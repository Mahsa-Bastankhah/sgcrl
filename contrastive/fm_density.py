"""Flow-matching density estimator  p_θ(g | s, a)  as a CRL drop-in.

Overview
--------
Instead of contrastive φ(s,a)·ψ(g) or an exact NF, we learn a conditional
OT / linear-interpolant flow that pushes Gaussian noise to future goals
drawn from the discounted occupancy  p_γ(s_f | s, a):

    x_0 ~ N(0, I)
    x_1 = g = obs_to_goal(s_f),   s_f ~ p_γ(· | s, a)
    t   ~ U[0, 1]  or Logit-Normal
    x_t = (1 − t) x_0 + t x_1
    v*  = x_1 − x_0
    L   = E[ ‖ v_θ(s, a, x_t, t) − v* ‖² ]

This module supports:
1. Fourier / Sinusoidal Time Embeddings for scalar time t
2. Heun's 2nd-order Predictor-Corrector ODE Solver for sampling & logp
3. Logit-Normal Timestep Sampling for training updates
4. Lightweight diagnostic logging (stage-wise loss, d v / dt norms)
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme.jax import networks as networks_lib

from contrastive.networks import _mlp_or_residual


# ---------------------------------------------------------------------------
# 1.  Time Embedding & Sampling Helpers
# ---------------------------------------------------------------------------

def fourier_time_embedding(
    t: jnp.ndarray,
    embed_dim: int = 32,
    scale: float = 16.0,
) -> jnp.ndarray:
  """Embed scalar time t in [0, 1] into a multi-frequency Fourier vector.

  Args:
    t: Shape (B, 1) or (B,).
    embed_dim: Output dimension (must be even).
    scale: Frequency scaling constant.
  """
  if t.ndim == 1:
    t = t[:, None]
  half_dim = max(1, embed_dim // 2)
  freqs = jnp.exp(jnp.linspace(0, jnp.log(scale), half_dim, dtype=t.dtype))
  args = t * freqs * (2.0 * jnp.pi)
  return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def sample_timesteps(
    key,
    shape: Tuple[int, ...],
    mode: str = 'uniform',
    logit_loc: float = 0.0,
    logit_scale: float = 1.0,
    dtype=jnp.float32,
) -> jnp.ndarray:
  """Sample timesteps t in [0, 1].

  mode:
    'uniform'      — standard t ~ U[0, 1]
    'logit_normal' — t = sigmoid(z), z ~ N(loc, scale)
  """
  if mode == 'logit_normal':
    z = jax.random.normal(key, shape, dtype=dtype) * logit_scale + logit_loc
    return jax.nn.sigmoid(z)
  return jax.random.uniform(key, shape, dtype=dtype)


# ---------------------------------------------------------------------------
# 2.  Network container & factory
# ---------------------------------------------------------------------------

class FMDensityNetworks(NamedTuple):
  """Velocity field for conditional OT flow matching on goals."""
  velocity_net: networks_lib.FeedForwardNetwork
  goal_dim: int
  flow_steps: int
  time_embedding: bool = False
  time_embed_dim: int = 32
  ode_solver: str = 'euler'


def make_fm_density_networks(
    obs_dim: int,
    act_dim: int,
    goal_dim: int,
    hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    flow_steps: int = 10,
    layer_norm: bool = False,
    time_embedding: bool = False,
    time_embed_dim: int = 32,
    ode_solver: str = 'euler',
) -> FMDensityNetworks:
  """Build v_θ(s, a, x_t, t) → R^{goal_dim}."""
  del layer_norm  # residual path always uses LayerNorm; API kept for flags
  assert goal_dim >= 1, f'FM density requires goal_dim >= 1, got {goal_dim}'
  hidden = tuple(hidden_layer_sizes)

  def _velocity_fn(
      state: jnp.ndarray,
      action: jnp.ndarray,
      x_t: jnp.ndarray,
      t: jnp.ndarray,
  ) -> jnp.ndarray:
    """(s, a, x_t, t) → velocity (B, goal_dim)."""
    if t.ndim == 1:
      t = t[:, None]
    if time_embedding:
      t_feat = fourier_time_embedding(t, embed_dim=time_embed_dim)
    else:
      t_feat = t
    inputs = jnp.concatenate([state, action, x_t, t_feat], axis=-1)
    w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    trunk = _mlp_or_residual(
        inputs,
        list(hidden),
        hidden_layer_sizes=hidden,
        name='velocity_trunk',
        activation=jax.nn.relu,
        activate_final=True,
        w_init=w_init,
    )
    return hk.Linear(goal_dim, w_init=w_init, name='vel_out')(trunk)

  transformed = hk.without_apply_rng(hk.transform(_velocity_fn))

  dummy_state = np.zeros((1, obs_dim), dtype=np.float32)
  dummy_action = np.zeros((1, act_dim), dtype=np.float32)
  dummy_x = np.zeros((1, goal_dim), dtype=np.float32)
  dummy_t = np.zeros((1, 1), dtype=np.float32)

  velocity_net = networks_lib.FeedForwardNetwork(
      init=lambda key: transformed.init(
          key, dummy_state, dummy_action, dummy_x, dummy_t),
      apply=transformed.apply,
  )
  return FMDensityNetworks(
      velocity_net=velocity_net,
      goal_dim=int(goal_dim),
      flow_steps=int(flow_steps),
      time_embedding=bool(time_embedding),
      time_embed_dim=int(time_embed_dim),
      ode_solver=str(ode_solver).lower(),
  )


def init_fm_params(fm_networks: FMDensityNetworks, key) -> networks_lib.Params:
  """Init velocity-field parameters."""
  return fm_networks.velocity_net.init(key)


def _tree_l2_norm(tree) -> jnp.ndarray:
  leaves = jax.tree_util.tree_leaves(tree)
  if not leaves:
    return jnp.array(0.0)
  return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


# ---------------------------------------------------------------------------
# 3.  ODE sample (noise → goal) — Euler or Heun
# ---------------------------------------------------------------------------

def fm_sample(
    fm_networks: FMDensityNetworks,
    params,
    state: jnp.ndarray,
    action: jnp.ndarray,
    key,
    flow_steps: Optional[int] = None,
    ode_solver: Optional[str] = None,
) -> jnp.ndarray:
  """Integrate the ODE from noise to a goal sample."""
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
  solver = str(ode_solver if ode_solver is not None else fm_networks.ode_solver).lower()
  b = state.shape[0]
  x = jax.random.normal(key, (b, fm_networks.goal_dim), dtype=state.dtype)

  if solver == 'heun':
    def body_heun(i, x_cur):
      t1 = jnp.full((b, 1), i / n, dtype=x_cur.dtype)
      t2 = jnp.full((b, 1), (i + 1) / n, dtype=x_cur.dtype)
      v1 = fm_networks.velocity_net.apply(params, state, action, x_cur, t1)
      x_pred = x_cur + v1 / n
      v2 = fm_networks.velocity_net.apply(params, state, action, x_pred, t2)
      return x_cur + 0.5 * (v1 + v2) / n

    return jax.lax.fori_loop(0, n, body_heun, x)

  def body_euler(i, x_cur):
    t = jnp.full((b, 1), i / n, dtype=x_cur.dtype)
    vel = fm_networks.velocity_net.apply(params, state, action, x_cur, t)
    return x_cur + vel / n

  return jax.lax.fori_loop(0, n, body_euler, x)


# ---------------------------------------------------------------------------
# 4.  Log-density via reverse ODE (Euler or Heun)
# ---------------------------------------------------------------------------

def _make_divergence_exact(v_apply):
  """v_apply: (s[B,S], a[B,A], x[B,G], t[B,1]) → vel[B,G]."""

  def single_div(s, a, x, t):
    def v_fn(x_in):
      return v_apply(s[None, :], a[None, :], x_in[None, :], t[None, None])[0]
    j = jax.jacrev(v_fn)(x)
    return jnp.trace(j)

  return jax.vmap(single_div, in_axes=(0, 0, 0, 0))


def _make_divergence_hutchinson(v_apply, probes: int = 8, gaussian: bool = False):
  """Hutchinson trace estimator: E[εᵀ J ε]."""

  def single_div(s, a, x, t, key):
    def v_fn(x_in):
      return v_apply(s[None, :], a[None, :], x_in[None, :], t[None, None])[0]

    def vjv(k):
      _, vjp = jax.vjp(v_fn, x)
      eps = (jax.random.normal(k, x.shape, dtype=x.dtype) if gaussian
             else jax.random.rademacher(k, x.shape, dtype=x.dtype))
      return jnp.dot(vjp(eps)[0], eps)

    keys = jax.random.split(key, probes)
    return jax.vmap(vjv)(keys).mean()

  return jax.vmap(single_div, in_axes=(0, 0, 0, 0, 0))


def fm_log_prob(
    fm_networks: FMDensityNetworks,
    params,
    state: jnp.ndarray,
    action: jnp.ndarray,
    goal: jnp.ndarray,
    rng=None,
    mode: str = 'exact',
    flow_steps: Optional[int] = None,
    hutch_probes: int = 8,
    ode_solver: Optional[str] = None,
) -> jnp.ndarray:
  """log p_θ(goal | state, action) via reverse ODE + base Gaussian."""
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
  solver = str(ode_solver if ode_solver is not None else fm_networks.ode_solver).lower()
  dt = 1.0 / n
  b, g_dim = goal.shape

  def v_apply(s_b, a_b, x_b, t_b):
    return fm_networks.velocity_net.apply(params, s_b, a_b, x_b, t_b)

  if mode == 'exact':
    div_fn = _make_divergence_exact(v_apply)
    need_key = False
  elif mode == 'hutch-rade':
    assert rng is not None, 'hutchinson logp requires rng'
    div_fn = _make_divergence_hutchinson(
        v_apply, probes=hutch_probes, gaussian=False)
    need_key = True
  elif mode == 'hutch-gaus':
    assert rng is not None, 'hutchinson logp requires rng'
    div_fn = _make_divergence_hutchinson(
        v_apply, probes=hutch_probes, gaussian=True)
    need_key = True
  else:
    raise ValueError(f'Unknown fm logp mode: {mode!r}')

  x_t = goal
  logp_acc = jnp.zeros((b,), dtype=goal.dtype)
  if need_key:
    rngs = jax.random.split(rng, n * b).reshape(n, b, 2)
  else:
    rngs = None

  if solver == 'heun':
    def body_fun_heun(k, carry):
      x_cur, logp_cur = carry
      t_k = (n - k) / n
      t_next = (n - k - 1) / n
      t_batch1 = jnp.full((b, 1), t_k, dtype=x_cur.dtype)
      t_batch2 = jnp.full((b, 1), t_next, dtype=x_cur.dtype)

      v1 = v_apply(state, action, x_cur, t_batch1)
      if need_key:
        div1 = div_fn(state, action, x_cur, t_batch1.squeeze(-1), rngs[k])
      else:
        div1 = div_fn(state, action, x_cur, t_batch1.squeeze(-1))

      x_pred = x_cur - v1 * dt
      v2 = v_apply(state, action, x_pred, t_batch2)
      if need_key:
        div2 = div_fn(state, action, x_pred, t_batch2.squeeze(-1), rngs[k])
      else:
        div2 = div_fn(state, action, x_pred, t_batch2.squeeze(-1))

      v_avg = 0.5 * (v1 + v2)
      div_avg = 0.5 * (div1 + div2)

      logp_new = logp_cur + div_avg * dt
      x_new = x_cur - v_avg * dt
      return (x_new, logp_new)

    x_0, logp_div = jax.lax.fori_loop(0, n, body_fun_heun, (x_t, logp_acc))
  else:
    def body_fun_euler(k, carry):
      x_cur, logp_cur = carry
      t_k = (n - k) / n
      t_batch = jnp.full((b, 1), t_k, dtype=x_cur.dtype)
      vels = v_apply(state, action, x_cur, t_batch)
      if need_key:
        divs = div_fn(state, action, x_cur, t_batch.squeeze(-1), rngs[k])
      else:
        divs = div_fn(state, action, x_cur, t_batch.squeeze(-1))
      logp_new = logp_cur + divs * dt
      x_new = x_cur - vels * dt
      return (x_new, logp_new)

    x_0, logp_div = jax.lax.fori_loop(0, n, body_fun_euler, (x_t, logp_acc))

  base_logp = (
      -0.5 * jnp.sum(x_0 ** 2, axis=-1)
      - 0.5 * g_dim * jnp.log(2.0 * jnp.pi)
  )
  return base_logp - logp_div


# ---------------------------------------------------------------------------
# 5.  OT-CFM training update & Diagnostic Metrics
# ---------------------------------------------------------------------------

def make_fm_density_update_fn(
    fm_networks: FMDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    t_sample_mode: str = 'uniform',
    t_logit_loc: float = 0.0,
    t_logit_scale: float = 1.0,
    goal_noise_std: float = 0.0,
):
  """Jitted OT conditional flow-matching update with diagnostic metrics."""

  def _loss(params, batch, key):
    obs = batch['obs']
    action = batch['action']
    state = obs[:, :obs_dim]
    x_1 = obs[:, obs_dim:]  # g = obs_to_goal(s_f), s_f ~ p_γ

    b = x_1.shape[0]
    key_x, key_t, key_g = jax.random.split(key, 3)
    if goal_noise_std > 0.0:
      x_1 = x_1 + jax.random.normal(key_g, x_1.shape, dtype=x_1.dtype) * goal_noise_std

    x_0 = jax.random.normal(key_x, x_1.shape, dtype=x_1.dtype)
    t = sample_timesteps(
        key_t, (b, 1), mode=t_sample_mode,
        logit_loc=t_logit_loc, logit_scale=t_logit_scale, dtype=x_1.dtype)
    x_t = (1.0 - t) * x_0 + t * x_1
    vel_target = x_1 - x_0

    vel_pred = fm_networks.velocity_net.apply(
        params, state, action, x_t, t)
    sq_err = jnp.mean((vel_pred - vel_target) ** 2, axis=-1)
    loss = jnp.mean(sq_err)

    # --- Diagnostic Metrics ---
    t_flat = t.squeeze(-1)
    mask_early = t_flat < 0.33
    mask_mid = (t_flat >= 0.33) & (t_flat < 0.67)
    mask_late = t_flat >= 0.67

    loss_early = jnp.sum(jnp.where(mask_early, sq_err, 0.0)) / jnp.maximum(jnp.sum(mask_early), 1.0)
    loss_mid = jnp.sum(jnp.where(mask_mid, sq_err, 0.0)) / jnp.maximum(jnp.sum(mask_mid), 1.0)
    loss_late = jnp.sum(jnp.where(mask_late, sq_err, 0.0)) / jnp.maximum(jnp.sum(mask_late), 1.0)

    # Velocity-time derivative norm diagnostic || d v_theta / dt ||_2 (via finite diff)
    eps_t = 1e-3
    t_plus = jnp.clip(t + eps_t, 0.0, 1.0)
    t_minus = jnp.clip(t - eps_t, 0.0, 1.0)
    v_plus = fm_networks.velocity_net.apply(params, state, action, x_t, t_plus)
    v_minus = fm_networks.velocity_net.apply(params, state, action, x_t, t_minus)
    vel_dt_norm = jnp.mean(jnp.linalg.norm((v_plus - v_minus) / (2.0 * eps_t), axis=-1))

    metrics = {
        'density_loss': loss,
        'fm_flow_loss': loss,
        'vel_pred_norm': jnp.mean(jnp.linalg.norm(vel_pred, axis=-1)),
        'vel_target_norm': jnp.mean(jnp.linalg.norm(vel_target, axis=-1)),
        'x1_norm': jnp.mean(jnp.linalg.norm(x_1, axis=-1)),
        'loss_t_early': loss_early,
        'loss_t_mid': loss_mid,
        'loss_t_late': loss_late,
        'vel_dt_norm': vel_dt_norm,
    }
    return loss, metrics

  grad_fn = jax.value_and_grad(_loss, has_aux=True)

  def update(params, opt_state, batch, key):
    (_, metrics), grads = grad_fn(params, batch, key)

    grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads))))
    loss_finite = jnp.isfinite(metrics['density_loss'])
    do_update = jnp.logical_and(grads_finite, loss_finite)

    def _apply(_):
      updates, new_opt_state = optimizer.update(grads, opt_state, params)
      return optax.apply_updates(params, updates), new_opt_state

    def _skip(_):
      return params, opt_state

    new_params, new_opt_state = jax.lax.cond(
        do_update, _apply, _skip, operand=None)
    metrics = dict(metrics)
    metrics['grad_norm'] = _tree_l2_norm(grads)
    metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
    return new_params, new_opt_state, metrics

  return jax.jit(update)


# ---------------------------------------------------------------------------
# 6.  Reward  r_t = log p_θ(g | s_t, a_t)
# ---------------------------------------------------------------------------

def make_fm_reward_fn(
    fm_networks: FMDensityNetworks,
    obs_dim: int,
    logp_mode: str = 'exact',
    flow_steps: Optional[int] = None,
    hutch_probes: int = 8,
    ode_solver: Optional[str] = None,
):
  """Jitted reward: log-density of the episode goal under the flow."""
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
  solver = str(ode_solver if ode_solver is not None else fm_networks.ode_solver).lower()
  need_key = 'hutch' in logp_mode

  if need_key:
    @jax.jit
    def reward_fn(params, obs, action, key):
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
      return fm_log_prob(
          fm_networks, params, state, action, goal,
          rng=key, mode=logp_mode, flow_steps=n,
          hutch_probes=hutch_probes, ode_solver=solver)
  else:
    @jax.jit
    def reward_fn(params, obs, action):
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
      return fm_log_prob(
          fm_networks, params, state, action, goal,
          rng=None, mode=logp_mode, flow_steps=n,
          hutch_probes=hutch_probes, ode_solver=solver)

  return reward_fn
