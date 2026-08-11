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
    goal_mean: Optional[jnp.ndarray] = None,
    goal_std: Optional[jnp.ndarray] = None,
    reward_clip: float = 0.0,
) -> jnp.ndarray:
  """log p_θ(goal | state, action) via reverse ODE + base Gaussian."""
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
  solver = str(ode_solver if ode_solver is not None else fm_networks.ode_solver).lower()
  dt = 1.0 / n
  b, g_dim = goal.shape

  if goal_mean is not None and goal_std is not None:
    x_t = (goal - goal_mean) / goal_std
    log_det_adj = -jnp.sum(jnp.log(goal_std))
  else:
    x_t = goal
    log_det_adj = 0.0

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
  logp = base_logp - logp_div + log_det_adj
  if reward_clip > 0.0:
    logp = jnp.clip(logp, -reward_clip, reward_clip)
  return logp


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
    cond_dropout: float = 0.0,
    cat_acc_mode: str = 'midpoint',
    cat_acc_subbatch: int = 128,
    cat_acc_flow_steps: int = -1,
    logp_mode: str = 'exact',
    ode_solver: str = 'euler',
    hutch_probes: int = 8,
    reward_clip: float = 0.0,
):
  """Jitted OT conditional flow-matching update with diagnostic metrics."""

  def _loss(params, batch, key, goal_mean=None, goal_std=None):
    obs = batch['obs']
    action = batch['action']
    state = obs[:, :obs_dim]
    x_1 = obs[:, obs_dim:]  # g = obs_to_goal(s_f), s_f ~ p_γ

    if goal_mean is not None and goal_std is not None:
      x_1 = (x_1 - goal_mean) / goal_std

    b = x_1.shape[0]
    key_x, key_t, key_g, key_mask, key_logp = jax.random.split(key, 5)
    if goal_noise_std > 0.0:
      x_1 = x_1 + jax.random.normal(key_g, x_1.shape, dtype=x_1.dtype) * goal_noise_std

    if cond_dropout > 0.0:
      mask = jax.random.bernoulli(key_mask, p=1.0 - cond_dropout, shape=(b, 1))
      state = state * mask
      action = action * mask

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

    # --- Sub-batch Pairwise Top-1 Goal Retrieval Categorical Accuracy ---
    b_sub = min(int(x_1.shape[0]), int(cat_acc_subbatch))
    sub_state = state[:b_sub]
    sub_action = action[:b_sub]
    sub_x1 = x_1[:b_sub]
    sub_x0 = x_0[:b_sub]

    if str(cat_acc_mode).lower() == 'logp':
      # Full reverse-ODE conditional log-likelihood matrix log p_FM(g_j | s_i, a_i)
      state_rep = jnp.repeat(sub_state, b_sub, axis=0)
      action_rep = jnp.repeat(sub_action, b_sub, axis=0)
      goal_rep = jnp.tile(sub_x1, (b_sub, 1))

      eval_steps = (int(cat_acc_flow_steps) if int(cat_acc_flow_steps) > 0
                    else int(fm_networks.flow_steps))
      logp_flat = fm_log_prob(
          fm_networks, params, state_rep, action_rep, goal_rep,
          rng=key_logp, mode=logp_mode, flow_steps=eval_steps,
          hutch_probes=hutch_probes, ode_solver=ode_solver,
          goal_mean=None, goal_std=None, reward_clip=reward_clip,
      )
      scores = logp_flat.reshape(b_sub, b_sub)
    else:
      # Midpoint velocity prediction error proxy at t = 0.5
      t_mid = jnp.full((b_sub, 1), 0.5, dtype=sub_x1.dtype)
      x0_expanded = sub_x0[:, None, :]
      x1_expanded = sub_x1[None, :, :]
      x_t_pair = 0.5 * x0_expanded + 0.5 * x1_expanded
      target_pair = x1_expanded - x0_expanded

      state_rep = jnp.repeat(sub_state, b_sub, axis=0)
      action_rep = jnp.repeat(sub_action, b_sub, axis=0)
      t_rep = jnp.repeat(t_mid, b_sub, axis=0)
      x_t_flat = x_t_pair.reshape(-1, sub_x1.shape[-1])

      vel_pair_pred = fm_networks.velocity_net.apply(
          params, state_rep, action_rep, x_t_flat, t_rep)
      vel_pair_pred = vel_pair_pred.reshape(b_sub, b_sub, -1)

      sq_err_pair = jnp.mean((vel_pair_pred - target_pair) ** 2, axis=-1)
      scores = -sq_err_pair

    correct = (jnp.argmax(scores, axis=1) == jnp.arange(b_sub))
    cat_acc = jnp.mean(correct.astype(jnp.float32))

    # Score gradient norms ||grad_a log p(g | s, a)||_2 and ||grad_s log p(g | s, a)||_2
    b_grad = min(int(x_1.shape[0]), 64)
    def _single_logp(a_i, s_i, g_i):
      return fm_log_prob(
          fm_networks, params, s_i[None, :], a_i[None, :], g_i[None, :],
          mode='exact', flow_steps=min(int(fm_networks.flow_steps), 5),
          ode_solver='euler'
      )[0]

    grad_a_fn = jax.vmap(jax.grad(_single_logp, argnums=0), in_axes=(0, 0, 0))
    grad_s_fn = jax.vmap(jax.grad(_single_logp, argnums=1), in_axes=(0, 0, 0))
    grads_a = grad_a_fn(action[:b_grad], state[:b_grad], x_1[:b_grad])
    grads_s = grad_s_fn(action[:b_grad], state[:b_grad], x_1[:b_grad])
    action_grad_norm = jax.lax.stop_gradient(
        jnp.mean(jnp.linalg.norm(grads_a, axis=-1)))
    state_grad_norm = jax.lax.stop_gradient(
        jnp.mean(jnp.linalg.norm(grads_s, axis=-1)))
    state_action_grad_ratio = jax.lax.stop_gradient(
        state_grad_norm / (action_grad_norm + 1e-6))
    labels = jnp.eye(b_sub)
    logits_pos = jnp.sum(scores * labels) / b_sub
    logits_neg = jnp.sum(scores * (1.0 - labels)) / jnp.maximum(b_sub * (b_sub - 1), 1)

    metrics = {
        'density_loss': loss,
        'fm_flow_loss': loss,
        'categorical_accuracy': cat_acc,
        'logits_pos': logits_pos,
        'logits_neg': logits_neg,
        'vel_pred_norm': jnp.mean(jnp.linalg.norm(vel_pred, axis=-1)),
        'vel_target_norm': jnp.mean(jnp.linalg.norm(vel_target, axis=-1)),
        'x1_norm': jnp.mean(jnp.linalg.norm(x_1, axis=-1)),
        'loss_t_early': loss_early,
        'loss_t_mid': loss_mid,
        'loss_t_late': loss_late,
        'vel_dt_norm': vel_dt_norm,
        'action_grad_norm': action_grad_norm,
        'state_grad_norm': state_grad_norm,
        'state_action_grad_ratio': state_action_grad_ratio,
    }
    return loss, metrics

  grad_fn = jax.value_and_grad(_loss, has_aux=True)

  def update(params, opt_state, batch, key, goal_mean=None, goal_std=None):
    (_, metrics), grads = grad_fn(params, batch, key, goal_mean, goal_std)

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


def make_scan_fm_update_fn(
    fm_networks: FMDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    repr_tau: float = 0.0,
):
  """Scan-based FM updater: N density steps in one JIT call.

  Caller pre-samples all N batches into ``(N, B, …)`` arrays.  Reward-param
  EMA (``0 < repr_tau < 1``) is applied inside the scan.
  """
  raw_update = make_fm_density_update_fn(
      fm_networks, optimizer, obs_dim=obs_dim)
  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def multi_update(params, opt_state, params_ema, batches, key):
    def scan_step(carry, batch):
      p, opt, ema, k = carry
      k, k_u = jax.random.split(k)
      p, opt, m = raw_update(p, opt, batch, k_u)
      if use_ema:
        ema = jax.tree_util.tree_map(
            lambda t, o: _tau * t + (1.0 - _tau) * o, ema, p)
      else:
        ema = p
      return (p, opt, ema, k), m

    (params, opt_state, params_ema, key), metrics = jax.lax.scan(
        scan_step, (params, opt_state, params_ema, key), batches)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return params, opt_state, params_ema, key, metrics

  return multi_update


# ---------------------------------------------------------------------------
# 5b. TD-Flow Bellman Probability Path Training Update
# ---------------------------------------------------------------------------

def generate_target_goals_bootstrapped(
    fm_networks: FMDensityNetworks,
    target_params: networks_lib.Params,
    next_state: jnp.ndarray,
    next_action: jnp.ndarray,
    key: jax.random.PRNGKey,
    boot_steps: int = 1,
    ode_solver: str = 'euler',
    goal_dim: Optional[int] = None,
) -> jnp.ndarray:
  """Integrate target velocity field v_phi starting from Gaussian noise x_0 ~ N(0, I)
  for `boot_steps` to produce bootstrapped goal samples x_1_boot from next_state s_1."""
  b = next_state.shape[0]
  if goal_dim is None:
    g_dim = next_state.shape[-1]
  else:
    g_dim = goal_dim

  key_x0, key_loop = jax.random.split(key)
  x_0 = jax.random.normal(key_x0, (b, g_dim), dtype=next_state.dtype)

  n = max(1, boot_steps)
  dt = 1.0 / float(n)

  if ode_solver == 'heun' and n > 1:
    def body_fun_heun(i, x_cur):
      t1 = i * dt
      t2 = (i + 1) * dt
      t1_vec = jnp.full((b, 1), t1, dtype=x_cur.dtype)
      t2_vec = jnp.full((b, 1), t2, dtype=x_cur.dtype)
      v1 = fm_networks.velocity_net.apply(
          target_params, next_state, next_action, x_cur, t1_vec)
      x_pred = x_cur + dt * v1
      v2 = fm_networks.velocity_net.apply(
          target_params, next_state, next_action, x_pred, t2_vec)
      return x_cur + 0.5 * dt * (v1 + v2)

    return jax.lax.fori_loop(0, n, body_fun_heun, x_0)
  else:
    def body_fun_euler(i, x_cur):
      t_val = i * dt
      t_vec = jnp.full((b, 1), t_val, dtype=x_cur.dtype)
      vel = fm_networks.velocity_net.apply(
          target_params, next_state, next_action, x_cur, t_vec)
      return x_cur + dt * vel

    return jax.lax.fori_loop(0, n, body_fun_euler, x_0)


def make_td_fm_density_update_fn(
    fm_networks: FMDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    gamma: float = 0.99,
    target_tau: float = 0.005,
    boot_steps: int = 1,
    ode_solver: str = 'euler',
    t_sample_mode: str = 'uniform',
    t_logit_loc: float = 0.0,
    t_logit_scale: float = 1.0,
    goal_noise_std: float = 0.0,
    cond_dropout: float = 0.0,
    cat_acc_mode: str = 'midpoint',
    cat_acc_subbatch: int = 128,
    cat_acc_flow_steps: int = -1,
    logp_mode: str = 'exact',
    hutch_probes: int = 8,
    reward_clip: float = 0.0,
):
  """Jitted TD-Flow update function leveraging Bellman targets on probability paths."""

  def _loss(online_params, target_params, batch, key, goal_mean=None, goal_std=None):
    obs = batch['obs']
    action = batch['action']
    next_obs = batch['next_obs']

    state = obs[:, :obs_dim]
    goal_dim = obs.shape[-1] - obs_dim

    # 1-step next state and action
    next_state = next_obs[:, :obs_dim]
    next_action = batch.get('next_action', action)

    # 1-step next state goal: g_1 = obs_to_goal(s_1)
    g_1 = next_obs[:, obs_dim:]
    if goal_mean is not None and goal_std is not None:
      g_1 = (g_1 - goal_mean) / goal_std

    b = state.shape[0]
    key_x0, key_t, key_m, key_boot, key_g, key_mask, key_logp = jax.random.split(key, 7)

    if goal_noise_std > 0.0:
      g_1 = g_1 + jax.random.normal(key_g, g_1.shape, dtype=g_1.dtype) * goal_noise_std

    if cond_dropout > 0.0:
      mask = jax.random.bernoulli(key_mask, p=1.0 - cond_dropout, shape=(b, 1))
      state = state * mask
      action = action * mask
      next_state = next_state * mask
      next_action = next_action * mask

    # Bootstrapped goal sample x_1_boot starting from s_1
    x_1_boot = generate_target_goals_bootstrapped(
        fm_networks, target_params, next_state, next_action,
        key_boot, boot_steps=boot_steps, ode_solver=ode_solver,
        goal_dim=goal_dim)

    # Bernoulli mixture mask m ~ Bernoulli(1 - gamma)
    # True -> 1-step immediate target g_1 = obs_to_goal(s_1)
    # False -> Bootstrapped target x_1_boot
    mask_1step = jax.random.bernoulli(key_m, p=1.0 - gamma, shape=(b, 1))
    x_1_target = jnp.where(mask_1step, g_1, x_1_boot)

    x_0 = jax.random.normal(key_x0, x_1_target.shape, dtype=x_1_target.dtype)
    t = sample_timesteps(
        key_t, (b, 1), mode=t_sample_mode,
        logit_loc=t_logit_loc, logit_scale=t_logit_scale, dtype=x_1_target.dtype)

    x_t = (1.0 - t) * x_0 + t * x_1_target
    vel_target = x_1_target - x_0

    vel_pred = fm_networks.velocity_net.apply(
        online_params, state, action, x_t, t)
    sq_err = jnp.mean((vel_pred - vel_target) ** 2, axis=-1)
    loss = jnp.mean(sq_err)

    # --- Sub-batch Pairwise Top-1 Goal Retrieval Categorical Accuracy ---
    b_sub = min(int(x_1_target.shape[0]), int(cat_acc_subbatch))
    sub_state = state[:b_sub]
    sub_action = action[:b_sub]
    sub_x1 = x_1_target[:b_sub]
    sub_x0 = x_0[:b_sub]

    if str(cat_acc_mode).lower() == 'logp':
      # Full reverse-ODE conditional log-likelihood matrix log p_FM(g_j | s_i, a_i)
      state_rep = jnp.repeat(sub_state, b_sub, axis=0)
      action_rep = jnp.repeat(sub_action, b_sub, axis=0)
      goal_rep = jnp.tile(sub_x1, (b_sub, 1))

      eval_steps = (int(cat_acc_flow_steps) if int(cat_acc_flow_steps) > 0
                    else int(fm_networks.flow_steps))
      logp_flat = fm_log_prob(
          fm_networks, online_params, state_rep, action_rep, goal_rep,
          rng=key_logp, mode=logp_mode, flow_steps=eval_steps,
          hutch_probes=hutch_probes, ode_solver=ode_solver,
          goal_mean=None, goal_std=None, reward_clip=reward_clip,
      )
      scores = logp_flat.reshape(b_sub, b_sub)
    else:
      # Midpoint velocity prediction error proxy at t = 0.5
      t_mid = jnp.full((b_sub, 1), 0.5, dtype=sub_x1.dtype)
      x0_expanded = sub_x0[:, None, :]
      x1_expanded = sub_x1[None, :, :]
      x_t_pair = 0.5 * x0_expanded + 0.5 * x1_expanded
      target_pair = x1_expanded - x0_expanded

      state_rep = jnp.repeat(sub_state, b_sub, axis=0)
      action_rep = jnp.repeat(sub_action, b_sub, axis=0)
      t_rep = jnp.repeat(t_mid, b_sub, axis=0)
      x_t_flat = x_t_pair.reshape(-1, sub_x1.shape[-1])

      vel_pair_pred = fm_networks.velocity_net.apply(
          online_params, state_rep, action_rep, x_t_flat, t_rep)
      vel_pair_pred = vel_pair_pred.reshape(b_sub, b_sub, -1)

      sq_err_pair = jnp.mean((vel_pair_pred - target_pair) ** 2, axis=-1)
      scores = -sq_err_pair

    correct = (jnp.argmax(scores, axis=1) == jnp.arange(b_sub))
    cat_acc = jnp.mean(correct.astype(jnp.float32))

    # Score gradient norms ||grad_a log p(g | s, a)||_2 and ||grad_s log p(g | s, a)||_2
    b_grad = min(int(x_1_target.shape[0]), 64)
    def _single_logp_td(a_i, s_i, g_i):
      return fm_log_prob(
          fm_networks, online_params, s_i[None, :], a_i[None, :], g_i[None, :],
          mode='exact', flow_steps=min(int(fm_networks.flow_steps), 5),
          ode_solver='euler'
      )[0]

    grad_a_fn = jax.vmap(jax.grad(_single_logp_td, argnums=0), in_axes=(0, 0, 0))
    grad_s_fn = jax.vmap(jax.grad(_single_logp_td, argnums=1), in_axes=(0, 0, 0))
    grads_a = grad_a_fn(action[:b_grad], state[:b_grad], x_1_target[:b_grad])
    grads_s = grad_s_fn(action[:b_grad], state[:b_grad], x_1_target[:b_grad])
    action_grad_norm = jax.lax.stop_gradient(
        jnp.mean(jnp.linalg.norm(grads_a, axis=-1)))
    state_grad_norm = jax.lax.stop_gradient(
        jnp.mean(jnp.linalg.norm(grads_s, axis=-1)))
    state_action_grad_ratio = jax.lax.stop_gradient(
        state_grad_norm / (action_grad_norm + 1e-6))
    labels = jnp.eye(b_sub)
    logits_pos = jnp.sum(scores * labels) / b_sub
    logits_neg = jnp.sum(scores * (1.0 - labels)) / jnp.maximum(b_sub * (b_sub - 1), 1)

    metrics = {
        'density_loss': loss,
        'td_fm_loss': loss,
        'categorical_accuracy': cat_acc,
        'logits_pos': logits_pos,
        'logits_neg': logits_neg,
        'vel_pred_norm': jnp.mean(jnp.linalg.norm(vel_pred, axis=-1)),
        'vel_target_norm': jnp.mean(jnp.linalg.norm(vel_target, axis=-1)),
        'x1_target_norm': jnp.mean(jnp.linalg.norm(x_1_target, axis=-1)),
        'x1_boot_norm': jnp.mean(jnp.linalg.norm(x_1_boot, axis=-1)),
        'ratio_1step': jnp.mean(mask_1step.astype(jnp.float32)),
        'action_grad_norm': action_grad_norm,
        'state_grad_norm': state_grad_norm,
        'state_action_grad_ratio': state_action_grad_ratio,
    }
    return loss, metrics

  grad_fn = jax.value_and_grad(_loss, has_aux=True)

  def update(online_params, target_params, opt_state, batch, key, goal_mean=None, goal_std=None):
    (_, metrics), grads = grad_fn(online_params, target_params, batch, key, goal_mean, goal_std)

    grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads))))
    loss_finite = jnp.isfinite(metrics['density_loss'])
    do_update = jnp.logical_and(grads_finite, loss_finite)

    def _apply(_):
      updates, new_opt_state = optimizer.update(grads, opt_state, online_params)
      new_online = optax.apply_updates(online_params, updates)
      new_target = jax.tree_util.tree_map(
          lambda t, o: (1.0 - target_tau) * t + target_tau * o,
          target_params, new_online)
      return new_online, new_target, new_opt_state

    def _skip(_):
      return online_params, target_params, opt_state

    new_online, new_target, new_opt_state = jax.lax.cond(
        do_update, _apply, _skip, operand=None)

    metrics = dict(metrics)
    metrics['grad_norm'] = _tree_l2_norm(grads)
    metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
    return new_online, new_target, new_opt_state, metrics

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
    reward_clip: float = 0.0,
):
  """Jitted reward: log-density of the episode goal under the flow."""
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
  solver = str(ode_solver if ode_solver is not None else fm_networks.ode_solver).lower()
  need_key = 'hutch' in logp_mode

  if need_key:
    @jax.jit
    def reward_fn(params, obs, action, key, goal_mean=None, goal_std=None):
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
      return fm_log_prob(
          fm_networks, params, state, action, goal,
          rng=key, mode=logp_mode, flow_steps=n,
          hutch_probes=hutch_probes, ode_solver=solver,
          goal_mean=goal_mean, goal_std=goal_std,
          reward_clip=reward_clip)
  else:
    @jax.jit
    def reward_fn(params, obs, action, goal_mean=None, goal_std=None):
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
      return fm_log_prob(
          fm_networks, params, state, action, goal,
          rng=None, mode=logp_mode, flow_steps=n,
          hutch_probes=hutch_probes, ode_solver=solver,
          goal_mean=goal_mean, goal_std=goal_std,
          reward_clip=reward_clip)

  return reward_fn


# ---------------------------------------------------------------------------
# 7.  Score gradient norm evaluator ||grad_s log p(g | s, a)|| and ||grad_a log p(g | s, a)||
# ---------------------------------------------------------------------------

def make_fm_grad_norm_fn(
    fm_networks: FMDensityNetworks,
    obs_dim: int,
    flow_steps: Optional[int] = 5,
    ode_solver: str = 'euler',
):
  """Jitted evaluator returning (state_grad_norm, action_grad_norm) for a batch of (obs, action)."""
  eval_steps = int(flow_steps if flow_steps is not None else min(int(fm_networks.flow_steps), 5))
  solver = str(ode_solver).lower()

  def _single_logp(a_i, s_i, g_i, params):
    return fm_log_prob(
        fm_networks, params, s_i[None, :], a_i[None, :], g_i[None, :],
        mode='exact', flow_steps=eval_steps,
        ode_solver=solver,
    )[0]

  @jax.jit
  def grad_norm_fn(params, obs, action):
    state = obs[:, :obs_dim]
    goal = obs[:, obs_dim:]
    grad_s_fn = jax.vmap(
        lambda a, s, g: jax.grad(_single_logp, argnums=1)(a, s, g, params),
        in_axes=(0, 0, 0))
    grad_a_fn = jax.vmap(
        lambda a, s, g: jax.grad(_single_logp, argnums=0)(a, s, g, params),
        in_axes=(0, 0, 0))
    grads_s = grad_s_fn(action, state, goal)
    grads_a = grad_a_fn(action, state, goal)
    s_norms = jnp.linalg.norm(grads_s, axis=-1)
    a_norms = jnp.linalg.norm(grads_a, axis=-1)
    return s_norms, a_norms

  return grad_norm_fn


# ---------------------------------------------------------------------------
# 8.  Seen vs Unseen Log-Probability Diagnostics Evaluator
# ---------------------------------------------------------------------------

def compute_fm_logp_diagnostics(
    fm_networks: FMDensityNetworks,
    params,
    state: jnp.ndarray,
    action: jnp.ndarray,
    seen_goals: jnp.ndarray,
    unseen_goals: jnp.ndarray,
    flow_steps: int = 5,
    ode_solver: str = 'euler',
    goal_mean: Optional[jnp.ndarray] = None,
    goal_std: Optional[jnp.ndarray] = None,
) -> dict:
  """Evaluate summary statistics of seen vs unseen log probabilities."""
  logp_seen = fm_log_prob(
      fm_networks, params, state, action, seen_goals,
      mode='exact', flow_steps=flow_steps, ode_solver=ode_solver,
      goal_mean=goal_mean, goal_std=goal_std)

  logp_unseen = fm_log_prob(
      fm_networks, params, state, action, unseen_goals,
      mode='exact', flow_steps=flow_steps, ode_solver=ode_solver,
      goal_mean=goal_mean, goal_std=goal_std)

  return {
      'fm_diag/logp_seen_mean': float(jnp.mean(logp_seen)),
      'fm_diag/logp_seen_min': float(jnp.min(logp_seen)),
      'fm_diag/logp_seen_max': float(jnp.max(logp_seen)),
      'fm_diag/logp_seen_std': float(jnp.std(logp_seen)),
      'fm_diag/logp_unseen_mean': float(jnp.mean(logp_unseen)),
      'fm_diag/logp_unseen_min': float(jnp.min(logp_unseen)),
      'fm_diag/logp_unseen_max': float(jnp.max(logp_unseen)),
      'fm_diag/logp_unseen_std': float(jnp.std(logp_unseen)),
      'fm_diag/logp_gap': float(jnp.mean(logp_seen) - jnp.mean(logp_unseen)),
  }
