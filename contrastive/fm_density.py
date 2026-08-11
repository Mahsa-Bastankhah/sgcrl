"""Flow-matching density estimator  p_θ(g | s, a)  as a CRL drop-in.

Overview
--------
Instead of contrastive φ(s,a)·ψ(g) or an exact NF, we learn a conditional
OT / linear-interpolant flow that pushes Gaussian noise to future goals
drawn from the discounted occupancy  p_γ(s_f | s, a):

    x_0 ~ N(0, I)
    x_1 = g = obs_to_goal(s_f),   s_f ~ p_γ(· | s, a)
    t   ~ U[0, 1]
    x_t = (1 − t) x_0 + t x_1
    v*  = x_1 − x_0
    L   = E[ ‖ v_θ(s, a, x_t, t) − v* ‖² ]

This matches the BC flow objective in FAC (`FAC/agents/fac.py` `bc_loss`),
with the data endpoint swapped from actions → future goals and conditioning
extended from observations → (s, a).

Sampling of (s, a, s_f)
-----------------------
Reuses the same EpisodeReplay batches as default CRL / Gaussian / NF:

    batch['obs']    = [ s_t ; obs_to_goal(s_f) ]
    batch['action'] = a_t

where s_f is drawn with offset d ~ TruncatedGeometric(1−γ) inside the
episode (handled entirely by EpisodeReplay.sample — this module never
re-samples futures).

Reward
------
At every PPO step, the per-env shaped reward is the flow log-density
obtained by integrating the reverse ODE (FAC `logprob_given_actions`):

    r_t = log p_θ(g | s_t, a_t)
        = log N(x_0; 0, I) − ∫_0^1 ∇·v_θ(s_t, a_t, x_t, t) dt

Divergence can be exact (Jacobian trace) or Hutchinson.

Usage in ppo_learner.py
-----------------------
Set config.ppo_repr_mode = 'fm'.  Wire via make_fm_density_networks /
make_fm_density_update_fn / make_fm_reward_fn (same pattern as gaussian/nf).
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme.jax import networks as networks_lib

from contrastive.networks import _mlp_or_residual


# ---------------------------------------------------------------------------
# 1.  Network container
# ---------------------------------------------------------------------------

class FMDensityNetworks(NamedTuple):
  """Velocity field for conditional OT flow matching on goals.

  velocity_net:
      FeedForwardNetwork  init(key) / apply(params, state, action, x_t, t)
      state.shape  = (B, obs_dim)
      action.shape = (B, act_dim)
      x_t.shape    = (B, goal_dim)   # interpolant / flow state
      t.shape      = (B, 1)          # time in [0, 1]
      returns velocity of shape (B, goal_dim)
  """
  velocity_net: networks_lib.FeedForwardNetwork
  goal_dim: int
  flow_steps: int


# ---------------------------------------------------------------------------
# 2.  Network factory  (ResidualMLP velocity field → Haiku)
# ---------------------------------------------------------------------------

def make_fm_density_networks(
    obs_dim: int,
    act_dim: int,
    goal_dim: int,
    hidden_layer_sizes: Sequence[int] = (512, 512, 512, 512),
    flow_steps: int = 10,
    layer_norm: bool = False,
) -> FMDensityNetworks:
  """Build v_θ(s, a, x_t, t) → R^{goal_dim}.

  Architecture matches CRL / Gaussian density trunks:
      concat([s, a, x_t, t]) → ResidualMLP(hidden…) → Linear(goal_dim)

  With ``len(hidden_layer_sizes) > 2`` (e.g. 6×256), this uses
  ``ResidualMLP`` (LayerNorm + Swish + skip_every=2). Shallow stacks
  fall back to ``hk.nets.MLP``. ``layer_norm`` is kept for API compat
  and is unused on the residual path (LN is always on there).
  """
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
    inputs = jnp.concatenate([state, action, x_t, t], axis=-1)
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
# 3.  Forward Euler sample  (noise → goal)  — FAC compute_flow_actions
# ---------------------------------------------------------------------------

def fm_sample(
    fm_networks: FMDensityNetworks,
    params,
    state: jnp.ndarray,
    action: jnp.ndarray,
    key,
    flow_steps: Optional[int] = None,
) -> jnp.ndarray:
  """Integrate the ODE from noise to a goal sample.

  x ← x + v_θ(s, a, x, t=i/N) / N,  i = 0..N−1,  x_0 ~ N(0,I).
  """
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
  b = state.shape[0]
  x = jax.random.normal(key, (b, fm_networks.goal_dim), dtype=state.dtype)

  def body(i, x_cur):
    t = jnp.full((b, 1), i / n, dtype=x_cur.dtype)
    vel = fm_networks.velocity_net.apply(params, state, action, x_cur, t)
    return x_cur + vel / n

  return jax.lax.fori_loop(0, n, body, x)


# ---------------------------------------------------------------------------
# 4.  Log-density via reverse ODE  — FAC logprob_given_actions
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
) -> jnp.ndarray:
  """log p_θ(goal | state, action) via reverse Euler + base Gaussian.

  mode:
    'exact'       — exact Jacobian trace (cheap when goal_dim is small)
    'hutch-rade'  — Hutchinson with Rademacher probes (needs rng)
    'hutch-gaus'  — Hutchinson with Gaussian probes (needs rng)
  """
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
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

  def body_fun(k, carry):
    x_cur, logp_cur = carry
    t_k = (n - k) / n
    t_batch = jnp.full((b, 1), t_k, dtype=x_cur.dtype)
    vels = v_apply(state, action, x_cur, t_batch)
    if need_key:
      divs = div_fn(state, action, x_cur, t_batch.squeeze(-1), rngs[k])
    else:
      divs = div_fn(state, action, x_cur, t_batch.squeeze(-1))
    # Accumulate ∫ ∇·v dt; return base_logp − that integral below.
    logp_new = logp_cur + divs * dt
    x_new = x_cur - vels * dt
    return (x_new, logp_new)

  x_0, logp_div = jax.lax.fori_loop(0, n, body_fun, (x_t, logp_acc))
  base_logp = (
      -0.5 * jnp.sum(x_0 ** 2, axis=-1)
      - 0.5 * g_dim * jnp.log(2.0 * jnp.pi)
  )
  return base_logp - logp_div


# ---------------------------------------------------------------------------
# 5.  OT-CFM training update  — FAC bc_loss on CRL future goals
# ---------------------------------------------------------------------------

def make_fm_density_update_fn(
    fm_networks: FMDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
):
  """Jitted OT conditional flow-matching update on EpisodeReplay batches.

  Signature matches make_gaussian_density_update_fn:

      update(params, opt_state, batch, key)
      → (new_params, new_opt_state, metrics)

  Training objective (FAC bc_loss with x_1 = future goal):
      x_0 ~ N(0,I),  x_1 = obs[:, obs_dim:],  t ~ U[0,1]
      x_t = (1−t) x_0 + t x_1,   v* = x_1 − x_0
      L   = mean ‖ v_θ(s, a, x_t, t) − v* ‖²
  """

  def _loss(params, batch, key):
    obs = batch['obs']
    action = batch['action']
    state = obs[:, :obs_dim]
    x_1 = obs[:, obs_dim:]  # g = obs_to_goal(s_f), s_f ~ p_γ

    b = x_1.shape[0]
    key_x, key_t = jax.random.split(key)
    x_0 = jax.random.normal(key_x, x_1.shape, dtype=x_1.dtype)
    t = jax.random.uniform(key_t, (b, 1), dtype=x_1.dtype)
    x_t = (1.0 - t) * x_0 + t * x_1
    vel_target = x_1 - x_0

    vel_pred = fm_networks.velocity_net.apply(
        params, state, action, x_t, t)
    loss = jnp.mean((vel_pred - vel_target) ** 2)

    metrics = {
        'density_loss': loss,
        'fm_flow_loss': loss,
        'vel_pred_norm': jnp.mean(jnp.linalg.norm(vel_pred, axis=-1)),
        'vel_target_norm': jnp.mean(jnp.linalg.norm(vel_target, axis=-1)),
        'x1_norm': jnp.mean(jnp.linalg.norm(x_1, axis=-1)),
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
# 6.  Reward  r_t = log p_θ(g | s_t, a_t)
# ---------------------------------------------------------------------------

def make_fm_reward_fn(
    fm_networks: FMDensityNetworks,
    obs_dim: int,
    logp_mode: str = 'exact',
    flow_steps: Optional[int] = None,
    hutch_probes: int = 8,
):
  """Jitted reward: log-density of the episode goal under the flow.

  For ``exact`` mode:
      reward_fn(params, obs, action) → (E,)
  For Hutchinson modes:
      reward_fn(params, obs, action, key) → (E,)
  """
  n = int(flow_steps if flow_steps is not None else fm_networks.flow_steps)
  need_key = 'hutch' in logp_mode

  if need_key:
    @jax.jit
    def reward_fn(params, obs, action, key):
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
      return fm_log_prob(
          fm_networks, params, state, action, goal,
          rng=key, mode=logp_mode, flow_steps=n,
          hutch_probes=hutch_probes)
  else:
    @jax.jit
    def reward_fn(params, obs, action):
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
      return fm_log_prob(
          fm_networks, params, state, action, goal,
          rng=None, mode=logp_mode, flow_steps=n,
          hutch_probes=hutch_probes)

  return reward_fn
