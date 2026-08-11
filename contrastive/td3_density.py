"""TD3-style twin Q(s, a, s_f) density estimator as a CRL drop-in for PPO.

Overview
--------
Instead of contrastive φ(s,a)·ψ(g), we learn a goal-conditioned Q:

    Q_θ(s, a, g)   ≈   E[ Σ_t γ^t  1{s_t ≈ g}  |  s_0=s, a_0=a ]

with twin critics + Polyak target networks (TD3-style clipped double Q).

Parameterizations
-----------------
  ``mlp`` (default): Q = MLP([s; g; a]) → scalar
  ``bilinear``:      Q(s,a,g) = x(s,a) · y(g)
                     where x, y match CRL φ / ψ (sa_encoder / g_encoder):
                     ResidualMLP(hidden… + [repr_dim]), same init / activation.

Sampling
--------
Reuses the same EpisodeReplay batches as CRL / Gaussian / NF:

    batch['obs']      = [ s  ; obs_to_goal(s_f) ]
    batch['action']   = a
    batch['next_obs'] = [ s' ; obs_to_goal(s_f) ]

where s_f is a geometrically-sampled future state (d ≥ 1).

TD3 critic objective
--------------------
  r     = 1{ ‖obs_to_goal(s') − g‖ < tol }          (indicator on NEXT state)
  a'    ~ π(· | s', g)                              (online π, or π̄ if enabled)
  y     = r + γ · min(Q1̄, Q2̄)(s', a', g)           (always bootstrap)
  L     = MSE(Q1(s,a,g), y) + MSE(Q2(s,a,g), y)
  targets ← (1 − τ) · targets + τ · online          (Polyak; config.tau)

FB critic objective (``fb_loss`` / ``ppo_td3_fb_loss``)
-------------------------------------------------------
Same twin Q(s,a,s_f) nets and s_f sampling, but no indicator reward::

  a'  ~ π(· | s', s_f)
  y   = γ · stopgrad( min(Q1̄, Q2̄)(s', a', s_f) )
  L_i = mean( (Qi(s,a,s_f) − y)^2 ) − mean( Qi(s, a, s') )
  L   = L_1 + L_2

i.e. Bellman residual on the future goal s_f, plus maximize Q on the
immediate next state s' (packed as ``obs_to_goal(s')``).

Cross-batch goals (``cross_batch_goals`` / ``ppo_td3_cross_batch_goals``):
  For a batch of size B, each transition ``(s_i, a_i, s'_i)`` is trained against
  **every** future goal ``g_j = obs_to_goal(s_f_j)`` in the batch (B² pairs)::

      Q(s_i, a_i, g_j) ← 1[s'_i ≈ g_j] + γ min Q̄(s'_i, a'_{ij}, g_j)

  With FB loss the indicator is dropped (``y = γ min Q̄``); the
  ``−Q(s,a,s')`` term stays per-transition (diagonal).  With the flag off,
  only the diagonal pairs ``(i, i)`` are used.

Optional target policy (``use_target_policy`` / ``ppo_td3_use_target_policy``):
  a' ~ π̄(· | s', g) and π̄ ← (1−τ)·π̄ + τ·π after each critic step
  (same τ as the Q targets).

This is the non-absorbing discounted occupancy backup:
  Q(s,a,s_f) ← 1[s'=s_f] + γ min_i Q̄_i(s', a', s_f)

``τ`` here is the TD3 target-network coefficient (``config.tau`` /
``ppo_td3_tau``).  It is **independent** of ``ppo_crl_repr_tau`` (EMA of
φ/ψ used only for CRL-mode PPO rewards).

PPO reward
----------
    r_t = Q1_θ(s_t, a_t, g)    (online twin critic)
  or, if ``ppo_td3_reward_tau`` ∈ (0, 1):
    r_t = Q1_ema(s_t, a_t, g)  with ema ← τ·ema + (1−τ)·online
                               (independent of Polyak target τ)
  If ``ppo_td3_log_reward`` is True, the scalar above is replaced by
    r_t = log((1 − γ) · max(Q, ε))
  so the PPO reward matches a log-occupancy / log-density scale
  (same units as Gaussian / NF ``log p`` rewards).

Usage
-----
Set ``config.ppo_repr_mode = 'td3'``.
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Sequence

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

class Td3DensityNetworks(NamedTuple):
  """Twin scalar Q networks for Q(s, a, g).

  Each ``qf*_net`` is a FeedForwardNetwork:
      init(key) / apply(params, obs, action) → (B,)
  where ``obs = [state; goal]``.
  """
  qf1_net: networks_lib.FeedForwardNetwork
  qf2_net: networks_lib.FeedForwardNetwork


# ---------------------------------------------------------------------------
# 2.  Network factory
# ---------------------------------------------------------------------------

def make_td3_density_networks(
    obs_dim: int,
    act_dim: int,
    goal_dim: int,
    hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256, 256),
    repr_dim: int = 64,
    bilinear: bool = False,
    repr_norm: bool = False,
) -> Td3DensityNetworks:
  """Build twin Q(s, a, g) critics.

  Args:
    bilinear: If False (default), each critic is an MLP on ``[s;g;a]``.
      If True, ``Q = x(s,a)·y(g)`` with x/y matching CRL φ/ψ
      (``sa_encoder`` / ``g_encoder``, widths ``hidden + [repr_dim]``).
    repr_dim: Output dim of x and y when ``bilinear`` (same as CRL).
    repr_norm: If True and bilinear, L2-normalize x and y (as CRL).
  """

  def _make_mlp_q_fn(net_name: str):
    def _q_fn(obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
      x = jnp.concatenate([obs, action], axis=-1)
      w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
      trunk = _mlp_or_residual(
          x,
          list(hidden_layer_sizes),
          hidden_layer_sizes=hidden_layer_sizes,
          name=net_name,
          activation=jax.nn.relu,
          activate_final=True,
          w_init=w_init,
      )
      head = hk.Linear(1, w_init=w_init, name=net_name + '_head')
      return jnp.squeeze(head(trunk), axis=-1)
    return _q_fn

  def _make_bilinear_q_fn(net_name: str):
    # Same arch as CRL φ(s,a) / ψ(g) in contrastive.networks._repr_fn.
    def _q_fn(obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
      state = obs[:, :obs_dim]
      goal = obs[:, obs_dim:]
      w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
      x_sa = _mlp_or_residual(
          jnp.concatenate([state, action], axis=-1),
          list(hidden_layer_sizes) + [repr_dim],
          hidden_layer_sizes=hidden_layer_sizes,
          name=net_name + '_sa_encoder',
          activation=jax.nn.relu,
          activate_final=False,
          w_init=w_init,
      )
      y_g = _mlp_or_residual(
          goal,
          list(hidden_layer_sizes) + [repr_dim],
          hidden_layer_sizes=hidden_layer_sizes,
          name=net_name + '_g_encoder',
          activation=jax.nn.relu,
          activate_final=False,
          w_init=w_init,
      )
      if repr_norm:
        x_sa = x_sa / jnp.linalg.norm(x_sa, axis=1, keepdims=True)
        y_g = y_g / jnp.linalg.norm(y_g, axis=1, keepdims=True)
      return jnp.sum(x_sa * y_g, axis=-1)
    return _q_fn

  _factory = _make_bilinear_q_fn if bilinear else _make_mlp_q_fn
  qf1_t = hk.without_apply_rng(hk.transform(_factory('qf1')))
  qf2_t = hk.without_apply_rng(hk.transform(_factory('qf2')))

  dummy_obs = np.zeros((1, obs_dim + goal_dim), dtype=np.float32)
  dummy_act = np.zeros((1, act_dim), dtype=np.float32)

  return Td3DensityNetworks(
      qf1_net=networks_lib.FeedForwardNetwork(
          init=lambda key: qf1_t.init(key, dummy_obs, dummy_act),
          apply=qf1_t.apply),
      qf2_net=networks_lib.FeedForwardNetwork(
          init=lambda key: qf2_t.init(key, dummy_obs, dummy_act),
          apply=qf2_t.apply),
  )


def init_td3_params(
    nets: Td3DensityNetworks,
    key: networks_lib.PRNGKey,
) -> Dict[str, networks_lib.Params]:
  """Init online twin Qs and copy them into Polyak targets."""
  k1, k2 = jax.random.split(key)
  qf1 = nets.qf1_net.init(k1)
  qf2 = nets.qf2_net.init(k2)
  return {
      'qf1': qf1,
      'qf2': qf2,
      'qf1_target': jax.tree_util.tree_map(lambda x: x, qf1),
      'qf2_target': jax.tree_util.tree_map(lambda x: x, qf2),
  }


def online_td3_params(q_params: Dict[str, networks_lib.Params]):
  """Optimizer / grad-bearing subset of ``q_params``."""
  return {'qf1': q_params['qf1'], 'qf2': q_params['qf2']}


# ---------------------------------------------------------------------------
# 3.  TD3 critic update
# ---------------------------------------------------------------------------

def make_td3_density_update_fn(
    density_nets: Td3DensityNetworks,
    policy_network: networks_lib.FeedForwardNetwork,
    sample_fn,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    start_index: int = 0,
    end_index: int = -1,
    discount: float = 0.99,
    tau: float = 0.005,
    goal_tol: float = 1e-2,
    use_target_policy: bool = False,
    cross_batch_goals: bool = True,
    normalize_obs: bool = False,
    obs_norm_clip: float = 10.0,
    goal_state_indices=None,
    fb_loss: bool = False,
):
  """Jitted TD3 twin-Q update over one EpisodeReplay batch.

  Signature::

      update(q_params, opt_state, batch, key, policy_params, policy_target_params)
      → (new_q_params, new_opt_state, metrics, new_policy_target_params)

  ``tau`` is the Polyak coefficient for target Q nets (and for π̄ when
  ``use_target_policy``).  Unrelated to ``ppo_crl_repr_tau``.

  When ``use_target_policy`` is True, bootstrap actions a' are sampled from
  ``policy_target_params`` and that target is Polyak-updated toward the
  online PPO policy after each successful critic step.  When False, a' comes
  from the online policy (stop-grad) and ``policy_target_params`` is returned
  unchanged.

  When ``cross_batch_goals`` is True (default), each ``(s_i, a_i, s'_i)`` is
  paired with every batch goal ``g_j`` (B² TD backups).  When False, only the
  paired goal ``g_i`` is used.

  When ``fb_loss`` is True, use the FB objective
  ``(Q(s,a,s_f) − γ sg Q(s',a',s_f))^2 − Q(s,a,s')`` instead of the
  indicator TD3 backup (``s_f`` still sampled as usual).
  """
  gamma = float(discount)
  polyak = float(tau)
  tol = float(goal_tol)
  si = int(start_index)
  ei = int(end_index)
  use_pi_bar = bool(use_target_policy)
  use_cross = bool(cross_batch_goals)
  use_fb = bool(fb_loss)
  use_obs_norm = bool(normalize_obs)
  norm_clip = float(obs_norm_clip)
  norm_ei = int(obs_dim if ei == -1 else ei)
  _gidx = None if goal_state_indices is None else jnp.asarray(
      goal_state_indices, dtype=jnp.int32)

  def _state_as_goal(state: jnp.ndarray) -> jnp.ndarray:
    if _gidx is not None:
      return state[:, _gidx]
    if ei == -1:
      return state[:, si:]
    return state[:, si:ei]

  def _normalize(obs, obs_mean, obs_var):
    if not use_obs_norm:
      return obs
    active = jnp.all(jnp.isfinite(obs_mean))
    obs_mean = jnp.nan_to_num(obs_mean)
    state = obs[..., :obs_dim]
    goal = obs[..., obs_dim:]
    state = (state - obs_mean) / jnp.sqrt(jnp.maximum(obs_var, 1e-8))
    if _gidx is not None:
      g_mean, g_var = obs_mean[_gidx], obs_var[_gidx]
    else:
      g_mean, g_var = obs_mean[si:norm_ei], obs_var[si:norm_ei]
    goal = (goal - g_mean) / jnp.sqrt(jnp.maximum(g_var, 1e-8))
    normalized = jnp.clip(jnp.concatenate([state, goal], axis=-1),
                          -norm_clip, norm_clip)
    return jnp.where(active, normalized, obs)

  def _critic_loss(
      online_params, batch, key, policy_for_bootstrap, target_params,
      obs_mean, obs_var):
    obs = batch['obs']            # [s ; g],  g = obs_to_goal(s_f)
    action = batch['action']
    next_obs = batch['next_obs']  # [s' ; g]

    goal = obs[:, obs_dim:]
    state = obs[:, :obs_dim]
    s_next = next_obs[:, :obs_dim]
    s_next_as_g = _state_as_goal(s_next)

    if use_cross:
      # (i, j) = transition i × goal j  →  flatten to B² rows.
      B = obs.shape[0]
      state_ij = jnp.broadcast_to(
          state[:, None, :], (B, B, state.shape[-1]))
      goal_ij = jnp.broadcast_to(
          goal[None, :, :], (B, B, goal.shape[-1]))
      action_ij = jnp.broadcast_to(
          action[:, None, :], (B, B, action.shape[-1]))
      s_next_ij = jnp.broadcast_to(
          s_next[:, None, :], (B, B, s_next.shape[-1]))
      obs_flat = jnp.concatenate(
          [state_ij, goal_ij], axis=-1).reshape(B * B, -1)
      next_obs_flat = jnp.concatenate(
          [s_next_ij, goal_ij], axis=-1).reshape(B * B, -1)
      action_flat = action_ij.reshape(B * B, -1)
      # r_ij = 1[s'_i ≈ g_j]
      rewards = (jnp.linalg.norm(
          s_next_as_g[:, None, :] - goal[None, :, :], axis=-1) < tol).astype(
              obs.dtype)  # (B, B)
      rewards_flat = rewards.reshape(B * B)
    else:
      obs_flat = obs
      next_obs_flat = next_obs
      action_flat = action
      rewards = (jnp.linalg.norm(s_next_as_g - goal, axis=-1) < tol).astype(
          obs.dtype)  # (B,)
      rewards_flat = rewards

    obs_network = _normalize(obs_flat, obs_mean, obs_var)
    next_obs_network = _normalize(next_obs_flat, obs_mean, obs_var)

    # a' ~ π or π̄ (· | s', g) with frozen weights.
    key, k_act = jax.random.split(key)
    next_dist = policy_network.apply(
        jax.lax.stop_gradient(policy_for_bootstrap), next_obs_network)
    next_action = sample_fn(next_dist, k_act)

    q1_next = density_nets.qf1_net.apply(
        target_params['qf1_target'], next_obs_network, next_action)
    q2_next = density_nets.qf2_net.apply(
        target_params['qf2_target'], next_obs_network, next_action)
    min_next = jnp.minimum(q1_next, q2_next)
    if use_fb:
      # FB: y = γ · stopgrad(min Q̄(s', a', s_f))  (no indicator)
      target_q = jax.lax.stop_gradient(gamma * min_next)
    else:
      # Non-absorbing backup: y = 1[s'≈g] + γ min Q̄(s', a', g)
      target_q = jax.lax.stop_gradient(rewards_flat + gamma * min_next)

    q1 = density_nets.qf1_net.apply(
        online_params['qf1'], obs_network, action_flat)
    q2 = density_nets.qf2_net.apply(
        online_params['qf2'], obs_network, action_flat)
    qf1_mse = jnp.mean((q1 - target_q) ** 2)
    qf2_mse = jnp.mean((q2 - target_q) ** 2)

    if use_fb:
      # −Q(s, a, s'): maximize Q on the immediate next state as goal.
      # Always diagonal over transitions (not cross-batch goals).
      obs_sprime = jnp.concatenate([state, s_next_as_g], axis=-1)
      obs_sprime_network = _normalize(obs_sprime, obs_mean, obs_var)
      q1_sprime = density_nets.qf1_net.apply(
          online_params['qf1'], obs_sprime_network, action)
      q2_sprime = density_nets.qf2_net.apply(
          online_params['qf2'], obs_sprime_network, action)
      q1_sprime_mean = jnp.mean(q1_sprime)
      q2_sprime_mean = jnp.mean(q2_sprime)
      qf1_loss = qf1_mse - q1_sprime_mean
      qf2_loss = qf2_mse - q2_sprime_mean
    else:
      q1_sprime_mean = jnp.array(0.0, dtype=obs.dtype)
      q2_sprime_mean = jnp.array(0.0, dtype=obs.dtype)
      qf1_loss = qf1_mse
      qf2_loss = qf2_mse
    loss = qf1_loss + qf2_loss

    metrics = {
        'td3_qf_loss': loss,
        'td3_qf1_loss': qf1_loss,
        'td3_qf2_loss': qf2_loss,
        'td3_qf1_mse': qf1_mse,
        'td3_qf2_mse': qf2_mse,
        'td3_q1_mean': jnp.mean(q1),
        'td3_q2_mean': jnp.mean(q2),
        'td3_target_mean': jnp.mean(target_q),
        'td3_reward_mean': jnp.mean(rewards_flat),
        'td3_goal_hit_frac': jnp.mean(rewards_flat),
        'td3_fb_loss': jnp.asarray(float(use_fb), dtype=obs.dtype),
        'td3_q1_sprime_mean': q1_sprime_mean,
        'td3_q2_sprime_mean': q2_sprime_mean,
    }
    if use_cross:
      # Diagonal = original paired (s_i, g_i) hits; useful sanity check.
      metrics['td3_goal_hit_frac_diag'] = jnp.mean(jnp.diag(rewards))
    return loss, metrics

  grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)

  @jax.jit
  def update(
      q_params, opt_state, batch, key, policy_params, policy_target_params,
      obs_mean=None, obs_var=None):
    online = online_td3_params(q_params)
    target = {
        'qf1_target': q_params['qf1_target'],
        'qf2_target': q_params['qf2_target'],
    }
    policy_for_bootstrap = (
        policy_target_params if use_pi_bar else policy_params)
    (_, metrics), grads = grad_fn(
        online, batch, key, policy_for_bootstrap, target, obs_mean, obs_var)

    grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))))
    loss_finite = jnp.isfinite(metrics['td3_qf_loss'])
    do_update = jnp.logical_and(grads_finite, loss_finite)

    def _apply(_):
      updates, new_opt = optimizer.update(grads, opt_state, online)
      new_online = optax.apply_updates(online, updates)
      # Polyak: target ← (1−τ)·target + τ·online
      new_qf1_t = jax.tree_util.tree_map(
          lambda t, o: (1.0 - polyak) * t + polyak * o,
          q_params['qf1_target'], new_online['qf1'])
      new_qf2_t = jax.tree_util.tree_map(
          lambda t, o: (1.0 - polyak) * t + polyak * o,
          q_params['qf2_target'], new_online['qf2'])
      new_q = {
          'qf1': new_online['qf1'],
          'qf2': new_online['qf2'],
          'qf1_target': new_qf1_t,
          'qf2_target': new_qf2_t,
      }
      if use_pi_bar:
        new_pi_t = jax.tree_util.tree_map(
            lambda t, o: (1.0 - polyak) * t + polyak * o,
            policy_target_params, policy_params)
      else:
        new_pi_t = policy_target_params
      return new_q, new_opt, new_pi_t

    def _skip(_):
      return q_params, opt_state, policy_target_params

    new_q_params, new_opt_state, new_pi_target = jax.lax.cond(
        do_update, _apply, _skip, operand=None)
    metrics = dict(metrics)
    metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
    return new_q_params, new_opt_state, metrics, new_pi_target

  return update


def make_scan_td3_update_fn(
    density_nets: Td3DensityNetworks,
    policy_network: networks_lib.FeedForwardNetwork,
    sample_fn,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    start_index: int = 0,
    end_index: int = -1,
    discount: float = 0.99,
    tau: float = 0.005,
    goal_tol: float = 1e-2,
    use_target_policy: bool = False,
    cross_batch_goals: bool = True,
    normalize_obs: bool = False,
    obs_norm_clip: float = 10.0,
    goal_state_indices=None,
    fb_loss: bool = False,
    repr_tau: float = 0.0,
):
  """Scan-based TD3 updater: N critic steps in one JIT call.

  Caller pre-samples all N batches into ``(N, B, …)`` arrays.  Reward-param
  EMA (``0 < repr_tau < 1``) is applied inside the scan.

  Returns:
    ``multi_update(q_params, opt_state, params_ema, batches, key,
                   policy_params, policy_target_params, obs_mean, obs_var)``
    → ``(new_q, new_opt, new_ema, new_key, new_policy_target, mean_metrics)``
  """
  raw_update = make_td3_density_update_fn(
      density_nets,
      policy_network=policy_network,
      sample_fn=sample_fn,
      optimizer=optimizer,
      obs_dim=obs_dim,
      start_index=start_index,
      end_index=end_index,
      discount=discount,
      tau=tau,
      goal_tol=goal_tol,
      use_target_policy=use_target_policy,
      cross_batch_goals=cross_batch_goals,
      normalize_obs=normalize_obs,
      obs_norm_clip=obs_norm_clip,
      goal_state_indices=goal_state_indices,
      fb_loss=fb_loss,
  )
  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def multi_update(
      q_params, opt_state, params_ema, batches, key,
      policy_params, policy_target_params, obs_mean, obs_var):
    def scan_step(carry, batch):
      q_p, opt, ema, k, pi_t = carry
      k, k_u = jax.random.split(k)
      q_p, opt, m, pi_t = raw_update(
          q_p, opt, batch, k_u, policy_params, pi_t, obs_mean, obs_var)
      if use_ema:
        ema = jax.tree_util.tree_map(
            lambda t, o: _tau * t + (1.0 - _tau) * o, ema, q_p)
      else:
        ema = q_p
      return (q_p, opt, ema, k, pi_t), m

    (q_params, opt_state, params_ema, key, policy_target_params), metrics = (
        jax.lax.scan(
            scan_step,
            (q_params, opt_state, params_ema, key, policy_target_params),
            batches))
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return (q_params, opt_state, params_ema, key, policy_target_params,
            metrics)

  return multi_update


# ---------------------------------------------------------------------------
# 4.  PPO reward: r = Q1(s, a, g)  or  log((1−γ) Q1)
# ---------------------------------------------------------------------------

def make_td3_reward_fn(
    density_nets: Td3DensityNetworks,
    obs_dim: int,
    discount: float = 0.99,
    log_reward: bool = False,
    q_eps: float = 1e-8,
    start_index: int = 0,
    end_index: int = -1,
    normalize_obs: bool = False,
    obs_norm_clip: float = 10.0,
    goal_state_indices=None,
):
  """Jitted PPO reward from the online (or EMA) twin critic.

  Default: ``r = Q1(s, a, g)``.
  If ``log_reward``: ``r = log((1 − γ) · max(Q1, q_eps))``.
  """
  one_m_gamma = float(1.0 - float(discount))
  use_log = bool(log_reward)
  eps = float(q_eps)
  si = int(start_index)
  ei = int(obs_dim if end_index == -1 else end_index)
  use_obs_norm = bool(normalize_obs)
  norm_clip = float(obs_norm_clip)
  _gidx = None if goal_state_indices is None else jnp.asarray(
      goal_state_indices, dtype=jnp.int32)

  @jax.jit
  def reward_fn(
      q_params, obs: jnp.ndarray, action: jnp.ndarray,
      obs_mean: jnp.ndarray = None,
      obs_var: jnp.ndarray = None) -> jnp.ndarray:
    if use_obs_norm:
      active = jnp.all(jnp.isfinite(obs_mean))
      obs_mean = jnp.nan_to_num(obs_mean)
      raw_obs = obs
      state = ((obs[..., :obs_dim] - obs_mean)
               / jnp.sqrt(jnp.maximum(obs_var, 1e-8)))
      if _gidx is not None:
        g_mean, g_var = obs_mean[_gidx], obs_var[_gidx]
      else:
        g_mean, g_var = obs_mean[si:ei], obs_var[si:ei]
      goal = ((obs[..., obs_dim:] - g_mean)
              / jnp.sqrt(jnp.maximum(g_var, 1e-8)))
      obs = jnp.clip(
          jnp.concatenate([state, goal], axis=-1), -norm_clip, norm_clip)
      obs = jnp.where(active, obs, raw_obs)
    q1 = density_nets.qf1_net.apply(q_params['qf1'], obs, action)
    if use_log:
      return jnp.log(one_m_gamma * jnp.maximum(q1, eps))
    return q1

  return reward_fn
