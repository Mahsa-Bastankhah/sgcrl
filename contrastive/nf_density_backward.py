"""Backward normalizing flow  p_θ(s_goal | s_f)  trained alongside the forward NF.

The forward model in ``nf_density.py`` learns ``log p(s_f | s, a)`` and is the
PPO reward.  This module is a *sidecar*: same replay batches, independent
params, never used for PPO.

    s_goal = obs_to_goal(s)     # waypoint xyz (goal-space slice of s)
    y      = encoder(s_f)       # condition on the later hindsight goal
    z, _   = RealNVP(s_goal_norm ; y)
    log p  = log N(z; 0, I) + log|det|

Train-time: normalize + Gaussian noise on **s_goal** (same knobs as forward
uses on s_f: ``nf_normalize_goals``, ``nf_noise_std``, ``nf_goal_std_min``).
Conditioning s_f is raw, matching how forward leaves (s, a) unnormalized.

Sampling later from a checkpoint::

    from contrastive.nf_density_backward import (
        make_nf_backward_networks, nf_backward_sample)
    extra = ckpt['extra_state']
    s_goal = nf_backward_sample(
        nets, extra['nf_backward_params'], sf, key,
        extra['nf_bwd_s_mean'], extra['nf_bwd_s_std'])
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from contrastive import nf_density as _nf


def state_as_goal(state, start_index: int, end_index: int,
                  goal_state_indices=None):
  """Packed state → waypoint / goal-space coords (numpy or jax).

  Matches ``EpisodeReplay._obs_to_goal`` / ``obs_to_goal_2d``.
  """
  if goal_state_indices is not None:
    idx = np.asarray(goal_state_indices, dtype=np.int32)
    return state[..., idx]
  si = int(start_index)
  ei = int(end_index)
  if ei == -1:
    return state[..., si:]
  return state[..., si:ei]


def make_nf_backward_networks(
    goal_dim: int,
    hidden_layer_sizes: Sequence[int] = (),
    rep_size: int = 64,
    num_blocks: int = 8,
    channels: int = 256,
    sa_hidden: int = 1024,
    sa_num_layers: int = 4,
    scale_tanh: bool = False,
    scale_tanh_c: float = 2.0,
) -> _nf.NFDensityNetworks:
  """Same RealNVP as forward, with roles swapped and no action.

  Encoder input is s_f (goal_dim).  Flow input is s_goal (goal_dim).
  ``goal_enc_size`` is always 0 so the flow is invertible for sampling.
  """
  return _nf.make_nf_density_networks(
      obs_dim=int(goal_dim),
      act_dim=1,
      goal_dim=int(goal_dim),
      hidden_layer_sizes=hidden_layer_sizes,
      rep_size=rep_size,
      num_blocks=num_blocks,
      channels=channels,
      goal_enc_size=0,
      sa_hidden=sa_hidden,
      sa_num_layers=sa_num_layers,
      state_only=True,
      scale_tanh=scale_tanh,
      scale_tanh_c=scale_tanh_c,
  )


def _dummy_action(n: int, dtype):
  return jnp.zeros((n, 1), dtype=dtype)


def nf_backward_log_prob(nf_networks: _nf.NFDensityNetworks, params,
                         s_goal, sf):
  """log p(s_goal | s_f).  ``s_goal`` must already be normalized."""
  dummy = _dummy_action(int(sf.shape[0]), sf.dtype)
  return _nf.nf_log_prob(nf_networks, params, sf, dummy, s_goal)


def nf_backward_sample(nf_networks: _nf.NFDensityNetworks, params,
                       sf, key, s_mean, s_std):
  """Draw raw waypoint samples  s_goal ~ p(s_goal | s_f).

  ``sf``, ``s_mean``, ``s_std`` are unnormalized (checkpoint extras
  ``nf_bwd_s_mean`` / ``nf_bwd_s_std``).  ``sf`` is the raw later goal.
  """
  dummy = _dummy_action(int(sf.shape[0]), sf.dtype)
  s_norm = _nf.nf_sample(nf_networks, params, sf, dummy, key)
  return s_norm * (s_std + 1e-8) + s_mean


def make_nf_backward_update_fn(
    nf_networks: _nf.NFDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    start_index: int,
    end_index: int,
    goal_state_indices=None,
    noise_std: float = 0.0,
):
  """Jitted NLL update for p(s_goal | s_f) on a forward-NF replay batch.

  Batch layout is the usual CRL dict: ``obs = [s ; s_f]``.  s_goal is
  ``obs_to_goal(s)``.  ``s_mean`` / ``s_std`` are per-dim stats of s_goal.
  """
  si = int(start_index)
  ei = int(end_index)
  _gidx = None if goal_state_indices is None else jnp.asarray(
      goal_state_indices, dtype=jnp.int32)
  _noise_std = float(noise_std)

  def _s_goal(state: jnp.ndarray) -> jnp.ndarray:
    if _gidx is not None:
      return state[:, _gidx]
    if ei == -1:
      return state[:, si:]
    return state[:, si:ei]

  def _loss(params, batch, key, s_mean, s_std):
    obs = batch['obs']
    state = obs[:, :obs_dim]
    sf = obs[:, obs_dim:]
    s_g = _s_goal(state)
    s_g = (s_g - s_mean) / (s_std + 1e-8)
    if _noise_std > 0.0:
      key, k_n = jax.random.split(key)
      s_g = s_g + _noise_std * jax.random.normal(k_n, s_g.shape)
    dummy = _dummy_action(int(sf.shape[0]), sf.dtype)
    y = nf_networks.sa_encoder_net.apply(params['sa_encoder'], sf, dummy)
    log_p, s_raw_mean, s_mean_c = nf_networks.flow_net.apply(
        params['nf_flow'], s_g, y, return_stats=True)
    nll = -jnp.mean(log_p)
    metrics = {
        'density_loss': nll,
        'log_p_mean': jnp.mean(log_p),
        'log_p_min': jnp.min(log_p),
        'log_p_max': jnp.max(log_p),
        'repr_norm': jnp.mean(jnp.linalg.norm(y, axis=-1)),
        's_raw_mean': s_raw_mean,
        's_mean': s_mean_c,
    }
    return nll, metrics

  grad_fn = jax.value_and_grad(_loss, has_aux=True)

  def update(params, opt_state, batch, key, s_mean, s_std):
    (_, metrics), grads = grad_fn(params, batch, key, s_mean, s_std)
    metrics = dict(metrics)
    metrics['encoder_grad_norm'] = _nf._tree_l2_norm(grads['sa_encoder'])
    metrics['flow_grad_norm'] = _nf._tree_l2_norm(grads['nf_flow'])

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
    metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
    return new_params, new_opt_state, metrics

  return jax.jit(update)


def make_scan_nf_backward_update_fn(
    nf_networks: _nf.NFDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
    start_index: int,
    end_index: int,
    goal_state_indices=None,
    noise_std: float = 0.0,
):
  """Scan-based backward NF: N NLL steps in one JIT, same stacked batches.

  Returns
    ``multi_update(params, opt_state, batches, key, s_mean, s_std)``
    → ``(new_params, new_opt_state, new_key, mean_metrics)``
  """
  raw_update = make_nf_backward_update_fn(
      nf_networks, optimizer, obs_dim=obs_dim,
      start_index=start_index, end_index=end_index,
      goal_state_indices=goal_state_indices, noise_std=noise_std)

  @jax.jit
  def multi_update(params, opt_state, batches, key, s_mean, s_std):
    def scan_step(carry, batch):
      p, opt, k = carry
      k, k_u = jax.random.split(k)
      p, opt, m = raw_update(p, opt, batch, k_u, s_mean, s_std)
      return (p, opt, k), m

    (params, opt_state, key), metrics = jax.lax.scan(
        scan_step, (params, opt_state, key), batches)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return params, opt_state, key, metrics

  return multi_update
