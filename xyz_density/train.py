#!/usr/bin/env python3
"""Train CRL / NF / TD3 density estimators on the controlled xyz env.

Collects data with a uniform random policy over 1024 parallel JAX envs
(episode length 50), stores episodes in ``EpisodeReplay`` (geometric future
sampling, same as PPO), and runs ``ppo_crl_steps_per_iter`` density updates
per iteration.

Only the density estimators / CRL update + EpisodeReplay are imported from
``contrastive/``; the env and loop live in this folder.

Example::

  python -u xyz_density/train.py --repr_mode=crl --num_envs=1024 \\
      --episode_length=50 --density_steps_per_iter=25 --max_replay_size=1000000
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict

# Repo root on sys.path when launched as ``python xyz_density/train.py``.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401  # must precede acme / jax imports

import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme import specs
from acme.jax import networks as networks_lib

from contrastive import fm_density as _fm
from contrastive import nf_density as _nf
from contrastive import td3_density as _td3
from contrastive.networks import make_networks
from contrastive.ppo_learner import (
    EpisodeReplay,
    make_crl_update_fn,
    make_td_infonce_update_fn,
)
from xyz_density.env import JaxXYZVecEnv

# Shared OOD density-probe action seed (identical ax, ay for every model).
_PROBE_ACTION_SEED = 42_4242


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tree_to_numpy(tree):
  return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _mean_metrics(agg: Dict[str, list]) -> Dict[str, float]:
  return {k: float(np.mean(v)) for k, v in agg.items() if v}


def _make_env_spec(obs_dim: int, act_dim: int, goal_dim: int) -> specs.EnvironmentSpec:
  """Packed observation ``[s; g]`` for CRL / TD3 policy nets."""
  total = obs_dim + goal_dim
  return specs.EnvironmentSpec(
      observations=specs.Array(shape=(total,), dtype=np.float32, name='obs'),
      actions=specs.BoundedArray(
          shape=(act_dim,), dtype=np.float32, name='action',
          minimum=-1.0, maximum=1.0),
      rewards=specs.Array(shape=(), dtype=np.float32, name='reward'),
      discounts=specs.BoundedArray(
          shape=(), dtype=np.float32, name='discount',
          minimum=0.0, maximum=1.0),
  )


def _make_uniform_bootstrap_policy(act_dim: int, zero_az: bool = False):
  """TD3 bootstrap a' ~ Uniform[-1, 1] (matches the data-collection policy)."""

  def init(_key):
    return {}

  def apply(_params, obs):
    b = obs.shape[0]

    class _UniformDist:
      def sample(self, seed):
        a = jax.random.uniform(
            seed, shape=(b, act_dim), minval=-1.0, maxval=1.0)
        if zero_az and act_dim >= 3:
          a = a.at[..., 2].set(0.0)
        return a

    return _UniformDist()

  net = networks_lib.FeedForwardNetwork(init=init, apply=apply)
  sample_fn = lambda dist, key: dist.sample(seed=key)
  return net, sample_fn


def _analytic_gaussian_log_prob(
    state: np.ndarray,
    action: np.ndarray,
    goal: np.ndarray,
    noise_std: float,
    active_dims: slice | None = None,
) -> np.ndarray:
  """One-step ground truth: log N(goal; state+action, σ² I).

  ``active_dims`` restricts the Gaussian to a coordinate slice (e.g. xy when
  ``zero_az`` freezes z).
  """
  if active_dims is not None:
    state = state[..., active_dims]
    action = action[..., active_dims]
    goal = goal[..., active_dims]
  resid = (goal - state - action) / noise_std
  d = state.shape[-1]
  log_norm = -0.5 * d * np.log(2.0 * np.pi) - d * np.log(noise_std)
  return log_norm - 0.5 * np.sum(resid ** 2, axis=-1)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
  """Spearman rank correlation; returns NaN if undefined."""
  x = np.asarray(x, dtype=np.float64).ravel()
  y = np.asarray(y, dtype=np.float64).ravel()
  if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
    return float('nan')
  # Rank via average ranks (scipy-free).
  def _rank(v):
    order = np.argsort(v, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    # Average ties.
    sorted_v = v[order]
    i = 0
    while i < len(v):
      j = i
      while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
        j += 1
      if j > i:
        avg = 0.5 * (i + j)
        ranks[order[i:j + 1]] = avg
      i = j + 1
    return ranks
  rx, ry = _rank(x), _rank(y)
  rx -= rx.mean()
  ry -= ry.mean()
  denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
  if denom < 1e-12:
    return float('nan')
  return float(np.sum(rx * ry) / denom)


def _collect_random_episodes(
    env: JaxXYZVecEnv,
    rng: np.random.Generator,
    discount: float,
) -> EpisodeReplay:
  """Roll one full episode per env into a fresh EpisodeReplay (validation)."""
  E = env.num_envs
  T = env.episode_length
  replay = EpisodeReplay(
      capacity=E * T + 1,
      obs_dim=env.obs_dim,
      discount=discount,
      start_index=0,
      end_index=env.obs_dim,
  )
  obs = env.reset()
  ep_obs = [[obs[i].copy()] for i in range(E)]
  ep_act: list = [[] for _ in range(E)]
  for _t in range(T):
    action = env.sample_uniform_actions(rng)
    next_obs, _rew, dones, terminal_obs, _info = env.step(action)
    for i in range(E):
      ep_act[i].append(action[i].copy())
      if dones[i]:
        ep_obs[i].append(terminal_obs[i].copy())
        try:
          replay.add_episode(
              np.stack(ep_obs[i], axis=0),
              np.stack(ep_act[i], axis=0),
              successful=False)
        except AssertionError:
          pass
        ep_obs[i] = [next_obs[i].copy()]
        ep_act[i] = []
      else:
        ep_obs[i].append(next_obs[i].copy())
    obs = next_obs
  return replay


def _score_matrix(
    mode: str,
    *,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    goals: jnp.ndarray,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> jnp.ndarray:
  """Return (B, B) scores where entry [i, j] scores goal_j under (s_i, a_i).

  Higher is better for all modes:
    CRL / TDInfoNCE → φ(s,a)·ψ(g)
    NF / FM → log p(g|s,a)
    TD3 → Q1(s,a,g)
  """
  b = int(states.shape[0])
  if mode in ('crl', 'tdinfonce'):
    packed = jnp.concatenate([states, goals], axis=-1)
    logits, _, _ = networks.q_network.apply(q_params, packed, actions)
    if logits.ndim == 3:
      logits = jnp.mean(logits, axis=-1)
    return logits

  if mode == 'nf':
    g_norm = (goals - jnp.asarray(nf_goal_mean)) / (
        jnp.asarray(nf_goal_std) + 1e-8)
    s_ij = jnp.repeat(states, b, axis=0)
    a_ij = jnp.repeat(actions, b, axis=0)
    g_ij = jnp.tile(g_norm, (b, 1))
    log_p = _nf.nf_log_prob(nf_nets, q_params, s_ij, a_ij, g_ij)
    return log_p.reshape(b, b)

  if mode == 'fm':
    # Chunk B×B exact ODE log-probs to avoid GPU OOM / long compiles.
    s_np = np.asarray(states)
    a_np = np.asarray(actions)
    g_np = np.asarray(goals)
    out = np.empty((b, b), dtype=np.float32)
    chunk = 256
    for i0 in range(0, b, chunk):
      i1 = min(i0 + chunk, b)
      s_blk = jnp.repeat(jnp.asarray(s_np[i0:i1]), b, axis=0)
      a_blk = jnp.repeat(jnp.asarray(a_np[i0:i1]), b, axis=0)
      g_blk = jnp.tile(jnp.asarray(g_np), (i1 - i0, 1))
      # Secondary chunk along columns for large B.
      col_chunk = 256
      parts = []
      n_blk = int(s_blk.shape[0])
      for j0 in range(0, n_blk, col_chunk):
        j1 = min(j0 + col_chunk, n_blk)
        parts.append(np.asarray(_fm.fm_log_prob(
            fm_nets, q_params, s_blk[j0:j1], a_blk[j0:j1], g_blk[j0:j1],
            mode='exact')))
      out[i0:i1] = np.concatenate(parts, axis=0).reshape(i1 - i0, b)
    return jnp.asarray(out)

  # td3
  s_ij = jnp.repeat(states, b, axis=0)
  a_ij = jnp.repeat(actions, b, axis=0)
  g_ij = jnp.tile(goals, (b, 1))
  packed = jnp.concatenate([s_ij, g_ij], axis=-1)
  q1 = td3_nets.qf1_net.apply(q_params['qf1'], packed, a_ij)
  return q1.reshape(b, b)


def _score_candidates(
    mode: str,
    *,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    state: np.ndarray,
    action: np.ndarray,
    goals: np.ndarray,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> np.ndarray:
  """Score many goals under a single fixed (s, a). Returns shape (N,)."""
  n = int(goals.shape[0])
  s = jnp.broadcast_to(jnp.asarray(state)[None, :], (n, state.shape[-1]))
  a = jnp.broadcast_to(jnp.asarray(action)[None, :], (n, action.shape[-1]))
  g = jnp.asarray(goals)
  if mode in ('crl', 'tdinfonce'):
    packed = jnp.concatenate([s, g], axis=-1)
    _, sa_repr, g_repr = networks.q_network.apply(q_params, packed, a)
    return np.asarray(jnp.sum(sa_repr * g_repr, axis=-1))
  if mode == 'nf':
    g_norm = (g - jnp.asarray(nf_goal_mean)) / (
        jnp.asarray(nf_goal_std) + 1e-8)
    return np.asarray(_nf.nf_log_prob(nf_nets, q_params, s, a, g_norm))
  if mode == 'fm':
    return np.asarray(
        _fm.fm_log_prob(fm_nets, q_params, s, a, g, mode='exact'))
  packed = jnp.concatenate([s, g], axis=-1)
  return np.asarray(td3_nets.qf1_net.apply(q_params['qf1'], packed, a))


def _softmax_normalize(scores: np.ndarray) -> np.ndarray:
  x = np.asarray(scores, dtype=np.float64)
  x = x - np.max(x)
  e = np.exp(x)
  return (e / np.maximum(e.sum(), 1e-12)).astype(np.float32)


def _make_1d_density_probe(
    *,
    state: np.ndarray,
    action: np.ndarray,
    noise_std: float,
    axis: int,
    half_width: float = 2.0,
    n_grid: int = 81,
    name: str,
) -> dict:
  """Fixed (s,a) probe: 1D grid of x' along ``axis``, other coords at μ=s+a."""
  state = np.asarray(state, dtype=np.float32)
  action = np.asarray(action, dtype=np.float32)
  mu = state + action
  grid = np.linspace(
      float(mu[axis] - half_width), float(mu[axis] + half_width), n_grid)
  goals = np.broadcast_to(mu[None, :], (n_grid, 3)).copy().astype(np.float32)
  goals[:, axis] = grid.astype(np.float32)
  gt = np.exp(-0.5 * ((grid - mu[axis]) / noise_std) ** 2)
  gt = (gt / gt.sum()).astype(np.float32)
  return {
      'name': name,
      'state': state,
      'action': action,
      'mu': mu.astype(np.float32),
      'axis': int(axis),
      'grid': grid.astype(np.float32),
      'goals': goals,
      'gt_prob': gt,
      'noise_std': float(noise_std),
  }


def _make_fixed_ood_probes(
    noise_std: float,
    high_xy: float = 15.0,
    axy_low: float = 2.0,
    axy_high: float = 4.0,
):
  """Fixed one-step probes shared across all model jobs.

  1. ``az_ood``: s=0, a_z=0.5 (action OOD) — histogram along z'
  2. ``state_ood``: s=(high,high,0), a_z=0 (state OOD) — histogram along x'
  3. ``axy_ood``: s=0, a_x,a_y ~ U[axy_low,axy_high], a_z=0 — hist along x'
  """
  rng = np.random.default_rng(_PROBE_ACTION_SEED)
  ax, ay = rng.uniform(-1.0, 1.0, size=2)
  az_probe = _make_1d_density_probe(
      state=np.zeros(3, dtype=np.float32),
      action=np.array([ax, ay, 0.5], dtype=np.float32),
      noise_std=noise_std,
      axis=2,
      name='az_ood',
  )
  # Fresh ax,ay for the high-state probe (still deterministic via same RNG).
  ax2, ay2 = rng.uniform(-1.0, 1.0, size=2)
  state_probe = _make_1d_density_probe(
      state=np.array([high_xy, high_xy, 0.0], dtype=np.float32),
      action=np.array([ax2, ay2, 0.0], dtype=np.float32),
      noise_std=noise_std,
      axis=0,
      name='state_ood',
  )
  # Sample AFTER az/state so existing probe actions stay bit-identical.
  ax3, ay3 = rng.uniform(float(axy_low), float(axy_high), size=2)
  axy_probe = _make_1d_density_probe(
      state=np.zeros(3, dtype=np.float32),
      action=np.array([ax3, ay3, 0.0], dtype=np.float32),
      noise_std=noise_std,
      axis=0,
      name='axy_ood',
  )
  return {
      'az_ood': az_probe,
      'state_ood': state_probe,
      'axy_ood': axy_probe,
  }


def _categorical_accuracy_from_batch(
    mode: str,
    batch: Dict[str, np.ndarray],
    *,
    obs_dim: int,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> float:
  states = jnp.asarray(batch['obs'][:, :obs_dim])
  goals = jnp.asarray(batch['obs'][:, obs_dim:])
  actions = jnp.asarray(batch['action'])
  scores = np.asarray(_score_matrix(
      mode,
      networks=networks,
      nf_nets=nf_nets,
      fm_nets=fm_nets,
      td3_nets=td3_nets,
      q_params=q_params,
      states=states,
      actions=actions,
      goals=goals,
      nf_goal_mean=nf_goal_mean,
      nf_goal_std=nf_goal_std,
  ))
  b = scores.shape[0]
  return float(np.mean(np.argmax(scores, axis=1) == np.arange(b)))


def _categorical_accuracy_on_env(
    mode: str,
    *,
    val_env: JaxXYZVecEnv,
    rng: np.random.Generator,
    discount: float,
    batch_size: int,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> float:
  """InfoNCE-style categorical accuracy on fresh geometric (s,a,g) pairs."""
  val_replay = _collect_random_episodes(val_env, rng, discount)
  if val_replay.size < batch_size:
    return float('nan')
  batch = val_replay.sample(batch_size, rng)
  return _categorical_accuracy_from_batch(
      mode, batch,
      obs_dim=val_env.obs_dim,
      networks=networks,
      nf_nets=nf_nets,
      fm_nets=fm_nets,
      td3_nets=td3_nets,
      q_params=q_params,
      nf_goal_mean=nf_goal_mean,
      nf_goal_std=nf_goal_std,
  )


def evaluate_categorical(
    mode: str,
    *,
    train_replay: EpisodeReplay,
    val_env_az0: JaxXYZVecEnv,
    val_env_az_ood: JaxXYZVecEnv,
    val_env_state_ood: JaxXYZVecEnv,
    val_env_axy_ood: JaxXYZVecEnv,
    rng: np.random.Generator,
    discount: float,
    batch_size: int,
    obs_dim: int,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> Dict[str, float]:
  """Train / in-dist / az-OOD / state-OOD / axy-OOD categorical accuracies."""
  out: Dict[str, float] = {}
  if train_replay.size >= batch_size:
    train_batch = train_replay.sample(batch_size, rng)
    out['train/cat_acc'] = _categorical_accuracy_from_batch(
        mode, train_batch,
        obs_dim=obs_dim,
        networks=networks,
        nf_nets=nf_nets,
        fm_nets=fm_nets,
        td3_nets=td3_nets,
        q_params=q_params,
        nf_goal_mean=nf_goal_mean,
        nf_goal_std=nf_goal_std,
    )
  kwargs = dict(
      rng=rng, discount=discount, batch_size=batch_size,
      networks=networks, nf_nets=nf_nets, fm_nets=fm_nets,
      td3_nets=td3_nets, q_params=q_params,
      nf_goal_mean=nf_goal_mean, nf_goal_std=nf_goal_std,
  )
  out['val/cat_acc_az0'] = _categorical_accuracy_on_env(
      mode, val_env=val_env_az0, **kwargs)
  out['val/cat_acc_az_ood'] = _categorical_accuracy_on_env(
      mode, val_env=val_env_az_ood, **kwargs)
  out['val/cat_acc_state_ood'] = _categorical_accuracy_on_env(
      mode, val_env=val_env_state_ood, **kwargs)
  out['val/cat_acc_axy_ood'] = _categorical_accuracy_on_env(
      mode, val_env=val_env_axy_ood, **kwargs)
  return out


def dump_ood_density_probe(
    mode: str,
    *,
    probe: dict,
    out_path: str,
    iteration: int,
    global_step: int,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> None:
  """Score fixed (s,a) on a 1D goal grid; save softmax-normalized p̂(x'|s,a)."""
  scores = _score_candidates(
      mode,
      networks=networks,
      nf_nets=nf_nets,
      fm_nets=fm_nets,
      td3_nets=td3_nets,
      q_params=q_params,
      state=probe['state'],
      action=probe['action'],
      goals=probe['goals'],
      nf_goal_mean=nf_goal_mean,
      nf_goal_std=nf_goal_std,
  )
  prob = _softmax_normalize(scores)
  np.savez_compressed(
      out_path,
      iteration=iteration,
      global_step=global_step,
      mode=np.asarray(mode),
      probe_name=np.asarray(probe['name']),
      state=probe['state'],
      action=probe['action'],
      mu=probe['mu'],
      axis=np.asarray(probe['axis']),
      grid=probe['grid'],
      # Backward-compatible aliases used by older plot code.
      z_grid=probe['grid'],
      scores=scores.astype(np.float32),
      prob=prob,
      gt_prob=probe['gt_prob'],
      gt_prob_z=probe['gt_prob'],
  )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
  os.makedirs(args.log_dir, exist_ok=True)
  seed = int(args.seed)
  E = int(args.num_envs)
  T = int(args.episode_length)
  mode = str(args.repr_mode).strip().lower()
  if mode not in ('crl', 'nf', 'td3', 'fm', 'tdinfonce'):
    raise ValueError(
        f'repr_mode must be crl|nf|td3|fm|tdinfonce, got {mode!r}')

  zero_az = bool(args.zero_az)
  print(f'[xyz] JAX backend={jax.default_backend()} devices={jax.devices()}',
        flush=True)
  print(f'[xyz] mode={mode}  num_envs={E}  episode_length={T}  '
        f'density_steps_per_iter={args.density_steps_per_iter}  '
        f'max_replay_size={args.max_replay_size}  batch_size={args.batch_size}  '
        f'discount={args.discount}  noise_std={args.noise_std}  '
        f'zero_az={zero_az}  val_every={args.val_every}', flush=True)

  rng = np.random.default_rng(seed)
  env = JaxXYZVecEnv(
      num_envs=E,
      episode_length=T,
      noise_std=float(args.noise_std),
      seed=seed,
      zero_az=zero_az,
  )
  high_xy = float(args.ood_high_xy)
  axy_low = float(args.axy_ood_low)
  axy_high = float(args.axy_ood_high)
  # Held-out validation envs:
  #   az0: in-dist (a_z=0, start ~0, a_x,a_y ~ U[-1,1])
  #   az_ood: action OOD (a_z ~ Uniform[-1,1], vs train a_z=0)
  #   state_ood: state OOD (a_z=0, start at (high,high,0))
  #   axy_ood: action OOD (a_x,a_y ~ U[axy_low,axy_high], a_z=0)
  val_env_az0 = JaxXYZVecEnv(
      num_envs=int(args.val_num_envs),
      episode_length=T,
      noise_std=float(args.noise_std),
      seed=seed + 10_000,
      zero_az=True,
  )
  val_env_az_ood = JaxXYZVecEnv(
      num_envs=int(args.val_num_envs),
      episode_length=T,
      noise_std=float(args.noise_std),
      seed=seed + 20_000,
      zero_az=False,
      force_az_positive=False,
  )
  val_env_state_ood = JaxXYZVecEnv(
      num_envs=int(args.val_num_envs),
      episode_length=T,
      noise_std=float(args.noise_std),
      seed=seed + 30_000,
      zero_az=True,
      reset_center=np.array([high_xy, high_xy, 0.0], dtype=np.float32),
  )
  val_env_axy_ood = JaxXYZVecEnv(
      num_envs=int(args.val_num_envs),
      episode_length=T,
      noise_std=float(args.noise_std),
      seed=seed + 40_000,
      zero_az=True,
      action_low=axy_low,
      action_high=axy_high,
  )
  obs_dim = env.obs_dim
  act_dim = env.act_dim
  goal_dim = obs_dim  # full state is the goal

  replay = EpisodeReplay(
      capacity=int(args.max_replay_size),
      obs_dim=obs_dim,
      discount=float(args.discount),
      start_index=0,
      end_index=goal_dim,
  )

  # ---- density estimator setup -----------------------------------------
  key = jax.random.PRNGKey(seed)
  env_spec = _make_env_spec(obs_dim, act_dim, goal_dim)
  hidden = tuple(int(x) for x in args.hidden_layer_sizes.split(',') if x)

  nf_nets = None
  fm_nets = None
  td3_nets = None
  networks = None
  nf_goal_mean = np.zeros(goal_dim, dtype=np.float32)
  nf_goal_std = np.ones(goal_dim, dtype=np.float32)
  policy_params = None
  td3_policy_target = None
  td_infonce_target_q = None
  policy_network = None
  sample_fn = None

  # Fixed OOD probes shared across all model jobs.
  ood_probes = _make_fixed_ood_probes(
      float(args.noise_std),
      high_xy=high_xy,
      axy_low=axy_low,
      axy_high=axy_high,
  )
  probe_dir = os.path.join(args.log_dir, 'ood_density_probe')
  os.makedirs(probe_dir, exist_ok=True)
  with open(os.path.join(args.log_dir, 'probe_config.txt'), 'w') as f:
    for name, pr in ood_probes.items():
      f.write(
          f"[{name}] state={pr['state'].tolist()} "
          f"action={pr['action'].tolist()} mu={pr['mu'].tolist()} "
          f"axis={pr['axis']}\n"
      )
    f.write(f"probe_action_seed={_PROBE_ACTION_SEED}\n")
    f.write(f"ood_high_xy={high_xy}\n")
    f.write(f"axy_ood_action=U[{axy_low},{axy_high}]\n")
  for name, pr in ood_probes.items():
    print(f"[xyz] density probe[{name}] s={pr['state'].tolist()} "
          f"a={pr['action'].tolist()} axis={pr['axis']}", flush=True)

  if mode == 'crl':
    networks = make_networks(
        env_spec,
        obs_dim=obs_dim,
        repr_dim=int(args.repr_dim),
        repr_norm=bool(args.repr_norm),
        hidden_layer_sizes=hidden,
        twin_q=bool(args.twin_q),
    )
    key, k_q = jax.random.split(key)
    q_params = networks.q_network.init(k_q)
    q_optimizer = optax.adam(float(args.learning_rate))
    q_opt_state = q_optimizer.init(q_params)
    density_update = make_crl_update_fn(
        networks, q_optimizer, backward=False, config=None)
    print(f'[xyz] CRL φ·ψ  repr_dim={args.repr_dim}  hidden={hidden}  '
          f'twin_q={args.twin_q}', flush=True)

  elif mode == 'nf':
    nf_nets = _nf.make_nf_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        rep_size=int(args.nf_rep_size),
        num_blocks=int(args.nf_num_blocks),
        channels=int(args.nf_coupling_width),
        goal_enc_size=int(args.nf_goal_enc_size),
        sa_hidden=int(args.nf_sa_hidden),
        sa_num_layers=int(args.nf_sa_num_layers),
    )
    key, k_q = jax.random.split(key)
    q_params = _nf.init_nf_params(nf_nets, k_q)
    q_optimizer = _nf.make_nf_optimizers(
        encoder_lr=float(args.nf_encoder_lr),
        critic_lr=float(args.nf_critic_lr),
        critic_weight_decay=float(args.nf_critic_weight_decay),
        grad_clip=float(args.nf_grad_clip),
        has_goal_encoder=(nf_nets.goal_encoder_net is not None),
    )
    q_opt_state = q_optimizer.init(q_params)
    density_update = _nf.make_nf_density_update_fn(
        nf_nets, q_optimizer, obs_dim=obs_dim,
        noise_std=float(args.nf_noise_std))
    print(f'[xyz] NF RealNVP  rep_size={args.nf_rep_size}  '
          f'blocks={args.nf_num_blocks}  width={args.nf_coupling_width}  '
          f'sa={args.nf_sa_num_layers}x{args.nf_sa_hidden}  '
          f'flow_dim={nf_nets.flow_dim}', flush=True)

  elif mode == 'fm':
    fm_nets = _fm.make_fm_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        hidden_layer_sizes=hidden,
        flow_steps=int(args.fm_flow_steps),
    )
    key, k_q = jax.random.split(key)
    q_params = _fm.init_fm_params(fm_nets, k_q)
    q_optimizer = optax.adam(float(args.learning_rate))
    q_opt_state = q_optimizer.init(q_params)
    density_update = _fm.make_fm_density_update_fn(
        fm_nets, q_optimizer, obs_dim=obs_dim)
    print(f'[xyz] FM OT-CFM  hidden={hidden}  '
          f'flow_steps={args.fm_flow_steps}', flush=True)

  elif mode == 'tdinfonce':
    networks = make_networks(
        env_spec,
        obs_dim=obs_dim,
        repr_dim=int(args.repr_dim),
        repr_norm=bool(args.repr_norm),
        hidden_layer_sizes=hidden,
        twin_q=False,
    )
    key, k_q = jax.random.split(key)
    q_params = networks.q_network.init(k_q)
    td_infonce_target_q = jax.tree_util.tree_map(
        lambda x: jnp.array(x), q_params)
    q_optimizer = optax.adam(float(args.learning_rate))
    q_opt_state = q_optimizer.init(q_params)
    policy_params = networks.policy_network.init(jax.random.PRNGKey(0))
    if zero_az:
      a_low = np.array([-1.0, -1.0, 0.0], dtype=np.float32)
      a_high = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    else:
      a_low, a_high = -1.0, 1.0
    # Uniform-bootstrap variant (matches random data-collection policy).
    _, density_update = make_td_infonce_update_fn(
        networks,
        q_optimizer,
        obs_dim=obs_dim,
        discount=float(args.discount),
        target_tau=float(args.td_infonce_target_tau),
        start_index=0,
        end_index=goal_dim,
        twin_q=False,
        policy_goal=np.zeros(goal_dim, dtype=np.float32),
        action_low=a_low,
        action_high=a_high,
        logsumexp_penalty_coef=float(args.td_infonce_logsumexp_coef),
    )
    print(
        f'[xyz] TDInfoNCE φ·ψ  repr_dim={args.repr_dim}  hidden={hidden}  '
        f'mix_gamma={args.discount}  target_tau={args.td_infonce_target_tau}  '
        f"a'~Uniform (zero_az={zero_az})",
        flush=True)

  else:  # td3
    td3_nets = _td3.make_td3_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        hidden_layer_sizes=hidden,
        repr_dim=int(args.repr_dim),
        bilinear=bool(args.td3_bilinear),
        repr_norm=bool(args.repr_norm) if args.td3_bilinear else False,
    )
    key, k_q = jax.random.split(key)
    q_params = _td3.init_td3_params(td3_nets, k_q)
    q_optimizer = optax.adam(float(args.learning_rate))
    q_opt_state = q_optimizer.init(_td3.online_td3_params(q_params))
    policy_network, sample_fn = _make_uniform_bootstrap_policy(
        act_dim, zero_az=zero_az)
    policy_params = policy_network.init(jax.random.PRNGKey(0))
    td3_policy_target = policy_params
    density_update = _td3.make_td3_density_update_fn(
        td3_nets,
        policy_network=policy_network,
        sample_fn=sample_fn,
        optimizer=q_optimizer,
        obs_dim=obs_dim,
        start_index=0,
        end_index=goal_dim,
        discount=float(args.discount),
        tau=float(args.td3_tau),
        goal_tol=float(args.td3_goal_tol),
        use_target_policy=False,
        cross_batch_goals=bool(args.td3_cross_batch_goals),
    )
    print(f'[xyz] TD3 twin-Q  hidden={hidden}  goal_tol={args.td3_goal_tol}  '
          f'cross_batch_goals={args.td3_cross_batch_goals}  '
          f"a'~Uniform[-1,1]", flush=True)

  # ---- CSV logger (fixed schema so early iters without updates still work)
  csv_path = os.path.join(args.log_dir, f'metrics_{mode}_seed{seed}.csv')
  fieldnames = [
      'iteration', 'global_step', 'replay_size', 'num_episodes',
      'episodes_added', 'iter_time_s', 'elapsed_s',
      'crl_loss', 'categorical_accuracy', 'logits_pos', 'logits_neg',
      'density_loss', 'log_p_mean',
      'td3_qf_loss', 'td3_q1_mean', 'td3_goal_hit_frac',
      'probe/gt_logp_next_mean', 'probe/gt_logp_goal_mean',
      'probe/goal_is_next_frac',
      'probe/nf_logp_goal_mean', 'probe/crl_diag_mean', 'probe/crl_cat_acc',
      'probe/td3_q1_mean',
      # Shared categorical accuracies (train / in-dist val / OOD val).
      'train/cat_acc', 'val/cat_acc_az0',
      'val/cat_acc_az_ood', 'val/cat_acc_state_ood',
      'val/cat_acc_axy_ood',
  ]
  csv_file = open(csv_path, 'w', newline='')  # pylint: disable=consider-using-with
  csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction='ignore')
  csv_writer.writeheader()
  csv_file.flush()

  # ---- rollout state ---------------------------------------------------
  obs = env.reset()
  ep_obs = [[obs[i].copy()] for i in range(E)]
  ep_act: list = [[] for _ in range(E)]

  steps_per_iter = E * T
  num_iterations = int(np.ceil(int(args.num_steps) / max(1, steps_per_iter)))
  print(f'[xyz] steps_per_iter={steps_per_iter}  num_iterations={num_iterations}  '
        f'log_csv={csv_path}', flush=True)

  global_step = 0
  t0 = time.time()

  for iteration in range(num_iterations):
    iter_t0 = time.time()
    episodes_added = 0
    if iteration == 0:
      print('[xyz] starting iter 0 rollout…', flush=True)

    # One full episode per env (= T steps), matching PPO rollout_length=50.
    for _t in range(T):
      action = env.sample_uniform_actions(rng)
      next_obs, _rew, dones, terminal_obs, _info = env.step(action)
      for i in range(E):
        ep_act[i].append(action[i].copy())
        if dones[i]:
          ep_obs[i].append(terminal_obs[i].copy())
          try:
            replay.add_episode(
                np.stack(ep_obs[i], axis=0),
                np.stack(ep_act[i], axis=0),
                successful=False)
            episodes_added += 1
          except AssertionError:
            pass
          ep_obs[i] = [next_obs[i].copy()]
          ep_act[i] = []
        else:
          ep_obs[i].append(next_obs[i].copy())
      obs = next_obs
      global_step += E

    # ---- density updates ----------------------------------------------
    metrics_agg: Dict[str, list] = {}
    probe_log: Dict[str, float] = {}
    if replay.size >= int(args.min_replay_size):
      if iteration == 0:
        print(f'[xyz] iter 0: replay={replay.size} → density updates…',
              flush=True)
      if mode == 'nf':
        stat_batch = replay.sample(min(2048, replay.size), rng)
        goals = stat_batch['obs'][:, obs_dim:]
        nf_goal_mean = goals.mean(axis=0).astype(np.float32)
        nf_goal_std = goals.std(axis=0).astype(np.float32)
        nf_goal_std = np.maximum(
            nf_goal_std, float(args.nf_goal_std_min)).astype(np.float32)

      n_upd = int(args.density_steps_per_iter)
      for _ in range(n_upd):
        batch_np = replay.sample(int(args.batch_size), rng)
        if mode == 'tdinfonce':
          # Same random_goal construction as PPO TD-InfoNCE (roll future goals).
          fut_g = batch_np['obs'][:, obs_dim:]
          batch_np = dict(batch_np)
          batch_np['random_goal'] = np.roll(fut_g, -1, axis=0)
        batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
        key, k_upd = jax.random.split(key)

        if mode == 'crl':
          q_params, q_opt_state, m = density_update(
              q_params, q_opt_state, batch, k_upd)
        elif mode == 'nf':
          q_params, q_opt_state, m = density_update(
              q_params, q_opt_state, batch, k_upd,
              jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std))
        elif mode == 'fm':
          q_params, q_opt_state, m = density_update(
              q_params, q_opt_state, batch, k_upd)
        elif mode == 'tdinfonce':
          q_params, q_opt_state, td_infonce_target_q, m = density_update(
              q_params, q_opt_state, td_infonce_target_q, policy_params,
              batch, k_upd)
        else:
          q_params, q_opt_state, m, td3_policy_target = density_update(
              q_params, q_opt_state, batch, k_upd,
              policy_params, td3_policy_target)

        m_np = _tree_to_numpy(m)
        for k, v in m_np.items():
          arr = np.asarray(v)
          # Skip vector metrics (e.g. TD-InfoNCE a_prime_hist).
          if arr.ndim == 0 or arr.size == 1:
            metrics_agg.setdefault(k, []).append(float(arr))

      if iteration == 0:
        print('[xyz] iter 0: density updates done → diagnostics…', flush=True)

      # Analytic one-step Gaussian probe (diagnostic vs geometric goals).
      probe = replay.sample(int(args.batch_size), rng)
      s = probe['obs'][:, :obs_dim]
      g = probe['obs'][:, obs_dim:]
      a = probe['action']
      s_next = probe['next_obs'][:, :obs_dim]
      active = slice(0, 2) if zero_az else None
      gt_next = _analytic_gaussian_log_prob(
          s, a, s_next, float(args.noise_std), active_dims=active)
      gt_goal = _analytic_gaussian_log_prob(
          s, a, g, float(args.noise_std), active_dims=active)
      probe_log['probe/gt_logp_next_mean'] = float(gt_next.mean())
      probe_log['probe/gt_logp_goal_mean'] = float(gt_goal.mean())
      probe_log['probe/goal_is_next_frac'] = float(
          np.mean(np.linalg.norm(g - s_next, axis=-1) < 1e-6))

      if mode == 'nf':
        log_p = np.asarray(_nf.nf_log_prob(
            nf_nets, q_params,
            jnp.asarray(s), jnp.asarray(a),
            jnp.asarray((g - nf_goal_mean) / (nf_goal_std + 1e-8))))
        probe_log['probe/nf_logp_goal_mean'] = float(log_p.mean())
      elif mode == 'fm':
        # Exact FM log-prob is expensive; only log it on probe iters.
        if (int(args.probe_every) > 0
            and (iteration % int(args.probe_every) == 0
                 or iteration == num_iterations - 1)):
          log_p = np.asarray(_fm.fm_log_prob(
              fm_nets, q_params, jnp.asarray(s), jnp.asarray(a),
              jnp.asarray(g), mode='exact'))
          probe_log['probe/nf_logp_goal_mean'] = float(log_p.mean())
      elif mode in ('crl', 'tdinfonce'):
        packed = jnp.asarray(probe['obs'])
        logits, sa_repr, g_repr = networks.q_network.apply(
            q_params, packed, jnp.asarray(a))
        if logits.ndim == 3:
          logits = logits.mean(axis=-1)
        diag = np.asarray(jnp.einsum('bd,bd->b', sa_repr, g_repr))
        probe_log['probe/crl_diag_mean'] = float(diag.mean())
        probe_log['probe/crl_cat_acc'] = float(
            np.mean(np.argmax(np.asarray(logits), axis=1)
                    == np.arange(logits.shape[0])))
      elif mode == 'td3':
        q1 = np.asarray(td3_nets.qf1_net.apply(
            q_params['qf1'], jnp.asarray(probe['obs']), jnp.asarray(a)))
        probe_log['probe/td3_q1_mean'] = float(q1.mean())

    # Train / in-dist val / OOD val categorical accuracy.
    val_log: Dict[str, float] = {}
    do_val = (
        replay.size >= int(args.min_replay_size)
        and int(args.val_every) > 0
        and (iteration % int(args.val_every) == 0
             or iteration == num_iterations - 1)
    )
    if do_val:
      val_log = evaluate_categorical(
          mode,
          train_replay=replay,
          val_env_az0=val_env_az0,
          val_env_az_ood=val_env_az_ood,
          val_env_state_ood=val_env_state_ood,
          val_env_axy_ood=val_env_axy_ood,
          rng=rng,
          discount=float(args.discount),
          batch_size=int(args.val_batch_size),
          obs_dim=obs_dim,
          networks=networks,
          nf_nets=nf_nets,
          fm_nets=fm_nets,
          td3_nets=td3_nets,
          q_params=q_params,
          nf_goal_mean=nf_goal_mean,
          nf_goal_std=nf_goal_std,
      )

    # Fixed-(s,a) OOD density histogram probes (az / state / axy).
    do_probe = (
        replay.size >= int(args.min_replay_size)
        and int(args.probe_every) > 0
        and (iteration % int(args.probe_every) == 0
             or iteration == num_iterations - 1)
    )
    if do_probe:
      for pname, pr in ood_probes.items():
        probe_path = os.path.join(
            probe_dir, f'probe_{pname}_{mode}_iter{iteration:06d}.npz')
        dump_ood_density_probe(
            mode,
            probe=pr,
            out_path=probe_path,
            iteration=iteration,
            global_step=global_step,
            networks=networks,
            nf_nets=nf_nets,
            fm_nets=fm_nets,
            td3_nets=td3_nets,
            q_params=q_params,
            nf_goal_mean=nf_goal_mean,
            nf_goal_std=nf_goal_std,
        )
        print(f'[xyz][probe] wrote {probe_path}', flush=True)

    metrics = _mean_metrics(metrics_agg)
    row = {
        'iteration': iteration,
        'global_step': global_step,
        'replay_size': replay.size,
        'num_episodes': replay.num_episodes,
        'episodes_added': episodes_added,
        'iter_time_s': time.time() - iter_t0,
        'elapsed_s': time.time() - t0,
        **metrics,
        **probe_log,
        **val_log,
    }
    csv_writer.writerow(row)
    csv_file.flush()

    if iteration % int(args.log_interval) == 0 or iteration == num_iterations - 1:
      metric_str = ' '.join(f'{k}={v:.4f}' for k, v in metrics.items())
      probe_str = ' '.join(f'{k}={v:.4f}' for k, v in probe_log.items())
      val_str = ' '.join(f'{k}={v:.4f}' for k, v in val_log.items())
      print(
          f'[xyz] iter={iteration}/{num_iterations} step={global_step} '
          f'replay={replay.size} eps={replay.num_episodes} '
          f't={row["iter_time_s"]:.2f}s  {metric_str}  {probe_str}',
          flush=True)
      if val_str:
        print(f'[xyz][val] iter={iteration}  {val_str}', flush=True)

    if (args.checkpoint_interval > 0
        and iteration > 0
        and iteration % int(args.checkpoint_interval) == 0):
      import pickle
      ckpt = {
          'iteration': iteration,
          'global_step': global_step,
          'q_params': q_params,
          'q_opt_state': q_opt_state,
          'mode': mode,
          'args': vars(args),
      }
      if mode == 'tdinfonce' and td_infonce_target_q is not None:
        ckpt['td_infonce_target_q'] = td_infonce_target_q
      ckpt_path = os.path.join(
          args.log_dir, f'ckpt_{mode}_seed{seed}_iter{iteration}.pkl')
      with open(ckpt_path, 'wb') as f:
        pickle.dump(ckpt, f)
      print(f'[xyz] wrote {ckpt_path}', flush=True)

  csv_file.close()
  print(f'[xyz] done. metrics -> {csv_path}', flush=True)


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
      description='xyz density probe (CRL / NF / TD3 / FM / TDInfoNCE)')
  p.add_argument('--repr_mode', type=str, default='crl',
                 choices=['crl', 'nf', 'td3', 'fm', 'tdinfonce'])
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--num_envs', type=int, default=1024)
  p.add_argument('--episode_length', type=int, default=50)
  p.add_argument('--num_steps', type=int, default=100_000_000,
                 help='Total env steps (default: 100M).')
  p.add_argument('--max_replay_size', type=int, default=1_000_000)
  p.add_argument('--min_replay_size', type=int, default=10_000)
  p.add_argument('--batch_size', type=int, default=256)
  p.add_argument('--density_steps_per_iter', type=int, default=25,
                 help='Density SGD updates per iteration (PPO: ppo_crl_steps_per_iter).')
  p.add_argument('--discount', type=float, default=0.5)
  p.add_argument('--noise_std', type=float, default=0.01)
  p.add_argument('--zero_az', action='store_true', default=True,
                 help='Force a_z=0 and freeze z (xy motion only).')
  p.add_argument('--no_zero_az', dest='zero_az', action='store_false')
  p.add_argument('--axy_ood_low', type=float, default=2.0,
                 help='axy_ood val: lower bound for a_x,a_y (train uses -1).')
  p.add_argument('--axy_ood_high', type=float, default=4.0,
                 help='axy_ood val: upper bound for a_x,a_y (train uses +1).')
  p.add_argument('--ood_high_xy', type=float, default=15.0,
                 help='Start (x,y) for state-OOD val / density probe.')
  p.add_argument('--val_every', type=int, default=5,
                 help='Run train/ID/OOD categorical eval every N iters.')
  p.add_argument('--probe_every', type=int, default=10,
                 help='Dump fixed-(s,a) OOD density histograms every N iters.')
  p.add_argument('--val_num_envs', type=int, default=128,
                 help='Parallel envs used to generate validation episodes.')
  p.add_argument('--val_batch_size', type=int, default=256)
  p.add_argument('--fm_flow_steps', type=int, default=10)
  p.add_argument('--td_infonce_target_tau', type=float, default=0.995,
                 help='EMA keep-rate for TD-InfoNCE target critic.')
  p.add_argument('--td_infonce_logsumexp_coef', type=float, default=0.01,
                 help='Logsumexp penalty coef for TD-InfoNCE CE loss.')
  p.add_argument('--learning_rate', type=float, default=3e-4)
  p.add_argument('--hidden_layer_sizes', type=str,
                 default='256,256,256,256,256,256')
  p.add_argument('--repr_dim', type=int, default=64)
  p.add_argument('--repr_norm', action='store_true')
  p.add_argument('--twin_q', action='store_true')
  # NF
  p.add_argument('--nf_rep_size', type=int, default=256)
  p.add_argument('--nf_num_blocks', type=int, default=12)
  p.add_argument('--nf_coupling_width', type=int, default=512)
  p.add_argument('--nf_sa_hidden', type=int, default=1024)
  p.add_argument('--nf_sa_num_layers', type=int, default=4)
  p.add_argument('--nf_goal_enc_size', type=int, default=0)
  p.add_argument('--nf_encoder_lr', type=float, default=3e-4)
  p.add_argument('--nf_critic_lr', type=float, default=1e-4)
  p.add_argument('--nf_critic_weight_decay', type=float, default=1e-6)
  p.add_argument('--nf_grad_clip', type=float, default=1.0)
  p.add_argument('--nf_noise_std', type=float, default=0.05)
  p.add_argument('--nf_goal_std_min', type=float, default=0.1)
  # TD3
  p.add_argument('--td3_tau', type=float, default=0.005)
  p.add_argument('--td3_goal_tol', type=float, default=0.05)
  p.add_argument('--td3_cross_batch_goals', action='store_true', default=True)
  p.add_argument('--no_td3_cross_batch_goals',
                 dest='td3_cross_batch_goals', action='store_false')
  p.add_argument('--td3_bilinear', action='store_true')
  # Logging
  p.add_argument('--log_dir', type=str, default='logs/xyz_density/')
  p.add_argument('--log_interval', type=int, default=1)
  p.add_argument('--checkpoint_interval', type=int, default=0)
  return p


def main():
  args = build_parser().parse_args()
  train(args)


if __name__ == '__main__':
  main()
