"""Probe categorical select probs from PPO-catselect checkpoints.

Loads policy checkpoints, rolls out a short batch of env states, and prints
the 7D (or N-cube) categorical probability vector + entropy stats.

Example:
  python scripts/probe_catselect_probs.py \\
    --run_dir=logs/.../ppo_builderbench_creative_7_task2_0 \\
    --env=builderbench_creative_7_task2 \\
    --iters=0,750,1500,1650,1800
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import argparse
import math
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from contrastive import ppo_learner
from envs.builderbench_utils import (
    filter_pd_policy_state_obs,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)
from scripts.ppo_builderbench_rollout_video import (
    _build_networks,
    _load_train_ctx,
    _make_bb_env,
    force_video_nopermute_norand,
)


def _parse_iters(s: str) -> List[Optional[int]]:
  out: List[Optional[int]] = []
  for part in s.split(','):
    part = part.strip()
    if not part:
      continue
    if part.lower() == 'latest':
      out.append(None)
    else:
      out.append(int(part))
  return out


def _ckpt_path(run_dir: str, it: Optional[int]) -> str:
  ckpt_dir = os.path.join(run_dir, 'checkpoints')
  if it is None:
    return os.path.join(ckpt_dir, 'latest.pkl')
  return os.path.join(ckpt_dir, f'ckpt_iter_{it:07d}.pkl')


def _softmax(logits: np.ndarray) -> np.ndarray:
  x = logits - logits.max(axis=-1, keepdims=True)
  e = np.exp(x)
  return e / e.sum(axis=-1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
  p = np.clip(probs, 1e-12, 1.0)
  return -(p * np.log(p)).sum(axis=-1)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_dir', required=True)
  parser.add_argument('--env', default='builderbench_creative_7_task2')
  parser.add_argument('--iters', default='0,750,1500,1650,1800,latest')
  parser.add_argument('--num_envs', type=int, default=64)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--steps', type=int, default=8,
                      help='Macro steps of stochastic rollout to pool states.')
  parser.add_argument('--match_run_init', action='store_true')
  args = parser.parse_args()

  env_id = sgcrl_env_name_to_bb_env_id(args.env)
  num_cubes, _ = parse_bb_env_id(env_id)
  uniform_h = math.log(num_cubes)

  # Use a checkpoint path under run_dir so run_config resolves.
  probe_ckpt = _ckpt_path(args.run_dir, 0)
  if not os.path.isfile(probe_ckpt):
    probe_ckpt = _ckpt_path(args.run_dir, None)
  ctx = _load_train_ctx(args.env, probe_ckpt)
  if not args.match_run_init:
    force_video_nopermute_norand(ctx, fixed_start_x=0.1)

  print(f'[probe] env={args.env} cubes={num_cubes} '
        f'cat_classes={ctx.categorical_select_classes} '
        f'obs_dim={ctx.obs_dim} actor_min_std={ctx.actor_min_std}',
        flush=True)
  if ctx.categorical_select_classes is None:
    raise SystemExit('run is not categorical-select; nothing to probe')

  networks = _build_networks(args.env, seed=args.seed, ctx=ctx)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)
  print(f'[probe] ep_len={ep_len} num_envs={args.num_envs} '
        f'uniform_entropy={uniform_h:.4f} nats '
        f'({uniform_h / math.log(2):.3f} bits)', flush=True)

  goal = np.asarray(ctx.fixed_target_goal, dtype=np.float32)
  if goal is None:
    raise SystemExit('fixed_target_goal missing from run_config')
  goals = np.broadcast_to(goal[None, :], (args.num_envs, goal.shape[0])).copy()

  @jax.jit
  def select_stats(policy_params, packed):
    dist = networks.policy_network.apply(policy_params, packed)
    # HybridSelectDistribution exposes categorical_dist with logits.
    logits = dist.categorical_dist.logits
    probs = jax.nn.softmax(logits, axis=-1)
    ent = dist.categorical_dist.entropy()
    mode = jnp.argmax(probs, axis=-1)
    maxp = jnp.max(probs, axis=-1)
    return probs, ent, mode, maxp, logits

  for it in _parse_iters(args.iters):
    path = _ckpt_path(args.run_dir, it)
    if not os.path.isfile(path):
      print(f'\n=== iter={it} MISSING {path} ===')
      continue
    ckpt = ppo_learner.load_checkpoint(path)
    policy_params = ckpt['policy_params']
    it_ck = ckpt.get('iteration', it)
    print(f'\n=== ckpt iter={it_ck}  global_step={ckpt.get("global_step")} '
          f'({os.path.basename(path)}) ===', flush=True)

    obs_mean = obs_var = None
    mean_j = jnp.zeros((ctx.obs_dim,), dtype=jnp.float32)
    var_j = jnp.ones((ctx.obs_dim,), dtype=jnp.float32)
    if ctx.ppo_norm_obs:
      extra = ckpt.get('extra_state') or {}
      obs_state = extra.get('obs_rms')
      if not isinstance(obs_state, dict):
        raise RuntimeError(f'missing obs_rms in {path}')
      mean_j = jnp.asarray(obs_state['mean'], dtype=jnp.float32)
      var_j = jnp.asarray(obs_state['var'], dtype=jnp.float32)

    rng = jax.random.PRNGKey(args.seed + int(it_ck or 0))
    key, rng = jax.random.split(rng)
    state = env.reset(jax.random.split(key, args.num_envs))

    all_probs = []
    all_ent = []
    all_maxp = []
    all_mode = []

    for t in range(int(args.steps)):
      obs = state.obs
      if ctx.filter_policy_obs:
        obs = filter_pd_policy_state_obs(obs, num_cubes)
      packed = jnp.concatenate(
          [jnp.asarray(obs, dtype=jnp.float32),
           jnp.asarray(goals, dtype=jnp.float32)], axis=-1)
      packed = ppo_learner._normalize_packed_obs(
          packed, mean_j, var_j,
          obs_dim=ctx.obs_dim,
          start_index=ctx.start_index,
          end_index=ctx.end_index,
          clip=ctx.ppo_obs_norm_clip,
          enabled=bool(ctx.ppo_norm_obs),
      )
      probs, ent, mode, maxp, _logits = select_stats(policy_params, packed)
      probs_np = np.asarray(probs)
      all_probs.append(probs_np)
      all_ent.append(np.asarray(ent))
      all_maxp.append(np.asarray(maxp))
      all_mode.append(np.asarray(mode))

      # Advance with stochastic hybrid sample so we see varied states.
      key, rng = jax.random.split(rng)
      dist = networks.policy_network.apply(policy_params, packed)
      act = networks.sample(dist, key)
      state = env.step(state, act)

    probs_all = np.concatenate(all_probs, axis=0)  # [T*E, C]
    ent_all = np.concatenate(all_ent, axis=0)
    maxp_all = np.concatenate(all_maxp, axis=0)
    mode_all = np.concatenate(all_mode, axis=0)

    mean_probs = probs_all.mean(axis=0)
    # Also show one representative vector (env 0, step 0) and a high-conf one.
    first = all_probs[0][0]
    # Most peaked state in the pool
    peak_i = int(maxp_all.argmax())
    peak = probs_all[peak_i]
    # Most uniform state
    flat_i = int(ent_all.argmax())
    flat = probs_all[flat_i]

    print(f'  mean_probs (over {probs_all.shape[0]} states): '
          f'{np.array2string(mean_probs, precision=3, floatmode="fixed")}')
    print(f'  example env0/t0:                 '
          f'{np.array2string(first, precision=3, floatmode="fixed")}')
    print(f'  most peaked (maxp={maxp_all[peak_i]:.3f}): '
          f'{np.array2string(peak, precision=3, floatmode="fixed")}')
    print(f'  most flat   (H={ent_all[flat_i]:.3f}): '
          f'{np.array2string(flat, precision=3, floatmode="fixed")}')
    print(f'  entropy: mean={ent_all.mean():.4f}  '
          f'med={np.median(ent_all):.4f}  '
          f'min={ent_all.min():.4f}  max={ent_all.max():.4f}  '
          f'(uniform={uniform_h:.4f})')
    print(f'  max_prob: mean={maxp_all.mean():.4f}  '
          f'med={np.median(maxp_all):.4f}  '
          f'min={maxp_all.min():.4f}  max={maxp_all.max():.4f}')
    # Mode histogram
    hist = np.bincount(mode_all.astype(np.int64), minlength=num_cubes)
    hist = hist / hist.sum()
    print(f'  mode_frac: '
          f'{np.array2string(hist, precision=3, floatmode="fixed")}')
    # Effective support: how many cubes with p>0.05 on average
    support = (probs_all > 0.05).sum(axis=-1)
    print(f'  support(p>0.05): mean={support.mean():.2f}  '
          f'med={np.median(support):.1f}')


if __name__ == '__main__':
  main()
