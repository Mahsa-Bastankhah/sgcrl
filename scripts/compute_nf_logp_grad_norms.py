"""Compute ||∇_s log p_NF|| / ||∇_a log p_NF|| for listed checkpoints (no video).

Matches ``render_nf_traj_reward_video.py`` rollout + reward settings so the
grad series lines up with an existing reward-probe CSV/video.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BB = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import sgcrl_jax_acme_compat  # noqa: F401

import importlib.util as _ilu

import jax
import jax.numpy as jnp
import numpy as np

from contrastive import nf_density as _nf
from contrastive import ppo_learner
from envs.builderbench_utils import (
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

_nfvid_path = os.path.join(_REPO, 'scripts', 'render_nf_traj_reward_video.py')
_nfvid_spec = _ilu.spec_from_file_location('render_nf_traj_reward_video', _nfvid_path)
nfvid = _ilu.module_from_spec(_nfvid_spec)
assert _nfvid_spec.loader is not None
_nfvid_spec.loader.exec_module(nfvid)


def _parse():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint_dir', required=True)
  p.add_argument('--iters', required=True, help='comma-separated iteration ints')
  p.add_argument('--env', default='builderbench_creative_5_task2')
  p.add_argument('--out_json', required=True)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--max_tries', type=int, default=20)
  p.add_argument('--fixed_start_x', type=float, default=nfvid.VIDEO_FIXED_START_X)
  return p.parse_args()


def main():
  args = _parse()
  iters = [int(x) for x in args.iters.split(',') if x.strip()]
  env_name = args.env
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, _ = parse_bb_env_id(env_id)
  first = os.path.join(args.checkpoint_dir, f'ckpt_iter_{iters[0]:07d}.pkl')
  ctx = nfvid._load_train_ctx(env_name, first)
  nfvid.force_video_nopermute_norand(ctx, fixed_start_x=float(args.fixed_start_x))
  run_dir = nfvid._run_dir_from_ckpt(first)
  arch = nfvid._load_nf_arch(run_dir)
  networks, nf_nets, goal_dim = nfvid._build_networks(
      env_name, args.seed, ctx, arch)
  env, _base, mocap_targets, ep_len = nfvid._make_bb_env(env_id, ctx)
  reward_fn = _nf.make_nf_reward_fn(nf_nets, obs_dim=int(ctx.obs_dim))
  grad_fn = nfvid.make_nf_logp_grad_norm_fn(nf_nets, obs_dim=int(ctx.obs_dim))
  policy_fn_maker = nfvid._make_policy_fn
  run_maker = nfvid._compile_rollout_and_states

  out = {}
  print(f'[grad] jax={jax.default_backend()} devices={jax.devices()} '
        f'state_only={arch["nf_state_only"]}', flush=True)

  for it in iters:
    ckpt_path = os.path.join(args.checkpoint_dir, f'ckpt_iter_{it:07d}.pkl')
    print(f'[grad] === iter {it} {ckpt_path} ===', flush=True)
    ckpt = ppo_learner.load_checkpoint(ckpt_path)
    nf_params = ckpt['q_params']
    iteration = int(ckpt.get('iteration') or 0)
    goal_mean, goal_std = nfvid._load_goal_stats(run_dir, iteration, goal_dim)
    goal_std = np.maximum(goal_std, arch['nf_goal_std_min']).astype(np.float32)
    gmean = jnp.asarray(goal_mean)
    gstd = jnp.asarray(goal_std)

    policy = policy_fn_maker(
        networks, ckpt['policy_params'], stochastic=False,
        filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes,
        normalize_obs=False, obs_dim=ctx.obs_dim,
        start_index=ctx.start_index, end_index=ctx.end_index)
    run = run_maker(
        policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
        ctx.filter_policy_obs)
    key = jax.random.PRNGKey(args.seed)
    key, warm_key = jax.random.split(key)
    print('[grad] warming compile...', flush=True)
    _ = jax.block_until_ready(run(warm_key))
    print('[grad] compile done', flush=True)

    best = None
    for attempt in range(int(args.max_tries)):
      key, roll_key = jax.random.split(key)
      traj, states = run(roll_key)
      packed = np.asarray(traj['packed'], dtype=np.float32)
      actions = np.asarray(traj['action'], dtype=np.float32)
      succ = np.asarray(traj['success'], dtype=np.float32)
      rewards = np.asarray(
          reward_fn(nf_params, jnp.asarray(packed), jnp.asarray(actions),
                    gmean, gstd), dtype=np.float32)
      reached = bool(np.any(succ >= 0.5))
      print(f'[grad] try={attempt} success={reached} logp_sum={rewards.sum():.2f}',
            flush=True)
      cur = dict(rewards=rewards, success=succ, packed=packed, actions=actions,
                 attempt=attempt)
      if reached:
        best = cur
        break
      if best is None or rewards.sum() > best['rewards'].sum():
        best = cur

    rewards = best['rewards']
    success = best['success']
    packed = jnp.asarray(best['packed'])
    actions = jnp.asarray(best['actions'])
    gs, ga = grad_fn(nf_params, packed, actions, gmean, gstd)
    gs = np.asarray(gs, dtype=np.float32)
    ga = np.asarray(ga, dtype=np.float32)
    first_succ = (int(np.argmax(success >= 0.5))
                  if np.any(success >= 0.5) else None)
    rec = {
        'iter': it,
        'attempt': int(best['attempt']),
        'first_succ_t': first_succ,
        'T': int(len(gs)),
        'state_only': bool(arch['nf_state_only']),
        'logp_sum': float(rewards.sum()),
        'gs_min': float(gs.min()),
        'gs_max': float(gs.max()),
        'gs_mean': float(gs.mean()),
        'ga_min': float(ga.min()),
        'ga_max': float(ga.max()),
        'ga_mean': float(ga.mean()),
        'gs': [round(float(x), 4) for x in gs],
        'ga': [round(float(x), 4) for x in ga],
        'r': [round(float(x), 4) for x in rewards],
        'success': [int(x >= 0.5) for x in success],
    }
    if first_succ is not None:
      rec['gs_pre_mean'] = float(gs[:first_succ].mean()) if first_succ else float(gs[0])
      rec['gs_post_mean'] = float(gs[first_succ:].mean())
      rec['gs_before_succ'] = float(gs[first_succ - 1]) if first_succ else float(gs[0])
      rec['gs_at_succ'] = float(gs[first_succ])
    else:
      rec['gs_pre_mean'] = float(gs.mean())
    out[f'{it:07d}'] = rec
    print(f'[grad] iter={it} ||∇s|| range=[{gs.min():.4g},{gs.max():.4g}] '
          f'mean={gs.mean():.4g} ||∇a|| range=[{ga.min():.4g},{ga.max():.4g}]',
          flush=True)

  os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
  with open(args.out_json, 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=2)
  print(f'[grad] wrote {args.out_json}', flush=True)


if __name__ == '__main__':
  main()
