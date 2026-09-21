#!/usr/bin/env python3
"""Roll out Sawyer-bin checkpoints at an override episode horizon.

Used to test whether a T=50-trained policy can succeed if given T=150.

  python -u scripts/eval_sawyer_bin_horizon_success.py \\
      --run_dir=logs/.../ppo_sawyer_bin_0 \\
      --iters=0,4000,8000,12000,16000,20000,latest \\
      --horizon=150 --episodes=20 --stochastic
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import sgcrl_jax_acme_compat  # noqa: F401

import importlib.util as _ilu
import jax
import numpy as np

from contrastive import ppo_learner
import env_utils
from ppo_contrastive import fixed_goal_dict
from eval_sawyer_nf_binary_accuracy import (  # noqa: E402
    CKPT_RE, _env_kwargs, _latest_ckpt)

_vid_path = os.path.join(REPO, 'scripts', 'render_sawyer_nf_traj_reward_video.py')
_spec = _ilu.spec_from_file_location('sawyer_nf_vid', _vid_path)
_vid = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_vid)


def _ckpt_path(run_dir: str, token: str) -> tuple[int, str]:
  if token in ('latest', '-1'):
    return _latest_ckpt(run_dir)
  iteration = int(token)
  path = os.path.join(
      run_dir, 'checkpoints', f'ckpt_iter_{iteration:07d}.pkl')
  if not os.path.isfile(path):
    raise FileNotFoundError(path)
  return iteration, path


def _eval_ckpt(gym_env, act_fn, policy_params, horizon: int, n_ep: int,
               seed: int) -> dict:
  succ = 0
  lengths = []
  first_hits = []
  rng = jax.random.PRNGKey(int(seed))
  for ep in range(n_ep):
    np.random.seed(int(seed) + 17 * (ep + 1))
    obs = np.asarray(gym_env.reset(), dtype=np.float32)
    hit = False
    hit_t = -1
    t_end = 0
    for t in range(horizon):
      rng, k = jax.random.split(rng)
      action = np.asarray(act_fn(policy_params, obs[None], k)[0], dtype=np.float32)
      obs, r, done, _ = gym_env.step(action)
      obs = np.asarray(obs, dtype=np.float32)
      t_end = t + 1
      if (not hit) and float(r) > 0.0:
        hit = True
        hit_t = t_end
      if done:
        break
    succ += int(hit)
    lengths.append(t_end)
    first_hits.append(hit_t)
  n = float(n_ep)
  rate = succ / n
  return {
      'episodes': n_ep,
      'successes': int(succ),
      'success_rate': rate,
      'success_se': float(np.sqrt(rate * (1.0 - rate) / n)),
      'episode_length_mean': float(np.mean(lengths)),
      'first_success_step_mean': (
          float(np.mean([x for x in first_hits if x > 0]))
          if succ else None),
  }


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True)
  p.add_argument('--iters', default='latest')
  p.add_argument('--horizon', type=int, default=150)
  p.add_argument('--episodes', type=int, default=20)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--stochastic', action='store_true')
  args = p.parse_args()
  run_dir = os.path.abspath(args.run_dir)
  tokens = [t.strip() for t in str(args.iters).split(',') if t.strip()]

  _, probe_ckpt = _ckpt_path(run_dir, tokens[0])
  settings = _vid._load_run_settings(probe_ckpt, 'sawyer_bin')
  arch = _vid._load_nf_arch(run_dir)
  networks, _, _, _ = _vid._build_networks(
      'sawyer_bin', seed=args.seed, settings=settings, arch=arch)
  env_kw = _env_kwargs(run_dir)
  env_kw['max_episode_steps'] = int(args.horizon)
  gym_env, _, max_steps = env_utils.load(
      'sawyer_bin', fixed_start_end=fixed_goal_dict['sawyer_bin'],
      seed=args.seed, **env_kw)
  horizon = int(args.horizon) if args.horizon > 0 else int(max_steps)

  @jax.jit
  def _act(params, obs, key):
    dist = networks.policy_network.apply(params, obs)
    if args.stochastic:
      return networks.sample(dist, key)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  print(
      f'EVAL run={os.path.basename(run_dir)} horizon={horizon} '
      f'episodes={args.episodes} stochastic={bool(args.stochastic)} '
      f'iters={tokens}',
      flush=True)
  for token in tokens:
    iteration, ckpt_path = _ckpt_path(run_dir, token)
    ckpt = ppo_learner.load_checkpoint(ckpt_path)
    stats = _eval_ckpt(
        gym_env, _act, ckpt['policy_params'], horizon,
        int(args.episodes), int(args.seed) + iteration)
    row = {
        'run_dir': run_dir,
        'iteration': iteration,
        'horizon': horizon,
        'stochastic': bool(args.stochastic),
        **stats,
    }
    print(
        f'HORIZON_SUCC iter={iteration} rate={stats["success_rate"]:.3f} '
        f'se={stats["success_se"]:.3f} succ={stats["successes"]}/'
        f'{stats["episodes"]} eplen={stats["episode_length_mean"]:.1f} '
        f'first_hit={stats["first_success_step_mean"]}',
        flush=True)
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
  main()
