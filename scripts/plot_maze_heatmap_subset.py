#!/usr/bin/env python3
"""Plot heatmaps for N evenly spaced checkpoints of a PPO maze run.

Examples::

  python scripts/plot_maze_heatmap_subset.py --seed 123 --num_checkpoints 10
  python scripts/plot_maze_heatmap_subset.py --seed 124 --env point_FourRooms \\
      --log_root ppo_fourrooms --num_checkpoints 10
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

HEATMAP_SUBCELLS = 25
NUM_TRAJ = 5
FIG_SCALE = 2.0


def _list_iters(ckpt_dir: str) -> list[int]:
  iters = []
  for p in glob.glob(os.path.join(ckpt_dir, 'ckpt_iter_*.pkl')):
    m = re.search(r'ckpt_iter_(\d+)\.pkl$', p)
    if m:
      iters.append(int(m.group(1)))
  return sorted(iters)


def _pick_evenly(iters: list[int], n: int) -> list[int]:
  if not iters:
    return []
  if len(iters) <= n:
    return iters
  out = []
  for i in range(n):
    idx = round(i * (len(iters) - 1) / (n - 1))
    out.append(iters[idx])
  # dedupe while preserving order
  seen = set()
  unique = []
  for it in out:
    if it not in seen:
      seen.add(it)
      unique.append(it)
  return unique


def _plot_one(repo: str, pkl_path: str, out_dir: str, env_name: str, seed: int) -> int:
  os.makedirs(out_dir, exist_ok=True)
  out_arg = out_dir if out_dir.endswith(os.sep) else out_dir + os.sep
  cmd = [
      sys.executable, '-u', os.path.join(repo, 'ppo_rollout_maze.py'),
      '--checkpoint', pkl_path,
      '--env', env_name,
      '--output', out_arg,
      '--seed', str(seed),
      '--num_trajectories', str(NUM_TRAJ),
      '--heatmap_subcells', str(HEATMAP_SUBCELLS),
      '--fig_scale', str(FIG_SCALE),
  ]
  subproc_env = {**os.environ}
  subproc_env.setdefault('JAX_PLATFORMS', 'cpu')
  subproc_env.setdefault('MPLBACKEND', 'Agg')
  print(f'[subset] {" ".join(cmd)}', flush=True)
  return subprocess.call(cmd, cwd=repo, env=subproc_env)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--env', default='point_FourRooms')
  ap.add_argument('--log_root', default='ppo_fourrooms')
  ap.add_argument('--seed', type=int, required=True)
  ap.add_argument('--num_checkpoints', type=int, default=10)
  ap.add_argument('--repo_root', default='.')
  ap.add_argument(
      '--output_dir',
      default=None,
      help='PNG dir (default: plots/<slug>_<seed>_subset<N>_sub25_traj5/...).')
  args = ap.parse_args()

  repo = os.path.abspath(os.path.expanduser(args.repo_root))
  slug = args.log_root.removeprefix('ppo_')
  ckpt_dir = os.path.join(
      repo, 'logs', args.log_root, f'ppo_{args.env}_{args.seed}', 'checkpoints')
  out_dir = args.output_dir or os.path.join(
      repo,
      f'plots/ppo_{slug}_{args.seed}_subset{args.num_checkpoints}_sub25_traj5',
      f'ppo_{args.env}_{args.seed}',
  )

  iters = _list_iters(ckpt_dir)
  if not iters:
    print(f'[subset] no checkpoints in {ckpt_dir}', flush=True)
    sys.exit(1)

  picked = _pick_evenly(iters, args.num_checkpoints)
  print(
      f'[subset] seed={args.seed}  available={len(iters)}  '
      f'plotting {len(picked)} iters: {picked}',
      flush=True,
  )

  failed = 0
  for it in picked:
    pkl = os.path.join(ckpt_dir, f'ckpt_iter_{it:07d}.pkl')
    if not os.path.isfile(pkl):
      print(f'[subset] WARNING: missing {pkl}', flush=True)
      failed += 1
      continue
    if _plot_one(repo, pkl, out_dir, args.env, args.seed) != 0:
      failed += 1

  if failed:
    sys.exit(1)


if __name__ == '__main__':
  main()
