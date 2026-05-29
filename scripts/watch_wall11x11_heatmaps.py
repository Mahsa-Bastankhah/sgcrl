#!/usr/bin/env python3
"""Plot new point-maze checkpoint heatmaps as they appear.

Examples::

  python scripts/watch_wall11x11_heatmaps.py --seed 300
  python scripts/watch_wall11x11_heatmaps.py --env point_FourRooms \\
      --log_root ppo_fourrooms --seed 123
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HEATMAP_SUBCELLS = 25
NUM_TRAJ = 5
FIG_SCALE = 2.0


def _ckpt_label(pkl_path: str) -> str:
  base = os.path.splitext(os.path.basename(pkl_path))[0]
  m = re.search(r'ckpt_iter_(\d+)\.pkl$', pkl_path)
  if m:
    return f'iter_{int(m.group(1)):07d}'
  if base == 'latest':
    return 'latest'
  return base[len('ckpt_'):] if base.startswith('ckpt_') else base


def _list_checkpoints(ckpt_dir: str) -> list[tuple[str, str, float]]:
  out = []
  for p in sorted(glob.glob(os.path.join(ckpt_dir, '*.pkl'))):
    if os.path.isfile(p):
      out.append((_ckpt_label(p), p, os.path.getmtime(p)))
  return out


def _load_state(path: str) -> dict:
  if not os.path.isfile(path):
    return {}
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def _save_state(path: str, state: dict) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)


def _expected_png(out_dir: str, env: str, label: str) -> str:
  name = f'{env}_{label}_trajset{NUM_TRAJ:02d}.png' if NUM_TRAJ > 1 else f'{env}_{label}.png'
  return os.path.join(out_dir, name)


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
  print(f'[watch_maze] {" ".join(cmd)}', flush=True)
  return subprocess.call(cmd, cwd=repo, env=subproc_env)


def _scan(
    repo: str,
    env: str,
    log_root: str,
    seed: int,
    out_dir: str,
    state_path: str,
    force: bool,
) -> int:
  ckpt_dir = os.path.join(
      repo, 'logs', log_root, f'ppo_{env}_{seed}', 'checkpoints')
  if not os.path.isdir(ckpt_dir):
    print(f'[watch_maze] missing {ckpt_dir}', flush=True)
    return 0
  state = _load_state(state_path)
  n_new = 0
  for label, pkl_path, mtime in _list_checkpoints(ckpt_dir):
    key = f'{seed}|{label}'
    prev = state.get(key)
    out_png = _expected_png(out_dir, env, label)
    needs = force or prev is None or float(prev) < mtime
    if not needs and os.path.isfile(out_png):
      continue
    if not needs:
      needs = not os.path.isfile(out_png)
    if not needs:
      continue
    print(f'[watch_maze] seed={seed} {label} ({pkl_path})', flush=True)
    if _plot_one(repo, pkl_path, out_dir, env, seed) != 0:
      print(f'[watch_maze] WARNING: failed {pkl_path}', flush=True)
      continue
    state[key] = mtime
    n_new += 1
    print(f'[watch_maze] saved {out_png}', flush=True)
  _save_state(state_path, state)
  return n_new


def _log_slug(log_root: str) -> str:
  return log_root.removeprefix('ppo_')


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--env', default='point_Wall11x11')
  ap.add_argument('--log_root', default='ppo_wall11x11',
                  help='Subdir under logs/, e.g. ppo_fourrooms.')
  ap.add_argument('--seed', type=int, required=True)
  ap.add_argument('--repo_root', default='.')
  ap.add_argument(
      '--output_dir',
      default=None,
      help='PNG dir (default: plots/<log_slug>_<seed>_all_ckpts_sub25_traj5/...).')
  ap.add_argument('--state_file', default=None)
  ap.add_argument('--once', action='store_true')
  ap.add_argument('--force', action='store_true')
  ap.add_argument('--watch_interval', type=float, default=120.0)
  args = ap.parse_args()

  repo = os.path.abspath(os.path.expanduser(args.repo_root))
  slug = _log_slug(args.log_root)
  out_dir = args.output_dir or os.path.join(
      repo,
      f'plots/ppo_{slug}_{args.seed}_all_ckpts_sub25_traj5',
      f'ppo_{args.env}_{args.seed}',
  )
  state_path = args.state_file or os.path.join(
      repo, 'plots', f'.watch_{slug}_{args.seed}_heatmaps_state.json')

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(
          f'\n[watch_maze] === scan @ {ts} env={args.env} seed={args.seed} ===',
          flush=True,
      )
      n = _scan(
          repo, args.env, args.log_root, args.seed,
          out_dir, state_path, args.force,
      )
      print(f'[watch_maze] plotted {n} checkpoint(s)', flush=True)
      if args.once:
        break
      time.sleep(max(30.0, float(args.watch_interval)))
  except KeyboardInterrupt:
    print('\n[watch_maze] stopped.', flush=True)


if __name__ == '__main__':
  main()
