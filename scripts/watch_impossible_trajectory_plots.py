#!/usr/bin/env python3
"""Re-render ``ppo_rollout_maze`` PNGs for every ``ppo_point_Impossible_*`` run.

Each cycle glob-searches under ``--log_roots`` for
``.../ppo_point_Impossible_<seed>/checkpoints`` (must contain at least one
``.pkl``), then invokes ``ppo_rollout_maze.py`` so **new** ``ckpt_iter_*.pkl``
files are picked up automatically.

Uses the same settings as the manual Impossible sweep: ``point_Impossible``,
``--num_trajectories 5``, ``--heatmap_subcells 25``, ``--fig_scale 2``.

Examples::

  # One shot (no loop)
  python scripts/watch_impossible_trajectory_plots.py --once

  # Refresh every 3 minutes (good in tmux on a GPU node)
  python scripts/watch_impossible_trajectory_plots.py --watch_interval 180

  # CPU-only (e.g. login node without GPU)
  JAX_PLATFORMS=cpu python scripts/watch_impossible_trajectory_plots.py --once
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time


def _iter_checkpoint_dirs(log_roots: list[str]) -> list[str]:
  out: list[str] = []
  for root in log_roots:
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
      continue
    pat = os.path.join(root, '**', 'ppo_point_Impossible_*', 'checkpoints')
    for d in sorted(glob.glob(pat, recursive=True)):
      if not os.path.isdir(d):
        continue
      pkls = glob.glob(os.path.join(d, '*.pkl'))
      if not pkls:
        continue  # includes latest.pkl-only dirs
      out.append(os.path.normpath(d))
  # Stable unique order
  return sorted(set(out))


def _run_name_and_seed(ckpt_dir: str) -> tuple[str, int]:
  run_dir = os.path.dirname(ckpt_dir)
  run_name = os.path.basename(run_dir)
  m = re.search(r'ppo_point_Impossible_(\d+)$', run_name)
  seed = int(m.group(1)) if m else 0
  return run_name, seed


def _out_subdir(out_base: str, ckpt_dir: str, nested_by_parent: bool) -> str:
  """Match manual runs: ``.../ppo_point_Impossible_<seed>/`` unless nested."""
  run_name, _ = _run_name_and_seed(ckpt_dir)
  if not nested_by_parent:
    return os.path.join(out_base, run_name)
  cfg_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_dir)))
  return os.path.join(out_base, f'{cfg_name}__{run_name}')


def _render_one(
    repo_root: str,
    ckpt_dir: str,
    out_sub: str,
    extra_env: dict[str, str],
) -> int:
  os.makedirs(out_sub, exist_ok=True)
  _, seed = _run_name_and_seed(ckpt_dir)
  cmd = [
      sys.executable,
      '-u',
      os.path.join(repo_root, 'ppo_rollout_maze.py'),
      '--checkpoint',
      ckpt_dir,
      '--env',
      'point_Impossible',
      '--output',
      out_sub + os.sep,
      '--seed',
      str(seed),
      '--num_trajectories',
      '5',
      '--heatmap_subcells',
      '25',
      '--fig_scale',
      '2.0',
  ]
  env = {**os.environ, **extra_env}
  print(f'[watch_impossible] {" ".join(cmd)}', flush=True)
  return subprocess.call(cmd, cwd=repo_root, env=env)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--log_roots',
      nargs='+',
      default=['logs/ppo_fourrooms'],
      help='Directories to search recursively (default: logs/ppo_fourrooms).')
  ap.add_argument(
      '--output_base',
      default='plots/ppo_fourrooms_impossible_sub25_traj5',
      help='PNG tree root (default: plots/ppo_fourrooms_impossible_sub25_traj5).')
  ap.add_argument(
      '--repo_root',
      default='.',
      help='Repo root (default: cwd).')
  ap.add_argument(
      '--watch_interval',
      type=float,
      default=120.0,
      help='Seconds between full scan cycles (default: 120).')
  ap.add_argument(
      '--once',
      action='store_true',
      help='Run a single cycle then exit (no watch loop).')
  ap.add_argument(
      '--nested_by_parent',
      action='store_true',
      help='Put PNGs under ``<parent_cfg>__ppo_point_Impossible_<seed>/`` '
           'so different log parents with the same seed do not overwrite.')
  args = ap.parse_args()

  repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
  out_base = os.path.join(repo_root, args.output_base)
  extra_env = {}
  # Respect JAX_PLATFORMS if user already exported it; do not override.

  cycle = 0
  try:
    while True:
      cycle += 1
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[watch_impossible] === cycle {cycle} @ {ts} ===', flush=True)
      dirs = _iter_checkpoint_dirs(list(args.log_roots))
      if not dirs:
        print('[watch_impossible] no Impossible checkpoint dirs found yet.',
              flush=True)
      for ck in dirs:
        sub = _out_subdir(out_base, ck, args.nested_by_parent)
        rc = _render_one(repo_root, ck, sub, extra_env)
        if rc != 0:
          print(f'[watch_impossible] WARNING: exit {rc} for {ck}', flush=True)
      if args.once:
        break
      delay = max(15.0, float(args.watch_interval))
      print(f'[watch_impossible] next cycle in {delay:.0f}s  (Ctrl+C to stop)',
            flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[watch_impossible] stopped.', flush=True)


if __name__ == '__main__':
  main()
