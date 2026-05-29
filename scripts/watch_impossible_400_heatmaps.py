#!/usr/bin/env python3
"""Plot new ``ppo_point_Impossible_400`` checkpoints (heatmap style).

Watches:
  - logs/ppo_impossible/ppo_point_Impossible_400/checkpoints
  - logs/ppo_impossible_dirac_baseline/ppo_point_Impossible_400/checkpoints

When a new ``.pkl`` appears (or ``latest.pkl`` mtime changes), runs
``ppo_rollout_maze.py`` for that file only (subcells=25, 5 trajectories).

State: ``plots/.watch_impossible_400_heatmaps_state.json``

Examples::

  python scripts/watch_impossible_400_heatmaps.py --once
  python scripts/watch_impossible_400_heatmaps.py --watch_interval 120
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

SEED = 400
ENV = 'point_Impossible'
RUNS = (
    ('default', 'ppo_impossible'),
    ('dirac_baseline', 'ppo_impossible_dirac_baseline'),
)
HEATMAP_SUBCELLS = 25
NUM_TRAJ = 5
FIG_SCALE = 2.0


def _ckpt_label(pkl_path: str) -> str:
  """Match ppo_rollout_maze._enumerate_checkpoints labels."""
  base = os.path.splitext(os.path.basename(pkl_path))[0]
  m = re.search(r'ckpt_iter_(\d+)\.pkl$', pkl_path)
  if m:
    return f'iter_{int(m.group(1)):07d}'
  if base == 'latest':
    return 'latest'
  return base[len('ckpt_'):] if base.startswith('ckpt_') else base


def _list_checkpoints(ckpt_dir: str) -> list[tuple[str, str, float]]:
  """(label, path, mtime) for each .pkl in dir."""
  out = []
  for p in sorted(glob.glob(os.path.join(ckpt_dir, '*.pkl'))):
    if not os.path.isfile(p):
      continue
    out.append((_ckpt_label(p), p, os.path.getmtime(p)))
  return out


def _state_key(log_tag: str, label: str) -> str:
  return f'{log_tag}|{label}'


def _load_state(path: str) -> dict:
  if not os.path.isfile(path):
    return {}
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def _save_state(path: str, state: dict) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)


def _expected_png(out_base: str, log_tag: str, label: str) -> str:
  # ppo_rollout_maze: dir output -> <env>_<label>.png + trajset suffix
  name = f'{ENV}_{label}_trajset{NUM_TRAJ:02d}.png' if NUM_TRAJ > 1 else f'{ENV}_{label}.png'
  return os.path.join(out_base, log_tag, name)


def _plot_one(repo: str, pkl_path: str, out_dir: str, seed: int) -> int:
  os.makedirs(out_dir, exist_ok=True)
  cmd = [
      sys.executable,
      '-u',
      os.path.join(repo, 'ppo_rollout_maze.py'),
      '--checkpoint',
      pkl_path,
      '--env',
      ENV,
      '--output',
      out_dir if out_dir.endswith(os.sep) else out_dir + os.sep,
      '--seed',
      str(seed),
      '--num_trajectories',
      str(NUM_TRAJ),
      '--heatmap_subcells',
      str(HEATMAP_SUBCELLS),
      '--fig_scale',
      str(FIG_SCALE),
  ]
  env = {**os.environ}
  env.setdefault('JAX_PLATFORMS', 'cpu')
  env.setdefault('MPLBACKEND', 'Agg')
  print(f'[watch400] {" ".join(cmd)}', flush=True)
  return subprocess.call(cmd, cwd=repo, env=env)


def _scan_and_plot(repo: str, out_base: str, state_path: str, force: bool) -> int:
  state = _load_state(state_path)
  n_new = 0
  for log_tag, log_root in RUNS:
    ckpt_dir = os.path.join(repo, 'logs', log_root, f'ppo_{ENV}_{SEED}', 'checkpoints')
    if not os.path.isdir(ckpt_dir):
      print(f'[watch400] skip missing {ckpt_dir}', flush=True)
      continue
    for label, pkl_path, mtime in _list_checkpoints(ckpt_dir):
      key = _state_key(log_tag, label)
      prev = state.get(key)
      out_dir = os.path.join(out_base, log_tag)
      out_png = _expected_png(out_base, log_tag, label)
      needs = force or prev is None or float(prev) < mtime
      if not needs and os.path.isfile(out_png):
        continue
      if not needs:
        # state says done but png missing — replot
        needs = not os.path.isfile(out_png)
      if not needs:
        continue
      print(f'[watch400] new/changed: {log_tag} {label} ({pkl_path})', flush=True)
      rc = _plot_one(repo, pkl_path, out_dir, SEED)
      if rc != 0:
        print(f'[watch400] WARNING: plot failed rc={rc} for {pkl_path}', flush=True)
        continue
      state[key] = mtime
      n_new += 1
      print(f'[watch400] saved {out_png}', flush=True)
  _save_state(state_path, state)
  return n_new


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--repo_root', default='.',
      help='Repo root (default: cwd).')
  ap.add_argument(
      '--output_base',
      default='plots/ppo_impossible_400_heatmaps',
      help='PNG output root (subdirs: default/, dirac_baseline/).')
  ap.add_argument(
      '--state_file',
      default='plots/.watch_impossible_400_heatmaps_state.json',
      help='JSON tracking plotted checkpoint mtimes.')
  ap.add_argument(
      '--once', action='store_true',
      help='Single scan then exit.')
  ap.add_argument(
      '--force', action='store_true',
      help='Replot all checkpoints regardless of state.')
  ap.add_argument(
      '--watch_interval', type=float, default=120.0,
      help='Seconds between scans (default: 120).')
  args = ap.parse_args()

  repo = os.path.abspath(os.path.expanduser(args.repo_root))
  out_base = os.path.join(repo, args.output_base)
  state_path = os.path.join(repo, args.state_file)

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[watch400] === scan @ {ts} ===', flush=True)
      n = _scan_and_plot(repo, out_base, state_path, args.force)
      print(f'[watch400] plotted {n} checkpoint(s)', flush=True)
      if args.once:
        break
      delay = max(15.0, float(args.watch_interval))
      print(f'[watch400] next scan in {delay:.0f}s', flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[watch400] stopped.', flush=True)


if __name__ == '__main__':
  main()
