#!/usr/bin/env python3
"""Compare eval success: preimage matching (default) vs dirac target (baseline).

Reads ``logs/<log_root>/<prefix>_point_<Env>_<seed>/logs/eval/logs.csv`` for
seeds 123–128 (override with ``--seeds``) and plots mean ± SE across seeds
(shaded band).  Run prefix is ``ppo`` or ``wbc`` (auto from log folder name).

Outputs PNGs under ``--output_dir`` (``success_*`` and ``success_1000_*``).

Examples::

  python scripts/plot_success_default_vs_dirac.py --once
  python scripts/plot_success_default_vs_dirac.py --watch_interval 120
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS_DEFAULT = list(range(123, 129))

# (legend label, log folder under logs/)
CONDITIONS = {
    'Impossible': [
        ('preimage matching (default)', 'ppo_impossible'),
    ],
    'Spiral11x11': [
        ('preimage matching (default)', 'ppo_spiral11x11'),
    ],
    'Wall11x11': [
        ('preimage matching (default)', 'ppo_wall11x11'),
    ],
    'EightRooms': [
        ('preimage matching (default)', 'ppo_eightrooms'),
        ('KDE dirac', 'ppo_eightrooms_kde_dirac'),
    ],
    'SixteenRooms': [
        ('preimage matching (default)', 'ppo_sixteenrooms'),
        ('KDE dirac', 'ppo_eightrooms_kde_dirac'),
    ],
    'SixteenRooms4D': [
        ('preimage matching (default)', 'ppo_sixteenrooms4d_1d'),
        ('KDE dirac', 'ppo_sixteenrooms4d_1d_kde_dirac'),
    ],
    'SixteenRoomsActual4D': [
        ('PPO nouniform', 'ppo_sixteenroomsactual4d_nouniform'),
        ('WBC nouniform locclip', 'wbc_sixteenroomsactual4d_nouniform_locclip'),
    ],
}


def _run_prefix(log_root: str) -> str:
  base = log_root.strip('/').split('/')[-1]
  return 'wbc' if base.startswith('wbc_') else 'ppo'


def _read_eval_series(path: str, metric: str) -> Dict[int, float]:
  with open(path, 'rb') as fh:
    raw = fh.read().replace(b'\x00', b'')
  text = raw.decode('utf-8', errors='replace')
  lines = [ln for ln in text.splitlines() if ln.strip()]
  if not lines:
    return {}
  out: Dict[int, float] = {}
  header = lines[0].split(',')
  try:
    it_idx = header.index('iteration')
    met_idx = header.index(metric)
  except ValueError:
    return {}
  for ln in lines[1:]:
    parts = ln.split(',')
    if len(parts) <= max(it_idx, met_idx):
      continue
    try:
      it = int(float(parts[it_idx].strip()))
      y = float(parts[met_idx].strip())
      out[it] = y
    except (TypeError, ValueError):
      continue
  return out


def _aggregate_seeds(
    series_list: List[Dict[int, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Mean and SE (std / sqrt(n)) across seeds at each iteration."""
  all_iters = set()
  for s in series_list:
    all_iters.update(s.keys())
  iters = sorted(all_iters)
  means, ses = [], []
  for it in iters:
    vals = np.array([s[it] for s in series_list if it in s], dtype=np.float64)
    if vals.size == 0:
      continue
    means.append(float(vals.mean()))
    if vals.size > 1:
      ses.append(float(vals.std(ddof=1) / np.sqrt(vals.size)))
    else:
      ses.append(0.0)
  return np.asarray(iters, dtype=np.int32), np.asarray(means), np.asarray(ses)


def _load_condition(
    log_root: str,
    env: str,
    seeds: List[int],
    metric: str,
    run_prefix: str | None = None,
) -> List[Dict[int, float]]:
  prefix = run_prefix or _run_prefix(log_root)
  series = []
  for seed in seeds:
    path = os.path.join(
        log_root, f'{prefix}_point_{env}_{seed}', 'logs', 'eval', 'logs.csv')
    if not os.path.isfile(path):
      print(f'[warn] missing {path}')
      continue
    s = _read_eval_series(path, metric)
    if s:
      series.append(s)
      print(f'  loaded {path} ({len(s)} points)')
  return series


def _plot_env(
    env: str,
    conditions: List[Tuple[str, str]],
    seeds: List[int],
    metric: str,
    output_path: str,
    x_axis: str = 'iteration',
) -> None:
  fig, ax = plt.subplots(figsize=(9, 5))
  _palette = ['#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed', '#0891b2']
  colors = [_palette[i % len(_palette)] for i in range(len(conditions))]

  for (label, log_root), color in zip(conditions, colors):
    print(f'[{env}] {label} <- logs/{log_root} ({_run_prefix(log_root)})')
    series_list = _load_condition(
        os.path.join('logs', log_root), env, seeds, metric)
    if not series_list:
      print(f'  skip (no data)')
      continue
    xs, mean, se = _aggregate_seeds(series_list)
    if xs.size == 0:
      continue
    ax.plot(xs, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(xs, mean - se, mean + se, color=color, alpha=0.25, linewidth=0)

  ax.set_xlabel('PPO iteration' if x_axis == 'iteration' else 'learner steps')
  ylab = 'success' if metric == 'success' else 'success_1000 (rolling eval mean)'
  ax.set_ylabel(ylab)
  ax.set_title(f'{env}: eval {ylab} (seeds {seeds[0]}–{seeds[-1]}, mean ± SE)')
  ax.set_ylim(-0.02, 1.02)
  ax.legend(loc='best', framealpha=0.9)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
  fig.savefig(output_path, dpi=150)
  plt.close(fig)
  print(f'[saved] {output_path}')


def _run_all_plots(output_dir: str, metrics: List[str], seeds: List[int],
                   envs: List[str]) -> None:
  for metric in metrics:
    for env, conds in CONDITIONS.items():
      if envs and env not in envs:
        continue
      out = os.path.join(output_dir, f'{metric}_{env}.png')
      _plot_env(env, conds, seeds, metric, out)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--output_dir', default='plots/success_default_vs_dirac',
      help='Directory for output PNGs.')
  ap.add_argument(
      '--metric', default='',
      choices=('', 'success', 'success_1000'),
      help='Single metric to plot. Empty = both success and success_1000.')
  ap.add_argument(
      '--metrics', nargs='+', default=None,
      choices=('success', 'success_1000'),
      help='Explicit metric list (overrides --metric).')
  ap.add_argument(
      '--seeds', type=int, nargs='+', default=SEEDS_DEFAULT,
      help='Seeds to average (default: 123 … 128).')
  ap.add_argument(
      '--env', nargs='+', default=None,
      choices=list(CONDITIONS.keys()),
      help='Restrict to these environment(s). Default: all.')
  ap.add_argument(
      '--once', action='store_true',
      help='Single refresh then exit (default if --watch_interval not set).')
  ap.add_argument(
      '--watch_interval', type=float, default=0.0,
      help='Seconds between refreshes. >0 runs until interrupted.')
  args = ap.parse_args()

  repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
  os.chdir(repo)

  if args.metrics:
    metrics = list(args.metrics)
  elif args.metric:
    metrics = [args.metric]
  else:
    metrics = ['success', 'success_1000']

  watch = float(args.watch_interval) > 0
  if not watch and not args.once:
    args.once = True

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[success_plot] === refresh @ {ts} ===', flush=True)
      _run_all_plots(args.output_dir, metrics, args.seeds, args.env or [])
      if args.once:
        break
      delay = max(30.0, float(args.watch_interval))
      print(f'[success_plot] next refresh in {delay:.0f}s', flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[success_plot] stopped.', flush=True)


if __name__ == '__main__':
  main()
