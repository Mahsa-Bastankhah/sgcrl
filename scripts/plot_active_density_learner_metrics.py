#!/usr/bin/env python3
"""Plot learner metrics for active TD3 / Gaussian / NF BuilderBench runs.

Produces 9 figures (3 measures × 3 density types), each overlaying variants
of that density type with a legend:

  1. log_p_mean  (NF / Gaussian)  or  td3_q1_mean  (TD3; no log_p)
  2. ppo/policy_loc_abs_mean      (action-mean magnitude)
  3. density / critic loss

  python scripts/plot_active_density_learner_metrics.py
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(REPO, 'figs', 'builderbench', 'density_learner_metrics')


def _read_csv(path: str) -> List[dict]:
  with open(path, 'rb') as fh:
    raw = fh.read().replace(b'\x00', b'')
  return list(csv.DictReader(io.StringIO(raw.decode('utf-8', errors='replace'))))


def _f(v) -> Optional[float]:
  if v is None:
    return None
  s = str(v).strip()
  if not s:
    return None
  try:
    x = float(s)
  except ValueError:
    return None
  if math.isnan(x) or math.isinf(x):
    return None
  return x


def _series(rows: Sequence[dict], y_key: str, x_key: str = 'global_step'
            ) -> Tuple[np.ndarray, np.ndarray]:
  xs, ys = [], []
  for r in rows:
    x = _f(r.get(x_key))
    y = _f(r.get(y_key))
    if x is None or y is None:
      continue
    xs.append(x)
    ys.append(y)
  if not xs:
    return np.asarray([]), np.asarray([])
  order = np.argsort(xs)
  return np.asarray(xs)[order], np.asarray(ys)[order]


def _smooth(y: np.ndarray, win: int) -> np.ndarray:
  if win <= 1 or len(y) < 3:
    return y
  w = min(win, max(3, len(y) // 20 * 2 + 1))
  if w % 2 == 0:
    w += 1
  kernel = np.ones(w, dtype=np.float64) / w
  pad = w // 2
  yp = np.pad(y.astype(np.float64), (pad, pad), mode='edge')
  return np.convolve(yp, kernel, mode='valid')


# Active runs grouped by density estimator type.
# (label, learner csv path relative to REPO)
RUNS: Dict[str, List[Tuple[str, str]]] = {
    'td3': [
        ('q1online',
         'logs/ppo_builderbench_creative3_task1_e1024_pd_td3_q1online/'
         'ppo_builderbench_creative_3_task1_0/logs/learner/logs.csv'),
        ('bilinear',
         'logs/ppo_builderbench_creative3_task1_e1024_pd_td3_bilinear/'
         'ppo_builderbench_creative_3_task1_0/logs/learner/logs.csv'),
        ('tau0.5',
         'logs/ppo_builderbench_creative3_task1_e1024_pd_td3_tau05/'
         'ppo_builderbench_creative_3_task1_0/logs/learner/logs.csv'),
    ],
    'gaussian': [
        ('no-tau',
         'logs/ppo_builderbench_creative3_task1_e1024_pd_gaussian/'
         'ppo_builderbench_creative_3_task1_0/logs/learner/logs.csv'),
        ('tau0.5',
         'logs/ppo_builderbench_creative3_task1_e1024_pd_gaussian_tau05/'
         'ppo_builderbench_creative_3_task1_0/logs/learner/logs.csv'),
    ],
    'nf': [
        ('c3t1 seed1',
         'logs/ppo_builderbench_creative3_task1_e1024_pd_nf/'
         'ppo_builderbench_creative_3_task1_1/logs/learner/logs.csv'),
        ('c3t1 tau0.5 seed2',
         'logs/ppo_builderbench_creative3_task1_e1024_pd_nf_tau05/'
         'ppo_builderbench_creative_3_task1_2/logs/learner/logs.csv'),
        ('c3t2 tau0.5',
         'logs/ppo_builderbench_creative3_task2_e1024_pd_nf_tau05/'
         'ppo_builderbench_creative_3_task2_0/logs/learner/logs.csv'),
    ],
}

# measure_key -> (ylabel, {density_type: csv_column})
MEASURES = {
    'log_p_or_q': (
        'log p mean  /  TD3 Q1 mean',
        {
            'td3': 'td3/td3_q1_mean',
            'gaussian': 'gaussian/log_p_mean',
            'nf': 'nf/log_p_mean',
        },
    ),
    'action_mean': (
        'policy loc |mean|  (ppo/policy_loc_abs_mean)',
        {
            'td3': 'ppo/policy_loc_abs_mean',
            'gaussian': 'ppo/policy_loc_abs_mean',
            'nf': 'ppo/policy_loc_abs_mean',
        },
    ),
    'density_loss': (
        'density / critic loss',
        {
            'td3': 'td3/td3_qf_loss',
            'gaussian': 'gaussian/gaussian_density_loss',
            'nf': 'nf/density_loss',
        },
    ),
}


def _plot_one(
    density: str,
    measure: str,
    ylabel: str,
    col: str,
    out_path: str,
    smooth: int,
) -> None:
  fig, ax = plt.subplots(figsize=(8.5, 4.8))
  n_drawn = 0
  for label, rel in RUNS[density]:
    path = os.path.join(REPO, rel)
    if not os.path.isfile(path):
      print(f'[skip] missing {path}', flush=True)
      continue
    rows = _read_csv(path)
    x, y = _series(rows, col)
    if len(x) == 0:
      print(f'[skip] empty {label} col={col}', flush=True)
      continue
    y_plot = _smooth(y, smooth)
    ax.plot(x, y_plot, label=f'{label} (n={len(x)})', linewidth=1.6)
    n_drawn += 1

  ax.set_xlabel('global_step')
  ax.set_ylabel(ylabel)
  title_measure = {
      'log_p_or_q': 'log_p (or Q1 for TD3)',
      'action_mean': 'action mean (|policy loc|)',
      'density_loss': 'density estimator loss',
  }[measure]
  ax.set_title(f'{density.upper()} — {title_measure}')
  if n_drawn:
    ax.legend(frameon=False, loc='best')
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
  fig.savefig(out_path, dpi=160)
  plt.close(fig)
  print(f'[wrote] {out_path}  ({n_drawn} curves)', flush=True)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--out_dir', default=FIGS,
      help='Output directory for PNGs.')
  ap.add_argument(
      '--smooth', type=int, default=21,
      help='Odd rolling-mean window (1 = off).')
  args = ap.parse_args()
  out_dir = args.out_dir
  if not os.path.isabs(out_dir):
    out_dir = os.path.join(REPO, out_dir)

  for density in ('td3', 'gaussian', 'nf'):
    for measure, (ylabel, col_by_type) in MEASURES.items():
      col = col_by_type[density]
      out = os.path.join(out_dir, f'{density}_{measure}.png')
      _plot_one(density, measure, ylabel, col, out, args.smooth)

  print(f'\nAll figures under: {out_dir}', flush=True)


if __name__ == '__main__':
  main()
