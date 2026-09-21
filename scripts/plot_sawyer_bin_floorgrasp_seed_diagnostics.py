#!/usr/bin/env python3
"""Compare PPO/NF diagnostics across Sawyer-bin floor-grasp seeds 0, 1, 2."""
from __future__ import annotations

import csv
import math
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import plot_builderbench_train_success1000 as base  # noqa: E402

RUN_ROOT = os.path.join(
    REPO, 'logs', 'final_metaworld_runs',
    'ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_mixtaskg')
OUT_STEM = os.path.join(
    REPO, 'figs', 'metaworld', 'final_metaworld_runs',
    'sawyer_bin_floorgrasp_seed_diagnostics')

PPO_STEPS_PER_ITER = 1024
SMOOTH_ITERS = 200
DRAW_EVERY = 20

SEEDS = (
    (0, 'Seed 0', '#6A3D9A'),
    (1, 'Seed 1', '#D55E00'),
    (2, 'Seed 2', '#009E73'),
)

PANELS = (
    ('reward_repr_raw_mean', 'Raw PPO representation reward',
     'Mean raw reward per step'),
    ('reward_return_norm_std', 'Return-normalizer standard deviation',
     'Running return std'),
    ('ppo/policy_loc_abs_mean', 'Policy mean absolute magnitude (pre-tanh)',
     r'Mean $|\mu|$'),
    ('ppo/policy_scale_mean', 'Policy scale (action std)',
     'Mean policy scale'),
    ('nf/log_p_min', 'Replay-batch NF log p minimum',
     'Minimum log p'),
    ('nf/log_p_max', 'Replay-batch NF log p maximum',
     'Maximum log p'),
    ('ppo/pg_loss', 'PPO policy-gradient loss',
     'PG loss'),
)


def _coerce(value):
  try:
    value = float(value)
  except (TypeError, ValueError):
    return None
  return value if math.isfinite(value) else None


def _read_metric(seed: int, key: str) -> tuple[np.ndarray, np.ndarray]:
  path = os.path.join(
      RUN_ROOT, f'ppo_sawyer_bin_{seed}', 'logs', 'learner', 'logs.csv')
  if not os.path.isfile(path):
    return np.asarray([]), np.asarray([])
  xs, ys = [], []
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    if key not in (reader.fieldnames or []):
      return np.asarray([]), np.asarray([])
    for row in reader:
      y = _coerce(row.get(key))
      x = _coerce(row.get('global_step'))
      if x is None:
        iteration = _coerce(row.get('iteration'))
        x = None if iteration is None else iteration * PPO_STEPS_PER_ITER
      if x is not None and y is not None:
        xs.append(x)
        ys.append(y)
  return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _centered_mean(values: np.ndarray, window: int) -> np.ndarray:
  if values.size == 0 or window <= 1:
    return values.copy()
  half = window // 2
  prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
  idx = np.arange(values.size)
  lo = np.maximum(0, idx - half)
  hi = np.minimum(values.size, idx + half + 1)
  return (prefix[hi] - prefix[lo]) / (hi - lo)


def main() -> None:
  plt.rcParams.update({
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
      'font.size': 12,
      'axes.titlesize': 14,
      'axes.labelsize': 13,
      'legend.fontsize': 12,
      'axes.linewidth': 1.05,
      'pdf.fonttype': 42,
      'savefig.dpi': 300,
  })
  fig, axes = plt.subplots(4, 2, figsize=(13.0, 14.0), sharex=True)

  for ax, (key, title, ylabel) in zip(axes.flat, PANELS):
    for seed, label, color in SEEDS:
      xs, ys = _read_metric(seed, key)
      if xs.size == 0:
        print(f'{label} {key}: no data')
        continue
      smooth = _centered_mean(ys, SMOOTH_ITERS)
      draw = slice(None, None, DRAW_EVERY)
      ax.plot(xs[draw], ys[draw], color=color, lw=0.7, alpha=0.20,
              zorder=1)
      ax.plot(xs[draw], smooth[draw], color=color, lw=2.2, alpha=0.98,
              zorder=2)
      print(f'{label} {key}: steps={xs[-1]/1e6:.2f}M '
            f'last={ys[-1]:.5g} smooth={smooth[-1]:.5g}')
    ax.set_title(title, fontweight='bold', pad=7)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    ax.grid(axis='y', linestyle='--', alpha=0.30)
    ax.spines[['top', 'right']].set_visible(False)

  for ax in axes.flat[len(PANELS):]:
    ax.set_visible(False)

  for ax in axes[-1]:
    ax.set_xlabel('Environment steps')

  handles = [
      Line2D([0], [0], color=color, lw=2.6, label=label)
      for _, label, color in SEEDS
  ]
  fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
             bbox_to_anchor=(0.5, 0.995), columnspacing=2.2)
  fig.suptitle(
      'Sawyer bin PPO+NF floor-grasp diagnostics',
      y=1.025, fontsize=16, fontweight='bold')
  fig.text(
      0.5, 0.008,
      f'Source: learner logs · full available run · faint raw, bold centered '
      f'rolling mean ({SMOOTH_ITERS} PPO iterations).',
      ha='center', fontsize=10, color='#444444')
  fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.07,
                      hspace=0.34, wspace=0.23)

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  pid = os.getpid()
  for ext in ('.png', '.pdf'):
    output = OUT_STEM + ext
    tmp = f'{OUT_STEM}.{pid}.tmp{ext}'
    fig.savefig(tmp, bbox_inches='tight')
    os.replace(tmp, output)
    print(f'→ {output}')
  plt.close(fig)


if __name__ == '__main__':
  main()
