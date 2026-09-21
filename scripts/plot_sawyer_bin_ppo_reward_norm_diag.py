#!/usr/bin/env python3
"""Sawyer bin PPO+NF: raw reward + return-normalizer std for the three live settings.

Orange: ent=0.05 rand mixtaskg
Green:  ent=0.005 rand mixtaskg
Pink:   norand seed 2 (current code)

  python scripts/plot_sawyer_bin_ppo_reward_norm_diag.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import plot_builderbench_train_success1000 as base  # noqa: E402

LOG_ROOT = os.path.join(REPO, 'logs', 'final_metaworld_runs')
OUT_DIR = os.path.join(REPO, 'figs', 'metaworld', 'final_metaworld_runs')
OUT_STEM = os.path.join(OUT_DIR, 'sawyer_bin_reward_norm_diag')

PPO_STEPS_PER_ITER = 1024
SUBSAMPLE_EVERY = 20  # learner CSVs are ~30k–40k rows

RUNS = (
    {
        'label': 'ent=0.05 rand (orange)',
        'color': '#D55E00',
        'path': os.path.join(
            LOG_ROOT,
            'ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
            '_minstd1e5_extrew1_rand_mixtaskg',
            'ppo_sawyer_bin_0',
            'logs', 'learner', 'logs.csv'),
    },
    {
        'label': 'ent=0.005 rand (green)',
        'color': '#009E73',
        'path': os.path.join(
            LOG_ROOT,
            'ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
            '_minstd1e5_extrew1_rand_ent0005_mixtaskg',
            'ppo_sawyer_bin_0',
            'logs', 'learner', 'logs.csv'),
    },
    {
        'label': 'norand s2 current (pink)',
        'color': '#CC79A7',
        'path': os.path.join(
            LOG_ROOT,
            'ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
            '_minstd1e5_extrew1_norand',
            'ppo_sawyer_bin_2',
            'logs', 'learner', 'logs.csv'),
    },
)

Y_KEYS = (
    ('reward_repr_raw_mean', 'Raw PPO reward  (reward_repr_raw_mean)'),
    ('reward_return_norm_std', 'Return normalizer std  (reward_return_norm_std)'),
)


def _finite(v):
  return v is not None and math.isfinite(v)


def _read_series(path: str, y_key: str) -> list[tuple[int, float]]:
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return []
  pts: list[tuple[int, float]] = []
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    if y_key not in (reader.fieldnames or []):
      return []
    for i, row in enumerate(reader):
      if i % SUBSAMPLE_EVERY != 0:
        continue
      y = base._coerce(row.get(y_key))
      if not _finite(y):
        continue
      x = base._coerce(row.get('global_step'))
      if not _finite(x):
        it = base._coerce(row.get('iteration'))
        if _finite(it):
          x = it * PPO_STEPS_PER_ITER
      if not _finite(x):
        continue
      pts.append((int(x), float(y)))
  pts.sort(key=lambda p: p[0])
  return pts


def main() -> None:
  plt.rcParams.update({
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
      'font.size': 13,
      'axes.titlesize': 15,
      'axes.labelsize': 14,
      'legend.fontsize': 12,
      'axes.linewidth': 1.1,
      'savefig.dpi': 300,
  })
  fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)

  for ax, (y_key, title) in zip(axes, Y_KEYS):
    for run in RUNS:
      pts = _read_series(run['path'], y_key)
      if not pts:
        print(f'  {run["label"]} {y_key}: no data')
        continue
      xs = [p[0] for p in pts]
      ys = [p[1] for p in pts]
      ax.plot(xs, ys, color=run['color'], lw=2.0, alpha=0.95,
              label=run['label'], solid_capstyle='round')
      print(f'  {run["label"]} {y_key}: n={len(pts)} '
            f'last={ys[-1]:.4g} @ {xs[-1]/1e6:.2f}M')
    ax.set_title(title, fontweight='bold', pad=8)
    ax.set_xlabel('Environment steps')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

  handles = [
      Line2D([0], [0], color=r['color'], lw=2.6, label=r['label'])
      for r in RUNS
  ]
  fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
             bbox_to_anchor=(0.5, 1.04), handlelength=2.2, columnspacing=1.6)
  fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14, wspace=0.22)

  os.makedirs(OUT_DIR, exist_ok=True)
  for ext in ('.png', '.pdf'):
    out = OUT_STEM + ext
    tmp = out + '.tmp' + ext
    fig.savefig(tmp, dpi=300, bbox_inches='tight')
    os.replace(tmp, out)
    print(f'→ {out}')
  plt.close(fig)


if __name__ == '__main__':
  main()
