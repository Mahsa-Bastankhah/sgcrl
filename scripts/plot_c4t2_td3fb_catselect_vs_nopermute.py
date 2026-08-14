#!/usr/bin/env python3
"""c4t2 TD3-FB: catselect vs nopermute+norand+normobs (mean±SE train success).

Writes:
  figs/builderbench/active_train_eval/creative4_task2_td3fb_catselect_vs_nopermute_train_success1000.png

Usage:
  python scripts/plot_c4t2_td3fb_catselect_vs_nopermute.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plot_builderbench_train_success1000 as base  # noqa: E402

OUT_NAME = (
    'creative4_task2_td3fb_catselect_vs_nopermute_train_success1000.png'
)

# (legend label, linestyle, [slurm log basenames])
GROUPS = [
    (
        'catselect',
        '-',
        [
            'ppo_builderbench_creative4_task2_e1024_pd_td3_fb_logq_tau05_catselect_3677735_0.log',
            'ppo_builderbench_creative4_task2_e1024_pd_td3_fb_logq_tau05_catselect_3677735_1.log',
        ],
    ),
    (
        'noperm+norand+norm',
        '--',
        [
            'ppo_builderbench_creative4_task2_e1024_pd_td3_fb_logq_tau05_nopermute_norand_normobs_catselect_3672299_0.log',
            'ppo_builderbench_creative4_task2_e1024_pd_td3_fb_logq_tau05_nopermute_norand_normobs_catselect_3677734_1.log',
        ],
    ),
]


def plot_once(figs_dir: str = base.FIGS_DIR,
              slurm_dir: str = base.SLURM_DIR) -> str:
  os.makedirs(figs_dir, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10.5, 5.2))
  plotted = 0

  for i, (label, ls, log_names) in enumerate(GROUPS):
    seed_series = []
    for log_name in log_names:
      path = os.path.join(slurm_dir, log_name)
      pts = base._parse_train_slurm_path(path)
      if pts:
        seed_series.append(pts)
    xs, mean, se, n_seeds = base._aggregate_mean_stderr(seed_series)
    if not xs:
      print(f'  {label}: no train data')
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    ax.plot(
        xs, mean,
        color=color, linewidth=2.2, linestyle=ls,
        label=f'{label} (n={n_seeds})', alpha=0.95, zorder=3 + i)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0,
                      zorder=2 + i)
    print(f'  {label}: n_seeds={n_seeds}, {len(xs)} pts  '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(mean):.3f}..{max(mean):.3f}]')
    plotted += 1

  ax.set_title(
      'Creative-4 Task2 — TD3-FB train success_1000\n'
      'catselect vs nopermute + norand + normobs',
      fontsize=12, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel('Train Success (last 1000)', fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if plotted > 0:
    ax.legend(loc='best', fontsize=9, framealpha=0.95)

  out_path = os.path.join(figs_dir, OUT_NAME)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}')
  return out_path


if __name__ == '__main__':
  plot_once()
