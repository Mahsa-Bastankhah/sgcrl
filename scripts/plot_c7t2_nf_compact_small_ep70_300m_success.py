#!/usr/bin/env python3
"""Dual-panel train/eval success for c7t2 NF compact-small catwp ep70 300m.

The smaller-net analog of the current best-method NF pick (sa3x256). Seed 0
only; rose then collapsed.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5).

  python scripts/plot_c7t2_nf_compact_small_ep70_300m_success.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small_sa3x192_r64_b6_'
    'w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep70_300m'
)
RUN = 'ppo_builderbench_creative_7_task2_0'
LEARNER_CSV = os.path.join(
    REPO, 'logs', LOG_DIR, RUN, 'logs', 'learner', 'logs.csv')
EVAL_CSV = os.path.join(
    REPO, 'logs', LOG_DIR, RUN, 'logs', 'eval', 'logs.csv')
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench',
    'c7t2_nf_compact_small_ep70_300m_success.png')

# 1024 envs × rollout 70 (do not use DEFAULT_STEPS_PER_ITER=50×1024).
STEPS_PER_ITER = 1024 * 70
COLOR = '#4C9BE8'


def _load_xy(path: str, x_col: str, y_col: str) -> tuple[list[float], list[float]]:
  xs: list[float] = []
  ys: list[float] = []
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      x = base._coerce(row.get(x_col))
      y = base._coerce(row.get(y_col))
      if x is None or y is None:
        continue
      xs.append(float(x))
      ys.append(float(y))
  return xs, ys


def main() -> None:
  train_steps, train_ys = _load_xy(
      LEARNER_CSV, 'global_step', 'train_success_1000')
  eval_iters, eval_ys = _load_xy(EVAL_CSV, 'iteration', 'success')
  if not train_steps and not eval_iters:
    raise SystemExit(f'no success data under {LOG_DIR}/{RUN}')

  eval_xs = [it * STEPS_PER_ITER for it in eval_iters]

  fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)

  ax = axes[0]
  if train_steps:
    ax.plot(
        train_steps, train_ys, color=COLOR, linewidth=2.0, alpha=0.95,
        label='train success_1000', zorder=3,
    )
    peak_i = max(range(len(train_ys)), key=lambda i: train_ys[i])
    print(
        f'train: {len(train_steps)} pts  '
        f'peak={train_ys[peak_i]:.3f} @ {train_steps[peak_i]:.0f} steps  '
        f'last={train_ys[-1]:.3f} @ {train_steps[-1]:.0f} steps'
    )
  ax.set_title(
      'c7t2 NF compact-small ep70 300m — train success (last 1000)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('Train Success (last 1000)', fontsize=10)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

  ax = axes[1]
  if eval_xs:
    sm = base._plot_eval_smoothed(
        ax, eval_xs, eval_ys, color=COLOR,
        label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})',
        linewidth=2.0,
    )
    print(
        f'eval:  {len(eval_xs)} pts  '
        f'raw=[{min(eval_ys):.3f}..{max(eval_ys):.3f}] last={eval_ys[-1]:.3f}  '
        f'smooth=[{min(sm):.3f}..{max(sm):.3f}] last={sm[-1]:.3f}'
    )
  ax.set_title(
      f'c7t2 NF compact-small ep70 300m — eval success '
      f'(roll mean w={base.EVAL_SMOOTH_WINDOW})',
      fontsize=11, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.set_ylabel(
      f'Eval Success (roll mean w={base.EVAL_SMOOTH_WINDOW})', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

  fig.suptitle(
      'BuilderBench Creative 7 Task 2  [NF compact-small · catwp · ep70 300m]',
      fontsize=12, fontweight='bold', y=1.01,
  )
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
