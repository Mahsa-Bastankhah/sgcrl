#!/usr/bin/env python3
"""Dual-panel train/eval success for the c8t2 NF compact catwp run.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5). Shades the 2cm success lock (iters 1680–1872).

  python scripts/plot_c8t2_nf_success.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256'
    '_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10'
)
RUN = 'ppo_builderbench_creative_8_task2_0'
LEARNER_CSV = os.path.join(
    REPO, 'logs', LOG_DIR, RUN, 'logs', 'learner', 'logs.csv')
EVAL_CSV = os.path.join(
    REPO, 'logs', LOG_DIR, RUN, 'logs', 'eval', 'logs.csv')
OUT_PATH = os.path.join(REPO, 'figs', 'builderbench', 'c8t2_nf_success.png')

LOCK_LO, LOCK_HI = 1680, 1872  # inclusive; collapse starts 1873
SPI = base.DEFAULT_STEPS_PER_ITER
COLOR = '#4C9BE8'
LOCK_COLOR = '#2A9D8F'


def _load_xy(path: str, x_col: str, y_col: str) -> tuple[list[int], list[float]]:
  xs: list[int] = []
  ys: list[float] = []
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      x = base._coerce(row.get(x_col))
      y = base._coerce(row.get(y_col))
      if x is None or y is None:
        continue
      xs.append(int(x))
      ys.append(float(y))
  return xs, ys


def _shade_lock(ax):
  ax.axvspan(
      LOCK_LO * SPI, (LOCK_HI + 0.5) * SPI,
      color=LOCK_COLOR, alpha=0.12, lw=0, zorder=0,
  )


def main() -> None:
  train_iters, train_ys = _load_xy(
      LEARNER_CSV, 'iteration', 'train_success_1000')
  eval_iters, eval_ys = _load_xy(EVAL_CSV, 'iteration', 'success')
  if not train_iters and not eval_iters:
    raise SystemExit(f'no success data under {LOG_DIR}/{RUN}')

  train_xs = [it * SPI for it in train_iters]
  eval_xs = [it * SPI for it in eval_iters]

  fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)

  # --- train (raw) ---
  ax = axes[0]
  _shade_lock(ax)
  if train_xs:
    ax.plot(
        train_xs, train_ys, color=COLOR, linewidth=2.0, alpha=0.95,
        label='train success_1000', zorder=3,
    )
    peak_i = max(range(len(train_ys)), key=lambda i: train_ys[i])
    print(
        f'train: {len(train_xs)} pts  '
        f'peak={train_ys[peak_i]:.3f} @ iter {train_iters[peak_i]}  '
        f'last={train_ys[-1]:.3f} @ iter {train_iters[-1]}'
    )
  ax.set_title('c8t2 NF compact catwp — train success (last 1000)',
               fontsize=11, fontweight='bold')
  ax.set_ylabel('Train Success (last 1000)', fontsize=10)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

  # --- eval (smoothed) ---
  ax = axes[1]
  _shade_lock(ax)
  if eval_xs:
    sm = base._plot_eval_smoothed(
        ax, eval_xs, eval_ys, color=COLOR,
        label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})',
        linewidth=2.0,
    )
    lock_evals = [
        (it, y) for it, y in zip(eval_iters, eval_ys)
        if LOCK_LO <= it <= LOCK_HI
    ]
    print(
        f'eval:  {len(eval_xs)} pts  '
        f'raw=[{min(eval_ys):.3f}..{max(eval_ys):.3f}]  '
        f'smooth=[{min(sm):.3f}..{max(sm):.3f}]  '
        f'lock iters with success>0: '
        f'{[it for it, y in lock_evals if y > 0]}'
    )
  ax.set_title(
      f'c8t2 NF compact catwp — eval success '
      f'(roll mean w={base.EVAL_SMOOTH_WINDOW})',
      fontsize=11, fontweight='bold',
  )
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.set_ylabel(
      f'Eval Success (roll mean w={base.EVAL_SMOOTH_WINDOW})', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  lock_patch = Patch(
      facecolor=LOCK_COLOR, alpha=0.25, edgecolor='none',
      label=f'lock {LOCK_LO}–{LOCK_HI}',
  )
  handles, labels = ax.get_legend_handles_labels()
  ax.legend(
      handles + [lock_patch], labels + [f'lock {LOCK_LO}–{LOCK_HI}'],
      loc='upper left', fontsize=9, framealpha=0.95,
  )

  fig.suptitle(
      'BuilderBench Creative 8 Task 2 [NF compact · catwp]',
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
