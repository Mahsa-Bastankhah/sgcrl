#!/usr/bin/env python3
"""Overlay train/eval success for Sawyer-peg tiny-NF mixtaskg ablations.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5) of per-checkpoint ``success``.

  python scripts/plot_peg_nf_tiny_mixtaskg_success.py
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
# num_envs=4 × rollout_length=256 (not BuilderBench's 50×1024).
STEPS_PER_ITER = 1024
OUT_PATH = os.path.join(
    REPO, 'figs', 'metaworld', 'peg_nf_tiny_mixtaskg_train_eval_success.png')

_LOG_PREFIX = (
    'ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_rand'
    '_minstd1e5_ent0005_mixtaskg'
)
# (legend, log_dir, seed run dir, color)
RUNS = (
    ('mixtaskg s0', _LOG_PREFIX, 'ppo_sawyer_peg_0', '#4C9BE8'),
    ('mixtaskg s1', _LOG_PREFIX, 'ppo_sawyer_peg_1', '#9B59B6'),
    ('mask10', f'{_LOG_PREFIX}_mask10', 'ppo_sawyer_peg_0', '#2A9D8F'),
    ('grdual', f'{_LOG_PREFIX}_grdual', 'ppo_sawyer_peg_0', '#E07A3D'),
    ('retnormW1e6', f'{_LOG_PREFIX}_retnormW1e6', 'ppo_sawyer_peg_0', '#C44E52'),
)


def _csv_path(log_dir: str, run: str, kind: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, run, 'logs', kind, 'logs.csv')


def _load_xy(path: str, x_col: str, y_col: str) -> tuple[list[float], list[float]]:
  xs: list[float] = []
  ys: list[float] = []
  if not os.path.isfile(path):
    return xs, ys
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
  fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True)
  ax_tr, ax_ev = axes

  for label, log_dir, run, color in RUNS:
    train_steps, train_ys = _load_xy(
        _csv_path(log_dir, run, 'learner'), 'global_step', 'train_success_1000')
    eval_iters, eval_ys = _load_xy(
        _csv_path(log_dir, run, 'eval'), 'iteration', 'success')
    eval_xs = [it * STEPS_PER_ITER for it in eval_iters]

    if train_steps:
      ax_tr.plot(
          train_steps, train_ys, color=color, linewidth=1.6, alpha=0.95,
          label=label, zorder=3)
      peak_i = max(range(len(train_ys)), key=lambda i: train_ys[i])
      print(
          f'{label} train: {len(train_steps)} pts  '
          f'peak={train_ys[peak_i]:.3f} @ {train_steps[peak_i]:.0f} steps  '
          f'last={train_ys[-1]:.3f} @ {train_steps[-1]:.0f} steps')
    else:
      print(f'{label} train: no data under {log_dir}/{run}')

    if eval_xs:
      sm = base._plot_eval_smoothed(
          ax_ev, eval_xs, eval_ys, color=color, label=label, linewidth=2.0)
      print(
          f'{label} eval:  {len(eval_xs)} pts  '
          f'raw=[{min(eval_ys):.3f}..{max(eval_ys):.3f}] last={eval_ys[-1]:.3f}  '
          f'smooth=[{min(sm):.3f}..{max(sm):.3f}] last={sm[-1]:.3f}')
    else:
      print(f'{label} eval: no data under {log_dir}/{run}')

  ax_tr.set_title('Sawyer peg · tiny NF — train success (last 1000)',
                  fontsize=11, fontweight='bold')
  ax_tr.set_ylabel('Train Success (last 1000)', fontsize=10)
  ax_tr.set_ylim(-0.05, 1.05)
  ax_tr.spines[['top', 'right']].set_visible(False)
  ax_tr.grid(axis='y', linestyle='--', alpha=0.4)
  ax_tr.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=3)

  ax_ev.set_title(
      f'Sawyer peg · tiny NF — eval success '
      f'(roll mean w={base.EVAL_SMOOTH_WINDOW})',
      fontsize=11, fontweight='bold')
  ax_ev.set_xlabel('Env Steps', fontsize=10)
  ax_ev.set_ylabel(
      f'Eval Success (roll mean w={base.EVAL_SMOOTH_WINDOW})', fontsize=10)
  ax_ev.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax_ev.set_ylim(-0.05, 1.05)
  ax_ev.spines[['top', 'right']].set_visible(False)
  ax_ev.grid(axis='y', linestyle='--', alpha=0.4)
  ax_ev.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=3)

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
