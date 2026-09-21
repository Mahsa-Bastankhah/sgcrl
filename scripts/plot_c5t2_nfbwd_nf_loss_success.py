#!/usr/bin/env python3
"""c5t2 nfbwd seed0: NF NLL vs train/eval success.

Train success is raw ``train_success_1000``. Eval hard success is a centered
rolling mean (window=5) with faint raw underneath. NF loss is raw
``nf/density_loss``.

  python scripts/plot_c5t2_nfbwd_nf_loss_success.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect_extrew1'
    '_minstd1e5_ent05to001_ep70_T50_100m_crl10_nfbwd'
)
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c5t2',
    'c5t2_nfbwd_nf_loss_success.png')
C_TRAIN = '#2A9D8F'
C_EVAL = '#4C9BE8'
C_NLL = '#9B2226'
# Same ckpts as the stochastic pre/post binary-acc probe.
PROBE_ITERS = (300, 400, 1000, 1400)


def _curve(split: str, x_col: str, y_col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, LOG_DIR, split=split, x_col=x_col, y_col=y_col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  return xs, mean, n


def _first_above(xs, ys, thresh: float):
  for x, y in zip(xs, ys):
    if y == y and y >= thresh:
      return int(x)
  return None


def main() -> None:
  xs_tr, ys_tr, n_tr = _curve('learner', 'iteration', 'train_success_1000')
  xs_nll, ys_nll, n_nll = _curve('learner', 'iteration', 'nf/density_loss')
  ev_xs, ev_ys, n_ev = _curve('eval', 'iteration', 'success')
  t_takeoff = _first_above(xs_tr, ys_tr, 0.5)

  print(f'train_success_1000: n={n_tr} takeoff>=0.5 iter={t_takeoff}')
  print(f'nf/density_loss: n={n_nll} last={ys_nll[-1] if ys_nll else float("nan"):.3f}')
  print(f'eval success: n={n_ev}')

  fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.0), sharex=True)
  ax_s, ax_l = axes

  if xs_tr:
    ax_s.plot(
        xs_tr, ys_tr, color=C_TRAIN, lw=1.4, alpha=0.95,
        label='train success (2cm, raw)', zorder=3)
  if ev_xs:
    base._plot_eval_smoothed(
        ax_s, ev_xs, ev_ys, color=C_EVAL,
        label=f'eval hard (roll mean w={base.EVAL_SMOOTH_WINDOW})')
  if t_takeoff is not None:
    ax_s.axvline(t_takeoff, color=C_TRAIN, lw=1.1, ls=':', zorder=2)
    ax_l.axvline(t_takeoff, color=C_TRAIN, lw=1.1, ls=':', zorder=2)
    ax_s.annotate(
        f'success ≥0.5\niter {t_takeoff}',
        xy=(t_takeoff, 0.5), xytext=(12, 18),
        textcoords='offset points', fontsize=8, color=C_TRAIN)
  for it in PROBE_ITERS:
    ax_s.axvline(it, color='0.75', lw=0.8, ls='--', zorder=1)
    ax_l.axvline(it, color='0.75', lw=0.8, ls='--', zorder=1)
  ax_s.set_ylim(-0.05, 1.14)
  ax_s.set_ylabel('Success', fontsize=10)
  ax_s.set_title(
      'c5t2 nfbwd  ·  NF NLL and success (seed 0)',
      fontsize=11, fontweight='bold')
  ax_s.spines[['top', 'right']].set_visible(False)
  ax_s.grid(axis='y', linestyle='--', alpha=0.4)
  ax_s.legend(loc='lower right', fontsize=8, framealpha=0.95)

  if xs_nll:
    ax_l.plot(
        xs_nll, ys_nll, color=C_NLL, lw=1.4, alpha=0.95,
        label=r'NF NLL  ($-\log p$)', zorder=3)
  ax_l.set_ylabel('NF density loss', fontsize=10)
  ax_l.set_xlabel('Iteration', fontsize=10)
  ax_l.xaxis.set_major_locator(mticker.MultipleLocator(200))
  ax_l.spines[['top', 'right']].set_visible(False)
  ax_l.grid(axis='y', linestyle='--', alpha=0.4)
  ax_l.legend(loc='upper right', fontsize=8, framealpha=0.95)

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
