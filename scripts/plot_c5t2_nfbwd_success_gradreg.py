#!/usr/bin/env python3
"""c5t2 NF compact-small T=50 nfbwd (no gradreg in the loss): success vs ∇_s.

Train success is raw ``train_success_1000``. Eval hard success is a centered
rolling mean (window=5) with faint raw underneath. Grad traces are raw.

  python scripts/plot_c5t2_nfbwd_success_gradreg.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    'c5t2_nfbwd_success_gradreg.png')
GRAD_REG_C = 100.0
C_TRAIN = '#2A9D8F'
C_EVAL = '#4C9BE8'
C_GNORM = '#9B2226'
C_RAW = '#1B7A6E'
# Success-window probe ckpts (same as job_bb_probe_c5t2_nfbwd_succ_warp).
PROBE_ITERS = (300, 400, 1000, 1400, 1600, 1800, 1900)


def _curve(split: str, x_col: str, y_col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, LOG_DIR, split=split, x_col=x_col, y_col=y_col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  return xs, mean, n


def main() -> None:
  xs_tr, ys_tr, n_tr = _curve('learner', 'iteration', 'train_success_1000')
  xs_g, ys_g, n_g = _curve(
      'learner', 'iteration', 'nf/nf_logp_grad_s_norm_mean')
  xs_raw, ys_raw, n_raw = _curve(
      'learner', 'iteration', 'nf/nf_grad_reg_raw')
  ev_xs, ev_ys, n_ev = _curve('eval', 'iteration', 'success')

  print(f'train_success_1000: n={n_tr} last={ys_tr[-1] if ys_tr else float("nan"):.3f} '
        f'peak={max(ys_tr) if ys_tr else float("nan"):.3f}')
  print(f'eval success: n={n_ev} last={ev_ys[-1] if ev_ys else float("nan"):.3f}')
  print(f'||∇_s||: n={n_g} last={ys_g[-1] if ys_g else float("nan"):.1f} '
        f'peak={max(ys_g) if ys_g else float("nan"):.1f}')
  print(f'grad_reg_raw: n={n_raw} last={ys_raw[-1] if ys_raw else float("nan"):.1f} '
        f'peak={max(ys_raw) if ys_raw else float("nan"):.1f}')

  fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.0), sharex=True)
  ax_s, ax_g = axes

  if xs_tr:
    ax_s.plot(
        xs_tr, ys_tr, color=C_TRAIN, lw=1.4, alpha=0.95,
        label='train success (2cm, raw)', zorder=3)
  if ev_xs:
    base._plot_eval_smoothed(
        ax_s, ev_xs, ev_ys, color=C_EVAL,
        label=f'eval hard (roll mean w={base.EVAL_SMOOTH_WINDOW})')
  for it in PROBE_ITERS:
    ax_s.axvline(it, color='0.75', lw=0.8, ls='--', zorder=1)
    ax_g.axvline(it, color='0.75', lw=0.8, ls='--', zorder=1)
    ax_s.annotate(
        str(it), xy=(it, 1.0), xytext=(0, 6), textcoords='offset points',
        ha='center', va='bottom', fontsize=7, color='0.35')
  ax_s.set_ylim(-0.05, 1.14)
  ax_s.set_ylabel('Success', fontsize=10)
  ax_s.set_title(
      'c5t2 NF compact-small T=50 nfbwd  ·  no gradreg in the loss',
      fontsize=11, fontweight='bold')
  ax_s.spines[['top', 'right']].set_visible(False)
  ax_s.grid(axis='y', linestyle='--', alpha=0.4)
  ax_s.legend(loc='lower right', fontsize=8, framealpha=0.95)

  if xs_g:
    ax_g.plot(
        xs_g, ys_g, color=C_GNORM, lw=1.4, alpha=0.95,
        label=r'mean $\|\nabla_s \log p\|$', zorder=3)
  if xs_raw:
    ax_g.plot(
        xs_raw, ys_raw, color=C_RAW, lw=1.25, ls='--', alpha=0.9,
        label=r'hinge raw  $\mathrm{mean}(\max(\|\nabla_s\|-c,0))$',
        zorder=2)
  ax_g.axhline(
      GRAD_REG_C, color='0.35', lw=0.9, ls=':', zorder=0,
      label=f'$c={GRAD_REG_C:g}$ (not in loss)')
  ax_g.set_ylabel(r'$\|\nabla_s \log p\|$', fontsize=10)
  ax_g.set_xlabel('Iteration', fontsize=10)
  ax_g.xaxis.set_major_locator(mticker.MultipleLocator(200))
  ax_g.spines[['top', 'right']].set_visible(False)
  ax_g.grid(axis='y', linestyle='--', alpha=0.4)
  ax_g.legend(loc='upper left', fontsize=8, framealpha=0.95)

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
