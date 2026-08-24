#!/usr/bin/env python3
"""c5t2 dualgradreg horizon ablation (ent=0.05 fixed): ep70/T50 vs ep70/T70.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5). X-axis is env steps (learner ``global_step``), so T=50 and
T=70 line up.

  python scripts/plot_c5t2_ep70_horizon_success.py
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
PREFIX = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect'
    '_dualgradreg_c100_lamlr1e6_ent05'
)
RUNS = (
    (PREFIX + '_ep70_T50', 'ep=70, T=50', base.ACCENT_COLORS[1], '-'),
    (PREFIX + '_ep70', 'ep=70, T=70', base.ACCENT_COLORS[2], '-'),
)
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c5t2', 'c5t2_ep70_horizon_success.png')


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.8), sharex=True)

  for i, (name, label, color, ls) in enumerate(RUNS):
    t_seeds = base._read_train_seed_series(base.LOG_ROOT, name)
    e_seeds = base._read_eval_seed_series(base.LOG_ROOT, name)
    txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
    exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)

    if txs:
      xs, mean, _se = base._subsample_curve(txs, tmean, tse)
      axes[0].plot(
          xs, mean, color=color, linestyle=ls, linewidth=2.0, alpha=0.95,
          label=label, zorder=3 + i)
    if exs:
      sm = base._plot_eval_smoothed(
          axes[1], exs, emean, color=color, label=label, linestyle=ls,
          zorder=3 + i)
      print(
          f'{label}: n={max(n_t, n_e)}  '
          f'train last={tmean[-1]:.3f} peak={max(tmean):.3f}  '
          f'eval raw last={emean[-1]:.3f} peak={max(emean):.3f}  '
          f'smooth last={sm[-1]:.3f} peak={max(sm):.3f}'
      )
    else:
      print(f'{label}: n={n_t}  no eval points')

  ax = axes[0]
  ax.set_title(
      'c5t2 NF compact-small dualgradreg — train success (last 1000)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('Train Success (last 1000)', fontsize=10)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

  ax = axes[1]
  ax.set_title(
      f'c5t2 NF compact-small dualgradreg — eval success '
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
      'BuilderBench Creative 5 Task 2  [pyramid horizon · ent=0.05 fixed]',
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
