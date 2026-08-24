#!/usr/bin/env python3
"""c5t2 NF dualgradreg ablations: λ_lr, entropy, ep70/T50, tiny.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5). X-axis is env steps. Catwp and ep70/T70 dropped.

  python scripts/plot_c5t2_dualgradreg_ablations_success.py
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
CS = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01'
)
TINY = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_tiny'
    '_sa2x128_r32_b4_w128_tau05_nopermute_fixedx01'
)
C = base.ACCENT_COLORS
RUNS = (
    (f'{CS}_catselect_dualgradreg_c100_lamlr1e2_ent05_s2',
     r'cs · catselect · $\lambda$lr=1e-2 · ent=0.05 · ep50', C[0], '-'),
    (f'{CS}_catselect_dualgradreg_c100_lamlr1e2_ent05to001_s2',
     r'cs · catselect · $\lambda$lr=1e-2 · ent 0.05→0.01 · ep50', C[1], '-'),
    (f'{CS}_catselect_dualgradreg_c100_lamlr1e6_ent05_s2',
     r'cs · catselect · $\lambda$lr=1e-6 · ent=0.05 · ep50', C[2], '-'),
    (f'{CS}_catselect_dualgradreg_c100_lamlr1e6_ent05_ep70_T50',
     r'cs · catselect · $\lambda$lr=1e-6 · ent=0.05 · ep70/T50', C[4], '-'),
    (f'{TINY}_catselect_dualgradreg_c100_lamlr1e6_ent05',
     r'tiny · catselect · $\lambda$lr=1e-6 · ent=0.05 · ep50', C[6], '-'),
)
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c5t2',
    'c5t2_dualgradreg_ablations_success.png')


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(11.4, 7.0), sharex=True)

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
    elif txs:
      print(
          f'{label}: n={n_t}  '
          f'train last={tmean[-1]:.3f} peak={max(tmean):.3f}  no eval'
      )
    else:
      print(f'{label}: no data')

  ax = axes[0]
  ax.set_title(
      'c5t2 NF dualgradreg — train success (last 1000)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('Train Success (last 1000)', fontsize=10)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

  ax = axes[1]
  ax.set_title(
      f'c5t2 NF dualgradreg — eval success '
      f'(roll mean w={base.EVAL_SMOOTH_WINDOW})',
      fontsize=11, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.set_ylabel(
      f'Eval Success (roll mean w={base.EVAL_SMOOTH_WINDOW})', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5),
      fontsize=8.5, framealpha=0.95, title='ablation', title_fontsize=9)
  fig.suptitle(
      'BuilderBench Creative 5 Task 2  '
      '[NF dualgradreg c=100 · nopermute + fixedx]',
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
