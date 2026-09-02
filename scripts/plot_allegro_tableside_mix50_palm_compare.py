#!/usr/bin/env python3
"""Compare tableside mix50 palm-behind / palm-on-top / table-spawn touch.

The failed table-spawn hover run (3782118) is excluded — palm never
reached the cube.

Train: raw ``train_success_1000``.
Eval: faint raw + bold rolling mean (window=5).

  python scripts/plot_allegro_tableside_mix50_palm_compare.py
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
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_tableside_mix50_palm_compare.png')

RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_'
        'tableside_initrand_noshape_mixtaskg_mix50_actorreset_palmbehind_extrew1_4h',
        'palm-behind (0.14,-0.10,0.585)',
        0,
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_'
        'tableside_initrand_noshape_mixtaskg_mix50_actorreset_palmgoal_extrew1_4h',
        'palm-on-top (0.20,-0.15,0.620)',
        1,
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_'
        'tableside_tablespawn_grasp_noshape_mixtaskg_mix50_palmgoal_extrew1_4h',
        'table-spawn grasp (palm-down on cube, no AR)',
        2,
    ),
)


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True)

  for log_dir, label, ci in RUNS:
    color = base.ACCENT_COLORS[ci]
    train_seeds = base._read_train_seed_series(base.LOG_ROOT, log_dir)
    tx, tmean, _, tn = base._aggregate_mean_stderr(train_seeds)
    eval_seeds = base._read_eval_seed_series(base.LOG_ROOT, log_dir)
    ex, emean, _, en = base._aggregate_mean_stderr(eval_seeds)

    ax = axes[0]
    if tx:
      ax.plot(tx, tmean, color=color, lw=2.0, label=f'{label}  train')
      print(f'{label} train: n={tn} pts={len(tx)} last={tmean[-1]:.4f} '
            f'peak={max(tmean):.4g} at {tx[tmean.index(max(tmean))]/1e6:.1f}M')
    else:
      print(f'{label} train: no data')

    ax = axes[1]
    if ex:
      base._plot_eval_smoothed(
          ax, ex, emean, color=color,
          label=f'{label}  eval (roll mean w={base.EVAL_SMOOTH_WINDOW})')
      print(f'{label} eval: n={en} pts={len(ex)} last={emean[-1]:.4f} '
            f'peak={max(emean):.4g} at {ex[emean.index(max(emean))]/1e6:.1f}M')
    else:
      print(f'{label} eval: no data')

  axes[0].set_title('Allegro tableside mix50 — train (raw)',
                    fontsize=11, fontweight='bold')
  axes[0].set_ylabel('train_success_1000', fontsize=10)
  axes[0].legend(loc='upper left', fontsize=8, framealpha=0.95)
  axes[0].spines[['top', 'right']].set_visible(False)
  axes[0].grid(axis='y', linestyle='--', alpha=0.4)

  axes[1].set_title(
      f'eval success  (rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=11, fontweight='bold')
  axes[1].set_ylabel('eval success (roll mean, w=5)', fontsize=10)
  axes[1].set_xlabel('Env Steps', fontsize=10)
  axes[1].legend(loc='upper left', fontsize=8, framealpha=0.95)
  axes[1].spines[['top', 'right']].set_visible(False)
  axes[1].grid(axis='y', linestyle='--', alpha=0.4)
  axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
