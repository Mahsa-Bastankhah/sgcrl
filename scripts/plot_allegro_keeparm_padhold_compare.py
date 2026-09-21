#!/usr/bin/env python3
"""Train/eval success for the keep-arm palm-up pad-hold runs.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5).

  python scripts/plot_allegro_keeparm_padhold_compare.py
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
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_keeparm_padhold_train_eval_success.png')

RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_200m_seed0_10h',
        'hide table · bucket (0.00,-0.22) · ep50  (3870715)',
        base.ACCENT_COLORS[0],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_200m_fallm4_nosettle_seed0_4h',
        'hide table · bucket · ep50 · fall −4 · no mid-reset settle  (3870338)',
        base.ACCENT_COLORS[4],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_200m_q10shift_seed0_7h',
        'hide table · bucket · ep50 · NF P10 shift (ema 0.1)  (3870369)',
        base.ACCENT_COLORS[2],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_500m_q10shift_seed0_14h',
        'hide table · bucket · ep50 · NF P10 shift · 500M  (3871188)',
        base.ACCENT_COLORS[8],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed0_14h',
        'hide table · bucket · ep50 · NF P5 shift · 500M  (3871639)',
        base.ACCENT_COLORS[4],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_200m_q20shift_ema001_seed0_10h',
        'hide table · bucket · ep50 · NF P20 shift (ema 0.01)  (3870724)',
        base.ACCENT_COLORS[5],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_200m_q20shift_ema001_extrew5_seed0_10h',
        'hide table · bucket · ep50 · NF P20 shift (ema 0.01) · extrew 5  (3871048)',
        base.ACCENT_COLORS[7],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep70_200m_q20shift_ema001_seed0_10h',
        'hide table · bucket · ep70 · NF P20 shift (ema 0.01)  (3871232)',
        base.ACCENT_COLORS[10],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep70_200m_q10shift_gamma097_seed0_10h',
        'hide table · bucket · ep70 · NF P10 shift · γ=0.97  (3871642)',
        '#5C4B7A',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep100_200m_q20shift_ema001_gamma097_seed0_10h',
        'hide table · bucket · ep100 · NF P20 · γ=0.97  (3871620)',
        base.ACCENT_COLORS[11],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_largetable15_objgoal_norand_ep50_200m_seed0_4h',
        'large table · on-desk objgoal · ep50  (3868985)',
        base.ACCENT_COLORS[1],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_largetable15_objgoal_norand_ep50_200m_extrew5_seed0_10h',
        'large table · on-desk objgoal · ep50 · extrew 5  (3870737)',
        base.ACCENT_COLORS[6],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_largetable15_objgoal_norand_ep50_200m_fallm4_nosettle_seed0_7h',
        'large table · objgoal · ep50 · fall −4 · no mid-reset settle  (3870332)',
        base.ACCENT_COLORS[3],
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_largetable15_objgoal_norand_ep50_500m_fallm4_nosettle_seed0_14h',
        'large table · objgoal · ep50 · fall −4 · 500M  (3871189)',
        base.ACCENT_COLORS[9],
    ),
)


def _style(ax, ylabel: str, *, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=10)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.set_ylim(-0.05, 1.05)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.8), sharex=True)
  n_ok = 0
  for log_dir, label, color in RUNS:
    train_seeds = base._read_train_seed_series(base.LOG_ROOT, log_dir)
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train_seeds)
    eval_seeds = base._read_eval_seed_series(base.LOG_ROOT, log_dir)
    ex, emean, _, en = base._aggregate_mean_stderr(eval_seeds)
    if tx:
      tx, tmean, _ = base._subsample_curve(tx, tmean, tse)
      peak_i = tmean.index(max(tmean))
      axes[0].plot(tx, tmean, color=color, lw=2.0, label=label)
      print(f'{label} train: n={tn} last={tmean[-1]:.4f} '
            f'peak={tmean[peak_i]:.4g} @ {tx[peak_i]/1e6:.1f}M '
            f'steps={tx[-1]/1e6:.1f}M')
      n_ok += 1
    else:
      print(f'{label} train: no data')
    if ex:
      sm = base._plot_eval_smoothed(
          axes[1], ex, emean, color=color,
          label=f'{label}  (roll mean w={base.EVAL_SMOOTH_WINDOW})')
      ys = sm if sm else emean
      peak_i = ys.index(max(ys))
      print(f'{label} eval: n={en} last={ys[-1]:.4f} '
            f'peak={ys[peak_i]:.4g} @ {ex[peak_i]/1e6:.1f}M')
    else:
      print(f'{label} eval: no data')

  axes[0].set_title(
      'keep-arm palm-up pad-hold — train (raw last-1000)',
      fontsize=12, fontweight='bold')
  _style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper right', fontsize=7.0, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=12, fontweight='bold')
  _style(axes[1], f'eval success (roll mean, w={base.EVAL_SMOOTH_WINDOW})',
         xlabel=True)
  axes[1].legend(loc='upper right', fontsize=7.0, framealpha=0.95)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=160, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'plotted {n_ok}/{len(RUNS)}  → {OUT_PATH}')


if __name__ == '__main__':
  main()
