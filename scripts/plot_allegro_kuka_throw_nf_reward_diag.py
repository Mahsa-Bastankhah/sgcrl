#!/usr/bin/env python3
"""AllegroKukaThrow 300M NF run: log p, raw PPO NF reward, return-norm std, NLL.

Main panel: log mean / max p, raw PPO NF reward, NLL.
``|log min p|`` (log y-scale) and return-norm std each get their own panel.
Train is not smoothed.

  python scripts/plot_allegro_kuka_throw_nf_reward_diag.py
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
    'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
    '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
)
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_nfcsm_300m_nf_reward_diag.png')

MAIN_SERIES = (
    ('nf/log_p_mean', 'log mean p', '#264653', 1.8, 0.95),
    ('nf/log_p_max', 'log max p', '#4C9BE8', 1.4, 0.95),
    ('reward_repr_raw_mean', 'PPO NF reward (raw)', '#d1495b', 1.8, 0.95),
    ('nf/density_loss', 'NLL (density_loss)', '#A84CE8', 1.5, 0.95),
)
MIN_SERIES = ('nf/log_p_min', 'log min p', '#2A9D8F', 1.4, 0.95)
STD_SERIES = (
    'reward_return_norm_std', 'return-norm std', '#E8834C', 1.6, 0.95)


def _load(col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, LOG_DIR, split='learner',
      x_col='global_step', y_col=col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    raise SystemExit(f'no data for {col}')
  xs, mean, _ = base._subsample_curve(xs, mean, se)
  return xs, mean, n


def _style(ax, ylabel: str, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.axhline(0.0, color='0.55', lw=0.7)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)


def main() -> None:
  fig, axes = plt.subplots(
      3, 1, figsize=(10.8, 9.2), sharex=True,
      gridspec_kw={'height_ratios': [1.35, 1.0, 0.9], 'hspace': 0.18})

  ax = axes[0]
  for col, label, color, lw, alpha in MAIN_SERIES:
    xs, mean, n = _load(col)
    ax.plot(xs, mean, color=color, linewidth=lw, alpha=alpha, label=label)
    print(f'{label}: n={n} last={mean[-1]:.4f} steps={xs[-1]:.0f}')
  ax.set_title(
      'AllegroKukaThrow 300M — log mean/max p, raw PPO NF reward, NLL',
      fontsize=11, fontweight='bold')
  _style(ax, 'value')
  ax.legend(loc='best', fontsize=8.5, framealpha=0.95, ncol=2)

  ax = axes[1]
  col, label, color, lw, alpha = MIN_SERIES
  xs, mean, n = _load(col)
  # log_p_min is always negative; log y-axis needs |log min p|.
  mag = [-y if y == y else y for y in mean]
  ax.plot(xs, mag, color=color, linewidth=lw, alpha=alpha, label='|log min p|')
  ax.set_yscale('log')
  print(f'{label}: n={n} last={mean[-1]:.4f} min={min(mean):.1f}')
  ax.set_title('|log min p|  (log scale)', fontsize=11, fontweight='bold')
  ax.set_ylabel('|log min p|', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')
  ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)

  ax = axes[2]
  col, label, color, lw, alpha = STD_SERIES
  xs, mean, n = _load(col)
  ax.plot(xs, mean, color=color, linewidth=lw, alpha=alpha, label=label)
  print(f'{label}: n={n} last={mean[-1]:.4f}')
  ax.set_title(
      'return-norm std  (used to normalize reward)',
      fontsize=11, fontweight='bold')
  _style(ax, r'$\sigma_R$', xlabel=True)
  ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

  fig.suptitle(
      'NF compact-small sa3x192 · τ=0.5 · NF reward only · job 3749904',
      fontsize=10, fontweight='bold', y=0.995)
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout(rect=[0, 0, 1, 0.97])
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
