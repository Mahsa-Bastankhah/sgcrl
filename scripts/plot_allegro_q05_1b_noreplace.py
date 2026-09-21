#!/usr/bin/env python3
"""q05 keep-arm 1B runs: T=50/70 noreplace vs leftover T=70 replace=10.

Train is raw ``train_success_1000``. Eval is faint raw + bold rolling mean
(window=5). For the replace=10 run also draw per-iter env-hit
(succ_envs / 1024), which is the un-duplicated train rate.

  python scripts/plot_allegro_q05_1b_noreplace.py
"""
from __future__ import annotations

import csv
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
    'akt_q05_1b_noreplace_train_eval_success.png')

METHODS = (
    (
        'T=50 noreplace',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_1b_q05shift_noreplace_seed0_14h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_1b_q05shift_noreplace_seed1_14h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_1b_q05shift_noreplace_seed2_14h',
        ),
        base.ACCENT_COLORS[2],
        '-',
    ),
    (
        'T=70 noreplace',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep70_1b_q05shift_noreplace_seed0_14h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep70_1b_q05shift_noreplace_seed1_14h',
        ),
        base.ACCENT_COLORS[0],
        '-',
    ),
    (
        'T=70 replace=10',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep70_1000m_q05shift_seed0_15h',
        ),
        base.ACCENT_COLORS[1],
        '--',
    ),
)


def _read_env_hit(log_dir_name: str):
  root = os.path.join(base.LOG_ROOT, log_dir_name)
  paths = []
  if os.path.isdir(root):
    for name in sorted(os.listdir(root)):
      cand = os.path.join(root, name, 'logs', 'learner', 'logs.csv')
      if os.path.isfile(cand):
        paths.append(cand)
  pts = []
  for path in paths:
    with open(path, newline='') as fh:
      for row in csv.DictReader(fh):
        x = base._coerce(row.get('global_step', ''))
        n = base._coerce(row.get('ppo/succ_replace_n_succ_envs', ''))
        if x is None or n is None:
          continue
        pts.append((int(x), float(n) / 1024.0))
  return pts


def _style(ax, ylabel: str, *, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=10)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.set_ylim(-0.02, 1.05)
  if xlabel:
    ax.set_xlabel('Env steps', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.6), sharex=True)
  n_ok = 0
  for label, log_dirs, color, ls in METHODS:
    train_seeds = []
    for log_dir in log_dirs:
      train_seeds.extend(base._read_train_seed_series(base.LOG_ROOT, log_dir))
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train_seeds)
    if tx:
      xs, mean, se = base._subsample_curve(tx, tmean, tse)
      axes[0].plot(xs, mean, color=color, lw=2.0, linestyle=ls,
                   label=f'{label}  train_1000 (n={tn})')
      if tn > 1 and any(s > 0 for s in se):
        axes[0].fill_between(
            xs, [m - s for m, s in zip(mean, se)],
            [m + s for m, s in zip(mean, se)],
            color=color, alpha=0.18, linewidth=0)
      print(f'{label} train: n={tn} last={mean[-1]:.4f} '
            f'steps={xs[-1]/1e6:.1f}M')
      n_ok += 1
    else:
      print(f'{label} train: no data')

    if 'replace=10' in label:
      hit = _read_env_hit(log_dirs[0])
      if hit:
        hx, hy = zip(*hit)
        axes[0].plot(
            hx, hy, color=color, lw=1.3, linestyle=':', alpha=0.9,
            label=f'{label}  env-hit (succ/1024)')

    eval_seeds = []
    for log_dir in log_dirs:
      eval_seeds.extend(base._read_eval_seed_series(base.LOG_ROOT, log_dir))
    ex, emean, ese, en = base._aggregate_mean_stderr(eval_seeds)
    if ex:
      base._plot_eval_smoothed(
          axes[1], ex, emean, color=color,
          label=f'{label}  (n={en}, roll mean w={base.EVAL_SMOOTH_WINDOW})',
          linestyle=ls)
      print(f'{label} eval: n={en} last={emean[-1]:.4f} '
            f'steps={ex[-1]/1e6:.1f}M')
    else:
      print(f'{label} eval: no data')

  _style(axes[0], 'train success (raw)')
  _style(axes[1], f'eval success (roll mean, w={base.EVAL_SMOOTH_WINDOW})',
         xlabel=True)
  axes[0].set_title(
      'q05 keeparm 1B  noreplace vs replace=10',
      fontsize=11, fontweight='bold')
  axes[0].legend(loc='upper left', fontsize=7.5, framealpha=0.95)
  axes[1].legend(loc='upper left', fontsize=7.5, framealpha=0.95)
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  fig.savefig(OUT_PATH, dpi=160)
  print('wrote', OUT_PATH, 'series=', n_ok)


if __name__ == '__main__':
  main()
