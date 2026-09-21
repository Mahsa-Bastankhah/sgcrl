#!/usr/bin/env python3
"""Success for PPO+RND on the hover init that had NF signal (job 3861665).

RND writes eval_metrics.csv (no BraX train_success_1000). Overlay the
completed hover-only NF run. Eval: faint raw + bold rolling mean (window=5).

  python scripts/plot_allegro_rnd_hover_success.py
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_rnd_ab12_c15_success.png')

RND = (
    'ppo_rnd_allegro_kuka_throw_e1024_tableside_largetable15'
    '_above12_corr03_ep50_200m_objgoal_curl15_frand10_seed0_4h')
HOVER = (
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tableside_largetable15'
    '_above12_corr03_ep50_200m_objgoal_curl15_frand10_seed0_4h')

C_RND = base.ACCENT_COLORS[0]
C_REF = '#9AA3AD'


def _read_rnd_eval(log_dir: str):
  path = os.path.join(base.LOG_ROOT, log_dir, 'eval_metrics.csv')
  if not os.path.isfile(path):
    return [], []
  xs, ys = [], []
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      try:
        x = float(row['env_steps'])
        y = float(row['eval/episode_success'])
      except (KeyError, TypeError, ValueError):
        continue
      if y != y:
        continue
      xs.append(x)
      ys.append(y)
  return xs, ys


def _style(ax, ylabel: str):
  ax.set_ylabel(ylabel, fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.grid(True, alpha=0.28)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)


def main() -> int:
  fig, ax = plt.subplots(1, 1, figsize=(10.4, 4.8))

  rx, ry = _read_rnd_eval(RND)
  if rx:
    base._plot_eval_smoothed(
        ax, rx, ry, color=C_RND,
        label=f'RND  eval/episode_success  (roll mean w={base.EVAL_SMOOTH_WINDOW})')
    print(f'RND eval: pts={len(rx)} last={ry[-1]:.4g} '
          f'peak={max(ry):.4g} at {rx[ry.index(max(ry))]/1e6:.1f}M '
          f'nonzero={sum(1 for v in ry if v > 0)}/{len(ry)}')
  else:
    print(f'RND eval: no {RND}/eval_metrics.csv yet')

  tx_seeds = base._read_csv_seed_series(
      base.LOG_ROOT, HOVER, split='learner',
      x_col='global_step', y_col='train_success_1000')
  tx, tmean, tse, tn = base._aggregate_mean_stderr(tx_seeds)
  if tx:
    tx, tmean, _ = base._subsample_curve(tx, tmean, tse)
    ax.plot(tx, tmean, color=C_REF, lw=1.6, ls='--',
            label='NF hover-only  train_success_1000  (ref, raw)')
    print(f'NF train: n={tn} pts={len(tx)} last={tmean[-1]:.4g} '
          f'peak={max(tmean):.4g}')

  href = base._read_eval_seed_series(base.LOG_ROOT, HOVER)
  hx, hmean, _, hn = base._aggregate_mean_stderr(href)
  if hx:
    base._plot_eval_smoothed(
        ax, hx, hmean, color=C_REF, linestyle=':',
        label='NF hover-only  eval  (ref)')
    print(f'NF eval: n={hn} pts={len(hx)} last={hmean[-1]:.4g} '
          f'peak={max(hmean):.4g}')

  xmax = 0.0
  for xs in (rx, tx, hx):
    if xs:
      xmax = max(xmax, xs[-1])
  if xmax > 0:
    ax.set_xlim(0, max(xmax * 1.08, 5e6))
  ax.set_title(
      'Allegro hover init  —  PPO+RND success   vs completed NF hover-only',
      fontsize=11, fontweight='bold')
  _style(ax, 'success')
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)

  now = datetime.now().strftime('%Y-%m-%d %H:%M')
  fig.suptitle(f'updated {now}', fontsize=9, color='0.35', y=1.02)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
