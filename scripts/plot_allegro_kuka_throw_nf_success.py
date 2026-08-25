#!/usr/bin/env python3
"""AllegroKukaThrow NF compact-small train success.

Eval is off (``ppo_eval_interval=0``). Train is raw ``train_success_1000``
(object within env success_tolerance of the fixed bucket).

  python scripts/plot_allegro_kuka_throw_nf_success.py
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
RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001',
        '300M · ent 0.05→0.01 · job 3749904',
        base.ACCENT_COLORS[0],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_200m_crl10_ent05to001_smoke30m',
        'smoke 30min · same recipe',
        base.ACCENT_COLORS[1],
        '--',
    ),
)
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_nfcsm_ent05to001_train_success.png')


def main() -> None:
  fig, ax = plt.subplots(figsize=(10.4, 4.8))
  any_pts = False
  for name, label, color, ls in RUNS:
    seeds = base._read_csv_seed_series(
        base.LOG_ROOT, name, split='learner',
        x_col='global_step', y_col='train_success_1000')
    xs, mean, se, n = base._aggregate_mean_stderr(seeds)
    if not xs:
      print(f'{label}: no data')
      continue
    xs, mean, _ = base._subsample_curve(xs, mean, se)
    ax.plot(
        xs, mean, color=color, linestyle=ls, linewidth=2.0, alpha=0.95,
        label=label)
    any_pts = True
    print(f'{label}: n={n} last={mean[-1]:.4f} peak={max(mean):.4f} '
          f'steps={xs[-1]:.0f}')

  if not any_pts:
    raise SystemExit('no train_success_1000 data')

  ax.set_title(
      'AllegroKukaThrow — train success (last 1000)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('Train Success (last 1000)', fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
  fig.suptitle(
      'NF compact-small sa3x192 · τ=0.5 · minstd 1e-5 · ent 0.05→0.01 · '
      'ep300 · NF reward only  (eval_interval=0)',
      fontsize=11, fontweight='bold', y=1.02,
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
