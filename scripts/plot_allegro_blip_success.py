#!/usr/bin/env python3
"""Train success for Allegro runs that logged a 0.001 blip.

Eval is off. Train is raw ``train_success_1000`` (no smoothing) with y
zoomed so a single 1/1000-episode tick is visible.

  python scripts/plot_allegro_blip_success.py
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
    REPO, 'figs', 'allegro_kuka_throw', 'akt_blip_train_success.png')
C = base.ACCENT_COLORS

PALM = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm022_stateonly'
        '_actorreset_4h',
        'palm 0.22 + AR',
        C[0],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm022_stateonly'
        '_mixtaskg_actorreset_4h',
        'palm 0.22 mix + AR',
        C[1],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palm025_stateonly'
        '_mixtaskg_actorreset_4h',
        'palm 0.25 mix + AR',
        C[3],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_stateonly'
        '_actorreset_4h',
        'palm 0.17 + AR',
        C[4],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_stateonly'
        '_norand_mixtaskg_4h',
        'palm 0.17 norand mix',
        C[5],
        '--',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_palmgoal_stateonly_2h',
        'palm 0.17 ent anneal',
        C[6],
        '-',
    ),
)

TABLESIDE = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_tableside_objxyrand_noshape_mixtaskg_mix75_4h',
        'tableside mix75',
        C[2],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_tableside_objxyrand_noshape_mixtaskg_mix75_actorreset_4h',
        'tableside mix75 + AR',
        C[7],
        '-',
    ),
)


def _curve(log_dir: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, log_dir, split='learner',
      x_col='global_step', y_col='train_success_1000')
  xs, mean, _se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return None
  return xs, mean, n


def _plot_panel(ax, runs, title: str) -> None:
  any_pts = False
  for log_dir, label, color, ls in runs:
    got = _curve(log_dir)
    if got is None:
      print(f'{label}: no data')
      continue
    xs, mean, n = got
    peak = max(mean)
    peak_i = mean.index(peak)
    hits = [(x, y) for x, y in zip(xs, mean) if y > 0]
    ax.plot(xs, mean, color=color, linestyle=ls, lw=1.5, alpha=0.9,
            label=label)
    if hits:
      hx, hy = zip(*hits)
      ax.scatter(hx, hy, color=color, s=28, zorder=3, edgecolors='white',
                 linewidths=0.4)
    any_pts = True
    hit_s = ', '.join(f'{x/1e6:.1f}M' for x, _y in hits) or 'none'
    print(f'{label}: n={n} last={mean[-1]:.4f} peak={peak:.4g} '
          f'at {xs[peak_i]/1e6:.1f}M  hits={hit_s}')
  if not any_pts:
    return
  ax.axhline(0.001, color='0.75', lw=0.6, linestyle=':')
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_ylabel('train_success_1000', fontsize=10)
  ax.set_ylim(-0.00012, 0.00155)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=2)


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.2), sharex=True)
  _plot_panel(
      axes[0], PALM,
      r'palm-goal  (compact sa3$\times$256 · $p(g|s)$ · bucket 0.37,0.08,0.312)')
  _plot_panel(
      axes[1], TABLESIDE,
      r'tableside push  (compact-small sa3$\times$192 · $p(g|s,a)$ · mix75)')
  axes[1].set_xlabel('Env Steps', fontsize=10)
  fig.suptitle(
      'Allegro throw — train success on runs with a 0.001 blip  '
      '(1 / 1000 eps; raw, not smoothed)',
      fontsize=11, fontweight='bold', y=1.01,
  )
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
