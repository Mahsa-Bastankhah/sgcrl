#!/usr/bin/env python3
"""Train success for recent Allegro tableside (on-desk slide) runs.

Eval is off. Train is raw ``train_success_1000`` (no smoothing).

  python scripts/plot_allegro_tableside_initrand_success.py
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
    REPO, 'figs', 'allegro_kuka_throw', 'akt_tableside_initrand_train_success.png')
C = base.ACCENT_COLORS
P = (
    'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
    '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_'
)

RUNS = (
    (
        P + 'tableside_initrand_noshape_mixtaskg_mix75_actorreset'
        '_palmgoal_extrew1_4h',
        'initrand + 6D palm + mix75 + AR + extrew1',
        C[0],
        '-',
        2.2,
    ),
    (
        P + 'tableside_initrand_noshape_mixtaskg_mix75_actorreset_extrew1_4h',
        'initrand + 3D + mix75 + AR + extrew1',
        C[1],
        '-',
        1.5,
    ),
    (
        P + 'tableside_initrand_noshape_extrew1_4h',
        'initrand + 3D + no mix / no AR + extrew1',
        C[3],
        '--',
        1.5,
    ),
    (
        P + 'tableside_objxyrand_noshape_mixtaskg_mix75_actorreset'
        '_palmgoal_extrew1_4h',
        'obj-xy + 6D palm + mix75 + AR + extrew1',
        C[4],
        '-',
        1.5,
    ),
    (
        P + 'tableside_objxyrand_noshape_mixtaskg_mix75_actorreset_extrew1_4h',
        'obj-xy + 3D + mix75 + AR + extrew1',
        C[2],
        ':',
        1.3,
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


def main() -> None:
  fig, ax = plt.subplots(figsize=(10.8, 4.6))
  ymax = 0.0024
  for log_dir, label, color, ls, lw in RUNS:
    got = _curve(log_dir)
    if got is None:
      print(f'{label}: no data')
      continue
    xs, mean, n = got
    peak = max(mean)
    peak_i = mean.index(peak)
    hits = [(x, y) for x, y in zip(xs, mean) if y > 0]
    ax.plot(xs, mean, color=color, linestyle=ls, lw=lw, alpha=0.92,
            label=label, zorder=3 if lw > 2 else 2)
    if hits:
      hx, hy = zip(*hits)
      ax.scatter(hx, hy, color=color, s=22, zorder=4, edgecolors='white',
                 linewidths=0.35)
    hit_s = ', '.join(f'{x/1e6:.0f}M' for x, _y in hits[:8]) or 'none'
    more = '' if len(hits) <= 8 else f' +{len(hits)-8}'
    print(f'{label}: n={n} last={mean[-1]:.4f} peak={peak:.4g} '
          f'at {xs[peak_i]/1e6:.1f}M  npos={len(hits)}  hits={hit_s}{more}')
    ymax = max(ymax, peak * 1.18 if peak > 0 else ymax)

  ax.axhline(0.001, color='0.75', lw=0.6, linestyle=':')
  ax.set_title(
      'Allegro tableside — train_success_1000  (raw, not smoothed)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('train_success_1000', fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.set_ylim(-0.00015, ymax)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
