#!/usr/bin/env python3
"""Episode length + actor-reset marks for palm 0.17 + AR (ent 0.05).

Resets come from slurm ``[ppo] actor reset at iter=`` (last-layer), not
from every short-episode tick. This run never logged object xyz.

  python scripts/plot_allegro_palm17_ar_eplen_reset.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
    '_tau05_minstd1e5_ent05_ep300_300m_crl10_palmgoal_stateonly'
    '_actorreset_4h'
)
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_palm17_ar_eplen_reset.png')
RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')
RESET_EP_LEN = 200.0
NOMINAL_EP = 300.0
COLOR = base.ACCENT_COLORS[4]


def _curve():
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, LOG_DIR, split='learner',
      x_col='global_step', y_col='ep_length_mean')
  xs, mean, _se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    raise SystemExit('no ep_length_mean')
  return xs, mean, n


def _reset_events() -> list[tuple[int, int]]:
  mapping = base._iter_to_env_steps_map(base.LOG_ROOT, LOG_DIR)
  events: list[tuple[int, int]] = []
  for path in sorted(glob.glob(os.path.join(REPO, 'slurm', f'{LOG_DIR}_*.log'))):
    with open(path, encoding='utf-8', errors='replace') as fh:
      for line in fh:
        m = RESET_RE.search(line)
        if not m:
          continue
        it = int(m.group(1))
        if it in mapping:
          events.append((it, int(mapping[it])))
  return sorted(set(events))


def _y_at(xs, ys, x):
  best_i = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
  return ys[best_i]


def main() -> None:
  xs, mean, n = _curve()
  resets = _reset_events()
  print(f'eplen n={n} last={mean[-1]:.1f} min={min(mean):.1f} '
        f'steps={xs[-1]:.0f}  actor-resets={len(resets)}')

  fig, ax = plt.subplots(figsize=(10.8, 4.6))
  ax.plot(xs, mean, color=COLOR, lw=1.6, label='palm 0.17 + AR')
  ax.axhline(NOMINAL_EP, color='0.55', lw=0.8, linestyle='--',
             label='nominal 300')
  ax.axhline(RESET_EP_LEN, color='0.45', lw=1.0, linestyle=':',
             label='reset thresh 200')

  for it, step in resets:
    y = _y_at(xs, mean, step)
    ax.axvline(step, color=COLOR, lw=0.8, linestyle='--', alpha=0.45)
    ax.scatter([step], [y], color=COLOR, marker='x', s=64, linewidths=1.6,
               zorder=5)
    ax.annotate(
        f'reset iter {it}\n{step/1e6:.1f}M  eplen={y:.1f}',
        xy=(step, y), xytext=(12, -28 if y < 220 else 14),
        textcoords='offset points', fontsize=8, color=COLOR,
        arrowprops=dict(arrowstyle='-', color=COLOR, lw=0.7))
    print(f'  actor reset iter={it} step={step} ({step/1e6:.1f}M) eplen={y:.2f}')

  ax.set_title(
      'palm 0.17 + AR — episode length  '
      r'($\times$ = last-layer actor-reset)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('ep_length_mean', fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='lower left', fontsize=8.5, framealpha=0.95)
  ax.set_ylim(170, 310)

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
