#!/usr/bin/env python3
"""Zoom the palm-goal train-success blip (one 0.001 tick at ~152M).

  python scripts/plot_allegro_palmgoal_success_zoom.py
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
    'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
    '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
    '_palmgoal_stateonly_2h'
)
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_palmgoal_success_zoom.png')
COLOR = base.ACCENT_COLORS[3]
# Logged spike: iter 493, global_step 151756800, train_success_1000=0.001
SPIKE_STEPS = 151756800
CKPT_STEPS = (
    (400 * 1024 * 300, 'ckpt 400'),
    (500 * 1024 * 300, 'ckpt 500'),
)


def _load(col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, LOG_DIR, split='learner',
      x_col='global_step', y_col=col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    raise SystemExit(f'no {col}')
  return xs, mean, n


def main() -> None:
  xs, succ, n = _load('train_success_1000')
  exs, elen, _ = _load('ep_length_mean')
  peak = max(succ)
  peak_i = succ.index(peak)
  print(f'success n={n} peak={peak:.4g} at steps={xs[peak_i]:.0f} '
        f'iter~{xs[peak_i] / (1024 * 300):.0f}')

  fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.4), sharex=True)
  ax = axes[0]
  ax.plot(xs, succ, color=COLOR, lw=1.6, label='nfc palmgoal p(g|s)')
  ax.scatter(
      [xs[peak_i]], [succ[peak_i]], color=COLOR, s=36, zorder=3,
      label=f'peak {peak:.3f}  (1 / 1000 eps)')
  ax.axvline(SPIKE_STEPS, color='0.45', lw=0.8, linestyle='--')
  for steps, name in CKPT_STEPS:
    ax.axvline(steps, color='0.7', lw=0.7, linestyle=':')
    ax.text(steps, 0.00135, name, fontsize=8, color='0.35',
            rotation=90, va='bottom', ha='right')
  ax.set_ylim(-0.00015, 0.0016)
  ax.set_title(
      'palm-goal train success — zoomed  (one blip at 151.8M)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('train_success_1000')
  ax.legend(loc='upper left', fontsize=8.5, framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

  ax = axes[1]
  ax.plot(exs, elen, color=COLOR, lw=1.6)
  ax.axhline(300.0, color='0.55', lw=0.7, linestyle='--')
  ax.axvline(SPIKE_STEPS, color='0.45', lw=0.8, linestyle='--')
  for steps, _name in CKPT_STEPS:
    ax.axvline(steps, color='0.7', lw=0.7, linestyle=':')
  ax.set_title('episode length around the blip', fontsize=11, fontweight='bold')
  ax.set_ylabel('ep_length_mean')
  ax.set_xlabel('Env Steps')
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.set_xlim(1.20e8, 1.62e8)

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
