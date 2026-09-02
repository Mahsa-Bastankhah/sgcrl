#!/usr/bin/env python3
"""c7t2 dgr+tKL=0.1 per-seed success, with 'success on' spans.

Train raw. Eval faint raw + roll mean w=5. Green bands = train_success_1000>=0.6.

  python scripts/plot_c7t2_final_dgr_tkl01_per_seed_success.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = (
    'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
    '_minstd1e5_ent05to001_ep70_300m_crl10_dualgradreg_c100'
    '_lamlr1e6_tkl01_warp_4h'
)
LOG_DIR = os.path.join(REPO, 'logs', 'final_runs', NAME)
OUT = os.path.join(
    REPO, 'figs', 'builderbench', 'final_runs',
    'c7t2_extrew1_dgr_tkl01_per_seed_success.png')

SUCC_ON = 0.6
MERGE_ITERS = 5
VIDEO_ITERS = (2800, 3200)
SPI = 1024 * 70
C_TRAIN = '#2A9D8F'
C_EVAL = '#4C9BE8'
C_ON = '#7CB342'


def _load_xy(path: str, x_col: str, y_col: str):
  xs, ys = [], []
  if not os.path.isfile(path):
    return xs, ys
  with open(path, newline='') as f:
    for row in csv.DictReader(f):
      x = base._coerce(row.get(x_col, ''))
      y = base._coerce(row.get(y_col, ''))
      if x is not None and y is not None:
        xs.append(int(x))
        ys.append(float(y))
  return xs, ys


def _spans(steps, succ, *, thresh: float, merge_steps: int):
  on = [s >= thresh for s in succ]
  raw = []
  i = 0
  n = len(on)
  while i < n:
    if not on[i]:
      i += 1
      continue
    j = i + 1
    while j < n and on[j]:
      j += 1
    raw.append((steps[i], steps[j - 1]))
    i = j
  if not raw:
    return []
  merged = [raw[0]]
  for a, b in raw[1:]:
    prev_a, prev_b = merged[-1]
    if a - prev_b <= merge_steps:
      merged[-1] = (prev_a, b)
    else:
      merged.append((a, b))
  return merged


def _style(ax, *, title: str) -> None:
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_ylabel('Success', fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.12)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def main() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.6), sharex=True)
  fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16, hspace=0.28)
  for seed, ax in ((0, axes[0]), (1, axes[1])):
    run = os.path.join(LOG_DIR, f'ppo_builderbench_creative_7_task2_{seed}')
    txs, tys = _load_xy(
        os.path.join(run, 'logs', 'learner', 'logs.csv'),
        base.TRAIN_X_COL, base.TRAIN_METRIC)
    exs, eys = _load_xy(
        os.path.join(run, 'logs', 'eval', 'logs.csv'),
        'iteration', base.EVAL_METRIC)
    exs = [(it + 1) * SPI for it in exs]
    spans = _spans(txs, tys, thresh=SUCC_ON, merge_steps=MERGE_ITERS * SPI)
    for a, b in spans:
      ax.axvspan(a, b, color=C_ON, alpha=0.18, lw=0, zorder=1)
      ax.plot(
          [a, b], [1.06, 1.06], color=C_ON, lw=6.0, solid_capstyle='butt',
          alpha=0.95, zorder=4)
    if txs:
      ax.plot(txs, tys, color=C_TRAIN, lw=1.7, label='train (raw last-1000)',
              zorder=3)
    if exs:
      base._plot_eval_smoothed(
          ax, exs, eys, color=C_EVAL, label='eval (roll mean w=5)',
          zorder=3, linewidth=2.0)
    if seed == 1:
      for it in VIDEO_ITERS:
        ax.axvline(
            (it + 1) * SPI, color='#C45C26', ls='--', lw=1.1, alpha=0.85,
            zorder=5)
        ax.text(
            (it + 1) * SPI, 1.10, f'vid {it}', ha='center', va='bottom',
            fontsize=7.0, color='#C45C26')
    n_on = len(spans)
    last = tys[-1] if tys else float('nan')
    peak = max(tys) if tys else float('nan')
    print(f'  seed {seed}: last={last:.3f} peak={peak:.3f}  '
          f'success-on spans={n_on}')
    for a, b in spans:
      print(f'    {a/1e6:.1f}–{b/1e6:.1f}M')
    _style(ax, title=f'seed {seed}')
    ax.legend(loc='lower left', fontsize=8.0, framealpha=0.92)

  extra = [Patch(facecolor=C_ON, alpha=0.45, label=f'success on  (s1000≥{SUCC_ON:g})')]
  h, lab = axes[1].get_legend_handles_labels()
  axes[1].legend(
      h + extra, lab + [e.get_label() for e in extra],
      loc='lower left', fontsize=8.0, framealpha=0.92)
  fig.suptitle(
      'c7t2  ·  dgr c=100  ·  tKL=0.1  ·  extrew  ·  per seed',
      fontsize=12, fontweight='bold')
  fig.text(
      0.5, 0.015,
      'Green bar at the top + shading = train_success_1000 ≥ 0.6 '
      f'(gaps ≤{MERGE_ITERS} iters merged). '
      'Train raw; eval faint raw + bold roll mean w=5. '
      'Orange dashed on seed 1 = existing in-train videos (2800, 3200).',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
