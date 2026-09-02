#!/usr/bin/env python3
"""Sawyer bin PPO+RND eval success from slurm logs (still-running 40M job).

Eval is faint raw + bold centered rolling mean (window=5). Two seeds plus
mean ±1 SE.

  python paper_plot_scripts/plot_sawyer_bin_rnd_success.py
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import plot_builderbench_train_success1000 as base  # noqa: E402

SLURM = os.path.join(REPO, 'slurm')
OUT_DIR = os.path.join(REPO, 'figs', 'metaworld', 'final_metaworld_runs')
OUT_STEM = os.path.join(OUT_DIR, 'sawyer_bin_rnd_success')

LOGS = (
    (0, 'ppo_rnd_bin_40m_3782328_0.log'),
    (1, 'ppo_rnd_bin_40m_3782328_1.log'),
)
STEPS_PER_ITER = 2048
EVAL_RE = re.compile(
    r'\[eval\] iter=(\d+) return=([0-9.eE+-]+) success=([0-9.eE+-]+)')

COLOR_MEAN = '#0072B2'
COLOR_S0 = '#56B4E9'
COLOR_S1 = '#E69F00'

TITLE_FS = 18
LABEL_FS = 16
TICK_FS = 14
LEGEND_FS = 13
CAPTION_FS = 11


def _paper_rc() -> dict:
  return {
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
      'mathtext.fontset': 'stix',
      'font.size': TICK_FS,
      'axes.titlesize': TITLE_FS,
      'axes.labelsize': LABEL_FS,
      'xtick.labelsize': TICK_FS,
      'ytick.labelsize': TICK_FS,
      'legend.fontsize': LEGEND_FS,
      'axes.linewidth': 1.15,
      'pdf.fonttype': 42,
      'ps.fonttype': 42,
      'savefig.dpi': 300,
  }


def _parse_eval(path: str) -> list[tuple[int, float]]:
  text = open(path, encoding='utf-8', errors='replace').read()
  pts = []
  for it, _ret, suc in EVAL_RE.findall(text):
    pts.append((int(it) * STEPS_PER_ITER, float(suc)))
  pts.sort(key=lambda p: p[0])
  return pts


def main() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  seeds: list[list[tuple[int, float]]] = []
  for seed, name in LOGS:
    path = os.path.join(SLURM, name)
    pts = _parse_eval(path)
    print(f'  seed {seed}: {len(pts)} evals last={pts[-1] if pts else None}')
    if pts:
      seeds.append(pts)

  plt.rcParams.update(_paper_rc())
  fig, ax = plt.subplots(figsize=(8.2, 4.8))

  seed_colors = (COLOR_S0, COLOR_S1)
  for i, pts in enumerate(seeds):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sm = base._plot_eval_smoothed(
        ax, xs, ys, color=seed_colors[i],
        label=f'seed {i}  (roll. mean $w$={w})',
        zorder=3, linewidth=2.0)
    if sm:
      print(f'  seed {i} raw last={ys[-1]:.3f} peak={max(ys):.3f}  '
            f'smooth last={sm[-1]:.3f} peak={max(sm):.3f}')

  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if xs and n >= 1:
    if n > 1:
      lo = base._rolling_mean([m - s for m, s in zip(mean, se)], w)
      hi = base._rolling_mean([m + s for m, s in zip(mean, se)], w)
      ax.fill_between(xs, lo, hi, color=COLOR_MEAN, alpha=0.18, lw=0, zorder=2)
    sm = base._plot_eval_smoothed(
        ax, xs, mean, color=COLOR_MEAN,
        label=f'mean  ($n$={n}, roll. mean $w$={w})',
        zorder=4, linewidth=2.8)
    print(f'  mean raw last={mean[-1]:.3f} peak={max(mean):.3f}  '
          f'smooth last={sm[-1] if sm else float("nan"):.3f}')

  ax.set_title('Sawyer bin  ·  PPO+RND eval', fontweight='bold', pad=8)
  ax.set_xlabel('Environment steps')
  ax.set_ylabel(f'Eval success (roll. mean $w$={w})')
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.legend(frameon=False, loc='upper left')
  last_m = max((pts[-1][0] for pts in seeds if pts), default=0) / 1e6
  fig.text(
      0.5, 0.01,
      f'Job 3782328, still running ({last_m:.0f}M / 40M).  '
      f'Faint: raw eval.  Bold: centered rolling mean (window={w}).  '
      r'Shade: $\pm$1 s.e. of the seed mean.',
      ha='center', va='bottom', fontsize=CAPTION_FS, color='#333333',
  )
  fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.18)

  os.makedirs(OUT_DIR, exist_ok=True)
  png = OUT_STEM + '.png'
  pdf = OUT_STEM + '.pdf'
  tmp = png + '.tmp.png'
  fig.savefig(tmp, dpi=300, bbox_inches='tight')
  os.replace(tmp, png)
  fig.savefig(pdf, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {png}')
  print(f'→ {pdf}')


if __name__ == '__main__':
  main()
