#!/usr/bin/env python3
"""PPO+RND creative-1-task1 eval success from the live slurm log.

Eval is faint raw + bold centered rolling mean (window=5). Hard / easy /
very-hard on one panel.

  python paper_plot_scripts/plot_ppo_rnd_c1t1_success.py
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

LOG = os.path.join(
    REPO, 'slurm',
    'ppo_rnd_bb_c1t1_pd_res_e1024_ep30_200m_warp_3803351_0.log')
OUT_DIR = os.path.join(REPO, 'figs', 'builderbench', 'rnd')
OUT_STEM = os.path.join(OUT_DIR, 'c1t1_ppo_rnd_eval_success')

SERIES = (
    ('eval/episode_success_rate', 'hard', '#0072B2'),
    ('eval/episode_easy_success_rate', 'easy', '#009E73'),
    ('eval/episode_very_hard_success_rate', 'very-hard', '#D55E00'),
)

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


def _parse(path: str) -> dict[str, list[tuple[int, float]]]:
  text = open(path, encoding='utf-8', errors='replace').read()
  steps = [
      float(v) for v in re.findall(
          r"'training/env_steps': Array\(([^,]+),", text)
  ]
  out: dict[str, list[tuple[int, float]]] = {}
  for key, _lab, _c in SERIES:
    ys = [
        float(v) for v in re.findall(
            rf"'{re.escape(key)}': Array\(([^,]+),", text)
    ]
    n = min(len(steps), len(ys))
    out[key] = [(int(steps[i]), ys[i]) for i in range(n)]
  return out


def main() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  series = _parse(LOG)
  plt.rcParams.update(_paper_rc())
  fig, ax = plt.subplots(figsize=(8.2, 4.8))

  last_steps = 0
  for key, lab, color in SERIES:
    pts = series.get(key, [])
    if not pts:
      print(f'  {lab}: no data')
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    last_steps = xs[-1]
    sm = base._plot_eval_smoothed(
        ax, xs, ys, color=color, label=lab, zorder=3, linewidth=2.4)
    print(f'  {lab}: n={len(ys)} raw last={ys[-1]:.3f} peak={max(ys):.3f}  '
          f'smooth last={sm[-1]:.3f} peak={max(sm):.3f}')

  ax.set_title('creative-1-task1  ·  PPO+RND eval', fontweight='bold', pad=8)
  ax.set_xlabel('Environment steps')
  ax.set_ylabel(f'Eval success (roll. mean $w$={w})')
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.legend(frameon=False, loc='upper right')
  fig.text(
      0.5, 0.01,
      f'Job 3803351 seed 0, still running '
      f'({last_steps/1e6:.0f}M / 200M).  '
      f'Faint: raw eval.  Bold: centered rolling mean (window={w}).',
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
