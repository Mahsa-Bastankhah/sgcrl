#!/usr/bin/env python3
"""Plot easy in-bucket eval for policies trained with 7.5 cm goal-ball success.

Reads CSVs written by ``scripts/eval_allegro_goalball_inbucket.py``.
Eval is faint raw + bold rolling mean (window=5).
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'figs', 'allegro_kuka_throw', 'goalball_inbucket_eval')
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_live_queue_goalball_inbucket_eval.png')
C = base.ACCENT_COLORS

# More-specific tokens first. These are 7.5cm-trained runs; the series is
# *eval* success under the easier in-bucket cylinder.
STYLES = (
    ('xy045m060',
     'hover near10 T=70 q40 200M  trained 7.5cm  (0.45,-0.60)', C[0]),
    ('xy050030',
     'keep-arm T=50 q05 500M  trained 7.5cm  (0.50,-0.30)', C[2]),
    ('xy000m040',
     'keep-arm hidetable T=50 q05 500M  trained 7.5cm  (0.00,-0.40)', C[1]),
    ('xy0022',
     'keep-arm hidetable T=50 q05 400M  trained 7.5cm  (0.00,-0.22)', C[3]),
)


def _label_color(stem: str) -> tuple[str, str]:
  for token, label, color in STYLES:
    if token in stem:
      return label, color
  extra = ['#9B2226', '#2A9D8F', '#C45C26', '#6B8E9F']
  idx = abs(hash(stem)) % len(extra)
  return stem, extra[idx]


def _load_rows(path: str) -> list[dict]:
  if not os.path.isfile(path):
    return []
  with open(path, newline='') as fh:
    return list(csv.DictReader(fh))


def plot(out_path: str = OUT_PATH) -> str:
  paths = sorted(glob.glob(os.path.join(OUT_DIR, '*.csv')))
  fig, ax = plt.subplots(figsize=(12.0, 5.2))
  n_series = 0
  for path in paths:
    rows = _load_rows(path)
    if not rows:
      continue
    xs, ys = [], []
    for row in rows:
      if not row.get('in_bucket'):
        continue
      xs.append(float(row['env_steps']))
      ys.append(float(row['in_bucket']))
    if not xs:
      continue
    stem = os.path.splitext(os.path.basename(path))[0]
    label, color = _label_color(stem)
    base._plot_eval_smoothed(
        ax, xs, ys, color=color,
        label=f'{label}  s0', linestyle='-')
    n_series += 1
  if n_series == 0:
    ax.text(0.5, 0.5, 'no in-bucket eval CSVs yet',
            ha='center', va='center', transform=ax.transAxes)
  else:
    ax.legend(loc='upper left', fontsize=7.6, framealpha=0.95)
  ax.set_title(
      '7.5 cm-trained policies — eval success = ever in bucket  '
      f'(faint raw + bold rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=12, fontweight='bold')
  ax.set_ylabel(
      f'in-bucket success (roll mean, w={base.EVAL_SMOOTH_WINDOW})',
      fontsize=10)
  ax.set_xlabel('Train env steps', fontsize=10)
  ax.set_ylim(-0.02, 1.05)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.tight_layout()
  fig.savefig(out_path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}  n={n_series}', flush=True)
  return out_path


def main() -> None:
  plot()


if __name__ == '__main__':
  main()
