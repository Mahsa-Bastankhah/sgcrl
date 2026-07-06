#!/usr/bin/env python3
"""Plot TD InfoNCE bootstrap action a' histograms from side CSV.

Reads ``logs/.../logs/a_prime_hist/vectors.csv`` (columns ``d{d}_b{b}`` =
counts for action dimension *d*, bin *b* over env action bounds).

Examples::

  python scripts/plot_a_prime_hist.py \\
      --csv logs/ppo_td_infonce_drawer_norm16/ppo_sawyer_drawer_open_0/logs/a_prime_hist/vectors.csv

  python scripts/plot_a_prime_hist.py --csv ... --row -1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load_row(csv_path: Path, row_idx: int) -> tuple[int, int, np.ndarray]:
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)
  if not rows:
    raise SystemExit(f'No data rows in {csv_path}')
  row = rows[row_idx]
  iteration = int(float(row['iteration']))
  crl_step = int(float(row['crl_step']))
  d_cols = sorted(
      (k for k in row if k.startswith('d') and '_b' in k),
      key=lambda k: (int(k.split('_b')[0][1:]), int(k.split('_b')[1])),
  )
  if not d_cols:
    raise SystemExit(f'No d*_b* columns in {csv_path}')
  n_bins = max(int(k.split('_b')[1]) for k in d_cols) + 1
  n_dims = max(int(k.split('_b')[0][1:]) for k in d_cols) + 1
  hist = np.zeros((n_dims, n_bins), dtype=np.float64)
  for k in d_cols:
    d = int(k.split('_b')[0][1:])
    b = int(k.split('_b')[1])
    hist[d, b] = float(row[k])
  return iteration, crl_step, hist


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--csv', type=Path, required=True)
  ap.add_argument('--row', type=int, default=-1,
                  help='Which CRL update row to plot (-1 = latest).')
  ap.add_argument('--out', type=Path,
                  default=ROOT / 'figs' / 'a_prime_hist_latest.png')
  ap.add_argument('--action_low', type=float, default=-1.0)
  ap.add_argument('--action_high', type=float, default=1.0)
  args = ap.parse_args()

  iteration, crl_step, hist = _load_row(args.csv, args.row)
  n_dims, n_bins = hist.shape
  edges = np.linspace(args.action_low, args.action_high, n_bins + 1)
  centers = 0.5 * (edges[:-1] + edges[1:])

  ncols = min(4, n_dims)
  nrows = int(np.ceil(n_dims / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                           squeeze=False)
  for d in range(n_dims):
    ax = axes.ravel()[d]
    ax.bar(centers, hist[d], width=(edges[1] - edges[0]) * 0.9,
           color='#9B59B6', edgecolor='white', alpha=0.9)
    ax.set_title(f'dim {d}', fontsize=9)
    ax.set_xlim(args.action_low, args.action_high)
    ax.grid(True, alpha=0.2)
  for ax in axes.ravel()[n_dims:]:
    ax.set_axis_off()

  run_name = args.csv.parents[2].name
  fig.suptitle(
      f"TD InfoNCE a' histogram — {run_name}\n"
      f'iter={iteration}  crl_step={crl_step}  ({n_bins} bins/dim)',
      fontsize=11,
  )
  fig.supxlabel("a' component value")
  fig.supylabel('count in batch')
  fig.tight_layout()
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(str(args.out), dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Wrote {args.out}')


if __name__ == '__main__':
  main()
