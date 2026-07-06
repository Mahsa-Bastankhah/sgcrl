#!/usr/bin/env python3
"""Plot historical TD InfoNCE w_diag batch histograms from vectors.csv.

Every CRL update appends one row (full batch diagonal w_0 … w_{B-1}) to
``logs/.../logs/w_diag/vectors.csv``.  This script samples past rows and
builds a grid of histograms so you can see how the batch distribution
evolved over training.

Examples::

  # 16 evenly spaced snapshots through the whole log.
  python scripts/plot_w_diag_history.py \\
      --csv logs/ppo_td_infonce_drawer_norm16/ppo_sawyer_drawer_open_0/logs/w_diag/vectors.csv

  # One snapshot per 100 PPO iterations (last CRL step of each iter).
  python scripts/plot_w_diag_history.py --csv ... --every_k_iters 100

  # Specific iterations; also write individual PNGs.
  python scripts/plot_w_diag_history.py --csv ... --iterations 50,200,500,1000 \\
      --save_individual figs/w_diag_history/
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _parse_row(row: dict) -> tuple[int, int, np.ndarray]:
  iteration = int(float(row['iteration']))
  crl_step = int(float(row['crl_step']))
  w_cols = sorted(
      (k for k in row if k.startswith('w_')),
      key=lambda k: int(k.split('_', 1)[1]),
  )
  w = np.array([float(row[k]) for k in w_cols], dtype=np.float64)
  return iteration, crl_step, w


def _count_data_rows(csv_path: Path) -> int:
  n = 0
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    reader = csv.reader(fh)
    next(reader, None)  # header
    for _ in reader:
      n += 1
  return n


def _row_indices_evenly_spaced(n_rows: int, n_snapshots: int) -> list[int]:
  if n_rows <= 0:
    return []
  if n_snapshots >= n_rows:
    return list(range(n_rows))
  if n_snapshots == 1:
    return [n_rows - 1]
  return [
      int(round(i * (n_rows - 1) / (n_snapshots - 1)))
      for i in range(n_snapshots)
  ]


def _collect_by_row_indices(
    csv_path: Path,
    row_indices: set[int],
) -> dict[int, tuple[int, int, np.ndarray]]:
  """Map row_index -> (iteration, crl_step, w)."""
  want = set(row_indices)
  found: dict[int, tuple[int, int, np.ndarray]] = {}
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    reader = csv.DictReader(fh)
    for row_idx, row in enumerate(reader):
      if row_idx not in want:
        continue
      try:
        found[row_idx] = _parse_row(row)
      except (KeyError, ValueError):
        continue
      if len(found) == len(want):
        break
  return found


def _collect_last_per_iteration(
    csv_path: Path,
    *,
    every_k_iters: int | None = None,
    iterations: set[int] | None = None,
) -> list[tuple[int, int, np.ndarray]]:
  """Return rows sorted by iteration, one per selected PPO iteration."""
  last_by_iter: dict[int, tuple[int, np.ndarray]] = {}
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      try:
        iteration, crl_step, w = _parse_row(row)
      except (KeyError, ValueError):
        continue
      if iterations is not None and iteration not in iterations:
        continue
      if every_k_iters is not None and iteration % every_k_iters != 0:
        continue
      prev = last_by_iter.get(iteration)
      if prev is None or crl_step >= prev[0]:
        last_by_iter[iteration] = (crl_step, w)

  out: list[tuple[int, int, np.ndarray]] = []
  for iteration in sorted(last_by_iter):
    crl_step, w = last_by_iter[iteration]
    out.append((iteration, crl_step, w))
  return out


def _plot_single_hist(
    ax: plt.Axes,
    w: np.ndarray,
    iteration: int,
    crl_step: int,
    *,
    xlim: tuple[float, float] | None = None,
) -> None:
  ax.hist(
      w,
      bins=min(40, max(10, len(w) // 4)),
      color='#4C9BE8',
      edgecolor='white',
      alpha=0.9,
  )
  mean = float(np.mean(w))
  ax.axvline(mean, color='#E74C3C', lw=1.2)
  ax.set_title(f'iter={iteration}  crl={crl_step}\nμ={mean:.4f}', fontsize=8)
  ax.tick_params(labelsize=7)
  if xlim is not None:
    ax.set_xlim(xlim)
  ax.grid(True, alpha=0.2)


def _grid_shape(n: int) -> tuple[int, int]:
  if n <= 0:
    return 1, 1
  ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
  nrows = int(math.ceil(n / ncols))
  return nrows, ncols


def _plot_grid(
    snapshots: list[tuple[int, int, np.ndarray]],
    *,
    title: str,
    out_path: Path,
    share_x: bool = True,
) -> None:
  if not snapshots:
    print('No snapshots to plot.')
    return

  if share_x:
    all_w = np.concatenate([w for _, _, w in snapshots])
    pad = 0.05 * (float(all_w.max()) - float(all_w.min()) + 1e-9)
    xlim = (float(all_w.min()) - pad, float(all_w.max()) + pad)
  else:
    xlim = None

  nrows, ncols = _grid_shape(len(snapshots))
  fig, axes = plt.subplots(
      nrows, ncols,
      figsize=(3.2 * ncols, 2.6 * nrows),
      squeeze=False,
  )
  flat = axes.ravel()
  for ax, (iteration, crl_step, w) in zip(flat, snapshots):
    _plot_single_hist(ax, w, iteration, crl_step, xlim=xlim)
  for ax in flat[len(snapshots):]:
    ax.set_axis_off()

  fig.suptitle(title, fontsize=11)
  fig.supxlabel('w_diag[i]  (IS weight diagonal)', fontsize=10)
  fig.tight_layout()

  out_path.parent.mkdir(parents=True, exist_ok=True)
  tmp = out_path.with_name(out_path.stem + '.tmp' + out_path.suffix)
  fig.savefig(str(tmp), dpi=150, bbox_inches='tight')
  os.replace(str(tmp), str(out_path))
  plt.close(fig)
  print(f'Wrote {out_path}  ({len(snapshots)} snapshots)')


def _save_individual(
    snapshots: list[tuple[int, int, np.ndarray]],
    out_dir: Path,
) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  for iteration, crl_step, w in snapshots:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    _plot_single_hist(ax, w, iteration, crl_step)
    ax.set_xlabel('w_diag[i]')
    ax.set_ylabel('count')
    out = out_dir / f'w_diag_iter{iteration:05d}_crl{crl_step:03d}.png'
    tmp = out.with_name(out.stem + '.tmp' + out.suffix)
    fig.savefig(str(tmp), dpi=120, bbox_inches='tight')
    os.replace(str(tmp), str(out))
    plt.close(fig)
  print(f'Wrote {len(snapshots)} individual PNGs → {out_dir}')


def main() -> None:
  ap = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument('--csv', type=Path, required=True,
                  help='Path to logs/.../logs/w_diag/vectors.csv')
  ap.add_argument(
      '--snapshots', type=int, default=16,
      help='Number of evenly spaced row snapshots (default: 16). '
           'Ignored if --every_k_iters or --iterations is set.')
  ap.add_argument(
      '--every_k_iters', type=int, default=0,
      help='One histogram per PPO iteration divisible by K (uses last CRL step).')
  ap.add_argument(
      '--iterations', type=str, default='',
      help='Comma-separated PPO iterations to plot (last CRL step each).')
  ap.add_argument(
      '--out', type=Path,
      default=ROOT / 'figs' / 'w_diag_history_grid.png',
      help='Output grid PNG.')
  ap.add_argument(
      '--save_individual', type=Path, default=None,
      help='If set, also save one PNG per snapshot in this directory.')
  args = ap.parse_args()

  csv_path = args.csv
  if not csv_path.is_file():
    raise SystemExit(f'CSV not found: {csv_path}')

  run_name = csv_path.parents[2].name
  ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  if args.iterations.strip():
    iters = {int(x.strip()) for x in args.iterations.split(',') if x.strip()}
    snapshots = _collect_last_per_iteration(csv_path, iterations=iters)
    mode = f'iterations={sorted(iters)}'
  elif args.every_k_iters > 0:
    snapshots = _collect_last_per_iteration(
        csv_path, every_k_iters=args.every_k_iters)
    mode = f'every_k_iters={args.every_k_iters}'
  else:
    n_rows = _count_data_rows(csv_path)
    indices = _row_indices_evenly_spaced(n_rows, args.snapshots)
    by_idx = _collect_by_row_indices(csv_path, set(indices))
    snapshots = [by_idx[i] for i in sorted(indices) if i in by_idx]
    mode = f'{args.snapshots} evenly spaced rows (n_rows={n_rows})'

  title = (
      f'TD InfoNCE w_diag history — {run_name}\n'
      f'{mode}  ·  updated {ts}'
  )
  _plot_grid(snapshots, title=title, out_path=args.out)
  if args.save_individual is not None:
    _save_individual(snapshots, args.save_individual)


if __name__ == '__main__':
  main()
