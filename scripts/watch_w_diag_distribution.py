#!/usr/bin/env python3
"""Watch TD InfoNCE w_diag vector log and refresh distribution plots.

Reads ``logs/.../logs/w_diag/vectors.csv`` (one row per CRL update; columns
``w_0`` … ``w_{B-1}`` are the batch diagonal of IS weights).  Each time a
new row is appended, saves an updated figure.

Examples::

  # Plot latest row once and exit.
  python scripts/watch_w_diag_distribution.py --once \\
      --csv logs/ppo_td_infonce_drawer_norm16/ppo_sawyer_drawer_open_0/logs/w_diag/vectors.csv

  # Watch live during training (default poll 5s).
  python scripts/watch_w_diag_distribution.py --watch \\
      --csv logs/ppo_td_infonce_drawer_norm16/ppo_sawyer_drawer_open_0/logs/w_diag/vectors.csv

  # Auto-discover the newest vectors.csv under logs/.
  python scripts/watch_w_diag_distribution.py --watch --find_latest
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / 'figs' / 'w_diag_distribution.png'
STATE_PATH = ROOT / 'figs' / '.watch_w_diag_state.json'


def _find_latest_vectors_csv(logs_root: Path) -> Path | None:
  candidates = sorted(
      logs_root.glob('**/logs/w_diag/vectors.csv'),
      key=lambda p: p.stat().st_mtime,
      reverse=True,
  )
  return candidates[0] if candidates else None


def _load_state(path: Path) -> dict:
  if not path.is_file():
    return {}
  import json
  with path.open('r', encoding='utf-8') as fh:
    return json.load(fh)


def _save_state(path: Path, state: dict) -> None:
  import json
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)


def _parse_row(row: dict) -> tuple[int, int, np.ndarray]:
  iteration = int(float(row['iteration']))
  crl_step = int(float(row['crl_step']))
  w_cols = sorted(
      (k for k in row if k.startswith('w_')),
      key=lambda k: int(k.split('_', 1)[1]),
  )
  w = np.array([float(row[k]) for k in w_cols], dtype=np.float64)
  return iteration, crl_step, w


def _read_rows_from_offset(csv_path: Path, byte_offset: int) -> tuple[list, int]:
  """Return (parsed_rows, new_byte_offset)."""
  if not csv_path.is_file():
    return [], byte_offset
  _ensure_header(csv_path)
  key = str(csv_path)
  rows: list[tuple[int, int, np.ndarray]] = []
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    fh.seek(byte_offset)
    if byte_offset == 0:
      reader = csv.DictReader(fh)
    else:
      reader = csv.DictReader(fh, fieldnames=_HEADER_NAMES[key])
    for row in reader:
      if not row or row.get('iteration') in (None, 'iteration'):
        continue
      try:
        rows.append(_parse_row(row))
      except (KeyError, ValueError):
        continue
    new_offset = fh.tell()
  return rows, new_offset


_HEADER_NAMES: dict[str, list[str]] = {}


def _ensure_header(csv_path: Path) -> None:
  key = str(csv_path)
  if key in _HEADER_NAMES:
    return
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader, None)
  if header:
    _HEADER_NAMES[key] = header


def _atomic_savefig(fig: plt.Figure, out: Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.stem + '.tmp' + out.suffix)
  fig.savefig(str(tmp), dpi=150, bbox_inches='tight')
  if not tmp.is_file():
    raise RuntimeError(f'savefig did not create {tmp}')
  os.replace(str(tmp), str(out))


def _plot_distribution(
    *,
    latest_w: np.ndarray,
    iteration: int,
    crl_step: int,
    history: list[tuple[int, int, float, float, float, float]],
    csv_path: Path,
    out_path: Path,
    total_rows: int,
) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

  ax0 = axes[0]
  ax0.hist(latest_w, bins=min(50, max(10, len(latest_w) // 4)),
           color='#4C9BE8', edgecolor='white', alpha=0.9)
  ax0.axvline(float(np.mean(latest_w)), color='#E74C3C', lw=1.5,
              label=f'mean={np.mean(latest_w):.5f}')
  ax0.axvline(float(np.median(latest_w)), color='#2ECC71', lw=1.5, ls='--',
              label=f'median={np.median(latest_w):.5f}')
  ax0.set_xlabel('w_diag[i]  (IS weight diagonal)')
  ax0.set_ylabel('count')
  ax0.set_title(f'Latest batch  iter={iteration}  crl_step={crl_step}')
  ax0.legend(fontsize=8, loc='upper right')
  ax0.grid(True, alpha=0.25)

  ax1 = axes[1]
  if history:
    steps = np.arange(len(history))
    means = [h[2] for h in history]
    stds = [h[3] for h in history]
    ax1.plot(steps, means, color='#4C9BE8', lw=0.8, label='batch mean')
    ax1.fill_between(
        steps,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        color='#4C9BE8', alpha=0.2, label='mean ± std')
    ax1.set_xlabel('CRL update index (chronological)')
    ax1.set_ylabel('w_diag stats')
    ax1.set_title(f'Running stats  ({len(history)} updates logged)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.25)
  else:
    ax1.text(0.5, 0.5, 'no history yet', ha='center', va='center',
             transform=ax1.transAxes)
    ax1.set_axis_off()

  fig.suptitle(
      f'TD InfoNCE w_diag distribution — {csv_path.parents[2].name}\n'
      f'updated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  '
      f'rows={total_rows}  batch_size={len(latest_w)}',
      fontsize=11,
  )
  fig.tight_layout()
  _atomic_savefig(fig, out_path)
  plt.close(fig)


def _append_history(
    history: list[tuple[int, int, float, float, float, float]],
    iteration: int,
    crl_step: int,
    w: np.ndarray,
) -> None:
  history.append((
      iteration,
      crl_step,
      float(np.mean(w)),
      float(np.std(w)),
      float(np.min(w)),
      float(np.max(w)),
  ))


def _process_new_rows(
    csv_path: Path,
    new_rows: list[tuple[int, int, np.ndarray]],
    history: list[tuple[int, int, float, float, float, float]],
    out_path: Path,
    total_rows: int,
    *,
    plot_each: bool,
) -> None:
  if not new_rows:
    return
  if not plot_each:
    for iteration, crl_step, w in new_rows[:-1]:
      _append_history(history, iteration, crl_step, w)
    iteration, crl_step, w = new_rows[-1]
    _append_history(history, iteration, crl_step, w)
    _plot_distribution(
        latest_w=w,
        iteration=iteration,
        crl_step=crl_step,
        history=history,
        csv_path=csv_path,
        out_path=out_path,
        total_rows=total_rows,
    )
    if len(new_rows) > 1:
      print(
          f'  (+{len(new_rows) - 1} older rows ingested into history only)',
          flush=True,
      )
    print(
        f'[{datetime.now().strftime("%H:%M:%S")}] '
        f'iter={iteration} crl_step={crl_step}  '
        f'w_diag mean={history[-1][2]:.6f} std={history[-1][3]:.6f}  '
        f'min={history[-1][4]:.6f} max={history[-1][5]:.6f}  '
        f'→ {out_path}',
        flush=True,
    )
    return

  for iteration, crl_step, w in new_rows:
    _append_history(history, iteration, crl_step, w)
    _plot_distribution(
        latest_w=w,
        iteration=iteration,
        crl_step=crl_step,
        history=history,
        csv_path=csv_path,
        out_path=out_path,
        total_rows=total_rows,
    )
    print(
        f'[{datetime.now().strftime("%H:%M:%S")}] '
        f'iter={iteration} crl_step={crl_step}  '
        f'w_diag mean={history[-1][2]:.6f} std={history[-1][3]:.6f}  '
        f'min={history[-1][4]:.6f} max={history[-1][5]:.6f}  '
        f'→ {out_path}',
        flush=True,
    )


def run_once(
    csv_path: Path,
    out_path: Path,
    state_path: Path,
    *,
    plot_each: bool,
) -> int:
  if not csv_path.is_file():
    print(f'CSV not found: {csv_path}')
    return 1
  _ensure_header(csv_path)
  state = _load_state(state_path)
  if state.get('csv_path') != str(csv_path):
    state = {}
  history: list[tuple[int, int, float, float, float, float]] = list(
      state.get('history', []))
  offset = int(state.get('byte_offset', 0))

  rows, offset = _read_rows_from_offset(csv_path, offset)

  total_rows = int(state.get('total_rows', 0)) + len(rows)
  if rows:
    _process_new_rows(
        csv_path, rows, history, out_path, total_rows, plot_each=plot_each)
  else:
    if not history:
      print(f'CSV exists but no data rows yet: {csv_path}')
      return 0
    return 0

  _save_state(state_path, {
      'csv_path': str(csv_path),
      'byte_offset': offset,
      'total_rows': total_rows,
      'history': history[-500:],  # cap state size
      'last_update': datetime.now().isoformat(timespec='seconds'),
  })
  return 0


def run_watch(
    csv_path: Path,
    out_path: Path,
    state_path: Path,
    poll_interval: float,
    *,
    plot_each: bool,
) -> None:
  print(f'Watching {csv_path}')
  print(f'Output   {out_path}')
  print(f'Poll every {poll_interval:.1f}s  (Ctrl+C to stop)')
  try:
    while True:
      if csv_path.is_file():
        run_once(csv_path, out_path, state_path, plot_each=plot_each)
      else:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] waiting for {csv_path}...')
      time.sleep(max(1.0, poll_interval))
  except KeyboardInterrupt:
    print('\nStopped watch.')


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument(
      '--csv', type=Path, default=None,
      help='Path to logs/.../logs/w_diag/vectors.csv')
  ap.add_argument(
      '--find_latest', action='store_true',
      help='Use the most recently modified vectors.csv under logs/.')
  ap.add_argument(
      '--out', type=Path, default=DEFAULT_OUT,
      help=f'Output PNG (default: {DEFAULT_OUT})')
  ap.add_argument(
      '--state', type=Path, default=STATE_PATH,
      help='JSON state file tracking read offset / history')
  ap.add_argument(
      '--watch', action='store_true',
      help='Poll for new rows and replot on each append.')
  ap.add_argument(
      '--poll_interval', type=float, default=5.0,
      help='Seconds between polls in --watch mode (default: 5).')
  ap.add_argument(
      '--once', action='store_true',
      help='Process pending rows once and exit (default if not --watch).')
  ap.add_argument(
      '--plot_each', action='store_true',
      help='Save a PNG for every new row (slow if many rows arrive at once). '
           'Default: update history for all new rows, plot the latest only.')
  args = ap.parse_args()

  csv_path = args.csv
  if args.find_latest or csv_path is None:
    csv_path = _find_latest_vectors_csv(ROOT / 'logs')
    if csv_path is None:
      ap.error('No vectors.csv found under logs/; pass --csv explicitly.')
    print(f'Using latest CSV: {csv_path}')

  if args.watch:
    run_watch(csv_path, args.out, args.state, args.poll_interval,
              plot_each=args.plot_each)
  else:
    raise SystemExit(run_once(csv_path, args.out, args.state,
                              plot_each=args.plot_each))


if __name__ == '__main__':
  main()
