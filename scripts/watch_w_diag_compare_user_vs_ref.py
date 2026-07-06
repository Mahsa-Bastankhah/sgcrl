#!/usr/bin/env python3
"""Live w_diag monitoring for official ref TD-InfoNCE (README tasks).

By default plots the upstream ``fetch_reach`` run under
``logs/ref_td_infonce_fetch_reach/``: w_diag histogram, logits_w diagonal,
and evaluator ``success_1000``.  Pass ``--user_csv`` to overlay your PPO run
(on a different env) for informal comparison.

Examples::

  python scripts/watch_w_diag_compare_user_vs_ref.py --once

  python scripts/watch_w_diag_compare_user_vs_ref.py --watch --poll_interval 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / 'figs' / 'w_diag_compare_user_vs_ref.png'
DEFAULT_STATE = ROOT / 'figs' / '.watch_w_diag_compare_state.json'

REF_LOG_ROOT = ROOT / 'logs' / 'ref_td_infonce_fetch_reach'
DEFAULT_REF_CSV = REF_LOG_ROOT / 'logs' / 'w_diag' / 'vectors.csv'
DEFAULT_REF_LOGITS_CSV = REF_LOG_ROOT / 'logs' / 'logits_w' / 'vectors.csv'
DEFAULT_REF_EVAL_CSV = REF_LOG_ROOT / 'logs' / 'evaluator' / 'logs.csv'

# Optional overlay: your PPO TD-InfoNCE run on a *different* env/task.
DEFAULT_USER_CSV = None
DEFAULT_USER_EVAL_CSV = None


def _load_state(path: Path) -> dict:
  if not path.is_file():
    return {}
  with path.open('r', encoding='utf-8') as fh:
    return json.load(fh)


def _save_state(path: Path, state: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)


def _w_columns(row: dict) -> list[str]:
  return sorted(
      (k for k in row if k.startswith('w_')),
      key=lambda k: int(k.split('_', 1)[1]),
  )


def _lw_columns(row: dict) -> list[str]:
  return sorted(
      (k for k in row if k.startswith('lw_')),
      key=lambda k: int(k.split('_', 1)[1]),
  )


def _parse_w_row(row: dict) -> np.ndarray:
  cols = _w_columns(row)
  return np.array([float(row[k]) for k in cols], dtype=np.float64)


def _parse_lw_row(row: dict) -> np.ndarray:
  cols = _lw_columns(row)
  return np.array([float(row[k]) for k in cols], dtype=np.float64)


def _ensure_header(csv_path: Path, cache: dict[str, list[str]]) -> list[str] | None:
  key = str(csv_path)
  if key in cache:
    return cache[key]
  if not csv_path.is_file():
    return None
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    header = next(csv.reader(fh), None)
  if header:
    cache[key] = header
  return header


def _read_rows_from_offset(
    csv_path: Path,
    byte_offset: int,
    header_cache: dict[str, list[str]],
) -> tuple[list[np.ndarray], int, int]:
  """Return (parsed rows as w vectors, new byte offset, rows read)."""
  if not csv_path.is_file():
    return [], byte_offset, 0
  header = _ensure_header(csv_path, header_cache)
  if not header:
    return [], byte_offset, 0
  rows: list[np.ndarray] = []
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    fh.seek(byte_offset)
    if byte_offset == 0:
      reader = csv.DictReader(fh)
    else:
      reader = csv.DictReader(fh, fieldnames=header)
    for row in reader:
      if not row or row.get('iteration') in (None, 'iteration'):
        continue
      try:
        rows.append(_parse_w_row(row))
      except (KeyError, ValueError):
        continue
    new_offset = fh.tell()
  return rows, new_offset, len(rows)


def _read_tail_lines(csv_path: Path, n_data_rows: int) -> list[str]:
  """Return up to ``n_data_rows`` trailing data lines (excluding header)."""
  if not csv_path.is_file() or n_data_rows <= 0:
    return []
  chunk = 65536
  lines: list[bytes] = []
  with csv_path.open('rb') as fh:
    fh.seek(0, os.SEEK_END)
    pos = fh.tell()
    while pos > 0 and len(lines) <= n_data_rows:
      read_size = min(chunk, pos)
      pos -= read_size
      fh.seek(pos)
      data = fh.read(read_size)
      lines = data.splitlines() + lines
      if len(lines) > n_data_rows + 1:
        break
  text_lines = [ln.decode('utf-8', errors='replace') for ln in lines if ln.strip()]
  if not text_lines:
    return []
  if text_lines[0].startswith('iteration,'):
    data_lines = text_lines[1:]
  else:
    data_lines = text_lines
  return data_lines[-n_data_rows:]


def _tail_w_values(csv_path: Path, last_n_rows: int) -> tuple[np.ndarray, int]:
  """Read last ``last_n_rows`` w_diag rows without scanning the full file."""
  if not csv_path.is_file():
    return np.array([], dtype=np.float64), 0
  data_lines = _read_tail_lines(csv_path, last_n_rows)
  if not data_lines:
    return np.array([], dtype=np.float64), 0
  header = _ensure_header(csv_path, {})
  if not header:
    return np.array([], dtype=np.float64), 0
  vecs: list[np.ndarray] = []
  for line in data_lines:
    row = dict(zip(header, next(csv.reader([line]))))
    try:
      vecs.append(_parse_w_row(row))
    except (KeyError, ValueError):
      continue
  total = _count_data_rows(csv_path)
  if not vecs:
    return np.array([], dtype=np.float64), total
  return np.concatenate(vecs), total


def _count_data_rows(csv_path: Path) -> int:
  """Fast approximate row count via newline count minus header."""
  if not csv_path.is_file():
    return 0
  with csv_path.open('rb') as fh:
    n_lines = sum(1 for _ in fh)
  return max(0, n_lines - 1)


def _accumulate_w(
    csv_path: Path,
    last_n_batches: int,
    state: dict,
    header_cache: dict[str, list[str]],
) -> tuple[np.ndarray, int]:
  """Incremental read with rolling buffer capped at ``last_n_batches`` rows."""
  key = str(csv_path)
  entry = state.get('csv', {}).get(key, {})
  buffer: list[list[float]] = entry.get('buffer', [])
  offset = int(entry.get('byte_offset', 0))

  if offset == 0 and not buffer:
    flat, total = _tail_w_values(csv_path, last_n_batches)
    if total == 0:
      return flat, 0
    with csv_path.open('rb') as fh:
      fh.seek(0, os.SEEK_END)
      end_offset = fh.tell()
    state.setdefault('csv', {})[key] = {
        'byte_offset': end_offset,
        'buffer': [v.tolist() for v in _split_rows(flat, _batch_size_from_file(csv_path))],
        'total_rows': total,
    }
    return flat, total

  new_rows, new_offset, n_new = _read_rows_from_offset(
      csv_path, offset, header_cache)
  for row in new_rows:
    buffer.append(row.tolist())
  if len(buffer) > last_n_batches:
    buffer = buffer[-last_n_batches:]
  total_rows = int(entry.get('total_rows', 0)) + n_new
  state.setdefault('csv', {})[key] = {
      'byte_offset': new_offset,
      'buffer': buffer,
      'total_rows': total_rows,
  }
  if not buffer:
    return np.array([], dtype=np.float64), total_rows
  return np.concatenate([np.asarray(b, dtype=np.float64) for b in buffer]), total_rows


def _batch_size_from_file(csv_path: Path) -> int:
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    header = next(csv.reader(fh), [])
  w_cols = [c for c in header if c.startswith('w_')]
  return len(w_cols) if w_cols else 256


def _split_rows(flat: np.ndarray, batch_size: int) -> list[np.ndarray]:
  if flat.size == 0 or batch_size <= 0:
    return []
  n = flat.size // batch_size
  if n == 0:
    return [flat]
  trimmed = flat[: n * batch_size]
  return [trimmed[i * batch_size:(i + 1) * batch_size] for i in range(n)]


def _read_eval_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
  if not csv_path.is_file():
    return np.array([]), np.array([])
  steps, success = [], []
  with csv_path.open('r', encoding='utf-8', newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      if not row or 'success_1000' not in row:
        continue
      try:
        step = float(row.get(
            'actor_steps',
            row.get('learner_steps', row.get('iteration', len(steps)))))
        val = float(row['success_1000'])
      except (TypeError, ValueError):
        continue
      steps.append(step)
      success.append(val)
  return np.asarray(steps, dtype=np.float64), np.asarray(success, dtype=np.float64)


def _tail_logits(csv_path: Path, last_n_rows: int) -> np.ndarray:
  if not csv_path.is_file():
    return np.array([], dtype=np.float64)
  header = _ensure_header(csv_path, {})
  if not header:
    return np.array([], dtype=np.float64)
  vecs: list[np.ndarray] = []
  for line in _read_tail_lines(csv_path, last_n_rows):
    row = dict(zip(header, next(csv.reader([line]))))
    try:
      vecs.append(_parse_lw_row(row))
    except (KeyError, ValueError):
      continue
  if not vecs:
    return np.array([], dtype=np.float64)
  return np.concatenate(vecs)


def _atomic_savefig(fig: plt.Figure, out: Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.stem + '.tmp' + out.suffix)
  fig.savefig(str(tmp), dpi=150, bbox_inches='tight')
  if not tmp.is_file():
    raise RuntimeError(f'savefig did not create {tmp}')
  os.replace(str(tmp), str(out))


def _plot(
    *,
    user_w: np.ndarray,
    ref_w: np.ndarray,
    ref_logits: np.ndarray,
    ref_steps: np.ndarray,
    ref_success: np.ndarray,
    user_steps: np.ndarray,
    user_success: np.ndarray,
    user_rows: int,
    ref_rows: int,
    last_n_batches: int,
    user_label: str,
    ref_label: str,
    out_path: Path,
) -> None:
  fig = plt.figure(figsize=(12, 7))
  gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.38, wspace=0.28)

  ax0 = fig.add_subplot(gs[0, 0])
  bins = 40
  if user_w.size:
    ax0.hist(
        user_w, bins=bins, alpha=0.55, color='#1f77b4', edgecolor='white',
        label=f'{user_label}\nmean={user_w.mean():.5f} std={user_w.std():.5f}')
  if ref_w.size:
    ax0.hist(
        ref_w, bins=bins, alpha=0.55, color='#ff7f0e', edgecolor='white',
        label=f'{ref_label}\nmean={ref_w.mean():.5f} std={ref_w.std():.5f}')
  ax0.set_xlabel('w_diag (softmax IS weight)')
  ax0.set_ylabel('count')
  ax0.set_title(f'w_diag distribution (last {last_n_batches} batches)')
  ax0.legend(fontsize=8, loc='upper right')
  ax0.grid(True, alpha=0.25)

  ax1 = fig.add_subplot(gs[0, 1])
  if ref_logits.size:
    ax1.hist(
        ref_logits, bins=bins, color='#2ca02c', edgecolor='white', alpha=0.9)
    ax1.set_title(
        f'ref logits_w diag\nmean={ref_logits.mean():.3f}  std={ref_logits.std():.3f}')
  else:
    ax1.text(0.5, 0.5, 'no ref logits_w yet', ha='center', va='center',
             transform=ax1.transAxes)
  ax1.set_xlabel('logits_w diagonal (pre-softmax, min-Q)')
  ax1.set_ylabel('count')
  ax1.grid(True, alpha=0.25)

  ax2 = fig.add_subplot(gs[1, :])
  if ref_steps.size:
    ax2.plot(ref_steps, ref_success, color='#ff7f0e', lw=1.5, marker='o',
             ms=3, label='ref success_1000')
  if user_steps.size:
    ax2.plot(user_steps, user_success, color='#1f77b4', lw=1.0, alpha=0.65,
             label='user success_1000')
  if ref_steps.size or user_steps.size:
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel('actor steps')
    ax2.set_ylabel('success_1000')
    ax2.set_title('Eval success (ref fetch_reach should rise toward 1)')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.25)
  else:
    ax2.text(
        0.5, 0.5,
        'Waiting for ref logs/evaluator/logs.csv\n'
        '(from official lp_td_infonce.py run)',
        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
    ax2.set_axis_off()

  ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  title_env = 'official ref fetch_reach'
  if user_w.size:
    title_env += '  (+ optional user overlay)'
  fig.suptitle(
      f'TD InfoNCE w_diag — {title_env}  |  updated {ts}\n'
      f'ref rows={ref_rows}' + (f'  user rows={user_rows}' if user_w.size else ''),
      fontsize=11,
  )
  _atomic_savefig(fig, out_path)
  plt.close(fig)


def run_once(
    *,
    user_csv: Path | None,
    ref_csv: Path,
    ref_logits_csv: Path,
    ref_eval_csv: Path,
    user_eval_csv: Path | None,
    out_path: Path,
    state_path: Path,
    last_n_batches: int,
    user_label: str,
    ref_label: str,
) -> int:
  state = _load_state(state_path)
  header_cache: dict[str, list[str]] = {}

  user_w = np.array([], dtype=np.float64)
  user_rows = 0
  if user_csv is not None:
    user_w, user_rows = _accumulate_w(
        user_csv, last_n_batches, state, header_cache)
  ref_w, ref_rows = _accumulate_w(ref_csv, last_n_batches, state, header_cache)

  ref_logits = _tail_logits(ref_logits_csv, last_n_batches)
  ref_steps, ref_success = _read_eval_series(ref_eval_csv)
  user_steps, user_success = (
      _read_eval_series(user_eval_csv)
      if user_eval_csv is not None else (np.array([]), np.array([])))

  if user_w.size == 0 and ref_w.size == 0:
    print('No w_diag data yet in ref CSV' +
          (' or user CSV.' if user_csv else '.'))
    return 0

  _plot(
      user_w=user_w,
      ref_w=ref_w,
      ref_logits=ref_logits,
      ref_steps=ref_steps,
      ref_success=ref_success,
      user_steps=user_steps,
      user_success=user_success,
      user_rows=user_rows,
      ref_rows=ref_rows,
      last_n_batches=last_n_batches,
      user_label=user_label,
      ref_label=ref_label,
      out_path=out_path,
  )

  state['last_update'] = datetime.now().isoformat(timespec='seconds')
  _save_state(state_path, state)

  u_mean = float(user_w.mean()) if user_w.size else float('nan')
  r_mean = float(ref_w.mean()) if ref_w.size else float('nan')
  ref_s = float(ref_success[-1]) if ref_success.size else float('nan')
  user_part = (
      f'user w mean={u_mean:.6f} ({user_rows} rows)  '
      if user_w.size else '')
  print(
      f'[{datetime.now().strftime("%H:%M:%S")}] '
      f'{user_part}'
      f'ref w mean={r_mean:.6f} ({ref_rows} rows)  '
      f'ref success_1000={ref_s:.3f}  → {out_path}',
      flush=True,
  )
  return 0


def main() -> None:
  ap = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument('--user_csv', type=Path, default=DEFAULT_USER_CSV,
                  help='Optional PPO w_diag CSV to overlay (different env OK).')
  ap.add_argument('--ref_csv', type=Path, default=DEFAULT_REF_CSV)
  ap.add_argument('--ref_logits_csv', type=Path, default=DEFAULT_REF_LOGITS_CSV)
  ap.add_argument('--ref_eval_csv', type=Path, default=DEFAULT_REF_EVAL_CSV)
  ap.add_argument('--user_eval_csv', type=Path, default=DEFAULT_USER_EVAL_CSV)
  ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
  ap.add_argument('--state', type=Path, default=DEFAULT_STATE)
  ap.add_argument('--last_n_batches', type=int, default=80)
  ap.add_argument('--user_label', default='your PPO TD-InfoNCE')
  ap.add_argument('--ref_label', default='ref fetch_reach (official README)')
  ap.add_argument('--watch', action='store_true')
  ap.add_argument('--poll_interval', type=float, default=30.0)
  ap.add_argument('--once', action='store_true')
  args = ap.parse_args()

  kwargs = dict(
      user_csv=args.user_csv,
      ref_csv=args.ref_csv,
      ref_logits_csv=args.ref_logits_csv,
      ref_eval_csv=args.ref_eval_csv,
      user_eval_csv=args.user_eval_csv,
      out_path=args.out,
      state_path=args.state,
      last_n_batches=args.last_n_batches,
      user_label=args.user_label,
      ref_label=args.ref_label,
  )

  if args.watch:
    print(f'Watching ref={args.ref_csv}')
    if args.user_csv:
      print(f'         user={args.user_csv}')
    print(f'Output   {args.out}')
    print(f'Poll every {args.poll_interval:.0f}s  (Ctrl+C to stop)')
    try:
      while True:
        run_once(**kwargs)
        time.sleep(max(5.0, args.poll_interval))
    except KeyboardInterrupt:
      print('\nStopped watch.')
  else:
    raise SystemExit(run_once(**kwargs))


if __name__ == '__main__':
  main()
