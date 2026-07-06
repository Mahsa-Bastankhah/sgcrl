#!/usr/bin/env python3
"""Live dashboard for TD InfoNCE w_diag and a' action histograms.

Training appends rows to two side CSVs under each run:

  logs/<run>/logs/w_diag/vectors.csv       — batch IS weights w_0 … w_{B-1}
  logs/<run>/logs/a_prime_hist/vectors.csv — action bins d{d}_b{b} per dim

This script reads new rows and refreshes a combined PNG.  Use ``--watch``
during training (polls every few seconds).

Examples::

  # Auto-find the newest run and watch live.
  python scripts/watch_td_infonce_stats.py --watch --find_latest

  # Explicit run directory.
  python scripts/watch_td_infonce_stats.py --watch \\
      --run_dir logs/ppo_td_infonce_drawer_norm16/ppo_sawyer_drawer_open_0

  # One-shot snapshot of whatever is logged so far.
  python scripts/watch_td_infonce_stats.py --once --find_latest
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
DEFAULT_OUT = ROOT / 'figs' / 'td_infonce_stats_live.png'
DEFAULT_STATE = ROOT / 'figs' / '.watch_td_infonce_stats_state.json'


def _find_latest_run_dir(logs_root: Path) -> Path | None:
  candidates = sorted(
      logs_root.glob('**/logs/w_diag/vectors.csv'),
      key=lambda p: p.stat().st_mtime,
      reverse=True,
  )
  if not candidates:
    return None
  # .../ppo_env_seed/logs/w_diag/vectors.csv → run dir is parents[2]
  return candidates[0].parents[2]


def _run_paths(run_dir: Path) -> tuple[Path, Path]:
  w_csv = run_dir / 'logs' / 'w_diag' / 'vectors.csv'
  a_csv = run_dir / 'logs' / 'a_prime_hist' / 'vectors.csv'
  return w_csv, a_csv


def _load_state(path: Path) -> dict:
  if not path.is_file():
    return {}
  with path.open('r', encoding='utf-8') as fh:
    return json.load(fh)


def _save_state(path: Path, state: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)


def _parse_w_row(row: dict) -> tuple[int, int, np.ndarray]:
  step_key = 'iteration' if 'iteration' in row else 'learner_step'
  iteration = int(float(row[step_key]))
  crl_step = int(float(row['crl_step']))
  w_cols = sorted(
      (k for k in row if k.startswith('w_')),
      key=lambda k: int(k.split('_', 1)[1]),
  )
  w = np.array([float(row[k]) for k in w_cols], dtype=np.float64)
  return iteration, crl_step, w


def _parse_a_prime_row(row: dict) -> tuple[int, int, np.ndarray]:
  step_key = 'iteration' if 'iteration' in row else 'learner_step'
  iteration = int(float(row[step_key]))
  crl_step = int(float(row['crl_step']))
  d_cols = sorted(
      (k for k in row if k.startswith('d') and '_b' in k),
      key=lambda k: (int(k.split('_b')[0][1:]), int(k.split('_b')[1])),
  )
  if not d_cols:
    raise KeyError('no d*_b* columns')
  n_bins = max(int(k.split('_b')[1]) for k in d_cols) + 1
  n_dims = max(int(k.split('_b')[0][1:]) for k in d_cols) + 1
  hist = np.zeros((n_dims, n_bins), dtype=np.float64)
  for k in d_cols:
    d = int(k.split('_b')[0][1:])
    b = int(k.split('_b')[1])
    hist[d, b] = float(row[k])
  return iteration, crl_step, hist


_HEADER: dict[str, list[str]] = {}


def _sanitize_bytes(raw: bytes) -> str:
  return raw.replace(b'\0', b'').decode('utf-8', errors='replace').strip()


def _ensure_header(csv_path: Path) -> None:
  key = str(csv_path)
  if key in _HEADER:
    return
  with csv_path.open('rb') as fh:
    header_line = _sanitize_bytes(fh.readline())
  if header_line:
    _HEADER[key] = next(csv.reader([header_line]))


def _row_dict(line: str, header: list[str]) -> dict | None:
  if not line or line.startswith('iteration') or line.startswith('learner_step'):
    return None
  try:
    values = next(csv.reader([line]))
  except csv.Error:
    return None
  if len(values) < 2:
    return None
  if len(values) < len(header):
    values = values + [''] * (len(header) - len(values))
  return dict(zip(header, values[:len(header)]))


def _read_tail_row(csv_path: Path, parser) -> tuple | None:
  """Read the last complete CSV row (robust to NUL bytes / concurrent writes)."""
  if not csv_path.is_file():
    return None
  _ensure_header(csv_path)
  header = _HEADER[str(csv_path)]
  with csv_path.open('rb') as fh:
    fh.seek(0, os.SEEK_END)
    pos = fh.tell()
    chunk = b''
    while pos > 0 and chunk.count(b'\n') < 2:
      step = min(65536, pos)
      pos -= step
      fh.seek(pos)
      chunk = fh.read(step) + chunk
  lines = [_sanitize_bytes(ln) for ln in chunk.splitlines() if _sanitize_bytes(ln)]
  for line in reversed(lines):
    row = _row_dict(line, header)
    if row is None:
      continue
    try:
      return parser(row)
    except (KeyError, ValueError):
      continue
  return None


def _parse_w_summary(row: dict) -> tuple[int, int, float, float, float, float]:
  step_key = 'iteration' if 'iteration' in row else 'learner_step'
  iteration = int(float(row[step_key]))
  crl_step = int(float(row['crl_step']))
  w_cols = sorted(
      (k for k in row if k.startswith('w_')),
      key=lambda k: int(k.split('_', 1)[1]),
  )
  w = np.array([float(row[k]) for k in w_cols], dtype=np.float64)
  return (
      iteration, crl_step,
      float(np.mean(w)), float(np.std(w)),
      float(np.min(w)), float(np.max(w)),
  )


def _read_new_rows(
    csv_path: Path,
    byte_offset: int,
    parser,
    *,
    summary_parser=None,
    keep_full: str = 'last',
) -> tuple[list, list, int]:
  """Return (full_rows, summary_rows, new_byte_offset).

  ``summary_parser`` avoids storing large vectors for every row when catching up.
  ``keep_full='last'`` only materialises the parser result for the final new row.
  """
  if not csv_path.is_file():
    return [], [], byte_offset
  _ensure_header(csv_path)
  header = _HEADER[str(csv_path)]
  full_rows: list = []
  summary_rows: list = []
  new_offset = byte_offset
  with csv_path.open('rb') as fh:
    fh.seek(byte_offset)
    while True:
      raw = fh.readline()
      if not raw:
        new_offset = fh.tell()
        break
      line = _sanitize_bytes(raw)
      if not line:
        new_offset = fh.tell()
        continue
      row = _row_dict(line, header)
      if row is None:
        new_offset = fh.tell()
        continue
      try:
        if summary_parser is not None:
          summary_rows.append(summary_parser(row))
        if keep_full == 'all':
          full_rows.append(parser(row))
        elif keep_full == 'last':
          full_rows = [parser(row)]
      except (KeyError, ValueError):
        pass
      new_offset = fh.tell()
  if keep_full == 'last' and summary_rows and not full_rows:
    # parser failed but summary worked — drop broken tail row
    summary_rows.pop()
  return full_rows, summary_rows, new_offset


def _atomic_savefig(fig: plt.Figure, out: Path) -> None:
  out.parent.mkdir(parents=True, exist_ok=True)
  tmp = out.with_name(out.stem + '.tmp' + out.suffix)
  fig.savefig(str(tmp), dpi=150, bbox_inches='tight')
  if not tmp.is_file():
    raise RuntimeError(f'savefig did not create {tmp}')
  os.replace(str(tmp), str(out))


def _plot_dashboard(
    *,
    run_dir: Path,
    out_path: Path,
    latest_w: np.ndarray | None,
    w_iteration: int,
    w_crl_step: int,
    w_history: list[tuple[int, int, float, float, float, float]],
    latest_hist: np.ndarray | None,
    a_iteration: int,
    a_crl_step: int,
    action_low: float,
    action_high: float,
    w_total_rows: int,
    a_total_rows: int,
) -> None:
  n_a_dims = 0 if latest_hist is None else latest_hist.shape[0]
  n_a_cols = min(4, max(1, n_a_dims))
  n_a_plot_rows = int(np.ceil(n_a_dims / n_a_cols)) if n_a_dims else 0
  fig_h = 6.0 + 2.5 * n_a_plot_rows
  fig = plt.figure(figsize=(12, fig_h))
  top = fig.add_gridspec(1, 2, top=0.92, bottom=0.55 if n_a_plot_rows else 0.12)

  ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  fig.suptitle(
      f'TD InfoNCE stats — {run_dir.name}\n'
      f'updated {ts}  ·  w rows={w_total_rows}  ·  a′ rows={a_total_rows}',
      fontsize=11,
  )

  ax_w_hist = fig.add_subplot(top[0, 0])
  if latest_w is not None and latest_w.size:
    ax_w_hist.hist(
        latest_w,
        bins=min(50, max(10, len(latest_w) // 4)),
        color='#4C9BE8', edgecolor='white', alpha=0.9)
    ax_w_hist.axvline(float(np.mean(latest_w)), color='#E74C3C', lw=1.5,
                      label=f'mean={np.mean(latest_w):.5f}')
    ax_w_hist.axvline(float(np.median(latest_w)), color='#2ECC71', lw=1.5, ls='--',
                      label=f'median={np.median(latest_w):.5f}')
    ax_w_hist.legend(fontsize=8, loc='upper right')
    ax_w_hist.set_title(f'w_diag batch  iter={w_iteration}  crl={w_crl_step}')
  else:
    ax_w_hist.text(0.5, 0.5, 'no w_diag rows yet', ha='center', va='center',
                   transform=ax_w_hist.transAxes)
  ax_w_hist.set_xlabel('w_diag[i]')
  ax_w_hist.set_ylabel('count')
  ax_w_hist.grid(True, alpha=0.25)

  ax_w_ts = fig.add_subplot(top[0, 1])
  if w_history:
    steps = np.arange(len(w_history))
    means = [h[2] for h in w_history]
    stds = [h[3] for h in w_history]
    ax_w_ts.plot(steps, means, color='#4C9BE8', lw=0.8, label='batch mean')
    ax_w_ts.fill_between(
        steps,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        color='#4C9BE8', alpha=0.2, label='mean ± std')
    ax_w_ts.set_xlabel('CRL update index')
    ax_w_ts.set_ylabel('w_diag')
    ax_w_ts.set_title(f'w_diag over time ({len(w_history)} updates)')
    ax_w_ts.legend(fontsize=8, loc='upper right')
    ax_w_ts.grid(True, alpha=0.25)
  else:
    ax_w_ts.text(0.5, 0.5, 'no w history yet', ha='center', va='center',
                 transform=ax_w_ts.transAxes)

  if latest_hist is not None and n_a_dims:
    n_bins = latest_hist.shape[1]
    edges = np.linspace(action_low, action_high, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bottom_gs = fig.add_gridspec(
        n_a_plot_rows, n_a_cols,
        top=0.50, bottom=0.08, hspace=0.35, wspace=0.25)
    for d in range(n_a_dims):
      r, c = divmod(d, n_a_cols)
      ax = fig.add_subplot(bottom_gs[r, c])
      ax.bar(centers, latest_hist[d], width=(edges[1] - edges[0]) * 0.9,
             color='#9B59B6', edgecolor='white', alpha=0.9)
      ax.set_title(f"a′ dim {d}", fontsize=9)
      ax.set_xlim(action_low, action_high)
      ax.grid(True, alpha=0.2)
    batch_n = int(latest_hist.sum()) // max(n_bins, 1)
    fig.text(
        0.5, 0.52,
        f"a′ histograms  iter={a_iteration}  crl={a_crl_step}  "
        f'({n_bins} bins/dim; phase 1 = uniform a′, phase 3 = policy a′)',
        ha='center', fontsize=9)

  _atomic_savefig(fig, out_path)
  plt.close(fig)


def _append_w_history(
    history: list[tuple[int, int, float, float, float, float]],
    iteration: int,
    crl_step: int,
    w: np.ndarray,
) -> None:
  history.append((
      iteration, crl_step,
      float(np.mean(w)), float(np.std(w)),
      float(np.min(w)), float(np.max(w)),
  ))


def run_once(
    run_dir: Path,
    out_path: Path,
    state_path: Path,
    *,
    action_low: float,
    action_high: float,
    tail_only: bool = False,
) -> int:
  w_csv, a_csv = _run_paths(run_dir)
  state = _load_state(state_path)
  if state.get('run_dir') != str(run_dir):
    state = {}

  w_history: list = list(state.get('w_history', []))
  w_offset = int(state.get('w_byte_offset', 0))
  a_offset = int(state.get('a_byte_offset', 0))
  w_total = int(state.get('w_total_rows', 0))
  a_total = int(state.get('a_total_rows', 0))

  if tail_only:
    w_tail = _read_tail_row(w_csv, _parse_w_row)
    a_tail = _read_tail_row(a_csv, _parse_a_prime_row)
    if w_tail:
      latest_w, w_iter, w_crl = w_tail[2], w_tail[0], w_tail[1]
      s = _parse_w_summary({'iteration': str(w_iter), 'crl_step': str(w_crl),
                            **{f'w_{i}': str(v) for i, v in enumerate(latest_w)}})
      if not w_history or w_history[-1][:2] != (w_iter, w_crl):
        w_history.append(s)
        w_total += 1
    else:
      latest_w = None
      w_iter = w_crl = -1
    if a_tail:
      latest_hist, a_iter, a_crl = a_tail[2], a_tail[0], a_tail[1]
      a_total = max(a_total, a_total + 1) if a_total else 1
    else:
      latest_hist = None
      a_iter = a_crl = -1
    w_offset = w_csv.stat().st_size if w_csv.is_file() else w_offset
    a_offset = a_csv.stat().st_size if a_csv.is_file() else a_offset
  else:
    w_full, w_summaries, w_offset = _read_new_rows(
        w_csv, w_offset, _parse_w_row,
        summary_parser=_parse_w_summary, keep_full='last')
    a_full, _, a_offset = _read_new_rows(
        a_csv, a_offset, _parse_a_prime_row, keep_full='last')
    a_rows = a_full

    w_history.extend(w_summaries)
    w_total += len(w_summaries)
    a_total += len(a_full)

    if w_full:
      latest_w = w_full[-1][2]
      w_iter, w_crl = w_full[-1][0], w_full[-1][1]
    elif w_summaries:
      tail = _read_tail_row(w_csv, _parse_w_row)
      if tail:
        latest_w, w_iter, w_crl = tail[2], tail[0], tail[1]
      else:
        latest_w = np.array(state.get('latest_w', []), dtype=np.float64)
        latest_w = latest_w if latest_w.size else None
        s = w_summaries[-1]
        w_iter, w_crl = s[0], s[1]
    else:
      latest_w = np.array(state.get('latest_w', []), dtype=np.float64)
      latest_w = latest_w if latest_w.size else None
      w_iter = int(state.get('w_iteration', -1))
      w_crl = int(state.get('w_crl_step', -1))

    if a_rows:
      latest_hist = a_rows[-1][2]
      a_iter, a_crl = a_rows[-1][0], a_rows[-1][1]
    else:
      tail = _read_tail_row(a_csv, _parse_a_prime_row)
      if tail:
        latest_hist, a_iter, a_crl = tail[2], tail[0], tail[1]
      else:
        cached = state.get('latest_hist')
        latest_hist = np.array(cached, dtype=np.float64) if cached else None
        a_iter = int(state.get('a_iteration', -1))
        a_crl = int(state.get('a_crl_step', -1))

  if latest_w is None and not w_history and not tail_only:
    print(f'No data yet under {run_dir}')
    return 0

  _plot_dashboard(
      run_dir=run_dir,
      out_path=out_path,
      latest_w=latest_w,
      w_iteration=w_iter,
      w_crl_step=w_crl,
      w_history=w_history,
      latest_hist=latest_hist,
      a_iteration=a_iter,
      a_crl_step=a_crl,
      action_low=action_low,
      action_high=action_high,
      w_total_rows=w_total,
      a_total_rows=a_total,
  )

  _save_state(state_path, {
      'run_dir': str(run_dir),
      'w_byte_offset': w_offset,
      'a_byte_offset': a_offset,
      'w_total_rows': w_total,
      'a_total_rows': a_total,
      'w_history': w_history[-2000:],
      'latest_w': latest_w.tolist() if latest_w is not None else [],
      'w_iteration': w_iter,
      'w_crl_step': w_crl,
      'latest_hist': latest_hist.tolist() if latest_hist is not None else [],
      'a_iteration': a_iter,
      'a_crl_step': a_crl,
      'last_update': datetime.now().isoformat(timespec='seconds'),
  })

  if w_history or latest_hist is not None:
    w_msg = ''
    if w_history:
      h = w_history[-1]
      w_msg = (f'w_diag mean={h[2]:.6f} std={h[3]:.6f} '
               f'iter={h[0]} crl={h[1]}')
    a_msg = ''
    if latest_hist is not None:
      a_msg = f"a′ logged iter={a_iter} crl={a_crl}"
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {w_msg}  {a_msg}  → {out_path}',
          flush=True)
  return 0


def run_watch(
    run_dir: Path,
    out_path: Path,
    state_path: Path,
    poll_interval: float,
    *,
    action_low: float,
    action_high: float,
) -> None:
  print(f'Run dir   {run_dir}')
  w_csv, a_csv = _run_paths(run_dir)
  print(f'w_diag    {w_csv}')
  print(f"a′ hist   {a_csv}")
  print(f'Output    {out_path}')
  print(f'Poll every {poll_interval:.1f}s  (Ctrl+C to stop)')
  try:
    while True:
      if w_csv.is_file() or a_csv.is_file():
        run_once(run_dir, out_path, state_path,
                 action_low=action_low, action_high=action_high)
      else:
        print(f'[{datetime.now().strftime("%H:%M:%S")}] waiting for CSV logs...')
      time.sleep(max(1.0, poll_interval))
  except KeyboardInterrupt:
    print('\nStopped watch.')


def main() -> None:
  ap = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  ap.add_argument(
      '--run_dir', type=Path, default=None,
      help='Run directory, e.g. logs/.../ppo_sawyer_drawer_open_0')
  ap.add_argument(
      '--find_latest', action='store_true',
      help='Use the run with the newest w_diag/vectors.csv under logs/.')
  ap.add_argument(
      '--out', type=Path, default=DEFAULT_OUT,
      help=f'Output PNG (default: {DEFAULT_OUT})')
  ap.add_argument(
      '--state', type=Path, default=DEFAULT_STATE,
      help='JSON state file (byte offsets + w history)')
  ap.add_argument('--watch', action='store_true',
                  help='Poll and refresh the figure during training.')
  ap.add_argument('--once', action='store_true',
                  help='Process pending rows once and exit.')
  ap.add_argument('--poll_interval', type=float, default=10.0,
                  help='Seconds between polls in --watch mode (default: 10).')
  ap.add_argument('--tail_only', action='store_true',
                  help='Only read the latest CSV row (fast refresh).')
  ap.add_argument('--action_low', type=float, default=-1.0)
  ap.add_argument('--action_high', type=float, default=1.0)
  args = ap.parse_args()

  run_dir = args.run_dir
  if args.find_latest or run_dir is None:
    run_dir = _find_latest_run_dir(ROOT / 'logs')
    if run_dir is None:
      ap.error('No w_diag/vectors.csv found under logs/; pass --run_dir.')
    print(f'Using latest run: {run_dir}')

  if args.watch:
    run_watch(run_dir, args.out, args.state, args.poll_interval,
              action_low=args.action_low, action_high=args.action_high)
  else:
    raise SystemExit(run_once(run_dir, args.out, args.state,
                              action_low=args.action_low,
                              action_high=args.action_high,
                              tail_only=args.tail_only))


if __name__ == '__main__':
  main()
