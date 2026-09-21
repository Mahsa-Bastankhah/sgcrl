#!/usr/bin/env python3
"""Plot DISCOVER eval success vs global_step (NF-comparable any-microstep).

Groups seeds that share the same task + recipe (e.g. all c3t1 catwp) and
plots mean ± stderr as a shaded band. Curves are also time-smoothed over a
centered env-step window.

``--carry-forward-peak`` freezes each seed at its first max success for all
later steps (early-stop-at-peak counterfactual) and skips time smoothing.

For BuilderBench CSVs, uses ``eval_success_any_microstep`` when present.
Sawyer uses ``success``.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]

OUT_DIR = _REPO / 'figs' / 'builderbench' / 'discover_eval'
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
DEFAULT_SMOOTH_STEPS = 5_000_000


def _group_key(csv_path: Path) -> str:
  """Task + recipe label shared across seeds."""
  run_dir = csv_path.parent
  log_dir = run_dir.parent.name
  name = run_dir.name
  m = re.search(r'creative_(\d+)_task(\d+)_\d+$', name)
  if m:
    tag = f'c{m.group(1)}t{m.group(2)}'
    if '_tanh_' in log_dir:
      tag += ' tanh'
    elif '_catwp_' in log_dir:
      tag += ' catwp'
    return tag
  m = re.search(r'discover_sawyer_(bin|peg)_\d+$', name)
  if m:
    return f'sawyer_{m.group(1)}'
  return name.rsplit('_', 1)[0]


def _load_series(csv_path: Path):
  with csv_path.open() as fh:
    rows = list(csv.DictReader(fh))
  if not rows:
    return [], []
  keys = rows[0].keys()
  ykey = (
      'eval_success_any_microstep'
      if 'eval_success_any_microstep' in keys else 'success')
  xs, ys = [], []
  for r in rows:
    try:
      x = float(r['global_step'])
      y = float(r[ykey])
      ep = int(float(r['epoch'])) if 'epoch' in r and r['epoch'] != '' else None
    except (KeyError, TypeError, ValueError):
      continue
    if y != y:
      continue
    path = csv_path.as_posix()
    if ep is not None and 'builderbench' in path:
      if not (ep == 1 or ep % 10 == 0):
        continue
    elif ep is not None and 'sawyer' in path:
      if not (ep == 1 or ep % 17 == 0):
        continue
    xs.append(x)
    ys.append(y)
  return xs, ys


def _rolling_mean_by_steps(xs, ys, window_steps: float):
  if not xs or window_steps <= 0:
    return list(ys)
  half = 0.5 * float(window_steps)
  out = []
  n = len(xs)
  lo = 0
  hi = 0
  s = 0.0
  for i, x in enumerate(xs):
    while hi < n and xs[hi] <= x + half:
      s += ys[hi]
      hi += 1
    while lo < hi and xs[lo] < x - half:
      s -= ys[lo]
      lo += 1
    cnt = hi - lo
    out.append(s / float(cnt) if cnt else ys[i])
  return out


def _carry_forward_peak(xs, ys):
  """Running max: hold each seed at best success seen so far.

  Counterfactual of early-stopping whenever eval peaks (and never going
  back down). Returns (xs, ys_cummax, (step_at_global_max, global_max)).
  """
  if not ys:
    return list(xs), list(ys), None
  peak_i = int(np.argmax(ys))
  peak = float(ys[peak_i])
  out = []
  cur = float('-inf')
  for y in ys:
    cur = max(cur, float(y))
    out.append(cur)
  return list(xs), out, (float(xs[peak_i]), peak)


def _fmt_steps(n: float) -> str:
  if n >= 1e6:
    v = n / 1e6
    return f'{v:.0f}M' if abs(v - round(v)) < 1e-6 else f'{v:.1f}M'
  if n >= 1e3:
    return f'{n/1e3:.0f}k'
  return str(int(n))


def _aggregate_mean_stderr(seed_series: list[tuple[list[float], list[float]]]):
  """Interpolate seeds onto shared x-grid; return mean and stderr."""
  usable = [(np.asarray(xs, float), np.asarray(ys, float))
            for xs, ys in seed_series if len(xs) >= 2]
  if not usable:
    if seed_series and seed_series[0][0]:
      xs0, ys0 = seed_series[0]
      return list(xs0), list(ys0), [0.0] * len(xs0), 1
    return [], [], [], 0

  # Shared grid = sorted unique steps across seeds (cap density).
  all_x = np.unique(np.concatenate([xs for xs, _ in usable]))
  # Keep at most ~400 points for readability.
  if len(all_x) > 400:
    idx = np.linspace(0, len(all_x) - 1, 400).astype(int)
    all_x = all_x[idx]

  mat = []
  for xs, ys in usable:
    # Only interpolate within each seed's observed range.
    yi = np.interp(all_x, xs, ys, left=np.nan, right=np.nan)
    # Clip to [xmin, xmax] of this seed (interp already nan outside via left/right
    # only if we set nan — np.interp doesn't; manually mask).
    yi = np.where((all_x >= xs[0]) & (all_x <= xs[-1]),
                  np.interp(all_x, xs, ys), np.nan)
    mat.append(yi)
  mat = np.asarray(mat)  # (n_seeds, n_x)
  mean = np.nanmean(mat, axis=0)
  n = np.sum(~np.isnan(mat), axis=0).astype(float)
  # Sample std across seeds (only where n>1); stderr = std / sqrt(n).
  sq = np.nansum((mat - mean) ** 2, axis=0)
  with np.errstate(invalid='ignore', divide='ignore'):
    std = np.where(n > 1, np.sqrt(sq / np.maximum(n - 1.0, 1.0)), 0.0)
    se = np.where(n > 1, std / np.sqrt(n), 0.0)
  # Drop columns with no data.
  ok = n > 0
  return (all_x[ok].tolist(), mean[ok].tolist(), se[ok].tolist(),
          int(np.max(n)))


def main():
  p = argparse.ArgumentParser()
  p.add_argument(
      '--out', type=Path,
      default=OUT_DIR / 'discover_eval_success_any_microstep.png')
  p.add_argument(
      '--glob', default='logs/discover_*/discover_*/logs.csv')
  p.add_argument(
      '--smooth-steps', type=float, default=DEFAULT_SMOOTH_STEPS,
      help='Centered env-step window for time smoothing (default 5e6).')
  p.add_argument(
      '--band', choices=('stderr', 'std'), default='stderr',
      help='Shaded band: stderr (default) or std across seeds.')
  p.add_argument(
      '--carry-forward-peak', action='store_true',
      help='Per seed: running-max success (early-stop-at-best counterfactual).')
  p.add_argument(
      '--recipe', choices=('catwp', 'tanh', 'all'), default='catwp',
      help='BuilderBench recipe filter (default: catwp only).')
  p.add_argument(
      '--min-steps', type=float, default=1_000_000,
      help='Drop seeds whose last logged step is below this (default 1e6).')
  args = p.parse_args()

  csvs = sorted(_REPO.glob(args.glob)) if '*' in args.glob else [
      Path(args.glob) if Path(args.glob).is_absolute()
      else _REPO / args.glob]
  if not csvs:
    raise SystemExit(f'no CSVs matched {_REPO / args.glob}')

  groups: dict[str, list[tuple[list[float], list[float]]]] = defaultdict(list)
  peak_info: dict[str, list[tuple[float, float]]] = defaultdict(list)
  for csv_path in csvs:
    log_dir = csv_path.parent.parent.name
    if 'builderbench' in log_dir:
      if args.recipe == 'catwp' and '_catwp_' not in log_dir:
        continue
      if args.recipe == 'tanh' and '_tanh_' not in log_dir:
        continue
    xs, ys = _load_series(csv_path)
    if not xs:
      continue
    if xs[-1] < args.min_steps:
      print(f'skip short run ({xs[-1]:.0f} < {args.min_steps:.0f}): {csv_path}')
      continue
    key = _group_key(csv_path)
    # Drop recipe suffix from label when filtering to one recipe.
    if args.recipe == 'catwp' and key.endswith(' catwp'):
      key = key[: -len(' catwp')]
    elif args.recipe == 'tanh' and key.endswith(' tanh'):
      key = key[: -len(' tanh')]
    if args.carry_forward_peak:
      xs, ys, peak = _carry_forward_peak(xs, ys)
      if peak is not None:
        peak_info[key].append(peak)
    groups[key].append((xs, ys))

  if not groups:
    raise SystemExit('no series loaded')

  smooth_steps = args.smooth_steps
  wlabel = _fmt_steps(smooth_steps) if smooth_steps > 0 else None
  args.out.parent.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(9.5, 5.2))

  for i, key in enumerate(sorted(groups.keys())):
    series = groups[key]
    xs, mean, se, n_seeds = _aggregate_mean_stderr(series)
    if not xs:
      continue
    # For std band, recompute from se: se = std/sqrt(n) ⇒ std = se*sqrt(n)
    if args.band == 'std' and n_seeds > 1:
      band = [s * math.sqrt(n_seeds) for s in se]
      band_name = 'std'
    else:
      band = se
      band_name = 'stderr'

    color = COLORS[i % len(COLORS)]
    # Faint per-seed for context when n>1.
    if n_seeds > 1:
      for sxs, sys_ in series:
        sm_s = _rolling_mean_by_steps(sxs, sys_, smooth_steps)
        ax.plot(sxs, sm_s, color=color, linewidth=0.9, alpha=0.22,
                zorder=2 + i)

    sm = _rolling_mean_by_steps(xs, mean, smooth_steps)
    lo = _rolling_mean_by_steps(xs, [m - b for m, b in zip(mean, band)],
                                smooth_steps)
    hi = _rolling_mean_by_steps(xs, [m + b for m, b in zip(mean, band)],
                                smooth_steps)
    if n_seeds > 1 and any(b > 0 for b in band):
      ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0,
                      zorder=2 + i)
    label = f'{key} (n={n_seeds}, mean±{band_name})'
    ax.plot(xs, sm, color=color, linewidth=2.2, label=label, alpha=0.95,
            zorder=3 + i)

  ax.set_xlabel('env steps')
  if args.carry_forward_peak:
    ylab = 'eval success (running max / seed'
    if wlabel:
      ylab += f', then roll {wlabel}'
    ylab += ')'
    ax.set_ylabel(ylab)
    title = (
        f'DISCOVER eval success by task (mean±{args.band}; '
        f'carry-forward peak'
        + (f', smoothed {wlabel}' if wlabel else '')
        + ')')
  else:
    ax.set_ylabel(
        f'eval success any-microstep (time roll mean window={wlabel})')
    title = (
        f'DISCOVER eval success by task (mean±{args.band}; '
        f'time-smoothed {wlabel})')
  ax.set_ylim(-0.05, 1.05)
  ax.grid(True, alpha=0.3)
  ax.legend(loc='best', fontsize=8, framealpha=0.9)
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(args.out, dpi=160)
  pdf = args.out.with_suffix('.pdf')
  fig.savefig(pdf)
  print(f'wrote {args.out}')
  print(f'wrote {pdf}')
  for k, v in sorted(groups.items()):
    print(f'  {k}: {len(v)} seed(s)')
    for j, (px, py) in enumerate(peak_info.get(k, [])):
      print(f'    seed[{j}] peak={py:.3f} @ {px:.0f} steps')


if __name__ == '__main__':
  main()
