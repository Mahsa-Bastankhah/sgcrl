#!/usr/bin/env python3
"""Plot Sawyer bin tiny-NF eval success (single unified config).

Merges the canonical norand log dir with the legacy `_norand_noobs` dir
(same recipe: ppo_norm_obs=false by default; the noobs tag was redundant).

Reads logs/eval/logs.csv success_1000 (falls back to success).

Usage:
  python scripts/plot_sawyer_bin_tiny_nf_success.py
  python scripts/plot_sawyer_bin_tiny_nf_success.py --watch 900
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO = '/n/fs/mislresearch/sgcrl'
FIGS_DIR = os.path.join(REPO, 'figs', 'metaworld', 'bin_tiny_nf')
_SEED_RE = re.compile(r'^ppo_.+_(\d+)$')

# num_envs=4 × rollout_length=256
STEPS_PER_ITER = 1024

# One config; multiple log roots so legacy `_noobs` trials count too.
RUN = (
    'bin_tiny_nf_norand',
    [
        'logs/ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_minstd1e5_extrew1_norand',
        'logs/ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_minstd1e5_extrew1_norand_noobs',
    ],
    'Sawyer bin · tiny NF · τ=0.85 · crl10 · norand',
    '#2F6FED',
)


def _coerce(v) -> Optional[float]:
  if v is None:
    return None
  s = str(v).strip()
  if not s:
    return None
  try:
    x = float(s)
  except ValueError:
    return None
  if math.isnan(x) or math.isinf(x):
    return None
  return x


def _fmt_steps(v, _):
  if v >= 1e6:
    s = f'{v / 1e6:.1f}M'
    return s.replace('.0M', 'M')
  if v >= 1e3:
    return f'{v / 1e3:.0f}K'
  return str(int(v))


def seed_from_run_dir(run_dir: str) -> int:
  m = _SEED_RE.match(os.path.basename(run_dir))
  return int(m.group(1)) if m else -1


def load_eval_success(run_dir: str) -> List[Tuple[float, float]]:
  path = os.path.join(run_dir, 'logs', 'eval', 'logs.csv')
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return []
  pts: List[Tuple[float, float]] = []
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    if not reader.fieldnames:
      return []
    fields = set(reader.fieldnames)
    y_key = 'success_1000' if 'success_1000' in fields else (
        'success' if 'success' in fields else None)
    if y_key is None:
      return []
    for row in reader:
      y = _coerce(row.get(y_key))
      if y is None:
        continue
      it = _coerce(row.get('iteration'))
      if it is None:
        it = _coerce(row.get('learner_steps'))
      if it is None:
        continue
      pts.append((float(it) * STEPS_PER_ITER, y))
  pts.sort(key=lambda p: p[0])
  return pts


def collect_series(log_dirs: List[str]) -> Dict[str, List[Tuple[float, float]]]:
  """Load all trials; keys are unique even when seed numbers collide."""
  series: Dict[str, List[Tuple[float, float]]] = {}
  for log_dir in log_dirs:
    abs_log = log_dir if os.path.isabs(log_dir) else os.path.join(REPO, log_dir)
    if not os.path.isdir(abs_log):
      continue
    tag = os.path.basename(abs_log.rstrip('/'))
    for run_dir in sorted(glob.glob(os.path.join(abs_log, 'ppo_*'))):
      if not os.path.isdir(run_dir):
        continue
      pts = load_eval_success(run_dir)
      if not pts:
        continue
      seed = seed_from_run_dir(run_dir)
      # Disambiguate duplicate seed=0 across legacy log roots.
      key = f'{seed}'
      if key in series:
        key = f'{seed}@{tag}'
      series[key] = pts
  return series


def plot_variant(
    key: str, log_dirs: List[str], title: str, color: str, figs_dir: str,
) -> Optional[str]:
  series = collect_series(log_dirs)
  if not series:
    joined = ', '.join(log_dirs)
    print(f'[{key}] no eval data yet under {joined}')
    return None

  all_x = sorted({x for pts in series.values() for x, _ in pts})
  xs = np.asarray(all_x, dtype=np.float64)
  mats = []
  trial_order = sorted(series, key=lambda k: (k.split('@')[0], k))
  for trial in trial_order:
    px = np.asarray([p[0] for p in series[trial]], dtype=np.float64)
    py = np.asarray([p[1] for p in series[trial]], dtype=np.float64)
    mats.append(np.interp(xs, px, py, left=np.nan, right=np.nan))
  mat = np.vstack(mats)
  mean = np.nanmean(mat, axis=0)
  n = mat.shape[0]
  se = np.zeros_like(mean)
  if n > 1:
    counts = np.sum(~np.isnan(mat), axis=0)
    ok = counts >= 2
    if np.any(ok):
      std = np.nanstd(mat[:, ok], axis=0, ddof=1)
      se[ok] = std / np.sqrt(counts[ok].astype(np.float64))

  os.makedirs(figs_dir, exist_ok=True)
  out_path = os.path.join(figs_dir, f'{key}_eval_success.png')

  fig, ax = plt.subplots(figsize=(8.0, 4.5))
  if n > 1 and np.nanmax(se) > 0:
    ax.fill_between(xs, mean - se, mean + se, color=color, alpha=0.28,
                    linewidth=0, zorder=2, label='±stderr')
  for trial in trial_order:
    px = np.asarray([p[0] for p in series[trial]], dtype=np.float64)
    py = np.asarray([p[1] for p in series[trial]], dtype=np.float64)
    ax.plot(px, py, color=color, lw=1.0, alpha=0.35, zorder=2)
  ax.plot(xs, mean, color=color, lw=2.2, zorder=3,
          label=f'mean success_1000 (n={n} run{"s" if n != 1 else ""})')
  ax.set_ylim(-0.05, 1.05)
  ax.set_xlabel('env steps')
  ax.set_ylabel('eval success_1000')
  ax.set_title(title)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  ax.grid(True, alpha=0.25)
  ax.legend(loc='best', frameon=False, fontsize=9)
  fig.tight_layout()
  fig.savefig(out_path, dpi=140)
  plt.close(fig)

  lasts = {t: series[t][-1][1] for t in trial_order}
  peaks = {t: max(y for _, y in series[t]) for t in trial_order}
  print(f'[{key}] wrote {out_path}  (trials={trial_order}, '
        f'points={[len(series[t]) for t in trial_order]}, '
        f'last={lasts}, peak={peaks})')
  return out_path


def run_once(figs_dir: str = FIGS_DIR) -> list[str]:
  outs: list[str] = []
  key, log_dirs, title, color = RUN
  path = plot_variant(key, log_dirs, title, color, figs_dir)
  if path:
    outs.append(path)
  stamp = os.path.join(figs_dir, 'LAST_UPDATE.txt')
  os.makedirs(figs_dir, exist_ok=True)
  with open(stamp, 'w', encoding='utf-8') as fh:
    fh.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    for p in outs:
      fh.write(p + '\n')
  print(f'done: {len(outs)} plot(s) → {figs_dir}')
  return outs


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--figs_dir', default=FIGS_DIR)
  p.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only login-node watcher).')
  args = p.parse_args()

  if args.watch <= 0:
    run_once(args.figs_dir)
    return

  print(f'[bin_tiny_nf] watching every {args.watch}s → {args.figs_dir}',
        flush=True)
  while True:
    try:
      outs = run_once(args.figs_dir)
      print(f'[bin_tiny_nf] refreshed {len(outs)} plot(s)', flush=True)
    except Exception as exc:  # noqa: BLE001 — keep watcher alive
      print(f'[bin_tiny_nf] error: {exc}', flush=True)
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
