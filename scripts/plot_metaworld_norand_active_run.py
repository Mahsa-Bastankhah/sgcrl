#!/usr/bin/env python3
"""Plot Sawyer norand CRL eval success into figs/metaworld/active_run/.

Tracks:
  logs/ppo_{bin,peg,box}_crl_crl10_40m_minstd1e5_extrew1_norand/

One PNG per env (mean ± stderr across available seeds). Uses the rolling
``success_1000`` metric (mean of last ≤1000 eval episodes) so curves are not
pointy discrete 0/0.2/…/1 spikes. CPU-only: reads logs/eval/logs.csv.

Usage:
  python scripts/plot_metaworld_norand_active_run.py
  python scripts/plot_metaworld_norand_active_run.py --watch 3600
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
FIGS_DIR = os.path.join(REPO, 'figs', 'metaworld', 'active_run')
_SEED_RE = re.compile(r'^ppo_.+_(\d+)$')

# Steps per PPO iteration for these runs (num_envs=4 × rollout_length=256).
STEPS_PER_ITER = 1024

RUNS = [
    ('bin', 'logs/ppo_bin_crl_crl10_40m_minstd1e5_extrew1_norand',
     'Sawyer bin · CRL norand · minstd1e-5 · extrew1'),
    ('peg', 'logs/ppo_peg_crl_crl10_40m_minstd1e5_extrew1_norand',
     'Sawyer peg · CRL norand · minstd1e-5 · extrew1'),
    ('box', 'logs/ppo_box_crl_crl10_40m_minstd1e5_extrew1_norand',
     'Sawyer box · CRL norand · minstd1e-5 · extrew1'),
]


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
  """Return [(env_steps, success_1000)] from logs/eval/logs.csv.

  Prefers the rolling ``success_1000`` average; falls back to raw ``success``.
  """
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


def plot_env(env_key: str, log_dir: str, title: str, figs_dir: str) -> Optional[str]:
  abs_log = log_dir if os.path.isabs(log_dir) else os.path.join(REPO, log_dir)
  series: Dict[int, List[Tuple[float, float]]] = {}
  for run_dir in sorted(glob.glob(os.path.join(abs_log, 'ppo_*'))):
    if not os.path.isdir(run_dir):
      continue
    seed = seed_from_run_dir(run_dir)
    pts = load_eval_success(run_dir)
    if pts:
      series[seed] = pts
  if not series:
    print(f'[{env_key}] no eval data yet under {abs_log}')
    return None

  all_x = sorted({x for pts in series.values() for x, _ in pts})
  xs = np.asarray(all_x, dtype=np.float64)
  mats = []
  seed_order = sorted(series)
  for seed in seed_order:
    px = np.asarray([p[0] for p in series[seed]], dtype=np.float64)
    py = np.asarray([p[1] for p in series[seed]], dtype=np.float64)
    mats.append(np.interp(xs, px, py, left=np.nan, right=np.nan))
  mat = np.vstack(mats)
  mean = np.nanmean(mat, axis=0)
  n = mat.shape[0]
  if n > 1:
    se = np.nanstd(mat, axis=0, ddof=1) / math.sqrt(n)
  else:
    se = np.zeros_like(mean)

  os.makedirs(figs_dir, exist_ok=True)
  out_path = os.path.join(figs_dir, f'{env_key}_eval_success.png')

  fig, ax = plt.subplots(figsize=(8.0, 4.5))
  color = '#2F6FED'
  # Shade first so the mean line sits on top.
  if n > 1 and np.nanmax(se) > 0:
    ax.fill_between(xs, mean - se, mean + se, color=color, alpha=0.28,
                    linewidth=0, zorder=2, label='±stderr')
  ax.plot(xs, mean, color=color, lw=2.2, zorder=3,
          label=f'mean success_1000 (n={n} seed{"s" if n != 1 else ""})')
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
  print(f'[{env_key}] wrote {out_path}  (seeds={seed_order}, '
        f'points={[len(series[s]) for s in seed_order]})')
  return out_path


def run_once(figs_dir: str = FIGS_DIR) -> list[str]:
  outs: list[str] = []
  for env_key, log_dir, title in RUNS:
    path = plot_env(env_key, log_dir, title, figs_dir)
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

  print(f'[active_run] watching every {args.watch}s → {args.figs_dir}',
        flush=True)
  while True:
    try:
      outs = run_once(args.figs_dir)
      print(f'[active_run] refreshed {len(outs)} plot(s)', flush=True)
    except Exception as exc:  # noqa: BLE001 — keep watcher alive
      print(f'[active_run] error: {exc}', flush=True)
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
