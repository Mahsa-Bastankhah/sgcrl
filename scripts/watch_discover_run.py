#!/usr/bin/env python3
"""Login-node watcher for a DISCOVER CSV: print latest row + plot train/eval.

  python scripts/watch_discover_run.py \\
      --csv logs/discover_.../discover_builderbench_creative_2_task1_0/logs.csv \\
      --interval 60 --max_minutes 55
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _read(path: Path):
  if not path.is_file() or path.stat().st_size == 0:
    return []
  with path.open() as f:
    return list(csv.DictReader(f))


def _plot(rows, out: Path):
  if len(rows) < 2:
    return
  xs = [float(r['global_step']) for r in rows]
  tr = [float(r['train_success_1000']) for r in rows]
  ev = [float(r['success']) for r in rows]
  fig, ax = plt.subplots(figsize=(7.2, 4.2))
  ax.plot(xs, tr, color='#4C9BE8', lw=1.8, label='train_success_1000 (BB task)')
  ax.plot(xs, ev, color='#E8834C', lw=1.8, label='eval success (rolling mean n/a; raw)')
  ax.set_xlabel('env steps')
  ax.set_ylabel('success')
  ax.set_ylim(-0.02, 1.02)
  ax.legend(frameon=False)
  ax.grid(True, alpha=0.25)
  out.parent.mkdir(parents=True, exist_ok=True)
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--csv', required=True)
  p.add_argument('--interval', type=int, default=60)
  p.add_argument('--max_minutes', type=float, default=55)
  args = p.parse_args()
  csv_path = Path(args.csv)
  out = csv_path.parent / 'train_eval_success.png'
  t0 = time.time()
  last_n = -1
  print(f'[watch_discover] csv={csv_path} max={args.max_minutes}m', flush=True)
  while True:
    rows = _read(csv_path)
    if len(rows) != last_n and rows:
      last = rows[-1]
      print(
          f"[watch_discover] n={len(rows)} step={last.get('global_step')} "
          f"train_success_1000={last.get('train_success_1000')} "
          f"eval_success={last.get('success')} "
          f"goal_hit={last.get('goal_hit_rate')} sps={last.get('sps')}",
          flush=True)
      try:
        _plot(rows, out)
        print(f'[watch_discover] plot={out}', flush=True)
      except Exception as exc:  # noqa: BLE001
        print(f'[watch_discover] plot failed: {exc}', flush=True)
      last_n = len(rows)
    elif not rows:
      print('[watch_discover] waiting for csv...', flush=True)
    if (time.time() - t0) / 60.0 >= args.max_minutes:
      print('[watch_discover] done (time cap)', flush=True)
      return
    time.sleep(args.interval)


if __name__ == '__main__':
  main()
