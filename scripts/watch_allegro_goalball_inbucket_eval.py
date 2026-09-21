#!/usr/bin/env python3
"""Login-node watcher: sbatch in-bucket eval when new 7.5cm-run ckpts appear.

Does not use a GPU. Stop with the printed PID / watch-max-sec.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import eval_allegro_goalball_inbucket as ev  # noqa: E402
import plot_allegro_goalball_inbucket_eval as plot_mod  # noqa: E402

JOB = os.path.join(REPO, 'jobs/job_allegro_goalball_inbucket_eval.slurm')
JOB_NAME = 'akt_goalball_inbucket_eval'


def _squeue_names() -> set[str]:
  out = subprocess.check_output(
      ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%j'],
      text=True)
  return {line.strip() for line in out.splitlines() if line.strip()}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--interval', type=int, default=180)
  parser.add_argument('--watch-max-sec', type=int, default=0)
  args = parser.parse_args()
  t0 = time.time()
  print(
      f'[goalball inbucket watch] every {args.interval}s '
      f'(max {args.watch_max_sec or "∞"}s)',
      flush=True)
  while True:
    runs = ev.discover()
    pending = [(os.path.basename(r['root']), r['pending']) for r in runs
               if r['pending']]
    names = _squeue_names()
    eval_running = JOB_NAME in names
    print(
        f'pending={pending} eval_job={eval_running} runs={len(runs)}',
        flush=True)
    if pending and not eval_running:
      subprocess.check_call(['sbatch', JOB], cwd=REPO)
      print(f'submitted {JOB_NAME}', flush=True)
    try:
      plot_mod.plot()
    except Exception as exc:  # noqa: BLE001
      print(f'plot skipped: {exc}', flush=True)
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[goalball inbucket watch] watch-max-sec reached; exiting',
            flush=True)
      return
    time.sleep(args.interval)


if __name__ == '__main__':
  main()
