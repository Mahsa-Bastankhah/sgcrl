#!/usr/bin/env python3
"""Login-node watcher: sbatch T=150 stoch eval when new q30 ep70 ckpts appear.

Does not use a GPU. Stop with the printed PID / watch-max-sec.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(
    REPO,
    'logs/ppo_allegro_kuka_nf_keeparm_bucket_ep70_500m_q30shift_seed0'
    '/ppo_allegro_kuka_throw_0/checkpoints')
OUT_DIR = os.path.join(
    REPO, 'figs/allegro_kuka_throw/q30_ep70_ckpt_eval_t150_stoch')
CSV_PATH = os.path.join(OUT_DIR, 'ckpt_eval_t150_stoch.csv')
PLOT_PATH = os.path.join(OUT_DIR, 'ckpt_eval_t150_stoch.png')
PLOT_TITLE = (
    'q30 keep-arm NVIDIA-bucket  train T=70  eval T=150 stoch  (roll mean w=5)')
JOB = os.path.join(
    REPO, 'jobs/job_allegro_q30_ep70_ckpt_eval_t150_stoch.slurm')
JOB_NAME = 'akt_q30_ep70_ckpt_eval_t150'
VIDEO_JOB = os.path.join(
    REPO, 'jobs/job_allegro_q30_ep70_ckpt_videos_t150_stoch.slurm')
VIDEO_JOB_NAME = 'akt_q30_ep70_ckpt_vid_t150'
VIDEO_DIR = os.path.join(
    REPO,
    'logs/ppo_allegro_kuka_nf_keeparm_bucket_ep70_500m_q30shift_seed0'
    '/ppo_allegro_kuka_throw_0/videos_eval_t150_stoch')
TRAIN_NAME = 'akt_nf_keeparm_nvbucket_q30_ep70_s0'
CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')
LAST_N = 10


def _ckpt_iters() -> list[int]:
  paths = sorted(glob.glob(os.path.join(CKPT_DIR, 'ckpt_iter_*.pkl')))
  iters = []
  for path in paths:
    match = CKPT_RE.search(os.path.basename(path))
    if match:
      iters.append(int(match.group(1)))
  return iters


def _done_iters() -> set[int]:
  if not os.path.isfile(CSV_PATH):
    return set()
  with open(CSV_PATH, newline='') as fh:
    return {int(r['iteration']) for r in csv.DictReader(fh) if r.get('iteration')}


def _squeue_names() -> set[str]:
  out = subprocess.check_output(
      ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%j'],
      text=True)
  return {line.strip() for line in out.splitlines() if line.strip()}


def _refresh_plot() -> None:
  if not os.path.isfile(CSV_PATH):
    return
  import sys
  sys.path.insert(0, os.path.join(REPO, 'scripts'))
  import eval_allegro_ckpt_success as ev  # noqa: E402
  rows = ev._load_csv_rows(CSV_PATH)
  if not rows:
    return
  ev._plot(PLOT_PATH, rows, 150, title=PLOT_TITLE)
  print(f'plotted n={len(rows)} → {PLOT_PATH}', flush=True)


def _pending_last_n() -> list[int]:
  iters = _ckpt_iters()
  if not iters:
    return []
  return [i for i in iters[-LAST_N:] if i not in _done_iters()]


def _pending_videos() -> list[int]:
  iters = _ckpt_iters()
  if not iters:
    return []
  pending = []
  for it in iters[-LAST_N:]:
    path = os.path.join(VIDEO_DIR, f'iter_{it:06d}_t150_stoch.mp4')
    if not os.path.isfile(path):
      pending.append(it)
  return pending


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--interval', type=int, default=180)
  parser.add_argument('--watch-max-sec', type=int, default=0)
  args = parser.parse_args()
  t0 = time.time()
  print(
      f'[q30 ckpt eval watch] every {args.interval}s '
      f'(max {args.watch_max_sec or "∞"}s) last-n={LAST_N}',
      flush=True)
  while True:
    names = _squeue_names()
    pending = _pending_last_n()
    pending_vid = _pending_videos()
    eval_running = JOB_NAME in names
    video_running = VIDEO_JOB_NAME in names
    train_running = TRAIN_NAME in names
    print(
        f'pending={pending} pending_vid={pending_vid} '
        f'eval_job={eval_running} vid_job={video_running} '
        f'train={train_running} done={len(_done_iters())} '
        f'ckpts={len(_ckpt_iters())}',
        flush=True)
    try:
      _refresh_plot()
    except Exception as exc:  # noqa: BLE001
      print(f'plot error: {exc}', flush=True)
    if pending and not eval_running:
      out = subprocess.check_output(['sbatch', JOB], text=True).strip()
      print(out, flush=True)
    if pending_vid and not video_running:
      out = subprocess.check_output(['sbatch', VIDEO_JOB], text=True).strip()
      print(out, flush=True)
    if (not train_running and not pending and not pending_vid
        and not eval_running and not video_running and _done_iters()):
      print('[q30 ckpt eval watch] train done and csv/videos caught up; exiting',
            flush=True)
      return
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[q30 ckpt eval watch] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.interval)


if __name__ == '__main__':
  main()
