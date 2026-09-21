#!/usr/bin/env python3
"""Offline stochastic eval of every 7.5 cm (goal_ball) Allegro throw run.

Training success is the 7.5 cm ball. This rollout scores the easier NVIDIA
in-bucket cylinder (r=0.12, h=0.198) and appends it to a per-run CSV.

  python scripts/eval_allegro_goalball_inbucket.py
  python scripts/eval_allegro_goalball_inbucket.py --plot-only
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import plot_allegro_goalball_inbucket_eval as plot_mod  # noqa: E402

CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')
OUT_DIR = plot_mod.OUT_DIR


def _run_roots() -> list[str]:
  roots = []
  for path in sorted(glob.glob(os.path.join(REPO, 'logs', '*goalball75*'))):
    if os.path.isdir(path):
      roots.append(path)
  return roots


def _ckpt_dir(root: str) -> str:
  matches = glob.glob(os.path.join(root, '*', 'checkpoints'))
  matches = [p for p in matches if glob.glob(os.path.join(p, 'ckpt_iter_*.pkl'))]
  if not matches:
    return ''
  return sorted(matches)[0]


def _run_config(ckpt_dir: str) -> dict:
  run_dir = os.path.dirname(ckpt_dir)
  for name in ('run_config.json', 'flags.json'):
    path = os.path.join(run_dir, name)
    if os.path.isfile(path):
      with open(path, encoding='utf-8') as fh:
        data = json.load(fh)
      if isinstance(data, dict) and 'flags' in data:
        return data
      return {'flags': data} if isinstance(data, dict) else {}
  return {}


def _episode_length(cfg: dict) -> int:
  flags = cfg.get('flags') or {}
  resolved = cfg.get('resolved_config') or {}
  for key in ('isaacgym_episode_length', 'max_episode_steps'):
    val = resolved.get(key, flags.get(key))
    if val not in (None, '', -1):
      return int(val)
  return 50


def _csv_path(root: str) -> str:
  return os.path.join(OUT_DIR, os.path.basename(root.rstrip('/')) + '.csv')


def _done_iters(csv_path: str) -> set[int]:
  if not os.path.isfile(csv_path):
    return set()
  with open(csv_path, newline='') as fh:
    return {int(r['iteration']) for r in csv.DictReader(fh) if r.get('iteration')}


def _ckpt_iters(ckpt_dir: str) -> list[int]:
  iters = []
  for path in glob.glob(os.path.join(ckpt_dir, 'ckpt_iter_*.pkl')):
    match = CKPT_RE.search(os.path.basename(path))
    if match:
      iters.append(int(match.group(1)))
  return sorted(iters)


def discover() -> list[dict]:
  rows = []
  for root in _run_roots():
    ckpt_dir = _ckpt_dir(root)
    if not ckpt_dir:
      continue
    cfg = _run_config(ckpt_dir)
    flags = cfg.get('flags') or {}
    if str(flags.get('isaacgym_throw_success', 'goal_ball')) != 'goal_ball':
      continue
    ep = _episode_length(cfg)
    csv_path = _csv_path(root)
    iters = _ckpt_iters(ckpt_dir)
    pending = [i for i in iters if i not in _done_iters(csv_path)]
    rows.append({
        'root': root,
        'ckpt_dir': ckpt_dir,
        'csv': csv_path,
        'episode_length': ep,
        'iters': iters,
        'pending': pending,
    })
  return rows


def eval_run(run: dict) -> None:
  if not run['pending']:
    print(f"[goalball-inbucket] skip {os.path.basename(run['root'])} "
          f"(have {len(run['iters'])} ckpts)", flush=True)
    return
  os.makedirs(OUT_DIR, exist_ok=True)
  cmd = [
      sys.executable, '-u',
      os.path.join(REPO, 'scripts', 'eval_allegro_ckpt_success.py'),
      f"--checkpoint-dir={run['ckpt_dir']}",
      f"--output-csv={run['csv']}",
      f"--episode-length={int(run['episode_length'])}",
      '--num-envs=256',
      '--seed=0',
      '--pipeline=gpu',
      '--stochastic',
      f"--steps-per-iter={1024 * int(run['episode_length'])}",
  ]
  print(
      f"[goalball-inbucket] {os.path.basename(run['root'])} "
      f"T={run['episode_length']} pending={run['pending']}",
      flush=True)
  subprocess.check_call(cmd, cwd=REPO)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--plot-only', action='store_true')
  args = parser.parse_args()
  if args.plot_only:
    plot_mod.plot()
    return
  runs = discover()
  if not runs:
    print('[goalball-inbucket] no goalball75 runs with ckpts', flush=True)
    return
  for run in runs:
    eval_run(run)
  plot_mod.plot()


if __name__ == '__main__':
  main()
