#!/usr/bin/env python3
"""Render rollout videos for PPO sawyer_bin runs as checkpoints appear.

Watches:
  - logs/ppo_sawyer_bin/ppo_sawyer_bin_<seed>/checkpoints
  - logs/ppo_sawyer_bin_dirac_baseline/ppo_sawyer_bin_<seed>/checkpoints

When a new ``.pkl`` appears (or ``latest.pkl`` mtime changes), runs
``ppo_rollout_video.py`` for that file only.

State: ``videos/.watch_sawyer_bin_videos_state.json``

Examples::

  python scripts/watch_sawyer_bin_videos.py --once
  python scripts/watch_sawyer_bin_videos.py --watch_interval 120
  python scripts/watch_sawyer_bin_videos.py --seeds 123,124
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ENV = 'sawyer_bin'
RUNS = (
    ('default', 'ppo_sawyer_bin'),
    ('dirac_baseline', 'ppo_sawyer_bin_dirac_baseline'),
    ('nouniform', 'ppo_sawyer_bin_nouniform'),
)
RUN_DIR_RE = re.compile(r'^ppo_sawyer_bin_(\d+)$')


def _ckpt_label(pkl_path: str) -> str:
  """Match ppo_rollout_video._enumerate_checkpoints labels."""
  base = os.path.splitext(os.path.basename(pkl_path))[0]
  m = re.search(r'ckpt_iter_(\d+)\.pkl$', pkl_path)
  if m:
    return f'iter_{int(m.group(1)):07d}'
  if base == 'latest':
    return 'latest'
  return base[len('ckpt_'):] if base.startswith('ckpt_') else base


def _list_checkpoints(ckpt_dir: str) -> list[tuple[str, str, float]]:
  """(label, path, mtime) for each .pkl in dir."""
  out = []
  for p in sorted(glob.glob(os.path.join(ckpt_dir, '*.pkl'))):
    if not os.path.isfile(p):
      continue
    out.append((_ckpt_label(p), p, os.path.getmtime(p)))
  return out


def _state_key(log_tag: str, seed: int, label: str) -> str:
  return f'{log_tag}|{seed}|{label}'


def _load_state(path: str) -> dict:
  if not os.path.isfile(path):
    return {}
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def _save_state(path: str, state: dict) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  with open(path, 'w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)


def _expected_mp4(out_dir: str, label: str) -> str:
  return os.path.join(out_dir, f'{ENV}_{label}.mp4')


def _discover_runs(repo: str, log_root: str, seeds: set[int] | None) -> list[tuple[int, str]]:
  """(seed, ckpt_dir) for each ppo_sawyer_bin_<seed> under logs/<log_root>."""
  base = os.path.join(repo, 'logs', log_root)
  if not os.path.isdir(base):
    return []
  found = []
  for name in sorted(os.listdir(base)):
    m = RUN_DIR_RE.match(name)
    if not m:
      continue
    seed = int(m.group(1))
    if seeds is not None and seed not in seeds:
      continue
    ckpt_dir = os.path.join(base, name, 'checkpoints')
    if os.path.isdir(ckpt_dir):
      found.append((seed, ckpt_dir))
  return found


def _render_one(repo: str, pkl_path: str, out_dir: str, seed: int) -> int:
  os.makedirs(out_dir, exist_ok=True)
  out_arg = out_dir if out_dir.endswith(os.sep) else out_dir + os.sep
  cmd = [
      sys.executable,
      '-u',
      os.path.join(repo, 'ppo_rollout_video.py'),
      '--checkpoint',
      pkl_path,
      '--env',
      ENV,
      '--output',
      out_arg,
      '--seed',
      str(seed),
  ]
  env = {**os.environ}
  env.setdefault('JAX_PLATFORMS', 'cpu')
  env.setdefault('MUJOCO_GL', 'osmesa')
  env.setdefault('MUJOCO_PY_MUJOCO_PATH', os.path.expanduser('~/.mujoco/mujoco210'))
  mujoco_bin = os.path.expanduser('~/.mujoco/mujoco210/bin')
  extra_libs = [mujoco_bin, '/usr/lib/nvidia']
  existing = env.get('LD_LIBRARY_PATH', '')
  parts = [p for p in existing.split(':') if p]
  for lib in extra_libs:
    if lib not in parts:
      parts.append(lib)
  env['LD_LIBRARY_PATH'] = ':'.join(parts)
  print(f'[watch_sawyer] {" ".join(cmd)}', flush=True)
  return subprocess.call(cmd, cwd=repo, env=env)


def _scan_and_render(
    repo: str,
    out_base: str,
    state_path: str,
    force: bool,
    seeds: set[int] | None,
) -> int:
  state = _load_state(state_path)
  n_new = 0
  for log_tag, log_root in RUNS:
    for seed, ckpt_dir in _discover_runs(repo, log_root, seeds):
      out_dir = os.path.join(out_base, log_tag, f'ppo_sawyer_bin_{seed}')
      for label, pkl_path, mtime in _list_checkpoints(ckpt_dir):
        key = _state_key(log_tag, seed, label)
        prev = state.get(key)
        out_mp4 = _expected_mp4(out_dir, label)
        needs = force or prev is None or float(prev) < mtime
        if not needs and os.path.isfile(out_mp4):
          continue
        if not needs:
          needs = not os.path.isfile(out_mp4)
        if not needs:
          continue
        print(
            f'[watch_sawyer] new/changed: {log_tag} seed={seed} {label} ({pkl_path})',
            flush=True,
        )
        rc = _render_one(repo, pkl_path, out_dir, seed)
        if rc != 0:
          print(
              f'[watch_sawyer] WARNING: render failed rc={rc} for {pkl_path}',
              flush=True,
          )
          continue
        state[key] = mtime
        n_new += 1
        print(f'[watch_sawyer] saved {out_mp4}', flush=True)
  _save_state(state_path, state)
  return n_new


def _parse_seeds(s: str | None) -> set[int] | None:
  if not s:
    return None
  return {int(x.strip()) for x in s.split(',') if x.strip()}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--repo_root', default='.',
      help='Repo root (default: cwd).')
  ap.add_argument(
      '--output_base',
      default='videos/sawyer_bin/ppo_sawyer_bin_all_ckpts',
      help='MP4 output root (subdirs: default/, dirac_baseline/).')
  ap.add_argument(
      '--state_file',
      default='videos/sawyer_bin/.watch_sawyer_bin_videos_state.json',
      help='JSON tracking rendered checkpoint mtimes.')
  ap.add_argument(
      '--seeds',
      default=None,
      help='Comma-separated seeds to watch (default: all runs found).')
  ap.add_argument(
      '--once', action='store_true',
      help='Single scan then exit.')
  ap.add_argument(
      '--force', action='store_true',
      help='Re-render all checkpoints regardless of state.')
  ap.add_argument(
      '--watch_interval', type=float, default=120.0,
      help='Seconds between scans (default: 120).')
  args = ap.parse_args()

  repo = os.path.abspath(os.path.expanduser(args.repo_root))
  out_base = os.path.join(repo, args.output_base)
  state_path = os.path.join(repo, args.state_file)
  seeds = _parse_seeds(args.seeds)

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[watch_sawyer] === scan @ {ts} ===', flush=True)
      n = _scan_and_render(repo, out_base, state_path, args.force, seeds)
      print(f'[watch_sawyer] rendered {n} checkpoint(s)', flush=True)
      if args.once:
        break
      delay = max(30.0, float(args.watch_interval))
      print(f'[watch_sawyer] next scan in {delay:.0f}s', flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[watch_sawyer] stopped.', flush=True)


if __name__ == '__main__':
  main()
