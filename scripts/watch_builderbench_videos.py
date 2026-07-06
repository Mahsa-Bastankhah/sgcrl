#!/usr/bin/env python3
"""Render BuilderBench rollout videos as checkpoints appear.

Watches ``logs/<log_root>/ppo_<env>_<seed>/checkpoints/*.pkl`` and runs
``ppo_builderbench_rollout_video.py`` when a file is new or its mtime
changes.

State: ``videos/builderbench/.watch_state.json``

Videos are written under ``videos/builderbench/<log_dir_basename>/``.

Examples::

  # Single run:
  python scripts/watch_builderbench_videos.py \\
      --run logs/ppo_builderbench_creative1_task2_e1024:builderbench_creative_1_task2:task2_e1024 \\
      --once

  # Multiple runs (one GPU job polls all):
  python scripts/watch_builderbench_videos.py \\
      --run logs/ppo_builderbench_creative2_task1_e256:builderbench_creative_2_task1:c2t1_e256 \\
      --run logs/ppo_builderbench_creative2_task1_e1024:builderbench_creative_2_task1:c2t1_e1024 \\
      --watch_interval 300
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
from dataclasses import dataclass

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)
from envs.builderbench_utils import (
    run_config_path,
    video_render_skip_reason,
)


def _ckpt_label(pkl_path: str) -> str:
  base = os.path.splitext(os.path.basename(pkl_path))[0]
  m = re.search(r'ckpt_iter_(\d+)\.pkl$', pkl_path)
  if m:
    return f'iter_{int(m.group(1)):07d}'
  if base == 'latest':
    return 'latest'
  return base[len('ckpt_'):] if base.startswith('ckpt_') else base


def _list_checkpoints(ckpt_dir: str) -> list[tuple[str, str, float]]:
  out = []
  for p in sorted(glob.glob(os.path.join(ckpt_dir, '*.pkl'))):
    if os.path.isfile(p):
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


def _expected_mp4(out_dir: str, run_tag: str, label: str) -> str:
  return os.path.join(out_dir, f'{run_tag}_{label}.mp4')


def _discover_ckpt_dirs(log_dir: str, env: str, seeds: set[int] | None):
  """Yield (seed, ckpt_dir) under logs/<log_dir>/ppo_<env>_<seed>/."""
  base = os.path.abspath(log_dir)
  if not os.path.isdir(base):
    return
  pattern = os.path.join(base, f'ppo_{env}_*', 'checkpoints')
  for ckpt_dir in sorted(glob.glob(pattern)):
    run_name = os.path.basename(os.path.dirname(ckpt_dir))
    m = re.fullmatch(rf'ppo_{re.escape(env)}_(\d+)', run_name)
    if not m:
      continue
    seed = int(m.group(1))
    if seeds is not None and seed not in seeds:
      continue
    yield seed, ckpt_dir


def _render_one(
    repo: str,
    pkl_path: str,
    out_dir: str,
    env: str,
    run_tag: str,
    seed: int,
) -> int:
  os.makedirs(out_dir, exist_ok=True)
  python = os.environ.get('CONDA_PREFIX')
  if python:
    python = os.path.join(python, 'bin', 'python')
  else:
    python = sys.executable
  cmd = [
      python,
      '-u',
      os.path.join(repo, 'scripts', 'ppo_builderbench_rollout_video.py'),
      '--checkpoint',
      pkl_path,
      '--env',
      env,
      '--output',
      out_dir if out_dir.endswith(os.sep) else out_dir + os.sep,
      '--run_tag',
      run_tag,
      '--seed',
      str(seed),
  ]
  env_vars = {**os.environ}
  env_vars.setdefault('JAX_PLATFORMS', 'cpu')
  env_vars.setdefault('MUJOCO_GL', 'egl')
  env_vars.setdefault('BUILDERBENCH_MJX_IMPL', 'jax')
  env_vars.setdefault(
      'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
  print(f'[watch_bb] {" ".join(cmd)}', flush=True)
  return subprocess.call(cmd, cwd=repo, env=env_vars)


_skip_logged_runs: set[str] = set()


def _scan_and_render(
    repo: str,
    run: _RunSpec,
    state: dict,
    state_path: str,
    force: bool,
    seeds: set[int] | None,
) -> int:
  log_tag = os.path.basename(os.path.normpath(run.log_dir))
  out_dir = run.out_dir or _default_out_dir(repo, run.log_dir)
  n_new = 0
  for seed, ckpt_dir in _discover_ckpt_dirs(run.log_dir, run.env, seeds):
    skip_key = f'{log_tag}|{seed}'
    skip_reason = video_render_skip_reason(
        run_config_path(run.log_dir, run.env, seed))
    if skip_reason is not None:
      if skip_key not in _skip_logged_runs:
        print(
            f'[watch_bb] skip {log_tag} seed={seed}: {skip_reason}',
            flush=True,
        )
        _skip_logged_runs.add(skip_key)
      continue
    for label, pkl_path, mtime in _list_checkpoints(ckpt_dir):
      key = _state_key(log_tag, seed, label)
      prev = state.get(key)
      out_mp4 = _expected_mp4(out_dir, run.run_tag, label)
      needs = force or prev is None or float(prev) < mtime
      if not needs and os.path.isfile(out_mp4):
        continue
      if not needs:
        needs = not os.path.isfile(out_mp4)
      if not needs:
        continue
      print(
          f'[watch_bb] new/changed: {log_tag} seed={seed} {label} ({pkl_path})',
          flush=True,
      )
      rc = _render_one(repo, pkl_path, out_dir, run.env, run.run_tag, seed)
      if rc != 0:
        print(
            f'[watch_bb] WARNING: render failed rc={rc} for {pkl_path}',
            flush=True,
        )
        continue
      state[key] = mtime
      n_new += 1
      print(f'[watch_bb] saved {out_mp4}', flush=True)
  _save_state(state_path, state)
  return n_new


def _parse_seeds(s: str | None) -> set[int] | None:
  if not s:
    return None
  return {int(x.strip()) for x in s.split(',') if x.strip()}


@dataclass
class _RunSpec:
  log_dir: str
  env: str
  run_tag: str
  out_dir: str | None = None


def _default_out_dir(repo: str, log_dir: str) -> str:
  log_tag = os.path.basename(os.path.normpath(log_dir))
  return os.path.join(repo, 'videos', 'builderbench', log_tag)


def _parse_run_spec(spec: str, repo: str) -> _RunSpec:
  parts = spec.split(':')
  if len(parts) not in (3, 4):
    raise ValueError(
        f'--run must be LOG_DIR:ENV:RUN_TAG[:OUTPUT_DIR], got {spec!r}')
  log_dir = parts[0]
  if not os.path.isabs(log_dir):
    log_dir = os.path.join(repo, log_dir)
  out_dir = None
  if len(parts) == 4:
    out_dir = parts[3]
    if not os.path.isabs(out_dir):
      out_dir = os.path.join(repo, out_dir)
  return _RunSpec(log_dir=log_dir, env=parts[1], run_tag=parts[2], out_dir=out_dir)


def _resolve_runs(args, repo: str) -> list[_RunSpec]:
  if args.run:
    return [_parse_run_spec(s, repo) for s in args.run]
  if not args.log_dir or not args.env or not args.run_tag:
    raise SystemExit('Provide --run LOG_DIR:ENV:RUN_TAG or --log_dir/--env/--run_tag')
  log_dir = args.log_dir
  if not os.path.isabs(log_dir):
    log_dir = os.path.join(repo, log_dir)
  out_dir = args.output_dir
  if out_dir and not os.path.isabs(out_dir):
    out_dir = os.path.join(repo, out_dir)
  elif out_dir is None:
    out_dir = _default_out_dir(repo, log_dir)
  return [_RunSpec(log_dir=log_dir, env=args.env, run_tag=args.run_tag, out_dir=out_dir)]


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--repo_root', default='.', help='Repo root (default: cwd).')
  ap.add_argument(
      '--run', action='append', default=None,
      help='LOG_DIR:ENV:RUN_TAG[:OUTPUT_DIR]. Repeat for multiple runs.')
  ap.add_argument(
      '--log_dir', default=None,
      help='Training log root (legacy single-run mode).')
  ap.add_argument(
      '--env', default=None,
      help='sgcrl env name (legacy single-run mode).')
  ap.add_argument(
      '--run_tag', default=None,
      help='MP4 filename prefix (legacy single-run mode).')
  ap.add_argument(
      '--output_dir', default=None,
      help='Video output dir (legacy single-run mode).')
  ap.add_argument(
      '--state_file',
      default='videos/builderbench/.watch_state.json',
      help='JSON tracking rendered checkpoint mtimes.')
  ap.add_argument(
      '--seeds', default=None,
      help='Comma-separated seeds (default: all runs found).')
  ap.add_argument('--once', action='store_true', help='Single scan then exit.')
  ap.add_argument(
      '--force', action='store_true',
      help='Re-render all checkpoints regardless of state.')
  ap.add_argument(
      '--watch_interval', type=float, default=300.0,
      help='Seconds between scans (default: 300).')
  args = ap.parse_args()

  repo = os.path.abspath(os.path.expanduser(args.repo_root))
  runs = _resolve_runs(args, repo)
  state_path = os.path.join(repo, args.state_file) if not os.path.isabs(args.state_file) else args.state_file
  seeds = _parse_seeds(args.seeds)

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[watch_bb] === scan @ {ts} ===', flush=True)
      state = _load_state(state_path)
      n_total = 0
      for run in runs:
        print(
            f'[watch_bb] --- {run.run_tag}  ({run.log_dir}) ---',
            flush=True,
        )
        n_total += _scan_and_render(
            repo, run, state, state_path, args.force, seeds)
      print(f'[watch_bb] rendered {n_total} checkpoint(s)', flush=True)
      if args.once:
        break
      delay = max(60.0, float(args.watch_interval))
      print(f'[watch_bb] next scan in {delay:.0f}s', flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[watch_bb] stopped.', flush=True)


if __name__ == '__main__':
  main()
