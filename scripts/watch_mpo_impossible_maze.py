#!/usr/bin/env python3
"""Login-node watcher: MPO point_Impossible trajectory PNGs for new ckpts.

Watches the uniform-neg Impossible runs from
``slurm/mpo_crl_impossible_*`` and renders new ``ckpt_iter_*.pkl`` files
with ``scripts/mpo_rollout_maze.py`` on CPU (no GPU job).

Default runs::

  logs/mpo_crl_impossible_uniform_neg/mpo_crl_point_Impossible_{0,1}

Outputs: ``plots/mpo_impossible_uniform_neg/mpo_point_Impossible_<seed>/``
State:   ``plots/mpo_impossible_uniform_neg/.watch_state.json``

Examples::

  python scripts/watch_mpo_impossible_maze.py --once
  python scripts/watch_mpo_impossible_maze.py --watch_interval 300
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
from typing import Dict, List, Optional, Set, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOT_ROOT = os.path.join(_REPO, 'plots', 'mpo_impossible_uniform_neg')
_DEFAULT_STATE = os.path.join(_PLOT_ROOT, '.watch_state.json')
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')
_RENDER = os.path.join(_REPO, 'scripts', 'mpo_rollout_maze.py')

_DEFAULT_RUNS: List[dict] = [
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_impossible_uniform_neg/mpo_crl_point_Impossible_0'),
        'seed': 0,
        'short': 'mpo_imp_uneg',
        'log': 'slurm/mpo_crl_impossible_3550178_0.log',
    },
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_impossible_uniform_neg/mpo_crl_point_Impossible_1'),
        'seed': 1,
        'short': 'mpo_imp_uneg',
        'log': 'slurm/mpo_crl_impossible_3550178_1.log',
    },
]


def _load_state(path: str) -> dict:
  if not os.path.isfile(path):
    return {'runs': {}}
  with open(path, 'r', encoding='utf-8') as fh:
    data = json.load(fh)
  if 'runs' not in data:
    data = {'runs': data}
  return data


def _save_state(path: str, state: dict) -> None:
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  tmp = path + '.tmp'
  with open(tmp, 'w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)
  os.replace(tmp, path)


def _run_state(state: dict, run_dir: str) -> dict:
  runs = state.setdefault('runs', {})
  if run_dir not in runs:
    runs[run_dir] = {
        'done_labels': [],
        'bootstrapped': False,
        'last_action': None,
    }
  return runs[run_dir]


def list_checkpoint_labels(ckpt_dir: str) -> List[str]:
  if not os.path.isdir(ckpt_dir):
    return []
  out: List[Tuple[int, str]] = []
  for path in glob.glob(os.path.join(ckpt_dir, 'ckpt_iter_*.pkl')):
    m = _CKPT_RE.search(os.path.basename(path))
    if m:
      out.append((int(m.group(1)), f'iter_{int(m.group(1)):07d}'))
  out.sort()
  return [lab for _, lab in out]


def plot_labels_on_disk(out_dir: str, env: str = 'point_Impossible') -> Set[str]:
  if not os.path.isdir(out_dir):
    return set()
  labels: Set[str] = set()
  prefix = f'{env}_'
  for path in glob.glob(os.path.join(out_dir, f'{env}_*.png')):
    base = os.path.basename(path)
    if not base.startswith(prefix) or not base.endswith('.png'):
      continue
    lab = base[len(prefix):-len('.png')]
    if lab.startswith('iter_') or lab == 'latest':
      labels.add(lab)
  return labels


def render_labels(run_dir: str, out_dir: str, seed: int,
                  labels: List[str]) -> int:
  """Render specific checkpoint labels; return subprocess rc."""
  ckpt_dir = os.path.join(run_dir, 'checkpoints')
  paths = []
  for lab in labels:
    m = re.fullmatch(r'iter_(\d+)', lab)
    if not m:
      continue
    p = os.path.join(ckpt_dir, f'ckpt_iter_{int(m.group(1)):07d}.pkl')
    if os.path.isfile(p):
      paths.append(p)
  if not paths:
    return 0
  os.makedirs(out_dir, exist_ok=True)
  # Stage symlink farm so mpo_rollout_maze only sees requested ckpts.
  stage = os.path.join(out_dir, f'.stage_{os.getpid()}')
  os.makedirs(stage, exist_ok=True)
  try:
    for p in paths:
      dst = os.path.join(stage, os.path.basename(p))
      if not os.path.exists(dst):
        os.symlink(os.path.abspath(p), dst)
    # Also link run_config resolution via realpath (script follows symlinks).
    cmd = [
        sys.executable, '-u', _RENDER,
        '--checkpoint', stage,
        '--env', 'point_Impossible',
        '--output', out_dir + os.sep,
        '--seed', str(seed),
        '--num_trajectories', '5',
        '--fig_scale', '2.0',
        '--skip_existing',
    ]
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'MPLBACKEND': 'Agg',
           'PYTHONUNBUFFERED': '1'}
    print(f'[watch_mpo_imp] {" ".join(cmd)}', flush=True)
    return subprocess.call(cmd, cwd=_REPO, env=env)
  finally:
    for name in os.listdir(stage):
      try:
        os.unlink(os.path.join(stage, name))
      except OSError:
        pass
    try:
      os.rmdir(stage)
    except OSError:
      pass


def handle_run(item: dict, rs: dict, labels: List[str]) -> str:
  run_dir = item['run_dir']
  seed = int(item['seed'])
  short = f"{item['short']}_s{seed}"
  out_dir = os.path.join(_PLOT_ROOT, f'mpo_point_Impossible_{seed}')
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(out_dir, '.run_dir'), 'w', encoding='utf-8') as fh:
    fh.write(run_dir + '\n')

  disk = {lab for lab in plot_labels_on_disk(out_dir) if lab.startswith('iter_')}
  if disk:
    rs['done_labels'] = sorted(disk | set(rs.get('done_labels') or []))

  if not rs.get('bootstrapped') and not disk:
    rs['bootstrapped'] = True
    if not labels:
      return 'bootstrap(empty)'
    latest = labels[-1]
    rs['done_labels'] = [lab for lab in labels if lab != latest]
    rc = render_labels(run_dir, out_dir, seed, [latest])
    disk = {lab for lab in plot_labels_on_disk(out_dir) if lab.startswith('iter_')}
    rs['done_labels'] = sorted(set(rs.get('done_labels') or []) | disk)
    rs['last_action'] = f'bootstrap_latest:{latest}:rc={rc}'
    return f'bootstrap_latest:{latest}:rc={rc}'

  done = set(rs.get('done_labels') or [])
  new = [lab for lab in labels if lab not in done]
  if not new:
    return f'ok(new=0)'

  batch = new[:4]
  rc = render_labels(run_dir, out_dir, seed, batch)
  disk = {lab for lab in plot_labels_on_disk(out_dir) if lab.startswith('iter_')}
  rs['done_labels'] = sorted(done | disk)
  rs['last_action'] = f'render:{",".join(batch)}:rc={rc}'
  return f'render:n={len(batch)}/{len(new)}:rc={rc}'


def scan_once(state: dict) -> None:
  os.makedirs(_PLOT_ROOT, exist_ok=True)
  for item in _DEFAULT_RUNS:
    run_dir = os.path.abspath(item['run_dir'])
    if not os.path.isdir(run_dir):
      print(f'[watch_mpo_imp] missing run_dir (waiting): {run_dir}', flush=True)
      continue
    rs = _run_state(state, run_dir)
    labels = list_checkpoint_labels(os.path.join(run_dir, 'checkpoints'))
    action = handle_run({**item, 'run_dir': run_dir}, rs, labels)
    print(
        f'[watch_mpo_imp] {item["short"]}_s{item["seed"]}: '
        f'ckpts={len(labels)} → {action}',
        flush=True)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--once', action='store_true')
  ap.add_argument('--watch_interval', type=int, default=300)
  ap.add_argument('--state', default=_DEFAULT_STATE)
  args = ap.parse_args()

  state = _load_state(args.state)
  if args.once:
    scan_once(state)
    _save_state(args.state, state)
    return

  print(f'[watch_mpo_imp] daemon interval={args.watch_interval}s '
        f'state={args.state}', flush=True)
  while True:
    try:
      scan_once(state)
      _save_state(args.state, state)
    except Exception as exc:  # noqa: BLE001
      print(f'[watch_mpo_imp] scan error: {exc!r}', flush=True)
    time.sleep(max(30, int(args.watch_interval)))


if __name__ == '__main__':
  main()
