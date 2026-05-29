#!/usr/bin/env python3
"""Plot stochastic WBC trajectories as new checkpoints appear.

Scans ``logs/wbc_eightrooms_v2/wbc_point_EightRooms_*/checkpoints`` and runs
``ppo_rollout_maze.py`` each cycle so new ``ckpt_iter_*.pkl`` files are picked
up automatically.

Examples::

  python scripts/watch_wbc_eightrooms_trajectories.py --once

  nohup python -u scripts/watch_wbc_eightrooms_trajectories.py \\
      --watch_interval 180 > slurm/watch_wbc_eightrooms_traj.out 2>&1 &

  JAX_PLATFORMS=cpu python scripts/watch_wbc_eightrooms_trajectories.py --once
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time


def _iter_checkpoint_dirs(log_roots: list[str]) -> list[str]:
    out: list[str] = []
    for root in log_roots:
        root = os.path.abspath(os.path.expanduser(root))
        if not os.path.isdir(root):
            continue
        pat = os.path.join(root, '**', 'wbc_point_EightRooms_*', 'checkpoints')
        for d in sorted(glob.glob(pat, recursive=True)):
            if not os.path.isdir(d):
                continue
            if not glob.glob(os.path.join(d, '*.pkl')):
                continue
            out.append(os.path.normpath(d))
    return sorted(set(out))


def _run_name_and_seed(ckpt_dir: str) -> tuple[str, int]:
    run_dir = os.path.dirname(ckpt_dir)
    run_name = os.path.basename(run_dir)
    m = re.search(r'wbc_point_EightRooms_(\d+)$', run_name)
    seed = int(m.group(1)) if m else 0
    return run_name, seed


def _out_subdir(out_base: str, ckpt_dir: str) -> str:
    run_name, _ = _run_name_and_seed(ckpt_dir)
    return os.path.join(out_base, run_name)


def _render_one(repo_root: str, ckpt_dir: str, out_sub: str) -> int:
    os.makedirs(out_sub, exist_ok=True)
    _, seed = _run_name_and_seed(ckpt_dir)
    cmd = [
        sys.executable,
        '-u',
        os.path.join(repo_root, 'ppo_rollout_maze.py'),
        '--checkpoint',
        ckpt_dir,
        '--env',
        'point_EightRooms',
        '--output',
        out_sub + os.sep,
        '--seed',
        '42',
        '--num_trajectories',
        '5',
        '--stochastic',
        '--no_repr_overlay',
    ]
    print(f'[watch_wbc_traj] {" ".join(cmd)}', flush=True)
    return subprocess.call(cmd, cwd=repo_root, env=os.environ.copy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--log_roots',
        nargs='+',
        default=['logs/wbc_eightrooms_v2'],
        help='Directories to search recursively.')
    ap.add_argument(
        '--output_base',
        default='plots/wbc_eightrooms_v2/stochastic',
        help='PNG output root.')
    ap.add_argument(
        '--repo_root',
        default='.',
        help='Repo root (default: cwd).')
    ap.add_argument(
        '--watch_interval',
        type=float,
        default=180.0,
        help='Seconds between scan cycles (default: 180).')
    ap.add_argument(
        '--once',
        action='store_true',
        help='Single cycle then exit.')
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.expanduser(args.repo_root))
    out_base = os.path.join(repo_root, args.output_base)

    cycle = 0
    try:
        while True:
            cycle += 1
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f'\n[watch_wbc_traj] === cycle {cycle} @ {ts} ===', flush=True)
            dirs = _iter_checkpoint_dirs(list(args.log_roots))
            if not dirs:
                print('[watch_wbc_traj] no WBC checkpoint dirs found yet.',
                      flush=True)
            for ck in dirs:
                sub = _out_subdir(out_base, ck)
                rc = _render_one(repo_root, ck, sub)
                if rc != 0:
                    print(f'[watch_wbc_traj] WARNING: exit {rc} for {ck}',
                          flush=True)
            if args.once:
                break
            delay = max(30.0, float(args.watch_interval))
            print(f'[watch_wbc_traj] next cycle in {delay:.0f}s  (Ctrl+C to stop)',
                  flush=True)
            time.sleep(delay)
    except KeyboardInterrupt:
        print('\n[watch_wbc_traj] stopped.', flush=True)


if __name__ == '__main__':
    main()
