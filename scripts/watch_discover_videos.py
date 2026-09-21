#!/usr/bin/env python3
"""Login-node CPU watcher: render DISCOVER ckpts as they appear.

  python scripts/watch_discover_videos.py \\
      --ckpt_dir logs/.../checkpoints \\
      --out_dir videos/discover_c2t1_1h \\
      --interval 60 --max_minutes 50
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _ckpts(ckpt_dir: Path):
  out = []
  for p in sorted(ckpt_dir.glob('ckpt_epoch_*.pkl')):
    out.append((p.name, p, p.stat().st_mtime))
  latest = ckpt_dir / 'latest.pkl'
  if latest.is_file():
    out.append(('latest.pkl', latest, latest.stat().st_mtime))
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ckpt_dir', required=True)
  p.add_argument('--out_dir', required=True)
  p.add_argument('--env', default='builderbench_creative_2_task1')
  p.add_argument('--interval', type=int, default=60)
  p.add_argument('--max_minutes', type=float, default=50)
  p.add_argument('--seed', type=int, default=0)
  args = p.parse_args()

  ckpt_dir = Path(args.ckpt_dir)
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  done: dict[str, float] = {}
  t0 = time.time()
  py = sys.executable
  script = str(_REPO / 'scripts' / 'discover_builderbench_rollout_video.py')
  env = {**os.environ}
  env.setdefault('JAX_PLATFORMS', 'cpu')
  env.setdefault('MUJOCO_GL', 'egl')
  env.setdefault('BUILDERBENCH_MJX_IMPL', 'jax')
  env.setdefault('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
  print(f'[watch_discover_vid] ckpt_dir={ckpt_dir} out={out_dir}', flush=True)

  while True:
    if ckpt_dir.is_dir():
      for name, path, mtime in _ckpts(ckpt_dir):
        prev = done.get(name)
        if prev is not None and prev >= mtime:
          continue
        tag = Path(name).stem
        cmd = [
            py, '-u', script,
            '--checkpoint', str(path),
            '--env', args.env,
            '--output', str(out_dir) + '/',
            '--run_tag', tag,
            '--seed', str(args.seed),
        ]
        print(f'[watch_discover_vid] render {name}', flush=True)
        rc = subprocess.call(cmd, cwd=str(_REPO), env=env)
        if rc == 0:
          done[name] = mtime
        else:
          print(f'[watch_discover_vid] render failed rc={rc} {name}',
                flush=True)
    else:
      print('[watch_discover_vid] waiting for checkpoints dir...', flush=True)
    if (time.time() - t0) / 60.0 >= args.max_minutes:
      print('[watch_discover_vid] done (time cap)', flush=True)
      return
    time.sleep(args.interval)


if __name__ == '__main__':
  main()
