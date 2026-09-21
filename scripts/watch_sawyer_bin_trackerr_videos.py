#!/usr/bin/env python3
"""Login-node watcher: submit short tracking-error overlay jobs as ckpts appear.

Reusable for future Sawyer-bin NF runs. Example:

  python scripts/watch_sawyer_bin_trackerr_videos.py \\
    --train_job_id=3856801 \\
    --run_root=logs/final_metaworld_runs/... \\
    --out_root=videos/final_metaworld_runs/bin_nf_mpoinit_trackerrterm20cm_overlay \\
    --seeds=0,1 --ctrl=terminate --init_seed_base=30000 --also_latest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB_SCRIPT = os.path.join(REPO, 'jobs', 'job_render_sawyer_bin_trackerr_overlay.slurm')
MILESTONE_ITERS = 4000
MAX_SUBMIT_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 600
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')


def _load_state(path: str) -> dict:
  if not os.path.isfile(path):
    return {'jobs': {}}
  with open(path, encoding='utf-8') as fh:
    state = json.load(fh)
  state.setdefault('jobs', {})
  return state


def _save_state(path: str, state: dict) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + '.tmp'
  with open(tmp, 'w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)
  os.replace(tmp, path)


def _active_job_ids() -> set[str]:
  proc = subprocess.run(
      ['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%A'],
      check=True, capture_output=True, text=True)
  return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _checkpoint_iters(run_root: str, seed: int) -> list[int]:
  pattern = os.path.join(
      run_root, f'ppo_sawyer_bin_{seed}', 'checkpoints', 'ckpt_iter_*.pkl')
  out = []
  for path in glob.glob(pattern):
    match = _CKPT_RE.search(path)
    if match:
      out.append(int(match.group(1)))
  return sorted(set(out))


def _init_seed(base: int, policy_seed: int, checkpoint_iter: int) -> int:
  return int(base) + 1000 * int(policy_seed) + int(checkpoint_iter) // MILESTONE_ITERS


def _output_path(out_root: str, seed: int, iteration: int, init_seed: int) -> str:
  return os.path.join(
      out_root, f'seed{seed}',
      f'sawyer_bin_iter_{iteration:07d}_init{init_seed}_trackerr.mp4')


def _submit(args, seed: int, iteration: int, init_seed: int, out_dir: str) -> str:
  export = (
      f'ALL,POLICY_SEED={seed},CKPT_ITER={iteration},INIT_SEED={init_seed},'
      f'RUN_ROOT={args.run_root},OUT_DIR={out_dir},CTRL={args.ctrl}')
  if args.sawyer_max_episode_steps > 0:
    export += f',SAWYER_MAX_EPISODE_STEPS={int(args.sawyer_max_episode_steps)}'
  proc = subprocess.run(
      ['sbatch', '--parsable', f'--export={export}', JOB_SCRIPT],
      cwd=REPO, check=True, capture_output=True, text=True)
  return proc.stdout.strip().split(';', maxsplit=1)[0]


def scan_once(args, state: dict) -> tuple[int, bool]:
  active = _active_job_ids()
  training_active = str(args.train_job_id) in active
  now = time.time()
  submitted = 0
  eligible = []
  for seed in args.seeds:
    found = _checkpoint_iters(args.run_root, seed)
    chosen = [i for i in found if i % MILESTONE_ITERS == 0]
    if args.also_latest and found and found[-1] not in chosen:
      chosen.append(found[-1])
    for iteration in chosen:
      eligible.append((seed, iteration))

  for seed, iteration in eligible:
    init_seed = _init_seed(args.init_seed_base, seed, iteration)
    out_dir = os.path.join(args.out_root, f'seed{seed}')
    output = _output_path(args.out_root, seed, iteration, init_seed)
    if os.path.isfile(output) and os.path.getsize(output) > 0:
      continue
    key = f'seed{seed}:iter{iteration}'
    record = state['jobs'].get(key, {})
    job_id = str(record.get('job_id', ''))
    if job_id in active:
      continue
    attempts = int(record.get('attempts', 0))
    submitted_at = float(record.get('submitted_at', 0.0))
    if attempts >= MAX_SUBMIT_ATTEMPTS:
      continue
    if attempts and now - submitted_at < RETRY_DELAY_SECONDS:
      continue
    job_id = _submit(args, seed, iteration, init_seed, out_dir)
    state['jobs'][key] = {
        'job_id': job_id,
        'attempts': attempts + 1,
        'submitted_at': now,
        'output': output,
    }
    submitted += 1
    print(
        f'SUBMITTED_VIDEO job={job_id} policy_seed={seed} '
        f'checkpoint_iter={iteration} init_seed={init_seed}',
        flush=True)

  _save_state(args.state_path, state)
  render_active = any(
      str(record.get('job_id', '')) in active
      for record in state['jobs'].values())
  all_ready = bool(eligible) and all(
      os.path.isfile(_output_path(
          args.out_root, seed, iteration,
          _init_seed(args.init_seed_base, seed, iteration)))
      and os.path.getsize(_output_path(
          args.out_root, seed, iteration,
          _init_seed(args.init_seed_base, seed, iteration))) > 0
      for seed, iteration in eligible)
  done = not training_active and not render_active and all_ready
  return submitted, done


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--train_job_id', required=True)
  p.add_argument('--run_root', required=True)
  p.add_argument('--out_root', required=True)
  p.add_argument('--seeds', default='0')
  p.add_argument('--ctrl', default='vanilla',
                 choices=('vanilla', 'antiwindup', 'terminate', 'bounded'))
  p.add_argument('--init_seed_base', type=int, default=30000)
  p.add_argument('--sawyer_max_episode_steps', type=int, default=-1)
  p.add_argument('--also_latest', action='store_true',
                 help='Also render the newest checkpoint, even if not a 4000 milestone.')
  p.add_argument('--once', action='store_true')
  p.add_argument('--watch_interval', type=float, default=120.0)
  args = p.parse_args()
  args.run_root = os.path.join(REPO, args.run_root) if not os.path.isabs(args.run_root) else args.run_root
  args.out_root = os.path.join(REPO, args.out_root) if not os.path.isabs(args.out_root) else args.out_root
  args.seeds = tuple(int(s) for s in str(args.seeds).split(',') if s.strip() != '')
  args.state_path = os.path.join(args.out_root, '.watch_state.json')
  state = _load_state(args.state_path)
  while True:
    try:
      submitted, done = scan_once(args, state)
      print(f'SCAN submitted={submitted} complete={done}', flush=True)
      if args.once or done:
        return
    except Exception as exc:  # noqa: BLE001
      print(f'WATCH_ERROR {type(exc).__name__}: {exc}', flush=True)
      if args.once:
        raise
    time.sleep(max(30.0, args.watch_interval))


if __name__ == '__main__':
  main()
