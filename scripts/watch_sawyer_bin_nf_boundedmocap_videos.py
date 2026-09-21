#!/usr/bin/env python3
"""Submit short render jobs as bounded-mocap NF checkpoints appear."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_ROOT = os.path.join(
    REPO, 'logs', 'final_metaworld_runs',
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_boundedmocap_mixtaskg_'
    'dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6')
OUT_ROOT = os.path.join(
    REPO, 'videos', 'final_metaworld_runs',
    'bin_nf_mpoinit_boundedmocap_deterministic')
JOB_SCRIPT = os.path.join(
    REPO, 'jobs',
    'job_render_sawyer_bin_nf_mpoinit_boundedmocap_deterministic.slurm')
STATE_PATH = os.path.join(OUT_ROOT, '.watch_state.json')
TRAIN_JOB_ID = '3847407'
SEEDS = (0, 1, 2)
MILESTONE_ITERS = 4000
MAX_SUBMIT_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 600
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')


def _load_state() -> dict:
  if not os.path.isfile(STATE_PATH):
    return {'jobs': {}, 'first_notified': False}
  with open(STATE_PATH, 'r', encoding='utf-8') as fh:
    state = json.load(fh)
  state.setdefault('jobs', {})
  state.setdefault('first_notified', False)
  return state


def _save_state(state: dict) -> None:
  os.makedirs(OUT_ROOT, exist_ok=True)
  tmp = STATE_PATH + '.tmp'
  with open(tmp, 'w', encoding='utf-8') as fh:
    json.dump(state, fh, indent=2, sort_keys=True)
  os.replace(tmp, STATE_PATH)


def _active_job_ids() -> set[str]:
  proc = subprocess.run(
      ['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%A'],
      check=True, capture_output=True, text=True)
  return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def _checkpoint_iters(seed: int) -> list[int]:
  pattern = os.path.join(
      RUN_ROOT, f'ppo_sawyer_bin_{seed}', 'checkpoints', 'ckpt_iter_*.pkl')
  found = []
  for path in glob.glob(pattern):
    match = _CKPT_RE.search(path)
    if match:
      found.append(int(match.group(1)))
  return sorted(set(found))


def _init_seed(policy_seed: int, checkpoint_iter: int) -> int:
  return 10_000 + 1_000 * policy_seed + checkpoint_iter // MILESTONE_ITERS


def _output_path(policy_seed: int, checkpoint_iter: int) -> str:
  init_seed = _init_seed(policy_seed, checkpoint_iter)
  return os.path.join(
      OUT_ROOT, f'seed{policy_seed}',
      f'sawyer_bin_iter_{checkpoint_iter:07d}_init{init_seed}.mp4')


def _submit(policy_seed: int, checkpoint_iter: int) -> str:
  init_seed = _init_seed(policy_seed, checkpoint_iter)
  export = (
      f'ALL,POLICY_SEED={policy_seed},CKPT_ITER={checkpoint_iter},'
      f'INIT_SEED={init_seed}')
  proc = subprocess.run(
      ['sbatch', '--parsable', f'--export={export}', JOB_SCRIPT],
      cwd=REPO, check=True, capture_output=True, text=True)
  return proc.stdout.strip().split(';', maxsplit=1)[0]


def scan_once(state: dict) -> tuple[int, bool]:
  active = _active_job_ids()
  training_active = TRAIN_JOB_ID in active
  now = time.time()
  submitted = 0
  eligible = []

  for seed in SEEDS:
    available = _checkpoint_iters(seed)
    selected = [i for i in available if i % MILESTONE_ITERS == 0]
    # Once training exits, also render its last checkpoint.
    if not training_active and available and available[-1] not in selected:
      selected.append(available[-1])
    for checkpoint_iter in selected:
      eligible.append((seed, checkpoint_iter))

  for seed, checkpoint_iter in eligible:
    key = f'seed{seed}:iter{checkpoint_iter}'
    output = _output_path(seed, checkpoint_iter)
    if os.path.isfile(output) and os.path.getsize(output) > 0:
      if not state['first_notified']:
        print(f'FIRST_VIDEO_READY {output}', flush=True)
        state['first_notified'] = True
      continue

    record = state['jobs'].get(key, {})
    job_id = str(record.get('job_id', ''))
    if job_id and job_id in active:
      continue
    attempts = int(record.get('attempts', 0))
    submitted_at = float(record.get('submitted_at', 0.0))
    if attempts >= MAX_SUBMIT_ATTEMPTS:
      continue
    if attempts and now - submitted_at < RETRY_DELAY_SECONDS:
      continue

    job_id = _submit(seed, checkpoint_iter)
    state['jobs'][key] = {
        'job_id': job_id,
        'attempts': attempts + 1,
        'submitted_at': now,
        'output': output,
    }
    submitted += 1
    print(
        f'SUBMITTED_VIDEO job={job_id} policy_seed={seed} '
        f'checkpoint_iter={checkpoint_iter} '
        f'init_seed={_init_seed(seed, checkpoint_iter)}',
        flush=True)

  _save_state(state)
  pending = any(
      str(record.get('job_id', '')) in active
      for record in state['jobs'].values())
  done = not training_active and not pending and all(
      os.path.isfile(_output_path(seed, checkpoint_iter))
      for seed, checkpoint_iter in eligible)
  return submitted, done


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--once', action='store_true')
  parser.add_argument('--watch_interval', type=float, default=120.0)
  args = parser.parse_args()
  state = _load_state()

  while True:
    try:
      submitted, done = scan_once(state)
      print(
          f'SCAN submitted={submitted} complete={done}',
          flush=True)
      if args.once or done:
        return
    except Exception as exc:  # noqa: BLE001 - watcher must survive transient Slurm errors.
      print(f'WATCH_ERROR {type(exc).__name__}: {exc}', flush=True)
      if args.once:
        raise
    time.sleep(max(30.0, args.watch_interval))


if __name__ == '__main__':
  main()
