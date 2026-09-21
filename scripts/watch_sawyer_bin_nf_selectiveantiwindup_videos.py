#!/usr/bin/env python3
"""Login-node watcher submitting short anti-windup overlay render jobs."""
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
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_selectiveantiwindup_mixtaskg_'
    'dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6')
OUT_ROOT = os.path.join(
    REPO, 'videos', 'final_metaworld_runs',
    'bin_nf_mpoinit_selectiveantiwindup_trackerr_overlay')
JOB_SCRIPT = os.path.join(
    REPO, 'jobs',
    'job_render_sawyer_bin_nf_mpoinit_selectiveantiwindup_trackerr.slurm')
STATE_PATH = os.path.join(OUT_ROOT, '.watch_state.json')
SEEDS = (0, 1, 2)
MILESTONE_ITERS = 4000
MAX_SUBMIT_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 600
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')


def _load_state() -> dict:
  if not os.path.isfile(STATE_PATH):
    return {'jobs': {}}
  with open(STATE_PATH, encoding='utf-8') as fh:
    state = json.load(fh)
  state.setdefault('jobs', {})
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
  out = []
  for path in glob.glob(pattern):
    match = _CKPT_RE.search(path)
    if match:
      out.append(int(match.group(1)))
  return sorted(set(out))


def _init_seed(policy_seed: int, checkpoint_iter: int) -> int:
  return 20_000 + 1_000 * policy_seed + checkpoint_iter // MILESTONE_ITERS


def _output_path(policy_seed: int, checkpoint_iter: int) -> str:
  return os.path.join(
      OUT_ROOT, f'seed{policy_seed}',
      f'sawyer_bin_iter_{checkpoint_iter:07d}_'
      f'init{_init_seed(policy_seed, checkpoint_iter)}_trackerr.mp4')


def _submit(policy_seed: int, checkpoint_iter: int) -> str:
  export = (
      f'ALL,POLICY_SEED={policy_seed},CKPT_ITER={checkpoint_iter},'
      f'INIT_SEED={_init_seed(policy_seed, checkpoint_iter)}')
  proc = subprocess.run(
      ['sbatch', '--parsable', f'--export={export}', JOB_SCRIPT],
      cwd=REPO, check=True, capture_output=True, text=True)
  return proc.stdout.strip().split(';', maxsplit=1)[0]


def scan_once(state: dict, train_job_id: str) -> tuple[int, bool]:
  active = _active_job_ids()
  training_active = train_job_id in active
  now = time.time()
  submitted = 0
  eligible = []
  for seed in SEEDS:
    eligible.extend(
        (seed, iteration) for iteration in _checkpoint_iters(seed)
        if iteration % MILESTONE_ITERS == 0)

  for seed, iteration in eligible:
    output = _output_path(seed, iteration)
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
    job_id = _submit(seed, iteration)
    state['jobs'][key] = {
        'job_id': job_id,
        'attempts': attempts + 1,
        'submitted_at': now,
        'output': output,
    }
    submitted += 1
    print(
        f'SUBMITTED_VIDEO job={job_id} policy_seed={seed} '
        f'checkpoint_iter={iteration} '
        f'init_seed={_init_seed(seed, iteration)}',
        flush=True)

  _save_state(state)
  render_active = any(
      str(record.get('job_id', '')) in active
      for record in state['jobs'].values())
  all_outputs_ready = all(
      os.path.isfile(_output_path(seed, iteration))
      and os.path.getsize(_output_path(seed, iteration)) > 0
      for seed, iteration in eligible)
  done = not training_active and not render_active and all_outputs_ready
  return submitted, done


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_job_id', required=True)
  parser.add_argument('--once', action='store_true')
  parser.add_argument('--watch_interval', type=float, default=120.0)
  args = parser.parse_args()
  state = _load_state()
  while True:
    try:
      submitted, done = scan_once(state, str(args.train_job_id))
      print(f'SCAN submitted={submitted} complete={done}', flush=True)
      if args.once or done:
        return
    except Exception as exc:  # noqa: BLE001 - survive transient Slurm errors.
      print(f'WATCH_ERROR {type(exc).__name__}: {exc}', flush=True)
      if args.once:
        raise
    time.sleep(max(30.0, args.watch_interval))


if __name__ == '__main__':
  main()
