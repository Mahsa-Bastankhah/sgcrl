#!/usr/bin/env python3
"""Login-node watcher: render videos for specific MPO BuilderBench runs.

Watches fixed run directories (from training logs). When a new
``ckpt_iter_*.pkl`` appears, submit a short GPU job that renders only those
new checkpoints. Never holds a GPU on the login node.

Default runs (c2/c3 validation jobs)::

  logs/mpo_crl_builderbench_creative2_task1_gpu_validation/mpo_crl_builderbench_creative_2_task1_{0,1}
  logs/mpo_crl_builderbench_creative3_task1_gpu_validated/mpo_crl_builderbench_creative_3_task1_{0,1}

Videos: ``videos/builderbench/mpo_active_runs/<short>_s<seed>/``
State:  ``videos/builderbench/mpo_active_runs/.watch_state.json``

Examples::

  python scripts/watch_mpo_builderbench_video_updates.py --once
  python scripts/watch_mpo_builderbench_video_updates.py --watch_interval 180
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ACTIVE_ROOT = os.path.join(_REPO, 'videos', 'builderbench', 'mpo_active_runs')
_DEFAULT_STATE = os.path.join(_ACTIVE_ROOT, '.watch_state.json')
_BB_VIDEO_JOB = os.path.join(_REPO, 'jobs', 'job_mpo_bb_video_new_ckpts.slurm')
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')

_MIN_PER_CKPT = 4
_TIME_CAP_MIN = 25
_TIME_FLOOR_MIN = 8

# Fixed runs corresponding to the user's four training logs.
_DEFAULT_RUNS: List[dict] = [
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_builderbench_creative2_task1_gpu_validation/'
            'mpo_crl_builderbench_creative_2_task1_0'),
        'seed': 0,
        'short': 'mpo_c2t1_gpuval',
        'log': 'slurm/mpo_crl_builderbench_creative2_task1_e1024_pd_3550113_0.log',
    },
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_builderbench_creative2_task1_gpu_validation/'
            'mpo_crl_builderbench_creative_2_task1_1'),
        'seed': 1,
        'short': 'mpo_c2t1_gpuval',
        'log': 'slurm/mpo_crl_builderbench_creative2_task1_e1024_pd_3550121_1.log',
    },
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_builderbench_creative3_task1_gpu_validated/'
            'mpo_crl_builderbench_creative_3_task1_0'),
        'seed': 0,
        'short': 'mpo_c3t1_gpuval',
        'log': 'slurm/mpo_crl_builderbench_creative3_task1_e1024_pd_3550122_0.log',
    },
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_builderbench_creative3_task1_gpu_validated/'
            'mpo_crl_builderbench_creative_3_task1_1'),
        'seed': 1,
        'short': 'mpo_c3t1_gpuval',
        'log': 'slurm/mpo_crl_builderbench_creative3_task1_e1024_pd_3550122_1.log',
    },
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_builderbench_creative3_task2_e1024_pd_fast_succ/'
            'mpo_crl_builderbench_creative_3_task2_0'),
        'seed': 0,
        'short': 'mpo_c3t2',
        'log': 'slurm/mpo_crl_builderbench_creative3_task2_e1024_pd_3550292_0.log',
    },
    {
        'run_dir': os.path.join(
            _REPO,
            'logs/mpo_crl_builderbench_creative3_task2_e1024_pd_fast_succ/'
            'mpo_crl_builderbench_creative_3_task2_1'),
        'seed': 1,
        'short': 'mpo_c3t2',
        'log': 'slurm/mpo_crl_builderbench_creative3_task2_e1024_pd_3550292_1.log',
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
        'pending_labels': [],
        'pending_job_id': None,
        'last_action': None,
        'run_tag': None,
        'bootstrapped': False,
        'short_name': None,
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


def label_to_ckpt_path(run_dir: str, label: str) -> Optional[str]:
  m = re.fullmatch(r'iter_(\d+)', label)
  if not m:
    return None
  path = os.path.join(
      run_dir, 'checkpoints', f'ckpt_iter_{int(m.group(1)):07d}.pkl')
  return path if os.path.isfile(path) else None


def read_run_config(run_dir: str) -> dict:
  path = os.path.join(run_dir, 'run_config.json')
  if not os.path.isfile(path):
    return {}
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def env_from_run(run_dir: str, run_cfg: dict) -> str:
  env = str(run_cfg.get('env') or '')
  if env:
    return env
  base = os.path.basename(run_dir)
  m = re.match(r'mpo_crl_(builderbench_creative_\d+_task\d+)_\d+$', base)
  return m.group(1) if m else ''


def out_dir_for(short_name: str) -> str:
  return os.path.join(_ACTIVE_ROOT, short_name)


def video_labels_on_disk(out_dir: str, run_tag: str) -> Set[str]:
  if not run_tag or not os.path.isdir(out_dir):
    return set()
  labels: Set[str] = set()
  prefix = f'{run_tag}_'
  for path in glob.glob(os.path.join(out_dir, f'{run_tag}_*.mp4')):
    base = os.path.basename(path)
    if not base.startswith(prefix) or not base.endswith('.mp4'):
      continue
    lab = base[len(prefix):-len('.mp4')]
    if lab.startswith('iter_') or lab == 'latest':
      labels.add(lab)
  return labels


def squeue_has_job(job_id: Optional[str]) -> bool:
  if not job_id:
    return False
  try:
    r = subprocess.run(
        ['squeue', '-j', str(job_id), '-h', '-o', '%i'],
        capture_output=True, text=True, check=False)
  except FileNotFoundError:
    return False
  return bool(r.stdout.strip())


def walltime_for(n_ckpts: int) -> str:
  mins = min(_TIME_CAP_MIN, max(_TIME_FLOOR_MIN, _MIN_PER_CKPT * max(1, n_ckpts)))
  return f'00:{mins:02d}:00'


def submit_new_ckpt_videos(
    run_dir: str,
    env_name: str,
    run_tag: str,
    seed: int,
    out_dir: str,
    labels: Sequence[str],
) -> Optional[str]:
  paths = []
  for lab in labels:
    p = label_to_ckpt_path(run_dir, lab)
    if p:
      paths.append(p)
  if not paths:
    print(f'[watch_mpo_bb] no ckpt paths for labels={list(labels)}', flush=True)
    return None
  if not os.path.isfile(_BB_VIDEO_JOB):
    print(f'[watch_mpo_bb] missing job script: {_BB_VIDEO_JOB}', flush=True)
    return None

  list_dir = os.path.join(_ACTIVE_ROOT, '.ckpt_lists')
  os.makedirs(list_dir, exist_ok=True)
  list_file = os.path.join(
      list_dir, f'{run_tag}_{int(time.time())}_{os.getpid()}.txt')
  with open(list_file, 'w', encoding='utf-8') as fh:
    for p in paths:
      fh.write(p + '\n')

  time_lim = walltime_for(len(paths))
  export = (
      f'ALL,ENV_NAME={env_name},RUN_TAG={run_tag},SEED={seed},'
      f'OUT_DIR={out_dir},CKPT_LIST_FILE={list_file}'
  )
  cmd = [
      'sbatch', '--parsable',
      f'--time={time_lim}',
      f'--export={export}',
      _BB_VIDEO_JOB,
  ]
  print(f'[watch_mpo_bb] sbatch ({time_lim}, n={len(paths)}): {" ".join(cmd)}',
        flush=True)
  r = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if r.returncode != 0:
    print(f'[watch_mpo_bb] sbatch failed rc={r.returncode}: {r.stderr.strip()}',
          flush=True)
    return None
  job_id = r.stdout.strip().split(';')[0].strip()
  print(f'[watch_mpo_bb] submitted {job_id} → {out_dir} labels={list(labels)}',
        flush=True)
  return job_id


def handle_run(item: dict, rs: dict, labels: Sequence[str], run_cfg: dict,
               threshold: int) -> str:
  run_dir = item['run_dir']
  seed = int(item['seed'])
  env_name = env_from_run(run_dir, run_cfg)
  if not env_name:
    return 'skip(no env)'

  short = f"{item['short']}_s{seed}"
  rs['short_name'] = short
  out_dir = out_dir_for(short)
  os.makedirs(out_dir, exist_ok=True)
  marker = os.path.join(out_dir, '.run_dir')
  with open(marker, 'w', encoding='utf-8') as fh:
    fh.write(run_dir + '\n')
  run_tag = rs.get('run_tag') or short
  rs['run_tag'] = run_tag

  disk_iters = {
      lab for lab in video_labels_on_disk(out_dir, run_tag)
      if lab.startswith('iter_')}
  if disk_iters:
    rs['done_labels'] = sorted(disk_iters | set(rs.get('done_labels') or []))

  pending_job = rs.get('pending_job_id')
  if pending_job and not squeue_has_job(str(pending_job)):
    print(f'[watch_mpo_bb] job {pending_job} finished for {short}', flush=True)
    rs['pending_job_id'] = None
    rs['pending_labels'] = []
    disk_iters = {
        lab for lab in video_labels_on_disk(out_dir, run_tag)
        if lab.startswith('iter_')}
    rs['done_labels'] = sorted(
        set(rs.get('done_labels') or []) | disk_iters)
    rs['last_action'] = f'job_done:{pending_job}'

  # First sighting: render latest only; mark older as done.
  if not rs.get('bootstrapped') and not disk_iters:
    rs['bootstrapped'] = True
    if not labels:
      rs['done_labels'] = []
      return 'bootstrap(empty)'
    latest = labels[-1]
    rs['done_labels'] = [lab for lab in labels if lab != latest]
    if rs.get('pending_job_id') and squeue_has_job(str(rs['pending_job_id'])):
      return f'waiting_job={rs["pending_job_id"]}'
    job_id = submit_new_ckpt_videos(
        run_dir, env_name, run_tag, seed, out_dir, [latest])
    if not job_id:
      rs['done_labels'] = list(labels)
      return 'bootstrap_sbatch_failed'
    rs['pending_job_id'] = job_id
    rs['pending_labels'] = [latest]
    rs['last_action'] = f'bootstrap_latest:{job_id}:{latest}'
    return f'bootstrap_latest:{job_id}:{latest}'

  done = set(rs.get('done_labels') or [])
  pending = set(rs.get('pending_labels') or [])
  new = [lab for lab in labels if lab not in done and lab not in pending]
  if len(new) <= threshold:
    return f'ok(new={len(new)})'

  if rs.get('pending_job_id') and squeue_has_job(str(rs['pending_job_id'])):
    return f'waiting_job={rs["pending_job_id"]}(new={len(new)})'

  batch = new[:4]
  job_id = submit_new_ckpt_videos(
      run_dir, env_name, run_tag, seed, out_dir, batch)
  if not job_id:
    return 'sbatch_failed'
  rs['pending_job_id'] = job_id
  rs['pending_labels'] = list(batch)
  rs['last_action'] = f'submit:{job_id}:{",".join(batch)}'
  return f'submit:{job_id}:n={len(batch)}'


def scan_once(state: dict, threshold: int) -> None:
  os.makedirs(_ACTIVE_ROOT, exist_ok=True)
  for item in _DEFAULT_RUNS:
    run_dir = os.path.abspath(item['run_dir'])
    if not os.path.isdir(run_dir):
      print(f'[watch_mpo_bb] missing run_dir (waiting): {run_dir}', flush=True)
      continue
    rs = _run_state(state, run_dir)
    labels = list_checkpoint_labels(os.path.join(run_dir, 'checkpoints'))
    run_cfg = read_run_config(run_dir)
    action = handle_run(
        {**item, 'run_dir': run_dir}, rs, labels, run_cfg, threshold)
    print(
        f'[watch_mpo_bb] {item["short"]}_s{item["seed"]}: '
        f'ckpts={len(labels)} → {action}',
        flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--once', action='store_true')
  parser.add_argument('--watch_interval', type=int, default=180)
  parser.add_argument('--state', default=_DEFAULT_STATE)
  parser.add_argument('--threshold', type=int, default=0,
                      help='Trigger when new checkpoints > threshold.')
  args = parser.parse_args()

  state = _load_state(args.state)
  if args.once:
    scan_once(state, args.threshold)
    _save_state(args.state, state)
    return

  print(f'[watch_mpo_bb] daemon interval={args.watch_interval}s '
        f'state={args.state}', flush=True)
  while True:
    try:
      scan_once(state, args.threshold)
      _save_state(args.state, state)
    except Exception as exc:  # noqa: BLE001
      print(f'[watch_mpo_bb] scan error: {exc!r}', flush=True)
    time.sleep(max(30, int(args.watch_interval)))


if __name__ == '__main__':
  main()
