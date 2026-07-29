#!/usr/bin/env python3
"""Login-node watcher: render videos only for *currently running* BuilderBench jobs.

When a live run gains new ``ckpt_iter_*.pkl`` files (default: any new ckpt),
submit one short GPU job that renders **only those new checkpoints**, then
keep watching. Never holds a GPU on the login node.

Videos go under ``videos/builderbench/active_runs/<short>_s<seed>/``.
Each scan rewrites ``README.txt`` to list live runs only. When a run leaves
the queue, its video folder is **moved** to
``videos/builderbench/completed_runs/`` (never deleted), so mid-training
mp4s are kept.

State: ``videos/builderbench/active_runs/.watch_state.json``

Examples::

  python scripts/watch_builderbench_video_updates.py --once
  python scripts/watch_builderbench_video_updates.py --watch_interval 300
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ACTIVE_ROOT = os.path.join(_REPO, 'videos', 'builderbench', 'active_runs')
_COMPLETED_ROOT = os.path.join(_REPO, 'videos', 'builderbench', 'completed_runs')
_DEFAULT_STATE = os.path.join(_ACTIVE_ROOT, '.watch_state.json')
_BB_VIDEO_JOB = os.path.join(_REPO, 'jobs', 'job_bb_video_new_ckpts.slurm')
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')
_SEED_RE = re.compile(r'^ppo_(.+)_(\d+)$')
_BB_JOB_RE = re.compile(r'^ppo_bb_', re.I)

# Minutes of walltime per new checkpoint (capped).
_MIN_PER_CKPT = 4
_TIME_CAP_MIN = 25
_TIME_FLOOR_MIN = 8

NEW_CKPT_THRESHOLD = 0  # trigger when new_ckpts > 0 for live runs


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


def seed_from_run_dir(run_dir: str) -> int:
  m = _SEED_RE.match(os.path.basename(run_dir))
  return int(m.group(2)) if m else 0


def env_from_run(run_dir: str, run_cfg: dict) -> str:
  env = str(run_cfg.get('env') or '')
  if env:
    return env
  m = _SEED_RE.match(os.path.basename(run_dir))
  return m.group(1) if m else ''


def log_basename(run_dir: str) -> str:
  return os.path.basename(os.path.dirname(run_dir))


def short_name_for(run_dir: str, seed: int) -> str:
  """Human-friendly folder name under active_runs/."""
  parent = log_basename(run_dir)
  tag = parent
  if tag.startswith('ppo_builderbench_'):
    tag = tag[len('ppo_builderbench_'):]
  # compress common prefixes
  tag = (tag
         .replace('creative3_task1_', 'c3t1_')
         .replace('creative3_task2_', 'c3t2_')
         .replace('creative4_task1_', 'c4t1_')
         .replace('creative4_task2_', 'c4t2_')
         .replace('creative5_task1_', 'c5t1_')
         .replace('creative5_task2_', 'c5t2_')
         .replace('e1024_pd_', '')
         .replace('td3_logq_tau05', 'td3_lq05')
         .replace('nf_tau05', 'nf_t05')
         .replace('crl_tau05', 'crl_t05'))
  return f'{tag}_s{seed}'


def out_dir_for(short_name: str) -> str:
  return os.path.join(_ACTIVE_ROOT, short_name)


def _video_render_skip_reason(run_cfg_path: str) -> Optional[str]:
  if not os.path.isfile(run_cfg_path):
    return None
  with open(run_cfg_path, 'r', encoding='utf-8') as fh:
    run_cfg = json.load(fh)
  flags = run_cfg.get('flags', {})
  if not bool(flags.get('builderbench_use_pd', False)):
    return None
  resolved = run_cfg.get('resolved_config', {}) or {}
  obs_dim = int(resolved.get('obs_dim', -1))
  env = str(run_cfg.get('env', ''))
  m = re.fullmatch(r'builderbench_creative_(\d+)_task\d+', env)
  if m:
    pd_dim = int(m.group(1)) * 3 + 1
    if obs_dim == pd_dim:
      return None
  else:
    pd_dim = '?'
  return (
      f'legacy PD without filtered policy obs '
      f'(obs_dim={obs_dim}, filtered={pd_dim})'
  )


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


def _parse_seed_from_training_log(log_path: str) -> Optional[int]:
  try:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as fh:
      # seed usually appears early
      for i, line in enumerate(fh):
        if i > 80:
          break
        m = re.search(r'\bseed=(\d+)\b', line)
        if m:
          return int(m.group(1))
        m = re.search(r'/ppo_builderbench_creative_\d+_task\d+_(\d+)/checkpoints', line)
        if m:
          return int(m.group(1))
  except OSError:
    return None
  return None


def _run_dir_from_training_log(log_path: str) -> Optional[str]:
  try:
    with open(log_path, 'r', encoding='utf-8', errors='replace') as fh:
      for i, line in enumerate(fh):
        if i > 120:
          break
        m = re.search(
            r'checkpoints\s*→\s*(logs/ppo_builderbench_[^\s]+/checkpoints)',
            line)
        if m:
          ckpt = m.group(1)
          run_dir = os.path.dirname(ckpt)
          return os.path.join(_REPO, run_dir)
        m = re.search(
            r'(logs/ppo_builderbench_[^\s]+/ppo_builderbench_creative_\d+_task\d+_\d+)/checkpoints',
            line)
        if m:
          return os.path.join(_REPO, m.group(1))
  except OSError:
    return None
  return None


def discover_running_bb_runs() -> List[dict]:
  """Return [{run_dir, seed, job_id, job_name}] for live ppo_bb_* jobs."""
  try:
    r = subprocess.run(
        ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i %j %T'],
        capture_output=True, text=True, check=False)
  except FileNotFoundError:
    return []
  out: List[dict] = []
  seen: Set[str] = set()
  for line in r.stdout.splitlines():
    parts = line.split()
    if len(parts) < 3:
      continue
    job_id, job_name, state = parts[0], parts[1], parts[2]
    if state not in ('RUNNING', 'PENDING'):
      continue
    if not _BB_JOB_RE.match(job_name):
      continue
    # Find matching training log: slurm/*_<jobid_base>_<array>.log
    base = job_id.split('_')[0]
    array = job_id.split('_')[1] if '_' in job_id else '0'
    candidates = sorted(
        glob.glob(os.path.join(_REPO, 'slurm', f'*_{base}_{array}.log')),
        key=os.path.getmtime, reverse=True)
    run_dir = None
    seed = None
    for log_path in candidates[:5]:
      run_dir = _run_dir_from_training_log(log_path)
      seed = _parse_seed_from_training_log(log_path)
      if run_dir:
        break
    if not run_dir or not os.path.isdir(run_dir):
      print(f'[watch_bb_vid] WARN: could not resolve run_dir for {job_id} {job_name}',
            flush=True)
      continue
    run_dir = os.path.abspath(run_dir)
    if seed is None:
      seed = seed_from_run_dir(run_dir)
    if run_dir in seen:
      continue
    seen.add(run_dir)
    out.append({
        'run_dir': run_dir,
        'seed': int(seed),
        'job_id': job_id,
        'job_name': job_name,
    })
  out.sort(key=lambda d: d['run_dir'])
  return out


def rebuild_active_index(live: Sequence[dict]) -> None:
  """Rewrite active_runs/ to only list currently running runs (dirs + README)."""
  os.makedirs(_ACTIVE_ROOT, exist_ok=True)
  wanted_shorts: Set[str] = set()
  lines = [
      'Currently-running BuilderBench video folders (auto-maintained).',
      'Watcher writes new-ckpt mp4s here. When a run ends, its folder is moved',
      'to videos/builderbench/completed_runs/ (videos are never deleted).',
      '',
  ]
  for item in live:
    seed = int(item['seed'])
    short = short_name_for(item['run_dir'], seed)
    wanted_shorts.add(short)
    out_dir = out_dir_for(short)
    os.makedirs(out_dir, exist_ok=True)
    # Marker pointing back to the training run
    marker = os.path.join(out_dir, '.run_dir')
    with open(marker, 'w', encoding='utf-8') as fh:
      fh.write(item['run_dir'] + '\n')
    n_mp4 = len(glob.glob(os.path.join(out_dir, '*.mp4')))
    lines.append(
        f'{short}/  (seed={seed}, job={item["job_id"]}, '
        f'mp4s={n_mp4}, run={item["run_dir"]})')

  readme = os.path.join(_ACTIVE_ROOT, 'README.txt')
  with open(readme, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines) + '\n')

  # Move finished runs out of active_runs/ — never delete their videos.
  os.makedirs(_COMPLETED_ROOT, exist_ok=True)
  for name in os.listdir(_ACTIVE_ROOT):
    if name.startswith('.') or name == 'README.txt':
      continue
    path = os.path.join(_ACTIVE_ROOT, name)
    if name in wanted_shorts:
      continue
    dest = os.path.join(_COMPLETED_ROOT, name)
    if os.path.islink(path):
      os.unlink(path)
      print(f'[watch_bb_vid] removed stale link {name}', flush=True)
      continue
    if not os.path.isdir(path):
      continue
    if os.path.exists(dest):
      os.makedirs(dest, exist_ok=True)
      for src in glob.glob(os.path.join(path, '*')):
        target = os.path.join(dest, os.path.basename(src))
        if os.path.isdir(src):
          continue
        if not os.path.exists(target):
          shutil.move(src, target)
        else:
          # Keep the newer file; never drop an mp4 on the floor.
          try:
            if os.path.getmtime(src) >= os.path.getmtime(target):
              os.replace(src, target)
            else:
              os.remove(src)
          except OSError:
            # If replace fails, leave src and skip rmtree below.
            print(f'[watch_bb_vid] WARN: could not merge {src} → {target}',
                  flush=True)
      leftover = glob.glob(os.path.join(path, '*'))
      if leftover:
        print(f'[watch_bb_vid] WARN: not removing {path}; leftover={leftover}',
              flush=True)
      else:
        shutil.rmtree(path)
        print(f'[watch_bb_vid] merged+archived {name} → completed_runs/',
              flush=True)
    else:
      shutil.move(path, dest)
      print(f'[watch_bb_vid] archived {name} → completed_runs/', flush=True)


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
    print(f'[watch_bb_vid] no ckpt paths for labels={list(labels)}', flush=True)
    return None
  if not os.path.isfile(_BB_VIDEO_JOB):
    print(f'[watch_bb_vid] missing job script: {_BB_VIDEO_JOB}', flush=True)
    return None

  list_dir = os.path.join(_ACTIVE_ROOT, '.ckpt_lists')
  os.makedirs(list_dir, exist_ok=True)
  # Include a nonce so rapid successive submits do not overwrite the same file.
  list_file = os.path.join(
      list_dir,
      f'{run_tag}_{int(time.time())}_{os.getpid()}_{len(paths)}_{id(paths)}.txt')
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
  print(f'[watch_bb_vid] sbatch ({time_lim}, n={len(paths)}): {" ".join(cmd)}',
        flush=True)
  r = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if r.returncode != 0:
    print(f'[watch_bb_vid] sbatch failed rc={r.returncode}: {r.stderr.strip()}',
          flush=True)
    return None
  job_id = r.stdout.strip().split(';')[0].strip()
  print(f'[watch_bb_vid] submitted {job_id} → {out_dir} labels={list(labels)}',
        flush=True)
  return job_id


def handle_run(item: dict, rs: dict, labels: Sequence[str], run_cfg: dict,
               threshold: int) -> str:
  run_dir = item['run_dir']
  seed = int(item['seed'])
  env_name = env_from_run(run_dir, run_cfg)
  if not env_name:
    return 'skip(no env)'

  cfg_path = os.path.join(run_dir, 'run_config.json')
  skip = _video_render_skip_reason(cfg_path)
  if skip is not None:
    rs['done_labels'] = list(labels)
    rs['bootstrapped'] = True
    rs['pending_job_id'] = None
    rs['pending_labels'] = []
    rs['last_action'] = f'skip_render:{skip}'
    return f'skip_render({skip})'

  short = short_name_for(run_dir, seed)
  rs['short_name'] = short
  out_dir = out_dir_for(short)
  os.makedirs(out_dir, exist_ok=True)
  run_tag = rs.get('run_tag') or short
  rs['run_tag'] = run_tag

  disk_iters = {
      lab for lab in video_labels_on_disk(out_dir, run_tag)
      if lab.startswith('iter_')}
  if disk_iters:
    rs['done_labels'] = sorted(disk_iters | set(rs.get('done_labels') or []))

  pending_job = rs.get('pending_job_id')
  if pending_job and not squeue_has_job(str(pending_job)):
    print(f'[watch_bb_vid] job {pending_job} finished for {short}', flush=True)
    rs['pending_job_id'] = None
    rs['pending_labels'] = []
    disk_iters = {
        lab for lab in video_labels_on_disk(out_dir, run_tag)
        if lab.startswith('iter_')}
    rs['done_labels'] = sorted(
        set(rs.get('done_labels') or []) | disk_iters)
    rs['last_action'] = f'job_done:{pending_job}'

  # First sighting: bootstrap historical ckpts, but render the *latest* only
  # so active_runs is not empty while we wait for future checkpoints.
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
      # Still mark bootstrap so we do not storm; retry next scan via new logic.
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

  # Cap per submission so the GPU is released quickly; leftover new ckpts
  # are picked up on the next watcher scan.
  batch = new[:5]
  job_id = submit_new_ckpt_videos(
      run_dir, env_name, run_tag, seed, out_dir, batch)
  if not job_id:
    return 'sbatch_failed'
  rs['pending_job_id'] = job_id
  rs['pending_labels'] = list(batch)
  rs['last_action'] = f'submitted:{job_id}:n={len(batch)}'
  return f'submitted:{job_id}:new={len(batch)}/{len(new)}'


def scan_once(state: dict, threshold: int) -> Dict[str, int]:
  counts = defaultdict(int)
  live = discover_running_bb_runs()
  print(f'[watch_bb_vid] live BB training runs: {len(live)}', flush=True)
  rebuild_active_index(live)
  counts['live'] = len(live)

  live_dirs = {item['run_dir'] for item in live}
  # Drop pending tracking for runs no longer live (do not cancel their video jobs).
  for run_dir in list(state.get('runs', {})):
    if run_dir not in live_dirs:
      rs = state['runs'][run_dir]
      if rs.get('pending_job_id') and not squeue_has_job(str(rs['pending_job_id'])):
        rs['pending_job_id'] = None
        rs['pending_labels'] = []

  for item in live:
    run_dir = item['run_dir']
    run_cfg = read_run_config(run_dir)
    labels = list_checkpoint_labels(os.path.join(run_dir, 'checkpoints'))
    if not labels:
      counts['no_ckpts'] += 1
      continue
    rs = _run_state(state, run_dir)
    msg = handle_run(item, rs, labels, run_cfg, threshold)
    short = rs.get('short_name') or short_name_for(run_dir, item['seed'])
    print(f'[watch_bb_vid] {short}: {msg}', flush=True)
    counts['bb'] += 1
    if msg.startswith('submitted') or msg.startswith('bootstrap_latest'):
      counts['submitted'] += 1
    elif msg.startswith('ok'):
      counts['ok'] += 1
    elif msg.startswith('waiting'):
      counts['waiting'] += 1
    elif msg.startswith('skip_render'):
      counts['skip_render'] += 1
  return dict(counts)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--state_file', default=_DEFAULT_STATE)
  ap.add_argument('--once', action='store_true')
  ap.add_argument('--watch_interval', type=float, default=300.0)
  ap.add_argument('--threshold', type=int, default=NEW_CKPT_THRESHOLD,
                  help='Trigger when new checkpoints > threshold (default 0).')
  args = ap.parse_args()

  state_path = (args.state_file if os.path.isabs(args.state_file)
                else os.path.join(_REPO, args.state_file))
  threshold = int(args.threshold)

  print(f'[watch_bb_vid] active_root={_ACTIVE_ROOT}', flush=True)
  print(f'[watch_bb_vid] state={state_path}', flush=True)
  print(f'[watch_bb_vid] threshold=new_ckpts>{threshold}', flush=True)
  print(f'[watch_bb_vid] job={_BB_VIDEO_JOB}', flush=True)

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[watch_bb_vid] === scan @ {ts} ===', flush=True)
      state = _load_state(state_path)
      counts = scan_once(state, threshold)
      _save_state(state_path, state)
      print(f'[watch_bb_vid] counts={counts}', flush=True)
      if args.once:
        break
      delay = max(60.0, float(args.watch_interval))
      print(f'[watch_bb_vid] next scan in {delay:.0f}s', flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[watch_bb_vid] stopped.', flush=True)


if __name__ == '__main__':
  main()
