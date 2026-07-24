#!/usr/bin/env python3
"""Login-node watcher: MPO BuilderBench checkpoint eval + success-vs-step plots.

Watches live MPO BuilderBench runs that do **not** already log periodic
eval during training. When new ``ckpt_iter_*.pkl`` files appear, submits a
short GPU oneshot (``jobs/job_mpo_bb_ckpt_eval_oneshot.slurm``) that
evaluates deterministic success and appends rows to a per-seed CSV, then
refreshes an aggregate mean±SE plot of success vs global env steps.

Skipped (in-training ``logs/eval/logs.csv`` already has success)::

  logs/mpo_crl_builderbench_creative3_task1_gpu_validated/…  (3550122)

Default runs::

  logs/mpo_crl_builderbench_creative2_task1_gpu_validation/mpo_crl_..._{0,1}
  logs/mpo_crl_builderbench_creative3_task2_e1024_pd_fast_succ/mpo_crl_..._{0,1}

Plots::

  figs/builderbench/mpo_c2t1_gpuval_checkpoint_success.png
  figs/builderbench/mpo_c3t2_checkpoint_success.png

State: ``figs/builderbench/checkpoint_eval/.watch_mpo_bb_eval_state.json``

Examples::

  python scripts/watch_mpo_builderbench_checkpoint_eval.py --once
  python scripts/watch_mpo_builderbench_checkpoint_eval.py --watch_interval 300
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVAL_DIR = os.path.join(_REPO, 'figs', 'builderbench', 'checkpoint_eval')
_DEFAULT_STATE = os.path.join(_EVAL_DIR, '.watch_mpo_bb_eval_state.json')
_BB_EVAL_JOB = os.path.join(_REPO, 'jobs', 'job_mpo_bb_ckpt_eval_oneshot.slurm')
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')

_MIN_PER_CKPT = 3
_TIME_CAP_MIN = 55
_TIME_FLOOR_MIN = 12

# Fixed runs matching live MPO BuilderBench jobs (offline ckpt eval plots).
_DEFAULT_GROUPS: List[dict] = [
    {
        'plot_tag': 'mpo_c2t1_gpuval',
        'env': 'builderbench_creative_2_task1',
        'runs': [
            {
                'run_dir': os.path.join(
                    _REPO,
                    'logs/mpo_crl_builderbench_creative2_task1_gpu_validation/'
                    'mpo_crl_builderbench_creative_2_task1_0'),
                'seed': 0,
            },
            {
                'run_dir': os.path.join(
                    _REPO,
                    'logs/mpo_crl_builderbench_creative2_task1_gpu_validation/'
                    'mpo_crl_builderbench_creative_2_task1_1'),
                'seed': 1,
            },
        ],
    },
    {
        'plot_tag': 'mpo_c3t1_gpuval',
        'env': 'builderbench_creative_3_task1',
        'runs': [
            {
                'run_dir': os.path.join(
                    _REPO,
                    'logs/mpo_crl_builderbench_creative3_task1_gpu_validated/'
                    'mpo_crl_builderbench_creative_3_task1_0'),
                'seed': 0,
            },
            {
                'run_dir': os.path.join(
                    _REPO,
                    'logs/mpo_crl_builderbench_creative3_task1_gpu_validated/'
                    'mpo_crl_builderbench_creative_3_task1_1'),
                'seed': 1,
            },
        ],
    },
    {
        'plot_tag': 'mpo_c3t2',
        'env': 'builderbench_creative_3_task2',
        'runs': [
            {
                'run_dir': os.path.join(
                    _REPO,
                    'logs/mpo_crl_builderbench_creative3_task2_e1024_pd_fast_succ/'
                    'mpo_crl_builderbench_creative_3_task2_0'),
                'seed': 0,
            },
            {
                'run_dir': os.path.join(
                    _REPO,
                    'logs/mpo_crl_builderbench_creative3_task2_e1024_pd_fast_succ/'
                    'mpo_crl_builderbench_creative_3_task2_1'),
                'seed': 1,
            },
        ],
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
        'bootstrapped': False,
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


def csv_labels(plot_tag: str, seed: int) -> Set[str]:
  path = os.path.join(
      _EVAL_DIR, f'{plot_tag}_seed{seed}_checkpoint_success.csv')
  if not os.path.isfile(path):
    return set()
  labels: Set[str] = set()
  with open(path, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      lab = (row.get('label') or '').strip()
      if lab:
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


def submit_eval_job(
    run_dir: str,
    env_name: str,
    plot_tag: str,
    seed: int,
    labels: Sequence[str],
    num_eval_episodes: int,
) -> Optional[str]:
  if not os.path.isfile(_BB_EVAL_JOB):
    print(f'[watch_mpo_bb_eval] missing job: {_BB_EVAL_JOB}', flush=True)
    return None
  # Use ';' not ',' — Slurm --export splits on commas.
  only = ';'.join(labels)
  export = (
      f'ALL,RUN_DIR={run_dir},ENV_NAME={env_name},PLOT_TAG={plot_tag},'
      f'SEED={seed},NUM_EVAL_EPISODES={num_eval_episodes},ONLY_LABELS={only}'
  )
  time_lim = walltime_for(len(labels))
  cmd = [
      'sbatch', '--parsable',
      f'--time={time_lim}',
      f'--export={export}',
      _BB_EVAL_JOB,
  ]
  print(f'[watch_mpo_bb_eval] sbatch ({time_lim}, n={len(labels)}): '
        f'{" ".join(cmd)}', flush=True)
  r = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if r.returncode != 0:
    print(f'[watch_mpo_bb_eval] sbatch failed: {r.stderr.strip()}', flush=True)
    return None
  job_id = r.stdout.strip().split(';')[0].strip()
  print(f'[watch_mpo_bb_eval] submitted {job_id} seed={seed} labels={list(labels)}',
        flush=True)
  return job_id


def _write_aggregate_plot(plot_tag: str, env_name: str, seeds: Sequence[int]) -> None:
  """Refresh mean±SE success-vs-step plot across seeds (CSV + matplotlib only)."""
  # Keep this free of JAX / BuilderBench imports so it stays login-node safe.
  import math

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker
  import numpy as np

  by_x: Dict[int, List[float]] = {}
  n_seeds_used = 0
  for seed in seeds:
    csv_path = os.path.join(
        _EVAL_DIR, f'{plot_tag}_seed{seed}_checkpoint_success.csv')
    if not os.path.isfile(csv_path):
      continue
    rows = 0
    with open(csv_path, newline='', encoding='utf-8') as fh:
      for row in csv.DictReader(fh):
        try:
          x = int(float(row['global_step']))
          y = float(row['success_mean'])
        except (KeyError, TypeError, ValueError):
          continue
        by_x.setdefault(x, []).append(y)
        rows += 1
    if rows:
      n_seeds_used += 1
  if not by_x or n_seeds_used == 0:
    return

  xs = np.asarray(sorted(by_x), dtype=np.float64)
  means = np.empty_like(xs)
  ses = np.empty_like(xs)
  for i, x in enumerate(xs):
    ys = np.asarray(by_x[int(x)], dtype=np.float64)
    means[i] = float(ys.mean())
    if ys.size > 1:
      ses[i] = float(ys.std(ddof=1) / math.sqrt(ys.size))
    else:
      ses[i] = 0.0

  def _fmt_steps(v, _):
    if v >= 1e6:
      s = f'{v / 1e6:.1f}M'
      return s.replace('.0M', 'M')
    if v >= 1e3:
      return f'{v / 1e3:.0f}K'
    return str(int(v))

  out_path = os.path.join(
      _REPO, 'figs', 'builderbench', f'{plot_tag}_checkpoint_success.png')
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig, ax = plt.subplots(figsize=(8, 4))
  color = '#4C9BE8'
  ax.plot(xs, means, color=color, linewidth=1.8, marker='o', markersize=4,
          label=f'eval success (mean ± SE, n={n_seeds_used} seed'
                f'{"s" if n_seeds_used != 1 else ""})')
  ax.fill_between(xs, means - ses, means + ses, color=color, alpha=0.25,
                  linewidth=0)
  ax.set_title(
      f'MPO BuilderBench eval success — {env_name} '
      f'({plot_tag}, seeds {list(seeds)})',
      fontsize=13, fontweight='bold')
  ax.set_xlabel('Global env steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  ax.set_ylabel('Eval success rate', fontsize=11)
  ax.set_ylim(-0.02, 1.02)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='best', fontsize=9, framealpha=0.9)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'[watch_mpo_bb_eval] wrote aggregate plot (n={n_seeds_used}): {out_path}',
        flush=True)


def handle_run(
    item: dict,
    group: dict,
    rs: dict,
    labels: List[str],
    num_eval_episodes: int,
) -> str:
  run_dir = item['run_dir']
  seed = int(item['seed'])
  plot_tag = group['plot_tag']
  env_name = group['env']

  disk = csv_labels(plot_tag, seed)
  if disk:
    rs['done_labels'] = sorted(disk | set(rs.get('done_labels') or []))

  pending_job = rs.get('pending_job_id')
  if pending_job and not squeue_has_job(str(pending_job)):
    print(f'[watch_mpo_bb_eval] job {pending_job} finished for '
          f'{plot_tag}_s{seed}', flush=True)
    rs['pending_job_id'] = None
    rs['pending_labels'] = []
    disk = csv_labels(plot_tag, seed)
    rs['done_labels'] = sorted(set(rs.get('done_labels') or []) | disk)
    rs['last_action'] = f'job_done:{pending_job}'

  # Bootstrap: evaluate all existing checkpoints on first sighting (capped).
  if not rs.get('bootstrapped'):
    rs['bootstrapped'] = True
    if not labels:
      return 'bootstrap(empty)'
    # Cap first submission so walltime stays reasonable; leftovers next scan.
    batch = labels[:6]
    if rs.get('pending_job_id') and squeue_has_job(str(rs['pending_job_id'])):
      return f'waiting_job={rs["pending_job_id"]}'
    job_id = submit_eval_job(
        run_dir, env_name, plot_tag, seed, batch, num_eval_episodes)
    if not job_id:
      return 'bootstrap_sbatch_failed'
    rs['pending_job_id'] = job_id
    rs['pending_labels'] = list(batch)
    # Do not mark as done until CSV rows exist.
    rs['last_action'] = f'bootstrap:{job_id}:{",".join(batch)}'
    return f'bootstrap:{job_id}:n={len(batch)}'

  done = set(rs.get('done_labels') or [])
  pending = set(rs.get('pending_labels') or [])
  new = [lab for lab in labels if lab not in done and lab not in pending]
  if not new:
    return 'ok(new=0)'

  if rs.get('pending_job_id') and squeue_has_job(str(rs['pending_job_id'])):
    return f'waiting_job={rs["pending_job_id"]}(new={len(new)})'

  batch = new[:4]
  job_id = submit_eval_job(
      run_dir, env_name, plot_tag, seed, batch, num_eval_episodes)
  if not job_id:
    return 'sbatch_failed'
  rs['pending_job_id'] = job_id
  rs['pending_labels'] = list(batch)
  rs['last_action'] = f'submit:{job_id}:{",".join(batch)}'
  return f'submit:{job_id}:n={len(batch)}'


def scan_once(state: dict, num_eval_episodes: int) -> None:
  os.makedirs(_EVAL_DIR, exist_ok=True)
  for group in _DEFAULT_GROUPS:
    seeds = []
    for item in group['runs']:
      run_dir = os.path.abspath(item['run_dir'])
      seeds.append(int(item['seed']))
      if not os.path.isdir(run_dir):
        print(f'[watch_mpo_bb_eval] missing run_dir (waiting): {run_dir}',
              flush=True)
        continue
      rs = _run_state(state, run_dir)
      labels = list_checkpoint_labels(os.path.join(run_dir, 'checkpoints'))
      action = handle_run(
          {**item, 'run_dir': run_dir}, group, rs, labels, num_eval_episodes)
      print(
          f'[watch_mpo_bb_eval] {group["plot_tag"]}_s{item["seed"]}: '
          f'ckpts={len(labels)} → {action}',
          flush=True)
    try:
      _write_aggregate_plot(group['plot_tag'], group['env'], seeds)
    except Exception as exc:  # noqa: BLE001
      print(f'[watch_mpo_bb_eval] aggregate plot error '
            f'({group["plot_tag"]}): {exc!r}', flush=True)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--once', action='store_true')
  ap.add_argument('--watch_interval', type=int, default=300)
  ap.add_argument('--state', default=_DEFAULT_STATE)
  ap.add_argument('--num_eval_episodes', type=int, default=20)
  args = ap.parse_args()

  state = _load_state(args.state)
  if args.once:
    scan_once(state, args.num_eval_episodes)
    _save_state(args.state, state)
    return

  print(f'[watch_mpo_bb_eval] daemon interval={args.watch_interval}s '
        f'state={args.state}', flush=True)
  while True:
    try:
      scan_once(state, args.num_eval_episodes)
      _save_state(args.state, state)
    except Exception as exc:  # noqa: BLE001
      print(f'[watch_mpo_bb_eval] scan error: {exc!r}', flush=True)
    time.sleep(max(60, int(args.watch_interval)))


if __name__ == '__main__':
  main()
