#!/usr/bin/env python3
"""Login-node watcher: refresh eval-success plots when >2 new checkpoints appear.

Runs on the login node (CPU). Never holds a GPU.

* Metaworld / Sawyer runs: training already writes ``logs/eval/logs.csv`` with
  a ``success`` column. When >2 new ``ckpt_iter_*.pkl`` appear since the last
  refresh, replot from that CSV (all seeds under the same log root).
* BuilderBench runs: deterministic eval needs a GPU. When >2 new checkpoints
  are not yet in the checkpoint-eval CSV, submit a short one-shot SLURM job
  (``jobs/job_bb_ckpt_eval_oneshot.slurm``, time≤1h).

State: ``figs/.watch_eval_success_state.json``

Examples::

  # one scan
  python scripts/watch_eval_success_updates.py --once

  # daemon (login node)
  python scripts/watch_eval_success_updates.py --watch_interval 300
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_STATE = os.path.join(_REPO, 'figs', '.watch_eval_success_state.json')
_BB_EVAL_JOB = os.path.join(_REPO, 'jobs', 'job_bb_ckpt_eval_oneshot.slurm')
_CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')
_SEED_RE = re.compile(r'^ppo_(.+)_(\d+)$')

# Trigger when strictly more than this many *new* checkpoints appear.
NEW_CKPT_THRESHOLD = 2


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


def read_run_config(run_dir: str) -> dict:
  path = os.path.join(run_dir, 'run_config.json')
  if not os.path.isfile(path):
    return {}
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def classify_run(run_dir: str, run_cfg: dict) -> Optional[str]:
  """Return 'builderbench', 'sawyer', or None to skip."""
  env = str(run_cfg.get('env', '') or '')
  base = os.path.basename(run_dir)
  parent = os.path.basename(os.path.dirname(run_dir))
  if env.startswith('builderbench_') or 'builderbench' in parent or base.startswith(
      'ppo_builderbench_'):
    return 'builderbench'
  if env.startswith('sawyer_') or 'sawyer' in base or any(
      k in parent for k in ('reach', 'drawer', 'push', 'button', 'bin')):
    # Heuristic: Metaworld / Sawyer family with in-training eval CSV.
    return 'sawyer'
  if os.path.isfile(os.path.join(run_dir, 'logs', 'eval', 'logs.csv')):
    return 'sawyer'
  return None


def plot_tag_for(run_dir: str) -> str:
  parent = os.path.basename(os.path.dirname(run_dir))
  return parent.replace('ppo_', '', 1) if parent.startswith('ppo_') else parent


def resolve_bb_plot_tag(run_dir: str, seed: int) -> str:
  """Prefer an existing checkpoint-eval CSV tag that already covers this run."""
  default = plot_tag_for(run_dir)
  default_csv = os.path.join(
      _REPO, 'figs', 'builderbench', 'checkpoint_eval',
      f'{default}_seed{seed}_checkpoint_success.csv')
  if os.path.isfile(default_csv):
    return default

  # Fall back: any CSV whose paths point into this run_dir's checkpoints/.
  ckpt_prefix = os.path.join(os.path.abspath(run_dir), 'checkpoints')
  eval_dir = os.path.join(_REPO, 'figs', 'builderbench', 'checkpoint_eval')
  if os.path.isdir(eval_dir):
    for path in sorted(glob.glob(os.path.join(
        eval_dir, f'*_seed{seed}_checkpoint_success.csv'))):
      try:
        with open(path, newline='', encoding='utf-8') as fh:
          for row in csv.DictReader(fh):
            p = (row.get('path') or '').strip()
            if p.startswith(ckpt_prefix):
              base = os.path.basename(path)
              # "{tag}_seed{seed}_checkpoint_success.csv"
              suf = f'_seed{seed}_checkpoint_success.csv'
              if base.endswith(suf):
                return base[: -len(suf)]
              return default
      except Exception:
        continue
  return default


def seed_from_run_dir(run_dir: str) -> int:
  m = _SEED_RE.match(os.path.basename(run_dir))
  return int(m.group(2)) if m else 0


def discover_run_dirs(log_root: str) -> List[str]:
  out: List[str] = []
  if not os.path.isdir(log_root):
    return out
  for log_dir in sorted(glob.glob(os.path.join(log_root, 'ppo_*'))):
    if not os.path.isdir(log_dir):
      continue
    for run_dir in sorted(glob.glob(os.path.join(log_dir, 'ppo_*'))):
      ckpt = os.path.join(run_dir, 'checkpoints')
      if os.path.isdir(ckpt) and glob.glob(os.path.join(ckpt, 'ckpt_iter_*.pkl')):
        out.append(os.path.abspath(run_dir))
  return out


def bb_eval_csv_labels(plot_tag: str, seed: int) -> Set[str]:
  path = os.path.join(
      _REPO, 'figs', 'builderbench', 'checkpoint_eval',
      f'{plot_tag}_seed{seed}_checkpoint_success.csv')
  if not os.path.isfile(path):
    return set()
  labels: Set[str] = set()
  with open(path, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      lab = (row.get('label') or '').strip()
      if lab:
        labels.add(lab)
  return labels


def _coerce(v) -> Optional[float]:
  try:
    x = float(v)
    if math.isnan(x) or math.isinf(x):
      return None
    return x
  except Exception:
    return None


def _fmt_steps(v, _):
  if v >= 1e6:
    s = f'{v / 1e6:.1f}M'
    return s.replace('.0M', 'M')
  if v >= 1e3:
    return f'{v / 1e3:.0f}K'
  return str(int(v))


def load_sawyer_eval_curve(run_dir: str) -> List[Tuple[float, float]]:
  """Return [(x, success)] from logs/eval/logs.csv (prefer learner_steps)."""
  path = os.path.join(run_dir, 'logs', 'eval', 'logs.csv')
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return []
  pts: List[Tuple[float, float]] = []
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    if not reader.fieldnames or 'success' not in reader.fieldnames:
      return []
    for row in reader:
      y = _coerce(row.get('success'))
      if y is None:
        continue
      x = _coerce(row.get('learner_steps'))
      if x is None:
        x = _coerce(row.get('iteration'))
      if x is None:
        continue
      pts.append((x, y))
  pts.sort(key=lambda p: p[0])
  return pts


def plot_sawyer_log_dir(log_dir: str, plot_tag: str) -> Optional[str]:
  """Aggregate mean±SE across seeds under log_dir; write PNG. Return path."""
  series_by_seed: Dict[int, List[Tuple[float, float]]] = {}
  for run_dir in sorted(glob.glob(os.path.join(log_dir, 'ppo_*'))):
    if not os.path.isdir(run_dir):
      continue
    seed = seed_from_run_dir(run_dir)
    pts = load_sawyer_eval_curve(run_dir)
    if pts:
      series_by_seed[seed] = pts
  if not series_by_seed:
    return None

  # Align on union of x values (nearest-left style via interpolation per seed).
  all_x = sorted({x for pts in series_by_seed.values() for x, _ in pts})
  if not all_x:
    return None
  xs = np.asarray(all_x, dtype=np.float64)
  mats = []
  for seed in sorted(series_by_seed):
    pts = series_by_seed[seed]
    px = np.asarray([p[0] for p in pts], dtype=np.float64)
    py = np.asarray([p[1] for p in pts], dtype=np.float64)
    mats.append(np.interp(xs, px, py, left=np.nan, right=np.nan))
  mat = np.vstack(mats)
  mean = np.nanmean(mat, axis=0)
  if mat.shape[0] > 1:
    se = np.nanstd(mat, axis=0, ddof=1) / math.sqrt(mat.shape[0])
  else:
    se = np.zeros_like(mean)

  out_dir = os.path.join(_REPO, 'figs', 'metaworld')
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f'{plot_tag}_eval_success.png')

  fig, ax = plt.subplots(figsize=(7.5, 4.2))
  ax.plot(xs, mean, color='#2F6FED', lw=2.0,
          label=f'eval success (n={mat.shape[0]} seed'
                f'{"s" if mat.shape[0] != 1 else ""})')
  if mat.shape[0] > 1:
    ax.fill_between(xs, mean - se, mean + se, color='#2F6FED', alpha=0.2)
  ax.set_ylim(-0.05, 1.05)
  ax.set_xlabel('learner steps')
  ax.set_ylabel('eval success')
  ax.set_title(f'Metaworld eval success — {plot_tag}')
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  ax.grid(True, alpha=0.25)
  ax.legend(loc='best', frameon=False)
  fig.tight_layout()
  fig.savefig(out_path, dpi=140)
  plt.close(fig)
  return out_path


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


def submit_bb_eval(run_dir: str, env_name: str, plot_tag: str, seed: int) -> Optional[str]:
  if not os.path.isfile(_BB_EVAL_JOB):
    print(f'[watch_eval] missing job script: {_BB_EVAL_JOB}', flush=True)
    return None
  env = os.environ.copy()
  env.update({
      'RUN_DIR': run_dir,
      'ENV_NAME': env_name,
      'PLOT_TAG': plot_tag,
      'SEED': str(seed),
  })
  cmd = [
      'sbatch', '--parsable',
      f'--export=ALL,RUN_DIR={run_dir},ENV_NAME={env_name},'
      f'PLOT_TAG={plot_tag},SEED={seed}',
      _BB_EVAL_JOB,
  ]
  print(f'[watch_eval] sbatch BB eval: {" ".join(cmd)}', flush=True)
  r = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
  if r.returncode != 0:
    print(f'[watch_eval] sbatch failed rc={r.returncode}: {r.stderr.strip()}',
          flush=True)
    return None
  job_id = r.stdout.strip().split(';')[0].strip()
  print(f'[watch_eval] submitted job {job_id} for {plot_tag} seed={seed}',
        flush=True)
  return job_id


def handle_builderbench(run_dir: str, rs: dict, labels: Sequence[str],
                        run_cfg: dict, threshold: int) -> str:
  seed = seed_from_run_dir(run_dir)
  plot_tag = rs.get('plot_tag') or resolve_bb_plot_tag(run_dir, seed)
  rs['plot_tag'] = plot_tag
  env_name = str(run_cfg.get('env') or '')
  if not env_name:
    m = _SEED_RE.match(os.path.basename(run_dir))
    env_name = m.group(1) if m else ''
  if not env_name:
    return 'skip(no env)'

  # Refresh done labels from CSV; clear pending if job finished.
  csv_done = bb_eval_csv_labels(plot_tag, seed)
  if csv_done:
    rs['done_labels'] = sorted(csv_done)
  pending_job = rs.get('pending_job_id')
  if pending_job and not squeue_has_job(str(pending_job)):
    print(f'[watch_eval] BB job {pending_job} finished for {plot_tag}',
          flush=True)
    rs['pending_job_id'] = None
    rs['pending_labels'] = []
    rs['done_labels'] = sorted(bb_eval_csv_labels(plot_tag, seed))
    rs['last_action'] = f'job_done:{pending_job}'

  # First sighting with no eval CSV yet: bootstrap so we only track *future*
  # checkpoints (avoids submitting a backlog storm on daemon start).
  if not rs.get('done_labels') and not csv_done and not rs.get('bootstrapped'):
    rs['done_labels'] = list(labels)
    rs['bootstrapped'] = True
    rs['last_action'] = f'bootstrap:n={len(labels)}'
    return f'bootstrap(n={len(labels)})'

  done = set(rs.get('done_labels') or [])
  pending = set(rs.get('pending_labels') or [])
  new = [lab for lab in labels if lab not in done and lab not in pending]
  if len(new) <= threshold:
    return f'ok(new={len(new)})'

  if rs.get('pending_job_id') and squeue_has_job(str(rs['pending_job_id'])):
    return f'waiting_job={rs["pending_job_id"]}(new={len(new)})'

  job_id = submit_bb_eval(run_dir, env_name, plot_tag, seed)
  if not job_id:
    return 'sbatch_failed'
  rs['pending_job_id'] = job_id
  rs['pending_labels'] = list(new)
  rs['last_action'] = f'submitted:{job_id}:n={len(new)}'
  return f'submitted:{job_id}:new={len(new)}'

  if rs.get('pending_job_id') and squeue_has_job(str(rs['pending_job_id'])):
    return f'waiting_job={rs["pending_job_id"]}(new={len(new)})'

  job_id = submit_bb_eval(run_dir, env_name, plot_tag, seed)
  if not job_id:
    return 'sbatch_failed'
  rs['pending_job_id'] = job_id
  rs['pending_labels'] = list(new)
  rs['last_action'] = f'submitted:{job_id}:n={len(new)}'
  return f'submitted:{job_id}:new={len(new)}'


def handle_sawyer(run_dir: str, rs: dict, labels: Sequence[str]) -> str:
  done = set(rs.get('done_labels') or [])
  new = [lab for lab in labels if lab not in done]
  if len(new) <= NEW_CKPT_THRESHOLD:
    return f'ok(new={len(new)})'

  log_dir = os.path.dirname(run_dir)
  plot_tag = plot_tag_for(run_dir)
  out = plot_sawyer_log_dir(log_dir, plot_tag)
  # Mark all current checkpoints under this seed as processed.
  rs['done_labels'] = list(labels)
  rs['last_action'] = f'plotted:{out}'
  # Also bump sibling seeds' done_labels to current if they have fewer new.
  return f'plotted:{out}:new={len(new)}'


def scan_once(log_root: str, state: dict, threshold: int) -> Dict[str, int]:
  counts = defaultdict(int)
  run_dirs = discover_run_dirs(log_root)
  print(f'[watch_eval] discovered {len(run_dirs)} run dir(s) under {log_root}',
        flush=True)

  # Group Sawyer triggers by log_dir so we only replot once per parent.
  sawyer_pending_parents: Set[str] = set()

  for run_dir in run_dirs:
    run_cfg = read_run_config(run_dir)
    kind = classify_run(run_dir, run_cfg)
    if kind is None:
      counts['skipped'] += 1
      continue
    labels = list_checkpoint_labels(os.path.join(run_dir, 'checkpoints'))
    if not labels:
      counts['no_ckpts'] += 1
      continue
    rs = _run_state(state, run_dir)

    if kind == 'builderbench':
      msg = handle_builderbench(run_dir, rs, labels, run_cfg, threshold)
      print(f'[watch_eval] BB  {os.path.basename(os.path.dirname(run_dir))}/'
            f'{os.path.basename(run_dir)}: {msg}', flush=True)
      counts['bb'] += 1
      if msg.startswith('submitted'):
        counts['bb_submitted'] += 1
    else:
      done = set(rs.get('done_labels') or [])
      if not done and not rs.get('bootstrapped'):
        # Bootstrap Sawyer too: only react to future checkpoints.
        rs['done_labels'] = list(labels)
        rs['bootstrapped'] = True
        rs['last_action'] = f'bootstrap:n={len(labels)}'
        counts['sawyer_bootstrap'] += 1
        continue
      new_n = sum(1 for lab in labels if lab not in done)
      if new_n > threshold:
        sawyer_pending_parents.add(os.path.dirname(run_dir))
        counts['sawyer_due'] += 1
      else:
        counts['sawyer_ok'] += 1

  for log_dir in sorted(sawyer_pending_parents):
    plot_tag = os.path.basename(log_dir)
    if plot_tag.startswith('ppo_'):
      plot_tag = plot_tag[len('ppo_'):]
    out = plot_sawyer_log_dir(log_dir, plot_tag)
    print(f'[watch_eval] Sawyer plot {plot_tag} → {out}', flush=True)
    counts['sawyer_plotted'] += 1
    for run_dir in glob.glob(os.path.join(log_dir, 'ppo_*')):
      if not os.path.isdir(run_dir):
        continue
      labels = list_checkpoint_labels(os.path.join(run_dir, 'checkpoints'))
      rs = _run_state(state, os.path.abspath(run_dir))
      rs['done_labels'] = list(labels)
      rs['last_action'] = f'plotted:{out}'

  return dict(counts)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--repo_root', default=_REPO)
  ap.add_argument('--log_root', default=None,
                  help='Root containing logs/ppo_* dirs (default: <repo>/logs).')
  ap.add_argument('--state_file', default=_DEFAULT_STATE)
  ap.add_argument('--once', action='store_true')
  ap.add_argument('--watch_interval', type=float, default=300.0)
  ap.add_argument('--threshold', type=int, default=NEW_CKPT_THRESHOLD,
                  help='Trigger when new checkpoints > threshold (default 2).')
  args = ap.parse_args()

  repo = os.path.abspath(args.repo_root)
  log_root = (args.log_root if args.log_root
              else os.path.join(repo, 'logs'))
  state_path = (args.state_file if os.path.isabs(args.state_file)
                else os.path.join(repo, args.state_file))

  threshold = int(args.threshold)
  print(f'[watch_eval] log_root={log_root}', flush=True)
  print(f'[watch_eval] state={state_path}', flush=True)
  print(f'[watch_eval] threshold=new_ckpts>{threshold}', flush=True)
  print(f'[watch_eval] BB job={_BB_EVAL_JOB}', flush=True)

  try:
    while True:
      ts = time.strftime('%Y-%m-%d %H:%M:%S')
      print(f'\n[watch_eval] === scan @ {ts} ===', flush=True)
      state = _load_state(state_path)
      counts = scan_once(log_root, state, threshold)
      _save_state(state_path, state)
      print(f'[watch_eval] counts={counts}', flush=True)
      if args.once:
        break
      delay = max(60.0, float(args.watch_interval))
      print(f'[watch_eval] next scan in {delay:.0f}s', flush=True)
      time.sleep(delay)
  except KeyboardInterrupt:
    print('\n[watch_eval] stopped.', flush=True)


if __name__ == '__main__':
  main()
