#!/usr/bin/env python3
"""Plot BuilderBench train/eval success from recent Slurm-backed runs.

By default discovers config dirs that have a matching ``slurm/*.log`` (recently
run jobs), maps each to its ``logs/...`` dir, groups by ``creativeN_taskK``,
and overlays variants. Crowded tasks are also split by method family
(CRL / NF / TD3 / …) under ``{task}_by_method/``. Legends use human-readable
labels and mark currently RUNNING squeue jobs. Multiple seeds of the same
config are averaged with a ±1 standard-error shade. CPU-only (CSV / slurm
text reads; no rollouts).

Writes:
  figs/builderbench/active_train_eval/{task}_train_success1000.png
  figs/builderbench/active_train_eval/{task}_eval_success.png
  figs/builderbench/active_train_eval/{task}_by_method/{task}_*_crl|nf|td3….png

Usage:
  python scripts/plot_builderbench_train_success1000.py
  python scripts/plot_builderbench_train_success1000.py --all
  python scripts/plot_builderbench_train_success1000.py --active
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import time
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_ROOT = '/n/fs/mislresearch/sgcrl/logs'
SLURM_DIR = '/n/fs/mislresearch/sgcrl/slurm'
FIGS_DIR = '/n/fs/mislresearch/sgcrl/figs/builderbench/active_train_eval'
TRAIN_METRIC = 'train_success_1000'
TRAIN_X_COL = 'global_step'  # env steps
EVAL_METRIC = 'success'  # per-eval episode mean
EVAL_X_COL = 'iteration'  # converted to env steps for plotting
DEFAULT_STEPS_PER_ITER = 1024 * 50  # num_envs * rollout_length

ACCENT_COLORS = [
    '#4C9BE8', '#E8834C', '#4CE87A', '#E84C6F',
    '#A84CE8', '#E8D44C', '#4CE8D4', '#E84CA8',
    '#6B8E9F', '#C45C26', '#2A9D8F', '#9B2226',
]
EVAL_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p']
EVAL_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

# ppo_builderbench_creative4_task2_... or ppo_rnd_builderbench_creative4_task2
DIR_RE = re.compile(
    r'^ppo(?:_rnd)?_builderbench_'
    r'(creative(?P<creative>\d+)_task(?P<task>\d+))'
    r'(?P<suffix>.*)$'
)
LOG_DIR_RE = re.compile(r'log_dir=logs/([^/\s]+)/?')
WANDB_DIR_RE = re.compile(r'wandb_dir=logs/([^/\s]+)/?')
# Printed by ppo_learner after each periodic eval (CSV may lag if unflushed).
EVAL_SLURM_RE = re.compile(
    r'\[ppo\] eval iter=(?P<iter>\d+):\s*.*?success=(?P<success>[-+eE0-9.nanINF]+)'
)
# Printed by acme terminal logger each PPO iter (CSV may lag / still empty at start).
TRAIN_SLURM_RE = re.compile(
    r'\[Learner\].*?'
    r'Global Step = (?P<step>[-+eE0-9.nanINF]+).*?'
    r'Train Success 1000 = (?P<success>[-+eE0-9.nanINF]+)'
)


def _fmt_steps(v, _):
  if v == 0:
    return '0'
  if v >= 1e6:
    s = f'{v / 1e6:.1f}M'
    return s.replace('.0M', 'M')
  if v >= 1e3:
    return f'{v / 1e3:.0f}K'
  return str(int(v))


def _variant_label(log_dir_name: str) -> str:
  m = DIR_RE.match(log_dir_name)
  if not m:
    return log_dir_name
  s = (m.group('suffix') or '').lstrip('_')
  prefix = 'rnd_' if log_dir_name.startswith('ppo_rnd_') else ''
  if not s or s == 'e1024':
    return prefix + 'vanilla' if prefix else 'vanilla'
  if s.startswith('e1024_'):
    s = s[len('e1024_'):]
  return prefix + (s or 'default')


def _coerce(v):
  try:
    x = float(v)
    return None if (math.isnan(x) or math.isinf(x)) else x
  except Exception:
    return None


def _squeue_job_ids(user: str | None = None) -> list[str]:
  """Return active job ids as 'JOBID' or 'JOBID_ARRAYTASK'."""
  user = user or os.environ.get('USER', 'mb6458')
  try:
    out = subprocess.check_output(
        ['squeue', '-u', user, '-h', '-o', '%i'],
        text=True,
        stderr=subprocess.DEVNULL,
    )
  except Exception as exc:
    print(f'[warn] squeue failed: {exc}')
    return []
  return [line.strip() for line in out.splitlines() if line.strip()]


def _log_dir_from_slurm_log(path: str) -> str | None:
  """Extract config log-dir name from a slurm stdout log."""
  try:
    with open(path, 'r', errors='replace') as f:
      # Header usually has log_dir= within the first ~120 lines.
      for i, line in enumerate(f):
        if i > 200:
          break
        m = LOG_DIR_RE.search(line)
        if m:
          return m.group(1)
        m = WANDB_DIR_RE.search(line)
        if m:
          return m.group(1)
  except OSError:
    return None

  # Fallback: strip _{jobid}_{task}.log from basename.
  base = os.path.basename(path)
  m = re.match(r'^(?P<name>.+)_(?P<job>\d+)(?:_(?P<task>\d+))?\.log$', base)
  if not m:
    return None
  name = m.group('name')
  if name.startswith('ppo_builderbench_') or name.startswith('ppo_rnd_builderbench_'):
    return name
  return None


def _log_dir_from_sbatch_script(job_id: str) -> str | None:
  """Parse LOG_DIR from the job's .slurm script (works for PENDING jobs)."""
  bare = job_id.split('_')[0]
  try:
    out = subprocess.check_output(
        ['scontrol', 'show', 'job', bare, '-o'],
        text=True,
        stderr=subprocess.DEVNULL,
    )
  except Exception:
    return None
  m = re.search(r'Command=(\S+)', out)
  if not m:
    return None
  script = m.group(1)
  if not os.path.isfile(script):
    return None
  try:
    with open(script, 'r', errors='replace') as f:
      text = f.read()
  except OSError:
    return None
  # Prefer the default LOG_DIR= assignment used by our job scripts.
  for line in text.splitlines():
    if 'LOG_DIR=' not in line or line.strip().startswith('#'):
      continue
    m = re.search(r'logs/([^/"\'\s]+)/?', line)
    if m:
      name = m.group(1)
      if name.startswith('ppo_builderbench_') or name.startswith(
          'ppo_rnd_builderbench_'):
        return name
  return None


def _active_log_dirs(slurm_dir: str) -> set[str]:
  """Map currently queued jobs → unique config log-dir names."""
  active: set[str] = set()
  for job_id in _squeue_job_ids():
    # Prefer exact array-task log: ..._3608680_0.log
    patterns = [
        os.path.join(slurm_dir, f'*_{job_id}.log'),
    ]
    # Also allow basename without array when job_id already includes _task.
    matches = []
    for pat in patterns:
      matches.extend(glob.glob(pat))
    if not matches:
      # job_id may be bare JOBID; try any array task logs for that job.
      bare = job_id.split('_')[0]
      matches = glob.glob(os.path.join(slurm_dir, f'*_{bare}_*.log'))
      matches += glob.glob(os.path.join(slurm_dir, f'*_{bare}.log'))

    # Prefer newest log if multiple.
    matches = sorted(matches, key=os.path.getmtime, reverse=True)
    found = False
    for path in matches[:3]:
      name = _log_dir_from_slurm_log(path)
      if name:
        active.add(name)
        found = True
        break
    if found:
      continue
    # PENDING / just-submitted: no stdout log yet → read batch script.
    name = _log_dir_from_sbatch_script(job_id)
    if name:
      active.add(name)
  return active


def _log_dirs_from_slurm_files(slurm_dir: str) -> set[str]:
  """Config dirs that have a BuilderBench PPO/RND stdout log under slurm/."""
  found: set[str] = set()
  patterns = [
      os.path.join(slurm_dir, 'ppo_builderbench_*.log'),
      os.path.join(slurm_dir, 'ppo_rnd_builderbench_*.log'),
  ]
  for pat in patterns:
    for path in glob.glob(pat):
      name = _log_dir_from_slurm_log(path)
      if name:
        found.add(name)
        continue
      # Fallback: strip trailing _JOBID[_TASK].log from basename.
      base = os.path.basename(path)
      m = re.match(
          r'^(?P<name>ppo(?:_rnd)?_builderbench_.+)_\d+(?:_\d+)?\.log$',
          base)
      if m:
        found.add(m.group('name'))
  return found


def _has_csv(log_root: str, log_dir_name: str, split: str,
             min_size: int = 200) -> bool:
  base = os.path.join(log_root, log_dir_name)
  try:
    run_names = os.listdir(base)
  except OSError:
    return False
  for run_name in run_names:
    path = os.path.join(base, run_name, 'logs', split, 'logs.csv')
    if os.path.isfile(path) and os.path.getsize(path) > min_size:
      return True
  return False


def _discover_runs(log_root: str, mode: str = 'slurm',
                   slurm_dir: str = SLURM_DIR):
  """Return {group_key: [(log_dir_name, label), ...]} for runs with learner data.

  mode:
    'slurm'  — configs with a matching slurm/*.log (default; recent runs)
    'active' — currently RUNNING/PENDING squeue jobs only
    'all'    — every historical log dir under logs/
  """
  groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
  if not os.path.isdir(log_root):
    return groups

  if mode == 'active':
    candidates = sorted(_active_log_dirs(slurm_dir))
    print(f'Active (squeue) training log dirs ({len(candidates)}):')
    for c in candidates:
      print(f'  - {c}')
  elif mode == 'slurm':
    candidates = sorted(_log_dirs_from_slurm_files(slurm_dir))
    print(f'Slurm-backed training log dirs ({len(candidates)}):')
    for c in candidates:
      print(f'  - {c}')
  else:
    candidates = sorted(
        n for n in os.listdir(log_root)
        if n.startswith('ppo_builderbench_creative')
        or n.startswith('ppo_rnd_builderbench_creative')
    )
    print(f'All historical training log dirs ({len(candidates)}):')

  seen: set[str] = set()
  for name in candidates:
    if name in seen:
      continue
    seen.add(name)
    m = DIR_RE.match(name)
    if not m:
      print(f'  skip (unparsed): {name}')
      continue

    log_dir = os.path.join(log_root, name)
    # PENDING jobs are discovered from sbatch scripts before mkdir; create the
    # config dir so the run stays tracked and plots once learner/eval appear.
    if not os.path.isdir(log_dir):
      try:
        os.makedirs(log_dir, exist_ok=True)
        print(f'  created missing log dir: {name}')
      except OSError as exc:
        print(f'  skip (missing dir, mkdir failed): {name} ({exc})')
        continue

    has_csv = _has_csv(log_root, name, 'learner', min_size=200)
    has_slurm_train = bool(_read_train_from_slurm(slurm_dir, name))
    if not has_csv and not has_slurm_train:
      print(f'  skip (no learner csv/slurm yet): {name}')
      continue

    group_key = f"creative{m.group('creative')}_task{m.group('task')}"
    groups[group_key].append((name, _variant_label(name)))

  for key in groups:
    groups[key].sort(key=lambda x: (x[1], x[0]))
  return groups


def _aggregate_mean_stderr(seed_series: list[list[tuple[int, float]]]):
  """Per-x mean and standard error across seeds. Returns (xs, mean, se, n)."""
  by_x: dict[int, list[float]] = defaultdict(list)
  n_seeds = 0
  for pts in seed_series:
    if not pts:
      continue
    n_seeds += 1
    for x, y in pts:
      by_x[int(x)].append(float(y))
  if not by_x:
    return [], [], [], 0
  xs, mean, se = [], [], []
  for x in sorted(by_x):
    ys = by_x[x]
    m = sum(ys) / len(ys)
    if len(ys) > 1:
      var = sum((y - m) ** 2 for y in ys) / (len(ys) - 1)
      s = math.sqrt(var / len(ys))  # stderr
    else:
      s = 0.0
    xs.append(x)
    mean.append(m)
    se.append(s)
  return xs, mean, se, n_seeds


def _mean_seed_series(seed_series: list[list[tuple[int, float]]]):
  xs, mean, _se, _n = _aggregate_mean_stderr(seed_series)
  return list(zip(xs, mean))


def _read_csv_seed_series(log_root: str, log_dir_name: str, *, split: str,
                          x_col: str, y_col: str
                          ) -> list[list[tuple[int, float]]]:
  """One series per seed/run folder under the config dir."""
  base = os.path.join(log_root, log_dir_name)
  seed_series: list[list[tuple[int, float]]] = []
  try:
    run_names = sorted(os.listdir(base))
  except OSError:
    return []

  for run_name in run_names:
    path = os.path.join(base, run_name, 'logs', split, 'logs.csv')
    if not os.path.isfile(path):
      continue
    pts: list[tuple[int, float]] = []
    try:
      with open(path, newline='') as f:
        for row in csv.DictReader(f):
          x = _coerce(row.get(x_col, ''))
          y = _coerce(row.get(y_col, ''))
          if x is not None and y is not None:
            pts.append((int(x), y))
    except Exception:
      continue
    if pts:
      seed_series.append(pts)
  return seed_series


def _read_series(log_root: str, log_dir_name: str, *, split: str,
                 x_col: str, y_col: str):
  """Average seeds if multiple run folders exist under the config dir."""
  return _mean_seed_series(
      _read_csv_seed_series(
          log_root, log_dir_name, split=split, x_col=x_col, y_col=y_col))


def _iter_to_env_steps_map(log_root: str, log_dir_name: str) -> dict[int, int]:
  """Map PPO iteration → env steps (``global_step``) from learner CSV."""
  base = os.path.join(log_root, log_dir_name)
  mapping: dict[int, int] = {}
  try:
    run_names = sorted(os.listdir(base))
  except OSError:
    return mapping
  for run_name in run_names:
    path = os.path.join(base, run_name, 'logs', 'learner', 'logs.csv')
    if not os.path.isfile(path):
      continue
    try:
      with open(path, newline='') as f:
        for row in csv.DictReader(f):
          it = _coerce(row.get('iteration', ''))
          gs = _coerce(row.get('global_step', ''))
          if it is not None and gs is not None:
            mapping[int(it)] = int(gs)
    except Exception:
      continue
  return mapping


def _infer_steps_per_iter(iter_to_steps: dict[int, int]) -> int:
  if len(iter_to_steps) < 2:
    return DEFAULT_STEPS_PER_ITER
  items = sorted(iter_to_steps.items())
  deltas = []
  for (i0, s0), (i1, s1) in zip(items, items[1:]):
    di = i1 - i0
    if di > 0:
      deltas.append((s1 - s0) / di)
  if not deltas:
    return DEFAULT_STEPS_PER_ITER
  deltas.sort()
  return max(1, int(round(deltas[len(deltas) // 2])))


def _iters_to_env_steps(pts_iter: list[tuple[int, float]],
                        log_root: str, log_dir_name: str):
  """Convert (iteration, y) → (env_steps, y)."""
  mapping = _iter_to_env_steps_map(log_root, log_dir_name)
  spi = _infer_steps_per_iter(mapping)
  out: list[tuple[int, float]] = []
  for it, y in pts_iter:
    if it in mapping:
      out.append((mapping[it], y))
    else:
      # Learner logs global_step after each iter: (iter+1) * steps_per_iter.
      out.append((int((it + 1) * spi), y))
  return out


def _slurm_logs_for_log_dir(slurm_dir: str, log_dir_name: str) -> list[str]:
  """Slurm stdout logs that belong to ``log_dir_name`` (not prefix siblings).

  Exact basename form: ``{log_dir_name}_{jobid}.log`` or
  ``{log_dir_name}_{jobid}_{arraytask}.log``.

  The naive glob ``{log_dir_name}_*.log`` wrongly matches longer sibling
  configs (e.g. ``...catselect_*.log`` also hits ``...catselect_extrew1_*.log``).

  If no exact basename match exists (job ``#SBATCH --output`` stem differs
  from ``LOG_DIR``, e.g. ``...catselect_%A_%a.log`` vs ``...catselect_residual``),
  fall back to logs under the same creative/task prefix whose header declares
  this ``log_dir``.
  """
  exact_re = re.compile(rf'^{re.escape(log_dir_name)}_\d+(?:_\d+)?\.log$')
  paths = [
      p for p in glob.glob(os.path.join(slurm_dir, f'{log_dir_name}_*.log'))
      if exact_re.match(os.path.basename(p))
  ]
  if paths:
    return sorted(paths, key=os.path.getmtime, reverse=True)

  m = DIR_RE.match(log_dir_name)
  if not m:
    return []
  if log_dir_name.startswith('ppo_rnd_'):
    prefix = (
        f"ppo_rnd_builderbench_creative{m.group('creative')}"
        f"_task{m.group('task')}"
    )
  else:
    prefix = (
        f"ppo_builderbench_creative{m.group('creative')}"
        f"_task{m.group('task')}"
    )
  fallback: list[str] = []
  for path in glob.glob(os.path.join(slurm_dir, f'{prefix}*.log')):
    if _log_dir_from_slurm_log(path) == log_dir_name:
      fallback.append(path)
  return sorted(fallback, key=os.path.getmtime, reverse=True)


def _parse_eval_slurm_path(path: str) -> list[tuple[int, float]]:
  by_x: dict[int, float] = {}
  try:
    with open(path, 'r', errors='replace') as f:
      for line in f:
        m = EVAL_SLURM_RE.search(line)
        if not m:
          continue
        y = _coerce(m.group('success'))
        if y is None:
          continue
        by_x[int(m.group('iter'))] = y
  except OSError:
    return []
  return [(x, by_x[x]) for x in sorted(by_x)]


def _parse_train_slurm_path(path: str) -> list[tuple[int, float]]:
  by_x: dict[int, float] = {}
  try:
    with open(path, 'r', errors='replace') as f:
      for line in f:
        if '[Learner]' not in line or 'Train Success 1000' not in line:
          continue
        m = TRAIN_SLURM_RE.search(line)
        if not m:
          continue
        x = _coerce(m.group('step'))
        y = _coerce(m.group('success'))
        if x is None or y is None:
          continue
        by_x[int(x)] = y
  except OSError:
    return []
  return [(x, by_x[x]) for x in sorted(by_x)]


def _read_eval_seed_series_from_slurm(
    slurm_dir: str, log_dir_name: str,
) -> list[list[tuple[int, float]]]:
  """One eval series per seed log (array task), as (iteration, success)."""
  out = []
  for path in sorted(_slurm_logs_for_log_dir(slurm_dir, log_dir_name)):
    pts = _parse_eval_slurm_path(path)
    if pts:
      out.append(pts)
  return out


def _read_train_seed_series_from_slurm(
    slurm_dir: str, log_dir_name: str,
) -> list[list[tuple[int, float]]]:
  """One train series per seed log (array task), as (env_steps, success)."""
  out = []
  for path in sorted(_slurm_logs_for_log_dir(slurm_dir, log_dir_name)):
    pts = _parse_train_slurm_path(path)
    if pts:
      out.append(pts)
  return out


def _read_eval_from_slurm(slurm_dir: str, log_dir_name: str):
  """Merged mean series (legacy helper for discovery / has-data checks)."""
  return _mean_seed_series(
      _read_eval_seed_series_from_slurm(slurm_dir, log_dir_name))


def _read_train_from_slurm(slurm_dir: str, log_dir_name: str):
  """Merged mean series (legacy helper for discovery / has-data checks)."""
  return _mean_seed_series(
      _read_train_seed_series_from_slurm(slurm_dir, log_dir_name))


def _prefer_seed_series(
    csv_series: list[list[tuple[int, float]]],
    slurm_series: list[list[tuple[int, float]]],
) -> list[list[tuple[int, float]]]:
  """Prefer the source with more seeds, else more total points."""
  if not csv_series:
    return slurm_series
  if not slurm_series:
    return csv_series
  if len(slurm_series) != len(csv_series):
    return slurm_series if len(slurm_series) > len(csv_series) else csv_series
  n_csv = sum(len(s) for s in csv_series)
  n_slurm = sum(len(s) for s in slurm_series)
  return slurm_series if n_slurm > n_csv else csv_series


def _read_train_seed_series(log_root: str, log_dir_name: str,
                            slurm_dir: str = SLURM_DIR
                            ) -> list[list[tuple[int, float]]]:
  csv_series = _read_csv_seed_series(
      log_root, log_dir_name, split='learner',
      x_col=TRAIN_X_COL, y_col=TRAIN_METRIC)
  slurm_series = _read_train_seed_series_from_slurm(slurm_dir, log_dir_name)
  return _prefer_seed_series(csv_series, slurm_series)


def _read_eval_seed_series(log_root: str, log_dir_name: str,
                           slurm_dir: str = SLURM_DIR
                           ) -> list[list[tuple[int, float]]]:
  """Eval success vs env steps, one series per seed."""
  csv_series = _read_csv_seed_series(
      log_root, log_dir_name, split='eval',
      x_col=EVAL_X_COL, y_col=EVAL_METRIC)
  slurm_series = _read_eval_seed_series_from_slurm(slurm_dir, log_dir_name)
  series_iter = _prefer_seed_series(csv_series, slurm_series)
  out = []
  for pts in series_iter:
    out.append(_iters_to_env_steps(pts, log_root, log_dir_name))
  return out


def _read_train_series(log_root: str, log_dir_name: str,
                       slurm_dir: str = SLURM_DIR):
  """Train success_1000 vs env steps (mean over seeds)."""
  return _mean_seed_series(
      _read_train_seed_series(log_root, log_dir_name, slurm_dir=slurm_dir))


def _read_eval_series(log_root: str, log_dir_name: str,
                      slurm_dir: str = SLURM_DIR):
  """Eval success vs env steps (mean over seeds)."""
  return _mean_seed_series(
      _read_eval_seed_series(log_root, log_dir_name, slurm_dir=slurm_dir))


def _subsample_curve(xs, mean, se, max_pts=500):
  n = len(xs)
  if n <= max_pts:
    return xs, mean, se
  step = max(1, n // max_pts)
  xs_s = xs[::step]
  mean_s = mean[::step]
  se_s = se[::step]
  if xs[-1] != xs_s[-1]:
    xs_s = xs_s + [xs[-1]]
    mean_s = mean_s + [mean[-1]]
    se_s = se_s + [se[-1]]
  return xs_s, mean_s, se_s


def _short_label(label: str) -> str:
  """Compress long BuilderBench variant suffixes for legends."""
  s = label
  for old, new in (
      ('pd_nf_tiny_sa2x128_r32_b4_w128_tau05_', 'nf_tiny_'),
      ('pd_nf_compact_sa3x256_r64_b6_w256_tau05_', 'nf_compact_'),
      ('pd_nf_td_tau05_', 'nf+td_'),
      ('pd_nf_tau05_', 'nf_'),
      ('pd_crl_tau05_', 'crl_'),
      ('pd_td3_fb_logq_tau05_', 'td3_fb_'),
      ('pd_td3_logq_tau05_', 'td3_'),
      ('pd_tdinfonce_tau05_', 'tdinfonce_'),
      ('actorreset_', ''),
      ('evalvid_', ''),
      ('catselect_', 'cat_'),
      ('catwp_', 'catwp_'),
      ('minstd1e5_', 'minstd1e-5_'),
      ('minstd1e4_', 'minstd1e-4_'),
      ('entanneal_', 'ent_'),
      ('nopermute_', 'noperm_'),
      ('permute_rand_', 'permute_'),
      ('fixedx01_', 'fixx_'),
      ('norand_normobs_', 'norand+norm_'),
      ('normobs_', 'norm_'),
      ('extrew10', 'ext10'),
      ('extrew1', 'ext1'),
  ):
    s = s.replace(old, new)
  return s


def _method_family(log_dir_name: str) -> str:
  """Coarse method bucket for splitting crowded task overlays."""
  low = log_dir_name.lower()
  if '_tdinfonce_' in low:
    return 'tdinfonce'
  if '_td3_' in low or '_td3fb_' in low:
    return 'td3'
  if '_nf_td_' in low:
    return 'nf_td'
  if re.search(r'(^|_)nf(_|$)', low) or '_nf_compact_' in low or '_nf_tiny_' in low:
    return 'nf'
  if re.search(r'(^|_)crl(_|$)', low):
    return 'crl'
  return 'other'


FAMILY_TITLE = {
    'crl': 'CRL',
    'nf': 'NF',
    'nf_td': 'NF+TD',
    'td3': 'TD3',
    'tdinfonce': 'TDInfoNCE',
    'other': 'Other',
}


def _friendly_label(log_dir_name: str, raw_suffix: str | None = None) -> str:
  """Human-readable legend for common BuilderBench ablations."""
  s = raw_suffix if raw_suffix is not None else _variant_label(log_dir_name)
  low = log_dir_name.lower()
  fam = FAMILY_TITLE.get(_method_family(log_dir_name), 'Run')

  if 'hitbonus_goal' in low:
    return 'CRL · hit-bonus on goal'
  if 'hitbonus_sf_tol003' in low:
    return 'CRL · hit-bonus SF (tol=0.03)'
  if 'hitbonus_sf_tol005' in low:
    return 'CRL · hit-bonus SF (tol=0.05)'
  if 'hitbonus_sf_tol008' in low:
    return 'CRL · hit-bonus SF (tol=0.08)'
  if 'sfpert' in low:
    return 'CRL · SF perturbation'
  if low.endswith('_succ3') or '_catselect_succ3' in low:
    return f'{fam} · success×3 reward'
  if '_tdinfonce_' in low:
    return 'TDInfoNCE · catselect'
  if '_td3_fb_' in low:
    return 'TD3-FB · noperm + norand + norm'
  if '_td3_logq_' in low and 'tol0013' in low:
    return 'TD3 · catwp tol=0.013'
  if '_td3_logq_' in low:
    return 'TD3 · logQ'
  if '_nf_td_' in low:
    return 'NF+TD · catselect + extrew1'
  if '_nf_compact_' in low:
    bits = ['NF compact']
    if 'permute_rand' in low:
      bits.append('permute')
    elif 'nopermute' in low and 'fixedx01' in low:
      bits.append('nopermute + fixedx')
    elif 'nopermute' in low:
      bits.append('nopermute')
    if 'catwp' in low:
      bits.append('catwp')
    elif 'catselect' in low:
      bits.append('catselect')
    if 'extrew10' in low:
      bits.append('extrew10')
    elif 'extrew1' in low:
      bits.append('extrew1')
    if 'minstd1e5' in low:
      bits.append('minstd1e-5')
    elif 'minstd1e4' in low:
      bits.append('minstd1e-4')
    m = re.search(r'(?:^|_)crl(\d+)(?:_|$)', low)
    if m:
      bits.append(f'CRL steps/iter={m.group(1)}')
    if 'entanneal' in low:
      bits.append('ent-anneal')
    if 'ep60' in low:
      bits.append('ep60 / 300M')
    elif 'ep70' in low:
      bits.append('ep70 / 300M')
    elif 'ep50' in low:
      bits.append('ep50 / 300M')
    return ' · '.join(bits)
  m = re.search(r'(?:^|_)crl(\d+)(?:_|$)', low)
  if '_crl_' in low and m and 'nf_' not in low:
    return f'CRL · steps/iter={m.group(1)}'
  if '_crl_' in low and 'nopermute' in low and 'extrew1' in low:
    return 'CRL · nopermute + fixedx + extrew1'

  return _short_label(s)


def _plot_group(log_root: str, figs_dir: str, group_key: str,
                runs: list[tuple[str, str]], *,
                split: str, x_col: str, y_col: str,
                out_suffix: str, title_metric: str,
                xlabel: str, ylabel: str,
                fmt_x_steps: bool, marker: str | None = None,
                slurm_dir: str = SLURM_DIR,
                active_dirs: set[str] | None = None,
                family_title: str | None = None,
                out_name: str | None = None) -> str | None:
  fig, ax = plt.subplots(figsize=(11.5, 4.8))
  plotted = 0
  active_dirs = active_dirs or set()

  for i, (log_dir_name, label) in enumerate(runs):
    if split == 'eval':
      seed_series = _read_eval_seed_series(
          log_root, log_dir_name, slurm_dir=slurm_dir)
    elif split == 'learner':
      seed_series = _read_train_seed_series(
          log_root, log_dir_name, slurm_dir=slurm_dir)
    else:
      seed_series = _read_csv_seed_series(
          log_root, log_dir_name, split=split, x_col=x_col, y_col=y_col)
    xs, mean, se, n_seeds = _aggregate_mean_stderr(seed_series)
    if not xs:
      print(f'  {group_key}/{label}: no {split} data, skipping')
      continue
    xs, mean, se = _subsample_curve(xs, mean, se)
    color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
    nice = _friendly_label(log_dir_name, label)
    status = ' · RUNNING' if log_dir_name in active_dirs else ''
    legend_label = f'{nice} (n={n_seeds}){status}'
    kwargs = dict(
        color=color, linewidth=2.4 if log_dir_name in active_dirs else 2.0,
        label=legend_label, alpha=0.95,
        zorder=3 + i,
    )
    if split == 'eval' or marker:
      kwargs.update(
          marker=EVAL_MARKERS[i % len(EVAL_MARKERS)],
          markersize=7,
          linestyle=EVAL_LINESTYLES[i % len(EVAL_LINESTYLES)],
          markerfacecolor=color,
          markeredgecolor='white',
          markeredgewidth=0.6,
      )
    ax.plot(xs, mean, **kwargs)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0,
                      zorder=2 + i)
    print(f'  {group_key}/{nice}: n_seeds={n_seeds}, {len(xs)} {split} pts '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(mean):.3f}..{max(mean):.3f}]'
          f'{status}')
    plotted += 1

  if plotted == 0:
    plt.close(fig)
    return None

  creative, task = group_key.split('_task')
  creative_num = creative.replace('creative', '')
  fam_tag = f' [{family_title}]' if family_title else ''
  ax.set_title(
      f'BuilderBench Creative {creative_num} Task {task}{fam_tag} — '
      f'{title_metric} (mean ± stderr)',
      fontsize=12,
      fontweight='bold',
  )
  ax.set_xlabel(xlabel, fontsize=11)
  if fmt_x_steps:
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  ax.set_ylabel(ylabel, fontsize=11)
  # Keep room so y=0 and y=1 markers are not clipped by the frame.
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  leg = ax.legend(
      loc='upper left', bbox_to_anchor=(1.01, 1.0),
      fontsize=9.5, framealpha=1.0, ncol=1,
      edgecolor='#333333', fancybox=False, borderpad=0.6,
      handlelength=2.4, labelspacing=0.45, borderaxespad=0.0,
  )
  leg.get_frame().set_facecolor('white')
  leg.get_frame().set_linewidth(1.2)
  for text in leg.get_texts():
    text.set_fontweight('bold')

  fname = out_name or f'{group_key}_{out_suffix}.png'
  out_path = os.path.join(figs_dir, fname)
  fig.subplots_adjust(right=0.58)
  fig.savefig(
      out_path, dpi=150, bbox_inches='tight',
      bbox_extra_artists=(leg,),
  )
  plt.close(fig)
  return out_path


def _split_runs_by_family(
    runs: list[tuple[str, str]],
) -> dict[str, list[tuple[str, str]]]:
  by_fam: dict[str, list[tuple[str, str]]] = defaultdict(list)
  for name, label in runs:
    by_fam[_method_family(name)].append((name, label))
  for fam in by_fam:
    by_fam[fam].sort(key=lambda x: (_friendly_label(x[0], x[1]), x[0]))
  return by_fam


def run_once(log_root: str, figs_dir: str, mode: str = 'slurm',
             slurm_dir: str = SLURM_DIR) -> list[str]:
  os.makedirs(figs_dir, exist_ok=True)
  groups = _discover_runs(log_root, mode=mode, slurm_dir=slurm_dir)
  if not groups:
    print('No BuilderBench runs with learner CSV/slurm data found.')
    return []

  active_dirs = _active_log_dirs(slurm_dir)
  if active_dirs:
    print(f'Currently RUNNING/PENDING ({len(active_dirs)}):')
    for d in sorted(active_dirs):
      print(f'  * {d}')

  print(f'Found {len(groups)} task group(s)')
  outs: list[str] = []

  def _plot_both(group_key: str, runs: list[tuple[str, str]], *,
                 family_title: str | None = None,
                 subdir: str | None = None) -> None:
    nonlocal outs
    dest = os.path.join(figs_dir, subdir) if subdir else figs_dir
    os.makedirs(dest, exist_ok=True)
    fam_slug = (
        family_title.lower().replace('+', 'p').replace(' ', '_')
        if family_title else None)
    train_name = (
        f'{group_key}_train_success1000_{fam_slug}.png'
        if fam_slug else f'{group_key}_train_success1000.png')
    eval_name = (
        f'{group_key}_eval_success_{fam_slug}.png'
        if fam_slug else f'{group_key}_eval_success.png')

    labels = ', '.join(_friendly_label(n, l) for n, l in runs)
    print(f'  {group_key} train'
          f'{f" [{family_title}]" if family_title else ""}: '
          f'{len(runs)} variants ({labels})')
    out = _plot_group(
        log_root, dest, group_key, runs,
        split='learner', x_col=TRAIN_X_COL, y_col=TRAIN_METRIC,
        out_suffix='train_success1000',
        title_metric=f'train {TRAIN_METRIC}',
        xlabel='Env Steps',
        ylabel='Train Success (last 1000)',
        fmt_x_steps=True,
        slurm_dir=slurm_dir,
        active_dirs=active_dirs,
        family_title=family_title,
        out_name=train_name,
    )
    if out:
      print(f'    → {out}')
      outs.append(out)

    eval_runs = []
    for name, label in runs:
      has_csv = _has_csv(log_root, name, 'eval', min_size=100)
      has_slurm = bool(_read_eval_from_slurm(slurm_dir, name))
      if has_csv or has_slurm:
        eval_runs.append((name, label))
    if not eval_runs:
      print(f'  {group_key} eval'
            f'{f" [{family_title}]" if family_title else ""}: '
            f'no runs with eval CSV/slurm lines')
      return
    elabels = ', '.join(_friendly_label(n, l) for n, l in eval_runs)
    print(f'  {group_key} eval'
          f'{f" [{family_title}]" if family_title else ""}: '
          f'{len(eval_runs)} variants ({elabels})')
    eout = _plot_group(
        log_root, dest, group_key, eval_runs,
        split='eval', x_col=EVAL_X_COL, y_col=EVAL_METRIC,
        out_suffix='eval_success',
        title_metric=f'eval {EVAL_METRIC}',
        xlabel='Env Steps',
        ylabel='Eval Success',
        fmt_x_steps=True,
        marker='o',
        slurm_dir=slurm_dir,
        active_dirs=active_dirs,
        family_title=family_title,
        out_name=eval_name,
    )
    if eout:
      print(f'    → {eout}')
      outs.append(eout)

  for group_key, runs in sorted(groups.items()):
    # Always write the combined task overlay.
    _plot_both(group_key, runs)

    # When a task mixes methods (or is crowded), also write per-family plots
    # so legends stay readable.
    by_fam = _split_runs_by_family(runs)
    if len(by_fam) > 1 or len(runs) > 4:
      fam_dir = os.path.join(group_key + '_by_method')
      print(f'  {group_key}: splitting into {len(by_fam)} method families → '
            f'{fam_dir}/')
      for fam, fam_runs in sorted(
          by_fam.items(), key=lambda kv: FAMILY_TITLE.get(kv[0], kv[0])):
        _plot_both(
            group_key, fam_runs,
            family_title=FAMILY_TITLE.get(fam, fam),
            subdir=fam_dir,
        )
  return outs


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--log_root', default=LOG_ROOT)
  p.add_argument('--figs_dir', default=FIGS_DIR)
  p.add_argument('--slurm_dir', default=SLURM_DIR)
  src = p.add_mutually_exclusive_group()
  src.add_argument(
      '--all', action='store_true',
      help='Plot every historical log dir under logs/.')
  src.add_argument(
      '--active', action='store_true',
      help='Plot only currently RUNNING/PENDING squeue jobs.')
  p.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only; never deletes PNGs).')
  args = p.parse_args()
  if args.all:
    mode = 'all'
  elif args.active:
    mode = 'active'
  else:
    mode = 'slurm'

  if args.watch <= 0:
    run_once(args.log_root, args.figs_dir, mode=mode,
             slurm_dir=args.slurm_dir)
    return

  print(f'[success_plots] watching every {args.watch}s '
        f'(mode={mode}; train+eval; no prune)', flush=True)
  while True:
    t0 = time.time()
    try:
      outs = run_once(args.log_root, args.figs_dir, mode=mode,
                      slurm_dir=args.slurm_dir)
      print(f'[success_plots] updated {len(outs)} plot(s) '
            f'in {time.time() - t0:.1f}s', flush=True)
    except Exception as exc:
      print(f'[success_plots] ERROR: {exc}', flush=True)
    time.sleep(args.watch)


if __name__ == '__main__':
  main()

