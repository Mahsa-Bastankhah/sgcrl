#!/usr/bin/env python3
"""Plot *current* BuilderBench batch into figs/builderbench/active_run/.

Only uses data from jobs that are RUNNING/PENDING in squeue for the tracked
configs (plus job IDs previously seen in this batch, so finished seeds stay).
Never reads leftover CSV/slurm from older runs that share the same log dir.

Tracks:
  - creative4_task2 NF compact: 1 config × 3 seeds → mean ± stderr shade
  - creative5_task2 NF: 3 configs × 3 seeds → mean ± stderr shade
  - creative4_task1 CRL / NF / TD3 / TDInfoNCE: 3 settings × 1 seed each

CPU-only (slurm stdout parse; no rollouts / no GPU).

Usage:
  python scripts/plot_builderbench_active_run.py
  python scripts/plot_builderbench_active_run.py --watch 7200
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

LOG_ROOT = base.LOG_ROOT
SLURM_DIR = base.SLURM_DIR
FIGS_DIR = '/n/fs/mislresearch/sgcrl/figs/builderbench/active_run'
TRACKED_JOBS_PATH = os.path.join(FIGS_DIR, 'tracked_job_ids.json')

COLORS = base.ACCENT_COLORS
MARKERS = base.EVAL_MARKERS
LINESTYLES = base.EVAL_LINESTYLES
DEFAULT_SPI = base.DEFAULT_STEPS_PER_ITER

# (short_label, log_dir_name)
C4T2_NF = [
    ('compact NF (SA 3×256, r64, b6, w256)',
     'ppo_builderbench_creative4_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_fixedx01_catselect_extrew1'),
]

C5T2_NF = [
    ('permute (no extrew)',
     'ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_evalvid_catselect'),
    ('permute + extrew1',
     'ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_evalvid_catselect_extrew1'),
    ('nopermute + extrew1',
     'ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_nopermute_evalvid_catselect_extrew1'),
]

C4T1_SETTINGS = [
    ('random start + extrew1',
     'actorreset_nopermute_catselect_minstd1e4_extrew1'),
    ('fixed_start_x=0.1 + extrew1',
     'actorreset_nopermute_catselect_minstd1e4_fixedx01_extrew1'),
    ('norand + normobs + extrew1',
     'actorreset_nopermute_norand_normobs_catselect_minstd1e4_extrew1'),
]

C4T1_FAMILIES = {
    'crl': ('CRL', 'pd_crl_tau05'),
    'nf': ('NF', 'pd_nf_tau05'),
    'td3': ('TD3', 'pd_td3_logq_tau05'),
    'tdinfonce': ('TDInfoNCE', 'pd_tdinfonce_tau05'),
}


def _all_tracked_log_dirs() -> list[str]:
  dirs = [d for _, d in C4T2_NF] + [d for _, d in C5T2_NF]
  for mid in (m for _, m in C4T1_FAMILIES.values()):
    for _, suffix in C4T1_SETTINGS:
      dirs.append(f'ppo_builderbench_creative4_task1_e1024_{mid}_{suffix}')
  return dirs


def _coerce(v):
  return base._coerce(v)


def _load_tracked_jobs(path: str = TRACKED_JOBS_PATH) -> dict[str, list[str]]:
  if not os.path.isfile(path):
    return {}
  try:
    with open(path) as f:
      data = json.load(f)
    if isinstance(data, dict):
      return {k: [str(x) for x in v] for k, v in data.items()
              if isinstance(v, list)}
  except Exception:
    pass
  return {}


def _save_tracked_jobs(mapping: dict[str, set[str]],
                       path: str = TRACKED_JOBS_PATH):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  payload = {k: sorted(v) for k, v in sorted(mapping.items()) if v}
  tmp = path + '.tmp'
  with open(tmp, 'w') as f:
    json.dump(payload, f, indent=2)
    f.write('\n')
  os.replace(tmp, path)


def _squeue_log_dir_to_job_ids(tracked: set[str]) -> dict[str, set[str]]:
  """Map tracked log_dir → bare job ids currently in squeue."""
  out: dict[str, set[str]] = defaultdict(set)
  for job_id in base._squeue_job_ids():
    bare = job_id.split('_')[0]
    name = base._log_dir_from_sbatch_script(job_id)
    if name is None:
      # Running jobs usually already have a stdout log.
      patterns = [
          os.path.join(SLURM_DIR, f'*_{job_id}.log'),
          os.path.join(SLURM_DIR, f'*_{bare}_*.log'),
      ]
      matches = []
      for pat in patterns:
        matches.extend(__import__('glob').glob(pat))
      matches = sorted(matches, key=os.path.getmtime, reverse=True)
      for path in matches[:5]:
        name = base._log_dir_from_slurm_log(path)
        if name:
          break
    if name in tracked:
      out[name].add(bare)
  return out


def _refresh_allowed_job_ids(
    tracked_dirs: list[str],
) -> dict[str, set[str]]:
  """Union of previously tracked batch job ids + currently queued/running."""
  tracked_set = set(tracked_dirs)
  allowed: dict[str, set[str]] = defaultdict(set)
  for k, ids in _load_tracked_jobs().items():
    if k in tracked_set:
      allowed[k].update(ids)
  live = _squeue_log_dir_to_job_ids(tracked_set)
  for k, ids in live.items():
    before = set(allowed[k])
    allowed[k].update(ids)
    new = allowed[k] - before
    if new:
      print(f'  track jobs {k}: +{sorted(new)} (live={sorted(ids)})')
  _save_tracked_jobs(allowed)
  return allowed


def _slurm_logs_for_allowed_jobs(
    log_dir_name: str, job_ids: set[str],
) -> list[str]:
  """Only ``{log_dir}_{jobid}[_task].log`` for allowed job ids."""
  if not job_ids:
    return []
  paths: list[str] = []
  for job_id in sorted(job_ids):
    exact_re = re.compile(
        rf'^{re.escape(log_dir_name)}_{re.escape(job_id)}(?:_\d+)?\.log$'
    )
    for path in __import__('glob').glob(
        os.path.join(SLURM_DIR, f'{log_dir_name}_{job_id}*.log')
    ):
      if exact_re.match(os.path.basename(path)):
        paths.append(path)
  return sorted(paths)


def _parse_train_from_path(path: str) -> list[tuple[int, float]]:
  by_x: dict[int, float] = {}
  try:
    with open(path, 'r', errors='replace') as f:
      for line in f:
        if '[Learner]' not in line or 'Train Success 1000' not in line:
          continue
        m = base.TRAIN_SLURM_RE.search(line)
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


def _parse_eval_from_path(path: str) -> list[tuple[int, float]]:
  """Return (env_steps, success) using default steps-per-iter."""
  by_iter: dict[int, float] = {}
  try:
    with open(path, 'r', errors='replace') as f:
      for line in f:
        m = base.EVAL_SLURM_RE.search(line)
        if not m:
          continue
        y = _coerce(m.group('success'))
        if y is None:
          continue
        by_iter[int(m.group('iter'))] = y
  except OSError:
    return []
  # Avoid CSV mapping (would pull old seed folders). Use (iter+1)*spi.
  return [(int((it + 1) * DEFAULT_SPI), y)
          for it, y in sorted(by_iter.items())]


def _read_train_seed_series(log_dir_name: str, job_ids: set[str]):
  out = []
  for path in _slurm_logs_for_allowed_jobs(log_dir_name, job_ids):
    pts = _parse_train_from_path(path)
    if pts:
      out.append(pts)
  return out


def _read_eval_seed_series(log_dir_name: str, job_ids: set[str]):
  out = []
  for path in _slurm_logs_for_allowed_jobs(log_dir_name, job_ids):
    pts = _parse_eval_from_path(path)
    if pts:
      out.append(pts)
  return out


def _aggregate_mean_stderr(seed_series: list[list[tuple[int, float]]]):
  """Per-x mean and standard error across seeds (shade = ±1 stderr)."""
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
      s = math.sqrt(var / len(ys))
    else:
      s = 0.0
    xs.append(x)
    mean.append(m)
    se.append(s)
  return xs, mean, se, n_seeds


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


def _plot_overlay(
    runs: list[tuple[str, str]],
    *,
    split: str,
    title: str,
    out_path: str,
    shade: bool,
    allowed_jobs: dict[str, set[str]],
) -> str | None:
  fig, ax = plt.subplots(figsize=(9.5, 4.8))
  plotted = 0
  ylabel = ('Eval Success' if split == 'eval'
            else 'Train Success (last 1000)')
  for i, (label, log_dir_name) in enumerate(runs):
    job_ids = allowed_jobs.get(log_dir_name, set())
    if not job_ids:
      print(f'  skip {label}: no active/tracked job ids yet')
      continue
    if split == 'eval':
      seed_series = _read_eval_seed_series(log_dir_name, job_ids)
      use_markers = True
    else:
      seed_series = _read_train_seed_series(log_dir_name, job_ids)
      use_markers = False
    xs, mean, se, n = _aggregate_mean_stderr(seed_series)
    if not xs:
      print(f'  skip {label}: jobs={sorted(job_ids)} but no {split} '
            f'slurm lines yet')
      continue
    xs, mean, se = _subsample_curve(xs, mean, se)
    color = COLORS[i % len(COLORS)]
    legend = f'{label} (n={n})'
    kwargs = dict(color=color, linewidth=2.0, label=legend, alpha=0.95,
                  zorder=3 + i)
    if use_markers:
      kwargs.update(
          marker=MARKERS[i % len(MARKERS)],
          markersize=7,
          linestyle=LINESTYLES[i % len(LINESTYLES)],
          markerfacecolor=color,
          markeredgecolor='white',
          markeredgewidth=0.6,
      )
    ax.plot(xs, mean, **kwargs)
    if shade and n > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0,
                      zorder=2 + i)
    print(f'  {label}: jobs={sorted(job_ids)} {len(xs)} {split} pts, '
          f'n_seeds={n}, x=[{xs[0]}..{xs[-1]}] '
          f'y=[{min(mean):.3f}..{max(mean):.3f}]')
    plotted += 1

  if plotted == 0:
    plt.close(fig)
    # Drop stale PNG that may contain old-run dust.
    if os.path.isfile(out_path):
      os.remove(out_path)
      print(f'  removed stale (no active data): {out_path}')
    return None

  ax.set_title(title, fontsize=12, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel(ylabel, fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='best', fontsize=10, framealpha=0.95)
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.tight_layout()
  tmp = out_path + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, out_path)
  plt.close(fig)
  return out_path


def run_once(figs_dir: str = FIGS_DIR) -> list[str]:
  global TRACKED_JOBS_PATH
  TRACKED_JOBS_PATH = os.path.join(figs_dir, 'tracked_job_ids.json')
  os.makedirs(figs_dir, exist_ok=True)
  outs: list[str] = []

  tracked = _all_tracked_log_dirs()
  print('Refreshing allowed job ids from squeue (no old-run CSV)...')
  allowed = _refresh_allowed_job_ids(tracked)
  n_jobs = sum(len(v) for v in allowed.values())
  print(f'  tracking {n_jobs} job id(s) across {len(allowed)} config(s)')

  print('=== creative4_task2 NF compact (1 config × 3 seeds) ===')
  for split, suffix, metric in (
      ('learner', 'train_success1000', 'train success_1000'),
      ('eval', 'eval_success', 'eval success'),
  ):
    out = _plot_overlay(
        C4T2_NF, split=split, shade=True, allowed_jobs=allowed,
        title=f'BuilderBench Creative 4 Task 2 [NF compact] — {metric}',
        out_path=os.path.join(figs_dir, f'creative4_task2_nf_{suffix}.png'),
    )
    if out:
      print(f'  → {out}')
      outs.append(out)

  print('=== creative5_task2 NF (3 configs × 3 seeds) ===')
  for split, suffix, metric in (
      ('learner', 'train_success1000', 'train success_1000'),
      ('eval', 'eval_success', 'eval success'),
  ):
    out = _plot_overlay(
        C5T2_NF, split=split, shade=True, allowed_jobs=allowed,
        title=f'BuilderBench Creative 5 Task 2 [NF] — {metric}',
        out_path=os.path.join(figs_dir, f'creative5_task2_nf_{suffix}.png'),
    )
    if out:
      print(f'  → {out}')
      outs.append(out)

  print('=== creative4_task1 per density estimator (3 settings × 1 seed) ===')
  for fam_key, (fam_title, mid) in C4T1_FAMILIES.items():
    runs = [
        (setting_label,
         f'ppo_builderbench_creative4_task1_e1024_{mid}_{suffix}')
        for setting_label, suffix in C4T1_SETTINGS
    ]
    print(f'-- {fam_title} --')
    for split, suffix, metric in (
        ('learner', 'train_success1000', 'train success_1000'),
        ('eval', 'eval_success', 'eval success'),
    ):
      out = _plot_overlay(
          runs, split=split, shade=False, allowed_jobs=allowed,
          title=(f'BuilderBench Creative 4 Task 1 [{fam_title}] — '
                 f'{metric}'),
          out_path=os.path.join(
              figs_dir, f'creative4_task1_{fam_key}_{suffix}.png'),
      )
      if out:
        print(f'  → {out}')
        outs.append(out)

  stamp = os.path.join(figs_dir, 'LAST_UPDATE.txt')
  with open(stamp, 'w') as f:
    f.write(time.strftime('%Y-%m-%d %H:%M:%S') + f'  plots={len(outs)}\n')
  print(f'done: {len(outs)} plot(s) → {figs_dir}')
  return outs


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--figs_dir', default=FIGS_DIR)
  p.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only login-node watcher).')
  p.add_argument(
      '--reset-tracked-jobs', action='store_true',
      help='Clear tracked_job_ids.json before this pass (fresh batch only).')
  args = p.parse_args()

  if args.reset_tracked_jobs:
    path = os.path.join(args.figs_dir, 'tracked_job_ids.json')
    if os.path.isfile(path):
      os.remove(path)
      print(f'cleared {path}')

  if args.watch <= 0:
    run_once(args.figs_dir)
    return

  print(f'[active_run] watching every {args.watch}s → {args.figs_dir}',
        flush=True)
  while True:
    t0 = time.time()
    try:
      outs = run_once(args.figs_dir)
      print(f'[active_run] updated {len(outs)} plot(s) '
            f'in {time.time() - t0:.1f}s', flush=True)
    except Exception as exc:
      print(f'[active_run] ERROR: {exc}', flush=True)
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
