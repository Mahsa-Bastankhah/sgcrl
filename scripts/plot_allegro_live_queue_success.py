#!/usr/bin/env python3
"""Train/eval success for currently running/queued Allegro slurm jobs.

Re-reads squeue each refresh. Train is raw last-1000. Eval is faint raw +
bold rolling mean (window=5).

  python scripts/plot_allegro_live_queue_success.py
  python scripts/plot_allegro_live_queue_success.py --watch 120 --watch-max-sec 21600

Writes NF/MPO to ``akt_live_queue_success.png`` and RND (own colors) to
``akt_live_queue_rnd_success.png``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_allegro_hidetable_running_success as hidetable  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_live_queue_success.png')
OUT_PATH_RND = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_live_queue_rnd_success.png')
AUTO_RND_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'auto_nf_spawned_rnd.json')
EVAL_WINDOW = 5
C = base.ACCENT_COLORS
RND_COLORS = [
    '#1B9E4B', '#E67E22', '#4C9BE8', '#E84C6F',
    '#A84CE8', '#2A9D8F', '#C45C26', '#6B8E9F',
]

# Job-name / log-dir token → legend label. Discovery still plots unknown jobs.
LABELS = (
    ('hard_bucket_hover_near10', 'hard-bucket hover near10  z=0.55 T=80 300M  in-bucket  (0.45,-0.60)  hard 0.73m'),
    ('mid_bucket_hover_near10', 'mid-bucket hover near10  z=0.55 T=80 300M  in-bucket  (0.50,-0.45)  mid 0.61m'),
    ('hard_bucket_arm_init', 'hard-bucket arm-init  z=0.55 T=80 300M  in-bucket  (0.45,-0.60)  hard 0.73m'),
    ('nvidiainit_ep150_fut80_z055', 'NVIDIA-init T=150 fut80 z=0.55 500M  norand  in-bucket  (0.50,-0.30)  default 0.54m'),
    ('nvidiainit_ep100_fut50_norand', 'NVIDIA-init T=100 fut50 500M  norand  in-bucket  (0.50,-0.30)  default 0.54m'),
    ('nvidiainit_ep100_fut50', 'NVIDIA-init T=100 fut50 1B  in-bucket  (0.50,-0.30)  default 0.54m'),
    ('bkt_045m060_q40_ep70', 'keep-arm T=70 q40 1B  in-bucket  (0.45,-0.60)  hard 0.73m'),
    ('mpo_near10_045m060_ep70_300m', 'MPO hover T=70 300M  in-bucket  (0.45,-0.60)  hard 0.73m'),
    ('near10_045m060_q40_ball75', 'hover near10 T=70 q40 200M  goal-ball 7.5cm  (0.45,-0.60)  hard 0.73m'),
    ('near10_045m060_q40', 'hover near10 T=70 q40 1B  in-bucket  (0.45,-0.60)  hard 0.73m'),
    ('near10_050045_q05', 'hover near10 T=70 q05 500M  in-bucket  (0.50,-0.45)  mid 0.61m'),
    ('rnd_keeparm_050m030_ball75', 'RND keep-arm T=50 200M  goal-ball 7.5cm  (0.50,-0.30)  easy 0.54m'),
    ('rnd_keeparm_045m060_ep70_200m', 'RND keep-arm T=70 200M  in-bucket  (0.45,-0.60)  hard 0.73m'),
    ('rnd_near10_045m060_ep70_200m', 'RND hover T=70 200M  in-bucket  (0.45,-0.60)  hard 0.73m'),
    ('rnd_near10_050045_ep70_200m', 'RND hover T=70 200M  in-bucket  (0.50,-0.45)  mid 0.61m'),
    ('rnd_near10_050045', 'RND hover T=70 500M  in-bucket  (0.50,-0.45)  mid 0.61m'),
    ('rnd_keeparm_xy0022_ball75', 'RND keep-arm hidetable T=50 50M  goal-ball 7.5cm  (0.00,-0.22)'),
    ('rnd_keeparm_000m040_ball75', 'RND keep-arm hidetable T=50 50M  goal-ball 7.5cm  (0.00,-0.40)'),
    ('keeparm_050030_ball75', 'keep-arm T=50 q05 500M  goal-ball 7.5cm  (0.50,-0.30)  default 0.54m'),
    ('mpo_keeparm_050m030_ball75', 'MPO keep-arm T=50 300M  goal-ball 7.5cm  (0.50,-0.30)  easy 0.54m'),
    ('mpo_keeparm_000m040_ball75', 'MPO keep-arm hidetable T=50 50M  goal-ball 7.5cm  (0.00,-0.40)'),
    ('keeparm_000m040_ball75', 'keep-arm hidetable T=50 q05 500M  goal-ball 7.5cm  (0.00,-0.40)'),
    ('near10_nvg_500m', 'hover near10 T=50 q05 500M  NV-goal-ball 11.25cm  (0.50,-0.30)  default 0.54m'),
    ('xy000m040', 'keep-arm hidetable T=50 q05 500M  goal-ball 7.5cm  (0.00,-0.40)'),
    ('xy0022_ball75', 'keep-arm hidetable T=50 q05 400M  goal-ball 7.5cm  (0.00,-0.22)'),
    ('xy0022_nvg', 'keep-arm hidetable T=50 q05 400M  NV-goal-ball 11.25cm  (0.00,-0.22)'),
)

# NF success-candidate parents + z=0.55 variants live on the dedicated figure.
SKIP_FROM_LIVE_NF = (
    'successful_candidate',
    'hard_bucket',
    'mid_bucket',
    'z055',
    'bkt_045m060_q40_ep70',
    'near10_045m060_q40_ep70_1b',
    'near10_050045_q05',
    'tablespawn_near10_above10_corr03_nvbucket_xy045m060_ep70_1b',
    'tablespawn_near10_above10_corr03_nvbucket_xy050045_noshape_ep70_500m',
    'keeparm_palmup_padhold_nvbucket_xy045m060_norand_ep70_1b',
)


def _is_success_candidate_nf(job_name: str, log_dir: str, kind: str) -> bool:
  if kind != 'nf':
    return False
  blob = f'{job_name} {log_dir}'
  # NVIDIA-init z=0.55 stays on the live queue (parent already lives there).
  if 'nvidiainit' in blob:
    return False
  return any(token in blob for token in SKIP_FROM_LIVE_NF)


# Finished / cancelled runs kept on the figure for comparison.
EXTRA_COMPARISON = (
    (
        'RND hover T=70 200M  in-bucket  (0.45,-0.60)  hard 0.73m  s0',
        'ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03'
        '_nvbucket_xy045m060_noshape_ep70_200m_inbucket_seed0_3h',
        'rnd',
        '#1B9E4B',
        'cmp-rnd-near10-045m060',
    ),
    (
        'RND hover T=70 200M  in-bucket  (0.50,-0.45)  mid 0.61m  s0',
        'ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03'
        '_nvbucket_xy050045_noshape_ep70_200m_inbucket_seed0_3h',
        'rnd',
        '#E67E22',
        'cmp-rnd-near10-050045',
    ),
)


def _auto_rnd() -> dict:
  if not os.path.isfile(AUTO_RND_PATH):
    return {'labels': [], 'extras': []}
  try:
    with open(AUTO_RND_PATH, encoding='utf-8') as fh:
      data = json.load(fh)
  except (OSError, json.JSONDecodeError):
    return {'labels': [], 'extras': []}
  if not isinstance(data, dict):
    return {'labels': [], 'extras': []}
  return {
      'labels': list(data.get('labels') or []),
      'extras': list(data.get('extras') or []),
  }


def _with_hardness(label: str) -> str:
  if 'hard 0.73m' in label or 'mid 0.61m' in label or 'default 0.54m' in label:
    return label
  if '(0.45,-0.60)' in label:
    return f'{label}  hard 0.73m'
  if '(0.50,-0.45)' in label:
    return f'{label}  mid 0.61m'
  if '(0.50,-0.30)' in label:
    return f'{label}  default 0.54m'
  return label


def _label_for(job_name: str, log_dir: str, seed: str) -> str:
  blob = f'{job_name} {log_dir}'
  for token, label in _auto_rnd().get('labels', []):
    if token and token in blob:
      return _with_hardness(f'{label}  s{seed}')
  for token, label in LABELS:
    if token in blob:
      return _with_hardness(f'{label}  s{seed}')
  return _with_hardness(f'{job_name}  s{seed}')


def _kind(job_name: str, log_dir: str) -> str:
  blob = f'{job_name} {log_dir}'
  if ('_rnd_' in blob or blob.startswith('akt_rnd')
      or '/ppo_rnd_' in blob or log_dir.startswith('ppo_rnd_')):
    return 'rnd'
  if ('_mpo_' in blob or blob.startswith('akt_mpo')
      or log_dir.startswith('mpo_')):
    return 'mpo'
  return 'nf'


def _squeue_jobs() -> list[tuple[str, str, str]]:
  """(job_id, job_name, state) for this user."""
  try:
    out = subprocess.check_output(
        ['squeue', '-u', os.environ.get('USER', ''),
         '-h', '-o', '%i|%j|%T'],
        text=True)
  except (subprocess.CalledProcessError, FileNotFoundError):
    return []
  rows = []
  for line in out.splitlines():
    parts = line.strip().split('|')
    if len(parts) != 3:
      continue
    rows.append((parts[0], parts[1], parts[2]))
  return rows


def _scontrol_command(job_id: str) -> str:
  try:
    out = subprocess.check_output(
        ['scontrol', 'show', 'job', job_id, '-o'], text=True)
  except (subprocess.CalledProcessError, FileNotFoundError):
    return ''
  m = re.search(r'Command=(\S+)', out)
  return m.group(1) if m else ''


def _log_dir_from_script(script: str, seed: str) -> str:
  if not script or not os.path.isfile(script):
    return ''
  text = open(script, encoding='utf-8', errors='replace').read()
  m = re.search(r'LOG_DIR="([^"]+)"', text)
  if not m:
    return ''
  raw = m.group(1)
  raw = raw.replace('${SEED}', seed).replace('${SLURM_ARRAY_TASK_ID}', seed)
  raw = raw.replace('${seed}', seed)
  raw = raw.split('logs/', 1)[-1] if 'logs/' in raw else raw
  return raw.strip().strip('/')


def _seed_from_job_id(job_id: str) -> str:
  if '_' in job_id:
    return job_id.rsplit('_', 1)[-1]
  return '0'


def discover_methods():
  jobs = _squeue_jobs()
  methods = []
  seen = set()
  color_i = 0
  rnd_i = 0
  extra_colors = ['#9B2226', '#2A9D8F', '#C45C26', '#6B8E9F']
  palette = list(C) + extra_colors
  for job_id, job_name, state in jobs:
    if not (job_name.startswith('akt_') or 'allegro' in job_name.lower()):
      continue
    seed = _seed_from_job_id(job_id)
    script = _scontrol_command(job_id)
    log_dir = _log_dir_from_script(script, seed)
    if not log_dir:
      print(f'skip {job_id} {job_name}: no LOG_DIR', flush=True)
      continue
    key = (log_dir, seed)
    if key in seen:
      continue
    seen.add(key)
    label = _label_for(job_name, log_dir, seed)
    if state not in ('RUNNING', 'R'):
      label = f'{label}  [{state}]'
    kind = _kind(job_name, log_dir)
    if _is_success_candidate_nf(job_name, log_dir, kind):
      print(f'skip {job_id} {job_name}: success-candidate → dedicated plot',
            flush=True)
      continue
    if kind == 'rnd':
      color = RND_COLORS[rnd_i % len(RND_COLORS)]
      rnd_i += 1
    elif kind == 'mpo':
      color = '#6B2D8B'
    else:
      color = palette[color_i % len(palette)]
      color_i += 1
    methods.append((label, log_dir, kind, color, job_id))
  auto_extras = []
  for extra in _auto_rnd().get('extras', []):
    if len(extra) >= 5:
      auto_extras.append(tuple(extra[:5]))
  for extra in tuple(auto_extras) + EXTRA_COMPARISON:
    label, log_dir, kind, color, job_id = extra
    key = (log_dir, 'cmp')
    if key in seen or any(m[1] == log_dir for m in methods):
      continue
    seen.add(key)
    methods.append((label, log_dir, kind, color, job_id))
  return methods


def _draw_methods(axes, methods, *, rnd: bool) -> list[str]:
  missing = []
  if not methods:
    axes[0].text(0.5, 0.5,
                 'no Allegro RND/MPO jobs' if rnd else 'no Allegro jobs in squeue',
                 ha='center', va='center', transform=axes[0].transAxes)
  for label, log_dir, kind, color, job_id in methods:
    ls = hidetable._ls(kind)
    print(f'[{job_id}] {label}  {log_dir}', flush=True)
    if kind == 'rnd':
      ev = [hidetable._read_rnd_eval(log_dir)]
      ev = [e for e in ev if e]
      ex, emean, ese, en = base._aggregate_mean_stderr(ev)
      if ex:
        hidetable._plot_train(
            axes[0], ex, emean, ese, en, color,
            f'{label}  (eval only)', ls='-')
        hidetable._plot_eval(axes[1], ex, emean, ese, en, color, label, ls='-')
        for ax in axes:
          ax.plot(ex, emean, color=color, ls='none', marker='D',
                  markersize=6, zorder=6)
      else:
        missing.append(label)
        print(f'{label} eval: no data', flush=True)
      continue
    train = base._read_train_seed_series(base.LOG_ROOT, log_dir)
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train)
    if tx:
      hidetable._plot_train(axes[0], tx, tmean, tse, tn, color, label, ls=ls)
    else:
      missing.append(label)
      print(f'{label} train: no data', flush=True)
    ev = base._read_eval_seed_series(base.LOG_ROOT, log_dir)
    ex, emean, ese, en = base._aggregate_mean_stderr(ev)
    if ex:
      hidetable._plot_eval(axes[1], ex, emean, ese, en, color, label, ls=ls)
    else:
      print(f'{label} eval: no data', flush=True)
  return missing


def _save_pair(methods, out_path: str, title: str, *, rnd: bool) -> None:
  fig, axes = plt.subplots(2, 1, figsize=(12.0, 9.2), sharex=True)
  missing = _draw_methods(axes, methods, rnd=rnd)
  axes[0].set_title(f'{title} — train success (raw last-1000)',
                    fontsize=12, fontweight='bold')
  hidetable._style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper left', fontsize=7.4, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (faint raw + bold rolling mean, window={EVAL_WINDOW})'
      '\nbucket hardness from hover cube (0.20,0.08):  '
      '(0.45,−0.60) hard 0.73m  >  (0.50,−0.45) mid 0.61m  >  '
      '(0.50,−0.30) default 0.54m',
      fontsize=11, fontweight='bold')
  hidetable._style(axes[1], f'eval success (roll mean, w={EVAL_WINDOW})',
                   xlabel=True)
  axes[1].legend(loc='upper left', fontsize=7.4, framealpha=0.95)
  if missing:
    axes[0].text(
        0.99, 0.02,
        'no logs yet:\n' + '\n'.join('• ' + m for m in missing),
        transform=axes[0].transAxes, ha='right', va='bottom',
        fontsize=7.4, color='#555555')
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}', flush=True)


def run_once() -> None:
  methods = discover_methods()
  methods = [m for m in methods
             if not _is_success_candidate_nf(m[0], m[1], m[2])]
  nf = [m for m in methods if m[2] != 'rnd']
  rnd = [m for m in methods if m[2] in ('rnd', 'mpo')]
  _save_pair(nf, OUT_PATH, 'running + queued Allegro NF/MPO', rnd=False)
  _save_pair(rnd, OUT_PATH_RND, 'running + queued Allegro RND + MPO', rnd=True)
  try:
    import plot_allegro_successful_candidate as cand  # noqa: E402
    cand.run_once()
  except Exception as exc:  # noqa: BLE001
    print(f'successful-candidate plot skipped: {exc}', flush=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--watch', type=int, default=0,
                      help='Refresh interval in seconds; 0 = once.')
  parser.add_argument('--watch-max-sec', type=int, default=0,
                      help='Stop after this many seconds; 0 = until killed.')
  args = parser.parse_args()
  if args.watch <= 0:
    run_once()
    return
  t0 = time.time()
  print(
      f'[live queue] watching every {args.watch}s '
      f'(max {args.watch_max_sec or "∞"}s) → {OUT_PATH} + {OUT_PATH_RND}',
      flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001
      print(f'[live queue] error: {exc}', flush=True)
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[live queue] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
