#!/usr/bin/env python3
"""Refresh creative5_task2 train/eval success into active_train_eval/.

Pins the NF baselines already on those plots plus the live norand /
nopermute runs (tiny / compact / original large NF), so success updates
keep overlaying without depending on leftover ``slurm/*.log`` files for
finished configs.

Writes:
  figs/builderbench/active_train_eval/creative5_task2_train_success1000.png
  figs/builderbench/active_train_eval/creative5_task2_eval_success.png

CPU-only (CSV / slurm text; no GPU / no rollouts).

Usage:
  python scripts/plot_c5t2_active_train_eval.py
  python scripts/plot_c5t2_active_train_eval.py --watch 3600
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

LOG_ROOT = base.LOG_ROOT
SLURM_DIR = base.SLURM_DIR
FIGS_DIR = base.FIGS_DIR
GROUP = 'creative5_task2'

# Keep legend order: live norand runs first, then prior NF overlays.
C5T2_RUNS = [
    'ppo_builderbench_creative5_task2_e1024_pd_nf_tiny_sa2x128_r32_b4_w128_tau05_actorreset_nopermute_norand_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_nopermute_norand_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_evalvid_catselect',
    'ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_evalvid_catselect_extrew1',
    'ppo_builderbench_creative5_task2_e1024_pd_nf_tau05_actorreset_nopermute_evalvid_catselect_extrew1',
]


def _runs_with_data(log_root: str, slurm_dir: str) -> list[tuple[str, str]]:
  out: list[tuple[str, str]] = []
  for name in C5T2_RUNS:
    has_csv = base._has_csv(log_root, name, 'learner', min_size=200)
    has_slurm = bool(base._read_train_from_slurm(slurm_dir, name))
    if not has_csv and not has_slurm:
      print(f'  skip (no learner data yet): {name}')
      continue
    out.append((name, base._variant_label(name)))
  return out


def run_once(log_root: str = LOG_ROOT, figs_dir: str = FIGS_DIR,
             slurm_dir: str = SLURM_DIR) -> list[str]:
  os.makedirs(figs_dir, exist_ok=True)
  runs = _runs_with_data(log_root, slurm_dir)
  if not runs:
    print('No creative5_task2 pinned runs with learner data.')
    return []

  labels = ', '.join(base._short_label(lab) for _, lab in runs)
  print(f'{GROUP}: {len(runs)} variants ({labels})')
  outs: list[str] = []

  train_out = base._plot_group(
      log_root, figs_dir, GROUP, runs,
      split='learner', x_col=base.TRAIN_X_COL, y_col=base.TRAIN_METRIC,
      out_suffix='train_success1000',
      title_metric=f'train {base.TRAIN_METRIC}',
      xlabel='Env Steps',
      ylabel='Train Success (last 1000)',
      fmt_x_steps=True,
      slurm_dir=slurm_dir,
  )
  if train_out:
    print(f'  → {train_out}')
    outs.append(train_out)

  eval_runs = []
  for name, label in runs:
    has_csv = base._has_csv(log_root, name, 'eval', min_size=100)
    has_slurm = bool(base._read_eval_from_slurm(slurm_dir, name))
    if has_csv or has_slurm:
      eval_runs.append((name, label))
  if not eval_runs:
    print(f'{GROUP} eval: no runs with eval CSV/slurm lines yet')
    return outs

  eval_out = base._plot_group(
      log_root, figs_dir, GROUP, eval_runs,
      split='eval', x_col=base.EVAL_X_COL, y_col=base.EVAL_METRIC,
      out_suffix='eval_success',
      title_metric=f'eval {base.EVAL_METRIC}',
      xlabel='Env Steps',
      ylabel='Eval Success',
      fmt_x_steps=True,
      marker='o',
      slurm_dir=slurm_dir,
  )
  if eval_out:
    print(f'  → {eval_out}')
    outs.append(eval_out)
  return outs


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--log_root', default=LOG_ROOT)
  p.add_argument('--figs_dir', default=FIGS_DIR)
  p.add_argument('--slurm_dir', default=SLURM_DIR)
  p.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only; login node).')
  args = p.parse_args()

  if args.watch <= 0:
    run_once(args.log_root, args.figs_dir, args.slurm_dir)
    return

  print(f'[c5t2_active_train_eval] watching every {args.watch}s '
        f'→ {args.figs_dir}', flush=True)
  while True:
    t0 = time.time()
    try:
      outs = run_once(args.log_root, args.figs_dir, args.slurm_dir)
      print(f'[c5t2_active_train_eval] updated {len(outs)} plot(s) '
            f'in {time.time() - t0:.1f}s', flush=True)
    except Exception as exc:
      print(f'[c5t2_active_train_eval] ERROR: {exc}', flush=True)
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
