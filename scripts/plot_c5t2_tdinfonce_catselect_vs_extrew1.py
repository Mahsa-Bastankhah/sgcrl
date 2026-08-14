#!/usr/bin/env python3
"""c5t2 TD-InfoNCE: catselect vs catselect+extrew1 (train + eval success).

Writes:
  figs/builderbench/active_train_eval/creative5_task2_tdinfonce_catselect_vs_extrew1_train_success1000.png
  figs/builderbench/active_train_eval/creative5_task2_tdinfonce_catselect_vs_extrew1_eval_success.png

Usage:
  python scripts/plot_c5t2_tdinfonce_catselect_vs_extrew1.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'scripts'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plot_builderbench_train_success1000 as base  # noqa: E402

OUT_TRAIN = (
    'creative5_task2_tdinfonce_catselect_vs_extrew1_train_success1000.png'
)
OUT_EVAL = (
    'creative5_task2_tdinfonce_catselect_vs_extrew1_eval_success.png'
)

# (slurm log basename, legend label, log_dir for iter→steps)
RUNS = [
    (
        'ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_catselect_3672460_0.log',
        'catselect · seed0',
        'ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_catselect',
    ),
    (
        'ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_catselect_extrew1_3677758_0.log',
        'catselect + extrew1 · seed0',
        'ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_catselect_extrew1',
    ),
]


def _subsample(pts: list[tuple[int, float]], max_pts: int = 500):
  n = len(pts)
  if n <= max_pts:
    return pts
  step = max(1, n // max_pts)
  out = pts[::step]
  if out[-1] != pts[-1]:
    out = out + [pts[-1]]
  return out


def plot_train(figs_dir: str = base.FIGS_DIR,
               slurm_dir: str = base.SLURM_DIR) -> str:
  os.makedirs(figs_dir, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10.5, 5.2))
  plotted = 0

  for i, (log_name, label, _) in enumerate(RUNS):
    path = os.path.join(slurm_dir, log_name)
    pts = _subsample(base._parse_train_slurm_path(path))
    if not pts:
      print(f'  train {label}: no data')
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    ls = '--' if 'extrew' in label else '-'
    ax.plot(
        xs, ys,
        color=color, linewidth=2.0, linestyle=ls,
        label=f'{label} (n={len(pts)})', alpha=0.95, zorder=3 + i)
    print(f'  train {label}: {len(pts)} pts  '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(ys):.3f}..{max(ys):.3f}]')
    plotted += 1

  ax.set_title(
      'Creative-5 Task2 — TD-InfoNCE train success_1000\n'
      'catselect vs catselect + extrew1',
      fontsize=12, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel('Train Success (last 1000)', fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if plotted > 0:
    ax.legend(loc='best', fontsize=9, framealpha=0.95)

  out_path = os.path.join(figs_dir, OUT_TRAIN)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}')
  return out_path


def plot_eval(figs_dir: str = base.FIGS_DIR,
              log_root: str = base.LOG_ROOT,
              slurm_dir: str = base.SLURM_DIR) -> str:
  os.makedirs(figs_dir, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10.5, 5.2))
  plotted = 0

  for i, (log_name, label, log_dir_name) in enumerate(RUNS):
    path = os.path.join(slurm_dir, log_name)
    pts_iter = base._parse_eval_slurm_path(path)
    pts = base._iters_to_env_steps(pts_iter, log_root, log_dir_name)
    if not pts:
      print(f'  eval {label}: no data')
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    ls = '--' if 'extrew' in label else '-'
    ax.plot(
        xs, ys,
        color=color, linewidth=2.2, linestyle=ls,
        marker=base.EVAL_MARKERS[i % len(base.EVAL_MARKERS)],
        markersize=6.5, markerfacecolor=color, markeredgecolor='white',
        markeredgewidth=0.6,
        label=f'{label} (n={len(pts)})', alpha=0.95, zorder=3 + i)
    print(f'  eval {label}: {len(pts)} pts  '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(ys):.3f}..{max(ys):.3f}]')
    plotted += 1

  ax.set_title(
      'Creative-5 Task2 — TD-InfoNCE eval success\n'
      'catselect vs catselect + extrew1',
      fontsize=12, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel('Eval Success', fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if plotted > 0:
    ax.legend(loc='best', fontsize=9, framealpha=0.95)

  out_path = os.path.join(figs_dir, OUT_EVAL)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}')
  return out_path


def plot_once(figs_dir: str = base.FIGS_DIR,
              log_root: str = base.LOG_ROOT,
              slurm_dir: str = base.SLURM_DIR) -> list[str]:
  return [
      plot_train(figs_dir=figs_dir, slurm_dir=slurm_dir),
      plot_eval(figs_dir=figs_dir, log_root=log_root, slurm_dir=slurm_dir),
  ]


if __name__ == '__main__':
  plot_once()
