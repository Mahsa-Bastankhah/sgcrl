#!/usr/bin/env python3
"""c4t2 TD-NF: catselect vs catselect+extrew1 (train + eval success).

Mean ± stderr across seeds from the given Slurm array tasks.

Writes:
  figs/builderbench/active_train_eval/creative4_task2_nf_td_catselect_vs_extrew1_train_success1000.png
  figs/builderbench/active_train_eval/creative4_task2_nf_td_catselect_vs_extrew1_eval_success.png

Usage:
  python scripts/plot_c4t2_nf_td_catselect_vs_extrew1.py
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
    'creative4_task2_nf_td_catselect_vs_extrew1_train_success1000.png'
)
OUT_EVAL = (
    'creative4_task2_nf_td_catselect_vs_extrew1_eval_success.png'
)

# (legend label, linestyle, log_dir for iter→steps, [slurm log basenames])
GROUPS = [
    (
        'catselect',
        '-',
        'ppo_builderbench_creative4_task2_e1024_pd_nf_td_tau05_catselect',
        [
            'ppo_builderbench_creative4_task2_e1024_pd_nf_td_tau05_catselect_3677642_0.log',
            'ppo_builderbench_creative4_task2_e1024_pd_nf_td_tau05_catselect_3677642_1.log',
        ],
    ),
    (
        'catselect + extrew1',
        '--',
        'ppo_builderbench_creative4_task2_e1024_pd_nf_td_tau05_catselect_extrew1',
        [
            'ppo_builderbench_creative4_task2_e1024_pd_nf_td_tau05_catselect_extrew1_3677618_0.log',
            'ppo_builderbench_creative4_task2_e1024_pd_nf_td_tau05_catselect_extrew1_3677618_1.log',
        ],
    ),
]


def plot_train(figs_dir: str = base.FIGS_DIR,
               slurm_dir: str = base.SLURM_DIR) -> str:
  os.makedirs(figs_dir, exist_ok=True)
  fig, ax = plt.subplots(figsize=(10.5, 5.2))
  plotted = 0

  for i, (label, ls, _log_dir, log_names) in enumerate(GROUPS):
    seed_series = []
    for log_name in log_names:
      path = os.path.join(slurm_dir, log_name)
      pts = base._parse_train_slurm_path(path)
      if pts:
        seed_series.append(pts)
        print(f'  train {label} · {log_name}: {len(pts)} pts')
      else:
        print(f'  train {label} · {log_name}: no data')
    xs, mean, se, n_seeds = base._aggregate_mean_stderr(seed_series)
    if not xs:
      print(f'  train {label}: no aggregated data')
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    ax.plot(
        xs, mean,
        color=color, linewidth=2.2, linestyle=ls,
        label=f'{label} (n={n_seeds})', alpha=0.95, zorder=3 + i)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0,
                      zorder=2 + i)
    print(f'  train {label}: n_seeds={n_seeds}, {len(xs)} pts  '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(mean):.3f}..{max(mean):.3f}]')
    plotted += 1

  ax.set_title(
      'Creative-4 Task2 — TD-NF train success_1000\n'
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

  for i, (label, ls, log_dir_name, log_names) in enumerate(GROUPS):
    seed_series = []
    for log_name in log_names:
      path = os.path.join(slurm_dir, log_name)
      pts_iter = base._parse_eval_slurm_path(path)
      pts = base._iters_to_env_steps(pts_iter, log_root, log_dir_name)
      if pts:
        seed_series.append(pts)
        print(f'  eval {label} · {log_name}: {len(pts)} pts')
      else:
        print(f'  eval {label} · {log_name}: no data')
    xs, mean, se, n_seeds = base._aggregate_mean_stderr(seed_series)
    if not xs:
      print(f'  eval {label}: no aggregated data')
      continue
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    ax.plot(
        xs, mean,
        color=color, linewidth=2.2, linestyle=ls,
        marker=base.EVAL_MARKERS[i % len(base.EVAL_MARKERS)],
        markersize=6.5, markerfacecolor=color, markeredgecolor='white',
        markeredgewidth=0.6,
        label=f'{label} (n={n_seeds})', alpha=0.95, zorder=3 + i)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0,
                      zorder=2 + i)
    print(f'  eval {label}: n_seeds={n_seeds}, {len(xs)} pts  '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(mean):.3f}..{max(mean):.3f}]')
    plotted += 1

  ax.set_title(
      'Creative-4 Task2 — TD-NF eval success\n'
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
