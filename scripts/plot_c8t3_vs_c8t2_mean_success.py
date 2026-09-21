#!/usr/bin/env python3
"""C8 scattered task3 seed 0 vs final task2 mean success."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(REPO, 'logs', 'final_runs')
OUT = os.path.join(
    REPO, 'figs', 'builderbench', 'final_runs',
    'c8t3_seed0_vs_c8t2_mean_success.png')
MAX_STEPS = 100_000_000

C8T3 = (
    'ppo_builderbench_creative8_task3_e1024_pd_nf_compact_small_'
    'sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_'
    'minstd1e5_ent005_to001_ep100_100m_crl10_dualgradreg_c100_lamlr1e6_'
    'valuedgr_c100_lamlr1e6_warp_2h'
)
C8T2 = (
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact_small_'
    'sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_'
    'minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100_lamlr1e6_'
    'valuedgr_c100_lamlr1e6_warp_4h'
)


def _paths(run_dir: str) -> tuple[str, str]:
  return (
      os.path.join(run_dir, 'logs', 'learner', 'logs.csv'),
      os.path.join(run_dir, 'logs', 'eval', 'logs.csv'),
  )


def _load_run(run_dir: str) -> tuple[list[tuple[int, float]],
                                      list[tuple[int, float]]]:
  learner, eval_path = _paths(run_dir)
  train, iteration_steps = [], {}
  with open(learner, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      step = base._coerce(row.get('global_step', ''))
      iteration = base._coerce(row.get('iteration', ''))
      success = base._coerce(row.get('train_success_1000', ''))
      if step is not None and iteration is not None:
        iteration_steps[int(iteration)] = int(step)
      if step is not None and success is not None and step <= MAX_STEPS:
        train.append((int(step), float(success)))

  spi = base._infer_steps_per_iter(iteration_steps)
  eval_points = []
  with open(eval_path, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      iteration = base._coerce(row.get('iteration', ''))
      success = base._coerce(row.get('success', ''))
      if iteration is None or success is None:
        continue
      iteration = int(iteration)
      step = iteration_steps.get(iteration, int((iteration + 1) * spi))
      if step <= MAX_STEPS:
        eval_points.append((step, float(success)))
  return train, eval_points


def _mean_series(
    series: list[list[tuple[int, float]]],
) -> tuple[list[int], list[float]]:
  grouped: dict[int, list[float]] = defaultdict(list)
  for points in series:
    for x, y in points:
      grouped[x].append(y)
  xs = sorted(grouped)
  return xs, [float(np.mean(grouped[x])) for x in xs]


def main() -> None:
  c8t3_run = os.path.join(
      LOG_ROOT, C8T3, 'ppo_builderbench_creative_8_task3_0')
  t3_train, t3_eval = _load_run(c8t3_run)

  c8t2_root = os.path.join(LOG_ROOT, C8T2)
  seed_dirs = sorted(
      os.path.join(c8t2_root, name)
      for name in os.listdir(c8t2_root)
      if os.path.isdir(os.path.join(c8t2_root, name))
  )
  t2_loaded = [_load_run(path) for path in seed_dirs]
  t2_train = _mean_series([item[0] for item in t2_loaded])
  t2_eval = _mean_series([item[1] for item in t2_loaded])
  t3_train_xy = _mean_series([t3_train])
  t3_eval_xy = _mean_series([t3_eval])

  print(f'c8t3: train={len(t3_train)} eval={len(t3_eval)}')
  print(f'c8t2: seeds={len(seed_dirs)} '
        f'train_mean_points={len(t2_train[0])} eval_mean_points={len(t2_eval[0])}')

  fig, axes = plt.subplots(
      2, 2, figsize=(12.5, 7.6), sharex=True, sharey='row')
  columns = (
      ('c8t3 scattered planar · seed 0', t3_train_xy, t3_eval_xy, '#7B2CBF'),
      (f'c8t2 final · mean of {len(seed_dirs)} seeds',
       t2_train, t2_eval, '#2A9D8F'),
  )
  for col, (title, train, eval_data, color) in enumerate(columns):
    axes[0, col].plot(
        train[0], train[1], color=color, linewidth=1.8,
        label='train_success_1000')
    base._plot_eval_smoothed(
        axes[1, col], eval_data[0], eval_data[1], color=color,
        label=f'eval success (rolling mean, w={base.EVAL_SMOOTH_WINDOW})',
        linewidth=2.2)
    axes[0, col].set_title(title, fontsize=11, fontweight='bold')
    axes[0, col].legend(loc='lower right', fontsize=8.5)
    axes[1, col].legend(loc='lower right', fontsize=8.5)

  axes[0, 0].set_ylabel('Train success (last 1000)')
  axes[1, 0].set_ylabel(
      f'Eval success (rolling mean, window={base.EVAL_SMOOTH_WINDOW})')
  for ax in axes.flat:
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='both', linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  axes[1, 0].set_xlabel('Environment steps')
  axes[1, 1].set_xlabel('Environment steps')
  fig.suptitle(
      'BuilderBench creative-8 success comparison\n'
      'train raw · eval faint raw + centered rolling mean',
      fontsize=13, fontweight='bold')

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'wrote {OUT}')


if __name__ == '__main__':
  main()
