#!/usr/bin/env python3
"""Plot the completed C5T2 TD3 external-reward observation-norm ablation."""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402


REPO_ROOT = '/n/fs/mislresearch/sgcrl'
LOG_ROOT = os.path.join(
    REPO_ROOT, 'logs/final_runs/other_density_estimators')
OUT_DIR = os.path.join(
    REPO_ROOT, 'figs/builderbench/final_runs/other_density_estimators')
OUT_PATH = os.path.join(
    OUT_DIR, 'creative5_task2_td3_extrew_obsnorm_ablation_success.png')
EXPECTED_SEEDS = {2, 3}
MIN_FINAL_STEP = 190_000_000
MAX_PLOT_STEP = 200_000_000


@dataclass(frozen=True)
class Variant:
  label: str
  run_dir: str
  norm_obs: bool
  updates: int
  color: str


VARIANTS = (
    Variant(
        'NormObs/25',
        'ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_'
        'actorreset_nopermute_normobs_evalvid_catselect_extrew1_'
        'ep70_200m_crl25_eval10_warp_4h',
        True,
        25,
        '#3274A1',
    ),
    Variant(
        'NoNorm/25',
        'ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_'
        'actorreset_nopermute_nonormobs_evalvid_catselect_extrew1_'
        'ep70_200m_crl25_eval10_warp_4h',
        False,
        25,
        '#E1812C',
    ),
    Variant(
        'NoNorm/45',
        'ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_'
        'actorreset_nopermute_nonormobs_evalvid_catselect_extrew1_'
        'ep70_200m_crl45_eval10_warp_4h',
        False,
        45,
        '#3A923A',
    ),
)


def _read_csv(path: str, x_col: str, y_col: str) -> list[tuple[int, float]]:
  points: list[tuple[int, float]] = []
  with open(path, newline='') as f:
    for row in csv.DictReader(f):
      x = base._coerce(row.get(x_col, ''))
      y = base._coerce(row.get(y_col, ''))
      if x is not None and y is not None:
        points.append((int(x), float(y)))
  if not points:
    raise RuntimeError(f'no {y_col} data in {path}')
  return sorted(points)


def _validate_config(config: dict, variant: Variant, seed: int) -> None:
  flags = config['flags']
  expected = {
      'env': 'builderbench_creative_5_task2',
      'seed': seed,
      'num_steps': 200_000_000,
      'ppo_repr_mode': 'td3',
      'ppo_norm_obs': variant.norm_obs,
      'ppo_crl_steps_per_iter': variant.updates,
      'ppo_use_external_reward': True,
      'ppo_external_reward_scale': 1.0,
  }
  mismatches = [
      f'{key}={flags.get(key)!r} (expected {value!r})'
      for key, value in expected.items() if flags.get(key) != value
  ]
  expected_log_dir = f'logs/final_runs/other_density_estimators/{variant.run_dir}/'
  if flags.get('log_dir_path') != expected_log_dir:
    mismatches.append(
        f"log_dir_path={flags.get('log_dir_path')!r} "
        f'(expected {expected_log_dir!r})')
  if mismatches:
    raise RuntimeError(
        f'{variant.label} seed {seed} config mismatch: '
        + '; '.join(mismatches))


def _load_variant(
    variant: Variant,
) -> tuple[list[list[tuple[int, float]]],
           list[list[tuple[int, float]]]]:
  root = os.path.join(LOG_ROOT, variant.run_dir)
  if not os.path.isdir(root):
    raise RuntimeError(f'missing exact executed log directory: {root}')

  by_seed: dict[int, tuple[list[tuple[int, float]],
                           list[tuple[int, float]]]] = {}
  for run_name in sorted(os.listdir(root)):
    run_root = os.path.join(root, run_name)
    config_path = os.path.join(run_root, 'run_config.json')
    if not os.path.isfile(config_path):
      continue
    with open(config_path) as f:
      config = json.load(f)
    seed = int(config['seed'])
    if seed not in EXPECTED_SEEDS:
      raise RuntimeError(
          f'{variant.label}: unexpected seed {seed} in {config_path}')
    if seed in by_seed:
      raise RuntimeError(f'{variant.label}: duplicate seed {seed}')
    _validate_config(config, variant, seed)

    train = _read_csv(
        os.path.join(run_root, 'logs/learner/logs.csv'),
        base.TRAIN_X_COL,
        base.TRAIN_METRIC,
    )
    eval_by_iter = _read_csv(
        os.path.join(run_root, 'logs/eval/logs.csv'),
        base.EVAL_X_COL,
        base.EVAL_METRIC,
    )
    iteration_to_step = {
        int(x): int(step)
        for x, step in _read_csv(
            os.path.join(run_root, 'logs/learner/logs.csv'),
            'iteration',
            'global_step',
        )
    }
    steps_per_iter = base._infer_steps_per_iter(iteration_to_step)
    eval_points = [
        (
            iteration_to_step.get(
                iteration, int((iteration + 1) * steps_per_iter)),
            success,
        )
        for iteration, success in eval_by_iter
    ]

    if train[-1][0] < MIN_FINAL_STEP:
      raise RuntimeError(
          f'{variant.label} seed {seed}: train ends at {train[-1][0]:,}')
    if eval_points[-1][0] < MIN_FINAL_STEP:
      raise RuntimeError(
          f'{variant.label} seed {seed}: eval ends at '
          f'{eval_points[-1][0]:,}')
    by_seed[seed] = (train, eval_points)

  if set(by_seed) != EXPECTED_SEEDS:
    raise RuntimeError(
        f'{variant.label}: expected seeds {sorted(EXPECTED_SEEDS)}, '
        f'found {sorted(by_seed)}')

  train_series = [by_seed[seed][0] for seed in sorted(EXPECTED_SEEDS)]
  eval_series = [by_seed[seed][1] for seed in sorted(EXPECTED_SEEDS)]
  print(
      f'{variant.label}: validated seeds 2/3; '
      f'train horizons={[pts[-1][0] for pts in train_series]}; '
      f'eval horizons={[pts[-1][0] for pts in eval_series]}')
  return train_series, eval_series


def _overlap(
    series: list[list[tuple[int, float]]],
) -> list[list[tuple[int, float]]]:
  start = max(points[0][0] for points in series)
  end = min(MAX_PLOT_STEP, min(points[-1][0] for points in series))
  return [
      [(x, y) for x, y in points if start <= x <= end]
      for points in series
  ]


def _plot_sem(ax, xs, mean, sem, color, *, smooth: bool) -> None:
  lower = [max(0.0, m - s) for m, s in zip(mean, sem)]
  upper = [min(1.0, m + s) for m, s in zip(mean, sem)]
  if smooth:
    lower = base._rolling_mean(lower, base.EVAL_SMOOTH_WINDOW)
    upper = base._rolling_mean(upper, base.EVAL_SMOOTH_WINDOW)
  ax.fill_between(xs, lower, upper, color=color, alpha=0.17, linewidth=0)


def main() -> None:
  os.makedirs(OUT_DIR, exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.0), sharey=True)

  for variant in VARIANTS:
    train_series, eval_series = _load_variant(variant)
    train_series = _overlap(train_series)
    eval_series = _overlap(eval_series)
    tx, train_mean, train_sem, train_n = base._aggregate_mean_stderr(
        train_series)
    ex, eval_mean, eval_sem, eval_n = base._aggregate_mean_stderr(eval_series)
    if train_n != 2 or eval_n != 2:
      raise RuntimeError(
          f'{variant.label}: aggregation lost a seed '
          f'(train={train_n}, eval={eval_n})')

    tx, train_mean, train_sem = base._subsample_curve(
        tx, train_mean, train_sem, max_pts=700)
    axes[0].plot(
        tx, train_mean, color=variant.color, linewidth=2.3,
        label=variant.label)
    _plot_sem(
        axes[0], tx, train_mean, train_sem, variant.color, smooth=False)

    eval_smoothed = base._plot_eval_smoothed(
        axes[1],
        ex,
        eval_mean,
        color=variant.color,
        label=variant.label,
        linewidth=2.3,
        window=base.EVAL_SMOOTH_WINDOW,
    )
    _plot_sem(
        axes[1], ex, eval_mean, eval_sem, variant.color, smooth=True)

    print(
        f'{variant.label}: '
        f'train final={train_mean[-1]:.3f}, peak={max(train_mean):.3f}; '
        f'eval smooth final={eval_smoothed[-1]:.3f}, '
        f'peak={max(eval_smoothed):.3f}')

  panel_titles = (
      'Train success (1000 episodes; unsmoothed)',
      f'Eval success (centered rolling mean, '
      f'window={base.EVAL_SMOOTH_WINDOW})',
  )
  for ax, title in zip(axes, panel_titles):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('environment steps')
    ax.set_xlim(0, MAX_PLOT_STEP)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
  axes[0].set_ylabel('success')
  axes[1].legend(
      loc='lower right', fontsize=9.5, framealpha=0.95,
      edgecolor='#555555', fancybox=False)

  fig.suptitle(
      'BuilderBench C5T2 — TD3 + external reward (scale 1), ep70',
      fontsize=13,
      fontweight='bold',
      y=0.98,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
  fig.savefig(
      OUT_PATH, dpi=180, bbox_inches='tight', pad_inches=0.15,
      facecolor='white')
  plt.close(fig)
  print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
  main()
