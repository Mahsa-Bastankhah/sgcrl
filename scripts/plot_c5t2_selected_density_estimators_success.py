#!/usr/bin/env python3
"""Plot the three selected (not config-controlled) C5T2 density recipes."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


REPO_ROOT = '/n/fs/mislresearch/sgcrl'
LOG_ROOT = os.path.join(REPO_ROOT, 'logs')
SLURM_ROOT = os.path.join(
    REPO_ROOT, 'slurm/final_runs/other_density_estimators')
OUT_PATH = os.path.join(
    REPO_ROOT,
    'figs/builderbench/final_runs/other_density_estimators',
    'creative5_task2_selected_crl10_td3_extrew_normobs25_'
    'tdinfonce25_success.png',
)
MIN_FINAL_STEP = 190_000_000
EXPECTED_SEEDS = 2


@dataclass(frozen=True)
class Recipe:
  label: str
  run_dir: str
  slurm_logs: tuple[str, str]
  header_tokens: tuple[tuple[str, ...], tuple[str, ...]]
  color: str


RECIPES = (
    Recipe(
        label='CRL (10 updates, n=2)',
        run_dir=(
            'final_runs/other_density_estimators/'
            'ppo_builderbench_creative5_task2_e1024_pd_crl_tau05_'
            'actorreset_evalvid_catselect_extrew1_ep70_200m_crl10_'
            'eval10_warp_2h30'
        ),
        slurm_logs=(
            'bb_c5_ep70_crl10_3811587_0.log',
            'bb_c5_ep70_crl10_3811587_1.log',
        ),
        header_tokens=(
            ('array=0 task=creative5_task2 method=crl seed=0',
             'crl_steps=10', 'extrew=true'),
            ('array=1 task=creative5_task2 method=crl seed=1',
             'crl_steps=10', 'extrew=true'),
        ),
        color='#E8834C',
    ),
    Recipe(
        label='TD3 + ext. reward + NormObs (25 updates, n=2)',
        run_dir=(
            'final_runs/other_density_estimators/'
            'ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_'
            'actorreset_nopermute_normobs_evalvid_catselect_extrew1_'
            'ep70_200m_crl25_eval10_warp_4h'
        ),
        slurm_logs=(
            'bb_c5t2_td3_ablate_3811711_0.log',
            'bb_c5t2_td3_ablate_3811711_1.log',
        ),
        header_tokens=(
            ('array=0 config=0 task=creative5_task2 method=td3 seed=2',
             'crl_steps=25', 'norm_obs=true', 'external_reward_scale=1'),
            ('array=1 config=0 task=creative5_task2 method=td3 seed=3',
             'crl_steps=25', 'norm_obs=true', 'external_reward_scale=1'),
        ),
        color='#4C9BE8',
    ),
    Recipe(
        label='TD-InfoNCE (25 updates, n=2)',
        run_dir=(
            'final_runs/other_density_estimators/'
            'ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_'
            'catselect_extrew1_ep70_200m_crl25_eval10_warp_4h'
        ),
        slurm_logs=(
            'bb_c5_ep70_crl25_3811588_4.log',
            'bb_c5_ep70_crl25_3811588_5.log',
        ),
        header_tokens=(
            ('array=4 task=creative5_task2 method=tdinfonce seed=0',
             'crl_steps=25', 'extrew=true'),
            ('array=5 task=creative5_task2 method=tdinfonce seed=1',
             'crl_steps=25', 'extrew=true'),
        ),
        color='#E84C6F',
    ),
)


def _load_recipe(recipe: Recipe):
  train_series = []
  eval_series = []
  for filename, tokens in zip(recipe.slurm_logs, recipe.header_tokens):
    path = os.path.join(SLURM_ROOT, filename)
    if not os.path.isfile(path):
      raise FileNotFoundError(f'missing retained Slurm log: {path}')
    with open(path, 'r', errors='replace') as f:
      header = ''.join(f.readline() for _ in range(8))
    missing = [token for token in tokens if token not in header]
    if missing:
      raise RuntimeError(f'{path}: header missing expected tokens {missing}')

    train = base._parse_train_slurm_path(path)
    eval_iters = base._parse_eval_slurm_path(path)
    if not train or train[-1][0] < MIN_FINAL_STEP:
      final = train[-1][0] if train else 'missing'
      raise RuntimeError(
          f'{path}: incomplete train log; final step={final}, '
          f'required >= {MIN_FINAL_STEP}')
    if not eval_iters:
      raise RuntimeError(f'{path}: no eval success records')
    train_series.append(train)
    eval_series.append(
        base._iters_to_env_steps(eval_iters, LOG_ROOT, recipe.run_dir))
    print(
        f'validated {filename}: final_train_step={train[-1][0]:,}, '
        f'final_eval_step={eval_series[-1][-1][0]:,}')

  if len(train_series) != EXPECTED_SEEDS or len(eval_series) != EXPECTED_SEEDS:
    raise RuntimeError(f'{recipe.label}: expected exactly two retained seeds')
  return train_series, eval_series


def _plot_sem(ax, xs, mean, sem, color, *, smooth: bool) -> None:
  lower = [max(0.0, m - s) for m, s in zip(mean, sem)]
  upper = [min(1.0, m + s) for m, s in zip(mean, sem)]
  if smooth:
    lower = base._rolling_mean(lower, base.EVAL_SMOOTH_WINDOW)
    upper = base._rolling_mean(upper, base.EVAL_SMOOTH_WINDOW)
  ax.fill_between(xs, lower, upper, color=color, alpha=0.17, linewidth=0)


def main() -> None:
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
  summaries = []

  for recipe in RECIPES:
    train_series, eval_series = _load_recipe(recipe)
    tx, tm, tse, tn = base._aggregate_mean_stderr(train_series)
    ex, em, ese, en = base._aggregate_mean_stderr(eval_series)
    if tn != EXPECTED_SEEDS or en != EXPECTED_SEEDS:
      raise RuntimeError(
          f'{recipe.label}: aggregate seed counts train={tn}, eval={en}')

    axes[0].plot(tx, tm, color=recipe.color, linewidth=2.25,
                 label=recipe.label)
    _plot_sem(axes[0], tx, tm, tse, recipe.color, smooth=False)
    eval_smoothed = base._plot_eval_smoothed(
        axes[1], ex, em, color=recipe.color, label=recipe.label,
        linewidth=2.25, window=base.EVAL_SMOOTH_WINDOW)
    _plot_sem(axes[1], ex, em, ese, recipe.color, smooth=True)

    summary = (
        f'{recipe.label}: final/peak train={tm[-1]:.3f}/{max(tm):.3f}; '
        f'final/peak smoothed eval='
        f'{eval_smoothed[-1]:.3f}/{max(eval_smoothed):.3f}'
    )
    summaries.append(summary)

  panel_titles = (
      'Train success (last 1000 episodes; unsmoothed)',
      f'Eval success (faint raw mean + centered rolling mean, '
      f'w={base.EVAL_SMOOTH_WINDOW})',
  )
  for ax, title in zip(axes, panel_titles):
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=10)
    ax.set_xlabel('environment steps')
    ax.set_xlim(left=0)
    ax.set_ylim(-0.035, 1.035)
    ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
  axes[0].set_ylabel('success (mean ± SEM)')

  handles, labels = axes[1].get_legend_handles_labels()
  legend = axes[1].legend(
      handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0),
      fontsize=8.7, framealpha=1.0, edgecolor='#333333', fancybox=False,
      handlelength=2.6,
  )
  suptitle = fig.suptitle(
      'BuilderBench C5T2 ep70 — selected density-estimator recipes',
      fontsize=13, fontweight='bold', y=0.98)
  subtitle = fig.text(
      0.5, 0.91,
      'Selected best recipes (configs differ) · seeds: CRL/TD-InfoNCE 0–1; '
      'TD3 2–3',
      ha='center', fontsize=9.5)
  fig.subplots_adjust(left=0.07, right=0.79, top=0.80, bottom=0.14, wspace=0.20)
  fig.savefig(
      OUT_PATH, dpi=170, bbox_inches='tight',
      bbox_extra_artists=(legend, suptitle, subtitle),
      pad_inches=0.18, facecolor='white')
  plt.close(fig)

  print(f'wrote {OUT_PATH}')
  print('seed IDs: CRL=0/1; TD3=2/3; TD-InfoNCE=0/1')
  for summary in summaries:
    print(f'  {summary}')


if __name__ == '__main__':
  main()
