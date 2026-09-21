#!/usr/bin/env python3
"""Plot pooled C4T2 density-estimator score diagnostics.

For each estimator, this pools the 10- and 25-update variants and seeds 0/1,
giving four trajectories. Bands are SEM across that pooled set.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import plot_builderbench_train_success1000 as base  # noqa: E402

LOG_ROOT = os.path.join(REPO, 'logs')
RUN_ROOT = 'final_runs/other_density_estimators'
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'final_runs',
    'other_density_estimators',
    'creative4_task2_logp_diagnostics.png')
OUT_DIR = os.path.dirname(OUT_PATH)
MIN_FINAL_STEP = 190_000_000
ROLLING_WINDOW = 21
POST_SUCCESS_STEPS = 20_000_000
EXPECTED_TRAJECTORIES = 4
FINAL_EVAL_METRIC = base.EVAL_METRIC
FINAL_TRAIN_METRIC = base.TRAIN_METRIC
NF_RUN = os.path.join(
    LOG_ROOT,
    'final_runs',
    'visualizations',
    'ppo_builderbench_creative4_task2_e1024_pd_nf_compact_small_'
    'sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_'
    'minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_'
    'valuedgr_c100_lamlr1e6_warp_logp_s0_4h',
    'ppo_builderbench_creative_4_task2_0',
    'logs',
    'learner',
    'logs.csv',
)

ESTIMATORS = (
    {
        'label': 'CRL',
        'prefix': 'crl',
        'color': '#E8834C',
        'runs': (
            'ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_'
            'catselect_stateonly_extrew1_ep50_200m_crl10_eval10_warp_2h30',
            'ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_'
            'catselect_stateonly_extrew1_ep50_200m_crl25_eval10_warp_4h',
        ),
    },
    {
        'label': 'TD3',
        'prefix': 'td3',
        'color': '#4CE87A',
        'runs': (
            'ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_'
            'catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30',
            'ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_'
            'catselect_extrew1_ep50_200m_crl25_eval10_warp_4h',
        ),
    },
    {
        'label': 'TD-InfoNCE',
        'prefix': 'tdinfonce',
        'color': '#E84C6F',
        'runs': (
            'ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_'
            'catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30',
            'ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_'
            'catselect_extrew1_ep50_200m_crl25_eval10_warp_4h',
        ),
    },
)

PANELS = (
    ('replay_pos_logp_p10', 'Replay positive p10\n(lower percentile; 90% above)'),
    ('replay_pos_logp_mean', 'Replay positive mean'),
    ('replay_pos_logp_p90', 'Replay positive p90\n(upper percentile; 90% below)'),
    (
        'reward_repr_raw_mean',
        'PPO task-goal score mean\n(pre-normalization; evaluated at task goal g)',
    ),
)

METRIC_STYLES = (
    ('replay_pos_logp_p10', 'Replay positive p10 (90% above)', '#0072B2', '-'),
    ('replay_pos_logp_mean', 'Replay positive mean', '#009E73', '--'),
    ('replay_pos_logp_p90', 'Replay positive p90 (90% below)', '#E69F00', '-.'),
    (
        'reward_repr_raw_mean',
        'PPO task-goal mean (pre-normalization)',
        '#CC79A7',
        ':',
    ),
)


def _finite(value: float | None) -> bool:
  return value is not None and math.isfinite(value)


def _metric_key(prefix: str, metric: str) -> str:
  if metric == 'reward_repr_raw_mean':
    return metric
  return f'{prefix}/{metric}'


def _read_csv(path: str, metric: str) -> list[tuple[int, float]]:
  if not os.path.isfile(path):
    raise FileNotFoundError(f'missing learner CSV: {path}')
  values_by_step: dict[int, float] = {}
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    fields = reader.fieldnames or []
    for required in ('global_step', metric):
      if required not in fields:
        raise KeyError(f'{path}: missing required column {required!r}')
    for row in reader:
      x = base._coerce(row.get('global_step'))
      y = base._coerce(row.get(metric))
      if _finite(x) and _finite(y):
        values_by_step[int(x)] = float(y)
  points = sorted(values_by_step.items())
  if not points:
    raise RuntimeError(f'{path}: no finite values for {metric}')
  if points[-1][0] < MIN_FINAL_STEP:
    raise RuntimeError(
        f'{path}: incomplete {metric} horizon ({points[-1][0]:,} steps)')
  return points


def _trajectory_paths(run: str) -> list[str]:
  run_dir = os.path.join(LOG_ROOT, RUN_ROOT, run)
  if not os.path.isdir(run_dir):
    raise FileNotFoundError(f'missing run directory: {run_dir}')
  paths = []
  for child in sorted(os.listdir(run_dir)):
    path = os.path.join(run_dir, child, 'logs', 'learner', 'logs.csv')
    if os.path.isfile(path):
      paths.append(path)
  if len(paths) != 2:
    raise RuntimeError(
        f'{run_dir}: expected exactly two seed learner CSVs, found {len(paths)}')
  return paths


def _first_success(path: str) -> tuple[str, int, int]:
  """Return selected success metric, iteration, and step for one run."""
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    fields = reader.fieldnames or []
    rows = list(reader)
  for required in ('global_step', 'iteration', 'reward_env_mean'):
    if required not in fields:
      raise KeyError(f'{path}: missing first-success column {required!r}')

  success_col = 'reward_env_mean'
  if 'train_success_mean' in fields:
    train_success = [
        base._coerce(row.get('train_success_mean')) for row in rows
    ]
    if any(_finite(value) and value > 0 for value in train_success):
      success_col = 'train_success_mean'

  for row in rows:
    value = base._coerce(row.get(success_col))
    iteration = base._coerce(row.get('iteration'))
    step = base._coerce(row.get('global_step'))
    if (
        _finite(value) and value > 0
        and _finite(iteration) and _finite(step)
    ):
      return success_col, int(iteration), int(step)
  raise RuntimeError(f'{path}: no finite positive {success_col} value')


def _seed_from_path(path: str) -> int:
  run_name = os.path.basename(
      os.path.dirname(os.path.dirname(os.path.dirname(path))))
  match = re.search(r'_(\d+)$', run_name)
  if not match:
    raise RuntimeError(f'cannot identify seed from learner CSV path: {path}')
  return int(match.group(1))


def _last_finite(path: str, metric: str) -> float:
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    fields = reader.fieldnames or []
    if metric not in fields:
      raise KeyError(f'{path}: missing final-score column {metric!r}')
    values = [base._coerce(row.get(metric)) for row in reader]
  finite = [value for value in values if _finite(value)]
  if not finite:
    raise RuntimeError(f'{path}: no finite values for {metric}')
  return float(finite[-1])


def _candidate(path: str, updates: int) -> dict[str, object]:
  eval_path = path.replace('/logs/learner/logs.csv', '/logs/eval/logs.csv')
  return {
      'path': path,
      'updates': updates,
      'seed': _seed_from_path(path),
      'final_eval': _last_finite(eval_path, FINAL_EVAL_METRIC),
      'final_train': _last_finite(path, FINAL_TRAIN_METRIC),
  }


def _select_method_run(spec: dict[str, object]) -> dict[str, object]:
  if spec['prefix'] == 'nf':
    selected = _candidate(NF_RUN, 10)
    reason = 'only available NF diagnostic run'
  else:
    candidates = []
    for run in spec['runs']:
      update_match = re.search(r'_crl(\d+)_', str(run))
      if not update_match:
        raise RuntimeError(f'cannot identify update count from run: {run}')
      updates = int(update_match.group(1))
      candidates.extend(
          _candidate(path, updates) for path in _trajectory_paths(str(run)))
    if len(candidates) != EXPECTED_TRAJECTORIES:
      raise RuntimeError(
          f'{spec["label"]}: expected {EXPECTED_TRAJECTORIES} selection '
          f'candidates, found {len(candidates)}')
    selected = max(
        candidates,
        key=lambda item: (item['final_eval'], item['final_train']))
    reason = (
        f'highest final {FINAL_EVAL_METRIC}, tie-broken by final '
        f'{FINAL_TRAIN_METRIC}')
    for item in candidates:
      print(
          f'{spec["label"]} candidate u{item["updates"]} s{item["seed"]}: '
          f'final eval={item["final_eval"]:.6g}; '
          f'final train={item["final_train"]:.6g}')
  print(
      f'{spec["label"]} selected u{selected["updates"]} '
      f's{selected["seed"]}: final eval={selected["final_eval"]:.6g}; '
      f'final train={selected["final_train"]:.6g}; reason={reason}')
  return selected


def _interp(points: list[tuple[int, float]], x: int) -> float:
  """Interpolate within one trajectory; callers guarantee no extrapolation."""
  lo, hi = 0, len(points) - 1
  while lo <= hi:
    mid = (lo + hi) // 2
    if points[mid][0] < x:
      lo = mid + 1
    elif points[mid][0] > x:
      hi = mid - 1
    else:
      return points[mid][1]
  left, right = points[hi], points[lo]
  weight = (x - left[0]) / (right[0] - left[0])
  return left[1] + weight * (right[1] - left[1])


def _aggregate_complete(
    trajectories: list[list[tuple[int, float]]],
) -> tuple[list[int], list[float], list[float]]:
  """Mean and SEM on the union grid where every trajectory has support."""
  start = max(points[0][0] for points in trajectories)
  stop = min(points[-1][0] for points in trajectories)
  grid = sorted({
      x for points in trajectories for x, _ in points if start <= x <= stop
  })
  means, sems = [], []
  for x in grid:
    values = [_interp(points, x) for points in trajectories]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (
        len(values) - 1)
    means.append(mean)
    sems.append(math.sqrt(variance / len(values)))
  return grid, means, sems


def _load_pooled(spec: dict[str, object], metric: str):
  key = _metric_key(str(spec['prefix']), metric)
  trajectories = []
  for run in spec['runs']:
    trajectories.extend(_read_csv(path, key) for path in _trajectory_paths(run))
  if len(trajectories) != EXPECTED_TRAJECTORIES:
    raise RuntimeError(
        f'{spec["label"]} {key}: expected {EXPECTED_TRAJECTORIES} pooled '
        f'trajectories, found {len(trajectories)}')
  xs, mean, sem = _aggregate_complete(trajectories)
  if not xs or xs[-1] < MIN_FINAL_STEP:
    raise RuntimeError(
        f'{spec["label"]} {key}: pooled horizon is '
        f'{xs[-1] if xs else "empty"}')
  return xs, mean, sem


def _plot_panel(ax, metric: str, title: str) -> None:
  for spec in ESTIMATORS:
    xs, mean, sem = _load_pooled(spec, metric)
    smooth_mean = base._rolling_mean(mean, ROLLING_WINDOW)
    color = str(spec['color'])
    ax.plot(xs, mean, color=color, linewidth=0.8, alpha=0.24, zorder=2)
    ax.fill_between(
        xs,
        [m - s for m, s in zip(mean, sem)],
        [m + s for m, s in zip(mean, sem)],
        color=color, alpha=0.12,
        linewidth=0, zorder=1)
    ax.plot(
        xs, smooth_mean, color=color, linewidth=2.2,
        label=f'{spec["label"]} — pooled 10/25 variants (n=4 runs)',
        solid_capstyle='round', zorder=3)
    print(
        f'{spec["label"]} {metric}: pooled n=4; '
        f'range=[{min(mean):.5g}, {max(mean):.5g}]; '
        f'final raw={mean[-1]:.5g}; final MA={smooth_mean[-1]:.5g}; '
        f'max_step={xs[-1]:,}')

  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_xlabel('Environment steps')
  ax.set_ylabel('Raw estimator score')
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.spines[['top', 'right']].set_visible(False)


def _method_series(
    spec: dict[str, object],
    metric: str,
    path: str,
) -> tuple[list[int], list[float]]:
  key = _metric_key(str(spec['prefix']), metric)
  points = _read_csv(path, key)
  return [x for x, _ in points], [y for _, y in points]


def _plot_method_diagnostics(spec: dict[str, object]) -> str:
  is_nf = spec['prefix'] == 'nf'
  selected = _select_method_run(spec)
  selected_path = str(selected['path'])
  success_column, iteration, success_step = _first_success(selected_path)
  window_end = success_step + POST_SUCCESS_STEPS
  fig, ax = plt.subplots(figsize=(12.4, 6.8))
  series_data = []
  series_means = {}

  for metric, label, color, linestyle in METRIC_STYLES:
    xs, values = _method_series(spec, metric, selected_path)
    in_window = [(x, y) for x, y in zip(xs, values) if x <= window_end]
    if not in_window:
      raise RuntimeError(
          f'{spec["label"]} {metric}: no points through step {window_end:,}')
    window_xs = [x for x, _ in in_window]
    window_values = [y for _, y in in_window]
    series_data.append((
        metric, label, color, linestyle, window_xs, window_values))
    series_means[metric] = sum(window_values) / len(window_values)

  low_metric = min(series_means, key=series_means.get)
  high_metric = max(series_means, key=series_means.get)
  mu_low = series_means[low_metric]
  mu_high = series_means[high_metric]
  if math.isclose(mu_high, mu_low, rel_tol=1e-12, abs_tol=1e-12):
    raise RuntimeError(
        f'{spec["label"]}: in-window diagnostic means are indistinguishable '
        f'({mu_low:.12g} vs {mu_high:.12g})')
  scale = mu_high - mu_low

  for metric, label, color, linestyle, xs, values in series_data:
    normalized = [(value - mu_low) / scale for value in values]
    smooth_values = base._rolling_mean(normalized, ROLLING_WINDOW)
    ax.plot(
        xs, normalized, color=color, linestyle=linestyle,
        linewidth=0.8, alpha=0.24, zorder=2)
    ax.plot(
        xs, smooth_values, color=color, linestyle=linestyle,
        linewidth=2.4, label=label, solid_capstyle='round', zorder=3)
    print(
        f'{spec["label"]} {metric}: selected u{selected["updates"]} '
        f's{selected["seed"]}; window_mean={series_means[metric]:.6g}; '
        f'normalized_range=[{min(normalized):.5g}, '
        f'{max(normalized):.5g}]; window_last_step={xs[-1]:,}')

  print(
      f'{spec["label"]} normalization window: [0, {window_end:,}]; '
      f'low={low_metric} mean={mu_low:.6g} -> 0; '
      f'high={high_metric} mean={mu_high:.6g} -> 1')
  run_label = f'u{selected["updates"]} s{selected["seed"]}'
  ax.axvline(
      success_step, color='#555555', linestyle='--', linewidth=1.15,
      alpha=0.82, zorder=4)
  print(
      f'{spec["label"]} first success {run_label}: '
      f'column={success_column}; iteration={iteration}; '
      f'step={success_step:,}')
  ax.text(
      0.99, 0.98,
      f'Selected {run_label}\n'
      f'final eval={selected["final_eval"]:.3f}; '
      f'train={selected["final_train"]:.3f}\n'
      f'first success: i{iteration} / '
      f'{success_step / 1_000_000:.2f}M',
      transform=ax.transAxes, ha='right', va='top', fontsize=8.2,
      linespacing=1.25,
      bbox={
          'boxstyle': 'round,pad=0.35',
          'facecolor': 'white',
          'edgecolor': '#BBBBBB',
          'alpha': 0.90,
      },
      zorder=6)

  sample_text = 'single diagnostic run' if is_nf else 'best of four runs'
  ax.set_title(
      f'BuilderBench C4T2 — {spec["label"]} log-p diagnostics\n'
      f'{sample_text}: {run_label}; '
      f'final eval={selected["final_eval"]:.3f}, '
      f'train={selected["final_train"]:.3f}; n=1, no SEM',
      fontsize=13, fontweight='bold', pad=12)
  ax.set_xlabel('Environment steps')
  ax.set_ylabel('Relative diagnostic score (shared affine scale)')
  ax.set_xlim(0, window_end)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.margins(y=0.08)
  ax.spines[['top', 'right']].set_visible(False)
  ax.legend(
      loc='best', framealpha=0.94, edgecolor='#BBBBBB',
      handlelength=3.2)
  fig.text(
      0.5, 0.018,
      f'Bold: centered rolling mean (window={ROLLING_WINDOW} iterations); '
      f'faint: unsmoothed selected run; no uncertainty band. '
      f'Relative scale: in-window mean of {low_metric} → 0, '
      f'{high_metric} → 1 (values not clipped). '
      f'Window: 0–{window_end:,} steps; first success: '
      f'{success_column} > 0.',
      ha='center', va='bottom', fontsize=9)
  fig.subplots_adjust(left=0.085, right=0.98, top=0.85, bottom=0.14)

  max_plotted_step = max(
      float(x)
      for line in ax.lines
      for x in line.get_xdata()
  )
  if max_plotted_step > window_end:
    raise RuntimeError(
        f'{spec["label"]}: plotted step {max_plotted_step:g} exceeds '
        f'window end {window_end:,}')
  # Lock the requested range after every artist has been added so a later
  # autoscale cannot restore the full 200M-step trajectory extent.
  ax.set_xlim(0.0, float(window_end))
  ax.set_autoscalex_on(False)
  actual_xlim = ax.get_xlim()
  if not (
      math.isclose(actual_xlim[0], 0.0)
      and math.isclose(actual_xlim[1], float(window_end))
  ):
    raise RuntimeError(
        f'{spec["label"]}: requested xlim=(0, {window_end}) but got '
        f'{actual_xlim}')
  print(
      f'{spec["label"]} plotted_xmax={max_plotted_step:,.0f}; '
      f'axis_xlim=({actual_xlim[0]:,.0f}, {actual_xlim[1]:,.0f})')

  out = os.path.join(
      OUT_DIR,
      f'creative4_task2_{spec["prefix"]}_logp_diagnostics.png')
  tmp = out + '.tmp.png'
  fig.savefig(tmp, dpi=180, bbox_inches='tight', facecolor='white')
  os.replace(tmp, out)
  plt.close(fig)
  print(f'wrote {out}')
  return out


def _plot_combined() -> None:
  fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), sharex=True)
  for ax, (metric, title) in zip(axes.flat, PANELS):
    _plot_panel(ax, metric, title)

  handles, labels = axes.flat[0].get_legend_handles_labels()
  fig.legend(
      handles, labels, loc='upper center', ncol=3, frameon=False,
      bbox_to_anchor=(0.5, 0.925), handlelength=2.5, columnspacing=1.4)
  fig.suptitle(
      'BuilderBench C4T2 density-estimator diagnostics\n'
      'Pooled 10/25-update variants, n=4 runs per estimator',
      fontsize=14, fontweight='bold', y=0.985)
  fig.text(
      0.5, 0.012,
      f'Bold: centered rolling mean (window={ROLLING_WINDOW} iterations); '
      'faint: unsmoothed pooled mean; shading: unsmoothed SEM across four '
      'runs. Curves use only the common supported step range (no '
      'extrapolation).',
      ha='center', va='bottom', fontsize=9)
  fig.subplots_adjust(
      left=0.075, right=0.98, top=0.83, bottom=0.085,
      hspace=0.32, wspace=0.20)

  os.makedirs(OUT_DIR, exist_ok=True)
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=180, bbox_inches='tight', facecolor='white')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'wrote {OUT_PATH}')


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Plot C4T2 density-estimator score diagnostics.')
  parser.add_argument(
      '--combined', action='store_true',
      help='Also regenerate the pooled four-panel multi-method figure.')
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  plt.rcParams.update({
      'font.size': 10,
      'axes.labelsize': 10,
      'legend.fontsize': 9,
      'axes.linewidth': 1.0,
  })
  os.makedirs(OUT_DIR, exist_ok=True)
  if args.combined:
    _plot_combined()
  _plot_method_diagnostics({
      'label': 'NF',
      'prefix': 'nf',
  })
  for spec in ESTIMATORS:
    _plot_method_diagnostics(spec)


if __name__ == '__main__':
  main()
