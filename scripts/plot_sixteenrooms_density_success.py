#!/usr/bin/env python3
"""Compare eval success for five SixteenRooms density estimators.

Each method has one seed. Eval success is shown as faint raw checkpoints plus
a bold centered rolling mean with window five; no uncertainty is estimated.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


LOG_ROOT = (
    '/n/fs/mislresearch/sgcrl/logs/ppo_sixteenrooms_density_diag_1m'
)
OUTPUT = (
    '/n/fs/mislresearch/sgcrl/figs/maze/'
    'sixteenrooms_density_estimators_eval_success.png'
)
MIN_FINAL_STEP = 980_000
SMOOTH_WINDOW = 5

METHODS = (
    ('NF compact-small', 'nf_compact_small', '#3B82C4'),
    ('NF tiny', 'nf_tiny', '#8B5FBF'),
    ('CRL', 'crl', '#E07A35'),
    ('TD3 log-Q', 'td3_logq', '#2E9B62'),
    ('TD-InfoNCE', 'tdinfonce', '#D64D68'),
)


def _finite_float(value: str | None, *, field: str, path: str) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f'{path}: invalid {field}={value!r}') from exc
  if not math.isfinite(result):
    raise ValueError(f'{path}: non-finite {field}={value!r}')
  return result


def _read_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
  if not os.path.isfile(path):
    raise FileNotFoundError(path)
  with open(path, newline='') as handle:
    reader = csv.DictReader(handle)
    fields = list(reader.fieldnames or ())
    rows = list(reader)
  if not fields or not rows:
    raise ValueError(f'{path}: empty CSV')
  return fields, rows


def _learner_step_map(run_dir: str) -> tuple[dict[int, int], int]:
  path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
  fields, rows = _read_rows(path)
  required = {'iteration', 'global_step'}
  if not required.issubset(fields):
    raise ValueError(f'{path}: expected columns {sorted(required)}')

  mapping: dict[int, int] = {}
  final_step = 0
  for row in rows:
    iteration = int(_finite_float(
        row.get('iteration'), field='iteration', path=path))
    global_step = int(_finite_float(
        row.get('global_step'), field='global_step', path=path))
    mapping[iteration] = global_step
    final_step = max(final_step, global_step)
  return mapping, final_step


def _load_eval(run_dir: str) -> tuple[list[int], list[float], str, int]:
  path = os.path.join(run_dir, 'logs', 'eval', 'logs.csv')
  fields, rows = _read_rows(path)
  if 'success' not in fields:
    raise ValueError(f'{path}: missing success column')

  learner_map, learner_final_step = _learner_step_map(run_dir)
  has_global_step = 'global_step' in fields
  if not has_global_step and 'iteration' not in fields:
    raise ValueError(f'{path}: needs global_step or iteration column')

  by_step: dict[int, float] = {}
  for row in rows:
    success = _finite_float(
        row.get('success'), field='success', path=path)
    if not 0.0 <= success <= 1.0:
      raise ValueError(f'{path}: success outside [0, 1]: {success}')
    if has_global_step:
      step = int(_finite_float(
          row.get('global_step'), field='global_step', path=path))
    else:
      iteration = int(_finite_float(
          row.get('iteration'), field='iteration', path=path))
      if iteration not in learner_map:
        raise ValueError(
            f'{path}: eval iteration {iteration} has no learner global_step')
      step = learner_map[iteration]
    by_step[step] = success

  points = sorted(by_step.items())
  xs = [step for step, _ in points]
  ys = [success for _, success in points]
  if len(xs) < SMOOTH_WINDOW:
    raise ValueError(
        f'{path}: only {len(xs)} eval points; need {SMOOTH_WINDOW}')
  if xs[-1] < MIN_FINAL_STEP:
    raise ValueError(
        f'{path}: incomplete eval horizon {xs[-1]:,} < '
        f'{MIN_FINAL_STEP:,}')
  if xs[-1] > learner_final_step:
    raise ValueError(
        f'{path}: eval horizon {xs[-1]:,} exceeds learner horizon '
        f'{learner_final_step:,}')

  schema = (
      'eval global_step'
      if has_global_step
      else 'eval iteration mapped through learner iteration/global_step'
  )
  return xs, ys, schema, learner_final_step


def plot(log_root: str, output: str) -> None:
  loaded = []
  for label, subdir, color in METHODS:
    run_dir = os.path.join(
        log_root, subdir, 'ppo_point_SixteenRooms_0')
    xs, ys, schema, learner_final = _load_eval(run_dir)
    loaded.append((label, color, xs, ys, schema, learner_final))

  horizons = {(len(xs), xs[0], xs[-1]) for _, _, xs, _, _, _ in loaded}
  if len(horizons) != 1:
    details = ', '.join(
        f'{label}: n={len(xs)}, [{xs[0]:,}..{xs[-1]:,}]'
        for label, _, xs, _, _, _ in loaded)
    raise ValueError(f'eval horizons do not match: {details}')

  fig, ax = plt.subplots(figsize=(10.8, 6.1))
  for label, color, xs, ys, schema, learner_final in loaded:
    smoothed = base._plot_eval_smoothed(
        ax,
        xs,
        ys,
        color=color,
        label=f'{label} (smoothed, window=5)',
        linewidth=2.35,
        window=SMOOTH_WINDOW,
    )
    print(
        f'{label}: eval_points={len(xs)}, eval_steps={xs[0]}..{xs[-1]}, '
        f'learner_final_step={learner_final}, '
        f'final_raw={ys[-1]:.6f}, final_smoothed={smoothed[-1]:.6f}, '
        f'schema={schema}'
    )

  final_horizon = loaded[0][2][-1]
  ax.set_xlim(0, final_horizon)
  ax.set_ylim(0, 1)
  ax.set_xlabel('Environment steps', fontsize=11)
  ax.set_ylabel(
      'Eval success (centered rolling mean, window=5)', fontsize=11)
  ax.set_title(
      'SixteenRooms — density-estimator eval success\n'
      'Centered rolling mean (window=5); faint markers are raw eval '
      'checkpoints · n=1 seed per method',
      fontsize=13,
      fontweight='bold',
      pad=12,
  )
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
  ax.grid(axis='both', linestyle='--', linewidth=0.7, alpha=0.35)
  ax.spines[['top', 'right']].set_visible(False)
  ax.legend(
      loc='lower right',
      fontsize=9.5,
      framealpha=0.95,
      edgecolor='#555555',
      handlelength=2.8,
  )
  fig.tight_layout()
  os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
  fig.savefig(output, dpi=180, facecolor='white', bbox_inches='tight')
  plt.close(fig)
  print(f'wrote {output}')


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Plot five-method SixteenRooms eval success.')
  parser.add_argument('--log-root', default=LOG_ROOT)
  parser.add_argument('--output', default=OUTPUT)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  plot(args.log_root, args.output)


if __name__ == '__main__':
  main()
