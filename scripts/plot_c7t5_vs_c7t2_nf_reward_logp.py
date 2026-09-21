#!/usr/bin/env python3
"""Plot rollout NF reward and replay log-p range for c7t5 and c7t2.

Writes one four-line plot per task. Both figures use identical axis limits so
they can be compared directly. The c7t5 CSV may be read while training.
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(REPO, 'logs', 'final_runs')
OUT_DIR = os.path.join(
    REPO, 'figs', 'builderbench', 'c7t5_vs_c7t2_nf_reward_logp')
MAX_STEPS = 300_000_000

RECIPE = (
    'e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_'
    'nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_ep70_'
    '300m_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6_'
    'warp_4h'
)

RUNS = (
    {
        'task': 'c7t5',
        'title': 'creative-7 task5 · random planar spread · seed 0',
        'csv': os.path.join(
            LOG_ROOT,
            f'ppo_builderbench_creative7_task5_{RECIPE}',
            'ppo_builderbench_creative_7_task5_0',
            'logs', 'learner', 'logs.csv'),
        'eval_csv': os.path.join(
            LOG_ROOT,
            f'ppo_builderbench_creative7_task5_{RECIPE}',
            'ppo_builderbench_creative_7_task5_0',
            'logs', 'eval', 'logs.csv'),
    },
    {
        'task': 'c7t2',
        'title': 'creative-7 task2 · zig-zag tower easy · seed 0',
        'csv': os.path.join(
            LOG_ROOT,
            f'ppo_builderbench_creative7_task2_{RECIPE}',
            'ppo_builderbench_creative_7_task2_0',
            'logs', 'learner', 'logs.csv'),
        'eval_csv': os.path.join(
            LOG_ROOT,
            f'ppo_builderbench_creative7_task2_{RECIPE}',
            'ppo_builderbench_creative_7_task2_0',
            'logs', 'eval', 'logs.csv'),
    },
)

SERIES = (
    ('reward_repr_raw_mean',
     r'rollout PPO NF reward ($\tau\log p$, $\tau=0.5$)',
     '#D1495B', 1.8, '-'),
    ('nf/log_p_min', 'replay log p min', '#2A9D8F', 1.2, '-'),
    ('nf/log_p_mean', 'replay log p mean', '#264653', 1.8, '-'),
    ('nf/log_p_max', 'replay log p max', '#4C9BE8', 1.4, '--'),
)


def _load(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
  columns = ('global_step',) + tuple(item[0] for item in SERIES)
  values: dict[str, list[float]] = {name: [] for name in columns}
  with open(path, newline='', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      parsed: dict[str, float] = {}
      try:
        for name in columns:
          parsed[name] = float(row[name])
      except (KeyError, TypeError, ValueError):
        # The final row can be incomplete while the learner is writing it.
        continue
      if not all(np.isfinite(parsed[name]) for name in columns):
        continue
      for name in columns:
        values[name].append(parsed[name])

  x = np.asarray(values['global_step'], dtype=np.float64)
  return {
      name: (x, np.asarray(values[name], dtype=np.float64))
      for name, *_ in SERIES
  }


def _fmt_steps(value: float, _position=None) -> str:
  if abs(value) >= 1e6:
    return f'{value / 1e6:g}M'
  if abs(value) >= 1e3:
    return f'{value / 1e3:g}k'
  return f'{value:g}'


def _load_eval(path: str) -> tuple[np.ndarray, np.ndarray]:
  xs, ys = [], []
  with open(path, newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      try:
        iteration = float(row['iteration'])
        success = float(row['success'])
      except (KeyError, TypeError, ValueError):
        continue
      if np.isfinite(iteration) and np.isfinite(success):
        xs.append(iteration * 1024 * 70)
        ys.append(success)
  return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def main() -> None:
  loaded = []
  pooled_y = []
  for run in RUNS:
    if not os.path.isfile(run['csv']):
      raise FileNotFoundError(run['csv'])
    if not os.path.isfile(run['eval_csv']):
      raise FileNotFoundError(run['eval_csv'])
    data = _load(run['csv'])
    eval_data = _load_eval(run['eval_csv'])
    n = len(next(iter(data.values()))[0])
    if n == 0:
      raise RuntimeError(f'No complete learner rows in {run["csv"]}')
    last_step = next(iter(data.values()))[0][-1]
    print(f'{run["task"]}: rows={n}, last_step={last_step:.0f}')
    for metric, *_ in SERIES:
      pooled_y.append(data[metric][1])
    print(f'  eval checkpoints={len(eval_data[0])}')
    loaded.append((run, data, eval_data, last_step))

  finite_y = np.concatenate(pooled_y)
  y_lo = float(np.min(finite_y))
  y_hi = float(np.max(finite_y))
  pad = max(1.0, 0.04 * (y_hi - y_lo))
  shared_ylim = (y_lo - pad, y_hi + pad)

  os.makedirs(OUT_DIR, exist_ok=True)
  for run, data, eval_data, last_step in loaded:
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for metric, label, color, width, linestyle in SERIES:
      x, y = data[metric]
      ax.plot(
          x, y, label=label, color=color, linewidth=width,
          linestyle=linestyle, alpha=0.95)

    state = 'complete' if last_step >= 0.99 * MAX_STEPS else 'in progress'
    ax.set_title(
        f'{run["title"]}\n'
        f'raw learner traces · {state} at {_fmt_steps(last_step)}',
        fontsize=11, fontweight='bold')
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Reward / log probability')
    ax.set_xlim(0, MAX_STEPS)
    ax.set_ylim(*shared_ylim)
    ax.axhline(0.0, color='0.65', linewidth=0.7, zorder=0)
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))

    ax.legend(loc='best', framealpha=0.95)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'{run["task"]}_nf_reward_logp.png')
    tmp = out + '.tmp.png'
    fig.savefig(tmp, dpi=150, bbox_inches='tight')
    os.replace(tmp, out)
    plt.close(fig)
    print(f'wrote {out}')

  fig, ax = plt.subplots(figsize=(10.5, 5.4))
  eval_colors = {'c7t5': '#7B2CBF', 'c7t2': '#2A9D8F'}
  for run, _data, eval_data, _last_step in loaded:
    eval_x, eval_y = eval_data
    base._plot_eval_smoothed(
        ax, eval_x.tolist(), eval_y.tolist(),
        color=eval_colors[run['task']],
        label=(
            f'{run["task"]} eval success '
            f'(rolling mean, w={base.EVAL_SMOOTH_WINDOW})'),
        linestyle='-',
        linewidth=2.2)
  ax.set_title(
      'creative-7 task5 vs task2 · seed 0 eval success\n'
      f'faint raw checkpoints + centered rolling mean '
      f'(window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=11, fontweight='bold')
  ax.set_xlabel('Environment steps')
  ax.set_ylabel(
      f'Eval success (rolling mean, window={base.EVAL_SMOOTH_WINDOW})')
  ax.set_xlim(0, MAX_STEPS)
  ax.set_ylim(-0.05, 1.05)
  ax.grid(axis='both', linestyle='--', alpha=0.3)
  ax.legend(loc='best', framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  fig.tight_layout()
  out = os.path.join(OUT_DIR, 'c7t5_vs_c7t2_eval_success.png')
  tmp = out + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, out)
  plt.close(fig)
  print(f'wrote {out}')


if __name__ == '__main__':
  main()
