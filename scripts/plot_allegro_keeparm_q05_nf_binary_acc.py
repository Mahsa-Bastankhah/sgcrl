#!/usr/bin/env python3
"""Hide-table NF P5 seeds 0–2: eval NF binary accuracy (policy vs random).

``nf_binary_accuracy`` is P[log p(g+|s,a) > log p(g-|s,a)] on eval π.
``nf_binary_accuracy_random`` is the same after Uniform[-1,1] actions.
There is no held-out NF train/val split in the learner logs.

  python scripts/plot_allegro_keeparm_q05_nf_binary_acc.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_keeparm_q05_nf_binary_acc.png')
OUT_PATH_T70 = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_keeparm_q05_ep70_nf_binary_acc.png')
OUT_PATH_BOTH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_keeparm_q05_ep50_ep70_nf_binary_acc.png')
EVAL_WINDOW = 21
EVAL_WINDOW_T70 = 5
EVAL_WINDOW_BOTH = 5

RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed0_14h',
        'T=50 seed 0',
        base.ACCENT_COLORS[4],
        1024 * 50,
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed1_14h',
        'T=50 seed 1',
        base.ACCENT_COLORS[0],
        1024 * 50,
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed2_14h',
        'T=50 seed 2',
        base.ACCENT_COLORS[2],
        1024 * 50,
        '-',
    ),
)

# Running hide-table Q5 T=70 jobs (1B).
T70_RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep70_1000m_q05shift_seed0_15h',
        'T=70 seed 0  (replace=10)',
        base.ACCENT_COLORS[3],
        1024 * 70,
        '--',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep70_1b_q05shift_noreplace_seed0_14h',
        'T=70 seed 0  noreplace',
        base.ACCENT_COLORS[1],
        1024 * 70,
        '--',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_hidetable_bucket_xy0022_norand_ep70_1b_q05shift_noreplace_seed1_14h',
        'T=70 seed 1  noreplace',
        base.ACCENT_COLORS[5],
        1024 * 70,
        '--',
    ),
)


def _read_eval(log_dir: str, steps_per_iter: int = 1024 * 50):
  root = os.path.join(base.LOG_ROOT, log_dir)
  path = None
  if os.path.isdir(root):
    for name in sorted(os.listdir(root)):
      cand = os.path.join(root, name, 'logs', 'eval', 'logs.csv')
      if os.path.isfile(cand):
        path = cand
        break
  if path is None:
    return [], [], []
  xs, pol, rnd = [], [], []
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      it = base._coerce(row.get('iteration', ''))
      yp = base._coerce(row.get('nf_binary_accuracy', ''))
      yr = base._coerce(row.get('nf_binary_accuracy_random', ''))
      if it is None or yp is None or yp != yp:
        continue
      xs.append(int(it) * int(steps_per_iter))
      pol.append(float(yp))
      rnd.append(float(yr) if yr is not None and yr == yr else float('nan'))
  return xs, pol, rnd


def _style(ax, ylabel: str, *, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=10)
  ax.axhline(0.5, color='0.45', lw=1.0, ls=':', zorder=1)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.set_ylim(0.25, 1.02)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))


def _plot_group(runs, out_path: str, *, window: int, title_suffix: str) -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.0), sharex=True)
  for item in runs:
    log_dir, label, color = item[0], item[1], item[2]
    spi = int(item[3]) if len(item) > 3 else 1024 * 50
    ls = item[4] if len(item) > 4 else '-'
    xs, pol, rnd = _read_eval(log_dir, spi)
    if not xs:
      print(f'{label}: no eval nf_binary_accuracy')
      continue
    base._plot_eval_smoothed(
        axes[0], xs, pol, color=color, linestyle=ls,
        label=f'{label}  (roll mean w={window})',
        window=window)
    if any(y == y for y in rnd):
      base._plot_eval_smoothed(
          axes[1], xs, rnd, color=color, linestyle=ls,
          label=f'{label}  (roll mean w={window})',
          window=window)
    print(f'{label} policy last={pol[-1]:.3f} peak={max(pol):.3f}  '
          f'random last={rnd[-1]:.3f} peak={max(rnd):.3f}  '
          f'steps={xs[-1]/1e6:.1f}M n={len(xs)}')

  axes[0].set_title(
      f'NF binary acc  on eval π  {title_suffix}',
      fontsize=12, fontweight='bold')
  _style(axes[0], f'policy  (roll mean, w={window})')
  axes[0].legend(loc='lower right', fontsize=8.5, framealpha=0.95)
  axes[1].set_title(
      'NF binary acc  on Uniform[-1,1] actions  (same eval env, fresh reset)',
      fontsize=12, fontweight='bold')
  _style(axes[1], f'random  (roll mean, w={window})', xlabel=True)
  axes[1].legend(loc='lower right', fontsize=8.5, framealpha=0.95)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}')


def main() -> None:
  _plot_group(
      RUNS, OUT_PATH, window=EVAL_WINDOW,
      title_suffix='(P[log p(g+|s,a) > log p(g-|s,a)])  T=50 500M')
  _plot_group(
      T70_RUNS, OUT_PATH_T70, window=EVAL_WINDOW_T70,
      title_suffix='(P[log p(g+|s,a) > log p(g-|s,a)])  running T=70 1B')
  _plot_group(
      RUNS + T70_RUNS, OUT_PATH_BOTH, window=EVAL_WINDOW_BOTH,
      title_suffix='(P[log p(g+|s,a) > log p(g-|s,a)])  T=50 500M + running T=70')


if __name__ == '__main__':
  main()
