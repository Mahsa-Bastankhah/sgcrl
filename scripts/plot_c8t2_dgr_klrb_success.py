#!/usr/bin/env python3
"""c8t2 blue dgr + skip PPO after 4× train_success_1000 > 0.8.

Train: raw train_success_1000. Eval: faint raw + roll mean w=5.

  python scripts/plot_c8t2_dgr_klrb_success.py
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

LOG_ROOT = os.path.join(base.LOG_ROOT, 'final_runs')
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'figs', 'builderbench', 'final_runs',
    'c8t2_dgr_trainsucc08x4_skipppo_train_eval_success.png')
OUT_SEEDS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'figs', 'builderbench', 'final_runs',
    'c8t2_dgr_trainsucc08x4_skipppo_per_seed_success.png')
C = base.ACCENT_COLORS

NAME = (
    'ppo_builderbench_creative8_task2_e1024_pd'
    '_nf_compact_small_sa3x192_r64_b6_w192_tau05_'
    'nopermute_fixedx01_catwp_extrew1_minstd1e5_ent005_to001_'
    'ep100_300m_crl10_dualgradreg_c100_lamlr1e6_'
    'trainsucc08x4_skipppo_warp_4h'
)


def _shade(ax, xs, mean, se, *, color, n: int) -> None:
  if n <= 1 or not se:
    return
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)


def _style(ax, *, title: str, ylabel: str, xlabel: bool) -> None:
  ax.set_title(title, fontsize=12, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=10)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def _seed_label(run_name: str) -> str:
  if run_name.rsplit('_', 1)[-1].isdigit():
    return f'seed {run_name.rsplit("_", 1)[-1]}'
  return run_name


def _plot_per_seed() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  root = os.path.join(LOG_ROOT, NAME)
  try:
    runs = sorted(
        n for n in os.listdir(root)
        if os.path.isdir(os.path.join(root, n)))
  except OSError:
    runs = []
  fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.0), sharex=True)
  fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16, hspace=0.22)
  n_plotted = 0
  for i, run in enumerate(runs):
    color = C[i % len(C)]
    lab = _seed_label(run)
    t_path = os.path.join(root, run, 'logs', 'learner', 'logs.csv')
    e_path = os.path.join(root, run, 'logs', 'eval', 'logs.csv')
    txs, tys = [], []
    if os.path.isfile(t_path):
      with open(t_path, newline='') as f:
        for row in csv.DictReader(f):
          x = base._coerce(row.get(base.TRAIN_X_COL, ''))
          y = base._coerce(row.get(base.TRAIN_METRIC, ''))
          if x is not None and y is not None:
            txs.append(int(x))
            tys.append(y)
    exs, eys = [], []
    if os.path.isfile(e_path):
      it_map = {}
      if os.path.isfile(t_path):
        with open(t_path, newline='') as f:
          for row in csv.DictReader(f):
            it = base._coerce(row.get('iteration', ''))
            gs = base._coerce(row.get(base.TRAIN_X_COL, ''))
            if it is not None and gs is not None:
              it_map[int(it)] = int(gs)
      spi = base._infer_steps_per_iter(it_map) if it_map else 1024 * 100
      with open(e_path, newline='') as f:
        for row in csv.DictReader(f):
          it = base._coerce(row.get(base.EVAL_X_COL, ''))
          y = base._coerce(row.get(base.EVAL_METRIC, ''))
          if it is None or y is None:
            continue
          it = int(it)
          exs.append(it_map[it] if it in it_map else int((it + 1) * spi))
          eys.append(y)
    if not txs and not exs:
      print(f'  {lab}: no data')
      continue
    n_plotted += 1
    if txs:
      axes[0].plot(
          txs, tys, color=color, lw=1.7, alpha=0.95, label=lab, zorder=3)
      print(f'  {lab} train last={tys[-1]:.3f} peak={max(tys):.3f} '
            f'@ {txs[-1]/1e6:.1f}M')
    if exs:
      sm = base._plot_eval_smoothed(
          axes[1], exs, eys, color=color, label=lab, zorder=3, linewidth=2.0)
      if sm:
        print(f'  {lab} eval smooth last={sm[-1]:.3f} raw last={eys[-1]:.3f}')
  _style(
      axes[0],
      title='c8t2  ·  dgr + train-latch skip PPO  ·  train by seed',
      ylabel='Train Success (last 1000)', xlabel=False)
  _style(
      axes[1],
      title=f'c8t2  ·  dgr + train-latch skip PPO  ·  eval by seed (roll mean w={w})',
      ylabel=f'Eval Success (roll mean w={w})', xlabel=True)
  if n_plotted:
    axes[0].legend(
        loc='upper left', fontsize=8.5, framealpha=0.95,
        fancybox=False, edgecolor='#333333')
  fig.text(
      0.5, 0.015,
      'One line per seed. Train: raw train_success_1000. '
      f'Eval: faint raw + bold rolling mean, window={w}.',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  os.makedirs(os.path.dirname(OUT_SEEDS), exist_ok=True)
  tmp = OUT_SEEDS + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT_SEEDS)
  plt.close(fig)
  print(f'→ {OUT_SEEDS}')


def main() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.0), sharex=True)
  fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16, hspace=0.22)
  label = 'dgr + skip PPO after 4× train>0.8'
  color, ls = C[0], '-'
  t_seeds = [
      pts for pts in base._read_csv_seed_series(
          LOG_ROOT, NAME, split='learner',
          x_col=base.TRAIN_X_COL, y_col=base.TRAIN_METRIC)
      if pts]
  e_raw = base._read_csv_seed_series(
      LOG_ROOT, NAME, split='eval',
      x_col=base.EVAL_X_COL, y_col=base.EVAL_METRIC)
  e_seeds = [
      base._iters_to_env_steps(pts, LOG_ROOT, NAME)
      for pts in e_raw if pts]
  txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
  exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
  n = max(n_t, n_e)
  lab = f'{label}  (n={n})'
  if txs:
    xs, mean, se = base._subsample_curve(txs, tmean, tse)
    _shade(axes[0], xs, mean, se, color=color, n=n_t)
    axes[0].plot(
        xs, mean, color=color, linestyle=ls, linewidth=2.2,
        alpha=0.95, label=lab, zorder=3)
  sm = []
  if exs:
    _shade(axes[1], exs, emean, ese, color=color, n=n_e)
    sm = base._plot_eval_smoothed(
        axes[1], exs, emean, color=color, label=lab, linestyle=ls,
        zorder=3, linewidth=2.2)
  if tmean:
    print(f'  train last={tmean[-1]:.3f} peak={max(tmean):.3f} '
          f'@ {txs[-1]/1e6:.1f}M  n={n_t}')
  if emean:
    print(f'  eval raw last={emean[-1]:.3f} peak={max(emean):.3f}')
  if sm:
    print(f'  eval smooth last={sm[-1]:.3f}')
  if not txs and not exs:
    print('  no data yet')

  _style(
      axes[0],
      title='c8t2  ·  dgr + train-latch skip PPO  ·  train',
      ylabel='Train Success (last 1000)', xlabel=False)
  _style(
      axes[1],
      title=f'c8t2  ·  dgr + train-latch skip PPO  ·  eval (roll mean w={w})',
      ylabel=f'Eval Success (roll mean w={w})', xlabel=True)
  axes[0].legend(
      loc='upper left', fontsize=8.5, framealpha=0.95,
      fancybox=False, edgecolor='#333333')
  fig.text(
      0.5, 0.015,
      'Same compact-small dgr as the blue c8 run. After 4 consecutive '
      'iters with train_success_1000 > 0.8, later PPO updates are skipped. '
      f'Train raw. Eval faint + roll mean w={w}. Mean ±1 SE.',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')
  _plot_per_seed()


if __name__ == '__main__':
  main()
