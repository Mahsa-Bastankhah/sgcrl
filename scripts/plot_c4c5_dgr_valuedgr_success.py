#!/usr/bin/env python3
"""c4t1 / c4t2 / c5t2 / c5t4 valuedgr: per-seed success and ‖∇_s V‖.

Train raw. Eval faint raw + roll mean w=5. Value-grad from learner CSV.

  python scripts/plot_c4c5_dgr_valuedgr_success.py
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
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(REPO, 'figs', 'builderbench', 'final_runs')
C = base.ACCENT_COLORS
GRAD_REG_C = 100.0
GRAD_COL = 'ppo/value_grad_s_norm_mean'
GRAD_MAX_COL = 'ppo/value_grad_s_norm_max'
SPI = 1024 * 50

STEM = (
    'e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_'
    'nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_'
    'ep50_200m_crl10_dualgradreg_c100_lamlr1e6_'
    'valuedgr_c100_lamlr1e6_warp_2h30'
)
TASKS = (
    ('c4t1', 'creative4_task1', 'c4t1'),
    ('c4t2', 'creative4_task2', 'c4t2 two towers'),
    ('c5t2', 'creative5_task2', 'c5t2 pyramid'),
    ('c5t4', 'creative5_task4', 'c5t4 overhang-5'),
)


def _seed_label(run_name: str) -> str:
  if run_name.rsplit('_', 1)[-1].isdigit():
    return f'seed {run_name.rsplit("_", 1)[-1]}'
  return run_name


def _load_xy(path: str, x_col: str, y_col: str):
  xs, ys = [], []
  if not os.path.isfile(path):
    return xs, ys
  with open(path, newline='') as f:
    for row in csv.DictReader(f):
      x = base._coerce(row.get(x_col, ''))
      y = base._coerce(row.get(y_col, ''))
      if x is not None and y is not None:
        xs.append(int(x))
        ys.append(float(y))
  return xs, ys


def _load_eval(t_path: str, e_path: str):
  it_map = {}
  if os.path.isfile(t_path):
    with open(t_path, newline='') as f:
      for row in csv.DictReader(f):
        it = base._coerce(row.get('iteration', ''))
        gs = base._coerce(row.get(base.TRAIN_X_COL, ''))
        if it is not None and gs is not None:
          it_map[int(it)] = int(gs)
  spi = base._infer_steps_per_iter(it_map) if it_map else SPI
  xs, ys = [], []
  if not os.path.isfile(e_path):
    return xs, ys
  with open(e_path, newline='') as f:
    for row in csv.DictReader(f):
      it = base._coerce(row.get(base.EVAL_X_COL, ''))
      y = base._coerce(row.get(base.EVAL_METRIC, ''))
      if it is None or y is None:
        continue
      it = int(it)
      xs.append(it_map[it] if it in it_map else int((it + 1) * spi))
      ys.append(float(y))
  return xs, ys


def _plot_one(tag: str, name: str, title: str) -> str:
  w = base.EVAL_SMOOTH_WINDOW
  root = os.path.join(LOG_ROOT, f'ppo_builderbench_{name}_{STEM}')
  out = os.path.join(FIGS, f'{tag}_dgr_valuedgr_per_seed_success_grad_s.png')
  try:
    runs = sorted(
        n for n in os.listdir(root)
        if os.path.isdir(os.path.join(root, n)))
  except OSError:
    runs = []

  fig, axes = plt.subplots(3, 1, figsize=(8.8, 10.2), sharex=True)
  fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.10, hspace=0.20)
  n_plotted = 0
  print(f'== {tag} ==')
  for i, run in enumerate(runs):
    color = C[i % len(C)]
    lab = _seed_label(run)
    t_path = os.path.join(root, run, 'logs', 'learner', 'logs.csv')
    e_path = os.path.join(root, run, 'logs', 'eval', 'logs.csv')
    txs, tys = _load_xy(t_path, base.TRAIN_X_COL, base.TRAIN_METRIC)
    gxs, gys = _load_xy(t_path, base.TRAIN_X_COL, GRAD_COL)
    _, gmax = _load_xy(t_path, base.TRAIN_X_COL, GRAD_MAX_COL)
    exs, eys = _load_eval(t_path, e_path)
    if not txs and not exs and not gxs:
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
    if gxs:
      axes[2].plot(
          gxs, gys, color=color, lw=1.8, alpha=0.95, label=lab, zorder=3)
      if gmax:
        axes[2].plot(
            gxs, gmax, color=color, lw=1.0, ls=':', alpha=0.55, zorder=2)
      print(f'  {lab} ‖∇_s V‖ mean last={gys[-1]:.3f} peak={max(gys):.3f}')

  for ax, ttl, ylabel, ylim in (
      (axes[0], f'{title}  ·  NF+value dgr  ·  train by seed',
       'Train Success (last 1000)', (-0.05, 1.05)),
      (axes[1],
       f'{title}  ·  NF+value dgr  ·  eval by seed (roll mean w={w})',
       f'Eval Success (roll mean w={w})', (-0.05, 1.05)),
      (axes[2], rf'{title}  ·  $\|\nabla_s V\|$  by seed',
       r'$\|\nabla_s V\|$', None),
  ):
    ax.set_title(ttl, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if ylim is not None:
      ax.set_ylim(*ylim)
  axes[2].axhline(
      GRAD_REG_C, color='0.35', lw=0.9, ls='--', zorder=1,
      label=f'c={GRAD_REG_C:g}')
  axes[2].set_xlabel('Env Steps', fontsize=10)
  axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  if n_plotted:
    axes[0].legend(
        loc='upper left', fontsize=8.5, framealpha=0.95,
        fancybox=False, edgecolor='#333333')
    axes[2].legend(
        loc='upper right', fontsize=8.0, framealpha=0.95,
        fancybox=False, edgecolor='#333333')
  fig.text(
      0.5, 0.012,
      'One line per seed. Train: raw train_success_1000. '
      f'Eval: faint raw + bold rolling mean, window={w}. '
      r'T=ep=50 · 200M · NF + value dualgradreg $c=100$ $\lambda$lr$=10^{-6}$.',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  os.makedirs(os.path.dirname(out), exist_ok=True)
  tmp = out + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, out)
  plt.close(fig)
  print(f'→ {out}')
  return out


def main() -> None:
  for tag, name, title in TASKS:
    _plot_one(tag, name, title)


if __name__ == '__main__':
  main()
