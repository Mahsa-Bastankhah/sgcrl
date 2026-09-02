#!/usr/bin/env python3
"""LP contrastive (SGCRL) Sawyer bin + peg: train/eval success mean ± SE.

Paper version (SGCRL + PPO+NF, serif fonts) lives at
paper_plot_scripts/plot_sawyer_bin_peg_success.py.

Reads the three seeds under logs/final_metaworld_runs/lp_contrastive_sawyer_{bin,peg}_40m/
(symlinks to the live run dirs). Train is raw actor success_1000. Eval is
evaluator success_1000: faint raw + centered rolling mean, window=5.

  python scripts/plot_lp_contrastive_sawyer_bin_peg_success.py
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
LOG_ROOT = os.path.join(REPO, 'logs', 'final_metaworld_runs')
OUT = os.path.join(
    REPO, 'figs', 'metaworld', 'final_metaworld_runs',
    'lp_contrastive_sawyer_bin_peg_success.png')

ENVS = [
    ('bin', 'lp_contrastive_sawyer_bin_40m', 'Sawyer bin'),
    ('peg', 'lp_contrastive_sawyer_peg_40m', 'Sawyer peg'),
]
COLOR = base.ACCENT_COLORS[0]


def _read_success(run_dir: str, split: str) -> list[tuple[int, float]]:
  path = os.path.join(run_dir, 'logs', split, 'logs.csv')
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return []
  pts: list[tuple[int, float]] = []
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    if not reader.fieldnames:
      return []
    fields = set(reader.fieldnames)
    y_key = 'success_1000' if 'success_1000' in fields else (
        'success' if 'success' in fields else None)
    if y_key is None:
      return []
    x_key = 'actor_steps' if 'actor_steps' in fields else None
    if x_key is None:
      return []
    for row in reader:
      x = base._coerce(row.get(x_key))
      y = base._coerce(row.get(y_key))
      if x is None or y is None:
        continue
      pts.append((int(x), float(y)))
  pts.sort(key=lambda p: p[0])
  return pts


def _seed_series(cfg_dir: str, split: str) -> list[list[tuple[int, float]]]:
  if not os.path.isdir(cfg_dir):
    return []
  out = []
  for name in sorted(os.listdir(cfg_dir)):
    run_dir = os.path.join(cfg_dir, name)
    if not os.path.isdir(run_dir):
      continue
    pts = _read_success(run_dir, split)
    if pts:
      out.append(pts)
  return out


def _shade(ax, xs, mean, se, *, color, n: int) -> None:
  if n <= 1 or not se:
    return
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  ax.fill_between(xs, lo, hi, color=color, alpha=0.22, lw=0, zorder=2)


def _style(ax, *, title: str, ylabel: str, xlabel: bool) -> None:
  ax.set_title(title, fontsize=12, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=10)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def main() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.6), sharex='col', sharey=True)
  fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.12,
                      wspace=0.18, hspace=0.28)

  for col, (key, cfg_name, title) in enumerate(ENVS):
    cfg_dir = os.path.join(LOG_ROOT, cfg_name)
    t_seeds = _seed_series(cfg_dir, 'actor')
    e_seeds = _seed_series(cfg_dir, 'evaluator')
    txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
    exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
    n = max(n_t, n_e)
    lab = f'SGCRL / LP contrastive  (n={n})'
    ax_t, ax_e = axes[0, col], axes[1, col]

    if txs:
      xs, mean, se = base._subsample_curve(txs, tmean, tse)
      _shade(ax_t, xs, mean, se, color=COLOR, n=n_t)
      ax_t.plot(xs, mean, color=COLOR, lw=2.2, alpha=0.95, label=lab, zorder=3)
      print(f'  {key} train last={tmean[-1]:.3f} peak={max(tmean):.3f} '
            f'@ {txs[-1]/1e6:.2f}M  n={n_t}')
    else:
      print(f'  {key} train: no data under {cfg_dir}')

    sm = []
    if exs:
      xs, mean, se = base._subsample_curve(exs, emean, ese)
      lo = base._rolling_mean([m - s for m, s in zip(mean, se)], w)
      hi = base._rolling_mean([m + s for m, s in zip(mean, se)], w)
      if n_e > 1:
        ax_e.fill_between(xs, lo, hi, color=COLOR, alpha=0.18, lw=0, zorder=2)
      sm = base._plot_eval_smoothed(
          ax_e, xs, mean, color=COLOR, label=lab, zorder=3, linewidth=2.2)
      print(f'  {key} eval raw last={emean[-1]:.3f} peak={max(emean):.3f} '
            f'@ {exs[-1]/1e6:.2f}M  n={n_e}')
      if sm:
        print(f'  {key} eval smooth last={sm[-1]:.3f}')
    else:
      print(f'  {key} eval: no data under {cfg_dir}')

    _style(ax_t, title=f'{title}  ·  train',
           ylabel='Train Success (last 1000)', xlabel=False)
    _style(ax_e, title=f'{title}  ·  eval (roll mean w={w})',
           ylabel=f'Eval Success (roll mean w={w})', xlabel=True)
    if n:
      ax_t.legend(loc='upper left', fontsize=8.5, framealpha=0.95,
                  fancybox=False, edgecolor='#333333')

  fig.text(
      0.5, 0.015,
      'LP contrastive (contrastive_cpc) 40M, seeds 0–2. Solid = mean across seeds; '
      'shade = ±1 SE. Train: raw actor success_1000. '
      f'Eval: faint raw + bold rolling mean, window={w}.',
      ha='center', va='bottom', fontsize=7.4, color='#555555')
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
