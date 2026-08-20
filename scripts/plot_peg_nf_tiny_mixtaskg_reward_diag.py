#!/usr/bin/env python3
"""Overlay success + NF reward / density diagnostics for peg mixtaskg runs.

Train success is raw ``train_success_1000``. Eval success is faint raw +
bold rolling mean (window=5). Remaining panels are raw.

  python scripts/plot_peg_nf_tiny_mixtaskg_reward_diag.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'metaworld', 'peg_nf_tiny_mixtaskg_reward_diag.png')
STEPS_PER_ITER = 1024  # num_envs=4 × rollout_length=256

_LOG_PREFIX = (
    'ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_rand'
    '_minstd1e5_ent0005_mixtaskg'
)

# (legend, log_dir, seed run dir, color)
RUNS = (
    ('mixtaskg s0', _LOG_PREFIX, 'ppo_sawyer_peg_0', '#4C9BE8'),
    ('mixtaskg s1', _LOG_PREFIX, 'ppo_sawyer_peg_1', '#9B59B6'),
    ('mask10', f'{_LOG_PREFIX}_mask10', 'ppo_sawyer_peg_0', '#2A9D8F'),
    ('grdual', f'{_LOG_PREFIX}_grdual', 'ppo_sawyer_peg_0', '#E07A3D'),
    ('retnormW1e6', f'{_LOG_PREFIX}_retnormW1e6', 'ppo_sawyer_peg_0', '#C44E52'),
)

# (source, column, ylabel, yscale)  source is 'learner' or 'eval'
PANELS = (
    ('learner', 'train_success_1000', 'Train success (last 1000)', 'success'),
    ('eval', 'success', 'Eval success (roll mean w=5)', 'success'),
    ('learner', 'reward_repr_raw_mean', 'NF reward (raw log p)', 'linear'),
    ('learner', 'nf/log_p_mean', 'replay log p mean', 'linear'),
    ('learner', 'nf/log_p_min', 'replay log p min', 'linear'),
    ('learner', 'nf/log_p_max', 'replay log p max', 'linear'),
    ('learner', 'nf/nf_grad_reg_raw', 'grad-reg raw', 'linear'),
    ('learner', 'reward_return_norm_std', 'return-norm std', 'log'),
)
PLOT_MAX_PTS = 2500


def _csv_path(log_dir: str, run: str, kind: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, run, 'logs', kind, 'logs.csv')


def _downsample(xs, ys, max_pts: int = PLOT_MAX_PTS):
  n = len(xs)
  if n <= max_pts:
    return xs, ys
  stride = max(1, n // max_pts)
  xs_d = list(xs[::stride])
  ys_d = list(ys[::stride])
  if xs_d[-1] != xs[-1]:
    xs_d.append(xs[-1])
    ys_d.append(ys[-1])
  return xs_d, ys_d


def _load_cols(
    path: str, x_col: str, y_cols: tuple[str, ...],
) -> dict[str, tuple[list, list]]:
  out = {c: ([], []) for c in y_cols}
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return out
  with open(path, newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader, None)
    if not header:
      return out
    try:
      x_i = header.index(x_col)
      y_is = [(c, header.index(c)) for c in y_cols if c in header]
    except ValueError:
      return out
    for row in reader:
      if len(row) <= x_i:
        continue
      x = base._coerce(row[x_i])
      if x is None:
        continue
      for c, i in y_is:
        if i >= len(row):
          continue
        y = base._coerce(row[i])
        if y is None:
          continue
        out[c][0].append(float(x))
        out[c][1].append(float(y))
  return out


def main() -> None:
  learn_cols = tuple(c for src, c, _, _ in PANELS if src == 'learner')
  loaded = []
  for label, log_dir, run, color in RUNS:
    series = _load_cols(
        _csv_path(log_dir, run, 'learner'), 'global_step', learn_cols)
    ev_raw = _load_cols(
        _csv_path(log_dir, run, 'eval'), 'iteration', ('success',))
    ev_it, ev_y = ev_raw.get('success', ([], []))
    ev = ([it * STEPS_PER_ITER for it in ev_it], ev_y)
    n_tr = len(series.get('train_success_1000', ([], []))[0])
    print(f'{label}: train={n_tr} pts  eval={len(ev[0])} pts')
    loaded.append((label, color, series, ev))

  fig, axes = plt.subplots(
      len(PANELS), 1, figsize=(10.0, 16.0), sharex=True)

  for ax, (src, col, ylabel, yscale) in zip(axes, PANELS):
    for label, color, series, ev in loaded:
      if src == 'eval':
        xs, ys = ev
        if not xs:
          continue
        base._plot_eval_smoothed(
            ax, xs, ys, color=color, label=label, linewidth=1.8)
        continue
      xs, ys = series.get(col, ([], []))
      if not xs:
        continue
      xs, ys = _downsample(xs, ys)
      ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.92, label=label)
    ax.set_ylabel(ylabel, fontsize=9)
    if yscale == 'success':
      ax.set_ylim(-0.05, 1.05)
    elif yscale == 'log':
      ax.set_yscale('log')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if yscale == 'linear':
      ax.axhline(0.0, color='0.75', lw=0.6, zorder=0)

  axes[0].set_title(
      'Sawyer peg · tiny NF · mixtaskg — success + NF diagnostics',
      fontsize=11, fontweight='bold')
  axes[0].legend(loc='best', fontsize=8, framealpha=0.95, ncol=3)
  axes[-1].set_xlabel('Env Steps', fontsize=10)
  axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
