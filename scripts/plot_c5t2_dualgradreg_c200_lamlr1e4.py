#!/usr/bin/env python3
"""c5t2 compact-small dualgradreg (c=200, lam_lr=1e-4): success + grad-reg.

Train success is raw ``train_success_1000``. Eval is faint raw + bold rolling
mean (window=5). Grad-reg / λ / ‖∇_s log p‖ are raw learner traces.

  python scripts/plot_c5t2_dualgradreg_c200_lamlr1e4.py
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
LOG_DIR = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_norand_catselect_dualgradreg'
    '_c200_lamlr1e4'
)
RUN = 'ppo_builderbench_creative_5_task2_0'
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c5t2',
    'c5t2_nf_dualgradreg_c200_lamlr1e4.png')
SPI = 1024 * 50  # num_envs × episode_length (ep50)
PLOT_MAX_PTS = 2500
GRAD_REG_C = 200.0

LEARN_COLS = (
    'train_success_1000',
    'nf/nf_grad_reg_raw',
    'nf/nf_grad_reg',
    'nf/nf_grad_reg_lam',
    'nf/nf_logp_grad_s_norm_mean',
)

C_TRAIN = '#2A9D8F'
C_EVAL = '#4C9BE8'
C_RAW = '#C45C26'
C_PEN = '#6B8E9F'
C_LAM = '#A84CE8'
C_GNORM = '#9B2226'


def _csv_path(kind: str) -> str:
  return os.path.join(
      REPO, 'logs', LOG_DIR, RUN, 'logs', kind, 'logs.csv')


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
    except ValueError:
      return out
    y_is = [(c, header.index(c)) for c in y_cols if c in header]
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


def _summarize(name: str, xs, ys) -> None:
  if not ys:
    print(f'  {name}: no data')
    return
  peak_i = max(range(len(ys)), key=lambda i: ys[i])
  print(
      f'  {name}: n={len(ys)}  last={ys[-1]:.4g} @ {xs[-1]:.0f}  '
      f'peak={ys[peak_i]:.4g} @ {xs[peak_i]:.0f}')


def _style(ax):
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def main() -> None:
  series = _load_cols(_csv_path('learner'), 'global_step', LEARN_COLS)
  ev_raw = _load_cols(_csv_path('eval'), 'iteration', ('success',))
  ev_it, ev_y = ev_raw.get('success', ([], []))
  ev = ([it * SPI for it in ev_it], ev_y)

  print(LOG_DIR)
  for col in LEARN_COLS:
    xs, ys = series.get(col, ([], []))
    _summarize(col, xs, ys)
  _summarize('eval success', ev[0], ev[1])

  fig, axes = plt.subplots(5, 1, figsize=(10.0, 12.4), sharex=True)

  # --- train success ---
  ax = axes[0]
  xs, ys = series.get('train_success_1000', ([], []))
  if xs:
    xs, ys = _downsample(xs, ys)
    ax.plot(xs, ys, color=C_TRAIN, lw=1.4, label='train_success_1000')
  ax.set_ylim(-0.05, 1.05)
  ax.set_ylabel('Train success (last 1000)', fontsize=9)
  ax.legend(fontsize=8, frameon=False, loc='upper right')
  _style(ax)

  # --- eval success ---
  ax = axes[1]
  if ev[0]:
    base._plot_eval_smoothed(
        ax, ev[0], ev[1], color=C_EVAL,
        label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})')
  ax.set_ylim(-0.05, 1.05)
  ax.set_ylabel('Eval success (E=5, rolling mean w=5)', fontsize=9)
  ax.legend(fontsize=8, frameon=False, loc='upper right')
  _style(ax)

  # --- grad-reg: raw hinge + λ·hinge ---
  ax = axes[2]
  for col, color, label in (
      ('nf/nf_grad_reg_raw', C_RAW, r'raw  mean$(\max(\|\nabla_s\|-c,0))$'),
      ('nf/nf_grad_reg', C_PEN, r'$\lambda\cdot$hinge'),
  ):
    xs, ys = series.get(col, ([], []))
    if not xs:
      continue
    xs, ys = _downsample(xs, ys)
    ax.plot(xs, ys, color=color, lw=1.3, label=label)
  ax.axhline(0.0, color='0.75', lw=0.6, zorder=0)
  ax.set_ylabel('Grad-reg', fontsize=9)
  ax.legend(fontsize=8, frameon=False, loc='upper left')
  _style(ax)

  # --- lambda ---
  ax = axes[3]
  xs, ys = series.get('nf/nf_grad_reg_lam', ([], []))
  if xs:
    xs, ys = _downsample(xs, ys)
    ax.plot(xs, ys, color=C_LAM, lw=1.4)
  ax.set_yscale('log')
  ax.set_ylabel(r'grad-reg $\lambda$  (log)', fontsize=9)
  _style(ax)

  # --- ‖∇_s log p‖ ---
  ax = axes[4]
  xs, ys = series.get('nf/nf_logp_grad_s_norm_mean', ([], []))
  if xs:
    xs, ys = _downsample(xs, ys)
    ax.plot(xs, ys, color=C_GNORM, lw=1.4, label=r'mean $\|\nabla_s \log p\|$')
  ax.axhline(
      GRAD_REG_C, color=C_GNORM, lw=0.8, ls=':', alpha=0.85,
      label=f'c = {GRAD_REG_C:g}')
  ax.set_ylabel(r'mean $\|\nabla_s \log p\|$', fontsize=9)
  ax.legend(fontsize=8, frameon=False, loc='upper left')
  _style(ax)

  axes[0].set_title(
      'c5t2 NF compact-small · dualgradreg c=200  λ_lr=1e-4  ·  seed 0',
      fontsize=11, fontweight='bold')
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
