#!/usr/bin/env python3
"""c5t2 compact-small: dualgradreg vs no-gradreg log-p / score / NLL overlay.

No exact no-gradreg twin of dualgradreg (norand catselect τ=0.5 ep50). Closest
finished compact-small run with ‖∇_s log p‖ logged is τ=0.9 ep60 (no gradreg).
Train traces are raw.

  python scripts/plot_c5t2_dualgradreg_vs_gradreg_logp.py
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
    REPO, 'figs', 'builderbench', 'c5t2',
    'c5t2_dualgradreg_vs_nogradreg_logp.png')
PLOT_MAX_PTS = 2500
GRAD_REG_C = 100.0

RUNS = (
    (
        'dualgradreg (τ=0.5 · ep50)',
        'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_norand_catselect_dualgradreg',
        '#2A9D8F',
    ),
    (
        'no gradreg (τ=0.9 · ep60)',
        'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau09_nopermute_fixedx01_catselect'
        '_minstd1e5_entanneal_ep60_300m_crl10',
        '#4C9BE8',
    ),
)
RUN = 'ppo_builderbench_creative_5_task2_0'

PANELS = (
    ('nf/log_p_min', 'replay log p min'),
    ('nf/log_p_mean', 'replay log p mean'),
    ('reward_repr_raw_mean', 'rollout log p (reward)'),
    ('nf/log_p_max', 'replay log p max'),
    ('nf/nf_logp_grad_s_norm_mean', r'mean $\|\nabla_s \log p\|$'),
    ('nf/density_loss', 'NF NLL'),
)
LEARN_COLS = tuple(c for c, _ in PANELS) + ('train_success_1000',)


def _csv_path(log_dir: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, RUN, 'logs', 'learner', 'logs.csv')


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


def _load_cols(path: str, y_cols: tuple[str, ...]) -> dict:
  out = {c: ([], []) for c in y_cols}
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return out
  with open(path, newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader, None)
    if not header:
      return out
    try:
      x_i = header.index('global_step')
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


def main() -> None:
  loaded = []
  for label, log_dir, color in RUNS:
    series = _load_cols(_csv_path(log_dir), LEARN_COLS)
    print(label)
    for col, _ in PANELS:
      xs, ys = series.get(col, ([], []))
      if not ys:
        print(f'  {col}: no data')
        continue
      peak_i = max(range(len(ys)), key=lambda i: ys[i])
      trough_i = min(range(len(ys)), key=lambda i: ys[i])
      print(
          f'  {col}: n={len(ys)} last={ys[-1]:.4g} @ {xs[-1]:.0f}  '
          f'peak={ys[peak_i]:.4g} trough={ys[trough_i]:.4g}')
    xs, ys = series.get('train_success_1000', ([], []))
    if ys:
      peak_i = max(range(len(ys)), key=lambda i: ys[i])
      print(
          f'  train_success_1000: last={ys[-1]:.3f} peak={ys[peak_i]:.3f} '
          f'@ {xs[peak_i]:.0f}')
    loaded.append((label, color, series))

  fig, axes = plt.subplots(
      len(PANELS), 1, figsize=(10.0, 13.6), sharex=True)

  for ax, (col, ylabel) in zip(axes, PANELS):
    for label, color, series in loaded:
      xs, ys = series.get(col, ([], []))
      if not xs:
        continue
      xs, ys = _downsample(xs, ys)
      ax.plot(xs, ys, color=color, linewidth=1.35, alpha=0.92, label=label)
    ax.set_ylabel(ylabel, fontsize=9)
    if col == 'nf/nf_logp_grad_s_norm_mean':
      ax.axhline(
          GRAD_REG_C, color='0.35', lw=0.8, ls=':', zorder=0,
          label=f'c={GRAD_REG_C:g}')
    else:
      ax.axhline(0.0, color='0.75', lw=0.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

  axes[0].set_title(
      'c5t2 NF compact-small · dualgradreg vs no gradreg',
      fontsize=11, fontweight='bold')
  axes[0].legend(loc='best', fontsize=9, framealpha=0.95)
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
