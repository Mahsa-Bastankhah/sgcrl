#!/usr/bin/env python3
"""c8t2 NF compact catwp: on-policy raw repr reward for three reward modes.

Stacked panels (own y-scales) because χ² explodes, reverse sits at exp(-80),
and forward log p is O(10–100). Train metric: no smoothing.

  python scripts/plot_c8t2_nf_reward_mode_raw.py
"""
from __future__ import annotations

import csv
import math
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
    REPO, 'figs', 'builderbench', 'c8t2',
    'c8t2_nf_chisq_dualgradreg_revrew_reward_repr_raw.png')
PLOT_MAX_PTS = 2500
RUN = 'ppo_builderbench_creative_8_task2_0'

# (label, job, log_dir, yscale, color, formula)
RUNS = (
    (
        r'χ²  $r=-\exp(-\mathrm{clip}(\log p,\pm80))$  (3718483)',
        'logs/ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256'
        '_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5'
        '_ent005_anneal_ep60_300m_crl10_chisqrew',
        'symlog',
        '#E07A3D',
        1e10,
    ),
    (
        r'forward + dualgradreg $c{=}200$  $r=\log p$  (3718386)',
        'logs/ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256'
        '_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5'
        '_ent0005_anneal_ep60_300m_crl10_dualgradreg_c200_lamlr1e4',
        'linear',
        '#2A9D8F',
        None,
    ),
    (
        r'reverse  $r=\exp(\mathrm{clip}(\log p,\pm80))$  (3718482)',
        'logs/ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256'
        '_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5'
        '_ent005_anneal_ep60_300m_crl10_revrew',
        'log',
        '#6A4C93',
        None,
    ),
)


def _csv_path(log_dir: str) -> str:
  return os.path.join(
      REPO, log_dir, RUN, 'logs', 'learner', 'logs.csv')


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


def _load_raw(path: str) -> tuple[list[float], list[float], int, int]:
  """Return (xs, ys finite), n_rows, n_nonfinite. ±inf/nan dropped from ys."""
  xs, ys = [], []
  n_rows = 0
  n_nonfinite = 0
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return xs, ys, n_rows, n_nonfinite
  with open(path, newline='') as fh:
    reader = csv.reader(fh)
    header = next(reader, None)
    if not header:
      return xs, ys, n_rows, n_nonfinite
    try:
      x_i = header.index('global_step')
      y_i = header.index('reward_repr_raw_mean')
    except ValueError:
      return xs, ys, n_rows, n_nonfinite
    for row in reader:
      if len(row) <= max(x_i, y_i):
        continue
      x = base._coerce(row[x_i])
      if x is None:
        continue
      n_rows += 1
      try:
        y = float(row[y_i])
      except Exception:
        n_nonfinite += 1
        continue
      if math.isnan(y) or math.isinf(y):
        n_nonfinite += 1
        continue
      xs.append(float(x) / 1e6)
      ys.append(float(y))
  return xs, ys, n_rows, n_nonfinite


def main() -> None:
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.4), sharex=True)
  fig.suptitle(
      'c8t2 NF compact catwp  ·  on-policy raw repr reward  '
      r'($\mathtt{reward\_repr\_raw\_mean}$, pre return-norm / extrew)',
      fontsize=11)

  for ax, (label, log_dir, yscale, color, linthresh) in zip(axes, RUNS):
    xs, ys, n_rows, n_bad = _load_raw(_csv_path(log_dir))
    xs, ys = _downsample(xs, ys)
    ax.plot(xs, ys, color=color, lw=1.25, label=label)
    ax.set_ylabel('raw $r$', fontsize=9)
    if yscale == 'symlog':
      ax.set_yscale('symlog', linthresh=linthresh)
    elif yscale == 'log':
      ax.set_yscale('log')
    ax.grid(True, alpha=0.28)
    ax.legend(loc='best', fontsize=8, framealpha=0.92)
    note = f'n={n_rows}'
    if n_bad:
      note += f'  dropped ±inf/nan={n_bad} ({100.0 * n_bad / max(n_rows, 1):.0f}%)'
    if xs:
      note += f'  last={xs[-1]:.1f}M'
    ax.text(
        0.99, 0.08, note, transform=ax.transAxes, ha='right', va='bottom',
        fontsize=7.5, color='0.35')

  axes[-1].set_xlabel('env steps (M)')
  axes[-1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(OUT_PATH, dpi=140)
  print('wrote', OUT_PATH)


if __name__ == '__main__':
  main()
