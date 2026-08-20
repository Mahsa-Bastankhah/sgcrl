#!/usr/bin/env python3
"""Train diagnostics around the c8t2 NF catwp success lock and collapse.

Shades the lock window (eval/train 2cm success) and marks the KL spike
at iter 1872. Train metrics are not smoothed.

  python scripts/plot_c8t2_nf_collapse_diagnostics.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(
    REPO,
    'logs',
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256'
    '_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_300m_crl10',
    'ppo_builderbench_creative_8_task2_0',
    'logs',
    'learner',
    'logs.csv',
)
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c8t2_nf_collapse_diagnostics.png')

LOCK_LO, LOCK_HI = 1680, 1872  # inclusive lock; collapse starts 1873
SPIKE_ITER = 1872
FULL_LO = 1500
ZOOM_LO, ZOOM_HI = 1865, 1880

C_SUCC = '#2A9D8F'
C_SCALE = '#4C9BE8'
C_SMIN = '#C45C26'
C_KL = '#9B2226'
C_CLIP = '#6B8E9F'
C_ENT = '#A84CE8'
C_RNORM = '#E8834C'
C_RAW = '#4CE8D4'
C_RETSTD = '#E8D44C'
C_VAL = '#4C9BE8'
C_RET = '#E84C6F'
C_ADV = '#2A9D8F'


def _load(path: str) -> dict:
  with open(path, newline='') as fh:
    rows = list(csv.DictReader(fh))
  out = {k: [] for k in rows[0]}
  for r in rows:
    for k, v in r.items():
      try:
        out[k].append(float(v))
      except (TypeError, ValueError):
        out[k].append(float('nan'))
  return out


def _shade(ax, ymin=None, ymax=None):
  ax.axvspan(LOCK_LO, LOCK_HI + 0.5, color=C_SUCC, alpha=0.12, lw=0, zorder=0)
  ax.axvline(SPIKE_ITER, color=C_KL, ls='--', lw=0.9, alpha=0.85, zorder=2)
  if ymin is not None and ymax is not None:
    ax.set_ylim(ymin, ymax)


def _style(ax, ylabel: str, xlabel: bool = False):
  ax.set_ylabel(ylabel, fontsize=8)
  ax.tick_params(labelsize=7)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  if xlabel:
    ax.set_xlabel('iteration', fontsize=8)


def _ylim_visible(ax, it, *yss, lo: float, hi: float, pad: float = 0.08):
  """Y-limits from points inside [lo, hi], not the whole training run."""
  vals = []
  for xs, ys in zip([it] * len(yss), yss):
    for x, y in zip(xs, ys):
      if lo <= x <= hi and y == y:
        vals.append(y)
  if not vals:
    return
  ymin, ymax = min(vals), max(vals)
  if ymin == ymax:
    span = max(abs(ymin) * 0.05, 1e-3)
    ax.set_ylim(ymin - span, ymax + span)
    return
  extra = (ymax - ymin) * pad
  ax.set_ylim(ymin - extra, ymax + extra)


def main() -> None:
  d = _load(CSV_PATH)
  it = d['iteration']

  def series(*names):
    for n in names:
      if n in d:
        return d[n]
    raise KeyError(names)

  succ = series('train_success_mean')
  smean = series('ppo/policy_scale_mean')
  smin = series('ppo/policy_scale_min')
  kl = series('ppo/approx_kl')
  clip = series('ppo/clipfrac')
  ent = series('ppo/entropy')
  rnorm = series('reward_repr_mean')
  raw = series('reward_repr_raw_mean')
  retstd = series('reward_return_norm_std')
  vmean = series('value_mean')
  rmean = series('returns_mean')
  vloss = series('ppo/v_loss')
  advs = series('advantage_std')
  advm = series('advantage_mean')

  fig = plt.figure(figsize=(11.2, 12.4))
  gs = GridSpec(
      6, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1.15],
      hspace=0.38, wspace=0.28, left=0.07, right=0.98, top=0.93, bottom=0.05)

  def full_ax(r, c):
    ax = fig.add_subplot(gs[r, c])
    ax.set_xlim(FULL_LO, it[-1])
    return ax

  # --- row 0: success, entropy ---
  ax = full_ax(0, 0)
  ax.plot(it, succ, color=C_SUCC, lw=1.3)
  _shade(ax, -0.05, 1.05)
  _style(ax, 'train success')
  ax.set_title('c8t2 NF catwp  ·  lock 1680–1872,  KL spike 1872',
               fontsize=10, loc='left', pad=6)

  ax = full_ax(0, 1)
  ax.plot(it, ent, color=C_ENT, lw=1.3)
  _shade(ax)
  _style(ax, 'entropy')

  # --- row 1: scales, kl ---
  ax = full_ax(1, 0)
  ax.plot(it, smean, color=C_SCALE, lw=1.3, label='mean')
  ax.plot(it, smin, color=C_SMIN, lw=1.3, label='min')
  _shade(ax)
  _style(ax, 'policy scale (pre-tanh)')
  ax.legend(fontsize=7, frameon=False, loc='upper right')

  ax = full_ax(1, 1)
  ax.plot(it, kl, color=C_KL, lw=1.2, label='approx KL')
  ax.set_ylabel('approx KL', fontsize=8, color=C_KL)
  ax.tick_params(axis='y', labelcolor=C_KL, labelsize=7)
  ax2 = ax.twinx()
  ax2.plot(it, clip, color=C_CLIP, lw=1.1, alpha=0.9, label='clipfrac')
  ax2.set_ylabel('clipfrac', fontsize=8, color=C_CLIP)
  ax2.tick_params(axis='y', labelcolor=C_CLIP, labelsize=7)
  ax2.spines['top'].set_visible(False)
  _shade(ax)
  ax.set_xlim(FULL_LO, it[-1])
  ax.tick_params(labelsize=7)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left')

  # --- row 2: normalized reward vs raw NF logp ---
  ax = full_ax(2, 0)
  ax.plot(it, rnorm, color=C_RNORM, lw=1.3)
  _shade(ax)
  _style(ax, 'reward after return-norm\n(+1 bonus if 2cm)')
  _ylim_visible(ax, it, rnorm, lo=FULL_LO, hi=it[-1])

  ax = full_ax(2, 1)
  ax.plot(it, raw, color='#1B7A6E', lw=1.3)
  _shade(ax)
  _style(ax, 'raw NF log p  (on-policy)')

  # --- row 3: return std, value/returns ---
  ax = full_ax(3, 0)
  ax.plot(it, retstd, color='#B8860B', lw=1.3)
  ax.set_xlim(0, it[-1])  # full run: early decay then lock
  _shade(ax)
  _style(ax, 'return-norm std (full run)', xlabel=True)
  _ylim_visible(ax, it, retstd, lo=1, hi=it[-1])  # skip dummy std=1 at iter 0

  ax = full_ax(3, 1)
  ax.plot(it, vmean, color=C_VAL, lw=1.3, label='value mean')
  ax.plot(it, rmean, color=C_RET, lw=1.2, label='returns mean')
  ax2 = ax.twinx()
  ax2.plot(it, vloss, color='#6B8E9F', lw=1.0, alpha=0.85, label='v_loss')
  ax2.set_ylabel('v_loss', fontsize=8, color='#6B8E9F')
  ax2.tick_params(axis='y', labelcolor='#6B8E9F', labelsize=7)
  ax2.spines['top'].set_visible(False)
  _shade(ax)
  _style(ax, 'value / GAE returns')
  _ylim_visible(ax, it, vmean, rmean, lo=FULL_LO, hi=it[-1])
  _ylim_visible(ax2, it, vloss, lo=FULL_LO, hi=it[-1])
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left')

  # --- row 4: advantage mean (full run) + std (1500→end, includes fall) ---
  ax = full_ax(4, 0)
  ax.plot(it, advm, color=C_ADV, lw=1.3)
  ax.set_xlim(0, it[-1])
  _shade(ax)
  _style(ax, 'advantage mean (full run)', xlabel=True)
  _ylim_visible(ax, it, advm, lo=2, hi=it[-1])  # skip iter 0–1 (−55 / −6.8)

  ax = full_ax(4, 1)
  ax.plot(it, advs, color=C_ADV, lw=1.3)
  _shade(ax)
  _style(ax, 'advantage std', xlabel=True)
  _ylim_visible(ax, it, advs, lo=FULL_LO, hi=it[-1])  # skip dummy std~410 at iter 0

  # --- row 5: zoom 1865-1880 spanning both columns ---
  gs_zoom = gs[5, :].subgridspec(1, 5, wspace=0.30)
  zoom_specs = [
      ('train success', succ, C_SUCC, (-0.05, 1.05)),
      ('scale mean / min', None, None, None),
      ('approx KL', kl, C_KL, None),
      ('value / returns', None, None, None),
      ('v_loss', vloss, '#6B8E9F', None),
  ]
  for i, (ylab, ys, color, ylim) in enumerate(zoom_specs):
    ax = fig.add_subplot(gs_zoom[0, i])
    ax.set_xlim(ZOOM_LO, ZOOM_HI)
    ax.axvspan(LOCK_LO, LOCK_HI + 0.5, color=C_SUCC, alpha=0.12, lw=0, zorder=0)
    ax.axvline(SPIKE_ITER, color=C_KL, ls='--', lw=0.9, alpha=0.85)
    if i == 1:
      ax.plot(it, smean, color=C_SCALE, lw=1.4, label='mean')
      ax.plot(it, smin, color=C_SMIN, lw=1.4, label='min')
      ax.legend(fontsize=6, frameon=False, loc='upper left')
    elif i == 3:
      ax.plot(it, vmean, color=C_VAL, lw=1.4, label='V')
      ax.plot(it, rmean, color=C_RET, lw=1.3, label='ret')
      ax.legend(fontsize=6, frameon=False, loc='upper right')
    else:
      ax.plot(it, ys, color=color, lw=1.4)
    if ylim is not None:
      ax.set_ylim(*ylim)
    elif i == 3:
      _ylim_visible(ax, it, vmean, rmean, lo=ZOOM_LO, hi=ZOOM_HI, pad=0.12)
    elif i == 4:
      _ylim_visible(ax, it, vloss, lo=ZOOM_LO, hi=ZOOM_HI, pad=0.12)
    _style(ax, ylab, xlabel=True)
    if i == 0:
      ax.set_title('zoom  1865–1880', fontsize=9, loc='left', pad=4)

  legend = [
      Patch(facecolor=C_SUCC, alpha=0.25, label='success lock (1680–1872)'),
      Line2D([0], [0], color=C_KL, ls='--', lw=1.0, label='KL spike (1872)'),
  ]
  fig.legend(handles=legend, loc='upper right', fontsize=8, frameon=False,
             bbox_to_anchor=(0.98, 0.995))

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=160)
  plt.close(fig)
  print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
  main()
