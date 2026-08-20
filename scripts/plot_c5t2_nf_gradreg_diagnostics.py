#!/usr/bin/env python3
"""c5t2 compact-small NF + gradreg: success vs density / PPO diagnostics.

Single run (seed 0). Train curves are raw. Eval success is a centered rolling
mean (window=5) with faint raw underneath.

  python scripts/plot_c5t2_nf_gradreg_diagnostics.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(
    REPO, 'logs',
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_norand_catselect_gradreg',
    'ppo_builderbench_creative_5_task2_0',
)
LEARNER_CSV = os.path.join(RUN_DIR, 'logs', 'learner', 'logs.csv')
EVAL_CSV = os.path.join(RUN_DIR, 'logs', 'eval', 'logs.csv')
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c5t2_nf_gradreg_diagnostics.png')

# Train-success windows (2cm). First lock collapses ~510; main lock peaks
# at iter 1051 (train_success_1000=0.93) then decays.
LOCKS = ((390, 480), (960, 1230))
GRAD_REG_C = 100.0

C_SUCC = '#2A9D8F'
C_SUCC2 = '#4C9BE8'
C_EVAL = '#2A9D8F'
C_NLL = '#C45C26'
C_TOTAL = '#6B8E9F'
C_FLOW = '#9B2226'
C_ENC = '#E8834C'
C_SCALE = '#4C9BE8'
C_SMIN = '#C45C26'
C_ENT = '#A84CE8'
C_KL = '#9B2226'
C_CLIP = '#6B8E9F'
C_GNORM = '#9B2226'
C_FRAC = '#4C9BE8'
C_LAM = '#A84CE8'
C_RAW = '#1B7A6E'
C_RETSTD = '#B8860B'

# Task goal (pyramid): 5 cubes × (x,y,z), dim=15
# Source: slurm log line "hard_goal (flat)="
TASK_GOAL = np.array([
    0.26999998, -0.035,  0.02,   # cube 0
    0.26999998,  0.035,  0.02,   # cube 1
    0.26999998, -0.02,   0.06,   # cube 2
    0.26999998,  0.02,   0.06,   # cube 3
    0.26999998,  0.0,    0.1,    # cube 4
], dtype=float)
GOAL_DIMS = 15


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


def _series(d: dict, *names):
  for n in names:
    if n in d:
      return d[n]
  raise KeyError(names)


def _shade(ax):
  for lo, hi in LOCKS:
    ax.axvspan(lo, hi + 0.5, color=C_SUCC, alpha=0.10, lw=0, zorder=0)


def _style(ax, ylabel: str, xlabel: bool = False):
  ax.set_ylabel(ylabel, fontsize=8)
  ax.tick_params(labelsize=7)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  if xlabel:
    ax.set_xlabel('iteration', fontsize=8)


def _normalized_goal_stats(d: dict) -> tuple:
  """Return (l2_norm, per_dim_abs_max, per_dim_matrix) arrays over iters."""
  n = len(d['iteration'])
  per_dim = np.full((n, GOAL_DIMS), float('nan'))
  for di in range(GOAL_DIMS):
    gm = np.array(d.get(f'nf/goal_mean_{di}', [float('nan')] * n))
    gs = np.array(d.get(f'nf/goal_std_{di}',  [float('nan')] * n))
    per_dim[:, di] = (TASK_GOAL[di] - gm) / (gs + 1e-8)
  l2  = np.sqrt(np.nansum(per_dim ** 2, axis=1))
  dmax = np.nanmax(np.abs(per_dim), axis=1)
  return l2, dmax, per_dim


def main() -> None:
  d = _load(LEARNER_CSV)
  ev = _load(EVAL_CSV)
  it = _series(d, 'iteration')

  norm_l2, norm_dmax, norm_per_dim = _normalized_goal_stats(d)

  fig = plt.figure(figsize=(11.4, 14.6))
  gs = GridSpec(
      6, 2, figure=fig, hspace=0.42, wspace=0.30,
      left=0.08, right=0.97, top=0.93, bottom=0.04)

  def ax_at(r, c):
    ax = fig.add_subplot(gs[r, c])
    ax.set_xlim(0, it[-1])
    _shade(ax)
    return ax

  # --- row 0: train success, eval success ---
  ax = ax_at(0, 0)
  ax.plot(it, _series(d, 'train_success_1000'), color=C_SUCC, lw=1.4,
          label='train_success_1000')
  ax.plot(it, _series(d, 'train_success_mean'), color=C_SUCC2, lw=1.1,
          alpha=0.85, label='train_success_mean')
  ax.set_ylim(-0.05, 1.05)
  _style(ax, 'train success (2cm)')
  ax.legend(fontsize=7, frameon=False, loc='upper right')
  ax.set_title(
      'c5t2 NF compact-small + gradreg  ·  catselect, no extrew  ·  seed 0',
      fontsize=10, loc='left', pad=6)

  ax = ax_at(0, 1)
  base._plot_eval_smoothed(
      ax, _series(ev, 'iteration'), _series(ev, 'success'),
      color=C_EVAL, label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})')
  ax.set_ylim(-0.05, 1.05)
  _style(ax, 'eval success (E=5, rolling mean w=5)')
  ax.legend(fontsize=7, frameon=False, loc='upper right')

  # --- row 1: NF loss, param grads ---
  ax = ax_at(1, 0)
  ax.plot(it, _series(d, 'nf/density_loss'), color=C_NLL, lw=1.3, label='NLL')
  ax.plot(it, _series(d, 'nf/nf_total_loss'), color=C_TOTAL, lw=1.1, ls='--',
          label='NLL + λ·hinge')
  ax.axhline(0.0, color='0.5', lw=0.6, ls=':')
  _style(ax, 'NF loss')
  ax.legend(fontsize=7, frameon=False, loc='upper right')

  ax = ax_at(1, 1)
  ax.plot(it, _series(d, 'nf/flow_grad_norm'), color=C_FLOW, lw=1.3,
          label='flow')
  ax.plot(it, _series(d, 'sa/encoder_grad_norm'), color=C_ENC, lw=1.2,
          label='sa encoder')
  _style(ax, 'param grad norm')
  ax.legend(fontsize=7, frameon=False, loc='upper right')

  # --- row 2: policy scale, entropy / KL ---
  ax = ax_at(2, 0)
  ax.plot(it, _series(d, 'ppo/policy_scale_mean'), color=C_SCALE, lw=1.3,
          label='mean')
  ax.plot(it, _series(d, 'ppo/policy_scale_min'), color=C_SMIN, lw=1.2,
          label='min')
  _style(ax, 'policy scale (pre-tanh)')
  ax.legend(fontsize=7, frameon=False, loc='upper right')

  ax = ax_at(2, 1)
  ax.plot(it, _series(d, 'ppo/entropy'), color=C_ENT, lw=1.3, label='entropy')
  ax.set_ylabel('entropy', fontsize=8, color=C_ENT)
  ax.tick_params(axis='y', labelcolor=C_ENT, labelsize=7)
  ax2 = ax.twinx()
  ax2.plot(it, _series(d, 'ppo/approx_kl'), color=C_KL, lw=1.1, label='approx KL')
  ax2.plot(it, _series(d, 'ppo/clipfrac'), color=C_CLIP, lw=1.0, alpha=0.9,
           label='clipfrac')
  ax2.set_ylabel('approx KL / clipfrac', fontsize=8, color=C_KL)
  ax2.tick_params(axis='y', labelcolor=C_KL, labelsize=7)
  ax2.spines['top'].set_visible(False)
  ax.tick_params(labelsize=7)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper right')

  # --- row 3: ∇_s log p, λ ---
  ax = ax_at(3, 0)
  ax.plot(it, _series(d, 'nf/nf_logp_grad_s_norm_mean'), color=C_GNORM, lw=1.3,
          label='mean ‖∇_s log p‖')
  ax.axhline(GRAD_REG_C, color=C_GNORM, lw=0.7, ls=':', alpha=0.8,
             label=f'c = {GRAD_REG_C:g}')
  ax.set_ylabel('‖∇_s log p‖ mean', fontsize=8, color=C_GNORM)
  ax.tick_params(axis='y', labelcolor=C_GNORM, labelsize=7)
  ax2 = ax.twinx()
  ax2.plot(it, _series(d, 'nf/nf_logp_grad_s_frac_above_c'), color=C_FRAC,
           lw=1.1, label='frac > c')
  ax2.set_ylabel('frac above c', fontsize=8, color=C_FRAC)
  ax2.tick_params(axis='y', labelcolor=C_FRAC, labelsize=7)
  ax2.set_ylim(-0.05, 1.05)
  ax2.spines['top'].set_visible(False)
  ax.tick_params(labelsize=7)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc='upper left')

  ax = ax_at(3, 1)
  ax.plot(it, _series(d, 'nf/nf_grad_reg_lam'), color=C_LAM, lw=1.3)
  ax.set_yscale('log')
  ax.axhline(1e-8, color='0.5', lw=0.6, ls=':')
  _style(ax, 'grad-reg λ  (log)')

  # --- row 4: on-policy NF reward, return-norm std ---
  ax = ax_at(4, 0)
  ax.plot(it, _series(d, 'reward_repr_raw_mean'), color=C_RAW, lw=1.3)
  ax.axhline(0.0, color='0.5', lw=0.6, ls=':')
  _style(ax, 'raw NF log p  (on-policy)')

  ax = ax_at(4, 1)
  ax.plot(it, _series(d, 'reward_return_norm_std'), color=C_RETSTD, lw=1.3)
  _style(ax, 'return-norm std  σ_R')

  # --- row 5: normalized task goal magnitude ---
  ax = ax_at(5, 0)
  ax.plot(it, norm_l2, color='#4C9BE8', lw=1.4, label='‖norm goal‖₂')
  ax.plot(it, norm_dmax, color='#E84C6F', lw=1.2, ls='--',
          label='max |norm goal dim|')
  ax.axhline(3.0, color='0.5', lw=0.6, ls=':', alpha=0.8)
  ax.text(it[-1] * 0.02, 3.1, '3σ', fontsize=7, color='0.4')
  _style(ax, '(task_goal − μ_goal) / σ_goal', xlabel=True)
  ax.legend(fontsize=7, frameon=False, loc='upper right')

  ax = ax_at(5, 1)
  cmap = plt.get_cmap('tab20', GOAL_DIMS)
  # find which dims are worst late in training (last 10% of iters)
  late_start = max(0, int(0.9 * len(it)))
  late_max = np.nanmean(np.abs(norm_per_dim[late_start:]), axis=0)
  order = np.argsort(late_max)[::-1]
  for rank, di in enumerate(order):
    vals = norm_per_dim[:, di]
    lw = 1.5 if rank < 3 else 0.7
    alpha = 0.9 if rank < 3 else 0.35
    label = f'dim {di}' if rank < 5 else None
    ax.plot(it, np.abs(vals), color=cmap(di), lw=lw, alpha=alpha, label=label)
  ax.axhline(3.0, color='0.5', lw=0.6, ls=':', alpha=0.8)
  _style(ax, '|norm goal| per dim', xlabel=True)
  ax.legend(fontsize=6, frameon=False, loc='upper right', ncol=2)

  fig.legend(
      handles=[Patch(facecolor=C_SUCC, alpha=0.25,
                     label='train-success locks  (390–480, 960–1230)')],
      loc='upper right', fontsize=8, frameon=False,
      bbox_to_anchor=(0.97, 0.995))

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=160)
  plt.close(fig)
  print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
  main()
