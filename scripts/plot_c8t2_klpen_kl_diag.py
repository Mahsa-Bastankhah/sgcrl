#!/usr/bin/env python3
"""c8t2 NF compact catwp + KL-penalty: β, KL, early-stop, actor-reset, success.

Single run (seed 0, job 3714601). Train success is raw. Eval success is a
centered rolling mean (window=5) with faint raw underneath.

  python scripts/plot_c8t2_klpen_kl_diag.py
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR_NAME = (
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
    '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp'
    '_extrew1_minstd1e5_entanneal_ep60_300m_crl10_klpen1_tkl005')
RUN_DIR = os.path.join(
    REPO, 'logs', LOG_DIR_NAME, 'ppo_builderbench_creative_8_task2_0')
LEARNER_CSV = os.path.join(RUN_DIR, 'logs', 'learner', 'logs.csv')
EVAL_CSV = os.path.join(RUN_DIR, 'logs', 'eval', 'logs.csv')
SLURM_GLOB = os.path.join(REPO, 'slurm', f'{LOG_DIR_NAME}_*.log')
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'active_train_eval',
    'creative8_task2_nf_compact_klpen1_tkl005_kl_diag.png')

TARGET_KL = 0.05
KL_PENALTY_TARGET = 0.02
BETA_FLOOR = 1e-16
ACTOR_RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')

C_TRAIN = '#2A9D8F'
C_EVAL = '#4C9BE8'
C_APPROX = '#9B2226'
C_APPROX_MAX = '#E84C6F'
C_ANALYTIC = '#E8834C'
C_BETA = '#A84CE8'
C_ES = '#C45C26'
C_RESET = '#4C9BE8'


def _load(path: str) -> dict:
  with open(path, newline='') as fh:
    rows = list(csv.DictReader(fh))
  if not rows:
    raise SystemExit(f'empty csv: {path}')
  out = {k: [] for k in rows[0]}
  for r in rows:
    for k, v in r.items():
      try:
        x = float(v)
        out[k].append(x if math.isfinite(x) else float('nan'))
      except (TypeError, ValueError):
        out[k].append(float('nan'))
  return out


def _series(d: dict, *names):
  for n in names:
    if n in d:
      return d[n]
  raise KeyError(names)


def _actor_reset_steps(slurm_path: str, iter_to_step: dict) -> list[float]:
  if not slurm_path or not os.path.isfile(slurm_path):
    return []
  steps = []
  with open(slurm_path) as fh:
    for line in fh:
      m = ACTOR_RESET_RE.search(line)
      if not m:
        continue
      it = int(m.group(1))
      if it in iter_to_step:
        steps.append(iter_to_step[it])
  return steps


def _vlines(ax, xs, *, color, alpha, lw, zorder=1):
  if not xs:
    return
  ax.vlines(xs, 0, 1, transform=ax.get_xaxis_transform(),
            colors=color, alpha=alpha, lw=lw, zorder=zorder)


def _style(ax, ylabel: str, xlabel: bool = False):
  ax.set_ylabel(ylabel, fontsize=9)
  ax.tick_params(labelsize=8)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  if xlabel:
    ax.set_xlabel('env steps', fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(base._fmt_steps))
  else:
    ax.tick_params(labelbottom=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(base._fmt_steps))


def main() -> None:
  d = _load(LEARNER_CSV)
  ev = _load(EVAL_CSV)
  steps = _series(d, 'global_step')
  it = _series(d, 'iteration')
  iter_to_step = {int(i): s for i, s in zip(it, steps)}

  es_epoch = _series(d, 'ppo/early_stop_epoch')
  es_steps = [s for s, e in zip(steps, es_epoch) if e >= 0]

  slurm_matches = sorted(glob.glob(SLURM_GLOB), key=os.path.getmtime,
                         reverse=True)
  slurm_path = slurm_matches[0] if slurm_matches else ''
  reset_steps = _actor_reset_steps(slurm_path, iter_to_step)

  eval_it = _series(ev, 'iteration')
  eval_succ = _series(ev, 'success')
  spi = float(steps[0]) if steps and it and it[0] == 0 else 1024 * 60
  eval_steps = [
      iter_to_step.get(int(i), (i + 1.0) * spi) for i in eval_it]

  beta = _series(d, 'ppo/kl_beta')
  beta_plot = [max(b, BETA_FLOOR) if math.isfinite(b) else float('nan')
               for b in beta]

  fig, axes = plt.subplots(
      3, 1, figsize=(11.6, 8.4), sharex=True,
      gridspec_kw={'height_ratios': [1.05, 1.15, 0.95], 'hspace': 0.08})
  ax_s, ax_k, ax_b = axes
  xmax = steps[-1] if steps else 1.0
  for ax in axes:
    ax.set_xlim(0, xmax)
    _vlines(ax, es_steps, color=C_ES, alpha=0.18, lw=0.7)
    _vlines(ax, reset_steps, color=C_RESET, alpha=0.7, lw=1.4)

  ax_s.plot(steps, _series(d, 'train_success_1000'), color=C_TRAIN, lw=1.5,
            label='train success (last 1000)')
  base._plot_eval_smoothed(
      ax_s, eval_steps, eval_succ, color=C_EVAL,
      label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})',
      linewidth=2.0)
  ax_s.set_ylim(-0.05, 1.05)
  _style(ax_s, 'success')
  ax_s.set_title(
      'c8t2 NF compact catwp+extrew1  ·  KL-pen β₀=1  target=0.02  ·  '
      'early-stop tkl=0.05  ·  seed 0',
      fontsize=10, loc='left', pad=8)

  ax_k.plot(steps, _series(d, 'ppo/approx_kl_max'), color=C_APPROX_MAX,
            lw=0.9, alpha=0.55, label='approx KL max')
  ax_k.plot(steps, _series(d, 'ppo/approx_kl'), color=C_APPROX, lw=1.35,
            label='approx KL (exact)')
  ax_k.plot(steps, _series(d, 'ppo/analytic_kl'), color=C_ANALYTIC, lw=1.25,
            label='analytic KL')
  ax_k.axhline(TARGET_KL, color=C_ES, lw=0.9, ls='--',
               label=f'target KL early-stop = {TARGET_KL:g}')
  ax_k.axhline(KL_PENALTY_TARGET, color=C_BETA, lw=0.8, ls=':',
               label=f'β-adapt target = {KL_PENALTY_TARGET:g}')
  ax_k.set_ylim(bottom=-0.005)
  _style(ax_k, 'KL')

  ax_b.plot(steps, beta_plot, color=C_BETA, lw=1.5, label='β (KL penalty)')
  ax_b.set_yscale('log')
  ax_b.set_ylim(BETA_FLOOR, 8.0)
  ax_b.axhline(1.0, color='0.5', lw=0.6, ls=':')
  zero_i = next((i for i, b in enumerate(beta) if b == 0.0), None)
  if zero_i is not None:
    zx = steps[zero_i]
    ax_b.annotate(
        f'β = 0 from {base._fmt_steps(zx, None)}  (iter {int(it[zero_i])})',
        xy=(zx, BETA_FLOOR * 2), xytext=(xmax * 0.12, 3e-4),
        fontsize=8, color=C_BETA,
        arrowprops=dict(arrowstyle='->', color=C_BETA, lw=0.8),
    )
  _style(ax_b, 'β  (log)', xlabel=True)

  extra = [
      Line2D([0], [0], color=C_ES, lw=1.4,
             label=f'early-stop ({len(es_steps)} iters)'),
      Line2D([0], [0], color=C_RESET, lw=1.6,
             label=f'actor reset ({len(reset_steps)})'),
  ]
  h, lab = [], []
  for ax in axes:
    hi, li = ax.get_legend_handles_labels()
    h.extend(hi)
    lab.extend(li)
  seen = set()
  h2, lab2 = [], []
  for handle, label in zip(h, lab):
    if label in seen:
      continue
    seen.add(label)
    h2.append(handle)
    lab2.append(label)
  ax_s.legend(h2 + extra, lab2 + [e.get_label() for e in extra],
              fontsize=7.5, frameon=False, loc='upper right', ncol=2)

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=160)
  plt.close(fig)
  print(f'wrote {OUT_PATH}')
  print(f'  iters={len(steps)}  early-stops={len(es_steps)}  '
        f'actor-resets={len(reset_steps)}  '
        f'slurm={os.path.basename(slurm_path) or "missing"}')


if __name__ == '__main__':
  main()
