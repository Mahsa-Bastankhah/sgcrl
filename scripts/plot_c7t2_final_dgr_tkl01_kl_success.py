#!/usr/bin/env python3
"""c7t2 final-run dgr+tKL=0.1+extrew: policy KL, KL fires, actor reset, success.

One column per seed. Train success is raw. Eval is faint raw + roll mean w=5.

  python scripts/plot_c7t2_final_dgr_tkl01_kl_success.py
"""
from __future__ import annotations

import csv
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
NAME = (
    'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
    '_minstd1e5_ent05to001_ep70_300m_crl10_dualgradreg_c100'
    '_lamlr1e6_tkl01_warp_4h'
)
LOG_DIR = os.path.join(REPO, 'logs', 'final_runs', NAME)
SLURM = {
    0: os.path.join(
        REPO, 'slurm', 'final_runs',
        'c7t2_dgr_tkl_extrew1_3765976_0.log'),
    1: os.path.join(
        REPO, 'slurm', 'final_runs',
        'c7t2_dgr_tkl_extrew1_3765976_1.log'),
}
OUT = os.path.join(
    REPO, 'figs', 'builderbench', 'final_runs',
    'c7t2_extrew1_dgr_tkl01_kl_success.png')

TARGET_KL = 0.1
ALT_KLS = (0.01, 0.02)
ALT_STYLES = {
    0.01: dict(color='#1D6F6A', ls=':', lw=1.15),
    0.02: dict(color='#6B4C9A', ls='-.', lw=1.15),
}
SPI = 1024 * 70
RESET_RE = re.compile(
    r'\[ppo\] actor reset at iter=(\d+):\s*(.*)')
FIRE_RE = re.compile(
    r'target_kl early-stop iter=(\d+) epoch=(\d+)/(\d+) mb=(\d+)/(\d+) '
    r'threshold=([0-9.eE+-]+) approx_kl=([0-9.eE+-]+)')

C_TRAIN = '#2A9D8F'
C_EVAL = '#4C9BE8'
C_APPROX = '#9B2226'
C_APPROX_MAX = '#E84C6F'
C_FIRE = '#C45C26'
C_RESET = '#4C4CE8'


def _load_csv(path: str) -> dict:
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


def _parse_slurm(path: str):
  resets = []
  fires = []
  with open(path) as fh:
    for line in fh:
      m = RESET_RE.search(line)
      if m:
        resets.append((int(m.group(1)), m.group(2).strip()))
        continue
      m = FIRE_RE.search(line)
      if m:
        fires.append((
            int(m.group(1)), int(m.group(2)), int(m.group(4)),
            float(m.group(7))))
  return resets, fires


def _vlines(ax, xs, *, color, alpha, lw, zorder=1):
  if not xs:
    return
  ax.vlines(
      xs, 0, 1, transform=ax.get_xaxis_transform(),
      colors=color, alpha=alpha, lw=lw, zorder=zorder)


def _style(ax, ylabel: str, xlabel: bool):
  ax.set_ylabel(ylabel, fontsize=9)
  ax.tick_params(labelsize=8)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.xaxis.set_major_formatter(plt.FuncFormatter(base._fmt_steps))
  if xlabel:
    ax.set_xlabel('env steps', fontsize=9)
  else:
    ax.tick_params(labelbottom=False)


def _plot_seed(ax_s, ax_k, seed: int) -> None:
  run = os.path.join(LOG_DIR, f'ppo_builderbench_creative_7_task2_{seed}')
  d = _load_csv(os.path.join(run, 'logs', 'learner', 'logs.csv'))
  ev = _load_csv(os.path.join(run, 'logs', 'eval', 'logs.csv'))
  steps = _series(d, 'global_step')
  it = _series(d, 'iteration')
  iter_to_step = {int(i): s for i, s in zip(it, steps)}

  resets, fires = _parse_slurm(SLURM[seed])
  reset_steps = [iter_to_step.get(i, (i + 1) * SPI) for i, _ in resets]
  fire_steps = [iter_to_step.get(i, (i + 1) * SPI) for i, *_ in fires]
  fire_kls = [kl for *_, kl in fires]

  eval_it = _series(ev, 'iteration')
  eval_succ = _series(ev, 'success')
  eval_steps = [
      iter_to_step.get(int(i), (i + 1.0) * SPI) for i in eval_it]

  xmax = steps[-1] if steps else 1.0
  for ax in (ax_s, ax_k):
    ax.set_xlim(0, xmax)
    _vlines(ax, fire_steps, color=C_FIRE, alpha=0.12, lw=0.6)
    _vlines(ax, reset_steps, color=C_RESET, alpha=0.85, lw=1.35)

  ax_s.plot(
      steps, _series(d, 'train_success_1000'), color=C_TRAIN, lw=1.45,
      label='train success (last 1000)', zorder=3)
  base._plot_eval_smoothed(
      ax_s, eval_steps, eval_succ, color=C_EVAL,
      label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})',
      linewidth=2.0)
  ax_s.set_ylim(-0.05, 1.05)
  _style(ax_s, 'success', xlabel=False)
  last_tr = _series(d, 'train_success_1000')[-1]
  ax_s.set_title(
      f'seed {seed}  ·  train last={last_tr:.3f}  ·  '
      f'KL fires={len(fires)}  ·  actor resets={len(resets)}',
      fontsize=10, loc='left', pad=6)

  ax_k.plot(
      steps, _series(d, 'ppo/approx_kl_max'), color=C_APPROX_MAX,
      lw=0.8, alpha=0.5, label='approx KL max (over mb)', zorder=2)
  ax_k.plot(
      steps, _series(d, 'ppo/approx_kl'), color=C_APPROX, lw=1.2,
      label='approx KL (mean)', zorder=3)
  ax_k.axhline(
      TARGET_KL, color=C_FIRE, lw=0.95, ls='--',
      label=f'target KL = {TARGET_KL:g} (used)', zorder=2)
  for alt in ALT_KLS:
    ax_k.axhline(
        alt, label=f'ref KL = {alt:g}', zorder=2, **ALT_STYLES[alt])
  if fire_steps:
    ax_k.scatter(
        fire_steps, fire_kls, s=14, c=C_FIRE, alpha=0.75,
        zorder=4, label='KL fire (approx_kl at trip)', edgecolors='none')
  ax_k.set_ylim(bottom=-0.01)
  _style(ax_k, 'policy KL', xlabel=True)

  kl_mean = _series(d, 'ppo/approx_kl')
  kl_max = _series(d, 'ppo/approx_kl_max')
  n = sum(1 for x in kl_max if math.isfinite(x))
  bits = []
  for thr in ALT_KLS + (TARGET_KL,):
    n_max = sum(1 for x in kl_max if math.isfinite(x) and x > thr)
    n_mean = sum(1 for x in kl_mean if math.isfinite(x) and x > thr)
    frac_max = 100.0 * n_max / max(n, 1)
    bits.append(f'{thr:g}: max>{thr:g} {n_max}/{n} ({frac_max:.0f}%)')
    print(f'  approx_kl_max > {thr:g}: {n_max}/{n} ({frac_max:.0f}%)  '
          f'mean > {thr:g}: {n_mean}/{n}')
  ax_k.text(
      0.01, 0.98, '  ·  '.join(bits[:2]),
      transform=ax_k.transAxes, va='top', ha='left', fontsize=7.0,
      color='#333333')

  print(f'seed {seed}: fires={len(fires)}  resets={len(resets)}')
  for it_r, why in resets:
    print(f'  reset iter={it_r}  ({(it_r + 1) * SPI / 1e6:.2f}M)  {why}')


def main() -> None:
  fig, axes = plt.subplots(
      2, 2, figsize=(13.2, 7.6), sharex=False,
      gridspec_kw={'hspace': 0.10, 'wspace': 0.16})
  for seed, col in ((0, 0), (1, 1)):
    _plot_seed(axes[0, col], axes[1, col], seed)

  extra = [
      Line2D([0], [0], color=C_FIRE, lw=1.4,
             label='KL early-stop fire'),
      Line2D([0], [0], color=C_RESET, lw=1.8,
             label='actor reset (last-layer)'),
  ]
  h, lab = axes[0, 0].get_legend_handles_labels()
  hk, lk = axes[1, 0].get_legend_handles_labels()
  seen, h2, lab2 = set(), [], []
  for handle, label in zip(h + hk + extra, lab + lk + [e.get_label() for e in extra]):
    if label in seen:
      continue
    seen.add(label)
    h2.append(handle)
    lab2.append(label)
  axes[0, 1].legend(
      h2, lab2, fontsize=7.4, frameon=False, loc='lower right', ncol=2)

  fig.suptitle(
      'c7t2 final  ·  compact-small  ·  dualgradreg c=100 λlr=1e-6  ·  '
      'tKL=0.1  ·  extrew',
      fontsize=11, fontweight='bold', y=0.98)
  fig.text(
      0.5, 0.01,
      'Orange dashed = used tKL=0.1. Teal dotted = 0.01, purple dash-dot = 0.02 '
      '(refs only; this run early-stopped at 0.1). '
      'Panel text = fraction of iters with approx_kl_max above each ref. '
      'Orange vlines = target_kl early-stop. Blue vlines = actor reset '
      '(last-layer when ep_length_mean < 56). Train raw; eval roll mean w=5. '
      'Source: logs/final_runs/ …_tkl01_warp_4h + slurm/final_runs/c7t2_dgr_tkl_extrew1_3765976_{0,1}.log',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
