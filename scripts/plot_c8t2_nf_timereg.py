#!/usr/bin/env python3
"""c8t2 NF compact catwp + time-reg: is η·(Δlog p)² a sensible share of NLL?

Overlays the two live jobs (ema η=10, iter η=20) against the no-time-reg
dualgradreg clone. Train curves are raw. Eval success is a centered rolling
mean (window=5) with faint raw underneath.

  python scripts/plot_c8t2_nf_timereg.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = (
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
    '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp'
    '_extrew1_minstd1e5_ent005_to001_ep60_300m_crl10'
    '_dualgradreg_c100_lamlr1e6')
RUN = 'ppo_builderbench_creative_8_task2_0'

RUNS = {
    'ema': {
        'label': 'ema η=10  (3748478)',
        'dir': PREFIX + '_timereg_ema_eta10',
        'color': '#2A9D8F',
    },
    'iter': {
        'label': 'iter η=20  (3748477)',
        'dir': PREFIX + '_timereg_iter_eta20',
        'color': '#C45C26',
    },
    'base': {
        'label': 'no time-reg (clone)',
        'dir': PREFIX,
        'color': '#6B8E9F',
    },
}

OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c8t2',
    'c8t2_nf_timereg_ema_vs_iter.png')


def _csv(log_dir: str, split: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, RUN, 'logs', split, 'logs.csv')


def _load(path: str) -> dict:
  with open(path, newline='') as fh:
    rows = list(csv.DictReader(fh))
  if not rows:
    return {}
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
  return None


def _finite(xs):
  return [x for x in xs if x == x and math.isfinite(x)]


def _mean(xs) -> float:
  xs = _finite(xs)
  return sum(xs) / len(xs) if xs else float('nan')


def _style(ax, ylabel: str, xlabel: bool = False):
  ax.set_ylabel(ylabel, fontsize=8)
  ax.tick_params(labelsize=7)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  if xlabel:
    ax.set_xlabel('iteration', fontsize=8)


def _clip_to(d: dict, it_max: float) -> dict:
  it = d.get('iteration') or []
  keep = [i for i, x in enumerate(it) if x <= it_max + 0.5]
  if not keep:
    return d
  return {k: [v[i] for i in keep] for k, v in d.items()}


def main() -> None:
  data = {}
  evals = {}
  for key, spec in RUNS.items():
    lp = _csv(spec['dir'], 'learner')
    ep = _csv(spec['dir'], 'eval')
    if not os.path.isfile(lp):
      raise SystemExit(f'missing {lp}')
    data[key] = _load(lp)
    evals[key] = _load(ep) if os.path.isfile(ep) else {}

  it_max = max(
      data['ema']['iteration'][-1],
      data['iter']['iteration'][-1],
  )
  data['base'] = _clip_to(data['base'], it_max)
  if evals['base']:
    evals['base'] = _clip_to(evals['base'], it_max)

  fig = plt.figure(figsize=(11.6, 13.8))
  gs = GridSpec(
      5, 2, figure=fig, hspace=0.42, wspace=0.28,
      left=0.08, right=0.97, top=0.93, bottom=0.05)

  def ax_at(r, c):
    ax = fig.add_subplot(gs[r, c])
    ax.set_xlim(0, it_max)
    return ax

  # --- row 0: NLL vs ratio ---
  ax = ax_at(0, 0)
  for key in ('base', 'ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ls = '--' if key == 'base' else '-'
    lw = 1.15 if key == 'base' else 1.45
    ax.plot(d['iteration'], _series(d, 'nf/density_loss'),
            color=spec['color'], lw=lw, ls=ls, label=spec['label'])
  _style(ax, 'NF NLL  (density_loss)')
  ax.legend(fontsize=6.5, frameon=False, loc='upper right')
  ax.set_title(
      'c8t2 NF compact catwp  ·  time-reg vs no-time-reg clone',
      fontsize=10, loc='left', pad=6)

  ax = ax_at(0, 1)
  for key in ('ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ratio = _series(d, 'nf/nf_time_reg_nll_ratio')
    pct = [100.0 * x if x == x else float('nan') for x in ratio]
    ax.plot(d['iteration'], pct, color=spec['color'], lw=1.45,
            label=spec['label'])
  for y, lab, ls in (
      (1.0, '1%', ':'),
      (5.0, '5%', '--'),
      (10.0, '10%', '-.'),
  ):
    ax.axhline(y, color='0.55', lw=0.7, ls=ls)
    ax.text(it_max * 0.99, y, f' {lab}', va='bottom', ha='right',
            fontsize=6.5, color='0.4')
  _style(ax, r'time-reg / |NLL|  (%)')
  ax.legend(fontsize=6.5, frameon=False, loc='upper right')

  # --- row 1: weighted time-reg + raw (Δlog p)² ---
  ax = ax_at(1, 0)
  for key in ('ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ax.plot(d['iteration'], _series(d, 'nf/nf_time_reg'),
            color=spec['color'], lw=1.4, label=spec['label'])
  _style(ax, r'time-reg  $\eta\,\mathbb{E}[(\Delta\log p)^2]$')
  ax.legend(fontsize=6.5, frameon=False, loc='upper right')

  ax = ax_at(1, 1)
  for key in ('ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ax.plot(d['iteration'], _series(d, 'nf/nf_time_reg_raw'),
            color=spec['color'], lw=1.4, label=spec['label'])
  _style(ax, r'time-reg raw  $\mathbb{E}[(\log p-\log p_{\mathrm{old}})^2]$')
  ax.legend(fontsize=6.5, frameon=False, loc='upper right')

  # --- row 2: log p mean / max ---
  ax = ax_at(2, 0)
  for key in ('base', 'ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ls = '--' if key == 'base' else '-'
    lw = 1.15 if key == 'base' else 1.4
    ax.plot(d['iteration'], _series(d, 'nf/log_p_mean'),
            color=spec['color'], lw=lw, ls=ls, label=spec['label'])
  _style(ax, 'log p mean')
  ax.legend(fontsize=6.5, frameon=False, loc='lower right')

  ax = ax_at(2, 1)
  for key in ('base', 'ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ls = '--' if key == 'base' else '-'
    lw = 1.15 if key == 'base' else 1.4
    ax.plot(d['iteration'], _series(d, 'nf/log_p_max'),
            color=spec['color'], lw=lw, ls=ls, label=spec['label'])
  _style(ax, 'log p max')
  ax.legend(fontsize=6.5, frameon=False, loc='lower right')

  # --- row 3: param grads ---
  ax = ax_at(3, 0)
  for key in ('base', 'ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ls = '--' if key == 'base' else '-'
    lw = 1.15 if key == 'base' else 1.4
    ax.plot(d['iteration'], _series(d, 'nf/flow_grad_norm'),
            color=spec['color'], lw=lw, ls=ls, label=spec['label'])
  ax.set_yscale('log')
  _style(ax, 'flow param grad (log)')
  ax.legend(fontsize=6.5, frameon=False, loc='upper right')

  ax = ax_at(3, 1)
  for key in ('base', 'ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ls = '--' if key == 'base' else '-'
    lw = 1.15 if key == 'base' else 1.4
    ax.plot(d['iteration'], _series(d, 'sa/encoder_grad_norm'),
            color=spec['color'], lw=lw, ls=ls, label=spec['label'])
  ax.set_yscale('log')
  _style(ax, 'SA encoder grad (log)')
  ax.legend(fontsize=6.5, frameon=False, loc='upper right')

  # --- row 4: PPO health + success ---
  ax = ax_at(4, 0)
  for key in ('ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ax.plot(d['iteration'], _series(d, 'ppo/entropy'),
            color=spec['color'], lw=1.4, label=spec['label'] + ' ent')
  ax2 = ax.twinx()
  for key in ('ema', 'iter'):
    spec = RUNS[key]
    d = data[key]
    ax2.plot(d['iteration'], _series(d, 'ppo/approx_kl'),
             color=spec['color'], lw=1.05, ls=':', alpha=0.9)
  _style(ax, 'PPO entropy', xlabel=True)
  ax2.set_ylabel('approx KL (dotted)', fontsize=8)
  ax2.tick_params(labelsize=7)
  ax2.spines['top'].set_visible(False)
  handles = [
      Line2D([0], [0], color=RUNS['ema']['color'], lw=1.4, label='ema ent'),
      Line2D([0], [0], color=RUNS['iter']['color'], lw=1.4, label='iter ent'),
      Line2D([0], [0], color='0.35', lw=1.05, ls=':', label='approx KL'),
  ]
  ax.legend(handles=handles, fontsize=6.5, frameon=False, loc='upper right')

  ax = ax_at(4, 1)
  for key in ('ema', 'iter', 'base'):
    spec = RUNS[key]
    d = data[key]
    ls = '--' if key == 'base' else '-'
    ax.plot(d['iteration'], _series(d, 'train_success_1000'),
            color=spec['color'], lw=1.2, ls=ls, label=spec['label'])
  for key in ('ema', 'iter'):
    spec = RUNS[key]
    ev = evals[key]
    if ev and 'success' in ev:
      base._plot_eval_smoothed(
          ax, ev['iteration'], ev['success'],
          color=spec['color'], linestyle=':',
          label=spec['label'].split()[0] + ' eval (roll mean w=5)')
  ax.set_ylim(-0.05, 1.05)
  _style(ax, 'train success_1000 / eval', xlabel=True)
  ax.legend(fontsize=6.0, frameon=False, loc='upper right')

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=140)
  print(f'wrote {OUT_PATH}')

  print('\n--- last-20-iter means ---')
  for key in ('ema', 'iter', 'base'):
    d = data[key]
    nll = _series(d, 'nf/density_loss')
    tr = _series(d, 'nf/nf_time_reg')
    ratio = _series(d, 'nf/nf_time_reg_nll_ratio')
    raw = _series(d, 'nf/nf_time_reg_raw')
    print(
        f"{key:5s}  n={len(d['iteration']):4d}  last_iter={d['iteration'][-1]:.0f}"
        f"  NLL={_mean(nll[-20:]):.3f}"
        f"  time_reg={_mean(tr[-20:]) if tr else float('nan'):.4g}"
        f"  ratio%={100*_mean(ratio[-20:]) if ratio else float('nan'):.3f}"
        f"  raw={_mean(raw[-20:]) if raw else float('nan'):.4g}"
        f"  logp_mean={_mean(_series(d, 'nf/log_p_mean')[-20:]):.3f}")


if __name__ == '__main__':
  main()
