#!/usr/bin/env python3
"""c8t2 next-step time-reg 2h probes vs the no-time-reg dualgradreg clone.

Two figures (not a shared axis):
  1. NF NLL  (nf/density_loss)
  2. time-reg already x eta  (nf/nf_time_reg)

  python scripts/plot_c8t2_nf_timereg_next_2h.py
"""
from __future__ import annotations

import csv
import math
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = (
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
    '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp'
    '_extrew1_minstd1e5_ent005_to001_ep60_300m_crl10'
    '_dualgradreg_c100_lamlr1e6')
RUN = 'ppo_builderbench_creative_8_task2_0'
OUT_DIR = os.path.join(REPO, 'figs', 'builderbench', 'c8t2')

RUNS = {
    'base': {
        'label': 'no time-reg (clone)',
        'dir': PREFIX,
        'color': '#6B8E9F',
        'ls': '--',
        'lw': 1.2,
    },
    'iter': {
        'label': r'next-step iter  $\eta=1$  (3749613)',
        'dir': PREFIX + '_timereg_next_iter_eta1_2h',
        'color': '#C45C26',
        'ls': '-',
        'lw': 1.6,
    },
    'ema': {
        'label': r'next-step ema  $\eta=0.1$  (3749614)',
        'dir': PREFIX + '_timereg_next_ema_eta01_2h',
        'color': '#2A9D8F',
        'ls': '-',
        'lw': 1.6,
    },
}


def _csv(log_dir: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, RUN, 'logs', 'learner', 'logs.csv')


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


def _clip_to(d: dict, it_max: float) -> dict:
  it = d.get('iteration') or []
  keep = [i for i, x in enumerate(it) if x <= it_max + 0.5]
  if not keep:
    return d
  return {k: [v[i] for i in keep] for k, v in d.items()}


def _style(ax, ylabel: str, title: str):
  ax.set_title(title, fontsize=11, loc='left', pad=8)
  ax.set_xlabel('iteration', fontsize=9)
  ax.set_ylabel(ylabel, fontsize=9)
  ax.tick_params(labelsize=8)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.legend(fontsize=8, frameon=False, loc='best')


def _save(fig, name: str) -> str:
  os.makedirs(OUT_DIR, exist_ok=True)
  path = os.path.join(OUT_DIR, name)
  fig.savefig(path, dpi=140)
  plt.close(fig)
  print(f'wrote {path}')
  return path


def main() -> None:
  data = {}
  for key, spec in RUNS.items():
    path = _csv(spec['dir'])
    if not os.path.isfile(path):
      raise SystemExit(f'missing {path}')
    data[key] = _load(path)

  it_max = max(data['iter']['iteration'][-1], data['ema']['iteration'][-1])
  data['base'] = _clip_to(data['base'], it_max)

  fig, ax = plt.subplots(figsize=(8.2, 4.6))
  for key in ('base', 'iter', 'ema'):
    spec = RUNS[key]
    d = data[key]
    ax.plot(
        d['iteration'], d['nf/density_loss'],
        color=spec['color'], ls=spec['ls'], lw=spec['lw'],
        label=spec['label'])
  ax.set_xlim(0, it_max)
  _style(
      ax, 'NF NLL  (density_loss)',
      'c8t2  ·  NF NLL  ·  next-step time-reg 2h vs clone')
  fig.tight_layout()
  _save(fig, 'c8t2_nf_timereg_next_2h_nll.png')

  fig, ax = plt.subplots(figsize=(8.2, 4.6))
  for key in ('iter', 'ema'):
    spec = RUNS[key]
    d = data[key]
    ax.plot(
        d['iteration'], d['nf/nf_time_reg'],
        color=spec['color'], ls=spec['ls'], lw=spec['lw'],
        label=spec['label'])
  ax.set_xlim(0, it_max)
  _style(
      ax, r'time-reg  $\eta\,\mathbb{E}[(\Delta\log p)^2]$',
      r'c8t2  ·  time-reg $\times\eta$  ·  next-step 2h probes')
  fig.tight_layout()
  _save(fig, 'c8t2_nf_timereg_next_2h_reg.png')

  print('\n--- last-20-iter means ---')
  for key in ('base', 'iter', 'ema'):
    d = data[key]
    nll = d['nf/density_loss'][-20:]
    nll_m = sum(nll) / len(nll)
    tr = d.get('nf/nf_time_reg')
    if tr:
      tail = [x for x in tr[-20:] if x == x and math.isfinite(x)]
      tr_m = sum(tail) / len(tail) if tail else float('nan')
    else:
      tr_m = float('nan')
    print(
        f"{key:5s}  n={len(d['iteration']):4d}  last_iter={d['iteration'][-1]:.0f}"
        f"  NLL={nll_m:.3f}  time_reg={tr_m:.4g}")


if __name__ == '__main__':
  main()
