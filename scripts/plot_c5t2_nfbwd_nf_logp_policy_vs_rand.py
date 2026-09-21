#!/usr/bin/env python3
"""E[log p(s_f|s,a)] on checkpoint-policy vs random rollouts, ckpts 100–500.

Uses ``positive_logp_mean`` from the binary-acc CSVs (same future-goal pairing).

  python scripts/plot_c5t2_nfbwd_nf_logp_policy_vs_rand.py
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(REPO, 'figs', 'builderbench')
CSVS = (
    os.path.join(
        FIG, 'c5t2_nfbwd_nf_binary_acc_ckpts100_200_300',
        'c5t2_nfbwd_s0_policy_ckpts100_200_300.csv'),
    os.path.join(
        FIG, 'c5t2_nfbwd_nf_binary_acc_ckpts100_200_300',
        'c5t2_nfbwd_s0_rand_ckpts100_200_300.csv'),
    os.path.join(
        FIG, 'c5t2_nfbwd_nf_binary_acc_ckpts400_500',
        'c5t2_nfbwd_s0_policy_ckpts400_500.csv'),
    os.path.join(
        FIG, 'c5t2_nfbwd_nf_binary_acc_ckpts400_500',
        'c5t2_nfbwd_s0_rand_ckpts400_500.csv'),
)
OUT = os.path.join(
    FIG, 'c5t2_nfbwd_nf_binary_acc_ckpts400_500',
    'c5t2_nfbwd_s0_logp_policy_vs_rand_100to500.png')
FIRST_LOCK = 260
KEEP = {100, 200, 300, 400, 500}


def _row_policy(row: dict) -> str:
  value = row.get('policy', '')
  if value in ('checkpoint', 'random'):
    return value
  return 'checkpoint'


def _load() -> dict[str, dict[int, float]]:
  out = {'checkpoint': {}, 'random': {}}
  for path in CSVS:
    if not os.path.isfile(path):
      print(f'missing {path}')
      continue
    with open(path, newline='', encoding='utf-8') as fh:
      for row in csv.DictReader(fh):
        if row['split'] != 'overall':
          continue
        it = int(float(row['iteration']))
        if it not in KEEP:
          continue
        if float(row['num_samples']) <= 0:
          continue
        out[_row_policy(row)][it] = float(row['positive_logp_mean'])
  return out


def main() -> None:
  data = _load()
  fig, ax = plt.subplots(figsize=(8.8, 4.8))
  ax.axvline(FIRST_LOCK, color='#2A9D8F', lw=1.1, ls=':', zorder=1)
  ax.annotate(
      f'first success lock\niter {FIRST_LOCK}',
      xy=(FIRST_LOCK, 0.02), xycoords=('data', 'axes fraction'),
      xytext=(8, 0), textcoords='offset points', fontsize=8, color='#2A9D8F')
  styles = (
      ('checkpoint', '#4C9BE8', 'o', r'checkpoint policy data'),
      ('random', '#6C757D', 'D', r'random uniform data'),
  )
  for key, color, marker, label in styles:
    pts = data[key]
    xs = np.array(sorted(pts))
    ys = np.array([pts[int(x)] for x in xs])
    if not len(xs):
      print(f'no points for {key}')
      continue
    ax.plot(xs, ys, color=color, marker=marker, lw=1.8, label=label)
    for x, y in zip(xs, ys):
      print(f'{label}: iter={int(x)} E[log p]={y:+.3f}')
  ax.axhline(0.0, color='0.7', lw=0.8, ls='--')
  ax.set_xlabel('checkpoint iteration')
  ax.set_ylabel(r'$\mathbb{E}[\log p_\theta(s_f\mid s,a)]$')
  ax.set_title(
      'c5t2 nfbwd  ·  future-state log p, policy vs random',
      fontsize=11, fontweight='bold')
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.legend(fontsize=8, loc='best', framealpha=0.95)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.tight_layout()
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
