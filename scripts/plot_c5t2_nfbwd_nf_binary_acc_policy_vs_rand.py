#!/usr/bin/env python3
"""Compare NF future-goal rank acc: checkpoint policy vs random actions.

Merges pre-lock (0/100/200) CSVs with the earlier 300/400/1000/1400 probes.

  python scripts/plot_c5t2_nfbwd_nf_binary_acc_policy_vs_rand.py
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
        FIG, 'c5t2_nfbwd_nf_binary_acc_prepost_stoch',
        'c5t2_nfbwd_s0_stoch_ckpts300_400_1000_1400.csv'),
    os.path.join(
        FIG, 'c5t2_nfbwd_nf_binary_acc_prepost_rand',
        'c5t2_nfbwd_s0_rand_ckpts300_400_1000_1400.csv'),
    os.path.join(
        FIG, 'c5t2_nfbwd_nf_binary_acc_prelock_vs_post',
        'c5t2_nfbwd_s0_stoch_ckpts0_100_200.csv'),
    os.path.join(
        FIG, 'c5t2_nfbwd_nf_binary_acc_prelock_vs_post',
        'c5t2_nfbwd_s0_rand_ckpts0_100_200.csv'),
)
OUT = os.path.join(
    FIG, 'c5t2_nfbwd_nf_binary_acc_prelock_vs_post',
    'c5t2_nfbwd_s0_policy_vs_rand.png')
FIRST_LOCK = 260
C_POL = '#4C9BE8'
C_RAND = '#6C757D'
C_AFTER = '#E07A5F'


def _load() -> list[dict]:
  rows = []
  for path in CSVS:
    if not os.path.isfile(path):
      print(f'missing {path}')
      continue
    with open(path, newline='', encoding='utf-8') as fh:
      rows.extend(csv.DictReader(fh))
  return rows


def _row_policy(row: dict) -> str:
  value = row.get('policy', '')
  if value in ('checkpoint', 'random'):
    return value
  return 'checkpoint'


def _xy(rows: list[dict], policy: str, split: str):
  pts = {}
  for row in rows:
    if _row_policy(row) != policy:
      continue
    if row['split'] != split:
      continue
    n = int(float(row['num_samples']))
    if n <= 0:
      continue
    it = int(float(row['iteration']))
    pts[it] = (
        float(row['rank_accuracy']),
        float(row['rank_accuracy_se']),
        n,
    )
  if not pts:
    return np.array([]), np.array([]), np.array([])
  xs = np.array(sorted(pts))
  acc = np.array([pts[int(x)][0] for x in xs])
  se = np.array([pts[int(x)][1] for x in xs])
  return xs, acc, se


def main() -> None:
  rows = _load()
  # Infer policy for older CSVs that omitted the column.
  for row in rows:
    if 'policy' not in row or row['policy'] == '':
      src = row.get('nf_params_source', '')
      del src
      # stoch file path is not on the row; leave empty and handle in _xy.

  fig, ax = plt.subplots(figsize=(9.2, 4.8))
  ax.axvline(FIRST_LOCK, color='#2A9D8F', lw=1.1, ls=':', zorder=1)
  ax.annotate(
      f'first success lock\niter {FIRST_LOCK}',
      xy=(FIRST_LOCK, 0.98), xytext=(8, 0),
      textcoords='offset points', va='top', fontsize=8, color='#2A9D8F')

  series = (
      ('checkpoint', 'overall', C_POL, 'o', 'checkpoint policy (overall)'),
      ('checkpoint', 'after', C_AFTER, 's', 'checkpoint policy (after success)'),
      ('random', 'overall', C_RAND, 'D', 'random uniform (overall)'),
  )
  for policy, split, color, marker, label in series:
    xs, acc, se = _xy(rows, policy, split)
    if not len(xs):
      print(f'no points for {policy}/{split}')
      continue
    ax.errorbar(
        xs, acc, yerr=1.96 * se, color=color, marker=marker, lw=1.6,
        capsize=3, label=label, zorder=3)
    for x, y in zip(xs, acc):
      print(f'{label}: iter={int(x)} acc={y:.3f}')

  ax.axhline(0.5, color='0.45', lw=1.0, ls='--', label='chance = 50%')
  ax.set_ylim(0.0, 1.05)
  ax.set_xlabel('checkpoint iteration')
  ax.set_ylabel(r'$P[\log p(g^+) > \log p(g^-)]$')
  ax.set_title(
      'c5t2 nfbwd  ·  NF rank acc, policy data vs random',
      fontsize=11, fontweight='bold')
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.legend(fontsize=8, loc='lower right', framealpha=0.95)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.tight_layout()
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
