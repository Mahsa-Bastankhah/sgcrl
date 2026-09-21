#!/usr/bin/env python3
"""Plot DISCOVER UCB goals logged as ucb_goals[:8] on the Allegro throw run."""
from __future__ import annotations

import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(
    REPO, 'slurm',
    'newbl_discover_nvidiainit_z055_ep150_3911122.log')
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'discover_nvidiainit_ucb_goals.png')
BUCKET = np.array([0.50, -0.30, 0.55], dtype=np.float32)
GOAL_RE = re.compile(r'\[(-?\d+\.\d+),(-?\d+\.\d+),(-?\d+\.\d+)\]')


def _load(path: str) -> np.ndarray:
  goals = []
  with open(path) as f:
    for line in f:
      if 'ucb_goals[:8]=' not in line:
        continue
      for m in GOAL_RE.finditer(line):
        goals.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
  if not goals:
    raise SystemExit(f'no ucb_goals in {path}')
  return np.asarray(goals, dtype=np.float32)


def main() -> None:
  g = _load(LOG)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
  ax = axes[0]
  ax.scatter(g[:, 0], g[:, 1], s=6, alpha=0.12, c='#4C9BE8', linewidths=0,
             label='UCB pick (log :8)')
  ax.scatter([0.0], [0.0], s=70, c='#1B9E4B', marker='x', linewidths=2,
             label='table spawn (0, 0)')
  ax.scatter([BUCKET[0]], [BUCKET[1]], s=90, c='#D62728', marker='*',
             label='bucket (0.50, −0.30)')
  # NVIDIA table_narrow footprint, top-down.
  ax.add_patch(plt.Rectangle((-0.2375, -0.20), 0.475, 0.40, fill=False,
                             ls='--', lw=1.0, ec='#666666',
                             label='table_narrow'))
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlabel('x (m)')
  ax.set_ylabel('y (m)')
  ax.set_title('selected UCB object-xyz (xy)')
  ax.legend(loc='upper right', fontsize=8, frameon=False)
  ax.set_xlim(-0.85, 0.70)
  ax.set_ylim(-0.55, 0.70)

  ax = axes[1]
  ax.hist(g[:, 2], bins=40, color='#4C9BE8', alpha=0.85)
  ax.axvline(0.55, color='#1B9E4B', ls='--', lw=1.2, label='z=0.55 (table / rim)')
  ax.axvline(BUCKET[2], color='#D62728', ls=':', lw=1.2, label='bucket goal z')
  ax.set_xlabel('z (m)')
  ax.set_ylabel('count')
  ax.set_title('selected UCB z')
  ax.legend(fontsize=8, frameon=False)
  fig.suptitle(
      f'DISCOVER NVIDIA-init  ·  {len(g)} logged UCB goals  ·  '
      f'{100.0 * np.mean((np.abs(g[:, 0]) < 0.05) & (np.abs(g[:, 1]) < 0.05)):.0f}%'
      f' within 5 cm of table origin',
      fontsize=11)
  fig.tight_layout()
  fig.savefig(OUT, dpi=140)
  print(f'wrote {OUT} n={len(g)}', flush=True)


if __name__ == '__main__':
  sys.exit(main() or 0)
