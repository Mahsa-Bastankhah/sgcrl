#!/usr/bin/env python3
"""Where ``ppo_target_kl`` fired on the finished c8t2 tkl=0.05 run (3748425).

Parses ``[ppo] target_kl early-stop`` lines. x = env steps; y = PPO epoch +
minibatch fraction (10 epochs × 4 minibatches). Color = approx_kl at trip.

  python scripts/plot_c8t2_tkl005_kl_triggers.py
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'slurm',
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
    '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
    '_minstd1e5_ent005_to001_ep60_300m_crl10_dualgradreg_c100'
    '_lamlr1e6_tkl005_3748425_0.log',
)
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'figs', 'builderbench', 'c8t2_tkl005_kl_triggers.png',
)
SPI = 1024 * 60  # num_envs * T
N_EPOCHS = 10
N_MB = 4
THRESH = 0.05
PAT = re.compile(
    r'target_kl early-stop iter=(\d+) epoch=(\d+)/(\d+) mb=(\d+)/(\d+) '
    r'threshold=([0-9.eE+-]+) approx_kl=([0-9.eE+-]+)'
)


def _load(path: str):
  xs, ys, kls, epochs, mbs = [], [], [], [], []
  with open(path) as fh:
    for line in fh:
      m = PAT.search(line)
      if not m:
        continue
      it, ep, _ne, mb, _nm, _th, kl = m.groups()
      it = int(it)
      ep = int(ep)
      mb = int(mb)
      kl = float(kl)
      xs.append(it * SPI)
      ys.append(ep + (mb + 0.5) / N_MB)
      kls.append(kl)
      epochs.append(ep)
      mbs.append(mb)
  return xs, ys, kls, epochs, mbs


def main() -> None:
  xs, ys, kls, epochs, mbs = _load(LOG)
  if not xs:
    raise SystemExit(f'no early-stop lines in {LOG}')
  print(f'{len(xs)} trips  steps {xs[0]/1e6:.1f}M–{xs[-1]/1e6:.1f}M  '
        f'kl min={min(kls):.3f} med={sorted(kls)[len(kls)//2]:.3f} '
        f'max={max(kls):.3f}')

  fig, axes = plt.subplots(
      2, 1, figsize=(11.2, 7.2), sharex=False,
      gridspec_kw={'height_ratios': [2.4, 1.15], 'hspace': 0.32})
  ax, ax_kl = axes
  cmap = plt.cm.plasma
  vmax = max(0.25, sorted(kls)[int(0.95 * (len(kls) - 1))])
  norm = Normalize(vmin=THRESH, vmax=vmax)
  sc = ax.scatter(
      xs, ys, c=kls, cmap=cmap, norm=norm, s=28, alpha=0.88,
      edgecolors='none', zorder=3)
  for e in range(N_EPOCHS + 1):
    ax.axhline(e, color='#dddddd', linewidth=0.6, zorder=1)
  ax.set_ylim(-0.15, N_EPOCHS)
  ax.set_yticks([i + 0.5 for i in range(N_EPOCHS)])
  ax.set_yticklabels([str(i) for i in range(N_EPOCHS)])
  ax.set_ylabel('PPO epoch at stop\n(dot = epoch + minibatch/4)', fontsize=10)
  ax.set_title(
      'c8t2 3748425  ·  compact dualgradreg  ·  ep=T=60  ·  '
      r'target KL $=0.05$  ·  keep minibatch, stop rest',
      fontsize=11, fontweight='bold')
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.spines[['top', 'right']].set_visible(False)
  cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.035)
  cbar.set_label(r'approx KL at trip', fontsize=9)

  ax_kl.axhline(THRESH, color='#333333', linestyle='--', linewidth=1.0,
                label=f'threshold {THRESH:g}', zorder=2)
  ax_kl.scatter(
      xs, kls, c=kls, cmap=cmap, norm=norm, s=22, alpha=0.88,
      edgecolors='none', zorder=3)
  ax_kl.set_ylabel(r'approx KL at trip', fontsize=10)
  ax_kl.set_xlabel('Env Steps', fontsize=10)
  ax_kl.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax_kl.set_ylim(0.0, max(1.25, max(kls) * 1.05))
  ax_kl.spines[['top', 'right']].set_visible(False)
  ax_kl.legend(loc='upper right', fontsize=8, framealpha=0.95)
  ax_kl.grid(axis='y', linestyle='--', alpha=0.35)

  n_ep = [epochs.count(i) for i in range(N_EPOCHS)]
  n_mb = [mbs.count(i) for i in range(N_MB)]
  bits = '  '.join(f'ep{i}={n}' for i, n in enumerate(n_ep) if n)
  bits2 = '  '.join(f'mb{i}={n}' for i, n in enumerate(n_mb))
  fig.text(
      0.5, 0.01,
      f'{len(xs)} trips   {bits}   |   {bits2}',
      ha='center', fontsize=8.5, color='#333333')

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
