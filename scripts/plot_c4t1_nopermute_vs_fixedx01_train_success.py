#!/usr/bin/env python3
"""Compare c4t1 nopermute vs fixedx01 (8 runs) train_success_1000.

Shared knobs: actorreset (BB default) + nopermute + catselect + minstd1e-4 +
extrew1; methods CRL / NF / TD3 / TD-InfoNCE; ± fixed_start_x=0.1.

Writes:
  figs/builderbench/active_train_eval/creative4_task1_nopermute_vs_fixedx01_train_success1000.png

Usage:
  python scripts/plot_c4t1_nopermute_vs_fixedx01_train_success.py
  python scripts/plot_c4t1_nopermute_vs_fixedx01_train_success.py --watch 120
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'scripts'))

import plot_builderbench_train_success1000 as base  # noqa: E402

OUT_NAME = 'creative4_task1_nopermute_vs_fixedx01_train_success1000.png'

# (log_dir_name, legend_label)
RUNS = [
    # baseline: nopermute only (x still sampled in [0.05, 0.10])
    ('ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_actorreset_nopermute_catselect_minstd1e4_extrew1',
     'CRL nopermute'),
    ('ppo_builderbench_creative4_task1_e1024_pd_nf_tau05_actorreset_nopermute_catselect_minstd1e4_extrew1',
     'NF nopermute'),
    ('ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset_nopermute_catselect_minstd1e4_extrew1',
     'TD3 nopermute'),
    ('ppo_builderbench_creative4_task1_e1024_pd_tdinfonce_tau05_actorreset_nopermute_catselect_minstd1e4_extrew1',
     'TD-InfoNCE nopermute'),
    # fixedx01: nopermute + fixed_start_x=0.1
    ('ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_actorreset_nopermute_catselect_minstd1e4_fixedx01_extrew1',
     'CRL fixedx01'),
    ('ppo_builderbench_creative4_task1_e1024_pd_nf_tau05_actorreset_nopermute_catselect_minstd1e4_fixedx01_extrew1',
     'NF fixedx01'),
    ('ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset_nopermute_catselect_minstd1e4_fixedx01_extrew1',
     'TD3 fixedx01'),
    ('ppo_builderbench_creative4_task1_e1024_pd_tdinfonce_tau05_actorreset_nopermute_catselect_minstd1e4_fixedx01_extrew1',
     'TD-InfoNCE fixedx01'),
]


def plot_once(figs_dir: str = base.FIGS_DIR,
              log_root: str = base.LOG_ROOT,
              slurm_dir: str = base.SLURM_DIR) -> str | None:
  os.makedirs(figs_dir, exist_ok=True)
  # Reuse group plotter but with a custom out suffix / title via a thin wrapper.
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker

  fig, ax = plt.subplots(figsize=(10.5, 5.2))
  plotted = 0
  for i, (log_dir_name, label) in enumerate(RUNS):
    pts = base._subsample(base._read_train_series(
        log_root, log_dir_name, slurm_dir=slurm_dir))
    if not pts:
      print(f'  {label}: no train data yet')
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    # dashed = fixedx01, solid = nopermute
    ls = '--' if 'fixedx01' in label else '-'
    ax.plot(
        xs, ys, color=color, linewidth=2.2, linestyle=ls,
        label=f'{label} (n={len(pts)})', alpha=0.95, zorder=3 + i)
    print(f'  {label}: {len(pts)} pts  '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(ys):.3f}..{max(ys):.3f}]')
    plotted += 1

  if plotted == 0:
    ax.text(0.5, 0.5, 'waiting for train_success_1000…',
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    print('  (no series yet; writing placeholder figure)')

  ax.set_title(
      'Creative-4 Task1 — train success_1000\n'
      'nopermute vs fixedx01  (catselect, minstd1e-4, extrew1)',
      fontsize=12, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel('Train Success (last 1000)', fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if plotted > 0:
    ax.legend(loc='best', fontsize=8, framealpha=0.95, ncol=2)

  out_path = os.path.join(figs_dir, OUT_NAME)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}')
  return out_path


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--figs_dir', default=base.FIGS_DIR)
  p.add_argument('--log_root', default=base.LOG_ROOT)
  p.add_argument('--slurm_dir', default=base.SLURM_DIR)
  p.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds on the login node (CPU-only).')
  args = p.parse_args()
  if args.watch > 0:
    print(f'[watch] refresh every {args.watch}s → {args.figs_dir}/{OUT_NAME}')
    while True:
      plot_once(args.figs_dir, args.log_root, args.slurm_dir)
      time.sleep(args.watch)
  else:
    plot_once(args.figs_dir, args.log_root, args.slurm_dir)


if __name__ == '__main__':
  main()
