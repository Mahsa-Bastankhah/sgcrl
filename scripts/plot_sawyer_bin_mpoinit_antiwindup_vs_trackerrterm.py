#!/usr/bin/env python3
"""Eval success only for the running Sawyer-bin no-grad-reg MPO-init jobs.

Reads logs/eval/logs.csv. Mean ±1 SE across seeds. Faint raw mean + bold
centered rolling mean, window=5 (EVAL_SMOOTH_WINDOW).

  python scripts/plot_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.py
  python scripts/plot_sawyer_bin_mpoinit_antiwindup_vs_trackerrterm.py --watch 900
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_sawyer_bin_dgr_valuedgr_success as sawyer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    REPO, 'figs', 'metaworld', 'final_metaworld_runs',
    'sawyer_bin_mpoinit_selectiveantiwindup_vs_trackerrterm20cm.png')

# (kind, cfg, tag, ls, z, color, seeds)
CONFIGS = (
    ('ppo', sawyer.MPOINIT_NOGRADREG_DIR,
     'NF MPO XY±5cm no grad-reg', (0, (4, 1.5)), 7, '#332288', (0, 1)),
    ('ppo', sawyer.MPOINIT_NOGRADREG_EP50_DIR,
     'NF MPO XY±5cm no grad-reg ep50', (0, (6, 1.5)), 8, '#88CCEE', (0,)),
    ('ppo', sawyer.MPOINIT_NOGRADREG_EP50_ENT05_DIR,
     'NF MPO XY±5cm no grad-reg ep50 ent0.05', (0, (5, 1.5)), 9,
     '#E69F00', (0,)),
    ('ppo', sawyer.MPOINIT_NOGRADREG_EP100_ENT05_G095_DIR,
     'NF MPO XY±5cm no grad-reg ep100 ent0.05 γ0.95', (0, (3, 1.5)), 10,
     '#009E73', (0,)),
)

def run_once() -> str | None:
  w = base.EVAL_SMOOTH_WINDOW
  fig, ax = plt.subplots(figsize=(8.6, 4.4))
  n_plotted = 0
  last_bits = []

  for kind, cfg, tag, ls, z, color, seeds in CONFIGS:
    seed_xy, missing = sawyer._collect_seeds(kind, cfg, seeds)
    for seed in missing:
      print(f'  {tag} seed {seed}: no eval data yet')
    if not seed_xy:
      continue
    xs, mean, se, n = sawyer._mean_se_at_grid(seed_xy)
    sm = sawyer._plot_mean_eval(
        ax, xs, mean, se, color=color, ls=ls, z=z, w=w, n=n,
        label=f'{tag}  (n={n})')
    n_plotted += 1
    last = sm[-1] if sm else mean[-1]
    print(f'  {tag}: n={n} last_smooth={last:.3f} raw={mean[-1]:.3f} '
          f'@ {xs[-1]/1e6:.2f}M')
    last_bits.append(
        f'{tag} n={n} last={last:.3f} raw={mean[-1]:.3f} @ {xs[-1]/1e6:.2f}M')

  fig.suptitle(
      'Sawyer bin  ·  running no-grad-reg  ·  eval success',
      fontsize=11, fontweight='bold', y=0.995)
  ax.set_ylabel(f'Eval success_1000 (roll mean w={w})', fontsize=10)
  ax.set_xlabel('Env steps', fontsize=10)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  if not n_plotted:
    ax.text(
        0.5, 0.5, 'no eval data yet', transform=ax.transAxes,
        ha='center', va='center', fontsize=11, color='#888888')

  if n_plotted:
    handles, labels = ax.get_legend_handles_labels()
    ncol = 2 if n_plotted >= 2 else 1
    fig.legend(
        handles, labels, loc='upper center', ncol=ncol, fontsize=8,
        frameon=True, framealpha=0.95, fancybox=False, edgecolor='#333333',
        bbox_to_anchor=(0.5, 0.93), columnspacing=1.4, handlelength=2.4)

  fig.text(
      0.5, 0.01,
      'MPO-safe init, no density/value grad-reg. Shade = ±1 SE. '
      f'Faint raw + bold centered rolling mean, window={w}.',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.16)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  stamp = os.path.join(os.path.dirname(OUT), 'LAST_UPDATE_antiwindup_vs_trackerrterm.txt')
  with open(stamp, 'w', encoding='utf-8') as fh:
    fh.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    fh.write(OUT + '\n')
    for line in last_bits:
      fh.write(line + '\n')
    if n_plotted == 0:
      fh.write('no eval data yet\n')
  if n_plotted == 0:
    print(f'no eval data yet → {OUT} (axes only)')
  else:
    print(f'→ {OUT}')
  return OUT


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only login-node watcher).')
  args = p.parse_args()
  if args.watch <= 0:
    run_once()
    return
  print(
      f'[antiwindup_vs_trackerr] watching every {args.watch}s → {OUT}',
      flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001 — keep watcher alive
      print(f'[antiwindup_vs_trackerr] error: {exc}', flush=True)
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
