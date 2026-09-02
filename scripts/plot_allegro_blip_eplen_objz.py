#!/usr/bin/env python3
"""Episode length, object z, and train success for Allegro blip runs.

Raw train curves (no smoothing). ``×`` is a last-layer actor-reset from
slurm ``[ppo] actor reset at iter=``, not every short-episode tick.
Object z is missing on the earlier palm jobs (logging was added later).
Success y-axis is zoomed so a 1/1000-episode tick is visible.

  python scripts/plot_allegro_blip_eplen_objz.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_allegro_blip_success as blip  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_blip_eplen_objz.png')
RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')
RESET_EP_LEN = 200.0
NOMINAL_EP = 300.0
TABLE_Z = 0.55
FALL_Z = 0.1

RUNS = blip.PALM + blip.TABLESIDE


def _curve(log_dir: str, col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, log_dir, split='learner',
      x_col='global_step', y_col=col)
  xs, mean, _se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return None
  return xs, mean, n


def _reset_steps(log_dir: str) -> list[int]:
  mapping = base._iter_to_env_steps_map(base.LOG_ROOT, log_dir)
  steps: list[int] = []
  for path in sorted(glob.glob(os.path.join(REPO, 'slurm', f'{log_dir}_*.log'))):
    with open(path, encoding='utf-8', errors='replace') as fh:
      for line in fh:
        m = RESET_RE.search(line)
        if not m:
          continue
        it = int(m.group(1))
        if it in mapping:
          steps.append(int(mapping[it]))
  return sorted(set(steps))


def _y_at(xs, ys, x):
  best_i = min(range(len(xs)), key=lambda i: abs(xs[i] - x))
  return ys[best_i]


def _mark(ax, xs, ys, resets, color):
  for rx in resets:
    ax.axvline(rx, color=color, lw=0.7, linestyle='--', alpha=0.35)
    ax.scatter([rx], [_y_at(xs, ys, rx)], color=color, marker='x',
               s=36, linewidths=1.2, zorder=5)


def main() -> None:
  n = len(RUNS)
  fig, axes = plt.subplots(n, 3, figsize=(16.2, 2.15 * n), sharex=True)
  if n == 1:
    axes = axes.reshape(1, 3)

  for row, (log_dir, label, color, ls) in enumerate(RUNS):
    ax_z, ax_ep, ax_s = axes[row]
    resets = _reset_steps(log_dir)
    z = _curve(log_dir, 'object_z_mean')
    ep = _curve(log_dir, 'ep_length_mean')
    succ = _curve(log_dir, 'train_success_1000')

    if z is None:
      ax_z.text(0.5, 0.5, 'object z not logged', transform=ax_z.transAxes,
                ha='center', va='center', fontsize=9, color='0.45')
      print(f'{label}: no object_z_mean  resets={len(resets)}')
    else:
      xs, mean, _n = z
      ax_z.plot(xs, mean, color=color, linestyle=ls, lw=1.5)
      _mark(ax_z, xs, mean, resets, color)
      print(f'{label}: object_z last={mean[-1]:.3f}  resets={len(resets)}')
    ax_z.axhline(TABLE_Z, color='0.55', lw=0.7, linestyle='--')
    ax_z.axhline(FALL_Z, color='0.55', lw=0.7, linestyle=':')
    ax_z.set_ylabel(r'$z$ (m)', fontsize=8)
    ax_z.set_ylim(-0.02, 0.72)
    ax_z.set_title(f'{label}  —  object $z$', fontsize=9, fontweight='bold',
                   loc='left')
    ax_z.spines[['top', 'right']].set_visible(False)
    ax_z.grid(axis='y', linestyle='--', alpha=0.35)
    ax_z.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

    if ep is None:
      print(f'{label}: no ep_length_mean')
    else:
      xs, mean, _n = ep
      ax_ep.plot(xs, mean, color=color, linestyle=ls, lw=1.5)
      _mark(ax_ep, xs, mean, resets, color)
      print(f'  eplen last={mean[-1]:.1f} min={min(mean):.1f}')
    ax_ep.axhline(NOMINAL_EP, color='0.55', lw=0.7, linestyle='--')
    ax_ep.axhline(RESET_EP_LEN, color='0.45', lw=0.9, linestyle=':')
    ax_ep.set_ylabel('ep len', fontsize=8)
    ax_ep.set_ylim(40, 320)
    ax_ep.set_title(
        rf'{label}  —  episode length  ($\times$ = actor-reset, n={len(resets)})',
        fontsize=9, fontweight='bold', loc='left')
    ax_ep.spines[['top', 'right']].set_visible(False)
    ax_ep.grid(axis='y', linestyle='--', alpha=0.35)
    ax_ep.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

    if succ is None:
      print(f'  no train_success_1000')
    else:
      sxs, smean, _n = succ
      hits = [(x, y) for x, y in zip(sxs, smean) if y > 0]
      ax_s.plot(sxs, smean, color=color, linestyle=ls, lw=1.5)
      if hits:
        hx, hy = zip(*hits)
        ax_s.scatter(hx, hy, color=color, s=22, zorder=4,
                     edgecolors='white', linewidths=0.4)
      for rx in resets:
        ax_s.axvline(rx, color=color, lw=0.7, linestyle='--', alpha=0.35)
      hit_s = ', '.join(f'{x/1e6:.1f}M' for x, _y in hits) or 'none'
      print(f'  success peak={max(smean):.4g}  hits={hit_s}')
    ax_s.axhline(0.001, color='0.75', lw=0.6, linestyle=':')
    ax_s.set_ylabel('succ', fontsize=8)
    ax_s.set_ylim(-0.00012, 0.00155)
    ax_s.set_title(f'{label}  —  train_success_1000', fontsize=9,
                   fontweight='bold', loc='left')
    ax_s.spines[['top', 'right']].set_visible(False)
    ax_s.grid(axis='y', linestyle='--', alpha=0.35)
    ax_s.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  axes[-1, 0].set_xlabel('Env Steps', fontsize=10)
  axes[-1, 1].set_xlabel('Env Steps', fontsize=10)
  axes[-1, 2].set_xlabel('Env Steps', fontsize=10)
  fig.suptitle(
      'Allegro throw blip runs — object $z$, episode length, train success  '
      r'($\times$ / dashed = last-layer actor-reset; '
      r'dotted $z$=0.1 fall, dotted eplen=200; success zoomed to 0.001)',
      fontsize=11, fontweight='bold', y=1.002)
  fig.tight_layout(rect=[0, 0, 1, 0.985])
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
