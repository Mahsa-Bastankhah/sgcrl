#!/usr/bin/env python3
"""Hard-hover T=80: working NF vs one SGCRL run (300M).

Mid-hover and keep-arm SGCRL were cancelled (too easy).

Separate from scripts/plot_allegro_new_baselines.py (T=150 RND/MPO).

  python scripts/plot_allegro_sgcrl_ep80.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_allegro_hidetable_running_success as hidetable  # noqa: E402
import plot_allegro_new_baselines as bl  # noqa: E402
import plot_builderbench_train_success1000 as base  # noqa: E402

FIG_PATH = os.path.join(bl.FIG_DIR, 'akt_sgcrl_ep80_300m_success.png')
EVAL_WINDOW = bl.EVAL_WINDOW
C = bl.C

# One row per working T=80 recipe: NF reference + SGCRL 300M.
ROWS = (
    (
        'hard hover  near10  (0.45,-0.60)',
        (
            (
                'NF  T=80 q40  300M  (working)',
                'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
                'nf', C[3], '--',
            ),
            (
                'SGCRL  T=80  300M',
                'sac_crl_allegro_kuka_throw_e1024_hard_hover_near10_z055_ep80_300m_seed0_4h',
                'sac_crl', '#D62728', '-.',
            ),
        ),
    ),
)


def _plot_row(ax_train, ax_eval, title, methods):
  notes = []
  for label, log_dir, kind, color, ls in methods:
    print(f'{title} | {label}  {log_dir}', flush=True)
    train = bl._read_split(log_dir, kind, 'train')
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train)
    z = 20 if kind == 'sac_crl' else 3
    lw = 2.8 if kind == 'sac_crl' else 2.0
    if tx:
      hidetable._plot_train(
          ax_train, tx, tmean, tse, tn, color,
          label + ('  (eval)' if kind == 'sac_crl' else ''), ls=ls)
      if kind == 'sac_crl':
        ax_train.lines[-1].set_zorder(z)
        ax_train.lines[-1].set_linewidth(lw)
    else:
      print(f'{label} train: no data', flush=True)
    ev = bl._read_split(log_dir, kind, 'eval')
    ex, emean, ese, en = base._aggregate_mean_stderr(ev)
    if ex:
      if kind == 'sac_crl':
        sm = base._plot_eval_smoothed(
            ax_eval, ex, emean, color=color,
            label=f'{label}  (n={en}, roll mean w={EVAL_WINDOW})',
            linestyle=ls, window=EVAL_WINDOW, zorder=z, linewidth=lw)
        ys = sm if sm else emean
        print(f'{label} eval: n={en} last={ys[-1]:.4f} peak={max(ys):.4g} '
              f'pts={len(ex)} steps={ex[-1]/1e6:.1f}M', flush=True)
        notes.append(f'{label}: {ys[-1]:.3f} @ {ex[-1]/1e6:.1f}M')
      else:
        hidetable._plot_eval(ax_eval, ex, emean, ese, en, color, label, ls=ls)
    else:
      print(f'{label} eval: no data', flush=True)
  ax_train.set_title(f'{title} — train (SGCRL = greedy eval)', fontsize=11,
                     fontweight='bold')
  hidetable._style(ax_train, 'train_success_1000')
  ax_train.legend(loc='upper left', fontsize=7.6, framealpha=0.95)
  ax_eval.set_title(
      f'{title} — eval (faint raw + roll mean w={EVAL_WINDOW})',
      fontsize=11, fontweight='bold')
  hidetable._style(ax_eval, f'eval success (roll mean, w={EVAL_WINDOW})',
                   xlabel=True)
  ax_eval.legend(loc='upper left', fontsize=7.6, framealpha=0.95)
  if notes:
    ax_eval.text(
        0.98, 0.16, 'SGCRL\n' + '\n'.join(notes),
        transform=ax_eval.transAxes, ha='right', va='bottom',
        fontsize=7.6, color='#B41E5C', zorder=30,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#B41E5C', alpha=0.95))


def run_once() -> None:
  n = len(ROWS)
  fig, axes = plt.subplots(n, 2, figsize=(12.4, 4.8 * n), sharex=True)
  if n == 1:
    axes = axes.reshape(1, 2)
  for i, (title, methods) in enumerate(ROWS):
    _plot_row(axes[i, 0], axes[i, 1], title, methods)
  fig.suptitle(
      'SGCRL vs working NF  ·  T=ep=80  ·  300M  ·  in-bucket z=0.55',
      fontsize=13, fontweight='bold', y=0.995)
  fig.tight_layout(rect=(0, 0, 1, 0.98))
  os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
  fig.savefig(FIG_PATH, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {FIG_PATH}', flush=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--watch', type=int, default=0)
  parser.add_argument('--watch-max-sec', type=int, default=0)
  args = parser.parse_args()
  if args.watch <= 0:
    run_once()
    return
  t0 = time.time()
  print(
      f'[sgcrl ep80] watching every {args.watch}s '
      f'(max {args.watch_max_sec or "∞"}s)',
      flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001
      print(f'[sgcrl ep80] error: {exc}', flush=True)
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[sgcrl ep80] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
