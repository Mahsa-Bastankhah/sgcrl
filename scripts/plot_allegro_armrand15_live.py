#!/usr/bin/env python3
"""Live diagnostics for the hover + arm±15% Allegro push run (3861584).

Overlays the completed hover-only run (curl15 / finger±10%, no arm noise).
Train is raw. Eval is faint raw + bold rolling mean (window=5).

  python scripts/plot_allegro_armrand15_live.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_armrand15_live.png')

ARMRAND = (
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tableside_largetable15'
    '_above12_corr03_ep50_200m_objgoal_curl15_frand10_armrand15_seed0_4h')
HOVER = (
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tableside_largetable15'
    '_above12_corr03_ep50_200m_objgoal_curl15_frand10_seed0_4h')

C_LIVE = base.ACCENT_COLORS[0]
C_REF = '#9AA3AD'
SPAWN_Z = 0.555
GOAL_DIST0 = 0.23  # ~cube (0.17,0.08) → goal (0.20,-0.15)


def _curve(log_dir: str, y_col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, log_dir, split='learner',
      x_col='global_step', y_col=y_col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return [], [], n
  xs, mean, _se = base._subsample_curve(xs, mean, se)
  return xs, mean, n


def _style(ax, ylabel: str, xlabel: bool = False):
  ax.set_ylabel(ylabel, fontsize=10)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.grid(True, alpha=0.28)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)


def _plot_train(ax, log_dir, color, label, ls='-'):
  xs, ys, n = _curve(log_dir, 'train_success_1000')
  if xs:
    ax.plot(xs, ys, color=color, lw=2.0, ls=ls, label=label)
  return xs, ys, n


def main() -> int:
  fig, axes = plt.subplots(4, 1, figsize=(10.6, 12.4), sharex=False)
  fig.subplots_adjust(hspace=0.38)

  ax = axes[0]
  tx, ty, tn = _plot_train(
      ax, ARMRAND, C_LIVE, 'armrand15  train_success_1000  (raw)')
  _plot_train(
      ax, HOVER, C_REF, 'hover-only  train  (ref, raw)', ls='--')
  if tx:
    print(f'train live: n={tn} pts={len(tx)} last={ty[-1]:.4f} '
          f'peak={max(ty):.4g} at {tx[ty.index(max(ty))]/1e6:.1f}M')
  else:
    print('train live: no data')
  eval_seeds = base._read_eval_seed_series(base.LOG_ROOT, ARMRAND)
  ex, emean, _, en = base._aggregate_mean_stderr(eval_seeds)
  if ex:
    base._plot_eval_smoothed(
        ax, ex, emean, color=C_LIVE,
        label=f'armrand15  eval  (roll mean w={base.EVAL_SMOOTH_WINDOW})')
    print(f'eval live: n={en} pts={len(ex)} last={emean[-1]:.4f} '
          f'peak={max(emean):.4g}')
  else:
    print('eval live: no data')
  href = base._read_eval_seed_series(base.LOG_ROOT, HOVER)
  hx, hmean, _, hn = base._aggregate_mean_stderr(href)
  if hx:
    base._plot_eval_smoothed(
        ax, hx, hmean, color=C_REF, linestyle='--',
        label='hover-only  eval  (ref)')
  if tx:
    ax.set_xlim(0, max(tx[-1] * 1.08, 5e6))
  ax.set_title(
      'Allegro hover + arm±15%  —  success   (live vs completed hover-only)',
      fontsize=11, fontweight='bold')
  _style(ax, 'success')
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)

  ax = axes[1]
  for log_dir, color, ls, name in (
      (ARMRAND, C_LIVE, '-', 'armrand15'),
      (HOVER, C_REF, '--', 'hover-only'),
  ):
    xs, ys, n = _curve(log_dir, 'object_goal_dist_mean')
    if not xs:
      print(f'{name} dist: no data')
      continue
    ax.plot(xs, ys, color=color, lw=1.9, ls=ls, label=f'{name}  ||obj-goal||')
    print(f'{name} dist: last={ys[-1]:.3f} min={min(ys):.3f}')
  ax.axhline(GOAL_DIST0, color='0.45', lw=0.9, ls=':',
             label=f'spawn→goal ≈ {GOAL_DIST0:.2f} m')
  ax.axhline(0.075, color='#C45C26', lw=0.9, ls=':', label='success rad 7.5 cm')
  xs_d, _, _ = _curve(ARMRAND, 'object_goal_dist_mean')
  if xs_d:
    ax.set_xlim(0, max(xs_d[-1] * 1.08, 5e6))
  ax.set_title('cube–goal distance  (rollout mean)', fontsize=11,
               fontweight='bold')
  _style(ax, 'm')
  ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=2)

  ax = axes[2]
  for col, label, color in (
      ('object_x_mean', 'x', '#4C9BE8'),
      ('object_y_mean', 'y', '#E8834C'),
      ('object_z_mean', 'z', '#2A9D8F'),
  ):
    xs, ys, n = _curve(ARMRAND, col)
    if xs:
      ax.plot(xs, ys, color=color, lw=1.7, label=label)
      print(f'{col}: last={ys[-1]:.3f}')
  ax.axhline(SPAWN_Z, color='0.5', lw=0.8, ls=':', label=f'table z={SPAWN_Z}')
  ax.axhline(0.20, color='#4C9BE8', lw=0.7, ls=':', alpha=0.7)
  ax.axhline(-0.15, color='#E8834C', lw=0.7, ls=':', alpha=0.7)
  ax.set_title('object xyz  (armrand15 rollout mean; dotted = goal x/y, table z)',
               fontsize=11, fontweight='bold')
  _style(ax, 'm')
  ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=4)

  ax = axes[3]
  for col, label, color in (
      ('object_z_frac_above_1', r'frac  $z>1$ m  (fly)', '#E84C6F'),
      ('object_z_frac_below_01', r'frac  $z<0.1$ m  (fall)', '#C45C26'),
      ('ep_length_mean', 'ep length / 50', '#A84CE8'),
  ):
    xs, ys, n = _curve(ARMRAND, col)
    if not xs:
      continue
    if col == 'ep_length_mean':
      ys = [v / 50.0 for v in ys]
    ax.plot(xs, ys, color=color, lw=1.7, label=label)
    print(f'{col}: last={ys[-1]:.3f}')
  ax.set_ylim(-0.02, 1.15)
  ax.set_title('fly / fall / episode-length fraction   (armrand15)',
               fontsize=11, fontweight='bold')
  _style(ax, 'fraction', xlabel=True)
  ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=3)

  now = datetime.now().strftime('%Y-%m-%d %H:%M')
  fig.suptitle(f'updated {now}', fontsize=9, color='0.35', y=0.995)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
