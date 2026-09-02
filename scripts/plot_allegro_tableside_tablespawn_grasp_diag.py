#!/usr/bin/env python3
"""Diagnostics for tableside table-spawn grasp mix50, no palm_goal (3803362).

Panels: train+eval success, episode length, PPO rewards, object / reset
positions, return-norm std.

Train curves are raw. Eval is faint raw + bold rolling mean (window=5).
Object xyz is the learner rollout mean (not a dedicated reset log). Spawn
and goal are the known table-spawn / table-push constants. Goal is 3-D
object-only (no palm in goal).

  python scripts/plot_allegro_tableside_tablespawn_grasp_diag.py
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
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
    '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_'
    'tableside_tablespawn_grasp_noshape_mixtaskg_mix50_extrew1_4h'
)
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_tableside_tablespawn_grasp_nopalmgoal_diag.png')
RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')
NOMINAL_EP = 300.0
# Table-spawn cube and on-desk slide target (world m).
SPAWN_XYZ = (0.17, 0.08, 0.555)
GOAL_XYZ = (0.20, -0.15, 0.555)
SUCC_RAD = 0.075
# Noiseless IK confirm from the train slurm header (one env).
RESET_PALM_XY = (0.165, 0.105)
RESET_OBJ_XY = (0.161, 0.082)
COLOR = base.ACCENT_COLORS[2]
C_EVAL = base.ACCENT_COLORS[0]
C_RAW = '#C45C26'
C_NORM = '#4C9BE8'
C_ENV = '#2A9D8F'
C_RET = '#A84CE8'


def _curve(y_col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, LOG_DIR, split='learner',
      x_col='global_step', y_col=y_col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return [], [], n
  xs, mean, _se = base._subsample_curve(xs, mean, se)
  return xs, mean, n


def _reset_events() -> list[tuple[int, int]]:
  mapping = base._iter_to_env_steps_map(base.LOG_ROOT, LOG_DIR)
  events: list[tuple[int, int]] = []
  for path in sorted(glob.glob(os.path.join(REPO, 'slurm', f'{LOG_DIR}_*.log'))):
    with open(path, encoding='utf-8', errors='replace') as fh:
      for line in fh:
        m = RESET_RE.search(line)
        if not m:
          continue
        it = int(m.group(1))
        step = mapping.get(it)
        if step is None:
          continue
        events.append((it, int(step)))
  return sorted(set(events))


def _vlines(ax, steps, *, color=COLOR):
  if not steps:
    return
  ax.vlines(
      steps, 0, 1, transform=ax.get_xaxis_transform(),
      colors=color, alpha=0.28, lw=0.9, linestyles='--')


def _style(ax, ylabel: str, *, xlabel: bool = False):
  ax.set_ylabel(ylabel, fontsize=9)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))


def _draw_reset_xy(ax) -> None:
  """Top-down spawn / goal (not a timeseries — reset xyz is not logged)."""
  circ = Circle(
      (GOAL_XYZ[0], GOAL_XYZ[1]), SUCC_RAD,
      facecolor='#E8834C', edgecolor='#C45C26', alpha=0.18, lw=1.4,
      zorder=1, label='success disk  7.5 cm')
  ax.add_patch(circ)
  ax.scatter(
      [SPAWN_XYZ[0]], [SPAWN_XYZ[1]], marker='s', s=70, color='#4C9BE8',
      zorder=4, label=f'spawn  {SPAWN_XYZ[:2]}')
  ax.scatter(
      [GOAL_XYZ[0]], [GOAL_XYZ[1]], marker='*', s=110, color='#C45C26',
      zorder=4, label=f'object goal (3D)  {GOAL_XYZ[:2]}')
  ax.scatter(
      [RESET_OBJ_XY[0]], [RESET_OBJ_XY[1]], marker='o', s=36, color='#4C9BE8',
      facecolors='none', linewidths=1.3, zorder=5,
      label=f'IK confirm obj  {RESET_OBJ_XY}')
  ax.scatter(
      [RESET_PALM_XY[0]], [RESET_PALM_XY[1]], marker='P', s=42, color='#2A9D8F',
      zorder=5, label=f'IK confirm palm  {RESET_PALM_XY}')
  ax.annotate(
      '', xy=(GOAL_XYZ[0], GOAL_XYZ[1]),
      xytext=(SPAWN_XYZ[0], SPAWN_XYZ[1]),
      arrowprops=dict(arrowstyle='->', color='0.45', lw=1.1))
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlim(-0.05, 0.35)
  ax.set_ylim(-0.28, 0.20)
  ax.set_xlabel('x (m)', fontsize=9)
  ax.set_ylabel('y (m)', fontsize=9)
  ax.set_title(
      'reset / goal  (desk xy; one-env IK confirm, not rollout mean)',
      fontsize=10, fontweight='bold')
  ax.legend(loc='lower left', fontsize=7, framealpha=0.95, ncol=1)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(True, linestyle='--', alpha=0.35)


def main() -> None:
  resets = _reset_events()
  reset_steps = [s for _it, s in resets]
  print(f'actor-resets={len(resets)} '
        + ', '.join(f'iter={it}@{s/1e6:.1f}M' for it, s in resets[:12])
        + (' …' if len(resets) > 12 else ''))

  fig = plt.figure(figsize=(11.2, 16.4))
  gs = fig.add_gridspec(6, 2, height_ratios=[1.05, 0.95, 1.05, 1.05, 1.15, 1.0],
                        hspace=0.38, wspace=0.28)

  # --- success (train raw, eval smoothed) ---
  ax = fig.add_subplot(gs[0, :])
  tx, ty, tn = _curve('train_success_1000')
  if tx:
    ax.plot(tx, ty, color=COLOR, lw=2.0, label='train_success_1000  (raw)')
    print(f'train success: n={tn} pts={len(tx)} last={ty[-1]:.4f} '
          f'peak={max(ty):.4g} at {tx[ty.index(max(ty))]/1e6:.1f}M')
  else:
    print('train success: no data')
  eval_seeds = base._read_eval_seed_series(base.LOG_ROOT, LOG_DIR)
  ex, emean, _, en = base._aggregate_mean_stderr(eval_seeds)
  if ex:
    base._plot_eval_smoothed(
        ax, ex, emean, color=C_EVAL,
        label=f'eval  (roll mean w={base.EVAL_SMOOTH_WINDOW})')
    print(f'eval success: n={en} pts={len(ex)} last={emean[-1]:.4f} '
          f'peak={max(emean):.4g} at {ex[emean.index(max(emean))]/1e6:.1f}M')
  else:
    print('eval success: no data')
  _vlines(ax, reset_steps)
  ax.set_title('Allegro table-spawn grasp (no palm_goal) — success',
               fontsize=11, fontweight='bold')
  _style(ax, 'success')
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)

  # --- episode length ---
  ax = fig.add_subplot(gs[1, :])
  xs, ys, n = _curve('ep_length_mean')
  if xs:
    ax.plot(xs, ys, color=COLOR, lw=2.0, label='ep_length_mean')
    print(f'ep_length: n={n} pts={len(xs)} last={ys[-1]:.1f} '
          f'min={min(ys):.1f} max={max(ys):.1f}')
  ax.axhline(NOMINAL_EP, color='0.45', lw=0.9, ls=':', label='nominal T=300')
  _vlines(ax, reset_steps)
  ax.set_title('episode length  (actor-reset off)', fontsize=11, fontweight='bold')
  _style(ax, 'steps')
  ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=2)

  # --- PPO rewards ---
  ax = fig.add_subplot(gs[2, :])
  for col, label, color, ls in (
      ('reward_repr_raw_mean', r'raw NF $r$  (pre-norm $\tau\log p$)', C_RAW, '-'),
      ('reward_repr_mean', r'NF $r$ after return-norm  (PPO policy)', C_NORM, '-'),
      ('reward_env_mean', 'env / extrew mean  (+1 after norm)', C_ENV, '--'),
  ):
    xs, ys, n = _curve(col)
    if not xs:
      print(f'{col}: no data')
      continue
    ax.plot(xs, ys, color=color, lw=1.8, ls=ls, label=label)
    print(f'{col}: n={n} last={ys[-1]:.4g} min={min(ys):.4g} max={max(ys):.4g}')
  xs, ys, n = _curve('ep_return_mean')
  if xs:
    ax2 = ax.twinx()
    ax2.plot(xs, ys, color=C_RET, lw=1.3, alpha=0.85,
             label='ep_return_mean  (right)')
    ax2.set_ylabel('episode return', fontsize=9, color=C_RET)
    ax2.spines['top'].set_visible(False)
    print(f'ep_return_mean: n={n} last={ys[-1]:.1f} '
          f'min={min(ys):.1f} max={max(ys):.1f}')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7.5,
              framealpha=0.95, ncol=2)
  else:
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
  _vlines(ax, reset_steps)
  ax.set_title('PPO rewards', fontsize=11, fontweight='bold')
  _style(ax, 'reward')

  # --- object / reset positions ---
  ax = fig.add_subplot(gs[3, :])
  for col, label, color in (
      ('object_x_mean', r'object $x$ mean', '#4C9BE8'),
      ('object_y_mean', r'object $y$ mean', '#E8834C'),
      ('object_z_mean', r'object $z$ mean', '#2A9D8F'),
      ('object_goal_dist_mean', r'$\|obj-goal\|$ mean', '#C45C26'),
  ):
    xs, ys, n = _curve(col)
    if not xs:
      print(f'{col}: no data')
      continue
    ax.plot(xs, ys, color=color, lw=1.6, label=label)
    print(f'{col}: n={n} last={ys[-1]:.3f} min={min(ys):.3f} max={max(ys):.3f}')
  ax.axhline(SPAWN_XYZ[2], color='0.5', lw=0.8, ls=':',
             label=f'table / spawn z={SPAWN_XYZ[2]}')
  _vlines(ax, reset_steps)
  ax.set_title(
      'object position  (rollout mean over E×T; not reset-only)',
      fontsize=11, fontweight='bold')
  _style(ax, 'm  (or obs units)')
  ax.legend(loc='upper right', fontsize=7.5, framealpha=0.95, ncol=3)

  ax_xy = fig.add_subplot(gs[4, 0])
  _draw_reset_xy(ax_xy)

  ax_z = fig.add_subplot(gs[4, 1])
  xs, ys, n = _curve('object_z_frac_below_01')
  if xs:
    ax_z.plot(xs, ys, color='#C45C26', lw=1.8, label=r'frac  $z<0.1$m')
    print(f'object_z_frac_below_01: n={n} last={ys[-1]:.3f} max={max(ys):.3f}')
  _vlines(ax_z, reset_steps)
  ax_z.set_title('fallen objects  (z < 0.1 m)', fontsize=10, fontweight='bold')
  _style(ax_z, 'fraction')
  ax_z.set_ylim(-0.02, 1.02)
  ax_z.legend(loc='upper left', fontsize=8, framealpha=0.95)
  ax_z.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax_z.set_xlabel('Env Steps', fontsize=9)

  # --- return std ---
  ax = fig.add_subplot(gs[5, :])
  xs, ys, n = _curve('reward_return_norm_std')
  if xs:
    ax.plot(xs, ys, color=COLOR, lw=2.0,
            label=r'return-norm $\sigma$  (PPO divides $r$ by this)')
    print(f'return_norm_std: n={n} last={ys[-1]:.3g} '
          f'min={min(ys):.3g} max={max(ys):.3g}')
  xs2, ys2, n2 = _curve('reward_repr_raw_std')
  if xs2:
    ax.plot(xs2, ys2, color=C_RAW, lw=1.5, ls='--',
            label=r'raw NF $r$ std  (per-step)')
    print(f'reward_repr_raw_std: n={n2} last={ys2[-1]:.3g}')
  _vlines(ax, reset_steps)
  ax.set_title(r'std of the return', fontsize=11, fontweight='bold')
  _style(ax, r'$\sigma$', xlabel=True)
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
