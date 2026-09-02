#!/usr/bin/env python3
"""Palm-behind action means: per-dim executed a, plus pre-tanh loc/scale.

The actor is Independent(Tanh(Normal(loc, scale))) with loc also soft-clamped
via 10*tanh(raw/10). CSV only logs scalars averaged over the 23 dims.
Per-dim means come from latest.pkl replay (post-tanh samples in [-1, 1]).

  python scripts/plot_allegro_tableside_palmbehind_actions.py
"""
from __future__ import annotations

import os
import pickle
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402
from plot_allegro_actorreset_diag import ACTION_DIM_LABELS, ARM_DIMS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
    '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_'
    'tableside_initrand_noshape_mixtaskg_mix50_actorreset_palmbehind_extrew1_4h'
)
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_tableside_palmbehind_actions.png')
ACCENT_COLORS = base.ACCENT_COLORS
COLOR = ACCENT_COLORS[0]


def _curve(y_col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, LOG_DIR, split='learner',
      x_col='global_step', y_col=y_col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return [], [], n
  xs, mean, _se = base._subsample_curve(xs, mean, se)
  return xs, mean, n


def _replay_actions():
  cache = os.path.join(
      REPO, 'figs', 'allegro_kuka_throw', '_palmbehind_replay_actions.npy')
  if os.path.isfile(cache):
    act = np.load(cache)
    print(f'loaded cached actions {cache} shape={act.shape}')
    return act if act.ndim == 2 else None
  path = os.path.join(
      base.LOG_ROOT, LOG_DIR, 'ppo_allegro_kuka_throw_0',
      'checkpoints', 'latest.pkl')
  if not os.path.isfile(path):
    print(f'no latest.pkl: {path}')
    return None
  print(f'loading {path} ...', flush=True)
  with open(path, 'rb') as fh:
    ckpt = pickle.load(fh)
  act = ((ckpt.get('extra_state') or {}).get('replay') or {}).get('action')
  del ckpt
  if act is None:
    print('latest.pkl has no extra_state.replay.action')
    return None
  act = np.asarray(act, dtype=np.float32)
  if act.ndim != 2:
    print(f'unexpected action shape {act.shape}')
    return None
  os.makedirs(os.path.dirname(cache), exist_ok=True)
  np.save(cache, act)
  return act


def main() -> None:
  fig, axes = plt.subplots(
      3, 1, figsize=(12.2, 10.4),
      gridspec_kw={'height_ratios': [1.15, 1.0, 1.0]})

  ax = axes[0]
  series = (
      ('ppo/policy_loc_mean', r'mean loc  (pre-tanh, all dims)',
       '-', ACCENT_COLORS[0]),
      ('ppo/policy_loc_abs_mean', r'mean $|$loc$|$',
       '--', ACCENT_COLORS[1]),
      ('ppo/policy_scale_mean', r'mean scale  (pre-tanh $\sigma$)',
       '-', ACCENT_COLORS[2]),
  )
  for col, label, ls, color in series:
    xs, ys, n = _curve(col)
    if not xs:
      print(f'{col}: no data')
      continue
    ax.plot(xs, ys, color=color, lw=1.8, ls=ls, label=label)
    print(f'{col}: n={n} last={ys[-1]:.4g}')
  ax.axhline(0.0, color='0.55', lw=0.7)
  ax.axhline(1e-5, color='0.55', lw=0.7, ls=':', label='min_std=1e-5')
  ax.set_title(
      'Palm-behind — Gaussian loc / scale  (CSV scalars, not per-dim)',
      fontsize=11, fontweight='bold')
  ax.set_ylabel('pre-tanh', fontsize=10)
  ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  act = _replay_actions()
  ax_m, ax_a = axes[1], axes[2]
  if act is not None:
    mu = act.mean(axis=0)
    sd = act.std(axis=0)
    mag = np.abs(act).mean(axis=0)
    xs = np.arange(len(mu))
    labels = ACTION_DIM_LABELS[:len(mu)]
    print(f'actions n={act.shape[0]} dim={act.shape[1]} '
          f'arm_mean={mu[:ARM_DIMS].mean():+.3f} '
          f'hand_mean={mu[ARM_DIMS:].mean():+.3f} '
          f'mean|a|={mag.mean():.3f}')
    for i, (name, m, s, a) in enumerate(zip(labels, mu, sd, mag)):
      print(f'  {name:5s}  mean={m:+.3f}  std={s:.3f}  mean|a|={a:.3f}')
    ax_m.plot(xs, mu, color=COLOR, lw=1.7, marker='o', ms=4)
    ax_m.fill_between(xs, mu - sd, mu + sd, color=COLOR, alpha=0.15, lw=0)
    ax_a.plot(xs, mag, color=COLOR, lw=1.7, marker='o', ms=4)
    ax_m.axhline(0.0, color='0.55', lw=0.7)
    ax_m.axvline(ARM_DIMS - 0.5, color='0.6', lw=0.8, ls='--')
    ax_a.axvline(ARM_DIMS - 0.5, color='0.6', lw=0.8, ls='--')
    ax_m.set_ylim(-1.05, 1.05)
    ax_a.set_ylim(0.0, 1.05)
    ax_a.set_xticks(xs)
    ax_a.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax_m.set_title(
        rf'Executed $a=\tanh(z)$ per dim  (latest.pkl replay, $n$={act.shape[0]:,})',
        fontsize=11, fontweight='bold')
    ax_a.set_title(r'mean $|a|$ per dim', fontsize=11, fontweight='bold')
  else:
    ax_m.text(0.5, 0.5, 'no replay actions', ha='center', va='center',
              transform=ax_m.transAxes)
    ax_a.text(0.5, 0.5, 'no replay actions', ha='center', va='center',
              transform=ax_a.transAxes)
  ax_m.set_ylabel(r'mean $a$  (band = $\pm$1 std)')
  ax_a.set_ylabel(r'mean $|a|$')
  ax_a.set_xlabel('action dim  (J1–J7 arm, then fingers)')
  for ax in (ax_m, ax_a):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
