#!/usr/bin/env python3
"""Train/eval success for recent Allegro tablespawn-grasp ablations.

One table-center goal run exists (``_center_extrew1_zmax07_``). This
script plots that run alone and overlays every sibling grasp ablation.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5).

  python scripts/plot_allegro_recent_ablations_eval_success.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, 'figs', 'allegro_kuka_throw')
OUT_ALL = os.path.join(FIG_DIR, 'akt_recent_ablations_train_eval_success.png')
OUT_CENTER = os.path.join(FIG_DIR, 'akt_center_zmax07_train_eval_success.png')
OUT_CENTER_EP_LEN = os.path.join(
    FIG_DIR, 'akt_center_zmax07_episode_length.png')

_P = (
    'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
    '_tau05_minstd1e5_'
)
# (log_dir basename, legend label, highlight)
RUNS = (
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew1_4h',
        'desk (0.20,-0.15) · ent 0.05→0.01 · extrew1  (baseline)',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent0005to0001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew1_4h',
        'desk · ent 0.005→0.001 · extrew1',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew10_4h',
        'desk · ent 0.05→0.01 · extrew10',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew1_succreplace10_4h',
        'desk · success→fail replace×10',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew1_zmax07_4h',
        'desk · reset z>0.7',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew1_zmax06_4h',
        'desk · reset z>0.6',
        False,
    ),
    (
        _P + 'ent05_ep300_300m_crl10_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew1_zmax1_4h',
        'desk · ent=0.05 fixed · reset z>1',
        False,
    ),
    (
        _P + 'ent05_ep300_300m_crl10_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_extrew1_zmax06_4h',
        'desk · ent=0.05 fixed · reset z>0.6',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_palmgoal_extrew1_4h',
        'desk · palm+object goal',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_handgoal_extrew1_4h',
        'desk · hand-joint goal',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_handgoal_extrew1_zmax1_4h',
        'desk · hand-joint · reset z>1',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_handgoal_extrew1_zmax06_4h',
        'desk · hand-joint · reset z>0.6',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_bucket_extrew1_4h',
        'bucket throw · grasp init',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_bucket_standoff18_zmax1_extrew1_4h',
        'bucket · standoff18 · reset z>1',
        False,
    ),
    (
        _P + 'entanneal_ep300_300m_crl10_ent05to001_tableside_tablespawn_grasp_'
        'noshape_mixtaskg_mix50_center_extrew1_zmax07_4h',
        'CENTER (0,0) · reset z>0.7',
        True,
    ),
)

CENTER_DIR = RUNS[-1][0]


def _style(ax, ylabel: str, *, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=10)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))


def _plot_run(axes, log_dir: str, label: str, color: str, *,
              highlight: bool = False, linestyle: str = '-') -> dict:
  train_seeds = base._read_train_seed_series(base.LOG_ROOT, log_dir)
  tx, tmean, tse, tn = base._aggregate_mean_stderr(train_seeds)
  eval_seeds = base._read_eval_seed_series(base.LOG_ROOT, log_dir)
  ex, emean, _, en = base._aggregate_mean_stderr(eval_seeds)
  stats = {
      'label': label, 'tn': tn, 'en': en,
      'train_last': None, 'train_peak': None, 'train_peak_m': None,
      'eval_last': None, 'eval_peak': None, 'eval_peak_m': None,
  }
  lw = 2.6 if highlight else 1.7
  z = 5 if highlight else 3
  if tx:
    tx, tmean, _ = base._subsample_curve(tx, tmean, tse)
    peak_i = tmean.index(max(tmean))
    stats['train_last'] = tmean[-1]
    stats['train_peak'] = tmean[peak_i]
    stats['train_peak_m'] = tx[peak_i] / 1e6
    axes[0].plot(
        tx, tmean, color=color, lw=lw, zorder=z, linestyle=linestyle,
        label=label)
    print(f'{label} train: n={tn} pts={len(tx)} last={tmean[-1]:.4f} '
          f'peak={tmean[peak_i]:.4g} @ {tx[peak_i]/1e6:.1f}M')
  else:
    print(f'{label} train: no data')
  if ex:
    sm = base._plot_eval_smoothed(
        axes[1], ex, emean, color=color,
        label=f'{label}  (roll mean w={base.EVAL_SMOOTH_WINDOW})',
        linestyle=linestyle, linewidth=lw, zorder=z)
    peak_i = sm.index(max(sm)) if sm else emean.index(max(emean))
    ys = sm if sm else emean
    stats['eval_last'] = ys[-1]
    stats['eval_peak'] = ys[peak_i]
    stats['eval_peak_m'] = ex[peak_i] / 1e6
    print(f'{label} eval: n={en} pts={len(ex)} last={ys[-1]:.4f} '
          f'peak={ys[peak_i]:.4g} @ {ex[peak_i]/1e6:.1f}M')
  else:
    print(f'{label} eval: no data')
  return stats


def _save(fig, path: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + '.tmp.png'
  fig.savefig(tmp, dpi=160, bbox_inches='tight')
  os.replace(tmp, path)
  plt.close(fig)
  print(f'→ {path}')


def _center_only() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), sharex=True)
  log_dir, label, _hl = RUNS[-1]
  color = base.ACCENT_COLORS[2]
  _plot_run(axes, log_dir, label, color, highlight=True)
  axes[0].set_title(
      'Allegro table-center goal  (0, 0, 0.555)  ·  train (raw)',
      fontsize=12, fontweight='bold')
  _style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper left', fontsize=8, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=12, fontweight='bold')
  _style(axes[1], f'eval success (roll mean, w={base.EVAL_SMOOTH_WINDOW})',
         xlabel=True)
  axes[1].legend(loc='upper left', fontsize=8, framealpha=0.95)
  fig.tight_layout()
  _save(fig, OUT_CENTER)


def _center_episode_length() -> None:
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, CENTER_DIR, split='learner',
      x_col='global_step', y_col='ep_length_mean')
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    raise RuntimeError(f'No ep_length_mean data found for {CENTER_DIR}')
  xs, mean, _ = base._subsample_curve(xs, mean, se)

  fig, ax = plt.subplots(figsize=(10.8, 4.2))
  ax.plot(xs, mean, color=base.ACCENT_COLORS[2], linewidth=1.8,
          label='episode length mean (raw)')
  ax.axhline(300, color='0.35', linewidth=1.2, linestyle='--',
             label='episode cap = 300')
  ax.set_title(
      'Allegro table-center goal (0, 0, 0.555) — episode length',
      fontsize=12, fontweight='bold')
  _style(ax, 'mean episode length (steps)', xlabel=True)
  ax.set_ylim(bottom=0)
  ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
  fig.tight_layout()
  _save(fig, OUT_CENTER_EP_LEN)
  print(f'episode length: n={n} points={len(xs)} '
        f'first={mean[0]:.2f} min={min(mean):.2f} '
        f'max={max(mean):.2f} last={mean[-1]:.2f}')


def _all_ablations() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(13.4, 8.8), sharex=True)
  n_ok = 0
  n_color = len(base.ACCENT_COLORS)
  for i, (log_dir, label, highlight) in enumerate(RUNS):
    color = base.ACCENT_COLORS[i % n_color]
    ls = '-' if i < n_color else '--'
    stats = _plot_run(
        axes, log_dir, label, color, highlight=highlight, linestyle=ls)
    if stats['tn'] or stats['en']:
      n_ok += 1
  axes[0].set_title(
      'Allegro tablespawn-grasp ablations — train (raw)',
      fontsize=12, fontweight='bold')
  _style(axes[0], 'train_success_1000')
  axes[0].legend(
      loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7.2,
      framealpha=0.95, borderaxespad=0.3)
  axes[1].set_title(
      f'eval success  (rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=12, fontweight='bold')
  _style(axes[1], f'eval success (roll mean, w={base.EVAL_SMOOTH_WINDOW})',
         xlabel=True)
  axes[1].legend(
      loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7.2,
      framealpha=0.95, borderaxespad=0.3)
  fig.tight_layout()
  _save(fig, OUT_ALL)
  print(f'plotted {n_ok}/{len(RUNS)} runs')


def main() -> None:
  print(f'center-goal runs: 1  ({CENTER_DIR})')
  print('--- center only ---')
  _center_only()
  print('--- center episode length ---')
  _center_episode_length()
  print('--- all grasp ablations ---')
  _all_ablations()


if __name__ == '__main__':
  main()
