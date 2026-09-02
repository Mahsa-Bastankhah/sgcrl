#!/usr/bin/env python3
"""Allegro throw/slide: return-norm std, r_T-r_0, and failure diagnostics.

Train curves are raw (not smoothed). Eval is off on these runs.

  python scripts/plot_allegro_kuka_throw_fail_diag.py
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
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'akt_fail_returnstd_rTr0.png')
C = base.ACCENT_COLORS

RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001',
        'nfcsm 300M throw',
        C[0],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep150_T150_300m_crl10_ent05to001_2h',
        'nfcsm T=150',
        C[1],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_norand_2h',
        'nfcsm norand',
        C[2],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_sa3x256_r64_b6_w256'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_palmgoal_stateonly_2h',
        'nfc palmgoal p(g|s)',
        C[3],
        '-',
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
        '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001'
        '_norand_mixtaskg_2h',
        'nfcsm norand + mix g_task',
        C[4],
        '-',
    ),
)

PANELS = (
    ('train_success_1000', 'train success (last 1000)', False),
    ('ep_length_mean', 'episode length  (nominal 300)', False),
    ('reward_return_norm_std', r'return-norm $\sigma$  (PPO divides $r$ by this)', False),
    ('repr_rT_minus_r0_mean_fail', r'$r_T-r_0$  (failed episodes)', True),
    ('reward_repr_raw_mean', 'raw NF reward  (pre-norm)', True),
    ('nf/log_p_mean', r'NF $\log p$ mean', True),
    ('ppo/entropy', 'PPO policy entropy', False),
    ('nf/goal_mean_0', r'NF goal $\mu_x$  (bucket $x=0.5$)', False),
)


def _curve(log_dir: str, col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, log_dir, split='learner',
      x_col='global_step', y_col=col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return None
  xs, mean, _ = base._subsample_curve(xs, mean, se)
  return xs, mean, n


def _style(ax, ylabel: str, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=9)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)


def main() -> None:
  fig, axes = plt.subplots(4, 2, figsize=(12.4, 12.2), sharex=True)
  axes = axes.ravel()
  for ax, (col, title, hline0) in zip(axes, PANELS):
    any_pts = False
    for log_dir, label, color, ls in RUNS:
      got = _curve(log_dir, col)
      if got is None:
        print(f'{label}: no {col}')
        continue
      xs, mean, n = got
      ax.plot(xs, mean, color=color, linestyle=ls, lw=1.7, alpha=0.95,
              label=label if ax is axes[0] else None)
      any_pts = True
      print(f'{label} {col}: n={n} last={mean[-1]:.4g} steps={xs[-1]:.0f}')
    if col == 'repr_rT_minus_r0_mean_fail':
      for log_dir, label, color, _ls in RUNS:
        got = _curve(log_dir, 'repr_rT_minus_r0_mean_succ')
        if got is None:
          continue
        xs, mean, n = got
        if not any(y == y for y in mean):
          continue
        ax.plot(xs, mean, color=color, linestyle='--', lw=1.2, alpha=0.75,
                label=f'{label} succ' if ax is axes[0] else None)
        print(f'{label} rT-r0 succ: n={n} last={mean[-1]:.4g}')
    if not any_pts:
      ax.text(0.5, 0.5, f'no {col}', ha='center', va='center',
              transform=ax.transAxes, color='0.5')
    ax.set_title(title, fontsize=10, fontweight='bold')
    if hline0:
      ax.axhline(0.0, color='0.55', lw=0.7)
    if col == 'ep_length_mean':
      ax.axhline(300.0, color='0.55', lw=0.7, linestyle='--')
    if col == 'nf/goal_mean_0':
      ax.axhline(0.5, color='0.55', lw=0.7, linestyle='--')
    _style(ax, '', xlabel=ax in axes[-2:])
  axes[0].legend(loc='upper right', fontsize=7.5, framealpha=0.95)
  fig.suptitle(
      'AllegroKukaThrow — why no success: ep length, return-norm std, '
      r'$r_T-r_0$ (fail solid / succ dashed), raw NF $r$, $\log p$, entropy, goal $\mu_x$',
      fontsize=11, fontweight='bold', y=0.995)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout(rect=[0, 0, 1, 0.96])
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=140, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
