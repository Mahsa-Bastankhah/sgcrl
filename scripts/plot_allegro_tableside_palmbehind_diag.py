#!/usr/bin/env python3
"""Palm-behind tableside diagnostics: success + actor-reset, r_T-r_0, ∇_s r.

Train curves are raw. Actor-reset marks come from slurm
``[ppo] actor reset at iter=``.

  python scripts/plot_allegro_tableside_palmbehind_diag.py
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = (
    'ppo_allegro_kuka_throw_e1024_nf_compact_small_sa3x192_r64_b6_w192'
    '_tau05_minstd1e5_entanneal_ep300_300m_crl10_ent05to001_'
    'tableside_initrand_noshape_mixtaskg_mix50_actorreset_palmbehind_extrew1_4h'
)
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_tableside_palmbehind_diag.png')
RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')
COLOR = base.ACCENT_COLORS[0]
C_SUCC = '#2A9D8F'
C_FAIL = '#C45C26'


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
      colors=color, alpha=0.45, lw=1.0, linestyles='--',
      label='actor reset' if steps else None)


def main() -> None:
  resets = _reset_events()
  reset_steps = [s for _it, s in resets]
  print(f'actor-resets={len(resets)} '
        + ', '.join(f'iter={it}@{s/1e6:.1f}M' for it, s in resets))

  fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.4), sharex=True)

  ax = axes[0]
  xs, ys, n = _curve('train_success_1000')
  if xs:
    ax.plot(xs, ys, color=COLOR, lw=2.0, label='train_success_1000  (raw)')
    print(f'train success: n={n} pts={len(xs)} last={ys[-1]:.4f} '
          f'peak={max(ys):.4g} at {xs[ys.index(max(ys))]/1e6:.1f}M')
  else:
    print('train success: no data')
  _vlines(ax, reset_steps)
  for it, step in resets:
    if not xs:
      break
    y = ys[min(range(len(xs)), key=lambda i: abs(xs[i] - step))]
    ax.scatter([step], [y], color=COLOR, marker='x', s=56, linewidths=1.6,
               zorder=5)
    ax.annotate(
        f'reset iter {it}',
        xy=(step, y), xytext=(10, 12), textcoords='offset points',
        fontsize=8, color=COLOR)
  ax.set_title('Allegro tableside palm-behind — train success + actor reset',
               fontsize=11, fontweight='bold')
  ax.set_ylabel('train_success_1000', fontsize=10)
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

  ax = axes[1]
  for col, label, color, ls in (
      ('repr_rT_minus_r0_mean_succ', r'succ  $r_T-r_0$', C_SUCC, '-'),
      ('repr_rT_minus_r0_mean_fail', r'fail  $r_T-r_0$', C_FAIL, '--'),
  ):
    xs, ys, n = _curve(col)
    if not xs:
      print(f'{col}: no data')
      continue
    ax.plot(xs, ys, color=color, lw=1.8, ls=ls, label=label)
    print(f'{col}: n={n} pts={len(xs)} last={ys[-1]:+.3f} '
          f'min={min(ys):+.3f} max={max(ys):+.3f}')
  ax.axhline(0.0, color='#8899aa', lw=1.0, ls=':', alpha=0.8, zorder=1)
  _vlines(ax, reset_steps)
  ax.set_title(r'NF reward  $r_T-r_0$  (completed episodes; succ often NaN)',
               fontsize=11, fontweight='bold')
  ax.set_ylabel(r'mean $r_T-r_0$', fontsize=10)
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

  ax = axes[2]
  xs, ys, n = _curve('nf/nf_logp_grad_s_norm_mean')
  if xs:
    ax.plot(xs, ys, color=COLOR, lw=2.0,
            label=r'mean $\|\nabla_s \log p\|$  ($\approx\|\nabla_s r\|$)')
    print(f'grad_s logp: n={n} pts={len(xs)} last={ys[-1]:.3f} '
          f'max={max(ys):.3f}')
  else:
    print('grad_s logp: no data')
  ax.axhline(100.0, color='0.35', lw=0.9, ls=':', label='c=100')
  _vlines(ax, reset_steps)
  ax.set_title(r'$\|\nabla_s r(s,a)\|$  from learner CSV  (V grad is not logged)',
               fontsize=11, fontweight='bold')
  ax.set_ylabel(r'mean $\|\nabla_s \log p\|$', fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
