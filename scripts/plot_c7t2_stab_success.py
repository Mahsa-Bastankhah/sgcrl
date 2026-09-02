#!/usr/bin/env python3
"""c7t2 150M stability ablations: train (raw) + eval (roll mean w=5).

Mean ±1 SE across seeds. One legend name per ablation.

  python scripts/plot_c7t2_stab_success.py
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

LOG_ROOT = os.path.join(base.LOG_ROOT, 'final_runs')
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'figs', 'builderbench', 'final_runs',
    'c7t2_stab5_train_eval_success.png')
OUT_KLPEN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'figs', 'builderbench', 'final_runs',
    'c7t2_stab_klpen_train_eval_success.png')
C = base.ACCENT_COLORS

CSM = (
    'ppo_builderbench_creative7_task2_e1024_pd'
    '_nf_compact_small_sa3x192_r64_b6_w192_tau05_'
)
TINY = (
    'ppo_builderbench_creative7_task2_e1024_pd'
    '_nf_tiny_sa2x128_r32_b4_w128_tau05_'
)
MID = '_catwp_extrew1_minstd1e5_'
S150 = '_ep70_150m_crl10_'

# Five live jobs after the cancel (each n=2, 150M).
ABLATIONS = (
    (
        f'{CSM}nopermute_fixedx01{MID}ent05to001{S150}'
        'dualgradreg_c100_lamlr1e6_tkl01_klpen1_kltgt005_betamin005'
        '_replaysub30_warp_2h',
        'KL-pen + replay sub 30%',
        C[0],
        '-',
    ),
    (
        f'{CSM}nopermute_fixedx01{MID}ent05{S150}'
        'dualgradreg_c100_lamlr1e6_tkl01_ent05_mixtaskg_klpen1'
        '_kltgt005_betamin005_warp_2h',
        'combo  (ent05 + mixg + KL-pen)',
        C[1],
        '--',
    ),
    (
        f'{TINY}nopermute_fixedx01{MID}ent05{S150}'
        'dualgradreg_c100_lamlr1e6_tkl01_ent05_mixtaskg_klpen1'
        '_kltgt005_betamin005_warp_2h',
        'combo tiny NF',
        C[2],
        '-.',
    ),
    (
        f'{CSM}nopermute_fixedx01{MID}ent05{S150}'
        'dualgradreg_c100_lamlr1e6_tkl01_ent05_mixtaskg_klpen1'
        '_kltgt005_betamin005_freezenf10_warp_2h',
        'combo + freeze NF @10',
        C[3],
        ':',
    ),
    (
        f'{CSM}nopermute_fixedx01{MID}ent05to001{S150}'
        'dualgradreg_c20_lamlr1e6_tkl01_klpen1_kltgt005_betamin005'
        '_warp_2h',
        r'KL-pen  $c=20$',
        C[4],
        '-',
    ),
)


def _shade(ax, xs, mean, se, *, color, n: int) -> None:
  if n <= 1 or not se:
    return
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)


def _style(ax, *, title: str, ylabel: str, xlabel: bool) -> None:
  ax.set_title(title, fontsize=12, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=10)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def main() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.4), sharex=True)
  fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.18, hspace=0.22)
  missing = []
  for name, label, color, ls in ABLATIONS:
    t_seeds = [
        pts for pts in base._read_csv_seed_series(
            LOG_ROOT, name, split='learner',
            x_col=base.TRAIN_X_COL, y_col=base.TRAIN_METRIC)
        if pts]
    e_raw = base._read_csv_seed_series(
        LOG_ROOT, name, split='eval',
        x_col=base.EVAL_X_COL, y_col=base.EVAL_METRIC)
    e_seeds = [
        base._iters_to_env_steps(pts, LOG_ROOT, name)
        for pts in e_raw if pts]
    txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
    exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
    n = max(n_t, n_e)
    lab = f'{label}  (n={n})'
    if not txs and not exs:
      missing.append(label)
      print(f'  {label}: no data')
      continue
    if txs:
      xs, mean, se = base._subsample_curve(txs, tmean, tse)
      _shade(axes[0], xs, mean, se, color=color, n=n_t)
      axes[0].plot(
          xs, mean, color=color, linestyle=ls, linewidth=2.2,
          alpha=0.95, label=lab, zorder=3)
    sm = []
    if exs:
      _shade(axes[1], exs, emean, ese, color=color, n=n_e)
      sm = base._plot_eval_smoothed(
          axes[1], exs, emean, color=color, label=lab, linestyle=ls,
          zorder=3, linewidth=2.2)
    bits = [f'  {label}']
    if tmean:
      bits.append(
          f'train last={tmean[-1]:.3f} peak={max(tmean):.3f} '
          f'@ {txs[-1]/1e6:.1f}M')
    if emean:
      bits.append(f'eval raw last={emean[-1]:.3f} peak={max(emean):.3f}')
    if sm:
      bits.append(f'smooth last={sm[-1]:.3f}')
    print('  '.join(bits))

  _style(
      axes[0],
      title='c7t2  ·  live 5  ·  train',
      ylabel='Train Success (last 1000)', xlabel=False)
  _style(
      axes[1],
      title=f'c7t2  ·  live 5  ·  eval (roll mean w={w})',
      ylabel=f'Eval Success (roll mean w={w})', xlabel=True)
  handles, labels = axes[0].get_legend_handles_labels()
  if missing:
    from matplotlib.lines import Line2D
    for lab in missing:
      handles.append(Line2D([0], [0], color='#888888', lw=1.4, ls=':'))
      labels.append(f'{lab}  (n=0, no logs)')
  leg = axes[0].legend(
      handles, labels,
      loc='upper left', fontsize=8.0, framealpha=0.95, handlelength=2.6,
      fancybox=False, edgecolor='#333333')
  fig.text(
      0.5, 0.015,
      'Five live 150M jobs. Mean ±1 SE across seeds. '
      'Train: raw train_success_1000. '
      f'Eval: faint raw + bold rolling mean, window={w}. '
      'combo = fixed ent 0.05 + mix task g + KL-pen. '
      'KL-pen is βmin=0.05, d_targ=0.05, tKL=0.1 unless noted.',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight', bbox_extra_artists=(leg,))
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


def _plot_klpen() -> None:
  """KL-pen only, one line per seed (not averaged)."""
  w = base.EVAL_SMOOTH_WINDOW
  name = (
      f'{PREFIX}nopermute_fixedx01{MID}ent05to001{SUFFIX}'
      'dualgradreg_c100_lamlr1e6_tkl01_klpen1_kltgt005_betamin005_warp_2h')
  train = [
      pts for pts in base._read_csv_seed_series(
          LOG_ROOT, name, split='learner',
          x_col=base.TRAIN_X_COL, y_col=base.TRAIN_METRIC)
      if pts]
  evals = []
  for pts in base._read_csv_seed_series(
      LOG_ROOT, name, split='eval',
      x_col=base.EVAL_X_COL, y_col=base.EVAL_METRIC):
    if pts:
      evals.append(base._iters_to_env_steps(pts, LOG_ROOT, name))
  print('c7t2 stab KL-pen  per-seed')
  fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.2), sharex=True)
  fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16, hspace=0.22)
  seed_colors = (C[0], C[3], C[1], C[2])
  for i, pts in enumerate(train):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = seed_colors[i % len(seed_colors)]
    xs_p, ys_p, _ = base._subsample_curve(xs, ys, [0.0] * len(ys))
    axes[0].plot(
        xs_p, ys_p, color=color, linewidth=2.2, label=f'seed {i}', zorder=3)
    print(
        f'  train seed{i}: last={ys[-1]:.3f} peak={max(ys):.3f} '
        f'@{xs[-1]/1e6:.1f}M')
  for i, pts in enumerate(evals):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = seed_colors[i % len(seed_colors)]
    sm = base._plot_eval_smoothed(
        axes[1], xs, ys, color=color, label=f'seed {i}',
        linestyle='-', zorder=3, linewidth=2.2)
    bits = [f'  eval  seed{i}: last={ys[-1]:.3f} peak={max(ys):.3f}']
    if sm:
      bits.append(f'smooth last={sm[-1]:.3f}')
    print('  '.join(bits))
  _style(
      axes[0],
      title=r'c7t2  ·  KL-pen  $\beta_{\min}=0.05$  $d_{\mathrm{targ}}=0.05$  ·  train',
      ylabel='Train Success (last 1000)', xlabel=False)
  _style(
      axes[1],
      title=rf'c7t2  ·  KL-pen  ·  eval (roll mean w={w})',
      ylabel=f'Eval Success (roll mean w={w})', xlabel=True)
  leg = axes[0].legend(
      loc='lower right', fontsize=9.0, framealpha=0.95, handlelength=2.4,
      fancybox=False, edgecolor='#333333')
  fig.text(
      0.5, 0.02,
      r'KL-pen only (not averaged). $\beta_0=1$, $\beta_{\min}=0.05$, '
      r'$d_{\mathrm{targ}}=0.05$, adapt once/epoch. '
      'Base still dgr c=100 λlr=1e-6 + tKL=0.1 + extrew. '
      f'Train raw; eval faint raw + bold roll mean w={w}.',
      ha='center', va='bottom', fontsize=7.2, color='#555555')
  os.makedirs(os.path.dirname(OUT_KLPEN), exist_ok=True)
  tmp = OUT_KLPEN + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight', bbox_extra_artists=(leg,))
  os.replace(tmp, OUT_KLPEN)
  plt.close(fig)
  print(f'→ {OUT_KLPEN}')


if __name__ == '__main__':
  main()

