#!/usr/bin/env python3
"""Train/eval success for the live c7t2 + c8t2 NF dualgradreg ablations.

Each listed slurm log is a unique config with one seed (c7t2 seed 1,
c8t2 seed 0). Legend is two lines: config knobs + job id.

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5).

  python scripts/plot_c7t2_c8t2_nf_horizon_ablation_success.py
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

C = base.ACCENT_COLORS

C7_RUNS = (
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep70_200m_crl10_dualgradreg_c100_lamlr1e6_s1',
        'c7t2 NF compact-small sa3x192 · catwp · dualgradreg c=100\n'
        'ep70 T=70 · ent=0.05 · crl10 · 200M · seed 1 · job 3748350',
        C[0],
        '-',
    ),
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep90_200m_crl10_dualgradreg_c100_lamlr1e6_s1',
        'c7t2 NF compact-small sa3x192 · catwp · dualgradreg c=100\n'
        'ep90 T=90 · ent=0.05 · crl10 · 200M · seed 1 · job 3748351',
        C[1],
        '-',
    ),
)

C8_RUNS = (
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep100_300m_crl10_dualgradreg_c100_lamlr1e6_36h',
        'c8t2 NF compact sa3x256 · catwp · dualgradreg c=100\n'
        'ep100 T=100 · ent=0.05 · crl10 · 300M · seed 0 · job 3748245',
        C[0],
        '-',
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_36h',
        'c8t2 NF compact sa3x256 · catwp · dualgradreg c=100\n'
        'ep100 T=100 · ent 0.05→0.01 · crl10 · 300M · seed 0 · job 3748244',
        C[1],
        '-',
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_T50_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_36h',
        'c8t2 NF compact sa3x256 · catwp · dualgradreg c=100\n'
        'ep100 T=50 · ent 0.05→0.01 · crl10 · 300M · seed 0 · job 3748347',
        C[2],
        '--',
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep60_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_timereg_ema_eta10',
        'c8t2 NF compact sa3x256 · catwp · dualgradreg c=100\n'
        'ep60 T=60 · time-reg EMA η=10 · ent 0.05→0.01 · 300M · seed 0 · job 3748478',
        '#2A9D8F',
        '-',
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep60_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_timereg_iter_eta20',
        'c8t2 NF compact sa3x256 · catwp · dualgradreg c=100\n'
        'ep60 T=60 · time-reg iter η=20 · ent 0.05→0.01 · 300M · seed 0 · job 3748477',
        '#C45C26',
        '--',
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep60_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_tkl005',
        'c8t2 NF compact sa3x256 · catwp · dualgradreg c=100\n'
        'ep60 T=60 · target-KL=0.05 · ent 0.05→0.01 · 300M · seed 0 · job 3748425',
        C[4],
        '-.',
    ),
)

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'figs', 'builderbench', 'c7t2_c8t2_nf_horizon_ablation_success.png')


def _plot_runs(ax_train, ax_eval, runs, *, tag: str) -> None:
  for i, (name, label, color, ls) in enumerate(runs):
    t_seeds = base._read_train_seed_series(base.LOG_ROOT, name)
    e_seeds = base._read_eval_seed_series(base.LOG_ROOT, name)
    txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
    exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
    n = max(n_t, n_e)
    if n != 1:
      print(f'  [warn] {tag}/{label}: n_seeds={n} (expected 1)')

    if txs:
      xs, mean, _se = base._subsample_curve(txs, tmean, tse)
      ax_train.plot(
          xs, mean, color=color, linestyle=ls, linewidth=2.0, alpha=0.95,
          label=label, zorder=3 + i)
    sm = []
    if exs:
      sm = base._plot_eval_smoothed(
          ax_eval, exs, emean, color=color, label=label, linestyle=ls,
          zorder=3 + i, linewidth=2.0)
    bits = [f'{tag}  {label}: n={n}']
    if tmean:
      bits.append(
          f'train last={tmean[-1]:.3f} peak={max(tmean):.3f} '
          f'@ {txs[tmean.index(max(tmean))]:.0f}')
    if emean:
      bits.append(
          f'eval raw last={emean[-1]:.3f} peak={max(emean):.3f}')
    if sm:
      bits.append(f'smooth last={sm[-1]:.3f} peak={max(sm):.3f}')
    print('  '.join(bits))


def _style_success(ax, *, ylabel: str, title: str, xlabel: bool) -> None:
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=10)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def _legend_below(ax, *, fontsize: float) -> object:
  return ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, -0.28),
      fontsize=fontsize, framealpha=0.95, handlelength=2.8,
      labelspacing=0.85, borderpad=0.55, fancybox=False,
      edgecolor='#333333',
  )


def main() -> None:
  fig, axes = plt.subplots(2, 2, figsize=(16.2, 10.4), sharex='col')
  fig.subplots_adjust(
      left=0.055, right=0.99, top=0.91, bottom=0.30,
      wspace=0.16, hspace=0.28)
  w = base.EVAL_SMOOTH_WINDOW

  _plot_runs(axes[0, 0], axes[1, 0], C7_RUNS, tag='c7t2')
  _plot_runs(axes[0, 1], axes[1, 1], C8_RUNS, tag='c8t2')

  _style_success(
      axes[0, 0],
      ylabel='Train Success (last 1000)',
      title='c7t2 compact-small · seed 1 — train',
      xlabel=False,
  )
  _style_success(
      axes[1, 0],
      ylabel=f'Eval Success (roll mean w={w})',
      title=f'c7t2 compact-small · seed 1 — eval (w={w})',
      xlabel=True,
  )
  _style_success(
      axes[0, 1],
      ylabel='Train Success (last 1000)',
      title='c8t2 compact · seed 0 — train',
      xlabel=False,
  )
  _style_success(
      axes[1, 1],
      ylabel=f'Eval Success (roll mean w={w})',
      title=f'c8t2 compact · seed 0 — eval (w={w})',
      xlabel=True,
  )

  # One legend per column (eval axes), below the plots — not covering curves.
  axes[0, 0].legend().remove()
  axes[0, 1].legend().remove()
  leg_c7 = _legend_below(axes[1, 0], fontsize=7.6)
  leg_c8 = _legend_below(axes[1, 1], fontsize=6.9)
  for text in list(leg_c7.get_texts()) + list(leg_c8.get_texts()):
    text.set_fontfamily('monospace')

  fig.suptitle(
      'NF catwp · dualgradreg c=100 λ-lr=1e-6 · nopermute+fixedx · extrew1  '
      '(one seed per config)',
      fontsize=12, fontweight='bold', y=0.98,
  )
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(
      tmp, dpi=150, bbox_inches='tight',
      bbox_extra_artists=(leg_c7, leg_c8),
  )
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
