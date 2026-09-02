#!/usr/bin/env python3
"""Hard success for the live c7t2 tKL pair + the c8t2 tKL=0.1 warp job.

c7t2 is a matched tKL ablation (0.05 vs 0.10) at λlr=1e-2 compact-small.
c8t2 is a different task/net/λlr/ent (not a matched ablation of the c7 pair).

Train: raw ``train_success_1000``. Eval: faint raw + bold rolling mean
(window=5).

  python scripts/plot_c7t2_c8t2_tkl_lamlr_live_success.py
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
FIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'figs', 'builderbench')
OUT = os.path.join(FIGS, 'c7t2_c8t2_tkl_lamlr_live_success.png')

# (log_dir_name, legend, color, linestyle)
C7_RUNS = (
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep70_200m_crl10_dualgradreg_c100_lamlr1e2_tkl005_s1',
        'tKL=0.05   λlr=1e-2   compact-small sa3x192\n'
        'ent=0.05 fixed   ep=70   seed 1   3749899 TIMEOUT',
        C[2],
        '-',
    ),
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep70_200m_crl10_dualgradreg_c100_lamlr1e2_tkl01'
        '_s2_warp_2h30',
        'tKL=0.10   λlr=1e-2   compact-small sa3x192\n'
        'ent=0.05 fixed   ep=70   seed 2   3752999 COMPLETED',
        C[0],
        '-',
    ),
)
C8_RUNS = (
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_tkl01_warp_2h30',
        'tKL=0.10   λlr=1e-6   compact sa3x256\n'
        'ent 0.05→0.01   ep=100   seed 0   3752549 TIMEOUT',
        C[1],
        '-',
    ),
)
PANELS = (
    ('c7t2  ·  tKL ablation  ·  λlr=1e-2', C7_RUNS),
    ('c8t2  ·  tKL=0.10  ·  λlr=1e-6', C8_RUNS),
)


def _load(name: str, kind: str):
  if kind == 'train':
    seeds = base._read_train_seed_series(base.LOG_ROOT, name)
  else:
    seeds = base._read_eval_seed_series(base.LOG_ROOT, name)
  return base._aggregate_mean_stderr(seeds)


def _plot_runs(ax_train, ax_eval, runs, *, tag: str) -> None:
  for i, (name, label, color, ls) in enumerate(runs):
    txs, tmean, tse, n_t = _load(name, 'train')
    exs, emean, ese, n_e = _load(name, 'eval')
    n = max(n_t, n_e)
    if n != 1:
      print(f'  [warn] {tag} n_seeds={n}  {label.split(chr(10))[0]}')
    if txs:
      xs, mean, _ = base._subsample_curve(txs, tmean, tse)
      ax_train.plot(
          xs, mean, color=color, linestyle=ls, linewidth=2.2, alpha=0.95,
          label=label, zorder=3 + i)
    sm = []
    if exs:
      sm = base._plot_eval_smoothed(
          ax_eval, exs, emean, color=color, label=label, linestyle=ls,
          zorder=3 + i, linewidth=2.2)
    bits = [f'{tag}  {label.split(chr(10))[-1]}: n={n}']
    if tmean:
      peak_i = tmean.index(max(tmean))
      bits.append(
          f'train last={tmean[-1]:.3f} peak={max(tmean):.3f} '
          f'@ {txs[peak_i] / 1e6:.1f}M')
    if emean:
      bits.append(f'eval raw last={emean[-1]:.3f} peak={max(emean):.3f}')
    if sm:
      bits.append(f'smooth last={sm[-1]:.3f} peak={max(sm):.3f}')
    print('  '.join(bits))


def _style(ax, *, title: str, ylabel: str, xlabel: bool) -> None:
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=10)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def _legend_below(ax, *, fontsize: float):
  return ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, -0.26),
      fontsize=fontsize, framealpha=0.95, handlelength=2.8,
      labelspacing=0.75, borderpad=0.5, fancybox=False,
      edgecolor='#333333',
  )


def main() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.2), sharex='col', sharey=True)
  fig.subplots_adjust(
      left=0.055, right=0.99, top=0.88, bottom=0.28,
      wspace=0.10, hspace=0.28)

  _plot_runs(axes[0, 0], axes[1, 0], C7_RUNS, tag='c7t2')
  _plot_runs(axes[0, 1], axes[1, 1], C8_RUNS, tag='c8t2')

  _style(
      axes[0, 0], title='c7t2 train',
      ylabel='Train Success (last 1000)', xlabel=False)
  _style(
      axes[1, 0], title=f'c7t2 eval (roll mean w={w})',
      ylabel=f'Eval Success (roll mean w={w})', xlabel=True)
  _style(
      axes[0, 1], title='c8t2 train',
      ylabel='Train Success (last 1000)', xlabel=False)
  _style(
      axes[1, 1], title=f'c8t2 eval (roll mean w={w})',
      ylabel=f'Eval Success (roll mean w={w})', xlabel=True)
  axes[0, 1].set_ylabel('')
  axes[1, 1].set_ylabel('')

  axes[0, 0].legend().remove()
  axes[0, 1].legend().remove()
  leg_c7 = _legend_below(axes[1, 0], fontsize=7.8)
  leg_c8 = _legend_below(axes[1, 1], fontsize=7.8)
  for text in list(leg_c7.get_texts()) + list(leg_c8.get_texts()):
    text.set_fontfamily('monospace')

  fig.suptitle(
      'NF catwp · dualgradreg c=100 · nopermute+fixedx · extrew1\n'
      'c7t2: tKL 0.05 vs 0.10 (same λlr / net / ent / ep).  '
      'c8t2: different task, bigger net, λlr=1e-6, ent anneal, ep=100.',
      fontsize=12, fontweight='bold', y=0.99,
  )
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(
      tmp, dpi=150, bbox_inches='tight',
      bbox_extra_artists=(leg_c7, leg_c8),
  )
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
