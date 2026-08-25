#!/usr/bin/env python3
"""Best held c7t2 / c8t2 configs + the closest variants.

Winner families (held 2cm train success):
  c7: NF compact sa3x256 catwp ent-anneal ep70 (2 seeds)
  c8: NF compact sa3x256 catwp dualgradreg c=100  T=100  ent 0.05→0.01

Train: raw ``train_success_1000`` (mean ±1 SE if n>1). Eval: faint raw +
bold rolling mean (window=5).

  python scripts/plot_c7t2_c8t2_best_config_success.py
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
TRAIN_OUT = os.path.join(FIGS, 'c7t2_c8t2_best_config_train_success.png')
EVAL_OUT = os.path.join(FIGS, 'c7t2_c8t2_best_config_eval_success.png')

C7 = (
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_entanneal_ep70_300m',
        'NF compact sa3x256 · catwp · ent anneal · ep70\n'
        '2 seeds · held  (best c7)',
        C[2],
        '-',
        2.4,
    ),
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_entanneal_ep60_300m_crl10',
        'same net/catwp/ent · ep60 T=60 crl10\n'
        '2 seeds · peaked then collapsed',
        C[0],
        '--',
        1.7,
    ),
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_entanneal_ep70_300m',
        'compact-small sa3x192 · catwp · ent anneal · ep70\n'
        '1 seed · peaked then collapsed',
        C[1],
        ':',
        1.7,
    ),
    (
        'ppo_builderbench_creative7_task2_e1024_pd_crl_tau05'
        '_nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60'
        '_200m_crl10_gradreg_vhextrew1',
        'CRL catwp · vhextrew1 · gradreg · ep60\n'
        '1 seed · still held at stop',
        C[4],
        '-.',
        1.7,
    ),
)
C8 = (
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_36h',
        'NF compact · catwp · dualgradreg c=100\n'
        'ep100 T=100 · ent 0.05→0.01 · 36h  (best c8)',
        C[1],
        '-',
        2.4,
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100'
        '_lamlr1e6',
        'same flags · shorter job (stopped ~125M)\n'
        'still climbing',
        C[1],
        '--',
        1.7,
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep100_300m_crl10_dualgradreg_c100_lamlr1e6_36h',
        'same except ent=0.05 fixed · T=100',
        C[0],
        '-',
        1.8,
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_T50_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_36h',
        'same except T=50 (collapsed)',
        C[2],
        ':',
        1.7,
    ),
)
PANELS = (('c7t2', C7), ('c8t2', C8))


def _load_train(name: str):
  seeds = base._read_train_seed_series(base.LOG_ROOT, name)
  return base._aggregate_mean_stderr(seeds)


def _load_eval_seeds(name: str):
  """Per-seed eval series. Do not interpolate onto a union grid — eval is
  a handful of 0/1 checkpoints and interp fabricates mid values."""
  return [pts for pts in base._read_eval_seed_series(base.LOG_ROOT, name) if pts]


def _plot_panel(ax, runs, *, kind: str, tag: str) -> None:
  for i, (name, label, color, ls, lw) in enumerate(runs):
    if kind == 'train':
      xs, mean, se, n = _load_train(name)
      if not xs:
        print(f'  [warn] {tag}/{kind} no data {name[-40:]}')
        continue
      xs_p, mean_p, se_p = base._subsample_curve(xs, mean, se)
      if n > 1:
        lo = [m - s for m, s in zip(mean_p, se_p)]
        hi = [m + s for m, s in zip(mean_p, se_p)]
        ax.fill_between(xs_p, lo, hi, color=color, alpha=0.18, lw=0, zorder=2 + i)
      ax.plot(
          xs_p, mean_p, color=color, linestyle=ls, linewidth=lw,
          alpha=0.95, label=label, zorder=3 + i)
      print(f'{tag} {kind} n={n}  {label.split(chr(10))[0]}  '
            f'last={mean[-1]:.3f} peak={max(mean):.3f}')
      continue

    seed_series = _load_eval_seeds(name)
    if not seed_series:
      print(f'  [warn] {tag}/{kind} no data {name[-40:]}')
      continue
    n = len(seed_series)
    for si, pts in enumerate(seed_series):
      xs = [p[0] for p in pts]
      ys = [p[1] for p in pts]
      lab = label if si == 0 else None
      sm = base._plot_eval_smoothed(
          ax, xs, ys, color=color, label=lab, linestyle=ls,
          zorder=3 + i, linewidth=lw)
      bits = [f'{tag} {kind} seed{si}/{n}  {label.split(chr(10))[0]}']
      bits.append(f'last={ys[-1]:.3f} peak={max(ys):.3f}')
      if sm:
        bits.append(f'smooth last={sm[-1]:.3f} peak={max(sm):.3f}')
      print('  '.join(bits))


def _style(ax, *, title: str, ylabel: str) -> None:
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=10)
  ax.set_xlabel('Env Steps', fontsize=10)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)


def _legend_below(ax, *, fontsize: float):
  return ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, -0.22),
      fontsize=fontsize, framealpha=0.95, handlelength=2.8,
      labelspacing=0.8, borderpad=0.5, fancybox=False,
      edgecolor='#333333',
  )


def _make_figure(kind: str, out_path: str) -> None:
  w = base.EVAL_SMOOTH_WINDOW
  if kind == 'train':
    ylabel = 'Train Success (last 1000)'
    sup = 'Hard train success — best held configs + closest variants'
  else:
    ylabel = f'Eval Success (roll mean w={w})'
    sup = (
        f'Hard eval success (rolling mean, window={w}) — '
        'best held configs + closest variants'
    )

  fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.2), sharey=True)
  fig.subplots_adjust(
      left=0.055, right=0.995, top=0.84, bottom=0.42, wspace=0.12)
  legs = []
  for ax, (tag, runs) in zip(axes, PANELS):
    _plot_panel(ax, runs, kind=kind, tag=tag)
    _style(ax, title=tag, ylabel=ylabel if ax is axes[0] else '')
    if ax is not axes[0]:
      ax.set_ylabel('')
    legs.append(_legend_below(ax, fontsize=7.0))
    for text in legs[-1].get_texts():
      text.set_fontfamily('monospace')

  fig.suptitle(sup, fontsize=12, fontweight='bold', y=0.98)
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  tmp = out_path + '.tmp.png'
  fig.savefig(
      tmp, dpi=150, bbox_inches='tight', bbox_extra_artists=tuple(legs))
  os.replace(tmp, out_path)
  plt.close(fig)
  print(f'→ {out_path}')


def main() -> None:
  _make_figure('train', TRAIN_OUT)
  _make_figure('eval', EVAL_OUT)


if __name__ == '__main__':
  main()
