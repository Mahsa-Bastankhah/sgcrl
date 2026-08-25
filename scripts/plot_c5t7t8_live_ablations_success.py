#!/usr/bin/env python3
"""Hard success for currently running / just-cancelled c7t2 and c8t2 jobs.

Two figures (train vs eval), each faceted by task. Train is raw
``train_success_1000``. Eval is faint raw + bold rolling mean (window=5).

  python scripts/plot_c5t7t8_live_ablations_success.py
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
TRAIN_OUT = os.path.join(FIGS, 'c7t2_c8t2_live_ablations_train_success.png')
EVAL_OUT = os.path.join(FIGS, 'c7t2_c8t2_live_ablations_eval_success.png')

# Currently RUNNING, plus 3748244 cancelled this morning (~09:29).
# Pending 3751949 / 3751950 (c7 timereg warp 4h) have no logs yet.
# (log_dir_name, legend, color, linestyle)
C7_RUNS = (
    (
        'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
        '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep70_200m_crl10_dualgradreg_c100_lamlr1e2_tkl005_s1',
        'NF compact-small · catwp · dualgradreg\n'
        r'ep70 · ent=0.05 · $\lambda$lr=1e-2 · tKL=0.05 · RUNNING 3749899',
        C[2],
        '-',
    ),
    (
        'mpo_crl_builderbench_creative7_task2_e1024_pd_fast_succ',
        'MPO-CRL  ·  T=90  ·  RUNNING 3749778',
        C[0],
        '--',
    ),
)
C8_RUNS = (
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_36h',
        'NF compact · catwp · dualgradreg c=100\n'
        'T=100 · ent 0.05→0.01 · cancelled 09:29  ·  3748244',
        C[1],
        '-',
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep100_300m_crl10_dualgradreg_c100_lamlr1e6_36h',
        'NF compact · catwp · dualgradreg c=100\n'
        'T=100 · ent=0.05 fixed  ·  RUNNING 3748245',
        C[0],
        '-',
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_T50_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_36h',
        'NF compact · catwp · dualgradreg c=100\n'
        'T=50 · ent 0.05→0.01  ·  RUNNING 3748347',
        C[2],
        '--',
    ),
)
PANELS = (
    ('c7t2', C7_RUNS),
    ('c8t2', C8_RUNS),
)


def _load(name: str, kind: str):
  if kind == 'train':
    seeds = base._read_train_seed_series(base.LOG_ROOT, name)
  else:
    seeds = base._read_eval_seed_series(base.LOG_ROOT, name)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  return xs, mean, se, n


def _plot_panel(ax, runs, *, kind: str, tag: str) -> None:
  for i, (name, label, color, ls) in enumerate(runs):
    xs, mean, se, n = _load(name, kind)
    if n != 1:
      print(f'  [warn] {tag}/{kind} {label.split(chr(10))[0]}: n_seeds={n}')
    if not xs:
      print(f'  [warn] {tag}/{kind} no data for {name}')
      continue
    if kind == 'train':
      xs_p, mean_p, _ = base._subsample_curve(xs, mean, se)
      ax.plot(
          xs_p, mean_p, color=color, linestyle=ls, linewidth=2.0,
          alpha=0.95, label=label, zorder=3 + i)
      sm = mean
    else:
      sm = base._plot_eval_smoothed(
          ax, xs, mean, color=color, label=label, linestyle=ls,
          zorder=3 + i, linewidth=2.0)
    bits = [f'{tag} {kind}  {label.split(chr(10))[-1]}: n={n}']
    bits.append(f'last={mean[-1]:.3f} peak={max(mean):.3f}')
    if kind == 'eval' and sm:
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
      labelspacing=0.75, borderpad=0.5, fancybox=False,
      edgecolor='#333333',
  )


def _make_figure(kind: str, out_path: str) -> None:
  w = base.EVAL_SMOOTH_WINDOW
  if kind == 'train':
    ylabel = 'Train Success (last 1000)'
    sup = (
        'Hard train success (last 1000)  ·  '
        'currently running / cancelled this morning'
    )
    legend_fs = (7.2, 6.8)
  else:
    ylabel = f'Eval Success (roll mean w={w})'
    sup = (
        f'Hard eval success (rolling mean, window={w})  ·  '
        'currently running / cancelled this morning'
    )
    legend_fs = (7.2, 6.8)

  fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), sharey=True)
  fig.subplots_adjust(
      left=0.06, right=0.995, top=0.84, bottom=0.40, wspace=0.12)

  legs = []
  for ax, (tag, runs), fs in zip(axes, PANELS, legend_fs):
    _plot_panel(ax, runs, kind=kind, tag=tag)
    _style(ax, title=tag, ylabel=ylabel if ax is axes[0] else '')
    if ax is not axes[0]:
      ax.set_ylabel('')
    legs.append(_legend_below(ax, fontsize=fs))
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
