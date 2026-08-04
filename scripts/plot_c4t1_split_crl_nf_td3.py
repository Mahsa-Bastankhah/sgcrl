#!/usr/bin/env python3
"""One-off: split creative4_task1 train/eval success into CRL / NF / TD3 plots.

Only the runs that were on the previous combined creative4_task1 plot.
Does not modify scripts/plot_builderbench_train_success1000.py or other tasks.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

FIGS_DIR = base.FIGS_DIR
LOG_ROOT = base.LOG_ROOT
SLURM_DIR = base.SLURM_DIR
GROUP = 'creative4_task1'
LEGEND_FONTSIZE = 11  # larger than the combined-plot default

# Exact log-dir names that were on the previous creative4_task1 overlay.
RUNS_BY_FAMILY = {
    'crl': [
        'ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_actorreset_nopermute_evalvid_catselect',
        'ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_actorreset_nopermute_normobs_evalvid_catselect',
        'ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_actorreset_nopermute_normobs_evalvid_catselect_minstd1e4_entanneal',
        'ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_nopermute_catselect_fixedx01',
    ],
    'nf': [
        'ppo_builderbench_creative4_task1_e1024_pd_nf_tau05_actorreset_nopermute_evalvid_catselect',
    ],
    'td3': [
        'ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset_nopermute_evalvid_catselect',
        'ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset_nopermute_normobs_evalvid_catselect',
        'ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_catselect_extrew1',
    ],
}
FAMILY_TITLE = {'crl': 'CRL', 'nf': 'NF', 'td3': 'TD3'}


def _plot(family: str, log_dirs: list[str], *, split: str) -> str | None:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker

  runs = [(d, base._variant_label(d)) for d in log_dirs]
  fig, ax = plt.subplots(figsize=(9.5, 4.8))
  plotted = 0
  for i, (log_dir_name, label) in enumerate(runs):
    if split == 'eval':
      pts = base._subsample(base._read_eval_series(
          LOG_ROOT, log_dir_name, slurm_dir=SLURM_DIR))
      out_suffix = 'eval_success'
      title_metric = f'eval {base.EVAL_METRIC}'
      ylabel = 'Eval Success'
      use_markers = True
    else:
      pts = base._subsample(base._read_train_series(
          LOG_ROOT, log_dir_name, slurm_dir=SLURM_DIR))
      out_suffix = 'train_success1000'
      title_metric = f'train {base.TRAIN_METRIC}'
      ylabel = 'Train Success (last 1000)'
      use_markers = False
    if not pts:
      print(f'  skip {label}: no {split} data')
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    kwargs = dict(
        color=color, linewidth=2.0, label=f'{label} (n={len(pts)})',
        alpha=0.95, zorder=3 + i,
    )
    if use_markers:
      kwargs.update(
          marker=base.EVAL_MARKERS[i % len(base.EVAL_MARKERS)],
          markersize=7,
          linestyle=base.EVAL_LINESTYLES[i % len(base.EVAL_LINESTYLES)],
          markerfacecolor=color,
          markeredgecolor='white',
          markeredgewidth=0.6,
      )
    ax.plot(xs, ys, **kwargs)
    print(f'  {family}/{label}: {len(pts)} {split} pts')
    plotted += 1

  if plotted == 0:
    plt.close(fig)
    return None

  ax.set_title(
      f'BuilderBench Creative 4 Task 1 [{FAMILY_TITLE[family]}] — '
      f'{title_metric}',
      fontsize=12, fontweight='bold',
  )
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel(ylabel, fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='best', fontsize=LEGEND_FONTSIZE, framealpha=0.95, ncol=1)

  out_path = os.path.join(FIGS_DIR, f'{GROUP}_{family}_{out_suffix}.png')
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  return out_path


def main():
  os.makedirs(FIGS_DIR, exist_ok=True)
  outs = []
  for family in ('crl', 'nf', 'td3'):
    log_dirs = RUNS_BY_FAMILY[family]
    print(f'=== {family} ({len(log_dirs)} runs) ===')
    for split in ('learner', 'eval'):
      out = _plot(family, log_dirs, split=split)
      if out:
        print(f'  → {out}')
        outs.append(out)
  # Remove stale combined overlay for this task only (family plots replace it).
  for stale in (
      f'{GROUP}_train_success1000.png',
      f'{GROUP}_eval_success.png',
  ):
    path = os.path.join(FIGS_DIR, stale)
    if os.path.isfile(path):
      os.remove(path)
      print(f'removed combined: {path}')
  print(f'done: {len(outs)} plots')


if __name__ == '__main__':
  main()
