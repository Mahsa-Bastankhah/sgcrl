#!/usr/bin/env python3
"""One-off: split creative4_task2 train/eval success into CRL / NF / TD3.

Only catselect runs (the previous combined overlay + any other catselect
dirs that still have learner data). Does not touch the general plotter.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

FIGS_DIR = base.FIGS_DIR
LOG_ROOT = base.LOG_ROOT
SLURM_DIR = base.SLURM_DIR
GROUP = 'creative4_task2'
LEGEND_FONTSIZE = 14

# Previous combined plot runs. Extra catselect dirs with data are appended
# at runtime. Note: extrew10_catselect / prenorm logs are gone from disk.
RUNS_BY_FAMILY: dict[str, list[str]] = {
    'crl': [
        'ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_catselect',
        'ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_extrew1_catselect',
        # contrastive family (still r=φ·ψ)
        'ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_catselect_extrew1',
    ],
    'nf': [
        'ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_catselect',
        'ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_catselect_extrew1',
    ],
    'td3': [
        'ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_catselect_extrew1',
    ],
}
FAMILY_TITLE = {'crl': 'CRL', 'nf': 'NF', 'td3': 'TD3'}


def _family_of(name: str) -> str | None:
  low = name.lower()
  if '_tdinfonce_' in low:
    return 'crl'
  if re.search(r'(^|_)td3(_|$)', low):
    return 'td3'
  if re.search(r'(^|_)nf(_|$)', low):
    return 'nf'
  if re.search(r'(^|_)crl(_|$)', low):
    return 'crl'
  return None


def _has_data(log_dir_name: str) -> bool:
  return bool(base._read_train_series(
      LOG_ROOT, log_dir_name, slurm_dir=SLURM_DIR))


def _plot(family: str, log_dirs: list[str], *, split: str) -> str | None:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker

  runs = [(d, base._variant_label(d)) for d in log_dirs if _has_data(d)]
  if not runs:
    print(f'  {family}/{split}: no data')
    return None

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
      f'BuilderBench Creative 4 Task 2 [{FAMILY_TITLE[family]}] — '
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
  known = {d for dirs in RUNS_BY_FAMILY.values() for d in dirs}
  for name in sorted(os.listdir(LOG_ROOT)):
    if not name.startswith('ppo_builderbench_creative4_task2'):
      continue
    if 'catselect' not in name or 'prenorm' in name:
      continue
    if name in known or not _has_data(name):
      continue
    fam = _family_of(name)
    if fam is None:
      print(f'extra catselect (unknown family, skip): {name}')
      continue
    print(f'adding extra catselect → {fam}: {name}')
    RUNS_BY_FAMILY[fam].append(name)

  outs = []
  for family in ('crl', 'nf', 'td3'):
    print(f'=== {family} ===')
    for split in ('learner', 'eval'):
      out = _plot(family, RUNS_BY_FAMILY[family], split=split)
      if out:
        print(f'  → {out}')
        outs.append(out)

  for stale in (f'{GROUP}_train_success1000.png', f'{GROUP}_eval_success.png'):
    path = os.path.join(FIGS_DIR, stale)
    if os.path.isfile(path):
      os.remove(path)
      print(f'removed combined: {path}')
  print(f'done: {len(outs)} plots')


if __name__ == '__main__':
  main()
