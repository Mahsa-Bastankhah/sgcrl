#!/usr/bin/env python3
"""Plot currently RUNNING creative7_task2 train/eval success.

Discovers squeue jobs whose log_dir is ppo_builderbench_creative7_task2_*,
and writes a dual-panel overlay plus per-family splits.

CPU-only (CSV / slurm stdout; no GPU / no rollouts).

Usage:
  python scripts/plot_c7t2_active_train_eval.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import plot_builderbench_train_success1000 as base  # noqa: E402

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

GROUP = 'creative7_task2'
PREFIX = 'ppo_builderbench_creative7_task2_'
OUT_DIR = base.FIGS_DIR
COMBINED_NAME = 'creative7_task2_running_recent_success.png'


def _c7_label(log_dir_name: str, raw_suffix: str | None = None) -> str:
  """Short legend that distinguishes the live c7t2 ablations."""
  del raw_suffix
  low = log_dir_name.lower()
  fam = base._method_family(log_dir_name)
  bits: list[str] = []
  if fam == 'crl':
    bits.append('CRL')
  elif fam == 'td3':
    bits.append('TD3')
  elif '_nf_compact_small_' in low:
    bits.append('NF compact-small')
  elif '_nf_compact_' in low:
    bits.append('NF compact')
  else:
    bits.append('NF')

  mt = re.search(r'tau(\d+)', low)
  if mt:
    raw = mt.group(1)
    tau = float(raw[0] + '.' + raw[1:]) if len(raw) > 1 else float(raw)
    if abs(tau - 0.5) > 1e-9:
      bits.append(f'τ={tau:g}')

  me = re.search(r'task\d+_e(\d+)_', low)
  if me and me.group(1) != '1024':
    bits.append(f'e{me.group(1)}')

  m = re.search(r'(?:^|_)crl(\d+)(?:_|$)', low)
  if m:
    bits.append(f'steps/iter={m.group(1)}')
  if 'h4x256' in low:
    bits.append('h4×256')
  if 'sparseafter' in low:
    bits.append('sparse-after')
  if 'freezeafter' in low or 'freezerepr' in low:
    bits.append('freeze-repr')
  if 'sfpert' in low:
    meps = re.search(r'sfpert_eps(\d+)', low)
    if meps:
      digits = meps.group(1)
      bits.append(f'SF-pert ε={float(digits) / (10 ** (len(digits) - 1)):g}')
    else:
      bits.append('SF-pert')
  if 'tol0013' in low:
    bits.append('tol=0.013')
  if 'actorreset' in low:
    bits.append('actor-reset')
  if 'normobs' in low and 'nonormobs' not in low:
    bits.append('normobs')
  ms = re.search(r'(?:^|_)seed(\d+)(?:_|$)', low)
  if ms:
    bits.append(f'seed{ms.group(1)}')
  if 'ep50' in low:
    bits.append('ep50')
  elif 'ep60' in low:
    bits.append('ep60')
  elif 'ep70' in low:
    bits.append('ep70')
  return ' · '.join(bits)


def _active_c7_runs() -> tuple[set[str], list[tuple[str, str]]]:
  active = {
      n for n in base._active_log_dirs(base.SLURM_DIR) if n.startswith(PREFIX)
  }
  runs: list[tuple[str, str]] = []
  for name in sorted(active):
    has_csv = base._has_csv(base.LOG_ROOT, name, 'learner', min_size=200)
    has_slurm = bool(base._read_train_from_slurm(base.SLURM_DIR, name))
    if not has_csv and not has_slurm:
      print(f'  skip (no train data yet): {name}')
      continue
    runs.append((name, _c7_label(name)))
  return active, runs


def _plot_dual(runs: list[tuple[str, str]], active: set[str],
               out_path: str, title: str) -> str | None:
  fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), sharey=True)
  plotted = 0
  for i, (name, label) in enumerate(runs):
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    t_seeds = base._read_train_seed_series(
        base.LOG_ROOT, name, slurm_dir=base.SLURM_DIR)
    e_seeds = base._read_eval_seed_series(
        base.LOG_ROOT, name, slurm_dir=base.SLURM_DIR)
    txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
    exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
    n_seeds = max(n_t, n_e)
    status = ' · RUNNING' if name in active else ''
    legend = f'{label} (n={n_seeds}){status}'
    if txs:
      xs, mean, se = base._subsample_curve(txs, tmean, tse)
      axes[0].plot(
          xs, mean, color=color, linewidth=2.5, label=legend, alpha=0.95,
          zorder=3 + i)
      if n_t > 1 and any(s > 0 for s in se):
        axes[0].fill_between(
            xs,
            [m - s for m, s in zip(mean, se)],
            [m + s for m, s in zip(mean, se)],
            color=color, alpha=0.18, linewidth=0, zorder=2 + i)
    if exs:
      xs, mean, se = base._subsample_curve(exs, emean, ese)
      axes[1].plot(
          xs, mean, color=color, linewidth=2.5, label=legend, alpha=0.95,
          zorder=3 + i)
      if n_e > 1 and any(s > 0 for s in se):
        axes[1].fill_between(
            xs,
            [m - s for m, s in zip(mean, se)],
            [m + s for m, s in zip(mean, se)],
            color=color, alpha=0.18, linewidth=0, zorder=2 + i)
    peak_t = max(tmean) if tmean else 0.0
    peak_e = max(emean) if emean else 0.0
    print(f'  {label}: n={n_seeds} peak_train={peak_t:.3f} peak_eval={peak_e:.3f}')
    if txs or exs:
      plotted += 1

  if plotted == 0:
    plt.close(fig)
    return None

  for ax, panel_title, ylabel in (
      (axes[0], f'{GROUP} — train success (last 1000)',
       'Train Success (last 1000)'),
      (axes[1], f'{GROUP} — eval success', 'Eval Success'),
  ):
    ax.set_title(panel_title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Env Steps', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
    ax.set_ylim(-0.05, 1.05)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

  handles, labels = axes[0].get_legend_handles_labels()
  if not handles:
    handles, labels = axes[1].get_legend_handles_labels()
  fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
  extra = ()
  if handles:
    leg = fig.legend(
        handles, labels, loc='upper left', bbox_to_anchor=(1.01, 0.98),
        fontsize=8.5, framealpha=1.0, ncol=1, edgecolor='#333333',
        fancybox=False, borderpad=0.6, handlelength=2.2, labelspacing=0.55,
        borderaxespad=0.0)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_linewidth(1.1)
    for text in leg.get_texts():
      text.set_fontweight('bold')
    extra = (leg,)
  fig.tight_layout(rect=[0, 0, 0.70 if handles else 1.0, 0.92])
  fig.savefig(out_path, dpi=150, bbox_inches='tight', bbox_extra_artists=extra)
  plt.close(fig)
  return out_path


def run() -> list[str]:
  os.makedirs(OUT_DIR, exist_ok=True)
  base._friendly_label = _c7_label
  active, runs = _active_c7_runs()
  if not runs:
    print('No running creative7_task2 jobs with train data.')
    return []
  print(f'Running creative7_task2 configs with data: {len(runs)}')
  for name, label in runs:
    print(f'  - {label}\n      {name}')

  outs: list[str] = []
  combined = os.path.join(OUT_DIR, COMBINED_NAME)
  path = _plot_dual(
      runs, active, combined,
      title='Creative 7 Task 2 — currently RUNNING jobs (mean ± stderr)',
  )
  if path:
    print(f'  → {path}')
    outs.append(path)

  by_fam = base._split_runs_by_family(runs)
  fam_dir = os.path.join(OUT_DIR, f'{GROUP}_by_method')
  os.makedirs(fam_dir, exist_ok=True)
  for fam, fam_runs in sorted(by_fam.items()):
    title = base.FAMILY_TITLE.get(fam, fam)
    out = os.path.join(fam_dir, f'{GROUP}_{fam}_running_train_eval.png')
    p = _plot_dual(
        fam_runs, active, out,
        title=f'Creative 7 Task 2 [{title}] — currently RUNNING',
    )
    if p:
      print(f'  → {p}')
      outs.append(p)
  return outs


if __name__ == '__main__':
  written = run()
  print(f'\nWrote {len(written)} figure(s).')
