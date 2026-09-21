#!/usr/bin/env python3
"""NF binary accuracy + grad-reg raw for successful-candidate z=0.55 runs.

Eval binary acc: faint raw + bold rolling mean (window=5).
Train ``nf/nf_grad_reg_raw`` is not smoothed.

  python scripts/plot_allegro_successful_candidate_nf_diag.py
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
import plot_allegro_hidetable_running_success as hidetable  # noqa: E402
import plot_allegro_successful_candidate as cand  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_successful_candidate_nf_diag.png')
EVAL_WINDOW = hidetable.EVAL_WINDOW

# Same colors / labels as the success figure. Only the z=0.55 runs listed.
RUNS = (
    (
        'hard hover  z=0.55 T=80 q40  s0  (0.45,-0.60)',
        'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
        cand.C[3],
        '--',
    ),
    (
        'hard hover  z=0.55 T=80 q40  s1  (0.45,-0.60)',
        'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed1_4h',
        cand.C[7],
        '--',
    ),
    (
        'hard hover  z=0.55 T=80 q40  s2  (0.45,-0.60)',
        'successful_candidate/hard_bucket_hover_near10_z055_ep80_q40_300m_seed2_4h',
        cand.C[11],
        '--',
    ),
    (
        'mid hover  z=0.55 T=80 q40  300M  (0.50,-0.45)',
        'successful_candidate/mid_bucket_hover_near10_z055_ep80_q40_300m_seed0_4h',
        cand.C[0],
        '--',
    ),
    (
        'hard arm-init  z=0.55 T=80 q40  300M  (0.45,-0.60)',
        'successful_candidate/hard_bucket_arm_init_z055_ep80_q40_300m_seed0_4h',
        cand.C[1],
        '--',
    ),
)


def _read_eval_metric(log_dir: str, y_col: str):
  series = []
  for d in cand._expand_log_dirs(log_dir):
    raw = base._read_csv_seed_series(
        base.LOG_ROOT, d, split='eval',
        x_col='iteration', y_col=y_col)
    series.extend(
        base._iters_to_env_steps(pts, base.LOG_ROOT, d) for pts in raw)
  return series


def _read_learner_metric(log_dir: str, y_col: str):
  series = []
  for d in cand._expand_log_dirs(log_dir):
    series.extend(base._read_csv_seed_series(
        base.LOG_ROOT, d, split='learner',
        x_col='global_step', y_col=y_col))
  return series


def run_once(out_path: str = OUT_PATH) -> str:
  fig, axes = plt.subplots(2, 1, figsize=(12.0, 9.0), sharex=True)
  for label, log_dir, color, ls in RUNS:
    print(f'{label}  {log_dir}', flush=True)
    acc = _read_eval_metric(log_dir, 'nf_binary_accuracy')
    ax, am, ase, an = base._aggregate_mean_stderr(acc)
    if ax:
      base._plot_eval_smoothed(
          axes[0], ax, am, color=color, linestyle=ls, label=label)
      print(f'{label} binacc: n={an} last={am[-1]:.3f} peak={max(am):.3f}',
            flush=True)
    else:
      print(f'{label} binacc: no data', flush=True)

    grad = _read_learner_metric(log_dir, 'nf/nf_grad_reg_raw')
    gx, gm, gse, gn = base._aggregate_mean_stderr(grad)
    if gx:
      gx, gm, gse = base._subsample_curve(gx, gm, gse)
      axes[1].plot(gx, gm, color=color, lw=1.7, linestyle=ls, label=label)
      if gn > 1 and any(s > 0 for s in gse):
        lo = [m - s for m, s in zip(gm, gse)]
        hi = [m + s for m, s in zip(gm, gse)]
        axes[1].fill_between(gx, lo, hi, color=color, alpha=0.12, linewidth=0)
      print(f'{label} grad_reg_raw: n={gn} last={gm[-1]:.3f} peak={max(gm):.3f}',
            flush=True)
    else:
      print(f'{label} grad_reg_raw: no data', flush=True)

  axes[0].set_title(
      f'successful candidates — NF binary accuracy  '
      f'(faint raw + bold rolling mean, window={EVAL_WINDOW})',
      fontsize=12, fontweight='bold')
  hidetable._style(axes[0], f'NF binary acc (roll mean, w={EVAL_WINDOW})')
  axes[0].set_ylim(0.45, 1.02)
  axes[0].axhline(0.5, color='#888888', lw=0.8, ls=':')
  axes[0].legend(loc='lower right', fontsize=7.4, framealpha=0.95)

  axes[1].set_title(
      r'NF grad-reg raw   mean$(\max(\|\nabla_s \log p\|-c,0))$  (λ=0)',
      fontsize=12, fontweight='bold')
  axes[1].set_ylabel('nf/nf_grad_reg_raw', fontsize=10)
  axes[1].spines[['top', 'right']].set_visible(False)
  axes[1].grid(axis='y', linestyle='--', alpha=0.4)
  axes[1].set_xlabel('Env Steps', fontsize=10)
  axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  axes[1].legend(loc='upper left', fontsize=7.4, framealpha=0.95)

  fig.tight_layout()
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  fig.savefig(out_path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}', flush=True)
  return out_path


def main() -> None:
  run_once()


if __name__ == '__main__':
  main()
