#!/usr/bin/env python3
"""NF binary accuracy + ‖∇_s log p‖ for the three succeeding throws.

Eval binary acc is faint raw + bold rolling mean (window=5).
Train ``nf/nf_logp_grad_s_norm_mean`` is not smoothed.

  python scripts/plot_allegro_success_nf_binacc_gradual_rew.py
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_success_nf_binacc_grad_s.png')
C = base.ACCENT_COLORS

RUNS = (
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03'
        '_nvbucket_xy045m060_ep70_1b_q40shift_noreplace_inbucket_seed0_14h',
        'hover T=70 q40 1B  in-bucket  (0.45,-0.60)',
        C[0],
        1024 * 70,
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03'
        '_nvbucket_xy050045_noshape_ep70_500m_q05shift_noreplace_inbucket_seed0_7h',
        'hover T=70 q05 500M  in-bucket  (0.50,-0.45)',
        C[2],
        1024 * 70,
    ),
    (
        'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
        '_nvbucket_xy045m060_norand_ep70_1b_q40shift_noreplace_inbucket_seed0_14h',
        'keep-arm T=70 q40 1B  in-bucket  (0.45,-0.60)',
        C[1],
        1024 * 70,
    ),
)


def _eval_csv(log_dir: str) -> str:
  root = os.path.join(base.LOG_ROOT, log_dir)
  if not os.path.isdir(root):
    return ''
  for name in sorted(os.listdir(root)):
    cand = os.path.join(root, name, 'logs', 'eval', 'logs.csv')
    if os.path.isfile(cand):
      return cand
  return ''


def _read_binacc(log_dir: str, steps_per_iter: int):
  path = _eval_csv(log_dir)
  if not path:
    return [], [], []
  xs, pol, rnd = [], [], []
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      it = base._coerce(row.get('iteration', ''))
      yp = base._coerce(row.get('nf_binary_accuracy', ''))
      yr = base._coerce(row.get('nf_binary_accuracy_random', ''))
      if it is None or yp is None or yp != yp:
        continue
      xs.append(int(it) * int(steps_per_iter))
      pol.append(float(yp))
      rnd.append(float(yr) if yr is not None and yr == yr else float('nan'))
  return xs, pol, rnd


def _style(ax, ylabel: str, *, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=10)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(11.2, 8.4), sharex=True)
  for log_dir, label, color, spi in RUNS:
    xs, pol, rnd = _read_binacc(log_dir, spi)
    if xs:
      base._plot_eval_smoothed(
          axes[0], xs, pol, color=color, linestyle='-',
          label=f'{label}  π')
      if any(y == y for y in rnd):
        base._plot_eval_smoothed(
            axes[0], xs, rnd, color=color, linestyle='--',
            label=f'{label}  random a', linewidth=1.4)
      print(f'{label} binacc π last={pol[-1]:.3f} peak={max(pol):.3f} '
            f'rand last={rnd[-1]:.3f} n={len(xs)}', flush=True)
    else:
      print(f'{label}: no nf_binary_accuracy', flush=True)

    series = base._read_csv_seed_series(
        base.LOG_ROOT, log_dir, split='learner',
        x_col='global_step', y_col='nf/nf_logp_grad_s_norm_mean')
    tx, tmean, tse, tn = base._aggregate_mean_stderr(series)
    if tx:
      tx, tmean, tse = base._subsample_curve(tx, tmean, tse)
      axes[1].plot(tx, tmean, color=color, lw=1.6, label=label)
      print(f'{label} ||∇_s log p|| last={tmean[-1]:.3f} n={tn}', flush=True)
    else:
      print(f'{label}: no nf/nf_logp_grad_s_norm_mean', flush=True)

  axes[0].axhline(0.5, color='0.45', lw=1.0, ls=':', zorder=1)
  axes[0].set_title(
      'NF binary acc  P[log p(g+|s,a) > log p(g-|s,a)]  '
      f'(faint raw + bold rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=12, fontweight='bold')
  _style(axes[0], f'binary acc (roll mean, w={base.EVAL_SMOOTH_WINDOW})')
  axes[0].set_ylim(0.35, 1.02)
  axes[0].legend(loc='lower right', fontsize=7.4, framealpha=0.95, ncol=2)

  axes[1].set_title(
      r'mean $\|\nabla_s \log p(g\mid s,a)\|$  (train, unsmoothed)',
      fontsize=12, fontweight='bold')
  _style(axes[1], r'mean $\|\nabla_s \log p\|$', xlabel=True)
  axes[1].legend(loc='upper right', fontsize=7.6, framealpha=0.95)

  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=160, bbox_inches='tight')
  old = os.path.join(
      os.path.dirname(OUT_PATH), 'akt_success_nf_binacc_gradual_rew.png')
  fig.savefig(old, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {OUT_PATH}', flush=True)
  print(f'→ {old}', flush=True)


if __name__ == '__main__':
  main()
