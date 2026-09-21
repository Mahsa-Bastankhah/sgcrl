#!/usr/bin/env python3
"""Train/eval success for live Allegro throw jobs, grouped by goal.

Seeds that share a recipe are mean ± SE. Train is raw last-1000. Eval is
faint raw + bold rolling mean (window=5).

  python scripts/plot_allegro_hidetable_running_success.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_hidetable_running_success.png')
EVAL_WINDOW = 5
C = base.ACCENT_COLORS

# Live / pending train jobs only. Same goal shares a color; NF solid, RND dashed.
# Seeds of one recipe are mean ± SE.
METHODS = (
    (
        'NF NVIDIA-init  (0.50,-0.30)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
            '_nvbucket_xy050030_noshape_ep50_300m_q05shift_noreplace_inbucket_seed0_4h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
            '_nvbucket_xy050030_noshape_ep50_300m_q05shift_noreplace_inbucket_seed1_4h',
        ),
        'nf',
        C[1],
    ),
    (
        'NF NVIDIA-init T=150 fut50  (0.50,-0.30)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
            '_nvbucket_xy050030_noshape_ep150_fut50_300m_q05shift_noreplace_inbucket_seed0_4h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
            '_nvbucket_xy050030_noshape_ep150_fut50_300m_q05shift_noreplace_inbucket_seed1_4h',
        ),
        'nf',
        C[6],
    ),
    (
        'NF NVIDIA-init T=100 fut50 1B  (0.50,-0.30)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
            '_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed0_14h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
            '_nvbucket_xy050030_noshape_ep100_fut50_1b_q05shift_noreplace_inbucket_seed1_14h',
        ),
        'nf',
        C[3],
    ),
    (
        'NF near-hover  (0.50,-0.30)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03'
            '_nvbucket_xy050030_noshape_ep50_300m_q05shift_noreplace_inbucket_seed0_4h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03'
            '_nvbucket_xy050030_noshape_ep50_300m_q05shift_noreplace_inbucket_seed1_4h',
        ),
        'nf',
        C[0],
    ),
    (
        'RND near-hover  (0.50,-0.30)',
        (
            'ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03'
            '_nvbucket_xy050030_noshape_ep50_300m_inbucket_seed0_4h',
        ),
        'rnd',
        C[0],
    ),
    (
        'NF keep-arm  (0.45,-0.60)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_nvbucket_xy045m060_norand_ep50_300m_q05shift_noreplace_inbucket_seed0_4h',
        ),
        'nf',
        C[4],
    ),
    (
        'RND keep-arm  (0.45,-0.60)',
        (
            'ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_nvbucket_xy045m060_norand_ep50_200m_inbucket_seed0_4h',
        ),
        'rnd',
        C[4],
    ),
    (
        'NF keep-arm  (0.00,-0.80)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_nvbucket_xy000m080_norand_ep50_300m_q05shift_noreplace_inbucket_seed0_4h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_nvbucket_xy000m080_norand_ep50_300m_q05shift_noreplace_inbucket_seed1_4h',
        ),
        'nf',
        C[5],
    ),
    (
        'NF hover T=70 7.5cm  (0.45,-0.60)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03'
            '_nvbucket_xy045m060_ep70_200m_q40shift_noreplace_goalball75_seed0_3h',
        ),
        'nf',
        C[11],
    ),
    (
        'NF hover T=70  (0.45,-0.60)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03'
            '_nvbucket_xy045m060_ep70_1b_q40shift_noreplace_inbucket_seed0_14h',
        ),
        'nf',
        C[2],
    ),
    (
        'RND hover T=70 200M  (0.45,-0.60)',
        (
            'ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03'
            '_nvbucket_xy045m060_noshape_ep70_200m_inbucket_seed0_3h',
        ),
        'rnd',
        C[2],
    ),
    (
        'NF hover T=70  (0.50,-0.45)',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_tablespawn_near10_above10_corr03'
            '_nvbucket_xy050045_noshape_ep70_500m_q05shift_noreplace_inbucket_seed0_7h',
        ),
        'nf',
        C[10],
    ),
    (
        'RND hover T=70 200M  (0.50,-0.45)',
        (
            'ppo_rnd_allegro_kuka_throw_e1024_tablespawn_near10_above10_corr03'
            '_nvbucket_xy050045_noshape_ep70_200m_inbucket_seed0_3h',
        ),
        'rnd',
        C[10],
    ),
)


def _read_rnd_eval(log_dir: str):
  root = base.resolve_run_dir(base.LOG_ROOT, log_dir)
  candidates = [
      os.path.join(root, 'eval_metrics.csv'),
  ]
  if os.path.isdir(root):
    for name in sorted(os.listdir(root)):
      cand = os.path.join(root, name, 'eval_metrics.csv')
      candidates.append(cand)
  pts = []
  for path in candidates:
    if not os.path.isfile(path):
      continue
    with open(path, newline='') as fh:
      for row in csv.DictReader(fh):
        x = base._coerce(row.get('env_steps', ''))
        y = base._coerce(row.get('eval/episode_success', ''))
        if x is not None and y is not None and y == y:
          pts.append((int(x), float(y)))
    if pts:
      break
  return pts


def _style(ax, ylabel: str, *, xlabel: bool = False) -> None:
  ax.set_ylabel(ylabel, fontsize=10)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.set_ylim(-0.05, 1.05)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))


def _ls(kind: str) -> str:
  if kind == 'rnd':
    return '--'
  if kind == 'mpo':
    return '-.'
  return '-'


def _plot_train(ax, xs, mean, se, n, color, label, ls='-'):
  xs, mean, se = base._subsample_curve(xs, mean, se)
  ax.plot(xs, mean, color=color, lw=2.0, linestyle=ls, label=f'{label}  (n={n})')
  if n > 1 and any(s > 0 for s in se):
    lo = [m - s for m, s in zip(mean, se)]
    hi = [m + s for m, s in zip(mean, se)]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0)
  print(f'{label} train: n={n} last={mean[-1]:.4f} peak={max(mean):.4g} '
        f'steps={xs[-1]/1e6:.1f}M')


def _plot_eval(ax, xs, mean, se, n, color, label, ls='-'):
  sm = base._plot_eval_smoothed(
      ax, xs, mean, color=color,
      label=f'{label}  (n={n}, roll mean w={EVAL_WINDOW})',
      linestyle=ls,
      window=EVAL_WINDOW)
  if n > 1 and any(s > 0 for s in se):
    lo = base._rolling_mean([m - s for m, s in zip(mean, se)], window=EVAL_WINDOW)
    hi = base._rolling_mean([m + s for m, s in zip(mean, se)], window=EVAL_WINDOW)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
  ys = sm if sm else mean
  print(f'{label} eval: n={n} last={ys[-1]:.4f} peak={max(ys):.4g}')


def run_once() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(11.2, 9.2), sharex=True)
  for label, log_dirs, kind, color in METHODS:
    ls = _ls(kind)
    if kind == 'rnd':
      print(f'{label} train: RND has no train_success_1000 (eval only)')
    else:
      train = []
      for log_dir in log_dirs:
        train.extend(base._read_train_seed_series(base.LOG_ROOT, log_dir))
      tx, tmean, tse, tn = base._aggregate_mean_stderr(train)
      if tx:
        _plot_train(axes[0], tx, tmean, tse, tn, color, label, ls=ls)
      else:
        print(f'{label} train: no data')
    ev = []
    if kind == 'rnd':
      ev = [_read_rnd_eval(d) for d in log_dirs if _read_rnd_eval(d)]
    else:
      for log_dir in log_dirs:
        ev.extend(base._read_eval_seed_series(base.LOG_ROOT, log_dir))
    ex, emean, ese, en = base._aggregate_mean_stderr(ev)
    if ex:
      _plot_eval(axes[1], ex, emean, ese, en, color, label, ls=ls)
    else:
      print(f'{label} eval: no data')

  axes[0].set_title(
      'live jobs — train (raw last-1000, mean ± SE; NF solid, RND dashed)',
      fontsize=12, fontweight='bold')
  _style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper left', fontsize=8.5, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (rolling mean, window={EVAL_WINDOW}, mean ± SE)',
      fontsize=12, fontweight='bold')
  _style(axes[1], f'eval success (roll mean, w={EVAL_WINDOW})', xlabel=True)
  axes[1].legend(loc='upper left', fontsize=8.5, framealpha=0.95)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {OUT_PATH}', flush=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--watch', type=int, default=0,
                      help='Refresh interval in seconds; 0 = once.')
  parser.add_argument('--watch-max-sec', type=int, default=0,
                      help='Stop after this many seconds; 0 = until killed.')
  args = parser.parse_args()
  if args.watch <= 0:
    run_once()
    return
  t0 = time.time()
  print(
      f'[nvbucket running] watching every {args.watch}s '
      f'(max {args.watch_max_sec or "∞"}s) → {OUT_PATH}',
      flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001
      print(f'[nvbucket running] error: {exc}', flush=True)
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[nvbucket running] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
