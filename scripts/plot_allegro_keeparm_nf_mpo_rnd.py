#!/usr/bin/env python3
"""Keep-arm pad-hold hide-table bucket: NF P5 vs MPO-CRL vs PPO+RND.

Seeds are averaged. Shaded band is ±1 standard error. Train is raw
``train_success_1000``. Eval is faint raw + bold rolling mean (window=5);
the SE band is the rolling mean of mean±SE.

  python scripts/plot_allegro_keeparm_nf_mpo_rnd.py
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
    'akt_keeparm_bucket_nf_mpo_rnd_success.png')
OUT_PATH_ROOT = os.path.join(REPO, 'akt_keeparm_bucket_nf_mpo_rnd_success.png')
# Allegro eval is every 10 iters (~0.5M steps), so w=5 still looks raw.
EVAL_WINDOW = 21
OUT_PATH_SEEDS12 = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw',
    'akt_keeparm_bucket_nf_p05_seed1_seed2_success.png')

METHODS = (
    (
        'NF P5 shift',
        (
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed0_14h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed1_14h',
            'ppo_allegro_kuka_throw_e1024_nf_compactsmall_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_q05shift_seed2_14h',
        ),
        'nf',
        base.ACCENT_COLORS[4],
    ),
    (
        'MPO-CRL',
        (
            'mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_seed0_14h',
            'mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h',
            'mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_seed2_20h',
            'mpo_crl_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_seed3_20h',
        ),
        'mpo',
        base.ACCENT_COLORS[0],
    ),
    (
        'PPO+RND',
        (
            'ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_seed0_14h',
            'ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_seed1_14h',
            'ppo_rnd_allegro_kuka_throw_e1024_keeparm_palmup_padhold'
            '_hidetable_bucket_xy0022_norand_ep50_500m_seed2_14h',
        ),
        'rnd',
        base.ACCENT_COLORS[2],
    ),
)

NF_SEED_RUNS = (
    (METHODS[0][1][1], 'NF P5  seed 1', base.ACCENT_COLORS[0]),
    (METHODS[0][1][2], 'NF P5  seed 2', base.ACCENT_COLORS[2]),
)


def _collect_train(log_dirs):
  seeds = []
  for log_dir in log_dirs:
    seeds.extend(base._read_train_seed_series(base.LOG_ROOT, log_dir))
  return seeds


def _collect_eval(log_dirs, kind: str):
  if kind == 'rnd':
    return [_read_rnd_eval(d) for d in log_dirs if _read_rnd_eval(d)]
  seeds = []
  for log_dir in log_dirs:
    seeds.extend(base._read_eval_seed_series(base.LOG_ROOT, log_dir))
  return seeds


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


def _plot_train(ax, xs, mean, se, n, color, label):
  xs, mean, se = base._subsample_curve(xs, mean, se)
  ax.plot(xs, mean, color=color, lw=2.0, label=f'{label}  (n={n})')
  if n > 1 and any(s > 0 for s in se):
    lo = [m - s for m, s in zip(mean, se)]
    hi = [m + s for m, s in zip(mean, se)]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.22, linewidth=0)
  peak_i = mean.index(max(mean))
  print(f'{label} train: n={n} last={mean[-1]:.4f} '
        f'peak={mean[peak_i]:.4g} @ {xs[peak_i]/1e6:.1f}M '
        f'steps={xs[-1]/1e6:.1f}M')


def _plot_eval(ax, xs, mean, se, n, color, label):
  sm = base._plot_eval_smoothed(
      ax, xs, mean, color=color,
      label=f'{label}  (n={n}, roll mean w={EVAL_WINDOW})',
      linewidth=2.6, window=EVAL_WINDOW)
  if n > 1 and any(s > 0 for s in se):
    lo = base._rolling_mean([m - s for m, s in zip(mean, se)], window=EVAL_WINDOW)
    hi = base._rolling_mean([m + s for m, s in zip(mean, se)], window=EVAL_WINDOW)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, linewidth=0)
  ys = sm if sm else mean
  peak_i = ys.index(max(ys))
  print(f'{label} eval: n={n} last={ys[-1]:.4f} '
        f'peak={ys[peak_i]:.4g} @ {xs[peak_i]/1e6:.1f}M')


def run_once() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.2), sharex=True)
  n_ok = 0
  for label, log_dirs, kind, color in METHODS:
    train_seeds = [] if kind == 'rnd' else _collect_train(log_dirs)
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train_seeds)
    if tx:
      _plot_train(axes[0], tx, tmean, tse, tn, color, label)
      n_ok += 1
    elif kind == 'rnd':
      print(f'{label} train: RND has no train_success_1000 (eval only)')
    else:
      print(f'{label} train: no data')

    eval_seeds = _collect_eval(log_dirs, kind)
    ex, emean, ese, en = base._aggregate_mean_stderr(eval_seeds)
    if ex:
      _plot_eval(axes[1], ex, emean, ese, en, color, label)
      if kind == 'rnd':
        n_ok += 1
    else:
      print(f'{label} eval: no data')

  axes[0].set_title(
      'keep-arm pad-hold bucket — train (raw last-1000, mean ± SE)',
      fontsize=12, fontweight='bold')
  _style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper left', fontsize=8.5, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (rolling mean, window={EVAL_WINDOW}, mean ± SE)',
      fontsize=12, fontweight='bold')
  _style(axes[1], f'eval success (roll mean, w={EVAL_WINDOW})',
         xlabel=True)
  axes[1].legend(loc='upper left', fontsize=8.5, framealpha=0.95)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=160, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  tmp_root = OUT_PATH_ROOT + '.tmp.png'
  fig.savefig(tmp_root, dpi=160, bbox_inches='tight')
  os.replace(tmp_root, OUT_PATH_ROOT)
  plt.close(fig)
  print(f'plotted {n_ok} methods  → {OUT_PATH}', flush=True)
  print(f'                         → {OUT_PATH_ROOT}', flush=True)
  _run_once_seeds12()


def _run_once_seeds12() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.0), sharex=True)
  for log_dir, label, color in NF_SEED_RUNS:
    train = base._read_train_seed_series(base.LOG_ROOT, log_dir)
    tx, tmean, tse, tn = base._aggregate_mean_stderr(train)
    ev = base._read_eval_seed_series(base.LOG_ROOT, log_dir)
    ex, emean, _, en = base._aggregate_mean_stderr(ev)
    if tx:
      tx, tmean, tse = base._subsample_curve(tx, tmean, tse)
      axes[0].plot(tx, tmean, color=color, lw=2.0, label=label)
      peak_i = tmean.index(max(tmean))
      print(f'{label} train: n={tn} last={tmean[-1]:.4f} '
            f'peak={tmean[peak_i]:.4g} @ {tx[peak_i]/1e6:.1f}M '
            f'steps={tx[-1]/1e6:.1f}M', flush=True)
    else:
      print(f'{label} train: no data', flush=True)
    if ex:
      sm = base._plot_eval_smoothed(
          axes[1], ex, emean, color=color,
          label=f'{label}  (roll mean w={base.EVAL_SMOOTH_WINDOW})')
      ys = sm if sm else emean
      peak_i = ys.index(max(ys))
      print(f'{label} eval: n={en} last={ys[-1]:.4f} '
            f'peak={ys[peak_i]:.4g} @ {ex[peak_i]/1e6:.1f}M', flush=True)
    else:
      print(f'{label} eval: no data', flush=True)
  axes[0].set_title(
      'NF P5 seeds 1 and 2 — train (raw last-1000)',
      fontsize=12, fontweight='bold')
  _style(axes[0], 'train_success_1000')
  axes[0].legend(loc='upper left', fontsize=9, framealpha=0.95)
  axes[1].set_title(
      f'eval success  (rolling mean, window={base.EVAL_SMOOTH_WINDOW})',
      fontsize=12, fontweight='bold')
  _style(axes[1], f'eval success (roll mean, w={base.EVAL_SMOOTH_WINDOW})',
         xlabel=True)
  axes[1].legend(loc='upper left', fontsize=9, framealpha=0.95)
  fig.tight_layout()
  os.makedirs(os.path.dirname(OUT_PATH_SEEDS12), exist_ok=True)
  tmp = OUT_PATH_SEEDS12 + '.tmp.png'
  fig.savefig(tmp, dpi=160, bbox_inches='tight')
  os.replace(tmp, OUT_PATH_SEEDS12)
  plt.close(fig)
  print(f'→ {OUT_PATH_SEEDS12}', flush=True)


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
      f'[keeparm nf/mpo/rnd] watching every {args.watch}s '
      f'(max {args.watch_max_sec or "∞"}s) → {OUT_PATH}',
      flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001
      print(f'[keeparm nf/mpo/rnd] error: {exc}', flush=True)
    if args.watch_max_sec > 0 and time.time() - t0 >= args.watch_max_sec:
      print('[keeparm nf/mpo/rnd] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
