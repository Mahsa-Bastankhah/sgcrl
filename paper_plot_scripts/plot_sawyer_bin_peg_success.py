#!/usr/bin/env python3
"""Paper figure: Sawyer bin / peg success, SGCRL vs PPO+NF vs MPO+CRL.

2x2: columns = bin, peg; rows = train, eval. Mean ±1 SE across seeds.
Train is raw. Eval is faint raw + bold centered rolling mean (window=5).

SGCRL: logs/final_metaworld_runs/lp_contrastive_sawyer_{bin,peg}_40m/
PPO+NF: logs/final_metaworld_runs/ppo_{bin,peg}_nf_tiny_...  (seeds 0, 1, 2)
MPO+CRL: logs/final_metaworld_runs/mpo_crl_sawyer_{bin,peg}_tau0p85_40m/

  python paper_plot_scripts/plot_sawyer_bin_peg_success.py
  python paper_plot_scripts/plot_sawyer_bin_peg_success.py --watch 300
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
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import plot_builderbench_train_success1000 as base  # noqa: E402

LOG_ROOT = os.path.join(REPO, 'logs', 'final_metaworld_runs')
OUT_DIR = os.path.join(REPO, 'figs', 'metaworld', 'final_metaworld_runs')
OUT_STEM = os.path.join(OUT_DIR, 'sawyer_bin_peg_success')
# Keep the original LP-only filename in sync so the open figure updates.
OUT_LEGACY = os.path.join(OUT_DIR, 'lp_contrastive_sawyer_bin_peg_success.png')

PPO_STEPS_PER_ITER = 1024  # num_envs=4 × rollout_length=256

# Okabe–Ito (colorblind-safe).
COLOR_SGCRL = '#0072B2'
COLOR_PPO = '#D55E00'
COLOR_MPO = '#009E73'

METHODS = (
    {
        'key': 'sgcrl',
        'label': 'SGCRL',
        'color': COLOR_SGCRL,
        'dirs': {
            'bin': 'lp_contrastive_sawyer_bin_40m',
            'peg': 'lp_contrastive_sawyer_peg_40m',
        },
        'kind': 'lp',
    },
    {
        'key': 'ppo_nf',
        'label': 'PPO+NF',
        'color': COLOR_PPO,
        'dirs': {
            'bin': ('ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
                    '_minstd1e5_extrew1_rand_mixtaskg'),
            'peg': ('ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
                    '_extrew1_rand_minstd1e5_ent0005_mixtaskg'),
        },
        'kind': 'ppo',
    },
    {
        'key': 'mpo_crl',
        'label': 'MPO+CRL',
        'color': COLOR_MPO,
        'dirs': {
            'bin': 'mpo_crl_sawyer_bin_tau0p85_40m',
            'peg': 'mpo_crl_sawyer_peg_tau0p85_40m',
        },
        # Same CSV layout as PPO: learner train_success_1000, eval success_1000.
        'kind': 'ppo',
    },
)

ENVS = (
    ('bin', 'Sawyer bin'),
    ('peg', 'Sawyer peg'),
)

TITLE_FS = 18
LABEL_FS = 16
TICK_FS = 14
LEGEND_FS = 13
CAPTION_FS = 11


def _paper_rc() -> dict:
  return {
      'font.family': 'serif',
      'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
      'mathtext.fontset': 'stix',
      'font.size': TICK_FS,
      'axes.titlesize': TITLE_FS,
      'axes.labelsize': LABEL_FS,
      'xtick.labelsize': TICK_FS,
      'ytick.labelsize': TICK_FS,
      'legend.fontsize': LEGEND_FS,
      'axes.linewidth': 1.15,
      'pdf.fonttype': 42,
      'ps.fonttype': 42,
      'savefig.dpi': 300,
  }


def _read_xy(path: str, x_keys: tuple[str, ...], y_keys: tuple[str, ...],
             x_scale: float = 1.0) -> list[tuple[int, float]]:
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return []
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.reader(fh)
    try:
      header = next(reader)
    except StopIteration:
      return []
    name_to_i = {k: i for i, k in enumerate(header)}
    try:
      xi = next(name_to_i[k] for k in x_keys if k in name_to_i)
      yi = next(name_to_i[k] for k in y_keys if k in name_to_i)
    except StopIteration:
      return []
    pts: list[tuple[int, float]] = []
    for row in reader:
      if max(xi, yi) >= len(row):
        continue
      x = base._coerce(row[xi])
      y = base._coerce(row[yi])
      if x is None or y is None:
        continue
      pts.append((int(x * x_scale), float(y)))
  pts.sort(key=lambda p: p[0])
  return pts


def _seed_dirs(cfg_dir: str) -> list[str]:
  if not os.path.isdir(cfg_dir):
    return []
  out = []
  for name in sorted(os.listdir(cfg_dir)):
    run_dir = os.path.join(cfg_dir, name)
    if os.path.isdir(run_dir):
      out.append(run_dir)
  return out


def _lp_series(cfg_dir: str, split: str) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir):
    pts = _read_xy(
        os.path.join(run_dir, 'logs', split, 'logs.csv'),
        ('actor_steps',),
        ('success_1000', 'success'),
    )
    if pts:
      series.append(pts)
  return series


def _ppo_train_series(cfg_dir: str) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir):
    pts = _read_xy(
        os.path.join(run_dir, 'logs', 'learner', 'logs.csv'),
        ('global_step',),
        ('train_success_1000',),
    )
    if pts:
      series.append(pts)
  return series


def _ppo_eval_series(cfg_dir: str) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir):
    pts = _read_xy(
        os.path.join(run_dir, 'logs', 'eval', 'logs.csv'),
        ('iteration',),
        ('success_1000', 'success'),
        x_scale=PPO_STEPS_PER_ITER,
    )
    if pts:
      series.append(pts)
  return series


def _method_series(method: dict, env_key: str):
  cfg_dir = os.path.join(LOG_ROOT, method['dirs'][env_key])
  if method['kind'] == 'lp':
    return _lp_series(cfg_dir, 'actor'), _lp_series(cfg_dir, 'evaluator')
  return _ppo_train_series(cfg_dir), _ppo_eval_series(cfg_dir)


def _shade(ax, xs, mean, se, *, color, n: int) -> None:
  if n <= 1 or not se:
    return
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  ax.fill_between(xs, lo, hi, color=color, alpha=0.22, lw=0, zorder=2)


def _style(ax, *, title: str, ylabel: str, xlabel: bool) -> None:
  ax.set_title(title, fontweight='bold', pad=8)
  ax.set_ylabel(ylabel)
  if xlabel:
    ax.set_xlabel('Environment steps')
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.tick_params(length=4, width=1.0)


def _draw() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  plt.rcParams.update(_paper_rc())
  fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex='col', sharey=True)

  for col, (env_key, env_title) in enumerate(ENVS):
    ax_t, ax_e = axes[0, col], axes[1, col]
    for method in METHODS:
      t_seeds, e_seeds = _method_series(method, env_key)
      txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
      exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
      color = method['color']
      lab_t = f'{method["label"]}  ($n$={n_t})' if n_t else method['label']
      lab_e = f'{method["label"]}  ($n$={n_e})' if n_e else method['label']

      if txs:
        xs, mean, se = base._subsample_curve(txs, tmean, tse)
        _shade(ax_t, xs, mean, se, color=color, n=n_t)
        ax_t.plot(xs, mean, color=color, lw=2.6, alpha=0.95, label=lab_t,
                  zorder=3, solid_capstyle='round')
        print(f'  {env_key} {method["key"]} train last={tmean[-1]:.3f} '
              f'peak={max(tmean):.3f} @ {txs[-1]/1e6:.2f}M  n={n_t}')
      else:
        print(f'  {env_key} {method["key"]} train: no data')

      if exs:
        xs, mean, se = base._subsample_curve(exs, emean, ese)
        if n_e > 1:
          lo = base._rolling_mean([m - s for m, s in zip(mean, se)], w)
          hi = base._rolling_mean([m + s for m, s in zip(mean, se)], w)
          ax_e.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
        sm = base._plot_eval_smoothed(
            ax_e, xs, mean, color=color, label=lab_e, zorder=3, linewidth=2.6)
        print(f'  {env_key} {method["key"]} eval raw last={emean[-1]:.3f} '
              f'peak={max(emean):.3f} @ {exs[-1]/1e6:.2f}M  n={n_e}')
        if sm:
          print(f'  {env_key} {method["key"]} eval smooth last={sm[-1]:.3f}')
      else:
        print(f'  {env_key} {method["key"]} eval: no data')

    _style(ax_t, title=f'{env_title}  ·  train',
           ylabel='Train success (last 1000)', xlabel=False)
    _style(ax_e, title=f'{env_title}  ·  eval (roll. mean $w$={w})',
           ylabel=f'Eval success (roll. mean $w$={w})', xlabel=True)

  # Shared legend from both methods (proxy artists so n is not panel-specific).
  handles = [
      Line2D([0], [0], color=m['color'], lw=2.8, label=m['label'])
      for m in METHODS
  ]
  fig.legend(
      handles=handles, loc='upper center', ncol=3, frameon=False,
      bbox_to_anchor=(0.5, 1.02), handlelength=2.4, columnspacing=1.8,
  )
  fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14,
                      wspace=0.16, hspace=0.32)
  fig.text(
      0.5, 0.02,
      r'Solid: mean across seeds; shade: $\pm$1 s.e.  '
      'Train is raw.  Eval: faint raw + bold rolling mean '
      f'(window={w}).',
      ha='center', va='bottom', fontsize=CAPTION_FS, color='#333333',
  )

  os.makedirs(OUT_DIR, exist_ok=True)
  png = OUT_STEM + '.png'
  pdf = OUT_STEM + '.pdf'
  tmp = png + '.tmp.png'
  fig.savefig(tmp, dpi=300, bbox_inches='tight')
  os.replace(tmp, png)
  fig.savefig(pdf, bbox_inches='tight')
  # Mirror onto the filename already used in the paper-draft folder.
  tmp_legacy = OUT_LEGACY + '.tmp.png'
  fig.savefig(tmp_legacy, dpi=300, bbox_inches='tight')
  os.replace(tmp_legacy, OUT_LEGACY)
  plt.close(fig)
  print(f'→ {png}')
  print(f'→ {pdf}')
  print(f'→ {OUT_LEGACY}')


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--watch', type=float, default=0.0,
      help='If >0, redraw every this many seconds (local CPU; Ctrl-C to stop).')
  args = ap.parse_args()
  _draw()
  if args.watch <= 0:
    return
  print(f'watching every {args.watch:.0f}s')
  while True:
    time.sleep(args.watch)
    _draw()


if __name__ == '__main__':
  main()
