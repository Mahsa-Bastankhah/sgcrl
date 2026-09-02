#!/usr/bin/env python3
"""Live comparison: original-repo SGCRL (graliuce) vs PPO+NF on Sawyer bin/peg.

2x2: columns = bin, peg; rows = train, eval. Mean ±1 SE across seeds.
Train is raw. Eval is faint raw + bold centered rolling mean (window=5).

SGCRL (2 seeds, live): /n/fs/mislresearch/old-sgcrl/logs/sawyer_{bin,peg}_s{0,1}/
PPO+NF bin ent=0.05 (1 seed, rand): ..._rand_mixtaskg/
PPO+NF bin ent=0.005 (1 seed, rand): ..._rand_ent0005_mixtaskg/
PPO+NF peg (3 done, ent=0.005): ..._rand_minstd1e5_ent0005_mixtaskg/
PPO+NF bin norand orig (seeds 0,1 Aug-12 + seed 2 new): ..._norand/

  python scripts/plot_old_sgcrl_vs_ppo_nf_success.py
  python scripts/plot_old_sgcrl_vs_ppo_nf_success.py --watch 300
"""
from __future__ import annotations

import argparse
import csv
import math
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

OLD_SGCRL = '/n/fs/mislresearch/old-sgcrl/logs'
LOG_ROOT = os.path.join(REPO, 'logs', 'final_metaworld_runs')
OUT_DIR = os.path.join(REPO, 'figs', 'metaworld', 'final_metaworld_runs')
OUT_STEM = os.path.join(OUT_DIR, 'sawyer_bin_peg_success')
OUT_LIVE = os.path.join(OUT_DIR, 'old_sgcrl_vs_ppo_nf_success')

PPO_STEPS_PER_ITER = 1024  # num_envs=4 × rollout_length=256
LP_NUM_ACTORS = 4

COLOR_SGCRL = '#0072B2'
COLOR_PPO_ENT05 = '#D55E00'
COLOR_PPO_ENT0005 = '#009E73'
COLOR_PPO_NORAND = '#CC79A7'

PPO_DIR_BIN_ENT05 = (
    'ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
    '_minstd1e5_extrew1_rand_mixtaskg')
PPO_DIR_BIN_ENT0005 = (
    'ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
    '_minstd1e5_extrew1_rand_ent0005_mixtaskg')
PPO_DIR_BIN_NORAND = (
    'ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
    '_minstd1e5_extrew1_norand')
PPO_DIR_PEG = (
    'ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m'
    '_extrew1_rand_minstd1e5_ent0005_mixtaskg')

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


def _finite(v):
  return v is not None and math.isfinite(v)


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
    yi = next((name_to_i[k] for k in y_keys if k in name_to_i), None)
    if yi is None:
      return []
    xi = next((name_to_i[k] for k in x_keys if k in name_to_i), None)
    # Launchpad evaluator CSVs sometimes omit actor_steps; scale
    # evaluator_steps by num_actors so the x-axis is env steps.
    eval_i = name_to_i.get('evaluator_steps')
    pts: list[tuple[int, float]] = []
    for row in reader:
      if yi >= len(row):
        continue
      y = base._coerce(row[yi])
      if not _finite(y):
        continue
      x = None
      if xi is not None and xi < len(row):
        x = base._coerce(row[xi])
      if not _finite(x) and eval_i is not None and eval_i < len(row):
        ev = base._coerce(row[eval_i])
        if _finite(ev):
          x = ev * LP_NUM_ACTORS
      if not _finite(x):
        continue
      pts.append((int(x * x_scale), float(y)))
  pts.sort(key=lambda p: p[0])
  return pts


def _seed_dirs(cfg_dir: str) -> list[str]:
  if not os.path.isdir(cfg_dir):
    return []
  return [os.path.join(cfg_dir, name)
          for name in sorted(os.listdir(cfg_dir))
          if os.path.isdir(os.path.join(cfg_dir, name))]


def _lp_run_dirs(env_key: str) -> list[str]:
  parent = os.path.join(OLD_SGCRL, f'sawyer_{env_key}_s')
  out = []
  if not os.path.isdir(OLD_SGCRL):
    return out
  for name in sorted(os.listdir(OLD_SGCRL)):
    if not name.startswith(f'sawyer_{env_key}_s'):
      continue
    seed_root = os.path.join(OLD_SGCRL, name)
    if not os.path.isdir(seed_root):
      continue
    for child in sorted(os.listdir(seed_root)):
      run_dir = os.path.join(seed_root, child)
      if os.path.isdir(run_dir) and os.path.isdir(os.path.join(run_dir, 'logs')):
        out.append(run_dir)
  del parent
  return out


def _lp_series(env_key: str, split: str) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _lp_run_dirs(env_key):
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


def _series_xmax(seed_series: list[list[tuple[int, float]]]) -> int:
  m = 0
  for pts in seed_series:
    if pts:
      m = max(m, pts[-1][0])
  return m


def _plot_train(ax, xs, mean, se, *, color, n, ls, z, lw=2.6) -> None:
  if not xs:
    return
  xs, mean, se = base._subsample_curve(xs, mean, se)
  _shade(ax, xs, mean, se, color=color, n=n)
  ax.plot(xs, mean, color=color, lw=lw, ls=ls, alpha=0.95, zorder=z,
          solid_capstyle='round')


def _plot_eval(ax, xs, mean, se, *, color, n, ls, z, w: int) -> None:
  if not xs:
    return
  xs, mean, se = base._subsample_curve(xs, mean, se)
  if n > 1:
    lo = base._rolling_mean([m - s for m, s in zip(mean, se)], w)
    hi = base._rolling_mean([m + s for m, s in zip(mean, se)], w)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=z - 1)
  base._plot_eval_smoothed(
      ax, xs, mean, color=color, label='_nolegend_', linestyle=ls,
      zorder=z, linewidth=2.6, window=w)


def _faint_seeds(ax, seed_series, *, color, ls='-', z=2) -> None:
  for pts in seed_series:
    if len(pts) < 2:
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=1.0, ls=ls, alpha=0.35, zorder=z)


def _draw() -> None:
  w = base.EVAL_SMOOTH_WINDOW
  plt.rcParams.update(_paper_rc())
  # Independent x per panel: peg's 40M axis must not crush the live bin/SGCRL
  # curves (sharex='col' + a finished 40M series made them look empty).
  fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=False, sharey=True)

  methods = (
      {
          'key': 'ppo_nf_ent05',
          'label': 'PPO+NF ent=0.05 (rand)',
          'color': COLOR_PPO_ENT05,
          'kind': 'ppo',
          'ls': '-',
          'z': 3,
          'envs': ('bin',),
          'ppo_dir': PPO_DIR_BIN_ENT05,
      },
      {
          'key': 'ppo_nf_ent0005',
          'label': 'PPO+NF ent=0.005 (rand)',
          'color': COLOR_PPO_ENT0005,
          'kind': 'ppo',
          'ls': '-',
          'z': 4,
          'envs': ('bin', 'peg'),
          'ppo_dir': {
              'bin': PPO_DIR_BIN_ENT0005,
              'peg': PPO_DIR_PEG,
          },
      },
      {
          'key': 'ppo_nf_norand',
          'label': 'PPO+NF (norand orig.)',
          'color': COLOR_PPO_NORAND,
          'kind': 'ppo',
          'ls': '-',
          'z': 4,
          'envs': ('bin',),
          'ppo_dir': PPO_DIR_BIN_NORAND,
      },
      {
          'key': 'sgcrl',
          'label': 'SGCRL (original)',
          'color': COLOR_SGCRL,
          'kind': 'lp',
          'ls': '--',
          'z': 5,
          'envs': ('bin', 'peg'),
      },
  )

  for col, (env_key, env_title) in enumerate(ENVS):
    ax_t, ax_e = axes[0, col], axes[1, col]
    packed = []
    col_xmax = 0
    for method in methods:
      if env_key not in method['envs']:
        continue
      if method['kind'] == 'lp':
        t_seeds = _lp_series(env_key, 'actor')
        e_seeds = _lp_series(env_key, 'evaluator')
      else:
        ppo_dir = method['ppo_dir']
        if isinstance(ppo_dir, dict):
          ppo_dir = ppo_dir[env_key]
        cfg_dir = os.path.join(LOG_ROOT, ppo_dir)
        t_seeds = _ppo_train_series(cfg_dir)
        e_seeds = _ppo_eval_series(cfg_dir)
      txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
      exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
      col_xmax = max(col_xmax, _series_xmax(t_seeds), _series_xmax(e_seeds))
      packed.append((method, t_seeds, e_seeds, txs, tmean, tse, n_t,
                     exs, emean, ese, n_e))
      if txs:
        print(f'  {env_key} {method["key"]} train last={tmean[-1]:.3f} '
              f'peak={max(tmean):.3f} @ {txs[-1]/1e6:.2f}M  n={n_t}')
      else:
        print(f'  {env_key} {method["key"]} train: no data')
      if exs:
        print(f'  {env_key} {method["key"]} eval raw last={emean[-1]:.3f} '
              f'peak={max(emean):.3f} @ {exs[-1]/1e6:.2f}M  n={n_e}')
      else:
        print(f'  {env_key} {method["key"]} eval: no data')

    for method, t_seeds, e_seeds, txs, tmean, tse, n_t, exs, emean, ese, n_e in packed:
      _faint_seeds(ax_t, t_seeds, color=method['color'], ls=method['ls'], z=2)
      _plot_train(ax_t, txs, tmean, tse, color=method['color'], n=n_t,
                  ls=method['ls'], z=method['z'])
      _faint_seeds(ax_e, e_seeds, color=method['color'], ls=method['ls'], z=2)
      _plot_eval(ax_e, exs, emean, ese, color=method['color'], n=n_e,
                 ls=method['ls'], z=method['z'], w=w)

    xmax = int(col_xmax * 1.05) if col_xmax else 1
    ax_t.set_xlim(0, xmax)
    ax_e.set_xlim(0, xmax)
    _style(ax_t, title=f'{env_title}  ·  train',
           ylabel='Train success (last 1000)', xlabel=False)
    _style(ax_e, title=f'{env_title}  ·  eval (roll. mean $w$={w})',
           ylabel=f'Eval success (roll. mean $w$={w})', xlabel=True)

  handles = [
      Line2D([0], [0], color=m['color'], lw=2.8, ls=m['ls'], label=m['label'])
      for m in methods
  ]
  fig.legend(
      handles=handles, loc='upper center', ncol=4, frameon=False,
      bbox_to_anchor=(0.5, 1.02), handlelength=2.4, columnspacing=1.6,
  )
  fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14,
                      wspace=0.16, hspace=0.32)
  fig.text(
      0.5, 0.02,
      r'Dashed blue: SGCRL. Orange: PPO+NF ent=0.05 rand (bin). '
      r'Green: PPO+NF ent=0.005 rand (bin/peg). '
      r'Pink: PPO+NF norand orig (bin; seeds 0,1=Aug-12, seed 2=current code). '
      r'Thin: per-seed. Shade: $\pm$1 s.e.  Train raw; eval roll. mean '
      f'$w$={w}.',
      ha='center', va='bottom', fontsize=CAPTION_FS, color='#333333',
  )

  os.makedirs(OUT_DIR, exist_ok=True)
  for stem in (OUT_STEM, OUT_LIVE):
    png = stem + '.png'
    pdf = stem + '.pdf'
    tmp = png + '.tmp.png'
    fig.savefig(tmp, dpi=300, bbox_inches='tight')
    os.replace(tmp, png)
    fig.savefig(pdf, bbox_inches='tight')
    print(f'→ {png}')
    print(f'→ {pdf}')
  plt.close(fig)


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
