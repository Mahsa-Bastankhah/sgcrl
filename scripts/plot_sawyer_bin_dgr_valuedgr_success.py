#!/usr/bin/env python3
"""Sawyer bin NF + dualgradreg + valuedgr: eval success and episode length.

Reads logs/eval/logs.csv. One mean line per config with ±1 SE shade across
seeds. Both eval metrics use faint raw means plus bold centered rolling means,
window=5.

  python scripts/plot_sawyer_bin_dgr_valuedgr_success.py
  python scripts/plot_sawyer_bin_dgr_valuedgr_success.py --watch 900
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(REPO, 'logs', 'final_metaworld_runs')
OUT = os.path.join(
    REPO, 'figs', 'metaworld', 'final_metaworld_runs',
    'sawyer_bin_dgr_valuedgr_eval_success.png')
MPOINIT_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_mixtaskg_'
    'dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6')
MPOINIT_BOUNDED_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_boundedmocap_mixtaskg_'
    'dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6')
MPOINIT_ANTIWINDUP_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_selectiveantiwindup_mixtaskg_'
    'dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6')
MPOINIT_TRACKERRTERM_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_trackerrterm20cm_mixtaskg_'
    'dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6')
MPOINIT_TRACKERRTERM_POLICYRESET_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_trackerrterm20cm_policyreset_'
    'lastlayerreset_cooldown_mixtaskg_dualgradreg_c100_lamlr1e6_valuedgr_'
    'c100_lamlr1e6')
Z3TO6_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_z3to6_mixtaskg_'
    'dualgradreg_c100_lamlr1e6_valuedgr_c100_lamlr1e6')
BASE_DIR = 'ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_mixtaskg'
MPO_DIR = 'mpo_crl_sawyer_bin_safeinit_z3to6_40m'
RND_DIR = 'ppo_rnd_sawyer_bin_safeinit_z3to6_40m'
RND_MPOINIT_DIR = 'ppo_rnd_sawyer_bin_safeinit_mpoinit_xy5_40m'
MPOINIT_NOGRADREG_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_nogradreg_mixtaskg')
MPOINIT_NOGRADREG_EP50_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_nogradreg_ep50_mixtaskg')
MPOINIT_NOGRADREG_EP50_ENT05_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_nogradreg_ep50_ent05_mixtaskg')
MPOINIT_NOGRADREG_EP100_ENT05_G095_DIR = (
    'ppo_bin_nf_floorgrasp_safeinit_mpoinit_nogradreg_ep100_ent05_g095_mixtaskg')
DISCOVER_DIR = 'discover_sawyer_bin_e4_40m_safeinit_z3to6_ucbstd0'
PPO_SPI = 1024  # num_envs=4 × rollout_length=256
MPO_SPI = 1024
DISCOVER_EVAL_EVERY = 17
BASE_COLOR = '#555555'
# (kind, cfg, tag, ls, z, color, seeds)
CONFIGS = (
    ('ppo', BASE_DIR, 'NF no DGR', '--', 3, BASE_COLOR, (0, 1, 2)),
    ('ppo', Z3TO6_DIR, 'NF z Uniform 3–6cm', (0, (5, 1.5)), 6, '#D55E00',
     (0, 1, 2)),
    ('ppo', MPOINIT_DIR, 'NF MPO-safe XY±5cm', '-.', 7, '#0072B2', (0, 1, 2)),
    ('ppo', MPOINIT_NOGRADREG_DIR, 'NF MPO XY±5cm no grad-reg',
     (0, (4, 1.5)), 16, '#332288', (0, 1)),
    ('ppo', MPOINIT_NOGRADREG_EP50_DIR, 'NF MPO XY±5cm no grad-reg ep50',
     (0, (6, 1.5)), 17, '#88CCEE', (0,)),
    ('ppo', MPOINIT_NOGRADREG_EP50_ENT05_DIR,
     'NF MPO XY±5cm no grad-reg ep50 ent0.05',
     (0, (5, 1.5)), 18, '#AA3377', (0,)),
    ('ppo', MPOINIT_BOUNDED_DIR, 'NF MPO-init + bounded mocap', '-',
     11, '#E69F00', (0, 1, 2)),
    ('mpo', MPO_DIR, 'MPO-CRL z3–6', '-', 8, '#56B4E9', (0, 1)),
    ('rnd', RND_DIR, 'PPO+RND z3–6', (0, (3, 1.2)), 9, '#CC79A7', (0, 1)),
    ('discover', DISCOVER_DIR, 'DISCOVER z3–6', ':', 10, '#009E73', (0, 1)),
)
_RUN_PREFIX = {
    'ppo': 'ppo_sawyer_bin',
    'mpo': 'mpo_crl_sawyer_bin',
    'rnd': 'ppo_rnd_sawyer_bin',
    'discover': 'discover_sawyer_bin',
}


def _eval_path(kind: str, cfg: str, seed: int) -> str:
  run = f'{_RUN_PREFIX[kind]}_{seed}'
  if kind == 'discover':
    return os.path.join(LOG_ROOT, cfg, run, 'logs.csv')
  return os.path.join(LOG_ROOT, cfg, run, 'logs', 'eval', 'logs.csv')


def _load_eval(path: str, kind: str, metric: str = 'success'):
  xs, ys = [], []
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return xs, ys
  with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    reader = csv.DictReader(fh)
    fields = set(reader.fieldnames or [])
    if metric == 'episode_length':
      y_key = 'episode_length' if 'episode_length' in fields else None
    else:
      y_key = 'success_1000' if 'success_1000' in fields else (
          'success' if 'success' in fields else None)
    if y_key is None:
      return xs, ys
    for row in reader:
      if kind == 'discover':
        ep = base._coerce(row.get('epoch', ''))
        if ep is None:
          continue
        ep_i = int(ep)
        if ep_i != 1 and ep_i % DISCOVER_EVAL_EVERY != 0:
          continue
      y = base._coerce(row.get(y_key, ''))
      if y is None or (isinstance(y, float) and math.isnan(y)):
        continue
      if kind == 'rnd':
        x = base._coerce(row.get('learner_steps', ''))
        if x is None:
          it = base._coerce(row.get('iteration', ''))
          if it is None:
            continue
          x = it * 2048
      elif kind == 'discover':
        x = base._coerce(row.get('global_step', ''))
      else:
        it = base._coerce(row.get('iteration', ''))
        if it is None:
          it = base._coerce(row.get('learner_steps', ''))
        if it is None:
          continue
        spi = MPO_SPI if kind == 'mpo' else PPO_SPI
        x = it * spi
      if x is None:
        continue
      xs.append(int(x))
      ys.append(float(y))
  return xs, ys


def _mean_se_at_grid(seed_xy: list[tuple[list, list]]):
  all_x = sorted({x for xs, _ in seed_xy for x in xs})
  if not all_x:
    return [], [], [], 0
  means, ses = [], []
  n_max = 0
  for x in all_x:
    vals = []
    for xs, ys in seed_xy:
      if not xs or x < xs[0] or x > xs[-1]:
        continue
      y = ys[0]
      for xi, yi in zip(xs, ys):
        if xi > x:
          break
        y = yi
      vals.append(y)
    n = len(vals)
    n_max = max(n_max, n)
    if n == 0:
      means.append(float('nan'))
      ses.append(0.0)
      continue
    m = sum(vals) / n
    means.append(m)
    if n < 2:
      ses.append(0.0)
    else:
      var = sum((v - m) ** 2 for v in vals) / (n - 1)
      ses.append((var ** 0.5) / (n ** 0.5))
  return all_x, means, ses, n_max


def _plot_mean_eval(ax, xs, mean, se, *, color, label, ls, z, w, n):
  if not xs:
    return []
  sm = base._rolling_mean(mean, w)
  if n > 1 and any(s > 0 for s in se):
    lo = base._rolling_mean([m - s for m, s in zip(mean, se)], w)
    hi = base._rolling_mean([m + s for m, s in zip(mean, se)], w)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.20, lw=0, zorder=z - 1)
  ax.plot(
      xs, mean, color=color, linewidth=1.0, alpha=0.28, linestyle=ls,
      marker='o', markersize=2.6, markeredgewidth=0.0, zorder=z)
  ax.plot(
      xs, sm, color=color, linewidth=2.3, linestyle=ls, alpha=0.95,
      zorder=z + 1, label=label)
  return sm


def _collect_seeds(kind: str, cfg: str, seeds, metric: str = 'success'):
  seed_xy = []
  missing = []
  for seed in seeds:
    xs, ys = _load_eval(_eval_path(kind, cfg, seed), kind, metric)
    if xs:
      seed_xy.append((xs, ys))
    else:
      missing.append(seed)
  return seed_xy, missing


def run_once() -> str | None:
  w = base.EVAL_SMOOTH_WINDOW
  fig, (ax, ax_len) = plt.subplots(
      2, 1, figsize=(9.2, 7.2), sharex=True,
      gridspec_kw={'height_ratios': (1.25, 1.0)})
  n_plotted = 0
  last_bits = []

  for kind, cfg, tag, ls, z, color, seeds in CONFIGS:
    seed_xy, missing = _collect_seeds(kind, cfg, seeds)
    for seed in missing:
      print(f'  {tag} seed {seed}: no eval data yet')
    if not seed_xy:
      continue
    xs, mean, se, n = _mean_se_at_grid(seed_xy)
    sm = _plot_mean_eval(
        ax, xs, mean, se, color=color, ls=ls, z=z, w=w, n=n,
        label=f'{tag}  (n={n})')
    n_plotted += 1
    last = sm[-1] if sm else mean[-1]
    print(f'  {tag}: n={n} last_smooth={last:.3f} raw={mean[-1]:.3f} '
          f'@ {xs[-1]/1e6:.2f}M')
    last_bits.append(
        f'{tag} n={n} last={last:.3f} raw={mean[-1]:.3f} @ {xs[-1]/1e6:.2f}M')

    skip_eplen = cfg in (
        MPOINIT_NOGRADREG_DIR, MPOINIT_NOGRADREG_EP50_DIR,
        MPOINIT_NOGRADREG_EP50_ENT05_DIR)
    len_xy, len_missing = ([], []) if skip_eplen else _collect_seeds(
        kind, cfg, seeds, metric='episode_length')
    if len_xy:
      lxs, lmean, lse, ln = _mean_se_at_grid(len_xy)
      _plot_mean_eval(
          ax_len, lxs, lmean, lse, color=color, ls=ls, z=z, w=w, n=ln,
          label=f'{tag}  (n={ln})')
    elif not missing and len_missing:
      print(f'  {tag}: no eval episode_length data yet')

  ax.set_title(
      'Sawyer bin  ·  eval mean ± SE  ·  z Uniform 3–6cm init',
      fontsize=12, fontweight='bold')
  ax.set_ylabel(f'Eval success_1000 (roll mean w={w})', fontsize=10)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  if n_plotted:
    ax.legend(
        loc='upper left', fontsize=8.0, framealpha=0.95,
        fancybox=False, edgecolor='#333333')
  ax_len.set_ylabel(f'Eval episode length (roll mean w={w})', fontsize=10)
  ax_len.set_xlabel('Env steps', fontsize=10)
  ax_len.set_ylim(0, 155)
  ax_len.spines[['top', 'right']].set_visible(False)
  ax_len.grid(axis='y', linestyle='--', alpha=0.4)
  ax_len.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  if ax_len.lines:
    ax_len.legend(
        loc='lower left', fontsize=7.0, framealpha=0.95, ncol=2,
        fancybox=False, edgecolor='#333333')
  fig.text(
      0.5, 0.01,
      'MPO-safe init: object XY random, TCP XY ±5cm, z~U(3–6cm), safe servo. '
      'Other orange-init baselines keep TCP XY on object. '
      'Shade = ±1 SE. Both eval panels: faint raw mean + '
      f'bold centered rolling mean, window={w}. NF DGR: $c=100$ '
      '$\\lambda$lr$=10^{-6}$.',
      ha='center', va='bottom', fontsize=7.0, color='#555555')
  fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.14, hspace=0.12)
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  stamp = os.path.join(os.path.dirname(OUT), 'LAST_UPDATE.txt')
  with open(stamp, 'w', encoding='utf-8') as fh:
    fh.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    fh.write(OUT + '\n')
    for line in last_bits:
      fh.write(line + '\n')
  if n_plotted == 0:
    print(f'no eval data yet → {OUT} (axes only)')
  else:
    print(f'→ {OUT}')
  return OUT


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument(
      '--watch', type=int, default=0,
      help='If >0, refresh every N seconds (CPU-only login-node watcher).')
  args = p.parse_args()
  if args.watch <= 0:
    run_once()
    return
  print(f'[sawyer_bin_dgr] watching every {args.watch}s → {OUT}', flush=True)
  while True:
    try:
      run_once()
    except Exception as exc:  # noqa: BLE001 — keep watcher alive
      print(f'[sawyer_bin_dgr] error: {exc}', flush=True)
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
