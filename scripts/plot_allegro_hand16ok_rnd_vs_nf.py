#!/usr/bin/env python3
"""OK-hand success: live PPO-RND vs NF (same pose).

RND: logs/ppo_rnd_allegro_hand16ok_.../eval_metrics.csv
NF:  learner/eval CSVs or slurm [ppo] / [Learner] lines.

Eval series: faint raw + bold rolling mean, window=5.
Train series: raw (no smooth).

  python scripts/plot_allegro_hand16ok_rnd_vs_nf.py
  python scripts/plot_allegro_hand16ok_rnd_vs_nf.py --watch 30 --watch-max-sec 3600
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
import plot_allegro_hand16_joint_frac as jf  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RND_LOG_DIR = 'ppo_rnd_allegro_hand16ok_e1024_200m_ep50_4h_seed0'
NF_LOG_DIRS = (
    'ppo_allegro_hand16ok_e1024_nf_tiny_fully_scaled_ep50_200m_dgr_valuedgr_seed0_4h',
)
MPO_LOG_DIRS = (
    'mpo_crl_allegro_hand16ok_e1024_ep50_200m_seed0_4h',
)
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'hand16ok_rnd_3861661_success.png')
STEPS_PER_ITER = 1024 * 50


def _xy_from_pairs(pairs):
  if not pairs:
    return [], []
  xs, ys = zip(*pairs)
  return list(xs), list(ys)


def _nf_train(log_dir: str):
  path = jf._seed_csv('learner', log_dir)
  if path:
    data = jf._read_cols(path, 'global_step', ['train_success_1000'])
    xs, ys = data['train_success_1000']
    if xs:
      return xs, ys, f'csv:{os.path.basename(os.path.dirname(path))}'
  slurm = os.path.join(REPO, 'slurm')
  series = base._read_train_seed_series_from_slurm(slurm, log_dir)
  if series:
    xs, ys = _xy_from_pairs(series[0])
    return xs, ys, 'slurm-train'
  return [], [], ''


def _nf_eval(log_dir: str):
  path = jf._seed_csv('eval', log_dir)
  if path:
    data = jf._read_cols(
        path, 'iteration', ['success'], x_scale=float(STEPS_PER_ITER))
    xs, ys = data['success']
    if xs:
      return xs, ys, f'csv:{os.path.basename(os.path.dirname(path))}'
  slurm = os.path.join(REPO, 'slurm')
  series = base._read_eval_seed_series_from_slurm(slurm, log_dir)
  if series:
    xs = [it * STEPS_PER_ITER for it, _ in series[0]]
    ys = [y for _, y in series[0]]
    return xs, ys, 'slurm-eval'
  return [], [], ''


def _first_nf():
  for log_dir in NF_LOG_DIRS:
    tx, ty, tsrc = _nf_train(log_dir)
    ex, ey, esrc = _nf_eval(log_dir)
    if tx or ex:
      return log_dir, (tx, ty, tsrc), (ex, ey, esrc)
  return None, ([], [], ''), ([], [], '')


def _first_mpo():
  for log_dir in MPO_LOG_DIRS:
    tx, ty, tsrc = _nf_train(log_dir)
    ex, ey, esrc = _nf_eval(log_dir)
    if tx or ex:
      return log_dir, (tx, ty, tsrc), (ex, ey, esrc)
  return None, ([], [], ''), ([], [], '')


def run_once(*, out: str = OUT) -> str:
  rnd_path = jf._rnd_eval_csv(RND_LOG_DIR)
  if rnd_path is None:
    raise SystemExit(f'no RND eval_metrics.csv under logs/{RND_LOG_DIR}/')
  rnd = jf._read_rnd_eval(rnd_path, jf.EVAL_COLS)
  nf_dir, (nfx_tr, nfy_tr, nf_tr_src), (nfx_ev, nfy_ev, nf_ev_src) = _first_nf()
  mpo_dir, (mpx_tr, mpy_tr, mpo_tr_src), (mpx_ev, mpy_ev, mpo_ev_src) = (
      _first_mpo())

  C = base.ACCENT_COLORS
  fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.0), sharex=True)
  w = base.EVAL_SMOOTH_WINDOW

  ax = axes[0]
  for key, label, color, ls in (
      ('success', 'RND eval hard (≤0.10)', C[3], '-'),
      ('easy_success', 'RND eval easy (≤0.20)', C[4], '--'),
      ('very_easy_success', 'RND eval very-easy (≤0.30)', C[2], ':'),
  ):
    xs, ys = rnd[key]
    if not xs:
      continue
    base._plot_eval_smoothed(ax, xs, ys, color=color, label=label, linestyle=ls)
    print(f'RND {key}: n={len(xs)} last={ys[-1]:.4f} peak={max(ys):.4f} '
          f'at {xs[ys.index(max(ys))]/1e6:.2f}M')

  if nfx_ev:
    base._plot_eval_smoothed(
        ax, nfx_ev, nfy_ev, color=C[0],
        label=f'NF eval hard ({nf_ev_src})', linestyle='-')
    print(f'NF eval: n={len(nfx_ev)} last={nfy_ev[-1]:.4f} peak={max(nfy_ev):.4f} '
          f'src={nf_ev_src}')
  if nfx_tr:
    ax.plot(nfx_tr, nfy_tr, color=C[1], lw=2.0,
            label=f'NF train_success_1000 ({nf_tr_src})')
    print(f'NF train: n={len(nfx_tr)} last={nfy_tr[-1]:.4f} peak={max(nfy_tr):.4f} '
          f'src={nf_tr_src}')
  if mpx_ev:
    base._plot_eval_smoothed(
        ax, mpx_ev, mpy_ev, color=C[6],
        label=f'MPO-CRL eval hard ({mpo_ev_src})', linestyle='-.')
    print(f'MPO eval: n={len(mpx_ev)} last={mpy_ev[-1]:.4f} '
          f'peak={max(mpy_ev):.4f} src={mpo_ev_src}')
  if mpx_tr:
    ax.plot(mpx_tr, mpy_tr, color=C[7], lw=2.0,
            label=f'MPO-CRL train_success_1000 ({mpo_tr_src})')
    print(f'MPO train: n={len(mpx_tr)} last={mpy_tr[-1]:.4f} '
          f'peak={max(mpy_tr):.4f} src={mpo_tr_src}')
  if not nfx_ev and not nfx_tr:
    ax.text(
        0.02, 0.92,
        'NF 3858696 logs were deleted — no curve to overlay',
        transform=ax.transAxes, fontsize=8, color='#666666', va='top')
    print('NF: no learner/eval/slurm series found')
  if not mpx_ev and not mpx_tr:
    print('MPO: no learner/eval/slurm series found')

  last_m = 0.0
  if rnd['success'][0]:
    last_m = rnd['success'][0][-1] / 1e6
  ax.set_title(
      f'OK-hand · RND / NF / MPO-CRL · eval rolling mean w={w} · '
      f'RND @ {last_m:.1f}M',
      fontsize=11, fontweight='bold')
  ax.set_ylabel(f'success (eval roll mean w={w}; train raw)', fontsize=10)
  ax.set_ylim(-0.02, 1.05)
  ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3,
      fontsize=8, framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)

  ax = axes[1]
  for key, label, color, ls in (
      ('joint_frac_010', 'RND joint_frac@0.10', C[0], '-'),
      ('joint_frac_020', 'RND joint_frac@0.20', C[1], '--'),
  ):
    xs, ys = rnd[key]
    if not xs:
      continue
    base._plot_eval_smoothed(ax, xs, ys, color=color, label=label, linestyle=ls)
    print(f'RND {key}: n={len(xs)} last={ys[-1]:.4f} peak={max(ys):.4f}')
  xs_e, ys_e = rnd['mean_abs_joint_err']
  if xs_e:
    ax2 = ax.twinx()
    sm = base._rolling_mean(ys_e)
    ax2.plot(xs_e, ys_e, color=C[5], lw=1.0, alpha=0.28)
    ax2.plot(xs_e, sm, color=C[5], lw=2.0, label='RND mean |q-q*|')
    ax2.set_ylabel('mean abs joint err (rad)', fontsize=10, color=C[5])
    ax2.spines[['top']].set_visible(False)
    print(f'RND mae: last={ys_e[-1]:.4f} min={min(ys_e):.4f}')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.22),
        ncol=3, fontsize=8, framealpha=0.95)
  else:
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=3,
        fontsize=8, framealpha=0.95)
  ax.set_title(
      f'RND joint fraction + MAE (rolling mean, window={w})',
      fontsize=11, fontweight='bold')
  ax.set_ylabel(f'joint fraction (roll mean, w={w})', fontsize=10)
  ax.set_xlabel('Env steps', fontsize=10)
  ax.set_ylim(-0.02, 1.05)
  ax.spines[['top']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  for a in axes:
    a.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.tight_layout()
  fig.subplots_adjust(hspace=0.55, bottom=0.16)
  fig.savefig(out, dpi=160)
  plt.close(fig)
  print(f'wrote {out}')
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', default=OUT)
  parser.add_argument('--watch', type=int, default=0)
  parser.add_argument('--watch-max-sec', type=int, default=3600)
  args = parser.parse_args()
  if args.watch <= 0:
    run_once(out=args.out)
    return
  t0 = time.time()
  print(
      f'[hand16ok] watching every {args.watch}s '
      f'(max {args.watch_max_sec}s) → {args.out}',
      flush=True)
  while True:
    try:
      run_once(out=args.out)
    except SystemExit as exc:
      print(f'[hand16ok] {exc}', flush=True)
    except Exception as exc:  # noqa: BLE001
      print(f'[hand16ok] error: {exc}', flush=True)
    if time.time() - t0 >= args.watch_max_sec:
      print('[hand16ok] watch-max-sec reached; exiting', flush=True)
      break
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
