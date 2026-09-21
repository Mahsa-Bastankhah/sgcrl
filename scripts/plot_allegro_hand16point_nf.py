#!/usr/bin/env python3
"""Point-finger NF screen (compact-small, no grad-reg).

NF-only until this pose is solved.

  python scripts/plot_allegro_hand16point_nf.py
  python scripts/plot_allegro_hand16point_nf.py --watch 30 --watch-max-sec 3600
"""
from __future__ import annotations

import argparse
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
NF_LOG_DIR = (
    'ppo_allegro_hand16point_e1024_nf_compactsmall_fully_scaled_ep50_200m_seed0_4h'
)
MPO_LOG_DIR = 'mpo_crl_allegro_hand16point_e1024_ep50_200m_seed0_4h'
OUT = os.path.join(
    REPO, 'figs', 'allegro_kuka_throw', 'hand16point_nf_success.png')
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


def run_once(*, out: str = OUT) -> str:
  nfx_tr, nfy_tr, nf_tr_src = _nf_train(NF_LOG_DIR)
  nfx_ev, nfy_ev, nf_ev_src = _nf_eval(NF_LOG_DIR)
  mpx_tr, mpy_tr, mpo_tr_src = _nf_train(MPO_LOG_DIR)
  mpx_ev, mpy_ev, mpo_ev_src = _nf_eval(MPO_LOG_DIR)

  C = base.ACCENT_COLORS
  fig, ax = plt.subplots(1, 1, figsize=(10.4, 4.6))
  w = base.EVAL_SMOOTH_WINDOW

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
        'waiting for NF learner/eval/slurm series',
        transform=ax.transAxes, fontsize=8, color='#666666', va='top')
    print('NF: no learner/eval/slurm series found')

  last_m = 0.0
  if nfx_tr:
    last_m = nfx_tr[-1] / 1e6
  elif nfx_ev:
    last_m = nfx_ev[-1] / 1e6
  ax.set_title(
      f'Point · NF + MPO · eval roll mean w={w} · '
      f'@ {last_m:.1f}M',
      fontsize=11, fontweight='bold')
  ax.set_ylabel(f'success (eval roll mean w={w}; train raw)', fontsize=10)
  ax.set_xlabel('Env steps', fontsize=10)
  ax.set_ylim(-0.02, 1.05)
  ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2,
      fontsize=8, framealpha=0.95)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))

  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.22)
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
      f'[hand16point] watching every {args.watch}s '
      f'(max {args.watch_max_sec}s) → {args.out}',
      flush=True)
  while True:
    try:
      run_once(out=args.out)
    except Exception as exc:  # noqa: BLE001
      print(f'[hand16point] error: {exc}', flush=True)
    if time.time() - t0 >= args.watch_max_sec:
      print('[hand16point] watch-max-sec reached; exiting', flush=True)
      return
    time.sleep(args.watch)


if __name__ == '__main__':
  main()
