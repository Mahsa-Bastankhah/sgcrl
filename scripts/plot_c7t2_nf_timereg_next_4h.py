#!/usr/bin/env python3
"""c7t2 next-step time-reg 4h probes: success, NF NLL, time-reg / |NLL|.

Train is raw ``train_success_1000``. Eval is faint raw + bold rolling mean
(window=5). Ratio is ``nf_time_reg / |NLL|`` as a percent (log y).

  python scripts/plot_c7t2_nf_timereg_next_4h.py
"""
from __future__ import annotations

import csv
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = 'ppo_builderbench_creative_7_task2_0'
# 1024 envs × rollout 70.
STEPS_PER_ITER = 1024 * 70
OUT = os.path.join(
    REPO, 'figs', 'builderbench', 'c7t2',
    'c7t2_nf_timereg_next_4h.png')

PREFIX = (
    'ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1'
    '_minstd1e5_ent05to001_ep70_300m_crl10_timereg_next')

RUNS = (
    {
        'key': 'iter',
        'dir': PREFIX + '_iter_eta1_warp_4h',
        'label': r'iter  $\eta=1$   3751949  done',
        'color': '#C45C26',
    },
    {
        'key': 'ema',
        'dir': PREFIX + '_ema_eta001_warp_4h',
        'label': r'ema  $\eta=0.01$   3751950  live',
        'color': '#2A9D8F',
    },
)


def _csv(log_dir: str, split: str) -> str:
  return os.path.join(
      REPO, 'logs', log_dir, RUN, 'logs', split, 'logs.csv')


def _load(path: str) -> dict:
  with open(path, newline='') as fh:
    rows = list(csv.DictReader(fh))
  if not rows:
    return {}
  out = {k: [] for k in rows[0]}
  for r in rows:
    for k, v in r.items():
      x = base._coerce(v)
      out[k].append(float('nan') if x is None else x)
  return out


def _series(d: dict, *names):
  for n in names:
    if n in d:
      return d[n]
  return None


def _xy(d: dict, yname: str):
  ys = _series(d, yname)
  xs = _series(d, 'global_step')
  if xs is None or ys is None:
    return [], []
  xo, yo = [], []
  for x, y in zip(xs, ys):
    if x == x and y == y and math.isfinite(x) and math.isfinite(y):
      xo.append(x)
      yo.append(y)
  return xo, yo


def _peak(xs, ys):
  i = max(range(len(ys)), key=lambda k: ys[k])
  return ys[i], xs[i]


def _style(ax, *, title: str, ylabel: str, xlabel: bool):
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.set_ylabel(ylabel, fontsize=9)
  if xlabel:
    ax.set_xlabel('Env Steps', fontsize=9)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.tick_params(labelsize=8)


def main() -> None:
  data = []
  for spec in RUNS:
    lp = _csv(spec['dir'], 'learner')
    ep = _csv(spec['dir'], 'eval')
    if not os.path.isfile(lp):
      raise SystemExit(f'missing {lp}')
    learner = _load(lp)
    ev = _load(ep) if os.path.isfile(ep) else {}
    data.append((spec, learner, ev))

  xmax = 0.0
  for _, learner, _ in data:
    xs = _series(learner, 'global_step') or [0.0]
    xmax = max(xmax, xs[-1])

  w = base.EVAL_SMOOTH_WINDOW
  fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), sharex=True)
  fig.subplots_adjust(
      left=0.07, right=0.98, top=0.88, bottom=0.18, wspace=0.22, hspace=0.32)
  ax_tr, ax_ev, ax_nll, ax_ratio = (
      axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])

  for spec, learner, ev in data:
    txs, tys = _xy(learner, 'train_success_1000')
    txs, tys, _ = base._subsample_curve(txs, tys, [0.0] * len(tys))
    ax_tr.plot(
        txs, tys, color=spec['color'], lw=2.0, label=spec['label'], zorder=3)

    e_it = _series(ev, 'iteration') or []
    e_ys = _series(ev, 'success') or []
    exs, eys = [], []
    for it, y in zip(e_it, e_ys):
      if it == it and y == y:
        exs.append(it * STEPS_PER_ITER)
        eys.append(y)
    if exs:
      base._plot_eval_smoothed(
          ax_ev, exs, eys, color=spec['color'], linestyle='-',
          label=spec['label'], linewidth=2.0)

    nxs, nys = _xy(learner, 'nf/density_loss')
    nxs, nys, _ = base._subsample_curve(nxs, nys, [0.0] * len(nys))
    ax_nll.plot(nxs, nys, color=spec['color'], lw=1.6, label=spec['label'])

    rxs, rys = _xy(learner, 'nf/nf_time_reg_nll_ratio')
    pct = [max(100.0 * y, 1e-4) for y in rys]
    rxs, pct, _ = base._subsample_curve(rxs, pct, [0.0] * len(pct))
    ax_ratio.plot(rxs, pct, color=spec['color'], lw=1.6, label=spec['label'])

    t_all_x, t_all_y = _xy(learner, 'train_success_1000')
    pk, px = _peak(t_all_x, t_all_y)
    nll_tail = nys[-20:]
    ratio_tail = pct[-20:]
    print(
        f"{spec['key']:4s}  n={len(t_all_y):4d}  "
        f"steps={t_all_x[-1]/1e6:.1f}M  "
        f"train last={t_all_y[-1]:.3f} peak={pk:.3f} @{px/1e6:.1f}M  "
        f"NLL={sum(nll_tail)/len(nll_tail):.3f}  "
        f"ratio%={sum(ratio_tail)/len(ratio_tail):.3g}")

  for ax in (ax_tr, ax_ev):
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, xmax)

  _style(
      ax_tr, title='c7t2 train',
      ylabel='Train Success (last 1000)', xlabel=False)
  _style(
      ax_ev, title=f'c7t2 eval (roll mean w={w})',
      ylabel=f'Eval Success (roll mean w={w})', xlabel=False)
  _style(
      ax_nll, title='NF NLL',
      ylabel='NF NLL  (density_loss)', xlabel=True)
  _style(
      ax_ratio, title=r'time-reg share of |NLL|',
      ylabel=r'time-reg $/$ $|$NLL$|$  (%)', xlabel=True)
  ax_ratio.set_yscale('log')
  for y, lab, ls in ((1.0, '1%', ':'), (10.0, '10%', '--'), (100.0, '100%', '-.')):
    ax_ratio.axhline(y, color='0.55', lw=0.7, ls=ls)
    ax_ratio.text(
        xmax * 0.99, y, f' {lab}', va='bottom', ha='right',
        fontsize=7, color='0.4')

  ax_tr.legend(fontsize=8, frameon=False, loc='lower right')
  ax_ev.legend().remove()
  ax_nll.legend(fontsize=8, frameon=False, loc='upper right')
  ax_ratio.legend(fontsize=8, frameon=False, loc='upper right')

  fig.suptitle(
      'c7t2 NF compact-small catwp  ·  next-step time-reg  ·  no grad-reg\n'
      r'ent 0.05$\to$0.01  ·  ep=70  ·  warp 4h  ·  eval = rolling mean, window=5',
      fontsize=12, fontweight='bold', y=0.99,
  )
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  tmp = OUT + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT)
  plt.close(fig)
  print(f'→ {OUT}')


if __name__ == '__main__':
  main()
