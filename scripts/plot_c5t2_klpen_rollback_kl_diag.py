#!/usr/bin/env python3
"""c5t2 compact-small NF + KL-penalty (rollback, β_min, epoch-adapt): diagnostics.

Job 3718271 seed 0. Train success is raw. Eval success is a centered rolling
mean (window=5) with faint raw underneath. Marks target-KL rollback, actor
reset, and post-reset cooldown. Main figure also shows pg loss and KL-penalty
loss (β · analytic KL). Also writes NF NLL + entropy-loss, and a single-panel
pg / KL-penalty / overall PPO loss figure.

  python scripts/plot_c5t2_klpen_rollback_kl_diag.py
"""
from __future__ import annotations

import csv
import glob
import math
import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR_NAME = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect'
    '_minstd1e5_entanneal_ep60_300m_crl10_klpen1_kltgt001_betamin005'
    '_epbeta_tkl005_rollback')
RUN_DIR = os.path.join(
    REPO, 'logs', LOG_DIR_NAME, 'ppo_builderbench_creative_5_task2_0')
LEARNER_CSV = os.path.join(RUN_DIR, 'logs', 'learner', 'logs.csv')
EVAL_CSV = os.path.join(RUN_DIR, 'logs', 'eval', 'logs.csv')
SLURM_GLOB = os.path.join(REPO, 'slurm', f'{LOG_DIR_NAME}_*.log')
OUT_DIR = os.path.join(REPO, 'figs', 'builderbench', 'active_train_eval')
OUT_PATH = os.path.join(
    OUT_DIR,
    'creative5_task2_nf_compact_small_klpen1_kltgt001_betamin005'
    '_epbeta_tkl005_rollback_kl_diag.png')
OUT_LOSS_PATH = os.path.join(
    OUT_DIR,
    'creative5_task2_nf_compact_small_klpen1_kltgt001_betamin005'
    '_epbeta_tkl005_rollback_nf_ent_loss.png')
OUT_PPO_LOSS_PATH = os.path.join(
    OUT_DIR,
    'creative5_task2_nf_compact_small_klpen1_kltgt001_betamin005'
    '_epbeta_tkl005_rollback_ppo_losses.png')

TARGET_KL = 0.05            # ppo_target_kl (rollback threshold)
KL_PENALTY_TARGET = 0.01    # ppo_kl_penalty_target (adaptive β)
BETA_MIN = 0.05             # ppo_kl_penalty_beta_min
ACTOR_RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')

C_TRAIN = '#2A9D8F'
C_EVAL = '#4C9BE8'
C_APPROX = '#9B2226'
C_APPROX_MAX = '#E84C6F'
C_ANALYTIC = '#E8834C'
C_BETA = '#A84CE8'
C_RB = '#C45C26'
C_RESET = '#4C9BE8'
C_COOL = '#4C9BE8'
C_NF = '#1D6F6A'
C_ENT = '#C45C26'
C_ENT_BETA = '#A84CE8'
C_PG = '#2A9D8F'
C_KL_LOSS = '#A84CE8'
C_TOTAL = '#1D3557'


def _load(path: str) -> dict:
  with open(path, newline='') as fh:
    rows = list(csv.DictReader(fh))
  if not rows:
    raise SystemExit(f'empty csv: {path}')
  out = {k: [] for k in rows[0]}
  for r in rows:
    for k, v in r.items():
      try:
        x = float(v)
        out[k].append(x if math.isfinite(x) else float('nan'))
      except (TypeError, ValueError):
        out[k].append(float('nan'))
  return out


def _series(d: dict, *names):
  for n in names:
    if n in d:
      return d[n]
  raise KeyError(names)


def _actor_reset_steps(slurm_path: str, iter_to_step: dict) -> list[float]:
  if not slurm_path or not os.path.isfile(slurm_path):
    return []
  steps = []
  with open(slurm_path) as fh:
    for line in fh:
      m = ACTOR_RESET_RE.search(line)
      if not m:
        continue
      it = int(m.group(1))
      if it in iter_to_step:
        steps.append(iter_to_step[it])
  return steps


def _cooldown_spans(steps, cooldown) -> list[tuple[float, float]]:
  """Merge consecutive iters with target_kl_cooldown > 0 into [left, right]."""
  spans = []
  i = 0
  n = len(steps)
  while i < n:
    if not (math.isfinite(cooldown[i]) and cooldown[i] > 0):
      i += 1
      continue
    j = i
    while j + 1 < n and math.isfinite(cooldown[j + 1]) and cooldown[j + 1] > 0:
      j += 1
    left = steps[i] if i == 0 else 0.5 * (steps[i - 1] + steps[i])
    right = steps[j] if j + 1 >= n else 0.5 * (steps[j] + steps[j + 1])
    spans.append((left, right))
    i = j + 1
  return spans


def _vlines(ax, xs, *, color, alpha, lw, zorder=1):
  if not xs:
    return
  ax.vlines(xs, 0, 1, transform=ax.get_xaxis_transform(),
            colors=color, alpha=alpha, lw=lw, zorder=zorder)


def _shade_spans(ax, spans, *, color, alpha):
  for left, right in spans:
    ax.axvspan(left, right, color=color, alpha=alpha, lw=0, zorder=0)


def _mark_events(ax, rb_steps, reset_steps, cool_spans):
  _shade_spans(ax, cool_spans, color=C_COOL, alpha=0.10)
  _vlines(ax, rb_steps, color=C_RB, alpha=0.35, lw=0.9)
  _vlines(ax, reset_steps, color=C_RESET, alpha=0.7, lw=1.4)


def _event_handles(rb_steps, reset_steps, cool_spans):
  return [
      Line2D([0], [0], color=C_RB, lw=1.4,
             label=f'KL rollback ({len(rb_steps)} iters)'),
      Line2D([0], [0], color=C_RESET, lw=1.6,
             label=(f'actor reset ({len(reset_steps)})' if reset_steps
                    else 'actor reset (none)')),
      Patch(facecolor=C_COOL, alpha=0.25, edgecolor='none',
            label=f'KL-stop cooldown ({len(cool_spans)} span'
                  f'{"s" if len(cool_spans) != 1 else ""})'),
  ]


def _legend(ax, axes, extra):
  h, lab = [], []
  for a in axes:
    hi, li = a.get_legend_handles_labels()
    h.extend(hi)
    lab.extend(li)
  seen = set()
  h2, lab2 = [], []
  for handle, label in zip(h, lab):
    if label in seen:
      continue
    seen.add(label)
    h2.append(handle)
    lab2.append(label)
  ax.legend(h2 + extra, lab2 + [e.get_label() for e in extra],
            fontsize=7.5, frameon=False, loc='upper right', ncol=2)


def _style(ax, ylabel: str, xlabel: bool = False):
  ax.set_ylabel(ylabel, fontsize=9)
  ax.tick_params(labelsize=8)
  ax.grid(True, alpha=0.25, lw=0.4)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  if xlabel:
    ax.set_xlabel('env steps', fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(base._fmt_steps))
  else:
    ax.tick_params(labelbottom=False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(base._fmt_steps))


def _plot_nf_ent_loss(d, steps, beta, rb_steps, reset_steps, cool_spans,
                      xmax) -> None:
  nll = _series(d, 'nf/density_loss')
  ent_loss = _series(d, 'ppo/entropy_loss')
  ent_beta = [e * b for e, b in zip(ent_loss, beta)]

  fig, axes = plt.subplots(
      2, 1, figsize=(11.6, 6.4), sharex=True,
      gridspec_kw={'height_ratios': [1.0, 1.05], 'hspace': 0.08})
  ax_n, ax_e = axes
  for ax in axes:
    ax.set_xlim(0, xmax)
    _mark_events(ax, rb_steps, reset_steps, cool_spans)

  ax_n.plot(steps, nll, color=C_NF, lw=1.5, label='NF NLL (density_loss)')
  ax_n.axhline(0.0, color='0.5', lw=0.5, ls=':')
  _style(ax_n, 'NF NLL')
  ax_n.set_title(
      'c5t2 NF compact-small  ·  NF loss + entropy loss  ·  seed 0',
      fontsize=10, loc='left', pad=8)

  ax_e.plot(steps, ent_loss, color=C_ENT, lw=1.5,
            label='entropy loss  (−ent_coef · H)')
  ax_e.plot(steps, ent_beta, color=C_ENT_BETA, lw=1.35,
            label='β × entropy loss')
  ax_e.axhline(0.0, color='0.5', lw=0.5, ls=':')
  # One β=51 point at the actor reset would squash the rest of the series.
  finite_eb = [x for x in ent_beta if math.isfinite(x)]
  finite_el = [x for x in ent_loss if math.isfinite(x)]
  spike = min(finite_eb) if finite_eb else 0.0
  typical = sorted(finite_eb)[max(0, int(0.02 * len(finite_eb)))]
  if spike < typical * 4:  # large negative outlier
    ymin = min(min(finite_el), typical) * 1.25
    ax_e.set_ylim(ymin, 0.02)
    if reset_steps:
      ax_e.annotate(
          f'β×ent={spike:.2f} at actor reset',
          xy=(reset_steps[0], ymin * 0.92),
          xytext=(xmax * 0.18, ymin * 0.55),
          fontsize=8, color=C_ENT_BETA,
          arrowprops=dict(arrowstyle='->', color=C_ENT_BETA, lw=0.8),
      )
  _style(ax_e, 'entropy loss', xlabel=True)

  extra = _event_handles(rb_steps, reset_steps, cool_spans)
  _legend(ax_n, axes, extra)

  os.makedirs(os.path.dirname(OUT_LOSS_PATH), exist_ok=True)
  fig.savefig(OUT_LOSS_PATH, dpi=160)
  plt.close(fig)
  print(f'wrote {OUT_LOSS_PATH}')


def _plot_ppo_losses(d, steps, rb_steps, reset_steps, cool_spans,
                     xmax) -> None:
  pg = _series(d, 'ppo/pg_loss')
  kl = _series(d, 'ppo/kl_penalty_loss')  # already β · analytic KL
  # Logged ppo_total_loss is the core objective (pg − ent + vf·v) *before*
  # the KL penalty is added. Reconstruct the optimized total.
  core = _series(d, 'ppo/ppo_total_loss')
  overall = [c + k for c, k in zip(core, kl)]

  fig, ax = plt.subplots(figsize=(11.6, 4.8))
  ax.set_xlim(0, xmax)
  _mark_events(ax, rb_steps, reset_steps, cool_spans)

  ax.plot(steps, overall, color=C_TOTAL, lw=1.7, label='overall  (core + β·KL)')
  ax.plot(steps, pg, color=C_PG, lw=1.35, label='pg loss')
  ax.plot(steps, kl, color=C_KL_LOSS, lw=1.35, label='KL loss  (β · analytic KL)')
  ax.axhline(0.0, color='0.5', lw=0.5, ls=':')

  # Iters 0–1 have v_loss ~1e4, which would squash the rest of the run.
  skip = 2
  later = (overall[skip:] + pg[skip:] + kl[skip:])
  later = [x for x in later if math.isfinite(x)]
  if later:
    lo, hi = min(later), max(later)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    if abs(overall[0]) > hi + pad:
      ax.annotate(
          f'overall={overall[0]:.0f} at iter 0  (v_loss spike)',
          xy=(steps[0], hi),
          xytext=(xmax * 0.12, hi - pad * 0.2),
          fontsize=8, color=C_TOTAL,
          arrowprops=dict(arrowstyle='->', color=C_TOTAL, lw=0.8),
      )

  _style(ax, 'loss', xlabel=True)
  ax.set_title(
      'c5t2 NF compact-small  ·  pg / KL / overall PPO loss  ·  seed 0',
      fontsize=10, loc='left', pad=8)
  extra = _event_handles(rb_steps, reset_steps, cool_spans)
  _legend(ax, [ax], extra)

  os.makedirs(os.path.dirname(OUT_PPO_LOSS_PATH), exist_ok=True)
  fig.savefig(OUT_PPO_LOSS_PATH, dpi=160)
  plt.close(fig)
  print(f'wrote {OUT_PPO_LOSS_PATH}')


def main() -> None:
  d = _load(LEARNER_CSV)
  ev = _load(EVAL_CSV)
  steps = _series(d, 'global_step')
  it = _series(d, 'iteration')
  iter_to_step = {int(i): s for i, s in zip(it, steps)}

  rb = _series(d, 'ppo/kl_rollback')
  rb_steps = [s for s, x in zip(steps, rb) if x > 0.5]
  cool = _series(d, 'ppo/target_kl_cooldown')
  cool_spans = _cooldown_spans(steps, cool)

  slurm_matches = sorted(glob.glob(SLURM_GLOB), key=os.path.getmtime,
                         reverse=True)
  slurm_path = slurm_matches[0] if slurm_matches else ''
  reset_steps = _actor_reset_steps(slurm_path, iter_to_step)

  eval_it = _series(ev, 'iteration')
  eval_succ = _series(ev, 'success')
  spi = float(steps[0]) if steps and it and it[0] == 0 else 1024 * 60
  eval_steps = [
      iter_to_step.get(int(i), (i + 1.0) * spi) for i in eval_it]

  beta = _series(d, 'ppo/kl_beta')

  pg = _series(d, 'ppo/pg_loss')
  kl_loss = _series(d, 'ppo/kl_penalty_loss')  # already β · analytic KL

  fig, axes = plt.subplots(
      4, 1, figsize=(11.6, 10.4), sharex=True,
      gridspec_kw={'height_ratios': [1.05, 1.0, 1.15, 0.95], 'hspace': 0.08})
  ax_s, ax_l, ax_k, ax_b = axes
  xmax = steps[-1] if steps else 1.0
  for ax in axes:
    ax.set_xlim(0, xmax)
    _mark_events(ax, rb_steps, reset_steps, cool_spans)

  ax_s.plot(steps, _series(d, 'train_success_1000'), color=C_TRAIN, lw=1.5,
            label='train success (last 1000)')
  base._plot_eval_smoothed(
      ax_s, eval_steps, eval_succ, color=C_EVAL,
      label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})',
      linewidth=2.0)
  ax_s.set_ylim(-0.05, 1.05)
  _style(ax_s, 'success')
  ax_s.set_title(
      'c5t2 NF compact-small  ·  KL-pen β₀=1  target=0.01  β_min=0.05  ·  '
      'rollback tkl=0.05  ·  catselect ep60  ·  seed 0',
      fontsize=10, loc='left', pad=8)

  ax_l.plot(steps, pg, color=C_PG, lw=1.35, label='pg loss')
  ax_l.plot(steps, kl_loss, color=C_KL_LOSS, lw=1.35,
            label='KL loss  (β · analytic KL)')
  ax_l.axhline(0.0, color='0.5', lw=0.5, ls=':')
  skip = 2
  later = [x for x in (pg[skip:] + kl_loss[skip:]) if math.isfinite(x)]
  if later:
    lo, hi = min(later), max(later)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    ax_l.set_ylim(lo - pad, hi + pad)
  _style(ax_l, 'loss')

  ax_k.plot(steps, _series(d, 'ppo/approx_kl_max'), color=C_APPROX_MAX,
            lw=0.9, alpha=0.35, label='approx KL max')
  ax_k.plot(steps, _series(d, 'ppo/approx_kl'), color=C_APPROX, lw=1.35,
            label='approx KL (exact)')
  ax_k.plot(steps, _series(d, 'ppo/analytic_kl'), color=C_ANALYTIC, lw=1.25,
            label='analytic KL')
  ax_k.axhline(TARGET_KL, color=C_RB, lw=0.9, ls='--',
               label=f'target KL rollback = {TARGET_KL:g}')
  ax_k.axhline(KL_PENALTY_TARGET, color=C_BETA, lw=0.8, ls=':',
               label=f'β-adapt target = {KL_PENALTY_TARGET:g}')
  ax_k.set_ylim(bottom=-0.005)
  _style(ax_k, 'KL')

  ax_b.plot(steps, beta, color=C_BETA, lw=1.5, label='β (KL penalty)')
  ax_b.set_yscale('log')
  bmax = max(b for b in beta if math.isfinite(b)) if beta else 1.0
  ax_b.set_ylim(BETA_MIN * 0.6, max(bmax * 1.4, 2.0))
  ax_b.axhline(1.0, color='0.5', lw=0.6, ls=':')
  ax_b.axhline(BETA_MIN, color=C_BETA, lw=0.8, ls='--',
               label=f'β_min = {BETA_MIN:g}')
  _style(ax_b, 'β  (log)', xlabel=True)

  extra = _event_handles(rb_steps, reset_steps, cool_spans)
  _legend(ax_s, axes, extra)

  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.savefig(OUT_PATH, dpi=160)
  plt.close(fig)
  print(f'wrote {OUT_PATH}')
  print(f'  iters={len(steps)}  rollbacks={len(rb_steps)}  '
        f'actor-resets={len(reset_steps)}  cooldown_spans={len(cool_spans)}  '
        f'slurm={os.path.basename(slurm_path) or "missing"}')

  _plot_nf_ent_loss(d, steps, beta, rb_steps, reset_steps, cool_spans, xmax)
  _plot_ppo_losses(d, steps, rb_steps, reset_steps, cool_spans, xmax)


if __name__ == '__main__':
  main()
