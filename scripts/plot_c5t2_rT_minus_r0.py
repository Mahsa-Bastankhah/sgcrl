#!/usr/bin/env python3
"""c5t2 r_T-r_0 vs train/eval success and policy KL.

Panels: ``repr_rT_minus_r0`` (raw), train/eval success, ``ppo/approx_kl``
(raw; this run has no KL penalty so no analytic KL).

  python scripts/plot_c5t2_rT_minus_r0.py
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(REPO, 'figs', 'builderbench', 'c5t2')

RUN_GR = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect'
    '_dualgradreg_c100_lamlr1e6_ent05to001'
)
RUN_NOGR = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect'
    '_ent0005to0001'
)
RUN_C8_EP60 = (
    'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
    '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp'
    '_extrew1_minstd1e5_ent005_to001_ep60_300m_crl10'
    '_dualgradreg_c100_lamlr1e6'
)
RUN_TGF20 = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect'
    '_dualgradreg_c100_lamlr1e6_ent05to001_taskgfrac20'
)
RUN_TGF50 = (
    'ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small'
    '_sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catselect'
    '_dualgradreg_c100_lamlr1e6_ent05to001_taskgfrac50'
)

RESET_RE = re.compile(r'\[ppo\] actor reset at iter=(\d+)')


def _actor_reset_steps(log_dir: str) -> list[int]:
  """Env-step of each slurm ``actor reset`` (not in the learner CSV)."""
  it_to_step = base._iter_to_env_steps_map(base.LOG_ROOT, log_dir)
  iters: list[int] = []
  for path in base._slurm_logs_for_log_dir(base.SLURM_DIR, log_dir):
    try:
      with open(path, 'r', errors='replace') as fh:
        for line in fh:
          m = RESET_RE.search(line)
          if m:
            iters.append(int(m.group(1)))
    except OSError:
      continue
  steps = []
  for it in sorted(set(iters)):
    if it in it_to_step:
      steps.append(it_to_step[it])
    elif it_to_step:
      spi = base._infer_steps_per_iter(it_to_step)
      steps.append(it * spi)
  return steps


SERIES = (
    ('repr_rT_minus_r0_mean_succ', 'successful episodes',
     base.ACCENT_COLORS[2]),
    ('repr_rT_minus_r0_mean_fail', 'unsuccessful episodes',
     base.ACCENT_COLORS[1]),
)


def _plot_col(ax_r, ax_s, ax_kl, log_dir: str, title: str) -> None:
  for y_col, label, color in SERIES:
    seeds = base._read_csv_seed_series(
        base.LOG_ROOT, log_dir, split='learner',
        x_col='global_step', y_col=y_col)
    xs, mean, se, n = base._aggregate_mean_stderr(seeds)
    if not xs:
      print(f'  {title} / {label}: no points')
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    ax_r.plot(xs, mean, color=color, linewidth=2.0, alpha=0.95, label=label)
    print(f'  {title} / {label}: n={n} last={mean[-1]:+.3f} '
          f'peak={max(mean):+.3f} trough={min(mean):+.3f}')

  ax_r.axhline(0.0, color='#8899aa', lw=1.0, ls='--', alpha=0.8, zorder=1)
  ax_r.set_title(title, fontsize=10, fontweight='bold')
  ax_r.set_ylabel(r'mean $r(s_T,a_T)-r(s_0,a_0)$', fontsize=9)
  ax_r.spines[['top', 'right']].set_visible(False)
  ax_r.grid(axis='y', linestyle='--', alpha=0.4)
  ax_r.legend(loc='upper right', fontsize=8, framealpha=0.95)

  t_seeds = base._read_train_seed_series(base.LOG_ROOT, log_dir)
  e_seeds = base._read_eval_seed_series(base.LOG_ROOT, log_dir)
  txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
  exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
  if txs:
    xs, mean, _se = base._subsample_curve(txs, tmean, tse)
    ax_s.plot(
        xs, mean, color=base.ACCENT_COLORS[0], linewidth=2.0, alpha=0.95,
        label='train success (last 1000)', zorder=3)
    print(f'  {title} / train: n={n_t} last={tmean[-1]:.3f} peak={max(tmean):.3f}')
  if exs:
    sm = base._plot_eval_smoothed(
        ax_s, exs, emean, color='#7a5cff',
        label=f'eval success (roll mean w={base.EVAL_SMOOTH_WINDOW})',
        zorder=4)
    print(f'  {title} / eval: n={n_e} raw last={emean[-1]:.3f} '
          f'smooth last={sm[-1]:.3f} peak={max(emean):.3f}')

  ax_s.set_ylabel('Success', fontsize=9)
  ax_s.set_ylim(-0.05, 1.05)
  ax_s.spines[['top', 'right']].set_visible(False)
  ax_s.grid(axis='y', linestyle='--', alpha=0.4)
  ax_s.legend(loc='upper right', fontsize=8, framealpha=0.95)

  def _pctl(vals, q):
    if not vals:
      return 0.0
    s = sorted(vals)
    return float(s[max(0, min(len(s) - 1, int(q * (len(s) - 1))))])

  kl_hi_candidates = []
  for y_col, label, color, lw, alpha, q_ylim in (
      ('ppo/approx_kl',
       r'approx KL mean  $\mathbb{E}[(r-1)-\log r]$',
       '#c44e52', 2.0, 0.95, 0.99),
      ('ppo/approx_kl_max',
       'approx KL max (over PPO minibatches / epoch)',
       '#e08a8c', 1.4, 0.85, 0.95),
  ):
    seeds = base._read_csv_seed_series(
        base.LOG_ROOT, log_dir, split='learner',
        x_col='global_step', y_col=y_col)
    xs, mean, se, n = base._aggregate_mean_stderr(seeds)
    if not xs:
      print(f'  {title} / {label}: no points')
      continue
    xs_s, mean_s, _se = base._subsample_curve(xs, mean, se)
    ax_kl.plot(
        xs_s, mean_s, color=color, linewidth=lw, alpha=alpha, label=label)
    kl_hi_candidates.append(_pctl(mean, q_ylim))
    print(f'  {title} / {label}: n={n} last={mean[-1]:.4g} '
          f'peak={max(mean):.4g}')
  if kl_hi_candidates:
    hi = max(max(kl_hi_candidates) * 1.15, 1e-3)
    ax_kl.set_ylim(0.0, hi)
    print(f'  {title} / KL ylim={hi:.4g}')

  ax_kl.set_xlabel('Env Steps', fontsize=9)
  ax_kl.set_ylabel('Policy KL', fontsize=9)
  ax_kl.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax_kl.spines[['top', 'right']].set_visible(False)
  ax_kl.grid(axis='y', linestyle='--', alpha=0.4)
  ax_kl.legend(loc='upper right', fontsize=8, framealpha=0.95)

  resets = _actor_reset_steps(log_dir)
  print(f'  {title} / actor resets: {len(resets)}  steps={resets}')
  for i, x in enumerate(resets):
    lbl = 'actor reset (last-layer)' if i == 0 else None
    for ax in (ax_r, ax_s, ax_kl):
      ax.axvline(
          x, color='#6b4c9a', ls='--', lw=1.15, alpha=0.75,
          zorder=5, label=lbl)
      lbl = None
  if resets:
    ax_r.legend(loc='upper right', fontsize=8, framealpha=0.95)


def _plot_taskgfrac_succ() -> None:
  """Sparse successful-episode r_T-r_0 for taskgfrac 20 vs 50."""
  fig, (ax_r, ax_s) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True)
  runs = (
      (RUN_TGF20, 'taskgfrac=0.20  (3744807)', base.ACCENT_COLORS[0]),
      (RUN_TGF50, 'taskgfrac=0.50  (3744806)', base.ACCENT_COLORS[1]),
  )
  for log_dir, label, color in runs:
    seeds = base._read_csv_seed_series(
        base.LOG_ROOT, log_dir, split='learner',
        x_col='global_step', y_col='repr_rT_minus_r0_mean_succ')
    xs, mean, se, n = base._aggregate_mean_stderr(seeds)
    if not xs:
      print(f'  {label} / succ rT-r0: no points')
      continue
    ax_r.plot(
        xs, mean, color=color, linewidth=1.4, alpha=0.9, marker='o',
        markersize=3.5, label=label)
    print(f'  {label} / succ rT-r0: n={n} last={mean[-1]:+.3f} '
          f'peak={max(mean):+.3f} trough={min(mean):+.3f}')

    t_seeds = base._read_train_seed_series(base.LOG_ROOT, log_dir)
    txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
    if txs:
      xs_t, mean_t, _ = base._subsample_curve(txs, tmean, tse)
      ax_s.plot(
          xs_t, mean_t, color=color, linewidth=2.0, alpha=0.95, label=label)
      print(f'  {label} / train: n={n_t} last={tmean[-1]:.3f} '
            f'peak={max(tmean):.3f}')

  ax_r.axhline(0.0, color='#8899aa', lw=1.0, ls='--', alpha=0.8, zorder=1)
  ax_r.set_ylabel(r'mean $r(s_T,a_T)-r(s_0,a_0)$  (successes only)', fontsize=9)
  ax_r.set_title(
      r'c5t2 dualgradreg  —  successful-episode $r_T-r_0$',
      fontsize=11, fontweight='bold')
  ax_r.spines[['top', 'right']].set_visible(False)
  ax_r.grid(axis='y', linestyle='--', alpha=0.4)
  ax_r.legend(loc='upper right', fontsize=8, framealpha=0.95)

  ax_s.set_ylabel('train success (last 1000)', fontsize=9)
  ax_s.set_xlabel('Env Steps', fontsize=9)
  ax_s.set_ylim(-0.002, 0.012)
  ax_s.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax_s.spines[['top', 'right']].set_visible(False)
  ax_s.grid(axis='y', linestyle='--', alpha=0.4)
  ax_s.legend(loc='upper right', fontsize=8, framealpha=0.95)

  fig.tight_layout()
  _save(fig, os.path.join(FIG_DIR, 'c5t2_taskgfrac20_50_succ_rT_minus_r0.png'))


def _save(fig, path: str) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = path + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, path)
  plt.close(fig)
  print(f'→ {path}')


def main() -> None:
  fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.6), sharex=True)
  print('single-run 3744774')
  _plot_col(
      axes[0], axes[1], axes[2], RUN_GR,
      r'dualgradreg c=100  ep=50 T=50 (3744774)  —  $r_T-r_0$ / success / policy KL')
  fig.tight_layout()
  _save(fig, os.path.join(FIG_DIR, 'c5t2_ent05to001_ep50_T50_rT_minus_r0.png'))

  fig, axes = plt.subplots(3, 2, figsize=(12.8, 8.6), sharey=False)
  print('comparison')
  _plot_col(
      axes[0, 0], axes[1, 0], axes[2, 0], RUN_GR,
      r'dualgradreg c=100 · ent 0.05→0.01  (200M)')
  _plot_col(
      axes[0, 1], axes[1, 1], axes[2, 1], RUN_NOGR,
      r'no grad-reg · ent 0.0005→0.0001  (stopped ~25M)')
  y0 = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
  y1 = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
  axes[0, 0].set_ylim(y0, y1)
  axes[0, 1].set_ylim(y0, y1)
  fig.suptitle(
      r'c5t2 compact-small  —  $r_T-r_0$ vs success vs policy KL',
      fontsize=12, fontweight='bold', y=1.01)
  fig.tight_layout()
  _save(fig, os.path.join(FIG_DIR, 'c5t2_gradreg_vs_nogradreg_rT_minus_r0.png'))

  fig, axes = plt.subplots(3, 1, figsize=(9.6, 8.6), sharex=True)
  print('c8t2 ep60 3744740')
  _plot_col(
      axes[0], axes[1], axes[2], RUN_C8_EP60,
      r'c8t2 dualgradreg c=100  ep=60  (3744740)  —  $r_T-r_0$ / success / policy KL')
  fig.tight_layout()
  _save(fig, os.path.join(
      REPO, 'figs', 'builderbench', 'c8t2',
      'c8t2_ep60_dgr_c100_rT_minus_r0.png'))

  print('taskgfrac 20 vs 50 successful rT-r0')
  _plot_taskgfrac_succ()


if __name__ == '__main__':
  main()
