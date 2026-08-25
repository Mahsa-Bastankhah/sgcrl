#!/usr/bin/env python3
"""c8t2 T=100 dualgradreg successes: ∇_s log p and r_T-r_0.

Raw learner traces. Overlay the two T=100 jobs that locked 2cm success.

  python scripts/plot_c8t2_t100_success_rT_grad_s.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAD_REG_C = 100.0
RUNS = (
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent005_to001_ep100_300m_crl10_dualgradreg_c100'
        '_lamlr1e6_36h',
        'ent 0.05→0.01 · T=100 · job 3748244',
        base.ACCENT_COLORS[1],
    ),
    (
        'ppo_builderbench_creative8_task2_e1024_pd_nf_compact'
        '_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catwp_extrew1'
        '_minstd1e5_ent05_ep100_300m_crl10_dualgradreg_c100_lamlr1e6_36h',
        'ent=0.05 · T=100 · job 3748245',
        base.ACCENT_COLORS[0],
    ),
)
OUT_PATH = os.path.join(
    REPO, 'figs', 'builderbench', 'c8t2',
    'c8t2_t100_success_rT_grad_s.png')


def _curve(log_dir: str, y_col: str):
  seeds = base._read_csv_seed_series(
      base.LOG_ROOT, log_dir, split='learner',
      x_col='global_step', y_col=y_col)
  xs, mean, se, n = base._aggregate_mean_stderr(seeds)
  if not xs:
    return [], [], n
  xs, mean, _se = base._subsample_curve(xs, mean, se)
  return xs, mean, n


def main() -> None:
  fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=True)
  ax_g, ax_r = axes

  for name, label, color in RUNS:
    xs, ys, n = _curve(name, 'nf/nf_logp_grad_s_norm_mean')
    if xs:
      ax_g.plot(xs, ys, color=color, lw=2.0, alpha=0.95, label=label)
      print(f'{label} / grad_s mean: n={n} last={ys[-1]:.3f} '
            f'peak={max(ys):.3f}')
    else:
      print(f'{label} / grad_s mean: no data')

    for y_col, ep_label, ls, lw in (
        ('repr_rT_minus_r0_mean_succ', 'succ', '-', 2.0),
        ('repr_rT_minus_r0_mean_fail', 'fail', '--', 1.6),
    ):
      xs, ys, n = _curve(name, y_col)
      if not xs:
        print(f'{label} / rT-r0 {ep_label}: no data')
        continue
      ax_r.plot(
          xs, ys, color=color, lw=lw, ls=ls, alpha=0.95,
          label=f'{label} · {ep_label}')
      print(f'{label} / rT-r0 {ep_label}: n={n} last={ys[-1]:+.3f} '
            f'peak={max(ys):+.3f} trough={min(ys):+.3f}')

  ax_g.axhline(
      GRAD_REG_C, color='0.35', lw=0.9, ls=':', zorder=0,
      label=f'c={GRAD_REG_C:g}')
  ax_g.set_title(
      r'c8t2 NF dualgradreg T=100 — mean $\|\nabla_s \log p\|$',
      fontsize=11, fontweight='bold')
  ax_g.set_ylabel(r'mean $\|\nabla_s \log p\|$', fontsize=10)
  ax_g.spines[['top', 'right']].set_visible(False)
  ax_g.grid(axis='y', linestyle='--', alpha=0.4)
  ax_g.legend(loc='upper right', fontsize=8, framealpha=0.95)

  ax_r.axhline(0.0, color='#8899aa', lw=1.0, ls='--', alpha=0.8, zorder=1)
  ax_r.set_title(
      r'c8t2 NF dualgradreg T=100 — $r(s_T,a_T)-r(s_0,a_0)$',
      fontsize=11, fontweight='bold')
  ax_r.set_ylabel(r'mean $r_T-r_0$', fontsize=10)
  ax_r.set_xlabel('Env Steps', fontsize=10)
  ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax_r.spines[['top', 'right']].set_visible(False)
  ax_r.grid(axis='y', linestyle='--', alpha=0.4)
  ax_r.legend(loc='upper right', fontsize=7.5, framealpha=0.95)

  fig.suptitle(
      'BuilderBench Creative 8 Task 2  ·  compact sa3x256 · catwp · '
      r'dualgradreg $c=100$ $\lambda$lr$=10^{-6}$',
      fontsize=12, fontweight='bold', y=1.01,
  )
  os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
  fig.tight_layout()
  tmp = OUT_PATH + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, OUT_PATH)
  plt.close(fig)
  print(f'→ {OUT_PATH}')


if __name__ == '__main__':
  main()
