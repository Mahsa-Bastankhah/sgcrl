#!/usr/bin/env python3
"""Plot creative4_task2 CRL / NF catselect train_success_1000 variants.

Overlays requested ``*_catselect_crl10`` jobs against the other same-family
cube-4 task-2 runs (mean ± stderr over seeds).

Writes:
  figs/builderbench/active_train_eval/creative4_task2_by_method/
    creative4_task2_train_success1000_crl.png
    creative4_task2_train_success1000_nf.png

Usage:
  python scripts/plot_c4t2_catselect_crl_variants.py
  python scripts/plot_c4t2_catselect_crl_variants.py --family nf
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import plot_builderbench_train_success1000 as base  # noqa: E402

FIGS_DIR = os.path.join(base.FIGS_DIR, 'creative4_task2_by_method')
GROUP = 'creative4_task2'

# (log_dir_name, legend_label, emphasize)
CRL_RUNS: list[tuple[str, str, bool]] = [
    ('ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_catselect',
     'catselect (no extrew)', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_extrew1_catselect',
     'extrew1_catselect', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_catselect_crl10',
     'catselect_crl10 (steps/iter=10)', True),
    ('ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_catselect_sfpert',
     'catselect_sfpert', False),
]

NF_RUNS: list[tuple[str, str, bool]] = [
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_catselect',
     'nf_catselect', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_catselect_extrew1',
     'nf_catselect_extrew1', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_actorreset_evalvid_catselect',
     'nf_actor_evalvid_cat', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_actorreset_evalvid_catselect_extrew1',
     'nf_actor_evalvid_cat_ext1', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_tau05_actorreset_nopermute_fixedx01_evalvid_catselect_extrew1',
     'nf_noperm_fixx_ext1', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_fixedx01_catselect_extrew1',
     'nf_compact_noperm_fixx_ext1', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_nopermute_fixedx01_catselect_succ3',
     'nf_compact_noperm_fixx_succ3', False),
    ('ppo_builderbench_creative4_task2_e1024_pd_nf_compact_sa3x256_r64_b6_w256_tau05_actorreset_nopermute_fixedx01_catselect_crl10',
     'nf_compact_noperm_fixx_crl10', True),
]

FAMILIES = {
    'crl': ('CRL catselect', CRL_RUNS),
    'nf': ('NF', NF_RUNS),
}


def plot_family(family: str, figs_dir: str = FIGS_DIR) -> str | None:
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker

  title_tag, runs = FAMILIES[family]
  os.makedirs(figs_dir, exist_ok=True)
  fig, ax = plt.subplots(figsize=(11.0, 5.0))
  plotted = 0

  for i, (log_dir_name, label, emphasize) in enumerate(runs):
    seed_series = base._read_train_seed_series(
        base.LOG_ROOT, log_dir_name, slurm_dir=base.SLURM_DIR)
    xs, mean, se, n_seeds = base._aggregate_mean_stderr(seed_series)
    if not xs:
      print(f'  [{family}] {label}: no train data, skip')
      continue
    xs, mean, se = base._subsample_curve(xs, mean, se)
    color = base.ACCENT_COLORS[i % len(base.ACCENT_COLORS)]
    lw = 2.8 if emphasize else 2.0
    ax.plot(
        xs, mean, color=color, linewidth=lw,
        label=f'{label} (n={n_seeds})', alpha=0.95, zorder=3 + i)
    if n_seeds > 1 and any(s > 0 for s in se):
      lo = [m - s for m, s in zip(mean, se)]
      hi = [m + s for m, s in zip(mean, se)]
      ax.fill_between(
          xs, lo, hi, color=color, alpha=0.22, linewidth=0, zorder=2 + i)
    print(f'  [{family}] {label}: n_seeds={n_seeds}, {len(xs)} pts  '
          f'x=[{xs[0]}..{xs[-1]}] y=[{min(mean):.3f}..{max(mean):.3f}]')
    plotted += 1

  if plotted == 0:
    plt.close(fig)
    print(f'[{family}] no series to plot')
    return None

  ax.set_title(
      f'BuilderBench Creative 4 Task 2 [{title_tag}] — '
      'train_success_1000 (mean ± stderr)',
      fontsize=12, fontweight='bold')
  ax.set_xlabel('Env Steps', fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylabel('Train Success (last 1000)', fontsize=11)
  ax.set_ylim(-0.05, 1.05)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(
      loc='upper left', bbox_to_anchor=(1.01, 1.0),
      fontsize=10, framealpha=1.0, ncol=1,
      edgecolor='#333333', fancybox=False)

  out_path = os.path.join(
      figs_dir, f'{GROUP}_train_success1000_{family}.png')
  fig.tight_layout()
  fig.savefig(out_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {out_path}')
  return out_path


def main():
  p = argparse.ArgumentParser()
  p.add_argument(
      '--family', choices=('crl', 'nf', 'all'), default='all',
      help='Which family plot(s) to write.')
  p.add_argument('--figs_dir', default=FIGS_DIR)
  args = p.parse_args()
  families = ('crl', 'nf') if args.family == 'all' else (args.family,)
  for fam in families:
    print(f'=== {fam} ===')
    plot_family(fam, figs_dir=args.figs_dir)


if __name__ == '__main__':
  main()
