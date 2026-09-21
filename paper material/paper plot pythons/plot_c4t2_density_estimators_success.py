#!/usr/bin/env python3
"""Paper-ready C4T2 success, pooling 10/25-update variants per estimator.

NF is the compact-small valuedgr recipe (n=3; seeds 0/1 stop short of 200M
and stay in the mean).

Averaging matches plot_bb_nf_rnd_success.py: train raw; eval faint raw +
centered rolling mean w=21; mean on the union of seed x (no min-end clip).
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import paper_style as ps  # noqa: E402
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_bb_nf_rnd_success as nf_plot  # noqa: E402

ps.apply()

LOG_ROOT = os.path.join(REPO, "logs")
OUT_STEM = os.path.join(
    PAPER_DIR, "paper plots", "bb_c4t2_density_estimators_train_eval")
EVAL_SMOOTH_WINDOW = nf_plot.EVAL_SMOOTH_WINDOW
MIN_FINAL_STEP = 190_000_000

RUNS = (
    {
        "label": "CRL · pooled 10/25 updates (n=4 runs)",
        "color": ps.C["blue"],
        "runs": (
            (
                "final_runs/other_density_estimators/"
                "ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_"
                "catselect_stateonly_extrew1_ep50_200m_crl10_eval10_warp_2h30"
            ),
            (
                "final_runs/other_density_estimators/"
                "ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_"
                "catselect_stateonly_extrew1_ep50_200m_crl25_eval10_warp_4h"
            ),
        ),
    },
    {
        "label": "TD3 · pooled 10/25 updates (n=4 runs)",
        "color": ps.C["vermillion"],
        "runs": (
            (
                "final_runs/other_density_estimators/"
                "ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_"
                "catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30"
            ),
            (
                "final_runs/other_density_estimators/"
                "ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_"
                "catselect_extrew1_ep50_200m_crl25_eval10_warp_4h"
            ),
        ),
    },
    {
        "label": "TD-InfoNCE · pooled 10/25 updates (n=4 runs)",
        "color": ps.C["green"],
        "runs": (
            (
                "final_runs/other_density_estimators/"
                "ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_"
                "catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30"
            ),
            (
                "final_runs/other_density_estimators/"
                "ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_"
                "catselect_extrew1_ep50_200m_crl25_eval10_warp_4h"
            ),
        ),
    },
    {
        "label": "NF · valuedgr (n=3)",
        "color": ps.C["orange"],
        "runs": (
            (
                "final_runs/"
                "ppo_builderbench_creative4_task2_e1024_pd_nf_compact_small_"
                "sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_"
                "minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_"
                "lamlr1e6_valuedgr_c100_lamlr1e6_warp_2h30"
            ),
        ),
        "expected_per_run": 3,
        "expected_n": 3,
        "min_final_step": 0,
    },
)


def _load(spec: dict, *, eval_split: bool):
  series = []
  split_name = "eval" if eval_split else "train"
  expected_per_run = spec.get("expected_per_run", 2)
  expected_n = spec.get("expected_n", 4)
  min_final = int(spec.get("min_final_step", MIN_FINAL_STEP))
  for run in spec["runs"]:
    if eval_split:
      run_series = base._read_eval_seed_series(LOG_ROOT, run)
    else:
      run_series = base._read_train_seed_series(LOG_ROOT, run)
    if len(run_series) != expected_per_run:
      raise RuntimeError(
          f"{run}: expected {expected_per_run} {split_name} trajectories, "
          f"found {len(run_series)}")
    finals = [max((x for x, _ in trajectory), default=0)
              for trajectory in run_series]
    print(f"validated {run} {split_name}: n={len(run_series)}, finals={finals}")
    if min_final and any(step < min_final for step in finals):
      raise RuntimeError(
          f"{run}: incomplete {split_name} trajectory "
          f"(finals={finals}, need>={min_final})")
    series.extend(run_series)

  if len(series) != expected_n:
    raise RuntimeError(
        f"{spec['runs']}: expected {expected_n} pooled {split_name} "
        f"trajectories, found {len(series)}")
  return series


def _style_axis(ax, *, title: str, ylabel: str) -> None:
  ax.set_title(title)
  ax.set_xlabel("Environment steps")
  ax.set_ylabel(ylabel)
  ax.set_ylim(-0.05, 1.05)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ps.style_axes(ax, which="major")


def main() -> None:
  fig, (ax_train, ax_eval) = ps.figure("sidebyside")

  for spec in RUNS:
    train_series = _load(spec, eval_split=False)
    eval_series = _load(spec, eval_split=True)
    train_n = nf_plot._draw_train(
        ax_train, train_series, color=spec["color"], label=spec["label"])
    eval_n = nf_plot._draw_eval(
        ax_eval, eval_series, color=spec["color"], label=spec["label"],
        window=EVAL_SMOOTH_WINDOW)
    tx, tm, _, _ = base._aggregate_mean_stderr(train_series)
    ex, em, _, _ = base._aggregate_mean_stderr(eval_series)
    eval_smooth = base._rolling_mean(em, EVAL_SMOOTH_WINDOW)
    print(
        f'{spec["label"]}: train/eval runs={train_n}/{eval_n}; '
        f'train final={tm[-1]:.3f}; '
        f'eval MA final={eval_smooth[-1]:.3f}; '
        f'train/eval max={tx[-1]}/{ex[-1]}')

  _style_axis(ax_train, title="Two towers · train", ylabel="Success")
  _style_axis(
      ax_eval,
      title=rf"Two towers · eval (roll. mean $w$={EVAL_SMOOTH_WINDOW})",
      ylabel="")

  handles = [
      Line2D([0], [0], color=spec["color"], lw=ps.LW, label=spec["label"])
      for spec in RUNS
  ]
  ps.fig_legend(
      fig, handles=handles, labels=[h.get_label() for h in handles],
      ncol=4, loc="outside upper center")

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM)
  plt.close(fig)


if __name__ == "__main__":
  main()
