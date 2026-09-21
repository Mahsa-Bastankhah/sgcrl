#!/usr/bin/env python3
"""Paper-ready C7T2 success for CRL dual-dgr, historical TD3, TD-InfoNCE, NF.

CRL is dual-dgr on φ·ψ plus value-dgr (Warp 200M / 10-update, two seeds).
TD-InfoNCE is the Warp 200M / 10-update run (two seeds).
TD3 never succeeded on this task; the curve is the historical locked pick.
NF is compact-small + dual-dgr + valuedgr at the full 300M budget.

Averaging matches plot_bb_nf_rnd_success.py: train raw; eval faint raw +
centered rolling mean w=21; mean on the union of seed x (no min-end clip).
"""
from __future__ import annotations

import argparse
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
    PAPER_DIR, "paper plots", "bb_c7t2_density_estimators_train_eval")
FIGS_STEM = os.path.join(
    REPO, "figs", "builderbench", "final_runs", "other_density_estimators",
    "creative7_task2_crl_td3_tdinfonce_crl10_success")
EVAL_SMOOTH_WINDOW = nf_plot.EVAL_SMOOTH_WINDOW
MIN_FINAL_STEP = 190_000_000
EXPECTED_SEEDS = 2

RUNS = (
    {
        "label": "CRL · dual-dgr + valuedgr · 10 updates",
        "color": ps.C["blue"],
        "run": (
            "final_runs/"
            "ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_"
            "nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_"
            "200m_crl10_dualgradreg_c100_lamlr1e2_valuedgr_c100_"
            "lamlr1e6_warp_2h30"
        ),
        "min_final_step": MIN_FINAL_STEP,
    },
    {
        "label": "TD3 · historical (did not solve)",
        "color": ps.C["vermillion"],
        "run": (
            "ppo_builderbench_creative7_task2_e1024_pd_td3_logq_tau05_"
            "actorreset_nopermute_normobs_evalvid_catwp_tol0013"
        ),
        # Historical jax pick; seed 1 stopped early. Still a flat failure.
        "min_final_step": 80_000_000,
    },
    {
        "label": "TD-InfoNCE · 10 updates",
        "color": ps.C["green"],
        "run": (
            "final_runs/"
            "ppo_builderbench_creative7_task2_e1024_pd_tdinfonce_tau05_"
            "nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_"
            "200m_crl10_warp_2h30"
        ),
        "min_final_step": MIN_FINAL_STEP,
    },
    {
        "label": "NF · compact-small + valuedgr",
        "color": ps.C["orange"],
        "run": (
            "final_runs/"
            "ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small_"
            "sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_"
            "minstd1e5_ent05to001_ep70_300m_crl10_dualgradreg_c100_"
            "lamlr1e6_valuedgr_c100_lamlr1e6_warp_4h"
        ),
        "min_final_step": MIN_FINAL_STEP,
    },
)


def _load(spec: dict, *, allow_partial: bool) -> tuple:
  train_series = base._read_train_seed_series(LOG_ROOT, spec["run"])
  eval_series = base._read_eval_seed_series(LOG_ROOT, spec["run"])
  min_final = int(spec["min_final_step"])
  for split, series in (("train", train_series), ("eval", eval_series)):
    if not series:
      raise RuntimeError(f'{spec["run"]}: no {split} trajectories')
    finals = [max((x for x, _ in t), default=0) for t in series]
    print(
        f'validated {spec["label"]} {split}: n={len(series)}, finals={finals}')
    if not allow_partial and any(step < min_final for step in finals):
      raise RuntimeError(
          f'{spec["run"]}: incomplete {split} trajectory '
          f"(finals={finals}, need>={min_final}); pass --allow-partial")

  return train_series, eval_series


def _style_axis(ax, *, title: str, ylabel: str) -> None:
  ax.set_title(title)
  ax.set_xlabel("Environment steps")
  ax.set_ylabel(ylabel)
  ax.set_ylim(-0.05, 1.05)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ps.style_axes(ax, which="major")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--allow-partial", action="store_true",
      help="Plot before CRL / TD-InfoNCE reach 190M env steps.")
  args = parser.parse_args()

  fig, (ax_train, ax_eval) = ps.figure("sidebyside")
  max_step = 0
  is_partial = False

  for spec in RUNS:
    train_series, eval_series = _load(
        spec, allow_partial=args.allow_partial)
    tx, tm, _, train_n = base._aggregate_mean_stderr(train_series)
    ex, em, _, eval_n = base._aggregate_mean_stderr(eval_series)
    spec_max = max(tx[-1], ex[-1]) if tx and ex else 0
    max_step = max(max_step, spec_max)
    if spec_max < spec["min_final_step"]:
      is_partial = True

    label = f'{spec["label"]} (n={train_n})'
    nf_plot._draw_train(
        ax_train, train_series, color=spec["color"], label=label)
    nf_plot._draw_eval(
        ax_eval, eval_series, color=spec["color"], label=label,
        window=EVAL_SMOOTH_WINDOW)
    eval_smooth = base._rolling_mean(em, EVAL_SMOOTH_WINDOW)
    print(
        f'{spec["label"]}: train/eval runs={train_n}/{eval_n}; '
        f"train final={tm[-1]:.3f}; eval MA final={eval_smooth[-1]:.3f}; "
        f"train/eval max={tx[-1]}/{ex[-1]}")

  train_title = "Creative 7 Task 2 · train"
  eval_title = (
      rf"Creative 7 Task 2 · eval (roll. mean $w$={EVAL_SMOOTH_WINDOW})")
  if is_partial:
    train_title += " (partial)"
    eval_title += " (partial)"

  _style_axis(ax_train, title=train_title, ylabel="Success")
  _style_axis(ax_eval, title=eval_title, ylabel="")
  ax_train.set_xlim(0, max(max_step, 1))
  ax_eval.set_xlim(0, max(max_step, 1))

  handles = [
      Line2D(
          [0], [0], color=spec["color"], lw=ps.LW,
          label=spec["label"])
      for spec in RUNS
  ]
  ps.fig_legend(
      fig, handles=handles, labels=[h.get_label() for h in handles],
      ncol=4, loc="outside upper center")

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  os.makedirs(os.path.dirname(FIGS_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM)
  ps.savefig(fig, FIGS_STEM)
  plt.close(fig)


if __name__ == "__main__":
  main()
