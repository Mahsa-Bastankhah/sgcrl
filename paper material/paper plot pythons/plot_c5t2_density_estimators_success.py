#!/usr/bin/env python3
"""Paper-ready C5T2 success for the three retained estimator recipes plus NF.

Each CRL/TD recipe is a single update setting with three seeds from the run
dirs. NF is the compact-small valuedgr recipe (n=3; all seeds stop short of
200M and stay in the mean).

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
    PAPER_DIR, "paper plots", "bb_c5t2_density_estimators_train_eval")
EVAL_SMOOTH_WINDOW = nf_plot.EVAL_SMOOTH_WINDOW
MIN_FINAL_STEP = 190_000_000
EXPECTED_SEEDS = 3

RUNS = (
    {
        "label": "CRL · 10 updates (n=3)",
        "color": ps.C["blue"],
        "run": (
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task2_e1024_pd_crl_tau05_"
            "actorreset_evalvid_catselect_extrew1_ep70_200m_crl10_"
            "eval10_warp_2h30"
        ),
    },
    {
        "label": "TD3 · ext. reward + NormObs · 25 updates (n=3)",
        "color": ps.C["vermillion"],
        "run": (
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_"
            "actorreset_nopermute_normobs_evalvid_catselect_extrew1_"
            "ep70_200m_crl25_eval10_warp_4h"
        ),
    },
    {
        "label": "TD-InfoNCE · 25 updates (n=3)",
        "color": ps.C["green"],
        "run": (
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_"
            "catselect_extrew1_ep70_200m_crl25_eval10_warp_4h"
        ),
    },
    {
        "label": "NF · compact-small + valuedgr (n=3)",
        "color": ps.C["orange"],
        "run": (
            "final_runs/"
            "ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_"
            "sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_"
            "minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_"
            "valuedgr_c100_lamlr1e6_warp_2h30"
        ),
        "min_final_step": 0,
    },
)


def _load(spec: dict) -> tuple:
  train_series = base._read_train_seed_series(LOG_ROOT, spec["run"])
  eval_series = base._read_eval_seed_series(LOG_ROOT, spec["run"])
  min_final = int(spec.get("min_final_step", MIN_FINAL_STEP))
  for split, series in (("train", train_series), ("eval", eval_series)):
    if len(series) != EXPECTED_SEEDS:
      raise RuntimeError(
          f'{spec["run"]}: expected {EXPECTED_SEEDS} {split} trajectories, '
          f"found {len(series)}")
    finals = [max((x for x, _ in t), default=0) for t in series]
    print(
        f'validated {spec["label"]} {split}: n={len(series)}, finals={finals}')
    if min_final and any(step < min_final for step in finals):
      raise RuntimeError(
          f'{spec["run"]}: incomplete {split} trajectory '
          f"(finals={finals}, need>={min_final})")

  return train_series, eval_series


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
    train_series, eval_series = _load(spec)
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
        f"train final={tm[-1]:.3f}; eval MA final={eval_smooth[-1]:.3f}; "
        f"train/eval max={tx[-1]}/{ex[-1]}")

  _style_axis(ax_train, title="Pyramid · train", ylabel="Success")
  _style_axis(
      ax_eval,
      title=rf"Pyramid · eval (roll. mean $w$={EVAL_SMOOTH_WINDOW})",
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
