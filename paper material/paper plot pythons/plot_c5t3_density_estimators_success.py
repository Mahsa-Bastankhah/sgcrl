#!/usr/bin/env python3
"""Paper-ready C5T3 success comparison for CRL-10, TD3-10, TD-InfoNCE-10,
and the older compact NF (sa3x256, nopermute; n=3).

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
    PAPER_DIR, "paper plots", "bb_c5t3_density_estimators_train_eval")
EVAL_SMOOTH_WINDOW = nf_plot.EVAL_SMOOTH_WINDOW
MIN_FINAL_STEP = 190_000_000

RUNS = (
    {
        "label": "CRL-10",
        "color": ps.C["blue"],
        "run": (
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task3_e1024_pd_crl_tau05_"
            "actorreset_nopermute_norand_normobs_evalvid_catselect_extrew1_"
            "ep70_200m_crl10_eval10_warp_2h30"
        ),
        "expected_n": 4,
    },
    {
        "label": "TD3-10",
        "color": ps.C["vermillion"],
        "run": (
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task3_e1024_pd_td3_logq_tau05_"
            "actorreset_permute_rand_evalvid_catselect_extrew1_ep70_200m_"
            "crl10_eval10_warp_2h30"
        ),
        "expected_n": 4,
    },
    {
        "label": "TD-InfoNCE-10",
        "color": ps.C["green"],
        "run": (
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task3_e1024_pd_tdinfonce_tau05_"
            "catselect_extrew1_ep70_200m_crl10_eval10_warp_2h30"
        ),
        "expected_n": 4,
    },
    {
        "label": "NF (compact)",
        "color": ps.C["orange"],
        "run": (
            "ppo_builderbench_creative5_task3_e1024_pd_nf_compact_"
            "sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_"
            "minstd1e5_entanneal_evalvid_catselect_extrew1"
        ),
        "expected_n": 3,
    },
)


def _load(spec: dict, *, eval_split: bool):
  run = spec["run"]
  expected_n = int(spec["expected_n"])
  split_name = "eval" if eval_split else "train"
  if eval_split:
    series = base._read_eval_seed_series(LOG_ROOT, run)
  else:
    series = base._read_train_seed_series(LOG_ROOT, run)
  if len(series) != expected_n:
    raise RuntimeError(
        f"{run}: expected {expected_n} {split_name} trajectories, "
        f"found {len(series)}")
  final_steps = [max((x for x, _ in trajectory), default=0)
                 for trajectory in series]
  if any(step < MIN_FINAL_STEP for step in final_steps):
    raise RuntimeError(
        f"{run}: incomplete {split_name} trajectories "
        f"(final steps {final_steps})")
  return series, final_steps


def _style_axis(ax, *, title: str, ylabel: str) -> None:
  ax.set_title(title)
  ax.set_xlabel("Environment steps")
  ax.set_ylabel(ylabel)
  ax.set_xlim(0, 200_000_000)
  ax.set_ylim(-0.05, 1.05)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ps.style_axes(ax, which="major")


def main() -> None:
  fig, (ax_train, ax_eval) = ps.figure("sidebyside")

  for spec in RUNS:
    train_series, train_finals = _load(spec, eval_split=False)
    eval_series, eval_finals = _load(spec, eval_split=True)
    train_n = len(train_series)
    eval_n = len(eval_series)
    if train_n != eval_n:
      raise RuntimeError(
          f'{spec["label"]}: train/eval seed mismatch '
          f"({train_n}/{eval_n})")
    curve_label = f'{spec["label"]} (n={train_n})'
    nf_plot._draw_train(
        ax_train, train_series, color=spec["color"], label=curve_label)
    nf_plot._draw_eval(
        ax_eval, eval_series, color=spec["color"], label=curve_label,
        window=EVAL_SMOOTH_WINDOW)
    tx, tm, _, _ = base._aggregate_mean_stderr(train_series)
    ex, em, _, _ = base._aggregate_mean_stderr(eval_series)
    eval_smooth = base._rolling_mean(em, EVAL_SMOOTH_WINDOW)
    print(
        f'{spec["label"]}: train/eval runs={train_n}/{eval_n}; '
        f"train final={tm[-1]:.3f}, peak={max(tm):.3f}; "
        f"eval MA final={eval_smooth[-1]:.3f}, peak={max(eval_smooth):.3f}; "
        f"train finals={train_finals}; eval finals={eval_finals}")

  _style_axis(
      ax_train, title="Underspecified towers · train", ylabel="Success")
  _style_axis(
      ax_eval,
      title=(
          rf"Underspecified towers · eval "
          rf"(roll. mean $w$={EVAL_SMOOTH_WINDOW})"),
      ylabel="")

  handles = [
      Line2D(
          [0], [0], color=spec["color"], lw=ps.LW,
          label=f'{spec["label"]} (n={spec["expected_n"]})')
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
