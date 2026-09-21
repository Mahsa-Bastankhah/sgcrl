#!/usr/bin/env python3
"""Paper-ready selected-recipe C4T1 density-estimator comparison.

CRL-25 and TD3-25 use three seeds from the run directories. TD-InfoNCE stays
the pooled 10/25-update pair (n=4 trajectories). NF is the compact-small
valuedgr recipe (n=3; seeds 0/1 stop short of 200M and stay in the mean).

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
SLURM_ROOT = os.path.join(
    REPO, "slurm", "final_runs", "other_density_estimators")
OUT_STEM = os.path.join(
    PAPER_DIR, "paper plots", "bb_c4t1_density_estimators_train_eval")
EVAL_SMOOTH_WINDOW = nf_plot.EVAL_SMOOTH_WINDOW
MIN_FINAL_STEP = 190_000_000

RUN_CRL25 = (
    "final_runs/other_density_estimators/"
    "ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_actorreset_"
    "nopermute_norand_normobs_catselect_minstd1e4_extrew1_ep50_200m_"
    "crl25_eval10_warp_4h"
)
RUN_TD325 = (
    "final_runs/other_density_estimators/"
    "ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_actorreset_"
    "nopermute_catselect_minstd1e4_fixedx01_extrew1_ep50_200m_crl25_"
    "eval10_warp_4h"
)
RUN_TDINFO10 = (
    "final_runs/other_density_estimators/"
    "ppo_builderbench_creative4_task1_e1024_pd_tdinfonce_tau05_actorreset_"
    "nopermute_catselect_minstd1e4_extrew1_ep50_200m_crl10_eval10_"
    "warp_2h30"
)
RUN_TDINFO25 = (
    "final_runs/other_density_estimators/"
    "ppo_builderbench_creative4_task1_e1024_pd_tdinfonce_tau05_actorreset_"
    "nopermute_catselect_minstd1e4_extrew1_ep50_200m_crl25_eval10_warp_4h"
)
RUN_NF = (
    "final_runs/"
    "ppo_builderbench_creative4_task1_e1024_pd_nf_compact_small_"
    "sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_"
    "minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_"
    "valuedgr_c100_lamlr1e6_warp_2h30"
)

RUNS = (
    {
        "label": "CRL-25 (n=3 seeds)",
        "color": ps.C["blue"],
        "run": RUN_CRL25,
        "expected_n": 3,
    },
    {
        "label": "TD3-25 (n=3 seeds)",
        "color": ps.C["vermillion"],
        "run": RUN_TD325,
        "expected_n": 3,
    },
    {
        "label": "TD-InfoNCE (pooled 10/25 updates; n=4)",
        "color": ps.C["green"],
        "trajectories": (
            ("bb_ode_crl10_3811416_10.log", RUN_TDINFO10, 0, 10),
            ("bb_ode_crl10_3811416_11.log", RUN_TDINFO10, 1, 10),
            ("bb_ode_crl25_3811417_10.log", RUN_TDINFO25, 0, 25),
            ("bb_ode_crl25_3811417_11.log", RUN_TDINFO25, 1, 25),
        ),
    },
    {
        "label": "NF (n=3 seeds)",
        "color": ps.C["orange"],
        "run": RUN_NF,
        "expected_n": 3,
        "min_final_step": 0,
    },
)


def _load_run_dir(
    run: str, expected_n: int, *, min_final_step: int = MIN_FINAL_STEP,
) -> tuple:
  train_series = base._read_train_seed_series(LOG_ROOT, run)
  eval_series = base._read_eval_seed_series(LOG_ROOT, run)
  for split, series in (("train", train_series), ("eval", eval_series)):
    if len(series) != expected_n:
      raise RuntimeError(
          f"{run}: expected {expected_n} {split} trajectories, "
          f"found {len(series)}")
    finals = [max((x for x, _ in trajectory), default=0)
              for trajectory in series]
    print(f"validated {run} {split}: n={len(series)}, finals={finals}")
    if min_final_step and any(step < min_final_step for step in finals):
      raise RuntimeError(
          f"{run}: incomplete {split} trajectory "
          f"(finals={finals}, need>={min_final_step})")
  print(f"validated {run}: train/eval n={len(train_series)}/{len(eval_series)}")
  return train_series, eval_series


def _load(spec: dict) -> tuple:
  if "run" in spec:
    return _load_run_dir(
        spec["run"], spec["expected_n"],
        min_final_step=int(spec.get("min_final_step", MIN_FINAL_STEP)))

  train_series = []
  eval_series = []
  for filename, run, seed, updates in spec["trajectories"]:
    path = os.path.join(SLURM_ROOT, filename)
    if not os.path.isfile(path):
      raise FileNotFoundError(f"missing retained Slurm log: {path}")
    with open(path, "r", errors="replace") as f:
      header = "".join(f.readline() for _ in range(8))
    expected = (
        "task=creative4_task1",
        f"seed={seed}",
        f"crl_steps={updates}",
        f"log_dir=logs/{run}/",
    )
    missing = [token for token in expected if token not in header]
    if missing:
      raise RuntimeError(f"{path}: header missing expected tokens {missing}")

    train = base._parse_train_slurm_path(path)
    eval_iters = base._parse_eval_slurm_path(path)
    eval_curve = base._iters_to_env_steps(eval_iters, LOG_ROOT, run)
    for split, curve in (("train", train), ("eval", eval_curve)):
      final_step = curve[-1][0] if curve else 0
      if final_step < MIN_FINAL_STEP:
        raise RuntimeError(
            f"{path}: incomplete {split} trajectory (final step {final_step})")
    train_series.append(train)
    eval_series.append(eval_curve)
    print(
        f"validated {filename}: seed={seed}, updates={updates}, "
        f"train/eval final steps={train[-1][0]}/{eval_curve[-1][0]}")

  expected_n = len(spec["trajectories"])
  if len(train_series) != expected_n or len(eval_series) != expected_n:
    raise RuntimeError(
        f'{spec["label"]}: expected {expected_n} train/eval trajectories, '
        f"found {len(train_series)}/{len(eval_series)}")
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
        f'{spec["label"]}: train/eval trajectories={train_n}/{eval_n}; '
        f"train final={tm[-1]:.3f}; eval MA final={eval_smooth[-1]:.3f}; "
        f"train/eval max={tx[-1]}/{ex[-1]}")

  _style_axis(
      ax_train, title="Creative 4 Task 1 · train", ylabel="Success")
  _style_axis(
      ax_eval,
      title=(
          rf"Creative 4 Task 1 · eval "
          rf"(roll. mean $w$={EVAL_SMOOTH_WINDOW})"),
      ylabel="")

  handles = [
      Line2D([0], [0], color=spec["color"], lw=ps.LW, label=spec["label"])
      for spec in RUNS
  ]
  ps.fig_legend(
      fig, handles=handles, labels=[h.get_label() for h in handles],
      ncol=4, loc="outside upper center",
      title=(
          "Selected recipes · mean ± SEM; pooled TD-InfoNCE uncertainty "
          "includes seed + update-setting variation"))

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM)
  plt.close(fig)


if __name__ == "__main__":
  main()
