#!/usr/bin/env python3
"""Time-align and average three floor-grasp safe-reset evaluations."""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts import plot_builderbench_train_success1000 as plot_base  # noqa: E402

SMOOTH_WINDOW = 5
COLORS = ("#1F78B4", "#E31A1C", "#33A02C")


def _read(path: str) -> dict[int, float]:
  with open(path, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
  if not rows:
    raise ValueError(f"empty evaluation CSV: {path}")
  return {
      int(row["global_step"]): float(row["success_rate"])
      for row in rows
  }


def _write(path: str, rows: list[dict]) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed_csv", action="append", required=True)
  parser.add_argument("--csv_output", required=True)
  parser.add_argument("--plot_output", required=True)
  args = parser.parse_args()
  if len(args.seed_csv) != 3:
    raise ValueError(f"expected exactly 3 --seed_csv inputs, got {len(args.seed_csv)}")

  curves = [_read(path) for path in args.seed_csv]
  common_steps = sorted(set.intersection(*(set(curve) for curve in curves)))
  if not common_steps:
    raise ValueError("seed evaluations have no exact global_step values in common")

  values = np.asarray([
      [curve[step] for step in common_steps] for curve in curves],
      dtype=np.float64)
  means = values.mean(axis=0)
  sems = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
  rows = []
  for index, step in enumerate(common_steps):
    rows.append({
        "global_step": step,
        "seed0_success_rate": values[0, index],
        "seed1_success_rate": values[1, index],
        "seed2_success_rate": values[2, index],
        "mean_success_rate": means[index],
        "seed_sem": sems[index],
        "n_seeds": 3,
        "episodes_per_seed": 10,
    })
  _write(args.csv_output, rows)

  fig, ax = plt.subplots(figsize=(9.0, 4.8))
  for seed, (curve, color) in enumerate(zip(curves, COLORS)):
    xs = [step for step in common_steps if step in curve]
    ys = [curve[step] for step in xs]
    ax.plot(xs, ys, color=color, lw=0.9, alpha=0.22,
            label=f"Seed {seed} raw")

  smoothed_mean = np.asarray(
      plot_base._plot_eval_smoothed(
          ax, common_steps, means.tolist(), color="#6A3D9A",
          label=f"3-seed mean (rolling window={SMOOTH_WINDOW})",
          window=SMOOTH_WINDOW, linewidth=2.7),
      dtype=np.float64)
  smoothed_sem = np.asarray(
      plot_base._rolling_mean(sems.tolist(), SMOOTH_WINDOW),
      dtype=np.float64)
  ax.fill_between(
      common_steps,
      np.clip(smoothed_mean - smoothed_sem, 0.0, 1.0),
      np.clip(smoothed_mean + smoothed_sem, 0.0, 1.0),
      color="#6A3D9A", alpha=0.16, linewidth=0, label="Seed SEM")
  ax.set_xlim(common_steps[0], common_steps[-1])
  ax.set_ylim(-0.05, 1.05)
  ax.set_xlabel("Environment steps")
  ax.set_ylabel(
      f"Eval success over 10 episodes/seed (rolling mean, window={SMOOTH_WINDOW})")
  ax.set_title(
      "Sawyer bin floor-grasp · healthy-reset deterministic evaluation\n"
      "Three-seed mean on exact shared training steps")
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(plot_base._fmt_steps))
  ax.grid(axis="y", linestyle=":", alpha=0.4)
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(frameon=False, ncol=2)
  fig.tight_layout()
  fig.savefig(args.plot_output, dpi=180, bbox_inches="tight")
  plt.close(fig)
  print(
      f"[overall] shared_checkpoints={len(common_steps)} "
      f"range={common_steps[0]}..{common_steps[-1]} n_seeds=3")
  print(f"→ {args.csv_output}")
  print(f"→ {args.plot_output}")


if __name__ == "__main__":
  main()
