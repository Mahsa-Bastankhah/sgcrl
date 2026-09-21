#!/usr/bin/env python3
"""Combine matched Allegro NF rank-accuracy CSVs into one CSV and PNG."""
from __future__ import annotations

import argparse
import csv
import os


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", action="append", required=True,
                      help="LABEL=CSV_PATH (repeat once per variant)")
  parser.add_argument("--output-csv", required=True)
  parser.add_argument("--output-plot", required=True)
  args = parser.parse_args()

  rows = []
  for spec in args.input:
    if "=" not in spec:
      parser.error(f"--input must be LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    with open(path, newline="", encoding="utf-8") as fh:
      for row in csv.DictReader(fh):
        rows.append({"variant": label, **row})
  if not rows:
    raise RuntimeError("input CSVs contained no rows")

  os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
  with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(7.2, 4.5))
  labels = list(dict.fromkeys(row["variant"] for row in rows))
  for label in labels:
    selected = [row for row in rows if row["variant"] == label]
    x = [int(row["iteration"]) for row in selected]
    y = [float(row["rank_accuracy"]) for row in selected]
    se = [float(row["rank_accuracy_se"]) for row in selected]
    ax.errorbar(x, y, yerr=se, marker="o", linewidth=2, capsize=3,
                label=label)
  ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
  ax.set(xlabel="checkpoint iteration", ylabel="binary NF rank accuracy",
         ylim=(0.0, 1.0))
  ax.grid(axis="y", linestyle="--", alpha=0.3)
  ax.spines[["top", "right"]].set_visible(False)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(args.output_plot, dpi=180, bbox_inches="tight")
  plt.close(fig)
  print(f"wrote {args.output_csv}")
  print(f"wrote {args.output_plot}")


if __name__ == "__main__":
  main()
