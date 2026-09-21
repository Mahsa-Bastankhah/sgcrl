#!/usr/bin/env python3
"""Overlay offline NF vs production CRL learnability curves (CPU, no JAX)."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

from scripts.offline_nf.core import (  # noqa: E402
    _annotate_early_steps,
    _load_pyplot,
    read_csv,
    write_csv,
    write_json,
)


def _rows_for_size(rows: list[dict[str, Any]], label: int) -> list[dict[str, Any]]:
  selected = [row for row in rows if int(row["dataset_size_label"]) == label]
  selected.sort(key=lambda row: float(row["step"]))
  return selected


def _loss_series(rows: list[dict[str, Any]], method: str) -> list[float]:
  key = "train_nf_loss" if method == "nf" else "train_crl_loss"
  if rows and key not in rows[0] and method == "nf":
    key = "train_nll"
  return [float(row[key]) for row in rows]


def write_overlay_plots(
    *,
    nf_dir: str,
    crl_dir: str,
    output_dir: str,
    domain_label: str,
    batch_size: int = 256,
) -> list[str]:
  """Read immutable NF metrics plus CRL metrics and write overlay figures."""
  nf_all = read_csv(os.path.join(nf_dir, "all_metrics.csv"))
  crl_all = read_csv(os.path.join(crl_dir, "all_metrics.csv"))
  labels = sorted({int(row["dataset_size_label"]) for row in crl_all})
  os.makedirs(output_dir, exist_ok=True)
  plt = _load_pyplot()
  paths: list[str] = []

  def draw(path: str, nf_rows, crl_rows, title: str, early_max: int | None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    nf_x = np.asarray([row["step"] for row in nf_rows], dtype=float)
    crl_x = np.asarray([row["step"] for row in crl_rows], dtype=float)
    axes[0, 0].plot(
        nf_x, [row["binary_accuracy"] for row in nf_rows],
        marker="o", color="#4C9BE8", label="NF")
    axes[0, 0].plot(
        crl_x, [row["binary_accuracy"] for row in crl_rows],
        marker="s", color="#E07A3D", label="CRL")
    axes[0, 0].axhline(0.5, color="0.5", ls="--", label="chance 50%")
    axes[0, 0].set_ylabel("held-out binary accuracy")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(fontsize=8)
    chance = float(crl_rows[0]["categorical_chance"])
    axes[0, 1].plot(
        nf_x, [row["categorical_accuracy"] for row in nf_rows],
        marker="o", color="#4C9BE8", label="NF")
    axes[0, 1].plot(
        crl_x, [row["categorical_accuracy"] for row in crl_rows],
        marker="s", color="#E07A3D", label="CRL")
    axes[0, 1].axhline(
        chance, color="0.5", ls="--", label=f"chance {100 * chance:g}%")
    axes[0, 1].set_ylabel("held-out K=32 accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(
        nf_x, _loss_series(nf_rows, "nf"),
        marker="o", color="#4C9BE8", label="NF train NLL/loss")
    axes[1, 0].set_ylabel("NF loss")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(
        crl_x, _loss_series(crl_rows, "crl"),
        marker="s", color="#E07A3D", label="CRL InfoNCE")
    axes[1, 1].set_ylabel("CRL loss")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
      axis.grid(axis="y", alpha=0.25)
      axis.set_xlabel("gradient updates")
    if early_max is not None:
      _annotate_early_steps(axes.flat, batch_size)
      axes[1, 0].set_xlim(0, early_max)
    fig.suptitle(title)
    fig.text(
        0.5, 0.005,
        "Same held-out fixture (seed+91003). NF scores log p(g|s,a); "
        "CRL scores φ(s,a)·ψ(g). Losses are not on a shared scale. "
        f"Step 0 = pre-train; first trained eval = 250 updates "
        f"({250 * batch_size:,} draws).",
        ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    os.replace(tmp, path)
    plt.close(fig)
    paths.append(path)

  comparison_rows: list[dict[str, Any]] = []
  for label in labels:
    nf_rows = _rows_for_size(nf_all, label)
    crl_rows = _rows_for_size(crl_all, label)
    if not nf_rows or not crl_rows:
      raise ValueError(f"missing metrics for dataset size {label}")
    size_dir = os.path.join(output_dir, f"size_{label}")
    os.makedirs(size_dir, exist_ok=True)
    draw(
        os.path.join(size_dir, "learning_curves_nf_vs_crl.png"),
        nf_rows, crl_rows,
        f"{domain_label} offline learnability — {label:,}  NF vs CRL",
        None)
    draw(
        os.path.join(size_dir, "learning_curves_early_0_500_nf_vs_crl.png"),
        [row for row in nf_rows if float(row["step"]) <= 500],
        [row for row in crl_rows if float(row["step"]) <= 500],
        f"{domain_label} offline learnability — {label:,}  NF vs CRL — early 0–500",
        500)
    comparison_rows.append({
        "dataset_size_label": label,
        "nf_binary_accuracy": nf_rows[-1]["binary_accuracy"],
        "crl_binary_accuracy": crl_rows[-1]["binary_accuracy"],
        "nf_categorical_accuracy": nf_rows[-1]["categorical_accuracy"],
        "crl_categorical_accuracy": crl_rows[-1]["categorical_accuracy"],
        "nf_train_loss": _loss_series(nf_rows, "nf")[-1],
        "crl_train_loss": _loss_series(crl_rows, "crl")[-1],
        "categorical_chance": crl_rows[0]["categorical_chance"],
        "train_transitions": crl_rows[-1]["train_transitions"],
    })

  write_csv(os.path.join(output_dir, "final_size_comparison.csv"), comparison_rows)
  x = np.asarray([row["train_transitions"] for row in comparison_rows], dtype=float)
  fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
  axes[0].plot(
      x, [row["nf_binary_accuracy"] for row in comparison_rows],
      marker="o", color="#4C9BE8", label="NF")
  axes[0].plot(
      x, [row["crl_binary_accuracy"] for row in comparison_rows],
      marker="s", color="#E07A3D", label="CRL")
  axes[0].axhline(0.5, color="0.5", ls="--")
  axes[0].set_ylabel("final binary accuracy")
  axes[0].set_ylim(0, 1)
  axes[0].legend(fontsize=8)
  axes[1].plot(
      x, [row["nf_categorical_accuracy"] for row in comparison_rows],
      marker="o", color="#4C9BE8", label="NF")
  axes[1].plot(
      x, [row["crl_categorical_accuracy"] for row in comparison_rows],
      marker="s", color="#E07A3D", label="CRL")
  axes[1].axhline(
      float(comparison_rows[0]["categorical_chance"]), color="0.5", ls="--")
  axes[1].set_ylabel("final K=32 accuracy")
  axes[1].set_ylim(0, 1)
  axes[1].legend(fontsize=8)
  for axis in axes:
    axis.set_xscale("log")
    axis.set_xlabel("actual train transitions")
    axis.grid(axis="y", alpha=0.25)
  fig.suptitle(f"{domain_label} offline learnability: final NF vs CRL")
  scaling = os.path.join(output_dir, "final_size_comparison_nf_vs_crl.png")
  tmp = scaling + ".tmp.png"
  fig.tight_layout()
  fig.savefig(tmp, dpi=150, bbox_inches="tight")
  os.replace(tmp, scaling)
  plt.close(fig)
  paths.append(scaling)
  write_json(os.path.join(output_dir, "summary.json"), {
      "nf_dir": os.path.abspath(nf_dir),
      "crl_dir": os.path.abspath(crl_dir),
      "nf_immutable": True,
      "rows": comparison_rows,
  })
  print(f"[overlay] wrote {len(paths)} figures -> {output_dir}", flush=True)
  return paths


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--nf_dir", required=True)
  parser.add_argument("--crl_dir", required=True)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--domain_label", required=True)
  parser.add_argument("--batch_size", type=int, default=256)
  args = parser.parse_args()
  for path in write_overlay_plots(
      nf_dir=args.nf_dir, crl_dir=args.crl_dir,
      output_dir=args.output_dir, domain_label=args.domain_label,
      batch_size=args.batch_size):
    print(path)


if __name__ == "__main__":
  main()
