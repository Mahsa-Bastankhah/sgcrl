#!/usr/bin/env python3
"""Probe grid: top-5 NF goal-preimage samples with post_success_steps=2.

Does not touch the paper ghost figure. Writes a separate PDF for comparison.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)

import paper_style as ps  # noqa: E402

ps.apply()

_CKPT10 = os.path.join(
    _REPO, "figs", "builderbench",
    "creative5_task2_nf_nopermute_fixedx01_goal_preimage_post2_nomocap_ckpt10_front",
)
_S2 = os.path.join(
    _REPO, "figs", "builderbench",
    "creative5_task2_nf_nopermute_fixedx01_goal_preimage_post2_nomocap_s2_front",
)
OUT_STEM = os.path.join(
    _PAPER, "paper plots", "bb_c5t2_nf_goal_preimage_top5_post2_probe")

ROWS = (
    (200, 0, _CKPT10),
    (400, 0, _CKPT10),
    (600, 0, _CKPT10),
    (650, 0, _CKPT10),
    (800, 2, _S2),
    (1200, 2, _S2),
)


def main() -> None:
  fig, axes = plt.subplots(
      len(ROWS), 5, figsize=(18.0, 20.5), layout="constrained")
  fig.set_constrained_layout_pads(
      w_pad=0.025, h_pad=0.04, wspace=0.015, hspace=0.025)
  fig.suptitle(
      "post_success_steps=2  ·  top-5 samples per checkpoint",
      fontsize=22, y=1.01)

  for row, (iteration, seed, root) in enumerate(ROWS):
    stage = os.path.join(root, f"iter_{iteration:07d}")
    metadata_path = os.path.join(stage, "selected_samples.npz")
    if not os.path.isfile(metadata_path):
      raise FileNotFoundError(metadata_path)
    with np.load(metadata_path) as metadata:
      traj = metadata["traj_index"]
      step = metadata["traj_step"]
      weights = metadata["normalized_top_weights"]

    for rank, ax in enumerate(axes[row], start=1):
      image_path = os.path.join(stage, f"rank{rank:02d}_clean.png")
      if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
      ax.imshow(Image.open(image_path).convert("RGB"))
      ax.set_title(
          f"rank {rank} · traj {int(traj[rank - 1])}, t={int(step[rank - 1])}"
          f" · w={float(weights[rank - 1]):.2f}",
          fontsize=12, pad=4)
      ax.set_xticks([])
      ax.set_yticks([])
      for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("#B0B0B0")

    axes[row, 0].set_ylabel(
        f"checkpoint {iteration}\nseed {seed}", fontsize=16, labelpad=8)

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM, dpi=220)
  plt.close(fig)


if __name__ == "__main__":
  main()
