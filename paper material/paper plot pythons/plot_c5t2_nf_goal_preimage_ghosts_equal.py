#!/usr/bin/env python3
"""Paper layout: equal-weight C5T2 NF goal-preimage ghosts (post2).

Layout: goal | 200 | 400 | 600 | 1200 | 800
(+ group title + training-progress arrow). Each ghost uses identical
weights over the top-5 frames (no NF / softmax). Fonts bumped for paper.

Note: 1200 is placed left of 800 by request (visual sequence, not iter order).

Does not overwrite ``bb_c5t2_nf_goal_preimage_ghosts.pdf``.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import numpy as np

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
    _PAPER, "paper plots", "final final plots",
    "bb_c5t2_nf_goal_preimage_ghosts_equal")

_TOP_CROP_FRAC = 0.045

# Larger than the NF-weighted paper fig (32 / 30 / 26).
_FS_PANEL = 38
_FS_GROUP = 36
_FS_PROGRESS = 32

PANELS = (
    ("goal", os.path.join(
        _CKPT10, "goal_solid_policycolors.png")),
    (None, os.path.join(_CKPT10, "iter_0000200", "ghost_blend_equal.png")),
    (None, os.path.join(_CKPT10, "iter_0000400", "ghost_blend_equal.png")),
    (None, os.path.join(_CKPT10, "iter_0000600", "ghost_blend_equal.png")),
    (None, os.path.join(_S2, "iter_0001200", "ghost_blend_equal.png")),
    (None, os.path.join(_S2, "iter_0000800", "ghost_blend_equal.png")),
)


def _load_frame(path: str) -> np.ndarray:
  img = np.asarray(Image.open(path).convert("RGB"))
  y0 = int(round(img.shape[0] * _TOP_CROP_FRAC))
  return img[y0:]


def main() -> None:
  n = len(PANELS)
  fig = plt.figure(figsize=(24.5, 5.55))
  gs = fig.add_gridspec(
      3, n,
      height_ratios=[0.18, 1.0, 0.16],
      hspace=0.06,
      wspace=0.03,
      left=0.012, right=0.988, top=0.96, bottom=0.035,
  )

  ax_goal_title = fig.add_subplot(gs[0, 0])
  ax_goal_title.axis("off")
  ax_goal_title.text(
      0.5, 0.35, "goal",
      ha="center", va="center",
      fontsize=_FS_PANEL, color="#222222",
  )

  ax_group = fig.add_subplot(gs[0, 1:])
  ax_group.axis("off")
  ax_group.text(
      0.5, 0.35, "goal preimage distribution samples",
      ha="center", va="center",
      fontsize=_FS_GROUP, color="#222222",
  )

  axes = [fig.add_subplot(gs[1, i]) for i in range(n)]
  for ax, (title, path) in zip(axes, PANELS):
    if not os.path.isfile(path):
      raise FileNotFoundError(path)
    ax.imshow(_load_frame(path))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_linewidth(0.95)
      spine.set_color("#B0B0B0")
    if title == "goal":
      for spine in ax.spines.values():
        spine.set_linewidth(1.45)
        spine.set_color("#4A4A4A")

  ax_bar = fig.add_subplot(gs[2, 1:])
  ax_bar.set_xlim(0, 1)
  ax_bar.set_ylim(0, 1)
  ax_bar.axis("off")
  ax_bar.add_patch(FancyArrowPatch(
      (0.01, 0.58), (0.99, 0.58),
      arrowstyle="-|>", mutation_scale=58,
      linewidth=4.8, color="#2A2A2A",
      shrinkA=0, shrinkB=0,
  ))
  ax_bar.text(
      0.5, 0.08, "training progress",
      ha="center", va="center",
      fontsize=_FS_PROGRESS, color="#2F2F2F",
  )

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM, dpi=220)
  plt.close(fig)


if __name__ == "__main__":
  main()
