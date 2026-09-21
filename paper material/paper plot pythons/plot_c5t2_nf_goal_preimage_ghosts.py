#!/usr/bin/env python3
"""Paper figure: C5T2 NF goal-preimage ghosts with a goal reference panel.

Layout (left → right):
  goal | goal preimage distribution samples
       └──── training progress → ────┘

Uses post-success+3 truncated samples with mocap markers hidden.
Panels 200/400/600 are from the 40M seed-0 run; 800 from the 200M seed-2 run.

  python "paper material/paper plot pythons/plot_c5t2_nf_goal_preimage_ghosts.py"
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
    "creative5_task2_nf_nopermute_fixedx01_goal_preimage_post3_nomocap_ckpt10_front",
)
_S2 = os.path.join(
    _REPO, "figs", "builderbench",
    "creative5_task2_nf_nopermute_fixedx01_goal_preimage_post3_nomocap_s2_front",
)
OUT_STEM = os.path.join(_PAPER, "paper plots", "bb_c5t2_nf_goal_preimage_ghosts")

# Drop the thin starry sky strip at the top of every MuJoCo frame.
_TOP_CROP_FRAC = 0.045

# Larger than paper_style defaults so titles stay readable when the figure
# is shrunk into a paper column / two-column layout.
_FS_PANEL = 32
_FS_GROUP = 30
_FS_PROGRESS = 26

PANELS = (
    ("goal", os.path.join(_CKPT10, "goal_solid.png")),
    (None, os.path.join(_CKPT10, "iter_0000200", "ghost_blend.png")),
    (None, os.path.join(_CKPT10, "iter_0000400", "ghost_blend.png")),
    (None, os.path.join(_CKPT10, "iter_0000600", "ghost_blend.png")),
    (None, os.path.join(_S2, "iter_0000800", "ghost_blend.png")),
)


def _load_frame(path: str) -> np.ndarray:
  img = np.asarray(Image.open(path).convert("RGB"))
  y0 = int(round(img.shape[0] * _TOP_CROP_FRAC))
  return img[y0:]


def main() -> None:
  n = len(PANELS)
  fig = plt.figure(figsize=(21.0, 5.35))
  # Row 0: titles; row 1: images; row 2: progress arrow.
  gs = fig.add_gridspec(
      3, n,
      height_ratios=[0.16, 1.0, 0.14],
      hspace=0.06,
      wspace=0.03,
      left=0.012, right=0.988, top=0.96, bottom=0.035,
  )

  # Goal title over panel 0.
  ax_goal_title = fig.add_subplot(gs[0, 0])
  ax_goal_title.axis("off")
  ax_goal_title.text(
      0.5, 0.35, "goal",
      ha="center", va="center",
      fontsize=_FS_PANEL, color="#222222",
  )

  # Shared title over the four training panels.
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
      (0.01, 0.62), (0.99, 0.62),
      arrowstyle="-|>", mutation_scale=20,
      linewidth=1.9, color="#3A3A3A",
      shrinkA=0, shrinkB=0,
  ))
  ax_bar.text(
      0.5, 0.12, "training progress",
      ha="center", va="center",
      fontsize=_FS_PROGRESS, color="#2F2F2F",
  )

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM)
  plt.close(fig)


if __name__ == "__main__":
  main()
