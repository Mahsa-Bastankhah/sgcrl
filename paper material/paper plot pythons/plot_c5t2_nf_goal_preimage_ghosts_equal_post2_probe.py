#!/usr/bin/env python3
"""Equal-weight ghost blends from existing top-5 clean frames (post2 probe).

Does not re-score with NF / softmax weights — every rank frame gets w=1/K.
Does not overwrite the paper ghost figure.
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
    _PAPER, "paper plots", "bb_c5t2_nf_goal_preimage_ghosts_equal_post2_probe")

STAGES = (
    (200, 0, _CKPT10),
    (400, 0, _CKPT10),
    (600, 0, _CKPT10),
    (650, 0, _CKPT10),
    (800, 2, _S2),
    (1200, 2, _S2),
)


def _equal_ghost(frames: list[np.ndarray], floor: float = 0.15) -> np.ndarray:
  """Same compositing as ``_ghost_max``, but with identical weights."""
  k = len(frames)
  w = np.full(k, 1.0 / k, dtype=np.float64)
  blend = np.zeros(frames[0].shape, dtype=np.float64)
  for f, wi in zip(frames, w):
    blend += wi * f.astype(np.float64)
  ghost = np.zeros_like(blend)
  for f, wi in zip(frames, w):
    a = floor + (1.0 - floor) * wi
    ghost = np.maximum(ghost, a * f.astype(np.float64))
  out = 0.55 * blend + 0.45 * ghost
  return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _load_ranks(stage_dir: str) -> list[np.ndarray]:
  frames = []
  for rank in range(1, 6):
    path = os.path.join(stage_dir, f"rank{rank:02d}_clean.png")
    if not os.path.isfile(path):
      raise FileNotFoundError(path)
    frames.append(np.asarray(Image.open(path).convert("RGB")))
  return frames


def main() -> None:
  ghosts = []
  labels = []
  for iteration, seed, root in STAGES:
    stage = os.path.join(root, f"iter_{iteration:07d}")
    frames = _load_ranks(stage)
    ghost = _equal_ghost(frames)
    out = os.path.join(stage, "ghost_blend_equal.png")
    Image.fromarray(ghost).save(out)
    ghosts.append(ghost)
    labels.append(f"iter {iteration}  seed {seed}")
    print(f"wrote {out}")

  fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), layout="constrained")
  fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.08, wspace=0.03, hspace=0.08)
  fig.suptitle(
      "equal-weight ghosts  (post_success_steps=2, no NF softmax weights)",
      fontsize=22)
  for ax, ghost, lab in zip(axes.ravel(), ghosts, labels):
    ax.imshow(ghost)
    ax.set_title(lab, fontsize=16, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_linewidth(0.8)
      spine.set_color("#B0B0B0")

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM, dpi=220)
  plt.close(fig)


if __name__ == "__main__":
  main()
