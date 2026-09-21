#!/usr/bin/env python3
"""Paper figure: maze schematic occupancy → preimage across 3 iterations.

Layout
        Iteration 1  →  Iteration 2  →  Iteration 3
  occ   p^π_γ(s)
  pre   p_←^π(s)

  python "paper material/paper plot pythons/plot_maze_concept_progression.py"
"""
from __future__ import annotations

import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionStyle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)

import paper_style as ps  # noqa: E402

ps.apply()

OUT_STEM = os.path.join(_PAPER, "paper plots", "maze_concept_progression")
FIGS_PNG = os.path.join(_REPO, "figs", "maze_concept", "maze_concept_progression.png")

# Start of the painted t1 policy cloud (bottom-left free cell).
START_RC = (4, 0)
GOAL_RC = (4, 3)


def _load_scenes():
  path = os.path.join(_REPO, "scripts", "make_maze_concept_occpre.py")
  spec = importlib.util.spec_from_file_location("maze_concept_occpre", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  t1 = mod._scene_bottom_halls_wallbehind()
  t2 = mod._scene_t2_from_t1_preimage()
  t3 = mod._scene_t3_from_t2_preimage()
  walls = t1[0]
  return walls, (
      ("Iteration 1", t1[1], t1[2]),
      ("Iteration 2", t2[1], t2[2]),
      ("Iteration 3", t3[1], t3[2]),
  )


def _shared_norm(*grids):
  vmax = 0.0
  for grid in grids:
    pos = grid[np.isfinite(grid) & (grid > 0.0)]
    if pos.size:
      vmax = max(vmax, float(pos.max()))
  return PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax if vmax > 0.0 else 1e-12)


def _free_clip_path(walls):
  h, w = walls.shape
  verts, codes = [], []
  for r in range(h):
    for c in range(w):
      if int(walls[r, c]) != 0:
        continue
      verts.extend([(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1), (c, r)])
      codes.extend([
          MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
          MplPath.LINETO, MplPath.CLOSEPOLY])
  return MplPath(verts, codes)


def _dot_xy(grid, walls, jitter=0.42, seed=11):
  raw = np.asarray(grid, dtype=np.float64)
  h, w = walls.shape
  sub = int(raw.shape[0] // h)
  ii, jj = np.nonzero(np.isfinite(raw) & (raw > 0.0))
  rng = np.random.default_rng(int(seed) + 17 * int(ii.size))
  ys = (ii.astype(np.float64) + 0.5 + rng.uniform(-jitter, jitter, ii.size)) / sub
  xs = (jj.astype(np.float64) + 0.5 + rng.uniform(-jitter, jitter, jj.size)) / sub
  return xs, ys, raw[ii, jj]


def _draw_maze(ax, walls, heatmap, norm, border="#6A6A6A"):
  h, w = walls.shape
  floor = np.where(np.asarray(walls) == 1, 0.18, 0.94)
  ax.imshow(
      floor, cmap="gray", origin="upper", extent=[0, w, h, 0],
      interpolation="nearest", vmin=0.0, vmax=1.0, zorder=0)
  xs, ys, mass = _dot_xy(heatmap, walls)
  colors = plt.get_cmap("YlOrRd")(norm(mass))
  clip = _free_clip_path(walls)
  halo = ax.scatter(
      xs, ys, s=62, c=colors, marker="o", linewidths=0,
      alpha=0.38, zorder=2)
  core = ax.scatter(
      xs, ys, s=26, c=colors, marker="o", linewidths=0,
      alpha=0.98, zorder=3)
  for sc in (halo, core):
    sc.set_clip_path(clip, ax.transData)
    sc.set_clip_on(True)
  wall_rgba = np.zeros((h, w, 4), dtype=np.float32)
  wall_rgba[np.asarray(walls) == 1] = (0.18, 0.18, 0.18, 1.0)
  ax.imshow(
      wall_rgba, origin="upper", extent=[0, w, h, 0],
      interpolation="nearest", zorder=4)
  sx, sy = START_RC[1] + 0.5, START_RC[0] + 0.5
  gx, gy = GOAL_RC[1] + 0.5, GOAL_RC[0] + 0.5
  ax.plot(
      sx, sy, marker="o", markersize=22,
      markerfacecolor="white", markeredgecolor=ps.C["green"],
      markeredgewidth=3.0, linestyle="None", zorder=5, clip_on=False,
      label="start")
  ax.plot(
      gx, gy, marker="*", markersize=36,
      markerfacecolor=ps.C["blue"], markeredgecolor="black",
      markeredgewidth=1.35, linestyle="None", zorder=6, clip_on=False,
      label="goal")
  ax.set_xlim(0, w)
  ax.set_ylim(h, 0)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])
  thick = 10.0 if border != "#6A6A6A" else 1.15
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(thick)
    spine.set_color(border)
  for x in range(w + 1):
    ax.axvline(x, color="0.55", linewidth=0.45, alpha=0.42, zorder=5)
  for y in range(h + 1):
    ax.axhline(y, color="0.55", linewidth=0.45, alpha=0.42, zorder=5)


_KL_STYLE = "simple,head_length=0.85,head_width=0.72,tail_width=0.32"
C_STEP1 = ps.C["purple"]
# Teal, not Okabe green — that green is the start marker.
C_STEP2 = "#007A8C"


def _connect_point(posA, posB, t=0.52):
  """Point at fraction t along the angle3 connector, figure coords."""
  conn = ConnectionStyle("angle3,angleA=90,angleB=18")
  path = conn.connect(posA, posB)
  verts = np.asarray(path.vertices, dtype=np.float64)
  seg = np.sqrt(((verts[1:] - verts[:-1]) ** 2).sum(axis=1))
  s = np.concatenate([[0.0], np.cumsum(seg)])
  if s[-1] <= 1e-12:
    return np.array(posA, dtype=np.float64)
  target = float(t) * s[-1]
  i = int(np.searchsorted(s, target) - 1)
  i = max(0, min(i, len(verts) - 2))
  u = 0.0 if seg[i] <= 1e-12 else (target - s[i]) / seg[i]
  return verts[i] * (1.0 - u) + verts[i + 1] * u


def _copy_icon(fig, cx, cy, color):
  """Two overlapping pages centered at (cx, cy) in figure coords."""
  pw, ph = 0.0185, 0.033
  ox, oy = 0.008, 0.010
  tw, th = pw + ox, ph + oy
  x, y = cx - 0.5 * tw, cy - 0.5 * th
  pad = 0.005
  fig.add_artist(FancyBboxPatch(
      (x - pad, y - pad), tw + 2 * pad, th + 2 * pad,
      boxstyle="round,pad=0.0008,rounding_size=0.008",
      transform=fig.transFigure, clip_on=False,
      facecolor="white", edgecolor="none", linewidth=0, zorder=8))
  rgb = np.asarray(matplotlib.colors.to_rgb(color))
  back_face = tuple(0.42 * rgb + 0.58)
  def page(px, py, face, z, lw):
    fig.add_artist(FancyBboxPatch(
        (px, py), pw, ph,
        boxstyle="round,pad=0.0006,rounding_size=0.0035",
        transform=fig.transFigure, clip_on=False,
        facecolor=face, edgecolor=color, linewidth=lw, zorder=z))
  page(x + ox, y + oy, back_face, 10, 1.7)
  page(x, y, "white", 11, 1.85)


def _kl_arrow(fig, ax_from, ax_to, color, text_x, text_ha="left", icon_t=0.52):
  """preimage_t → occupancy_{t+1}  (policy update)."""
  b0 = ax_from.get_position()
  b1 = ax_to.get_position()
  p0 = (b0.x0 + 0.92 * b0.width, b0.y1 + 0.004)
  p1 = (b1.x0 + 0.16 * b1.width, b1.y0 - 0.004)
  kw = dict(
      posA=p0, posB=p1, transform=fig.transFigure,
      connectionstyle="angle3,angleA=90,angleB=18",
      arrowstyle=_KL_STYLE,
      shrinkA=1, shrinkB=1, clip_on=False,
  )
  fig.add_artist(FancyArrowPatch(
      **kw, mutation_scale=34, facecolor=color, edgecolor=color,
      linewidth=0.0, zorder=6))
  fig.add_artist(FancyArrowPatch(
      **kw, mutation_scale=29, facecolor="white", edgecolor=color,
      linewidth=1.8, zorder=7))
  mid = _connect_point(p0, p1, t=icon_t)
  _copy_icon(fig, mid[0], mid[1], color)
  fig.text(
      text_x,
      b0.y1 + 0.018,
      "state matching",
      ha=text_ha, va="bottom", fontsize=22, color=color,
      zorder=10, clip_on=False)


def main() -> None:
  walls, scenes = _load_scenes()
  occs = [s[1] for s in scenes]
  pres = [s[2] for s in scenes]
  occ_norm = _shared_norm(*occs)
  pre_norm = _shared_norm(*pres)

  fig = plt.figure(figsize=(12.2, 6.55))
  gs = fig.add_gridspec(
      2, 4,
      height_ratios=[1.0, 1.0],
      width_ratios=[0.58, 1.0, 1.0, 1.0],
      wspace=0.035,
      hspace=0.19,
      left=0.018, right=0.980, top=0.922, bottom=0.048,
  )

  handles = [
      Line2D(
          [0], [0], marker="o", linestyle="None",
          markerfacecolor="white", markeredgecolor=ps.C["green"],
          markeredgewidth=3.0, markersize=22, label="start"),
      Line2D(
          [0], [0], marker="*", linestyle="None",
          markerfacecolor=ps.C["blue"], markeredgecolor="black",
          markeredgewidth=1.35, markersize=28, label="goal"),
  ]

  row_specs = (
      (0, occs, occ_norm, r"$p^{\pi}_{\gamma}(s)$", "policy\noccupancy\nmeasure"),
      (1, pres, pre_norm, r"$p_{\leftarrow}^{\pi}(s)$", "goal\npreimage\ndistribution"),
  )
  occ_axes, pre_axes = [], []
  for row, grids, norm, math, words in row_specs:
    ax_l = fig.add_subplot(gs[row, 0])
    ax_l.axis("off")
    ax_l.text(
        0.50, 0.68, math, ha="center", va="center",
        fontsize=40, color="#222222", clip_on=False)
    ax_l.text(
        0.50, 0.28, words, ha="center", va="center",
        fontsize=24, color="#333333", linespacing=1.12, clip_on=False)
    row_axes = []
    for i, grid in enumerate(grids):
      ax = fig.add_subplot(gs[row, 1 + i])
      if row == 0:
        border = (C_STEP1 if i == 1 else C_STEP2 if i == 2 else "#6A6A6A")
      else:
        border = (C_STEP1 if i == 0 else C_STEP2 if i == 1 else "#6A6A6A")
      _draw_maze(ax, walls, grid, norm, border=border)
      row_axes.append(ax)
    if row == 0:
      occ_axes = row_axes
    else:
      pre_axes = row_axes

  fig.canvas.draw()
  for ax, title in zip(occ_axes, ("iter 1", "iter 100", "iter 200")):
    b = ax.get_position()
    fig.text(
        0.5 * (b.x0 + b.x1), b.y1 + 0.004, title,
        ha="center", va="bottom", fontsize=22, color="#222222",
        clip_on=False)
  leg = fig.legend(
      handles, [h.get_label() for h in handles],
      loc="upper left", bbox_to_anchor=(0.014, 0.978),
      ncol=2, frameon=True, fancybox=False, framealpha=1.0,
      facecolor="white", edgecolor="#666666",
      fontsize=22, handlelength=1.05, handletextpad=0.35,
      columnspacing=0.70, borderpad=0.32, borderaxespad=0.0)
  leg.get_frame().set_linewidth(1.15)
  try:
    leg.get_frame().set_boxstyle("square", pad=0.2)
  except Exception:
    pass
  b_pre0 = pre_axes[0].get_position()
  b_pre1 = pre_axes[1].get_position()
  _kl_arrow(
      fig, pre_axes[0], occ_axes[1], C_STEP1,
      text_x=b_pre0.x0 - 0.022, text_ha="left", icon_t=0.46)
  _kl_arrow(
      fig, pre_axes[1], occ_axes[2], C_STEP2,
      text_x=b_pre1.x0 + 0.48 * b_pre1.width, text_ha="center",
      icon_t=0.78)

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  os.makedirs(os.path.dirname(FIGS_PNG), exist_ok=True)
  ps.savefig(fig, OUT_STEM, pad_inches=0.18)
  fig.savefig(FIGS_PNG, dpi=160, facecolor="white", bbox_inches="tight",
              pad_inches=0.18)
  print("Saved:", FIGS_PNG)
  plt.close(fig)


if __name__ == "__main__":
  main()
