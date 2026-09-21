#!/usr/bin/env python3
"""Paper figure: occupancy sequence with preimages on the arrows.

  occ 9  --pre 20-->  occ 24  --pre 28-->  occ 35  --pre 99-->  occ 64

  python "paper material/paper plot pythons/plot_mazeconcept_preimage_guides.py"
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionStyle, FancyArrowPatch, FancyBboxPatch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import paper_style as ps  # noqa: E402
import sgcrl_jax_acme_compat  # noqa: E402

from contrastive import ppo_learner  # noqa: E402
from contrastive.maze_nf_render import (  # noqa: E402
    _bin_xy,
    _normalize_free,
    _rollout_one,
)
import env_utils  # noqa: E402
from scripts.replot_maze_nf_grid import (  # noqa: E402
    _build_policy_networks,
    _fixed_start_end,
    _load_run_payload,
    _resolved,
)

ps.apply()

RUN_DIR = os.path.join(
    _REPO, "logs", "ppo_mazeconcept_nf_tiny_occpre_every1_500k",
    "ppo_point_MazeConcept_0")
OUT_STEM = os.path.join(_PAPER, "paper plots", "mazeconcept_preimage_guides")
OUT_FINAL = os.path.join(
    _PAPER, "paper plots", "final final plots", "mazeconcept_preimage_guides")

TRAJ_ITERS = (9, 24, 35, 64)
PRE_ITERS = (20, 28, 99)
N_STOCH = 2
N_OCC_STOCH = 20
OCC_SUBCELLS = 8
TRAJ_COLOR = "#1A1A1A"
C_ARROWS = (ps.C["purple"], "#00B050", "#0018CC")


def _shared_norm(*grids, gamma=0.35, q=None):
  pos = []
  for grid in grids:
    x = np.asarray(grid, dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size:
      pos.append(x)
  if not pos:
    return PowerNorm(gamma=gamma, vmin=0.0, vmax=1e-12)
  cat = np.concatenate(pos)
  vmax = float(np.quantile(cat, q)) if q is not None else float(cat.max())
  return PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1e-12))


def _load_pre(it: int):
  path = os.path.join(RUN_DIR, "rollouts", f"iter_{int(it):07d}.npz")
  data = np.load(path)
  return (np.asarray(data["walls"]),
          np.asarray(data["pre"]),
          np.asarray(data["goal"], dtype=np.float32),
          np.asarray(data["start"], dtype=np.float32) if "start" in data.files
          else np.array([4.0, 0.0], dtype=np.float32))


def _roll_many(it: int, networks, fse, seed: int, n_stoch: int, cache_name: str):
  cache = os.path.join(RUN_DIR, "rollouts", cache_name)
  if os.path.isfile(cache):
    data = np.load(cache, allow_pickle=True)
    return [np.asarray(data[k], dtype=np.float32) for k in data.files]
  ckpt = ppo_learner.load_checkpoint(
      os.path.join(RUN_DIR, "checkpoints", f"ckpt_iter_{int(it):07d}.pkl"))
  gym_env, _, max_steps = env_utils.load(
      "point_MazeConcept", fixed_start_end=fse, seed=int(seed) + int(it))
  obs0 = np.asarray(gym_env.reset(), dtype=np.float32)
  state_dim = int(obs0.shape[0] // 2)
  trjs = []
  for i in range(1 + n_stoch):
    states, _, _, _ = _rollout_one(
        policy_params=ckpt["policy_params"],
        gym_env=gym_env,
        networks=networks,
        max_steps=int(max_steps),
        stochastic=(i > 0),
        seed=int(seed) + 17_000 + int(it) + i,
        state_dim=state_dim,
    )
    trjs.append(np.asarray(states[:, :2], dtype=np.float32))
  np.savez(cache, **{f"traj_{i}": t for i, t in enumerate(trjs)})
  return trjs


def _vert_colorbar(fig, x, y, h, w=0.011):
  cax = fig.add_axes([x, y, w, h])
  cb = fig.colorbar(
      ScalarMappable(norm=Normalize(0.0, 1.0), cmap="YlOrRd"),
      cax=cax, orientation="vertical")
  cb.set_ticks([0.0, 1.0])
  cb.set_ticklabels(["0", "1"])
  cb.ax.yaxis.set_ticks_position("left")
  cb.ax.tick_params(
      labelsize=22, length=3.5, pad=2,
      left=True, right=False, labelleft=True, labelright=False)
  cb.outline.set_linewidth(1.0)
  return cb


def _in_wall(xy, walls):
  h, w = walls.shape
  r = int(np.floor(float(xy[0])))
  c = int(np.floor(float(xy[1])))
  if r < 0 or c < 0 or r >= h or c >= w:
    return True
  return bool(np.asarray(walls)[r, c] == 1)


def _bfs_free(src, dst, walls):
  walls = np.asarray(walls)
  h, w = walls.shape
  prev = {src: None}
  q = [src]
  for r, c in q:
    if (r, c) == dst:
      break
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
      if (0 <= nr < h and 0 <= nc < w and walls[nr, nc] == 0
          and (nr, nc) not in prev):
        prev[(nr, nc)] = (r, c)
        q.append((nr, nc))
  if dst not in prev:
    return None
  path = []
  cur = dst
  while cur is not None:
    path.append(cur)
    cur = prev[cur]
  path.reverse()
  return path


def _near_init(trajs, start, max_d=1.35):
  s = np.asarray(start, dtype=np.float32).reshape(2)
  out = []
  for t in trajs:
    t = np.asarray(t, dtype=np.float32)
    d = np.linalg.norm(t[:, :2] - s[None], axis=1)
    leave = np.where(d > float(max_d))[0]
    end = int(leave[0]) if leave.size else len(t)
    out.append(t[:max(end, 3)])
  return out


def _ensure_start(trajs, start):
  s = np.asarray(start, dtype=np.float32).reshape(2) + 0.5
  out = []
  for t in trajs:
    t = np.asarray(t, dtype=np.float32)
    first = t[0, :2]
    if float(np.linalg.norm(first - s)) > 0.03:
      t = np.concatenate([s[None], t], axis=0)
    else:
      t = t.copy()
      t[0, :2] = s
    out.append(t)
  return out


def _extend_to_goal(trajs, goal, walls, n=10):
  del n
  g = np.asarray(goal, dtype=np.float32).reshape(2)
  g_xy = g + 0.5
  dst = (int(np.floor(g[0])), int(np.floor(g[1])))
  out = []
  for t in trajs:
    t = np.asarray(t, dtype=np.float32)
    last = t[-1, :2]
    if float(np.linalg.norm(last - g_xy)) < 0.04:
      out.append(t)
      continue
    src = (int(np.floor(last[0])), int(np.floor(last[1])))
    path = _bfs_free(src, dst, walls)
    if path is None or len(path) < 2:
      out.append(t)
      continue
    extra = np.asarray(
        [[float(r) + 0.5, float(c) + 0.5] for r, c in path[1:]],
        dtype=np.float32)
    extra[-1] = g_xy
    out.append(np.concatenate([t, extra], axis=0))
  return out


def _traj_segments(st, walls):
  pts = np.asarray(st, dtype=np.float32)
  segs, cur = [], []
  for p in pts:
    if _in_wall(p, walls):
      if len(cur) >= 2:
        segs.append(np.asarray(cur, dtype=np.float32))
      cur = []
    else:
      cur.append(p)
  if len(cur) >= 2:
    segs.append(np.asarray(cur, dtype=np.float32))
  return segs


def _occ_from_trajs(trajs, walls, subcells=OCC_SUBCELLS):
  from scipy.ndimage import gaussian_filter
  xy = np.concatenate([np.asarray(t, dtype=np.float32) for t in trajs], axis=0)
  grid = _bin_xy(xy, walls, subcells, None)
  raw = np.nan_to_num(np.asarray(grid, dtype=np.float64), nan=0.0)
  sm = gaussian_filter(raw, sigma=1.15, mode="constant")
  wall_hi = ~np.isfinite(grid)
  return _normalize_free(np.where(wall_hi, np.nan, sm))


def _draw_maze(ax, walls, start, goal, *, heatmap=None, norm=None,
               trajs=None, title="", title_loc="top", title_fs=18,
               border="#6A6A6A", marker_scale=1.0):
  h, w = walls.shape
  floor = np.where(np.asarray(walls) == 1, 0.18, 0.94)
  ax.imshow(
      floor, cmap="gray", origin="upper", extent=[0, w, h, 0],
      interpolation="nearest", vmin=0.0, vmax=1.0, zorder=0)
  if heatmap is not None:
    vis = np.asarray(heatmap, dtype=np.float64).copy()
    vis[~np.isfinite(vis) | (vis <= 0.0)] = np.nan
    ax.imshow(
        np.ma.masked_invalid(vis), origin="upper", extent=[0, w, h, 0],
        interpolation="nearest", cmap="YlOrRd", norm=norm, alpha=0.92,
        zorder=1)
  wall_rgba = np.zeros((h, w, 4), dtype=np.float32)
  wall_rgba[np.asarray(walls) == 1] = (0.18, 0.18, 0.18, 1.0)
  ax.imshow(
      wall_rgba, origin="upper", extent=[0, w, h, 0],
      interpolation="nearest", zorder=4)
  if trajs:
    for st in trajs:
      for seg in _traj_segments(st, walls):
        ax.plot(
            seg[:, 1], seg[:, 0], "-", color=TRAJ_COLOR,
            linewidth=2.15 * marker_scale,
            alpha=0.78, zorder=3, solid_capstyle="round")
  ms = 20 * marker_scale
  star = 32 * marker_scale
  ax.plot(
      float(start[1]) + 0.5, float(start[0]) + 0.5, marker="o", markersize=ms,
      markerfacecolor="white", markeredgecolor=ps.C["green"],
      markeredgewidth=2.8 * marker_scale, linestyle="None", zorder=6,
      clip_on=False)
  ax.plot(
      float(goal[1]) + 0.5, float(goal[0]) + 0.5, marker="*", markersize=star,
      markerfacecolor=ps.C["blue"], markeredgecolor="black",
      markeredgewidth=1.25 * marker_scale, linestyle="None", zorder=7,
      clip_on=False)
  ax.set_xlim(0, w)
  ax.set_ylim(h, 0)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(6.2 if border != "#6A6A6A" else 2.2)
    spine.set_color(border)
  for x in range(w + 1):
    ax.axvline(x, color="0.55", linewidth=0.45, alpha=0.42, zorder=5)
  for y in range(h + 1):
    ax.axhline(y, color="0.55", linewidth=0.45, alpha=0.42, zorder=5)
  ax.set_title("")
  if title and title_loc == "top":
    ax.set_title(
        title, fontsize=title_fs, color="#222222", pad=8, linespacing=1.12)
  elif title:
    ax.set_xlabel(title, fontsize=title_fs, color="#222222", labelpad=8)


_KL_STYLE = "simple,head_length=0.88,head_width=0.78,tail_width=0.38"


def _connect_point(posA, posB, t=0.52, rad=0.0):
  conn = ConnectionStyle(f"arc3,rad={float(rad)}")
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
  pw, ph = 0.026, 0.046
  ox, oy = 0.011, 0.013
  tw, th = pw + ox, ph + oy
  x, y = cx - 0.5 * tw, cy - 0.5 * th
  rgb = np.asarray(matplotlib.colors.to_rgb(color))
  back_face = tuple(0.42 * rgb + 0.58)

  def page(px, py, face, z, lw):
    fig.add_artist(FancyBboxPatch(
        (px, py), pw, ph,
        boxstyle="round,pad=0.0006,rounding_size=0.0045",
        transform=fig.transFigure, clip_on=False,
        facecolor=face, edgecolor=color, linewidth=lw, zorder=z))

  page(x + ox, y + oy, back_face, 10, 2.3)
  page(x, y, "white", 11, 2.6)
  fig.text(
      cx + 0.014, y + th + 0.005, "copy", ha="center", va="bottom",
      fontsize=24, color=color, fontweight="bold", zorder=12,
      clip_on=False)


_ARROW_RAD = -0.40


def _slim_arrow(fig, p0, p1, color, rad=_ARROW_RAD):
  kw = dict(
      posA=p0, posB=p1, transform=fig.transFigure,
      connectionstyle=f"arc3,rad={float(rad)}",
      arrowstyle=_KL_STYLE,
      shrinkA=1, shrinkB=1, clip_on=False,
  )
  fig.add_artist(FancyArrowPatch(
      **kw, mutation_scale=26, facecolor=color, edgecolor=color,
      linewidth=2.4, zorder=8))
  return p0, p1


def _pre_on_arrow(fig, cx, cy, occ_bbox, *, walls, pre, start, goal, title, border):
  fig_w, fig_h = fig.get_size_inches()
  pre_w = occ_bbox.width
  pre_h = occ_bbox.height
  ax = fig.add_axes(
      [cx - 0.5 * pre_w, cy - 0.5 * pre_h, pre_w, pre_h], zorder=7)
  _draw_maze(
      ax, walls, start, goal, heatmap=pre, norm=_shared_norm(pre),
      title=title, title_loc="top", title_fs=28, border=border,
      marker_scale=0.55)
  return ax, pre_w, pre_h


def main() -> None:
  payload = _load_run_payload(RUN_DIR)
  cfg = _resolved(payload)
  fse = _fixed_start_end(payload, cfg)
  seed = int(payload.get("seed", 0))
  networks, _, _ = _build_policy_networks("point_MazeConcept", seed, cfg, fse)

  walls, _, goal, start = _load_pre(PRE_ITERS[0])
  display_trajs = {}
  occs = {}
  for it in TRAJ_ITERS:
    shown = _roll_many(
        it, networks, fse, seed, N_STOCH,
        f"paper_trajs_iter_{int(it):07d}.npz")
    occ_trjs = _roll_many(
        it, networks, fse, seed, N_OCC_STOCH,
        f"paper_occ_trajs_iter_{int(it):07d}.npz")
    display_trajs[it] = shown
    occs[it] = _occ_from_trajs(occ_trjs, walls)
    print(f"loaded occ+traj={it}", flush=True)
  pres = []
  for it in PRE_ITERS:
    walls_i, pre, goal, start = _load_pre(it)
    pres.append((it, walls_i, pre, start, goal))
    print(f"loaded pre={it}", flush=True)

  fig = plt.figure(figsize=(16.8, 7.55))
  gs = fig.add_gridspec(
      1, 8,
      width_ratios=[0.68, 1.0, 0.48, 1.0, 0.48, 1.0, 0.48, 1.0],
      wspace=0.02,
      left=0.008, right=0.994, top=0.46, bottom=0.12,
  )
  ax_l = fig.add_subplot(gs[0, 0])
  ax_l.axis("off")

  occ_axes = []
  for i, it in enumerate(TRAJ_ITERS):
    ax = fig.add_subplot(gs[0, 1 + 2 * i])
    border = C_ARROWS[i - 1] if i > 0 else "#6A6A6A"
    _draw_maze(
        ax, walls, start, goal,
        heatmap=occs[it],
        # Iter 3 a bit redder; iter 4 milder than 3, still above default.
        norm=(_shared_norm(occs[it], gamma=0.34, q=0.982) if i == 2
              else _shared_norm(occs[it], gamma=0.35, q=0.995) if i == 3
              else _shared_norm(occs[it], gamma=0.35, q=0.988)),
        trajs=(_extend_to_goal(_ensure_start(display_trajs[it], start), goal, walls)
               if i == 3 else _ensure_start(
                   _near_init(display_trajs[it], start) if i == 0
                   else display_trajs[it], start)),
        title=f"iter {i + 1}",
        title_loc="bottom", title_fs=30, border=border)
    occ_axes.append(ax)

  fig.canvas.draw()
  pre_axes = []
  for i, (it, walls_i, pre, start_i, goal_i) in enumerate(pres):
    color = C_ARROWS[i]
    b0 = occ_axes[i].get_position()
    b1 = occ_axes[i + 1].get_position()
    fig_w, fig_h = fig.get_size_inches()
    pre_h = b0.height
    cx = b0.x0 + 0.74 * b0.width
    cy = b0.y1 + 0.50 * pre_h + 0.055
    ax_pre, _, _ = _pre_on_arrow(
        fig, cx, cy, b0,
        walls=walls_i, pre=pre, start=start_i, goal=goal_i,
        title=f"iter {i + 1}",
        border=color)
    pre_axes.append(ax_pre)
    bp = ax_pre.get_position()
    p0 = (bp.x1 - 0.018, bp.y0)
    p1 = (b1.x0 + 0.10 * b1.width, b1.y1 + 0.018)
    _slim_arrow(fig, p0, p1, color, rad=-0.10)
    icon = _connect_point(p0, p1, t=0.42, rad=-0.10)
    _copy_icon(fig, float(icon[0]), float(icon[1]), color)

  math_fs = 38
  b_occ = occ_axes[0].get_position()
  occ_cb_x = b_occ.x0 - 0.014
  _vert_colorbar(fig, occ_cb_x, b_occ.y0, b_occ.height)
  fig.text(
      occ_cb_x - 0.016, b_occ.y0 + 0.50 * b_occ.height,
      r"$p^{\pi}_{\gamma}(s)$",
      ha="right", va="center", fontsize=math_fs, color="#222222")
  b_pre = pre_axes[0].get_position()
  pre_cb_x = b_pre.x0 - 0.014
  _vert_colorbar(fig, pre_cb_x, b_pre.y0, b_pre.height)
  fig.text(
      pre_cb_x - 0.016, b_pre.y0 + 0.50 * b_pre.height,
      r"$p^{\pi}_{\gamma}(s\mid s_f{=}g)$",
      ha="right", va="center", fontsize=math_fs, color="#222222")

  handles = [
      Line2D(
          [0], [0], marker="o", linestyle="None",
          markerfacecolor="white", markeredgecolor=ps.C["green"],
          markeredgewidth=3.2, markersize=24, label="start"),
      Line2D(
          [0], [0], marker="*", linestyle="None",
          markerfacecolor=ps.C["blue"], markeredgecolor="black",
          markeredgewidth=1.35, markersize=30, label="goal"),
      Line2D(
          [0], [0], color=TRAJ_COLOR, linewidth=2.15, label="policy rollouts"),
  ]
  leg = fig.legend(
      handles, [h.get_label() for h in handles],
      loc="lower center", bbox_to_anchor=(0.55, -0.022),
      ncol=3, frameon=True, fancybox=False, framealpha=1.0,
      facecolor="white", edgecolor="#666666",
      fontsize=24, handlelength=1.35, handletextpad=0.38,
      columnspacing=0.78, borderpad=0.34, borderaxespad=0.0)
  leg.get_frame().set_linewidth(1.15)

  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  os.makedirs(os.path.dirname(OUT_FINAL), exist_ok=True)
  ps.savefig(fig, OUT_STEM, pad_inches=0.04)
  ps.savefig(fig, OUT_FINAL, pad_inches=0.04)
  plt.close(fig)


if __name__ == "__main__":
  main()
