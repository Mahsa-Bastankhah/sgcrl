#!/usr/bin/env python3
"""Paper figure: SixteenRooms PPO reward heatmaps at iterations 5 and 30.

2x4 grid, columns PIM (NF) / PIM (TD3) / PIM (CRL) / PIM (TD-InfoNCE). Each panel uses that
frame's free-cell p02-p98 color scale (same as density_evolution_autoscale).

  python "paper material/paper plot pythons/plot_sixteenrooms_density_reward_heatmaps.py"
"""
from __future__ import annotations

import importlib.util
import os
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import paper_style as ps  # noqa: E402

ps.apply()

LOG_ROOT = os.path.join(_REPO, "logs", "ppo_sixteenrooms_density_diag_1m")
OUT_STEM = os.path.join(
    _PAPER, "paper plots", "final final plots",
    "sixteenrooms_density_reward_heatmaps")

ITERS = (5, 30)
# This 2x4 heatmap is included at ~textwidth, so paper_style defaults
# shrink too far. Titles / legend / row labels are the exception.
FS_TITLE = 42
FS_LEGEND = 36
FS_ROW = 36
METHODS = (
    ("nf_tiny", "PIM (NF)", r"$\log p(g \mid s,a)$"),
    ("td3_logq", "PIM (TD3)", r"$\log((1-\gamma)Q(s,a,g))$"),
    ("crl", "PIM (CRL)", r"$\phi(s,a)^{\top}\psi(g)$"),
    ("tdinfonce", "PIM (TD-InfoNCE)", r"$\phi(s,a)^{\top}\psi(g)$"),
)


def _load_evo():
  path = os.path.join(_REPO, "scripts", "plot_sixteenrooms_reward_evolution.py")
  spec = importlib.util.spec_from_file_location("sixteenrooms_reward_evo", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _snapshot_path(method: str, iteration: int) -> str:
  return os.path.join(
      LOG_ROOT, method, "ppo_point_SixteenRooms_0", "density_snapshots",
      f"snapshot_iter_{int(iteration):07d}.pkl")


def _load_snapshot(path: str) -> dict:
  if not os.path.isfile(path):
    raise FileNotFoundError(path)
  with open(path, "rb") as handle:
    item = pickle.load(handle)
  if int(item.get("schema_version", -1)) != 1:
    raise ValueError(f"Unsupported snapshot schema in {path}")
  item["_path"] = path
  return item


def _walls(evo):
  gym_env, _, _ = evo.env_utils.load(
      "point_SixteenRooms",
      fixed_start_end=evo.fixed_goal_dict["point_SixteenRooms"],
      seed=0)
  return np.asarray(gym_env._walls, dtype=np.int8)


def _panel_field(evo, snapshot, models, rows, cols, walls):
  field = evo._reward_field(snapshot, models, evo._free_cell_centers(walls)[2])
  p02, _median, p98, vmin, vmax = evo._frame_robust_stats(field)
  grid = np.full(walls.shape, np.nan, dtype=np.float32)
  grid[rows, cols] = field
  return grid, float(p02), float(p98), float(vmin), float(vmax)


def _draw_maze(ax, walls, grid, vmin, vmax):
  h, w = walls.shape
  image = ax.imshow(
      np.ma.masked_where(walls != 0, grid),
      origin="upper", extent=[0, w, h, 0], cmap="magma",
      vmin=vmin, vmax=vmax, interpolation="nearest", zorder=1)
  wall_rgba = np.zeros((h, w, 4), dtype=np.float32)
  wall_rgba[np.asarray(walls) == 1] = (0.12, 0.12, 0.12, 1.0)
  ax.imshow(
      wall_rgba, origin="upper", extent=[0, w, h, 0],
      interpolation="nearest", zorder=2)
  ax.plot(
      0.5, 0.5, marker="o", markersize=13,
      markerfacecolor="white", markeredgecolor=ps.C["green"],
      markeredgewidth=2.4, linestyle="None", zorder=5, clip_on=True)
  ax.plot(
      20.5, 20.5, marker="*", markersize=18,
      markerfacecolor=ps.C["sky"], markeredgecolor="black",
      markeredgewidth=1.15, linestyle="None", zorder=6, clip_on=True)
  ax.set_xlim(0, w)
  ax.set_ylim(h, 0)
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])
  ax.tick_params(length=0)
  for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(ps.SPINE)
    spine.set_color("black")
  ax.grid(False)
  return image


def _compute_panels(evo, walls):
  rows, cols, _positions = evo._free_cell_centers(walls)
  panels = []
  for method, name, formula in METHODS:
    models = None
    for iteration in ITERS:
      snapshot = _load_snapshot(_snapshot_path(method, iteration))
      if int(snapshot["iteration"]) != int(iteration):
        raise ValueError(
            f'{snapshot["_path"]}: expected iteration {iteration}, '
            f'got {snapshot["iteration"]}')
      if models is None:
        models = evo._build_models(snapshot)
      grid, p02, p98, vmin, vmax = _panel_field(
          evo, snapshot, models, rows, cols, walls)
      print(
          f"{method} iter {iteration}: steps={int(snapshot['global_step']):,} "
          f"p02={p02:.6g} p98={p98:.6g}",
          flush=True)
      panels.append(dict(
          method=method, name=name, formula=formula, iteration=iteration,
          grid=grid, vmin=vmin, vmax=vmax))
  return panels


def main() -> None:
  evo = _load_evo()
  walls = _walls(evo)
  panels = _compute_panels(evo, walls)

  fig = plt.figure(figsize=(20.8, 11.2))
  grid = ImageGrid(
      fig, (0.08, 0.03, 0.90, 0.70),
      nrows_ncols=(2, 4),
      axes_pad=(0.42, 0.58),
      cbar_mode="each",
      cbar_location="right",
      cbar_pad=0.04,
      cbar_size="3.6%",
      share_all=True,
  )

  for col, (method, name, formula) in enumerate(METHODS):
    for row, iteration in enumerate(ITERS):
      panel = next(
          item for item in panels
          if item["method"] == method and item["iteration"] == iteration)
      ax = grid[row * 4 + col]
      image = _draw_maze(ax, walls, panel["grid"], panel["vmin"], panel["vmax"])
      if row == 0:
        ax.set_title(f"{name}\n{formula}", fontsize=FS_TITLE, pad=16)
      if col == 0:
        ax.set_ylabel(f"Iteration\n{iteration}", fontsize=FS_ROW, labelpad=12)
      cbar = grid.cbar_axes[row * 4 + col]
      cb = fig.colorbar(image, cax=cbar, extend="both")
      cb.set_ticks([])
      cb.ax.tick_params(size=0, labelleft=False, labelright=False)
      cb.outline.set_linewidth(ps.SPINE)

  handles = [
      Line2D(
          [0], [0], marker="o", color="none",
          markerfacecolor="white", markeredgecolor=ps.C["green"],
          markeredgewidth=2.6, markersize=16, label="start"),
      Line2D(
          [0], [0], marker="*", color="none",
          markerfacecolor=ps.C["sky"], markeredgecolor="black",
          markeredgewidth=1.3, markersize=22, label="goal"),
  ]
  fig.legend(
      handles, ["start", "goal"], loc="upper center", ncol=2,
      bbox_to_anchor=(0.5, 0.995), frameon=True, fancybox=False,
      framealpha=0.96, edgecolor="#B0B0B0", fontsize=FS_LEGEND,
      handletextpad=0.5, columnspacing=1.8, borderpad=0.45)
  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM)
  plt.close(fig)


if __name__ == "__main__":
  main()
