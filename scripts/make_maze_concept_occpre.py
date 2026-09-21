"""Schematic occupancy / goal-preimage panels for paper figures.

Paints hand-specified mass on ImpossibleOpen and writes the same two-panel
PNG as in-train maze renders (YlOrRd, shared PowerNorm, goal star).
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paper schematic: drop left col; the (3,0) wall stub moves +1 so it remains.
GOAL_RC = (4, 4)
GOAL = np.array([4.0, 4.0], dtype=np.float32)
PAPER_WALLS = np.array(
    [[0, 0, 0, 0, 0],
     [0, 1, 1, 1, 0],
     [0, 0, 1, 0, 0],
     [1, 0, 1, 0, 1],
     [0, 0, 1, 0, 0]],
    dtype=np.int32)


def _paper_walls() -> np.ndarray:
  return PAPER_WALLS.copy()


def _bin_xy(xy, walls, subcells, weights):
  sub = max(1, int(subcells))
  h, w = walls.shape
  hh, ww = h * sub, w * sub
  grid = np.zeros((hh, ww), dtype=np.float64)
  i = np.floor(xy[:, 0] * sub).astype(np.int32)
  j = np.floor(xy[:, 1] * sub).astype(np.int32)
  valid = (i >= 0) & (i < hh) & (j >= 0) & (j < ww)
  if weights is None:
    np.add.at(grid, (i[valid], j[valid]), 1.0)
  else:
    wwts = np.asarray(weights, dtype=np.float64).reshape(-1)
    np.add.at(grid, (i[valid], j[valid]), wwts[valid])
  wall_hi = np.repeat(np.repeat(np.asarray(walls) == 1, sub, axis=0), sub, axis=1)
  return np.where(wall_hi, np.nan, grid).astype(np.float32)


def _normalize_free(grid):
  out = np.asarray(grid, dtype=np.float64).copy()
  mass = np.nansum(out)
  if mass > 0.0:
    out /= mass
  return out.astype(np.float32)


def _shared_power_norm(*grids):
  vmax = 0.0
  for grid in grids:
    pos = np.asarray(grid, dtype=np.float64)
    pos = pos[np.isfinite(pos) & (pos > 0.0)]
    if pos.size:
      vmax = max(vmax, float(np.max(pos)))
  return PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax if vmax > 0.0 else 1e-12)


def _draw_occ_panel(ax, walls, heatmap, goal, heat_label, panel_title,
                    norm=None, add_colorbar=True):
  h, w = walls.shape
  fig = ax.figure
  floor = np.where(np.asarray(walls) == 1, 0.18, 0.94)
  ax.imshow(
      floor, cmap='gray', origin='upper', extent=[0, w, h, 0],
      interpolation='nearest', vmin=0.0, vmax=1.0)
  vis = np.asarray(heatmap, dtype=np.float64).copy()
  vis[~np.isfinite(vis) | (vis <= 0.0)] = np.nan
  pos = vis[np.isfinite(vis)]
  divider = make_axes_locatable(ax)
  if pos.size > 0:
    if norm is None:
      vmax = float(np.max(pos))
      vmin = float(np.min(pos))
      if vmax <= vmin:
        vmax = vmin * 1.01 + 1e-12
      norm = PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
    hm = ax.imshow(
        np.ma.masked_invalid(vis), origin='upper',
        extent=[0, w, h, 0], interpolation='nearest', alpha=0.88,
        cmap='YlOrRd', norm=norm)
    if add_colorbar:
      cax = divider.append_axes('right', size='4%', pad=0.12)
      cbar = fig.colorbar(hm, cax=cax)
      if heat_label:
        cbar.ax.set_ylabel(heat_label, rotation=270, labelpad=14)
  ax.plot(goal[1], goal[0], marker='*', color='tab:blue', markersize=16,
          markeredgecolor='black', linewidth=0, label='goal')
  ax.set_xlim(0, w)
  ax.set_ylim(h, 0)
  ax.set_aspect('equal')
  ax.set_xlabel('col  (x)')
  ax.set_ylabel('row  (y, inverted)')
  ax.set_title(panel_title, fontsize=11)
  ax.set_xticks(np.arange(0, w + 1, 1))
  ax.set_yticks(np.arange(0, h + 1, 1))
  ax.grid(True, color='0.55', linewidth=0.4, alpha=0.45)
  ax.legend(loc='upper right', fontsize=8, framealpha=0.85)


def save_occupancy_preimage_figure(
    out_path, walls, occ, pre, goal, title, subtitle='', shared_scale=True):
  walls = np.asarray(walls)
  h, w = walls.shape
  shared = _shared_power_norm(occ, pre) if shared_scale else None
  fig, axes = plt.subplots(
      1, 2, figsize=(9.8, 5.6 * h / max(w, 1)), layout='constrained')
  fig.suptitle(f'{title}\n{subtitle}'.rstrip(), fontsize=11)
  _draw_occ_panel(
      axes[0], walls, occ, goal,
      None, r'on-policy occupancy $\mu(s)$',
      norm=shared, add_colorbar=False)
  _draw_occ_panel(
      axes[1], walls, pre, goal,
      'probability mass',
      r'goal preimage $\mu(s)\,p(g\mid s,a)/Z$',
      norm=shared, add_colorbar=True)
  fig.savefig(out_path, dpi=140)
  plt.close(fig)
  return out_path


def _clip_to_free(xy: np.ndarray, walls: np.ndarray) -> np.ndarray:
  """Keep points on free floor. Crossing a free–free cell edge is allowed."""
  return xy[_on_free_floor(xy, walls)]


def _on_free_floor(xy: np.ndarray, walls: np.ndarray) -> np.ndarray:
  h, w = walls.shape
  r = np.floor(xy[:, 0]).astype(np.int32)
  c = np.floor(xy[:, 1]).astype(np.int32)
  ok = (xy[:, 0] >= 0.0) & (xy[:, 0] < h) & (xy[:, 1] >= 0.0) & (xy[:, 1] < w)
  ok &= (r >= 0) & (r < h) & (c >= 0) & (c < w)
  ok &= walls[np.clip(r, 0, h - 1), np.clip(c, 0, w - 1)] == 0
  ok &= ~((r == GOAL_RC[0]) & (c == GOAL_RC[1]))
  return ok


def _spill_keep_weights(xy, weights, walls, rng, frac=0.26, scale=0.13):
  """Bleed a few points across free–free borders; keep each point's color."""
  xy = np.asarray(xy, dtype=np.float64).copy()
  n = int(xy.shape[0])
  if n == 0:
    if weights is None:
      return xy.astype(np.float32), None
    return xy.astype(np.float32), np.asarray(weights, dtype=np.float64)
  w = None if weights is None else np.asarray(weights, dtype=np.float64).copy()
  xy[:, 0] += rng.normal(0.0, 0.05, n)
  xy[:, 1] += rng.normal(0.0, 0.05, n)
  k = max(1, int(round(frac * n)))
  idx = rng.choice(n, size=min(k, n), replace=False)
  step = rng.uniform(0.05, 0.20, size=idx.size)
  edge = rng.integers(0, 4, size=idx.size)
  xy[idx, 1] += np.where(edge == 0, step, np.where(edge == 1, -step, 0.0))
  xy[idx, 0] += np.where(edge == 2, step, np.where(edge == 3, -step, 0.0))
  xy[idx, 0] += rng.normal(0.0, scale * 0.35, idx.size)
  xy[idx, 1] += rng.normal(0.0, scale * 0.35, idx.size)
  ok = _on_free_floor(xy, walls)
  if w is None:
    return xy[ok].astype(np.float32), None
  return xy[ok].astype(np.float32), w[ok]


def _sample_cells(walls, rng, cells, blob=False):
  chunks = []
  for row, col, n_typ in cells:
    if row < 0 or col < 0 or row >= walls.shape[0] or col >= walls.shape[1]:
      continue
    if int(walls[row, col]) != 0:
      continue
    if int(row) == GOAL_RC[0] and int(col) == GOAL_RC[1]:
      continue
    # Large cells: same jitter as the approved t1 cloud.
    # Tiny/faint cells: keep a tight count so they stay specks.
    if n_typ >= 20:
      lo, hi = n_typ - 10, n_typ + 12
    else:
      lo = max(4, n_typ - max(2, n_typ // 4))
      hi = n_typ + max(3, n_typ // 5)
    n = int(rng.integers(lo, hi + 1))
    if blob:
      # Off-center cloud so a cell does not stamp a filled square.
      cr = row + float(rng.uniform(0.22, 0.78))
      cc = col + float(rng.uniform(0.22, 0.78))
      sig = float(rng.uniform(0.14, 0.26))
      rr = rng.normal(cr, sig, n)
      cc = rng.normal(cc, sig, n)
    else:
      rr = row + rng.uniform(0.04, 0.96, n)
      cc = col + rng.uniform(0.04, 0.96, n)
      rr += rng.normal(0.0, 0.11, n)
      cc += rng.normal(0.0, 0.11, n)
    chunks.append(np.stack([rr, cc], axis=1))
  if not chunks:
    return np.zeros((0, 2), dtype=np.float32)
  return _clip_to_free(np.concatenate(chunks, axis=0), walls).astype(np.float32)


def _sample_right_of_cell(walls, rng, row: int, col: int, n_typ: int):
  """Points inside (row, col) piled toward the right edge (x=col+1)."""
  if int(walls[row, col]) != 0:
    return np.zeros((0, 2), dtype=np.float32)
  n = int(rng.integers(max(10, n_typ - 6), n_typ + 8))
  u = rng.beta(3.6, 1.35, n)
  cc = col + 0.22 + 0.74 * u
  rr = row + rng.uniform(0.08, 0.92, n)
  rr += rng.normal(0.0, 0.06, n)
  cc += rng.normal(0.0, 0.04, n)
  return _clip_to_free(np.stack([rr, cc], axis=1), walls).astype(np.float32)


def _t1_weights(xy, rng):
  """Within-cell: orange on the wall face; red only at goal height + nearest wall."""
  r = xy[:, 0]
  c = xy[:, 1]
  r_i = np.floor(r).astype(np.int32)
  c_i = np.floor(c).astype(np.int32)
  noise = rng.lognormal(0.0, 0.12, size=xy.shape[0])
  # Wall face is x=2. Hug it without painting the whole cell one color.
  hug_wall = np.exp(-((2.0 - c) ** 2) / (2.0 * 0.20 ** 2))
  # Star sits at y=4; a short band at that height in (4,1) goes red.
  same_h = np.exp(-((r - 4.0) ** 2) / (2.0 * 0.22 ** 2))
  on_col1 = c_i == 1
  in_41 = (r_i == 4) & (c_i == 1)
  # Pale leftover everywhere else.
  weights = 0.16 * noise
  # Orange: col-1 points near the wall (varies inside the cell).
  weights = np.where(on_col1, (0.45 + 1.15 * hug_wall) * noise, weights)
  # (1,0) against the bar — yellow, and only near the right edge.
  on_10 = (r_i == 1) & (c_i == 0)
  hug_bar = np.exp(-((1.0 - c) ** 2) / (2.0 * 0.18 ** 2))
  weights = np.where(on_10, (0.16 + 0.22 * hug_bar) * noise, weights)
  # Red: same height as the star AND closest to the wall, inside (4,1).
  red = in_41.astype(np.float64) * hug_wall * same_h
  return weights + 4.6 * red * noise


def _t2_weights(xy, rng):
  """New top-right pocket. Col-1 wall face is only a faint leftover."""
  r_i = np.floor(xy[:, 0]).astype(np.int32)
  c_i = np.floor(xy[:, 1]).astype(np.int32)
  c = xy[:, 1]
  weights = rng.lognormal(0.0, 0.14, size=xy.shape[0])
  focus = (
      ((r_i == 1) & (c_i == 4))
      | ((r_i == 2) & ((c_i == 3) | (c_i == 4)))
  )
  in_01 = (r_i == 0) & (c_i == 1)
  in_02 = (r_i == 0) & (c_i == 2)
  in_03 = (r_i == 0) & (c_i == 3)
  in_04 = (r_i == 0) & (c_i == 4)
  # Cross the (0,1)|(0,2) wall smoothly; rest of col 1 stays faint.
  fade01 = 0.40 + 2.50 * np.exp(-((2.0 - c) ** 2) / (2.0 * 0.40 ** 2))
  fade02 = 0.72 + 0.28 * np.clip(1.0 - (c - 2.0), 0.0, 1.0)
  # (0,3) → (0,4): continuous rightward ramp, no color cliff.
  fade_top = 3.2 + 1.6 * np.clip((c - 3.0) / 2.0, 0.0, 1.0)
  trail = (c_i == 1) & np.isin(r_i, (2, 3, 4))
  weights = np.where(focus, weights * 5.0, weights)
  weights = np.where(in_01, weights * fade01, weights)
  weights = np.where(in_02, weights * 2.9 * fade02, weights)
  weights = np.where(in_03 | in_04, weights * fade_top, weights)
  weights = np.where(trail, weights * 0.32, weights)
  leftover = ~(focus | trail | in_01 | in_02 | in_03 | in_04)
  return np.where(leftover, weights * 0.16, weights)


def _t3_weights(xy, rng):
  """Reddest next to the goal; orange scatter back up the right corridor."""
  r = xy[:, 0]
  c = xy[:, 1]
  r_i = np.floor(r).astype(np.int32)
  c_i = np.floor(c).astype(np.int32)
  noise = rng.lognormal(0.0, 0.13, size=xy.shape[0])
  dist = np.hypot(r - GOAL[0], c - GOAL[1])
  prox = np.exp(-0.72 * dist)
  on_up = (
      ((r_i == 4) & (c_i == 3))
      | ((r_i == 3) & (c_i == 3))
      | ((r_i == 2) & ((c_i == 3) | (c_i == 4)))
      | ((r_i == 1) & (c_i == 4))
      | ((r_i == 0) & (c_i >= 2))
  )
  trail01 = (r_i == 0) & (c_i == 1)
  weights = 0.12 * noise
  weights = np.where(on_up, (0.50 + 2.6 * prox) * noise, weights)
  weights = np.where(trail01, 0.32 * noise, weights)
  # Red only in (4,3), nearest the star — not the whole cell one color.
  in_43 = (r_i == 4) & (c_i == 3)
  hug_goal = (
      np.exp(-((4.0 - c) ** 2) / (2.0 * 0.22 ** 2))
      * np.exp(-((4.0 - r) ** 2) / (2.0 * 0.22 ** 2)))
  return weights + 4.4 * in_43.astype(np.float64) * hug_goal * noise


def _t4_weights(xy, rng):
  """Gradient toward the goal along the right corridor, not a jump to the well."""
  r = xy[:, 0]
  c = xy[:, 1]
  r_i = np.floor(r).astype(np.int32)
  c_i = np.floor(c).astype(np.int32)
  on_path = (
      (np.isin(r_i, (0, 1, 2)) & (c_i >= 1))
      | ((r_i == 3) & (c_i == 3))
      | ((r_i == 4) & (c_i == 3))
  )
  dist = np.hypot(r - GOAL[0], c - GOAL[1])
  prox = np.exp(-0.40 * dist)
  weights = prox * rng.lognormal(0.0, 0.14, size=xy.shape[0])
  weights = np.where(on_path, weights * 5.5, weights * 0.10)
  well = ((r_i == 3) & (c_i == 3)) | ((r_i == 4) & (c_i == 3))
  return np.where(well, weights * 1.55, weights)


# (0,2) is sampled as a full-cell bridge from (0,1), not a left-edge pile.
T2_FAINT_CELLS = [
    (0, 3, 8), (0, 4, 8),
    (1, 4, 8),
    (2, 3, 8), (2, 4, 8),
]


def _bridge02_mu_weights(xy, rng):
  """Gentle left→right fade inside (0,2): continues (0,1), does not dump left."""
  c = xy[:, 1]
  fade = 0.38 + 0.62 * np.clip(1.0 - (c - 2.0), 0.0, 1.0)
  return fade * rng.lognormal(0.0, 0.12, size=xy.shape[0])

# (3,4) is a wall; (4,4) is the goal — never paint the goal cell.
T3_EXTRA_CELLS = [
    (3, 3, 30),
    (4, 3, 34),
]

# (3,4) is a wall; (4,4) is the goal.
T4_EXTRA_CELLS = [
    (3, 3, 9),
    (4, 3, 9),
]


def _soft_dots(grid: np.ndarray, walls: np.ndarray, sigma: float = 1.45) -> np.ndarray:
  """Widen each hit into a small Gaussian splat. Does not fill whole cells."""
  from scipy.ndimage import gaussian_filter
  g = np.nan_to_num(np.asarray(grid, dtype=np.float64), nan=0.0)
  out = gaussian_filter(g, sigma=float(sigma), mode='constant')
  h, w = walls.shape
  sub = int(grid.shape[0] // h)
  wall_hi = np.repeat(np.repeat(np.asarray(walls) == 1, sub, axis=0), sub, axis=1)
  return np.where(wall_hi, np.nan, out)


def _sample_bottom_hall_xy(walls: np.ndarray, rng: np.random.Generator) -> np.ndarray:
  """Connected dusty occupancy: col-0 links row 4 to the bottom halls.

  Points are biased toward cell interiors and jittered across free
  neighbors so the cloud does not stamp out clean cell rectangles.
  """
  cells = [
      (4, 0, 70), (4, 1, 96),
      (3, 1, 48),
      (2, 0, 32), (2, 1, 42),
      (0, 0, 28), (0, 1, 32),
  ]
  xy = _sample_cells(walls, rng, cells)
  right10 = _sample_right_of_cell(walls, rng, 1, 0, 34)
  if right10.size:
    xy = np.concatenate([xy, right10], axis=0)
  return xy


def _scene_bottom_halls_wallbehind(seed: int = 19) -> tuple:
  walls = _paper_walls()
  rng = np.random.default_rng(int(seed))
  xy = _sample_bottom_hall_xy(walls, rng)
  weights = _t1_weights(xy, rng)
  xy, weights = _spill_keep_weights(xy, weights, walls, rng)
  sub = 16
  occ = _normalize_free(_bin_xy(xy, walls, sub, None))
  pre = _normalize_free(_bin_xy(xy, walls, sub, weights))
  return walls, occ, pre, GOAL, xy.shape[0], xy, weights


def _mix_occ(core: np.ndarray, extra: np.ndarray, extra_frac: float) -> np.ndarray:
  """Add a small extra histogram onto a core histogram, then renormalize."""
  core_s = float(np.nansum(core))
  extra_s = float(np.nansum(extra))
  if extra_s <= 0:
    return _normalize_free(core)
  scale = (float(extra_frac) * core_s) / extra_s
  mixed = np.where(np.isnan(core), np.nan, np.nan_to_num(core) + scale * np.nan_to_num(extra))
  return _normalize_free(mixed)


def _t2_points(seed: int = 21):
  walls = _paper_walls()
  rng_t1 = np.random.default_rng(19)
  xy1 = _sample_bottom_hall_xy(walls, rng_t1)
  w1 = _t1_weights(xy1, rng_t1)
  rng = np.random.default_rng(int(seed))
  faint = _sample_cells(walls, rng, T2_FAINT_CELLS)
  bridge = _sample_cells(walls, rng, [(0, 2, 40)])
  extras = np.concatenate([faint, bridge], axis=0)
  xy = np.concatenate([xy1, extras], axis=0)
  w2 = _t2_weights(xy, rng)
  return walls, xy1, w1, extras, xy, w2, faint, bridge


def _t2_occ_weights(xy1, w1):
  """Occupancy in the top hall: (0,1) matches (0,0) yellow, no wall-hug cliff."""
  r_i = np.floor(xy1[:, 0]).astype(np.int32)
  c = xy1[:, 1]
  c_i = np.floor(c).astype(np.int32)
  w = np.asarray(w1, dtype=np.float64).copy()
  in_00 = (r_i == 0) & (c_i == 0)
  in_01 = (r_i == 0) & (c_i == 1)
  if np.any(in_00) and np.any(in_01):
    ref = float(np.median(w[in_00]))
    ranks = w[in_01] / (float(np.median(w[in_01])) + 1e-12)
    rise = 1.0 + 0.10 * np.clip(c[in_01] - 1.0, 0.0, 1.0)
    w[in_01] = ref * ranks * rise
  return w


def _scene_t2_from_t1_preimage(seed: int = 21) -> tuple:
  """μ ≈ t1 preimage + (0,1)→(0,2) bridge + faint top-right."""
  walls, xy1, w1, extras, xy, weights, faint, bridge = _t2_points(seed)
  rng = np.random.default_rng(int(seed) + 11)
  sub = 16
  w_occ = _t2_occ_weights(xy1, w1)
  xy1s, w_occs = _spill_keep_weights(xy1, w_occ, walls, rng)
  occ_core = _bin_xy(xy1s, walls, sub, w_occs)
  br_w = _bridge02_mu_weights(bridge, np.random.default_rng(seed + 3))
  br_s, br_ws = _spill_keep_weights(bridge, br_w, walls, np.random.default_rng(seed + 13))
  occ = _mix_occ(occ_core, _bin_xy(br_s, walls, sub, br_ws), extra_frac=0.09)
  ft_s, _ = _spill_keep_weights(faint, None, walls, np.random.default_rng(seed + 15))
  occ = _mix_occ(occ, _bin_xy(ft_s, walls, sub, None), extra_frac=0.04)
  xy_s, w_s = _spill_keep_weights(xy, weights, walls, np.random.default_rng(seed + 17))
  pre = _normalize_free(_bin_xy(xy_s, walls, sub, w_s))
  return walls, occ, pre, GOAL, xy_s.shape[0]


def _t3_points(seed: int = 23):
  walls, _, _, _, xy2, w2, _, _ = _t2_points(21)
  rng = np.random.default_rng(int(seed))
  extra = _sample_cells(walls, rng, T3_EXTRA_CELLS)
  xy = np.concatenate([xy2, extra], axis=0)
  w3 = _t3_weights(xy, rng)
  return walls, xy2, w2, extra, xy, w3


def _scene_t3_from_t2_preimage(seed: int = 23) -> tuple:
  """μ ≈ t2 preimage + corridor to the goal. Preimage reddest at the well."""
  walls, xy2, w2, extra, xy, weights = _t3_points(seed)
  rng = np.random.default_rng(int(seed) + 11)
  sub = 16
  xy2s, w2s = _spill_keep_weights(xy2, w2, walls, rng)
  occ_core = _bin_xy(xy2s, walls, sub, w2s)
  ex_s, _ = _spill_keep_weights(extra, None, walls, np.random.default_rng(seed + 13))
  occ_extra = _bin_xy(ex_s, walls, sub, None)
  occ = _mix_occ(occ_core, occ_extra, extra_frac=0.12)
  xy_s, w_s = _spill_keep_weights(xy, weights, walls, np.random.default_rng(seed + 17))
  pre = _normalize_free(_bin_xy(xy_s, walls, sub, w_s))
  return walls, occ, pre, GOAL, xy_s.shape[0]


def _scene_t4_from_t3_preimage(seed: int = 25) -> tuple:
  """μ ≈ t3 preimage + faint (3,5)/(4,5)/(4,6)/(5,6). Preimage on those cells."""
  walls, _, _, _, xy3, w3 = _t3_points(23)
  rng = np.random.default_rng(int(seed))
  extra = _sample_cells(walls, rng, T4_EXTRA_CELLS)
  xy = np.concatenate([xy3, extra], axis=0)
  weights = _t4_weights(xy, rng)
  sub = 16
  occ_core = _bin_xy(xy3, walls, sub, w3)
  occ_extra = _bin_xy(extra, walls, sub, None)
  occ = _mix_occ(occ_core, occ_extra, extra_frac=0.055)
  pre = _normalize_free(_bin_xy(xy, walls, sub, weights))
  return walls, occ, pre, GOAL, xy.shape[0]


def _write_scene(out_path, walls, occ, pre, goal, n, title):
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  save_occupancy_preimage_figure(
      out_path, walls, occ, pre, goal,
      title=title,
      subtitle=f'n={n}  each panel sums to 1',
      shared_scale=True)
  np.savez_compressed(
      os.path.splitext(out_path)[0] + '.npz',
      occ=occ, pre=pre, goal=goal, walls=walls)
  print(f'wrote {out_path}')
  print(f'sum μ(s)={float(np.nansum(occ)):.6f}  '
        f'sum preimage={float(np.nansum(pre)):.6f}')


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--scene', choices=('t1', 't2', 't3', 't4', 'both'), default='both')
  args = parser.parse_args()
  paper_dir = os.path.join(ROOT, 'paper material', 'paper plots')
  log_dir = os.path.join(ROOT, 'logs', 'paper_maze_concept')
  figs_dir = os.path.join(ROOT, 'figs', 'maze_concept')
  if args.scene in ('t1', 'both'):
    walls, occ, pre, goal, n, _, _ = _scene_bottom_halls_wallbehind()
    for out in (
        os.path.join(log_dir, 'maze_concept_t1_early.png'),
        os.path.join(paper_dir, 'maze_concept_t1_early.png'),
        os.path.join(figs_dir, 'maze_concept_t1_early.png'),
        os.path.join(log_dir, 'bottom_halls_wallbehind_goal.png'),
    ):
      _write_scene(out, walls, occ, pre, goal, n,
                   'schematic  t1  (early, before goal)')
  if args.scene in ('t2', 'both'):
    walls, occ, pre, goal, n = _scene_t2_from_t1_preimage()
    for out in (
        os.path.join(log_dir, 'maze_concept_t2_mid.png'),
        os.path.join(paper_dir, 'maze_concept_t2_mid.png'),
        os.path.join(figs_dir, 'maze_concept_t2_mid.png'),
    ):
      _write_scene(out, walls, occ, pre, goal, n,
                   'schematic  t2  (μ ≈ t1 preimage + faint top)')
  if args.scene in ('t3', 'both'):
    walls, occ, pre, goal, n = _scene_t3_from_t2_preimage()
    for out in (
        os.path.join(log_dir, 'maze_concept_t3_late.png'),
        os.path.join(paper_dir, 'maze_concept_t3_late.png'),
        os.path.join(figs_dir, 'maze_concept_t3_late.png'),
    ):
      _write_scene(out, walls, occ, pre, goal, n,
                   'schematic  t3  (μ ≈ t2 preimage → goal)')
  if args.scene in ('t4', 'both'):
    walls, occ, pre, goal, n = _scene_t4_from_t3_preimage()
    for out in (
        os.path.join(log_dir, 'maze_concept_t4_near.png'),
        os.path.join(paper_dir, 'maze_concept_t4_near.png'),
        os.path.join(figs_dir, 'maze_concept_t4_near.png'),
    ):
      _write_scene(out, walls, occ, pre, goal, n,
                   'schematic  t4  (μ ≈ t3 preimage + faint right well)')


if __name__ == '__main__':
  main()
