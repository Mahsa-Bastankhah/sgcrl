"""JAX rasterizer: BuilderBench compact state/goal xyz → 64×64 RGB.

Used only when ``ppo_bb_pixel_obs`` is on. Env obs and replay stay compact
vectors; networks call these helpers at apply time (inside jit).

Isometric hard cubes (3 faces). ``u = y + k x``, ``v = z + k_d x`` so
y is left-right, z is up, and **x recedes into the page** (start vs goal
depth). Top / front / side faces; 1 px linear edges keep ∇_s log p.
Camera bounds come from ``num_cubes`` so state and goal share the window.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from envs.builderbench_utils import (
    is_builderbench_creative_env,
    load_task_cube_mask,
    parse_sgcrl_builderbench_env_name,
)

IMAGE_HW = 64
IMAGE_C = 3
PIXEL_EMBED_DIM = 256

# RGB only; matches builderbench.constants._CUSTOM_COLORS.
CUBE_RGB = np.asarray(
    [
        [1.00, 0.49, 0.43],
        [0.00, 0.58, 0.55],
        [0.93, 0.78, 0.28],
        [0.38, 0.31, 0.86],
        [0.98, 0.67, 0.78],
        [0.13, 0.55, 0.13],
        [0.85, 0.37, 0.00],
        [0.29, 0.63, 0.92],
        [0.56, 0.27, 0.68],
        [0.44, 0.50, 0.56],
    ],
    dtype=np.float32,
)

_BG = np.asarray([0.12, 0.12, 0.14], dtype=np.float32)
# Sprite half-extent (m). Physics cubes are 2 cm half; draw a bit larger.
_CUBE_HALF = 0.022
# Oblique: x shears right (_X_TO_U) and up (_X_TO_V = depth into the page).
_X_TO_U = 0.35
_X_TO_V = 0.55
_TABLE_X_MAX = 0.30
_TABLE_X_MIN = 0.08
_EDGE_PX = 1.0
# Face shades: top lit, front, darker side (depth).
_SHADE_TOP = 1.05
_SHADE_FRONT = 0.88
_SHADE_SIDE = 0.62


def resolve_bb_pixel_layout(env_name: str) -> Tuple[int, Tuple[int, ...]]:
  """``(num_cubes, goal_color_indices)`` for a ``builderbench_creative_*`` env."""
  if not is_builderbench_creative_env(env_name):
    raise ValueError(
        'ppo_bb_pixel_obs requires a builderbench_creative_* env, '
        f'got {env_name!r}')
  _, num_cubes, task_index = parse_sgcrl_builderbench_env_name(env_name)
  mask = load_task_cube_mask(num_cubes, task_index)
  color_idx = tuple(int(i) for i in np.flatnonzero(mask))
  if not color_idx:
    raise ValueError(f'{env_name}: empty task cube mask')
  return int(num_cubes), color_idx


def pixel_network_kwargs(env_name: str, enabled: bool) -> dict:
  """Kwargs for ``make_networks`` / ``make_nf_density_networks``."""
  if not enabled:
    return dict(
        bb_pixel_obs=False,
        bb_num_cubes=0,
        bb_goal_color_indices=(),
    )
  num_cubes, color_idx = resolve_bb_pixel_layout(env_name)
  return dict(
      bb_pixel_obs=True,
      bb_num_cubes=num_cubes,
      bb_goal_color_indices=color_idx,
  )


def cube_colors(indices: Sequence[int]) -> jnp.ndarray:
  palette = jnp.asarray(CUBE_RGB, dtype=jnp.float32)
  idx = jnp.asarray(tuple(int(i) % int(palette.shape[0]) for i in indices),
                    dtype=jnp.int32)
  return palette[idx]


def split_bb_state(
    state: jnp.ndarray,
    num_cubes: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
  """``state → ((B, N, 3) xyz, (B,) select or None)``.

  Accepts PD-filtered ``N*3+1`` or full ``N*13+1`` (xyz / quat / vel / select).
  """
  nc = int(num_cubes)
  pos = state[..., : 3 * nc].reshape(state.shape[:-1] + (nc, 3))
  d = int(state.shape[-1])
  if d in (nc * 3 + 1, nc * 13 + 1):
    return pos, state[..., -1]
  return pos, None


def _view_bounds(num_cubes: int) -> Tuple[float, float, float, float]:
  """Isotropic window covering y-lanes, stack z, and x-depth shear."""
  n = max(int(num_cubes), 1)
  h = float(_CUBE_HALF)
  y_half = 0.04 * (n - 1) + 0.04
  u_min = -(y_half + h) + _X_TO_U * (_TABLE_X_MIN - h)
  u_max = (y_half + h) + _X_TO_U * (_TABLE_X_MAX + h)
  v_min = -0.02 + _X_TO_V * (_TABLE_X_MIN - h)
  v_max = 0.04 * n + 0.05 + _X_TO_V * (_TABLE_X_MAX + h)
  span = max(u_max - u_min, v_max - v_min)
  u_mid = 0.5 * (u_min + u_max)
  u_min, u_max = u_mid - 0.5 * span, u_mid + 0.5 * span
  v_max = v_min + span
  return float(u_min), float(u_max), float(v_min), float(v_max)


def _project_uv(
    x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """``u = y + 0.35 x`` (lanes), ``v = z + 0.55 x`` (up + depth)."""
  u = y + jnp.float32(_X_TO_U) * x
  v = z + jnp.float32(_X_TO_V) * x
  return u, v


def _uv_to_pix(
    u: jnp.ndarray,
    v: jnp.ndarray,
    u_min: jnp.ndarray,
    u_span: jnp.ndarray,
    v_min: jnp.ndarray,
    v_span: jnp.ndarray,
    hw_f: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  px = (u - u_min) / u_span * hw_f
  py = (jnp.float32(1.0) - (v - v_min) / v_span) * hw_f
  return px, py


def _para_alpha(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    p0x: jnp.ndarray,
    p0y: jnp.ndarray,
    e1x: jnp.ndarray,
    e1y: jnp.ndarray,
    e2x: jnp.ndarray,
    e2y: jnp.ndarray,
    edge_px: jnp.ndarray,
) -> jnp.ndarray:
  """Parallelogram occupancy ``(B, H, W)`` with a linear edge."""
  qx = grid_x[None, :, :] - p0x[:, None, None]
  qy = grid_y[None, :, :] - p0y[:, None, None]
  det = e1x * e2y - e1y * e2x
  det = jnp.where(jnp.abs(det) < jnp.float32(1e-6), jnp.ones_like(det), det)
  w1 = (qx * e2y[:, None, None] - qy * e2x[:, None, None]) / det[:, None, None]
  w2 = (e1x[:, None, None] * qy - e1y[:, None, None] * qx) / det[:, None, None]
  len1 = jnp.hypot(e1x, e1y)
  len2 = jnp.hypot(e2x, e2y)
  d1 = jnp.minimum(w1, jnp.float32(1.0) - w1) * len1[:, None, None]
  d2 = jnp.minimum(w2, jnp.float32(1.0) - w2) * len2[:, None, None]
  return jnp.clip(d1 / edge_px, 0.0, 1.0) * jnp.clip(d2 / edge_px, 0.0, 1.0)


def rasterize_cubes(
    pos: jnp.ndarray,
    colors: jnp.ndarray,
    select: Optional[jnp.ndarray] = None,
    resolution: int = IMAGE_HW,
    view_cubes: Optional[int] = None,
) -> jnp.ndarray:
  """Isometric cubes (top/front/side). ``pos (B, N, 3)`` → ``(B, H, W, 3)``."""
  pos = jnp.asarray(pos, dtype=jnp.float32)
  colors = jnp.asarray(colors, dtype=jnp.float32)
  if pos.ndim != 3 or pos.shape[-1] != 3:
    raise ValueError(f'pos must be (B, N, 3), got {pos.shape}')
  n_cubes = int(pos.shape[1])
  hw = int(resolution)
  if colors.shape != (n_cubes, 3):
    raise ValueError(f'colors must be ({n_cubes}, 3), got {colors.shape}')
  n_view = int(view_cubes) if view_cubes is not None else n_cubes
  u_min_f, u_max_f, v_min_f, v_max_f = _view_bounds(n_view)
  u_min = jnp.float32(u_min_f)
  u_span = jnp.float32(u_max_f - u_min_f)
  v_min = jnp.float32(v_min_f)
  v_span = jnp.float32(v_max_f - v_min_f)
  hw_f = jnp.float32(hw - 1)
  h = jnp.float32(_CUBE_HALF)
  edge_px = jnp.maximum(jnp.float32(_EDGE_PX), jnp.float32(1e-3))

  def to_pix(xw, yw, zw):
    uu, vv = _project_uv(xw, yw, zw)
    return _uv_to_pix(uu, vv, u_min, u_span, v_min, v_span, hw_f)

  ys = jnp.arange(hw, dtype=jnp.float32)
  xs = jnp.arange(hw, dtype=jnp.float32)
  grid_y, grid_x = jnp.meshgrid(ys, xs, indexing='ij')

  col = colors[None, :, :]
  if select is not None:
    sel = jnp.asarray(select, dtype=jnp.float32).reshape(-1)
    cont = (sel + 1.0) * 0.5 * jnp.float32(max(n_cubes - 1, 0))
    ids = jnp.arange(n_cubes, dtype=jnp.float32)[None, :]
    highlight = jax.nn.softmax(-jnp.square(ids - cont[:, None]) / 0.15, axis=-1)
    col = jnp.clip(col * (1.0 + 0.45 * highlight[:, :, None]), 0.0, 1.0)

  bg = jnp.asarray(_BG, dtype=jnp.float32)
  rgb = jnp.broadcast_to(bg, pos.shape[:1] + (hw, hw, 3))
  zbuf = jnp.full(pos.shape[:1] + (hw, hw), jnp.float32(-1e6))

  def blit(p0x, p0y, e1x, e1y, e2x, e2y, shade, depth):
    nonlocal rgb, zbuf
    occ = _para_alpha(
        grid_x, grid_y, p0x, p0y, e1x, e1y, e2x, e2y, edge_px)
    depth_img = depth[:, None, None]
    closer = (occ > jnp.float32(1e-4)) & (depth_img >= zbuf)
    a = occ[..., None]
    rgb = jnp.where(closer[..., None], (1.0 - a) * rgb + a * shade, rgb)
    zbuf = jnp.where(closer, depth_img, zbuf)

  # Nearer = smaller x (camera at start, looking toward the goal).
  for i in range(n_cubes):
    x = pos[:, i, 0]
    y = pos[:, i, 1]
    z = pos[:, i, 2]
    shade_base = col[:, i, :][:, None, None, :]
    depth_x = -x * jnp.float32(100.0) + z

    # Front (near x = x-h): axis-aligned in uv. Origin (x-h, y-h, z-h).
    p0x, p0y = to_pix(x - h, y - h, z - h)
    p1x, p1y = to_pix(x - h, y + h, z - h)
    p2x, p2y = to_pix(x - h, y - h, z + h)
    blit(p0x, p0y, p1x - p0x, p1y - p0y, p2x - p0x, p2y - p0y,
         jnp.clip(shade_base * jnp.float32(_SHADE_FRONT), 0.0, 1.0),
         depth_x + jnp.float32(0.01))

    # Side (y = y+h): depth face. Origin (x-h, y+h, z-h).
    s0x, s0y = to_pix(x - h, y + h, z - h)
    s1x, s1y = to_pix(x + h, y + h, z - h)
    s2x, s2y = to_pix(x - h, y + h, z + h)
    blit(s0x, s0y, s1x - s0x, s1y - s0y, s2x - s0x, s2y - s0y,
         jnp.clip(shade_base * jnp.float32(_SHADE_SIDE), 0.0, 1.0),
         depth_x)

    # Top (z = z+h). Origin (x-h, y-h, z+h).
    t0x, t0y = to_pix(x - h, y - h, z + h)
    t1x, t1y = to_pix(x + h, y - h, z + h)
    t2x, t2y = to_pix(x - h, y + h, z + h)
    blit(t0x, t0y, t1x - t0x, t1y - t0y, t2x - t0x, t2y - t0y,
         jnp.clip(shade_base * jnp.float32(_SHADE_TOP), 0.0, 1.0),
         depth_x + jnp.float32(0.02))
  return rgb


def rasterize_bb_state(
    state: jnp.ndarray,
    num_cubes: int,
    resolution: int = IMAGE_HW,
) -> jnp.ndarray:
  pos, select = split_bb_state(state, num_cubes)
  colors = cube_colors(range(int(num_cubes)))
  return rasterize_cubes(
      pos, colors, select=select, resolution=resolution,
      view_cubes=int(num_cubes))


def rasterize_bb_goal(
    goal: jnp.ndarray,
    color_indices: Sequence[int],
    resolution: int = IMAGE_HW,
    num_cubes: Optional[int] = None,
) -> jnp.ndarray:
  idx = tuple(int(i) for i in color_indices)
  n_task = len(idx)
  pos = jnp.asarray(goal, dtype=jnp.float32).reshape(-1, n_task, 3)
  n_view = int(num_cubes) if num_cubes is not None else n_task
  return rasterize_cubes(
      pos, cube_colors(idx), select=None, resolution=resolution,
      view_cubes=n_view)


class BBPixelTorso(hk.Module):
  """Small CNN: ``(B, 64, 64, 3)`` in ``[0, 1]`` → ``(B, 256)``."""

  def __init__(self, embed_dim: int = PIXEL_EMBED_DIM, name: str = 'bb_pixel_torso'):
    super().__init__(name=name)
    self.embed_dim = int(embed_dim)

  def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
    w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    x = image
    x = hk.Conv2D(32, 5, stride=2, padding='SAME', w_init=w_init, name='c0')(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(64, 3, stride=2, padding='SAME', w_init=w_init, name='c1')(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(64, 3, stride=2, padding='SAME', w_init=w_init, name='c2')(x)
    x = jax.nn.relu(x)
    x = hk.Flatten()(x)
    return hk.Linear(self.embed_dim, w_init=w_init, name='out')(x)


def encode_state_goal_images(
    state: jnp.ndarray,
    goal: jnp.ndarray,
    num_cubes: int,
    goal_color_indices: Sequence[int],
    torso_name: str = 'bb_pixel_torso',
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Rasterize then shared CNN. Call from inside a Haiku transform."""
  torso = BBPixelTorso(name=torso_name)
  s_img = rasterize_bb_state(state, num_cubes)
  g_img = rasterize_bb_goal(
      goal, goal_color_indices, num_cubes=num_cubes)
  return torso(s_img), torso(g_img)
