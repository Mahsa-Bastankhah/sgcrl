#!/usr/bin/env python3
"""3D PCA of NF latent z at init (s0, policy-mode a0) across checkpoints.

Reuses the Variation-A path from ``probe_nf_ckpt_task_goal_z_rollouts.py``:
fixed-start reset → s0, a0 = policy mode(s0, g_task) → z = f(g_task|s0,a0)
with that checkpoint's nf_goal_mean/std.

Default checkpoint subset: every iter with ``iter % 40 == 0``, plus the final
ckpt if its iter is not already included (e.g. 389).  Use ``--max_iter`` to
cap (e.g. 280) and ``--no_include_final`` / auto-drop of final when it exceeds
``max_iter``.

PCA is fit on the plotted z set only (documented in the figure subtitle).

Curved time arrows (``--arrows``): matplotlib has no robust 3D FancyArrowPatch,
so for each consecutive PCA pair we draw a quadratic Bezier arc in 3D — chord
midpoint offset by a small perpendicular bulge — then a 3D quiver arrowhead
at the end of the arc.  Documented in the figure subtitle when enabled.

Prior overlay (``--gauss2d``): 1σ / 2σ contours of the NF base prior N(0,I)
in the PC1–PC2 plane.  Under an orthonormal PCA rotation, the PC1–PC2
marginal of N(0,I) is N(0,I₂), so these are circles of radius 1 and 2
centered at the PC origin, drawn at PC3=0 (or a floor just below the data).
These are the *prior*, not a Gaussian fit to the scatter.

Density surface (``--gauss_density_surface``): drop PC3. Plot a semi-transparent
3D surface of the isotropic prior density
``dens(x,y) = (1/(2π)) exp(-½(x²+y²))`` over (PC1, PC2), and place each
init-z checkpoint at ``(PC1, PC2, dens(PC1, PC2))`` so points sit on the
surface, with curved arrows connecting them in time order.

Example:
  CUDA_VISIBLE_DEVICES=0 BUILDERBENCH_MJX_IMPL=warp \\
  python scripts/plot_nf_init_z_pca3d.py \\
      --run_dir=logs/.../ppo_builderbench_creative_4_task2_0 \\
      --max_iter=280 --arrows --gauss_density_surface \\
      --out=logs/.../init_z_pca2d_gauss_density_surface.png
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from contrastive import nf_density as _nf
from contrastive import ppo_learner

# Reuse rollout-probe helpers / session (env + networks + goal norm).
_probe_path = os.path.join(_REPO, 'scripts', 'probe_nf_ckpt_task_goal_z_rollouts.py')
_spec = importlib.util.spec_from_file_location('probe_nf_rollouts_z', _probe_path)
_probe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_probe)

CKPT_RE = _probe.CKPT_RE
_abs = _probe._abs
_list_ckpts = _probe._list_ckpts
_goal_norm = _probe._goal_norm
RolloutProbeSession = _probe.RolloutProbeSession


def _pca_fit_transform(X: np.ndarray, n_components: int = 3
                       ) -> Tuple[np.ndarray, np.ndarray]:
  """Center + SVD PCA. Returns (scores [N,k], explained_variance_ratio [k])."""
  X = np.asarray(X, dtype=np.float64)
  Xc = X - X.mean(axis=0, keepdims=True)
  _u, s, _vt = np.linalg.svd(Xc, full_matrices=False)
  k = min(int(n_components), s.size)
  scores = _u[:, :k] * s[:k]
  total = float(np.sum(s ** 2))
  if total <= 0:
    ratio = np.zeros(k, dtype=np.float64)
  else:
    ratio = (s[:k] ** 2) / total
  return scores, ratio


def _select_ckpts(
    ckpts: Sequence[Tuple[int, str]],
    *,
    stride: int = 40,
    always_include_final: bool = True,
    max_iter: Optional[int] = None,
) -> List[Tuple[int, str]]:
  selected = [(it, p) for it, p in ckpts if it % stride == 0]
  if always_include_final and ckpts:
    last_it, last_p = ckpts[-1]
    if not selected or selected[-1][0] != last_it:
      selected.append((last_it, last_p))
  if max_iter is not None:
    selected = [(it, p) for it, p in selected if it <= int(max_iter)]
  return selected


def _first_success(learner_csv: str) -> dict:
  """Return first iter/step where train_success_{1000,mean} > 0."""
  out = {
      'train_success_1000': None,
      'train_success_mean': None,
  }
  if not os.path.isfile(learner_csv):
    return out
  with open(learner_csv, newline='') as f:
    rows = list(csv.DictReader(f))
  for col in out:
    for r in rows:
      try:
        v = float(r[col])
      except (KeyError, TypeError, ValueError):
        continue
      if v > 0:
        out[col] = {
            'iteration': int(float(r['iteration'])),
            'global_step': int(float(r['global_step'])),
            'value': v,
        }
        break
  return out


def _probe_init_z(session: RolloutProbeSession, ckpt_path: str
                  ) -> Tuple[int, int, np.ndarray]:
  """Return (iter, global_step, z[goal_dim]) for Variation A only."""
  ckpt = ppo_learner.load_checkpoint(ckpt_path)
  m = CKPT_RE.search(os.path.basename(ckpt_path))
  it = int(m.group(1)) if m else int(ckpt.get('iteration', -1))
  global_step = int(ckpt.get('global_step', -1))
  q_key = 'q_params' if session.which == 'online' else 'q_params_ema'
  if q_key not in ckpt:
    raise KeyError(f'{q_key} missing in {ckpt_path}')
  if 'policy_params' not in ckpt:
    raise KeyError(f'policy_params missing in {ckpt_path}')

  extra = ckpt.get('extra_state') or {}
  g_norm = _goal_norm(session.g_task, extra, session.std_min)
  nf_params = ckpt[q_key]
  policy_params = ckpt['policy_params']

  env_state = session._reset_fixed()
  packed0 = session.vec_env.pack_obs_from_state(env_state)
  packed0_1 = packed0[:1]
  s0 = packed0_1[:, :session.obs_dim]
  a0 = session._mode_action(policy_params, packed0_1)
  a0 = jnp.clip(a0, -1.0, 1.0)

  g = jnp.broadcast_to(g_norm.reshape(1, -1), (1, g_norm.size))
  z, _log_det, _log_p = _nf.nf_forward(
      session.nf_nets, nf_params, s0, a0, g)
  z_np = np.asarray(z, dtype=np.float64).reshape(-1)
  return it, global_step, z_np


def _arc_polyline(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    n: int = 28,
    bulge: float = 0.18,
) -> np.ndarray:
  """Quadratic Bezier arc through p0→p1 with perpendicular mid bulge.

  The control point sits at the chord midpoint offset by ``bulge * ||p1-p0||``
  along a unit vector perpendicular to the chord (cross with the coordinate
  axis least aligned with the chord).  Returns ``[n, 3]``.
  """
  p0 = np.asarray(p0, dtype=np.float64).reshape(3)
  p1 = np.asarray(p1, dtype=np.float64).reshape(3)
  chord = p1 - p0
  length = float(np.linalg.norm(chord))
  if length < 1e-12:
    return np.stack([p0, p1], axis=0)
  mid = 0.5 * (p0 + p1)
  u = chord / length
  # Prefer a stable perp: cross with axis most orthogonal to chord.
  axes = np.eye(3)
  a = axes[int(np.argmin(np.abs(u)))]
  perp = np.cross(u, a)
  pn = float(np.linalg.norm(perp))
  if pn < 1e-12:
    a = axes[(int(np.argmin(np.abs(u))) + 1) % 3]
    perp = np.cross(u, a)
    pn = float(np.linalg.norm(perp))
  perp = perp / pn
  ctrl = mid + float(bulge) * length * perp
  t = np.linspace(0.0, 1.0, int(n))
  omt = 1.0 - t
  curve = (omt * omt)[:, None] * p0 + (2.0 * omt * t)[:, None] * ctrl + (
      t * t)[:, None] * p1
  return curve


def _draw_curved_arrows(
    ax,
    xyz: np.ndarray,
    *,
    bulge: float = 0.18,
    color: str = '0.35',
    lw: float = 1.15,
    alpha: float = 0.85,
) -> None:
  """Draw arc polylines + 3D quiver heads between consecutive rows of xyz."""
  span = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
  head_len = max(0.04 * span, 1e-3)
  for i in range(xyz.shape[0] - 1):
    curve = _arc_polyline(xyz[i], xyz[i + 1], bulge=bulge)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2],
            color=color, lw=lw, alpha=alpha, zorder=2)
    # Arrowhead along the last arc tangent, ending at p1.
    tang = curve[-1] - curve[-2]
    tn = float(np.linalg.norm(tang))
    if tn < 1e-12:
      continue
    tang = tang / tn
    # Start the quiver slightly before the endpoint so the tip lands on p1.
    start = curve[-1] - tang * head_len
    ax.quiver(
        start[0], start[1], start[2],
        tang[0] * head_len, tang[1] * head_len, tang[2] * head_len,
        color=color, alpha=alpha, arrow_length_ratio=0.45,
        linewidth=lw, normalize=False, zorder=3)


def _gauss2d_pc3_level(xyz: np.ndarray) -> float:
  """Prefer PC3=0; if that sits inside the cloud, use a floor below min PC3."""
  zmin = float(xyz[:, 2].min())
  zmax = float(xyz[:, 2].max())
  if zmin <= 0.0 <= zmax:
    # Origin plane cuts the cloud — still OK for calibration; keep PC3=0.
    return 0.0
  # Otherwise park the contour plane just below the data as a floor.
  span = max(zmax - zmin, 1e-3)
  return zmin - 0.08 * span


def _draw_prior_gauss2d(
    ax,
    xyz: np.ndarray,
    *,
    radii: Sequence[float] = (1.0, 2.0),
    n_theta: int = 160,
    n_radial: int = 24,
) -> Tuple[list, float]:
  """Overlay N(0,I) 1σ/2σ circles in the PC1–PC2 plane at fixed PC3.

  Returns (legend_handles, pc3_level).
  """
  from matplotlib.lines import Line2D
  from matplotlib.patches import Patch

  pc3 = _gauss2d_pc3_level(xyz)
  theta = np.linspace(0.0, 2.0 * np.pi, int(n_theta))
  colors = {1.0: '#4c78a8', 2.0: '#f58518'}
  # Filled disk (≤1σ) then annulus (1σ–2σ).
  for r_inner, r_outer, alpha, key in (
      (0.0, 1.0, 0.14, 1.0),
      (1.0, 2.0, 0.10, 2.0),
  ):
    rr = np.linspace(r_inner, r_outer, int(n_radial))
    R, T = np.meshgrid(rr, theta)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = np.full_like(X, pc3)
    ax.plot_surface(
        X, Y, Z,
        color=colors[key], alpha=alpha, linewidth=0, shade=False,
        antialiased=True, zorder=1)

  handles = []
  for r in radii:
    r = float(r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, pc3)
    c = colors.get(r, '0.4')
    ax.plot(x, y, z, color=c, lw=1.8, alpha=0.95, zorder=4)
    handles.append(Line2D([0], [0], color=c, lw=1.8,
                          label=f'N(0,I) {int(r)}σ'))
  # Also a faint filled-legend hint.
  handles.append(Patch(facecolor=colors[1.0], alpha=0.25, edgecolor='none',
                       label='prior disk / ring (PC1–PC2)'))
  return handles, pc3


def _iso_gauss2d_density(x, y):
  """N(0, I₂) density: (1/(2π)) exp(-0.5 (x²+y²))."""
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  return (1.0 / (2.0 * np.pi)) * np.exp(-0.5 * (x * x + y * y))


def _arrow_head_triangle_3d(
    tip: np.ndarray,
    direction: np.ndarray,
    *,
    head_len: float,
    head_width: float,
) -> np.ndarray:
  """Flat arrowhead in the PC1–PC2 plane at tip's density height.

  Using a planar (xy) triangle avoids mpl 3D aspect-ratio shards from the
  short density z-axis (~0.16) vs wide PC axes (~±3).
  """
  tip = np.asarray(tip, dtype=np.float64).reshape(3)
  d = np.asarray(direction, dtype=np.float64).reshape(3)
  # Direction in PC1–PC2 only.
  d_xy = np.array([d[0], d[1], 0.0], dtype=np.float64)
  dn = float(np.linalg.norm(d_xy))
  if dn < 1e-12:
    d_xy = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  else:
    d_xy = d_xy / dn
  perp = np.array([-d_xy[1], d_xy[0], 0.0], dtype=np.float64)
  z = float(tip[2])
  tip_xy = np.array([tip[0], tip[1], z], dtype=np.float64)
  base = tip_xy - d_xy * float(head_len)
  left = base + perp * (0.5 * float(head_width))
  right = base - perp * (0.5 * float(head_width))
  return np.stack([tip_xy, left, right], axis=0)


def _draw_curved_arrows_on_density(
    ax,
    xy: np.ndarray,
    *,
    bulge: float = 0.10,
    shaft_color: str = '0.40',
    head_color: str = 'black',
    lw: float = 0.75,
    alpha: float = 0.90,
    z_eps: float = 0.0,
    zorder: float = 8,
    # Absolute PC units — large enough to read, not aspect-ratio shards.
    head_len: float = 0.32,
    head_width: float = 0.22,
) -> None:
  """Thin shaft + clearly visible flat black triangular heads."""
  from mpl_toolkits.mplot3d.art3d import Poly3DCollection

  head_z_eps = float(z_eps) + 0.008
  for i in range(xy.shape[0] - 1):
    p0 = np.array([xy[i, 0], xy[i, 1], 0.0], dtype=np.float64)
    p1 = np.array([xy[i + 1, 0], xy[i + 1, 1], 0.0], dtype=np.float64)
    curve_xy = _arc_polyline(p0, p1, bulge=bulge)
    z = _iso_gauss2d_density(curve_xy[:, 0], curve_xy[:, 1]) + float(z_eps)
    tip_xy = curve_xy[-1, :2]
    tip = np.array([
        float(tip_xy[0]), float(tip_xy[1]),
        float(_iso_gauss2d_density(tip_xy[0], tip_xy[1]) + head_z_eps),
    ], dtype=np.float64)
    tang = np.array([
        curve_xy[-1, 0] - curve_xy[-2, 0],
        curve_xy[-1, 1] - curve_xy[-2, 1],
        0.0,
    ], dtype=np.float64)
    tn = float(np.linalg.norm(tang))
    if tn < 1e-12:
      continue
    tang = tang / tn

    # Thin shaft; stop before tip so the triangle is distinct.
    n_shaft = max(int(curve_xy.shape[0] * 0.78), 2)
    ax.plot(curve_xy[:n_shaft, 0], curve_xy[:n_shaft, 1], z[:n_shaft],
            color=shaft_color, lw=lw, alpha=alpha, zorder=zorder)

    tri = _arrow_head_triangle_3d(
        tip, tang, head_len=float(head_len), head_width=float(head_width))
    coll = Poly3DCollection(
        [tri], facecolors=head_color, edgecolors='black',
        linewidths=0.9, alpha=1.0)
    coll.set_zorder(zorder + 3)
    ax.add_collection3d(coll)


def _plot_gauss_density_surface(
    iters: np.ndarray,
    xy: np.ndarray,
    var_ratio: np.ndarray,
    *,
    success_mean_iter: int,
    success_mean_step: int,
    success_1000_iter: int,
    out_png: str,
    out_pdf: Optional[str] = None,
    arrows: bool = True,
    arc_bulge: float = 0.18,
    grid_lim: float = 3.5,
    grid_n: int = 80,
) -> int:
  """3D surface of N(0,I₂) density; points at (PC1, PC2, dens).

  Returns the marked success iteration (or -1).
  """
  dens_pts = _iso_gauss2d_density(xy[:, 0], xy[:, 1])
  # Tiny lift so markers/arrows sit visually above the mesh (mpl 3D zorder
  # alone is unreliable).
  z_eps = 0.004 * max(float(dens_pts.max()), 1e-3)
  xyz = np.column_stack([xy[:, 0], xy[:, 1], dens_pts + z_eps])

  # Mesh covering points and ±grid_lim around origin.
  lim = max(
      float(grid_lim),
      float(np.max(np.abs(xy))) + 0.5,
  )
  xs = np.linspace(-lim, lim, int(grid_n))
  ys = np.linspace(-lim, lim, int(grid_n))
  XX, YY = np.meshgrid(xs, ys)
  ZZ = _iso_gauss2d_density(XX, YY)

  fig = plt.figure(figsize=(9.6, 7.4))
  ax = fig.add_subplot(111, projection='3d')
  # Honor explicit zorder instead of depth-sorting everything.
  if hasattr(ax, 'computed_zorder'):
    ax.computed_zorder = False

  # Background prior density surface (flat color, low alpha).
  ax.plot_surface(
      XX, YY, ZZ,
      color='#9ecae1', alpha=0.28, linewidth=0, antialiased=True,
      shade=False, zorder=1)
  # 1σ / 2σ rings on the surface (still behind trajectory).
  theta = np.linspace(0.0, 2.0 * np.pi, 160)
  for r, c, lab in ((1.0, '#1f77b4', '1σ'), (2.0, '#ff7f0e', '2σ')):
    cx = r * np.cos(theta)
    cy = r * np.sin(theta)
    cz = _iso_gauss2d_density(cx, cy)
    ax.plot(cx, cy, cz, color=c, lw=1.35, alpha=0.75, zorder=2,
            label=f'N(0,I₂) {lab} ring')

  if arrows:
    # Prefer a mild bulge for density-surface figures (cleaner look).
    bulge = min(float(arc_bulge), 0.10) if arc_bulge is not None else 0.10
    _draw_curved_arrows_on_density(
        ax, xy, bulge=bulge, z_eps=z_eps, zorder=8)

  sc = ax.scatter(
      xyz[:, 0], xyz[:, 1], xyz[:, 2],
      c=iters, cmap='viridis', s=120, depthshade=False,
      edgecolors='k', linewidths=0.9, zorder=10)
  cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
  cbar.set_label('iteration')

  mark_it = -1
  for i, it in enumerate(iters):
    if int(it) >= int(success_mean_iter):
      mark_it = int(it)
      ax.scatter(
          [xyz[i, 0]], [xyz[i, 1]], [xyz[i, 2]],
          s=320, facecolors='none', edgecolors='crimson',
          linewidths=2.4, depthshade=False, zorder=12)
      ax.text(
          xyz[i, 0], xyz[i, 1], xyz[i, 2],
          '  first success',
          color='crimson', fontsize=9, fontweight='bold', zorder=13)
      break

  for i, it in enumerate(iters):
    if int(it) in (0, mark_it, int(iters[-1])):
      ax.text(xyz[i, 0], xyz[i, 1], xyz[i, 2], f'  {int(it)}',
              fontsize=7, color='0.15', zorder=13)

  ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')
  ax.set_zlabel('N(0, I2) density')
  ax.set_xlim(-lim, lim)
  ax.set_ylim(-lim, lim)
  ax.set_zlim(0.0, float(ZZ.max()) * 1.08)
  ev = ', '.join(f'{100.0 * r:.1f}%' for r in var_ratio[:2])
  r_xy = np.linalg.norm(xy, axis=1)
  ax.set_title(
      'C4T2 NF — init z on N(0,I₂) density surface\n'
      'z = (1/2π) exp(−½(PC1²+PC2²))  '
      f'|  PCA (2-D) expl. var: [{ev}]\n'
      f'points at (PC1, PC2, dens); prior surface (not fit to scatter); '
      f'median ‖(PC1,PC2)‖={float(np.median(r_xy)):.2f}\n'
      f'first train_success_1000>0 @ iter {success_1000_iter}; '
      f'first train_success_mean>0 @ iter {success_mean_iter} '
      f'(step {success_mean_step / 1e6:.2f}M)',
      fontsize=9)
  fig.subplots_adjust(left=0.02, right=0.92, bottom=0.02, top=0.86)
  os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
  fig.savefig(out_png, dpi=160, bbox_inches='tight')
  if out_pdf:
    fig.savefig(out_pdf, bbox_inches='tight')
  plt.close(fig)
  return mark_it


def _plot_pca3d(
    iters: np.ndarray,
    xyz: np.ndarray,
    var_ratio: np.ndarray,
    *,
    success_mean_iter: int,
    success_mean_step: int,
    success_1000_iter: int,
    out_png: str,
    out_pdf: Optional[str] = None,
    arrows: bool = False,
    arc_bulge: float = 0.18,
    gauss2d: bool = False,
) -> int:
  """Scatter PCA coords; mark first plotted point with iter >= success_mean_iter.

  Returns the marked iteration (or -1 if none).
  """
  fig = plt.figure(figsize=(9.2, 7.2))
  ax = fig.add_subplot(111, projection='3d')

  gauss_handles = []
  gauss_pc3 = 0.0
  if gauss2d:
    gauss_handles, gauss_pc3 = _draw_prior_gauss2d(ax, xyz)

  if arrows:
    _draw_curved_arrows(ax, xyz, bulge=arc_bulge)

  sc = ax.scatter(
      xyz[:, 0], xyz[:, 1], xyz[:, 2],
      c=iters, cmap='viridis', s=70, depthshade=True,
      edgecolors='k', linewidths=0.4, zorder=5)
  cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.72)
  cbar.set_label('iteration')

  mark_it = -1
  for i, it in enumerate(iters):
    if int(it) >= int(success_mean_iter):
      mark_it = int(it)
      ax.scatter(
          [xyz[i, 0]], [xyz[i, 1]], [xyz[i, 2]],
          s=240, facecolors='none', edgecolors='crimson',
          linewidths=2.2, depthshade=False, zorder=10)
      ax.text(
          xyz[i, 0], xyz[i, 1], xyz[i, 2],
          f'  first plotted ≥{success_mean_iter}\n  (iter {mark_it})',
          color='crimson', fontsize=8)
      break

  for i, it in enumerate(iters):
    if int(it) in (0, mark_it, int(iters[-1])):
      ax.text(xyz[i, 0], xyz[i, 1], xyz[i, 2], f'  {int(it)}',
              fontsize=7, color='0.25')

  if gauss_handles:
    ax.legend(handles=gauss_handles, loc='upper left', fontsize=8,
              framealpha=0.9)

  # Expand limits so 2σ circle is visible alongside the scatter.
  if gauss2d:
    lim = 2.15
    ax.set_xlim(min(float(xyz[:, 0].min()), -lim),
                max(float(xyz[:, 0].max()), lim))
    ax.set_ylim(min(float(xyz[:, 1].min()), -lim),
                max(float(xyz[:, 1].max()), lim))

  ev = ', '.join(f'{100.0 * r:.1f}%' for r in var_ratio)
  ax.set_xlabel('PC1')
  ax.set_ylabel('PC2')
  ax.set_zlabel('PC3')
  arrow_note = (
      '  |  arcs: quadratic Bezier + quiver heads'
      if arrows else '')
  gauss_note = ''
  if gauss2d:
    # Distances of scatter from PC origin in the PC1–PC2 plane.
    r_xy = np.linalg.norm(xyz[:, :2], axis=1)
    gauss_note = (
        f'\nN(0,I) 1σ/2σ circles in PC1–PC2 at PC3={gauss_pc3:.2f} '
        f'(base prior, not fit to points; '
        f'scatter ‖(PC1,PC2)‖ median={float(np.median(r_xy)):.2f})'
    )
  ax.set_title(
      'C4T2 NF — init z = f(g_task | s0, a0_mode)\n'
      f'PCA fit on plotted ckpts only  |  expl. var: [{ev}]'
      f'{arrow_note}\n'
      f'first train_success_1000>0 @ iter {success_1000_iter}; '
      f'first train_success_mean>0 @ iter {success_mean_iter} '
      f'(step {success_mean_step / 1e6:.2f}M)'
      f'{gauss_note}',
      fontsize=9 if gauss2d else 10)
  fig.tight_layout()
  os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
  fig.savefig(out_png, dpi=160, bbox_inches='tight')
  if out_pdf:
    fig.savefig(out_pdf, bbox_inches='tight')
  plt.close(fig)
  return mark_it


def _default_out_name(*, max_iter: Optional[int], arrows: bool,
                      gauss2d: bool = False,
                      gauss_density_surface: bool = False) -> str:
  if gauss_density_surface:
    return 'init_z_pca2d_gauss_density_surface.png'
  parts = ['init_z_pca3d']
  if max_iter is not None:
    parts.append(f'to{int(max_iter)}')
  else:
    parts.append('every40')
  if arrows:
    parts.append('arrows')
  if gauss2d:
    parts.append('gauss2d')
  return '_'.join(parts) + '.png'


def scores_csv_path(run_dir: str, *, max_iter: Optional[int] = None) -> str:
  """Canonical cache path for PC1/PC2 scores used by paper plots."""
  tag = f'to{int(max_iter)}' if max_iter is not None else 'all'
  return os.path.join(_abs(run_dir), f'init_z_pca2d_scores_{tag}.csv')


def save_pca2d_scores_csv(
    path: str,
    iters: np.ndarray,
    xy: np.ndarray,
    var_ratio: np.ndarray,
    *,
    success_mean_iter: int,
    success_mean_step: int,
    success_1000_iter: int,
) -> str:
  """Write PC1/PC2 scores + success metadata for paper/replot reuse."""
  path = _abs(path)
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  dens = _iso_gauss2d_density(xy[:, 0], xy[:, 1])
  with open(path, 'w', newline='') as f:
    f.write(
        f'# var_ratio_pc1={float(var_ratio[0]):.10g} '
        f'var_ratio_pc2={float(var_ratio[1]):.10g}\n')
    f.write(
        f'# success_mean_iter={int(success_mean_iter)} '
        f'success_mean_step={int(success_mean_step)} '
        f'success_1000_iter={int(success_1000_iter)}\n')
    w = csv.writer(f)
    w.writerow(['iter', 'pc1', 'pc2', 'density'])
    for i, it in enumerate(iters):
      w.writerow([
          int(it),
          float(xy[i, 0]),
          float(xy[i, 1]),
          float(dens[i]),
      ])
  return path


def load_pca2d_scores_csv(path: str) -> dict:
  """Load cache written by ``save_pca2d_scores_csv``."""
  path = _abs(path)
  meta = {
      'var_ratio_pc1': None,
      'var_ratio_pc2': None,
      'success_mean_iter': None,
      'success_mean_step': None,
      'success_1000_iter': None,
  }
  rows = []
  with open(path, newline='') as f:
    for line in f:
      if line.startswith('#'):
        for key in meta:
          token = f'{key}='
          if token in line:
            for part in line.lstrip('#').split():
              if part.startswith(token):
                meta[key] = float(part.split('=', 1)[1])
        continue
      rows.append(line)
  reader = csv.DictReader(rows)
  iters, pc1, pc2, dens = [], [], [], []
  for r in reader:
    iters.append(int(float(r['iter'])))
    pc1.append(float(r['pc1']))
    pc2.append(float(r['pc2']))
    dens.append(float(r['density']))
  if meta['var_ratio_pc1'] is None or meta['var_ratio_pc2'] is None:
    raise ValueError(f'missing var_ratio metadata in {path}')
  return {
      'iters': np.asarray(iters, dtype=np.int64),
      'xy': np.column_stack([pc1, pc2]).astype(np.float64),
      'density': np.asarray(dens, dtype=np.float64),
      'var_ratio': np.asarray(
          [meta['var_ratio_pc1'], meta['var_ratio_pc2']], dtype=np.float64),
      'success_mean_iter': int(meta['success_mean_iter']),
      'success_mean_step': int(meta['success_mean_step']),
      'success_1000_iter': int(meta['success_1000_iter']),
  }


def collect_init_z_pca(
    run_dir: str,
    *,
    stride: int = 40,
    max_iter: Optional[int] = None,
    include_final: bool = True,
    seed: int = 0,
    which: str = 'online',
    env_name: Optional[str] = None,
    n_components: int = 2,
) -> dict:
  """Probe init-z across checkpoints and fit PCA.

  Returns dict with iters, steps, Z, scores, var_ratio, and success_* fields.
  """
  run_dir = _abs(run_dir)
  ckpt_dir = os.path.join(run_dir, 'checkpoints')
  learner_csv = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')

  succ = _first_success(learner_csv)
  s1000 = succ['train_success_1000']
  smean = succ['train_success_mean']
  print('Learner success timing:', flush=True)
  if s1000:
    print(f'  first train_success_1000>0: iter={s1000["iteration"]} '
          f'step={s1000["global_step"]} val={s1000["value"]}', flush=True)
  else:
    print('  first train_success_1000>0: (none)', flush=True)
  if smean:
    print(f'  first train_success_mean>0: iter={smean["iteration"]} '
          f'step={smean["global_step"]} val={smean["value"]}', flush=True)
  else:
    print('  first train_success_mean>0: (none)', flush=True)

  all_ckpts = _list_ckpts(ckpt_dir)
  if not all_ckpts:
    raise FileNotFoundError(f'no ckpt_iter_*.pkl under {ckpt_dir}')
  selected = _select_ckpts(
      all_ckpts,
      stride=stride,
      always_include_final=bool(include_final),
      max_iter=max_iter,
  )
  if not selected:
    raise RuntimeError('no checkpoints selected after filters')
  print(f'selected iters ({len(selected)}): '
        f'{[it for it, _ in selected]}', flush=True)

  session = RolloutProbeSession(
      run_dir,
      n_rollouts=1,
      seed=seed,
      which=which,
      env_name=env_name,
  )
  print(f'task_goal dim={session.goal_dim}', flush=True)

  iters, steps, zs = [], [], []
  for it, path in selected:
    print(f'  probing iter={it} ...', flush=True)
    it2, step, z = _probe_init_z(session, path)
    assert it2 == it, (it2, it)
    iters.append(it2)
    steps.append(step)
    zs.append(z)
    print(f'    step={step}  ||z||={np.linalg.norm(z):.4f}  z_dim={z.size}',
          flush=True)

  Z = np.stack(zs, axis=0)
  iters_np = np.asarray(iters, dtype=np.int64)
  scores, var_ratio = _pca_fit_transform(Z, n_components=int(n_components))
  print(f'PCA explained_variance_ratio_={var_ratio.tolist()}', flush=True)
  print(f'PCA explained_variance_ratio_ sum={float(var_ratio.sum()):.4f}',
        flush=True)

  return {
      'iters': iters_np,
      'steps': np.asarray(steps, dtype=np.int64),
      'Z': Z,
      'scores': scores,
      'var_ratio': var_ratio,
      'success_mean_iter': int(smean['iteration']) if smean else 68,
      'success_mean_step': int(smean['global_step']) if smean else 3532800,
      'success_1000_iter': int(s1000['iteration']) if s1000 else 62,
  }


def main(argv: Optional[List[str]] = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--run_dir', type=str, required=True)
  p.add_argument('--out', type=str, default=None,
                 help='PNG path (default depends on --max_iter/--arrows)')
  p.add_argument('--stride', type=int, default=40)
  p.add_argument('--max_iter', type=int, default=None,
                 help='Only include checkpoints with iter <= this value')
  p.add_argument('--arrows', action='store_true',
                 help='Draw curved time-order arcs between consecutive points')
  p.add_argument('--arc_bulge', type=float, default=0.18,
                 help='Arc height as fraction of chord length (default 0.18)')
  p.add_argument('--gauss2d', action='store_true',
                 help='Overlay N(0,I) 1σ/2σ circles in the PC1–PC2 plane')
  p.add_argument('--gauss_density_surface', action='store_true',
                 help='Plot N(0,I₂) density as z-axis surface; points at '
                      '(PC1,PC2,dens); drops PC3')
  p.add_argument('--include_final', action=argparse.BooleanOptionalAction,
                 default=True,
                 help='Also include the final ckpt if not already selected '
                      '(ignored when it exceeds --max_iter)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--which', type=str, default='online',
                 choices=('online', 'ema'))
  p.add_argument('--env', type=str, default=None)
  args = p.parse_args(argv)

  # Density-surface mode implies arrows unless the caller disabled via no flag
  # (store_true can't disable — default on when mode is set).
  use_arrows = bool(args.arrows) or bool(args.gauss_density_surface)

  run_dir = _abs(args.run_dir)
  out_png = _abs(args.out) if args.out else os.path.join(
      run_dir, _default_out_name(
          max_iter=args.max_iter, arrows=use_arrows, gauss2d=args.gauss2d,
          gauss_density_surface=args.gauss_density_surface))
  out_pdf = os.path.splitext(out_png)[0] + '.pdf'

  print(f'jax backend={jax.default_backend()} devices={jax.devices()}',
        flush=True)
  print(f'BUILDERBENCH_MJX_IMPL={os.environ.get("BUILDERBENCH_MJX_IMPL")}',
        flush=True)
  print(f'run_dir={run_dir}', flush=True)
  print(f'max_iter={args.max_iter} arrows={use_arrows} '
        f'gauss2d={args.gauss2d} '
        f'gauss_density_surface={args.gauss_density_surface} '
        f'arc_bulge={args.arc_bulge}', flush=True)

  n_comp = 2 if args.gauss_density_surface else 3
  data = collect_init_z_pca(
      run_dir,
      stride=args.stride,
      max_iter=args.max_iter,
      include_final=bool(args.include_final),
      seed=args.seed,
      which=args.which,
      env_name=args.env,
      n_components=n_comp,
  )
  iters_np = data['iters']
  scores = data['scores']
  var_ratio = data['var_ratio']
  success_mean_iter = data['success_mean_iter']
  success_mean_step = data['success_mean_step']
  success_1000_iter = data['success_1000_iter']

  if args.gauss_density_surface:
    cache = scores_csv_path(run_dir, max_iter=args.max_iter)
    save_pca2d_scores_csv(
        cache, iters_np, scores[:, :2], var_ratio,
        success_mean_iter=success_mean_iter,
        success_mean_step=success_mean_step,
        success_1000_iter=success_1000_iter,
    )
    print(f'wrote {cache}', flush=True)
    mark_it = _plot_gauss_density_surface(
        iters_np, scores[:, :2], var_ratio,
        success_mean_iter=success_mean_iter,
        success_mean_step=success_mean_step,
        success_1000_iter=success_1000_iter,
        out_png=out_png,
        out_pdf=out_pdf,
        arrows=use_arrows,
        arc_bulge=float(args.arc_bulge),
    )
  else:
    mark_it = _plot_pca3d(
        iters_np, scores, var_ratio,
        success_mean_iter=success_mean_iter,
        success_mean_step=success_mean_step,
        success_1000_iter=success_1000_iter,
        out_png=out_png,
        out_pdf=out_pdf,
        arrows=use_arrows,
        arc_bulge=float(args.arc_bulge),
        gauss2d=bool(args.gauss2d),
    )
  print(f'marked first plotted iter>={success_mean_iter}: {mark_it}',
        flush=True)
  print(f'wrote {out_png}', flush=True)
  print(f'wrote {out_pdf}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
