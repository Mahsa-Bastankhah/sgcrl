"""Visualize a compact, irregular BuilderBench 8-cube planar goal."""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('BUILDERBENCH_MJX_IMPL', 'jax')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BB = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import numpy as np

from envs.builderbench_utils import (
    apply_fixed_start_x,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

# Same midpoint as envs.builderbench_utils._TARGET_SAMPLING_MID.
_TARGET_ORIGIN = np.array([0.27, 0.0, 0.02], dtype=np.float32)

# Cubes are 4cm along each axis.
_CUBE = 0.04

# Seed-8 constrained random sample. It has no row, ring, cross, or other
# imposed structure. Constraints used while drawing it:
#   x offset in [-0.16, 0.06], y offset in [-0.21, 0.21],
#   pairwise center distance >= 0.08m.
# Adding any legal target origin x∈[0.22,0.32], y∈[-0.10,0.10] keeps every
# cube inside the BuilderBench workspace.
_RANDOM_OFFSETS = np.array([
    [-0.05915,  0.05923, 0.0],
    [-0.07940,  0.20277, 0.0],
    [ 0.00745,  0.14962, 0.0],
    [ 0.05836, -0.11016, 0.0],
    [-0.02915, -0.16634, 0.0],
    [-0.13704,  0.09583, 0.0],
    [-0.11696, -0.04927, 0.0],
    [ 0.02755, -0.03470, 0.0],
], dtype=np.float32)

_START_Y = np.array(
    [-0.28, -0.20, -0.12, -0.04, 0.04, 0.12, 0.20, 0.28],
    dtype=np.float32)


def _start_xyz(fixed_start_x: float = 0.1) -> np.ndarray:
  out = np.zeros((8, 3), dtype=np.float32)
  out[:, 0] = float(fixed_start_x)
  out[:, 1] = _START_Y
  out[:, 2] = 0.02
  return out


def _make_env():
  from builderbench.creative_cube import CreativeCube, default_config
  env_name = 'builderbench_creative_8_task3'
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.permute_start_boxes = False
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  env = CreativeCube(config=cfg)
  apply_fixed_start_x(env, 0.1)
  return env


def _camera(model, lookat, zoom, azimuth, elevation):
  import mujoco
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.lookat[:] = np.asarray(lookat, dtype=np.float64).reshape(3)
  extent = float(model.stat.extent) if float(model.stat.extent) > 0 else 0.8
  cam.distance = (1.5 * extent) / max(float(zoom), 1e-3)
  cam.azimuth = float(azimuth)
  cam.elevation = float(elevation)
  return cam


def _render_cubes(env, cube_xyz, *, lookat, zoom, azimuth, elevation,
                  height: int, width: int) -> np.ndarray:
  import mujoco
  num_task = int(env._num_task_cubes)
  qpos = np.array(env._init_q, copy=True)
  qpos[np.asarray(env._objs_pos_qpos_idxs)] = np.asarray(
      cube_xyz, dtype=np.float32).reshape(-1)
  qpos[np.asarray(env._objs_quat_qpos_idxs)] = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 8)
  d = mujoco.MjData(env._mj_model)
  d.qpos[:] = qpos
  d.qvel[:] = 0
  # Park mocap ghosts off-table so the figure shows the actual cubes.
  d.mocap_pos[env._task_mocap_targets] = np.tile(
      np.array([10.0, 10.0, 10.0]), (num_task, 1))
  d.mocap_quat[env._task_mocap_targets] = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0]), (num_task, 1))
  mujoco.mj_forward(env._mj_model, d)
  cam = _camera(env._mj_model, lookat, zoom, azimuth, elevation)
  renderer = mujoco.Renderer(env._mj_model, height=height, width=width)
  renderer.update_scene(d, camera=cam)
  img = np.asarray(renderer.render(), dtype=np.uint8)
  renderer.close()
  return img


def _fmt_xyz(pts: np.ndarray) -> str:
  lines = []
  for i, p in enumerate(pts):
    lines.append(f'c{i}: [{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]')
  return '\n'.join(lines)


def _min_center_dist(pts: np.ndarray) -> float:
  dmin = np.inf
  for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
      dmin = min(dmin, float(np.linalg.norm(pts[i] - pts[j])))
  return dmin


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--output', default='figs/builderbench/c8_random_planar_goal.png')
  parser.add_argument(
      '--schematic-output',
      default='figs/builderbench/c8_random_planar_goal_schematic.png')
  parser.add_argument('--width', type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  parser.add_argument('--schematic-only', action='store_true')
  args = parser.parse_args()

  origin = np.asarray(_TARGET_ORIGIN, dtype=np.float32)
  start = _start_xyz(0.1)
  goal = origin + _RANDOM_OFFSETS

  print(f'target origin (sampling mid) = {origin}')
  print(f'start (fixed x=0.1):\n{_fmt_xyz(start)}')
  nn = _min_center_dist(goal)
  print(f'random goal abs xyz  min center-dist={nn:.3f}m  '
        f'face-gap≈{nn - _CUBE:.3f}m\n{_fmt_xyz(goal)}')

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.patches import Rectangle

  fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
  colors = plt.cm.tab10(np.arange(8))
  for ax, title, pts in zip(
      axes, ('init: fixed x=0.10', 'compact irregular planar goal'),
      (start, goal)):
    ax.add_patch(Rectangle(
        (-0.05, -0.35), 0.50, 0.70, facecolor='#eef2f5',
        edgecolor='#7f8c8d'))
    ax.add_patch(Rectangle(
        (0.22, -0.10), 0.10, 0.20, facecolor='#f5b041',
        edgecolor='#d35400', linestyle='--', alpha=0.25))
    for i, (p, color) in enumerate(zip(pts, colors)):
      ax.add_patch(Rectangle(
          (float(p[0] - _CUBE / 2), float(p[1] - _CUBE / 2)),
          _CUBE, _CUBE, facecolor=color, edgecolor='black'))
      ax.text(p[0], p[1], str(i), ha='center', va='center',
              color='white', fontsize=8, fontweight='bold')
    ax.plot(origin[0], origin[1], 'k+', markersize=10)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-0.07, 0.47)
    ax.set_ylim(-0.30, 0.30)
    ax.set_xlabel('x (m)')
    ax.grid(alpha=0.2)
  axes[0].set_ylabel('y (m)')
  fig.suptitle(
      'BuilderBench creative-8: fixed random goal on table\n'
      f'min center distance={nn:.3f}m; cube face gap≈{nn - _CUBE:.3f}m',
      fontweight='bold')
  os.makedirs(os.path.dirname(args.schematic_output) or '.', exist_ok=True)
  fig.tight_layout()
  fig.savefig(args.schematic_output, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {args.schematic_output}')

  if args.schematic_only:
    return

  env = _make_env()
  table_lookat = np.array([0.20, 0.00, 0.02], dtype=np.float64)
  cams = [
      ('3/4', dict(lookat=table_lookat, zoom=2.2, azimuth=120, elevation=-25)),
      ('side', dict(lookat=table_lookat, zoom=2.2, azimuth=90, elevation=-15)),
      ('top', dict(lookat=table_lookat, zoom=2.4, azimuth=90, elevation=-90)),
  ]

  rows = [('init (start line)', start), ('random planar goal', goal)]

  frames = []
  for _, pts in rows:
    frames.append([
        _render_cubes(env, pts, height=args.height, width=args.width, **cam)
        for _, cam in cams
    ])

  n_rows, n_cols = len(rows), len(cams)
  fig, axes = plt.subplots(
      n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows))
  if n_rows == 1:
    axes = np.expand_dims(axes, 0)
  fig.suptitle(
      'BuilderBench creative-8 — compact random planar goal\n'
      f'target origin = {origin.tolist()}   |   ghosts hidden   |   '
      'registered as task3',
      fontsize=12, fontweight='bold')

  for r, ((title, pts), imgs) in enumerate(zip(rows, frames)):
    nn = _min_center_dist(pts)
    note = (
        f'{title}\nmin ‖ci−cj‖={nn:.3f}m  face-gap≈{nn - _CUBE:.3f}m\n'
        + _fmt_xyz(pts)
    )
    for c, ((cam_name, _), img) in enumerate(zip(cams, imgs)):
      ax = axes[r, c]
      ax.imshow(img)
      ax.set_title(f'{title} — {cam_name}', fontsize=10)
      ax.axis('off')
      if c == 0:
        ax.text(
            8, args.height - 8, note, fontsize=7, family='monospace',
            va='bottom', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.55))

  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
  fig.tight_layout()
  fig.savefig(args.output, dpi=130, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {args.output}')


if __name__ == '__main__':
  main()
