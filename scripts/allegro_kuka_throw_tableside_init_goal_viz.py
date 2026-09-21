#!/usr/bin/env python3
"""Stills of tableside object-push: table_spawn init vs on-desk goal.

``--mode=above --large-table`` is the current train init: palm-down 12 cm
above the cube on the 1.5x1.5 m table. ``behind`` is the older +y paddle
standoff; ``grasp`` is IK onto the cube. Episode T=50.

Two goal packings (same scene; markers differ):

  v1 object-only  --isaacgym_table_push  (no palm_goal)
      state 49 = [q23, qd23, obj3]   goal 3 = object_xyz
  v2 palm+object  --isaacgym_table_push --isaacgym_palm_goal
      state 52 = [q23, qd23, palm3, obj3]   goal 6 = [palm_xyz, object_xyz]

  python scripts/allegro_kuka_throw_tableside_init_goal_viz.py --variant=object
  python scripts/allegro_kuka_throw_tableside_init_goal_viz.py --variant=palm
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    GOAL_DIM,
    GOAL_DIM_PALM,
    LARGE_TABLE_BOUNDS_XY,
    STATE_DIM,
    STATE_DIM_PALM,
    TABLE_OBJECT_Z,
    TABLE_SIDE_GOAL_XYZ,
    TABLE_SIDE_PALM_XYZ,
    TABLE_SPAWN_BEHIND_OBJECT_XY,
    TABLE_SPAWN_BEHIND_PALM_ABOVE,
    TABLE_SPAWN_BEHIND_PALM_DY,
    TABLE_SPAWN_OBJECT_XY,
    AllegroKukaThrowVecEnv,
)

EPISODE_T = 50
CUBE_HALF = 0.025  # ~5 cm cube
SUCCESS_TOL = 0.075  # env success_tolerance
# Hover-above: palm-down, 12 cm above cube center (~9.5 cm above cube top).
HOVER_ABOVE_DY = 0.0
HOVER_ABOVE_Z = 0.12
OBLIQUE_EYE = (0.54, -0.36, 0.84)
OBLIQUE_TGT = (0.20, 0.00, 0.56)
# Slightly off-axis so look_at up=(0,0,1) is not singular (straight-down = black).
TOP_EYE = (0.32, -0.28, 1.02)
TOP_TGT = (0.18, -0.03, 0.55)
# Pulled back so the 1.5x1.5 table is in frame.
WIDE_EYE = (1.45, -1.20, 1.90)
WIDE_TGT = (0.10, -0.25, 0.53)
WIDE_TOP_EYE = (0.15, -0.40, 2.55)
WIDE_TOP_TGT = (0.10, -0.25, 0.53)
HFOV = 34.0
WIDE_HFOV = 48.0
WH = (960, 720)


def _planar_cm(a_xy, b_xy) -> float:
  d = np.asarray(a_xy, dtype=np.float64) - np.asarray(b_xy, dtype=np.float64)
  return float(np.linalg.norm(d) * 100.0)


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
  eye = np.asarray(eye, dtype=np.float64)
  target = np.asarray(target, dtype=np.float64)
  up = np.asarray(up, dtype=np.float64)
  f = target - eye
  f = f / (np.linalg.norm(f) + 1e-12)
  s = np.cross(f, up)
  s = s / (np.linalg.norm(s) + 1e-12)
  u = np.cross(s, f)
  view = np.eye(4, dtype=np.float64)
  view[0, 0:3] = s
  view[1, 0:3] = u
  view[2, 0:3] = -f
  view[0, 3] = -np.dot(s, eye)
  view[1, 3] = -np.dot(u, eye)
  view[2, 3] = np.dot(f, eye)
  return view


def _perspective(hfov_deg, aspect, near=0.05, far=8.0):
  r = np.tan(np.deg2rad(float(hfov_deg)) * 0.5) * near
  t = r / max(float(aspect), 1e-6)
  proj = np.zeros((4, 4), dtype=np.float64)
  proj[0, 0] = near / r
  proj[1, 1] = near / t
  proj[2, 2] = -(far + near) / (far - near)
  proj[2, 3] = -2.0 * far * near / (far - near)
  proj[3, 2] = -1.0
  return proj


def _project(xyz, view, proj, width, height):
  p = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
  clip = proj @ (view @ p)
  if abs(clip[3]) < 1e-8:
    return None
  ndc = clip[:3] / clip[3]
  u = (ndc[0] * 0.5 + 0.5) * width
  v = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
  if not np.isfinite(u) or not np.isfinite(v):
    return None
  return float(u), float(v)


def _restore_noiseless(env) -> None:
  from isaacgym import gymtorch

  task = env._env
  task.arm_hand_dof_pos[:, :] = task.hand_arm_default_dof_pos
  task.cur_targets[:, :] = task.hand_arm_default_dof_pos
  task.prev_targets[:, :] = task.hand_arm_default_dof_pos
  task._set_object_root(None)
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))
  for _ in range(8):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._set_object_root(None)
  task.compute_observations()


def _set_cam(env, eye, tgt, hfov=None) -> None:
  from isaacgym import gymapi

  task = env._env
  env._cam_eye = tuple(float(v) for v in eye)
  env._cam_tgt = tuple(float(v) for v in tgt)
  env._cam_hfov = float(HFOV if hfov is None else hfov)
  task.gym.set_camera_location(
      env._cam_handle, task.envs[0],
      gymapi.Vec3(*env._cam_eye), gymapi.Vec3(*env._cam_tgt))


def _ghost_cube_edges(center, half=CUBE_HALF):
  c = np.asarray(center, dtype=np.float64)
  signs = (-1.0, 1.0)
  corners = [
      c + np.array([sx * half, sy * half, sz * half], dtype=np.float64)
      for sx in signs for sy in signs for sz in signs]
  edges = []
  for i, a in enumerate(corners):
    for j, b in enumerate(corners):
      if j <= i:
        continue
      if int(np.sum(np.abs(a - b) > 1e-9)) == 1:
        edges.append((a, b))
  return edges


def _overlay(rgb, env, *, palm_markers: bool, caption_lines):
  from PIL import Image, ImageDraw

  h, w = rgb.shape[:2]
  view = _look_at(env._cam_eye, env._cam_tgt)
  hfov = float(getattr(env, '_cam_hfov', HFOV) or HFOV)
  proj = _perspective(hfov, float(w) / max(float(h), 1.0))
  goal = np.asarray(TABLE_SIDE_GOAL_XYZ, dtype=np.float64)
  palm_g = np.asarray(TABLE_SIDE_PALM_XYZ, dtype=np.float64)
  obj = env._object_xyz()[0].detach().cpu().numpy()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  ring = []
  for k in range(48):
    ang = 2.0 * np.pi * k / 48.0
    xyz = goal + np.array(
        [SUCCESS_TOL * np.cos(ang), SUCCESS_TOL * np.sin(ang), 0.0])
    uv = _project(xyz, view, proj, w, h)
    if uv is not None:
      ring.append(uv)
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  if len(ring) >= 3:
    poly = [(int(u), int(v)) for u, v in ring]
    draw.polygon(poly, fill=(255, 140, 40, 70), outline=(255, 180, 50, 230))
    draw.line(poly + poly[:1], fill=(255, 200, 60, 255), width=4)
  for a, b in _ghost_cube_edges(goal):
    ua = _project(a, view, proj, w, h)
    ub = _project(b, view, proj, w, h)
    if ua and ub:
      draw.line(
          [(int(ua[0]), int(ua[1])), (int(ub[0]), int(ub[1]))],
          fill=(255, 170, 60, 220), width=2)
  g_uv = _project(goal, view, proj, w, h)
  o_uv = _project(obj, view, proj, w, h)
  p_uv = _project(palm, view, proj, w, h)
  pg_uv = _project(palm_g, view, proj, w, h)
  if o_uv and g_uv:
    draw.line(
        [(int(o_uv[0]), int(o_uv[1])), (int(g_uv[0]), int(g_uv[1]))],
        fill=(255, 220, 80, 220), width=3)

  def _dot(uv, fill, label, dy=-18):
    if uv is None:
      return
    u, v = int(uv[0]), int(uv[1])
    draw.ellipse([u - 7, v - 7, u + 7, v + 7], fill=fill,
                 outline=(255, 255, 255, 255))
    draw.text((u + 10, v + dy), label, fill=fill)

  _dot(g_uv, (255, 170, 50, 255), 'object goal  (ghost cube + 7.5cm disk)')
  _dot(o_uv, (240, 240, 240, 255), 'cube start')
  _dot(p_uv, (80, 220, 255, 255), 'palm (live)')
  if palm_markers:
    _dot(pg_uv, (40, 200, 255, 255), 'palm goal  (v2)', dy=10)
    if p_uv and pg_uv:
      draw.line(
          [(int(p_uv[0]), int(p_uv[1])), (int(pg_uv[0]), int(pg_uv[1]))],
          fill=(80, 200, 255, 180), width=2)
  bar_h = 18 + 16 * len(caption_lines)
  canvas = np.full((h + bar_h, w, 3), 18, dtype=np.uint8)
  canvas[bar_h:] = np.asarray(img.convert('RGB'))
  out = Image.fromarray(canvas)
  drawer = ImageDraw.Draw(out)
  for i, line in enumerate(caption_lines):
    drawer.text((12, 6 + 16 * i), line, fill=(240, 240, 240))
  return np.asarray(out, dtype=np.uint8)


def _save(path: str, rgb) -> None:
  from PIL import Image

  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  Image.fromarray(rgb).save(path)
  print(f'wrote {path}', flush=True)


def _spawn_xy(mode: str):
  if mode in ('behind', 'above'):
    return TABLE_SPAWN_BEHIND_OBJECT_XY
  return TABLE_SPAWN_OBJECT_XY


def _print_dims(*, variant: str, mode: str, env, palm_dy: float,
                palm_above: float, large_table: bool) -> None:
  spawn_xy = _spawn_xy(mode)
  goal_xy = TABLE_SIDE_GOAL_XYZ[:2]
  cm = _planar_cm(spawn_xy, goal_xy)
  obj = env._object_xyz()[0].detach().cpu().numpy()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  packed = env._pack_obs()
  print('[tableside_init_goal] === packing ===', flush=True)
  print(
      f'  current/proposed v1 object-only: state={STATE_DIM} '
      f'[q23, qd23, obj3]  goal={GOAL_DIM}  (table_push, no palm_goal)',
      flush=True)
  print(
      f'  current/proposed v2 palm+object: state={STATE_DIM_PALM} '
      f'[q23, qd23, palm3, obj3]  goal={GOAL_DIM_PALM}  '
      f'(table_push + palm_goal)',
      flush=True)
  print(
      f'  this process variant={variant} mode={mode}  '
      f'env.obs_dim={env.obs_dim}  env.goal_dim={env.goal_dim}  '
      f'packed={tuple(packed.shape)}  episode_length={EPISODE_T}',
      flush=True)
  print(
      f'  start_xy={spawn_xy}  start_z={TABLE_OBJECT_Z:.3f}  '
      f'goal_xyz={TABLE_SIDE_GOAL_XYZ}  palm_goal_xyz={TABLE_SIDE_PALM_XYZ}',
      flush=True)
  print(
      f'  planar start→goal = {cm:.1f} cm  '
      f'(dx={(TABLE_SIDE_GOAL_XYZ[0] - spawn_xy[0]) * 100:.1f} cm, '
      f'dy={(TABLE_SIDE_GOAL_XYZ[1] - spawn_xy[1]) * 100:.1f} cm)  '
      f'historical job text said ~25cm from table center (0,0)',
      flush=True)
  print(
      f'  live after noiseless restore: obj={tuple(float(v) for v in obj)}  '
      f'palm={tuple(float(v) for v in palm)}  '
      f'|palm-obj|={float(np.linalg.norm(palm - obj)) * 100:.1f} cm',
      flush=True)
  flags = (
      '--env=allegro_kuka_throw --isaacgym_table_push '
      '--isaacgym_table_push_xyz=0.20,-0.15,0.555 --isaacgym_table_spawn '
      '--noisaacgym_randomize_object_shape '
      '--isaacgym_episode_length=50 --ppo_rollout_length=50')
  if mode in ('behind', 'above'):
    flags += (
        ' --isaacgym_table_spawn_behind '
        f'--isaacgym_table_spawn_behind_dy={palm_dy:g} '
        f'--isaacgym_table_spawn_behind_above={palm_above:g}')
  if large_table:
    flags += ' --isaacgym_large_table'
  if variant == 'palm':
    flags += ' --isaacgym_palm_goal --isaacgym_palm_goal_xyz=0.20,-0.15,0.620'
  print(f'  flags: {flags}', flush=True)
  if large_table:
    print(
        f'  large_table=1.5x1.5  bounds x={LARGE_TABLE_BOUNDS_XY[0]} '
        f'y={LARGE_TABLE_BOUNDS_XY[1]}',
        flush=True)
  print(
      '  env change for [q,qd,obj]: none (table_push already packs full '
      'arm+hand q,qd + object; not ITS trim-SA)',
      flush=True)


def _caption_lines(variant: str, mode: str, view_name: str, *,
                   palm_dy: float, palm_above: float,
                   large_table: bool) -> list[str]:
  spawn_xy = _spawn_xy(mode)
  cm = _planar_cm(spawn_xy, TABLE_SIDE_GOAL_XYZ[:2])
  if mode == 'above':
    init = (
        f'palm-down hover (dy={palm_dy*100:.0f}cm, '
        f'+{palm_above*100:.0f}cm above cube center)')
    init_flags = (
        f'--isaacgym_table_spawn --isaacgym_table_spawn_behind '
        f'dy={palm_dy:g} above={palm_above:g}')
  elif mode == 'behind':
    init = (
        f'table_spawn_behind (palm +{palm_dy*100:.0f}cm y / '
        f'+{palm_above*100:.0f}cm z)')
    init_flags = '--isaacgym_table_spawn --isaacgym_table_spawn_behind'
  else:
    init = 'table_spawn grasp (IK palm on cube)'
    init_flags = '--isaacgym_table_spawn'
  table = '1.5x1.5 large table' if large_table else 'narrow NVIDIA table'
  if variant == 'palm':
    dims = (
        f'v2 palm+object  state {STATE_DIM_PALM}=[q23,qd23,palm3,obj3]  '
        f'goal {GOAL_DIM_PALM}=[palm,obj]')
    flags = (
        f'--isaacgym_table_push {init_flags} --isaacgym_palm_goal '
        'palm=(0.20,-0.15,0.620)')
  else:
    dims = (
        f'v1 object-only  state {STATE_DIM}=[q23,qd23,obj3]  '
        f'goal {GOAL_DIM}=object_xyz')
    flags = f'--isaacgym_table_push {init_flags}  (palm_goal off)'
  return [
      f'tableside object-push  T={EPISODE_T}  {init}  {table}  {view_name}',
      (f'start cube {spawn_xy[0]:.2f},{spawn_xy[1]:.2f},'
       f'{TABLE_OBJECT_Z:.3f}  →  object goal '
       f'{TABLE_SIDE_GOAL_XYZ[0]:.2f},{TABLE_SIDE_GOAL_XYZ[1]:.2f},'
       f'{TABLE_SIDE_GOAL_XYZ[2]:.3f}  planar {cm:.1f} cm'),
      f'{dims}  {flags}',
  ]


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--variant', choices=('object', 'palm'), required=True)
  p.add_argument(
      '--mode', choices=('grasp', 'behind', 'above'), default='grasp',
      help='grasp = IK-on-cube; behind = +y paddle standoff; '
           'above = palm-down hover over the cube.')
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'tableside_init_goal'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--large-table', action='store_true',
                 help='Use the 1.5x1.5 m away-shifted table.')
  p.add_argument('--palm-dy', type=float, default=None)
  p.add_argument('--palm-above', type=float, default=None)
  p.add_argument('--video', action='store_true',
                 help='Also write a 2s still-hold mp4 of the oblique view.')
  args = p.parse_args()
  palm_goal = args.variant == 'palm'
  behind = args.mode in ('behind', 'above')
  if args.mode == 'above':
    palm_dy = HOVER_ABOVE_DY if args.palm_dy is None else float(args.palm_dy)
    palm_above = (
        HOVER_ABOVE_Z if args.palm_above is None else float(args.palm_above))
  else:
    palm_dy = (
        TABLE_SPAWN_BEHIND_PALM_DY if args.palm_dy is None
        else float(args.palm_dy))
    palm_above = (
        TABLE_SPAWN_BEHIND_PALM_ABOVE if args.palm_above is None
        else float(args.palm_above))
  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=0,
      episode_length=EPISODE_T,
      table_spawn=True,
      table_spawn_behind=behind,
      table_spawn_behind_dy=palm_dy,
      table_spawn_behind_above=palm_above,
      table_spawn_finger_curl_scale=0.50,
      table_push=True,
      table_push_xyz=TABLE_SIDE_GOAL_XYZ,
      palm_goal=palm_goal,
      palm_goal_xyz=TABLE_SIDE_PALM_XYZ,
      large_table=bool(args.large_table),
      randomize_init=True,
      randomize_object_shape=False,
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=OBLIQUE_EYE,
      camera_tgt=OBLIQUE_TGT,
      camera_hfov=HFOV,
      pipeline=args.pipeline,
  )
  env.reset()
  _restore_noiseless(env)
  _print_dims(
      variant=args.variant, mode=args.mode, env=env,
      palm_dy=palm_dy, palm_above=palm_above, large_table=bool(args.large_table))
  tag = f'{args.mode}_{"v1_object" if not palm_goal else "v2_palm"}'
  if args.large_table:
    tag += '_largetable15'
  cap_kw = dict(
      palm_dy=palm_dy, palm_above=palm_above, large_table=bool(args.large_table))
  close_eye, close_tgt = (OBLIQUE_EYE, OBLIQUE_TGT)
  top_eye, top_tgt = (TOP_EYE, TOP_TGT)
  close_hfov = HFOV
  if args.large_table:
    top_eye, top_tgt = (WIDE_TOP_EYE, WIDE_TOP_TGT)
  _set_cam(env, close_eye, close_tgt, hfov=close_hfov)
  rgb = env.render_rgb()
  if rgb is None:
    print('camera returned no frame', flush=True)
    return 1
  oblique = _overlay(
      rgb, env, palm_markers=palm_goal,
      caption_lines=_caption_lines(
          args.variant, args.mode, 'oblique close', **cap_kw))
  _save(os.path.join(args.out_dir, f'tableside_init_goal_{tag}_oblique.png'),
        oblique)
  _set_cam(env, top_eye, top_tgt, hfov=WIDE_HFOV if args.large_table else HFOV)
  rgb = env.render_rgb()
  top = _overlay(
      rgb, env, palm_markers=palm_goal,
      caption_lines=_caption_lines(
          args.variant, args.mode, 'top', **cap_kw))
  _save(os.path.join(args.out_dir, f'tableside_init_goal_{tag}_top.png'), top)
  if args.large_table:
    _set_cam(env, WIDE_EYE, WIDE_TGT, hfov=WIDE_HFOV)
    rgb = env.render_rgb()
    wide = _overlay(
        rgb, env, palm_markers=palm_goal,
        caption_lines=_caption_lines(
            args.variant, args.mode, 'wide table', **cap_kw))
    _save(os.path.join(args.out_dir, f'tableside_init_goal_{tag}_wide.png'),
          wide)
  if args.video:
    try:
      import imageio
    except ImportError:
      print('imageio missing; skip video', flush=True)
      return 0
    vid_path = os.path.join(
        args.out_dir, f'tableside_init_goal_{tag}_oblique.mp4')
    writer = imageio.get_writer(vid_path, fps=15, codec='libx264', quality=8)
    try:
      for _ in range(30):
        writer.append_data(oblique)
    finally:
      writer.close()
    print(f'wrote {vid_path}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
