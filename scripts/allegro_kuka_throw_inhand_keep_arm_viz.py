#!/usr/bin/env python3
"""Stills: pose-v1 arm, wrist roll, curled fingers, cube in the palm.

Large 1.5x1.5 table. No IK. Orange ghost + 7.5 cm disk = object goal
``IN_HAND_KEEP_ARM_GOAL_XYZ`` (on the desk, ~26 cm in front of the hold).

  python scripts/allegro_kuka_throw_inhand_keep_arm_viz.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    IN_HAND_KEEP_ARM_GOAL_XYZ,
    TABLE_SPAWN_IN_HAND_OBJ_NOISE,
    TABLE_SPAWN_IN_HAND_OFFSET,
    TABLE_SPAWN_IN_HAND_WRIST_NOISE,
    TABLE_SPAWN_IN_HAND_WRIST_OFFSET,
    AllegroKukaThrowVecEnv,
)

EPISODE_T = 50
N_INITS = 4
CURL = 1.0
FINGER_NOISE = 0.04
WH = (960, 720)
HFOV = 42.0
CUBE_HALF = 0.025
SUCCESS_TOL = 0.075
# Look between the palm-up hold and the on-table goal.
VIEWS = {
    'oblique': ((1.00, -1.00, 1.25), (0.00, -0.08, 0.56)),
    'side': ((1.30, -0.08, 1.00), (0.00, -0.08, 0.58)),
    'front': ((0.00, -1.30, 1.05), (0.00, -0.06, 0.56)),
}


def _font(size: int):
  try:
    return ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', size)
  except OSError:
    return ImageFont.load_default()


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


def _overlay(rgb, env, caption_lines):
  from PIL import Image, ImageDraw

  h, w = rgb.shape[:2]
  view = _look_at(env._cam_eye, env._cam_tgt)
  hfov = float(getattr(env, '_cam_hfov', HFOV) or HFOV)
  proj = _perspective(hfov, float(w) / max(float(h), 1.0))
  goal = np.asarray(IN_HAND_KEEP_ARM_GOAL_XYZ, dtype=np.float64)
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
          fill=(255, 170, 60, 220), width=3)
  g_uv = _project(goal, view, proj, w, h)
  o_uv = _project(obj, view, proj, w, h)
  p_uv = _project(palm, view, proj, w, h)
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

  _dot(g_uv, (255, 170, 50, 255), 'object goal  (on table)')
  _dot(o_uv, (240, 240, 240, 255), 'cube in hand')
  _dot(p_uv, (80, 220, 255, 255), 'palm')
  bar_h = 8 + 20 * len(caption_lines)
  canvas = np.full((h + bar_h, w, 3), 18, dtype=np.uint8)
  canvas[bar_h:] = np.asarray(img.convert('RGB'))
  out = Image.fromarray(canvas)
  drawer = ImageDraw.Draw(out)
  font = _font(16)
  for i, line in enumerate(caption_lines):
    drawer.text((8, 4 + 20 * i), line, fill=(240, 240, 240), font=font)
  return np.asarray(out, dtype=np.uint8)


def _set_cam(env, eye, tgt, hfov=HFOV) -> None:
  from isaacgym import gymapi

  env._cam_eye = tuple(float(v) for v in eye)
  env._cam_tgt = tuple(float(v) for v in tgt)
  env._cam_hfov = float(hfov)
  env._env.gym.set_camera_location(
      env._cam_handle, env._env.envs[0],
      gymapi.Vec3(*env._cam_eye), gymapi.Vec3(*env._cam_tgt))


def _caption(rgb, lines) -> np.ndarray:
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  bar_h = 8 + 20 * len(lines)
  draw.rectangle((0, 0, img.width, bar_h), fill=(0, 0, 0, 175))
  font = _font(16)
  y = 4
  for line in lines:
    draw.text((8, y), line, fill=(255, 255, 255, 255), font=font)
    y += 20
  return np.asarray(img)


def _grid(frames, cols: int = 2) -> Image.Image:
  h, w = frames[0].shape[:2]
  rows = (len(frames) + cols - 1) // cols
  canvas = Image.new('RGB', (cols * w, rows * h), (20, 20, 20))
  for i, fr in enumerate(frames):
    r, c = divmod(i, cols)
    canvas.paste(Image.fromarray(fr), (c * w, r * h))
  return canvas


def _show_src(env, src: int, q_all) -> None:
  from isaacgym import gymtorch
  import torch

  task = env._env
  q = q_all[src]
  task.arm_hand_dof_pos[0] = q
  task.arm_hand_dof_vel[0] = 0.0
  task.prev_targets[0] = q
  task.cur_targets[0] = q
  hand = task.allegro_hand_indices[0:1].to(torch.int32)
  task.gym.set_dof_position_target_tensor_indexed(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets),
      gymtorch.unwrap_tensor(hand), 1)
  task.gym.set_dof_state_tensor_indexed(
      task.sim, gymtorch.unwrap_tensor(task.dof_state),
      gymtorch.unwrap_tensor(hand), 1)
  for _ in range(4):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
  task.compute_observations()
  ids = torch.tensor([0], device=task.device, dtype=torch.long)
  task._place_cube_in_hand(ids, write_init=False, use_stored=False)
  for _ in range(2):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._place_cube_in_hand(ids, write_init=False, use_stored=False)
  task.compute_observations()


def _lines(i: int, env, view: str) -> list[str]:
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  obj = env._object_xyz()[0].detach().cpu().numpy()
  goal = np.asarray(IN_HAND_KEEP_ARM_GOAL_XYZ, dtype=np.float64)
  dist = float(np.linalg.norm(palm - obj))
  to_goal = float(np.linalg.norm(obj - goal))
  arm = tuple(
      round(float(v), 3)
      for v in env._env.arm_hand_dof_pos[0, :7].tolist())
  return [
      (f'keep-arm in-hand {i}  {view}  pose v1 + wrist {TABLE_SPAWN_IN_HAND_WRIST_OFFSET:g} '
       f'±{TABLE_SPAWN_IN_HAND_WRIST_NOISE:g}  curl={CURL:g}  '
       f'large 1.5x1.5 table'),
      (f'arm={arm}  palm=({palm[0]:.3f},{palm[1]:.3f},{palm[2]:.3f})  '
       f'cube=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})  '
       f'|palm-cube|={dist*100:.1f}cm'),
      (f'object goal={tuple(round(float(v), 3) for v in goal)}  '
       f'|cube-goal|={to_goal*100:.1f}cm  success={SUCCESS_TOL*100:.1f}cm disk'),
  ]


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'inhand_keep_arm'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--n', type=int, default=N_INITS)
  p.add_argument('--seed', type=int, default=0)
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  n = max(int(args.n), 1)
  env = AllegroKukaThrowVecEnv(
      num_envs=n,
      seed=int(args.seed),
      episode_length=EPISODE_T,
      table_spawn=True,
      table_spawn_object_xy=(0.0, 0.0),
      table_spawn_in_hand=True,
      table_spawn_in_hand_keep_arm=True,
      table_spawn_in_hand_offset=TABLE_SPAWN_IN_HAND_OFFSET,
      table_spawn_in_hand_obj_noise=TABLE_SPAWN_IN_HAND_OBJ_NOISE,
      table_spawn_in_hand_wrist_offset=TABLE_SPAWN_IN_HAND_WRIST_OFFSET,
      table_spawn_in_hand_wrist_noise=TABLE_SPAWN_IN_HAND_WRIST_NOISE,
      table_spawn_finger_curl_scale=CURL,
      table_spawn_finger_noise=FINGER_NOISE,
      table_spawn_arm_noise=0.0,
      table_push=True,
      table_push_xyz=IN_HAND_KEEP_ARM_GOAL_XYZ,
      large_table=True,
      randomize_init=True,
      randomize_object_shape=False,
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=VIEWS['oblique'][0],
      camera_tgt=VIEWS['oblique'][1],
      camera_hfov=HFOV,
      pipeline=args.pipeline,
  )
  env.reset()
  q_all = env._env.arm_hand_dof_pos.detach().clone()
  side_frames = []
  for i in range(n):
    _show_src(env, i, q_all)
    palm = env._palm_xyz()[0].detach().cpu().numpy()
    obj = env._object_xyz()[0].detach().cpu().numpy()
    dist = float(np.linalg.norm(palm - obj))
    print(
        f'[keep_arm] env {i}  palm={tuple(round(float(v), 3) for v in palm)}  '
        f'obj={tuple(round(float(v), 3) for v in obj)}  '
        f'|palm-obj|={dist:.3f}m',
        flush=True)
    for name, (eye, tgt) in VIEWS.items():
      _set_cam(env, eye, tgt)
      rgb = env.render_rgb()
      if rgb is None:
        print('camera returned no frame', flush=True)
        return 1
      labeled = _overlay(rgb, env, _lines(i, env, name))
      path = os.path.join(args.out_dir, f'keep_arm_{i:02d}_{name}.png')
      Image.fromarray(labeled).save(path)
      print(f'wrote {path}', flush=True)
      if name == 'side':
        side_frames.append(labeled)
  grid_path = os.path.join(args.out_dir, 'keep_arm_grid_side.png')
  _grid(side_frames, cols=2).save(grid_path)
  print(f'wrote {grid_path}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
