#!/usr/bin/env python3
"""Zoomed stills of table-spawn push inits (no coordinates in the caption).

  python scripts/allegro_kuka_throw_push_spawn_viz.py --mode=current
  python scripts/allegro_kuka_throw_push_spawn_viz.py --mode=behind
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    TABLE_SIDE_GOAL_XYZ,
    TABLE_SIDE_PALM_XYZ,
    AllegroKukaThrowVecEnv,
)

# Tight desk views: cube, hand, and the orange success disk in-frame.
OBLIQUE_EYE = (0.54, -0.36, 0.84)
OBLIQUE_TGT = (0.20, 0.00, 0.56)
TOP_EYE = (0.20, -0.03, 1.08)
TOP_TGT = (0.20, -0.03, 0.55)
HFOV = 34.0
WH = (960, 720)


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
    # Keep the cube parked so finger contact cannot yeet it for the still.
    task._set_object_root(None)
  task.compute_observations()


def _set_cam(env, eye, tgt) -> None:
  from isaacgym import gymapi

  task = env._env
  env._cam_eye = tuple(float(v) for v in eye)
  env._cam_tgt = tuple(float(v) for v in tgt)
  env._cam_hfov = float(HFOV)
  task.gym.set_camera_location(
      env._cam_handle, task.envs[0],
      gymapi.Vec3(*env._cam_eye), gymapi.Vec3(*env._cam_tgt))


def _overlay(rgb, env, caption: str):
  from PIL import Image, ImageDraw

  h, w = rgb.shape[:2]
  view = _look_at(env._cam_eye, env._cam_tgt)
  proj = _perspective(HFOV, float(w) / max(float(h), 1.0))
  goal = np.asarray(TABLE_SIDE_GOAL_XYZ, dtype=np.float64)
  obj = env._object_xyz()[0].detach().cpu().numpy()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  tol = 0.075
  ring = []
  for k in range(48):
    ang = 2.0 * np.pi * k / 48.0
    xyz = goal + np.array([tol * np.cos(ang), tol * np.sin(ang), 0.0])
    uv = _project(xyz, view, proj, w, h)
    if uv is not None:
      ring.append(uv)
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  if len(ring) >= 3:
    poly = [(int(u), int(v)) for u, v in ring]
    draw.polygon(poly, fill=(255, 140, 40, 70), outline=(255, 180, 50, 230))
    draw.line(poly + poly[:1], fill=(255, 200, 60, 255), width=4)
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
  _dot(g_uv, (255, 170, 50, 255), 'goal  (orange disk = success)')
  _dot(o_uv, (240, 240, 240, 255), 'cube')
  _dot(p_uv, (80, 220, 255, 255), 'palm')
  bar_h = 40
  canvas = np.full((h + bar_h, w, 3), 18, dtype=np.uint8)
  canvas[bar_h:] = np.asarray(img.convert('RGB'))
  out = Image.fromarray(canvas)
  ImageDraw.Draw(out).text((12, 10), caption, fill=(240, 240, 240))
  return np.asarray(out, dtype=np.uint8)


def _save(path: str, rgb) -> None:
  from PIL import Image

  os.makedirs(os.path.dirname(path), exist_ok=True)
  Image.fromarray(rgb).save(path)
  print(f'wrote {path}', flush=True)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--mode', choices=('current', 'behind'), required=True)
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw'))
  p.add_argument('--pipeline', default='gpu')
  args = p.parse_args()
  behind = args.mode == 'behind'
  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=0,
      episode_length=300,
      table_spawn=True,
      table_spawn_behind=behind,
      table_push=True,
      table_push_xyz=TABLE_SIDE_GOAL_XYZ,
      palm_goal=True,
      palm_goal_xyz=TABLE_SIDE_PALM_XYZ,
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
  caption = (
      'proposed: palm behind the cube, ready to push toward the orange disk'
      if behind else
      'current: palm on top of the cube  (easy to knock it off)')
  _set_cam(env, OBLIQUE_EYE, OBLIQUE_TGT)
  rgb = env.render_rgb()
  if rgb is None:
    print('camera returned no frame', flush=True)
    return 1
  _save(os.path.join(args.out_dir, f'push_spawn_{args.mode}_oblique.png'),
        _overlay(rgb, env, caption))
  _set_cam(env, TOP_EYE, TOP_TGT)
  rgb = env.render_rgb()
  _save(os.path.join(args.out_dir, f'push_spawn_{args.mode}_top.png'),
        _overlay(rgb, env, caption + '   (top view)'))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
