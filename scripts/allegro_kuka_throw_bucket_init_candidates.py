#!/usr/bin/env python3
"""Render table_spawn init candidates aimed at the throw bucket.

Four stills (oblique + top each):
  current     — today's palm-on-cube table_spawn (tiny joint noise base pose)
  standoff14  — palm opposite the bucket, 14 cm stand-off, paddle toward bucket
  standoff10  — same, tighter 10 cm / 5 cm above
  standoff18  — same, longer 18 cm / 8 cm above

  python scripts/allegro_kuka_throw_bucket_init_candidates.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    DEFAULT_FIXED_TARGET_XYZ,
    TABLE_OBJECT_Z,
    TABLE_SPAWN_BEHIND_FINGER_Q,
    TABLE_SPAWN_OBJECT_XY,
    AllegroKukaThrowVecEnv,
)

# Wide enough to see cube, palm, and bucket.
OBLIQUE_EYE = (0.72, -0.55, 0.95)
OBLIQUE_TGT = (0.30, -0.10, 0.50)
TOP_EYE = (0.32, -0.12, 1.25)
TOP_TGT = (0.32, -0.12, 0.50)
HFOV = 48.0
WH = (960, 720)

# (name, stand-off along cube→bucket in xy [m], palm z above cube center [m])
# None = keep env's default table_spawn touch IK (current).
STANDOFFS = {
    'current': None,
    'standoff14': (0.14, 0.08),
    'standoff10': (0.10, 0.05),
    'standoff18': (0.18, 0.08),
}


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


def _bucket_goal_xyz():
  # Match AllegroKukaThrowVecEnv: object goal = bucket xy, z + 0.05.
  b = DEFAULT_FIXED_TARGET_XYZ
  return (float(b[0]), float(b[1]), float(b[2]) + 0.05)


def _dir_cube_to_bucket_xy():
  cube = np.asarray(TABLE_SPAWN_OBJECT_XY, dtype=np.float64)
  buck = np.asarray(DEFAULT_FIXED_TARGET_XYZ[:2], dtype=np.float64)
  d = buck - cube
  n = float(np.linalg.norm(d))
  if n < 1e-8:
    return np.array([1.0, 0.0], dtype=np.float64)
  return d / n


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


def _acquire_jacobian(task):
  from isaacgym import gymtorch

  raw = task.gym.acquire_jacobian_tensor(task.sim, 'allegro')
  return gymtorch.wrap_tensor(raw)


def _ik_standoff_toward_bucket(task, jac, dist: float, above: float):
  """Stand palm opposite the bucket so a shove goes toward the bucket."""
  import torch

  dxy = _dir_cube_to_bucket_xy()
  # Palm on the anti-bucket side of the cube.
  palm_xy = (
      TABLE_SPAWN_OBJECT_XY[0] - float(dist) * float(dxy[0]),
      TABLE_SPAWN_OBJECT_XY[1] - float(dist) * float(dxy[1]),
  )
  # Paddle faces the cube / bucket (+xy toward bucket, slight down).
  z_raw = np.array([dxy[0], dxy[1], -0.85], dtype=np.float64)
  z_raw = z_raw / (np.linalg.norm(z_raw) + 1e-12)
  z_des = (float(z_raw[0]), float(z_raw[1]), float(z_raw[2]))

  obj = task.object_init_state[:, 0:3].clone()
  stand = obj.clone()
  stand[:, 0] = float(palm_xy[0])
  stand[:, 1] = float(palm_xy[1])
  stand[:, 2] = TABLE_OBJECT_Z + float(above)
  hover = stand.clone()
  hover[:, 2] = TABLE_OBJECT_Z + 0.18

  task._stow_object_for_ik()
  task.compute_observations()
  task._ik_palm_stage(
      jac, hover, n_iters=220, step=0.18, lam=0.12,
      name='hover', stop_at=0.04)
  task._ik_palm_stage(
      jac, hover, n_iters=220, step=0.12, lam=0.12,
      name='face-bucket', stop_at=0.05, use_orn=True, orn_w=0.40,
      stop_ang=0.35, z_des_xyz=z_des)
  last_err = task._ik_palm_stage(
      jac, stand, n_iters=220, step=0.10, lam=0.10,
      name='standoff', stop_at=0.03, use_orn=True, orn_w=0.30,
      stop_ang=0.40, z_des_xyz=z_des)

  fq = torch.tensor(
      TABLE_SPAWN_BEHIND_FINGER_Q, dtype=torch.float, device=task.device)
  task.arm_hand_dof_pos[:, 7:23] = fq
  task.cur_targets[:, 7:23] = fq
  task.prev_targets[:, 7:23] = fq
  task.hand_arm_default_dof_pos[:7] = task.arm_hand_dof_pos[0, :7].detach()
  task.hand_arm_default_dof_pos[7:23] = fq
  task._ik_apply_q(task.arm_hand_dof_pos[:, :7])
  task._set_object_root(None)
  for _ in range(8):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._set_object_root(None)
  task.compute_observations()
  return last_err, palm_xy, z_des


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
  goal = np.asarray(_bucket_goal_xyz(), dtype=np.float64)
  bucket = np.asarray(DEFAULT_FIXED_TARGET_XYZ, dtype=np.float64)
  obj = env._object_xyz()[0].detach().cpu().numpy()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  tol = 0.075
  ring = []
  for k in range(48):
    ang = 2.0 * math.pi * k / 48.0
    xyz = goal + np.array([tol * math.cos(ang), tol * math.sin(ang), 0.0])
    uv = _project(xyz, view, proj, w, h)
    if uv is not None:
      ring.append(uv)
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  if len(ring) >= 3:
    poly = [(int(u), int(v)) for u, v in ring]
    draw.polygon(poly, fill=(255, 140, 40, 70), outline=(255, 180, 50, 230))
    draw.line(poly + poly[:1], fill=(255, 200, 60, 255), width=4)
  o_uv = _project(obj, view, proj, w, h)
  g_uv = _project(goal, view, proj, w, h)
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

  _dot(g_uv, (255, 170, 50, 255), 'goal (in bucket)')
  _dot(_project(bucket, view, proj, w, h), (180, 120, 255, 255), 'bucket')
  _dot(o_uv, (240, 240, 240, 255), 'cube')
  _dot(_project(palm, view, proj, w, h), (80, 220, 255, 255), 'palm')
  bar_h = 44
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
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'videos', 'allegro_kuka_throw', 'bucket_init_candidates'))
  p.add_argument('--pipeline', default='gpu')
  args = p.parse_args()

  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=0,
      episode_length=300,
      table_spawn=True,
      table_spawn_behind=False,
      table_push=False,
      palm_goal=False,
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
  task = env._env
  jac = _acquire_jacobian(task)
  # Keep a copy of the original touch IK so we can restore `current`.
  q_touch = task.hand_arm_default_dof_pos.detach().clone()

  for name, spec in STANDOFFS.items():
    if spec is None:
      task.hand_arm_default_dof_pos[:] = q_touch
      _restore_noiseless(env)
      caption = (
          'current: palm on cube (table_spawn touch) — throw away from bucket')
      print(f'[{name}] restored touch IK', flush=True)
    else:
      dist, above = spec
      err, palm_xy, z_des = _ik_standoff_toward_bucket(
          task, jac, dist, above)
      palm = env._palm_xyz()[0].detach().cpu().numpy()
      obj = env._object_xyz()[0].detach().cpu().numpy()
      dist_po = float(np.linalg.norm(palm - obj))
      caption = (
          f'{name}: standoff={dist:.2f}m above={above:.2f}m  '
          f'|palm-obj|={dist_po:.3f}m  palm_xy=({palm_xy[0]:.2f},{palm_xy[1]:.2f})')
      print(
          f'[{name}] ik_err={err:.3f}m |palm-obj|={dist_po:.3f}m '
          f'z_des={tuple(round(v, 3) for v in z_des)}',
          flush=True)

    _set_cam(env, OBLIQUE_EYE, OBLIQUE_TGT)
    rgb = env.render_rgb()
    if rgb is None:
      print(f'[{name}] camera returned no frame', flush=True)
      return 1
    _save(os.path.join(args.out_dir, f'{name}_oblique.png'),
          _overlay(rgb, env, caption))
    _set_cam(env, TOP_EYE, TOP_TGT)
    rgb = env.render_rgb()
    _save(os.path.join(args.out_dir, f'{name}_top.png'),
          _overlay(rgb, env, caption + '   (top)'))

  print(f'done → {args.out_dir}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
