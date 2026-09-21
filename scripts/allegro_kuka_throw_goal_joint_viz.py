#!/usr/bin/env python3
"""IK a pushing pose *at the table-side goal* and dump the 23-D joint q.

Cube sits on the orange success disk. Palm is on the robot (+y) side, facing
−y (paddle), same finger curl as table_spawn_behind. No palm xyz packing —
this is only to pick a q* you like.

  python scripts/allegro_kuka_throw_goal_joint_viz.py --variant=standoff14
  python scripts/allegro_kuka_throw_goal_joint_viz.py --variant=tight10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    TABLE_OBJECT_Z,
    TABLE_SIDE_GOAL_XYZ,
    TABLE_SIDE_JOINT_GOAL_Q,
    TABLE_SPAWN_BEHIND_Z_DES,
    AllegroKukaThrowVecEnv,
)

# Cube at (0.20, −0.15); palm ~14 cm toward +y. Slight top tilt so look-at
# is not parallel to world-up (Isaac Gym returns a black frame if it is).
OBLIQUE_EYE = (0.58, -0.44, 0.86)
OBLIQUE_TGT = (0.20, -0.08, 0.57)
TOP_EYE = (0.28, -0.22, 1.05)
TOP_TGT = (0.20, -0.08, 0.56)
HFOV = 38.0
WH = (960, 720)

# (name, palm +y behind cube, palm z above table-object center)
VARIANTS = {
    'standoff14': (0.14, 0.08),  # same stand-off as table_spawn_behind, at the goal
    'tight10': (0.10, 0.05),
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


def _acquire_jacobian(task):
  from isaacgym import gymtorch

  raw = task.gym.acquire_jacobian_tensor(task.sim, 'allegro')
  jac = gymtorch.wrap_tensor(raw)
  print(f'[goal_joint_viz] jacobian {tuple(jac.shape)}', flush=True)
  return jac


def _seed_nvidia_default(task, q_seed) -> None:
  from isaacgym import gymtorch

  task.arm_hand_dof_pos[:, :] = q_seed
  task.cur_targets[:, :] = q_seed
  task.prev_targets[:, :] = q_seed
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))
  for _ in range(4):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)


def _ik_behind_goal(task, jac, goal_xyz, dy: float, above: float):
  """Waypoint DLS: hover, face −y, stand behind the cube at the goal."""
  import torch

  goal = torch.tensor(
      [list(goal_xyz)], dtype=torch.float, device=task.device)
  stand = goal.clone()
  stand[:, 1] = stand[:, 1] + float(dy)
  stand[:, 2] = TABLE_OBJECT_Z + float(above)
  hover = stand.clone()
  hover[:, 2] = TABLE_OBJECT_Z + 0.18
  z_des = TABLE_SPAWN_BEHIND_Z_DES
  task._stow_object_for_ik()
  task.compute_observations()
  task._ik_palm_stage(
      jac, hover, n_iters=220, step=0.18, lam=0.12,
      name='hover', stop_at=0.04)
  task._ik_palm_stage(
      jac, hover, n_iters=220, step=0.12, lam=0.12,
      name='face-goal', stop_at=0.05, use_orn=True, orn_w=0.40,
      stop_ang=0.35, z_des_xyz=z_des)
  last_err = task._ik_palm_stage(
      jac, stand, n_iters=220, step=0.10, lam=0.10,
      name='behind-goal', stop_at=0.03, use_orn=True, orn_w=0.30,
      stop_ang=0.40, z_des_xyz=z_des)
  fq = torch.tensor(
      TABLE_SIDE_JOINT_GOAL_Q[7:23], dtype=torch.float, device=task.device)
  task.arm_hand_dof_pos[:, 7:23] = fq
  task.cur_targets[:, 7:23] = fq
  task.prev_targets[:, 7:23] = fq
  task._ik_apply_q(task.arm_hand_dof_pos[:, :7])
  task._set_object_root(goal.expand(task.num_envs, 3))
  for _ in range(8):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._set_object_root(goal.expand(task.num_envs, 3))
  task.compute_observations()
  return last_err


def _set_cam(env, eye, tgt) -> None:
  from isaacgym import gymapi

  task = env._env
  env._cam_eye = tuple(float(v) for v in eye)
  env._cam_tgt = tuple(float(v) for v in tgt)
  env._cam_hfov = float(HFOV)
  task.gym.set_camera_location(
      env._cam_handle, task.envs[0],
      gymapi.Vec3(*env._cam_eye), gymapi.Vec3(*env._cam_tgt))


def _overlay(rgb, env, goal_xyz, caption: str):
  from PIL import Image, ImageDraw

  h, w = rgb.shape[:2]
  view = _look_at(env._cam_eye, env._cam_tgt)
  proj = _perspective(HFOV, float(w) / max(float(h), 1.0))
  goal = np.asarray(goal_xyz, dtype=np.float64)
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

  def _dot(uv, fill, label, dy=-18):
    if uv is None:
      return
    u, v = int(uv[0]), int(uv[1])
    draw.ellipse([u - 7, v - 7, u + 7, v + 7], fill=fill,
                 outline=(255, 255, 255, 255))
    draw.text((u + 10, v + dy), label, fill=fill)

  _dot(_project(goal, view, proj, w, h), (255, 170, 50, 255), 'goal')
  _dot(_project(obj, view, proj, w, h), (240, 240, 240, 255), 'cube')
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


def _fmt_q(q) -> str:
  return ', '.join(f'{float(v):.6f}' for v in q)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--variant', choices=tuple(VARIANTS.keys()), default='standoff14')
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--goal', choices=('tableside', 'center'), default='tableside')
  p.add_argument('--large-table', action='store_true')
  args = p.parse_args()
  dy, above = VARIANTS[args.variant]
  goal_xyz = (
      (0.0, 0.0, TABLE_OBJECT_Z)
      if args.goal == 'center' else TABLE_SIDE_GOAL_XYZ)
  goal_x, goal_y, _ = goal_xyz
  oblique_eye = (goal_x + 0.38, goal_y - 0.29, 0.86)
  oblique_tgt = (goal_x, goal_y + 0.07, 0.57)
  top_eye = (goal_x + 0.08, goal_y - 0.07, 1.05)
  top_tgt = (goal_x, goal_y + 0.07, 0.56)

  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=0,
      episode_length=300,
      table_spawn=False,
      table_push=True,
      table_push_xyz=goal_xyz,
      palm_goal=False,
      randomize_init=False,
      randomize_object_shape=False,
      large_table=args.large_table,
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=oblique_eye,
      camera_tgt=oblique_tgt,
      camera_hfov=HFOV,
      pipeline=args.pipeline,
  )
  env.reset()
  task = env._env
  q_seed = task.hand_arm_default_dof_pos.detach().clone()
  jac = _acquire_jacobian(task)
  _seed_nvidia_default(task, q_seed)
  ik_err = _ik_behind_goal(task, jac, goal_xyz, dy, above)

  q = task.arm_hand_dof_pos[0].detach().cpu().numpy()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  obj = env._object_xyz()[0].detach().cpu().numpy()
  dist = float(np.linalg.norm(palm - obj))
  os.makedirs(args.out_dir, exist_ok=True)
  suffix = (
      f'{args.goal}_{"large_table_" if args.large_table else ""}'
      f'{args.variant}')
  q_path = os.path.join(args.out_dir, f'goal_joint_{suffix}_q.txt')
  with open(q_path, 'w', encoding='utf-8') as fh:
    fh.write(f'variant={args.variant}  dy={dy:.3f}  above={above:.3f}\n')
    fh.write(f'object_goal={tuple(float(v) for v in goal_xyz)}\n')
    fh.write(f'palm={tuple(round(float(v), 4) for v in palm)}\n')
    fh.write(f'obj={tuple(round(float(v), 4) for v in obj)}\n')
    fh.write(f'|palm-obj|={dist:.4f}  ik_err={ik_err:.4f}\n')
    fh.write(f'arm_q  = [{_fmt_q(q[:7])}]\n')
    fh.write(f'hand_q = [{_fmt_q(q[7:23])}]\n')
    fh.write(f'q23    = [{_fmt_q(q)}]\n')
  print(open(q_path, encoding='utf-8').read(), flush=True)

  caption = (
      f'q* behind goal  {args.variant}  '
      f'dy=+{dy:.2f}m  z=+{above:.2f}m  |palm-obj|={dist:.3f}m')
  _set_cam(env, oblique_eye, oblique_tgt)
  rgb = env.render_rgb()
  if rgb is None:
    print('camera returned no frame', flush=True)
    return 1
  _save(os.path.join(args.out_dir, f'goal_joint_{suffix}_oblique.png'),
        _overlay(rgb, env, goal_xyz, caption))
  _set_cam(env, top_eye, top_tgt)
  rgb = env.render_rgb()
  _save(os.path.join(args.out_dir, f'goal_joint_{suffix}_top.png'),
        _overlay(rgb, env, goal_xyz, caption + '   (top view)'))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
