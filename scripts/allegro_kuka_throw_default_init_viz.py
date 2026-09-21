#!/usr/bin/env python3
"""Stills of the NVIDIA throw-task default reset and bucket goal.

No table_spawn / in-hand. Arm is pose v1 over the narrow table, fingers
open (0). Init: cube at NVIDIA table-center spawn. Goal: same arm, cube
teleported into the fixed bucket (success xyz = bucket + 5 cm z).

  python scripts/allegro_kuka_throw_default_init_viz.py
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
    DEFAULT_FIXED_TARGET_XYZ,
    OBJECT_ABOVE_BUCKET,
    AllegroKukaThrowVecEnv,
)

EPISODE_T = 300
WH = (960, 720)
HFOV = 48.0
# Wide enough for table-center cube (0, 0, ~0.63) and bucket (0.5, -0.3, 0.4).
VIEWS = {
    'oblique': ((1.25, -1.15, 1.35), (0.20, -0.12, 0.48)),
    'side': ((1.45, 0.05, 1.05), (0.20, -0.10, 0.48)),
    'front': ((0.20, -1.35, 1.10), (0.20, -0.05, 0.48)),
    # Offset look-down so the cube inside the bucket is visible.
    'into_bucket': ((0.78, -0.68, 1.15), (0.50, -0.30, 0.42)),
}
POSE_V1 = (-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 2.358)
GOAL_XYZ = (
    DEFAULT_FIXED_TARGET_XYZ[0],
    DEFAULT_FIXED_TARGET_XYZ[1],
    DEFAULT_FIXED_TARGET_XYZ[2] + OBJECT_ABOVE_BUCKET,
)


def _font(size: int):
  try:
    return ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', size)
  except OSError:
    return ImageFont.load_default()


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


def _hold_arm(env) -> None:
  from isaacgym import gymtorch

  task = env._env
  task.arm_hand_dof_pos[:, :] = task.hand_arm_default_dof_pos
  task.arm_hand_dof_vel[:, :] = 0.0
  task.cur_targets[:, :] = task.hand_arm_default_dof_pos
  task.prev_targets[:, :] = task.hand_arm_default_dof_pos
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))


def _hold_default(env) -> None:
  task = env._env
  _hold_arm(env)
  task._set_object_root(None)
  for _ in range(8):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._set_object_root(None)
    _hold_arm(env)
  task.compute_observations()


def _place_cube_in_bucket(env) -> None:
  import torch

  task = env._env
  xyz = torch.tensor(
      [GOAL_XYZ], dtype=torch.float, device=task.device).expand(
          task.num_envs, 3).contiguous()
  _hold_arm(env)
  task._set_object_root(xyz)
  for _ in range(8):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    task._set_object_root(xyz)
    _hold_arm(env)
  task.compute_observations()


def _scene_stats(env):
  task = env._env
  arm = tuple(
      round(float(v), 3)
      for v in task.hand_arm_default_dof_pos[:7].tolist())
  fingers = tuple(
      round(float(v), 3)
      for v in task.hand_arm_default_dof_pos[7:23].tolist())
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  obj = env._object_xyz()[0].detach().cpu().numpy()
  return arm, fingers, palm, obj


def _write_views(env, out_dir, stem, lines):
  paths = []
  for name, (eye, tgt) in VIEWS.items():
    _set_cam(env, eye, tgt)
    rgb = env.render_rgb()
    if rgb is None:
      raise RuntimeError('camera returned no frame')
    labeled = _caption(rgb, [f'{lines[0]}  {name}', lines[1]])
    path = os.path.join(out_dir, f'{stem}_{name}.png')
    Image.fromarray(labeled).save(path)
    print(f'wrote {path}', flush=True)
    paths.append(path)
  return paths


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'throw_default_init_goal'))
  p.add_argument('--pipeline', default='gpu')
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=0,
      episode_length=EPISODE_T,
      table_spawn=False,
      table_push=False,
      large_table=False,
      randomize_init=False,
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
  _hold_default(env)
  arm, fingers, palm, obj = _scene_stats(env)
  print(
      f'[throw_default] INIT arm={arm}\n'
      f'  fingers={fingers}\n'
      f'  palm={tuple(round(float(v), 3) for v in palm)}\n'
      f'  obj={tuple(round(float(v), 3) for v in obj)}\n'
      f'  want_arm={POSE_V1}  bucket={DEFAULT_FIXED_TARGET_XYZ}  '
      f'goal={GOAL_XYZ}',
      flush=True)
  init_lines = [
      (f'NVIDIA throw INIT  pose v1  arm={arm}  '
       f'fingers=0 (open)  randomize_init=off'),
      (f'palm=({palm[0]:.3f},{palm[1]:.3f},{palm[2]:.3f})  '
       f'obj=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})  '
       f'cube on table center  bucket={DEFAULT_FIXED_TARGET_XYZ}'),
  ]
  init_paths = _write_views(env, args.out_dir, 'throw_default_init', init_lines)

  _place_cube_in_bucket(env)
  arm, fingers, palm, obj = _scene_stats(env)
  print(
      f'[throw_default] GOAL palm={tuple(round(float(v), 3) for v in palm)}  '
      f'obj={tuple(round(float(v), 3) for v in obj)}  want_goal={GOAL_XYZ}',
      flush=True)
  goal_lines = [
      (f'NVIDIA throw GOAL  same arm  cube in bucket  '
       f'success xyz={GOAL_XYZ}'),
      (f'palm=({palm[0]:.3f},{palm[1]:.3f},{palm[2]:.3f})  '
       f'obj=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})  '
       f'bucket={DEFAULT_FIXED_TARGET_XYZ}  +5cm z'),
  ]
  goal_paths = _write_views(env, args.out_dir, 'throw_default_goal', goal_lines)

  for init_path, goal_path in zip(init_paths, goal_paths):
    name = os.path.basename(init_path).split('_')[-1]
    init_im = np.asarray(Image.open(init_path))
    goal_im = np.asarray(Image.open(goal_path))
    pair = np.concatenate([init_im, goal_im], axis=1)
    pair_path = os.path.join(args.out_dir, f'throw_default_init_goal_{name}')
    Image.fromarray(pair).save(pair_path)
    print(f'wrote {pair_path}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
