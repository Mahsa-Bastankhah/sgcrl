#!/usr/bin/env python3
"""Stills of tableside in-hand init: cube in a curled palm, small jitter.

Creates 4 vectorized envs so correlated xy gives 4 different hand poses,
copies each onto env 0 for the camera, and writes close + oblique frames
plus a grid.

  python scripts/allegro_kuka_throw_tableside_inhand_init_viz.py
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
    TABLE_OBJECT_Z,
    TABLE_SIDE_GOAL_XYZ,
    TABLE_SPAWN_IN_HAND_ABOVE,
    TABLE_SPAWN_IN_HAND_OBJ_NOISE,
    TABLE_SPAWN_IN_HAND_OFFSET,
    AllegroKukaThrowVecEnv,
)

EPISODE_T = 50
N_INITS = 4
CORRELATED_XY = 0.03
FINGER_NOISE = 0.04
CURL = 1.0
WH = (960, 720)
# Isaac Gym locks HFOV at camera create. Pull back so the arm, hand,
# cube, and nearby table are in frame (the 0.3 m / 28° shots were a
# cube close-up).
CAM_HFOV = 42.0
SIDE_OFF = np.array([0.18, -0.62, 0.28], dtype=np.float64)
THREE_OFF = np.array([0.48, -0.42, 0.36], dtype=np.float64)


def _font(size: int):
  try:
    return ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', size)
  except OSError:
    return ImageFont.load_default()


def _set_cam(env, eye, tgt, hfov) -> None:
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


def _snapshot_q(env):
  return env._env.arm_hand_dof_pos.detach().clone()


def _show_src(env, src: int, q_all) -> None:
  """Copy env ``src`` joints onto env 0 and snap the cube into that palm."""
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
  dist = float(np.linalg.norm(palm - obj))
  return [
      (f'in-hand init {i}  {view}  curl={CURL:g}  '
       f'hand xy ±{CORRELATED_XY*100:.0f}cm  '
       f'obj ±{TABLE_SPAWN_IN_HAND_OBJ_NOISE*100:.1f}cm  '
       f'finger±{FINGER_NOISE:g}'),
      (f'palm=({palm[0]:.3f},{palm[1]:.3f},{palm[2]:.3f})  '
       f'obj=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})  '
       f'|palm-obj|={dist*100:.1f}cm  '
       f'goal={TABLE_SIDE_GOAL_XYZ}  table_z={TABLE_OBJECT_Z:.3f}'),
  ]


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'tableside_inhand_init'))
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
      table_spawn_behind=True,
      table_spawn_behind_dy=0.0,
      table_spawn_behind_above=TABLE_SPAWN_IN_HAND_ABOVE,
      table_spawn_correlated_xy=CORRELATED_XY,
      table_spawn_finger_curl_scale=CURL,
      table_spawn_finger_noise=FINGER_NOISE,
      table_spawn_arm_noise=0.0,
      table_spawn_in_hand=True,
      table_spawn_in_hand_offset=TABLE_SPAWN_IN_HAND_OFFSET,
      table_spawn_in_hand_obj_noise=TABLE_SPAWN_IN_HAND_OBJ_NOISE,
      table_push=True,
      table_push_xyz=TABLE_SIDE_GOAL_XYZ,
      large_table=True,
      randomize_init=True,
      randomize_object_shape=False,
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=(0.36, -0.52, 0.90),
      camera_tgt=(0.18, 0.10, 0.60),
      camera_hfov=CAM_HFOV,
      pipeline=args.pipeline,
  )
  env.reset()
  q_all = _snapshot_q(env)
  side_frames = []
  for i in range(n):
    _show_src(env, i, q_all)
    palm = env._palm_xyz()[0].detach().cpu().numpy()
    obj = env._object_xyz()[0].detach().cpu().numpy()
    dist = float(np.linalg.norm(palm - obj))
    print(
        f'[inhand_init] env {i}  palm={tuple(round(float(v), 3) for v in palm)}  '
        f'obj={tuple(round(float(v), 3) for v in obj)}  '
        f'|palm-obj|={dist:.3f}m',
        flush=True)
    side_eye = palm.astype(np.float64) + SIDE_OFF
    three_eye = palm.astype(np.float64) + THREE_OFF
    _set_cam(env, side_eye, palm, CAM_HFOV)
    rgb = env.render_rgb()
    if rgb is None:
      print('camera returned no frame', flush=True)
      return 1
    side = _caption(rgb, _lines(i, env, 'side'))
    side_path = os.path.join(args.out_dir, f'inhand_init_{i:02d}_side.png')
    Image.fromarray(side).save(side_path)
    print(f'wrote {side_path}', flush=True)
    side_frames.append(side)
    _set_cam(env, three_eye, palm, CAM_HFOV)
    rgb = env.render_rgb()
    three = _caption(rgb, _lines(i, env, 'three-quarter'))
    three_path = os.path.join(args.out_dir, f'inhand_init_{i:02d}_three.png')
    Image.fromarray(three).save(three_path)
    print(f'wrote {three_path}', flush=True)
  grid_path = os.path.join(args.out_dir, 'inhand_init_grid_side.png')
  _grid(side_frames, cols=2).save(grid_path)
  print(f'wrote {grid_path}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
