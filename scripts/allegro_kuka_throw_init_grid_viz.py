#!/usr/bin/env python3
"""Render several table-spawn resets (hover + finger/arm joint noise).

  python scripts/allegro_kuka_throw_init_grid_viz.py \
    --finger-noise=0.10 --arm-noise=0.15 --n=8
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
    AllegroKukaThrowVecEnv,
)

EPISODE_T = 50
# Pulled back so a ±15% arm jitter still keeps the hand in frame.
WIDE_EYE = (1.35, 0.95, 1.55)
WIDE_TGT = (0.12, 0.05, 0.55)
HFOV = 48.0
WH = (960, 720)


def _set_cam(env, eye, tgt, hfov=None) -> None:
  from isaacgym import gymapi

  env._cam_eye = tuple(float(v) for v in eye)
  env._cam_tgt = tuple(float(v) for v in tgt)
  env._cam_hfov = float(hfov if hfov is not None else HFOV)
  env._env.gym.set_camera_location(
      env._cam_handle, env._env.envs[0],
      gymapi.Vec3(*env._cam_eye), gymapi.Vec3(*env._cam_tgt))


def _settle(env, n: int = 6) -> None:
  task = env._env
  for _ in range(int(n)):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
  task.compute_observations()


def _caption(rgb, lines) -> np.ndarray:
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  draw.rectangle((0, 0, img.width, 52), fill=(0, 0, 0, 170))
  try:
    font = ImageFont.truetype(
        '/usr/share/fonts/dejavu/DejaVuSans.ttf', 16)
  except OSError:
    font = ImageFont.load_default()
  y = 4
  for line in lines:
    draw.text((8, y), line, fill=(255, 255, 255, 255), font=font)
    y += 22
  return np.asarray(img)


def _grid(frames, cols: int = 4) -> Image.Image:
  h, w = frames[0].shape[:2]
  rows = (len(frames) + cols - 1) // cols
  canvas = Image.new('RGB', (cols * w, rows * h), (20, 20, 20))
  for i, fr in enumerate(frames):
    r, c = divmod(i, cols)
    canvas.paste(Image.fromarray(fr), (c * w, r * h))
  return canvas


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'armrand15_inits'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--n', type=int, default=8)
  p.add_argument('--curl', type=float, default=0.15)
  p.add_argument('--finger-noise', type=float, default=0.10)
  p.add_argument('--arm-noise', type=float, default=0.15)
  p.add_argument('--seed', type=int, default=0)
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=int(args.seed),
      episode_length=EPISODE_T,
      table_spawn=True,
      table_spawn_behind=True,
      table_spawn_behind_dy=0.0,
      table_spawn_behind_above=0.12,
      table_spawn_correlated_xy=0.03,
      table_spawn_finger_curl_scale=float(args.curl),
      table_spawn_finger_noise=float(args.finger_noise),
      table_spawn_arm_noise=float(args.arm_noise),
      table_push=True,
      table_push_xyz=TABLE_SIDE_GOAL_XYZ,
      large_table=True,
      randomize_init=True,
      randomize_object_shape=False,
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=WIDE_EYE,
      camera_tgt=WIDE_TGT,
      camera_hfov=HFOV,
      pipeline=args.pipeline,
  )
  _set_cam(env, WIDE_EYE, WIDE_TGT, hfov=HFOV)
  frames = []
  for i in range(int(args.n)):
    env.reset()
    _settle(env)
    rgb = env.render_rgb()
    if rgb is None:
      print('camera returned no frame', flush=True)
      return 1
    palm = env._palm_xyz()[0].detach().cpu().numpy()
    obj = env._object_xyz()[0].detach().cpu().numpy()
    dist = float(np.linalg.norm(palm - obj))
    labeled = _caption(rgb, [
        (f'init {i}  curl={args.curl:g}  finger±{args.finger_noise:g}  '
         f'arm±{args.arm_noise:g}'),
        (f'palm=({palm[0]:.2f},{palm[1]:.2f},{palm[2]:.2f})  '
         f'obj=({obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f})  '
         f'|palm-obj|={dist:.3f}m  cube_z={TABLE_OBJECT_Z:.3f}'),
    ])
    path = os.path.join(args.out_dir, f'init_{i:02d}.png')
    Image.fromarray(labeled).save(path)
    frames.append(labeled)
    print(f'wrote {path}  |palm-obj|={dist:.3f}m', flush=True)
  grid_path = os.path.join(args.out_dir, 'init_grid.png')
  _grid(frames, cols=4).save(grid_path)
  print(f'wrote {grid_path}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
