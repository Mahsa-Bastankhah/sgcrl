#!/usr/bin/env python3
"""Hold the keep-arm pose for 30 steps; cube must stay in the palm.

No policy. Actions are the authored hold. Writes stills at t=0/10/30.

  python scripts/allegro_kuka_throw_inhand_hold_stability.py
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
    TABLE_SPAWN_IN_HAND_OFFSET,
    TABLE_SPAWN_IN_HAND_OBJ_NOISE,
    TABLE_SPAWN_IN_HAND_WRIST_NOISE,
    TABLE_SPAWN_IN_HAND_WRIST_OFFSET,
    AllegroKukaThrowVecEnv,
)

HOLD_STEPS = 30
WH = (960, 720)
HFOV = 48.0
VIEWS = {
    'side': ((1.30, -0.08, 1.00), (0.00, -0.08, 0.58)),
}


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


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'inhand_keep_arm', 'hold_stable'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--steps', type=int, default=HOLD_STEPS)
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=0,
      episode_length=50,
      table_spawn=True,
      table_spawn_object_xy=(0.0, 0.0),
      table_spawn_in_hand=True,
      table_spawn_in_hand_keep_arm=True,
      table_spawn_in_hand_offset=TABLE_SPAWN_IN_HAND_OFFSET,
      table_spawn_in_hand_obj_noise=0.0,
      table_spawn_in_hand_wrist_offset=TABLE_SPAWN_IN_HAND_WRIST_OFFSET,
      table_spawn_in_hand_wrist_noise=0.0,
      table_spawn_finger_curl_scale=1.0,
      table_spawn_finger_noise=0.0,
      table_spawn_arm_noise=0.0,
      table_push=True,
      table_push_xyz=IN_HAND_KEEP_ARM_GOAL_XYZ,
      large_table=True,
      randomize_init=False,
      randomize_object_shape=False,
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=VIEWS['side'][0],
      camera_tgt=VIEWS['side'][1],
      camera_hfov=HFOV,
      pipeline=args.pipeline,
  )
  env.reset()
  hold = env._hold_pose_actions()
  dists = []
  snapshots = (0, 10, int(args.steps))
  for t in range(int(args.steps) + 1):
    palm = env._palm_xyz()[0].detach().cpu().numpy()
    obj = env._object_xyz()[0].detach().cpu().numpy()
    dist = float(np.linalg.norm(palm - obj))
    dists.append(dist)
    print(
        f'[hold] t={t:02d}  palm=({palm[0]:.3f},{palm[1]:.3f},{palm[2]:.3f})  '
        f'cube=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})  '
        f'|palm-cube|={dist*100:.1f}cm',
        flush=True)
    if t in snapshots:
      _set_cam(env, *VIEWS['side'])
      rgb = env.render_rgb()
      if rgb is None:
        return 1
      labeled = _caption(rgb, [
          f'hold-pose only  t={t}/{args.steps}  NO policy',
          (f'cube=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})  '
           f'|palm-cube|={dist*100:.1f}cm  '
           f'goal={IN_HAND_KEEP_ARM_GOAL_XYZ}'),
      ])
      path = os.path.join(args.out_dir, f'hold_t{t:02d}_side.png')
      Image.fromarray(labeled).save(path)
      print(f'wrote {path}', flush=True)
    if t < int(args.steps):
      env.step(hold)
  d0, d30 = dists[0], dists[-1]
  print(
      f'[hold] |palm-cube| t0={d0*100:.1f}cm  t{args.steps}={d30*100:.1f}cm',
      flush=True)
  if d30 > 0.08:
    print(
        f'[hold] FAIL cube left the palm ({d30:.3f}m)',
        flush=True)
    return 2
  print('[hold] OK cube stayed in the palm under hold actions', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
