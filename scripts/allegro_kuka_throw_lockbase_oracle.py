#!/usr/bin/env python3
"""Prove wrist+fingers (A1–A4 locked) can hit the 7.5 cm object goal.

Sweeps constant A5/A6/A7 actions × finger-release times. Not a train.

  python scripts/allegro_kuka_throw_lockbase_oracle.py
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    IN_HAND_KEEP_ARM_GOAL_XYZ,
    TABLE_SPAWN_IN_HAND_OFFSET,
    TABLE_SPAWN_IN_HAND_WRIST_OFFSET,
    AllegroKukaThrowVecEnv,
)

WH = (960, 720)
HFOV = 48.0
WRIST_VALS = (-1.0, -0.5, 0.0, 0.5, 1.0)
OPEN_AT = (999, 8, 16)  # 999 = never open
HOLD_STEPS = 4
TOL = 0.075
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


def _recipes():
  return list(itertools.product(WRIST_VALS, WRIST_VALS, WRIST_VALS, OPEN_AT))


def _actions_for_t(env, t, hold, wrist, open_at):
  """(E, 19) lock-arm actions: hold, then wrist flick, optional finger open."""
  import torch

  a = hold.clone()
  if t >= HOLD_STEPS:
    a[:, 0:3] = wrist
    open_now = open_at <= t
    if bool(open_now.any()):
      a[open_now, 3:] = 0.0
  return a


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'inhand_keep_arm',
      'lockbase_oracle'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--ep-len', type=int, default=50)
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)
  recipes = _recipes()
  n = len(recipes)
  print(f'[oracle] n_recipes={n} wrist={WRIST_VALS} open_at={OPEN_AT} '
        f'hold={HOLD_STEPS} tol={TOL}', flush=True)
  env = AllegroKukaThrowVecEnv(
      num_envs=n,
      seed=0,
      episode_length=int(args.ep_len),
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
      lock_arm_base=True,
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
  assert int(env.action_dim) == 19, env.action_dim
  env.reset()
  import torch

  hold = env._hold_pose_actions()
  wrist = torch.zeros((n, 3), dtype=torch.float, device=env.device)
  open_at = torch.full((n,), 999, dtype=torch.int32, device=env.device)
  for i, (a5, a6, a7, ot) in enumerate(recipes):
    wrist[i, 0] = float(a5)
    wrist[i, 1] = float(a6)
    wrist[i, 2] = float(a7)
    open_at[i] = int(ot)
  goal = np.asarray(IN_HAND_KEEP_ARM_GOAL_XYZ, dtype=np.float32)
  min_dist = np.full((n,), 1e9, dtype=np.float32)
  best_t = np.zeros((n,), dtype=np.int32)
  hit = np.zeros((n,), dtype=np.bool_)
  t0_obj = env._object_xyz().detach().cpu().numpy()
  t0_dist = np.linalg.norm(t0_obj - goal[None, :], axis=-1)
  print(
      f'[oracle] t0 |obj-g| mean={float(t0_dist.mean()):.3f} '
      f'min={float(t0_dist.min()):.3f} action_dim={env.action_dim}',
      flush=True)
  for t in range(int(args.ep_len)):
    obj = env._object_xyz().detach().cpu().numpy()
    dist = np.linalg.norm(obj - goal[None, :], axis=-1)
    improved = dist < min_dist
    min_dist = np.minimum(min_dist, dist)
    best_t[improved] = t
    succ = env.success().detach().cpu().numpy().astype(bool)
    hit |= succ
    env.step(_actions_for_t(env, t, hold, wrist, open_at))
  n_hit = int(hit.sum())
  best_i = int(np.argmin(min_dist))
  rec = recipes[best_i]
  print(
      f'[oracle] hits={n_hit}/{n}  best_i={best_i} '
      f'A5={rec[0]:g} A6={rec[1]:g} A7={rec[2]:g} open_at={rec[3]} '
      f'min|obj-g|={min_dist[best_i]*100:.1f}cm at t={int(best_t[best_i])}',
      flush=True)
  order = np.argsort(min_dist)[:8]
  print('[oracle] top-8 min|obj-g|:', flush=True)
  for i in order:
    r = recipes[int(i)]
    print(
        f'  {min_dist[i]*100:5.1f}cm  hit={int(hit[i])}  '
        f'A5={r[0]:+.1f} A6={r[1]:+.1f} A7={r[2]:+.1f} open={r[3]}',
        flush=True)
  # Replay the winner on env 0 for stills (same sim; copy winner recipe).
  env.reset()
  hold = env._hold_pose_actions()
  w0 = torch.zeros((n, 3), dtype=torch.float, device=env.device)
  o0 = torch.full((n,), 999, dtype=torch.int32, device=env.device)
  w0[:, :] = wrist[best_i]
  o0[:] = int(rec[3])
  still_ts = {0, int(best_t[best_i]), 15, 30, 49}
  for t in range(int(args.ep_len)):
    if t in still_ts:
      obj = env._object_xyz()[0].detach().cpu().numpy()
      d = float(np.linalg.norm(obj - goal))
      _set_cam(env, *VIEWS['side'])
      rgb = env.render_rgb()
      if rgb is not None:
        labeled = _caption(rgb, [
            (f'lockbase oracle  best A5={rec[0]:g} A6={rec[1]:g} '
             f'A7={rec[2]:g} open_at={rec[3]}'),
            (f't={t}  cube=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})  '
             f'|obj-g|={d*100:.1f}cm  goal={tuple(goal)}'),
        ])
        path = os.path.join(args.out_dir, f'best_t{t:02d}.png')
        Image.fromarray(labeled).save(path)
        print(f'wrote {path}', flush=True)
    env.step(_actions_for_t(env, t, hold, w0, o0))
  out_txt = os.path.join(args.out_dir, 'oracle_summary.txt')
  with open(out_txt, 'w') as f:
    f.write(
        f'hits={n_hit}/{n}\n'
        f'best_A5={rec[0]} A6={rec[1]} A7={rec[2]} open_at={rec[3]}\n'
        f'best_min_dist_m={float(min_dist[best_i]):.4f}\n'
        f'best_t={int(best_t[best_i])}\n')
  if n_hit < 1:
    print(
        f'[oracle] FAIL no recipe reached {TOL*100:.1f}cm '
        f'(best {min_dist[best_i]*100:.1f}cm)',
        flush=True)
    return 2
  print(
      f'[oracle] OK {n_hit} recipes reached {TOL*100:.1f}cm',
      flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
