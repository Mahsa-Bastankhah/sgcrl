#!/usr/bin/env python3
"""Can *some* motion put the cube in the candidate buckets?

Keep-arm pad-hold + NVIDIA table. For each xyz:
  1) DLS IK palm to a hover over the bucket (place test)
  2) snap cube back in the hand, interpolate start→IK-best while opening
     the fingers (scripted throw)
  3) current NVIDIA bucket is a positive control

  python scripts/allegro_kuka_throw_bucket_feasibility.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    BUCKET_INNER_RADIUS,
    TABLE_SPAWN_IN_HAND_OFFSET,
    TABLE_SPAWN_IN_HAND_WRIST_OFFSET,
    AllegroKukaThrowVecEnv,
)

# iiwa 7 R800 workspace radius (mount → wrist), not a hard sim limit.
IIWA_REACH = 0.80
BASE_XY = (0.0, 0.80)
HOVER_Z = 0.68
PLACE_OK = 0.08
WH = (960, 720)
HFOV = 58.0
CAM = ((1.55, -1.55, 1.70), (0.00, -0.22, 0.42))

LOCATIONS = (
    ('control_nvidia', (0.50, -0.30, 0.40), 'control  (0.50, -0.30)'),
    ('mirror_minus_x', (-0.50, -0.30, 0.40), 'proposed  (-0.50, -0.30)'),
    ('far_minus_y', (0.00, -0.80, 0.40), 'backup  (0.00, -0.80)'),
)


def _font(size: int):
  try:
    return ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf', size)
  except OSError:
    return ImageFont.load_default()


def _caption(rgb, lines):
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  bar_h = 8 + 20 * len(lines)
  canvas = np.full((img.height + bar_h, img.width, 3), 18, dtype=np.uint8)
  canvas[bar_h:] = np.asarray(img)
  out = Image.fromarray(canvas)
  draw = ImageDraw.Draw(out)
  font = _font(16)
  for i, line in enumerate(lines):
    draw.text((8, 4 + 20 * i), line, fill=(240, 240, 240), font=font)
  return out


def _set_cam(env, eye, tgt) -> None:
  from isaacgym import gymapi

  env._cam_eye = tuple(float(v) for v in eye)
  env._cam_tgt = tuple(float(v) for v in tgt)
  env._cam_hfov = HFOV
  env._env.gym.set_camera_location(
      env._cam_handle, env._env.envs[0],
      gymapi.Vec3(*env._cam_eye), gymapi.Vec3(*env._cam_tgt))


def _move_bucket(env, xyz) -> None:
  import torch
  from isaacgym import gymtorch

  task = env._env
  tgt = torch.tensor(xyz, dtype=torch.float, device=task.device)
  idx = task.bucket_object_indices
  task.root_state_tensor[idx, 0:3] = tgt
  task.root_state_tensor[idx, 7:13] = 0.0
  task.goal_states[:, 0:3] = tgt
  task.goal_states[:, 2] = float(xyz[2]) + 0.05
  i32 = idx.to(torch.int32)
  task.gym.set_actor_root_state_tensor_indexed(
      task.sim,
      gymtorch.unwrap_tensor(task.root_state_tensor),
      gymtorch.unwrap_tensor(i32),
      int(task.num_envs),
  )
  task.gym.simulate(task.sim)
  task.gym.fetch_results(task.sim, True)
  task.compute_observations()


def _set_arm_q(task, q) -> None:
  from isaacgym import gymtorch

  task.cur_targets[:, :7] = q
  task.prev_targets[:, :7] = q
  task.arm_hand_dof_pos[:, :7] = q
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))


def _open_fingers(task) -> None:
  from isaacgym import gymtorch

  # Absolute finger targets: 0 is mid-range / open (see _hold_pose_actions).
  task.cur_targets[:, 7:] = 0.0
  task.prev_targets[:, 7:] = 0.0
  task.arm_hand_dof_pos[:, 7:] = 0.0
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))


def _sim(task, n: int) -> None:
  for _ in range(int(n)):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
  task.compute_observations()


def _in_bucket(env) -> bool:
  return bool(env.in_bucket()[0].item() > 0.5)


def _ik_hover(task, hover_xyz, n_iters=180) -> float:
  import torch
  from isaacgym import gymtorch

  try:
    raw = task.gym.acquire_jacobian_tensor(task.sim, 'allegro')
  except Exception as exc:
    raise RuntimeError(f'jacobian acquire failed: {exc}') from exc
  jac = gymtorch.wrap_tensor(raw)
  goal = torch.tensor(
      [list(hover_xyz)], dtype=torch.float, device=task.device)
  return task._ik_palm_stage(
      jac, goal, n_iters=n_iters, step=0.35, lam=0.08,
      name='feas_hover', stop_at=0.04, use_orn=True, orn_w=0.25,
      stop_ang=0.35, z_des_xyz=(0.0, 0.0, -1.0))


def _make_env(pipeline: str, seed: int) -> AllegroKukaThrowVecEnv:
  return AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=int(seed),
      episode_length=80,
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
      hide_table=False,
      large_table=False,
      randomize_init=False,
      randomize_object_shape=False,
      fixed_target_xyz=LOCATIONS[0][1],
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=CAM[0],
      camera_tgt=CAM[1],
      camera_hfov=HFOV,
      pipeline=pipeline,
  )


def _snap(env) -> None:
  env.reset()
  task = env._env
  ids = task.object_indices.new_zeros((task.num_envs,), dtype=task.object_indices.dtype)
  # object_indices is 1-d of env object ids; reset already placed the cube
  task.compute_observations()


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'bucket_feasibility'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--seed', type=int, default=0)
  args = p.parse_args()
  os.makedirs(args.out_dir, exist_ok=True)

  env = _make_env(args.pipeline, args.seed)
  env.reset()
  _set_cam(env, *CAM)
  task = env._env
  import torch

  rows = []
  for key, xyz, label in LOCATIONS:
    print(f'\n======== {label} ========', flush=True)
    env.reset()
    _move_bucket(env, xyz)
    palm0 = env._palm_xyz()[0].detach().cpu().numpy()
    obj0 = env._object_xyz()[0].detach().cpu().numpy()
    q0 = task.arm_hand_dof_pos[:, :7].detach().clone()
    hover = (float(xyz[0]), float(xyz[1]), HOVER_Z)
    d_palm = float(math.hypot(palm0[0] - xyz[0], palm0[1] - xyz[1]))
    d_base = float(math.hypot(BASE_XY[0] - xyz[0], BASE_XY[1] - xyz[1]))
    print(
        f'start palm={tuple(round(float(v), 3) for v in palm0)}  '
        f'cube-to-bucket xy={d_palm:.2f}m  base-to-bucket xy={d_base:.2f}m  '
        f'iiwa_reach~{IIWA_REACH:.2f}m',
        flush=True)

    ik_err = _ik_hover(task, hover)
    palm_ik = env._palm_xyz()[0].detach().cpu().numpy()
    q_best = task.arm_hand_dof_pos[:, :7].detach().clone()
    place_ok = bool(ik_err < PLACE_OK)
    print(f'IK hover |palm-target|={ik_err:.3f}m  place_ok={place_ok}',
          flush=True)

    # Put the cube back in the hand at the IK pose, then drop.
    ids = torch.arange(task.num_envs, device=task.device)
    task._place_cube_in_hand(ids, write_init=False, use_stored=False)
    _sim(task, 4)
    rgb = env.render_rgb()
    if rgb is not None:
      _caption(rgb, [
          f'{label}  after IK hover',
          f'|palm-hover|={ik_err*100:.1f}cm  place_ok={place_ok}  '
          f'base_xy={d_base:.2f}m',
      ]).save(os.path.join(args.out_dir, f'{key}_after_ik.png'))

    _open_fingers(task)
    ever_drop = False
    for _ in range(70):
      _sim(task, 1)
      ever_drop = ever_drop or _in_bucket(env)
    rgb = env.render_rgb()
    if rgb is not None:
      _caption(rgb, [
          f'{label}  after open-finger drop',
          f'in_bucket={ever_drop}',
      ]).save(os.path.join(args.out_dir, f'{key}_after_drop.png'))

    # Scripted throw: reset to keep-arm, interpolate toward IK-best, release.
    env.reset()
    _move_bucket(env, xyz)
    ever_throw = False
    n_swing = 35
    for t in range(n_swing + 55):
      if t <= n_swing:
        a = float(t) / float(n_swing)
        q = (1.0 - a) * q0 + a * q_best
        _set_arm_q(task, q)
      if t == 12:
        _open_fingers(task)
      _sim(task, 1)
      ever_throw = ever_throw or _in_bucket(env)
    rgb = env.render_rgb()
    if rgb is not None:
      _caption(rgb, [
          f'{label}  after scripted swing+release',
          f'in_bucket={ever_throw}',
      ]).save(os.path.join(args.out_dir, f'{key}_after_throw.png'))

    any_ok = bool(ever_drop or ever_throw)
    row = {
        'name': key,
        'label': label,
        'bucket_xyz': list(xyz),
        'start_palm_xy_m': d_palm,
        'base_xy_m': d_base,
        'ik_hover_err_m': ik_err,
        'place_ik_ok': place_ok,
        'drop_in_bucket': ever_drop,
        'throw_in_bucket': ever_throw,
        'any_scripted_success': any_ok,
        'likely_needs_throw': d_base > IIWA_REACH + 0.15,
    }
    rows.append(row)
    print(
        f'RESULT  drop={ever_drop}  throw={ever_throw}  any={any_ok}',
        flush=True)

  summary_path = os.path.join(args.out_dir, 'feasibility.json')
  with open(summary_path, 'w', encoding='utf-8') as fh:
    json.dump(rows, fh, indent=2)
  print(f'wrote {summary_path}', flush=True)
  print('\n==== summary ====', flush=True)
  for row in rows:
    print(
        f"{row['label']}: IK={row['ik_hover_err_m']:.3f}m "
        f"place={row['place_ik_ok']} drop={row['drop_in_bucket']} "
        f"throw={row['throw_in_bucket']} any={row['any_scripted_success']}",
        flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
