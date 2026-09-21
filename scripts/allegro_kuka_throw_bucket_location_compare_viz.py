#!/usr/bin/env python3
"""Stills: keep-arm pad-hold + NVIDIA table, three candidate bucket xyz.

Same camera for every column so placements are comparable.

  python scripts/allegro_kuka_throw_bucket_location_compare_viz.py
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
    BUCKET_INNER_RADIUS,
    TABLE_SPAWN_IN_HAND_OFFSET,
    TABLE_SPAWN_IN_HAND_WRIST_OFFSET,
    AllegroKukaThrowVecEnv,
)

WH = (960, 720)
HFOV = 58.0
# Wide enough for palm ~(0, 0.04, 0.72), current (0.50,-0.30),
# -x (-0.50,-0.30), and far -y (0.00,-0.80).
OVERVIEW = {
    'oblique': ((1.55, -1.55, 1.70), (0.00, -0.22, 0.42)),
    # Tiny xy offset: a true top-down look-at is parallel to world +z and
    # Isaac Gym returns a black frame.
    'top': ((0.12, -0.18, 2.15), (0.00, -0.22, 0.40)),
}

LOCATIONS = (
    ('current_nvidia', (0.50, -0.30, 0.40), 'current  (0.50, -0.30, 0.40)'),
    ('harder_plusx_far', (0.45, -0.60, 0.40), 'proposed  (0.45, -0.60, 0.40)'),
    ('far_minus_y', (0.00, -0.80, 0.40), 'far  (0.00, -0.80, 0.40)'),
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


def _perspective(hfov_deg, aspect, near=0.05, far=10.0):
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


def _overlay(rgb, env, bucket_xyz, caption_lines):
  h, w = rgb.shape[:2]
  view = _look_at(env._cam_eye, env._cam_tgt)
  hfov = float(getattr(env, '_cam_hfov', HFOV) or HFOV)
  proj = _perspective(hfov, float(w) / max(float(h), 1.0))
  bucket = np.asarray(bucket_xyz, dtype=np.float64)
  obj = env._object_xyz()[0].detach().cpu().numpy()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  ring = []
  for k in range(48):
    ang = 2.0 * np.pi * k / 48.0
    xyz = bucket + np.array(
        [BUCKET_INNER_RADIUS * np.cos(ang),
         BUCKET_INNER_RADIUS * np.sin(ang), 0.10])
    uv = _project(xyz, view, proj, w, h)
    if uv is not None:
      ring.append(uv)
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  if len(ring) >= 3:
    poly = [(int(u), int(v)) for u, v in ring]
    draw.polygon(poly, fill=(40, 200, 80, 55), outline=(40, 230, 90, 230))
    draw.line(poly + poly[:1], fill=(60, 255, 110, 255), width=3)
  b_uv = _project(bucket + np.array([0.0, 0.0, 0.10]), view, proj, w, h)
  o_uv = _project(obj, view, proj, w, h)
  p_uv = _project(palm, view, proj, w, h)
  if o_uv and b_uv:
    draw.line(
        [(int(o_uv[0]), int(o_uv[1])), (int(b_uv[0]), int(b_uv[1]))],
        fill=(255, 220, 80, 220), width=3)

  def _dot(uv, fill, label, dy=-18):
    if uv is None:
      return
    u, v = int(uv[0]), int(uv[1])
    draw.ellipse([u - 7, v - 7, u + 7, v + 7], fill=fill,
                 outline=(255, 255, 255, 255))
    draw.text((u + 10, v + dy), label, fill=fill)

  _dot(b_uv, (60, 230, 100, 255), 'bucket  (r=12cm)')
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


def _grid(frames, cols):
  if not frames:
    raise ValueError('no frames')
  h, w = frames[0].shape[:2]
  rows = (len(frames) + cols - 1) // cols
  canvas = np.full((rows * h, cols * w, 3), 18, dtype=np.uint8)
  for i, fr in enumerate(frames):
    r, c = divmod(i, cols)
    canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
  return Image.fromarray(canvas)


def _parse_locations(raw):
  if not raw:
    return LOCATIONS
  out = []
  for item in raw:
    parts = item.split(':', 2)
    if len(parts) < 2:
      raise ValueError(
          'location must be name:x,y,z or name:x,y,z:label, got '
          f'{item!r}')
    name, xyz_s = parts[0], parts[1]
    xyz = tuple(float(v) for v in xyz_s.split(','))
    if len(xyz) != 3:
      raise ValueError(f'location xyz must have 3 floats, got {item!r}')
    label = parts[2] if len(parts) == 3 else name
    out.append((name, xyz, label))
  return tuple(out)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument('--out-dir', default=os.path.join(
      _REPO, 'figs', 'allegro_kuka_throw', 'bucket_location_compare'))
  p.add_argument('--pipeline', default='gpu')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument(
      '--location', action='append', default=[],
      help='name:x,y,z[:label]  (repeat; default is the built-in 3-way)')
  args = p.parse_args()
  locations = _parse_locations(args.location)
  os.makedirs(args.out_dir, exist_ok=True)

  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=int(args.seed),
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
      hide_table=False,
      large_table=False,
      randomize_init=False,
      randomize_object_shape=False,
      fixed_target_xyz=locations[0][1],
      enable_cameras=True,
      camera_width=WH[0],
      camera_height=WH[1],
      camera_eye=OVERVIEW['oblique'][0],
      camera_tgt=OVERVIEW['oblique'][1],
      camera_hfov=HFOV,
      pipeline=args.pipeline,
  )
  env.reset()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  obj = env._object_xyz()[0].detach().cpu().numpy()
  print(
      f'palm={tuple(round(float(v), 3) for v in palm)}  '
      f'cube={tuple(round(float(v), 3) for v in obj)}',
      flush=True)

  oblique_row = []
  top_row = []
  for key, xyz, label in locations:
    _move_bucket(env, xyz)
    dxy = float(np.hypot(obj[0] - xyz[0], obj[1] - xyz[1]))
    for view_name, (eye, tgt) in OVERVIEW.items():
      _set_cam(env, eye, tgt)
      rgb = env.render_rgb()
      if rgb is None:
        print('camera returned no frame', flush=True)
        return 1
      lines = [
          f'{label}   {view_name}   NVIDIA table   keep-arm pad-hold',
          (f'palm=({palm[0]:.2f},{palm[1]:.2f},{palm[2]:.2f})  '
           f'cube-to-bucket xy={dxy*100:.0f}cm  cylinder r=12cm'),
      ]
      labeled = _overlay(rgb, env, xyz, lines)
      path = os.path.join(args.out_dir, f'{key}_{view_name}.png')
      Image.fromarray(labeled).save(path)
      print(f'wrote {path}', flush=True)
      if view_name == 'oblique':
        oblique_row.append(labeled)
      else:
        top_row.append(labeled)

  grid = _grid(oblique_row + top_row, cols=len(locations))
  grid_path = os.path.join(args.out_dir, 'bucket_location_compare_grid.png')
  grid.save(grid_path)
  print(f'wrote {grid_path}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
