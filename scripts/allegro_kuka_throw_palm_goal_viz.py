#!/usr/bin/env python3
"""Visualize Allegro palm goals for throw / place / slide.

Scenes (``--scene``):

  place    palm at the bucket rim, object in the bucket.
  windup   far throw bucket stays put; palm high over the table, wrist aimed
           at the bucket; object at the palm (release pose).
  slide    bucket against the table +x edge, rim 2 cm below the top; palm on
           the desk next to the lip (palm down, fingers toward the bucket);
           object in the bucket.
  slide_compare  three slide palms: current (0.17), lip (0.22), over-gap (0.25).

Markers: cyan=palm goal, magenta=IK palm, orange=object, yellow=bucket.

  python scripts/allegro_kuka_throw_palm_goal_viz.py \
      --output-dir=figs/allegro_kuka_throw --scene=slide_compare
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Bucket.obj: origin at the floor, height ≈ 0.198 m, xy radius ≈ 0.12 m.
BUCKET_HEIGHT = 0.198
PALM_ABOVE_RIM = 0.04
OBJECT_ABOVE_FLOOR = 0.05
CURRENT_BUCKET = (0.5, -0.3, 0.4)
REACHABLE_BUCKET = (0.30, 0.25, 0.38)
ROBOT_BASE = np.array([0.0, 0.8, 0.0], dtype=np.float64)
# Over the robot-side / +x of the table, in reach, offset toward the bucket.
WINDUP_PALM = (0.18, 0.28, 0.70)
WINDUP_AIM = (0.50, -0.30, 0.50)  # bucket interior, slightly above origin
# NVIDIA table_narrow: 0.475 x 0.4 x 0.3 box at (0, 0, 0.38) → top z = 0.53.
# Slide: bucket against +x / robot-near edge; rim 2 cm below the table top.
TABLE_TOP_Z = 0.38 + 0.15
SLIDE_BUCKET = (0.37, 0.08, TABLE_TOP_Z - BUCKET_HEIGHT - 0.02)
SLIDE_PALM = (0.17, 0.08, TABLE_TOP_Z + PALM_ABOVE_RIM)
# Closer-to-bucket variants (same y as the bucket). Table +x edge ≈ 0.24;
# bucket wall ≈ 0.25. Lip is the proposed train target.
SLIDE_PALM_LIP = (0.22, 0.08, TABLE_TOP_Z + PALM_ABOVE_RIM)
SLIDE_PALM_GAP = (0.25, 0.08, 0.55)


def _palm_goal(bucket_xyz):
  b = np.asarray(bucket_xyz, dtype=np.float64)
  g = b.copy()
  g[2] = b[2] + BUCKET_HEIGHT + PALM_ABOVE_RIM
  return g


def _object_goal(bucket_xyz):
  b = np.asarray(bucket_xyz, dtype=np.float64)
  g = b.copy()
  g[2] = b[2] + OBJECT_ABOVE_FLOOR
  return g


def _skew(v):
  import torch
  z = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
  x, y, zz = v[:, 0], v[:, 1], v[:, 2]
  row0 = torch.stack([z, -zz, y], dim=-1)
  row1 = torch.stack([zz, z, -x], dim=-1)
  row2 = torch.stack([-y, x, z], dim=-1)
  return torch.stack([row0, row1, row2], dim=-2)


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


def _draw_disk(img, uv, color, radius=7, ring=2):
  if uv is None:
    return
  h, w = img.shape[:2]
  u, v = int(round(uv[0])), int(round(uv[1]))
  r = int(radius + ring)
  yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
  dist2 = xx * xx + yy * yy
  for dy, dx in zip(*np.where(dist2 <= (r * r))):
    y = v + dy - r
    x = u + dx - r
    if 0 <= y < h and 0 <= x < w:
      img[y, x] = (255, 255, 255) if dist2[dy, dx] > (radius * radius) else color


def _annotate(img, pts, colors):
  out = np.ascontiguousarray(img.copy())
  for uv, color in zip(pts, colors):
    _draw_disk(out, uv, color)
  return out


def _caption_bar(img, text):
  """Black strip above ``img`` with ``text`` (PIL). Returns img if PIL is missing."""
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return img
  bar_h = 36
  canvas = np.full((img.shape[0] + bar_h, img.shape[1], 3), 20, dtype=np.uint8)
  canvas[bar_h:] = img
  pil = Image.fromarray(canvas)
  ImageDraw.Draw(pil).text((10, 8), text, fill=(240, 240, 240))
  return np.asarray(pil)


def _hcat(imgs, pad=8):
  h = max(im.shape[0] for im in imgs)
  chunks = []
  for i, im in enumerate(imgs):
    if i:
      chunks.append(np.full((h, pad, 3), 20, dtype=np.uint8))
    if im.shape[0] < h:
      top = (h - im.shape[0]) // 2
      im = np.pad(im, ((top, h - im.shape[0] - top), (0, 0), (0, 0)))
    chunks.append(im)
  return np.concatenate(chunks, axis=1)


def _vcat(imgs, pad=8):
  w = max(im.shape[1] for im in imgs)
  chunks = []
  for i, im in enumerate(imgs):
    if i:
      chunks.append(np.full((pad, w, 3), 20, dtype=np.uint8))
    if im.shape[1] < w:
      left = (w - im.shape[1]) // 2
      im = np.pad(im, ((0, 0), (left, w - im.shape[1] - left), (0, 0)))
    chunks.append(im)
  return np.concatenate(chunks, axis=0)


def _make_cam(task, wh, pos, tgt):
  from isaacgym import gymapi
  props = gymapi.CameraProperties()
  props.width = int(wh[0])
  props.height = int(wh[1])
  props.enable_tensors = False
  env_ptr = task.envs[0]
  hdl = task.gym.create_camera_sensor(env_ptr, props)
  task.gym.set_camera_location(
      hdl, env_ptr, gymapi.Vec3(*pos), gymapi.Vec3(*tgt))
  return hdl


def _read_rgb(task, handle, wh):
  from isaacgym import gymapi
  img = task.gym.get_camera_image(
      task.sim, task.envs[0], handle, gymapi.IMAGE_COLOR)
  w, h = int(wh[0]), int(wh[1])
  return np.asarray(img, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()


def _cam_matrices(task, handle):
  view = np.asarray(
      task.gym.get_camera_view_matrix(task.sim, task.envs[0], handle),
      dtype=np.float64).reshape(4, 4)
  proj = np.asarray(
      task.gym.get_camera_proj_matrix(task.sim, task.envs[0], handle),
      dtype=np.float64).reshape(4, 4)
  return view, proj


def _set_root_xyz(task, actor_indices, xyz):
  import torch
  from isaacgym import gymtorch
  n = int(len(actor_indices))
  tgt = torch.as_tensor(xyz, dtype=torch.float, device=task.device).view(1, 3)
  task.root_state_tensor[actor_indices, 0:3] = tgt.expand(n, 3)
  task.root_state_tensor[actor_indices, 7:13] = 0.0
  idx = actor_indices.to(torch.int32)
  task.gym.set_actor_root_state_tensor_indexed(
      task.sim,
      gymtorch.unwrap_tensor(task.root_state_tensor),
      gymtorch.unwrap_tensor(idx),
      n,
  )


def _place_bucket_and_object(vec, bucket_xyz):
  task = vec._env
  obj_g = _object_goal(bucket_xyz)
  _set_root_xyz(task, task.bucket_object_indices, bucket_xyz)
  _set_root_xyz(task, task.object_indices, obj_g)
  b = np.asarray(bucket_xyz, dtype=np.float32)
  task.goal_states[:, 0:3] = task.goal_states.new_tensor(b)
  task.goal_states[:, 2] = float(b[2]) + OBJECT_ABOVE_FLOOR
  vec._goal_batch[:, 0] = float(b[0])
  vec._goal_batch[:, 1] = float(b[1])
  vec._goal_batch[:, 2] = float(b[2])


def _acquire_jacobian(task):
  from isaacgym import gymtorch
  try:
    raw = task.gym.acquire_jacobian_tensor(task.sim, 'allegro')
  except Exception as exc:
    print(f'[viz] jacobian acquire failed: {exc}', flush=True)
    return None
  jac = gymtorch.wrap_tensor(raw)
  print(f'[viz] jacobian shape={tuple(jac.shape)} '
        f'palm_handle={int(task.allegro_palm_handle)}', flush=True)
  return jac


def _ik_to_palm(
    vec, jac, palm_goal, n_iters=180, aim_xyz=None, z_world=None,
    x_world=None, orn_w=0.35):
  """DLS IK on arm DOFs (0:7) targeting palm_center_pos.

  If ``aim_xyz`` is set, points link_7 +z at that world point (throw aim).
  ``z_world`` / ``x_world`` override that with explicit palm axes (slide:
  +z down, +x toward the bucket).
  """
  import torch
  from isaacgym import gymtorch
  from isaacgym.torch_utils import quat_rotate, tensor_clamp

  task = vec._env
  device = task.device
  goal = torch.as_tensor(palm_goal, dtype=torch.float, device=device).view(1, 3)
  palm_off = torch.as_tensor(
      task.palm_offset, dtype=torch.float, device=device).view(1, 3)
  body = int(task.allegro_palm_handle)
  if jac is None:
    return None
  if body >= int(jac.shape[1]):
    body = int(jac.shape[1]) - 1
  lo = task.arm_hand_dof_lower_limits[:7]
  hi = task.arm_hand_dof_upper_limits[:7]
  z_des = None
  x_des = None
  if z_world is not None:
    z_des = torch.as_tensor(z_world, dtype=torch.float, device=device).view(1, 3)
    z_des = z_des / (torch.norm(z_des, dim=-1, keepdim=True) + 1e-8)
  elif aim_xyz is not None:
    aim = torch.as_tensor(aim_xyz, dtype=torch.float, device=device).view(1, 3)
    z_des = aim - goal
    z_des = z_des / (torch.norm(z_des, dim=-1, keepdim=True) + 1e-8)
  if x_world is not None:
    if z_des is None:
      raise ValueError('x_world requires aim_xyz or z_world')
    x_raw = torch.as_tensor(x_world, dtype=torch.float, device=device).view(1, 3)
    x_raw = x_raw - (x_raw * z_des).sum(dim=-1, keepdim=True) * z_des
    x_des = x_raw / (torch.norm(x_raw, dim=-1, keepdim=True) + 1e-8)
  last_err = None
  last_ang = None
  z_axis = torch.zeros((1, 3), device=device, dtype=goal.dtype)
  z_axis[:, 2] = 1.0
  x_axis = torch.zeros((1, 3), device=device, dtype=goal.dtype)
  x_axis[:, 0] = 1.0
  for i in range(int(n_iters)):
    task.gym.refresh_dof_state_tensor(task.sim)
    task.gym.refresh_rigid_body_state_tensor(task.sim)
    task.gym.refresh_jacobian_tensors(task.sim)
    task.compute_observations()
    palm = task.palm_center_pos
    pos_err = goal - palm
    last_err = float(torch.norm(pos_err, dim=-1)[0].item())
    j_full = jac[:, body, :, :7]
    j_lin = j_full[:, 0:3, :]
    j_ang = j_full[:, 3:6, :]
    off_w = quat_rotate(task._palm_rot, palm_off.expand(task.num_envs, 3))
    j_palm = j_lin + torch.bmm(-_skew(off_w), j_ang)
    if z_des is None:
      jjt = torch.bmm(j_palm, j_palm.transpose(1, 2))
      lam = (0.08 ** 2) * torch.eye(3, device=device, dtype=j_palm.dtype)
      dq = torch.bmm(
          j_palm.transpose(1, 2),
          torch.linalg.solve(jjt + lam, pos_err.unsqueeze(-1)),
      ).squeeze(-1)
      step = 0.35
    else:
      z_cur = quat_rotate(task._palm_rot, z_axis.expand(task.num_envs, 3))
      orn_err = torch.cross(z_cur, z_des.expand_as(z_cur), dim=-1)
      if x_des is not None:
        x_cur = quat_rotate(task._palm_rot, x_axis.expand(task.num_envs, 3))
        orn_err = orn_err + 0.5 * torch.cross(
            x_cur, x_des.expand_as(x_cur), dim=-1)
      cosang = torch.clamp((z_cur * z_des).sum(dim=-1), -1.0, 1.0)
      last_ang = float(torch.acos(cosang)[0].item())
      err6 = torch.cat([pos_err, float(orn_w) * orn_err], dim=-1)
      j6 = torch.cat([j_palm, j_ang], dim=1)
      jjt = torch.bmm(j6, j6.transpose(1, 2))
      lam = (0.10 ** 2) * torch.eye(6, device=device, dtype=j6.dtype)
      dq = torch.bmm(
          j6.transpose(1, 2),
          torch.linalg.solve(jjt + lam, err6.unsqueeze(-1)),
      ).squeeze(-1)
      step = 0.25
    q = task.arm_hand_dof_pos[:, :7] + step * dq
    q = tensor_clamp(q, lo, hi)
    task.cur_targets[:, :7] = q
    task.prev_targets[:, :7] = q
    task.arm_hand_dof_pos[:, :7] = q
    task.gym.set_dof_position_target_tensor(
        task.sim, gymtorch.unwrap_tensor(task.cur_targets))
    task.gym.set_dof_state_tensor(
        task.sim, gymtorch.unwrap_tensor(task.dof_state))
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
    if i % 40 == 0 or i == n_iters - 1:
      extra = '' if last_ang is None else f'  aim_err={last_ang * 180.0 / np.pi:.1f} deg'
      print(f'[viz] ik iter {i:3d}  palm_err={last_err:.3f} m{extra}', flush=True)
    done = last_err < 0.02
    if last_ang is not None:
      done = done and last_ang < 0.15
    if done:
      break
  task.compute_observations()
  return last_err


def _render_scene(vec, cams, wh, palm_g, obj_g, extra=()):
  from isaacgym import gymapi
  task = vec._env
  task.gym.fetch_results(task.sim, True)
  task.gym.step_graphics(task.sim)
  task.gym.render_all_camera_sensors(task.sim)
  task.compute_observations()
  palm_a = task.palm_center_pos[0].detach().cpu().numpy()
  obj_a = task.object_pos[0].detach().cpu().numpy()
  frames = []
  for hdl in cams:
    img = _read_rgb(task, hdl, wh)
    view, proj = _cam_matrices(task, hdl)
    pts = [
        _project(palm_g, view, proj, wh[0], wh[1]),
        _project(palm_a, view, proj, wh[0], wh[1]),
        _project(obj_g, view, proj, wh[0], wh[1]),
    ]
    for xyz in extra:
      pts.append(_project(xyz, view, proj, wh[0], wh[1]))
    frames.append((img, pts, palm_a, obj_a))
  return frames, palm_a, obj_a


def _panel_title(name, bucket, palm_g, palm_a, obj_a, ik_err, obj_ref=None):
  b = np.asarray(bucket, dtype=np.float64)
  pg = np.asarray(palm_g, dtype=np.float64)
  reach = np.linalg.norm(pg - ROBOT_BASE)
  pa = np.linalg.norm(np.asarray(palm_a) - pg)
  oref = _object_goal(bucket) if obj_ref is None else np.asarray(obj_ref)
  oa = np.linalg.norm(np.asarray(obj_a) - oref)
  ik_s = 'n/a' if ik_err is None else f'{ik_err:.3f} m'
  return (
      f'{name}  bucket=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f})  '
      f'base→palm-goal {reach:.2f} m\n'
      f'IK residual {ik_s}   |palm−goal|={pa:.3f} m   |obj−ref|={oa:.3f} m'
  )


def _slide_ik_and_render(vec, jac, cams, wh, bucket, palm_g, obj_g, name,
                         ik_iters, colors):
  """IK palm-down / +x fingers, then 3-cam row with a caption."""
  task = vec._env
  _place_bucket_and_object(vec, bucket)
  _seed_arm_default(task)
  _place_bucket_and_object(vec, bucket)
  ik_err = _ik_to_palm(
      vec, jac, palm_g, n_iters=int(ik_iters),
      z_world=(0.0, 0.0, -1.0), x_world=(1.0, 0.0, 0.0))
  _place_bucket_and_object(vec, bucket)
  task.compute_observations()
  frames, palm_a, obj_a = _render_scene(
      vec, cams, wh, palm_g, obj_g, extra=(bucket,))
  title = _panel_title(name, bucket, palm_g, palm_a, obj_a, ik_err)
  annotated = [_annotate(img, pts, colors) for img, pts, _, _ in frames]
  row = _caption_bar(_hcat(annotated), title.replace('\n', ' | '))
  return row, title


def _seed_arm_default(task):
  from isaacgym import gymtorch
  task.progress_buf[:] = 0
  task.reset_buf[:] = 0
  task.arm_hand_dof_pos[:, :] = task.hand_arm_default_dof_pos
  task.cur_targets[:, :] = task.hand_arm_default_dof_pos
  task.prev_targets[:, :] = task.hand_arm_default_dof_pos
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))
  for _ in range(4):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--output-dir', default='figs/allegro_kuka_throw')
  p.add_argument('--pipeline', default='gpu', choices=('gpu', 'cpu'))
  p.add_argument('--ik-iters', type=int, default=180)
  p.add_argument('--width', type=int, default=640)
  p.add_argument('--height', type=int, default=480)
  p.add_argument('--scene', default='all',
                 choices=('all', 'place', 'windup', 'slide', 'slide_compare'))
  args = p.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
  import imageio

  marker_rgb = ((61, 219, 232), (232, 60, 154), (232, 131, 76))  # palm goal, actual, obj
  yellow = (240, 210, 70)
  wh = (int(args.width), int(args.height))
  vec = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=0,
      episode_length=4000,
      pipeline=str(args.pipeline),
      enable_cameras=True,
      camera_width=wh[0],
      camera_height=wh[1],
      headless=True,
      randomize_init=False,
      fixed_target_xyz=CURRENT_BUCKET,
  )
  vec.reset()
  task = vec._env
  jac = _acquire_jacobian(task)

  do_place = args.scene in ('all', 'place')
  do_windup = args.scene in ('all', 'windup')
  do_slide = args.scene in ('all', 'slide')
  do_slide_compare = args.scene == 'slide_compare'
  summary_lines = []
  row_imgs = []

  if do_place:
    scenes = (
        ('current throw bucket', CURRENT_BUCKET,
         (1.35, -1.45, 1.15), (0.25, -0.15, 0.40),
         (1.05, -0.85, 0.95), (0.50, -0.30, 0.52)),
        ('reachable table bucket', REACHABLE_BUCKET,
         (1.20, -1.20, 1.10), (0.10, 0.15, 0.40),
         (0.85, -0.25, 0.95), (0.30, 0.25, 0.50)),
    )
    print('[viz] markers: cyan=palm goal (rim)  magenta=IK palm  orange=object',
          flush=True)
    for row, (name, bucket, wide_pos, wide_tgt, close_pos, close_tgt) in enumerate(scenes):
      palm_g = _palm_goal(bucket)
      obj_g = _object_goal(bucket)
      print(f'\n[viz] === {name} ===', flush=True)
      print(f'[viz] bucket={bucket}  palm_goal={tuple(np.round(palm_g, 3))}  '
            f'object_goal={tuple(np.round(obj_g, 3))}', flush=True)
      print(f'[viz] |base→palm_goal|={np.linalg.norm(palm_g - ROBOT_BASE):.3f} m '
            f'(iiwa reach ~0.8 m + hand ~0.16 m)', flush=True)
      _place_bucket_and_object(vec, bucket)
      _seed_arm_default(task)
      _place_bucket_and_object(vec, bucket)
      ik_err = _ik_to_palm(vec, jac, palm_g, n_iters=int(args.ik_iters))
      _place_bucket_and_object(vec, bucket)
      task.compute_observations()
      cams = [
          _make_cam(task, wh, wide_pos, wide_tgt),
          _make_cam(task, wh, close_pos, close_tgt),
      ]
      frames, palm_a, obj_a = _render_scene(vec, cams, wh, palm_g, obj_g)
      title = _panel_title(name, bucket, palm_g, palm_a, obj_a, ik_err)
      summary_lines.append(title.replace('\n', ' | '))
      annotated = [_annotate(img, pts, marker_rgb) for img, pts, _, _ in frames]
      slug = 'current' if row == 0 else 'reachable'
      out_s = os.path.join(args.output_dir, f'palm_goal_{slug}_bucket.png')
      imageio.imwrite(out_s, _hcat(annotated))
      print(f'[viz] wrote {out_s}', flush=True)
      row_imgs.append(_hcat(annotated))

    if row_imgs:
      out = os.path.join(args.output_dir, 'palm_goal_compare.png')
      imageio.imwrite(out, _vcat(row_imgs, pad=12))
      print(f'[viz] wrote {out}', flush=True)

  if do_windup:
    palm_g = np.asarray(WINDUP_PALM, dtype=np.float64)
    bucket = CURRENT_BUCKET
    print('\n[viz] === throw wind-up (palm over table, aim at far bucket) ===',
          flush=True)
    print(f'[viz] bucket={bucket}  palm_goal={WINDUP_PALM}  aim={WINDUP_AIM}',
          flush=True)
    print(f'[viz] |base→palm_goal|={np.linalg.norm(palm_g - ROBOT_BASE):.3f} m',
          flush=True)
    print('[viz] markers: cyan=palm goal  magenta=IK palm  orange=object  '
          'yellow=bucket', flush=True)
    _place_bucket_and_object(vec, bucket)
    _seed_arm_default(task)
    _set_root_xyz(task, task.bucket_object_indices, bucket)
    ik_err = _ik_to_palm(
        vec, jac, palm_g, n_iters=int(args.ik_iters), aim_xyz=WINDUP_AIM)
    task.compute_observations()
    palm_now = task.palm_center_pos[0].detach().cpu().numpy()
    _set_root_xyz(task, task.object_indices, palm_now)
    _set_root_xyz(task, task.bucket_object_indices, bucket)
    task.compute_observations()
    cams = [
        _make_cam(task, wh, (1.35, -1.45, 1.15), (0.25, -0.15, 0.40)),
        _make_cam(task, wh, (1.05, -0.40, 0.95), (0.18, 0.22, 0.58)),
        _make_cam(task, wh, (-0.25, 1.15, 1.05), (0.30, 0.00, 0.50)),
    ]
    frames, palm_a, obj_a = _render_scene(
        vec, cams, wh, palm_g, palm_now, extra=(bucket,))
    colors = marker_rgb + (yellow,)
    title = _panel_title(
        'throw wind-up', bucket, palm_g, palm_a, obj_a, ik_err, obj_ref=palm_g)
    summary_lines.append(title.replace('\n', ' | '))
    annotated = [_annotate(img, pts, colors) for img, pts, _, _ in frames]
    out_s = os.path.join(args.output_dir, 'palm_goal_throw_windup.png')
    imageio.imwrite(out_s, _hcat(annotated))
    print(f'[viz] wrote {out_s}', flush=True)

  if do_slide or do_slide_compare:
    bucket = SLIDE_BUCKET
    obj_g = _object_goal(bucket)
    colors = marker_rgb + (yellow,)
    cams = [
        _make_cam(task, wh, (1.25, -1.15, 1.05), (0.15, 0.00, 0.45)),
        _make_cam(task, wh, (0.95, -0.35, 0.90), (0.25, 0.08, 0.52)),
        _make_cam(task, wh, (-0.15, 0.95, 1.05), (0.20, 0.05, 0.50)),
    ]
    print('[viz] markers: cyan=palm goal  magenta=IK palm  orange=object  '
          'yellow=bucket', flush=True)
    if do_slide:
      palm_g = np.asarray(SLIDE_PALM, dtype=np.float64)
      print('\n[viz] === slide (palm on desk, bucket at +x lip) ===', flush=True)
      print(f'[viz] table top z={TABLE_TOP_Z:.3f}  '
            f'bucket={tuple(np.round(bucket, 3))}  '
            f'palm_goal={tuple(np.round(palm_g, 3))}', flush=True)
      row, title = _slide_ik_and_render(
          vec, jac, cams, wh, bucket, palm_g, obj_g,
          'slide desk-to-bucket', args.ik_iters, colors)
      summary_lines.append(title.replace('\n', ' | '))
      out_s = os.path.join(args.output_dir, 'palm_goal_slide.png')
      imageio.imwrite(out_s, row)
      print(f'[viz] wrote {out_s}', flush=True)
    if do_slide_compare:
      poses = (
          ('current 0.17 (desk)', SLIDE_PALM),
          ('lip 0.22 (+5 cm x)', SLIDE_PALM_LIP),
          ('over-gap 0.25 (bucket wall)', SLIDE_PALM_GAP),
      )
      print('\n[viz] === slide_compare current / lip / over-gap ===', flush=True)
      rows = []
      for name, palm in poses:
        palm_g = np.asarray(palm, dtype=np.float64)
        print(f'\n[viz] --- {name}  palm={tuple(np.round(palm_g, 3))} ---',
              flush=True)
        row, title = _slide_ik_and_render(
            vec, jac, cams, wh, bucket, palm_g, obj_g,
            name, args.ik_iters, colors)
        summary_lines.append(title.replace('\n', ' | '))
        rows.append(row)
      out_s = os.path.join(args.output_dir, 'palm_goal_slide_compare.png')
      imageio.imwrite(out_s, _vcat(rows, pad=12))
      print(f'[viz] wrote {out_s}', flush=True)

  for line in summary_lines:
    print('[viz] ' + line, flush=True)


if __name__ == '__main__':
  main()
