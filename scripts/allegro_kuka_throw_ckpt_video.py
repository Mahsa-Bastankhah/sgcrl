"""Roll out a trained AllegroKukaThrow PPO policy to mp4 (GPU camera).

PhysX is created *before* JAX so Preview 4 GPU kernels still register.
Env packing (throw vs --isaacgym_palm_goal) is read from the run's
``run_config.json`` so the obs dim matches the checkpoint.

Control-sanity joint modes (four2mmx, hand16, …) render a hand close-up
with a static GOAL pose stacked above the live policy frame.

  python scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint=logs/.../checkpoints/ckpt_iter_0000100.pkl \
      --output=videos/allegro_kuka_throw/run_iter100.mp4 \
      --deterministic --episodes=2

  # Stochastic + raw NF log p strip on top:
  python scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint=.../ckpt_iter_0000100.pkl \
      --output=videos/.../iter100.mp4 \
      --episodes=1 --reward-overlay

  # Flax PPO+RND params_*.pkl (no run_config.json — pass env flags):
  python scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint=logs/.../params_976.pkl \
      --flags-json=hidetable_rnd_flags.json \
      --output=videos/.../params_976_t150_stoch.mp4 \
      --num-steps=150 --episodes=2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _run_config_path(ckpt_path: str) -> str:
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
  return os.path.join(run_dir, 'run_config.json')


def _load_flags(ckpt_path: str) -> dict:
  cfg_path = _run_config_path(ckpt_path)
  if not os.path.isfile(cfg_path):
    return {}
  with open(cfg_path, 'r', encoding='utf-8') as fh:
    return dict((json.load(fh).get('flags') or {}))


def _as_xyz(raw, default, dims=3):
  if raw is None:
    return default
  if isinstance(raw, str):
    parts = [p for p in raw.replace(' ', '').split(',') if p]
  else:
    parts = list(raw)
  if len(parts) != dims:
    return default
  return tuple(float(x) for x in parts)


def _load_run_hparams(flags: dict):
  hidden = (256, 256, 256, 256, 256, 256)
  raw = str(flags.get('hidden_layer_sizes') or '')
  if raw.strip():
    hidden = tuple(int(x) for x in raw.split(',') if x.strip())
  # SAC writes --actor_min_std (stock 1e-6). PPO writes --ppo_actor_min_std
  # (typically 1e-5). Fall back to the historical PPO floor.
  min_std = 1e-5
  if flags.get('actor_min_std') is not None:
    min_std = float(flags['actor_min_std'])
  elif flags.get('ppo_actor_min_std') is not None:
    min_std = float(flags['ppo_actor_min_std'])
  return hidden, min_std


# Joint-goal control-sanity modes rendered with a hand close-up + goal strip.
_CONTROL_SANITY_JOINT_MODES = (
    'finger', 'hand16', 'hand16fig', 'hand16ok', 'hand16peace',
    'hand16point', 'hand16gun', 'arm23wave', 'index',
    'six', 'two_finger', 'three2',
    'four2', 'four2h', 'four2m', 'four2mh', 'four2mm', 'four2mmh',
    'four2mmx', 'four2w', 'index_thumb_straight',
)


def _load_env_kwargs(flags: dict, num_steps: int, seed: int, pipeline: str):
  ep = int(flags.get('isaacgym_episode_length') or 300)
  if int(num_steps) > 0:
    ep = int(num_steps)
  mode = str(flags.get('isaacgym_control_sanity_mode', 'off') or 'off')
  kwargs = dict(
      num_envs=1,
      seed=int(seed),
      episode_length=int(ep),
      pipeline=str(pipeline or flags.get('isaacgym_pipeline') or 'gpu'),
      enable_cameras=True,
      headless=True,
      fixed_target_xyz=_as_xyz(
          flags.get('isaacgym_fixed_target_xyz'), (0.5, -0.3, 0.4)),
      randomize_init=bool(flags.get('isaacgym_randomize_init', True)),
      randomize_object_xyz=bool(flags.get('isaacgym_randomize_object_xyz', False)),
      randomize_object_shape=bool(
          flags.get('isaacgym_randomize_object_shape', True)),
      palm_goal=bool(flags.get('isaacgym_palm_goal', False)),
      palm_goal_xyz=_as_xyz(
          flags.get('isaacgym_palm_goal_xyz'), (0.17, 0.08, 0.57)),
      joint_goal=bool(flags.get('isaacgym_joint_goal', False)),
      control_sanity_mode=mode,
      control_sanity_palm_xyz=_as_xyz(
          flags.get('isaacgym_control_sanity_palm_xyz'),
          (0.0, 0.0, 0.80)),
      control_sanity_finger_tol=float(
          flags.get('isaacgym_control_sanity_finger_tol', 0.15)),
      control_sanity_palm_tol=float(
          flags.get('isaacgym_control_sanity_palm_tol', 0.05)),
      control_sanity_trim_sa=bool(
          flags.get('isaacgym_control_sanity_trim_sa', False)),
      control_sanity_trim_init_range_frac=float(
          flags.get(
              'isaacgym_control_sanity_trim_init_range_frac', 0.10)),
      control_sanity_trim_init_mode=str(
          flags.get(
              'isaacgym_control_sanity_trim_init_mode', 'curled')
          or 'curled').strip().lower(),
      control_sanity_q_only=bool(
          flags.get('isaacgym_control_sanity_q_only', False)),
      control_sanity_goal_include_qd=bool(
          flags.get(
              'isaacgym_control_sanity_goal_include_qd', False)),
      # Absent from historical run_config files: preserve legacy mixed packing.
      coordinate_mode=str(
          flags.get('isaacgym_coordinate_mode', 'mixed')
          or 'mixed').strip().lower(),
      table_push=bool(flags.get('isaacgym_table_push', False)),
      table_push_xyz=_as_xyz(
          flags.get('isaacgym_table_push_xyz'), None),
      table_spawn=bool(flags.get('isaacgym_table_spawn', False)),
      table_spawn_object_xy=_as_xyz(
          flags.get('isaacgym_table_spawn_object_xy'), None, dims=2),
      table_spawn_behind=bool(
          flags.get('isaacgym_table_spawn_behind', False)),
      table_spawn_behind_dy=float(
          flags.get('isaacgym_table_spawn_behind_dy', 0.14)),
      table_spawn_behind_above=float(
          flags.get('isaacgym_table_spawn_behind_above', 0.08)),
      table_spawn_correlated_xy=float(
          flags.get('isaacgym_table_spawn_correlated_xy', 0.0)),
      table_spawn_finger_curl_scale=float(
          flags.get('isaacgym_table_spawn_finger_curl_scale', 1.0)),
      table_spawn_finger_noise=float(
          flags.get('isaacgym_table_spawn_finger_noise', 0.0)),
      table_spawn_arm_noise=float(
          flags.get('isaacgym_table_spawn_arm_noise', 0.0)),
      table_spawn_toward_bucket=bool(
          flags.get('isaacgym_table_spawn_toward_bucket', False)),
      table_spawn_in_hand=bool(
          flags.get('isaacgym_table_spawn_in_hand', False)),
      table_spawn_in_hand_offset=float(
          flags.get('isaacgym_table_spawn_in_hand_offset', 0.042)),
      table_spawn_in_hand_obj_noise=float(
          flags.get('isaacgym_table_spawn_in_hand_obj_noise', 0.012)),
      table_spawn_in_hand_keep_arm=bool(
          flags.get('isaacgym_table_spawn_in_hand_keep_arm', False)),
      table_spawn_in_hand_wrist_offset=float(
          flags.get(
              'isaacgym_table_spawn_in_hand_wrist_offset',
              -3.141592653589793)),
      table_spawn_in_hand_wrist_noise=float(
          flags.get('isaacgym_table_spawn_in_hand_wrist_noise', 0.10)),
      large_table=bool(flags.get('isaacgym_large_table', False)),
      hide_table=bool(flags.get('isaacgym_hide_table', False)),
      palm_and_object_success=bool(
          flags.get('isaacgym_palm_and_object_success', False)),
      throw_success=str(flags.get('isaacgym_throw_success', 'in_bucket')),
      reset_z_above=float(flags.get('isaacgym_reset_z_above', 0.0) or 0.0),
      lock_arm_base=bool(flags.get('isaacgym_lock_arm_base', False)),
  )
  _goal_z = flags.get('isaacgym_goal_z', -1.0)
  try:
    _goal_z = float(_goal_z)
  except (TypeError, ValueError):
    _goal_z = -1.0
  if _goal_z >= 0.0:
    kwargs['goal_z'] = _goal_z
  # Narrow FOV at construction (Isaac Gym cannot change HFOV after create).
  if mode in _CONTROL_SANITY_JOINT_MODES:
    kwargs['camera_hfov'] = 35.0
    kwargs['camera_width'] = 640
    kwargs['camera_height'] = 480
  elif bool(flags.get('isaacgym_table_spawn_in_hand_keep_arm', False)):
    # Do not hardcode eye/tgt: env auto-frames the start palm + goal/bucket
    # so far targets (e.g. y=−0.80) stay on screen.
    kwargs['camera_hfov'] = 48.0
    kwargs['camera_width'] = 960
    kwargs['camera_height'] = 720
  elif bool(flags.get('isaacgym_table_spawn', False)):
    # Same auto-frame as keep-arm. The old table_spawn eye sat at y≈−0.62
    # and hid far buckets (0.45, −0.60).
    kwargs['camera_hfov'] = 48.0
    kwargs['camera_width'] = 960
    kwargs['camera_height'] = 720
  elif mode not in _CONTROL_SANITY_JOINT_MODES and not bool(
      flags.get('isaacgym_table_push', False)):
    # NVIDIA default pose. Stock cam is eye=(1.4,-1.6,1.2) tgt=(0,0,0.35)
    # hfov=75 — too wide; the magenta cube marker also hid the object.
    hfov = 42.0
    bucket = _as_xyz(flags.get('isaacgym_fixed_target_xyz'), (0.5, -0.3, 0.4))
    eye, tgt = _nvidia_init_video_cam(bucket, _goal_z, hfov)
    kwargs['camera_hfov'] = hfov
    kwargs['camera_width'] = 960
    kwargs['camera_height'] = 720
    kwargs['camera_eye'] = eye
    kwargs['camera_tgt'] = tgt
  return kwargs


def _nvidia_init_video_cam(bucket, goal_z, hfov=42.0):
  """Closer side/elevated cam on NVIDIA in-hand start + the throw bucket."""
  palm = np.array([0.03, 0.02, 0.56], dtype=np.float64)
  bucket = np.asarray(bucket, dtype=np.float64).reshape(3)
  gz = float(goal_z) if float(goal_z) >= 0.0 else float(bucket[2] + 0.05)
  goal = np.array([bucket[0], bucket[1], gz], dtype=np.float64)
  pts = np.stack([palm, bucket, goal], axis=0)
  tgt = pts.mean(axis=0)
  span = float(np.max(np.linalg.norm(pts - tgt[None, :], axis=1))) + 0.12
  half = np.deg2rad(float(hfov) * 0.5)
  dist = max(1.15, span / max(float(np.tan(half)), 1e-3) * 1.12)
  direction = np.array([0.78, -0.72, 0.48], dtype=np.float64)
  direction /= float(np.linalg.norm(direction))
  eye = tgt + direction * dist
  return (float(eye[0]), float(eye[1]), float(eye[2])), (
      float(tgt[0]), float(tgt[1]), float(tgt[2]))


def _diag(env):
  task = env._env
  control_mode = str(getattr(env, 'control_sanity_mode', '') or '')
  if control_mode in _CONTROL_SANITY_JOINT_MODES:
    full_q = bool(getattr(env, 'control_sanity_full_q', False))
    idxs = getattr(env, 'control_sanity_hand_indices', None)
    if full_q:
      q = task.arm_hand_dof_pos[0, :23].detach().cpu().numpy()
    else:
      hand_q = task.arm_hand_dof_pos[0, 7:23]
      if idxs is not None:
        q = hand_q[list(idxs)].detach().cpu().numpy()
      else:
        width = int(env.goal_dim)
        q = hand_q[:width].detach().cpu().numpy()
    # Success always uses physical radians, independent of packed coordinates.
    phys = getattr(env, '_control_sanity_goal_q_batch', None)
    if phys is not None:
      goal = phys[0].detach().cpu().numpy()
    else:
      goal = env._goal_batch[0].detach().cpu().numpy()
    max_err = float(np.max(np.abs(q - goal)))
    mean_err = float(np.mean(np.abs(q - goal)))
    if full_q:
      full_idxs = list(range(len(goal)))
    else:
      full_idxs = [7 + int(i) for i in idxs] if idxs is not None else list(
          range(7, 7 + len(goal)))
    qd_raw = task.arm_hand_dof_vel[0, full_idxs].detach().cpu().numpy()
    qd_limits = env._joint_velocity_limits[full_idxs].detach().cpu().numpy()
    qd_scaled = np.clip(qd_raw / qd_limits, -1.0, 1.0)
    if control_mode == 'index_thumb_straight':
      success = mean_err <= 0.15 and max_err <= 0.30
      easy_success = success
      very_easy_success = success
    else:
      success = max_err <= 0.10
      easy_success = max_err <= 0.20
      very_easy_success = max_err <= 0.30
    return {
        'd_obj': max_err,
        'mean_abs_joint_err': mean_err,
        'mean_abs_qd_scaled': float(np.mean(np.abs(qd_scaled))),
        'max_abs_qd_scaled': float(np.max(np.abs(qd_scaled))),
        'succ': success,
        'obj': q,
        'goal': goal,
        'tol': 0.10,
        'd_palm': None,
        'control_mode': control_mode,
        'index_q': q,
        'index_goal': goal,
        'easy_succ': easy_success,
        'very_easy_succ': very_easy_success,
        'in_bucket': False,
    }
  obj = task.object_pos[0].detach().cpu().numpy()
  goal = task.goal_pos[0].detach().cpu().numpy()
  palm = task.palm_center_pos[0].detach().cpu().numpy()
  d_obj = float(np.linalg.norm(obj - goal))
  tol = float(getattr(task, 'success_tolerance', 0.075))
  goal_z = getattr(env, 'goal_z', None)
  if getattr(env, '_bucket_xyz', None) is not None:
    bucket = np.asarray(env._bucket_xyz, dtype=np.float64)
  else:
    bucket = np.asarray(goal, dtype=np.float64).copy()
    if goal_z is None:
      bucket[2] -= 0.05
  if hasattr(env, 'in_bucket'):
    in_bucket = bool(env.in_bucket()[0].item())
    counted = bool(env.success()[0].item())
  else:
    xy = float(np.hypot(obj[0] - bucket[0], obj[1] - bucket[1]))
    z_in = float(obj[2] - bucket[2])
    in_bucket = bool(xy <= 0.12 and 0.0 <= z_in <= 0.198)
    counted = in_bucket
  throw_success = str(getattr(env, 'throw_success', 'in_bucket') or 'in_bucket')
  scale = float(getattr(task, 'keypoint_scale', 1.5))
  out = {
      'd_obj': d_obj,
      'succ': counted,
      'in_bucket': in_bucket,
      'throw_success': throw_success,
      'obj': obj,
      'goal': goal,
      'bucket': bucket,
      'tol': tol,
      'd_palm': None,
      'palm': palm,
      'd_palm_obj': float(np.linalg.norm(palm - obj)),
      'control_mode': control_mode,
  }
  if getattr(env, 'palm_goal', False):
    palm = task.palm_center_pos[0].detach().cpu().numpy()
    palm_g = env._goal_batch[0, :3].detach().cpu().numpy()
    out['d_palm'] = float(np.linalg.norm(palm - palm_g))
    out['palm'] = palm
    out['palm_g'] = palm_g
  if not bool(getattr(env, 'table_push', False)):
    if goal_z is not None:
      # Orange marker is the NF / env.goal_pos (near-rim), not the floor.
      out['goal'] = np.asarray(goal, dtype=np.float64)
      out['tol'] = tol
      out['goal_label'] = f'NF GOAL  z={float(goal_z):g}'
    elif throw_success == 'nvidia_goal':
      out['tol'] = tol * scale
      out['goal_label'] = f'NVIDIA GOAL  r={out["tol"]*100:.1f}cm'
    elif throw_success == 'goal_ball':
      out['tol'] = tol
      out['goal_label'] = f'GOAL BALL  r={out["tol"]*100:.1f}cm'
    else:
      # Default throw success is the bucket cylinder, not the goal ball.
      out['goal'] = bucket
      out['tol'] = 0.12
      out['goal_label'] = 'BUCKET  r=12cm'
  return out


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


def _cam_matrices(env, width, height):
  eye = getattr(env, '_cam_eye', None)
  tgt = getattr(env, '_cam_tgt', None)
  hfov = float(getattr(env, '_cam_hfov', 75.0) or 75.0)
  if eye is None or tgt is None:
    return None, None
  view = _look_at(eye, tgt)
  proj = _perspective(hfov, float(width) / max(float(height), 1.0))
  return view, proj


def _draw_goal_markers(rgb, env, diag):
  """Draw object-goal center + 7.5 cm success ring in the camera image."""
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return rgb
  h, w = rgb.shape[:2]
  if diag.get('control_mode') in _CONTROL_SANITY_JOINT_MODES:
    img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    goal = np.asarray(diag['index_goal'])
    mode = diag['control_mode']
    if mode in (
            'three2', 'four2', 'four2h', 'four2m', 'four2mh', 'four2mm',
            'four2mmh', 'four2mmx', 'four2w'):
      labels = ('idx', 'mid', 'ring', 'thumb')
      parts = []
      for i, lab in enumerate(labels):
        if 2 * i + 1 >= len(goal):
          break
        parts.append(
            f'{lab}=({goal[2 * i]:.2f},{goal[2 * i + 1]:.2f})')
      goal_text = 'target ' + ' '.join(parts)
    elif mode == 'index_thumb_straight':
      goal_text = (
          'target index=(' + ', '.join(f'{v:.2f}' for v in goal[:4]) + ')'
          '  thumb=(' + ', '.join(f'{v:.2f}' for v in goal[4:8]) + ')')
    else:
      goal_text = (
          'target index=(' + ', '.join(f'{v:.2f}' for v in goal[:4]) + ')')
      if len(goal) >= 6:
        goal_text += (
            '  middle=(' + ', '.join(f'{v:.2f}' for v in goal[4:]) + ')')
    y0 = h - 48
    draw.rectangle([6, y0 - 4, w - 6, h - 5], fill=(0, 0, 0, 175))
    draw.text((12, y0), goal_text, fill=(245, 245, 245, 255))
    if mode == 'index_thumb_straight':
      status_text = (
          f'mean error={diag["mean_abs_joint_err"]:.3f} rad '
          f'max={diag["d_obj"]:.3f} | qd*=0 scaled abs '
          f'mean={diag["mean_abs_qd_scaled"]:.3f} '
          f'max={diag["max_abs_qd_scaled"]:.3f} | '
          f'position-only success:{int(diag["succ"])}')
    else:
      status_text = (
          f'max error={diag["d_obj"]:.3f} rad | '
          f'hard:{int(diag["succ"])}  easy:{int(diag["easy_succ"])}  '
          f'very easy:{int(diag["very_easy_succ"])}')
    draw.text(
        (12, y0 + 20), status_text, fill=(255, 230, 160, 255))
    return np.asarray(img.convert('RGB'), dtype=np.uint8)
  view, proj = _cam_matrices(env, w, h)
  if view is None:
    return rgb
  goal = np.asarray(diag['goal'], dtype=np.float64)
  tol = float(diag['tol'])
  obj_uv = _project(goal, view, proj, w, h)
  ring = []
  for k in range(48):
    ang = 2.0 * np.pi * k / 48.0
    xyz = goal + np.array([tol * np.cos(ang), tol * np.sin(ang), 0.0])
    uv = _project(xyz, view, proj, w, h)
    if uv is not None:
      ring.append(uv)
  palm_uv = None
  if diag.get('palm_g') is not None:
    palm_uv = _project(np.asarray(diag['palm_g'], dtype=np.float64), view, proj, w, h)
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  if len(ring) >= 3:
    draw.polygon([(int(u), int(v)) for u, v in ring],
                 fill=(255, 140, 40, 55), outline=(255, 170, 50, 230))
    draw.line([(int(u), int(v)) for u, v in ring + ring[:1]],
              fill=(255, 200, 60, 255), width=3)
  live = np.asarray(diag['obj'], dtype=np.float64)
  palm_xyz = diag.get('palm')
  palm_live_uv = None
  if palm_xyz is not None:
    palm_live_uv = _project(np.asarray(palm_xyz, dtype=np.float64), view, proj, w, h)
  if obj_uv is not None:
    u, v = int(obj_uv[0]), int(obj_uv[1])
    draw.ellipse([u - 7, v - 7, u + 7, v + 7], fill=(255, 120, 20, 255),
                 outline=(255, 255, 255, 255))
    goal_label = str(diag.get('goal_label') or 'OBJ GOAL  r=7.5cm')
    draw.text((u + 12, v - 16), goal_label, fill=(255, 200, 80, 255))
  if palm_live_uv is not None:
    u, v = int(palm_live_uv[0]), int(palm_live_uv[1])
    draw.ellipse([u - 6, v - 6, u + 6, v + 6], fill=(80, 220, 255, 255),
                 outline=(255, 255, 255, 255))
    draw.text((u + 10, v + 8), 'palm', fill=(80, 220, 255, 255))
  if palm_uv is not None:
    u, v = int(palm_uv[0]), int(palm_uv[1])
    draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(80, 220, 255, 255),
                 outline=(255, 255, 255, 255))
    draw.text((u + 12, v + 6), 'PALM GOAL', fill=(80, 220, 255, 255))
  gx, gy, gz = (float(goal[0]), float(goal[1]), float(goal[2]))
  ox, oy, oz = (float(live[0]), float(live[1]), float(live[2]))
  d_po = diag.get('d_palm_obj')
  extra = '' if d_po is None else f'  |palm-cube|={float(d_po)*100:.1f}cm'
  draw.text((8, h - 22),
            f'cube=({ox:.2f},{oy:.2f},{oz:.2f})  '
            f'goal=({gx:.2f},{gy:.2f},{gz:.3f})  '
            f'orange=goal{extra}',
            fill=(255, 230, 180, 255))
  return np.asarray(img.convert('RGB'), dtype=np.uint8)


def _goal_stats_from_learner(ckpt_path: str, iteration: int, goal_dim: int):
  """NF goal mean/std at this ckpt iteration (needed for normalize_goals)."""
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
  path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
  mean = np.zeros((goal_dim,), dtype=np.float32)
  std = np.ones((goal_dim,), dtype=np.float32)
  if not os.path.isfile(path):
    return mean, std
  best = None
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      try:
        it = int(float(row.get('iteration', '')))
      except (TypeError, ValueError):
        continue
      if best is None or abs(it - int(iteration)) < abs(best[0] - int(iteration)):
        best = (it, row)
  if best is None:
    return mean, std
  row = best[1]
  for i in range(int(goal_dim)):
    try:
      mean[i] = float(row[f'nf/goal_mean_{i}'])
      std[i] = float(row[f'nf/goal_std_{i}'])
    except (KeyError, TypeError, ValueError):
      pass
  std = np.maximum(std, 1e-6)
  print(f'[ckpt_video] NF goal stats from learner iter={best[0]} '
        f'(ckpt iter={iteration})', flush=True)
  return mean, std


def _render_reward_strip(rewards, success, t, width, height=160, title='',
                         value_label='log p', line_color=(125, 255, 176)):
  """PIL-only sparkline (sgcrl_isaacgym has no matplotlib)."""
  from PIL import Image, ImageDraw

  T = max(len(rewards), 1)
  r_now = float(rewards[t])
  r_min = float(np.min(rewards)) - 0.35
  r_max = float(np.max(rewards)) + 0.35
  if r_max <= r_min:
    r_max = r_min + 1.0
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  succ_now = bool(success[t] >= 0.5)
  img = Image.new('RGB', (int(width), int(height)), (15, 20, 25))
  draw = ImageDraw.Draw(img)
  pad_l, pad_r, pad_t, pad_b = 10, 150, 28, 10
  x0, y0 = pad_l, pad_t
  x1, y1 = int(width) - pad_r, int(height) - pad_b
  plot_w = max(x1 - x0, 1)
  plot_h = max(y1 - y0, 1)

  def _xy(i, r):
    x = x0 + int(round((i / max(T - 1, 1)) * plot_w))
    y = y1 - int(round(((float(r) - r_min) / (r_max - r_min)) * plot_h))
    return x, y

  draw.rectangle([x0, y0, x1, y1], outline=(58, 70, 84))
  fade = [_xy(i, rewards[i]) for i in range(T)]
  if len(fade) >= 2:
    draw.line(fade, fill=(90, 106, 122), width=1)
  live = [_xy(i, rewards[i]) for i in range(t + 1)]
  if len(live) >= 2:
    draw.line(live, fill=line_color, width=2)
  cx, cy = _xy(t, r_now)
  draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 229, 102))
  draw.line([(cx, y0), (cx, y1)], fill=(255, 229, 102))
  if first_succ >= 0:
    sx, _ = _xy(first_succ, rewards[first_succ])
    draw.line([(sx, y0), (sx, y1)], fill=(255, 138, 76))
  draw.text((8, 6), title or 'raw NF  r = log p(g|s,a)', fill=(232, 238, 244))
  badge = f'{value_label} = {r_now:+.2f}'
  status = 'SUCCESS' if succ_now else 'no success'
  draw.text((x1 + 8, y0 + 8), value_label, fill=(154, 167, 181))
  draw.text((x1 + 8, y0 + 32), badge, fill=line_color)
  draw.text((x1 + 8, y0 + 56), status, fill=(125, 255, 176) if succ_now else (255, 138, 76))
  draw.text((x1 + 8, y0 + 80), f't = {t}/{T - 1}', fill=(200, 208, 216))
  return np.asarray(img, dtype=np.uint8)


def _stack_reward_overlay(rgb, rewards, success, t, title, *,
                          values=None, advantages=None):
  from PIL import Image
  resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
  w = int(rgb.shape[1])
  if w % 2:
    w -= 1
  strips = [
      _render_reward_strip(
          rewards, success, t, width=w, title=title,
          value_label='raw log p', line_color=(125, 255, 176))
  ]
  if values is not None:
    strips.append(_render_reward_strip(
        values, success, t, width=w, height=135,
        title='critic value  V(s, g_task)',
        value_label='V', line_color=(90, 170, 255)))
  if advantages is not None:
    strips.append(_render_reward_strip(
        advantages, success, t, width=w, height=135,
        title='GAE advantage  (frozen reward-normalizer scale)',
        value_label='A_GAE', line_color=(210, 125, 255)))
  rh = int(round(rgb.shape[0] * (w / rgb.shape[1])))
  render_r = np.asarray(
      Image.fromarray(rgb).resize((w, rh), resample))
  out = np.concatenate(strips + [render_r], axis=0)
  h, ww = out.shape[:2]
  if h % 2 or ww % 2:
    out = out[: h - (h % 2), : ww - (ww % 2)]
  return out


def _annotate(rgb, lines):
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return rgb
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img)
  pad = 6
  line_h = 16
  w = img.size[0]
  h = pad * 2 + line_h * len(lines)
  draw.rectangle((0, 0, w, h), fill=(0, 0, 0))
  for i, line in enumerate(lines):
    draw.text((8, pad + i * line_h), line, fill=(255, 255, 255))
  return np.asarray(img, dtype=np.uint8)


def _horizon_status(t, mark):
  if int(mark) <= 0:
    return None
  mark = int(mark)
  t = int(t)
  if t < mark:
    return (
        (40, 90, 50),
        f'before T={mark}  t={t}  ({mark - t} steps until T={mark} ends)')
  if t == mark:
    return ((255, 210, 40), f'T={mark} ENDS  (train-length marker)')
  return (
      (220, 90, 30),
      f'PAST T={mark}  t={t}  (+{t - mark} extra steps)')


def _right_hud_box(img_w, box_w=200):
  box_w = min(int(box_w), max(80, img_w - 10))
  return img_w - box_w - 6, box_w


def _draw_horizon_bar(rgb, t, mark):
  """No-op. The T=50 / past-horizon chip is no longer drawn."""
  del t, mark
  return rgb


def _set_camera(env, eye, target, hfov=None):
  """Reposition env-0 RGB camera (and optionally record a new hfov for overlays)."""
  from isaacgym import gymapi

  task = env._env
  eye = np.asarray(eye, dtype=np.float64).reshape(3)
  target = np.asarray(target, dtype=np.float64).reshape(3)
  task.gym.set_camera_location(
      env._cam_handle, task.envs[0],
      gymapi.Vec3(float(eye[0]), float(eye[1]), float(eye[2])),
      gymapi.Vec3(float(target[0]), float(target[1]), float(target[2])))
  env._cam_eye = (float(eye[0]), float(eye[1]), float(eye[2]))
  env._cam_tgt = (float(target[0]), float(target[1]), float(target[2]))
  if hfov is not None:
    env._cam_hfov = float(hfov)


def _zoom_camera_on_hand(env):
  """Close-up palm-centered camera so the hand fills the frame."""
  palm = np.asarray(env._palm_xyz()[0].detach().cpu().numpy(), dtype=np.float64)
  # Close eye + construction-time hfov=35 so only the hand is visible.
  eye = palm + np.array([0.22, -0.20, 0.12], dtype=np.float64)
  _set_camera(env, eye, palm, hfov=35.0)
  print(f'[ckpt_video] hand close-up cam eye={tuple(np.round(eye, 3))} '
        f'tgt={tuple(np.round(palm, 3))} hfov=35', flush=True)


def _follow_cube_camera(env):
  """Side camera locked on the live cube so a throw stays in frame."""
  obj = np.asarray(env._object_xyz()[0].detach().cpu().numpy(), dtype=np.float64)
  tgt = obj.copy()
  eye = tgt + np.array([1.15, 0.12, 0.42], dtype=np.float64)
  hfov = float(getattr(env, '_cam_hfov', 48.0) or 48.0)
  _set_camera(env, eye, tgt, hfov=hfov)


def _policy_caption(ep, t, d):
  if d.get('control_mode') in _CONTROL_SANITY_JOINT_MODES:
    if d['control_mode'] == 'index_thumb_straight':
      return (
          f'ep{ep} t={t:3d} index+thumb '
          f'mean|q-q*|={d["mean_abs_joint_err"]:.3f}rad '
          f'max={d["d_obj"]:.3f}rad balanced={int(d["succ"])}')
    return (
        f'ep{ep} t={t:3d} finger max|q-q*|={d["d_obj"]:.3f}rad '
        f'hard={int(d["succ"])} easy={int(d["easy_succ"])} '
        f'very_easy={int(d["very_easy_succ"])}')
  obj = d['obj']
  g = d['goal']
  dist_cm = float(d['d_obj']) * 100.0
  tol_cm = float(d.get('tol', 0.075)) * 100.0
  in_b = 'IN BUCKET' if d.get('in_bucket') else 'NOT IN BUCKET'
  counted = 'SUCCESS' if d['succ'] else 'NO SUCCESS'
  line = (
      f'episode {ep}  step {t:3d}  '
      f'{in_b}  {counted}  '
      f'|cube-goal|={dist_cm:.1f}cm  '
      f'cube=({obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f})  '
      f'goal=({g[0]:.2f},{g[1]:.2f},{g[2]:.3f})')
  if d.get('d_palm_obj') is not None:
    line += f'  |palm-cube|={d["d_palm_obj"]*100:.1f}cm'
  if d['d_palm'] is not None:
    line += f'  |palm-g|={d["d_palm"]:.3f}m'
  return line


def _load_banner_fonts(big=42, small=22):
  try:
    from PIL import ImageFont
    font = ImageFont.truetype(
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', int(big))
    small_f = ImageFont.truetype(
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', int(small))
    return font, small_f
  except (OSError, IOError, ImportError):
    from PIL import ImageFont
    d = ImageFont.load_default()
    return d, d


def _text_w(draw, text, font):
  try:
    return draw.textlength(text, font=font)
  except AttributeError:
    return font.getsize(text)[0]


def _draw_status_hud(rgb, d):
  """Compact right-side IN BUCKET / SUCCESS chips (below the T= marker)."""
  if d.get('control_mode') in _CONTROL_SANITY_JOINT_MODES:
    return rgb
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return rgb
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img)
  font, small = _load_banner_fonts(14, 11)
  w, h = img.size
  in_b = bool(d.get('in_bucket'))
  succ = bool(d.get('succ'))
  dist_cm = float(d.get('d_obj', 0.0)) * 100.0
  x0, box_w = _right_hud_box(w)
  y0 = 60
  row_h = 18
  c1 = (20, 150, 70) if in_b else (140, 40, 40)
  draw.rectangle((x0, y0, x0 + box_w, y0 + row_h), fill=c1)
  draw.text(
      (x0 + 6, y0 + 2),
      'in bucket' if in_b else 'not in bucket',
      fill=(255, 255, 255), font=font)
  y1 = y0 + row_h + 2
  c2 = (20, 170, 60) if succ else (150, 50, 20)
  draw.rectangle((x0, y1, x0 + box_w, y1 + row_h), fill=c2)
  draw.text(
      (x0 + 6, y1 + 2),
      'success' if succ else 'no success',
      fill=(255, 255, 255), font=font)
  y2 = y1 + row_h + 2
  draw.rectangle((x0, y2, x0 + box_w, y2 + 16), fill=(0, 0, 0))
  draw.text(
      (x0 + 6, y2 + 2),
      f'|cube-goal|={dist_cm:.1f}cm',
      fill=(255, 230, 160), font=small)
  return np.asarray(img.convert('RGB'), dtype=np.uint8)


def _draw_event_banner(rgb, *, reset=False, succ_now=False, first_succ_t=None,
                       t=0, episode=0):
  """Compact right-side RESET / SUCCESS chip. Does not cover the scene."""
  if not (reset or succ_now or first_succ_t is not None):
    return rgb
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return rgb
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img)
  w, h = img.size
  font, small = _load_banner_fonts(14, 11)
  x0, box_w = _right_hud_box(w)
  y0 = 118
  if reset:
    draw.rectangle((x0, y0, x0 + box_w, y0 + 36), fill=(210, 40, 40))
    draw.text((x0 + 6, y0 + 3), f'reset  ep {episode}',
              fill=(255, 255, 255), font=font)
    draw.text((x0 + 6, y0 + 18), 'new episode',
              fill=(255, 240, 180), font=small)
  elif succ_now:
    draw.rectangle((x0, y0, x0 + box_w, y0 + 36), fill=(20, 170, 60))
    draw.text((x0 + 6, y0 + 3), 'success this frame',
              fill=(255, 255, 255), font=font)
    draw.text((x0 + 6, y0 + 18), f'ep {episode}  t={t}',
              fill=(230, 255, 230), font=small)
  elif first_succ_t is not None:
    draw.rectangle((x0, y0, x0 + box_w, y0 + 18), fill=(10, 90, 40))
    draw.text(
        (x0 + 6, y0 + 2),
        f'success at t={int(first_succ_t)}',
        fill=(200, 255, 210), font=small)
  return np.asarray(img.convert('RGB'), dtype=np.uint8)


def _make_episode_break_card(shape, ended_ep, next_ep, n_frames=60):
  """Full-screen pause card between concatenated episodes."""
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return [np.zeros(shape, dtype=np.uint8)] * int(n_frames)
  h, w = int(shape[0]), int(shape[1])
  img = Image.new('RGB', (w, h), (25, 20, 20))
  draw = ImageDraw.Draw(img)
  font, small = _load_banner_fonts(52, 26)
  lines = [
      (font, (255, 230, 80), 'PAUSED  —  EPISODE RESET'),
      (small, (255, 255, 255),
       f'episode {ended_ep} ended   →   episode {next_ep} starts'),
      (small, (255, 210, 160),
       'The cube jumping back into the PALM is a NEW rollout.'),
      (small, (255, 210, 160),
       'This video concatenates 2 independent episodes.'),
      (small, (200, 220, 255),
       'Next clip starts from the usual in-hand keep-arm reset.'),
  ]
  y = int(h * 0.26)
  for fnt, fill, text in lines:
    draw.text(((w - _text_w(draw, text, fnt)) * 0.5, y), text, fill=fill, font=fnt)
    y += 64 if fnt is font else 42
  card = np.asarray(img, dtype=np.uint8)
  return [card] * int(n_frames)


def _capture_labeled_frame(env, ckpt_name, line, d, goal_ref_rgb=None,
                           *, t=0, mark_horizon=0, episode=0,
                           reset=False, first_succ_t=None):
  rgb = env.render_rgb()
  if rgb is None:
    raise RuntimeError('camera returned no frame; enable_cameras failed')
  rgb = _annotate(rgb, [ckpt_name, line])
  rgb = _draw_horizon_bar(rgb, t, mark_horizon)
  rgb = _draw_goal_markers(rgb, env, d)
  rgb = _draw_status_hud(rgb, d)
  rgb = _draw_event_banner(
      rgb, reset=reset, succ_now=bool(d.get('succ')) and not reset,
      first_succ_t=None if reset or d.get('succ') else first_succ_t,
      t=t, episode=episode)
  if goal_ref_rgb is not None:
    rgb = _stack_goal_above_policy(goal_ref_rgb, rgb)
  return rgb


def _set_finger_goal_pose(env):
  """Teleport controlled hand joints to the physical radian goal."""
  import torch
  from isaacgym import gymtorch

  task = env._env
  phys = getattr(env, '_control_sanity_goal_q_batch', None)
  if phys is None:
    raise RuntimeError('control-sanity video needs _control_sanity_goal_q_batch')
  q = phys[0].detach().to(dtype=torch.float, device=task.device)
  idxs = getattr(env, 'control_sanity_hand_indices', None)
  if bool(getattr(env, 'control_sanity_full_q', False)):
    n = int(q.numel())
    task.arm_hand_dof_pos[:, :n] = q
    task.cur_targets[:, :n] = q
    task.prev_targets[:, :n] = q
  elif idxs is None:
    n = int(q.numel())
    task.arm_hand_dof_pos[:, 7:7 + n] = q
    task.cur_targets[:, 7:7 + n] = q
    task.prev_targets[:, 7:7 + n] = q
  else:
    dof_idx = torch.tensor(
        [7 + int(i) for i in idxs], dtype=torch.long, device=task.device)
    task.arm_hand_dof_pos[:, dof_idx] = q
    task.cur_targets[:, dof_idx] = q
    task.prev_targets[:, dof_idx] = q
  task.arm_hand_dof_vel[:, :] = 0.0
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))
  for _ in range(12):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
  task.compute_observations()


def _capture_goal_reference_rgb(env, caption_prefix=''):
  """Render one static GOAL hand pose frame (after teleport)."""
  _set_finger_goal_pose(env)
  rgb = env.render_rgb()
  if rgb is None:
    raise RuntimeError('camera returned no frame for GOAL reference')
  d = _diag(env)
  line = (
      f'GOAL pose (static)  max|q-q*|={d["d_obj"]:.3f}rad  '
      f'(should be ~0 after teleport)')
  rgb = _annotate(rgb, [caption_prefix or 'GOAL', line])
  return rgb


def _stack_goal_above_policy(goal_rgb, live_rgb):
  """Vertical composite: GOAL on top, live policy below."""
  from PIL import Image, ImageDraw

  g = np.ascontiguousarray(goal_rgb)
  l = np.ascontiguousarray(live_rgb)
  w = min(int(g.shape[1]), int(l.shape[1]))
  if w % 2:
    w -= 1
  resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)

  def _fit(arr):
    h = int(round(arr.shape[0] * (w / max(arr.shape[1], 1))))
    return np.asarray(Image.fromarray(arr).resize((w, h), resample))

  g2, l2 = _fit(g), _fit(l)
  # Thin label bars so the split is obvious even without reading overlays.
  bar_h = 22
  top_bar = Image.new('RGB', (w, bar_h), (24, 48, 28))
  bot_bar = Image.new('RGB', (w, bar_h), (28, 28, 48))
  ImageDraw.Draw(top_bar).text((8, 4), 'GOAL (target finger config)', fill=(180, 255, 180))
  ImageDraw.Draw(bot_bar).text((8, 4), 'POLICY (live)', fill=(180, 200, 255))
  out = np.concatenate(
      [np.asarray(top_bar), g2, np.asarray(bot_bar), l2], axis=0)
  h, ww = out.shape[:2]
  if h % 2 or ww % 2:
    out = out[: h - (h % 2), : ww - (ww % 2)]
  return out


def _build_env(kwargs):
  from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
  return AllegroKukaThrowVecEnv(**kwargs)


def _set_reset_seed(seed: int) -> None:
  """Seed reset-time NumPy / Python / Torch randomness reproducibly."""
  import torch

  random.seed(int(seed))
  np.random.seed(int(seed))
  torch.manual_seed(int(seed))
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(seed))


def _load_ckpt(ckpt_path: str):
  with open(ckpt_path, 'rb') as fh:
    return pickle.load(fh)


def _is_rnd_ckpt(ckpt) -> bool:
  if not (isinstance(ckpt, tuple) and len(ckpt) >= 2):
    return False
  params = ckpt[0]
  return isinstance(params, dict) and 'policy' in params


def _is_mpo_ckpt(ckpt) -> bool:
  return (
      isinstance(ckpt, dict)
      and 'state' in ckpt
      and hasattr(ckpt['state'], 'target_policy_params'))


def _rnd_iteration(ckpt_path: str) -> int:
  stem = os.path.splitext(os.path.basename(ckpt_path))[0]
  if stem.startswith('params_'):
    try:
      return int(stem.rsplit('_', 1)[-1])
    except ValueError:
      return -1
  return -1


def _load_ppo_rnd():
  """Load ``baseline-agents/ppo-rnd.py`` by path (hyphenated filename)."""
  import importlib.util

  module_path = os.path.join(_REPO, 'baseline-agents', 'ppo-rnd.py')
  spec = importlib.util.spec_from_file_location('sgcrl_ppo_rnd', module_path)
  if spec is None or spec.loader is None:
    raise ImportError(f'Could not load PPO+RND from {module_path}')
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _build_rnd_actor(env, ckpt, deterministic: bool, ckpt_path: str):
  import sgcrl_jax_acme_compat  # noqa: F401
  import jax
  import jax.numpy as jnp

  ppo_rnd = _load_ppo_rnd()
  params, normalizer = ckpt[0], ckpt[1]
  policy_params = {'policy': params['policy'], 'normalizer': normalizer}
  iteration = _rnd_iteration(ckpt_path)
  rnd_args = ppo_rnd.Args()
  ppo_network = ppo_rnd.make_ppo_networks(
      rnd_args, int(env.action_dim), include_auxiliary=False)
  policy = ppo_rnd.make_inference_fn(ppo_network)(
      policy_params, deterministic=bool(deterministic))
  obs_dim = int(env.obs_dim)
  print(f'[ckpt_video] loaded RND params={os.path.basename(ckpt_path)}  '
        f'eval_step={iteration}  det={deterministic}  '
        f'obs_dim={env.obs_dim} goal_dim={env.goal_dim} '
        f'palm_goal={getattr(env, "palm_goal", False)} '
        f'joint_goal={getattr(env, "joint_goal", False)}',
        flush=True)

  @jax.jit
  def act(_params, obs, key):
    action, _ = policy(obs[..., :obs_dim], obs[..., obs_dim:], key)
    return jnp.clip(action, -1.0, 1.0)

  gpu = jax.devices('gpu')
  if gpu:
    policy_params = jax.device_put(policy_params, gpu[0])
  return act, policy_params, iteration, None


def _build_mpo_actor(env, ckpt, flags: dict, deterministic: bool,
                     ckpt_path: str = ''):
  import sgcrl_jax_acme_compat  # noqa: F401
  import jax
  import jax.numpy as jnp
  from acme import specs as acme_specs

  sys.path.insert(0, os.path.join(_REPO, 'baseline-agents'))
  import mpo_crl_learner  # noqa: E402

  hidden, _ = _load_run_hparams(flags)
  raw_h = flags.get('mpo_policy_hidden_sizes')
  if raw_h:
    hidden = tuple(int(x) for x in str(raw_h).split(',') if str(x).strip())
  init_scale = float(flags.get('mpo_policy_init_scale', 0.7))
  packed_dim = int(env.obs_dim + env.goal_dim)
  spec = acme_specs.EnvironmentSpec(
      observations=acme_specs.Array(
          shape=(packed_dim,), dtype=np.float32, name='observation'),
      actions=acme_specs.BoundedArray(
          shape=(int(env.action_dim),), dtype=np.float32,
          minimum=-1.0, maximum=1.0, name='action'),
      rewards=acme_specs.Array(shape=(), dtype=np.float32, name='reward'),
      discounts=acme_specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0,
          name='discount'),
  )
  policy_net = mpo_crl_learner.make_mpo_policy(
      spec, hidden_sizes=hidden, init_scale=init_scale)
  policy_params = ckpt['state'].target_policy_params
  iteration = int(ckpt.get('iteration', -1))
  print(f'[ckpt_video] loaded MPO {os.path.basename(ckpt_path) or "ckpt"}  '
        f'iter={iteration}  det={deterministic}  hidden={hidden}',
        flush=True)

  @jax.jit
  def act(params, obs, key):
    dist = policy_net.apply(params, obs)
    action = dist.mode() if deterministic else dist.sample(seed=key)
    return jnp.clip(action, -1.0, 1.0)

  gpu = jax.devices('gpu')
  if gpu:
    policy_params = jax.device_put(policy_params, gpu[0])
  return act, policy_params, iteration, None


def _build_actor(env, ckpt, flags: dict, deterministic: bool,
                 ckpt_path: str = ''):
  if _is_rnd_ckpt(ckpt):
    return _build_rnd_actor(env, ckpt, deterministic, ckpt_path)
  if _is_mpo_ckpt(ckpt):
    return _build_mpo_actor(env, ckpt, flags, deterministic, ckpt_path)

  import sgcrl_jax_acme_compat  # noqa: F401
  import jax
  import jax.numpy as jnp
  from acme import specs as acme_specs
  from contrastive.networks import make_networks

  hidden, min_std = _load_run_hparams(flags)
  packed_dim = int(env.obs_dim + env.goal_dim)
  obs_spec = acme_specs.Array(
      shape=(packed_dim,), dtype=np.float32, name='observation')
  act_spec = acme_specs.BoundedArray(
      shape=(int(env.action_dim),), dtype=np.float32,
      minimum=-1.0, maximum=1.0, name='action')
  spec = acme_specs.EnvironmentSpec(
      observations=obs_spec, actions=act_spec,
      rewards=acme_specs.Array(shape=(), dtype=np.float32, name='reward'),
      discounts=acme_specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0,
          name='discount'),
  )
  networks = make_networks(
      spec,
      obs_dim=int(env.obs_dim),
      hidden_layer_sizes=hidden,
      actor_min_std=float(min_std),
  )
  policy_params = ckpt['policy_params']
  iteration = int(ckpt.get('iteration', -1))
  print(f'[ckpt_video] loaded iter={iteration}  '
        f'hidden={hidden}  min_std={min_std}  det={deterministic}  '
        f'obs_dim={env.obs_dim} goal_dim={env.goal_dim} '
        f'palm_goal={getattr(env, "palm_goal", False)} '
        f'joint_goal={getattr(env, "joint_goal", False)}',
        flush=True)

  sample_fn = networks.sample_eval if deterministic else networks.sample

  @jax.jit
  def act(params, obs, key):
    dist = networks.policy_network.apply(params, obs)
    action = sample_fn(dist, key)
    return jnp.clip(action, -1.0, 1.0)

  gpu = jax.devices('gpu')
  if gpu:
    policy_params = jax.device_put(policy_params, gpu[0])
  return act, policy_params, iteration, networks


def _build_nf_reward(env, ckpt, flags: dict):
  import jax
  import jax.numpy as jnp
  from contrastive import nf_density as _nf

  nf_nets = _nf.make_nf_density_networks(
      obs_dim=int(env.obs_dim),
      act_dim=int(env.action_dim),
      goal_dim=int(env.goal_dim),
      rep_size=int(flags.get('nf_rep_size') or 64),
      num_blocks=int(flags.get('nf_num_blocks') or 6),
      channels=int(flags.get('nf_coupling_width') or 192),
      goal_enc_size=int(flags.get('nf_goal_enc_size') or 0),
      sa_hidden=int(flags.get('nf_sa_hidden') or 192),
      sa_num_layers=int(flags.get('nf_sa_num_layers') or 3),
      state_only=bool(flags.get('nf_state_only', False)),
      scale_tanh=bool(flags.get('nf_scale_tanh', False)),
  )
  reward_fn = _nf.make_nf_reward_fn(
      nf_nets, obs_dim=int(env.obs_dim),
      tanh_scale=0.0, reward_mode='forward')
  nf_params = ckpt.get('q_params_ema', ckpt['q_params'])
  src = 'q_params_ema' if 'q_params_ema' in ckpt else 'q_params'
  gpu = jax.devices('gpu')
  if gpu:
    nf_params = jax.device_put(nf_params, gpu[0])
  print(f'[ckpt_video] NF reward from {src}  (raw log p)', flush=True)
  return reward_fn, nf_params


def _build_value_fn(ckpt, networks):
  import jax
  import jax.numpy as jnp

  value_params = ckpt.get('value_params')
  if value_params is None:
    raise KeyError('checkpoint has no value_params')
  gpu = jax.devices('gpu')
  if gpu:
    value_params = jax.device_put(value_params, gpu[0])

  @jax.jit
  def value_fn(params, packed):
    value = networks.value_network.apply(params, packed)
    return jnp.reshape(value, (packed.shape[0],))

  return value_fn, value_params


def _frozen_return_norm_std(ckpt, ckpt_path: str, iteration: int) -> float:
  extra = ckpt.get('extra_state') or {}
  state = extra.get('reward_normalizer')
  if isinstance(state, dict) and 'rms_var' in state:
    var = float(np.asarray(state['rms_var']).reshape(-1)[0])
    return float(np.sqrt(max(var, 0.0) + 1e-8))
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
  path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
  best = None
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      try:
        row_it = int(float(row.get('iteration', '')))
        std = float(row.get('reward_return_norm_std', ''))
      except (TypeError, ValueError):
        continue
      distance = abs(row_it - int(iteration))
      if best is None or distance < best[0]:
        best = (distance, std)
  if best is None:
    raise RuntimeError('no reward_return_norm_std in checkpoint or learner CSV')
  return float(best[1])


def _gae(rewards, values, dones_before, next_value, next_done,
         gamma: float, gae_lambda: float):
  """Match learner GAE, including rollout bootstrap and reset boundaries."""
  rewards = np.asarray(rewards, dtype=np.float32)
  values = np.asarray(values, dtype=np.float32)
  dones_before = np.asarray(dones_before, dtype=np.float32)
  next_values = np.concatenate(
      [values[1:], np.asarray([next_value], dtype=np.float32)])
  next_dones = np.concatenate(
      [dones_before[1:], np.asarray([next_done], dtype=np.float32)])
  advantage = np.zeros_like(rewards)
  last = 0.0
  for t in reversed(range(len(rewards))):
    nonterminal = 1.0 - next_dones[t]
    delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
    last = delta + gamma * gae_lambda * nonterminal * last
    advantage[t] = last
  return advantage


def _search_best_reset(env, act, policy_params, *, base_seed: int,
                       num_seeds: int, n_steps: int):
  """Find the deterministic-policy reset seed with smallest object-goal gap."""
  import jax
  import jax.numpy as jnp
  import torch

  best = None
  for offset in range(int(num_seeds)):
    seed = int(base_seed) + offset
    _set_reset_seed(seed)
    obs_t = env.reset()
    key = jax.random.PRNGKey(seed)
    min_dist = float('inf')
    first_min_t = -1
    ever_succ = False
    for t in range(int(n_steps)):
      obs = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
      if obs.ndim == 1:
        obs = obs[None]
      key, sub = jax.random.split(key)
      action = np.asarray(
          act(policy_params, jnp.asarray(obs), sub), dtype=np.float32).copy()
      obs_t, _, _ = env.step(
          torch.as_tensor(action, dtype=torch.float32, device=env.device))
      diag = _diag(env)
      if diag['d_obj'] < min_dist:
        min_dist = float(diag['d_obj'])
        first_min_t = int(t)
      ever_succ = ever_succ or bool(diag['succ'])
    candidate = (min_dist, seed, first_min_t, ever_succ)
    if best is None or candidate[0] < best[0]:
      best = candidate
    print(
        f'[ckpt_video] reset search seed={seed} '
        f'min|obj-g|={min_dist:.3f}m t={first_min_t} '
        f'succ={int(ever_succ)}',
        flush=True)
  print(
      f'[ckpt_video] reset search BEST seed={best[1]} '
      f'min|obj-g|={best[0]:.3f}m t={best[2]} succ={int(best[3])}',
      flush=True)
  return best


def _episode_grads(networks, value_params, reward_fn, nf_params,
                   packed, actions, gmean, gstd, obs_dim: int):
  import jax
  import jax.numpy as jnp

  packed_j = jnp.asarray(packed)
  actions_j = jnp.asarray(actions)
  gmean_j = jnp.asarray(gmean)
  gstd_j = jnp.asarray(gstd)
  g = packed_j[:, int(obs_dim):]
  s = packed_j[:, :int(obs_dim)]

  def _r_of_s(state, action, goal):
    obs = jnp.concatenate([state, goal], axis=-1)[None]
    return jnp.reshape(reward_fn(nf_params, obs, action[None], gmean_j, gstd_j), ())

  def _v_of_s(state, goal):
    obs = jnp.concatenate([state, goal], axis=-1)[None]
    return jnp.reshape(networks.value_network.apply(value_params, obs), ())

  def _r_norm(state, action, goal):
    return jnp.linalg.norm(jax.grad(_r_of_s)(state, action, goal))

  def _v_norm(state, goal):
    return jnp.linalg.norm(jax.grad(_v_of_s)(state, goal))

  r_n = jax.vmap(_r_norm)(s, actions_j, g)
  v_n = jax.vmap(_v_norm)(s, g)
  return np.asarray(r_n), np.asarray(v_n)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--num-steps', type=int, default=0,
                 help='0 = isaacgym_episode_length from run_config')
  p.add_argument('--episodes', type=int, default=2)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pipeline', default='gpu', choices=('gpu', 'cpu'))
  p.add_argument('--deterministic', action='store_true')
  p.add_argument('--reward-overlay', action='store_true',
                 help='Stack raw NF log p(g|s,a) timeline on top of the camera')
  p.add_argument('--value-overlay', action='store_true',
                 help='Also show checkpoint critic V(s,g) above the camera')
  p.add_argument('--gae-overlay', action='store_true',
                 help='Also show PPO GAE advantage using checkpoint reward scale')
  p.add_argument('--search-reset-seeds', type=int, default=0,
                 help='For deterministic policy, search this many consecutive '
                      'reset seeds and render the closest-to-goal trajectory')
  p.add_argument('--dump-grads', action='store_true',
                 help='Write per-step ||∇_s r|| and ||∇_s V|| next to the mp4')
  p.add_argument('--follow-cube', action='store_true', default=False,
                 help='Side camera tracks the live cube. Off by default so '
                      'the auto-framed arm+goal shot stays put.')
  p.add_argument('--no-follow-cube', dest='follow_cube', action='store_false')
  p.add_argument('--write-still', action='store_true',
                 help='Also write the first annotated frame as a sibling PNG.')
  p.add_argument('--mark-horizon', type=int, default=0,
                 help='Unused. Horizon / T=50 chip is no longer drawn.')
  p.add_argument('--flags-json', default='',
                 help='JSON object merged into run_config flags. Required '
                      'for Flax PPO+RND params_*.pkl (no run_config.json).')
  p.add_argument('--reset-banner', action='store_true',
                 help='Insert the large red RESET pause and inter-episode '
                      'break cards. Off by default (in-train videos).')
  args = p.parse_args()

  os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
  flags = _load_flags(args.checkpoint)
  if args.flags_json:
    with open(args.flags_json, 'r', encoding='utf-8') as fh:
      extra = json.load(fh)
    if not isinstance(extra, dict):
      raise ValueError('--flags-json must be a JSON object')
    flags.update(extra)
    print(f'[ckpt_video] merged flags from {args.flags_json}', flush=True)
  env_kw = _load_env_kwargs(
      flags, args.num_steps, args.seed, args.pipeline)
  print(f'[ckpt_video] env kwargs: { {k: env_kw[k] for k in env_kw if k != "seed"} }',
        flush=True)

  env = _build_env(env_kw)
  follow_cube = bool(args.follow_cube)
  ckpt = _load_ckpt(args.checkpoint)
  act, policy_params, iteration, networks = _build_actor(
      env, ckpt, flags, bool(args.deterministic), args.checkpoint)
  reward_fn = nf_params = None
  gmean = gstd = None
  need_reward = bool(
      args.reward_overlay or args.value_overlay or args.gae_overlay
      or args.dump_grads)
  need_value = bool(args.value_overlay or args.gae_overlay or args.dump_grads)
  if (need_reward or need_value) and (networks is None or _is_rnd_ckpt(ckpt)):
    raise SystemExit(
        'NF/value overlays are not supported for PPO+RND checkpoints')
  if need_reward:
    reward_fn, nf_params = _build_nf_reward(env, ckpt, flags)
    gmean, gstd = _goal_stats_from_learner(
        args.checkpoint, iteration, int(env.goal_dim))
  value_fn = value_params = None
  if need_value:
    value_fn, value_params = _build_value_fn(ckpt, networks)
  return_norm_std = None
  if args.gae_overlay and bool(flags.get('ppo_norm_reward', True)):
    return_norm_std = _frozen_return_norm_std(
        ckpt, args.checkpoint, iteration)
    print(f'[ckpt_video] frozen return-normalizer std={return_norm_std:.6g}',
          flush=True)

  import torch
  import jax
  import jax.numpy as jnp
  import imageio

  n_steps = int(env.max_episode_steps)
  render_seed = int(args.seed)
  searched_best = None
  if int(args.search_reset_seeds) > 0:
    if not args.deterministic:
      raise ValueError('--search-reset-seeds requires --deterministic')
    searched_best = _search_best_reset(
        env, act, policy_params, base_seed=int(args.seed),
        num_seeds=int(args.search_reset_seeds), n_steps=n_steps)
    render_seed = int(searched_best[1])
  frames = []
  key = jax.random.PRNGKey(render_seed)
  summaries = []
  grad_rows = []
  for ep in range(int(args.episodes)):
    _set_reset_seed(render_seed + ep)
    obs_t = env.reset()
    control_mode = str(getattr(env, 'control_sanity_mode', '') or '')
    hand_closeup = (
        control_mode in _CONTROL_SANITY_JOINT_MODES
        and control_mode != 'arm23wave')
    goal_ref_rgb = None
    if hand_closeup:
      _zoom_camera_on_hand(env)
      goal_ref_rgb = _capture_goal_reference_rgb(
          env, caption_prefix=os.path.basename(args.checkpoint))
      # Restore the randomized start so the policy episode matches training reset.
      _set_reset_seed(render_seed + ep)
      obs_t = env.reset()
      _zoom_camera_on_hand(env)
    min_obj = 1e9
    min_palm = 1e9
    ever_succ = False
    packed_list = []
    action_list = []
    succ_list = []
    done_after_list = []
    rgb_list = []
    ckpt_name = os.path.basename(args.checkpoint)
    d0 = _diag(env)
    if follow_cube:
      _follow_cube_camera(env)
    mark_h = int(args.mark_horizon)
    first_succ_t = None
    rgb_list.append(_capture_labeled_frame(
        env, ckpt_name, _policy_caption(ep, 0, d0), d0,
        goal_ref_rgb, t=0, mark_horizon=mark_h, episode=ep, reset=False))
    if args.reset_banner:
      reset_pause = 60  # 2.0s at 30fps
      reset_frame = _capture_labeled_frame(
          env, ckpt_name, _policy_caption(ep, 0, d0), d0,
          goal_ref_rgb, t=0, mark_horizon=mark_h, episode=ep, reset=True)
      rgb_list = [reset_frame] * reset_pause + rgb_list
      print(f'[ckpt_video] {_policy_caption(ep, 0, d0)}  RESET pause',
            flush=True)
    else:
      print(f'[ckpt_video] {_policy_caption(ep, 0, d0)}', flush=True)
    for t in range(n_steps):
      obs = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
      if obs.ndim == 1:
        obs = obs[None]
      key, sub = jax.random.split(key)
      action = act(policy_params, jnp.asarray(obs), sub)
      a_np = np.asarray(action, dtype=np.float32).reshape(-1)
      a_t = torch.as_tensor(a_np[None], device=env.device)
      packed_list.append(obs.reshape(-1).copy())
      action_list.append(a_np)
      obs_t, _, done_t = env.step(a_t)
      done_after_list.append(
          float(np.asarray(done_t.detach().cpu().numpy()).reshape(-1)[0]))
      d = _diag(env)
      min_obj = min(min_obj, d['d_obj'])
      ever_succ = ever_succ or d['succ']
      if d['succ'] and first_succ_t is None:
        first_succ_t = t + 1
      succ_list.append(1.0 if d['succ'] else 0.0)
      if d['d_palm'] is not None:
        min_palm = min(min_palm, d['d_palm'])
      line = _policy_caption(ep, t + 1, d)
      if follow_cube:
        _follow_cube_camera(env)
      rgb_list.append(_capture_labeled_frame(
          env, ckpt_name, line, d, goal_ref_rgb,
          t=t + 1, mark_horizon=mark_h, episode=ep,
          reset=False,
          first_succ_t=first_succ_t))
      if t % 50 == 0:
        print(f'[ckpt_video] {line}', flush=True)

    rewards = None
    if reward_fn is not None:
      try:
        packed = np.asarray(packed_list, dtype=np.float32)
        if packed.ndim == 3:
          packed = packed.reshape(packed.shape[0], -1)
        actions = np.asarray(action_list, dtype=np.float32)
        if actions.ndim == 3:
          actions = actions.reshape(actions.shape[0], -1)
        rewards = np.asarray(
            reward_fn(
                nf_params, jnp.asarray(packed), jnp.asarray(actions),
                jnp.asarray(gmean), jnp.asarray(gstd)),
            dtype=np.float32)
        success = np.asarray(succ_list, dtype=np.float32)
        values = None
        advantages = None
        if value_fn is not None:
          values = np.asarray(
              value_fn(value_params, jnp.asarray(packed)), dtype=np.float32)
        if args.gae_overlay:
          final_packed = np.asarray(
              obs_t.detach().cpu().numpy(), dtype=np.float32).reshape(1, -1)
          next_value = float(np.asarray(
              value_fn(value_params, jnp.asarray(final_packed)),
              dtype=np.float32).reshape(-1)[0])
          done_after = np.asarray(done_after_list, dtype=np.float32)
          dones_before = np.concatenate(
              [np.zeros(1, dtype=np.float32), done_after[:-1]])
          ext_scale = (
              float(flags.get('ppo_external_reward_scale', 1.0))
              if bool(flags.get('ppo_use_external_reward', False)) else 0.0)
          ext = ext_scale * success
          if return_norm_std is not None:
            if bool(flags.get('ppo_external_reward_before_norm', False)):
              ppo_rewards = (rewards + ext) / return_norm_std
            else:
              ppo_rewards = rewards / return_norm_std + ext
          else:
            ppo_rewards = rewards + ext
          gamma = float(flags.get('ppo_discount', 0.99))
          if gamma <= 0.0:
            gamma = 0.99
          gae_lambda = float(flags.get('ppo_gae_lambda', 0.95))
          advantages = _gae(
              ppo_rewards, values, dones_before, next_value, done_after[-1],
              gamma, gae_lambda)
          print(
              f'[ckpt_video] ep{ep} GAE: mean={np.mean(advantages):+.4f} '
              f'std={np.std(advantages):.4f} '
              f'min={np.min(advantages):+.4f} max={np.max(advantages):+.4f}',
              flush=True)
        title = f'{os.path.basename(args.checkpoint)}  raw log p'
        for t, rgb in enumerate(rgb_list):
          frames.append(_stack_reward_overlay(
              rgb, rewards, success, t, title,
              values=values if args.value_overlay or args.gae_overlay else None,
              advantages=advantages))
      except Exception as exc:
        print(f'[ckpt_video] reward/overlay failed ({exc}); '
              f'writing camera frames', flush=True)
        frames.extend(rgb_list)
      if args.dump_grads and value_params is not None:
        try:
          r_n, v_n = _episode_grads(
              networks, value_params, reward_fn, nf_params,
              packed, actions, gmean, gstd, int(env.obs_dim))
          grad_rows.append({
              'iteration': iteration,
              'episode': ep,
              'grad_s_r_mean': float(np.mean(r_n)),
              'grad_s_r_max': float(np.max(r_n)),
              'grad_s_v_mean': float(np.mean(v_n)),
              'grad_s_v_max': float(np.max(v_n)),
              'ever_succ': int(ever_succ),
              'logp_mean': float(np.mean(rewards)),
          })
          print(f'[ckpt_video] grads ep{ep}: ||∇_s r|| mean={np.mean(r_n):.3f} '
                f'max={np.max(r_n):.3f}  ||∇_s V|| mean={np.mean(v_n):.3f} '
                f'max={np.max(v_n):.3f}', flush=True)
        except Exception as exc:
          print(f'[ckpt_video] dump-grads failed ({exc}); continuing',
                flush=True)
    else:
      frames.extend(rgb_list)

    extra = '' if min_palm > 1e8 else f'  min|palm-g|={min_palm:.3f}m'
    summary = (
        f'ep{ep}: min|obj-g|={min_obj:.3f}m  ever_succ={int(ever_succ)}{extra}')
    summaries.append(summary)
    print(f'[ckpt_video] {summary}', flush=True)
    if searched_best is not None and ep == 0:
      print(
          f'[ckpt_video] reset search verification: '
          f'searched={searched_best[0]:.4f}m rendered={min_obj:.4f}m '
          f'|delta|={abs(searched_best[0] - min_obj):.4g}m',
          flush=True)
    if args.reset_banner and ep + 1 < int(args.episodes):
      frames.extend(_make_episode_break_card(
          frames[-1].shape, ended_ep=ep, next_ep=ep + 1, n_frames=60))

  imageio.mimsave(args.output, frames, fps=int(args.fps))
  print(f'[ckpt_video] wrote {len(frames)} frames iter={iteration} '
        f'-> {args.output}', flush=True)
  if args.write_still and frames:
    from PIL import Image
    still = os.path.splitext(args.output)[0] + '_t0.png'
    Image.fromarray(frames[0]).save(still)
    mid = frames[min(10, len(frames) - 1)]
    still10 = os.path.splitext(args.output)[0] + '_t10.png'
    Image.fromarray(mid).save(still10)
    print(f'[ckpt_video] wrote {still} and {still10}', flush=True)
  if grad_rows:
    side = os.path.splitext(args.output)[0] + '_grads.json'
    with open(side, 'w', encoding='utf-8') as fh:
      json.dump(grad_rows, fh, indent=2)
    print(f'[ckpt_video] wrote {side}', flush=True)
  for line in summaries:
    print('[ckpt_video] ' + line, flush=True)


if __name__ == '__main__':
  main()
