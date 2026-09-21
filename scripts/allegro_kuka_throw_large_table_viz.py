#!/usr/bin/env python3
"""Render Allegro center-task inits on the away-shifted 1.5 m table.

Supports:
  --init-style behind : palm stand-off behind the cube (hover/touch sweeps)
  --init-style grasp  : exact successful-run touch init — cube at (0.17,0.08),
                        IK palm on the cube, ±2 cm xy / small joint noise
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    LARGE_TABLE_BOUNDS_XY,
    LARGE_TABLE_CENTER_XY,
    LARGE_TABLE_SIZE_XY,
    TABLE_OBJECT_Z,
    TABLE_SPAWN_OBJECT_XY,
    AllegroKukaThrowVecEnv,
)

WIDTH, HEIGHT = 1200, 900
HFOV = 42.0
CENTER_GOAL = (0.0, 0.0, TABLE_OBJECT_Z)
BEHIND_SPAWN_XY = (0.17, 0.04)
GRASP_SPAWN_XY = TABLE_SPAWN_OBJECT_XY  # (0.17, 0.08)
CORRELATED_XY_RANDOMIZATION = 0.005

CUBE_Z_TOL = 0.030
MAX_XY_DISPLACEMENT = 0.005
FINGERTIP_MIN = 0.055
FINGERTIP_MAX = 0.075
GRASP_MAX_PALM_OBJ = 0.12

VIEWS = {
    "robot": ((0.78, 0.86, 0.96), (0.17, 0.10, 0.59)),
    "away": ((0.72, -0.48, 0.92), (0.17, 0.09, 0.59)),
    "top": ((0.19, 0.08, 1.38), (0.17, 0.07, 0.55)),
}


def _set_camera(env, eye, target) -> None:
  from isaacgym import gymapi

  task = env._env
  task.gym.set_camera_location(
      env._cam_handle, task.envs[0],
      gymapi.Vec3(*eye), gymapi.Vec3(*target))


def _hand_cube_contacts(env) -> list:
  task = env._env
  object_bodies = {int(v) for v in task.object_rb_handles.tolist()}
  hand_bodies = set(range(int(task.num_hand_arm_bodies)))
  pairs = []
  for contact in task.gym.get_env_rigid_contacts(task.envs[0]):
    try:
      b0, b1 = int(contact["body0"]), int(contact["body1"])
    except (KeyError, TypeError):
      b0, b1 = int(contact.body0), int(contact.body1)
    if ((b0 in object_bodies and b1 in hand_bodies)
        or (b1 in object_bodies and b0 in hand_bodies)):
      pairs.append((b0, b1))
  return pairs


def _evaluate_behind(obj, fingertip_dist, initial_pairs, cur_pairs, xy_disp,
                     fmin, fmax, max_xy_disp, mode):
  reasons = []
  if abs(float(obj[2]) - TABLE_OBJECT_Z) > CUBE_Z_TOL:
    reasons.append(
        f"cube off table z={float(obj[2]):.3f} (want {TABLE_OBJECT_Z:.3f})")
  if xy_disp > max_xy_disp:
    reasons.append(f"cube shoved xy={xy_disp:.3f}m")
  has_contact = bool(initial_pairs or cur_pairs)
  if mode == "touch":
    if not has_contact:
      reasons.append("no hand-cube contact (want fingers touching)")
  else:
    if has_contact:
      reasons.append(f"hand-cube contact {initial_pairs or cur_pairs}")
    if not fmin <= fingertip_dist <= fmax:
      reasons.append(
          f"fingertip-center {fingertip_dist:.3f}m outside "
          f"[{fmin:.3f},{fmax:.3f}]")
  return (not reasons), reasons


def _evaluate_grasp(obj, palm_obj):
  reasons = []
  if abs(float(obj[2]) - TABLE_OBJECT_Z) > CUBE_Z_TOL:
    reasons.append(
        f"cube off table z={float(obj[2]):.3f} (want {TABLE_OBJECT_Z:.3f})")
  if palm_obj > GRASP_MAX_PALM_OBJ:
    reasons.append(
        f"palm too far from cube |palm-obj|={palm_obj:.3f}m "
        f"(limit {GRASP_MAX_PALM_OBJ:.2f}m)")
  return (not reasons), reasons


def _caption(rgb, view, obj_xyz, palm_xyz, seed, fingertip_dist, contact_pairs,
             init_style, spawn_xy, extra):
  image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
  bar_h = 82
  canvas = Image.new("RGB", (image.width, image.height + bar_h), (18, 18, 18))
  canvas.paste(image, (0, bar_h))
  draw = ImageDraw.Draw(canvas)
  draw.text(
      (14, 10),
      f"Allegro center | 1.5x1.5 large table | {init_style} init | {view}",
      fill=(245, 245, 245))
  draw.text(
      (14, 34),
      f"cube={tuple(round(float(v), 3) for v in obj_xyz)} | "
      f"palm={tuple(round(float(v), 3) for v in palm_xyz)} | "
      f"spawn={spawn_xy} | seed={seed}",
      fill=(205, 205, 205))
  draw.text(
      (14, 56),
      f"{extra} | min fingertip-center={fingertip_dist:.3f}m | "
      f"hand-cube contacts={contact_pairs}",
      fill=(205, 205, 205))
  return canvas


def _build_env(args):
  if args.init_style == "grasp":
    return AllegroKukaThrowVecEnv(
        num_envs=1,
        seed=args.seed,
        episode_length=300,
        table_spawn=True,
        table_spawn_object_xy=GRASP_SPAWN_XY,
        table_spawn_behind=False,
        table_push=True,
        table_push_xyz=CENTER_GOAL,
        large_table=True,
        reset_z_above=0.7,
        randomize_init=True,
        randomize_object_shape=False,
        enable_cameras=True,
        camera_width=WIDTH,
        camera_height=HEIGHT,
        camera_eye=VIEWS["robot"][0],
        camera_tgt=VIEWS["robot"][1],
        camera_hfov=HFOV,
        pipeline=args.pipeline,
    )
  return AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=args.seed,
      episode_length=300,
      table_spawn=True,
      table_spawn_object_xy=BEHIND_SPAWN_XY,
      table_spawn_behind=True,
      table_spawn_behind_dy=args.palm_dy,
      table_spawn_behind_above=args.palm_above,
      table_spawn_correlated_xy=CORRELATED_XY_RANDOMIZATION,
      table_spawn_finger_curl_scale=args.curl,
      table_spawn_behind_tilt=args.tilt,
      table_push=True,
      table_push_xyz=CENTER_GOAL,
      large_table=True,
      reset_z_above=0.7,
      randomize_init=False,
      randomize_object_shape=False,
      enable_cameras=True,
      camera_width=WIDTH,
      camera_height=HEIGHT,
      camera_eye=VIEWS["robot"][0],
      camera_tgt=VIEWS["robot"][1],
      camera_hfov=HFOV,
      pipeline=args.pipeline,
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--out-dir",
      default=os.path.join(REPO, "figs", "allegro_kuka_throw",
                           "large_table_1p5x1p5"))
  parser.add_argument("--pipeline", default="gpu", choices=("gpu", "cpu"))
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--init-style", choices=("behind", "grasp"), default="behind")
  parser.add_argument("--palm-dy", type=float, default=0.10)
  parser.add_argument("--palm-above", type=float, default=0.05)
  parser.add_argument("--curl", type=float, default=0.0)
  parser.add_argument("--tag", default="hover")
  parser.add_argument("--fingertip-min", type=float, default=FINGERTIP_MIN)
  parser.add_argument("--fingertip-max", type=float, default=FINGERTIP_MAX)
  parser.add_argument(
      "--max-xy-disp", type=float, default=MAX_XY_DISPLACEMENT)
  parser.add_argument(
      "--mode", choices=("hover", "touch"), default="hover",
      help="behind-style only: hover vs touch acceptance")
  parser.add_argument("--tilt", type=float, default=None)
  parser.add_argument(
      "--enforce", action="store_true",
      help="hard-fail on criteria and render the views")
  args = parser.parse_args()

  if LARGE_TABLE_SIZE_XY != (1.5, 1.5):
    raise AssertionError(f"unexpected large-table size: {LARGE_TABLE_SIZE_XY}")
  if LARGE_TABLE_BOUNDS_XY != ((-0.75, 0.75), (-1.30, 0.20)):
    raise AssertionError(
        f"unexpected large-table bounds: {LARGE_TABLE_BOUNDS_XY}")

  env = _build_env(args)
  env.reset()
  obj = env._object_xyz()[0].detach().cpu().numpy()
  palm = env._palm_xyz()[0].detach().cpu().numpy()
  palm_obj = float(np.linalg.norm(palm - obj))
  fingertip_dist = float(
      env._env.curr_fingertip_distances[0].min().detach().cpu().item())
  contact_pairs = _hand_cube_contacts(env)
  initial_contact_pairs = getattr(
      env._env, "_table_spawn_initial_hand_contacts", [])
  init_xy = env._env.object_init_state[0, :2].detach().cpu().numpy()
  xy_displacement = float(np.linalg.norm(obj[:2] - init_xy))

  if args.init_style == "grasp":
    passed, reasons = _evaluate_grasp(obj, palm_obj)
    spawn_xy = GRASP_SPAWN_XY
    extra = "IK palm-on-cube grasp (±2cm xy, small joint noise)"
  else:
    passed, reasons = _evaluate_behind(
        obj, fingertip_dist, initial_contact_pairs, contact_pairs,
        xy_displacement, args.fingertip_min, args.fingertip_max,
        args.max_xy_disp, args.mode)
    spawn_xy = BEHIND_SPAWN_XY
    extra = (
        f"palm offset=(+y {args.palm_dy}, +z {args.palm_above}) | "
        f"curl={args.curl:.2f} | tilt={args.tilt}")

  print(
      f"[SWEEP] init={args.init_style} mode={args.mode} "
      f"seed={args.seed} cube={obj.tolist()} palm={palm.tolist()} "
      f"|palm-obj|={palm_obj:.4f} fingertip_center={fingertip_dist:.4f} "
      f"xy_disp={xy_displacement:.4f} contacts={contact_pairs} "
      f"=> {'PASS' if passed else 'FAIL: ' + '; '.join(reasons)}",
      flush=True)

  if not args.enforce:
    return 0
  if not passed:
    raise RuntimeError("; ".join(reasons))

  os.makedirs(args.out_dir, exist_ok=True)
  for view, (eye, target) in VIEWS.items():
    _set_camera(env, eye, target)
    rgb = env.render_rgb()
    if rgb is None:
      raise RuntimeError(f"camera returned no frame for {view}")
    path = os.path.join(
        args.out_dir,
        f"center_large_table_{args.tag}_s{args.seed}_{view}.png")
    _caption(
        rgb, view, obj, palm, args.seed, fingertip_dist, contact_pairs,
        args.init_style, spawn_xy, extra).save(path)
    print(f"wrote {path}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
