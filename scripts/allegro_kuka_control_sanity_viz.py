#!/usr/bin/env python3
"""Render object-free Allegro finger and palm control sanity tasks."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    CONTROL_SANITY_FINGER_GOAL_Q,
    CONTROL_SANITY_HAND16_GOAL_Q,
    CONTROL_SANITY_HAND16_FIGURE_GOAL_Q,
    CONTROL_SANITY_HAND16_OK_GOAL_Q,
    CONTROL_SANITY_HAND16_PEACE_GOAL_Q,
    CONTROL_SANITY_HAND16_POINT_GOAL_Q,
    CONTROL_SANITY_HAND16_GUN_GOAL_Q,
    CONTROL_SANITY_ARM23_WAVE_GOAL_Q,
    CONTROL_SANITY_INDEX_GOAL_Q,
    CONTROL_SANITY_SIX_GOAL_Q,
    CONTROL_SANITY_TWO_FINGER_GOAL_Q,
    CONTROL_SANITY_THREE2_GOAL_Q,
    CONTROL_SANITY_THREE2_HAND_INDICES,
    CONTROL_SANITY_FOUR2_GOAL_Q,
    CONTROL_SANITY_FOUR2H_GOAL_Q,
    CONTROL_SANITY_FOUR2M_GOAL_Q,
    CONTROL_SANITY_FOUR2MH_GOAL_Q,
    CONTROL_SANITY_FOUR2MM_GOAL_Q,
    CONTROL_SANITY_FOUR2MMH_GOAL_Q,
    CONTROL_SANITY_FOUR2MMX_GOAL_Q,
    CONTROL_SANITY_FOUR2W_GOAL_Q,
    CONTROL_SANITY_FOUR2_HAND_INDICES,
    CONTROL_SANITY_INDEX_THUMB_STRAIGHT_GOAL_Q,
    CONTROL_SANITY_INDEX_THUMB_HAND_INDICES,
    CONTROL_SANITY_PALM_GOAL_XYZ,
    AllegroKukaThrowVecEnv,
)

WIDTH, HEIGHT = 1000, 760


def _set_camera(env, eye, target) -> None:
  from isaacgym import gymapi

  task = env._env
  task.gym.set_camera_location(
      env._cam_handle, task.envs[0],
      gymapi.Vec3(*eye), gymapi.Vec3(*target))


def _save(env, path: str, caption: str) -> None:
  rgb = env.render_rgb()
  if rgb is None:
    raise RuntimeError("camera returned no frame")
  image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
  canvas = Image.new("RGB", (image.width, image.height + 58), (18, 18, 18))
  canvas.paste(image, (0, 58))
  draw = ImageDraw.Draw(canvas)
  draw.text((14, 10), caption, fill=(245, 245, 245))
  draw.text(
      (14, 34),
      "No task object in workspace; required dummy cube is parked off-camera.",
      fill=(205, 205, 205))
  os.makedirs(os.path.dirname(path), exist_ok=True)
  canvas.save(path)
  print(f"wrote {path}", flush=True)


def _set_finger_goal(env, goal, hand_indices=None, full_q=False) -> None:
  import torch
  from isaacgym import gymtorch

  task = env._env
  q = torch.tensor(goal, dtype=torch.float, device=task.device)
  if full_q:
    n = int(q.numel())
    task.arm_hand_dof_pos[:, :n] = q
    task.cur_targets[:, :n] = q
    task.prev_targets[:, :n] = q
  elif hand_indices is None:
    task.arm_hand_dof_pos[:, 7:7 + len(goal)] = q
    task.cur_targets[:, 7:7 + len(goal)] = q
    task.prev_targets[:, 7:7 + len(goal)] = q
  else:
    idx = torch.tensor(
        [7 + int(i) for i in hand_indices],
        dtype=torch.long, device=task.device)
    task.arm_hand_dof_pos[:, idx] = q
    task.cur_targets[:, idx] = q
    task.prev_targets[:, idx] = q
  task.arm_hand_dof_vel[:, :] = 0.0
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))
  n_settle = 80 if len(goal) >= 16 else 12
  for _ in range(n_settle):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
  task.compute_observations()


def _set_palm_goal(env) -> float:
  import torch
  from isaacgym import gymtorch

  task = env._env
  jac = gymtorch.wrap_tensor(
      task.gym.acquire_jacobian_tensor(task.sim, "allegro"))
  goal = torch.tensor(
      [CONTROL_SANITY_PALM_GOAL_XYZ],
      dtype=torch.float, device=task.device)
  err = task._ik_palm_stage(
      jac, goal, n_iters=240, step=0.12, lam=0.10,
      name="palm-sanity-goal", stop_at=0.015)
  task.arm_hand_dof_vel[:, :] = 0.0
  task.gym.set_dof_position_target_tensor(
      task.sim, gymtorch.unwrap_tensor(task.cur_targets))
  task.gym.set_dof_state_tensor(
      task.sim, gymtorch.unwrap_tensor(task.dof_state))
  for _ in range(12):
    task.gym.simulate(task.sim)
    task.gym.fetch_results(task.sim, True)
  task.compute_observations()
  return float(err)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--mode", required=True,
      choices=("finger", "hand16", "hand16fig", "hand16ok", "hand16peace",
               "hand16point", "hand16gun", "arm23wave", "index", "six",
               "two_finger",
               "three2", "four2",
               "four2h", "four2m", "four2mh", "four2mm", "four2mmh", "four2mmx",
               "four2w", "index_thumb_straight", "palm"))
  parser.add_argument(
      "--out-dir",
      default=os.path.join(
          REPO, "figs", "allegro_kuka_throw", "control_sanity"))
  parser.add_argument("--pipeline", default="gpu", choices=("gpu", "cpu"))
  parser.add_argument("--trim-sa", action="store_true")
  parser.add_argument(
      "--trim-init-range-frac", type=float, default=0.10,
      help="controlled-joint reset radius / full physical joint range "
           "(curled mode only)")
  parser.add_argument(
      "--trim-init-mode", default="curled",
      choices=("curled", "full_range"),
      help="curled=center±frac; full_range=Uniform[lo,hi] per joint")
  parser.add_argument("--randomize-init", action="store_true")
  parser.add_argument("--oracle-action", action="store_true")
  parser.add_argument(
      "--coordinate-mode", default="mixed",
      choices=("mixed", "physical", "fully_scaled"))
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--steps", type=int, default=150)
  args = parser.parse_args()
  if (args.oracle_action and not args.trim_sa
      and args.mode != "index_thumb_straight"):
    parser.error("--oracle-action requires --trim-sa")

  close = args.mode in (
      "hand16fig", "hand16ok", "hand16peace", "hand16point", "hand16gun",
      "hand16",
      "finger")
  env = AllegroKukaThrowVecEnv(
      num_envs=args.num_envs,
      seed=0,
      episode_length=150,
      pipeline=args.pipeline,
      control_sanity_mode=args.mode,
      control_sanity_trim_sa=args.trim_sa,
      control_sanity_trim_init_range_frac=args.trim_init_range_frac,
      control_sanity_trim_init_mode=args.trim_init_mode,
      coordinate_mode=args.coordinate_mode,
      control_sanity_palm_xyz=CONTROL_SANITY_PALM_GOAL_XYZ,
      randomize_init=args.randomize_init,
      randomize_object_shape=False,
      large_table=True,
      enable_cameras=not args.oracle_action,
      camera_width=WIDTH,
      camera_height=HEIGHT,
      camera_eye=(0.42, -0.36, 0.90) if close else None,
      camera_tgt=(0.00, 0.00, 0.68) if close else None,
      camera_hfov=42.0 if close else None,
  )
  obs = env.reset()
  expected_total = (
      env.obs_dim + env.goal_dim
      if env.control_sanity_trim_sa else {
      "finger": 46 + 16,
      "hand16": 46 + 16,
      "hand16fig": 46 + 16,
      "hand16ok": 46 + 16,
      "hand16peace": 46 + 16,
      "hand16point": 46 + 16,
      "hand16gun": 46 + 16,
      "arm23wave": 46 + 23,
      "index": 46 + 4,
      "six": 46 + 6,
      "two_finger": 46 + 8,
      "three2": 46 + 6,
      "four2": 46 + 8,
      "four2h": 46 + 8,
      "four2m": 46 + 8,
      "four2mh": 46 + 8,
      "four2mm": 46 + 8,
      "four2mmh": 46 + 8,
      "four2mmx": 46 + 8,
      "four2w": 46 + 8,
      "index_thumb_straight": 46 + 8,
      "palm": 49 + 3,
      }[args.mode])
  if tuple(obs.shape) != (args.num_envs, expected_total):
    raise RuntimeError(
        f"bad packed observation shape {tuple(obs.shape)}, "
        f"expected ({args.num_envs}, {expected_total})")
  level_modes = (
      "index", "six", "two_finger", "three2", "four2", "four2h", "four2m",
      "four2mh", "four2mm", "four2mmh", "four2mmx", "four2w",
      "index_thumb_straight")
  if args.mode in level_modes:
    levels = env.success_levels()
    hand_q = env._env.arm_hand_dof_pos[0, 7:23]
    if env.control_sanity_hand_indices is not None:
      q0 = hand_q[list(env.control_sanity_hand_indices)]
    else:
      q0 = hand_q[: env.goal_dim]
    if env._control_sanity_goal_q_batch is None:
      raise RuntimeError("joint-control task has no physical goal batch")
    g0 = env._control_sanity_goal_q_batch[0]
    initial_max_error = float((q0 - g0).abs().max().item())
    print(
        f"[control_sanity_viz] {args.mode} initial_max_error="
        f"{initial_max_error:.5f} "
        f"hard={int(levels['hard'][0].item())} "
        f"easy={int(levels['easy'][0].item())} "
        f"very_easy={int(levels['very_easy'][0].item())}",
        flush=True)
    if args.mode == "index_thumb_straight" and levels["hard"][0].item() >= 0.5:
      raise RuntimeError(f"{args.mode} reset is already successful")
    if args.mode != "index_thumb_straight" and initial_max_error <= 0.30:
      raise RuntimeError(
          f"{args.mode} goal is already inside the very-easy threshold at reset")
  if args.mode == "arm23wave":
    q0 = env._env.arm_hand_dof_pos[0, :23]
    if env._control_sanity_goal_q_batch is None:
      raise RuntimeError("arm23wave has no physical goal batch")
    g0 = env._control_sanity_goal_q_batch[0]
    initial_max_error = float((q0 - g0).abs().max().item())
    print(
        f"[control_sanity_viz] arm23wave initial_max_error="
        f"{initial_max_error:.5f} "
        f"hard={int(env.success()[0].item())}",
        flush=True)
    if initial_max_error <= 0.30:
      raise RuntimeError(
          "arm23wave goal is already inside the very-easy threshold at reset")

  if args.oracle_action:
    import torch

    if env.action_dim != env.goal_dim:
      raise RuntimeError(
          f"oracle requires action_dim == goal_dim, got "
          f"{env.action_dim} != {env.goal_dim}")
    oracle_action = env.normalized_goal_action().detach().clone()
    ever_hard = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device)
    best_hard = 0.0
    checkpoints = {0, 1, 2, 4, 9, 19, 49, 99, args.steps - 1}
    for step in range(args.steps):
      env.step(oracle_action)
      levels = env.success_levels()
      ever_hard |= levels["hard"].bool()
      best_hard = max(best_hard, float(levels["hard"].mean().item()))
      if step in checkpoints:
        print(
            "[control_sanity_oracle] "
            f"step={step + 1} hard={levels['hard'].mean().item():.6f} "
            f"easy={levels['easy'].mean().item():.6f} "
            f"very_easy={levels['very_easy'].mean().item():.6f} "
            f"jfrac010={levels['joint_frac_010'].mean().item():.6f} "
            f"jfrac020={levels['joint_frac_020'].mean().item():.6f} "
            f"mae={levels['mean_abs_joint_err'].mean().item():.6f}",
            flush=True)
    final = env.success_levels()
    print(
        "[control_sanity_oracle] RESULT "
        f"envs={env.num_envs} steps={args.steps} "
        f"ever_hard={ever_hard.float().mean().item():.6f} "
        f"best_hard={best_hard:.6f} "
        f"last_observation_hard={final['hard'].mean().item():.6f} "
        "(last observation may be post auto-reset)",
        flush=True)
    if ever_hard.float().mean().item() < 0.99 or best_hard < 0.99:
      raise RuntimeError("oracle action failed to solve at least 99% of envs")
    return 0

  start_palm = env._palm_xyz()[0].detach().cpu().numpy()
  camera_target = (
      np.asarray(CONTROL_SANITY_PALM_GOAL_XYZ, dtype=np.float64)
      if args.mode == "palm" else start_palm.astype(np.float64))
  if args.mode in (
      "hand16fig", "hand16ok", "hand16peace", "hand16point", "hand16gun",
      "hand16",
      "finger"):
    # Look at the fingertips, not the back of the palm.
    eye = camera_target + np.array([0.42, -0.40, 0.20])
    side_eye = camera_target + np.array([0.48, 0.08, 0.16])
    camera_target = camera_target + np.array([0.00, -0.10, -0.02])
  elif args.mode == "arm23wave":
    # Fixed wide shot aimed high enough to see an upright arm.
    camera_target = np.array([0.00, 0.05, 0.95], dtype=np.float64)
    eye = np.array([1.15, -1.35, 1.20], dtype=np.float64)
    side_eye = np.array([1.35, 0.35, 1.10], dtype=np.float64)
  else:
    eye = camera_target + np.array([0.62, -0.55, 0.28])
    side_eye = None
  _set_camera(env, eye, camera_target)
  _save(
      env, os.path.join(args.out_dir, f"{args.mode}_start.png"),
      f"{args.mode} sanity START | state={env.obs_dim} goal={env.goal_dim} "
      f"| palm={tuple(round(float(v), 3) for v in start_palm)}")

  if args.mode in (
          "finger", "hand16", "hand16fig", "hand16ok", "hand16peace",
          "hand16point", "hand16gun", "arm23wave", "index", "six",
          "two_finger",
          "three2", "four2",
          "four2h", "four2m", "four2mh", "four2mm", "four2mmh", "four2mmx",
          "four2w", "index_thumb_straight"):
    goal = {
        "finger": CONTROL_SANITY_FINGER_GOAL_Q,
        "hand16": CONTROL_SANITY_HAND16_GOAL_Q,
        "hand16fig": CONTROL_SANITY_HAND16_FIGURE_GOAL_Q,
        "hand16ok": CONTROL_SANITY_HAND16_OK_GOAL_Q,
        "hand16peace": CONTROL_SANITY_HAND16_PEACE_GOAL_Q,
        "hand16point": CONTROL_SANITY_HAND16_POINT_GOAL_Q,
        "hand16gun": CONTROL_SANITY_HAND16_GUN_GOAL_Q,
        "arm23wave": CONTROL_SANITY_ARM23_WAVE_GOAL_Q,
        "index": CONTROL_SANITY_INDEX_GOAL_Q,
        "six": CONTROL_SANITY_SIX_GOAL_Q,
        "two_finger": CONTROL_SANITY_TWO_FINGER_GOAL_Q,
        "three2": CONTROL_SANITY_THREE2_GOAL_Q,
        "four2": CONTROL_SANITY_FOUR2_GOAL_Q,
        "four2h": CONTROL_SANITY_FOUR2H_GOAL_Q,
        "four2m": CONTROL_SANITY_FOUR2M_GOAL_Q,
        "four2mh": CONTROL_SANITY_FOUR2MH_GOAL_Q,
        "four2mm": CONTROL_SANITY_FOUR2MM_GOAL_Q,
        "four2mmh": CONTROL_SANITY_FOUR2MMH_GOAL_Q,
        "four2mmx": CONTROL_SANITY_FOUR2MMX_GOAL_Q,
        "four2w": CONTROL_SANITY_FOUR2W_GOAL_Q,
        "index_thumb_straight": CONTROL_SANITY_INDEX_THUMB_STRAIGHT_GOAL_Q,
    }[args.mode]
    hand_idx = {
        "three2": CONTROL_SANITY_THREE2_HAND_INDICES,
        "four2": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "four2h": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "four2m": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "four2mh": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "four2mm": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "four2mmh": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "four2mmx": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "four2w": CONTROL_SANITY_FOUR2_HAND_INDICES,
        "index_thumb_straight": CONTROL_SANITY_INDEX_THUMB_HAND_INDICES,
    }.get(args.mode)
    _set_finger_goal(
        env, goal, hand_indices=hand_idx, full_q=(args.mode == "arm23wave"))
    if args.mode == "arm23wave":
      q = env._env.arm_hand_dof_pos[0, :23]
    elif hand_idx is None:
      q = env._env.arm_hand_dof_pos[0, 7:7 + len(goal)]
    else:
      q = env._env.arm_hand_dof_pos[0, 7:23][list(hand_idx)]
    if env._control_sanity_goal_q_batch is None:
      raise RuntimeError("joint-control task has no physical goal batch")
    target = env._control_sanity_goal_q_batch[0]
    error = float((q - target).abs().max().item())
    caption = (
        f"{args.mode} sanity GOAL | {len(goal)} q positions only | "
        f"max joint error={error:.3f} rad")
  else:
    ik_err = _set_palm_goal(env)
    palm = env._palm_xyz()[0]
    target = env._goal_batch[0]
    error = float(np.linalg.norm(
        palm.detach().cpu().numpy() - target.detach().cpu().numpy()))
    caption = (
        f"palm sanity GOAL | xyz={CONTROL_SANITY_PALM_GOAL_XYZ} | "
        f"error={error:.3f} m (IK final={ik_err:.3f} m)")

  success = float(env.success()[0].item())
  print(
      f"[control_sanity_viz] mode={args.mode} obs_dim={env.obs_dim} "
      f"goal_dim={env.goal_dim} error={error:.5f} success={success:g}",
      flush=True)
  live_palm = env._palm_xyz()[0].detach().cpu().numpy()
  if args.mode == "arm23wave":
    camera_target = np.array([0.00, 0.05, 0.95], dtype=np.float64)
    eye = np.array([1.15, -1.35, 1.20], dtype=np.float64)
    side_eye = np.array([1.35, 0.35, 1.10], dtype=np.float64)
  elif args.mode != "palm":
    camera_target = live_palm.astype(np.float64)
    if args.mode in (
        "hand16fig", "hand16ok", "hand16peace", "hand16point", "hand16gun",
      "hand16",
        "finger"):
      eye = camera_target + np.array([0.42, -0.40, 0.20])
      side_eye = camera_target + np.array([0.48, 0.08, 0.16])
      camera_target = camera_target + np.array([0.00, -0.10, -0.02])
  _set_camera(env, eye, camera_target)
  _save(
      env, os.path.join(args.out_dir, f"{args.mode}_goal.png"),
      caption + f" | success={success:g}")
  if side_eye is not None:
    _set_camera(env, side_eye, camera_target)
    _save(
        env, os.path.join(args.out_dir, f"{args.mode}_goal_side.png"),
        caption + f" | side | success={success:g}")
  if success != 1.0 and args.mode not in (
      "hand16fig", "hand16ok", "hand16peace", "hand16point", "hand16gun"):
    raise RuntimeError(
        f"rendered {args.mode} target does not satisfy success: error={error}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
