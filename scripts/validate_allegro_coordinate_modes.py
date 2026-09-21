#!/usr/bin/env python3
"""Validate matched physical dynamics across Allegro coordinate packings."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
MODES = ("mixed", "physical", "fully_scaled")


def _seed_everything(seed: int) -> None:
  import torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _worker(args) -> None:
  from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv

  # Importing the env loads Isaac Gym before Torch, as Preview 4 requires.
  # Seed Torch/CUDA immediately afterward and before construction/reset.
  _seed_everything(args.seed)
  env = AllegroKukaThrowVecEnv(
      num_envs=args.num_envs,
      seed=args.seed,
      episode_length=args.steps,
      pipeline=args.pipeline,
      control_sanity_mode="index_thumb_straight",
      control_sanity_trim_sa=True,
      control_sanity_trim_init_mode="full_range",
      coordinate_mode=args.worker_mode,
      randomize_init=True,
      randomize_object_shape=False,
      large_table=True,
      enable_cameras=False,
  )
  _seed_everything(args.seed)
  obs = env.reset()
  if (env.obs_dim, env.action_dim, env.goal_dim) != (16, 8, 8):
    raise RuntimeError(
        f"bad dimensions {(env.obs_dim, env.action_dim, env.goal_dim)}")
  physical_q = env._env.arm_hand_dof_pos[
      :, env._trim_full_action_idx].detach().cpu().numpy().copy()
  action = env.normalized_goal_action().detach().cpu().numpy().copy()
  trajectories = [physical_q]
  ever = env.success_levels()["hard"].bool()
  for _ in range(args.steps):
    env.step(env.normalized_goal_action())
    trajectories.append(
        env._env.arm_hand_dof_pos[
            :, env._trim_full_action_idx].detach().cpu().numpy().copy())
    ever |= env.success_levels()["hard"].bool()
  result = {
      "mode": args.worker_mode,
      "dims": [env.obs_dim, env.action_dim, env.goal_dim],
      "oracle_ever_success": float(ever.float().mean().item()),
      "packed_q_min": float(obs[:, :8].min().item()),
      "packed_q_max": float(obs[:, :8].max().item()),
      "packed_qd_min": float(obs[:, 8:16].min().item()),
      "packed_qd_max": float(obs[:, 8:16].max().item()),
      "velocity_limits_rad_s": [
          float(v) for v in env._joint_velocity_limits[
              env._trim_full_action_idx].tolist()],
  }
  np.savez_compressed(
      args.worker_output,
      physical_q0=physical_q,
      normalized_action=action,
      physical_trajectory=np.asarray(trajectories),
      result_json=np.asarray(json.dumps(result)),
  )
  print(json.dumps(result, indent=2), flush=True)
  if result["oracle_ever_success"] < 0.99:
    raise RuntimeError("oracle solved fewer than 99% of environments")


def _parent(args) -> None:
  os.makedirs(args.output_dir, exist_ok=True)
  records = {}
  arrays = {}
  for mode in MODES:
    path = os.path.join(args.output_dir, f"{mode}.npz")
    cmd = [
        sys.executable, "-u", os.path.abspath(__file__),
        f"--worker-mode={mode}", f"--worker-output={path}",
        f"--num-envs={args.num_envs}", f"--steps={args.steps}",
        f"--seed={args.seed}", f"--pipeline={args.pipeline}",
    ]
    subprocess.run(cmd, cwd=REPO, check=True)
    data = np.load(path)
    arrays[mode] = data
    records[mode] = json.loads(str(data["result_json"].item()))

  reference = arrays["mixed"]
  comparisons = {}
  for mode in ("physical", "fully_scaled"):
    current = arrays[mode]
    q0_max_abs = float(np.max(np.abs(
        reference["physical_q0"] - current["physical_q0"])))
    action_max_abs = float(np.max(np.abs(
        reference["normalized_action"] - current["normalized_action"])))
    trajectory_max_abs = float(np.max(np.abs(
        reference["physical_trajectory"] - current["physical_trajectory"])))
    comparisons[mode] = {
        "initial_physical_q_max_abs": q0_max_abs,
        "normalized_action_max_abs": action_max_abs,
        "physical_trajectory_max_abs": trajectory_max_abs,
    }
    if q0_max_abs > args.atol or action_max_abs > args.atol:
      raise RuntimeError(f"{mode}: reset/action mismatch: {comparisons[mode]}")
    if trajectory_max_abs > args.trajectory_atol:
      raise RuntimeError(
          f"{mode}: physical trajectory mismatch: {comparisons[mode]}")

  summary = {
      "seed": args.seed,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "atol": args.atol,
      "trajectory_atol": args.trajectory_atol,
      "modes": records,
      "comparisons_to_mixed": comparisons,
  }
  summary_path = os.path.join(args.output_dir, "summary.json")
  with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
  print(f"[coordinate-validation] PASS wrote {summary_path}", flush=True)
  print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", default=os.path.join(
      REPO, "figs", "allegro_kuka_throw", "coordinate_mode_validation"))
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=150)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--pipeline", choices=("gpu", "cpu"), default="gpu")
  parser.add_argument("--atol", type=float, default=1e-6)
  parser.add_argument("--trajectory-atol", type=float, default=1e-5)
  parser.add_argument("--worker-mode", choices=MODES)
  parser.add_argument("--worker-output")
  args = parser.parse_args()
  if args.worker_mode:
    if not args.worker_output:
      parser.error("--worker-output is required with --worker-mode")
    _worker(args)
  else:
    _parent(args)


if __name__ == "__main__":
  main()
