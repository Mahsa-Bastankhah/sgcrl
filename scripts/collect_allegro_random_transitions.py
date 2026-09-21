#!/usr/bin/env python3
"""Collect episode-major random-policy Allegro ITS transitions.

The dataset is stored as uncompressed, directly memory-mappable ``.npy``
arrays.  Logical episodes have 150 actions and 151 observations.  The
effective simulator horizon after the recorded initial observation is T+1.
The bootstrap ``episode_length`` must be T+3 because the wrapper's
initializing reset consumes one step and Isaac Gym times out at
``progress >= max_episode_length - 1``.  This exposes the observation after
logical action 150 before automatic reset; the environment is manually reset
between logical episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
  sys.path.insert(0, REPO)

# This must run before importing JAX or torch.
from envs import isaacgym_physx_bootstrap as _bootstrap  # noqa: E402

_bootstrap.maybe_create_from_argv()


def _atomic_json(path: str, value: dict) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
  os.replace(tmp, path)


def main() -> None:
  parser = argparse.ArgumentParser()
  # Bootstrap-compatible environment arguments.
  parser.add_argument("--env", default="allegro_kuka_throw")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--ppo_num_envs", type=int, default=1024)
  parser.add_argument("--isaacgym_episode_length", type=int, default=153)
  parser.add_argument("--isaacgym_pipeline", default="gpu")
  parser.add_argument("--isaacgym_randomize_init", default="true")
  parser.add_argument("--isaacgym_randomize_object_shape", default="false")
  parser.add_argument(
      "--isaacgym_control_sanity_mode", default="index_thumb_straight")
  parser.add_argument("--isaacgym_control_sanity_trim_sa", default="true")
  parser.add_argument(
      "--isaacgym_control_sanity_trim_init_mode", default="full_range")
  parser.add_argument("--isaacgym_coordinate_mode", default="fully_scaled")
  parser.add_argument("--isaacgym_large_table", default="true")
  # Collector arguments.
  parser.add_argument("--logical_episode_length", type=int, default=150)
  parser.add_argument("--target_transitions", type=int, default=8_000_000)
  parser.add_argument("--output_dir", required=True)
  args = parser.parse_args()

  if args.logical_episode_length + 3 != args.isaacgym_episode_length:
    raise ValueError(
        "bootstrap episode_length must be logical_episode_length + 3 because "
        "the initializing reset consumes one step and Isaac Gym times out at "
        "max_episode_length - 1")
  env = _bootstrap.take_prebuilt_env()
  if env is None:
    raise RuntimeError("Isaac Gym bootstrap did not create an environment")

  import torch

  state_dim, action_dim, goal_dim = (
      int(env.obs_dim), int(env.action_dim), int(env.goal_dim))
  if (state_dim, action_dim, goal_dim) != (16, 8, 8):
    raise ValueError(
        f"expected ITS trim-SA dimensions 16/8/8, got "
        f"{state_dim}/{action_dim}/{goal_dim}")
  if str(env.coordinate_mode) != "fully_scaled":
    raise ValueError(f"expected fully_scaled, got {env.coordinate_mode!r}")

  horizon = int(args.logical_episode_length)
  # Complete episodes only: nearest count not exceeding the requested scale.
  num_episodes = int(args.target_transitions) // horizon
  actual_transitions = num_episodes * horizon
  num_envs = int(env.num_envs)
  os.makedirs(args.output_dir, exist_ok=True)
  obs_path = os.path.join(args.output_dir, "observations.npy")
  action_path = os.path.join(args.output_dir, "actions.npy")
  obs_out = np.lib.format.open_memmap(
      obs_path, mode="w+", dtype=np.float32,
      shape=(num_episodes, horizon + 1, state_dim))
  action_out = np.lib.format.open_memmap(
      action_path, mode="w+", dtype=np.float32,
      shape=(num_episodes, horizon, action_dim))

  action_rng = np.random.default_rng(int(args.seed))
  task_goal = None
  written = 0
  started = time.time()
  while written < num_episodes:
    packed = env.reset()
    packed_np = packed.detach().cpu().numpy().astype(np.float32, copy=False)
    if task_goal is None:
      task_goal = packed_np[0, state_dim:].copy()
    take = min(num_envs, num_episodes - written)
    obs_out[written:written + take, 0] = packed_np[:take, :state_dim]
    for step in range(horizon):
      action_np = action_rng.uniform(
          -1.0, 1.0, size=(num_envs, action_dim)).astype(np.float32)
      next_packed, _, done = env.step(torch.as_tensor(
          action_np, dtype=torch.float32, device=env.device))
      if bool(torch.any(done).item()):
        raise RuntimeError(
            f"automatic reset at logical step {step + 1}; successor state "
            "would be invalid")
      next_np = (
          next_packed.detach().cpu().numpy().astype(np.float32, copy=False))
      action_out[written:written + take, step] = action_np[:take]
      obs_out[written:written + take, step + 1] = next_np[:take, :state_dim]
    written += take
    obs_out.flush()
    action_out.flush()
    elapsed = time.time() - started
    print(
        f"[collect] episodes={written}/{num_episodes} "
        f"transitions={written * horizon:,} elapsed={elapsed:.1f}s",
        flush=True)

  metadata = {
      "format_version": 1,
      "format": "episode-major numpy memmap",
      "observations_file": os.path.basename(obs_path),
      "actions_file": os.path.basename(action_path),
      "observation_shape": list(obs_out.shape),
      "action_shape": list(action_out.shape),
      "dtype": "float32",
      "state_layout": ["q_norm[8]", "qd_scaled[8]"],
      "action_layout": "absolute normalized joint targets[8]",
      "achieved_goal_state_indices": list(range(8)),
      "task_goal": np.asarray(task_goal, dtype=np.float32).tolist(),
      "collection_policy": "iid Uniform[-1,1]^8 per transition",
      "action_seed": int(args.seed),
      "environment_seed": int(args.seed),
      "num_envs": num_envs,
      "num_episodes": num_episodes,
      "logical_episode_length": horizon,
      "simulator_episode_length": int(args.isaacgym_episode_length),
      "requested_transitions": int(args.target_transitions),
      "actual_transitions": actual_transitions,
      "environment": {
          "name": "allegro_kuka_throw",
          "control_sanity_mode": "index_thumb_straight",
          "trim_sa": True,
          "trim_init_mode": "full_range",
          "coordinate_mode": "fully_scaled",
          "randomize_init": True,
          "pipeline": str(args.isaacgym_pipeline),
      },
      "terminal_observation_semantics": (
          "effective sim horizon after initial observation is logical horizon "
          "+ 1; bootstrap episode_length is logical horizon + 3 because reset "
          "initialization consumes one step and timeout occurs at "
          "max_episode_length-1; manual reset after 150 actions"),
      "elapsed_seconds": time.time() - started,
  }
  _atomic_json(os.path.join(args.output_dir, "metadata.json"), metadata)
  print(
      f"[collect] complete: {num_episodes} episodes, "
      f"{actual_transitions:,} transitions -> {args.output_dir}",
      flush=True)


if __name__ == "__main__":
  main()
