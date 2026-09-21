#!/usr/bin/env python3
"""Collect complete BuilderBench PD episodes under iid uniform actions."""
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


def _write_json(path: str, value: dict) -> None:
  tmp = path + ".tmp"
  with open(tmp, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
  os.replace(tmp, path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", default="builderbench_creative_3_task1")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--num_envs", type=int, default=1024)
  parser.add_argument("--logical_episode_length", type=int, default=50)
  parser.add_argument("--pd_duration", type=int, default=5)
  parser.add_argument("--target_transitions", type=int, default=8_000_000)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument(
      "--builderbench_permute_start_boxes", default="false",
      choices=("false",))
  parser.add_argument(
      "--builderbench_fixed_start_x", type=float, default=0.1)
  args = parser.parse_args()

  if os.environ.get("BUILDERBENCH_MJX_IMPL") != "warp":
    raise ValueError("BUILDERBENCH_MJX_IMPL must be exactly 'warp'")
  if args.env != "builderbench_creative_3_task1":
    raise ValueError(f"this diagnostic expects c3t1, got {args.env!r}")
  if args.logical_episode_length != 50 or args.pd_duration != 5:
    raise ValueError("c3t1 diagnostic requires T=50 and PD duration=5")
  if args.builderbench_fixed_start_x != 0.1:
    raise ValueError("NF BuilderBench collection requires fixed_start_x=0.1")

  import jax
  import jax.numpy as jnp
  from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv
  from envs.builderbench_utils import default_fixed_target_goal

  env = JaxBuilderBenchVecEnv(
      env_name=args.env, num_envs=int(args.num_envs), seed=int(args.seed),
      use_pd=True, pd_duration=int(args.pd_duration),
      pd_filter_policy_obs=True, permute_start_boxes=False,
      mj_episode_length=args.logical_episode_length * args.pd_duration,
      fixed_start_x=float(args.builderbench_fixed_start_x),
      success_terminate_steps=0)
  state_dim, action_dim, goal_dim = (
      env.observation_shape[0] - 9, env.action_shape[0], 9)
  if (state_dim, action_dim, goal_dim) != (10, 5, 9):
    raise ValueError(
        f"expected state/action/goal 10/5/9, got "
        f"{state_dim}/{action_dim}/{goal_dim}")
  horizon = int(args.logical_episode_length)
  num_episodes = int(args.target_transitions) // horizon
  actual_transitions = num_episodes * horizon
  os.makedirs(args.output_dir, exist_ok=True)
  obs_out = np.lib.format.open_memmap(
      os.path.join(args.output_dir, "observations.npy"), mode="w+",
      dtype=np.float32, shape=(num_episodes, horizon + 1, state_dim))
  action_out = np.lib.format.open_memmap(
      os.path.join(args.output_dir, "actions.npy"), mode="w+",
      dtype=np.float32, shape=(num_episodes, horizon, action_dim))

  collect = env.compile_fixed_action_episode_collector(horizon)
  action_rng = np.random.default_rng(int(args.seed))
  written = 0
  started = time.time()
  initial_samples: list[np.ndarray] = []
  target_goal_samples: list[np.ndarray] = []
  discarded_boundary_rollouts = 0
  while written < num_episodes:
    env_state = env.reset_state()
    actions = action_rng.uniform(
        -1.0, 1.0, size=(horizon, env.num_envs, action_dim)).astype(
            np.float32)
    _, episode_obs, dones = collect(env_state, jnp.asarray(actions))
    episode_obs_np, dones_np = jax.device_get((episode_obs, dones))
    valid = np.logical_and(
        ~np.any(dones_np[:-1], axis=0), dones_np[-1].astype(bool))
    valid_indices = np.flatnonzero(valid)
    discarded_boundary_rollouts += int(env.num_envs - len(valid_indices))
    if len(valid_indices) == 0:
      raise RuntimeError("all vectorized rollouts had invalid boundaries")
    take = min(len(valid_indices), num_episodes - written)
    chosen = valid_indices[:take]
    state_episode = np.asarray(
        episode_obs_np[chosen, :, :state_dim], dtype=np.float32)
    if not np.isfinite(state_episode).all() or not np.isfinite(actions).all():
      raise RuntimeError("non-finite values encountered during collection")
    obs_out[written:written + take] = state_episode
    action_out[written:written + take] = np.swapaxes(
        actions[:, chosen], 0, 1)
    initial_samples.append(state_episode[:, 0, :9].copy())
    target_goal_samples.append(np.asarray(
        env_state.info["target_goal"][chosen], dtype=np.float32))
    written += take
    if written % (10 * env.num_envs) == 0 or written == num_episodes:
      obs_out.flush()
      action_out.flush()
      print(
          f"[collect] episodes={written}/{num_episodes} "
          f"transitions={written * horizon:,} "
          f"discarded_boundary_rollouts={discarded_boundary_rollouts} "
          f"elapsed={time.time() - started:.1f}s", flush=True)

  initial = np.concatenate(initial_samples, axis=0)
  sampled_goals = np.concatenate(target_goal_samples, axis=0)
  cube_xyz = initial.reshape(num_episodes, 3, 3)
  reset_stats = {
      "cube_x_mean": cube_xyz[:, :, 0].mean(axis=0).tolist(),
      "cube_x_std": cube_xyz[:, :, 0].std(axis=0).tolist(),
      "cube_y_std": cube_xyz[:, :, 1].std(axis=0).tolist(),
      "distinct_initial_xyz_rows": int(np.unique(initial, axis=0).shape[0]),
      "target_goal_std": sampled_goals.std(axis=0).tolist(),
      "distinct_target_goal_rows": int(
          np.unique(sampled_goals, axis=0).shape[0]),
  }
  if reset_stats["distinct_target_goal_rows"] <= 1:
    raise RuntimeError(
        "reset randomness check failed: sampled target goals all match")
  metadata = {
      "format_version": 1,
      "format": "episode-major numpy memmap",
      "observations_file": "observations.npy",
      "actions_file": "actions.npy",
      "observation_shape": list(obs_out.shape),
      "action_shape": list(action_out.shape),
      "dtype": "float32",
      "state_layout": ["cube_xyz[9]", "select_action[1]"],
      "action_layout": "PD action[5]",
      "achieved_goal_state_indices": list(range(9)),
      "task_goal": default_fixed_target_goal(3, 0).tolist(),
      "collection_policy": "iid Uniform[-1,1]^5 per macro transition",
      "action_seed": int(args.seed),
      "environment_seed": int(args.seed),
      "num_envs": int(args.num_envs),
      "num_episodes": num_episodes,
      "logical_episode_length": horizon,
      "pd_duration": int(args.pd_duration),
      "requested_transitions": int(args.target_transitions),
      "actual_transitions": actual_transitions,
      "environment": {
          "name": args.env,
          "builderbench_id": "creative-3-task1",
          "mjx_impl": "warp",
          "use_pd": True,
          "pd_filter_policy_obs": True,
          "permute_start_boxes": False,
          "fixed_start_x": 0.1,
          "remaining_reset_randomness": "BuilderBench defaults",
      },
      "reset_randomness_check": reset_stats,
      "episode_boundary_check": (
          "stored rollouts have no done in macro steps 1..49 and done after "
          "action 50; invalid per-env rollouts are discarded; "
          "stored observation 51 is pre-autoreset terminal observation"),
      "discarded_boundary_rollouts": discarded_boundary_rollouts,
      "elapsed_seconds": time.time() - started,
  }
  _write_json(os.path.join(args.output_dir, "metadata.json"), metadata)
  print(
      f"[collect] complete: observations={obs_out.shape} "
      f"actions={action_out.shape} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
  main()
