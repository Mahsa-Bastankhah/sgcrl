"""Preflight validation for fully-scaled Allegro [q, qd] HER goals."""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--trials", type=int, default=4)
  parser.add_argument("--episode-length", type=int, default=150)
  args = parser.parse_args()

  from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
  import torch

  env = AllegroKukaThrowVecEnv(
      num_envs=args.num_envs,
      seed=0,
      episode_length=args.episode_length,
      pipeline="gpu",
      headless=True,
      randomize_init=True,
      randomize_object_shape=False,
      control_sanity_mode="index_thumb_straight",
      control_sanity_trim_sa=True,
      control_sanity_trim_init_mode="full_range",
      control_sanity_goal_include_qd=True,
      coordinate_mode="fully_scaled",
  )
  obs = env.reset()
  assert (env.obs_dim, env.action_dim, env.goal_dim) == (16, 8, 16)
  assert tuple(obs.shape) == (args.num_envs, 32)

  oracle = env.normalized_goal_action()
  goal = env._goal_batch
  assert tuple(oracle.shape) == (args.num_envs, 8)
  assert bool(torch.all((oracle >= -1.0) & (oracle <= 1.0)).item())
  torch.testing.assert_close(goal[:, :8], oracle, atol=1e-6, rtol=0)
  torch.testing.assert_close(goal[:, 8:], torch.zeros_like(goal[:, 8:]))

  ctrl = env._trim_full_action_idx
  assert ctrl is not None
  raw_q = env._env.arm_hand_dof_pos[:, ctrl]
  lo = env._env.arm_hand_dof_lower_limits[ctrl]
  hi = env._env.arm_hand_dof_upper_limits[ctrl]
  oracle_raw = lo + 0.5 * (oracle + 1.0) * (hi - lo)
  torch.testing.assert_close(
      oracle_raw, env._control_sanity_goal_q_batch, atol=1e-5, rtol=1e-5)

  # HER achieved goal is the entire packed state, including scaled/clipped qd.
  env.step(torch.zeros_like(oracle))
  packed = env._pack_obs()
  actual_qd_raw = env._env.arm_hand_dof_vel[:, ctrl]
  velocity_limits = env._joint_velocity_limits[ctrl]
  expected_qd_scaled = torch.clamp(
      actual_qd_raw / velocity_limits.unsqueeze(0), -1.0, 1.0)
  torch.testing.assert_close(
      packed[:, 8:16], expected_qd_scaled, atol=1e-6, rtol=0)
  if not bool(torch.any(torch.abs(expected_qd_scaled) > 1e-6).item()):
    raise AssertionError("expected nonzero achieved-goal velocity after step")
  future_state = packed[:, :16].clone()
  her_goal = future_state[:, torch.arange(16, device=env.device)]
  torch.testing.assert_close(her_goal, future_state, atol=0, rtol=0)

  # Position-only success must be invariant to deliberately large velocity.
  saved_q = env._env.arm_hand_dof_pos[:, ctrl].clone()
  saved_qd = env._env.arm_hand_dof_vel[:, ctrl].clone()
  env._env.arm_hand_dof_pos[:, ctrl] = env._control_sanity_goal_q_batch
  env._env.arm_hand_dof_vel[:, ctrl] = 7.0
  success_fast = env.success()
  env._env.arm_hand_dof_vel[:, ctrl] = 0.0
  success_still = env.success()
  if not (bool(torch.all(success_fast == 1).item())
          and torch.equal(success_fast, success_still)):
    raise AssertionError("success changed with velocity at identical goal q")
  env._env.arm_hand_dof_pos[:, ctrl] = saved_q
  env._env.arm_hand_dof_vel[:, ctrl] = saved_qd

  solved = 0
  total = 0
  for _ in range(args.trials):
    env.reset()
    reached = torch.zeros(
        args.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(args.episode_length):
      env.step(env.normalized_goal_action())
      reached |= env.success().bool()
    solved += int(reached.sum().item())
    total += int(reached.numel())
  rate = solved / total
  if rate < 0.99:
    raise AssertionError(f"oracle solve rate {rate:.6f} < 0.99")

  qd_abs_mean = float(torch.mean(torch.abs(expected_qd_scaled)).item())
  qd_abs_max = float(torch.max(torch.abs(expected_qd_scaled)).item())
  print(
      "VALIDATION_OK "
      f"dims=obs{env.obs_dim}/action{env.action_dim}/goal{env.goal_dim} "
      "packed_obs=32 task_goal=[q_norm,zero_qd_scaled] "
      "her_goal=future_full_state=[q_norm,qd_scaled] "
      "oracle_action=8D_normalized_roundtrip_ok "
      "oracle_physics_matches_q_only=identical_8D_action_path "
      "success_position_only=ok "
      f"observed_qd_scaled_abs_mean={qd_abs_mean:.6g} "
      f"observed_qd_scaled_abs_max={qd_abs_max:.6g} "
      f"oracle_solve_rate={rate:.6f} ({solved}/{total})",
      flush=True,
  )


if __name__ == "__main__":
  main()
