"""BuilderBench helpers for DISCOVER: pack obs, write curriculum goals."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from envs.builderbench_jax_vec import _pack_obs
from envs.builderbench_utils import (
    default_fixed_target_goal,
    goal_xyz_state_indices,
    load_task_cube_mask,
    parse_sgcrl_builderbench_env_name,
    pd_policy_state_obs_dim,
)


def env_layout(env_name: str, pd_filter: bool = True):
  """Return (bb_env_id, num_cubes, task_index, state_dim, goal_dim, goal_idx)."""
  bb_env_id, num_cubes, task_index = parse_sgcrl_builderbench_env_name(env_name)
  if pd_filter:
    state_dim = pd_policy_state_obs_dim(num_cubes)
    goal_idx = goal_xyz_state_indices(num_cubes, task_index, start_index=0)
  else:
    raise NotImplementedError('DISCOVER BB adapter expects PD-filtered obs')
  mask = load_task_cube_mask(num_cubes, task_index)
  goal_dim = int(3 * int(np.sum(mask)))
  return bb_env_id, num_cubes, task_index, state_dim, goal_dim, goal_idx


def task_goal(num_cubes: int, task_index: int) -> np.ndarray:
  return default_fixed_target_goal(num_cubes, task_index)


def pack_obs(state_obs, target_goal, num_cubes: int, pd_filter: bool = True):
  return _pack_obs(
      state_obs, target_goal, num_cubes=num_cubes, filter_pd_policy=pd_filter)


def achieved_from_state_obs(state_obs, goal_idx):
  return state_obs[..., goal_idx]


def set_per_env_goals(state, goals, mocap_targets, num_task_cubes: int):
  """Write a different target_goal (+ mocap) into each env.

  ``goals`` is ``(E, goal_dim)`` with ``goal_dim = 3 * num_task_cubes``.
  """
  goals = jnp.asarray(goals, dtype=jnp.float32)
  n_task = int(num_task_cubes)
  pos = goals.reshape(goals.shape[0], n_task, 3)
  info = dict(state.info)
  info['target_goal'] = goals
  info['target_mocap_pos'] = pos
  mocap_pos = state.data.mocap_pos
  mocap_pos = mocap_pos.at[:, jnp.asarray(mocap_targets)].set(pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


def set_shared_goal(state, goal, mocap_targets, num_task_cubes: int):
  """Broadcast one goal vector to every env (eval / task g*)."""
  goal = jnp.asarray(goal, dtype=jnp.float32).reshape(-1)
  n_env = int(state.obs.shape[0])
  goals = jnp.broadcast_to(goal, (n_env, goal.shape[0]))
  return set_per_env_goals(state, goals, mocap_targets, num_task_cubes)


def cube_success(achieved, target, thresh: float):
  """BB-style all-cubes-within-thresh indicator. Shapes (..., goal_dim)."""
  n = achieved.shape[-1] // 3
  err = jnp.linalg.norm(
      achieved.reshape(achieved.shape[:-1] + (n, 3))
      - target.reshape(target.shape[:-1] + (n, 3)),
      axis=-1)
  return jnp.all(err < thresh, axis=-1).astype(jnp.float32)


def sparse_ball_reward(achieved, goal, thresh: float):
  """DISCOVER-style 1{||φ(s)-g|| < ε} on the concatenated xyz vector."""
  dist = jnp.linalg.norm(achieved - goal, axis=-1)
  return (dist < thresh).astype(jnp.float32)

