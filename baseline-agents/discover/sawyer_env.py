"""Sawyer bin/peg helpers for DISCOVER: 3-D object-xyz goals, vec env, task success.

Goal representation
-------------------
DISCOVER uses object xyz (3D) as the achieved goal, matching the original
repo's ``goal_indices_2`` (task-relevant dims only).  The raw gym obs is still
14D ([state_7D | env_goal_7D]), but we only pack [state_7D | goal_3D] = 10D
for the actor.

  OBJ_IDX = slice(4, 7)   ← object xyz within the 7D state
  GOAL_DIM = 3             ← DISCOVER goal space (object xyz only)
  _GYM_OBS_DIM = 14        ← raw gym obs size (used only in SawyerVec._obs)
"""
from __future__ import annotations

import concurrent.futures
from typing import Tuple

import jax.numpy as jnp
import numpy as np

import env_utils

TASK_XYZ = {
    'sawyer_bin': np.array([0.12, 0.7, 0.02], dtype=np.float32),
    'sawyer_peg': np.array([-0.3, 0.6, 0.0], dtype=np.float32),
}

STATE_DIM = 7
GOAL_DIM = 3          # object xyz only (was 7 – caused HER reward to never fire)
OBJ_IDX = slice(4, 7)  # object xyz within the 7D state vector
_GYM_OBS_DIM = 14     # raw MetaWorld obs; separate from actor obs_dim=STATE_DIM+GOAL_DIM
ACTION_DIM = 4
EPISODE_LENGTH = 150

# MetaWorld success (object / pegHead vs task xyz). Not the DISCOVER ball.
_MW_THRESH = {
    'sawyer_bin': 0.05,
    'sawyer_peg': 0.07,
}


def env_layout(env_name: str):
  """Return (state_dim, goal_dim, action_dim, ep_len, task_xyz)."""
  if env_name not in TASK_XYZ:
    raise ValueError(f'DISCOVER Sawyer adapter supports {list(TASK_XYZ)}, got {env_name}')
  return STATE_DIM, GOAL_DIM, ACTION_DIM, EPISODE_LENGTH, TASK_XYZ[env_name].copy()


def make_gym_env(
    env_name: str,
    seed: int,
    randomize_init: bool = True,
    safe_grasp_reset: bool = False,
    randomize_tcp_z: bool = False,
):
  np.random.seed(int(seed))
  load_kwargs = {'randomize_init': bool(randomize_init)}
  if env_name == 'sawyer_bin' and safe_grasp_reset:
    load_kwargs['safe_grasp_reset'] = True
  if env_name == 'sawyer_bin' and randomize_tcp_z:
    load_kwargs['randomize_tcp_z'] = True
  gym_env, obs_dim, max_episode_steps = env_utils.load(
      env_name, **load_kwargs)
  assert int(obs_dim) == STATE_DIM, obs_dim
  assert int(max_episode_steps) == EPISODE_LENGTH
  assert int(gym_env.action_space.shape[0]) == ACTION_DIM
  return gym_env


def apply_task_goal(gym_env, env_name: str, task_xyz: np.ndarray) -> np.ndarray:
  """Write the official 3-D task target so env reward is MetaWorld success."""
  xyz = np.asarray(task_xyz, dtype=np.float64).reshape(3)
  if env_name == 'sawyer_bin':
    gym_env._goal = xyz.copy()
    gym_env._target_pos = xyz.copy()
  else:
    gym_env._goal_pos = xyz.copy()
    gym_env._target_pos = xyz.copy()
  return np.asarray(gym_env._get_obs(), dtype=np.float32)


def task_goal_3d(env_name: str) -> np.ndarray:
  """3-D goal = the official task xyz (object target position).

  DISCOVER's goal space is object xyz only, so this is just TASK_XYZ.
  """
  return TASK_XYZ[env_name].copy()


def sparse_ball_reward(achieved, goal, thresh: float):
  dist = jnp.linalg.norm(achieved - goal, axis=-1)
  return (dist < thresh).astype(jnp.float32)


def mw_task_success(next_obs: np.ndarray, env_name: str, task_xyz: np.ndarray) -> np.ndarray:
  """MetaWorld success on the object / pegHead slice of the 7-D state."""
  obj = np.asarray(next_obs[..., 4:7], dtype=np.float32)
  tgt = np.asarray(task_xyz, dtype=np.float32).reshape(3)
  if env_name == 'sawyer_peg':
    scale = np.array([1.0, 2.0, 2.0], dtype=np.float32)
    dist = np.linalg.norm((obj - tgt) * scale, axis=-1)
  else:
    dist = np.linalg.norm(obj - tgt, axis=-1)
  return (dist < float(_MW_THRESH[env_name])).astype(np.float32)


class SawyerVec:
  """Threaded gym vec. No auto-reset; caller resets each episode."""

  def __init__(
      self,
      env_name: str,
      num_envs: int,
      seed: int,
      randomize_init: bool = True,
      safe_grasp_reset: bool = False,
      randomize_tcp_z: bool = False,
  ):
    self.env_name = env_name
    self.num_envs = int(num_envs)
    self.episode_length = EPISODE_LENGTH
    self._envs = [
        make_gym_env(
            env_name,
            seed=int(seed) + i,
            randomize_init=randomize_init,
            safe_grasp_reset=safe_grasp_reset,
            randomize_tcp_z=randomize_tcp_z,
        )
        for i in range(self.num_envs)
    ]
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=self.num_envs, thread_name_prefix='discover_sawyer')
    self._t = np.zeros((self.num_envs,), dtype=np.int32)
    self._obs = np.zeros((self.num_envs, _GYM_OBS_DIM), dtype=np.float32)

  def reset(self) -> np.ndarray:
    def _one(i: int):
      return np.asarray(self._envs[i].reset(), dtype=np.float32)

    futs = [self._executor.submit(_one, i) for i in range(self.num_envs)]
    self._obs = np.stack([f.result() for f in futs], axis=0)
    self._t[:] = 0
    return self._obs.copy()

  def reset_with_task_goal(self, task_xyz: np.ndarray) -> np.ndarray:
    def _one(i: int):
      self._envs[i].reset()
      return apply_task_goal(self._envs[i], self.env_name, task_xyz)

    futs = [self._executor.submit(_one, i) for i in range(self.num_envs)]
    self._obs = np.stack([f.result() for f in futs], axis=0)
    self._t[:] = 0
    return self._obs.copy()

  def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _one(i: int):
      a = np.asarray(actions[i], dtype=np.float32)
      a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
      a = np.clip(a, -1.0, 1.0)
      obs, rew, done, _info = self._envs[i].step(a)
      return (
          np.asarray(obs, dtype=np.float32),
          float(rew),
          bool(done),
      )

    futs = [self._executor.submit(_one, i) for i in range(self.num_envs)]
    next_obs = np.zeros_like(self._obs)
    env_r = np.zeros((self.num_envs,), dtype=np.float32)
    env_done = np.zeros((self.num_envs,), dtype=np.float32)
    for i, fut in enumerate(futs):
      o, r, d = fut.result()
      next_obs[i] = o
      env_r[i] = r
      env_done[i] = float(d)
    self._t += 1
    trunc = (self._t >= self.episode_length).astype(np.float32)
    done = np.maximum(env_done, trunc)
    self._obs = next_obs
    return next_obs, env_r, done

  @property
  def obs(self) -> np.ndarray:
    return self._obs

  def close(self) -> None:
    self._executor.shutdown(wait=True)
