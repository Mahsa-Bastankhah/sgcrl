"""Gym adapter for BuilderBench creative-cube tasks (sgcrl / PPO-CRL).

Wraps a single BuilderBench ``CreativeCube`` MJX env (+ optional ``PDWrapper``)
as a classic ``gym.Env`` returning ``[state, target_goal]`` so
``contrastive.utils.make_environment`` can append the BuilderBench task goal.

Supported env ids: ``creative-{N}-task{K}`` (e.g. ``creative-1-task2``).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import gym
import numpy as np

_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

# Headless MJX rendering defaults (override in shell if needed).
os.environ.setdefault('MUJOCO_GL', 'egl')

_BUILDERBENCH_IMPORT_ERROR: Optional[BaseException] = None
try:
  import jax
  import jax.numpy as jnp
  from builderbench.constants import _MJX_PARAMS
  from builderbench.creative_cube import CreativeCube, default_config
  from utils.wrapper import EpisodeWrapper, PDWrapper
except Exception as _e:  # noqa: BLE001
  jax = None  # type: ignore
  jnp = None  # type: ignore
  CreativeCube = None  # type: ignore
  default_config = None  # type: ignore
  EpisodeWrapper = None  # type: ignore
  PDWrapper = None  # type: ignore
  _BUILDERBENCH_IMPORT_ERROR = _e

from envs.builderbench_utils import (
    creative_cube_mj_episode_length,
    default_fixed_target_goal,
    filter_pd_policy_state_obs,
    get_filtered_obs_dim,
    parse_bb_env_id,
    scaled_episode_length,
    validate_pd_episode_length,
    uniform_goal_obs_bounds as _uniform_goal_obs_bounds,
)


def _require_builderbench(env_name: str) -> None:
  if _BUILDERBENCH_IMPORT_ERROR is not None:
    raise RuntimeError(
        f'Cannot build {env_name}: builderbench failed to import.\n'
        f'  Original error: {type(_BUILDERBENCH_IMPORT_ERROR).__name__}: '
        f'{_BUILDERBENCH_IMPORT_ERROR}\n'
        f'Install with: pip install -e {_BUILDERBENCH_ROOT}\n'
        f'And set BUILDERBENCH_ROOT if the repo lives elsewhere.'
    ) from _BUILDERBENCH_IMPORT_ERROR


def parse_creative_env_id(env_id: str) -> tuple[int, int]:
  """Parse ``creative-{N}-task{K}`` ΓåÆ (num_cubes, task_id)."""
  return parse_bb_env_id(env_id)


class BuilderBenchCreativeGymEnv(gym.Env):
  """Single-env Gym wrapper around BuilderBench ``CreativeCube``.

  Observation layout returned to sgcrl:
    ``[ state (obs_dim) | target_goal (goal_dim) ]``

  ``target_goal`` is BuilderBench's ``info['target_goal']`` (absolute cube
  target xyz), *not* a slice of the state observation.
  """

  metadata = {'render_modes': []}

  def __init__(
      self,
      env_id: str = 'creative-1-task1',
      seed: Optional[int] = None,
      use_pd: bool = False,
      pd_duration: int = 5,
      pd_filter_policy_obs: bool = True,
      fixed_target_goal: Optional[np.ndarray] = None,
      obs_space_list: Optional[list[str]] = None,
      episode_length_multiplier: float = 1.0,
  ):
    super().__init__()
    _require_builderbench(env_id)

    self._env_id = env_id
    self._seed = 0 if seed is None else int(seed)
    self._use_pd = bool(use_pd)
    self._pd_duration = int(pd_duration)
    self._pd_filter_policy_obs = bool(pd_filter_policy_obs) and self._use_pd
    self._obs_space_list = obs_space_list or ["xy", "select"]
    self._fixed_target_goal = (
        None if fixed_target_goal is None
        else np.asarray(fixed_target_goal, dtype=np.float32).reshape(-1))

    num_cubes, task_id = parse_creative_env_id(env_id)
    self._num_cubes = num_cubes
    self._task_id = task_id
    cfg = default_config()
    cfg.num_cubes = num_cubes
    cfg.task_id = task_id
    self._episode_length_multiplier = float(episode_length_multiplier)
    cfg.episode_length = creative_cube_mj_episode_length(
        num_cubes, self._task_id)
    # Use JAX MJX backend (no warp-lang required). Set BUILDERBENCH_MJX_IMPL=warp
    # if warp-lang is installed and you want the faster path.
    cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
    if env_id in _MJX_PARAMS:
      cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]

    base = CreativeCube(config=cfg)
    if self._use_pd:
      validate_pd_episode_length(cfg.episode_length, self._pd_duration)
      inner = PDWrapper(base, duration=self._pd_duration)
      episode_length = cfg.episode_length // self._pd_duration
    else:
      inner = base
      episode_length = cfg.episode_length

    self._env = EpisodeWrapper(inner, episode_length=episode_length,
                               action_repeat=1)
    self._base_env = base
    self._episode_length = int(episode_length)
    self._state = None
    self._rng = jax.random.PRNGKey(self._seed)

    # Probe dims with a dry reset.
    probe = self._env.reset(jax.random.PRNGKey(0))
    self._full_state_obs_dim = int(np.asarray(probe.obs).shape[-1])
    if self._pd_filter_policy_obs:
      self._state_obs_dim = get_filtered_obs_dim(self._num_cubes, self._obs_space_list)
    else:
      self._state_obs_dim = self._full_state_obs_dim
    self._goal_dim = int(np.asarray(probe.info['target_goal']).shape[-1])
    self._action_dim = int(self._env.action_size)

    obs_high = np.full(self._state_obs_dim + self._goal_dim, np.inf,
                       dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=-obs_high, high=obs_high, dtype=np.float32)
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)
    self._max_episode_steps = self._episode_length

  @property
  def state_obs_dim(self) -> int:
    return self._state_obs_dim

  @property
  def goal_dim(self) -> int:
    return self._goal_dim

  @property
  def target_goal(self) -> np.ndarray:
    if self._state is None:
      if self._fixed_target_goal is not None:
        return self._fixed_target_goal.copy()
      return default_fixed_target_goal(self._num_cubes, self._task_id)
    return np.asarray(self._state.info['target_goal'], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    """Bounds on the goal slice for CRL uniform-sampling negatives."""
    return _uniform_goal_obs_bounds(self._num_cubes, self._task_id)

  def _pack_obs(self, state) -> np.ndarray:
    state_obs = np.asarray(state.obs, dtype=np.float32)
    if self._pd_filter_policy_obs:
      state_obs = filter_pd_policy_state_obs(state_obs, self._num_cubes, self._obs_space_list)
    goal = np.asarray(state.info['target_goal'], dtype=np.float32)
    return np.concatenate([state_obs, goal], axis=0)

  def _maybe_fix_target(self, state):
    if self._fixed_target_goal is None:
      return state
    fixed = jnp.asarray(self._fixed_target_goal, dtype=jnp.float32).reshape(-1)
    num_cubes = self._num_cubes
    fixed_pos = fixed.reshape(num_cubes, 3)
    info = dict(state.info)
    info['target_goal'] = fixed
    mocap_pos = state.data.mocap_pos.at[self._base_env._mocap_targets].set(
        fixed_pos)
    data = state.data.replace(mocap_pos=mocap_pos)
    return state.replace(data=data, info=info)

  def reset(self):
    self._rng, subkey = jax.random.split(self._rng)
    state = self._env.reset(subkey)
    state = self._maybe_fix_target(state)
    self._state = state
    return self._pack_obs(state)

  def step(self, action):
    if self._state is None:
      raise RuntimeError('step() called before reset()')
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    self._state = self._env.step(
        self._state, jnp.asarray(action, dtype=jnp.float32))
    obs = self._pack_obs(self._state)
    reward = float(np.asarray(self._state.reward))
    done = bool(np.asarray(self._state.done))
    info = {
        'success': float(np.asarray(self._state.metrics.get('success', 0.0))),
        'easy_success': float(
            np.asarray(self._state.metrics.get('easy_success', 0.0))),
        'target_goal': np.asarray(
            self._state.info['target_goal'], dtype=np.float32),
        'achieved_goal': np.asarray(
            self._state.info['achieved_goal'], dtype=np.float32),
    }
    if done:
      self._state = None
    return obs, reward, done, info


def make_builderbench_creative_env(
    env_id: str = 'creative-1-task1',
    seed: Optional[int] = None,
    use_pd: bool = False,
    pd_duration: int = 5,
    pd_filter_policy_obs: bool = True,
    fixed_target_goal: Optional[np.ndarray] = None,
    obs_space_list: Optional[list[str]] = None,
    episode_length_multiplier: float = 1.0,
) -> BuilderBenchCreativeGymEnv:
  return BuilderBenchCreativeGymEnv(
      env_id=env_id,
      seed=seed,
      use_pd=use_pd,
      pd_duration=pd_duration,
      pd_filter_policy_obs=pd_filter_policy_obs,
      fixed_target_goal=fixed_target_goal,
      obs_space_list=obs_space_list,
      episode_length_multiplier=episode_length_multiplier
  )
