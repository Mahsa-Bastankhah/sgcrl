"""Batched JAX vec-env for BuilderBench (sgcrl PPO-CRL).

Uses the same wrapper stack as ``builderbench/utils/wrapper.py::wrap_env``
(``VmapWrapper`` + ``EpisodeWrapper`` + ``AutoResetWrapper``) but exposes the
``VecEnv`` API expected by ``contrastive.ppo_learner``:

    obs = vec_env.reset()                         # (E, obs_dim_total)
    next_obs, rew, dones, terminal_obs, info_rew = vec_env.step(actions)

Training logic (PPO, CRL, GAE) stays unchanged; only rollout collection is
accelerated via a single ``jax.jit`` batched env step on GPU.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

os.environ.setdefault('MUJOCO_GL', 'egl')

_BUILDERBENCH_IMPORT_ERROR: Optional[BaseException] = None
try:
  import jax
  import jax.numpy as jnp
  from builderbench.constants import _MJX_PARAMS
  from builderbench.creative_cube import CreativeCube, default_config
  from builderbench.env_utils import State
  from utils.wrapper import (
      AutoResetWrapper,
      EpisodeWrapper,
      PDWrapper,
      VmapWrapper,
      Wrapper,
  )
  from envs.builderbench_utils import (
      filter_pd_policy_state_obs,
      get_filtered_obs_dim,
      parse_bb_env_id,
      scaled_episode_length,
      validate_pd_episode_length,
      sgcrl_env_name_to_bb_env_id,
      uniform_goal_obs_bounds as _uniform_goal_obs_bounds,
  )
except Exception as _e:  # noqa: BLE001
  jax = None  # type: ignore
  jnp = None  # type: ignore
  CreativeCube = None  # type: ignore
  default_config = None  # type: ignore
  State = None  # type: ignore
  AutoResetWrapper = None  # type: ignore
  EpisodeWrapper = None  # type: ignore
  PDWrapper = None  # type: ignore
  VmapWrapper = None  # type: ignore
  Wrapper = None  # type: ignore
  _BUILDERBENCH_IMPORT_ERROR = _e


def _require_builderbench(env_name: str) -> None:
  if _BUILDERBENCH_IMPORT_ERROR is not None:
    raise RuntimeError(
        f'Cannot build {env_name}: builderbench failed to import.\n'
        f'  Original error: {type(_BUILDERBENCH_IMPORT_ERROR).__name__}: '
        f'{_BUILDERBENCH_IMPORT_ERROR}\n'
        f'Install with: pip install -e {_BUILDERBENCH_ROOT}\n'
        f'And set BUILDERBENCH_ROOT if the repo lives elsewhere.'
    ) from _BUILDERBENCH_IMPORT_ERROR


from envs.builderbench_utils import (
    filter_pd_policy_state_obs,
    get_filtered_obs_dim,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
    uniform_goal_obs_bounds as _uniform_goal_obs_bounds,
)


class TerminalObsWrapper(Wrapper):
  """Record pre-autoreset observations so CRL can store true episode terminals."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    info = dict(state.info)
    info['terminal_obs'] = state.obs
    info['terminal_target_goal'] = state.info['target_goal']
    return state.replace(info=info)

  def step(self, state: State, action: jax.Array) -> State:
    state = self.env.step(state, action)
    info = dict(state.info)
    info['terminal_obs'] = state.obs
    info['terminal_target_goal'] = state.info['target_goal']
    return state.replace(info=info)


def _wrap_batched_env(env, episode_length: int):
  env = VmapWrapper(env)
  env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
  env = TerminalObsWrapper(env)
  env = AutoResetWrapper(env)
  return env


def _pack_obs(
    state_obs: jax.Array,
    target_goal: jax.Array,
    *,
    num_cubes: int = 0,
    filter_pd_policy: bool = False,
    obs_space_list: Optional[list[str]] = None,
) -> jax.Array:
  if filter_pd_policy:
    state_obs = filter_pd_policy_state_obs(state_obs, num_cubes, obs_space_list or ["xy", "select"])
  return jnp.concatenate([state_obs, target_goal], axis=-1)


def _maybe_fix_target(
    state: State,
    fixed_target_goal: Optional[jax.Array],
    mocap_targets: jax.Array,
    num_cubes: int,
) -> State:
  if fixed_target_goal is None:
    return state
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(num_cubes, 3)
  goal = jnp.broadcast_to(fixed, state.info['target_goal'].shape)
  info = dict(state.info)
  info['target_goal'] = goal
  mocap_pos = state.data.mocap_pos.at[mocap_targets].set(fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


class JaxBuilderBenchVecEnv:
  """GPU-batched BuilderBench env with the ``ppo_learner.VecEnv`` API."""

  def __init__(
      self,
      env_name: str,
      num_envs: int,
      seed: int,
      use_pd: bool = False,
      pd_duration: int = 5,
      pd_filter_policy_obs: bool = True,
      fixed_target_goal: Optional[np.ndarray] = None,
      obs_space_list: Optional[list[str]] = None,
    episode_length_multiplier: float = 1.0,

  ):
    _require_builderbench(env_name)
    self._env_name = str(env_name)
    self._bb_env_id = sgcrl_env_name_to_bb_env_id(env_name)
    self._num_cubes, self._task_id = parse_bb_env_id(self._bb_env_id)
    self._num_envs = int(num_envs)
    self._seed = int(seed)
    self._use_pd = bool(use_pd)
    self._pd_duration = int(pd_duration)
    self._pd_filter_policy_obs = bool(pd_filter_policy_obs) and self._use_pd
    self._fixed_target_goal = (
        None if fixed_target_goal is None
        else jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1))

    num_cubes, task_id = self._num_cubes, self._task_id
    cfg = default_config()
    cfg.num_cubes = num_cubes
    cfg.task_id = task_id
    cfg.episode_length = scaled_episode_length(
              num_cubes, episode_length_multiplier)
    self._episode_length_multiplier = float(episode_length_multiplier)
    cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
    if self._bb_env_id in _MJX_PARAMS:
      ncon, njmax = _MJX_PARAMS[self._bb_env_id]
      cfg.nconmax = int(ncon) * self._num_envs
      cfg.njmax = int(njmax)

    base = CreativeCube(config=cfg)
    self._mocap_targets = base._mocap_targets
    if self._use_pd:
      validate_pd_episode_length(cfg.episode_length, self._pd_duration)
      inner = PDWrapper(base, duration=self._pd_duration)
      episode_length = cfg.episode_length // self._pd_duration
    else:
      inner = base
      episode_length = cfg.episode_length

    self._env = _wrap_batched_env(inner, episode_length=int(episode_length))
    self._episode_length = int(episode_length)
    self._obs_space_list = obs_space_list or ["xy", "select"]

    probe_key = jax.random.split(jax.random.PRNGKey(0), self._num_envs)
    probe = self._env.reset(probe_key)
    self._full_state_obs_dim = int(probe.obs.shape[-1])
    if self._pd_filter_policy_obs:
      self._state_obs_dim = get_filtered_obs_dim(self._num_cubes, self._obs_space_list)
    else:
      self._state_obs_dim = self._full_state_obs_dim
    self._goal_dim = int(probe.info['target_goal'].shape[-1])
    self._action_dim = int(self._env.action_size)
    self._obs_dim_total = self._state_obs_dim + self._goal_dim

    self._rng = jax.random.PRNGKey(self._seed)
    self._state: Optional[State] = None

    self._reset_fn = jax.jit(self._reset_impl)
    self._step_fn = jax.jit(self._step_impl)

    _pd_obs_msg = (
        f' pd_policy_obs={self._state_obs_dim}'
        f' (full_state={self._full_state_obs_dim})'
        if self._pd_filter_policy_obs else '')
    print(f'[jax_vec] BuilderBench {self._bb_env_id}: '
            f'E={self._num_envs} obs={self._obs_dim_total} '
            f'act={self._action_dim} ep_len={self._episode_length} '
            f'use_pd={self._use_pd}{_pd_obs_msg} '
            f'ep_len_mult={self._episode_length_multiplier}')

  def _reset_impl(self, rng: jax.Array) -> State:
    state = self._env.reset(rng)
    return _maybe_fix_target(
        state, self._fixed_target_goal, self._mocap_targets, self._num_cubes)

  def _step_impl(self, state: State, actions: jax.Array) -> State:
    actions = jnp.clip(actions, -1.0, 1.0)
    state = self._env.step(state, actions)
    return _maybe_fix_target(
        state, self._fixed_target_goal, self._mocap_targets, self._num_cubes)

  def uniform_goal_obs_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    return _uniform_goal_obs_bounds(self._num_cubes, self._task_id)

  def pack_obs_from_state(self, state: State) -> jnp.ndarray:
    return _pack_obs(
        state.obs,
        state.info['target_goal'],
        num_cubes=self._num_cubes,
        filter_pd_policy=self._pd_filter_policy_obs,
        obs_space_list=self._obs_space_list,
    )

  def compile_generate_unroll(
      self,
      act_and_value_fn: Callable,
      unroll_length: int,
      obs_dim: int,
  ):
    """Build a ``jax.lax.scan`` rollout collector (BuilderBench-style).

    ``act_and_value_fn(policy_p, value_p, packed_obs, key)``
    must return ``(actions, logprobs, values)`` ΓÇö same contract as
    ``ppo_learner.act_and_value``.
    """
    step_fn = self._step_fn
    _obs_dim = int(obs_dim)
    _T = int(unroll_length)
    _num_cubes = self._num_cubes
    _filter_pd = self._pd_filter_policy_obs
    _obs_space_list = self._obs_space_list

    @jax.jit
    def generate_unroll(
        env_state: State,
        policy_params: Any,
        value_params: Any,
        key: jax.Array,
        next_done_init: jax.Array,
        s0_states_init: jax.Array,
    ):
      def f(carry, _):
        env_state, key, next_done, s0_states = carry
        packed_obs = _pack_obs(
            env_state.obs, env_state.info['target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        key, k_act = jax.random.split(key)
        actions, logprobs, values = act_and_value_fn(
            policy_params, value_params, packed_obs, k_act)
        next_state = step_fn(env_state, actions)

        terminal_obs = _pack_obs(
            next_state.info['terminal_obs'],
            next_state.info['terminal_target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        next_packed = _pack_obs(
            next_state.obs, next_state.info['target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        dones = next_state.done
        term_obs = jnp.where(dones[:, None], terminal_obs, next_packed)

        step_out = {
            'obs': packed_obs,
            'roll_dones': next_done,
            'actions': actions,
            'logprobs': logprobs,
            'values': values,
            'env_rew': next_state.reward,
            'success': next_state.metrics['success'],
            'step_dones': dones.astype(jnp.float32),
            'terminal_obs': term_obs,
            'next_obs': next_packed,
            's0_states': s0_states,
        }
        s0_next = jnp.where(
            dones[:, None], next_packed[:, :_obs_dim], s0_states)
        next_done_out = dones.astype(jnp.float32)
        return (next_state, key, next_done_out, s0_next), step_out

      (final_state, key, final_next_done, final_s0), steps = jax.lax.scan(
          f,
          (env_state, key, next_done_init, s0_states_init),
          (),
          length=_T,
      )
      return (final_state, key, final_next_done, final_s0), steps

    return generate_unroll

  def compile_eval_unroll(
      self,
      eval_policy_fn: Callable,
      unroll_length: int,
  ):
    """Batched eval rollout: deterministic policy + env scan on GPU.

    ``eval_policy_fn(policy_params, packed_obs)`` must return actions in
    ``[-1, 1]`` with batch shape ``(E, act_dim)``.
    """
    step_fn = self._step_fn
    _T = int(unroll_length)
    _num_cubes = self._num_cubes
    _filter_pd = self._pd_filter_policy_obs
    _obs_space_list = self._obs_space_list

    @jax.jit
    def eval_unroll(env_state: State, policy_params: Any):
      def f(carry, _):
        env_state = carry
        packed_obs = _pack_obs(
            env_state.obs, env_state.info['target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        actions = eval_policy_fn(policy_params, packed_obs)
        next_state = step_fn(env_state, actions)
        step_out = {
            'reward': next_state.reward,
            'success': next_state.metrics['success'],
            'state_obs': next_state.obs,
            'goal': next_state.info['target_goal'],
        }
        return next_state, step_out

      _, steps = jax.lax.scan(f, env_state, (), length=_T)
      return steps

    return eval_unroll

  @property
  def episode_length(self) -> int:
    return self._episode_length

  def reset_state(self) -> State:
    """Reset and return the internal JAX env state (for eval scans)."""
    self._rng, subkey = jax.random.split(self._rng)
    keys = jax.random.split(subkey, self._num_envs)
    self._state = self._reset_fn(keys)
    return self._state

  def reset(self) -> np.ndarray:
    self._rng, subkey = jax.random.split(self._rng)
    keys = jax.random.split(subkey, self._num_envs)
    self._state = self._reset_fn(keys)
    packed = self.pack_obs_from_state(self._state)
    return np.asarray(packed, dtype=np.float32)

  def step(
      self, actions: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if self._state is None:
      raise RuntimeError('step() called before reset()')

    actions = np.asarray(actions, dtype=np.float32)
    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = np.clip(actions, -1.0, 1.0)

    self._state = self._step_fn(self._state, jnp.asarray(actions))

    next_obs = self.pack_obs_from_state(self._state)
    terminal_obs = _pack_obs(
        self._state.info['terminal_obs'],
        self._state.info['terminal_target_goal'],
        num_cubes=self._num_cubes,
        filter_pd_policy=self._pd_filter_policy_obs,
        obs_space_list=self._obs_space_list,
    )
    dones = self._state.done.astype(bool)
    env_rewards = self._state.reward.astype(np.float32)
    info_rewards = np.full(self._num_envs, np.nan, dtype=np.float32)

    next_np = np.asarray(next_obs, dtype=np.float32)
    term_np = np.asarray(terminal_obs, dtype=np.float32)
    # Match VecEnv: terminal_obs == next_obs when the episode did not end.
    term_np = np.where(dones[:, None], term_np, next_np)

    return (
        next_np,
        np.asarray(env_rewards, dtype=np.float32),
        np.asarray(dones, dtype=bool),
        term_np,
        info_rewards,
    )

  @property
  def num_envs(self) -> int:
    return self._num_envs

  @property
  def observation_shape(self) -> Tuple[int, ...]:
    return (self._obs_dim_total,)

  @property
  def action_shape(self) -> Tuple[int, ...]:
    return (self._action_dim,)

  def stagger_resets(self):
    """Randomly advances environments by t ~ Uniform(0, episode_length) to desynchronize rollouts."""
    if self._state is None:
      self.reset_state()

    self._rng, k_offset, k_scan = jax.random.split(self._rng, 3)
    offsets = jax.random.randint(
        k_offset, (self._num_envs,), 0, self._episode_length)

    @jax.jit
    def _warmup_scan(carry, step_idx):
      state, rng = carry
      rng, a_key = jax.random.split(rng)

      # Sample random actions to progress the environment
      actions = jax.random.uniform(
          a_key, (self._num_envs, self._action_dim), minval=-1.0, maxval=1.0)
      next_state = self._step_impl(state, actions)

      # Mask: Only accept the next_state if this env hasn't reached its target offset yet
      mask = step_idx < offsets
      state = jax.tree_util.tree_map(
          lambda ns, s: jnp.where(mask.reshape(
              (-1,) + (1,) * (ns.ndim - 1)), ns, s),
          next_state, state
      )
      return (state, rng), None

    (self._state, _), _ = jax.lax.scan(_warmup_scan,
                                       (self._state, k_scan), jnp.arange(self._episode_length))
    print(
        f'[jax_vec] Staggered resets applied (max offset: {self._episode_length})')
