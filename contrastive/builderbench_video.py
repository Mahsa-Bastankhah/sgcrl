"""BuilderBench deterministic rollout video helpers (in-train + offline).

Uses the CreativeCube + PD/Episode wrapper stack (same as
``scripts/ppo_builderbench_rollout_video.py``) so rendered frames match
training macro-steps. Policy actions are deterministic (``dist.mode()``)
and optionally apply the same packed-obs normalization as PPO.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

from contrastive.ppo_learner import _normalize_packed_obs
from envs.builderbench_utils import (
    creative_cube_mj_episode_length,
    filter_pd_policy_state_obs,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

from builderbench.constants import _MJX_PARAMS
from builderbench.creative_cube import CreativeCube, default_config
from utils.wrapper import (
    AutoResetWrapper,
    EpisodeWrapper,
    PDWrapper,
    VmapWrapper,
)


def make_bb_video_env(
    env_name: str,
    *,
    use_pd: bool,
    pd_duration: int,
    permute_start_boxes: bool,
    filter_policy_obs: bool,
):
  """Build a single-env batched BuilderBench stack with render helpers."""
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.episode_length = creative_cube_mj_episode_length(num_cubes, task_id)
  cfg.permute_start_boxes = bool(permute_start_boxes)
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  if env_id in _MJX_PARAMS:
    cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]

  base = CreativeCube(config=cfg)
  mocap_targets = base._mocap_targets

  if use_pd:
    assert cfg.episode_length % pd_duration == 0, (
        f'episode_length {cfg.episode_length} must divide pd_duration '
        f'{pd_duration}')
    inner = PDWrapper(base, duration=int(pd_duration))
    macro_ep_len = cfg.episode_length // int(pd_duration)
  else:
    inner = base
    macro_ep_len = cfg.episode_length

  env = VmapWrapper(inner)
  env = EpisodeWrapper(env, episode_length=int(macro_ep_len), action_repeat=1)
  env = AutoResetWrapper(env)
  env.render_from_info = base.render_from_info  # type: ignore[attr-defined]
  env._mocap_targets_geom = base._mocap_targets_geom  # type: ignore[attr-defined]
  env.model = base._mj_model  # type: ignore[attr-defined]
  return env, mocap_targets, int(macro_ep_len), int(num_cubes), bool(filter_policy_obs)


def _maybe_fix_target(state, fixed_target_goal, mocap_targets, num_cubes: int):
  if fixed_target_goal is None:
    return state
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(num_cubes, 3)
  info = dict(state.info)
  info['target_goal'] = jnp.broadcast_to(
      fixed, state.info['target_goal'].shape)
  info['target_mocap_pos'] = jnp.broadcast_to(
      fixed_pos, state.info['target_mocap_pos'].shape)
  mocap_pos = state.data.mocap_pos.at[mocap_targets].set(fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


def write_video(frames, path: str, fps: int = 10):
  import imageio.v2 as imageio
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


def render_deterministic_episode(
    *,
    video_env,
    mocap_targets,
    num_cubes: int,
    episode_length: int,
    networks,
    policy_params,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    fixed_target_goal: Optional[np.ndarray],
    seed: int,
    filter_policy_obs: bool,
    normalize_obs: bool,
    obs_dim: int,
    start_index: int,
    end_index: int,
    obs_norm_clip: float = 10.0,
    fps: int = 10,
    out_path: str,
) -> Tuple[str, int]:
  """Roll out one deterministic episode with live obs stats and write mp4.

  Returns ``(out_path, n_frames)``.
  """
  obs_mean_j = jnp.asarray(obs_mean, dtype=jnp.float32)
  obs_var_j = jnp.asarray(obs_var, dtype=jnp.float32)
  _filter = bool(filter_policy_obs)
  _norm = bool(normalize_obs)
  _obs_dim = int(obs_dim)
  _si = int(start_index)
  _ei = int(end_index if end_index != -1 else obs_dim)
  _clip = float(obs_norm_clip)
  policy_apply = networks.policy_network.apply

  @jax.jit
  def _policy(obs, goals, obs_mean, obs_var):
    if _filter:
      obs = filter_pd_policy_state_obs(obs, num_cubes)
    packed = jnp.concatenate([obs, goals], axis=-1)
    packed = _normalize_packed_obs(
        packed, obs_mean, obs_var,
        obs_dim=_obs_dim, start_index=_si, end_index=_ei,
        clip=_clip, enabled=_norm)
    dist = policy_apply(policy_params, packed)
    return dist.mode()

  @jax.jit
  def _run(key, obs_mean, obs_var):
    env_key, key = jax.random.split(key)
    state = video_env.reset(jax.random.split(env_key, 1))
    state = _maybe_fix_target(
        state, fixed_target_goal, mocap_targets, num_cubes)

    def f(carry, _):
      state, key = carry
      key, _ = jax.random.split(key)
      action = _policy(
          state.obs, state.info['target_goal'], obs_mean, obs_var)
      next_state = video_env.step(state, action)
      next_state = _maybe_fix_target(
          next_state, fixed_target_goal, mocap_targets, num_cubes)
      return (next_state, key), next_state

    _, states = jax.lax.scan(
        f, (state, key), (), length=int(episode_length))
    return states

  states = _run(jax.random.PRNGKey(int(seed)), obs_mean_j, obs_var_j)
  jax.block_until_ready(states.data.qpos)

  frames = []
  for i in range(int(episode_length)):
    if i % 2 != 0:
      continue
    frames.append(video_env.render_from_info(
        np.asarray(states.data.qpos[i][0]),
        np.asarray(states.data.qvel[i][0]),
        np.asarray(states.info['target_mocap_pos'][i][0]),
        np.asarray(states.info['target_mocap_quat'][i][0]),
    ))
  write_video(frames, out_path, fps=int(fps))
  return out_path, len(frames)
