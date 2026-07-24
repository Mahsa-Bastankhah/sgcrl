"""Render BuilderBench rollout videos from MPO-CRL checkpoints.

Matches training by reading ``run_config.json`` next to the checkpoint
(PD wrapper, macro episode length, obs packing / PD policy-obs filter,
fixed goals, MPO policy sizes). Uses ``target_policy_params`` (same as
training collection / eval).

Examples:
  python scripts/mpo_builderbench_rollout_video.py \\
      --checkpoint=logs/mpo_crl_.../checkpoints/latest.pkl \\
      --output=videos/builderbench/mpo_c2t1/
"""
from __future__ import annotations

import json
import os
import pickle
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'baseline-agents'))
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401 — must precede acme/jax imports

import argparse
import glob
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from acme import specs

import contrastive
from contrastive import utils as contrastive_utils
from ppo_contrastive import fixed_goal_for_env, ppo_env_defaults_for_env
import mpo_crl_learner
from envs.builderbench_utils import (
    creative_cube_mj_episode_length,
    creative_cube_full_state_obs_dim,
    filter_pd_policy_state_obs,
    is_builderbench_creative_env,
    parse_bb_env_id,
    pd_policy_state_obs_dim,
    sgcrl_env_name_to_bb_env_id,
    video_render_skip_reason,
)

from builderbench.constants import _MJX_PARAMS
from builderbench.creative_cube import CreativeCube, default_config
from utils.wrapper import (
    AutoResetWrapper,
    EpisodeWrapper,
    PDWrapper,
    VmapWrapper,
)


@dataclass
class _TrainCtx:
  use_pd: bool
  pd_duration: int
  filter_policy_obs: bool
  episode_length: int
  obs_dim: int
  policy_hidden_sizes: Tuple[int, ...]
  policy_init_scale: float
  fixed_target_goal: Optional[np.ndarray]
  start_index: int
  end_index: int
  permute_start_boxes: bool


def _get_video(
    inference_policy,
    video_env,
    video_key,
    episode_length: int,
    *,
    fixed_target_goal: Optional[np.ndarray],
    mocap_targets,
    num_cubes: int,
):
  video_env_states = _get_trajectory(
      inference_policy,
      video_env,
      video_key,
      episode_length,
      fixed_target_goal=fixed_target_goal,
      mocap_targets=mocap_targets,
      num_cubes=num_cubes,
  )
  mocap_key = 'target_mocap'
  video_images = []
  for i in range(episode_length):
    if i % 2 == 0:
      video_images.append(video_env.render_from_info(
          np.asarray(video_env_states.data.qpos[i][0]),
          np.asarray(video_env_states.data.qvel[i][0]),
          np.asarray(video_env_states.info[f'{mocap_key}_pos'][i][0]),
          np.asarray(video_env_states.info[f'{mocap_key}_quat'][i][0]),
      ))
  return video_images


def _maybe_force_target(
    state,
    fixed_target_goal: Optional[np.ndarray],
    mocap_targets,
    num_cubes: int,
):
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


def _get_trajectory(
    policy,
    env,
    key,
    unroll_length: int,
    *,
    fixed_target_goal: Optional[np.ndarray],
    mocap_targets,
    num_cubes: int,
):
  @jax.jit
  def _run(key):
    env_key, key = jax.random.split(key)
    state = env.reset(jax.random.split(env_key, 1))
    state = _maybe_force_target(
        state, fixed_target_goal, mocap_targets, num_cubes)

    def f(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      action, _ = policy(state.obs, state.info['target_goal'], act_key)
      next_state = env.step(state, action)
      next_state = _maybe_force_target(
          next_state, fixed_target_goal, mocap_targets, num_cubes)
      return (next_state, key), next_state

    _, states = jax.lax.scan(f, (state, key), (), length=unroll_length)
    return states

  return _run(key)


def _run_config_path_for_checkpoint(checkpoint_path: str) -> Optional[str]:
  if os.path.isfile(checkpoint_path):
    ckpt_dir = os.path.dirname(os.path.realpath(checkpoint_path))
  else:
    abspath = os.path.abspath(checkpoint_path)
    candidates = sorted(glob.glob(os.path.join(abspath, 'ckpt_iter_*.pkl')))
    if candidates:
      ckpt_dir = os.path.dirname(os.path.realpath(candidates[0]))
    else:
      latest = os.path.join(abspath, 'latest.pkl')
      if os.path.isfile(latest):
        ckpt_dir = os.path.dirname(os.path.realpath(latest))
      else:
        ckpt_dir = os.path.realpath(abspath)
  run_dir = os.path.dirname(ckpt_dir)
  path = os.path.join(run_dir, 'run_config.json')
  return path if os.path.isfile(path) else None


def _load_train_ctx(env_name: str, checkpoint_path: str) -> _TrainCtx:
  num_cubes, task_index = parse_bb_env_id(sgcrl_env_name_to_bb_env_id(env_name))
  mj_ep_len = creative_cube_mj_episode_length(num_cubes, task_index)
  full_obs_dim = creative_cube_full_state_obs_dim(num_cubes)
  pd_obs_dim = pd_policy_state_obs_dim(num_cubes)

  cfg_path = _run_config_path_for_checkpoint(checkpoint_path)
  if cfg_path is not None:
    with open(cfg_path, 'r', encoding='utf-8') as fh:
      run_cfg: Dict[str, Any] = json.load(fh)
    flags = run_cfg.get('flags', {})
    resolved = run_cfg.get('resolved_config', {})
    mpo_cfg = run_cfg.get('mpo_config', {})
    env_defaults = run_cfg.get('env_defaults', {})
    use_pd = bool(flags.get('builderbench_use_pd', False))
    pd_duration = int(flags.get('builderbench_pd_duration', 5))
    obs_dim = int(resolved.get('obs_dim', full_obs_dim))
    hidden = tuple(int(x) for x in mpo_cfg.get(
        'policy_hidden_sizes', (256, 256, 256)))
    init_scale = float(mpo_cfg.get('policy_init_scale', 0.7))
    start_index = int(resolved.get(
        'start_index', env_defaults.get('start_index', 0)))
    end_index = int(resolved.get(
        'end_index', env_defaults.get('end_index', num_cubes * 3)))
    fixed = run_cfg.get('fixed_start_end')
    fixed_goal = (None if fixed is None
                  else np.asarray(fixed, dtype=np.float32))
    permute_start_boxes = bool(
        flags.get('builderbench_permute_start_boxes', True))
  else:
    use_pd = False
    pd_duration = 5
    obs_dim = full_obs_dim
    defaults = ppo_env_defaults_for_env(env_name) or {}
    hidden = (256, 256, 256)
    init_scale = 0.7
    start_index = int(defaults.get('start_index', 0))
    end_index = int(defaults.get('end_index', num_cubes * 3))
    fixed_goal = fixed_goal_for_env(env_name)
    permute_start_boxes = True

  if use_pd:
    macro_ep_len = mj_ep_len // pd_duration
    filter_policy = (obs_dim == pd_obs_dim)
    if obs_dim not in (pd_obs_dim, full_obs_dim):
      print(f'[mpo_bb_video] WARNING: obs_dim={obs_dim} unexpected for PD; '
            f'assuming filter={obs_dim == pd_obs_dim}')
  else:
    macro_ep_len = mj_ep_len
    filter_policy = False

  return _TrainCtx(
      use_pd=use_pd,
      pd_duration=pd_duration,
      filter_policy_obs=filter_policy,
      episode_length=int(macro_ep_len),
      obs_dim=int(obs_dim),
      policy_hidden_sizes=hidden,
      policy_init_scale=init_scale,
      fixed_target_goal=fixed_goal,
      start_index=start_index,
      end_index=end_index,
      permute_start_boxes=permute_start_boxes,
  )


def _build_policy_network(env_name: str, seed: int, ctx: _TrainCtx):
  env_kwargs: Dict[str, Any] = {}
  if ctx.use_pd:
    env_kwargs['builderbench_use_pd'] = True
    env_kwargs['builderbench_pd_duration'] = ctx.pd_duration
    env_kwargs['builderbench_pd_filter_policy_obs'] = ctx.filter_policy_obs
  env_kwargs['builderbench_permute_start_boxes'] = ctx.permute_start_boxes

  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name,
      ctx.start_index,
      ctx.end_index,
      seed=seed,
      fixed_start_end=ctx.fixed_target_goal,
      **env_kwargs,
  )
  if int(obs_dim) != int(ctx.obs_dim):
    print(f'[mpo_bb_video] WARNING: probe obs_dim={obs_dim} != '
          f'run_config obs_dim={ctx.obs_dim}')
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env
  return mpo_crl_learner.make_mpo_policy(
      env_spec,
      hidden_sizes=ctx.policy_hidden_sizes,
      init_scale=ctx.policy_init_scale,
  )


def _enumerate_checkpoints(path: str) -> List[Tuple[str, str]]:
  if os.path.isfile(path):
    base = os.path.splitext(os.path.basename(path))[0]
    label = base[len('ckpt_'):] if base.startswith('ckpt_') else base
    return [(label, path)]

  if not os.path.isdir(path):
    raise FileNotFoundError(f'Checkpoint path not found: {path}')

  iter_files = glob.glob(os.path.join(path, 'ckpt_iter_*.pkl'))

  def _it(fname):
    m = re.search(r'ckpt_iter_(\d+)\.pkl$', fname)
    return int(m.group(1)) if m else -1

  iter_files.sort(key=_it)
  entries = [(f'iter_{_it(f):07d}', f) for f in iter_files]
  latest = os.path.join(path, 'latest.pkl')
  if os.path.isfile(latest):
    entries.append(('latest', latest))
  return entries


def _make_bb_env(env_id: str, ctx: _TrainCtx):
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.episode_length = creative_cube_mj_episode_length(num_cubes, task_id)
  cfg.permute_start_boxes = bool(ctx.permute_start_boxes)
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  if env_id in _MJX_PARAMS:
    cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]

  base = CreativeCube(config=cfg)
  mocap_targets = base._mocap_targets

  if ctx.use_pd:
    assert cfg.episode_length % ctx.pd_duration == 0, (
        f'episode_length {cfg.episode_length} must divide pd_duration '
        f'{ctx.pd_duration}')
    inner = PDWrapper(base, duration=ctx.pd_duration)
    macro_ep_len = cfg.episode_length // ctx.pd_duration
  else:
    inner = base
    macro_ep_len = cfg.episode_length

  env = VmapWrapper(inner)
  env = EpisodeWrapper(env, episode_length=int(macro_ep_len), action_repeat=1)
  env = AutoResetWrapper(env)
  env.render_from_info = base.render_from_info  # type: ignore[attr-defined]
  env._mocap_targets_geom = base._mocap_targets_geom  # type: ignore[attr-defined]
  env.model = base._mj_model  # type: ignore[attr-defined]
  return env, base, mocap_targets, int(macro_ep_len)


def _make_policy_fn(
    policy_network,
    policy_params,
    stochastic: bool,
    *,
    filter_policy_obs: bool,
    num_cubes: int,
):
  @jax.jit
  def policy(obs, goals, key):
    if filter_policy_obs:
      obs = filter_pd_policy_state_obs(obs, num_cubes)
    packed = jnp.concatenate([obs, goals], axis=-1)
    distribution = policy_network.apply(policy_params, packed)
    if stochastic:
      action = distribution.sample(seed=key)
    else:
      action = distribution.mode()
    return jnp.clip(action, -1.0, 1.0), {}

  return policy


def _load_mpo_checkpoint(path: str) -> Dict[str, Any]:
  with open(path, 'rb') as handle:
    payload = pickle.load(handle)
  state = payload['state']
  return {
      'policy_params': state.target_policy_params,
      'iteration': int(payload.get('iteration', -1)),
      'global_step': int(payload.get('global_step', -1)),
  }


def _write_video(frames, path: str, fps: int):
  import imageio.v2 as imageio
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


def _resolve_output_path(output_arg: str, run_tag: str, label: str,
                         multi: bool) -> str:
  is_dir_like = output_arg.endswith(os.sep) or os.path.isdir(output_arg)
  if is_dir_like:
    return os.path.join(output_arg, f'{run_tag}_{label}.mp4')
  if not multi:
    return output_arg
  stem, ext = os.path.splitext(output_arg)
  ext = ext or '.mp4'
  return f'{stem}_{label}{ext}'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env', default='builderbench_creative_2_task1')
  parser.add_argument('--output', required=True)
  parser.add_argument('--fps', type=int, default=10)
  parser.add_argument('--stochastic', action='store_true')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--run_tag', default=None,
                      help='Prefix for output filenames (default: env name).')
  parser.add_argument(
      '--skip_existing', action='store_true',
      help='Skip checkpoints whose output mp4 already exists.')
  args = parser.parse_args()

  if not is_builderbench_creative_env(args.env):
    parser.error(
        f'--env must match builderbench_creative_{{N}}_task{{K}} '
        f'(got {args.env!r})')

  env_id = sgcrl_env_name_to_bb_env_id(args.env)
  num_cubes, _ = parse_bb_env_id(env_id)
  run_tag = args.run_tag or args.env

  ckpt_entries = _enumerate_checkpoints(args.checkpoint)
  if not ckpt_entries:
    print(f'[mpo_bb_video] no checkpoints found at {args.checkpoint!r}')
    return
  print(f'[mpo_bb_video] found {len(ckpt_entries)} checkpoint(s)')

  ctx = _load_train_ctx(args.env, args.checkpoint)
  cfg_path = _run_config_path_for_checkpoint(args.checkpoint)
  if cfg_path is not None:
    skip = video_render_skip_reason(cfg_path)
    if skip is not None:
      print(f'[mpo_bb_video] skip: {skip}')
      return
  print(f'[mpo_bb_video] training context: use_pd={ctx.use_pd} '
        f'pd_duration={ctx.pd_duration} filter_policy_obs={ctx.filter_policy_obs} '
        f'obs_dim={ctx.obs_dim} ep_len={ctx.episode_length} '
        f'hidden={ctx.policy_hidden_sizes}')

  print('[mpo_bb_video] building MPO policy and builderbench env...')
  policy_network = _build_policy_network(args.env, seed=args.seed, ctx=ctx)
  video_env, _base, mocap_targets, episode_length = _make_bb_env(env_id, ctx)
  print(f'[mpo_bb_video] env_id={env_id}  episode_length={episode_length}')

  key = jax.random.PRNGKey(args.seed)
  multi = len(ckpt_entries) > 1
  out_dir = args.output
  if out_dir.endswith(os.sep) or not out_dir.endswith('.mp4'):
    os.makedirs(out_dir, exist_ok=True)

  for label, path in ckpt_entries:
    out_path = _resolve_output_path(out_dir, run_tag, label, multi)
    if args.skip_existing and os.path.isfile(out_path):
      print(f'[mpo_bb_video] skip existing {out_path}', flush=True)
      continue
    print(f'[mpo_bb_video] === {label}  ({path}) ===')
    ckpt = _load_mpo_checkpoint(path)
    policy_params = ckpt['policy_params']
    print(f'[mpo_bb_video]   iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}')

    policy = _make_policy_fn(
        policy_network,
        policy_params,
        args.stochastic,
        filter_policy_obs=ctx.filter_policy_obs,
        num_cubes=num_cubes,
    )
    key, video_key = jax.random.split(key)
    frames = _get_video(
        policy,
        video_env,
        video_key,
        episode_length,
        fixed_target_goal=ctx.fixed_target_goal,
        mocap_targets=mocap_targets,
        num_cubes=num_cubes,
    )
    _write_video(frames, out_path, fps=args.fps)
    print(f'[mpo_bb_video] wrote {out_path} ({len(frames)} frames)', flush=True)


if __name__ == '__main__':
  main()
