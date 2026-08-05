"""Render BuilderBench rollout videos from sgcrl PPO-CRL checkpoints.

Matches training setup by reading ``run_config.json`` next to the checkpoint
(PD wrapper, macro episode length, obs packing / PD policy-obs filter, fixed
goals, network sizes).

Examples:
  python scripts/ppo_builderbench_rollout_video.py \\
      --checkpoint=logs/ppo_builderbench_creative1_task1/.../checkpoints/latest.pkl \\
      --output=videos/ppo_builderbench_creative1_task1/
"""
import json
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401 ΓÇö must precede acme/jax imports

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
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from ppo_contrastive import fixed_goal_for_env, ppo_env_defaults_for_env
from envs.builderbench_utils import (
    creative_cube_mj_episode_length,
    creative_cube_full_state_obs_dim,
    filter_pd_policy_state_obs,
    get_filtered_obs_dim,
    is_builderbench_creative_env,
    parse_bb_env_id,
    pd_policy_state_obs_dim,
    scaled_episode_length,
    set_task_mocap_pos,
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
  hidden_layer_sizes: Tuple[int, ...]
  fixed_target_goal: Optional[np.ndarray]
  actor_min_std: float
  start_index: int
  end_index: int
  obs_space_list: list[str]
  permute_start_boxes: bool
  episode_length_multiplier: float = 1.0
  obs_norm_mode: str = 'none'
  z_scale_multiplier: float = 3.0
  rsnorm_clip: float = 10.0
  ppo_cleanrl_actor: bool = True
  ppo_norm_obs: bool = False
  ppo_obs_norm_clip: float = 10.0
  categorical_select_classes: Optional[int] = None


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
  """Roll out one episode and render frames (matches training PD macro steps)."""
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


def _maybe_fix_target(
    state,
    fixed_target_goal: Optional[np.ndarray],
    mocap_targets,
    num_cubes: int,
):
  del num_cubes  # full scene count; masked goals may be shorter
  if fixed_target_goal is None:
    return state
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(int(fixed.shape[0] // 3), 3)
  info = dict(state.info)
  info['target_goal'] = jnp.broadcast_to(
      fixed, state.info['target_goal'].shape)
  # render_from_info draws translucent targets from this info field.
  info['target_mocap_pos'] = jnp.broadcast_to(
      fixed_pos, state.info['target_mocap_pos'].shape)
  mocap_pos = set_task_mocap_pos(
      state.data.mocap_pos, mocap_targets, fixed_pos)
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
    state = _maybe_fix_target(
        state, fixed_target_goal, mocap_targets, num_cubes)

    def f(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      action, _ = policy(state.obs, state.info['target_goal'], act_key)
      next_state = env.step(state, action)
      # Re-pin after step (AutoReset can reintroduce randomized goals).
      next_state = _maybe_fix_target(
          next_state, fixed_target_goal, mocap_targets, num_cubes)
      return (next_state, key), next_state

    _, states = jax.lax.scan(f, (state, key), (), length=unroll_length)
    return states

  return _run(key)


def _run_config_path_for_checkpoint(checkpoint_path: str) -> Optional[str]:
  """Locate run_config.json for a ckpt file or directory.

  Follows symlinks so stride/staging dirs (symlink farms of ckpt_iter_*.pkl)
  still resolve to the real ``.../run_dir/run_config.json``.
  """
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
  """Infer training settings from run_config.json or env defaults."""
  num_cubes, task_index = parse_bb_env_id(sgcrl_env_name_to_bb_env_id(env_name))
  mj_ep_len = creative_cube_mj_episode_length(num_cubes, task_index)
  full_obs_dim = creative_cube_full_state_obs_dim(num_cubes)

  cfg_path = _run_config_path_for_checkpoint(checkpoint_path)
  if cfg_path is not None:
    with open(cfg_path, 'r', encoding='utf-8') as fh:
      run_cfg: Dict[str, Any] = json.load(fh)
    flags = run_cfg.get('flags', {})
    obs_space_str = flags.get('obs_space', 'xy,select')
    obs_space_list = [s.strip() for s in obs_space_str.split(',')]
    ep_mult = float(flags.get('builderbench_episode_length_multiplier', 1.0))
    mj_ep_len = scaled_episode_length(num_cubes, ep_mult)
    pd_obs_dim = get_filtered_obs_dim(num_cubes, obs_space_list)
    resolved = run_cfg.get('resolved_config', {})
    ppo_defaults = run_cfg.get('ppo_env_defaults', {})
    use_pd = bool(flags.get('builderbench_use_pd', False))
    pd_duration = int(flags.get('builderbench_pd_duration', 5))
    obs_dim = int(resolved.get('obs_dim', full_obs_dim))
    hidden = tuple(int(x) for x in resolved.get(
        'hidden_layer_sizes', contrastive.ContrastiveConfig().hidden_layer_sizes))
    actor_min_std = float(resolved.get(
        'ppo_actor_min_std', contrastive.ContrastiveConfig().ppo_actor_min_std))
    start_index = int(resolved.get(
        'start_index', ppo_defaults.get('start_index', 0)))
    end_index = int(resolved.get(
        'end_index', ppo_defaults.get('end_index', num_cubes * 3)))
    fixed = run_cfg.get('fixed_start_end')
    fixed_goal = (None if fixed is None
                  else np.asarray(fixed, dtype=np.float32))
    permute_start_boxes = bool(
        flags.get('builderbench_permute_start_boxes', True))
    obs_norm_mode = str(resolved.get(
        'obs_norm_mode', flags.get('obs_norm_mode', 'none'))).lower()
    z_scale_multiplier = float(resolved.get(
        'z_scale_multiplier', flags.get('z_scale_multiplier', 3.0)))
    rsnorm_clip = float(resolved.get(
        'rsnorm_clip', flags.get('rsnorm_clip', 10.0)))
    ppo_cleanrl_actor = bool(flags.get('ppo_cleanrl_actor', True))
    ppo_norm_obs = bool(flags.get(
        'ppo_norm_obs', resolved.get('ppo_norm_obs', False)))
    ppo_obs_norm_clip = float(flags.get(
        'ppo_obs_norm_clip',
        resolved.get('ppo_obs_norm_clip', 10.0)))
    cat_select = bool(flags.get(
        'ppo_categorical_select',
        resolved.get('ppo_categorical_select', False)))
    categorical_select_classes = int(num_cubes) if cat_select else None
  else:
    use_pd = False
    pd_duration = 5
    ep_mult = 1.0
    mj_ep_len = scaled_episode_length(num_cubes, ep_mult)
    obs_dim = full_obs_dim
    defaults = ppo_env_defaults_for_env(env_name) or {}
    hidden = tuple(contrastive.ContrastiveConfig().hidden_layer_sizes)
    actor_min_std = float(contrastive.ContrastiveConfig().ppo_actor_min_std)
    start_index = int(defaults.get('start_index', 0))
    end_index = int(defaults.get('end_index', num_cubes * 3))
    fixed_goal = fixed_goal_for_env(env_name)
    obs_space_list = ['xy', 'select']
    pd_obs_dim = pd_policy_state_obs_dim(num_cubes)
    permute_start_boxes = True
    obs_norm_mode = 'none'
    z_scale_multiplier = 3.0
    rsnorm_clip = 10.0
    ppo_cleanrl_actor = True
    ppo_norm_obs = False
    ppo_obs_norm_clip = 10.0
    categorical_select_classes = None

  if use_pd:
    macro_ep_len = mj_ep_len // pd_duration
    filter_policy = (obs_dim == pd_obs_dim)
    if obs_dim not in (pd_obs_dim, full_obs_dim):
      print(f'[bb_video] WARNING: obs_dim={obs_dim} unexpected for PD; '
            f'assuming filter={obs_dim == pd_obs_dim}')
  else:
    macro_ep_len = mj_ep_len
    filter_policy = False

  print(f'[bb_video] training context: use_pd={use_pd} pd_duration={pd_duration} '
        f'filter_policy_obs={filter_policy} obs_dim={obs_dim} '
        f'ep_len={macro_ep_len} hidden={hidden} '
        f'fixed_goal={fixed_goal is not None} '
        f'actor_min_std={actor_min_std} '
        f'start_index={start_index} end_index={end_index} '
        f'obs_space_list={obs_space_list} ep_mult={ep_mult} '
        f'obs_norm_mode={obs_norm_mode}')

  return _TrainCtx(
      use_pd=use_pd,
      pd_duration=pd_duration,
      filter_policy_obs=filter_policy,
      episode_length=int(macro_ep_len),
      obs_dim=int(obs_dim),
      hidden_layer_sizes=hidden,
      fixed_target_goal=fixed_goal,
      actor_min_std=actor_min_std,
      start_index=start_index,
      end_index=end_index,
      obs_space_list=obs_space_list,
      episode_length_multiplier=ep_mult,
      permute_start_boxes=permute_start_boxes,
      obs_norm_mode=obs_norm_mode,
      z_scale_multiplier=z_scale_multiplier,
      rsnorm_clip=rsnorm_clip,
      ppo_cleanrl_actor=ppo_cleanrl_actor,
      ppo_norm_obs=ppo_norm_obs,
      ppo_obs_norm_clip=ppo_obs_norm_clip,
      categorical_select_classes=categorical_select_classes,
  )


def _build_networks(env_name: str, seed: int, ctx: _TrainCtx):
  env_kwargs: Dict[str, Any] = {}
  env_kwargs['builderbench_episode_length_multiplier'] = ctx.episode_length_multiplier
  if ctx.use_pd:
    env_kwargs['builderbench_use_pd'] = True
    env_kwargs['builderbench_pd_duration'] = ctx.pd_duration
    env_kwargs['builderbench_pd_filter_policy_obs'] = ctx.filter_policy_obs
    env_kwargs['obs_space_list'] = ctx.obs_space_list
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
    print(f'[bb_video] WARNING: probe obs_dim={obs_dim} != '
          f'run_config obs_dim={ctx.obs_dim}')
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env

  cfg = contrastive.ContrastiveConfig()
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=int(ctx.obs_dim),
      repr_dim=cfg.repr_dim,
      repr_norm=cfg.repr_norm,
      twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs,
      hidden_layer_sizes=ctx.hidden_layer_sizes,
      actor_min_std=ctx.actor_min_std,
      ppo_cleanrl_actor=ctx.ppo_cleanrl_actor,
      categorical_select_classes=ctx.categorical_select_classes,
  )
  return networks


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
  """Build a single-env batched BuilderBench stack matching training."""
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
  mocap_targets = base._task_mocap_targets

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
  # Expose render helper from the underlying CreativeCube instance.
  env.render_from_info = base.render_from_info  # type: ignore[attr-defined]
  env._mocap_targets_geom = base._mocap_targets_geom  # type: ignore[attr-defined]
  env.model = base._mj_model  # type: ignore[attr-defined]

  return env, base, mocap_targets, int(macro_ep_len)


def _make_policy_fn(
    networks,
    policy_params,
    stochastic: bool,
    *,
    filter_policy_obs: bool,
    num_cubes: int,
    obs_space_list: list[str] = ['xy', 'select'],
    obs_normalizer = None,
    normalize_obs: bool = False,
    obs_dim: int = 0,
    start_index: int = 0,
    end_index: int = -1,
    obs_norm_clip: float = 10.0,
    obs_mean: Optional[np.ndarray] = None,
    obs_var: Optional[np.ndarray] = None,
):
  _norm = bool(normalize_obs)
  _obs_dim = int(obs_dim)
  _si = int(start_index)
  _ei = int(end_index if end_index != -1 else obs_dim)
  _clip = float(obs_norm_clip)
  if _norm:
    if obs_mean is None or obs_var is None:
      raise ValueError('normalize_obs requires obs_mean and obs_var')
    mean_j = jnp.asarray(obs_mean, dtype=jnp.float32)
    var_j = jnp.asarray(obs_var, dtype=jnp.float32)
  else:
    mean_j = jnp.zeros((_obs_dim,), dtype=jnp.float32)
    var_j = jnp.ones((_obs_dim,), dtype=jnp.float32)

  @jax.jit
  def policy(obs, goals, key):
    if filter_policy_obs:
      obs = filter_pd_policy_state_obs(obs, num_cubes, obs_space_list)
    packed = jnp.concatenate([obs, goals], axis=-1)
    if obs_normalizer is not None and getattr(obs_normalizer, 'mode', 'none') != 'none':
      packed = jnp.asarray(
          obs_normalizer.normalize(np.asarray(packed), update_stats=False),
          dtype=jnp.float32)
    elif _norm:
      packed = ppo_learner._normalize_packed_obs(
          packed, mean_j, var_j,
          obs_dim=_obs_dim, start_index=_si, end_index=_ei,
          clip=_clip, enabled=_norm)
    dist = networks.policy_network.apply(policy_params, packed)
    if stochastic:
      action = networks.sample(dist, key)
    else:
      action = dist.mode()
    return action, {}

  return policy


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
  parser.add_argument('--env', default='builderbench_creative_1_task1')
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
    print(f'[bb_video] no checkpoints found at {args.checkpoint!r}')
    return
  print(f'[bb_video] found {len(ckpt_entries)} checkpoint(s)')

  ctx = _load_train_ctx(args.env, args.checkpoint)
  cfg_path = _run_config_path_for_checkpoint(args.checkpoint)
  if cfg_path is not None:
    skip = video_render_skip_reason(cfg_path)
    if skip is not None:
      print(f'[bb_video] skip: {skip}')
      return
  cfg_note = (f'use_pd={ctx.use_pd} pd_duration={ctx.pd_duration} '
              f'filter_policy_obs={ctx.filter_policy_obs}')
  print(f'[bb_video] training context: {cfg_note} '
        f'obs_dim={ctx.obs_dim} ep_len={ctx.episode_length}')

  print('[bb_video] building networks and builderbench env...')
  networks = _build_networks(args.env, seed=args.seed, ctx=ctx)
  video_env, _base, mocap_targets, episode_length = _make_bb_env(env_id, ctx)
  print(f'[bb_video] env_id={env_id}  episode_length={episode_length}')

  key = jax.random.PRNGKey(args.seed)
  multi = len(ckpt_entries) > 1
  out_dir = args.output
  if out_dir.endswith(os.sep) or not out_dir.endswith('.mp4'):
    os.makedirs(out_dir, exist_ok=True)

  from envs.builderbench_obs_norm import BuilderBenchObsNormalizer

  for label, path in ckpt_entries:
    out_path = _resolve_output_path(out_dir, run_tag, label, multi)
    if args.skip_existing and os.path.isfile(out_path):
      print(f'[bb_video] skip existing {out_path}', flush=True)
      continue
    print(f'[bb_video] === {label}  ({path}) ===')
    ckpt = ppo_learner.load_checkpoint(path)
    policy_params = ckpt['policy_params']

    obs_normalizer = BuilderBenchObsNormalizer(
        mode=ctx.obs_norm_mode,
        num_cubes=num_cubes,
        state_obs_dim=ctx.obs_dim,
        goal_dim=num_cubes * 3,
        z_scale_multiplier=ctx.z_scale_multiplier,
        rsnorm_clip=ctx.rsnorm_clip,
    )
    if 'obs_norm_mean' in ckpt:
      obs_normalizer.mean = np.asarray(ckpt['obs_norm_mean'], dtype=np.float64)
      obs_normalizer.var = np.asarray(ckpt['obs_norm_var'], dtype=np.float64)
      obs_normalizer.count = float(ckpt.get('obs_norm_count', 1.0))

    obs_mean = obs_var = None
    if ctx.ppo_norm_obs:
      extra = ckpt.get('extra_state') or {}
      obs_state = extra.get('obs_rms')
      if not isinstance(obs_state, dict):
        raise RuntimeError(
            f'ppo_norm_obs checkpoint missing extra_state.obs_rms: {path}')
      obs_mean = np.asarray(obs_state['mean'], dtype=np.float32)
      obs_var = np.asarray(obs_state['var'], dtype=np.float32)
      print(f'[bb_video]   obs_rms count={obs_state.get("count")} '
            f'mean_abs={float(np.mean(np.abs(obs_mean))):.4f}')

    policy = _make_policy_fn(
        networks,
        policy_params,
        args.stochastic,
        filter_policy_obs=ctx.filter_policy_obs,
        num_cubes=num_cubes,
        obs_space_list=ctx.obs_space_list,
        obs_normalizer=obs_normalizer,
        normalize_obs=ctx.ppo_norm_obs,
        obs_dim=ctx.obs_dim,
        start_index=ctx.start_index,
        end_index=ctx.end_index,
        obs_norm_clip=ctx.ppo_obs_norm_clip,
        obs_mean=obs_mean,
        obs_var=obs_var,
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
    print(f'[bb_video]   frames={len(frames)}')

    _write_video(frames, out_path, args.fps)
    print(f'[bb_video]   wrote {out_path}')


if __name__ == '__main__':
  main()
