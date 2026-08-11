"""Render BuilderBench traj video with live online TD3 log-Q reward strip.

Stacked frame: reward timeline on top + MuJoCo below. Reward is
``r = log((1-γ) · max(Q1(s,a,g), ε))`` from online ``q_params`` (not the
``ppo_td3_reward_tau`` EMA), matching the NF online logp probe layout.

Examples:
  python scripts/render_td3_traj_reward_video.py \\
      --checkpoint_dir=logs/.../checkpoints \\
      --env=builderbench_creative_5_task2 \\
      --tag_prefix=c5t2_td3_normobs_extrew1_s0_online \\
      --allow_no_success --skip_existing
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BB = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import sgcrl_jax_acme_compat  # noqa: F401

import importlib.util as _ilu

import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from PIL import Image

import contrastive
from contrastive import ppo_learner
from contrastive import td3_density as _td3
from contrastive import utils as contrastive_utils
from envs.builderbench_utils import (
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

_bbv_path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_rollout_video.py')
_bbv_spec = _ilu.spec_from_file_location('ppo_builderbench_rollout_video', _bbv_path)
_bbv = _ilu.module_from_spec(_bbv_spec)
assert _bbv_spec.loader is not None
_bbv_spec.loader.exec_module(_bbv)
_load_train_ctx = _bbv._load_train_ctx
_make_bb_env = _bbv._make_bb_env
_make_policy_fn = _bbv._make_policy_fn
_maybe_fix_target = _bbv._maybe_fix_target
force_video_nopermute_norand = _bbv.force_video_nopermute_norand
VIDEO_FIXED_START_X = _bbv.VIDEO_FIXED_START_X

_crl_path = os.path.join(_REPO, 'scripts', 'render_frozen_crl_traj_reward_video.py')
_crl_spec = _ilu.spec_from_file_location('render_frozen_crl_traj_reward_video', _crl_path)
_crl = _ilu.module_from_spec(_crl_spec)
assert _crl_spec.loader is not None
_crl_spec.loader.exec_module(_crl)
_render_reward_strip = _crl._render_reward_strip
_compose_frame = _crl._compose_frame
_write_mp4 = _crl._write_mp4
_compile_rollout_and_states = _crl._compile_rollout_and_states

import argparse

DEFAULT_ENV = 'builderbench_creative_5_task2'
OUT_DIR = 'figs/builderbench/td3_logq_reward_probe'
SEED = 0
MAX_TRIES = 1
FPS = 8
HOLD_LAST = 8


def _run_dir_from_ckpt(ckpt_path: str) -> str:
  ckpt_dir = os.path.dirname(os.path.realpath(ckpt_path))
  return os.path.dirname(ckpt_dir)


def _load_td3_flags(run_dir: str) -> dict:
  cfg_path = os.path.join(run_dir, 'run_config.json')
  with open(cfg_path, 'r', encoding='utf-8') as fh:
    payload = json.load(fh)
  resolved = payload.get('resolved_config', {})
  flags = payload.get('flags', {})

  def _get(key, default):
    if key in flags and flags[key] is not None:
      return flags[key]
    if key in resolved and resolved[key] is not None:
      return resolved[key]
    return default

  def _sizes(val, default=(256,) * 6):
    if val is None:
      return tuple(default)
    if isinstance(val, str):
      parts = [p.strip() for p in val.replace('(', '').replace(')', '').split(',')
               if p.strip()]
      return tuple(int(p) for p in parts)
    return tuple(int(x) for x in val)

  # Prefer resolved values: flags often store CLI sentinels (e.g. discount=-1).
  hidden = resolved.get('hidden_layer_sizes', flags.get('hidden_layer_sizes'))
  discount = resolved.get('discount', flags.get('discount', 0.99))
  discount = float(discount)
  if discount < 0.0 or discount >= 1.0:
    discount = float(contrastive.ContrastiveConfig().discount)

  return {
      'discount': discount,
      'log_reward': bool(_get('ppo_td3_log_reward', True)),
      'bilinear': bool(_get('ppo_td3_bilinear', False)),
      'repr_dim': int(_get('repr_dim', contrastive.ContrastiveConfig().repr_dim)),
      'repr_norm': bool(_get('repr_norm', False)),
      'hidden_layer_sizes': _sizes(hidden),
  }


def _build_networks(env_name: str, seed: int, ctx, td3_flags: dict):
  env_kwargs = {}
  if ctx.use_pd:
    env_kwargs['builderbench_use_pd'] = True
    env_kwargs['builderbench_pd_duration'] = ctx.pd_duration
    env_kwargs['builderbench_pd_filter_policy_obs'] = ctx.filter_policy_obs
  env_kwargs['builderbench_permute_start_boxes'] = ctx.permute_start_boxes
  if getattr(ctx, 'fixed_start_x', None) is not None:
    env_kwargs['builderbench_fixed_start_x'] = float(ctx.fixed_start_x)
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, ctx.start_index, ctx.end_index, seed=seed,
      fixed_start_end=ctx.fixed_target_goal, **env_kwargs)
  env_spec = specs.make_environment_spec(probe_env)
  act_dim = int(np.prod(env_spec.actions.shape))
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
      categorical_select_classes=ctx.categorical_select_classes,
      categorical_select_waypoint=bool(getattr(
          ctx, 'categorical_select_waypoint', False)),
  )
  goal_dim = int(ctx.end_index - ctx.start_index) if ctx.end_index != -1 else int(ctx.obs_dim)
  print(f'[vid] TD3 arch: hidden={td3_flags["hidden_layer_sizes"]} '
        f'bilinear={td3_flags["bilinear"]} log_reward={td3_flags["log_reward"]} '
        f'discount={td3_flags["discount"]} goal_dim={goal_dim} '
        f'act_dim={act_dim} norm_obs={ctx.ppo_norm_obs}', flush=True)
  td3_nets = _td3.make_td3_density_networks(
      obs_dim=int(ctx.obs_dim),
      act_dim=act_dim,
      goal_dim=goal_dim,
      hidden_layer_sizes=td3_flags['hidden_layer_sizes'],
      repr_dim=td3_flags['repr_dim'],
      bilinear=td3_flags['bilinear'],
      repr_norm=td3_flags['repr_norm'] if td3_flags['bilinear'] else False,
  )
  return networks, td3_nets, goal_dim


def _obs_rms_from_ckpt(ckpt) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
  extra = ckpt.get('extra_state') or {}
  obs_state = extra.get('obs_rms') if isinstance(extra, dict) else None
  if not isinstance(obs_state, dict):
    return None, None
  mean = np.asarray(obs_state['mean'], dtype=np.float32)
  var = np.asarray(obs_state['var'], dtype=np.float32)
  return mean, var


def _enumerate_ckpts(checkpoint: str, checkpoint_dir: str):
  if checkpoint_dir:
    files = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'ckpt_iter_*.pkl')),
        key=lambda p: int(re.search(r'ckpt_iter_(\d+)\.pkl$', p).group(1)))
    out = []
    for p in files:
      it = int(re.search(r'ckpt_iter_(\d+)\.pkl$', p).group(1))
      out.append((f'iter_{it:07d}', p))
    return out
  base = os.path.splitext(os.path.basename(checkpoint))[0]
  label = base[len('ckpt_'):] if base.startswith('ckpt_') else base
  return [(label, checkpoint)]


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', default='')
  p.add_argument('--checkpoint_dir', default='')
  p.add_argument('--env', default=DEFAULT_ENV)
  p.add_argument('--out_dir', default=OUT_DIR)
  p.add_argument('--tag', default='')
  p.add_argument('--tag_prefix', default='c5t2_td3_online')
  p.add_argument('--title', default='')
  p.add_argument('--allow_no_success', action='store_true')
  p.add_argument('--max_tries', type=int, default=MAX_TRIES)
  p.add_argument('--seed', type=int, default=SEED)
  p.add_argument('--fps', type=int, default=FPS)
  p.add_argument('--skip_existing', action='store_true')
  p.add_argument('--use_ema', action='store_true',
                 help='Use q_params_ema (ppo_td3_reward_tau) instead of online')
  p.add_argument('--match_run_init', action='store_true',
                 help='Use permute/fixed_start_x from run_config (default: '
                      'force nopermute + fixed_start_x=0.1)')
  p.add_argument('--fixed_start_x', type=float, default=VIDEO_FIXED_START_X,
                 help='Start-box x when forcing norand')
  return p.parse_args()


def _render_one(args, *, label, ckpt_path, ctx, networks, td3_nets, env,
                mocap_targets, ep_len, num_cubes, td3_flags, reward_fn, out_dir):
  tag = args.tag if (args.tag and not args.checkpoint_dir) else (
      f'{args.tag_prefix}_{label}')
  out_mp4 = os.path.join(out_dir, f'{tag}.mp4')
  still_path = os.path.join(out_dir, f'{tag}_still.png')
  csv_path = os.path.join(out_dir, f'{tag}.csv')
  if args.skip_existing and os.path.isfile(out_mp4):
    print(f'[vid] skip existing {out_mp4}', flush=True)
    return

  print(f'[vid] === {label} ({ckpt_path}) ===', flush=True)
  ckpt = ppo_learner.load_checkpoint(ckpt_path)
  if args.use_ema and ckpt.get('q_params_ema') is not None:
    q_params = ckpt['q_params_ema']
    q_src = 'q_params_ema'
  else:
    q_params = ckpt['q_params']
    q_src = 'q_params(online)'
  iteration = int(ckpt.get('iteration') or 0)

  obs_mean, obs_var = None, None
  if ctx.ppo_norm_obs:
    obs_mean, obs_var = _obs_rms_from_ckpt(ckpt)
    if obs_mean is None:
      raise RuntimeError(
          f'ppo_norm_obs run missing extra_state.obs_rms in {ckpt_path}')
    print(f'[vid] obs_rms loaded mean_shape={obs_mean.shape}', flush=True)

  print(f'[vid] policy_iter={iteration} ep_len={ep_len} reward_src={q_src}',
        flush=True)

  policy = _make_policy_fn(
      networks, ckpt['policy_params'], stochastic=False,
      filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes,
      normalize_obs=bool(ctx.ppo_norm_obs),
      obs_dim=ctx.obs_dim,
      start_index=ctx.start_index, end_index=ctx.end_index,
      obs_norm_clip=float(ctx.ppo_obs_norm_clip),
      obs_mean=obs_mean, obs_var=obs_var)
  run = _compile_rollout_and_states(
      policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
      ctx.filter_policy_obs)

  key = jax.random.PRNGKey(args.seed)
  key, warm_key = jax.random.split(key)
  print('[vid] warming compile...', flush=True)
  _ = jax.block_until_ready(run(warm_key))
  print('[vid] compile done', flush=True)

  mean_j = (jnp.asarray(obs_mean) if obs_mean is not None
            else jnp.zeros((ctx.obs_dim,), dtype=jnp.float32))
  var_j = (jnp.asarray(obs_var) if obs_var is not None
           else jnp.ones((ctx.obs_dim,), dtype=jnp.float32))

  best = None
  for attempt in range(int(args.max_tries)):
    key, roll_key = jax.random.split(key)
    traj, states = run(roll_key)
    packed = np.asarray(traj['packed'], dtype=np.float32)
    actions = np.asarray(traj['action'], dtype=np.float32)
    succ = np.asarray(traj['success'], dtype=np.float32)
    rewards = np.asarray(
        reward_fn(q_params, jnp.asarray(packed), jnp.asarray(actions),
                  mean_j, var_j), dtype=np.float32)
    reached = bool(np.any(succ >= 0.5))
    print(f'[vid] try={attempt} success={reached} '
          f'rew_sum={rewards.sum():.2f} first_succ='
          f'{int(np.argmax(succ >= 0.5)) if reached else -1}', flush=True)
    cur = dict(rewards=rewards, success=succ, states=states, attempt=attempt)
    if reached:
      best = cur
      break
    if best is None or rewards.sum() > best['rewards'].sum():
      best = cur

  if best is None:
    raise RuntimeError('no trajectory collected')
  if (not np.any(best['success'] >= 0.5)) and (not args.allow_no_success):
    raise RuntimeError('no successful trajectory found '
                       '(pass --allow_no_success)')

  rewards = best['rewards']
  success = best['success']
  states = best['states']
  T = len(rewards)
  first_succ = (int(np.argmax(success >= 0.5))
                if np.any(success >= 0.5) else -1)
  print(f'[vid] using attempt={best["attempt"]} first_success_t={first_succ} '
        f'rew_sum={rewards.sum():.3f}', flush=True)

  np.savetxt(
      csv_path,
      np.stack([np.arange(T), rewards, success,
                np.full(T, np.nan, dtype=np.float32)], axis=1),
      delimiter=',',
      header='t,reward_logq_online,success,dist',
      comments='')

  print('[vid] rendering frames...', flush=True)
  renders = []
  for i in range(T):
    renders.append(np.asarray(env.render_from_info(
        np.asarray(states.data.qpos[i][0]),
        np.asarray(states.data.qvel[i][0]),
        np.asarray(states.info['target_mocap_pos'][i][0]),
        np.asarray(states.info['target_mocap_quat'][i][0]),
    )))
  width = int(renders[0].shape[1])
  if width % 2:
    width -= 1

  if args.title:
    title = args.title
  elif td3_flags['log_reward']:
    title = rf'online TD3  $r=\log((1-\gamma)Q_1(s,a,g))$  ·  {tag}'
  else:
    title = rf'online TD3  $r=Q_1(s,a,g)$  ·  {tag}'
  ylabel = r'$\log Q$' if td3_flags['log_reward'] else r'$Q$'

  print('[vid] composing reward overlay...', flush=True)
  frames = []
  for t in range(T):
    strip = _render_reward_strip(
        rewards, success, t, width=width, height=220,
        title=title, ylabel=ylabel, line_color='#ff8a4c', ylim=None)
    composed = _compose_frame(
        strip, renders[t], float(rewards[t]), bool(success[t] >= 0.5))
    h, w = composed.shape[:2]
    if h % 2 or w % 2:
      composed = composed[: h - (h % 2), : w - (w % 2)]
    frames.append(composed)
    if (t + 1) % 10 == 0 or t == T - 1:
      print(f'[vid]   framed {t + 1}/{T}', flush=True)
  frames.extend([frames[-1]] * HOLD_LAST)
  _write_mp4(frames, out_mp4, args.fps)
  still_t = min(first_succ + 2, T - 1) if first_succ >= 0 else T // 2
  Image.fromarray(frames[still_t]).save(still_path)
  print(f'[vid] wrote {out_mp4} ({os.path.getsize(out_mp4) / 1e6:.2f} MB)',
        flush=True)


def main():
  args = _parse_args()
  if not args.checkpoint and not args.checkpoint_dir:
    raise SystemExit('need --checkpoint or --checkpoint_dir')
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)
  entries = _enumerate_ckpts(args.checkpoint, args.checkpoint_dir)
  if not entries:
    raise FileNotFoundError('no checkpoints found')
  print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[vid] {len(entries)} checkpoint(s) → {out_dir}')

  first_path = entries[0][1]
  env_name = args.env
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, _ = parse_bb_env_id(env_id)
  ctx = _load_train_ctx(env_name, first_path)
  if not args.match_run_init:
    force_video_nopermute_norand(ctx, fixed_start_x=float(args.fixed_start_x))
    print(f'[vid] forcing nopermute + fixed_start_x={ctx.fixed_start_x}',
          flush=True)
  run_dir = _run_dir_from_ckpt(first_path)
  td3_flags = _load_td3_flags(run_dir)
  networks, td3_nets, _goal_dim = _build_networks(
      env_name, args.seed, ctx, td3_flags)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)

  reward_fn = _td3.make_td3_reward_fn(
      td3_nets,
      obs_dim=int(ctx.obs_dim),
      discount=float(td3_flags['discount']),
      log_reward=bool(td3_flags['log_reward']),
      start_index=int(ctx.start_index),
      end_index=int(ctx.end_index),
      normalize_obs=bool(ctx.ppo_norm_obs),
      obs_norm_clip=float(ctx.ppo_obs_norm_clip),
  )

  for label, path in entries:
    _render_one(
        args, label=label, ckpt_path=path, ctx=ctx, networks=networks,
        td3_nets=td3_nets, env=env, mocap_targets=mocap_targets, ep_len=ep_len,
        num_cubes=num_cubes, td3_flags=td3_flags, reward_fn=reward_fn,
        out_dir=out_dir)


if __name__ == '__main__':
  main()
