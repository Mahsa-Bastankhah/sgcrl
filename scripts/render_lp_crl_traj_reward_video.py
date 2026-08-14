"""Render LP contrastive (Acme TF) traj videos with live φ·ψ reward strip.

Loads ``checkpoints/learner/ckpt-*`` from an ``lp_contrastive.py`` run
(``TrainingState`` via acme TF CheckpointManager), rolls out the policy,
and overlays online ``r = φ(s,a)·ψ(g)`` like
``render_sawyer_crl_traj_reward_video.py``.

Examples:
  python scripts/render_lp_crl_traj_reward_video.py \\
      --run_dir=logs/lp_contrastive_sawyer_bin/contrastive_cpc_sawyer_bin_42 \\
      --env=sawyer_bin --ckpt_stride=2 --allow_no_success \\
      --out_dir=figs/sawyer_bin/frozen_crl_reward_probe/lp_cpc_s42_eo \\
      --tag_prefix=lp_cpc_sawyer_bin_s42_eo
"""
from __future__ import annotations

import argparse
import glob
import importlib.util as _ilu
import os
import re
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from PIL import Image

import contrastive
from contrastive import ContrastiveConfig
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
import env_utils
from ppo_contrastive import fixed_goal_dict

_qs_path = os.path.join(_REPO, 'scripts', 'q_sac_rollout_video.py')
_qs_spec = _ilu.spec_from_file_location('q_sac_rollout_video', _qs_path)
_qs = _ilu.module_from_spec(_qs_spec)
assert _qs_spec.loader is not None
_qs_spec.loader.exec_module(_qs)
_load_training_state = _qs._load_training_state
_enumerate_checkpoints = _qs._enumerate_checkpoints
_load_sweep_config = _qs._load_sweep_config
_get_render_fn = _qs._get_render_fn
_DEFAULT_CAMERA = _qs._DEFAULT_CAMERA

_jp_before = os.environ.get('JAX_PLATFORMS')
_crl_path = os.path.join(_REPO, 'scripts', 'render_frozen_crl_traj_reward_video.py')
_crl_spec = _ilu.spec_from_file_location(
    'render_frozen_crl_traj_reward_video', _crl_path)
_crl = _ilu.module_from_spec(_crl_spec)
assert _crl_spec.loader is not None
_crl_spec.loader.exec_module(_crl)
if _jp_before is None:
  os.environ.pop('JAX_PLATFORMS', None)
else:
  os.environ['JAX_PLATFORMS'] = _jp_before
_render_reward_strip = _crl._render_reward_strip
_compose_frame = _crl._compose_frame
_write_mp4 = _crl._write_mp4

OUT_DIR = 'figs/sawyer_bin/frozen_crl_reward_probe'
SEED = 0
FPS = 15
HOLD_LAST = 8
MAX_TRIES = 1


def _build_networks(env_name: str, seed: int, cfg_dict: dict):
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed,
      fixed_start_end=fixed_goal_dict[env_name])
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env
  # Match ContrastiveConfig / lp_contrastive defaults (repr_norm=False).
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=int(cfg_dict.get('repr_dim', ContrastiveConfig().repr_dim)),
      repr_norm=bool(cfg_dict.get('repr_norm', ContrastiveConfig().repr_norm)),
      twin_q=bool(cfg_dict.get('twin_q', ContrastiveConfig().twin_q)),
      use_image_obs=bool(cfg_dict.get(
          'use_image_obs', ContrastiveConfig().use_image_obs)),
      hidden_layer_sizes=tuple(cfg_dict.get(
          'hidden_layer_sizes', ContrastiveConfig().hidden_layer_sizes)),
  )
  return networks, int(obs_dim)


def _enumerate_ckpts(run_dir: str, only_latest: bool, ckpt_stride: int,
                     max_ckpts: int):
  ckpt_dir = os.path.join(run_dir, 'checkpoints', 'learner')
  entries = _enumerate_checkpoints(ckpt_dir, only_latest=only_latest)
  if ckpt_stride > 1:
    entries = entries[:: int(ckpt_stride)]
  if max_ckpts and len(entries) > int(max_ckpts):
    idxs = [round(i * (len(entries) - 1) / (int(max_ckpts) - 1))
            for i in range(int(max_ckpts))]
    entries = [entries[i] for i in sorted(set(idxs))]
  return entries


def _make_policy_fns(networks):
  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  @jax.jit
  def policy_sample(params, obs, rng):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample(dist, rng)

  return policy_mode, policy_sample


def _rollout_collect(policy_params, gym_env, networks, render, reward_fn,
                     q_params, max_steps, seed, stochastic, mean0, var1):
  policy_mode, policy_sample = _make_policy_fns(networks)
  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  packed_list = []
  action_list = []
  success_list = []
  env_r_list = []
  frames = []
  total_env_r = 0.0
  any_success = False
  rng = jax.random.PRNGKey(seed)
  for _ in range(max_steps):
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = policy_sample(policy_params, obs[None], k)
    else:
      action_j = policy_mode(policy_params, obs[None])
    action = np.asarray(action_j)[0].astype(np.float32)
    packed_list.append(obs.copy())
    action_list.append(action)
    obs_next, r, done, _info = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    r = float(r)
    succ = 1.0 if r > 0.0 else 0.0
    if succ:
      any_success = True
    total_env_r += r
    success_list.append(succ)
    env_r_list.append(r)
    frames.append(render())
    if done:
      break

  packed = np.asarray(packed_list, dtype=np.float32)
  actions = np.asarray(action_list, dtype=np.float32)
  success = np.asarray(success_list, dtype=np.float32)
  rewards = np.asarray(
      reward_fn(q_params, jnp.asarray(packed), jnp.asarray(actions),
                mean0, var1),
      dtype=np.float32)
  return dict(
      frames=frames,
      rewards=rewards,
      success=success,
      env_reward=np.asarray(env_r_list, dtype=np.float32),
      packed=packed,
      actions=actions,
      any_success=any_success,
      total_env_r=total_env_r,
  )


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True,
                 help='LP run dir with checkpoints/learner/ckpt-*.')
  p.add_argument('--env', default='',
                 help='Override env; else sweep_config / run-dir inference.')
  p.add_argument('--out_dir', default=OUT_DIR)
  p.add_argument('--tag', default='')
  p.add_argument('--tag_prefix', default='lp_crl_online')
  p.add_argument('--title', default='')
  p.add_argument('--allow_no_success', action='store_true')
  p.add_argument('--max_tries', type=int, default=MAX_TRIES)
  p.add_argument('--seed', type=int, default=SEED)
  p.add_argument('--fps', type=int, default=FPS)
  p.add_argument('--skip_existing', action='store_true')
  p.add_argument('--stochastic', action='store_true')
  p.add_argument('--max_steps', type=int, default=-1)
  p.add_argument('--width', type=int, default=640)
  p.add_argument('--height', type=int, default=480)
  p.add_argument('--camera', default=None)
  p.add_argument('--rotate', type=int, default=180)
  p.add_argument('--only_latest', action='store_true')
  p.add_argument('--max_ckpts', type=int, default=0,
                 help='If >0, evenly subsample this many after stride.')
  p.add_argument('--ckpt_stride', type=int, default=1,
                 help='Keep every Nth learner checkpoint (2 = every other).')
  return p.parse_args()


def _render_one(args, *, label, ckpt_prefix, networks, gym_env, render,
                reward_fn, mean0, var1, max_steps, out_dir):
  tag = args.tag if (args.tag and args.only_latest) else (
      f'{args.tag_prefix}_{label}')
  out_mp4 = os.path.join(out_dir, f'{tag}.mp4')
  still_path = os.path.join(out_dir, f'{tag}_still.png')
  csv_path = os.path.join(out_dir, f'{tag}.csv')
  if args.skip_existing and os.path.isfile(out_mp4):
    print(f'[vid] skip existing {out_mp4}', flush=True)
    return

  print(f'[vid] === {label} ({ckpt_prefix}) ===', flush=True)
  state = _load_training_state(ckpt_prefix)
  q_params = state.q_params
  policy_params = state.policy_params
  print(f'[vid] max_steps={max_steps} reward_src=q_params(online) φ·ψ only',
        flush=True)

  best = None
  for attempt in range(int(args.max_tries)):
    roll = _rollout_collect(
        policy_params, gym_env, networks, render, reward_fn, q_params,
        max_steps=max_steps, seed=int(args.seed) + attempt,
        stochastic=bool(args.stochastic), mean0=mean0, var1=var1)
    print(f'[vid] try={attempt} success={roll["any_success"]} '
          f'rew_sum={roll["rewards"].sum():.2f} env_r={roll["total_env_r"]:.1f} '
          f'T={len(roll["rewards"])}', flush=True)
    if roll['any_success']:
      best = roll
      break
    if best is None or roll['rewards'].sum() > best['rewards'].sum():
      best = roll

  if best is None:
    raise RuntimeError('no trajectory collected')
  if (not best['any_success']) and (not args.allow_no_success):
    raise RuntimeError('no successful trajectory found '
                       '(pass --allow_no_success to keep best return)')

  rewards = best['rewards']
  success = best['success']
  frames_rgb = best['frames']
  T = len(rewards)
  first_succ = (int(np.argmax(success >= 0.5))
                if np.any(success >= 0.5) else -1)
  print(f'[vid] using first_success_t={first_succ} '
        f'rew_sum={rewards.sum():.3f}', flush=True)

  np.savetxt(
      csv_path,
      np.stack([
          np.arange(T), rewards, success, best['env_reward'],
      ], axis=1),
      delimiter=',',
      header='t,reward_online,success,env_reward',
      comments='')

  width = int(frames_rgb[0].shape[1])
  if width % 2:
    width -= 1
  if args.title:
    title = f'{args.title}  ·  {tag}'
  else:
    title = rf'online CRL  $r=\varphi(s,a)\cdot\psi(g)$  ·  {tag}'
  ylabel = r'$r=\varphi(s,a)\cdot\psi(g)$'

  print('[vid] composing reward overlay...', flush=True)
  frames = []
  for t in range(T):
    strip = _render_reward_strip(
        rewards, success, t, width=width, height=220,
        title=title, ylabel=ylabel, ylim=None)
    composed = _compose_frame(
        strip, frames_rgb[t], float(rewards[t]), bool(success[t] >= 0.5))
    h, w = composed.shape[:2]
    if h % 2 or w % 2:
      composed = composed[: h - (h % 2), : w - (w % 2)]
    frames.append(composed)
    if (t + 1) % 25 == 0 or t == T - 1:
      print(f'[vid]   framed {t + 1}/{T}', flush=True)
  frames.extend([frames[-1]] * HOLD_LAST)
  _write_mp4(frames, out_mp4, args.fps)
  still_t = min(first_succ + 2, T - 1) if first_succ >= 0 else T // 2
  Image.fromarray(frames[still_t]).save(still_path)
  print(f'[vid] wrote {out_mp4} ({os.path.getsize(out_mp4) / 1e6:.2f} MB)',
        flush=True)


def main():
  args = _parse_args()
  run_dir = os.path.abspath(args.run_dir)
  if not os.path.isdir(run_dir):
    raise FileNotFoundError(f'run_dir not found: {run_dir}')
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)

  cfg_dict = _load_sweep_config(run_dir)
  env_name = args.env or cfg_dict.get('env_name')
  if not env_name:
    raise SystemExit('pass --env (could not infer from run_dir)')
  if env_name not in fixed_goal_dict:
    raise SystemExit(f'unsupported env={env_name!r}')

  entries = _enumerate_ckpts(
      run_dir, only_latest=bool(args.only_latest),
      ckpt_stride=int(args.ckpt_stride), max_ckpts=int(args.max_ckpts))
  if not entries:
    raise FileNotFoundError(
        f'no checkpoints under {run_dir}/checkpoints/learner')
  print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[vid] run_dir={run_dir}')
  print(f'[vid] env={env_name} repr_norm={cfg_dict.get("repr_norm")} '
        f'twin_q={cfg_dict.get("twin_q")} '
        f'hidden={cfg_dict.get("hidden_layer_sizes")}')
  print(f'[vid] {len(entries)} checkpoint(s) stride={args.ckpt_stride} '
        f'→ {out_dir}')

  networks, obs_dim = _build_networks(env_name, seed=args.seed,
                                      cfg_dict=cfg_dict)
  gym_env, _, env_max_steps = env_utils.load(
      env_name, fixed_start_end=fixed_goal_dict[env_name], seed=args.seed)
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  camera = args.camera or _DEFAULT_CAMERA.get(env_name, 'corner')
  print(f'[vid] max_steps={max_steps} camera={camera} rotate={args.rotate} '
        f'obs_dim={obs_dim}', flush=True)
  render = _get_render_fn(gym_env, args.width, args.height, camera,
                          rotate_deg=args.rotate)

  cfg = ContrastiveConfig()
  cfg.obs_dim = int(obs_dim)
  cfg.start_index = 0
  cfg.end_index = -1
  cfg.ppo_norm_obs = False
  cfg.ppo_crl_hit_bonus = ''
  reward_fn = ppo_learner.make_reward_fn(networks, cfg)
  mean0 = jnp.zeros((obs_dim,), dtype=jnp.float32)
  var1 = jnp.ones((obs_dim,), dtype=jnp.float32)

  for label, prefix in entries:
    _render_one(
        args, label=label, ckpt_prefix=prefix, networks=networks,
        gym_env=gym_env, render=render, reward_fn=reward_fn,
        mean0=mean0, var1=var1, max_steps=max_steps, out_dir=out_dir)


if __name__ == '__main__':
  main()
