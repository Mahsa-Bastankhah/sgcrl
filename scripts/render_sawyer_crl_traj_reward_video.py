"""Render Sawyer traj video with live online φ·ψ reward strip.

Like ``scripts/render_frozen_crl_traj_reward_video.py`` but for MetaWorld /
Sawyer (mujoco_py offscreen), not BuilderBench.

By default the strip is the CRL representation reward r = φ(s,a)·ψ(g)
only (no hit-indicator, no external success bonus), matching runs that
did not enable ``ppo_crl_hit_bonus``.

Examples:
  python scripts/render_sawyer_crl_traj_reward_video.py \\
      --checkpoint_dir=logs/.../ppo_sawyer_bin_0/checkpoints \\
      --env=sawyer_bin --tag_prefix=bin_norand_s0_online \\
      --allow_no_success --max_ckpts=24
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import sgcrl_jax_acme_compat  # noqa: F401

import argparse
import importlib.util as _ilu

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

_roll_path = os.path.join(_REPO, 'scripts', 'ppo_rollout_video.py')
_roll_spec = _ilu.spec_from_file_location('ppo_rollout_video', _roll_path)
_roll = _ilu.module_from_spec(_roll_spec)
assert _roll_spec.loader is not None
_roll_spec.loader.exec_module(_roll)
_get_render_fn = _roll._get_render_fn
_DEFAULT_CAMERA = _roll._DEFAULT_CAMERA

# Reuse strip/compose helpers from the BuilderBench probe script without
# letting that module's setdefault(JAX_PLATFORMS=cpu) override the job env.
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


def _run_config_path_for_checkpoint(checkpoint_path: str):
  abspath = os.path.abspath(checkpoint_path)
  if os.path.isfile(abspath):
    ckpt_dir = os.path.dirname(abspath)
  else:
    ckpt_dir = abspath
  # .../run_dir/checkpoints[/file] → run_dir/run_config.json
  if os.path.basename(ckpt_dir) == 'checkpoints':
    run_dir = os.path.dirname(ckpt_dir)
  else:
    run_dir = os.path.dirname(ckpt_dir)
  path = os.path.join(run_dir, 'run_config.json')
  return path if os.path.isfile(path) else None


def _load_run_settings(ckpt_path: str, env_name: str):
  """Read network/env knobs from run_config next to the checkpoint."""
  cfg_path = _run_config_path_for_checkpoint(ckpt_path)
  actor_min_std = float(ContrastiveConfig().ppo_actor_min_std)
  hidden = tuple(ContrastiveConfig().hidden_layer_sizes)
  randomize_init = True
  hit_bonus = ''
  if cfg_path:
    with open(cfg_path, 'r', encoding='utf-8') as fh:
      run_cfg = json.load(fh)
    flags = run_cfg.get('flags', {}) or {}
    resolved = run_cfg.get('resolved_config', {}) or {}
    actor_min_std = float(flags.get(
        'ppo_actor_min_std',
        resolved.get('ppo_actor_min_std', actor_min_std)))
    hl = flags.get('hidden_layer_sizes',
                   resolved.get('hidden_layer_sizes', None))
    if isinstance(hl, str) and hl.strip():
      hidden = tuple(int(x) for x in hl.split(',') if x.strip())
    elif isinstance(hl, (list, tuple)) and hl:
      hidden = tuple(int(x) for x in hl)
    if 'sawyer_randomize_init' in flags:
      randomize_init = bool(flags['sawyer_randomize_init'])
    hit_bonus = str(flags.get(
        'ppo_crl_hit_bonus',
        resolved.get('ppo_crl_hit_bonus', '')) or '').strip().lower()
    print(f'[vid] run_config={cfg_path}', flush=True)
  else:
    print('[vid] no run_config.json; using ContrastiveConfig defaults',
          flush=True)
  fixed = fixed_goal_dict[env_name]
  print(f'[vid] actor_min_std={actor_min_std} hidden={hidden} '
        f'randomize_init={randomize_init} hit_bonus={hit_bonus!r}',
        flush=True)
  return dict(
      actor_min_std=actor_min_std,
      hidden_layer_sizes=hidden,
      randomize_init=randomize_init,
      hit_bonus=hit_bonus,
      fixed_goal=fixed,
      cfg_path=cfg_path,
  )


def _build_networks(env_name: str, seed: int, settings: dict):
  env_kwargs = {}
  if env_name in ('sawyer_bin', 'sawyer_peg'):
    env_kwargs['randomize_init'] = bool(settings['randomize_init'])
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed,
      fixed_start_end=settings['fixed_goal'], **env_kwargs)
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=ContrastiveConfig().repr_dim,
      repr_norm=ContrastiveConfig().repr_norm,
      twin_q=ContrastiveConfig().twin_q,
      use_image_obs=ContrastiveConfig().use_image_obs,
      hidden_layer_sizes=settings['hidden_layer_sizes'],
      actor_min_std=float(settings['actor_min_std']),
  )
  return networks, int(obs_dim)


def _enumerate_ckpts(checkpoint: str, checkpoint_dir: str,
                     max_ckpts: int = 0, ckpt_stride: int = 1):
  if checkpoint_dir:
    files = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'ckpt_iter_*.pkl')),
        key=lambda p: int(re.search(r'ckpt_iter_(\d+)\.pkl$', p).group(1)))
  else:
    files = [checkpoint]
  if ckpt_stride > 1:
    files = files[:: int(ckpt_stride)]
  if max_ckpts and len(files) > int(max_ckpts):
    idxs = [round(i * (len(files) - 1) / (int(max_ckpts) - 1))
            for i in range(int(max_ckpts))]
    files = [files[i] for i in sorted(set(idxs))]
  out = []
  for p in files:
    base = os.path.splitext(os.path.basename(p))[0]
    if base.startswith('ckpt_'):
      label = base[len('ckpt_'):]
      m = re.search(r'iter_(\d+)$', label)
      if m:
        label = f'iter_{int(m.group(1)):07d}'
    else:
      label = base
    out.append((label, p))
  return out


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
                     q_params, obs_dim, max_steps, seed, stochastic,
                     mean0, var1):
  del obs_dim  # packed obs already includes goal; mean0 length is state dim
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
    frames.append(render())  # post-action frame aligned with reward at t
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


def _make_psi_psi_reward_fn(networks, obs_dim: int):
  """r = ψ(s)·ψ(g): encode current state and goal both through the g-encoder."""
  obs_dim = int(obs_dim)

  @jax.jit
  def reward_fn(q_params, obs, action, obs_mean, obs_var):
    del obs_mean, obs_var  # no obs-norm in this probe; action unused by ψ
    s = obs[:, :obs_dim]
    # Goal slot ← state so g_encoder yields ψ(s).
    obs_ss = jnp.concatenate([s, s], axis=-1)
    _, _, psi_g = networks.q_network.apply(q_params, obs, action)
    _, _, psi_s = networks.q_network.apply(q_params, obs_ss, action)
    return jnp.sum(psi_s * psi_g, axis=-1)

  return reward_fn


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', default='')
  p.add_argument('--checkpoint_dir', default='',
                 help='If set, render ckpt_iter_*.pkl in this dir.')
  p.add_argument('--env', default='sawyer_bin',
                 choices=sorted(fixed_goal_dict.keys()))
  p.add_argument('--out_dir', default=OUT_DIR)
  p.add_argument('--tag', default='')
  p.add_argument('--tag_prefix', default='sawyer_bin_online')
  p.add_argument('--title', default='')
  p.add_argument('--score_mode', default='phi_psi',
                 choices=('phi_psi', 'psi_psi'),
                 help='Top-strip score: φ(s,a)·ψ(g) (default) or ψ(s)·ψ(g).')
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
  p.add_argument('--max_ckpts', type=int, default=0,
                 help='If >0, evenly subsample this many milestones.')
  p.add_argument('--ckpt_stride', type=int, default=1,
                 help='Keep every Nth checkpoint before max_ckpts.')
  return p.parse_args()


def _render_one(args, *, label, ckpt_path, networks, gym_env, render,
                reward_fn, mean0, var1, max_steps, out_dir):
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
  q_params = ckpt['q_params']
  policy_params = ckpt['policy_params']
  score_mode = str(getattr(args, 'score_mode', 'phi_psi') or 'phi_psi')
  score_name = (r'ψ(s)·ψ(g)' if score_mode == 'psi_psi'
                else r'φ(s,a)·ψ(g)')
  print(f'[vid] policy_iter={ckpt.get("iteration")} max_steps={max_steps} '
        f'reward_src=q_params(online) {score_name}', flush=True)

  best = None
  for attempt in range(int(args.max_tries)):
    roll = _rollout_collect(
        policy_params, gym_env, networks, render, reward_fn, q_params,
        obs_dim=int(mean0.shape[0]), max_steps=max_steps,
        seed=int(args.seed) + attempt, stochastic=bool(args.stochastic),
        mean0=mean0, var1=var1)
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
  score_mode = str(getattr(args, 'score_mode', 'phi_psi') or 'phi_psi')
  if args.title:
    title = f'{args.title}  ·  {tag}'
    ylabel = r'$r$'
  elif score_mode == 'psi_psi':
    title = rf'online CRL  $r=\psi(s)\cdot\psi(g)$  ·  {tag}'
    ylabel = r'$r=\psi(s)\cdot\psi(g)$'
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
  if not args.checkpoint and not args.checkpoint_dir:
    raise SystemExit('pass --checkpoint or --checkpoint_dir')
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)
  entries = _enumerate_ckpts(
      args.checkpoint, args.checkpoint_dir,
      max_ckpts=int(args.max_ckpts), ckpt_stride=int(args.ckpt_stride))
  if not entries:
    raise FileNotFoundError('no checkpoints found')
  print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[vid] {len(entries)} checkpoint(s) → {out_dir}')

  first_path = entries[0][1]
  settings = _load_run_settings(first_path, args.env)
  if settings['hit_bonus']:
    print(f'[vid] run_config hit_bonus={settings["hit_bonus"]!r}; '
          f'plotting φ·ψ only (no indicator)', flush=True)
  settings['hit_bonus'] = ''

  networks, obs_dim = _build_networks(args.env, seed=args.seed,
                                      settings=settings)
  env_kwargs = {}
  if args.env in ('sawyer_bin', 'sawyer_peg'):
    env_kwargs['randomize_init'] = bool(settings['randomize_init'])
  gym_env, _, env_max_steps = env_utils.load(
      args.env, fixed_start_end=settings['fixed_goal'], seed=args.seed,
      **env_kwargs)
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  camera = args.camera or _DEFAULT_CAMERA.get(args.env, 'corner')
  print(f'[vid] env={args.env} max_steps={max_steps} camera={camera} '
        f'rotate={args.rotate} obs_dim={obs_dim}', flush=True)
  render = _get_render_fn(gym_env, args.width, args.height, camera,
                          rotate_deg=args.rotate)

  cfg = ContrastiveConfig()
  cfg.obs_dim = int(obs_dim)
  cfg.start_index = 0
  cfg.end_index = -1
  cfg.ppo_norm_obs = False
  cfg.ppo_crl_hit_bonus = ''  # score only (no hit indicator)
  score_mode = str(args.score_mode or 'phi_psi')
  if score_mode == 'psi_psi':
    reward_fn = _make_psi_psi_reward_fn(networks, obs_dim)
    print('[vid] score_mode=psi_psi → r=ψ(s)·ψ(g)', flush=True)
  else:
    reward_fn = ppo_learner.make_reward_fn(networks, cfg)
    print('[vid] score_mode=phi_psi → r=φ(s,a)·ψ(g)', flush=True)
  mean0 = jnp.zeros((obs_dim,), dtype=jnp.float32)
  var1 = jnp.ones((obs_dim,), dtype=jnp.float32)

  for label, path in entries:
    _render_one(
        args, label=label, ckpt_path=path, networks=networks,
        gym_env=gym_env, render=render, reward_fn=reward_fn,
        mean0=mean0, var1=var1, max_steps=max_steps, out_dir=out_dir)


if __name__ == '__main__':
  main()
