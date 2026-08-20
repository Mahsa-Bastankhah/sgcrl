"""Render Sawyer traj video with live online NF log-p reward strips.

Stacked frame: raw log p_NF, optional r/σ, optional ‖∇_s log p‖ / ‖∇_a log p‖,
plus MuJoCo below. Reward uses online ``q_params`` (not the tau-weighted EMA).

Examples:
  python scripts/render_sawyer_nf_traj_reward_video.py \\
      --checkpoint_dir=logs/.../ppo_sawyer_peg_0/checkpoints \\
      --env=sawyer_peg --tag_prefix=peg_nft_s0_online \\
      --allow_no_success --min_iter=1 --normalize_reward --show_logp_grad
  python scripts/render_sawyer_nf_traj_reward_video.py \\
      --checkpoint_dir=logs/.../ppo_sawyer_peg_0/checkpoints \\
      --env=sawyer_peg --tag_prefix=peg_nft_tanhr_s0 \\
      --allow_no_success --max_ckpts=8 --min_iter=1 --show_tanh
"""
from __future__ import annotations

import argparse
import csv
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

import importlib.util as _ilu

import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from PIL import Image

import contrastive
from contrastive import nf_density as _nf
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
import env_utils
from ppo_contrastive import fixed_goal_dict

_sw_path = os.path.join(_REPO, 'scripts', 'render_sawyer_crl_traj_reward_video.py')
_sw_spec = _ilu.spec_from_file_location('render_sawyer_crl_traj_reward_video', _sw_path)
_sw = _ilu.module_from_spec(_sw_spec)
assert _sw_spec.loader is not None
_sw_spec.loader.exec_module(_sw)

_load_run_settings = _sw._load_run_settings
_rollout_collect = _sw._rollout_collect
_get_render_fn = _sw._get_render_fn
_DEFAULT_CAMERA = _sw._DEFAULT_CAMERA
_render_reward_strip = _sw._render_reward_strip
_compose_frame = _sw._compose_frame
_write_mp4 = _sw._write_mp4
_render_grad_norm_strip = _sw._crl._render_grad_norm_strip

OUT_DIR = 'figs/sawyer_peg/nf_logp_reward_probe'
SEED = 0
FPS = 15
HOLD_LAST = 8
MAX_TRIES = 1


def _run_dir_from_ckpt(ckpt_path: str) -> str:
  ckpt_dir = os.path.dirname(os.path.realpath(ckpt_path))
  return os.path.dirname(ckpt_dir)


def _tanh_scale(run_dir: str, override: float | None) -> float:
  """``nf_reward_tanh_scale`` from run_config, or ``override`` if set."""
  if override is not None:
    return float(override)
  cfg_path = os.path.join(run_dir, 'run_config.json')
  if os.path.isfile(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as fh:
      payload = json.load(fh)
    for blob in (payload.get('resolved_config') or {},
                 payload.get('flags') or {}, payload):
      if isinstance(blob, dict) and 'nf_reward_tanh_scale' in blob:
        return float(blob['nf_reward_tanh_scale'])
  return 50.0


def _load_nf_arch(run_dir: str):
  with open(os.path.join(run_dir, 'run_config.json'), 'r', encoding='utf-8') as fh:
    payload = json.load(fh)
  resolved = payload.get('resolved_config', {})
  flags = payload.get('flags', {})
  hidden = resolved.get('hidden_layer_sizes', flags.get('hidden_layer_sizes', (256,) * 6))
  if isinstance(hidden, str):
    hidden = tuple(int(x) for x in hidden.split(',') if x.strip())
  else:
    hidden = tuple(int(x) for x in hidden)
  return {
      'nf_rep_size': int(resolved.get('nf_rep_size', flags.get('nf_rep_size', 64))),
      'nf_num_blocks': int(resolved.get('nf_num_blocks', flags.get('nf_num_blocks', 8))),
      'nf_coupling_width': int(resolved.get(
          'nf_coupling_width', flags.get('nf_coupling_width', 256))),
      'nf_goal_enc_size': int(resolved.get(
          'nf_goal_enc_size', flags.get('nf_goal_enc_size', 0))),
      'nf_sa_hidden': int(resolved.get('nf_sa_hidden', flags.get('nf_sa_hidden', 1024))),
      'nf_sa_num_layers': int(resolved.get(
          'nf_sa_num_layers', flags.get('nf_sa_num_layers', 4))),
      'nf_state_only': bool(resolved.get(
          'nf_state_only', flags.get('nf_state_only', False))),
      'nf_scale_tanh': bool(resolved.get(
          'nf_scale_tanh', flags.get('nf_scale_tanh', False))),
      'nf_scale_tanh_c': float(resolved.get(
          'nf_scale_tanh_c', flags.get('nf_scale_tanh_c', 2.0))),
      'nf_normalize_goals': bool(resolved.get(
          'nf_normalize_goals', flags.get('nf_normalize_goals', True))),
      'nf_goal_std_min': float(resolved.get(
          'nf_goal_std_min', flags.get('nf_goal_std_min', 0.02))),
      'hidden_layer_sizes': hidden,
      'obs_dim': int(resolved.get('obs_dim', flags.get('obs_dim', 0))),
      'start_index': int(resolved.get('start_index', flags.get('start_index', 0))),
      'end_index': int(resolved.get('end_index', flags.get('end_index', -1))),
  }


def _load_goal_stats(run_dir: str, iteration: int, goal_dim: int):
  csv_path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
  best, best_dist = None, None
  with open(csv_path, 'r', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      ls = int(float(row.get('learner_steps') or row.get('iteration') or 0))
      dist = abs(ls - int(iteration))
      v = row.get('nf/goal_mean_0')
      if v in (None, '', 'nan'):
        continue
      if best_dist is None or dist < best_dist:
        best_dist = dist
        best = row
  if best is None:
    print('[vid] WARNING: no nf/goal_* stats; using mean=0 std=1')
    return (np.zeros(goal_dim, dtype=np.float32),
            np.ones(goal_dim, dtype=np.float32))
  mean = np.asarray(
      [float(best[f'nf/goal_mean_{i}']) for i in range(goal_dim)],
      dtype=np.float32)
  std = np.asarray(
      [float(best[f'nf/goal_std_{i}']) for i in range(goal_dim)],
      dtype=np.float32)
  print(f'[vid] goal stats from learner_steps={best.get("learner_steps")} '
        f'(|Δ|={best_dist})', flush=True)
  return mean, std


def make_nf_logp_grad_norm_fn(nf_nets, obs_dim: int):
  """‖∇_s log p_NF(g|s,a)‖ and ‖∇_a log p_NF(g|s,a)‖, g held fixed."""

  def _logp(nf_params, s, a, g_norm):
    lp = _nf.nf_log_prob(
        nf_nets, nf_params, s[None], a[None], g_norm[None])
    return jnp.sum(lp)

  @jax.jit
  def grad_norm_fn(nf_params, packed, action, goal_mean, goal_std):
    s = packed[:, :obs_dim]
    g = packed[:, obs_dim:]
    g_norm = (g - goal_mean) / (goal_std + 1e-8)

    def one(s_i, a_i, g_i):
      gs = jax.grad(_logp, argnums=1)(nf_params, s_i, a_i, g_i)
      ga = jax.grad(_logp, argnums=2)(nf_params, s_i, a_i, g_i)
      return jnp.linalg.norm(gs), jnp.linalg.norm(ga)

    gs_n, ga_n = jax.vmap(one)(s, action, g_norm)
    return gs_n, ga_n

  return grad_norm_fn


def _build_networks(env_name: str, seed: int, settings: dict, arch: dict):
  env_kwargs = {}
  if env_name in ('sawyer_bin', 'sawyer_peg'):
    env_kwargs['randomize_init'] = bool(settings['randomize_init'])
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed,
      fixed_start_end=settings['fixed_goal'], **env_kwargs)
  env_spec = specs.make_environment_spec(probe_env)
  act_dim = int(np.prod(env_spec.actions.shape))
  del probe_env

  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=contrastive.ContrastiveConfig().repr_dim,
      repr_norm=contrastive.ContrastiveConfig().repr_norm,
      twin_q=contrastive.ContrastiveConfig().twin_q,
      use_image_obs=contrastive.ContrastiveConfig().use_image_obs,
      hidden_layer_sizes=settings['hidden_layer_sizes'],
      actor_min_std=float(settings['actor_min_std']),
  )
  end_index = int(arch['end_index'])
  start_index = int(arch['start_index'])
  goal_dim = int(end_index - start_index) if end_index != -1 else int(obs_dim)
  print(f'[vid] NF arch: rep={arch["nf_rep_size"]} blocks={arch["nf_num_blocks"]} '
        f'channels={arch["nf_coupling_width"]} sa={arch["nf_sa_num_layers"]}x'
        f'{arch["nf_sa_hidden"]} state_only={arch["nf_state_only"]} '
        f'scale_tanh={arch["nf_scale_tanh"]}'
        f'{f"(c={arch["nf_scale_tanh_c"]:g})" if arch["nf_scale_tanh"] else ""} '
        f'goal_dim={goal_dim} act_dim={act_dim}', flush=True)
  nf_nets = _nf.make_nf_density_networks(
      obs_dim=int(obs_dim),
      act_dim=act_dim,
      goal_dim=goal_dim,
      hidden_layer_sizes=arch['hidden_layer_sizes'],
      rep_size=arch['nf_rep_size'],
      num_blocks=arch['nf_num_blocks'],
      channels=arch['nf_coupling_width'],
      goal_enc_size=arch['nf_goal_enc_size'],
      sa_hidden=arch['nf_sa_hidden'],
      sa_num_layers=arch['nf_sa_num_layers'],
      state_only=bool(arch['nf_state_only']),
      scale_tanh=bool(arch['nf_scale_tanh']),
      scale_tanh_c=float(arch['nf_scale_tanh_c']),
  )
  return networks, nf_nets, int(obs_dim), int(goal_dim)


def _enumerate_ckpts(checkpoint: str, checkpoint_dir: str,
                     max_ckpts: int = 0, ckpt_stride: int = 1,
                     min_iter: int = 0):
  if checkpoint_dir:
    files = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'ckpt_iter_*.pkl')),
        key=lambda p: int(re.search(r'ckpt_iter_(\d+)\.pkl$', p).group(1)))
  else:
    files = [checkpoint]
  kept = []
  for p in files:
    m = re.search(r'ckpt_iter_(\d+)\.pkl$', p)
    it = int(m.group(1)) if m else 0
    if it < int(min_iter):
      continue
    kept.append(p)
  files = kept
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


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', default='')
  p.add_argument('--checkpoint_dir', default='')
  p.add_argument('--env', default='sawyer_peg',
                 choices=sorted(fixed_goal_dict.keys()))
  p.add_argument('--out_dir', default=OUT_DIR)
  p.add_argument('--tag', default='')
  p.add_argument('--tag_prefix', default='sawyer_peg_nf_online')
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
  p.add_argument('--max_ckpts', type=int, default=0)
  p.add_argument('--ckpt_stride', type=int, default=1)
  p.add_argument('--min_iter', type=int, default=1,
                 help='Skip checkpoints with iter < this (default 1 skips iter 0).')
  p.add_argument('--normalize_reward', action='store_true')
  p.add_argument('--show_logp_grad', action='store_true')
  p.add_argument('--show_tanh', action='store_true',
                 help='Add a second strip: scale*tanh(log_p/scale), same y-lim '
                      'as raw log p so clipping is visible.')
  p.add_argument('--tanh_scale', type=float, default=None,
                 help='Override nf_reward_tanh_scale (default: run_config or 50).')
  return p.parse_args()


def _render_one(args, *, label, ckpt_path, networks, nf_nets, gym_env, render,
                obs_dim, goal_dim, arch, max_steps, out_dir):
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
  nf_params = ckpt['q_params']
  policy_params = ckpt['policy_params']
  iteration = int(ckpt.get('iteration') or 0)
  run_dir = _run_dir_from_ckpt(ckpt_path)
  if not arch.get('nf_normalize_goals', True):
    goal_mean = np.zeros(goal_dim, dtype=np.float32)
    goal_std = np.ones(goal_dim, dtype=np.float32)
    print('[vid] NF goal norm OFF: mean=0 std=1 (matches training)', flush=True)
  else:
    goal_mean, goal_std = _load_goal_stats(run_dir, iteration, goal_dim)
    goal_std = np.maximum(goal_std, arch['nf_goal_std_min']).astype(np.float32)
  gmean = jnp.asarray(goal_mean)
  gstd = jnp.asarray(goal_std)
  reward_fn = _nf.make_nf_reward_fn(nf_nets, obs_dim=int(obs_dim))
  print(f'[vid] policy_iter={iteration} max_steps={max_steps} '
        f'reward_src=q_params(online) state_only={arch["nf_state_only"]}',
        flush=True)

  best = None
  for attempt in range(int(args.max_tries)):
    roll = _rollout_collect(
        policy_params, gym_env, networks, render, reward_fn, nf_params,
        obs_dim=int(obs_dim), max_steps=max_steps,
        seed=int(args.seed) + attempt, stochastic=bool(args.stochastic),
        mean0=gmean, var1=gstd)
    print(f'[vid] try={attempt} success={roll["any_success"]} '
          f'logp_sum={roll["rewards"].sum():.2f} env_r={roll["total_env_r"]:.1f} '
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

  rewards = np.asarray(best['rewards'], dtype=np.float32)
  success = best['success']
  frames_rgb = best['frames']
  packed = np.asarray(best['packed'], dtype=np.float32)
  actions = np.asarray(best['actions'], dtype=np.float32)
  show_tanh = bool(args.show_tanh)
  tanh_scale = _tanh_scale(run_dir, args.tanh_scale) if show_tanh else 0.0
  rewards_tanh = (
      tanh_scale * np.tanh(np.asarray(rewards, dtype=np.float64) / tanh_scale)
      ).astype(np.float32) if show_tanh else None
  if show_tanh:
    gap = float(np.max(np.abs(rewards - rewards_tanh)))
    print(f'[vid] tanh scale={tanh_scale:g}  max|logp-tanh|={gap:.4g}  '
          f'logp range=[{float(np.min(rewards)):.3g},{float(np.max(rewards)):.3g}]  '
          f'tanh range=[{float(np.min(rewards_tanh)):.3g},'
          f'{float(np.max(rewards_tanh)):.3g}]',
          flush=True)
  show_grad = bool(args.show_logp_grad)
  grad_s = np.zeros(len(packed), dtype=np.float32)
  grad_a = np.zeros(len(packed), dtype=np.float32)
  if show_grad:
    grad_fn = make_nf_logp_grad_norm_fn(nf_nets, obs_dim=int(obs_dim))
    gs, ga = grad_fn(
        nf_params, jnp.asarray(packed), jnp.asarray(actions), gmean, gstd)
    grad_s = np.asarray(gs, dtype=np.float32)
    grad_a = np.asarray(ga, dtype=np.float32)
    print(f'[vid] ||∇_s logp|| range=[{grad_s.min():.4g},{grad_s.max():.4g}] '
          f'||∇_a logp|| range=[{grad_a.min():.4g},{grad_a.max():.4g}]',
          flush=True)

  T = len(rewards)
  first_succ = (int(np.argmax(success >= 0.5))
                if np.any(success >= 0.5) else -1)
  print(f'[vid] using first_success_t={first_succ} '
        f'logp_sum={rewards.sum():.3f}', flush=True)

  csv_cols = [
      np.arange(T), rewards, success, best['env_reward'],
      grad_s, grad_a,
  ]
  csv_header = 't,reward_logp_online,success,env_reward,grad_s_norm,grad_a_norm'
  if show_tanh:
    csv_cols.append(rewards_tanh)
    csv_header += ',reward_tanh'
  np.savetxt(
      csv_path,
      np.stack(csv_cols, axis=1),
      delimiter=',',
      header=csv_header,
      comments='')

  width = int(frames_rgb[0].shape[1])
  if width % 2:
    width -= 1
  if args.title:
    title = f'{args.title}  ·  {tag}'
  elif arch['nf_state_only']:
    title = rf'online NF  $r=\log p(g\mid s)$  ·  {tag}'
  else:
    title = rf'online NF  $r=\log p(g\mid s,a)$  ·  {tag}'
  ylabel = r'$\log p_{\mathrm{NF}}$'
  rewards_norm = rewards / (float(np.std(rewards)) + 1e-8)
  shared_ylim = None
  if show_tanh:
    pad = 0.35
    lo = float(min(np.min(rewards), np.min(rewards_tanh)) - pad)
    hi = float(max(np.max(rewards), np.max(rewards_tanh)) + pad)
    shared_ylim = (lo, hi)
    title = rf'before tanh  $r=\log p_{{\mathrm{{NF}}}}$  ·  {tag}'

  print('[vid] composing reward overlay...', flush=True)
  frames = []
  for t in range(T):
    strip = _render_reward_strip(
        rewards, success, t, width=width, height=220,
        title=title, ylabel=ylabel, line_color='#5ec8ff', ylim=shared_ylim)
    extra = []
    if show_tanh:
      extra.append(_render_reward_strip(
          rewards_tanh, success, t, width=width, height=220,
          title=(
              rf'after tanh  $r={tanh_scale:g}\,\tanh(\log p/{tanh_scale:g})$'
              rf'  ·  rails $\pm{tanh_scale:g}$  ·  {tag}'),
          ylabel=rf'${tanh_scale:g}\tanh(\log p/{tanh_scale:g})$',
          line_color='#ffb347', ylim=shared_ylim))
    if args.normalize_reward:
      extra.append(_render_reward_strip(
          rewards_norm, success, t, width=width, height=180,
          title=rf'normalised reward  $r/\sigma_r$  ·  {tag}',
          ylabel=r'$r/\sigma_r$',
          line_color='#ffb347'))
    if show_grad:
      if arch['nf_state_only']:
        grad_title = (
            r'$\Vert\nabla_s\log p_{\mathrm{NF}}(g\mid s)\Vert$  /  '
            r'$\Vert\nabla_a\log p_{\mathrm{NF}}\Vert$ (unused)'
            rf'  ·  {tag}')
      else:
        grad_title = (
            r'$\Vert\nabla_s\log p_{\mathrm{NF}}(g\mid s,a)\Vert$  /  '
            r'$\Vert\nabla_a\log p_{\mathrm{NF}}(g\mid s,a)\Vert$'
            rf'  ·  {tag}')
      extra.append(_render_grad_norm_strip(
          grad_s, grad_a, success, t, width=width, height=220,
          title=grad_title,
          ylabel_s=r'$\Vert\nabla_s\log p_{\mathrm{NF}}\Vert$',
          ylabel_a=r'$\Vert\nabla_a\log p_{\mathrm{NF}}\Vert$'))
    composed = _compose_frame(
        strip, frames_rgb[t], float(rewards[t]), bool(success[t] >= 0.5),
        extra_strips=extra or None)
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
      max_ckpts=int(args.max_ckpts), ckpt_stride=int(args.ckpt_stride),
      min_iter=int(args.min_iter))
  if not entries:
    raise FileNotFoundError('no checkpoints found')
  print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[vid] {len(entries)} checkpoint(s) (min_iter={args.min_iter}) → {out_dir}')

  first_path = entries[0][1]
  settings = _load_run_settings(first_path, args.env)
  run_dir = _run_dir_from_ckpt(first_path)
  arch = _load_nf_arch(run_dir)
  networks, nf_nets, obs_dim, goal_dim = _build_networks(
      args.env, seed=args.seed, settings=settings, arch=arch)

  env_kwargs = {}
  if args.env in ('sawyer_bin', 'sawyer_peg'):
    env_kwargs['randomize_init'] = bool(settings['randomize_init'])
  gym_env, _, env_max_steps = env_utils.load(
      args.env, fixed_start_end=settings['fixed_goal'], seed=args.seed,
      **env_kwargs)
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  camera = args.camera or _DEFAULT_CAMERA.get(args.env, 'corner')
  print(f'[vid] env={args.env} max_steps={max_steps} camera={camera} '
        f'rotate={args.rotate} obs_dim={obs_dim} goal_dim={goal_dim} '
        f'randomize_init={settings["randomize_init"]}', flush=True)
  render = _get_render_fn(gym_env, args.width, args.height, camera,
                          rotate_deg=args.rotate)

  for label, path in entries:
    _render_one(
        args, label=label, ckpt_path=path, networks=networks, nf_nets=nf_nets,
        gym_env=gym_env, render=render, obs_dim=obs_dim, goal_dim=goal_dim,
        arch=arch, max_steps=max_steps, out_dir=out_dir)


if __name__ == '__main__':
  main()
