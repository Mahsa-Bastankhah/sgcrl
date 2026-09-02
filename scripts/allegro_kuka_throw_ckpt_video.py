"""Roll out a trained AllegroKukaThrow PPO policy to mp4 (GPU camera).

PhysX is created *before* JAX so Preview 4 GPU kernels still register.
Env packing (throw vs --isaacgym_palm_goal) is read from the run's
``run_config.json`` so the obs dim matches the checkpoint.

  python scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint=logs/.../checkpoints/ckpt_iter_0000100.pkl \
      --output=videos/allegro_kuka_throw/run_iter100.mp4 \
      --deterministic --episodes=2

  # Stochastic + raw NF log p strip on top:
  python scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint=.../ckpt_iter_0000100.pkl \
      --output=videos/.../iter100.mp4 \
      --episodes=1 --reward-overlay
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _run_config_path(ckpt_path: str) -> str:
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
  return os.path.join(run_dir, 'run_config.json')


def _load_flags(ckpt_path: str) -> dict:
  cfg_path = _run_config_path(ckpt_path)
  if not os.path.isfile(cfg_path):
    return {}
  with open(cfg_path, 'r', encoding='utf-8') as fh:
    return dict((json.load(fh).get('flags') or {}))


def _as_xyz(raw, default):
  if raw is None:
    return default
  if isinstance(raw, str):
    parts = [p for p in raw.replace(' ', '').split(',') if p]
  else:
    parts = list(raw)
  if len(parts) != 3:
    return default
  return tuple(float(x) for x in parts)


def _load_run_hparams(flags: dict):
  hidden = (256, 256, 256, 256, 256, 256)
  raw = str(flags.get('hidden_layer_sizes') or '')
  if raw.strip():
    hidden = tuple(int(x) for x in raw.split(',') if x.strip())
  # SAC writes --actor_min_std (stock 1e-6). PPO writes --ppo_actor_min_std
  # (typically 1e-5). Fall back to the historical PPO floor.
  min_std = 1e-5
  if flags.get('actor_min_std') is not None:
    min_std = float(flags['actor_min_std'])
  elif flags.get('ppo_actor_min_std') is not None:
    min_std = float(flags['ppo_actor_min_std'])
  return hidden, min_std


def _load_env_kwargs(flags: dict, num_steps: int, seed: int, pipeline: str):
  ep = int(flags.get('isaacgym_episode_length') or 300)
  if int(num_steps) > 0:
    ep = int(num_steps)
  return dict(
      num_envs=1,
      seed=int(seed),
      episode_length=int(ep),
      pipeline=str(pipeline or flags.get('isaacgym_pipeline') or 'gpu'),
      enable_cameras=True,
      headless=True,
      fixed_target_xyz=_as_xyz(
          flags.get('isaacgym_fixed_target_xyz'), (0.5, -0.3, 0.4)),
      randomize_init=bool(flags.get('isaacgym_randomize_init', True)),
      randomize_object_xyz=bool(flags.get('isaacgym_randomize_object_xyz', False)),
      randomize_object_shape=bool(
          flags.get('isaacgym_randomize_object_shape', True)),
      palm_goal=bool(flags.get('isaacgym_palm_goal', False)),
      palm_goal_xyz=_as_xyz(
          flags.get('isaacgym_palm_goal_xyz'), (0.17, 0.08, 0.57)),
      table_push=bool(flags.get('isaacgym_table_push', False)),
      table_push_xyz=_as_xyz(
          flags.get('isaacgym_table_push_xyz'), None),
      table_spawn=bool(flags.get('isaacgym_table_spawn', False)),
  )


def _diag(env):
  task = env._env
  obj = task.object_pos[0].detach().cpu().numpy()
  goal = task.goal_pos[0].detach().cpu().numpy()
  d_obj = float(np.linalg.norm(obj - goal))
  tol = float(getattr(task, 'success_tolerance', 0.075))
  out = {
      'd_obj': d_obj,
      'succ': d_obj <= tol,
      'obj': obj,
      'goal': goal,
      'tol': tol,
      'd_palm': None,
  }
  if getattr(env, 'palm_goal', False):
    palm = task.palm_center_pos[0].detach().cpu().numpy()
    palm_g = env._goal_batch[0, :3].detach().cpu().numpy()
    out['d_palm'] = float(np.linalg.norm(palm - palm_g))
    out['palm'] = palm
    out['palm_g'] = palm_g
  return out


def _look_at(eye, target, up=(0.0, 0.0, 1.0)):
  eye = np.asarray(eye, dtype=np.float64)
  target = np.asarray(target, dtype=np.float64)
  up = np.asarray(up, dtype=np.float64)
  f = target - eye
  f = f / (np.linalg.norm(f) + 1e-12)
  s = np.cross(f, up)
  s = s / (np.linalg.norm(s) + 1e-12)
  u = np.cross(s, f)
  view = np.eye(4, dtype=np.float64)
  view[0, 0:3] = s
  view[1, 0:3] = u
  view[2, 0:3] = -f
  view[0, 3] = -np.dot(s, eye)
  view[1, 3] = -np.dot(u, eye)
  view[2, 3] = np.dot(f, eye)
  return view


def _perspective(hfov_deg, aspect, near=0.05, far=8.0):
  r = np.tan(np.deg2rad(float(hfov_deg)) * 0.5) * near
  t = r / max(float(aspect), 1e-6)
  proj = np.zeros((4, 4), dtype=np.float64)
  proj[0, 0] = near / r
  proj[1, 1] = near / t
  proj[2, 2] = -(far + near) / (far - near)
  proj[2, 3] = -2.0 * far * near / (far - near)
  proj[3, 2] = -1.0
  return proj


def _project(xyz, view, proj, width, height):
  p = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=np.float64)
  clip = proj @ (view @ p)
  if abs(clip[3]) < 1e-8:
    return None
  ndc = clip[:3] / clip[3]
  u = (ndc[0] * 0.5 + 0.5) * width
  v = (1.0 - (ndc[1] * 0.5 + 0.5)) * height
  if not np.isfinite(u) or not np.isfinite(v):
    return None
  return float(u), float(v)


def _cam_matrices(env, width, height):
  eye = getattr(env, '_cam_eye', None)
  tgt = getattr(env, '_cam_tgt', None)
  hfov = float(getattr(env, '_cam_hfov', 75.0) or 75.0)
  if eye is None or tgt is None:
    return None, None
  view = _look_at(eye, tgt)
  proj = _perspective(hfov, float(width) / max(float(height), 1.0))
  return view, proj


def _draw_goal_markers(rgb, env, diag):
  """Draw object-goal center + 7.5 cm success ring in the camera image."""
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return rgb
  h, w = rgb.shape[:2]
  view, proj = _cam_matrices(env, w, h)
  if view is None:
    return rgb
  goal = np.asarray(diag['goal'], dtype=np.float64)
  tol = float(diag['tol'])
  obj_uv = _project(goal, view, proj, w, h)
  ring = []
  for k in range(48):
    ang = 2.0 * np.pi * k / 48.0
    xyz = goal + np.array([tol * np.cos(ang), tol * np.sin(ang), 0.0])
    uv = _project(xyz, view, proj, w, h)
    if uv is not None:
      ring.append(uv)
  palm_uv = None
  if diag.get('palm_g') is not None:
    palm_uv = _project(np.asarray(diag['palm_g'], dtype=np.float64), view, proj, w, h)
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img, 'RGBA')
  if len(ring) >= 3:
    draw.polygon([(int(u), int(v)) for u, v in ring],
                 fill=(255, 140, 40, 55), outline=(255, 170, 50, 230))
    draw.line([(int(u), int(v)) for u, v in ring + ring[:1]],
              fill=(255, 200, 60, 255), width=3)
  if obj_uv is not None:
    u, v = int(obj_uv[0]), int(obj_uv[1])
    draw.ellipse([u - 7, v - 7, u + 7, v + 7], fill=(255, 120, 20, 255),
                 outline=(255, 255, 255, 255))
    draw.text((u + 12, v - 16), 'OBJ GOAL  r=7.5cm', fill=(255, 200, 80, 255))
  if palm_uv is not None:
    u, v = int(palm_uv[0]), int(palm_uv[1])
    draw.ellipse([u - 5, v - 5, u + 5, v + 5], fill=(80, 220, 255, 255),
                 outline=(255, 255, 255, 255))
    draw.text((u + 12, v + 6), 'PALM GOAL', fill=(80, 220, 255, 255))
  gx, gy, gz = (float(goal[0]), float(goal[1]), float(goal[2]))
  draw.text((8, h - 22),
            f'object goal=({gx:.2f},{gy:.2f},{gz:.3f}) on desk  '
            f'orange ring = success (7.5 cm)',
            fill=(255, 230, 180, 255))
  return np.asarray(img.convert('RGB'), dtype=np.uint8)


def _goal_stats_from_learner(ckpt_path: str, iteration: int, goal_dim: int):
  """NF goal mean/std at this ckpt iteration (needed for normalize_goals)."""
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
  path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
  mean = np.zeros((goal_dim,), dtype=np.float32)
  std = np.ones((goal_dim,), dtype=np.float32)
  if not os.path.isfile(path):
    return mean, std
  best = None
  with open(path, newline='') as fh:
    for row in csv.DictReader(fh):
      try:
        it = int(float(row.get('iteration', '')))
      except (TypeError, ValueError):
        continue
      if best is None or abs(it - int(iteration)) < abs(best[0] - int(iteration)):
        best = (it, row)
  if best is None:
    return mean, std
  row = best[1]
  for i in range(int(goal_dim)):
    try:
      mean[i] = float(row[f'nf/goal_mean_{i}'])
      std[i] = float(row[f'nf/goal_std_{i}'])
    except (KeyError, TypeError, ValueError):
      pass
  std = np.maximum(std, 1e-6)
  print(f'[ckpt_video] NF goal stats from learner iter={best[0]} '
        f'(ckpt iter={iteration})', flush=True)
  return mean, std


def _render_reward_strip(rewards, success, t, width, height=160, title=''):
  """PIL-only sparkline (sgcrl_isaacgym has no matplotlib)."""
  from PIL import Image, ImageDraw

  T = max(len(rewards), 1)
  r_now = float(rewards[t])
  r_min = float(np.min(rewards)) - 0.35
  r_max = float(np.max(rewards)) + 0.35
  if r_max <= r_min:
    r_max = r_min + 1.0
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  succ_now = bool(success[t] >= 0.5)
  img = Image.new('RGB', (int(width), int(height)), (15, 20, 25))
  draw = ImageDraw.Draw(img)
  pad_l, pad_r, pad_t, pad_b = 10, 150, 28, 10
  x0, y0 = pad_l, pad_t
  x1, y1 = int(width) - pad_r, int(height) - pad_b
  plot_w = max(x1 - x0, 1)
  plot_h = max(y1 - y0, 1)

  def _xy(i, r):
    x = x0 + int(round((i / max(T - 1, 1)) * plot_w))
    y = y1 - int(round(((float(r) - r_min) / (r_max - r_min)) * plot_h))
    return x, y

  draw.rectangle([x0, y0, x1, y1], outline=(58, 70, 84))
  fade = [_xy(i, rewards[i]) for i in range(T)]
  if len(fade) >= 2:
    draw.line(fade, fill=(90, 106, 122), width=1)
  live = [_xy(i, rewards[i]) for i in range(t + 1)]
  if len(live) >= 2:
    draw.line(live, fill=(125, 255, 176), width=2)
  cx, cy = _xy(t, r_now)
  draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 229, 102))
  draw.line([(cx, y0), (cx, y1)], fill=(255, 229, 102))
  if first_succ >= 0:
    sx, _ = _xy(first_succ, rewards[first_succ])
    draw.line([(sx, y0), (sx, y1)], fill=(255, 138, 76))
  draw.text((8, 6), title or 'raw NF  r = log p(g|s,a)', fill=(232, 238, 244))
  badge = f'log p = {r_now:+.2f}'
  status = 'SUCCESS' if succ_now else 'no success'
  draw.text((x1 + 8, y0 + 8), 'raw log p', fill=(154, 167, 181))
  draw.text((x1 + 8, y0 + 32), badge, fill=(125, 255, 176) if succ_now else (255, 229, 102))
  draw.text((x1 + 8, y0 + 56), status, fill=(125, 255, 176) if succ_now else (255, 138, 76))
  draw.text((x1 + 8, y0 + 80), f't = {t}/{T - 1}', fill=(200, 208, 216))
  return np.asarray(img, dtype=np.uint8)


def _stack_reward_overlay(rgb, rewards, success, t, title):
  from PIL import Image
  resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
  w = int(rgb.shape[1])
  if w % 2:
    w -= 1
  strip = _render_reward_strip(rewards, success, t, width=w, title=title)
  rh = int(round(rgb.shape[0] * (strip.shape[1] / rgb.shape[1])))
  render_r = np.asarray(
      Image.fromarray(rgb).resize((strip.shape[1], rh), resample))
  out = np.concatenate([strip, render_r], axis=0)
  h, ww = out.shape[:2]
  if h % 2 or ww % 2:
    out = out[: h - (h % 2), : ww - (ww % 2)]
  return out


def _annotate(rgb, lines):
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return rgb
  img = Image.fromarray(np.ascontiguousarray(rgb)).convert('RGB')
  draw = ImageDraw.Draw(img)
  pad = 6
  line_h = 16
  w = img.size[0]
  h = pad * 2 + line_h * len(lines)
  draw.rectangle((0, 0, w, h), fill=(0, 0, 0))
  for i, line in enumerate(lines):
    draw.text((8, pad + i * line_h), line, fill=(255, 255, 255))
  return np.asarray(img, dtype=np.uint8)


def _build_env(kwargs):
  from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
  return AllegroKukaThrowVecEnv(**kwargs)


def _load_ckpt(ckpt_path: str):
  with open(ckpt_path, 'rb') as fh:
    return pickle.load(fh)


def _build_actor(env, ckpt, flags: dict, deterministic: bool):
  import sgcrl_jax_acme_compat  # noqa: F401
  import jax
  import jax.numpy as jnp
  from acme import specs as acme_specs
  from contrastive.networks import make_networks

  hidden, min_std = _load_run_hparams(flags)
  packed_dim = int(env.obs_dim + env.goal_dim)
  obs_spec = acme_specs.Array(
      shape=(packed_dim,), dtype=np.float32, name='observation')
  act_spec = acme_specs.BoundedArray(
      shape=(int(env.action_dim),), dtype=np.float32,
      minimum=-1.0, maximum=1.0, name='action')
  spec = acme_specs.EnvironmentSpec(
      observations=obs_spec, actions=act_spec,
      rewards=acme_specs.Array(shape=(), dtype=np.float32, name='reward'),
      discounts=acme_specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0,
          name='discount'),
  )
  networks = make_networks(
      spec,
      obs_dim=int(env.obs_dim),
      hidden_layer_sizes=hidden,
      actor_min_std=float(min_std),
  )
  policy_params = ckpt['policy_params']
  iteration = int(ckpt.get('iteration', -1))
  print(f'[ckpt_video] loaded iter={iteration}  '
        f'hidden={hidden}  min_std={min_std}  det={deterministic}  '
        f'obs_dim={env.obs_dim} goal_dim={env.goal_dim} '
        f'palm_goal={getattr(env, "palm_goal", False)}',
        flush=True)

  sample_fn = networks.sample_eval if deterministic else networks.sample

  @jax.jit
  def act(params, obs, key):
    dist = networks.policy_network.apply(params, obs)
    action = sample_fn(dist, key)
    return jnp.clip(action, -1.0, 1.0)

  gpu = jax.devices('gpu')
  if gpu:
    policy_params = jax.device_put(policy_params, gpu[0])
  return act, policy_params, iteration, networks


def _build_nf_reward(env, ckpt, flags: dict):
  import jax
  import jax.numpy as jnp
  from contrastive import nf_density as _nf

  nf_nets = _nf.make_nf_density_networks(
      obs_dim=int(env.obs_dim),
      act_dim=int(env.action_dim),
      goal_dim=int(env.goal_dim),
      rep_size=int(flags.get('nf_rep_size') or 64),
      num_blocks=int(flags.get('nf_num_blocks') or 6),
      channels=int(flags.get('nf_coupling_width') or 192),
      goal_enc_size=int(flags.get('nf_goal_enc_size') or 0),
      sa_hidden=int(flags.get('nf_sa_hidden') or 192),
      sa_num_layers=int(flags.get('nf_sa_num_layers') or 3),
      state_only=bool(flags.get('nf_state_only', False)),
      scale_tanh=bool(flags.get('nf_scale_tanh', False)),
  )
  reward_fn = _nf.make_nf_reward_fn(
      nf_nets, obs_dim=int(env.obs_dim),
      tanh_scale=0.0, reward_mode='forward')
  nf_params = ckpt.get('q_params_ema', ckpt['q_params'])
  src = 'q_params_ema' if 'q_params_ema' in ckpt else 'q_params'
  gpu = jax.devices('gpu')
  if gpu:
    nf_params = jax.device_put(nf_params, gpu[0])
  print(f'[ckpt_video] NF reward from {src}  (raw log p)', flush=True)
  return reward_fn, nf_params


def _episode_grads(networks, value_params, reward_fn, nf_params,
                   packed, actions, gmean, gstd, obs_dim: int):
  import jax
  import jax.numpy as jnp

  packed_j = jnp.asarray(packed)
  actions_j = jnp.asarray(actions)
  gmean_j = jnp.asarray(gmean)
  gstd_j = jnp.asarray(gstd)
  g = packed_j[:, int(obs_dim):]
  s = packed_j[:, :int(obs_dim)]

  def _r_of_s(state, action, goal):
    obs = jnp.concatenate([state, goal], axis=-1)[None]
    return jnp.reshape(reward_fn(nf_params, obs, action[None], gmean_j, gstd_j), ())

  def _v_of_s(state, goal):
    obs = jnp.concatenate([state, goal], axis=-1)[None]
    return jnp.reshape(networks.value_network.apply(value_params, obs), ())

  def _r_norm(state, action, goal):
    return jnp.linalg.norm(jax.grad(_r_of_s)(state, action, goal))

  def _v_norm(state, goal):
    return jnp.linalg.norm(jax.grad(_v_of_s)(state, goal))

  r_n = jax.vmap(_r_norm)(s, actions_j, g)
  v_n = jax.vmap(_v_norm)(s, g)
  return np.asarray(r_n), np.asarray(v_n)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--num-steps', type=int, default=0,
                 help='0 = isaacgym_episode_length from run_config')
  p.add_argument('--episodes', type=int, default=2)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pipeline', default='gpu', choices=('gpu', 'cpu'))
  p.add_argument('--deterministic', action='store_true')
  p.add_argument('--reward-overlay', action='store_true',
                 help='Stack raw NF log p(g|s,a) timeline on top of the camera')
  p.add_argument('--dump-grads', action='store_true',
                 help='Write per-step ||∇_s r|| and ||∇_s V|| next to the mp4')
  args = p.parse_args()

  os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
  flags = _load_flags(args.checkpoint)
  env_kw = _load_env_kwargs(
      flags, args.num_steps, args.seed, args.pipeline)
  print(f'[ckpt_video] env kwargs: { {k: env_kw[k] for k in env_kw if k != "seed"} }',
        flush=True)

  env = _build_env(env_kw)
  ckpt = _load_ckpt(args.checkpoint)
  act, policy_params, iteration, networks = _build_actor(
      env, ckpt, flags, bool(args.deterministic))
  reward_fn = nf_params = None
  gmean = gstd = None
  if args.reward_overlay or args.dump_grads:
    reward_fn, nf_params = _build_nf_reward(env, ckpt, flags)
    gmean, gstd = _goal_stats_from_learner(
        args.checkpoint, iteration, int(env.goal_dim))

  import torch
  import jax
  import jax.numpy as jnp
  import imageio

  n_steps = int(env.max_episode_steps)
  frames = []
  key = jax.random.PRNGKey(int(args.seed))
  summaries = []
  grad_rows = []
  value_params = ckpt.get('value_params')
  if args.dump_grads and value_params is not None:
    gpu = jax.devices('gpu')
    if gpu:
      value_params = jax.device_put(value_params, gpu[0])
  for ep in range(int(args.episodes)):
    obs_t = env.reset()
    min_obj = 1e9
    min_palm = 1e9
    ever_succ = False
    packed_list = []
    action_list = []
    succ_list = []
    rgb_list = []
    for t in range(n_steps):
      obs = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
      if obs.ndim == 1:
        obs = obs[None]
      key, sub = jax.random.split(key)
      action = act(policy_params, jnp.asarray(obs), sub)
      a_np = np.asarray(action, dtype=np.float32).reshape(-1)
      a_t = torch.as_tensor(a_np[None], device=env.device)
      packed_list.append(obs.reshape(-1).copy())
      action_list.append(a_np)
      obs_t, _, _ = env.step(a_t)
      d = _diag(env)
      min_obj = min(min_obj, d['d_obj'])
      ever_succ = ever_succ or d['succ']
      succ_list.append(1.0 if d['succ'] else 0.0)
      if d['d_palm'] is not None:
        min_palm = min(min_palm, d['d_palm'])
      line = (
          f'ep{ep} t={t:3d}  |obj-g|={d["d_obj"]:.3f}m  '
          f'succ={int(d["succ"])}  tol={d["tol"]:.3f}'
      )
      if d['d_palm'] is not None:
        line += f'  |palm-g|={d["d_palm"]:.3f}m'
      g = d['goal']
      line += f'  obj_goal=({g[0]:.2f},{g[1]:.2f},{g[2]:.3f})'
      rgb = env.render_rgb()
      if rgb is None:
        raise RuntimeError('camera returned no frame; enable_cameras failed')
      rgb = _annotate(rgb, [os.path.basename(args.checkpoint), line])
      rgb = _draw_goal_markers(rgb, env, d)
      rgb_list.append(rgb)
      if t % 50 == 0:
        print(f'[ckpt_video] {line}', flush=True)

    rewards = None
    if reward_fn is not None:
      try:
        packed = np.asarray(packed_list, dtype=np.float32)
        if packed.ndim == 3:
          packed = packed.reshape(packed.shape[0], -1)
        actions = np.asarray(action_list, dtype=np.float32)
        if actions.ndim == 3:
          actions = actions.reshape(actions.shape[0], -1)
        rewards = np.asarray(
            reward_fn(
                nf_params, jnp.asarray(packed), jnp.asarray(actions),
                jnp.asarray(gmean), jnp.asarray(gstd)),
            dtype=np.float32)
        success = np.asarray(succ_list, dtype=np.float32)
        title = f'{os.path.basename(args.checkpoint)}  raw log p'
        for t, rgb in enumerate(rgb_list):
          frames.append(_stack_reward_overlay(rgb, rewards, success, t, title))
      except Exception as exc:
        print(f'[ckpt_video] reward/overlay failed ({exc}); '
              f'writing camera frames', flush=True)
        frames.extend(rgb_list)
      if args.dump_grads and value_params is not None:
        try:
          r_n, v_n = _episode_grads(
              networks, value_params, reward_fn, nf_params,
              packed, actions, gmean, gstd, int(env.obs_dim))
          grad_rows.append({
              'iteration': iteration,
              'episode': ep,
              'grad_s_r_mean': float(np.mean(r_n)),
              'grad_s_r_max': float(np.max(r_n)),
              'grad_s_v_mean': float(np.mean(v_n)),
              'grad_s_v_max': float(np.max(v_n)),
              'ever_succ': int(ever_succ),
              'logp_mean': float(np.mean(rewards)),
          })
          print(f'[ckpt_video] grads ep{ep}: ||∇_s r|| mean={np.mean(r_n):.3f} '
                f'max={np.max(r_n):.3f}  ||∇_s V|| mean={np.mean(v_n):.3f} '
                f'max={np.max(v_n):.3f}', flush=True)
        except Exception as exc:
          print(f'[ckpt_video] dump-grads failed ({exc}); continuing',
                flush=True)
    else:
      frames.extend(rgb_list)

    extra = '' if min_palm > 1e8 else f'  min|palm-g|={min_palm:.3f}m'
    summary = (
        f'ep{ep}: min|obj-g|={min_obj:.3f}m  ever_succ={int(ever_succ)}{extra}')
    summaries.append(summary)
    print(f'[ckpt_video] {summary}', flush=True)
    if ep + 1 < int(args.episodes):
      gap = np.zeros_like(frames[-1])
      frames.extend([gap] * 8)

  imageio.mimsave(args.output, frames, fps=int(args.fps))
  print(f'[ckpt_video] wrote {len(frames)} frames iter={iteration} '
        f'-> {args.output}', flush=True)
  if grad_rows:
    side = os.path.splitext(args.output)[0] + '_grads.json'
    with open(side, 'w', encoding='utf-8') as fh:
      json.dump(grad_rows, fh, indent=2)
    print(f'[ckpt_video] wrote {side}', flush=True)
  for line in summaries:
    print('[ckpt_video] ' + line, flush=True)


if __name__ == '__main__':
  main()
