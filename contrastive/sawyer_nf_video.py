"""In-train Sawyer episode video with NF log-p reward overlay.

Same stacked layout as ``scripts/render_sawyer_crl_traj_reward_video.py``:
reward timeline on top, MuJoCo below, current NF reward badge.
"""
from __future__ import annotations

import csv
import importlib.util as _ilu
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name: str, path: str):
  spec = _ilu.spec_from_file_location(name, path)
  mod = _ilu.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(mod)
  return mod


def _unwrap_gym(env):
  cur = env
  seen = set()
  while id(cur) not in seen:
    seen.add(id(cur))
    if hasattr(cur, 'sim'):
      return cur
    nxt = getattr(cur, '_environment', None)
    if nxt is None:
      nxt = getattr(cur, 'environment', None)
    if nxt is None:
      nxt = getattr(cur, 'env', None)
    if nxt is None or nxt is cur:
      break
    cur = nxt
  return cur


def _dm_obs(ts):
  return np.asarray(ts.observation, dtype=np.float32)


def render_nf_reward_episode(
    *,
    env,
    networks,
    policy_params,
    nf_reward_fn,
    nf_params,
    nf_params_target=None,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
    max_steps: int,
    out_path: str,
    fps: int = 15,
    width: int = 640,
    height: int = 480,
    camera: str = 'corner',
    rotate_deg: int = 180,
    title: str = '',
    stochastic: bool = False,
    rng_seed: int = 0,
    reward_norm_std: Optional[float] = None,
    use_external_reward: bool = False,
    external_reward_scale: float = 1.0,
    external_reward_before_norm: bool = False,
) -> tuple[str, int]:
  """Roll one episode; overlay online logp + PPO (target-net) normalized r.

  Top strip: online NF ``log p(g|s,a)`` (unnormalized).
  Bottom strip: reward the value net uses — target/EMA NF logp, then
  ``r / σ_G`` with frozen ``ReturnNormalizer.std``, plus the external
  success bonus in the same before/after-norm order as training.
  """
  _roll = _load_script(
      'ppo_rollout_video',
      os.path.join(_REPO, 'scripts', 'ppo_rollout_video.py'))
  _jp_before = os.environ.get('JAX_PLATFORMS')
  _crl = _load_script(
      'render_frozen_crl_traj_reward_video',
      os.path.join(_REPO, 'scripts', 'render_frozen_crl_traj_reward_video.py'))
  if _jp_before is None:
    os.environ.pop('JAX_PLATFORMS', None)
  else:
    os.environ['JAX_PLATFORMS'] = _jp_before

  gym_env = _unwrap_gym(env)
  render = _roll._get_render_fn(
      gym_env, width, height, camera, rotate_deg=rotate_deg)

  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  @jax.jit
  def policy_sample(params, obs, rng):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample(dist, rng)

  ts = env.reset()
  obs = _dm_obs(ts)
  packed_list = []
  action_list = []
  success_list = []
  env_r_list = []
  frames_rgb = []
  rng = jax.random.PRNGKey(int(rng_seed))
  for _ in range(int(max_steps)):
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = policy_sample(policy_params, obs[None], k)
    else:
      action_j = policy_mode(policy_params, obs[None])
    action = np.asarray(action_j)[0].astype(np.float32)
    packed_list.append(obs.copy())
    action_list.append(action)
    ts = env.step(action)
    r = 0.0 if ts.reward is None else float(ts.reward)
    succ = 1.0 if r >= 0.5 else 0.0
    success_list.append(succ)
    env_r_list.append(r)
    frames_rgb.append(np.asarray(render()))
    obs = _dm_obs(ts)
    if bool(ts.last()):
      break

  packed = np.asarray(packed_list, dtype=np.float32)
  actions = np.asarray(action_list, dtype=np.float32)
  success = np.asarray(success_list, dtype=np.float32)
  env_r = np.asarray(env_r_list, dtype=np.float32)
  gmean = jnp.asarray(nf_goal_mean, dtype=jnp.float32)
  gstd = jnp.asarray(nf_goal_std, dtype=jnp.float32)
  rewards = np.asarray(
      nf_reward_fn(
          nf_params, jnp.asarray(packed), jnp.asarray(actions), gmean, gstd),
      dtype=np.float32)
  _tgt = nf_params_target if nf_params_target is not None else nf_params
  rewards_tgt = np.asarray(
      nf_reward_fn(
          _tgt, jnp.asarray(packed), jnp.asarray(actions), gmean, gstd),
      dtype=np.float32)
  _std = float(reward_norm_std) if reward_norm_std is not None else float('nan')
  ext = (
      float(external_reward_scale) * (success >= 0.5).astype(np.float32)
      if use_external_reward else np.zeros_like(rewards_tgt))
  r_for_norm = rewards_tgt.astype(np.float32, copy=True)
  if use_external_reward and external_reward_before_norm:
    r_for_norm = r_for_norm + ext
  if np.isfinite(_std) and _std > 0.0:
    rewards_norm = (r_for_norm / np.float32(_std)).astype(np.float32)
  else:
    rewards_norm = r_for_norm
  if use_external_reward and not external_reward_before_norm:
    rewards_norm = rewards_norm + ext

  csv_path = os.path.splitext(out_path)[0] + '_reward.csv'
  os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
  with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh)
    w.writerow([
        't', 'nf_logp_online', 'nf_logp_target', 'ppo_reward_norm',
        'env_reward', 'success', 'return_norm_std',
    ])
    for t in range(len(rewards)):
      w.writerow([
          t, float(rewards[t]), float(rewards_tgt[t]),
          float(rewards_norm[t]), float(env_r[t]), float(success[t]), _std,
      ])

  T = len(frames_rgb)
  if T == 0:
    raise RuntimeError('Sawyer NF video rollout produced 0 frames')
  vid_w = int(frames_rgb[0].shape[1])
  if vid_w % 2:
    vid_w -= 1
  ylabel = r'$r_{\mathrm{online}}=\log p_{\mathrm{NF}}$'
  plot_title = title or 'online NF logp (raw)'
  frames = []
  have_norm = bool(np.isfinite(rewards_norm).any())
  for t in range(T):
    strip = _crl._render_reward_strip(
        rewards, success, t, width=vid_w, height=220,
        title=plot_title, ylabel=ylabel, ylim=None,
        line_color='#7dffb0')
    extra = None
    if have_norm:
      std_txt = f'{_std:.3g}' if np.isfinite(_std) else 'n/a'
      extra = [_crl._render_reward_strip(
          rewards_norm, success, t, width=vid_w, height=180,
          title=(rf'PPO / value target  $r_{{\mathrm{{tgt}}}}/\sigma_G$'
                 rf'  ($\sigma_G$={std_txt}, frozen)'),
          ylabel=r'$r_{\mathrm{PPO}}$', ylim=None,
          line_color='#ffe566')]
    r_now = float(rewards[t])
    rn_now = float(rewards_norm[t]) if have_norm else float('nan')
    composed = _crl._compose_frame(
        strip, frames_rgb[t], r_now, bool(success[t] >= 0.5),
        extra_strips=extra)
    if have_norm and np.isfinite(rn_now):
      composed = _stamp_norm_badge(composed, r_now, rn_now, _std)
    h, w = composed.shape[:2]
    if h % 2 or w % 2:
      composed = composed[: h - (h % 2), : w - (w % 2)]
    frames.append(composed)
  frames.extend([frames[-1]] * 8)
  _crl._write_mp4(frames, out_path, int(fps))
  still_path = os.path.splitext(out_path)[0] + '.png'
  Image.fromarray(frames[min(T // 2, len(frames) - 1)]).save(still_path)
  return out_path, T


def _stamp_norm_badge(
    frame: np.ndarray, r_raw: float, r_norm: float, std: float) -> np.ndarray:
  """Second line on the existing r_online badge: frozen r/σ_G."""
  from PIL import ImageDraw, ImageFont
  im = Image.fromarray(frame).convert('RGBA')
  draw = ImageDraw.Draw(im, 'RGBA')
  try:
    font = ImageFont.truetype(
        '/usr/share/fonts/dejavu/DejaVuSans.ttf', 22)
  except OSError:
    font = ImageFont.load_default()
  std_s = f'{std:.3g}' if np.isfinite(std) else 'n/a'
  text = f'r_PPO = {r_norm:+.3f}   σ_G = {std_s}'
  bb = draw.textbbox((0, 0), text, font=font)
  pad_x, pad_y = 12, 6
  bw, bh = bb[2] - bb[0] + 2 * pad_x, bb[3] - bb[1] + 2 * pad_y
  x, y = 12, 58
  draw.rounded_rectangle(
      [x, y, x + bw, y + bh], radius=8,
      fill=(16, 22, 28, 210), outline=(255, 229, 102, 255), width=2)
  draw.text((x + pad_x, y + pad_y - 2), text, fill=(255, 229, 102), font=font)
  return np.asarray(im.convert('RGB'))
