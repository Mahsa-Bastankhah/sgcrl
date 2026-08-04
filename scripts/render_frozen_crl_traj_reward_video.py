"""Render creative-3-task1 successful traj video with live φ·ψ reward strip.

Produces a stacked frame: reward timeline on top + MuJoCo render below, with
a moving cursor tracking the current reward.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Force CPU before importing the video helper (it setdefaults JAX_PLATFORMS=cpu).
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

from contrastive import ContrastiveConfig
from contrastive import ppo_learner
from envs.builderbench_utils import (
    filter_pd_policy_state_obs,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

_bbv_path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_rollout_video.py')
_bbv_spec = _ilu.spec_from_file_location('ppo_builderbench_rollout_video', _bbv_path)
_bbv = _ilu.module_from_spec(_bbv_spec)
assert _bbv_spec.loader is not None
_bbv_spec.loader.exec_module(_bbv)
_build_networks = _bbv._build_networks
_load_train_ctx = _bbv._load_train_ctx
_make_bb_env = _bbv._make_bb_env
_make_policy_fn = _bbv._make_policy_fn
_maybe_fix_target = _bbv._maybe_fix_target


import argparse

ENV = 'builderbench_creative_3_task1'
DEFAULT_CKPT = (
    'logs/ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_catselect_extrew1/'
    'ppo_builderbench_creative_3_task1_0/checkpoints/latest.pkl')
OUT_DIR = 'figs/builderbench/frozen_crl_reward_probe'
SEED = 0
MAX_TRIES = 40
FPS = 8
HOLD_LAST = 8  # extra frames at end so the final reward lingers


def _font(size: int):
  for path in (
      '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf',
      '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
      '/usr/share/fonts/liberation/LiberationSans-Bold.ttf',
  ):
    if os.path.isfile(path):
      return ImageFont.truetype(path, size=size)
  return ImageFont.load_default()


def _compile_rollout_and_states(policy, env, ep_len, fixed_goal, mocap_targets,
                                num_cubes, filter_policy_obs):
  """Return (traj_dict, env_states) for one episode under a single jit."""

  @jax.jit
  def _run(key):
    env_key, key = jax.random.split(key)
    state = env.reset(jax.random.split(env_key, 1))
    state = _maybe_fix_target(state, fixed_goal, mocap_targets, num_cubes)

    def step(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      goals = state.info['target_goal']
      action, _ = policy(state.obs, goals, act_key)
      if filter_policy_obs:
        policy_obs = filter_pd_policy_state_obs(state.obs, num_cubes)
      else:
        policy_obs = state.obs
      packed = jnp.concatenate([policy_obs, goals], axis=-1)
      next_state = env.step(state, action)
      next_state = _maybe_fix_target(
          next_state, fixed_goal, mocap_targets, num_cubes)
      succ = jnp.asarray(next_state.metrics['success']).reshape(-1)[0]
      out = {
          'packed': packed[0],
          'action': action[0],
          'success': succ,
      }
      return (next_state, key), (out, next_state)

    _, (traj, states) = jax.lax.scan(step, (state, key), (), length=ep_len)
    return traj, states

  return _run


def _render_reward_strip(
    rewards: np.ndarray,
    success: np.ndarray,
    t: int,
    width: int,
    height: int = 220,
    *,
    title: str = '',
    ylim: tuple[float, float] | None = None,
) -> np.ndarray:
  """Matplotlib strip: full curve ghosted, revealed to t, live cursor + value."""
  T = len(rewards)
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  r_now = float(rewards[t])
  if ylim is None:
    r_min = float(np.min(rewards)) - 0.35
    r_max = float(np.max(rewards)) + 0.35
  else:
    r_min, r_max = float(ylim[0]), float(ylim[1])

  dpi = 120
  fig_w = width / dpi
  fig_h = height / dpi
  fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='#0f1419')
  ax = fig.add_axes([0.07, 0.22, 0.72, 0.62])
  ax.set_facecolor('#0f1419')

  xs = np.arange(T)
  # Ghost full curve
  ax.plot(xs, rewards, color='#5a6a7a', lw=1.6, alpha=0.45, zorder=1)
  # Revealed prefix
  ax.plot(xs[: t + 1], rewards[: t + 1], color='#5ec8ff', lw=2.4, zorder=2)
  ax.fill_between(
      xs[: t + 1], rewards[: t + 1], r_min,
      color='#5ec8ff', alpha=0.12, zorder=1)
  # Current marker
  ax.scatter([t], [r_now], s=55, color='#ffe566', edgecolors='#1a1a1a',
             linewidths=0.8, zorder=4)
  ax.axvline(t, color='#ffe566', ls=':', lw=1.0, alpha=0.7, zorder=3)

  if first_succ >= 0:
    ax.axvline(first_succ, color='#ff8a4c', ls='--', lw=1.2, alpha=0.85,
               zorder=2)
    if t >= first_succ:
      ax.scatter([first_succ], [rewards[first_succ]], s=40,
                 color='#ff8a4c', zorder=3)

  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(r_min, r_max)
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(r'$r=\varphi\!\cdot\!\psi$', color='#c8d0d8', fontsize=9)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)

  # Big live readout on the right
  ax_txt = fig.add_axes([0.80, 0.22, 0.18, 0.62])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  succ_now = bool(success[t] >= 0.5)
  ax_txt.text(
      0.05, 0.78, 'reward now', transform=ax_txt.transAxes,
      color='#9aa7b5', fontsize=9, va='center')
  ax_txt.text(
      0.05, 0.52, f'{r_now:+.3f}', transform=ax_txt.transAxes,
      color='#ffe566' if not succ_now else '#7dffb0',
      fontsize=16, fontweight='bold', va='center',
      family='DejaVu Sans')
  ax_txt.text(
      0.05, 0.28, f't = {t}/{T - 1}', transform=ax_txt.transAxes,
      color='#c8d0d8', fontsize=10, va='center')
  ax_txt.text(
      0.05, 0.10,
      'SUCCESS' if succ_now else 'no success',
      transform=ax_txt.transAxes,
      color='#7dffb0' if succ_now else '#ff8a4c',
      fontsize=11, fontweight='bold', va='center')

  fig.suptitle(
      title or (r'frozen CRL reward  $r=\varphi(s,a)\cdot\psi(g)$'
                '  ·  creative-3-task1'),
      color='#e8eef4', fontsize=10, y=0.96)

  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    buf = np.asarray(
        Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf


def _compose_frame(strip: np.ndarray, render: np.ndarray, t: int,
                   reward: float, success: bool) -> np.ndarray:
  """Stack strip above render; badge current reward on the render corner."""
  rw = strip.shape[1]
  # Match render width to strip
  rh = int(round(render.shape[0] * (rw / render.shape[1])))
  render_r = np.asarray(
      Image.fromarray(render).resize((rw, rh), Image.Resampling.LANCZOS))

  im = Image.fromarray(render_r).convert('RGBA')
  draw = ImageDraw.Draw(im, 'RGBA')
  badge = f'r = {reward:+.3f}'
  font = _font(28)
  bb = draw.textbbox((0, 0), badge, font=font)
  pad_x, pad_y = 14, 8
  bw, bh = bb[2] - bb[0] + 2 * pad_x, bb[3] - bb[1] + 2 * pad_y
  x0, y0 = 12, 12
  fill = (16, 22, 28, 210)
  outline = (125, 255, 176, 255) if success else (255, 229, 102, 255)
  draw.rounded_rectangle(
      [x0, y0, x0 + bw, y0 + bh], radius=10, fill=fill, outline=outline, width=2)
  draw.text((x0 + pad_x, y0 + pad_y - 2), badge, fill=outline[:3], font=font)
  if success:
    sfont = _font(22)
    st = 'SUCCESS'
    sbb = draw.textbbox((0, 0), st, font=sfont)
    sw, sh = sbb[2] - sbb[0] + 20, sbb[3] - sbb[1] + 10
    sx = rw - sw - 12
    draw.rounded_rectangle(
        [sx, 12, sx + sw, 12 + sh], radius=8,
        fill=(20, 60, 40, 220), outline=(125, 255, 176, 255), width=2)
    draw.text((sx + 10, 14), st, fill=(125, 255, 176), font=sfont)

  render_rgb = np.asarray(im.convert('RGB'))
  return np.concatenate([strip, render_rgb], axis=0)


def _write_mp4(frames, path: str, fps: int) -> None:
  os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
  # Prefer imageio; scrub LD_LIBRARY_PATH so system ffmpeg isn't broken by conda.
  old_ld = os.environ.pop('LD_LIBRARY_PATH', None)
  try:
    import imageio.v2 as imageio
    imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)
  finally:
    if old_ld is not None:
      os.environ['LD_LIBRARY_PATH'] = old_ld
  if not os.path.isfile(path) or os.path.getsize(path) < 1000:
    raise RuntimeError(f'video write failed: {path}')


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', default=DEFAULT_CKPT)
  p.add_argument(
      '--reward_ckpt', default='',
      help='Optional separate ckpt for φ/ψ (default: use --checkpoint).')
  p.add_argument('--tag', default='c3t1_frozen_phi_psi_successful_traj',
                 help='Output filename stem under OUT_DIR.')
  p.add_argument('--title', default='',
                 help='Title shown on the reward strip.')
  p.add_argument('--allow_no_success', action='store_true',
                 help='If no success, keep the highest-return traj.')
  p.add_argument(
      '--share_ylim_csv', default='',
      help='CSV with reward_phi_dot_psi column; lock y-limits to that curve '
           '(for fair side-by-side compare).')
  p.add_argument('--max_tries', type=int, default=MAX_TRIES)
  p.add_argument('--seed', type=int, default=SEED)
  p.add_argument('--fps', type=int, default=FPS)
  return p.parse_args()


def main():
  args = _parse_args()
  os.makedirs(OUT_DIR, exist_ok=True)
  out_mp4 = os.path.join(OUT_DIR, f'{args.tag}.mp4')
  still_path = os.path.join(OUT_DIR, f'{args.tag}_still.png')
  csv_path = os.path.join(OUT_DIR, f'{args.tag}.csv')

  print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[vid] policy_ckpt={args.checkpoint}')
  env_id = sgcrl_env_name_to_bb_env_id(ENV)
  num_cubes, _ = parse_bb_env_id(env_id)

  ctx = _load_train_ctx(ENV, args.checkpoint)
  networks = _build_networks(ENV, seed=args.seed, ctx=ctx)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)

  ckpt = ppo_learner.load_checkpoint(args.checkpoint)
  reward_ckpt_path = args.reward_ckpt.strip() or args.checkpoint
  if reward_ckpt_path == args.checkpoint:
    rew_ckpt = ckpt
  else:
    rew_ckpt = ppo_learner.load_checkpoint(reward_ckpt_path)
  q_params = rew_ckpt.get('q_params_ema') or rew_ckpt['q_params']
  q_src = ('q_params_ema' if rew_ckpt.get('q_params_ema') is not None
           else 'q_params')
  print(f'[vid] policy_iter={ckpt.get("iteration")} ep_len={ep_len} '
        f'reward_src={q_src} reward_ckpt={reward_ckpt_path}')

  ylim = None
  if args.share_ylim_csv:
    ref = np.loadtxt(args.share_ylim_csv, delimiter=',', skiprows=1)
    # columns: t, reward, success, dist
    ref_r = ref[:, 1]
    ylim = (float(ref_r.min()) - 0.35, float(ref_r.max()) + 0.35)
    print(f'[vid] shared ylim from {args.share_ylim_csv}: '
          f'[{ylim[0]:.3f}, {ylim[1]:.3f}]')

  policy = _make_policy_fn(
      networks, ckpt['policy_params'], stochastic=False,
      filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes,
      normalize_obs=False, obs_dim=ctx.obs_dim,
      start_index=ctx.start_index, end_index=ctx.end_index)

  cfg = ContrastiveConfig()
  cfg.obs_dim = int(ctx.obs_dim)
  cfg.start_index = int(ctx.start_index)
  cfg.end_index = int(ctx.end_index)
  cfg.ppo_norm_obs = False
  reward_fn = ppo_learner.make_reward_fn(networks, cfg)

  run = _compile_rollout_and_states(
      policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
      ctx.filter_policy_obs)

  key = jax.random.PRNGKey(args.seed)
  key, warm_key = jax.random.split(key)
  print('[vid] warming compile...', flush=True)
  _ = jax.block_until_ready(run(warm_key))
  print('[vid] compile done', flush=True)

  mean0 = jnp.zeros((ctx.obs_dim,), dtype=jnp.float32)
  var1 = jnp.ones((ctx.obs_dim,), dtype=jnp.float32)

  best = None
  for attempt in range(int(args.max_tries)):
    key, roll_key = jax.random.split(key)
    traj, states = run(roll_key)
    packed = np.asarray(traj['packed'], dtype=np.float32)
    actions = np.asarray(traj['action'], dtype=np.float32)
    succ = np.asarray(traj['success'], dtype=np.float32)
    rewards = np.asarray(
        reward_fn(q_params, jnp.asarray(packed), jnp.asarray(actions),
                  mean0, var1), dtype=np.float32)
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
                       '(pass --allow_no_success to keep best return)')

  rewards = best['rewards']
  success = best['success']
  states = best['states']
  T = len(rewards)
  first_succ = (int(np.argmax(success >= 0.5))
                if np.any(success >= 0.5) else -1)
  print(f'[vid] using attempt={best["attempt"]} first_success_t={first_succ} '
        f'rew_sum={rewards.sum():.3f}')

  np.savetxt(
      csv_path,
      np.stack([np.arange(T), rewards, success,
                np.full(T, np.nan, dtype=np.float32)], axis=1),
      delimiter=',',
      header='t,reward_phi_dot_psi,success,dist',
      comments='')
  print(f'[vid] wrote {csv_path}')

  # Render every macro step (1:1 with reward).
  print('[vid] rendering frames...', flush=True)
  renders = []
  for i in range(T):
    frame = env.render_from_info(
        np.asarray(states.data.qpos[i][0]),
        np.asarray(states.data.qvel[i][0]),
        np.asarray(states.info['target_mocap_pos'][i][0]),
        np.asarray(states.info['target_mocap_quat'][i][0]),
    )
    renders.append(np.asarray(frame))
  width = int(renders[0].shape[1])
  if width % 2:
    width -= 1

  title = args.title or (
      r'frozen CRL reward  $r=\varphi(s,a)\cdot\psi(g)$'
      f'  ·  {args.tag}')

  print('[vid] composing reward overlays...', flush=True)
  frames = []
  for t in range(T):
    strip = _render_reward_strip(
        rewards, success, t, width=width, height=220,
        title=title, ylim=ylim)
    composed = _compose_frame(
        strip, renders[t], t, float(rewards[t]), bool(success[t] >= 0.5))
    h, w = composed.shape[:2]
    if h % 2 or w % 2:
      composed = composed[: h - (h % 2), : w - (w % 2)]
    frames.append(composed)
    if (t + 1) % 10 == 0 or t == T - 1:
      print(f'[vid]   framed {t + 1}/{T}', flush=True)

  frames.extend([frames[-1]] * HOLD_LAST)

  print(f'[vid] writing {out_mp4} ({len(frames)} frames @ {args.fps}fps)...',
        flush=True)
  _write_mp4(frames, out_mp4, args.fps)
  still_t = min(first_succ + 2, T - 1) if first_succ >= 0 else T // 2
  Image.fromarray(frames[still_t]).save(still_path)
  print(f'[vid] wrote {out_mp4}')
  print(f'[vid] wrote {still_path}')
  print(f'[vid] size={os.path.getsize(out_mp4) / 1e6:.2f} MB')


if __name__ == '__main__':
  main()

