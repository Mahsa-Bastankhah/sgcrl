#!/usr/bin/env python3
"""One-process multi-seed Allegro ckpt videos + start-pose montage.

Isaac Gym cold-start dominates walltime, so we reuse one env and reset with
different seeds (same pattern as ``_search_best_reset`` in the single-seed
script). Stochastic policy by default (no --deterministic).

  python scripts/allegro_ckpt_init_diversity_video.py \
    --checkpoint=logs/.../checkpoints/ckpt_iter_0000700.pkl \
    --outdir=videos/.../ckpt700_stoch \
    --seeds=0,1,2,7,13,42,100,400,600,700,1100,9999 \
    --episodes=1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'scripts'))

import allegro_kuka_throw_ckpt_video as cv  # noqa: E402


def _parse_seeds(raw: str):
  parts = [p.strip() for p in str(raw).replace(' ', ',').split(',') if p.strip()]
  if not parts:
    raise ValueError('--seeds must list at least one integer')
  return [int(p) for p in parts]


def _save_montage(first_frames, seeds, out_path: str, tile_cols: int = 4):
  """Stack labeled first frames into a grid PNG (no system ffmpeg/ImageMagick)."""
  from PIL import Image, ImageDraw, ImageFont

  labeled = []
  for rgb, seed in zip(first_frames, seeds):
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    pad = 28
    canvas = Image.new('RGB', (img.width, img.height + pad), (0, 0, 0))
    canvas.paste(img, (0, pad))
    draw = ImageDraw.Draw(canvas)
    try:
      font = ImageFont.load_default()
    except Exception:
      font = None
    draw.text((8, 6), f'seed={seed}', fill=(255, 255, 255), font=font)
    labeled.append(canvas)

  cols = max(1, int(tile_cols))
  rows = (len(labeled) + cols - 1) // cols
  cell_w = max(im.width for im in labeled)
  cell_h = max(im.height for im in labeled)
  gap = 6
  grid = Image.new(
      'RGB',
      (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap),
      (17, 17, 17))
  for i, im in enumerate(labeled):
    r, c = divmod(i, cols)
    x = gap + c * (cell_w + gap)
    y = gap + r * (cell_h + gap)
    grid.paste(im, (x, y))
  os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
  grid.save(out_path)
  print(f'[init_diversity] wrote montage {out_path}', flush=True)


def _rollout_one(env, act, policy_params, flags, *, seed: int, n_steps: int,
                 ckpt_name: str, iteration: int):
  """Return (frames, first_rgb, summary_line)."""
  import jax
  import jax.numpy as jnp
  import torch

  frames = []
  key = jax.random.PRNGKey(int(seed))
  cv._set_reset_seed(int(seed))
  obs_t = env.reset()
  control_mode = str(getattr(env, 'control_sanity_mode', '') or '')
  hand_closeup = control_mode in cv._CONTROL_SANITY_JOINT_MODES
  goal_ref_rgb = None
  if hand_closeup:
    cv._zoom_camera_on_hand(env)
    goal_ref_rgb = cv._capture_goal_reference_rgb(
        env, caption_prefix=ckpt_name)
    cv._set_reset_seed(int(seed))
    obs_t = env.reset()
    cv._zoom_camera_on_hand(env)

  min_obj = 1e9
  ever_succ = False
  # Start-pose frame: right after randomized reset (before any policy step).
  d0 = cv._diag(env)
  line0 = f'seed={seed} t=  0 START mean|q-q*|={d0.get("mean_abs_joint_err", d0["d_obj"]):.3f}'
  rgb0 = env.render_rgb()
  if rgb0 is None:
    raise RuntimeError('camera returned no frame; enable_cameras failed')
  rgb0 = cv._annotate(rgb0, [ckpt_name, line0])
  rgb0 = cv._draw_goal_markers(rgb0, env, d0)
  if goal_ref_rgb is not None:
    rgb0 = cv._stack_goal_above_policy(goal_ref_rgb, rgb0)
  first_rgb = np.asarray(rgb0, dtype=np.uint8).copy()
  frames.append(rgb0)
  print(f'[init_diversity] {line0}', flush=True)

  for t in range(int(n_steps)):
    obs = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
    if obs.ndim == 1:
      obs = obs[None]
    key, sub = jax.random.split(key)
    action = act(policy_params, jnp.asarray(obs), sub)
    a_np = np.asarray(action, dtype=np.float32).reshape(-1)
    a_t = torch.as_tensor(a_np[None], device=env.device)
    obs_t, _, _ = env.step(a_t)
    d = cv._diag(env)
    min_obj = min(min_obj, d['d_obj'])
    ever_succ = ever_succ or d['succ']
    if d.get('control_mode') == 'index_thumb_straight':
      line = (
          f'seed={seed} t={t:3d} index+thumb '
          f'mean|q-q*|={d["mean_abs_joint_err"]:.3f}rad '
          f'max={d["d_obj"]:.3f}rad balanced={int(d["succ"])}')
    elif d.get('control_mode') in cv._CONTROL_SANITY_JOINT_MODES:
      line = (
          f'seed={seed} t={t:3d} finger max|q-q*|={d["d_obj"]:.3f}rad '
          f'hard={int(d["succ"])}')
    else:
      line = (
          f'seed={seed} t={t:3d} |obj-g|={d["d_obj"]:.3f}m '
          f'succ={int(d["succ"])}')
    rgb = env.render_rgb()
    if rgb is None:
      raise RuntimeError('camera returned no frame; enable_cameras failed')
    rgb = cv._annotate(rgb, [ckpt_name, line])
    rgb = cv._draw_goal_markers(rgb, env, d)
    if goal_ref_rgb is not None:
      rgb = cv._stack_goal_above_policy(goal_ref_rgb, rgb)
    frames.append(rgb)
    if t % 50 == 0:
      print(f'[init_diversity] {line}', flush=True)

  summary = (
      f'seed={seed}: min|obj-g|={min_obj:.3f}m ever_succ={int(ever_succ)} '
      f'iter={iteration}')
  print(f'[init_diversity] {summary}', flush=True)
  return frames, first_rgb, summary


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--outdir', required=True)
  p.add_argument('--seeds', default='0,1,2,7,13,42,100,400,600,700,1100,9999')
  p.add_argument('--num-steps', type=int, default=0)
  p.add_argument('--episodes', type=int, default=1,
                 help='episodes per seed (usually 1)')
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--pipeline', default='gpu', choices=('gpu', 'cpu'))
  p.add_argument('--deterministic', action='store_true')
  p.add_argument('--full-video-seeds', default='',
                 help='optional subset for full mp4s; empty = all seeds. '
                      'Other seeds still contribute first-frame montage '
                      '(rolled with --montage-steps).')
  p.add_argument('--montage-steps', type=int, default=0,
                 help='if >0 and --full-video-seeds is set, non-full seeds '
                      'only run this many steps for the montage frame')
  p.add_argument(
      '--trim-init-mode', default='',
      choices=('', 'curled', 'full_range'),
      help='override run_config trim init mode (empty = keep checkpoint)')
  args = p.parse_args()

  seeds = _parse_seeds(args.seeds)
  full_seeds = set(seeds)
  if str(args.full_video_seeds).strip():
    full_seeds = set(_parse_seeds(args.full_video_seeds))

  os.makedirs(args.outdir, exist_ok=True)
  flags = cv._load_flags(args.checkpoint)
  if str(args.trim_init_mode).strip():
    flags['isaacgym_control_sanity_trim_init_mode'] = str(
        args.trim_init_mode).strip().lower()
    print(f'[init_diversity] override trim_init_mode='
          f'{flags["isaacgym_control_sanity_trim_init_mode"]}', flush=True)
  # Build env with first seed; later seeds re-seed via _set_reset_seed.
  env_kw = cv._load_env_kwargs(
      flags, args.num_steps, seeds[0], args.pipeline)
  print(f'[init_diversity] seeds={seeds}', flush=True)
  print(f'[init_diversity] full_video_seeds={sorted(full_seeds)}', flush=True)
  print(f'[init_diversity] deterministic={bool(args.deterministic)}',
        flush=True)
  print(f'[init_diversity] env kwargs: '
        f'{ {k: env_kw[k] for k in env_kw if k != "seed"} }', flush=True)

  env = cv._build_env(env_kw)
  ckpt = cv._load_ckpt(args.checkpoint)
  act, policy_params, iteration, _networks = cv._build_actor(
      env, ckpt, flags, bool(args.deterministic))
  ckpt_name = os.path.basename(args.checkpoint)
  n_steps_full = int(env.max_episode_steps)
  n_steps_montage = int(args.montage_steps) if int(args.montage_steps) > 0 else 1

  import imageio

  first_frames = []
  for seed in seeds:
    do_full = seed in full_seeds
    n_steps = n_steps_full if do_full else n_steps_montage
    print(f'[init_diversity] === seed={seed} steps={n_steps} '
          f'full_video={do_full} ===', flush=True)
    all_frames = []
    first_rgb = None
    for ep in range(int(args.episodes)):
      # Distinct episode seeds if episodes>1; keep primary seed for ep0.
      ep_seed = int(seed) + ep
      frames, rgb0, _ = _rollout_one(
          env, act, policy_params, flags,
          seed=ep_seed, n_steps=n_steps, ckpt_name=ckpt_name,
          iteration=iteration)
      if first_rgb is None:
        first_rgb = rgb0
      all_frames.extend(frames)
      if ep + 1 < int(args.episodes):
        all_frames.extend([np.zeros_like(frames[-1])] * 8)
    first_frames.append(first_rgb)
    if do_full:
      out_mp4 = os.path.join(args.outdir, f'seed_{int(seed):04d}.mp4')
      imageio.mimsave(out_mp4, all_frames, fps=int(args.fps))
      print(f'[init_diversity] wrote {len(all_frames)} frames -> {out_mp4}',
            flush=True)
    # Always keep a first-frame PNG for inspection.
    out_png = os.path.join(args.outdir, f'seed_{int(seed):04d}_t0.png')
    imageio.imwrite(out_png, first_rgb)
    print(f'[init_diversity] wrote {out_png}', flush=True)

  montage = os.path.join(args.outdir, 'start_pose_montage.png')
  _save_montage(first_frames, seeds, montage, tile_cols=4)
  print('[init_diversity] DONE', flush=True)


if __name__ == '__main__':
  main()
