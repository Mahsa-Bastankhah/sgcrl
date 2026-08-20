#!/usr/bin/env python3
"""Render BuilderBench rollouts with RealNVP ||z|| and log|det| probes.

Each video compares the fixed task goal against the mean of K
truncated-geometric future-goal positives, matching NF replay sampling.
"""
from __future__ import annotations

import argparse
import os
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
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from contrastive import ppo_learner
from envs.builderbench_utils import (
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)


def _load_module(name: str, path: str):
  spec = _ilu.spec_from_file_location(name, path)
  module = _ilu.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


_probe = _load_module(
    'probe_nf_flow_forward',
    os.path.join(_REPO, 'scripts', 'probe_nf_flow_forward.py'))
_nfvid = _load_module(
    'render_nf_traj_reward_video',
    os.path.join(_REPO, 'scripts', 'render_nf_traj_reward_video.py'))

_load_train_ctx = _nfvid._load_train_ctx
_make_bb_env = _nfvid._make_bb_env
_make_policy_fn = _nfvid._make_policy_fn
_compile_rollout_and_states = _nfvid._compile_rollout_and_states
_load_nf_arch = _nfvid._load_nf_arch
_load_goal_stats = _nfvid._load_goal_stats
_build_networks = _nfvid._build_networks
_run_dir_from_ckpt = _nfvid._run_dir_from_ckpt
force_video_nopermute_norand = _nfvid.force_video_nopermute_norand
_write_mp4 = _nfvid._write_mp4

DEFAULT_ENV = 'builderbench_creative_5_task2'
DEFAULT_ITERS = '600,1050,1500,3905'
DEFAULT_OUT = (
    'figs/builderbench/nf_flow_forward/'
    'c5t2_nf_compact_extrew1_s1_videos')
TASK_COLOR = '#ff6b5f'
POS_COLOR = '#5ec8ff'
SUCCESS_COLOR = '#7dffb0'
CURRENT_COLOR = '#ffe566'
HOLD_LAST = 8


def _render_compare_strip(
    task: np.ndarray,
    positive: np.ndarray,
    success: np.ndarray,
    t: int,
    *,
    width: int,
    title: str,
    ylabel: str,
    task_mean: float,
    pos_mean: float,
) -> np.ndarray:
  """Dark timeline strip with task and positive curves."""
  task = np.asarray(task, dtype=float)
  positive = np.asarray(positive, dtype=float)
  success = np.asarray(success, dtype=float)
  T = len(task)
  xs = np.arange(T)
  first = _probe._first_success(success)
  both = np.concatenate([task, positive])
  lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
  pad = 0.08 * max(hi - lo, 1e-6)

  height = 220
  dpi = 120
  fig = plt.figure(
      figsize=(width / dpi, height / dpi), dpi=dpi, facecolor='#0f1419')
  ax = fig.add_axes([0.075, 0.22, 0.69, 0.63])
  ax.set_facecolor('#0f1419')
  ax.plot(xs, task, color='#73454b', lw=1.5, alpha=0.45)
  ax.plot(xs, positive, color='#375b6b', lw=1.5, alpha=0.45)
  ax.plot(xs[:t + 1], task[:t + 1], color=TASK_COLOR, lw=2.3,
          label=rf'task  mean={task_mean:.3g}')
  ax.plot(xs[:t + 1], positive[:t + 1], color=POS_COLOR, lw=2.3,
          label=rf'positive mean={pos_mean:.3g}')
  ax.scatter([t], [task[t]], color=TASK_COLOR, edgecolors='#111820',
             linewidths=0.6, s=48, zorder=5)
  ax.scatter([t], [positive[t]], color=POS_COLOR, edgecolors='#111820',
             linewidths=0.6, s=48, zorder=5)
  ax.axvline(t, color=CURRENT_COLOR, ls=':', lw=1.0, alpha=0.8)
  if first >= 0:
    ax.axvline(first, color=SUCCESS_COLOR, ls='--', lw=1.2, alpha=0.9)
    ax.axvspan(first, T - 1, color=SUCCESS_COLOR, alpha=0.06)
  ax.set_xlim(-0.5, T - 0.5)
  ax.set_ylim(lo - pad, hi + pad)
  ax.set_xlabel('macro step t', color='#c8d0d8', fontsize=9)
  ax.set_ylabel(ylabel, color='#c8d0d8', fontsize=9)
  ax.tick_params(colors='#9aa7b5', labelsize=8)
  for spine in ax.spines.values():
    spine.set_color('#3a4654')
  ax.grid(True, color='#2a3540', alpha=0.7, lw=0.6)
  legend = ax.legend(
      fontsize=7.5, loc='best', facecolor='#0f1419', edgecolor='#3a4654')
  for text in legend.get_texts():
    text.set_color('#d9e0e7')

  txt = fig.add_axes([0.79, 0.20, 0.20, 0.66])
  txt.axis('off')
  txt.text(0.02, 0.78, 'task now', color='#b9c4cf', fontsize=8)
  txt.text(0.02, 0.60, f'{task[t]:+.3f}', color=TASK_COLOR,
           fontsize=14, fontweight='bold')
  txt.text(0.02, 0.39, 'positive now', color='#b9c4cf', fontsize=8)
  txt.text(0.02, 0.21, f'{positive[t]:+.3f}', color=POS_COLOR,
           fontsize=14, fontweight='bold')
  txt.text(
      0.02, 0.02,
      ('SUCCESS' if success[t] >= 0.5 else f't={t}/{T - 1}'),
      color=SUCCESS_COLOR if success[t] >= 0.5 else '#c8d0d8',
      fontsize=9, fontweight='bold')
  fig.suptitle(title, color='#e8eef4', fontsize=10, y=0.96)

  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if image.shape[:2] != (height, width):
    image = np.asarray(
        Image.fromarray(image).resize(
            (width, height), Image.Resampling.LANCZOS))
  return image


def _stack_frame(
    z_strip: np.ndarray,
    det_strip: np.ndarray,
    render: np.ndarray,
) -> np.ndarray:
  width = z_strip.shape[1]
  render_h = int(round(render.shape[0] * width / render.shape[1]))
  render = np.asarray(
      Image.fromarray(render).resize(
          (width, render_h), Image.Resampling.LANCZOS))
  frame = np.concatenate([z_strip, det_strip, render], axis=0)
  h, w = frame.shape[:2]
  return frame[:h - (h % 2), :w - (w % 2)]


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint_dir', required=True)
  p.add_argument('--iters', default=DEFAULT_ITERS)
  p.add_argument('--env', default=DEFAULT_ENV)
  p.add_argument('--out_dir', default=DEFAULT_OUT)
  p.add_argument('--tag_prefix', default='c5t2_nf_compact_extrew1_s1')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--n_pos', type=int, default=8)
  p.add_argument('--fps', type=int, default=8)
  p.add_argument('--fixed_start_x', type=float, default=0.1)
  p.add_argument('--skip_existing', action='store_true')
  return p.parse_args()


def main():
  args = _parse_args()
  iterations = [int(x) for x in args.iters.split(',') if x.strip()]
  entries = []
  for iteration in iterations:
    path = os.path.join(
        args.checkpoint_dir, f'ckpt_iter_{iteration:07d}.pkl')
    if not os.path.isfile(path):
      raise FileNotFoundError(path)
    entries.append((iteration, path))
  os.makedirs(args.out_dir, exist_ok=True)

  first_path = entries[0][1]
  env_id = sgcrl_env_name_to_bb_env_id(args.env)
  num_cubes, _ = parse_bb_env_id(env_id)
  ctx = _load_train_ctx(args.env, first_path)
  force_video_nopermute_norand(ctx, fixed_start_x=args.fixed_start_x)
  run_dir = _run_dir_from_ckpt(first_path)
  arch = _load_nf_arch(run_dir)
  discount = _probe._load_discount(run_dir)
  networks, nf_nets, goal_dim = _build_networks(
      args.env, args.seed, ctx, arch)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)
  forward_fn = _probe.make_nf_forward_fn(nf_nets)
  obs_dim = int(ctx.obs_dim)

  print(
      f'[flowvid] iters={iterations} K={args.n_pos} gamma={discount:g} '
      f'ep_len={ep_len}', flush=True)
  for iteration, ckpt_path in entries:
    tag = f'{args.tag_prefix}_iter_{iteration:07d}_flow_probe'
    mp4_path = os.path.join(args.out_dir, f'{tag}.mp4')
    still_path = os.path.join(args.out_dir, f'{tag}_still.png')
    csv_path = os.path.join(args.out_dir, f'{tag}.csv')
    if args.skip_existing and os.path.isfile(mp4_path):
      print(f'[flowvid] skip {mp4_path}', flush=True)
      continue

    ckpt = ppo_learner.load_checkpoint(ckpt_path)
    goal_mean, goal_std = _load_goal_stats(run_dir, iteration, goal_dim)
    goal_std = np.maximum(
        goal_std, arch['nf_goal_std_min']).astype(np.float32)
    policy = _make_policy_fn(
        networks, ckpt['policy_params'], stochastic=False,
        filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes,
        normalize_obs=False, obs_dim=ctx.obs_dim,
        start_index=ctx.start_index, end_index=ctx.end_index)
    run = _compile_rollout_and_states(
        policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
        ctx.filter_policy_obs)
    traj, states = run(jax.random.PRNGKey(args.seed))
    packed = np.asarray(traj['packed'], dtype=np.float32)
    actions = np.asarray(traj['action'], dtype=np.float32)
    success = np.asarray(traj['success'], dtype=np.float32)
    state = packed[:, :obs_dim]
    task_goal = packed[:, obs_dim:]
    achieved = _probe._achieved_goals(states, ctx)
    rng = np.random.default_rng(
        int(args.seed) * 1_000_003 + int(iteration))
    positive_goals = _probe._sample_pos_goals(
        achieved, discount, rng, args.n_pos)

    z_task, ld_task, lp_task = forward_fn(
        ckpt['q_params'], jnp.asarray(state), jnp.asarray(actions),
        jnp.asarray(task_goal), jnp.asarray(goal_mean), jnp.asarray(goal_std))
    z_pos, ld_pos, lp_pos = forward_fn(
        ckpt['q_params'], jnp.asarray(state), jnp.asarray(actions),
        jnp.asarray(positive_goals), jnp.asarray(goal_mean),
        jnp.asarray(goal_std))
    z_task = np.asarray(z_task, dtype=np.float64)
    ld_task = np.asarray(ld_task, dtype=np.float64)
    lp_task = np.asarray(lp_task, dtype=np.float64)
    z_pos = np.asarray(z_pos, dtype=np.float64)
    ld_pos = np.asarray(ld_pos, dtype=np.float64)
    lp_pos = np.asarray(lp_pos, dtype=np.float64)
    error = max(
        _probe._recon_err(z_task, ld_task, lp_task),
        _probe._recon_err(z_pos, ld_pos, lp_pos))
    if error > 1e-3:
      raise RuntimeError(f'flow reconstruction error {error}')

    z_norm_task = np.linalg.norm(z_task, axis=-1)
    z_norm_pos = np.linalg.norm(z_pos, axis=-1).mean(axis=1)
    logdet_pos = ld_pos.mean(axis=1)
    first_success = _probe._first_success(success)
    print(
        f'[flowvid] iter={iteration} first_success={first_success} '
        f'task mean||z||={z_norm_task.mean():.3f} '
        f'pos mean||z||={z_norm_pos.mean():.3f} '
        f'task mean log|det|={ld_task.mean():.3f} '
        f'pos mean log|det|={logdet_pos.mean():.3f}', flush=True)
    np.savetxt(
        csv_path,
        np.stack([
            np.arange(len(success)), success, z_norm_task, z_norm_pos,
            ld_task, logdet_pos,
        ], axis=1),
        delimiter=',',
        header=(
            't,success,z_norm_task,z_norm_pos_mean,'
            'log_absdet_task,log_absdet_pos_mean'),
        comments='')

    renders = []
    for i in range(len(success)):
      renders.append(np.asarray(env.render_from_info(
          np.asarray(states.data.qpos[i][0]),
          np.asarray(states.data.qvel[i][0]),
          np.asarray(states.info['target_mocap_pos'][i][0]),
          np.asarray(states.info['target_mocap_quat'][i][0]),
      )))
    width = int(renders[0].shape[1])
    width -= width % 2
    phase = 'pre-success checkpoint' if iteration < 1500 else (
        'post-success checkpoint')
    frames = []
    for t in range(len(success)):
      z_strip = _render_compare_strip(
          z_norm_task, z_norm_pos, success, t, width=width,
          title=(
              rf'iter {iteration} ({phase})  ·  '
              rf'$\|z\|_2=\|f(g_{{norm}}\mid s,a)\|_2$'),
          ylabel=r'$\|z\|_2$',
          task_mean=float(z_norm_task.mean()),
          pos_mean=float(z_norm_pos.mean()))
      det_strip = _render_compare_strip(
          ld_task, logdet_pos, success, t, width=width,
          title=(
              rf'iter {iteration}  ·  $\log|\det\,\partial z/\partial g_{{norm}}|$'),
          ylabel=r'$\log|\det J|$',
          task_mean=float(ld_task.mean()),
          pos_mean=float(logdet_pos.mean()))
      frames.append(_stack_frame(z_strip, det_strip, renders[t]))
      if (t + 1) % 10 == 0 or t == len(success) - 1:
        print(
            f'[flowvid] iter={iteration} framed {t + 1}/{len(success)}',
            flush=True)
    frames.extend([frames[-1]] * HOLD_LAST)
    _write_mp4(frames, mp4_path, args.fps)
    still_t = min(first_success + 2, len(success) - 1) if (
        first_success >= 0) else len(success) // 2
    Image.fromarray(frames[still_t]).save(still_path)
    print(f'[flowvid] wrote {mp4_path}', flush=True)


if __name__ == '__main__':
  main()
