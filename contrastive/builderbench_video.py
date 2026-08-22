"""BuilderBench deterministic rollout video helpers (in-train + offline).

Uses the CreativeCube + PD/Episode wrapper stack (same as
``scripts/ppo_builderbench_rollout_video.py``) so rendered frames match
training macro-steps. Policy actions are deterministic (``dist.mode()``)
and optionally apply the same packed-obs normalization as PPO.
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

from contrastive.ppo_learner import _normalize_packed_obs
from envs.builderbench_utils import (
    creative_cube_mj_episode_length,
    filter_pd_policy_state_obs,
    parse_bb_env_id,
    set_task_mocap_pos,
    sgcrl_env_name_to_bb_env_id,
)

from builderbench.constants import _MJX_PARAMS
from builderbench.creative_cube import CreativeCube, default_config
from utils.wrapper import (
    AutoResetWrapper,
    EpisodeWrapper,
    PDWrapper,
    VmapWrapper,
)


def make_bb_video_env(
    env_name: str,
    *,
    use_pd: bool,
    pd_duration: int,
    permute_start_boxes: bool,
    filter_policy_obs: bool,
):
  """Build a single-env batched BuilderBench stack with render helpers."""
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.episode_length = creative_cube_mj_episode_length(num_cubes, task_id)
  cfg.permute_start_boxes = bool(permute_start_boxes)
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  if env_id in _MJX_PARAMS:
    cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]

  base = CreativeCube(config=cfg)
  # Masked-in goal mocaps only (equals all mocaps when mask is all-True).
  mocap_targets = getattr(base, '_task_mocap_targets', getattr(base, '_mocap_targets', getattr(base, 'mocap_targets', None)))

  if use_pd:
    assert cfg.episode_length % pd_duration == 0, (
        f'episode_length {cfg.episode_length} must divide pd_duration '
        f'{pd_duration}')
    inner = PDWrapper(base, duration=int(pd_duration))
    macro_ep_len = cfg.episode_length // int(pd_duration)
  else:
    inner = base
    macro_ep_len = cfg.episode_length

  env = VmapWrapper(inner)
  env = EpisodeWrapper(env, episode_length=int(macro_ep_len), action_repeat=1)
  env = AutoResetWrapper(env)
  env.render_from_info = base.render_from_info  # type: ignore[attr-defined]
  env._mocap_targets_geom = base._mocap_targets_geom  # type: ignore[attr-defined]
  env.model = base._mj_model  # type: ignore[attr-defined]
  return env, mocap_targets, int(macro_ep_len), int(num_cubes), bool(filter_policy_obs)


def _maybe_fix_target(state, fixed_target_goal, mocap_targets, num_cubes: int):
  """Overwrite target goal / mocaps. ``num_cubes`` kept for call-site compat.

  Goal length may be ``3 * num_task_cubes`` when cube masks are active; reshape
  uses ``len(fixed) // 3`` rather than ``num_cubes``.
  """
  del num_cubes  # full scene cube count; goal may be shorter under masks
  if fixed_target_goal is None:
    return state
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  num_task_cubes = int(fixed.shape[0] // 3)
  fixed_pos = fixed.reshape(num_task_cubes, 3)
  info = dict(state.info)
  info['target_goal'] = jnp.broadcast_to(
      fixed, state.info['target_goal'].shape)
  info['target_mocap_pos'] = jnp.broadcast_to(
      fixed_pos, state.info['target_mocap_pos'].shape)
  mocap_pos = set_task_mocap_pos(
      state.data.mocap_pos, mocap_targets, fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


def write_video(frames, path: str, fps: int = 10):
  import imageio.v2 as imageio
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


def _render_dual_reward_strip(
    *,
    rewards_raw: np.ndarray,
    rewards_total: np.ndarray,
    success: np.ndarray,
    t: int,
    width: int,
    height: int,
    title: str = '',
    state_grad_norms: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Matplotlib dual-axis strip: raw intrinsic reward (left) vs total PPO reward (right)."""
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.backends.backend_agg import FigureCanvasAgg
  from PIL import Image

  T = len(rewards_raw)
  first_succ = int(np.argmax(success >= 0.5)) if np.any(success >= 0.5) else -1
  r_raw_now = float(rewards_raw[t])
  r_tot_now = float(rewards_total[t])
  grad_s_now = float(state_grad_norms[t]) if state_grad_norms is not None else None

  raw_min, raw_max = float(np.min(rewards_raw)), float(np.max(rewards_raw))
  if raw_min == raw_max:
    raw_min -= 1.0
    raw_max += 1.0
  else:
    span = raw_max - raw_min
    raw_min -= 0.12 * span
    raw_max += 0.12 * span

  tot_min, tot_max = float(np.min(rewards_total)), float(np.max(rewards_total))
  if tot_min == tot_max:
    tot_min -= 1.0
    tot_max += 1.0
  else:
    span = tot_max - tot_min
    tot_min -= 0.12 * span
    tot_max += 0.12 * span

  ret_tot_now = float(np.sum(rewards_total[: t + 1]))

  dpi = 120
  fig_w = width / dpi
  fig_h = height / dpi
  fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor='#0f1419')
  # 0.12 left margin, 0.54 width -> leaves 0.66 to 0.83 for right y-axis numbers
  ax_raw = fig.add_axes([0.12, 0.22, 0.54, 0.60])
  ax_raw.set_facecolor('#0f1419')

  xs = np.arange(T)
  # Left Y-axis: Raw Intrinsic Reward (Solid Cyan)
  ax_raw.plot(xs, rewards_raw, color='#5a6a7a', lw=1.5, alpha=0.4, zorder=1)
  ax_raw.plot(xs[: t + 1], rewards_raw[: t + 1], color='#00e5ff', lw=2.2, label='r_raw', zorder=2)
  ax_raw.scatter([t], [r_raw_now], s=45, color='#00e5ff', edgecolors='#0f1419', zorder=4)
  ax_raw.set_xlim(-0.5, T - 0.5)
  ax_raw.set_ylim(raw_min, raw_max)
  ax_raw.set_xlabel('macro step t', color='#c8d0d8', fontsize=8)
  ax_raw.set_ylabel('r_raw', color='#00e5ff', fontsize=8)
  ax_raw.tick_params(axis='y', colors='#00e5ff', labelsize=8)
  ax_raw.tick_params(axis='x', colors='#9aa7b5', labelsize=8)
  for spine in ax_raw.spines.values():
    spine.set_color('#3a4654')
  ax_raw.grid(True, color='#2a3540', alpha=0.6, lw=0.5)

  # Right Y-axis: Total PPO Reward (Dashed Gold)
  ax_tot = ax_raw.twinx()
  ax_tot.plot(xs, rewards_total, color='#6a5a3a', lw=1.5, ls='--', alpha=0.4, zorder=1)
  ax_tot.plot(xs[: t + 1], rewards_total[: t + 1], color='#ffea00', lw=2.2, ls='--', label='r_total', zorder=3)
  ax_tot.scatter([t], [r_tot_now], s=45, color='#ffea00', edgecolors='#0f1419', zorder=4)
  ax_tot.set_ylim(tot_min, tot_max)
  ax_tot.set_ylabel('r_total', color='#ffea00', fontsize=8)
  ax_tot.tick_params(axis='y', colors='#ffea00', labelsize=8)
  ax_tot.spines['right'].set_color('#ffea00')

  # Vertical cursor and success indicator
  ax_raw.axvline(t, color='#ffffff', ls=':', lw=1.0, alpha=0.6, zorder=3)
  if first_succ >= 0:
    ax_raw.axvline(first_succ, color='#00ff66', ls='--', lw=1.2, alpha=0.85, zorder=2)
    if t >= first_succ:
      ax_raw.scatter([first_succ], [rewards_raw[first_succ]], s=35, color='#00ff66', zorder=5)

  # Right side text panel (positioned at 0.83 to avoid overlapping right y-axis)
  ax_txt = fig.add_axes([0.83, 0.22, 0.15, 0.60])
  ax_txt.set_facecolor('#0f1419')
  ax_txt.axis('off')
  succ_now = bool(success[t] >= 0.5)
  if grad_s_now is not None:
    ax_txt.text(0.02, 0.90, 'raw reward:', transform=ax_txt.transAxes, color='#9aa7b5', fontsize=7)
    ax_txt.text(0.02, 0.76, f'{r_raw_now:+.3f}', transform=ax_txt.transAxes, color='#00e5ff', fontsize=10, fontweight='bold')
    ax_txt.text(0.02, 0.61, 'total reward:', transform=ax_txt.transAxes, color='#9aa7b5', fontsize=7)
    ax_txt.text(0.02, 0.47, f'{r_tot_now:+.3f}', transform=ax_txt.transAxes, color='#ffea00', fontsize=10, fontweight='bold')
    ax_txt.text(0.02, 0.33, f'grad_s: {grad_s_now:.3f}', transform=ax_txt.transAxes, color='#ff70a6', fontsize=8, fontweight='bold')
    ax_txt.text(0.02, 0.21, f'return: {ret_tot_now:+.1f}', transform=ax_txt.transAxes, color='#7dffb0', fontsize=7)
    ax_txt.text(0.02, 0.10, f'step: {t}/{T-1}', transform=ax_txt.transAxes, color='#c8d0d8', fontsize=7)
    ax_txt.text(0.02, 0.00, 'SUCCESS' if succ_now else 'running', transform=ax_txt.transAxes, color='#00ff66' if succ_now else '#ff9900', fontsize=8, fontweight='bold')
  else:
    ax_txt.text(0.02, 0.88, 'raw reward:', transform=ax_txt.transAxes, color='#9aa7b5', fontsize=7)
    ax_txt.text(0.02, 0.73, f'{r_raw_now:+.3f}', transform=ax_txt.transAxes, color='#00e5ff', fontsize=11, fontweight='bold')
    ax_txt.text(0.02, 0.55, 'total reward:', transform=ax_txt.transAxes, color='#9aa7b5', fontsize=7)
    ax_txt.text(0.02, 0.40, f'{r_tot_now:+.3f}', transform=ax_txt.transAxes, color='#ffea00', fontsize=11, fontweight='bold')
    ax_txt.text(0.02, 0.25, f'return: {ret_tot_now:+.1f}', transform=ax_txt.transAxes, color='#7dffb0', fontsize=8)
    ax_txt.text(0.02, 0.12, f'step: {t}/{T-1}', transform=ax_txt.transAxes, color='#c8d0d8', fontsize=8)
    ax_txt.text(0.02, 0.00, 'SUCCESS' if succ_now else 'running', transform=ax_txt.transAxes, color='#00ff66' if succ_now else '#ff9900', fontsize=9, fontweight='bold')

  fig.suptitle(title or 'Dual Reward Timeline (Lockstep)', color='#e8eef4', fontsize=9, y=0.96)

  canvas = FigureCanvasAgg(fig)
  canvas.draw()
  buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
  plt.close(fig)
  if buf.shape[1] != width or buf.shape[0] != height:
    from PIL import Image
    buf = np.asarray(Image.fromarray(buf).resize((width, height), Image.Resampling.LANCZOS))
  return buf



def _compose_frame(strip: np.ndarray, render: np.ndarray) -> np.ndarray:
  """Stack strip vertically above render, adjusting width if needed."""
  from PIL import Image
  rw = strip.shape[1]
  if render.shape[1] != rw:
    rh = int(round(render.shape[0] * (rw / render.shape[1])))
    render = np.asarray(Image.fromarray(render).resize((rw, rh), Image.Resampling.LANCZOS))
  return np.concatenate([strip, render], axis=0)


def render_deterministic_episode(
    *,
    video_env,
    mocap_targets,
    num_cubes: int,
    episode_length: int,
    networks,
    policy_params,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    fixed_target_goal: Optional[np.ndarray],
    seed: int,
    filter_policy_obs: bool,
    normalize_obs: bool,
    obs_dim: int,
    start_index: int,
    end_index: int,
    obs_norm_clip: float = 10.0,
    fps: int = 10,
    out_path: str,
    goal_state_indices=None,
    reward_fn=None,
    reward_params=None,
    grad_norm_fn=None,
    reward_normalizer=None,
    use_external_reward: bool = False,
    external_reward_scale: float = 100.0,
    external_reward_before_norm: bool = False,
    include_reward_plot: bool = True,
    title: str = '',
) -> Tuple[str, int]:
  """Roll out one deterministic episode with live obs stats and write mp4.

  Returns ``(out_path, n_frames)``.
  """
  obs_mean_j = jnp.asarray(obs_mean, dtype=jnp.float32)
  obs_var_j = jnp.asarray(obs_var, dtype=jnp.float32)
  _filter = bool(filter_policy_obs)
  _norm = bool(normalize_obs)
  _obs_dim = int(obs_dim)
  _si = int(start_index)
  _ei = int(end_index if end_index != -1 else obs_dim)
  _clip = float(obs_norm_clip)
  _gidx = goal_state_indices
  policy_apply = networks.policy_network.apply

  @jax.jit
  def _policy(obs, goals, obs_mean, obs_var):
    if _filter:
      obs = filter_pd_policy_state_obs(obs, num_cubes)
    packed = jnp.concatenate([obs, goals], axis=-1)
    packed = _normalize_packed_obs(
        packed, obs_mean, obs_var,
        obs_dim=_obs_dim, start_index=_si, end_index=_ei,
        clip=_clip, enabled=_norm, goal_state_indices=_gidx)
    dist = policy_apply(policy_params, packed)
    return dist.mode()

  @jax.jit
  def _run(key, obs_mean, obs_var):
    env_key, key = jax.random.split(key)
    state = video_env.reset(jax.random.split(env_key, 1))
    state = _maybe_fix_target(
        state, fixed_target_goal, mocap_targets, num_cubes)

    def f(carry, _):
      state, key = carry
      key, _ = jax.random.split(key)
      action = _policy(
          state.obs, state.info['target_goal'], obs_mean, obs_var)
      next_state = video_env.step(state, action)
      next_state = _maybe_fix_target(
          next_state, fixed_target_goal, mocap_targets, num_cubes)
      return (next_state, key), (next_state, action, state.obs, state.info['target_goal'])

    _, (states, actions, raw_obs, goals) = jax.lax.scan(
        f, (state, key), (), length=int(episode_length))
    return states, actions, raw_obs, goals

  states, actions, raw_obs, goals = _run(jax.random.PRNGKey(int(seed)), obs_mean_j, obs_var_j)
  jax.block_until_ready(states.data.qpos)

  T_steps = int(episode_length)
  successes = np.asarray(states.metrics['success'][:, 0] >= 0.5, dtype=np.float32)

  # Compute per-step rewards and state gradient norms for lockstep plot if requested
  state_grad_norms = None
  if include_reward_plot:
    if reward_fn is not None and reward_params is not None:
      if _filter:
        p_obs = filter_pd_policy_state_obs(raw_obs[:, 0], num_cubes)
      else:
        p_obs = raw_obs[:, 0]
      packed_traj = jnp.concatenate([p_obs, goals[:, 0]], axis=-1)
      try:
        r_raw_j = reward_fn(reward_params, packed_traj, actions[:, 0])
        rewards_raw = np.asarray(r_raw_j, dtype=np.float32).reshape(-1)
      except Exception:
        rewards_raw = np.asarray(states.reward[:, 0], dtype=np.float32)

      if grad_norm_fn is not None:
        try:
          s_norms_j, _ = grad_norm_fn(reward_params, packed_traj, actions[:, 0])
          state_grad_norms = np.asarray(s_norms_j, dtype=np.float32).reshape(-1)
        except Exception:
          state_grad_norms = None
    else:
      rewards_raw = np.asarray(states.reward[:, 0], dtype=np.float32)

    rewards_total = np.copy(rewards_raw)
    if reward_normalizer is not None and getattr(reward_normalizer, 'std', None) is not None:
      std_val = float(reward_normalizer.std)
      if std_val > 1e-8:
        rewards_total = rewards_total / std_val
    if use_external_reward:
      rewards_total += float(external_reward_scale) * successes
  else:
    rewards_raw = None
    rewards_total = None

  frames = []
  for i in range(T_steps):
    if i % 2 != 0:
      continue
    render_img = video_env.render_from_info(
        np.asarray(states.data.qpos[i][0]),
        np.asarray(states.data.qvel[i][0]),
        np.asarray(states.info['target_mocap_pos'][i][0]),
        np.asarray(states.info['target_mocap_quat'][i][0]),
    )
    if include_reward_plot and rewards_raw is not None and rewards_total is not None:
      strip = _render_dual_reward_strip(
          rewards_raw=rewards_raw,
          rewards_total=rewards_total,
          success=successes,
          t=i,
          width=render_img.shape[1],
          height=int(round(render_img.shape[0] * 0.45)),
          title=title,
          state_grad_norms=state_grad_norms,
      )
      frame = _compose_frame(strip, render_img)
    else:
      frame = render_img
    frames.append(frame)

  write_video(frames, out_path, fps=int(fps))
  return out_path, len(frames)


def render_rollout_trajectory_video(
    *,
    video_env,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_mocap_pos: np.ndarray,
    target_mocap_quat: np.ndarray,
    rewards_raw: np.ndarray,
    rewards_total: np.ndarray,
    success: np.ndarray,
    fps: int = 10,
    out_path: str,
    include_reward_plot: bool = True,
    title: str = '',
    state_grad_norms: Optional[np.ndarray] = None,
) -> Tuple[str, int]:
  """Render a recorded rollout trajectory (e.g. from an on-policy training env) with lockstep dual-reward strip."""
  T_steps = len(qpos)
  frames = []
  for i in range(T_steps):
    if i % 2 != 0:
      continue
    render_img = video_env.render_from_info(
        np.asarray(qpos[i]),
        np.asarray(qvel[i]),
        np.asarray(target_mocap_pos[i]),
        np.asarray(target_mocap_quat[i]),
    )
    if include_reward_plot and rewards_raw is not None and rewards_total is not None:
      strip = _render_dual_reward_strip(
          rewards_raw=rewards_raw,
          rewards_total=rewards_total,
          success=success,
          t=i,
          width=render_img.shape[1],
          height=int(round(render_img.shape[0] * 0.45)),
          title=title,
          state_grad_norms=state_grad_norms,
      )
      frame = _compose_frame(strip, render_img)
    else:
      frame = render_img
    frames.append(frame)

  write_video(frames, out_path, fps=int(fps))
  return out_path, len(frames)

