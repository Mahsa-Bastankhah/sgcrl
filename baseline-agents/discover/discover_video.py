"""In-train DISCOVER rollout video (deterministic policy vs task g*)."""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

import networks as nets
from contrastive.builderbench_video import (
    make_bb_video_env,
    _maybe_fix_target,
    write_video,
)
from envs.builderbench_utils import apply_fixed_start_x, filter_pd_policy_state_obs


def overlay_success_banner(frame: np.ndarray, on: bool) -> np.ndarray:
  img = np.asarray(frame)
  if img.dtype != np.uint8:
    mx = float(np.max(img)) if img.size else 1.0
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if mx <= 1.0 else img.astype(np.uint8)
  out = np.ascontiguousarray(img.copy())
  bar_h = max(32, int(out.shape[0] * 0.08))
  out[:bar_h] = (34, 139, 34) if on else (55, 55, 55)
  try:
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    try:
      font = ImageFont.truetype(
          '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf', 22)
    except OSError:
      font = ImageFont.load_default()
    draw.text(
        (12, 6), 'SUCCESS = 1' if on else 'success = 0',
        fill=(255, 255, 255), font=font)
    out = np.asarray(pil)
  except Exception:
    pass
  return out


def _step_success_series(states):
  if hasattr(states, 'metrics') and states.metrics is not None:
    succ = states.metrics.get('success')
    if succ is not None:
      arr = np.asarray(succ, dtype=np.float32)
      if arr.ndim >= 2:
        arr = arr[:, 0]
      return arr
  return None


def init_video_env(
    env_name: str,
    *,
    pd_duration: int,
    permute_start_boxes: bool,
    fixed_start_x: Optional[float],
):
  video_env, mocap_targets, macro_ep_len, num_cubes, _ = make_bb_video_env(
      env_name,
      use_pd=True,
      pd_duration=int(pd_duration),
      permute_start_boxes=bool(permute_start_boxes),
      filter_policy_obs=True,
  )
  apply_fixed_start_x(video_env.unwrapped, fixed_start_x)
  return video_env, mocap_targets, int(macro_ep_len), int(num_cubes)


def make_rollout_fn(
    video_env, mocap_targets, actor_def, n_cubes, g_star, ep_len,
    categorical_select: bool = True,
    categorical_select_waypoint: bool = False):
  g_star_j = jnp.asarray(g_star, dtype=jnp.float32)
  use_wp = bool(categorical_select_waypoint)
  use_cat = bool(categorical_select) or use_wp

  @jax.jit
  def _run(params, key):
    state = video_env.reset(jax.random.split(key, 1))
    state = _maybe_fix_target(state, g_star_j, mocap_targets, n_cubes)

    def f(carry, _):
      state, key = carry
      key, _ = jax.random.split(key)
      obs = filter_pd_policy_state_obs(state.obs, n_cubes)
      packed = jnp.concatenate([obs, state.info['target_goal']], axis=-1)
      if use_wp:
        wp, yaw, logits = actor_def.apply(params, packed)
        action = nets.catwp_det_action(wp, yaw, logits, n_cubes)
      elif use_cat:
        mean, logits = actor_def.apply(params, packed)
        action = nets.actor_det_action(mean, logits, n_cubes)
      else:
        action = actor_def.apply(params, packed)
      nstate = video_env.step(state, action)
      nstate = _maybe_fix_target(nstate, g_star_j, mocap_targets, n_cubes)
      return (nstate, key), nstate

    _, states = jax.lax.scan(f, (state, key), (), length=int(ep_len))
    return states

  return _run


def write_rollout_mp4(video_env, states, out_path: str, fps: int = 10) -> tuple:
  ep_len = int(states.data.qpos.shape[0])
  step_succ = _step_success_series(states)
  frames = []
  n_on = 0
  for i in range(ep_len):
    if i % 2 != 0:
      continue
    frame = video_env.render_from_info(
        np.asarray(states.data.qpos[i][0]),
        np.asarray(states.data.qvel[i][0]),
        np.asarray(states.info['target_mocap_pos'][i][0]),
        np.asarray(states.info['target_mocap_quat'][i][0]),
    )
    on = bool(step_succ is not None and float(step_succ[i]) >= 0.5)
    n_on += int(on)
    frames.append(overlay_success_banner(frame, on))
  write_video(frames, out_path, fps=int(fps))
  ep_succ = float(np.max(step_succ)) if step_succ is not None else float('nan')
  return len(frames), n_on, ep_succ
