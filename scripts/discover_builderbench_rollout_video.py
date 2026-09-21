"""Render a DISCOVER (Flax catselect actor) BuilderBench rollout video.

  python scripts/discover_builderbench_rollout_video.py \\
      --checkpoint=logs/.../checkpoints/latest.pkl \\
      --env=builderbench_creative_2_task1 \\
      --output=videos/discover_c2t1/
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('BUILDERBENCH_MJX_IMPL', 'jax')

_REPO = Path(__file__).resolve().parents[1]
_DISC = _REPO / 'baseline-agents' / 'discover'
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_DISC))

_BB = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import sgcrl_jax_acme_compat  # noqa: E402,F401

import jax
import jax.numpy as jnp
import numpy as np

import bb_env  # noqa: E402
import networks as nets  # noqa: E402
from contrastive.builderbench_video import (  # noqa: E402
    make_bb_video_env,
    write_video,
)
from envs.builderbench_utils import (  # noqa: E402
    apply_fixed_start_x,
    filter_pd_policy_state_obs,
)


def _step_success_series(states):
  if hasattr(states, 'metrics') and states.metrics is not None:
    succ = states.metrics.get('success')
    if succ is not None:
      arr = np.asarray(succ, dtype=np.float32)
      if arr.ndim >= 2:
        arr = arr[:, 0]
      return arr
  return None


def _episode_success_from_states(states) -> float:
  succ = _step_success_series(states)
  if succ is None:
    return float('nan')
  return float(np.max(succ))


def _overlay_success_banner(frame: np.ndarray, on: bool) -> np.ndarray:
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


def _maybe_fix_target(state, fixed_target_goal, mocap_targets, num_cubes: int):
  del num_cubes
  if fixed_target_goal is None:
    return state
  from envs.builderbench_utils import set_task_mocap_pos
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(int(fixed.shape[0] // 3), 3)
  info = dict(state.info)
  info['target_goal'] = jnp.broadcast_to(fixed, state.info['target_goal'].shape)
  info['target_mocap_pos'] = jnp.broadcast_to(
      fixed_pos, state.info['target_mocap_pos'].shape)
  mocap_pos = set_task_mocap_pos(
      state.data.mocap_pos, mocap_targets, fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--env', default='builderbench_creative_2_task1')
  p.add_argument('--output', default='videos/discover_c2t1/')
  p.add_argument('--run_tag', default='')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pd_duration', type=int, default=5)
  args = p.parse_args()

  ckpt_path = Path(args.checkpoint)
  payload = pickle.loads(ckpt_path.read_bytes())
  hidden = tuple(int(x) for x in payload.get('hidden', (256, 256)))
  env_name = str(payload.get('env', args.env))
  n_cubes = int(payload.get('n_cubes', 2))
  epoch = int(payload.get('epoch', 0))
  steps = int(payload.get('env_steps', 0))
  actor_params = payload['actor_params']

  _, num_cubes, task_index, _, _, _ = bb_env.env_layout(env_name)
  n_cubes = int(num_cubes)
  g_star = bb_env.task_goal(n_cubes, task_index)
  actor_def = nets.Actor(n_cubes=n_cubes, n_continuous=4, hidden=hidden)

  video_env, mocap_targets, macro_ep_len, num_cubes_env, filter_pd = (
      make_bb_video_env(
          env_name,
          use_pd=True,
          pd_duration=int(args.pd_duration),
          permute_start_boxes=False,
          filter_policy_obs=True,
      ))
  apply_fixed_start_x(video_env.unwrapped, 0.1)

  @jax.jit
  def _run(key):
    state = video_env.reset(jax.random.split(key, 1))
    state = _maybe_fix_target(state, g_star, mocap_targets, n_cubes)

    def f(carry, _):
      state, key = carry
      key, _ = jax.random.split(key)
      obs = filter_pd_policy_state_obs(state.obs, n_cubes)
      packed = jnp.concatenate([obs, state.info['target_goal']], axis=-1)
      mean, logits = actor_def.apply(actor_params, packed)
      action = nets.actor_det_action(mean, logits, n_cubes)
      nstate = video_env.step(state, action)
      nstate = _maybe_fix_target(nstate, g_star, mocap_targets, n_cubes)
      return (nstate, key), nstate

    _, states = jax.lax.scan(f, (state, key), (), length=int(macro_ep_len))
    return states

  print(f'[discover_vid] ckpt={ckpt_path} epoch={epoch} steps={steps} '
        f'env={env_name} ep_len={macro_ep_len}', flush=True)
  states = _run(jax.random.PRNGKey(int(args.seed)))
  jax.block_until_ready(states.data.qpos)
  step_succ = _step_success_series(states)
  frames = []
  n_on = 0
  for i in range(int(macro_ep_len)):
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
    frames.append(_overlay_success_banner(frame, on))
  ep_succ = _episode_success_from_states(states)
  out_dir = Path(args.output)
  out_dir.mkdir(parents=True, exist_ok=True)
  tag = args.run_tag or f'epoch_{epoch:07d}_s{steps}'
  out_path = out_dir / f'{tag}.mp4'
  write_video(frames, str(out_path), fps=10)
  print(f'[discover_vid] wrote {out_path} frames={len(frames)} '
        f'overlay_on={n_on} ep_success={ep_succ:.3f}', flush=True)


if __name__ == '__main__':
  main()
