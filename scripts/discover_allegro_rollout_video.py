"""Roll out a DISCOVER Allegro TanhActor checkpoint to mp4.

Uses the same NVIDIA-init throw packing as training eval: actor sees
``[state_49 | g_star_3]`` with g_star from the run_config (bucket xy, goal_z).

  python scripts/discover_allegro_rollout_video.py \
      --checkpoint=logs/.../checkpoints/ckpt_epoch_0001800.pkl \
      --output=videos/discover_nvidiainit/epoch_1800.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DISC = _REPO / 'baseline-agents' / 'discover'
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_DISC))

# PhysX before JAX.
import isaacgym  # noqa: E402,F401
from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv  # noqa: E402

import sgcrl_jax_acme_compat  # noqa: E402,F401

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

import networks as nets  # noqa: E402


def _run_dir(ckpt_path: Path) -> Path:
  return ckpt_path.resolve().parent.parent


def _load_run_cfg(ckpt_path: Path) -> dict:
  cfg = _run_dir(ckpt_path) / 'run_config.json'
  if not cfg.is_file():
    return {}
  return json.loads(cfg.read_text())


def _annotate(rgb: np.ndarray, lines) -> np.ndarray:
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


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--episodes', type=int, default=2)
  p.add_argument('--num-steps', type=int, default=150)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pipeline', default='gpu')
  p.add_argument(
      '--goal-xyz', default='',
      help='Override actor goal, e.g. 0.00,0.00,0.55 for a typical UCB pick. '
           'Empty = run_config g_star (eval bucket).')
  p.add_argument(
      '--explore', action='store_true',
      help='Training collect policy: tanh mean + clipped Gaussian noise '
           '(noise=0.4, clip=0.5), then clip to [-1, 1].')
  p.add_argument('--exploration-noise', type=float, default=0.4)
  p.add_argument('--noise-clip', type=float, default=0.5)
  args = p.parse_args()

  ckpt_path = Path(args.checkpoint)
  payload = pickle.loads(ckpt_path.read_bytes())
  if 'actor_params' not in payload:
    raise SystemExit(f'not a DISCOVER ckpt (no actor_params): {ckpt_path}')
  hidden = tuple(int(x) for x in payload.get('hidden', (256, 256)))
  act_dim = int(payload.get('action_dim', 23))
  epoch = int(payload.get('epoch', 0))
  steps = int(payload.get('env_steps', 0))
  actor_params = payload['actor_params']
  run_cfg = _load_run_cfg(ckpt_path)
  if str(args.goal_xyz).strip():
    g_star = np.asarray(
        [float(x) for x in str(args.goal_xyz).split(',')],
        dtype=np.float32).reshape(3)
    goal_tag = 'UCB-style goal (override)'
  else:
    g_star = np.asarray(
        run_cfg.get('g_star_3d', [0.5, -0.3, 0.55]), dtype=np.float32).reshape(3)
    goal_tag = 'eval task goal'
  state_dim = int(run_cfg.get('state_dim', 49))

  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=int(args.seed),
      episode_length=int(args.num_steps),
      pipeline=str(args.pipeline),
      enable_cameras=True,
      headless=True,
      fixed_target_xyz=(0.50, -0.30, 0.40),
      goal_z=0.55,
      randomize_init=False,
      randomize_object_xyz=False,
      randomize_object_shape=False,
      throw_success='in_bucket',
  )
  if int(env.obs_dim) != state_dim or int(env.action_dim) != act_dim:
    raise SystemExit(
        f'env packing mismatch: obs={env.obs_dim} act={env.action_dim} '
        f'ckpt state={state_dim} act={act_dim}')

  actor_def = nets.TanhActor(action_dim=act_dim, hidden=hidden)

  @jax.jit
  def det_act(obs):
    return actor_def.apply(actor_params, obs)

  @jax.jit
  def explore_act(obs, key):
    mean = actor_def.apply(actor_params, obs)
    return nets.tanh_actor_action(
        mean, key, True, float(args.exploration_noise), float(args.noise_clip))

  device = env.device
  n_steps = int(args.num_steps)
  frames = []
  policy_tag = (
      f'train-explore n={args.exploration_noise:g} clip={args.noise_clip:g}'
      if args.explore else 'eval-det')
  print(
      f'[discover_vid] ckpt={ckpt_path} epoch={epoch} steps={steps} '
      f'goal={g_star.tolist()} ({goal_tag}) policy={policy_tag} '
      f'ep_len={n_steps} episodes={args.episodes}',
      flush=True)

  rng = jax.random.PRNGKey(int(args.seed) + 17)
  for ep in range(int(args.episodes)):
    raw = np.asarray(env.reset().detach().cpu().numpy(), dtype=np.float32)
    ever_succ = False
    act_abs = []
    for t in range(n_steps):
      state = raw[:, :state_dim]
      packed = np.concatenate(
          [state, np.broadcast_to(g_star, (1, 3))], axis=-1)
      if args.explore:
        rng, k_act = jax.random.split(rng)
        a = np.asarray(explore_act(jnp.asarray(packed), k_act))
      else:
        a = np.asarray(det_act(jnp.asarray(packed)))
      a = np.clip(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
      act_abs.append(float(np.mean(np.abs(a))))
      nxt, _, _done = env.step(torch.from_numpy(a.astype(np.float32)).to(device))
      succ = bool(float(env.success().detach().cpu().numpy().reshape(-1)[0]))
      ever_succ = ever_succ or succ
      obj = state[0, state_dim - 3:state_dim]
      dist = float(np.linalg.norm(obj - g_star))
      rgb = env.render_rgb()
      if rgb is None:
        raise RuntimeError('camera returned no frame')
      rgb = _annotate(rgb, [
          f'DISCOVER  {ckpt_path.name}  epoch={epoch}  steps={steps}',
          f'ep={ep + 1}/{args.episodes}  t={t}/{n_steps}  '
          f'succ={int(succ)}  ever={int(ever_succ)}',
          f'obj=({obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f})  '
          f'dist_task={dist:.3f}  |a|={act_abs[-1]:.3f}',
          f'g=({g_star[0]:.2f},{g_star[1]:.2f},{g_star[2]:.2f})  '
          f'({goal_tag})  {policy_tag}',
      ])
      frames.append(rgb)
      raw = np.asarray(nxt.detach().cpu().numpy(), dtype=np.float32)
    print(
        f'[discover_vid] ep={ep + 1} ever_succ={int(ever_succ)} '
        f'mean_|a|={float(np.mean(act_abs)):.4f} '
        f'max_|a|={float(np.max(act_abs)):.4f}',
        flush=True)

  import imageio
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(str(out), frames, fps=int(args.fps))
  print(f'[discover_vid] wrote {out} frames={len(frames)}', flush=True)


if __name__ == '__main__':
  main()
