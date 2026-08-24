"""Roll out a trained AllegroKukaThrow PPO policy to mp4 (GPU camera).

PhysX is created *before* JAX so Preview 4 GPU kernels still register.

  python scripts/allegro_kuka_throw_ckpt_video.py \
      --checkpoint=logs/.../checkpoints/ckpt_iter_0000100.pkl \
      --output=videos/allegro_kuka_throw_ckpt100.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _run_config_path(ckpt_path: str) -> str:
  run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
  return os.path.join(run_dir, 'run_config.json')


def _load_run_hparams(ckpt_path: str):
  cfg_path = _run_config_path(ckpt_path)
  hidden = (256, 256, 256, 256, 256, 256)
  min_std = 1e-5
  if os.path.isfile(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as fh:
      cfg = json.load(fh)
    flags = cfg.get('flags') or {}
    raw = str(flags.get('hidden_layer_sizes') or '')
    if raw.strip():
      hidden = tuple(int(x) for x in raw.split(',') if x.strip())
    if flags.get('ppo_actor_min_std') is not None:
      min_std = float(flags['ppo_actor_min_std'])
  return hidden, min_std


def _build_env(num_steps: int, seed: int, pipeline: str):
  from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
  return AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=int(seed),
      episode_length=int(num_steps),
      pipeline=str(pipeline),
      enable_cameras=True,
      headless=True,
  )


def _build_actor(env, ckpt_path: str, deterministic: bool):
  import sgcrl_jax_acme_compat  # noqa: F401
  import jax
  import jax.numpy as jnp
  import numpy as np
  from acme import specs as acme_specs
  from contrastive.networks import make_networks

  hidden, min_std = _load_run_hparams(ckpt_path)
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
  with open(ckpt_path, 'rb') as fh:
    ckpt = pickle.load(fh)
  policy_params = ckpt['policy_params']
  iteration = int(ckpt.get('iteration', -1))
  print(f'[ckpt_video] loaded {ckpt_path}  iter={iteration}  '
        f'hidden={hidden}  min_std={min_std}  det={deterministic}',
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
  return act, policy_params, jax.random.PRNGKey(0), iteration


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--output', required=True)
  p.add_argument('--num-steps', type=int, default=300)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pipeline', default='gpu', choices=('gpu', 'cpu'))
  p.add_argument('--deterministic', action='store_true')
  args = p.parse_args()

  os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

  env = _build_env(args.num_steps, args.seed, args.pipeline)
  act, policy_params, key, iteration = _build_actor(
      env, args.checkpoint, bool(args.deterministic))

  import numpy as np
  import torch
  import jax
  import jax.numpy as jnp

  obs_t = env.reset()
  frames = []
  key = jax.random.PRNGKey(int(args.seed))
  for t in range(int(args.num_steps)):
    obs = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
    key, sub = jax.random.split(key)
    action = act(policy_params, jnp.asarray(obs), sub)
    a_np = np.asarray(action, dtype=np.float32)
    a_t = torch.as_tensor(a_np, device=env.device)
    obs_t, _, _ = env.step(a_t)
    rgb = env.render_rgb()
    if rgb is None:
      raise RuntimeError('camera returned no frame; enable_cameras failed')
    frames.append(rgb)
    if t % 50 == 0:
      print(f'[ckpt_video] frame {t}/{args.num_steps} shape={rgb.shape}',
            flush=True)

  import imageio
  imageio.mimsave(args.output, frames, fps=int(args.fps))
  print(f'[ckpt_video] wrote {len(frames)} frames iter={iteration} '
        f'-> {args.output}', flush=True)


if __name__ == '__main__':
  main()
