"""Render one AllegroKukaThrow episode to mp4 (GPU camera, no JAX).

Used to visually check the scene: Kuka+Allegro, object on the table, fixed
bucket. Default policy is random actions (no trained checkpoint yet).

  python scripts/allegro_kuka_throw_video.py \
      --output=videos/allegro_kuka_throw_preview.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Isaac Gym must be imported before torch (handled inside the env module).
from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
import numpy as np
import torch


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--output', default='videos/allegro_kuka_throw_preview.mp4')
  p.add_argument('--num-steps', type=int, default=300)
  p.add_argument('--fps', type=int, default=30)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pipeline', default='gpu', choices=('gpu', 'cpu'))
  args = p.parse_args()

  os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

  env = AllegroKukaThrowVecEnv(
      num_envs=1,
      seed=int(args.seed),
      episode_length=int(args.num_steps),
      pipeline=str(args.pipeline),
      enable_cameras=True,
      headless=True,
  )
  env.reset()
  rng = np.random.RandomState(int(args.seed))
  frames = []
  for t in range(int(args.num_steps)):
    act = rng.uniform(-0.4, 0.4, size=(1, env.action_dim)).astype(np.float32)
    a_t = torch.as_tensor(act, device=env.device)
    env.step(a_t)
    rgb = env.render_rgb()
    if rgb is None:
      raise RuntimeError('camera returned no frame; enable_cameras failed')
    frames.append(rgb)
    if t % 50 == 0:
      print(f'[video] frame {t}/{args.num_steps} shape={rgb.shape}', flush=True)

  import imageio
  imageio.mimsave(args.output, frames, fps=int(args.fps))
  print(f'[video] wrote {len(frames)} frames -> {args.output}', flush=True)


if __name__ == '__main__':
  main()
