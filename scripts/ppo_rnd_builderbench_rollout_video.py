#!/usr/bin/env python3
"""Render stochastic (or deterministic) videos from builderbench PPO+RND params_*.pkl.

Run with the builderbench venv and PYTHONPATH including builderbench root, e.g.:

  python /n/fs/mislresearch/sgcrl/scripts/ppo_rnd_builderbench_rollout_video.py \\
      --checkpoint_dir=/abs/path/to/checkpoints/creative-4-task2__0__ppo-rnd__... \\
      --env_id=creative-4-task2 --output=/abs/path/out --stochastic --num_episodes=2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax
import mediapy


def _parse_args(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint_dir', required=True,
                 help='Directory containing params_*.pkl')
  p.add_argument('--env_id', default='creative-4-task2')
  p.add_argument('--output', required=True, help='Output directory for mp4s')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--fps', type=int, default=10)
  p.add_argument('--stochastic', action='store_true')
  p.add_argument('--num_episodes', type=int, default=2,
                 help='Rollouts per checkpoint (different PRNG seeds)')
  p.add_argument('--ckpt_glob', default='params_*.pkl')
  p.add_argument('--max_ckpts', type=int, default=0,
                 help='If >0, only the latest N checkpoints by index')
  p.add_argument('--skip_existing', action='store_true',
                 help='Skip outputs that already exist on disk')
  return p.parse_args(argv)


def _write_video(frames, path: str, fps: int) -> None:
  """Write mp4 via mediapy/ffmpeg with a clean LD_LIBRARY_PATH.

  Conda/CUDA libs on LD_LIBRARY_PATH break system ffmpeg (g_memdup2 / pango).
  """
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  old_ld = os.environ.pop('LD_LIBRARY_PATH', None)
  try:
    mediapy.write_video(path, frames, fps=fps)
  finally:
    if old_ld is not None:
      os.environ['LD_LIBRARY_PATH'] = old_ld
  if not os.path.isfile(path) or os.path.getsize(path) < 1000:
    raise RuntimeError(f'video write produced empty/missing file: {path}')


def main(args):
  ckpt_dir = Path(args.checkpoint_dir)
  out_dir = Path(args.output)
  out_dir.mkdir(parents=True, exist_ok=True)

  from builderbench.env_utils import make_env
  from utils.wrapper import wrap_env
  from utils.networks import load_params
  from utils.evaluation import get_video
  from ppo_rnd import Args as RndArgs, PPONetworks, Actor, Value, make_inference_fn

  rnd_args = RndArgs(env_id=args.env_id, seed=args.seed, num_envs=1)
  env_class, default_config = make_env(rnd_args)
  # BuilderBench defaults to MJX warp; match training (BUILDERBENCH_MJX_IMPL=jax).
  default_config.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  print(f'[rnd_video] MJX impl={default_config.impl} ckpt_dir={ckpt_dir}')
  env = wrap_env(env_class(config=default_config), default_config.episode_length)
  action_size = env.action_size

  # make_inference_fn only uses policy_network; other fields unused for video.
  ppo_network = PPONetworks(
      policy_network=Actor(
          layer_sizes=rnd_args.policy_hidden_sizes + [action_size * 2]),
      value_network=Value(
          layer_sizes=rnd_args.value_hidden_sizes + [1]),
      int_value_network=None,
      rnd_network=None,
  )
  make_policy = make_inference_fn(ppo_network)

  ckpts = sorted(
      ckpt_dir.glob(args.ckpt_glob),
      key=lambda p: int(p.stem.split('_')[-1]) if p.stem.split('_')[-1].isdigit()
      else p.stem,
  )
  if not ckpts:
    raise SystemExit(f'No checkpoints matching {args.ckpt_glob} in {ckpt_dir}')
  if args.max_ckpts > 0:
    ckpts = ckpts[-args.max_ckpts:]

  mode = 'stoch' if args.stochastic else 'det'
  print(f'[rnd_video] env={args.env_id} mode={mode} ep_len={default_config.episode_length}')
  print(f'[rnd_video] ckpts={[c.name for c in ckpts]}')
  print(f'[rnd_video] out={out_dir}')
  print(f'[rnd_video] jax={jax.default_backend()} devices={jax.devices()}')

  key = jax.random.PRNGKey(args.seed)
  for param_file in ckpts:
    params_tuple = load_params(str(param_file))
    params, normalize_params, _ = params_tuple
    actor_params = params['policy']
    policy_fn = jax.jit(
        make_policy(
            {'policy': actor_params, 'normalizer': normalize_params},
            deterministic=not args.stochastic,
        )
    )
    for ep in range(args.num_episodes):
      key, video_key = jax.random.split(key)
      out_path = out_dir / f'{param_file.stem}_{mode}_ep{ep}_seed{args.seed}.mp4'
      if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f'[rnd_video] skip existing {out_path.name}')
        continue
      print(f'[rnd_video] rendering {param_file.name} ep={ep} -> {out_path.name}',
            flush=True)
      frames = get_video(
          args.env_id, policy_fn, env, video_key, default_config.episode_length)
      _write_video(frames, str(out_path), args.fps)
      print(f'[rnd_video] wrote {out_path} ({len(frames)} frames)')

  print('[rnd_video] done')


if __name__ == '__main__':
  # Resolve CLI paths while still in the submit cwd, then chdir to builderbench.
  args = _parse_args()
  args.checkpoint_dir = str(Path(args.checkpoint_dir).expanduser().resolve())
  args.output = str(Path(args.output).expanduser().resolve())

  bb_root = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
  sgcrl_root = os.environ.get('SGCRL_ROOT', '/n/fs/mislresearch/sgcrl')
  baseline_agents = os.path.join(sgcrl_root, 'baseline-agents')
  for p in (bb_root, baseline_agents, sgcrl_root):
    if p and p not in sys.path:
      sys.path.insert(0, p)
  os.environ.setdefault('BUILDERBENCH_MJX_IMPL', 'jax')
  os.chdir(bb_root)
  main(args)
