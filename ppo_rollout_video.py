"""Load PPO checkpoint(s), run one rollout each, save the videos.

Examples:
  # Single checkpoint -> single mp4.
  python ppo_rollout_video.py \
      --checkpoint=logs/ppo/ppo_sawyer_bin_0/checkpoints/latest.pkl \
      --env=sawyer_bin \
      --output=videos/sawyer_bin_0.mp4

  # A directory -> one mp4 per ckpt_iter_*.pkl (+ latest.pkl).
  python ppo_rollout_video.py \
      --checkpoint=logs/ppo/ppo_sawyer_bin_0/checkpoints \
      --env=sawyer_bin \
      --output=videos/sawyer_bin_0/

  # ManiSkill envs render via SAPIEN automatically (no --camera needed).
  python ppo_rollout_video.py \
      --checkpoint=logs/ppo/ppo_maniskill_pushcube_0/checkpoints/latest.pkl \
      --env=maniskill_pushcube \
      --output=videos/pushcube_0.mp4

Deterministic actions by default (policy.mode()); pass `--stochastic`
to sample instead.  For MuJoCo/metaworld envs, frames are rendered
offscreen via mujoco_py's `MjRenderContextOffscreen`; on a headless
cluster make sure MUJOCO_GL is set (typically `egl` for GPU nodes,
`osmesa` for CPU nodes).  ManiSkill envs render via their own SAPIEN
`.render()` (see `ppo_video_utils.get_render_fn`) -- no MUJOCO_GL needed.

The `--rotate` flag applies an N*90-degree rotation to each rendered
MuJoCo/metaworld frame (default 180; ignored for ManiSkill envs).
Mujoco_py's pixel convention plus whatever the metaworld scene camera
yields means the raw image can come out upside-down or mirrored
depending on the build; rotating in software is cheap and robust —
easier than chasing the right camera matrix.

`--hidden_layer_sizes` must match the checkpoint's training-time
architecture exactly (else loaded params won't plug in). If omitted,
it's read from the checkpoint's own stored `hidden_layer_sizes` field
(present on checkpoints saved after this field was added); older
checkpoints without it fall back to the `ContrastiveConfig` default
with a printed warning.
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports

import argparse
import glob
import os
import re

import contrastive
from contrastive import ppo_learner
from ppo_contrastive import fixed_goal_dict
import ppo_video_utils


def _enumerate_checkpoints(path: str):
  """Resolve --checkpoint to a list of (label, pkl_path) pairs.

  If `path` is a file, returns it as a single entry whose label is
  derived from the filename (`ckpt_iter_5000.pkl` → `iter_5000`,
  `latest.pkl` → `latest`).  If `path` is a directory, enumerates every
  `ckpt_iter_*.pkl` inside it in ascending iteration order, followed by
  `latest.pkl` (if present).  Files that don't match either naming
  convention are ignored; this is intentional so arbitrary pickles
  dropped in the dir don't silently get rendered.
  """
  if os.path.isfile(path):
    base = os.path.splitext(os.path.basename(path))[0]  # strip .pkl
    label = base[len('ckpt_'):] if base.startswith('ckpt_') else base
    return [(label, path)]

  if not os.path.isdir(path):
    raise FileNotFoundError(f'Checkpoint path not found: {path}')

  iter_files = glob.glob(os.path.join(path, 'ckpt_iter_*.pkl'))
  def _it(fname):
    m = re.search(r'ckpt_iter_(\d+)\.pkl$', fname)
    return int(m.group(1)) if m else -1
  iter_files.sort(key=_it)

  entries = [(f'iter_{_it(f):07d}', f) for f in iter_files]
  latest = os.path.join(path, 'latest.pkl')
  if os.path.isfile(latest):
    entries.append(('latest', latest))
  return entries


def _resolve_output_path(output_arg: str, env: str, label: str,
                         multi: bool) -> str:
  """Translate the user's --output into a per-checkpoint filename.

  Rules:
    - `--output foo.mp4` + single checkpoint  -> `foo.mp4`
    - `--output foo.mp4` + multi              -> `foo_<label>.mp4`
    - `--output dir/`    (trailing slash)     -> `dir/<env>_<label>.mp4`
    - `--output dir`     (existing dir)       -> `dir/<env>_<label>.mp4`
  This way a single `--output videos/sawyer_bin_0/` produces a clean
  `videos/sawyer_bin_0/sawyer_bin_iter_0001000.mp4`, ..., `latest.mp4`.
  """
  is_dir_like = output_arg.endswith(os.sep) or os.path.isdir(output_arg)
  if is_dir_like:
    return os.path.join(output_arg, f'{env}_{label}.mp4')
  if not multi:
    return output_arg
  stem, ext = os.path.splitext(output_arg)
  ext = ext or '.mp4'
  return f'{stem}_{label}{ext}'


def _resolve_hidden_layer_sizes(args_value: str, first_ckpt: dict):
  if args_value:
    return tuple(int(x) for x in args_value.split(','))
  stored = first_ckpt.get('hidden_layer_sizes')
  if stored is not None:
    return tuple(stored)
  default = contrastive.ContrastiveConfig().hidden_layer_sizes
  print('[rollout] WARNING: checkpoint has no hidden_layer_sizes field '
        f'(pre-dates this feature); assuming default {default}. Pass '
        '--hidden_layer_sizes explicitly if this run used a custom '
        'architecture.')
  return default


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True,
                      help='Path to a single .pkl OR a directory containing '
                           'ckpt_iter_*.pkl and/or latest.pkl.')
  parser.add_argument('--env', default='sawyer_bin',
                      choices=list(fixed_goal_dict.keys()))
  parser.add_argument('--output', required=True,
                      help='Output .mp4/.gif path, OR a directory (ends in /, '
                           'or already exists) when rendering many checkpoints.')
  parser.add_argument('--width', type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--camera', default=None,
                      help='Mujoco camera name; default depends on --env. '
                           'Ignored for ManiSkill envs.')
  parser.add_argument('--rotate', type=int, default=180,
                      help='Rotate each frame by N*90 degrees before writing. '
                           'Default 180 (upright for metaworld sawyer scenes). '
                           'Ignored for ManiSkill envs.')
  parser.add_argument('--stochastic', action='store_true',
                      help='Sample from the policy instead of using mode().')
  parser.add_argument('--max_steps', type=int, default=-1,
                      help='Override rollout length; -1 = env default.')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--hidden_layer_sizes', default='',
                      help='Comma-separated widths, e.g. '
                           '"256,256,256,256,256,256". If empty (default), '
                           'read from the checkpoint\'s stored '
                           '`hidden_layer_sizes` field; falls back to the '
                           'ContrastiveConfig default (256,256) with a '
                           'warning for older checkpoints without it.')
  args = parser.parse_args()

  # ----- 1. Enumerate checkpoints ----------------------------------------
  ckpt_entries = _enumerate_checkpoints(args.checkpoint)
  if not ckpt_entries:
    print(f'[rollout] no checkpoints found at {args.checkpoint!r}')
    return
  print(f'[rollout] found {len(ckpt_entries)} checkpoint(s) under '
        f'{args.checkpoint}')

  first_ckpt = ppo_learner.load_checkpoint(ckpt_entries[0][1])
  hidden_layer_sizes = _resolve_hidden_layer_sizes(
      args.hidden_layer_sizes, first_ckpt)
  print(f'[rollout] hidden_layer_sizes={hidden_layer_sizes}')

  # ----- 2. Build networks + env + render ONCE and reuse -----------------
  # The env, render context, and jitted policy graph are all reusable
  # across checkpoints — only the params change.
  print('[rollout] building networks and inferring spec...')
  render_mode = 'rgb_array' if args.env.startswith('maniskill_') else None
  networks, _, gym_env, env_max_steps = ppo_video_utils.build_networks(
      args.env, args.seed, hidden_layer_sizes,
      fixed_start_end=fixed_goal_dict[args.env], render_mode=render_mode)

  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  camera = args.camera
  print(f'[rollout] env={args.env}  max_steps={max_steps}  '
        f'camera={camera}  rotate={args.rotate}°')

  render = ppo_video_utils.get_render_fn(
      args.env, gym_env, args.width, args.height, camera,
      rotate_deg=args.rotate)

  # ----- 3. Render one video per checkpoint ------------------------------
  multi = len(ckpt_entries) > 1
  for label, path in ckpt_entries:
    print(f'[rollout] === {label}  ({path}) ===')
    ckpt = first_ckpt if path == ckpt_entries[0][1] else (
        ppo_learner.load_checkpoint(path))
    policy_params = ckpt['policy_params']
    print(f'[rollout]   iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}')

    _obs_norm_state = ckpt.get('obs_normalizer_state')
    obs_normalizer = (
        ppo_learner.ObsNormalizer.from_state_dict(_obs_norm_state)
        if _obs_norm_state is not None else None)
    if obs_normalizer is None:
      print('[rollout]   WARNING: checkpoint has no obs_normalizer_state '
            '(older checkpoint, or trained with ppo_norm_obs=False) -- '
            'feeding the policy raw observations.')

    frames, stats = ppo_video_utils.rollout_and_render(
        policy_params=policy_params,
        gym_env=gym_env,
        networks=networks,
        render_fn=render,
        max_steps=max_steps,
        stochastic=args.stochastic,
        seed=args.seed,
        obs_normalizer=obs_normalizer,
    )
    print(f'[rollout]   length={stats["length"]}  '
          f'total_reward={stats["total_reward"]:.3f}  '
          f'success={stats["success"]}')

    out_path = _resolve_output_path(args.output, args.env, label, multi)
    ppo_video_utils.write_video(frames, out_path, args.fps)
    print(f'[rollout]   wrote {out_path}')

  gym_env.close()


if __name__ == '__main__':
  main()
