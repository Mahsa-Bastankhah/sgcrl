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

Deterministic actions by default (policy.mode()); pass `--stochastic`
to sample instead.  Frames are rendered offscreen via mujoco_py's
`MjRenderContextOffscreen`; on a headless cluster make sure MUJOCO_GL is
set (typically `egl` for GPU nodes, `osmesa` for CPU nodes).

The `--rotate` flag applies an N*90-degree rotation to each rendered
frame (default 180).  Mujoco_py's pixel convention plus whatever the
metaworld scene camera yields means the raw image can come out upside-
down or mirrored depending on the build; rotating in software is cheap
and robust — easier than chasing the right camera matrix.
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede acme/jax imports

import argparse
import glob
import os
import re

import numpy as np
import jax
from acme import specs

import contrastive
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
import env_utils
from ppo_contrastive import fixed_goal_dict

# Metaworld/Sawyer cameras.  'corner' works for bin/box/peg; 'topview'
# and 'behindGripper' exist too if you want a different angle.
_DEFAULT_CAMERA = {
    'sawyer_bin': 'corner',
    'sawyer_box': 'corner',
    'sawyer_peg': 'corner',
}


def _build_networks(env_name, seed):
  """Build networks matching the PPO training setup.

  The architectural hyperparameters (hidden layer sizes, repr_dim, twin_q,
  actor_min_std) must match the training-time values exactly, or the
  loaded params won't plug in cleanly.  We read them off a fresh
  ContrastiveConfig so this file stays in sync with config.py defaults.
  """
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed,
      fixed_start_end=fixed_goal_dict[env_name])
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env

  cfg = contrastive.ContrastiveConfig()
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=cfg.repr_dim,
      repr_norm=cfg.repr_norm,
      twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      actor_min_std=float(cfg.ppo_actor_min_std),
  )
  return networks, obs_dim


def _get_render_fn(gym_env, width, height, camera_name, rotate_deg=180):
  """Return a callable that produces one RGB frame per invocation.

  Uses mujoco_py's `MjRenderContextOffscreen` explicitly so we don't
  depend on whether the gym/metaworld wrapper exposes a `render` method
  with the right signature (the metaworld commit we pin has an older
  convention that differs from gym's).

  `rotate_deg` is applied to the final frame as an N*90-degree rotation
  (any multiple of 90 is accepted; other values are rounded to the
  nearest multiple).  Useful because mujoco_py's raw buffer layout plus
  the metaworld scene camera sometimes yields an inverted / mirrored
  image depending on the build.
  """
  from mujoco_py import MjRenderContextOffscreen

  sim = gym_env.sim
  camera_id = sim.model.camera_name2id(camera_name)

  # Build (or reuse) an offscreen context.  mujoco_py caches it on the sim.
  ctx = None
  for c in getattr(sim, 'render_contexts', []) or []:
    if getattr(c, 'offscreen', False):
      ctx = c
      break
  if ctx is None:
    ctx = MjRenderContextOffscreen(sim, device_id=-1)

  # Pre-compute the rotation count so the per-frame path is a single call.
  k_rot = int(round((rotate_deg % 360) / 90)) % 4   # 0,1,2,3

  def _render():
    ctx.render(width, height, camera_id)
    rgb = ctx.read_pixels(width, height, depth=False)
    # mujoco_py's read_pixels returns rows in OpenGL order (bottom-up),
    # so the baseline fix is a vertical flip.  Any additional user-
    # requested rotation is applied on top of that.
    frame = rgb[::-1]
    if k_rot:
      frame = np.rot90(frame, k=k_rot)
    return np.ascontiguousarray(frame)

  return _render


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


def _rollout_one(policy_params, gym_env, networks, render,
                 max_steps: int, stochastic: bool, seed: int):
  """Run one deterministic (or stochastic) rollout, return (frames, stats)."""
  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  @jax.jit
  def policy_sample(params, obs, rng):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample(dist, rng)

  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  frames = [render()]
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(seed)
  for t in range(max_steps):
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = policy_sample(policy_params, obs[None], k)
    else:
      action_j = policy_mode(policy_params, obs[None])
    action = np.asarray(action_j)[0].astype(np.float32)
    obs_next, r, done, info = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    total_reward += float(r)
    if float(r) > 0.0:
      success = True
    frames.append(render())
    if done:
      break
  return frames, dict(total_reward=total_reward, success=success,
                      length=len(frames))


def _write_video(frames, path: str, fps: int):
  import imageio.v2 as imageio
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  ext = os.path.splitext(path)[1].lower()
  if ext == '.gif':
    imageio.mimsave(path, frames, fps=fps)
  else:
    imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


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
                      help='Mujoco camera name; default depends on --env.')
  parser.add_argument('--rotate', type=int, default=180,
                      help='Rotate each frame by N*90 degrees before writing. '
                           'Default 180 (upright for metaworld sawyer scenes).')
  parser.add_argument('--stochastic', action='store_true',
                      help='Sample from the policy instead of using mode().')
  parser.add_argument('--max_steps', type=int, default=-1,
                      help='Override rollout length; -1 = env default.')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  # ----- 1. Enumerate checkpoints ----------------------------------------
  ckpt_entries = _enumerate_checkpoints(args.checkpoint)
  if not ckpt_entries:
    print(f'[rollout] no checkpoints found at {args.checkpoint!r}')
    return
  print(f'[rollout] found {len(ckpt_entries)} checkpoint(s) under '
        f'{args.checkpoint}')

  # ----- 2. Build networks + env + render ONCE and reuse -----------------
  # The env, mujoco context, and jitted policy graph are all reusable
  # across checkpoints — only the params change.  This avoids the
  # multi-second mujoco_py startup per checkpoint.
  print('[rollout] building networks and inferring spec...')
  networks, _ = _build_networks(args.env, seed=args.seed)

  gym_env, _, env_max_steps = env_utils.load(
      args.env, fixed_start_end=fixed_goal_dict[args.env])
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  camera = args.camera or _DEFAULT_CAMERA.get(args.env, 'corner')
  print(f'[rollout] env={args.env}  max_steps={max_steps}  '
        f'camera={camera}  rotate={args.rotate}°')

  render = _get_render_fn(gym_env, args.width, args.height, camera,
                          rotate_deg=args.rotate)

  # ----- 3. Render one video per checkpoint ------------------------------
  multi = len(ckpt_entries) > 1
  for label, path in ckpt_entries:
    print(f'[rollout] === {label}  ({path}) ===')
    ckpt = ppo_learner.load_checkpoint(path)
    policy_params = ckpt['policy_params']
    print(f'[rollout]   iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}')

    frames, stats = _rollout_one(
        policy_params=policy_params,
        gym_env=gym_env,
        networks=networks,
        render=render,
        max_steps=max_steps,
        stochastic=args.stochastic,
        seed=args.seed,
    )
    print(f'[rollout]   length={stats["length"]}  '
          f'total_reward={stats["total_reward"]:.3f}  '
          f'success={stats["success"]}')

    out_path = _resolve_output_path(args.output, args.env, label, multi)
    _write_video(frames, out_path, args.fps)
    print(f'[rollout]   wrote {out_path}')


if __name__ == '__main__':
  main()
