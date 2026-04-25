"""Load q_sac checkpoint(s) from a sweep run, roll out, save videos.

The q_sac runs are persisted by acme's `savers.CheckpointingRunner`, which
uses `tf.train.CheckpointManager` under the hood with a
`SaveableAdapter` (i.e. the whole `TrainingState` NamedTuple is pickled
into the TF checkpoint's PythonState).  This is different from the PPO
pipeline (`ppo_rollout_video.py`), which pickles `policy_params` on its
own schedule — hence a separate script rather than a branch inside the
PPO one.

Layout produced by the sweep:

  logs/q_sac_sweep/<sweep_tag>/q_sac_<env>_<seed>/
    sweep_config.json            # full ContrastiveConfig dump
    sweep_config.txt             # human-readable summary
    checkpoints/learner/ckpt-*   # TF checkpoints (the learner state)

Examples:

  # Whole run dir -> one mp4 per TF checkpoint (ckpt-1, ckpt-2, ...).
  python q_sac_rollout_video.py \
      --run_dir=logs/q_sac_sweep/her10_spi64/q_sac_sawyer_bin_2 \
      --output=videos/her10_spi64/

  # Just the latest checkpoint.
  python q_sac_rollout_video.py \
      --run_dir=logs/q_sac_sweep/her10_spi64/q_sac_sawyer_bin_2 \
      --only_latest \
      --output=videos/her10_spi64/latest.mp4

  # Sweep over every <sweep_tag>/q_sac_<env>_<seed> under a root.
  python q_sac_rollout_video.py \
      --sweep_root=logs/q_sac_sweep --only_latest \
      --output=videos/q_sac_sweep/

Frame rendering and rotation follow the same conventions as
`ppo_rollout_video.py`; see its module docstring for the details.
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede acme/jax imports

import argparse
import glob
import json
import os
import re

import numpy as np
import jax
from acme import specs

import contrastive
from contrastive import utils as contrastive_utils
import env_utils
from ppo_contrastive import fixed_goal_dict

_DEFAULT_CAMERA = {
    'sawyer_bin': 'corner',
    'sawyer_box': 'corner',
    'sawyer_peg': 'corner',
}


# ---------------------------------------------------------------------------
# Checkpoint loading (acme TF -> TrainingState -> policy_params)
# ---------------------------------------------------------------------------

def _load_training_state(ckpt_prefix: str):
  """Restore a pickled `TrainingState` from an acme TF checkpoint prefix.

  `ckpt_prefix` is the path *without* the `.index`/`.data-*` suffix, e.g.
  `.../checkpoints/learner/ckpt-8`.  acme stores the learner state by
  wrapping the learner (a `core.Saveable`) in a `SaveableAdapter`, which
  is itself a `tf.train.experimental.PythonState` that serializes with
  pickle.  To read it back without spinning up a full learner we plug in
  a tiny placeholder that exposes the same `save`/`restore` interface,
  and let the adapter stuff the unpickled `TrainingState` into it.
  """
  import tensorflow as tf
  from acme.tf.savers import SaveableAdapter
  # Ensure TrainingState is importable so pickle can resolve the class.
  from contrastive.learning import TrainingState  # noqa: F401

  class _Holder:
    state = None
    def save(self):
      return self.state
    def restore(self, s):
      self.state = s

  holder = _Holder()
  ckpt = tf.train.Checkpoint(learner=SaveableAdapter(holder))
  # `expect_partial()` silences warnings about not restoring the
  # optimizer/counter slots — we only need the params for rollouts.
  ckpt.restore(ckpt_prefix).expect_partial()
  if holder.state is None:
    raise RuntimeError(f'Failed to restore TrainingState from {ckpt_prefix!r}')
  return holder.state


def _enumerate_checkpoints(ckpt_dir: str, only_latest: bool):
  """Return an ordered list of (label, prefix) tuples for this run.

  `ckpt_dir` is the directory containing the TF checkpoints (typically
  `<run_dir>/checkpoints/learner`).  The acme manager writes `ckpt-N`
  triplets (`.index`, `.data-00000-of-00002`, ...); we key off `.index`
  files to enumerate them and sort by N ascending.
  """
  if not os.path.isdir(ckpt_dir):
    raise FileNotFoundError(f'Checkpoint dir not found: {ckpt_dir}')

  index_files = glob.glob(os.path.join(ckpt_dir, 'ckpt-*.index'))
  def _n(fname):
    m = re.search(r'ckpt-(\d+)\.index$', fname)
    return int(m.group(1)) if m else -1
  index_files.sort(key=_n)
  entries = [(f'ckpt_{_n(f):05d}', f[:-len('.index')]) for f in index_files]
  if only_latest and entries:
    entries = entries[-1:]
    # Relabel the sole entry as "latest" for nicer filenames.
    entries = [('latest', entries[0][1])]
  return entries


# ---------------------------------------------------------------------------
# Config + networks + env (mirror of ppo_rollout_video)
# ---------------------------------------------------------------------------

def _load_sweep_config(run_dir: str):
  """Read this run's config, falling back to ContrastiveConfig() defaults.

  Sweep runs (job_sac_sweep.slurm) drop a sweep_config.json that captures
  every effective hyperparameter.  Other runs — anything launched via
  lp_contrastive.py directly, including stock contrastive_cpc/c_learning
  runs under logs/sgcrl/ — don't have that file.  Rather than refuse to
  render those, fall back to the ContrastiveConfig defaults and let the
  caller pass `--env` explicitly.  This works as long as the run was
  trained with the current defaults for `repr_dim`, `repr_norm`,
  `twin_q`, `hidden_layer_sizes`, `use_image_obs` — which is the typical
  case across all the algorithms that share contrastive/make_networks.
  If you customised any of those for a non-sweep run, add a sweep_config
  .json to its run dir (see job_sac_sweep.slurm lines ~147-171 for the
  exact dump recipe) and this function will pick it up.
  """
  cfg_path = os.path.join(run_dir, 'sweep_config.json')
  if os.path.isfile(cfg_path):
    with open(cfg_path) as fh:
      return json.load(fh)

  # Fallback: populate a dict that looks like sweep_config.json using the
  # ContrastiveConfig defaults so _build_networks_from_cfg just works.
  default_cfg = contrastive.ContrastiveConfig()
  env_name = _infer_env_name_from_run_dir(run_dir)
  fallback = {
      'env_name': env_name,
      'repr_dim': default_cfg.repr_dim,
      'repr_norm': default_cfg.repr_norm,
      'twin_q': default_cfg.twin_q,
      'use_image_obs': default_cfg.use_image_obs,
      'hidden_layer_sizes': list(default_cfg.hidden_layer_sizes),
  }
  print(f'[rollout] {cfg_path} not found; using ContrastiveConfig() '
        f'defaults (env={env_name}, repr_dim={fallback["repr_dim"]}, '
        f'twin_q={fallback["twin_q"]}, '
        f'hidden_layer_sizes={fallback["hidden_layer_sizes"]}).')
  return fallback


def _infer_env_name_from_run_dir(run_dir: str):
  """Best-effort env_name inference from the run directory basename.

  Acme's naming convention (see contrastive/agents.py:44) is
  `<alg>_<env>_<seed>`, e.g. `contrastive_cpc_sawyer_bin_2`.  We strip
  the trailing `_<digits>` and then match the longest known env suffix.
  Returns None if no match — _build_networks_from_cfg will then surface
  a clearer error requiring --env on the CLI.
  """
  base = os.path.basename(os.path.normpath(run_dir))
  base = re.sub(r'_\d+$', '', base)  # strip seed suffix
  for env in sorted(fixed_goal_dict.keys(), key=len, reverse=True):
    if base.endswith(env):
      return env
  return None


def _build_networks_from_cfg(env_name, seed, cfg_dict):
  """Build contrastive networks with architecture taken from sweep_config.json.

  We intentionally do NOT use `contrastive.ContrastiveConfig()` defaults
  here: the sweep may have run with different `hidden_layer_sizes`,
  `repr_dim`, `repr_norm`, `twin_q`, etc., and the restored params must
  plug into a network with the exact same shapes.
  """
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed,
      fixed_start_end=fixed_goal_dict[env_name])
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env

  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=int(cfg_dict.get('repr_dim', 64)),
      repr_norm=bool(cfg_dict.get('repr_norm', True)),
      twin_q=bool(cfg_dict.get('twin_q', False)),
      use_image_obs=bool(cfg_dict.get('use_image_obs', False)),
      hidden_layer_sizes=tuple(cfg_dict.get('hidden_layer_sizes',
                                            (256, 256))),
  )
  return networks, obs_dim


def _get_render_fn(gym_env, width, height, camera_name, rotate_deg=180):
  from mujoco_py import MjRenderContextOffscreen

  sim = gym_env.sim
  camera_id = sim.model.camera_name2id(camera_name)

  ctx = None
  for c in getattr(sim, 'render_contexts', []) or []:
    if getattr(c, 'offscreen', False):
      ctx = c
      break
  if ctx is None:
    ctx = MjRenderContextOffscreen(sim, device_id=-1)

  k_rot = int(round((rotate_deg % 360) / 90)) % 4

  def _render():
    ctx.render(width, height, camera_id)
    rgb = ctx.read_pixels(width, height, depth=False)
    frame = rgb[::-1]
    if k_rot:
      frame = np.rot90(frame, k=k_rot)
    return np.ascontiguousarray(frame)

  return _render


def _rollout_one(policy_params, gym_env, networks, render,
                 max_steps: int, stochastic: bool, seed: int):
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
  for _ in range(max_steps):
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = policy_sample(policy_params, obs[None], k)
    else:
      action_j = policy_mode(policy_params, obs[None])
    action = np.asarray(action_j)[0].astype(np.float32)
    obs_next, r, done, _ = gym_env.step(action)
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


def _resolve_output_path(output_arg: str, env: str, run_tag: str,
                         label: str, multi_runs: bool,
                         multi_ckpts: bool) -> str:
  """Build a per-(run, checkpoint) output filename.

  Rules roughly mirror `ppo_rollout_video._resolve_output_path`, extended
  to also inject a run tag when iterating multiple sweep runs:
    - `foo.mp4` + single ckpt + single run   -> `foo.mp4`
    - `foo.mp4` + multi                      -> `foo_<run_tag>_<label>.mp4`
    - `dir/`    (or existing dir)            -> `dir/<run_tag>_<env>_<label>.mp4`
  """
  parts = [run_tag] if run_tag else []
  parts.append(env)
  parts.append(label)
  stem = '_'.join(p for p in parts if p)

  is_dir_like = output_arg.endswith(os.sep) or os.path.isdir(output_arg)
  if is_dir_like:
    return os.path.join(output_arg, f'{stem}.mp4')
  if not (multi_runs or multi_ckpts):
    return output_arg
  base, ext = os.path.splitext(output_arg)
  ext = ext or '.mp4'
  return f'{base}_{stem}{ext}'


# ---------------------------------------------------------------------------
# Sweep discovery
# ---------------------------------------------------------------------------

def _enumerate_run_dirs(run_dir: str, sweep_root: str):
  """Return (run_tag, run_path) pairs to process.

  If `--run_dir` is given, the run_tag is the sweep-cell directory name
  (e.g. `her10_spi64`) when that layout is detected, else empty.  If
  `--sweep_root` is given, every descendant that contains a
  `sweep_config.json` and a `checkpoints/learner` dir is included.
  If none match but `sweep_root` itself has `checkpoints/learner`, that path
  is treated as one run (Launchpad logs without sweep_config.json).
  """
  if run_dir:
    tag = ''
    # job_sac_sweep.slurm layout: <sweep_root>/<tag>/q_sac_<env>_<seed>/
    parent = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    if parent and parent != 'q_sac_sweep':
      tag = parent
    return [(tag, run_dir)]

  if not sweep_root:
    raise ValueError('Provide either --run_dir or --sweep_root.')

  matches = []
  for cfg_path in glob.glob(
      os.path.join(sweep_root, '**', 'sweep_config.json'), recursive=True):
    run_path = os.path.dirname(cfg_path)
    learner_dir = os.path.join(run_path, 'checkpoints', 'learner')
    if not os.path.isdir(learner_dir):
      continue
    tag = os.path.basename(os.path.dirname(run_path))
    matches.append((tag, run_path))
  matches.sort()
  if matches:
    return matches

  # `sweep_root` passed but no sweep cells: treat root as one run if it has
  # Acme learner checkpoints (contrastive_cpc / sgcrl layout, no sweep_config).
  root = os.path.normpath(sweep_root)
  if os.path.isdir(os.path.join(root, 'checkpoints', 'learner')):
    return [(os.path.basename(root), root)]
  return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser()
  src = parser.add_mutually_exclusive_group(required=True)
  src.add_argument('--run_dir',
                   help='Path to a single sweep run, e.g. '
                        'logs/q_sac_sweep/her10_spi64/q_sac_sawyer_bin_2.')
  src.add_argument('--sweep_root',
                   help='Sweep root: every descendant with sweep_config.json + '
                        'checkpoints/learner is processed. If none match but '
                        'this path itself has checkpoints/learner, it is treated '
                        'as a single run (same as --run_dir).')
  parser.add_argument('--output', required=True,
                      help='Output .mp4/.gif OR a directory (ends in / or '
                           'already exists) when rendering multiple videos.')
  parser.add_argument('--env', default=None,
                      help='Override env; otherwise read from '
                           'sweep_config.json.')
  parser.add_argument('--only_latest', action='store_true',
                      help='Render only the latest checkpoint per run.')
  parser.add_argument('--width', type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--camera', default=None)
  parser.add_argument('--rotate', type=int, default=180)
  parser.add_argument('--stochastic', action='store_true')
  parser.add_argument('--max_steps', type=int, default=-1)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  # ----- 1. Enumerate sweep runs ----------------------------------------
  run_entries = _enumerate_run_dirs(args.run_dir, args.sweep_root)
  if not run_entries:
    print('[rollout] no runs found.')
    return
  multi_runs = len(run_entries) > 1
  print(f'[rollout] found {len(run_entries)} run(s) to process.')

  # Cache env+renderer per distinct env so mujoco_py doesn't rebuild the
  # offscreen context for every checkpoint — that's the slow part.
  env_cache = {}

  for run_tag, run_dir in run_entries:
    print(f'\n[rollout] === run: {run_tag or "(single)"}  ({run_dir}) ===')
    cfg_dict = _load_sweep_config(run_dir)
    env_name = args.env or cfg_dict.get('env_name')
    if env_name is None:
      raise ValueError(f'env_name missing from {run_dir}/sweep_config.json; '
                       f'pass --env.')
    print(f'[rollout]   env={env_name}  '
          f'her_aux={cfg_dict.get("q_actor_her_aux_coef")}  '
          f'spi={cfg_dict.get("samples_per_insert")}')

    ckpt_dir = os.path.join(run_dir, 'checkpoints', 'learner')
    ckpts = _enumerate_checkpoints(ckpt_dir, only_latest=args.only_latest)
    if not ckpts:
      print(f'[rollout]   no ckpt-*.index files under {ckpt_dir}; skipping.')
      continue
    multi_ckpts = len(ckpts) > 1
    print(f'[rollout]   {len(ckpts)} checkpoint(s) to render.')

    # Build (or reuse) networks + env + render for this env.
    if env_name not in env_cache:
      print(f'[rollout]   building networks/env for {env_name}...')
      networks, _ = _build_networks_from_cfg(env_name, args.seed, cfg_dict)
      gym_env, _, env_max_steps = env_utils.load(
          env_name, fixed_start_end=fixed_goal_dict[env_name],
          seed=args.seed)
      camera = args.camera or _DEFAULT_CAMERA.get(env_name, 'corner')
      render = _get_render_fn(gym_env, args.width, args.height, camera,
                              rotate_deg=args.rotate)
      env_cache[env_name] = (networks, gym_env, env_max_steps, render)
    networks, gym_env, env_max_steps, render = env_cache[env_name]
    max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)

    for label, prefix in ckpts:
      print(f'[rollout]   --- {label}  ({prefix}) ---')
      state = _load_training_state(prefix)
      policy_params = state.policy_params

      frames, stats = _rollout_one(
          policy_params=policy_params,
          gym_env=gym_env,
          networks=networks,
          render=render,
          max_steps=max_steps,
          stochastic=args.stochastic,
          seed=args.seed,
      )
      print(f'[rollout]     length={stats["length"]}  '
            f'total_reward={stats["total_reward"]:.3f}  '
            f'success={stats["success"]}')

      out_path = _resolve_output_path(
          args.output, env_name, run_tag, label,
          multi_runs=multi_runs, multi_ckpts=multi_ckpts)
      _write_video(frames, out_path, args.fps)
      print(f'[rollout]     wrote {out_path}')


if __name__ == '__main__':
  main()
