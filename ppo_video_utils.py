"""Shared PPO rollout-and-render helpers.

Factored out so the network-reconstruction / rollout / mp4-writing logic
lives in exactly one place instead of being duplicated between the
standalone `ppo_rollout_video.py` CLI (loads a checkpoint from disk) and
the automatic end-of-training video hook inside `ppo_contrastive.py:main()`
(uses in-memory `policy_params` from the just-finished training run, no
checkpoint round-trip needed).
"""
import os

import numpy as np
import jax
from acme import specs
from acme.wrappers import gym_wrapper

import contrastive
import env_utils

# Mujoco/metaworld camera names, used only for non-ManiSkill envs.
_DEFAULT_CAMERA = {
    'sawyer_bin': 'corner',
    'sawyer_box': 'corner',
    'sawyer_peg': 'corner',
}


def build_networks(env_name, seed, hidden_layer_sizes, fixed_start_end=None,
                   config=None, render_mode=None):
  """Builds (networks, obs_dim, gym_env, max_episode_steps) for a rollout.

  `hidden_layer_sizes` MUST match the checkpoint's training-time value
  exactly, or the loaded params won't plug into the reconstructed network
  cleanly (shape mismatch).

  `fixed_start_end` is passed straight through to `env_utils.load` --
  callers should pass their own `fixed_goal_dict[env_name]` lookup
  (this module deliberately does NOT import `ppo_contrastive` itself:
  when `ppo_contrastive.py` is run as `__main__` and its own
  end-of-training video hook calls into this function, `import
  ppo_contrastive` would re-execute that file as a second, differently
  -named module, re-registering all its absl flags and raising
  `DuplicateFlagError`).

  `gym_env` is the raw (old-gym-API) environment -- callers use it
  directly for stepping/rendering; `render_mode='rgb_array'` enables
  `.render()` for ManiSkill envs (ignored by other env families).
  """
  cfg = config if config is not None else contrastive.ContrastiveConfig()

  gym_env, obs_dim, max_episode_steps = env_utils.load(
      env_name, fixed_start_end=fixed_start_end, seed=seed,
      render_mode=render_mode)
  # Wrapping (not re-building) gym_env is side-effect free: GymWrapper's
  # __init__ only converts gym.spaces to acme specs, no reset()/step() --
  # so this avoids spinning up a second, throwaway ManiSkill/SAPIEN scene
  # just to infer the spec.
  env_spec = specs.make_environment_spec(gym_wrapper.GymWrapper(gym_env))

  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=cfg.repr_dim,
      repr_norm=cfg.repr_norm,
      twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs,
      hidden_layer_sizes=tuple(hidden_layer_sizes),
      actor_min_std=float(cfg.ppo_actor_min_std),
  )
  return networks, obs_dim, gym_env, max_episode_steps


def _get_mujoco_render_fn(gym_env, width, height, camera_name, rotate_deg=180):
  """Return a callable that produces one RGB frame per invocation.

  Uses mujoco_py's `MjRenderContextOffscreen` explicitly so we don't
  depend on whether the gym/metaworld wrapper exposes a `render` method
  with the right signature (the metaworld commit we pin has an older
  convention that differs from gym's).

  `rotate_deg` is applied to the final frame as an N*90-degree rotation
  (any multiple of 90 is accepted; other values are rounded to the
  nearest multiple).  Useful because mujoco_py's raw buffer layout plus
  the metaworld scene camera sometimes yields an inverted / mirrored
  image depending on the build; rotating in software is cheap and
  robust — easier than chasing the right camera matrix.
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


def get_render_fn(env_name, gym_env, width=640, height=480, camera=None,
                  rotate_deg=180):
  """Returns a callable producing one RGB frame per invocation.

  Dispatches to ManiSkill's own `.render()` (SAPIEN; already returns a
  frame directly, no offscreen-context setup needed) for
  `env_name.startswith('maniskill_')`, else falls back to the
  MuJoCo/metaworld offscreen-rendering path.
  """
  if env_name.startswith('maniskill_'):
    return gym_env.render
  cam = camera or _DEFAULT_CAMERA.get(env_name, 'corner')
  return _get_mujoco_render_fn(gym_env, width, height, cam,
                               rotate_deg=rotate_deg)


def rollout_and_render(policy_params, gym_env, networks, render_fn,
                       max_steps, stochastic=False, seed=0,
                       obs_normalizer=None, success_fn=None):
  """Runs one rollout, returns (frames: list[np.ndarray], stats: dict).

  `obs_normalizer` (a `contrastive.ppo_learner.ObsNormalizer`, optional):
  if the checkpoint was trained with `ppo_norm_obs=True`, its policy
  expects normalized observations -- pass the normalizer reconstructed
  from that checkpoint's `obs_normalizer_state` (see
  `ppo_rollout_video.py`) or the training run's in-memory one (see
  `record_rollout_video` below) so the policy sees the same scale it was
  trained on. Frozen here: only used to normalize, never updated, so
  rendering a rollout never perturbs the saved stats.

  `success_fn` (optional): `(reward, done, info) -> bool`, checked every
  step; `stats['success']` is True if it ever returns True. Defaults to
  `reward > 0.0` -- correct for envs whose `reward` IS their success flag,
  but NOT for maniskill_close_subtask_train: that env's `reward` is
  mshab's *strict* success (joint closed AND arm at rest AND static),
  while its own training-time eval logging
  (`contrastive.utils.DrawerClosedSuccessObserver`) reads the narrower
  `info['drawer_closed']` (joint closed only) -- pass
  `success_fn=lambda r, d, info: info.get('drawer_closed', False)` there
  to match the logged metric, else this will read as far fewer successes
  than training's own eval reports.
  """
  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  @jax.jit
  def policy_sample(params, obs, rng):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample(dist, rng)

  def _norm(o):
    return obs_normalizer(o) if obs_normalizer is not None else o

  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  frames = [render_fn()]
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(seed)
  for _ in range(max_steps):
    policy_obs = np.asarray(_norm(obs), dtype=np.float32)
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = policy_sample(policy_params, policy_obs[None], k)
    else:
      action_j = policy_mode(policy_params, policy_obs[None])
    action = np.asarray(action_j)[0].astype(np.float32)
    obs_next, r, done, info = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    total_reward += float(r)
    step_success = (success_fn(r, done, info) if success_fn is not None
                    else float(r) > 0.0)
    if step_success:
      success = True
    frames.append(render_fn())
    if done:
      break
  return frames, dict(total_reward=total_reward, success=success,
                      length=len(frames))


def write_video(frames, path, fps=30):
  import imageio.v2 as imageio
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  ext = os.path.splitext(path)[1].lower()
  if ext == '.gif':
    imageio.mimsave(path, frames, fps=fps)
  else:
    imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


def record_rollout_video(env_name, seed, policy_params, hidden_layer_sizes,
                         output_path, fixed_start_end=None, max_steps=None,
                         fps=30, config=None, stochastic=False,
                         obs_normalizer=None):
  """End-to-end: build networks+env, roll out once, write an mp4.

  `obs_normalizer`: see `rollout_and_render` -- pass through when the
  policy was trained with `ppo_norm_obs=True`.

  Returns `output_path`. Raises on failure -- callers wanting best-effort
  (non-fatal) behavior, e.g. so a rendering bug never loses a completed
  training run's results, should wrap this call in their own try/except.
  """
  networks, _, gym_env, env_max_steps = build_networks(
      env_name, seed, hidden_layer_sizes, fixed_start_end=fixed_start_end,
      config=config, render_mode='rgb_array')
  try:
    render_fn = get_render_fn(env_name, gym_env)
    steps = int(max_steps) if max_steps else env_max_steps
    frames, stats = rollout_and_render(
        policy_params, gym_env, networks, render_fn, steps,
        stochastic=stochastic, seed=seed, obs_normalizer=obs_normalizer)
    write_video(frames, output_path, fps=fps)
    print(f'[ppo_video_utils] wrote {output_path} '
          f'(len={stats["length"]}, success={stats["success"]})')
  finally:
    gym_env.close()
  return output_path
