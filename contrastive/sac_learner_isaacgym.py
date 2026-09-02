"""Single-process SAC + contrastive CPC on Isaac Gym (AllegroKukaThrow).

Does **not** go through Launchpad / Reverb.  PhysX is created before JAX
(see ``envs/isaacgym_physx_bootstrap.py``); this loop reuses:

  * ``IsaacGymVecEnv`` (GPU PhysX, packed 49+3 obs)
  * ``EpisodeReplay`` (same future-goal sampling as ``builder.py::flatten_fn``)
  * ``ContrastiveLearner`` (InfoNCE critic + SAC actor on φ·ψ, adaptive α)

Stock ``contrastive_cpc``: actor maximizes ``diag(Q) = φ·ψ``.  ``q_sac`` is
an optional ``reward_shaping_mode='q'`` switch on the same learner.

UTD: Launchpad uses ``samples_per_insert`` SGD samples per *finished
episode*.  With E parallel envs that is spread as
``round(E * SPI / (T * batch_size))`` ContrastiveLearner steps per PhysX
step (default ~3 at E=1024, T=300, B=256, SPI=256).  Each learner step is
one SGD (``num_sgd_steps_per_step=1``); do not copy LP's 64.
"""
from __future__ import annotations

import functools
import os
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme import types
from acme.utils import counting
from acme.utils.loggers import base as loggers_base

from contrastive import config as contrastive_config
from contrastive import learning
from contrastive import networks as contrastive_networks
from contrastive import ppo_learner_isaacgym as _ppo_ig
from contrastive import utils as contrastive_utils


def _default_updates_per_env_step(
    num_envs: int,
    max_episode_steps: int,
    batch_size: int,
    samples_per_insert: float,
) -> int:
  """Map LP SPI (samples per finished episode) onto a vec-env sim step."""
  t = max(1, int(max_episode_steps))
  b = max(1, int(batch_size))
  e = max(1, int(num_envs))
  spi = float(samples_per_insert)
  return max(1, int(round(e * spi / (float(t) * float(b)))))


def _as_float(value) -> float:
  arr = np.asarray(loggers_base.to_numpy(value))
  if arr.size == 0:
    return float('nan')
  if arr.size == 1:
    return float(arr.reshape(-1)[0])
  return float(arr.mean())


class _LastMetricsLogger:
  """Store the last ContrastiveLearner.write() payload (no CSV)."""

  def __init__(self):
    self.last: Dict[str, float] = {}

  def write(self, data):
    out = {}
    for key, value in dict(data).items():
      try:
        out[str(key)] = _as_float(value)
      except Exception:
        continue
    self.last = out

  def close(self):
    pass


class _ReplaySampleIterator:
  """Yield Acme Transitions from EpisodeReplay for ContrastiveLearner.step()."""

  def __init__(self, replay: _ppo_ig.EpisodeReplay, rng: np.random.Generator,
               batch_size: int, num_sgd_steps: int):
    self._replay = replay
    self._rng = rng
    self._n = int(batch_size) * int(num_sgd_steps)

  def __iter__(self):
    return self

  def __next__(self):
    batch = self._replay.sample(self._n, self._rng)
    bsz = int(batch['obs'].shape[0])
    data = types.Transition(
        observation=batch['obs'],
        action=batch['action'],
        reward=np.zeros((bsz,), dtype=np.float32),
        discount=np.ones((bsz,), dtype=np.float32),
        next_observation=batch['next_obs'],
        extras={'next_action': batch['next_action']},
    )
    return SimpleNamespace(data=data)


def run_sac_training(
    config: contrastive_config.ContrastiveConfig,
    network_factory: Callable,
    logger_fn: Callable,
    total_steps: int,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
    isaacgym_kwargs: Optional[Dict[str, Any]] = None,
    updates_per_env_step: int = 1,
    log_interval_sim_steps: int = 300,
    actor_min_std: float = 1e-6,
):
  """Off-policy SAC+CPC loop on a native-batched AllegroKukaThrow env."""
  from acme import specs as _specs

  env_name = str(getattr(config, 'env_name', '') or '')
  if not env_name.startswith('allegro_kuka'):
    raise ValueError(
        'sac_learner_isaacgym only supports allegro_kuka_* envs, '
        f'got {env_name!r}')

  ig_kw = dict(isaacgym_kwargs or {})
  vec_env = _ppo_ig.IsaacGymVecEnv(
      env_name=env_name,
      num_envs=int(config.ppo_num_envs),
      seed=int(seed * 31),
      isaacgym_kwargs={
          'episode_length': int(ig_kw.get('isaacgym_episode_length', 300)),
          'fixed_target_xyz': ig_kw.get(
              'isaacgym_fixed_target_xyz', (0.5, -0.3, 0.4)),
          'pipeline': str(ig_kw.get('isaacgym_pipeline', 'gpu')),
          'randomize_init': bool(ig_kw.get('isaacgym_randomize_init', True)),
          'randomize_object_shape': bool(
              ig_kw.get('isaacgym_randomize_object_shape', True)),
          'palm_goal': bool(ig_kw.get('isaacgym_palm_goal', False)),
          'palm_goal_xyz': ig_kw.get('isaacgym_palm_goal_xyz'),
          'table_push': bool(ig_kw.get('isaacgym_table_push', False)),
      },
  )
  print(
      f'[sac] Isaac Gym vec env E={vec_env.num_envs} '
      f'pipeline={ig_kw.get("isaacgym_pipeline", "gpu")} '
      f'randomize_init={bool(ig_kw.get("isaacgym_randomize_init", True))} '
      f'palm_goal={bool(ig_kw.get("isaacgym_palm_goal", False))} '
      f'table_push={bool(ig_kw.get("isaacgym_table_push", False))} '
      f'prebuilt={getattr(vec_env, "used_prebuilt", False)} '
      f'shaping={config.reward_shaping_mode!r} use_cpc={config.use_cpc}',
      flush=True)

  goal_dim = int(getattr(config, 'goal_dim', 0) or 0)
  obs_total = int(config.obs_dim) + (goal_dim if goal_dim > 0 else 3)
  act_dim = 23
  spec = _specs.EnvironmentSpec(
      observations=_specs.Array(
          shape=(obs_total,), dtype=np.float32, name='observation'),
      actions=_specs.BoundedArray(
          shape=(act_dim,), dtype=np.float32,
          minimum=-1.0, maximum=1.0, name='action'),
      rewards=_specs.Array(shape=(), dtype=np.float32, name='reward'),
      discounts=_specs.BoundedArray(
          shape=(), dtype=np.float32, minimum=0.0, maximum=1.0,
          name='discount'),
  )
  networks = network_factory(spec=spec)

  e_envs = int(vec_env.num_envs)
  t_max = int(config.max_episode_steps)
  n_sgd = max(1, int(config.num_sgd_steps_per_step))
  n_updates = max(1, int(updates_per_env_step))
  log_every = max(1, int(log_interval_sim_steps))
  min_replay = max(1, int(config.min_replay_size))
  use_random_actor = bool(getattr(config, 'use_random_actor', True))
  num_sim_steps = int(total_steps) // e_envs
  num_iterations = max(1, num_sim_steps // log_every)

  replay = _ppo_ig.EpisodeReplay(
      capacity=int(config.max_replay_size),
      obs_dim=int(config.obs_dim),
      discount=float(config.discount),
      start_index=int(config.start_index),
      end_index=int(config.end_index),
      goal_state_indices=getattr(config, 'goal_state_indices', None),
  )
  np_rng = np.random.default_rng(int(seed) + 17)
  iterator = _ReplaySampleIterator(
      replay, np_rng, int(config.batch_size), n_sgd)

  policy_optimizer = optax.adam(
      learning_rate=config.actor_learning_rate, eps=1e-7)
  q_optimizer = optax.adam(learning_rate=config.learning_rate, eps=1e-7)
  learner_metrics = _LastMetricsLogger()

  print('[sac] init: ContrastiveLearner (Haiku on CPU, then GPU)...',
        flush=True)
  with _ppo_ig._haiku_init_device(True):
    learner = learning.ContrastiveLearner(
        networks=networks,
        rng=jax.random.PRNGKey(int(seed)),
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        iterator=iterator,
        counter=counting.Counter(),
        logger=learner_metrics,
        obs_to_goal=functools.partial(
            contrastive_utils.obs_to_goal_2d,
            start_index=config.start_index,
            end_index=config.end_index),
        config=config,
    )
  learner._state = _ppo_ig._isaacgym_init_on_cpu_then_gpu(True, learner._state)

  sample_fn = contrastive_networks.apply_policy_and_sample(
      networks, eval_mode=False)

  @jax.jit
  def batched_act(params, key, obs):
    return sample_fn(params, key, obs)

  learner_logger = logger_fn(label='learner')

  obs = np.asarray(vec_env.reset(), dtype=np.float32)
  packed_dim = int(obs.shape[-1])
  ep_obs = np.zeros((e_envs, t_max + 1, packed_dim), dtype=np.float32)
  ep_act = np.zeros((e_envs, t_max, act_dim), dtype=np.float32)
  ep_len = np.zeros(e_envs, dtype=np.int32)
  ep_return = np.zeros(e_envs, dtype=np.float64)
  ep_success_max = np.zeros(e_envs, dtype=np.float64)
  ep_obs[:, 0] = obs
  rows = np.arange(e_envs)

  recent_returns: list = []
  recent_lengths: list = []
  recent_success: list = []

  start_iteration = 0
  global_step = 0
  start_sim = 0
  key = jax.random.PRNGKey(int(seed) + 1)
  compiled_update = False

  ckpt_interval = int(getattr(config, 'ppo_checkpoint_interval', 0) or 0)
  ckpt_keep_last = int(getattr(config, 'ppo_checkpoint_keep_last', 0) or 0)
  ckpt_replay_max = int(getattr(config, 'ppo_checkpoint_replay_max', 0) or 0)
  if ckpt_interval > 0 and checkpoint_dir is not None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'[sac] checkpoints → {checkpoint_dir} '
          f'every {ckpt_interval} log ticks (keep_last={ckpt_keep_last})',
          flush=True)

  if checkpoint_dir is not None:
    latest = os.path.join(checkpoint_dir, 'latest.pkl')
    if os.path.exists(latest):
      ckpt = _ppo_ig.load_checkpoint(latest)
      extra = ckpt.get('extra_state') or {}
      sac_state = extra.get('sac_state')
      if sac_state is not None:
        learner.restore(sac_state)
        learner._state = _ppo_ig._isaacgym_init_on_cpu_then_gpu(
            True, learner._state)
      else:
        learner._state = learner._state._replace(
            policy_params=ckpt['policy_params'],
            q_params=ckpt['q_params'],
        )
      start_iteration = int(ckpt['iteration']) + 1
      global_step = int(ckpt['global_step'])
      start_sim = global_step // max(1, e_envs)
      key = ckpt.get('key', key)
      if 'replay' in extra:
        if replay.load_state_dict(extra['replay']):
          print(f'[sac] resumed replay: {replay.size} transitions / '
                f'{replay.num_episodes} episodes', flush=True)
        else:
          print('[sac] replay in checkpoint incompatible; starting empty',
                flush=True)
      run_dir = os.path.dirname(checkpoint_dir)
      _ppo_ig._truncate_csv_to_iteration(
          os.path.join(run_dir, 'logs', 'learner', 'logs.csv'),
          int(ckpt['iteration']))
      print(f'[sac] resumed from checkpoint: start_iteration='
            f'{start_iteration} global_step={global_step}', flush=True)

  print(
      f'[sac] loop: sim_steps={num_sim_steps} log_every={log_every} '
      f'iterations={num_iterations} updates/step={n_updates} '
      f'batch={config.batch_size} n_sgd={n_sgd} min_replay={min_replay} '
      f'max_replay={config.max_replay_size} actor_min_std={actor_min_std} '
      f'random_actor={use_random_actor}',
      flush=True)

  start_time = time.time()
  # Resume already has a trained actor; do not drop back to uniform actions.
  n_learner_calls = 1 if start_iteration > 0 else 0
  last_entropy = float('nan')

  def _save_ckpt(iteration: int, path: str, include_replay: bool):
    extra = {'sac_state': learner._state}
    if include_replay:
      replay_state = replay.state_dict(ckpt_replay_max)
      if replay_state is not None:
        extra['replay'] = replay_state
    _ppo_ig._save_checkpoint(
        path,
        policy_params=learner._state.policy_params,
        value_params=None,
        q_params=learner._state.q_params,
        ppo_opt_state=learner._state.policy_optimizer_state,
        q_opt_state=learner._state.q_optimizer_state,
        iteration=int(iteration),
        global_step=int(global_step),
        key=key,
        extra_state=extra,
    )

  for sim in range(start_sim, num_sim_steps):
    if use_random_actor and n_learner_calls == 0:
      action = np_rng.uniform(-1.0, 1.0, size=(e_envs, act_dim)).astype(
          np.float32)
    else:
      key, k_act = jax.random.split(key)
      action = np.asarray(
          batched_act(
              learner._state.policy_params, k_act, jnp.asarray(obs)),
          dtype=np.float32)

    next_obs, env_rew, dones, terminal_obs, _info = vec_env.step(action)
    success = np.asarray(vec_env.last_success, dtype=np.float32)

    # Store a_t at current length, then the post-step obs (reset obs if done).
    overflow = ep_len >= t_max
    if np.any(overflow):
      # Horizon should already set done; force-flush to keep buffers in range.
      dones = np.logical_or(dones, overflow)

    write_t = np.minimum(ep_len, t_max - 1)
    ep_act[rows, write_t] = action
    ep_len = np.minimum(ep_len + 1, t_max)
    ep_obs[rows, ep_len] = next_obs
    ep_return += env_rew.astype(np.float64)
    ep_success_max = np.maximum(ep_success_max, success.astype(np.float64))

    done_ids = np.flatnonzero(dones)
    for i in done_ids.tolist():
      t_i = int(ep_len[i])
      t_i = min(t_i, t_max)
      obs_traj = ep_obs[i, :t_i + 1].copy()
      # Terminal frame is the pre-reset observation when Isaac Gym auto-resets.
      obs_traj[t_i] = terminal_obs[i]
      act_traj = ep_act[i, :t_i].copy()
      succeeded = bool(ep_success_max[i] >= 0.5)
      try:
        replay.add_episode(obs_traj, act_traj, successful=succeeded)
      except AssertionError:
        pass
      recent_returns.append(float(ep_return[i]))
      recent_lengths.append(int(t_i))
      recent_success.append(float(succeeded))
      if len(recent_success) > 1000:
        del recent_success[:-1000]
        del recent_returns[:-1000]
        del recent_lengths[:-1000]
      ep_obs[i, 0] = next_obs[i]
      ep_len[i] = 0
      ep_return[i] = 0.0
      ep_success_max[i] = 0.0

    obs = next_obs
    global_step += e_envs

    if replay.size >= min_replay:
      if not compiled_update:
        print('[sac] first ContrastiveLearner.step() (JIT compile)...',
              flush=True)
        t_jit = time.time()
      for _ in range(n_updates):
        learner.step()
        n_learner_calls += 1
      if not compiled_update:
        print(f'[sac] JIT compile done in {time.time() - t_jit:.1f}s',
              flush=True)
        compiled_update = True
      last_entropy = float(learner_metrics.last.get('entropy_mean', last_entropy))

    # ---- log / checkpoint on PPO-comparable ticks (T sim steps) ----
    sim_done = sim + 1
    aligned = (sim_done % log_every == 0)
    is_last = sim_done == num_sim_steps
    if not aligned and not is_last:
      continue
    iteration = ((sim_done // log_every) - 1 if aligned
                 else sim_done // log_every)
    if iteration < start_iteration:
      continue

    elapsed = time.time() - start_time
    log = {
        'iteration': float(iteration),
        'learner_steps': float(iteration),
        'global_step': float(global_step),
        'sac_sgd_steps': float(n_learner_calls * n_sgd),
        'sps': float(global_step / max(1e-6, elapsed)),
        'replay_size': float(replay.size),
        'replay_episodes': float(replay.num_episodes),
        'ep_return_mean': (
            float(np.mean(recent_returns)) if recent_returns else float('nan')),
        'ep_length_mean': (
            float(np.mean(recent_lengths)) if recent_lengths else float('nan')),
        'train_success_mean': (
            float(np.mean(recent_success[-100:])) if recent_success
            else float('nan')),
        'train_success_1000': (
            float(np.mean(recent_success[-1000:])) if recent_success
            else float('nan')),
        'random_actor': float(n_learner_calls == 0),
        'updates_per_env_step': float(n_updates),
        'actor_min_std': float(actor_min_std),
    }
    for key_m, val_m in learner_metrics.last.items():
      if key_m in ('steps', 'walltime', 'learner_steps'):
        continue
      log[key_m] = val_m
    if learner._state.alpha_params is not None:
      log['alpha'] = float(np.exp(np.asarray(
          loggers_base.to_numpy(learner._state.alpha_params))))
    learner_logger.write(log)
    print(
        f'[sac] iter={iteration} step={global_step} '
        f'sps={log["sps"]:.0f} replay={int(replay.size)} '
        f'ep_len={log["ep_length_mean"]:.1f} '
        f'succ1000={log["train_success_1000"]:.4f} '
        f'ent={log.get("entropy_mean", last_entropy):.3f}',
        flush=True)

    if (ckpt_interval > 0
        and checkpoint_dir is not None
        and (iteration % ckpt_interval == 0
             or sim_done == num_sim_steps)):
      milestone = os.path.join(
          checkpoint_dir, f'ckpt_iter_{iteration:07d}.pkl')
      _save_ckpt(iteration, milestone, include_replay=False)
      _save_ckpt(
          iteration,
          os.path.join(checkpoint_dir, 'latest.pkl'),
          include_replay=True)
      _ppo_ig._prune_old_checkpoints(checkpoint_dir, ckpt_keep_last)

  return learner
