"""Standalone GPU-batched SAC + Contrastive RL (SGCRL) learner implementation.

Combines:
  1. Off-policy SAC actor and InfoNCE representation critic from contrastive RL.
  2. Native GPU-batched vectorized environment stepping (ManiskillVecEnv).
  3. In-memory episodic replay buffer with geometric future hindsight goal relabeling (HER).
  4. Optax optimizers and JIT-compiled updates for fast training in JAX.
"""
import os
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

from acme import specs as _specs
from acme.jax import networks as networks_lib
import contrastive
from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
from contrastive import utils as _cu
from contrastive.ppo_learner import EpisodeReplay, VecEnv
import env_utils as _env_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax


# ---------------------------------------------------------------------------
# Training State
# ---------------------------------------------------------------------------
class SACTrainingState(NamedTuple):
  """Contains training state for the SAC + CRL learner."""
  policy_params: networks_lib.Params
  policy_opt_state: optax.OptState
  q_params: networks_lib.Params
  q_opt_state: optax.OptState
  alpha_params: Optional[networks_lib.Params]
  alpha_opt_state: Optional[optax.OptState]
  key: networks_lib.PRNGKey


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------
def _save_checkpoint(
    path: str,
    training_state: SACTrainingState,
    iteration: int,
    global_step: int,
    gradient_steps: int,
    hidden_layer_sizes=None,
):
  """Writes a pickle checkpoint atomically."""
  import pickle as _pkl
  ckpt = {
      'policy_params': training_state.policy_params,
      'policy_opt_state': training_state.policy_opt_state,
      'q_params': training_state.q_params,
      'q_opt_state': training_state.q_opt_state,
      'alpha_params': training_state.alpha_params,
      'alpha_opt_state': training_state.alpha_opt_state,
      'key': training_state.key,
      'iteration': int(iteration),
      'global_step': int(global_step),
      'gradient_steps': int(gradient_steps),
      'hidden_layer_sizes': (
          tuple(hidden_layer_sizes) if hidden_layer_sizes is not None else None
      ),
  }
  tmp_path = path + '.tmp'
  with open(tmp_path, 'wb') as fh:
    _pkl.dump(ckpt, fh, protocol=_pkl.HIGHEST_PROTOCOL)
  os.replace(tmp_path, path)


def _prune_old_checkpoints(ckpt_dir: str, keep_last: int):
  """Deletes all but the `keep_last` most recent ckpt_iter_*.pkl files."""
  if keep_last <= 0:
    return
  files = [
      f for f in os.listdir(ckpt_dir)
      if f.startswith('ckpt_iter_') and f.endswith('.pkl')
  ]

  def _iter_of(fname):
    try:
      return int(fname[len('ckpt_iter_'):-len('.pkl')])
    except ValueError:
      return -1

  files.sort(key=_iter_of)
  to_remove = files[:-keep_last] if len(files) > keep_last else []
  for f in to_remove:
    try:
      os.remove(os.path.join(ckpt_dir, f))
    except OSError:
      pass


# ---------------------------------------------------------------------------
# JIT-Compiled SAC + CRL Update Functions
# ---------------------------------------------------------------------------
def make_sac_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    alpha_optimizer: Optional[optax.GradientTransformation],
):
  """Builds JIT-compiled single and scanned update functions for SAC + CRL."""
  batch_size = int(config.batch_size)
  I = jnp.eye(batch_size)
  use_adaptive_alpha = alpha_optimizer is not None
  target_entropy = float(config.target_entropy)

  def critic_loss_fn(q_params, batch):
    obs = batch['obs']
    actions = batch['action']
    logits, phi_sa, psi_g = networks.q_network.apply(q_params, obs, actions)

    if 'extra_goals' in batch:
      # Uniform negative goals: (B, num_negatives, goal_dim)
      extra_goals = batch['extra_goals']
      B, K, G_dim = extra_goals.shape
      flat_extra = extra_goals.reshape(B * K, G_dim)
      dummy_state = jnp.zeros((B * K, config.obs_dim), dtype=flat_extra.dtype)
      dummy_obs = jnp.concatenate([dummy_state, flat_extra], axis=1)
      dummy_act = jnp.zeros((B * K,) + actions.shape[1:], dtype=actions.dtype)
      _, _, extra_psi = networks.q_network.apply(q_params, dummy_obs, dummy_act)
      extra_psi = extra_psi.reshape(B, K, -1)

      if len(logits.shape) == 3:
        # twin_q
        extra_logits1 = jnp.einsum('ik,ijk->ij', phi_sa[:, :, 0], extra_psi)
        extra_logits2 = jnp.einsum('ik,ijk->ij', phi_sa[:, :, 1], extra_psi)
        extra_logits = jnp.stack([extra_logits1, extra_logits2], axis=-1)
        full_logits = jnp.concatenate([logits, extra_logits], axis=1)
      else:
        extra_logits = jnp.einsum('ik,ijk->ij', phi_sa, extra_psi)
        full_logits = jnp.concatenate([logits, extra_logits], axis=1)
    else:
      full_logits = logits

    def _loss_single(_l):
      return (
          optax.softmax_cross_entropy(logits=_l, labels=I)
          + 0.01 * jax.nn.logsumexp(_l, axis=1) ** 2
      )

    if len(full_logits.shape) == 3:
      loss = jax.vmap(_loss_single, in_axes=2, out_axes=-1)(full_logits)
      loss = jnp.mean(loss)
      eval_logits = jnp.mean(full_logits, axis=-1)
    else:
      loss = jnp.mean(_loss_single(full_logits))
      eval_logits = full_logits

    logits_cpc = eval_logits[:, :batch_size]
    correct = jnp.argmax(logits_cpc, axis=1) == jnp.argmax(I, axis=1)
    logits_pos = jnp.sum(logits_cpc * I) / jnp.sum(I)
    logits_neg = jnp.sum(logits_cpc * (1.0 - I)) / jnp.sum(1.0 - I)
    logsumexp_val = jax.nn.logsumexp(eval_logits, axis=1) ** 2

    metrics = {
        'crl_loss': loss,
        'categorical_accuracy': jnp.mean(correct),
        'binary_accuracy': jnp.mean((logits_cpc > 0) == I),
        'logits_pos': logits_pos,
        'logits_neg': logits_neg,
        'logsumexp': jnp.mean(logsumexp_val),
        'repr_phi_norm_mean': jnp.mean(jnp.linalg.norm(phi_sa, axis=-1)),
        'repr_psi_norm_mean': jnp.mean(jnp.linalg.norm(psi_g, axis=-1)),
    }
    return loss, metrics

  def actor_loss_fn(policy_params, q_params, alpha_val, batch, key):
    obs = batch['obs']
    dist = networks.policy_network.apply(policy_params, obs)
    action = networks.sample(dist, key)
    log_prob = networks.log_prob(dist, action)

    # Evaluate action under frozen CRL critic
    q_val, _, _ = networks.q_network.apply(
        jax.lax.stop_gradient(q_params), obs, action
    )
    if len(q_val.shape) == 3:
      q_val = jnp.min(q_val, axis=-1)
    q_diag = jnp.diag(q_val)

    actor_loss = jnp.mean(alpha_val * log_prob - q_diag)
    metrics = {
        'actor_loss': actor_loss,
        'entropy_mean': jnp.mean(-log_prob),
        'q_mean': jnp.mean(q_diag),
        'log_prob_mean': jnp.mean(log_prob),
    }
    return actor_loss, metrics

  def alpha_loss_fn(alpha_params, policy_params, batch, key):
    dist = networks.policy_network.apply(
        jax.lax.stop_gradient(policy_params), batch['obs']
    )
    action = networks.sample(dist, key)
    log_prob = networks.log_prob(dist, action)
    alpha = jnp.exp(alpha_params)
    loss = -alpha * jnp.mean(log_prob + target_entropy)
    metrics = {
        'alpha_loss': loss,
        'alpha': alpha,
    }
    return loss, metrics

  @jax.jit
  def update_step(training_state: SACTrainingState, batch: Dict[str, jnp.ndarray]):
    key, key_actor, key_alpha = jax.random.split(training_state.key, 3)

    # 1. Critic Update
    (c_loss, c_metrics), c_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
        training_state.q_params, batch
    )
    q_updates, new_q_opt_state = q_optimizer.update(
        c_grads, training_state.q_opt_state, training_state.q_params
    )
    new_q_params = optax.apply_updates(training_state.q_params, q_updates)

    # 2. Alpha Update
    if use_adaptive_alpha:
      (a_loss, a_metrics), a_grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(
          training_state.alpha_params, training_state.policy_params, batch, key_alpha
      )
      alpha_updates, new_alpha_opt_state = alpha_optimizer.update(
          a_grads, training_state.alpha_opt_state, training_state.alpha_params
      )
      new_alpha_params = optax.apply_updates(
          training_state.alpha_params, alpha_updates
      )
      alpha_val = jnp.exp(new_alpha_params)
    else:
      new_alpha_params = training_state.alpha_params
      new_alpha_opt_state = training_state.alpha_opt_state
      alpha_val = float(config.entropy_coefficient)
      a_metrics = {'alpha': alpha_val, 'alpha_loss': 0.0}

    # 3. Actor Update
    (p_loss, p_metrics), p_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
        training_state.policy_params,
        new_q_params,
        alpha_val,
        batch,
        key_actor,
    )
    p_updates, new_policy_opt_state = policy_optimizer.update(
        p_grads, training_state.policy_opt_state, training_state.policy_params
    )
    new_policy_params = optax.apply_updates(
        training_state.policy_params, p_updates
    )

    new_state = SACTrainingState(
        policy_params=new_policy_params,
        policy_opt_state=new_policy_opt_state,
        q_params=new_q_params,
        q_opt_state=new_q_opt_state,
        alpha_params=new_alpha_params,
        alpha_opt_state=new_alpha_opt_state,
        key=key,
    )
    all_metrics = {**c_metrics, **p_metrics, **a_metrics}
    return new_state, all_metrics

  @jax.jit
  def scan_update_step(
      training_state: SACTrainingState,
      stacked_batches: Dict[str, jnp.ndarray],
  ):
    """Runs N updates in a single lax.scan."""
    def _body_fn(carry_state, batch_slice):
      next_state, metrics = update_step(carry_state, batch_slice)
      return next_state, metrics

    final_state, stacked_metrics = jax.lax.scan(
        _body_fn, training_state, stacked_batches
    )
    mean_metrics = jax.tree_map(lambda x: jnp.mean(x, axis=0), stacked_metrics)
    return final_state, mean_metrics

  return update_step, scan_update_step


# ---------------------------------------------------------------------------
# Training Orchestration Loop
# ---------------------------------------------------------------------------
def run_sac_crl_training(
    config: contrastive_config.ContrastiveConfig,
    env_factory: Callable[[int], Any],
    eval_env_factory: Callable[[int], Any],
    network_factory: Callable[[Any], contrastive_networks.ContrastiveNetworks],
    logger_fn: Callable[..., Any],
    total_steps: int,
    seed: int = 0,
    video_fn: Optional[Callable[..., None]] = None,
    video_every_steps: int = 0,
    checkpoint_dir: Optional[str] = None,
) -> SACTrainingState:
  """Main training loop for standalone SAC + CRL on GPU-batched environments."""
  key = jax.random.PRNGKey(seed)
  np_rng = np.random.default_rng(seed + 1000)

  E = int(config.ppo_num_envs) if config.ppo_num_envs > 0 else 64
  T = int(config.ppo_rollout_length) if config.ppo_rollout_length > 0 else 256
  steps_per_iter = E * T
  num_iterations = int(np.ceil(total_steps / steps_per_iter))
  crl_steps_per_iter = int(config.ppo_crl_steps_per_iter) if config.ppo_crl_steps_per_iter > 0 else 128
  min_replay_size = int(config.min_replay_size) if config.min_replay_size > 0 else 10000

  print(
      f'[sac_crl] Initializing: E={E}, T={T}, steps_per_iter={steps_per_iter}, '
      f'total_steps={total_steps}, crl_steps_per_iter={crl_steps_per_iter}'
  )

  # 1. Build Environment
  if config.ppo_maniskill_native_vec:
    success_key = (
        'drawer_closed'
        if config.env_name == 'maniskill_close_subtask_train'
        else ('drawer_open' if config.env_name == 'maniskill_open_subtask_train' else 'success')
    )
    vec_env = _env_utils.ManiskillVecEnv(
        env_name=config.env_name,
        num_envs=E,
        obs_dim=int(config.obs_dim),
        start_index=int(config.start_index),
        end_index=int(config.end_index),
        success_key=success_key,
    )
  else:
    vec_env = VecEnv(env_factory, E, seed=seed * 31)

  # 2. Build Networks & Optimizers from Spec
  probe_env = env_factory(seed)
  spec = _specs.make_environment_spec(probe_env)
  del probe_env

  networks = network_factory(spec=spec)

  k_pol, k_q, key = jax.random.split(key, 3)

  policy_params = networks.policy_network.init(k_pol)
  q_params = networks.q_network.init(k_q)

  policy_optimizer = optax.adam(learning_rate=float(config.actor_learning_rate))
  q_optimizer = optax.adam(learning_rate=float(config.learning_rate))

  if config.entropy_coefficient is None:
    alpha_params = jnp.array(0.0, dtype=jnp.float32)
    alpha_optimizer = optax.adam(learning_rate=float(config.learning_rate))
    alpha_opt_state = alpha_optimizer.init(alpha_params)
  else:
    alpha_params = None
    alpha_optimizer = None
    alpha_opt_state = None

  policy_opt_state = policy_optimizer.init(policy_params)
  q_opt_state = q_optimizer.init(q_params)

  training_state = SACTrainingState(
      policy_params=policy_params,
      policy_opt_state=policy_opt_state,
      q_params=q_params,
      q_opt_state=q_opt_state,
      alpha_params=alpha_params,
      alpha_opt_state=alpha_opt_state,
      key=key,
  )

  # 3. JIT Update Functions
  _, scan_update_fn = make_sac_crl_update_fn(
      networks, config, policy_optimizer, q_optimizer, alpha_optimizer
  )

  @jax.jit
  def act_fn(p_params, obs, rng):
    dist = networks.policy_network.apply(p_params, obs)
    return networks.sample(dist, rng)

  @jax.jit
  def greedy_act_fn(p_params, obs):
    dist = networks.policy_network.apply(p_params, obs)
    return networks.sample_eval(dist, None)

  # 4. Replay Buffer
  replay = EpisodeReplay(
      capacity=int(config.max_replay_size),
      obs_dim=int(config.obs_dim),
      discount=float(config.discount),
      start_index=int(config.start_index),
      end_index=int(config.end_index),
  )

  uniform_sampling = bool(getattr(config, 'uniform_sampling', False))
  goal_low = goal_high = None
  if uniform_sampling:
    _obs_min = np.asarray(spec.observations.minimum, dtype=np.float32)
    _obs_max = np.asarray(spec.observations.maximum, dtype=np.float32)
    _end = int(config.end_index) if int(config.end_index) != -1 else int(config.obs_dim)
    goal_low = _obs_min[int(config.start_index):_end]
    goal_high = _obs_max[int(config.start_index):_end]
    print(f'[sac_crl] uniform_sampling enabled: goal_low={goal_low}, goal_high={goal_high}')

  # 5. Logging & Observers
  learner_logger = logger_fn(label='learner')
  eval_logger = logger_fn(label='eval')

  _env_str = str(getattr(config, 'env_name', '') or '').lower()
  if _env_str == 'maniskill_close_subtask_train':
    eval_success_obs = _cu.DrawerClosedSuccessObserver()
  elif _env_str == 'maniskill_open_subtask_train':
    eval_success_obs = _cu.DrawerOpenSuccessObserver()
  else:
    eval_success_obs = _cu.SuccessObserver()
  eval_dist_obs = _cu.DistanceObserver(
      obs_dim=int(config.obs_dim),
      start_index=int(config.start_index),
      end_index=int(config.end_index),
  )

  eval_env = eval_env_factory(seed + 42)

  # 6. Checkpoint Resuming
  start_iteration = 0
  global_step = 0
  gradient_steps = 0
  ckpt_interval = int(getattr(config, 'ppo_checkpoint_interval', 500))
  ckpt_keep_last = 10

  if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, 'latest.pkl')):
    latest_ckpt_path = os.path.join(checkpoint_dir, 'latest.pkl')
    import pickle as _pkl
    with open(latest_ckpt_path, 'rb') as fh:
      _ckpt = _pkl.load(fh)
    training_state = SACTrainingState(
        policy_params=_ckpt['policy_params'],
        policy_opt_state=_ckpt['policy_opt_state'],
        q_params=_ckpt['q_params'],
        q_opt_state=_ckpt['q_opt_state'],
        alpha_params=_ckpt['alpha_params'],
        alpha_opt_state=_ckpt['alpha_opt_state'],
        key=_ckpt['key'],
    )
    start_iteration = int(_ckpt['iteration']) + 1
    global_step = int(_ckpt['global_step'])
    gradient_steps = int(_ckpt.get('gradient_steps', 0))
    print(f'[sac_crl] Resumed from checkpoint: iteration={start_iteration}, global_step={global_step}')

  # Rollout storage tracking
  act_shape = spec.actions.shape
  ep_obs = [[] for _ in range(E)]
  ep_act = [[] for _ in range(E)]
  obs = vec_env.reset()
  for i in range(E):
    ep_obs[i].append(obs[i].copy())

  next_video_step = (
      ((global_step // video_every_steps) + 1) * video_every_steps
      if (video_fn is not None and video_every_steps > 0)
      else None
  )
  start_time = time.time()

  # =========================================================================
  # Main Training Loop
  # =========================================================================
  for iteration in range(start_iteration, num_iterations):
    # -----------------------------------------------------------------------
    # 1. Environment Rollouts
    # -----------------------------------------------------------------------
    for t in range(T):
      key, k_step = jax.random.split(key)
      if replay.size < min_replay_size and getattr(config, 'use_random_actor', True):
        actions = np_rng.uniform(-1.0, 1.0, size=(E,) + act_shape).astype(np.float32)
      else:
        actions_j = act_fn(training_state.policy_params, jnp.asarray(obs), k_step)
        actions = np.asarray(actions_j)

      next_obs, rewards, dones, terminal_obs = vec_env.step(actions)
      global_step += E

      for i in range(E):
        ep_act[i].append(actions[i].copy())
        if dones[i]:
          ep_obs[i].append(terminal_obs[i].copy())
          replay.add_episode(np.asarray(ep_obs[i]), np.asarray(ep_act[i]))
          ep_obs[i] = [next_obs[i].copy()]
          ep_act[i] = []
        else:
          ep_obs[i].append(next_obs[i].copy())

      obs = next_obs

    # -----------------------------------------------------------------------
    # 2. Off-Policy SAC + CRL Gradient Updates
    # -----------------------------------------------------------------------
    crl_metrics = {}
    if replay.size >= min_replay_size and crl_steps_per_iter > 0:
      batch_size = int(config.batch_size)
      if uniform_sampling:
        samples = [
            replay.sample_with_uniform_negatives(
                batch_size, np_rng, goal_low, goal_high
            )
            for _ in range(crl_steps_per_iter)
        ]
      else:
        samples = [
            replay.sample(batch_size, np_rng) for _ in range(crl_steps_per_iter)
        ]

      stacked_batch = {
          k: jnp.asarray(np.stack([s[k] for s in samples], axis=0))
          for k in samples[0]
      }
      training_state, crl_metrics = scan_update_fn(training_state, stacked_batch)
      gradient_steps += crl_steps_per_iter

    # -----------------------------------------------------------------------
    # 3. Evaluation Rollouts (Deterministic Mode)
    # -----------------------------------------------------------------------
    if iteration % 10 == 0 or iteration == num_iterations - 1:
      eval_ts = eval_env.reset()
      eval_success_obs.observe_first(eval_env, eval_ts)
      eval_dist_obs.observe_first(eval_env, eval_ts)
      eval_done = False
      while not eval_done:
        eval_act = greedy_act_fn(
            training_state.policy_params, jnp.asarray(eval_ts.observation)[None]
        )
        eval_act_np = np.asarray(eval_act)[0]
        eval_ts = eval_env.step(eval_act_np)
        eval_success_obs.observe(eval_env, eval_ts, eval_act_np)
        eval_dist_obs.observe(eval_env, eval_ts, eval_act_np)
        eval_done = eval_ts.last() if hasattr(eval_ts, 'last') else False

      eval_metrics_dict = {
          'iteration': iteration,
          'learner_steps': iteration,
          'global_step': global_step,
          **eval_success_obs.get_metrics(),
          **eval_dist_obs.get_metrics(),
      }
      eval_logger.write(eval_metrics_dict)

    # -----------------------------------------------------------------------
    # 4. Metrics Logging
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    log_dict = {
        'iteration': iteration,
        'learner_steps': iteration,
        'global_step': global_step,
        'gradient_steps': gradient_steps,
        'sps': global_step / max(1e-6, elapsed),
        'replay_size': int(replay.size),
        'num_episodes': int(replay.num_episodes),
    }
    for k, v in crl_metrics.items():
      log_dict[f'crl/{k}'] = float(v)

    learner_logger.write(log_dict)

    # -----------------------------------------------------------------------
    # 5. Periodic Video
    # -----------------------------------------------------------------------
    if next_video_step is not None and global_step >= next_video_step:
      if video_fn is not None:
        video_fn(training_state.policy_params, None, global_step, iteration)
      next_video_step += video_every_steps

    # -----------------------------------------------------------------------
    # 6. Checkpointing
    # -----------------------------------------------------------------------
    if checkpoint_dir and (iteration % ckpt_interval == 0 or iteration == num_iterations - 1):
      os.makedirs(checkpoint_dir, exist_ok=True)
      ckpt_file = os.path.join(checkpoint_dir, f'ckpt_iter_{iteration}.pkl')
      latest_file = os.path.join(checkpoint_dir, 'latest.pkl')
      _save_checkpoint(
          ckpt_file,
          training_state,
          iteration,
          global_step,
          gradient_steps,
          hidden_layer_sizes=config.hidden_layer_sizes,
      )
      _save_checkpoint(
          latest_file,
          training_state,
          iteration,
          global_step,
          gradient_steps,
          hidden_layer_sizes=config.hidden_layer_sizes,
      )
      _prune_old_checkpoints(checkpoint_dir, ckpt_keep_last)

  return training_state
