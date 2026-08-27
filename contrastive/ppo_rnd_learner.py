"""PPO + RND (Random Network Distillation) learner.

Standalone single-process implementation, sibling to `ppo_learner.py`
(PPO-on-φ·ψ). No CRL critic, no replay buffer -- reward is:

  * extrinsic:  sparse evaluator success (1 if the goal is reached, per
                `contrastive.utils.SuccessObserver` /
                `DrawerClosedSuccessObserver` semantics -- NOT ManiSkill's own
                dense reward, and for `maniskill_close_subtask_train`, NOT
                mshab's own (broader) `info['success']` either; see
                `env_utils.ManiskillVecEnv`'s `success_key` kwarg).
  * intrinsic:  RND prediction error `0.5*||predictor(s) - target(s)||^2` on
                the STATE-ONLY slice of the observation (excludes the
                per-episode goal slice -- see module-level design note in the
                repo's PPO+RND integration plan).

Two independent value heads (extrinsic, intrinsic) and two independent GAE
streams (with their own discounts) are combined into one PPO advantage:
    advantages = rnd_ext_coef * ext_adv + rnd_int_coef * int_adv

Everything else (obs normalization, reward normalization, vec-env stepping,
checkpointing, eval loop, logging) mirrors `ppo_learner.py` and reuses its
generic pieces directly (no duplication of `RunningMeanStd`, `ObsNormalizer`,
`ReturnNormalizer`, `VecEnv`, `load_checkpoint`, `_truncate_csv_to_iteration`).
"""
import os
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme.jax import networks as networks_lib

from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
from contrastive.ppo_learner import (
    ObsNormalizer,
    ReturnNormalizer,
    VecEnv,
    load_checkpoint,
    _truncate_csv_to_iteration,
)


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------
class PPORNDTrainingState(NamedTuple):
  """All trainable state held by the PPO + RND learner."""
  policy_params: networks_lib.Params
  ext_value_params: networks_lib.Params
  int_value_params: networks_lib.Params
  rnd_predictor_params: networks_lib.Params
  rnd_target_params: networks_lib.Params  # fixed at init, never trained
  ppo_optimizer_state: optax.OptState
  key: networks_lib.PRNGKey
  obs_normalizer: Optional['ObsNormalizer'] = None


# ---------------------------------------------------------------------------
# GAE (parameterized by discount directly -- ext/int streams use different
# discounts, so this isn't closed over a single `config` like
# `ppo_learner.make_gae_fn`).
# ---------------------------------------------------------------------------
def _make_gae_fn(discount: float, gae_lambda: float):
  discount = float(discount)
  gae_lambda = float(gae_lambda)

  @jax.jit
  def gae_fn(rewards, values, dones, next_value, next_done):
    next_vals_seq = jnp.concatenate([values[1:], next_value[None]], axis=0)
    next_dones_seq = jnp.concatenate([dones[1:], next_done[None]], axis=0)
    next_nonterm = 1.0 - next_dones_seq.astype(jnp.float32)

    deltas = rewards + discount * next_vals_seq * next_nonterm - values

    def scan_fn(last_gae, inputs):
      delta_t, nnt_t = inputs
      new_gae = delta_t + discount * gae_lambda * nnt_t * last_gae
      return new_gae, new_gae

    init = jnp.zeros(rewards.shape[1], dtype=rewards.dtype)
    _, adv_rev = jax.lax.scan(
        scan_fn, init, (deltas[::-1], next_nonterm[::-1]))
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns
  return gae_fn


# ---------------------------------------------------------------------------
# Combined PPO + RND update.
# ---------------------------------------------------------------------------
def make_rnd_ppo_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    rnd_networks: contrastive_networks.RNDNetworks,
    config: contrastive_config.ContrastiveConfig,
    ppo_optimizer: optax.GradientTransformation,
    obs_dim: int,
):
  """One combined minibatch update over {policy, ext_value, int_value,
  rnd_predictor}, matching the collaborator's PPO+RND scripts' single
  optimizer / single combined loss design.
  """
  clip_coef = float(config.ppo_clip_coef)
  vf_coef = float(config.ppo_vf_coef)
  ent_coef = float(config.ppo_ent_coef)
  clip_vloss = bool(config.ppo_clip_vloss)
  norm_adv = bool(config.ppo_norm_adv)
  obs_dim = int(obs_dim)

  def rnd_ppo_loss(params, rnd_target_params, batch, key):
    # ---- policy forward ----
    dist = networks.policy_network.apply(params['policy'], batch['obs'])
    new_logprob = networks.log_prob(dist, batch['actions'])
    fresh_action = networks.sample(dist, key)
    entropy_est = -networks.log_prob(dist, fresh_action)

    # ---- value forwards (extrinsic + intrinsic) ----
    new_ext_value = networks.value_network.apply(params['ext_value'], batch['obs'])
    new_int_value = rnd_networks.int_value_network.apply(
        params['int_value'], batch['obs'])

    # ---- advantage normalization (per-minibatch, on the COMBINED advantage) ----
    adv = batch['advantages']
    if norm_adv:
      adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ---- clipped surrogate ----
    logratio = new_logprob - batch['old_logprobs']
    ratio = jnp.exp(logratio)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

    # ---- value losses ----
    def _v_loss(new_value, old_value, returns):
      if clip_vloss:
        v_unclipped = (new_value - returns) ** 2
        v_clipped_pred = old_value + jnp.clip(
            new_value - old_value, -clip_coef, clip_coef)
        v_clipped = (v_clipped_pred - returns) ** 2
        return 0.5 * jnp.mean(jnp.maximum(v_unclipped, v_clipped))
      return 0.5 * jnp.mean((new_value - returns) ** 2)

    ext_v_loss = _v_loss(new_ext_value, batch['old_ext_values'], batch['ext_returns'])
    int_v_loss = _v_loss(new_int_value, batch['old_int_values'], batch['int_returns'])

    # ---- entropy bonus ----
    entropy_mean = jnp.mean(entropy_est)

    # ---- RND forward loss (predictor vs fixed target, state-only) ----
    # `rnd_target_params` is a fixed argument (not part of `params`), so no
    # gradient flows to it regardless -- it never appears in optax's state.
    state = batch['rnd_state']
    prediction = rnd_networks.rnd_predictor_network.apply(
        params['rnd_predictor'], state)
    target = rnd_networks.rnd_target_network.apply(rnd_target_params, state)
    rnd_loss = 0.5 * jnp.mean(jnp.sum((prediction - target) ** 2, axis=-1))

    total = (pg_loss - ent_coef * entropy_mean
             + vf_coef * (ext_v_loss + int_v_loss) + rnd_loss)

    approx_kl = jnp.mean((ratio - 1.0) - logratio)
    old_approx_kl = jnp.mean(-logratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))

    metrics = {
        'ppo_total_loss': total,
        'pg_loss': pg_loss,
        'ext_v_loss': ext_v_loss,
        'int_v_loss': int_v_loss,
        'rnd_loss': rnd_loss,
        'entropy': entropy_mean,
        'approx_kl': approx_kl,
        'old_approx_kl': old_approx_kl,
        'clipfrac': clipfrac,
        'ratio_mean': jnp.mean(ratio),
    }
    return total, metrics

  grad_fn = jax.value_and_grad(rnd_ppo_loss, has_aux=True)

  @jax.jit
  def update(params, opt_state, rnd_target_params, batch, key):
    (_, metrics), grads = grad_fn(params, rnd_target_params, batch, key)
    updates, new_opt_state = ppo_optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics

  return update


# ---------------------------------------------------------------------------
# Checkpoint helpers (RND-specific fields; same atomic-write pattern as
# `ppo_learner._save_checkpoint`; `load_checkpoint` is reused unchanged).
# ---------------------------------------------------------------------------
def _save_rnd_checkpoint(path: str,
                         policy_params, ext_value_params, int_value_params,
                         rnd_predictor_params, rnd_target_params,
                         ppo_opt_state,
                         iteration: int, global_step: int, key,
                         hidden_layer_sizes=None,
                         obs_normalizer_state=None):
  import pickle as _pkl
  import os as _os

  ckpt = {
      'policy_params':        policy_params,
      'ext_value_params':     ext_value_params,
      'int_value_params':     int_value_params,
      'rnd_predictor_params': rnd_predictor_params,
      'rnd_target_params':    rnd_target_params,
      'ppo_optimizer_state':  ppo_opt_state,
      'iteration':            int(iteration),
      'global_step':          int(global_step),
      'key':                  key,
      'hidden_layer_sizes':   (tuple(hidden_layer_sizes)
                               if hidden_layer_sizes is not None else None),
      'obs_normalizer_state': obs_normalizer_state,
  }
  tmp_path = path + '.tmp'
  with open(tmp_path, 'wb') as fh:
    _pkl.dump(ckpt, fh, protocol=_pkl.HIGHEST_PROTOCOL)
  _os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Top-level training loop.
# ---------------------------------------------------------------------------
def run_ppo_rnd_training(
    config: contrastive_config.ContrastiveConfig,
    env_factory: Callable,
    eval_env_factory: Callable,
    network_factory: Callable,
    logger_fn: Callable,
    total_steps: int,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
    video_fn: Optional[Callable] = None,
    video_every_steps: int = 0,
):
  """Top-level PPO+RND training loop. See module docstring."""
  # ---- vec env ------------------------------------------------------------
  # Same GPU-PhysX-first-touch ordering constraint as `ppo_learner
  # .run_ppo_training` -- see that function's matching comment.
  _env_name = str(getattr(config, 'env_name', '') or '').lower()
  _use_maniskill_native_vec = (
      bool(getattr(config, 'ppo_maniskill_native_vec', False))
      and _env_name in ('maniskill_pushcube', 'maniskill_pickcube',
                        'maniskill_open_cabinet_drawer',
                        'maniskill_close_cabinet_drawer',
                        'maniskill_close_subtask_train',
                        'maniskill_open_subtask_train'))
  if _use_maniskill_native_vec:
    import env_utils as _env_utils
    if _env_name == 'maniskill_close_subtask_train':
      _success_key = 'drawer_closed'
    elif _env_name == 'maniskill_open_subtask_train':
      _success_key = 'drawer_open'
    else:
      _success_key = 'success'
    vec_env = _env_utils.ManiskillVecEnv(
        _env_name, config.ppo_num_envs,
        obs_dim=int(config.obs_dim),
        start_index=int(config.start_index),
        end_index=int(config.end_index),
        success_key=_success_key)
    print(f'[ppo_rnd] maniskill native vec env: {_env_name}, '
          f'num_envs={config.ppo_num_envs}, success_key={_success_key!r}')
  else:
    vec_env = VecEnv(env_factory, config.ppo_num_envs, seed=seed * 31)
  E = vec_env.num_envs

  # ---- build networks from env spec ---------------------------------------
  from acme import specs as _specs
  import contrastive.utils as _cu

  probe_env = env_factory(seed)
  spec = _specs.make_environment_spec(probe_env)
  networks = network_factory(spec=spec)
  obs_dim = int(config.obs_dim)
  rnd_networks = contrastive_networks.make_rnd_extra_networks(
      spec=spec,
      obs_dim=obs_dim,
      hidden_layer_sizes=config.hidden_layer_sizes,
      rnd_output_size=int(config.rnd_output_size))
  del probe_env
  obs_shape = vec_env.observation_shape
  act_shape = vec_env.action_shape
  T = int(config.ppo_rollout_length)
  batch_per_iter = T * E
  mb_size = batch_per_iter // int(config.ppo_num_minibatches)
  assert mb_size * int(config.ppo_num_minibatches) == batch_per_iter, (
      'batch_per_iter must divide evenly into ppo_num_minibatches')
  num_iterations = int(total_steps) // (T * E)

  # ---- observation normalizer ----------------------------------------------
  norm_obs = bool(getattr(config, 'ppo_norm_obs', True))
  obs_normalizer = (
      ObsNormalizer(obs_dim=obs_dim,
                    start_index=int(config.start_index),
                    end_index=int(config.end_index))
      if norm_obs else None)

  def _obs_update(raw_state: np.ndarray):
    if obs_normalizer is not None:
      obs_normalizer.update(raw_state[..., :obs_dim])

  def _obs_norm(raw_obs: np.ndarray) -> np.ndarray:
    return obs_normalizer(raw_obs) if obs_normalizer is not None else raw_obs

  def _rnd_state_norm(raw_obs: np.ndarray) -> np.ndarray:
    """State-only slice, normalized with RND's tighter +/-5 clip (vs the
    policy/value input's +/-10 clip)."""
    state = raw_obs[..., :obs_dim]
    if obs_normalizer is not None:
      return obs_normalizer.normalize_state(state, clip=5.0)
    return np.clip(state, -5.0, 5.0).astype(np.float32)

  # ---- init params ----------------------------------------------------------
  key = jax.random.PRNGKey(seed)
  k_pol, k_ext_val, k_int_val, k_pred, k_target, key = jax.random.split(key, 6)
  policy_params = networks.policy_network.init(k_pol)
  ext_value_params = networks.value_network.init(k_ext_val)
  int_value_params = rnd_networks.int_value_network.init(k_int_val)
  rnd_predictor_params = rnd_networks.rnd_predictor_network.init(k_pred)
  rnd_target_params = rnd_networks.rnd_target_network.init(k_target)
  # `rnd_target_params` is deliberately NOT in `ppo_params`: it's fixed at
  # init and never trained, so it must not appear in the optax-tracked
  # pytree (which would otherwise waste an Adam moment-estimate slot on a
  # permanently-zero gradient).
  ppo_params = {
      'policy': policy_params,
      'ext_value': ext_value_params,
      'int_value': int_value_params,
      'rnd_predictor': rnd_predictor_params,
  }

  # ---- optimizer --------------------------------------------------------
  if config.ppo_anneal_lr:
    total_ppo_updates = (
        num_iterations
        * int(config.ppo_num_epochs)
        * int(config.ppo_num_minibatches))
    lr_schedule = optax.linear_schedule(
        init_value=float(config.learning_rate),
        end_value=0.0,
        transition_steps=max(1, total_ppo_updates))
    ppo_optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(lambda count: -lr_schedule(count)))
  else:
    ppo_optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
        optax.adam(float(config.learning_rate), eps=1e-5))
  ppo_opt_state = ppo_optimizer.init(ppo_params)

  # ---- resume from checkpoint if one exists -----------------------------
  start_iteration = 0
  global_step = 0
  ppo_sgd_step = 0
  if checkpoint_dir is not None:
    _latest = os.path.join(checkpoint_dir, 'latest.pkl')
    if os.path.exists(_latest):
      _ckpt = load_checkpoint(_latest)
      ppo_params = {
          'policy': _ckpt['policy_params'],
          'ext_value': _ckpt['ext_value_params'],
          'int_value': _ckpt['int_value_params'],
          'rnd_predictor': _ckpt['rnd_predictor_params'],
      }
      rnd_target_params = _ckpt['rnd_target_params']
      ppo_opt_state = _ckpt['ppo_optimizer_state']
      _obs_norm_state = _ckpt.get('obs_normalizer_state', None)
      if obs_normalizer is not None and _obs_norm_state is not None:
        obs_normalizer.load_state_dict(_obs_norm_state)
      start_iteration = int(_ckpt['iteration']) + 1
      global_step = int(_ckpt['global_step'])
      key = _ckpt['key']
      ppo_sgd_step = (start_iteration
                      * int(config.ppo_num_epochs)
                      * int(config.ppo_num_minibatches))
      print(f'[ppo_rnd] resumed from checkpoint: '
            f'start_iteration={start_iteration}, global_step={global_step}')
      _run_dir = os.path.dirname(checkpoint_dir)
      for _label in ('learner', 'eval'):
        _csv_path = os.path.join(_run_dir, 'logs', _label, 'logs.csv')
        _truncate_csv_to_iteration(_csv_path, int(_ckpt['iteration']))

  # ---- jitted helpers -----------------------------------------------------
  ext_gamma = (float(config.ppo_discount)
               if float(getattr(config, 'ppo_discount', -1.0)) > 0.0
               else float(config.discount))
  int_gamma = float(config.rnd_int_discount)
  gae_lambda = float(config.ppo_gae_lambda)
  ext_gae_fn = _make_gae_fn(ext_gamma, gae_lambda)
  int_gae_fn = _make_gae_fn(int_gamma, gae_lambda)
  rnd_ppo_update = make_rnd_ppo_update_fn(
      networks, rnd_networks, config, ppo_optimizer, obs_dim)

  @jax.jit
  def act_and_value(policy_p, ext_value_p, int_value_p, obs, rng):
    dist = networks.policy_network.apply(policy_p, obs)
    action = networks.sample(dist, rng)
    logprob = networks.log_prob(dist, action)
    ext_value = networks.value_network.apply(ext_value_p, obs)
    int_value = rnd_networks.int_value_network.apply(int_value_p, obs)
    return action, logprob, ext_value, int_value

  @jax.jit
  def value_only(ext_value_p, int_value_p, obs):
    ext_value = networks.value_network.apply(ext_value_p, obs)
    int_value = rnd_networks.int_value_network.apply(int_value_p, obs)
    return ext_value, int_value

  @jax.jit
  def rnd_intrinsic_reward(rnd_predictor_p, rnd_target_p, state):
    prediction = rnd_networks.rnd_predictor_network.apply(rnd_predictor_p, state)
    target = rnd_networks.rnd_target_network.apply(rnd_target_p, state)
    return 0.5 * jnp.sum((prediction - target) ** 2, axis=-1)

  @jax.jit
  def greedy_action(policy_p, obs):
    dist = networks.policy_network.apply(policy_p, obs)
    return networks.sample_eval(dist, None)

  # ---- per-env episode accounting (logging only -- no replay buffer) ------
  ep_return = np.zeros(E, dtype=np.float32)
  ep_len = np.zeros(E, dtype=np.int32)
  recent_returns: list = []
  recent_lengths: list = []

  obs = vec_env.reset()
  _obs_update(obs)
  obs = _obs_norm(obs)
  next_done = np.zeros(E, dtype=np.float32)

  # ---- loggers --------------------------------------------------------------
  learner_logger = logger_fn(label='learner')
  eval_logger = logger_fn(label='eval')

  _env = str(getattr(config, 'env_name', '') or '').lower()
  if _env == 'riverswim':
    eval_success_obs = _cu.RiverSwimGoalVisitSuccessObserver(obs_dim=obs_dim)
  elif _env == 'maniskill_close_subtask_train':
    eval_success_obs = _cu.DrawerClosedSuccessObserver()
  elif _env == 'maniskill_open_subtask_train':
    eval_success_obs = _cu.DrawerOpenSuccessObserver()
  else:
    eval_success_obs = _cu.SuccessObserver()
  eval_dist_obs = _cu.DistanceObserver(
      obs_dim=obs_dim,
      start_index=int(config.start_index),
      end_index=int(config.end_index))

  # ---- rollout storage (reused each iteration) -----------------------------
  roll_obs = np.zeros((T, E) + obs_shape, dtype=np.float32)
  roll_acts = np.zeros((T, E) + act_shape, dtype=np.float32)
  roll_logp = np.zeros((T, E), dtype=np.float32)
  roll_ext_rew = np.zeros((T, E), dtype=np.float32)       # normalized (or raw)
  roll_ext_rew_raw = np.zeros((T, E), dtype=np.float32)   # log only
  roll_int_rew = np.zeros((T, E), dtype=np.float32)       # normalized
  roll_int_rew_raw = np.zeros((T, E), dtype=np.float32)   # log only
  roll_dones = np.zeros((T, E), dtype=np.float32)
  roll_ext_vals = np.zeros((T, E), dtype=np.float32)
  roll_int_vals = np.zeros((T, E), dtype=np.float32)
  roll_step_dones = np.zeros((T, E), dtype=np.float32)
  roll_next_state = np.zeros((T, E, obs_dim), dtype=np.float32)  # for RND fwd

  # ---- reward normalizers (extrinsic + intrinsic, independent streams) ----
  norm_reward = bool(getattr(config, 'ppo_norm_reward', True))
  ext_reward_normalizer = (
      ReturnNormalizer(num_envs=E, discount=ext_gamma) if norm_reward else None)
  int_reward_normalizer = ReturnNormalizer(num_envs=E, discount=int_gamma)

  # ---- checkpointing --------------------------------------------------------
  ckpt_interval = int(getattr(config, 'ppo_checkpoint_interval', 0))
  ckpt_keep_last = int(getattr(config, 'ppo_checkpoint_keep_last', 10))
  if ckpt_interval > 0 and checkpoint_dir is not None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'[ppo_rnd] checkpoints -> {checkpoint_dir} '
          f'(every {ckpt_interval} iters)')

  start_time = time.time()
  np_rng = np.random.default_rng(seed + 12345)

  next_video_step = (
      ((global_step // video_every_steps) + 1) * video_every_steps
      if (video_fn is not None and video_every_steps > 0) else None)

  for iteration in range(start_iteration, num_iterations):
    # =================================================================
    # 1. Rollout (on-policy)
    # =================================================================
    for t in range(T):
      roll_obs[t] = obs
      roll_dones[t] = next_done

      key, k_act = jax.random.split(key)
      action_j, logprob_j, ext_val_j, int_val_j = act_and_value(
          ppo_params['policy'], ppo_params['ext_value'],
          ppo_params['int_value'], jnp.asarray(obs), k_act)
      action = np.asarray(action_j)
      roll_acts[t] = action
      roll_logp[t] = np.asarray(logprob_j)
      roll_ext_vals[t] = np.asarray(ext_val_j)
      roll_int_vals[t] = np.asarray(int_val_j)

      next_obs, env_rew, dones, terminal_obs = vec_env.step(action)
      _obs_update(terminal_obs)
      if dones.any():
        _obs_update(next_obs[dones])
      terminal_obs = _obs_norm(terminal_obs)
      next_obs = _obs_norm(next_obs)
      roll_ext_rew_raw[t] = env_rew
      roll_step_dones[t] = dones.astype(np.float32)
      # RND always looks at the TRUE post-step state (terminal_obs), never
      # the auto-reset observation -- matches the collaborator's gym script's
      # `rnd_source = np.where(done, terminal_obs, next_obs)` convention.
      roll_next_state[t] = np.where(
          dones[:, None], terminal_obs[:, :obs_dim], next_obs[:, :obs_dim])

      for i in range(E):
        ep_return[i] += float(env_rew[i])
        ep_len[i] += 1
        if dones[i]:
          recent_returns.append(float(ep_return[i]))
          recent_lengths.append(int(ep_len[i]))
          ep_return[i] = 0.0
          ep_len[i] = 0
          if len(recent_returns) > 100:
            recent_returns.pop(0)
            recent_lengths.pop(0)

      obs = next_obs
      next_done = dones.astype(np.float32)
      global_step += E

    # =================================================================
    # 1b. Batched intrinsic reward (single call over the whole rollout).
    # =================================================================
    _flat_state = roll_next_state.reshape(batch_per_iter, obs_dim)
    _flat_state_norm = _rnd_state_norm(_flat_state)
    _int_rew_flat = np.asarray(rnd_intrinsic_reward(
        ppo_params['rnd_predictor'], rnd_target_params,
        jnp.asarray(_flat_state_norm)))
    roll_int_rew_raw[:] = _int_rew_flat.reshape(T, E)

    for _t in range(T):
      roll_int_rew[_t] = int_reward_normalizer(
          roll_int_rew_raw[_t], roll_step_dones[_t])
      if ext_reward_normalizer is not None:
        roll_ext_rew[_t] = ext_reward_normalizer(
            roll_ext_rew_raw[_t], roll_step_dones[_t])
    if ext_reward_normalizer is None:
      roll_ext_rew[:] = roll_ext_rew_raw

    # =================================================================
    # 2. GAE (extrinsic + intrinsic streams, independently)
    # =================================================================
    next_ext_val_j, next_int_val_j = value_only(
        ppo_params['ext_value'], ppo_params['int_value'], jnp.asarray(obs))
    next_ext_val = np.asarray(next_ext_val_j)
    next_int_val = np.asarray(next_int_val_j)

    ext_adv_j, ext_ret_j = ext_gae_fn(
        jnp.asarray(roll_ext_rew), jnp.asarray(roll_ext_vals),
        jnp.asarray(roll_dones), jnp.asarray(next_ext_val), jnp.asarray(next_done))
    int_adv_j, int_ret_j = int_gae_fn(
        jnp.asarray(roll_int_rew), jnp.asarray(roll_int_vals),
        jnp.asarray(roll_dones), jnp.asarray(next_int_val), jnp.asarray(next_done))
    ext_adv = np.asarray(ext_adv_j)
    ext_ret = np.asarray(ext_ret_j)
    int_adv = np.asarray(int_adv_j)
    int_ret = np.asarray(int_ret_j)

    ext_coef = float(config.rnd_ext_coef)
    int_coef = float(config.rnd_int_coef)
    combined_adv = ext_coef * ext_adv + int_coef * int_adv

    # =================================================================
    # 3. PPO updates (epochs x minibatches over flat T*E batch)
    # =================================================================
    flat_obs = roll_obs.reshape((batch_per_iter,) + obs_shape)
    flat_acts = roll_acts.reshape((batch_per_iter,) + act_shape)
    flat_logp = roll_logp.reshape(batch_per_iter)
    flat_adv = combined_adv.reshape(batch_per_iter)
    flat_ext_ret = ext_ret.reshape(batch_per_iter)
    flat_int_ret = int_ret.reshape(batch_per_iter)
    flat_ext_vals = roll_ext_vals.reshape(batch_per_iter)
    flat_int_vals = roll_int_vals.reshape(batch_per_iter)
    flat_rnd_state = _flat_state_norm

    ppo_metrics_agg: Dict[str, list] = {}
    early_stop = False
    for epoch in range(int(config.ppo_num_epochs)):
      perm = np_rng.permutation(batch_per_iter)
      last_kl = None
      for start in range(0, batch_per_iter, mb_size):
        mb = perm[start:start + mb_size]
        batch = {
            'obs':            jnp.asarray(flat_obs[mb]),
            'actions':        jnp.asarray(flat_acts[mb]),
            'old_logprobs':   jnp.asarray(flat_logp[mb]),
            'advantages':     jnp.asarray(flat_adv[mb]),
            'ext_returns':    jnp.asarray(flat_ext_ret[mb]),
            'int_returns':    jnp.asarray(flat_int_ret[mb]),
            'old_ext_values': jnp.asarray(flat_ext_vals[mb]),
            'old_int_values': jnp.asarray(flat_int_vals[mb]),
            'rnd_state':      jnp.asarray(flat_rnd_state[mb]),
        }
        key, k_mb = jax.random.split(key)
        ppo_params, ppo_opt_state, m = rnd_ppo_update(
            ppo_params, ppo_opt_state, rnd_target_params, batch, k_mb)
        ppo_sgd_step += 1
        last_kl = float(m['approx_kl'])
        for k_, v in m.items():
          ppo_metrics_agg.setdefault(k_, []).append(float(v))
      if (config.ppo_target_kl is not None and last_kl is not None
          and last_kl > float(config.ppo_target_kl)):
        early_stop = True
        break

    # =================================================================
    # 4. Logging
    # =================================================================
    elapsed = time.time() - start_time
    log = {
        'iteration':     iteration,
        'learner_steps': iteration,
        'global_step':   global_step,
        'sps':           global_step / max(1e-6, elapsed),
        'reward_env_mean':       float(roll_ext_rew_raw.mean()),
        'rnd/int_reward_mean':     float(roll_int_rew.mean()),
        'rnd/int_reward_raw_mean': float(roll_int_rew_raw.mean()),
        'rnd/int_reward_raw_std':  float(roll_int_rew_raw.std()),
        'rnd/int_return_norm_std': float(int_reward_normalizer.std),
        'ext_value_mean':    float(roll_ext_vals.mean()),
        'int_value_mean':    float(roll_int_vals.mean()),
        'ext_returns_mean':  float(ext_ret.mean()),
        'int_returns_mean':  float(int_ret.mean()),
        'advantage_mean':    float(combined_adv.mean()),
        'advantage_std':     float(combined_adv.std()),
        'early_stop_epochs': int(early_stop),
        'ep_return_mean':    float(np.mean(recent_returns)) if recent_returns else float('nan'),
        'ep_length_mean':    float(np.mean(recent_lengths)) if recent_lengths else float('nan'),
    }
    for k_, vs in ppo_metrics_agg.items():
      log[f'ppo/{k_}'] = float(np.mean(vs))
    if config.ppo_anneal_lr:
      lr_log = float(lr_schedule(max(0, ppo_sgd_step - 1)))
    else:
      lr_log = float(config.learning_rate)
    log['ppo/learning_rate'] = round(lr_log, 7)
    learner_logger.write(log)

    # =================================================================
    # 5. Periodic evaluation (5 episodes every 10 iters) -- reuses the same
    # observers the CRL agent uses, so success/success_1000 are directly
    # comparable across the two baselines.
    # =================================================================
    if iteration % 10 == 0:
      ep_metrics_list = []
      for e_i in range(5):
        env = eval_env_factory(seed + 900_000 + iteration * 100 + e_i)
        ts = env.reset()
        eval_success_obs.observe_first(env, ts)
        eval_dist_obs.observe_first(env, ts)
        ret_e, n_e = 0.0, 0
        while not ts.last():
          a = greedy_action(
              ppo_params['policy'],
              jnp.asarray(_obs_norm(ts.observation))[None])
          action = np.asarray(a)[0].astype(np.float32)
          action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
          action = np.clip(action, -1.0, 1.0)
          ts = env.step(action)
          eval_success_obs.observe(env, ts, action)
          eval_dist_obs.observe(env, ts, action)
          ret_e += float(ts.reward or 0.0)
          n_e += 1
        ep_metrics = {'episode_return': ret_e, 'episode_length': n_e}
        ep_metrics.update(eval_success_obs.get_metrics())
        ep_metrics.update(eval_dist_obs.get_metrics())
        ep_metrics_list.append(ep_metrics)

      agg = {'iteration': iteration, 'learner_steps': iteration}
      for k_ in ep_metrics_list[0].keys():
        agg[k_] = float(np.nanmean([m[k_] for m in ep_metrics_list]))
      eval_logger.write(agg)

    # =================================================================
    # 6. Periodic mid-training rollout video.
    # =================================================================
    if next_video_step is not None and global_step >= next_video_step:
      video_fn(ppo_params['policy'], obs_normalizer, global_step, iteration)
      next_video_step += video_every_steps

    # =================================================================
    # 7. Checkpointing.
    # =================================================================
    if (ckpt_interval > 0
        and checkpoint_dir is not None
        and (iteration % ckpt_interval == 0
             or iteration == num_iterations - 1)):
      _save_rnd_checkpoint(
          os.path.join(checkpoint_dir, 'latest.pkl'),
          policy_params=ppo_params['policy'],
          ext_value_params=ppo_params['ext_value'],
          int_value_params=ppo_params['int_value'],
          rnd_predictor_params=ppo_params['rnd_predictor'],
          rnd_target_params=rnd_target_params,
          ppo_opt_state=ppo_opt_state,
          iteration=iteration,
          global_step=global_step,
          key=key,
          hidden_layer_sizes=config.hidden_layer_sizes,
          obs_normalizer_state=(obs_normalizer.state_dict()
                                if obs_normalizer is not None else None))

  return PPORNDTrainingState(
      policy_params=ppo_params['policy'],
      ext_value_params=ppo_params['ext_value'],
      int_value_params=ppo_params['int_value'],
      rnd_predictor_params=ppo_params['rnd_predictor'],
      rnd_target_params=rnd_target_params,
      ppo_optimizer_state=ppo_opt_state,
      key=key,
      obs_normalizer=obs_normalizer,
  )
