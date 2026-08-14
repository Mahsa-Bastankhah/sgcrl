"""Standalone online MPO policy learning with CRL representations.

The policy and representation objectives intentionally use different replay
views:

* MPO samples the original goal-conditioned observations collected online.
  By default it directly scores actions with target ``phi(s, a) dot psi(g)``.
  With ``use_td_critic``, that score instead becomes a fixed-goal reward for a
  scalar TD critic ``G(s, a)``, and MPO scores actions with target ``G``.
* CRL samples complete episodes through ``EpisodeReplay``, which relabels each
  transition with a geometrically sampled future goal.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import pickle
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))
import sgcrl_jax_acme_compat  # noqa: E402,F401

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme import specs
from acme.jax import networks as networks_lib

from contrastive import networks as contrastive_networks
from contrastive import ppo_learner
import contrastive.utils as contrastive_utils


def _load_acme_mpo_loss():
  """Load ../acme's MPO module without importing Acme's Reverb-heavy package."""
  path = Path(__file__).resolve().parents[2] / 'acme/acme/jax/losses/mpo.py'
  if not path.exists():
    raise FileNotFoundError(f'Expected Acme MPO loss at {path}')
  module_name = '_sgcrl_external_acme_mpo_loss'
  if module_name in sys.modules:
    return sys.modules[module_name]
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec is None or spec.loader is None:
    raise ImportError(f'Could not load Acme MPO loss from {path}')
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


mpo_losses = _load_acme_mpo_loss()


@dataclasses.dataclass
class MPOCRLConfig:
  """MPO-specific settings; environment and CRL settings stay in ContrastiveConfig."""

  num_envs: int = 8
  rollout_length: int = 128
  policy_batch_size: int = 256
  policy_updates_per_iter: int = 32
  crl_updates_per_iter: int = 64
  min_replay_size: int = 1_000
  num_action_samples: int = 20

  policy_learning_rate: float = 1e-4
  dual_learning_rate: float = 1e-2
  policy_grad_norm_clip: float = 40.0
  policy_hidden_sizes: Sequence[int] = (256, 256, 256)
  policy_init_scale: float = 0.7

  # Optional fixed-goal scalar TD critic G(s, a).  CRL target scores are the
  # rewards; G supplies the long-horizon action values used by MPO.
  use_td_critic: bool = False
  critic_learning_rate: float = 1e-4
  critic_grad_norm_clip: float = 40.0
  critic_hidden_sizes: Sequence[int] = (512, 512, 256)
  critic_target_update_period: int = 100
  critic_target_update_rate: float = 0.005
  bootstrap_action_samples: int = 20
  normalize_critic_reward: bool = False
  critic_reward_norm_rate: float = 0.001

  epsilon: float = 0.1
  epsilon_mean: float = 0.0025
  epsilon_stddev: float = 1e-6
  epsilon_penalty: float = 0.001
  init_log_temperature: float = 10.0
  init_log_alpha_mean: float = 10.0
  init_log_alpha_stddev: float = 1000.0
  per_dim_constraining: bool = True
  action_penalization: bool = True

  # Exactly one policy-target update mode is active. A positive period wins;
  # set it to zero to use incremental_update with target_update_rate.
  target_update_period: int = 100
  target_update_rate: float = 0.005
  # Acme MPO incremental-update convention:
  # target <- (1-rate) * target + rate * online.
  repr_target_update_rate: float = 0.005

  eval_interval: int = 30
  eval_episodes: int = 5
  checkpoint_interval: int = 200
  checkpoint_keep_last: int = 0

  def __post_init__(self):
    if self.num_action_samples <= 0:
      raise ValueError('num_action_samples must be positive')
    if self.target_update_period <= 0 and not 0.0 < self.target_update_rate <= 1.0:
      raise ValueError(
          'target_update_rate must be in (0, 1] when target_update_period <= 0')
    if not 0.0 < self.repr_target_update_rate <= 1.0:
      raise ValueError('repr_target_update_rate must be in (0, 1]')
    if self.use_td_critic:
      if (self.critic_target_update_period <= 0
          and not 0.0 < self.critic_target_update_rate <= 1.0):
        raise ValueError(
            'critic_target_update_rate must be in (0, 1] when '
            'critic_target_update_period <= 0')
      if self.bootstrap_action_samples <= 0:
        raise ValueError('bootstrap_action_samples must be positive')
      if (self.normalize_critic_reward
          and not 0.0 < self.critic_reward_norm_rate <= 1.0):
        raise ValueError('critic_reward_norm_rate must be in (0, 1]')


class MPOCRLTrainingState(NamedTuple):
  policy_params: networks_lib.Params
  target_policy_params: networks_lib.Params
  dual_params: mpo_losses.MPOParams
  policy_optimizer_state: optax.OptState
  dual_optimizer_state: optax.OptState
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  q_optimizer_state: optax.OptState
  policy_steps: jax.Array
  key: networks_lib.PRNGKey


class ObservationReplay:
  """FIFO replay of original goal-conditioned observations."""

  def __init__(self, capacity: int, observation_shape: Sequence[int]):
    self._capacity = int(capacity)
    self._storage = np.empty(
        (self._capacity,) + tuple(observation_shape), dtype=np.float32)
    self._size = 0
    self._cursor = 0

  @property
  def size(self) -> int:
    return self._size

  def add(self, observations: np.ndarray) -> None:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim == self._storage.ndim - 1:
      observations = observations[None]
    observations = observations.reshape((-1,) + self._storage.shape[1:])
    if observations.shape[0] >= self._capacity:
      observations = observations[-self._capacity:]
    n = observations.shape[0]
    first = min(n, self._capacity - self._cursor)
    self._storage[self._cursor:self._cursor + first] = observations[:first]
    remaining = n - first
    if remaining:
      self._storage[:remaining] = observations[first:]
    self._cursor = (self._cursor + n) % self._capacity
    self._size = min(self._capacity, self._size + n)

  def sample(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    if self._size < batch_size:
      raise ValueError(f'replay has {self._size} observations, need {batch_size}')
    indices = rng.integers(0, self._size, size=int(batch_size))
    return self._storage[indices]

  def sample_batches(
      self,
      num_batches: int,
      batch_size: int,
      rng: np.random.Generator,
  ) -> np.ndarray:
    """Sample a stack of batches for one compiled learner scan."""
    if self._size < batch_size:
      raise ValueError(f'replay has {self._size} observations, need {batch_size}')
    indices = rng.integers(
        0, self._size, size=(int(num_batches), int(batch_size)))
    return self._storage[indices]


class TransitionReplay:
  """FIFO replay of original fixed-goal transitions for scalar TD learning."""

  def __init__(
      self,
      capacity: int,
      observation_shape: Sequence[int],
      action_shape: Sequence[int],
  ):
    self._capacity = int(capacity)
    self._obs = np.empty(
        (self._capacity,) + tuple(observation_shape), dtype=np.float32)
    self._action = np.empty(
        (self._capacity,) + tuple(action_shape), dtype=np.float32)
    self._next_obs = np.empty_like(self._obs)
    self._done = np.empty((self._capacity,), dtype=np.float32)
    self._size = 0
    self._cursor = 0

  @property
  def size(self) -> int:
    return self._size

  def add(
      self,
      obs: np.ndarray,
      action: np.ndarray,
      next_obs: np.ndarray,
      done: np.ndarray,
  ) -> None:
    obs = np.asarray(obs, dtype=np.float32).reshape(
        (-1,) + self._obs.shape[1:])
    action = np.asarray(action, dtype=np.float32).reshape(
        (-1,) + self._action.shape[1:])
    next_obs = np.asarray(next_obs, dtype=np.float32).reshape(
        (-1,) + self._next_obs.shape[1:])
    done = np.asarray(done, dtype=np.float32).reshape(-1)
    n = obs.shape[0]
    if not (action.shape[0] == next_obs.shape[0] == done.shape[0] == n):
      raise ValueError('transition fields must have equal leading sizes')
    if n >= self._capacity:
      obs = obs[-self._capacity:]
      action = action[-self._capacity:]
      next_obs = next_obs[-self._capacity:]
      done = done[-self._capacity:]
      n = self._capacity
    first = min(n, self._capacity - self._cursor)
    sl = slice(self._cursor, self._cursor + first)
    self._obs[sl] = obs[:first]
    self._action[sl] = action[:first]
    self._next_obs[sl] = next_obs[:first]
    self._done[sl] = done[:first]
    remaining = n - first
    if remaining:
      self._obs[:remaining] = obs[first:]
      self._action[:remaining] = action[first:]
      self._next_obs[:remaining] = next_obs[first:]
      self._done[:remaining] = done[first:]
    self._cursor = (self._cursor + n) % self._capacity
    self._size = min(self._capacity, self._size + n)

  def sample_batches(
      self,
      num_batches: int,
      batch_size: int,
      rng: np.random.Generator,
  ) -> Dict[str, np.ndarray]:
    if self._size < batch_size:
      raise ValueError(f'replay has {self._size} transitions, need {batch_size}')
    indices = rng.integers(
        0, self._size, size=(int(num_batches), int(batch_size)))
    return {
        'obs': self._obs[indices],
        'action': self._action[indices],
        'next_obs': self._next_obs[indices],
        'done': self._done[indices],
    }


def _tree_copy(tree):
  return jax.tree_util.tree_map(lambda x: x, tree)


def _incremental_update_tree(target, online, rate: float):
  return jax.tree_util.tree_map(
      lambda t, o: (1.0 - rate) * t + rate * o, target, online)


def make_mpo_policy(
    environment_spec: specs.EnvironmentSpec,
    hidden_sizes: Sequence[int],
    init_scale: float,
) -> networks_lib.FeedForwardNetwork:
  """Build the unsquashed Gaussian policy expected by Acme's MPO loss."""
  action_dim = int(np.prod(environment_spec.actions.shape))
  dummy_obs = jax.tree_util.tree_map(
      lambda x: jnp.zeros((1,) + x.shape, x.dtype),
      environment_spec.observations)

  def policy_fn(observation):
    embedding = networks_lib.LayerNormMLP(
        tuple(hidden_sizes), activate_final=True)(observation)
    return networks_lib.MultivariateNormalDiagHead(
        action_dim, init_scale=float(init_scale))(embedding)

  transformed = hk.without_apply_rng(hk.transform(policy_fn))
  return networks_lib.FeedForwardNetwork(
      init=lambda key: transformed.init(key, dummy_obs),
      apply=transformed.apply)


def make_scalar_critic(
    environment_spec: specs.EnvironmentSpec,
    obs_dim: int,
    hidden_sizes: Sequence[int],
) -> networks_lib.FeedForwardNetwork:
  """Build the fixed-goal scalar critic G(s, a); goal is not an input."""
  action_dim = int(np.prod(environment_spec.actions.shape))
  dummy_state = jnp.zeros((1, int(obs_dim)), dtype=jnp.float32)
  dummy_action = jnp.zeros((1, action_dim), dtype=jnp.float32)

  def critic_fn(state, action):
    inputs = jnp.concatenate([state, action], axis=-1)
    embedding = networks_lib.LayerNormMLP(
        tuple(hidden_sizes), activate_final=True)(inputs)
    return hk.Linear(
        1, w_init=hk.initializers.VarianceScaling(1e-4))(embedding)[..., 0]

  transformed = hk.without_apply_rng(hk.transform(critic_fn))
  return networks_lib.FeedForwardNetwork(
      init=lambda key: transformed.init(key, dummy_state, dummy_action),
      apply=transformed.apply)


def make_mpo_update_fn(
    policy_network: networks_lib.FeedForwardNetwork,
    crl_networks: contrastive_networks.ContrastiveNetworks,
    action_min: jnp.ndarray,
    action_max: jnp.ndarray,
    mpo_config: MPOCRLConfig,
    policy_optimizer: optax.GradientTransformation,
    dual_optimizer: optax.GradientTransformation,
):
  """Create one MPO update using target CRL values and target policy samples."""
  mpo_loss = mpo_losses.MPO(
      epsilon=mpo_config.epsilon,
      epsilon_mean=mpo_config.epsilon_mean,
      epsilon_stddev=mpo_config.epsilon_stddev,
      epsilon_penalty=mpo_config.epsilon_penalty,
      init_log_temperature=mpo_config.init_log_temperature,
      init_log_alpha_mean=mpo_config.init_log_alpha_mean,
      init_log_alpha_stddev=mpo_config.init_log_alpha_stddev,
      per_dim_constraining=mpo_config.per_dim_constraining,
      action_penalization=mpo_config.action_penalization)
  num_samples = int(mpo_config.num_action_samples)
  target_period = int(mpo_config.target_update_period)
  target_rate = float(mpo_config.target_update_rate)

  def loss_fn(policy_params, dual_params, target_policy_params,
              target_q_params, observations, key):
    online_dist = policy_network.apply(policy_params, observations)
    target_dist = policy_network.apply(target_policy_params, observations)
    sampled_actions = target_dist.sample(num_samples, seed=key)  # (N, B, A)

    # CRL was trained only inside the action specification. As in Acme MPO,
    # score clipped actions while leaving the raw samples available to MO-MPO's
    # out-of-bounds penalty.
    clipped_actions = jnp.clip(sampled_actions, action_min, action_max)
    n, b, a = clipped_actions.shape
    tiled_obs = jnp.broadcast_to(
        observations[None], (n,) + observations.shape).reshape(
            (n * b,) + observations.shape[1:])
    flat_actions = clipped_actions.reshape((n * b, a))
    sa_repr, g_repr, _ = crl_networks.repr_fn(
        target_q_params, tiled_obs, flat_actions)
    q_values = jnp.sum(sa_repr * g_repr, axis=-1).reshape(n, b)
    q_values = jax.lax.stop_gradient(q_values)

    loss, stats = mpo_loss(
        params=dual_params,
        online_action_distribution=online_dist,
        target_action_distribution=target_dist,
        actions=sampled_actions,
        q_values=q_values)
    metrics = {name: jnp.mean(value) if value is not None else jnp.nan
               for name, value in stats._asdict().items()}
    metrics['total_loss'] = jnp.mean(loss)
    metrics['q_mean'] = jnp.mean(q_values)
    return jnp.mean(loss), metrics

  grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

  @jax.jit
  def update(state: MPOCRLTrainingState, observations: jnp.ndarray):
    key, sample_key = jax.random.split(state.key)
    (_, metrics), (policy_grads, dual_grads) = grad_fn(
        state.policy_params,
        state.dual_params,
        state.target_policy_params,
        state.target_q_params,
        observations,
        sample_key)
    policy_updates, policy_opt_state = policy_optimizer.update(
        policy_grads, state.policy_optimizer_state, state.policy_params)
    dual_updates, dual_opt_state = dual_optimizer.update(
        dual_grads, state.dual_optimizer_state, state.dual_params)
    policy_params = optax.apply_updates(state.policy_params, policy_updates)
    dual_params = optax.apply_updates(state.dual_params, dual_updates)
    dual_params = mpo_losses.clip_mpo_params(
        dual_params, mpo_config.per_dim_constraining)
    policy_steps = state.policy_steps + 1
    if target_period > 0:
      target_policy_params = optax.periodic_update(
          policy_params, state.target_policy_params, policy_steps, target_period)
    else:
      target_policy_params = optax.incremental_update(
          policy_params, state.target_policy_params, target_rate)
    metrics['policy_grad_norm'] = optax.global_norm(policy_grads)
    metrics['dual_grad_norm'] = optax.global_norm(dual_grads)
    return state._replace(
        policy_params=policy_params,
        target_policy_params=target_policy_params,
        dual_params=dual_params,
        policy_optimizer_state=policy_opt_state,
        dual_optimizer_state=dual_opt_state,
        policy_steps=policy_steps,
        key=key), metrics

  return update, mpo_loss


def make_mpo_td_update_fn(
    policy_network: networks_lib.FeedForwardNetwork,
    crl_networks: contrastive_networks.ContrastiveNetworks,
    critic_network: networks_lib.FeedForwardNetwork,
    obs_dim: int,
    discount: float,
    action_min: jnp.ndarray,
    action_max: jnp.ndarray,
    mpo_config: MPOCRLConfig,
    policy_optimizer: optax.GradientTransformation,
    dual_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
):
  """Jointly update scalar G(s,a), MPO policy, and MPO dual variables.

  CRL target scores provide a fixed-goal reward.  Expected-SARSA targets
  average G_target over actions sampled from the target policy.  MPO policy
  improvement also uses G_target, never the online G being optimized.
  """
  mpo_loss = mpo_losses.MPO(
      epsilon=mpo_config.epsilon,
      epsilon_mean=mpo_config.epsilon_mean,
      epsilon_stddev=mpo_config.epsilon_stddev,
      epsilon_penalty=mpo_config.epsilon_penalty,
      init_log_temperature=mpo_config.init_log_temperature,
      init_log_alpha_mean=mpo_config.init_log_alpha_mean,
      init_log_alpha_stddev=mpo_config.init_log_alpha_stddev,
      per_dim_constraining=mpo_config.per_dim_constraining,
      action_penalization=mpo_config.action_penalization)
  num_samples = int(mpo_config.num_action_samples)
  bootstrap_samples = int(mpo_config.bootstrap_action_samples)
  target_period = int(mpo_config.target_update_period)
  target_rate = float(mpo_config.target_update_rate)
  critic_target_period = int(mpo_config.critic_target_update_period)
  critic_target_rate = float(mpo_config.critic_target_update_rate)
  normalize_reward = bool(mpo_config.normalize_critic_reward)
  reward_norm_rate = float(mpo_config.critic_reward_norm_rate)
  obs_dim = int(obs_dim)
  discount = float(discount)

  def _target_critic_values(target_critic_params, observations, actions):
    """Evaluate G_target for actions shaped (N, B, A)."""
    clipped_actions = jnp.clip(actions, action_min, action_max)
    n, b, a = clipped_actions.shape
    tiled_state = jnp.broadcast_to(
        observations[None, :, :obs_dim],
        (n, b, obs_dim)).reshape(n * b, obs_dim)
    flat_actions = clipped_actions.reshape(n * b, a)
    return critic_network.apply(
        target_critic_params, tiled_state, flat_actions).reshape(n, b)

  def critic_loss_fn(
      critic_params,
      target_critic_params,
      target_policy_params,
      reward,
      raw_reward,
      reward_std,
      batch,
      key,
  ):
    obs = batch['obs']
    action = batch['action']
    next_obs = batch['next_obs']
    done = batch['done']

    next_dist = policy_network.apply(target_policy_params, next_obs)
    next_actions = next_dist.sample(bootstrap_samples, seed=key)
    next_values = _target_critic_values(
        target_critic_params, next_obs, next_actions)
    next_value = jnp.mean(next_values, axis=0)
    td_target = jax.lax.stop_gradient(
        reward + discount * (1.0 - done) * next_value)
    prediction = critic_network.apply(
        critic_params, obs[:, :obs_dim], action)
    td_error = td_target - prediction
    loss = 0.5 * jnp.mean(jnp.square(td_error))
    metrics = {
        'critic_loss': loss,
        'critic_prediction_mean': jnp.mean(prediction),
        'critic_reward_mean': jnp.mean(reward),
        'critic_reward_raw_mean': jnp.mean(raw_reward),
        'critic_reward_norm_std': reward_std,
        'critic_target_mean': jnp.mean(td_target),
        'critic_td_error_abs': jnp.mean(jnp.abs(td_error)),
    }
    return loss, metrics

  critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)

  def policy_loss_fn(
      policy_params,
      dual_params,
      target_policy_params,
      target_critic_params,
      observations,
      key,
  ):
    online_dist = policy_network.apply(policy_params, observations)
    target_dist = policy_network.apply(target_policy_params, observations)
    sampled_actions = target_dist.sample(num_samples, seed=key)
    q_values = jax.lax.stop_gradient(_target_critic_values(
        target_critic_params, observations, sampled_actions))
    loss, stats = mpo_loss(
        params=dual_params,
        online_action_distribution=online_dist,
        target_action_distribution=target_dist,
        actions=sampled_actions,
        q_values=q_values)
    metrics = {name: jnp.mean(value) if value is not None else jnp.nan
               for name, value in stats._asdict().items()}
    metrics['total_loss'] = jnp.mean(loss)
    metrics['q_mean'] = jnp.mean(q_values)
    return jnp.mean(loss), metrics

  policy_grad_fn = jax.value_and_grad(
      policy_loss_fn, argnums=(0, 1), has_aux=True)

  @jax.jit
  def update(state: MPOCRLTrainingState, batch: Dict[str, jnp.ndarray]):
    key, critic_key, policy_key = jax.random.split(state.key, 3)
    crl_params = state.q_params['crl']
    critic_params = state.q_params['critic']
    target_crl_params = state.target_q_params['crl']
    target_critic_params = state.target_q_params['critic']

    # Track the slowly moving CRL reward with EMA moments. Initializing from
    # the first minibatch avoids a large artificial transient from zero-valued
    # moments. Centering and scaling make G's target insensitive to CRL logit
    # offset/scale drift.
    sa_repr, g_repr, _ = crl_networks.repr_fn(
        target_crl_params, batch['obs'], batch['action'])
    raw_reward = jax.lax.stop_gradient(
        jnp.sum(sa_repr * g_repr, axis=-1))
    reward_mean = state.q_params['critic_reward_mean']
    reward_var = state.q_params['critic_reward_var']
    reward_initialized = state.q_params['critic_reward_initialized']
    if normalize_reward:
      batch_mean = jnp.mean(raw_reward)
      batch_var = jnp.var(raw_reward)
      mean_delta = batch_mean - reward_mean
      ema_mean = reward_mean + reward_norm_rate * mean_delta
      ema_var = (
          (1.0 - reward_norm_rate) * reward_var
          + reward_norm_rate * batch_var
          + reward_norm_rate * (1.0 - reward_norm_rate)
          * jnp.square(mean_delta))
      reward_mean = jnp.where(
          reward_initialized, ema_mean, batch_mean)
      reward_var = jnp.where(
          reward_initialized, ema_var, batch_var)
      reward_initialized = jnp.asarray(True)
      reward_std = jnp.sqrt(jnp.maximum(reward_var, 1e-6))
      reward = (raw_reward - reward_mean) / reward_std
    else:
      reward_std = jnp.asarray(1.0, dtype=raw_reward.dtype)
      reward = raw_reward

    (_, critic_metrics), critic_grads = critic_grad_fn(
        critic_params,
        target_critic_params,
        state.target_policy_params,
        reward,
        raw_reward,
        reward_std,
        batch,
        critic_key)
    (_, policy_metrics), (policy_grads, dual_grads) = policy_grad_fn(
        state.policy_params,
        state.dual_params,
        state.target_policy_params,
        target_critic_params,
        batch['obs'],
        policy_key)

    critic_updates, critic_opt_state = critic_optimizer.update(
        critic_grads, state.q_optimizer_state['critic'], critic_params)
    policy_updates, policy_opt_state = policy_optimizer.update(
        policy_grads, state.policy_optimizer_state, state.policy_params)
    dual_updates, dual_opt_state = dual_optimizer.update(
        dual_grads, state.dual_optimizer_state, state.dual_params)
    critic_params = optax.apply_updates(critic_params, critic_updates)
    policy_params = optax.apply_updates(state.policy_params, policy_updates)
    dual_params = optax.apply_updates(state.dual_params, dual_updates)
    dual_params = mpo_losses.clip_mpo_params(
        dual_params, mpo_config.per_dim_constraining)

    policy_steps = state.policy_steps + 1
    if target_period > 0:
      target_policy_params = optax.periodic_update(
          policy_params, state.target_policy_params, policy_steps, target_period)
    else:
      target_policy_params = optax.incremental_update(
          policy_params, state.target_policy_params, target_rate)
    if critic_target_period > 0:
      target_critic_params = optax.periodic_update(
          critic_params,
          target_critic_params,
          policy_steps,
          critic_target_period)
    else:
      target_critic_params = optax.incremental_update(
          critic_params, target_critic_params, critic_target_rate)

    metrics = dict(policy_metrics)
    metrics.update(critic_metrics)
    # These are pre-clipping norms.  The optimizer chains below apply the
    # configured global-norm clips before Adam.
    metrics['policy_grad_norm'] = optax.global_norm(policy_grads)
    metrics['dual_grad_norm'] = optax.global_norm(dual_grads)
    metrics['critic_grad_norm'] = optax.global_norm(critic_grads)
    return state._replace(
        policy_params=policy_params,
        target_policy_params=target_policy_params,
        dual_params=dual_params,
        policy_optimizer_state=policy_opt_state,
        dual_optimizer_state=dual_opt_state,
        q_params={
            'crl': crl_params,
            'critic': critic_params,
            'critic_reward_mean': reward_mean,
            'critic_reward_var': reward_var,
            'critic_reward_initialized': reward_initialized,
        },
        target_q_params={
            'crl': target_crl_params,
            'critic': target_critic_params,
        },
        q_optimizer_state={
            'crl': state.q_optimizer_state['crl'],
            'critic': critic_opt_state,
        },
        policy_steps=policy_steps,
        key=key), metrics

  return update, mpo_loss


def _save_checkpoint(path: str, state: MPOCRLTrainingState,
                     iteration: int, global_step: int) -> None:
  payload = {
      'state': state,
      'iteration': int(iteration),
      'global_step': int(global_step),
  }
  tmp_path = path + '.tmp'
  with open(tmp_path, 'wb') as handle:
    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
  os.replace(tmp_path, path)


def _prune_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
  if keep_last <= 0:
    return
  names = sorted(
      (name for name in os.listdir(checkpoint_dir)
       if name.startswith('ckpt_iter_') and name.endswith('.pkl')),
      key=lambda name: int(name[len('ckpt_iter_'):-len('.pkl')]))
  for name in names[:-keep_last]:
    os.remove(os.path.join(checkpoint_dir, name))


def _mean_metrics(metrics: Dict[str, list]) -> Dict[str, float]:
  return {
      key: float(np.mean(values)) if values else float('nan')
      for key, values in metrics.items()
  }


def _append_metrics(destination: Dict[str, list], metrics: Dict[str, Any]) -> None:
  for key, value in metrics.items():
    destination.setdefault(key, []).append(float(np.asarray(value)))


def _mean_device_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
  """Transfer one reduced metric value per key after a compiled update scan."""
  reduced = jax.device_get(
      jax.tree_util.tree_map(lambda value: jnp.mean(value), metrics))
  return {key: float(value) for key, value in reduced.items()}


def run_mpo_crl_training(
    config,
    mpo_config: MPOCRLConfig,
    env_factory: Callable,
    eval_env_factory: Callable,
    network_factory: Callable,
    logger_fn: Callable,
    total_steps: int,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
    builderbench_kwargs: Optional[Dict[str, Any]] = None,
) -> MPOCRLTrainingState:
  """Run online target-policy collection with MPO and geometric CRL updates."""
  probe_env = env_factory(seed)
  environment_spec = specs.make_environment_spec(probe_env)
  crl_networks = network_factory(spec=environment_spec)
  action_spec = environment_spec.actions
  action_min = jnp.asarray(action_spec.minimum, dtype=jnp.float32)
  action_max = jnp.asarray(action_spec.maximum, dtype=jnp.float32)
  del probe_env

  policy_network = make_mpo_policy(
      environment_spec,
      hidden_sizes=mpo_config.policy_hidden_sizes,
      init_scale=mpo_config.policy_init_scale)
  critic_network = (
      make_scalar_critic(
          environment_spec,
          obs_dim=int(config.obs_dim),
          hidden_sizes=mpo_config.critic_hidden_sizes)
      if mpo_config.use_td_critic else None)
  policy_optimizer = optax.chain(
      optax.clip_by_global_norm(mpo_config.policy_grad_norm_clip),
      optax.adam(mpo_config.policy_learning_rate))
  dual_optimizer = optax.chain(
      optax.clip_by_global_norm(mpo_config.policy_grad_norm_clip),
      optax.adam(mpo_config.dual_learning_rate))
  critic_optimizer = optax.chain(
      optax.clip_by_global_norm(mpo_config.critic_grad_norm_clip),
      optax.adam(mpo_config.critic_learning_rate))
  q_optimizer = optax.adam(float(config.learning_rate))

  key = jax.random.PRNGKey(seed)
  key, policy_key, q_key, critic_key = jax.random.split(key, 4)
  policy_params = policy_network.init(policy_key)
  crl_params = crl_networks.q_network.init(q_key)
  critic_params = (
      critic_network.init(critic_key) if critic_network is not None else None)
  mpo_module = mpo_losses.MPO(
      epsilon=mpo_config.epsilon,
      epsilon_mean=mpo_config.epsilon_mean,
      epsilon_stddev=mpo_config.epsilon_stddev,
      epsilon_penalty=mpo_config.epsilon_penalty,
      init_log_temperature=mpo_config.init_log_temperature,
      init_log_alpha_mean=mpo_config.init_log_alpha_mean,
      init_log_alpha_stddev=mpo_config.init_log_alpha_stddev,
      per_dim_constraining=mpo_config.per_dim_constraining,
      action_penalization=mpo_config.action_penalization)
  action_dim = int(np.prod(action_spec.shape))
  dual_params = mpo_module.init_params(action_dim=action_dim)
  if mpo_config.use_td_critic:
    q_params = {
        'crl': crl_params,
        'critic': critic_params,
        'critic_reward_mean': jnp.asarray(0.0, dtype=jnp.float32),
        'critic_reward_var': jnp.asarray(1.0, dtype=jnp.float32),
        'critic_reward_initialized': jnp.asarray(False),
    }
    target_q_params = {
        'crl': _tree_copy(crl_params),
        'critic': _tree_copy(critic_params),
    }
    q_optimizer_state = {
        'crl': q_optimizer.init(crl_params),
        'critic': critic_optimizer.init(critic_params),
    }
  else:
    q_params = crl_params
    target_q_params = _tree_copy(crl_params)
    q_optimizer_state = q_optimizer.init(crl_params)
  state = MPOCRLTrainingState(
      policy_params=policy_params,
      target_policy_params=_tree_copy(policy_params),
      dual_params=dual_params,
      policy_optimizer_state=policy_optimizer.init(policy_params),
      dual_optimizer_state=dual_optimizer.init(dual_params),
      q_params=q_params,
      target_q_params=target_q_params,
      q_optimizer_state=q_optimizer_state,
      policy_steps=jnp.asarray(0, dtype=jnp.int32),
      key=key)

  if mpo_config.use_td_critic:
    mpo_update, _ = make_mpo_td_update_fn(
        policy_network, crl_networks, critic_network,
        obs_dim=int(config.obs_dim), discount=float(config.discount),
        action_min=action_min, action_max=action_max, mpo_config=mpo_config,
        policy_optimizer=policy_optimizer, dual_optimizer=dual_optimizer,
        critic_optimizer=critic_optimizer)
  else:
    mpo_update, _ = make_mpo_update_fn(
        policy_network, crl_networks, action_min, action_max, mpo_config,
        policy_optimizer, dual_optimizer)
  backward = (
      str(getattr(config, 'ppo_crl_loss_direction', 'forward')).lower()
      == 'backward')
  crl_update = ppo_learner.make_crl_update_fn(
      crl_networks, q_optimizer, backward=backward)

  @jax.jit
  def mpo_update_scan(
      scan_state: MPOCRLTrainingState,
      replay_batches,
  ):
    """Run all replay updates sequentially on device without host syncs."""
    def body(carry, batch):
      return mpo_update(carry, batch)
    return jax.lax.scan(body, scan_state, replay_batches)

  @jax.jit
  def crl_update_scan(
      scan_state: MPOCRLTrainingState,
      batches: Dict[str, jnp.ndarray],
  ):
    """Run CRL updates and target updates in one compiled device scan."""
    def body(carry, batch):
      state_key, update_key = jax.random.split(carry.key)
      current_q_params = (
          carry.q_params['crl'] if mpo_config.use_td_critic
          else carry.q_params)
      current_target_q_params = (
          carry.target_q_params['crl'] if mpo_config.use_td_critic
          else carry.target_q_params)
      current_q_opt_state = (
          carry.q_optimizer_state['crl'] if mpo_config.use_td_critic
          else carry.q_optimizer_state)
      q_params, q_opt_state, metrics = crl_update(
          current_q_params, current_q_opt_state, batch, update_key)
      target_q_params = _incremental_update_tree(
          current_target_q_params, q_params,
          mpo_config.repr_target_update_rate)
      if mpo_config.use_td_critic:
        all_q_params = dict(carry.q_params)
        all_q_params['crl'] = q_params
        all_target_q_params = dict(carry.target_q_params)
        all_target_q_params['crl'] = target_q_params
        all_q_opt_state = dict(carry.q_optimizer_state)
        all_q_opt_state['crl'] = q_opt_state
      else:
        all_q_params = q_params
        all_target_q_params = target_q_params
        all_q_opt_state = q_opt_state
      carry = carry._replace(
          q_params=all_q_params,
          target_q_params=all_target_q_params,
          q_optimizer_state=all_q_opt_state,
          key=state_key)
      return carry, metrics
    return jax.lax.scan(body, scan_state, batches)

  env_name = str(getattr(config, 'env_name', '') or '')
  use_jax_bb = env_name.startswith('builderbench_')
  if use_jax_bb:
    import importlib.util as _ilu
    _jax_vec_path = os.path.join(
        str(_ROOT), 'envs', 'builderbench_jax_vec.py')
    _spec = _ilu.spec_from_file_location('builderbench_jax_vec', _jax_vec_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _bb_kw = dict(builderbench_kwargs or {})
    vec_env = _mod.JaxBuilderBenchVecEnv(
        env_name=env_name,
        num_envs=int(mpo_config.num_envs),
        seed=int(seed * 31),
        use_pd=bool(_bb_kw.get('builderbench_use_pd', False)),
        pd_duration=int(_bb_kw.get('builderbench_pd_duration', 5)),
        pd_filter_policy_obs=bool(
            _bb_kw.get('builderbench_pd_filter_policy_obs', True)),
        fixed_target_goal=_bb_kw.get('fixed_target_goal'),
        permute_start_boxes=bool(
            _bb_kw.get('builderbench_permute_start_boxes', True)),
    )
    print(f'[mpo-crl] using JAX-batched BuilderBench vec env '
          f'(E={mpo_config.num_envs})')
  else:
    vec_env = ppo_learner.VecEnv(
        env_factory, int(mpo_config.num_envs), seed=seed * 31)
  episode_replay = ppo_learner.EpisodeReplay(
      capacity=int(config.max_replay_size),
      obs_dim=int(config.obs_dim),
      discount=float(config.discount),
      start_index=int(config.start_index),
      end_index=int(config.end_index))
  policy_replay = (
      TransitionReplay(
          int(config.max_replay_size),
          vec_env.observation_shape,
          action_spec.shape)
      if mpo_config.use_td_critic
      else ObservationReplay(
          int(config.max_replay_size), vec_env.observation_shape))
  rng = np.random.default_rng(seed + 12345)

  # ---- uniform-sampling goal bounds (mirrors ppo_learner.py) -------------
  uniform_sampling = bool(getattr(config, 'uniform_sampling', False))
  goal_low = goal_high = None
  if uniform_sampling:
    import env_utils as _env_utils
    if hasattr(vec_env, 'uniform_goal_obs_bounds'):
      glo, ghi = vec_env.uniform_goal_obs_bounds()
      si, ei = int(config.start_index), int(config.end_index)
      if ei == -1:
        ei = int(config.obs_dim)
      goal_low = np.asarray(glo[si:ei], dtype=np.float32)
      goal_high = np.asarray(ghi[si:ei], dtype=np.float32)
    else:
      goal_low, goal_high = _env_utils.resolve_uniform_goal_bounds(
          environment_spec, vec_env._envs[0], int(config.obs_dim),
          int(config.start_index), int(config.end_index))
    print(f'[mpo-crl] uniform_sampling: goal_low={goal_low}, '
          f'goal_high={goal_high}')

  start_iteration = 0
  global_step = 0
  if checkpoint_dir is not None:
    latest = os.path.join(checkpoint_dir, 'latest.pkl')
    if os.path.exists(latest):
      with open(latest, 'rb') as handle:
        payload = pickle.load(handle)
      loaded_state = payload['state']
      loaded_has_td = (
          isinstance(loaded_state.q_params, dict)
          and 'crl' in loaded_state.q_params
          and 'critic' in loaded_state.q_params)
      if mpo_config.use_td_critic and not loaded_has_td:
        # Allow starting the new TD critic from an existing direct-CRL
        # checkpoint without changing the checkpoint tuple schema.
        state = loaded_state._replace(
            q_params={
                'crl': loaded_state.q_params,
                'critic': state.q_params['critic'],
            },
            target_q_params={
                'crl': loaded_state.target_q_params,
                'critic': state.target_q_params['critic'],
            },
            q_optimizer_state={
                'crl': loaded_state.q_optimizer_state,
                'critic': state.q_optimizer_state['critic'],
            })
        print('[mpo-crl] initialized a fresh TD critic while resuming '
              'policy/CRL parameters')
      elif not mpo_config.use_td_critic and loaded_has_td:
        raise ValueError(
            'checkpoint contains a TD critic; resume with --mpo_use_td_critic')
      else:
        state = loaded_state
      if mpo_config.use_td_critic and 'critic_reward_mean' not in state.q_params:
        # Older TD-critic checkpoints predate reward normalization.
        loaded_q_params = dict(state.q_params)
        loaded_q_params.update({
            'critic_reward_mean': jnp.asarray(0.0, dtype=jnp.float32),
            'critic_reward_var': jnp.asarray(1.0, dtype=jnp.float32),
            'critic_reward_initialized': jnp.asarray(False),
        })
        state = state._replace(q_params=loaded_q_params)
      start_iteration = int(payload['iteration']) + 1
      global_step = int(payload['global_step'])
      print(f'[mpo-crl] resumed at iteration={start_iteration}, '
            f'global_step={global_step}; replay will refill')

  @jax.jit
  def target_action(params, observation, sample_key):
    distribution = policy_network.apply(params, observation)
    return distribution.sample(seed=sample_key)

  @jax.jit
  def target_mode(params, observation):
    return policy_network.apply(params, observation).mode()

  bb_mpo_unroll = (
      vec_env.compile_mpo_unroll(
          target_action, int(mpo_config.rollout_length))
      if use_jax_bb else None)
  bb_eval_vec = None
  bb_eval_unroll = None
  if use_jax_bb and mpo_config.eval_interval > 0:
    bb_eval_vec = _mod.JaxBuilderBenchVecEnv(
        env_name=env_name,
        num_envs=int(mpo_config.eval_episodes),
        seed=int(seed * 31 + 77),
        use_pd=bool(_bb_kw.get('builderbench_use_pd', False)),
        pd_duration=int(_bb_kw.get('builderbench_pd_duration', 5)),
        pd_filter_policy_obs=bool(
            _bb_kw.get('builderbench_pd_filter_policy_obs', True)),
        fixed_target_goal=_bb_kw.get('fixed_target_goal'),
        permute_start_boxes=bool(
            _bb_kw.get('builderbench_permute_start_boxes', True)),
    )
    bb_eval_unroll = bb_eval_vec.compile_eval_unroll(
        target_mode, unroll_length=bb_eval_vec.episode_length)
    print(f'[mpo-crl] BuilderBench eval: jax.lax.scan '
          f'(E={mpo_config.eval_episodes}, '
          f'ep_len={bb_eval_vec.episode_length})')

  obs = vec_env.reset()
  num_envs = vec_env.num_envs
  episode_obs = [[obs[i].copy()] for i in range(num_envs)]
  episode_actions = [[] for _ in range(num_envs)]
  episode_returns = np.zeros(num_envs, dtype=np.float32)
  episode_lengths = np.zeros(num_envs, dtype=np.int32)
  recent_returns: list[float] = []
  recent_lengths: list[int] = []
  # BuilderBench: track episode success from env metrics (dense reward ≠ success).
  track_train_success = bool(use_jax_bb and hasattr(vec_env, 'last_success'))
  recent_success: list[float] = []
  recent_very_hard_success: list[float] = []
  ep_success_max = np.zeros(num_envs, dtype=np.float32)
  ep_very_hard_success_max = np.zeros(num_envs, dtype=np.float32)

  learner_logger = logger_fn(label='learner')
  eval_logger = logger_fn(label='eval')
  # Keep observers alive across evaluation episodes and rounds. Their internal
  # histories are what define success_1000 and smoothed distance metrics.
  if env_name.lower() == 'riverswim':
    eval_success_observer = (
        contrastive_utils.RiverSwimGoalVisitSuccessObserver(
            obs_dim=int(config.obs_dim)))
  elif env_name.lower().startswith('builderbench_'):
    eval_success_observer = contrastive_utils.BuilderBenchSuccessObserver()
  else:
    eval_success_observer = contrastive_utils.SuccessObserver()
  eval_distance_observer = contrastive_utils.DistanceObserver(
      obs_dim=int(config.obs_dim),
      start_index=int(config.start_index),
      end_index=int(config.end_index))
  if checkpoint_dir is not None and mpo_config.checkpoint_interval > 0:
    os.makedirs(checkpoint_dir, exist_ok=True)

  iterations = int(total_steps) // (
      int(mpo_config.rollout_length) * num_envs)
  start_time = time.time()
  for iteration in range(start_iteration, iterations):
    iteration_start = time.perf_counter()
    if bb_mpo_unroll is not None:
      # One policy+environment launch for the entire rollout. Device outputs
      # cross to the host once for replay insertion and episode accounting.
      (vec_env._state, collect_key), steps_j = bb_mpo_unroll(
          vec_env._state, state.target_policy_params, state.key)
      state = state._replace(key=collect_key)
      steps = jax.device_get(steps_j)
      rollout_obs = np.asarray(steps['obs'], dtype=np.float32)
      rollout_actions = np.asarray(steps['actions'], dtype=np.float32)
      env_rewards = np.asarray(steps['env_rew'], dtype=np.float32)
      rollout_success = np.asarray(steps['success'], dtype=np.float32)
      if 'very_hard_success' in steps:
        rollout_very_hard = np.asarray(
            steps['very_hard_success'], dtype=np.float32)
      else:
        rollout_very_hard = np.zeros_like(rollout_success)
      rollout_dones = np.asarray(steps['step_dones'], dtype=bool)
      terminal_obs_rollout = np.asarray(
          steps['terminal_obs'], dtype=np.float32)
      next_obs_rollout = np.asarray(steps['next_obs'], dtype=np.float32)
      if mpo_config.use_td_critic:
        replay_next_obs = np.where(
            rollout_dones[..., None],
            terminal_obs_rollout,
            next_obs_rollout)
        policy_replay.add(
            rollout_obs, rollout_actions, replay_next_obs, rollout_dones)
      else:
        policy_replay.add(rollout_obs)

      # BuilderBench PD jobs use T == episode length, so all synchronized
      # environments normally terminate once at the final scan step. Handle
      # that common case with one Python loop over E instead of T*E.
      aligned_episodes = (
          not np.any(rollout_dones[:-1])
          and np.all(rollout_dones[-1])
          and all(not actions for actions in episode_actions))
      if aligned_episodes:
        rollout_returns = np.sum(env_rewards, axis=0)
        rollout_successes = np.max(rollout_success, axis=0) >= 0.5
        rollout_very_hard_successes = np.max(rollout_very_hard, axis=0) >= 0.5
        for index in range(num_envs):
          episode_replay.add_episode(
              np.concatenate(
                  [rollout_obs[:, index],
                   terminal_obs_rollout[-1, index][None]],
                  axis=0),
              rollout_actions[:, index])
        recent_returns.extend(rollout_returns.astype(float).tolist())
        recent_lengths.extend(
            [int(mpo_config.rollout_length)] * num_envs)
        recent_success.extend(
            rollout_successes.astype(float).tolist())
        recent_very_hard_success.extend(
            rollout_very_hard_successes.astype(float).tolist())
        recent_returns[:] = recent_returns[-100:]
        recent_lengths[:] = recent_lengths[-100:]
        recent_success[:] = recent_success[-1000:]
        recent_very_hard_success[:] = recent_very_hard_success[-1000:]
        obs = next_obs_rollout[-1]
        episode_obs = [[obs[index].copy()] for index in range(num_envs)]
      else:
        # Generic fallback for early termination or a resumed partial episode.
        for timestep in range(int(mpo_config.rollout_length)):
          for index in range(num_envs):
            episode_actions[index].append(
                rollout_actions[timestep, index].copy())
            episode_returns[index] += float(env_rewards[timestep, index])
            episode_lengths[index] += 1
            ep_success_max[index] = max(
                ep_success_max[index],
                float(rollout_success[timestep, index]))
            ep_very_hard_success_max[index] = max(
                ep_very_hard_success_max[index],
                float(rollout_very_hard[timestep, index]))
            if rollout_dones[timestep, index]:
              episode_obs[index].append(
                  terminal_obs_rollout[timestep, index].copy())
              episode_replay.add_episode(
                  np.stack(episode_obs[index]),
                  np.stack(episode_actions[index]))
              recent_returns.append(float(episode_returns[index]))
              recent_lengths.append(int(episode_lengths[index]))
              recent_success.append(
                  float(ep_success_max[index] >= 0.5))
              recent_very_hard_success.append(
                  float(ep_very_hard_success_max[index] >= 0.5))
              recent_returns[:] = recent_returns[-100:]
              recent_lengths[:] = recent_lengths[-100:]
              recent_success[:] = recent_success[-1000:]
              recent_very_hard_success[:] = recent_very_hard_success[-1000:]
              episode_obs[index] = [
                  next_obs_rollout[timestep, index].copy()]
              episode_actions[index] = []
              episode_returns[index] = 0.0
              episode_lengths[index] = 0
              ep_success_max[index] = 0.0
              ep_very_hard_success_max[index] = 0.0
            else:
              episode_obs[index].append(
                  next_obs_rollout[timestep, index].copy())
        obs = next_obs_rollout[-1]
      global_step += int(mpo_config.rollout_length) * num_envs
    else:
      env_rewards = []
      for _ in range(int(mpo_config.rollout_length)):
        next_key, action_key = jax.random.split(state.key)
        state = state._replace(key=next_key)
        raw_action = np.asarray(target_action(
            state.target_policy_params, jnp.asarray(obs), action_key))
        action = np.clip(
            np.nan_to_num(raw_action, nan=0.0),
            np.asarray(action_spec.minimum),
            np.asarray(action_spec.maximum)).astype(np.float32)
        next_obs, reward, dones, terminal_obs, _ = vec_env.step(action)
        if mpo_config.use_td_critic:
          replay_next_obs = np.where(
              np.asarray(dones)[:, None],
              np.asarray(terminal_obs),
              np.asarray(next_obs))
          policy_replay.add(obs, action, replay_next_obs, dones)
        else:
          policy_replay.add(obs)
        env_rewards.append(reward)
        for index in range(num_envs):
          episode_actions[index].append(action[index].copy())
          episode_returns[index] += float(reward[index])
          episode_lengths[index] += 1
          if dones[index]:
            episode_obs[index].append(terminal_obs[index].copy())
            episode_replay.add_episode(
                np.stack(episode_obs[index]),
                np.stack(episode_actions[index]))
            recent_returns.append(float(episode_returns[index]))
            recent_lengths.append(int(episode_lengths[index]))
            recent_returns[:] = recent_returns[-100:]
            recent_lengths[:] = recent_lengths[-100:]
            episode_obs[index] = [next_obs[index].copy()]
            episode_actions[index] = []
            episode_returns[index] = 0.0
            episode_lengths[index] = 0
          else:
            episode_obs[index].append(next_obs[index].copy())
        obs = next_obs
        global_step += num_envs

    collect_seconds = time.perf_counter() - iteration_start
    crl_start = time.perf_counter()
    crl_metrics_mean: Dict[str, float] = {}
    if (episode_replay.size >= int(mpo_config.min_replay_size)
        and episode_replay.num_episodes > 0):
      if uniform_sampling:
        crl_batch_list = [
            episode_replay.sample_with_uniform_negatives(
                int(config.batch_size), rng, goal_low, goal_high)
            for _ in range(int(mpo_config.crl_updates_per_iter))
        ]
      else:
        crl_batch_list = [
            episode_replay.sample(int(config.batch_size), rng)
            for _ in range(int(mpo_config.crl_updates_per_iter))
        ]
      crl_batches = {
          key_: jnp.asarray(
              np.stack([batch[key_] for batch in crl_batch_list], axis=0))
          for key_ in crl_batch_list[0]
      }
      state, crl_metrics = crl_update_scan(state, crl_batches)
      crl_metrics_mean = _mean_device_metrics(crl_metrics)

    crl_seconds = time.perf_counter() - crl_start
    mpo_start = time.perf_counter()
    mpo_metrics_mean: Dict[str, float] = {}
    if policy_replay.size >= max(
        int(mpo_config.min_replay_size), int(mpo_config.policy_batch_size)):
      sampled_batches = policy_replay.sample_batches(
          int(mpo_config.policy_updates_per_iter),
          int(mpo_config.policy_batch_size),
          rng)
      policy_batches = jax.tree_util.tree_map(jnp.asarray, sampled_batches)
      state, mpo_metrics = mpo_update_scan(state, policy_batches)
      mpo_metrics_mean = _mean_device_metrics(mpo_metrics)
    mpo_seconds = time.perf_counter() - mpo_start

    elapsed = time.time() - start_time
    log = {
        'iteration': iteration,
        'learner_steps': iteration,
        'global_step': global_step,
        'sps': global_step / max(elapsed, 1e-6),
        'policy_replay_size': policy_replay.size,
        'crl_replay_size': episode_replay.size,
        'crl_replay_episodes': episode_replay.num_episodes,
        'reward_env_mean': float(np.mean(env_rewards)),
        'ep_return_mean': (
            float(np.mean(recent_returns)) if recent_returns else float('nan')),
        'ep_length_mean': (
            float(np.mean(recent_lengths)) if recent_lengths else float('nan')),
        'mpo/policy_steps': int(state.policy_steps),
        'timing/collect_seconds': collect_seconds,
        'timing/crl_seconds': crl_seconds,
        'timing/mpo_seconds': mpo_seconds,
        'timing/iteration_seconds': time.perf_counter() - iteration_start,
    }
    if track_train_success:
      # Seed columns on first write so CSV keeps train success for the run.
      log['train_success_mean'] = (
          float(np.mean(recent_success[-100:]))
          if recent_success else float('nan'))
      log['train_success_1000'] = (
          float(np.mean(recent_success[-1000:]))
          if recent_success else float('nan'))
      log['train_very_hard_success_mean'] = (
          float(np.mean(recent_very_hard_success[-100:]))
          if recent_very_hard_success else float('nan'))
      log['train_very_hard_success_1000'] = (
          float(np.mean(recent_very_hard_success[-1000:]))
          if recent_very_hard_success else float('nan'))
    # Acme's CSV logger fixes its columns on the first write. Seed update
    # columns even while replay is warming up so later MPO/CRL metrics persist.
    for key_ in (
        'dual_alpha_mean', 'dual_alpha_stddev', 'dual_grad_norm',
        'dual_temperature', 'kl_mean_rel', 'kl_q_rel', 'kl_stddev_rel',
        'loss_alpha', 'loss_policy', 'loss_temperature',
        'penalty_kl_q_rel', 'pi_stddev_cond', 'pi_stddev_max',
        'pi_stddev_min', 'policy_grad_norm', 'q_max', 'q_mean', 'q_min',
        'total_loss', 'critic_grad_norm', 'critic_loss',
        'critic_prediction_mean', 'critic_reward_mean',
        'critic_reward_raw_mean', 'critic_reward_norm_std',
        'critic_target_mean', 'critic_td_error_abs'):
      log[f'mpo/{key_}'] = float('nan')
    for key_ in (
        'binary_accuracy', 'categorical_accuracy', 'crl_loss', 'logits_neg',
        'logits_pos', 'logsumexp', 'update_skipped_nonfinite'):
      log[f'crl/{key_}'] = float('nan')
    log.update({f'mpo/{key_}': value
                for key_, value in mpo_metrics_mean.items()})
    log.update({f'crl/{key_}': value
                for key_, value in crl_metrics_mean.items()})
    learner_logger.write(log)

    if mpo_config.eval_interval > 0 and iteration % mpo_config.eval_interval == 0:
      if bb_eval_unroll is not None:
        eval_steps = bb_eval_unroll(
            bb_eval_vec.reset_state(), state.target_policy_params)
        episode_metrics = ppo_learner._bb_ep_metrics_from_eval_steps(
            eval_steps,
            obs_dim=int(config.obs_dim),
            start_index=int(config.start_index),
            end_index=int(config.end_index),
            episode_length=bb_eval_vec.episode_length)
        episode_metrics = ppo_learner._smooth_bb_eval_metrics(
            episode_metrics, eval_success_observer, eval_distance_observer)
      else:
        episode_metrics = []
        for eval_index in range(int(mpo_config.eval_episodes)):
          env = eval_env_factory(
              seed + 900_000 + iteration * 100 + eval_index)
          timestep = env.reset()
          eval_success_observer.observe_first(env, timestep)
          eval_distance_observer.observe_first(env, timestep)
          episode_return = 0.0
          episode_length = 0
          while not timestep.last():
            action = np.asarray(target_mode(
                state.target_policy_params,
                jnp.asarray(timestep.observation)[None]))[0]
            action = np.clip(
                np.nan_to_num(action, nan=0.0),
                np.asarray(action_spec.minimum),
                np.asarray(action_spec.maximum)).astype(action_spec.dtype)
            timestep = env.step(action)
            eval_success_observer.observe(env, timestep, action)
            eval_distance_observer.observe(env, timestep, action)
            episode_return += float(timestep.reward or 0.0)
            episode_length += 1
          metrics = {
              'episode_return': episode_return,
              'episode_length': episode_length,
          }
          metrics.update(eval_success_observer.get_metrics())
          metrics.update(eval_distance_observer.get_metrics())
          episode_metrics.append(metrics)
      eval_logger.write(
          contrastive_utils.aggregate_eval_metrics(episode_metrics, iteration))

    if (checkpoint_dir is not None
        and mpo_config.checkpoint_interval > 0
        and (iteration % mpo_config.checkpoint_interval == 0
             or iteration == iterations - 1)):
      milestone = os.path.join(
          checkpoint_dir, f'ckpt_iter_{iteration:07d}.pkl')
      _save_checkpoint(milestone, state, iteration, global_step)
      _save_checkpoint(
          os.path.join(checkpoint_dir, 'latest.pkl'),
          state, iteration, global_step)
      _prune_checkpoints(
          checkpoint_dir, int(mpo_config.checkpoint_keep_last))

  return state
