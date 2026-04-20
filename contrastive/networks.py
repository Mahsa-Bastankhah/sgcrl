"""Contrastive RL networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from itertools import product


# modified Tanh mean to be mapped to tanh(mean) to keep within [-1, 1]
from distributional import NormalTanhDistribution

@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None
  # κ network: maps (obs, action) → R^repr_dim.
  # κ(s,a) approximates E[Σ γ^t φ(s_t,a_t)] along on-policy trajectories.
  kappa_network: Optional[networks_lib.FeedForwardNetwork] = None
  kappa_network_2: Optional[networks_lib.FeedForwardNetwork] = None


def apply_policy_and_sample(
    networks,
    eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample


def make_networks(
    spec,
    obs_dim,
    repr_dim = 64,
    repr_norm = False,
    repr_norm_temp = False,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-6,
    twin_q = False,
    use_image_obs = False):
  """Creates networks used by the agent."""

  num_dimensions = np.prod(spec.actions.shape, dtype=int)
  TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

  def _unflatten_obs(obs):
    state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
    goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    return state, goal

  def _repr_fn(obs, action, hidden=None):
    # The optional input hidden is the image representations. We include this
    # as an input for the second Q value when twin_q = True, so that the two Q
    # values use the same underlying image representation.
    if hidden is None:
      if use_image_obs:
        state, goal = _unflatten_obs(obs)
        img_encoder = TORSO()
        state = img_encoder(state)
        goal = img_encoder(goal)
      else:
        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]
    else:
      state, goal = hidden

    sa_encoder = hk.nets.MLP(
        list(hidden_layer_sizes) + [repr_dim],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        activation=jax.nn.relu,
        name='sa_encoder')
    sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))

    g_encoder = hk.nets.MLP(
        list(hidden_layer_sizes) + [repr_dim],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        activation=jax.nn.relu,
        name='g_encoder')
    g_repr = g_encoder(goal)

    if repr_norm:
      sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
      g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)

      if repr_norm_temp:
        log_scale = hk.get_parameter('repr_log_scale', [], dtype=sa_repr.dtype,
                                     init=jnp.zeros)
        sa_repr = sa_repr / jnp.exp(log_scale)
    return sa_repr, g_repr, (state, goal)

    
  def _combine_repr(sa_repr, g_repr):
    return jax.numpy.einsum('ik,jk->ij', sa_repr, g_repr)

  def _critic_fn(obs, action):
    sa_repr, g_repr, hidden = _repr_fn(obs, action)
    critic_val = _combine_repr(sa_repr, g_repr)
    if twin_q:
      sa_repr2, g_repr2, _ = _repr_fn(obs, action, hidden=hidden)
      product2 = _combine_repr(sa_repr2, g_repr2)
      # outer.shape = [batch_size, batch_size, 2]
      critic_val = jnp.stack([product, product2], axis=-1)
      sa_repr = sa_repr2
      g_repr = g_repr2
    return critic_val, sa_repr, g_repr

  def _actor_fn(obs):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      obs = jnp.concatenate([state, goal], axis=-1)
      obs = TORSO()(obs)
    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            activate_final=True),
        NormalTanhDistribution(num_dimensions, min_scale=actor_min_std),
    ])
    return network(obs)

  def _make_kappa_fn(net_name):
    """Factory for a κ network with a given Haiku variable scope name.

    Each call produces an independent set of parameters, enabling twin-κ.
    Uses only the state slice of obs (goal-agnostic, matching φ(s,a)).
    """
    def _kappa_fn(obs, action):
      state = obs[:, :obs_dim]
      net = hk.nets.MLP(
          list(hidden_layer_sizes) + [repr_dim],
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          activation=jax.nn.relu,
          name=net_name)
      return net(jnp.concatenate([state, action], axis=-1))
    return _kappa_fn

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))
  kappa = hk.without_apply_rng(hk.transform(_make_kappa_fn('kappa_net')))
  kappa2 = hk.without_apply_rng(hk.transform(_make_kappa_fn('kappa_net_2')))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  return ContrastiveNetworks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, dummy_obs), policy.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      repr_fn=repr_fn.apply,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      kappa_network=networks_lib.FeedForwardNetwork(
          lambda key: kappa.init(key, dummy_obs, dummy_action), kappa.apply),
      kappa_network_2=networks_lib.FeedForwardNetwork(
          lambda key: kappa2.init(key, dummy_obs, dummy_action), kappa2.apply),
      )
