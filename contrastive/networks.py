"""Contrastive RL networks definition."""
import dataclasses
from typing import Callable, Optional, Sequence, Tuple

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


def _use_residual_mlp(hidden_layer_sizes: Sequence[int]) -> bool:
  """Use ResidualMLP instead of hk.nets.MLP when there are more than two hidden layers."""
  return len(tuple(hidden_layer_sizes)) > 2


# When `_use_residual_mlp` is true, all such stacks share this layout (repr-style).
# skip_every=2 means residual blocks span 2 hidden layers each.  With 6 hidden
# layers this gives 2 real residual additions (at layers 3 and 5); with 4 layers
# it gives 1.  Keeping it at 2 (not 4) ensures skips are actually used for the
# default 6-layer depth.
_RESIDUAL_SKIP_EVERY = 2
_RESIDUAL_ACTIVATION = jax.nn.swish
_RESIDUAL_USE_LAYER_NORM = True


class ResidualMLP(hk.Module):
  """MLP with optional residual skips and LayerNorm.

  Residual adds ``skip + h`` only when widths match and ``skip_every`` divides
  the layer index. Final layer has no residual from prior blocks.
  """

  def __init__(
      self,
      widths: Sequence[int],
      skip_every: int = 0,
      activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
      use_layer_norm: bool = False,
      name: Optional[str] = None,
      activate_final: bool = False,
      w_init: Optional[hk.initializers.Initializer] = None,
  ):
    super().__init__(name=name)
    self._widths = list(widths)
    self._skip_every = skip_every
    self._activation = activation
    self._use_ln = use_layer_norm
    self._activate_final = activate_final
    if w_init is None:
      self._w_init = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")
    else:
      self._w_init = w_init

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    h = x
    skip = None
    for i, w in enumerate(self._widths[:-1]):
      h = hk.Linear(w, name=f"linear_{i}", w_init=self._w_init)(h)
      if self._use_ln:
        h = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name=f"ln_{i}",
        )(h)
      h = self._activation(h)
      if self._skip_every and (i + 1) % self._skip_every == 0:
        if skip is not None and skip.shape[-1] == h.shape[-1]:
          h = skip + h
        skip = h
    final_w = self._widths[-1]
    h = hk.Linear(
        final_w,
        w_init=self._w_init,
        name=f"linear_{len(self._widths) - 1}",
    )(h)
    if self._activate_final:
      h = self._activation(h)
    return h


def _mlp_or_residual(
    x: jnp.ndarray,
    widths: Sequence[int],
    *,
    hidden_layer_sizes: Sequence[int],
    name: str,
    activation: Callable[[jnp.ndarray], jnp.ndarray],
    activate_final: bool,
    w_init: hk.initializers.Initializer,
) -> jnp.ndarray:
  """hk.nets.MLP for shallow stacks; ResidualMLP when len(hidden_layer_sizes) > 2.

  Deep (residual) stacks use a fixed layout: LayerNorm, Swish, skip_every=4, and
  the caller's ``w_init`` (e.g. fan_avg uniform for encoders / κ / q-goal trunks).
  """
  if _use_residual_mlp(hidden_layer_sizes):
    return ResidualMLP(
        widths,
        skip_every=_RESIDUAL_SKIP_EVERY,
        activation=_RESIDUAL_ACTIVATION,
        use_layer_norm=_RESIDUAL_USE_LAYER_NORM,
        name=name,
        activate_final=activate_final,
        w_init=w_init,
    )(x)
  return hk.nets.MLP(
      list(widths),
      w_init=w_init,
      activation=activation,
      activate_final=activate_final,
      name=name,
  )(x)


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
  # V network used by the PPO-on-φ·ψ actor (reward_shaping_mode='ppo').
  # Maps full obs=[state;goal] → scalar V(s,g).
  value_network: Optional[networks_lib.FeedForwardNetwork] = None
  # Scalar Q networks when reward_shaping_mode='q'.  Input is obs=[s;g] but the
  # MLP only uses the state slice (see _make_q_goal_fn).  Trained TD3-style on
  # r = sg(φ(s,a)·ψ(g)) with clipped-double-Q target.
  q_goal_network: Optional[networks_lib.FeedForwardNetwork] = None
  q_goal_network_2: Optional[networks_lib.FeedForwardNetwork] = None


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

    sa_repr = _mlp_or_residual(
        jnp.concatenate([state, action], axis=-1),
        list(hidden_layer_sizes) + [repr_dim],
        hidden_layer_sizes=hidden_layer_sizes,
        name='sa_encoder',
        activation=jax.nn.relu,
        activate_final=False,
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    )

    g_repr = _mlp_or_residual(
        goal,
        list(hidden_layer_sizes) + [repr_dim],
        hidden_layer_sizes=hidden_layer_sizes,
        name='g_encoder',
        activation=jax.nn.relu,
        activate_final=False,
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
    )

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
      critic_val = jnp.stack([critic_val, product2], axis=-1)
      sa_repr = sa_repr2
      g_repr = g_repr2
    return critic_val, sa_repr, g_repr

  def _actor_fn(obs):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      obs = jnp.concatenate([state, goal], axis=-1)
      obs = TORSO()(obs)
    h = _mlp_or_residual(
        obs,
        list(hidden_layer_sizes),
        hidden_layer_sizes=hidden_layer_sizes,
        name='policy_mlp',
        activation=jax.nn.relu,
        activate_final=True,
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
    )
    return NormalTanhDistribution(num_dimensions, min_scale=actor_min_std)(h)

  def _value_fn(obs):
    """V(s, g): scalar value network for PPO on r = φ·ψ.

    Takes the full obs = [state; goal] (same format the policy consumes).
    CleanRL uses tanh activations + orthogonal init for PPO; we follow suit
    since it's notably more stable than relu+fan_avg for value learning.
    """
    if _use_residual_mlp(hidden_layer_sizes):
      h = _mlp_or_residual(
          obs,
          list(hidden_layer_sizes),
          hidden_layer_sizes=hidden_layer_sizes,
          name='value_mlp',
          activation=jnp.tanh,
          activate_final=True,
          w_init=hk.initializers.Orthogonal(scale=np.sqrt(2.0)),
      )
      out = hk.Linear(1, w_init=hk.initializers.Orthogonal(scale=1.0))(h)
      return jnp.squeeze(out, axis=-1)
    net = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),
            w_init=hk.initializers.Orthogonal(scale=np.sqrt(2.0)),
            activation=jnp.tanh,
            activate_final=True),
        hk.Linear(1, w_init=hk.initializers.Orthogonal(scale=1.0)),
    ])
    return jnp.squeeze(net(obs), axis=-1)

  def _make_q_goal_fn(net_name):
    """Factory for a state-action scalar Q network.

    Takes obs=[state;goal] and action, but only uses the state slice
    `obs[:, :obs_dim]` (goal slice is ignored). Used when reward_shaping_mode='q' and
    trained on
    r = sg(φ(s,a)·ψ(g)) with TD(0) + clipped-double-Q.  Each call creates
    an independent Haiku scope so Q1 and Q2 have independent parameters.
    """
    def _q_goal_fn(obs, action):
      state = obs[:, :obs_dim]
      trunk_in = jnp.concatenate([state, action], axis=-1)
      trunk = _mlp_or_residual(
          trunk_in,
          list(hidden_layer_sizes),
          hidden_layer_sizes=hidden_layer_sizes,
          name=net_name,
          activation=jax.nn.relu,
          activate_final=True,
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
      )
      head = hk.Linear(
          1, w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
          name=net_name + '_head')
      return jnp.squeeze(head(trunk), axis=-1)
    return _q_goal_fn

  def _make_kappa_fn(net_name):
    """Factory for a κ network with a given Haiku variable scope name.

    Each call produces an independent set of parameters, enabling twin-κ.
    Uses only the state slice of obs (goal-agnostic, matching φ(s,a)).
    """
    def _kappa_fn(obs, action):
      state = obs[:, :obs_dim]
      return _mlp_or_residual(
          jnp.concatenate([state, action], axis=-1),
          list(hidden_layer_sizes) + [repr_dim],
          hidden_layer_sizes=hidden_layer_sizes,
          name=net_name,
          activation=jax.nn.relu,
          activate_final=False,
          w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
      )
    return _kappa_fn

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))
  kappa = hk.without_apply_rng(hk.transform(_make_kappa_fn('kappa_net')))
  kappa2 = hk.without_apply_rng(hk.transform(_make_kappa_fn('kappa_net_2')))
  value = hk.without_apply_rng(hk.transform(_value_fn))
  q_goal = hk.without_apply_rng(hk.transform(_make_q_goal_fn('q_goal_net')))
  q_goal2 = hk.without_apply_rng(hk.transform(_make_q_goal_fn('q_goal_net_2')))

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
      value_network=networks_lib.FeedForwardNetwork(
          lambda key: value.init(key, dummy_obs), value.apply),
      q_goal_network=networks_lib.FeedForwardNetwork(
          lambda key: q_goal.init(key, dummy_obs, dummy_action), q_goal.apply),
      q_goal_network_2=networks_lib.FeedForwardNetwork(
          lambda key: q_goal2.init(key, dummy_obs, dummy_action), q_goal2.apply),
      )
