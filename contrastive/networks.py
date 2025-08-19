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

import haiku as hk
import jax.numpy as jnp
from typing import Sequence
import typing

import haiku as hk
import jax.numpy as jnp
from typing import Sequence, Callable
# modified Tanh mean to be mapped to tanh(mean) to keep within [-1, 1]
from distributional import NormalTanhDistribution
from itertools import product
import itertools


class ResidualMLP(hk.Module):
    """
    Simple MLP with optional skip connections and LayerNorm.  Adds residual
    connection only when input and output dims match for a block.
    """
    def __init__(
        self,
        widths: Sequence[int],
        skip_every: int = 0,
        activation: Callable = jax.nn.relu,
        use_layer_norm: bool = False,
        name: str = None,
        activate_final: bool = False,
        # w_init: Optional[Callable] = None,
    ):
        super().__init__(name=name)
        self._widths = list(widths)
        self._skip_every = skip_every
        self._activation = activation
        self._use_ln = use_layer_norm
        self._activate_final = activate_final
        self._w_init = hk.initializers.VarianceScaling(1.0, "fan_in", "uniform")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        skip = None
        for i, w in enumerate(self._widths[:-1]):
            # Linear layer
            h = hk.Linear(w, name=f"linear_{i}", w_init=self._w_init)(h)
            # print(f"linear_{i}")
            # Optional LayerNorm
            if self._use_ln:
                # print("Applying LayerNorm at layer", i, flush=True)
                h = hk.LayerNorm(axis=-1,
                                 create_scale=True,
                                 create_offset=True,
                                 name=f"ln_{i}")(h)
            # Activation
            h = self._activation(h)

            # Residual: add skip when dims match and it's a skip point
            if self._skip_every and (i + 1) % self._skip_every == 0:
                if skip is not None and skip.shape[-1] == h.shape[-1]:
                    # print("Adding skip connection at layer", i, flush=True)
                    h = skip + h
                # update skip to current output
                skip = h
        # Final layer
        # ---- final projection (no residual) -----------------------------
        final_w = self._widths[-1]
        h = hk.Linear(final_w, w_init=self._w_init,
                      name=f"linear_{len(self._widths)-1}")(h)
        # print(f"linear_{len(self._widths)-1}")

        if self._activate_final:
            h = self._activation(h)

        return h



@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


# def apply_policy_and_sample(
#     networks,
#     eval_mode = False):
#   """Returns a function that computes actions."""
#   sample_fn = networks.sample if not eval_mode else networks.sample_eval
#   if not sample_fn:
#     raise ValueError('sample function is not provided')

#   def apply_and_sample(params, key, obs):
#     return sample_fn(networks.policy_network.apply(params, obs), key)
#   return apply_and_sample



def apply_policy_and_sample(
    networks,
    Q_max = False,
    eval_mode = False):
  
  if Q_max is False:
      print("Using actor", flush = True)
      """Returns a function that computes actions."""
      sample_fn = networks.sample if not eval_mode else networks.sample_eval
      if not sample_fn:
        raise ValueError('sample function is not provided')

      def apply_and_sample(params, key, obs):
        # print("param tree structure: {}", jax.tree_util.tree_structure(params), flush=True)
        # print("top-level keys: {}", list(params.keys()), flush= True)
        # policy_params = params if isinstance(params, jax.Array) else params['policy_network']
        # q_params = params if isinstance(params, jax.Array) else params['critic']
        policy_params, q_params = params 
        
        return sample_fn(networks.policy_network.apply(policy_params, obs), key)
      return apply_and_sample
  
  else:
      def select_action(params, key, obs):
          print("Using Q_max to select actions", flush = True)
          # print("param tree structure: {}", jax.tree_util.tree_structure(params), flush=True)
          # print("top-level keys: {}", list(params.keys()), flush= True)
          # policy_params = params if isinstance(params, jax.Array) else params['policy_network']
          # q_params = params if isinstance(params, jax.Array) else params['critic']
          policy_params, q_params = params      # unpack
          return maximize_q_action(networks.q_network, q_params, obs)
      return select_action


    



def maximize_q_action(q_network, q_params, obs, grid_size=20, epsilon = 0.01):
    # 1) Build candidate actions
    # 1) Infer dimensions
    obs = obs.reshape(-1)  # flatten in case it's shape (1, obs_dim)
    obs_dim = obs.shape[0] // 2

    input_dim = q_params['sa_encoder/~/linear_0']['w'].shape[0]
    action_dim = input_dim - obs_dim
    #debug.print("obs_dim: {}, action_dim: {}", obs_dim, action_dim)

    grid = jnp.linspace(-1.0, 1.0, num=grid_size)        # 10 points per axis
    action_grid = jnp.array(list(itertools.product(grid, repeat=action_dim)))
    #    → (1000, 3)

    # 2) Tile obs into shape (1000, obs_dim*2)
    #    If obs is (6,) or (1,6), this yields (1000,6).
    repeated_obs = jnp.tile(obs.reshape(-1), (action_grid.shape[0], 1))

    # —— Debug prints ——
    # debug.print("obs shape:         ", obs.shape)
    # debug.print("repeated_obs shape:", repeated_obs.shape)
    # debug.print("action_grid shape: ", action_grid.shape)

    # 3) Call the critic: returns (critic_val, sa_repr, g_repr)
    critic_val, sa_repr, g_repr = q_network.apply(q_params,
                                                  repeated_obs,
                                                  action_grid)
    # Now sa_repr, g_repr are both (1000, repr_dim)

    # 4) Compute per-action Q: elementwise dot → (1000,)
    q_values = jnp.sum(sa_repr * g_repr, axis=-1)
    #debug.print("q_values shape:    ", q_values.shape)

    # 5) Pick the best index (a JAX scalar) and slice
    best_idx    = jnp.argmax(q_values)          # shape=(), dtype=int32
    best_action = action_grid[best_idx]         # shape=(3,)
    #debug.print("best_idx:", best_idx, "→ best_action shape:", best_action.shape)
    # Add small Gaussian noise to the best action
    # key = jax.random.PRNGKey(0)  # Initialize once
    # key, noise_key, eps_key = jax.random.split(key, 3)

    # rand_idx = jax.random.randint(jax.random.PRNGKey(0), shape=(), minval=0, maxval=action_grid.shape[0])
    # random_action = action_grid[rand_idx]

    # # 8) Epsilon-greedy selection
    # choose_random = jax.random.uniform(eps_key) < epsilon
    # selected_action = jax.lax.cond(choose_random, lambda: random_action, lambda: best_action)


    return jnp.expand_dims(best_action, axis=0)


def make_networks(
    spec,
    obs_dim,
    repr_dim = 64,
    repr_norm = False,
    repr_norm_temp = False,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-6,
    twin_q = False,
    use_image_obs = False,
    config = None):
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

    if config.use_residual_mlp is False:
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


    if config.use_residual_mlp:
      sa_encoder = ResidualMLP(
        list(hidden_layer_sizes) + [repr_dim],
        skip_every=4,              # <- every 2 layers get a skip; tune as you like
        activation=jax.nn.swish,
        use_layer_norm=True,
        #w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        name='sa_encoder')
      sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))

      g_encoder = ResidualMLP(
          list(hidden_layer_sizes) + [repr_dim],
          skip_every=4,
          activation=jax.nn.swish,
          use_layer_norm=True,
          #w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
          name='g_encoder')
      g_repr = g_encoder(goal)


    # # ❶ Add these two helpers – place them near _repr_fn for clarity
    # def _in_region(x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    #     """Returns a (batch,) boolean mask: True if each state lies inside [lower, upper] box."""
    #     # jax.debug.print("x {x}", x=x)
    #     mask = jnp.all((x >= lower) & (x <= upper), axis=-1)
    #     # jax.debug.print(mask)
    #     return mask
    

    def _in_region(x: jnp.ndarray,
               lower: jnp.ndarray,
               upper: jnp.ndarray) -> jnp.ndarray:
        """
        Returns a (batch,) mask that is True when each state in `x` lies in
        *at least one* axis-aligned box defined by `lower, upper`.

        Parameters
        ----------
        x      : (B, D)  batch of states
        lower  : (D,) or (R, D)  lower corner(s) of region box(es)
        upper  : (D,) or (R, D)  upper corner(s) of region box(es)
        """
        # Ensure JAX arrays with matching dtype
        lower = jnp.asarray(lower, dtype=x.dtype)
        upper = jnp.asarray(upper, dtype=x.dtype)

        if lower.ndim == 1:
            # ---- single region --------------------------------------------------
            return jnp.all((x >= lower) & (x <= upper), axis=-1)   # (B,)
        else:
            # ---- multiple regions ----------------------------------------------
            # lower/upper shape  (R, D)
            # Build a (R, B, D) boolean tensor: True if inside that coord
            in_each_region = (x[None, :, :] >= lower[:, None, :]) & \
                            (x[None, :, :] <= upper[:, None, :])      # (R,B,D)

            # Collapse the D dimension ⇒ (R, B)  then OR across regions ⇒ (B,)
            return jnp.any(jnp.all(in_each_region, axis=-1), axis=0)

    



    def _replace_with_goal(mask: jnp.ndarray,
                          g_repr: jnp.ndarray,
                          goal_repr: jnp.ndarray,
                          stop_grad_to_goal: bool = False) -> jnp.ndarray:
        """Batch-select between g_repr and goal_repr using mask."""
        if stop_grad_to_goal:
            goal_repr = jax.lax.stop_gradient(goal_repr)

        out = jnp.where(mask[:, None], goal_repr, g_repr)

    
        # jax.debug.print(
        #     "⏩ replace_with_goal: masked {m}/{b}  stop_grad={sg}, goal {goal}",
        #     m=mask.sum(), b=mask.size, sg=stop_grad_to_goal, goal=goal)
        #small peek at first 3 dims of the first vector

      # -----------------------------------------------------------------
        return out


    # --- NEW: selective goal substitution ----------------------------
    if (config.fixed_goal is not None) and (config.region_bounds is not None):
        lower, upper = config.region_bounds               # each shape (state_dim,)
        # NEW: ensure they are JAX arrays (works for list / tuple / array)
        lower = jnp.asarray(lower, dtype=state.dtype)
        upper = jnp.asarray(upper, dtype=state.dtype)
        # jax.debug.print("region_bounds {lower} {upper}", lower=lower, upper=upper)
        # jax.debug.print("goal {goal}", goal=goal)
        # jax.debug.print("fixed goal {fixed_goal}", fixed_goal =config.fixed_goal)
        # jax.debug.print("stop grad fixed {stop_grad}", stop_grad = config.stop_grad_fixed)
        mask = _in_region(goal, lower, upper)     # (batch,)

        # ▸ 1.  Convert whatever the user put in `config.fixed_goal`
        #       (tuple, list, np.array, etc.) into a JAX array **once**:
        goal_fixed = jnp.asarray(config.fixed_goal, dtype=goal.dtype)   # shape (goal_dim,)

        # ▸ 2.  Tile it to match the batch:
        goal_fixed = jnp.broadcast_to(goal_fixed, goal.shape)           # shape (B, goal_dim)

        # ▸ 3.  Pass it through the same encoder:
        if config.negative_goal_repr:
          goal_fixed_repr = -1 *  g_encoder(goal_fixed) 
        else:
          goal_fixed_repr = g_encoder(goal_fixed)            

        g_repr = _replace_with_goal(mask, g_repr,
                                    goal_fixed_repr,
                                    config.stop_grad_fixed)
        




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

    if config.use_residual_mlp:

      trunk = ResidualMLP(                            # your critic style
          list(hidden_layer_sizes),
          skip_every=4,
          activation=jax.nn.swish,
          use_layer_norm=True,
          activate_final=True,
          #w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
          name="policy_trunk",
      )
      seq_layers = [trunk]
      # ― final distribution head
      seq_layers.append(NormalTanhDistribution(num_dimensions, min_scale=actor_min_std))

      # ── 3. assemble and call ──────────────────────────────────────────────────
      network = hk.Sequential(seq_layers)

    else:
      network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            activate_final=True),
        NormalTanhDistribution(num_dimensions, min_scale=actor_min_std),
      ])
#  if config.use_residual_mlp else hk.nets.MLP(  # fallback to plain MLP
#         list(hidden_layer_sizes),
#         w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "uniform"),
#         activation=jax.nn.relu,
#         activate_final=True,
#         name="policy_trunk",
#     )
   
    return network(obs)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))

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
      )
