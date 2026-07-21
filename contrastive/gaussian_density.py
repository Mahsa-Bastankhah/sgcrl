"""Gaussian density estimator  p_θ(g | s)  as a CRL drop-in for ppo_learner.

Overview
--------
Instead of the contrastive φ(s,a)·ψ(g) representation, we model the
discounted future-state density directly as a diagonal Gaussian:

    p_θ(g | s, a) = N(g;  μ_θ(s,a),  diag(σ_θ(s,a)²))

where μ_θ and σ_θ are the outputs of a single MLP that takes
concat([state, action]) as input — directly mirroring φ(s,a) in CRL.

Training
--------
We reuse the same  (s, s_f)  pairs that CRL draws from EpisodeReplay:

  1. Sample anchor s = obs[t] and future s_f = obs[t+d] with offset
     d ~ TruncatedGeometric(1-γ).   (handled by EpisodeReplay.sample)
  2. Convert s_f to the goal representation g = obs_to_goal(s_f).
     (already done by EpisodeReplay — batch['obs'] = [s; g])
  3. Loss:   L = -mean_i[ log p_θ(g_i | s_i, a_i) ]
             = mean_i[ 0.5·‖(g_i − μ_i)/σ_i‖² + Σ_k log σ_ik
                       + 0.5·goal_dim·log(2π) ]

   Includes the full −0.5·D·log(2π) normalizing constant (does not
   change gradients wrt μ/σ, but shifts the absolute log-p / PPO reward
   scale).  σ is clamped to [exp(-5), exp(2)] ≈ [0.007, 7.4].

Reward
------
At every PPO step, the per-env shaped reward is:

    r_t = log p_θ(g | s_t, a_t)
        = -0.5·‖(g − μ_θ(s_t,a_t))/σ_θ(s_t,a_t)‖² − Σ_k log σ_θk(s_t,a_t) + const

This is a dense scalar that peaks (≈ 0) when the policy takes an action
from which the goal is reached with high probability, and is deeply
negative far from the goal.  Unlike φ·ψ it has a natural scale and does
not require reward normalisation — though the existing ReturnNormalizer
is still applied for stability.

Usage in ppo_learner.py
-----------------------
Set config.ppo_repr_mode = 'gaussian'.  The run_ppo_training loop will:
  * Build a GaussianDensityNetworks instead of ContrastiveNetworks for
    the density side.
  * Replace make_crl_update_fn  →  make_gaussian_density_update_fn.
  * Replace the  φ·ψ  reward      →  make_gaussian_reward_fn.
  * The PPO policy and value networks are **unchanged**.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme.jax import networks as networks_lib
from contrastive.networks import _mlp_or_residual


# ---------------------------------------------------------------------------
# 1.  Network container
# ---------------------------------------------------------------------------

class GaussianDensityNetworks(NamedTuple):
    """All callable network objects needed by the Gaussian density mode.

    density_net:
        FeedForwardNetwork  init(key)  /  apply(params, state, action)
        state.shape  = (B, obs_dim)
        action.shape = (B, act_dim)
        returns (mu, log_std) each of shape (B, goal_dim)
    """
    density_net: networks_lib.FeedForwardNetwork


# ---------------------------------------------------------------------------
# 2.  Network factory
# ---------------------------------------------------------------------------

def make_gaussian_density_networks(
    obs_dim: int,
    act_dim: int,
    goal_dim: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    log_std_min: float = -5.0,
    log_std_max: float = 2.0,
) -> GaussianDensityNetworks:
    """Build the density network p_θ(g | s, a).

    Architecture
    ------------
    Input: concat([state, action])  shape (B, obs_dim + act_dim)
    Shared MLP trunk  (relu activations, VarianceScaling fan-avg init)
    Two independent linear heads:
        μ_head  : R^goal_dim   (unconstrained mean)
        σ_head  : R^goal_dim   log σ, clamped to [log_std_min, log_std_max]

    Taking (s, a) as input directly mirrors φ(s, a) in CRL so the two
    methods are directly comparable in terms of network expressivity.

    Args:
        obs_dim:  Dimension of the state part (obs[:obs_dim]).
        act_dim:  Dimension of the action.
        goal_dim: Dimension of the goal (obs[obs_dim:]).  Output dim of heads.
        hidden_layer_sizes: Widths of the shared trunk hidden layers.
        log_std_min / log_std_max: Clamp range for log σ.

    Returns:
        GaussianDensityNetworks with one field: density_net.
    """

    def _density_fn(
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """(state, action): (B, obs_dim), (B, act_dim) → (mu, log_std) (B, goal_dim)."""
        # ------------------------------------------------------------------
        # Step 1: concatenate state and action, then run shared trunk.
        # Uses ResidualMLP when len(hidden_layer_sizes) > 2 (same rule as
        # the CRL φ network in networks.py), so architecture is identical
        # to φ(s,a) for all hidden_layer_sizes configs.
        # ------------------------------------------------------------------
        sa = jnp.concatenate([state, action], axis=-1)   # (B, obs_dim + act_dim)
        w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
        trunk = _mlp_or_residual(
            sa,
            list(hidden_layer_sizes),
            hidden_layer_sizes=hidden_layer_sizes,
            name='density_trunk',
            activation=jax.nn.relu,
            activate_final=True,
            w_init=w_init,
        )

        # ------------------------------------------------------------------
        # Step 2: two independent linear heads
        # ------------------------------------------------------------------
        mu = hk.Linear(goal_dim, w_init=w_init, name='mu_head')(trunk)
        log_std_raw = hk.Linear(goal_dim, w_init=w_init, name='log_std_head')(trunk)

        # ------------------------------------------------------------------
        # Step 3: clamp log σ for numerical safety
        # ------------------------------------------------------------------
        log_std = jnp.clip(log_std_raw, log_std_min, log_std_max)

        return mu, log_std

    transformed = hk.without_apply_rng(hk.transform(_density_fn))

    # Dummy inputs for parameter init.
    dummy_state  = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_action = np.zeros((1, act_dim),  dtype=np.float32)

    density_net = networks_lib.FeedForwardNetwork(
        init=lambda key: transformed.init(key, dummy_state, dummy_action),
        apply=transformed.apply,
    )

    return GaussianDensityNetworks(density_net=density_net)


# ---------------------------------------------------------------------------
# 3.  Loss function  L = -E[log p_θ(g | s)]
# ---------------------------------------------------------------------------

def _diagonal_gaussian_log_prob(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    log_std: jnp.ndarray,
) -> jnp.ndarray:
    """Per-sample log N(x; mu, diag(exp(log_std))²).

    Shape: (B,) — the goal_dim terms are *summed* (not averaged) to get
    the correct log-probability of the full goal vector.

    log p(x) = -0.5·‖z‖² − Σ_k log σ_k − 0.5·D·log(2π)
    where z_k = (x_k − μ_k) / exp(log_std_k)
    """
    std = jnp.exp(log_std)                          # (B, goal_dim)
    z = (x - mu) / std                              # (B, goal_dim)
    d = log_std.shape[-1]
    log_p = -0.5 * jnp.sum(z ** 2, axis=-1) \
            - jnp.sum(log_std, axis=-1) \
            - 0.5 * d * jnp.log(2.0 * jnp.pi)       # (B,)
    return log_p


def make_gaussian_density_update_fn(
    density_networks: GaussianDensityNetworks,
    optimizer: optax.GradientTransformation,
    obs_dim: int,
):
    """Returns a jitted update function for the Gaussian density parameters.

    The returned function has the same signature as make_crl_update_fn's
    output so it can be swapped in without changing the training loop:

        update(density_params, opt_state, batch, key)
        → (new_density_params, new_opt_state, metrics)

    Training objective
    ------------------
    Given a replay batch with:
        batch['obs']    = [s_t ; g]    (shape (B, obs_dim+goal_dim))
        batch['action'] = a_t          (shape (B, act_dim))

        state        = obs[:, :obs_dim]
        future_goal  = obs[:, obs_dim:]
        action       = batch['action']

    We maximise  Σ_i log p_θ(future_goal_i | state_i, action_i), i.e.:

        L = -mean_i[ log N(future_goal_i; μ_θ(state_i, action_i),
                                          σ_θ(state_i, action_i)) ]

    Args:
        density_networks: GaussianDensityNetworks (only density_net used).
        optimizer: optax gradient transform (e.g. optax.adam(lr)).
        obs_dim: state slice length; obs[:obs_dim] is passed to the network.

    Returns:
        A jitted callable  update(params, opt_state, batch, key)
        →  (new_params, new_opt_state, metrics_dict)
    """

    def _loss(params, batch, key):
        del key  # not needed for this loss (no stochasticity in forward pass)

        obs = batch['obs']                          # (B, obs_dim + goal_dim)
        action = batch['action']                    # (B, act_dim)
        state = obs[:, :obs_dim]                    # (B, obs_dim)
        future_goal = obs[:, obs_dim:]              # (B, goal_dim)

        # ---- forward pass -------------------------------------------------
        # density_net.apply(params, state, action) → (mu, log_std) (B, goal_dim)
        mu, log_std = density_networks.density_net.apply(params, state, action)

        # ---- negative log-likelihood -------------------------------------
        log_p = _diagonal_gaussian_log_prob(future_goal, mu, log_std)  # (B,)
        loss = -jnp.mean(log_p)

        # ---- diagnostics --------------------------------------------------
        std = jnp.exp(log_std)
        metrics = {
            'gaussian_density_loss': loss,
            'log_p_mean': jnp.mean(log_p),
            'mu_norm': jnp.mean(jnp.linalg.norm(mu, axis=-1)),
            'std_mean': jnp.mean(std),
            'std_min': jnp.min(std),
            'std_max': jnp.max(std),
        }
        return loss, metrics

    grad_fn = jax.value_and_grad(_loss, has_aux=True)

    def update(params, opt_state, batch, key):
        (_, metrics), grads = grad_fn(params, batch, key)

        # Skip update if any gradient or loss is non-finite.
        grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))))
        loss_finite = jnp.isfinite(metrics['gaussian_density_loss'])
        do_update = jnp.logical_and(grads_finite, loss_finite)

        def _apply(_):
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state

        def _skip(_):
            return params, opt_state

        new_params, new_opt_state = jax.lax.cond(
            do_update, _apply, _skip, operand=None)
        metrics = dict(metrics)
        metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
        return new_params, new_opt_state, metrics

    return jax.jit(update)


# ---------------------------------------------------------------------------
# 4.  Reward function   r_t = log p_θ(g | s_t)
# ---------------------------------------------------------------------------

def make_gaussian_reward_fn(
    density_networks: GaussianDensityNetworks,
    obs_dim: int,
):
    """Returns a jitted per-step reward function for the Gaussian density mode.

    Reward computation
    ------------------
    Given the current policy observation  obs  (shape (E, obs_dim+goal_dim))
    and the action  action  just sampled from the policy (shape (E, act_dim)):

        state  = obs[:, :obs_dim]        — current state  s_t
        goal   = obs[:, obs_dim:]        — fixed episode goal  g
        action = action                  — policy action  a_t

    Reward:
        r_t = log p_θ(g | s_t, a_t)
            = -0.5·‖(g − μ_θ(s_t,a_t)) / σ_θ(s_t,a_t)‖²
              − Σ_k log σ_θk(s_t,a_t) − 0.5·D·log(2π)

    Returns:
        A jitted callable:
            reward_fn(density_params, obs, action) → rewards  (shape (E,))
    """

    @jax.jit
    def reward_fn(
        density_params,
        obs: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """obs: (E, obs_dim+goal_dim), action: (E, act_dim)  →  rewards: (E,)."""
        state = obs[:, :obs_dim]          # (E, obs_dim)
        goal  = obs[:, obs_dim:]          # (E, goal_dim)

        mu, log_std = density_networks.density_net.apply(
            density_params, state, action)
        rewards = _diagonal_gaussian_log_prob(goal, mu, log_std)  # (E,)
        return rewards

    return reward_fn
