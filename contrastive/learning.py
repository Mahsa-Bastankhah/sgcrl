"""Contrastive RL learner implementation."""
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
import jax
import jax.numpy as jnp
import optax
import reverb
from jax.experimental.host_callback import id_print
from jax import debug
from jax.scipy.special import logsumexp
import numpy as np
from jax import random
import os
from default import make_default_logger
from pathlib import Path

class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  policy_params_prev: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None
  # κ networks (use_kappa=True only; _2 fields also used when twin_kappa=True)
  kappa_params: Optional[networks_lib.Params] = None
  target_kappa_params: Optional[networks_lib.Params] = None
  kappa_optimizer_state: Optional[optax.OptState] = None
  kappa_params_2: Optional[networks_lib.Params] = None
  target_kappa_params_2: Optional[networks_lib.Params] = None
  kappa_optimizer_state_2: Optional[optax.OptState] = None
  # Goal-conditioned scalar Q networks (use_q_repr=True only; _2 fields
  # also used when twin_q_repr=True).  Trained TD3-style on r = sg(φ·ψ).
  q_repr_params: Optional[networks_lib.Params] = None
  target_q_repr_params: Optional[networks_lib.Params] = None
  q_repr_optimizer_state: Optional[optax.OptState] = None
  q_repr_params_2: Optional[networks_lib.Params] = None
  target_q_repr_params_2: Optional[networks_lib.Params] = None
  q_repr_optimizer_state_2: Optional[optax.OptState] = None


def _scale_kappa_rows_to_phi_norm(kappa, phi_sa):
  """Rescale each κ row so ‖κ‖_2 = ‖φ‖_2 (same batch as φ = sa_repr).

  Used in the κ actor loss when config.kappa_actor_match_phi_norm is True.
  """
  eps = jnp.asarray(1e-8, dtype=kappa.dtype)
  n_phi = jnp.linalg.norm(phi_sa, axis=-1, keepdims=True) + eps
  n_k = jnp.linalg.norm(kappa, axis=-1, keepdims=True) + eps
  return kappa * (n_phi / n_k)


class ContrastiveLearner(acme.Learner):
  """Contrastive RL learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      policy_optimizer,
      q_optimizer,
      iterator,
      counter,
      logger,
      obs_to_goal,
      config):
    """Initialize the Contrastive RL learner.

    Args:
      networks: Contrastive RL networks.
      rng: a key for random number generation.
      policy_optimizer: the policy optimizer.
      q_optimizer: the Q-function optimizer.
      iterator: an iterator over training data.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      obs_to_goal: a function for extracting the goal coordinates.
      config: the experiment config file.
    """
    if config.add_mc_to_td:
      assert config.use_td
    adaptive_entropy_coefficient = config.entropy_coefficient is None
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim
    self._use_td = config.use_td
    
    if adaptive_entropy_coefficient:
      # alpha is the temperature parameter that determines the relative
      # importance of the entropy term versus the reward.
      log_alpha = jnp.asarray(0., dtype=jnp.float32)
      alpha_optimizer = optax.adam(learning_rate=3e-4)
      alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    else:
      if config.target_entropy:
        raise ValueError('target_entropy should not be set when '
                         'entropy_coefficient is provided')

    if config.use_kappa:
      # κ uses a separate, lower LR than the critic — see config.py for why.
      kappa_tx = [optax.adam(learning_rate=config.learning_rate_kappa)]
      if config.kappa_max_grad_norm > 0:
        kappa_tx.insert(
            0, optax.clip_by_global_norm(config.kappa_max_grad_norm))
      kappa_optimizer = optax.chain(*kappa_tx)

    if config.use_q_repr:
      # Scalar Q(s,a,g) trained TD3-style on r = sg(φ·ψ).  Optional
      # gradient-norm clip — with repr_norm=True the reward is bounded,
      # but the bootstrap target can still spike before Q has settled.
      q_repr_tx = [optax.adam(learning_rate=config.learning_rate_q)]
      if config.q_repr_max_grad_norm > 0:
        q_repr_tx.insert(0, optax.clip_by_global_norm(
            config.q_repr_max_grad_norm))
      q_repr_optimizer = optax.chain(*q_repr_tx)

    # NB: reward_shaping_mode='kappa' reuses the SAME adaptive-α pipeline as the stock
    # CRL actor (state.alpha_params, alpha_loss, alpha_optimizer) with
    # target = config.target_entropy.  Keeping a separate `log_alpha_repr`
    # was redundant — there's only ever one policy, so one temperature
    # suffices — and led to two α's training in parallel with one of them
    # unused.  To "encourage more entropy" just raise config.target_entropy.

    def alpha_loss(log_alpha,
                   policy_params,
                   transitions,
                   key):
      """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      alpha = jnp.exp(log_alpha)
      alpha_loss = alpha * jax.lax.stop_gradient(
          -log_prob - config.target_entropy)
      return jnp.mean(alpha_loss)


    def critic_loss(q_params,
                    policy_params,
                    target_q_params,
                    transitions,
                    key):
      batch_size = transitions.observation.shape[0]
      # Note: We might be able to speed up the computation for some of the
      # baselines to making a single network that returns all the values. This
      # avoids computing some of the underlying representations multiple times.
      if config.use_td:
        # For TD learning, the diagonal elements are the immediate next state.
        s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
        next_s, _ = jnp.split(transitions.next_observation, [config.obs_dim],
                              axis=1)
        if config.add_mc_to_td:
          next_fraction = (1 - config.discount) / ((1 - config.discount) + 1)
          num_next = int(batch_size * next_fraction)
          new_g = jnp.concatenate([
              obs_to_goal(next_s[:num_next]),
              g[num_next:],
          ], axis=0)
        else:
          new_g = obs_to_goal(next_s)
        obs = jnp.concatenate([s, new_g], axis=1)
        transitions = transitions._replace(observation=obs)
      I = jnp.eye(batch_size)  # pylint: disable=invalid-name

      logits, phi_repr, psi_repr = networks.q_network.apply(
          q_params, transitions.observation, transitions.action)
      repr_phi_norm_mean = jnp.mean(jnp.linalg.norm(phi_repr, axis=-1))
      repr_psi_norm_mean = jnp.mean(jnp.linalg.norm(psi_repr, axis=-1))

      if config.use_td:
        # Make sure to use the twin Q trick.
        assert len(logits.shape) == 3

        # We evaluate the next-state Q function using random goals
        s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
        del s
        next_s = transitions.next_observation[:, :config.obs_dim]
        goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)
        g = g[goal_indices]
        transitions = transitions._replace(
            next_observation=jnp.concatenate([next_s, g], axis=1))
        next_dist_params = networks.policy_network.apply(
            policy_params, transitions.next_observation)
        next_action = networks.sample(next_dist_params, key)
        
        next_q, _, _ = networks.q_network.apply(target_q_params,
                                          transitions.next_observation,
                                          next_action)  # This outputs logits.
        next_q = jax.nn.sigmoid(next_q)
        next_v = jnp.min(next_q, axis=-1)
        next_v = jax.lax.stop_gradient(next_v)
        next_v = jnp.diag(next_v)
        # diag(logits) are predictions for future states.
        # diag(next_q) are predictions for random states, which correspond to
        # the predictions logits[range(B), goal_indices].
        # So, the only thing that's meaningful for next_q is the diagonal. Off
        # diagonal entries are meaningless and shouldn't be used.
        w = next_v / (1 - next_v)
        w_clipping = 20.0
        w = jnp.clip(w, 0, w_clipping)
        # (B, B, 2) --> (B, 2), computes diagonal of each twin Q.
        pos_logits = jax.vmap(jnp.diag, -1, -1)(logits)
        loss_pos = optax.sigmoid_binary_cross_entropy(
            logits=pos_logits, labels=1)  # [B, 2]

        neg_logits = logits[jnp.arange(batch_size), goal_indices]
        loss_neg1 = w[:, None] * optax.sigmoid_binary_cross_entropy(
            logits=neg_logits, labels=1)  # [B, 2]
        loss_neg2 = optax.sigmoid_binary_cross_entropy(
            logits=neg_logits, labels=0)  # [B, 2]

        if config.add_mc_to_td:
          loss = ((1 + (1 - config.discount)) * loss_pos
                  + config.discount * loss_neg1 + 2 * loss_neg2)
        else:
          loss = ((1 - config.discount) * loss_pos
                  + config.discount * loss_neg1 + loss_neg2)
        # Take the mean here so that we can compute the accuracy.
        logits = jnp.mean(logits, axis=-1)

      else:  # For the MC losses.
        def loss_fn(_logits):  # pylint: disable=invalid-name
          if config.use_cpc:
            return (optax.softmax_cross_entropy(logits=_logits, labels=I)
                    + 0.01 * jax.nn.logsumexp(_logits, axis=1)**2)
          else:
            return optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I)
        if len(logits.shape) == 3:  # twin q
          # loss.shape = [.., num_q]
          loss = jax.vmap(loss_fn, in_axes=2, out_axes=-1)(logits)
          loss = jnp.mean(loss, axis=-1)
          # Take the mean here so that we can compute the accuracy.
          logits = jnp.mean(logits, axis=-1)
        else:
          loss = loss_fn(logits)

      loss = jnp.mean(loss)
      correct = (jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1))
      logits_pos = jnp.sum(logits * I) / jnp.sum(I)
      logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
      if len(logits.shape) == 3:
        logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1)**2
      else:
        logsumexp = jax.nn.logsumexp(logits, axis=1)**2
      metrics = {
          'binary_accuracy': jnp.mean((logits > 0) == I),
          'categorical_accuracy': jnp.mean(correct),
          'logits_pos': logits_pos,
          'logits_neg': logits_neg,
          'logsumexp': logsumexp.mean(),
          'repr_phi_norm_mean': repr_phi_norm_mean,
          'repr_psi_norm_mean': repr_psi_norm_mean,
      }

      return loss, metrics

    def actor_loss(policy_params,
                   policy_params_prev,
                   q_params,
                   kappa_params,
                   kappa_params_2,
                   alpha,
                   transitions,
                   key,
                   ):
      """Policy gradient target: CRL logits diag, or min(κ·ψ) when reward_shaping_mode='kappa'.

      `kappa_params` / `kappa_params_2` are unused unless reward_shaping_mode is ``kappa``;
      they may be None for stock CRL runs.
      `policy_params_prev` is only used when reward_shaping_mode='kappa' and
      ``q_actor_kl_to_prev_coef`` > 0 (same trust-region estimator as q_sac).
      """
      obs = transitions.observation

      state = obs[:, :config.obs_dim]
      goal = obs[:, config.obs_dim:]

      if config.random_goals == 0.0:
        new_state = state
        new_goal = goal
      elif config.random_goals == 0.5:
        new_state = jnp.concatenate([state, state], axis=0)
        new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
      else:
        assert config.random_goals == 1.0
        new_state = state
        new_goal = jnp.roll(goal, 1, axis=0)

      new_obs = jnp.concatenate([new_state, new_goal], axis=1)
      dist_params = networks.policy_network.apply(policy_params, new_obs)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)

      if config.reward_shaping_mode == 'kappa':
        # Same HER-shaped batch as stock CRL, but maximise min(κ1·ψ, κ2·ψ) instead
        # of the contrastive logits diagonal.  Critic + κ params are stop-gradiented.
        _, phi_sa, psi_g = networks.q_network.apply(
            jax.lax.stop_gradient(q_params), new_obs, action)
        phi_sa = jax.lax.stop_gradient(phi_sa)
        psi_g = jax.lax.stop_gradient(psi_g)
        kappa1 = networks.kappa_network.apply(
            jax.lax.stop_gradient(kappa_params), new_obs, action)
        if config.kappa_actor_match_phi_norm:
          kappa1 = _scale_kappa_rows_to_phi_norm(kappa1, phi_sa)
        q1 = jnp.sum(kappa1 * psi_g, axis=-1)
        if config.twin_kappa:
          kappa2 = networks.kappa_network_2.apply(
              jax.lax.stop_gradient(kappa_params_2), new_obs, action)
          if config.kappa_actor_match_phi_norm:
            kappa2 = _scale_kappa_rows_to_phi_norm(kappa2, phi_sa)
          q2 = jnp.sum(kappa2 * psi_g, axis=-1)
          q_scalar = jnp.minimum(q1, q2)
        else:
          q_scalar = q1
        actor_loss_vec = -q_scalar
        metrics = {
            'entropy_mean': jnp.mean(-log_prob),
            'repr_q_min_mean': jnp.mean(q_scalar),
            'repr_log_prob_mean': jnp.mean(log_prob),
        }
        if (config.hard_goal is not None
            and float(config.kappa_actor_hard_goal_coef) != 0.0):
          hg = jnp.asarray(config.hard_goal, dtype=new_obs.dtype)
          hg = jnp.broadcast_to(
              hg[None, :], (new_state.shape[0], hg.shape[0]))
          hard_obs = jnp.concatenate([new_state, hg], axis=1)
          _, phi_h, psi_h = networks.q_network.apply(
              jax.lax.stop_gradient(q_params), hard_obs, action)
          phi_h = jax.lax.stop_gradient(phi_h)
          psi_h = jax.lax.stop_gradient(psi_h)
          k1h = networks.kappa_network.apply(
              jax.lax.stop_gradient(kappa_params), hard_obs, action)
          if config.kappa_actor_match_phi_norm:
            k1h = _scale_kappa_rows_to_phi_norm(k1h, phi_h)
          qh1 = jnp.sum(k1h * psi_h, axis=-1)
          if config.twin_kappa:
            k2h = networks.kappa_network_2.apply(
                jax.lax.stop_gradient(kappa_params_2), hard_obs, action)
            if config.kappa_actor_match_phi_norm:
              k2h = _scale_kappa_rows_to_phi_norm(k2h, phi_h)
            q_hard = jnp.minimum(qh1, jnp.sum(k2h * psi_h, axis=-1))
          else:
            q_hard = qh1
          c = jnp.asarray(
              float(config.kappa_actor_hard_goal_coef), dtype=q_hard.dtype)
          actor_loss_vec = actor_loss_vec - c * q_hard
          metrics['repr_actor_hard_goal_mean'] = jnp.mean(q_hard)
      else:
        q_action, _, _ = networks.q_network.apply(q_params, new_obs, action)

        if len(q_action.shape) == 3:  # twin q trick
          assert q_action.shape[2] == 2
          q_action = jnp.min(q_action, axis=-1)

        actor_loss_vec = -jnp.diag(q_action)  # negative -(Q): maximize Q
        metrics = {
            'entropy_mean': jnp.mean(-log_prob),
        }

      # action entropy loss (same as stock CRL)
      approx_entropy = -log_prob
      if config.use_action_entropy:
        actor_loss_vec = actor_loss_vec - alpha * approx_entropy

      if (config.reward_shaping_mode == 'kappa'
          and config.q_actor_kl_to_prev_coef > 0):
        dist_prev = networks.policy_network.apply(
            jax.lax.stop_gradient(policy_params_prev), new_obs)
        log_prob_prev = networks.log_prob(dist_prev, action)
        kl_prev = jnp.mean(log_prob - log_prob_prev)
        actor_loss_vec = (
            actor_loss_vec + config.q_actor_kl_to_prev_coef * kl_prev)
        metrics['q_actor_kl_prev'] = kl_prev

      if config.reward_shaping_mode == 'kappa':
        metrics['repr_actor_loss'] = jnp.mean(actor_loss_vec)

      return jnp.mean(actor_loss_vec), metrics

    def _make_kappa_loss(kappa_net, name_prefix):
      """Factory returning a Bellman loss for one κ network.

      κ(s,a) ∈ R^repr_dim: discounted sum of φ(s_t,a_t) along on-policy
      trajectories.  Loss: mean‖κ(s,a) − sg[φ(s,a) + γ·κ_target(s',a')]‖².
      φ and the policy are stop-gradiented so only κ's own params are trained.
      """
      def kappa_loss(kappa_params, target_kappa_params, q_params, policy_params,
                     transitions, key):
        obs = transitions.observation
        action = transitions.action
        g = obs[:, config.obs_dim:]

        # φ(s,a) from frozen critic — stop-gradient so critic is unaffected.
        _, phi_sa, _ = networks.q_network.apply(
            jax.lax.stop_gradient(q_params), obs, action)
        phi_sa = jax.lax.stop_gradient(phi_sa)  # (B, repr_dim)

        kappa_sa = kappa_net.apply(kappa_params, obs, action)

        next_s = transitions.next_observation[:, :config.obs_dim]
        # π(a'|s',g) and κ(s',a'): use fixed env goal when set (κ loss only).
        if config.hard_goal is not None:
          hard_goal = jnp.asarray(config.hard_goal, dtype=obs.dtype)
          hard_goal = jnp.broadcast_to(
              hard_goal[None, :], (obs.shape[0], hard_goal.shape[0]))
          next_obs = jnp.concatenate([next_s, hard_goal], axis=1)
        else:
          next_obs = jnp.concatenate([next_s, g], axis=1)
        next_dist = networks.policy_network.apply(
            jax.lax.stop_gradient(policy_params), next_obs)
        next_action = networks.sample(next_dist, key)

        kappa_next = kappa_net.apply(target_kappa_params, next_obs, next_action)
        # γ_κ is separate from the critic's discount; see config.discount_kappa.
        target = jax.lax.stop_gradient(
            phi_sa + config.discount_kappa * kappa_next)

        # Mean (not sum) over the repr_dim axis: with repr_dim=64, summing
        # effectively multiplies the gradient by 64 and makes the nominal
        # LR behave like ~2e-2 per coordinate, which is a major driver of
        # κ divergence.  Meaning keeps the per-dim gradient on the same
        # scale as any other MLP regression and lets `learning_rate_kappa`
        # set the actual update magnitude.
        loss = jnp.mean(jnp.mean((kappa_sa - target) ** 2, axis=-1))
        # How far κ(s,a) is from φ(s,a) (same batch) — Bellman target can differ
        # from φ when γ_κ>0; these scalars summarize the κ–φ alignment.
        diff = kappa_sa - phi_sa
        phi_l2 = jnp.linalg.norm(phi_sa, axis=-1)
        kap_l2 = jnp.linalg.norm(kappa_sa, axis=-1)
        eps = jnp.asarray(1e-8, dtype=diff.dtype)
        cos_num = jnp.sum(kappa_sa * phi_sa, axis=-1)
        cos_den = kap_l2 * phi_l2 + eps
        return loss, {
            f'{name_prefix}_loss': loss,
            f'{name_prefix}_norm_mean': jnp.mean(jnp.linalg.norm(kappa_sa, axis=-1)),
            f'{name_prefix}_target_norm_mean':
                jnp.mean(jnp.linalg.norm(target, axis=-1)),
            f'{name_prefix}_phi_l2_mean': jnp.mean(jnp.linalg.norm(diff, axis=-1)),
            f'{name_prefix}_phi_mse_mean': jnp.mean(jnp.mean(diff ** 2, axis=-1))
        }
      return kappa_loss

    def q_repr_loss(q_repr_params, q_repr_params_2,
                    target_q_repr_params, target_q_repr_params_2,
                    q_params, policy_params, transitions, key):
      """Clipped-double-Q TD loss for scalar Q(s,a,g) on r = sg(φ·ψ).

      Trains both Q1 and Q2 simultaneously against a shared target
      y = r + γ_q · min(Q̄1(s',a',g), Q̄2(s',a',g)), with a' ~ π(·|s',g).
      φ, ψ and π are stop-gradiented.
      """
      obs = transitions.observation
      action = transitions.action
      g = obs[:, config.obs_dim:]

      # Reward can be anchored to a fixed "hard goal" (from lp_contrastive)
      # instead of using the goal slice currently present in the replay obs.
      use_hard_goal = ((config.use_q_repr or config.use_kappa) and
                       config.hard_goal is not None)
      reward_obs = obs
      if use_hard_goal:
        hard_goal = jnp.asarray(config.hard_goal, dtype=obs.dtype)
        hard_goal = jnp.broadcast_to(hard_goal[None, :], (obs.shape[0], hard_goal.shape[0]))
        s = obs[:, :config.obs_dim]
        reward_obs = jnp.concatenate([s, hard_goal], axis=1)
      # r = φ(s,a) · ψ(g).  Pull both from the frozen CRL critic —
      # gradients must NOT flow into the critic or this becomes a
      # very indirect contrastive objective.
      _, phi_sa, psi_g = networks.q_network.apply(
          jax.lax.stop_gradient(q_params), reward_obs, action)
      reward = jnp.sum(jax.lax.stop_gradient(phi_sa) *
                       jax.lax.stop_gradient(psi_g), axis=-1)  # (B,)

      # Next state with the SAME goal (matches the HER convention used
      # by kappa_loss and critic_loss).
      next_s = transitions.next_observation[:, :config.obs_dim]
      next_obs = jnp.concatenate([next_s, g], axis=1)
      next_dist = networks.policy_network.apply(
          jax.lax.stop_gradient(policy_params), next_obs)
      next_action = networks.sample(next_dist, key)

      # Clipped-double-Q target.
      q1_next = networks.q_goal_network.apply(
          target_q_repr_params, next_obs, next_action)
      if config.twin_q_repr:
        q2_next = networks.q_goal_network_2.apply(
            target_q_repr_params_2, next_obs, next_action)
        q_next_min = jnp.minimum(q1_next, q2_next)
      else:
        q_next_min = q1_next
      target = jax.lax.stop_gradient(reward + config.discount_q * q_next_min)

      # Current Q predictions — gradients flow into each Q's own params.
      q1_pred = networks.q_goal_network.apply(q_repr_params, obs, action)
      loss1 = jnp.mean((q1_pred - target) ** 2)
      if config.twin_q_repr:
        q2_pred = networks.q_goal_network_2.apply(
            q_repr_params_2, obs, action)
        loss2 = jnp.mean((q2_pred - target) ** 2)
        loss = loss1 + loss2
      else:
        loss2 = jnp.zeros_like(loss1)
        loss = loss1

      metrics = {
          'q_repr_loss': loss,
          'q_repr_loss_1': loss1,
          'q_repr_loss_2': loss2,
          'q_repr_reward_std': jnp.std(reward),
          'q_repr_target_mean': jnp.mean(target),
          'q_repr_q1_mean': jnp.mean(q1_pred),
      }
      return loss, metrics

    if config.reward_shaping_mode == 'q':
      def actor_loss_q(policy_params, policy_params_prev,
                       q_repr_params, q_repr_params_2,
                       q_params,
                       alpha, transitions, key):
        """Soft actor on min(Q1,Q2)(s,a,g) (entropy-regularised like CRL).

        Mirrors the original `actor_loss`: gradients flow policy → action
        → Q networks (Q params frozen), α arrives as a scalar and is
        implicitly stop-gradiented via argnums=0.  Uses the shared
        adaptive-α pipeline (see alpha_loss / config.target_entropy).

        When `config.q_actor_kl_to_prev_coef > 0`, adds a trust-region
        penalty  β · KL(π_new ‖ π_prev)  estimated per-minibatch as
          mean_{a~π_new}[ log π_new(a|s) - log π_prev(a|s) ].
        `policy_params_prev` is stop-gradiented (see `argnums=0` below),
        so gradients only flow into `policy_params`.  This is the
        sample-based PPO-style estimator; it's unbiased for KL in
        expectation and can be slightly negative per-batch (that's fine
        for a soft penalty — the mean is non-negative).
        """
        obs = transitions.observation
        use_hard_goal = ((config.use_q_repr or config.use_kappa) and
                         config.hard_goal is not None)
        actor_obs = obs
        if use_hard_goal:
          hard_goal = jnp.asarray(config.hard_goal, dtype=obs.dtype)
          hard_goal = jnp.broadcast_to(
              hard_goal[None, :], (obs.shape[0], hard_goal.shape[0]))
          s = obs[:, :config.obs_dim]
          actor_obs = jnp.concatenate([s, hard_goal], axis=1)

        dist_params = networks.policy_network.apply(policy_params, actor_obs)
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)

        q1 = networks.q_goal_network.apply(
            jax.lax.stop_gradient(q_repr_params), actor_obs, action)
        if config.twin_q_repr:
          q2 = networks.q_goal_network_2.apply(
              jax.lax.stop_gradient(q_repr_params_2), actor_obs, action)
          q_min = jnp.minimum(q1, q2)
        else:
          q_min = q1

        loss = jnp.mean(alpha * log_prob - q_min)

        # --- HER-relabeled auxiliary via the frozen CRL critic ---
        # Follows the stock `actor_loss` (random_goals=0.5) pattern: build
        # a doubled batch whose second half pairs each state with a
        # goal rolled from another trajectory, then maximize the CRL
        # critic's diagonal on that batch.  `q_params` is stop-gradiented
        # so this provides NO training signal to φ/ψ — pure policy
        # gradient that uses the already-contrastively-trained critic
        # as a free auxiliary for goal generalization.
        her_aux_loss = jnp.zeros((), dtype=loss.dtype)
        her_q_diag_mean = jnp.zeros((), dtype=loss.dtype)
        her_log_prob_mean = jnp.zeros((), dtype=loss.dtype)
        if config.q_actor_her_aux_coef > 0:
          s_ = obs[:, :config.obs_dim]
          g_ = obs[:, config.obs_dim:]

          if config.random_goals == 0.0:
            her_state, her_goal = s_, g_
          elif config.random_goals == 0.5:
            her_state = jnp.concatenate([s_, s_], axis=0)
            her_goal = jnp.concatenate([g_, jnp.roll(g_, 1, axis=0)], axis=0)
          else:
            # random_goals == 1.0 → purely shuffled goals
            her_state = s_
            her_goal = jnp.roll(g_, 1, axis=0)

          her_obs = jnp.concatenate([her_state, her_goal], axis=1)

          # Fresh subkey for the HER sample (independent of the main
          # `action` sample above under JAX's PRNG model).
          key_her = jax.random.fold_in(key, 0xBABE)
          her_dist = networks.policy_network.apply(policy_params, her_obs)
          her_action = networks.sample(her_dist, key_her)
          her_log_prob = networks.log_prob(her_dist, her_action)

          # Frozen CRL critic — stop-gradient into q_params so the actor
          # cannot drag φ/ψ around to make the reward look better.
          her_q, _, _ = networks.q_network.apply(
              jax.lax.stop_gradient(q_params), her_obs, her_action)
          if len(her_q.shape) == 3:  # twin_q → min over the twin axis
            her_q = jnp.min(her_q, axis=-1)
          her_q_diag = jnp.diag(her_q)  # (2B,) or (B,) per random_goals

          # Soft-actor-style auxiliary: minimise α·log π - Q_CRL (= maximise Q + α·H).
          her_aux_loss = jnp.mean(alpha * her_log_prob - her_q_diag)
          loss = loss + config.q_actor_her_aux_coef * her_aux_loss

          her_q_diag_mean = jnp.mean(her_q_diag)
          her_log_prob_mean = jnp.mean(her_log_prob)

        kl_prev = jnp.zeros((), dtype=loss.dtype)
        if config.q_actor_kl_to_prev_coef > 0:
          # Evaluate log π_prev(a|s) at the SAME action sampled from π_new.
          # `policy_params_prev` was stop-gradiented at the call site; we
          # also wrap it here defensively so the apply's param pytree
          # carries no gradient wrt the previous snapshot.
          dist_prev = networks.policy_network.apply(
              jax.lax.stop_gradient(policy_params_prev), obs)
          log_prob_prev = networks.log_prob(dist_prev, action)
          kl_prev = jnp.mean(log_prob - log_prob_prev)
          loss = loss + config.q_actor_kl_to_prev_coef * kl_prev

        return loss, {
            'q_actor_loss': loss,
            'q_min_mean': jnp.mean(q_min),
            'q_log_prob_mean': jnp.mean(log_prob),
            'entropy_mean': jnp.mean(-log_prob),
            'q_actor_kl_prev': kl_prev,
            'q_actor_her_aux_loss': her_aux_loss,
            'q_actor_her_crl_diag_mean': her_q_diag_mean,
            'q_actor_her_log_prob_mean': her_log_prob_mean,
        }

    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)
    if config.reward_shaping_mode == 'q':
      actor_q_grad = jax.value_and_grad(actor_loss_q, argnums=0, has_aux=True)
    if config.use_q_repr:
      # Grads wrt both Q1 and Q2 (argnums=(0,1)) so we can update them in
      # one pass; the target is shared, so this is mathematically identical
      # to two independent updates for each Q.
      q_repr_grad = jax.value_and_grad(q_repr_loss, argnums=(0, 1),
                                       has_aux=True)
    if config.use_kappa:
      kappa_grad_1 = jax.value_and_grad(
          _make_kappa_loss(networks.kappa_network, 'kappa'), has_aux=True)
      if config.twin_kappa:
        kappa_grad_2 = jax.value_and_grad(
            _make_kappa_loss(networks.kappa_network_2, 'kappa2'), has_aux=True)

    def update_step(
        state,
        transitions
    ):
  
      key, key_alpha, key_critic, key_actor, key_kappa = (
          jax.random.split(state.key, 5))
      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient

      (critic_loss, critic_metrics), critic_grads = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          transitions, key_critic)

      # Apply critic gradients
      critic_update, q_optimizer_state = q_optimizer.update(critic_grads, state.q_optimizer_state)

      q_params = optax.apply_updates(state.q_params, critic_update)
       
      new_target_q_params = jax.tree_map(lambda x, y: x * (1 - config.tau) + y * config.tau, 
                                         state.target_q_params, q_params)
      metrics = critic_metrics
      
      # compute actor loss — all branches consume the shared adaptive α.
      if config.reward_shaping_mode == 'q':
        (actor_loss_val, actor_metrics), actor_grads = actor_q_grad(
            state.policy_params, state.policy_params_prev,
            state.q_repr_params, state.q_repr_params_2,
            state.q_params,
            alpha, transitions, key_actor)
      else:
        # Stock CRL (`reward_shaping_mode=''`) and κ-shaping (`'kappa'`) share
        # `actor_loss`; κ path is selected inside the loss via config.
        (actor_loss_val, actor_metrics), actor_grads = actor_grad(
            state.policy_params,
            state.policy_params_prev,
            state.q_params,
            state.kappa_params,
            state.kappa_params_2,
            alpha, transitions, key_actor)

      # Apply policy gradients
      policy_params_prev = state.policy_params
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

      metrics.update({
          'critic_loss': critic_loss,
          'actor_loss': actor_loss_val,
      })

      metrics.update(actor_metrics)
  
      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          policy_params_prev=policy_params_prev,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key,
          # κ fields — carried forward unchanged; overwritten below if use_kappa.
          kappa_params=state.kappa_params,
          target_kappa_params=state.target_kappa_params,
          kappa_optimizer_state=state.kappa_optimizer_state,
          kappa_params_2=state.kappa_params_2,
          target_kappa_params_2=state.target_kappa_params_2,
          kappa_optimizer_state_2=state.kappa_optimizer_state_2,
          # Q-repr fields — carried forward; overwritten below if use_q_repr.
          q_repr_params=state.q_repr_params,
          target_q_repr_params=state.target_q_repr_params,
          q_repr_optimizer_state=state.q_repr_optimizer_state,
          q_repr_params_2=state.q_repr_params_2,
          target_q_repr_params_2=state.target_q_repr_params_2,
          q_repr_optimizer_state_2=state.q_repr_optimizer_state_2,
      )

      if config.use_kappa:
        key_kappa, key_kappa2 = jax.random.split(key_kappa)

        # --- κ1 update ---
        (_, k1_metrics), k1_grads = kappa_grad_1(
            state.kappa_params, state.target_kappa_params, state.q_params,
            state.policy_params, transitions, key_kappa)
        k1_update, new_k1_opt_state = kappa_optimizer.update(
            k1_grads, state.kappa_optimizer_state)
        new_kappa_params = optax.apply_updates(state.kappa_params, k1_update)
        new_target_kappa_params = jax.tree_map(
            lambda x, y: x * (1 - config.tau) + y * config.tau,
            state.target_kappa_params, new_kappa_params)
        metrics.update(k1_metrics)
        new_state = new_state._replace(
            kappa_params=new_kappa_params,
            target_kappa_params=new_target_kappa_params,
            kappa_optimizer_state=new_k1_opt_state,
        )

        # --- κ2 update (twin_kappa only) ---
        if config.twin_kappa:
          (_, k2_metrics), k2_grads = kappa_grad_2(
              state.kappa_params_2, state.target_kappa_params_2, state.q_params,
              state.policy_params, transitions, key_kappa2)
          k2_update, new_k2_opt_state = kappa_optimizer.update(
              k2_grads, state.kappa_optimizer_state_2)
          new_kappa_params_2 = optax.apply_updates(state.kappa_params_2, k2_update)
          new_target_kappa_params_2 = jax.tree_map(
              lambda x, y: x * (1 - config.tau) + y * config.tau,
              state.target_kappa_params_2, new_kappa_params_2)
          metrics.update(k2_metrics)
          new_state = new_state._replace(
              kappa_params_2=new_kappa_params_2,
              target_kappa_params_2=new_target_kappa_params_2,
              kappa_optimizer_state_2=new_k2_opt_state,
          )

      if config.use_q_repr:
        key_q_repr, _ = jax.random.split(key_kappa)  # reuse κ key slot
        (_, q_repr_metrics), (q1_grads, q2_grads) = q_repr_grad(
            state.q_repr_params, state.q_repr_params_2,
            state.target_q_repr_params, state.target_q_repr_params_2,
            state.q_params, state.policy_params, transitions, key_q_repr)

        q1_update, new_q1_opt_state = q_repr_optimizer.update(
            q1_grads, state.q_repr_optimizer_state)
        new_q_repr_params = optax.apply_updates(state.q_repr_params, q1_update)
        new_target_q_repr_params = jax.tree_map(
            lambda x, y: x * (1 - config.tau) + y * config.tau,
            state.target_q_repr_params, new_q_repr_params)
        new_state = new_state._replace(
            q_repr_params=new_q_repr_params,
            target_q_repr_params=new_target_q_repr_params,
            q_repr_optimizer_state=new_q1_opt_state,
        )
        if config.twin_q_repr:
          q2_update, new_q2_opt_state = q_repr_optimizer.update(
              q2_grads, state.q_repr_optimizer_state_2)
          new_q_repr_params_2 = optax.apply_updates(
              state.q_repr_params_2, q2_update)
          new_target_q_repr_params_2 = jax.tree_map(
              lambda x, y: x * (1 - config.tau) + y * config.tau,
              state.target_q_repr_params_2, new_q_repr_params_2)
          new_state = new_state._replace(
              q_repr_params_2=new_q_repr_params_2,
              target_q_repr_params_2=new_target_q_repr_params_2,
              q_repr_optimizer_state_2=new_q2_opt_state,
          )
        metrics.update(q_repr_metrics)

      if adaptive_entropy_coefficient:
        # Apply alpha gradients
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
        metrics.update({
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params),
        })
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params)

      return_state = new_state
      return return_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray,
        time_delta=10.0)

    # Iterator on demonstration transitions.
    self._iterator = iterator

    update_step = utils.process_multiple_batches(update_step,config.num_sgd_steps_per_step)
    # Use the JIT compiler.
    if config.jit:
      self._update_step = jax.jit(update_step)
    else:
      self._update_step = update_step

    def make_initial_state(key):
      """Initialises the training state (parameters and optimiser state)."""
      (key_policy, key_q, key_kappa, key_kappa_2,
       key_q_repr, key_q_repr_2, key) = jax.random.split(key, 7)

      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          policy_params_prev=policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key)

      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)

      if config.use_kappa:
        kappa_params = networks.kappa_network.init(key_kappa)
        state = state._replace(
            kappa_params=kappa_params,
            target_kappa_params=kappa_params,
            kappa_optimizer_state=kappa_optimizer.init(kappa_params),
        )
        if config.twin_kappa:
          kappa_params_2 = networks.kappa_network_2.init(key_kappa_2)
          state = state._replace(
              kappa_params_2=kappa_params_2,
              target_kappa_params_2=kappa_params_2,
              kappa_optimizer_state_2=kappa_optimizer.init(kappa_params_2),
          )

      if config.use_q_repr:
        q_repr_params = networks.q_goal_network.init(key_q_repr)
        state = state._replace(
            q_repr_params=q_repr_params,
            target_q_repr_params=q_repr_params,
            q_repr_optimizer_state=q_repr_optimizer.init(q_repr_params),
        )
        if config.twin_q_repr:
          q_repr_params_2 = networks.q_goal_network_2.init(key_q_repr_2)
          state = state._replace(
              q_repr_params_2=q_repr_params_2,
              target_q_repr_params_2=q_repr_params_2,
              q_repr_optimizer_state_2=q_repr_optimizer.init(q_repr_params_2),
          )

      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online
    # and fill the replay buffer.
    self._timestamp = None

  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)
      self._state, metrics = self._update_step(self._state, transitions) 
    
    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp
    
    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    
    if elapsed_time > 0:
      metrics['steps_per_second'] = (
          self._num_sgd_steps_per_step / elapsed_time)
    else:
      metrics['steps_per_second'] = 0.
    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names):
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.q_params,
    }
    if self._state.kappa_params is not None:
      variables['kappa'] = self._state.kappa_params
    if self._state.kappa_params_2 is not None:
      variables['kappa_2'] = self._state.kappa_params_2
    if self._state.q_repr_params is not None:
      variables['q_repr'] = self._state.q_repr_params
    if self._state.q_repr_params_2 is not None:
      variables['q_repr_2'] = self._state.q_repr_params_2
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
