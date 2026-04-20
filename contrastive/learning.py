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
  # repr_reward_actor='sac': separate temperature for the κ·ψ actor objective.
  log_alpha_repr: Optional[networks_lib.Params] = None
  alpha_repr_optimizer_state: Optional[optax.OptState] = None


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
      kappa_optimizer = optax.adam(learning_rate=config.learning_rate)

    if config.repr_reward_actor == 'sac':
      log_alpha_repr_init = jnp.asarray(0., dtype=jnp.float32)
      alpha_repr_optimizer = optax.adam(learning_rate=3e-4)

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
      
      logits, _, _ = networks.q_network.apply(q_params, transitions.observation, transitions.action)

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
      }

      return loss, metrics

    def actor_loss(policy_params,
                   q_params,
                   alpha,
                   transitions,
                   key,
                   ):
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

      q_action, sa_repr, sf_repr = networks.q_network.apply(q_params, new_obs, action)

      if len(q_action.shape) == 3:  # twin q trick
        assert q_action.shape[2] == 2
        q_action = jnp.min(q_action, axis=-1)

      actor_loss = -jnp.diag(q_action) # negative -(Q): maximize Q

      # action entropy loss
      approx_entropy = -log_prob

      if config.use_action_entropy:
        actor_loss -= alpha * approx_entropy # negative -(-log prob): maximize entropy

      metrics = {
          'entropy_mean': jnp.mean(approx_entropy),
      }

      
      return jnp.mean(actor_loss), metrics

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
        next_obs = jnp.concatenate([next_s, g], axis=1)
        next_dist = networks.policy_network.apply(
            jax.lax.stop_gradient(policy_params), next_obs)
        next_action = networks.sample(next_dist, key)

        kappa_next = kappa_net.apply(target_kappa_params, next_obs, next_action)
        target = jax.lax.stop_gradient(phi_sa + config.discount * kappa_next)

        loss = jnp.mean(jnp.sum((kappa_sa - target) ** 2, axis=-1))
        return loss, {
            f'{name_prefix}_loss': loss,
            f'{name_prefix}_norm_mean': jnp.mean(jnp.linalg.norm(kappa_sa, axis=-1)),
        }
      return kappa_loss

    if config.repr_reward_actor == 'sac':
      def alpha_repr_loss(log_alpha_repr, policy_params, transitions, key):
        """SAC temperature loss for the repr-based actor (same form as alpha_loss)."""
        dist_params = networks.policy_network.apply(
            policy_params, transitions.observation)
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)
        alpha_val = jnp.exp(log_alpha_repr)
        return jnp.mean(
            alpha_val * jax.lax.stop_gradient(
                -log_prob - config.repr_reward_target_entropy))

      def actor_loss_repr(policy_params, kappa_params, kappa_params_2,
                          q_params, log_alpha_repr, transitions, key):
        """SAC actor loss: maximise min(κ1·ψ, κ2·ψ) subject to entropy >= H0.

        Gradients flow policy_params → action → κ networks (κ params frozen).
        Both ψ and κ params are stop-gradiented so only the policy is trained.
        """
        obs = transitions.observation
        dist_params = networks.policy_network.apply(policy_params, obs)
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)

        # ψ(g) from the frozen critic; action-independent but we must pass one.
        _, _, psi_g = networks.q_network.apply(
            jax.lax.stop_gradient(q_params), obs, action)
        psi_g = jax.lax.stop_gradient(psi_g)  # (B, repr_dim)

        # κ1(s,a)·ψ(g) — gradients flow through action → policy_params.
        kappa1 = networks.kappa_network.apply(
            jax.lax.stop_gradient(kappa_params), obs, action)  # (B, repr_dim)
        q1 = jnp.sum(kappa1 * psi_g, axis=-1)  # (B,)

        if config.twin_kappa:
          kappa2 = networks.kappa_network_2.apply(
              jax.lax.stop_gradient(kappa_params_2), obs, action)
          q2 = jnp.sum(kappa2 * psi_g, axis=-1)
          q_min = jnp.minimum(q1, q2)
        else:
          q_min = q1

        alpha_repr = jnp.exp(log_alpha_repr)
        loss = jnp.mean(alpha_repr * log_prob - q_min)
        return loss, {
            'repr_actor_loss': loss,
            'repr_q_min_mean': jnp.mean(q_min),
            'repr_log_prob_mean': jnp.mean(log_prob),
            'repr_alpha': alpha_repr,
        }

    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)
    if config.repr_reward_actor == 'sac':
      alpha_repr_grad = jax.value_and_grad(alpha_repr_loss)
      actor_repr_grad = jax.value_and_grad(actor_loss_repr, argnums=0, has_aux=True)
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
  
      key, key_alpha, key_critic, key_actor, key_kappa, key_alpha_repr = (
          jax.random.split(state.key, 6))
      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient

      if config.repr_reward_actor == 'sac':
        alpha_repr_loss_val, alpha_repr_grads = alpha_repr_grad(
            state.log_alpha_repr, state.policy_params, transitions, key_alpha_repr)

                       
      (critic_loss, critic_metrics), critic_grads = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          transitions, key_critic)

      # Apply critic gradients
      critic_update, q_optimizer_state = q_optimizer.update(critic_grads, state.q_optimizer_state)

      q_params = optax.apply_updates(state.q_params, critic_update)
       
      new_target_q_params = jax.tree_map(lambda x, y: x * (1 - config.tau) + y * config.tau, 
                                         state.target_q_params, q_params)
      metrics = critic_metrics
      
      # compute actor loss
      if config.repr_reward_actor == 'sac':
        (actor_loss_val, actor_metrics), actor_grads = actor_repr_grad(
            state.policy_params, state.kappa_params, state.kappa_params_2,
            state.q_params, state.log_alpha_repr, transitions, key_actor)
      else:
        (actor_loss_val, actor_metrics), actor_grads = actor_grad(
            state.policy_params, state.q_params, alpha, transitions, key_actor)

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
          # repr SAC α — carried forward; overwritten below if repr_reward_actor='sac'.
          log_alpha_repr=state.log_alpha_repr,
          alpha_repr_optimizer_state=state.alpha_repr_optimizer_state,
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

      if config.repr_reward_actor == 'sac':
        alpha_repr_update, new_alpha_repr_opt_state = alpha_repr_optimizer.update(
            alpha_repr_grads, state.alpha_repr_optimizer_state)
        new_log_alpha_repr = optax.apply_updates(state.log_alpha_repr, alpha_repr_update)
        metrics.update({
            'repr_alpha_loss': alpha_repr_loss_val,
            'repr_alpha': jnp.exp(new_log_alpha_repr),
        })
        new_state = new_state._replace(
            log_alpha_repr=new_log_alpha_repr,
            alpha_repr_optimizer_state=new_alpha_repr_opt_state,
        )

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
      key_policy, key_q, key_kappa, key_kappa_2, key = jax.random.split(key, 5)

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

      if config.repr_reward_actor == 'sac':
        state = state._replace(
            log_alpha_repr=log_alpha_repr_init,
            alpha_repr_optimizer_state=alpha_repr_optimizer.init(log_alpha_repr_init),
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
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
