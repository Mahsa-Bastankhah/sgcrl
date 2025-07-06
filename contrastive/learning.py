"""Contrastive RL learner implementation."""
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable
import pickle
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
import tensorflow as tf
from acme.jax import savers        # for SaveableAdapter
from contrastive import utils as contrastive_utils     # obs_to_goal helpers
import functools
from acme.tf.savers import SaveableAdapter
from acme import core
import glob, re, os, tensorflow as tf
import os, functools, tensorflow as tf, jax
import os, pickle
from pathlib import Path
import haiku as hk, jax, jax.numpy as jnp
from typing import Mapping


def load_ckpt(path: str, *, first: bool = True):
    # --------------------------------------------------------
    # 1) Resolve prefix  (dir → ckpt-0  …  or latest)
    # --------------------------------------------------------
    if os.path.isdir(path):
        if first:
            idx_files = sorted(
                glob.glob(os.path.join(path, "ckpt-*.index")),
                key=lambda p: int(re.search(r"ckpt-(\d+)\.index", p).group(1)),
            )
            if not idx_files:
                raise FileNotFoundError(f"No ckpt-*.index files inside {path}")
            ckpt_prefix = idx_files[0][:-6]  # strip ".index"
        else:
            ckpt_prefix = tf.train.latest_checkpoint(path)
            if ckpt_prefix is None:
                raise FileNotFoundError(f"No checkpoints inside {path}")
    else:
        ckpt_prefix = path

    print(f"[warm-start] reading weights from {ckpt_prefix}", flush=True)

    # --------------------------------------------------------
    # 2) Grab the PythonState blob & extract the two param trees
    # --------------------------------------------------------
    reader = tf.train.load_checkpoint(ckpt_prefix)
    blob   = reader.get_tensor("learner/.ATTRIBUTES/py_state")
    state  = pickle.loads(blob)

    policy_params = state.policy_params
    q_params      = state.q_params
    return policy_params, q_params


def save_params(params: Dict[str, Any], save_dir: str, name: str = "initial_params.pkl"):
    """
    params: dict, e.g. {"policy_params": ..., "q_params": ...}
    save_dir: the directory where you want to write the pickle
    """
    os.makedirs(save_dir, exist_ok=True)
    path = Path(save_dir) / name
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"[init] saved params to {path}", flush = True)

def load_saved_params(save_dir: str, name: str = "initial_params.pkl"):
    """
    Returns the dict you originally saved.
    """
    path = Path(save_dir) / name
    with open(path, "rb") as f:
        return pickle.load(f)

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
    self.config = config
    self.actor_steps = 0
    
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
                    key,
                    use_goal_neg: bool = False):
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
            fixed_goal = self.config.fixed_goal
            fixed_goal = jnp.asarray(fixed_goal, dtype=jnp.float32)   # shape (d,)
            B = _logits.shape[0]
            
            # ## Adding goal as a negative example
            labels = I
            if (fixed_goal is not None) and use_goal_neg:
              # ensure fixed_goal is a JAX array, not a Python list
              # debug.print("[DBG] fixed_goal is not None, using it as a negative example")
              
              


              # split current states  s  |  g
              s, _ = jnp.split(transitions.observation,
                              [config.obs_dim], axis=1)
              

      

              # replicate the fixed goal so we have B copies
              g_fixed = jnp.broadcast_to(fixed_goal, (B, fixed_goal.shape[-1]))

              obs_fixed = jnp.concatenate([s, g_fixed], axis=1)        # (B , 2*obs_dim)
              fixed_logits, _, _ = networks.q_network.apply(
                  q_params, obs_fixed, transitions.action)             # (B [,2])
              fixed_logits = fixed_logits[:, 0]    

              # ensure shape is (B ,1)  or  (B ,1 ,2) in twin-Q case
              if fixed_logits.ndim == 1:
                  fixed_logits = fixed_logits[:, None]
              else:
                  fixed_logits = fixed_logits[:, None, :]

              # append as a new column on the right
              _logits = jnp.concatenate([_logits, fixed_logits], axis=1)
              #debug.print("[DBG] logits shape after concat {}", _logits.shape)

              # extend the label matrix (all zeros → still a negative)
              labels = jnp.concatenate(
                      [I, jnp.zeros((B, 1), I.dtype)],
                      axis=1)
              #debug.print("[DBG] labels shape after concat {}", labels.shape)
            if self.config.perturbed_negatives_num > 0:
              obs_dim = config.obs_dim
              s, _ = jnp.split(transitions.observation, [obs_dim], axis=1)
              num_negatives = self.config.perturbed_negatives_num
              

              # ---- 1. draw uniform noise in [-2 , 2]  ----
              scale  = 2.0
              noise  = jax.random.uniform(
                  key,
                  shape=(B, num_negatives, obs_dim),
                  minval=-scale,
                  maxval= scale
              )
              s_perturbed = s[:, None, :] + noise          # (B , 5 , obs_dim)

              # ---- 2. build [s , s′] pairs  ----
              s_rep             = jnp.repeat(s, num_negatives, axis=0)              # (B*5 , obs_dim)
              s_perturbed_flat  = s_perturbed.reshape(B * num_negatives, obs_dim)   # (B*5 , obs_dim)
              obs_neg           = jnp.concatenate([s_rep, s_perturbed_flat], axis=1)  # (B*5 , 2*obs_dim)

              action_rep        = jnp.repeat(transitions.action, num_negatives, axis=0)



              neg_logits, _, _  = networks.q_network.apply(q_params, obs_neg, action_rep)
              neg_logits        = neg_logits[:, 0]
              neg_logits        = neg_logits[:, None] if neg_logits.ndim == 1 else neg_logits[:, None, :]
              neg_logits        = neg_logits.reshape(B, num_negatives, -1)          # (B , 5 , 1) or (B , 5 , 2)
              neg_logits = jnp.squeeze(neg_logits, axis=-1)      # (B*K,)  ← get rid of last dim
              neg_logits = neg_logits.reshape(B, num_negatives)  # (B , K)  rank-2 matrix
              # ---- 3. concatenate & label  ----
              _logits = jnp.concatenate([_logits, neg_logits], axis=1)
              labels  = jnp.concatenate([labels, jnp.zeros((B, num_negatives), I.dtype)], axis=1)                        
            

            # ------------------------------------------------------------------
            # (2-alt)  Perturbed-state w/ fixed-goal negatives  q(s′ , g_fixed)
            # ------------------------------------------------------------------
            if (self.config.perturbed_negatives_goal_num > 0) and (fixed_goal is not None):
                obs_dim       = config.obs_dim
                num_negatives = self.config.perturbed_negatives_goal_num

                # ── split observation into state s | g_original (unused here) ──
                s, _ = jnp.split(transitions.observation, [obs_dim], axis=1)  # s: (B , obs_dim)
                B    = s.shape[0]

                # ── 1. draw uniform noise in [-scale , +scale] and perturb s ──
                
                scale = 2.0
                noise = jax.random.uniform(
                    key,
                    shape=(B, num_negatives, obs_dim),
                    minval=-scale,
                    maxval= scale
                )
                s_perturbed = s[:, None, :] + noise                             # (B , K , obs_dim)

                # ── 2. build observations [s′ , g_fixed]  ──
                s_perturbed_flat = s_perturbed.reshape(B * num_negatives, obs_dim)  # (B*K , obs_dim)
                g_fixed          = jnp.asarray(self.config.fixed_goal, dtype=jnp.float32)
                g_fixed_rep      = jnp.broadcast_to(g_fixed, (B * num_negatives, g_fixed.shape[-1]))
                obs_neg          = jnp.concatenate([s_perturbed_flat, g_fixed_rep], axis=1)  # (B*K , 2*obs_dim)

                action_rep = jnp.repeat(transitions.action, num_negatives, axis=0)  # (B*K , act_dim)

                # ── 3. critic forward: q(s′ , g_fixed) ──
                # ---- 3. critic forward: full similarity matrix (N × N) ----
                neg_logits_full, _, _ = networks.q_network.apply(q_params, obs_neg, action_rep)

                # ---- 4. grab just the diagonal  ----------------------------
                neg_logits_vec = jnp.diag(neg_logits_full)          # (B*K,)

                # If you have twin-Q (rank-3, N × N × 2), use:
                # neg_logits_vec = jax.vmap(jnp.diag, in_axes=2, out_axes=-1)(neg_logits_full)
                # neg_logits_vec = jnp.min(neg_logits_vec, axis=-1)  # twin-Q trick

                # ---- 5. reshape to (B , K) so rank = 2 ---------------------
                neg_logits = neg_logits_vec.reshape(B, num_negatives)   # (B , K)

                # ---- 6. concatenate & label -------------------------------
                _logits = jnp.concatenate([_logits, neg_logits], axis=1)        # both rank-2
                labels  = jnp.concatenate(
                    [labels, jnp.zeros((B, num_negatives), I.dtype)],
                    axis=1
                )


                # ── 5. sanity prints (work in JIT via jax.debug.print) ──
                # jax.debug.print("\n[DBG-PG] ----- perturbed+fixed-goal branch -----")
                # jax.debug.print("[DBG-PG] s.shape              = {}", s.shape)
                # jax.debug.print("[DBG-PG] s_perturbed.shape     = {}", s_perturbed.shape)
                # jax.debug.print("[DBG-PG] obs_neg.shape         = {}", obs_neg.shape)
                # jax.debug.print("[DBG-PG] neg_logits.shape      = {}", neg_logits.shape)
                # jax.debug.print("[DBG-PG] _logits final shape   = {}", _logits.shape)
                # jax.debug.print("[DBG-PG] labels  final shape   = {}", labels.shape)

            return (optax.softmax_cross_entropy(logits=_logits, labels=labels)
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

    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

    def update_step(
            state,
            transitions,
            *,                       # keep it a keyword-only arg
            use_goal_neg: bool,
    ):
  
      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient

      # (critic_loss, critic_metrics), critic_grads = critic_grad(
      #     state.q_params, state.policy_params, state.target_q_params,
      #     transitions, key_critic)
      (critic_loss, critic_metrics), critic_grads = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          transitions, key_critic, use_goal_neg=use_goal_neg)

      # Apply critic gradients
      critic_update, q_optimizer_state = q_optimizer.update(critic_grads, state.q_optimizer_state)

      q_params = optax.apply_updates(state.q_params, critic_update)
       
      new_target_q_params = jax.tree_map(lambda x, y: x * (1 - config.tau) + y * config.tau, 
                                         state.target_q_params, q_params)
      metrics = critic_metrics
      
      # compute actor loss                               
      (actor_loss, actor_metrics), actor_grads = actor_grad(state.policy_params, state.q_params, 
                                                            alpha, transitions, key_actor)
                                     
      # Apply policy gradients
      policy_params_prev = state.policy_params
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)
                                     
      metrics.update({
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
      })
      
      metrics.update(actor_metrics)
  
      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          policy_params_prev=policy_params_prev,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key
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
    
      return_state = new_state
      return return_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger(
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray,
        time_delta=10.0)

    # Iterator on demonstration transitions.
    self._iterator = iterator

    # update_step = utils.process_multiple_batches(update_step,config.num_sgd_steps_per_step)
    # #Use the JIT compiler.
    # if config.jit:
    #   self._update_step = jax.jit(update_step)
    # else:
    #   self._update_step = update_step

    def make_update(use_goal_neg: bool):
      # 1. freeze the flag so the inner fn now has *only* (state, trans)
      step_fn = functools.partial(update_step, use_goal_neg=use_goal_neg)
      # 2. let Acme split big batches if requested
      step_fn = utils.process_multiple_batches(
          step_fn, config.num_sgd_steps_per_step
      )
      # 3. JIT if desired (no extra static args now)
      return jax.jit(step_fn) if config.jit else step_fn

    self._update_step_true  = make_update(True)   # goal is a negative
    self._update_step_false = make_update(False)  # stop using it

    def make_initial_state(key):
      log_subdir = os.path.join(config.log_dir,
                              f"{config.alg_name}_{config.env_name}_{config.seed}")
      ckpt_dir   = os.path.join(log_subdir, "checkpoints", "learner")


      if config.init_weight is not None:
          # policy_params, q_params = load_ckpt(config.init_ckpt)
          try:
            saved = load_saved_params(config.init_weight)
            
            #policy_params = saved["policy_params"]
            q_params      = saved["q_params"]
            print(q_params, flush= True)  
            print(f"[warm-start] loaded initial weights from pickle in {config.init_weight}", flush=True)
          except FileNotFoundError:
            # Fallback: grab them from the very first TF checkpoint
            print(f"[warm-start] No pickel was found", flush=True)
          # Fresh optimiser slots
          
          q_optimizer_state      = q_optimizer.init(q_params)
          key_policy, key_q, key = jax.random.split(key, 3)
          policy_params = networks.policy_network.init(key_policy)
          policy_optimizer_state = policy_optimizer.init(policy_params)



      else:
        print(f"[Init] random initialisation of weights {config.init_weight}", flush=True)
        key_policy, key_q, key = jax.random.split(key, 3)
        
        ### cold initialization helper function
        def uniform_coldify_last_linear(params: hk.Params, rng_key,
                                scale: float = 1e-12) -> hk.Params:
          mut = hk.data_structures.to_mutable_dict(params)

          for prefix in ("sa_encoder", "g_encoder"):
              # collect all leaves that belong to a linear_* module under this prefix
              linear_items = [
                  (k, v)
                  for k, v in mut.items()
                  if re.search(rf"^{prefix}.*?/linear_(\d+)(/w|/b)?$", k)
              ]
              if not linear_items:
                  print(f"[ColdInit] WARNING: no Linear layer found in {prefix}", flush=True)
                  continue

              # pick the highest index
              last_idx = max(int(re.search(r"linear_(\d+)", k).group(1))
                            for k, _ in linear_items)
              print(f"[ColdInit] Found {len(linear_items)} linear layers in {prefix}, ", flush=True)
              print(f"[ColdInit] last index is {last_idx}", flush=True) 
              print(f"[ColdInit] linear_items: {linear_items}", flush=True)
              # re-initialise every leaf that belongs to that index
              # --- after you computed `last_idx` ---------------------------------
              for k, block in linear_items:          # block is {'w': array, 'b': array}
                  if f"linear_{last_idx}" not in k:  # skip earlier linear layers
                      continue

                  # re-seed the weight matrix only
                  rng_key, sub = jax.random.split(rng_key)
                  w_arr = block["w"]                 # current weights
                  block["w"] = jax.random.uniform(
                      sub,
                      w_arr.shape,
                      minval=-scale,
                      maxval=scale,
                      dtype=w_arr.dtype,
                  )

                  mut[k] = block   
                                  # write the updated block back
          print(mut, flush=True)
          return hk.data_structures.to_immutable_dict(mut)

        policy_params = networks.policy_network.init(key_policy)
        policy_optimizer_state = policy_optimizer.init(policy_params)
        q_params = networks.q_network.init(key_q)


        # NEW: cold-start the final layer if requested
        if config.cold_q_init:
            print(f"[ColdInit] Initialising last Q layer with U[-{config.cold_q_scale}, {config.cold_q_scale}]", flush=True)
            q_params = uniform_coldify_last_linear(
                q_params,
                rng_key=key_q,                 # reuse the same sub-key is fine
                scale=config.cold_q_scale      # 1e-12 by default
            )


        q_optimizer_state = q_optimizer.init(q_params)
        if config.save_init_weight:
          # save them for future reuse
          save_params({
              "policy_params": policy_params,
              "q_params":      q_params
          }, save_dir=ckpt_dir)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          policy_params_prev = policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key)

      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                              alpha_params=log_alpha)
        
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

      ## Added logic to support adding fixed goal as negative example
      counts = self._counter.get_counts()          # Python dict
      actor_steps = counts.get('actor_steps', 0)
      # If the number of actor steps is less than goal_neg_actor_steps then use the goal as a negative example
      use_goal_neg  = actor_steps < self.config.goal_neg_actor_steps


      # ──────────────────────────────────────────────────────────────
      # NEW -- randomly replace 20 % of future states with the fixed goal
      # ──────────────────────────────────────────────────────────────
      if self.config.fixed_goal is not None and actor_steps < self.config.goal_pos_actor_steps:
        obs        = transitions.observation           # (B, 2*obs_dim)
        B          = obs.shape[0]
        obs_dim    = self.config.obs_dim               # e.g. 2
        frac       = 0.2                              # 10 %
        k1, k2     = jax.random.split(self._state.key) # reuse learner RNG
        idx        = jax.random.choice(                # (⌈0.1 B⌉,)
                       k1, B,
                       (max(1, int(B * frac)),),       # at least one row
                       replace=False)
        fixed_goal = jnp.asarray(
                       self.config.fixed_goal,
                       dtype=obs.dtype)                # (obs_dim,)

        # broadcast & set the goal part (last obs_dim cols) at idx rows
        obs  = obs.at[idx, obs_dim:].set(
                 jnp.broadcast_to(fixed_goal, (idx.size, obs_dim)))
        #print(f"[DBG] using fixed goal {fixed_goal} as a posuitve example for {idx.size} rows", flush=True)
        
        transitions = transitions._replace(observation=obs)
        # keep the new RNG key in learner state so next step differs
        self._state = self._state._replace(key=k2)

      update_fn = (self._update_step_true
                 if use_goal_neg
                 else self._update_step_false)

      
      self._state, metrics = update_fn(self._state, transitions)
      #self._state, metrics = self._update_step(self._state, transitions) 
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
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
