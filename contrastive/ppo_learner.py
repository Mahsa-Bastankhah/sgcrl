"""PPO learner on representation-based rewards (r = φ·ψ).

Standalone single-process implementation.  No Launchpad, no Reverb.

Design (matches user requirements — SAC code stays untouched):
  * The policy that collects data IS the PPO policy being optimized.  PPO
    updates are strictly on-policy over each fresh rollout batch.
  * Per-step reward:  r_t = φ(s_t, a_t) · ψ(g_t), using the *current* CRL
    representations (frozen for the reward computation but updated
    off-policy from the replay buffer alongside PPO).
  * Rollouts are written to an in-memory FIFO replay buffer; the CRL
    critic (φ, ψ) is trained off-policy from the replay, unchanged from
    the existing InfoNCE / C-learning losses.
  * No κ network is used — PPO brings its own value net V(s, g).

Implementation style mirrors CleanRL's ppo_continuous_action.py:
  * Flat rollout storage (T × num_envs tensors)
  * GAE advantages, clipped surrogate, clipped value loss
  * Multiple PPO epochs + minibatches per update

Everything runs in JAX for consistency with the rest of this codebase.
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


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------
class PPOTrainingState(NamedTuple):
  """All trainable state held by the PPO + CRL learner."""
  # PPO-side params (policy + value) share a single Adam optimizer, CleanRL-style.
  policy_params: networks_lib.Params
  value_params: networks_lib.Params
  ppo_optimizer_state: optax.OptState
  # CRL-side params trained from replay.
  q_params: networks_lib.Params
  q_optimizer_state: optax.OptState
  key: networks_lib.PRNGKey


# ---------------------------------------------------------------------------
# Rollout storage  (populated on-policy, consumed by GAE + PPO update)
# ---------------------------------------------------------------------------
class Rollout(NamedTuple):
  """A (T, num_envs, ...) block of on-policy experience."""
  obs: jnp.ndarray           # (T, E, obs_dim_total)
  actions: jnp.ndarray       # (T, E, act_dim)
  logprobs: jnp.ndarray      # (T, E)
  rewards: jnp.ndarray       # (T, E)           r_t = φ·ψ
  dones: jnp.ndarray         # (T, E)           terminal OR truncation at step t
  values: jnp.ndarray        # (T, E)           V(s_t, g_t)
  next_obs: jnp.ndarray      # (E, obs_dim_total)  final bootstrap state
  next_done: jnp.ndarray     # (E,)


# ---------------------------------------------------------------------------
# Reward normalization (CleanRL `NormalizeReward` wrapper, numpy version).
# ---------------------------------------------------------------------------
class RunningMeanStd:
  """Welford-style online mean / variance tracker.

  Matches OpenAI Baselines' `RunningMeanStd` and CleanRL's vector-env
  reward normalizer.  Maintains mean, variance, and sample count; `update`
  accepts a batch and folds it in via the parallel-algorithm formula.
  """

  def __init__(self, shape=(), epsilon: float = 1e-4):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = float(epsilon)

  def update(self, x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    batch_mean = x.mean(axis=0)
    batch_var = x.var(axis=0)
    batch_count = x.shape[0]
    delta = batch_mean - self.mean
    tot = self.count + batch_count
    new_mean = self.mean + delta * (batch_count / tot)
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot)
    self.mean = new_mean
    self.var = M2 / tot
    self.count = tot


class ReturnNormalizer:
  """Normalizes a per-env reward stream by the running std of discounted
  returns (NOT by the std of raw rewards).

  This is CleanRL's `NormalizeReward` wrapper, translated to a plain
  object we can call inside the rollout loop.  For each env we keep a
  *running discounted return* G_i that is updated on every step; the
  RunningMeanStd is updated with the current G's, and the *instantaneous*
  reward r_t is divided by sqrt(Var(G) + eps).  On episode boundaries G_i
  is reset to 0.

  Rationale: dividing the raw reward by its own std would shrink the
  signal to O(1) but leave very high variance advantages because the
  discounted *return* — what GAE actually bootstraps — can still be much
  larger.  Scaling by std(G) directly normalizes what the value head has
  to fit, which is what empirically stabilizes PPO with learned rewards.
  """

  def __init__(self, num_envs: int, discount: float, epsilon: float = 1e-8):
    self._rms = RunningMeanStd(shape=())
    self._returns = np.zeros(num_envs, dtype=np.float64)
    self._gamma = float(discount)
    self._eps = float(epsilon)

  def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Rescale a (num_envs,) reward vector and advance internal state.

    Args:
      reward: per-env reward at step t (float32/64).
      done:   per-env termination mask at step t (bool / 0-1).
    Returns:
      Rescaled reward, same shape as input.
    """
    reward = np.asarray(reward, dtype=np.float64)
    done = np.asarray(done).astype(bool)
    self._returns = self._returns * self._gamma + reward
    self._rms.update(self._returns)
    scaled = reward / np.sqrt(self._rms.var + self._eps)
    # Reset per-env running return at episode boundaries so the normalizer
    # tracks the *in-episode* discounted return distribution.
    if done.any():
      self._returns = np.where(done, 0.0, self._returns)
    return scaled.astype(np.float32)

  @property
  def std(self) -> float:
    return float(np.sqrt(self._rms.var + self._eps))


# ---------------------------------------------------------------------------
# In-memory FIFO replay buffer (CRL side only).
# ---------------------------------------------------------------------------
class EpisodeReplay:
  """In-memory episode buffer matching the naive CRL pipeline exactly.

  Replicates the observable behavior of the Reverb-based pipeline used
  by the original SAC-free CRL code (`builder.py::flatten_fn`
  followed by `batch(B) → transpose → unbatch → unbatch` on a Reverb
  Uniform-selector episode table).  After that tf.data chain, each
  individual training sample has the following distribution:

      k  ~ Uniform(stored episodes)
      t  ~ Uniform[0, T_k - 1]
      d  ~ TruncatedGeometric(1 - γ,  range=[1, T_k - t])
      j  = t + d
      obs       = [ s_t       ;  obs_to_goal_2d(s_j) ]
      action    =   a_t
      next_obs  = [ s_{t+1}   ;  obs_to_goal_2d(s_j) ]

  with the B samples in a training minibatch drawn i.i.d. (so, with
  high probability, each InfoNCE batch has B distinct source episodes —
  the property that makes the negatives semantically diverse).

  The `tf.roll` shift inside `flatten_fn` is irrelevant here: it only
  decorrelates *contiguous* transitions within an episode, and in this
  equivalent reformulation we never emit two transitions from the same
  episode in a single batch call (except by coincidence, same as the
  naive pipeline).

  Storage: list of episodes, each {'obs': (T+1, D), 'action': (T, A)}.
  FIFO eviction is at the episode granularity once total stored
  transitions exceed `capacity`.
  """

  def __init__(self, capacity: int, obs_dim: int, discount: float,
               start_index: int, end_index: int):
    self._cap = capacity
    self._obs_dim = obs_dim               # state slice size
    self._discount = float(discount)
    self._start_index = start_index
    self._end_index = end_index
    self._episodes: list = []             # list[{'obs': (T+1, D), 'action': (T, A)}]
    self._ep_lens: list = []              # parallel list of T (== len(action))
    self._total_transitions = 0
    # Pre-compute log(γ) once; used per sample call for the truncated
    # geometric.  γ must be in (0, 1) for the formula to be well-defined.
    assert 0.0 < self._discount < 1.0, (
        f'discount must be in (0, 1), got {self._discount}')
    self._log_gamma = float(np.log(self._discount))

  @property
  def size(self) -> int:
    """Number of stored (s, a) transitions across all retained episodes."""
    return self._total_transitions

  @property
  def num_episodes(self) -> int:
    return len(self._episodes)

  def add_episode(self, obs: np.ndarray, action: np.ndarray):
    """Store one complete episode.

    Args:
      obs:    (T+1, obs_dim_total) — includes the terminal observation.
      action: (T,   act_dim)
    Episodes shorter than 1 transition are rejected.
    """
    T = action.shape[0]
    assert obs.shape[0] == T + 1, f'obs len {obs.shape[0]} != action len {T} + 1'
    if T < 1:
      return
    self._episodes.append({
        'obs': np.asarray(obs, dtype=np.float32),
        'action': np.asarray(action, dtype=np.float32),
    })
    self._ep_lens.append(T)
    self._total_transitions += T
    # FIFO eviction at the episode granularity.
    while self._total_transitions > self._cap and len(self._episodes) > 1:
      self._episodes.pop(0)
      self._total_transitions -= self._ep_lens.pop(0)

  # ---------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------
  def _obs_to_goal(self, states: np.ndarray) -> np.ndarray:
    """Equivalent to `contrastive/utils.py::obs_to_goal_2d`."""
    if self._end_index == -1:
      return states[:, self._start_index:]
    return states[:, self._start_index:self._end_index]

  # ---------------------------------------------------------------------
  # Sampling — matches the naive CRL pipeline exactly
  # ---------------------------------------------------------------------
  def sample(self, batch_size: int,
             rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Draw B i.i.d. transitions `[s_t; goal_j]` matching naive CRL.

    See class docstring for the per-sample distribution; this
    implementation vectorizes index sampling across the batch and uses
    a tiny Python loop only for the final gather into arrays (episodes
    have different lengths so a single numpy take isn't possible
    without padding).  The loop is O(B), not O(B·T).

    Returns a dict with the exact keys consumed by `critic_loss`:
      obs        : (B, obs_dim_total)   [ s_t       ; obs_to_goal(s_j) ]
      action     : (B, act_dim)
      next_obs   : (B, obs_dim_total)   [ s_{t+1}   ; obs_to_goal(s_j) ]
    """
    assert self.size > 0, 'Cannot sample from an empty buffer.'
    B = int(batch_size)
    num_eps = len(self._episodes)
    ep_lens = np.asarray(self._ep_lens, dtype=np.int64)      # (K,)

    # (1) Episode index: Uniform over stored episodes (matches reverb Uniform).
    ep_ids = rng.integers(0, num_eps, size=B)
    lens = ep_lens[ep_ids]                                   # (B,)  == T_k

    # (2) Starting timestep t: Uniform[0, T_k - 1].  Flatten_fn's valid t
    #     range is exactly this — it slices obs[:-1] (last obs has no action).
    u_t = rng.random(B)
    t = np.floor(u_t * lens).astype(np.int64)
    t = np.minimum(t, lens - 1)                              # numerical guard

    # (3) Future offset d ~ TruncatedGeometric(1-γ, [1, T_k - t]).
    #     Continuous-time derivation — for U ~ Uniform(0, 1 - γ^max_d),
    #       d = 1 + floor( log(1 - U) / log(γ) )
    #     distributes U over the truncated geometric support exactly.
    #     Matches `flatten_fn`'s categorical with probs ∝ γ^(j-t) normalized
    #     to the in-episode future states only.
    max_d = lens - t                                         # (B,)  ≥ 1
    trunc_cdf = 1.0 - np.power(self._discount,
                               max_d.astype(np.float64))     # CDF at max_d
    u_d = rng.random(B) * trunc_cdf                          # U ∈ [0, trunc_cdf)
    # `log1p(-u_d)` is `log(1 - u_d)` but numerically stable near 0.
    d = 1 + np.floor(np.log1p(-u_d) / self._log_gamma).astype(np.int64)
    d = np.clip(d, 1, max_d)                                 # guard rounding

    j = t + d                                                # (B,)  ∈ [t+1, T_k]

    # (4) Gather.  Per-sample index into the i-th chosen episode.
    obs_out = np.empty((B, 2 * self._obs_dim), dtype=np.float32)
    # Note: final obs dim = self._obs_dim (state) + goal_dim; for the
    # standard configurations goal_dim == self._obs_dim, hence `2*obs_dim`.
    # For other envs we resize on the first fill.
    act_dim = self._episodes[0]['action'].shape[1]
    act_out = np.empty((B, act_dim), dtype=np.float32)
    # We don't know goal_dim until we call _obs_to_goal on the first sample.
    next_obs_out = None
    goal_dim = None
    for i in range(B):
      ep = self._episodes[int(ep_ids[i])]
      full_obs = ep['obs']                      # (T+1, D_full)
      act = ep['action']                        # (T, A)
      ti = int(t[i]);  ji = int(j[i])
      s_t       = full_obs[ti,     :self._obs_dim]
      s_tp1     = full_obs[ti + 1, :self._obs_dim]
      s_j       = full_obs[ji,     :self._obs_dim]
      goal      = self._obs_to_goal(s_j[None])[0]
      if goal_dim is None:
        goal_dim = goal.shape[0]
        obs_out = np.empty((B, self._obs_dim + goal_dim), dtype=np.float32)
        next_obs_out = np.empty_like(obs_out)
      obs_out[i, :self._obs_dim] = s_t
      obs_out[i,  self._obs_dim:] = goal
      next_obs_out[i, :self._obs_dim] = s_tp1
      next_obs_out[i,  self._obs_dim:] = goal
      act_out[i] = act[ti]

    return {
        'obs': obs_out,
        'action': act_out,
        'next_obs': next_obs_out,
    }

  def sample_with_uniform_negatives(
      self, batch_size: int, rng: np.random.Generator,
      goal_low: np.ndarray, goal_high: np.ndarray,
  ) -> Dict[str, np.ndarray]:
    """Like sample(), but the second half of the batch has goals replaced
    by goals sampled uniformly from [goal_low, goal_high].

    This gives a 50/50 mix: half the in-batch negatives seen by the InfoNCE
    loss come from the replay future-state distribution, half from the
    uniform goal distribution — without changing the loss function.
    """
    batch = self.sample(batch_size, rng)
    half = batch_size // 2
    goal_dim = batch['obs'].shape[1] - self._obs_dim
    uniform_goals = rng.uniform(
        goal_low, goal_high, size=(half, goal_dim)).astype(np.float32)
    obs = batch['obs'].copy()
    next_obs = batch['next_obs'].copy()
    obs[half:, self._obs_dim:] = uniform_goals
    next_obs[half:, self._obs_dim:] = uniform_goals
    return {'obs': obs, 'action': batch['action'], 'next_obs': next_obs}


# ---------------------------------------------------------------------------
# Core factories.  Each returns a jitted function plus any needed metadata.
# ---------------------------------------------------------------------------
def make_reward_fn(networks: contrastive_networks.ContrastiveNetworks):
  """Returns r(q_params, obs, action) = φ(s,a) · ψ(g).

  The q_network's apply() returns `(critic_val, sa_repr, g_repr)`; we
  ignore the outer-product critic_val and just take the per-sample dot
  product `Σ_k sa_repr[b,k] * g_repr[b,k]` — that is φ(s,a)·ψ(g).
  Note: PPO does not use twin_q (so no min() needed here).
  """
  @jax.jit
  def reward_fn(q_params: networks_lib.Params,
                obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    _, sa_repr, g_repr = networks.q_network.apply(q_params, obs, action)
    return jnp.sum(sa_repr * g_repr, axis=-1)  # (B,)
  return reward_fn


def make_value_fn(networks: contrastive_networks.ContrastiveNetworks):
  """Returns V(value_params, obs) — scalar value per sample."""
  @jax.jit
  def value_fn(value_params: networks_lib.Params,
               obs: jnp.ndarray) -> jnp.ndarray:
    return networks.value_network.apply(value_params, obs)  # (B,)
  return value_fn


def make_gae_fn(config: contrastive_config.ContrastiveConfig):
  """Returns a jitted GAE advantage + return computation.

  Matches CleanRL's `ppo_continuous_action.py` computation:

    δ_t         = r_t + γ·V(s_{t+1})·(1−done_{t+1}) − V(s_t)
    A_t         = δ_t + γ·λ·(1−done_{t+1})·A_{t+1}
    returns_t   = A_t + V(s_t)

  Inputs are rollout tensors shaped (T, E).  The extra `next_value` and
  `next_done` correspond to the state after the last collected step
  (used for bootstrapping A_{T-1}).
  """
  ppo_gamma = (float(config.ppo_discount)
               if float(getattr(config, 'ppo_discount', -1.0)) > 0.0
               else float(config.discount))
  gae_lambda = float(config.ppo_gae_lambda)

  @jax.jit
  def gae_fn(rewards: jnp.ndarray, values: jnp.ndarray,
             dones: jnp.ndarray, next_value: jnp.ndarray,
             next_done: jnp.ndarray
             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Build per-timestep "next" tensors: for t in [0, T-1) we use
    # values[t+1] / dones[t+1]; for t = T-1 we use next_value / next_done.
    next_vals_seq = jnp.concatenate([values[1:], next_value[None]], axis=0)  # (T, E)
    next_dones_seq = jnp.concatenate([dones[1:], next_done[None]], axis=0)   # (T, E)
    next_nonterm = 1.0 - next_dones_seq.astype(jnp.float32)

    deltas = rewards + ppo_gamma * next_vals_seq * next_nonterm - values  # (T, E)

    def scan_fn(last_gae, inputs):
      delta_t, nnt_t = inputs
      new_gae = delta_t + ppo_gamma * gae_lambda * nnt_t * last_gae
      return new_gae, new_gae

    init = jnp.zeros(rewards.shape[1], dtype=rewards.dtype)
    _, adv_rev = jax.lax.scan(
        scan_fn, init,
        (deltas[::-1], next_nonterm[::-1]))
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns
  return gae_fn


def make_ppo_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
    ppo_optimizer: optax.GradientTransformation,
):
  """Returns a jitted one-minibatch PPO update.

  The update operates on a pytree of trainable params keyed as:
      {'policy': policy_params, 'value': value_params}

  Combined loss (CleanRL-style):
      L  =  pg_loss  -  ent_coef * entropy  +  vf_coef * v_loss

  * pg_loss:   clipped surrogate,  max(-adv*ratio, -adv*clip(ratio))
  * v_loss:    clipped MSE (optional) against `returns`
  * entropy:   single-sample MC estimate  -log π(ã|s) with ã ~ π(·|s)

  Returns a function `update(params, opt_state, batch, key)` that does
  one SGD step and returns (new_params, new_opt_state, metrics_dict).
  """
  clip_coef = float(config.ppo_clip_coef)
  vf_coef = float(config.ppo_vf_coef)
  ent_coef = float(config.ppo_ent_coef)
  clip_vloss = bool(config.ppo_clip_vloss)
  norm_adv = bool(config.ppo_norm_adv)

  def ppo_loss(params, batch, key):
    # ---- policy forward ----
    dist = networks.policy_network.apply(params['policy'], batch['obs'])
    new_logprob = networks.log_prob(dist, batch['actions'])         # (B,)
    # MC-estimate entropy: H(π) ≈ -log π(ã|s), ã ~ π.
    fresh_action = networks.sample(dist, key)
    entropy_est = -networks.log_prob(dist, fresh_action)            # (B,)

    # ---- value forward ----
    new_value = networks.value_network.apply(params['value'], batch['obs'])  # (B,)

    # ---- advantage normalization (per-minibatch) ----
    adv = batch['advantages']
    if norm_adv:
      adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ---- clipped surrogate ----
    logratio = new_logprob - batch['old_logprobs']
    ratio = jnp.exp(logratio)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

    # ---- value loss ----
    if clip_vloss:
      v_unclipped = (new_value - batch['returns']) ** 2
      v_clipped_pred = batch['old_values'] + jnp.clip(
          new_value - batch['old_values'], -clip_coef, clip_coef)
      v_clipped = (v_clipped_pred - batch['returns']) ** 2
      v_loss = 0.5 * jnp.mean(jnp.maximum(v_unclipped, v_clipped))
    else:
      v_loss = 0.5 * jnp.mean((new_value - batch['returns']) ** 2)

    # ---- entropy bonus ----
    entropy_mean = jnp.mean(entropy_est)
    # Term as it enters the minimized objective: L includes -coef * H.
    entropy_loss_term = -ent_coef * entropy_mean

    # ---- combined loss ----
    total = pg_loss - ent_coef * entropy_mean + vf_coef * v_loss

    # ---- diagnostics (stop_gradient is implicit for metrics) ----
    approx_kl = jnp.mean((ratio - 1.0) - logratio)    # http://joschu.net/blog/kl-approx.html
    old_approx_kl = jnp.mean(-logratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))

    metrics = {
        'ppo_total_loss': total,
        'pg_loss': pg_loss,
        'v_loss': v_loss,
        'entropy': entropy_mean,
        'entropy_loss': entropy_loss_term,
        'approx_kl': approx_kl,
        'old_approx_kl': old_approx_kl,
        'clipfrac': clipfrac,
        'ratio_mean': jnp.mean(ratio),
    }
    return total, metrics

  grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

  @jax.jit
  def update(params, opt_state, batch, key):
    (_, metrics), grads = grad_fn(params, batch, key)
    updates, new_opt_state = ppo_optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics

  return update


def make_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
):
  """Returns a jitted CRL critic update over one replay batch.

  PPO CRL uses a single loss: in-batch InfoNCE (softmax cross-entropy on the
  diagonal) plus ``0.01 * logsumexp(logits, axis=1)^2``, matching the CPC path
  in ``learning.py``.  Supports ``(B, B)`` or twin ``(B, B, 2)`` logits.

  Args:
    networks: ContrastiveNetworks (``q_network`` only is used here).
    q_optimizer: Adam (or other) transform for Q / representation params.

  Returns:
    ``update(q_params, q_optimizer_state, batch, key)``
    → ``(new_q_params, new_q_optimizer_state, metrics)``
  """
  _logsumexp_penalty_coef = 0.01

  def critic_loss(q_params, batch, key):
    del key
    obs = batch['obs']
    action = batch['action']
    batch_size = obs.shape[0]
    labels = jnp.eye(batch_size)

    logits, _, _ = networks.q_network.apply(q_params, obs, action)

    def loss_fn(_logits, _labels):
      return (
          optax.softmax_cross_entropy(logits=_logits, labels=_labels)
          + _logsumexp_penalty_coef * jax.nn.logsumexp(_logits, axis=1) ** 2)

    if logits.ndim == 3:
      loss = jax.vmap(loss_fn, in_axes=(2, None), out_axes=-1)(logits, labels)
      loss = jnp.mean(loss, axis=-1)
    else:
      loss = loss_fn(logits, labels)

    loss = jnp.mean(loss)
    train_logits = logits

    if train_logits.ndim == 2:
      narrow_logits = train_logits[:, :batch_size]
    else:
      narrow_logits = train_logits[:, :batch_size, :]

    if narrow_logits.ndim == 3:
      logits_flat = jnp.mean(narrow_logits, axis=-1)
    else:
      logits_flat = narrow_logits

    correct = (jnp.argmax(logits_flat, axis=1) == jnp.argmax(labels, axis=1))
    logits_pos = jnp.sum(logits_flat * labels) / jnp.sum(labels)
    logits_neg = jnp.sum(logits_flat * (1 - labels)) / jnp.sum(1 - labels)
    if train_logits.ndim == 3:
      logsumexp_val = jax.nn.logsumexp(train_logits[:, :, 0], axis=1) ** 2
    else:
      logsumexp_val = jax.nn.logsumexp(train_logits, axis=1) ** 2

    metrics = {
        'crl_loss': loss,
        'binary_accuracy': jnp.mean((logits_flat > 0) == labels),
        'categorical_accuracy': jnp.mean(correct),
        'logits_pos': logits_pos,
        'logits_neg': logits_neg,
        'logsumexp': logsumexp_val.mean(),
    }
    return loss, metrics

  grad_fn = jax.value_and_grad(critic_loss, has_aux=True)

  def update(q_params, q_optimizer_state, batch, key):
    (_, metrics), grads = grad_fn(q_params, batch, key)
    grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))))
    loss_finite = jnp.isfinite(metrics['crl_loss'])
    do_update = jnp.logical_and(grads_finite, loss_finite)

    def _apply(_):
      updates, new_opt_state = q_optimizer.update(
          grads, q_optimizer_state, q_params)
      new_q_params = optax.apply_updates(q_params, updates)
      return new_q_params, new_opt_state

    def _skip(_):
      return q_params, q_optimizer_state

    new_q_params, new_opt_state = jax.lax.cond(
        do_update, _apply, _skip, operand=None)
    metrics = dict(metrics)
    metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
    return new_q_params, new_opt_state, metrics

  return jax.jit(update)


# ---------------------------------------------------------------------------
# Rollout collection (env stepping — runs in host/numpy, not jitted).
# ---------------------------------------------------------------------------
class VecEnv:
  """A tiny synchronous wrapper around N acme/dm_env environments.

  Uses dm_env's step/reset interface (not gym's) to stay compatible with
  contrastive_utils.make_environment.  Exposes a CleanRL-like API:

      obs = vec_env.reset()                                    # (E, obs_dim_total)
      next_obs, env_rew, dones, terminal_obs = vec_env.step(actions)

  Envs auto-reset on episode end (matching gymnasium semantics), so
  `next_obs[i]` is the fresh initial observation of the new episode when
  `dones[i]` is True.  `terminal_obs[i]` is the true final observation
  of the just-finished episode — needed by the CRL episode buffer so it
  can store complete (T+1)-length trajectories.  When `dones[i]` is
  False, `terminal_obs[i]` equals `next_obs[i]`.

  The env reward is ignored by PPO (we use φ·ψ instead) but is returned
  so the caller can also log ground-truth return for sanity.
  """

  def __init__(self, env_factory: Callable, num_envs: int, seed: int):
    self._envs = [env_factory(seed + i) for i in range(num_envs)]
    self._num_envs = num_envs
    self._obs_spec = self._envs[0].observation_spec()
    self._action_spec = self._envs[0].action_spec()

  def reset(self) -> np.ndarray:
    obs = np.stack([e.reset().observation for e in self._envs], axis=0)
    return obs.astype(np.float32)

  def step(
      self, actions: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_dtype = self._action_spec.dtype
    shape = self._obs_spec.shape
    next_obs = np.zeros((self._num_envs,) + shape, dtype=np.float32)
    terminal_obs = np.zeros((self._num_envs,) + shape, dtype=np.float32)
    env_rewards = np.zeros(self._num_envs, dtype=np.float32)
    dones = np.zeros(self._num_envs, dtype=bool)
    for i, env in enumerate(self._envs):
      a = np.asarray(actions[i], dtype=np.float32)
      a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
      if hasattr(self._action_spec, 'minimum') and hasattr(self._action_spec, 'maximum'):
        a = np.clip(a, self._action_spec.minimum, self._action_spec.maximum)
      ts = env.step(a.astype(action_dtype))
      env_rewards[i] = 0.0 if ts.reward is None else float(ts.reward)
      terminal_obs[i] = ts.observation
      if ts.last():
        dones[i] = True
        # Auto-reset: next_obs is the start of the next episode.
        next_obs[i] = env.reset().observation
      else:
        next_obs[i] = ts.observation
    return next_obs, env_rewards, dones, terminal_obs

  @property
  def num_envs(self) -> int:
    return self._num_envs

  @property
  def observation_shape(self) -> Tuple[int, ...]:
    return tuple(self._obs_spec.shape)

  @property
  def action_shape(self) -> Tuple[int, ...]:
    return tuple(self._action_spec.shape)


# ---------------------------------------------------------------------------
# Checkpoint helpers.
# ---------------------------------------------------------------------------
def _save_checkpoint(path: str,
                     policy_params, value_params, q_params,
                     ppo_opt_state, q_opt_state,
                     iteration: int, global_step: int, key):
  """Write a pickle checkpoint atomically (write to tmp → rename)."""
  import pickle as _pkl
  import os as _os

  ckpt = {
      'policy_params':       policy_params,
      'value_params':        value_params,
      'q_params':            q_params,
      'ppo_optimizer_state': ppo_opt_state,
      'q_optimizer_state':   q_opt_state,
      'iteration':           int(iteration),
      'global_step':         int(global_step),
      'key':                 key,
  }
  tmp_path = path + '.tmp'
  with open(tmp_path, 'wb') as fh:
    _pkl.dump(ckpt, fh, protocol=_pkl.HIGHEST_PROTOCOL)
  _os.replace(tmp_path, path)


def _prune_old_checkpoints(ckpt_dir: str, keep_last: int):
  """Delete all but the `keep_last` most recent ckpt_iter_*.pkl files."""
  import os as _os
  if keep_last <= 0:
    return
  files = [f for f in _os.listdir(ckpt_dir)
           if f.startswith('ckpt_iter_') and f.endswith('.pkl')]

  def _iter_of(fname):
    try:
      return int(fname[len('ckpt_iter_'):-len('.pkl')])
    except ValueError:
      return -1

  files.sort(key=_iter_of)
  to_remove = files[:-keep_last] if len(files) > keep_last else []
  for f in to_remove:
    try:
      _os.remove(_os.path.join(ckpt_dir, f))
    except OSError:
      pass


def load_checkpoint(path: str):
  """Load a checkpoint file produced by `_save_checkpoint`.

  Returns the raw pickled dict.  Caller is responsible for plugging the
  params back into networks / optimizers.
  """
  import pickle as _pkl
  with open(path, 'rb') as fh:
    return _pkl.load(fh)


def _truncate_csv_to_iteration(csv_path: str, max_iteration: int) -> None:
  """Rewrite a log CSV keeping only rows with learner_steps <= max_iteration.

  Called on resume to remove log entries written after the last checkpoint
  (which would otherwise create non-monotonic step sequences in the CSV).
  """
  import csv as _csv
  if not os.path.exists(csv_path):
    return
  try:
    with open(csv_path, 'r', newline='') as fh:
      reader = _csv.DictReader(fh)
      fieldnames = reader.fieldnames
      if not fieldnames:
        return
      rows = [r for r in reader
              if int(float(r.get('learner_steps', max_iteration + 1)))
              <= max_iteration]
    with open(csv_path, 'w', newline='') as fh:
      writer = _csv.DictWriter(fh, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)
    print(f'[ppo] truncated {csv_path} to learner_steps<={max_iteration} '
          f'({len(rows)} rows kept)')
  except Exception as exc:
    print(f'[ppo] warning: could not truncate {csv_path}: {exc}')


# ---------------------------------------------------------------------------
# Top-level training loop.
# ---------------------------------------------------------------------------
def run_ppo_training(
    config: contrastive_config.ContrastiveConfig,
    env_factory: Callable,
    eval_env_factory: Callable,
    network_factory: Callable,
    logger_fn: Callable,
    total_steps: int,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
):
  """Top-level PPO-on-φ·ψ training loop.

  Mirrors CleanRL's ppo_continuous_action.py main block:
    for iteration in 1..num_iterations:
      collect rollout of T steps × E envs   (logprob, value, reward = φ·ψ)
      compute GAE advantages / returns
      for epoch in 1..num_epochs:
        for minibatch: PPO update on flat (T·E) batch
      for _ in 1..ppo_crl_steps_per_iter:   # CRL off-policy
        sample future-goal batch from EpisodeReplay, CRL SGD step
      log metrics; periodically evaluate

  The CRL side is InfoNCE + logsumexp penalty on replay batches only.
  The PPO side follows CleanRL.
  """
  # ---- build networks from env spec -------------------------------------
  from acme import specs as _specs
  import contrastive.utils as _cu

  probe_env = env_factory(seed)
  spec = _specs.make_environment_spec(probe_env)
  networks = network_factory(spec=spec)
  del probe_env

  # ---- vec env ----------------------------------------------------------
  vec_env = VecEnv(env_factory, config.ppo_num_envs, seed=seed * 31)
  E = vec_env.num_envs
  obs_shape = vec_env.observation_shape
  act_shape = vec_env.action_shape
  T = int(config.ppo_rollout_length)
  batch_per_iter = T * E
  mb_size = batch_per_iter // int(config.ppo_num_minibatches)
  assert mb_size * int(config.ppo_num_minibatches) == batch_per_iter, (
      'batch_per_iter must divide evenly into ppo_num_minibatches')
  num_iterations = int(total_steps) // (T * E)

  # ---- init params ------------------------------------------------------
  key = jax.random.PRNGKey(seed)
  k_pol, k_val, k_q, key = jax.random.split(key, 4)
  policy_params = networks.policy_network.init(k_pol)
  value_params = networks.value_network.init(k_val)
  q_params = networks.q_network.init(k_q)
  ppo_params = {'policy': policy_params, 'value': value_params}

  # ---- optimizers -------------------------------------------------------
  if config.ppo_anneal_lr:
    total_ppo_updates = (
      num_iterations
      * int(config.ppo_num_epochs)
      * int(config.ppo_num_minibatches)
    )

    lr_schedule = optax.linear_schedule(
      init_value=float(config.learning_rate),
      end_value=0.0,
      transition_steps=max(1, total_ppo_updates),
    )
    # A single Adam with schedule + grad-clip (CleanRL-style).
    ppo_optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(lambda count: -lr_schedule(count)))
  else:
    ppo_optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
        optax.adam(float(config.learning_rate), eps=1e-5))
  ppo_opt_state = ppo_optimizer.init(ppo_params)

  q_optimizer = optax.adam(float(config.learning_rate))
  q_opt_state = q_optimizer.init(q_params)

  # ---- resume from checkpoint if one exists -----------------------------
  start_iteration = 0
  global_step = 0
  ppo_sgd_step = 0
  if checkpoint_dir is not None:
    _latest = os.path.join(checkpoint_dir, 'latest.pkl')
    if os.path.exists(_latest):
      _ckpt = load_checkpoint(_latest)
      policy_params = _ckpt['policy_params']
      value_params  = _ckpt['value_params']
      q_params      = _ckpt['q_params']
      ppo_opt_state = _ckpt['ppo_optimizer_state']
      q_opt_state   = _ckpt['q_optimizer_state']
      ppo_params    = {'policy': policy_params, 'value': value_params}
      start_iteration = int(_ckpt['iteration']) + 1
      global_step     = int(_ckpt['global_step'])
      key             = _ckpt['key']
      ppo_sgd_step    = (start_iteration
                         * int(config.ppo_num_epochs)
                         * int(config.ppo_num_minibatches))
      print(f'[ppo] resumed from checkpoint: '
            f'start_iteration={start_iteration}, global_step={global_step}')
      # Truncate CSV logs to remove any entries written after the checkpoint
      # (can happen if training ran past the last checkpoint before preemption).
      _run_dir = os.path.dirname(checkpoint_dir)
      for _label in ('learner', 'eval'):
        _csv_path = os.path.join(_run_dir, 'logs', _label, 'logs.csv')
        _truncate_csv_to_iteration(_csv_path, int(_ckpt['iteration']))

  # ---- jitted helpers ---------------------------------------------------
  reward_fn = make_reward_fn(networks)
  gae_fn = make_gae_fn(config)
  ppo_update = make_ppo_update_fn(networks, config, ppo_optimizer)

  crl_update = make_crl_update_fn(networks, q_optimizer)

  @jax.jit
  def act_and_value(policy_p, value_p, obs, rng):
    dist = networks.policy_network.apply(policy_p, obs)
    action = networks.sample(dist, rng)
    logprob = networks.log_prob(dist, action)
    value = networks.value_network.apply(value_p, obs)
    return action, logprob, value

  @jax.jit
  def value_only(value_p, obs):
    return networks.value_network.apply(value_p, obs)

  @jax.jit
  def greedy_action(policy_p, obs):
    # Deterministic policy mean for evaluation.
    dist = networks.policy_network.apply(policy_p, obs)
    return networks.sample(dist, jax.random.PRNGKey(0))  # sample still; we log both below

  # ---- uniform-sampling goal bounds (extracted once from env spec) ------
  uniform_sampling = bool(getattr(config, 'uniform_sampling', False))
  goal_low = goal_high = None
  if uniform_sampling:
    _obs_min = np.asarray(spec.observations.minimum, dtype=np.float32)
    _obs_max = np.asarray(spec.observations.maximum, dtype=np.float32)
    _end = int(config.end_index) if int(config.end_index) != -1 else int(config.obs_dim)
    goal_low = _obs_min[int(config.start_index):_end]
    goal_high = _obs_max[int(config.start_index):_end]
    print(f'[ppo] uniform_sampling: goal_low={goal_low}, goal_high={goal_high}')

  # ---- replay buffer (episodes) -----------------------------------------
  replay = EpisodeReplay(
      capacity=int(config.max_replay_size),
      obs_dim=int(config.obs_dim),
      discount=float(config.discount),
      start_index=int(config.start_index),
      end_index=int(config.end_index))
  np_rng = np.random.default_rng(seed + 12345)

  # ---- per-env episode buffers (for flushing complete trajectories) -----
  ep_obs: list = [[] for _ in range(E)]
  ep_act: list = [[] for _ in range(E)]
  ep_return = np.zeros(E, dtype=np.float32)
  ep_len = np.zeros(E, dtype=np.int32)
  recent_returns: list = []
  recent_lengths: list = []

  obs = vec_env.reset()
  next_done = np.zeros(E, dtype=np.float32)
  for i in range(E):
    ep_obs[i].append(obs[i].copy())

  # ---- loggers ----------------------------------------------------------
  learner_logger = logger_fn(label='learner')
  eval_logger = logger_fn(label='eval')

  # Persistent eval observers (mirrors Acme's evaluator loop).  Keeping
  # them alive across iterations is what lets `success_1000` and
  # `*_dist_{10,100,1000}` smooth over eval history.
  # RiverSwim: success = visited goal cell at least once (not env +1 reward).
  _env = str(getattr(config, 'env_name', '') or '').lower()
  if _env == 'riverswim':
    eval_success_obs = _cu.RiverSwimGoalVisitSuccessObserver(
        obs_dim=int(config.obs_dim))
  else:
    eval_success_obs = _cu.SuccessObserver()
  eval_dist_obs = _cu.DistanceObserver(
      obs_dim=int(config.obs_dim),
      start_index=int(config.start_index),
      end_index=int(config.end_index))

  # ---- rollout storage (reused each iteration) --------------------------
  roll_obs = np.zeros((T, E) + obs_shape, dtype=np.float32)
  roll_acts = np.zeros((T, E) + act_shape, dtype=np.float32)
  roll_logp = np.zeros((T, E), dtype=np.float32)
  roll_rew = np.zeros((T, E), dtype=np.float32)            # reps reward (possibly normalized)
  roll_rew_raw = np.zeros((T, E), dtype=np.float32)        # reps reward (pre-normalization, log only)
  roll_env_rew = np.zeros((T, E), dtype=np.float32)        # gt reward (log only)
  roll_dones = np.zeros((T, E), dtype=np.float32)
  roll_vals = np.zeros((T, E), dtype=np.float32)

  # ---- reward normalizer (CleanRL NormalizeReward) ----------------------
  # Normalizes the reps-based reward by the running std of discounted
  # returns.  Critical for PPO with learned rewards — raw φ·ψ values can
  # be O(10) and non-stationary, leading to unbounded advantages and
  # policy collapse within a handful of updates.
  norm_reward = bool(getattr(config, 'ppo_norm_reward', True))
  ppo_gamma = (float(config.ppo_discount)
               if float(getattr(config, 'ppo_discount', -1.0)) > 0.0
               else float(config.discount))
  reward_normalizer = (
      ReturnNormalizer(num_envs=E, discount=ppo_gamma)
      if norm_reward else None)

  # ---- checkpointing ----------------------------------------------------
  ckpt_interval = int(getattr(config, 'ppo_checkpoint_interval', 0))
  ckpt_keep_last = int(getattr(config, 'ppo_checkpoint_keep_last', 10))
  if ckpt_interval > 0 and checkpoint_dir is not None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'[ppo] checkpoints → {checkpoint_dir} '
          f'(every {ckpt_interval} iters, keep last {ckpt_keep_last})')

  start_time = time.time()
  # global_step, ppo_sgd_step, start_iteration set above (0 for fresh runs,
  # restored from checkpoint on resume).

  for iteration in range(start_iteration, num_iterations):
    # =================================================================
    # 1. Rollout (on-policy, CleanRL convention)
    # =================================================================
    for t in range(T):
      roll_obs[t] = obs
      roll_dones[t] = next_done

      key, k_act = jax.random.split(key)
      action_j, logprob_j, value_j = act_and_value(
          ppo_params['policy'], ppo_params['value'],
          jnp.asarray(obs), k_act)
      action = np.asarray(action_j)
      roll_acts[t] = action
      roll_logp[t] = np.asarray(logprob_j)
      roll_vals[t] = np.asarray(value_j)

      # reps-based reward, computed BEFORE env step (frozen q_params).
      rep_rew = reward_fn(q_params, jnp.asarray(obs), action_j)
      rep_rew_np = np.asarray(rep_rew)
      roll_rew_raw[t] = rep_rew_np

      next_obs, env_rew, dones, terminal_obs = vec_env.step(action)
      roll_env_rew[t] = env_rew

      # Scale reps reward by the running std of discounted returns.
      # Applied AFTER the env step so `dones` is available to reset the
      # running-return tracker at episode boundaries.
      if reward_normalizer is not None:
        roll_rew[t] = reward_normalizer(rep_rew_np, dones)
      else:
        roll_rew[t] = rep_rew_np

      # Episode flushing / per-env accounting.
      for i in range(E):
        ep_act[i].append(action[i].copy())
        ep_return[i] += float(env_rew[i])
        ep_len[i] += 1
        if dones[i]:
          ep_obs[i].append(terminal_obs[i].copy())
          try:
            replay.add_episode(
                np.stack(ep_obs[i], axis=0),
                np.stack(ep_act[i], axis=0))
          except AssertionError:
            pass  # degenerate len-0 episodes; skip
          ep_obs[i] = [next_obs[i].copy()]  # auto-reset state seeds next ep
          ep_act[i] = []
          recent_returns.append(float(ep_return[i]))
          recent_lengths.append(int(ep_len[i]))
          ep_return[i] = 0.0
          ep_len[i] = 0
          if len(recent_returns) > 100:
            recent_returns.pop(0)
            recent_lengths.pop(0)
        else:
          ep_obs[i].append(next_obs[i].copy())

      obs = next_obs
      next_done = dones.astype(np.float32)
      global_step += E

    # =================================================================
    # 2. GAE advantages / returns
    # =================================================================
    next_val = np.asarray(value_only(ppo_params['value'], jnp.asarray(obs)))
    adv_j, ret_j = gae_fn(
        jnp.asarray(roll_rew), jnp.asarray(roll_vals),
        jnp.asarray(roll_dones),
        jnp.asarray(next_val), jnp.asarray(next_done))
    adv = np.asarray(adv_j)
    ret = np.asarray(ret_j)

    # =================================================================
    # 3. PPO updates (epochs × minibatches over flat T·E batch)
    # =================================================================
    flat_obs = roll_obs.reshape((batch_per_iter,) + obs_shape)
    flat_acts = roll_acts.reshape((batch_per_iter,) + act_shape)
    flat_logp = roll_logp.reshape(batch_per_iter)
    flat_adv = adv.reshape(batch_per_iter)
    flat_ret = ret.reshape(batch_per_iter)
    flat_vals = roll_vals.reshape(batch_per_iter)

    ppo_metrics_agg: Dict[str, list] = {}
    early_stop = False
    for epoch in range(int(config.ppo_num_epochs)):
      perm = np_rng.permutation(batch_per_iter)
      last_kl = None
      for start in range(0, batch_per_iter, mb_size):
        mb = perm[start:start + mb_size]
        batch = {
            'obs':          jnp.asarray(flat_obs[mb]),
            'actions':      jnp.asarray(flat_acts[mb]),
            'old_logprobs': jnp.asarray(flat_logp[mb]),
            'advantages':   jnp.asarray(flat_adv[mb]),
            'returns':      jnp.asarray(flat_ret[mb]),
            'old_values':   jnp.asarray(flat_vals[mb]),
        }
        key, k_mb = jax.random.split(key)
        ppo_params, ppo_opt_state, m = ppo_update(
            ppo_params, ppo_opt_state, batch, k_mb)
        ppo_sgd_step += 1
        last_kl = float(m['approx_kl'])
        for k_, v in m.items():
          ppo_metrics_agg.setdefault(k_, []).append(float(v))
      if (config.ppo_target_kl is not None and last_kl is not None
          and last_kl > float(config.ppo_target_kl)):
        early_stop = True
        break

    pg_vals = ppo_metrics_agg.get('pg_loss', [])
    mean_pg = float(np.mean(pg_vals)) if pg_vals else float('inf')

    # =================================================================
    # 4. CRL updates (off-policy, from replay)
    # =================================================================
    crl_metrics_agg: Dict[str, list] = {}
    if replay.size >= int(config.ppo_min_replay_size):
      for _ in range(int(config.ppo_crl_steps_per_iter)):
        if uniform_sampling:
          crl_batch_np = replay.sample_with_uniform_negatives(
              int(config.batch_size), np_rng, goal_low, goal_high)
        else:
          crl_batch_np = replay.sample(int(config.batch_size), np_rng)
        crl_batch = {k_: jnp.asarray(v) for k_, v in crl_batch_np.items()}
        key, k_crl = jax.random.split(key)
        q_params, q_opt_state, m = crl_update(
            q_params, q_opt_state, crl_batch, k_crl)
        for k_, v in m.items():
          crl_metrics_agg.setdefault(k_, []).append(float(v))

    # =================================================================
    # 5. Logging
    # =================================================================
    elapsed = time.time() - start_time
    log = {
        'iteration':         iteration,
        'learner_steps':     iteration,
        'global_step':       global_step,
        'sps':               global_step / max(1e-6, elapsed),
        'replay_size':       int(replay.size),
        'reward_repr_mean':      float(roll_rew.mean()),
        'reward_repr_raw_mean':  float(roll_rew_raw.mean()),
        'reward_repr_raw_std':   float(roll_rew_raw.std()),
        'reward_return_norm_std': (
            float(reward_normalizer.std) if reward_normalizer is not None
            else float('nan')),
        'reward_env_mean':       float(roll_env_rew.mean()),
        'value_mean':        float(roll_vals.mean()),
        'returns_mean':      float(ret.mean()),
        'advantage_mean':    float(adv.mean()),
        'advantage_std':     float(adv.std()),
        'early_stop_epochs': int(early_stop),
        'ppo/mean_pg_loss': mean_pg,
        # Always emit these keys so the CSV header is fixed at iter 0,
        # even if the first iteration has no completed episodes or
        # skips CRL for lack of replay data.
        'ep_return_mean':    float(np.mean(recent_returns)) if recent_returns else float('nan'),
        'ep_length_mean':    float(np.mean(recent_lengths)) if recent_lengths else float('nan'),
        'crl/crl_loss':           float('nan'),
        'crl/categorical_accuracy': float('nan'),
        'crl/binary_accuracy':    float('nan'),
        'crl/logits_pos':         float('nan'),
        'crl/logits_neg':         float('nan'),
        'crl/logsumexp':          float('nan'),
    }
    for k_, vs in ppo_metrics_agg.items():
      log[f'ppo/{k_}'] = float(np.mean(vs))
    for k_, vs in crl_metrics_agg.items():
      log[f'crl/{k_}'] = float(np.mean(vs))
    if config.ppo_anneal_lr:
      lr_log = float(lr_schedule(max(0, ppo_sgd_step - 1)))
    else:
      lr_log = float(config.learning_rate)
    # Seven fractional digits so CSV / terminal show stable small LRs.
    log['ppo/learning_rate'] = round(lr_log, 7)
    learner_logger.write(log)

    # =================================================================
    # 6. Periodic evaluation  (5 episodes every 10 iters)
    #
    # Uses the same observers the SAC side uses (`SuccessObserver` or
    # `RiverSwimGoalVisitSuccessObserver` + `DistanceObserver` from
    # contrastive/utils.py) so the eval CSV
    # schema matches the kappa_sac runs:
    #   success, success_1000, init_dist, final_dist, delta_dist,
    #   min_dist, *_10, *_100, *_1000, episode_return, episode_length
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
          key, k_eval = jax.random.split(key)
          a, _, _ = act_and_value(
              ppo_params['policy'], ppo_params['value'],
              jnp.asarray(ts.observation)[None], k_eval)
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

      # Average across the 5 eval episodes.  `success_1000` is already
      # a running statistic inside the success observer, so we just take its
      # last value (same as Acme's evaluator loop).
      agg = {
          'iteration':     iteration,
          'learner_steps': iteration,
      }
      for k_ in ep_metrics_list[0].keys():
        agg[k_] = float(np.nanmean([m[k_] for m in ep_metrics_list]))
      eval_logger.write(agg)

    # =================================================================
    # 7. Checkpointing (every `ppo_checkpoint_interval` iterations,
    #    plus a rolling `latest.pkl` and FIFO pruning of older
    #    milestone files).
    # =================================================================
    if (ckpt_interval > 0
        and checkpoint_dir is not None
        and (iteration % ckpt_interval == 0
             or iteration == num_iterations - 1)):
      _save_checkpoint(
          os.path.join(checkpoint_dir, 'latest.pkl'),
          policy_params=ppo_params['policy'],
          value_params=ppo_params['value'],
          q_params=q_params,
          ppo_opt_state=ppo_opt_state,
          q_opt_state=q_opt_state,
          iteration=iteration,
          global_step=global_step,
          key=key)

  # ---- return final state in case the caller wants to checkpoint --------
  return PPOTrainingState(
      policy_params=ppo_params['policy'],
      value_params=ppo_params['value'],
      ppo_optimizer_state=ppo_opt_state,
      q_params=q_params,
      q_optimizer_state=q_opt_state,
      key=key,
  )
