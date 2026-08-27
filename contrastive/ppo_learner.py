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
from contrastive import nf_density as _nf


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
  # Trained running mean/std for observation normalization (None if
  # `ppo_norm_obs=False`); see `ObsNormalizer`. Callers that render a
  # rollout right after training (e.g. `ppo_contrastive.py`'s end-of-run
  # video) need this to feed the policy observations on the same scale it
  # was trained on.
  obs_normalizer: Optional['ObsNormalizer'] = None


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


class ObsNormalizer:
  """Online per-dimension observation normalizer (z-score via `RunningMeanStd`).

  Observations are `concat([state(obs_dim,), goal(goal_dim,)])`. Goal
  column j is the SAME feature slot as state column `start_index + j`
  (that's what `obs_to_goal` slicing means -- the goal is always a literal
  slice of some state), so goal columns are normalized by reusing the
  corresponding STATE column's stats, not their own independently-fit
  stats. This matters because the "goal" a rollout sees (an env-provided
  heuristic, e.g. a chased target position) and the "goal" the CRL loss
  trains on (a literal future-state slice, via HER) are two different
  samples of conceptually the same feature -- sharing stats keeps both on
  the same scale instead of letting them drift apart.

  Running stats are updated ONLY from the state portion of observations:
  the goal segment is a derived/heuristic quantity (not an independently
  sampled state), so it shouldn't skew the estimate of the state's own
  distribution.

  Output is clipped to +/-`clip` after normalizing, matching CleanRL's
  `NormalizeObservation` wrapper -- keeps stray high-variance dims from
  producing huge inputs to the policy/value/CRL encoders early in
  training, before the running stats have settled.
  """

  def __init__(self, obs_dim: int, start_index: int, end_index: int,
               epsilon: float = 1e-4, clip: float = 10.0):
    self.obs_dim = int(obs_dim)
    self.start_index = int(start_index)
    self.end_index = (self.obs_dim if int(end_index) == -1
                      else int(end_index))
    self._rms = RunningMeanStd(shape=(self.obs_dim,), epsilon=epsilon)
    self._clip = float(clip)

  def update(self, state: np.ndarray):
    """Folds a batch of RAW state observations into the running stats.

    Args:
      state: (..., obs_dim) array -- the STATE portion only (not goal).
    """
    flat = np.asarray(state, dtype=np.float64).reshape(-1, self.obs_dim)
    self._rms.update(flat)

  def _mean_std(self):
    return self._rms.mean, np.sqrt(self._rms.var) + 1e-8

  def normalize_state(self, state: np.ndarray, clip=None) -> np.ndarray:
    mean, std = self._mean_std()
    out = (np.asarray(state, dtype=np.float64) - mean) / std
    clip = self._clip if clip is None else float(clip)
    return np.clip(out, -clip, clip).astype(np.float32)

  def normalize_goal(self, goal: np.ndarray) -> np.ndarray:
    mean, std = self._mean_std()
    s, e = self.start_index, self.end_index
    out = (np.asarray(goal, dtype=np.float64) - mean[s:e]) / std[s:e]
    return np.clip(out, -self._clip, self._clip).astype(np.float32)

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    """Normalizes a raw `concat([state, goal])` batch; same shape out."""
    obs = np.asarray(obs, dtype=np.float64)
    state = self.normalize_state(obs[..., :self.obs_dim])
    goal = self.normalize_goal(obs[..., self.obs_dim:])
    return np.concatenate([state, goal], axis=-1)

  def state_dict(self) -> dict:
    return {
        'mean': self._rms.mean.copy(),
        'var': self._rms.var.copy(),
        'count': self._rms.count,
        'obs_dim': self.obs_dim,
        'start_index': self.start_index,
        'end_index': self.end_index,
    }

  def load_state_dict(self, d: dict):
    self._rms.mean = np.asarray(d['mean'], dtype=np.float64)
    self._rms.var = np.asarray(d['var'], dtype=np.float64)
    self._rms.count = float(d['count'])

  @classmethod
  def from_state_dict(cls, d: dict) -> 'ObsNormalizer':
    """Reconstructs a normalizer from a `state_dict()`, no config needed.

    Useful for `ppo_rollout_video.py`, which loads a checkpoint standalone
    (no access to the `ContrastiveConfig` that produced it).
    """
    norm = cls(obs_dim=d['obs_dim'], start_index=d['start_index'],
               end_index=d['end_index'])
    norm.load_state_dict(d)
    return norm


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
      num_negatives: int = -1,
  ) -> Dict[str, np.ndarray]:
    """Like sample(), but adds uniform goals as *extra negatives only*.

    Returns the standard B-sample batch where every pair has a true
    replay-future-state positive on the InfoNCE diagonal, plus an
    'extra_goals' key with K goals sampled uniformly from [goal_low, goal_high].
    K = num_negatives if >= 0, else batch_size // 2.
    The CRL loss treats those extra goals as additional off-diagonal negatives
    for all B anchors — they are never used as positives for any row.
    """
    batch = self.sample(batch_size, rng)
    K = num_negatives if num_negatives >= 0 else batch_size // 2
    goal_dim = batch['obs'].shape[1] - self._obs_dim
    uniform_goals = rng.uniform(
        goal_low, goal_high, size=(K, goal_dim)).astype(np.float32)
    return {
        'obs': batch['obs'],
        'action': batch['action'],
        'next_obs': batch['next_obs'],
        'extra_goals': uniform_goals,
    }


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


def _ema_tree(target, online, tau: float):
  """target <- tau*target + (1-tau)*online, leaf-wise over a pytree."""
  return jax.tree_util.tree_map(
      lambda t, o: tau * t + (1.0 - tau) * o, target, online)


def make_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    _return_raw: bool = False,
):
  """Returns a jitted CRL critic update over one replay batch.

  PPO CRL uses a single loss: in-batch InfoNCE (softmax cross-entropy on the
  diagonal) plus ``0.01 * logsumexp(logits, axis=1)^2``, matching the CPC path
  in ``learning.py``.  Supports ``(B, B)`` or twin ``(B, B, 2)`` logits.

  Args:
    networks: ContrastiveNetworks (``q_network`` only is used here).
    q_optimizer: Adam (or other) transform for Q / representation params.
    _return_raw: if True, also return the un-jitted ``update`` function so
      ``make_scan_crl_update_fn`` can call it once per ``jax.lax.scan`` step
      (jitting the whole scan, not each individual step).

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
    labels_replay = jnp.eye(batch_size)

    # If 'extra_goals' is present, augment the forward pass with K dummy rows
    # whose state/action are zeroed out so only g_encoder sees the uniform
    # goals.  We then take only the first B rows of the (B+K, B+K) logit
    # matrix: each true anchor (row i < B) now has B+K logit columns, with
    # the extra K columns being φ(s_i,a_i)·ψ(uniform_goal_k).  The diagonal
    # remains the true replay-future-state positive; the uniform goals are
    # purely off-diagonal negatives for every anchor row.
    extra_goals = batch.get('extra_goals', None)
    if extra_goals is not None:
      K = extra_goals.shape[0]
      obs_dim_inf = obs.shape[1] - extra_goals.shape[1]
      act_dim = action.shape[1]
      obs_extra = jnp.concatenate(
          [jnp.zeros((K, obs_dim_inf)), extra_goals], axis=1)
      action_extra = jnp.zeros((K, act_dim))
      obs_fwd = jnp.concatenate([obs, obs_extra], axis=0)
      action_fwd = jnp.concatenate([action, action_extra], axis=0)
      logits_full, _, _ = networks.q_network.apply(q_params, obs_fwd, action_fwd)
      if logits_full.ndim == 3:
        logits = logits_full[:batch_size, :, :]   # (B, B+K, 2)
      else:
        logits = logits_full[:batch_size, :]       # (B, B+K)
      labels = jnp.concatenate(
          [labels_replay, jnp.zeros((batch_size, K))], axis=1)
    else:
      logits, _, _ = networks.q_network.apply(q_params, obs, action)
      labels = labels_replay

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

    # Metrics are computed on the B×B replay-goal submatrix so they are
    # comparable whether or not extra uniform-goal negatives are present.
    if train_logits.ndim == 2:
      narrow_logits = train_logits[:, :batch_size]
    else:
      narrow_logits = train_logits[:, :batch_size, :]

    if narrow_logits.ndim == 3:
      logits_flat = jnp.mean(narrow_logits, axis=-1)
    else:
      logits_flat = narrow_logits

    correct = (jnp.argmax(logits_flat, axis=1) == jnp.argmax(labels_replay, axis=1))
    logits_pos = jnp.sum(logits_flat * labels_replay) / jnp.sum(labels_replay)
    logits_neg = jnp.sum(logits_flat * (1 - labels_replay)) / jnp.sum(1 - labels_replay)
    if train_logits.ndim == 3:
      logsumexp_val = jax.nn.logsumexp(train_logits[:, :, 0], axis=1) ** 2
    else:
      logsumexp_val = jax.nn.logsumexp(train_logits, axis=1) ** 2

    metrics = {
        'crl_loss': loss,
        'binary_accuracy': jnp.mean((logits_flat > 0) == labels_replay),
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

  if _return_raw:
    return jax.jit(update), update
  return jax.jit(update)


def make_scan_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    repr_tau: float = 0.0,
):
  """Scan-based CRL updater: runs N CRL steps in a single JIT call.

  The caller pre-samples all N replay batches in NumPy, stacks them into
  ``(N, B, dim)`` arrays, and transfers to GPU once; a single
  ``jax.lax.scan`` call replaces N separate dispatch-and-sync rounds
  through ``make_crl_update_fn``'s per-step ``update``.

  When ``0 < repr_tau < 1``, the "slow" EMA copy of q_params used for the
  PPO reward (see ``run_ppo_training``'s ``q_params_reward``) is also
  updated inside the scan, eliminating N un-jitted ``_ema_tree`` calls
  per iteration.

  Returns:
    ``multi_update(q_params, q_opt_state, q_params_reward, batches, key)``
    → ``(new_q_params, new_q_opt_state, new_q_params_reward, new_key,
         mean_metrics)``
    where ``batches`` is a dict of ``(N, B, dim)`` JAX arrays.
  """
  _, raw_update = make_crl_update_fn(networks, q_optimizer, _return_raw=True)

  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def multi_update(q_params, q_opt_state, q_params_reward, batches, key):
    def scan_step(carry, batch):
      q_p, q_opt, q_reward, k = carry
      k, k_crl = jax.random.split(k)
      q_p, q_opt, m = raw_update(q_p, q_opt, batch, k_crl)
      if use_ema:
        q_reward = _ema_tree(q_reward, q_p, _tau)
      else:
        q_reward = q_p
      return (q_p, q_opt, q_reward, k), m

    (q_params, q_opt_state, q_params_reward, key), metrics = jax.lax.scan(
        scan_step, (q_params, q_opt_state, q_params_reward, key), batches)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return q_params, q_opt_state, q_params_reward, key, metrics

  return multi_update


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
                     iteration: int, global_step: int, key,
                     hidden_layer_sizes=None,
                     q_params_reward=None,
                     obs_normalizer_state=None):
  """Write a pickle checkpoint atomically (write to tmp → rename).

  `hidden_layer_sizes` (if given) is stored so a later rollout/video
  script can reconstruct the exact trained network architecture without
  the caller having to remember and re-pass a custom override correctly.

  `q_params_reward` (if given) is the EMA "slow" copy of q_params used for
  the PPO reward when `ppo_crl_repr_tau` is enabled; stored separately so
  resuming from a checkpoint doesn't reset the EMA back to the online
  q_params.

  `obs_normalizer_state` (if given) is `ObsNormalizer.state_dict()` --
  stored so resuming doesn't reset the running mean/std back to zero, and
  so `ppo_rollout_video.py` can reconstruct the exact same normalizer a
  saved policy was trained against (its own `from_state_dict` carries the
  obs_dim/start_index/end_index needed, no separate config required).
  """
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
      'hidden_layer_sizes':  (tuple(hidden_layer_sizes)
                             if hidden_layer_sizes is not None else None),
      'q_params_reward':     q_params_reward,
      'obs_normalizer_state': obs_normalizer_state,
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
    video_fn: Optional[Callable] = None,
    video_every_steps: int = 0,
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
  # ---- vec env ------------------------------------------------------------
  # Built BEFORE any other ManiSkill env in this process (see below): SAPIEN
  # requires `physx.enable_gpu()` -- triggered by ManiSkill itself the
  # moment a `num_envs > 1` env is constructed, auto sim_backend='physx_cuda'
  # -- to be the FIRST PhysX-touching call ever made in the process, or it
  # raises "GPU PhysX can only be enabled once before any other code
  # involving PhysX". The `probe_env`/spec-building step right after this
  # constructs a `num_envs=1` (CPU, sim_backend='physx_cpu') ManiSkill env;
  # if that ran first, the native-vec GPU env below would fail to init.
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
    # When ppo_crl_add_extrinsic_reward is on, use the narrow drawer-only
    # success signal (matching the evaluator/ppo_rnd_learner.py) instead of
    # the default 'success' key, so `roll_env_rew` below is exactly the 0/1
    # bonus to add to the reps reward. No-op (default 'success', unused) for
    # every other env_name or when the flag is off.
    _success_key = 'success'
    if bool(getattr(config, 'ppo_crl_add_extrinsic_reward', False)):
      if _env_name == 'maniskill_close_subtask_train':
        _success_key = 'drawer_closed'
      elif _env_name == 'maniskill_open_subtask_train':
        _success_key = 'drawer_open'
    vec_env = _env_utils.ManiskillVecEnv(
        _env_name, config.ppo_num_envs,
        obs_dim=int(config.obs_dim),
        start_index=int(config.start_index),
        end_index=int(config.end_index),
        success_key=_success_key)
    print(f'[ppo] maniskill native vec env: {_env_name}, '
          f'num_envs={config.ppo_num_envs} (single GPU-batched sim), '
          f'success_key={_success_key!r}')
  else:
    vec_env = VecEnv(env_factory, config.ppo_num_envs, seed=seed * 31)
  E = vec_env.num_envs

  # ---- build networks from env spec -------------------------------------
  from acme import specs as _specs
  import contrastive.utils as _cu

  probe_env = env_factory(seed)
  spec = _specs.make_environment_spec(probe_env)
  networks = network_factory(spec=spec)
  del probe_env
  obs_shape = vec_env.observation_shape
  act_shape = vec_env.action_shape

  # ---- density estimator selection (φ·ψ contrastive vs. NF RealNVP) -----
  repr_mode = (str(getattr(config, 'ppo_repr_mode', 'crl') or 'crl')).strip().lower()
  if repr_mode not in ('crl', 'nf'):
    raise ValueError(
        f"Unknown config.ppo_repr_mode={repr_mode!r}; expected 'crl' or 'nf'")
  use_nf = repr_mode == 'nf'
  obs_dim_cfg = int(config.obs_dim)
  goal_dim_cfg = int(np.prod(obs_shape)) - obs_dim_cfg
  act_dim_cfg = int(np.prod(act_shape))

  nf_density_nets = None
  if use_nf:
    nf_density_nets = _nf.make_nf_density_networks(
        obs_dim=obs_dim_cfg, act_dim=act_dim_cfg, goal_dim=goal_dim_cfg,
        rep_size=int(config.nf_rep_size),
        num_blocks=int(config.nf_num_blocks),
        channels=int(config.nf_coupling_width),
        goal_enc_size=int(config.nf_goal_enc_size),
        sa_hidden=int(config.nf_sa_hidden),
        sa_num_layers=int(config.nf_sa_num_layers))
    print(f'[ppo] repr_mode=nf (RealNVP)  obs_dim={obs_dim_cfg} '
          f'act_dim={act_dim_cfg} goal_dim={goal_dim_cfg} '
          f'rep_size={config.nf_rep_size} num_blocks={config.nf_num_blocks} '
          f'coupling_width={config.nf_coupling_width} '
          f'flow_dim={nf_density_nets.flow_dim} '
          f'sa_encoder={config.nf_sa_num_layers}x{config.nf_sa_hidden}+swish')

  T = int(config.ppo_rollout_length)
  batch_per_iter = T * E
  mb_size = batch_per_iter // int(config.ppo_num_minibatches)
  assert mb_size * int(config.ppo_num_minibatches) == batch_per_iter, (
      'batch_per_iter must divide evenly into ppo_num_minibatches')
  num_iterations = int(total_steps) // (T * E)

  # ---- observation normalizer --------------------------------------------
  # See `ObsNormalizer` docstring: normalizes state+goal per-column online,
  # goal columns sharing stats with their corresponding state column.
  norm_obs = bool(getattr(config, 'ppo_norm_obs', True))
  obs_normalizer = (
      ObsNormalizer(obs_dim=int(config.obs_dim),
                    start_index=int(config.start_index),
                    end_index=int(config.end_index))
      if norm_obs else None)

  def _obs_update(raw_state: np.ndarray):
    """Folds raw STATE observations into the running stats (no-op if off)."""
    if obs_normalizer is not None:
      obs_normalizer.update(raw_state[..., :int(config.obs_dim)])

  def _obs_norm(raw_obs: np.ndarray) -> np.ndarray:
    """Normalizes a raw `concat([state, goal])` batch (identity if off)."""
    return obs_normalizer(raw_obs) if obs_normalizer is not None else raw_obs

  # ---- init params ------------------------------------------------------
  key = jax.random.PRNGKey(seed)
  k_pol, k_val, k_q, key = jax.random.split(key, 4)
  policy_params = networks.policy_network.init(k_pol)
  value_params = networks.value_network.init(k_val)
  q_params = (_nf.init_nf_params(nf_density_nets, k_q) if use_nf
              else networks.q_network.init(k_q))
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

  if use_nf:
    q_optimizer = _nf.make_nf_optimizers(
        encoder_lr=float(config.nf_encoder_lr),
        critic_lr=float(config.nf_critic_lr),
        critic_weight_decay=float(config.nf_critic_weight_decay),
        grad_clip=float(config.nf_grad_clip),
        has_goal_encoder=(nf_density_nets.goal_encoder_net is not None))
  else:
    q_optimizer = optax.adam(float(config.learning_rate))
  q_opt_state = q_optimizer.init(q_params)

  # ---- slow "reward" copy of q_params (see ppo_crl_repr_tau) -------------
  # When enabled, `q_params_reward` is an EMA-averaged copy of q_params
  # used only to compute the PPO reward; the CRL loss keeps training the
  # live `q_params` every step regardless. 0 (default) disables this: the
  # reward reads the live q_params directly (q_params_reward == q_params).
  _repr_tau = (float(config.ppo_nf_reward_tau) if use_nf
               else float(getattr(config, 'ppo_crl_repr_tau', 0.0)))
  _use_repr_ema = 0.0 < _repr_tau < 1.0
  q_params_reward = q_params

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
      q_params_reward = (_ckpt.get('q_params_reward', None)
                        if _use_repr_ema else q_params)
      if q_params_reward is None:
        q_params_reward = q_params
      _obs_norm_state = _ckpt.get('obs_normalizer_state', None)
      if obs_normalizer is not None and _obs_norm_state is not None:
        obs_normalizer.load_state_dict(_obs_norm_state)
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

  def _reward_q_params():
    return q_params_reward if _use_repr_ema else q_params

  # ---- jitted helpers ---------------------------------------------------
  gae_fn = make_gae_fn(config)
  ppo_update = make_ppo_update_fn(networks, config, ppo_optimizer)

  reward_fn = None
  nf_reward_fn = None
  crl_scan_update = None
  nf_scan_update = None
  if use_nf:
    nf_reward_fn = _nf.make_nf_reward_fn(nf_density_nets, obs_dim=obs_dim_cfg)
    nf_scan_update = _nf.make_scan_nf_update_fn(
        nf_density_nets, q_optimizer, obs_dim=obs_dim_cfg,
        noise_std=float(config.nf_noise_std), repr_tau=_repr_tau)
    if _use_repr_ema:
      print(f'[ppo] NF reward repr EMA: tau={_repr_tau} '
            f'(reward uses a slow copy of NF params)')
  else:
    reward_fn = make_reward_fn(networks)
    crl_scan_update = make_scan_crl_update_fn(
        networks, q_optimizer, repr_tau=_repr_tau)
    if _use_repr_ema:
      print(f'[ppo] CRL reward repr EMA: tau={_repr_tau} '
            f'(reward uses a slow copy of q_params)')

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
    # Deterministic policy mode (NormalTanhDistribution.mode(), no rng
    # needed) -- used for eval rollouts so success_rate/success_rate_1000
    # reflect the policy's actual behavior, not sampling noise.
    dist = networks.policy_network.apply(policy_p, obs)
    return networks.sample_eval(dist, None)

  # ---- uniform-sampling goal bounds (extracted once from env spec) ------
  uniform_sampling = bool(getattr(config, 'uniform_sampling', False))
  if uniform_sampling and use_nf:
    print('[ppo] WARNING: uniform_sampling has no effect under '
          'ppo_repr_mode=nf (NF density loss ignores extra_goals); '
          'disabling for this run.')
    uniform_sampling = False
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

  # ---- NF goal-normalization running stats (only used when use_nf) ------
  nf_goal_mean = np.zeros(goal_dim_cfg, dtype=np.float32) if use_nf else None
  nf_goal_std = np.ones(goal_dim_cfg, dtype=np.float32) if use_nf else None
  _nf_goal_std_min = float(getattr(config, 'nf_goal_std_min', 0.1))
  _nf_normalizer_reset_done = False

  # ---- per-env episode buffers (for flushing complete trajectories) -----
  ep_obs: list = [[] for _ in range(E)]
  ep_act: list = [[] for _ in range(E)]
  ep_return = np.zeros(E, dtype=np.float32)
  ep_len = np.zeros(E, dtype=np.int32)
  recent_returns: list = []
  recent_lengths: list = []

  obs = vec_env.reset()
  _obs_update(obs)
  obs = _obs_norm(obs)
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
  elif _env == 'maniskill_close_subtask_train':
    # Drawer-closed only -- mshab's own info['success'] (== env reward,
    # still logged separately below) also requires the arm back at rest
    # and the robot static, which the CRL goal never trains for. See
    # `contrastive.utils.DrawerClosedSuccessObserver`.
    eval_success_obs = _cu.DrawerClosedSuccessObserver()
  elif _env == 'maniskill_open_subtask_train':
    # Drawer-open only, mirroring the close branch above -- see
    # `contrastive.utils.DrawerOpenSuccessObserver`.
    eval_success_obs = _cu.DrawerOpenSuccessObserver()
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
  # Per-step env `dones` (post-step), used to reset the reward normalizer's
  # running-return tracker at episode boundaries once reward is computed
  # in a single batched call after the rollout (see step 1b below).
  roll_step_dones = np.zeros((T, E), dtype=np.float32)

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

  # ---- periodic mid-training rollout video ------------------------------
  # `video_fn` (if given) is called with (policy_params, obs_normalizer,
  # global_step) every time `global_step` crosses a `video_every_steps`
  # boundary. Resuming from a checkpoint already past one or more boundaries
  # must not immediately re-trigger them, so the next boundary is computed
  # from the restored `global_step`, not from 0.
  next_video_step = (
      ((global_step // video_every_steps) + 1) * video_every_steps
      if (video_fn is not None and video_every_steps > 0) else None)

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

      next_obs, env_rew, dones, terminal_obs = vec_env.step(action)
      # `terminal_obs` is always a genuine raw post-step state; `next_obs`
      # rows where `dones` is True are ALSO a genuine raw state (the fresh
      # auto-reset observation) -- both are worth folding into the running
      # stats. Rows where `dones` is False have next_obs == terminal_obs,
      # so only counting them once (via terminal_obs) avoids double-weighting.
      _obs_update(terminal_obs)
      if dones.any():
        _obs_update(next_obs[dones])
      terminal_obs = _obs_norm(terminal_obs)
      next_obs = _obs_norm(next_obs)
      roll_env_rew[t] = env_rew
      roll_step_dones[t] = dones.astype(np.float32)

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
    # 1b. Batched reward computation (single GPU call over the whole
    # rollout, instead of T separate per-step calls). Equivalent to the
    # per-step version: q_params (or its EMA "reward" copy, see
    # ppo_crl_repr_tau) doesn't change during rollout collection, only
    # between iterations, so computing it before vs. after the T-loop
    # gives identical values.
    # =================================================================
    _flat_obs_j = jnp.asarray(roll_obs.reshape(batch_per_iter, -1))
    _flat_acts_j = jnp.asarray(roll_acts.reshape(batch_per_iter, -1))
    if use_nf:
      _rew_flat = np.asarray(nf_reward_fn(
          _reward_q_params(), _flat_obs_j, _flat_acts_j,
          jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std)))
    else:
      _rew_flat = np.asarray(
          reward_fn(_reward_q_params(), _flat_obs_j, _flat_acts_j))
    roll_rew_raw[:] = _rew_flat.reshape(T, E)
    if bool(getattr(config, 'ppo_crl_add_extrinsic_reward', False)):
      # Add the sparse extrinsic success reward (roll_env_rew, already the
      # narrow drawer-only 0/1 flag -- see the success_key resolution above)
      # on top of the reps reward, BEFORE normalization, so the combined
      # signal is scaled by reward_normalizer exactly like the pure φ·ψ
      # reward was before.
      roll_rew_raw[:] += roll_env_rew

    # Scale reps reward by the running std of discounted returns.  Reset
    # at episode boundaries via the per-step `dones` recorded above.
    if reward_normalizer is not None:
      for _t in range(T):
        roll_rew[_t] = reward_normalizer(roll_rew_raw[_t], roll_step_dones[_t])
    else:
      roll_rew[:] = roll_rew_raw

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
    # 4. Density-estimator updates (off-policy, from replay): CRL InfoNCE
    #    or NF NLL, dispatched on config.ppo_repr_mode.
    # =================================================================
    crl_metrics_agg: Dict[str, list] = {}
    _n_crl = int(config.ppo_crl_steps_per_iter)
    if replay.size >= int(config.ppo_min_replay_size) and _n_crl > 0:
      if use_nf:
        if not _nf_normalizer_reset_done:
          # First NF activation: reset the return normalizer so the
          # untrained flow's wild initial log p_NF rewards don't
          # permanently corrupt the running return-std PPO uses to scale
          # advantages. CRL's cold start is comparatively much milder so
          # it doesn't need this.
          if reward_normalizer is not None:
            reward_normalizer._rms = RunningMeanStd(shape=())
            reward_normalizer._returns = np.zeros(E, dtype=np.float64)
          _nf_normalizer_reset_done = True

        # Refresh goal normalization stats from a fresh replay sample.
        _mix_task = bool(getattr(config, 'nf_mix_task_goal_stats', False))
        _stat_n = int(config.batch_size) if _mix_task else min(2048, replay.size)
        _stat_n = min(_stat_n, int(replay.size))
        _stat_batch = replay.sample(_stat_n, np_rng)
        _goals = _stat_batch['obs'][:, obs_dim_cfg:]
        if _mix_task:
          # Mix in batch_size copies of the current rollout's actual env
          # task goal so it's in-distribution for the normalizer, not just
          # hindsight-relabeled goals -- otherwise r = log p_NF(g_task|s,a)
          # can be wildly out-of-distribution (and explode) early on.
          _g_task = roll_obs.reshape(-1, roll_obs.shape[-1])[
              0, obs_dim_cfg:].astype(np.float32)
          _n_task = int(config.batch_size)
          _goals = np.concatenate(
              [_goals, np.repeat(_g_task[None], _n_task, axis=0)], axis=0)
        nf_goal_mean = _goals.mean(axis=0).astype(np.float32)
        nf_goal_std = np.maximum(
            _goals.std(axis=0), _nf_goal_std_min).astype(np.float32)

        _samples = [replay.sample(int(config.batch_size), np_rng)
                    for _ in range(_n_crl)]
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        (q_params, q_opt_state, q_params_reward,
         key, _nf_lam_unused, m) = nf_scan_update(
            q_params, q_opt_state, q_params_reward, _stacked, key,
            jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std))
        crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}
      else:
        # Pre-sample all N batches in NumPy, stack to (N, B, dim), and run
        # all N CRL steps in a single jax.lax.scan (one JIT dispatch instead
        # of N), also updating the EMA "reward" copy of q_params inside the
        # scan when ppo_crl_repr_tau is enabled.
        if uniform_sampling:
          # `goal_low`/`goal_high` are RAW env bounds (real physical units --
          # correct support to sample from). `replay` already stores
          # normalized obs (see `_obs_norm` above), so the resulting
          # `extra_goals` must be normalized too before joining the batch,
          # or these "negatives" would sit at a wildly different scale than
          # the (already normalized) positives/other negatives.
          _samples = []
          for _ in range(_n_crl):
            s = replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high,
                num_negatives=int(getattr(config, 'uniform_num_negatives', -1)))
            if obs_normalizer is not None:
              s['extra_goals'] = obs_normalizer.normalize_goal(s['extra_goals'])
            _samples.append(s)
        else:
          _samples = [
              replay.sample(int(config.batch_size), np_rng)
              for _ in range(_n_crl)]
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        (q_params, q_opt_state, q_params_reward,
         key, m) = crl_scan_update(
            q_params, q_opt_state, q_params_reward, _stacked, key)
        crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}

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
    }
    if use_nf:
      log['crl/density_loss'] = float('nan')
      log['crl/log_p_mean'] = float('nan')
    else:
      log['crl/crl_loss'] = float('nan')
      log['crl/categorical_accuracy'] = float('nan')
      log['crl/binary_accuracy'] = float('nan')
      log['crl/logits_pos'] = float('nan')
      log['crl/logits_neg'] = float('nan')
      log['crl/logsumexp'] = float('nan')
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
          # Deterministic policy mode (not a stochastic training-time
          # sample) so success_rate/success_rate_1000 reflect the policy's
          # actual behavior, not sampling noise. Frozen normalization: uses
          # the current running stats but does NOT update them from eval
          # data (keeps eval deterministic/comparable across iterations,
          # and `eval_dist_obs` below still reads raw meters/radians off
          # `ts` directly, unaffected).
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
    # 6b. Periodic mid-training rollout video (every `video_every_steps`
    #     global steps). `video_fn` owns rendering + wandb logging; it
    #     always writes to the same local path (caller's responsibility)
    #     so disk usage doesn't grow with training length.
    # =================================================================
    if next_video_step is not None and global_step >= next_video_step:
      video_fn(ppo_params['policy'], obs_normalizer, global_step, iteration)
      next_video_step += video_every_steps

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
          key=key,
          hidden_layer_sizes=config.hidden_layer_sizes,
          q_params_reward=(q_params_reward if _use_repr_ema else None),
          obs_normalizer_state=(obs_normalizer.state_dict()
                                if obs_normalizer is not None else None))

  # ---- return final state in case the caller wants to checkpoint --------
  return PPOTrainingState(
      policy_params=ppo_params['policy'],
      value_params=ppo_params['value'],
      ppo_optimizer_state=ppo_opt_state,
      q_params=q_params,
      q_optimizer_state=q_opt_state,
      key=key,
      obs_normalizer=obs_normalizer,
  )
