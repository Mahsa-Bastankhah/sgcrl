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
import concurrent.futures
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
from contrastive import gaussian_density as _gd
from contrastive import nf_density as _nf
from contrastive import fm_density as _fm
from contrastive import td3_density as _td3
from contrastive.utils import extract_info_reward
from distributional import TanhTransformedDistribution

import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

# Near-zero pre-tanh scale used when `ppo_deterministic_select_dim` forces the
# last action dim (BuilderBench PD select) to μ/mode.
_SELECT_DET_SCALE = 1e-6


def _policy_dist_maybe_det_select(dist, enabled: bool):
  """Optionally force last action dim to near-zero std (select → μ only).

  Policy head is ``Independent(TanhTransformedDistribution(Normal))``.  When
  ``enabled``, rebuild with ``scale[..., -1] = _SELECT_DET_SCALE`` so sample /
  log_prob / entropy treat select as deterministic while other dims are
  unchanged.  No-op for hybrid categorical-select actors (select is already
  discrete).
  """
  if not enabled:
    return dist
  if getattr(dist, 'hybrid_select', False):
    return dist
  tanh_td = dist.distribution
  normal = tanh_td.distribution
  new_scale = normal.scale.at[..., -1].set(
      jnp.asarray(_SELECT_DET_SCALE, dtype=normal.scale.dtype))
  new_normal = tfd.Normal(loc=normal.loc, scale=new_scale)
  threshold = float(getattr(tanh_td, '_threshold', 0.999))
  new_tanh = TanhTransformedDistribution(new_normal, threshold=threshold)
  reinterpreted = int(dist.reinterpreted_batch_ndims)
  return tfd.Independent(new_tanh, reinterpreted_batch_ndims=reinterpreted)


def _policy_normal_loc_scale(dist):
  """Pre-tanh Gaussian ``loc`` / ``scale`` for PPO diagnostics.

  Default actor: ``Independent(TanhTransformed(Normal))``.
  Hybrid categorical-select: same Gaussian lives on ``dist.continuous_dist``
  (shared continuous head, or mode-cube packed ``[xyz,yaw]`` for the
  per-cube waypoint actor).
  """
  cont = (dist.continuous_dist
          if getattr(dist, 'hybrid_select', False) else dist)
  normal = cont.distribution.distribution
  return normal.loc, normal.scale


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
  """Welford-style online mean / variance tracker with optional window cap.

  Matches OpenAI Baselines' `RunningMeanStd` and CleanRL's vector-env
  reward normalizer.  Maintains mean, variance, and sample count; `update`
  accepts a batch and folds it in via the parallel-algorithm formula.

  When ``max_count > 0`` the effective count is capped at ``max_count`` after
  each update.  Once the cap is reached every new batch of size B gets weight
  B / max_count instead of B / (old_count + B), making the estimate track
  recent data more closely (soft sliding-window effect).
  """

  def __init__(self, shape=(), epsilon: float = 1e-4, max_count: float = 0):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = float(epsilon)
    self.max_count = float(max_count) if max_count > 0 else 0.0

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
    # Soft window cap: once count exceeds max_count, pin it so new data
    # gets proportionally more weight in future updates.
    if self.max_count > 0 and self.count > self.max_count:
      self.count = self.max_count


def _normalize_packed_obs(
    obs: jnp.ndarray,
    mean: jnp.ndarray,
    var: jnp.ndarray,
    *,
    obs_dim: int,
    start_index: int,
    end_index: int,
    clip: float,
    enabled: bool,
    goal_state_indices=None,
) -> jnp.ndarray:
  """Normalize packed ``[state; goal]`` observations from state-only stats."""
  if not enabled:
    return obs
  active = jnp.all(jnp.isfinite(mean))
  mean = jnp.nan_to_num(mean)
  state = obs[..., :obs_dim]
  goal = obs[..., obs_dim:]
  state_norm = (state - mean) / jnp.sqrt(jnp.maximum(var, 1e-8))
  if goal_state_indices is not None:
    idx = jnp.asarray(goal_state_indices, dtype=jnp.int32)
    goal_mean = mean[idx]
    goal_var = var[idx]
  else:
    goal_mean = mean[start_index:end_index]
    goal_var = var[start_index:end_index]
  goal_norm = (goal - goal_mean) / jnp.sqrt(jnp.maximum(goal_var, 1e-8))
  normalized = jnp.clip(
      jnp.concatenate([state_norm, goal_norm], axis=-1), -clip, clip)
  return jnp.where(active, normalized, obs)


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

  window_size > 0 caps the effective sample count so the variance estimate
  stays responsive to recent changes (useful when the reward distribution
  shifts rapidly, e.g. as the NF density model improves).
  """

  def __init__(self, num_envs: int, discount: float, epsilon: float = 1e-8,
               window_size: int = 0):
    self._rms = RunningMeanStd(shape=(), max_count=float(window_size))
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

      k  ~ Categorical(w_k / sum_i w_i) over stored episodes
            (w_k = success_sample_weight if episode k succeeded else 1;
             default weight 1 recovers Uniform)
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

  Storage: list of episodes, each
  {'obs': (T+1, D), 'action': (T, A), 'successful': bool}.
  FIFO eviction is at the episode granularity once total stored
  transitions exceed `capacity`.

  Eviction uses a lazy start-index (O(1) popleft, O(1) random access)
  instead of ``list.pop(0)`` which is O(n) and dominates once the buffer
  holds hundreds of thousands of episodes.  Dead prefix is compacted
  periodically.
  """

  # Compact when the dead prefix is at least this large *and* at least
  # half of the backing list (amortized O(1) eviction).
  _COMPACT_MIN_DEAD = 4096

  def __init__(self, capacity: int, obs_dim: int, discount: float,
               start_index: int, end_index: int,
               success_sample_weight: float = 1.0,
               goal_state_indices=None):
    self._cap = capacity
    self._obs_dim = obs_dim               # state slice size
    self._discount = float(discount)
    self._start_index = start_index
    self._end_index = end_index
    if goal_state_indices is None:
      self._goal_state_indices = None
    else:
      self._goal_state_indices = np.asarray(
          goal_state_indices, dtype=np.int32).reshape(-1)
    self._success_sample_weight = float(success_sample_weight)
    if self._success_sample_weight <= 0.0:
      raise ValueError(
          'success_sample_weight must be > 0, got '
          f'{self._success_sample_weight}')
    self._episodes: list = []             # list[{'obs': (T+1, D), 'action': (T, A)}]
    self._ep_lens: list = []              # parallel list of T (== len(action))
    self._ep_weights: list = []           # parallel sampling weights
    self._start = 0                       # first live episode index
    self._weight_sum = 0.0
    self._total_transitions = 0
    self._act_dim: Optional[int] = None
    self._goal_dim: Optional[int] = None
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
    return len(self._episodes) - self._start

  def _compact_if_needed(self, force: bool = False):
    """Drop the dead FIFO prefix so random access stays dense."""
    dead = self._start
    if dead <= 0:
      return
    n = len(self._episodes)
    if force or (dead >= self._COMPACT_MIN_DEAD and dead >= (n >> 1)):
      self._episodes = self._episodes[dead:]
      self._ep_lens = self._ep_lens[dead:]
      self._ep_weights = self._ep_weights[dead:]
      self._start = 0

  def add_episode(self, obs: np.ndarray, action: np.ndarray,
                  successful: bool = False):
    """Store one complete episode.

    Args:
      obs:    (T+1, obs_dim_total) — includes the terminal observation.
      action: (T,   act_dim)
      successful: Whether the episode saw the environment goal at least once.
    Episodes shorter than 1 transition are rejected.
    """
    T = action.shape[0]
    assert obs.shape[0] == T + 1, f'obs len {obs.shape[0]} != action len {T} + 1'
    if T < 1:
      return
    weight = (
        self._success_sample_weight if bool(successful) else 1.0)
    act_arr = np.asarray(action, dtype=np.float32)
    obs_arr = np.asarray(obs, dtype=np.float32)
    if self._act_dim is None:
      self._act_dim = int(act_arr.shape[1])
      # Infer goal dim once from the first stored episode.
      self._goal_dim = int(
          self._obs_to_goal(obs_arr[:1, :self._obs_dim]).shape[1])
    self._episodes.append({
        'obs': obs_arr,
        'action': act_arr,
        'successful': bool(successful),
    })
    self._ep_lens.append(T)
    self._ep_weights.append(float(weight))
    self._weight_sum += float(weight)
    self._total_transitions += T
    # FIFO eviction: O(1) logical popleft (no list.pop(0)).
    while (self._total_transitions > self._cap
           and self.num_episodes > 1):
      i = self._start
      self._total_transitions -= self._ep_lens[i]
      self._weight_sum -= float(self._ep_weights[i])
      self._episodes[i] = None  # drop ref for GC
      self._start = i + 1
    self._compact_if_needed()

  # ---------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------
  def _obs_to_goal(self, states: np.ndarray) -> np.ndarray:
    """Equivalent to `contrastive/utils.py::obs_to_goal_2d`.

    When ``goal_state_indices`` is set (BuilderBench cube masks), gather those
    coordinates instead of a contiguous ``start:end`` slice.
    """
    if self._goal_state_indices is not None:
      return states[:, self._goal_state_indices]
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
    start = self._start
    num_eps = len(self._episodes) - start
    assert num_eps > 0, 'Cannot sample from an empty buffer.'
    # Slice views of the live prefix (cheap; no full-list copy of episodes).
    ep_lens = np.asarray(self._ep_lens[start:], dtype=np.int64)  # (K,)

    # (1) Episode index: uniform, or weighted by success_sample_weight.
    if self._success_sample_weight == 1.0:
      ep_ids = rng.integers(0, num_eps, size=B)
    else:
      weights = np.asarray(self._ep_weights[start:], dtype=np.float64)
      ep_ids = rng.choice(
          num_eps, size=B, replace=True,
          p=weights / self._weight_sum)
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

    # (4) Gather states/actions, then one batched obs→goal.
    obs_dim = self._obs_dim
    act_dim = (self._act_dim if self._act_dim is not None
               else int(self._episodes[start]['action'].shape[1]))
    s_t = np.empty((B, obs_dim), dtype=np.float32)
    s_tp1 = np.empty((B, obs_dim), dtype=np.float32)
    s_j = np.empty((B, obs_dim), dtype=np.float32)
    act_out = np.empty((B, act_dim), dtype=np.float32)
    next_act_out = np.empty((B, act_dim), dtype=np.float32)
    episodes = self._episodes
    # Local binds for the tight gather loop.
    ep_ids_l = ep_ids.tolist()
    t_l = t.tolist()
    j_l = j.tolist()
    lens_l = lens.tolist()
    for i in range(B):
      ep = episodes[start + ep_ids_l[i]]
      full_obs = ep['obs']
      full_act = ep['action']
      ti = t_l[i]
      ji = j_l[i]
      Ti = lens_l[i]
      s_t[i] = full_obs[ti, :obs_dim]
      s_tp1[i] = full_obs[ti + 1, :obs_dim]
      s_j[i] = full_obs[ji, :obs_dim]
      act_out[i] = full_act[ti]
      # a' = a_{t+1} when available; else fall back to a_t (terminal step).
      next_act_out[i] = full_act[ti + 1] if (ti + 1) < Ti else full_act[ti]

    goals = self._obs_to_goal(s_j)  # (B, goal_dim)
    goal_dim = goals.shape[1]
    obs_out = np.empty((B, obs_dim + goal_dim), dtype=np.float32)
    next_obs_out = np.empty_like(obs_out)
    obs_out[:, :obs_dim] = s_t
    obs_out[:, obs_dim:] = goals
    next_obs_out[:, :obs_dim] = s_tp1
    next_obs_out[:, obs_dim:] = goals

    return {
        'obs': obs_out,
        'action': act_out,
        'next_obs': next_obs_out,
        'next_action': next_act_out,  # a_{t+1}; used by TD-NF
        'future_state': s_j,  # s_j (full state); used by TD3
    }

  def sample_states(self, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Return up to *n* raw states (shape ``(n, obs_dim)``) sampled uniformly
    from all stored transitions.  Used for KDE fitting."""
    live = self._episodes[self._start:]
    all_states = np.concatenate(
        [ep['obs'][:, :self._obs_dim] for ep in live], axis=0)
    n = min(n, len(all_states))
    idx = rng.choice(len(all_states), size=n, replace=False)
    return all_states[idx].astype(np.float32)

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
    out = {'obs': obs, 'action': batch['action'], 'next_obs': next_obs}
    if 'next_action' in batch:
      out['next_action'] = batch['next_action']
    if 'future_state' in batch:
      out['future_state'] = batch['future_state']
    return out


def _flush_rollout_episodes_to_replay(
    *,
    actions: np.ndarray,
    env_rew: np.ndarray,
    dones: np.ndarray,
    terminal_obs: np.ndarray,
    next_obs: np.ndarray,
    success: Optional[np.ndarray],
    ep_obs: list,
    ep_act: list,
    ep_return: np.ndarray,
    ep_len: np.ndarray,
    ep_success_max: np.ndarray,
    s0_states: np.ndarray,
    obs_dim: int,
    recent_returns: list,
    recent_lengths: list,
    recent_success: list,
    use_crl_td3_switch: bool,
    hard_goal_visit_count: int,
    replay: 'EpisodeReplay',
) -> int:
  """Flush completed episodes from a (T, E) rollout into ``replay``.

  Same semantics as the old nested ``for t / for i`` stitcher (including
  time-major flush order for replay FIFO / recent-stat windows), but walks
  done events only and slices trajectory arrays once per finished episode.

  ``ep_obs[i]`` / ``ep_act[i]`` are ndarray carries of shape ``(L, D)`` /
  ``(L-1, A)`` (or ``(0, A)`` with a single seed obs).
  """
  T, E = int(dones.shape[0]), int(dones.shape[1])
  track_success = success is not None
  act_dim = int(actions.shape[-1])
  # Per-env open-episode state (mirrors the old running buffers).
  t0 = np.zeros(E, dtype=np.int32)
  ret = ep_return.astype(np.float64, copy=True)
  ln = ep_len.astype(np.int64, copy=True)
  succ_max = ep_success_max.astype(np.float64, copy=True)
  # np.nonzero on (T, E) yields time-major then env order — same as
  # ``for t in range(T): for i in range(E)``.
  done_ts, done_is = np.nonzero(dones)

  for t_end, i in zip(done_ts.tolist(), done_is.tolist()):
    start = int(t0[i])
    seg_act = actions[start:t_end + 1, i]
    prefix_obs = ep_obs[i]
    prefix_act = ep_act[i]
    if t_end > start:
      ep_o = np.concatenate(
          [prefix_obs, next_obs[start:t_end, i],
           terminal_obs[t_end:t_end + 1, i]],
          axis=0)
    else:
      ep_o = np.concatenate(
          [prefix_obs, terminal_obs[t_end:t_end + 1, i]], axis=0)
    if prefix_act.shape[0]:
      ep_a = np.concatenate([prefix_act, seg_act], axis=0)
    else:
      ep_a = np.asarray(seg_act, dtype=np.float32)

    ret[i] += float(env_rew[start:t_end + 1, i].sum())
    ln[i] += (t_end - start + 1)
    if track_success:
      succ_max[i] = max(
          float(succ_max[i]), float(success[start:t_end + 1, i].max()))
    episode_succeeded = bool(track_success and succ_max[i] >= 0.5)
    try:
      replay.add_episode(ep_o, ep_a, successful=episode_succeeded)
    except AssertionError:
      pass

    recent_returns.append(float(ret[i]))
    recent_lengths.append(int(ln[i]))
    if track_success:
      recent_success.append(float(episode_succeeded))
      if use_crl_td3_switch and episode_succeeded:
        hard_goal_visit_count += 1
      if len(recent_success) > 1000:
        del recent_success[:-1000]
    if len(recent_returns) > 100:
      del recent_returns[:-100]
      del recent_lengths[:-100]

    # Auto-reset obs seeds the next episode.
    ep_obs[i] = next_obs[t_end:t_end + 1, i].copy()
    ep_act[i] = np.zeros((0, act_dim), dtype=np.float32)
    s0_states[i] = next_obs[t_end, i, :obs_dim].copy()
    ret[i] = 0.0
    ln[i] = 0
    succ_max[i] = 0.0
    t0[i] = t_end + 1

  # Append unfinished tails (no episode flush).
  for i in range(E):
    start = int(t0[i])
    if start < T:
      seg_act = actions[start:T, i]
      ep_obs[i] = np.concatenate([ep_obs[i], next_obs[start:T, i]], axis=0)
      if ep_act[i].shape[0]:
        ep_act[i] = np.concatenate([ep_act[i], seg_act], axis=0)
      else:
        ep_act[i] = np.asarray(seg_act, dtype=np.float32)
      ret[i] += float(env_rew[start:T, i].sum())
      ln[i] += (T - start)
      if track_success:
        succ_max[i] = max(
            float(succ_max[i]), float(success[start:T, i].max()))
    ep_return[i] = np.float32(ret[i])
    ep_len[i] = np.int32(ln[i])
    ep_success_max[i] = np.float32(succ_max[i])

  return hard_goal_visit_count


# ---------------------------------------------------------------------------
# Core factories.  Each returns a jitted function plus any needed metadata.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Gaussian KDE density estimator (pure numpy, fitted to replay buffer states)
# ---------------------------------------------------------------------------
class GaussianKDE:
  """Isotropic Gaussian KDE fitted to a set of training points.

  All maths is in numpy so it can be called from the (non-jitted) rollout
  loop without JAX overhead or shape-recompilation issues.

  Log density at a batch of test points ``x`` (shape ``(B, d)``)::

      log p(x) = logsumexp_n [ log(1/N) − d·log(h) − d/2·log(2π)
                                − ½·‖(x − x_n)/h‖² ]

  Bandwidth *h* defaults to Scott's rule: ``N^{-1/(d+4)}``.
  """

  def __init__(self, train_xs: np.ndarray, bandwidth: float):
    self.train_xs = train_xs.astype(np.float32)    # (N, d)
    self.bandwidth = float(bandwidth)
    n, d = train_xs.shape
    self._log_w = -np.log(float(n))                # uniform log-weight
    self._log_norm = (0.5 * d * np.log(2.0 * np.pi)
                      + d * np.log(self.bandwidth))

  def log_density(self, test_x: np.ndarray,
                  chunk_size: int = 2000) -> np.ndarray:
    """Return log p(test_x) for test_x of shape ``(B, d)``.

    Processes in chunks of *chunk_size* to keep peak memory bounded at
    ``chunk_size × N_train × d × 4`` bytes (≈64 MB for 2000×2000×2).
    """
    test_x = np.asarray(test_x, dtype=np.float32)
    B = len(test_x)
    if B <= chunk_size:
      return self._log_density_chunk(test_x)
    out = np.empty(B, dtype=np.float32)
    for i in range(0, B, chunk_size):
      out[i:i + chunk_size] = self._log_density_chunk(test_x[i:i + chunk_size])
    return out

  def _log_density_chunk(self, test_x: np.ndarray) -> np.ndarray:
    diffs = (test_x[:, None, :] - self.train_xs[None, :, :]) / self.bandwidth
    log_k = -0.5 * np.sum(diffs ** 2, axis=-1) - self._log_norm  # (B, N)
    log_contrib = self._log_w + log_k                              # (B, N)
    lc_max = log_contrib.max(axis=-1, keepdims=True)
    log_p = (np.log(np.exp(log_contrib - lc_max).sum(axis=-1))
             + lc_max[:, 0])
    return log_p  # (B,)

  @classmethod
  def fit(cls,
          replay: 'EpisodeReplay',
          obs_dim: int,
          max_points: int,
          rng: np.random.Generator,
          bandwidth: Optional[float] = None) -> 'GaussianKDE':
    """Sample *max_points* states from *replay* and fit the KDE."""
    states = replay.sample_states(max_points, rng)
    n, d = states.shape
    if bandwidth is None:
      bandwidth = float(n) ** (-1.0 / (d + 4))
    return cls(states, bandwidth)


def _obs_to_goal_jax(states, start_index, end_index, goal_state_indices):
  """Goal features from raw/packed state rows (matches LHER ``obs_to_goal``)."""
  if goal_state_indices is not None:
    return states[:, jnp.asarray(goal_state_indices, dtype=jnp.int32)]
  if end_index == -1:
    return states[:, start_index:]
  return states[:, start_index:end_index]


def _crl_hit_bonus_matrix(
    s_goal: jnp.ndarray,
    goals: jnp.ndarray,
    mode: str,
    tol: float,
    scale: float,
    task_goal: Optional[jnp.ndarray],
) -> jnp.ndarray:
  """(B,B) hit-indicator bonus added to φ·ψ logits / reward.

  ``mode='sf'``:   scale · 1{‖obs_to_goal(s_i)−g_j‖ < tol}  (pairwise)
  ``mode='goal'``: scale · 1{‖obs_to_goal(s_i)−task_goal‖ < tol}
                   broadcast over all g_j (independent of sf).
  """
  b_s, b_g = s_goal.shape[0], goals.shape[0]
  if mode == 'goal':
    if task_goal is None:
      hit = jnp.zeros((b_s, b_g), dtype=bool)
    else:
      s_near = jnp.linalg.norm(
          s_goal - task_goal[None, :], axis=-1) < float(tol)  # (B,)
      hit = jnp.broadcast_to(s_near[:, None], (b_s, b_g))
  else:
    # 'sf': pairwise state↔goal hit (covers negatives too)
    diff = s_goal[:, None, :] - goals[None, :, :]
    hit = jnp.linalg.norm(diff, axis=-1) < float(tol)
  return float(scale) * hit.astype(s_goal.dtype)


def make_reward_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
):
  """Factory for the per-step PPO reward used during rollouts.

  Modes (``config.ppo_reward_mode``):
    * ``''`` (default): r = φ(s,a) · ψ(g)
      (or r = φ(s)·ψ(g) when networks were built with state_only / crl_state_only).
    * ``'dirac_target'``: for s ≠ g, r = log(eps) − φ(s0,a)·ψ(s);
      at s = g, r = −φ(s0,a)·ψ(g).  s0 is fixed per episode; a ~ π(·|s).
    * ``'kde_dirac'``: same formula as ``dirac_target`` but the CRL dot
      products are replaced by Gaussian KDE log-densities fitted on replay
      buffer states.  Requires a ``GaussianKDE`` object to be maintained
      externally and passed as the first argument of the returned function.

  Optional ``config.ppo_crl_hit_bonus`` adds an indicator on top of φ·ψ:
  ``'sf'`` → scale·1{‖obs_to_goal(s)−g‖<tol}; ``'goal'`` →
  scale·1{‖obs_to_goal(s)−task_goal‖<tol}.
  """
  mode = (getattr(config, 'ppo_reward_mode', '') or '').strip().lower()
  if mode == 'dirac_target':
    return make_dirac_target_reward_fn(networks, config)
  if mode == 'kde_dirac':
    return make_kde_dirac_reward_fn(config)
  if mode not in ('', 'phi_psi'):
    raise ValueError(
        f'Unknown ppo_reward_mode={config.ppo_reward_mode!r}; '
        f"supported: '', 'phi_psi', 'dirac_target', 'kde_dirac'")
  norm_obs = bool(getattr(config, 'ppo_norm_obs', False))
  obs_dim = int(config.obs_dim)
  si = int(config.start_index)
  ei = int(config.end_index if config.end_index != -1 else obs_dim)
  norm_clip = float(getattr(config, 'ppo_obs_norm_clip', 10.0))
  gidx = getattr(config, 'goal_state_indices', None)
  hit_mode = (getattr(config, 'ppo_crl_hit_bonus', '') or '').strip().lower()
  if hit_mode not in ('', 'sf', 'goal'):
    raise ValueError(
        f'Unknown ppo_crl_hit_bonus={hit_mode!r}; '
        f"supported: '', 'sf', 'goal'")
  hit_tol = float(getattr(config, 'ppo_crl_hit_bonus_tol', 1e-2))
  hit_scale = float(getattr(config, 'ppo_crl_hit_bonus_scale', 1.0))
  _task_goal_raw = getattr(config, 'ppo_crl_hit_bonus_goal', None)
  task_goal = (None if _task_goal_raw is None
               else jnp.asarray(_task_goal_raw, dtype=jnp.float32).reshape(-1))
  if hit_mode == 'goal' and task_goal is None:
    raise ValueError(
        "ppo_crl_hit_bonus='goal' requires config.ppo_crl_hit_bonus_goal")

  @jax.jit
  def reward_fn(q_params: networks_lib.Params,
                obs: jnp.ndarray, action: jnp.ndarray,
                obs_mean: jnp.ndarray, obs_var: jnp.ndarray) -> jnp.ndarray:
    obs = _normalize_packed_obs(
        obs, obs_mean, obs_var, obs_dim=obs_dim, start_index=si,
        end_index=ei, clip=norm_clip, enabled=norm_obs,
        goal_state_indices=gidx)
    _, sa_repr, g_repr = networks.q_network.apply(q_params, obs, action)
    rew = jnp.sum(sa_repr * g_repr, axis=-1)  # (B,)
    if hit_mode:
      s = obs[:, :obs_dim]
      g = obs[:, obs_dim:]
      s_goal = _obs_to_goal_jax(s, si, ei, gidx)
      # Diagonal of the (B,B) hit matrix = per-env bonus for packed (s,g).
      bonus = _crl_hit_bonus_matrix(
          s_goal, g, hit_mode, hit_tol, hit_scale, task_goal)
      rew = rew + jnp.diag(bonus)
    return rew
  return reward_fn


def make_dirac_target_reward_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
):
  """Dirac-target baseline: log(eps) − φ(s0,a)·ψ(s) off-goal, −φ(s0,a)·ψ(g) at g."""
  obs_dim = int(config.obs_dim)
  eps = float(getattr(config, 'ppo_dirac_eps', 1e-6))
  goal_tol = 1e-2

  @jax.jit
  def reward_fn(q_params: networks_lib.Params,
                obs: jnp.ndarray,
                action: jnp.ndarray,
                s0_states: jnp.ndarray) -> jnp.ndarray:
    s = obs[:, :obs_dim]
    g = obs[:, obs_dim:]
    obs_s0 = jnp.concatenate([s0_states, g], axis=-1)
    obs_ss = jnp.concatenate([s, s], axis=-1)
    _, phi_s0, psi_g = networks.q_network.apply(q_params, obs_s0, action)
    _, _, psi_s = networks.q_network.apply(q_params, obs_ss, action)
    dot_ps = jnp.sum(phi_s0 * psi_s, axis=-1)
    dot_pg = jnp.sum(phi_s0 * psi_g, axis=-1)
    at_goal = jnp.linalg.norm(s - g, axis=-1) < goal_tol
    log_rew = jnp.log(jnp.maximum(eps, 1e-10)) - dot_ps
    return jnp.where(at_goal, -dot_pg, log_rew)
  return reward_fn


def make_kde_dirac_reward_fn(config: contrastive_config.ContrastiveConfig):
  """KDE-dirac reward: same sign convention as dirac_target but CRL dot
  products are replaced by Gaussian KDE log-densities from the replay buffer.

  Off-goal:  r = log(eps) − log p_kde(s)
  At goal:   r = −log p_kde(g)

  The returned function signature is::

      reward_fn(kde: GaussianKDE, obs: np.ndarray) -> np.ndarray

  *kde* is a :class:`GaussianKDE` fitted to recent replay-buffer states
  and must be updated by the caller (see ``kde_refit_interval`` config).
  The function operates entirely in numpy so it can be called without JAX.
  """
  obs_dim = int(config.obs_dim)
  eps = float(getattr(config, 'ppo_dirac_eps', 1e-6))
  log_eps = float(np.log(max(eps, 1e-10)))
  goal_tol = 1e-2

  def reward_fn(kde: GaussianKDE,
                obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    s = obs[:, :obs_dim]
    g = obs[:, obs_dim:]
    log_ps = kde.log_density(s)
    log_pg = kde.log_density(g)
    at_goal = np.linalg.norm(s - g, axis=-1) < goal_tol
    log_rew = log_eps - log_ps
    return np.where(at_goal, -log_pg, log_rew)

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
    ent_coef_schedule: Optional[Callable[[jax.Array], jax.Array]] = None,
):
  """Returns a jitted one-minibatch PPO update.

  The update operates on a pytree of trainable params keyed as:
      {'policy': policy_params, 'value': value_params}

  Combined loss (CleanRL-style):
      L  =  pg_loss  -  ent_coef * entropy  +  vf_coef * v_loss

  * pg_loss:   clipped surrogate,  max(-adv*ratio, -adv*clip(ratio))
  * v_loss:    clipped MSE (optional) against `returns`
  * entropy:   single-sample MC estimate  -log π(ã|s) with ã ~ π(·|s)

  Args:
    ent_coef_schedule: if given, an optax-style schedule `step -> ent_coef`
      evaluated on the running SGD-step counter (see `ppo_anneal_ent_coef`);
      `update()` then takes an extra `step` argument. If None (default),
      `ent_coef` is the static `config.ppo_ent_coef` and `update()` keeps its
      original 4-argument signature.

  Returns a function `update(params, opt_state, batch, key[, step])` that
  does one SGD step and returns (new_params, new_opt_state, metrics_dict).
  """
  clip_coef = float(config.ppo_clip_coef)
  vf_coef = float(config.ppo_vf_coef)
  ent_coef_const = float(config.ppo_ent_coef)
  clip_vloss = bool(config.ppo_clip_vloss)
  norm_adv = bool(config.ppo_norm_adv)
  norm_obs = bool(getattr(config, 'ppo_norm_obs', False))
  obs_dim = int(config.obs_dim)
  si = int(config.start_index)
  ei = int(config.end_index if config.end_index != -1 else obs_dim)
  norm_clip = float(getattr(config, 'ppo_obs_norm_clip', 10.0))
  gidx = getattr(config, 'goal_state_indices', None)
  det_select = bool(getattr(config, 'ppo_deterministic_select_dim', False))

  def ppo_loss(params, batch, key, obs_mean, obs_var, step=None):
    ent_coef = (ent_coef_schedule(step) if ent_coef_schedule is not None
                else ent_coef_const)
    network_obs = _normalize_packed_obs(
        batch['obs'], obs_mean, obs_var, obs_dim=obs_dim, start_index=si,
        end_index=ei, clip=norm_clip, enabled=norm_obs,
        goal_state_indices=gidx)
    # ---- policy forward ----
    dist = _policy_dist_maybe_det_select(
        networks.policy_network.apply(params['policy'], network_obs),
        det_select)
    new_logprob = networks.log_prob(dist, batch['actions'])         # (B,)
    # MC-estimate entropy: H(π) ≈ -log π(ã|s), ã ~ π.
    fresh_action = networks.sample(dist, key)
    entropy_est = -networks.log_prob(dist, fresh_action)            # (B,)

    # ---- value forward ----
    new_value = networks.value_network.apply(params['value'], network_obs)  # (B,)

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

    # Pre-tanh Gaussian loc (μ) and scale (σ) from the continuous policy head.
    policy_loc, policy_scale = _policy_normal_loc_scale(dist)

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
        'policy_loc_mean': jnp.mean(policy_loc),
        'policy_loc_abs_mean': jnp.mean(jnp.abs(policy_loc)),
        'policy_scale_mean': jnp.mean(policy_scale),
        'policy_scale_min': jnp.min(policy_scale),
        'ent_coef': jnp.asarray(ent_coef, dtype=jnp.float32),
    }
    return total, metrics

  grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

  if ent_coef_schedule is not None:
    @jax.jit
    def update(params, opt_state, batch, key, obs_mean, obs_var, step):
      (_, metrics), grads = grad_fn(
          params, batch, key, obs_mean, obs_var, step)
      updates, new_opt_state = ppo_optimizer.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, metrics
  else:
    @jax.jit
    def update(params, opt_state, batch, key, obs_mean, obs_var):
      (_, metrics), grads = grad_fn(
          params, batch, key, obs_mean, obs_var)
      updates, new_opt_state = ppo_optimizer.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, metrics

  return update


def make_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    backward: bool = False,
    _return_raw: bool = False,
    config: Optional[contrastive_config.ContrastiveConfig] = None,
):
  """Returns a jitted CRL critic update over one replay batch.

  PPO CRL uses in-batch InfoNCE (softmax cross-entropy on the diagonal) plus
  ``0.01 * logsumexp(logits, axis=1)^2``, matching the CPC path in
  ``learning.py``.  Supports ``(B, B)`` or twin ``(B, B, 2)`` logits.

  Args:
    networks: ContrastiveNetworks (``q_network`` only is used here).
    q_optimizer: Adam (or other) transform for Q / representation params.
    backward: If False (default), use forward InfoNCE — fix anchor (sᵢ,aᵢ),
      treat all goals gⱼ as negatives (denominator = row i of L).
      If True, use backward InfoNCE — fix goal gᵢ, treat all anchors (sⱼ,aⱼ)
      as negatives (denominator = col i of L), implemented by transposing
      the logit matrix before the loss.

  Returns:
    ``update(q_params, q_optimizer_state, batch, key)``
    → ``(new_q_params, new_q_optimizer_state, metrics)``
  """
  _logsumexp_penalty_coef = 0.01
  norm_obs = bool(
      config is not None and getattr(config, 'ppo_norm_obs', False))
  obs_dim = int(config.obs_dim) if config is not None else 0
  si = int(config.start_index) if config is not None else 0
  ei = int(
      config.end_index if config is not None and config.end_index != -1
      else obs_dim)
  norm_clip = float(
      getattr(config, 'ppo_obs_norm_clip', 10.0)
      if config is not None else 10.0)
  gidx = (
      getattr(config, 'goal_state_indices', None)
      if config is not None else None)
  hit_mode = (
      (getattr(config, 'ppo_crl_hit_bonus', '') or '').strip().lower()
      if config is not None else '')
  if hit_mode not in ('', 'sf', 'goal'):
    raise ValueError(
        f'Unknown ppo_crl_hit_bonus={hit_mode!r}; '
        f"supported: '', 'sf', 'goal'")
  hit_tol = float(
      getattr(config, 'ppo_crl_hit_bonus_tol', 1e-2)
      if config is not None else 1e-2)
  hit_scale = float(
      getattr(config, 'ppo_crl_hit_bonus_scale', 1.0)
      if config is not None else 1.0)
  _task_goal_raw = (
      getattr(config, 'ppo_crl_hit_bonus_goal', None)
      if config is not None else None)
  task_goal = (None if _task_goal_raw is None
               else jnp.asarray(_task_goal_raw, dtype=jnp.float32).reshape(-1))
  if hit_mode == 'goal' and task_goal is None:
    raise ValueError(
        "ppo_crl_hit_bonus='goal' requires config.ppo_crl_hit_bonus_goal")
  sf_pert_prob = float(
      getattr(config, 'ppo_crl_sf_perturb_prob', 0.0)
      if config is not None else 0.0)
  sf_pert_eps = float(
      getattr(config, 'ppo_crl_sf_perturb_eps', 1e-2)
      if config is not None else 1e-2)
  if not (0.0 <= sf_pert_prob <= 1.0):
    raise ValueError(
        f'ppo_crl_sf_perturb_prob must be in [0, 1], got {sf_pert_prob}')
  if sf_pert_eps < 0.0:
    raise ValueError(
        f'ppo_crl_sf_perturb_eps must be >= 0, got {sf_pert_eps}')

  def critic_loss(q_params, batch, key, obs_mean, obs_var):
    obs_raw = batch['obs']
    sf_pert_frac = jnp.array(0.0, dtype=obs_raw.dtype)
    sf_pert_norm = jnp.array(0.0, dtype=obs_raw.dtype)
    if sf_pert_prob > 0.0 and sf_pert_eps > 0.0 and obs_dim > 0:
      # Augment future goals s_f = packed goal slice before InfoNCE.
      # With prob p, add δ with ‖δ‖₂ ≤ eps (uniform radius × unit direction).
      key, k_dir, k_rad, k_mask = jax.random.split(key, 4)
      state = obs_raw[:, :obs_dim]
      goal = obs_raw[:, obs_dim:]
      direction = jax.random.normal(k_dir, goal.shape, dtype=goal.dtype)
      direction = direction / jnp.maximum(
          jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
      radius = jax.random.uniform(
          k_rad, (goal.shape[0], 1), dtype=goal.dtype) * sf_pert_eps
      noise = direction * radius
      do_pert = jax.random.bernoulli(
          k_mask, sf_pert_prob, shape=(goal.shape[0],)).astype(goal.dtype)
      noise = noise * do_pert[:, None]
      goal = goal + noise
      obs_raw = jnp.concatenate([state, goal], axis=-1)
      sf_pert_frac = jnp.mean(do_pert)
      sf_pert_norm = jnp.mean(jnp.linalg.norm(noise, axis=-1))
    obs = _normalize_packed_obs(
        obs_raw, obs_mean, obs_var, obs_dim=obs_dim, start_index=si,
        end_index=ei, clip=norm_clip, enabled=norm_obs,
        goal_state_indices=gidx)
    action = batch['action']
    batch_size = obs.shape[0]
    labels = jnp.eye(batch_size)

    logits, _, _ = networks.q_network.apply(q_params, obs, action)

    hit_frac = jnp.array(0.0, dtype=logits.dtype)
    if hit_mode:
      s = obs[:, :obs_dim]
      g = obs[:, obs_dim:]
      s_goal = _obs_to_goal_jax(s, si, ei, gidx)
      bonus = _crl_hit_bonus_matrix(
          s_goal, g, hit_mode, hit_tol, hit_scale, task_goal)
      hit_frac = jnp.mean((bonus > 0).astype(logits.dtype))
      if logits.ndim == 3:
        logits = logits + bonus[:, :, None]
      else:
        logits = logits + bonus

    # Transpose: L^T[i,j] = L[j,i]  →  row i of L^T = col i of L
    # diagonal is unchanged so positive pairs are preserved.
    if backward:
      if logits.ndim == 3:
        logits = jnp.transpose(logits, (1, 0, 2))
      else:
        logits = logits.T

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
        'crl_hit_bonus_frac': hit_frac,
        'crl_sf_perturb_frac': sf_pert_frac,
        'crl_sf_perturb_norm': sf_pert_norm,
    }
    return loss, metrics

  grad_fn = jax.value_and_grad(critic_loss, has_aux=True)

  def update(
      q_params, q_optimizer_state, batch, key,
      obs_mean=None, obs_var=None):
    (_, metrics), grads = grad_fn(
        q_params, batch, key, obs_mean, obs_var)
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


def _action_bin_counts(actions, low, high, n_bins: int):
  """Histogram counts of actions into ``n_bins`` equal-width bins per dim.

  Used only for TD-InfoNCE logging (``a_prime_hist``).
  """
  # actions: (B, A), low/high: (A,)
  span = jnp.maximum(high - low, 1e-8)
  x = (actions - low) / span
  idx = jnp.clip(jnp.floor(x * n_bins).astype(jnp.int32), 0, n_bins - 1)
  # Aggregate over action dims into one hist for compact logging.
  flat = idx.reshape(-1)
  return jnp.bincount(flat, length=n_bins).astype(jnp.float32)


def make_td_infonce_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    obs_dim: int,
    discount: float,
    target_tau: float = 0.995,
    start_index: int = 0,
    end_index: int = -1,
    twin_q: bool = False,
    policy_goal: Optional[np.ndarray] = None,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    n_action_bins: int = 100,
    logsumexp_penalty_coef: float = 0.01,
    goal_state_indices=None,
):
  """TD InfoNCE CRL critic update (Zheng et al. 2023 style).

  Combines a one-step forward InfoNCE term with a bootstrapped TD term that
  uses importance-sampling weights from a slow-moving target network::

      L = (1-γ) * L_InfoNCE(φ·ψ_{g'})  +  γ * L_IS-CE(φ·ψ_randg, w)

  where ``g' = obs_to_goal(s_{t+1})`` (immediate next-state goals, matching
  ``learning.py`` with ``use_td``), and
  ``w[i,j] = softmax_j( min_k target_φ(s_{t+1}_i, a'_i) · target_ψ(randg_j) )``
  are stop-gradient IS weights from the target network; ``rand_g`` is
  ``jnp.roll(obs_to_goal(s_j), -1, axis=0)`` — geometrically sampled future
  goals (same ``s_j`` already in the CRL batch) shifted by one row so every
  sample pairs with a future goal from a *different* batch row.

  The target network is updated inside each call via EMA::

      target_q ← target_tau * target_q + (1 - target_tau) * new_q

  Args:
    networks:    ContrastiveNetworks with ``q_network``, ``policy_network``,
                 and ``sample``.
    q_optimizer: Optimizer for the critic / representation params.
    obs_dim:     State dimension; batch obs = [state(obs_dim) ; goal(*)].
    discount:    γ — weight between term 1 (current) and term 2 (TD).
    target_tau:  EMA rate for the target network.  0 → no memory, 1 → frozen.
    start_index: Goal coordinate slice start (``obs_to_goal``).
    end_index:   Goal coordinate slice end; ``-1`` means ``states[:, start:]``.

  Returns:
    ``(update_policy_boot, update_uniform_boot)`` each with signature
    ``update(q_params, q_opt_state, target_q_params, policy_params, batch, key)``
    → ``(new_q_params, new_q_opt_state, new_target_q_params, metrics_dict)``
  """
  del twin_q  # twin vs single inferred from logit ndim at runtime
  _gidx = None if goal_state_indices is None else jnp.asarray(
      goal_state_indices, dtype=jnp.int32)

  def obs_to_goal(obs):
    if _gidx is not None:
      return obs[:, _gidx]
    if end_index == -1:
      return obs[:, start_index:]
    return obs[:, start_index:end_index]

  _policy_goal = (None if policy_goal is None
                  else jnp.asarray(policy_goal, dtype=jnp.float32))
  if _policy_goal is None:
    raise ValueError('policy_goal is required for TD InfoNCE (env task goal for π bootstrap)')

  _action_low_j = jnp.asarray(
      -1.0 if action_low is None else action_low, dtype=jnp.float32)
  _action_high_j = jnp.asarray(
      1.0 if action_high is None else action_high, dtype=jnp.float32)
  if _action_low_j.ndim == 0:
    _action_low_j = jnp.full((1,), _action_low_j)
  if _action_high_j.ndim == 0:
    _action_high_j = jnp.full((1,), _action_high_j)
  _n_action_bins = int(n_action_bins)

  def critic_loss(
    q_params, target_q_params, policy_params, batch, key,
    use_uniform_bootstrap: bool):
    obs = batch['obs']
    action = batch['action']
    next_obs = batch['next_obs']
    batch_size = obs.shape[0]
    s = obs[:, :obs_dim]
    next_s = next_obs[:, :obs_dim]
    td_g = obs_to_goal(next_s)
    obs_td = jnp.concatenate([s, td_g], axis=1)

    pos_logits, _, _ = networks.q_network.apply(q_params, obs_td, action)
    I = jnp.eye(batch_size)
    # Same logit-scale regularizer as CPC/CRL InfoNCE (learning.py / make_crl).
    # Set coef=0 to disable (e.g. when using repr_norm alone).
    _lse_coef = float(logsumexp_penalty_coef)

    def _ce_with_lse(logits, labels):
      """Softmax CE + coef * logsumexp(logits)^2 to keep φ·ψ from exploding."""
      ce = optax.softmax_cross_entropy(logits=logits, labels=labels)
      if _lse_coef == 0.0:
        return ce
      return ce + _lse_coef * jax.nn.logsumexp(logits, axis=1) ** 2

    # --- necessary: single-Q (2D) vs twin-Q (3D) ---
    if pos_logits.ndim == 3:
      I3 = I[:, :, None].repeat(pos_logits.shape[-1], axis=-1)
      loss1 = jax.vmap(_ce_with_lse, -1, -1)(pos_logits, I3)
    else:
      loss1 = _ce_with_lse(pos_logits, I)

    # term 2 — bootstrap action a' at (s', g_task): uniform random during
    # exploration warmup, else a' ~ π(·|s', g_task).
    env_goal = jnp.broadcast_to(
        _policy_goal[None, :], (batch_size, _policy_goal.shape[0]))
    next_policy_obs = jnp.concatenate([next_s, env_goal], axis=1)
    key, k_boot = jax.random.split(key)
    act_dim = _action_low_j.shape[0]

    def _uniform_a_prime(_):
      return jax.random.uniform(
          k_boot, (batch_size, act_dim),
          minval=_action_low_j, maxval=_action_high_j)

    def _policy_a_prime(_):
      next_dist_params = networks.policy_network.apply(
          policy_params, next_policy_obs)
      return networks.sample(next_dist_params, k_boot)

    next_action = jax.lax.cond(
        use_uniform_bootstrap, _uniform_a_prime, _policy_a_prime, operand=None)
    a_prime_hist = _action_bin_counts(
        next_action, _action_low_j, _action_high_j, _n_action_bins)

    obs_rand = jnp.concatenate([s, batch['random_goal']], axis=1)
    neg_logits, _, _ = networks.q_network.apply(q_params, obs_rand, action)

    next_critic_obs = jnp.concatenate([next_s, batch['random_goal']], axis=1)
    logits_w, _, _ = networks.q_network.apply(
        target_q_params, next_critic_obs, next_action)

    logits_w1 = logits_w[..., 0] if logits_w.ndim == 3 else logits_w
    logits_w2 = logits_w[..., 1] if logits_w.ndim == 3 else logits_w
    # --- necessary: min over twin axis only when twin ---
    logits_w_sc = (jnp.min(logits_w, axis=-1) if logits_w.ndim == 3
                   else logits_w)
    w = jax.nn.softmax(logits_w_sc, axis=1)
    w_diag_vals = jnp.diag(w)

    # Note that we remove the multiplier N for w to balance
    # one term of loss1 with N terms of loss2 in each row.
    w = jax.lax.stop_gradient(w)  # (N, N)
    # --- necessary: twin labels are (B,B,2); single-Q labels are (B,B) ---
    if neg_logits.ndim == 3:
      loss2 = jax.vmap(_ce_with_lse, -1, -1)(
          neg_logits,
          w[:, :, None].repeat(neg_logits.shape[-1], axis=-1)
      )
    else:
      loss2 = _ce_with_lse(neg_logits, w)

    loss = (1 - discount) * loss1 + discount * loss2
    loss = jnp.mean(loss)

    # --- necessary: entropy / accuracy metrics for 2D vs 3D logits ---
    pos_for_ent = (jnp.min(pos_logits, axis=-1) if pos_logits.ndim == 3
                   else pos_logits)
    neg_for_ent = (jnp.min(neg_logits, axis=-1) if neg_logits.ndim == 3
                   else neg_logits)
    logits_pos_entropy = -jnp.sum(
        jax.nn.softmax(pos_for_ent, axis=1)
        * jax.nn.log_softmax(pos_for_ent, axis=1), axis=1)
    logits_neg_entropy = -jnp.sum(
        jax.nn.softmax(neg_for_ent, axis=1)
        * jax.nn.log_softmax(neg_for_ent, axis=1), axis=1)
    logits_w_entropy = -jnp.sum(
        jax.nn.softmax(logits_w_sc, axis=1)
        * jax.nn.log_softmax(logits_w_sc, axis=1), axis=1)

    if pos_logits.ndim == 3:
      logits_pos = jnp.mean(jax.vmap(jnp.diag, -1, -1)(pos_logits))
      logits_pos1 = jnp.mean(jnp.diag(pos_logits[..., 0]))
      logits_pos2 = jnp.mean(jnp.diag(pos_logits[..., 1]))
      logits_neg1 = jnp.mean(neg_logits[..., 0])
      logits_neg2 = jnp.mean(neg_logits[..., 1])
      logits_w1_m = jnp.mean(jnp.diag(logits_w1))
      logits_w2_m = jnp.mean(jnp.diag(logits_w2))
    else:
      logits_pos = jnp.mean(jnp.diag(pos_logits))
      logits_pos1 = logits_pos
      logits_pos2 = logits_pos
      logits_neg1 = jnp.mean(neg_logits)
      logits_neg2 = logits_neg1
      logits_w1_m = jnp.mean(jnp.diag(logits_w_sc))
      logits_w2_m = logits_w1_m

    # Term-1 sanity: mean f(s_i,a_i,s'_i) vs mean_{i≠j} f(s_i,a_i,s'_j).
    # With mix-γ≈0 these should separate if InfoNCE term 1 is learning.
    _b = jnp.asarray(batch_size, dtype=pos_for_ent.dtype)
    pos_diag_mean = jnp.mean(jnp.diag(pos_for_ent))
    pos_offdiag_mean = (
        (jnp.sum(pos_for_ent) - jnp.sum(jnp.diag(pos_for_ent)))
        / jnp.maximum(_b * (_b - 1.0), 1.0))

    metrics = {
        'crl_loss': loss,
        'td_loss1': jnp.mean(loss1),
        'td_loss2': jnp.mean(loss2),
        'categorical_accuracy': jnp.mean(
            jnp.argmax(pos_for_ent, axis=1) == jnp.arange(batch_size)),
        'binary_accuracy': jnp.mean(
            (pos_for_ent > 0) == jnp.eye(batch_size)),
        'pos_diag_mean': pos_diag_mean,
        'pos_offdiag_mean': pos_offdiag_mean,
        'pos_diag_offdiag_gap': pos_diag_mean - pos_offdiag_mean,
        'logsumexp': jnp.mean(jax.nn.logsumexp(pos_for_ent, axis=1) ** 2),
        'logits_pos': logits_pos,
        'logits_pos1': logits_pos1,
        'logits_pos2': logits_pos2,
        'logits_pos_entropy': jnp.mean(logits_pos_entropy),
        'logits_neg': jnp.mean(neg_for_ent),
        'logits_neg1': logits_neg1,
        'logits_neg2': logits_neg2,
        'logits_neg_entropy': jnp.mean(logits_neg_entropy),
        'logits_w1': logits_w1_m,
        'logits_w2': logits_w2_m,
        'w_diag': jnp.mean(w_diag_vals),
        'w_diag_std': jnp.std(w_diag_vals),
        'w_diag_min': jnp.min(w_diag_vals),
        'w_diag_max': jnp.max(w_diag_vals),
        'a_prime_mean': jnp.mean(next_action),
        'a_prime_std': jnp.std(next_action),
        'a_prime_hist': a_prime_hist,
        'w': jnp.mean(w),
        'logits_w_entropy': jnp.mean(logits_w_entropy),
    }

    return loss, metrics     # column goal pool (hindsight / random_goal)

  # --- necessary: was nested under critic_loss after `return` (dead code) ---
  def _make_jitted_update(use_uniform_bootstrap: bool):
    def _loss(q_params, target_q_params, policy_params, batch, key):
      return critic_loss(
          q_params, target_q_params, policy_params, batch, key,
          use_uniform_bootstrap)

    grad_fn = jax.value_and_grad(_loss, has_aux=True)

    @jax.jit
    def update(q_params, q_opt_state, target_q_params, policy_params, batch, key):
      (_, metrics), grads = grad_fn(
          q_params, target_q_params, policy_params, batch, key)

      grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
          jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))))
      loss_finite  = jnp.isfinite(metrics['crl_loss'])
      do_update    = jnp.logical_and(grads_finite, loss_finite)

      def _apply(_):
        updates, new_opt_state = q_optimizer.update(grads, q_opt_state, q_params)
        return optax.apply_updates(q_params, updates), new_opt_state

      def _skip(_):
        return q_params, q_opt_state

      new_q_params, new_opt_state = jax.lax.cond(
          do_update, _apply, _skip, operand=None)

      new_target = jax.tree_util.tree_map(
          lambda t, o: target_tau * t + (1.0 - target_tau) * o,
          target_q_params, new_q_params)

      metrics = dict(metrics)
      metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
      return new_q_params, new_opt_state, new_target, metrics

    return update

  return _make_jitted_update(False), _make_jitted_update(True)


def make_scan_td_infonce_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    obs_dim: int,
    discount: float,
    target_tau: float = 0.995,
    start_index: int = 0,
    end_index: int = -1,
    policy_goal: Optional[np.ndarray] = None,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    logsumexp_penalty_coef: float = 0.01,
    goal_state_indices=None,
    repr_tau: float = 0.0,
):
  """Scan-based TD-InfoNCE updater: N critic steps in one JIT call.

  Batches must already include ``random_goal`` (rolled future goals).
  Reward-param EMA (``0 < repr_tau < 1``) is applied inside the scan.

  Returns:
    ``multi_update(q_params, q_opt_state, target_q, params_ema, batches,
                   key, policy_params)``
    → ``(new_q, new_opt, new_target, new_ema, new_key, mean_metrics)``
  """
  raw_update, _ = make_td_infonce_update_fn(
      networks, q_optimizer,
      obs_dim=obs_dim,
      discount=discount,
      target_tau=target_tau,
      start_index=start_index,
      end_index=end_index,
      twin_q=False,
      policy_goal=policy_goal,
      action_low=action_low,
      action_high=action_high,
      logsumexp_penalty_coef=logsumexp_penalty_coef,
      goal_state_indices=goal_state_indices,
  )
  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def multi_update(
      q_params, q_opt_state, target_q, params_ema, batches, key,
      policy_params):
    def scan_step(carry, batch):
      q_p, opt, tgt, ema, k = carry
      k, k_u = jax.random.split(k)
      q_p, opt, tgt, m = raw_update(
          q_p, opt, tgt, policy_params, batch, k_u)
      if use_ema:
        ema = jax.tree_util.tree_map(
            lambda t, o: _tau * t + (1.0 - _tau) * o, ema, q_p)
      else:
        ema = q_p
      return (q_p, opt, tgt, ema, k), m

    (q_params, q_opt_state, target_q, params_ema, key), metrics = (
        jax.lax.scan(
            scan_step,
            (q_params, q_opt_state, target_q, params_ema, key),
            batches))
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return q_params, q_opt_state, target_q, params_ema, key, metrics

  return multi_update


def make_scan_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    backward: bool = False,
    repr_tau: float = 0.0,
    config: Optional[contrastive_config.ContrastiveConfig] = None,
):
  """Scan-based CRL updater: runs N steps in one JIT call.

  Caller pre-samples all N batches in NumPy, stacks them into
  ``(N, B, dim)`` arrays, and transfers to GPU once.  A single
  ``jax.lax.scan`` call replaces N separate dispatch-and-sync rounds.

  The φ/ψ EMA for the PPO reward (``repr_tau > 0``) is also updated
  inside the scan, eliminating the 128 un-JIT-compiled ``_ema_tree``
  calls per iteration.

  Returns:
    ``multi_update(q_params, q_opt_state, q_params_ema, batches, key)``
    → ``(new_q_params, new_q_opt_state, new_q_params_ema, new_key,
         mean_metrics)``
    where ``batches`` is a dict of ``(N, B, dim)`` JAX arrays.
  """
  _, raw_update = make_crl_update_fn(
      networks, q_optimizer, backward=backward, _return_raw=True,
      config=config)

  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def multi_update(
      q_params, q_opt_state, q_params_ema, batches, key, obs_mean, obs_var):
    def scan_step(carry, batch):
      q_p, q_opt, q_ema, k = carry
      k, k_crl = jax.random.split(k)
      q_p, q_opt, m = raw_update(
          q_p, q_opt, batch, k_crl, obs_mean, obs_var)
      if use_ema:
        q_ema = jax.tree_util.tree_map(
            lambda t, o: _tau * t + (1.0 - _tau) * o, q_ema, q_p)
      else:
        q_ema = q_p
      return (q_p, q_opt, q_ema, k), m

    (q_params, q_opt_state, q_params_ema, key), metrics = jax.lax.scan(
        scan_step, (q_params, q_opt_state, q_params_ema, key), batches)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return q_params, q_opt_state, q_params_ema, key, metrics

  return multi_update


# ---------------------------------------------------------------------------
# Rollout collection (env stepping — runs in host/numpy, not jitted).
# ---------------------------------------------------------------------------
class VecEnv:
  """A tiny synchronous wrapper around N acme/dm_env environments.

  Uses dm_env's step/reset interface (not gym's) to stay compatible with
  contrastive_utils.make_environment.  Exposes a CleanRL-like API:

      obs = vec_env.reset()                                    # (E, obs_dim_total)
      next_obs, env_rew, dones, terminal_obs, info_rew = vec_env.step(actions)

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
    # Thread pool: MuJoCo C extensions release the GIL during sim.step(),
    # so threads can run env physics in parallel.
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=num_envs,
        thread_name_prefix='vecenv_worker')

  def reset(self) -> np.ndarray:
    obs = np.stack([e.reset().observation for e in self._envs], axis=0)
    return obs.astype(np.float32)

  def step(
      self, actions: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_dtype = self._action_spec.dtype
    shape = self._obs_spec.shape
    next_obs    = np.zeros((self._num_envs,) + shape, dtype=np.float32)
    terminal_obs = np.zeros((self._num_envs,) + shape, dtype=np.float32)
    env_rewards  = np.zeros(self._num_envs, dtype=np.float32)
    info_rewards = np.full(self._num_envs, np.nan, dtype=np.float32)
    dones        = np.zeros(self._num_envs, dtype=bool)

    def _step_one(i: int):
      env = self._envs[i]
      a = np.asarray(actions[i], dtype=np.float32)
      a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
      if hasattr(self._action_spec, 'minimum') and hasattr(self._action_spec, 'maximum'):
        a = np.clip(a, self._action_spec.minimum, self._action_spec.maximum)
      ts = env.step(a.astype(action_dtype))
      env_r  = 0.0 if ts.reward is None else float(ts.reward)
      info_r = extract_info_reward(env)
      term   = ts.observation.copy()
      done   = bool(ts.last())
      nxt    = env.reset().observation.copy() if done else ts.observation.copy()
      return env_r, info_r, term, nxt, done

    futures = [self._executor.submit(_step_one, i) for i in range(self._num_envs)]
    for i, fut in enumerate(futures):
      env_r, info_r, term, nxt, done = fut.result()
      env_rewards[i]  = env_r
      info_rewards[i] = info_r
      terminal_obs[i] = term
      next_obs[i]     = nxt
      dones[i]        = done

    return next_obs, env_rewards, dones, terminal_obs, info_rewards

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
def _tree_copy(params):
  return jax.tree_util.tree_map(lambda x: x, params)


def _merge_policy_opt_state(old_opt_state, fresh_opt_state):
  """Keep value-network Adam state; replace policy branches from ``fresh``."""

  def _pick(path, old, fresh):
    for p in path:
      key = getattr(p, 'key', None)
      if key == 'policy':
        return fresh
      if key == 'value':
        return old
    return old

  return jax.tree_util.tree_map_with_path(
      _pick, old_opt_state, fresh_opt_state)


def _ema_tree(target, online, tau: float):
  """target ← τ·target + (1−τ)·online elementwise over a param pytree."""
  tau = float(tau)
  return jax.tree_util.tree_map(
      lambda t, o: tau * t + (1.0 - tau) * o, target, online)


def _save_checkpoint(path: str,
                     policy_params, value_params, q_params,
                     ppo_opt_state, q_opt_state,
                     iteration: int, global_step: int, key,
                     q_params_ema=None,
                     td3_policy_target=None,
                     extra_state=None):
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
  if q_params_ema is not None:
    ckpt['q_params_ema'] = q_params_ema
  if td3_policy_target is not None:
    ckpt['td3_policy_target'] = td3_policy_target
  if extra_state is not None:
    ckpt['extra_state'] = extra_state
  tmp_path = path + '.tmp'
  with open(tmp_path, 'wb') as fh:
    _pkl.dump(ckpt, fh, protocol=_pkl.HIGHEST_PROTOCOL)
  _os.replace(tmp_path, path)


def _prune_old_checkpoints(ckpt_dir: str, keep_last: int):
  """Delete oldest ckpt_iter_*.pkl until at most `keep_last` remain.

  No-op when ``keep_last <= 0`` (retain every milestone checkpoint).
  """
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


def _bb_ep_metrics_from_eval_steps(
    steps,
    obs_dim: int,
    start_index: int,
    end_index: int,
    episode_length: int,
    goal_state_indices=None,
) -> list:
  """Per-env eval metrics from a BuilderBench GPU eval scan.

  When ``goal_state_indices`` is set (masked creative tasks), gather those
  state coords so distances match the (shorter) goal vector. Contiguous
  ``start_index:end_index`` alone is wrong for non-contiguous masks — e.g.
  creative-5-task3 state xyz is 15-D while goals are 9-D.
  """
  del obs_dim  # kept for call-site compat; width comes from goals / indices
  rewards = np.asarray(steps['reward'], dtype=np.float32)
  success = np.asarray(steps['success'], dtype=np.float32)
  state_obs = np.asarray(steps['state_obs'], dtype=np.float32)
  goals = np.asarray(steps['goal'], dtype=np.float32)
  if goal_state_indices is not None:
    gidx = np.asarray(goal_state_indices, dtype=np.int32).reshape(-1)
    state_slice = state_obs[..., gidx]
    goal_slice = goals[..., : int(gidx.shape[0])]
  else:
    ei = int(goals.shape[-1] if end_index == -1 else end_index)
    si = int(start_index)
    gdim = int(goals.shape[-1])
    # Clamp to goal width so a too-wide end_index cannot crash dist metrics.
    state_slice = state_obs[..., si:si + gdim]
    goal_slice = goals[..., :gdim]
    if state_slice.shape[-1] != goal_slice.shape[-1]:
      raise ValueError(
          'eval dist slice mismatch: '
          f'state{state_slice.shape} vs goal{goal_slice.shape} '
          f'(start_index={si}, end_index={ei}, goal_dim={gdim})')
  dists = np.linalg.norm(state_slice - goal_slice, axis=-1)

  ep_metrics_list = []
  for i in range(rewards.shape[1]):
    ep_metrics_list.append({
        'episode_return': float(rewards[:, i].sum()),
        'episode_length': int(episode_length),
        'success': float(np.max(success[:, i]) >= 0.5),
        'init_dist': float(dists[0, i]),
        'final_dist': float(dists[-1, i]),
        'delta_dist': float(dists[0, i] - dists[-1, i]),
        'min_dist': float(np.min(dists[:, i])),
    })
  return ep_metrics_list


def _smooth_bb_eval_metrics(
    ep_metrics_list: list,
    success_obs,
    dist_obs,
) -> list:
  """Attach running success_1000 / dist_* smoothers (matches observer semantics)."""
  smoothed = []
  for ep_m in ep_metrics_list:
    success_obs._success.append(bool(ep_m['success'] >= 0.5))
    ep_out = dict(ep_m)
    ep_out['success_1000'] = float(np.mean(success_obs._success[-1000:]))
    for key in ('init_dist', 'final_dist', 'delta_dist', 'min_dist'):
      dist_obs._history.setdefault(key, []).append(float(ep_m[key]))
    if dist_obs._smooth:
      for key, vec in dist_obs._history.items():
        for size in (10, 100, 1000):
          ep_out[f'{key}_{size}'] = float(np.nanmean(vec[-size:]))
    smoothed.append(ep_out)
  return smoothed


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
    builderbench_kwargs: Optional[Dict[str, Any]] = None,
    fixed_start_end: Optional[Any] = None,
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
  _env_name = str(getattr(config, 'env_name', '') or '')
  _use_jax_bb_vec = _env_name.startswith('builderbench_')
  if _use_jax_bb_vec:
    import importlib.util as _ilu
    _jax_vec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'envs', 'builderbench_jax_vec.py')
    _spec = _ilu.spec_from_file_location('builderbench_jax_vec', _jax_vec_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    JaxBuilderBenchVecEnv = _mod.JaxBuilderBenchVecEnv
    _bb_kw = dict(builderbench_kwargs or {})
    vec_env = JaxBuilderBenchVecEnv(
        env_name=_env_name,
        num_envs=int(config.ppo_num_envs),
        seed=int(seed * 31),
        use_pd=bool(_bb_kw.get('builderbench_use_pd', False)),
        pd_duration=int(_bb_kw.get('builderbench_pd_duration', 5)),
        pd_filter_policy_obs=bool(
            _bb_kw.get('builderbench_pd_filter_policy_obs', True)),
        fixed_target_goal=_bb_kw.get('fixed_target_goal'),
        permute_start_boxes=bool(
            _bb_kw.get('builderbench_permute_start_boxes', True)),
        mj_episode_length=_bb_kw.get('builderbench_mj_episode_length'),
        fixed_start_x=_bb_kw.get('builderbench_fixed_start_x'),
        success_terminate_steps=int(
            _bb_kw.get('builderbench_success_terminate_steps', 0) or 0),
    )
    print(f'[ppo] using JAX-batched BuilderBench vec env '
          f'(E={config.ppo_num_envs})')
  else:
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

  # ---- density estimator mode ------------------------------------------
  # 'crl'      (default) — φ(s,a)·ψ(g) contrastive representations.
  # 'gaussian' — diagonal Gaussian p_θ(g|s,a);  reward = log p_θ(g|s_t,a_t).
  # 'nf'       — conditional RealNVP  log p_NF(g|s,a)
  #              (or log p_NF(g|s) when nf_state_only=True).
  # 'fm'       — OT flow-matching; reward = log p_FM(g|s,a) via reverse ODE.
  # 'td3'      — twin Q(s,a,s_f) TD3-style; reward = Q1(s,a,g).
  repr_mode    = (getattr(config, 'ppo_repr_mode', 'crl') or 'crl').strip().lower()
  use_gaussian = repr_mode == 'gaussian'
  use_nf       = repr_mode == 'nf'
  use_fm       = repr_mode == 'fm'
  use_td3      = repr_mode == 'td3'
  use_td_infonce = repr_mode in ('tdinfonce', 'td_infonce')
  use_crl_td3_switch = repr_mode == 'crl_td3_switch'
  if use_crl_td3_switch and not _use_jax_bb_vec:
    raise ValueError(
        'crl_td3_switch currently requires a BuilderBench environment so '
        'hard-goal visits can be counted from its success signal')
  if repr_mode not in (
      'crl', 'gaussian', 'nf', 'fm', 'td3', 'crl_td3_switch',
      'tdinfonce', 'td_infonce'):
    raise ValueError(f'Unknown ppo_repr_mode={repr_mode!r}')
  if use_td_infonce and bool(getattr(config, 'twin_q', False)):
    raise ValueError(
        'ppo_repr_mode=tdinfonce currently requires twin_q=False '
        '(single critic); set twin_q=False or turn twin support on later')
  _switch_goal_visits = int(
      getattr(config, 'ppo_reward_switch_goal_visits', 5))
  if use_crl_td3_switch and _switch_goal_visits <= 0:
    raise ValueError(
        'ppo_reward_switch_goal_visits must be > 0 in crl_td3_switch mode')
  _switch_blend_iters = int(
      getattr(config, 'ppo_reward_switch_blend_iters', 0))
  if use_crl_td3_switch and _switch_blend_iters < 0:
    raise ValueError(
        'ppo_reward_switch_blend_iters must be >= 0 in crl_td3_switch mode')
  density_nets    = None
  nf_density_nets = None
  fm_density_nets = None
  td3_density_nets = None

  obs_dim_cfg   = int(config.obs_dim)
  total_obs_dim = int(np.prod(obs_shape))
  goal_dim_cfg  = total_obs_dim - obs_dim_cfg
  act_dim_cfg   = int(np.prod(act_shape))
  norm_obs = bool(getattr(config, 'ppo_norm_obs', False))
  obs_norm_clip = float(getattr(config, 'ppo_obs_norm_clip', 10.0))
  norm_si = int(config.start_index)
  norm_ei = int(
      config.end_index if int(config.end_index) != -1 else obs_dim_cfg)
  _goal_state_indices = getattr(config, 'goal_state_indices', None)
  if _goal_state_indices is not None:
    _goal_state_indices = tuple(int(x) for x in _goal_state_indices)
    if len(_goal_state_indices) != goal_dim_cfg:
      raise ValueError(
          'goal_state_indices length must equal goal_dim; '
          f'got len={len(_goal_state_indices)}, goal_dim={goal_dim_cfg}')
    if (min(_goal_state_indices) < 0
        or max(_goal_state_indices) >= obs_dim_cfg):
      raise ValueError(
          'goal_state_indices must lie in [0, obs_dim); '
          f'got {_goal_state_indices}, obs_dim={obs_dim_cfg}')
  _configured_reward_mode = (
      getattr(config, 'ppo_reward_mode', '') or '').strip().lower()
  if norm_obs:
    if _goal_state_indices is None:
      if not 0 <= norm_si < norm_ei <= obs_dim_cfg:
        raise ValueError(
            'ppo_norm_obs requires 0 <= start_index < end_index <= obs_dim; '
            f'got start_index={norm_si}, end_index={norm_ei}, '
            f'obs_dim={obs_dim_cfg}')
      if goal_dim_cfg != norm_ei - norm_si:
        raise ValueError(
            'ppo_norm_obs maps each goal coordinate to state stats from '
            'start_index:end_index, so dimensions must match; '
            f'goal_dim={goal_dim_cfg}, state slice={norm_ei - norm_si}')
      _goal_stats_msg = f'state[{norm_si}:{norm_ei}]'
    else:
      _goal_stats_msg = f'state[goal_state_indices]={_goal_state_indices}'
    if not np.isfinite(obs_norm_clip) or obs_norm_clip <= 0.0:
      raise ValueError(
          f'ppo_obs_norm_clip must be finite and > 0, got {obs_norm_clip}')
    if (use_gaussian or use_nf or use_fm
        or _configured_reward_mode not in ('', 'phi_psi')):
      raise ValueError(
          'ppo_norm_obs currently supports ppo_repr_mode=crl, td3, '
          'tdinfonce, or crl_td3_switch with ppo_reward_mode="" or '
          '"phi_psi"; Gaussian, NF, FM, dirac_target, and kde_dirac have '
          'separate density-coordinate semantics and are intentionally '
          'rejected.')
    print(f'[ppo] observation normalization enabled: state_dim={obs_dim_cfg}, '
          f'goal_stats={_goal_stats_msg}, clip={obs_norm_clip}; '
          'rollout/replay storage remains raw')
  else:
    print('[ppo] observation normalization disabled')
  if _goal_state_indices is not None:
    print(f'[ppo] LHER obs_to_goal uses goal_state_indices='
          f'{_goal_state_indices} (goal_dim={goal_dim_cfg})')
  # Zero pseudo-count plus mean=0/var=1 makes iteration zero exactly identity.
  obs_rms = RunningMeanStd(shape=(obs_dim_cfg,), epsilon=0.0)

  if use_gaussian:
    density_nets = _gd.make_gaussian_density_networks(
        obs_dim=obs_dim_cfg,
        act_dim=act_dim_cfg,
        goal_dim=goal_dim_cfg,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )
    print(f'[ppo] repr_mode=gaussian  obs_dim={obs_dim_cfg}  '
          f'act_dim={act_dim_cfg}  goal_dim={goal_dim_cfg}  '
          f'hidden_layers={config.hidden_layer_sizes}')
  elif use_fm:
    _fm_flow_steps = int(getattr(config, 'fm_flow_steps', 10))
    _fm_logp_mode = (
        getattr(config, 'fm_logp_mode', 'exact') or 'exact').strip().lower()
    _fm_hutch_probes = int(getattr(config, 'fm_hutch_probes', 8))
    _fm_layer_norm = bool(getattr(config, 'fm_layer_norm', False))
    fm_density_nets = _fm.make_fm_density_networks(
        obs_dim=obs_dim_cfg,
        act_dim=act_dim_cfg,
        goal_dim=goal_dim_cfg,
        hidden_layer_sizes=config.hidden_layer_sizes,
        flow_steps=_fm_flow_steps,
        layer_norm=_fm_layer_norm,
    )
    print(f'[ppo] repr_mode=fm (OT flow matching)  obs_dim={obs_dim_cfg}  '
          f'act_dim={act_dim_cfg}  goal_dim={goal_dim_cfg}  '
          f'hidden_layers={config.hidden_layer_sizes}  '
          f'flow_steps={_fm_flow_steps}  logp_mode={_fm_logp_mode}  '
          f'velocity=ResidualMLP')
  elif use_nf:
    nf_rep_size      = int(getattr(config, 'nf_rep_size', 64))
    nf_num_blocks    = int(getattr(config, 'nf_num_blocks', 8))
    nf_coupling_w    = int(getattr(config, 'nf_coupling_width', 256))
    nf_goal_enc_size = int(getattr(config, 'nf_goal_enc_size', 0))
    nf_sa_hidden     = int(getattr(config, 'nf_sa_hidden', 1024))
    nf_sa_num_layers = int(getattr(config, 'nf_sa_num_layers', 4))
    nf_state_only    = bool(getattr(config, 'nf_state_only', False))
    nf_density_nets = _nf.make_nf_density_networks(
        obs_dim=obs_dim_cfg,
        act_dim=act_dim_cfg,
        goal_dim=goal_dim_cfg,
        hidden_layer_sizes=config.hidden_layer_sizes,
        rep_size=nf_rep_size,
        num_blocks=nf_num_blocks,
        channels=nf_coupling_w,
        goal_enc_size=nf_goal_enc_size,
        sa_hidden=nf_sa_hidden,
        sa_num_layers=nf_sa_num_layers,
        state_only=nf_state_only,
    )
    _goal_enc_desc = (f'goal_encoder=2x256+swish→{nf_goal_enc_size}'
                      if nf_goal_enc_size > 0 else 'goal_encoder=none (raw goal)')
    _cond_desc = 'p(g|s) / r(s)' if nf_state_only else 'p(g|s,a) / r(s,a)'
    print(f'[ppo] repr_mode=nf (RealNVP)  obs_dim={obs_dim_cfg}  '
          f'act_dim={act_dim_cfg}  goal_dim={goal_dim_cfg}  '
          f'rep_size={nf_rep_size}  num_blocks={nf_num_blocks}  '
          f'coupling_width={nf_coupling_w}  flow_dim={nf_density_nets.flow_dim}  '
          f'sa_encoder={nf_sa_num_layers}x{nf_sa_hidden}+swish  '
          f'state_only={nf_state_only} ({_cond_desc})  {_goal_enc_desc}')
  elif use_td3 or use_crl_td3_switch:
    _td3_bilinear = bool(getattr(config, 'ppo_td3_bilinear', False))
    td3_density_nets = _td3.make_td3_density_networks(
        obs_dim=obs_dim_cfg,
        act_dim=act_dim_cfg,
        goal_dim=goal_dim_cfg,
        hidden_layer_sizes=config.hidden_layer_sizes,
        repr_dim=int(config.repr_dim),
        bilinear=_td3_bilinear,
        repr_norm=bool(config.repr_norm) if _td3_bilinear else False,
    )
    _td3_tau_cfg = float(getattr(config, 'ppo_td3_tau', -1.0))
    _td3_tau = (_td3_tau_cfg if _td3_tau_cfg >= 0.0
                else float(config.tau))
    _q_param = (
        f'bilinear x(s,a)·y(g) repr_dim={config.repr_dim}'
        if _td3_bilinear else 'mlp([s;g;a])')
    _mode_desc = (
        f'crl_td3_switch (CRL→TD3 after {_switch_goal_visits} goal visits)'
        if use_crl_td3_switch else 'td3')
    print(f'[ppo] repr_mode={_mode_desc} (twin Q(s,a,s_f))  obs_dim={obs_dim_cfg}  '
          f'act_dim={act_dim_cfg}  goal_dim={goal_dim_cfg}  '
          f'hidden_layers={config.hidden_layer_sizes}  '
          f'q_param={_q_param}  '
          f'target_tau={_td3_tau} (≠ ppo_crl_repr_tau)  '
          f'goal_tol={getattr(config, "ppo_td3_goal_tol", 1e-2)}  '
          f'cross_batch_goals={getattr(config, "ppo_td3_cross_batch_goals", False)}  '
          f'discount={config.discount}', flush=True)
  else:
    if use_td_infonce:
      print(f'[ppo] repr_mode=tdinfonce (TD InfoNCE φ·ψ, single Q)  '
            f'obs_dim={obs_dim_cfg}  discount={config.discount}', flush=True)
    else:
      print(f'[ppo] repr_mode=crl (φ·ψ contrastive)', flush=True)

  # ---- init params ------------------------------------------------------
  print('[ppo] init: policy/value/density params...', flush=True)
  key = jax.random.PRNGKey(seed)
  k_pol, k_val, k_q, key = jax.random.split(key, 4)
  policy_params = networks.policy_network.init(k_pol)
  print('[ppo] init: policy done', flush=True)
  value_params = networks.value_network.init(k_val)
  print('[ppo] init: value done', flush=True)
  # q_params holds the density-estimator params in all modes:
  #   crl mode      → CRL (φ, ψ) contrastive params from networks.q_network
  #   gaussian mode → Gaussian density p_θ(g|s,a) params from density_nets
  #   nf mode       → RealNVP log p_NF(g|s,a) params from nf_density_nets
  #   fm mode       → velocity field v_θ(s,a,x_t,t) params from fm_density_nets
  #   td3 mode      → twin Q + Polyak targets from td3_density_nets
  if use_gaussian:
    q_params = density_nets.density_net.init(k_q)
  elif use_fm:
    q_params = _fm.init_fm_params(fm_density_nets, k_q)
  elif use_nf:
    q_params = _nf.init_nf_params(nf_density_nets, k_q)
  elif use_td3:
    q_params = _td3.init_td3_params(td3_density_nets, k_q)
    print('[ppo] init: td3 density done', flush=True)
  else:
    q_params = networks.q_network.init(k_q)
  hybrid_td3_params = None
  if use_crl_td3_switch:
    key, k_hybrid_td3 = jax.random.split(key)
    hybrid_td3_params = _td3.init_td3_params(
        td3_density_nets, k_hybrid_td3)
  ppo_params = {'policy': policy_params, 'value': value_params}

  # ---- optimizers -------------------------------------------------------
  # Total PPO SGD steps over the whole run; used by both the LR anneal and
  # the (optional) entropy-coefficient anneal below.
  print('[ppo] init: optimizers...', flush=True)
  total_ppo_updates = (
      num_iterations
      * int(config.ppo_num_epochs)
      * int(config.ppo_num_minibatches)
  )

  if config.ppo_anneal_lr:
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
  print('[ppo] init: ppo_opt done', flush=True)

  # Optional linear anneal of the PPO entropy-bonus coefficient, mirroring
  # the LR schedule above. Off unless --ppo_anneal_ent_coef is set.
  ent_coef_schedule = None
  if bool(getattr(config, 'ppo_anneal_ent_coef', False)):
    ent_coef_schedule = optax.linear_schedule(
        init_value=float(config.ppo_ent_coef),
        end_value=float(getattr(config, 'ppo_ent_coef_final', 0.0)),
        transition_steps=max(1, total_ppo_updates),
    )
    print(f'[ppo] entropy anneal: ent_coef {config.ppo_ent_coef} -> '
          f'{getattr(config, "ppo_ent_coef_final", 0.0)} over '
          f'{total_ppo_updates} SGD steps')

  if use_nf:
    q_optimizer = _nf.make_nf_optimizers(
        encoder_lr=float(getattr(config, 'nf_encoder_lr', 3e-4)),
        critic_lr=float(getattr(config, 'nf_critic_lr', 1e-4)),
        critic_weight_decay=float(getattr(config, 'nf_critic_weight_decay', 1e-6)),
        grad_clip=float(getattr(config, 'nf_grad_clip', 1.0)),
        has_goal_encoder=(nf_density_nets.goal_encoder_net is not None),
    )
    _ge_opt_desc = (f', goal_encoder Adam(lr={config.nf_encoder_lr})'
                    if nf_density_nets.goal_encoder_net is not None else '')
    print(f'[ppo] NF optimizers: SA encoder Adam(lr={config.nf_encoder_lr})'
          f'{_ge_opt_desc}, '
          f'flow AdamW(lr={config.nf_critic_lr}, wd={config.nf_critic_weight_decay}), '
          f'grad_clip={getattr(config, "nf_grad_clip", 1.0)}')
  else:
    q_optimizer = optax.adam(float(config.learning_rate))
  if use_td3:
    # Optimizer tracks online Q1/Q2 only; targets are Polyak-updated outside Adam.
    q_opt_state = q_optimizer.init(_td3.online_td3_params(q_params))
  else:
    q_opt_state = q_optimizer.init(q_params)
  print('[ppo] init: q_opt done', flush=True)
  hybrid_td3_opt_state = None
  if use_crl_td3_switch:
    hybrid_td3_opt_state = q_optimizer.init(
        _td3.online_td3_params(hybrid_td3_params))

  # ---- optional EMA of reward-network params ----------------------------
  # CRL / TD-InfoNCE: EMA of φ, ψ for r = φ·ψ (ppo_crl_repr_tau).
  # NF:  EMA of flow/encoder params for r = log p_NF (ppo_nf_reward_tau).
  # Gaussian: EMA of density params for r = log p_θ (ppo_gaussian_reward_tau).
  # FM:  EMA of velocity-field params for r = log p_FM (ppo_fm_reward_tau).
  # TD3: EMA of twin-Q params for r = Q1(s,a,g) or log((1−γ)Q1)
  #       (ppo_td3_reward_tau).
  # Density / critic training always uses online params; only PPO reward
  # reads the EMA copy.  Independent of TD3 Polyak target τ (ppo_td3_tau)
  # and of TD-InfoNCE target keep-rate (ppo_td_infonce_target_tau).
  if use_nf:
    _repr_tau = float(getattr(config, 'ppo_nf_reward_tau', 0.0))
    _use_repr_ema = 0.0 < _repr_tau < 1.0
  elif use_gaussian:
    _repr_tau = float(getattr(config, 'ppo_gaussian_reward_tau', 0.0))
    _use_repr_ema = 0.0 < _repr_tau < 1.0
  elif use_fm:
    _repr_tau = float(getattr(config, 'ppo_fm_reward_tau', 0.0))
    _use_repr_ema = 0.0 < _repr_tau < 1.0
  elif use_td3:
    _repr_tau = float(getattr(config, 'ppo_td3_reward_tau', 0.0))
    _use_repr_ema = 0.0 < _repr_tau < 1.0
  else:
    # crl and tdinfonce both reward with φ·ψ → same reward EMA knob
    _repr_tau = float(getattr(config, 'ppo_crl_repr_tau', 0.0))
    _use_repr_ema = 0.0 < _repr_tau < 1.0
  q_params_reward = _tree_copy(q_params) if _use_repr_ema else q_params
  # TD-InfoNCE slow target critic (separate from reward EMA).
  td_infonce_target_q = (
      _tree_copy(q_params) if use_td_infonce else None)
  # TD-NF slow target density (separate from reward EMA / ppo_nf_reward_tau).
  use_nf_td = use_nf and bool(getattr(config, 'ppo_nf_td', False))
  nf_td_target = _tree_copy(q_params) if use_nf_td else None
  nf_scan_td_update = None
  _hybrid_td3_reward_tau = float(
      getattr(config, 'ppo_td3_reward_tau', 0.0))
  _use_hybrid_td3_reward_ema = (
      use_crl_td3_switch and 0.0 < _hybrid_td3_reward_tau < 1.0)
  hybrid_td3_params_reward = (
      _tree_copy(hybrid_td3_params)
      if _use_hybrid_td3_reward_ema else hybrid_td3_params)

  # ---- resume from checkpoint if one exists -----------------------------
  start_iteration = 0
  global_step = 0
  ppo_sgd_step = 0
  _td3_policy_target_from_ckpt = None
  hard_goal_visit_count = 0
  reward_switched_to_td3 = False
  reward_switch_iteration = -1
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
      if _use_repr_ema:
        q_params_reward = (_ckpt['q_params_ema']
                           if 'q_params_ema' in _ckpt
                           else _tree_copy(q_params))
      _extra = _ckpt.get('extra_state', {})
      if use_td_infonce and 'td_infonce_target_q' in _extra:
        td_infonce_target_q = _extra['td_infonce_target_q']
      elif use_td_infonce:
        td_infonce_target_q = _tree_copy(q_params)
      if use_nf_td and 'nf_td_target' in _extra:
        nf_td_target = _extra['nf_td_target']
      elif use_nf_td:
        nf_td_target = _tree_copy(q_params)
      if norm_obs and 'obs_rms' in _extra:
        _obs_state = _extra['obs_rms']
        obs_rms.mean = np.asarray(
            _obs_state.get('mean', obs_rms.mean), dtype=np.float64)
        obs_rms.var = np.asarray(
            _obs_state.get('var', obs_rms.var), dtype=np.float64)
        obs_rms.count = float(_obs_state.get('count', obs_rms.count))
      if (use_td3
          and bool(getattr(config, 'ppo_td3_use_target_policy', False))
          and 'td3_policy_target' in _ckpt):
        _td3_policy_target_from_ckpt = _ckpt['td3_policy_target']
      if use_crl_td3_switch:
        hybrid_td3_params = _extra.get(
            'hybrid_td3_params', hybrid_td3_params)
        hybrid_td3_opt_state = _extra.get(
            'hybrid_td3_opt_state', hybrid_td3_opt_state)
        hybrid_td3_params_reward = _extra.get(
            'hybrid_td3_params_reward',
            (_tree_copy(hybrid_td3_params)
             if _use_hybrid_td3_reward_ema else hybrid_td3_params))
        hard_goal_visit_count = int(
            _extra.get('hard_goal_visit_count', 0))
        reward_switched_to_td3 = bool(
            _extra.get('reward_switched_to_td3', False))
        reward_switch_iteration = int(
            _extra.get('reward_switch_iteration', -1))
        if (bool(getattr(config, 'ppo_td3_use_target_policy', False))
            and 'td3_policy_target' in _ckpt):
          _td3_policy_target_from_ckpt = _ckpt['td3_policy_target']
      print(f'[ppo] resumed from checkpoint: '
            f'start_iteration={start_iteration}, global_step={global_step}')
      # Truncate CSV logs to remove any entries written after the checkpoint
      # (can happen if training ran past the last checkpoint before preemption).
      _run_dir = os.path.dirname(checkpoint_dir)
      for _label in ('learner', 'eval'):
        _csv_path = os.path.join(_run_dir, 'logs', _label, 'logs.csv')
        _truncate_csv_to_iteration(_csv_path, int(_ckpt['iteration']))

  # ---- optional frozen pretrained reward (stationary φ·ψ) ---------------
  # Load φ/ψ from a successful CRL run and keep them fixed while training a
  # fresh PPO policy/value. On resume of *this* run, q_params already come
  # from this run's latest.pkl above, so skip reloading the foreign ckpt.
  _frozen_reward_ckpt = str(
      getattr(config, 'ppo_frozen_reward_ckpt', '') or '').strip()
  if _frozen_reward_ckpt and start_iteration == 0:
    if not os.path.exists(_frozen_reward_ckpt):
      raise FileNotFoundError(
          f'ppo_frozen_reward_ckpt not found: {_frozen_reward_ckpt}')
    _fr = load_checkpoint(_frozen_reward_ckpt)
    if (_fr.get('q_params_ema') is not None
        and not use_gaussian and not use_nf and not use_fm and not use_td3):
      _frozen_q = _fr['q_params_ema']
      _frozen_src = 'q_params_ema'
    else:
      _frozen_q = _fr['q_params']
      _frozen_src = 'q_params'
    q_params = _frozen_q
    q_params_reward = _tree_copy(_frozen_q)
    # Stationary: reward reads q_params directly; no EMA / CRL updates.
    _use_repr_ema = False
    _repr_tau = 0.0
    if int(config.ppo_crl_steps_per_iter) != 0:
      print(f'[ppo] frozen reward: forcing ppo_crl_steps_per_iter '
            f'{config.ppo_crl_steps_per_iter} -> 0')
      config.ppo_crl_steps_per_iter = 0
    # Re-init q optimizer state so it matches the loaded tree (unused when
    # crl_steps=0, but keeps checkpoint dumps consistent).
    if use_td3:
      q_opt_state = q_optimizer.init(_td3.online_td3_params(q_params))
    else:
      q_opt_state = q_optimizer.init(q_params)
    if use_td_infonce:
      td_infonce_target_q = _tree_copy(q_params)
    if use_nf_td:
      nf_td_target = _tree_copy(q_params)
    print(f'[ppo] frozen reward from {_frozen_reward_ckpt} '
          f'(source={_frozen_src}, crl_steps=0, fresh policy/value)')
  elif _frozen_reward_ckpt and start_iteration > 0:
    print(f'[ppo] frozen reward ckpt configured but resuming this run '
          f'(iter={start_iteration}); keeping q_params from latest.pkl')

  # TD3 optional Polyak target policy for bootstrap a' ~ π̄(·|s',g).
  _use_td3_target_policy = (
      (use_td3 or use_crl_td3_switch)
      and bool(getattr(config, 'ppo_td3_use_target_policy', False)))
  if _use_td3_target_policy:
    td3_policy_target = (
        _td3_policy_target_from_ckpt
        if _td3_policy_target_from_ckpt is not None
        else _tree_copy(ppo_params['policy']))
  else:
    td3_policy_target = ppo_params['policy']  # unused; keeps update signature stable

  # ---- jitted helpers ---------------------------------------------------
  print('[ppo] init: building jitted helpers...', flush=True)
  reward_mode = (getattr(config, 'ppo_reward_mode', '') or '').strip().lower()
  use_dirac_target = reward_mode == 'dirac_target'
  use_kde_dirac    = reward_mode == 'kde_dirac'
  reward_fn = make_reward_fn(networks, config)
  print('[ppo] init: reward_fn ready', flush=True)
  if use_dirac_target:
    print(f'[ppo] reward mode: dirac_target (eps={config.ppo_dirac_eps})')
  if use_kde_dirac:
    kde_max_points     = int(getattr(config, 'kde_max_points', 2000))
    kde_refit_interval = int(getattr(config, 'kde_refit_interval', 1))
    kde_bandwidth_cfg  = float(getattr(config, 'kde_bandwidth', 0.0))
    kde_bandwidth_arg  = kde_bandwidth_cfg if kde_bandwidth_cfg > 0.0 else None
    kde_state: Optional[GaussianKDE] = None
    print(f'[ppo] reward mode: kde_dirac  (eps={config.ppo_dirac_eps}, '
          f'max_points={kde_max_points}, refit_interval={kde_refit_interval}, '
          f'bandwidth={"Scott" if kde_bandwidth_arg is None else kde_bandwidth_cfg})')
  gae_fn = make_gae_fn(config)
  print('[ppo] init: gae_fn ready', flush=True)
  ppo_update = make_ppo_update_fn(
      networks, config, ppo_optimizer, ent_coef_schedule=ent_coef_schedule)
  print('[ppo] init: ppo_update ready', flush=True)

  td3_scan_update = None
  fm_scan_update = None
  td_infonce_scan_update = None
  td_infonce_crl_warmup_scan = None
  if use_gaussian:
    crl_update = _gd.make_gaussian_density_update_fn(
        density_nets, q_optimizer, obs_dim=int(config.obs_dim))
    gaussian_reward_fn = _gd.make_gaussian_reward_fn(
        density_nets, obs_dim=int(config.obs_dim))
    nf_reward_fn = None
    fm_reward_fn = None
    td3_reward_fn = None
    if _use_repr_ema:
      print(f'[ppo] Gaussian reward param EMA: tau={_repr_tau} '
            f'(density training still uses online Gaussian params)')
  elif use_fm:
    crl_update = _fm.make_fm_density_update_fn(
        fm_density_nets, q_optimizer, obs_dim=int(config.obs_dim))
    fm_scan_update = _fm.make_scan_fm_update_fn(
        fm_density_nets, q_optimizer, obs_dim=int(config.obs_dim),
        repr_tau=_repr_tau)
    fm_reward_fn = _fm.make_fm_reward_fn(
        fm_density_nets,
        obs_dim=int(config.obs_dim),
        logp_mode=str(getattr(config, 'fm_logp_mode', 'exact') or 'exact'),
        flow_steps=int(getattr(config, 'fm_flow_steps', 10)),
        hutch_probes=int(getattr(config, 'fm_hutch_probes', 8)),
    )
    gaussian_reward_fn = None
    nf_reward_fn = None
    td3_reward_fn = None
    if _use_repr_ema:
      print(f'[ppo] FM reward param EMA: tau={_repr_tau} '
            f'(density training still uses online FM params)')
    else:
      print('[ppo] FM reward uses online velocity-field params')
    print('[ppo] FM density updates: jax.lax.scan multi-step', flush=True)
  elif use_nf:
    crl_update = _nf.make_nf_density_update_fn(
        nf_density_nets, q_optimizer, obs_dim=int(config.obs_dim),
        noise_std=float(getattr(config, 'nf_noise_std', 0.0)))
    # Pre-sample N batches → one H→D transfer → one scanned JIT (like CRL).
    nf_scan_update = _nf.make_scan_nf_update_fn(
        nf_density_nets, q_optimizer, obs_dim=int(config.obs_dim),
        noise_std=float(getattr(config, 'nf_noise_std', 0.0)),
        repr_tau=_repr_tau)
    if use_nf_td:
      _nf_td_tau = float(getattr(config, 'ppo_nf_td_target_tau', 0.995))
      _nf_td_mix = float(getattr(config, 'ppo_nf_td_discount', -1.0))
      if _nf_td_mix < 0.0:
        _nf_td_mix = float(config.discount)
      if not (0.0 <= _nf_td_mix <= 1.0):
        raise ValueError(
            'ppo_nf_td_discount (TD-NF mix γ) must be in [0, 1], '
            f'got {_nf_td_mix}')
      _nf_td_mask = float(getattr(config, 'ppo_nf_td_mask_prob', 0.2))
      if not (0.0 <= _nf_td_mask <= 1.0):
        raise ValueError(
            'ppo_nf_td_mask_prob must be in [0, 1], '
            f'got {_nf_td_mask}')
      _nf_policy_goal = None
      if (builderbench_kwargs
          and builderbench_kwargs.get('fixed_target_goal') is not None):
        _nf_policy_goal = np.asarray(
            builderbench_kwargs['fixed_target_goal'],
            dtype=np.float32).reshape(-1)
      elif fixed_start_end is not None:
        fse = fixed_start_end
        if (isinstance(fse, (list, tuple)) and len(fse) == 2
            and not isinstance(fse[0], (int, float, np.floating))):
          _nf_policy_goal = np.asarray(
              fse[1], dtype=np.float32).reshape(-1)
        else:
          _nf_policy_goal = np.asarray(
              fse, dtype=np.float32).reshape(-1)
      if _nf_policy_goal is None:
        raise ValueError(
            'ppo_nf_td=True requires a fixed task goal for a\'∼π(·|s\',g) '
            '(builderbench fixed_target_goal / fixed_start_end)')
      crl_update = _nf.make_nf_td_density_update_fn(
          nf_density_nets, q_optimizer,
          obs_dim=int(config.obs_dim),
          discount=_nf_td_mix,
          policy_network_apply=networks.policy_network.apply,
          sample_fn=networks.sample,
          policy_goal=_nf_policy_goal,
          target_tau=_nf_td_tau,
          noise_std=float(getattr(config, 'nf_noise_std', 0.0)),
          mask_prob=_nf_td_mask,
          start_index=int(config.start_index),
          end_index=int(config.end_index),
          goal_state_indices=_goal_state_indices,
      )
      nf_scan_update = None
      nf_scan_td_update = _nf.make_scan_nf_td_update_fn(
          nf_density_nets, q_optimizer,
          obs_dim=int(config.obs_dim),
          discount=_nf_td_mix,
          policy_network_apply=networks.policy_network.apply,
          sample_fn=networks.sample,
          policy_goal=_nf_policy_goal,
          target_tau=_nf_td_tau,
          noise_std=float(getattr(config, 'nf_noise_std', 0.0)),
          mask_prob=_nf_td_mask,
          start_index=int(config.start_index),
          end_index=int(config.end_index),
          goal_state_indices=_goal_state_indices,
          repr_tau=_repr_tau,
      )
      print(f'[ppo] TD-NF density: mix_gamma={_nf_td_mix} '
            f'(HER/discount={config.discount}), '
            f'target_keep_tau={_nf_td_tau}, mask_prob={_nf_td_mask}, '
            f'a\'∼π(·|s\',g_task)  '
            f'policy_goal_dim={_nf_policy_goal.shape[0]}', flush=True)
    nf_reward_fn = _nf.make_nf_reward_fn(
        nf_density_nets, obs_dim=int(config.obs_dim))
    gaussian_reward_fn = None
    fm_reward_fn = None
    td3_reward_fn = None
    if _use_repr_ema:
      print(f'[ppo] NF reward param EMA: tau={_repr_tau} '
            f'(density training still uses online NF params)')
    print('[ppo] NF density updates: jax.lax.scan multi-step', flush=True)
  elif use_td3:
    _td3_tau_cfg = float(getattr(config, 'ppo_td3_tau', -1.0))
    _td3_tau = (_td3_tau_cfg if _td3_tau_cfg >= 0.0
                else float(config.tau))
    print('[ppo] init: make_td3_density_update_fn...', flush=True)
    _td3_update_kwargs = dict(
        density_nets=td3_density_nets,
        policy_network=networks.policy_network,
        sample_fn=networks.sample,
        optimizer=q_optimizer,
        obs_dim=int(config.obs_dim),
        start_index=int(config.start_index),
        end_index=int(config.end_index),
        discount=float(config.discount),
        tau=_td3_tau,
        goal_tol=float(getattr(config, 'ppo_td3_goal_tol', 1e-2)),
        use_target_policy=_use_td3_target_policy,
        cross_batch_goals=bool(
            getattr(config, 'ppo_td3_cross_batch_goals', False)),
        normalize_obs=norm_obs,
        obs_norm_clip=obs_norm_clip,
        goal_state_indices=_goal_state_indices,
        fb_loss=bool(getattr(config, 'ppo_td3_fb_loss', False)),
    )
    crl_update = _td3.make_td3_density_update_fn(**_td3_update_kwargs)
    td3_scan_update = _td3.make_scan_td3_update_fn(
        **_td3_update_kwargs, repr_tau=_repr_tau)
    print('[ppo] init: make_td3_reward_fn...', flush=True)
    td3_reward_fn = _td3.make_td3_reward_fn(
        td3_density_nets,
        obs_dim=int(config.obs_dim),
        discount=float(config.discount),
        log_reward=bool(getattr(config, 'ppo_td3_log_reward', False)),
        start_index=norm_si,
        end_index=norm_ei,
        normalize_obs=norm_obs,
        obs_norm_clip=obs_norm_clip,
        goal_state_indices=_goal_state_indices,
    )
    gaussian_reward_fn = None
    nf_reward_fn = None
    fm_reward_fn = None
    print('[ppo] TD3 density updates: jax.lax.scan multi-step', flush=True)
    _pi_src = 'target_policy' if _use_td3_target_policy else 'online_policy'
    _cross = bool(getattr(config, 'ppo_td3_cross_batch_goals', False))
    _bilin = bool(getattr(config, 'ppo_td3_bilinear', False))
    _use_log_r = bool(getattr(config, 'ppo_td3_log_reward', False))
    _use_fb = bool(getattr(config, 'ppo_td3_fb_loss', False))
    _q_src = (
        f'Q1_ema(tau={_repr_tau})' if _use_repr_ema else 'Q1_online')
    _rew_src = (
        f'log((1-γ)·{_q_src})' if _use_log_r else _q_src)
    _loss_name = (
        'FB [(Q−γ sg Q\')^2 − Q(s,a,s\')]' if _use_fb
        else 'TD3 [1[s\'≈sf]+γ Q\']')
    print(f'[ppo] TD3 Q update: loss={_loss_name}, '
          f'discount={config.discount}, '
          f'target_tau={_td3_tau}, '
          f'goal_tol={getattr(config, "ppo_td3_goal_tol", 1e-2)}, '
          f'reward={_rew_src}, a\'={_pi_src}, '
          f'cross_batch_goals={_cross}, bilinear={_bilin}', flush=True)
    if _use_repr_ema:
      print(f'[ppo] TD3 reward Q EMA: tau={_repr_tau} '
            f'(critic TD still uses online Q + Polyak targets)', flush=True)
  else:
    gaussian_reward_fn = None
    nf_reward_fn = None
    fm_reward_fn = None
    td3_reward_fn = None
    td_infonce_crl_warmup_update = None
    _tdi_crl_warmup_iters = 0
    if use_td_infonce:
      _tdi_tau = float(getattr(config, 'ppo_td_infonce_target_tau', 0.995))
      _tdi_mix = float(getattr(config, 'ppo_td_infonce_discount', -1.0))
      if _tdi_mix < 0.0:
        _tdi_mix = float(config.discount)
      if not (0.0 <= _tdi_mix <= 1.0):
        raise ValueError(
            'ppo_td_infonce_discount (TD-InfoNCE mix γ) must be in [0, 1], '
            f'got {_tdi_mix}')
      _tdi_crl_warmup_iters = int(
          getattr(config, 'ppo_tdinfonce_crl_warmup_iters', 0))
      if _tdi_crl_warmup_iters < 0:
        raise ValueError(
            'ppo_tdinfonce_crl_warmup_iters must be >= 0, '
            f'got {_tdi_crl_warmup_iters}')
      _policy_goal = None
      if builderbench_kwargs and builderbench_kwargs.get('fixed_target_goal') is not None:
        _policy_goal = np.asarray(
            builderbench_kwargs['fixed_target_goal'], dtype=np.float32).reshape(-1)
      elif fixed_start_end is not None:
        # Maze-style fixed_start_end is [start, goal]; BB passes a flat goal.
        fse = fixed_start_end
        if (isinstance(fse, (list, tuple)) and len(fse) == 2
            and not isinstance(fse[0], (int, float, np.floating))):
          _policy_goal = np.asarray(fse[1], dtype=np.float32).reshape(-1)
        else:
          _policy_goal = np.asarray(fse, dtype=np.float32).reshape(-1)
      if _policy_goal is None:
        raise ValueError(
            'ppo_repr_mode=tdinfonce requires a fixed task goal '
            '(builderbench fixed_target_goal / fixed_start_end)')
      _act_low = np.asarray(spec.actions.minimum, dtype=np.float32).reshape(-1)
      _act_high = np.asarray(spec.actions.maximum, dtype=np.float32).reshape(-1)
      _tdi_lse = float(
          getattr(config, 'ppo_td_infonce_logsumexp_coef', 0.01))
      if _tdi_lse < 0.0:
        raise ValueError(
            'ppo_td_infonce_logsumexp_coef must be >= 0, '
            f'got {_tdi_lse}')
      td_infonce_update, td_infonce_update_uniform = make_td_infonce_update_fn(
          networks, q_optimizer,
          obs_dim=int(config.obs_dim),
          discount=_tdi_mix,
          target_tau=_tdi_tau,
          start_index=int(config.start_index),
          end_index=int(config.end_index),
          twin_q=False,
          policy_goal=_policy_goal,
          action_low=_act_low,
          action_high=_act_high,
          logsumexp_penalty_coef=_tdi_lse,
          goal_state_indices=_goal_state_indices,
      )
      del td_infonce_update_uniform  # unused (policy bootstrap only)
      crl_update = td_infonce_update  # policy-bootstrap variant
      crl_scan_update = None
      td_infonce_scan_update = make_scan_td_infonce_update_fn(
          networks, q_optimizer,
          obs_dim=int(config.obs_dim),
          discount=_tdi_mix,
          target_tau=_tdi_tau,
          start_index=int(config.start_index),
          end_index=int(config.end_index),
          policy_goal=_policy_goal,
          action_low=_act_low,
          action_high=_act_high,
          logsumexp_penalty_coef=_tdi_lse,
          goal_state_indices=_goal_state_indices,
          repr_tau=_repr_tau,
      )
      td_infonce_crl_warmup_update = None
      td_infonce_crl_warmup_scan = None
      if _tdi_crl_warmup_iters > 0:
        _direction = str(
            getattr(config, 'ppo_crl_loss_direction', 'forward')).lower()
        td_infonce_crl_warmup_update = make_crl_update_fn(
            networks, q_optimizer,
            backward=(_direction == 'backward'), config=config)
        td_infonce_crl_warmup_scan = make_scan_crl_update_fn(
            networks, q_optimizer, backward=(_direction == 'backward'),
            repr_tau=_repr_tau, config=config)
        print(f'[ppo] TD-InfoNCE CRL warm-start: '
              f'{_tdi_crl_warmup_iters} iters of standard InfoNCE, '
              f'then switch critic to TD-InfoNCE')
      print(f'[ppo] TD-InfoNCE update: mix_gamma={_tdi_mix} '
            f'(HER/discount={config.discount}), '
            f'target_keep_tau={_tdi_tau}, twin_q=False, '
            f'logsumexp_coef={_tdi_lse}, '
            f'repr_norm={bool(config.repr_norm)}, '
            f'policy_goal_dim={_policy_goal.shape[0]}')
      print('[ppo] TD-InfoNCE density updates: jax.lax.scan multi-step',
            flush=True)
      if _use_repr_ema:
        print(f'[ppo] TD-InfoNCE reward φ·ψ EMA: tau={_repr_tau} '
              f'(critic uses online φ,ψ; target keep-tau={_tdi_tau})')
    else:
      _direction = str(getattr(config, 'ppo_crl_loss_direction', 'forward')).lower()
      _backward = (_direction == 'backward')
      crl_update = make_crl_update_fn(
          networks, q_optimizer, backward=_backward, config=config)
      # Scan-based multi-step updater: one JIT dispatch for all CRL steps.
      crl_scan_update = make_scan_crl_update_fn(
          networks, q_optimizer, backward=_backward, repr_tau=_repr_tau,
          config=config)
      print(f'[ppo] CRL loss direction: {_direction}')
      _sf_p = float(getattr(config, 'ppo_crl_sf_perturb_prob', 0.0))
      _sf_eps = float(getattr(config, 'ppo_crl_sf_perturb_eps', 1e-2))
      if _sf_p > 0.0 and _sf_eps > 0.0:
        print(f'[ppo] CRL s_f perturb: prob={_sf_p}  eps={_sf_eps}  '
              f'(‖δ‖₂≤eps on packed goal before InfoNCE)')
      _hit = (getattr(config, 'ppo_crl_hit_bonus', '') or '').strip().lower()
      if _hit:
        _hit_g = getattr(config, 'ppo_crl_hit_bonus_goal', None)
        print(f'[ppo] CRL hit-indicator param: mode={_hit!r}  '
              f'tol={getattr(config, "ppo_crl_hit_bonus_tol", 1e-2)}  '
              f'scale={getattr(config, "ppo_crl_hit_bonus_scale", 1.0)}  '
              f'task_goal_set={_hit_g is not None}  '
              f'(logits + PPO reward = φ·ψ + scale·1{{‖sg−g‖<tol}})')
      if _use_repr_ema:
        print(f'[ppo] CRL reward repr EMA: tau={_repr_tau} '
              f'(InfoNCE still uses online φ, ψ)')

  hybrid_td3_update = None
  hybrid_td3_scan_update = None
  hybrid_td3_reward_fn = None
  if use_crl_td3_switch:
    _td3_tau_cfg = float(getattr(config, 'ppo_td3_tau', -1.0))
    _td3_tau = (_td3_tau_cfg if _td3_tau_cfg >= 0.0
                else float(config.tau))
    _hybrid_td3_kwargs = dict(
        density_nets=td3_density_nets,
        policy_network=networks.policy_network,
        sample_fn=networks.sample,
        optimizer=q_optimizer,
        obs_dim=int(config.obs_dim),
        start_index=int(config.start_index),
        end_index=int(config.end_index),
        discount=float(config.discount),
        tau=_td3_tau,
        goal_tol=float(getattr(config, 'ppo_td3_goal_tol', 1e-2)),
        use_target_policy=_use_td3_target_policy,
        cross_batch_goals=bool(
            getattr(config, 'ppo_td3_cross_batch_goals', False)),
        normalize_obs=norm_obs,
        obs_norm_clip=obs_norm_clip,
        goal_state_indices=_goal_state_indices,
        fb_loss=bool(getattr(config, 'ppo_td3_fb_loss', False)),
    )
    hybrid_td3_update = _td3.make_td3_density_update_fn(**_hybrid_td3_kwargs)
    hybrid_td3_scan_update = _td3.make_scan_td3_update_fn(
        **_hybrid_td3_kwargs, repr_tau=_hybrid_td3_reward_tau)
    print('[ppo] hybrid TD3 density updates: jax.lax.scan multi-step',
          flush=True)
    hybrid_td3_reward_fn = _td3.make_td3_reward_fn(
        td3_density_nets,
        obs_dim=int(config.obs_dim),
        discount=float(config.discount),
        log_reward=bool(getattr(config, 'ppo_td3_log_reward', False)),
        start_index=norm_si,
        end_index=norm_ei,
        normalize_obs=norm_obs,
        obs_norm_clip=obs_norm_clip,
        goal_state_indices=_goal_state_indices,
    )
    print(f'[ppo] CRL→TD3 switch enabled: threshold='
          f'{_switch_goal_visits} completed hard-goal visits; '
          f'CRL and TD3 updates/iter={config.ppo_crl_steps_per_iter}; '
          f'CRL training stops at threshold, TD3 continues; '
          f'reward_blend_iters={_switch_blend_iters} '
          f'({"hard switch" if _switch_blend_iters == 0 else "linear CRL→TD3"})')
    if _use_hybrid_td3_reward_ema:
      print(f'[ppo] hybrid TD3 reward Q EMA: '
            f'tau={_hybrid_td3_reward_tau}')

  def _reward_q_params():
    return q_params_reward if _use_repr_ema else q_params

  def _hybrid_reward_td3_params():
    return (hybrid_td3_params_reward
            if _use_hybrid_td3_reward_ema else hybrid_td3_params)

  _det_select = bool(getattr(config, 'ppo_deterministic_select_dim', False))
  if _det_select:
    print('[ppo] deterministic_select_dim=True: last action dim (select) '
          'uses μ only (std deactivated for sample/logprob/entropy)')

  @jax.jit
  def act_and_value(policy_p, value_p, obs, rng, obs_mean, obs_var):
    network_obs = _normalize_packed_obs(
        obs, obs_mean, obs_var, obs_dim=obs_dim_cfg, start_index=norm_si,
        end_index=norm_ei, clip=obs_norm_clip, enabled=norm_obs,
        goal_state_indices=_goal_state_indices)
    dist = _policy_dist_maybe_det_select(
        networks.policy_network.apply(policy_p, network_obs), _det_select)
    action = networks.sample(dist, rng)
    logprob = networks.log_prob(dist, action)
    value = networks.value_network.apply(value_p, network_obs)
    return action, logprob, value

  bb_generate_unroll = None
  bb_eval_unroll = None
  bb_eval_vec = None
  if _use_jax_bb_vec:
    print(f'[ppo] init: compile_generate_unroll (T={T})...', flush=True)
    bb_generate_unroll = vec_env.compile_generate_unroll(
        act_and_value,
        unroll_length=T,
        obs_dim=int(config.obs_dim),
        dynamic_obs_stats=True)
    print(f'[ppo] BuilderBench rollout: jax.lax.scan (T={T})', flush=True)
    _eval_iv = int(getattr(config, 'ppo_eval_interval', 10))
    if _eval_iv > 0:
      _n_eval = int(getattr(config, 'ppo_eval_episodes', 5))
      bb_eval_vec = JaxBuilderBenchVecEnv(
          env_name=_env_name,
          num_envs=_n_eval,
          seed=int(seed * 31 + 77),
          use_pd=bool(_bb_kw.get('builderbench_use_pd', False)),
          pd_duration=int(_bb_kw.get('builderbench_pd_duration', 5)),
          fixed_target_goal=_bb_kw.get('fixed_target_goal'),
          permute_start_boxes=bool(
              _bb_kw.get('builderbench_permute_start_boxes', True)),
          mj_episode_length=_bb_kw.get('builderbench_mj_episode_length'),
          fixed_start_x=_bb_kw.get('builderbench_fixed_start_x'),
          success_terminate_steps=int(
              _bb_kw.get('builderbench_success_terminate_steps', 0) or 0),
      )

      @jax.jit
      def eval_policy_action(policy_p, obs, obs_mean, obs_var):
        network_obs = _normalize_packed_obs(
            obs, obs_mean, obs_var, obs_dim=obs_dim_cfg, start_index=norm_si,
            end_index=norm_ei, clip=obs_norm_clip, enabled=norm_obs,
        goal_state_indices=_goal_state_indices)
        dist = networks.policy_network.apply(policy_p, network_obs)
        return dist.mode()

      print('[ppo] init: compile_eval_unroll...', flush=True)
      bb_eval_unroll = bb_eval_vec.compile_eval_unroll(
          eval_policy_action,
          unroll_length=bb_eval_vec.episode_length,
          dynamic_obs_stats=True)
      print(f'[ppo] BuilderBench eval: jax.lax.scan '
            f'(E={_n_eval}, ep_len={bb_eval_vec.episode_length}, '
            f'every {_eval_iv} iters)', flush=True)
      # Warm-compile once at startup so the first logged eval does not stall
      # training (XLA compile of the E_eval scan can take minutes otherwise).
      _warm_t0 = time.time()
      print('[ppo] warming BuilderBench eval unroll (XLA compile)...',
            flush=True)
      _warm_mean = jnp.asarray(obs_rms.mean, dtype=jnp.float32)
      _warm_var = jnp.asarray(obs_rms.var, dtype=jnp.float32)
      _warm_steps = bb_eval_unroll(
          bb_eval_vec.reset_state(), ppo_params['policy'],
          _warm_mean, _warm_var)
      jax.block_until_ready(_warm_steps['reward'])
      print(f'[ppo] BuilderBench eval warm-compile done in '
            f'{time.time() - _warm_t0:.1f}s', flush=True)
      del _warm_steps, _warm_mean, _warm_var
    else:
      print('[ppo] BuilderBench: periodic eval disabled; '
            'logging train_success_mean / train_success_1000 from rollouts')

  # Optional in-train deterministic video (BuilderBench only).
  bb_video_env = None
  bb_video_mocap = None
  bb_video_ep_len = None
  bb_video_num_cubes = None
  bb_video_filter = False
  bb_video_dir = None
  _video_interval = int(getattr(config, 'ppo_video_interval', 0) or 0)
  _video_fps = int(getattr(config, 'ppo_video_fps', 10) or 10)
  _skip_first_video = bool(getattr(config, 'ppo_skip_first_video', True))
  if _use_jax_bb_vec and _video_interval > 0:
    from contrastive import builderbench_video as _bb_vid
    _filter_pol = bool(_bb_kw.get('builderbench_pd_filter_policy_obs', True))
    if not bool(_bb_kw.get('builderbench_use_pd', False)):
      _filter_pol = False
    (bb_video_env, bb_video_mocap, bb_video_ep_len, bb_video_num_cubes,
     bb_video_filter) = _bb_vid.make_bb_video_env(
         _env_name,
         use_pd=bool(_bb_kw.get('builderbench_use_pd', False)),
         pd_duration=int(_bb_kw.get('builderbench_pd_duration', 5)),
         permute_start_boxes=bool(
             _bb_kw.get('builderbench_permute_start_boxes', True)),
         filter_policy_obs=_filter_pol,
    )
    if checkpoint_dir is not None:
      bb_video_dir = os.path.join(os.path.dirname(checkpoint_dir), 'videos')
    else:
      bb_video_dir = os.path.join('videos', 'builderbench', 'in_train')
    os.makedirs(bb_video_dir, exist_ok=True)
    print(f'[ppo] BuilderBench in-train video: every {_video_interval} iters, '
          f'deterministic, norm_obs={norm_obs}, fps={_video_fps}, '
          f'dir={bb_video_dir}', flush=True)

  @jax.jit
  def value_only(value_p, obs, obs_mean, obs_var):
    obs = _normalize_packed_obs(
        obs, obs_mean, obs_var, obs_dim=obs_dim_cfg, start_index=norm_si,
        end_index=norm_ei, clip=obs_norm_clip, enabled=norm_obs,
        goal_state_indices=_goal_state_indices)
    return networks.value_network.apply(value_p, obs)

  @jax.jit
  def greedy_action(policy_p, obs, obs_mean, obs_var):
    # Deterministic policy mean for evaluation.
    obs = _normalize_packed_obs(
        obs, obs_mean, obs_var, obs_dim=obs_dim_cfg, start_index=norm_si,
        end_index=norm_ei, clip=obs_norm_clip, enabled=norm_obs,
        goal_state_indices=_goal_state_indices)
    dist = networks.policy_network.apply(policy_p, obs)
    return networks.sample(dist, jax.random.PRNGKey(0))  # sample still; we log both below

  # ---- uniform-sampling goal bounds (extracted once from env spec) ------
  uniform_sampling = bool(getattr(config, 'uniform_sampling', False))
  goal_low = goal_high = None
  if uniform_sampling:
    import env_utils as _env_utils
    if hasattr(vec_env, 'uniform_goal_obs_bounds'):
      glo, ghi = vec_env.uniform_goal_obs_bounds()
      glo = np.asarray(glo, dtype=np.float32).reshape(-1)
      ghi = np.asarray(ghi, dtype=np.float32).reshape(-1)
      # Bounds are already in goal space (masked offsets). Use directly when
      # length matches goal_dim; otherwise fall back to state-slice indexing.
      if glo.shape[0] == goal_dim_cfg:
        goal_low, goal_high = glo, ghi
      else:
        si, ei = int(config.start_index), int(config.end_index)
        if ei == -1:
          ei = int(config.obs_dim)
        goal_low = glo[si:ei]
        goal_high = ghi[si:ei]
    else:
      goal_low, goal_high = _env_utils.resolve_uniform_goal_bounds(
          spec, vec_env._envs[0], int(config.obs_dim),
          int(config.start_index), int(config.end_index))
    print(f'[ppo] uniform_sampling: goal_low={goal_low}, goal_high={goal_high}')

  # ---- replay buffer (episodes) -----------------------------------------
  _success_sample_weight = float(
      getattr(config, 'ppo_success_sample_weight', 1.0))
  if _success_sample_weight <= 0.0:
    raise ValueError(
        'ppo_success_sample_weight must be > 0, got '
        f'{_success_sample_weight}')
  replay = EpisodeReplay(
      capacity=int(config.max_replay_size),
      obs_dim=int(config.obs_dim),
      discount=float(config.discount),
      start_index=int(config.start_index),
      end_index=int(config.end_index),
      success_sample_weight=_success_sample_weight,
      goal_state_indices=_goal_state_indices)
  np_rng = np.random.default_rng(seed + 12345)
  if _success_sample_weight != 1.0:
    print('[ppo] CRL episode sampling: successful trajectories weight='
          f'{_success_sample_weight:g}, others weight=1 '
          '(sample proportional to w / sum w)')

  use_external_reward = bool(
      getattr(config, 'ppo_use_external_reward', False))
  external_reward_scale = float(
      getattr(config, 'ppo_external_reward_scale', 100.0))
  external_reward_before_norm = bool(
      getattr(config, 'ppo_external_reward_before_norm', False))
  if use_external_reward:
    _sawyer_extrew = str(_env_name).startswith('sawyer_')
    if not (_use_jax_bb_vec or _sawyer_extrew):
      raise ValueError(
          'ppo_use_external_reward requires BuilderBench (metrics["success"]) '
          'or a Sawyer MetaWorld env (sparse env reward as hard success)')
    _when = (
        'BEFORE reward normalisation'
        if external_reward_before_norm else
        'AFTER reward normalisation')
    _src = (
        'metrics["success"]' if _use_jax_bb_vec else
        'env_reward>=0.5 (Sawyer sparse success)')
    print('[ppo] external hard-success bonus enabled: '
          f'scale={external_reward_scale:g} '
          f'(added {_when} on {_src} steps)')

  # ---- per-env episode buffers (for flushing complete trajectories) -----
  ep_obs: list = [[] for _ in range(E)]
  ep_act: list = [[] for _ in range(E)]
  ep_return = np.zeros(E, dtype=np.float32)
  ep_flow_dense_return = np.zeros(E, dtype=np.float32)
  ep_has_flow_dense = np.zeros(E, dtype=bool)
  ep_len = np.zeros(E, dtype=np.int32)
  recent_returns: list = []
  recent_flow_dense_returns: list = []
  recent_lengths: list = []
  recent_success: list = []
  ep_success_max = np.zeros(E, dtype=np.float32)
  _track_train_success = _use_jax_bb_vec
  # Nominal PD/MJ episode length (BuilderBench). Used to detect collapse via
  # short episodes (e.g. repeated OOB early terminations) and reinit the actor.
  _nominal_ep_len = (
      int(vec_env.episode_length) if _use_jax_bb_vec else 0)
  _actor_reset_ep_frac = 0.8
  _force_reset_iters: set = set()
  _raw_force = str(getattr(config, 'ppo_actor_reset_iters', '') or '').strip()
  if _raw_force:
    for _tok in _raw_force.replace(';', ',').split(','):
      _tok = _tok.strip()
      if _tok:
        _force_reset_iters.add(int(_tok))
  if _nominal_ep_len > 0:
    print(f'[ppo] actor-reset guard: reinit policy if ep_length_mean < '
          f'{_actor_reset_ep_frac:.0%} of nominal ({_nominal_ep_len}) '
          f'= {_actor_reset_ep_frac * _nominal_ep_len:.1f}')
  if _force_reset_iters:
    print(f'[ppo] actor-reset schedule: force reinit at iters '
          f'{sorted(_force_reset_iters)}')

  # Running goal normalisation stats for NF mode.
  # Updated from replay buffer each iteration; broadcast-compatible with goals.
  nf_goal_mean = np.zeros(goal_dim_cfg, dtype=np.float32)
  nf_goal_std  = np.ones(goal_dim_cfg,  dtype=np.float32)

  obs = vec_env.reset()
  next_done = np.zeros(E, dtype=np.float32)
  s0_states = np.asarray(obs[:, :int(config.obs_dim)], dtype=np.float32).copy()
  if _use_jax_bb_vec:
    # ndarray carries for the vectorized episode flush (BB GPU path).
    ep_obs = [obs[i:i + 1].astype(np.float32).copy() for i in range(E)]
    ep_act = [
        np.zeros((0, act_dim_cfg), dtype=np.float32) for _ in range(E)]
  else:
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
  elif _env.startswith('builderbench_'):
    eval_success_obs = _cu.BuilderBenchSuccessObserver()
  else:
    eval_success_obs = _cu.SuccessObserver()
  eval_dist_obs = _cu.DistanceObserver(
      obs_dim=int(config.obs_dim),
      start_index=int(config.start_index),
      end_index=int(config.end_index))

  # ---- rollout storage (reused each iteration) --------------------------
  roll_obs   = np.zeros((T, E) + obs_shape, dtype=np.float32)
  roll_acts  = np.zeros((T, E) + act_shape, dtype=np.float32)
  roll_logp  = np.zeros((T, E), dtype=np.float32)
  roll_rew   = np.zeros((T, E), dtype=np.float32)          # reps reward (possibly normalized)
  roll_rew_raw = np.zeros((T, E), dtype=np.float32)        # reps reward (pre-normalization, log only)
  roll_env_rew = np.zeros((T, E), dtype=np.float32)        # gt reward (log only)
  roll_flow_dense_rew = np.full((T, E), np.nan, dtype=np.float32)
  roll_dones = np.zeros((T, E), dtype=np.float32)
  roll_vals  = np.zeros((T, E), dtype=np.float32)
  # Per-step dones from env (used for reward normalization after rollout).
  roll_step_dones = np.zeros((T, E), dtype=np.float32)
  # s0 state tracked for dirac_target: batched reward call after rollout.
  roll_s0_states = np.zeros(
      (T, E, int(config.obs_dim)), dtype=np.float32)

  # ---- reward normalizer (CleanRL NormalizeReward) ----------------------
  # Normalizes the reps-based reward by the running std of discounted
  # returns.  Critical for PPO with learned rewards — raw φ·ψ values can
  # be O(10) and non-stationary, leading to unbounded advantages and
  # policy collapse within a handful of updates.
  norm_reward = bool(getattr(config, 'ppo_norm_reward', True))
  ppo_gamma = (float(config.ppo_discount)
               if float(getattr(config, 'ppo_discount', -1.0)) > 0.0
               else float(config.discount))
  _return_norm_window = int(getattr(config, 'ppo_return_norm_window', 0))
  reward_normalizer = (
      ReturnNormalizer(num_envs=E, discount=ppo_gamma,
                       window_size=_return_norm_window)
      if norm_reward else None)
  _nf_normalizer_reset_done = False  # reset once when NF first activates
  _nf_stat_log: Dict[str, float] = {}  # per-dim goal mean/std, updated each iter
  # Sparse success-step reward dump (intrinsic / extrinsic / full).
  _extrew_diag_prints = 0
  _extrew_diag_max = 20
  _extrew_diag_warmup_iters = 10

  # ---- checkpointing ----------------------------------------------------
  ckpt_interval = int(getattr(config, 'ppo_checkpoint_interval', 0))
  ckpt_keep_last = int(getattr(config, 'ppo_checkpoint_keep_last', 0))
  if ckpt_interval > 0 and checkpoint_dir is not None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    keep_msg = ('keep all milestones'
                if ckpt_keep_last <= 0
                else f'keep last {ckpt_keep_last} milestones')
    print(f'[ppo] checkpoints → {checkpoint_dir} '
          f'(every {ckpt_interval} iters, {keep_msg})')

  start_time = time.time()
  # global_step, ppo_sgd_step, start_iteration set above (0 for fresh runs,
  # restored from checkpoint on resume).

  for iteration in range(start_iteration, num_iterations):
    # One immutable snapshot is shared by every network call in this
    # iteration. Stats are advanced from raw rollout states only after eval.
    iter_obs_mean = np.asarray(obs_rms.mean, dtype=np.float32).copy()
    iter_obs_var = np.asarray(obs_rms.var, dtype=np.float32).copy()
    if norm_obs and obs_rms.count <= 0.0:
      # Sentinel consumed by _normalize_packed_obs: first iteration is raw,
      # including no clipping, then rollout data initializes the RMS.
      iter_obs_mean.fill(np.nan)
    iter_obs_mean_j = jnp.asarray(iter_obs_mean)
    iter_obs_var_j = jnp.asarray(iter_obs_var)
    # Freeze the reward-transition decision for the whole iteration. If the
    # threshold is crossed in this rollout, blending / TD3 starts next iter.
    _reward_uses_td3_this_iter = (
        use_crl_td3_switch and reward_switched_to_td3)
    _reward_td3_weight = 0.0
    if _reward_uses_td3_this_iter:
      if _switch_blend_iters <= 0:
        _reward_td3_weight = 1.0
      else:
        _reward_td3_weight = float(np.clip(
            (int(iteration) - int(reward_switch_iteration))
            / float(_switch_blend_iters),
            0.0, 1.0))
    # =================================================================
    # 1. Rollout (on-policy, CleanRL convention)
    # =================================================================
    _roll_success = None
    # When set, reward / GAE / PPO consume these device arrays directly
    # instead of round-tripping the full (T, E, …) traj through NumPy.
    rollout_j = None
    if bb_generate_unroll is not None:
      # BuilderBench: fused policy + env unroll via jax.lax.scan (GPU).
      (vec_env._state, key, next_done, _), _steps_j = bb_generate_unroll(
          vec_env._state,
          ppo_params['policy'],
          ppo_params['value'],
          key,
          jnp.asarray(next_done),
          jnp.asarray(s0_states),
          iter_obs_mean_j,
          iter_obs_var_j,
      )
      # Keep the heavy rollout on device for reward → GAE → PPO.
      rollout_j = {
          'obs': _steps_j['obs'],
          'actions': _steps_j['actions'],
          'logprobs': _steps_j['logprobs'],
          'values': _steps_j['values'],
          'roll_dones': _steps_j['roll_dones'],
          'step_dones': _steps_j['step_dones'],
          'env_rew': _steps_j['env_rew'],
      }
      if use_dirac_target:
        rollout_j['s0_states'] = _steps_j['s0_states']
        roll_s0_states[:] = np.asarray(_steps_j['s0_states'], dtype=np.float32)
      # Host copies only for episode→replay flush + reward-normalizer state.
      _flush_acts = np.asarray(_steps_j['actions'], dtype=np.float32)
      _flush_env_rew = np.asarray(_steps_j['env_rew'], dtype=np.float32)
      _flush_step_dones = np.asarray(
          _steps_j['step_dones'], dtype=np.float32)
      roll_acts[:] = _flush_acts
      roll_env_rew[:] = _flush_env_rew
      roll_step_dones[:] = _flush_step_dones
      if use_kde_dirac:
        # KDE reward is NumPy-only; materialize obs for that path.
        roll_obs[:] = np.asarray(_steps_j['obs'], dtype=np.float32)
        for _t in range(T):
          rep_rew_np = (np.zeros(E, dtype=np.float32) if kde_state is None
                        else reward_fn(kde_state, roll_obs[_t]).astype(np.float32))
          roll_rew_raw[_t] = rep_rew_np
        rollout_j = None  # fall back to host rollout buffers below
      obs = np.asarray(
          vec_env.pack_obs_from_state(vec_env._state), dtype=np.float32)
      _roll_success = (
          np.asarray(_steps_j['success'], dtype=np.float32)
          if _track_train_success else None)
      _term_all = np.asarray(_steps_j['terminal_obs'], dtype=np.float32)
      _next_all = np.asarray(_steps_j['next_obs'], dtype=np.float32)
      hard_goal_visit_count = _flush_rollout_episodes_to_replay(
          actions=_flush_acts,
          env_rew=_flush_env_rew,
          dones=_flush_step_dones.astype(bool),
          terminal_obs=_term_all,
          next_obs=_next_all,
          success=_roll_success,
          ep_obs=ep_obs,
          ep_act=ep_act,
          ep_return=ep_return,
          ep_len=ep_len,
          ep_success_max=ep_success_max,
          s0_states=s0_states,
          obs_dim=int(config.obs_dim),
          recent_returns=recent_returns,
          recent_lengths=recent_lengths,
          recent_success=recent_success,
          use_crl_td3_switch=use_crl_td3_switch,
          hard_goal_visit_count=hard_goal_visit_count,
          replay=replay,
      )
      global_step += T * E
    else:
      for t in range(T):
        roll_obs[t] = obs
        roll_dones[t] = next_done

        key, k_act = jax.random.split(key)
        action_j, logprob_j, value_j = act_and_value(
            ppo_params['policy'], ppo_params['value'],
            jnp.asarray(obs), k_act, iter_obs_mean_j, iter_obs_var_j)
        action = np.asarray(action_j)
        roll_acts[t] = action
        roll_logp[t] = np.asarray(logprob_j)
        roll_vals[t] = np.asarray(value_j)

        # For dirac_target, record s0 for the batched reward call later.
        if use_dirac_target:
          roll_s0_states[t] = s0_states
        # kde_dirac reward is pure NumPy — compute per-step to avoid storing KDE.
        elif use_kde_dirac:
          rep_rew_np = (np.zeros(E, dtype=np.float32) if kde_state is None
                        else reward_fn(kde_state, obs).astype(np.float32))
          roll_rew_raw[t] = rep_rew_np

        next_obs, env_rew, dones, terminal_obs, info_rew = vec_env.step(action)
        roll_env_rew[t] = env_rew
        roll_flow_dense_rew[t] = info_rew
        # Store step-level dones for the post-rollout reward normalizer loop.
        roll_step_dones[t] = dones.astype(np.float32)

        # Episode flushing / per-env accounting.
        for i in range(E):
          ep_act[i].append(action[i].copy())
          ep_return[i] += float(env_rew[i])
          if not np.isnan(info_rew[i]):
            ep_flow_dense_return[i] += float(info_rew[i])
            ep_has_flow_dense[i] = True
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
            s0_states[i] = next_obs[i, :int(config.obs_dim)].copy()
            ep_act[i] = []
            recent_returns.append(float(ep_return[i]))
            if ep_has_flow_dense[i]:
              recent_flow_dense_returns.append(float(ep_flow_dense_return[i]))
            recent_lengths.append(int(ep_len[i]))
            ep_return[i] = 0.0
            ep_flow_dense_return[i] = 0.0
            ep_has_flow_dense[i] = False
            ep_len[i] = 0
            if len(recent_returns) > 100:
              recent_returns.pop(0)
              recent_lengths.pop(0)
            if len(recent_flow_dense_returns) > 100:
              recent_flow_dense_returns.pop(0)
          else:
            ep_obs[i].append(next_obs[i].copy())

        obs = next_obs
        next_done = dones.astype(np.float32)
        global_step += E

      # Sawyer MetaWorld envs use sparse 0/1 env reward as hard success.
      if use_external_reward and _roll_success is None:
        _roll_success = (
            (np.asarray(roll_env_rew, dtype=np.float32) >= 0.5)
            .astype(np.float32))

    _switch_after_this_iteration = (
        use_crl_td3_switch
        and not reward_switched_to_td3
        and hard_goal_visit_count >= _switch_goal_visits)

    # Occasional raw-vs-normalized sanity print (2 early iters only).
    if norm_obs and iteration in (1, 10) and np.all(np.isfinite(iter_obs_mean)):
      if rollout_j is not None:
        _raw = np.asarray(rollout_j['obs'][0, 0], dtype=np.float32)
      else:
        _raw = np.asarray(roll_obs[0, 0], dtype=np.float32)
      _norm = np.asarray(_normalize_packed_obs(
          jnp.asarray(_raw[None]), iter_obs_mean_j, iter_obs_var_j,
          obs_dim=obs_dim_cfg, start_index=norm_si, end_index=norm_ei,
          clip=obs_norm_clip, enabled=True,
          goal_state_indices=_goal_state_indices)[0])
      print(f'[ppo][obs_norm] iter={iteration} raw  '
            f'state={np.array2string(_raw[:obs_dim_cfg], precision=3)} '
            f'goal={np.array2string(_raw[obs_dim_cfg:], precision=3)}')
      print(f'[ppo][obs_norm] iter={iteration} norm '
            f'state={np.array2string(_norm[:obs_dim_cfg], precision=3)} '
            f'goal={np.array2string(_norm[obs_dim_cfg:], precision=3)}')

    # =================================================================
    # 1b. Batched reward computation (single GPU call over full rollout)
    # =================================================================
    # kde_dirac rewards were already filled per-step above (CPU-only).
    _rew_flat_j = None
    if not use_kde_dirac:
      if rollout_j is not None:
        _flat_obs_j = jnp.reshape(rollout_j['obs'], (T * E, -1))
        _flat_acts_j = jnp.reshape(rollout_j['actions'], (T * E, -1))
      else:
        _flat_obs_j = jnp.asarray(roll_obs.reshape(T * E, -1))
        _flat_acts_j = jnp.asarray(roll_acts.reshape(T * E, -1))
      if use_nf:
        _rew_flat_j = nf_reward_fn(
            _reward_q_params(), _flat_obs_j, _flat_acts_j,
            jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std))
      elif use_gaussian:
        _rew_flat_j = gaussian_reward_fn(
            _reward_q_params(), _flat_obs_j, _flat_acts_j)
      elif use_fm:
        _fm_mode = (
            getattr(config, 'fm_logp_mode', 'exact') or 'exact').strip().lower()
        if 'hutch' in _fm_mode:
          key, k_fm_rew = jax.random.split(key)
          _rew_flat_j = fm_reward_fn(
              _reward_q_params(), _flat_obs_j, _flat_acts_j, k_fm_rew)
        else:
          _rew_flat_j = fm_reward_fn(
              _reward_q_params(), _flat_obs_j, _flat_acts_j)
      elif use_td3:
        _rew_flat_j = td3_reward_fn(
            _reward_q_params(), _flat_obs_j, _flat_acts_j,
            iter_obs_mean_j, iter_obs_var_j)
      elif use_crl_td3_switch and _reward_uses_td3_this_iter:
        _rew_td3_j = hybrid_td3_reward_fn(
            _hybrid_reward_td3_params(), _flat_obs_j, _flat_acts_j,
            iter_obs_mean_j, iter_obs_var_j)
        if _reward_td3_weight >= 1.0 - 1e-8:
          _rew_flat_j = _rew_td3_j
        else:
          # Still mix in (frozen) CRL reward during the blend window.
          _rew_crl_j = reward_fn(
              _reward_q_params(), _flat_obs_j, _flat_acts_j,
              iter_obs_mean_j, iter_obs_var_j)
          _w = float(_reward_td3_weight)
          _rew_flat_j = (1.0 - _w) * _rew_crl_j + _w * _rew_td3_j
      elif use_dirac_target:
        if rollout_j is not None and 's0_states' in rollout_j:
          _flat_s0_j = jnp.reshape(rollout_j['s0_states'], (T * E, -1))
        else:
          _flat_s0_j = jnp.asarray(roll_s0_states.reshape(T * E, -1))
        _rew_flat_j = reward_fn(
            _reward_q_params(), _flat_obs_j, _flat_acts_j, _flat_s0_j)
      else:
        # Default CRL: r = φ(s,a)·ψ(g)
        _rew_flat_j = reward_fn(
            _reward_q_params(), _flat_obs_j, _flat_acts_j,
            iter_obs_mean_j, iter_obs_var_j)
      # ReturnNormalizer is stateful NumPy — one small (T·E,) D2H only.
      roll_rew_raw[:] = np.asarray(_rew_flat_j, dtype=np.float32).reshape(T, E)

    # Hard-success external bonus (opt-in).
    # Default: after return-norm so the fixed scale is not washed out.
    # Optional: before return-norm (absorbed into running return std).
    _ext = None
    if use_external_reward and _roll_success is not None:
      _ext = (
          external_reward_scale
          * (_roll_success >= 0.5).astype(np.float32))
      if external_reward_before_norm:
        roll_rew_raw += _ext

    # Apply reward normalisation (cheap NumPy loop; normalizer state is shared
    # across the rollout, reset on episode boundaries via roll_step_dones).
    if reward_normalizer is not None:
      for _t in range(T):
        roll_rew[_t] = reward_normalizer(roll_rew_raw[_t], roll_step_dones[_t])
    else:
      roll_rew[:] = roll_rew_raw

    if (use_external_reward and _roll_success is not None
            and not external_reward_before_norm):
      roll_rew += _ext

    # Occasional success-step dump (after warmup): intrinsic / extrinsic / full.
    if (use_external_reward and _roll_success is not None
            and iteration >= _extrew_diag_warmup_iters
            and _extrew_diag_prints < _extrew_diag_max):
      _succ_ij = np.argwhere(_roll_success >= 0.5)
      if _succ_ij.size > 0:
        _t_i, _e_i = (int(x) for x in _succ_ij[0])
        if external_reward_before_norm:
          _intr_raw = float(roll_rew_raw[_t_i, _e_i] - _ext[_t_i, _e_i])
          _intr_norm = float('nan')  # mixed into return-norm with extrinsic
        else:
          _intr_raw = float(roll_rew_raw[_t_i, _e_i])
          _intr_norm = float(roll_rew[_t_i, _e_i] - _ext[_t_i, _e_i])
        _extr = float(_ext[_t_i, _e_i])
        _full = float(roll_rew[_t_i, _e_i])
        _extrew_diag_prints += 1
        print(
            f'[ppo][extrew] goal seen iter={iteration} '
            f't={_t_i} env={_e_i} '
            f'({_extrew_diag_prints}/{_extrew_diag_max}) '
            f'intrinsic_raw={_intr_raw:.4g} '
            f'extrinsic={_extr:.4g} '
            f'intrinsic_after_norm={_intr_norm:.4g} '
            f'full_reward={_full:.4g}',
            flush=True)

    # =================================================================
    # 2. GAE advantages / returns
    # =================================================================
    next_val_j = value_only(
        ppo_params['value'], jnp.asarray(obs),
        iter_obs_mean_j, iter_obs_var_j)
    if rollout_j is not None:
      _gae_vals_j = rollout_j['values']
      _gae_dones_j = rollout_j['roll_dones']
    else:
      _gae_vals_j = jnp.asarray(roll_vals)
      _gae_dones_j = jnp.asarray(roll_dones)
    adv_j, ret_j = gae_fn(
        jnp.asarray(roll_rew), _gae_vals_j, _gae_dones_j,
        next_val_j, jnp.asarray(next_done))

    # =================================================================
    # 3. PPO updates (epochs × minibatches over flat T·E batch)
    # =================================================================
    if rollout_j is not None:
      flat_obs_j = jnp.reshape(
          rollout_j['obs'], (batch_per_iter,) + obs_shape)
      flat_acts_j = jnp.reshape(
          rollout_j['actions'], (batch_per_iter,) + act_shape)
      flat_logp_j = jnp.reshape(rollout_j['logprobs'], (batch_per_iter,))
      flat_vals_j = jnp.reshape(rollout_j['values'], (batch_per_iter,))
    else:
      flat_obs_j = jnp.asarray(
          roll_obs.reshape((batch_per_iter,) + obs_shape))
      flat_acts_j = jnp.asarray(
          roll_acts.reshape((batch_per_iter,) + act_shape))
      flat_logp_j = jnp.asarray(roll_logp.reshape(batch_per_iter))
      flat_vals_j = jnp.asarray(roll_vals.reshape(batch_per_iter))
    flat_adv_j = jnp.reshape(adv_j, (batch_per_iter,))
    flat_ret_j = jnp.reshape(ret_j, (batch_per_iter,))

    ppo_metrics_device: list = []
    early_stop = False
    _need_kl_sync = config.ppo_target_kl is not None
    for epoch in range(int(config.ppo_num_epochs)):
      perm = np_rng.permutation(batch_per_iter)
      last_kl = None
      for start in range(0, batch_per_iter, mb_size):
        mb = jnp.asarray(perm[start:start + mb_size])
        batch = {
            'obs':          flat_obs_j[mb],
            'actions':      flat_acts_j[mb],
            'old_logprobs': flat_logp_j[mb],
            'advantages':   flat_adv_j[mb],
            'returns':      flat_ret_j[mb],
            'old_values':   flat_vals_j[mb],
        }
        key, k_mb = jax.random.split(key)
        if ent_coef_schedule is not None:
          ppo_params, ppo_opt_state, m = ppo_update(
              ppo_params, ppo_opt_state, batch, k_mb,
              iter_obs_mean_j, iter_obs_var_j,
              jnp.asarray(ppo_sgd_step, dtype=jnp.int32))
        else:
          ppo_params, ppo_opt_state, m = ppo_update(
              ppo_params, ppo_opt_state, batch, k_mb,
              iter_obs_mean_j, iter_obs_var_j)
        ppo_sgd_step += 1
        ppo_metrics_device.append(m)
        # Only sync KL when early-stopping is enabled (default: off).
        if _need_kl_sync:
          last_kl = float(m['approx_kl'])
      if (_need_kl_sync and last_kl is not None
          and last_kl > float(config.ppo_target_kl)):
        early_stop = True
        break

    # One host sync for all PPO metrics after the epoch loop.
    ppo_metrics_agg: Dict[str, list] = {}
    if ppo_metrics_device:
      _m_keys = list(ppo_metrics_device[0].keys())
      _stacked = {
          k_: jnp.stack([m[k_] for m in ppo_metrics_device])
          for k_ in _m_keys}
      _stacked_np = {k_: np.asarray(v) for k_, v in _stacked.items()}
      for k_, arr in _stacked_np.items():
        ppo_metrics_agg[k_] = [float(x) for x in arr.reshape(-1)]

    pg_vals = ppo_metrics_agg.get('pg_loss', [])
    mean_pg = float(np.mean(pg_vals)) if pg_vals else float('inf')
    # Host mirrors used by logging (single sync).
    adv = np.asarray(adv_j)
    ret = np.asarray(ret_j)
    if rollout_j is not None:
      roll_vals[:] = np.asarray(rollout_j['values'], dtype=np.float32)
      # Lazy materialize of obs only if later code needs the host buffer.
      roll_dones[:] = np.asarray(
          rollout_j['roll_dones'], dtype=np.float32)
      roll_logp[:] = np.asarray(
          rollout_j['logprobs'], dtype=np.float32)

    # =================================================================
    # 4. CRL updates (off-policy, from replay)
    # =================================================================
    crl_metrics_agg: Dict[str, list] = {}
    hybrid_td3_metrics_agg: Dict[str, list] = {}
    if replay.size >= int(config.ppo_min_replay_size):
      # Update goal normalisation stats from a fresh replay sample (NF only).
      if use_nf and not _nf_normalizer_reset_done:
        # First time NF activates: reset the return normalizer so the extreme
        # rewards from the untrained flow don't permanently corrupt the running
        # std that PPO uses to scale advantages.
        if reward_normalizer is not None:
          reward_normalizer._rms = RunningMeanStd(shape=())
          reward_normalizer._returns = np.zeros(E, dtype=np.float64)
        _nf_normalizer_reset_done = True
      if use_nf:
        _std_floor = float(getattr(config, 'nf_goal_std_min', 0.02))
        _stat_batch = replay.sample(min(2048, replay.size), np_rng)
        _goals = _stat_batch['obs'][:, int(config.obs_dim):]
        # Optionally mix in the actual env goals from the current rollout so
        # that the running stats cover both hindsight goals AND reward goals.
        if bool(getattr(config, 'nf_mix_env_goal_stats', False)):
          if rollout_j is not None:
            _env_goals = np.asarray(
                rollout_j['obs'].reshape(-1, rollout_j['obs'].shape[-1])
                [:, int(config.obs_dim):],
                dtype=np.float32)
          else:
            _env_goals = roll_obs.reshape(
                -1, roll_obs.shape[-1])[:, int(config.obs_dim):]
          _goals = np.concatenate([_goals, _env_goals], axis=0)
        nf_goal_mean = _goals.mean(axis=0).astype(np.float32)
        nf_goal_std  = _goals.std(axis=0).astype(np.float32)
        nf_goal_std  = np.maximum(nf_goal_std, _std_floor).astype(np.float32)
        for _di, (_gm, _gs) in enumerate(zip(nf_goal_mean, nf_goal_std)):
          _nf_stat_log[f'nf/goal_mean_{_di}'] = float(_gm)
          _nf_stat_log[f'nf/goal_std_{_di}']  = float(_gs)

      _n_crl = int(config.ppo_crl_steps_per_iter)
      # Frozen-reward / stationary φ·ψ sets crl_steps=0: skip updates. The
      # pre-sample+stack paths below would IndexError on an empty _samples.
      if _n_crl <= 0:
        pass
      elif use_crl_td3_switch:
        # TD3 learns from the beginning and continues after the reward switch.
        _samples_td3 = [
            (replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
             if uniform_sampling
             else replay.sample(int(config.batch_size), np_rng))
            for _ in range(_n_crl)]
        _stacked_td3 = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples_td3], axis=0))
            for k_ in _samples_td3[0]}
        (hybrid_td3_params, hybrid_td3_opt_state, hybrid_td3_params_reward,
         key, td3_policy_target, m) = hybrid_td3_scan_update(
            hybrid_td3_params, hybrid_td3_opt_state,
            hybrid_td3_params_reward, _stacked_td3, key,
            ppo_params['policy'], td3_policy_target,
            iter_obs_mean_j, iter_obs_var_j)
        hybrid_td3_metrics_agg = {k_: [float(v)] for k_, v in m.items()}

        # Once the goal threshold is reached CRL is no longer trained. The
        # frozen CRL critic still supplied this iteration's already-computed
        # rewards; TD3 becomes active only on the next iteration.
        if not _switch_after_this_iteration and not reward_switched_to_td3:
          _samples = [
              (replay.sample_with_uniform_negatives(
                  int(config.batch_size), np_rng, goal_low, goal_high)
               if uniform_sampling
               else replay.sample(int(config.batch_size), np_rng))
              for _ in range(_n_crl)]
          _stacked = {
              k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
              for k_ in _samples[0]}
          (q_params, q_opt_state, q_params_reward,
           key, m) = crl_scan_update(
              q_params, q_opt_state, q_params_reward, _stacked, key,
              iter_obs_mean_j, iter_obs_var_j)
          crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}
      elif use_nf:
        # Pre-sample all NF batches → one H→D transfer → one scanned JIT.
        _samples = [
            (replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
             if uniform_sampling
             else replay.sample(int(config.batch_size), np_rng))
            for _ in range(_n_crl)]
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        if use_nf_td:
          (q_params, q_opt_state, nf_td_target, q_params_reward,
           key, m) = nf_scan_td_update(
              q_params, q_opt_state, nf_td_target, q_params_reward,
              _stacked, key,
              jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std),
              ppo_params['policy'])
        else:
          (q_params, q_opt_state, q_params_reward,
           key, m) = nf_scan_update(
              q_params, q_opt_state, q_params_reward, _stacked, key,
              jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std))
        crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}
      elif use_td3:
        _samples = [
            (replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
             if uniform_sampling
             else replay.sample(int(config.batch_size), np_rng))
            for _ in range(_n_crl)]
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        (q_params, q_opt_state, q_params_reward,
         key, td3_policy_target, m) = td3_scan_update(
            q_params, q_opt_state, q_params_reward, _stacked, key,
            ppo_params['policy'], td3_policy_target,
            iter_obs_mean_j, iter_obs_var_j)
        crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}
      elif use_td_infonce:
        _tdi_in_crl_warmup = (
            td_infonce_crl_warmup_scan is not None
            and int(iteration) < int(_tdi_crl_warmup_iters))
        if (td_infonce_crl_warmup_scan is not None
            and int(iteration) == int(_tdi_crl_warmup_iters)):
          # Cold-start TD target from warm CRL params at the switch iter.
          td_infonce_target_q = _tree_copy(q_params)
          print(f'[ppo] iter={iteration}: CRL warm-start done '
                f'({_tdi_crl_warmup_iters} iters); switching critic to '
                f'TD-InfoNCE (target Q reinit from online φ,ψ)',
                flush=True)
        _samples = []
        for _ in range(_n_crl):
          if uniform_sampling:
            crl_batch_np = replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
          else:
            crl_batch_np = replay.sample(int(config.batch_size), np_rng)
          if not _tdi_in_crl_warmup:
            # Column goals = rolled geometric future goals (s_j from sample()).
            _si = int(config.start_index)
            _ei = int(config.end_index)
            if 'future_state' in crl_batch_np:
              _fs = crl_batch_np['future_state']
            else:
              _fs = None
            if _fs is not None:
              _fut_g = (_fs[:, _si:] if _ei == -1 else _fs[:, _si:_ei])
            else:
              _fut_g = crl_batch_np['obs'][:, int(config.obs_dim):]
            crl_batch_np = dict(crl_batch_np)
            crl_batch_np['random_goal'] = np.roll(_fut_g, -1, axis=0)
          _samples.append(crl_batch_np)
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        if _tdi_in_crl_warmup:
          (q_params, q_opt_state, q_params_reward,
           key, m) = td_infonce_crl_warmup_scan(
              q_params, q_opt_state, q_params_reward, _stacked, key,
              iter_obs_mean_j, iter_obs_var_j)
        else:
          (q_params, q_opt_state, td_infonce_target_q, q_params_reward,
           key, m) = td_infonce_scan_update(
              q_params, q_opt_state, td_infonce_target_q, q_params_reward,
              _stacked, key, ppo_params['policy'])
        crl_metrics_agg = {}
        for k_, v in m.items():
          try:
            crl_metrics_agg[k_] = [float(v)]
          except (TypeError, ValueError):
            pass  # skip non-scalars (e.g. a_prime_hist)
        crl_metrics_agg['crl_warmup_active'] = [
            1.0 if _tdi_in_crl_warmup else 0.0]
      elif use_fm:
        _samples = [
            (replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
             if uniform_sampling
             else replay.sample(int(config.batch_size), np_rng))
            for _ in range(_n_crl)]
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        (q_params, q_opt_state, q_params_reward,
         key, m) = fm_scan_update(
            q_params, q_opt_state, q_params_reward, _stacked, key)
        crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}
      elif use_gaussian:
        # Gaussian still on the Python loop (unused for now).
        for _ in range(_n_crl):
          if uniform_sampling:
            crl_batch_np = replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
          else:
            crl_batch_np = replay.sample(int(config.batch_size), np_rng)
          crl_batch = {k_: jnp.asarray(v) for k_, v in crl_batch_np.items()}
          key, k_crl = jax.random.split(key)
          q_params, q_opt_state, m = crl_update(
              q_params, q_opt_state, crl_batch, k_crl)
          if _use_repr_ema:
            q_params_reward = _ema_tree(q_params_reward, q_params, _repr_tau)
          for k_, v in m.items():
            try:
              crl_metrics_agg.setdefault(k_, []).append(float(v))
            except (TypeError, ValueError):
              pass
      else:
        # Standard CRL: pre-sample all batches → one H→D transfer → one JIT.
        # This eliminates _n_crl rounds of dispatch + host sync.
        _samples = [
            (replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
             if uniform_sampling
             else replay.sample(int(config.batch_size), np_rng))
            for _ in range(_n_crl)]
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        (q_params, q_opt_state, q_params_reward,
         key, m) = crl_scan_update(
            q_params, q_opt_state, q_params_reward, _stacked, key,
            iter_obs_mean_j, iter_obs_var_j)
        crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}

    # =================================================================
    # 4b. KDE refit (kde_dirac mode only)
    # =================================================================
    if use_kde_dirac and replay.size >= int(config.ppo_min_replay_size):
      if iteration % kde_refit_interval == 0:
        kde_state = GaussianKDE.fit(
            replay, int(config.obs_dim),
            kde_max_points, np_rng, kde_bandwidth_arg)

    # =================================================================
    # 5. Logging
    # =================================================================
    elapsed = time.time() - start_time
    # flow-dense reward: only log if this env actually emits it.
    _has_flow_dense = not np.all(np.isnan(roll_flow_dense_rew))

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
        'ep_return_mean':    float(np.mean(recent_returns)) if recent_returns else float('nan'),
        'ep_length_mean':    float(np.mean(recent_lengths)) if recent_lengths else float('nan'),
        'ppo/mean_pg_loss':  mean_pg,
    }
    if _track_train_success:
      log['train_success_mean'] = (
          float(np.mean(recent_success[-100:]))
          if recent_success else float('nan'))
      log['train_success_1000'] = (
          float(np.mean(recent_success[-1000:]))
          if recent_success else float('nan'))
    if use_crl_td3_switch:
      log['reward_source_td3'] = float(_reward_uses_td3_this_iter)
      log['reward_td3_weight'] = float(_reward_td3_weight)
      log['hard_goal_visit_count'] = int(hard_goal_visit_count)
      log['reward_switch_pending'] = float(_switch_after_this_iteration)
      log['reward_switch_iteration'] = int(reward_switch_iteration)
    log['obs_norm_enabled'] = float(norm_obs)
    log['obs_norm_count'] = float(obs_rms.count)
    log['obs_norm_mean_abs'] = float(np.mean(np.abs(obs_rms.mean)))
    log['obs_norm_std_mean'] = float(np.mean(np.sqrt(iter_obs_var + 1e-8)))

    # Acme CSVLogger fixes columns on the *first* write and drops any later
    # keys.  Seed density-mode columns from iter 0 so training metrics land in CSV.
    if use_fm:
      log.update({
          'fm/density_loss': float('nan'),
          'fm/fm_flow_loss': float('nan'),
          'fm/vel_pred_norm': float('nan'),
          'fm/vel_target_norm': float('nan'),
          'fm/x1_norm': float('nan'),
          'fm/grad_norm': float('nan'),
          'fm/update_skipped_nonfinite': float('nan'),
          'fm/update_steps': 0,
      })
    if use_nf:
      log.update({
          'nf/density_loss': float('nan'),
          'nf/log_p_mean': float('nan'),
          'nf/log_p_min': float('nan'),
          'nf/log_p_max': float('nan'),
          'nf/flow_grad_norm': float('nan'),
          'nf/update_skipped_nonfinite': float('nan'),
          'nf/update_steps': 0,
          'sa/repr_norm': float('nan'),
          'sa/encoder_grad_norm': float('nan'),
      })
      if nf_density_nets.goal_encoder_net is not None:
        log['nf/goal_enc_grad_norm'] = float('nan')
      for _di in range(goal_dim_cfg):
        log[f'nf/goal_mean_{_di}'] = float('nan')
        log[f'nf/goal_std_{_di}']  = float('nan')

    # Flow-dense benchmark reward: only when the env provides it.
    if _has_flow_dense:
      log['reward_flow_dense_mean'] = float(np.nanmean(roll_flow_dense_rew))
      log['ep_flow_dense_return_mean'] = (
          float(np.mean(recent_flow_dense_returns))
          if recent_flow_dense_returns else float('nan'))

    # PPO update metrics (always present).
    for k_, vs in ppo_metrics_agg.items():
      log[f'ppo/{k_}'] = float(np.mean(vs))

    # Density-estimator metrics: only the active mode.
    if use_gaussian:
      for k_, vs in crl_metrics_agg.items():
        log[f'gaussian/{k_}'] = float(np.mean(vs))
    elif use_fm:
      for k_, vs in crl_metrics_agg.items():
        log[f'fm/{k_}'] = float(np.mean(vs))
      log['fm/update_steps'] = (
          int(config.ppo_crl_steps_per_iter) if crl_metrics_agg else 0)
    elif use_nf:
      _sa_keys = frozenset({'repr_norm', 'encoder_grad_norm'})
      _ge_keys = frozenset({'goal_enc_grad_norm'})
      for k_, vs in crl_metrics_agg.items():
        if k_ in _sa_keys:
          prefix = 'sa'
        elif k_ in _ge_keys:
          prefix = 'nf'
        else:
          prefix = 'nf'
        log[f'{prefix}/{k_}'] = float(np.mean(vs))
      # Scan path stores mean metrics (len=1); report configured step count.
      log['nf/update_steps'] = (
          int(config.ppo_crl_steps_per_iter) if crl_metrics_agg else 0)
    elif use_td3:
      for k_, vs in crl_metrics_agg.items():
        log[f'td3/{k_}'] = float(np.mean(vs))
      log['td3/update_steps'] = (
          int(config.ppo_crl_steps_per_iter) if crl_metrics_agg else 0)
    elif use_td_infonce:
      for k_, vs in crl_metrics_agg.items():
        log[f'tdinfonce/{k_}'] = float(np.mean(vs))
      log['tdinfonce/update_steps'] = (
          int(config.ppo_crl_steps_per_iter) if crl_metrics_agg else 0)
    elif use_crl_td3_switch:
      for k_, vs in crl_metrics_agg.items():
        log[f'crl/{k_}'] = float(np.mean(vs))
      for k_, vs in hybrid_td3_metrics_agg.items():
        log[f'td3/{k_}'] = float(np.mean(vs))
      log['crl/update_steps'] = (
          int(config.ppo_crl_steps_per_iter) if crl_metrics_agg else 0)
      log['td3/update_steps'] = (
          int(config.ppo_crl_steps_per_iter)
          if hybrid_td3_metrics_agg else 0)
    else:
      for k_, vs in crl_metrics_agg.items():
        log[f'crl/{k_}'] = float(np.mean(vs))
    if config.ppo_anneal_lr:
      lr_log = float(lr_schedule(max(0, ppo_sgd_step - 1)))
    else:
      lr_log = float(config.learning_rate)
    # Seven fractional digits so CSV / terminal show stable small LRs.
    log['ppo/learning_rate'] = round(lr_log, 7)
    log.update(_nf_stat_log)
    learner_logger.write(log)

    if _switch_after_this_iteration:
      reward_switched_to_td3 = True
      reward_switch_iteration = int(iteration) + 1
      if _switch_blend_iters > 0:
        print(f'[ppo] hard-goal visit threshold reached at iter={iteration}: '
              f'visits={hard_goal_visit_count}; linear CRL→TD3 blend starts '
              f'at iter={reward_switch_iteration} over '
              f'{_switch_blend_iters} iters '
              f'(w=0→1); CRL training stopped; '
              f'PPO/value/replay/reward-normalizer state preserved')
      else:
        print(f'[ppo] hard-goal visit threshold reached at iter={iteration}: '
              f'visits={hard_goal_visit_count}; TD3 reward starts at '
              f'iter={reward_switch_iteration}; CRL training stopped; '
              f'PPO/value/replay/reward-normalizer state preserved')

    # =================================================================
    # 5b. Actor reset if episodes are collapsing OR on a forced schedule
    # =================================================================
    _ep_mean = (float(np.mean(recent_lengths)) if recent_lengths
                else float('nan'))
    _short_ep = (
        _nominal_ep_len > 0 and recent_lengths
        and _ep_mean < _actor_reset_ep_frac * float(_nominal_ep_len))
    _forced = int(iteration) in _force_reset_iters
    if _short_ep or _forced:
      _thresh = _actor_reset_ep_frac * float(_nominal_ep_len)
      key, k_pol = jax.random.split(key)
      _new_policy = networks.policy_network.init(k_pol)
      ppo_params = {'policy': _new_policy, 'value': ppo_params['value']}
      _fresh_opt = ppo_optimizer.init(ppo_params)
      ppo_opt_state = _merge_policy_opt_state(ppo_opt_state, _fresh_opt)
      if _use_td3_target_policy:
        td3_policy_target = _tree_copy(_new_policy)
      else:
        td3_policy_target = ppo_params['policy']
      recent_lengths.clear()
      recent_returns.clear()
      if _forced and _short_ep:
        _why = (f'forced+short ep_length_mean={_ep_mean:.2f} < {_thresh:.1f}')
      elif _forced:
        _why = 'forced schedule'
      else:
        _why = (f'ep_length_mean={_ep_mean:.2f} < {_thresh:.1f} '
                f'(nominal={_nominal_ep_len})')
      print(f'[ppo] actor reset at iter={iteration}: {_why}')

    # =================================================================
    # 6. Periodic evaluation
    # =================================================================
    _skip_first = bool(getattr(config, 'ppo_skip_first_eval', False))
    _eval_interval = int(getattr(config, 'ppo_eval_interval', 10))
    _n_eval = int(getattr(config, 'ppo_eval_episodes', 5))
    if (_eval_interval > 0
        and iteration % _eval_interval == 0
        and not (iteration == 0 and _skip_first)):
      _eval_t0 = time.time()
      _eval_gpu_s = None
      if bb_eval_unroll is not None and bb_eval_vec is not None:
        eval_state = bb_eval_vec.reset_state()
        steps_j = bb_eval_unroll(
            eval_state, ppo_params['policy'],
            iter_obs_mean_j, iter_obs_var_j)
        jax.block_until_ready(steps_j['reward'])
        _eval_gpu_s = time.time() - _eval_t0
        ep_metrics_list = _bb_ep_metrics_from_eval_steps(
            steps_j,
            obs_dim=int(config.obs_dim),
            start_index=int(config.start_index),
            end_index=int(config.end_index),
            episode_length=int(bb_eval_vec.episode_length),
            goal_state_indices=_goal_state_indices,
        )
        ep_metrics_list = _smooth_bb_eval_metrics(
            ep_metrics_list, eval_success_obs, eval_dist_obs)
      else:
        ep_metrics_list = []
        for e_i in range(_n_eval):
          env = eval_env_factory(seed + 900_000 + iteration * 100 + e_i)
          ts = env.reset()
          eval_success_obs.observe_first(env, ts)
          eval_dist_obs.observe_first(env, ts)
          ret_e, n_e = 0.0, 0
          flow_dense_ret_e = 0.0
          flow_dense_steps = 0
          while not ts.last():
            key, k_eval = jax.random.split(key)
            a, _, _ = act_and_value(
                ppo_params['policy'], ppo_params['value'],
                jnp.asarray(ts.observation)[None], k_eval,
                iter_obs_mean_j, iter_obs_var_j)
            action = np.asarray(a)[0].astype(np.float32)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
            ts = env.step(action)
            eval_success_obs.observe(env, ts, action)
            eval_dist_obs.observe(env, ts, action)
            ret_e += float(ts.reward or 0.0)
            dense_r = _cu.extract_info_reward(env)
            if not np.isnan(dense_r):
              flow_dense_ret_e += dense_r
              flow_dense_steps += 1
            n_e += 1
          ep_metrics = {'episode_return': ret_e, 'episode_length': n_e}
          if flow_dense_steps > 0:
            ep_metrics.update(
                _cu.flow_dense_eval_episode_metrics(
                    flow_dense_ret_e, flow_dense_steps))
          ep_metrics.update(eval_success_obs.get_metrics())
          ep_metrics.update(eval_dist_obs.get_metrics())
          ep_metrics_list.append(ep_metrics)

      agg = _cu.aggregate_eval_metrics(ep_metrics_list, iteration)
      eval_logger.write(agg)
      _eval_total_s = time.time() - _eval_t0
      _gpu_part = (f'gpu={_eval_gpu_s:.2f}s '
                   if _eval_gpu_s is not None else 'gpu=n/a ')
      print(f'[ppo] eval iter={iteration}: {_gpu_part}'
            f'total={_eval_total_s:.2f}s E={_n_eval} '
            f'success={agg.get("success", float("nan")):.3f}',
            flush=True)

    # =================================================================
    # 6b. Periodic deterministic video (same frozen obs_rms as eval)
    # =================================================================
    if (bb_video_env is not None
        and _video_interval > 0
        and iteration % _video_interval == 0
        and not (iteration == 0 and _skip_first_video)):
      from contrastive import builderbench_video as _bb_vid
      _vid_t0 = time.time()
      _vid_path = os.path.join(
          bb_video_dir, f'iter_{int(iteration):07d}.mp4')
      try:
        _out, _n_frames = _bb_vid.render_deterministic_episode(
            video_env=bb_video_env,
            mocap_targets=bb_video_mocap,
            num_cubes=int(bb_video_num_cubes),
            episode_length=int(bb_video_ep_len),
            networks=networks,
            policy_params=ppo_params['policy'],
            obs_mean=np.asarray(iter_obs_mean, dtype=np.float32),
            obs_var=np.asarray(iter_obs_var, dtype=np.float32),
            fixed_target_goal=_bb_kw.get('fixed_target_goal'),
            seed=int(seed + 17_000 + iteration),
            filter_policy_obs=bool(bb_video_filter),
            normalize_obs=bool(norm_obs),
            obs_dim=int(obs_dim_cfg),
            start_index=int(norm_si),
            end_index=int(norm_ei),
            obs_norm_clip=float(obs_norm_clip),
            fps=int(_video_fps),
            out_path=_vid_path,
            goal_state_indices=_goal_state_indices,
        )
        print(f'[ppo] video iter={iteration}: wrote {_out} '
              f'({_n_frames} frames) in {time.time() - _vid_t0:.1f}s',
              flush=True)
      except Exception as _vid_exc:
        print(f'[ppo] video iter={iteration}: FAILED after '
              f'{time.time() - _vid_t0:.1f}s: {_vid_exc}', flush=True)

    # Advance only after every network call in the iteration has consumed the
    # frozen snapshot. The first fresh run therefore uses identity/raw inputs.
    if norm_obs:
      if rollout_j is not None:
        _obs_for_rms = np.asarray(
            rollout_j['obs'].reshape(T * E, -1)[:, :obs_dim_cfg],
            dtype=np.float32)
      else:
        _obs_for_rms = roll_obs.reshape(T * E, -1)[:, :obs_dim_cfg]
      obs_rms.update(_obs_for_rms)

    # =================================================================
    # 7. Checkpointing: ckpt_iter_{iter}.pkl every `ppo_checkpoint_interval`
    #    iters + rolling latest.pkl.  Prune only if ppo_checkpoint_keep_last>0.
    # =================================================================
    if (ckpt_interval > 0
        and checkpoint_dir is not None
        and (iteration % ckpt_interval == 0
             or iteration == num_iterations - 1)):
      _checkpoint_extra = {
          'obs_rms': {
              'mean': np.asarray(obs_rms.mean, dtype=np.float64),
              'var': np.asarray(obs_rms.var, dtype=np.float64),
              'count': float(obs_rms.count),
          },
      } if norm_obs else {}
      if use_crl_td3_switch:
        _checkpoint_extra.update({
            'hybrid_td3_params': hybrid_td3_params,
            'hybrid_td3_opt_state': hybrid_td3_opt_state,
            'hybrid_td3_params_reward': hybrid_td3_params_reward,
            'hard_goal_visit_count': int(hard_goal_visit_count),
            'reward_switched_to_td3': bool(reward_switched_to_td3),
            'reward_switch_iteration': int(reward_switch_iteration),
        })
      if use_td_infonce:
        _checkpoint_extra['td_infonce_target_q'] = td_infonce_target_q
      if use_nf_td:
        _checkpoint_extra['nf_td_target'] = nf_td_target
      ckpt_kw = dict(
          policy_params=ppo_params['policy'],
          value_params=ppo_params['value'],
          q_params=q_params,
          ppo_opt_state=ppo_opt_state,
          q_opt_state=q_opt_state,
          iteration=iteration,
          global_step=global_step,
          key=key,
          q_params_ema=(q_params_reward if _use_repr_ema else None),
          td3_policy_target=(
              td3_policy_target if _use_td3_target_policy else None),
          extra_state=(_checkpoint_extra if _checkpoint_extra else None))
      milestone_path = os.path.join(
          checkpoint_dir, f'ckpt_iter_{iteration:07d}.pkl')
      _save_checkpoint(milestone_path, **ckpt_kw)
      _save_checkpoint(os.path.join(checkpoint_dir, 'latest.pkl'), **ckpt_kw)
      _prune_old_checkpoints(checkpoint_dir, ckpt_keep_last)

  # ---- return final state in case the caller wants to checkpoint --------
  return PPOTrainingState(
      policy_params=ppo_params['policy'],
      value_params=ppo_params['value'],
      ppo_optimizer_state=ppo_opt_state,
      q_params=q_params,
      q_optimizer_state=q_opt_state,
      key=key,
  )
