"""JAX vectorized 3D additive-noise environment.

Dynamics (per env, per step)::

    a  ~  Uniform[-1, 1]^3   (provided by the caller)
    s' = s + a + ε,   ε ~ N(0, σ² I),   σ = 0.01 by default

State and observation are both ``(x, y, z)``.  Episodes truncate at a
fixed horizon (default 50); there is no terminal success signal.

Optional ``single_axis_xy``: each episode locks motion to **either** the
x-axis **or** the y-axis (never both in one episode).  Useful as an OOD
protocol when eval uses unrestricted joint ``(a_x, a_y)`` actions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class JaxXYZVecEnv:
  """Batched JAX xyz random-walk env (``num_envs`` in parallel)."""

  STATE_DIM = 3
  ACTION_DIM = 3

  def __init__(
      self,
      num_envs: int = 1024,
      episode_length: int = 50,
      noise_std: float = 0.01,
      action_low: float = -1.0,
      action_high: float = 1.0,
      reset_scale: float = 0.0,
      seed: int = 0,
      zero_az: bool = False,
      force_az_positive: bool = False,
      single_axis_xy: bool = False,
      reset_center: Optional[np.ndarray] = None,
  ):
    self.num_envs = int(num_envs)
    self.episode_length = int(episode_length)
    self.noise_std = float(noise_std)
    self.action_low = float(action_low)
    self.action_high = float(action_high)
    self.reset_scale = float(reset_scale)
    # If True: force a_z=0 and no z-noise (motion only in x,y; z stays 0).
    self.zero_az = bool(zero_az)
    # If True: sample a_z ~ Uniform(0, action_high] (OOD vs zero_az training).
    self.force_az_positive = bool(force_az_positive)
    # If True: each episode picks x-only OR y-only actions (never both).
    self.single_axis_xy = bool(single_axis_xy)
    if self.zero_az and self.force_az_positive:
      raise ValueError('zero_az and force_az_positive are mutually exclusive')
    if reset_center is None:
      self.reset_center = np.zeros(self.STATE_DIM, dtype=np.float32)
    else:
      self.reset_center = np.asarray(reset_center, dtype=np.float32).reshape(
          self.STATE_DIM)
    self._key = jax.random.PRNGKey(int(seed))
    self._state = np.broadcast_to(
        self.reset_center, (self.num_envs, self.STATE_DIM)).copy()
    self._t = np.zeros(self.num_envs, dtype=np.int32)
    # Per-env locked axis for the current episode: 0=x-only, 1=y-only.
    self._axis = np.zeros(self.num_envs, dtype=np.int32)
    self._np_rng = np.random.default_rng(int(seed) + 17)
    if self.single_axis_xy:
      self._resample_axes(np.ones(self.num_envs, dtype=bool))
    self._step_jit = jax.jit(self._step_fn)
    self._reset_jit = jax.jit(self._reset_fn)

  @property
  def obs_dim(self) -> int:
    return self.STATE_DIM

  @property
  def act_dim(self) -> int:
    return self.ACTION_DIM

  def _split_key(self):
    self._key, sub = jax.random.split(self._key)
    return sub

  def _resample_axes(self, mask: np.ndarray) -> None:
    """Assign a fresh x-or-y lock for envs where ``mask`` is True."""
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
      return
    self._axis[mask] = self._np_rng.integers(0, 2, size=n, dtype=np.int32)

  def _reset_fn(self, key, mask, state, t, reset_scale, reset_center):
    """Reset envs where ``mask`` is True; leave others unchanged."""
    key_noise, = jax.random.split(key, 1)
    noise = reset_scale * jax.random.normal(
        key_noise, shape=state.shape, dtype=state.dtype)
    base = reset_center[None, :] + noise
    new_state = jnp.where(mask[:, None], base, state)
    new_t = jnp.where(mask, jnp.zeros_like(t), t)
    return new_state, new_t

  def _step_fn(self, key, state, action, t, noise_std, action_low, action_high,
               episode_length, zero_az, reset_center):
    action = jnp.clip(action, action_low, action_high)
    # zero_az: freeze z-action / z-noise / z-state (xy motion only).
    z_scale = jnp.where(zero_az, 0.0, 1.0).astype(action.dtype)
    action = action.at[..., 2].multiply(z_scale)
    eps = noise_std * jax.random.normal(key, shape=state.shape, dtype=state.dtype)
    eps = eps.at[..., 2].multiply(z_scale)
    next_state = state + action + eps
    next_state = next_state.at[..., 2].multiply(z_scale)
    next_t = t + 1
    done = next_t >= episode_length
    # Auto-reset finished envs to reset_center.
    reset_state = jnp.broadcast_to(reset_center, next_state.shape)
    out_state = jnp.where(done[:, None], reset_state, next_state)
    out_t = jnp.where(done, jnp.zeros_like(next_t), next_t)
    reward = jnp.zeros(state.shape[0], dtype=state.dtype)
    # Return pre-reset next_state as terminal_obs for episode buffers.
    return out_state, reward, done, next_state, out_t

  def reset(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Reset all envs (or those with ``mask`` True). Returns obs ``(E, 3)``."""
    if mask is None:
      mask = np.ones(self.num_envs, dtype=bool)
    else:
      mask = np.asarray(mask, dtype=bool)
    key = self._split_key()
    state_j, t_j = self._reset_jit(
        key,
        jnp.asarray(mask),
        jnp.asarray(self._state),
        jnp.asarray(self._t),
        jnp.asarray(self.reset_scale, dtype=jnp.float32),
        jnp.asarray(self.reset_center, dtype=jnp.float32),
    )
    self._state = np.asarray(state_j, dtype=np.float32)
    self._t = np.asarray(t_j, dtype=np.int32)
    if self.single_axis_xy:
      self._resample_axes(mask)
    return self._state.copy()

  def step(
      self, action: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step all envs.

    Returns:
      next_obs:     (E, 3) post-auto-reset observation
      reward:       (E,) zeros
      done:         (E,) bool, True on horizon truncate
      terminal_obs: (E, 3) pre-reset next state (for episode buffers)
      info_reward:  (E,) NaNs (no dense env reward)
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (self.num_envs, self.ACTION_DIM):
      raise ValueError(
          f'action shape {action.shape} != {(self.num_envs, self.ACTION_DIM)}')
    key = self._split_key()
    out_state, reward, done, terminal, out_t = self._step_jit(
        key,
        jnp.asarray(self._state),
        jnp.asarray(action),
        jnp.asarray(self._t),
        jnp.asarray(self.noise_std, dtype=jnp.float32),
        jnp.asarray(self.action_low, dtype=jnp.float32),
        jnp.asarray(self.action_high, dtype=jnp.float32),
        jnp.asarray(self.episode_length, dtype=jnp.int32),
        jnp.asarray(self.zero_az),
        jnp.asarray(self.reset_center, dtype=jnp.float32),
    )
    self._state = np.asarray(out_state, dtype=np.float32)
    self._t = np.asarray(out_t, dtype=np.int32)
    done_np = np.asarray(done, dtype=bool)
    if self.single_axis_xy and done_np.any():
      # New episode after auto-reset → new axis lock.
      self._resample_axes(done_np)
    info_rew = np.full(self.num_envs, np.nan, dtype=np.float32)
    return (
        self._state.copy(),
        np.asarray(reward, dtype=np.float32),
        done_np,
        np.asarray(terminal, dtype=np.float32),
        info_rew,
    )

  def sample_uniform_actions(self, rng: np.random.Generator) -> np.ndarray:
    """Uniform random actions in ``[action_low, action_high]^3``.

    When ``zero_az`` is set, the z-component is forced to 0.
    When ``force_az_positive`` is set, a_z ~ Uniform(0, action_high].
    When ``single_axis_xy`` is set, each env zeros the inactive axis
    (x-only or y-only for the current episode).
    """
    actions = rng.uniform(
        self.action_low, self.action_high,
        size=(self.num_envs, self.ACTION_DIM)).astype(np.float32)
    if self.zero_az:
      actions[:, 2] = 0.0
    elif self.force_az_positive:
      # Open at 0 so a_z > 0 almost surely.
      actions[:, 2] = rng.uniform(
          1e-6, self.action_high, size=(self.num_envs,)).astype(np.float32)
    if self.single_axis_xy:
      # axis==0 → keep x, zero y; axis==1 → keep y, zero x.
      x_only = self._axis == 0
      actions[x_only, 1] = 0.0
      actions[~x_only, 0] = 0.0
    return actions
