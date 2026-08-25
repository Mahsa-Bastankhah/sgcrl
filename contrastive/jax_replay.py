"""GPU (time, env) ring buffer + uniform-step / in-column geometric sampler.

Inserts a fused BuilderBench unroll as ``T`` rows. Sampling is uniform over
stored cells, then truncated-geometric future along that env column with the
same ``traj_id`` (not uniform-over-episodes).
"""
from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class TrajBuffer(NamedTuple):
  obs: jnp.ndarray       # (cap_T, E, obs_dim) state at t
  action: jnp.ndarray    # (cap_T, E, act_dim)
  next_obs: jnp.ndarray  # (cap_T, E, obs_dim) s_{t+1} or terminal
  traj_id: jnp.ndarray   # (cap_T, E) int32
  horizon: jnp.ndarray   # (cap_T, E) int32 max_d >= 1
  insert_ptr: jnp.ndarray
  size_t: jnp.ndarray


def _obs_to_goal(states, start_index, end_index, goal_idx):
  if goal_idx is not None:
    return states[..., goal_idx]
  if int(end_index) == -1:
    return states[..., int(start_index):]
  return states[..., int(start_index):int(end_index)]


def expand_traj_ids(traj_id_e: jnp.ndarray,
                    step_dones: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Per-step traj_id from env carry + dones. Done increments the *next* row."""
  incs = step_dones.astype(jnp.int32)
  prev = jnp.concatenate(
      [jnp.zeros((1, incs.shape[1]), dtype=jnp.int32), incs[:-1]], axis=0)
  ids = traj_id_e[None, :] + jnp.cumsum(prev, axis=0)
  new_id = traj_id_e + jnp.sum(incs, axis=0)
  return ids, new_id


def make_traj_replay(
    *,
    cap_T: int,
    num_envs: int,
    obs_dim: int,
    act_dim: int,
    discount: float,
    start_index: int,
    end_index: int,
    goal_state_indices=None,
):
  """Factory: jitted insert / sample closed over static shapes."""
  cap_T = int(cap_T)
  E = int(num_envs)
  obs_dim = int(obs_dim)
  act_dim = int(act_dim)
  gamma = float(discount)
  log_gamma = float(np.log(gamma))
  si = int(start_index)
  ei = int(end_index)
  gidx = None
  if goal_state_indices is not None:
    gidx = jnp.asarray(
        np.asarray(goal_state_indices).reshape(-1), dtype=jnp.int32)

  def init() -> TrajBuffer:
    return TrajBuffer(
        obs=jnp.zeros((cap_T, E, obs_dim), dtype=jnp.float32),
        action=jnp.zeros((cap_T, E, act_dim), dtype=jnp.float32),
        next_obs=jnp.zeros((cap_T, E, obs_dim), dtype=jnp.float32),
        traj_id=jnp.zeros((cap_T, E), dtype=jnp.int32),
        horizon=jnp.zeros((cap_T, E), dtype=jnp.int32),
        insert_ptr=jnp.zeros((), dtype=jnp.int32),
        size_t=jnp.zeros((), dtype=jnp.int32),
    )

  def _recompute_horizon(traj_id, insert_ptr, size_t):
    l = jnp.arange(cap_T, dtype=jnp.int32)
    phys = jnp.where(size_t == cap_T, (insert_ptr + l) % cap_T, l)
    tid_log = traj_id[phys]  # (cap_T, E)

    def body(h_next, l_rev):
      valid = l_rev < size_t
      valid_n = (l_rev + 1) < size_t
      l_n = jnp.minimum(l_rev + 1, cap_T - 1)
      same = valid & valid_n & (tid_log[l_rev] == tid_log[l_n])
      h = jnp.where(
          same, h_next + 1,
          jnp.where(valid, jnp.ones_like(h_next), jnp.zeros_like(h_next)))
      return h, h

    ls = jnp.arange(cap_T, dtype=jnp.int32)[::-1]
    _, h_rev = jax.lax.scan(
        body, jnp.zeros((E,), dtype=jnp.int32), ls)
    h_log = h_rev[::-1]
    horizon = jnp.zeros((cap_T, E), dtype=jnp.int32)
    horizon = horizon.at[phys].set(h_log)
    return horizon

  def _write(arr, rows, pos):
    t_idx = (pos + jnp.arange(rows.shape[0], dtype=jnp.int32)) % cap_T
    return arr.at[t_idx].set(rows)

  def insert(buf: TrajBuffer, obs, action, next_obs, traj_id_te) -> TrajBuffer:
    """Write a ``(T, E, …)`` unroll chunk."""
    pos = buf.insert_ptr
    t_len = obs.shape[0]
    obs_b = _write(buf.obs, obs.astype(jnp.float32), pos)
    act_b = _write(buf.action, action.astype(jnp.float32), pos)
    nxt_b = _write(buf.next_obs, next_obs.astype(jnp.float32), pos)
    tid_b = _write(buf.traj_id, traj_id_te.astype(jnp.int32), pos)
    new_ptr = (pos + t_len) % cap_T
    new_size = jnp.minimum(buf.size_t + t_len, jnp.asarray(cap_T, dtype=jnp.int32))
    horizon = _recompute_horizon(tid_b, new_ptr, new_size)
    return TrajBuffer(
        obs=obs_b, action=act_b, next_obs=nxt_b, traj_id=tid_b,
        horizon=horizon, insert_ptr=new_ptr, size_t=new_size)

  def _phys_plus(phys, k, size_t, insert_ptr):
    return jnp.where(size_t == cap_T, (phys + k) % cap_T, phys + k)

  def sample(buf: TrajBuffer, key, batch_size: int) -> Tuple[Dict[str, jnp.ndarray], TrajBuffer]:
    B = int(batch_size)
    n_cells = jnp.maximum(buf.size_t * E, 1)
    key, k_flat, k_d = jax.random.split(key, 3)
    u = jax.random.uniform(k_flat, (B,))
    flat = jnp.floor(u * n_cells.astype(jnp.float32)).astype(jnp.int32)
    flat = jnp.minimum(flat, n_cells - 1)
    logical = flat // E
    e = flat % E
    phys = jnp.where(
        buf.size_t == cap_T, (buf.insert_ptr + logical) % cap_T, logical)
    s_t = buf.obs[phys, e]
    a_t = buf.action[phys, e]
    s_tp1 = buf.next_obs[phys, e]
    h = jnp.maximum(buf.horizon[phys, e], 1)
    max_d = h.astype(jnp.float32)
    trunc = 1.0 - jnp.power(jnp.asarray(gamma, dtype=jnp.float32), max_d)
    u_d = jax.random.uniform(k_d, (B,)) * trunc
    d = 1 + jnp.floor(jnp.log1p(-u_d) / jnp.asarray(log_gamma, dtype=jnp.float32)).astype(jnp.int32)
    d = jnp.clip(d, 1, h)
    n_after = h - 1
    use_next = d > n_after
    k_obs = jnp.where(use_next, n_after, d)
    p_j = _phys_plus(phys, k_obs, buf.size_t, buf.insert_ptr)
    s_j = jnp.where(use_next[:, None], buf.next_obs[p_j, e], buf.obs[p_j, e])
    p1 = _phys_plus(phys, jnp.ones_like(n_after), buf.size_t, buf.insert_ptr)
    a_tp1 = jnp.where((n_after >= 1)[:, None], buf.action[p1, e], a_t)
    goals = _obs_to_goal(s_j, si, ei, gidx)
    obs_out = jnp.concatenate([s_t, goals], axis=-1)
    next_out = jnp.concatenate([s_tp1, goals], axis=-1)
    batch = {
        'obs': obs_out,
        'action': a_t,
        'next_obs': next_out,
        'next_action': a_tp1,
        'future_state': s_j,
    }
    return batch, buf

  def sample_n(buf: TrajBuffer, key, n_steps: int, batch_size: int):
    n_steps = int(n_steps)

    def body(carry, _):
      buf, key = carry
      key, k_s = jax.random.split(key)
      batch, buf = sample(buf, k_s, batch_size)
      return (buf, key), batch

    (buf, key), batches = jax.lax.scan(
        body, (buf, key), None, length=n_steps)
    return batches, buf, key

  def uniform_half_goals(batch, key, goal_low, goal_high):
    B = batch['obs'].shape[0]
    half = B // 2
    gdim = batch['obs'].shape[1] - obs_dim
    key, k_g = jax.random.split(key)
    u = jax.random.uniform(k_g, (half, gdim))
    low = jnp.asarray(goal_low, dtype=jnp.float32).reshape(-1)
    high = jnp.asarray(goal_high, dtype=jnp.float32).reshape(-1)
    uni = low + u * (high - low)
    obs = batch['obs'].at[half:, obs_dim:].set(uni)
    nxt = batch['next_obs'].at[half:, obs_dim:].set(uni)
    return {**batch, 'obs': obs, 'next_obs': nxt}, key

  def apply_uniform_n(batches, key, goal_low, goal_high):
    def body(k, batch):
      batch, k = uniform_half_goals(batch, k, goal_low, goal_high)
      return k, batch
    key, batches = jax.lax.scan(body, key, batches)
    return batches, key

  insert_jit = jax.jit(insert)
  sample_jit = jax.jit(sample, static_argnums=(2,))
  sample_n_jit = jax.jit(sample_n, static_argnums=(2, 3))
  uni_jit = jax.jit(uniform_half_goals)
  uni_n_jit = jax.jit(apply_uniform_n)
  expand_jit = jax.jit(expand_traj_ids)

  class _Api:
    pass

  _Api.cap_T = cap_T
  _Api.num_envs = E
  _Api.obs_dim = obs_dim
  _Api.init = staticmethod(init)
  _Api.insert = staticmethod(insert_jit)
  _Api.sample = staticmethod(sample_jit)
  _Api.sample_n = staticmethod(sample_n_jit)
  _Api.uniform_half_goals = staticmethod(uni_jit)
  _Api.apply_uniform_n = staticmethod(uni_n_jit)
  _Api.expand_traj_ids = staticmethod(expand_jit)

  return _Api()


def host_filled_after_insert(filled_t: int, unroll_t: int, cap_T: int) -> int:
  return min(int(filled_t) + int(unroll_t), int(cap_T))
