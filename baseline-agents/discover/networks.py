"""TD3 actors + ensemble Q, matching DISCOVER ``td3_networks.py``.

BuilderBench-only extras: ``Actor`` (catselect). The paper arm policy is
``TanhActor`` (MLP + tanh, lecun_uniform, collect noise clipped then added).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from distributional import select_cube_centers

_KERNEL = jax.nn.initializers.lecun_uniform()


class Actor(nn.Module):
  """Deterministic TD3 actor with a categorical cube-select head.

  Continuous dims (xyz+yaw) are tanh-bounded. Select is a softmax over
  ``n_cubes`` classes, mapped to the PD ``select_action`` bin center so the
  env API stays in ``[-1, 1]^5``.
  """
  n_cubes: int
  n_continuous: int = 4
  hidden: Sequence[int] = (256, 256)

  @nn.compact
  def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x = obs
    for i, w in enumerate(self.hidden):
      x = nn.relu(nn.Dense(w, name=f'h{i}', kernel_init=_KERNEL)(x))
    mean = jnp.tanh(nn.Dense(self.n_continuous, name='mean', kernel_init=_KERNEL)(x))
    logits = nn.Dense(self.n_cubes, name='select_logits', kernel_init=_KERNEL)(x)
    return mean, logits


def pack_action(mean: jnp.ndarray, cube_id: jnp.ndarray, n_cubes: int) -> jnp.ndarray:
  centers = select_cube_centers(n_cubes)
  select = centers[cube_id.astype(jnp.int32)][..., None]
  return jnp.concatenate([mean, select], axis=-1)


def actor_action(
    mean: jnp.ndarray,
    logits: jnp.ndarray,
    key: jax.Array,
    n_cubes: int,
    explore: bool,
    exploration_noise: float,
    noise_clip: float,
) -> jnp.ndarray:
  """Sample (explore) or take the mode (eval) and pack a 5-D PD action."""
  if explore:
    key_n, key_c = jax.random.split(key)
    noise = (jax.random.normal(key_n, mean.shape) * exploration_noise).clip(
        -noise_clip, noise_clip)
    cont = jnp.clip(mean + noise, -1.0, 1.0)
    cube_id = jax.random.categorical(key_c, logits)
  else:
    cont = mean
    cube_id = jnp.argmax(logits, axis=-1)
  return pack_action(cont, cube_id, n_cubes)


def actor_det_action(mean: jnp.ndarray, logits: jnp.ndarray, n_cubes: int) -> jnp.ndarray:
  return pack_action(mean, jnp.argmax(logits, axis=-1), n_cubes)


def pack_all_selects(mean: jnp.ndarray, n_cubes: int) -> jnp.ndarray:
  """Same xyz/yaw, every cube bin center. Shape ``(B, n_cubes, 5)``."""
  centers = select_cube_centers(n_cubes)
  b = mean.shape[0]
  mean_rep = jnp.broadcast_to(mean[:, None, :], (b, n_cubes, mean.shape[-1]))
  select = jnp.broadcast_to(centers[None, :, None], (b, n_cubes, 1))
  return jnp.concatenate([mean_rep, select], axis=-1)


class CatWpActor(nn.Module):
  """PPO-style catwp: per-cube tanh xyz, shared yaw, categorical select."""
  n_cubes: int
  hidden: Sequence[int] = (256, 256)

  @nn.compact
  def __call__(
      self, obs: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = obs
    for i, w in enumerate(self.hidden):
      x = nn.relu(nn.Dense(w, name=f'h{i}', kernel_init=_KERNEL)(x))
    wp = jnp.tanh(nn.Dense(self.n_cubes * 3, name='wp', kernel_init=_KERNEL)(x))
    wp = wp.reshape(obs.shape[:-1] + (self.n_cubes, 3))
    yaw = jnp.tanh(nn.Dense(1, name='yaw', kernel_init=_KERNEL)(x))
    logits = nn.Dense(self.n_cubes, name='select_logits', kernel_init=_KERNEL)(x)
    return wp, yaw, logits


def _gather_wp(wp: jnp.ndarray, cube_id: jnp.ndarray) -> jnp.ndarray:
  idx = cube_id.astype(jnp.int32)[..., None, None]
  idx = jnp.broadcast_to(idx, wp.shape[:-2] + (1, wp.shape[-1]))
  return jnp.take_along_axis(wp, idx, axis=-2)[..., 0, :]


def pack_all_catwp(
    wp: jnp.ndarray, yaw: jnp.ndarray, n_cubes: int) -> jnp.ndarray:
  """Per-cube xyz + shared yaw + that cube's select center. ``(B, n, 5)``."""
  centers = select_cube_centers(n_cubes)
  b = wp.shape[0]
  yaw_rep = jnp.broadcast_to(yaw[:, None, :], (b, n_cubes, 1))
  select = jnp.broadcast_to(centers[None, :, None], (b, n_cubes, 1))
  return jnp.concatenate([wp, yaw_rep, select], axis=-1)


def catwp_action(
    wp: jnp.ndarray,
    yaw: jnp.ndarray,
    logits: jnp.ndarray,
    key: jax.Array,
    n_cubes: int,
    explore: bool,
    exploration_noise: float,
    noise_clip: float,
) -> jnp.ndarray:
  if explore:
    key_n, key_c = jax.random.split(key)
    cube_id = jax.random.categorical(key_c, logits)
    xyz = _gather_wp(wp, cube_id)
    cont = jnp.concatenate([xyz, yaw], axis=-1)
    noise = (jax.random.normal(key_n, cont.shape) * exploration_noise).clip(
        -noise_clip, noise_clip)
    cont = jnp.clip(cont + noise, -1.0, 1.0)
  else:
    cube_id = jnp.argmax(logits, axis=-1)
    xyz = _gather_wp(wp, cube_id)
    cont = jnp.concatenate([xyz, yaw], axis=-1)
  return pack_action(cont, cube_id, n_cubes)


def catwp_det_action(
    wp: jnp.ndarray, yaw: jnp.ndarray, logits: jnp.ndarray, n_cubes: int
) -> jnp.ndarray:
  cube_id = jnp.argmax(logits, axis=-1)
  xyz = _gather_wp(wp, cube_id)
  cont = jnp.concatenate([xyz, yaw], axis=-1)
  return pack_action(cont, cube_id, n_cubes)


class TanhActor(nn.Module):
  """Original DISCOVER TD3 policy: MLP then tanh on all action dims.

  For BuilderBench PD that is 5-D: xyz, yaw, and a continuous select channel
  that the env bins onto a cube.
  """
  action_dim: int = 5
  hidden: Sequence[int] = (256, 256)

  @nn.compact
  def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
    x = obs
    for i, w in enumerate(self.hidden):
      x = nn.relu(nn.Dense(w, name=f'h{i}', kernel_init=_KERNEL)(x))
    return jnp.tanh(nn.Dense(self.action_dim, name='mean', kernel_init=_KERNEL)(x))


def tanh_actor_action(
    mean: jnp.ndarray,
    key: jax.Array,
    explore: bool,
    exploration_noise: float,
    noise_clip: float,
) -> jnp.ndarray:
  # DISCOVER ``make_policy_td3``: clip noise, add, do not clip the action.
  if explore:
    noise = (jax.random.normal(key, mean.shape) * exploration_noise).clip(
        -noise_clip, noise_clip)
    return mean + noise
  return mean


class EnsembleQ(nn.Module):
  """K independent Q(s,g,a) heads. Output shape ``(..., K)``."""
  n_critics: int = 6
  hidden: Sequence[int] = (256, 256)
  activate_final: bool = True

  @nn.compact
  def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    x0 = jnp.concatenate([obs, action], axis=-1)
    qs = []
    for k in range(self.n_critics):
      h = x0
      for i, w in enumerate(self.hidden):
        h = nn.relu(nn.Dense(w, name=f'c{k}_h{i}', kernel_init=_KERNEL)(h))
      q = nn.Dense(1, name=f'c{k}_out', kernel_init=_KERNEL)(h)
      if self.activate_final:
        q = nn.softplus(q)
      qs.append(q)
    return jnp.concatenate(qs, axis=-1)
