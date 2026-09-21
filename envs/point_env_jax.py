"""Batched JAX vec-env for the ``point_*`` mazes.

Matches ``envs/point_env.py::PointEnv`` dynamics (10 axis-wise substeps,
0.01 maze-action noise, sparse success reward, time-limit done) and the
``VecEnv`` API used by ``contrastive.ppo_learner``:

    obs = vec_env.reset()                         # (E, 2 * state_dim)
    next_obs, rew, dones, terminal_obs, info_rew = vec_env.step(actions)

Numpy ``PointEnv`` stays for eval, MPO, RND, and maze PNG scripts.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp

from envs.point_env import WALLS, resize_walls


def point_spec_from_env_name(env_name: str) -> Dict[str, Any]:
  """Mirror ``env_utils.load`` kwargs / horizon for ``point_*`` names."""
  if not env_name.startswith('point_'):
    raise ValueError(f'expected point_* env, got {env_name!r}')
  extra_dims = 0
  goal_tolerance = 1.0
  if env_name == 'point_SixteenRooms4D':
    walls_name = 'SixteenRooms'
    extra_dims = 1
    goal_tolerance = 2.0
  elif env_name == 'point_SixteenRoomsActual4D':
    walls_name = 'SixteenRooms'
    extra_dims = 2
    goal_tolerance = 2.0
  else:
    walls_name = env_name.split('_')[-1]
  if walls_name not in WALLS:
    raise ValueError(
        f'unknown maze walls {walls_name!r} for env {env_name!r}; '
        f'known={sorted(WALLS)}')
  if 'SixteenRooms' in env_name:
    max_episode_steps = 200
  elif ('11x11' in env_name or '9x9' in env_name or '7x7' in env_name
        or 'Impossible' in env_name or 'EightRooms' in env_name):
    max_episode_steps = 100
  else:
    max_episode_steps = 50
  return dict(
      walls_name=walls_name,
      extra_dims=int(extra_dims),
      goal_tolerance=float(goal_tolerance),
      max_episode_steps=int(max_episode_steps),
  )


def _walls_array(walls_name: str, resize_factor: int = 1) -> np.ndarray:
  walls = WALLS[walls_name]
  if int(resize_factor) > 1:
    walls = resize_walls(walls, int(resize_factor))
  return np.asarray(walls, dtype=np.int32)


def _free_cells(walls: np.ndarray) -> np.ndarray:
  rows, cols = np.where(np.asarray(walls) == 0)
  if rows.size == 0:
    raise ValueError('maze has no free cells')
  return np.stack([rows, cols], axis=1).astype(np.int32)


def _split_keys(keys: jax.Array) -> Tuple[jax.Array, jax.Array]:
  """Split a batch of PRNG keys. ``keys`` is ``(E, 2)``."""
  pair = jax.vmap(jax.random.split)(keys)  # (E, 2, 2)
  return pair[:, 0], pair[:, 1]


def _is_blocked(pos: jnp.ndarray, walls: jnp.ndarray,
                height: int, width: int) -> jnp.ndarray:
  """Vectorized ``PointEnv._is_blocked`` on the first two (maze) dims."""
  pos2 = pos[..., :2]
  high = jnp.array([float(height), float(width)], dtype=pos2.dtype)
  out = (pos2 < 0.0).any(axis=-1) | (pos2 > high).any(axis=-1)
  ij = jnp.floor(pos2).astype(jnp.int32)
  ij = jnp.clip(
      ij, 0, jnp.array([height - 1, width - 1], dtype=jnp.int32))
  wall = walls[ij[..., 0], ij[..., 1]] == 1
  return out | wall


def _move_axis(pos: jnp.ndarray, maze_act: jnp.ndarray, axis: int,
               walls: jnp.ndarray, height: int, width: int,
               dt: float) -> jnp.ndarray:
  new = pos.at[:, axis].add(dt * maze_act[:, axis])
  blocked = _is_blocked(new, walls, height, width)
  return jnp.where(blocked[:, None], pos, new)


def _sample_empty(key: jax.Array, free_cells: jnp.ndarray,
                  extra_dims: int) -> jnp.ndarray:
  k_idx, k_off = jax.random.split(key)
  idx = jax.random.randint(k_idx, (), 0, free_cells.shape[0])
  xy = free_cells[idx].astype(jnp.float32) + jax.random.uniform(
      k_off, (2,), minval=0.0, maxval=1.0)
  if extra_dims > 0:
    return jnp.concatenate(
        [xy, jnp.zeros((extra_dims,), dtype=jnp.float32)], axis=0)
  return xy


def _pack_obs(pos: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
  return jnp.concatenate([pos, goal], axis=-1)


class JaxPointVecEnv:
  """JAX-batched point-maze vec-env with auto-reset on the time limit."""

  _NUM_SUBSTEPS = 10
  _EXTRA_BOUND = 20.0

  def __init__(
      self,
      env_name: str,
      num_envs: int,
      seed: int,
      fixed_start_end: Optional[Any] = None,
      resize_factor: int = 1,
      action_noise: float = 0.01,
      randomize_start: bool = False,
      start_jitter_cells: int = 0,
      start_in_cell_jitter: float = 0.0,
      start_cells: Optional[Any] = None,
  ):
    spec = point_spec_from_env_name(env_name)
    walls_np = _walls_array(spec['walls_name'], resize_factor)
    free_np = _free_cells(walls_np)
    height, width = int(walls_np.shape[0]), int(walls_np.shape[1])
    extra_dims = int(spec['extra_dims'])
    state_dim = 2 + extra_dims
    max_steps = int(spec['max_episode_steps'])
    goal_tol = float(spec['goal_tolerance'])
    noise_std = float(action_noise)
    dt = 1.0 / float(self._NUM_SUBSTEPS)
    extra_bound = float(self._EXTRA_BOUND)

    self._env_name = str(env_name)
    self._num_envs = int(num_envs)
    self._state_dim = int(state_dim)
    self._extra_dims = extra_dims
    self._obs_dim_total = 2 * state_dim
    self._action_dim = state_dim
    self._episode_length = max_steps
    self._walls_np = walls_np
    self._height = height
    self._width = width
    self._goal_tolerance = goal_tol
    self._action_noise = noise_std
    self._randomize_start = bool(randomize_start)
    self._start_jitter_cells = int(start_jitter_cells)
    self._start_in_cell_jitter = float(start_in_cell_jitter)
    self._start_cells = None

    walls_j = jnp.asarray(walls_np, dtype=jnp.int32)
    free_j = jnp.asarray(free_np, dtype=jnp.int32)
    use_fixed = (fixed_start_end is not None
                 and len(fixed_start_end) == 2)
    rand_start = bool(randomize_start)
    jitter_cells = int(start_jitter_cells)
    in_cell_jitter = float(max(start_in_cell_jitter, 0.0))
    if use_fixed:
      start_j = jnp.asarray(
          np.asarray(fixed_start_end[0], dtype=np.float32).reshape(-1),
          dtype=jnp.float32)
      goal_j = jnp.asarray(
          np.asarray(fixed_start_end[1], dtype=np.float32).reshape(-1),
          dtype=jnp.float32)
      if int(start_j.size) != state_dim or int(goal_j.size) != state_dim:
        raise ValueError(
            f'fixed_start_end dims {int(start_j.size)}/{int(goal_j.size)} '
            f'!= state_dim={state_dim} for {env_name}')
    else:
      start_j = jnp.zeros((state_dim,), dtype=jnp.float32)
      goal_j = jnp.zeros((state_dim,), dtype=jnp.float32)

    start_cells_np = None
    if start_cells is not None:
      start_cells_np = np.asarray(start_cells, dtype=np.int32).reshape(-1, 2)
      if start_cells_np.size == 0:
        start_cells_np = None
      else:
        for r, c in start_cells_np:
          if not (0 <= int(r) < height and 0 <= int(c) < width):
            raise ValueError(
                f'start cell ({int(r)}, {int(c)}) out of bounds '
                f'{height}x{width} for {env_name}')
          if int(walls_np[int(r), int(c)]) != 0:
            raise ValueError(
                f'start cell ({int(r)}, {int(c)}) is a wall in {env_name}')

    start_free_np = free_np
    if start_cells_np is not None:
      start_free_np = start_cells_np
      rand_start = True
      in_cell_jitter = 0.0
      self._start_cells = tuple(
          (int(r), int(c)) for r, c in start_cells_np)
    elif rand_start and jitter_cells > 0 and use_fixed:
      origin = np.floor(np.asarray(fixed_start_end[0], dtype=np.float32)[:2])
      cheb = np.max(np.abs(free_np.astype(np.float32) - origin[None, :]),
                    axis=1)
      local = free_np[cheb <= float(jitter_cells)]
      if local.size == 0:
        raise ValueError(
            f'no free cells within {jitter_cells} of start {origin.tolist()}')
      start_free_np = local
    start_free_j = jnp.asarray(start_free_np, dtype=jnp.int32)

    def _reset_one(key: jax.Array):
      if use_fixed and in_cell_jitter > 0.0:
        k_pos, k_rest = jax.random.split(key)
        off = jax.random.uniform(
            k_pos, (2,), minval=0.0, maxval=in_cell_jitter)
        pos = start_j
        pos = pos.at[:2].add(off)
        cell0 = jnp.floor(start_j[:2])
        lo = cell0
        hi = cell0 + (1.0 - 1e-5)
        pos = pos.at[:2].set(jnp.clip(pos[:2], lo, hi))
        return pos, goal_j, k_rest
      if use_fixed and not rand_start:
        return start_j, goal_j, key
      if use_fixed and rand_start:
        k_pos, k_rest = jax.random.split(key)
        pos = _sample_empty(k_pos, start_free_j, extra_dims)
        return pos, goal_j, k_rest
      k_pos, k_goal, k_rest = jax.random.split(key, 3)
      pos = _sample_empty(k_pos, free_j, extra_dims)
      goal = _sample_empty(k_goal, free_j, extra_dims)
      return pos, goal, k_rest

    def _reset_batch(keys: jax.Array):
      pos, goal, keys = jax.vmap(_reset_one)(keys)
      t = jnp.zeros((keys.shape[0],), dtype=jnp.int32)
      return {'pos': pos, 'goal': goal, 't': t, 'key': keys}

    def _step_batch(state: Dict[str, jnp.ndarray], actions: jnp.ndarray):
      pos = state['pos']
      goal = state['goal']
      t = state['t']
      keys = state['key']
      keys, k_noise = _split_keys(keys)
      if noise_std > 0.0:
        noise = jax.vmap(
            lambda k: jax.random.normal(k, (2,), dtype=jnp.float32)
        )(k_noise) * noise_std
      else:
        noise = jnp.zeros((pos.shape[0], 2), dtype=jnp.float32)
      act = jnp.asarray(actions, dtype=jnp.float32)
      maze_act = jnp.clip(act[:, :2] + noise, -1.0, 1.0)

      def _one_substep(carry, _):
        p = carry
        p = _move_axis(p, maze_act, 0, walls_j, height, width, dt)
        p = _move_axis(p, maze_act, 1, walls_j, height, width, dt)
        return p, None

      pos, _ = jax.lax.scan(
          _one_substep, pos, None, length=int(self._NUM_SUBSTEPS))
      if extra_dims > 0:
        extra_act = jnp.clip(act[:, 2:], -1.0, 1.0)
        pos = pos.at[:, 2:].add(extra_act)
        pos = pos.at[:, 2:].set(jnp.clip(pos[:, 2:], 0.0, extra_bound))

      t = t + 1
      dist = jnp.linalg.norm(goal - pos, axis=-1)
      rew = (dist < goal_tol).astype(jnp.float32)
      done = t >= max_steps
      terminal_obs = _pack_obs(pos, goal)

      keys, k_reset = _split_keys(keys)
      reset_state = _reset_batch(k_reset)
      pos = jnp.where(done[:, None], reset_state['pos'], pos)
      goal = jnp.where(done[:, None], reset_state['goal'], goal)
      t = jnp.where(done, reset_state['t'], t)
      next_obs = _pack_obs(pos, goal)
      new_state = {'pos': pos, 'goal': goal, 't': t, 'key': keys}
      return new_state, next_obs, rew, done, terminal_obs

    self._reset_fn = jax.jit(_reset_batch)
    self._step_fn = jax.jit(_step_batch)
    self._rng = jax.random.PRNGKey(int(seed))
    self._state: Optional[Dict[str, jnp.ndarray]] = None

  def reset(self) -> np.ndarray:
    self._rng, subkey = jax.random.split(self._rng)
    keys = jax.random.split(subkey, self._num_envs)
    self._state = self._reset_fn(keys)
    return np.asarray(
        _pack_obs(self._state['pos'], self._state['goal']),
        dtype=np.float32)

  def step(
      self, actions: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if self._state is None:
      raise RuntimeError('step() called before reset()')
    actions = np.asarray(actions, dtype=np.float32)
    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = np.clip(actions, -1.0, 1.0)
    self._state, next_obs, rew, done, term = self._step_fn(
        self._state, jnp.asarray(actions, dtype=jnp.float32))
    dones = np.asarray(done, dtype=bool)
    next_np = np.asarray(next_obs, dtype=np.float32)
    term_np = np.asarray(term, dtype=np.float32)
    term_np = np.where(dones[:, None], term_np, next_np)
    return (
        next_np,
        np.asarray(rew, dtype=np.float32),
        dones,
        term_np,
        np.full(self._num_envs, np.nan, dtype=np.float32),
    )

  @property
  def walls(self) -> np.ndarray:
    return self._walls_np

  @property
  def episode_length(self) -> int:
    return self._episode_length

  @property
  def num_envs(self) -> int:
    return self._num_envs

  @property
  def observation_shape(self) -> Tuple[int, ...]:
    return (self._obs_dim_total,)

  @property
  def action_shape(self) -> Tuple[int, ...]:
    return (self._action_dim,)


def compare_numpy_vs_jax(
    env_name: str = 'point_Impossible',
    n_steps: int = 80,
    atol: float = 1e-5,
) -> float:
  """Noise-off lockstep: numpy ``PointEnv`` vs ``JaxPointVecEnv`` (E=1)."""
  import env_utils

  if env_name == 'point_Impossible':
    fse = [np.array([0.0, 0.0], dtype=float),
           np.array([6.0, 8.0], dtype=float)]
  else:
    from ppo_contrastive import fixed_goal_dict
    fse = fixed_goal_dict[env_name]
  gym_env, _, _ = env_utils.load(env_name, fixed_start_end=fse, seed=0)
  gym_env._action_noise = 0.0
  n_obs = np.asarray(gym_env.reset(), dtype=np.float32)

  vec = JaxPointVecEnv(
      env_name, num_envs=1, seed=0, fixed_start_end=fse, action_noise=0.0)
  j_obs = vec.reset()
  max_diff = float(np.max(np.abs(n_obs - j_obs[0])))
  rng = np.random.RandomState(0)
  for _ in range(int(n_steps)):
    act = rng.uniform(-1.0, 1.0, size=gym_env.action_space.shape).astype(
        np.float32)
    n_obs, n_rew, n_done, _ = gym_env.step(act.copy())
    if n_done:
      n_obs = np.asarray(gym_env.reset(), dtype=np.float32)
    j_obs, j_rew, j_done, _, _ = vec.step(act[None])
    max_diff = max(
        max_diff,
        float(np.max(np.abs(np.asarray(n_obs) - j_obs[0]))),
        float(abs(float(n_rew) - float(j_rew[0]))),
        float(abs(float(bool(n_done)) - float(bool(j_done[0])))),
    )
  if max_diff > float(atol):
    raise AssertionError(
        f'numpy vs JAX mismatch on {env_name}: max_diff={max_diff:g} '
        f'> atol={atol:g}')
  return max_diff
