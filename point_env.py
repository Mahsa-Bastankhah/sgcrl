"""Utility for loading the 2D navigation environments."""
from typing import Optional

import gym
import numpy as np
import scipy

# Only Spiral11x11 is supported, but the other walls can easily be added 
# by adding the start and ending coordinates to the fixed_goal_dict in lp_contrastive.py
WALLS = { 
    'Small':  
        np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]),
    'Cross':  
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'Impossible':
        np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 1],
                  [0, 1, 0, 0, 0, 1, 0, 1, 0]]),
    'FourRooms':  
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]]),
    'U': 
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0],
                  [0, 0, 0]]),
    'Spiral7x7':
        np.array([[1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 1, 0],
                  [1, 0, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 0]]),
    'Spiral9x9':
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    'Spiral11x11':
        np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    # 4×4 room grid (21×21).  Same staggered wall pattern as EightRooms,
    # extended to 3 horizontal divider levels (rows 5-6, 10-11, 15-16).
    # Vertical dividers at cols 5, 10, 15; doors in vertical walls at rows
    # 2, 8, 13, 18.  Horizontal wall doors at cols 1, 8, 11, 18 (repeated
    # at each level).  Episode length is 2× EightRooms = 200 steps.
    'SixteenRooms':
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    # Two FourRooms grids placed side by side (11×21).
    # Vertical dividers at cols 5, 10, 15; horizontal dividers at rows 5–6
    # (staggered: left sections wall at row 5, right sections at row 6).
    # Doors in vertical walls at rows 2 and 9; horizontal wall doors at
    # cols 1, 8, 11, 18.  Episode length is 2× FourRooms = 100 steps.
    'EightRooms':
        np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    'Wall11x11':  
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    'Maze11x11':  
        np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                  [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                  [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(factor)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls


class PointEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               walls=None, resize_factor=1, fixed_start_end=None,
               extra_dims=0, goal_tolerance=1.0):
    """Initialize the point environment.

    Args:
      walls: (str or array) binary, H x W array indicating locations of walls.
        Can also be the name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
      extra_dims: (int) Number of additional continuous dimensions appended to
        the 2-D maze state.  The agent moves freely in these extra dims (no
        wall constraints).  Start and goal have extra dims fixed at 0; the
        agent must return there to earn reward (full Euclidean distance check).
      goal_tolerance: (float) Success radius in full-state Euclidean distance.
        Default 1.0.  Use a larger value (e.g. 2.0) for envs with extra dims
        so that small deviations in those dims don't prevent success.
    """
    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._action_noise = 0.01
    self._fixed_start_end = fixed_start_end
    self._extra_dims = int(extra_dims)
    self._goal_tolerance = float(goal_tolerance)
    act_dim = 2 + self._extra_dims
    self.action_space = gym.spaces.Box(
        low=np.full(act_dim, -1.0, dtype=np.float32),
        high=np.full(act_dim,  1.0, dtype=np.float32),
        dtype=np.float32)
    # Extra dims are bounded to [0, 20] and clamped in step().
    extra_bound = 20.0
    self._extra_bound = extra_bound
    self.observation_space = gym.spaces.Box(
        low=np.concatenate([
            np.array([0.0, 0.0], dtype=np.float32),
            np.full(self._extra_dims, 0.0, dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            np.full(self._extra_dims, 0.0, dtype=np.float32),
        ]),
        high=np.concatenate([
            np.array([height, width], dtype=np.float32),
            np.full(self._extra_dims, extra_bound, dtype=np.float32),
            np.array([height, width], dtype=np.float32),
            np.full(self._extra_dims, extra_bound, dtype=np.float32),
        ]),
        dtype=np.float32)
    self._timestep = 0
    if '11x11' in walls or '9x9' in walls or '7x7' in walls or walls == 'Impossible':
      self._max_episode_steps = 100
    elif walls == 'EightRooms':
      self._max_episode_steps = 100   # 2× FourRooms (11×21 grid)
    elif walls == 'SixteenRooms':
      self._max_episode_steps = 200   # 2× EightRooms (21×21 grid)
    else:
      self._max_episode_steps = 50
    self.reset()

  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state_2d = np.array([candidate_states[0][state_index],
                         candidate_states[1][state_index]],
                        dtype=float)
    state_2d += np.random.uniform(size=2)
    assert not self._is_blocked(state_2d)
    if self._extra_dims > 0:
      return np.concatenate([state_2d, np.zeros(self._extra_dims)])
    return state_2d

  def _get_obs(self):
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self):
    self._timestep = 0
    if self._fixed_start_end is not None:
      self.state = np.asarray(self._fixed_start_end[0], dtype=float).copy()
      self.goal  = np.asarray(self._fixed_start_end[1], dtype=float).copy()
    else:
      self.goal  = self._sample_empty_state()
      self.state = self._sample_empty_state()
    return self._get_obs()

  def _discretize_state(self, state, resolution=1.0):
    # Use only the first 2 (maze) dimensions for discretisation.
    ij = np.floor(resolution * state[:2]).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    pos2 = np.asarray(state[:2])
    if (np.any(pos2 < self.observation_space.low[:2])
        or np.any(pos2 > self.observation_space.high[:2])):
      return True
    (i, j) = self._discretize_state(pos2)
    return (self._walls[i, j] == 1)

  def step(self, action):
    action = action.copy()
    if not self.action_space.contains(action):
      print('WARNING: clipping invalid action:', action)
    maze_noise = np.random.normal(0, self._action_noise, 2) if self._action_noise > 0 else 0.0
    action[:2] = np.clip(action[:2] + maze_noise,
                         self.action_space.low[:2], self.action_space.high[:2])
    if self._extra_dims > 0:
      action[2:] = np.clip(action[2:],
                           self.action_space.low[2:], self.action_space.high[2:])

    num_substeps = 10
    dt = 1.0 / num_substeps

    # Maze dims: wall-constrained sub-stepped movement.
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(2):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        if not self._is_blocked(new_state):
          self.state = new_state

    # Extra dims: free movement (full step, no wall check), clamped to [0, bound].
    if self._extra_dims > 0:
      for d in range(self._extra_dims):
        self.state[2 + d] += action[2 + d]
        self.state[2 + d] = np.clip(self.state[2 + d], 0.0, self._extra_bound)

    done = False
    obs = self._get_obs()
    dist = np.linalg.norm(self.goal - self.state)
    self._last_end_pos = self.state
    self._timestep += 1
    rew = float(dist < self._goal_tolerance)
    return obs, rew, done, {}

  @property
  def walls(self):
    return self._walls