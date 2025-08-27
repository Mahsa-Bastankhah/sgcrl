import gym
import numpy as np
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

class PointEnv3D(gym.Env):
    """
    3D point navigation:
      - Walls/blocking logic applies ONLY to (x, y) using the same 2D grid.
      - z-axis has no walls; same action dynamics as x and y.
      - Start and goal are forced to z = 0.
      - Observation = [x, y, z, gx, gy, gz]
      - Action = 3D vector in [-1, 1]^3 applied with the same substep integration.
    """

    def __init__(self,
                 walls=None,
                 resize_factor: int = 1,
                 fixed_start_end: Optional[tuple] = None,
                 z_max: float = 10.0,
                 action_noise: float = 0.01,
                 noisy_tv: bool = True,
                 noisy_tv_sigma: float = 1.0):
        """
        Args:
          walls: name of a map in WALLS or a binary HxW array; used for (x,y) only.
          resize_factor: integer scale factor for the (x,y) map.
          fixed_start_end: optional ((sx, sy[, sz]), (gx, gy[, gz])).
                           z components (if provided) are ignored and forced to 0.
          z_max: upper bound for z (lower bound is 0).
          action_noise: std of Gaussian action noise added before clipping.
        """
        # Handle walls / resizing compatible with your 2D env
        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]
        (height, width) = self._walls.shape
        self._height = height
        self._width = width

        self._z_min = 0.0
        self._z_max = float(z_max)

        self._action_noise = float(action_noise)
        self._fixed_start_end = fixed_start_end

        self._noisy_tv = bool(noisy_tv)
        self._noisy_tv_sigma = float(noisy_tv_sigma)
        if self._noisy_tv:
            print(f"[INIT] Noisy-TV mode enabled (σ={self._noisy_tv_sigma}). Actions are 2D only.")
        else:
            print("[INIT] Standard 3D mode enabled.")

        # 3D continuous actions
        # self.action_space = gym.spaces.Box(
        #     low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        #     high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        #     dtype=np.float32
        # )
                # 3D or 2D continuous actions depending on noisy_tv
        if self._noisy_tv:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([ 1.0,  1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
                dtype=np.float32
            )


        # Observations: [x, y, z, gx, gy, gz]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, self._z_min, 0.0, 0.0, self._z_min], dtype=np.float32),
            high=np.array([float(height), float(width), self._z_max,
                           float(height), float(width), self._z_max], dtype=np.float32),
            dtype=np.float32
        )

        self._timestep = 0
        self._max_episode_steps = 100 if (isinstance(walls, str) and '11x11' in walls) else 50
        self.reset()

    # --- helpers (x,y use 2D walls; z is free but bounded) ---
    def _in_bottom_left_room(self, xy):
        # Generic "bottom-left room" as the bottom-left quadrant.
        # x ~ rows (downwards), y ~ cols (rightwards)
        return (xy[0] >= self._height / 2.0) and (xy[1] < self._width / 2.0)

    def _apply_noisy_tv_z(self):
        if not self._noisy_tv:
            return
        prev_z = self.state[2]
        if self._in_bottom_left_room(self.state[:2]):
            z = np.random.normal(0.0, self._noisy_tv_sigma)
            self.state[2] = float(np.clip(z, self._z_min, self._z_max))
            #print(f"[NOISY_TV] In bottom-left room → sampled z={self.state[2]:.3f} (prev {prev_z:.3f})")
        else:
            self.state[2] = 0.0
            #print(f"[NOISY_TV] Outside bottom-left room → forced z=0 (prev {prev_z:.3f})")

        # Always force goal z=0
        self.goal[2] = 0.0


    def _sample_empty_xy(self):
        """Sample a free (x,y) cell and add uniform sub-cell jitter."""
        candidate_states = np.where(self._walls == 0)
        num = len(candidate_states[0])
        idx = np.random.choice(num)
        xy = np.array([candidate_states[0][idx], candidate_states[1][idx]], dtype=float)
        xy += np.random.uniform(size=2)  # jitter inside the cell
        assert not self._is_blocked_xy(xy)
        return xy

    def _get_obs(self):
        #print(f"[OBS] pos=({self.state[0]:.3f}, {self.state[1]:.3f}, {self.state[2]:.3f}) ")
        return np.concatenate([self.state, self.goal]).astype(np.float32)

    def reset(self):
        self._timestep = 0

        if self._fixed_start_end is not None:
            s, g = self._fixed_start_end
            # accept 2D or 3D tuples; force z = 0
            sx, sy = float(s[0]), float(s[1])
            gx, gy = float(g[0]), float(g[1])
            self.state = np.array([sx, sy, 0.0], dtype=float)
            self.goal  = np.array([gx, gy, 0.0], dtype=float)
        else:
            xy_goal = self._sample_empty_xy()
            xy_start = self._sample_empty_xy()
            self.goal = np.array([xy_goal[0],  xy_goal[1],  0.0], dtype=float)
            self.state = np.array([xy_start[0], xy_start[1], 0.0], dtype=float)

        # Ensure within bounds
        self.state = self._clip_state(self.state)
        self.goal  = self._clip_state(self.goal)
        # Enforce noisy TV behavior on z at the start
        self._apply_noisy_tv_z()
        return self._get_obs()

    def _discretize_xy(self, xy, resolution=1.0):
        ij = np.floor(resolution * xy).astype(int)
        ij = np.clip(ij, [0, 0], [self._height - 1, self._width - 1])
        return ij.astype(int)

    def _is_blocked_xy(self, xy):
        """Block check for (x,y) against walls; ignores z."""
        if (xy[0] < 0 or xy[0] > self._height or
            xy[1] < 0 or xy[1] > self._width):
            return True
        (i, j) = self._discretize_xy(xy)
        return (self._walls[i, j] == 1)

    def _clip_state(self, s):
        """Clip (x,y) to map bounds, z to [z_min, z_max]."""
        x = np.clip(s[0], 0.0, float(self._height))
        y = np.clip(s[1], 0.0, float(self._width))
        z = np.clip(s[2], self._z_min, self._z_max)
        return np.array([x, y, z], dtype=float)

    def step(self, action):
        action = np.array(action, dtype=np.float32).copy()

        # Validate/clip action with appropriate dimensionality
        if self._noisy_tv:
            if action.shape != (2,):
                # If someone passes a 3D action by mistake, ignore the z component
                action = action[:2]
            if self._action_noise > 0.0:
                action += np.random.normal(0.0, self._action_noise, size=(2,))
            action = np.clip(action, self.action_space.low, self.action_space.high)
            assert self.action_space.contains(action)
        else:
            if not self.action_space.contains(action):
                print('WARNING: clipping invalid action:', action)
            if self._action_noise > 0.0:
                action += np.random.normal(0.0, self._action_noise, size=(3,))
            action = np.clip(action, self.action_space.low, self.action_space.high)
            assert self.action_space.contains(action)

        num_substeps = 10
        dt = 1.0 / num_substeps

        for _ in np.linspace(0, 1, num_substeps):
            # x axis
            new_state = self.state.copy()
            new_state[0] += dt * action[0]
            if not self._is_blocked_xy(new_state[:2]):
                self.state[0] = new_state[0]

            # y axis
            new_state = self.state.copy()
            new_state[1] += dt * action[1]
            if not self._is_blocked_xy(new_state[:2]):
                self.state[1] = new_state[1]

            # z axis
            if not self._noisy_tv:
                new_state = self.state.copy()
                new_state[2] += dt * action[2]
                self.state[2] = np.clip(new_state[2], self._z_min, self._z_max)

        # Clip and enforce noisy-tv z after the dynamics
        self.state = self._clip_state(self.state)
        self._apply_noisy_tv_z()

        done = False
        obs = self._get_obs()

        if self._noisy_tv:
            # Reward ignores z because agent can't control it
            dist = np.linalg.norm(self.goal[:2] - self.state[:2])
        else:
            dist = np.linalg.norm(self.goal - self.state)

        self._last_end_pos = self.state.copy()
        self._timestep += 1
        rew = float(dist < 1.0)

        return obs, rew, done, {}


    @property
    def walls(self):
        return self._walls
