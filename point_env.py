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


class PointEnv(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               walls = None, resize_factor = 1, fixed_start_end = None):
    """Initialize the point environment.

    Args:
      walls: (str or array) binary, H x W array indicating locations of walls.
        Can also be the name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
    """
    if resize_factor > 1:
      print("resize_factor", resize_factor)
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width
    self._depth = 1
    self._action_noise = 0.01
    self._fixed_start_end = fixed_start_end
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0, -1]),
        high=np.array([1.0, 1.0, 1]),
        dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.array([0, 0, 0, 0, 0, 0]),
        high=np.array([height, width, self._depth, height, width, self._depth]),
        dtype=np.float32)
    self._timestep = 0
    if '11x11' in walls:
      self._max_episode_steps = 100
    else:
      self._max_episode_steps = 50
    self.reset()

## for sampling goals always use depth 0 and don't add any randomness just for simplicity
## note that function is costumized for the 3D env
  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index],
                      0],
                     dtype=float)
    # state[0] += np.random.uniform()
    # state[1] += np.random.uniform()
    assert not self._is_blocked(state[:2])
    return state

  def _get_obs(self):
    
    return np.concatenate([self.state, self.goal]).astype(np.float32)

  def reset(self):
    # wall_positions = np.argwhere(self._walls == 1)
    # for pos in wall_positions:
    #     print(tuple(pos))
    self._timestep = 0
    
    if self._fixed_start_end is not None:
        # fix the starting and ending position of the agent
        self.state = self._fixed_start_end[0]
        self.goal = self._fixed_start_end[1]
    else:
        self.goal = self._sample_empty_state()
        self.state = self._sample_empty_state()
    return self._get_obs()

  def _discretize_state(self, state, resolution=1.0):
    ij = np.floor(resolution * state).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(self.walls.shape) - 1)
    return ij.astype(int)

  def _is_blocked(self, state):
    assert len(state) == 2
    if (np.any(state < self.observation_space.low[:2])
        or np.any(state > self.observation_space.high[:2])):
      return True
    (i, j) = self._discretize_state(state)
    return (self._walls[i, j] == 1)

  def step(self, action):
    action = action.copy()
    if not self.action_space.contains(action):
      print('WARNING: clipping invalid action:', action)
    if self._action_noise > 0:
      action += np.random.normal(0, self._action_noise, action.shape)
    action = np.clip(action, self.action_space.low, self.action_space.high)
    assert self.action_space.contains(action)
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = self.state.copy()
        new_state[axis] += dt * action[axis]
        
        # Only block in x and y directions
        if axis < 2:
            if not self._is_blocked(new_state[:2]):
                self.state = new_state
        else:
            # z axis is always allowed but clipped between [0, 1]
            new_state[2] = np.clip(new_state[2], 0, 1)
            self.state = new_state

    done = False
    obs = self._get_obs()
    dist = np.linalg.norm(self.goal - self.state)
    self._last_end_pos = self.state
    self._timestep += 1
    rew = float(dist < 1.0)
    return obs, rew, done, {}

  @property
  def walls(self):
    return self._walls
  



class StochasticDepthPointEnv(PointEnv):
    """Modified environment with custom depth logic:
       - Positive depth action sets z to 0.
       - Negative depth action sets z to a random value in [0, 1].
    """
    def __init__(self, walls=None, resize_factor=1, fixed_start_end=None, depth_mode="normal_with_teleport"):
      super().__init__(walls, resize_factor, fixed_start_end)
      self.depth_mode = depth_mode
      self._teleport_spike_count = 0



    def step(self, action):
        action = action.copy()
        

        if not self.action_space.contains(action):
            print('WARNING: clipping invalid action:', action)

        if self._action_noise > 0:
            action += np.random.normal(0, self._action_noise, action.shape)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        assert self.action_space.contains(action)

        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        _spike_break = False          # ← tells the outer loop to stop
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = self.state.copy()
                
                         
                

                if axis < 2:  # x, y
                    new_state[axis] += dt * action[axis]
                    if not self._is_blocked(new_state[:2]):
                        self.state = new_state
                # elif axis == 2:  # z (depth)
                #     # Custom behavior: positive → 0, negative → random[0, 1]
                #     if action[2] > 0:
                #         self.state[2] = 0.0
                #     elif action[2] < 0:
                #         self.state[2] = np.random.uniform(0, 1)

                elif axis == 2:  # z (depth)
                    if self.depth_mode == "simple":
                        # Custom behavior: positive → 0, negative → random[0, 1]
                        if action[2] > 0:
                            self.state[2] = 0.0
                        elif action[2] < 0 and action[2] > -0.5:
                            self.state[2] = np.random.uniform(0, 1)
                        elif action[2] < -0.5:
                            self.state[2] = 1.0
                    elif self.depth_mode == "two-part":
                        # Define the unpredictable spike at e/π ≈ 0.865
                        spike_center = 0.9
                        spike_width = 0.05  # very narrow spike window
                        a = action[2]
                        # Always zero when action is negative
                        if a < 0:
                            self.state[2] = 0.0
                        # Sharp spike to 1
                        elif np.abs(a - spike_center) < spike_width:
                            self.state[2] = 1.0
                        # Smooth function starting at 0, rising slowly, bounded below 0.5
                        else:
                            base = (  # damped sine
                              0.5 * (1 - np.exp(-5 * a)))
                            self.state[2] = np.clip(base, 0.0, 1)


                    elif self.depth_mode == "complex":
                        a = action[2]

                        if a < 0:
                            self.state[2] = 0.0  # Always zero when action is negative
                        else:
                            # Define bin edges and values
                            bin_width = 0.05
                            num_bins = int(1.0 / bin_width)  # 20 bins
                            bin_index = int(min(a, 0.999) // bin_width)  # bin index in [0, 19]

                            # Create values for bins: random-like fixed values between 0 and 0.85
                            bin_values = [
                                0.05, 0.12, 0.08, 0.25, 0.3,
                                0.1,  0.15, 0.22, 0.35, 0.4,
                                0.85,  # 🔥 bin 10 
                                0.45, 0.38, 0.2, 0.28,
                                0.33, 0.18, 0.1, 0.05, 0.0
                            ]
                            # Spike bin
                            spike_bin_index = 10
                            if bin_index == spike_bin_index:
                                self.state[2] = 0.0
                                self.state[0] = 8
                                self.state[1] = 10

                            else:
                                self.state[2] = bin_values[bin_index]
                    
                    elif self.depth_mode == "normal_with_teleport":
                        a = action[2]
                        
                        # -------- spike definition (same 0.05‑wide bin around 0.50)
                        spike_bin_width = 0.05
                        if 0.50 <= a < 0.50 + spike_bin_width:   # spike detected
                            

                            ## if the mode is normal_with_teleport, the agent will teleport to the origin
                            if abs(self.state[0]-10) < 0.1 and abs(self.state[1]-0) < 0.1 and abs(self.state[2]-1) < 0.1:
                              self.state[0]  = 8
                              self.state[1]  = 10
                              self.state[2]  = 1
                              print("=========Teleporting to goal ==========")
                            else:
                              #first spike → send to origin
                              self.state[0] = 10
                              self.state[1] = 0
                              self.state[2] = 1
                              print("=========Teleporting to the shortcut state ==========")
                               
                             
                            
                            _spike_break = True   # <-- set flag
                            break                 # <-- exit inner loop

                            
                        else:
                            # not a spike → simple‑mode behaviour
                            self._teleport_spike_count = 0  # reset counter
                            new_state[2] += dt * a
                            self.state[2] = np.clip(new_state[2], 0, 1)
            if _spike_break:                 # <-- exit outer loop, too
                break


        done = False
        obs = self._get_obs()
        dist = np.linalg.norm(self.goal - self.state)
        self._last_end_pos = self.state
        self._timestep += 1
        rew = float(dist < 1.0)
        return obs, rew, done, {}