import jax
import haiku as hk

import jax.numpy as jnp
import numpy as np
import gym
from typing import NamedTuple
import random

class EnvState(NamedTuple):
    curr_state_onehot: jnp.ndarray    
    goal_state_onehot: jnp.ndarray    
    curr_state_idx: jnp.int32         
    goal_state_idx: jnp.int32         
    curr_timestep: jnp.int32          


class RiverSwim:
    def __init__(self, river_len, randomize_actions, horizon, seed, gamma=0.95, use_absorbing_states=False, action_mapping=None):
        # Original river states are shifted by 2 positions to make room for s^+ and s^-
        # State indices: 0=s^+, 1=s^-, 2=leftmost river, ..., river_len+1=rightmost river
        self.use_absorbing_states = use_absorbing_states
        self.gamma = gamma
        self.river_len = river_len
        self.fixed_action_mapping = action_mapping  # Optional fixed action mapping from gym env

        if use_absorbing_states:
            num_states = river_len + 2  # +2 for s^+ and s^-
            self.s_plus_idx = 0   # s^+ at index 0
            self.s_minus_idx = 1  # s^- at index 1
            self.river_start_idx = 2  # river states start at index 2
        else:
            num_states = river_len
            self.river_start_idx = 0  # river states start at index 0
            
        num_actions = 2
        self.num_states = num_states
        self.num_actions = num_actions
        self.horizon = horizon
        self.randomize_actions = randomize_actions
        self.reward_noise = 0.0
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(2 * num_states,),  
            dtype=np.int32
        )

        self.action_space = gym.spaces.Box(
            low=np.array([0]),
            high=np.array([num_actions]),
            dtype=np.float32)

        R = np.zeros((num_states, num_actions, num_states))
        T = np.zeros((num_states, num_actions, num_states))
        
        if use_absorbing_states:
            for a in range(num_actions):
                T[self.s_plus_idx, a, self.s_plus_idx] = 1.0
                T[self.s_minus_idx, a, self.s_minus_idx] = 1.0
        
        for s in range(river_len):
            # if use_absorbing_states, the river states are shifted by 2 positions to make room for s^+ and s^-
            river_state_idx = s + self.river_start_idx
            
            # Determine left and right action indices
            if self.fixed_action_mapping is not None:
                # Use the provided action mapping from gym environment
                # Gym env uses constants: LEFT=0, RIGHT=1 (not action indices!)
                # action_mapping[gym_state] = {action_idx: LEFT/RIGHT}
                gym_state_mapping = self.fixed_action_mapping[s]
                # Find which action index corresponds to LEFT (0) and RIGHT (1)
                left = [action_idx for action_idx, direction in gym_state_mapping.items() if direction == 0][0]
                right = [action_idx for action_idx, direction in gym_state_mapping.items() if direction == 1][0]
            elif self.randomize_actions:
                left = int(np.random.choice(2))
                right = 1 - left
            else:
                left, right = 0, 1
                
            if s == 0:  # Leftmost river state
                if use_absorbing_states:
                    reward_prob_left = 0.005  # small distractor reward
                    jump_prob = 1 - gamma
                    continue_prob = gamma
                    
                    T[river_state_idx, left, self.s_plus_idx] = jump_prob * reward_prob_left
                    T[river_state_idx, left, self.s_minus_idx] = jump_prob * (1 - reward_prob_left)
                    T[river_state_idx, left, river_state_idx] = continue_prob * 1.0
                    
                    reward_prob_right = 0
                    T[river_state_idx, right, self.s_plus_idx] = jump_prob * reward_prob_right
                    T[river_state_idx, right, self.s_minus_idx] = jump_prob * (1 - reward_prob_right)
                    T[river_state_idx, right, river_state_idx] = continue_prob * 0.4
                    T[river_state_idx, right, river_state_idx + 1] = continue_prob * 0.6
                else:
                    # Original dynamics
                    R[river_state_idx, left, :] = 0.0
                    T[river_state_idx, left, river_state_idx] = 1.
                    T[river_state_idx, right, river_state_idx] = 0.4
                    T[river_state_idx, right, river_state_idx + 1] = 0.6
                    
            elif s == river_len - 1:  # Rightmost river state
                if use_absorbing_states:
                    jump_prob = 1 - gamma
                    continue_prob = gamma
                    
                    reward_prob_left = 0.0
                    T[river_state_idx, left, self.s_plus_idx] = jump_prob * reward_prob_left
                    T[river_state_idx, left, self.s_minus_idx] = jump_prob * (1 - reward_prob_left)
                    T[river_state_idx, left, river_state_idx - 1] = continue_prob * 1.0
                    
                    reward_prob_right = 1.0
                    T[river_state_idx, right, self.s_plus_idx] = jump_prob * reward_prob_right
                    T[river_state_idx, right, self.s_minus_idx] = jump_prob * (1 - reward_prob_right)
                    T[river_state_idx, right, river_state_idx - 1] = continue_prob * 0.4
                    T[river_state_idx, right, river_state_idx] = continue_prob * 0.6
                else:
                    R[river_state_idx, right, river_state_idx] = 1.
                    T[river_state_idx, right, river_state_idx - 1] = 0.4
                    T[river_state_idx, right, river_state_idx] = 0.6
                    T[river_state_idx, left, river_state_idx - 1] = 1.
                    
            else:  # Middle river states
                if use_absorbing_states:
                    reward_prob = 0.0
                    jump_prob = 1 - gamma
                    continue_prob = gamma
                    
                    T[river_state_idx, left, self.s_plus_idx] = jump_prob * reward_prob
                    T[river_state_idx, left, self.s_minus_idx] = jump_prob * (1 - reward_prob)
                    T[river_state_idx, left, river_state_idx - 1] = continue_prob * 1.0
                    
                    T[river_state_idx, right, self.s_plus_idx] = jump_prob * reward_prob
                    T[river_state_idx, right, self.s_minus_idx] = jump_prob * (1 - reward_prob)
                    T[river_state_idx, right, river_state_idx + 1] = continue_prob * 0.35
                    T[river_state_idx, right, river_state_idx] = continue_prob * 0.6
                    T[river_state_idx, right, river_state_idx - 1] = continue_prob * 0.05
                else:
                    T[river_state_idx, left, river_state_idx - 1] = 1.
                    T[river_state_idx, right, river_state_idx + 1] = 0.35
                    T[river_state_idx, right, river_state_idx] = 0.6
                    T[river_state_idx, right, river_state_idx - 1] = 0.05
        
        R, T = jnp.array(R), jnp.array(T)
        self.rng = hk.PRNGSequence(seed)
        self.R = R
        self.T = T
        
        if use_absorbing_states:
            self.default_goal_idx = self.s_plus_idx  # Goal is to reach s^+
            self.anti_goal_idx = self.s_minus_idx
        else:
            self.default_goal_idx = num_states - 1  # Goal is the rightmost state
            self.anti_goal_idx = 0
        
        def _reset(key):
            init_state_idx = jnp.array(self.river_start_idx, dtype=jnp.int32)
            goal_state_idx = jnp.array(self.default_goal_idx, dtype=jnp.int32)
            anti_goal_state_idx = jnp.array(self.anti_goal_idx, dtype=jnp.int32)
            
            # Create one-hot vectors for initial state, goal state, and anti-goal state
            init_state_onehot = jax.nn.one_hot(init_state_idx, num_states, dtype=jnp.int32)
            goal_state_onehot = jax.nn.one_hot(goal_state_idx, num_states, dtype=jnp.int32)
            anti_goal_state_onehot = jax.nn.one_hot(anti_goal_state_idx, num_states, dtype=jnp.int32)

            return EnvState(
                curr_state_onehot=init_state_onehot,
                goal_state_onehot=goal_state_onehot,
                curr_state_idx=init_state_idx,
                goal_state_idx=goal_state_idx,
                curr_timestep=jnp.array(1, dtype=jnp.int32)
            )
        
        self._reset = jax.jit(_reset)
        self.reset()
        
        def _step(key, env_state, action):
            # Discrete actions 0 and 1 index rows of T (no off-by-one).
            action = jnp.asarray(action).astype(jnp.int32).reshape(-1)[0]
            action = jnp.clip(action, 0, num_actions - 1)
            next_state_idx = jax.random.choice(
                key, 
                jnp.arange(0, num_states), 
                p=T[env_state.curr_state_idx, action]
            )
            next_state_onehot = jax.nn.one_hot(next_state_idx, num_states, dtype=jnp.int32)
            
            reward = R[env_state.curr_state_idx, action, next_state_idx]
            
            new_env_state = EnvState(
                curr_state_onehot=next_state_onehot,
                goal_state_onehot=env_state.goal_state_onehot,
                curr_state_idx=next_state_idx,
                goal_state_idx=env_state.goal_state_idx,
                curr_timestep=env_state.curr_timestep + 1
            )
            
            return new_env_state, reward
        
        self._step = jax.jit(_step)

    def get_model(self):
        return np.array(self.R), np.array(self.T)
    
    def get_curr_state(self):
        return jnp.concatenate([
            self.env_state.curr_state_onehot,
            self.env_state.goal_state_onehot
        ])
    
    def get_curr_state_idx(self):
        return self.env_state.curr_state_idx.item()
    
    def get_goal_state_idx(self):
        return self.env_state.goal_state_idx.item()
    
    def get_trans_prob(self, state_idx, action, next_state_idx):
        a = int(
            np.clip(
                int(np.asarray(action).reshape(-1)[0]),
                0,
                self.num_actions - 1))
        return self.T[state_idx, a, next_state_idx]
    
    def is_absorbing_state(self, state_idx):
        """Check if a state is an absorbing state (s^+ or s^-)"""
        if not self.use_absorbing_states:
            return False
        return state_idx == self.s_plus_idx or state_idx == self.s_minus_idx
    
    def is_good_absorbing_state(self, state_idx):
        """Check if current state is s^+"""
        if not self.use_absorbing_states:
            return False
        return state_idx == self.s_plus_idx
    
    def is_bad_absorbing_state(self, state_idx):
        """Check if current state is s^-"""
        if not self.use_absorbing_states:
            return False
        return state_idx == self.s_minus_idx
    
    def reset(self, goal_idx=None):
        self.env_state = self._reset(next(self.rng))
        
        if goal_idx is not None:
            goal_idx = jnp.array(goal_idx, dtype=jnp.int32)
            goal_onehot = jax.nn.one_hot(goal_idx, self.num_states, dtype=jnp.int32)
            self.env_state = EnvState(
                curr_state_onehot=self.env_state.curr_state_onehot,
                goal_state_onehot=goal_onehot,
                curr_state_idx=self.env_state.curr_state_idx,
                goal_state_idx=goal_idx,
                curr_timestep=self.env_state.curr_timestep
            )
        
        return jnp.concatenate([
            self.env_state.curr_state_onehot,
            self.env_state.goal_state_onehot
        ])
    
    def step(self, action):
        """`action` is a length-1 array.  Values in ``[-1, 1]`` (tanh policy) map to
        ``{0, 1}`` via ``1 if x > 0 else 0``; values already in ``[0, 1]`` work
        as discrete 0/1 after the same rule (so ``[0]`` → 0, ``[1]`` → 1).
        """
        x = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        # Tanh policies in [-1, 1]: split at 0 (0 → action 0).  Values in [0, 1]
        # are treated as discrete action ids after rounding (for tests / scripts).
        if -1e-6 <= x <= 1.0 + 1e-6 and float(int(round(x))) == x:
            disc = int(np.clip(int(round(x)), 0, 1))
        else:
            disc = 1 if x > 0.0 else 0
        ja = jnp.asarray(disc, dtype=jnp.int32)
        curr_state_idx = self.env_state.curr_state_idx
        curr_state_onehot = self.env_state.curr_state_onehot
        self.env_state, reward = self._step(next(self.rng), self.env_state, ja)
        
        next_state_idx = self.env_state.curr_state_idx
        next_state_onehot = self.env_state.curr_state_onehot
        
        done = False
        if self.env_state.curr_timestep > self.horizon:
            done = True
        # Note: We don't terminate early when reaching absorbing states to maintain
        # consistent episode lengths for batching. The agent will stay in the 
        # absorbing state until the horizon is reached.
            
        concatenated_state = jnp.concatenate([
            self.env_state.curr_state_onehot,
            self.env_state.goal_state_onehot
        ])

        if self.use_absorbing_states:
            if next_state_idx == self.s_plus_idx:
                reward = 1.0
            else:
                reward = 0.0
        else:
            reward = float(jnp.asarray(reward))

        return concatenated_state, reward, done, {'reached_goal': next_state_idx == self.env_state.goal_state_idx}
    
    def set_goal(self, goal_idx):
        """Set a new goal state."""
        goal_idx = jnp.array(goal_idx, dtype=jnp.int32)
        goal_onehot = jax.nn.one_hot(goal_idx, self.num_states, dtype=jnp.int32)
        
        self.env_state = EnvState(
            curr_state_onehot=self.env_state.curr_state_onehot,
            goal_state_onehot=goal_onehot,
            curr_state_idx=self.env_state.curr_state_idx,
            goal_state_idx=goal_idx,
            curr_timestep=self.env_state.curr_timestep
        )
        
        return jnp.concatenate([
            self.env_state.curr_state_onehot,
            self.env_state.goal_state_onehot
        ])


if __name__ == '__main__':
    river_len = 6
    H = 20
    SEED = random.randint(0, 1000000)
    gamma = 0.95
    
    # print("=== Original RiverSwim ===")
    # env_original = RiverSwim(river_len, False, H, SEED, use_absorbing_states=False)
    # R_orig, T_orig = env_original.get_model()
    # print(f"States: {env_original.num_states}, Shape: {R_orig.shape}")
    # print("Reward matrix R:")
    # print(R_orig)
    # print("Transition matrix T (showing T[:,:,0] - transitions to state 0):")
    # print(T_orig[:,:,0])
    
    print("\n=== Modified RiverSwim with Absorbing States ===")
    env_modified = RiverSwim(river_len, False, H, SEED, gamma=gamma, use_absorbing_states=True)
    R_mod, T_mod = env_modified.get_model()
    print(f"States: {env_modified.num_states}, Shape: {R_mod.shape}")
    print(f"s^+ index: {env_modified.s_plus_idx}, s^- index: {env_modified.s_minus_idx}")
    print(f"River states: {env_modified.river_start_idx} to {env_modified.num_states-1}")
    print("Transition matrix T (showing transitions to s^+ and s^-):")
    print("T[:,:,0] (to s^+):")
    print(T_mod[:,:,0])
    print("T[:,:,1] (to s^-):")
    print(T_mod[:,:,1])
    
    # Test a few steps
    print("\n=== Testing Environment ===")
    state = env_modified.reset()
    print(f"Initial state idx: {env_modified.get_curr_state_idx()}")
    
    for step in range(20):
        action = np.array([1])  # Go right
        state, reward, done, info = env_modified.step(action)
        curr_idx = env_modified.get_curr_state_idx()
        print(f"Step {step+1}: State {curr_idx}, Reward {reward:.3f}, Done {done}")
        
        if env_modified.is_good_absorbing_state(curr_idx):
            print("  -> Reached s^+ (good absorbing state)!")
        elif env_modified.is_bad_absorbing_state(curr_idx):
            print("  -> Reached s^- (bad absorbing state)!")
        
        if done:
            break