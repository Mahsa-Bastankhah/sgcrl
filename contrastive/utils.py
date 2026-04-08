"""Utilities for the contrastive RL agent."""
import functools
from collections import deque
from typing import Dict
from typing import Optional, Sequence

from acme import types
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.utils.observers import base as observers_base
from acme.wrappers import base
from acme.wrappers import canonical_spec
from acme.wrappers import gym_wrapper
from acme.wrappers import step_limit
import dm_env
import env_utils
import jax
import numpy as np
import point_env
from torch.utils.tensorboard import SummaryWriter
import os

def obs_to_goal_1d(obs, start_index, end_index):
  assert len(obs.shape) == 1
  return obs_to_goal_2d(obs[None], start_index, end_index)[0]


def obs_to_goal_2d(obs, start_index, end_index):
  assert len(obs.shape) == 2
  if end_index == -1:
    return obs[:, start_index:]
  else:
    return obs[:, start_index:end_index]


class SuccessObserver(observers_base.EnvLoopObserver):
  """Measures success by whether any of the rewards in an episode are positive.
  """

  def __init__(self):
    self._rewards = []
    self._success = []

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._rewards:
      success = np.sum(self._rewards) >= 1
      self._success.append(success)
    self._rewards = []

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    assert timestep.reward in [0, 1]
    self._rewards.append(timestep.reward)

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    return {
        'success': float(np.sum(self._rewards) >= 1),
        'success_1000': np.mean(self._success[-1000:]),
    }


class DistanceObserver(observers_base.EnvLoopObserver):
  """Observer that measures the L2 distance to the goal."""

  def __init__(self, obs_dim, start_index, end_index,
               smooth = True):
    self._distances = []
    self._obs_dim = obs_dim
    self._obs_to_goal = functools.partial(
        obs_to_goal_1d, start_index=start_index, end_index=end_index)
    self._smooth = smooth
    self._history = {}

  def _get_distance(self, env,
                    timestep):
    if hasattr(env, '_dist'):
      assert env._dist  # pylint: disable=protected-access
      return env._dist[-1]  # pylint: disable=protected-access
    else:
      # Note that the timestep comes from the environment, which has already
      # had some goal coordinates removed.
      obs = timestep.observation[:self._obs_dim]
      goal = timestep.observation[self._obs_dim:]
      dist = np.linalg.norm(self._obs_to_goal(obs) - goal)
      return dist

  def observe_first(self, env, timestep
                    ):
    """Observes the initial state."""
    if self._smooth and self._distances:
      for key, value in self._get_current_metrics().items():
        self._history[key] = self._history.get(key, []) + [value]
    self._distances = [self._get_distance(env, timestep)]

  def observe(self, env, timestep,
              action):
    """Records one environment step."""
    self._distances.append(self._get_distance(env, timestep))

  def _get_current_metrics(self):
    metrics = {
        'init_dist': self._distances[0],
        'final_dist': self._distances[-1],
        'delta_dist': self._distances[0] - self._distances[-1],
        'min_dist': min(self._distances),
    }
    return metrics

  def get_metrics(self):
    """Returns metrics collected for the current episode."""
    metrics = self._get_current_metrics()
    if self._smooth:
      for key, vec in self._history.items():
        for size in [10, 100, 1000]:
          metrics['%s_%d' % (key, size)] = np.nanmean(vec[-size:])
    return metrics


class ObservationFilterWrapper(base.EnvironmentWrapper):
  """Wrapper that exposes just the desired goal coordinates."""

  def __init__(self, environment,
               idx):
    """Initializes a new ObservationFilterWrapper.

    Args:
      environment: Environment to wrap.
      idx: Sequence of indices of coordinates to keep.
    """
    super().__init__(environment)
    self._idx = idx
    observation_spec = environment.observation_spec()
    spec_min = self._convert_observation(observation_spec.minimum)
    spec_max = self._convert_observation(observation_spec.maximum)
    self._observation_spec = dm_env.specs.BoundedArray(
        shape=spec_min.shape,
        dtype=spec_min.dtype,
        minimum=spec_min,
        maximum=spec_max,
        name='state')

  def _convert_observation(self, observation):
    return observation[self._idx]

  def step(self, action):
    timestep = self._environment.step(action)
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def reset(self):
    timestep = self._environment.reset()
    return timestep._replace(
        observation=self._convert_observation(timestep.observation))

  def observation_spec(self):
    return self._observation_spec


def _point_maze_wall_key(env_name: str) -> Optional[str]:
  """Return WALLS dict key for point_* envs (e.g. point_Spiral11x11 -> Spiral11x11)."""
  if not env_name.startswith('point_'):
    return None
  return env_name.split('_', 1)[1]


class PointMazeTrajectorySnapshotWrapper(base.EnvironmentWrapper):
  """Logs recent 2D agent positions on point maze envs; saves PNGs periodically.

  Wraps the *filtered* observation (state || goal). Agent position is the first
  ``obs_dim`` components. Disabled for non-point environments at construction.
  """

  def __init__(
      self,
      environment,
      wall_map: np.ndarray,
      obs_dim: int,
      env_name: str,
      history_len: int = 100,
      snapshot_every: int = 5000,
      plot_dir: Optional[str] = None,
  ):
    super().__init__(environment)
    self._walls = np.asarray(wall_map, dtype=np.float32)
    self._obs_dim = obs_dim
    self._env_name = env_name
    self._history = deque(maxlen=history_len)
    self._snapshot_every = max(1, int(snapshot_every))
    self._step_count = 0
    base_dir = plot_dir or os.environ.get(
        'SGCRL_MAZE_TRAJ_DIR', os.path.join('logs', 'maze_traj_snapshots'))
    self._plot_dir = os.path.join(
        base_dir, f'{env_name}_pid{os.getpid()}')
    os.makedirs(self._plot_dir, exist_ok=True)

  def _push_position(self, observation):
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    pos = obs[:self._obs_dim].copy()
    self._history.append(pos)

  def reset(self):
    timestep = self._environment.reset()
    self._push_position(timestep.observation)
    return timestep

  def step(self, action):
    timestep = self._environment.step(action)
    self._push_position(timestep.observation)
    self._step_count += 1
    if (self._step_count % self._snapshot_every == 0
        and len(self._history) >= 2):
      self._save_plot(timestep.observation)
    return timestep

  def _save_plot(self, last_observation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    positions = np.stack(list(self._history), axis=0)
    rows, cols = positions[:, 0], positions[:, 1]
    t_idx = np.arange(len(positions))
    obs = np.asarray(last_observation, dtype=np.float64).reshape(-1)
    goal_row, goal_col = obs[self._obs_dim:self._obs_dim + 2]

    fig, ax = plt.subplots(figsize=(8, 8))
    h, w = self._walls.shape
    gray_background = np.ones((h, w), dtype=np.float32) * np.nan
    gray_background[self._walls > 0.5] = 1.0
    ax.imshow(
        gray_background, cmap='gray', vmin=0, vmax=1, origin='lower',
        interpolation='nearest', zorder=0, extent=(-0.5, w - 0.5, -0.5, h - 0.5))
    free_layer = np.where(self._walls < 0.5, 0.12, np.nan).astype(np.float32)
    ax.imshow(
        free_layer, cmap='Greens', vmin=0, vmax=1, origin='lower',
        interpolation='nearest', alpha=0.4, zorder=1,
        extent=(-0.5, w - 0.5, -0.5, h - 0.5))

    ax.plot(cols, rows, color='white', linewidth=1.1, alpha=0.45, zorder=2)
    sc = ax.scatter(
        cols, rows, c=t_idx, cmap='plasma', s=38, alpha=0.9,
        edgecolors='black', linewidths=0.35, zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04,
                 label='index in window (older→newer)')
    ax.scatter(
        goal_col, goal_row, marker='*', s=320, c='crimson',
        edgecolors='black', linewidths=0.9, zorder=5, label='goal (last obs)')

    ax.set_title(
        f'{self._env_name} — last {len(positions)} positions @ env step '
        f'{self._step_count}',
        fontsize=16, pad=10)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper left', fontsize=12, frameon=True)
    fig.tight_layout()
    path = os.path.join(
        self._plot_dir, f'step_{self._step_count:09d}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[maze-traj] saved {path}', flush=True)


class FakeEpisodeBoundaryWrapper(base.EnvironmentWrapper):
  """Wrapper that creates fake episode boundaries for continuous episodes.
  
  This allows us to have one long continuous episode but still chunk the data
  into fixed-length trajectories for the replay buffer by inserting LAST 
  timesteps periodically without actually resetting the environment.
  """

  def __init__(self, environment, steps_per_chunk, env_label = ''):
    """Initializes the wrapper.
    
    Args:
      environment: Environment to wrap.
      steps_per_chunk: Number of steps before creating a fake episode boundary.
      env_label: Optional name (e.g. env_name) for log messages.
    """
    super().__init__(environment)
    self._steps_per_chunk = steps_per_chunk
    self._env_label = env_label
    self._step_count = 0
    self._needs_reset = True
    self._last_observation = None
    self._chunk_index = 0
    self._n_fake_last_logged = 0
    self._n_fake_reset_logged = 0
    label = f' {env_label}' if env_label else ''
    print(
        f'[sgcrl continuous-episode pid={os.getpid()}{label}] '
        f'FakeEpisodeBoundaryWrapper: steps_per_chunk={steps_per_chunk} — '
        f'replay buffer gets a new trajectory every {steps_per_chunk} steps; '
        f'the underlying simulator is not reset until the first line below '
        f'("REAL reset").',
        flush=True)

  def reset(self):
    # Only do actual reset on first call or when explicitly needed
    if self._needs_reset:
      timestep = self._environment.reset()
      self._needs_reset = False
      self._last_observation = timestep.observation
      self._chunk_index = 0
      print(
          f'[sgcrl continuous-episode pid={os.getpid()}] REAL reset: '
          f'called underlying env.reset() (physical episode start).',
          flush=True)
    else:
      # Fake reset - just get current observation without resetting
      timestep = dm_env.TimeStep(
          step_type=dm_env.StepType.FIRST,
          reward=0.0,
          discount=1.0,
          observation=self._last_observation
      )
      self._chunk_index += 1
      self._n_fake_reset_logged += 1
      if self._n_fake_reset_logged <= 3:
        print(
            f'[sgcrl continuous-episode pid={os.getpid()}] FAKE reset '
            f'({self._n_fake_reset_logged}/3 logged, chunk '
            f'{self._chunk_index}): no underlying env.reset().',
            flush=True)
    self._step_count = 0
    return timestep

  def step(self, action):
    timestep = self._environment.step(action)
    self._last_observation = timestep.observation
    self._step_count += 1
    
    # Create fake episode boundary after steps_per_chunk steps
    if self._step_count >= self._steps_per_chunk:
      # Return LAST timestep to trigger episode boundary in replay buffer
      timestep = timestep._replace(step_type=dm_env.StepType.LAST)
      self._n_fake_last_logged += 1
      if self._n_fake_last_logged <= 3:
        print(
            f'[sgcrl continuous-episode pid={os.getpid()}] FAKE LAST '
            f'({self._n_fake_last_logged}/3 logged): after '
            f'{self._steps_per_chunk} env steps — replay sees episode end, '
            f'simulator continues.',
            flush=True)
    
    return timestep


def make_environment(env_name, start_index, end_index,
                     seed, fixed_start_end = None, extra_dim: int = 0):
  """Creates the environment.

  Args:
    env_name: name of the environment
    start_index: first index of the observation to use in the goal.
    end_index: final index of the observation to use in the goal. The goal
      is then obs[start_index:goal_index].
    seed: random seed.
    extra_dim: backwards-compatibility hook; legacy callers may request
      additional observation features. The current environments do not
      expose these extra features, so the parameter is ignored.
  Returns:
    env: the environment
    obs_dim: integer specifying the size of the observations, before
      the start_index/end_index is applied.
  """
  np.random.seed(seed)
  gym_env, obs_dim, max_episode_steps = env_utils.load(env_name, fixed_start_end)
  if extra_dim:
    print(f"[make_environment] Requested extra_dim={extra_dim}, but the current"
          " environment does not expose additional coordinates; ignoring.")
  goal_indices = obs_dim + obs_to_goal_1d(np.arange(obs_dim), start_index,
                                          end_index)
  indices = np.concatenate([
      np.arange(obs_dim),
      goal_indices
  ])
  env = gym_wrapper.GymWrapper(gym_env)
  # Use very large step limit for continuous episode (no actual resets during training)
  env = step_limit.StepLimitWrapper(env, step_limit=12_000_000)
  # Add fake episode boundaries to chunk continuous episode into trajectories
  env = FakeEpisodeBoundaryWrapper(
      env, steps_per_chunk=max_episode_steps, env_label=env_name)
  # Preserve the original episode length for config, not the large step limit
  env._step_limit = max_episode_steps
  env = ObservationFilterWrapper(env, indices)
  wall_key = _point_maze_wall_key(env_name)
  if wall_key is not None and wall_key in point_env.WALLS:
    if os.environ.get('SGCRL_MAZE_TRAJ', '1') != '0':
      every = int(os.environ.get('SGCRL_MAZE_TRAJ_EVERY', '5000'))
      hist = int(os.environ.get('SGCRL_MAZE_TRAJ_HISTORY', '100'))
      if every > 0:
        env = PointMazeTrajectorySnapshotWrapper(
            env,
            wall_map=point_env.WALLS[wall_key],
            obs_dim=obs_dim,
            env_name=env_name,
            history_len=max(2, hist),
            snapshot_every=every,
        )
  return env, obs_dim


class InitiallyRandomActor(actors.GenericActor):
  """Actor that takes actions uniformly at random until the actor is updated.
  """

  def select_action(self,
                    observation):
    if (self._params['mlp/~/linear_0']['b'] == 0).all():
      shape = self._params['Normal/~/linear']['b'].shape
      rng, self._state = jax.random.split(self._state)
      action = jax.random.uniform(key=rng, shape=shape,
                                  minval=-1.0, maxval=1.0)
    else:
      action, self._state = self._policy(self._params, observation,
                                         self._state)
    return utils.to_numpy(action)