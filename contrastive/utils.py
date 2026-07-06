"""Utilities for the contrastive RL agent."""
import functools
from typing import Dict, Optional, Sequence

from acme import types
from acme.agents.jax import actors
from acme.jax import networks as network_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils.observers import base as observers_base
from acme.wrappers import base
from acme.wrappers import canonical_spec
from acme.wrappers import gym_wrapper
from acme.wrappers import step_limit
import dm_env
import env_utils
import jax
import numpy as np
import os

FLOW_DENSE_REWARD_KEY = 'flow_dense_reward'


def extract_info_reward(env, key: str = FLOW_DENSE_REWARD_KEY) -> float:
  """Read a scalar diagnostic from the last env step (via Acme ``get_info``)."""
  get_info = getattr(env, 'get_info', None)
  if get_info is None:
    return float('nan')
  info = get_info()
  if not isinstance(info, dict) or key not in info:
    return float('nan')
  return float(info[key])


def flow_dense_eval_episode_metrics(
    episode_return: float, num_steps: int) -> Dict[str, float]:
  """Per-episode Flow ``desired_velocity`` return for eval logging."""
  if num_steps <= 0:
    return {
        'flow_dense_return': float('nan'),
        'flow_dense_reward_mean': float('nan'),
    }
  ret = float(episode_return)
  return {
      'flow_dense_return': ret,
      'flow_dense_reward_mean': ret / float(num_steps),
  }


_FLOW_DENSE_KEYS = frozenset({
    'flow_dense_return', 'flow_dense_reward_mean', 'ep_flow_dense_return_mean'})


def aggregate_eval_metrics(
    ep_metrics_list: Sequence[Dict],
    iteration: int) -> Dict[str, float]:
  """Mean eval metrics across episodes.

  Flow-dense keys (flow_dense_return, flow_dense_reward_mean,
  ep_flow_dense_return_mean) are only included when at least one episode
  actually produced a non-nan flow-dense return — i.e. the environment
  provides the flow benchmark reward signal.
  """
  agg: Dict[str, float] = {
      'iteration': float(iteration),
      'learner_steps': float(iteration),
  }
  if not ep_metrics_list:
    return agg
  all_keys = set()
  for m in ep_metrics_list:
    all_keys.update(m.keys())

  # Decide whether to include flow-dense keys at all.
  dense_returns = [
      m.get('flow_dense_return', float('nan')) for m in ep_metrics_list]
  include_flow_dense = any(not np.isnan(v) for v in dense_returns)

  for k_ in all_keys:
    if k_ in ('iteration', 'learner_steps'):
      continue
    if k_ in _FLOW_DENSE_KEYS and not include_flow_dense:
      continue
    agg[k_] = float(np.nanmean(
        [m.get(k_, float('nan')) for m in ep_metrics_list]))

  # Always emit flow-dense columns so eval CSV schema stays stable on resume.
  if include_flow_dense:
    agg['ep_flow_dense_return_mean'] = float(np.nanmean(dense_returns))
  else:
    agg['ep_flow_dense_return_mean'] = float('nan')
    agg['flow_dense_return'] = float('nan')
    agg['flow_dense_reward_mean'] = float('nan')

  return agg


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


class BuilderBenchSuccessObserver(observers_base.EnvLoopObserver):
  """Eval success from BuilderBench ``info['success']`` (dense env reward is not 0/1)."""

  def __init__(self):
    self._step_success = []
    self._success = []

  def observe_first(self, env, timestep):
    if self._step_success:
      self._success.append(bool(np.max(self._step_success) >= 0.5))
    self._step_success = []

  def observe(self, env, timestep, action):
    get_info = getattr(env, 'get_info', None)
    val = 0.0
    if get_info is not None:
      info = get_info() or {}
      val = float(info.get('success', info.get('easy_success', 0.0)))
    self._step_success.append(val)

  def get_metrics(self):
    hit = bool(np.max(self._step_success) >= 0.5) if self._step_success else False
    return {
        'success': float(hit),
        'success_1000': float(np.mean(self._success[-1000:]))
        if self._success else float('nan'),
    }


class RiverSwimGoalVisitSuccessObserver(observers_base.EnvLoopObserver):
  """Eval success for RiverSwim: agent reached the goal cell at least once.

  Uses the goal-conditioned observation ``[one_hot(state); one_hot(goal)]``
  (``obs_dim`` cells each). Env reward (+1 only when swimming right at the
  far bank) is **not** used, since PPO trains on φ·ψ and the sparse env
  bonus can diverge from "visited the goal state".
  """

  def __init__(self, obs_dim: int):
    self._obs_dim = int(obs_dim)
    self._visited_goal = []  # bool per env step this episode
    self._success = []  # bool per completed episode (for success_1000)

  def observe_first(self, env, timestep):
    if self._visited_goal:
      self._success.append(bool(np.any(self._visited_goal)))
    self._visited_goal = []

  def observe(self, env, timestep, action):
    obs = np.asarray(timestep.observation, dtype=np.float64).ravel()
    d = self._obs_dim
    if obs.size < 2 * d:
      self._visited_goal.append(False)
      return
    s = int(np.argmax(obs[:d]))
    g = int(np.argmax(obs[d:2 * d]))
    self._visited_goal.append(s == g)

  def get_metrics(self):
    hit = bool(np.any(self._visited_goal)) if self._visited_goal else False
    s1k = (
        float(np.mean(self._success[-1000:]))
        if self._success else float('nan'))
    return {'success': float(hit), 'success_1000': s1k}


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


class ReprCriticLogitObserver(observers_base.EnvLoopObserver):
  """Logs average φ(s,a)·ψ(g) along the episode.

  The contrastive critic's output for a single (s, a, g) triple is
  `φ(s,a)·ψ(g)` — the same quantity used as the logit in CRL's InfoNCE
  loss and (via `log σ(·)`) as the reward signal for the repr-reward
  actors (Launchpad `kappa_sac` / `q_sac`, or standalone PPO).  Tracking it at eval time tells
  us how the critic scores the trajectories the policy is actually
  producing, without imposing any training cost — useful especially for
  the "reward_shaping_mode disabled" case where the critic never feeds
  back into the actor and we have no other direct signal of φ·ψ scale.

  Wiring note: the critic params live on the learner, so this observer
  cannot be fully constructed in `agents.py` (no variable source yet).
  It is instantiated there as a bare shell and later "bound" by the
  evaluator / actor factory via `bind(variable_source, networks)` once
  the launchpad topology is stitched together.  Before binding it is a
  no-op.
  """

  def __init__(self, twin_q = False, obs_dim = None, hard_goal = None):
    self._twin_q = bool(twin_q)
    self._obs_dim = obs_dim
    self._hard_goal = None if hard_goal is None else np.asarray(hard_goal, dtype=np.float32)
    self._apply = None
    self._variable_client = None
    self._prev_obs = None
    self._logits = []          # current episode
    self._episode_means = []   # one entry per completed episode

  def _obs_with_hard_goal(self, obs):
    if self._hard_goal is None or self._obs_dim is None:
      return obs
    obs = np.asarray(obs).copy()
    obs[self._obs_dim:] = self._hard_goal
    return obs

  def bind(self, variable_source, networks):
    """Attach a variable client + network.  Called by the factory."""
    self._apply = jax.jit(networks.q_network.apply)
    self._variable_client = variable_utils.VariableClient(
        variable_source, 'critic', device='cpu')
    # Block once so we're ready for the very first episode.
    self._variable_client.update_and_wait()

  def observe_first(self, env, timestep):
    """Called with the initial timestep of a new episode."""
    del env
    if self._logits:
      self._episode_means.append(float(np.mean(self._logits)))
    self._logits = []
    self._prev_obs = np.asarray(timestep.observation)
    # Fire-and-forget refresh; params used in observe() are whatever the
    # client currently holds (cheap on an inter-process pipe).
    if self._variable_client is not None:
      self._variable_client.update(wait=False)

  def observe(self, env, timestep, action):
    """Called after each env step with the *post-step* timestep."""
    del env
    if self._apply is None or self._prev_obs is None:
      self._prev_obs = np.asarray(timestep.observation)
      return
    # q_network.apply returns (critic_val, sa_repr, g_repr) where
    # critic_val has shape [B, B] (no-twin) or [B, B, 2] (twin).  For a
    # single-sample query, B=1 and we want the diagonal entry, i.e.
    # critic_val[0, 0] averaged across the twin axis.
    obs_for_critic = self._obs_with_hard_goal(self._prev_obs)
    out = self._apply(self._variable_client.params,
                      obs_for_critic[None], np.asarray(action)[None])
    critic_val = np.asarray(out[0]).reshape(-1)
    self._logits.append(float(np.mean(critic_val)))
    self._prev_obs = np.asarray(timestep.observation)

  def get_metrics(self):
    """Per-episode metrics consumed by the environment loop logger."""
    if not self._logits:
      return {}
    cur = float(np.mean(self._logits))
    return {
        'repr_critic_logit_mean': cur,
        'repr_critic_logit_final': float(self._logits[-1]),
        'repr_critic_logit_max': float(np.max(self._logits)),
        'repr_critic_logit_mean_100':
            float(np.mean((self._episode_means + [cur])[-100:])),
    }


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


def make_environment(env_name, start_index, end_index,
                     seed, fixed_start_end=None, **env_kwargs):
  """Creates the environment.

  Args:
    env_name: name of the environment
    start_index: first index of the observation to use in the goal.
    end_index: final index of the observation to use in the goal. The goal
      is then obs[start_index:goal_index].
    seed: random seed.
    **env_kwargs: forwarded to ``env_utils.load`` (env-specific).
  Returns:
    env: the environment
    obs_dim: integer specifying the size of the observations, before
      the start_index/end_index is applied.
  """
  np.random.seed(seed)
  gym_env, obs_dim, max_episode_steps = env_utils.load(
      env_name, fixed_start_end, seed, **env_kwargs)
  goal_indices = obs_dim + obs_to_goal_1d(np.arange(obs_dim), start_index,
                                          end_index)
  indices = np.concatenate([
      np.arange(obs_dim),
      goal_indices
  ])
  env = gym_wrapper.GymWrapper(gym_env)
  env = step_limit.StepLimitWrapper(env, step_limit=max_episode_steps)
  env = ObservationFilterWrapper(env, indices)
  return env, obs_dim


def _policy_trunk_linear0_bias_all_zero(params) -> bool:
  """True before the first learner actor update (bias init is zeros).

  Policy trunk param names differ by architecture:
  ``mlp/~/linear_0`` (``hk.Sequential`` + unnamed ``hk.nets.MLP``) vs
  ``policy_mlp/linear_0`` (named ``ResidualMLP`` / ``_mlp_or_residual`` trunk).
  """
  preferred = ('mlp/~/linear_0', 'policy_mlp/linear_0')
  for key in preferred:
    block = params.get(key)
    if isinstance(block, dict) and 'b' in block:
      return bool((block['b'] == 0).all())
  for key in sorted(params):
    if key.endswith('/linear_0'):
      block = params[key]
      if isinstance(block, dict) and 'b' in block:
        return bool((block['b'] == 0).all())
  return False


class InitiallyRandomActor(actors.GenericActor):
  """Actor that takes actions uniformly at random until the actor is updated.
  """

  def select_action(self,
                    observation):
    if _policy_trunk_linear0_bias_all_zero(self._params):
      shape = self._params['Normal/~/linear']['b'].shape
      rng, self._state = jax.random.split(self._state)
      action = jax.random.uniform(key=rng, shape=shape,
                                  minval=-1.0, maxval=1.0)
    else:
      action, self._state = self._policy(self._params, observation,
                                         self._state)
    return utils.to_numpy(action)

