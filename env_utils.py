"""Utility for loading the goal-conditioned environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import gym
import numpy as np

_METAWORLD_IMPORT_ERROR = None
try:
  import metaworld as _metaworld
  _MW_BIN    = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']
  _MW_BOX    = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['box-close-v2']
  _MW_PEG    = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['peg-insert-side-v2']
  _MW_REACH  = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['reach-v2']
  _MW_PUSH   = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-v2']
  _MW_DRAWER = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['drawer-open-v2']
  _MW_BUTTON = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['button-press-v2']
except Exception as _e:  # noqa: BLE001  ImportError, OR mujoco_py's env-var
  # checks, OR any other metaworld/mujoco_py load-time failure.  We
  # deliberately cast a wide net so that point-only workflows (e.g.
  # ppo_rollout_maze.py) can run on hosts that don't have MuJoCo
  # installed/configured.  The sawyer_* envs below still reference these
  # class stubs; we re-raise the cached import error in SawyerBin/Box/Peg
  # `__init__` so users instantiating them get a clear message instead of
  # a downstream `super().reset()` AttributeError.
  _metaworld = None
  _MW_BIN    = object
  _MW_BOX    = object
  _MW_PEG    = object
  _MW_REACH  = object
  _MW_PUSH   = object
  _MW_DRAWER = object
  _MW_BUTTON = object
  _METAWORLD_IMPORT_ERROR = _e


def _require_metaworld(env_name: str):
  if _METAWORLD_IMPORT_ERROR is not None:
    raise RuntimeError(
        f'Cannot build {env_name}: metaworld/mujoco_py failed to import at '
        f'env_utils load time.  Original error:\n    '
        f'{type(_METAWORLD_IMPORT_ERROR).__name__}: '
        f'{_METAWORLD_IMPORT_ERROR}\n'
        f'Typical fix — export mujoco paths before launching Python:\n'
        f'    export LD_LIBRARY_PATH='
        f'$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia\n'
        f'    export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210\n'
        f'    export MUJOCO_GL=osmesa'
    ) from _METAWORLD_IMPORT_ERROR
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'envs'))
import point_env

os.environ['SDL_VIDEODRIVER'] = 'dummy'

try:
  from riverswim import RiverSwim
except ImportError:  # pragma: no cover
  RiverSwim = None


class GymRiverSwimEnv(gym.Env):
  """gym wrapper around ``RiverSwim`` (``use_absorbing_states=False``) for Acme/CRL.

  Observations are ``[one_hot(state); one_hot(goal)]`` as float32.  Actions are
  one-dimensional in ``[-1, 1]`` (same as point envs); see ``RiverSwim.step``
  for the mapping to discrete ``{0, 1}``.
  """

  def __init__(self, river_len, horizon, seed, randomize_actions=False,
               fixed_start_end=None, action_mapping=None):
    super().__init__()
    if RiverSwim is None:
      raise ImportError('riverswim.RiverSwim is unavailable')
    self._river_len = int(river_len)
    self._rs = RiverSwim(
        self._river_len,
        bool(randomize_actions),
        int(horizon),
        int(seed),
        use_absorbing_states=False,
        action_mapping=action_mapping)
    self._fixed = fixed_start_end
    d = 2 * self._river_len
    self.observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(d,), dtype=np.float32)
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0], dtype=np.float32),
        high=np.array([1.0], dtype=np.float32),
        dtype=np.float32)
    self._max_episode_steps = int(horizon)

  def _goal_idx_from_fixed(self):
    if self._fixed is None:
      return None
    arr = np.asarray(self._fixed, dtype=np.float64).ravel()
    if arr.size == self._river_len:
      return int(np.argmax(arr))
    if arr.size >= 1:
      return int(arr[0])
    return None

  def reset(self):
    g = self._goal_idx_from_fixed()
    obs = self._rs.reset(goal_idx=g)
    return np.asarray(obs, dtype=np.float32)

  def step(self, action):
    obs, rew, done, info = self._rs.step(action)
    return (np.asarray(obs, dtype=np.float32),
            float(rew), bool(done), info)

  @property
  def walls(self):
    return None


def euler2quat(euler):
  """Convert Euler angles to quaternions."""
  euler = np.asarray(euler, dtype=np.float64)
  assert euler.shape[-1] == 3, 'Invalid shape euler {}'.format(euler)

  ai, aj, ak = euler[Ellipsis, 2] / 2, -euler[Ellipsis, 1] / 2, euler[Ellipsis, 0] / 2
  si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
  ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
  cc, cs = ci * ck, ci * sk
  sc, ss = si * ck, si * sk

  quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
  quat[Ellipsis, 0] = cj * cc + sj * ss
  quat[Ellipsis, 3] = cj * sc - sj * cs
  quat[Ellipsis, 2] = -(cj * ss + sj * cc)
  quat[Ellipsis, 1] = cj * cs - sj * sc
  return quat


def load(env_name, fixed_start_end=None, seed=None, **env_kwargs):
  """Loads the train and eval environments, as well as the obs_dim.

  Args:
    env_name: Registered environment id.
    fixed_start_end: Env-specific fixed goal / start–goal (see each env).
    seed: Optional RNG seed (used by ``riverswim``; others may ignore it).
    **env_kwargs: Extra kwargs forwarded to env constructors (e.g.
      ``nf_closed_gripper_init`` for SawyerPush).
  """
  # pylint: disable=invalid-name
  kwargs = {}
  if env_name == 'sawyer_bin':
    CLASS = SawyerBin
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
    if 'randomize_gripper_init' in env_kwargs:
      kwargs['randomize_gripper_init'] = env_kwargs['randomize_gripper_init']
    if 'randomize_init' in env_kwargs:
      kwargs['randomize_init'] = env_kwargs['randomize_init']
    if 'metaworld_hand_init' in env_kwargs:
      kwargs['metaworld_hand_init'] = env_kwargs['metaworld_hand_init']
    if 'safe_grasp_reset' in env_kwargs:
      kwargs['safe_grasp_reset'] = env_kwargs['safe_grasp_reset']
    if 'randomize_tcp_z' in env_kwargs:
      kwargs['randomize_tcp_z'] = env_kwargs['randomize_tcp_z']
    if 'bounded_step_mocap' in env_kwargs:
      kwargs['bounded_step_mocap'] = env_kwargs['bounded_step_mocap']
    if 'selective_antiwindup_mocap' in env_kwargs:
      kwargs['selective_antiwindup_mocap'] = env_kwargs[
          'selective_antiwindup_mocap']
    if 'terminate_tracking_error' in env_kwargs:
      kwargs['terminate_tracking_error'] = env_kwargs[
          'terminate_tracking_error']
    if int(env_kwargs.get('max_episode_steps', 0) or 0) > 0:
      max_episode_steps = int(env_kwargs['max_episode_steps'])
  elif env_name == 'sawyer_box':
    CLASS = SawyerBox
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_peg':
    CLASS = SawyerPeg
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
    if 'randomize_init' in env_kwargs:
      kwargs['randomize_init'] = env_kwargs['randomize_init']
    if int(env_kwargs.get('max_episode_steps', 0) or 0) > 0:
      max_episode_steps = int(env_kwargs['max_episode_steps'])
  elif env_name == 'sawyer_reach':
    CLASS = SawyerReach
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
    gym_env = CLASS(**kwargs)
    obs_dim = 3  # hand_xyz only; goal_xyz appended separately
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'sawyer_push':
    CLASS = SawyerPush
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
    if 'nf_closed_gripper_init' in env_kwargs:
      kwargs['nf_closed_gripper_init'] = env_kwargs['nf_closed_gripper_init']
    gym_env = CLASS(**kwargs)
    obs_dim = 7  # hand_xyz(3) + gripper(1) + obj_xyz(3); goal is also 7-D
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'sawyer_drawer_open':
    CLASS = SawyerDrawerOpen
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
    gym_env = CLASS(**kwargs)
    obs_dim = 7  # hand_xyz(3) + gripper(1) + handle_xyz(3); goal is also 7-D
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'sawyer_button_press':
    CLASS = SawyerButtonPress
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
    gym_env = CLASS(**kwargs)
    obs_dim = 7  # hand_xyz(3) + gripper(1) + button_xyz(3); goal is also 7-D
    return gym_env, obs_dim, max_episode_steps
  elif env_name.startswith('point_'):
    CLASS = point_env.PointEnv
    # point_SixteenRooms4D  →  walls='SixteenRooms', extra_dims=2
    if env_name == 'point_SixteenRooms4D':
      kwargs['walls'] = 'SixteenRooms'
      kwargs['extra_dims'] = 1
      kwargs['goal_tolerance'] = 2.0
    elif env_name == 'point_SixteenRoomsActual4D':
      kwargs['walls'] = 'SixteenRooms'
      kwargs['extra_dims'] = 2
      kwargs['goal_tolerance'] = 2.0
    else:
      kwargs['walls'] = env_name.split('_')[-1]
    kwargs['fixed_start_end'] = fixed_start_end
    if 'SixteenRooms' in env_name:
      max_episode_steps = 200
    elif ('11x11' in env_name or '9x9' in env_name or '7x7' in env_name
          or 'Impossible' in env_name or 'EightRooms' in env_name):
      max_episode_steps = 100
    else:
      max_episode_steps = 50
  elif env_name == 'riverswim':
    if RiverSwim is None:
      raise ImportError(
          'riverswim is required for env riverswim (import failed).')
    river_len = int(os.environ.get('RIVERSWIM_LEN', '6'))
    horizon = int(os.environ.get(
        'RIVERSWIM_HORIZON', str(max(100, river_len * 20))))
    rs_seed = int(seed) if seed is not None else 0
    gym_env = GymRiverSwimEnv(
        river_len=river_len,
        horizon=horizon,
        seed=rs_seed,
        randomize_actions=False,
        fixed_start_end=fixed_start_end)
    obs_dim = river_len
    max_episode_steps = horizon
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight':
    from flow_env import FlowFigureEightEnv
    gym_env = FlowFigureEightEnv(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEightEnv.STATE_OBS_DIM  # 28: speeds + positions
    max_episode_steps = gym_env._max_episode_steps       # = 1500
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight_7rl':
    from flow_env import FlowFigureEight7RL
    gym_env = FlowFigureEight7RL(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEight7RL.STATE_OBS_DIM  # 28
    max_episode_steps = gym_env._max_episode_steps       # = 1500
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight_14rl':
    from flow_env import FlowFigureEight14RL
    gym_env = FlowFigureEight14RL(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEight14RL.STATE_OBS_DIM  # 28
    max_episode_steps = gym_env._max_episode_steps       # = 1500
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight_4v2rl':
    from flow_env import FlowFigureEight4V2RL
    gym_env = FlowFigureEight4V2RL(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEight4V2RL.STATE_OBS_DIM  # 8
    max_episode_steps = gym_env._max_episode_steps
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight_8v4rl':
    from flow_env import FlowFigureEight8V4RL
    gym_env = FlowFigureEight8V4RL(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEight8V4RL.STATE_OBS_DIM  # 16
    max_episode_steps = gym_env._max_episode_steps
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight_1v1rl':
    from flow_env import FlowFigureEight1V1RL
    gym_env = FlowFigureEight1V1RL(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEight1V1RL.STATE_OBS_DIM  # 2
    max_episode_steps = gym_env._max_episode_steps       # = 1500
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight_2v1rl':
    from flow_env import FlowFigureEight2V1RL
    gym_env = FlowFigureEight2V1RL(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEight2V1RL.STATE_OBS_DIM  # 4
    max_episode_steps = gym_env._max_episode_steps       # = 1500
    return gym_env, obs_dim, max_episode_steps
  elif env_name == 'flow_figureeight_2v2rl':
    from flow_env import FlowFigureEight2V2RL
    gym_env = FlowFigureEight2V2RL(fixed_start_end=fixed_start_end)
    obs_dim = FlowFigureEight2V2RL.STATE_OBS_DIM  # 4
    max_episode_steps = gym_env._max_episode_steps       # = 1500
    return gym_env, obs_dim, max_episode_steps
  elif env_name.startswith('builderbench_creative_'):
    from envs.builderbench_env import make_builderbench_creative_env
    from envs.builderbench_utils import sgcrl_env_name_to_bb_env_id
    fixed_target = None
    if fixed_start_end is not None:
      fixed_target = np.asarray(fixed_start_end, dtype=np.float32)
    bb_env_id = sgcrl_env_name_to_bb_env_id(env_name)
    gym_env = make_builderbench_creative_env(
        env_id=bb_env_id,
        seed=seed,
        use_pd=bool(env_kwargs.get('builderbench_use_pd', False)),
        pd_duration=int(env_kwargs.get('builderbench_pd_duration', 5)),
        pd_filter_policy_obs=bool(
            env_kwargs.get('builderbench_pd_filter_policy_obs', True)),
        fixed_target_goal=fixed_target,
        permute_start_boxes=bool(
            env_kwargs.get('builderbench_permute_start_boxes', True)),
    )
    obs_dim = gym_env.state_obs_dim
    max_episode_steps = gym_env._max_episode_steps
    return gym_env, obs_dim, max_episode_steps
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)

  # Disable type checking in line below because different environments have
  # different kwargs, which pytype doesn't reason about.
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  obs_dim = gym_env.observation_space.shape[0] // 2
  return gym_env, obs_dim, max_episode_steps


def unwrap_gym_env(env):
  """Unwrap acme/dm_env wrappers down to the underlying ``gym.Env``."""
  while hasattr(env, '_environment'):
    env = env._environment
  return env


def resolve_uniform_goal_bounds(
    spec,
    wrapped_env,
    obs_dim: int,
    start_index: int,
    end_index: int,
):
  """Goal-slice bounds for ``uniform_sampling`` CRL negatives.

  Uses ``spec.observations`` when finite; otherwise ``uniform_goal_obs_bounds``
  on Sawyer wrappers (see ``env_utils.Sawyer*``).
  """
  obs_dim = int(obs_dim)
  si = int(start_index)
  ei = int(end_index) if int(end_index) != -1 else obs_dim
  obs_min = np.asarray(spec.observations.minimum, dtype=np.float32)
  obs_max = np.asarray(spec.observations.maximum, dtype=np.float32)
  goal_low = obs_min[obs_dim + si:obs_dim + ei]
  goal_high = obs_max[obs_dim + si:obs_dim + ei]
  if np.all(np.isfinite(goal_low)) and np.all(np.isfinite(goal_high)):
    return goal_low, goal_high
  base = unwrap_gym_env(wrapped_env)
  if not hasattr(base, 'uniform_goal_obs_bounds'):
    raise ValueError(
        f'uniform_sampling requires finite goal bounds; got '
        f'goal_low={goal_low}, goal_high={goal_high}. '
        f'Underlying env {type(base).__name__!r} has no '
        f'uniform_goal_obs_bounds().')
  glo, ghi = base.uniform_goal_obs_bounds()
  return np.asarray(glo[si:ei], dtype=np.float32), np.asarray(
      ghi[si:ei], dtype=np.float32)


class SawyerBin(_MW_BIN):
  """Wrapper for the SawyerBin environment."""

  # Goal slice of ``_get_obs()`` (7-d): [grasp_hand, grip≈cube width, object=_goal].
  # ψ: fingers around the cube on the floor at the success target.
  # Success still only checks ‖object − _goal‖.
  PSI_TCP_ABOVE_OBJ = 0.03  # finger/TCP geometry for a grasp around the cube
  PSI_GRIPPER = 0.4         # ≈ cube width 0.04 m / 0.1 (tight hold, not 0)
  UNIFORM_GOAL_OBS_LOW = np.array(
      [-0.22, 0.64, 0.01, 0.2, -0.22, 0.64, 0.01], dtype=np.float32)
  UNIFORM_GOAL_OBS_HIGH = np.array(
      [0.17, 0.76, 0.16, 0.5, 0.17, 0.76, 0.13], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    """Bounds on the goal block appended in ``_get_obs`` (for CRL uniform negs)."""
    return self.UNIFORM_GOAL_OBS_LOW.copy(), self.UNIFORM_GOAL_OBS_HIGH.copy()

  def _set_gripper_obs_width(self, target: float) -> float:
    """Set finger joints to the closest requested normalized aperture."""
    target = float(np.clip(target, 0.0, 1.0))
    r_id = self.model.joint_name2id('r_close')
    l_id = self.model.joint_name2id('l_close')
    r_adr = int(self.model.jnt_qposadr[r_id])
    l_adr = int(self.model.jnt_qposadr[l_id])
    cached = getattr(self, '_gripper_width_alpha', None)
    if cached is not None:
      self.data.qpos[r_adr] = 0.04 * cached
      self.data.qpos[l_adr] = -0.03 * cached
      self.sim.forward()
      right = self._get_site_pos('rightEndEffector')
      left = self._get_site_pos('leftEndEffector')
      return float(np.clip(np.linalg.norm(right - left) / 0.1, 0., 1.))

    best = None
    # Calibrate against the actual finger-site distance instead of assuming
    # which endpoint is open in this MetaWorld XML.
    for alpha in np.linspace(0.0, 1.0, 101):
      self.data.qpos[r_adr] = 0.04 * alpha
      self.data.qpos[l_adr] = -0.03 * alpha
      self.sim.forward()
      right = self._get_site_pos('rightEndEffector')
      left = self._get_site_pos('leftEndEffector')
      width = float(np.clip(np.linalg.norm(right - left) / 0.1, 0., 1.))
      err = abs(width - target)
      if best is None or err < best[0]:
        best = (err, float(alpha), width)
    _, alpha, width = best
    self._gripper_width_alpha = alpha
    self.data.qpos[r_adr] = 0.04 * alpha
    self.data.qpos[l_adr] = -0.03 * alpha
    self.sim.forward()
    return width

  def __init__(self, fixed_start_end=None, randomize_gripper_init=False,
               randomize_init=True, metaworld_hand_init=False,
               safe_grasp_reset=False, randomize_tcp_z=False,
               bounded_step_mocap=False, selective_antiwindup_mocap=False,
               terminate_tracking_error=False):
    _require_metaworld('sawyer_bin')
    self._goal = np.zeros(3)
    self._gripper_width_alpha = None
    self._metaworld_hand_init = bool(metaworld_hand_init)
    self._safe_grasp_reset = bool(safe_grasp_reset)
    self._bounded_step_mocap = bool(bounded_step_mocap)
    self._selective_antiwindup_mocap = bool(selective_antiwindup_mocap)
    self._terminate_tracking_error = bool(terminate_tracking_error)
    if self._bounded_step_mocap and self._selective_antiwindup_mocap:
      raise ValueError(
          'bounded_step_mocap and selective_antiwindup_mocap are separate '
          'controller modes and cannot be combined')
    # Freeze-init runs must also pin the gripper TCP (no offset noise).
    self._randomize_init = bool(randomize_init)
    self._randomize_gripper_init = (
        bool(randomize_gripper_init) and self._randomize_init)
    self._randomize_tcp_z = (
        bool(randomize_tcp_z) and self._randomize_init
        and not self._randomize_gripper_init)
    super(SawyerBin, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end=fixed_start_end
    # MetaWorld: random_init controls object XY spawn on reset_model.
    if not self._randomize_init:
      self.random_init = False
    self.reset()

  def set_xyz_action(self, action):
    """Set the TCP target without accumulating mocap tracking error."""
    if self._selective_antiwindup_mocap:
      action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
      pos_delta = action * float(self.action_scale)
      tcp = np.asarray(self.get_endeff_pos(), dtype=np.float64)
      mocap = np.asarray(self.data.mocap_pos[0], dtype=np.float64)
      error = mocap - tcp
      error_norm = float(np.linalg.norm(error))

      # Below the threshold this is exactly MetaWorld's legacy integrator:
      # mocap_target <- mocap_target + action_scale * clipped_action.
      if error_norm < 0.15:
        return super(SawyerBin, self).set_xyz_action(action)

      # Once tracking error is large, remove only the proposed increment's
      # outward radial component. Tangential and error-reducing motion remains.
      if error_norm > 0.0:
        outward_dir = error / error_norm
        outward = float(np.dot(pos_delta, outward_dir))
        if outward > 0.0:
          pos_delta = pos_delta - outward * outward_dir
      target = np.clip(mocap + pos_delta, self.mocap_low, self.mocap_high)

      # Hard backstop on the integrator state. This is intentionally not
      # global action clipping: only mocap-target/TCP separation is projected.
      target_error = target - tcp
      target_error_norm = float(np.linalg.norm(target_error))
      if target_error_norm > 0.20:
        target = tcp + (0.20 / target_error_norm) * target_error
        target = np.clip(target, self.mocap_low, self.mocap_high)
      self.data.set_mocap_pos('mocap', target)
      self.data.set_mocap_quat(
          'mocap', np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64))
      return None

    if not self._bounded_step_mocap:
      return super(SawyerBin, self).set_xyz_action(action)

    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    pos_delta = action * float(self.action_scale)
    tcp = np.asarray(self.get_endeff_pos(), dtype=np.float64)
    target = np.clip(tcp + pos_delta, self.mocap_low, self.mocap_high)
    self.data.set_mocap_pos('mocap', target)
    self.data.set_mocap_quat(
        'mocap', np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64))

  def _set_reset_goal(self):
    body_id = self.model.body_name2id('bin_goal')
    pos1 = self.sim.data.body_xpos[body_id].copy()
    pos1 += np.random.uniform(-0.05, 0.05, 3)
    pos2 = self._get_pos_objects().copy()

    if self._fixed_start_end is not None:
        self._goal = self._fixed_start_end
    else:
        t = np.random.random()
        self._goal = t * pos1 + (1 - t) * pos2
        self._goal[2] = np.random.uniform(0.03, 0.12)
    self._target_pos = self._goal

  def _grasp_target(self, obj_pos):
    tcp_z = float(self.PSI_TCP_ABOVE_OBJ)
    if self._randomize_gripper_init:
      return obj_pos + np.array([
          np.random.uniform(-0.05, 0.05),
          np.random.uniform(-0.05, 0.05),
          np.random.uniform(tcp_z, tcp_z + 0.03),
      ], dtype=np.float64)
    if self._randomize_tcp_z:
      return obj_pos + np.array([
          0., 0., np.random.uniform(tcp_z, tcp_z + 0.03),
      ], dtype=np.float64)
    return obj_pos + np.array([0., 0., tcp_z], dtype=np.float64)

  def _legacy_grasp_reset(self):
    """Original unbounded mocap servo, retained for historical runs."""
    obj_pos = self._get_pos_objects().copy()
    grip_target = self._grasp_target(obj_pos)
    mocap_pos = grip_target.copy()
    mocap_quat = np.array([1., 0., 1., 0.], dtype=np.float64)
    for _ in range(100):
      mocap_pos += grip_target - self.get_endeff_pos()
      self.data.set_mocap_pos('mocap', mocap_pos)
      self.data.set_mocap_quat('mocap', mocap_quat)
      self.do_simulation([-1, 1], self.frame_skip)
    self.sim.forward()
    self._set_obj_xyz(obj_pos)
    self._set_gripper_obs_width(self.PSI_GRIPPER)
    self.sim.forward()
    return self._get_obs()

  def _bounded_grasp_reset(self, obj_pos, grip_target):
    """Move to ``grip_target`` without integrating tracking error."""
    mocap_quat = np.array([1., 0., 1., 0.], dtype=np.float64)
    target = np.asarray(grip_target, dtype=np.float64)

    for _ in range(150):
      tcp = np.asarray(self.get_endeff_pos(), dtype=np.float64)
      error = target - tcp
      if np.linalg.norm(error) <= 0.005:
        break
      mocap = np.asarray(self.data.mocap_pos[0], dtype=np.float64)
      correction = np.clip(0.5 * error, -0.01, 0.01)
      command = np.clip(mocap + correction, self.mocap_low, self.mocap_high)
      self.data.set_mocap_pos('mocap', command)
      self.data.set_mocap_quat('mocap', mocap_quat)
      self.do_simulation([-1, 1], self.frame_skip)
    else:
      return False

    # Let the arm settle while open, then restore the sampled object and set
    # the desired aperture as the final operation (simulation would otherwise
    # immediately drive the fingers open again).
    self._set_obj_xyz(obj_pos)
    for _ in range(5):
      self.do_simulation([-1, 1], self.frame_skip)
    self._set_obj_xyz(obj_pos)
    width = self._set_gripper_obs_width(self.PSI_GRIPPER)
    self.sim.forward()

    hand = np.asarray(self.get_endeff_pos(), dtype=np.float64)
    obj = np.asarray(self._get_pos_objects(), dtype=np.float64)
    # Centered hover: hand must stay over the cube. MPO-style jitter: hand
    # must reach the sampled TCP target and the object must stay put
    # (no unbounded mocap shove).
    if self._randomize_gripper_init or self._randomize_tcp_z:
      pose_ok = (
          np.linalg.norm(hand - target) <= 0.025
          and np.linalg.norm(obj - np.asarray(obj_pos, dtype=np.float64)) <= 0.015)
    else:
      delta = hand - obj
      pose_ok = (
          np.linalg.norm(delta[:2]) <= 0.02
          and 0.0 <= delta[2] <= 0.06)
    return bool(
        np.all(np.isfinite(hand))
        and np.all(np.isfinite(obj))
        and pose_ok
        and abs(float(width) - float(self.PSI_GRIPPER)) <= 0.05)

  def reset(self):
    # Preserve the exact historical reset unless the opt-in safe path is used.
    if not self._safe_grasp_reset:
      super(SawyerBin, self).reset()
      self._set_reset_goal()
      if self._metaworld_hand_init:
        return self._get_obs()
      return self._legacy_grasp_reset()

    # Rejection sampling defines the intended reset distribution conditioned
    # on a reachable, physically realized floor grasp.
    for _ in range(25):
      super(SawyerBin, self).reset()
      self._set_reset_goal()
      if self._metaworld_hand_init:
        return self._get_obs()
      obj_pos = self._get_pos_objects().copy()
      grip_target = self._grasp_target(obj_pos)
      if self._bounded_grasp_reset(obj_pos, grip_target):
        return self._get_obs()
    raise RuntimeError(
        'SawyerBin safe_grasp_reset could not produce a valid grasp in '
        '25 sampled resets')

  def step(self, action):
    super(SawyerBin, self).step(action)
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)  # Taken from metaworld
    tracking_error = float(np.linalg.norm(
        np.asarray(self.data.mocap_pos[0], dtype=np.float64)
        - np.asarray(self.get_endeff_pos(), dtype=np.float64)))
    tracking_error_termination = bool(
        self._terminate_tracking_error and tracking_error > 0.20)
    done = tracking_error_termination
    info = {}
    if self._terminate_tracking_error:
      info = {
          'mocap_tracking_error': tracking_error,
          'tracking_error_termination': tracking_error_termination,
      }
        
    return obs, r, done, info

  def _psi_held_object(self) -> np.ndarray:
    """Object pose in the ψ goal: on the floor at the success target ``_goal``."""
    return np.asarray(self._goal, dtype=np.float32).copy()

  def _ideal_grasp_hand(self) -> np.ndarray:
    """TCP pose for ψ goal: closed-grasp TCP just above the floor cube."""
    return (self._psi_held_object()
            + np.array([0.0, 0.0, float(self.PSI_TCP_ABOVE_OBJ)],
                       dtype=np.float32))

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          self._get_pos_objects()))
    # ψ goal: fingers around the cube sitting on the floor at ``_goal``.
    # Success (step) still only checks ‖object − _goal‖.
    #   object  = _goal (floor / bin target)
    #   hand    = object + (0,0,PSI_TCP_ABOVE_OBJ)
    #   gripper = PSI_GRIPPER (≈0.4 = cube-width hold)
    held_obj = self._psi_held_object()
    goal = np.concatenate(
        [self._ideal_grasp_hand(), [float(self.PSI_GRIPPER)], held_obj])

    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)


class SawyerBox(_MW_BOX):
  """Wrapper for the SawyerBox environment."""

  UNIFORM_GOAL_OBS_LOW = np.array(
      [-0.10, 0.50, 0.11, 0.4, -0.10, 0.50, 0.08,
       0.707, 0.0, 0.0, 0.707], dtype=np.float32)
  UNIFORM_GOAL_OBS_HIGH = np.array(
      [0.10, 0.80, 0.17, 0.4, 0.10, 0.80, 0.14,
       0.707, 0.0, 0.0, 0.707], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    return self.UNIFORM_GOAL_OBS_LOW.copy(), self.UNIFORM_GOAL_OBS_HIGH.copy()

  def __init__(self, fixed_start_end=None):
    _require_metaworld('sawyer_box')
    self._goal_pos = np.zeros(3)
    self._goal_quat = np.zeros(4)
    super(SawyerBox, self).__init__()
    self._fixed_start_end=fixed_start_end
    self._set_task_called = True
    self._partially_observable = False
    self._freeze_rand_vec = False
    self.reset()

  def reset(self):
    super(SawyerBox, self).reset()
    pos1 = self._target_pos.copy()
    pos2 = self._get_pos_objects().copy()
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal_pos = pos1 
    else:
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        t = np.random.random()
        self._goal_pos = t * pos1 + (1 - t) * pos2
        
    self._goal_quat = np.array([0.707, 0, 0, 0.707]) # ideal orientation of lid
    self._target_pos = self._goal_pos    
    return self._get_obs()

  def step(self, action):
    super(SawyerBox, self).step(action)
    obj_pos = self._get_pos_objects()
    obj_quat = self._get_quat_objects()
    
    dist_pos = np.linalg.norm(self._goal_pos - obj_pos)
    dist_quat = np.linalg.norm(self._goal_quat - obj_quat)
    
    obs = self._get_obs()
    r = float(dist_pos < 0.08 and dist_quat < 0.08)  # Taken from metaworld
    done = False
    info = {}
    
    return obs, r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    
    obj_pos = self._get_pos_objects()
    obj_quat = self._get_quat_objects()
    
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          obj_pos, obj_quat))
    # the ideal goal state has the lid on the box and the gripper slightly 
    # higher than the lid center
    goal = np.concatenate([self._goal_pos + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal_pos, self._goal_quat])
    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 11, -np.inf),
        high=np.full(2 * 11, np.inf),
        dtype=np.float32)

class SawyerReach(_MW_REACH):
  """Wrapper for reach-v2: move the end-effector to a 3-D goal position.

  Observation layout (6-dim):
    obs = [ tcp_xyz (3)    <- state / φ input  (gripper TCP centre)
            goal_xyz (3) ] <- goal  / ψ input  (target reach position)
    obs_dim = 3, start_index = 0, end_index = -1 (defaults)

  State uses ``tcp_center`` (midpoint of the two finger sites), matching
  MetaWorld's native reach success metric — not ``get_endeff_pos()`` (mocap
  hand body), which is ~4-5 cm offset in z.

  CRL negatives: obs_to_goal(future_state) = future tcp_xyz (3-D),
  which matches the 3-D goal slice perfectly.

  Fixed goal (``fixed_goal_dict``): centre of MW goal_space,
  ``[0.0, 0.85, 0.2]``  (x=0, y mid of [0.8,0.9], z=0.2).
  """

  # Goal space bounds from SawyerReachEnvV2: y ∈ [0.8, 0.9], z ∈ [0.05, 0.3].
  UNIFORM_GOAL_OBS_LOW  = np.array([-0.1, 0.8, 0.05], dtype=np.float32)
  UNIFORM_GOAL_OBS_HIGH = np.array([ 0.1, 0.9, 0.30], dtype=np.float32)

  # For reach the goal IS the effector position, which is also the state;
  # effector can start anywhere in the workspace before the goal is reached.
  NF_GOAL_NORM_LOW  = np.array([-0.1, 0.8, 0.05], dtype=np.float32)
  NF_GOAL_NORM_HIGH = np.array([ 0.1, 0.9, 0.30], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    return self.UNIFORM_GOAL_OBS_LOW.copy(), self.UNIFORM_GOAL_OBS_HIGH.copy()

  def nf_goal_norm_bounds(self):
    return self.NF_GOAL_NORM_LOW.copy(), self.NF_GOAL_NORM_HIGH.copy()

  def __init__(self, fixed_start_end=None):
    _require_metaworld('sawyer_reach')
    self._goal = np.zeros(3)
    super(SawyerReach, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    # Disable random goal/object spawn when using a fixed target so _target_pos
    # and obs[3:6] always match fixed_goal_dict (same pattern as button_press).
    if fixed_start_end is not None:
      self.random_init = False
    self.reset()

  def reset(self):
    super(SawyerReach, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.asarray(self._fixed_start_end, dtype=np.float32).ravel()
    else:
      self._goal = self._target_pos.copy().astype(np.float32)
    self._target_pos = self._goal
    return self._get_obs()

  def step(self, action):
    super(SawyerReach, self).step(action)
    dist = np.linalg.norm(self.tcp_center - self._goal)
    obs = self._get_obs()
    r = float(dist <= 0.05)
    return obs, r, False, {}

  def _get_obs(self):
    tcp_pos = self.tcp_center.astype(np.float32)
    goal = np.asarray(self._goal, dtype=np.float32)
    return np.concatenate([tcp_pos, goal])

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(6, -np.inf, dtype=np.float32),
        high=np.full(6,  np.inf, dtype=np.float32),
        dtype=np.float32)


class SawyerPush(_MW_PUSH):
  """Wrapper for push-v2: slide a puck to a 3-D goal position.

  Observation layout (14-dim):
    obs = [ hand_xyz (3)  gripper (1)  obj_xyz (3)        <- state (7) / φ
            ideal_hand_xyz (3)  gripper (1)  target_xyz (3) ] <- goal (7) / ψ

    obs_dim = 7, start_index = 0, end_index = -1 (defaults).

  The 3-D fixed goal ``_goal`` is the puck target (MetaWorld ``_target_pos``,
  table surface z ≈ 0.02).

  The 7-D ψ slice uses a simplified/easier hand target: gripper at the puck
  goal, slightly behind (−8 cm in y) and a bit above (+3 cm in z), gripper
  closed, object at ``_goal``:

      ideal_hand = _goal + [0, -0.08, 0.03]
      goal       = [ideal_hand, gripper=0.0, _goal]
  """

  # Goal bounds match goal_space in SawyerPushEnvV2.
  UNIFORM_GOAL_OBS_LOW  = np.array(
      [-0.1, 0.8, 0.01, 0.0, -0.1, 0.8, 0.01], dtype=np.float32)
  UNIFORM_GOAL_OBS_HIGH = np.array(
      [ 0.1, 0.9, 0.02, 0.4,  0.1, 0.9, 0.02], dtype=np.float32)

  # NF normalisation bounds: cover the FULL workspace so that hindsight goals
  # (where obj starts at y≈0.60–0.70) are within [-1, 1] after normalisation.
  # Layout: [hand_x, hand_y, hand_z, gripper, obj_x, obj_y, obj_z]
  #   hand = obj_goal + [0, -0.08, 0.03]  →  hand_y range offset by -0.08
  NF_GOAL_NORM_LOW  = np.array(
      [-0.15, 0.47, 0.00, 0.0, -0.15, 0.55, 0.01], dtype=np.float32)
  NF_GOAL_NORM_HIGH = np.array(
      [ 0.15, 0.87, 0.30, 0.4,  0.15, 0.95, 0.03], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    return self.UNIFORM_GOAL_OBS_LOW.copy(), self.UNIFORM_GOAL_OBS_HIGH.copy()

  def nf_goal_norm_bounds(self):
    return self.NF_GOAL_NORM_LOW.copy(), self.NF_GOAL_NORM_HIGH.copy()

  def __init__(self, fixed_start_end=None, nf_closed_gripper_init=False):
    _require_metaworld('sawyer_push')
    self._goal = np.zeros(3)
    self._nf_closed_gripper_init = bool(nf_closed_gripper_init)
    super(SawyerPush, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    if fixed_start_end is not None:
      self.random_init = False
    self.reset()

  def _close_gripper_at_reset(self) -> None:
    """Hold hand position and close fingers (NF push init only)."""
    mocap_quat = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    for _ in range(40):
      mocap_pos = self.tcp_center.copy()
      self.data.set_mocap_pos('mocap', mocap_pos)
      self.data.set_mocap_quat('mocap', mocap_quat)
      self.do_simulation([1.0, -1.0], self.frame_skip)
    self.sim.forward()

  def reset(self):
    super(SawyerPush, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.asarray(self._fixed_start_end, dtype=np.float32).ravel()
    else:
      self._goal = self._target_pos.copy().astype(np.float32)
    self._target_pos = self._goal
    if self._nf_closed_gripper_init:
      self._close_gripper_at_reset()
    return self._get_obs()

  def step(self, action):
    super(SawyerPush, self).step(action)
    dist = np.linalg.norm(self._get_pos_objects() - self._goal)
    obs = self._get_obs()
    r = float(dist <= 0.05)
    return obs, r, False, {}

  def _get_obs(self):
    pos_hand = self.tcp_center.astype(np.float32)
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper_distance_apart = np.clip(
        np.linalg.norm(finger_right - finger_left) / 0.1, 0., 1.)
    obj_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper_distance_apart], obj_pos))  # 7-D
    # ψ: hand just behind the target puck, slightly above, gripper closed.
    ideal_hand = self._goal + np.array([0.0, -0.08, 0.03], dtype=np.float32)
    goal = np.concatenate([ideal_hand, [0.0], self._goal])                 # 7-D
    return np.concatenate([state, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(14, -np.inf, dtype=np.float32),
        high=np.full(14,  np.inf, dtype=np.float32),
        dtype=np.float32)


class SawyerDrawerOpen(_MW_DRAWER):
  """Wrapper for drawer-open-v2: pull the drawer handle to the open position.

  Observation layout (14-dim):
    obs = [ hand_xyz (3)  gripper (1)  handle_xyz (3)          <- state (7) / φ
            ideal_hand_xyz (3)  gripper (1)  target_handle (3) ] <- goal (7) / ψ

  obs_dim = 7, start_index = 0, end_index = -1 (defaults).

  Goal-reaching formulation:
    The 3-D fixed goal ``_goal`` is the *open* handle position (MetaWorld
    ``_target_pos`` = closed site + [0, -maxDist, +0.09]).

    The 7-D ψ slice encodes an *opening* pose: drawer at the open target,
    gripper open and below the handle lip, pulling in −y (expert pull offset):

      ideal_hand = _goal + [0, 0, 0]   # TCP at handle bar: fingers straddle the bar
      goal       = [ideal_hand, gripper=1.0, _goal]
  """

  # Goal bounds: handle spans x∈[-0.2,0.2], target y∈[0.40,0.60] (open), z≈0.09.
  # Goal obs = [hand(3), gripper(1), target_handle(3)] — same layout as state.
  UNIFORM_GOAL_OBS_LOW  = np.array(
      [-0.2, 0.40, 0.05, 0.0, -0.2, 0.40, 0.05], dtype=np.float32)
  UNIFORM_GOAL_OBS_HIGH = np.array(
      [ 0.2, 0.60, 0.20, 1.0,  0.2, 0.60, 0.20], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    return self.UNIFORM_GOAL_OBS_LOW.copy(), self.UNIFORM_GOAL_OBS_HIGH.copy()

  def __init__(self, fixed_start_end=None):
    _require_metaworld('sawyer_drawer_open')
    self._goal = np.zeros(3)
    super(SawyerDrawerOpen, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end = fixed_start_end
    if fixed_start_end is not None:
      self.random_init = False
    self.reset()

  def reset(self):
    super(SawyerDrawerOpen, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.asarray(self._fixed_start_end, dtype=np.float32).ravel()
    else:
      self._goal = self._target_pos.copy().astype(np.float32)
    self._target_pos = self._goal
    return self._get_obs()

  def _ideal_pull_hand(self) -> np.ndarray:
    """TCP at the handle bar position: open fingers straddle the bar."""
    return self._goal.copy()

  def step(self, action):
    super(SawyerDrawerOpen, self).step(action)
    dist = np.linalg.norm(self._get_pos_objects() - self._goal)
    obs = self._get_obs()
    r = float(dist <= 0.03)
    return obs, r, False, {}

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper = np.clip(
        np.linalg.norm(finger_right - finger_left) / 0.1, 0., 1.)
    handle_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper], handle_pos))       # 7-D
    # ψ goal: open gripper below handle, pulling drawer to _goal.
    ideal_hand = self._ideal_pull_hand()
    goal = np.concatenate([ideal_hand, [1.0], self._goal])            # 7-D
    return np.concatenate([state, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(14, -np.inf, dtype=np.float32),
        high=np.full(14,  np.inf, dtype=np.float32),
        dtype=np.float32)


class SawyerButtonPress(_MW_BUTTON):
  """Wrapper for button-press-v2: push the button to the depressed position.

  Observation layout (14-dim):
    obs = [ hand_xyz (3)  gripper (1)  button_xyz (3)           <- state (7) / φ
            ideal_hand_xyz (3)  gripper (1)  target_button (3) ] <- goal (7) / ψ

  obs_dim = 7, start_index = 0, end_index = -1 (defaults).

  Goal-reaching formulation:
    The button moves along the y-axis when pressed (MetaWorld convention).
    φ(s, a) represents the current (hand, gripper, button) configuration.
    ψ(g) represents the terminal state — button depressed to _target_pos,
    hand directly above the button (pressing from above), gripper open.
    obs_to_goal(future_obs) = future_obs[0:7] provides the same layout,
    so future successful states naturally supply the positive training signal.
  """

  # Goal bounds: button spans x∈[-0.1,0.1], target y∈[0.74,0.80] (pressed), z≈0.115.
  UNIFORM_GOAL_OBS_LOW  = np.array(
      [-0.1, 0.74, 0.10, 0.0, -0.1, 0.74, 0.10], dtype=np.float32)
  UNIFORM_GOAL_OBS_HIGH = np.array(
      [ 0.1, 0.80, 0.25, 1.0,  0.1, 0.80, 0.12], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    return self.UNIFORM_GOAL_OBS_LOW.copy(), self.UNIFORM_GOAL_OBS_HIGH.copy()

  def __init__(self, fixed_start_end=None):
    _require_metaworld('sawyer_button_press')
    self._goal = np.zeros(3)
    super(SawyerButtonPress, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.random_init = False  # always spawn at default position so _goal is consistent
    self._fixed_start_end = fixed_start_end
    self.reset()

  def reset(self):
    super(SawyerButtonPress, self).reset()
    if self._fixed_start_end is not None:
      self._goal = np.asarray(self._fixed_start_end, dtype=np.float32).ravel()
    else:
      self._goal = self._target_pos.copy().astype(np.float32)
    self._target_pos = self._goal
    return self._get_obs()

  def step(self, action):
    super(SawyerButtonPress, self).step(action)
    dist = abs(self._get_pos_objects()[1] - self._goal[1])
    obs = self._get_obs()
    r = float(dist <= 0.02)
    return obs, r, False, {}

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector'),
    )
    gripper = np.clip(
        np.linalg.norm(finger_right - finger_left) / 0.1, 0., 1.)
    button_pos = self._get_pos_objects()
    state = np.concatenate((pos_hand, [gripper], button_pos))       # 7-D
    # Ideal end state: hand coincident with the depressed button target so the
    # agent is rewarded for driving the end-effector into the button.
    ideal_hand = self._goal + np.array([0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.concatenate([ideal_hand, [1.0], self._goal])          # 7-D
    return np.concatenate([state, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(14, -np.inf, dtype=np.float32),
        high=np.full(14,  np.inf, dtype=np.float32),
        dtype=np.float32)


class SawyerPeg(_MW_PEG):
  """Wrapper for the SawyerPeg environment."""

  UNIFORM_GOAL_OBS_LOW = np.array(
      [-0.20, 0.39, 0.05, 0.4, -0.32, 0.39, 0.02], dtype=np.float32)
  UNIFORM_GOAL_OBS_HIGH = np.array(
      [0.23, 0.71, 0.17, 0.4, 0.10, 0.71, 0.14], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    return self.UNIFORM_GOAL_OBS_LOW.copy(), self.UNIFORM_GOAL_OBS_HIGH.copy()

  def __init__(self, fixed_start_end=None, randomize_init=True):
    _require_metaworld('sawyer_peg')
    self._goal_pos = np.zeros(3)
    self._randomize_init = bool(randomize_init)
    super(SawyerPeg, self).__init__()
    self._fixed_start_end=fixed_start_end
    self._set_task_called = True
    self._partially_observable = False
    self._freeze_rand_vec = False
    # MetaWorld: random_init controls peg + hole box spawn on reset_model.
    if not self._randomize_init:
      self.random_init = False
    self.reset()

  def reset(self):
    super(SawyerPeg, self).reset()
    pos1 = self._target_pos.copy()
    pos2 = self._get_site_pos("pegHead")
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal_pos = pos1 
    else:
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        t = np.random.random()
        self._goal_pos = t * pos1 + (1 - t) * pos2
    self._target_pos = self._goal_pos    
    return self._get_obs()

  def step(self, action):
    super(SawyerPeg, self).step(action)
    obj_head = self._get_site_pos("pegHead")
    
    scale = np.array([1.0, 2.0, 2.0])
    dist_pos = float(np.linalg.norm((obj_head - self._goal_pos) * scale))
       
    r = float(dist_pos < 0.07)  # Taken from metaworld
    done = False
    info = {}
    return self._get_obs(), r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    
    obj_pos_head = self._get_site_pos("pegHead") 
    obj_pos_grasp = self._get_pos_objects()
    obs = np.concatenate((pos_hand, [gripper_distance_apart], obj_pos_head))
    # the ideal goal state has the peg head in the hole and the gripper slightly 
    # higher than the middle of the peg
    goal = np.concatenate([self._goal_pos + np.array([0.13, 0.0, 0.03]),
                           [0.4], self._goal_pos])
    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)
