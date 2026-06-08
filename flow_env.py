"""Goal-conditioned gym wrapper around Flow's figureeight0 benchmark.

Observation layout (42-dim):
  obs = [ v_0/v_max, ..., v_13/v_max,     <- speeds (14-dim, φ state)
          x_0/L,   ..., x_13/L,           <- positions (14-dim, φ state)
          g_0/v_max, ..., g_13/v_max ]     <- goal speeds only (14-dim, ψ)

  obs_dim = 28 (state = speeds + positions; contrastive/utils.py splits here)
  CRL ``obs_to_goal`` uses state[0:14] only (speeds) for ψ negatives.

Goal:
  The goal g is a state where every vehicle travels at target_velocity
  (20 m/s by default, normalized by network max_speed = 30 m/s).
  With fixed_start_end=None the default goal is used; pass a 14-dim
  float array to override (values should be in [0, 1], i.e. normalized).

  Each step also exposes Flow's dense ``desired_velocity`` reward in
  ``info['flow_dense_reward']`` (for logging only; PPO uses φ·ψ).

Action:
  1-D continuous in [-1, 1] (standard sgcrl convention).
  Internally scaled to [-max_decel, max_accel] = [-3, 3] m/s² before
  being passed to the RL vehicle in AccelEnv.

Environment:
  Backed by Flow's AccelEnv + FigureEightNetwork + SUMO traci.
  SUMO binaries and the Flow package are expected at the locations set
  by the SUMO_HOME / LD_LIBRARY_PATH env vars (written to ~/.bashrc
  during setup).  The wrapper sets sane defaults if those vars are not
  already present so that the env can be imported from the sgcrl venv.

Usage in ppo_contrastive.py:
  python ppo_contrastive.py --env=flow_figureeight --seed=0 \\
      --num_steps=4000000 --ppo_num_envs=4 --log_dir_path=logs/ppo/
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from typing import Optional

import gym
import numpy as np

# ---------------------------------------------------------------------------
# SUMO / Flow path bootstrap
# (only affects this process; harmless if already set in the shell)
# ---------------------------------------------------------------------------
_SUMO_HOME = '/u/mb6458/sumo_binaries/bin'
_FLOW_LIB = '/n/fs/mislresearch/miniconda3/envs/flow/lib'
_FLOW_SRC = '/n/fs/mislresearch/flow'

os.environ.setdefault('SUMO_HOME', _SUMO_HOME)

# Prepend SUMO binary dir to PATH so sub-processes (netconvert, sumo) are found.
_path = os.environ.get('PATH', '')
if _SUMO_HOME not in _path:
    os.environ['PATH'] = _SUMO_HOME + os.pathsep + _path

# Prepend conda-flow lib dir so SUMO shared libs resolve correctly.
_ld = os.environ.get('LD_LIBRARY_PATH', '')
if _FLOW_LIB not in _ld:
    os.environ['LD_LIBRARY_PATH'] = _FLOW_LIB + (os.pathsep + _ld if _ld else '')

# Make the flow package importable from the sgcrl venv (editable source).
if _FLOW_SRC not in sys.path:
    sys.path.insert(0, _FLOW_SRC)


def _patch_flow_traci_vehicle_add() -> None:
  """Flow passes departLane/departPos as str(lane); newer SUMO rejects '0.0'."""
  try:
    from flow.core.kernel.vehicle.traci import TraCIVehicle
  except ImportError:
    return
  if getattr(TraCIVehicle.add, '_sgcrl_patched', False):
    return

  def _add(self, veh_id, type_id, edge, pos, lane, speed):
    _VALID_LANES = frozenset({'random', 'free', 'allowed', 'best', 'first'})
    if isinstance(lane, (int, np.integer)) or (
        isinstance(lane, (float, np.floating)) and float(lane).is_integer()):
      lane_s = str(int(lane))
    elif isinstance(lane, str) and lane in _VALID_LANES:
      lane_s = lane
    else:
      lane_s = str(lane)
    pos_s = str(float(pos))
    speed_s = str(float(speed))

    if veh_id in self.master_kernel.network.rts:
      route_id = 'route{}_0'.format(veh_id)
    else:
      num_routes = len(self.master_kernel.network.rts[edge])
      frac = [val[1] for val in self.master_kernel.network.rts[edge]]
      route_id = 'route{}_{}'.format(
          edge, np.random.choice([i for i in range(num_routes)], size=1, p=frac)[0])

    self.kernel_api.vehicle.addFull(
        veh_id,
        route_id,
        typeID=str(type_id),
        departLane=lane_s,
        departPos=pos_s,
        departSpeed=speed_s)

  _add._sgcrl_patched = True
  TraCIVehicle.add = _add


_patch_flow_traci_vehicle_add()


class FlowFigureEightEnv(gym.Env):
    """Goal-conditioned figureeight0 env compatible with the sgcrl CRL stack.

    Supports variable numbers of RL agents via the NUM_RL class attribute.
    Total vehicles is always N_VEHICLES=14; the remaining (N_VEHICLES - NUM_RL)
    vehicles use IDM + noise.

    Parameters
    ----------
    fixed_start_end:
        Optional 14-element array of *normalized* goal speeds (values in
        [0, 1]).  If None, the default goal (all vehicles at
        target_velocity / max_speed) is used.
    """

    # figureeight0 constants (from flow/benchmarks/figureeight0.py)
    N_VEHICLES: int = 14   # total vehicles
    NUM_RL: int = 1        # RL-controlled vehicles (override in subclasses)
    STATE_OBS_DIM: int = 2 * N_VEHICLES   # speeds + positions → φ
    GOAL_OBS_DIM: int = N_VEHICLES        # goal speeds only → ψ
    MAX_ACCEL: float = 3.0  # m/s²
    TARGET_VELOCITY: float = 20.0  # m/s
    MAX_SPEED: float = 30.0  # m/s  (FigureEightNetwork speed_limit)
    HORIZON: int = 1500

    def __init__(self, fixed_start_end: Optional[np.ndarray] = None):
        super().__init__()
        self._fixed_start_end = fixed_start_end
        self._flow_env = None
        self._goal = self._make_goal(fixed_start_end)
        self._build_flow_env()

        # ---- gym spaces ----
        # Obs = [state (2N), goal (N)], all values in [0, 1].
        obs_len = self.STATE_OBS_DIM + self.GOAL_OBS_DIM
        self.observation_space = gym.spaces.Box(
            low=np.zeros(obs_len, dtype=np.float32),
            high=np.ones(obs_len, dtype=np.float32),
            dtype=np.float32)

        # Actions in [-1, 1] per RL vehicle; scaled internally to [-3, 3].
        self.action_space = gym.spaces.Box(
            low=np.full(self.NUM_RL, -1.0, dtype=np.float32),
            high=np.full(self.NUM_RL,  1.0, dtype=np.float32),
            dtype=np.float32)

        # Needed by Acme's StepLimitWrapper (queried as env._max_episode_steps).
        self._max_episode_steps = self.HORIZON

    @classmethod
    def _make_goal(cls, fixed_start_end: Optional[np.ndarray]) -> np.ndarray:
        if fixed_start_end is not None:
            goal = np.asarray(fixed_start_end, dtype=np.float32).ravel()
            if goal.shape[0] != cls.N_VEHICLES:
                raise ValueError(
                    f'fixed_start_end must have {cls.N_VEHICLES} elements '
                    f'(one normalized speed per vehicle); got {goal.shape[0]}.')
        else:
            goal = np.full(
                cls.N_VEHICLES,
                cls.TARGET_VELOCITY / cls.MAX_SPEED,
                dtype=np.float32)
        return goal

    def _build_flow_env(self) -> None:
        """Create a fresh AccelEnv + SUMO subprocess."""
        if self._flow_env is not None:
            try:
                self._flow_env.terminate()
            except Exception:
                pass

        from flow.benchmarks.figureeight0 import flow_params as _base_fp
        from flow.core.params import (InitialConfig, TrafficLightParams,
                                      VehicleParams, SumoCarFollowingParams)
        from flow.controllers import IDMController, ContinuousRouter, RLController

        fp = deepcopy(_base_fp)
        fp['sim'].render = False
        fp['sim'].port = None  # traci picks a free port per process

        # Rebuild the vehicle mix for the requested NUM_RL split.
        n_human = self.N_VEHICLES - self.NUM_RL
        vehicles = VehicleParams()
        if n_human > 0:
            vehicles.add(
                veh_id='human',
                acceleration_controller=(IDMController, {'noise': 0.2}),
                routing_controller=(ContinuousRouter, {}),
                car_following_params=SumoCarFollowingParams(
                    speed_mode='obey_safe_speed', decel=1.5),
                num_vehicles=n_human)
        vehicles.add(
            veh_id='rl',
            acceleration_controller=(RLController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode='obey_safe_speed'),
            num_vehicles=self.NUM_RL)
        fp['veh'] = vehicles

        env_class = fp['env_name']          # AccelEnv
        network_class = fp['network']       # FigureEightNetwork

        sim_params = fp['sim']
        env_params = fp['env']
        net_params = fp['net']
        vehicles = fp['veh']
        initial_config = fp.get('initial', InitialConfig())
        traffic_lights = fp.get('tls', TrafficLightParams())

        network = network_class(
            name=fp['exp_tag'],
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

        self._flow_env = env_class(
            env_params=env_params,
            sim_params=sim_params,
            network=network,
            simulator=fp['simulator'],
        )

    def _terminate_flow_env(self) -> None:
        if self._flow_env is None:
            return
        try:
            self._flow_env.terminate()
        except Exception:
            pass
        self._flow_env = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _obs(self, full_flow_obs: np.ndarray) -> np.ndarray:
        """Flow state is [speeds, positions]; append goal speeds for ψ only."""
        state = np.asarray(
            full_flow_obs[:self.STATE_OBS_DIM], dtype=np.float32)
        return np.concatenate([state, self._goal])

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale actor output in [-1, 1] to flow's [-max_decel, max_accel]."""
        return np.asarray(action, dtype=np.float32) * self.MAX_ACCEL

    # ------------------------------------------------------------------
    # gym.Env interface (old-style 4-tuple step / scalar obs reset)
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        # Flow's in-place reset can fail after teleports / long SUMO runs
        # (TraCI "Invalid departLane" on vehicle respawn).  Retry once with a
        # fresh SUMO subprocess before surfacing the error.
        last_exc = None
        for attempt in range(2):
            try:
                full_obs = self._flow_env.reset()
                return self._obs(full_obs)
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    self._build_flow_env()
                else:
                    raise
        raise last_exc  # pragma: no cover

    def step(self, action):
        flow_action = self._scale_action(action)
        full_obs, flow_rew, done, info = self._flow_env.step(flow_action)
        obs = self._obs(full_obs)

        # Sparse reward: 1 if every vehicle is within 1 m/s of target speed.
        speeds_ms = obs[:self.N_VEHICLES] * self.MAX_SPEED
        reward = float(np.all(np.abs(speeds_ms - self.TARGET_VELOCITY) < 1.0))

        info = dict(info) if info else {}
        # Flow's dense desired_velocity reward (for logging only; not used by PPO).
        info['flow_dense_reward'] = float(flow_rew)
        return obs, reward, bool(done), info

    def close(self):
        self._terminate_flow_env()

    @property
    def walls(self):
        """Satisfies unwrap_gym_env checks in env_utils."""
        return None


class FlowFigureEight7RL(FlowFigureEightEnv):
    """Figure-eight with 7 RL agents + 7 IDM human agents (7/14 split).

    Action space is 7-D (one acceleration per RL vehicle).
    Observation space is identical to the 1-RL variant (42-D): all 14
    vehicles' speeds + positions form the state, goal is all 14 target speeds.
    """
    NUM_RL: int = 7


class FlowFigureEight14RL(FlowFigureEightEnv):
    """Figure-eight with 14 RL agents (fully autonomous, 14/14 split).

    Action space is 14-D.  No IDM human vehicles; all vehicles are controlled
    by the shared PPO policy.
    """
    NUM_RL: int = 14


# ---------------------------------------------------------------------------
# 2-vehicle variants (1 RL + 1 human, or 2 RL)
# ---------------------------------------------------------------------------

class FlowFigureEight1V1RL(FlowFigureEightEnv):
    """Trivial figure-eight: a single RL vehicle, no other traffic.

    The task reduces to 'accelerate to target_velocity (20 m/s) and hold it'.
    Useful as a sanity-check lower bound — if this doesn't solve, something
    is wrong with the reward / observation pipeline.

    Observation: 3-D  [speed (1)  position (1)  goal_speed (1)]
    Action:      1-D
    """
    N_VEHICLES:    int = 1
    NUM_RL:        int = 1
    STATE_OBS_DIM: int = 2 * 1   # 2
    GOAL_OBS_DIM:  int = 1


class FlowFigureEight4V2RL(FlowFigureEightEnv):
    """Figure-eight with 4 vehicles total: 2 RL + 2 IDM human.

    Observation: 12-D  [speed×4, pos×4, goal_speed×4]
    Action:      2-D
    """
    N_VEHICLES:    int = 4
    NUM_RL:        int = 2
    STATE_OBS_DIM: int = 2 * 4   # 8
    GOAL_OBS_DIM:  int = 4


class FlowFigureEight8V4RL(FlowFigureEightEnv):
    """Figure-eight with 8 vehicles total: 4 RL + 4 IDM human.

    Observation: 24-D  [speed×8, pos×8, goal_speed×8]
    Action:      4-D
    """
    N_VEHICLES:    int = 8
    NUM_RL:        int = 4
    STATE_OBS_DIM: int = 2 * 8   # 16
    GOAL_OBS_DIM:  int = 8


class FlowFigureEight2V1RL(FlowFigureEightEnv):
    """Figure-eight with only 2 vehicles total: 1 RL + 1 IDM human.

    Observation space is 6-D:
      state (4): [speed_0, speed_1, pos_0, pos_1]  → φ
      goal  (2): [target_speed_0, target_speed_1]  → ψ
    Action space is 1-D.
    """
    N_VEHICLES:   int = 2
    NUM_RL:       int = 1
    STATE_OBS_DIM: int = 2 * 2   # 4
    GOAL_OBS_DIM:  int = 2


class FlowFigureEight2V2RL(FlowFigureEightEnv):
    """Figure-eight with only 2 vehicles total, both RL-controlled.

    Observation space is 6-D (same layout as 2V1RL).
    Action space is 2-D (one acceleration per RL vehicle).
    """
    N_VEHICLES:   int = 2
    NUM_RL:       int = 2
    STATE_OBS_DIM: int = 2 * 2   # 4
    GOAL_OBS_DIM:  int = 2
