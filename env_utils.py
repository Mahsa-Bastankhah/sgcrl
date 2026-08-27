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
  _MW_BIN = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']
  _MW_BOX = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['box-close-v2']
  _MW_PEG = _metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['peg-insert-side-v2']
except Exception as _e:  # noqa: BLE001  ImportError, OR mujoco_py's env-var
  # checks, OR any other metaworld/mujoco_py load-time failure.  We
  # deliberately cast a wide net so that point-only workflows (e.g.
  # ppo_rollout_maze.py) can run on hosts that don't have MuJoCo
  # installed/configured.  The sawyer_* envs below still reference these
  # class stubs; we re-raise the cached import error in SawyerBin/Box/Peg
  # `__init__` so users instantiating them get a clear message instead of
  # a downstream `super().reset()` AttributeError.
  _metaworld = None
  _MW_BIN = object
  _MW_BOX = object
  _MW_PEG = object
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
import point_env

os.environ['SDL_VIDEODRIVER'] = 'dummy'

_MANISKILL_IMPORT_ERROR = None
try:
  import gymnasium as _gymnasium
  import mani_skill.envs  # noqa: F401  registers 'PushCube-v1' etc.
  import torch as _torch
  from mani_skill.envs.tasks.mobile_manipulation.open_cabinet_drawer import (
      OpenCabinetDrawerEnv as _MsOpenCabinetDrawerEnv)
  from mani_skill.utils.registration import register_env as _ms_register_env
  from mani_skill.utils.structs.pose import Pose as _MsPose
except Exception as _e:  # noqa: BLE001  cast a wide net, same rationale as
  # metaworld above: hosts without ManiSkill/torch/sapien installed should
  # still be able to run point/sawyer/riverswim workflows.
  _gymnasium = None
  _torch = None
  _MsOpenCabinetDrawerEnv = None
  _ms_register_env = None
  _MsPose = None
  _MANISKILL_IMPORT_ERROR = _e


def _require_maniskill(env_name: str):
  if _MANISKILL_IMPORT_ERROR is not None:
    raise RuntimeError(
        f'Cannot build {env_name}: gymnasium/mani_skill failed to import at '
        f'env_utils load time.  Original error:\n    '
        f'{type(_MANISKILL_IMPORT_ERROR).__name__}: '
        f'{_MANISKILL_IMPORT_ERROR}'
    ) from _MANISKILL_IMPORT_ERROR


# ManiSkill3 ships OpenCabinetDrawer-v1 (and OpenCabinetDoor-v1) but no
# "close" counterpart: OpenCabinetDrawerEnv._initialize_episode
# unconditionally resets every cabinet to fully closed (the lower qlimit),
# and its target_qpos/evaluate() are hardcoded to the "open at least
# min_open_frac" direction (see that class's source). Register a sibling env
# here -- reusing the exact same PartNet-Mobility cabinet/drawer loading,
# robot spawn, and observation code via subclassing -- that instead starts
# each cabinet fully open and requires closing it back down.
if _MANISKILL_IMPORT_ERROR is None:

  @_ms_register_env(
      'CloseCabinetDrawer-v1',
      asset_download_ids=['partnet_mobility_cabinet'],
      max_episode_steps=100,
  )
  class _MsCloseCabinetDrawerEnv(_MsOpenCabinetDrawerEnv):
    """Close variant of ManiSkill3's ``OpenCabinetDrawer-v1``.

    Same Fetch-vs-PartNet-Mobility-cabinet setup as the parent class, with
    three pieces flipped: drawers start fully open (instead of fully
    closed), the target is a *near-closed* joint value (instead of a
    *mostly-open* one), and success requires the joint at or below that
    target (instead of at or above it).
    """

    # Fraction of the joint's [qmin, qmax] range that counts as "closed
    # enough" -- mirrors the parent's `min_open_frac = 0.75` ("open at
    # least 75% of the way"): 0.25 is the same distance from its respective
    # limit (qmax) as 0.75 is from qmin, so both directions of the task are
    # equally hard by this measure.
    max_close_frac = 0.25

    def _after_reconfigure(self, options):
      super()._after_reconfigure(options)
      target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
      qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
      self.target_qpos = qmin + (qmax - qmin) * self.max_close_frac

    def _initialize_episode(self, env_idx, options):
      # Runs the parent's robot-spawn/cabinet-placement logic first, which
      # (as its last step) sets every cabinet to fully closed -- the right
      # starting point for the open task, wrong for this one. Re-open them
      # here instead, redoing the same GPU-apply/kinematics-fetch sequence
      # the parent uses so the corrected qpos is actually visible to the
      # physics/render state afterward.
      super()._initialize_episode(env_idx, options)
      with _torch.device(self.device):
        qlimits = self.cabinet.get_qlimits()  # [b, max_dof, 2]
        self.cabinet.set_qpos(qlimits[env_idx, :, 1])
        self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)
        if self.gpu_sim_enabled:
          self.scene._gpu_apply_all()
          self.scene.px.gpu_update_articulation_kinematics()
          self.scene.px.step()
          self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            _MsPose.create_from_pq(p=self.handle_link_positions(env_idx)))

    def evaluate(self):
      closed_enough = self.handle_link.joint.qpos <= self.target_qpos
      handle_link_pos = self.handle_link_positions()
      link_is_static = (
          _torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
      ) & (_torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
      return {
          'success': closed_enough & link_is_static,
          'handle_link_pos': handle_link_pos,
          'closed_enough': closed_enough,
      }

    def compute_dense_reward(self, obs, action, info):
      # Mirrors the parent's compute_dense_reward, with "amount left to
      # open" (normalized by target_qpos, since the open task starts at
      # qmin=0) replaced by "amount left to close" (normalized by
      # qmax - target_qpos, since this task starts at qmax).
      tcp_to_handle_dist = _torch.linalg.norm(
          self.agent.tcp.pose.p - info['handle_link_pos'], axis=1)
      reaching_reward = 1 - _torch.tanh(5 * tcp_to_handle_dist)
      qmax = self.handle_link.joint.limits[..., 1]
      amount_to_close_left = _torch.div(
          self.handle_link.joint.qpos - self.target_qpos,
          qmax - self.target_qpos)
      close_reward = 2 * (1 - amount_to_close_left)
      reaching_reward[
          amount_to_close_left < 0.999] = 2  # matches parent's early bypass
      close_reward[info['closed_enough']] = 3
      reward = reaching_reward + close_reward
      reward[info['success']] = 5.0
      return reward

    def compute_normalized_dense_reward(self, obs, action, info):
      max_reward = 5.0
      return self.compute_dense_reward(
          obs=obs, action=action, info=info) / max_reward


# mshab (ManiSkill-HAB, https://github.com/arth-shukla/mshab) is a separate
# package from mani_skill itself, installed into its own conda env (it
# requires a specific `mshab` branch of ManiSkill3-beta, incompatible with
# the pip `mani_skill==3.0.1` release the other maniskill_* envs above use).
# Guarded independently so hosts with mani_skill but not mshab still get
# OpenCabinetDrawer-v1/PushCube-v1/PickCube-v1.
_MSHAB_IMPORT_ERROR = None
try:
  import mshab.envs  # noqa: F401  registers 'CloseSubtaskTrain-v0' etc.
  from mshab.envs.planner import plan_data_from_file as _mshab_plan_data_from_file
except Exception as _e:  # noqa: BLE001
  _mshab_plan_data_from_file = None
  _MSHAB_IMPORT_ERROR = _e


def _require_mshab(env_name: str):
  if _MSHAB_IMPORT_ERROR is not None:
    raise RuntimeError(
        f'Cannot build {env_name}: mshab failed to import at env_utils load '
        f'time.  Original error:\n    '
        f'{type(_MSHAB_IMPORT_ERROR).__name__}: {_MSHAB_IMPORT_ERROR}'
    ) from _MSHAB_IMPORT_ERROR


def _mshab_close_subtask_paths():
  """Resolves the CloseSubtaskTrain-v0 task-plan/spawn-data file paths.

  Defaults to task=set_table, split=train -- the only standard HAB task with
  `close` subtasks in mshab's task design (closing a drawer/fridge/cabinet
  after placing an object); confirmed against a downloaded asset listing
  (task_plans/set_table/close/{train,val}/{kitchen_counter,fridge,all}.json
  all exist; tidy_house/prepare_groceries have no `close` subtask dir at
  all). Override via MSHAB_TASK/MSHAB_SPLIT env vars if that changes.

  Uses the single-articulation-type `kitchen_counter.json` plan (not
  `all.json`, which mixes `kitchen_counter` and `fridge` subtasks): mshab's
  own batched-env code (`SequentialTaskEnv._merge_close_subtasks`) asserts
  that every parallel env's subtask at a given slot shares one
  `articulation_type`, so a task plan spanning two types breaks
  `ManiskillVecEnv`/any `num_envs>1` construction (verified: `all.json`
  raises `AssertionError` on `gym.make(..., num_envs=8)`, `kitchen_counter
  .json`/`fridge.json` don't). `kitchen_counter` chosen as the closer analog
  to OpenCabinetDrawer-v1's cabinet drawers; override via MSHAB_OBJ.
  Spawn data isn't split per-object (indexed internally by subtask uid), so
  no equivalent override is needed there.
  """
  ms_asset_dir = os.environ.get(
      'MS_ASSET_DIR', os.path.expanduser('~/.maniskill/data'))
  rearrange_dir = os.path.join(
      ms_asset_dir, 'data', 'scene_datasets', 'replica_cad_dataset',
      'rearrange')
  task = os.environ.get('MSHAB_TASK', 'set_table')
  split = os.environ.get('MSHAB_SPLIT', 'train')
  obj = os.environ.get('MSHAB_OBJ', 'kitchen_counter')
  task_plan_fp = os.path.join(
      rearrange_dir, 'task_plans', task, 'close', split, f'{obj}.json')
  spawn_data_fp = os.path.join(
      rearrange_dir, 'spawn_data', task, 'close', split, 'spawn_data.pt')
  return task_plan_fp, spawn_data_fp


def _mshab_open_subtask_paths():
  """Resolves the OpenSubtaskTrain-v0 task-plan/spawn-data file paths.

  Mirrors `_mshab_close_subtask_paths()` exactly (same task/split/obj env
  var overrides, same `kitchen_counter`-only rationale -- mshab's
  `_merge_open_subtasks` has the same single-articulation-type-per-plan
  requirement `_merge_close_subtasks` does), just pointing at the `open`
  subdirectory instead of `close`. `open/train/kitchen_counter.json`
  confirmed to exist alongside close's (see this repo's plan doc); every
  close-plan entry has a matching open-plan entry at the same list index
  (same build_config_name/init_config_name/articulation_id/
  articulation_handle_link_idx/articulation_handle_active_joint_idx, only
  the subtask uid differs) -- see `scripts/build_open_spawn_data.py`, which
  is how `spawn_data.pt` here was actually built (reusing close's
  already-validated adjacent-room robot spawn poses via that uid mapping,
  NOT mshab's own stock `gen_open_spawn_data`).
  """
  ms_asset_dir = os.environ.get(
      'MS_ASSET_DIR', os.path.expanduser('~/.maniskill/data'))
  rearrange_dir = os.path.join(
      ms_asset_dir, 'data', 'scene_datasets', 'replica_cad_dataset',
      'rearrange')
  task = os.environ.get('MSHAB_TASK', 'set_table')
  split = os.environ.get('MSHAB_SPLIT', 'train')
  obj = os.environ.get('MSHAB_OBJ', 'kitchen_counter')
  task_plan_fp = os.path.join(
      rearrange_dir, 'task_plans', task, 'open', split, f'{obj}.json')
  spawn_data_fp = os.path.join(
      rearrange_dir, 'spawn_data', task, 'open', split, 'spawn_data.pt')
  return task_plan_fp, spawn_data_fp


# TEMPORARY diagnostic restriction, open-subtask-only: most of the 63
# ReplicaCAD scenes' "adjacent room" spawn points are navigationally too
# hard for the current policy/training budget -- even
# maniskill_close_subtask_train, which does show real learning (Final Dist
# ~5m -> ~2m, Success 20-60% by iteration ~2100, see jobs 9342889/9342974),
# empirically only succeeds from episodes that happen to land in one
# specific room.
#
# This scene is not a visual guess -- it's the exact room mshab's own
# `SequentialTaskEnv._load_scene` deterministically assigns to env slot 0
# whenever `randomize_build_configs_per_env=False` (the library default,
# never overridden anywhere in this repo): with that flag off, the
# build_config_idx list is `np.repeat(sorted(build_config_idx_to_task_plans
# .keys()), ...)` -- a pure sort, no RNG -- so slot 0 always gets the
# *lowest* build_config_idx among the scenes present in the close task
# plan. That's build_config_idx=6 -> 'v3_sc0_staging_00.scene_instance
# .json' (confirmed by constructing the scene builder and reading
# `build_config_names_to_idxs` directly). Since `ManiskillCloseSubtaskTrain`
# (and `ppo_video_utils.build_networks`, which every rollout-video path --
# training's own periodic eval-video hook AND this repo's standalone
# render_checkpoint_videos.ipynb -- goes through) always constructs a
# num_envs=1 env off the *full, unfiltered* close task plan, this is
# provably the ONLY room any close-subtask checkpoint video has ever shown,
# regardless of seed/checkpoint -- including the specific success the user
# pointed to (training run seed 9, "Find successful rollouts" attempt 0).
# It also matches close's real num_envs=64 training run: env slot 0 there
# gets this same build_config_idx=6 for its entire lifetime (no per-episode
# reconfiguration randomizes it away).
#
# (An earlier attempt at this identification used visual matching --
# comparing checkpoint-video frames against renders of all 63 scenes -- and
# landed on 'v3_sc2_staging_07.scene_instance.json' as the closest-looking
# candidate. That guess was wrong: the user found its room layout actually
# blocks robot navigation with furniture. This code-derived value replaces
# it and needs no visual confirmation, since it's derived from the same
# deterministic assignment logic that produced every video the user has
# actually watched.)
#
# Restricting maniskill_open_subtask_train to just this one scene isolates
# the navigation-difficulty variable so the goal-vector fix (see
# ManiskillOpenSubtaskTrain's grasp-target-flip note) can actually be
# evaluated without also fighting an unsolved-for-close navigation problem.
# Not applied to close (already demonstrates learning across the full scene
# set) -- revisit/remove once open training is healthy enough to expand
# back to the full task plan.
_OPEN_SUBTASK_EASY_SCENE = 'v3_sc0_staging_00.scene_instance.json'

# Even within `_OPEN_SUBTASK_EASY_SCENE`, the 145 task-plan entries' spawn
# points (copied verbatim from close's own spawn_data.pt -- see
# `scripts/build_open_spawn_data.py`) fall into two well-separated
# world-frame-x clusters, confirmed by directly reading every entry's
# `robot_pos` out of `open`'s spawn_data.pt: 69 entries at x in
# [-1.20, 0.07] (near the two armchairs by the room's inner doorway) and 76
# entries at x in [2.94, 4.18] (near the sofa/coffee-table/staircase side,
# closer to an exterior wall). Visual inspection (render_goal_state.ipynb's
# "View spawn images" section, plan_idx 0/60/90 vs. 30/120/144 in the
# filtered list) confirmed the low-x cluster is the "near the armchairs"
# one the user asked to keep -- per-scene rendering of the high-x cluster
# wasn't independently confirmed as furniture-blocked, but the low-x
# cluster is the one the user has visually verified, so this only trains on
# that one. Threshold picked at the midpoint of the gap between the two
# clusters (0.07 to 2.94) -- not tuned finer than that, since both clusters
# are otherwise tight and well away from the boundary.
_OPEN_SUBTASK_EASY_SPAWN_MAX_X = 1.5


def _filter_open_subtask_plan_data(plan_data):
  """Restricts `plan_data.plans` to `_OPEN_SUBTASK_EASY_SCENE`, then further
  to the `_OPEN_SUBTASK_EASY_SPAWN_MAX_X` (near-armchairs) spawn cluster
  within it, in place.

  Returns `plan_data` (mutated, not copied -- `PlanData`/its `.plans` list
  aren't used elsewhere before this call in either caller). Asserts
  non-empty at each stage rather than silently training on zero episodes if
  the scene name/spawn data ever stop matching the on-disk task plan.
  """
  plan_data.plans = [
      tp for tp in plan_data.plans
      if tp.build_config_name == _OPEN_SUBTASK_EASY_SCENE]
  assert plan_data.plans, (
      f'_filter_open_subtask_plan_data: no plans found for '
      f'build_config_name={_OPEN_SUBTASK_EASY_SCENE!r} -- has the on-disk '
      f'open task plan changed?')

  _, spawn_data_fp = _mshab_open_subtask_paths()
  spawn_data = _torch.load(spawn_data_fp, map_location='cpu', weights_only=False)
  plan_data.plans = [
      tp for tp in plan_data.plans
      if float(spawn_data[tp.subtasks[0].uid]['robot_pos'].flatten()[0])
      < _OPEN_SUBTASK_EASY_SPAWN_MAX_X]
  assert plan_data.plans, (
      f'_filter_open_subtask_plan_data: no plans left after restricting to '
      f'robot_pos x < {_OPEN_SUBTASK_EASY_SPAWN_MAX_X} -- has the on-disk '
      f'open spawn data changed?')

  # Pin to a single fixed spawn (first entry of the near-armchairs cluster)
  # instead of randomizing among all N cluster entries -- removes all
  # initial-state randomness (mshab's `task_plan_idxs` sampling degenerates
  # to always-0 with only one plan entry to choose from; each entry's own
  # spawn_data.pt pose is already unique, i.e. N=1, so this makes every
  # reset fully deterministic).
  plan_data.plans = plan_data.plans[:1]
  return plan_data

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


def _ms_to_numpy(x):
  """Squeeze ManiSkill's batch-dim-1 torch tensor to a flat float32 array."""
  if hasattr(x, 'detach'):
    x = x.detach().cpu().numpy()
  return np.asarray(x, dtype=np.float32).reshape(-1)


def _ms_to_bool_scalar(x):
  if hasattr(x, 'item'):
    return bool(x.item())
  return bool(np.asarray(x).reshape(-1)[0])


def _ms_render_frame(gymnasium_env):
  """Renders one (H, W, 3) uint8 RGB frame from a ManiSkill gymnasium env.

  ManiSkill's `.render()` returns a torch tensor batched over num_envs
  (here always 1); squeeze that leading dim and cast to uint8.
  """
  img = gymnasium_env.render()
  if hasattr(img, 'detach'):
    img = img.detach().cpu().numpy()
  img = np.asarray(img)
  if img.ndim == 4:  # (num_envs, H, W, C) -> (H, W, C)
    img = img[0]
  return np.ascontiguousarray(img).astype(np.uint8)


class ManiskillPushCube(gym.Env):
  """gym (old-API) wrapper around ManiSkill3's ``PushCube-v1``.

  State layout (6,): [tcp_pos(3), obj_pos(3)]
  Goal  layout (6,): [tcp_target_pos(3), goal_pos(3)]

  This mirrors SawyerBin/Box/Peg's convention exactly: state/goal live in
  pure Cartesian task-space (end-effector + object position), with no raw
  joint angles/velocities. That choice isn't just cosmetic parity -- it's
  required for correctness under this codebase's CRL scheme.
  `EpisodeReplay.sample()` (ppo_learner.py) trains the goal encoder psi not
  on this class's own goal vector, but on `obs_to_goal(s_j)` = the full
  *state* slice of a random future timestep in the same episode (a
  HER-style relabeled "achieved goal"). For that relabeled goal to be
  in-distribution with the goal this class hands the policy at rollout
  time, "state" must already be a plausible goal (i.e. everything in it
  should be a sensible thing to want to reach) -- an earlier version of
  this class included qpos/qvel(18 dims) in "state" but zeroed them in the
  rollout-time goal, so the critic was trained on rich, real joint
  configurations as goals but evaluated at rollout on a near-all-zero
  vector, an extreme train/eval mismatch that made phi*psi diverge instead
  of learn. Restricting state/goal to the same 3-dim Cartesian positions
  (tcp/target-tcp and object/goal) that Sawyer uses avoids this: both the
  rollout-time goal and the replay-relabeled goal are ordinary, bounded
  table positions from the same distribution.

  ``control_mode='pd_ee_delta_pos'`` (4-dim: dx, dy, dz, gripper) is chosen
  to match: Cartesian end-effector control doesn't need qpos feedback to
  be controllable, unlike joint-space control.

  ``tcp_target_pos`` is the "get behind the cube to push" pose already
  used by the env's own dense reward (`obj_pos + [-(cube_half_size+0.005),
  0, 0]`) -- same role as SawyerBin's `self._goal + [0, 0, 0.03]` hand
  target slightly offset from the object goal.

  ``reward`` is the binary ``info['success']`` signal (matching the
  SawyerBin/Box/Peg convention); PPO here trains on phi*psi (the
  contrastive critic), not this reward, so this only affects logging
  (ep_return_mean / reward_env_mean) and the eval success metric.

  ``done`` is always False: episode truncation is owned entirely by the
  outer ``step_limit.StepLimitWrapper``.  Its ``max_episode_steps`` (set in
  ``load()`` below) must equal ManiSkill's own registered limit (50) so
  that this class is never stepped past the point where gymnasium's
  internal ``TimeLimit`` would have already fired once (documented as
  undefined behavior by gymnasium).
  """

  _CUBE_HALF_SIZE = 0.02  # matches PushCubeEnv.cube_half_size
  _PUSH_OFFSET = np.array([-(_CUBE_HALF_SIZE + 0.005), 0.0, 0.0], dtype=np.float32)
  STATE_DIM = 6

  def __init__(self, fixed_start_end=None, render_mode=None):
    super().__init__()
    _require_maniskill('maniskill_pushcube')
    del fixed_start_end  # PushCube randomizes cube+goal every reset itself.
    self._env = _gymnasium.make(
        'PushCube-v1',
        obs_mode='state_dict',
        control_mode='pd_ee_delta_pos',
        render_mode=render_mode,
        num_envs=1)
    # Fresh *old*-gym Box objects: acme's gym_wrapper checks
    # `isinstance(space, gym.spaces.Box)` against the old `gym` package, so
    # reusing ManiSkill's `gymnasium.spaces.Box` objects directly would fail
    # that check with `ValueError: Unexpected gym space`.
    #
    # Bounds must be *finite*: --uniform_sampling draws negative goals via
    # `np.random.uniform(goal_low, goal_high)`, where goal_low/goal_high come
    # straight from this Box's minimum/maximum over the state slice
    # (ppo_learner.py's `spec.observations.minimum/maximum`); +-inf bounds
    # make that draw raise `OverflowError: Range exceeds valid bounds`.
    pos_bound = 1.0  # meters; generous margin over the robot/table workspace
    pos_low = np.full(3, -pos_bound, dtype=np.float32)
    pos_high = np.full(3, pos_bound, dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.concatenate([pos_low, pos_low, pos_low, pos_low]),
        high=np.concatenate([pos_high, pos_high, pos_high, pos_high]),
        dtype=np.float32)
    action_dim = int(self._env.action_space.shape[-1])
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

  def reset(self):
    obs_dict, _info = self._env.reset()
    return self._get_obs(obs_dict)

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    obs_dict, _reward, _terminated, _truncated, info = self._env.step(action)
    reward = float(_ms_to_bool_scalar(info.get('success', False)))
    return self._get_obs(obs_dict), reward, False, {}

  def close(self):
    self._env.close()

  def render(self):
    """Returns one (H, W, 3) uint8 RGB frame. Requires render_mode='rgb_array'."""
    return _ms_render_frame(self._env)

  def _get_obs(self, obs_dict):
    tcp_pos = _ms_to_numpy(obs_dict['extra']['tcp_pose'])[:3]
    obj_pos = _ms_to_numpy(obs_dict['extra']['obj_pose'])[:3]
    goal_pos = _ms_to_numpy(obs_dict['extra']['goal_pos'])
    tcp_target_pos = obj_pos + self._PUSH_OFFSET

    state = np.concatenate([tcp_pos, obj_pos]).astype(np.float32)
    goal = np.concatenate([tcp_target_pos, goal_pos]).astype(np.float32)
    return np.concatenate([state, goal]).astype(np.float32)


class ManiskillPickCube(gym.Env):
  """gym (old-API) wrapper around ManiSkill3's ``PickCube-v1``.

  State layout (7,): [tcp_pos(3), gripper_aperture(1), obj_pos(3)]
  Goal  layout (7,): [tcp_target_pos(3), gripper_target(1), goal_pos(3)]

  Mirrors ManiskillPushCube's rationale (see its docstring) for why
  state/goal are restricted to Cartesian task-space features only.
  Unlike PushCube, PickCube's success condition requires *grasping*
  (info['is_grasped']) in addition to spatial proximity -- a pure
  [tcp_pos, obj_pos] state can't distinguish "nudged near the goal" from
  "actually gripped and placed," so a scalar gripper-aperture feature is
  added, mirroring SawyerBin's [pos_hand(3), gripper_dist(1), obj_pos(3)]
  convention exactly.

  ``gripper_aperture`` = qpos[7] + qpos[8], the sum of the two Panda
  finger-joint positions (each in [0, 0.04] per
  ``agent.robot.get_qlimits()``), i.e. the open/closed span in meters,
  range [0, 0.08].

  ``gripper_target`` = 2 * cube_half_size = 0.04: the physical width of
  the cube, i.e. "fingers spread exactly wide enough to have closed
  around this object." A shaping heuristic (not ground truth), same
  spirit as SawyerBin's own approximate gripper-target constant.

  ``tcp_target_pos`` = goal_pos (NOT obj_pos). Unlike PushCube's
  continuous "trail behind the current cube" target (which matches its
  own dense reward), PickCube's success requires tcp/object/goal to all
  coincide at the FINAL goal location (grasp + lift + place, no release
  phase) -- so the tcp target is the fixed goal position, matching
  SawyerBin's own convention of targeting the fixed final goal, not the
  object's current position.

  ``reward``/``done`` conventions match ManiskillPushCube exactly.
  """

  _CUBE_HALF_SIZE = 0.02  # matches PickCubeEnv.cube_half_size
  _GRIPPER_TARGET = np.array([2 * _CUBE_HALF_SIZE], dtype=np.float32)  # 0.04
  STATE_DIM = 7

  def __init__(self, fixed_start_end=None, render_mode=None):
    super().__init__()
    _require_maniskill('maniskill_pickcube')
    del fixed_start_end  # PickCube randomizes cube+goal every reset itself.
    self._env = _gymnasium.make(
        'PickCube-v1',
        obs_mode='state_dict',
        control_mode='pd_ee_delta_pos',
        render_mode=render_mode,
        num_envs=1)
    pos_bound = 1.0  # meters; generous margin over the robot/table workspace
    pos_low = np.full(3, -pos_bound, dtype=np.float32)
    pos_high = np.full(3, pos_bound, dtype=np.float32)
    grip_low = np.zeros(1, dtype=np.float32)
    grip_high = np.full(1, 0.08, dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=np.concatenate([pos_low, grip_low, pos_low,
                           pos_low, grip_low, pos_low]),
        high=np.concatenate([pos_high, grip_high, pos_high,
                            pos_high, grip_high, pos_high]),
        dtype=np.float32)
    action_dim = int(self._env.action_space.shape[-1])
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

  def reset(self):
    obs_dict, _info = self._env.reset()
    return self._get_obs(obs_dict)

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    obs_dict, _reward, _terminated, _truncated, info = self._env.step(action)
    reward = float(_ms_to_bool_scalar(info.get('success', False)))
    return self._get_obs(obs_dict), reward, False, {}

  def close(self):
    self._env.close()

  def render(self):
    """Returns one (H, W, 3) uint8 RGB frame. Requires render_mode='rgb_array'."""
    return _ms_render_frame(self._env)

  def _get_obs(self, obs_dict):
    tcp_pos = _ms_to_numpy(obs_dict['extra']['tcp_pose'])[:3]
    obj_pos = _ms_to_numpy(obs_dict['extra']['obj_pose'])[:3]
    goal_pos = _ms_to_numpy(obs_dict['extra']['goal_pos'])
    qpos = _ms_to_numpy(obs_dict['agent']['qpos'])
    gripper_aperture = np.array([qpos[7] + qpos[8]], dtype=np.float32)

    state = np.concatenate(
        [tcp_pos, gripper_aperture, obj_pos]).astype(np.float32)
    goal = np.concatenate(
        [goal_pos, self._GRIPPER_TARGET, goal_pos]).astype(np.float32)
    return np.concatenate([state, goal]).astype(np.float32)


class ManiskillOpenCabinetDrawer(gym.Env):
  """gym (old-API) wrapper around ManiSkill3's ``OpenCabinetDrawer-v1``.

  State layout (10,): [tcp_pos(3), drawer_qpos(1), gripper_aperture(1),
                        is_grasping(1), sin(base_theta)(1), cos(base_theta)(1),
                        base_x(1), base_y(1)]
  Goal  layout (6,): [handle_target_pos(3), drawer_target_qpos(1),
                       gripper_target(1)=OPEN, is_grasping_target(1)=0]

  Only the first 6 state dims mirror the goal dims 1:1 (same convention as
  ManiskillPushCube/PickCube); the last 4 (heading, base_x, base_y) are
  state-only and are never used as -- or matched against -- a goal. This
  is an *asymmetric* state/goal split: STATE_DIM (10) != GOAL_DIM (6). It
  works cleanly with the CRL critic because sa_encoder/g_encoder are
  separate MLP towers (`contrastive/networks.py::_repr_fn`) combined only
  via a dot product on their *output* embeddings -- their input widths
  are independent. It requires two plumbing pieces elsewhere, both
  already wired up for this env: (1) `env_utils.load()` must return
  `obs_dim=STATE_DIM=10`, not `total_obs // 2` (which would be wrong here);
  (2) `ppo_contrastive.py` must set `config.end_index=GOAL_DIM=6` (default
  -1 would try to read past the end of the raw per-step obs, since the
  wrapper's own goal is only 6 long) so `obs_to_goal` -- used both by
  `ObservationFilterWrapper` and by HER hindsight relabeling in
  `EpisodeReplay._obs_to_goal` -- slices heading/base_x/base_y out of any
  goal. Because that slicing is a plain prefix `state[:end_index]`,
  gripper_aperture/is_grasping (which DO have goal counterparts, below)
  must sit ahead of heading/base_xy (which don't) in the state ordering.

  gripper_aperture and is_grasping DO get goal counterparts, unlike
  heading/base position (see below): although OpenCabinetDrawerEnv's own
  `evaluate()`/`compute_dense_reward()` never reference grip state at all
  (success is purely `handle_link.joint.qpos >= target_qpos` plus the
  handle being near-static), "gripper open / not grasping" is the one
  asset-independent, physically-reasonable value to target -- after the
  drawer reaches its target position, there's no more need to hold the
  handle, so releasing it is a sensible terminal target regardless of
  which PartNet-Mobility cabinet or handle geometry this episode sampled
  (unlike e.g. a fixed *closed* aperture, which would vary per handle).
  `gripper_target = grip_high = 0.10` (this class's existing fully-open
  aperture bound) and `is_grasping_target = 0.0` are therefore reused as
  fixed per-step goal values for the whole episode.

  heading and base (x, y) deliberately get NO goal counterpart. Unlike
  gripper aperture, there is no asset-independent "reasonable" final
  heading/base position: OpenCabinetDrawerEnv's own reward/success ignore
  them completely, and fixing a target (e.g. the episode's reset-time
  pose) would make the CRL reward -- which is the literal φ·ψ dot product
  used as the PPO reward, see the `ppo_crl_repr_tau` slurm knob -- reward
  staying near the start pose, fighting the retreat-while-pulling motion
  this task plausibly needs as the handle recedes from the cabinet while
  the drawer opens. So, same as before, they stay state-only.

  Fetch (the mobile-base robot this task requires) has 15-dim qpos/qvel (3
  base + 3 torso/head + 7 arm + 2 gripper); 5 of those raw dims are used
  here (2 gripper-finger qpos, summed into one aperture scalar; base
  qpos[0:2], the (x, y) position; base qpos[2], the heading, reencoded as
  sin/cos):

  - Robot base (x, y) IS included (added alongside heading, unlike the
    single-env wrapper's original design). Reason: Fetch's action space
    is a `CombinedController` with independent `arm`(3)/`gripper`(1)/
    `body`(3)/`base`(2, forward-vel + turn-rate) sub-controllers -- base
    translation is a genuinely separate control channel from the
    arm/gripper, and the redundant `(base_xy, arm_joints) -> tcp_pos`
    kinematics mean `tcp_pos` alone can't tell the policy how much it has
    already retreated (or still can) as the drawer's handle recedes while
    opening, which risks running the arm out of reach if the base doesn't
    back away in time.
  - Robot base heading (theta) IS included. Reason: Fetch's base
    controller (`PDBaseForwardVelController.set_action`) is ego-centric --
    it rotates the 2D [forward_vel, turn_rate] action by the *current*
    heading (`qpos[:, 2]`) before applying it. Two states with identical
    tcp_pos but different (unobserved) heading respond differently to the
    same action, which breaks the Markov property for a memoryless PPO
    policy. This is a real effect here, not a rounding corner case:
    `OpenCabinetDrawerEnv._initialize_episode` spawns the robot at a
    randomized angle around the cabinet with heading noise of +/-0.05*pi
    on top of that, so heading varies by tens of degrees across episodes.
    Encoded as (sin, cos) rather than raw radians since the heading joint
    is an unbounded revolute (`qlimit = [-inf, inf]`).
  - is_grasping is included as an explicit boolean-valued feature (see
    above for its goal target): `gripper_aperture` alone is a continuous
    proxy for openness but doesn't capture contact/force -- e.g. fingers
    can be nearly closed without actually gripping the handle -- so
    `agent.is_grasping(handle_link)` (contact-force + angle based) gives a
    more direct grasp signal.
  - No object/handle orientation: the cabinet is spawned at an identity
    quaternion every episode (``OpenCabinetDrawerEnv._initialize_episode``
    randomizes only the robot's start pose, never the cabinet's), so
    unlike PickCube's randomized cube there is no rotational DOF to track.
  - No handle/robot velocity: the env's own success condition also
    requires the handle link to be near-static (in addition to open
    enough), but adding raw velocity here risks the same HER
    train/rollout-mismatch failure mode documented on ManiskillPushCube,
    so it's left out, same call as ManiskillPickCube.

  ``drawer_qpos`` = ``target_link_qpos``, ManiSkill's own name for the
  drawer's prismatic joint value (how far open it currently is).

  ``gripper_aperture`` = qpos[13] + qpos[14], the sum of Fetch's two
  finger-joint positions (each in [0, 0.05] per
  ``agent.robot.get_qlimits()``), range [0, 0.10]. Sum, not difference:
  Fetch's gripper controller (`PDJointPosMimicControllerConfig` in
  `fetch.py`, with `mimic={"r_gripper_finger_joint": {"joint":
  "l_gripper_finger_joint"}}`) drives the two fingers to (near-)identical
  positions, each moving outward from the gripper's centerline. Their
  *difference* is therefore always ~0 (mimic-tracking error/contact
  noise, not signal); their *sum* is the actual fingertip-to-fingertip
  gap, i.e. how open the gripper physically is -- the same convention
  ManiskillPickCube already uses for the Panda's two symmetric fingers.

  ``is_grasping`` = ``agent.is_grasping(handle_link)``
  (`mani_skill/agents/robots/fetch/fetch.py`), a contact-force- and
  -angle-based check (>= 0.5N on both fingers, within 85 degrees of the
  closing direction) of whether the gripper is actually gripping the
  drawer's handle link, cast from bool to float32 (0.0/1.0).

  ``handle_target_pos`` = the handle's current world position, which
  ManiSkill itself recomputes every control step for its own dense
  reward/visualization (see ``OpenCabinetDrawerEnv._after_control_step``);
  it moves as the drawer slides open, analogous to PushCube's "chase"
  ``tcp_target_pos``.

  ``drawer_target_qpos`` = ``OpenCabinetDrawerEnv.target_qpos``, the
  per-episode (per-cabinet) joint value that counts as "open enough"
  (``min_open_frac`` of the way through this cabinet's own joint limits).

  ``control_mode='pd_ee_delta_pos'`` matches the tabletop wrappers'
  choice of position-only end-effector control (no wrist rotation needed,
  since no object-orientation goal is ever reached); for Fetch this is a
  combined controller that also drives the mobile base (forward
  velocity + turn rate) and torso/head joints via the same action vector.
  """

  STATE_DIM = 10
  GOAL_DIM = 6

  def __init__(self, fixed_start_end=None, render_mode=None):
    super().__init__()
    _require_maniskill('maniskill_open_cabinet_drawer')
    del fixed_start_end  # OpenCabinetDrawer randomizes cabinet/robot pose itself.
    self._env = _gymnasium.make(
        'OpenCabinetDrawer-v1',
        obs_mode='state_dict',
        control_mode='pd_ee_delta_pos',
        render_mode=render_mode,
        num_envs=1)
    pos_bound = 3.0  # meters; generous margin over the mobile workspace
    pos_low = np.full(3, -pos_bound, dtype=np.float32)
    pos_high = np.full(3, pos_bound, dtype=np.float32)
    # Drawer joint travel observed empirically across PartNet-Mobility
    # cabinets as roughly [0, 0.5]; pad the upper bound for margin.
    qpos_low = np.zeros(1, dtype=np.float32)
    qpos_high = np.full(1, 0.6, dtype=np.float32)
    grip_low = np.zeros(1, dtype=np.float32)
    grip_high = np.full(1, 0.10, dtype=np.float32)
    grasp_low = np.zeros(1, dtype=np.float32)   # is_grasping is 0.0/1.0
    grasp_high = np.ones(1, dtype=np.float32)
    heading_low = np.full(2, -1.0, dtype=np.float32)   # (sin, cos) range
    heading_high = np.full(2, 1.0, dtype=np.float32)
    base_xy_low = pos_low[:2]     # reuse the same generous position margin
    base_xy_high = pos_high[:2]
    state_low = np.concatenate(
        [pos_low, qpos_low, grip_low, grasp_low, heading_low, base_xy_low])
    state_high = np.concatenate(
        [pos_high, qpos_high, grip_high, grasp_high, heading_high,
         base_xy_high])
    goal_low = np.concatenate([pos_low, qpos_low, grip_low, grasp_low])
    goal_high = np.concatenate([pos_high, qpos_high, grip_high, grasp_high])
    self.observation_space = gym.spaces.Box(
        low=np.concatenate([state_low, goal_low]),
        high=np.concatenate([state_high, goal_high]),
        dtype=np.float32)
    action_dim = int(self._env.action_space.shape[-1])
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

  def reset(self):
    obs_dict, _info = self._env.reset()
    return self._get_obs(obs_dict)

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    obs_dict, _reward, _terminated, _truncated, info = self._env.step(action)
    reward = float(_ms_to_bool_scalar(info.get('success', False)))
    return self._get_obs(obs_dict), reward, False, {}

  def close(self):
    self._env.close()

  def render(self):
    """Returns one (H, W, 3) uint8 RGB frame. Requires render_mode='rgb_array'."""
    return _ms_render_frame(self._env)

  _GRIPPER_TARGET = np.array([0.10], dtype=np.float32)   # fully open
  _GRASP_TARGET = np.array([0.0], dtype=np.float32)      # not grasping

  def _get_obs(self, obs_dict):
    tcp_pos = _ms_to_numpy(obs_dict['extra']['tcp_pose'])[:3]
    drawer_qpos = _ms_to_numpy(obs_dict['extra']['target_link_qpos'])
    handle_target_pos = _ms_to_numpy(obs_dict['extra']['target_handle_pos'])
    drawer_target_qpos = _ms_to_numpy(self._env.unwrapped.target_qpos)
    qpos = _ms_to_numpy(obs_dict['agent']['qpos'])
    gripper_aperture = np.array([qpos[13] + qpos[14]], dtype=np.float32)
    is_grasping = _ms_to_numpy(self._env.unwrapped.agent.is_grasping(
        self._env.unwrapped.handle_link))
    base_theta = float(qpos[2])
    heading = np.array(
        [np.sin(base_theta), np.cos(base_theta)], dtype=np.float32)
    base_xy = qpos[:2]

    state = np.concatenate([
        tcp_pos, drawer_qpos, gripper_aperture, is_grasping, heading,
        base_xy]).astype(np.float32)
    goal = np.concatenate([
        handle_target_pos, drawer_target_qpos, self._GRIPPER_TARGET,
        self._GRASP_TARGET]).astype(np.float32)
    return np.concatenate([state, goal]).astype(np.float32)


class ManiskillCloseCabinetDrawer(gym.Env):
  """gym (old-API) wrapper around this module's ``CloseCabinetDrawer-v1``.

  Sibling of `ManiskillOpenCabinetDrawer` for the close direction: same
  state/goal layout, same STATE_DIM/GOAL_DIM (10/6), same asymmetric
  state/goal split rationale (see that class's docstring -- it applies here
  unchanged, just with the underlying env swapped for the closing variant
  registered above). Drawers start fully open each episode and the goal is
  to push them back down to (near) fully closed; "gripper open / not
  grasping" is still the sensible per-step goal target for
  gripper_aperture/is_grasping once the drawer reaches its target (release
  the handle, same reasoning as the open task).
  """

  STATE_DIM = 10
  GOAL_DIM = 6

  def __init__(self, fixed_start_end=None, render_mode=None):
    super().__init__()
    _require_maniskill('maniskill_close_cabinet_drawer')
    del fixed_start_end  # CloseCabinetDrawer randomizes cabinet/robot pose itself.
    self._env = _gymnasium.make(
        'CloseCabinetDrawer-v1',
        obs_mode='state_dict',
        control_mode='pd_ee_delta_pos',
        render_mode=render_mode,
        num_envs=1)
    pos_bound = 3.0  # meters; generous margin over the mobile workspace
    pos_low = np.full(3, -pos_bound, dtype=np.float32)
    pos_high = np.full(3, pos_bound, dtype=np.float32)
    # Drawer joint travel observed empirically across PartNet-Mobility
    # cabinets as roughly [0, 0.5]; pad the upper bound for margin (same as
    # ManiskillOpenCabinetDrawer -- same underlying joints).
    qpos_low = np.zeros(1, dtype=np.float32)
    qpos_high = np.full(1, 0.6, dtype=np.float32)
    grip_low = np.zeros(1, dtype=np.float32)
    grip_high = np.full(1, 0.10, dtype=np.float32)
    grasp_low = np.zeros(1, dtype=np.float32)   # is_grasping is 0.0/1.0
    grasp_high = np.ones(1, dtype=np.float32)
    heading_low = np.full(2, -1.0, dtype=np.float32)   # (sin, cos) range
    heading_high = np.full(2, 1.0, dtype=np.float32)
    base_xy_low = pos_low[:2]     # reuse the same generous position margin
    base_xy_high = pos_high[:2]
    state_low = np.concatenate(
        [pos_low, qpos_low, grip_low, grasp_low, heading_low, base_xy_low])
    state_high = np.concatenate(
        [pos_high, qpos_high, grip_high, grasp_high, heading_high,
         base_xy_high])
    goal_low = np.concatenate([pos_low, qpos_low, grip_low, grasp_low])
    goal_high = np.concatenate([pos_high, qpos_high, grip_high, grasp_high])
    self.observation_space = gym.spaces.Box(
        low=np.concatenate([state_low, goal_low]),
        high=np.concatenate([state_high, goal_high]),
        dtype=np.float32)
    action_dim = int(self._env.action_space.shape[-1])
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

  def reset(self):
    obs_dict, _info = self._env.reset()
    return self._get_obs(obs_dict)

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    obs_dict, _reward, _terminated, _truncated, info = self._env.step(action)
    reward = float(_ms_to_bool_scalar(info.get('success', False)))
    return self._get_obs(obs_dict), reward, False, {}

  def close(self):
    self._env.close()

  def render(self):
    """Returns one (H, W, 3) uint8 RGB frame. Requires render_mode='rgb_array'."""
    return _ms_render_frame(self._env)

  _GRIPPER_TARGET = np.array([0.10], dtype=np.float32)   # fully open
  _GRASP_TARGET = np.array([0.0], dtype=np.float32)      # not grasping

  def _get_obs(self, obs_dict):
    tcp_pos = _ms_to_numpy(obs_dict['extra']['tcp_pose'])[:3]
    drawer_qpos = _ms_to_numpy(obs_dict['extra']['target_link_qpos'])
    handle_target_pos = _ms_to_numpy(obs_dict['extra']['target_handle_pos'])
    drawer_target_qpos = _ms_to_numpy(self._env.unwrapped.target_qpos)
    qpos = _ms_to_numpy(obs_dict['agent']['qpos'])
    gripper_aperture = np.array([qpos[13] + qpos[14]], dtype=np.float32)
    is_grasping = _ms_to_numpy(self._env.unwrapped.agent.is_grasping(
        self._env.unwrapped.handle_link))
    base_theta = float(qpos[2])
    heading = np.array(
        [np.sin(base_theta), np.cos(base_theta)], dtype=np.float32)
    base_xy = qpos[:2]

    state = np.concatenate([
        tcp_pos, drawer_qpos, gripper_aperture, is_grasping, heading,
        base_xy]).astype(np.float32)
    goal = np.concatenate([
        handle_target_pos, drawer_target_qpos, self._GRIPPER_TARGET,
        self._GRASP_TARGET]).astype(np.float32)
    return np.concatenate([state, goal]).astype(np.float32)


# Meters to pull the close-subtask "closed handle" goal position out from
# the literal closed position, back toward "open," along the joint's own
# local motion axis at target_qpos. Without this, the target sits flush
# with (or slightly inside) the cabinet face -- a policy chasing that point
# tends to swing over the top of the cabinet rather than approach head-on,
# since the literal closed position isn't a comfortable place to be with an
# open gripper anyway (a real "push it shut and let go" motion ends with
# the hand pulled back in front of the drawer, not jammed against its
# face). Shared by `ManiskillCloseSubtaskTrain` (single-env) and
# `ManiskillVecEnv` (GPU-batched training) via `_close_subtask_closed_handle_positions`
# below, so both paths use the same constant.
_CLOSE_SUBTASK_APPROACH_OFFSET_M = 0.05

# NOTE: five approaches to `base_target_xy` (below) were tried, in order:
#   1. Linearly extrapolating _CLOSE_SUBTASK_APPROACH_OFFSET_M's own
#      axis_dir out by a fixed 0.6m. That axis is only a *local tangent*,
#      measured from a tiny (5% of [qmin, qmax]) probe -- exact for
#      prismatic drawers (a straight axis), but for revolute joints
#      (cabinet doors/fridges) a 0.6m linear extrapolation of that tangent
#      diverges from the door's actual swept arc and can land the target
#      inside a wall the door is hinged near (job 9300593's rendered
#      goal-state images: 2/6 examples showed the camera embedded in flat
#      wall geometry).
#   2. Plain nearest-navmesh-vertex-to-handle (no reference distance at
#      all). Fixed the wall-clipping, but landed 1.3-5.5m from the handle
#      in the scenes checked -- much farther than Fetch can reach, per user
#      visual review of job 9300637's images.
#   3. mshab's own reference interaction distance (kitchen_counter:
#      [0.3, 1.5]m local-x box in front of the handle, per
#      `gen_spawn_positions.py::gen_close_spawn_data`'s own inline comment
#      -- see that file for the pre-adjacent-room-change original),
#      intersected with the navmesh. Verified via direct sim-state
#      read-back (job 9301610) to land consistently ~0.30m from the
#      handle -- correct math, but judged not close enough for the base to
#      actually support grasping/pushing the handle (user judgment call,
#      not a bug).
#   4. Nearest navmesh vertex within a small (0.1m) radius of the handle,
#      falling back to nearest-overall if none qualifies. Always fell back
#      (job 9301674: still ~0.30m) -- the counter has physical depth, so no
#      real floor exists within 0.1m of the handle's projected (x, y) at
#      all; the 0.3m floor from approach #3 is apparently close to the
#      *actual* physical minimum for this furniture, not an artifact of the
#      box choice.
#   5. #1's small-tangent-offset idea, at the user's requested 0.1m instead
#      of 0.6m, THEN snapped to the nearest navmesh vertex (combining #1
#      with #2's safety net). Still landed at ~0.30m (job 9301701, byte-
#      identical to #3/#4's images) -- because the raw 0.1m-offset point is
#      *itself* inside the counter's footprint (per #4's finding), its
#      nearest navmesh vertex is the exact same one the handle's own
#      nearest-vertex query already finds. Snapping and "close to the
#      handle" are fundamentally incompatible here, not fixable by tweaking
#      the offset distance.
# Dropped the navmesh snap for #6 (like `handle_target_p`, the gripper
# target, already does) and used the raw 0.1m tangent-offset point
# directly -- but a *separate*, serious bug (see `_get_obs`'s comment on
# `base_xy`: `agent.robot.get_qpos()[:2]` is relative to a per-scene root
# anchor pose, not world position) had been silently corrupting every
# distance measurement/render up to this point, including the "0.30m"
# figures above (which were self-consistently wrong, not right). After
# fixing that, an actual physics-engine collision check (base_standoff_
# explorer.ipynb's check_collision(), stepping physics and reading real
# penetration depth/impulse against the counter's own collision mesh, not
# a visual guess) swept distances across all 6 scenes and found a sharp,
# consistent transition: 0 penetrating contacts at 0.35m, 5+ at 0.32m and
# closer, worsening down to 0.1m (up to 22cm of overlap, contact impulses
# >300,000) -- 0.35m is the empirically-validated collision-free *minimum*
# for this furniture. 0.6m was the user's first choice after interactively
# exploring base_standoff_explorer.ipynb post-fix (visually reasonable with
# headroom above that bare 0.35m minimum); bumped further to 1.0m per
# explicit request for even more standoff room, still well clear of the
# 0.35m collision floor.
_CLOSE_SUBTASK_BASE_STANDOFF_M = 1.0

# Open-subtask's own base-standoff distance -- deliberately a SEPARATE
# constant from `_CLOSE_SUBTASK_BASE_STANDOFF_M` above, not a shared one,
# even though `_open_subtask_open_handle_positions` originally reused that
# close-tuned value verbatim (see that function's docstring for the
# separate directional bug this was tangled up with: the axis `base_target_xy`
# is offset along was pointing into the cabinet instead of out into the
# room, so no standoff *distance* choice could have looked right until that
# was fixed first). `_CLOSE_SUBTASK_BASE_STANDOFF_M` reflects an extensive,
# already-validated tuning history specific to close (physics-verified
# 0.35m collision-free minimum, user-selected 1.0m with headroom above
# that -- see this constant's own comment) that has nothing to do with
# open's geometry; decoupling them means retuning one can never silently
# move the other.
#
# 0.5m was the user's first choice after visually reviewing the
# (direction-corrected) rendered goal state, but was flagged as backwards:
# the robot should stand at LEAST as far back for open (where the drawer
# ends up occupying more of the space in front of the counter) as for
# close, not closer, and 0.5m left only ~0.15m of margin above the 0.35m
# physics-verified collision-free floor referenced above (vs. close's
# ~0.65m of margin at 1.0m). Re-rendered
# slurm_scripts/maniskill_open_subtask_train/render_goal_state.ipynb's
# base-standoff comparison at 0.5/0.7/1.0/1.5m: 0.5m still visibly hugs the
# counter, 1.5m exceeds the Fetch arm's reach (IK error jumps from ~0m to
# ~0.47m solving toward the same handle target), and both 0.7m and 1.0m
# render cleanly with near-zero IK error (0.000m and ~0.048m
# respectively). Set to 0.7m -- clears the "closer than close" ordering
# problem (close's own is 1.0m) while keeping essentially perfect arm
# reach, without going all the way out to close's own value.
_OPEN_SUBTASK_BASE_STANDOFF_M = 0.7


def _close_subtask_closed_handle_positions(unwrapped, env_idx):
  """FK peek: reachable points in front of the handle when closed.

  Generalized over num_envs so it can serve both
  `ManiskillCloseSubtaskTrain` (always `env_idx=[0]`) and
  `ManiskillVecEnv`'s GPU-batched training path, which must restrict this
  to only the envs actually resetting on a given step (`env_idx` a subset)
  -- every other row's qpos must come back byte-for-byte unchanged, since
  those envs are mid-episode.

  mshab has no forward-kinematics helper for this -- `goal_pos_wrt_base`
  (mshab's own `handle_world_poses`) is recomputed every step from the
  *live* articulation pose (`SequentialTaskEnv.evaluate()`), so it just
  tracks the handle as it physically moves. Once the gripper makes
  contact and starts pushing, the gripper moves together with the
  handle, so a goal built from that live position gives ~0 position
  error for the entire push, well before the drawer/door is actually
  closed -- no incentive left to finish the last bit of travel. This
  computes a fixed point near the handle's *closed* configuration instead,
  offset back toward "open" along the joint's own motion axis by
  `_CLOSE_SUBTASK_APPROACH_OFFSET_M` -- the gripper's target, a comfortable
  approach point in front of the drawer rather than flush with/inside the
  cabinet face.

  The axis direction (used for the gripper target only) is measured, not
  assumed (drawers are prismatic -- a straight axis -- but doors/fridges in
  the same task plan are revolute, where "the axis" is only well-defined
  locally): probe the handle position at target_qpos and again at a second
  nearby qpos (5% of the joint's [qmin, qmax] range further "open"), and use
  the vector between them as the local outward direction.

  The base's target parking spot ("the base should be in front of the
  drawer" -- used as the goal's `target_base_xy`) uses the *same*
  axis-tangent offset idea as the gripper target just above, with its own
  `_CLOSE_SUBTASK_BASE_STANDOFF_M` distance instead (1.0m -- see that
  constant's comment for the empirically-validated 0.35m collision-free
  minimum this was chosen relative to, and for why it's deliberately NOT
  snapped to the navmesh the way an earlier version was).

  Borrows the same trick mshab itself uses when spawning articulations
  (`SubtaskTrainEnv._apply_premade_spawns`): temporarily overwrite qpos
  for just `env_idx`'s rows, refresh the GPU kinematics cache, read the
  now-updated link pose, then restore the real (full-batch) qpos.
  Deliberately skips the `px.step()` that spawn path calls -- this is a
  kinematics-only pose query, not a physics step, and must leave
  contacts/velocities/wall-clock untouched.

  Returns `(handle_target_p, base_target_xy)`, both numpy arrays row-aligned
  with `env_idx`: shapes `(len(env_idx), 3)` and `(len(env_idx), 2)`.
  """
  articulation = unwrapped.articulation
  joint_idx = unwrapped.close_subtask.articulation_handle_active_joint_idx
  gpu_sim = unwrapped.gpu_sim_enabled
  env_idx_np = env_idx.detach().cpu().numpy()

  orig_qpos = articulation.qpos.clone()

  def _handle_pos_at(qpos_values):
    peek_qpos = orig_qpos.clone()
    peek_qpos[env_idx, joint_idx] = qpos_values
    articulation.set_qpos(peek_qpos)
    if gpu_sim:
      unwrapped.scene._gpu_apply_all()
      unwrapped.scene.px.gpu_update_articulation_kinematics()
      unwrapped.scene._gpu_fetch_all()
    pose = (unwrapped.link.pose
            * unwrapped.close_subtask.articulation_relative_handle_pos)
    # _ms_to_numpy_batched detaches to a fresh host array, so this survives
    # the next kinematics refresh (which would otherwise silently
    # invalidate a live GPU tensor view).
    return _ms_to_numpy_batched(pose.p)[env_idx_np]

  target_qpos = unwrapped.target_qpos[env_idx]
  qmax = unwrapped.qmax[env_idx]
  qmin = unwrapped.qmin[env_idx]
  probe_qpos = _torch.minimum(target_qpos + 0.05 * (qmax - qmin), qmax)

  closed_p = _handle_pos_at(target_qpos)
  probe_p = _handle_pos_at(probe_qpos)

  articulation.set_qpos(orig_qpos)
  if gpu_sim:
    unwrapped.scene._gpu_apply_all()
    unwrapped.scene.px.gpu_update_articulation_kinematics()
    unwrapped.scene._gpu_fetch_all()

  axis_dir = probe_p - closed_p
  axis_norm = np.linalg.norm(axis_dir, axis=1, keepdims=True)
  degenerate = axis_norm < 1e-6  # e.g. already at qmax; leave un-offset
  safe_norm = np.where(degenerate, 1.0, axis_norm)
  unit_axis = axis_dir / safe_norm
  gripper_offset = np.where(degenerate, 0.0, _CLOSE_SUBTASK_APPROACH_OFFSET_M)
  handle_target_p = closed_p + gripper_offset * unit_axis
  # base_target_xy: same tangent-offset idea as handle_target_p just above,
  # at _CLOSE_SUBTASK_BASE_STANDOFF_M instead -- deliberately NOT snapped to
  # the navmesh (an earlier version was; see that constant's comment for why
  # dropping the snap was necessary: at 0.1m, the offset point sits inside
  # the counter's own footprint, same as handle_target_p's own 0.05m offset
  # already does without any navmesh validation -- real collision geometry
  # keeps the robot's actual base out of the counter regardless of what the
  # goal coordinate says, exactly as it already does for the gripper).
  base_offset = np.where(degenerate, 0.0, _CLOSE_SUBTASK_BASE_STANDOFF_M)
  base_target_xy = (closed_p + base_offset * unit_axis)[:, :2]
  return handle_target_p, base_target_xy


def _close_subtask_handle_pos_wrt_base(unwrapped, closed_handle_world_p):
  """Transforms a cached fixed world-frame point into the current base frame.

  Shared by `ManiskillCloseSubtaskTrain._get_obs` and `ManiskillVecEnv`'s
  batched obs path -- both need this every step (the base can move even
  though the cached target point doesn't), but only ever call the FK peek
  above once per episode (see `reset()` in each).
  """
  base_pose_inv = unwrapped.agent.base_link.pose.inv()
  closed_handle_pose_world = _MsPose.create_from_pq(
      p=_torch.as_tensor(closed_handle_world_p,
                         device=base_pose_inv.p.device,
                         dtype=base_pose_inv.p.dtype))
  return (base_pose_inv * closed_handle_pose_world).p


def _open_subtask_open_handle_positions(unwrapped, env_idx):
  """FK peek: reachable points in front of the handle when open.

  Mirrors `_close_subtask_closed_handle_positions` exactly (same
  generalization over `num_envs`, same qpos-overwrite-and-restore FK-peek
  mechanics, same GPU kinematics refresh calls, same
  `(handle_target_p, base_target_xy)` return shape/row-alignment) -- see
  that function's docstring for the full mechanics rationale, only
  reproduced here rather than shared so neither path risks the other while
  being edited.

  Probes in the *same* direction close does -- toward qmax (more open) --
  not the mirror-image "toward qmin" an earlier version of this function
  used. That earlier version reasoned by qpos-space symmetry ("open's
  offset should mirror close's by pointing toward the opposite extreme"),
  which is wrong: for a prismatic drawer, "away from the cabinet, out into
  the room" is *always* the +qpos (more-open) direction, regardless of
  which qpos value the FK-peek is anchored at. Probing toward qmin from the
  open anchor pointed *into* the cabinet/counter body -- harmless for the
  small (0.05m) gripper-approach offset (still short of the handle's travel
  range), but catastrophic once scaled up to the 1m+ base-standoff offset:
  `base_target_xy` landed behind/inside the counter instead of out in the
  room in front of it (caught visually -- rendered goal-state screenshots
  showed the robot's base overlapping the counter, facing away from the
  handle, instead of parked in front of it).

  Reuses `_CLOSE_SUBTASK_APPROACH_OFFSET_M` (the small 0.05m gripper-
  approach offset) but uses its own `_OPEN_SUBTASK_BASE_STANDOFF_M`
  (0.5m) rather than close's `_CLOSE_SUBTASK_BASE_STANDOFF_M` -- see that
  constant's own comment for why the base-standoff distance is
  deliberately NOT shared between the two tasks (close's 1.0m reflects an
  extensive tuning history specific to close's geometry; open's 0.5m is
  the user's own choice after visually reviewing the direction-corrected
  render). The small approach offset stays shared since it was never
  flagged as wrong and there's no equivalent tuning history to decouple it
  from.

  Returns `(handle_target_p, base_target_xy)`, both numpy arrays row-aligned
  with `env_idx`: shapes `(len(env_idx), 3)` and `(len(env_idx), 2)`.
  """
  articulation = unwrapped.articulation
  joint_idx = unwrapped.open_subtask.articulation_handle_active_joint_idx
  gpu_sim = unwrapped.gpu_sim_enabled
  env_idx_np = env_idx.detach().cpu().numpy()

  orig_qpos = articulation.qpos.clone()

  def _handle_pos_at(qpos_values):
    peek_qpos = orig_qpos.clone()
    peek_qpos[env_idx, joint_idx] = qpos_values
    articulation.set_qpos(peek_qpos)
    if gpu_sim:
      unwrapped.scene._gpu_apply_all()
      unwrapped.scene.px.gpu_update_articulation_kinematics()
      unwrapped.scene._gpu_fetch_all()
    pose = (unwrapped.link.pose
            * unwrapped.open_subtask.articulation_relative_handle_pos)
    # _ms_to_numpy_batched detaches to a fresh host array, so this survives
    # the next kinematics refresh (which would otherwise silently
    # invalidate a live GPU tensor view).
    return _ms_to_numpy_batched(pose.p)[env_idx_np]

  target_qpos = unwrapped.target_qpos[env_idx]
  qmax = unwrapped.qmax[env_idx]
  qmin = unwrapped.qmin[env_idx]
  probe_qpos = _torch.minimum(target_qpos + 0.05 * (qmax - qmin), qmax)

  open_p = _handle_pos_at(target_qpos)
  probe_p = _handle_pos_at(probe_qpos)

  articulation.set_qpos(orig_qpos)
  if gpu_sim:
    unwrapped.scene._gpu_apply_all()
    unwrapped.scene.px.gpu_update_articulation_kinematics()
    unwrapped.scene._gpu_fetch_all()

  axis_dir = probe_p - open_p
  axis_norm = np.linalg.norm(axis_dir, axis=1, keepdims=True)
  degenerate = axis_norm < 1e-6  # e.g. already at qmax; leave un-offset
  safe_norm = np.where(degenerate, 1.0, axis_norm)
  unit_axis = axis_dir / safe_norm
  gripper_offset = np.where(degenerate, 0.0, _CLOSE_SUBTASK_APPROACH_OFFSET_M)
  handle_target_p = open_p + gripper_offset * unit_axis
  base_offset = np.where(degenerate, 0.0, _OPEN_SUBTASK_BASE_STANDOFF_M)
  base_target_xy = (open_p + base_offset * unit_axis)[:, :2]
  return handle_target_p, base_target_xy


def _open_subtask_handle_pos_wrt_base(unwrapped, open_handle_world_p):
  """Transforms a cached fixed world-frame point into the current base frame.

  Identical to `_close_subtask_handle_pos_wrt_base`, just named for the
  open-target point it's called with -- see that function's docstring.
  """
  base_pose_inv = unwrapped.agent.base_link.pose.inv()
  open_handle_pose_world = _MsPose.create_from_pq(
      p=_torch.as_tensor(open_handle_world_p,
                         device=base_pose_inv.p.device,
                         dtype=base_pose_inv.p.dtype))
  return (base_pose_inv * open_handle_pose_world).p


class ManiskillCloseSubtaskTrain(gym.Env):
  """gym (old-API) wrapper around mshab's ``CloseSubtaskTrain-v0``.

  State layout (10,): [tcp_pos_wrt_base(3), articulation_qpos(1),
                       gripper_aperture(1), is_grasped(1), base_xy(2),
                       heading(2)]
  Goal  layout (8,): [handle_pos_wrt_base(3), target_qpos(1),
                       gripper_target(1)=OPEN, is_grasped_target(1)=0,
                       target_base_xy(2)]

  An *asymmetric* state/goal split (10/8, same shape of asymmetry as
  ManiskillOpenCabinetDrawer's 10/6, not the earlier symmetric 6/6 this env
  used to have): ``base_xy`` (world-frame, from the agent's raw/unstripped
  qpos -- see ``gripper_aperture``'s comment below for why the stripped
  ``obs_dict['agent']['qpos']`` can't be used here) DOES have a goal
  counterpart, ``target_base_xy`` -- a fixed "park in front of the drawer"
  point, computed once per episode alongside the handle's own closed-position
  target (see ``_close_subtask_closed_handle_positions``'s
  ``base_target_xy``: a small tangent-offset from the handle's closed
  position, same idea as the gripper's own approach offset, snapped to the
  nearest real navmesh point -- see that function's docstring for why).
  Added because
  the adjacent-room spawn change (see gen_spawn_positions.py) means episodes
  now routinely require several meters of base travel before the drawer is
  even reachable; without an absolute position readout and a real target to
  compare it against, the policy/value nets have no way to tell "far from the
  handle because still crossing the room" apart from "far from the handle
  because lost/circling" -- both look identical in the purely base-relative
  tcp_pos/handle_pos dims, and a *relabeled* (hindsight) achieved goal for
  tcp_pos_wrt_base is always arm-reach bounded regardless of base travel (see
  the CRL replay's ``obs_to_goal``), so it alone can never expose the
  long-range navigation signal either.

  ``heading`` (sin/cos of the base's world yaw) is state-only, with no goal
  counterpart -- same reasoning as OpenCabinetDrawer's own heading dim (no
  natural "target heading" for this task) -- but is now *necessary*, not just
  a navigation nicety: once ``base_xy`` is a raw world-frame position, the
  observation is no longer purely base-relative, so two states with
  identical ``base_xy`` but different (unobserved) heading would respond
  differently to the same ego-centric base action -- the exact Markov
  violation OpenCabinetDrawer-v1's own heading dim exists to avoid. Both
  ``base_xy`` and ``heading`` must come *after* the goal-paired dims in the
  state ordering (see ``EpisodeReplay._obs_to_goal``'s
  ``start_index``/``end_index`` slicing), and ``heading`` (no goal
  counterpart) must come after ``base_xy`` (which has one) -- same ordering
  constraint ManiskillOpenCabinetDrawer's own heading/base_xy already follow.

  mshab expresses tcp/handle positions relative to
  the robot base (``tcp_pose_wrt_base``/``goal_pos_wrt_base`` in
  ``_get_obs_extra``), recomputed fresh every step from the *current* base
  pose. That base-relative framing already absorbs whatever the mobile base
  does for those dims specifically -- Fetch's base controller is ego-centric
  (rotates [forward_vel, turn_rate] by the *current* heading before applying
  it), so "move forward" always means "move along the robot's own local +x
  axis" regardless of world heading, and no heading dim is needed to keep
  *tcp_pos_wrt_base*/*handle_pos_wrt_base* Markov. ``base_xy``/``heading``
  are an orthogonal, world-frame addition layered on top of that, added
  specifically to carry the long-range navigation signal those base-relative
  dims cannot.

  Each of the first 8 state dims pairs with the *same physical quantity* in
  the corresponding goal dim (current value vs. target value), mirroring
  ManiskillOpenCabinetDrawer's convention (``heading``, the 9th/10th state
  dims, has no goal counterpart -- see above):

  - ``tcp_pos_wrt_base`` / ``handle_pos_wrt_base``: current tcp position vs.
    where the tcp should end up. Unlike mshab's own ``goal_pos_wrt_base``
    (which tracks the *live*, currently-moving handle position every step),
    this goal dim is the handle's position at the *closed* configuration --
    a genuinely fixed point in world space for the whole episode, computed
    once at reset via a forward-kinematics peek (see
    ``_close_subtask_closed_handle_positions``) and re-expressed in the
    current base frame every step. Using the live handle position here would
    make this goal dim collapse to ~0 error the instant the gripper makes
    contact and starts pushing (since gripper and handle then move
    together), well before the joint is actually closed -- no incentive left
    to finish the push. The fixed closed-position target keeps a real
    "distance still to go" signal throughout.
  - ``articulation_qpos`` / ``target_qpos``: current vs. target joint value
    of the door/drawer/fridge being closed. ``target_qpos`` is NOT exposed
    in mshab's observation dict (only used internally for reward/success),
    so it's read directly off the underlying env instance
    (``self._env.unwrapped.target_qpos``), the same pattern
    ManiskillOpenCabinetDrawer already uses for its own ``target_qpos``.
  - ``gripper_aperture`` / ``gripper_target=0.10``: sum of Fetch's two
    base-stripped gripper-finger qpos (indices 10, 11 of the 12-dim
    ``obs['agent']['qpos']`` -- mshab's ``_get_obs_agent`` strips the 3 base
    dims off the raw 15-dim Fetch qpos, unlike OpenCabinetDrawer-v1 which
    uses the full 15-dim qpos and reads indices 13/14 for the same fingers).
    Same "release the handle once done" target as OpenCabinetDrawer-v1.
  - ``is_grasped`` / ``grasp_target=0.0``: current grasp bool vs. "not
    grasping." Read directly from ``obs_dict['extra']['is_grasped']``,
    already computed by mshab itself during ``evaluate()`` via
    ``agent.is_grasping(handle_link, max_angle=30)`` -- simpler than
    ManiskillOpenCabinetDrawer, which has to call ``is_grasping`` itself.
  - ``base_xy`` / ``target_base_xy``: current world-frame base position vs.
    the fixed "park in front of the drawer" point from
    ``_close_subtask_closed_handle_positions``'s ``base_target_xy``
    (`_CLOSE_SUBTASK_BASE_STANDOFF_M` tangent-offset from the handle's
    closed position). Read off ``agent.base_link.pose`` -- NOT
    ``agent.robot.get_qpos()[:2]``, which is relative to a per-scene root
    anchor pose, not world origin (see ``_get_obs``'s comment for the
    empirical measurement that caught this) -- and the FK-peek,
    respectively -- see those for details. Unlike the other three
    goal-paired dims above, mshab's own reward/success genuinely doesn't
    care where the base ends up (only the joint qpos does), so this target
    is this file's own invention, not something read off the underlying
    env.

  mshab's own success condition (``SequentialTaskEnv._close_check_success``)
  also requires the end-effector/robot to return to a "rest" pose and be
  static, in addition to the joint being closed -- neither captured in this
  compact goal vector. This mirrors an existing gap: OpenCabinetDrawer-v1's
  true success also requires the handle to be near-static, which its own
  goal vector doesn't capture either. Both wrappers use the env's binary
  ``info['success']`` as `reward` regardless, so this is a pre-existing,
  accepted simplification, not a new one.

  Unlike OpenCabinetDrawer-v1's procedurally-sampled PartNet-Mobility
  cabinets, CloseSubtaskTrain-v0 requires externally-authored ``task_plans``
  and ``spawn_data_fp`` (ReplicaCAD-Rearrange assets); see
  ``_mshab_close_subtask_paths()``. ``num_envs`` need not divide the number
  of distinct scenes in the task plan because
  ``require_build_configs_repeated_equally_across_envs=False`` is passed.

  ``control_mode='pd_joint_delta_pos'`` -- matching mshab's own baseline
  convention, NOT this file's other maniskill_* wrappers' Cartesian-IK
  ``pd_ee_delta_pos`` choice. ``pd_ee_delta_pos`` was tried first for
  consistency with those wrappers and worked fine single-env (num_envs=1),
  but empirically broke under GPU-batched simulation (num_envs>1, i.e.
  exactly what `ManiskillVecEnv`/`--maniskill_native_vec` needs) with
  `RuntimeError: shape mismatch: value tensor of shape [E, 11] cannot be
  broadcast to indexing result of shape [E, 7]` inside ManiSkill's
  `pd_ee_pose.py` IK solve -- apparently an incompatibility between this
  fork's GPU-batched IK jacobian computation and CloseSubtaskTrainEnv's
  per-subtask merged articulation setup. `pd_joint_delta_pos` (raw 7-dim
  arm joint-space delta instead of 3-dim Cartesian IK) verified working
  under both num_envs=1 and num_envs=8.
  """

  STATE_DIM = 10
  GOAL_DIM = 8

  _GRIPPER_TARGET = np.array([0.10], dtype=np.float32)   # fully open
  _GRASP_TARGET = np.array([0.0], dtype=np.float32)      # not grasping

  def __init__(self, fixed_start_end=None, render_mode=None):
    super().__init__()
    _require_maniskill('maniskill_close_subtask_train')
    _require_mshab('maniskill_close_subtask_train')
    del fixed_start_end  # CloseSubtaskTrain-v0 randomizes scene/spawn itself.
    task_plan_fp, spawn_data_fp = _mshab_close_subtask_paths()
    plan_data = _mshab_plan_data_from_file(task_plan_fp)
    self._env = _gymnasium.make(
        'CloseSubtaskTrain-v0',
        # Overrides mshab's own `@register_env("CloseSubtaskTrain-v0",
        # max_episode_steps=400)` -- ManiSkill's `TimeLimitWrapper` (see
        # mani_skill/utils/registration.py) specifically inspects `gym.make`'s
        # own call-frame locals for a user-supplied `max_episode_steps` and,
        # if present, uses that instead of the value baked in at
        # registration. Must match `load()`'s `max_episode_steps` for this
        # env_name (see that function's docstring on gymnasium's
        # already-fired-TimeLimit undefined-behavior caveat) and
        # `_MANISKILL_MAX_EPISODE_STEPS['maniskill_close_subtask_train']`.
        max_episode_steps=600,
        obs_mode='state_dict',
        control_mode='pd_joint_delta_pos',
        reward_mode='normalized_dense',
        render_mode=render_mode,
        robot_uids='fetch',
        num_envs=1,
        task_plans=plan_data.plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False,
        # See `_maniskill_extra_env_kwargs`'s matching comment: doubled GPU
        # PhysX buffer sizes vs. mshab's static defaults, kept consistent
        # here even though num_envs=1 makes an overflow far less likely.
        #
        # NOTE: collision_stack_size is NOT configurable here -- the pinned
        # mani_skill==3.0.0b18 / sapien==3.0.0b1 (see close_subtask_train
        # .slurm's mshab_rl env) have no such field/kwarg anywhere in their
        # Python API (confirmed against both wheels directly); passing it
        # raises `dacite.exceptions.UnexpectedDataError` (job 9189763).
        sim_config=dict(
            gpu_memory_config=dict(
                temp_buffer_capacity=2**25,
                max_rigid_contact_count=2**24,
                found_lost_pairs_capacity=2**26,
                max_rigid_patch_count=2**22)))
    pos_bound = 2.0  # meters; generous margin, base-relative workspace
    pos_low = np.full(3, -pos_bound, dtype=np.float32)
    pos_high = np.full(3, pos_bound, dtype=np.float32)
    # Shared range for current/target articulation qpos: covers both
    # prismatic drawers (~[0, 0.5]m) and revolute doors/fridges (~[0, 2.5]
    # rad) since the task plan may mix articulation types. Empirical guess,
    # same caveat as ManiskillOpenCabinetDrawer's drawer_qpos bound -- pad
    # generously and tighten later if training data shows it's too loose.
    qpos_low = np.full(1, -0.1, dtype=np.float32)
    qpos_high = np.full(1, 3.0, dtype=np.float32)
    grip_low = np.zeros(1, dtype=np.float32)
    grip_high = np.full(1, 0.10, dtype=np.float32)
    grasp_low = np.zeros(1, dtype=np.float32)   # is_grasped is 0.0/1.0
    grasp_high = np.ones(1, dtype=np.float32)
    # base_xy / target_base_xy: world-frame robot base position and its
    # fixed "park in front of the drawer" target (see
    # `_close_subtask_closed_handle_positions`'s `base_target_xy`) -- a
    # *separate, larger* margin than `pos_bound` above. `pos_bound` covers
    # tcp_pos/handle_pos, which are base-relative and so stay arm-reach
    # bounded (~1-2m) regardless of how far the robot has traveled; base_xy
    # is a world-frame absolute position, which after the adjacent-room
    # spawn change can differ from the target by the same multi-meter,
    # room-crossing distances that change introduced (geodesic dist up to
    # ~10.3m, see verify_geodesic_shift.slurm's stats). Reusing `pos_bound`
    # here would immediately recreate, for this new dimension, the exact
    # goal_low/goal_high staleness issue already flagged for
    # --uniform_sampling on the old (arm-bounded-only) goal vector.
    world_pos_bound = 15.0
    base_xy_low = np.full(2, -world_pos_bound, dtype=np.float32)
    base_xy_high = np.full(2, world_pos_bound, dtype=np.float32)
    heading_low = np.full(2, -1.0, dtype=np.float32)   # (sin, cos) range
    heading_high = np.full(2, 1.0, dtype=np.float32)
    state_low = np.concatenate(
        [pos_low, qpos_low, grip_low, grasp_low, base_xy_low, heading_low])
    state_high = np.concatenate(
        [pos_high, qpos_high, grip_high, grasp_high, base_xy_high,
         heading_high])
    goal_low = np.concatenate(
        [pos_low, qpos_low, grip_low, grasp_low, base_xy_low])
    goal_high = np.concatenate(
        [pos_high, qpos_high, grip_high, grasp_high, base_xy_high])
    self.observation_space = gym.spaces.Box(
        low=np.concatenate([state_low, goal_low]),
        high=np.concatenate([state_high, goal_high]),
        dtype=np.float32)
    action_dim = int(self._env.action_space.shape[-1])
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

  def reset(self):
    obs_dict, _info = self._env.reset()
    # Fixed once per episode -- see `_close_subtask_closed_handle_positions`'s
    # docstring for why this can't just be read off the env every step the
    # way the rest of `_get_obs` reads its fields.
    env_idx = _torch.zeros(1, dtype=_torch.long,
                           device=self._env.unwrapped.device)
    handle_target_p, base_target_xy = _close_subtask_closed_handle_positions(
        self._env.unwrapped, env_idx)
    self._closed_handle_world_p = handle_target_p[0]
    self._base_target_xy = base_target_xy[0]
    return self._get_obs(obs_dict)

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    obs_dict, _reward, _terminated, _truncated, info = self._env.step(action)
    reward = float(_ms_to_bool_scalar(info.get('success', False)))
    # Drawer-only success signal, deliberately narrower than `reward`
    # above (mshab's `info['success']` also requires the arm to return to
    # its rest pose and the robot to be static -- see
    # `contrastive.utils.DrawerClosedSuccessObserver`, which reads this
    # instead of `reward` for the `success`/`success_1000` eval metrics).
    # Same formula as mshab's own `articulation_closed` check
    # (sequential_task.py's `_close_check_success`): `target_qpos` is
    # already `(qmax - qmin) * joint_qpos_close_thresh_frac + qmin`.
    unwrapped = self._env.unwrapped
    joint_idx = unwrapped.close_subtask.articulation_handle_active_joint_idx
    drawer_closed = _ms_to_bool_scalar(
        unwrapped.articulation.qpos[:, joint_idx] < unwrapped.target_qpos)
    return (self._get_obs(obs_dict), reward, False,
            {'drawer_closed': drawer_closed})

  def close(self):
    self._env.close()

  def render(self):
    """Returns one (H, W, 3) uint8 RGB frame. Requires render_mode='rgb_array'."""
    return _ms_render_frame(self._env)

  def _get_obs(self, obs_dict):
    tcp_pos = _ms_to_numpy(obs_dict['extra']['tcp_pose_wrt_base'])[:3]
    unwrapped = self._env.unwrapped
    handle_pos = _ms_to_numpy(_close_subtask_handle_pos_wrt_base(
        unwrapped, self._closed_handle_world_p[None]))[:3]
    is_grasped = _ms_to_numpy(obs_dict['extra']['is_grasped'])
    articulation_qpos = _ms_to_numpy(unwrapped.articulation.qpos[
        :, unwrapped.close_subtask.articulation_handle_active_joint_idx])
    target_qpos = _ms_to_numpy(unwrapped.target_qpos)
    qpos = _ms_to_numpy(obs_dict['agent']['qpos'])
    gripper_aperture = np.array([qpos[10] + qpos[11]], dtype=np.float32)
    # World-frame base (x, y) and heading -- NOT `agent.robot.get_qpos()[:3]`
    # (OpenCabinetDrawer-v1's own convention, which this originally copied):
    # unlike that env's procedural single-cabinet scene, this env's Fetch
    # articulation has a non-trivial root anchor pose per ReplicaCAD scene
    # (`agent.robot.root_pose`, e.g. measured [1.80, 0.70, 0.02] for one
    # scene) -- qpos[0:2] is a *delta from that anchor*, not a world
    # position, so reading it directly is off by the whole per-scene anchor
    # offset (confirmed empirically: qpos[:2]=[0.0007, 0.0002] while the
    # base link's actual world position was [1.80, 0.70]). `base_link.pose`
    # is always the fully-composed world pose regardless of how the root
    # anchor is set up, so read world position/heading from there instead.
    base_pose = unwrapped.agent.base_link.pose
    base_xy = _ms_to_numpy(base_pose.p)[:2]
    qw, qx, qy, qz = _ms_to_numpy(base_pose.q)
    base_theta = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    heading = np.array([np.sin(base_theta), np.cos(base_theta)],
                       dtype=np.float32)

    state = np.concatenate([
        tcp_pos, articulation_qpos, gripper_aperture, is_grasped,
        base_xy, heading]).astype(np.float32)
    goal = np.concatenate([
        handle_pos, target_qpos, self._GRIPPER_TARGET,
        self._GRASP_TARGET, self._base_target_xy]).astype(np.float32)
    return np.concatenate([state, goal]).astype(np.float32)


class ManiskillOpenSubtaskTrain(gym.Env):
  """gym (old-API) wrapper around mshab's ``OpenSubtaskTrain-v0``.

  Sibling of ``ManiskillCloseSubtaskTrain`` with the goal reversed: the
  robot spawns the same way (same scene, same "adjacent room" robot
  spawn poses -- see ``_mshab_open_subtask_paths``/
  ``scripts/build_open_spawn_data.py``), but the drawer starts **closed**
  and success means **opening** it. Mirrors that class end to end -- see
  its docstring for the full state/goal design rationale (asymmetric
  10/8 split, why ``base_xy``/``heading`` are needed, why each dim is
  base-relative vs. world-frame, etc.); only the differences are called
  out below.

  State layout (10,): [tcp_pos_wrt_base(3), articulation_qpos(1),
                       gripper_aperture(1), is_grasped(1), base_xy(2),
                       heading(2)]
  Goal  layout (8,): [handle_pos_wrt_base(3), target_qpos(1),
                       gripper_target(1)=OPEN, is_grasped_target(1)=0,
                       target_base_xy(2)]
  (Identical layout to close -- only what each goal dim targets differs,
  see below.)

  Differences from ``ManiskillCloseSubtaskTrain``:
  - ``gymnasium.make('OpenSubtaskTrain-v0', ...)`` instead of
    ``'CloseSubtaskTrain-v0'``; ``task_plans``/``scene_builder_cls`` come
    from ``_mshab_open_subtask_paths()``.
  - Success direction is flipped: mshab's own
    ``SequentialTaskEnv._open_check_success`` uses ``qpos > target_qpos``
    (vs. close's ``qpos < target_qpos``) -- ``step()`` here mirrors that
    with its own ``drawer_open`` info flag (vs. close's ``drawer_closed``).
  - ``unwrapped.close_subtask`` -> ``unwrapped.open_subtask`` (same
    attribute shape on the underlying env, just this env's own
    ``task_plan[0]``); ``target_qpos``/``qmax``/``qmin`` are available
    identically on both envs (set in ``_initialize_episode``), so no other
    plumbing changes.
  - The goal vector is built from the drawer's **open** configuration
    instead of its closed one: ``handle_pos`` is the handle's position at
    ``target_qpos`` (here the *open* target, i.e. ``qmax*0.9`` for
    kitchen_counter, offset slightly further **toward open** -- i.e. away
    from the cabinet, same direction close's own offset uses -- by the same
    approach-offset idea -- see ``_open_subtask_open_handle_positions``),
    NOT the closed-handle position close's goal uses. ``target_qpos``
    itself is the open target value (still just
    ``unwrapped.target_qpos``, same read close already does).

    ``gripper_target``/``grasp_target`` -- grasp-target flip, deliberately
    DIFFERENT from close, not a copy/paste oversight: close's goal targets
    "released, not grasping" (``_GRIPPER_TARGET=[0.10]`` fully open /
    ``_GRASP_TARGET=[0.0]``) because a drawer can be pushed shut via bare
    contact, no grasp required, so a rollout policy conditioned on "goal:
    released" for the whole episode can still find a working close strategy.
    Opening has no such contact-only equivalent -- pulling a drawer out
    physically requires the gripper to grip the handle first. Since the
    per-step PPO reward is the CRL critic's ``φ(s,a)·ψ(g)`` dot product
    against this *fixed* goal used during rollout (not the sparse env
    reward -- see ``contrastive/ppo_learner.py``'s ``make_reward_fn``), a
    goal that targets "released" the entire episode makes any state where
    the gripper actually grips the handle score as moving *away* from the
    goal -- a structural disincentive against the one behavior opening
    requires. So this class's own ``_GRIPPER_TARGET=[0.0]``/
    ``_GRASP_TARGET=[1.0]`` instead target "closed, gripping" -- diagnosed
    and changed after a real training run (jobs 9579370/9579375, ~50M
    steps) showed zero learning progress (`Success`/`Min Dist` flat) despite
    the goal-position (handle/base) fix above already being in place.
    ``target_base_xy`` is the same "park in front of the drawer" idea as
    close, just computed from the open-target handle point via
    ``_open_subtask_open_handle_positions`` instead of the closed one.

  Everything else (FK-peek-via-qpos-overwrite-and-restore mechanics, GPU
  kinematics refresh calls, ``control_mode='pd_joint_delta_pos'`` -- same
  GPU-batched-IK bug applies here, an mshab/ManiSkill-level limitation
  independent of open vs. close -- doubled GPU buffer sim_config, obs-space
  bounds) copies verbatim from ``ManiskillCloseSubtaskTrain``.
  """

  STATE_DIM = 10
  GOAL_DIM = 8

  # NOT inherited from close (deliberately) -- see this class's docstring's
  # "grasp-target flip" note. Opening requires the gripper to actually grip
  # the handle to pull it; there's no ungrasped-contact-push equivalent the
  # way there is for closing.
  _GRIPPER_TARGET = np.array([0.0], dtype=np.float32)   # closed (gripping the handle)
  _GRASP_TARGET = np.array([1.0], dtype=np.float32)     # grasping

  def __init__(self, fixed_start_end=None, render_mode=None):
    super().__init__()
    _require_maniskill('maniskill_open_subtask_train')
    _require_mshab('maniskill_open_subtask_train')
    del fixed_start_end  # OpenSubtaskTrain-v0 randomizes scene/spawn itself.
    task_plan_fp, spawn_data_fp = _mshab_open_subtask_paths()
    plan_data = _mshab_plan_data_from_file(task_plan_fp)
    plan_data = _filter_open_subtask_plan_data(plan_data)
    self._env = _gymnasium.make(
        'OpenSubtaskTrain-v0',
        # See ManiskillCloseSubtaskTrain.__init__'s matching comment: this
        # overrides mshab's own registered TimeLimit via `gym.make`'s
        # call-frame-local `max_episode_steps` detection. Must match
        # `load()`'s `max_episode_steps` for this env_name and
        # `_MANISKILL_MAX_EPISODE_STEPS['maniskill_open_subtask_train']`.
        max_episode_steps=600,
        obs_mode='state_dict',
        control_mode='pd_joint_delta_pos',
        reward_mode='normalized_dense',
        render_mode=render_mode,
        robot_uids='fetch',
        num_envs=1,
        task_plans=plan_data.plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False,
        # See `_maniskill_extra_env_kwargs`'s matching comment: doubled GPU
        # PhysX buffer sizes vs. mshab's static defaults, kept consistent
        # here even though num_envs=1 makes an overflow far less likely.
        sim_config=dict(
            gpu_memory_config=dict(
                temp_buffer_capacity=2**25,
                max_rigid_contact_count=2**24,
                found_lost_pairs_capacity=2**26,
                max_rigid_patch_count=2**22)))
    pos_bound = 2.0  # meters; generous margin, base-relative workspace
    pos_low = np.full(3, -pos_bound, dtype=np.float32)
    pos_high = np.full(3, pos_bound, dtype=np.float32)
    qpos_low = np.full(1, -0.1, dtype=np.float32)
    qpos_high = np.full(1, 3.0, dtype=np.float32)
    grip_low = np.zeros(1, dtype=np.float32)
    grip_high = np.full(1, 0.10, dtype=np.float32)
    grasp_low = np.zeros(1, dtype=np.float32)   # is_grasped is 0.0/1.0
    grasp_high = np.ones(1, dtype=np.float32)
    world_pos_bound = 15.0
    base_xy_low = np.full(2, -world_pos_bound, dtype=np.float32)
    base_xy_high = np.full(2, world_pos_bound, dtype=np.float32)
    heading_low = np.full(2, -1.0, dtype=np.float32)   # (sin, cos) range
    heading_high = np.full(2, 1.0, dtype=np.float32)
    state_low = np.concatenate(
        [pos_low, qpos_low, grip_low, grasp_low, base_xy_low, heading_low])
    state_high = np.concatenate(
        [pos_high, qpos_high, grip_high, grasp_high, base_xy_high,
         heading_high])
    goal_low = np.concatenate(
        [pos_low, qpos_low, grip_low, grasp_low, base_xy_low])
    goal_high = np.concatenate(
        [pos_high, qpos_high, grip_high, grasp_high, base_xy_high])
    self.observation_space = gym.spaces.Box(
        low=np.concatenate([state_low, goal_low]),
        high=np.concatenate([state_high, goal_high]),
        dtype=np.float32)
    action_dim = int(self._env.action_space.shape[-1])
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

  def reset(self):
    obs_dict, _info = self._env.reset()
    # Fixed once per episode -- see `_open_subtask_open_handle_positions`'s
    # docstring for why this can't just be read off the env every step the
    # way the rest of `_get_obs` reads its fields.
    env_idx = _torch.zeros(1, dtype=_torch.long,
                           device=self._env.unwrapped.device)
    handle_target_p, base_target_xy = _open_subtask_open_handle_positions(
        self._env.unwrapped, env_idx)
    self._open_handle_world_p = handle_target_p[0]
    self._base_target_xy = base_target_xy[0]
    return self._get_obs(obs_dict)

  def step(self, action):
    action = np.asarray(action, dtype=np.float32)
    obs_dict, _reward, _terminated, _truncated, info = self._env.step(action)
    reward = float(_ms_to_bool_scalar(info.get('success', False)))
    # Drawer-only success signal, deliberately narrower than `reward`
    # above (mshab's `info['success']` also requires the arm to return to
    # its rest pose and the robot to be static -- see
    # `contrastive.utils.DrawerOpenSuccessObserver`, which reads this
    # instead of `reward` for the `success`/`success_1000` eval metrics).
    # Same formula as mshab's own `articulation_open` check
    # (sequential_task.py's `_open_check_success`): `target_qpos` is
    # already `(qmax - qmin) * (1 - joint_qpos_open_thresh_frac) + qmin`.
    unwrapped = self._env.unwrapped
    joint_idx = unwrapped.open_subtask.articulation_handle_active_joint_idx
    drawer_open = _ms_to_bool_scalar(
        unwrapped.articulation.qpos[:, joint_idx] > unwrapped.target_qpos)
    return (self._get_obs(obs_dict), reward, False,
            {'drawer_open': drawer_open})

  def close(self):
    self._env.close()

  def render(self):
    """Returns one (H, W, 3) uint8 RGB frame. Requires render_mode='rgb_array'."""
    return _ms_render_frame(self._env)

  def _get_obs(self, obs_dict):
    tcp_pos = _ms_to_numpy(obs_dict['extra']['tcp_pose_wrt_base'])[:3]
    unwrapped = self._env.unwrapped
    handle_pos = _ms_to_numpy(_open_subtask_handle_pos_wrt_base(
        unwrapped, self._open_handle_world_p[None]))[:3]
    is_grasped = _ms_to_numpy(obs_dict['extra']['is_grasped'])
    articulation_qpos = _ms_to_numpy(unwrapped.articulation.qpos[
        :, unwrapped.open_subtask.articulation_handle_active_joint_idx])
    target_qpos = _ms_to_numpy(unwrapped.target_qpos)
    qpos = _ms_to_numpy(obs_dict['agent']['qpos'])
    gripper_aperture = np.array([qpos[10] + qpos[11]], dtype=np.float32)
    # World-frame base (x, y) and heading -- see
    # ManiskillCloseSubtaskTrain._get_obs's matching comment for why
    # `agent.robot.get_qpos()[:3]` can't be used here (per-scene root
    # anchor offset).
    base_pose = unwrapped.agent.base_link.pose
    base_xy = _ms_to_numpy(base_pose.p)[:2]
    qw, qx, qy, qz = _ms_to_numpy(base_pose.q)
    base_theta = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    heading = np.array([np.sin(base_theta), np.cos(base_theta)],
                       dtype=np.float32)

    state = np.concatenate([
        tcp_pos, articulation_qpos, gripper_aperture, is_grasped,
        base_xy, heading]).astype(np.float32)
    goal = np.concatenate([
        handle_pos, target_qpos, self._GRIPPER_TARGET,
        self._GRASP_TARGET, self._base_target_xy]).astype(np.float32)
    return np.concatenate([state, goal]).astype(np.float32)


# ---------------------------------------------------------------------------
# Batched (num_envs=E, single GPU-simulated scene) ManiSkill vec env.
#
# The classes above each open their own single-env (`num_envs=1`) ManiSkill
# scene; `ppo_learner.VecEnv` then instantiates E of them and steps them one
# at a time in a Python loop. ManiSkill natively supports simulating all E
# copies in one SAPIEN scene on the GPU (`num_envs=E`), which is what
# `ManiskillVecEnv` below uses instead -- same rollouts, same learning
# dynamics, just collected with far fewer host<->device round trips.
#
# These `_*_obs_batched` functions duplicate (rather than share) the math in
# the `_get_obs` methods above -- same formulas, kept batch-shaped (E, dim)
# instead of squeezed to (dim,). Left as separate standalone code, not a
# shared helper the single-env classes also call through, to avoid touching
# the already-working single-env path while adding this.
# ---------------------------------------------------------------------------
def _ms_to_numpy_batched(x):
  """Like `_ms_to_numpy` but keeps the leading (E, ...) batch dim."""
  if hasattr(x, 'detach'):
    x = x.detach().cpu().numpy()
  x = np.asarray(x, dtype=np.float32)
  return x.reshape(x.shape[0], -1)


def _pushcube_obs_batched(obs_dict):
  tcp_pos = _ms_to_numpy_batched(obs_dict['extra']['tcp_pose'])[:, :3]
  obj_pos = _ms_to_numpy_batched(obs_dict['extra']['obj_pose'])[:, :3]
  goal_pos = _ms_to_numpy_batched(obs_dict['extra']['goal_pos'])
  tcp_target_pos = obj_pos + ManiskillPushCube._PUSH_OFFSET[None, :]
  state = np.concatenate([tcp_pos, obj_pos], axis=1)
  goal = np.concatenate([tcp_target_pos, goal_pos], axis=1)
  return np.concatenate([state, goal], axis=1).astype(np.float32)


def _pickcube_obs_batched(obs_dict):
  tcp_pos = _ms_to_numpy_batched(obs_dict['extra']['tcp_pose'])[:, :3]
  obj_pos = _ms_to_numpy_batched(obs_dict['extra']['obj_pose'])[:, :3]
  goal_pos = _ms_to_numpy_batched(obs_dict['extra']['goal_pos'])
  qpos = _ms_to_numpy_batched(obs_dict['agent']['qpos'])
  gripper_aperture = (qpos[:, 7] + qpos[:, 8])[:, None]
  state = np.concatenate([tcp_pos, gripper_aperture, obj_pos], axis=1)
  grip_target = np.broadcast_to(
      ManiskillPickCube._GRIPPER_TARGET, (goal_pos.shape[0], 1))
  goal = np.concatenate([goal_pos, grip_target, goal_pos], axis=1)
  return np.concatenate([state, goal], axis=1).astype(np.float32)


def _open_cabinet_drawer_obs_batched(obs_dict, target_qpos, is_grasping):
  tcp_pos = _ms_to_numpy_batched(obs_dict['extra']['tcp_pose'])[:, :3]
  drawer_qpos = _ms_to_numpy_batched(obs_dict['extra']['target_link_qpos'])
  handle_target_pos = _ms_to_numpy_batched(
      obs_dict['extra']['target_handle_pos'])
  drawer_target_qpos = _ms_to_numpy_batched(target_qpos)
  qpos = _ms_to_numpy_batched(obs_dict['agent']['qpos'])
  gripper_aperture = (qpos[:, 13] + qpos[:, 14])[:, None]
  is_grasping = _ms_to_numpy_batched(is_grasping)
  base_theta = qpos[:, 2]
  heading = np.stack([np.sin(base_theta), np.cos(base_theta)], axis=1)
  base_xy = qpos[:, :2]
  state = np.concatenate(
      [tcp_pos, drawer_qpos, gripper_aperture, is_grasping, heading,
       base_xy], axis=1)
  grip_target = np.broadcast_to(
      ManiskillOpenCabinetDrawer._GRIPPER_TARGET, (handle_target_pos.shape[0], 1))
  grasp_target = np.broadcast_to(
      ManiskillOpenCabinetDrawer._GRASP_TARGET, (handle_target_pos.shape[0], 1))
  goal = np.concatenate(
      [handle_target_pos, drawer_target_qpos, grip_target, grasp_target],
      axis=1)
  return np.concatenate([state, goal], axis=1).astype(np.float32)


def _close_cabinet_drawer_obs_batched(obs_dict, target_qpos, is_grasping):
  """Mirrors `_open_cabinet_drawer_obs_batched` verbatim (same obs_dict
  fields/layout -- CloseCabinetDrawer-v1 only overrides target_qpos/
  evaluate()/the initial cabinet qpos, not any observation field), just
  reading `ManiskillCloseCabinetDrawer`'s own goal-target constants.
  """
  tcp_pos = _ms_to_numpy_batched(obs_dict['extra']['tcp_pose'])[:, :3]
  drawer_qpos = _ms_to_numpy_batched(obs_dict['extra']['target_link_qpos'])
  handle_target_pos = _ms_to_numpy_batched(
      obs_dict['extra']['target_handle_pos'])
  drawer_target_qpos = _ms_to_numpy_batched(target_qpos)
  qpos = _ms_to_numpy_batched(obs_dict['agent']['qpos'])
  gripper_aperture = (qpos[:, 13] + qpos[:, 14])[:, None]
  is_grasping = _ms_to_numpy_batched(is_grasping)
  base_theta = qpos[:, 2]
  heading = np.stack([np.sin(base_theta), np.cos(base_theta)], axis=1)
  base_xy = qpos[:, :2]
  state = np.concatenate(
      [tcp_pos, drawer_qpos, gripper_aperture, is_grasping, heading,
       base_xy], axis=1)
  grip_target = np.broadcast_to(
      ManiskillCloseCabinetDrawer._GRIPPER_TARGET, (handle_target_pos.shape[0], 1))
  grasp_target = np.broadcast_to(
      ManiskillCloseCabinetDrawer._GRASP_TARGET, (handle_target_pos.shape[0], 1))
  goal = np.concatenate(
      [handle_target_pos, drawer_target_qpos, grip_target, grasp_target],
      axis=1)
  return np.concatenate([state, goal], axis=1).astype(np.float32)


def _close_subtask_obs_batched(obs_dict, articulation_qpos, target_qpos,
                               handle_pos_wrt_base, base_link_pose,
                               target_base_xy):
  tcp_pos = _ms_to_numpy_batched(obs_dict['extra']['tcp_pose_wrt_base'])[:, :3]
  # Fixed closed-handle position (see `_close_subtask_closed_handle_positions`),
  # passed in already base-relative -- NOT mshab's own live-tracking
  # `obs_dict['extra']['goal_pos_wrt_base']`.
  handle_pos = _ms_to_numpy_batched(handle_pos_wrt_base)
  is_grasped = _ms_to_numpy_batched(obs_dict['extra']['is_grasped'])
  articulation_qpos = _ms_to_numpy_batched(articulation_qpos)
  target_qpos = _ms_to_numpy_batched(target_qpos)
  qpos = _ms_to_numpy_batched(obs_dict['agent']['qpos'])
  gripper_aperture = (qpos[:, 10] + qpos[:, 11])[:, None]
  # World-frame base (x, y) and heading -- NOT `agent.robot.get_qpos()[:,:3]`
  # (OpenCabinetDrawer-v1's own convention, which this originally copied):
  # this env's Fetch articulation has a non-trivial root anchor pose per
  # ReplicaCAD scene (`agent.robot.root_pose`), so qpos[:, 0:2] is a *delta
  # from that anchor*, not a world position -- see the single-env
  # counterpart's comment for the empirical measurement that caught this.
  # `base_link_pose` (`agent.base_link.pose`, passed in by the caller) is
  # always the fully-composed world pose regardless of the root anchor.
  base_xy = _ms_to_numpy_batched(base_link_pose.p)[:, :2]
  qwxyz = _ms_to_numpy_batched(base_link_pose.q)
  qw, qx, qy, qz = qwxyz[:, 0], qwxyz[:, 1], qwxyz[:, 2], qwxyz[:, 3]
  base_theta = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
  heading = np.stack([np.sin(base_theta), np.cos(base_theta)], axis=1)
  target_base_xy = _ms_to_numpy_batched(target_base_xy)[:, :2]
  state = np.concatenate(
      [tcp_pos, articulation_qpos, gripper_aperture, is_grasped, base_xy,
       heading], axis=1)
  grip_target = np.broadcast_to(
      ManiskillCloseSubtaskTrain._GRIPPER_TARGET, (handle_pos.shape[0], 1))
  grasp_target = np.broadcast_to(
      ManiskillCloseSubtaskTrain._GRASP_TARGET, (handle_pos.shape[0], 1))
  goal = np.concatenate(
      [handle_pos, target_qpos, grip_target, grasp_target, target_base_xy],
      axis=1)
  return np.concatenate([state, goal], axis=1).astype(np.float32)


def _open_subtask_obs_batched(obs_dict, articulation_qpos, target_qpos,
                              handle_pos_wrt_base, base_link_pose,
                              target_base_xy):
  """Mirrors `_close_subtask_obs_batched` verbatim (same signature shape),
  just sourced from `_open_subtask_open_handle_positions`'s outputs via the
  caller (`ManiskillVecEnv._raw_obs`)."""
  tcp_pos = _ms_to_numpy_batched(obs_dict['extra']['tcp_pose_wrt_base'])[:, :3]
  # Fixed open-handle position (see `_open_subtask_open_handle_positions`),
  # passed in already base-relative -- NOT mshab's own live-tracking
  # `obs_dict['extra']['goal_pos_wrt_base']`.
  handle_pos = _ms_to_numpy_batched(handle_pos_wrt_base)
  is_grasped = _ms_to_numpy_batched(obs_dict['extra']['is_grasped'])
  articulation_qpos = _ms_to_numpy_batched(articulation_qpos)
  target_qpos = _ms_to_numpy_batched(target_qpos)
  qpos = _ms_to_numpy_batched(obs_dict['agent']['qpos'])
  gripper_aperture = (qpos[:, 10] + qpos[:, 11])[:, None]
  # World-frame base (x, y) and heading -- see `_close_subtask_obs_batched`'s
  # matching comment for why `agent.robot.get_qpos()[:, :3]` can't be used
  # here (per-scene root anchor offset).
  base_xy = _ms_to_numpy_batched(base_link_pose.p)[:, :2]
  qwxyz = _ms_to_numpy_batched(base_link_pose.q)
  qw, qx, qy, qz = qwxyz[:, 0], qwxyz[:, 1], qwxyz[:, 2], qwxyz[:, 3]
  base_theta = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
  heading = np.stack([np.sin(base_theta), np.cos(base_theta)], axis=1)
  target_base_xy = _ms_to_numpy_batched(target_base_xy)[:, :2]
  state = np.concatenate(
      [tcp_pos, articulation_qpos, gripper_aperture, is_grasped, base_xy,
       heading], axis=1)
  grip_target = np.broadcast_to(
      ManiskillOpenSubtaskTrain._GRIPPER_TARGET, (handle_pos.shape[0], 1))
  grasp_target = np.broadcast_to(
      ManiskillOpenSubtaskTrain._GRASP_TARGET, (handle_pos.shape[0], 1))
  goal = np.concatenate(
      [handle_pos, target_qpos, grip_target, grasp_target, target_base_xy],
      axis=1)
  return np.concatenate([state, goal], axis=1).astype(np.float32)


_MANISKILL_TASK_IDS = {
    'maniskill_pushcube': 'PushCube-v1',
    'maniskill_pickcube': 'PickCube-v1',
    'maniskill_open_cabinet_drawer': 'OpenCabinetDrawer-v1',
    'maniskill_close_cabinet_drawer': 'CloseCabinetDrawer-v1',
    'maniskill_close_subtask_train': 'CloseSubtaskTrain-v0',
    'maniskill_open_subtask_train': 'OpenSubtaskTrain-v0',
}

# ManiSkill's own registered @register_env(..., max_episode_steps=...);
# must match env_utils.load()'s max_episode_steps for these env_names.
# maniskill_close_subtask_train's 600 is NOT the value baked into mshab's own
# `@register_env("CloseSubtaskTrain-v0", max_episode_steps=400)` -- it's
# overridden at `gymnasium.make()` call time (both
# `ManiskillCloseSubtaskTrain.__init__` and `_maniskill_extra_env_kwargs`
# pass `max_episode_steps=600` explicitly, which ManiSkill's
# `TimeLimitWrapper` specifically detects and uses instead of the registered
# default -- see those call sites' comments).
_MANISKILL_MAX_EPISODE_STEPS = {
    'maniskill_pushcube': 50,
    'maniskill_pickcube': 50,
    'maniskill_open_cabinet_drawer': 100,
    # CloseCabinetDrawer-v1's own @register_env(..., max_episode_steps=100)
    # (registered above in this file) -- same episode length as open, same
    # underlying task/asset.
    'maniskill_close_cabinet_drawer': 100,
    'maniskill_close_subtask_train': 600,
    # Same 600 override as close (see that entry's comment) -- the
    # adjacent-room navigation distance is identical (same scenes, same
    # robot spawn poses, only the drawer's start/target direction differs),
    # so the same episode-length reasoning applies. mshab's own registered
    # default for OpenSubtaskTrain-v0 is also 200 (overridden via
    # `max_episode_steps=` at `gymnasium.make()` call time, same mechanism
    # as close -- see `ManiskillOpenSubtaskTrain.__init__`).
    'maniskill_open_subtask_train': 600,
}

# Mirrors each wrapper's own STATE_DIM class attribute -- kept as a plain
# dict (not read via getattr on a constructed instance) so
# `maniskill_static_obs_info` below never has to build a SAPIEN env.
_MANISKILL_STATE_DIM = {
    'maniskill_pushcube': ManiskillPushCube.STATE_DIM,
    'maniskill_pickcube': ManiskillPickCube.STATE_DIM,
    'maniskill_open_cabinet_drawer': ManiskillOpenCabinetDrawer.STATE_DIM,
    'maniskill_close_cabinet_drawer': ManiskillCloseCabinetDrawer.STATE_DIM,
    'maniskill_close_subtask_train': ManiskillCloseSubtaskTrain.STATE_DIM,
    'maniskill_open_subtask_train': ManiskillOpenSubtaskTrain.STATE_DIM,
}


def _maniskill_extra_env_kwargs(env_name: str):
  """Extra per-task kwargs for `ManiskillVecEnv`'s batched `gymnasium.make`.

  Empty for the self-contained procedural tasks (pushcube/pickcube/
  open_cabinet_drawer); CloseSubtaskTrain-v0/OpenSubtaskTrain-v0 need their
  external task-plan/spawn-data assets wired in, same as
  `ManiskillCloseSubtaskTrain.__init__`/`ManiskillOpenSubtaskTrain.__init__`.
  """
  if env_name not in ('maniskill_close_subtask_train',
                      'maniskill_open_subtask_train'):
    return {}
  _require_mshab(env_name)
  if env_name == 'maniskill_open_subtask_train':
    task_plan_fp, spawn_data_fp = _mshab_open_subtask_paths()
  else:
    task_plan_fp, spawn_data_fp = _mshab_close_subtask_paths()
  plan_data = _mshab_plan_data_from_file(task_plan_fp)
  if env_name == 'maniskill_open_subtask_train':
    plan_data = _filter_open_subtask_plan_data(plan_data)
  return dict(
      # Overrides mshab's registered TimeLimit -- see
      # `ManiskillCloseSubtaskTrain.__init__`/
      # `ManiskillOpenSubtaskTrain.__init__`'s matching comments. Must equal
      # `_MANISKILL_MAX_EPISODE_STEPS[env_name]` and `load()`'s
      # `max_episode_steps` for this env_name.
      max_episode_steps=600,
      robot_uids='fetch',
      reward_mode='normalized_dense',
      task_plans=plan_data.plans,
      scene_builder_cls=plan_data.dataset,
      spawn_data_fp=spawn_data_fp,
      require_build_configs_repeated_equally_across_envs=False,
      # mshab's SequentialTaskEnv._default_sim_config hardcodes GPU PhysX
      # buffer sizes (max_rigid_contact_count=2**23,
      # max_rigid_patch_count=2**21, temp_buffer_capacity=2**24,
      # found_lost_pairs_capacity=2**25) with NO scaling by num_envs --
      # mshab's own quickstart validates this only up to num_envs=252 (63
      # RCAD scenes x 4). At num_envs=512 in a ~96-actor cluttered kitchen
      # scene per env, a real run (job 9136398) crashed all 3 seeds at
      # iteration ~30-35 with identical `CUDA_ERROR_ILLEGAL_ADDRESS` inside
      # PhysX GPU contact-solver kernels -- consistent with the fixed
      # contact/patch buffers overflowing once the policy's exploration
      # starts generating more scene contacts than at initialization.
      # Doubling every buffer here as a safety margin (still cheap in GPU
      # memory relative to the scene itself).
      #
      # collision_stack_size wasn't part of that first round -- ManiSkill's
      # own default (64*64*1024 = 4194304 bytes) is untouched by mshab's sim
      # config, and a later run at NUM_ENVS=256 (job 9143247) still
      # overflowed it: `PxgDynamicsMemoryConfig::collisionStackSize buffer
      # overflow ... increase its size to at least 67263240`, again
      # cascading into `CUDA_ERROR_ILLEGAL_ADDRESS` in subsequent kernels.
      #
      # NOT fixable via a kwarg here, though: the pinned mani_skill==3.0.0b18
      # / sapien==3.0.0b1 (close_subtask_train.slurm's mshab_rl env) expose
      # no collision_stack_size field/parameter anywhere in their Python API
      # (confirmed against both wheels directly -- mani_skill's
      # GPUMemoryConfig dataclass and sapien's compiled
      # physx.set_gpu_memory_config both lack it entirely; it was added in a
      # later sapien/mani_skill release than what's installed). Passing it
      # anyway raises `dacite.exceptions.UnexpectedDataError` before the env
      # even builds (job 9189763). Until the mshab_rl env is upgraded to a
      # sapien/mani_skill pair that exposes this knob, the only lever left
      # is reducing NUM_ENVS in close_subtask_train.slurm further.
      sim_config=dict(
          gpu_memory_config=dict(
              temp_buffer_capacity=2**25,
              max_rigid_contact_count=2**24,
              found_lost_pairs_capacity=2**26,
              max_rigid_patch_count=2**22)))


def _maniskill_control_mode(env_name: str) -> str:
  """Per-task ManiSkill `control_mode` for `ManiskillVecEnv`'s `gym.make`.

  `pd_ee_delta_pos` (Cartesian IK) for every task except
  maniskill_close_subtask_train/maniskill_open_subtask_train, which need
  `pd_joint_delta_pos` -- see `ManiskillCloseSubtaskTrain`'s docstring for
  the empirically-observed GPU-batched IK-controller bug that forced this
  deviation (open shares the same underlying SequentialTaskEnv/merged-
  articulation setup, so the same bug applies).
  """
  if env_name in ('maniskill_close_subtask_train',
                  'maniskill_open_subtask_train'):
    return 'pd_joint_delta_pos'
  return 'pd_ee_delta_pos'


def maniskill_vec_supported(env_name: str) -> bool:
  return env_name in _MANISKILL_TASK_IDS


def maniskill_static_obs_info(env_name: str):
  """Returns `(obs_dim, max_episode_steps)` without constructing any env.

  Callers that intend to use `--maniskill_native_vec` need these two
  integers *before* building the real (GPU, `num_envs>1`) vec env, but must
  NOT construct a throwaway `num_envs=1` (CPU) ManiSkill env to get them:
  SAPIEN requires `physx.enable_gpu()` to be the first PhysX-touching call
  in the process, and a CPU env construction beforehand permanently blocks
  it (raises "GPU PhysX can only be enabled once before any other code
  involving PhysX"). Both values are static per env_name, so a plain dict
  lookup is all that's needed.
  """
  return _MANISKILL_STATE_DIM[env_name], _MANISKILL_MAX_EPISODE_STEPS[env_name]


class ManiskillVecEnv:
  """Native GPU-batched vectorized ManiSkill env.

  One SAPIEN scene simulates all `num_envs` copies at once (PhysX GPU
  backend), instead of `num_envs` independent single-env SAPIEN scenes
  stepped one at a time in a Python loop (`ppo_learner.VecEnv`). Requires a
  CUDA GPU -- ManiSkill itself raises at construction time if `num_envs > 1`
  and no GPU is visible.

  Exposes the same `reset()`/`step()` contract as `ppo_learner.VecEnv`
  (stacked-over-E next_obs/env_rewards/dones/terminal_obs), so it's a
  drop-in replacement for ManiSkill env_names.

  `obs_dim`/`start_index`/`end_index` reproduce the same goal-index
  selection that `contrastive_utils.make_environment`'s
  `ObservationFilterWrapper` applies for the single-env path: this class's
  raw per-task obs builders return `concat([state, own_goal])`; the
  observation actually handed to the policy is
  `concat([obs[:obs_dim], obs[obs_dim+start_index : obs_dim+end_index]])`.
  """

  def __init__(self, env_name, num_envs, obs_dim, start_index, end_index,
              render_mode=None, success_key='success'):
    _require_maniskill(env_name)
    if env_name not in _MANISKILL_TASK_IDS:
      raise ValueError(f'ManiskillVecEnv: unsupported env_name {env_name!r}')
    if success_key not in ('success', 'drawer_closed', 'drawer_open'):
      raise ValueError(
          f'ManiskillVecEnv: unsupported success_key {success_key!r}')
    if success_key == 'drawer_closed' and env_name != 'maniskill_close_subtask_train':
      raise ValueError(
          "success_key='drawer_closed' is only valid for "
          "env_name='maniskill_close_subtask_train'")
    if success_key == 'drawer_open' and env_name != 'maniskill_open_subtask_train':
      raise ValueError(
          "success_key='drawer_open' is only valid for "
          "env_name='maniskill_open_subtask_train'")
    self._env_name = env_name
    self._success_key = success_key
    self._num_envs = int(num_envs)
    self._max_episode_steps = _MANISKILL_MAX_EPISODE_STEPS[env_name]
    extra_kwargs = _maniskill_extra_env_kwargs(env_name)
    self._env = _gymnasium.make(
        _MANISKILL_TASK_IDS[env_name],
        obs_mode='state_dict',
        control_mode=_maniskill_control_mode(env_name),
        render_mode=render_mode,
        num_envs=self._num_envs,
        **extra_kwargs)
    self._device = self._env.unwrapped.device
    # Populated by `reset()`/`step()` for maniskill_close_subtask_train /
    # maniskill_open_subtask_train only -- see
    # `_close_subtask_closed_handle_positions`/
    # `_open_subtask_open_handle_positions`. Shapes (num_envs, 3) and
    # (num_envs, 2) respectively. Named `_closed_handle_world_p` for
    # historical reasons (predates the open-subtask sibling); holds the
    # OPEN target's cached handle position when `self._env_name ==
    # 'maniskill_open_subtask_train'`.
    self._closed_handle_world_p = None
    self._base_target_xy = None

    from contrastive.utils import obs_to_goal_1d as _obs_to_goal_1d
    goal_indices = obs_dim + _obs_to_goal_1d(
        np.arange(obs_dim), start_index, end_index)
    self._indices = np.concatenate([np.arange(obs_dim), goal_indices])

    action_dim = int(self._env.action_space.shape[-1])
    self._action_shape = (action_dim,)
    obs = self.reset()
    self._obs_shape = (obs.shape[-1],)

  def _raw_obs(self, obs_dict):
    if self._env_name == 'maniskill_pushcube':
      return _pushcube_obs_batched(obs_dict)
    elif self._env_name == 'maniskill_pickcube':
      return _pickcube_obs_batched(obs_dict)
    elif self._env_name == 'maniskill_open_cabinet_drawer':
      return _open_cabinet_drawer_obs_batched(
          obs_dict, self._env.unwrapped.target_qpos,
          self._env.unwrapped.agent.is_grasping(
              self._env.unwrapped.handle_link))
    elif self._env_name == 'maniskill_close_cabinet_drawer':
      return _close_cabinet_drawer_obs_batched(
          obs_dict, self._env.unwrapped.target_qpos,
          self._env.unwrapped.agent.is_grasping(
              self._env.unwrapped.handle_link))
    elif self._env_name == 'maniskill_close_subtask_train':
      unwrapped = self._env.unwrapped
      handle_pos_wrt_base = _close_subtask_handle_pos_wrt_base(
          unwrapped, self._closed_handle_world_p)
      return _close_subtask_obs_batched(
          obs_dict,
          unwrapped.articulation.qpos[
              :, unwrapped.close_subtask.articulation_handle_active_joint_idx],
          unwrapped.target_qpos,
          handle_pos_wrt_base,
          unwrapped.agent.base_link.pose,
          self._base_target_xy)
    elif self._env_name == 'maniskill_open_subtask_train':
      unwrapped = self._env.unwrapped
      handle_pos_wrt_base = _open_subtask_handle_pos_wrt_base(
          unwrapped, self._closed_handle_world_p)
      return _open_subtask_obs_batched(
          obs_dict,
          unwrapped.articulation.qpos[
              :, unwrapped.open_subtask.articulation_handle_active_joint_idx],
          unwrapped.target_qpos,
          handle_pos_wrt_base,
          unwrapped.agent.base_link.pose,
          self._base_target_xy)
    else:
      raise ValueError(f'ManiskillVecEnv: unsupported env_name {self._env_name!r}')

  def _get_obs(self, obs_dict):
    return self._raw_obs(obs_dict)[:, self._indices]

  def reset(self):
    obs_dict, _info = self._env.reset()
    if self._env_name == 'maniskill_close_subtask_train':
      all_idx = _torch.arange(self._num_envs, dtype=_torch.long,
                              device=self._device)
      self._closed_handle_world_p, self._base_target_xy = (
          _close_subtask_closed_handle_positions(
              self._env.unwrapped, all_idx))
    elif self._env_name == 'maniskill_open_subtask_train':
      all_idx = _torch.arange(self._num_envs, dtype=_torch.long,
                              device=self._device)
      self._closed_handle_world_p, self._base_target_xy = (
          _open_subtask_open_handle_positions(
              self._env.unwrapped, all_idx))
    return self._get_obs(obs_dict)

  def step(self, actions):
    actions = np.asarray(actions, dtype=np.float32)
    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = np.clip(actions, -1.0, 1.0)
    actions_t = _torch.as_tensor(
        actions, dtype=_torch.float32, device=self._device)
    obs_dict, _reward_t, _terminated_t, truncated_t, info = self._env.step(
        actions_t)
    obs = self._get_obs(obs_dict)
    terminal_obs = obs.copy()

    if self._success_key == 'drawer_closed':
      # Drawer-only success signal, deliberately narrower than
      # `info['success']` (mshab's own success also requires the arm back at
      # rest and the robot static -- see
      # `contrastive.utils.DrawerClosedSuccessObserver`, and the single-env
      # `ManiskillCloseSubtaskTrain.step()`'s identical formula).
      unwrapped = self._env.unwrapped
      joint_idx = unwrapped.close_subtask.articulation_handle_active_joint_idx
      drawer_closed_t = (
          unwrapped.articulation.qpos[:, joint_idx] < unwrapped.target_qpos)
      if drawer_closed_t.ndim > 1:
        drawer_closed_t = drawer_closed_t.reshape(self._num_envs)
      env_rewards = drawer_closed_t.detach().cpu().numpy().astype(np.float32)
    elif self._success_key == 'drawer_open':
      # Mirrors the `drawer_closed` branch above, flipped: deliberately
      # narrower than `info['success']` (see
      # `contrastive.utils.DrawerOpenSuccessObserver`, and the single-env
      # `ManiskillOpenSubtaskTrain.step()`'s identical formula).
      unwrapped = self._env.unwrapped
      joint_idx = unwrapped.open_subtask.articulation_handle_active_joint_idx
      drawer_open_t = (
          unwrapped.articulation.qpos[:, joint_idx] > unwrapped.target_qpos)
      if drawer_open_t.ndim > 1:
        drawer_open_t = drawer_open_t.reshape(self._num_envs)
      env_rewards = drawer_open_t.detach().cpu().numpy().astype(np.float32)
    else:
      success = info.get('success', None)
      if success is not None:
        env_rewards = success.detach().cpu().numpy().astype(np.float32)
      else:
        env_rewards = np.zeros(self._num_envs, dtype=np.float32)

    # Truncation only (episode length limit), matching the single-env
    # wrappers' `done=False` convention: success never ends an episode
    # early here, only the fixed step budget does.
    dones = truncated_t.detach().cpu().numpy().astype(bool)
    next_obs = obs
    if dones.any():
      idx = np.nonzero(dones)[0]
      idx_t = _torch.as_tensor(idx, dtype=_torch.long, device=self._device)
      reset_obs_dict, _ = self._env.reset(options=dict(env_idx=idx_t))
      if self._env_name == 'maniskill_close_subtask_train':
        # Only these `idx` rows get a fresh closed-handle/base target --
        # every other (still mid-episode) env's cached targets must be left
        # alone.
        new_handle_p, new_base_xy = _close_subtask_closed_handle_positions(
            self._env.unwrapped, idx_t)
        self._closed_handle_world_p[idx] = new_handle_p
        self._base_target_xy[idx] = new_base_xy
      elif self._env_name == 'maniskill_open_subtask_train':
        # Same idea as close's branch above: only `idx` rows get a fresh
        # open-handle/base target.
        new_handle_p, new_base_xy = _open_subtask_open_handle_positions(
            self._env.unwrapped, idx_t)
        self._closed_handle_world_p[idx] = new_handle_p
        self._base_target_xy[idx] = new_base_xy
      reset_obs = self._get_obs(reset_obs_dict)
      next_obs = obs.copy()
      next_obs[idx] = reset_obs[idx]

    return next_obs, env_rewards, dones, terminal_obs

  def close(self):
    self._env.close()

  @property
  def num_envs(self) -> int:
    return self._num_envs

  @property
  def observation_shape(self):
    return self._obs_shape

  @property
  def action_shape(self):
    return self._action_shape

  @property
  def max_episode_steps(self) -> int:
    return self._max_episode_steps


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


def load(env_name, fixed_start_end=None, seed=None, render_mode=None,
        **env_kwargs):
  """Loads the train and eval environments, as well as the obs_dim.

  Args:
    env_name: Registered environment id.
    fixed_start_end: Env-specific fixed goal / start–goal (see each env).
    seed: Optional RNG seed (used by ``riverswim``; others may ignore it).
    render_mode: Forwarded to ManiSkill envs only (e.g. 'rgb_array' to
      enable `.render()` for video rollouts); ignored by other envs.
    **env_kwargs: extra env-specific kwargs (e.g. sawyer_bin's
      `randomize_gripper_init`); ignored by envs that don't accept them.
  """
  # pylint: disable=invalid-name
  kwargs = {}
  if env_name == 'sawyer_bin':
    CLASS = SawyerBin
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
    if 'randomize_gripper_init' in env_kwargs:
      kwargs['randomize_gripper_init'] = env_kwargs['randomize_gripper_init']
  elif env_name == 'sawyer_box':
    CLASS = SawyerBox
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name == 'sawyer_peg':
    CLASS = SawyerPeg
    max_episode_steps = 150
    kwargs['fixed_start_end'] = fixed_start_end
  elif env_name.startswith('point_'):
    CLASS = point_env.PointEnv
    kwargs['walls'] = env_name.split('_')[-1]
    kwargs['fixed_start_end'] = fixed_start_end
    if ('11x11' in env_name or '9x9' in env_name or '7x7' in env_name
        or 'Impossible' in env_name):
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
  elif env_name == 'maniskill_pushcube':
    CLASS = ManiskillPushCube
    max_episode_steps = 50  # ManiSkill3's own @register_env(..., max_episode_steps=50)
    kwargs['fixed_start_end'] = fixed_start_end
    kwargs['render_mode'] = render_mode
  elif env_name == 'maniskill_pickcube':
    CLASS = ManiskillPickCube
    max_episode_steps = 50  # PickCubeEnv's own @register_env(..., max_episode_steps=50)
    kwargs['fixed_start_end'] = fixed_start_end
    kwargs['render_mode'] = render_mode
  elif env_name == 'maniskill_open_cabinet_drawer':
    CLASS = ManiskillOpenCabinetDrawer
    max_episode_steps = 100  # OpenCabinetDrawerEnv's own @register_env(..., max_episode_steps=100)
    kwargs['fixed_start_end'] = fixed_start_end
    kwargs['render_mode'] = render_mode
  elif env_name == 'maniskill_close_cabinet_drawer':
    CLASS = ManiskillCloseCabinetDrawer
    max_episode_steps = 100  # CloseCabinetDrawerEnv's own @register_env(..., max_episode_steps=100)
    kwargs['fixed_start_end'] = fixed_start_end
    kwargs['render_mode'] = render_mode
  elif env_name == 'maniskill_close_subtask_train':
    CLASS = ManiskillCloseSubtaskTrain
    # 600, not CloseSubtaskTrainEnv's own registered 400 -- overridden via
    # `max_episode_steps=` passed to `gymnasium.make()` in
    # `ManiskillCloseSubtaskTrain.__init__` (see that call site's comment).
    max_episode_steps = 600
    kwargs['fixed_start_end'] = fixed_start_end
    kwargs['render_mode'] = render_mode
  elif env_name == 'maniskill_open_subtask_train':
    CLASS = ManiskillOpenSubtaskTrain
    # 600, mirroring close -- overridden via `max_episode_steps=` passed to
    # `gymnasium.make()` in `ManiskillOpenSubtaskTrain.__init__` (see that
    # call site's comment).
    max_episode_steps = 600
    kwargs['fixed_start_end'] = fixed_start_end
    kwargs['render_mode'] = render_mode
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)

  # Disable type checking in line below because different environments have
  # different kwargs, which pytype doesn't reason about.
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  # STATE_DIM is only defined by wrappers with an asymmetric state/goal
  # split (ManiskillOpenCabinetDrawer, ManiskillCloseCabinetDrawer,
  # ManiskillCloseSubtaskTrain); everything else keeps the historical
  # convention of an exact state/goal 50-50 split.
  obs_dim = getattr(CLASS, 'STATE_DIM', gym_env.observation_space.shape[0] // 2)
  return gym_env, obs_dim, max_episode_steps


class SawyerBin(_MW_BIN):
  """Wrapper for the SawyerBin environment."""

  def __init__(self, fixed_start_end=None, randomize_gripper_init=False):
    _require_metaworld('sawyer_bin')
    self._goal = np.zeros(3)
    self._randomize_gripper_init = bool(randomize_gripper_init)
    super(SawyerBin, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._fixed_start_end=fixed_start_end
    self.reset()

  def _reset_hand(self, steps=50):
    if not self._randomize_gripper_init:
      # Unmodified base-class behavior: reset to the fixed hand_init_pos.
      return super(SawyerBin, self)._reset_hand(steps=steps)
    # NOTE: the base class calls _reset_hand() from reset_model() BEFORE
    # placing the object for this episode (_set_obj_xyz runs after), so
    # `_get_pos_objects()` here reads the *previous* episode's object
    # position, not this episode's freshly-sampled one. Close enough for
    # a randomized-offset exploration bonus, but not an exact "grasp-ready
    # near this episode's object" pose.
    obj_pos = self._get_pos_objects().copy()
    grip_target = obj_pos + np.array([
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(-0.05, 0.05),
        np.random.uniform(0.01, 0.06),
    ], dtype=np.float64)
    for _ in range(steps):
      self.data.set_mocap_pos('mocap', grip_target)
      self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
      self.do_simulation([-1, 1], self.frame_skip)
    self.init_tcp = self.tcp_center

  def reset(self):
    super(SawyerBin, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    pos1 = self.sim.data.body_xpos[body_id].copy()
    pos1 += np.random.uniform(-0.05, 0.05, 3)
    pos2 = self._get_pos_objects().copy()
    
    if self._fixed_start_end is not None:
        # Set the goal to be a fixed location
        self._goal = self._fixed_start_end 
    else:
        t = np.random.random()
        # Set the goal to be a uniformly sampled location
        # between the starting and end point
        self._goal = t * pos1 + (1 - t) * pos2
        self._goal[2] = np.random.uniform(0.03, 0.12)
    self._target_pos = self._goal
    return self._get_obs()

  def step(self, action):
    super(SawyerBin, self).step(action)
    obj_pos = self._get_pos_objects()
    dist = np.linalg.norm(self._goal - obj_pos)
    obs = self._get_obs()
    r = float(dist < 0.05)  # Taken from metaworld
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
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          self._get_pos_objects()))
    # the ideal goal state has the block in the blue bin and the gripper slightly 
    # higher than the block center
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal])

    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)


class SawyerBox(_MW_BOX):
  """Wrapper for the SawyerBox environment."""

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

class SawyerPeg(_MW_PEG):
  """Wrapper for the SawyerPeg environment."""

  def __init__(self, fixed_start_end=None):
    _require_metaworld('sawyer_peg')
    self._goal_pos = np.zeros(3)
    super(SawyerPeg, self).__init__()
    self._fixed_start_end=fixed_start_end
    self._set_task_called = True
    self._partially_observable = False
    self._freeze_rand_vec = False
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
