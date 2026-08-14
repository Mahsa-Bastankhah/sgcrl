"""Shared helpers for BuilderBench creative-cube env ids in sgcrl."""
from __future__ import annotations

import glob
import json
import os
import re
from functools import lru_cache
from typing import List, Tuple

import numpy as np

_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')

_SGCRL_CREATIVE_RE = re.compile(
    r'^builderbench_creative_(\d+)_task(\d+)$')
_BB_CREATIVE_RE = re.compile(r'^creative-(\d+)-task(\d+)$')

_TARGET_SAMPLING_LOW = np.array([0.22, -0.10, 0.02], dtype=np.float32)
_TARGET_SAMPLING_HIGH = np.array([0.32, 0.10, 0.02], dtype=np.float32)
_TARGET_SAMPLING_MID = (_TARGET_SAMPLING_LOW + _TARGET_SAMPLING_HIGH) / 2


def scaled_episode_length(num_cubes: int, multiplier: float = 1.0) -> int:
  """Base creative-cube episode length (100 + num_cubes*50 raw MuJoCo
  steps), optionally scaled. Rounds to the nearest int.

  Callers using PDWrapper must ensure the result divides evenly by
  pd_duration -- see validate_pd_episode_length.
  """
  base = 100 + int(num_cubes) * 50
  return int(round(base * float(multiplier)))


def validate_pd_episode_length(episode_length: int, pd_duration: int) -> None:
  if episode_length % pd_duration != 0:
    raise ValueError(
        f'episode_length={episode_length} is not divisible by '
        f'pd_duration={pd_duration}. Choose a '
        f'--builderbench_episode_length_multiplier such that '
        f'(100 + num_cubes*50) * multiplier is a multiple of {pd_duration}, '
        f'or change --builderbench_pd_duration.')


def is_builderbench_creative_env(env_name: str) -> bool:
  return bool(_SGCRL_CREATIVE_RE.fullmatch(str(env_name)))


def parse_bb_env_id(bb_env_id: str) -> Tuple[int, int]:
  """Parse ``creative-{N}-task{K}`` → (num_cubes, task_index)."""
  m = _BB_CREATIVE_RE.fullmatch(str(bb_env_id))
  if not m:
    raise ValueError(f'Unsupported BuilderBench env_id: {bb_env_id!r}')
  return int(m.group(1)), int(m.group(2)) - 1


def parse_sgcrl_builderbench_env_name(
    env_name: str,
) -> Tuple[str, int, int]:
  """Parse ``builderbench_creative_{N}_task{K}`` → (bb_env_id, num_cubes, task_index)."""
  m = _SGCRL_CREATIVE_RE.fullmatch(str(env_name))
  if not m:
    raise ValueError(f'Not a builderbench creative env name: {env_name!r}')
  num_cubes = int(m.group(1))
  task_num = int(m.group(2))
  return f'creative-{num_cubes}-task{task_num}', num_cubes, task_num - 1


def sgcrl_env_name_to_bb_env_id(env_name: str) -> str:
  """Map ``builderbench_creative_1_task2`` → ``creative-1-task2``."""
  if not env_name.startswith('builderbench_'):
    raise ValueError(f'Not a builderbench env name: {env_name!r}')
  if is_builderbench_creative_env(env_name):
    bb_env_id, _, _ = parse_sgcrl_builderbench_env_name(env_name)
    return bb_env_id
  raise ValueError(f'Unsupported builderbench env name: {env_name!r}')


def _creative_task_npz_path(num_cubes: int) -> str:
  return os.path.join(
      _BUILDERBENCH_ROOT,
      'builderbench',
      'tasks',
      f'creative-{num_cubes}.npz',
  )


def num_creative_tasks(num_cubes: int) -> int:
  """Number of tasks defined in ``creative-{N}.npz``."""
  path = _creative_task_npz_path(num_cubes)
  if not os.path.isfile(path):
    raise FileNotFoundError(
        f'BuilderBench task file not found: {path!r} '
        f'(set BUILDERBENCH_ROOT if the repo lives elsewhere).')
  return int(np.load(path)['goals'].shape[0])


def list_creative_env_names() -> List[str]:
  """All sgcrl env ids for installed creative-cube task files."""
  pattern = os.path.join(
      _BUILDERBENCH_ROOT, 'builderbench', 'tasks', 'creative-*.npz')
  names: List[str] = []
  for path in sorted(glob.glob(pattern)):
    m = re.search(r'creative-(\d+)\.npz$', os.path.basename(path))
    if not m:
      continue
    num_cubes = int(m.group(1))
    n_tasks = num_creative_tasks(num_cubes)
    for task_num in range(1, n_tasks + 1):
      names.append(f'builderbench_creative_{num_cubes}_task{task_num}')
  return names


@lru_cache(maxsize=None)
def load_task_cube_mask(num_cubes: int, task_index: int) -> np.ndarray:
  """Per-cube goal mask (True = cube in target/achieved goal).

  Missing ``masks`` in the npz → all-True (full goal, backward compatible).
  """
  path = _creative_task_npz_path(num_cubes)
  if not os.path.isfile(path):
    raise FileNotFoundError(
        f'BuilderBench task file not found: {path!r} '
        f'(set BUILDERBENCH_ROOT if the repo lives elsewhere).')
  data = np.load(path)
  n_tasks = int(data['goals'].shape[0])
  if task_index < 0 or task_index >= n_tasks:
    raise ValueError(
        f'creative-{num_cubes} has {n_tasks} task(s) (task1..task{n_tasks}); '
        f'got task index {task_index} (task{task_index + 1}). '
        f'Valid env names: '
        f'{", ".join(f"builderbench_creative_{num_cubes}_task{t}" for t in range(1, n_tasks + 1))}')
  if 'masks' in data:
    mask = np.asarray(data['masks'][task_index], dtype=bool).reshape(-1)
  else:
    mask = np.ones(int(num_cubes), dtype=bool)
  if mask.shape[0] != int(num_cubes):
    raise ValueError(
        f'creative-{num_cubes} task{task_index + 1} mask length '
        f'{mask.shape[0]} != num_cubes={num_cubes}')
  if not np.any(mask):
    raise ValueError(
        f'creative-{num_cubes} task{task_index + 1} mask is empty')
  return mask


def goal_xyz_state_indices(
    num_cubes: int,
    task_index: int,
    start_index: int = 0,
) -> np.ndarray:
  """Flat state indices of masked-in cube xyz (for LHER ``obs_to_goal``)."""
  mask = load_task_cube_mask(num_cubes, task_index)
  idxs: list[int] = []
  base = int(start_index)
  for i, keep in enumerate(mask.tolist()):
    if keep:
      idxs.extend([base + 3 * i, base + 3 * i + 1, base + 3 * i + 2])
  return np.asarray(idxs, dtype=np.int32)


@lru_cache(maxsize=None)
def load_task_goal_offsets(num_cubes: int, task_index: int) -> np.ndarray:
  """Load masked-in per-cube goal offsets from ``creative-{N}.npz``.

  Returns shape ``(num_task_cubes, 3)`` — only cubes with mask True.
  """
  path = _creative_task_npz_path(num_cubes)
  if not os.path.isfile(path):
    raise FileNotFoundError(
        f'BuilderBench task file not found: {path!r} '
        f'(set BUILDERBENCH_ROOT if the repo lives elsewhere).')
  data = np.load(path)
  n_tasks = int(data['goals'].shape[0])
  if task_index < 0 or task_index >= n_tasks:
    raise ValueError(
        f'creative-{num_cubes} has {n_tasks} task(s) (task1..task{n_tasks}); '
        f'got task index {task_index} (task{task_index + 1}). '
        f'Valid env names: '
        f'{", ".join(f"builderbench_creative_{num_cubes}_task{t}" for t in range(1, n_tasks + 1))}')
  goals = np.asarray(data['goals'][task_index], dtype=np.float32).reshape(-1, 3)
  mask = load_task_cube_mask(num_cubes, task_index)
  return goals[mask]


def default_fixed_target_goal(num_cubes: int, task_index: int) -> np.ndarray:
  """Fixed ``target_goal`` = sampling midpoint + masked task offsets."""
  offsets = load_task_goal_offsets(num_cubes, task_index)
  return (_TARGET_SAMPLING_MID.reshape(1, 3) + offsets).reshape(-1).astype(
      np.float32)


def creative_goal_dim(num_cubes: int, task_index: int = 0) -> int:
  """Goal vector length = ``3 * num_masked_in_cubes``."""
  return int(3 * int(np.sum(load_task_cube_mask(num_cubes, task_index))))


def creative_cube_mj_episode_length(num_cubes: int, task_index: int = 0) -> int:
  """MuJoCo steps per episode (before PD macro division).

  Default: ``100 + num_cubes * 50``.  creative-4-task1 uses 500 MJ steps
  (= 100 PD macro steps at ``pd_duration=5``).
  """
  nc = int(num_cubes)
  ti = int(task_index)
  if nc == 4 and ti == 0:
    return 500
  return 100 + nc * 50


def creative_cube_pd_macro_episode_length(
    num_cubes: int,
    task_index: int = 0,
    pd_duration: int = 5,
) -> int:
  """RL episode length when wrapped in ``PDWrapper``."""
  mj_len = creative_cube_mj_episode_length(num_cubes, task_index)
  pd = int(pd_duration)
  if mj_len % pd != 0:
    raise ValueError(
        f'creative_cube_mj_episode_length={mj_len} must divide '
        f'pd_duration={pd}')
  return mj_len // pd


def creative_cube_full_state_obs_dim(num_cubes: int) -> int:
  """State obs length from ``CreativeCube.get_obs`` (non-delta control)."""
  nc = int(num_cubes)
  # obj_pos + obj_quat + obj_linvel + obj_angvel + select_action
  return nc * (3 + 4 + 3 + 3) + 1


def pd_policy_state_obs_dim(num_cubes: int) -> int:
  """High-level PD policy obs: cube xyz positions + select_action only."""
  return int(num_cubes) * 3 + 1


def get_filtered_obs_dim(num_cubes: int, obs_space_list: list[str]) -> int:
    """Calculate the dimension of the filtered observation."""
    nc = int(num_cubes)
    dim_map = {
        "xy": 3 * nc,
        "quaternions": 4 * nc,
        "linear_velocity": 3 * nc,
        "angular_velocity": 3 * nc,
        "select": 1
    }
    requested = ["xy"]
    requested.extend([s for s in obs_space_list if s in dim_map and s not in requested])
    if "select" not in requested:
        requested.append("select")
    return sum(dim_map[s] for s in requested)

def filter_pd_policy_state_obs(state_obs, num_cubes, obs_space_list: Optional[list[str]] = None):
    """Drop unwanted state components; keep components based on obs_space_list."""
    if obs_space_list is None:
        obs_space_list = ["xy", "select"]
    nc = int(num_cubes)
    components = {
        "xy": state_obs[..., :3 * nc],
        "quaternions": state_obs[..., 3 * nc : 7 * nc],
        "linear_velocity": state_obs[..., 7 * nc : 10 * nc],
        "angular_velocity": state_obs[..., 10 * nc : 13 * nc],
        "select": state_obs[..., -1:]
    }
    
    requested = ["xy"]
    requested.extend([s for s in obs_space_list if s in components and s not in requested])
    if "select" not in requested:
        requested.append("select")
        
    # Support both NumPy (single env) and JAX (batched env) arrays
    if isinstance(state_obs, np.ndarray):
        return np.concatenate([components[s] for s in requested], axis=-1)
    
    import jax.numpy as jnp
    return jnp.concatenate([components[s] for s in requested], axis=-1)


def set_task_mocap_pos(mocap_pos, mocap_targets, fixed_pos):
  """Write task-cube xyz into ``mocap_pos`` for unbatched or batched states.

  Batched MJX states have ``mocap_pos`` shape ``[E, n_mocap, 3]``. Indexing
  with ``.at[mocap_targets]`` incorrectly scatters along the *env* axis, which
  breaks masked tasks (e.g. creative-5-task3 with targets ``[1,3,4]``) and only
  updates the first ``n_task`` envs on full-mask tasks. Always index the mocap
  axis for rank-3 arrays.
  """
  import jax.numpy as jnp
  pos = jnp.asarray(mocap_pos)
  targets = jnp.asarray(mocap_targets)
  fixed = jnp.asarray(fixed_pos, dtype=jnp.float32)
  if pos.ndim == 2:
    return pos.at[targets].set(fixed)
  if pos.ndim != 3:
    raise ValueError(f'Unexpected mocap_pos rank {pos.ndim}; expected 2 or 3')
  fixed_b = jnp.broadcast_to(fixed, (pos.shape[0],) + tuple(fixed.shape))
  return pos.at[:, targets].set(fixed_b)


def apply_fixed_start_x(creative_cube, fixed_start_x: float | None) -> None:
  """Collapse start-box x low/high to a constant (in-place on ``_starts_data``).

  ``starts`` shape is ``(num_cubes, 2, 3)`` = [cube, {low,high}, xyz]. Setting
  both low and high x makes ``jax.random.uniform`` always return that x.
  Pass ``None`` or a negative value to leave starts unchanged.
  """
  if fixed_start_x is None:
    return
  x = float(fixed_start_x)
  if x < 0:
    return
  starts = np.asarray(creative_cube._starts_data, dtype=np.float32).copy()
  if starts.ndim != 3 or starts.shape[-1] < 1:
    raise ValueError(
        f'Unexpected _starts_data shape {starts.shape}; expected (N, 2, 3)')
  starts[:, :, 0] = x
  # Keep the same array backend as CreativeCube (jax array).
  try:
    import jax.numpy as jnp
    creative_cube._starts_data = jnp.asarray(starts, dtype=jnp.float32)
  except Exception:
    creative_cube._starts_data = starts


def pd_run_uses_filtered_policy_obs(run_cfg: dict) -> bool:
  flags = run_cfg.get('flags', {})
  if not bool(flags.get('builderbench_use_pd', False)):
    return False
  resolved = run_cfg.get('resolved_config', {})
  obs_dim = int(resolved.get('obs_dim', -1))
  env = str(run_cfg.get('env', ''))
  m = _SGCRL_CREATIVE_RE.fullmatch(env)
  if not m:
    return False
  num_cubes = int(m.group(1))
  
  # FIX: Calculate based on actual obs_space
  obs_space_str = flags.get('obs_space', 'xy,select')
  obs_space_list = [s.strip() for s in obs_space_str.split(',')]
  expected_dim = get_filtered_obs_dim(num_cubes, obs_space_list)
  
  return obs_dim == expected_dim

def video_render_skip_reason(run_cfg_path: str) -> str | None:
  if not os.path.isfile(run_cfg_path):
    return None
  with open(run_cfg_path, 'r', encoding='utf-8') as fh:
    run_cfg = json.load(fh)
  flags = run_cfg.get('flags', {})
  if not bool(flags.get('builderbench_use_pd', False)):
    return None
  if pd_run_uses_filtered_policy_obs(run_cfg):
    return None
  resolved = run_cfg.get('resolved_config', {})
  obs_dim = int(resolved.get('obs_dim', -1))
  env = str(run_cfg.get('env', ''))
  m = _SGCRL_CREATIVE_RE.fullmatch(env)
  
  # FIX: Calculate based on actual obs_space
  obs_space_str = flags.get('obs_space', 'xy,select')
  obs_space_list = [s.strip() for s in obs_space_str.split(',')]
  pd_dim = get_filtered_obs_dim(int(m.group(1)), obs_space_list) if m else '?'
  
  return (
      f'legacy PD without filtered policy obs '
      f'(obs_dim={obs_dim}, filtered={pd_dim})'
  )

def run_config_path(log_dir: str, env: str, seed: int) -> str:
  return os.path.join(
      os.path.abspath(log_dir), f'ppo_{env}_{seed}', 'run_config.json')


def uniform_goal_obs_bounds(
    num_cubes: int,
    task_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """Bounds on the goal slice for CRL uniform-sampling negatives."""
  offsets = load_task_goal_offsets(num_cubes, task_index)
  low = (_TARGET_SAMPLING_LOW.reshape(1, 3) + offsets).reshape(-1)
  high = (_TARGET_SAMPLING_HIGH.reshape(1, 3) + offsets).reshape(-1)
  return low.astype(np.float32), high.astype(np.float32)


def ppo_env_defaults(
    num_cubes: int,
    use_pd: bool = False,
    task_index: int = 0,
    pd_duration: int = 5,
    episode_length_multiplier: float = 1.0,
) -> dict:
  """Non-rollout/CRL PPO defaults scaled to BuilderBench creative episode length.

  NOTE: for PD mode (``use_pd=True``), this deliberately does NOT return
  ``rollout_length``/``crl_steps_per_iter`` any more. Those used to be
  silently auto-computed here (``rollout_length // 2``-style formula
  copy-pasted from the point-maze defaults) and would override whatever a
  job script implied, with no visible flag. Every BuilderBench PD job must
  now pass ``--ppo_rollout_length`` and ``--ppo_crl_steps_per_iter``
  explicitly; see ``contrastive.config.ContrastiveConfig`` for the raw
  fallback defaults if neither is a passed.
  """
  episode_length = creative_cube_mj_episode_length(num_cubes, task_index)
  # Full cube-xyz region in state (used when mask is all-True).
  pos_end = int(num_cubes) * 3
  goal_dim = creative_goal_dim(num_cubes, task_index)
  mask = load_task_cube_mask(num_cubes, task_index)
  # Non-contiguous masks need explicit gather indices for LHER obs_to_goal.
  goal_state_indices = None
  if int(np.sum(mask)) < int(num_cubes):
    goal_state_indices = goal_xyz_state_indices(
        num_cubes, task_index, start_index=0).tolist()
  if use_pd:
    out = dict(
        start_index=0,
        end_index=pos_end if goal_state_indices is not None else goal_dim,
        eval_interval=30,
        checkpoint_interval=300,
    )
  else:
    rollout_length = min(512, max(256, episode_length))
    out = dict(
        rollout_length=rollout_length,
        crl_steps_per_iter=rollout_length // 2,
        start_index=0,
        end_index=pos_end if goal_state_indices is not None else goal_dim,
        num_envs=64,
        eval_interval=0,
        checkpoint_interval=300,
    )
  if goal_state_indices is not None:
    out['goal_state_indices'] = goal_state_indices
    out['goal_dim'] = goal_dim
  return out


BUILDERBENCH_NUM_STEPS = 200_000_000


def builderbench_replay_size(num_envs: int) -> int:
  """CRL replay capacity scaled to parallel env count."""
  n = int(num_envs)
  if n >= 1024:
    return 10_000_000
  if n >= 256:
    return 1_000_000
  return 1_000_000


def builderbench_training_defaults(
    num_envs: int,
    num_cubes: int,
    use_pd: bool = False,
    task_index: int = 0,
    pd_duration: int = 5,
    episode_length_multiplier: float = 1.0,
) -> dict:
  """PPO / CRL scale defaults for BuilderBench (200M steps, replay ∝ E)."""
  out = ppo_env_defaults(
      num_cubes, use_pd=use_pd, task_index=task_index, pd_duration=pd_duration, episode_length_multiplier=episode_length_multiplier)
  out['num_steps'] = BUILDERBENCH_NUM_STEPS
  out['max_replay_size'] = builderbench_replay_size(num_envs)
  return out
