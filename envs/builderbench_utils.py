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
def load_task_goal_offsets(num_cubes: int, task_index: int) -> np.ndarray:
  """Load per-cube goal offsets from ``builderbench/tasks/creative-{N}.npz``."""
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
  goals = np.asarray(data['goals'][task_index], dtype=np.float32)
  return goals.reshape(-1, 3)


def default_fixed_target_goal(num_cubes: int, task_index: int) -> np.ndarray:
  """Nominal fixed ``target_goal`` at the sampling midpoint + task offsets."""
  offsets = load_task_goal_offsets(num_cubes, task_index)
  return (_TARGET_SAMPLING_MID.reshape(1, 3) + offsets).reshape(-1)


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

def filter_pd_policy_state_obs(state_obs, num_cubes, obs_space_list: list[str]):
    """Drop unwanted state components; keep components based on obs_space_list."""
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


def ppo_env_defaults(num_cubes: int, use_pd: bool = False,
                     episode_length_multiplier: float = 1.0) -> dict:
  """Rollout / CRL defaults scaled to BuilderBench creative episode length."""
  episode_length = scaled_episode_length(num_cubes, episode_length_multiplier)
  goal_dim = num_cubes * 3
  if use_pd:
    pd_episode_length = episode_length // 5
    return dict(
        rollout_length=max(32, pd_episode_length),
        crl_steps_per_iter=max(16, pd_episode_length // 2),
        start_index=0,
        end_index=goal_dim,
        checkpoint_interval=150,
    )
  rollout_length = min(512, max(256, episode_length))
  return dict(
      rollout_length=rollout_length,
      crl_steps_per_iter=rollout_length // 2,
      start_index=0,
      end_index=goal_dim,
      num_envs=64,
      eval_interval=0,
      checkpoint_interval=150,
  )


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
    episode_length_multiplier: float = 1.0,
) -> dict:
  """PPO / CRL scale defaults for BuilderBench (200M steps, replay ≥ E)."""
  out = ppo_env_defaults(num_cubes, use_pd=use_pd,
                         episode_length_multiplier=episode_length_multiplier)
  out['num_steps'] = BUILDERBENCH_NUM_STEPS
  out['max_replay_size'] = builderbench_replay_size(num_envs)
  return out
