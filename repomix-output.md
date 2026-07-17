This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: envs/builderbench_utils.py, envs/builderbench_env.py, envs/builderbench_jax_vec.py, ppo_contrastive.py, scripts/ppo_builderbench_rollout_video.py, scripts/ppo_builderbench_checkpoint_eval.py, contrastive/ppo_learner.py, mila/scratch_mila.sh
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
contrastive/
  ppo_learner.py
envs/
  builderbench_env.py
  builderbench_jax_vec.py
  builderbench_utils.py
mila/
  scratch_mila.sh
scripts/
  ppo_builderbench_checkpoint_eval.py
  ppo_builderbench_rollout_video.py
ppo_contrastive.py
```

# Files

## File: mila/scratch_mila.sh
```bash
#!/bin/bash
# ==============================================================================
# SGCRL PPO Contrastive Baseline + Evaluation
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ---------- USER CONFIGURATION ------------------------------------------------
# ENVS=( "builderbench_creative_3_task1" )
# ENVS=( "builderbench_creative_4_task1" "builderbench_creative_4_task6" "builderbench_creative_3_task2" "builderbench_creative_3_task5")
SEEDS=( 0 )

LOG_ROOT="/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"
# ------------------------------------------------------------------------------

BASE_FLAGS="--num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes=\"256,256,256,256,256,256\" --env=builderbench_creative_3_task1"

EXPERIMENTS=( 

    # "debug|--num_steps=20_000_000"
    # "reproduce|--env=builderbench_creative_4_task1"
    # "reproduce|--env=builderbench_creative_4_task6 --obs_space="xy,quaternions,select""
    # "reproduce|--env=builderbench_creative_3_task2 -obs_space="xy,quaternions,select""
    # "reproduce2|"
    # "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --ppo_rollout_length=128" # Roughly double which is automatically calculated to 60.
    # "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --ppo_rollout_length=256" # Roughly quadruple which is automatically calculated to 60.
    # "reproduce|--env=builderbench_creative_2_task2"
    # "reproduce|--env=builderbench_creative_2_task3"
    # "reproduce_warmup|--env=builderbench_creative_4_task1 --ppo_warmup_percent=0.05"
    # "reproduce_warmup|--env=builderbench_creative_4_task1 --ppo_warmup_percent=0.10"
    # "reproduce_warmup|--env=builderbench_creative_4_task1 --ppo_warmup_percent=0.20"
    # "reproduce_cleanr_actor|--env=builderbench_creative_4_task1 --ppo_cleanrl_actor=True"
    # "reproduce_longer_env_epi|--env=builderbench_creative_4_task1 --builderbench_episode_length_multiplier=5"
    # "reproduce_longer_env_epi|--env=builderbench_creative_4_task1 --builderbench_episode_length_multiplier=10"
    # "reproduce_longer_env_epi|--env=builderbench_creative_4_task6 --builderbench_episode_length_multiplier=5"
    #


    )

mkdir -p "$SCRIPT_DIR/slurm_logs"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    EXP_NAME="${EXPERIMENT%%|*}"
    EXP_FLAGS="${EXPERIMENT##*|}"

    if [[ "$EXP_FLAGS" =~ --env=([^ ]+) ]]; then
        ENV_ID="${BASH_REMATCH[1]}"
    else
        ENV_ID="builderbench_creative_3_task1" 
    fi
    echo "Running experiment: $EXP_NAME with env: $ENV_ID and flags: $EXP_FLAGS"


    # for ENV_ID in "${ENVS[@]}"; do
        for seed in "${SEEDS[@]}"; do


            # 1. Create a unique signature of the parameters and hash it
            SIG_STR="env=${ENV_ID}_seed=${seed}_base=${BASE_FLAGS}_exp=${EXP_FLAGS}"
            CMD_HASH=$(echo "$SIG_STR" | md5sum | cut -c1-8)
            SAFE_NAME="${EXP_NAME//+/_plus_}_${CMD_HASH}"

            # 2. Build the command using SAFE_NAME for the log_dir_path
            CMD="python -u ppo_contrastive.py \
                --seed=${seed} \
                --log_dir_path=${LOG_ROOT}/${SAFE_NAME} \
                --exp_name=${EXP_NAME} \
                ${BASE_FLAGS} \
                ${EXP_FLAGS}"


            echo "CMD: $CMD for SAFE_NAME: $SAFE_NAME"
            SLURM_SCRIPT="$SCRIPT_DIR/slurm_logs/${SAFE_NAME}_${ENV_ID}_s${seed}.slurm"

            # 3. Create Training Script
            cat <<EOT > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${SAFE_NAME}_s${seed}_${ENV_ID}
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --output=%j.out

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench
if [ -d "${LOG_ROOT}/${SAFE_NAME}" ]; then
    echo "Warning: Directory exists, deleting to ensure clean restart."
    rm -rf "${LOG_ROOT}/${SAFE_NAME}"
fi
${CMD}
EOT

            echo "Submitting ${EXP_NAME} | env=${ENV_ID} | seed=${seed}..."
            TRAIN_OUTPUT=$(sbatch "$SLURM_SCRIPT")
            TRAIN_JOB_ID=$(echo "$TRAIN_OUTPUT" | awk '{print $4}')
            echo "  Training Job ID: $TRAIN_JOB_ID"

            echo "Run dir is ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}"

            # 4. Create Evaluation Script (points to the SAFE_NAME directory)
            EVAL_SCRIPT="$SCRIPT_DIR/slurm_logs/${SAFE_NAME}_${ENV_ID}_s${seed}_eval.slurm"
            cat <<EOT > "$EVAL_SCRIPT"
#!/bin/bash
#SBATCH --job-name=eval_${SAFE_NAME}_s${seed}
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=%j.out
#SBATCH --dependency=afterany:${TRAIN_JOB_ID}

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench

# Ensure eval output goes where the WandB sync script expects it
mkdir -p ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/logs/eval/
echo "{\"command\": \"${CMD}\"}" > "${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/metadata.json"

# Run the evaluation
python scripts/ppo_builderbench_checkpoint_eval.py \
    --run_dir=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed} \
    --csv_output=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/logs/eval/logs.csv


mkdir -p ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/videos/
python scripts/ppo_builderbench_rollout_video.py \
    --checkpoint=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/checkpoints \
    --env=${ENV_ID} \
    --output=${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}/videos/ \
    --fps=10

# Immediately sync the results of this experiment to WandB
python scripts/csv_runs_to_wandb.py \
    --project dist-matching \
    --entity doina-precup \
    --run-dir ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}
EOT

            sbatch "$EVAL_SCRIPT"
            echo "  Submitted evaluation job dependent on $TRAIN_JOB_ID"

        done  # seeds
    # done  # envs
done  # experiments

echo ""
echo "All jobs submitted."
```

## File: envs/builderbench_env.py
```python
"""Gym adapter for BuilderBench creative-cube tasks (sgcrl / PPO-CRL).

Wraps a single BuilderBench ``CreativeCube`` MJX env (+ optional ``PDWrapper``)
as a classic ``gym.Env`` returning ``[state, target_goal]`` so
``contrastive.utils.make_environment`` can append the BuilderBench task goal.

Supported env ids: ``creative-{N}-task{K}`` (e.g. ``creative-1-task2``).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

import gym
import numpy as np

_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

# Headless MJX rendering defaults (override in shell if needed).
os.environ.setdefault('MUJOCO_GL', 'egl')

_BUILDERBENCH_IMPORT_ERROR: Optional[BaseException] = None
try:
  import jax
  import jax.numpy as jnp
  from builderbench.constants import _MJX_PARAMS
  from builderbench.creative_cube import CreativeCube, default_config
  from utils.wrapper import EpisodeWrapper, PDWrapper
except Exception as _e:  # noqa: BLE001
  jax = None  # type: ignore
  jnp = None  # type: ignore
  CreativeCube = None  # type: ignore
  default_config = None  # type: ignore
  EpisodeWrapper = None  # type: ignore
  PDWrapper = None  # type: ignore
  _BUILDERBENCH_IMPORT_ERROR = _e

from envs.builderbench_utils import (
    default_fixed_target_goal,
    filter_pd_policy_state_obs,
    get_filtered_obs_dim,
    parse_bb_env_id,
    scaled_episode_length,
    validate_pd_episode_length,
    uniform_goal_obs_bounds as _uniform_goal_obs_bounds,
)


def _require_builderbench(env_name: str) -> None:
  if _BUILDERBENCH_IMPORT_ERROR is not None:
    raise RuntimeError(
        f'Cannot build {env_name}: builderbench failed to import.\n'
        f'  Original error: {type(_BUILDERBENCH_IMPORT_ERROR).__name__}: '
        f'{_BUILDERBENCH_IMPORT_ERROR}\n'
        f'Install with: pip install -e {_BUILDERBENCH_ROOT}\n'
        f'And set BUILDERBENCH_ROOT if the repo lives elsewhere.'
    ) from _BUILDERBENCH_IMPORT_ERROR


def parse_creative_env_id(env_id: str) -> tuple[int, int]:
  """Parse ``creative-{N}-task{K}`` ΓåÆ (num_cubes, task_id)."""
  return parse_bb_env_id(env_id)


class BuilderBenchCreativeGymEnv(gym.Env):
  """Single-env Gym wrapper around BuilderBench ``CreativeCube``.

  Observation layout returned to sgcrl:
    ``[ state (obs_dim) | target_goal (goal_dim) ]``

  ``target_goal`` is BuilderBench's ``info['target_goal']`` (absolute cube
  target xyz), *not* a slice of the state observation.
  """

  metadata = {'render_modes': []}

  def __init__(
      self,
      env_id: str = 'creative-1-task1',
      seed: Optional[int] = None,
      use_pd: bool = False,
      pd_duration: int = 5,
      pd_filter_policy_obs: bool = True,
      fixed_target_goal: Optional[np.ndarray] = None,
      obs_space_list: Optional[list[str]] = None,
      episode_length_multiplier: float = 1.0,
  ):
    super().__init__()
    _require_builderbench(env_id)

    self._env_id = env_id
    self._seed = 0 if seed is None else int(seed)
    self._use_pd = bool(use_pd)
    self._pd_duration = int(pd_duration)
    self._pd_filter_policy_obs = bool(pd_filter_policy_obs) and self._use_pd
    self._obs_space_list = obs_space_list or ["xy", "select"]
    self._fixed_target_goal = (
        None if fixed_target_goal is None
        else np.asarray(fixed_target_goal, dtype=np.float32).reshape(-1))

    num_cubes, task_id = parse_creative_env_id(env_id)
    self._num_cubes = num_cubes
    self._task_id = task_id
    cfg = default_config()
    cfg.num_cubes = num_cubes
    cfg.task_id = task_id
    cfg.episode_length = scaled_episode_length(num_cubes, episode_length_multiplier)
    self._episode_length_multiplier = float(episode_length_multiplier)
    # Use JAX MJX backend (no warp-lang required). Set BUILDERBENCH_MJX_IMPL=warp
    # if warp-lang is installed and you want the faster path.
    cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
    if env_id in _MJX_PARAMS:
      cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]

    base = CreativeCube(config=cfg)
    if self._use_pd:
      validate_pd_episode_length(cfg.episode_length, self._pd_duration)
      inner = PDWrapper(base, duration=self._pd_duration)
      episode_length = cfg.episode_length // self._pd_duration
    else:
      inner = base
      episode_length = cfg.episode_length

    self._env = EpisodeWrapper(inner, episode_length=episode_length,
                               action_repeat=1)
    self._base_env = base
    self._episode_length = int(episode_length)
    self._state = None
    self._rng = jax.random.PRNGKey(self._seed)

    # Probe dims with a dry reset.
    probe = self._env.reset(jax.random.PRNGKey(0))
    self._full_state_obs_dim = int(np.asarray(probe.obs).shape[-1])
    if self._pd_filter_policy_obs:
      self._state_obs_dim = get_filtered_obs_dim(self._num_cubes, self._obs_space_list)
    else:
      self._state_obs_dim = self._full_state_obs_dim
    self._goal_dim = int(np.asarray(probe.info['target_goal']).shape[-1])
    self._action_dim = int(self._env.action_size)

    obs_high = np.full(self._state_obs_dim + self._goal_dim, np.inf,
                       dtype=np.float32)
    self.observation_space = gym.spaces.Box(
        low=-obs_high, high=obs_high, dtype=np.float32)
    self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)
    self._max_episode_steps = self._episode_length

  @property
  def state_obs_dim(self) -> int:
    return self._state_obs_dim

  @property
  def goal_dim(self) -> int:
    return self._goal_dim

  @property
  def target_goal(self) -> np.ndarray:
    if self._state is None:
      if self._fixed_target_goal is not None:
        return self._fixed_target_goal.copy()
      return default_fixed_target_goal(self._num_cubes, self._task_id)
    return np.asarray(self._state.info['target_goal'], dtype=np.float32)

  def uniform_goal_obs_bounds(self):
    """Bounds on the goal slice for CRL uniform-sampling negatives."""
    return _uniform_goal_obs_bounds(self._num_cubes, self._task_id)

  def _pack_obs(self, state) -> np.ndarray:
    state_obs = np.asarray(state.obs, dtype=np.float32)
    if self._pd_filter_policy_obs:
      state_obs = filter_pd_policy_state_obs(state_obs, self._num_cubes, self._obs_space_list)
    goal = np.asarray(state.info['target_goal'], dtype=np.float32)
    return np.concatenate([state_obs, goal], axis=0)

  def _maybe_fix_target(self, state):
    if self._fixed_target_goal is None:
      return state
    fixed = jnp.asarray(self._fixed_target_goal, dtype=jnp.float32).reshape(-1)
    num_cubes = self._num_cubes
    fixed_pos = fixed.reshape(num_cubes, 3)
    info = dict(state.info)
    info['target_goal'] = fixed
    mocap_pos = state.data.mocap_pos.at[self._base_env._mocap_targets].set(
        fixed_pos)
    data = state.data.replace(mocap_pos=mocap_pos)
    return state.replace(data=data, info=info)

  def reset(self):
    self._rng, subkey = jax.random.split(self._rng)
    state = self._env.reset(subkey)
    state = self._maybe_fix_target(state)
    self._state = state
    return self._pack_obs(state)

  def step(self, action):
    if self._state is None:
      raise RuntimeError('step() called before reset()')
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    self._state = self._env.step(
        self._state, jnp.asarray(action, dtype=jnp.float32))
    obs = self._pack_obs(self._state)
    reward = float(np.asarray(self._state.reward))
    done = bool(np.asarray(self._state.done))
    info = {
        'success': float(np.asarray(self._state.metrics.get('success', 0.0))),
        'easy_success': float(
            np.asarray(self._state.metrics.get('easy_success', 0.0))),
        'target_goal': np.asarray(
            self._state.info['target_goal'], dtype=np.float32),
        'achieved_goal': np.asarray(
            self._state.info['achieved_goal'], dtype=np.float32),
    }
    if done:
      self._state = None
    return obs, reward, done, info


def make_builderbench_creative_env(
    env_id: str = 'creative-1-task1',
    seed: Optional[int] = None,
    use_pd: bool = False,
    pd_duration: int = 5,
    pd_filter_policy_obs: bool = True,
    fixed_target_goal: Optional[np.ndarray] = None,
    obs_space_list: Optional[list[str]] = None,
    episode_length_multiplier: float = 1.0,
) -> BuilderBenchCreativeGymEnv:
  return BuilderBenchCreativeGymEnv(
      env_id=env_id,
      seed=seed,
      use_pd=use_pd,
      pd_duration=pd_duration,
      pd_filter_policy_obs=pd_filter_policy_obs,
      fixed_target_goal=fixed_target_goal,
      obs_space_list=obs_space_list,
      episode_length_multiplier=episode_length_multiplier
  )
```

## File: scripts/ppo_builderbench_checkpoint_eval.py
```python
#!/usr/bin/env python3
"""Evaluate BuilderBench PPO checkpoints with deterministic rollouts and plot success.

For each milestone checkpoint in a training run, loads the policy and runs
``num_eval_episodes`` batched eval rollouts using the deterministic policy
mean (``dist.mode()``), matching training-time BuilderBench eval.

Reads ``run_config.json`` beside the checkpoint directory so PD settings, obs
packing, fixed goals, and network sizes match training.

Examples:
  python scripts/ppo_builderbench_checkpoint_eval.py \\
      --run_dir=logs/ppo_builderbench_creative3_task1_e1024_pd/ppo_builderbench_creative_3_task1_0

  # Incremental: eval only checkpoints missing from the CSV, then replot.
  python scripts/ppo_builderbench_checkpoint_eval.py \\
      --run_dir=logs/.../ppo_builderbench_creative_3_task1_0 \\
      --incremental --csv_output=figs/builderbench/checkpoint_eval/run.csv \\
      --output=figs/builderbench/run_eval_success.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401 — must precede acme/jax imports

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import contrastive
from contrastive import ppo_learner
from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv
from envs.builderbench_utils import is_builderbench_creative_env

CSV_FIELDS = [
    'label', 'path', 'iteration', 'global_step',
    'success_mean', 'success_std', 'episode_successes',
]


def _load_bb_video_helpers():
  path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_rollout_video.py')
  spec = importlib.util.spec_from_file_location('bb_video', path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


_bb_video = _load_bb_video_helpers()
_TrainCtx = _bb_video._TrainCtx
_load_train_ctx = _bb_video._load_train_ctx
_build_networks = _bb_video._build_networks
_enumerate_checkpoints = _bb_video._enumerate_checkpoints


@dataclass
class CheckpointEvalResult:
  label: str
  path: str
  iteration: int
  global_step: int
  success_mean: float
  success_std: float
  episode_successes: Tuple[float, ...]


def resolve_paths(
    run_dir: Optional[str],
    checkpoint_dir: Optional[str],
) -> Tuple[str, str]:
  if run_dir:
    run_dir = os.path.abspath(run_dir)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
      raise FileNotFoundError(f'No checkpoints/ under run_dir: {run_dir!r}')
    return run_dir, ckpt_dir
  if checkpoint_dir:
    ckpt_dir = os.path.abspath(checkpoint_dir)
    run_dir = os.path.dirname(ckpt_dir)
    return run_dir, ckpt_dir
  raise ValueError('Provide run_dir or checkpoint_dir')


def read_run_config(run_dir: str) -> Dict[str, Any]:
  path = os.path.join(run_dir, 'run_config.json')
  if not os.path.isfile(path):
    raise FileNotFoundError(f'run_config.json not found: {path!r}')
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def env_from_run_config(run_cfg: Dict[str, Any], env_arg: Optional[str]) -> str:
  if env_arg:
    return str(env_arg)
  env = str(run_cfg.get('env', ''))
  if not env:
    raise ValueError('Could not infer env; pass --env')
  return env


def ckpt_label(pkl_path: str) -> str:
  base = os.path.splitext(os.path.basename(pkl_path))[0]
  m = re.search(r'ckpt_iter_(\d+)\.pkl$', pkl_path)
  if m:
    return f'iter_{int(m.group(1)):07d}'
  if base == 'latest':
    return 'latest'
  return base[len('ckpt_'):] if base.startswith('ckpt_') else base


def list_checkpoint_files(
    ckpt_dir: str,
    *,
    include_latest: bool = False,
) -> List[Tuple[str, str, float]]:
  """Return [(label, path, mtime), ...] sorted by iteration."""
  if not os.path.isdir(ckpt_dir):
    return []
  out: List[Tuple[str, str, float]] = []
  for path in sorted(glob.glob(os.path.join(ckpt_dir, 'ckpt_iter_*.pkl'))):
    if os.path.isfile(path):
      out.append((ckpt_label(path), path, os.path.getmtime(path)))
  if include_latest:
    latest = os.path.join(ckpt_dir, 'latest.pkl')
    if os.path.isfile(latest):
      out.append(('latest', latest, os.path.getmtime(latest)))
  out.sort(key=lambda x: _iteration_from_label(x[0]))
  return out


def _iteration_from_label(label: str) -> int:
  m = re.search(r'iter_(\d+)', label)
  if m:
    return int(m.group(1))
  if label == 'latest':
    return 10**12
  return -1


def read_csv_results(csv_path: str) -> List[CheckpointEvalResult]:
  if not os.path.isfile(csv_path):
    return []
  rows: List[CheckpointEvalResult] = []
  with open(csv_path, 'r', newline='', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      ep_raw = str(row.get('episode_successes', '') or '')
      ep = tuple(float(x) for x in ep_raw.split(';') if x.strip())
      rows.append(CheckpointEvalResult(
          label=str(row['label']),
          path=str(row['path']),
          iteration=int(row['iteration']),
          global_step=int(row['global_step']),
          success_mean=float(row['success_mean']),
          success_std=float(row['success_std']),
          episode_successes=ep,
      ))
  rows.sort(key=lambda r: r.iteration)
  return rows


def write_csv_results(path: str, results: Sequence[CheckpointEvalResult]) -> None:
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  ordered = sorted(results, key=lambda r: r.iteration)
  with open(path, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for r in ordered:
      writer.writerow({
          'label': r.label,
          'path': r.path,
          'iteration': r.iteration,
          'global_step': r.global_step,
          'success_mean': r.success_mean,
          'success_std': r.success_std,
          'episode_successes': ';'.join(f'{x:.0f}' for x in r.episode_successes),
      })


def merge_results(
    existing: Sequence[CheckpointEvalResult],
    new_rows: Sequence[CheckpointEvalResult],
) -> List[CheckpointEvalResult]:
  by_key: Dict[str, CheckpointEvalResult] = {}
  for r in existing:
    by_key[r.label] = r
  for r in new_rows:
    by_key[r.label] = r
  return sorted(by_key.values(), key=lambda r: r.iteration)


def episode_successes_from_steps(steps) -> np.ndarray:
  success = np.asarray(steps['success'], dtype=np.float32)
  return (np.max(success, axis=0) >= 0.5).astype(np.float32)


class CheckpointEvalSession:
  """Reusable eval session (networks + vec env + jitted unroll)."""

  def __init__(
      self,
      env_name: str,
      ctx: _TrainCtx,
      run_cfg: Dict[str, Any],
      *,
      num_eval_episodes: int = 20,
      eval_seed: int = 0,
      network_seed: int = 0,
  ):
    self.env_name = env_name
    self.ctx = ctx
    self.num_eval_episodes = int(num_eval_episodes)
    fixed_goal = ctx.fixed_target_goal
    if fixed_goal is None and run_cfg.get('fixed_start_end') is not None:
      fixed_goal = np.asarray(run_cfg['fixed_start_end'], dtype=np.float32)

    self.networks = _build_networks(env_name, seed=network_seed, ctx=ctx)
    self.vec_env = JaxBuilderBenchVecEnv(
        env_name=env_name,
        num_envs=self.num_eval_episodes,
        seed=int(eval_seed),
        use_pd=ctx.use_pd,
        pd_duration=ctx.pd_duration,
        pd_filter_policy_obs=ctx.filter_policy_obs,
        fixed_target_goal=fixed_goal,
        obs_space_list=ctx.obs_space_list,
        episode_length_multiplier=ctx.episode_length_multiplier
    )
    print(f'[bb_eval] eval session: env={env_name}  num_envs={self.num_eval_episodes}  '
          f'ep_len={self.vec_env.episode_length}  obs_dim={self.vec_env.obs_dim}  act_dim={self.vec_env.act_dim}  '
          f'use_pd={ctx.use_pd}  filter_policy_obs={ctx.filter_policy_obs}  pd_duration={ctx.pd_duration}  fixed_goal={fixed_goal}  ')

    @jax.jit
    def eval_policy_action(policy_p, obs):
      dist = self.networks.policy_network.apply(policy_p, obs)
      return dist.mode()

    self._eval_unroll = self.vec_env.compile_eval_unroll(
        eval_policy_action,
        unroll_length=self.vec_env.episode_length,
    )

  def eval_policy_params(self, policy_params: Any) -> Tuple[float, float, Tuple[float, ...]]:
    eval_state = self.vec_env.reset_state()
    steps = self._eval_unroll(eval_state, policy_params)
    ep_success = episode_successes_from_steps(steps)
    mean = float(ep_success.mean())
    n = int(ep_success.size)
    std = float(math.sqrt(mean * (1.0 - mean) / max(n, 1)))
    return mean, std, tuple(float(x) for x in ep_success)

  def eval_checkpoint_file(self, label: str, path: str) -> CheckpointEvalResult:
    ckpt = ppo_learner.load_checkpoint(path)
    iteration = int(ckpt.get('iteration', _iteration_from_label(label)))
    global_step = int(ckpt.get('global_step', iteration))
    mean, std, ep_succ = self.eval_policy_params(ckpt['policy_params'])
    return CheckpointEvalResult(
        label=label,
        path=path,
        iteration=iteration,
        global_step=global_step,
        success_mean=mean,
        success_std=std,
        episode_successes=ep_succ,
    )


def format_steps(v, _):
  if v == 0:
    return '0'
  if v >= 1e6:
    s = f'{v / 1e6:.1f}M'
    return s.replace('.0M', 'M')
  if v >= 1e3:
    return f'{v / 1e3:.0f}K'
  return str(int(v))


def plot_results(
    results: Sequence[CheckpointEvalResult],
    *,
    title: str,
    output_path: str,
    x_axis: str = 'global_step',
) -> None:
  if not results:
    print('[bb_eval] no rows to plot')
    return

  if x_axis == 'iteration':
    xs = [r.iteration for r in results]
    xlabel = 'Training iteration'
  else:
    xs = [r.global_step for r in results]
    xlabel = 'Global env steps'

  ys = [r.success_mean for r in results]
  yerr = [r.success_std for r in results]

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.errorbar(
      xs, ys, yerr=yerr, fmt='-o', color='#4C9BE8', linewidth=1.8,
      markersize=4, capsize=3, elinewidth=1.0,
      label='eval success (mean ± SE)')
  ax.set_title(title, fontsize=13, fontweight='bold')
  ax.set_xlabel(xlabel, fontsize=11)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_steps))
  ax.set_ylabel('Eval success rate', fontsize=11)
  ax.set_ylim(-0.02, 1.02)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  ax.legend(loc='best', fontsize=9, framealpha=0.9)

  out_dir = os.path.dirname(os.path.abspath(output_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fig.tight_layout()
  fig.savefig(output_path, dpi=150, bbox_inches='tight')
  plt.close(fig)


def default_output_paths(run_dir: str, plot_tag: Optional[str] = None):
  run_name = os.path.basename(run_dir.rstrip(os.sep))
  tag = plot_tag or run_name
  csv_path = os.path.join(
      _REPO, 'figs', 'builderbench', 'checkpoint_eval',
      f'{run_name}_checkpoint_success.csv')
  plot_path = os.path.join(
      _REPO, 'figs', 'builderbench', f'{tag}_checkpoint_success.png')
  return plot_path, csv_path


def eval_run(
    run_dir: str,
    env_name: str,
    *,
    num_eval_episodes: int = 20,
    eval_seed: int = 0,
    network_seed: int = 0,
    include_latest: bool = False,
    incremental: bool = False,
    plot_only: bool = False,
    csv_path: Optional[str] = None,
    plot_path: Optional[str] = None,
    plot_tag: Optional[str] = None,
    x_axis: str = 'global_step',
    max_checkpoints: int = -1,
    only_labels: Optional[Sequence[str]] = None,
) -> Tuple[List[CheckpointEvalResult], str, str]:
  """Eval (optionally incremental), write CSV, plot. Returns (results, csv, png)."""
  run_dir = os.path.abspath(run_dir)
  ckpt_dir = os.path.join(run_dir, 'checkpoints')
  run_cfg = read_run_config(run_dir)
  default_plot, default_csv = default_output_paths(run_dir, plot_tag=plot_tag)
  plot_path = plot_path or default_plot
  csv_path = csv_path or default_csv

  existing = read_csv_results(csv_path) if incremental else []
  done_labels = {r.label for r in existing}

  if plot_only:
    results = existing
    title = f'BuilderBench eval success — {env_name} ({os.path.basename(run_dir)})'
    plot_results(results, title=title, output_path=plot_path, x_axis=x_axis)
    return results, csv_path, plot_path

  ckpt_files = list_checkpoint_files(ckpt_dir, include_latest=include_latest)
  if only_labels is not None:
    allow = set(only_labels)
    ckpt_files = [x for x in ckpt_files if x[0] in allow]
  if incremental:
    ckpt_files = [x for x in ckpt_files if x[0] not in done_labels]
  if max_checkpoints > 0:
    ckpt_files = ckpt_files[:max_checkpoints]

  ctx = _load_train_ctx(env_name, ckpt_dir)
  print(f'[bb_eval] env={env_name}  pending={len(ckpt_files)}  '
        f'existing={len(existing)}  eval_episodes={num_eval_episodes}')
  print(f'[bb_eval] use_pd={ctx.use_pd}  filter_policy_obs={ctx.filter_policy_obs}  '
        f'obs_dim={ctx.obs_dim}  ep_len={ctx.episode_length}')

  new_results: List[CheckpointEvalResult] = []
  if ckpt_files:
    session = CheckpointEvalSession(
        env_name, ctx, run_cfg,
        num_eval_episodes=num_eval_episodes,
        eval_seed=eval_seed,
        network_seed=network_seed,
    )
    for i, (label, path, _mtime) in enumerate(ckpt_files):
      print(f'[bb_eval] ({i + 1}/{len(ckpt_files)}) {label}  ({path})')
      result = session.eval_checkpoint_file(label, path)
      print(f'[bb_eval]   iter={result.iteration}  step={result.global_step}  '
            f'success={result.success_mean:.3f} ± {result.success_std:.3f}')
      new_results.append(result)

  results = merge_results(existing, new_results)
  if results:
    write_csv_results(csv_path, results)
    print(f'[bb_eval] wrote csv: {csv_path}  ({len(results)} row(s))')
  title = f'BuilderBench eval success — {env_name} ({os.path.basename(run_dir)})'
  plot_results(results, title=title, output_path=plot_path, x_axis=x_axis)
  if results:
    print(f'[bb_eval] wrote plot: {plot_path}')
  return results, csv_path, plot_path


def discover_run_dirs(log_dir: str, env: str, seeds: Optional[set[int]] = None):
  """Yield (seed, run_dir, ckpt_dir) under logs/<log_dir>/ppo_<env>_<seed>/."""
  base = os.path.abspath(log_dir)
  if not os.path.isdir(base):
    return
  pattern = os.path.join(base, f'ppo_{env}_*')
  for run_dir in sorted(glob.glob(pattern)):
    if not os.path.isdir(run_dir):
      continue
    run_name = os.path.basename(run_dir)
    m = re.fullmatch(rf'ppo_{re.escape(env)}_(\d+)', run_name)
    if not m:
      continue
    seed = int(m.group(1))
    if seeds is not None and seed not in seeds:
      continue
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    yield seed, run_dir, ckpt_dir


def main():
  parser = argparse.ArgumentParser(
      description='Deterministic BuilderBench checkpoint eval + success plot')
  parser.add_argument('--run_dir',
                      help='Training run dir (run_config.json + checkpoints/).')
  parser.add_argument('--checkpoint_dir',
                      help='Checkpoint dir (parent must contain run_config.json).')
  parser.add_argument('--env', default=None)
  parser.add_argument('--num_eval_episodes', type=int, default=20)
  parser.add_argument('--eval_seed', type=int, default=0)
  parser.add_argument('--output', default=None, help='Output PNG path.')
  parser.add_argument('--csv_output', default=None, help='Output CSV path.')
  parser.add_argument('--plot_tag', default=None, help='Plot filename stem override.')
  parser.add_argument('--x_axis', choices=('global_step', 'iteration'),
                      default='global_step')
  parser.add_argument('--include_latest', action='store_true')
  parser.add_argument('--incremental', action='store_true',
                      help='Skip checkpoints already present in the CSV.')
  parser.add_argument('--plot_only', action='store_true',
                      help='Replot from CSV without running eval.')
  parser.add_argument('--max_checkpoints', type=int, default=-1)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  if not args.run_dir and not args.checkpoint_dir:
    parser.error('Provide --run_dir or --checkpoint_dir')
  run_dir, _ = resolve_paths(args.run_dir, args.checkpoint_dir)
  run_cfg = read_run_config(run_dir)
  env_name = env_from_run_config(run_cfg, args.env)
  if not is_builderbench_creative_env(env_name):
    parser.error(f'Not a builderbench creative env: {env_name!r}')

  eval_run(
      run_dir,
      env_name,
      num_eval_episodes=args.num_eval_episodes,
      eval_seed=args.eval_seed,
      network_seed=args.seed,
      include_latest=args.include_latest,
      incremental=args.incremental,
      plot_only=args.plot_only,
      csv_path=args.csv_output,
      plot_path=args.output,
      plot_tag=args.plot_tag,
      x_axis=args.x_axis,
      max_checkpoints=args.max_checkpoints,
  )


if __name__ == '__main__':
  main()
```

## File: envs/builderbench_jax_vec.py
```python
"""Batched JAX vec-env for BuilderBench (sgcrl PPO-CRL).

Uses the same wrapper stack as ``builderbench/utils/wrapper.py::wrap_env``
(``VmapWrapper`` + ``EpisodeWrapper`` + ``AutoResetWrapper``) but exposes the
``VecEnv`` API expected by ``contrastive.ppo_learner``:

    obs = vec_env.reset()                         # (E, obs_dim_total)
    next_obs, rew, dones, terminal_obs, info_rew = vec_env.step(actions)

Training logic (PPO, CRL, GAE) stays unchanged; only rollout collection is
accelerated via a single ``jax.jit`` batched env step on GPU.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

os.environ.setdefault('MUJOCO_GL', 'egl')

_BUILDERBENCH_IMPORT_ERROR: Optional[BaseException] = None
try:
  import jax
  import jax.numpy as jnp
  from builderbench.constants import _MJX_PARAMS
  from builderbench.creative_cube import CreativeCube, default_config
  from builderbench.env_utils import State
  from utils.wrapper import (
      AutoResetWrapper,
      EpisodeWrapper,
      PDWrapper,
      VmapWrapper,
      Wrapper,
  )
  from envs.builderbench_utils import (
      filter_pd_policy_state_obs,
      get_filtered_obs_dim,
      parse_bb_env_id,
      scaled_episode_length,
      validate_pd_episode_length,
      sgcrl_env_name_to_bb_env_id,
      uniform_goal_obs_bounds as _uniform_goal_obs_bounds,
  )
except Exception as _e:  # noqa: BLE001
  jax = None  # type: ignore
  jnp = None  # type: ignore
  CreativeCube = None  # type: ignore
  default_config = None  # type: ignore
  State = None  # type: ignore
  AutoResetWrapper = None  # type: ignore
  EpisodeWrapper = None  # type: ignore
  PDWrapper = None  # type: ignore
  VmapWrapper = None  # type: ignore
  Wrapper = None  # type: ignore
  _BUILDERBENCH_IMPORT_ERROR = _e


def _require_builderbench(env_name: str) -> None:
  if _BUILDERBENCH_IMPORT_ERROR is not None:
    raise RuntimeError(
        f'Cannot build {env_name}: builderbench failed to import.\n'
        f'  Original error: {type(_BUILDERBENCH_IMPORT_ERROR).__name__}: '
        f'{_BUILDERBENCH_IMPORT_ERROR}\n'
        f'Install with: pip install -e {_BUILDERBENCH_ROOT}\n'
        f'And set BUILDERBENCH_ROOT if the repo lives elsewhere.'
    ) from _BUILDERBENCH_IMPORT_ERROR


from envs.builderbench_utils import (
    filter_pd_policy_state_obs,
    get_filtered_obs_dim,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
    uniform_goal_obs_bounds as _uniform_goal_obs_bounds,
)


class TerminalObsWrapper(Wrapper):
  """Record pre-autoreset observations so CRL can store true episode terminals."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    info = dict(state.info)
    info['terminal_obs'] = state.obs
    info['terminal_target_goal'] = state.info['target_goal']
    return state.replace(info=info)

  def step(self, state: State, action: jax.Array) -> State:
    state = self.env.step(state, action)
    info = dict(state.info)
    info['terminal_obs'] = state.obs
    info['terminal_target_goal'] = state.info['target_goal']
    return state.replace(info=info)


def _wrap_batched_env(env, episode_length: int):
  env = VmapWrapper(env)
  env = EpisodeWrapper(env, episode_length=episode_length, action_repeat=1)
  env = TerminalObsWrapper(env)
  env = AutoResetWrapper(env)
  return env


def _pack_obs(
    state_obs: jax.Array,
    target_goal: jax.Array,
    *,
    num_cubes: int = 0,
    filter_pd_policy: bool = False,
    obs_space_list: Optional[list[str]] = None,
) -> jax.Array:
  if filter_pd_policy:
    state_obs = filter_pd_policy_state_obs(state_obs, num_cubes, obs_space_list or ["xy", "select"])
  return jnp.concatenate([state_obs, target_goal], axis=-1)


def _maybe_fix_target(
    state: State,
    fixed_target_goal: Optional[jax.Array],
    mocap_targets: jax.Array,
    num_cubes: int,
) -> State:
  if fixed_target_goal is None:
    return state
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(num_cubes, 3)
  goal = jnp.broadcast_to(fixed, state.info['target_goal'].shape)
  info = dict(state.info)
  info['target_goal'] = goal
  mocap_pos = state.data.mocap_pos.at[mocap_targets].set(fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


class JaxBuilderBenchVecEnv:
  """GPU-batched BuilderBench env with the ``ppo_learner.VecEnv`` API."""

  def __init__(
      self,
      env_name: str,
      num_envs: int,
      seed: int,
      use_pd: bool = False,
      pd_duration: int = 5,
      pd_filter_policy_obs: bool = True,
      fixed_target_goal: Optional[np.ndarray] = None,
      obs_space_list: Optional[list[str]] = None,
    episode_length_multiplier: float = 1.0,

  ):
    _require_builderbench(env_name)
    self._env_name = str(env_name)
    self._bb_env_id = sgcrl_env_name_to_bb_env_id(env_name)
    self._num_cubes, self._task_id = parse_bb_env_id(self._bb_env_id)
    self._num_envs = int(num_envs)
    self._seed = int(seed)
    self._use_pd = bool(use_pd)
    self._pd_duration = int(pd_duration)
    self._pd_filter_policy_obs = bool(pd_filter_policy_obs) and self._use_pd
    self._fixed_target_goal = (
        None if fixed_target_goal is None
        else jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1))

    num_cubes, task_id = self._num_cubes, self._task_id
    cfg = default_config()
    cfg.num_cubes = num_cubes
    cfg.task_id = task_id
    cfg.episode_length = scaled_episode_length(
              num_cubes, episode_length_multiplier)
    self._episode_length_multiplier = float(episode_length_multiplier)
    cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
    if self._bb_env_id in _MJX_PARAMS:
      ncon, njmax = _MJX_PARAMS[self._bb_env_id]
      cfg.nconmax = int(ncon) * self._num_envs
      cfg.njmax = int(njmax)

    base = CreativeCube(config=cfg)
    self._mocap_targets = base._mocap_targets
    if self._use_pd:
      validate_pd_episode_length(cfg.episode_length, self._pd_duration)
      inner = PDWrapper(base, duration=self._pd_duration)
      episode_length = cfg.episode_length // self._pd_duration
    else:
      inner = base
      episode_length = cfg.episode_length

    self._env = _wrap_batched_env(inner, episode_length=int(episode_length))
    self._episode_length = int(episode_length)
    self._obs_space_list = obs_space_list or ["xy", "select"]

    probe_key = jax.random.split(jax.random.PRNGKey(0), self._num_envs)
    probe = self._env.reset(probe_key)
    self._full_state_obs_dim = int(probe.obs.shape[-1])
    if self._pd_filter_policy_obs:
      self._state_obs_dim = get_filtered_obs_dim(self._num_cubes, self._obs_space_list)
    else:
      self._state_obs_dim = self._full_state_obs_dim
    self._goal_dim = int(probe.info['target_goal'].shape[-1])
    self._action_dim = int(self._env.action_size)
    self._obs_dim_total = self._state_obs_dim + self._goal_dim

    self._rng = jax.random.PRNGKey(self._seed)
    self._state: Optional[State] = None

    self._reset_fn = jax.jit(self._reset_impl)
    self._step_fn = jax.jit(self._step_impl)

    _pd_obs_msg = (
        f' pd_policy_obs={self._state_obs_dim}'
        f' (full_state={self._full_state_obs_dim})'
        if self._pd_filter_policy_obs else '')
    print(f'[jax_vec] BuilderBench {self._bb_env_id}: '
            f'E={self._num_envs} obs={self._obs_dim_total} '
            f'act={self._action_dim} ep_len={self._episode_length} '
            f'use_pd={self._use_pd}{_pd_obs_msg} '
            f'ep_len_mult={self._episode_length_multiplier}')

  def _reset_impl(self, rng: jax.Array) -> State:
    state = self._env.reset(rng)
    return _maybe_fix_target(
        state, self._fixed_target_goal, self._mocap_targets, self._num_cubes)

  def _step_impl(self, state: State, actions: jax.Array) -> State:
    actions = jnp.clip(actions, -1.0, 1.0)
    state = self._env.step(state, actions)
    return _maybe_fix_target(
        state, self._fixed_target_goal, self._mocap_targets, self._num_cubes)

  def uniform_goal_obs_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    return _uniform_goal_obs_bounds(self._num_cubes, self._task_id)

  def pack_obs_from_state(self, state: State) -> jnp.ndarray:
    return _pack_obs(
        state.obs,
        state.info['target_goal'],
        num_cubes=self._num_cubes,
        filter_pd_policy=self._pd_filter_policy_obs,
        obs_space_list=self._obs_space_list,
    )

  def compile_generate_unroll(
      self,
      act_and_value_fn: Callable,
      unroll_length: int,
      obs_dim: int,
  ):
    """Build a ``jax.lax.scan`` rollout collector (BuilderBench-style).

    ``act_and_value_fn(policy_p, value_p, packed_obs, key)``
    must return ``(actions, logprobs, values)`` ΓÇö same contract as
    ``ppo_learner.act_and_value``.
    """
    step_fn = self._step_fn
    _obs_dim = int(obs_dim)
    _T = int(unroll_length)
    _num_cubes = self._num_cubes
    _filter_pd = self._pd_filter_policy_obs
    _obs_space_list = self._obs_space_list

    @jax.jit
    def generate_unroll(
        env_state: State,
        policy_params: Any,
        value_params: Any,
        key: jax.Array,
        next_done_init: jax.Array,
        s0_states_init: jax.Array,
    ):
      def f(carry, _):
        env_state, key, next_done, s0_states = carry
        packed_obs = _pack_obs(
            env_state.obs, env_state.info['target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        key, k_act = jax.random.split(key)
        actions, logprobs, values = act_and_value_fn(
            policy_params, value_params, packed_obs, k_act)
        next_state = step_fn(env_state, actions)

        terminal_obs = _pack_obs(
            next_state.info['terminal_obs'],
            next_state.info['terminal_target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        next_packed = _pack_obs(
            next_state.obs, next_state.info['target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        dones = next_state.done
        term_obs = jnp.where(dones[:, None], terminal_obs, next_packed)

        step_out = {
            'obs': packed_obs,
            'roll_dones': next_done,
            'actions': actions,
            'logprobs': logprobs,
            'values': values,
            'env_rew': next_state.reward,
            'success': next_state.metrics['success'],
            'step_dones': dones.astype(jnp.float32),
            'terminal_obs': term_obs,
            'next_obs': next_packed,
            's0_states': s0_states,
        }
        s0_next = jnp.where(
            dones[:, None], next_packed[:, :_obs_dim], s0_states)
        next_done_out = dones.astype(jnp.float32)
        return (next_state, key, next_done_out, s0_next), step_out

      (final_state, key, final_next_done, final_s0), steps = jax.lax.scan(
          f,
          (env_state, key, next_done_init, s0_states_init),
          (),
          length=_T,
      )
      return (final_state, key, final_next_done, final_s0), steps

    return generate_unroll

  def compile_eval_unroll(
      self,
      eval_policy_fn: Callable,
      unroll_length: int,
  ):
    """Batched eval rollout: deterministic policy + env scan on GPU.

    ``eval_policy_fn(policy_params, packed_obs)`` must return actions in
    ``[-1, 1]`` with batch shape ``(E, act_dim)``.
    """
    step_fn = self._step_fn
    _T = int(unroll_length)
    _num_cubes = self._num_cubes
    _filter_pd = self._pd_filter_policy_obs
    _obs_space_list = self._obs_space_list

    @jax.jit
    def eval_unroll(env_state: State, policy_params: Any):
      def f(carry, _):
        env_state = carry
        packed_obs = _pack_obs(
            env_state.obs, env_state.info['target_goal'],
            num_cubes=_num_cubes, filter_pd_policy=_filter_pd, obs_space_list=_obs_space_list)
        actions = eval_policy_fn(policy_params, packed_obs)
        next_state = step_fn(env_state, actions)
        step_out = {
            'reward': next_state.reward,
            'success': next_state.metrics['success'],
            'state_obs': next_state.obs,
            'goal': next_state.info['target_goal'],
        }
        return next_state, step_out

      _, steps = jax.lax.scan(f, env_state, (), length=_T)
      return steps

    return eval_unroll

  @property
  def episode_length(self) -> int:
    return self._episode_length

  def reset_state(self) -> State:
    """Reset and return the internal JAX env state (for eval scans)."""
    self._rng, subkey = jax.random.split(self._rng)
    keys = jax.random.split(subkey, self._num_envs)
    self._state = self._reset_fn(keys)
    return self._state

  def reset(self) -> np.ndarray:
    self._rng, subkey = jax.random.split(self._rng)
    keys = jax.random.split(subkey, self._num_envs)
    self._state = self._reset_fn(keys)
    packed = self.pack_obs_from_state(self._state)
    return np.asarray(packed, dtype=np.float32)

  def step(
      self, actions: np.ndarray,
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if self._state is None:
      raise RuntimeError('step() called before reset()')

    actions = np.asarray(actions, dtype=np.float32)
    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = np.clip(actions, -1.0, 1.0)

    self._state = self._step_fn(self._state, jnp.asarray(actions))

    next_obs = self.pack_obs_from_state(self._state)
    terminal_obs = _pack_obs(
        self._state.info['terminal_obs'],
        self._state.info['terminal_target_goal'],
        num_cubes=self._num_cubes,
        filter_pd_policy=self._pd_filter_policy_obs,
        obs_space_list=self._obs_space_list,
    )
    dones = self._state.done.astype(bool)
    env_rewards = self._state.reward.astype(np.float32)
    info_rewards = np.full(self._num_envs, np.nan, dtype=np.float32)

    next_np = np.asarray(next_obs, dtype=np.float32)
    term_np = np.asarray(terminal_obs, dtype=np.float32)
    # Match VecEnv: terminal_obs == next_obs when the episode did not end.
    term_np = np.where(dones[:, None], term_np, next_np)

    return (
        next_np,
        np.asarray(env_rewards, dtype=np.float32),
        np.asarray(dones, dtype=bool),
        term_np,
        info_rewards,
    )

  @property
  def num_envs(self) -> int:
    return self._num_envs

  @property
  def observation_shape(self) -> Tuple[int, ...]:
    return (self._obs_dim_total,)

  @property
  def action_shape(self) -> Tuple[int, ...]:
    return (self._action_dim,)

  def stagger_resets(self):
    """Randomly advances environments by t ~ Uniform(0, episode_length) to desynchronize rollouts."""
    if self._state is None:
      self.reset_state()

    self._rng, k_offset, k_scan = jax.random.split(self._rng, 3)
    offsets = jax.random.randint(
        k_offset, (self._num_envs,), 0, self._episode_length)

    @jax.jit
    def _warmup_scan(carry, step_idx):
      state, rng = carry
      rng, a_key = jax.random.split(rng)

      # Sample random actions to progress the environment
      actions = jax.random.uniform(
          a_key, (self._num_envs, self._action_dim), minval=-1.0, maxval=1.0)
      next_state = self._step_impl(state, actions)

      # Mask: Only accept the next_state if this env hasn't reached its target offset yet
      mask = step_idx < offsets
      state = jax.tree_util.tree_map(
          lambda ns, s: jnp.where(mask.reshape(
              (-1,) + (1,) * (ns.ndim - 1)), ns, s),
          next_state, state
      )
      return (state, rng), None

    (self._state, _), _ = jax.lax.scan(_warmup_scan,
                                       (self._state, k_scan), jnp.arange(self._episode_length))
    print(
        f'[jax_vec] Staggered resets applied (max offset: {self._episode_length})')
```

## File: envs/builderbench_utils.py
```python
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
```

## File: scripts/ppo_builderbench_rollout_video.py
```python
"""Render BuilderBench rollout videos from sgcrl PPO-CRL checkpoints.

Matches training setup by reading ``run_config.json`` next to the checkpoint
(PD wrapper, macro episode length, obs packing / PD policy-obs filter, fixed
goals, network sizes).

Examples:
  python scripts/ppo_builderbench_rollout_video.py \\
      --checkpoint=logs/ppo_builderbench_creative1_task1/.../checkpoints/latest.pkl \\
      --output=videos/ppo_builderbench_creative1_task1/
"""
import json
import os
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401 ΓÇö must precede acme/jax imports

import argparse
import glob
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from acme import specs

import contrastive
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from ppo_contrastive import fixed_goal_for_env, ppo_env_defaults_for_env
from envs.builderbench_utils import (
    creative_cube_full_state_obs_dim,
    filter_pd_policy_state_obs,
    get_filtered_obs_dim,
    is_builderbench_creative_env,
    parse_bb_env_id,
    pd_policy_state_obs_dim,
    scaled_episode_length,
    sgcrl_env_name_to_bb_env_id,
    video_render_skip_reason,
)

from builderbench.constants import _MJX_PARAMS
from builderbench.creative_cube import CreativeCube, default_config
from utils.wrapper import (
    AutoResetWrapper,
    EpisodeWrapper,
    PDWrapper,
    VmapWrapper,
)


@dataclass
class _TrainCtx:
  use_pd: bool
  pd_duration: int
  filter_policy_obs: bool
  episode_length: int
  obs_dim: int
  hidden_layer_sizes: Tuple[int, ...]
  fixed_target_goal: Optional[np.ndarray]
  actor_min_std: float
  start_index: int
  end_index: int
  obs_space_list: list[str]
  episode_length_multiplier: float = 1.0


def _get_video(
    inference_policy,
    video_env,
    video_key,
    episode_length: int,
    *,
    fixed_target_goal: Optional[np.ndarray],
    mocap_targets,
    num_cubes: int,
):
  """Roll out one episode and render frames (matches training PD macro steps)."""
  video_env_states = _get_trajectory(
      inference_policy,
      video_env,
      video_key,
      episode_length,
      fixed_target_goal=fixed_target_goal,
      mocap_targets=mocap_targets,
      num_cubes=num_cubes,
  )
  mocap_key = 'target_mocap'
  video_images = []
  for i in range(episode_length):
    if i % 2 == 0:
      video_images.append(video_env.render_from_info(
          np.asarray(video_env_states.data.qpos[i][0]),
          np.asarray(video_env_states.data.qvel[i][0]),
          np.asarray(video_env_states.info[f'{mocap_key}_pos'][i][0]),
          np.asarray(video_env_states.info[f'{mocap_key}_quat'][i][0]),
      ))
  return video_images


def _maybe_fix_target(
    state,
    fixed_target_goal: Optional[np.ndarray],
    mocap_targets,
    num_cubes: int,
):
  if fixed_target_goal is None:
    return state
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(num_cubes, 3)
  info = dict(state.info)
  info['target_goal'] = jnp.broadcast_to(
      fixed, state.info['target_goal'].shape)
  mocap_pos = state.data.mocap_pos.at[mocap_targets].set(fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


def _get_trajectory(
    policy,
    env,
    key,
    unroll_length: int,
    *,
    fixed_target_goal: Optional[np.ndarray],
    mocap_targets,
    num_cubes: int,
):
  @jax.jit
  def _run(key):
    env_key, key = jax.random.split(key)
    state = env.reset(jax.random.split(env_key, 1))
    state = _maybe_fix_target(
        state, fixed_target_goal, mocap_targets, num_cubes)

    def f(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      action, _ = policy(state.obs, state.info['target_goal'], act_key)
      next_state = env.step(state, action)
      return (next_state, key), next_state

    _, states = jax.lax.scan(f, (state, key), (), length=unroll_length)
    return states

  return _run(key)


def _run_config_path_for_checkpoint(checkpoint_path: str) -> Optional[str]:
  if os.path.isfile(checkpoint_path):
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
  else:
    ckpt_dir = os.path.abspath(checkpoint_path)
  run_dir = os.path.dirname(ckpt_dir)
  path = os.path.join(run_dir, 'run_config.json')
  return path if os.path.isfile(path) else None


def _load_train_ctx(env_name: str, checkpoint_path: str) -> _TrainCtx:
  """Infer training settings from run_config.json or env defaults."""
  num_cubes, _ = parse_bb_env_id(sgcrl_env_name_to_bb_env_id(env_name))
  full_obs_dim = creative_cube_full_state_obs_dim(num_cubes)

  cfg_path = _run_config_path_for_checkpoint(checkpoint_path)
  if cfg_path is not None:
    with open(cfg_path, 'r', encoding='utf-8') as fh:
      run_cfg: Dict[str, Any] = json.load(fh)
    flags = run_cfg.get('flags', {})
    obs_space_str = flags.get('obs_space', 'xy,select')
    obs_space_list = [s.strip() for s in obs_space_str.split(',')]
    ep_mult = float(flags.get('builderbench_episode_length_multiplier', 1.0))
    mj_ep_len = scaled_episode_length(num_cubes, ep_mult)
    pd_obs_dim = get_filtered_obs_dim(num_cubes, obs_space_list)
    resolved = run_cfg.get('resolved_config', {})
    ppo_defaults = run_cfg.get('ppo_env_defaults', {})
    use_pd = bool(flags.get('builderbench_use_pd', False))
    pd_duration = int(flags.get('builderbench_pd_duration', 5))
    obs_dim = int(resolved.get('obs_dim', full_obs_dim))
    hidden = tuple(int(x) for x in resolved.get(
        'hidden_layer_sizes', contrastive.ContrastiveConfig().hidden_layer_sizes))
    actor_min_std = float(resolved.get(
        'ppo_actor_min_std', contrastive.ContrastiveConfig().ppo_actor_min_std))
    start_index = int(resolved.get(
        'start_index', ppo_defaults.get('start_index', 0)))
    end_index = int(resolved.get(
        'end_index', ppo_defaults.get('end_index', num_cubes * 3)))
    fixed = run_cfg.get('fixed_start_end')
    fixed_goal = (None if fixed is None
                  else np.asarray(fixed, dtype=np.float32))
  else:
    use_pd = False
    pd_duration = 5
    ep_mult = 1.0
    mj_ep_len = scaled_episode_length(num_cubes, ep_mult)
    obs_dim = full_obs_dim
    defaults = ppo_env_defaults_for_env(env_name) or {}
    hidden = tuple(contrastive.ContrastiveConfig().hidden_layer_sizes)
    actor_min_std = float(contrastive.ContrastiveConfig().ppo_actor_min_std)
    start_index = int(defaults.get('start_index', 0))
    end_index = int(defaults.get('end_index', num_cubes * 3))
    fixed_goal = fixed_goal_for_env(env_name)
    obs_space_list = ['xy', 'select']
    pd_obs_dim = pd_policy_state_obs_dim(num_cubes)

  if use_pd:
    macro_ep_len = mj_ep_len // pd_duration
    filter_policy = (obs_dim == pd_obs_dim)
    if obs_dim not in (pd_obs_dim, full_obs_dim):
      print(f'[bb_video] WARNING: obs_dim={obs_dim} unexpected for PD; '
            f'assuming filter={obs_dim == pd_obs_dim}')
  else:
    macro_ep_len = mj_ep_len
    filter_policy = False

  print(f'[bb_video] training context: use_pd={use_pd} pd_duration={pd_duration} '
        f'filter_policy_obs={filter_policy} obs_dim={obs_dim} '
        f'ep_len={macro_ep_len} hidden={hidden} '
        f'fixed_goal={fixed_goal is not None} '
        f'actor_min_std={actor_min_std} '
        f'start_index={start_index} end_index={end_index} '
        f'obs_space_list={obs_space_list} ep_mult={ep_mult}')

  return _TrainCtx(
      use_pd=use_pd,
      pd_duration=pd_duration,
      filter_policy_obs=filter_policy,
      episode_length=int(macro_ep_len),
      obs_dim=int(obs_dim),
      hidden_layer_sizes=hidden,
      fixed_target_goal=fixed_goal,
      actor_min_std=actor_min_std,
      start_index=start_index,
      end_index=end_index,
      obs_space_list=obs_space_list,
      episode_length_multiplier=ep_mult,
  )


def _build_networks(env_name: str, seed: int, ctx: _TrainCtx):
  env_kwargs: Dict[str, Any] = {}
  env_kwargs['builderbench_episode_length_multiplier'] = ctx.episode_length_multiplier
  if ctx.use_pd:
    env_kwargs['builderbench_use_pd'] = True
    env_kwargs['builderbench_pd_duration'] = ctx.pd_duration
    env_kwargs['builderbench_pd_filter_policy_obs'] = ctx.filter_policy_obs
    env_kwargs['obs_space_list'] = ctx.obs_space_list

  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name,
      ctx.start_index,
      ctx.end_index,
      seed=seed,
      fixed_start_end=ctx.fixed_target_goal,
      **env_kwargs,
  )
  if int(obs_dim) != int(ctx.obs_dim):
    print(f'[bb_video] WARNING: probe obs_dim={obs_dim} != '
          f'run_config obs_dim={ctx.obs_dim}')
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env

  cfg = contrastive.ContrastiveConfig()
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=int(ctx.obs_dim),
      repr_dim=cfg.repr_dim,
      repr_norm=cfg.repr_norm,
      twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs,
      hidden_layer_sizes=ctx.hidden_layer_sizes,
      actor_min_std=ctx.actor_min_std,
  )
  return networks


def _enumerate_checkpoints(path: str) -> List[Tuple[str, str]]:
  if os.path.isfile(path):
    base = os.path.splitext(os.path.basename(path))[0]
    label = base[len('ckpt_'):] if base.startswith('ckpt_') else base
    return [(label, path)]

  if not os.path.isdir(path):
    raise FileNotFoundError(f'Checkpoint path not found: {path}')

  iter_files = glob.glob(os.path.join(path, 'ckpt_iter_*.pkl'))

  def _it(fname):
    m = re.search(r'ckpt_iter_(\d+)\.pkl$', fname)
    return int(m.group(1)) if m else -1

  iter_files.sort(key=_it)
  entries = [(f'iter_{_it(f):07d}', f) for f in iter_files]
  latest = os.path.join(path, 'latest.pkl')
  if os.path.isfile(latest):
    entries.append(('latest', latest))
  return entries


def _make_bb_env(env_id: str, ctx: _TrainCtx):
  """Build a single-env batched BuilderBench stack matching training."""
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.episode_length = scaled_episode_length(num_cubes, ctx.episode_length_multiplier)
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  if env_id in _MJX_PARAMS:
    cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]

  base = CreativeCube(config=cfg)
  mocap_targets = base._mocap_targets

  if ctx.use_pd:
    assert cfg.episode_length % ctx.pd_duration == 0, (
        f'episode_length {cfg.episode_length} must divide pd_duration '
        f'{ctx.pd_duration}')
    inner = PDWrapper(base, duration=ctx.pd_duration)
    macro_ep_len = cfg.episode_length // ctx.pd_duration
  else:
    inner = base
    macro_ep_len = cfg.episode_length

  env = VmapWrapper(inner)
  env = EpisodeWrapper(env, episode_length=int(macro_ep_len), action_repeat=1)
  env = AutoResetWrapper(env)
  # Expose render helper from the underlying CreativeCube instance.
  env.render_from_info = base.render_from_info  # type: ignore[attr-defined]
  env._mocap_targets_geom = base._mocap_targets_geom  # type: ignore[attr-defined]
  env.model = base._mj_model  # type: ignore[attr-defined]

  return env, base, mocap_targets, int(macro_ep_len)


def _make_policy_fn(
    networks,
    policy_params,
    stochastic: bool,
    *,
    filter_policy_obs: bool,
    num_cubes: int,
    obs_space_list: list[str]
):
  @jax.jit
  def policy(obs, goals, key):
    if filter_policy_obs:
      obs = filter_pd_policy_state_obs(obs, num_cubes, obs_space_list)
    packed = jnp.concatenate([obs, goals], axis=-1)
    dist = networks.policy_network.apply(policy_params, packed)
    if stochastic:
      action = networks.sample(dist, key)
    else:
      action = networks.sample_eval(dist, jax.random.PRNGKey(0))
    return action, {}

  return policy


def _write_video(frames, path: str, fps: int):
  import imageio.v2 as imageio
  out_dir = os.path.dirname(os.path.abspath(path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


def _resolve_output_path(output_arg: str, run_tag: str, label: str,
                         multi: bool) -> str:
  is_dir_like = output_arg.endswith(os.sep) or os.path.isdir(output_arg)
  if is_dir_like:
    return os.path.join(output_arg, f'{run_tag}_{label}.mp4')
  if not multi:
    return output_arg
  stem, ext = os.path.splitext(output_arg)
  ext = ext or '.mp4'
  return f'{stem}_{label}{ext}'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env', default='builderbench_creative_1_task1')
  parser.add_argument('--output', required=True)
  parser.add_argument('--fps', type=int, default=10)
  parser.add_argument('--stochastic', action='store_true')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--run_tag', default=None,
                      help='Prefix for output filenames (default: env name).')
  args = parser.parse_args()

  if not is_builderbench_creative_env(args.env):
    parser.error(
        f'--env must match builderbench_creative_{{N}}_task{{K}} '
        f'(got {args.env!r})')

  env_id = sgcrl_env_name_to_bb_env_id(args.env)
  num_cubes, _ = parse_bb_env_id(env_id)
  run_tag = args.run_tag or args.env

  ckpt_entries = _enumerate_checkpoints(args.checkpoint)
  if not ckpt_entries:
    print(f'[bb_video] no checkpoints found at {args.checkpoint!r}')
    return
  print(f'[bb_video] found {len(ckpt_entries)} checkpoint(s)')

  ctx = _load_train_ctx(args.env, args.checkpoint)
  cfg_path = _run_config_path_for_checkpoint(args.checkpoint)
  if cfg_path is not None:
    skip = video_render_skip_reason(cfg_path)
    if skip is not None:
      print(f'[bb_video] skip: {skip}')
      return
  cfg_note = (f'use_pd={ctx.use_pd} pd_duration={ctx.pd_duration} '
              f'filter_policy_obs={ctx.filter_policy_obs}')
  print(f'[bb_video] training context: {cfg_note} '
        f'obs_dim={ctx.obs_dim} ep_len={ctx.episode_length}')

  print('[bb_video] building networks and builderbench env...')
  networks = _build_networks(args.env, seed=args.seed, ctx=ctx)
  video_env, _base, mocap_targets, episode_length = _make_bb_env(env_id, ctx)
  print(f'[bb_video] env_id={env_id}  episode_length={episode_length}')

  key = jax.random.PRNGKey(args.seed)
  multi = len(ckpt_entries) > 1
  out_dir = args.output
  if out_dir.endswith(os.sep) or not out_dir.endswith('.mp4'):
    os.makedirs(out_dir, exist_ok=True)

  for label, path in ckpt_entries:
    print(f'[bb_video] === {label}  ({path}) ===')
    ckpt = ppo_learner.load_checkpoint(path)
    policy_params = ckpt['policy_params']
    print(f'[bb_video]   iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}')

    policy = _make_policy_fn(
        networks,
        policy_params,
        args.stochastic,
        filter_policy_obs=ctx.filter_policy_obs,
        num_cubes=num_cubes,
      obs_space_list=ctx.obs_space_list
    )
    key, video_key = jax.random.split(key)
    frames = _get_video(
        policy,
        video_env,
        video_key,
        episode_length,
        fixed_target_goal=ctx.fixed_target_goal,
        mocap_targets=mocap_targets,
        num_cubes=num_cubes,
    )
    print(f'[bb_video]   frames={len(frames)}')

    out_path = _resolve_output_path(out_dir, run_tag, label, multi)
    _write_video(frames, out_path, args.fps)
    print(f'[bb_video]   wrote {out_path}')


if __name__ == '__main__':
  main()
```

## File: contrastive/ppo_learner.py
```python
"""PPO learner on representation-based rewards (r = φ·ψ).

Standalone single-process implementation.  No Launchpad, no Reverb.

Design (matches user requirements — SAC code stays untouched):
  * The policy that collects data IS the PPO policy being optimized.  PPO
    updates are strictly on-policy over each fresh rollout batch.
  * Per-step reward:  r_t = φ(s_t, a_t) · ψ(g_t), using the *current* CRL
    representations (frozen for the reward computation but updated
    off-policy from the replay buffer alongside PPO).
  * Rollouts are written to an in-memory FIFO replay buffer; the CRL
    critic (φ, ψ) is trained off-policy from the replay, unchanged from
    the existing InfoNCE / C-learning losses.
  * No κ network is used — PPO brings its own value net V(s, g).

Implementation style mirrors CleanRL's ppo_continuous_action.py:
  * Flat rollout storage (T × num_envs tensors)
  * GAE advantages, clipped surrogate, clipped value loss
  * Multiple PPO epochs + minibatches per update

Everything runs in JAX for consistency with the rest of this codebase.
"""
import concurrent.futures
import os
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme.jax import networks as networks_lib

from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks
from contrastive import gaussian_density as _gd
from contrastive import nf_density as _nf
from contrastive.utils import extract_info_reward
from metrics import compute_analysis_dict


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------
class PPOTrainingState(NamedTuple):
  """All trainable state held by the PPO + CRL learner."""
  # PPO-side params (policy + value) share a single Adam optimizer, CleanRL-style.
  policy_params: networks_lib.Params
  value_params: networks_lib.Params
  ppo_optimizer_state: optax.OptState
  # CRL-side params trained from replay.
  q_params: networks_lib.Params
  q_optimizer_state: optax.OptState
  key: networks_lib.PRNGKey


# ---------------------------------------------------------------------------
# Rollout storage  (populated on-policy, consumed by GAE + PPO update)
# ---------------------------------------------------------------------------
class Rollout(NamedTuple):
  """A (T, num_envs, ...) block of on-policy experience."""
  obs: jnp.ndarray           # (T, E, obs_dim_total)
  actions: jnp.ndarray       # (T, E, act_dim)
  logprobs: jnp.ndarray      # (T, E)
  rewards: jnp.ndarray       # (T, E)           r_t = φ·ψ
  dones: jnp.ndarray         # (T, E)           terminal OR truncation at step t
  values: jnp.ndarray        # (T, E)           V(s_t, g_t)
  next_obs: jnp.ndarray      # (E, obs_dim_total)  final bootstrap state
  next_done: jnp.ndarray     # (E,)


# ---------------------------------------------------------------------------
# Reward normalization (CleanRL `NormalizeReward` wrapper, numpy version).
# ---------------------------------------------------------------------------
class RunningMeanStd:
  """Welford-style online mean / variance tracker with optional window cap.

  Matches OpenAI Baselines' `RunningMeanStd` and CleanRL's vector-env
  reward normalizer.  Maintains mean, variance, and sample count; `update`
  accepts a batch and folds it in via the parallel-algorithm formula.

  When ``max_count > 0`` the effective count is capped at ``max_count`` after
  each update.  Once the cap is reached every new batch of size B gets weight
  B / max_count instead of B / (old_count + B), making the estimate track
  recent data more closely (soft sliding-window effect).
  """

  def __init__(self, shape=(), epsilon: float = 1e-4, max_count: float = 0):
    self.mean = np.zeros(shape, dtype=np.float64)
    self.var = np.ones(shape, dtype=np.float64)
    self.count = float(epsilon)
    self.max_count = float(max_count) if max_count > 0 else 0.0

  def update(self, x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    batch_mean = x.mean(axis=0)
    batch_var = x.var(axis=0)
    batch_count = x.shape[0]
    delta = batch_mean - self.mean
    tot = self.count + batch_count
    new_mean = self.mean + delta * (batch_count / tot)
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot)
    self.mean = new_mean
    self.var = M2 / tot
    self.count = tot
    # Soft window cap: once count exceeds max_count, pin it so new data
    # gets proportionally more weight in future updates.
    if self.max_count > 0 and self.count > self.max_count:
      self.count = self.max_count


class ReturnNormalizer:
  """Normalizes a per-env reward stream by the running std of discounted
  returns (NOT by the std of raw rewards).

  This is CleanRL's `NormalizeReward` wrapper, translated to a plain
  object we can call inside the rollout loop.  For each env we keep a
  *running discounted return* G_i that is updated on every step; the
  RunningMeanStd is updated with the current G's, and the *instantaneous*
  reward r_t is divided by sqrt(Var(G) + eps).  On episode boundaries G_i
  is reset to 0.

  Rationale: dividing the raw reward by its own std would shrink the
  signal to O(1) but leave very high variance advantages because the
  discounted *return* — what GAE actually bootstraps — can still be much
  larger.  Scaling by std(G) directly normalizes what the value head has
  to fit, which is what empirically stabilizes PPO with learned rewards.

  window_size > 0 caps the effective sample count so the variance estimate
  stays responsive to recent changes (useful when the reward distribution
  shifts rapidly, e.g. as the NF density model improves).
  """

  def __init__(self, num_envs: int, discount: float, epsilon: float = 1e-8,
               window_size: int = 0):
    self._rms = RunningMeanStd(shape=(), max_count=float(window_size))
    self._returns = np.zeros(num_envs, dtype=np.float64)
    self._gamma = float(discount)
    self._eps = float(epsilon)

  def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Rescale a (num_envs,) reward vector and advance internal state.

    Args:
      reward: per-env reward at step t (float32/64).
      done:   per-env termination mask at step t (bool / 0-1).
    Returns:
      Rescaled reward, same shape as input.
    """
    reward = np.asarray(reward, dtype=np.float64)
    done = np.asarray(done).astype(bool)
    self._returns = self._returns * self._gamma + reward
    self._rms.update(self._returns)
    scaled = reward / np.sqrt(self._rms.var + self._eps)
    # Reset per-env running return at episode boundaries so the normalizer
    # tracks the *in-episode* discounted return distribution.
    if done.any():
      self._returns = np.where(done, 0.0, self._returns)
    return scaled.astype(np.float32)

  @property
  def std(self) -> float:
    return float(np.sqrt(self._rms.var + self._eps))


# ---------------------------------------------------------------------------
# In-memory FIFO replay buffer (CRL side only).
# ---------------------------------------------------------------------------
class EpisodeReplay:
  """In-memory episode buffer matching the naive CRL pipeline exactly.

  Replicates the observable behavior of the Reverb-based pipeline used
  by the original SAC-free CRL code (`builder.py::flatten_fn`
  followed by `batch(B) → transpose → unbatch → unbatch` on a Reverb
  Uniform-selector episode table).  After that tf.data chain, each
  individual training sample has the following distribution:

      k  ~ Uniform(stored episodes)
      t  ~ Uniform[0, T_k - 1]
      d  ~ TruncatedGeometric(1 - γ,  range=[1, T_k - t])
      j  = t + d
      obs       = [ s_t       ;  obs_to_goal_2d(s_j) ]
      action    =   a_t
      next_obs  = [ s_{t+1}   ;  obs_to_goal_2d(s_j) ]

  with the B samples in a training minibatch drawn i.i.d. (so, with
  high probability, each InfoNCE batch has B distinct source episodes —
  the property that makes the negatives semantically diverse).

  The `tf.roll` shift inside `flatten_fn` is irrelevant here: it only
  decorrelates *contiguous* transitions within an episode, and in this
  equivalent reformulation we never emit two transitions from the same
  episode in a single batch call (except by coincidence, same as the
  naive pipeline).

  Storage: list of episodes, each {'obs': (T+1, D), 'action': (T, A)}.
  FIFO eviction is at the episode granularity once total stored
  transitions exceed `capacity`.
  """

  def __init__(self, capacity: int, obs_dim: int, discount: float,
               start_index: int, end_index: int):
    self._cap = capacity
    self._obs_dim = obs_dim               # state slice size
    self._discount = float(discount)
    self._start_index = start_index
    self._end_index = end_index
    self._episodes: list = []             # list[{'obs': (T+1, D), 'action': (T, A)}]
    self._ep_lens: list = []              # parallel list of T (== len(action))
    self._total_transitions = 0
    # Pre-compute log(γ) once; used per sample call for the truncated
    # geometric.  γ must be in (0, 1) for the formula to be well-defined.
    assert 0.0 < self._discount < 1.0, (
        f'discount must be in (0, 1), got {self._discount}')
    self._log_gamma = float(np.log(self._discount))

  @property
  def size(self) -> int:
    """Number of stored (s, a) transitions across all retained episodes."""
    return self._total_transitions

  @property
  def num_episodes(self) -> int:
    return len(self._episodes)

  def add_episode(self, obs: np.ndarray, action: np.ndarray):
    """Store one complete episode.

    Args:
      obs:    (T+1, obs_dim_total) — includes the terminal observation.
      action: (T,   act_dim)
    Episodes shorter than 1 transition are rejected.
    """
    T = action.shape[0]
    assert obs.shape[0] == T + 1, f'obs len {obs.shape[0]} != action len {T} + 1'
    if T < 1:
      return
    self._episodes.append({
        'obs': np.asarray(obs, dtype=np.float32),
        'action': np.asarray(action, dtype=np.float32),
    })
    self._ep_lens.append(T)
    self._total_transitions += T
    # FIFO eviction at the episode granularity.
    while self._total_transitions > self._cap and len(self._episodes) > 1:
      self._episodes.pop(0)
      self._total_transitions -= self._ep_lens.pop(0)

  # ---------------------------------------------------------------------
  # Internal helpers
  # ---------------------------------------------------------------------
  def _obs_to_goal(self, states: np.ndarray) -> np.ndarray:
    """Equivalent to `contrastive/utils.py::obs_to_goal_2d`."""
    if self._end_index == -1:
      return states[:, self._start_index:]
    return states[:, self._start_index:self._end_index]

  # ---------------------------------------------------------------------
  # Sampling — matches the naive CRL pipeline exactly
  # ---------------------------------------------------------------------
  def sample(self, batch_size: int,
             rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Draw B i.i.d. transitions `[s_t; goal_j]` matching naive CRL.

    See class docstring for the per-sample distribution; this
    implementation vectorizes index sampling across the batch and uses
    a tiny Python loop only for the final gather into arrays (episodes
    have different lengths so a single numpy take isn't possible
    without padding).  The loop is O(B), not O(B·T).

    Returns a dict with the exact keys consumed by `critic_loss`:
      obs        : (B, obs_dim_total)   [ s_t       ; obs_to_goal(s_j) ]
      action     : (B, act_dim)
      next_obs   : (B, obs_dim_total)   [ s_{t+1}   ; obs_to_goal(s_j) ]
    """
    assert self.size > 0, 'Cannot sample from an empty buffer.'
    B = int(batch_size)
    num_eps = len(self._episodes)
    ep_lens = np.asarray(self._ep_lens, dtype=np.int64)      # (K,)

    # (1) Episode index: Uniform over stored episodes (matches reverb Uniform).
    ep_ids = rng.integers(0, num_eps, size=B)
    lens = ep_lens[ep_ids]                                   # (B,)  == T_k

    # (2) Starting timestep t: Uniform[0, T_k - 1].  Flatten_fn's valid t
    #     range is exactly this — it slices obs[:-1] (last obs has no action).
    u_t = rng.random(B)
    t = np.floor(u_t * lens).astype(np.int64)
    t = np.minimum(t, lens - 1)                              # numerical guard

    # (3) Future offset d ~ TruncatedGeometric(1-γ, [1, T_k - t]).
    #     Continuous-time derivation — for U ~ Uniform(0, 1 - γ^max_d),
    #       d = 1 + floor( log(1 - U) / log(γ) )
    #     distributes U over the truncated geometric support exactly.
    #     Matches `flatten_fn`'s categorical with probs ∝ γ^(j-t) normalized
    #     to the in-episode future states only.
    max_d = lens - t                                         # (B,)  ≥ 1
    trunc_cdf = 1.0 - np.power(self._discount,
                               max_d.astype(np.float64))     # CDF at max_d
    u_d = rng.random(B) * trunc_cdf                          # U ∈ [0, trunc_cdf)
    # `log1p(-u_d)` is `log(1 - u_d)` but numerically stable near 0.
    d = 1 + np.floor(np.log1p(-u_d) / self._log_gamma).astype(np.int64)
    d = np.clip(d, 1, max_d)                                 # guard rounding

    j = t + d                                                # (B,)  ∈ [t+1, T_k]

    # (4) Gather.  Per-sample index into the i-th chosen episode.
    obs_out = np.empty((B, 2 * self._obs_dim), dtype=np.float32)
    # Note: final obs dim = self._obs_dim (state) + goal_dim; for the
    # standard configurations goal_dim == self._obs_dim, hence `2*obs_dim`.
    # For other envs we resize on the first fill.
    act_dim = self._episodes[0]['action'].shape[1]
    act_out = np.empty((B, act_dim), dtype=np.float32)
    # We don't know goal_dim until we call _obs_to_goal on the first sample.
    next_obs_out = None
    goal_dim = None
    for i in range(B):
      ep = self._episodes[int(ep_ids[i])]
      full_obs = ep['obs']                      # (T+1, D_full)
      act = ep['action']                        # (T, A)
      ti = int(t[i]);  ji = int(j[i])
      s_t       = full_obs[ti,     :self._obs_dim]
      s_tp1     = full_obs[ti + 1, :self._obs_dim]
      s_j       = full_obs[ji,     :self._obs_dim]
      goal      = self._obs_to_goal(s_j[None])[0]
      if goal_dim is None:
        goal_dim = goal.shape[0]
        obs_out = np.empty((B, self._obs_dim + goal_dim), dtype=np.float32)
        next_obs_out = np.empty_like(obs_out)
      obs_out[i, :self._obs_dim] = s_t
      obs_out[i,  self._obs_dim:] = goal
      next_obs_out[i, :self._obs_dim] = s_tp1
      next_obs_out[i,  self._obs_dim:] = goal
      act_out[i] = act[ti]

    return {
        'obs': obs_out,
        'action': act_out,
        'next_obs': next_obs_out,
    }

  def sample_states(self, n: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Return up to *n* raw states (shape ``(n, obs_dim)``) sampled uniformly
    from all stored transitions.  Used for KDE fitting."""
    all_states = np.concatenate(
        [ep['obs'][:, :self._obs_dim] for ep in self._episodes], axis=0)
    n = min(n, len(all_states))
    idx = rng.choice(len(all_states), size=n, replace=False)
    return all_states[idx].astype(np.float32)

  def sample_with_uniform_negatives(
      self, batch_size: int, rng: np.random.Generator,
      goal_low: np.ndarray, goal_high: np.ndarray,
  ) -> Dict[str, np.ndarray]:
    """Like sample(), but the second half of the batch has goals replaced
    by goals sampled uniformly from [goal_low, goal_high].

    This gives a 50/50 mix: half the in-batch negatives seen by the InfoNCE
    loss come from the replay future-state distribution, half from the
    uniform goal distribution — without changing the loss function.
    """
    batch = self.sample(batch_size, rng)
    half = batch_size // 2
    goal_dim = batch['obs'].shape[1] - self._obs_dim
    uniform_goals = rng.uniform(
        goal_low, goal_high, size=(half, goal_dim)).astype(np.float32)
    obs = batch['obs'].copy()
    next_obs = batch['next_obs'].copy()
    obs[half:, self._obs_dim:] = uniform_goals
    next_obs[half:, self._obs_dim:] = uniform_goals
    return {'obs': obs, 'action': batch['action'], 'next_obs': next_obs}


# ---------------------------------------------------------------------------
# Core factories.  Each returns a jitted function plus any needed metadata.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Gaussian KDE density estimator (pure numpy, fitted to replay buffer states)
# ---------------------------------------------------------------------------
class GaussianKDE:
  """Isotropic Gaussian KDE fitted to a set of training points.

  All maths is in numpy so it can be called from the (non-jitted) rollout
  loop without JAX overhead or shape-recompilation issues.

  Log density at a batch of test points ``x`` (shape ``(B, d)``)::

      log p(x) = logsumexp_n [ log(1/N) − d·log(h) − d/2·log(2π)
                                − ½·‖(x − x_n)/h‖² ]

  Bandwidth *h* defaults to Scott's rule: ``N^{-1/(d+4)}``.
  """

  def __init__(self, train_xs: np.ndarray, bandwidth: float):
    self.train_xs = train_xs.astype(np.float32)    # (N, d)
    self.bandwidth = float(bandwidth)
    n, d = train_xs.shape
    self._log_w = -np.log(float(n))                # uniform log-weight
    self._log_norm = (0.5 * d * np.log(2.0 * np.pi)
                      + d * np.log(self.bandwidth))

  def log_density(self, test_x: np.ndarray,
                  chunk_size: int = 2000) -> np.ndarray:
    """Return log p(test_x) for test_x of shape ``(B, d)``.

    Processes in chunks of *chunk_size* to keep peak memory bounded at
    ``chunk_size × N_train × d × 4`` bytes (≈64 MB for 2000×2000×2).
    """
    test_x = np.asarray(test_x, dtype=np.float32)
    B = len(test_x)
    if B <= chunk_size:
      return self._log_density_chunk(test_x)
    out = np.empty(B, dtype=np.float32)
    for i in range(0, B, chunk_size):
      out[i:i + chunk_size] = self._log_density_chunk(test_x[i:i + chunk_size])
    return out

  def _log_density_chunk(self, test_x: np.ndarray) -> np.ndarray:
    diffs = (test_x[:, None, :] - self.train_xs[None, :, :]) / self.bandwidth
    log_k = -0.5 * np.sum(diffs ** 2, axis=-1) - self._log_norm  # (B, N)
    log_contrib = self._log_w + log_k                              # (B, N)
    lc_max = log_contrib.max(axis=-1, keepdims=True)
    log_p = (np.log(np.exp(log_contrib - lc_max).sum(axis=-1))
             + lc_max[:, 0])
    return log_p  # (B,)

  @classmethod
  def fit(cls,
          replay: 'EpisodeReplay',
          obs_dim: int,
          max_points: int,
          rng: np.random.Generator,
          bandwidth: Optional[float] = None) -> 'GaussianKDE':
    """Sample *max_points* states from *replay* and fit the KDE."""
    states = replay.sample_states(max_points, rng)
    n, d = states.shape
    if bandwidth is None:
      bandwidth = float(n) ** (-1.0 / (d + 4))
    return cls(states, bandwidth)


def make_reward_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
):
  """Factory for the per-step PPO reward used during rollouts.

  Modes (``config.ppo_reward_mode``):
    * ``''`` (default): r = φ(s,a) · ψ(g).
    * ``'dirac_target'``: for s ≠ g, r = log(eps) − φ(s0,a)·ψ(s);
      at s = g, r = −φ(s0,a)·ψ(g).  s0 is fixed per episode; a ~ π(·|s).
    * ``'kde_dirac'``: same formula as ``dirac_target`` but the CRL dot
      products are replaced by Gaussian KDE log-densities fitted on replay
      buffer states.  Requires a ``GaussianKDE`` object to be maintained
      externally and passed as the first argument of the returned function.
  """
  mode = (getattr(config, 'ppo_reward_mode', '') or '').strip().lower()
  if mode == 'dirac_target':
    return make_dirac_target_reward_fn(networks, config)
  if mode == 'kde_dirac':
    return make_kde_dirac_reward_fn(config)
  if mode not in ('', 'phi_psi'):
    raise ValueError(
        f'Unknown ppo_reward_mode={config.ppo_reward_mode!r}; '
        f"supported: '', 'phi_psi', 'dirac_target', 'kde_dirac'")

  @jax.jit
  def reward_fn(q_params: networks_lib.Params,
                obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    _, sa_repr, g_repr = networks.q_network.apply(q_params, obs, action)
    return jnp.sum(sa_repr * g_repr, axis=-1)  # (B,)
  return reward_fn


def make_dirac_target_reward_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
):
  """Dirac-target baseline: log(eps) − φ(s0,a)·ψ(s) off-goal, −φ(s0,a)·ψ(g) at g."""
  obs_dim = int(config.obs_dim)
  eps = float(getattr(config, 'ppo_dirac_eps', 1e-6))
  goal_tol = 1e-2

  @jax.jit
  def reward_fn(q_params: networks_lib.Params,
                obs: jnp.ndarray,
                action: jnp.ndarray,
                s0_states: jnp.ndarray) -> jnp.ndarray:
    s = obs[:, :obs_dim]
    g = obs[:, obs_dim:]
    obs_s0 = jnp.concatenate([s0_states, g], axis=-1)
    obs_ss = jnp.concatenate([s, s], axis=-1)
    _, phi_s0, psi_g = networks.q_network.apply(q_params, obs_s0, action)
    _, _, psi_s = networks.q_network.apply(q_params, obs_ss, action)
    dot_ps = jnp.sum(phi_s0 * psi_s, axis=-1)
    dot_pg = jnp.sum(phi_s0 * psi_g, axis=-1)
    at_goal = jnp.linalg.norm(s - g, axis=-1) < goal_tol
    log_rew = jnp.log(jnp.maximum(eps, 1e-10)) - dot_ps
    return jnp.where(at_goal, -dot_pg, log_rew)
  return reward_fn


def make_kde_dirac_reward_fn(config: contrastive_config.ContrastiveConfig):
  """KDE-dirac reward: same sign convention as dirac_target but CRL dot
  products are replaced by Gaussian KDE log-densities from the replay buffer.

  Off-goal:  r = log(eps) − log p_kde(s)
  At goal:   r = −log p_kde(g)

  The returned function signature is::

      reward_fn(kde: GaussianKDE, obs: np.ndarray) -> np.ndarray

  *kde* is a :class:`GaussianKDE` fitted to recent replay-buffer states
  and must be updated by the caller (see ``kde_refit_interval`` config).
  The function operates entirely in numpy so it can be called without JAX.
  """
  obs_dim = int(config.obs_dim)
  eps = float(getattr(config, 'ppo_dirac_eps', 1e-6))
  log_eps = float(np.log(max(eps, 1e-10)))
  goal_tol = 1e-2

  def reward_fn(kde: GaussianKDE,
                obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    s = obs[:, :obs_dim]
    g = obs[:, obs_dim:]
    log_ps = kde.log_density(s)
    log_pg = kde.log_density(g)
    at_goal = np.linalg.norm(s - g, axis=-1) < goal_tol
    log_rew = log_eps - log_ps
    return np.where(at_goal, -log_pg, log_rew)

  return reward_fn


def make_value_fn(networks: contrastive_networks.ContrastiveNetworks):
  """Returns V(value_params, obs) — scalar value per sample."""
  @jax.jit
  def value_fn(value_params: networks_lib.Params,
               obs: jnp.ndarray) -> jnp.ndarray:
    return networks.value_network.apply(value_params, obs)  # (B,)
  return value_fn


def make_gae_fn(config: contrastive_config.ContrastiveConfig):
  """Returns a jitted GAE advantage + return computation.

  Matches CleanRL's `ppo_continuous_action.py` computation:

    δ_t         = r_t + γ·V(s_{t+1})·(1−done_{t+1}) − V(s_t)
    A_t         = δ_t + γ·λ·(1−done_{t+1})·A_{t+1}
    returns_t   = A_t + V(s_t)

  Inputs are rollout tensors shaped (T, E).  The extra `next_value` and
  `next_done` correspond to the state after the last collected step
  (used for bootstrapping A_{T-1}).
  """
  ppo_gamma = (float(config.ppo_discount)
               if float(getattr(config, 'ppo_discount', -1.0)) > 0.0
               else float(config.discount))
  gae_lambda = float(config.ppo_gae_lambda)

  @jax.jit
  def gae_fn(rewards: jnp.ndarray, values: jnp.ndarray,
             dones: jnp.ndarray, next_value: jnp.ndarray,
             next_done: jnp.ndarray
             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Build per-timestep "next" tensors: for t in [0, T-1) we use
    # values[t+1] / dones[t+1]; for t = T-1 we use next_value / next_done.
    next_vals_seq = jnp.concatenate([values[1:], next_value[None]], axis=0)  # (T, E)
    next_dones_seq = jnp.concatenate([dones[1:], next_done[None]], axis=0)   # (T, E)
    next_nonterm = 1.0 - next_dones_seq.astype(jnp.float32)

    deltas = rewards + ppo_gamma * next_vals_seq * next_nonterm - values  # (T, E)

    def scan_fn(last_gae, inputs):
      delta_t, nnt_t = inputs
      new_gae = delta_t + ppo_gamma * gae_lambda * nnt_t * last_gae
      return new_gae, new_gae

    init = jnp.zeros(rewards.shape[1], dtype=rewards.dtype)
    _, adv_rev = jax.lax.scan(
        scan_fn, init,
        (deltas[::-1], next_nonterm[::-1]))
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns
  return gae_fn


def make_ppo_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
    ppo_optimizer: optax.GradientTransformation,
):
  """Returns a jitted one-minibatch PPO update.

  The update operates on a pytree of trainable params keyed as:
      {'policy': policy_params, 'value': value_params}

  Combined loss (CleanRL-style):
      L  =  pg_loss  -  ent_coef * entropy  +  vf_coef * v_loss

  * pg_loss:   clipped surrogate,  max(-adv*ratio, -adv*clip(ratio))
  * v_loss:    clipped MSE (optional) against `returns`
  * entropy:   single-sample MC estimate  -log π(ã|s) with ã ~ π(·|s)

  Returns a function `update(params, opt_state, batch, key)` that does
  one SGD step and returns (new_params, new_opt_state, metrics_dict).
  """
  clip_coef = float(config.ppo_clip_coef)
  vf_coef = float(config.ppo_vf_coef)
  ent_coef = float(config.ppo_ent_coef)
  clip_vloss = bool(config.ppo_clip_vloss)
  norm_adv = bool(config.ppo_norm_adv)

  def ppo_loss(params, batch, key):
    # ---- policy forward ----
    dist = networks.policy_network.apply(params['policy'], batch['obs'])
    new_logprob = networks.log_prob(dist, batch['actions'])         # (B,)
    # MC-estimate entropy: H(π) ≈ -log π(ã|s), ã ~ π.
    fresh_action = networks.sample(dist, key)
    entropy_est = -networks.log_prob(dist, fresh_action)            # (B,)

    # ---- value forward ----
    new_value = networks.value_network.apply(params['value'], batch['obs'])  # (B,)

    # ---- advantage normalization (per-minibatch) ----
    adv = batch['advantages']
    if norm_adv:
      adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # ---- clipped surrogate ----
    logratio = new_logprob - batch['old_logprobs']
    ratio = jnp.exp(logratio)
    pg1 = -adv * ratio
    pg2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = jnp.mean(jnp.maximum(pg1, pg2))

    # ---- value loss ----
    if clip_vloss:
      v_unclipped = (new_value - batch['returns']) ** 2
      v_clipped_pred = batch['old_values'] + jnp.clip(
          new_value - batch['old_values'], -clip_coef, clip_coef)
      v_clipped = (v_clipped_pred - batch['returns']) ** 2
      v_loss = 0.5 * jnp.mean(jnp.maximum(v_unclipped, v_clipped))
    else:
      v_loss = 0.5 * jnp.mean((new_value - batch['returns']) ** 2)

    # ---- entropy bonus ----
    entropy_mean = jnp.mean(entropy_est)
    # Term as it enters the minimized objective: L includes -coef * H.
    entropy_loss_term = -ent_coef * entropy_mean

    # ---- combined loss ----
    total = pg_loss - ent_coef * entropy_mean + vf_coef * v_loss

    # ---- diagnostics (stop_gradient is implicit for metrics) ----
    approx_kl = jnp.mean((ratio - 1.0) - logratio)    # http://joschu.net/blog/kl-approx.html
    old_approx_kl = jnp.mean(-logratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))

    # Pre-tanh Gaussian loc (μ) and scale (σ) from the policy head.
    normal = dist.distribution.distribution
    policy_loc = normal.loc
    policy_scale = normal.scale

    metrics = {
        'ppo_total_loss': total,
        'pg_loss': pg_loss,
        'v_loss': v_loss,
        'entropy': entropy_mean,
        'entropy_loss': entropy_loss_term,
        'approx_kl': approx_kl,
        'old_approx_kl': old_approx_kl,
        'clipfrac': clipfrac,
        'ratio_mean': jnp.mean(ratio),
        'policy_loc_mean': jnp.mean(policy_loc),
        'policy_loc_abs_mean': jnp.mean(jnp.abs(policy_loc)),
        'policy_scale_mean': jnp.mean(policy_scale),
        'policy_scale_min': jnp.min(policy_scale),
    }
    return total, metrics

  grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

  @jax.jit
  def update(params, opt_state, batch, key):
    (_, metrics), grads = grad_fn(params, batch, key)
    print("DEBUG ACTOR GRADS SHAPE:", jax.tree_util.tree_map(lambda x: x.shape, grads['policy']))
    actor_metrics = compute_analysis_dict(
        prefix="actor",
        params=params['policy'],
        grads=grads['policy']
    )
    metrics.update(actor_metrics)
    updates, new_opt_state = ppo_optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics

  return update


def make_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    backward: bool = False,
    _return_raw: bool = False,
):
  """Returns a jitted CRL critic update over one replay batch.

  PPO CRL uses in-batch InfoNCE (softmax cross-entropy on the diagonal) plus
  ``0.01 * logsumexp(logits, axis=1)^2``, matching the CPC path in
  ``learning.py``.  Supports ``(B, B)`` or twin ``(B, B, 2)`` logits.

  Args:
    networks: ContrastiveNetworks (``q_network`` only is used here).
    q_optimizer: Adam (or other) transform for Q / representation params.
    backward: If False (default), use forward InfoNCE — fix anchor (sᵢ,aᵢ),
      treat all goals gⱼ as negatives (denominator = row i of L).
      If True, use backward InfoNCE — fix goal gᵢ, treat all anchors (sⱼ,aⱼ)
      as negatives (denominator = col i of L), implemented by transposing
      the logit matrix before the loss.

  Returns:
    ``update(q_params, q_optimizer_state, batch, key)``
    → ``(new_q_params, new_q_optimizer_state, metrics)``
  """
  _logsumexp_penalty_coef = 0.01

  def critic_loss(q_params, batch, key):
    del key
    obs = batch['obs']
    action = batch['action']
    batch_size = obs.shape[0]
    labels = jnp.eye(batch_size)

    logits, _, _ = networks.q_network.apply(q_params, obs, action)

    # Transpose: L^T[i,j] = L[j,i]  →  row i of L^T = col i of L
    # diagonal is unchanged so positive pairs are preserved.
    if backward:
      if logits.ndim == 3:
        logits = jnp.transpose(logits, (1, 0, 2))
      else:
        logits = logits.T

    def loss_fn(_logits, _labels):
      return (
          optax.softmax_cross_entropy(logits=_logits, labels=_labels)
          + _logsumexp_penalty_coef * jax.nn.logsumexp(_logits, axis=1) ** 2)

    if logits.ndim == 3:
      loss = jax.vmap(loss_fn, in_axes=(2, None), out_axes=-1)(logits, labels)
      loss = jnp.mean(loss, axis=-1)
    else:
      loss = loss_fn(logits, labels)

    loss = jnp.mean(loss)
    train_logits = logits

    if train_logits.ndim == 2:
      narrow_logits = train_logits[:, :batch_size]
    else:
      narrow_logits = train_logits[:, :batch_size, :]

    if narrow_logits.ndim == 3:
      logits_flat = jnp.mean(narrow_logits, axis=-1)
    else:
      logits_flat = narrow_logits

    correct = (jnp.argmax(logits_flat, axis=1) == jnp.argmax(labels, axis=1))
    logits_pos = jnp.sum(logits_flat * labels) / jnp.sum(labels)
    logits_neg = jnp.sum(logits_flat * (1 - labels)) / jnp.sum(1 - labels)
    if train_logits.ndim == 3:
      logsumexp_val = jax.nn.logsumexp(train_logits[:, :, 0], axis=1) ** 2
    else:
      logsumexp_val = jax.nn.logsumexp(train_logits, axis=1) ** 2

    metrics = {
        'crl_loss': loss,
        'binary_accuracy': jnp.mean((logits_flat > 0) == labels),
        'categorical_accuracy': jnp.mean(correct),
        'logits_pos': logits_pos,
        'logits_neg': logits_neg,
        'logsumexp': logsumexp_val.mean(),
    }
    return loss, metrics

  grad_fn = jax.value_and_grad(critic_loss, has_aux=True)

  def update(q_params, q_optimizer_state, batch, key):
    (_, metrics), grads = grad_fn(q_params, batch, key)
    print("DEBUG CRITIC GRADS SHAPE:", jax.tree_util.tree_map(lambda x: x.shape, grads))
    critic_metrics = compute_analysis_dict(
          prefix="critic",
          params=q_params,
          grads=grads,
    )
    metrics.update(critic_metrics)
    grads_finite = jnp.all(jnp.asarray(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x: jnp.all(jnp.isfinite(x)), grads))))
    loss_finite = jnp.isfinite(metrics['crl_loss'])
    do_update = jnp.logical_and(grads_finite, loss_finite)

    def _apply(_):
      updates, new_opt_state = q_optimizer.update(
          grads, q_optimizer_state, q_params)
      new_q_params = optax.apply_updates(q_params, updates)
      return new_q_params, new_opt_state

    def _skip(_):
      return q_params, q_optimizer_state

    new_q_params, new_opt_state = jax.lax.cond(
        do_update, _apply, _skip, operand=None)
    metrics = dict(metrics)
    metrics['update_skipped_nonfinite'] = 1.0 - do_update.astype(jnp.float32)
    return new_q_params, new_opt_state, metrics

  if _return_raw:
    return jax.jit(update), update
  return jax.jit(update)


def make_scan_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    backward: bool = False,
    repr_tau: float = 0.0,
):
  """Scan-based CRL updater: runs N steps in one JIT call.

  Caller pre-samples all N batches in NumPy, stacks them into
  ``(N, B, dim)`` arrays, and transfers to GPU once.  A single
  ``jax.lax.scan`` call replaces N separate dispatch-and-sync rounds.

  The φ/ψ EMA for the PPO reward (``repr_tau > 0``) is also updated
  inside the scan, eliminating the 128 un-JIT-compiled ``_ema_tree``
  calls per iteration.

  Returns:
    ``multi_update(q_params, q_opt_state, q_params_ema, batches, key)``
    → ``(new_q_params, new_q_opt_state, new_q_params_ema, new_key,
         mean_metrics)``
    where ``batches`` is a dict of ``(N, B, dim)`` JAX arrays.
  """
  _, raw_update = make_crl_update_fn(
      networks, q_optimizer, backward=backward, _return_raw=True)

  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def multi_update(q_params, q_opt_state, q_params_ema, batches, key):
    def scan_step(carry, batch):
      q_p, q_opt, q_ema, k = carry
      k, k_crl = jax.random.split(k)
      q_p, q_opt, m = raw_update(q_p, q_opt, batch, k_crl)
      if use_ema:
        q_ema = jax.tree_util.tree_map(
            lambda t, o: _tau * t + (1.0 - _tau) * o, q_ema, q_p)
      else:
        q_ema = q_p
      return (q_p, q_opt, q_ema, k), m

    (q_params, q_opt_state, q_params_ema, key), metrics = jax.lax.scan(
        scan_step, (q_params, q_opt_state, q_params_ema, key), batches)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return q_params, q_opt_state, q_params_ema, key, metrics

  return multi_update


def make_onpolicy_crl_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    q_optimizer: optax.GradientTransformation,
    config: contrastive_config.ContrastiveConfig,
    backward: bool = False,
    repr_tau: float = 0.0,
):
  """Pure on-policy CRL updater over fresh (T, E, ...) rollout tensors."""
  _, raw_update = make_crl_update_fn(
      networks, q_optimizer, backward=backward, _return_raw=True)

  obs_dim = int(config.obs_dim)
  start_index = int(config.start_index)
  end_index = int(config.end_index)
  discount = float(config.discount)
  batch_size = int(config.batch_size)
  use_ema = 0.0 < float(repr_tau) < 1.0
  _tau = float(repr_tau)

  @jax.jit
  def onpolicy_update(q_params, q_opt_state, q_params_ema, roll_obs, roll_acts, roll_dones, key):
    T, E, _ = roll_obs.shape

    # Reverse scan to compute distance to episode done boundary
    def rev_scan(carry, done_t):
      val = jnp.where(done_t, 0, carry + 1)
      return val, val

    _, dist_to_done = jax.lax.scan(rev_scan, jnp.zeros(E, dtype=jnp.int32), roll_dones[::-1])
    dist_to_done = dist_to_done[::-1]

    # Upper bound future distance by current tensor dimensions
    t_indices = jnp.arange(T)[:, None]
    max_d_tensor = jnp.minimum(dist_to_done, T - 1 - t_indices)

    # Sample anchors (t, e)
    key, k_t, k_e, k_d, k_crl = jax.random.split(key, 5)
    t_samp = jax.random.randint(k_t, (batch_size,), 0, T)
    e_samp = jax.random.randint(k_e, (batch_size,), 0, E)
    max_d_samp = max_d_tensor[t_samp, e_samp]

    # Sample offset d using truncated geometric distribution
    trunc_cdf = 1.0 - jnp.power(discount, max_d_samp.astype(jnp.float32))
    u_d = jax.random.uniform(k_d, (batch_size,)) * trunc_cdf
    d_samp = 1 + jnp.floor(jnp.log1p(-u_d) / jnp.log(discount)).astype(jnp.int32)
    d_samp = jnp.clip(d_samp, 1, jnp.maximum(1, max_d_samp))
    d_samp = jnp.where(max_d_samp == 0, 0, d_samp)

    j_samp = t_samp + d_samp

    s_t = roll_obs[t_samp, e_samp, :obs_dim]
    a_t = roll_acts[t_samp, e_samp]
    s_j = roll_obs[j_samp, e_samp, :obs_dim]

    if end_index == -1:
      goal_j = s_j[:, start_index:]
    else:
      goal_j = s_j[:, start_index:end_index]

    crl_obs = jnp.concatenate([s_t, goal_j], axis=-1)
    batch = {'obs': crl_obs, 'action': a_t}

    q_params, q_opt_state, metrics = raw_update(q_params, q_opt_state, batch, k_crl)

    if use_ema:
      q_params_ema = jax.tree_util.tree_map(
          lambda t_val, o_val: _tau * t_val + (1.0 - _tau) * o_val, q_params_ema, q_params)
    else:
      q_params_ema = q_params

    return q_params, q_opt_state, q_params_ema, key, metrics

  return onpolicy_update


# ---------------------------------------------------------------------------
# Rollout collection (env stepping — runs in host/numpy, not jitted).
# ---------------------------------------------------------------------------
class VecEnv:
  """A tiny synchronous wrapper around N acme/dm_env environments.

  Uses dm_env's step/reset interface (not gym's) to stay compatible with
  contrastive_utils.make_environment.  Exposes a CleanRL-like API:

      obs = vec_env.reset()                                    # (E, obs_dim_total)
      next_obs, env_rew, dones, terminal_obs, info_rew = vec_env.step(actions)

  Envs auto-reset on episode end (matching gymnasium semantics), so
  `next_obs[i]` is the fresh initial observation of the new episode when
  `dones[i]` is True.  `terminal_obs[i]` is the true final observation
  of the just-finished episode — needed by the CRL episode buffer so it
  can store complete (T+1)-length trajectories.  When `dones[i]` is
  False, `terminal_obs[i]` equals `next_obs[i]`.

  The env reward is ignored by PPO (we use φ·ψ instead) but is returned
  so the caller can also log ground-truth return for sanity.
  """

  def __init__(self, env_factory: Callable, num_envs: int, seed: int):
    self._envs = [env_factory(seed + i) for i in range(num_envs)]
    self._num_envs = num_envs
    self._obs_spec = self._envs[0].observation_spec()
    self._action_spec = self._envs[0].action_spec()
    # Thread pool: MuJoCo C extensions release the GIL during sim.step(),
    # so threads can run env physics in parallel.
    self._executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=num_envs,
        thread_name_prefix='vecenv_worker')

  def reset(self) -> np.ndarray:
    obs = np.stack([e.reset().observation for e in self._envs], axis=0)
    return obs.astype(np.float32)

  def step(
      self, actions: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_dtype = self._action_spec.dtype
    shape = self._obs_spec.shape
    next_obs    = np.zeros((self._num_envs,) + shape, dtype=np.float32)
    terminal_obs = np.zeros((self._num_envs,) + shape, dtype=np.float32)
    env_rewards  = np.zeros(self._num_envs, dtype=np.float32)
    info_rewards = np.full(self._num_envs, np.nan, dtype=np.float32)
    dones        = np.zeros(self._num_envs, dtype=bool)

    def _step_one(i: int):
      env = self._envs[i]
      a = np.asarray(actions[i], dtype=np.float32)
      a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
      if hasattr(self._action_spec, 'minimum') and hasattr(self._action_spec, 'maximum'):
        a = np.clip(a, self._action_spec.minimum, self._action_spec.maximum)
      ts = env.step(a.astype(action_dtype))
      env_r  = 0.0 if ts.reward is None else float(ts.reward)
      info_r = extract_info_reward(env)
      term   = ts.observation.copy()
      done   = bool(ts.last())
      nxt    = env.reset().observation.copy() if done else ts.observation.copy()
      return env_r, info_r, term, nxt, done

    futures = [self._executor.submit(_step_one, i) for i in range(self._num_envs)]
    for i, fut in enumerate(futures):
      env_r, info_r, term, nxt, done = fut.result()
      env_rewards[i]  = env_r
      info_rewards[i] = info_r
      terminal_obs[i] = term
      next_obs[i]     = nxt
      dones[i]        = done

    return next_obs, env_rewards, dones, terminal_obs, info_rewards

  @property
  def num_envs(self) -> int:
    return self._num_envs

  @property
  def observation_shape(self) -> Tuple[int, ...]:
    return tuple(self._obs_spec.shape)

  @property
  def action_shape(self) -> Tuple[int, ...]:
    return tuple(self._action_spec.shape)


# ---------------------------------------------------------------------------
# Checkpoint helpers.
# ---------------------------------------------------------------------------
def _tree_copy(params):
  return jax.tree_util.tree_map(lambda x: x, params)


def _ema_tree(target, online, tau: float):
  """target ← τ·target + (1−τ)·online elementwise over a param pytree."""
  tau = float(tau)
  return jax.tree_util.tree_map(
      lambda t, o: tau * t + (1.0 - tau) * o, target, online)


def _save_checkpoint(path: str,
                     policy_params, value_params, q_params,
                     ppo_opt_state, q_opt_state,
                     iteration: int, global_step: int, key,
                     q_params_ema=None):
  """Write a pickle checkpoint atomically (write to tmp → rename)."""
  import pickle as _pkl
  import os as _os

  ckpt = {
      'policy_params':       policy_params,
      'value_params':        value_params,
      'q_params':            q_params,
      'ppo_optimizer_state': ppo_opt_state,
      'q_optimizer_state':   q_opt_state,
      'iteration':           int(iteration),
      'global_step':         int(global_step),
      'key':                 key,
  }
  if q_params_ema is not None:
    ckpt['q_params_ema'] = q_params_ema
  tmp_path = path + '.tmp'
  with open(tmp_path, 'wb') as fh:
    _pkl.dump(ckpt, fh, protocol=_pkl.HIGHEST_PROTOCOL)
  _os.replace(tmp_path, path)


def _prune_old_checkpoints(ckpt_dir: str, keep_last: int):
  """Delete oldest ckpt_iter_*.pkl until at most `keep_last` remain.

  No-op when ``keep_last <= 0`` (retain every milestone checkpoint).
  """
  import os as _os
  if keep_last <= 0:
    return
  files = [f for f in _os.listdir(ckpt_dir)
           if f.startswith('ckpt_iter_') and f.endswith('.pkl')]

  def _iter_of(fname):
    try:
      return int(fname[len('ckpt_iter_'):-len('.pkl')])
    except ValueError:
      return -1

  files.sort(key=_iter_of)
  to_remove = files[:-keep_last] if len(files) > keep_last else []
  for f in to_remove:
    try:
      _os.remove(_os.path.join(ckpt_dir, f))
    except OSError:
      pass


def load_checkpoint(path: str):
  """Load a checkpoint file produced by `_save_checkpoint`.

  Returns the raw pickled dict.  Caller is responsible for plugging the
  params back into networks / optimizers.
  """
  import pickle as _pkl
  with open(path, 'rb') as fh:
    return _pkl.load(fh)


def _truncate_csv_to_iteration(csv_path: str, max_iteration: int) -> None:
  """Rewrite a log CSV keeping only rows with learner_steps <= max_iteration.

  Called on resume to remove log entries written after the last checkpoint
  (which would otherwise create non-monotonic step sequences in the CSV).
  """
  import csv as _csv
  if not os.path.exists(csv_path):
    return
  try:
    with open(csv_path, 'r', newline='') as fh:
      reader = _csv.DictReader(fh)
      fieldnames = reader.fieldnames
      if not fieldnames:
        return
      rows = [r for r in reader
              if int(float(r.get('learner_steps', max_iteration + 1)))
              <= max_iteration]
    with open(csv_path, 'w', newline='') as fh:
      writer = _csv.DictWriter(fh, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)
    print(f'[ppo] truncated {csv_path} to learner_steps<={max_iteration} '
          f'({len(rows)} rows kept)')
  except Exception as exc:
    print(f'[ppo] warning: could not truncate {csv_path}: {exc}')


def _bb_ep_metrics_from_eval_steps(
    steps,
    obs_dim: int,
    start_index: int,
    end_index: int,
    episode_length: int,
) -> list:
  """Per-env eval metrics from a BuilderBench GPU eval scan."""
  rewards = np.asarray(steps['reward'], dtype=np.float32)
  success = np.asarray(steps['success'], dtype=np.float32)
  state_obs = np.asarray(steps['state_obs'], dtype=np.float32)
  goals = np.asarray(steps['goal'], dtype=np.float32)
  ei = int(obs_dim if end_index == -1 else end_index)
  si = int(start_index)
  goal_slice = goals[..., : max(1, ei - si)]
  state_slice = state_obs[..., si:ei]
  dists = np.linalg.norm(state_slice - goal_slice, axis=-1)

  ep_metrics_list = []
  for i in range(rewards.shape[1]):
    ep_metrics_list.append({
        'episode_return': float(rewards[:, i].sum()),
        'episode_length': int(episode_length),
        'success': float(np.max(success[:, i]) >= 0.5),
        'init_dist': float(dists[0, i]),
        'final_dist': float(dists[-1, i]),
        'delta_dist': float(dists[0, i] - dists[-1, i]),
        'min_dist': float(np.min(dists[:, i])),
    })
  return ep_metrics_list


def _smooth_bb_eval_metrics(
    ep_metrics_list: list,
    success_obs,
    dist_obs,
) -> list:
  """Attach running success_1000 / dist_* smoothers (matches observer semantics)."""
  smoothed = []
  for ep_m in ep_metrics_list:
    success_obs._success.append(bool(ep_m['success'] >= 0.5))
    ep_out = dict(ep_m)
    ep_out['success_1000'] = float(np.mean(success_obs._success[-1000:]))
    for key in ('init_dist', 'final_dist', 'delta_dist', 'min_dist'):
      dist_obs._history.setdefault(key, []).append(float(ep_m[key]))
    if dist_obs._smooth:
      for key, vec in dist_obs._history.items():
        for size in (10, 100, 1000):
          ep_out[f'{key}_{size}'] = float(np.nanmean(vec[-size:]))
    smoothed.append(ep_out)
  return smoothed


# ---------------------------------------------------------------------------
# Top-level training loop.
# ---------------------------------------------------------------------------
def run_ppo_training(
    config: contrastive_config.ContrastiveConfig,
    env_factory: Callable,
    eval_env_factory: Callable,
    network_factory: Callable,
    logger_fn: Callable,
    total_steps: int,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
    builderbench_kwargs: Optional[Dict[str, Any]] = None,
):
  """Top-level PPO-on-φ·ψ training loop.

  Mirrors CleanRL's ppo_continuous_action.py main block:
    for iteration in 1..num_iterations:
      collect rollout of T steps × E envs   (logprob, value, reward = φ·ψ)
      compute GAE advantages / returns
      for epoch in 1..num_epochs:
        for minibatch: PPO update on flat (T·E) batch
      for _ in 1..ppo_crl_steps_per_iter:   # CRL off-policy
        sample future-goal batch from EpisodeReplay, CRL SGD step
      log metrics; periodically evaluate

  The CRL side is InfoNCE + logsumexp penalty on replay batches only.
  The PPO side follows CleanRL.
  """
  # ---- build networks from env spec -------------------------------------
  from acme import specs as _specs
  import contrastive.utils as _cu

  probe_env = env_factory(seed)
  spec = _specs.make_environment_spec(probe_env)
  networks = network_factory(spec=spec)
  del probe_env

  # ---- vec env ----------------------------------------------------------
  _env_name = str(getattr(config, 'env_name', '') or '')
  _use_jax_bb_vec = _env_name.startswith('builderbench_')
  if _use_jax_bb_vec:
    import importlib.util as _ilu
    _jax_vec_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'envs', 'builderbench_jax_vec.py')
    _spec = _ilu.spec_from_file_location('builderbench_jax_vec', _jax_vec_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    JaxBuilderBenchVecEnv = _mod.JaxBuilderBenchVecEnv
    _bb_kw = dict(builderbench_kwargs or {})
    vec_env = JaxBuilderBenchVecEnv(
        env_name=_env_name,
        num_envs=int(config.ppo_num_envs),
        seed=int(seed * 31),
        use_pd=bool(_bb_kw.get('builderbench_use_pd', False)),
        pd_duration=int(_bb_kw.get('builderbench_pd_duration', 5)),
        pd_filter_policy_obs=bool(
            _bb_kw.get('builderbench_pd_filter_policy_obs', True)),
        fixed_target_goal=_bb_kw.get('fixed_target_goal'),
        obs_space_list=_bb_kw.get('obs_space_list'), # <--- ADD THIS LINE
        episode_length_multiplier=float(
                _bb_kw.get('builderbench_episode_length_multiplier', 1.0)),

    )
    print(f'[ppo] using JAX-batched BuilderBench vec env '
          f'(E={config.ppo_num_envs})')
  else:
    vec_env = VecEnv(env_factory, config.ppo_num_envs, seed=seed * 31)
  E = vec_env.num_envs
  obs_shape = vec_env.observation_shape
  act_shape = vec_env.action_shape
  T = int(config.ppo_rollout_length)
  batch_per_iter = T * E
  mb_size = batch_per_iter // int(config.ppo_num_minibatches)
  assert mb_size * int(config.ppo_num_minibatches) == batch_per_iter, (
      'batch_per_iter must divide evenly into ppo_num_minibatches')
  num_iterations = int(total_steps) // (T * E)

  # ---- density estimator mode ------------------------------------------
  # 'crl'      (default) — φ(s,a)·ψ(g) contrastive representations.
  # 'gaussian' — diagonal Gaussian p_θ(g|s,a);  reward = log p_θ(g|s_t,a_t).
  # 'nf'       — conditional RealNVP  log p_NF(g|s,a).
  repr_mode    = (getattr(config, 'ppo_repr_mode', 'crl') or 'crl').strip().lower()
  use_gaussian = repr_mode == 'gaussian'
  use_nf       = repr_mode == 'nf'
  density_nets    = None
  nf_density_nets = None

  obs_dim_cfg   = int(config.obs_dim)
  total_obs_dim = int(np.prod(obs_shape))
  goal_dim_cfg  = total_obs_dim - obs_dim_cfg
  act_dim_cfg   = int(np.prod(act_shape))

  if use_gaussian:
    density_nets = _gd.make_gaussian_density_networks(
        obs_dim=obs_dim_cfg,
        act_dim=act_dim_cfg,
        goal_dim=goal_dim_cfg,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )
    print(f'[ppo] repr_mode=gaussian  obs_dim={obs_dim_cfg}  '
          f'act_dim={act_dim_cfg}  goal_dim={goal_dim_cfg}  '
          f'hidden_layers={config.hidden_layer_sizes}')
  elif use_nf:
    nf_rep_size      = int(getattr(config, 'nf_rep_size', 64))
    nf_num_blocks    = int(getattr(config, 'nf_num_blocks', 8))
    nf_coupling_w    = int(getattr(config, 'nf_coupling_width', 256))
    nf_goal_enc_size = int(getattr(config, 'nf_goal_enc_size', 0))
    nf_density_nets = _nf.make_nf_density_networks(
        obs_dim=obs_dim_cfg,
        act_dim=act_dim_cfg,
        goal_dim=goal_dim_cfg,
        hidden_layer_sizes=config.hidden_layer_sizes,
        rep_size=nf_rep_size,
        num_blocks=nf_num_blocks,
        channels=nf_coupling_w,
        goal_enc_size=nf_goal_enc_size,
    )
    _goal_enc_desc = (f'goal_encoder=2x256+swish→{nf_goal_enc_size}'
                      if nf_goal_enc_size > 0 else 'goal_encoder=none (raw goal)')
    print(f'[ppo] repr_mode=nf (RealNVP)  obs_dim={obs_dim_cfg}  '
          f'act_dim={act_dim_cfg}  goal_dim={goal_dim_cfg}  '
          f'rep_size={nf_rep_size}  num_blocks={nf_num_blocks}  '
          f'coupling_width={nf_coupling_w}  flow_dim={nf_density_nets.flow_dim}  '
          f'sa_encoder=4x1024+swish  {_goal_enc_desc}')
  else:
    print(f'[ppo] repr_mode=crl (φ·ψ contrastive)')

  # ---- init params ------------------------------------------------------
  key = jax.random.PRNGKey(seed)
  k_pol, k_val, k_q, key = jax.random.split(key, 4)
  policy_params = networks.policy_network.init(k_pol)
  value_params = networks.value_network.init(k_val)
  # q_params holds the density-estimator params in all modes:
  #   crl mode      → CRL (φ, ψ) contrastive params from networks.q_network
  #   gaussian mode → Gaussian density p_θ(g|s,a) params from density_nets
  #   nf mode       → RealNVP log p_NF(g|s,a) params from nf_density_nets
  if use_gaussian:
    q_params = density_nets.density_net.init(k_q)
  elif use_nf:
    q_params = _nf.init_nf_params(nf_density_nets, k_q)
  else:
    q_params = networks.q_network.init(k_q)
  ppo_params = {'policy': policy_params, 'value': value_params}

  # ---- optimizers -------------------------------------------------------
  if config.ppo_anneal_lr:
    total_ppo_updates = (
      num_iterations
      * int(config.ppo_num_epochs)
      * int(config.ppo_num_minibatches)
    )

    lr_schedule = optax.linear_schedule(
      init_value=float(config.learning_rate),
      end_value=0.0,
      transition_steps=max(1, total_ppo_updates),
    )
    # A single Adam with schedule + grad-clip (CleanRL-style).
    ppo_optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
        optax.scale_by_adam(eps=1e-5),
        optax.scale_by_schedule(lambda count: -lr_schedule(count)))
  else:
    ppo_optimizer = optax.chain(
        optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
        optax.adam(float(config.learning_rate), eps=1e-5))
  ppo_opt_state = ppo_optimizer.init(ppo_params)

  if use_nf:
    q_optimizer = _nf.make_nf_optimizers(
        encoder_lr=float(getattr(config, 'nf_encoder_lr', 3e-4)),
        critic_lr=float(getattr(config, 'nf_critic_lr', 1e-4)),
        critic_weight_decay=float(getattr(config, 'nf_critic_weight_decay', 1e-6)),
        grad_clip=float(getattr(config, 'nf_grad_clip', 1.0)),
        has_goal_encoder=(nf_density_nets.goal_encoder_net is not None),
    )
    _ge_opt_desc = (f', goal_encoder Adam(lr={config.nf_encoder_lr})'
                    if nf_density_nets.goal_encoder_net is not None else '')
    print(f'[ppo] NF optimizers: SA encoder Adam(lr={config.nf_encoder_lr})'
          f'{_ge_opt_desc}, '
          f'flow AdamW(lr={config.nf_critic_lr}, wd={config.nf_critic_weight_decay}), '
          f'grad_clip={getattr(config, "nf_grad_clip", 1.0)}')
  else:
    q_optimizer = optax.adam(float(config.learning_rate))
  q_opt_state = q_optimizer.init(q_params)

  # ---- optional EMA of φ, ψ for PPO reward (CRL mode only) --------------
  _repr_tau = float(getattr(config, 'ppo_crl_repr_tau', 0.0))
  _use_repr_ema = (
      not use_gaussian and not use_nf
      and _repr_tau > 0.0 and _repr_tau < 1.0)
  q_params_reward = _tree_copy(q_params) if _use_repr_ema else q_params

  # ---- resume from checkpoint if one exists -----------------------------
  start_iteration = 0
  global_step = 0
  ppo_sgd_step = 0
  if checkpoint_dir is not None:
    _latest = os.path.join(checkpoint_dir, 'latest.pkl')
    if os.path.exists(_latest):
      _ckpt = load_checkpoint(_latest)
      policy_params = _ckpt['policy_params']
      value_params  = _ckpt['value_params']
      q_params      = _ckpt['q_params']
      ppo_opt_state = _ckpt['ppo_optimizer_state']
      q_opt_state   = _ckpt['q_optimizer_state']
      ppo_params    = {'policy': policy_params, 'value': value_params}
      start_iteration = int(_ckpt['iteration']) + 1
      global_step     = int(_ckpt['global_step'])
      key             = _ckpt['key']
      ppo_sgd_step    = (start_iteration
                         * int(config.ppo_num_epochs)
                         * int(config.ppo_num_minibatches))
      if _use_repr_ema:
        q_params_reward = (_ckpt['q_params_ema']
                           if 'q_params_ema' in _ckpt
                           else _tree_copy(q_params))
      print(f'[ppo] resumed from checkpoint: '
            f'start_iteration={start_iteration}, global_step={global_step}')
      # Truncate CSV logs to remove any entries written after the checkpoint
      # (can happen if training ran past the last checkpoint before preemption).
      _run_dir = os.path.dirname(checkpoint_dir)
      for _label in ('learner', 'eval'):
        _csv_path = os.path.join(_run_dir, 'logs', _label, 'logs.csv')
        _truncate_csv_to_iteration(_csv_path, int(_ckpt['iteration']))

  # ---- jitted helpers ---------------------------------------------------
  reward_mode = (getattr(config, 'ppo_reward_mode', '') or '').strip().lower()
  use_dirac_target = reward_mode == 'dirac_target'
  use_kde_dirac    = reward_mode == 'kde_dirac'
  reward_fn = make_reward_fn(networks, config)
  if use_dirac_target:
    print(f'[ppo] reward mode: dirac_target (eps={config.ppo_dirac_eps})')
  if use_kde_dirac:
    kde_max_points     = int(getattr(config, 'kde_max_points', 2000))
    kde_refit_interval = int(getattr(config, 'kde_refit_interval', 1))
    kde_bandwidth_cfg  = float(getattr(config, 'kde_bandwidth', 0.0))
    kde_bandwidth_arg  = kde_bandwidth_cfg if kde_bandwidth_cfg > 0.0 else None
    kde_state: Optional[GaussianKDE] = None
    print(f'[ppo] reward mode: kde_dirac  (eps={config.ppo_dirac_eps}, '
          f'max_points={kde_max_points}, refit_interval={kde_refit_interval}, '
          f'bandwidth={"Scott" if kde_bandwidth_arg is None else kde_bandwidth_cfg})')
  gae_fn = make_gae_fn(config)
  ppo_update = make_ppo_update_fn(networks, config, ppo_optimizer)

  if use_gaussian:
    crl_update = _gd.make_gaussian_density_update_fn(
        density_nets, q_optimizer, obs_dim=int(config.obs_dim))
    gaussian_reward_fn = _gd.make_gaussian_reward_fn(
        density_nets, obs_dim=int(config.obs_dim))
    nf_reward_fn = None
  elif use_nf:
    crl_update = _nf.make_nf_density_update_fn(
        nf_density_nets, q_optimizer, obs_dim=int(config.obs_dim),
        noise_std=float(getattr(config, 'nf_noise_std', 0.0)))
    nf_reward_fn = _nf.make_nf_reward_fn(
        nf_density_nets, obs_dim=int(config.obs_dim))
    gaussian_reward_fn = None
  else:
    _direction = str(getattr(config, 'ppo_crl_loss_direction', 'forward')).lower()
    _backward = (_direction == 'backward')
    crl_update = make_crl_update_fn(networks, q_optimizer, backward=_backward)
    # Scan-based multi-step updater: one JIT dispatch for all CRL steps.
    crl_scan_update = make_scan_crl_update_fn(
        networks, q_optimizer, backward=_backward, repr_tau=_repr_tau)
    onpolicy_crl_update = make_onpolicy_crl_update_fn(
        networks, q_optimizer, config, backward=_backward, repr_tau=_repr_tau)
    print(f'[ppo] CRL loss direction: {_direction}')
    if _use_repr_ema:
      print(f'[ppo] CRL reward repr EMA: tau={_repr_tau} '
            f'(InfoNCE still uses online ╧å, ╧ê)')
    gaussian_reward_fn = None
    nf_reward_fn = None

  def _reward_q_params():
    return q_params_reward if _use_repr_ema else q_params

  @jax.jit
  def act_and_value(policy_p, value_p, obs, rng):
    dist = networks.policy_network.apply(policy_p, obs)
    action = networks.sample(dist, rng)
    logprob = networks.log_prob(dist, action)
    value = networks.value_network.apply(value_p, obs)
    return action, logprob, value

  bb_generate_unroll = None
  bb_eval_unroll = None
  bb_eval_vec = None
  if _use_jax_bb_vec:
    bb_generate_unroll = vec_env.compile_generate_unroll(
        act_and_value,
        unroll_length=T,
        obs_dim=int(config.obs_dim))
    print(f'[ppo] BuilderBench rollout: jax.lax.scan (T={T})')
    _eval_iv = int(getattr(config, 'ppo_eval_interval', 10))
    if _eval_iv > 0:
      _n_eval = int(getattr(config, 'ppo_eval_episodes', 5))
      bb_eval_vec = JaxBuilderBenchVecEnv(
          env_name=_env_name,
          num_envs=_n_eval,
          seed=int(seed * 31 + 77),
          use_pd=bool(_bb_kw.get('builderbench_use_pd', False)),
          pd_duration=int(_bb_kw.get('builderbench_pd_duration', 5)),
          fixed_target_goal=_bb_kw.get('fixed_target_goal'),
          episode_length_multiplier=float(
                  _bb_kw.get('builderbench_episode_length_multiplier', 1.0)),
          obs_space_list=_bb_kw.get('obs_space_list')
      )

      @jax.jit
      def eval_policy_action(policy_p, obs):
        dist = networks.policy_network.apply(policy_p, obs)
        return dist.mode()

      bb_eval_unroll = bb_eval_vec.compile_eval_unroll(
          eval_policy_action,
          unroll_length=bb_eval_vec.episode_length)
      print(f'[ppo] BuilderBench eval: jax.lax.scan '
            f'(E={_n_eval}, ep_len={bb_eval_vec.episode_length}, '
            f'every {_eval_iv} iters)')
    else:
      print('[ppo] BuilderBench: periodic eval disabled; '
            'logging train_success_mean / train_success_1000 from rollouts')

  @jax.jit
  def value_only(value_p, obs):
    return networks.value_network.apply(value_p, obs)

  @jax.jit
  def greedy_action(policy_p, obs):
    # Deterministic policy mean for evaluation.
    dist = networks.policy_network.apply(policy_p, obs)
    return networks.sample(dist, jax.random.PRNGKey(0))  # sample still; we log both below

  # ---- uniform-sampling goal bounds (extracted once from env spec) ------
  uniform_sampling = bool(getattr(config, 'uniform_sampling', False))
  goal_low = goal_high = None
  if uniform_sampling:
    import env_utils as _env_utils
    if hasattr(vec_env, 'uniform_goal_obs_bounds'):
      glo, ghi = vec_env.uniform_goal_obs_bounds()
      si, ei = int(config.start_index), int(config.end_index)
      if ei == -1:
        ei = int(config.obs_dim)
      goal_low = np.asarray(glo[si:ei], dtype=np.float32)
      goal_high = np.asarray(ghi[si:ei], dtype=np.float32)
    else:
      goal_low, goal_high = _env_utils.resolve_uniform_goal_bounds(
          spec, vec_env._envs[0], int(config.obs_dim),
          int(config.start_index), int(config.end_index))
    print(f'[ppo] uniform_sampling: goal_low={goal_low}, goal_high={goal_high}')

  # ---- replay buffer (episodes) -----------------------------------------
  replay = EpisodeReplay(
      capacity=int(config.max_replay_size),
      obs_dim=int(config.obs_dim),
      discount=float(config.discount),
      start_index=int(config.start_index),
      end_index=int(config.end_index))
  np_rng = np.random.default_rng(seed + 12345)

  # ---- per-env episode buffers (for flushing complete trajectories) -----
  ep_obs: list = [[] for _ in range(E)]
  ep_act: list = [[] for _ in range(E)]
  ep_return = np.zeros(E, dtype=np.float32)
  ep_flow_dense_return = np.zeros(E, dtype=np.float32)
  ep_has_flow_dense = np.zeros(E, dtype=bool)
  ep_len = np.zeros(E, dtype=np.int32)
  recent_returns: list = []
  recent_flow_dense_returns: list = []
  recent_lengths: list = []
  recent_success: list = []
  ep_success_max = np.zeros(E, dtype=np.float32)
  _track_train_success = _use_jax_bb_vec

  # Running goal normalisation stats for NF mode.
  # Updated from replay buffer each iteration; broadcast-compatible with goals.
  nf_goal_mean = np.zeros(goal_dim_cfg, dtype=np.float32)
  nf_goal_std  = np.ones(goal_dim_cfg,  dtype=np.float32)

  obs = vec_env.reset()
  next_done = np.zeros(E, dtype=np.float32)
  s0_states = np.asarray(obs[:, :int(config.obs_dim)], dtype=np.float32).copy()
  for i in range(E):
    ep_obs[i].append(obs[i].copy())

  if getattr(config, 'staggered_resets', False) and hasattr(vec_env, 'stagger_resets'):
    print(f'[ppo] Applying staggered resets to {E} environments...')
    vec_env.stagger_resets()
    if _use_jax_bb_vec:
      obs_packed = vec_env.pack_obs_from_state(vec_env._state)
      obs = np.asarray(obs_packed, dtype=np.float32)
      s0_states = np.asarray(obs[:, :int(config.obs_dim)], dtype=np.float32).copy()

  # ---- loggers ----------------------------------------------------------
  learner_logger = logger_fn(label='learner')
  eval_logger = logger_fn(label='eval')

  # Persistent eval observers (mirrors Acme's evaluator loop). Keeping
  # them alive across iterations is what lets `success_1000` and
  # `*_dist_{10,100,1000}` smooth over eval history.
  # RiverSwim: success = visited goal cell at least once (not env +1 reward).
  _env = str(getattr(config, 'env_name', '') or '').lower()
  if _env == 'riverswim':
    eval_success_obs = _cu.RiverSwimGoalVisitSuccessObserver(
        obs_dim=int(config.obs_dim))
  elif _env.startswith('builderbench_'):
    eval_success_obs = _cu.BuilderBenchSuccessObserver()
  else:
    eval_success_obs = _cu.SuccessObserver()
  eval_dist_obs = _cu.DistanceObserver(
      obs_dim=int(config.obs_dim),
      start_index=int(config.start_index),
      end_index=int(config.end_index))

  # ---- rollout storage (reused each iteration) --------------------------
  roll_obs   = np.zeros((T, E) + obs_shape, dtype=np.float32)
  roll_acts  = np.zeros((T, E) + act_shape, dtype=np.float32)
  roll_logp  = np.zeros((T, E), dtype=np.float32)
  roll_rew   = np.zeros((T, E), dtype=np.float32)          # reps reward (possibly normalized)
  roll_rew_raw = np.zeros((T, E), dtype=np.float32)        # reps reward (pre-normalization, log only)
  roll_env_rew = np.zeros((T, E), dtype=np.float32)        # gt reward (log only)
  roll_flow_dense_rew = np.full((T, E), np.nan, dtype=np.float32)
  roll_dones = np.zeros((T, E), dtype=np.float32)
  roll_vals  = np.zeros((T, E), dtype=np.float32)
  # Per-step dones from env (used for reward normalization after rollout).
  roll_step_dones = np.zeros((T, E), dtype=np.float32)
  # s0 state tracked for dirac_target: batched reward call after rollout.
  roll_s0_states = np.zeros(
      (T, E, int(config.obs_dim)), dtype=np.float32)

  # ---- reward normalizer (CleanRL NormalizeReward) ----------------------
  # Normalizes the reps-based reward by the running std of discounted
  # returns. Critical for PPO with learned rewards — raw ╧å┬╖╧ê values can
  # be O(10) and non-stationary, leading to unbounded advantages and
  # policy collapse within a handful of updates.
  norm_reward = bool(getattr(config, 'ppo_norm_reward', True))
  ppo_gamma = (float(config.ppo_discount)
               if float(getattr(config, 'ppo_discount', -1.0)) > 0.0
               else float(config.discount))
  _return_norm_window = int(getattr(config, 'ppo_return_norm_window', 0))
  reward_normalizer = (
      ReturnNormalizer(num_envs=E, discount=ppo_gamma,
                       window_size=_return_norm_window)
      if norm_reward else None)
  _nf_normalizer_reset_done = False  # reset once when NF first activates
  _nf_stat_log: Dict[str, float] = {}  # per-dim goal mean/std, updated each iter

  # ---- checkpointing ----------------------------------------------------
  ckpt_interval = int(getattr(config, 'ppo_checkpoint_interval', 0))
  ckpt_keep_last = int(getattr(config, 'ppo_checkpoint_keep_last', 0))
  if ckpt_interval > 0 and checkpoint_dir is not None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    keep_msg = ('keep all milestones'
                if ckpt_keep_last <= 0
                else f'keep last {ckpt_keep_last} milestones')
    print(f'[ppo] checkpoints -> {checkpoint_dir} '
          f'(every {ckpt_interval} iters, {keep_msg})')

  warmup_percent = float(getattr(config, 'ppo_warmup_percent', 0.0))
  warmup_iters = int(num_iterations * warmup_percent)
  if warmup_iters > 0:
    print(f'[ppo] Warmup phase: PPO updates delayed for first {warmup_iters} iterations '
          f'({warmup_percent * 100:.1f}%). CRL will train on random rollouts.')

  start_time = time.time()
  # global_step, ppo_sgd_step, start_iteration set above (0 for fresh runs,
  # restored from checkpoint on resume).

  for iteration in range(start_iteration, num_iterations):
    is_warmup = iteration < warmup_iters
    # =================================================================
    # 1. Rollout (on-policy, CleanRL convention)
    # =================================================================
    if bb_generate_unroll is not None:
      # BuilderBench: fused policy + env unroll via jax.lax.scan (GPU).
      (vec_env._state, key, next_done, _), _steps_j = bb_generate_unroll(
          vec_env._state,
          ppo_params['policy'],
          ppo_params['value'],
          key,
          jnp.asarray(next_done),
          jnp.asarray(s0_states),
      )
      roll_obs[:] = np.asarray(_steps_j['obs'], dtype=np.float32)
      if iteration == start_iteration:
        print("\n" + "="*80)
        print(f"[VERIFICATION] PPO Actor Observation Verification")
        print(f"Requested obs_space_list: {vec_env._obs_space_list}")
        print(f"Expected state_obs_dim: {vec_env._state_obs_dim}")
        print(f"Expected goal_dim:      {vec_env._goal_dim}")
        print(f"Rollout obs matrix shape: {roll_obs.shape} (T, E, total_obs_dim)")
        print(f"Actual total features fed to actor: {roll_obs.shape[-1]}")
        print("="*80 + "\n", flush=True)
      roll_dones[:] = np.asarray(_steps_j['roll_dones'], dtype=np.float32)
      roll_acts[:] = np.asarray(_steps_j['actions'], dtype=np.float32)
      roll_logp[:] = np.asarray(_steps_j['logprobs'], dtype=np.float32)
      roll_vals[:] = np.asarray(_steps_j['values'], dtype=np.float32)
      roll_env_rew[:] = np.asarray(_steps_j['env_rew'], dtype=np.float32)
      roll_step_dones[:] = np.asarray(_steps_j['step_dones'], dtype=np.float32)
      if use_dirac_target:
        roll_s0_states[:] = np.asarray(_steps_j['s0_states'], dtype=np.float32)
      if use_kde_dirac:
        for _t in range(T):
          rep_rew_np = (np.zeros(E, dtype=np.float32) if kde_state is None
                        else reward_fn(kde_state, roll_obs[_t]).astype(np.float32))
          roll_rew_raw[_t] = rep_rew_np
      obs = np.asarray(
          vec_env.pack_obs_from_state(vec_env._state), dtype=np.float32)
      _roll_success = (
          np.asarray(_steps_j['success'], dtype=np.float32)
          if _track_train_success else None)
      for _t in range(T):
        _actions_t = roll_acts[_t]
        _env_rew_t = roll_env_rew[_t]
        _dones_t = roll_step_dones[_t].astype(bool)
        _term_t = np.asarray(_steps_j['terminal_obs'][_t], dtype=np.float32)
        _next_t = np.asarray(_steps_j['next_obs'][_t], dtype=np.float32)
        for i in range(E):
          ep_act[i].append(_actions_t[i].copy())
          ep_return[i] += float(_env_rew_t[i])
          ep_len[i] += 1
          if _roll_success is not None:
            ep_success_max[i] = max(
                ep_success_max[i], float(_roll_success[_t, i]))
          if _dones_t[i]:
            ep_obs[i].append(_term_t[i].copy())
            if not getattr(config, 'crl_on_policy', False):
              try:
                replay.add_episode(
                    np.stack(ep_obs[i], axis=0),
                    np.stack(ep_act[i], axis=0))
              except AssertionError:
                pass
            ep_obs[i] = [_next_t[i].copy()]
            s0_states[i] = _next_t[i, :int(config.obs_dim)].copy()
            ep_act[i] = []
            recent_returns.append(float(ep_return[i]))
            recent_lengths.append(int(ep_len[i]))
            if _roll_success is not None:
              recent_success.append(float(ep_success_max[i] >= 0.5))
              ep_success_max[i] = 0.0
              if len(recent_success) > 1000:
                recent_success.pop(0)
            ep_return[i] = 0.0
            ep_len[i] = 0
            if len(recent_returns) > 100:
              recent_returns.pop(0)
              recent_lengths.pop(0)
          else:
            ep_obs[i].append(_next_t[i].copy())
      global_step += T * E
    else:
      for t in range(T):
        roll_obs[t] = obs
        roll_dones[t] = next_done

        key, k_act = jax.random.split(key)
        action_j, logprob_j, value_j = act_and_value(
            ppo_params['policy'], ppo_params['value'],
            jnp.asarray(obs), k_act)
        action = np.asarray(action_j)
        roll_acts[t] = action
        roll_logp[t] = np.asarray(logprob_j)
        roll_vals[t] = np.asarray(value_j)

        # For dirac_target, record s0 for the batched reward call later.
        if use_dirac_target:
          roll_s0_states[t] = s0_states
        # kde_dirac reward is pure NumPy — compute per-step to avoid storing KDE.
        elif use_kde_dirac:
          rep_rew_np = (np.zeros(E, dtype=np.float32) if kde_state is None
                        else reward_fn(kde_state, obs).astype(np.float32))
          roll_rew_raw[t] = rep_rew_np

        next_obs, env_rew, dones, terminal_obs, info_rew = vec_env.step(action)
        roll_env_rew[t] = env_rew
        roll_flow_dense_rew[t] = info_rew
        # Store step-level dones for the post-rollout reward normalizer loop.
        roll_step_dones[t] = dones.astype(np.float32)

        # Episode flushing / per-env accounting.
        for i in range(E):
          ep_act[i].append(action[i].copy())
          ep_return[i] += float(env_rew[i])
          if not np.isnan(info_rew[i]):
            ep_flow_dense_return[i] += float(info_rew[i])
            ep_has_flow_dense[i] = True
          ep_len[i] += 1
          if dones[i]:
            ep_obs[i].append(terminal_obs[i].copy())
            if not getattr(config, 'crl_on_policy', False):
              try:
                replay.add_episode(
                    np.stack(ep_obs[i], axis=0),
                    np.stack(ep_act[i], axis=0))
              except AssertionError:
                pass  # degenerate len-0 episodes; skip
            ep_obs[i] = [next_obs[i].copy()]  # auto-reset state seeds next ep
            s0_states[i] = next_obs[i, :int(config.obs_dim)].copy()
            ep_act[i] = []
            recent_returns.append(float(ep_return[i]))
            if ep_has_flow_dense[i]:
              recent_flow_dense_returns.append(float(ep_flow_dense_return[i]))
            recent_lengths.append(int(ep_len[i]))
            ep_return[i] = 0.0
            ep_flow_dense_return[i] = 0.0
            ep_has_flow_dense[i] = False
            ep_len[i] = 0
            if len(recent_returns) > 100:
              recent_returns.pop(0)
              recent_lengths.pop(0)
            if len(recent_flow_dense_returns) > 100:
              recent_flow_dense_returns.pop(0)
          else:
            ep_obs[i].append(next_obs[i].copy())

        obs = next_obs
        next_done = dones.astype(np.float32)
        global_step += E

    # =================================================================
    # 1b. Batched reward computation (single GPU call over full rollout)
    # =================================================================
    if not is_warmup:
      # kde_dirac rewards were already filled per-step above (CPU-only).
      if not use_kde_dirac:
        _flat_obs_j  = jnp.asarray(roll_obs.reshape(T * E, -1))
        _flat_acts_j = jnp.asarray(roll_acts.reshape(T * E, -1))
        if use_nf:
          _rew_flat = np.asarray(nf_reward_fn(
              q_params, _flat_obs_j, _flat_acts_j,
              jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std)))
        elif use_gaussian:
          _rew_flat = np.asarray(
              gaussian_reward_fn(q_params, _flat_obs_j, _flat_acts_j))
        elif use_dirac_target:
          _flat_s0_j = jnp.asarray(roll_s0_states.reshape(T * E, -1))
          _rew_flat  = np.asarray(
              reward_fn(_reward_q_params(), _flat_obs_j, _flat_acts_j, _flat_s0_j))
        else:
          # Default CRL: r = ╧å(s,a)┬╖╧ê(g)
          _rew_flat = np.asarray(
              reward_fn(_reward_q_params(), _flat_obs_j, _flat_acts_j))
        roll_rew_raw[:] = _rew_flat.reshape(T, E)

      # Apply reward normalisation
      if reward_normalizer is not None:
        for _t in range(T):
          roll_rew[_t] = reward_normalizer(roll_rew_raw[_t], roll_step_dones[_t])
      else:
        roll_rew[:] = roll_rew_raw
    else:
      roll_rew_raw[:] = np.nan
      roll_rew[:] = np.nan

    # =================================================================
    # 2. GAE advantages / returns
    # =================================================================
    if not is_warmup:
      next_val = np.asarray(value_only(ppo_params['value'], jnp.asarray(obs)))
      adv_j, ret_j = gae_fn(
          jnp.asarray(roll_rew), jnp.asarray(roll_vals),
          jnp.asarray(roll_dones),
          jnp.asarray(next_val), jnp.asarray(next_done))
      adv = np.asarray(adv_j)
      ret = np.asarray(ret_j)
    else:
      adv = np.full((T, E), np.nan, dtype=np.float32)
      ret = np.full((T, E), np.nan, dtype=np.float32)

    # =================================================================
    # 3. PPO updates (epochs ├ù minibatches over flat T┬╖E batch)
    # =================================================================
    ppo_metrics_agg: Dict[str, list] = {}
    early_stop = False
    
    if not is_warmup:
      flat_obs = roll_obs.reshape((batch_per_iter,) + obs_shape)
      flat_acts = roll_acts.reshape((batch_per_iter,) + act_shape)
      flat_logp = roll_logp.reshape(batch_per_iter)
      flat_adv = adv.reshape(batch_per_iter)
      flat_ret = ret.reshape(batch_per_iter)
      flat_vals = roll_vals.reshape(batch_per_iter)

      for epoch in range(int(config.ppo_num_epochs)):
        perm = np_rng.permutation(batch_per_iter)
        last_kl = None
        for start in range(0, batch_per_iter, mb_size):
          mb = perm[start:start + mb_size]
          batch = {
              'obs':          jnp.asarray(flat_obs[mb]),
              'actions':      jnp.asarray(flat_acts[mb]),
              'old_logprobs': jnp.asarray(flat_logp[mb]),
              'advantages':   jnp.asarray(flat_adv[mb]),
              'returns':      jnp.asarray(flat_ret[mb]),
              'old_values':   jnp.asarray(flat_vals[mb]),
          }
          key, k_mb = jax.random.split(key)
          ppo_params, ppo_opt_state, m = ppo_update(
              ppo_params, ppo_opt_state, batch, k_mb)
          ppo_sgd_step += 1
          last_kl = float(m['approx_kl'])
          for k_, v in m.items():
            ppo_metrics_agg.setdefault(k_, []).append(float(v))
        if (config.ppo_target_kl is not None and last_kl is not None
            and last_kl > float(config.ppo_target_kl)):
          early_stop = True
          break

      pg_vals = ppo_metrics_agg.get('pg_loss', [])
      mean_pg = float(np.mean(pg_vals)) if pg_vals else float('inf')
    else:
      mean_pg = float('nan')

    # =================================================================
    # 4. CRL updates (off-policy from replay OR on-policy from rollouts)
    # =================================================================
    crl_metrics_agg: Dict[str, list] = {}
    _n_crl = int(config.ppo_crl_steps_per_iter)

    if getattr(config, 'crl_on_policy', False):
      j_roll_obs = jnp.asarray(roll_obs)
      j_roll_acts = jnp.asarray(roll_acts)
      j_roll_dones = jnp.asarray(roll_step_dones)
      for _ in range(_n_crl):
        key, k_onp = jax.random.split(key)
        q_params, q_opt_state, q_params_reward, key, m = onpolicy_crl_update(
            q_params, q_opt_state, q_params_reward,
            j_roll_obs, j_roll_acts, j_roll_dones, k_onp)
        for k_, v in m.items():
          crl_metrics_agg.setdefault(k_, []).append(float(v))
    elif replay.size >= int(config.ppo_min_replay_size):
      # Update goal normalisation stats from a fresh replay sample (NF only).
      if use_nf and not _nf_normalizer_reset_done:
        # First time NF activates: reset the return normalizer so the extreme
        # rewards from the untrained flow don't permanently corrupt the running
        # std that PPO uses to scale advantages.
        if reward_normalizer is not None:
          reward_normalizer._rms = RunningMeanStd(shape=())
          reward_normalizer._returns = np.zeros(E, dtype=np.float64)
        _nf_normalizer_reset_done = True
      if use_nf:
        _std_floor = float(getattr(config, 'nf_goal_std_min', 0.02))
        _stat_batch = replay.sample(min(2048, replay.size), np_rng)
        _goals = _stat_batch['obs'][:, int(config.obs_dim):]
        # Optionally mix in the actual env goals from the current rollout so
        # that the running stats cover both hindsight goals AND reward goals.
        if bool(getattr(config, 'nf_mix_env_goal_stats', False)):
          _env_goals = roll_obs.reshape(-1, roll_obs.shape[-1])[:, int(config.obs_dim):]
          _goals = np.concatenate([_goals, _env_goals], axis=0)
        nf_goal_mean = _goals.mean(axis=0).astype(np.float32)
        nf_goal_std  = _goals.std(axis=0).astype(np.float32)
        nf_goal_std  = np.maximum(nf_goal_std, _std_floor).astype(np.float32)
        for _di, (_gm, _gs) in enumerate(zip(nf_goal_mean, nf_goal_std)):
          _nf_stat_log[f'nf/goal_mean_{_di}'] = float(_gm)
          _nf_stat_log[f'nf/goal_std_{_di}']  = float(_gs)

      if use_nf or use_gaussian:
        # NF / Gaussian loops take extra args — keep the Python loop for now.
        for _ in range(_n_crl):
          if uniform_sampling:
            crl_batch_np = replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
          else:
            crl_batch_np = replay.sample(int(config.batch_size), np_rng)
          crl_batch = {k_: jnp.asarray(v) for k_, v in crl_batch_np.items()}
          key, k_crl = jax.random.split(key)
          if use_nf:
            q_params, q_opt_state, m = crl_update(
                q_params, q_opt_state, crl_batch, k_crl,
                jnp.asarray(nf_goal_mean), jnp.asarray(nf_goal_std))
          else:
            q_params, q_opt_state, m = crl_update(
                q_params, q_opt_state, crl_batch, k_crl)
          if _use_repr_ema:
            q_params_reward = _ema_tree(q_params_reward, q_params, _repr_tau)
          for k_, v in m.items():
            crl_metrics_agg.setdefault(k_, []).append(float(v))
      else:
        # Standard CRL: pre-sample all batches -> one H->D transfer -> one JIT.
        # This eliminates _n_crl rounds of dispatch + host sync.
        _samples = [
            (replay.sample_with_uniform_negatives(
                int(config.batch_size), np_rng, goal_low, goal_high)
             if uniform_sampling
             else replay.sample(int(config.batch_size), np_rng))
            for _ in range(_n_crl)]
        _stacked = {
            k_: jnp.asarray(np.stack([s[k_] for s in _samples], axis=0))
            for k_ in _samples[0]}
        (q_params, q_opt_state, q_params_reward,
         key, m) = crl_scan_update(
            q_params, q_opt_state, q_params_reward, _stacked, key)
        crl_metrics_agg = {k_: [float(v)] for k_, v in m.items()}

    # =================================================================
    # 4b. KDE refit (kde_dirac mode only)
    # =================================================================
    if use_kde_dirac and replay.size >= int(config.ppo_min_replay_size):
      if iteration % kde_refit_interval == 0:
        kde_state = GaussianKDE.fit(
            replay, int(config.obs_dim),
            kde_max_points, np_rng, kde_bandwidth_arg)

    # =================================================================
    # 5. Logging
    # =================================================================
    elapsed = time.time() - start_time
    # flow-dense reward: only log if this env actually emits it.
    _has_flow_dense = not np.all(np.isnan(roll_flow_dense_rew))

    log = {
        'iteration':          iteration,
        'learner_steps':      iteration,
        'global_step':        global_step,
        'sps':                global_step / max(1e-6, elapsed),
        'replay_size':        int(replay.size),
        'reward_repr_mean':      float(np.nanmean(roll_rew)) if not is_warmup else float('nan'),
        'reward_repr_raw_mean':  float(np.nanmean(roll_rew_raw)) if not is_warmup else float('nan'),
        'reward_repr_raw_std':   float(np.nanstd(roll_rew_raw)) if not is_warmup else float('nan'),
        'reward_return_norm_std': (
            float(reward_normalizer.std) if reward_normalizer is not None and not is_warmup
            else float('nan')),
        'reward_env_mean':       float(roll_env_rew.mean()),
        'value_mean':        float(roll_vals.mean()),
        'returns_mean':      float(np.nanmean(ret)) if not is_warmup else float('nan'),
        'advantage_mean':    float(np.nanmean(adv)) if not is_warmup else float('nan'),
        'advantage_std':     float(np.nanstd(adv)) if not is_warmup else float('nan'),
        'early_stop_epochs': int(early_stop),
        'ep_return_mean':    float(np.mean(recent_returns)) if recent_returns else float('nan'),
        'ep_length_mean':    float(np.mean(recent_lengths)) if recent_lengths else float('nan'),
        'ppo/mean_pg_loss':  mean_pg,
    }

    if _track_train_success:
      log['train_success_mean'] = (
          float(np.mean(recent_success[-100:]))
          if recent_success else float('nan'))
      log['train_success_1000'] = (
          float(np.mean(recent_success[-1000:]))
          if recent_success else float('nan'))

    # Acme CSVLogger fixes columns on the *first* write and drops any later
    # keys. Seed NF/SA columns from iter 0 so training metrics land in CSV.
    if use_nf:
      log.update({
          'nf/density_loss': float('nan'),
          'nf/log_p_mean': float('nan'),
          'nf/log_p_min': float('nan'),
          'nf/log_p_max': float('nan'),
          'nf/flow_grad_norm': float('nan'),
          'nf/update_skipped_nonfinite': float('nan'),
          'nf/update_steps': 0,
          'sa/repr_norm': float('nan'),
          'sa/encoder_grad_norm': float('nan'),
      })
      if nf_density_nets.goal_encoder_net is not None:
        log['nf/goal_enc_grad_norm'] = float('nan')
      for _di in range(goal_dim_cfg):
        log[f'nf/goal_mean_{_di}'] = float('nan')
        log[f'nf/goal_std_{_di}']  = float('nan')

    # Flow-dense benchmark reward: only when the env provides it.
    if _has_flow_dense:
      log['reward_flow_dense_mean'] = float(np.nanmean(roll_flow_dense_rew))
      log['ep_flow_dense_return_mean'] = (
          float(np.mean(recent_flow_dense_returns))
          if recent_flow_dense_returns else float('nan'))

    if is_warmup:
      expected_ppo_keys = [
          'ppo_total_loss', 'pg_loss', 'v_loss', 'entropy', 'entropy_loss',
          'approx_kl', 'old_approx_kl', 'clipfrac', 'ratio_mean',
          'policy_loc_mean', 'policy_loc_abs_mean', 'policy_scale_mean', 'policy_scale_min'
      ]
      for k in expected_ppo_keys:
        log[f'ppo/{k}'] = float('nan')
    else:
      for k_, vs in ppo_metrics_agg.items():
        log[f'ppo/{k_}'] = float(np.mean(vs))

    # Density-estimator metrics: only the active mode.
    if use_gaussian:
      for k_, vs in crl_metrics_agg.items():
        log[f'gaussian/{k_}'] = float(np.mean(vs))
    elif use_nf:
      _sa_keys = frozenset({'repr_norm', 'encoder_grad_norm'})
      _ge_keys = frozenset({'goal_enc_grad_norm'})
      for k_, vs in crl_metrics_agg.items():
        if k_ in _sa_keys:
          prefix = 'sa'
        elif k_ in _ge_keys:
          prefix = 'nf'
        else:
          prefix = 'nf'
        log[f'{prefix}/{k_}'] = float(np.mean(vs))
      log['nf/update_steps'] = len(crl_metrics_agg.get('density_loss', []))
    else:
      for k_, vs in crl_metrics_agg.items():
        log[f'crl/{k_}'] = float(np.mean(vs))
    if config.ppo_anneal_lr:
      lr_log = float(lr_schedule(max(0, ppo_sgd_step - 1)))
    else:
      lr_log = float(config.learning_rate)
    # Seven fractional digits so CSV / terminal show stable small LRs.
    log['ppo/learning_rate'] = round(lr_log, 7)
    log.update(_nf_stat_log)
    learner_logger.write(log)

    # =================================================================
    # 6. Periodic evaluation
    # =================================================================
    _skip_first = bool(getattr(config, 'ppo_skip_first_eval', False))
    _eval_interval = int(getattr(config, 'ppo_eval_interval', 10))
    _n_eval = int(getattr(config, 'ppo_eval_episodes', 5))
    if (_eval_interval > 0
        and iteration % _eval_interval == 0
        and not (iteration == 0 and _skip_first)):
      if bb_eval_unroll is not None and bb_eval_vec is not None:
        eval_state = bb_eval_vec.reset_state()
        steps_j = bb_eval_unroll(eval_state, ppo_params['policy'])
        ep_metrics_list = _bb_ep_metrics_from_eval_steps(
            steps_j,
            obs_dim=int(config.obs_dim),
            start_index=int(config.start_index),
            end_index=int(config.end_index),
            episode_length=int(bb_eval_vec.episode_length),
        )
        ep_metrics_list = _smooth_bb_eval_metrics(
            ep_metrics_list, eval_success_obs, eval_dist_obs)
      else:
        ep_metrics_list = []
        for e_i in range(_n_eval):
          env = eval_env_factory(seed + 900_000 + iteration * 100 + e_i)
          ts = env.reset()
          eval_success_obs.observe_first(env, ts)
          eval_dist_obs.observe_first(env, ts)
          ret_e, n_e = 0.0, 0
          flow_dense_ret_e = 0.0
          flow_dense_steps = 0
          while not ts.last():
            key, k_eval = jax.random.split(key)
            a, _, _ = act_and_value(
                ppo_params['policy'], ppo_params['value'],
                jnp.asarray(ts.observation)[None], k_eval)
            action = np.asarray(a)[0].astype(np.float32)
            action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = np.clip(action, -1.0, 1.0)
            ts = env.step(action)
            eval_success_obs.observe(env, ts, action)
            eval_dist_obs.observe(env, ts, action)
            ret_e += float(ts.reward or 0.0)
            dense_r = _cu.extract_info_reward(env)
            if not np.isnan(dense_r):
              flow_dense_ret_e += dense_r
              flow_dense_steps += 1
            n_e += 1
          ep_metrics = {'episode_return': ret_e, 'episode_length': n_e}
          if flow_dense_steps > 0:
            ep_metrics.update(
                _cu.flow_dense_eval_episode_metrics(
                    flow_dense_ret_e, flow_dense_steps))
          ep_metrics.update(eval_success_obs.get_metrics())
          ep_metrics.update(eval_dist_obs.get_metrics())
          ep_metrics_list.append(ep_metrics)

      agg = _cu.aggregate_eval_metrics(ep_metrics_list, iteration)
      eval_logger.write(agg)

    # =================================================================
    # 7. Checkpointing: ckpt_iter_{iter}.pkl every `ppo_checkpoint_interval`
    #    iters + rolling latest.pkl. Prune only if ppo_checkpoint_keep_last>0.
    # =================================================================
    if (ckpt_interval > 0
        and checkpoint_dir is not None
        and (iteration % ckpt_interval == 0
             or iteration == num_iterations - 1)):
      ckpt_kw = dict(
          policy_params=ppo_params['policy'],
          value_params=ppo_params['value'],
          q_params=q_params,
          ppo_opt_state=ppo_opt_state,
          q_opt_state=q_opt_state,
          iteration=iteration,
          global_step=global_step,
          key=key,
          q_params_ema=(q_params_reward if _use_repr_ema else None))
      milestone_path = os.path.join(
          checkpoint_dir, f'ckpt_iter_{iteration:07d}.pkl')
      _save_checkpoint(milestone_path, **ckpt_kw)
      _save_checkpoint(os.path.join(checkpoint_dir, 'latest.pkl'), **ckpt_kw)
      _prune_old_checkpoints(checkpoint_dir, ckpt_keep_last)

  # ---- return final state in case the caller wants to checkpoint --------
  return PPOTrainingState(
      policy_params=ppo_params['policy'],
      value_params=ppo_params['value'],
      ppo_optimizer_state=ppo_opt_state,
      q_params=q_params,
      q_optimizer_state=q_opt_state,
      key=key,
  )
```

## File: ppo_contrastive.py
```python
"""Standalone PPO-on-φ·ψ entry point.

Run with:
  python ppo_contrastive.py \
      --env=point_FourRooms \
      --seed=0 \
      --num_steps=8000000 \
      --log_dir_path=logs/ppo/

Sawyer bin (fixed goal, vanilla CRL without task-goal extra negatives):
  python ppo_contrastive.py \
      --env=sawyer_bin \
      --seed=0 \
      --num_steps=8000000 \
      --log_dir_path=logs/ppo/

PPO is on-policy and single-process; this script does NOT go through
Launchpad.  The SAC-based kappa actor stays untouched and is still
launched via lp_contrastive.py --alg=kappa_sac.
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports
import functools
import json
import os

from absl import app
from absl import flags
import numpy as np

import contrastive
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
import env_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/ppo/', 'Where to log metrics')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('env', 'point_FourRooms', 'Environment type')
flags.DEFINE_integer('num_steps', 8_000_000, 'Total env steps', lower_bound=0)
flags.DEFINE_bool('sample_goals', False,
                  'Sample goal uniformly (else use the fixed goal dict)')
flags.DEFINE_bool(
    'repr_norm', False,
    'If True, L2-normalize critic φ and ψ before dot products.')
# Optional explicit overrides for the two per-env-defaulted knobs.
# If left at -1 (the default), the per-env lookup in PPO_ENV_DEFAULTS wins;
# any non-negative value supplied on the CLI overrides that table.  This lets
# sweeps pin T / crl_steps without editing the source.
flags.DEFINE_integer('ppo_rollout_length', -1,
                     'If >=0, overrides the per-env rollout length default.')
flags.DEFINE_integer('ppo_crl_steps_per_iter', -1,
                     'If >=0, overrides the per-env CRL-steps default.')
flags.DEFINE_integer('ppo_num_envs', -1,
                     'If >=0, overrides the number of parallel env rollouts.')
flags.DEFINE_float(
    'discount', -1.0,
    'If >=0, overrides ContrastiveConfig.discount (CRL discount).')
flags.DEFINE_float(
    'ppo_discount', -1.0,
    'If >0, overrides PPO discount for GAE/returns only; '
    '<=0 falls back to config.discount.')
flags.DEFINE_float(
    'ppo_clip_coef', -1.0,
    'If >0, overrides PPO clip coefficient.')
flags.DEFINE_float(
    'ppo_actor_min_std', -1.0,
    'If >0, overrides PPO actor min std.')
flags.DEFINE_float(
    'ppo_ent_coef', -1.0,
    'If >=0, overrides PPO entropy bonus coefficient; '
    '<0 keeps ContrastiveConfig default.')
flags.DEFINE_bool(
    'ppo_anneal_lr', True,
    'If True, linearly decay PPO Adam learning rate to 0 over training; '
    'if False, use a fixed learning_rate.  Pass --noppo_anneal_lr to disable.')
flags.DEFINE_bool(
    'uniform_sampling', False,
    'If True, mix 50% uniformly sampled goals into each CRL replay batch. '
    'Half the in-batch InfoNCE negatives come from the uniform goal '
    'distribution, half from the replay future-state distribution.')
flags.DEFINE_string(
    'ppo_reward_mode', '',
    "PPO rollout reward baseline. '' = φ(s,a)·ψ(g); "
    "'dirac_target' = log(eps)−φ(s0,a)·ψ(s) off-goal, −φ(s0,a)·ψ(g) at goal; "
    "'kde_dirac' = same formula but densities estimated via Gaussian KDE on "
    "the replay buffer instead of CRL dot products.")
flags.DEFINE_string(
    'ppo_repr_mode', 'crl',
    "Density estimator for the PPO shaped reward. "
    "'crl' (default) = contrastive φ(s,a)·ψ(g) representations; "
    "'gaussian' = diagonal Gaussian p_θ(g|s), reward = log p_θ(g|s_t); "
    "'nf' = conditional RealNVP log p_NF(g|s,a), reward = log p_NF.")
flags.DEFINE_integer(
    'nf_rep_size', 64,
    'NF mode: SA encoder output dim (conditioning vector size).')
flags.DEFINE_integer(
    'nf_num_blocks', 8,
    'NF mode: number of affine coupling blocks in RealNVP.')
flags.DEFINE_integer(
    'nf_coupling_width', 256,
    'NF mode: width of s/t sub-networks inside each coupling block.')
flags.DEFINE_float(
    'nf_encoder_lr', 3e-4,
    'NF mode: Adam learning rate for SA encoder (ref actor_lr).')
flags.DEFINE_float(
    'nf_critic_lr', 1e-4,
    'NF mode: AdamW learning rate for RealNVP flow (ref critic_lr).')
flags.DEFINE_float(
    'nf_critic_weight_decay', 1e-6,
    'NF mode: AdamW weight decay for RealNVP flow (ref critic_weight_decay).')
flags.DEFINE_float(
    'nf_grad_clip', 1.0,
    'NF mode: global-norm gradient clipping for both SA encoder and flow (0 = disabled).')
flags.DEFINE_float(
    'nf_noise_std', 0.05,
    'NF mode: std of Gaussian noise added to goals during training (0 = disabled).')
flags.DEFINE_float(
    'nf_goal_std_min', 0.02,
    'NF mode: minimum per-dim goal std from replay stats (prevents div-by-tiny-std on static dims).')
flags.DEFINE_string(
    'nf_no_norm_goal_dims', '',
    'Deprecated; ignored. NF uses running replay mean/std with nf_goal_std_min floor.')
flags.DEFINE_boolean(
    'ppo_skip_first_eval', False,
    'Skip logging the iteration-0 eval (avoids logging the checkpoint result as the first data point when resuming).')
flags.DEFINE_integer(
    'ppo_eval_interval', -1,
    'Run eval every N PPO iterations. <0 keeps config/env default (10). 0 disables eval.')
flags.DEFINE_integer(
    'ppo_eval_episodes', -1,
    'Number of eval episodes per eval round. <0 keeps config default (5).')
flags.DEFINE_boolean(
    'ppo_norm_reward', True,
    'Normalize the repr reward by the running std of discounted returns. Set False to pass raw reward directly to PPO.')
flags.DEFINE_boolean(
    'nf_mix_env_goal_stats', False,
    'NF mode: when computing normalisation stats, also include the actual env goals '
    '(obs[obs_dim:] from current rollout) so the normaliser covers reward goals too.')
flags.DEFINE_integer(
    'nf_goal_enc_size', 0,
    'Goal encoder output dim for NF mode. 0 = disabled (raw normalized goal fed to flow). '
    '>0 adds a compact 2×(Dense256+LN+swish)→Dense(N) encoder trained end-to-end with the flow.')
flags.DEFINE_integer(
    'ppo_return_norm_window', 0,
    'Soft sliding-window size for the return normalizer. 0 = infinite (standard Welford). '
    '>0 caps the effective sample count so the variance stays responsive to recent reward shifts.')
flags.DEFINE_float(
    'ppo_dirac_eps', 1e-6,
    'Epsilon in dirac_target / kde_dirac reward: log(eps) − log_p(s).')
flags.DEFINE_integer(
    'kde_max_points', 2000,
    'Number of replay-buffer states to fit the Gaussian KDE on (kde_dirac mode).')
flags.DEFINE_integer(
    'kde_refit_interval', 1,
    'Refit the KDE every N PPO iterations (kde_dirac mode). 1 = every iteration.')
flags.DEFINE_float(
    'kde_bandwidth', 0.0,
    'KDE bandwidth (kde_dirac mode). 0.0 = Scott\'s rule automatically.')
flags.DEFINE_integer(
    'max_replay_size', -1,
    'Max transitions in the PPO episode replay buffer. <0 keeps default (1e6).')
flags.DEFINE_integer(
    'ppo_min_replay_size', -1,
    'Min replay transitions before density/CRL updates start. <0 keeps default (1e4).')
flags.DEFINE_integer(
    'ppo_checkpoint_interval', -1,
    'Save checkpoints every N PPO iterations. <0 keeps config default (500).')
flags.DEFINE_integer(
    'ppo_checkpoint_keep_last', -1,
    'Max milestone ckpt_iter_*.pkl files to retain (FIFO). '
    '0 = keep all. <0 keeps config default (0 = keep all).')
flags.DEFINE_string(
    'ppo_crl_loss_direction', 'forward',
    "InfoNCE loss direction for PPO-CRL: 'forward' (fix anchor, vary goal) "
    "or 'backward' (fix goal, vary anchor = transpose logits).")
flags.DEFINE_float(
    'ppo_crl_repr_tau', -1.0,
    'CRL mode: EMA decay τ for φ, ψ used in PPO reward r=φ·ψ. '
    'ema ← τ·ema + (1−τ)·online after each CRL step. '
    '0 = use online params (default). Higher τ = slower reward tracking. '
    '<0 keeps config default.')
flags.DEFINE_boolean(
    'bin_randomize_gripper_init', False,
    'SawyerBin: randomize initial gripper TCP offset around the object at reset.')
flags.DEFINE_boolean(
    'builderbench_use_pd', False,
    'BuilderBench: wrap env in PDWrapper (short horizon). Default False = raw control.')
flags.DEFINE_integer(
    'builderbench_pd_duration', 5,
    'BuilderBench PDWrapper: low-level MuJoCo steps per RL step when use_pd=True.')
flags.DEFINE_string(
    'hidden_layer_sizes', '',
    'Comma-separated hidden layer widths, e.g. "256,256,256,256,256,256". '
    'Empty string keeps the ContrastiveConfig default. '
    'Stacks with >2 layers automatically use ResidualMLP (LayerNorm + Swish, '
    'skip every 2 layers).')

flags.DEFINE_string('exp_name', 'ppo_contrastive.py', 'Experiment name for logging')
flags.DEFINE_string('obs_space', 'xy,select', 'Comma-separated obs components')

flags.DEFINE_boolean(
      'ppo_cleanrl_actor', False,
      'If True, uses Tanh activations, Orthogonal init, and state-independent std for the Actor (Trick 2).')


flags.DEFINE_float(
    'ppo_warmup_percent', 0.0,
    'Percentage of total iterations to wait before starting PPO actor updates '
    '(CRL trains during this time).')

flags.DEFINE_boolean(
    'staggered_resets', False,
    'If True, randomly staggers the parallel environments before training begins to maximize batch diversity.')
flags.DEFINE_boolean(
    'crl_on_policy', False,
    'If True, disables the replay buffer and trains CRL strictly on the current (T, E) rollout tensor.')
flags.DEFINE_float(
      'builderbench_episode_length_multiplier', 1.0,
      'BuilderBench: scale factor on the base episode length '
        '(100 + num_cubes*50 raw MuJoCo steps), applied before optional '
        'PD-duration division. 1.0 = default; 2.0 = double, etc. Result must '
        'be divisible by --builderbench_pd_duration when '
        '--builderbench_use_pd=true.')
# ---------------------------------------------------------------------------
# Fixed-goal lookup reused from lp_contrastive.py.
# ---------------------------------------------------------------------------
fixed_goal_dict = {
    'point_Spiral7x7':   [np.array([3, 3], dtype=float),
                          np.array([6, 6], dtype=float)],
    'point_Spiral9x9':   [np.array([5, 5], dtype=float),
                          np.array([8, 8], dtype=float)],
    'point_Spiral11x11': [np.array([5, 5], dtype=float),
                          np.array([10, 10], dtype=float)],
    'point_FourRooms':   [np.array([0, 0], dtype=float),
                          np.array([2, 8],  dtype=float)],
    # point_EightRooms: start top-left corner, goal bottom-right corner.
    # Must traverse all 8 rooms (11×21 grid, 100-step episodes).
    'point_EightRooms':  [np.array([0, 0],   dtype=float),
                          np.array([10, 20], dtype=float)],
    # point_SixteenRooms: start top-left corner, goal bottom-right corner.
    # Must traverse all 16 rooms (21×21 grid, 200-step episodes).
    'point_SixteenRooms': [np.array([0, 0],   dtype=float),
                           np.array([20, 20], dtype=float)],
    # point_SixteenRooms4D: same maze + 2 free extra dims (all start/goal at 0).
    'point_SixteenRooms4D': [np.array([0, 0, 0],    dtype=float),
                              np.array([20, 20, 0], dtype=float)],
    # point_SixteenRoomsActual4D: same maze + 2 free extra dims.
    'point_SixteenRoomsActual4D': [np.array([0, 0, 0, 0],    dtype=float),
                                   np.array([20, 20, 0, 0], dtype=float)],
    # point_Impossible: start top-left (0,0), goal row 6 col 8 (reachable via
    # the long winding path through the maze).
    'point_Impossible': [np.array([0, 0], dtype=float),
                         np.array([6, 8], dtype=float)],
    # point_Maze11x11: start top-left (0,0), goal top-right (0,10).
    'point_Maze11x11':  [np.array([0, 0], dtype=float),
                         np.array([0, 10], dtype=float)],
    # point_Wall11x11: start top-left (0,0), goal bottom-right (10,10);
    # must go along the top corridor, down the right gap, then along the bottom.
    'point_Wall11x11':  [np.array([0, 0], dtype=float),
                         np.array([10, 10], dtype=float)],
    'sawyer_bin':   np.array([0.12, 0.7, 0.02]),
    'sawyer_box':   np.array([0.0, 0.75, 0.133]),
    'sawyer_peg':   np.array([-0.3, 0.6, 0.0]),
    # Reach: fixed goal = centre of the goal cube (x=0, y=0.85, z=0.2).
    'sawyer_reach': np.array([0.0, 0.85, 0.2]),
    # Push: fixed goal = puck target (z≈0.02); ψ hand = target − 8 cm y, +3 cm z.
    'sawyer_push':  np.array([0.0, 0.85, 0.02]),
    # Drawer-open: fixed goal = open handle (_target_pos); ψ also places
    # ideal_hand at closed-handle first-contact (+0.2 y, −0.02 z).
    'sawyer_drawer_open':  np.array([0.0, 0.54, 0.09]),
    # Button-press: fixed goal = button depressed to the hole site (y≈0.78).
    'sawyer_button_press': np.array([0, 0.8, 0.115]),
    # One-hot goal over river cells (length must match RIVERSWIM_LEN, default 6).
    'riverswim': np.array([0., 0., 0., 0., 0., 1.], dtype=float),
    # Flow figureeight variants: goal = all N vehicles at target_velocity
    # (20 m/s), normalized by the network max_speed (30 m/s).
    'flow_figureeight':       np.full(14, 20.0 / 30.0, dtype=float),
    'flow_figureeight_7rl':   np.full(14, 20.0 / 30.0, dtype=float),
    'flow_figureeight_14rl':  np.full(14, 20.0 / 30.0, dtype=float),
    'flow_figureeight_4v2rl': np.full(4,  20.0 / 30.0, dtype=float),
    'flow_figureeight_8v4rl': np.full(8,  20.0 / 30.0, dtype=float),
    'flow_figureeight_1v1rl': np.full(1,  20.0 / 30.0, dtype=float),
    'flow_figureeight_2v1rl': np.full(2,  20.0 / 30.0, dtype=float),
    'flow_figureeight_2v2rl': np.full(2,  20.0 / 30.0, dtype=float),
}

# ---------------------------------------------------------------------------
# Per-env PPO defaults.
#
# The single knob that really needs to scale with the environment is the
# rollout length T, because it interacts with episode length: if T < ep_len,
# most rollouts complete zero episodes and GAE must bootstrap the return off
# V(s_T), which hurts sample efficiency early in training.  The CRL step
# count scales proportionally so that the CRL-to-env-step ratio stays
# roughly constant (CRL-steps ≈ T / 2).
#
# point_FourRooms / point_Spiral11x11: 50/100-step episodes -> T=128 keeps
#   ~1-2 completed episodes per env per rollout.  These are the values the
#   tuned point-env runs use, so we keep them to avoid regressing.
# sawyer_{bin,box,peg}: 150-step episodes -> T=256 gives one full episode
#   per env per rollout, matching CleanRL's MuJoCo convention.
# ---------------------------------------------------------------------------
PPO_ENV_DEFAULTS = {
    'point_FourRooms':   dict(rollout_length=128, crl_steps_per_iter=64),
    # EightRooms: 100-step episodes (2× FourRooms) → T=256 keeps ~2 episodes/rollout.
    'point_EightRooms':   dict(rollout_length=256, crl_steps_per_iter=128),
    # SixteenRooms: 200-step episodes (2× EightRooms) → T=512 keeps ~2 episodes/rollout.
    'point_SixteenRooms': dict(rollout_length=512, crl_steps_per_iter=256),
    # SixteenRooms4D / SixteenRoomsActual4D: same episode length, same rollout budget.
    'point_SixteenRooms4D':       dict(rollout_length=512, crl_steps_per_iter=256),
    'point_SixteenRoomsActual4D': dict(rollout_length=512, crl_steps_per_iter=256),
    'point_Spiral7x7':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Spiral9x9':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Spiral11x11': dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Maze11x11':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Wall11x11':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Impossible':  dict(rollout_length=128, crl_steps_per_iter=64),
    'riverswim':         dict(rollout_length=128, crl_steps_per_iter=64),
    'sawyer_bin':        dict(rollout_length=256, crl_steps_per_iter=128),
    'sawyer_box':        dict(rollout_length=256, crl_steps_per_iter=128),
    'sawyer_peg':        dict(rollout_length=256, crl_steps_per_iter=128),
    # Reach: very short episodes (150 steps), tiny obs → fast.
    'sawyer_reach':      dict(rollout_length=256, crl_steps_per_iter=128),
    # Push: same budget as bin (same episode length, similar obs structure).
    'sawyer_push':       dict(rollout_length=256, crl_steps_per_iter=128),
    # Drawer-open: same episode length / obs structure as push.
    'sawyer_drawer_open':  dict(rollout_length=256, crl_steps_per_iter=128),
    # Button-press: same episode length / obs structure as push.
    'sawyer_button_press': dict(rollout_length=256, crl_steps_per_iter=128),
    # flow_figureeight: 1500-step episodes. T=1500 gives one complete episode
    # per env per rollout (important for GAE accuracy on a long-horizon env).
    # φ: state = speeds+positions; ψ: goal speeds only (end_index=14 on state).
    'flow_figureeight':  dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=14),
    # 7/14 RL variant: same horizon and indexing, wider action space (7-D).
    'flow_figureeight_7rl':  dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=14),
    # 14/14 RL variant: all vehicles RL-controlled, 14-D action space.
    'flow_figureeight_14rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=14),
    'flow_figureeight_4v2rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=4),
    'flow_figureeight_8v4rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=8),
    # 1-vehicle sanity check: 2-D state, 1-D goal. ψ sees only the 1 goal speed.
    'flow_figureeight_1v1rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=1),
    # 2-vehicle variants: 4-D state (speeds+positions), 2-D goal (target speeds).
    # end_index=2 so ψ only sees the 2 goal speeds.
    'flow_figureeight_2v1rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=2),
    'flow_figureeight_2v2rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=2),
}


def fixed_goal_for_env(env_name: str) -> np.ndarray:
  """Return the fixed goal vector for ``env_name`` (incl. all BuilderBench creative tasks)."""
  if env_name in fixed_goal_dict:
    return fixed_goal_dict[env_name]
  from envs.builderbench_utils import (
      default_fixed_target_goal,
      is_builderbench_creative_env,
      parse_sgcrl_builderbench_env_name,
  )
  if is_builderbench_creative_env(env_name):
    _, num_cubes, task_index = parse_sgcrl_builderbench_env_name(env_name)
    return default_fixed_target_goal(num_cubes, task_index)
  raise KeyError(f'No fixed goal configured for env {env_name!r}')


def ppo_env_defaults_for_env(env_name: str, use_pd: bool = False,
                             episode_length_multiplier: float = 1.0):
  """Return per-env PPO defaults, including dynamic BuilderBench creative tasks."""
  if env_name in PPO_ENV_DEFAULTS:
    return PPO_ENV_DEFAULTS[env_name]
  from envs.builderbench_utils import (
      is_builderbench_creative_env,
      parse_sgcrl_builderbench_env_name,
      ppo_env_defaults as builderbench_ppo_defaults,
  )
  if is_builderbench_creative_env(env_name):
    _, num_cubes, _ = parse_sgcrl_builderbench_env_name(env_name)
    return builderbench_ppo_defaults(
        num_cubes, use_pd=use_pd,
        episode_length_multiplier=episode_length_multiplier)
  return None


def _json_safe(value):
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if hasattr(value, 'tolist'):
    try:
      return value.tolist()
    except Exception:  # pragma: no cover
      pass
  return str(value)


def main(_):
  env_name = FLAGS.env
  seed = FLAGS.seed
  print(f'[ppo_contrastive] env={env_name} seed={seed}')

  # ---- Build config ------------------------------------------------------
  # Only parameters we explicitly want to override are set here; everything
  # else inherits the ContrastiveConfig defaults (including the new PPO_*
  # fields in contrastive/config.py).
  params = dict(
      seed=seed,
      env_name=env_name,
      alg_name='ppo',
      reward_shaping_mode='ppo',
      use_cpc=True,                   # CRL loss: InfoNCE / CPC (matches kappa_sac)
      max_number_of_steps=FLAGS.num_steps,
      log_dir=FLAGS.log_dir_path,
      add_uid=FLAGS.add_uid,
      fix_goals=not FLAGS.sample_goals,
  )
  config = contrastive.ContrastiveConfig(**params)
  config.repr_norm = bool(FLAGS.repr_norm)

  # ---- Per-env PPO defaults (CLI flags still override) -------------------
  _use_pd = (bool(FLAGS.builderbench_use_pd)
             if env_name.startswith('builderbench_') else False)
  _ep_mult = float(FLAGS.builderbench_episode_length_multiplier)
  env_defaults = ppo_env_defaults_for_env(env_name, use_pd=_use_pd, episode_length_multiplier=_ep_mult)
  if env_defaults is None:
    print(f'[ppo_contrastive] WARNING: no PPO_ENV_DEFAULTS entry for '
          f'{env_name!r}; falling back to ContrastiveConfig defaults '
          f'(T={config.ppo_rollout_length}, '
          f'crl_steps={config.ppo_crl_steps_per_iter}).')
  else:
    config.ppo_rollout_length = int(env_defaults['rollout_length'])
    config.ppo_crl_steps_per_iter = int(env_defaults['crl_steps_per_iter'])
    if 'start_index' in env_defaults:
      config.start_index = int(env_defaults['start_index'])
    if 'end_index' in env_defaults:
      config.end_index = int(env_defaults['end_index'])
    if 'num_envs' in env_defaults and FLAGS.ppo_num_envs < 0:
      config.ppo_num_envs = int(env_defaults['num_envs'])
    if ('eval_interval' in env_defaults
        and FLAGS.ppo_eval_interval < 0):
      config.ppo_eval_interval = int(env_defaults['eval_interval'])
    if ('checkpoint_interval' in env_defaults
        and FLAGS.ppo_checkpoint_interval < 0):
      config.ppo_checkpoint_interval = int(env_defaults['checkpoint_interval'])

  if FLAGS.ppo_rollout_length >= 0:
    config.ppo_rollout_length = int(FLAGS.ppo_rollout_length)
  if FLAGS.ppo_crl_steps_per_iter >= 0:
    config.ppo_crl_steps_per_iter = int(FLAGS.ppo_crl_steps_per_iter)
  if FLAGS.ppo_num_envs >= 0:
    config.ppo_num_envs = int(FLAGS.ppo_num_envs)

  total_steps = int(FLAGS.num_steps)
  if env_name.startswith('builderbench_'):
    from envs.builderbench_utils import (
        BUILDERBENCH_NUM_STEPS,
        builderbench_replay_size,
    )
    if FLAGS.max_replay_size < 0:
      config.max_replay_size = builderbench_replay_size(config.ppo_num_envs)
    if FLAGS.num_steps == 8_000_000:
      total_steps = BUILDERBENCH_NUM_STEPS
      config.max_number_of_steps = total_steps
    print(f'[ppo] builderbench scale: num_steps={total_steps} '
          f'max_replay_size={config.max_replay_size} '
          f'(E={config.ppo_num_envs})')

  if FLAGS.discount >= 0.0:
    config.discount = float(FLAGS.discount)
  if FLAGS.ppo_discount > 0.0:
    config.ppo_discount = float(FLAGS.ppo_discount)
  if FLAGS.ppo_clip_coef > 0.0:
    config.ppo_clip_coef = float(FLAGS.ppo_clip_coef)
  if FLAGS.ppo_actor_min_std > 0.0:
    config.ppo_actor_min_std = float(FLAGS.ppo_actor_min_std)
  if FLAGS.ppo_ent_coef >= 0.0:
    config.ppo_ent_coef = float(FLAGS.ppo_ent_coef)
  config.ppo_anneal_lr = bool(FLAGS.ppo_anneal_lr)
  config.uniform_sampling = bool(FLAGS.uniform_sampling)
  config.staggered_resets = bool(FLAGS.staggered_resets)
  config.crl_on_policy = bool(FLAGS.crl_on_policy)
  config.ppo_reward_mode = str(FLAGS.ppo_reward_mode).strip()
  config.ppo_repr_mode = str(FLAGS.ppo_repr_mode).strip()
  config.ppo_dirac_eps = float(FLAGS.ppo_dirac_eps)
  config.nf_rep_size = int(FLAGS.nf_rep_size)
  config.nf_num_blocks = int(FLAGS.nf_num_blocks)
  config.nf_coupling_width = int(FLAGS.nf_coupling_width)
  config.nf_encoder_lr = float(FLAGS.nf_encoder_lr)
  config.nf_critic_lr = float(FLAGS.nf_critic_lr)
  config.nf_critic_weight_decay = float(FLAGS.nf_critic_weight_decay)
  config.nf_grad_clip = float(FLAGS.nf_grad_clip)
  config.nf_noise_std = float(FLAGS.nf_noise_std)
  config.nf_goal_std_min = float(FLAGS.nf_goal_std_min)
  config.nf_mix_env_goal_stats = bool(FLAGS.nf_mix_env_goal_stats)
  config.ppo_skip_first_eval = bool(FLAGS.ppo_skip_first_eval)
  if FLAGS.ppo_eval_interval >= 0:
    config.ppo_eval_interval = int(FLAGS.ppo_eval_interval)
  if FLAGS.ppo_eval_episodes >= 0:
    config.ppo_eval_episodes = int(FLAGS.ppo_eval_episodes)
  config.ppo_norm_reward = bool(FLAGS.ppo_norm_reward)
  config.nf_goal_enc_size = int(FLAGS.nf_goal_enc_size)
  config.ppo_return_norm_window = int(FLAGS.ppo_return_norm_window)
  config.ppo_warmup_percent = float(FLAGS.ppo_warmup_percent)
  config.kde_max_points = int(FLAGS.kde_max_points)
  config.kde_refit_interval = int(FLAGS.kde_refit_interval)
  config.kde_bandwidth = float(FLAGS.kde_bandwidth)
  config.builderbench_episode_length_multiplier = float(FLAGS.builderbench_episode_length_multiplier)
  if FLAGS.max_replay_size >= 0:
    config.max_replay_size = int(FLAGS.max_replay_size)
  if FLAGS.ppo_min_replay_size >= 0:
    config.ppo_min_replay_size = int(FLAGS.ppo_min_replay_size)
  if FLAGS.ppo_checkpoint_interval >= 0:
    config.ppo_checkpoint_interval = int(FLAGS.ppo_checkpoint_interval)
  if FLAGS.ppo_checkpoint_keep_last >= 0:
    config.ppo_checkpoint_keep_last = int(FLAGS.ppo_checkpoint_keep_last)
  if FLAGS.ppo_crl_loss_direction.strip():
    config.ppo_crl_loss_direction = FLAGS.ppo_crl_loss_direction.strip().lower()
  if FLAGS.ppo_crl_repr_tau >= 0.0:
    config.ppo_crl_repr_tau = float(FLAGS.ppo_crl_repr_tau)
  if FLAGS.hidden_layer_sizes.strip():
    config.hidden_layer_sizes = tuple(
        int(x) for x in FLAGS.hidden_layer_sizes.split(',') if x.strip())

  print(f'[ppo_contrastive] PPO knobs: '
        f'rollout_length={config.ppo_rollout_length}, '
        f'crl_steps_per_iter={config.ppo_crl_steps_per_iter}, '
        f'num_envs={config.ppo_num_envs}, '
        f'clip_coef={config.ppo_clip_coef}, '
        f'actor_min_std={config.ppo_actor_min_std}, '
        f'ent_coef={config.ppo_ent_coef}, '
        f'discount_crl={config.discount}, '
        f'discount_ppo={config.ppo_discount if config.ppo_discount > 0 else config.discount}, '
        f'norm_reward={config.ppo_norm_reward}, '
        f'repr_norm={config.repr_norm}, '
        f'ppo_anneal_lr={config.ppo_anneal_lr}  '
        f'ppo_repr_mode={config.ppo_repr_mode!r}  '
        f'ppo_reward_mode={config.ppo_reward_mode!r}  '
        f'ppo_crl_repr_tau={config.ppo_crl_repr_tau}  '
        f'ppo_dirac_eps={config.ppo_dirac_eps}  '
        f'max_replay_size={config.max_replay_size}  '
        f'ppo_min_replay_size={config.ppo_min_replay_size}  '
        f'kde_max_points={config.kde_max_points}  '
        f'kde_refit_interval={config.kde_refit_interval}  '
        f'kde_bandwidth={config.kde_bandwidth}  '
        f'ckpt_interval={config.ppo_checkpoint_interval}  '
        f'ckpt_keep_last={config.ppo_checkpoint_keep_last} '
        f'({"all milestones" if config.ppo_checkpoint_keep_last <= 0 else "FIFO prune"})  '
        f'hidden_layers={config.hidden_layer_sizes}')

  # ---- Build env factories ----------------------------------------------
  fixed_start_end = (fixed_goal_for_env(env_name)
                     if config.fix_goals else None)

  # NF push: start episodes with gripper closed (does not affect CRL/Gaussian).
  _env_kwargs = {}
  if (str(config.ppo_repr_mode).strip().lower() == 'nf'
      and env_name == 'sawyer_push'):
    _env_kwargs['nf_closed_gripper_init'] = True
    print('[ppo] sawyer_push NF init: closed gripper at reset')
  if env_name == 'sawyer_bin' and FLAGS.bin_randomize_gripper_init:
    _env_kwargs['randomize_gripper_init'] = True
    print('[ppo] sawyer_bin init: randomized gripper position at reset')
  if env_name.startswith('builderbench_'):
    _env_kwargs['builderbench_use_pd'] = bool(FLAGS.builderbench_use_pd)
    _env_kwargs['builderbench_pd_duration'] = int(FLAGS.builderbench_pd_duration)
    _env_kwargs['builderbench_episode_length_multiplier'] = _ep_mult
    if FLAGS.builderbench_use_pd and FLAGS.ppo_rollout_length < 0:
      pd_defaults = ppo_env_defaults_for_env(
          env_name, use_pd=True, episode_length_multiplier=_ep_mult)
      if pd_defaults is not None:
        config.ppo_rollout_length = int(pd_defaults['rollout_length'])
        config.ppo_crl_steps_per_iter = int(pd_defaults['crl_steps_per_iter'])
    print(f'[ppo] builderbench: use_pd={FLAGS.builderbench_use_pd} '
          f'pd_duration={FLAGS.builderbench_pd_duration} '
          f'episode_length_multiplier={_ep_mult}'
          + (' pd_policy_obs=pos+select' if FLAGS.builderbench_use_pd else ''))


  _env_kwargs['obs_space_list'] = [s.strip() for s in FLAGS.obs_space.split(',')]

  def env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=fixed_start_end, **_env_kwargs)
    return env

  def eval_env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=fixed_goal_for_env(env_name), **_env_kwargs)
    return env

  # obs_dim / max_episode_steps inferred from one sample env.
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index, seed,
      fixed_start_end=fixed_start_end, **_env_kwargs)
  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(probe_env, '_step_limit') + 1
  del probe_env

  # ---- Network factory (adds value_network via networks.py changes) -----
  # NOTE: `actor_min_std` is raised from the shared default (1e-6) to the
  # PPO-specific floor (0.1 by default).  Without this, the tanh-squashed
  # Gaussian policy collapses to a near-point-mass within a handful of
  # PPO updates — SAC gets away with a 1e-6 floor because adaptive-α
  # actively regulates entropy; PPO has no such control loop and relies
  # on (a) an entropy bonus and (b) a hard std floor to stay exploratory.
  network_factory = functools.partial(
      contrastive.make_networks,
      obs_dim=obs_dim,
      repr_dim=config.repr_dim,
      repr_norm=config.repr_norm,
      twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
      actor_min_std=float(config.ppo_actor_min_std),
      ppo_cleanrl_actor=bool(FLAGS.ppo_cleanrl_actor)
  )

  # ---- Logger ------------------------------------------------------------
  run_dir = os.path.join(
      config.log_dir,
      f'{config.alg_name}_{config.env_name}_{seed}')
  os.makedirs(run_dir, exist_ok=True)
  run_config_path = os.path.join(run_dir, 'run_config.json')
  run_cfg_payload = {
      'entrypoint': 'ppo_contrastive.py',
      'env': env_name,
      'seed': int(seed),
      'flags': {k: _json_safe(v) for k, v in FLAGS.flag_values_dict().items()},
      'resolved_config': {
          k: _json_safe(v) for k, v in config.__dict__.items()
      },
      'fixed_start_end': _json_safe(fixed_start_end),
      'ppo_env_defaults': _json_safe(env_defaults),
  }
  with open(run_config_path, 'w', encoding='utf-8') as fh:
    json.dump(run_cfg_payload, fh, indent=2, sort_keys=True)
  print(f'[ppo_contrastive] wrote run config: {run_config_path}')
  from default import make_default_logger
  logger_fn = functools.partial(
      make_default_logger,
      save_dir=run_dir,
      add_uid=config.add_uid,
      steps_key='learner_steps')

  # ---- Go ----------------------------------------------------------------
  checkpoint_dir = os.path.join(run_dir, 'checkpoints')
  _bb_kwargs = None
  if env_name.startswith('builderbench_'):
    _bb_kwargs = dict(_env_kwargs)
    if fixed_start_end is not None:
      _bb_kwargs['fixed_target_goal'] = np.asarray(
          fixed_start_end, dtype=np.float32)

  ppo_learner.run_ppo_training(
      config=config,
      env_factory=env_factory,
      eval_env_factory=eval_env_factory,
      network_factory=network_factory,
      logger_fn=logger_fn,
      total_steps=total_steps,
      seed=seed,
      checkpoint_dir=checkpoint_dir,
      builderbench_kwargs=_bb_kwargs,
  )


if __name__ == '__main__':
  app.run(main)
```
