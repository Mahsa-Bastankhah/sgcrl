#!/usr/bin/env python3
"""Evaluate BuilderBench MPO-CRL checkpoints and plot eval success vs env steps.

Loads ``state.target_policy_params`` from MPO pickle checkpoints and runs
deterministic batched eval (``dist.mode()``) via ``JaxBuilderBenchVecEnv``,
matching training-time BuilderBench eval.

CSV / plot format matches ``ppo_builderbench_checkpoint_eval.py`` so plots
can be compared side-by-side.

Examples:
  python scripts/mpo_builderbench_checkpoint_eval.py \\
      --run_dir=logs/mpo_crl_builderbench_creative2_task1_gpu_validation/\\
mpo_crl_builderbench_creative_2_task1_0

  python scripts/mpo_builderbench_checkpoint_eval.py \\
      --run_dir=... --incremental \\
      --csv_output=figs/builderbench/checkpoint_eval/mpo_c2t1_seed0_checkpoint_success.csv \\
      --output=figs/builderbench/mpo_c2t1_checkpoint_success.png
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'baseline-agents'))
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import numpy as np

import mpo_crl_learner
from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv
from envs.builderbench_utils import is_builderbench_creative_env

# Reuse CSV / plot helpers from the PPO eval script for identical figure style.
_ppo_eval_path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_checkpoint_eval.py')
_ppo_spec = importlib.util.spec_from_file_location('bb_ckpt_eval_ppo', _ppo_eval_path)
_ppo_eval = importlib.util.module_from_spec(_ppo_spec)
sys.modules['bb_ckpt_eval_ppo'] = _ppo_eval
_ppo_spec.loader.exec_module(_ppo_eval)

CheckpointEvalResult = _ppo_eval.CheckpointEvalResult
CSV_FIELDS = _ppo_eval.CSV_FIELDS
episode_successes_from_steps = _ppo_eval.episode_successes_from_steps
list_checkpoint_files = _ppo_eval.list_checkpoint_files
merge_results = _ppo_eval.merge_results
plot_multi_seed_results = _ppo_eval.plot_multi_seed_results
plot_results = _ppo_eval.plot_results
read_csv_results = _ppo_eval.read_csv_results
seed_csv_path = _ppo_eval.seed_csv_path
write_csv_results = _ppo_eval.write_csv_results
_iteration_from_label = _ppo_eval._iteration_from_label

# Reuse MPO video helpers for train context + policy network construction.
_mpo_vid_path = os.path.join(_REPO, 'scripts', 'mpo_builderbench_rollout_video.py')
_mpo_vid_spec = importlib.util.spec_from_file_location('mpo_bb_video', _mpo_vid_path)
_mpo_vid = importlib.util.module_from_spec(_mpo_vid_spec)
sys.modules['mpo_bb_video'] = _mpo_vid
_mpo_vid_spec.loader.exec_module(_mpo_vid)

_load_train_ctx = _mpo_vid._load_train_ctx
_build_policy_network = _mpo_vid._build_policy_network
_TrainCtx = _mpo_vid._TrainCtx


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


def _load_mpo_checkpoint(path: str) -> Dict[str, Any]:
  with open(path, 'rb') as handle:
    payload = pickle.load(handle)
  state = payload['state']
  return {
      'policy_params': state.target_policy_params,
      'iteration': int(payload.get('iteration', -1)),
      'global_step': int(payload.get('global_step', -1)),
  }


class MPOCheckpointEvalSession:
  """Reusable MPO eval session (policy net + vec env + jitted unroll)."""

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

    self.policy_network = _build_policy_network(
        env_name, seed=network_seed, ctx=ctx)
    self.vec_env = JaxBuilderBenchVecEnv(
        env_name=env_name,
        num_envs=self.num_eval_episodes,
        seed=int(eval_seed),
        use_pd=ctx.use_pd,
        pd_duration=ctx.pd_duration,
        pd_filter_policy_obs=ctx.filter_policy_obs,
        fixed_target_goal=fixed_goal,
        permute_start_boxes=bool(getattr(ctx, 'permute_start_boxes', True)),
    )

    policy_network = self.policy_network

    @jax.jit
    def eval_policy_action(policy_p, obs):
      action = policy_network.apply(policy_p, obs).mode()
      return jnp.clip(action, -1.0, 1.0)

    self._eval_unroll = self.vec_env.compile_eval_unroll(
        eval_policy_action,
        unroll_length=self.vec_env.episode_length,
    )

  def eval_policy_params(
      self, policy_params: Any,
  ) -> Tuple[float, float, Tuple[float, ...]]:
    eval_state = self.vec_env.reset_state()
    steps = self._eval_unroll(eval_state, policy_params)
    ep_success = episode_successes_from_steps(steps)
    mean = float(ep_success.mean())
    n = int(ep_success.size)
    std = float(math.sqrt(mean * (1.0 - mean) / max(n, 1)))
    return mean, std, tuple(float(x) for x in ep_success)

  def eval_checkpoint_file(self, label: str, path: str) -> CheckpointEvalResult:
    ckpt = _load_mpo_checkpoint(path)
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
    title = (
        f'MPO BuilderBench eval success — {env_name} '
        f'({os.path.basename(run_dir)})')
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
  print(f'[mpo_bb_eval] env={env_name}  pending={len(ckpt_files)}  '
        f'existing={len(existing)}  eval_episodes={num_eval_episodes}')
  print(f'[mpo_bb_eval] use_pd={ctx.use_pd}  filter_policy_obs={ctx.filter_policy_obs}  '
        f'obs_dim={ctx.obs_dim}  ep_len={ctx.episode_length}')

  new_results: List[CheckpointEvalResult] = []
  if ckpt_files:
    session = MPOCheckpointEvalSession(
        env_name, ctx, run_cfg,
        num_eval_episodes=num_eval_episodes,
        eval_seed=eval_seed,
        network_seed=network_seed,
    )
    for i, (label, path, _mtime) in enumerate(ckpt_files):
      print(f'[mpo_bb_eval] ({i + 1}/{len(ckpt_files)}) {label}  ({path})')
      result = session.eval_checkpoint_file(label, path)
      print(f'[mpo_bb_eval]   iter={result.iteration}  step={result.global_step}  '
            f'success={result.success_mean:.3f} ± {result.success_std:.3f}')
      new_results.append(result)

  results = merge_results(existing, new_results)
  if results:
    write_csv_results(csv_path, results)
    print(f'[mpo_bb_eval] wrote csv: {csv_path}  ({len(results)} row(s))')
  title = (
      f'MPO BuilderBench eval success — {env_name} '
      f'({os.path.basename(run_dir)})')
  plot_results(results, title=title, output_path=plot_path, x_axis=x_axis)
  if results:
    print(f'[mpo_bb_eval] wrote plot: {plot_path}')
  return results, csv_path, plot_path


def discover_run_dirs(log_dir: str, env: str, seeds: Optional[set[int]] = None):
  """Yield (seed, run_dir, ckpt_dir) under logs/.../mpo_crl_<env>_<seed>/."""
  base = os.path.abspath(log_dir)
  if not os.path.isdir(base):
    return
  pattern = os.path.join(base, f'mpo_crl_{env}_*')
  for run_dir in sorted(glob.glob(pattern)):
    if not os.path.isdir(run_dir):
      continue
    run_name = os.path.basename(run_dir)
    m = re.fullmatch(rf'mpo_crl_{re.escape(env)}_(\d+)', run_name)
    if not m:
      continue
    seed = int(m.group(1))
    if seeds is not None and seed not in seeds:
      continue
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    yield seed, run_dir, ckpt_dir


def main():
  parser = argparse.ArgumentParser(
      description='Deterministic MPO BuilderBench checkpoint eval + success plot')
  parser.add_argument('--run_dir',
                      help='Training run dir (run_config.json + checkpoints/).')
  parser.add_argument('--checkpoint_dir',
                      help='Checkpoint dir (parent must contain run_config.json).')
  parser.add_argument('--env', default=None)
  parser.add_argument('--num_eval_episodes', type=int, default=20)
  parser.add_argument('--eval_seed', type=int, default=0)
  parser.add_argument('--seed', type=int, default=0,
                      help='Network-init probe seed.')
  parser.add_argument('--output', default=None, help='Output PNG path.')
  parser.add_argument('--csv_output', default=None, help='Output CSV path.')
  parser.add_argument('--plot_tag', default=None, help='Plot filename stem override.')
  parser.add_argument('--x_axis', choices=('global_step', 'iteration'),
                      default='global_step')
  parser.add_argument('--include_latest', action='store_true')
  parser.add_argument('--incremental', action='store_true')
  parser.add_argument('--plot_only', action='store_true')
  parser.add_argument('--max_checkpoints', type=int, default=-1)
  parser.add_argument('--only_labels', default=None,
                      help='Comma-separated labels to eval (e.g. iter_0000150).')
  args = parser.parse_args()

  if args.run_dir:
    run_dir = os.path.abspath(args.run_dir)
  elif args.checkpoint_dir:
    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    run_dir = os.path.dirname(ckpt_dir)
  else:
    parser.error('Provide --run_dir or --checkpoint_dir')

  run_cfg = read_run_config(run_dir)
  env_name = env_from_run_config(run_cfg, args.env)
  if not is_builderbench_creative_env(env_name):
    parser.error(f'env must be builderbench_creative_* (got {env_name!r})')

  only_labels = None
  if args.only_labels:
    only_labels = [x.strip() for x in args.only_labels.split(',') if x.strip()]

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
      only_labels=only_labels,
  )


if __name__ == '__main__':
  main()
