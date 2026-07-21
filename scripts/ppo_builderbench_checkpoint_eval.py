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
