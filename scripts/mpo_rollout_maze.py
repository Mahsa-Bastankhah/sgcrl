"""Plot MPO-CRL maze trajectories (PNG), one per checkpoint.

Companion to ``ppo_rollout_maze.py`` for ``baseline-agents/mpo-crl.py``
checkpoints (pickle payload with ``state.target_policy_params``).

Examples:
  python scripts/mpo_rollout_maze.py \\
      --checkpoint=logs/mpo_crl_impossible_uniform_neg/mpo_crl_point_Impossible_0/checkpoints \\
      --env=point_Impossible \\
      --output=plots/mpo_impossible_s0/ \\
      --skip_existing
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'baseline-agents'))
import sgcrl_jax_acme_compat  # noqa: F401

import argparse
import glob
import json
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from acme import specs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contrastive import utils as contrastive_utils
import env_utils
import mpo_crl_learner
from ppo_contrastive import fixed_goal_dict


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


def _run_config_path(checkpoint_path: str) -> Optional[str]:
  if os.path.isfile(checkpoint_path):
    ckpt_dir = os.path.dirname(os.path.realpath(checkpoint_path))
  else:
    abspath = os.path.abspath(checkpoint_path)
    cands = sorted(glob.glob(os.path.join(abspath, 'ckpt_iter_*.pkl')))
    if cands:
      ckpt_dir = os.path.dirname(os.path.realpath(cands[0]))
    else:
      latest = os.path.join(abspath, 'latest.pkl')
      ckpt_dir = (os.path.dirname(os.path.realpath(latest))
                  if os.path.isfile(latest) else os.path.realpath(abspath))
  path = os.path.join(os.path.dirname(ckpt_dir), 'run_config.json')
  return path if os.path.isfile(path) else None


def _policy_hidden_and_scale(checkpoint_path: str) -> Tuple[Tuple[int, ...], float]:
  cfg_path = _run_config_path(checkpoint_path)
  hidden = (256, 256, 256, 256, 256, 256)
  init_scale = 0.7
  if cfg_path is None:
    return hidden, init_scale
  with open(cfg_path, 'r', encoding='utf-8') as fh:
    run_cfg: Dict[str, Any] = json.load(fh)
  flags = run_cfg.get('flags', {})
  hs = flags.get('mpo_policy_hidden_sizes') or flags.get('hidden_layer_sizes')
  if hs:
    hidden = tuple(int(x) for x in str(hs).split(',') if str(x).strip())
  init_scale = float(flags.get('mpo_policy_init_scale', init_scale))
  return hidden, init_scale


def _load_mpo_checkpoint(path: str) -> Dict[str, Any]:
  with open(path, 'rb') as handle:
    payload = pickle.load(handle)
  state = payload['state']
  return {
      'policy_params': state.target_policy_params,
      'iteration': int(payload.get('iteration', -1)),
      'global_step': int(payload.get('global_step', -1)),
  }


def _build_policy(env_name: str, seed: int, hidden, init_scale):
  # Match training: acme-wrapped env via contrastive_utils (not raw gym).
  probe_env, _ = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed,
      fixed_start_end=fixed_goal_dict[env_name])
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env
  return mpo_crl_learner.make_mpo_policy(
      env_spec, hidden_sizes=hidden, init_scale=init_scale)


def _get_raw_point_env(env_name):
  if not env_name.startswith('point_'):
    raise ValueError(f'maze-only script; got env={env_name!r}')
  gym_env, obs_dim, max_steps = env_utils.load(
      env_name, fixed_start_end=fixed_goal_dict[env_name], seed=None)
  walls = np.asarray(gym_env._walls)
  return gym_env, obs_dim, max_steps, walls


def _rollout_one(policy_network, policy_params, gym_env, max_steps: int,
                 stochastic: bool, seed: int):
  act_min = np.asarray(gym_env.action_space.low, dtype=np.float32)
  act_max = np.asarray(gym_env.action_space.high, dtype=np.float32)

  @jax.jit
  def policy_mode(params, obs):
    dist = policy_network.apply(params, obs[None])
    return dist.mode()[0]

  @jax.jit
  def policy_sample(params, obs, key):
    dist = policy_network.apply(params, obs[None])
    return dist.sample(seed=key)[0]

  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  state_dim = len(obs) // 2
  goal = obs[state_dim:].copy()
  states = [obs[:state_dim].copy()]
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(seed)
  for _ in range(max_steps):
    if stochastic:
      rng, k = jax.random.split(rng)
      action = np.asarray(policy_sample(policy_params, obs, k), dtype=np.float32)
    else:
      action = np.asarray(policy_mode(policy_params, obs), dtype=np.float32)
    action = np.clip(action, act_min, act_max)
    obs_next, r, done, _ = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    states.append(obs[:state_dim].copy())
    total_reward += float(r)
    if float(r) > 0.0:
      success = True
    if done:
      break
  return np.stack(states, axis=0), goal, total_reward, success


def _plot_trajectory(walls, trajectories, goal, title, out_path, fig_scale=2.0):
  H, W = walls.shape
  fig, ax = plt.subplots(
      1, 1,
      figsize=(7 * float(fig_scale), (6 * H / max(W, 1)) * float(fig_scale)),
      layout='constrained')
  ax.imshow(walls, origin='upper', cmap='Greys',
            extent=(-0.5, W - 0.5, H - 0.5, -0.5), vmin=0, vmax=1, alpha=0.85)
  colors = plt.cm.tab10(np.linspace(0, 1, max(len(trajectories), 1)))
  for i, states in enumerate(trajectories):
    xy = states[:, :2]
    ax.plot(xy[:, 1], xy[:, 0], '-', color=colors[i], lw=1.5, alpha=0.85,
            label=f'ep{i}')
    ax.scatter(xy[0, 1], xy[0, 0], c=[colors[i]], s=40, marker='o', zorder=3)
    ax.scatter(xy[-1, 1], xy[-1, 0], c=[colors[i]], s=40, marker='x', zorder=3)
  g = goal[:2]
  ax.scatter([g[1]], [g[0]], c='lime', s=90, marker='*', zorder=4, label='goal')
  ax.set_xlim(-0.5, W - 0.5)
  ax.set_ylim(H - 0.5, -0.5)
  ax.set_aspect('equal')
  ax.set_xlabel('col (x)')
  ax.set_ylabel('row (y, inverted)')
  ax.set_title(title)
  ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
  ax.grid(True, color='gray', linewidth=0.3, alpha=0.3)
  out_dir = os.path.dirname(os.path.abspath(out_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fig.savefig(out_path, dpi=120)
  plt.close(fig)


def _resolve_output_path(output_arg: str, env: str, label: str, multi: bool) -> str:
  is_dir_like = output_arg.endswith(os.sep) or os.path.isdir(output_arg)
  if is_dir_like:
    return os.path.join(output_arg, f'{env}_{label}.png')
  if not multi:
    return output_arg
  stem, ext = os.path.splitext(output_arg)
  return f'{stem}_{label}{ext or ".png"}'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--env', default='point_Impossible')
  parser.add_argument('--output', required=True)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num_trajectories', type=int, default=5)
  parser.add_argument('--max_steps', type=int, default=-1)
  parser.add_argument('--fig_scale', type=float, default=2.0)
  parser.add_argument('--stochastic', action='store_true')
  parser.add_argument('--skip_existing', action='store_true')
  args = parser.parse_args()

  os.environ.setdefault('JAX_PLATFORMS', 'cpu')
  os.environ.setdefault('MPLBACKEND', 'Agg')

  entries = _enumerate_checkpoints(args.checkpoint)
  if not entries:
    print(f'[mpo_maze] no checkpoints at {args.checkpoint!r}')
    return
  print(f'[mpo_maze] found {len(entries)} checkpoint(s)')

  hidden, init_scale = _policy_hidden_and_scale(args.checkpoint)
  print(f'[mpo_maze] policy hidden={hidden} init_scale={init_scale}')
  policy_network = _build_policy(args.env, args.seed, hidden, init_scale)
  gym_env, _, env_max_steps, walls = _get_raw_point_env(args.env)
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  multi = len(entries) > 1
  stochastic = bool(args.stochastic or args.num_trajectories > 1)

  for label, path in entries:
    out_path = _resolve_output_path(args.output, args.env, label, multi)
    if args.skip_existing and os.path.isfile(out_path):
      print(f'[mpo_maze] skip existing {out_path}', flush=True)
      continue
    print(f'[mpo_maze] === {label} ({path}) ===', flush=True)
    ckpt = _load_mpo_checkpoint(path)
    policy_params = ckpt['policy_params']
    print(f'[mpo_maze]   iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}', flush=True)

    trajs = []
    rewards = []
    successes = []
    goal = None
    for i in range(max(1, int(args.num_trajectories))):
      states, goal_i, rew, succ = _rollout_one(
          policy_network, policy_params, gym_env, max_steps,
          stochastic=stochastic, seed=args.seed + i)
      trajs.append(states)
      rewards.append(rew)
      successes.append(succ)
      goal = goal_i
    title = (f'{args.env}  {label}  '
             f'rew_mean={float(np.mean(rewards)):.2f}  '
             f'success_frac={float(np.mean(successes)):.2f}')
    _plot_trajectory(walls, trajs, goal, title, out_path, args.fig_scale)
    print(f'[mpo_maze]   wrote {out_path}', flush=True)


if __name__ == '__main__':
  main()
