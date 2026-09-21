#!/usr/bin/env python3
"""Sawyer NF binary rank accuracy on a checkpoint.

Stochastic rollouts. For (s_t, a_t) the positive goal is a truncated-geometric
future achieved state (γ=0.99). The negative is an achieved state from a
random other timestep. Report

    P[log p(g_pos | s_t, a_t) > log p(g_neg | s_t, a_t)]

  python -u scripts/eval_sawyer_nf_binary_accuracy.py \\
      --run_dir=logs/.../ppo_sawyer_bin_0 --latest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import sgcrl_jax_acme_compat  # noqa: F401

import importlib.util as _ilu
import jax
import jax.numpy as jnp
import numpy as np

from contrastive import nf_density as _nf
from contrastive import ppo_learner
import env_utils
from ppo_contrastive import fixed_goal_dict

_vid_path = os.path.join(REPO, 'scripts', 'render_sawyer_nf_traj_reward_video.py')
_spec = _ilu.spec_from_file_location('sawyer_nf_vid', _vid_path)
_vid = _ilu.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_vid)

CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')


def _latest_ckpt(run_dir: str) -> tuple[int, str]:
  paths = glob.glob(os.path.join(run_dir, 'checkpoints', 'ckpt_iter_*.pkl'))
  if not paths:
    raise FileNotFoundError(f'no checkpoints in {run_dir}')
  best = max(paths, key=lambda p: int(CKPT_RE.search(os.path.basename(p)).group(1)))
  return int(CKPT_RE.search(os.path.basename(best)).group(1)), best


def _env_kwargs(run_dir: str) -> dict:
  cfg_path = os.path.join(run_dir, 'run_config.json')
  flags = {}
  if os.path.isfile(cfg_path):
    with open(cfg_path, encoding='utf-8') as fh:
      payload = json.load(fh)
    flags = payload.get('flags') or {}
  kw = {'randomize_init': bool(flags.get('sawyer_randomize_init', True))}
  if flags.get('sawyer_bin_safe_grasp_reset'):
    kw['safe_grasp_reset'] = True
  if flags.get('bin_randomize_gripper_init'):
    kw['randomize_gripper_init'] = True
  if flags.get('sawyer_bin_bounded_step_mocap'):
    kw['bounded_step_mocap'] = True
  if flags.get('sawyer_bin_selective_antiwindup'):
    kw['selective_antiwindup_mocap'] = True
  if flags.get('sawyer_bin_trackerr_terminate_20cm'):
    kw['terminate_tracking_error'] = True
  steps = int(flags.get('sawyer_max_episode_steps', -1) or -1)
  if steps > 0:
    kw['max_episode_steps'] = steps
  return kw


def _sample_future_offsets(n: int, remaining: np.ndarray, gamma: float, rng):
  offsets = np.ones(n, dtype=np.int32)
  live = remaining > 1
  if not np.any(live):
    return offsets
  u = rng.random(int(np.sum(live)))
  geo = 1 + np.floor(np.log1p(-u) / np.log(gamma)).astype(np.int32)
  rem = remaining[live]
  offsets[live] = np.minimum(geo, rem)
  return offsets


def _pairs(states, actions, terminals, gamma, rng):
  """states (T,D_s), terminals True on last step of each episode."""
  t_idx = np.where(terminals)[0]
  starts = np.concatenate([[0], t_idx[:-1] + 1]) if len(t_idx) else np.array([0])
  ends = t_idx if len(t_idx) else np.array([len(states) - 1])
  anchors, acts, pos, neg, deltas = [], [], [], [], []
  all_states = states
  for s0, s1 in zip(starts, ends):
    length = int(s1 - s0 + 1)
    if length < 2:
      continue
    remaining = np.arange(length - 1, -1, -1)
    valid = remaining > 0
    idx = np.where(valid)[0]
    if idx.size == 0:
      continue
    off = _sample_future_offsets(idx.size, remaining[idx], gamma, rng)
    for i, dt in zip(idx, off):
      j = int(i + dt)
      k = int(rng.integers(0, len(all_states)))
      anchors.append(states[s0 + i])
      acts.append(actions[s0 + i])
      pos.append(states[s0 + j])
      neg.append(all_states[k])
      deltas.append(int(dt))
  return (
      np.asarray(anchors), np.asarray(acts), np.asarray(pos),
      np.asarray(neg), np.asarray(deltas))


def _rollout(gym_env, networks, policy_params, max_steps, n_episodes, seed):
  @jax.jit
  def _mode(params, obs):
    return networks.policy_network.apply(params, obs).mode()

  states, acts, terms = [], [], []
  rng = np.random.default_rng(seed)
  for ep in range(n_episodes):
    obs = gym_env.reset()
    for t in range(max_steps):
      obs_np = np.asarray(obs, dtype=np.float32)
      action = np.asarray(_mode(policy_params, obs_np[None])[0], dtype=np.float32)
      states.append(obs_np[:7])
      acts.append(action)
      obs, _, done, _ = gym_env.step(action)
      last = bool(done) or t == max_steps - 1
      terms.append(last)
      if done:
        break
  return np.asarray(states), np.asarray(acts), np.asarray(terms)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True)
  p.add_argument('--latest', action='store_true')
  p.add_argument('--iteration', type=int, default=-1)
  p.add_argument('--episodes', type=int, default=20)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--future_discount', type=float, default=0.99)
  args = p.parse_args()
  run_dir = os.path.abspath(args.run_dir)
  if args.latest or args.iteration < 0:
    iteration, ckpt_path = _latest_ckpt(run_dir)
  else:
    iteration = int(args.iteration)
    ckpt_path = os.path.join(
        run_dir, 'checkpoints', f'ckpt_iter_{iteration:07d}.pkl')

  settings = _vid._load_run_settings(run_dir)
  arch = _vid._load_nf_arch(run_dir)
  networks, nf_nets, obs_dim, goal_dim = _vid._build_networks(
      'sawyer_bin', seed=args.seed, settings=settings, arch=arch)
  gym_env, _, max_steps = env_utils.load(
      'sawyer_bin', fixed_start_end=fixed_goal_dict['sawyer_bin'],
      seed=args.seed, **_env_kwargs(run_dir))
  ckpt = ppo_learner.load_checkpoint(ckpt_path)
  nf_key = 'q_params_ema' if 'q_params_ema' in ckpt else 'q_params'
  goal_mean, goal_std = _vid._load_goal_stats(run_dir, iteration, goal_dim)

  states, actions, terms = _rollout(
      gym_env, networks, ckpt['policy_params'], int(max_steps),
      int(args.episodes), int(args.seed))
  rng = np.random.default_rng(int(args.seed) + 17)
  anchors, acts, pos_g, neg_g, deltas = _pairs(
      states, actions, terms, float(args.future_discount), rng)
  if len(anchors) < 32:
    raise SystemExit(f'too few pairs: {len(anchors)}')

  @jax.jit
  def _score(nf_params, s, a, g, mean, std):
    g_n = (g - mean) / (std + 1e-8)
    return _nf.nf_log_prob(nf_nets, nf_params, s, a, g_n)

  nf_params = ckpt[nf_key]
  pos = np.asarray(_score(nf_params, anchors, acts, pos_g, goal_mean, goal_std))
  neg = np.asarray(_score(nf_params, anchors, acts, neg_g, goal_mean, goal_std))
  acc = float(np.mean(pos > neg))
  se = float(np.sqrt(acc * (1.0 - acc) / len(pos)))
  print(
      f'BINARY_ACC run={os.path.basename(run_dir)} iter={iteration} '
      f'n={len(pos)} acc={acc:.4f} se={se:.4f} '
      f'pos={float(np.mean(pos)):.3f} neg={float(np.mean(neg)):.3f} '
      f'margin={float(np.mean(pos-neg)):.3f} nf={nf_key}',
      flush=True)
  print(json.dumps({
      'run_dir': run_dir,
      'iteration': iteration,
      'num_pairs': int(len(pos)),
      'binary_accuracy': acc,
      'binary_accuracy_se': se,
      'positive_logp_mean': float(np.mean(pos)),
      'negative_logp_mean': float(np.mean(neg)),
      'margin_mean': float(np.mean(pos - neg)),
      'future_delta_mean': float(np.mean(deltas)),
      'nf_params_source': nf_key,
  }))


if __name__ == '__main__':
  main()
