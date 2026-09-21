#!/usr/bin/env python3
"""Probe whether p(·|s, a=g_norm) peaks at the true task goal g.

Rolls out a stochastic checkpoint policy, then at subsampled states evaluates
log p(sf | s, a_gnorm) for the true packed goal g and K uniform random future
goals sf ~ U([-1,1]^goal_dim). Action is fixed to the oracle a = g_norm
(counterfactual; not necessarily the stepped action).

  python scripts/probe_allegro_nf_action_logp.py \
      --checkpoint=logs/.../ckpt_iter_0000100.pkl \
      --out-dir=figs/allegro_kuka_throw/nf_sf_logp_probe
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from scripts import allegro_kuka_throw_ckpt_video as vid  # noqa: E402


def _eval_logp(reward_fn, nf_params, state, action, goal, gmean, gstd):
  """log p(goal|state,action) via make_nf_reward_fn packing."""
  import jax.numpy as jnp

  state = np.asarray(state, dtype=np.float32).reshape(1, -1)
  action = np.asarray(action, dtype=np.float32).reshape(1, -1)
  goal = np.asarray(goal, dtype=np.float32).reshape(1, -1)
  packed = np.concatenate([state, goal], axis=-1)
  out = reward_fn(
      nf_params,
      jnp.asarray(packed),
      jnp.asarray(action),
      jnp.asarray(gmean),
      jnp.asarray(gstd),
  )
  return float(np.asarray(out).reshape(-1)[0])


def _joint_max_err(env):
  """Physical radian max |q - q*| using env success bookkeeping when present."""
  levels = getattr(env, 'success_levels', None)
  if callable(levels):
    out = levels()
    mae = out.get('mean_abs_joint_err')
    if mae is not None:
      # Prefer max abs from packed hand if available via _diag-like path.
      pass
  d = vid._diag(env)
  return float(d['d_obj']), bool(d['succ'])


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', required=True)
  p.add_argument('--out-dir', required=True)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--pipeline', default='gpu', choices=('gpu', 'cpu'))
  p.add_argument('--num-sf', type=int, default=8,
                 help='Uniform random future goals in [-1,1]^d')
  p.add_argument('--stride', type=int, default=5,
                 help='Evaluate every N-th pre-action state')
  p.add_argument('--sf-seed', type=int, default=0,
                 help='RNG seed for random sf samples')
  args = p.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  flags = vid._load_flags(args.checkpoint)
  env_kw = vid._load_env_kwargs(flags, 0, args.seed, args.pipeline)
  env_kw['enable_cameras'] = False  # probe only; no video
  env = vid._build_env(env_kw)
  ckpt = vid._load_ckpt(args.checkpoint)
  act, policy_params, iteration, _networks = vid._build_actor(
      env, ckpt, flags, deterministic=False)
  reward_fn, nf_params = vid._build_nf_reward(env, ckpt, flags)
  gmean, gstd = vid._goal_stats_from_learner(
      args.checkpoint, iteration, int(env.goal_dim))

  import torch
  import jax
  import jax.numpy as jnp

  obs_dim = int(env.obs_dim)
  goal_dim = int(env.goal_dim)
  act_dim = int(env.action_dim)
  if act_dim != goal_dim:
    raise RuntimeError(
        f'oracle a=g_norm requires action_dim==goal_dim, got '
        f'{act_dim}!={goal_dim}')

  n_steps = int(env.max_episode_steps)
  stride = max(int(args.stride), 1)
  k_sf = int(args.num_sf)
  sf_rng = np.random.default_rng(int(args.sf_seed))

  vid._set_reset_seed(int(args.seed))
  obs_t = env.reset()
  key = jax.random.PRNGKey(int(args.seed) + 17)

  rows = []
  for t in range(n_steps):
    packed = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
    if packed.ndim == 1:
      packed = packed[None]
    packed = packed.reshape(-1)
    state = packed[:obs_dim].copy()
    g_true = packed[obs_dim:].copy()
    a_oracle = g_true.copy()  # trim_sa absolute normalized targets

    key, sub = jax.random.split(key)
    a_policy = np.asarray(
        act(policy_params, jnp.asarray(packed[None]), sub),
        dtype=np.float32).reshape(-1)

    evaluate = (t % stride == 0)
    if evaluate:
      logp_g = _eval_logp(
          reward_fn, nf_params, state, a_oracle, g_true, gmean, gstd)
      sf_list = sf_rng.uniform(-1.0, 1.0, size=(k_sf, goal_dim)).astype(
          np.float32)
      logp_sf = np.asarray([
          _eval_logp(reward_fn, nf_params, state, a_oracle, sf, gmean, gstd)
          for sf in sf_list
      ], dtype=np.float32)
      best_sf = float(np.max(logp_sf))
      mean_sf = float(np.mean(logp_sf))
      gap_best = float(logp_g - best_sf)
      g_wins = bool(logp_g > best_sf + 1e-8)
      max_err, hard = _joint_max_err(env)
      row = {
          't': t,
          'max_joint_err_rad': max_err,
          'hard_succ': int(hard),
          'logp_g': float(logp_g),
          'logp_sf_best': best_sf,
          'logp_sf_mean': mean_sf,
          'gap_g_minus_best_sf': gap_best,
          'g_beats_all_sf': int(g_wins),
          'logp_policy_at_g': float(_eval_logp(
              reward_fn, nf_params, state, a_policy, g_true, gmean, gstd)),
      }
      for i, (sf, lp) in enumerate(zip(sf_list, logp_sf)):
        row[f'logp_sf_{i}'] = float(lp)
        for j in range(goal_dim):
          row[f'sf{i}_{j}'] = float(sf[j])
      for j in range(goal_dim):
        row[f'g_{j}'] = float(g_true[j])
      rows.append(row)
      if t % (stride * 5) == 0:
        print(
            f'[nf_sf_probe] t={t:3d} logp(g)={logp_g:+.3f} '
            f'best_sf={best_sf:+.3f} gap={gap_best:+.3f} '
            f'g_wins={int(g_wins)} max|q-q*|={max_err:.3f}',
            flush=True)

    a_t = torch.as_tensor(a_policy[None], device=env.device)
    obs_t, _, done_t = env.step(a_t)
    done = float(np.asarray(done_t.detach().cpu().numpy()).reshape(-1)[0])
    if done >= 0.5:
      print(f'[nf_sf_probe] episode done at t={t}', flush=True)
      break

  if not rows:
    raise RuntimeError('no probe rows collected')

  tag = f'iter{iteration:07d}_seed{args.seed}_ksf{k_sf}_stride{stride}'
  csv_path = os.path.join(args.out_dir, f'nf_sf_logp_{tag}.csv')
  fieldnames = list(rows[0].keys())
  with open(csv_path, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  gaps = np.asarray([r['gap_g_minus_best_sf'] for r in rows], dtype=np.float64)
  wins = np.asarray([r['g_beats_all_sf'] for r in rows], dtype=np.float64)
  logp_g = np.asarray([r['logp_g'] for r in rows], dtype=np.float64)
  logp_best = np.asarray([r['logp_sf_best'] for r in rows], dtype=np.float64)
  logp_mean = np.asarray([r['logp_sf_mean'] for r in rows], dtype=np.float64)
  ts = np.asarray([r['t'] for r in rows], dtype=np.float64)

  print(
      f'[nf_sf_probe] N={len(rows)} states  K={k_sf} uniform sf in [-1,1]^{goal_dim}\n'
      f'  frac g beats all sf: {wins.mean():.3f}\n'
      f'  gap logp(g)-max_sf: mean={gaps.mean():+.4f}  '
      f'median={np.median(gaps):+.4f}  '
      f'frac>0={float(np.mean(gaps > 0)):.3f}',
      flush=True)

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(9, 4.5))
  ax.plot(ts, logp_g, 'o-', color='#1b9e77', lw=2.0, ms=4,
          label=r'$\log p(g \mid s, a{=}g)$')
  ax.plot(ts, logp_best, 's--', color='#d95f02', lw=1.5, ms=3.5,
          label=rf'max$_{{i}}$ $\log p(s_f^{{(i)}} \mid s, a{{=}}g)$  (K={k_sf})')
  ax.plot(ts, logp_mean, ':', color='#7570b3', lw=1.5,
          label=rf'mean$_i$ $\log p(s_f^{{(i)}} \mid s, a{{=}}g)$')
  ax.axhline(0.0, color='0.6', lw=0.8)
  ax.set_xlabel('timestep (policy rollout)')
  ax.set_ylabel('NF log-density')
  ax.set_title(
      f'Does p(·|s,a=g) peak at true g?  iter={iteration}  '
      f'frac g wins={wins.mean():.2f}  mean gap={gaps.mean():+.2f}')
  ax.legend(loc='best', fontsize=9)
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  png_path = os.path.join(args.out_dir, f'nf_sf_logp_{tag}.png')
  fig.savefig(png_path, dpi=140)
  plt.close(fig)

  print(f'[nf_sf_probe] wrote {csv_path}', flush=True)
  print(f'[nf_sf_probe] wrote {png_path}', flush=True)


if __name__ == '__main__':
  main()
