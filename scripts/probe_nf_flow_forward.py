#!/usr/bin/env python3
"""Roll out NF PPO checkpoints and dump RealNVP ``||z||`` and ``|det|``.

``z = f(g|s,a)``.  Compares the **task goal** against **NF-training positives**:
truncated-geometric future achieved goals ``s'`` on the same trajectory.
Plots ``||z||`` only (not the latent coords).  Marks first success.  No video.

  python scripts/probe_nf_flow_forward.py \\
      --checkpoint_dir=logs/.../checkpoints \\
      --env=builderbench_creative_5_task2 \\
      --ckpt_stride=3 --skip_iter0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BB = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import sgcrl_jax_acme_compat  # noqa: F401

import importlib.util as _ilu

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contrastive import nf_density as _nf
from contrastive import ppo_learner
from envs.builderbench_utils import (
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

_nfvid_path = os.path.join(_REPO, 'scripts', 'render_nf_traj_reward_video.py')
_nfvid_spec = _ilu.spec_from_file_location('render_nf_traj_reward_video',
                                           _nfvid_path)
_nfvid = _ilu.module_from_spec(_nfvid_spec)
assert _nfvid_spec.loader is not None
_nfvid_spec.loader.exec_module(_nfvid)

_load_train_ctx = _nfvid._load_train_ctx
_make_bb_env = _nfvid._make_bb_env
_make_policy_fn = _nfvid._make_policy_fn
force_video_nopermute_norand = _nfvid.force_video_nopermute_norand
VIDEO_FIXED_START_X = _nfvid.VIDEO_FIXED_START_X
_load_nf_arch = _nfvid._load_nf_arch
_load_goal_stats = _nfvid._load_goal_stats
_build_networks = _nfvid._build_networks
_run_dir_from_ckpt = _nfvid._run_dir_from_ckpt
_compile_rollout_and_states = _nfvid._compile_rollout_and_states
_enumerate_ckpts = _nfvid._enumerate_ckpts

DEFAULT_ENV = 'builderbench_creative_5_task2'
DEFAULT_OUT = 'figs/builderbench/nf_flow_forward/c5t2_nf_compact_extrew1_s1'
SEED = 0
C_SUCC = '#2A9D8F'
C_TASK = '#9B2226'
C_POS = '#4C9BE8'
LOGDET_CLIP = 80.0
N_POS_SHOW = 4


def make_nf_forward_fn(nf_nets):
  """z, log|det ∂z/∂g|, log p.  ``goal`` is (T, G) or (T, K, G)."""

  @jax.jit
  def _one(nf_params, state, action, goal, goal_mean, goal_std):
    g_norm = (goal - goal_mean) / (goal_std + 1e-8)
    return _nf.nf_forward(nf_nets, nf_params, state, action, g_norm)

  @jax.jit
  def _fn(nf_params, state, action, goal, goal_mean, goal_std):
    if goal.ndim == 3:
      def one_k(g_k):
        return _one(nf_params, state, action, g_k, goal_mean, goal_std)
      return jax.vmap(one_k, in_axes=1, out_axes=1)(goal)
    return _one(nf_params, state, action, goal, goal_mean, goal_std)

  return _fn


def _select_ckpts(entries, stride: int, skip_iter0: bool):
  parsed = []
  for label, path in entries:
    m = re.search(r'ckpt_iter_(\d+)', os.path.basename(path))
    it = int(m.group(1)) if m else 0
    parsed.append((it, label, path))
  parsed.sort()
  if skip_iter0:
    parsed = [p for p in parsed if p[0] > 0]
  if not parsed:
    return []
  picked = list(parsed[::max(int(stride), 1)])
  if picked[-1][0] != parsed[-1][0]:
    picked.append(parsed[-1])
  return [(lab, path, it) for it, lab, path in picked]


def _first_success(success: np.ndarray) -> int:
  hit = np.asarray(success, dtype=np.float32) >= 0.5
  return int(np.argmax(hit)) if np.any(hit) else -1


def _mark_success(ax, first_succ: int, t_last: float, *, label: bool):
  if first_succ < 0:
    return
  ax.axvline(first_succ, color=C_SUCC, ls='--', lw=1.3, zorder=4,
             label='first success' if label else None)
  ax.axvspan(first_succ, t_last, color=C_SUCC, alpha=0.10, zorder=0)


def _load_discount(run_dir: str) -> float:
  path = os.path.join(run_dir, 'run_config.json')
  with open(path, 'r', encoding='utf-8') as fh:
    payload = json.load(fh)
  resolved = payload.get('resolved_config', {})
  flags = payload.get('flags', {})
  d = resolved.get('discount', flags.get('discount', 0.99))
  d = float(d)
  if d <= 0.0 or d >= 1.0:
    d = float(resolved.get('ppo_discount', 0.99))
  return d


def _achieved_goals(states, ctx) -> np.ndarray:
  """Cube-xyz (or goal slice) of s_{t+1} at each rollout step. Shape (T, G)."""
  obs = np.asarray(states.obs)
  if obs.ndim == 3:
    obs = obs[:, 0]
  lo = int(ctx.start_index)
  hi = int(ctx.end_index if ctx.end_index != -1 else ctx.obs_dim)
  return np.asarray(obs[:, lo:hi], dtype=np.float32)


def _sample_pos_goals(achieved: np.ndarray, discount: float, rng, n_pos: int):
  """Truncated-geometric future goals, same as EpisodeReplay.sample().

  ``achieved[t]`` is obs_to_goal(s_{t+1}).  For action at t, j ∈ [t+1, T]
  and g_pos = achieved[j-1].
  """
  T, g_dim = achieved.shape
  n_pos = int(n_pos)
  goals = np.empty((T, n_pos, g_dim), dtype=np.float32)
  t_arr = np.arange(T)
  max_d = (T - t_arr).astype(np.int64)
  log_gamma = float(np.log(discount))
  for k in range(n_pos):
    trunc_cdf = 1.0 - np.power(float(discount), max_d.astype(np.float64))
    trunc_cdf = np.maximum(trunc_cdf, 1e-12)
    u = rng.random(T) * trunc_cdf
    d = 1 + np.floor(np.log1p(-u) / log_gamma).astype(np.int64)
    d = np.clip(d, 1, max_d)
    j = t_arr + d
    goals[:, k, :] = achieved[j - 1]
  return goals


def _clip_absdet(log_det: np.ndarray) -> np.ndarray:
  return np.exp(np.clip(np.asarray(log_det, dtype=np.float64),
                        -LOGDET_CLIP, LOGDET_CLIP))


def _plot_det(ax, t, log_det, *, color, lw, label, ls='-', alpha=1.0):
  absdet = _clip_absdet(log_det)
  finite = np.isfinite(absdet) & (absdet > 0)
  if np.any(finite):
    ax.plot(t, np.where(finite, absdet, np.nan), color=color, lw=lw, ls=ls,
            alpha=alpha, label=label)
    ax.set_yscale('log')
    ax.set_ylabel(r'$|\det \partial f/\partial g|$')
  else:
    ax.plot(t, log_det, color=color, lw=lw, ls=ls, alpha=alpha, label=label)
    ax.set_ylabel(r'$\log|\det \partial f/\partial g|$')


def _plot_one(path: str, *, iteration: int, t, success,
              z_norm_task, log_det_task, z_norm_pos, log_det_pos,
              mean_znorm_task, mean_znorm_pos, mean_logdet_task,
              mean_logdet_pos, discount: float, n_pos: int):
  first = _first_success(success)
  t_last = float(t[-1])
  n_show = min(N_POS_SHOW, z_norm_pos.shape[1])
  fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True)
  succ_note = (f'first success t={first}' if first >= 0 else 'no success')

  ax = axes[0]
  for k in range(n_show):
    ax.plot(t, z_norm_pos[:, k], color=C_POS, lw=0.7, alpha=0.28,
            label=r'pos $s_j$ samples' if k == 0 else None)
  ax.plot(t, z_norm_pos.mean(axis=1), color=C_POS, lw=1.7,
          label=rf'pos mean $\|z\|$={mean_znorm_pos:.3g}')
  ax.plot(t, z_norm_task, color=C_TASK, lw=1.8,
          label=rf'task $g$  $\|z\|$ mean={mean_znorm_task:.3g}')
  _mark_success(ax, first, t_last, label=True)
  ax.set_ylabel(r'$\|z\|_2=\|f(g\mid s,a)\|_2$')
  ax.set_title(
      rf'iter {iteration}  ·  {succ_note}  ·  '
      rf'pos = truncGeom($\gamma$={discount:g}) future $s_j$, $K$={n_pos}')
  ax.grid(True, alpha=0.25)
  ax.legend(fontsize=8, loc='best')

  ax = axes[1]
  for k in range(n_show):
    _plot_det(ax, t, log_det_pos[:, k], color=C_POS, lw=0.7, label=None,
              alpha=0.28)
  _plot_det(ax, t, log_det_pos.mean(axis=1), color=C_POS, lw=1.7,
            label=rf'pos mean $\log|\det|$={mean_logdet_pos:.3g}')
  _plot_det(ax, t, log_det_task, color=C_TASK, lw=1.8,
            label=rf'task $g$  mean $\log|\det|$={mean_logdet_task:.3g}')
  _mark_success(ax, first, t_last, label=False)
  ax.set_xlabel('timestep')
  ax.grid(True, alpha=0.25, which='both')
  ax.legend(fontsize=8, loc='best')

  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)
  print(f'[fwd] wrote {path}', flush=True)


def _plot_summary(path: str, rows: list, tag: str):
  if not rows:
    return
  fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
  xs = np.asarray([r['iteration'] for r in rows], dtype=float)
  zt = np.asarray([r['mean_znorm_task'] for r in rows], dtype=float)
  zp = np.asarray([r['mean_znorm_pos'] for r in rows], dtype=float)
  lt = np.asarray([r['mean_logdet_task'] for r in rows], dtype=float)
  lp = np.asarray([r['mean_logdet_pos'] for r in rows], dtype=float)

  ax = axes[0]
  ax.plot(xs, zt, color=C_TASK, lw=1.8, marker='o', ms=5, label=r'task $g$')
  ax.plot(xs, zp, color=C_POS, lw=1.8, marker='o', ms=5,
          label=r'pos future $s_j$')
  ax.set_ylabel(r'mean $\|z\|_2$ over trajectory')
  ax.set_title(rf'{tag}: is task $g$ in-distribution?  (small $\|z\|$ = yes)')
  ax.grid(True, alpha=0.25)
  ax.legend(fontsize=8, loc='best')

  ax = axes[1]
  ax.plot(xs, np.exp(np.clip(lt, -LOGDET_CLIP, LOGDET_CLIP)), color=C_TASK,
          lw=1.8, marker='o', ms=5, label=r'task $g$')
  ax.plot(xs, np.exp(np.clip(lp, -LOGDET_CLIP, LOGDET_CLIP)), color=C_POS,
          lw=1.8, marker='o', ms=5, label=r'pos future $s_j$')
  ax.set_yscale('log')
  ax.set_ylabel(r'mean $|\det|$  (geom. = $\exp(\mathrm{mean}\,\log|\det|)$)')
  ax.set_xlabel('checkpoint iteration')
  ax.grid(True, alpha=0.25, which='both')
  ax.legend(fontsize=8, loc='best')

  fig.tight_layout()
  fig.savefig(path, dpi=140)
  plt.close(fig)
  print(f'[fwd] wrote {path}', flush=True)


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', default='')
  p.add_argument('--checkpoint_dir', default='')
  p.add_argument('--env', default=DEFAULT_ENV)
  p.add_argument('--out_dir', default=DEFAULT_OUT)
  p.add_argument('--tag_prefix', default='c5t2_nf_compact_extrew1_s1')
  p.add_argument('--seed', type=int, default=SEED)
  p.add_argument('--ckpt_stride', type=int, default=3)
  p.add_argument('--skip_iter0', action='store_true', default=True)
  p.add_argument('--no_skip_iter0', action='store_false', dest='skip_iter0')
  p.add_argument('--allow_no_success', action='store_true')
  p.add_argument('--max_tries', type=int, default=1)
  p.add_argument('--n_pos', type=int, default=8,
                 help='Truncated-geometric future positives per (s,a)')
  p.add_argument('--match_run_init', action='store_true')
  p.add_argument('--fixed_start_x', type=float, default=VIDEO_FIXED_START_X)
  return p.parse_args()


def _recon_err(z, log_det, log_p) -> float:
  z = np.asarray(z, dtype=np.float64)
  log_det = np.asarray(log_det, dtype=np.float64)
  log_p = np.asarray(log_p, dtype=np.float64)
  flow_dim = int(z.shape[-1])
  log_norm = -0.5 * flow_dim * np.log(2.0 * np.pi)
  log_prior = -0.5 * np.sum(z ** 2, axis=-1) + log_norm
  return float(np.max(np.abs(log_prior + log_det - log_p)))


def _probe_one(args, *, label, ckpt_path, iteration, ctx, networks, nf_nets,
               env, mocap_targets, ep_len, num_cubes, goal_dim, arch, out_dir,
               fwd_fn, discount: float, obs_dim: int):
  print(f'[fwd] === iter {iteration} ({ckpt_path}) ===', flush=True)
  ckpt = ppo_learner.load_checkpoint(ckpt_path)
  nf_params = ckpt['q_params']
  run_dir = _run_dir_from_ckpt(ckpt_path)
  goal_mean, goal_std = _load_goal_stats(run_dir, iteration, goal_dim)
  goal_std = np.maximum(goal_std, arch['nf_goal_std_min']).astype(np.float32)

  policy = _make_policy_fn(
      networks, ckpt['policy_params'], stochastic=False,
      filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes,
      normalize_obs=False, obs_dim=ctx.obs_dim,
      start_index=ctx.start_index, end_index=ctx.end_index)
  run = _compile_rollout_and_states(
      policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
      ctx.filter_policy_obs)

  key = jax.random.PRNGKey(int(args.seed))
  best = None
  for attempt in range(int(args.max_tries)):
    key, roll_key = jax.random.split(key)
    traj, states = run(roll_key)
    packed = np.asarray(traj['packed'], dtype=np.float32)
    actions = np.asarray(traj['action'], dtype=np.float32)
    succ = np.asarray(traj['success'], dtype=np.float32)
    reached = bool(np.any(succ >= 0.5))
    print(f'[fwd] seed={args.seed} try={attempt} success={reached} '
          f'first_succ={int(np.argmax(succ >= 0.5)) if reached else -1}',
          flush=True)
    cur = dict(packed=packed, actions=actions, success=succ, states=states,
               attempt=attempt)
    if reached:
      best = cur
      break
    if best is None:
      best = cur

  if (not np.any(best['success'] >= 0.5)) and (not args.allow_no_success):
    raise RuntimeError('no successful trajectory '
                       '(pass --allow_no_success)')

  packed_np = best['packed']
  actions_np = best['actions']
  success = np.asarray(best['success'], dtype=np.float32)
  state = packed_np[:, :obs_dim]
  g_task = packed_np[:, obs_dim:]
  achieved = _achieved_goals(best['states'], ctx)
  rng = np.random.default_rng(int(args.seed) * 1_000_003 + int(iteration))
  g_pos = _sample_pos_goals(achieved, discount, rng, int(args.n_pos))

  gmean = jnp.asarray(goal_mean)
  gstd = jnp.asarray(goal_std)
  z_t, ld_t, lp_t = fwd_fn(
      nf_params, jnp.asarray(state), jnp.asarray(actions_np),
      jnp.asarray(g_task), gmean, gstd)
  z_p, ld_p, lp_p = fwd_fn(
      nf_params, jnp.asarray(state), jnp.asarray(actions_np),
      jnp.asarray(g_pos), gmean, gstd)
  z_t = np.asarray(z_t, dtype=np.float64)
  ld_t = np.asarray(ld_t, dtype=np.float64)
  lp_t = np.asarray(lp_t, dtype=np.float64)
  z_p = np.asarray(z_p, dtype=np.float64)
  ld_p = np.asarray(ld_p, dtype=np.float64)
  lp_p = np.asarray(lp_p, dtype=np.float64)

  err_t = _recon_err(z_t, ld_t, lp_t)
  err_p = _recon_err(z_p, ld_p, lp_p)
  print(f'[fwd] recon log p vs flow: task max|Δ|={err_t:.3e}  '
        f'pos max|Δ|={err_p:.3e}', flush=True)
  if max(err_t, err_p) > 1e-3:
    raise RuntimeError(
        f'nf_forward disagrees with log p (task={err_t}, pos={err_p})')

  z_norm_task = np.linalg.norm(z_t, axis=-1)
  z_norm_pos = np.linalg.norm(z_p, axis=-1)
  mean_znorm_task = float(z_norm_task.mean())
  mean_znorm_pos = float(z_norm_pos.mean())
  mean_logdet_task = float(ld_t.mean())
  mean_logdet_pos = float(ld_p.mean())
  T = z_norm_task.shape[0]
  t = np.arange(T)
  first = _first_success(success)
  tag = f'{args.tag_prefix}_{label}'

  print(
      f'[fwd] TASK  mean||z||={mean_znorm_task:.4g}  '
      f'mean log|det|={mean_logdet_task:.4g}  '
      f'mean |det|≈{np.exp(np.clip(mean_logdet_task, -LOGDET_CLIP, LOGDET_CLIP)):.4g}',
      flush=True)
  print(
      f'[fwd] POS   mean||z||={mean_znorm_pos:.4g}  '
      f'mean log|det|={mean_logdet_pos:.4g}  '
      f'mean |det|≈{np.exp(np.clip(mean_logdet_pos, -LOGDET_CLIP, LOGDET_CLIP)):.4g}  '
      f'(K={args.n_pos}, γ={discount:g})',
      flush=True)

  csv_path = os.path.join(out_dir, f'{tag}.csv')
  header = ('t,success,z_norm_task,log_absdet_task,z_norm_pos_mean,'
            'log_absdet_pos_mean')
  cols = [
      t.astype(np.float64), success.astype(np.float64), z_norm_task, ld_t,
      z_norm_pos.mean(axis=1), ld_p.mean(axis=1),
  ]
  np.savetxt(csv_path, np.stack(cols, axis=1), delimiter=',',
             header=header, comments='')
  print(f'[fwd] wrote {csv_path}  first_succ={first}', flush=True)

  png = os.path.join(out_dir, f'{tag}.png')
  _plot_one(
      png, iteration=iteration, t=t, success=success,
      z_norm_task=z_norm_task, log_det_task=ld_t,
      z_norm_pos=z_norm_pos, log_det_pos=ld_p,
      mean_znorm_task=mean_znorm_task, mean_znorm_pos=mean_znorm_pos,
      mean_logdet_task=mean_logdet_task, mean_logdet_pos=mean_logdet_pos,
      discount=discount, n_pos=int(args.n_pos))
  return dict(
      iteration=iteration, t=t, first_succ=first, success=success,
      mean_znorm_task=mean_znorm_task, mean_znorm_pos=mean_znorm_pos,
      mean_logdet_task=mean_logdet_task, mean_logdet_pos=mean_logdet_pos,
      z_norm_task=z_norm_task, z_norm_pos_mean=z_norm_pos.mean(axis=1),
      log_det_task=ld_t, log_det_pos_mean=ld_p.mean(axis=1),
  )


def main():
  args = _parse_args()
  if not args.checkpoint and not args.checkpoint_dir:
    raise SystemExit('need --checkpoint or --checkpoint_dir')
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)
  entries = _enumerate_ckpts(args.checkpoint, args.checkpoint_dir)
  selected = _select_ckpts(
      entries, stride=args.ckpt_stride, skip_iter0=args.skip_iter0)
  if not selected:
    raise FileNotFoundError('no checkpoints after stride/skip_iter0')
  print(f'[fwd] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[fwd] {len(selected)}/{len(entries)} checkpoint(s) → {out_dir}')
  print('[fwd] iters=' + ','.join(str(it) for _, _, it in selected), flush=True)

  first_path = selected[0][1]
  env_name = args.env
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, _ = parse_bb_env_id(env_id)
  ctx = _load_train_ctx(env_name, first_path)
  if not args.match_run_init:
    force_video_nopermute_norand(ctx, fixed_start_x=float(args.fixed_start_x))
    print(f'[fwd] forcing nopermute + fixed_start_x={ctx.fixed_start_x}',
          flush=True)
  run_dir = _run_dir_from_ckpt(first_path)
  arch = _load_nf_arch(run_dir)
  discount = _load_discount(run_dir)
  print(f'[fwd] NF train positives: truncGeom γ={discount:g}  K={args.n_pos}',
        flush=True)
  networks, nf_nets, goal_dim = _build_networks(
      env_name, args.seed, ctx, arch)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)
  fwd_fn = make_nf_forward_fn(nf_nets)
  obs_dim = int(ctx.obs_dim)

  rows = []
  for label, path, iteration in selected:
    rows.append(_probe_one(
        args, label=label, ckpt_path=path, iteration=iteration, ctx=ctx,
        networks=networks, nf_nets=nf_nets, env=env,
        mocap_targets=mocap_targets, ep_len=ep_len, num_cubes=num_cubes,
        goal_dim=goal_dim, arch=arch, out_dir=out_dir, fwd_fn=fwd_fn,
        discount=discount, obs_dim=obs_dim))

  means_path = os.path.join(out_dir, f'{args.tag_prefix}_means.csv')
  with open(means_path, 'w', encoding='utf-8') as fh:
    fh.write('iteration,mean_znorm_task,mean_znorm_pos,'
             'mean_logdet_task,mean_logdet_pos,first_succ\n')
    for r in rows:
      fh.write(
          f'{r["iteration"]},{r["mean_znorm_task"]},'
          f'{r["mean_znorm_pos"]},{r["mean_logdet_task"]},'
          f'{r["mean_logdet_pos"]},{r["first_succ"]}\n')
  print(f'[fwd] wrote {means_path}', flush=True)

  _plot_summary(
      os.path.join(out_dir, f'{args.tag_prefix}_summary.png'),
      rows, args.tag_prefix)


if __name__ == '__main__':
  main()
