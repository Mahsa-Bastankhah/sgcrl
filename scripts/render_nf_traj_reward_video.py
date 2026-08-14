"""Render BuilderBench traj video with live online NF log-prob reward strip.

Stacked frame: reward timeline on top + MuJoCo below. Reward is
``r = log p_NF(g|s)`` (state_only) or ``log p_NF(g|s,a)`` from online
``q_params`` (not the tau-weighted EMA).

Examples:
  python scripts/render_nf_traj_reward_video.py \\
      --checkpoint_dir=logs/.../checkpoints \\
      --env=builderbench_creative_5_task2 \\
      --tag_prefix=c5t2_nf_stateonly_s3 \\
      --allow_no_success --skip_existing
"""
from __future__ import annotations

import csv
import glob
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
from acme import specs
from PIL import Image

import contrastive
from contrastive import nf_density as _nf
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from envs.builderbench_utils import (
    filter_pd_policy_state_obs,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)

_bbv_path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_rollout_video.py')
_bbv_spec = _ilu.spec_from_file_location('ppo_builderbench_rollout_video', _bbv_path)
_bbv = _ilu.module_from_spec(_bbv_spec)
assert _bbv_spec.loader is not None
_bbv_spec.loader.exec_module(_bbv)
_load_train_ctx = _bbv._load_train_ctx
_make_bb_env = _bbv._make_bb_env
_make_policy_fn = _bbv._make_policy_fn
_maybe_fix_target = _bbv._maybe_fix_target
force_video_nopermute_norand = _bbv.force_video_nopermute_norand
VIDEO_FIXED_START_X = _bbv.VIDEO_FIXED_START_X

# Reuse strip / compose / write helpers from the CRL reward video script.
_crl_path = os.path.join(_REPO, 'scripts', 'render_frozen_crl_traj_reward_video.py')
_crl_spec = _ilu.spec_from_file_location('render_frozen_crl_traj_reward_video', _crl_path)
_crl = _ilu.module_from_spec(_crl_spec)
assert _crl_spec.loader is not None
_crl_spec.loader.exec_module(_crl)
_render_reward_strip = _crl._render_reward_strip
_render_grad_norm_strip = _crl._render_grad_norm_strip
_render_select_strip = _crl._render_select_strip
_render_pos_delta_strip = _crl._render_pos_delta_strip
_compose_frame = _crl._compose_frame
_write_mp4 = _crl._write_mp4
_compile_rollout_and_states = _crl._compile_rollout_and_states
select_action_to_cube = _crl.select_action_to_cube
cube_step_deltas_from_pos = _crl.cube_step_deltas_from_pos

import argparse

DEFAULT_ENV = 'builderbench_creative_5_task2'
OUT_DIR = 'figs/builderbench/nf_logp_reward_probe'
SEED = 0
MAX_TRIES = 1  # single det-policy rollout (env reset seed only)
FPS = 8
HOLD_LAST = 8


def _run_dir_from_ckpt(ckpt_path: str) -> str:
  ckpt_dir = os.path.dirname(os.path.realpath(ckpt_path))
  return os.path.dirname(ckpt_dir)


def _load_nf_arch(run_dir: str):
  with open(os.path.join(run_dir, 'run_config.json'), 'r', encoding='utf-8') as fh:
    payload = json.load(fh)
  resolved = payload.get('resolved_config', {})
  flags = payload.get('flags', {})
  return {
      'nf_rep_size': int(resolved.get('nf_rep_size', flags.get('nf_rep_size', 64))),
      'nf_num_blocks': int(resolved.get('nf_num_blocks', flags.get('nf_num_blocks', 8))),
      'nf_coupling_width': int(resolved.get(
          'nf_coupling_width', flags.get('nf_coupling_width', 256))),
      'nf_goal_enc_size': int(resolved.get(
          'nf_goal_enc_size', flags.get('nf_goal_enc_size', 0))),
      'nf_sa_hidden': int(resolved.get('nf_sa_hidden', flags.get('nf_sa_hidden', 1024))),
      'nf_sa_num_layers': int(resolved.get(
          'nf_sa_num_layers', flags.get('nf_sa_num_layers', 4))),
      'nf_state_only': bool(resolved.get(
          'nf_state_only', flags.get('nf_state_only', False))),
      'nf_goal_std_min': float(resolved.get(
          'nf_goal_std_min', flags.get('nf_goal_std_min', 0.02))),
      'hidden_layer_sizes': tuple(int(x) for x in resolved.get(
          'hidden_layer_sizes', (256,) * 6)),
  }


def _load_goal_stats(run_dir: str, iteration: int, goal_dim: int):
  csv_path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
  best, best_dist = None, None
  with open(csv_path, 'r', encoding='utf-8') as fh:
    for row in csv.DictReader(fh):
      ls = int(float(row.get('learner_steps') or row.get('iteration') or 0))
      dist = abs(ls - int(iteration))
      v = row.get('nf/goal_mean_0')
      if v in (None, '', 'nan'):
        continue
      if best_dist is None or dist < best_dist:
        best_dist = dist
        best = row
  if best is None:
    print('[vid] WARNING: no nf/goal_* stats; using mean=0 std=1')
    return (np.zeros(goal_dim, dtype=np.float32),
            np.ones(goal_dim, dtype=np.float32))
  mean = np.asarray(
      [float(best[f'nf/goal_mean_{i}']) for i in range(goal_dim)],
      dtype=np.float32)
  std = np.asarray(
      [float(best[f'nf/goal_std_{i}']) for i in range(goal_dim)],
      dtype=np.float32)
  print(f'[vid] goal stats from learner_steps={best.get("learner_steps")} '
        f'(|Δ|={best_dist})', flush=True)
  return mean, std


def _build_networks(env_name: str, seed: int, ctx, arch: dict):
  env_kwargs = {}
  if ctx.use_pd:
    env_kwargs['builderbench_use_pd'] = True
    env_kwargs['builderbench_pd_duration'] = ctx.pd_duration
    env_kwargs['builderbench_pd_filter_policy_obs'] = ctx.filter_policy_obs
  env_kwargs['builderbench_permute_start_boxes'] = ctx.permute_start_boxes
  if getattr(ctx, 'fixed_start_x', None) is not None:
    env_kwargs['builderbench_fixed_start_x'] = float(ctx.fixed_start_x)
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, ctx.start_index, ctx.end_index, seed=seed,
      fixed_start_end=ctx.fixed_target_goal, **env_kwargs)
  env_spec = specs.make_environment_spec(probe_env)
  act_dim = int(np.prod(env_spec.actions.shape))
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
      categorical_select_classes=ctx.categorical_select_classes,
      categorical_select_waypoint=bool(getattr(
          ctx, 'categorical_select_waypoint', False)),
  )
  goal_dim = int(ctx.end_index - ctx.start_index) if ctx.end_index != -1 else int(ctx.obs_dim)
  print(f'[vid] NF arch: rep={arch["nf_rep_size"]} blocks={arch["nf_num_blocks"]} '
        f'channels={arch["nf_coupling_width"]} sa={arch["nf_sa_num_layers"]}x'
        f'{arch["nf_sa_hidden"]} state_only={arch["nf_state_only"]} '
        f'goal_dim={goal_dim}', flush=True)
  nf_nets = _nf.make_nf_density_networks(
      obs_dim=int(ctx.obs_dim),
      act_dim=act_dim,
      goal_dim=goal_dim,
      hidden_layer_sizes=arch['hidden_layer_sizes'],
      rep_size=arch['nf_rep_size'],
      num_blocks=arch['nf_num_blocks'],
      channels=arch['nf_coupling_width'],
      goal_enc_size=arch['nf_goal_enc_size'],
      sa_hidden=arch['nf_sa_hidden'],
      sa_num_layers=arch['nf_sa_num_layers'],
      state_only=bool(arch['nf_state_only']),
  )
  return networks, nf_nets, goal_dim


def make_nf_logp_grad_norm_fn(nf_nets, obs_dim: int):
  """‖∇_s log p_NF(g|s,a)‖ and ‖∇_a log p_NF(g|s,a)‖, g held fixed.

  Goal is normalized the same way as ``make_nf_reward_fn``. Action is ignored
  by the encoder when the nets were built with ``state_only=True``.
  """

  def _logp(nf_params, s, a, g_norm):
    lp = _nf.nf_log_prob(
        nf_nets, nf_params, s[None], a[None], g_norm[None])
    return jnp.sum(lp)

  @jax.jit
  def grad_norm_fn(nf_params, packed, action, goal_mean, goal_std):
    s = packed[:, :obs_dim]
    g = packed[:, obs_dim:]
    g_norm = (g - goal_mean) / (goal_std + 1e-8)

    def one(s_i, a_i, g_i):
      gs = jax.grad(_logp, argnums=1)(nf_params, s_i, a_i, g_i)
      ga = jax.grad(_logp, argnums=2)(nf_params, s_i, a_i, g_i)
      return jnp.linalg.norm(gs), jnp.linalg.norm(ga)

    gs_n, ga_n = jax.vmap(one)(s, action, g_norm)
    return gs_n, ga_n

  return grad_norm_fn


def _enumerate_ckpts(checkpoint: str, checkpoint_dir: str):
  if checkpoint_dir:
    files = sorted(
        glob.glob(os.path.join(checkpoint_dir, 'ckpt_iter_*.pkl')),
        key=lambda p: int(re.search(r'ckpt_iter_(\d+)\.pkl$', p).group(1)))
    out = []
    for p in files:
      it = int(re.search(r'ckpt_iter_(\d+)\.pkl$', p).group(1))
      out.append((f'iter_{it:07d}', p))
    return out
  base = os.path.splitext(os.path.basename(checkpoint))[0]
  label = base[len('ckpt_'):] if base.startswith('ckpt_') else base
  return [(label, checkpoint)]


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint', default='')
  p.add_argument('--checkpoint_dir', default='')
  p.add_argument('--env', default=DEFAULT_ENV)
  p.add_argument('--out_dir', default=OUT_DIR)
  p.add_argument('--tag', default='')
  p.add_argument('--tag_prefix', default='c5t2_nf_online')
  p.add_argument('--title', default='')
  p.add_argument('--allow_no_success', action='store_true')
  p.add_argument('--max_tries', type=int, default=MAX_TRIES)
  p.add_argument('--seed', type=int, default=SEED)
  p.add_argument('--fps', type=int, default=FPS)
  p.add_argument('--skip_existing', action='store_true')
  p.add_argument('--show_select', action='store_true',
                 help='Add a timeline strip for state select_action / cube id')
  p.add_argument('--show_pos_delta', action='store_true',
                 help='Add per-cube ||Δxyz|| strip from the reward state s')
  p.add_argument('--normalize_reward', action='store_true',
                 help='Also show a 2nd reward strip normalised by episode std.')
  p.add_argument('--show_logp_grad', action='store_true',
                 help='Add ||∇_s log p_NF|| and ||∇_a log p_NF|| strips '
                      '(g held fixed)')
  p.add_argument('--match_run_init', action='store_true',
                 help='Use permute/fixed_start_x from run_config (default: '
                      'force nopermute + fixed_start_x=0.1)')
  p.add_argument('--fixed_start_x', type=float, default=VIDEO_FIXED_START_X,
                 help='Start-box x when forcing norand')
  return p.parse_args()


def _render_one(args, *, label, ckpt_path, ctx, networks, nf_nets, env,
                mocap_targets, ep_len, num_cubes, goal_dim, arch, out_dir):
  tag = args.tag if (args.tag and not args.checkpoint_dir) else (
      f'{args.tag_prefix}_{label}')
  out_mp4 = os.path.join(out_dir, f'{tag}.mp4')
  still_path = os.path.join(out_dir, f'{tag}_still.png')
  csv_path = os.path.join(out_dir, f'{tag}.csv')
  if args.skip_existing and os.path.isfile(out_mp4):
    print(f'[vid] skip existing {out_mp4}', flush=True)
    return

  print(f'[vid] === {label} ({ckpt_path}) ===', flush=True)
  ckpt = ppo_learner.load_checkpoint(ckpt_path)
  # Online only.
  nf_params = ckpt['q_params']
  iteration = int(ckpt.get('iteration') or 0)
  run_dir = _run_dir_from_ckpt(ckpt_path)
  goal_mean, goal_std = _load_goal_stats(run_dir, iteration, goal_dim)
  goal_std = np.maximum(goal_std, arch['nf_goal_std_min']).astype(np.float32)

  reward_fn = _nf.make_nf_reward_fn(nf_nets, obs_dim=int(ctx.obs_dim))
  print(f'[vid] policy_iter={iteration} ep_len={ep_len} '
        f'reward_src=q_params(online) state_only={arch["nf_state_only"]}',
        flush=True)

  policy = _make_policy_fn(
      networks, ckpt['policy_params'], stochastic=False,
      filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes,
      normalize_obs=False, obs_dim=ctx.obs_dim,
      start_index=ctx.start_index, end_index=ctx.end_index)
  run = _compile_rollout_and_states(
      policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
      ctx.filter_policy_obs)

  key = jax.random.PRNGKey(args.seed)
  key, warm_key = jax.random.split(key)
  print('[vid] warming compile...', flush=True)
  _ = jax.block_until_ready(run(warm_key))
  print('[vid] compile done', flush=True)

  gmean = jnp.asarray(goal_mean)
  gstd = jnp.asarray(goal_std)
  best = None
  for attempt in range(int(args.max_tries)):
    key, roll_key = jax.random.split(key)
    traj, states = run(roll_key)
    packed = np.asarray(traj['packed'], dtype=np.float32)
    actions = np.asarray(traj['action'], dtype=np.float32)
    succ = np.asarray(traj['success'], dtype=np.float32)
    rewards = np.asarray(
        reward_fn(nf_params, jnp.asarray(packed), jnp.asarray(actions),
                  gmean, gstd), dtype=np.float32)
    reached = bool(np.any(succ >= 0.5))
    print(f'[vid] try={attempt} success={reached} '
          f'logp_sum={rewards.sum():.2f} first_succ='
          f'{int(np.argmax(succ >= 0.5)) if reached else -1}', flush=True)
    # State select after each env step (PD info / obs last dim).
    sel = np.asarray(states.info['select_action'], dtype=np.float32)
    if sel.ndim > 1:
      sel = sel.reshape(sel.shape[0], -1)[:, 0]
    # packed state matches the reward input (policy obs before step).
    cur = dict(rewards=rewards, success=succ, states=states, select=sel,
               packed=packed, actions=actions, attempt=attempt)
    if reached:
      best = cur
      break
    if best is None or rewards.sum() > best['rewards'].sum():
      best = cur

  if best is None:
    raise RuntimeError('no trajectory collected')
  if (not np.any(best['success'] >= 0.5)) and (not args.allow_no_success):
    raise RuntimeError('no successful trajectory found '
                       '(pass --allow_no_success)')

  rewards = best['rewards']
  success = best['success']
  states = best['states']
  select = np.asarray(best['select'], dtype=np.float32)
  cube = select_action_to_cube(select, num_cubes)
  packed = np.asarray(best['packed'], dtype=np.float32)
  actions = np.asarray(best['actions'], dtype=np.float32)
  # Same xyz layout as PD policy obs / NF state_only input.
  cube_pos = packed[:, :3 * num_cubes].reshape(len(packed), num_cubes, 3)
  step_delta = cube_step_deltas_from_pos(cube_pos)
  show_grad = bool(args.show_logp_grad)
  grad_s = np.zeros(len(packed), dtype=np.float32)
  grad_a = np.zeros(len(packed), dtype=np.float32)
  if show_grad:
    grad_fn = make_nf_logp_grad_norm_fn(nf_nets, obs_dim=int(ctx.obs_dim))
    gs, ga = grad_fn(
        nf_params, jnp.asarray(packed), jnp.asarray(actions), gmean, gstd)
    grad_s = np.asarray(gs, dtype=np.float32)
    grad_a = np.asarray(ga, dtype=np.float32)
    print(f'[vid] ||∇_s logp|| range=[{grad_s.min():.4g},{grad_s.max():.4g}] '
          f'||∇_a logp|| range=[{grad_a.min():.4g},{grad_a.max():.4g}]',
          flush=True)
  T = len(rewards)
  first_succ = (int(np.argmax(success >= 0.5))
                if np.any(success >= 0.5) else -1)
  print(f'[vid] using attempt={best["attempt"]} first_success_t={first_succ} '
        f'logp_sum={rewards.sum():.3f}', flush=True)
  print(f'[vid] select range=[{select.min():+.3f},{select.max():+.3f}] '
        f'cubes_used={sorted(set(int(x) for x in cube))}', flush=True)
  print(f'[vid] pos Δ after t=45: sum={step_delta[45:].sum():.5f} '
        f'max_step={step_delta[45:].max():.5f} m', flush=True)

  csv_cols = [
      np.arange(T), rewards, success,
      np.full(T, np.nan, dtype=np.float32),
      select, cube.astype(np.float32),
      step_delta.sum(axis=1).astype(np.float32),
      grad_s.astype(np.float32),
      grad_a.astype(np.float32),
  ]
  header = ('t,reward_logp_online,success,dist,select_action,select_cube,'
            'pos_delta_sum,grad_s_norm,grad_a_norm')
  for c in range(num_cubes):
    csv_cols.append(step_delta[:, c].astype(np.float32))
    header += f',pos_delta_c{c}'
  np.savetxt(
      csv_path, np.stack(csv_cols, axis=1), delimiter=',',
      header=header, comments='')

  print('[vid] rendering frames...', flush=True)
  renders = []
  for i in range(T):
    renders.append(np.asarray(env.render_from_info(
        np.asarray(states.data.qpos[i][0]),
        np.asarray(states.data.qvel[i][0]),
        np.asarray(states.info['target_mocap_pos'][i][0]),
        np.asarray(states.info['target_mocap_quat'][i][0]),
    )))
  width = int(renders[0].shape[1])
  if width % 2:
    width -= 1

  if args.title:
    title = args.title
  elif arch['nf_state_only']:
    title = rf'online NF  $r=\log p(g\mid s)$  ·  {tag}'
  else:
    title = rf'online NF  $r=\log p(g\mid s,a)$  ·  {tag}'
  ylabel = r'$\log p_{\mathrm{NF}}$'

  rewards_norm = rewards / (float(np.std(rewards)) + 1e-8)
  print('[vid] composing reward overlay...', flush=True)
  frames = []
  for t in range(T):
    strip = _render_reward_strip(
        rewards, success, t, width=width, height=220,
        title=title, ylabel=ylabel, line_color='#5ec8ff', ylim=None)
    extra = []
    select_badge = None
    if args.normalize_reward:
      extra.append(_render_reward_strip(
          rewards_norm, success, t, width=width, height=180,
          title=rf'normalised reward  $r/\sigma_r$  ·  {tag}',
          ylabel=r'$r/\sigma_r$',
          line_color='#ffb347'))
    if show_grad:
      if arch['nf_state_only']:
        grad_title = (
            r'$\Vert\nabla_s\log p_{\mathrm{NF}}(g\mid s)\Vert$  /  '
            r'$\Vert\nabla_a\log p_{\mathrm{NF}}\Vert$ (unused)'
            rf'  ·  {tag}')
      else:
        grad_title = (
            r'$\Vert\nabla_s\log p_{\mathrm{NF}}(g\mid s,a)\Vert$  /  '
            r'$\Vert\nabla_a\log p_{\mathrm{NF}}(g\mid s,a)\Vert$'
            rf'  ·  {tag}')
      extra.append(_render_grad_norm_strip(
          grad_s, grad_a, success, t, width=width, height=220,
          title=grad_title,
          ylabel_s=r'$\Vert\nabla_s\log p_{\mathrm{NF}}\Vert$',
          ylabel_a=r'$\Vert\nabla_a\log p_{\mathrm{NF}}\Vert$'))
    if args.show_select:
      extra.append(_render_select_strip(
          select, cube, t, width=width, num_cubes=num_cubes, height=180,
          title=rf'state select_action → cube  ·  {tag}'))
      select_badge = (float(select[t]), int(cube[t]))
    if args.show_pos_delta:
      extra.append(_render_pos_delta_strip(
          step_delta, t, width=width, selected_cube=cube, height=200,
          title=rf'per-cube $\Vert\Delta xyz\Vert$ in $s$  ·  {tag}'))
    composed = _compose_frame(
        strip, renders[t], float(rewards[t]), bool(success[t] >= 0.5),
        extra_strips=extra or None, select_badge=select_badge)
    h, w = composed.shape[:2]
    if h % 2 or w % 2:
      composed = composed[: h - (h % 2), : w - (w % 2)]
    frames.append(composed)
    if (t + 1) % 10 == 0 or t == T - 1:
      print(f'[vid]   framed {t + 1}/{T}', flush=True)
  frames.extend([frames[-1]] * HOLD_LAST)
  _write_mp4(frames, out_mp4, args.fps)
  still_t = min(first_succ + 2, T - 1) if first_succ >= 0 else T // 2
  Image.fromarray(frames[still_t]).save(still_path)
  print(f'[vid] wrote {out_mp4} ({os.path.getsize(out_mp4) / 1e6:.2f} MB)',
        flush=True)


def main():
  args = _parse_args()
  if not args.checkpoint and not args.checkpoint_dir:
    raise SystemExit('need --checkpoint or --checkpoint_dir')
  out_dir = args.out_dir
  os.makedirs(out_dir, exist_ok=True)
  entries = _enumerate_ckpts(args.checkpoint, args.checkpoint_dir)
  if not entries:
    raise FileNotFoundError('no checkpoints found')
  print(f'[vid] jax={jax.default_backend()} devices={jax.devices()}')
  print(f'[vid] {len(entries)} checkpoint(s) → {out_dir}')

  first_path = entries[0][1]
  env_name = args.env
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, _ = parse_bb_env_id(env_id)
  ctx = _load_train_ctx(env_name, first_path)
  if not args.match_run_init:
    force_video_nopermute_norand(ctx, fixed_start_x=float(args.fixed_start_x))
    print(f'[vid] forcing nopermute + fixed_start_x={ctx.fixed_start_x}',
          flush=True)
  run_dir = _run_dir_from_ckpt(first_path)
  arch = _load_nf_arch(run_dir)
  networks, nf_nets, goal_dim = _build_networks(
      env_name, args.seed, ctx, arch)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)

  for label, path in entries:
    _render_one(
        args, label=label, ckpt_path=path, ctx=ctx, networks=networks,
        nf_nets=nf_nets, env=env, mocap_targets=mocap_targets, ep_len=ep_len,
        num_cubes=num_cubes, goal_dim=goal_dim, arch=arch, out_dir=out_dir)


if __name__ == '__main__':
  main()
