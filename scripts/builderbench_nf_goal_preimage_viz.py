"""Goal-preimage state gallery for BuilderBench PPO runs (NF or CRL).

Idea
----
Sample trajectories from the stochastic policy at a checkpoint.  Treat the
visited (s, a) pairs as an empirical occupancy ρ_π.  Reweight each sample by
the goal-conditioned score at the hard goal g*:

    NF:   w(s, a) ∝ ρ_π(s, a) · p_NF(g* | s, a)
    CRL:  w(s, a) ∝ ρ_π(s, a) · φ(s, a) · ψ(g*)

After normalizing w over the collected samples, render the top-K
highest-weight states.

Examples::

  # NF
  python scripts/builderbench_nf_goal_preimage_viz.py --repr_mode=nf \\
      --run_dir=logs/ppo_builderbench_creative3_task1_e1024_pd_nf/ppo_builderbench_creative_3_task1_0 \\
      --output=figs/builderbench/creative3_task1_nf_goal_preimage/

  # CRL (τ=0.5)
  python scripts/builderbench_nf_goal_preimage_viz.py --repr_mode=crl \\
      --run_dir=logs/ppo_builderbench_creative3_task1_e1024_pd_tau0p5/ppo_builderbench_creative_3_task1_0 \\
      --output=figs/builderbench/creative3_task1_crl_goal_preimage/
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from PIL import Image, ImageDraw, ImageFont

import contrastive
from contrastive import nf_density as _nf
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from envs.builderbench_utils import (
    creative_cube_full_state_obs_dim,
    creative_cube_mj_episode_length,
    filter_pd_policy_state_obs,
    is_builderbench_creative_env,
    parse_bb_env_id,
    pd_policy_state_obs_dim,
    set_task_mocap_pos,
    sgcrl_env_name_to_bb_env_id,
)
from ppo_contrastive import fixed_goal_for_env, ppo_env_defaults_for_env

from builderbench.constants import _MJX_PARAMS
from builderbench.creative_cube import CreativeCube, default_config
from utils.wrapper import AutoResetWrapper, EpisodeWrapper, PDWrapper, VmapWrapper


# Default stages for ~200M-step cube3 runs (ckpt every 150 iters; iter 0 = init).
_DEFAULT_STAGES: Tuple[Tuple[str, int], ...] = (
    ('very_early', 0),     # init / 0 steps
    ('early', 600),        # ~30.7M
    ('mid', 1950),         # ~99.9M
    ('upper_mid', 3000),   # ~153.6M
    ('late', 3900),        # ~199.7M
)


@dataclass
class _TrainCtx:
  use_pd: bool
  pd_duration: int
  filter_policy_obs: bool
  episode_length: int
  obs_dim: int
  goal_dim: int
  hidden_layer_sizes: Tuple[int, ...]
  fixed_target_goal: Optional[np.ndarray]
  actor_min_std: float
  start_index: int
  end_index: int
  permute_start_boxes: bool
  nf_rep_size: int
  nf_num_blocks: int
  nf_coupling_width: int
  nf_goal_enc_size: int
  nf_goal_std_min: float
  nf_state_only: bool
  act_dim: int


def _run_config_path(run_dir: str) -> str:
  path = os.path.join(run_dir, 'run_config.json')
  if not os.path.isfile(path):
    raise FileNotFoundError(f'missing run_config.json under {run_dir}')
  return path


def _load_train_ctx(env_name: str, run_dir: str) -> Tuple[_TrainCtx, Dict[str, Any]]:
  num_cubes, task_index = parse_bb_env_id(sgcrl_env_name_to_bb_env_id(env_name))
  mj_ep_len = creative_cube_mj_episode_length(num_cubes, task_index)
  full_obs_dim = creative_cube_full_state_obs_dim(num_cubes)
  pd_obs_dim = pd_policy_state_obs_dim(num_cubes)

  with open(_run_config_path(run_dir), 'r', encoding='utf-8') as fh:
    run_cfg: Dict[str, Any] = json.load(fh)
  flags = run_cfg.get('flags', {})
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
  fixed_goal = None if fixed is None else np.asarray(fixed, dtype=np.float32)
  if fixed_goal is None:
    fixed_goal = fixed_goal_for_env(env_name)
  permute_start_boxes = bool(flags.get('builderbench_permute_start_boxes', True))

  if use_pd:
    macro_ep_len = mj_ep_len // pd_duration
    filter_policy = (obs_dim == pd_obs_dim)
  else:
    macro_ep_len = mj_ep_len
    filter_policy = False

  goal_dim = int(end_index - start_index) if end_index != -1 else int(obs_dim)
  # PD action is cube xyz delta + select → 4? Actually CreativeCube PD uses
  # act_dim from env spec; infer later from networks probe. Placeholder 5
  # matches cube3 PD (3 pos + 1 unused? + select) — overwritten after probe.
  ctx = _TrainCtx(
      use_pd=use_pd,
      pd_duration=pd_duration,
      filter_policy_obs=filter_policy,
      episode_length=int(macro_ep_len),
      obs_dim=int(obs_dim),
      goal_dim=int(goal_dim),
      hidden_layer_sizes=hidden,
      fixed_target_goal=fixed_goal,
      actor_min_std=actor_min_std,
      start_index=start_index,
      end_index=end_index,
      permute_start_boxes=permute_start_boxes,
      nf_rep_size=int(resolved.get('nf_rep_size', 256)),
      nf_num_blocks=int(resolved.get('nf_num_blocks', 12)),
      nf_coupling_width=int(resolved.get('nf_coupling_width', 512)),
      nf_goal_enc_size=int(resolved.get('nf_goal_enc_size', 0)),
      nf_goal_std_min=float(resolved.get('nf_goal_std_min', 0.02)),
      nf_state_only=bool(resolved.get('nf_state_only', False)),
      act_dim=0,
  )
  return ctx, run_cfg


def _load_goal_stats(run_dir: str, iteration: int, goal_dim: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
  """Load nf/goal_mean_*, nf/goal_std_* from learner CSV at nearest iteration."""
  csv_path = os.path.join(run_dir, 'logs', 'learner', 'logs.csv')
  if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'missing learner CSV: {csv_path}')
  best = None
  best_dist = None
  with open(csv_path, 'r', encoding='utf-8') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
      ls = int(float(row['learner_steps']))
      dist = abs(ls - int(iteration))
      if best_dist is None or dist < best_dist:
        best_dist = dist
        best = row
  if best is None:
    raise RuntimeError(f'empty learner CSV: {csv_path}')
  mean = np.asarray(
      [float(best[f'nf/goal_mean_{i}']) for i in range(goal_dim)],
      dtype=np.float32)
  std = np.asarray(
      [float(best[f'nf/goal_std_{i}']) for i in range(goal_dim)],
      dtype=np.float32)
  print(f'[preimage] goal stats from learner_steps={best["learner_steps"]} '
        f'(requested iter={iteration}, |Δ|={best_dist})')
  return mean, std


def _build_networks(env_name: str, seed: int, ctx: _TrainCtx, repr_mode: str):
  env_kwargs: Dict[str, Any] = {}
  if ctx.use_pd:
    env_kwargs['builderbench_use_pd'] = True
    env_kwargs['builderbench_pd_duration'] = ctx.pd_duration
    env_kwargs['builderbench_pd_filter_policy_obs'] = ctx.filter_policy_obs
  env_kwargs['builderbench_permute_start_boxes'] = ctx.permute_start_boxes

  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name,
      ctx.start_index,
      ctx.end_index,
      seed=seed,
      fixed_start_end=ctx.fixed_target_goal,
      **env_kwargs,
  )
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
  )
  nf_nets = None
  if repr_mode == 'nf':
    nf_nets = _nf.make_nf_density_networks(
        obs_dim=int(ctx.obs_dim),
        act_dim=act_dim,
        goal_dim=int(ctx.goal_dim),
        hidden_layer_sizes=ctx.hidden_layer_sizes,
        rep_size=ctx.nf_rep_size,
        num_blocks=ctx.nf_num_blocks,
        channels=ctx.nf_coupling_width,
        goal_enc_size=ctx.nf_goal_enc_size,
        state_only=bool(ctx.nf_state_only),
    )
  return networks, nf_nets, act_dim, int(obs_dim)


def _make_bb_env(env_id: str, ctx: _TrainCtx):
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.episode_length = creative_cube_mj_episode_length(num_cubes, task_id)
  cfg.permute_start_boxes = bool(ctx.permute_start_boxes)
  cfg.impl = os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax')
  if env_id in _MJX_PARAMS:
    cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]

  base = CreativeCube(config=cfg)
  mocap_targets = base._task_mocap_targets

  if ctx.use_pd:
    inner = PDWrapper(base, duration=ctx.pd_duration)
    macro_ep_len = cfg.episode_length // ctx.pd_duration
  else:
    inner = base
    macro_ep_len = cfg.episode_length

  env = VmapWrapper(inner)
  env = EpisodeWrapper(env, episode_length=int(macro_ep_len), action_repeat=1)
  env = AutoResetWrapper(env)
  env.render_from_info = base.render_from_info  # type: ignore[attr-defined]
  return env, base, mocap_targets, int(macro_ep_len), num_cubes


def _maybe_fix_target(state, fixed_target_goal, mocap_targets, num_cubes: int):
  if fixed_target_goal is None:
    return state
  del num_cubes  # full scene count; masked goals may be shorter
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(int(fixed.shape[0] // 3), 3)
  info = dict(state.info)
  info['target_goal'] = jnp.broadcast_to(fixed, state.info['target_goal'].shape)
  info['target_mocap_pos'] = jnp.broadcast_to(
      fixed_pos, state.info['target_mocap_pos'].shape)
  mocap_pos = set_task_mocap_pos(
      state.data.mocap_pos, mocap_targets, fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


def _make_policy_fn(networks, policy_params, *, filter_policy_obs: bool,
                    num_cubes: int):
  @jax.jit
  def policy(obs, goals, key):
    if filter_policy_obs:
      obs = filter_pd_policy_state_obs(obs, num_cubes)
    packed = jnp.concatenate([obs, goals], axis=-1)
    dist = networks.policy_network.apply(policy_params, packed)
    action = networks.sample(dist, key)
    return action

  return policy


def _collect_trajectories(
    policy,
    env,
    key,
    *,
    num_trajs: int,
    episode_length: int,
    fixed_target_goal,
    mocap_targets,
    num_cubes: int,
    filter_policy_obs: bool,
    obs_dim: int,
):
  """Roll out stochastic trajs; return flat arrays of (s, a, render payloads)."""
  @jax.jit
  def _one_traj(key):
    env_key, key = jax.random.split(key)
    state = env.reset(jax.random.split(env_key, 1))
    state = _maybe_fix_target(
        state, fixed_target_goal, mocap_targets, num_cubes)

    def step(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      obs = state.obs
      goals = state.info['target_goal']
      action = policy(obs, goals, act_key)
      # State features matching training NF / policy obs.
      if filter_policy_obs:
        state_feat = filter_pd_policy_state_obs(obs, num_cubes)
      else:
        state_feat = obs[..., :obs_dim]
      payload = {
          'state': state_feat[0],
          'action': action[0],
          'qpos': state.data.qpos[0],
          'qvel': state.data.qvel[0],
          'mocap_pos': state.info['target_mocap_pos'][0],
          'mocap_quat': state.info['target_mocap_quat'][0],
      }
      next_state = env.step(state, action)
      next_state = _maybe_fix_target(
          next_state, fixed_target_goal, mocap_targets, num_cubes)
      return (next_state, key), payload

    _, traj = jax.lax.scan(step, (state, key), (), length=episode_length)
    return traj

  states, actions, qpos, qvel, mocap_pos, mocap_quat = [], [], [], [], [], []
  for i in range(num_trajs):
    key, sub = jax.random.split(key)
    traj = _one_traj(sub)
    # Bring to host once per traj.
    states.append(np.asarray(traj['state']))
    actions.append(np.asarray(traj['action']))
    qpos.append(np.asarray(traj['qpos']))
    qvel.append(np.asarray(traj['qvel']))
    mocap_pos.append(np.asarray(traj['mocap_pos']))
    mocap_quat.append(np.asarray(traj['mocap_quat']))
    print(f'[preimage]   traj {i + 1}/{num_trajs} done', flush=True)

  return {
      'state': np.concatenate(states, axis=0).astype(np.float32),
      'action': np.concatenate(actions, axis=0).astype(np.float32),
      'qpos': np.concatenate(qpos, axis=0),
      'qvel': np.concatenate(qvel, axis=0),
      'mocap_pos': np.concatenate(mocap_pos, axis=0),
      'mocap_quat': np.concatenate(mocap_quat, axis=0),
  }


def _normalize_weights(raw_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """w ∝ (1/N) · score, with score shifted to be non-negative for stability."""
  raw = np.asarray(raw_scores, dtype=np.float64)
  # Softmax-style shift so large dynamic range does not overflow.
  shifted = raw - np.max(raw)
  score = np.exp(shifted)
  n = float(len(raw))
  unnorm = (1.0 / n) * score
  weights = unnorm / np.sum(unnorm)
  return score.astype(np.float32), weights.astype(np.float32)


def _score_preimage_nf(
    nf_nets,
    nf_params,
    states: np.ndarray,
    actions: np.ndarray,
    hard_goal: np.ndarray,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return (log_p, score, weights) with score ∝ p_NF(g*|s,a)."""
  gm = jnp.asarray(goal_mean, dtype=jnp.float32)
  gs = jnp.asarray(goal_std, dtype=jnp.float32)
  g = (jnp.asarray(hard_goal, dtype=jnp.float32) - gm) / (gs + 1e-8)

  @jax.jit
  def _batch_logp(s, a):
    g_b = jnp.broadcast_to(g, (s.shape[0], g.shape[0]))
    return _nf.nf_log_prob(nf_nets, nf_params, s, a, g_b)

  log_p = np.asarray(_batch_logp(
      jnp.asarray(states), jnp.asarray(actions)), dtype=np.float64)
  score, weights = _normalize_weights(log_p)
  return log_p.astype(np.float32), score, weights


def _score_preimage_crl(
    networks,
    q_params,
    states: np.ndarray,
    actions: np.ndarray,
    hard_goal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return (phi_dot_psi, score, weights) with score ∝ φ(s,a)·ψ(g*)."""
  g = jnp.asarray(hard_goal, dtype=jnp.float32)

  @jax.jit
  def _batch_dot(s, a):
    g_b = jnp.broadcast_to(g, (s.shape[0], g.shape[0]))
    obs = jnp.concatenate([s, g_b], axis=-1)
    _, phi, psi = networks.q_network.apply(q_params, obs, a)
    return jnp.sum(phi * psi, axis=-1)

  dots = np.asarray(_batch_dot(
      jnp.asarray(states), jnp.asarray(actions)), dtype=np.float64)
  score, weights = _normalize_weights(dots)
  return dots.astype(np.float32), score, weights


def _render_state(env, qpos, qvel, mocap_pos, mocap_quat,
                  height: int = 480, width: int = 640) -> np.ndarray:
  return np.asarray(env.render_from_info(
      np.asarray(qpos),
      np.asarray(qvel),
      np.asarray(mocap_pos),
      np.asarray(mocap_quat),
      height=height,
      width=width,
  ), dtype=np.uint8)


def _annotate(img: np.ndarray, lines: Sequence[str]) -> np.ndarray:
  im = Image.fromarray(img)
  draw = ImageDraw.Draw(im)
  try:
    font = ImageFont.truetype(
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', 16)
  except Exception:
    font = ImageFont.load_default()
  y = 8
  for line in lines:
    # shadow then foreground for readability on bright scenes
    draw.text((9, y + 1), line, fill=(0, 0, 0), font=font)
    draw.text((8, y), line, fill=(255, 255, 255), font=font)
    y += 18
  return np.asarray(im)


def _make_montage(
    images: List[np.ndarray],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: str,
    cell_w: int = 320,
    cell_h: int = 240,
    label_h: int = 28,
    row_label_w: int = 110,
):
  n_rows = len(row_labels)
  n_cols = len(col_labels)
  assert len(images) == n_rows * n_cols
  canvas_w = row_label_w + n_cols * cell_w
  canvas_h = label_h + n_rows * (cell_h + label_h)
  canvas = Image.new('RGB', (canvas_w, canvas_h), color=(20, 20, 24))
  draw = ImageDraw.Draw(canvas)
  try:
    font = ImageFont.truetype(
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', 14)
    font_sm = ImageFont.truetype(
        '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf', 12)
  except Exception:
    font = ImageFont.load_default()
    font_sm = font

  for c, lab in enumerate(col_labels):
    x = row_label_w + c * cell_w + 8
    draw.text((x, 6), lab, fill=(230, 230, 230), font=font)

  for r, rlab in enumerate(row_labels):
    y0 = label_h + r * (cell_h + label_h)
    for li, line in enumerate(str(rlab).split('\n')):
      draw.text((8, y0 + cell_h // 2 - 14 + li * 16), line,
                fill=(230, 230, 230), font=font)
    for c in range(n_cols):
      idx = r * n_cols + c
      tile = Image.fromarray(images[idx]).resize(
          (cell_w, cell_h), Image.Resampling.BILINEAR)
      x = row_label_w + c * cell_w
      y = y0 + label_h
      # small caption strip
      draw.rectangle([x, y0, x + cell_w, y0 + label_h], fill=(35, 35, 42))
      draw.text((x + 6, y0 + 6), f'rank {c + 1}', fill=(200, 200, 210),
                font=font_sm)
      canvas.paste(tile, (x, y))

  os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
  canvas.save(out_path)
  print(f'[preimage] wrote montage {out_path}')


def _parse_stages(s: str) -> List[Tuple[str, int]]:
  """Parse 'very_early:150,early:600,...' or empty → defaults."""
  if not s.strip():
    return list(_DEFAULT_STAGES)
  out = []
  for part in s.split(','):
    name, it = part.strip().split(':')
    out.append((name.strip(), int(it)))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      '--run_dir',
      default=('logs/ppo_builderbench_creative3_task1_e1024_pd_nf/'
               'ppo_builderbench_creative_3_task1_0'))
  ap.add_argument('--env', default='builderbench_creative_3_task1')
  ap.add_argument('--repr_mode', choices=['nf', 'crl'], default='nf')
  ap.add_argument('--output', default='')
  ap.add_argument('--stages', default='',
                  help='name:iter,...  (default: very_early@0 / early / mid / '
                       'upper_mid / late)')
  ap.add_argument('--num_trajs', type=int, default=10)
  ap.add_argument('--top_k', type=int, default=5)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--width', type=int, default=640)
  ap.add_argument('--height', type=int, default=480)
  args = ap.parse_args()

  if not is_builderbench_creative_env(args.env):
    ap.error(f'unsupported env {args.env}')

  run_dir = os.path.abspath(args.run_dir)
  if args.output:
    out_dir = os.path.abspath(args.output)
  else:
    tag = 'nf' if args.repr_mode == 'nf' else 'crl'
    out_dir = os.path.abspath(
        f'figs/builderbench/creative3_task1_{tag}_goal_preimage/')
  os.makedirs(out_dir, exist_ok=True)
  stages = _parse_stages(args.stages)
  env_id = sgcrl_env_name_to_bb_env_id(args.env)
  score_name = 'logp' if args.repr_mode == 'nf' else 'phi·psi'

  print(f'[preimage] repr_mode={args.repr_mode}')
  print(f'[preimage] run_dir={run_dir}')
  print(f'[preimage] stages={stages}')
  ctx, run_cfg = _load_train_ctx(args.env, run_dir)
  resolved = run_cfg.get('resolved_config', {})
  train_mode = str(resolved.get('ppo_repr_mode', '')).lower()
  if args.repr_mode == 'nf' and train_mode not in ('', 'nf'):
    print(f'[preimage] WARNING: run_config ppo_repr_mode={train_mode!r} '
          f'but --repr_mode=nf')
  if args.repr_mode == 'crl' and train_mode not in ('', 'crl'):
    print(f'[preimage] WARNING: run_config ppo_repr_mode={train_mode!r} '
          f'but --repr_mode=crl')

  hard_goal = np.asarray(ctx.fixed_target_goal, dtype=np.float32).reshape(-1)
  assert hard_goal.shape[0] == ctx.goal_dim, (
      f'hard_goal dim {hard_goal.shape[0]} != goal_dim {ctx.goal_dim}')
  print(f'[preimage] hard_goal={hard_goal}')
  print(f'[preimage] ctx: obs_dim={ctx.obs_dim} goal_dim={ctx.goal_dim} '
        f'ep_len={ctx.episode_length} filter={ctx.filter_policy_obs}')

  print('[preimage] building networks / env...', flush=True)
  networks, nf_nets, act_dim, _ = _build_networks(
      args.env, args.seed, ctx, args.repr_mode)
  ctx.act_dim = act_dim
  video_env, _base, mocap_targets, episode_length, num_cubes = _make_bb_env(
      env_id, ctx)
  assert episode_length == ctx.episode_length

  key = jax.random.PRNGKey(args.seed)

  montage_imgs: List[np.ndarray] = []
  row_labels: List[str] = []
  meta_rows: List[Dict[str, Any]] = []

  for stage_name, iteration in stages:
    ckpt_path = os.path.join(
        run_dir, 'checkpoints', f'ckpt_iter_{iteration:07d}.pkl')
    if not os.path.isfile(ckpt_path):
      raise FileNotFoundError(ckpt_path)
    print(f'\n[preimage] === {stage_name}  iter={iteration} ===', flush=True)
    ckpt = ppo_learner.load_checkpoint(ckpt_path)
    policy_params = ckpt['policy_params']
    print(f'[preimage]   ckpt iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}')

    if args.repr_mode == 'crl':
      if 'q_params_ema' in ckpt and ckpt['q_params_ema'] is not None:
        q_params = ckpt['q_params_ema']
        print('[preimage]   using q_params_ema for φ·ψ')
      else:
        q_params = ckpt['q_params']
        print('[preimage]   using q_params for φ·ψ (no ema)')
      goal_mean = goal_std = None
    else:
      q_params = ckpt['q_params']
      goal_mean, goal_std = _load_goal_stats(run_dir, iteration, ctx.goal_dim)
      goal_std = np.maximum(goal_std, ctx.nf_goal_std_min).astype(np.float32)

    policy = _make_policy_fn(
        networks, policy_params,
        filter_policy_obs=ctx.filter_policy_obs, num_cubes=num_cubes)

    key, traj_key = jax.random.split(key)
    print(f'[preimage]   sampling {args.num_trajs} stochastic trajs...',
          flush=True)
    data = _collect_trajectories(
        policy, video_env, traj_key,
        num_trajs=args.num_trajs,
        episode_length=episode_length,
        fixed_target_goal=ctx.fixed_target_goal,
        mocap_targets=mocap_targets,
        num_cubes=num_cubes,
        filter_policy_obs=ctx.filter_policy_obs,
        obs_dim=ctx.obs_dim,
    )
    n = data['state'].shape[0]
    print(f'[preimage]   collected {n} (s,a) samples; scoring {score_name}...',
          flush=True)

    if args.repr_mode == 'nf':
      raw_score, score, weights = _score_preimage_nf(
          nf_nets, q_params, data['state'], data['action'],
          hard_goal, goal_mean, goal_std)
    else:
      raw_score, score, weights = _score_preimage_crl(
          networks, q_params, data['state'], data['action'], hard_goal)

    top_idx = np.argsort(-weights)[: args.top_k]
    stage_dir = os.path.join(out_dir, f'{stage_name}_iter{iteration:07d}')
    os.makedirs(stage_dir, exist_ok=True)

    save_kw = dict(
        raw_score=raw_score, score=score, weights=weights, top_idx=top_idx,
        state=data['state'], action=data['action'], hard_goal=hard_goal,
    )
    if goal_mean is not None:
      save_kw['goal_mean'] = goal_mean
      save_kw['goal_std'] = goal_std
    np.savez(os.path.join(stage_dir, 'preimage_weights.npz'), **save_kw)

    row_labels.append(f'{stage_name}\n@{iteration}')
    for rank, idx in enumerate(top_idx, start=1):
      img = _render_state(
          video_env,
          data['qpos'][idx], data['qvel'][idx],
          data['mocap_pos'][idx], data['mocap_quat'][idx],
          height=args.height, width=args.width,
      )
      step_in_pool = int(idx)
      traj_id = step_in_pool // episode_length
      t_in_traj = step_in_pool % episode_length
      lines = [
          f'{stage_name}  iter={iteration}  rank={rank}',
          f'traj={traj_id} t={t_in_traj}  {score_name}={raw_score[idx]:.2f}  '
          f'w={weights[idx]:.4f}',
      ]
      img_a = _annotate(img, lines)
      out_png = os.path.join(stage_dir, f'rank{rank:02d}_idx{idx:04d}.png')
      Image.fromarray(img_a).save(out_png)
      print(f'[preimage]   wrote {out_png}  {score_name}={raw_score[idx]:.3f} '
            f'w={weights[idx]:.5f}')
      montage_imgs.append(img_a)
      meta_rows.append({
          'stage': stage_name,
          'iteration': iteration,
          'rank': rank,
          'sample_idx': int(idx),
          'traj_id': int(traj_id),
          't': int(t_in_traj),
          'raw_score': float(raw_score[idx]),
          'weight': float(weights[idx]),
          'path': out_png,
      })

  col_labels = [f'rank {i}' for i in range(1, args.top_k + 1)]
  montage_path = os.path.join(out_dir, 'goal_preimage_montage.png')
  _make_montage(montage_imgs, row_labels, col_labels, montage_path)

  meta_path = os.path.join(out_dir, 'manifest.json')
  with open(meta_path, 'w', encoding='utf-8') as fh:
    json.dump({
        'run_dir': run_dir,
        'env': args.env,
        'repr_mode': args.repr_mode,
        'hard_goal': hard_goal.tolist(),
        'num_trajs': args.num_trajs,
        'top_k': args.top_k,
        'stages': [{'name': n, 'iteration': i} for n, i in stages],
        'images': meta_rows,
        'note': (
            'Weights are w ∝ (1/N) · score over stochastic-policy samples. '
            'NF score = p_NF(g*|s,a); CRL score = φ(s,a)·ψ(g*).'
        ),
    }, fh, indent=2)
  print(f'[preimage] wrote {meta_path}')
  print(f'[preimage] done — {len(meta_rows)} images under {out_dir}')


if __name__ == '__main__':
  main()
