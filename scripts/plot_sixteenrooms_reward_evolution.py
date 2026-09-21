#!/usr/bin/env python3
"""Render SixteenRooms density-reward snapshots as PNGs and a GIF.

Each frame evaluates the deterministic policy action at the center of every
free maze cell.  The default ``fixed`` scale shares limits across all frames
from one run.  ``frame_robust`` instead uses each frame's free-cell 2nd and
98th percentiles while keeping the colorbar in actual reward units.  The four
newest completed training episodes saved in each snapshot are overlaid; paths
stop at the first state within the PointEnv success radius (1.0 for
point_SixteenRooms).
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from acme import specs
from PIL import Image

import contrastive
from contrastive import nf_density
from contrastive import ppo_learner
from contrastive import td3_density
from contrastive import utils as contrastive_utils
import env_utils
from ppo_contrastive import fixed_goal_dict


def _load_snapshots(snapshot_dir: str) -> list[dict]:
  paths = sorted(glob.glob(os.path.join(snapshot_dir, 'snapshot_iter_*.pkl')))
  if not paths:
    raise FileNotFoundError(f'No snapshot_iter_*.pkl in {snapshot_dir}')
  snapshots = []
  for path in paths:
    with open(path, 'rb') as fh:
      item = pickle.load(fh)
    if int(item.get('schema_version', -1)) != 1:
      raise ValueError(f'Unsupported snapshot schema in {path}')
    item['_path'] = path
    snapshots.append(item)
  return snapshots


def _build_models(snapshot: dict):
  cfg = SimpleNamespace(**snapshot['config'])
  env_name = str(cfg.env_name)
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=int(getattr(cfg, 'seed', 0)),
      fixed_start_end=fixed_goal_dict[env_name])
  env_spec = specs.make_environment_spec(probe_env)
  act_dim = int(np.prod(env_spec.actions.shape))
  del probe_env
  hidden = tuple(int(x) for x in cfg.hidden_layer_sizes)
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=int(cfg.repr_dim),
      repr_norm=bool(cfg.repr_norm),
      twin_q=bool(cfg.twin_q),
      use_image_obs=False,
      hidden_layer_sizes=hidden,
      actor_min_std=float(cfg.ppo_actor_min_std),
      state_only=bool(getattr(cfg, 'crl_state_only', False)),
  )
  mode = str(cfg.ppo_repr_mode).strip().lower()
  density = None
  if mode == 'nf':
    if bool(getattr(cfg, 'nf_state_only', False)):
      raise ValueError('Diagnostic requires NF r(s,a), but snapshot is state-only')
    density = nf_density.make_nf_density_networks(
        obs_dim=obs_dim, act_dim=act_dim, goal_dim=obs_dim,
        rep_size=int(cfg.nf_rep_size),
        num_blocks=int(cfg.nf_num_blocks),
        channels=int(cfg.nf_coupling_width),
        goal_enc_size=int(cfg.nf_goal_enc_size),
        sa_hidden=int(cfg.nf_sa_hidden),
        sa_num_layers=int(cfg.nf_sa_num_layers),
        state_only=False,
        scale_tanh=bool(cfg.nf_scale_tanh),
        scale_tanh_c=float(cfg.nf_scale_tanh_c),
    )
  elif mode == 'td3':
    if not bool(getattr(cfg, 'ppo_td3_log_reward', False)):
      raise ValueError(
          'TD3 diagnostic must train/plot log((1-gamma)*Q); '
          'snapshot has ppo_td3_log_reward=False')
    density = td3_density.make_td3_density_networks(
        obs_dim=obs_dim, act_dim=act_dim, goal_dim=obs_dim,
        hidden_layer_sizes=hidden, repr_dim=int(cfg.repr_dim),
        bilinear=bool(cfg.ppo_td3_bilinear),
        repr_norm=(bool(cfg.repr_norm) if cfg.ppo_td3_bilinear else False),
    )
  elif mode not in ('crl', 'tdinfonce', 'td_infonce'):
    raise ValueError(f'Unsupported estimator mode {mode!r}')
  return cfg, networks, density, obs_dim, mode


def _free_cell_centers(walls: np.ndarray):
  rows, cols = np.where(walls == 0)
  pos = np.stack([rows + 0.5, cols + 0.5], axis=-1).astype(np.float32)
  return rows, cols, pos


def _reward_field(snapshot: dict, models, positions: np.ndarray) -> np.ndarray:
  cfg, networks, density, obs_dim, mode = models
  params = snapshot['reward_repr_params']
  goal = np.asarray(snapshot['fixed_goal'], dtype=np.float32).reshape(-1)
  pos = jnp.asarray(positions, dtype=jnp.float32)
  goal_j = jnp.asarray(goal, dtype=jnp.float32)
  packed = jnp.concatenate(
      [pos, jnp.broadcast_to(goal_j[None], (pos.shape[0], goal.shape[0]))],
      axis=-1)
  rms = snapshot['obs_rms']
  mean = jnp.asarray(rms['mean'], dtype=jnp.float32)
  var = jnp.asarray(rms['var'], dtype=jnp.float32)
  policy_obs = ppo_learner._normalize_packed_obs(
      packed, mean, var, obs_dim=obs_dim,
      start_index=int(cfg.start_index),
      end_index=(obs_dim if int(cfg.end_index) == -1 else int(cfg.end_index)),
      clip=float(cfg.ppo_obs_norm_clip),
      enabled=bool(cfg.ppo_norm_obs))
  dist = networks.policy_network.apply(snapshot['policy_params'], policy_obs)
  action = networks.sample_eval(dist, jax.random.PRNGKey(0))

  if mode == 'nf':
    nf_goal = goal_j
    if bool(cfg.nf_normalize_goals):
      nf_goal = (
          nf_goal - jnp.asarray(snapshot['nf_goal_mean'], dtype=jnp.float32)
      ) / jnp.asarray(snapshot['nf_goal_std'], dtype=jnp.float32)
    goals = jnp.broadcast_to(nf_goal[None], (pos.shape[0], goal.shape[0]))
    values = nf_density.nf_log_prob(density, params, pos, action, goals)
  elif mode == 'td3':
    reward_fn = td3_density.make_td3_reward_fn(
        density, obs_dim=obs_dim, discount=float(cfg.discount),
        log_reward=True, start_index=int(cfg.start_index),
        end_index=int(cfg.end_index),
        normalize_obs=bool(cfg.ppo_norm_obs),
        obs_norm_clip=float(cfg.ppo_obs_norm_clip))
    values = reward_fn(params, packed, action, mean, var)
  else:
    reward_cfg = contrastive.ContrastiveConfig()
    for name, value in snapshot['config'].items():
      setattr(reward_cfg, name, value)
    reward_fn = ppo_learner.make_reward_fn(networks, reward_cfg)
    values = reward_fn(params, packed, action, mean, var)
  return np.asarray(values, dtype=np.float32)


def _reward_label(mode: str) -> str:
  if mode == 'nf':
    return 'NF  log p(g | s,a)'
  if mode == 'td3':
    return 'TD3  log((1-gamma) Q1(s,a,g))'
  if mode in ('tdinfonce', 'td_infonce'):
    return 'TD-InfoNCE  phi(s,a) dot psi(g)'
  return 'CRL  phi(s,a) dot psi(g)'


def _render_frame(snapshot: dict, walls: np.ndarray, grid: np.ndarray,
                  vmin: float, vmax: float, label: str, output: str,
                  scale_note: str = '', colorbar_extend: str = 'neither'):
  h, w = walls.shape
  fig, ax = plt.subplots(figsize=(7.4, 6.8))
  ax.imshow(
      np.ma.masked_where(walls != 0, grid), origin='upper',
      extent=[0, w, h, 0], cmap='magma', vmin=vmin, vmax=vmax,
      interpolation='nearest')
  ax.imshow(
      np.ma.masked_where(walls == 0, walls), origin='upper',
      extent=[0, w, h, 0], cmap='gray_r', vmin=0, vmax=1,
      interpolation='nearest')
  colors = plt.get_cmap('tab10')
  paths = snapshot.get('recent_xy_paths', [])[-4:]
  for idx, path in enumerate(paths):
    xy = np.asarray(path, dtype=np.float32)
    if xy.shape[0] == 0:
      continue
    color = colors(idx)
    ax.plot(xy[:, 1], xy[:, 0], color=color, lw=1.7, alpha=0.9,
            label=f'train trajectory {idx + 1}')
    ax.scatter(xy[:, 1], xy[:, 0], color=color, s=4, alpha=0.35)
  goal = np.asarray(snapshot['fixed_goal'], dtype=np.float32)
  ax.scatter([0], [0], marker='o', s=70, c='lime',
             edgecolors='black', label='start [0,0]', zorder=5)
  ax.scatter([goal[1]], [goal[0]], marker='*', s=150, c='cyan',
             edgecolors='black', label='goal [20,20]', zorder=5)
  image = ax.images[0]
  cbar = fig.colorbar(
      image, ax=ax, fraction=0.046, pad=0.04, extend=colorbar_extend)
  cbar.set_label(label)
  scale_note = scale_note or f'fixed scale [{vmin:.3g}, {vmax:.3g}]'
  ax.set(
      xlim=(0, w), ylim=(h, 0), aspect='equal', xlabel='maze column',
      ylabel='maze row',
      title=(f'{label}\niteration {snapshot["iteration"]:,}  |  '
             f'env steps {snapshot["global_step"]:,}\n{scale_note}'))
  if paths:
    ax.legend(loc='upper left', fontsize=7, framealpha=0.8)
  fig.tight_layout()
  os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
  fig.savefig(output, dpi=130)
  plt.close(fig)


def _frame_robust_stats(field: np.ndarray):
  """Return p02/median/p98 and nondegenerate p02-p98 display limits."""
  finite = np.asarray(field, dtype=np.float32)
  finite = finite[np.isfinite(finite)]
  if finite.size == 0:
    raise ValueError('Reward field has no finite free-cell values')
  p02, median, p98 = np.percentile(finite, [2.0, 50.0, 98.0])
  vmin, vmax = float(p02), float(p98)
  if not vmax > vmin:
    actual_min, actual_max = float(finite.min()), float(finite.max())
    if actual_max > actual_min:
      vmin, vmax = actual_min, actual_max
    else:
      vmax = vmin + max(1e-6, abs(vmin) * 1e-6)
  return float(p02), float(median), float(p98), vmin, vmax


def render(snapshot_dir: str, output_dir: str, fps: int = 6,
           scale_mode: str = 'fixed', make_gif: bool = True):
  snapshots = _load_snapshots(snapshot_dir)
  models = _build_models(snapshots[0])
  cfg, _, _, _, mode = models
  gym_env, _, _ = env_utils.load(
      str(cfg.env_name), fixed_start_end=fixed_goal_dict[str(cfg.env_name)],
      seed=0)
  walls = np.asarray(gym_env._walls, dtype=np.int8)
  rows, cols, positions = _free_cell_centers(walls)
  fields = [_reward_field(s, models, positions) for s in snapshots]
  finite = np.concatenate([f[np.isfinite(f)] for f in fields])
  fixed_vmin, fixed_vmax = float(finite.min()), float(finite.max())
  if not fixed_vmax > fixed_vmin:
    fixed_vmax = fixed_vmin + 1e-6
  label = _reward_label(mode)
  os.makedirs(output_dir, exist_ok=True)
  pngs = []
  for snapshot, field in zip(snapshots, fields):
    if scale_mode == 'frame_robust':
      p02, median, p98, vmin, vmax = _frame_robust_stats(field)
      scale_note = (
          f'per-frame free cells: p02={p02:.4g}, '
          f'median={median:.4g}, p98={p98:.4g}')
      colorbar_extend = 'both'
    else:
      vmin, vmax = fixed_vmin, fixed_vmax
      scale_note = f'fixed scale [{vmin:.3g}, {vmax:.3g}]'
      colorbar_extend = 'neither'
    grid = np.full(walls.shape, np.nan, dtype=np.float32)
    grid[rows, cols] = field
    path = os.path.join(
        output_dir, f'reward_iter_{int(snapshot["iteration"]):07d}.png')
    _render_frame(
        snapshot, walls, grid, vmin, vmax, label, path,
        scale_note=scale_note, colorbar_extend=colorbar_extend)
    pngs.append(path)
  if scale_mode == 'frame_robust':
    scale_summary = 'per-frame free-cell p02/p98 scale'
  else:
    scale_summary = (
        f'estimator-fixed scale=[{fixed_vmin:.6g}, {fixed_vmax:.6g}]')
  if make_gif:
    gif_path = os.path.join(output_dir, f'{mode}_reward_evolution.gif')
    frames = [Image.open(path).convert('RGB') for path in pngs]
    frames[0].save(
        gif_path, save_all=True, append_images=frames[1:], loop=0,
        duration=max(1, int(round(1000 / max(1, fps)))))
    for frame in frames:
      frame.close()
    output_summary = f'{len(pngs)} PNGs and {gif_path}'
  else:
    output_summary = f'{len(pngs)} PNGs (GIF disabled)'
  print(f'[plot] wrote {output_summary}; {scale_summary}', flush=True)


def synthetic_test(output_dir: str):
  gym_env, _, _ = env_utils.load(
      'point_SixteenRooms',
      fixed_start_end=fixed_goal_dict['point_SixteenRooms'], seed=0)
  walls = np.asarray(gym_env._walls, dtype=np.int8)
  rows, cols, _ = _free_cell_centers(walls)
  grid = np.full(walls.shape, np.nan, dtype=np.float32)
  grid[rows, cols] = rows.astype(np.float32) + cols.astype(np.float32)
  path = np.array([[0, 0], [0, 4], [2, 8], [8, 12], [14, 18], [20, 20]],
                  dtype=np.float32)
  snap = {
      'iteration': 5, 'global_step': 10_240,
      'fixed_goal': np.array([20, 20], dtype=np.float32),
      'recent_xy_paths': [path],
  }
  os.makedirs(output_dir, exist_ok=True)
  png = os.path.join(output_dir, 'synthetic_sixteenrooms.png')
  _render_frame(
      snap, walls, grid, float(np.nanmin(grid)), float(np.nanmax(grid)),
      'synthetic reward', png)
  with Image.open(png) as image:
    image.convert('RGB').save(
        os.path.join(output_dir, 'synthetic_sixteenrooms.gif'),
        save_all=True, duration=200)
  print(f'[plot] synthetic smoke test wrote {png}', flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--snapshot-dir')
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--fps', type=int, default=6)
  parser.add_argument(
      '--no-gif', action='store_true',
      help='Write the PNG sequence only; default behavior still writes a GIF')
  parser.add_argument(
      '--scale-mode', choices=('fixed', 'frame_robust'), default='fixed',
      help='fixed: one min/max across time (default); frame_robust: each '
           'frame uses free-cell p02/p98 limits in actual reward units')
  parser.add_argument('--synthetic-test', action='store_true')
  args = parser.parse_args()
  if args.synthetic_test:
    synthetic_test(args.output_dir)
  else:
    if not args.snapshot_dir:
      parser.error('--snapshot-dir is required unless --synthetic-test')
    render(
        args.snapshot_dir, args.output_dir, fps=args.fps,
        scale_mode=args.scale_mode, make_gif=not args.no_gif)


if __name__ == '__main__':
  main()
