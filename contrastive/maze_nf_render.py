"""In-train point-maze PNGs: 5 trajs + log p(g|s,a) heatmap.

Kept self-contained so the PPO training process does not re-import
``ppo_contrastive`` / ``scripts.ppo_rollout_maze`` (those pull absl flags
and would crash when ``ppo_contrastive.py`` is already ``__main__``).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import env_utils


def render_nf_maze_rollouts(
    *,
    env_name: str,
    networks: Any,
    policy_params: Any,
    nf_reward_fn: Callable,
    nf_params: Any,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
    out_path: str,
    title: str,
    seed: int,
    fixed_start_end: Optional[Any] = None,
    num_trajectories: int = 5,
    heatmap_subcells: int = 5,
    max_steps: Optional[int] = None,
) -> str:
  """Write one PNG: NF log p(g|s, a_π) over free cells + N policy trajs.

  Traj 0 is deterministic (policy mode). The rest are stochastic so the
  panel shows both the mode path and rollout variation.
  """
  gym_env, _, env_max_steps = env_utils.load(
      env_name, fixed_start_end=fixed_start_end, seed=int(seed))
  walls = np.asarray(gym_env.walls)
  obs0 = np.asarray(gym_env.reset(), dtype=np.float32)
  state_dim = int(obs0.shape[0] // 2)
  steps = int(env_max_steps if max_steps is None else max_steps)
  n_traj = max(1, int(num_trajectories))

  trjs = []
  metrics = []
  goal0 = None
  for i in range(n_traj):
    states, goal, reward_sum, success = _rollout_one(
        policy_params=policy_params,
        gym_env=gym_env,
        networks=networks,
        max_steps=steps,
        stochastic=(i > 0),
        seed=int(seed) + i,
        state_dim=state_dim,
    )
    if goal0 is None:
      goal0 = np.asarray(goal, dtype=np.float32)
    trjs.append(np.asarray(states[:, :2], dtype=np.float32))
    metrics.append((float(reward_sum), bool(success)))

  heatmap = _nf_logp_grid(
      walls=walls,
      goal=goal0,
      policy_params=policy_params,
      networks=networks,
      nf_reward_fn=nf_reward_fn,
      nf_params=nf_params,
      nf_goal_mean=np.asarray(nf_goal_mean, dtype=np.float32),
      nf_goal_std=np.asarray(nf_goal_std, dtype=np.float32),
      state_dim=state_dim,
      subcells=int(heatmap_subcells),
  )

  n_succ = int(sum(1 for _, s in metrics if s))
  avg_r = float(np.mean([r for r, _ in metrics]))
  h, w = walls.shape
  fig, ax = plt.subplots(
      1, 1, figsize=(7.4, 6.0 * h / max(w, 1)), layout='constrained')
  fig.suptitle(
      f'{title}\n'
      f'trajs={n_traj} (1 mode + {max(0, n_traj - 1)} stoch)  '
      f'success={n_succ}/{n_traj}  mean env ret={avg_r:.2f}',
      fontsize=11)
  _draw_maze_panel(
      ax, walls, trjs, goal0[:2], heatmap,
      r'$\log p(g \mid s, a_\pi)$')
  ax.set_title(r'$\log p(g \mid s, a_\pi(s,g))$  + policy rollouts', fontsize=11)
  out_dir = os.path.dirname(os.path.abspath(out_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fig.savefig(out_path, dpi=140)
  plt.close(fig)
  return out_path


def render_maze_policy_rollouts(
    *,
    env_name: str,
    act_fn: Callable,
    out_path: str,
    title: str,
    seed: int,
    fixed_start_end: Optional[Any] = None,
    num_trajectories: int = 5,
    max_steps: Optional[int] = None,
) -> str:
  """Write one PNG: N policy trajs on the maze walls (no density heatmap).

  ``act_fn(obs, stochastic, rng) -> action`` is the only policy hook, so
  RND / other trainers can render without an NF networks object.
  Traj 0 is deterministic; the rest are stochastic.
  """
  gym_env, _, env_max_steps = env_utils.load(
      env_name, fixed_start_end=fixed_start_end, seed=int(seed))
  walls = np.asarray(gym_env.walls)
  obs0 = np.asarray(gym_env.reset(), dtype=np.float32)
  state_dim = int(obs0.shape[0] // 2)
  steps = int(env_max_steps if max_steps is None else max_steps)
  n_traj = max(1, int(num_trajectories))

  trjs = []
  metrics = []
  goal0 = None
  for i in range(n_traj):
    states, goal, reward_sum, success = _rollout_one_act(
        act_fn=act_fn,
        gym_env=gym_env,
        max_steps=steps,
        stochastic=(i > 0),
        seed=int(seed) + i,
        state_dim=state_dim,
    )
    if goal0 is None:
      goal0 = np.asarray(goal, dtype=np.float32)
    trjs.append(np.asarray(states[:, :2], dtype=np.float32))
    metrics.append((float(reward_sum), bool(success)))

  n_succ = int(sum(1 for _, s in metrics if s))
  avg_r = float(np.mean([r for r, _ in metrics]))
  h, w = walls.shape
  fig, ax = plt.subplots(
      1, 1, figsize=(7.4, 6.0 * h / max(w, 1)), layout='constrained')
  fig.suptitle(
      f'{title}\n'
      f'trajs={n_traj} (1 mode + {max(0, n_traj - 1)} stoch)  '
      f'success={n_succ}/{n_traj}  mean env ret={avg_r:.2f}',
      fontsize=11)
  _draw_maze_panel(ax, walls, trjs, goal0[:2], None, '')
  ax.set_title('policy rollouts', fontsize=11)
  out_dir = os.path.dirname(os.path.abspath(out_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fig.savefig(out_path, dpi=140)
  plt.close(fig)
  return out_path


def _rollout_one_act(
    act_fn: Callable,
    gym_env,
    max_steps: int,
    stochastic: bool,
    seed: int,
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  goal = obs[state_dim:].copy()
  states = [obs[:state_dim].copy()]
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(int(seed))
  for _ in range(int(max_steps)):
    rng, k = jax.random.split(rng)
    action = np.asarray(act_fn(obs, stochastic, k), dtype=np.float32)
    obs_next, r, done, _ = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    states.append(obs[:state_dim].copy())
    total_reward += float(r)
    if float(r) > 0.0:
      success = True
    if done:
      break
  return (np.asarray(states, dtype=np.float32), goal,
          float(total_reward), bool(success))


def _rollout_one(
    policy_params,
    gym_env,
    networks,
    max_steps: int,
    stochastic: bool,
    seed: int,
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  goal = obs[state_dim:].copy()
  states = [obs[:state_dim].copy()]
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(int(seed))
  for _ in range(int(max_steps)):
    obs_j = jnp.asarray(obs[None], dtype=jnp.float32)
    dist = networks.policy_network.apply(policy_params, obs_j)
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = networks.sample(dist, k)
    else:
      action_j = networks.sample_eval(dist, jax.random.PRNGKey(0))
    action = np.asarray(action_j)[0].astype(np.float32)
    obs_next, r, done, _ = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    states.append(obs[:state_dim].copy())
    total_reward += float(r)
    if float(r) > 0.0:
      success = True
    if done:
      break
  return (np.asarray(states, dtype=np.float32), goal,
          float(total_reward), bool(success))


def _nf_logp_grid(
    *,
    walls: np.ndarray,
    goal: np.ndarray,
    policy_params: Any,
    networks: Any,
    nf_reward_fn: Callable,
    nf_params: Any,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
    state_dim: int,
    subcells: int,
) -> np.ndarray:
  h, w = walls.shape
  subcells = max(1, int(subcells))
  rows, cols = np.where(walls == 0)
  rr = (np.arange(subcells, dtype=np.float32) + 0.5) / float(subcells)
  cc = (np.arange(subcells, dtype=np.float32) + 0.5) / float(subcells)
  offsets = np.stack(np.meshgrid(rr, cc, indexing='ij'), axis=-1).reshape(-1, 2)
  base = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=-1)
  pos_2d = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
  if state_dim > 2:
    pos = np.concatenate(
        [pos_2d, np.zeros((len(pos_2d), state_dim - 2), dtype=np.float32)],
        axis=-1)
  else:
    pos = pos_2d

  g = jnp.asarray(goal, dtype=jnp.float32)
  pos_j = jnp.asarray(pos, dtype=jnp.float32)
  gmean = jnp.asarray(nf_goal_mean, dtype=jnp.float32)
  gstd = jnp.asarray(nf_goal_std, dtype=jnp.float32)

  @jax.jit
  def _eval(policy_p, nf_p, pos_batch, goal_vec):
    g_b = jnp.broadcast_to(goal_vec[None, :], (pos_batch.shape[0], state_dim))
    obs = jnp.concatenate([pos_batch, g_b], axis=-1)
    dist = networks.policy_network.apply(policy_p, obs)
    act = networks.sample_eval(dist, jax.random.PRNGKey(0))
    return nf_reward_fn(nf_p, obs, act, gmean, gstd)

  values = np.asarray(
      _eval(policy_params, nf_params, pos_j, g), dtype=np.float32)
  hh, ww = h * subcells, w * subcells
  grid = np.full((hh, ww), np.nan, dtype=np.float32)
  i0 = np.repeat(rows * subcells, subcells * subcells)
  j0 = np.repeat(cols * subcells, subcells * subcells)
  di = np.tile(np.repeat(np.arange(subcells), subcells), rows.shape[0])
  dj = np.tile(np.tile(np.arange(subcells), subcells), rows.shape[0])
  grid[i0 + di, j0 + dj] = values
  return grid


def _draw_maze_panel(ax, walls, trajectories, goal, heatmap, heat_label):
  h, w = walls.shape
  fig = ax.figure
  ax.imshow(
      walls, cmap='gray_r', origin='upper', extent=[0, w, h, 0],
      interpolation='nearest', alpha=0.85)
  divider = make_axes_locatable(ax)
  if heatmap is not None:
    hm = ax.imshow(
        np.ma.masked_invalid(heatmap), origin='upper',
        extent=[0, w, h, 0], interpolation='nearest', alpha=0.5, cmap='magma')
    cax = divider.append_axes('right', size='4%', pad=0.12)
    cbar = fig.colorbar(hm, cax=cax)
    cbar.ax.set_ylabel(heat_label, rotation=270, labelpad=12)
  cmap_traj = plt.get_cmap('tab10')
  for t_i, st in enumerate(trajectories):
    xs, ys = st[:, 1], st[:, 0]
    color = cmap_traj(t_i % 10)
    label = f'traj {t_i}' + (' (mode)' if t_i == 0 else ' (stoch)')
    ax.plot(xs, ys, '-', color=color, linewidth=1.8, alpha=0.9, label=label)
    ax.plot(xs, ys, 'o', color=color, markersize=1.8, alpha=0.5)
    ax.plot(xs[0], ys[0], marker='o', color='lime', markersize=9,
            markeredgecolor='black', linewidth=0)
    ax.plot(xs[-1], ys[-1], marker='X', color='orange', markersize=9,
            markeredgecolor='black', linewidth=0)
  ax.plot(goal[1], goal[0], marker='*', color='red', markersize=18,
          markeredgecolor='black', linewidth=0, label='goal')
  ax.set_xlim(0, w)
  ax.set_ylim(h, 0)
  ax.set_aspect('equal')
  ax.set_xlabel('col  (x)')
  ax.set_ylabel('row  (y, inverted)')
  ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
  ax.grid(True, color='gray', linewidth=0.3, alpha=0.3)


def render_maze_occupancy_preimage(
    *,
    env_name: str,
    networks: Any,
    policy_params: Any,
    nf_reward_fn: Callable,
    nf_params: Any,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
    out_path: str,
    title: str,
    seed: int = 0,
    heatmap_subcells: int = 8,
    fixed_start_end: Optional[Any] = None,
    num_trajectories: int = 5,
    obs: Optional[np.ndarray] = None,
    actions: Optional[np.ndarray] = None,
) -> str:
  """Full-grid NF density heatmaps + policy traj lines.

  Left:  p_NF(s | s0, a0) with a0 = π(s0, g).
  Right: that density times p(g | s, a_π(s)), renormalized.

  Rollouts are only used to draw where the agent went. ``obs`` / ``actions``
  are ignored (kept so older call sites do not break).
  """
  del obs, actions
  gym_env, _, env_max_steps = env_utils.load(
      env_name, fixed_start_end=fixed_start_end, seed=int(seed))
  walls = np.asarray(gym_env.walls)
  obs0 = np.asarray(gym_env.reset(), dtype=np.float32)
  state_dim = int(obs0.shape[0] // 2)
  s0 = obs0[:state_dim].copy()
  goal = obs0[state_dim:].copy()
  n_traj = max(1, int(num_trajectories))

  trjs = []
  metrics = []
  for i in range(n_traj):
    states, _, reward_sum, success = _rollout_one(
        policy_params=policy_params,
        gym_env=gym_env,
        networks=networks,
        max_steps=int(env_max_steps),
        stochastic=(i > 0),
        seed=int(seed) + i,
        state_dim=state_dim,
    )
    trjs.append(np.asarray(states[:, :2], dtype=np.float32))
    metrics.append((float(reward_sum), bool(success)))

  pos, scatter_idx = _free_grid_positions(
      walls, int(heatmap_subcells), state_dim)
  gmean = jnp.asarray(nf_goal_mean, dtype=jnp.float32)
  gstd = jnp.asarray(nf_goal_std, dtype=jnp.float32)
  pos_j = jnp.asarray(pos, dtype=jnp.float32)
  s0_j = jnp.asarray(s0, dtype=jnp.float32)
  goal_j = jnp.asarray(goal, dtype=jnp.float32)
  _eval = _nf_grid_eval_fn(networks, nf_reward_fn, state_dim)
  logp_s, logp_g = _eval(
      policy_params, nf_params, pos_j, s0_j, goal_j, gmean, gstd)
  p_s = np.exp(np.clip(np.asarray(logp_s, dtype=np.float32), -80.0, 80.0))
  p_g = np.exp(np.clip(np.asarray(logp_g, dtype=np.float32), -80.0, 80.0))
  p_pre = p_s * p_g

  hh = int(walls.shape[0] * max(1, int(heatmap_subcells)))
  ww = int(walls.shape[1] * max(1, int(heatmap_subcells)))
  occ = _scatter_grid(p_s, scatter_idx, hh, ww, walls, int(heatmap_subcells))
  pre = _scatter_grid(p_pre, scatter_idx, hh, ww, walls, int(heatmap_subcells))
  occ = _normalize_free(occ)
  pre = _normalize_free(pre)

  n_succ = int(sum(1 for _, s in metrics if s))
  out_dir = os.path.dirname(os.path.abspath(out_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  np.savez_compressed(
      os.path.splitext(out_path)[0] + '.npz',
      occ=np.asarray(occ),
      pre=np.asarray(pre),
      goal=np.asarray(goal, dtype=np.float32),
      start=np.asarray(s0[:2], dtype=np.float32),
      walls=np.asarray(walls),
      n=np.int32(pos.shape[0]),
      mean_p=np.float32(np.mean(p_g)),
      median_p=np.float32(np.median(p_g)),
      title=np.asarray(title),
  )
  save_occupancy_preimage_figure(
      out_path, walls, occ, pre, goal, title=title,
      subtitle=(f'trajs={n_traj} (1 mode + {max(0, n_traj - 1)} stoch)  '
                f'success={n_succ}/{n_traj}  '
                f'mean p(g|s,a)={float(np.mean(p_g)):.3g}'),
      trajectories=trjs,
      start=s0[:2])
  return out_path


def save_occupancy_preimage_figure(
    out_path: str,
    walls: np.ndarray,
    occ: np.ndarray,
    pre: np.ndarray,
    goal: np.ndarray,
    title: str,
    subtitle: str = '',
    shared_scale: bool = True,
    trajectories: Optional[list] = None,
    start: Optional[np.ndarray] = None,
) -> str:
  """Write the two-panel PNG. Grids should already be normalized to sum 1."""
  walls = np.asarray(walls)
  h, w = walls.shape
  shared = _shared_power_norm(occ, pre) if shared_scale else None
  fig, axes = plt.subplots(
      1, 2, figsize=(12.4, 5.6 * h / max(w, 1)), layout='constrained')
  fig.suptitle(f'{title}\n{subtitle}'.rstrip(), fontsize=11)
  _draw_occ_panel(
      axes[0], walls, occ, goal,
      'probability mass' if not shared_scale else None,
      r'$p_{\mathrm{NF}}(s \mid s_0, a_0)$',
      norm=shared, add_colorbar=not shared_scale,
      trajectories=trajectories, start=start)
  _draw_occ_panel(
      axes[1], walls, pre, goal,
      'probability mass',
      r'$p_{\mathrm{NF}}(s \mid s_0, a_0)\, p(g \mid s, a_\pi)/Z$',
      norm=shared, add_colorbar=True,
      trajectories=trajectories, start=start)
  fig.savefig(out_path, dpi=140)
  plt.close(fig)
  return out_path


def replot_occupancy_preimage_npz(npz_path: str, out_path: Optional[str] = None) -> str:
  data = np.load(npz_path, allow_pickle=True)
  title = str(data['title']) if 'title' in data.files else os.path.basename(npz_path)
  n = int(data['n']) if 'n' in data.files else -1
  mean_p = float(data['mean_p']) if 'mean_p' in data.files else float('nan')
  median_p = float(data['median_p']) if 'median_p' in data.files else float('nan')
  if out_path is None:
    out_path = os.path.splitext(npz_path)[0] + '.png'
  return save_occupancy_preimage_figure(
      out_path,
      data['walls'],
      data['occ'],
      data['pre'],
      data['goal'],
      title=title,
      subtitle=(f'n={n}  mean p(g|s,a)={mean_p:.3g}  median p={median_p:.3g}'),
      start=data['start'] if 'start' in data.files else None)


def _shared_power_norm(*grids) -> PowerNorm:
  vmax = 0.0
  for grid in grids:
    pos = np.asarray(grid, dtype=np.float64)
    pos = pos[np.isfinite(pos) & (pos > 0.0)]
    if pos.size:
      vmax = max(vmax, float(np.max(pos)))
  if vmax <= 0.0:
    vmax = 1e-12
  return PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)


def _bin_xy(
    xy: np.ndarray,
    walls: np.ndarray,
    subcells: int,
    weights: Optional[np.ndarray],
) -> np.ndarray:
  sub = max(1, int(subcells))
  h, w = walls.shape
  hh, ww = h * sub, w * sub
  grid = np.zeros((hh, ww), dtype=np.float64)
  i = np.floor(xy[:, 0] * sub).astype(np.int32)
  j = np.floor(xy[:, 1] * sub).astype(np.int32)
  valid = (i >= 0) & (i < hh) & (j >= 0) & (j < ww)
  if weights is None:
    np.add.at(grid, (i[valid], j[valid]), 1.0)
  else:
    wwts = np.asarray(weights, dtype=np.float64).reshape(-1)
    np.add.at(grid, (i[valid], j[valid]), wwts[valid])
  wall_hi = np.repeat(np.repeat(np.asarray(walls) == 1, sub, axis=0), sub, axis=1)
  grid = np.where(wall_hi, np.nan, grid)
  return grid.astype(np.float32)


def _normalize_free(grid: np.ndarray) -> np.ndarray:
  out = np.asarray(grid, dtype=np.float64).copy()
  mass = np.nansum(out)
  if mass > 0.0:
    out /= mass
  return out.astype(np.float32)


_NF_GRID_EVAL = {}


def _nf_grid_eval_fn(networks: Any, nf_reward_fn: Callable, state_dim: int):
  """Compile once per (networks, reward_fn, state_dim); reuse across ckpts."""
  key = (id(networks), id(nf_reward_fn), int(state_dim))
  cached = _NF_GRID_EVAL.get(key)
  if cached is not None:
    return cached

  @jax.jit
  def _eval(policy_p, nf_p, pos_batch, start, goal_vec, gmean, gstd):
    n = pos_batch.shape[0]
    obs_s0g = jnp.concatenate([start, goal_vec], axis=-1)[None]
    dist0 = networks.policy_network.apply(policy_p, obs_s0g)
    a0 = networks.sample_eval(dist0, jax.random.PRNGKey(0))[0]
    s0_b = jnp.broadcast_to(start[None, :], (n, state_dim))
    a0_b = jnp.broadcast_to(a0[None, :], (n, a0.shape[0]))
    obs_fwd = jnp.concatenate([s0_b, pos_batch], axis=-1)
    logp_s = nf_reward_fn(nf_p, obs_fwd, a0_b, gmean, gstd)
    g_b = jnp.broadcast_to(goal_vec[None, :], (n, state_dim))
    obs_pre = jnp.concatenate([pos_batch, g_b], axis=-1)
    dist = networks.policy_network.apply(policy_p, obs_pre)
    act = networks.sample_eval(dist, jax.random.PRNGKey(0))
    logp_g = nf_reward_fn(nf_p, obs_pre, act, gmean, gstd)
    return logp_s, logp_g

  _NF_GRID_EVAL[key] = _eval
  return _eval


def _free_grid_positions(
    walls: np.ndarray, subcells: int, state_dim: int,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
  """Cell-center samples on free floor, plus scatter indices into the hi-res grid."""
  sub = max(1, int(subcells))
  rows, cols = np.where(np.asarray(walls) == 0)
  rr = (np.arange(sub, dtype=np.float32) + 0.5) / float(sub)
  cc = (np.arange(sub, dtype=np.float32) + 0.5) / float(sub)
  offsets = np.stack(np.meshgrid(rr, cc, indexing='ij'), axis=-1).reshape(-1, 2)
  base = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=-1)
  pos_2d = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
  if int(state_dim) > 2:
    pos = np.concatenate(
        [pos_2d, np.zeros((len(pos_2d), int(state_dim) - 2), dtype=np.float32)],
        axis=-1)
  else:
    pos = pos_2d
  i0 = np.repeat(rows * sub, sub * sub)
  j0 = np.repeat(cols * sub, sub * sub)
  di = np.tile(np.repeat(np.arange(sub), sub), rows.shape[0])
  dj = np.tile(np.tile(np.arange(sub), sub), rows.shape[0])
  return pos.astype(np.float32), (i0 + di, j0 + dj)


def _scatter_grid(
    values: np.ndarray,
    scatter_idx: Tuple[np.ndarray, np.ndarray],
    hh: int,
    ww: int,
    walls: np.ndarray,
    subcells: int,
) -> np.ndarray:
  grid = np.full((hh, ww), np.nan, dtype=np.float32)
  grid[scatter_idx] = np.asarray(values, dtype=np.float32).reshape(-1)
  wall_hi = np.repeat(
      np.repeat(np.asarray(walls) == 1, int(subcells), axis=0),
      int(subcells), axis=1)
  return np.where(wall_hi, np.nan, grid)


def _draw_occ_panel(ax, walls, heatmap, goal, heat_label, panel_title,
                    norm=None, add_colorbar=True, trajectories=None,
                    start=None):
  h, w = walls.shape
  fig = ax.figure
  # Light free cells + dark walls. Inferno-from-zero painted the whole
  # maze black because empty occupancy bins are 0.
  floor = np.where(np.asarray(walls) == 1, 0.18, 0.94)
  ax.imshow(
      floor, cmap='gray', origin='upper', extent=[0, w, h, 0],
      interpolation='nearest', vmin=0.0, vmax=1.0)
  vis = np.asarray(heatmap, dtype=np.float64).copy()
  vis[~np.isfinite(vis) | (vis <= 0.0)] = np.nan
  pos = vis[np.isfinite(vis)]
  divider = make_axes_locatable(ax)
  if pos.size > 0:
    if norm is None:
      vmax = float(np.max(pos))
      vmin = float(np.min(pos))
      if vmax <= vmin:
        vmax = vmin * 1.01 + 1e-12
      norm = PowerNorm(gamma=0.45, vmin=vmin, vmax=vmax)
    hm = ax.imshow(
        np.ma.masked_invalid(vis), origin='upper',
        extent=[0, w, h, 0], interpolation='nearest', alpha=0.92,
        cmap='YlOrRd', norm=norm)
    if add_colorbar:
      cax = divider.append_axes('right', size='4%', pad=0.12)
      cbar = fig.colorbar(hm, cax=cax)
      if heat_label:
        cbar.ax.set_ylabel(heat_label, rotation=270, labelpad=14)
  if trajectories:
    cmap_traj = plt.get_cmap('tab10')
    for t_i, st in enumerate(trajectories):
      xs, ys = np.asarray(st)[:, 1], np.asarray(st)[:, 0]
      color = cmap_traj(t_i % 10)
      ax.plot(
          xs, ys, '-', color=color, linewidth=1.8 if t_i == 0 else 1.2,
          alpha=0.95 if t_i == 0 else 0.65, zorder=5,
          label='policy' if t_i == 0 else None)
  if start is not None:
    ax.plot(
        float(start[1]), float(start[0]), marker='o', color='lime',
        markersize=9, markeredgecolor='black', linewidth=0, zorder=6,
        label='start')
  ax.plot(goal[1], goal[0], marker='*', color='tab:blue', markersize=16,
          markeredgecolor='black', linewidth=0, label='goal', zorder=7)
  ax.set_xlim(0, w)
  ax.set_ylim(h, 0)
  ax.set_aspect('equal')
  ax.set_xlabel('col  (x)')
  ax.set_ylabel('row  (y, inverted)')
  ax.set_title(panel_title, fontsize=11)
  ax.set_xticks(np.arange(0, w + 1, 1))
  ax.set_yticks(np.arange(0, h + 1, 1))
  ax.grid(True, color='0.55', linewidth=0.4, alpha=0.45)
  ax.legend(loc='upper right', fontsize=8, framealpha=0.85)
