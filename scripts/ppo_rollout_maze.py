"""Plot trajectories of a PPO policy on a 2D maze env, one PNG per checkpoint.

Companion to `ppo_rollout_video.py`; same CLI conventions but produces
static matplotlib PNGs instead of mp4s.  Target envs are the
`point_*` family (FourRooms, Spiral11x11, Impossible, ...) defined in
`point_env.py`, whose state is a 2-D (row, col) position with continuous
coordinates in `[0, H] x [0, W]` and a discrete wall grid on the
integer cells.

Examples:
  # Single checkpoint -> single PNG.
  python ppo_rollout_maze.py \\
      --checkpoint=logs/ppo/ppo_point_FourRooms_3/checkpoints/latest.pkl \\
      --env=point_FourRooms \\
      --output=plots/point_FourRooms_3.png

  # A directory -> one PNG per ckpt_iter_*.pkl (+ latest.pkl).
  python ppo_rollout_maze.py \\
      --checkpoint=logs/ppo/ppo_point_FourRooms_3/checkpoints \\
      --env=point_FourRooms \\
      --output=plots/point_FourRooms_3/

  # Synthetic diagnostic — no checkpoint needed, draws a deliberate L-shape
  # with a labeled start / end so the axis orientation is obvious.
  python ppo_rollout_maze.py --env=point_FourRooms --test_only \\
      --output=plots/orientation_test.png

When a checkpoint is loaded, each PNG defaults to a **1×2** figure: the
left panel overlays a scalar field over every free cell — **φ(s,a)·ψ(g)**
for default runs, or the **dirac_target PPO reward** when
``run_config.json`` has ``ppo_reward_mode=dirac_target`` (or the
checkpoint path contains ``dirac_baseline``).  ``a`` is the policy's
deterministic action at ``concat([s,g])``; dirac uses ``s0`` from the
plotted rollout's start state.  The right panel is always **V(s,g)**.
Use ``--no_repr_overlay`` to skip the heatmaps and recover a single
trajectory panel.

Y-axis convention
-----------------
The WALLS arrays in `point_env.py` are laid out as
`walls[row, col]`, with `row=0` at the **top** of the maze (this is how
they appear when printed literally in Python source).  The env's
observation stores `(row, col)` verbatim, so the natural mapping for
visualization is `x=col, y=row`.  matplotlib's default scatter/plot
places y=0 at the *bottom*, so we explicitly invert the y-axis (and
`imshow` is given `origin='upper'`) to keep row 0 at the top in the
rendered figure.  Without this, trajectories appear vertically mirrored
relative to the maze as it's printed.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcrl_jax_acme_compat  # noqa: F401  must precede acme/jax imports

import argparse
import glob
import json
import os
import re
from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from acme import specs
import matplotlib
matplotlib.use('Agg')   # headless-safe; must be before pyplot import
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

import contrastive
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
import env_utils
from ppo_contrastive import fixed_goal_dict


# ---------------------------------------------------------------------------
# Checkpoint / path helpers — copied verbatim from ppo_rollout_video.py so
# the two scripts stay in sync.  If you need to edit the discovery logic,
# update both places (or refactor into a shared util).
# ---------------------------------------------------------------------------
def _enumerate_checkpoints(path: str):
  """Resolve --checkpoint to a list of (label, pkl_path) pairs.

  If `path` is a file, returns it as a single entry whose label is
  derived from the filename (`ckpt_iter_5000.pkl` -> `iter_5000`,
  `latest.pkl` -> `latest`).  If `path` is a directory, enumerates every
  `ckpt_iter_*.pkl` inside it in ascending iteration order, followed by
  `latest.pkl` (if present).
  """
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


def _resolve_output_path(output_arg: str, env: str, label: str,
                         multi: bool, ext: str = '.png') -> str:
  """Translate the user's --output into a per-checkpoint filename.

  Rules (mirror ppo_rollout_video.py):
    - `--output foo.png` + single checkpoint  -> `foo.png`
    - `--output foo.png` + multi              -> `foo_<label>.png`
    - `--output dir/`    (trailing slash)     -> `dir/<env>_<label>.png`
    - `--output dir`     (existing dir)       -> `dir/<env>_<label>.png`
  """
  is_dir_like = output_arg.endswith(os.sep) or os.path.isdir(output_arg)
  if is_dir_like:
    return os.path.join(output_arg, f'{env}_{label}{ext}')
  if not multi:
    return output_arg
  stem, e = os.path.splitext(output_arg)
  e = e or ext
  return f'{stem}_{label}{e}'


def _append_suffix_to_path(path: str, suffix: str) -> str:
  stem, ext = os.path.splitext(path)
  if not ext:
    ext = '.png'
  return f'{stem}_{suffix}{ext}'


def _infer_run_config_from_checkpoint(ckpt_path: str) -> dict:
  """Best-effort read ``resolved_config`` from neighboring run_config.json."""
  # Expected layout:
  #   <run_dir>/checkpoints/ckpt_iter_*.pkl
  #   <run_dir>/run_config.json
  ckpt_abs = os.path.abspath(ckpt_path)
  ckpt_dir = os.path.dirname(ckpt_abs)
  run_dir = os.path.dirname(ckpt_dir)
  run_cfg = os.path.join(run_dir, 'run_config.json')
  if not os.path.isfile(run_cfg):
    return {}
  try:
    with open(run_cfg, 'r', encoding='utf-8') as fh:
      payload = json.load(fh)
    resolved = payload.get('resolved_config', {})
    if not isinstance(resolved, dict):
      return {}
    return dict(resolved)
  except Exception:
    return {}


def _infer_repr_norm_from_checkpoint(ckpt_path: str) -> Optional[bool]:
  v = _infer_run_config_from_checkpoint(ckpt_path).get('repr_norm', None)
  return v if isinstance(v, bool) else None


def _resolve_ppo_reward_overlay(ckpt_path: str,
                                cli_reward_mode: str,
                                cli_dirac_eps: float) -> Tuple[str, float]:
  """Return (overlay_mode, dirac_eps) for heatmap left panel."""
  resolved = _infer_run_config_from_checkpoint(ckpt_path)
  mode = (cli_reward_mode or resolved.get('ppo_reward_mode', '') or '').strip().lower()
  abs_path = os.path.abspath(ckpt_path)
  if not mode and 'kde_dirac' in abs_path:
    mode = 'kde_dirac'
  elif not mode and 'dirac_baseline' in abs_path:
    mode = 'dirac_target'
  eps = float(cli_dirac_eps) if cli_dirac_eps > 0 else float(
      resolved.get('ppo_dirac_eps', 1e-6))
  return mode, eps


# ---------------------------------------------------------------------------
# Network + env construction — matches PPO training-time config so the
# checkpoint's policy_params plug in cleanly.
# ---------------------------------------------------------------------------
def _build_networks(env_name, seed):
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed,
      fixed_start_end=fixed_goal_dict[env_name])
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env

  cfg = contrastive.ContrastiveConfig()
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=cfg.repr_dim,
      repr_norm=cfg.repr_norm,
      twin_q=cfg.twin_q,
      use_image_obs=cfg.use_image_obs,
      hidden_layer_sizes=cfg.hidden_layer_sizes,
      actor_min_std=float(cfg.ppo_actor_min_std),
  )
  return networks, obs_dim


def _get_raw_point_env(env_name):
  """Return the underlying PointEnv + its wall grid.

  `env_utils.load` already builds the right PointEnv subclass; we keep
  the returned object for stepping and pull its private `_walls` array
  for rendering.  We assert it's a point-family env because no other
  env in this codebase exposes a 2-D grid layout.
  """
  if not env_name.startswith('point_'):
    raise ValueError(
        f'This script is maze-specific; got env={env_name!r}.  '
        f'Use ppo_rollout_video.py for sawyer_*.')
  gym_env, obs_dim, max_steps = env_utils.load(
      env_name, fixed_start_end=fixed_goal_dict[env_name], seed=None)
  walls = np.asarray(gym_env._walls)   # (H, W), 1=wall, 0=free
  return gym_env, obs_dim, max_steps, walls


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _policy_gaussian_stats(dist) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return (mode_action, gaussian_loc, gaussian_scale) from a policy dist."""
  normal = dist.distribution.distribution
  loc = np.asarray(normal.loc)[0]
  scale = np.asarray(normal.scale)[0]
  mode = np.asarray(dist.mode())[0]
  return mode.astype(np.float32), loc.astype(np.float32), scale.astype(np.float32)


def _rollout_one(policy_params, gym_env, networks,
                 max_steps: int, stochastic: bool, seed: int,
                 state_dim: int = None,
                 log_actions: bool = False,
                 log_policy_stats: bool = False,
                 log_prefix: str = ''):
  """Run one rollout and return (states (T+1, state_dim), goal (state_dim,), reward_sum, success).

  state_dim defaults to half the observation length, which equals 2 for standard
  2D mazes and 4 for SixteenRooms4D (2 maze + 2 extra dims).
  """
  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  @jax.jit
  def policy_sample(params, obs, rng):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample(dist, rng)

  def _policy_dist(params, obs):
    return networks.policy_network.apply(params, obs[None])

  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  s_dim = state_dim if state_dim is not None else (len(obs) // 2)
  goal = obs[s_dim:].copy()
  states = [obs[:s_dim].copy()]
  action_rows: Optional[list] = [] if log_actions else None
  policy_stat_rows: Optional[list] = [] if log_policy_stats else None
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(seed)
  for step in range(max_steps):
    dist = _policy_dist(policy_params, obs)
    if log_policy_stats:
      mode_a, loc, scale = _policy_gaussian_stats(dist)
      policy_stat_rows.append((mode_a.copy(), loc.copy(), scale.copy()))
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = networks.sample(dist, k)
    else:
      action_j = networks.sample_eval(dist, jax.random.PRNGKey(0))
    action = np.asarray(action_j)[0].astype(np.float32)
    if action_rows is not None:
      action_rows.append(action.copy())
    obs_next, r, done, _ = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    states.append(obs[:s_dim].copy())
    total_reward += float(r)
    if float(r) > 0.0:
      success = True
    if done:
      break
  if log_actions:
    if action_rows:
      A = np.stack(action_rows, axis=0)
      l2 = np.linalg.norm(A, axis=-1)
      print(f'{log_prefix}[maze] policy actions: shape={A.shape} dtype={A.dtype} '
            f'action_dim={A.shape[-1]}  per-step ||a||_2: '
            f'min={l2.min():.4f} mean={l2.mean():.4f} max={l2.max():.4f}  '
            f'elem-wise min/max={A.min():.4f}/{A.max():.4f}')
    else:
      print(f'{log_prefix}[maze] policy actions: no env steps (empty rollout).')
  if log_policy_stats:
    if policy_stat_rows:
      modes = np.stack([r[0] for r in policy_stat_rows], axis=0)
      locs = np.stack([r[1] for r in policy_stat_rows], axis=0)
      scales = np.stack([r[2] for r in policy_stat_rows], axis=0)
      print(f'{log_prefix}[maze] policy stats over {len(policy_stat_rows)} steps '
            f'(mode=tanh-Gaussian mean, loc/scale=pre-tanh Normal):')
      for t in range(len(policy_stat_rows)):
        m, l, s = modes[t], locs[t], scales[t]
        print(f'{log_prefix}  step={t:3d}  mode={np.array2string(m, precision=4, floatmode="fixed")}  '
              f'loc={np.array2string(l, precision=4, floatmode="fixed")}  '
              f'scale={np.array2string(s, precision=6, floatmode="fixed")}')
      print(f'{log_prefix}[maze] loc  mean={locs.mean(axis=0)}  std={locs.std(axis=0)}  '
            f'min={locs.min(axis=0)}  max={locs.max(axis=0)}')
      print(f'{log_prefix}[maze] scale mean={scales.mean(axis=0)}  std={scales.std(axis=0)}  '
            f'min={scales.min(axis=0)}  max={scales.max(axis=0)}  '
            f'||scale|| mean={np.linalg.norm(scales, axis=-1).mean():.6f}')
    else:
      print(f'{log_prefix}[maze] policy stats: no env steps (empty rollout).')
  return (np.asarray(states, dtype=np.float32), goal,
          float(total_reward), bool(success))


# ---------------------------------------------------------------------------
# CRL heatmaps over free cells (point mazes: obs = [state, goal], dim 4).
# ---------------------------------------------------------------------------
def _point_state_goal_dim(gym_env) -> int:
  d = int(np.prod(gym_env.observation_space.shape))
  if d % 2 != 0:
    raise ValueError(
        f'Expected even-length obs for [state, goal] concat; got dim {d}')
  return d // 2


def _make_maze_repr_field_fn(
    networks,
    state_dim: int,
    normalize_repr: bool = False,
):
  """Jitted (policy, q, value, positions, goal) -> scalar fields."""

  @jax.jit
  def _eval(policy_p, q_p, value_p, pos_batch: jnp.ndarray, goal_vec: jnp.ndarray,
            s0_vec: jnp.ndarray):
    del s0_vec  # unused for default reward overlay
    def _l2_normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
      denom = jnp.linalg.norm(x, axis=-1, keepdims=True)
      return x / jnp.maximum(denom, eps)

    g = jnp.broadcast_to(goal_vec[None, :], (pos_batch.shape[0], state_dim))
    obs_pg = jnp.concatenate([pos_batch, g], axis=-1)
    dist = networks.policy_network.apply(policy_p, obs_pg)
    act = networks.sample_eval(dist, jax.random.PRNGKey(0))

    _, phi, psi_g = networks.q_network.apply(q_p, obs_pg, act)
    if normalize_repr:
      phi_l = _l2_normalize(phi)
      psi_g_l = _l2_normalize(psi_g)
      phi_dot = jnp.sum(phi_l * psi_g_l, axis=-1)
    else:
      phi_dot = jnp.sum(phi * psi_g, axis=-1)

    v = networks.value_network.apply(value_p, obs_pg)
    return phi_dot, v

  return _eval


def _make_maze_dirac_reward_field_fn(
    networks,
    state_dim: int,
    eps: float,
):
  """Jitted dirac_target reward field (matches ``ppo_learner`` rollout reward)."""

  goal_tol = 1e-2
  log_eps = float(np.log(max(float(eps), 1e-10)))

  @jax.jit
  def _eval(policy_p, q_p, value_p, pos_batch: jnp.ndarray, goal_vec: jnp.ndarray,
            s0_vec: jnp.ndarray):
    g = jnp.broadcast_to(goal_vec[None, :], (pos_batch.shape[0], state_dim))
    s0 = jnp.broadcast_to(s0_vec[None, :], (pos_batch.shape[0], state_dim))
    obs_pg = jnp.concatenate([pos_batch, g], axis=-1)
    dist = networks.policy_network.apply(policy_p, obs_pg)
    act = networks.sample_eval(dist, jax.random.PRNGKey(0))

    obs_s0 = jnp.concatenate([s0, g], axis=-1)
    obs_ss = jnp.concatenate([pos_batch, pos_batch], axis=-1)
    _, phi_s0, psi_g = networks.q_network.apply(q_p, obs_s0, act)
    _, _, psi_s = networks.q_network.apply(q_p, obs_ss, act)
    dot_ps = jnp.sum(phi_s0 * psi_s, axis=-1)
    dot_pg = jnp.sum(phi_s0 * psi_g, axis=-1)
    at_goal = jnp.linalg.norm(pos_batch - g, axis=-1) < goal_tol
    log_rew = jnp.asarray(log_eps, dtype=dot_ps.dtype) - dot_ps
    r = jnp.where(at_goal, -dot_pg, log_rew)

    v = networks.value_network.apply(value_p, obs_pg)
    return r, v

  return _eval


def _collect_rollout_states(
    gym_env,
    policy_params,
    networks,
    max_steps: int,
    num_rollouts: int,
    seed: int,
) -> np.ndarray:
  """Run *num_rollouts* stochastic episodes and return all visited states (N, 2)."""
  all_states = []
  for i in range(num_rollouts):
    states, _, _, _ = _rollout_one(
        policy_params, gym_env, networks, max_steps,
        stochastic=True, seed=seed + i)
    all_states.append(states)
  return np.concatenate(all_states, axis=0).astype(np.float32)


def _make_maze_kde_reward_field_fn(
    kde,   # GaussianKDE from contrastive.ppo_learner
    networks,
    state_dim: int,
    eps: float,
):
  """numpy-compatible eval_fields for the KDE-dirac reward heatmap.

  Left field  = log(ε) − log p_kde(s)    (off-goal)
              = −log p_kde(g)             (at goal)
  Right field = V(s, g) from the value network.
  KDE is fitted on policy rollout states collected at render time.
  """
  log_eps = float(np.log(max(float(eps), 1e-10)))
  goal_tol = 1e-2

  def _eval(policy_p, q_p, value_p,
            pos_batch: jnp.ndarray,
            goal_vec: jnp.ndarray,
            s0_vec: jnp.ndarray):
    del q_p, s0_vec  # not used for KDE reward
    pos_np = np.asarray(pos_batch, dtype=np.float32)
    goal_np = np.asarray(goal_vec, dtype=np.float32)
    g_np = np.broadcast_to(goal_np[None, :], (pos_np.shape[0], state_dim))

    log_ps = kde.log_density(pos_np)
    log_pg = float(kde.log_density(goal_np[None, :])[0])
    at_goal = np.linalg.norm(pos_np - g_np, axis=-1) < goal_tol
    kde_rew = np.where(at_goal, -log_pg, log_eps - log_ps)

    # Value still from the network
    g_j = jnp.broadcast_to(
        jnp.asarray(goal_np)[None, :], (pos_batch.shape[0], state_dim))
    obs_pg = jnp.concatenate([pos_batch, g_j], axis=-1)
    v = networks.value_network.apply(value_p, obs_pg)
    return jnp.asarray(kde_rew, dtype=jnp.float32), v

  return _eval


def _repr_grids_over_maze(
    walls: np.ndarray,
    goal: np.ndarray,
    policy_params,
    q_params,
    value_params,
    eval_fields,
    subcells: int = 1,
    s0_state: Optional[np.ndarray] = None,
    state_dim: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
  """Scalar fields on an (H*subcells)×(W*subcells) grid (NaN on walls).

  For environments with extra dimensions (state_dim > 2, e.g. SixteenRooms4D),
  the grid is evaluated at maze positions padded with zeros for the extra dims,
  i.e. at [row, col, 0, 0, ...].
  """
  H, W = walls.shape
  subcells = max(1, int(subcells))
  rows, cols = np.where(walls == 0)
  # Sample each free cell on a uniform sub-grid.
  rr = (np.arange(subcells, dtype=np.float32) + 0.5) / float(subcells)
  cc = (np.arange(subcells, dtype=np.float32) + 0.5) / float(subcells)
  offsets = np.stack(np.meshgrid(rr, cc, indexing='ij'), axis=-1).reshape(-1, 2)
  base = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=-1)
  pos_2d = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
  # Pad extra dims with zeros so eval_fields receives state_dim-wide positions.
  if state_dim > 2:
    pos_full = np.concatenate(
        [pos_2d, np.zeros((len(pos_2d), state_dim - 2), dtype=np.float32)],
        axis=-1)
  else:
    pos_full = pos_2d
  gvec = jnp.asarray(goal, dtype=jnp.float32)
  pos_j = jnp.asarray(pos_full, dtype=jnp.float32)
  s0_j = jnp.asarray(
      s0_state if s0_state is not None else np.zeros(state_dim, dtype=np.float32),
      dtype=jnp.float32)
  left_field, v = eval_fields(
      policy_params, q_params, value_params, pos_j, gvec, s0_j)
  HH, WW = H * subcells, W * subcells
  left_grid = np.full((HH, WW), np.nan, dtype=np.float32)
  value_grid = np.full((HH, WW), np.nan, dtype=np.float32)
  # Map each sampled subcell back into the upsampled heatmap lattice.
  i0 = np.repeat(rows * subcells, subcells * subcells)
  j0 = np.repeat(cols * subcells, subcells * subcells)
  di = np.tile(np.repeat(np.arange(subcells), subcells), rows.shape[0])
  dj = np.tile(np.tile(np.arange(subcells), subcells), rows.shape[0])
  ii = i0 + di
  jj = j0 + dj
  left_grid[ii, jj] = np.asarray(left_field, dtype=np.float32)
  value_grid[ii, jj] = np.asarray(v, dtype=np.float32)
  return left_grid, value_grid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _draw_maze_panel(
    ax,
    walls: np.ndarray,
    states: Optional[np.ndarray],
    trajectories: Optional[list[np.ndarray]],
    goal: np.ndarray,
    heatmap: Optional[np.ndarray],
    heat_label: str,
    extra_dims_trajs: Optional[list] = None,
):
  """Walls + optional heatmap + trajectory on one axis.

  extra_dims_trajs: list of (T, K) arrays (one per trajectory) with values for
  each extra dimension.  When provided, trajectory segments are colored by
  extra_dims_trajs[i][:, 0] (dim2 / z) using a plasma colormap.
  """
  H, W = walls.shape
  fig = ax.figure
  ax.imshow(
      walls,
      cmap='gray_r',
      origin='upper',
      extent=[0, W, H, 0],
      interpolation='nearest',
      alpha=0.85,
  )

  divider = make_axes_locatable(ax)
  if heatmap is not None:
    hm = ax.imshow(
        np.ma.masked_invalid(heatmap),
        origin='upper',
        extent=[0, W, H, 0],
        interpolation='nearest',
        alpha=0.5,
        cmap='magma',
    )
    cax = divider.append_axes('right', size='4%', pad=0.12)
    cbar = fig.colorbar(hm, cax=cax)
    cbar.ax.set_ylabel(heat_label, rotation=270, labelpad=12)

  trjs = trajectories if trajectories is not None else (
      [states] if states is not None else [])
  cmap_traj = plt.get_cmap('tab10')
  cmap_z = plt.get_cmap('plasma')
  lc_for_cbar = None  # (LineCollection, norm) for z colorbar

  for t_i, st in enumerate(trjs):
    xs = st[:, 1]   # col -> x
    ys = st[:, 0]   # row -> y
    label = f'traj {t_i}' if len(trjs) > 1 else 'trajectory'

    if extra_dims_trajs is not None and t_i < len(extra_dims_trajs):
      extra = extra_dims_trajs[t_i]          # (T, K)
      z_vals = extra[:, 0]                   # first extra dim
      z_min, z_max = float(z_vals.min()), float(z_vals.max())
      norm_z = plt.Normalize(z_min, z_max if z_max > z_min else z_min + 1e-6)
      # Line segments colored by z
      pts = np.stack([xs, ys], axis=-1)
      segs = np.stack([pts[:-1], pts[1:]], axis=1)
      z_seg = 0.5 * (z_vals[:-1] + z_vals[1:])
      lc = LineCollection(segs, cmap=cmap_z, norm=norm_z,
                          linewidth=2.0, alpha=0.9, label=label)
      lc.set_array(z_seg)
      ax.add_collection(lc)
      ax.scatter(xs, ys, c=z_vals, cmap=cmap_z, norm=norm_z,
                 s=6, alpha=0.4, zorder=4)
      lc_for_cbar = (lc, norm_z)
    else:
      color = cmap_traj(t_i % 10)
      ax.plot(xs, ys, '-', color=color, linewidth=1.8, alpha=0.9, label=label)
      ax.plot(xs, ys, 'o', color=color, markersize=1.8, alpha=0.5)

    ax.plot(xs[0], ys[0], marker='o', color='lime', markersize=9,
            markeredgecolor='black', linewidth=0)
    ax.plot(xs[-1], ys[-1], marker='X', color='orange', markersize=9,
            markeredgecolor='black', linewidth=0)

  if lc_for_cbar is not None:
    lc_obj, _ = lc_for_cbar
    cax_z = divider.append_axes('bottom', size='4%', pad=0.5)
    cbar_z = fig.colorbar(lc_obj, cax=cax_z, orientation='horizontal')
    cbar_z.ax.set_xlabel('dim 2  (z)', fontsize=8)

  ax.plot(goal[1], goal[0], marker='*', color='red', markersize=18,
          markeredgecolor='black', linewidth=0, label='goal')

  ax.set_xlim(0, W)
  ax.set_ylim(H, 0)
  ax.set_aspect('equal')
  ax.set_xlabel('col  (x)')
  ax.set_ylabel('row  (y, inverted)')
  ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
  ax.grid(True, color='gray', linewidth=0.3, alpha=0.3)


def _plot_trajectory(
    walls: np.ndarray,
    states: np.ndarray,
    trajectories: Optional[list[np.ndarray]],
    goal: np.ndarray,
    title: str,
    out_path: str,
    reward_sum: float,
    success: bool,
    heatmap_phi: Optional[np.ndarray] = None,
    heatmap_value: Optional[np.ndarray] = None,
    normalize_repr: bool = False,
    overlay_mode: str = 'phi_psi',
    dirac_eps: float = 1e-6,
    fig_scale: float = 1.0,
    extra_dims_trajs: Optional[list] = None,
):
  """Save a PNG: trajectory on maze; optional side-by-side CRL heatmaps.

  extra_dims_trajs: list of (T, K) arrays for each trajectory's extra dims.
  When present, trajectories are colored by dim2 (z) and a right-hand panel
  shows the time-series of all extra dimensions.
  """
  H, W = walls.shape
  h_in = (6 * H / max(W, 1)) * float(fig_scale)
  goal_2d = goal[:2]  # maze coords; safe for both 2D and 4D goals

  has_extra = (extra_dims_trajs is not None and len(extra_dims_trajs) > 0
               and extra_dims_trajs[0].shape[1] > 0)
  has_heatmaps = heatmap_phi is not None and heatmap_value is not None

  # Build label strings before creating axes
  if overlay_mode == 'dirac_target':
    heat0 = (r'$r_{\mathrm{dirac}}(s)$  '
             r'(off-goal: $\log\epsilon-\phi(s_0,a)\!\cdot\!\psi(s)$; '
             r'at $g$: $-\phi(s_0,a)\!\cdot\!\psi(g)$)')
    ax0_title = f'{heat0}  ($\\epsilon$={dirac_eps:g}, $s_0$=rollout start)'
  elif overlay_mode == 'kde_dirac':
    heat0 = (r'$r_{\mathrm{kde}}(s)$  '
             r'(off-goal: $\log\epsilon - \log p_{\mathrm{kde}}(s)$; '
             r'at $g$: $-\log p_{\mathrm{kde}}(g)$)')
    ax0_title = f'{heat0}  ($\\epsilon$={dirac_eps:g}, KDE on policy rollouts)'
  else:
    heat0 = (r'$\cos(\phi(s,a), \psi(g))$'
           if normalize_repr else r'$\phi(s,a)\cdot\psi(g)$')
    ax0_title = f'{heat0}  (policy mode $a$)'
  heat1 = r'$V(s,g)$'

  # Figure layout: cols = [maze(+heatmap)] [value] [extra-dims ts]
  n_cols = (2 if has_heatmaps else 1) + (1 if has_extra else 0)
  w_ratios = None
  if has_heatmaps and has_extra:
    w_ratios = [3, 3, 2]
  elif has_extra:
    w_ratios = [3, 2]

  fig, axes = plt.subplots(
      1, n_cols,
      figsize=((7 * n_cols) * float(fig_scale), h_in),
      gridspec_kw={'width_ratios': w_ratios} if w_ratios else None,
      layout='constrained')
  if n_cols == 1:
    axes = [axes]

  ax0 = axes[0]
  ax1 = axes[1] if has_heatmaps else None
  ax_ts = axes[-1] if has_extra else None

  supt = (f'{title}\n'
          f'len={len(states)}  reward_sum={reward_sum:.2f}  success={success}')
  fig.suptitle(supt, fontsize=11)

  if has_heatmaps:
    _draw_maze_panel(ax0, walls, states, trajectories, goal_2d, heatmap_phi,
                     heat0, extra_dims_trajs=extra_dims_trajs)
    ax0.set_title(ax0_title, fontsize=9 if overlay_mode == 'dirac_target' else 11)
    _draw_maze_panel(ax1, walls, states, trajectories, goal_2d, heatmap_value,
                     heat1, extra_dims_trajs=extra_dims_trajs)
    ax1.set_title(f'{heat1}  (PPO value head)')
  else:
    _draw_maze_panel(ax0, walls, states, trajectories, goal_2d, None, '',
                     extra_dims_trajs=extra_dims_trajs)
    ax0.set_title(f'{title}\nlen={len(states)}  '
                  f'reward_sum={reward_sum:.2f}  success={success}')

  # Extra-dims time-series panel
  if ax_ts is not None and extra_dims_trajs:
    n_extra = extra_dims_trajs[0].shape[1]
    dim_names = [f'dim{i+2}' for i in range(n_extra)]
    linestyles = ['-', '--', ':', '-.']
    cmap_ts = plt.get_cmap('tab10')
    trjs_for_ts = trajectories if trajectories is not None else (
        [states] if states is not None else [])
    for t_i, extra in enumerate(extra_dims_trajs):
      color = cmap_ts(t_i % 10)
      t = np.arange(len(extra))
      for d in range(n_extra):
        ls = linestyles[d % len(linestyles)]
        label = f'traj{t_i} {dim_names[d]}' if len(trjs_for_ts) > 1 else dim_names[d]
        ax_ts.plot(t, extra[:, d], ls=ls, color=color,
                   linewidth=1.5, alpha=0.85, label=label)
    ax_ts.axhline(0.0, color='black', linewidth=0.5, alpha=0.4)
    ax_ts.set_xlabel('step')
    ax_ts.set_ylabel('extra dim value')
    ax_ts.set_title('Extra dimensions over time')
    ax_ts.legend(fontsize=7, framealpha=0.8)
    ax_ts.grid(True, alpha=0.3)

  out_dir = os.path.dirname(os.path.abspath(out_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def _synthetic_test(env_name: str, out_path: str):
  """Draw an obviously-asymmetric synthetic trajectory to verify orientation.

  We deliberately pick a path that goes: start at (row=1, col=1),
  down to (row=H-2, col=1), then right to (row=H-2, col=W-2).  The
  resulting "L" shape should render with:
    - the green "start" dot near the TOP-LEFT,
    - a long vertical segment going DOWN,
    - a horizontal segment going RIGHT to the bottom-right,
    - the orange "end" X near the BOTTOM-RIGHT,
    - the red goal star placed near the bottom-right too.
  If instead you see the start at the bottom-left, the y-axis is not
  being reversed and something regressed.
  """
  _, _, _, walls = _get_raw_point_env(env_name)
  H, W = walls.shape
  # Synthetic path from (1, 1) -> (H-2, 1) -> (H-2, W-2) in 1-step increments.
  down = [(r + 0.5, 1 + 0.5) for r in range(1, H - 1)]
  right = [(H - 2 + 0.5, c + 0.5) for c in range(1, W - 1)]
  states = np.asarray(down + right[1:], dtype=np.float32)   # (N, 2)
  goal = np.array([H - 2 + 0.5, W - 2 + 0.5], dtype=np.float32)
  _plot_trajectory(
      walls, states, goal,
      title=f'ORIENTATION TEST  env={env_name}  '
            f'(start=top-left, end=bottom-right)',
      out_path=out_path, reward_sum=0.0, success=False)
  print(f'[maze] wrote synthetic orientation test -> {out_path}')
  # Also print the coordinates so we can cross-check if the image looks wrong.
  print(f'       walls shape = {walls.shape}  (rows, cols)')
  print(f'       start (row,col) = ({states[0, 0]:.1f}, {states[0, 1]:.1f})'
        f'  -> should render near TOP-LEFT')
  print(f'       end   (row,col) = ({states[-1, 0]:.1f}, {states[-1, 1]:.1f})'
        f'  -> should render near BOTTOM-RIGHT')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default=None,
                      help='Path to a single .pkl OR a directory containing '
                           'ckpt_iter_*.pkl and/or latest.pkl.  Required '
                           'unless --test_only is set.')
  parser.add_argument('--env', default='point_FourRooms',
                      help='Maze env name; must start with "point_".')
  parser.add_argument('--output', required=True,
                      help='Output .png path, OR a directory (ends in /, '
                           'or already exists) when rendering many checkpoints.')
  parser.add_argument('--stochastic', action='store_true',
                      help='Sample from the policy instead of using mode().')
  parser.add_argument('--num_trajectories', type=int, default=1,
                      help='Number of trajectories to render per checkpoint. '
                           'If >1, stochastic sampling is enabled.')
  parser.add_argument('--fig_scale', type=float, default=1.0,
                      help='Scale factor for figure size (e.g., 1.8 for larger plots).')
  parser.add_argument('--max_steps', type=int, default=-1,
                      help='Override rollout length; -1 = env default.')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--test_only', action='store_true',
                      help='Skip policy loading and just emit a synthetic '
                           'orientation-check plot to --output.  Useful to '
                           'verify the axis convention before running a '
                           'full sweep.')
  parser.add_argument('--no_repr_overlay', action='store_true',
                      help='Do not compute CRL heatmaps; single trajectory '
                           'panel only.')
  parser.add_argument(
      '--no_repr_normalize', action='store_true',
      help='Use raw dot products for heatmaps instead of cosine similarity.')
  parser.add_argument(
      '--repr_normalize', action='store_true',
      help='Use cosine similarity heatmaps (L2-normalized representations).')
  parser.add_argument(
      '--heatmap_subcells', type=int, default=1,
      help='Subdivisions per maze cell for overlay sampling. '
           'Use 5 for 25 samples per cell.')
  parser.add_argument(
      '--log_rollout_actions', action='store_true',
      help='Print action-vector shape and min/mean/max norms for each rollout.')
  parser.add_argument(
      '--log_policy_stats', action='store_true',
      help='Print per-step Gaussian loc/scale and tanh-mode during rollout.')
  parser.add_argument(
      '--ppo_reward_mode', default='',
      help='Override left heatmap: "" uses run_config (phi·psi); '
           '"dirac_target" plots PPO dirac reward; '
           '"kde_dirac" plots KDE reward fitted on policy rollouts.')
  parser.add_argument(
      '--ppo_dirac_eps', type=float, default=-1.0,
      help='Epsilon for dirac/kde_dirac heatmap; <0 uses run_config default.')
  parser.add_argument(
      '--kde_num_rollouts', type=int, default=20,
      help='Number of policy rollouts used to fit the KDE (kde_dirac mode).')
  args = parser.parse_args()
  if args.num_trajectories < 1:
    parser.error('--num_trajectories must be >= 1')

  # ----- Test-only branch: no checkpoint, no networks, no rollout -------
  if args.test_only:
    out_path = args.output
    if out_path.endswith(os.sep) or os.path.isdir(out_path):
      out_path = os.path.join(out_path, f'{args.env}_orientation_test.png')
    _synthetic_test(args.env, out_path)
    return

  if args.checkpoint is None:
    parser.error('--checkpoint is required unless --test_only is set.')

  # ----- 1. Enumerate checkpoints ----------------------------------------
  ckpt_entries = _enumerate_checkpoints(args.checkpoint)
  if not ckpt_entries:
    print(f'[maze] no checkpoints found at {args.checkpoint!r}')
    return
  print(f'[maze] found {len(ckpt_entries)} checkpoint(s) under '
        f'{args.checkpoint}')

  # ----- 2. Build networks + env ONCE and reuse ---------------------------
  print('[maze] building networks and loading env...')
  networks, _ = _build_networks(args.env, seed=args.seed)
  gym_env, _, env_max_steps, walls = _get_raw_point_env(args.env)
  state_dim = _point_state_goal_dim(gym_env)
  n_extra_dims = state_dim - 2  # 0 for standard 2D mazes, 2 for SixteenRooms4D
  force_repr_normalize = bool(args.repr_normalize and not args.no_repr_normalize)
  force_no_repr_normalize = bool(args.no_repr_normalize)
  eval_fields_raw = _make_maze_repr_field_fn(
      networks, state_dim, normalize_repr=False)
  eval_fields_norm = _make_maze_repr_field_fn(
      networks, state_dim, normalize_repr=True)
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  print(f'[maze] env={args.env}  max_steps={max_steps}  '
        f'walls shape={walls.shape} (rows, cols)  state_dim={state_dim}  '
        f'n_extra_dims={n_extra_dims}  '
        f'force_repr_normalize={force_repr_normalize}  '
        f'force_no_repr_normalize={force_no_repr_normalize}')
  if args.num_trajectories > 1 and not args.stochastic:
    print('[maze] num_trajectories>1 -> enabling stochastic policy sampling.')

  # ----- 3. Render one PNG per checkpoint --------------------------------
  multi = len(ckpt_entries) > 1
  for label, path in ckpt_entries:
    print(f'[maze] === {label}  ({path}) ===')
    ckpt = ppo_learner.load_checkpoint(path)
    policy_params = ckpt['policy_params']
    print(f'[maze]   iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}')

    if force_repr_normalize:
      use_repr_normalize = True
      repr_src = 'cli(--repr_normalize)'
    elif force_no_repr_normalize:
      use_repr_normalize = False
      repr_src = 'cli(--no_repr_normalize)'
    else:
      inferred = _infer_repr_norm_from_checkpoint(path)
      if inferred is None:
        use_repr_normalize = False
        repr_src = 'default(raw; no run_config repr_norm found)'
      else:
        use_repr_normalize = bool(inferred)
        repr_src = 'checkpoint(run_config.json)'
    overlay_mode, dirac_eps = _resolve_ppo_reward_overlay(
        path, args.ppo_reward_mode, float(args.ppo_dirac_eps))
    use_dirac_overlay = overlay_mode == 'dirac_target'
    use_kde_overlay   = overlay_mode == 'kde_dirac'
    if use_dirac_overlay:
      eval_fields = _make_maze_dirac_reward_field_fn(
          networks, state_dim, dirac_eps)
      overlay_src = 'dirac_target'
      if args.ppo_reward_mode:
        overlay_src += ' (cli)'
      elif _infer_run_config_from_checkpoint(path).get('ppo_reward_mode'):
        overlay_src += ' (run_config.json)'
      else:
        overlay_src += ' (path dirac_baseline)'
    elif use_kde_overlay:
      from contrastive.ppo_learner import GaussianKDE
      n_kde = int(args.kde_num_rollouts)
      print(f'[maze]   kde_dirac: collecting {n_kde} rollouts for KDE fit…',
            flush=True)
      kde_states = _collect_rollout_states(
          gym_env, policy_params, networks, max_steps, n_kde, seed=args.seed)
      bw = float(len(kde_states)) ** (-1.0 / (kde_states.shape[1] + 4))
      kde = GaussianKDE(kde_states, bandwidth=bw)
      eval_fields = _make_maze_kde_reward_field_fn(
          kde, networks, state_dim, dirac_eps)
      overlay_src = f'kde_dirac (N={len(kde_states)}, bw={bw:.4f})'
    else:
      eval_fields = eval_fields_norm if use_repr_normalize else eval_fields_raw
      overlay_src = repr_src
    print(f'[maze]   left_heatmap={overlay_src}  '
          f'repr_normalize='
          f'{use_repr_normalize if not (use_dirac_overlay or use_kde_overlay) else "n/a"}')

    # Build overlays once per checkpoint (goal is fixed-goal in point envs).
    states0, goal0, reward_sum0, success0 = _rollout_one(
        policy_params=policy_params,
        gym_env=gym_env,
        networks=networks,
        max_steps=max_steps,
        stochastic=(args.stochastic or args.num_trajectories > 1),
        seed=args.seed,
        state_dim=state_dim,
        log_actions=bool(args.log_rollout_actions),
        log_policy_stats=bool(args.log_policy_stats),
        log_prefix=f'[{label}] traj=0 ',
    )
    h_phi = h_value = None
    if not args.no_repr_overlay:
      q_params = ckpt.get('q_params')
      value_params = ckpt.get('value_params')
      if q_params is None or value_params is None:
        print('[maze]   warning: checkpoint missing q_params/value_params; '
              'skipping overlays (use --no_repr_overlay to silence).')
      else:
        h_phi, h_value = _repr_grids_over_maze(
            walls, goal0, policy_params, q_params, value_params, eval_fields,
            subcells=int(args.heatmap_subcells),
            s0_state=states0[0] if use_dirac_overlay else None,
            state_dim=state_dim)

    trjs = []       # list of (T, 2) maze-coords arrays
    extra_trjs = [] # list of (T, n_extra) extra-dim arrays (empty for 2D envs)
    metrics = []
    for traj_idx in range(int(args.num_trajectories)):
      if traj_idx == 0:
        states, goal, reward_sum, success = states0, goal0, reward_sum0, success0
      else:
        states, goal, reward_sum, success = _rollout_one(
            policy_params=policy_params,
            gym_env=gym_env,
            networks=networks,
            max_steps=max_steps,
            stochastic=(args.stochastic or args.num_trajectories > 1),
            seed=args.seed + traj_idx,
            state_dim=state_dim,
            log_actions=bool(args.log_rollout_actions),
            log_policy_stats=bool(args.log_policy_stats),
            log_prefix=f'[{label}] traj={traj_idx} ',
        )
      # Split maze coords and extra dims
      trjs.append(states[:, :2])
      if n_extra_dims > 0:
        extra_trjs.append(states[:, 2:])
      metrics.append((reward_sum, success))
      print(f'[maze]   traj={traj_idx}  length={len(states)}  '
            f'reward_sum={reward_sum:.3f}  success={success}  '
            f'goal=(row={goal[0]:.2f}, col={goal[1]:.2f})')

    out_path = _resolve_output_path(args.output, args.env, label, multi, ext='.png')
    avg_reward = float(np.mean([m[0] for m in metrics])) if metrics else 0.0
    any_success = any(m[1] for m in metrics)
    n_traj = int(args.num_trajectories)
    ttl = f'{args.env}  seed={args.seed}  ckpt={label}'
    if n_traj > 1:
      out_path = _append_suffix_to_path(out_path, f'trajset{n_traj:02d}')
      ttl += f'  stochastic trajectories={n_traj}'
    _plot_trajectory(
        walls=walls, states=trjs[0], trajectories=(trjs if n_traj > 1 else None),
        goal=goal0, title=ttl, out_path=out_path,
        reward_sum=avg_reward, success=any_success,
        heatmap_phi=h_phi, heatmap_value=h_value,
        normalize_repr=use_repr_normalize and not use_kde_overlay,
        overlay_mode=overlay_mode,
        dirac_eps=dirac_eps,
        fig_scale=float(max(0.5, args.fig_scale)),
        extra_dims_trajs=extra_trjs if n_extra_dims > 0 else None)
    print(f'[maze]   wrote {out_path}')


if __name__ == '__main__':
  main()
