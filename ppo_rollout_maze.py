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
left panel overlays a heatmap of **φ(s,a)·ψ(g)** over every free cell
(``a`` is the policy's deterministic action at ``concat([s,g])``,
matching ``ppo_learner.make_reward_fn``); the right panel overlays
**ψ(s)·ψ(g)**, where **ψ** is the goal encoder on the goal slice of the
observation, and **ψ(s)** means that encoder applied to cell-center
coordinates ``s`` placed in the goal slot (``obs = concat([s,s])``).
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

import sgcrl_jax_acme_compat  # noqa: F401  must precede acme/jax imports

import argparse
import glob
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
      env_name, fixed_start_end=fixed_goal_dict[env_name])
  walls = np.asarray(gym_env._walls)   # (H, W), 1=wall, 0=free
  return gym_env, obs_dim, max_steps, walls


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _rollout_one(policy_params, gym_env, networks,
                 max_steps: int, stochastic: bool, seed: int):
  """Run one rollout and return (states (T+1, 2), goal (2,), reward_sum, success)."""
  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  @jax.jit
  def policy_sample(params, obs, rng):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample(dist, rng)

  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  goal = obs[2:].copy()                       # (2,)
  states = [obs[:2].copy()]
  total_reward = 0.0
  success = False
  rng = jax.random.PRNGKey(seed)
  for _ in range(max_steps):
    if stochastic:
      rng, k = jax.random.split(rng)
      action_j = policy_sample(policy_params, obs[None], k)
    else:
      action_j = policy_mode(policy_params, obs[None])
    action = np.asarray(action_j)[0].astype(np.float32)
    obs_next, r, done, _ = gym_env.step(action)
    obs = np.asarray(obs_next, dtype=np.float32)
    states.append(obs[:2].copy())
    total_reward += float(r)
    if float(r) > 0.0:
      success = True
    if done:
      break
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


def _make_maze_repr_field_fn(networks, state_dim: int):
  """Jitted (policy, q, positions, goal) -> (phi_dot, psi_dot) per position."""

  @jax.jit
  def _eval(policy_p, q_p, pos_batch: jnp.ndarray, goal_vec: jnp.ndarray):
    g = jnp.broadcast_to(goal_vec[None, :], (pos_batch.shape[0], state_dim))
    obs_pg = jnp.concatenate([pos_batch, g], axis=-1)
    dist = networks.policy_network.apply(policy_p, obs_pg)
    act = networks.sample_eval(dist, jax.random.PRNGKey(0))

    _, phi, psi_g = networks.q_network.apply(q_p, obs_pg, act)
    phi_dot = jnp.sum(phi * psi_g, axis=-1)

    obs_ss = jnp.concatenate([pos_batch, pos_batch], axis=-1)
    _, _, psi_s = networks.q_network.apply(q_p, obs_ss, act)
    psi_dot = jnp.sum(psi_s * psi_g, axis=-1)
    return phi_dot, psi_dot

  return _eval


def _repr_grids_over_maze(
    walls: np.ndarray,
    goal: np.ndarray,
    policy_params,
    q_params,
    eval_fields,
) -> Tuple[np.ndarray, np.ndarray]:
  """Scalar fields on an H×W grid (NaN on walls) at cell centers."""
  H, W = walls.shape
  rows, cols = np.where(walls == 0)
  pos = np.stack(
      [rows.astype(np.float32) + 0.5, cols.astype(np.float32) + 0.5],
      axis=-1)
  gvec = jnp.asarray(goal, dtype=jnp.float32)
  pos_j = jnp.asarray(pos, dtype=jnp.float32)
  phi_dot, psi_dot = eval_fields(policy_params, q_params, pos_j, gvec)
  phi_grid = np.full((H, W), np.nan, dtype=np.float32)
  psi_grid = np.full((H, W), np.nan, dtype=np.float32)
  phi_grid[rows, cols] = np.asarray(phi_dot, dtype=np.float32)
  psi_grid[rows, cols] = np.asarray(psi_dot, dtype=np.float32)
  return phi_grid, psi_grid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _draw_maze_panel(
    ax,
    walls: np.ndarray,
    states: np.ndarray,
    goal: np.ndarray,
    heatmap: Optional[np.ndarray],
    heat_label: str,
):
  """Walls + optional heatmap + trajectory on one axis."""
  H, W = walls.shape
  ax.imshow(
      walls,
      cmap='gray_r',
      origin='upper',
      extent=[0, W, H, 0],
      interpolation='nearest',
      alpha=0.85,
  )
  if heatmap is not None:
    hm = ax.imshow(
        np.ma.masked_invalid(heatmap),
        origin='upper',
        extent=[0, W, H, 0],
        interpolation='nearest',
        alpha=0.5,
        cmap='magma',
    )
    # ``plt.colorbar(ax=ax)`` shrinks ``ax`` and breaks ``set_aspect('equal')``,
    # which visually misaligns walls / trajectory / goal.  Reserve a sibling
    # axes for the colorbar instead (see mpl_toolkits.axes_grid1 docs).
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.12)
    fig = ax.figure
    cbar = fig.colorbar(hm, cax=cax)
    cbar.ax.set_ylabel(heat_label, rotation=270, labelpad=12)

  xs = states[:, 1]
  ys = states[:, 0]
  ax.plot(xs, ys, '-', color='tab:blue', linewidth=1.8, alpha=0.95,
          label='trajectory')
  ax.plot(xs, ys, 'o', color='tab:blue', markersize=2.0, alpha=0.6)
  ax.plot(xs[0], ys[0], marker='o', color='lime', markersize=12,
          markeredgecolor='black', linewidth=0, label='start')
  ax.plot(xs[-1], ys[-1], marker='X', color='orange', markersize=12,
          markeredgecolor='black', linewidth=0, label='end')
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
    goal: np.ndarray,
    title: str,
    out_path: str,
    reward_sum: float,
    success: bool,
    heatmap_phi: Optional[np.ndarray] = None,
    heatmap_psi: Optional[np.ndarray] = None,
):
  """Save a PNG: trajectory on maze; optional side-by-side CRL heatmaps."""
  H, W = walls.shape
  h_in = 6 * H / max(W, 1)
  if heatmap_phi is not None and heatmap_psi is not None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, h_in), layout='constrained')
    supt = (
        f'{title}\n'
        f'len={len(states)}  reward_sum={reward_sum:.2f}  success={success}')
    fig.suptitle(supt, fontsize=11)
    _draw_maze_panel(
        ax0, walls, states, goal, heatmap_phi,
        r'$\phi(s,a)\cdot\psi(g)$',
    )
    ax0.set_title(r'$\phi(s,a)\cdot\psi(g)$  (policy mode $a$)')
    _draw_maze_panel(
        ax1, walls, states, goal, heatmap_psi,
        r'$\psi(s)\cdot\psi(g)$',
    )
    ax1.set_title(
        r'$\psi(s)\cdot\psi(g)$  ($\psi$ = goal encoder; $s$ in goal slot)')
  else:
    fig, ax0 = plt.subplots(figsize=(6, h_in))
    _draw_maze_panel(ax0, walls, states, goal, None, '')
    ax0.set_title(
        f'{title}\n'
        f'len={len(states)}  reward_sum={reward_sum:.2f}  success={success}')

  out_dir = os.path.dirname(os.path.abspath(out_path))
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  if heatmap_phi is None or heatmap_psi is None:
    fig.tight_layout()
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
  args = parser.parse_args()

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
  eval_fields = _make_maze_repr_field_fn(networks, state_dim)
  max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
  print(f'[maze] env={args.env}  max_steps={max_steps}  '
        f'walls shape={walls.shape} (rows, cols)')

  # ----- 3. Render one PNG per checkpoint --------------------------------
  multi = len(ckpt_entries) > 1
  for label, path in ckpt_entries:
    print(f'[maze] === {label}  ({path}) ===')
    ckpt = ppo_learner.load_checkpoint(path)
    policy_params = ckpt['policy_params']
    print(f'[maze]   iteration={ckpt.get("iteration")} '
          f'global_step={ckpt.get("global_step")}')

    states, goal, reward_sum, success = _rollout_one(
        policy_params=policy_params,
        gym_env=gym_env,
        networks=networks,
        max_steps=max_steps,
        stochastic=args.stochastic,
        seed=args.seed,
    )
    print(f'[maze]   length={len(states)}  reward_sum={reward_sum:.3f}  '
          f'success={success}  goal=(row={goal[0]:.2f}, col={goal[1]:.2f})')

    out_path = _resolve_output_path(args.output, args.env, label, multi,
                                    ext='.png')
    h_phi = h_psi = None
    if not args.no_repr_overlay:
      q_params = ckpt.get('q_params')
      if q_params is None:
        print('[maze]   warning: checkpoint has no q_params; skipping '
              'CRL heatmaps (use --no_repr_overlay to silence).')
      else:
        h_phi, h_psi = _repr_grids_over_maze(
            walls, goal, policy_params, q_params, eval_fields)
    _plot_trajectory(
        walls=walls, states=states, goal=goal,
        title=f'{args.env}  seed={args.seed}  ckpt={label}',
        out_path=out_path, reward_sum=reward_sum, success=success,
        heatmap_phi=h_phi, heatmap_psi=h_psi)
    print(f'[maze]   wrote {out_path}')


if __name__ == '__main__':
  main()
