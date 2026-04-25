#!/usr/bin/env python3
"""RiverSwim checkpoint rollouts: chain MDP + φ(s,a)·ψ(g) heatmap + trajectory.

For each checkpoint, runs one deterministic-policy rollout and draws:

  * Y-axis: river cells (chain / MDP states), x-axis: time.
  * Pixel (i, t): counterfactual **φ(s=i, a_t) · ψ(g_t)** with the episode's
    goal ``g_t`` and action ``a_t`` at step ``t``, but state one-hot replaced
    by cell ``i`` (so you see how the critic orders states along the chain).
  * Trajectory overlaid: actual visited state before each step.

Example::

  python plot_ppo_riverswim_ckpts.py \\
      --checkpoint_dir=logs/ppo_riverswim/ppo_riverswim_7/checkpoints \\
      --output=figures/riverswim_ckpt_compare.png \\
      --seed=7 \\
      --labels=iter_0000000,iter_0000250,iter_0000500,latest
"""
import argparse
import os
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import sgcrl_jax_acme_compat  # noqa: F401

import env_utils
from contrastive import ppo_learner
from ppo_contrastive import fixed_goal_dict
from ppo_rollout_video import _build_networks, _enumerate_checkpoints


def _pick_labels(
    entries: List[Tuple[str, str]],
    labels_csv: Optional[str],
    max_plots: int,
):
  if labels_csv:
    want = {x.strip() for x in labels_csv.split(',') if x.strip()}
    picked = [(lab, p) for lab, p in entries if lab in want]
    missing = want - {lab for lab, _ in picked}
    if missing:
      avail = [e[0] for e in entries]
      raise SystemExit(
          f'Unknown label(s): {sorted(missing)}.\n'
          f'Available labels (from directory): {avail}\n'
          f'Fix ``--labels`` or omit it to use ``--max_plots`` auto-selection.')
    return picked
  if max_plots >= len(entries):
    return entries
  idxs = sorted(set(
      int(np.round(i * (len(entries) - 1) / max(1, max_plots - 1)))
      for i in range(max_plots)))
  return [entries[i] for i in idxs]


def _rollout_with_obs(networks, policy_params, q_params, gym_env, obs_dim: int,
                      max_steps: int):
  """Rollout; returns obs before each step, actions, and state indices."""

  @jax.jit
  def policy_mode(params, obs):
    dist = networks.policy_network.apply(params, obs)
    return networks.sample_eval(dist, jax.random.PRNGKey(0))

  obs = np.asarray(gym_env.reset(), dtype=np.float32)
  s_idx = [int(np.argmax(obs[:obs_dim]))]
  obs_pres = []
  acts = []
  for _ in range(max_steps):
    obs_pres.append(obs.copy())
    obs_j = jnp.asarray(obs)[None]
    act_j = policy_mode(policy_params, obs_j)
    act = np.asarray(act_j[0], dtype=np.float32)
    acts.append(act)
    obs_next, _, done, _ = gym_env.step(act)
    obs = np.asarray(obs_next, dtype=np.float32)
    s_idx.append(int(np.argmax(obs[:obs_dim])))
    if done:
      break
  if not obs_pres:
    adim = int(np.asarray(gym_env.action_space.shape).prod()) or 1
    return {
        'obs_pres': np.zeros((0, 2 * obs_dim), dtype=np.float32),
        'acts': np.zeros((0, adim), dtype=np.float32),
        's_idx': np.asarray(s_idx, dtype=np.int32),
        'T': 0,
    }
  return {
      'obs_pres': np.stack(obs_pres, axis=0),
      'acts': np.stack(acts, axis=0),
      's_idx': np.asarray(s_idx, dtype=np.int32),
      'T': len(obs_pres),
  }


def _counterfactual_phi_dot_psi_matrix(
    networks,
    q_params,
    obs_pres: np.ndarray,
    acts: np.ndarray,
    obs_dim: int,
) -> np.ndarray:
  """Shape (L, T): dot for hypothetical state i, true g_t and a_t."""

  L = obs_dim
  t_steps = obs_pres.shape[0]
  if t_steps == 0 or q_params is None:
    return np.full((L, t_steps), np.nan, dtype=np.float32)

  @jax.jit
  def batched_dot(params, obs_bat, act_bat):
    _, sa, gg = networks.q_network.apply(params, obs_bat, act_bat)
    return jnp.sum(sa * gg, axis=-1)

  out = np.zeros((L, t_steps), dtype=np.float32)
  eye = np.eye(L, dtype=np.float32)
  for t in range(t_steps):
    g = obs_pres[t, L:2 * L]
    obs_b = np.concatenate([eye, np.broadcast_to(g, (L, L))], axis=1)
    act_b = np.broadcast_to(acts[t], (L,) + acts[t].shape)
    out[:, t] = np.asarray(
        batched_dot(q_params, jnp.asarray(obs_b), jnp.asarray(act_b)))
  return out


def _plot_one_ckpt(
    ax,
    label: str,
    heat: np.ndarray,
    s_idx: np.ndarray,
    obs_dim: int,
    cmap: str,
    vmin,
    vmax,
):
  """Draw chain heatmap + trajectory on ``ax`` (matplotlib axes)."""
  L, t_steps = heat.shape
  if t_steps == 0:
    ax.text(0.5, 0.5, 'no steps', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(label)
    return None

  # origin='lower': row 0 at bottom = leftmost river cell (index 0).
  im = ax.imshow(
      heat,
      aspect='auto',
      interpolation='nearest',
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
      extent=(0, t_steps, -0.5, L - 0.5),
      origin='lower',
  )

  traj_x = np.arange(t_steps, dtype=np.float32) + 0.5
  traj_y = s_idx[:t_steps].astype(np.float32)
  ax.plot(traj_x, traj_y, color='white', linewidth=2.2, alpha=0.95, zorder=5)
  ax.plot(traj_x, traj_y, color='black', linewidth=1.0, alpha=0.9, zorder=6)
  ax.scatter(traj_x, traj_y, c='none', edgecolors='black', s=28, linewidths=1.0,
             zorder=7)

  ax.set_yticks(np.arange(L))
  ax.set_yticklabels([str(i) for i in range(L)])
  ax.set_ylabel('river cell (0 = left end)')
  ax.set_xlabel('time step')
  ax.set_title(label, fontsize=10)
  ax.set_xlim(0, t_steps)

  # Thin chain sketch on the left margin (optional tick line)
  for yi in range(L):
    ax.axhline(yi - 0.5, color='0.85', linewidth=0.6, zorder=0)
  return im


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint_dir', type=str, required=True)
  p.add_argument('--output', type=str, required=True,
                 help='Output .png path (parent dirs created).')
  p.add_argument('--env', type=str, default='riverswim')
  p.add_argument('--seed', type=int, default=0,
                 help='Env + network probe seed (match training seed).')
  p.add_argument('--max_steps', type=int, default=-1,
                 help='Cap rollout length; -1 = env default.')
  p.add_argument('--labels', type=str, default='')
  p.add_argument('--max_plots', type=int, default=4)
  p.add_argument('--cmap', type=str, default='magma',
                 help='Colormap for φ·ψ heatmap.')
  args = p.parse_args()

  if args.env != 'riverswim':
    raise SystemExit('This script is tailored to riverswim (gym stepping).')

  entries = _enumerate_checkpoints(args.checkpoint_dir)
  if not entries:
    raise SystemExit(f'No checkpoints under {args.checkpoint_dir!r}.')

  picked = _pick_labels(entries, args.labels or None, args.max_plots)
  print('Plotting:', [lab for lab, _ in picked])

  gym_env, obs_dim, env_max = env_utils.load(
      args.env, fixed_start_end=fixed_goal_dict[args.env], seed=args.seed)
  max_steps = env_max if args.max_steps < 0 else int(args.max_steps)

  networks, obs_dim_net = _build_networks(args.env, seed=args.seed)
  if obs_dim_net != obs_dim:
    print(f'WARNING: obs_dim gym={obs_dim} vs probe={obs_dim_net}')

  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  per_ckpt = []
  for label, path in picked:
    ck = ppo_learner.load_checkpoint(path)
    pol = ck['policy_params']
    q_p = ck.get('q_params')
    if q_p is None:
      print(f'WARNING: {label} has no q_params; heatmap will be empty.')
    tr = _rollout_with_obs(networks, pol, q_p, gym_env, obs_dim, max_steps)
    heat = _counterfactual_phi_dot_psi_matrix(
        networks, q_p, tr['obs_pres'], tr['acts'], obs_dim)
    per_ckpt.append((label, heat, tr['s_idx']))

  # Shared color scale across panels (robust to NaNs)
  finite_parts = [h[np.isfinite(h)].ravel() for _, h, _ in per_ckpt if h.size]
  finite = np.concatenate(finite_parts) if finite_parts else np.array([])
  if finite.size:
    lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
    if lo >= hi:
      hi = lo + 1e-6
  else:
    lo, hi = 0.0, 1.0

  n = len(picked)
  fig_w = min(22, 3.2 * n + 2)
  fig_h = min(10, 4.0 + 0.04 * max((h.shape[1] for _, h, _ in per_ckpt), default=1))
  fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
  axes = axes[0]
  ims = []
  for ax, (label, heat, s_idx) in zip(axes, per_ckpt):
    im = _plot_one_ckpt(ax, label, heat, s_idx, obs_dim, args.cmap, lo, hi)
    if im is not None:
      ims.append(im)

  if ims:
    fig.colorbar(
        ims[0], ax=axes.tolist() if hasattr(axes, 'tolist') else list(axes),
        fraction=0.035, pad=0.02, label='φ(s=i, a_t) · ψ(g_t)')
  fig.suptitle(
      f'{args.env}  seed={args.seed}  |  heatmap = counterfactual dots over '
      f'chain states; line = actual state  (repr from q_params)',
      fontsize=10, y=1.02)
  fig.tight_layout()
  out = os.path.abspath(args.output)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  fig.savefig(out, dpi=160, bbox_inches='tight')
  print('Wrote', out)


if __name__ == '__main__':
  main()
