#!/usr/bin/env python3
"""Action-decomposition OOD eval on existing (joint-XY) xyz density checkpoints.

Protocol (models trained with free ``a_x,a_y``, ``a_z=0``):

  Fix state ``s=(x,y,0)`` and goal ``g=(x+Δ, y+Δ, 0)`` with ``Δ=0.5``.
  Constrain ``a_y=0``, ``a_z=0``, and sweep ``a_x`` over a discrete grid.
  Score ``log p(g|s,a)`` (or CRL/TD3 surrogate) for each ``a_x`` and pick
  ``a_x* = argmax``.  Ground truth under the 1-step Gaussian dynamics is
  ``a_x*=Δ=0.5`` (y residual is fixed at ``Δ`` when ``a_y=0``).

  Categorical accuracy = fraction of states where the model's argmax lands
  on the ``0.5`` bin — tests whether the density still attributes the
  needed x-displacement to ``a_x`` even though ``a_y`` is clamped to 0
  (action decomposition).

  Also reports the discrete posterior mean
    ``E[a_x] = Σ_ax w(ax) · ax / Z``, ``Z = Σ_ax w(ax)``,
  with weights from the model score (see ``_score_to_weights``).

Example::

  python -u xyz_density/eval_ax_decompose.py \\
      --log_root=logs/xyz_density_az0_g05_100m \\
      --out_dir=figs/xyz_density_az0_g05_100m \\
      --delta=0.5 --n_states=256
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

from contrastive import fm_density as _fm
from contrastive import nf_density as _nf
from xyz_density.env import JaxXYZVecEnv
from xyz_density.reeval_state_ood import _estimate_nf_goal_stats
from xyz_density.reprobe_wide import (
    MODE_COLORS,
    MODE_LABELS,
    MODES,
    _build_nets,
    _file_prefix,
    _list_ckpts,
    _score_mode,
)
from xyz_density.train import _analytic_gaussian_log_prob


def _score_actions(
    mode: str,
    *,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    state: np.ndarray,
    actions: np.ndarray,
    goal: np.ndarray,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> np.ndarray:
  """Score many actions under a fixed (s, g). Returns shape (N,).

  NF/FM → log p.  CRL/TDInfoNCE → φ·ψ logit.  TD3/FB → raw Q (density /
  occupancy), not log Q.
  """
  n = int(actions.shape[0])
  s = jnp.broadcast_to(jnp.asarray(state)[None, :], (n, state.shape[-1]))
  a = jnp.asarray(actions)
  g = jnp.broadcast_to(jnp.asarray(goal)[None, :], (n, goal.shape[-1]))
  if mode in ('crl', 'tdinfonce'):
    packed = jnp.concatenate([s, g], axis=-1)
    _, sa_repr, g_repr = networks.q_network.apply(q_params, packed, a)
    return np.asarray(jnp.sum(sa_repr * g_repr, axis=-1))
  if mode == 'nf':
    g_norm = (g - jnp.asarray(nf_goal_mean)) / (
        jnp.asarray(nf_goal_std) + 1e-8)
    return np.asarray(_nf.nf_log_prob(nf_nets, q_params, s, a, g_norm))
  if mode == 'fm':
    return np.asarray(
        _fm.fm_log_prob(fm_nets, q_params, s, a, g, mode='exact'))
  packed = jnp.concatenate([s, g], axis=-1)
  return np.asarray(td3_nets.qf1_net.apply(q_params['qf1'], packed, a))


def _score_to_weights(score_mode: str, scores: np.ndarray) -> np.ndarray:
  """Unnormalized weights w(a_x) for discrete posterior over the ax grid.

  - NF / FM: score = log p → w = exp(score)  (stable max-subtract)
  - CRL / TDInfoNCE: score = φ·ψ logit → same exp / softmax convention
  - TD3 / FB: score = raw Q density/occupancy → w = max(Q, 0)  (no exp)

  ``score_mode`` is the network scorer from ``_score_mode`` (fb → 'td3').
  """
  x = np.asarray(scores, dtype=np.float64).reshape(-1)
  if score_mode == 'td3':  # covers fb (mapped) and td3
    return np.maximum(x, 0.0)
  x = x - np.max(x)
  return np.exp(x)


def _posterior_mean_ax(
    score_mode: str, scores: np.ndarray, ax_grid: np.ndarray,
) -> float:
  """E[a_x] = Σ w(ax)·ax / Z with Z = Σ w(ax)."""
  w = _score_to_weights(score_mode, scores)
  z = float(np.sum(w))
  if not np.isfinite(z) or z <= 0.0:
    return float('nan')
  return float(np.dot(w, np.asarray(ax_grid, dtype=np.float64)) / z)


def _ax_grid(delta: float, n_grid: int) -> np.ndarray:
  """Symmetric ax sweep in [-1, 1] that always includes ``delta`` exactly."""
  grid = np.linspace(-1.0, 1.0, int(n_grid), dtype=np.float32)
  # Snap nearest bin to delta so the GT answer is on-grid.
  i = int(np.argmin(np.abs(grid - float(delta))))
  grid[i] = np.float32(delta)
  return grid


def _eval_ckpt(
    *,
    score_mode: str,
    networks,
    nf_nets,
    fm_nets,
    td3_nets,
    q_params,
    states: np.ndarray,
    ax_grid: np.ndarray,
    delta: float,
    noise_std: float,
    nf_goal_mean: np.ndarray,
    nf_goal_std: np.ndarray,
) -> Dict[str, float]:
  """Per-state argmax-ax accuracy + posterior-mean ax."""
  n_states = int(states.shape[0])
  n_ax = int(ax_grid.shape[0])
  gt_idx = int(np.argmin(np.abs(ax_grid - float(delta))))
  correct = 0
  gt_correct = 0
  model_ax = np.zeros(n_states, dtype=np.float32)
  mean_ax = np.zeros(n_states, dtype=np.float64)
  gt_mean_ax = np.zeros(n_states, dtype=np.float64)
  gt_ax = np.zeros(n_states, dtype=np.float32)
  score_at_gt = np.zeros(n_states, dtype=np.float64)
  score_margin = np.zeros(n_states, dtype=np.float64)  # score(gt)-max_other

  for i in range(n_states):
    s = states[i]
    g = np.array([s[0] + delta, s[1] + delta, s[2]], dtype=np.float32)
    actions = np.zeros((n_ax, 3), dtype=np.float32)
    actions[:, 0] = ax_grid
    # ay=0, az=0 already

    scores = _score_actions(
        score_mode,
        networks=networks,
        nf_nets=nf_nets,
        fm_nets=fm_nets,
        td3_nets=td3_nets,
        q_params=q_params,
        state=s,
        actions=actions,
        goal=g,
        nf_goal_mean=nf_goal_mean,
        nf_goal_std=nf_goal_std,
    )
    pred = int(np.argmax(scores))
    model_ax[i] = ax_grid[pred]
    mean_ax[i] = _posterior_mean_ax(score_mode, scores, ax_grid)
    correct += int(pred == gt_idx)
    score_at_gt[i] = float(scores[gt_idx])
    others = np.delete(scores, gt_idx)
    score_margin[i] = float(scores[gt_idx] - np.max(others))

    # Analytic GT (xy only; z frozen) — log p, so exp-normalize.
    gt_scores = _analytic_gaussian_log_prob(
        np.broadcast_to(s[None, :], (n_ax, 3)),
        actions,
        np.broadcast_to(g[None, :], (n_ax, 3)),
        noise_std,
        active_dims=slice(0, 2),
    )
    gt_pred = int(np.argmax(gt_scores))
    gt_ax[i] = ax_grid[gt_pred]
    gt_mean_ax[i] = _posterior_mean_ax('nf', gt_scores, ax_grid)
    gt_correct += int(gt_pred == gt_idx)

  return {
      'cat_acc': correct / max(n_states, 1),
      'gt_cat_acc': gt_correct / max(n_states, 1),
      'mean_pred_ax': float(np.mean(model_ax)),
      'mean_abs_err_ax': float(np.mean(np.abs(model_ax - delta))),
      'mean_ax': float(np.nanmean(mean_ax)),
      'mean_abs_err_mean_ax': float(np.nanmean(np.abs(mean_ax - delta))),
      'gt_mean_ax': float(np.nanmean(gt_mean_ax)),
      'mean_score_at_gt_ax': float(np.mean(score_at_gt)),
      'mean_score_margin': float(np.mean(score_margin)),
      'frac_pred_near_gt': float(np.mean(np.abs(model_ax - delta) < 0.05)),
  }


def _sample_states(
    n: int, seed: int, noise_std: float, episode_length: int = 50,
) -> np.ndarray:
  """In-dist states from zero_az joint-XY rollouts (matches az0 training)."""
  env = JaxXYZVecEnv(
      num_envs=min(128, n),
      episode_length=episode_length,
      noise_std=noise_std,
      seed=seed,
      zero_az=True,
      single_axis_xy=False,
  )
  rng = np.random.default_rng(seed + 1)
  obs = env.reset()
  buf = [obs.copy()]
  for _ in range(episode_length - 1):
    a = env.sample_uniform_actions(rng)
    obs, _, _, _, _ = env.step(a)
    buf.append(obs.copy())
  all_s = np.concatenate(buf, axis=0)  # (T*E, 3)
  idx = rng.choice(all_s.shape[0], size=n, replace=(all_s.shape[0] < n))
  return all_s[idx].astype(np.float32)


def run_mode(
    mode: str,
    *,
    log_root: str,
    out_dir: str,
    delta: float,
    n_states: int,
    n_ax: int,
    noise_std: float,
    seed: int,
    max_ckpts: int,
) -> Optional[str]:
  mode_dir = os.path.join(log_root, mode)
  ckpts = _list_ckpts(mode_dir, mode)
  if not ckpts:
    print(f'[axdec] no checkpoints for {mode}', flush=True)
    return None
  if max_ckpts > 0 and len(ckpts) > max_ckpts:
    idx = np.linspace(0, len(ckpts) - 1, max_ckpts).round().astype(int)
    ckpts = [ckpts[i] for i in sorted(set(idx.tolist()))]

  ax_grid = _ax_grid(delta, n_ax)
  states = _sample_states(n_states, seed=seed, noise_std=noise_std)
  print(f'[axdec] {mode}: {len(ckpts)} ckpts  n_states={n_states}  '
        f'ax_grid includes {delta} at idx='
        f'{int(np.argmin(np.abs(ax_grid - delta)))}  '
        f'n_ax={len(ax_grid)}', flush=True)

  score_mode = _score_mode(mode)
  nf_mean = np.zeros(3, dtype=np.float32)
  nf_std = np.ones(3, dtype=np.float32)
  if score_mode == 'nf':
    nf_mean, nf_std = _estimate_nf_goal_stats(
        discount=0.5, noise_std=noise_std, seed=seed)

  networks = nf_nets = fm_nets = td3_nets = None
  rows = []
  for it, path in ckpts:
    with open(path, 'rb') as f:
      ckpt = pickle.load(f)
    args = ckpt['args']
    q_params = ckpt['q_params']
    if (networks is None and nf_nets is None and fm_nets is None
        and td3_nets is None):
      networks, nf_nets, fm_nets, td3_nets = _build_nets(score_mode, args)

    metrics = _eval_ckpt(
        score_mode=score_mode,
        networks=networks,
        nf_nets=nf_nets,
        fm_nets=fm_nets,
        td3_nets=td3_nets,
        q_params=q_params,
        states=states,
        ax_grid=ax_grid,
        delta=delta,
        noise_std=noise_std,
        nf_goal_mean=nf_mean,
        nf_goal_std=nf_std,
    )
    row = {
        'iteration': it,
        'global_step': int(ckpt.get('global_step', -1)),
        'mode': mode,
        **metrics,
    }
    rows.append(row)
    print(f'[axdec] {mode} iter={it}  cat_acc={metrics["cat_acc"]:.3f}  '
          f'mean_ax={metrics["mean_ax"]:.3f}  '
          f'mean_pred_ax={metrics["mean_pred_ax"]:.3f}  '
          f'margin={metrics["mean_score_margin"]:.3f}  '
          f'gt_acc={metrics["gt_cat_acc"]:.3f}', flush=True)

  os.makedirs(out_dir, exist_ok=True)
  csv_path = os.path.join(out_dir, f'ax_decompose_{mode}.csv')
  with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  print(f'[axdec] wrote {csv_path}', flush=True)
  return csv_path


def plot_results(out_dir: str, delta: float, n_ax: int) -> None:
  """Overlay cat-acc and posterior-mean a_x curves across modes."""
  paths = sorted(glob.glob(os.path.join(out_dir, 'ax_decompose_*.csv')))
  if not paths:
    print('[axdec] no CSVs to plot', flush=True)
    return

  chance = 1.0 / float(n_ax)
  fig1, ax1 = plt.subplots(figsize=(8.2, 4.5))
  fig2, ax2 = plt.subplots(figsize=(8.2, 4.5))
  for path in paths:
    mode = os.path.basename(path)[len('ax_decompose_'):-4]
    with open(path, 'r', newline='') as f:
      rows = list(csv.DictReader(f))
    if not rows or 'mean_ax' not in rows[0]:
      if rows and 'mean_ax' not in rows[0]:
        print(f'[axdec] skip {path}: no mean_ax column (re-run eval)',
              flush=True)
      continue
    x = np.array([float(r['global_step']) for r in rows])
    acc = np.array([float(r['cat_acc']) for r in rows])
    mean_ax = np.array([float(r['mean_ax']) for r in rows])
    color = MODE_COLORS.get(mode)
    label = MODE_LABELS.get(mode, mode)
    ax1.plot(x, acc, color=color, linewidth=2.0, label=label)
    ax2.plot(x, mean_ax, color=color, linewidth=2.0, label=label)

  ax1.axhline(chance, color='0.45', linestyle='--', linewidth=1.5,
              label=f'chance 1/{n_ax}={chance:.3f}')
  ax1.axhline(1.0, color='0.7', linestyle=':', linewidth=1.0)
  ax1.set_xlabel('env steps')
  ax1.set_ylabel('categorical accuracy')
  ax1.set_title(
      rf'Ax-decompose OOD: $\arg\max_{{a_x}} \log p(x+{delta:g},y+{delta:g}'
      rf'\mid s,a_x,a_y{chr(61)}0)$  (correct $a_x={delta:g}$)')
  ax1.set_ylim(-0.02, 1.02)
  ax1.grid(True, alpha=0.3)
  ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
  fig1.tight_layout(rect=(0, 0, 0.82, 1))
  p1 = os.path.join(out_dir, 'cat_acc_ax_decompose.png')
  fig1.savefig(p1, dpi=160, bbox_inches='tight')
  plt.close(fig1)
  print(f'[axdec] wrote {p1}', flush=True)

  ax2.axhline(delta, color='0.45', linestyle='--', linewidth=1.5,
              label=rf'GT $a_x={delta:g}$')
  ax2.set_xlabel('env steps')
  ax2.set_ylabel(rf'mean $E[a_x]$  (correct ${delta:g}$)')
  ax2.set_title(
      rf'Ax-decompose OOD: $E[a_x]=\sum w(a_x)\,a_x/Z$, '
      rf'$w\propto p(x+{delta:g},y+{delta:g}\mid s,a_x,a_y{chr(61)}0)$')
  ax2.grid(True, alpha=0.3)
  ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
  fig2.tight_layout(rect=(0, 0, 0.82, 1))
  # Keep legacy filename so existing fig links still resolve.
  p2 = os.path.join(out_dir, 'ax_decompose_abs_err.png')
  fig2.savefig(p2, dpi=160, bbox_inches='tight')
  plt.close(fig2)
  print(f'[axdec] wrote {p2} (posterior mean ax)', flush=True)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--log_root', type=str,
                 default='logs/xyz_density_az0_g05_100m')
  p.add_argument('--out_dir', type=str,
                 default='figs/xyz_density_az0_g05_100m')
  p.add_argument('--modes', type=str, default=','.join(MODES),
                 help='Comma-separated mode dirs under log_root.')
  p.add_argument('--delta', type=float, default=0.5,
                 help='Target displacement; GT ax*=delta with ay=0.')
  p.add_argument('--n_states', type=int, default=256)
  p.add_argument('--n_ax', type=int, default=21,
                 help='Number of ax candidates in [-1,1] (includes delta).')
  p.add_argument('--noise_std', type=float, default=0.01)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--max_ckpts', type=int, default=0,
                 help='If >0, subsample this many checkpoints per mode.')
  p.add_argument('--plot_only', action='store_true')
  args = p.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  modes = [m.strip() for m in args.modes.split(',') if m.strip()]

  if not args.plot_only:
    for mode in modes:
      run_mode(
          mode,
          log_root=args.log_root,
          out_dir=args.out_dir,
          delta=float(args.delta),
          n_states=int(args.n_states),
          n_ax=int(args.n_ax),
          noise_std=float(args.noise_std),
          seed=int(args.seed),
          max_ckpts=int(args.max_ckpts),
      )

  plot_results(args.out_dir, delta=float(args.delta), n_ax=int(args.n_ax))
  print('[axdec] done', flush=True)


if __name__ == '__main__':
  main()
