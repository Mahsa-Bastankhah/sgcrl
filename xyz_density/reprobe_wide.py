#!/usr/bin/env python3
"""Re-score state_ood density probes on a wider x' grid from checkpoints.

Existing online probes used half_width=2. Checkpoints only store q_params (not
NF goal whitening stats), so for NF we calibrate mean/std against the saved
±2 probe scores at the same iteration before scoring the wider grid.

Example::

  python -u xyz_density/reprobe_wide.py \\
      --log_root=logs/xyz_density_az0_g05_100m \\
      --out_dir=figs/xyz_density_az0_g05_100m \\
      --half_width=10 --n_grid=201 --ood_high_xy=15
"""
from __future__ import annotations

import argparse
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
import numpy as np
from acme import specs
from scipy import optimize

from contrastive import nf_density as _nf
from contrastive import td3_density as _td3
from contrastive.networks import make_networks
from xyz_density.train import (
    _make_1d_density_probe,
    _make_fixed_ood_probes,
    _score_candidates,
    _softmax_normalize,
)

MODES = ('crl', 'nf', 'nf_compact', 'nf_tiny', 'nf_td', 'td3', 'fb', 'fm',
         'tdinfonce')
MODE_FILE_PREFIX = {
    'nf_compact': 'nf',
    'nf_tiny': 'nf',
    'nf_td': 'nf',
}
MODE_COLORS = {
    'crl': '#1f77b4',
    'nf': '#ff7f0e',
    'nf_compact': '#8c564b',
    'nf_tiny': '#e377c2',
    'nf_td': '#17becf',
    'td3': '#2ca02c',
    'fb': '#bcbd22',
    'fm': '#d62728',
    'tdinfonce': '#9467bd',
}
MODE_LABELS = {
    'crl': 'CRL',
    'nf': 'NF',
    'nf_compact': 'NF compact',
    'nf_tiny': 'NF tiny',
    'nf_td': 'TD-NF',
    'td3': 'TD3',
    'fb': 'FB',
    'fm': 'FM',
    'tdinfonce': 'TDInfoNCE',
}


def _make_env_spec(obs_dim: int, act_dim: int, goal_dim: int):
  total = obs_dim + goal_dim
  return specs.EnvironmentSpec(
      observations=specs.Array(shape=(total,), dtype=np.float32, name='obs'),
      actions=specs.BoundedArray(
          shape=(act_dim,), dtype=np.float32, name='action',
          minimum=-1.0, maximum=1.0),
      rewards=specs.Array(shape=(), dtype=np.float32, name='reward'),
      discounts=specs.BoundedArray(
          shape=(), dtype=np.float32, name='discount',
          minimum=0.0, maximum=1.0),
  )


def _build_nets(mode: str, args: dict):
  obs_dim = act_dim = goal_dim = 3
  hidden = tuple(
      int(x) for x in str(args.get('hidden_layer_sizes',
                                   '256,256,256,256,256,256')).split(',')
      if x)
  networks = nf_nets = fm_nets = td3_nets = None
  if mode in ('crl', 'tdinfonce'):
    networks = make_networks(
        _make_env_spec(obs_dim, act_dim, goal_dim),
        obs_dim=obs_dim,
        repr_dim=int(args.get('repr_dim', 64)),
        repr_norm=bool(args.get('repr_norm', False)),
        hidden_layer_sizes=hidden,
        twin_q=False if mode == 'tdinfonce' else bool(
            args.get('twin_q', False)),
    )
  elif mode == 'nf':
    nf_nets = _nf.make_nf_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        rep_size=int(args.get('nf_rep_size', 256)),
        num_blocks=int(args.get('nf_num_blocks', 12)),
        channels=int(args.get('nf_coupling_width', 512)),
        goal_enc_size=int(args.get('nf_goal_enc_size', 0)),
        sa_hidden=int(args.get('nf_sa_hidden', 1024)),
        sa_num_layers=int(args.get('nf_sa_num_layers', 4)),
        state_only=bool(args.get('nf_state_only', False)),
    )
  elif mode == 'fm':
    from contrastive import fm_density as _fm
    fm_nets = _fm.make_fm_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        hidden_layer_sizes=hidden,
        flow_steps=int(args.get('fm_flow_steps', 10)),
    )
  else:
    td3_nets = _td3.make_td3_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        hidden_layer_sizes=hidden,
        repr_dim=int(args.get('repr_dim', 64)),
        bilinear=bool(args.get('td3_bilinear', False)),
        repr_norm=bool(args.get('repr_norm', False))
        if args.get('td3_bilinear', False) else False,
    )
  return networks, nf_nets, fm_nets, td3_nets


def _file_prefix(mode: str) -> str:
  return MODE_FILE_PREFIX.get(mode, mode)


def _score_mode(mode: str) -> str:
  """Network / scoring mode name (nf_* / fb share NF / TD3 scorers)."""
  if mode in ('nf_compact', 'nf_tiny', 'nf_td'):
    return 'nf'
  if mode == 'fb':
    return 'td3'
  return mode


def _list_ckpts(mode_dir: str, mode: str) -> List[Tuple[int, str]]:
  prefix = _file_prefix(mode)
  paths = sorted(glob.glob(os.path.join(
      mode_dir, f'ckpt_{prefix}_seed*_iter*.pkl')))
  out = []
  for p in paths:
    base = os.path.basename(p)
    it = int(base.rsplit('iter', 1)[-1].split('.')[0])
    out.append((it, p))
  return sorted(out)


def _load_saved_probe(mode_dir: str, mode: str, iteration: int) -> Optional[dict]:
  prefix = _file_prefix(mode)
  path = os.path.join(
      mode_dir, 'ood_density_probe',
      f'probe_state_ood_{prefix}_iter{iteration:06d}.npz')
  if not os.path.exists(path):
    # nearest existing probe (including pre-reeval backup)
    paths = sorted(glob.glob(os.path.join(
        mode_dir, 'ood_density_probe',
        f'probe_state_ood_{prefix}_iter*.npz')))
    bak = sorted(glob.glob(os.path.join(
        mode_dir, 'ood_density_probe', 'state_ood_pre_reeval_bak',
        f'probe_state_ood_{prefix}_iter*.npz')))
    paths = paths or bak
    if not paths:
      return None
    best = min(paths, key=lambda p: abs(
        int(os.path.basename(p).rsplit('iter', 1)[-1].split('.')[0]) - iteration))
    path = best
  return dict(np.load(path, allow_pickle=True))


def _calibrate_nf_stats(
    nf_nets,
    q_params,
    state: np.ndarray,
    action: np.ndarray,
    goals: np.ndarray,
    saved_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Fit goal whitening so re-scored log-probs match saved ±2 probe scores."""
  saved = np.asarray(saved_scores, dtype=np.float64)
  saved_c = saved - saved.mean()
  p_tgt = _softmax_normalize(saved)

  def loss(theta):
    mx, my, log_sx, log_sy, log_sz = theta
    mean = np.array([mx, my, 0.0], dtype=np.float32)
    std = np.exp(np.array([log_sx, log_sy, log_sz], dtype=np.float64)).astype(
        np.float32)
    std = np.maximum(std, 0.1)
    sc = _score_candidates(
        'nf',
        networks=None,
        nf_nets=nf_nets,
        fm_nets=None,
        td3_nets=None,
        q_params=q_params,
        state=state,
        action=action,
        goals=goals,
        nf_goal_mean=mean,
        nf_goal_std=std,
    ).astype(np.float64)
    sc_c = sc - sc.mean()
    mse = float(np.mean((saved_c - sc_c) ** 2))
    q = _softmax_normalize(sc)
    kl = float(np.sum(p_tgt * np.log((p_tgt + 1e-12) / (q + 1e-12))))
    return mse + 100.0 * kl

  x0 = np.array([0.0, 0.0, np.log(5.0), np.log(5.0), np.log(0.5)])
  res = optimize.minimize(
      loss, x0, method='Nelder-Mead',
      options={'maxiter': 40, 'xatol': 1e-2, 'fatol': 1e-3})
  mx, my, log_sx, log_sy, log_sz = res.x
  mean = np.array([mx, my, 0.0], dtype=np.float32)
  std = np.exp(np.array([log_sx, log_sy, log_sz], dtype=np.float64)).astype(
      np.float32)
  std = np.maximum(std, 0.1).astype(np.float32)
  print(f'[reprobe] NF calibrate ok={res.success} loss={res.fun:.4g} '
        f'mean={mean.tolist()} std={std.tolist()}', flush=True)
  return mean, std


def _choose_iters(all_iters: List[int], max_panels: int) -> List[int]:
  if len(all_iters) <= max_panels:
    return list(all_iters)
  idx = np.linspace(0, len(all_iters) - 1, max_panels).round().astype(int)
  return [all_iters[i] for i in idx]


def reprobe_and_plot(
    log_root: str,
    out_dir: str,
    half_width: float,
    n_grid: int,
    max_panels: int,
    noise_std: float,
    high_xy: float,
) -> None:
  os.makedirs(out_dir, exist_ok=True)
  probe_out = os.path.join(out_dir, f'wide_probes_hw{half_width:g}')
  os.makedirs(probe_out, exist_ok=True)

  # Shared (s,a) from the original probe config.
  base_probes = _make_fixed_ood_probes(noise_std, high_xy=high_xy)
  base = base_probes['state_ood']
  wide = _make_1d_density_probe(
      state=base['state'],
      action=base['action'],
      noise_std=noise_std,
      axis=0,
      half_width=half_width,
      n_grid=n_grid,
      name='state_ood',
  )
  print(f"[reprobe] wide grid x' in "
        f"[{wide['grid'][0]:.3f}, {wide['grid'][-1]:.3f}]  n={n_grid}",
        flush=True)

  ckpts_by_mode: Dict[str, Dict[int, str]] = {}
  for mode in MODES:
    mode_dir = os.path.join(log_root, mode)
    ckpts = _list_ckpts(mode_dir, mode)
    if ckpts:
      ckpts_by_mode[mode] = dict(ckpts)
    else:
      print(f'[reprobe] no checkpoints for {mode}', flush=True)
  if not ckpts_by_mode:
    print('[reprobe] nothing to plot', flush=True)
    return

  # Prefer iters shared by well-covered modes so a mid-run mode (few ckpts)
  # cannot collapse the panel grid to a single iteration.
  counts = {m: len(d) for m, d in ckpts_by_mode.items()}
  max_n = max(counts.values())
  rich = [m for m, n in counts.items() if n >= max(max_panels, (max_n + 1) // 2)]
  if not rich:
    rich = list(ckpts_by_mode)
  print(f'[reprobe] ckpt counts={counts}  panel modes={rich}', flush=True)
  common = sorted(set.intersection(
      *(set(ckpts_by_mode[m].keys()) for m in rich)))
  if not common:
    # Fall back to union / nearest later; score each mode's own panel set.
    common = sorted(set.union(
        *(set(ckpts_by_mode[m].keys()) for m in rich)))
  chosen = _choose_iters(common, max_panels)
  print(f'[reprobe] panel iters {chosen}', flush=True)

  results: Dict[str, List[Tuple[int, dict]]] = {}
  for mode, ckpt_map in ckpts_by_mode.items():
    mode_dir = os.path.join(log_root, mode)
    mode_iters = []
    for it in chosen:
      if it in ckpt_map:
        mode_iters.append(it)
      else:
        nearest = min(ckpt_map.keys(), key=lambda x: abs(x - it))
        mode_iters.append(nearest)
    mode_iters = sorted(set(mode_iters))
    print(f'[reprobe] {mode}: scoring iters {mode_iters}', flush=True)
    results[mode] = []

    networks = nf_nets = fm_nets = td3_nets = None
    s_mode = _score_mode(mode)
    for it in mode_iters:
      ckpt_path = ckpt_map[it]
      with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
      args = ckpt['args']
      q_params = ckpt['q_params']
      if (networks is None and nf_nets is None and fm_nets is None
          and td3_nets is None):
        networks, nf_nets, fm_nets, td3_nets = _build_nets(s_mode, args)

      nf_mean = np.zeros(3, dtype=np.float32)
      nf_std = np.ones(3, dtype=np.float32)
      if s_mode == 'nf':
        saved = _load_saved_probe(mode_dir, mode, it)
        if saved is None:
          print(f'[reprobe] WARNING: no saved probe to calibrate NF iter={it}',
                flush=True)
        else:
          g_old = np.asarray(
              saved['grid'] if 'grid' in saved else saved['z_grid'])
          goals_old = np.broadcast_to(
              np.asarray(saved['mu'])[None, :], (len(g_old), 3)).copy()
          goals_old = goals_old.astype(np.float32)
          goals_old[:, 0] = g_old.astype(np.float32)
          nf_mean, nf_std = _calibrate_nf_stats(
              nf_nets, q_params,
              np.asarray(saved['state']),
              np.asarray(saved['action']),
              goals_old,
              np.asarray(saved['scores']),
          )

      scores = _score_candidates(
          s_mode,
          networks=networks,
          nf_nets=nf_nets,
          fm_nets=fm_nets,
          td3_nets=td3_nets,
          q_params=q_params,
          state=wide['state'],
          action=wide['action'],
          goals=wide['goals'],
          nf_goal_mean=nf_mean,
          nf_goal_std=nf_std,
      )
      prob = _softmax_normalize(scores)
      data = {
          'iteration': it,
          'global_step': int(ckpt.get('global_step', -1)),
          'mode': mode,
          'probe_name': 'state_ood',
          'state': wide['state'],
          'action': wide['action'],
          'mu': wide['mu'],
          'axis': wide['axis'],
          'grid': wide['grid'],
          'scores': scores.astype(np.float32),
          'prob': prob,
          'gt_prob': wide['gt_prob'],
          'half_width': np.asarray(half_width),
          'nf_goal_mean': nf_mean,
          'nf_goal_std': nf_std,
      }
      out_path = os.path.join(
          probe_out,
          f'probe_state_ood_{mode}_iter{it:06d}_hw{half_width:g}.npz')
      np.savez_compressed(out_path, **data)
      results[mode].append((it, data))
      print(f'[reprobe] wrote {out_path}', flush=True)

  if not results:
    print('[reprobe] nothing to plot', flush=True)
    return

  all_iters = sorted({it for lst in results.values() for it, _ in lst})
  chosen = _choose_iters(all_iters, max_panels)
  any_data = next(iter(results.values()))[0][1]
  action = np.asarray(any_data['action'])
  mu = np.asarray(any_data['mu'])
  grid = np.asarray(any_data['grid'])
  gt = np.asarray(any_data['gt_prob'])

  # Floor for log-y (softmax probs can hit exact 0 numerically).
  y_floor = 1e-6

  def _ylog(y):
    return np.maximum(np.asarray(y, dtype=np.float64), y_floor)

  n = len(chosen)
  fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
  if n == 1:
    axes = [axes]
  for ax, it in zip(axes, chosen):
    ax.plot(grid, _ylog(gt), color='k', linestyle='--', linewidth=1.5,
            label='true 1-step N' if it == chosen[0] else None)
    for mode, lst in results.items():
      nearest = min(lst, key=lambda t: abs(t[0] - it))
      data = nearest[1]
      ax.plot(
          np.asarray(data['grid']), _ylog(data['prob']),
          color=MODE_COLORS.get(mode),
          linewidth=2.0,
          label=(MODE_LABELS.get(mode, mode.upper())
                 if it == chosen[0] else None),
      )
    ax.set_xlabel("x'")
    ax.set_title(f'iter {it}')
    ax.set_yscale('log')
    ax.set_ylim(y_floor, 1.0)
    ax.grid(True, alpha=0.3, which='both')
  axes[0].set_ylabel(r"normalized $\hat p(x'\mid s,a)$")
  axes[0].legend(frameon=False, fontsize=8)
  fig.suptitle(
      f'state_ood (±{half_width:g})  s={np.asarray(any_data["state"]).tolist()}  '
      f'a=[{action[0]:.3f},{action[1]:.3f},{action[2]:.3f}]  '
      f'μ=[{mu[0]:.3f},{mu[1]:.3f},{mu[2]:.3f}]',
      fontsize=10)
  fig.tight_layout()
  # ±2 → unsuffixed (plot_results naming). Wider → *_hw{N}.png, and also
  # write unsuffixed aliases so callers that expect those names stay in sync.
  if abs(float(half_width) - 2.0) < 1e-9:
    over_names = ['ood_state_density_over_time.png']
    latest_names = ['ood_state_density_latest.png']
  else:
    hw = f'{half_width:g}'
    over_names = [
        f'ood_state_density_over_time_hw{hw}.png',
        'ood_state_density_over_time.png',
    ]
    latest_names = [
        f'ood_state_density_latest_hw{hw}.png',
        'ood_state_density_latest.png',
    ]
  for over_name in over_names:
    out_path = os.path.join(out_dir, over_name)
    fig.savefig(out_path, dpi=160)
    print(f'[reprobe] wrote {out_path}', flush=True)
  plt.close(fig)

  # Latest panel
  fig, ax = plt.subplots(figsize=(6.5, 4.2))
  ax.plot(grid, _ylog(gt), 'k--', linewidth=1.8, label='true 1-step N')
  for mode, lst in results.items():
    data = lst[-1][1]
    ax.plot(
        np.asarray(data['grid']), _ylog(data['prob']),
        color=MODE_COLORS.get(mode),
        linewidth=2.2,
        label=(f'{MODE_LABELS.get(mode, mode.upper())} '
               f'(iter {int(data["iteration"])})'),
    )
  ax.set_xlabel("x'")
  ax.set_ylabel(r"normalized $\hat p(x'\mid s,a)$")
  ax.set_title(f'Latest state_ood density probe (±{half_width:g})')
  ax.set_yscale('log')
  ax.set_ylim(y_floor, 1.0)
  ax.grid(True, alpha=0.3, which='both')
  ax.legend(frameon=False)
  fig.tight_layout()
  for latest_name in latest_names:
    out_path = os.path.join(out_dir, latest_name)
    fig.savefig(out_path, dpi=160)
    print(f'[reprobe] wrote {out_path}', flush=True)
  plt.close(fig)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--log_root', type=str,
                 default='logs/xyz_density_az0_g05_100m')
  p.add_argument('--out_dir', type=str,
                 default='figs/xyz_density_az0_g05_100m')
  p.add_argument('--half_width', type=float, default=10.0)
  p.add_argument('--n_grid', type=int, default=201)
  p.add_argument('--max_panels', type=int, default=6)
  p.add_argument('--noise_std', type=float, default=0.01)
  p.add_argument('--ood_high_xy', type=float, default=15.0)
  args = p.parse_args()
  reprobe_and_plot(
      log_root=args.log_root,
      out_dir=args.out_dir,
      half_width=float(args.half_width),
      n_grid=int(args.n_grid),
      max_panels=int(args.max_panels),
      noise_std=float(args.noise_std),
      high_xy=float(args.ood_high_xy),
  )


if __name__ == '__main__':
  main()
