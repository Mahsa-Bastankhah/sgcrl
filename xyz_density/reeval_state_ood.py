#!/usr/bin/env python3
"""Re-evaluate OOD cat-acc (+ optional density probes) from checkpoints.

Training weights are unchanged; only validation settings differ:
  - state_ood: start at (ood_high_xy, ood_high_xy, 0), a_z=0
  - az_ood:    start ~0, a ~ Uniform[-1,1]^3  (a_z free; train used a_z=0)
  - axy_ood:   start ~0, a_x,a_y ~ Uniform[axy_low,axy_high], a_z=0
               (train used a_x,a_y ~ Uniform[-1,1])

Example::

  python -u xyz_density/reeval_state_ood.py \\
      --log_root=logs/xyz_density_az0_g05_100m \\
      --target=axy_ood \\
      --val_batch_size=256 --discount=0.5
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import shutil
import sys
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import numpy as np

from xyz_density.env import JaxXYZVecEnv
from xyz_density.reprobe_wide import _build_nets
from xyz_density.train import (
    _categorical_accuracy_on_env,
    _collect_random_episodes,
    _make_fixed_ood_probes,
    dump_ood_density_probe,
)

MODES = ('crl', 'nf', 'nf_compact', 'nf_tiny', 'nf_td', 'td3', 'fb', 'fm',
         'tdinfonce')
# Directory tag → checkpoint / metrics / probe file prefix.
MODE_FILE_PREFIX = {
    'nf_compact': 'nf',
    'nf_tiny': 'nf',
    'nf_td': 'nf',
}


def _file_prefix(mode: str) -> str:
  return MODE_FILE_PREFIX.get(mode, mode)


def _list_ckpts(mode_dir: str, mode: str) -> List[Tuple[int, str]]:
  prefix = _file_prefix(mode)
  paths = sorted(glob.glob(os.path.join(
      mode_dir, f'ckpt_{prefix}_seed*_iter*.pkl')))
  out = []
  for p in paths:
    it = int(os.path.basename(p).rsplit('iter', 1)[-1].split('.')[0])
    out.append((it, p))
  return sorted(out)


def _find_metrics_csv(mode_dir: str, mode: str) -> Optional[str]:
  prefix = _file_prefix(mode)
  matches = sorted(glob.glob(
      os.path.join(mode_dir, f'metrics_{prefix}_seed*.csv')))
  return matches[-1] if matches else None


def _estimate_nf_goal_stats(
    *,
    discount: float,
    noise_std: float,
    seed: int,
    batch_size: int = 2048,
    std_min: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
  """Match train-time NF whitening: goal mean/std from in-dist random rollouts."""
  env = JaxXYZVecEnv(
      num_envs=128,
      episode_length=50,
      noise_std=noise_std,
      seed=seed + 77_000,
      zero_az=True,
  )
  rng = np.random.default_rng(seed + 77_001)
  replay = _collect_random_episodes(env, rng, discount)
  n = min(int(batch_size), int(replay.size))
  batch = replay.sample(n, rng)
  goals = batch['obs'][:, env.obs_dim:]
  mean = goals.mean(axis=0).astype(np.float32)
  std = goals.std(axis=0).astype(np.float32)
  std = np.maximum(std, float(std_min)).astype(np.float32)
  return mean, std


def _patch_metrics_csv(
    csv_path: str,
    *,
    updates: Dict[int, float],
    column: str = 'val/cat_acc_state_ood',
    bak_suffix: str = '.pre_ood_reeval.bak',
) -> None:
  """Backup CSV, clear ``column``, then fill values at given iterations."""
  bak = csv_path + bak_suffix
  if not os.path.exists(bak):
    shutil.copy2(csv_path, bak)
    print(f'[reeval] backed up metrics → {bak}', flush=True)

  with open(csv_path, 'r', newline='') as f:
    rows = list(csv.DictReader(f))
  if not rows:
    print(f'[reeval] empty metrics: {csv_path}', flush=True)
    return
  fieldnames = list(rows[0].keys())
  if column not in fieldnames:
    fieldnames.append(column)

  n_set = 0
  for row in rows:
    row[column] = ''
    try:
      it = int(float(row.get('iteration', 'nan')))
    except ValueError:
      continue
    if it in updates:
      row[column] = f'{updates[it]:.8g}'
      n_set += 1

  with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
  print(f'[reeval] patched {csv_path}: set {n_set}/{len(updates)} '
        f'{column} values (others cleared)', flush=True)


def _backup_state_ood_probes(mode_dir: str, mode: str) -> None:
  prefix = _file_prefix(mode)
  probe_dir = os.path.join(mode_dir, 'ood_density_probe')
  if not os.path.isdir(probe_dir):
    return
  paths = sorted(glob.glob(os.path.join(
      probe_dir, f'probe_state_ood_{prefix}_iter*.npz')))
  if not paths:
    return
  bak_dir = os.path.join(probe_dir, 'state_ood_pre_reeval_bak')
  os.makedirs(bak_dir, exist_ok=True)
  for p in paths:
    dst = os.path.join(bak_dir, os.path.basename(p))
    if not os.path.exists(dst):
      shutil.move(p, dst)
  print(f'[reeval] moved {len(paths)} old state_ood probes → {bak_dir}',
        flush=True)


def reeval_mode(
    mode: str,
    *,
    target: str,
    log_root: str,
    high_xy: float,
    axy_low: float,
    axy_high: float,
    discount: float,
    val_batch_size: int,
    val_num_envs: int,
    noise_std: float,
    dump_probes: bool,
) -> None:
  """Re-score one mode. ``target`` is state_ood|az_ood|axy_ood."""
  if target not in ('state_ood', 'az_ood', 'axy_ood'):
    raise ValueError(
        f'target must be state_ood|az_ood|axy_ood, got {target!r}')

  mode_dir = os.path.join(log_root, mode)
  ckpts = _list_ckpts(mode_dir, mode)
  if not ckpts:
    print(f'[reeval] no checkpoints for {mode} under {mode_dir}', flush=True)
    return

  metric_col = f'val/cat_acc_{target}'
  print(f'[reeval] {mode}: {len(ckpts)} checkpoints, target={target} '
        f'ood_high_xy={high_xy} axy=[{axy_low},{axy_high}]', flush=True)
  if target == 'state_ood':
    _backup_state_ood_probes(mode_dir, mode)

  # Shared val env + NF whitening (recomputed once; train-dist, not OOD).
  file_mode = _file_prefix(mode)
  if mode in ('nf_compact', 'nf_tiny', 'nf_td'):
    score_mode = 'nf'
  elif mode == 'fb':
    score_mode = 'td3'
  else:
    score_mode = mode
  nf_goal_mean = np.zeros(3, dtype=np.float32)
  nf_goal_std = np.ones(3, dtype=np.float32)
  if score_mode == 'nf':
    # Use first ckpt args for std_min / seed.
    with open(ckpts[0][1], 'rb') as f:
      ck0 = pickle.load(f)
    nf_goal_mean, nf_goal_std = _estimate_nf_goal_stats(
        discount=discount,
        noise_std=noise_std,
        seed=int(ck0['args'].get('seed', 0)),
        std_min=float(ck0['args'].get('nf_goal_std_min', 0.1)),
    )
    print(f'[reeval] {mode}: NF goal mean={nf_goal_mean.tolist()} '
          f'std={nf_goal_std.tolist()}', flush=True)

  if target == 'state_ood':
    val_env = JaxXYZVecEnv(
        num_envs=int(val_num_envs),
        episode_length=50,
        noise_std=float(noise_std),
        seed=30_000,
        zero_az=True,
        reset_center=np.array([high_xy, high_xy, 0.0], dtype=np.float32),
    )
    rng_seed = 12345
  elif target == 'az_ood':
    # Action OOD: free a_z ~ Uniform[-1, 1] (not forced positive).
    val_env = JaxXYZVecEnv(
        num_envs=int(val_num_envs),
        episode_length=50,
        noise_std=float(noise_std),
        seed=20_000,
        zero_az=False,
        force_az_positive=False,
    )
    rng_seed = 22345
  else:
    # Action magnitude OOD: a_x,a_y ~ U[axy_low, axy_high], a_z=0.
    val_env = JaxXYZVecEnv(
        num_envs=int(val_num_envs),
        episode_length=50,
        noise_std=float(noise_std),
        seed=40_000,
        zero_az=True,
        action_low=float(axy_low),
        action_high=float(axy_high),
    )
    rng_seed = 32345

  probes = _make_fixed_ood_probes(
      noise_std, high_xy=high_xy, axy_low=axy_low, axy_high=axy_high)
  probe_dir = os.path.join(mode_dir, 'ood_density_probe')
  os.makedirs(probe_dir, exist_ok=True)

  networks = nf_nets = fm_nets = td3_nets = None
  updates: Dict[int, float] = {}
  rng = np.random.default_rng(rng_seed)

  for it, ckpt_path in ckpts:
    with open(ckpt_path, 'rb') as f:
      ckpt = pickle.load(f)
    args = ckpt['args']
    q_params = ckpt['q_params']
    if (networks is None and nf_nets is None and fm_nets is None
        and td3_nets is None):
      networks, nf_nets, fm_nets, td3_nets = _build_nets(score_mode, args)

    acc = _categorical_accuracy_on_env(
        score_mode,
        val_env=val_env,
        rng=rng,
        discount=float(discount),
        batch_size=int(val_batch_size),
        networks=networks,
        nf_nets=nf_nets,
        fm_nets=fm_nets,
        td3_nets=td3_nets,
        q_params=q_params,
        nf_goal_mean=nf_goal_mean,
        nf_goal_std=nf_goal_std,
    )
    updates[int(it)] = float(acc)
    print(f'[reeval] {mode} iter={it}  {metric_col}={acc:.4f}  '
          f'step={ckpt.get("global_step")}', flush=True)

    if dump_probes and target in ('state_ood', 'axy_ood'):
      # Rewrite the matching fixed-(s,a) density probe for this target.
      dump_ood_density_probe(
          'nf' if mode in ('nf_compact', 'nf_tiny', 'nf_td') else score_mode,
          probe=probes[target],
          out_path=os.path.join(
              probe_dir,
              f'probe_{target}_{file_mode}_iter{it:06d}.npz'),
          iteration=int(it),
          global_step=int(ckpt.get('global_step', -1)),
          networks=networks,
          nf_nets=nf_nets,
          fm_nets=fm_nets,
          td3_nets=td3_nets,
          q_params=q_params,
          nf_goal_mean=nf_goal_mean,
          nf_goal_std=nf_goal_std,
      )

  csv_path = _find_metrics_csv(mode_dir, mode)
  if csv_path is None:
    print(f'[reeval] WARNING: no metrics CSV for {mode}', flush=True)
  else:
    _patch_metrics_csv(
        csv_path,
        updates=updates,
        column=metric_col,
        bak_suffix=f'.pre_{target}_reeval.bak',
    )

  cfg_path = os.path.join(mode_dir, f'probe_config_{target}_reeval.txt')
  with open(cfg_path, 'w') as f:
    f.write(f'target={target}\n')
    f.write(f'ood_high_xy={high_xy}\n')
    f.write(f'axy_ood_low={axy_low}\n')
    f.write(f'axy_ood_high={axy_high}\n')
    f.write(f'discount={discount}\n')
    f.write(f'val_batch_size={val_batch_size}\n')
    f.write(f'val_num_envs={val_num_envs}\n')
    f.write(f'n_ckpts={len(ckpts)}\n')
    if target == 'state_ood':
      f.write(f'state_ood_reset=({high_xy},{high_xy},0) zero_az=True\n')
    elif target == 'az_ood':
      f.write('az_ood: zero_az=False force_az_positive=False '
              '(a ~ U[-1,1]^3)\n')
    else:
      f.write(f'axy_ood: zero_az=True a_x,a_y ~ U[{axy_low},{axy_high}] '
              f'(train used U[-1,1])\n')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--log_root', type=str,
                 default='logs/xyz_density_az0_g05_100m')
  p.add_argument('--target', type=str, default='state_ood',
                 choices=['state_ood', 'az_ood', 'axy_ood'],
                 help='Which OOD val metric to re-score from checkpoints.')
  p.add_argument('--ood_high_xy', type=float, default=15.0)
  p.add_argument('--axy_ood_low', type=float, default=2.0)
  p.add_argument('--axy_ood_high', type=float, default=4.0)
  p.add_argument('--discount', type=float, default=0.5)
  p.add_argument('--val_batch_size', type=int, default=256)
  p.add_argument('--val_num_envs', type=int, default=128)
  p.add_argument('--noise_std', type=float, default=0.01)
  p.add_argument('--modes', type=str, default=','.join(MODES),
                 help='Comma-separated mode dirs to reeval.')
  p.add_argument('--no_probes', action='store_true',
                 help='Only patch cat-acc; skip rewriting density probes.')
  args = p.parse_args()

  modes = tuple(m.strip() for m in args.modes.split(',') if m.strip())
  print(f'[reeval] log_root={args.log_root} target={args.target} '
        f'ood_high_xy={args.ood_high_xy} '
        f'axy=[{args.axy_ood_low},{args.axy_ood_high}] modes={modes}',
        flush=True)
  for mode in modes:
    if mode not in MODES:
      print(f'[reeval] skip unknown mode {mode!r}', flush=True)
      continue
    reeval_mode(
        mode,
        target=str(args.target),
        log_root=args.log_root,
        high_xy=float(args.ood_high_xy),
        axy_low=float(args.axy_ood_low),
        axy_high=float(args.axy_ood_high),
        discount=float(args.discount),
        val_batch_size=int(args.val_batch_size),
        val_num_envs=int(args.val_num_envs),
        noise_std=float(args.noise_std),
        dump_probes=not bool(args.no_probes),
    )
  print('[reeval] done', flush=True)


if __name__ == '__main__':
  main()
