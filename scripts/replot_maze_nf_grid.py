#!/usr/bin/env python3
"""Re-render maze occupancy/preimage PNGs as full-grid NF density + traj lines.

Overwrites ``<run_dir>/rollouts/iter_XXXXXXX.png`` from each checkpoint.

  python scripts/replot_maze_nf_grid.py \
      --run_dir=logs/ppo_mazeconcept_nf_tiny_occpre_every1_500k/ppo_point_MazeConcept_0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcrl_jax_acme_compat  # noqa: F401

import numpy as np
from acme import specs

import contrastive
from contrastive import nf_density as _nf
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from contrastive.maze_nf_render import render_maze_occupancy_preimage


def _load_run_payload(run_dir: str) -> dict:
  path = os.path.join(run_dir, 'run_config.json')
  with open(path, 'r', encoding='utf-8') as fh:
    return json.load(fh)


def _resolved(payload: dict) -> dict:
  resolved = payload.get('resolved_config') or {}
  flags = payload.get('flags') or {}
  out = dict(flags)
  out.update(resolved)
  return out


def _fixed_start_end(payload: dict, cfg: dict):
  fse = payload.get('fixed_start_end') or cfg.get('fixed_start_end')
  if not fse or len(fse) != 2:
    return None
  return [np.asarray(fse[0], dtype=np.float32),
          np.asarray(fse[1], dtype=np.float32)]


def _hidden(cfg: dict):
  raw = cfg.get('hidden_layer_sizes', (256, 256, 256, 256, 256, 256))
  if isinstance(raw, str):
    return tuple(int(x) for x in raw.split(',') if x.strip())
  return tuple(int(x) for x in raw)


def _list_ckpts(ckpt_dir: str) -> list[tuple[int, str]]:
  out = []
  for p in glob.glob(os.path.join(ckpt_dir, 'ckpt_iter_*.pkl')):
    m = re.search(r'ckpt_iter_(\d+)\.pkl$', os.path.basename(p))
    if m:
      out.append((int(m.group(1)), p))
  out.sort()
  return out


def _build_policy_networks(env_name: str, seed: int, cfg: dict, fse):
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, start_index=0, end_index=-1, seed=seed, fixed_start_end=fse)
  env_spec = specs.make_environment_spec(probe_env)
  del probe_env
  networks = contrastive.make_networks(
      spec=env_spec,
      obs_dim=obs_dim,
      repr_dim=int(cfg.get('repr_dim', 64)),
      repr_norm=bool(cfg.get('repr_norm', False)),
      twin_q=bool(cfg.get('twin_q', False)),
      use_image_obs=bool(cfg.get('use_image_obs', False)),
      hidden_layer_sizes=_hidden(cfg),
      actor_min_std=(
          float(cfg['ppo_actor_min_std'])
          if float(cfg.get('ppo_actor_min_std', -1.0) or -1.0) > 0.0
          else 1e-5),
  )
  act_dim = int(np.prod(env_spec.actions.shape))
  return networks, obs_dim, act_dim


def _build_nf(obs_dim: int, act_dim: int, cfg: dict):
  return _nf.make_nf_density_networks(
      obs_dim=obs_dim,
      act_dim=act_dim,
      goal_dim=obs_dim,
      hidden_layer_sizes=_hidden(cfg),
      rep_size=int(cfg.get('nf_rep_size', 64)),
      num_blocks=int(cfg.get('nf_num_blocks', 8)),
      channels=int(cfg.get('nf_coupling_width', 256)),
      sa_hidden=int(cfg.get('nf_sa_hidden', 1024)),
      sa_num_layers=int(cfg.get('nf_sa_num_layers', 4)),
      state_only=bool(cfg.get('nf_state_only', False)),
  )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--run_dir',
      default=('logs/ppo_mazeconcept_nf_tiny_occpre_every1_500k/'
               'ppo_point_MazeConcept_0'))
  ap.add_argument('--heatmap_subcells', type=int, default=8)
  ap.add_argument('--num_trajectories', type=int, default=5)
  args = ap.parse_args()

  repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  run_dir = args.run_dir
  if not os.path.isabs(run_dir):
    run_dir = os.path.join(repo, run_dir)
  payload = _load_run_payload(run_dir)
  cfg = _resolved(payload)
  env_name = str(payload.get('env') or cfg.get('env'))
  seed = int(payload.get('seed', cfg.get('seed', 0)))
  fse = _fixed_start_end(payload, cfg)
  ckpt_dir = os.path.join(run_dir, 'checkpoints')
  out_dir = os.path.join(run_dir, 'rollouts')
  ckpts = _list_ckpts(ckpt_dir)
  if not ckpts:
    raise SystemExit(f'no ckpt_iter_*.pkl in {ckpt_dir}')

  print(f'[replot] env={env_name}  seed={seed}  n_ckpt={len(ckpts)}',
        flush=True)
  networks, obs_dim, act_dim = _build_policy_networks(
      env_name, seed, cfg, fse)
  nf_nets = _build_nf(obs_dim, act_dim, cfg)
  nf_reward_fn = _nf.make_nf_reward_fn(nf_nets, obs_dim=obs_dim)
  print(f'[replot] NF tiny? sa={cfg.get("nf_sa_num_layers")}x'
        f'{cfg.get("nf_sa_hidden")} r={cfg.get("nf_rep_size")} '
        f'b={cfg.get("nf_num_blocks")} w={cfg.get("nf_coupling_width")}',
        flush=True)

  ones = np.ones((obs_dim,), dtype=np.float32)
  zeros = np.zeros((obs_dim,), dtype=np.float32)
  for it, path in ckpts:
    ckpt = ppo_learner.load_checkpoint(path)
    extra = ckpt.get('extra_state') or {}
    nf_params = ckpt.get('q_params_ema') or ckpt['q_params']
    gmean = np.asarray(extra.get('nf_goal_mean', zeros), dtype=np.float32)
    gstd = np.asarray(extra.get('nf_goal_std', ones), dtype=np.float32)
    step = int(ckpt.get('global_step', -1))
    out_path = os.path.join(out_dir, f'iter_{it:07d}.png')
    render_maze_occupancy_preimage(
        env_name=env_name,
        networks=networks,
        policy_params=ckpt['policy_params'],
        nf_reward_fn=nf_reward_fn,
        nf_params=nf_params,
        nf_goal_mean=gmean,
        nf_goal_std=gstd,
        out_path=out_path,
        title=f'{env_name}  iter={it}  step={step}',
        seed=seed + 17_000 + it,
        heatmap_subcells=int(args.heatmap_subcells),
        fixed_start_end=fse,
        num_trajectories=int(args.num_trajectories),
    )
    print(f'[replot] wrote {out_path}', flush=True)
  print(f'[replot] done  {len(ckpts)} pngs in {out_dir}', flush=True)


if __name__ == '__main__':
  main()
