#!/usr/bin/env python3
"""Probe NF z / log|det| at task goal for PPO checkpoints (init + rollouts).

Variation A (init): after env reset to the run's fixed start, take s0 and the
**policy mode** action a0 at (s0, g_task), then compute z=f(g_task|s0,a0) and
log|det| with that checkpoint's NF params + nf_goal_mean/std.

Variation B (traj): roll out N stochastic episodes with the checkpoint policy
(different action keys per env). At every step compute ||z||_2 and log|det|
conditioned on (s_t, a_t); report mean/std pooled over all steps × rollouts.

Uses JaxBuilderBenchVecEnv (same stack as training eval / video), so set
BUILDERBENCH_MJX_IMPL=warp and the sgcrl_builderbench conda env to match
training.

Example:
  CUDA_VISIBLE_DEVICES=0 BUILDERBENCH_MJX_IMPL=warp \\
  python scripts/probe_nf_ckpt_task_goal_z_rollouts.py \\
      --run_dir=logs/.../ppo_builderbench_creative_4_task2_0 \\
      --n_rollouts=8 --seed=0
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Prefer GPU for warp BB; callers can force CPU via JAX_PLATFORMS=cpu.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from contrastive import nf_density as _nf
from contrastive import ppo_learner
from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv
from envs.builderbench_utils import is_builderbench_creative_env

CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')
CSV_FIELDS = (
    'iter', 'global_step',
    'init_z_l2', 'init_log_det',
    'traj_z_l2_mean', 'traj_z_l2_std',
    'traj_log_det_mean', 'traj_log_det_std',
    'n_steps', 'n_rollouts',
)


def _load_bb_video_helpers():
  path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_rollout_video.py')
  spec = importlib.util.spec_from_file_location('bb_video_probe_z', path)
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(mod)
  return mod


_bb = _load_bb_video_helpers()
_load_train_ctx = _bb._load_train_ctx
_build_networks = _bb._build_networks


def _abs(path: str) -> str:
  if os.path.isabs(path):
    return path
  return os.path.join(_REPO, path)


def _load_run_config(run_dir: str) -> dict:
  path = os.path.join(run_dir, 'run_config.json')
  if not os.path.isfile(path):
    raise FileNotFoundError(f'missing run_config.json under {run_dir}')
  with open(path) as f:
    return json.load(f)


def _std_min(run_cfg: dict) -> float:
  flags = run_cfg.get('flags') or {}
  resolved = run_cfg.get('resolved_config') or {}
  return float(resolved.get('nf_goal_std_min', flags.get('nf_goal_std_min', 0.02)))


def _make_nf_nets(run_cfg: dict, obs_dim: int, act_dim: int, goal_dim: int):
  flags = run_cfg.get('flags') or {}
  resolved = run_cfg.get('resolved_config') or {}
  hidden = resolved.get('hidden_layer_sizes', flags.get('hidden_layer_sizes'))
  if isinstance(hidden, str):
    hidden = [int(x) for x in hidden.split(',') if x.strip()]
  if not hidden:
    hidden = [256] * 6
  return _nf.make_nf_density_networks(
      obs_dim=obs_dim,
      act_dim=act_dim,
      goal_dim=goal_dim,
      rep_size=int(resolved.get('nf_rep_size', flags.get('nf_rep_size', 64))),
      num_blocks=int(resolved.get('nf_num_blocks', flags.get('nf_num_blocks', 6))),
      channels=int(resolved.get(
          'nf_coupling_width', flags.get('nf_coupling_width', 192))),
      hidden_layer_sizes=tuple(int(x) for x in hidden),
      goal_enc_size=int(resolved.get(
          'nf_goal_enc_size', flags.get('nf_goal_enc_size', 0))),
      sa_hidden=int(resolved.get(
          'nf_sa_hidden', flags.get('nf_sa_hidden', 192))),
      sa_num_layers=int(resolved.get(
          'nf_sa_num_layers', flags.get('nf_sa_num_layers', 3))),
      state_only=bool(resolved.get(
          'nf_state_only', flags.get('nf_state_only', False))),
      scale_tanh=bool(resolved.get(
          'nf_scale_tanh', flags.get('nf_scale_tanh', False))),
      scale_tanh_c=float(resolved.get(
          'nf_scale_tanh_c', flags.get('nf_scale_tanh_c', 2.0))),
  )


def _list_ckpts(ckpt_dir: str) -> List[Tuple[int, str]]:
  if not os.path.isdir(ckpt_dir):
    return []
  out = []
  for name in os.listdir(ckpt_dir):
    m = CKPT_RE.match(name)
    if not m:
      continue
    out.append((int(m.group(1)), os.path.join(ckpt_dir, name)))
  out.sort()
  return out


def _read_done_iters(csv_path: str) -> Set[int]:
  done: Set[int] = set()
  if not os.path.isfile(csv_path):
    return done
  with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      try:
        done.add(int(row['iter']))
      except (KeyError, ValueError, TypeError):
        continue
  return done


def _ensure_csv_header(csv_path: str) -> None:
  os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
  if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
    return
  with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()


def _append_csv(csv_path: str, row: Dict[str, Any]) -> None:
  _ensure_csv_header(csv_path)
  with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writerow({k: row[k] for k in CSV_FIELDS})


def _print_table(rows: List[Dict[str, Any]]) -> None:
  hdr = (f'{"iter":>6}  {"step":>10}  '
         f'{"init||z||":>10}  {"init_ldet":>10}  '
         f'{"traj||z||μ":>10}  {"traj||z||σ":>10}  '
         f'{"traj_ldetμ":>10}  {"traj_ldetσ":>10}  '
         f'{"n":>5}')
  print(hdr, flush=True)
  for r in rows:
    print(
        f'{r["iter"]:6d}  {r["global_step"]:10d}  '
        f'{r["init_z_l2"]:10.4f}  {r["init_log_det"]:10.4f}  '
        f'{r["traj_z_l2_mean"]:10.4f}  {r["traj_z_l2_std"]:10.4f}  '
        f'{r["traj_log_det_mean"]:10.4f}  {r["traj_log_det_std"]:10.4f}  '
        f'{r["n_steps"]:5d}',
        flush=True)


def _load_all_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
  if not os.path.isfile(csv_path):
    return []
  rows = []
  with open(csv_path, newline='') as f:
    for row in csv.DictReader(f):
      rows.append({
          'iter': int(row['iter']),
          'global_step': int(row['global_step']),
          'init_z_l2': float(row['init_z_l2']),
          'init_log_det': float(row['init_log_det']),
          'traj_z_l2_mean': float(row['traj_z_l2_mean']),
          'traj_z_l2_std': float(row['traj_z_l2_std']),
          'traj_log_det_mean': float(row['traj_log_det_mean']),
          'traj_log_det_std': float(row['traj_log_det_std']),
          'n_steps': int(row['n_steps']),
          'n_rollouts': int(row['n_rollouts']),
      })
  rows.sort(key=lambda r: r['iter'])
  return rows


def _goal_norm(g_task: np.ndarray, extra: dict, std_min: float
               ) -> jnp.ndarray:
  gmean = np.asarray(extra['nf_goal_mean'], dtype=np.float32).reshape(-1)
  gstd = np.asarray(extra['nf_goal_std'], dtype=np.float32).reshape(-1)
  gstd = np.maximum(gstd, std_min).astype(np.float32)
  g = np.asarray(g_task, dtype=np.float32).reshape(-1)
  return jnp.asarray((g - gmean) / (gstd + 1e-8), dtype=jnp.float32)


class RolloutProbeSession:
  """Shared env + networks for probing every checkpoint in a run."""

  def __init__(
      self,
      run_dir: str,
      *,
      n_rollouts: int = 8,
      seed: int = 0,
      which: str = 'online',
      env_name: Optional[str] = None,
  ):
    self.run_dir = run_dir
    self.run_cfg = _load_run_config(run_dir)
    self.n_rollouts = int(n_rollouts)
    self.seed = int(seed)
    self.which = which
    self.std_min = _std_min(self.run_cfg)

    env = env_name or str(self.run_cfg.get('env', ''))
    if not env or not is_builderbench_creative_env(env):
      raise ValueError(f'need builderbench creative env, got {env!r}')
    self.env_name = env

    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    # Prefer an early real ckpt so train-ctx inference matches training.
    ckpts = _list_ckpts(ckpt_dir)
    if not ckpts:
      raise FileNotFoundError(f'no ckpt_iter_*.pkl under {ckpt_dir}')
    probe_ckpt = ckpts[0][1]

    self.ctx = _load_train_ctx(env, probe_ckpt)
    fixed_goal = self.ctx.fixed_target_goal
    if fixed_goal is None and self.run_cfg.get('fixed_start_end') is not None:
      fixed_goal = np.asarray(self.run_cfg['fixed_start_end'], dtype=np.float32)
    self.g_task = np.asarray(fixed_goal, dtype=np.float32).reshape(-1)
    self.goal_dim = int(self.g_task.size)
    self.obs_dim = int(self.ctx.obs_dim)

    print(f'[probe_z] building networks (env={env})...', flush=True)
    self.networks = _build_networks(env, seed=seed, ctx=self.ctx)

    print(f'[probe_z] building vec env n={self.n_rollouts} '
          f'impl={os.environ.get("BUILDERBENCH_MJX_IMPL", "jax")}...',
          flush=True)
    self.vec_env = JaxBuilderBenchVecEnv(
        env_name=env,
        num_envs=self.n_rollouts,
        seed=self.seed,
        use_pd=self.ctx.use_pd,
        pd_duration=self.ctx.pd_duration,
        pd_filter_policy_obs=self.ctx.filter_policy_obs,
        fixed_target_goal=self.g_task,
        permute_start_boxes=bool(self.ctx.permute_start_boxes),
        fixed_start_x=self.ctx.fixed_start_x,
        mj_episode_length=int(
            (self.run_cfg.get('flags') or {}).get(
                'builderbench_mj_episode_length', 0) or 0) or None,
    )
    self.act_dim = int(self.vec_env.action_shape[0])
    self.ep_len = int(self.vec_env.episode_length)
    print(f'[probe_z] obs_dim={self.obs_dim} act_dim={self.act_dim} '
          f'goal_dim={self.goal_dim} ep_len={self.ep_len} '
          f'permute={self.ctx.permute_start_boxes} '
          f'fixed_start_x={self.ctx.fixed_start_x}',
          flush=True)

    self.nf_nets = _make_nf_nets(
        self.run_cfg, self.obs_dim, self.act_dim, self.goal_dim)

    policy_network = self.networks.policy_network
    sample_fn = self.networks.sample

    @jax.jit
    def mode_action(policy_params, packed_obs):
      dist = policy_network.apply(policy_params, packed_obs)
      return dist.mode()

    @jax.jit
    def sample_action(policy_params, packed_obs, key):
      dist = policy_network.apply(policy_params, packed_obs)
      return sample_fn(dist, key)

    self._mode_action = mode_action
    self._stoch_unroll = self.vec_env.compile_mpo_unroll(
        sample_action, unroll_length=self.ep_len)

    @jax.jit
    def nf_batch(params, state, action, g_norm):
      # Broadcast single g_norm over batch.
      g = jnp.broadcast_to(g_norm.reshape(1, -1), (state.shape[0], g_norm.size))
      z, log_det, _log_p = _nf.nf_forward(
          self.nf_nets, params, state, action, g)
      z_l2 = jnp.linalg.norm(z, axis=-1)
      return z_l2, log_det

    self._nf_batch = nf_batch

  def _reset_fixed(self):
    """Reset all envs from a fixed PRNG so starts match across ckpts."""
    self.vec_env._rng = jax.random.PRNGKey(self.seed)
    return self.vec_env.reset_state()

  def probe_ckpt(self, ckpt_path: str) -> Dict[str, Any]:
    ckpt = ppo_learner.load_checkpoint(ckpt_path)
    m = CKPT_RE.search(os.path.basename(ckpt_path))
    it = int(m.group(1)) if m else int(ckpt.get('iteration', -1))
    global_step = int(ckpt.get('global_step', -1))
    q_key = 'q_params' if self.which == 'online' else 'q_params_ema'
    if q_key not in ckpt:
      raise KeyError(f'{q_key} missing in {ckpt_path}')
    if 'policy_params' not in ckpt:
      raise KeyError(f'policy_params missing in {ckpt_path}')

    extra = ckpt.get('extra_state') or {}
    g_norm = _goal_norm(self.g_task, extra, self.std_min)
    nf_params = ckpt[q_key]
    policy_params = ckpt['policy_params']

    # --- Variation A: initial (s0, mode a0) ---
    env_state = self._reset_fixed()
    packed0 = self.vec_env.pack_obs_from_state(env_state)
    # Use env 0 only (fixed start → identical across rollouts).
    packed0_1 = packed0[:1]
    s0 = packed0_1[:, :self.obs_dim]
    a0 = self._mode_action(policy_params, packed0_1)
    a0 = jnp.clip(a0, -1.0, 1.0)
    z_l2_init, log_det_init = self._nf_batch(nf_params, s0, a0, g_norm)
    init_z_l2 = float(np.asarray(z_l2_init).reshape(-1)[0])
    init_log_det = float(np.asarray(log_det_init).reshape(-1)[0])

    # --- Variation B: N stochastic rollouts ---
    # Fresh reset with same start seed; action keys differ per env/step.
    env_state = self._reset_fixed()
    act_key = jax.random.PRNGKey(self.seed + 10_000 + it)
    (_final_state, _key), steps = self._stoch_unroll(
        env_state, policy_params, act_key)
    # steps['obs']: (T, E, obs+goal), steps['actions']: (T, E, A)
    obs = steps['obs']
    actions = steps['actions']
    t, e = int(obs.shape[0]), int(obs.shape[1])
    state = obs[:, :, :self.obs_dim].reshape(t * e, self.obs_dim)
    act = actions.reshape(t * e, self.act_dim)
    z_l2, log_det = self._nf_batch(nf_params, state, act, g_norm)
    z_l2 = np.asarray(z_l2, dtype=np.float64).reshape(-1)
    log_det = np.asarray(log_det, dtype=np.float64).reshape(-1)

    row = {
        'iter': it,
        'global_step': global_step,
        'init_z_l2': init_z_l2,
        'init_log_det': init_log_det,
        'traj_z_l2_mean': float(z_l2.mean()),
        'traj_z_l2_std': float(z_l2.std(ddof=0)),
        'traj_log_det_mean': float(log_det.mean()),
        'traj_log_det_std': float(log_det.std(ddof=0)),
        'n_steps': int(z_l2.size),
        'n_rollouts': int(e),
        'ckpt': ckpt_path,
    }
    return row


def main(argv: Optional[List[str]] = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--run_dir', type=str, required=True)
  p.add_argument('--out', type=str, default=None,
                 help='CSV path (default: <run_dir>/task_goal_z_rollout_stats.csv)')
  p.add_argument('--n_rollouts', type=int, default=8)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--which', type=str, default='online',
                 choices=('online', 'ema'))
  p.add_argument('--env', type=str, default=None)
  p.add_argument('--ckpt', type=str, default=None,
                 help='Optional single checkpoint; default = all ckpt_iter_*.pkl')
  p.add_argument('--force', action='store_true',
                 help='Recompute rows even if iter already in CSV')
  args = p.parse_args(argv)

  run_dir = _abs(args.run_dir)
  csv_path = _abs(args.out) if args.out else os.path.join(
      run_dir, 'task_goal_z_rollout_stats.csv')
  ckpt_dir = os.path.join(run_dir, 'checkpoints')

  print(f'jax backend={jax.default_backend()} devices={jax.devices()}',
        flush=True)
  print(f'BUILDERBENCH_MJX_IMPL={os.environ.get("BUILDERBENCH_MJX_IMPL")}',
        flush=True)
  print(f'run_dir={run_dir}', flush=True)
  print(f'csv={csv_path}', flush=True)
  print('Variation A: s0=env reset fixed start, a0=policy MODE (stable)',
        flush=True)
  print(f'Variation B: n_rollouts={args.n_rollouts} STOCHASTIC, '
        f'pool mean/std of ||z||_2 and log|det|',
        flush=True)

  session = RolloutProbeSession(
      run_dir,
      n_rollouts=args.n_rollouts,
      seed=args.seed,
      which=args.which,
      env_name=args.env,
  )
  print(f'task_goal dim={session.goal_dim}  {session.g_task.tolist()}',
        flush=True)

  if args.ckpt:
    ckpts = [(-1, _abs(args.ckpt))]
  else:
    ckpts = _list_ckpts(ckpt_dir)

  done = set() if args.force else _read_done_iters(csv_path)
  new_rows: List[Dict[str, Any]] = []
  for it, path in ckpts:
    if it >= 0 and it in done:
      print(f'[skip] iter={it} already in CSV', flush=True)
      continue
    try:
      if os.path.getsize(path) < 1024:
        print(f'[skip] iter={it} tiny file', flush=True)
        continue
    except OSError as e:
      print(f'[skip] iter={it} {e}', flush=True)
      continue
    print(f'[probe] iter={it if it >= 0 else "?"} {os.path.basename(path)}...',
          flush=True)
    try:
      row = session.probe_ckpt(path)
    except Exception as e:  # noqa: BLE001
      print(f'[fail] iter={it}: {e}', flush=True)
      raise
    _append_csv(csv_path, row)
    new_rows.append(row)
    print(json.dumps({k: row[k] for k in CSV_FIELDS}), flush=True)

  all_rows = _load_all_csv_rows(csv_path)
  print('=== full table ===', flush=True)
  _print_table(all_rows)

  # Highlight early / success-ish / late.
  highlight = {0, 20, 40, 60, 80, 100, 200, 300, 380, 389}
  focus = [r for r in all_rows if r['iter'] in highlight]
  if focus:
    print('=== highlight (early / ~success 60–80 / late) ===', flush=True)
    _print_table(focus)

  print(f'csv={csv_path} processed_new={len(new_rows)} '
        f'total={len(all_rows)}', flush=True)
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
