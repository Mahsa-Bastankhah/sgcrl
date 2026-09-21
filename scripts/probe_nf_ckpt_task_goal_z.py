#!/usr/bin/env python3
"""Probe NF RealNVP forward at the task goal for PPO checkpoints.

For each ckpt: load NF params + ``extra_state.nf_goal_mean/std``, normalize the
run_config ``fixed_start_end`` goal, condition on fixed zero ``(s,a)``, and
report ``||z||_2`` and ``log|det|`` from ``nf_forward`` (same path as training
log_prob).

Examples:
  # One checkpoint → JSON line + CSV append
  python scripts/probe_nf_ckpt_task_goal_z.py \\
      --run_dir=logs/.../ppo_builderbench_creative_4_task2_0 \\
      --ckpt=logs/.../checkpoints/ckpt_iter_0000020.pkl

  # Process all new ckpts once
  python scripts/probe_nf_ckpt_task_goal_z.py --run_dir=... --once

  # Poll until Slurm job finishes (CPU; not GPU)
  python scripts/probe_nf_ckpt_task_goal_z.py \\
      --run_dir=... --watch --job-id=3812404 --poll-sec=30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
  sys.path.insert(0, _REPO)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from contrastive import nf_density as _nf
from contrastive import ppo_learner

# BuilderBench creative-4 defaults (obs includes select; action 4+1).
DEFAULT_OBS_DIM = 13
DEFAULT_ACT_DIM = 5
DEFAULT_GOAL_DIM = 12
CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')
CSV_FIELDS = ('iter', 'global_step', 'z_l2', 'log_det', 'z_dim')


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


def _infer_dims(run_cfg: dict, obs_dim: int, act_dim: int, goal_dim: int
                ) -> Tuple[int, int, int]:
  g = run_cfg.get('fixed_start_end')
  if g is not None:
    goal_dim = int(np.asarray(g).size)
  flags = run_cfg.get('flags') or {}
  # Optional overrides if ever stored; otherwise keep BB defaults.
  if 'obs_dim' in flags:
    obs_dim = int(flags['obs_dim'])
  if 'act_dim' in flags:
    act_dim = int(flags['act_dim'])
  return obs_dim, act_dim, goal_dim


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


def _std_min(run_cfg: dict) -> float:
  flags = run_cfg.get('flags') or {}
  resolved = run_cfg.get('resolved_config') or {}
  return float(resolved.get('nf_goal_std_min', flags.get('nf_goal_std_min', 0.02)))


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
  print(f'{"iter":>6}  {"global_step":>12}  {"||z||_2":>12}  {"log|det|":>12}  {"z_dim":>5}')
  for r in rows:
    print(f'{r["iter"]:6d}  {r["global_step"]:12d}  {r["z_l2"]:12.6f}  '
          f'{r["log_det"]:12.6f}  {r["z_dim"]:5d}')


def _slurm_job_state(job_id: Optional[str]) -> Optional[str]:
  if not job_id:
    return None
  try:
    out = subprocess.check_output(
        ['sacct', '-j', str(job_id), '-n', '-X',
         '--format=State', '-P'],
        text=True, stderr=subprocess.DEVNULL)
  except (subprocess.CalledProcessError, FileNotFoundError):
    try:
      out = subprocess.check_output(
          ['squeue', '-j', str(job_id), '-h', '-o', '%T'],
          text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
      return None
  lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
  if not lines:
    return None
  # sacct may return JOBID|STATE; take last field.
  state = lines[0].split('|')[-1].strip()
  return state.upper().split()[0]  # e.g. COMPLETED, RUNNING, PENDING


def _job_finished(state: Optional[str]) -> bool:
  if state is None:
    return False
  return state in {
      'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
      'PREEMPTED', 'OUT_OF_MEMORY', 'BOOT_FAIL', 'DEADLINE',
  }


class ProbeContext:
  def __init__(self, run_dir: str, obs_dim: int, act_dim: int, goal_dim: int):
    self.run_dir = run_dir
    self.run_cfg = _load_run_config(run_dir)
    self.obs_dim, self.act_dim, self.goal_dim = _infer_dims(
        self.run_cfg, obs_dim, act_dim, goal_dim)
    self.g_task = np.asarray(
        self.run_cfg['fixed_start_end'], dtype=np.float32).reshape(1, self.goal_dim)
    self.std_min = _std_min(self.run_cfg)
    self.nf_nets = _make_nf_nets(
        self.run_cfg, self.obs_dim, self.act_dim, self.goal_dim)
    self.s0 = jnp.zeros((1, self.obs_dim), dtype=jnp.float32)
    self.a0 = jnp.zeros((1, self.act_dim), dtype=jnp.float32)

    @jax.jit
    def fwd(params, g_norm):
      return _nf.nf_forward(self.nf_nets, params, self.s0, self.a0, g_norm)

    self.fwd = fwd

  def probe_ckpt(self, ckpt_path: str, which: str = 'online') -> Dict[str, Any]:
    ckpt = ppo_learner.load_checkpoint(ckpt_path)
    m = CKPT_RE.search(os.path.basename(ckpt_path))
    it = int(m.group(1)) if m else int(ckpt.get('iteration', -1))
    key = 'q_params' if which == 'online' else 'q_params_ema'
    if key not in ckpt:
      raise KeyError(f'{key} missing in {ckpt_path}')
    extra = ckpt.get('extra_state') or {}
    gmean = np.asarray(extra['nf_goal_mean'], dtype=np.float32).reshape(self.goal_dim)
    gstd = np.asarray(extra['nf_goal_std'], dtype=np.float32).reshape(self.goal_dim)
    gstd = np.maximum(gstd, self.std_min).astype(np.float32)
    g_norm = (self.g_task - gmean) / (gstd + 1e-8)
    z, log_det, log_p = self.fwd(ckpt[key], jnp.asarray(g_norm, dtype=jnp.float32))
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    row = {
        'iter': it,
        'global_step': int(ckpt.get('global_step', -1)),
        'z_l2': float(np.linalg.norm(z)),
        'log_det': float(np.asarray(log_det).reshape(-1)[0]),
        'z_dim': int(z.size),
        'log_p': float(np.asarray(log_p).reshape(-1)[0]),
        'which': which,
        'ckpt': ckpt_path,
    }
    return row


def process_new(
    ctx: ProbeContext,
    ckpt_dir: str,
    csv_path: str,
    *,
    which: str = 'online',
    only_iters: Optional[Set[int]] = None,
) -> List[Dict[str, Any]]:
  done = _read_done_iters(csv_path)
  rows: List[Dict[str, Any]] = []
  for it, path in _list_ckpts(ckpt_dir):
    if it in done:
      continue
    if only_iters is not None and it not in only_iters:
      continue
    # Skip incomplete writes: require non-tiny files.
    try:
      if os.path.getsize(path) < 1024:
        continue
    except OSError:
      continue
    try:
      row = ctx.probe_ckpt(path, which=which)
    except Exception as e:  # noqa: BLE001 — keep watching through transient pickle races
      print(f'[skip] iter={it} load/probe failed: {e}', flush=True)
      continue
    _append_csv(csv_path, row)
    rows.append(row)
    print(json.dumps({k: row[k] for k in (*CSV_FIELDS, 'log_p', 'which')}), flush=True)
  return rows


def _load_all_csv_rows(csv_path: str) -> List[Dict[str, Any]]:
  if not os.path.isfile(csv_path):
    return []
  rows = []
  with open(csv_path, newline='') as f:
    for row in csv.DictReader(f):
      rows.append({
          'iter': int(row['iter']),
          'global_step': int(row['global_step']),
          'z_l2': float(row['z_l2']),
          'log_det': float(row['log_det']),
          'z_dim': int(row['z_dim']),
      })
  rows.sort(key=lambda r: r['iter'])
  return rows


def main(argv: Optional[List[str]] = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--run_dir', type=str, required=True,
                 help='Seed run dir containing run_config.json + checkpoints/')
  p.add_argument('--ckpt', type=str, default=None,
                 help='Optional single checkpoint path')
  p.add_argument('--out', type=str, default=None,
                 help='CSV path (default: <run_dir>/task_goal_z_logdet.csv)')
  p.add_argument('--which', type=str, default='online',
                 choices=('online', 'ema'))
  p.add_argument('--obs_dim', type=int, default=DEFAULT_OBS_DIM)
  p.add_argument('--act_dim', type=int, default=DEFAULT_ACT_DIM)
  p.add_argument('--goal_dim', type=int, default=DEFAULT_GOAL_DIM)
  p.add_argument('--once', action='store_true',
                 help='Process all new ckpts once and exit')
  p.add_argument('--watch', action='store_true',
                 help='Poll for new ckpts until job finishes')
  p.add_argument('--poll-sec', type=float, default=30.0)
  p.add_argument('--job-id', type=str, default=None,
                 help='Slurm job id; watch stops after it finishes')
  p.add_argument('--max-watch-sec', type=float, default=0.0,
                 help='If >0, stop watching after this many seconds '
                      '(leave CSV for resume)')
  p.add_argument('--log_stem_wait_sec', type=float, default=0.0,
                 help='If run_dir missing, wait this long for it to appear')
  args = p.parse_args(argv)

  run_dir = _abs(args.run_dir)
  csv_path = _abs(args.out) if args.out else os.path.join(
      run_dir, 'task_goal_z_logdet.csv')
  ckpt_dir = os.path.join(run_dir, 'checkpoints')

  print(f'jax backend={jax.default_backend()} devices={jax.devices()}', flush=True)
  print(f'run_dir={run_dir}', flush=True)
  print(f'csv={csv_path}', flush=True)
  print(f'conditioning: s=0_({args.obs_dim},), a=0_({args.act_dim},) raw', flush=True)

  deadline = None
  if args.max_watch_sec and args.max_watch_sec > 0:
    deadline = time.time() + float(args.max_watch_sec)

  # Wait for run_dir / run_config if training has not created it yet.
  wait_budget = float(args.log_stem_wait_sec)
  t0 = time.time()
  while not os.path.isfile(os.path.join(run_dir, 'run_config.json')):
    if not args.watch and wait_budget <= 0:
      print(f'ERROR: run_config.json not found under {run_dir}', flush=True)
      return 2
    if deadline is not None and time.time() >= deadline:
      print('[watch] max-watch-sec reached before run_dir appeared; resume later',
            flush=True)
      return 0
    if args.job_id:
      st = _slurm_job_state(args.job_id)
      print(f'[wait] run_dir missing; job {args.job_id} state={st}', flush=True)
      if _job_finished(st):
        print('[wait] job finished but run_dir never appeared', flush=True)
        return 3
    else:
      print('[wait] run_dir missing...', flush=True)
    if wait_budget > 0 and (time.time() - t0) >= wait_budget and not args.watch:
      return 2
    time.sleep(max(args.poll_sec, 5.0))

  ctx = ProbeContext(run_dir, args.obs_dim, args.act_dim, args.goal_dim)
  print(f'task_goal dim={ctx.goal_dim}  {ctx.g_task.reshape(-1).tolist()}',
        flush=True)

  if args.ckpt:
    row = ctx.probe_ckpt(_abs(args.ckpt), which=args.which)
    done = _read_done_iters(csv_path)
    if row['iter'] not in done:
      _append_csv(csv_path, row)
    print(json.dumps({k: row[k] for k in (*CSV_FIELDS, 'log_p', 'which')}), flush=True)
    _print_table(_load_all_csv_rows(csv_path))
    return 0

  def one_pass() -> List[Dict[str, Any]]:
    return process_new(ctx, ckpt_dir, csv_path, which=args.which)

  new_rows = one_pass()
  all_rows = _load_all_csv_rows(csv_path)
  if all_rows:
    print('=== table so far ===', flush=True)
    _print_table(all_rows)
  else:
    print('(no checkpoints processed yet)', flush=True)

  if not args.watch:
    print(f'processed_new={len(new_rows)} total_in_csv={len(all_rows)}', flush=True)
    return 0

  print(f'[watch] poll={args.poll_sec}s job_id={args.job_id} '
        f'max_watch_sec={args.max_watch_sec or "none"}', flush=True)
  while True:
    if deadline is not None and time.time() >= deadline:
      print('[watch] max-watch-sec reached; CSV left for resume', flush=True)
      _print_table(_load_all_csv_rows(csv_path))
      print(f'RESUME: python scripts/probe_nf_ckpt_task_goal_z.py '
            f'--run_dir={run_dir} --out={csv_path} --watch '
            f'--job-id={args.job_id or ""} --poll-sec={args.poll_sec}',
            flush=True)
      return 0
    st = _slurm_job_state(args.job_id) if args.job_id else None
    new_rows = one_pass()
    all_rows = _load_all_csv_rows(csv_path)
    if new_rows:
      print('=== table so far ===', flush=True)
      _print_table(all_rows)
    else:
      n_disk = len(_list_ckpts(ckpt_dir))
      print(f'[watch] no new ckpts (disk={n_disk}, csv={len(all_rows)}) '
            f'job_state={st}', flush=True)
    if args.job_id and _job_finished(st):
      # Final sweep in case last ckpt landed after terminal state.
      time.sleep(2.0)
      one_pass()
      all_rows = _load_all_csv_rows(csv_path)
      print(f'[watch] job {args.job_id} finished state={st}', flush=True)
      print('=== final table ===', flush=True)
      _print_table(all_rows)
      print(f'csv={csv_path} n={len(all_rows)}', flush=True)
      return 0 if st == 'COMPLETED' else 4
    time.sleep(max(float(args.poll_sec), 1.0))


if __name__ == '__main__':
  raise SystemExit(main())
