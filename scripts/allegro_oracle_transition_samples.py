#!/usr/bin/env python3
"""Print Allegro control-sanity (s, a, s') samples under the oracle action.

Oracle a = the separate normalized goal action. Useful for diagnosing whether
one-step PD toward the fixed joint goal is locally learnable for NF ranking.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    CONTROL_SANITY_INDEX_THUMB_HAND_INDICES,
    CONTROL_SANITY_INDEX_THUMB_STRAIGHT_GOAL_Q,
    AllegroKukaThrowVecEnv,
)


def _fmt(xs, nd=3):
  return "[" + ", ".join(f"{float(v):.{nd}f}" for v in xs) + "]"


def _split_state(packed_row: np.ndarray, n: int = 8):
  """packed = [q_norm (n), qd (n), g_norm (n)]."""
  q = packed_row[:n]
  qd = packed_row[n:2 * n]
  g = packed_row[2 * n:3 * n]
  return q, qd, g


def _phys_err(env, env_i: int):
  hand_q = env._env.arm_hand_dof_pos[env_i, 7:23]
  idx = list(CONTROL_SANITY_INDEX_THUMB_HAND_INDICES)
  q = hand_q[idx]
  g = env._control_sanity_goal_q_batch[env_i]
  abs_err = (q - g).abs()
  return (
      float(abs_err.max().item()),
      float(abs_err.mean().item()),
      q.detach().cpu().numpy(),
  )


def _balanced(env, env_i: int) -> int:
  return int(env.success_levels()["hard"][env_i].item())


def _state_delta(a: np.ndarray, b: np.ndarray) -> dict:
  dq = float(np.max(np.abs(a[:8] - b[:8])))
  dqd = float(np.max(np.abs(a[8:16] - b[8:16])))
  l2 = float(np.linalg.norm(a[:16] - b[:16]))
  return {"max_abs_dq_norm": dq, "max_abs_dqd": dqd, "l2_state": l2}


def _mean_key(ds, k):
  return float(np.mean([d[k] for d in ds]))


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--trim-init-mode", default="full_range",
      choices=("full_range", "curled"))
  parser.add_argument("--trim-init-range-frac", type=float, default=0.50)
  parser.add_argument("--pipeline", default="gpu", choices=("gpu", "cpu"))
  parser.add_argument(
      "--coordinate-mode", default="mixed",
      choices=("mixed", "physical", "fully_scaled"))
  parser.add_argument(
      "--out-dir",
      default=os.path.join(
          REPO, "figs", "allegro_kuka_throw", "oracle_transition_samples"))
  parser.add_argument("--horizon", type=int, default=20)
  args = parser.parse_args()

  n = 8
  env = AllegroKukaThrowVecEnv(
      num_envs=args.num_envs,
      seed=args.seed,
      episode_length=150,
      pipeline=args.pipeline,
      control_sanity_mode="index_thumb_straight",
      control_sanity_trim_sa=True,
      control_sanity_trim_init_range_frac=args.trim_init_range_frac,
      control_sanity_trim_init_mode=args.trim_init_mode,
      coordinate_mode=args.coordinate_mode,
      randomize_init=True,
      randomize_object_shape=False,
      large_table=True,
      enable_cameras=False,
  )
  assert env.obs_dim == 16 and env.action_dim == 8 and env.goal_dim == 8

  import torch

  g_raw = np.asarray(
      CONTROL_SANITY_INDEX_THUMB_STRAIGHT_GOAL_Q, dtype=np.float64)
  g_norm = env.normalized_goal_action()[0].detach().cpu().numpy()
  print("=" * 72, flush=True)
  print(
      "[oracle_samples] FIXED GOAL  "
      f"raw_rad={_fmt(g_raw, 4)}  "
      f"g_norm={_fmt(g_norm, 4)}",
      flush=True)
  print(
      f"[oracle_samples] mode=index_thumb_straight trim_sa "
      f"init={args.trim_init_mode} frac={args.trim_init_range_frac:g} "
      f"num_envs={args.num_envs} seed={args.seed}",
      flush=True)
  print(
      "[oracle_samples] state layout: s=[q_norm(8), qd(8)]; "
      "oracle a = g_norm (constant); success=balanced "
      "(mean|q-q*|<=0.15 & max<=0.30 rad)",
      flush=True)
  print("=" * 72, flush=True)

  # ---- Pass 1: oracle transitions ----
  obs0 = env.reset()
  oracle_a = env.normalized_goal_action().detach().clone()
  max0 = []
  mean0 = []
  bal0 = []
  print("\n--- PASS 1: oracle action a = g_norm ---\n", flush=True)
  for i in range(args.num_envs):
    packed0 = obs0[i].detach().cpu().numpy()
    q0, qd0, g0 = _split_state(packed0, n)
    mx, mn, q_raw0 = _phys_err(env, i)
    bal = _balanced(env, i)
    max0.append(mx)
    mean0.append(mn)
    bal0.append(bal)
    print(
        f"env={i} t=0  q_norm={_fmt(q0)}  qd={_fmt(qd0, 2)}  "
        f"a_oracle={_fmt(g0)}  "
        f"max|q-q*|={mx:.3f} mean|q-q*|={mn:.3f} bal={bal}  "
        f"q_raw={_fmt(q_raw0, 3)}",
        flush=True)

  obs1, _, _ = env.step(oracle_a)
  print("", flush=True)
  for i in range(args.num_envs):
    packed0 = obs0[i].detach().cpu().numpy()
    packed1 = obs1[i].detach().cpu().numpy()
    q1, qd1, _ = _split_state(packed1, n)
    mx, mn, _ = _phys_err(env, i)
    bal = _balanced(env, i)
    d = _state_delta(packed0, packed1)
    print(
        f"env={i} t=1  q_norm={_fmt(q1)}  qd={_fmt(qd1, 2)}  "
        f"max|q-q*|={mx:.3f} mean|q-q*|={mn:.3f} bal={bal}  "
        f"Δmax|q_n|={d['max_abs_dq_norm']:.4f} "
        f"Δmax|qd|={d['max_abs_dqd']:.3f} "
        f"Δl2(s)={d['l2_state']:.4f}",
        flush=True)

  checkpoints = sorted({10, 20, int(args.horizon)})
  for step in range(2, args.horizon + 1):
    obs_t, _, _ = env.step(oracle_a)
    if step not in checkpoints:
      continue
    print(f"\n--- oracle t={step} ---", flush=True)
    for i in range(args.num_envs):
      packed = obs_t[i].detach().cpu().numpy()
      q, qd, _ = _split_state(packed, n)
      mx, mn, _ = _phys_err(env, i)
      bal = _balanced(env, i)
      print(
          f"env={i} t={step}  q_norm={_fmt(q)}  qd={_fmt(qd, 2)}  "
          f"max|q-q*|={mx:.3f} mean|q-q*|={mn:.3f} bal={bal}",
          flush=True)

  # ---- Pass 2: same-seed reset, one-step random vs oracle ----
  print("\n--- PASS 2: one-step Δ under oracle vs random (same reset) ---\n",
        flush=True)
  obs_a = env.reset()
  starts = [
      obs_a[i].detach().cpu().numpy().copy() for i in range(args.num_envs)]
  max_before = [_phys_err(env, i)[0] for i in range(args.num_envs)]

  obs_or, _, _ = env.step(oracle_a)
  oracle_deltas = []
  for i in range(args.num_envs):
    d = _state_delta(starts[i], obs_or[i].detach().cpu().numpy())
    oracle_deltas.append(d)
    print(
        f"env={i} ORACLE 1-step  max|q-q*|: {max_before[i]:.3f}→"
        f"{_phys_err(env, i)[0]:.3f}  "
        f"Δmax|q_n|={d['max_abs_dq_norm']:.4f} "
        f"Δmax|qd|={d['max_abs_dqd']:.3f} "
        f"Δl2={d['l2_state']:.4f}",
        flush=True)

  obs_b = env.reset()
  starts_b = [
      obs_b[i].detach().cpu().numpy().copy() for i in range(args.num_envs)]
  match = all(
      np.allclose(starts[i][:16], starts_b[i][:16], atol=1e-4)
      for i in range(args.num_envs))
  print(f"[oracle_samples] reset replay match={match}", flush=True)

  rng = np.random.RandomState(123)
  rand_a = torch.as_tensor(
      rng.uniform(-1.0, 1.0, size=(args.num_envs, n)).astype(np.float32),
      device=env.device)
  obs_rnd, _, _ = env.step(rand_a)
  random_deltas = []
  for i in range(args.num_envs):
    d = _state_delta(starts_b[i], obs_rnd[i].detach().cpu().numpy())
    random_deltas.append(d)
    print(
        f"env={i} RANDOM 1-step  a={_fmt(rand_a[i].detach().cpu().numpy())}  "
        f"Δmax|q_n|={d['max_abs_dq_norm']:.4f} "
        f"Δmax|qd|={d['max_abs_dqd']:.3f} "
        f"Δl2={d['l2_state']:.4f}",
        flush=True)

  print("\n--- SUMMARY (mean over envs) ---", flush=True)
  print(
      f"oracle  Δmax|q_n|={_mean_key(oracle_deltas, 'max_abs_dq_norm'):.4f}  "
      f"Δmax|qd|={_mean_key(oracle_deltas, 'max_abs_dqd'):.3f}  "
      f"Δl2={_mean_key(oracle_deltas, 'l2_state'):.4f}",
      flush=True)
  print(
      f"random  Δmax|q_n|={_mean_key(random_deltas, 'max_abs_dq_norm'):.4f}  "
      f"Δmax|qd|={_mean_key(random_deltas, 'max_abs_dqd'):.3f}  "
      f"Δl2={_mean_key(random_deltas, 'l2_state'):.4f}",
      flush=True)
  ratio = (
      _mean_key(oracle_deltas, "l2_state")
      / max(_mean_key(random_deltas, "l2_state"), 1e-8))
  print(
      f"oracle/random Δl2 ratio={ratio:.3f} "
      "(~1 means oracle is not a clearer local move than random)",
      flush=True)

  os.makedirs(args.out_dir, exist_ok=True)
  csv_path = os.path.join(
      args.out_dir,
      f"its_{args.trim_init_mode}_oracle_vs_random_1step.csv")
  with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(
        fh,
        fieldnames=[
            "env", "max_err_t0",
            "oracle_delta_max_q_norm", "oracle_delta_max_qd", "oracle_delta_l2",
            "random_delta_max_q_norm", "random_delta_max_qd", "random_delta_l2",
            "q_norm_t0", "qd_t0", "a_oracle", "a_random",
        ])
    w.writeheader()
    for i in range(args.num_envs):
      q0, qd0, _ = _split_state(starts[i], n)
      w.writerow({
          "env": i,
          "max_err_t0": f"{max_before[i]:.4f}",
          "oracle_delta_max_q_norm":
              f"{oracle_deltas[i]['max_abs_dq_norm']:.6f}",
          "oracle_delta_max_qd": f"{oracle_deltas[i]['max_abs_dqd']:.6f}",
          "oracle_delta_l2": f"{oracle_deltas[i]['l2_state']:.6f}",
          "random_delta_max_q_norm":
              f"{random_deltas[i]['max_abs_dq_norm']:.6f}",
          "random_delta_max_qd": f"{random_deltas[i]['max_abs_dqd']:.6f}",
          "random_delta_l2": f"{random_deltas[i]['l2_state']:.6f}",
          "q_norm_t0": _fmt(q0),
          "qd_t0": _fmt(qd0, 2),
          "a_oracle": _fmt(g_norm),
          "a_random": _fmt(rand_a[i].detach().cpu().numpy()),
      })
  print(f"\nwrote {csv_path}", flush=True)

  txt_path = os.path.join(
      args.out_dir,
      f"its_{args.trim_init_mode}_oracle_transitions.txt")
  with open(txt_path, "w") as fh:
    fh.write(
        f"goal_raw_rad={_fmt(g_raw, 4)}\n"
        f"goal_norm={_fmt(g_norm, 4)}\n"
        f"init={args.trim_init_mode} frac={args.trim_init_range_frac:g}\n"
        f"oracle_mean_dL2={_mean_key(oracle_deltas, 'l2_state'):.4f}\n"
        f"random_mean_dL2={_mean_key(random_deltas, 'l2_state'):.4f}\n"
        f"oracle_over_random_dL2={ratio:.3f}\n")
  print(f"wrote {txt_path}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
