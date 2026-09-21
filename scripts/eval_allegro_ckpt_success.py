#!/usr/bin/env python3
"""Offline Allegro checkpoint success at a chosen episode length.

Reuses the run's ``run_config.json`` packing. PhysX is created before JAX.

  python scripts/eval_allegro_ckpt_success.py \
    --checkpoint-dir logs/.../checkpoints \
    --output-csv figs/.../ckpt_eval_ep200.csv \
    --output-plot figs/.../ckpt_eval_ep200.png \
    --episode-length 200 --num-envs 256 --deterministic
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts import allegro_kuka_throw_ckpt_video as ckpt_utils  # noqa: E402

CKPT_RE = re.compile(r"ckpt_iter_(\d+)\.pkl$")
PARAMS_RE = re.compile(r"params_(\d+)\.pkl$")


def _iteration(path: str) -> int:
  name = os.path.basename(path)
  match = CKPT_RE.search(name) or PARAMS_RE.search(name)
  if match is None:
    raise ValueError(f"not a checkpoint filename: {path}")
  return int(match.group(1))


def _list_ckpt_paths(ckpt_dir: str) -> list[str]:
  paths = []
  for pat in ("ckpt_iter_*.pkl", "params_*.pkl"):
    paths.extend(glob.glob(os.path.join(ckpt_dir, pat)))
  paths = [p for p in paths if not os.path.basename(p).startswith("latest")]
  return sorted(set(paths), key=_iteration)


def _apply_stride(paths: list[str], stride: int) -> list[str]:
  if int(stride) <= 1 or not paths:
    return paths
  picked = [p for p in paths if _iteration(p) % int(stride) == 0]
  if paths[-1] not in picked:
    picked.append(paths[-1])
  return sorted(picked, key=_iteration)


def _rollout_metrics(env, act, policy_params, key, n_steps: int):
  import jax
  import jax.numpy as jnp
  import torch

  obs_t = env.reset()
  n_env = int(env.num_envs)
  ep_hard = np.zeros((n_env,), dtype=np.float32)
  ep_easy = np.zeros((n_env,), dtype=np.float32)
  ep_veasy = np.zeros((n_env,), dtype=np.float32)
  ep_j10 = np.zeros((n_env,), dtype=np.float32)
  ep_j20 = np.zeros((n_env,), dtype=np.float32)
  first_hard = np.full((n_env,), np.nan, dtype=np.float32)
  last_hard = np.zeros((n_env,), dtype=np.float32)
  min_dist = np.full((n_env,), np.inf, dtype=np.float32)
  ep_in_bucket = np.zeros((n_env,), dtype=np.float32)
  first_in_bucket = np.full((n_env,), np.nan, dtype=np.float32)
  last_mae = None
  frozen = np.zeros((n_env,), dtype=bool)
  task = env._env
  prev_obj = None
  for t in range(int(n_steps)):
    obs = np.asarray(obs_t.detach().cpu().numpy(), dtype=np.float32)
    if obs.ndim == 1:
      obs = obs[None, :]
    key, subkey = jax.random.split(key)
    action = act(policy_params, jnp.asarray(obs), subkey)
    action_np = np.asarray(action, dtype=np.float32)
    obs_t, _, done = env.step(
        torch.as_tensor(action_np, dtype=torch.float32, device=env.device))
    levels = env.success_levels()
    hard = np.asarray(levels["hard"].detach().cpu().numpy(), dtype=np.float32)
    easy = np.asarray(levels["easy"].detach().cpu().numpy(), dtype=np.float32)
    veasy = np.asarray(
        levels["very_easy"].detach().cpu().numpy(), dtype=np.float32)
    j10 = np.asarray(
        levels["joint_frac_010"].detach().cpu().numpy(), dtype=np.float32)
    j20 = np.asarray(
        levels["joint_frac_020"].detach().cpu().numpy(), dtype=np.float32)
    last_mae = np.asarray(
        levels["mean_abs_joint_err"].detach().cpu().numpy(), dtype=np.float32)
    obj = np.asarray(task.object_pos.detach().cpu().numpy(), dtype=np.float32)
    goal = np.asarray(task.goal_pos.detach().cpu().numpy(), dtype=np.float32)
    dist = np.linalg.norm(obj - goal, axis=-1).astype(np.float32)
    # NVIDIA bucket.obj: floor at goal_z-0.05, height ≈ 0.198 m, radius ≈ 0.12 m.
    xy = np.hypot(obj[:, 0] - goal[:, 0], obj[:, 1] - goal[:, 1])
    z_in = obj[:, 2] - (goal[:, 2] - 0.05)
    in_bucket = (xy <= 0.12) & (z_in >= 0.0) & (z_in <= 0.198)
    # NVIDIA keypoint-success respawns the cube to object_init_state without
    # setting done. A >20 cm jump is that teleport, not physics.
    jumped = np.zeros((n_env,), dtype=bool)
    if prev_obj is not None:
      jumped = np.linalg.norm(obj - prev_obj, axis=-1) > 0.20
    prev_obj = obj
    alive = ~frozen
    new_hard = alive & (ep_hard < 0.5) & (hard >= 0.5)
    first_hard[new_hard] = float(t + 1)
    new_in = alive & (ep_in_bucket < 0.5) & in_bucket
    first_in_bucket[new_in] = float(t + 1)
    ep_hard[alive] = np.maximum(ep_hard[alive], hard[alive])
    ep_in_bucket[alive] = np.maximum(
        ep_in_bucket[alive], in_bucket[alive].astype(np.float32))
    ep_easy[alive] = np.maximum(ep_easy[alive], easy[alive])
    ep_veasy[alive] = np.maximum(ep_veasy[alive], veasy[alive])
    ep_j10[alive] = np.maximum(ep_j10[alive], j10[alive])
    ep_j20[alive] = np.maximum(ep_j20[alive], j20[alive])
    last_hard[alive] = hard[alive]
    min_dist[alive] = np.minimum(min_dist[alive], dist[alive])
    done_np = np.asarray(done.detach().cpu().numpy()).astype(bool).reshape(-1)
    frozen = frozen | done_np | jumped
  return {
      "success": float(ep_hard.mean()),
      "end_success": float(last_hard.mean()),
      "in_bucket": float(ep_in_bucket.mean()),
      "n_in_bucket": int(ep_in_bucket.sum()),
      "first_in_bucket_step": float(np.nanmean(first_in_bucket)),
      "easy_success": float(ep_easy.mean()),
      "very_easy_success": float(ep_veasy.mean()),
      "joint_frac_010": float(ep_j10.mean()),
      "joint_frac_020": float(ep_j20.mean()),
      "mean_abs_joint_err": float(last_mae.mean()) if last_mae is not None else float("nan"),
      "min_obj_goal_dist": float(min_dist.mean()),
      "first_success_step": float(np.nanmean(first_hard)),
      "n_success": int(ep_hard.sum()),
      "n_envs": n_env,
  }, key


def _load_csv_rows(path: str) -> list[dict]:
  if not os.path.isfile(path):
    return []
  with open(path, newline="") as fh:
    return list(csv.DictReader(fh))


def _write_csv(path: str, rows: list[dict]) -> None:
  os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
  with open(path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _plot(path: str, rows: list[dict], episode_length: int,
          title: str = "") -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  sys.path.insert(0, os.path.join(REPO, "scripts"))
  import plot_builderbench_train_success1000 as base  # noqa: E402

  xs = [float(r["env_steps"]) for r in rows]
  C = base.ACCENT_COLORS
  fig, ax = plt.subplots(figsize=(10.2, 4.6))
  throw_mode = all(
      abs(float(r["success"]) - float(r["easy_success"])) < 1e-8
      and float(r.get("joint_frac_010", 0.0)) == 0.0
      for r in rows)
  if throw_mode:
    series = (
        ("in_bucket", "ever inside bucket (cylinder, not 7.5cm)", C[4], "-"),
        ("success", "ever success (≤7.5 cm)", C[2], "-"),
        ("end_success", "end-of-ep still ≤7.5 cm", C[0], "--"),
    )
    ylabel = f"success (roll mean, w={base.EVAL_SMOOTH_WINDOW})"
  else:
    series = (
        ("success", "hard (all |err| ≤ 0.10)", C[3], "-"),
        ("easy_success", "easy (all |err| ≤ 0.20)", C[4], "--"),
        ("very_easy_success", "very-easy (all |err| ≤ 0.30)", C[2], ":"),
        ("joint_frac_010", "joint_frac@0.10 (ep-max)", C[0], "-"),
    )
    ylabel = f"success / jfrac (roll mean, w={base.EVAL_SMOOTH_WINDOW})"
  for key, label, color, ls in series:
    if key not in rows[0]:
      continue
    ys = [float(r[key]) for r in rows]
    base._plot_eval_smoothed(ax, xs, ys, color=color, label=label, linestyle=ls)
  ax.set_title(
      title or (
          f"offline ckpt eval  T={episode_length}  "
          f"(rolling mean, window={base.EVAL_SMOOTH_WINDOW})"),
      fontsize=11, fontweight="bold")
  ax.set_ylabel(ylabel, fontsize=10)
  ax.set_xlabel("Train env steps", fontsize=10)
  ax.set_ylim(-0.02, 1.05)
  ax.legend(loc="best", fontsize=8, framealpha=0.95)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.grid(axis="y", linestyle="--", alpha=0.4)
  import matplotlib.ticker as mticker
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint-dir", required=True)
  parser.add_argument("--output-csv", required=True)
  parser.add_argument("--output-plot", default="")
  parser.add_argument("--episode-length", type=int, default=200)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--iterations", default="",
                      help="comma-separated iters; empty = all ckpts")
  parser.add_argument("--last-n", type=int, default=0,
                      help="Only consider the N newest ckpts (0 = all).")
  parser.add_argument("--plot-title", default="")
  parser.add_argument("--pipeline", choices=("gpu", "cpu"), default="gpu")
  parser.add_argument("--deterministic", action="store_true", default=True)
  parser.add_argument("--stochastic", action="store_true",
                      help="Use stochastic π instead of deterministic.")
  parser.add_argument("--steps-per-iter", type=int, default=1024 * 50)
  parser.add_argument("--ckpt-stride", type=int, default=1,
                      help="Keep iters divisible by this (plus the last ckpt).")
  parser.add_argument("--flags-json", default="",
                      help="Merge extra env flags (needed for PPO+RND).")
  args = parser.parse_args()
  deterministic = not bool(args.stochastic)

  paths = _apply_stride(
      _list_ckpt_paths(args.checkpoint_dir), int(args.ckpt_stride))
  if args.iterations.strip():
    requested = {
        int(v.strip()) for v in args.iterations.split(",") if v.strip()}
    paths = [p for p in paths if _iteration(p) in requested]
    found = {_iteration(p) for p in paths}
    if found != requested:
      raise FileNotFoundError(
          f"missing ckpt iters: {sorted(requested - found)}")
  if int(args.last_n) > 0:
    paths = paths[-int(args.last_n):]
  existing = _load_csv_rows(args.output_csv)
  done_iters = {int(r["iteration"]) for r in existing if r.get("iteration")}
  todo = [p for p in paths if _iteration(p) not in done_iters]
  if not paths:
    raise FileNotFoundError(
        f"no ckpt_iter_*.pkl or params_*.pkl in {args.checkpoint_dir}")
  if not todo:
    print(f"[ckpt_eval] nothing new (have {len(existing)} rows)", flush=True)
    if existing and args.output_plot:
      try:
        _plot(args.output_plot, existing, int(args.episode_length),
              title=args.plot_title)
        print(f"wrote {args.output_plot}", flush=True)
      except Exception as exc:  # noqa: BLE001
        print(f"[ckpt_eval] plot skipped: {exc}", flush=True)
    return

  flags = ckpt_utils._load_flags(paths[0])
  if args.flags_json:
    with open(args.flags_json, "r", encoding="utf-8") as fh:
      extra = json.load(fh)
    if not isinstance(extra, dict):
      raise ValueError("--flags-json must be a JSON object")
    flags = {**flags, **extra}
    print(f"[ckpt_eval] merged flags from {args.flags_json}", flush=True)
  env_kwargs = ckpt_utils._load_env_kwargs(
      flags, int(args.episode_length), int(args.seed), args.pipeline)
  env_kwargs["num_envs"] = int(args.num_envs)
  env_kwargs["enable_cameras"] = False
  env = ckpt_utils._build_env(env_kwargs)
  print(
      f"[ckpt_eval] mode={getattr(env, 'control_sanity_mode', '')} "
      f"trim_sa={getattr(env, 'control_sanity_trim_sa', False)} "
      f"init={getattr(env, 'control_sanity_trim_init_mode', '')} "
      f"T={env.max_episode_steps} E={env.num_envs} "
      f"obs={env.obs_dim} act={env.action_dim} "
      f"det={deterministic}",
      flush=True)

  import jax

  rows_by_iter = {int(r["iteration"]): r for r in existing}
  print(
      f"[ckpt_eval] resume={len(existing)} todo={len(todo)} "
      f"last_n={int(args.last_n) or 'all'}",
      flush=True)
  for path in todo:
    ckpt = ckpt_utils._load_ckpt(path)
    act, policy_params, built_iter, _ = ckpt_utils._build_actor(
        env, ckpt, flags, deterministic=deterministic, ckpt_path=path)
    if isinstance(ckpt, dict) and ckpt.get("iteration") is not None:
      iteration = int(ckpt["iteration"])
    elif built_iter is not None and int(built_iter) >= 0:
      iteration = int(built_iter)
    else:
      iteration = _iteration(path)
    env_steps = int(iteration) * int(args.steps_per_iter)
    if isinstance(ckpt, dict) and ckpt.get("global_step") is not None:
      env_steps = int(ckpt["global_step"])
    key = jax.random.PRNGKey(int(args.seed) + iteration)
    metrics, _ = _rollout_metrics(
        env, act, policy_params, key, int(env.max_episode_steps))
    row = {
        "iteration": iteration,
        "env_steps": env_steps,
        "episode_length": int(env.max_episode_steps),
        "deterministic": int(deterministic),
        **metrics,
    }
    rows_by_iter[iteration] = row
    rows = [rows_by_iter[k] for k in sorted(rows_by_iter)]
    _write_csv(args.output_csv, rows)
    if args.output_plot:
      try:
        _plot(args.output_plot, rows, int(args.episode_length),
              title=args.plot_title)
      except Exception as exc:  # noqa: BLE001
        print(f'[ckpt_eval] plot skipped: {exc}', flush=True)
    print(
        f"[ckpt_eval] iter={iteration} steps={row['env_steps']/1e6:.1f}M "
        f"in_bkt={metrics.get('in_bucket', float('nan')):.4f} "
        f"ever={metrics['success']:.4f} end={metrics['end_success']:.4f} "
        f"min_d={metrics['min_obj_goal_dist']:.3f} "
        f"t_succ={metrics['first_success_step']:.1f} "
        f"n_bkt={metrics.get('n_in_bucket', 0)} "
        f"n_succ={metrics['n_success']}/{metrics['n_envs']}",
        flush=True)

  rows = [rows_by_iter[k] for k in sorted(rows_by_iter)]
  print(f"wrote {args.output_csv}  n={len(rows)}", flush=True)
  if args.output_plot:
    print(f"wrote {args.output_plot}", flush=True)


if __name__ == "__main__":
  main()
