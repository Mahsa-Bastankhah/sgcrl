#!/usr/bin/env python3
"""Re-eval PPO+RND BuilderBench params_*.pkl (eval-fix recipe) to CSV.

Writes logs/final_baselines/rnd_bb_{task}/eval_success/seed{seed}.csv with
eval_step, env_steps, eval_success. Matches training: PD, residual 6x256,
CatSelect, fixed start x, fixed goal, warp MJX.

  python scripts/ppo_rnd_builderbench_checkpoint_eval.py --task_key=c3t1 --seed=0
"""
from __future__ import annotations

import argparse
import csv
import functools
import importlib.util
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

_REPO = Path(__file__).resolve().parents[1]
_BB = Path(os.environ.get("BUILDERBENCH_ROOT", "/n/fs/mislresearch/builderbench"))
for p in (_REPO, _BB, _REPO / "baseline-agents"):
  s = str(p)
  if s not in sys.path:
    sys.path.insert(0, s)

TASKS = {
    "c3t1": dict(
        env_id="creative-3-task1", num_timesteps=200_000_000,
        env_episode_length=None),
    "c4t1": dict(
        env_id="creative-4-task1", num_timesteps=200_000_000,
        env_episode_length=250),
    "c4t2": dict(
        env_id="creative-4-task2", num_timesteps=200_000_000,
        env_episode_length=250),
    "c5t2": dict(
        env_id="creative-5-task2", num_timesteps=200_000_000,
        env_episode_length=None),
    "c5t4": dict(
        env_id="creative-5-task4", num_timesteps=200_000_000,
        env_episode_length=None),
    "c7t2": dict(
        env_id="creative-7-task2", num_timesteps=300_000_000,
        env_episode_length=None),
    "c8t2": dict(
        env_id="creative-8-task2", num_timesteps=300_000_000,
        env_episode_length=None),
}

NUM_EVAL_STEPS = 50
CSV_FIELDS = ("eval_step", "env_steps", "eval_success")


def _load_ppo_rnd():
  path = _REPO / "baseline-agents" / "ppo-rnd.py"
  spec = importlib.util.spec_from_file_location("sgcrl_ppo_rnd", path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {path}")
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def _ckpt_dir(task_key: str, seed: int) -> Path:
  root = _REPO / "logs" / "final_baselines" / f"rnd_bb_{task_key}" / "checkpoints"
  if not root.is_dir():
    raise FileNotFoundError(root)
  pat = re.compile(rf".*__{seed}__ppo-rnd__")
  matches = sorted(p for p in root.iterdir() if p.is_dir() and pat.search(p.name))
  if not matches:
    raise FileNotFoundError(f"no seed {seed} dir under {root}")
  return matches[0]


def _csv_path(task_key: str, seed: int) -> Path:
  return (
      _REPO / "logs" / "final_baselines" / f"rnd_bb_{task_key}"
      / "eval_success" / f"seed{seed}.csv"
  )


def _read_done(path: Path) -> set[int]:
  if not path.is_file():
    return set()
  done = set()
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      try:
        done.add(int(row["eval_step"]))
      except (KeyError, ValueError):
        continue
  return done


def _append_row(path: Path, row: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  new = not path.is_file()
  with path.open("a", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new:
      w.writeheader()
    w.writerow(row)


def write_zero_csv(task_key: str, seed: int) -> Path:
  meta = TASKS[task_key]
  path = _csv_path(task_key, seed)
  path.parent.mkdir(parents=True, exist_ok=True)
  step = int(meta["num_timesteps"] // NUM_EVAL_STEPS)
  with path.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    w.writeheader()
    for es in range(1, NUM_EVAL_STEPS + 1):
      w.writerow({
          "eval_step": es,
          "env_steps": es * step,
          "eval_success": 0.0,
      })
  return path


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--task_key", required=True, choices=sorted(TASKS))
  p.add_argument("--seed", type=int, required=True)
  p.add_argument("--num_eval_envs", type=int, default=128)
  p.add_argument("--stride", type=int, default=1)
  p.add_argument("--write_zeros", action="store_true",
                 help="Write an all-zero CSV and exit (no GPU).")
  p.add_argument("--checkpoint_dir", default="")
  p.add_argument("--csv_output", default="")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  meta = TASKS[args.task_key]
  out = Path(args.csv_output) if args.csv_output else _csv_path(
      args.task_key, args.seed)
  if args.write_zeros:
    write_zero_csv(args.task_key, args.seed)
    print(f"wrote zeros {out}")
    return

  import jax
  import jax.numpy as jnp
  from builderbench.env_utils import make_env
  from envs.builderbench_utils import apply_fixed_start_x, default_fixed_target_goal
  from utils.evaluation import Evaluator
  from utils.networks import load_params
  from utils.wrapper import wrap_env, PDWrapper

  ppo_rnd = _load_ppo_rnd()
  ckpt_dir = (
      Path(args.checkpoint_dir) if args.checkpoint_dir
      else _ckpt_dir(args.task_key, args.seed)
  )
  ckpts = sorted(
      ckpt_dir.glob("params_*.pkl"),
      key=lambda p: int(p.stem.split("_")[-1]),
  )
  if args.stride > 1:
    ckpts = [c for c in ckpts if int(c.stem.split("_")[-1]) % args.stride == 0]
  done = _read_done(out)
  pending = [c for c in ckpts if int(c.stem.split("_")[-1]) not in done]
  print(f"[rnd_eval] task={args.task_key} seed={args.seed} ckpt_dir={ckpt_dir}")
  print(f"[rnd_eval] {len(ckpts)} ckpts, {len(done)} done, {len(pending)} pending")
  print(f"[rnd_eval] csv={out} jax={jax.default_backend()} {jax.devices()}")
  if not pending:
    print("[rnd_eval] nothing to do")
    return

  rnd_args = ppo_rnd.Args(
      env_id=meta["env_id"],
      seed=args.seed,
      num_envs=args.num_eval_envs,
      num_eval_envs=args.num_eval_envs,
      num_timesteps=meta["num_timesteps"],
      env_episode_length=meta["env_episode_length"],
      use_pd=True,
      pd_duration=5,
      use_residual_mlp=True,
      categorical_select=True,
      permute_start_boxes=False,
      fixed_start_x=0.1,
      fix_goal=True,
  )
  env_class, default_config = make_env(rnd_args)
  default_config.impl = os.environ.get("BUILDERBENCH_MJX_IMPL", "warp")
  default_config.permute_start_boxes = False
  nc = int(re.search(r"creative-(\d+)", meta["env_id"]).group(1))
  ti = int(re.search(r"task(\d+)", meta["env_id"]).group(1)) - 1
  fixed_goal = jnp.asarray(default_fixed_target_goal(nc, ti), dtype=jnp.float32)
  print(f"[rnd_eval] MJX impl={default_config.impl} fixed_goal={fixed_goal.tolist()}")

  def _make_env():
    base = env_class(config=default_config)
    apply_fixed_start_x(base, 0.1)
    base = PDWrapper(base, duration=5)
    return ppo_rnd.FixedGoalWrapper(base, fixed_goal)

  assert default_config.episode_length % 5 == 0
  episode_length = default_config.episode_length // 5
  eval_env = wrap_env(ppo_rnd.HardSuccessRewardWrapper(_make_env()), episode_length)
  print(f"[rnd_eval] macro_ep_len={episode_length} action={eval_env.action_size}")

  ppo_network = ppo_rnd.make_ppo_networks(
      rnd_args, eval_env.action_size, include_auxiliary=False)
  make_policy = ppo_rnd.make_inference_fn(ppo_network)
  key = jax.random.PRNGKey(args.seed + 17_831)
  evaluator = Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=True),
      num_eval_envs=args.num_eval_envs,
      episode_length=episode_length,
      key=key,
  )
  step_size = int(meta["num_timesteps"] // NUM_EVAL_STEPS)

  for i, param_file in enumerate(pending):
    es = int(param_file.stem.split("_")[-1])
    t0 = time.time()
    params, normalizer, _ = load_params(str(param_file))
    metrics = evaluator.run_evaluation(
        policy_params={"policy": params["policy"], "normalizer": normalizer},
        training_metrics={},
    )
    y = float(metrics["eval/episode_success_rate"])
    x = es * step_size
    _append_row(out, {"eval_step": es, "env_steps": x, "eval_success": y})
    print(
        f"[rnd_eval] {param_file.name}  step={x/1e6:.1f}M  "
        f"success={y:.4f}  {time.time()-t0:.1f}s  "
        f"({i+1}/{len(pending)})",
        flush=True,
    )
  print(f"[rnd_eval] wrote {out}")


if __name__ == "__main__":
  main()
