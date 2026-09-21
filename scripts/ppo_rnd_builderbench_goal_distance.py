#!/usr/bin/env python3
"""Stochastic Hungarian goal-distance for PPO+RND checkpoints before success.

Loads eval_success CSV for first-nonzero and first eval_success >= 0.1
markers, rolls out N stochastic trajs on every checkpoint, and records
mean Hungarian cube-goal distance (env obj_goal_dist / n_task_cubes)
at the last episode step.

Also marks the first nonzero eval success on the training-steps plot.

  python scripts/ppo_rnd_builderbench_goal_distance.py --task_key=c3t1 --seed=1
  python scripts/ppo_rnd_builderbench_goal_distance.py --plot_only
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
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = Path(__file__).resolve().parents[1]
_BB = Path(os.environ.get("BUILDERBENCH_ROOT", "/n/fs/mislresearch/builderbench"))
_PAPER_PY = _REPO / "paper material" / "paper plot pythons"
_SCRIPTS = _REPO / "scripts"
for p in (_REPO, _BB, _REPO / "baseline-agents", _PAPER_PY, _SCRIPTS):
  s = str(p)
  if s not in sys.path:
    sys.path.insert(0, s)

TASKS = {
    "c3t1": dict(
        env_id="creative-3-task1", num_timesteps=200_000_000,
        env_episode_length=None),
}

NUM_EVAL_STEPS = 50
CSV_FIELDS = (
    "eval_step",
    "env_steps",
    "mean_term_dist",
    "se_term_dist",
    "traj0",
    "traj1",
    "traj2",
    "traj3",
    "traj4",
)


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


def _success_csv(task_key: str, seed: int) -> Path:
  return (
      _REPO / "logs" / "final_baselines" / f"rnd_bb_{task_key}"
      / "eval_success" / f"seed{seed}.csv"
  )


def _dist_csv(task_key: str, seed: int) -> Path:
  return (
      _REPO / "logs" / "final_baselines" / f"rnd_bb_{task_key}"
      / "goal_distance" / f"seed{seed}.csv"
  )


def _plot_path() -> Path:
  return (
      _REPO / "paper material" / "paper plots"
      / "bb_rnd_c3t1_presuccess_goal_dist"
  )


def _read_success_csv(path: Path) -> list[tuple[int, int, float]]:
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      es = int(row["eval_step"])
      x = int(float(row["env_steps"]))
      y = float(row["eval_success"])
      rows.append((es, x, y))
  rows.sort()
  return rows


def _success_markers(rows: list[tuple[int, int, float]], threshold: float):
  first_nz = next(((es, x, y) for es, x, y in rows if y > 0.0), None)
  first_ok = next(((es, x, y) for es, x, y in rows if y >= threshold), None)
  if first_ok is None:
    raise SystemExit(f"no eval_success >= {threshold} in success CSV")
  return first_nz, first_ok


def _read_dist_csv(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      rows.append({
          "eval_step": int(row["eval_step"]),
          "env_steps": int(float(row["env_steps"])),
          "mean_term_dist": float(row["mean_term_dist"]),
          "se_term_dist": float(row["se_term_dist"]),
          "trajs": [float(row[f"traj{i}"]) for i in range(5)],
      })
  rows.sort(key=lambda r: r["eval_step"])
  return rows


def _write_dist_csv(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    w.writeheader()
    for row in rows:
      out = {
          "eval_step": row["eval_step"],
          "env_steps": row["env_steps"],
          "mean_term_dist": f"{row['mean_term_dist']:.8f}",
          "se_term_dist": f"{row['se_term_dist']:.8f}",
      }
      trajs = list(row["trajs"])
      while len(trajs) < 5:
        trajs.append(float("nan"))
      for i in range(5):
        out[f"traj{i}"] = f"{trajs[i]:.8f}"
      w.writerow(out)


def _fmt_steps(v, _=None):
  if v == 0:
    return "0"
  if v >= 1e6:
    s = f"{v / 1e6:.1f}M"
    return s.replace(".0M", "M")
  if v >= 1e3:
    return f"{v / 1e3:.0f}K"
  return str(int(v))


def _fmt_m(x: int) -> str:
  return _fmt_steps(x)


def plot_distance(
    rows: list[dict],
    *,
    first_nz: tuple[int, int, float] | None,
    first_ok: tuple[int, int, float],
    out_path: Path,
) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker
  from matplotlib.lines import Line2D
  import paper_style as ps

  ps.apply()
  fig, ax = ps.figure("single")

  if rows and int(rows[0]["env_steps"]) > 0:
    head = dict(rows[0])
    head["env_steps"] = 0
    rows = [head] + list(rows)

  xs = [r["env_steps"] for r in rows]
  mean = [r["mean_term_dist"] for r in rows]
  se = [r["se_term_dist"] for r in rows]
  color = ps.C["vermillion"]
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  jx = int(first_ok[1])
  before = [r for r in rows if int(r["env_steps"]) < jx]
  after = [r for r in rows if int(r["env_steps"]) > jx]
  def _line(rs):
    if not rs:
      return
    xs_ = [r["env_steps"] for r in rs]
    mean_ = [r["mean_term_dist"] for r in rs]
    se_ = [r["se_term_dist"] for r in rs]
    lo_ = [m - s for m, s in zip(mean_, se_)]
    hi_ = [m + s for m, s in zip(mean_, se_)]
    ax.fill_between(xs_, lo_, hi_, color=color, alpha=0.22, lw=0, zorder=2)
    ax.plot(xs_, mean_, color=color, linewidth=ps.LW, solid_capstyle="round", zorder=3)
  _line(before)
  _line(after)
  if before and after:
    y0 = float(before[-1]["mean_term_dist"])
    y1 = float(after[0]["mean_term_dist"])
    ax.plot(
        [before[-1]["env_steps"], jx], [y0, y0], color=color, linewidth=ps.LW,
        solid_capstyle="butt", zorder=3)
    ax.plot(
        [jx, jx], [y0, y1], color=color, linewidth=ps.LW,
        solid_capstyle="butt", zorder=3)
    ax.plot(
        [jx, after[0]["env_steps"]], [y1, y1], color=color, linewidth=ps.LW,
        solid_capstyle="butt", zorder=3)
  for r in rows:
    ax.plot(
        [r["env_steps"]] * 5, r["trajs"], linestyle="None", marker="o",
        markersize=4.0, markeredgewidth=0.0, color=color, alpha=0.22,
        zorder=3)

  ok_x = first_ok[1]
  ax.axvline(ok_x, color=color, linestyle=ps.DASH, linewidth=2.2, zorder=1)
  ok_y = None
  for r in rows:
    if r["env_steps"] == ok_x:
      ok_y = r["mean_term_dist"]
      break
  if ok_y is not None:
    ax.plot(
        [ok_x], [ok_y], marker="o", markersize=ps.MS,
        markerfacecolor="white", markeredgewidth=ps.MEW,
        markeredgecolor=color, zorder=5)
  handles = [
      Line2D([0], [0], color=color, lw=ps.LW, label="RND"),
      Line2D(
          [0], [0], color=ps.C["gray"], linestyle=ps.DASH, linewidth=2.2,
          label="First success"),
  ]

  ax.set_xlabel("Environment steps")
  ax.set_ylabel("Hungarian distance to goal")
  ax.set_title("PPO+RND c3t1 seed 1")
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  if xs:
    ax.set_xlim(0, max(ok_x, xs[-1]) * 1.04)
  ax.set_ylim(bottom=0)
  ps.style_axes(ax, which="major")
  ps.nice_legend(ax, handles=handles, loc="upper right")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  ps.savefig(fig, out_path)
  plt.close(fig)


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--task_key", default="c3t1", choices=sorted(TASKS))
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--num_traj", type=int, default=5)
  p.add_argument("--success_threshold", type=float, default=0.1)
  p.add_argument("--checkpoint_dir", default="")
  p.add_argument("--success_csv", default="")
  p.add_argument("--csv_output", default="")
  p.add_argument("--plot_output", default="")
  p.add_argument(
      "--plot_only", action="store_true",
      help="Replot from an existing distance CSV (no GPU).")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  if args.task_key != "c3t1" or args.seed != 1:
    raise SystemExit("this script is wired for the successful RND run: c3t1 seed 1")
  if args.num_traj != 5:
    raise SystemExit("CSV columns are fixed at 5 trajs")

  meta = TASKS[args.task_key]
  success_path = (
      Path(args.success_csv) if args.success_csv
      else _success_csv(args.task_key, args.seed))
  dist_path = (
      Path(args.csv_output) if args.csv_output
      else _dist_csv(args.task_key, args.seed))
  plot_path = Path(args.plot_output) if args.plot_output else _plot_path()

  success_rows = _read_success_csv(success_path)
  first_nz, first_ok = _success_markers(success_rows, args.success_threshold)
  print(f"[rnd_dist] success_csv={success_path}")
  if first_nz is None:
    print("[rnd_dist] first nonzero: none")
  else:
    print(
        f"[rnd_dist] first nonzero: eval={first_nz[0]}  "
        f"steps={first_nz[1]/1e6:.1f}M  success={first_nz[2]:.4f}")
  print(
      f"[rnd_dist] first >= {args.success_threshold}: eval={first_ok[0]}  "
      f"steps={first_ok[1]/1e6:.1f}M  success={first_ok[2]:.4f}")

  if args.plot_only:
    rows = _read_dist_csv(dist_path)
    if not rows:
      raise SystemExit(f"no distance CSV at {dist_path}")
    plot_distance(rows, first_nz=first_nz, first_ok=first_ok, out_path=plot_path)
    return

  import jax
  import jax.numpy as jnp
  import numpy as np
  from builderbench.env_utils import make_env
  from envs.builderbench_utils import apply_fixed_start_x, default_fixed_target_goal
  from utils.networks import load_params
  from utils.wrapper import VmapWrapper, EpisodeWrapper, PDWrapper

  ppo_rnd = _load_ppo_rnd()
  ckpt_dir = (
      Path(args.checkpoint_dir) if args.checkpoint_dir
      else _ckpt_dir(args.task_key, args.seed)
  )
  ckpts = sorted(
      ckpt_dir.glob("params_*.pkl"),
      key=lambda p: int(p.stem.split("_")[-1]),
  )
  step_size = int(meta["num_timesteps"] // NUM_EVAL_STEPS)
  pending = []
  for c in ckpts:
    es = int(c.stem.split("_")[-1])
    env_steps = es * step_size
    pending.append((es, env_steps, c))
  done = {r["eval_step"]: r for r in _read_dist_csv(dist_path)}
  todo = [p for p in pending if p[0] not in done]
  print(f"[rnd_dist] ckpt_dir={ckpt_dir}")
  print(
      f"[rnd_dist] {len(pending)} ckpts, {len(done)} done, "
      f"{len(todo)} pending")
  print(f"[rnd_dist] csv={dist_path} jax={jax.default_backend()} {jax.devices()}")

  if todo:
    rnd_args = ppo_rnd.Args(
        env_id=meta["env_id"],
        seed=args.seed,
        num_envs=args.num_traj,
        num_eval_envs=args.num_traj,
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
    n_task = int(fixed_goal.size // 3)
    print(f"[rnd_dist] MJX impl={default_config.impl} n_task_cubes={n_task}")

    def _make_env():
      base = env_class(config=default_config)
      apply_fixed_start_x(base, 0.1)
      base = PDWrapper(base, duration=5)
      return ppo_rnd.FixedGoalWrapper(base, fixed_goal)

    assert default_config.episode_length % 5 == 0
    episode_length = default_config.episode_length // 5
    # Skip AutoResetWrapper: the last done step would replace terminal
    # obj_goal_dist with the reset (start-of-episode) distance.
    env = EpisodeWrapper(VmapWrapper(_make_env()), episode_length, 1)
    print(f"[rnd_dist] macro_ep_len={episode_length} action={env.action_size}")

    ppo_network = ppo_rnd.make_ppo_networks(
        rnd_args, env.action_size, include_auxiliary=False)
    make_policy = ppo_rnd.make_inference_fn(ppo_network)
    policy_from_params = functools.partial(make_policy, deterministic=False)
    n_traj = args.num_traj

    def generate_unroll(policy_params, key):
      reset_key, unroll_key = jax.random.split(key)
      state = env.reset(jax.random.split(reset_key, n_traj))
      policy = policy_from_params(policy_params)

      def body(_i, carry):
        env_state, key = carry
        key, act_key = jax.random.split(key)
        actions, _ = policy(
            env_state.obs, env_state.info["target_goal"], act_key)
        return (env.step(env_state, actions), key)

      final_state, _ = jax.lax.fori_loop(
          0, episode_length, body, (state, unroll_key))
      return final_state.metrics["obj_goal_dist"] / float(n_task)

    run_unroll = jax.jit(generate_unroll)
    key = jax.random.PRNGKey(args.seed + 91_007)

    for i, (es, env_steps, param_file) in enumerate(todo):
      t0 = time.time()
      params, normalizer, _ = load_params(str(param_file))
      key, unroll_key = jax.random.split(key)
      dists = np.asarray(run_unroll(
          {"policy": params["policy"], "normalizer": normalizer},
          unroll_key,
      ))
      dists = np.reshape(dists, (n_traj,))
      mean = float(dists.mean())
      se = float(dists.std(ddof=1) / np.sqrt(n_traj)) if n_traj > 1 else 0.0
      done[es] = {
          "eval_step": es,
          "env_steps": env_steps,
          "mean_term_dist": mean,
          "se_term_dist": se,
          "trajs": [float(x) for x in dists.tolist()],
      }
      ordered = [done[k] for k in sorted(done)]
      _write_dist_csv(dist_path, ordered)
      print(
          f"[rnd_dist] {param_file.name}  step={env_steps/1e6:.1f}M  "
          f"dist={mean:.4f}±{se:.4f}  {time.time()-t0:.1f}s  "
          f"({i+1}/{len(todo)})",
          flush=True,
      )

  rows = [done[k] for k in sorted(done)]
  if not rows:
    raise SystemExit("no distance rows to plot")
  plot_distance(rows, first_nz=first_nz, first_ok=first_ok, out_path=plot_path)
  print(f"[rnd_dist] wrote {dist_path}")


if __name__ == "__main__":
  main()
