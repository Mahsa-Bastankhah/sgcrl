#!/usr/bin/env python3
"""Cumulative unique discretized cube states from RND rollout trajectories.

Same method as scripts/ppo_nf_builderbench_novel_states.py: 5 stochastic
trajs per checkpoint, bin cube xyz with tolerance ε, accumulate bins that
have never been visited. Wired for the successful PPO+RND run (c3t1 seed 1).

  python scripts/ppo_rnd_builderbench_novel_states.py --task_key=c3t1 --seed=1
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

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = Path(__file__).resolve().parents[1]
_BB = Path(os.environ.get("BUILDERBENCH_ROOT", "/n/fs/mislresearch/builderbench"))
_PAPER_PY = _REPO / "paper material" / "paper plot pythons"
for p in (_REPO, _BB, _REPO / "baseline-agents", _PAPER_PY):
  s = str(p)
  if s not in sys.path:
    sys.path.insert(0, s)

TASKS = {
    "c3t1": dict(
        env_id="creative-3-task1", num_timesteps=200_000_000,
        env_episode_length=None, num_cubes=3),
}

NUM_EVAL_STEPS = 50
CSV_FIELDS = (
    "eval_step",
    "env_steps",
    "n_steps",
    "n_unique_ckpt",
    "n_new",
    "n_cumulative",
)


def _load_ppo_rnd():
  path = _REPO / "baseline-agents" / "ppo-rnd.py"
  spec = importlib.util.spec_from_file_location("sgcrl_ppo_rnd_novel", path)
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


def _novel_csv(task_key: str, seed: int, eps: float) -> Path:
  return (
      _REPO / "logs" / "final_baselines" / f"rnd_bb_{task_key}"
      / "novel_states" / f"seed{seed}_eps{eps:g}.csv"
  )


def _plot_path() -> Path:
  return (
      _REPO / "paper material" / "paper plots" / "bb_rnd_c3t1_novel_states"
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


def _read_csv(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      rows.append({k: int(float(row[k])) for k in CSV_FIELDS})
  rows.sort(key=lambda r: r["eval_step"])
  return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    w.writeheader()
    for row in rows:
      w.writerow({k: row[k] for k in CSV_FIELDS})


def _fmt_steps(v, _=None):
  if v == 0:
    return "0"
  if v >= 1e6:
    s = f"{v / 1e6:.1f}M"
    return s.replace(".0M", "M")
  if v >= 1e3:
    return f"{v / 1e3:.0f}K"
  return str(int(v))


def plot_novel(
    rows: list[dict],
    *,
    first_ok: tuple,
    eps: float,
    out_path: Path,
    title: str,
) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker
  from matplotlib.lines import Line2D
  import paper_style as ps

  ps.apply()
  fig, ax = ps.figure("single")
  xs = [r["env_steps"] for r in rows]
  cum = [r["n_cumulative"] for r in rows]
  new = [r["n_new"] for r in rows]
  line_c = ps.C["blue"]
  ax.plot(
      xs, cum, color=line_c, linewidth=ps.LW, solid_capstyle="round",
      zorder=3, marker="o", markersize=7)
  ax.bar(
      xs, new, width=max(xs[-1] * 0.018, 1) if xs else 1,
      color=ps.C["sky"], alpha=0.55, zorder=2, align="center")
  ok_x = int(first_ok[1])
  ax.axvline(ok_x, color=ps.C["green"], linestyle=ps.DASH, linewidth=2.2, zorder=1)
  ax.set_xlabel("Environment steps")
  ax.set_ylabel("Discretized cube states")
  ax.set_title(title)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  ax.set_xlim(0, (max(xs) if xs else ok_x) * 1.08)
  ax.set_ylim(bottom=0)
  ps.style_axes(ax, which="major")
  handles = [
      Line2D([0], [0], color=line_c, lw=ps.LW, marker="o",
             label="cumulative unique"),
      Line2D([0], [0], color=ps.C["sky"], lw=8, label="new this ckpt"),
      Line2D(
          [0], [0], color=ps.C["green"], linestyle=ps.DASH, linewidth=2.2,
          label=fr"first eval $\geq$ 0.1 ({_fmt_steps(ok_x)})"),
  ]
  ps.nice_legend(ax, handles=handles, loc="upper left")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  ps.savefig(fig, out_path)
  plt.close(fig)
  _ = eps


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--task_key", default="c3t1", choices=sorted(TASKS))
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--num_traj", type=int, default=5)
  p.add_argument("--success_threshold", type=float, default=0.1)
  p.add_argument(
      "--bin_size", type=float, default=0.02,
      help="Cube-xyz bin width (m). Default = env success_threshold.")
  p.add_argument("--checkpoint_dir", default="")
  p.add_argument("--success_csv", default="")
  p.add_argument("--csv_output", default="")
  p.add_argument("--plot_output", default="")
  p.add_argument("--plot_only", action="store_true")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  if args.task_key != "c3t1" or args.seed != 1:
    raise SystemExit("this script is wired for the successful RND run: c3t1 seed 1")

  meta = TASKS[args.task_key]
  success_path = (
      Path(args.success_csv) if args.success_csv
      else _success_csv(args.task_key, args.seed))
  dist_path = (
      Path(args.csv_output) if args.csv_output
      else _novel_csv(args.task_key, args.seed, args.bin_size))
  plot_path = Path(args.plot_output) if args.plot_output else _plot_path()
  title = fr"RND c3t1 unique cube states  ($\varepsilon$={args.bin_size:g}m)"

  success_rows = _read_success_csv(success_path)
  first_nz, first_ok = _success_markers(success_rows, args.success_threshold)
  print(f"[rnd_novel] success_csv={success_path}")
  if first_nz is None:
    print("[rnd_novel] first nonzero: none")
  else:
    print(
        f"[rnd_novel] first nonzero: eval={first_nz[0]}  "
        f"steps={first_nz[1]/1e6:.1f}M  success={first_nz[2]:.4f}")
  print(
      f"[rnd_novel] first >= {args.success_threshold}: eval={first_ok[0]}  "
      f"steps={first_ok[1]/1e6:.1f}M  success={first_ok[2]:.4f}")

  if args.plot_only:
    rows = _read_csv(dist_path)
    if not rows:
      raise SystemExit(f"no CSV at {dist_path}")
    plot_novel(
        rows, first_ok=first_ok, eps=args.bin_size, out_path=plot_path,
        title=title)
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
  print(f"[rnd_novel] ckpt_dir={ckpt_dir}")
  print(f"[rnd_novel] {len(pending)} ckpts  eps={args.bin_size:g}m")
  print(f"[rnd_novel] csv={dist_path} jax={jax.default_backend()} {jax.devices()}")

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
  nc = int(meta["num_cubes"])
  ti = int(re.search(r"task(\d+)", meta["env_id"]).group(1)) - 1
  fixed_goal = jnp.asarray(default_fixed_target_goal(nc, ti), dtype=jnp.float32)
  print(f"[rnd_novel] MJX impl={default_config.impl} n_cubes={nc}")

  def _make_env():
    base = env_class(config=default_config)
    apply_fixed_start_x(base, 0.1)
    base = PDWrapper(base, duration=5)
    return ppo_rnd.FixedGoalWrapper(base, fixed_goal)

  assert default_config.episode_length % 5 == 0
  episode_length = default_config.episode_length // 5
  env = EpisodeWrapper(VmapWrapper(_make_env()), episode_length, 1)
  print(f"[rnd_novel] macro_ep_len={episode_length} action={env.action_size}")

  ppo_network = ppo_rnd.make_ppo_networks(
      rnd_args, env.action_size, include_auxiliary=False)
  make_policy = ppo_rnd.make_inference_fn(ppo_network)
  policy_from_params = functools.partial(make_policy, deterministic=False)
  n_traj = args.num_traj
  pos_dim = nc * 3
  eps = float(args.bin_size)

  def generate_unroll(policy_params, key):
    reset_key, unroll_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, n_traj))
    policy = policy_from_params(policy_params)

    def body(carry, _):
      env_state, key = carry
      key, act_key = jax.random.split(key)
      actions, _ = policy(
          env_state.obs, env_state.info["target_goal"], act_key)
      next_state = env.step(env_state, actions)
      pos = next_state.obs[:, :pos_dim]
      return (next_state, key), pos

    pos0 = state.obs[:, :pos_dim]
    (_, _), pos_seq = jax.lax.scan(
        body, (state, unroll_key), None, length=episode_length)
    return jnp.concatenate([pos0[None], pos_seq], axis=0)

  run_unroll = jax.jit(generate_unroll)
  # Same stream as the RND Hungarian-distance probe.
  key = jax.random.PRNGKey(args.seed + 91_007)
  seen: set[tuple] = set()
  rows: list[dict] = []

  for i, (es, env_steps, param_file) in enumerate(pending):
    t0 = time.time()
    params, normalizer, _ = load_params(str(param_file))
    key, unroll_key = jax.random.split(key)
    pos = np.asarray(run_unroll(
        {"policy": params["policy"], "normalizer": normalizer},
        unroll_key,
    ))
    bins = np.floor(pos / eps).astype(np.int32).reshape(-1, pos_dim)
    n_steps = int(bins.shape[0])
    ckpt_keys = {tuple(r.tolist()) for r in bins}
    n_unique = len(ckpt_keys)
    n_new = sum(1 for k in ckpt_keys if k not in seen)
    seen.update(ckpt_keys)
    row = {
        "eval_step": es,
        "env_steps": env_steps,
        "n_steps": n_steps,
        "n_unique_ckpt": n_unique,
        "n_new": n_new,
        "n_cumulative": len(seen),
    }
    rows.append(row)
    _write_csv(dist_path, rows)
    print(
        f"[rnd_novel] {param_file.name}  step={env_steps/1e6:.1f}M  "
        f"new={n_new}  unique_ckpt={n_unique}  cum={len(seen)}  "
        f"{time.time()-t0:.1f}s  ({i+1}/{len(pending)})",
        flush=True,
    )

  plot_novel(
      rows, first_ok=first_ok, eps=args.bin_size, out_path=plot_path,
      title=title)
  print(f"[rnd_novel] wrote {dist_path}")
  print(f"[rnd_novel] plot {plot_path}")


if __name__ == "__main__":
  main()
