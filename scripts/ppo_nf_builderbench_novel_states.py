#!/usr/bin/env python3
"""Cumulative unique discretized cube states from NF rollout trajectories.

From checkpoint 0 onward, roll out the same 5 stochastic trajs as the
Hungarian-distance probe, bin cube xyz with tolerance ε, and accumulate
bins that have never been visited. Stops 5 checkpoints after first eval
success >= 0.1.

  python scripts/ppo_nf_builderbench_novel_states.py \\
    --run_dir=.../ppo_builderbench_creative_3_task1_0 --seed=0
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = Path(__file__).resolve().parents[1]
_BB = Path(os.environ.get("BUILDERBENCH_ROOT", "/n/fs/mislresearch/builderbench"))
_PAPER_PY = _REPO / "paper material" / "paper plot pythons"
for p in (_REPO, _BB, _PAPER_PY):
  s = str(p)
  if s not in sys.path:
    sys.path.insert(0, s)

import sgcrl_jax_acme_compat  # noqa: F401

CSV_FIELDS = (
    "iteration",
    "env_steps",
    "n_steps",
    "n_unique_ckpt",
    "n_new",
    "n_cumulative",
)


def _load_bb_video():
  path = _REPO / "scripts" / "ppo_builderbench_rollout_video.py"
  spec = importlib.util.spec_from_file_location("bb_video_novel", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _iter_to_steps(run_dir: Path) -> dict[int, int]:
  path = run_dir / "logs" / "learner" / "logs.csv"
  it_map: dict[int, int] = {}
  if not path.is_file():
    return it_map
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      try:
        it = int(float(row["iteration"]))
        gs = int(float(row["global_step"]))
      except (KeyError, ValueError, TypeError):
        continue
      it_map[it] = gs
  return it_map


def _eval_success_rows(run_dir: Path, it_map: dict[int, int]):
  path = run_dir / "logs" / "eval" / "logs.csv"
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      it = int(float(row["iteration"]))
      y = float(row["success"])
      x = it_map.get(it, it * 1024 * 50)
      rows.append((it, x, y))
  rows.sort()
  return rows


def _success_markers(rows, threshold: float):
  first_nz = next(((it, x, y) for it, x, y in rows if y > 0.0), None)
  first_ok = next(((it, x, y) for it, x, y in rows if y >= threshold), None)
  if first_ok is None:
    raise SystemExit(f"no eval success >= {threshold}")
  return first_nz, first_ok


def _read_csv(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      rows.append({k: int(float(row[k])) for k in CSV_FIELDS})
  rows.sort(key=lambda r: r["iteration"])
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
  nf_c = ps.C["blue"]
  ax.plot(
      xs, cum, color=nf_c, linewidth=ps.LW, solid_capstyle="round",
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
      Line2D([0], [0], color=nf_c, lw=ps.LW, marker="o",
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
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run_dir", required=True)
  p.add_argument("--num_traj", type=int, default=5)
  p.add_argument("--success_threshold", type=float, default=0.1)
  p.add_argument(
      "--bin_size", type=float, default=0.02,
      help="Cube-xyz bin width (m). Default = env success_threshold.")
  p.add_argument(
      "--after_success_ckpts", type=int, default=5,
      help="Keep this many ckpts after first eval>=0.1. "
           "Negative = all ckpts in run_dir.")
  p.add_argument(
      "--tail_run_dir", default="",
      help="Optional extra run dir (same recipe) for later ckpts.")
  p.add_argument(
      "--max_env_steps", type=int, default=0,
      help="Stop once a ckpt reaches this many env steps (0 = no cap).")
  p.add_argument("--csv_output", default="")
  p.add_argument("--plot_output", default="")
  p.add_argument("--plot_only", action="store_true")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  run_dir = Path(args.run_dir)
  dist_path = (
      Path(args.csv_output) if args.csv_output
      else run_dir / "novel_states" / f"seed{args.seed}_eps{args.bin_size:g}.csv")
  plot_path = Path(args.plot_output) if args.plot_output else (
      _REPO / "paper material" / "paper plots" / "bb_nf_c3t1_novel_states")
  cfg_path = run_dir / "run_config.json"
  with cfg_path.open(encoding="utf-8") as fh:
    run_cfg = json.load(fh)
  env_name = str(run_cfg.get("env"))
  flags = run_cfg.get("flags", {})
  mj_ep_len = flags.get("builderbench_mj_episode_length")
  it_map = _iter_to_steps(run_dir)
  eval_rows = _eval_success_rows(run_dir, it_map)
  first_nz, first_ok = _success_markers(eval_rows, args.success_threshold)
  ckpt_dir = run_dir / "checkpoints"
  ckpt_iters = sorted(
      int(p.stem.split("_")[-1])
      for p in ckpt_dir.glob("ckpt_iter_*.pkl"))
  if not ckpt_iters:
    raise SystemExit(f"no ckpts in {ckpt_dir}")
  # Interval from the dense probe (every 10).
  gaps = [b - a for a, b in zip(ckpt_iters, ckpt_iters[1:]) if b > a]
  interval = min(gaps) if gaps else 10
  if int(args.after_success_ckpts) < 0:
    last_it = ckpt_iters[-1]
  else:
    last_it = int(first_ok[0]) + int(args.after_success_ckpts) * interval
  pending = [it for it in ckpt_iters if it <= last_it]
  spi = 1024 * 50
  jobs: list[tuple[Path, int]] = [
      (ckpt_dir / f"ckpt_iter_{it:07d}.pkl", it) for it in pending
  ]
  if args.tail_run_dir:
    tail_dir = Path(args.tail_run_dir)
    tail_ckpt_dir = tail_dir / "checkpoints"
    tail_map = _iter_to_steps(tail_dir)
    probe_last_steps = max(
        it_map.get(it, (it + 1) * spi) for it in pending)
    max_steps = int(args.max_env_steps)
    tail_its = sorted(
        int(p.stem.split("_")[-1])
        for p in tail_ckpt_dir.glob("ckpt_iter_*.pkl"))
    for it in tail_its:
      est = int(tail_map.get(it, (it + 1) * spi))
      if est <= probe_last_steps:
        continue
      jobs.append((tail_ckpt_dir / f"ckpt_iter_{it:07d}.pkl", it))
      if max_steps > 0 and est >= max_steps:
        break
  print(f"[novel] run_dir={run_dir}")
  print(f"[novel] env={env_name}  eps={args.bin_size:g}m")
  print(
      f"[novel] first >= {args.success_threshold}: iter={first_ok[0]}  "
      f"steps={first_ok[1]/1e6:.1f}M")
  print(
      f"[novel] {len(jobs)} ckpts  {jobs[0][1]}-{jobs[-1][1]}  "
      f"max_env_steps={args.max_env_steps or 'none'}")
  title = fr"NF c3t1 unique cube states  ($\varepsilon$={args.bin_size:g}m)"

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
  from contrastive import ppo_learner
  from envs.builderbench_utils import (
      apply_fixed_start_x,
      filter_pd_policy_state_obs,
      parse_bb_env_id,
      sgcrl_env_name_to_bb_env_id,
  )
  from builderbench.constants import _MJX_PARAMS
  from builderbench.creative_cube import CreativeCube, default_config
  from utils.wrapper import EpisodeWrapper, PDWrapper, VmapWrapper

  bb_video = _load_bb_video()
  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, task_id = parse_bb_env_id(env_id)
  ctx = bb_video._load_train_ctx(env_name, str(ckpt_dir))
  networks = bb_video._build_networks(env_name, seed=args.seed, ctx=ctx)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  if mj_ep_len is None:
    raise SystemExit("run_config missing builderbench_mj_episode_length")
  cfg.episode_length = int(mj_ep_len)
  cfg.permute_start_boxes = bool(ctx.permute_start_boxes)
  cfg.impl = os.environ.get("BUILDERBENCH_MJX_IMPL", "warp")
  if env_id in _MJX_PARAMS:
    cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]
  cfg.nconmax = max(int(cfg.nconmax), 64)
  cfg.njmax = max(int(cfg.njmax), 784)
  base = CreativeCube(config=cfg)
  apply_fixed_start_x(base, ctx.fixed_start_x)
  mocap_targets = base._task_mocap_targets
  assert cfg.episode_length % ctx.pd_duration == 0
  episode_length = cfg.episode_length // ctx.pd_duration
  env = EpisodeWrapper(
      VmapWrapper(PDWrapper(base, duration=ctx.pd_duration)),
      episode_length, 1)
  fixed_goal = ctx.fixed_target_goal
  n_traj = args.num_traj
  filter_obs = bool(ctx.filter_policy_obs)
  pos_dim = int(num_cubes) * 3
  eps = float(args.bin_size)
  print(
      f"[novel] MJX impl={cfg.impl} n_cubes={num_cubes} "
      f"macro_ep_len={episode_length} jax={jax.default_backend()}")

  def generate_unroll(policy_params, key):
    reset_key, unroll_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, n_traj))
    state = bb_video._maybe_fix_target(
        state, fixed_goal, mocap_targets, num_cubes)

    def body(carry, _):
      env_state, key = carry
      key, act_key = jax.random.split(key)
      obs = env_state.obs
      if filter_obs:
        obs = filter_pd_policy_state_obs(obs, num_cubes)
      packed = jnp.concatenate(
          [obs, env_state.info["target_goal"]], axis=-1)
      dist = networks.policy_network.apply(policy_params, packed)
      action = networks.sample(dist, act_key)
      next_state = env.step(env_state, action)
      next_state = bb_video._maybe_fix_target(
          next_state, fixed_goal, mocap_targets, num_cubes)
      pos = next_state.obs[:, :pos_dim]
      return (next_state, key), pos

    pos0 = state.obs[:, :pos_dim]
    (_, _), pos_seq = jax.lax.scan(
        body, (state, unroll_key), None, length=episode_length)
    return jnp.concatenate([pos0[None], pos_seq], axis=0)

  run_unroll = jax.jit(generate_unroll)
  # Same stream as the Hungarian-distance probe.
  key = jax.random.PRNGKey(args.seed + 40_011)
  seen: set[tuple] = set()
  rows: list[dict] = []

  for i, (param_file, it) in enumerate(jobs):
    t0 = time.time()
    ckpt = ppo_learner.load_checkpoint(str(param_file))
    env_steps = int(ckpt.get("global_step", it_map.get(it, (it + 1) * 1024 * 50)))
    key, unroll_key = jax.random.split(key)
    pos = np.asarray(run_unroll(ckpt["policy_params"], unroll_key))
    bins = np.floor(pos / eps).astype(np.int32).reshape(-1, pos_dim)
    n_steps = int(bins.shape[0])
    ckpt_keys = {tuple(r.tolist()) for r in bins}
    n_unique = len(ckpt_keys)
    n_new = sum(1 for k in ckpt_keys if k not in seen)
    seen.update(ckpt_keys)
    row = {
        "iteration": it,
        "env_steps": env_steps,
        "n_steps": n_steps,
        "n_unique_ckpt": n_unique,
        "n_new": n_new,
        "n_cumulative": len(seen),
    }
    rows.append(row)
    _write_csv(dist_path, rows)
    print(
        f"[novel] {param_file}  step={env_steps/1e6:.1f}M  "
        f"new={n_new}  unique_ckpt={n_unique}  cum={len(seen)}  "
        f"{time.time()-t0:.1f}s  ({i+1}/{len(jobs)})",
        flush=True,
    )

  plot_novel(
      rows, first_ok=first_ok, eps=args.bin_size, out_path=plot_path,
      title=title)
  print(f"[novel] wrote {dist_path}")
  print(f"[novel] plot {plot_path}")


if __name__ == "__main__":
  main()
