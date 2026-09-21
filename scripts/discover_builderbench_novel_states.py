#!/usr/bin/env python3
"""Cumulative unique discretized cube states from DISCOVER rollouts.

Same method as scripts/ppo_nf_builderbench_novel_states.py: 5 stochastic
trajs per checkpoint, bin cube xyz with tolerance ε, accumulate bins that
have never been visited. Wired for the successful DISCOVER run (c3t1 catwp
seed 1).

  python scripts/discover_builderbench_novel_states.py --seed=1
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = Path(__file__).resolve().parents[1]
_BB = Path(os.environ.get("BUILDERBENCH_ROOT", "/n/fs/mislresearch/builderbench"))
_PAPER_PY = _REPO / "paper material" / "paper plot pythons"
_DISC = _REPO / "baseline-agents" / "discover"
for p in (_REPO, _BB, _DISC, _PAPER_PY):
  s = str(p)
  if s not in sys.path:
    sys.path.insert(0, s)

CSV_FIELDS = (
    "epoch",
    "env_steps",
    "n_steps",
    "n_unique_ckpt",
    "n_new",
    "n_cumulative",
)

RUN_DIR = (
    _REPO / "logs"
    / "discover_builderbench_creative3_task1_e1024_pd_catwp_ucbstd0_nopermute_fixedx01_warp_8h"
    / "discover_builderbench_creative_3_task1_1"
)


def _novel_csv(run_dir: Path, seed: int, eps: float) -> Path:
  return run_dir / "novel_states" / f"seed{seed}_eps{eps:g}.csv"


def _plot_path() -> Path:
  return (
      _REPO / "paper material" / "paper plots" / "bb_discover_c3t1_novel_states"
  )


def _read_success_csv(path: Path) -> list[tuple[int, int, float]]:
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      ep = int(float(row["epoch"]))
      x = int(float(row["global_step"]))
      y = float(row["eval_success_any_microstep"])
      rows.append((ep, x, y))
  rows.sort()
  return rows


def _success_markers(rows: list[tuple[int, int, float]], threshold: float):
  first_nz = next(((ep, x, y) for ep, x, y in rows if y > 0.0), None)
  first_ok = next(((ep, x, y) for ep, x, y in rows if y >= threshold), None)
  if first_ok is None:
    raise SystemExit(f"no eval_success_any_microstep >= {threshold}")
  return first_nz, first_ok


def _read_csv(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      rows.append({k: int(float(row[k])) for k in CSV_FIELDS})
  rows.sort(key=lambda r: r["epoch"])
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


def _maybe_fix_target(state, fixed_target_goal, mocap_targets, num_cubes: int):
  del num_cubes
  if fixed_target_goal is None:
    return state
  import jax.numpy as jnp
  from envs.builderbench_utils import set_task_mocap_pos
  fixed = jnp.asarray(fixed_target_goal, dtype=jnp.float32).reshape(-1)
  fixed_pos = fixed.reshape(int(fixed.shape[0] // 3), 3)
  info = dict(state.info)
  info["target_goal"] = jnp.broadcast_to(fixed, state.info["target_goal"].shape)
  info["target_mocap_pos"] = jnp.broadcast_to(
      fixed_pos, state.info["target_mocap_pos"].shape)
  mocap_pos = set_task_mocap_pos(
      state.data.mocap_pos, mocap_targets, fixed_pos)
  data = state.data.replace(mocap_pos=mocap_pos)
  return state.replace(data=data, info=info)


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
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--run_dir", default=str(RUN_DIR))
  p.add_argument("--num_traj", type=int, default=5)
  p.add_argument("--success_threshold", type=float, default=0.1)
  p.add_argument(
      "--bin_size", type=float, default=0.02,
      help="Cube-xyz bin width (m). Default = env success_threshold.")
  p.add_argument("--exploration_noise", type=float, default=0.4)
  p.add_argument("--noise_clip", type=float, default=0.5)
  p.add_argument("--csv_output", default="")
  p.add_argument("--plot_output", default="")
  p.add_argument("--plot_only", action="store_true")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  if args.seed != 1:
    raise SystemExit("this script is wired for the successful DISCOVER run: c3t1 catwp seed 1")
  run_dir = Path(args.run_dir)
  success_path = run_dir / "logs.csv"
  dist_path = (
      Path(args.csv_output) if args.csv_output
      else _novel_csv(run_dir, args.seed, args.bin_size))
  plot_path = Path(args.plot_output) if args.plot_output else _plot_path()
  title = fr"DISCOVER c3t1 unique cube states  ($\varepsilon$={args.bin_size:g}m)"

  success_rows = _read_success_csv(success_path)
  first_nz, first_ok = _success_markers(success_rows, args.success_threshold)
  print(f"[disc_novel] run_dir={run_dir}")
  print(f"[disc_novel] success_csv={success_path}")
  if first_nz is None:
    print("[disc_novel] first nonzero: none")
  else:
    print(
        f"[disc_novel] first nonzero: epoch={first_nz[0]}  "
        f"steps={first_nz[1]/1e6:.1f}M  success={first_nz[2]:.4f}")
  print(
      f"[disc_novel] first >= {args.success_threshold}: epoch={first_ok[0]}  "
      f"steps={first_ok[1]/1e6:.1f}M  success={first_ok[2]:.4f}")

  if args.plot_only:
    rows = _read_csv(dist_path)
    if not rows:
      raise SystemExit(f"no CSV at {dist_path}")
    plot_novel(
        rows, first_ok=first_ok, eps=args.bin_size, out_path=plot_path,
        title=title)
    return

  import sgcrl_jax_acme_compat  # noqa: F401
  import jax
  import jax.numpy as jnp
  import numpy as np
  import bb_env
  import networks as nets
  from builderbench.constants import _MJX_PARAMS
  from builderbench.creative_cube import CreativeCube, default_config
  from envs.builderbench_utils import (
      apply_fixed_start_x,
      creative_cube_mj_episode_length,
      filter_pd_policy_state_obs,
      parse_bb_env_id,
      sgcrl_env_name_to_bb_env_id,
  )
  from utils.wrapper import EpisodeWrapper, PDWrapper, VmapWrapper

  ckpt_dir = run_dir / "checkpoints"
  ckpts = sorted(
      ckpt_dir.glob("ckpt_epoch_*.pkl"),
      key=lambda p: int(p.stem.split("_")[-1]),
  )
  if not ckpts:
    raise SystemExit(f"no ckpts in {ckpt_dir}")
  print(f"[disc_novel] {len(ckpts)} ckpts  eps={args.bin_size:g}m")
  print(f"[disc_novel] csv={dist_path} jax={jax.default_backend()} {jax.devices()}")

  first_payload = pickle.loads(ckpts[0].read_bytes())
  env_name = str(first_payload.get("env", "builderbench_creative_3_task1"))
  hidden = tuple(int(x) for x in first_payload.get("hidden", (256, 256)))
  use_wp = bool(first_payload.get("categorical_select_waypoint", True))
  if not use_wp:
    raise SystemExit("expected catwp checkpoint")

  env_id = sgcrl_env_name_to_bb_env_id(env_name)
  num_cubes, task_id = parse_bb_env_id(env_id)
  n_cubes = int(num_cubes)
  g_star = bb_env.task_goal(n_cubes, task_id)
  actor_def = nets.CatWpActor(n_cubes=n_cubes, hidden=hidden)

  cfg = default_config()
  cfg.num_cubes = n_cubes
  cfg.task_id = task_id
  cfg.episode_length = int(creative_cube_mj_episode_length(n_cubes, task_id))
  cfg.permute_start_boxes = False
  cfg.impl = os.environ.get("BUILDERBENCH_MJX_IMPL", "warp")
  if env_id in _MJX_PARAMS:
    cfg.nconmax, cfg.njmax = _MJX_PARAMS[env_id]
  cfg.nconmax = max(int(cfg.nconmax), 64)
  cfg.njmax = max(int(cfg.njmax), 784)
  base = CreativeCube(config=cfg)
  apply_fixed_start_x(base, 0.1)
  mocap_targets = base._task_mocap_targets
  pd_duration = 5
  assert cfg.episode_length % pd_duration == 0
  episode_length = cfg.episode_length // pd_duration
  env = EpisodeWrapper(
      VmapWrapper(PDWrapper(base, duration=pd_duration)),
      episode_length, 1)
  n_traj = args.num_traj
  pos_dim = n_cubes * 3
  eps = float(args.bin_size)
  expl = float(args.exploration_noise)
  nclip = float(args.noise_clip)
  print(
      f"[disc_novel] MJX impl={cfg.impl} n_cubes={n_cubes} "
      f"macro_ep_len={episode_length} explore_noise={expl}")

  def generate_unroll(actor_params, key):
    reset_key, unroll_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, n_traj))
    state = _maybe_fix_target(state, g_star, mocap_targets, n_cubes)

    def body(carry, _):
      env_state, key = carry
      key, act_key = jax.random.split(key)
      obs = filter_pd_policy_state_obs(env_state.obs, n_cubes)
      packed = jnp.concatenate(
          [obs, env_state.info["target_goal"]], axis=-1)
      wp, yaw, logits = actor_def.apply(actor_params, packed)
      action = nets.catwp_action(
          wp, yaw, logits, act_key, n_cubes, True, expl, nclip)
      next_state = env.step(env_state, action)
      next_state = _maybe_fix_target(
          next_state, g_star, mocap_targets, n_cubes)
      pos = next_state.obs[:, :pos_dim]
      return (next_state, key), pos

    pos0 = state.obs[:, :pos_dim]
    (_, _), pos_seq = jax.lax.scan(
        body, (state, unroll_key), None, length=episode_length)
    return jnp.concatenate([pos0[None], pos_seq], axis=0)

  run_unroll = jax.jit(generate_unroll)
  key = jax.random.PRNGKey(args.seed + 91_007)
  seen: set[tuple] = set()
  rows: list[dict] = []

  for i, param_file in enumerate(ckpts):
    t0 = time.time()
    payload = pickle.loads(param_file.read_bytes())
    epoch = int(payload.get("epoch", int(param_file.stem.split("_")[-1])))
    env_steps = int(payload.get("env_steps", 0))
    actor_params = payload["actor_params"]
    key, unroll_key = jax.random.split(key)
    pos = np.asarray(run_unroll(actor_params, unroll_key))
    bins = np.floor(pos / eps).astype(np.int32).reshape(-1, pos_dim)
    n_steps = int(bins.shape[0])
    ckpt_keys = {tuple(r.tolist()) for r in bins}
    n_unique = len(ckpt_keys)
    n_new = sum(1 for k in ckpt_keys if k not in seen)
    seen.update(ckpt_keys)
    row = {
        "epoch": epoch,
        "env_steps": env_steps,
        "n_steps": n_steps,
        "n_unique_ckpt": n_unique,
        "n_new": n_new,
        "n_cumulative": len(seen),
    }
    rows.append(row)
    _write_csv(dist_path, rows)
    print(
        f"[disc_novel] {param_file.name}  step={env_steps/1e6:.1f}M  "
        f"new={n_new}  unique_ckpt={n_unique}  cum={len(seen)}  "
        f"{time.time()-t0:.1f}s  ({i+1}/{len(ckpts)})",
        flush=True,
    )

  plot_novel(
      rows, first_ok=first_ok, eps=args.bin_size, out_path=plot_path,
      title=title)
  print(f"[disc_novel] wrote {dist_path}")
  print(f"[disc_novel] plot {plot_path}")


if __name__ == "__main__":
  main()
