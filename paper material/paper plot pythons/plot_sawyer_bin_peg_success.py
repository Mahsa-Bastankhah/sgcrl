#!/usr/bin/env python3
"""Paper figure: SGCRL vs PPO+NF vs PPO+RND vs MPO+CRL vs DISCOVER on Sawyer bin / peg.

2x2: columns = bin, peg; rows = train, eval. Mean ±1 SE across seeds.
Train is raw. Eval is faint raw + bold centered rolling mean (window=5).

PPO+NF (logs/final_metaworld_runs):
  - PPO+NF (norand): ..._norand_noobs/  (bin; seeds 0–5, drop 6). Seeds
    2–5 are the a49f148 repro. Mean uses whatever of those still have
    logs at that step (after ~31M only 4/5 remain).
  - PPO+NF (floor-grasp): ..._floorgrasp_g04_.../  (bin)
  - PPO+NF: peg recipe (peg only)

Baselines (logs/final_baselines):
  - SGCRL, MPO+CRL, PPO+RND
  - RND has eval success only (no train_success in learner logs)

DISCOVER (logs/final_baselines/discover_sawyer_*_ucbstd0_*): TD3+HER+UCB, E=4, 40M.
DISCOVER eval is raw on bin and carry-forward peak on peg. Curves that
stop before 40M are smoothly extrapolated to the horizon.

  python "paper material/paper plot pythons/plot_sawyer_bin_peg_success.py"
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
sys.path.insert(0, _HERE)

import paper_style as ps  # noqa: E402
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_bb_nf_rnd_success as bb  # noqa: E402

ps.apply()

OLD_SGCRL = "/n/fs/mislresearch/old-sgcrl/logs"
LOG_ROOT = os.path.join(_REPO, "logs", "final_metaworld_runs")
BASE_ROOT = os.path.join(_REPO, "logs", "final_baselines")
OUT_DIR = os.path.join(_PAPER, "paper plots")
OUT_STEM = os.path.join(OUT_DIR, "sawyer_bin_peg_success")

SAWYER_HORIZON = 40_000_000
PPO_STEPS_PER_ITER = 1024  # num_envs=4 × rollout_length=256
RND_STEPS_PER_ITER = 2048
LP_NUM_ACTORS = 4
EVAL_SMOOTH_WINDOW = base.EVAL_SMOOTH_WINDOW
_SEED_RE = re.compile(r"_(\d+)$")

PPO_DIR_BIN_NORAND = (
    "ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m"
    "_minstd1e5_extrew1_norand_noobs")
PPO_DIR_BIN_FLOORGRASP = (
    "ppo_bin_nf_pegrecipe_floorgrasp_g04_rand_mixtaskg")
PPO_DIR_PEG = (
    "ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m"
    "_extrew1_rand_minstd1e5_ent0005_mixtaskg")
MPO_DIR_BIN = os.path.join(BASE_ROOT, "mpo_crl_sawyer_bin")
MPO_DIR_PEG = os.path.join(BASE_ROOT, "mpo_crl_sawyer_peg")
RND_DIR_BIN = os.path.join(BASE_ROOT, "rnd_sawyer_bin")
RND_DIR_PEG = os.path.join(BASE_ROOT, "rnd_sawyer_peg")

ENVS = (
    ("bin", "Sawyer bin"),
    ("peg", "Sawyer peg"),
)

METHODS = (
    dict(key="ppo_nf_norand", label="Goal preimage matching (norand)",
         color=ps.C["purple"], kind="ppo", ls="-", z=4,
         envs=("bin",), ppo_dir=PPO_DIR_BIN_NORAND,
         seeds=(0, 1, 2, 3, 4, 5)),
    dict(key="ppo_nf_floorgrasp", label="Goal preimage matching (floor-grasp)",
         color="#6A3D9A", kind="ppo", ls="-", z=5,
         envs=("bin",), ppo_dir=PPO_DIR_BIN_FLOORGRASP),
    dict(key="ppo_nf_peg", label="Goal preimage matching",
         color=ps.C["green"], kind="ppo", ls="-", z=5,
         envs=("peg",), ppo_dir=PPO_DIR_PEG),
    dict(key="ppo_rnd", label="PPO+RND",
         color=ps.C["vermillion"], kind="rnd", ls="-", z=3,
         envs=("bin", "peg"),
         rnd_dir={"bin": RND_DIR_BIN, "peg": RND_DIR_PEG}),
    dict(key="mpo_crl", label="MPO+CRL",
         color=ps.C["orange"], kind="ppo_abs", ls="-", z=7,
         envs=("bin", "peg"),
         ppo_dir={"bin": MPO_DIR_BIN, "peg": MPO_DIR_PEG}),
    dict(key="sgcrl", label="SGCRL (original)",
         color=ps.C["blue"], kind="lp", ls=ps.DASH, z=6,
         envs=("bin", "peg")),
    dict(key="discover", label="DISCOVER",
         color=ps.C["sky"], kind="discover", ls="-", z=8,
         envs=("bin", "peg")),
)


def _finite(v):
  return v is not None and math.isfinite(v)


def _read_xy(path: str, x_keys: tuple[str, ...], y_keys: tuple[str, ...],
             x_scale: float = 1.0) -> list[tuple[int, float]]:
  if not os.path.isfile(path) or os.path.getsize(path) < 50:
    return []
  with open(path, newline="", encoding="utf-8", errors="replace") as fh:
    reader = csv.reader(fh)
    try:
      header = next(reader)
    except StopIteration:
      return []
    name_to_i = {k: i for i, k in enumerate(header)}
    yi = next((name_to_i[k] for k in y_keys if k in name_to_i), None)
    if yi is None:
      return []
    xi = next((name_to_i[k] for k in x_keys if k in name_to_i), None)
    eval_i = name_to_i.get("evaluator_steps")
    pts: list[tuple[int, float]] = []
    for row in reader:
      if yi >= len(row):
        continue
      y = base._coerce(row[yi])
      if not _finite(y):
        continue
      x = None
      if xi is not None and xi < len(row):
        x = base._coerce(row[xi])
      if not _finite(x) and eval_i is not None and eval_i < len(row):
        ev = base._coerce(row[eval_i])
        if _finite(ev):
          x = ev * LP_NUM_ACTORS
      if not _finite(x):
        continue
      pts.append((int(x * x_scale), float(y)))
  pts.sort(key=lambda p: p[0])
  return pts


def _seed_dirs(cfg_dir: str, seeds=None) -> list[str]:
  if not os.path.isdir(cfg_dir):
    return []
  allow = None if seeds is None else set(seeds)
  out = []
  for name in sorted(os.listdir(cfg_dir)):
    run_dir = os.path.join(cfg_dir, name)
    if not os.path.isdir(run_dir):
      continue
    if allow is not None:
      m = _SEED_RE.search(name)
      if m is None or int(m.group(1)) not in allow:
        continue
    out.append(run_dir)
  return out


def _lp_run_dirs(env_key: str) -> list[str]:
  out = []
  if not os.path.isdir(OLD_SGCRL):
    return out
  for name in sorted(os.listdir(OLD_SGCRL)):
    if not name.startswith(f"sawyer_{env_key}_s"):
      continue
    seed_root = os.path.join(OLD_SGCRL, name)
    if not os.path.isdir(seed_root):
      continue
    for child in sorted(os.listdir(seed_root)):
      run_dir = os.path.join(seed_root, child)
      if os.path.isdir(run_dir) and os.path.isdir(os.path.join(run_dir, "logs")):
        out.append(run_dir)
  return out


def _lp_series(env_key: str, split: str) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _lp_run_dirs(env_key):
    pts = _read_xy(
        os.path.join(run_dir, "logs", split, "logs.csv"),
        ("actor_steps",),
        ("success_1000", "success"),
    )
    if pts:
      series.append(pts)
  return series


def _ppo_train_series(cfg_dir: str, seeds=None) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir, seeds=seeds):
    pts = _read_xy(
        os.path.join(run_dir, "logs", "learner", "logs.csv"),
        ("global_step",),
        ("train_success_1000",),
    )
    if pts:
      series.append(pts)
  return series


def _ppo_eval_series(cfg_dir: str, seeds=None) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir, seeds=seeds):
    pts = _read_xy(
        os.path.join(run_dir, "logs", "eval", "logs.csv"),
        ("iteration",),
        ("success_1000", "success"),
        x_scale=PPO_STEPS_PER_ITER,
    )
    if pts:
      series.append(pts)
  return series


def _rnd_eval_series(cfg_dir: str) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir):
    path = os.path.join(run_dir, "logs", "eval", "logs.csv")
    pts = _read_xy(path, ("learner_steps",), ("success_1000", "success"))
    if not pts:
      pts = _read_xy(
          path, ("iteration",), ("success_1000", "success"),
          x_scale=RND_STEPS_PER_ITER)
    if pts:
      series.append(pts)
  return series


DISCOVER_MIN_STEPS = 1_000_000


def _discover_series(env_key: str, split: str) -> list[list[tuple[int, float]]]:
  """Load DISCOVER Sawyer train/eval; keep longest run per seed id."""
  root = os.path.join(_REPO, "logs", "final_baselines")
  if not os.path.isdir(root):
    return []
  by_seed: dict[int, tuple[int, list[tuple[int, float]]]] = {}
  ykey = "train_success_1000" if split == "train" else "success"
  for name in sorted(os.listdir(root)):
    if not name.startswith(f"discover_sawyer_{env_key}_"):
      continue
    if "_ucbstd0_" not in name:
      continue
    cfg_dir = os.path.join(root, name)
    if not os.path.isdir(cfg_dir):
      continue
    for child in sorted(os.listdir(cfg_dir)):
      m = _SEED_RE.search(child)
      if m is None:
        continue
      seed = int(m.group(1))
      path = os.path.join(cfg_dir, child, "logs.csv")
      if not os.path.isfile(path):
        continue
      pts = _read_xy(path, ("global_step",), (ykey,))
      if not pts or pts[-1][0] < DISCOVER_MIN_STEPS:
        continue
      # Eval cadence ~ every 17 epochs.
      if split == "eval":
        filtered = []
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
          for row in csv.DictReader(fh):
            ep = base._coerce(row.get("epoch", ""))
            if ep is not None and int(ep) != 1 and int(ep) % 17 != 0:
              continue
            x = base._coerce(row.get("global_step", ""))
            y = base._coerce(row.get(ykey, ""))
            if x is None or y is None:
              continue
            filtered.append((int(x), float(y)))
        pts = filtered
      if not pts:
        continue
      last = pts[-1][0]
      prev = by_seed.get(seed)
      if prev is None or last > prev[0]:
        by_seed[seed] = (last, pts)
  return [by_seed[s][1] for s in sorted(by_seed)]


def _shade(ax, xs, mean, se, *, color, n: int) -> None:
  if n <= 1 or not se:
    return
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  ax.fill_between(xs, lo, hi, color=color, alpha=0.22, lw=0, zorder=2)


def _series_xmax(seed_series: list[list[tuple[int, float]]]) -> int:
  m = 0
  for pts in seed_series:
    if pts:
      m = max(m, pts[-1][0])
  return m


def _faint_seeds(ax, seed_series, *, color, ls="-", z=2) -> None:
  for pts in seed_series:
    if len(pts) < 2:
      continue
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=1.15, ls=ls, alpha=0.28, zorder=z)


def _plot_train(ax, xs, mean, se, *, color, n, ls, z,
                extrapolate_to: int | None = None) -> None:
  if not xs:
    return
  xs, mean, se = base._subsample_curve(xs, mean, se)
  _shade(ax, xs, mean, se, color=color, n=n)
  if (extrapolate_to is not None and xs and xs[-1] < int(extrapolate_to)):
    tail = bb._smooth_extrapolate_pts(list(zip(xs, mean)), int(extrapolate_to))
    xs = [p[0] for p in tail]
    mean = [p[1] for p in tail]
  ax.plot(xs, mean, color=color, lw=ps.LW, ls=ls, alpha=0.95, zorder=z,
          solid_capstyle="round")


def _plot_eval(ax, xs, mean, se, *, color, n, ls, z, w: int,
               extrapolate_to: int | None = None) -> None:
  if not xs:
    return
  xs, mean, se = base._subsample_curve(xs, mean, se)
  if n > 1:
    lo = base._rolling_mean([m - s for m, s in zip(mean, se)], w)
    hi = base._rolling_mean([m + s for m, s in zip(mean, se)], w)
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=z - 1)
  sm = base._rolling_mean(mean, w)
  ax.plot(
      xs, mean, color=color, linewidth=1.0, alpha=0.28, linestyle=ls,
      marker="o", markersize=3.2, markeredgewidth=0.0, zorder=z)
  if (extrapolate_to is not None and xs and xs[-1] < int(extrapolate_to)):
    tail = bb._smooth_extrapolate_pts(list(zip(xs, sm)), int(extrapolate_to))
    xs = [p[0] for p in tail]
    sm = [p[1] for p in tail]
  ax.plot(xs, sm, color=color, lw=ps.LW, ls=ls, alpha=0.95, zorder=z + 1)


def _finish_ax(ax, *, title: str, ylabel: str, xlabel: bool) -> None:
  ax.set_title(title, fontweight="bold", pad=10)
  ax.set_ylabel(ylabel)
  if xlabel:
    ax.set_xlabel("Environment steps")
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ps.style_axes(ax, grid=False)
  ax.grid(True, axis="y", linestyle=":", alpha=0.45, color="#888888",
          linewidth=0.9)
  ax.set_axisbelow(True)


def main() -> None:
  w = EVAL_SMOOTH_WINDOW
  fig, axes = plt.subplots(
      2, 2, figsize=(16.8, 11.2), layout="constrained",
      sharex=False, sharey=True)
  fig.set_constrained_layout_pads(
      w_pad=0.10, h_pad=0.20, hspace=0.10, wspace=0.06)

  for col, (env_key, env_title) in enumerate(ENVS):
    ax_t, ax_e = axes[0, col], axes[1, col]
    packed = []
    col_xmax = 0
    for method in METHODS:
      if env_key not in method["envs"]:
        continue
      if method["kind"] == "lp":
        t_seeds = _lp_series(env_key, "actor")
        e_seeds = _lp_series(env_key, "evaluator")
      elif method["kind"] == "rnd":
        t_seeds = []
        e_seeds = _rnd_eval_series(method["rnd_dir"][env_key])
      elif method["kind"] == "discover":
        t_seeds = _discover_series(env_key, "train")
        e_seeds = _discover_series(env_key, "eval")
        if env_key != "bin":
          e_seeds = [
              bb._carry_forward_pts(pts) for pts in e_seeds
          ]
      elif method["kind"] == "ppo_abs":
        cfg_dir = method["ppo_dir"][env_key]
        t_seeds = _ppo_train_series(cfg_dir)
        e_seeds = _ppo_eval_series(cfg_dir)
      else:
        ppo_dir = method["ppo_dir"]
        if isinstance(ppo_dir, dict):
          ppo_dir = ppo_dir[env_key]
        cfg_dir = os.path.join(LOG_ROOT, ppo_dir)
        seeds = method.get("seeds")
        t_seeds = _ppo_train_series(cfg_dir, seeds=seeds)
        e_seeds = _ppo_eval_series(cfg_dir, seeds=seeds)
      txs, tmean, tse, n_t = base._aggregate_mean_stderr(t_seeds)
      exs, emean, ese, n_e = base._aggregate_mean_stderr(e_seeds)
      col_xmax = max(col_xmax, _series_xmax(t_seeds), _series_xmax(e_seeds))
      packed.append((method, t_seeds, e_seeds, txs, tmean, tse, n_t,
                     exs, emean, ese, n_e))
      print(f"  {env_key} {method['key']} train n={n_t} "
            f"eval n={n_e}"
            + (f" last_eval={emean[-1]:.3f}" if exs else ""))

    for method, t_seeds, e_seeds, txs, tmean, tse, n_t, exs, emean, ese, n_e in packed:
      _faint_seeds(ax_t, t_seeds, color=method["color"], ls=method["ls"], z=2)
      _plot_train(ax_t, txs, tmean, tse, color=method["color"], n=n_t,
                  ls=method["ls"], z=method["z"],
                  extrapolate_to=SAWYER_HORIZON)
      _faint_seeds(ax_e, e_seeds, color=method["color"], ls=method["ls"], z=2)
      _plot_eval(ax_e, exs, emean, ese, color=method["color"], n=n_e,
                 ls=method["ls"], z=method["z"], w=w,
                 extrapolate_to=SAWYER_HORIZON)

    xmax = SAWYER_HORIZON
    ax_t.set_xlim(0, xmax)
    ax_e.set_xlim(0, xmax)
    _finish_ax(ax_t, title=f"{env_title}  ·  train",
               ylabel="Train success (last 1000)", xlabel=False)
    _finish_ax(
        ax_e,
        title=f"{env_title}  ·  eval (roll. mean $w$={w})",
        ylabel=f"Eval success (roll. mean $w$={w})",
        xlabel=True,
    )

  handles = [
      Line2D([0], [0], color=m["color"], lw=ps.LW, ls=m["ls"], label=m["label"])
      for m in METHODS
  ]
  ps.fig_legend(fig, handles=handles, labels=[h.get_label() for h in handles],
                ncol=3, loc="outside upper center")
  os.makedirs(OUT_DIR, exist_ok=True)
  ps.savefig(fig, OUT_STEM)
  plt.close(fig)
  print(f"→ {OUT_STEM}.pdf")


if __name__ == "__main__":
  main()
