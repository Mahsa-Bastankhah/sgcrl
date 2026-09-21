#!/usr/bin/env python3
"""BuilderBench PPO+NF vs SGCRL vs PPO+RND vs MPO-CRL vs DISCOVER success.

Per-task side-by-side train | eval, plus combined 2x4 grids (train, eval).
Mean ±1 SE across seeds. Train is raw. Eval is faint raw + rolling mean w=21.
Mean uses the union of seed x (no min-end clip): shorter timeout seeds
contribute until they stop, longer finished seeds keep the curve going.
Keep incomplete original seeds until at least 3 runs finish ~200M;
extra in-progress seeds are then dropped.

SGCRL vanilla CRL from wandb sgcrl-vanilla-baselines (eval/episode_success_rate,
1 eval env). Cached under logs/final_baselines/sgcrl_vanilla_bb/*/eval_success.
c3t1/c4t1/c4t2/c5t2 have real seeds; c5t4/c7t2/c8t2 are zeros.
RND numbers come from logs/final_baselines/rnd_bb_*/eval_success/*.csv
(checkpoint re-eval of hard success; training stdout logs were not kept).
c3t1/c4t2 are re-evaled from params_*.pkl. Other tasks are saved as zeros.
MPO-CRL lives under logs/final_baselines/mpo_crl_builderbench_* (c3t1/c4t1/c4t2/c5t2/c5t4).
DISCOVER (catwp, UCB std=0) under
logs/final_baselines/discover_builderbench_*_warp_8h/.
DISCOVER eval curves use a per-seed running-max (carry-forward peak).

  python "paper material/paper plot pythons/plot_bb_nf_rnd_success.py"
"""
from __future__ import annotations

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import paper_style as ps  # noqa: E402
import plot_builderbench_train_success1000 as base  # noqa: E402

ps.apply()

# Wider than the default eval window=5 so the bold series is less jagged.
EVAL_SMOOTH_WINDOW = 21

OUT_DIR = os.path.join(_PAPER, "paper plots")
NF_ROOT = os.path.join(_REPO, "logs", "final_runs")
LOGS_ROOT = os.path.join(_REPO, "logs")
RND_ROOT = os.path.join(_REPO, "logs", "final_baselines")
MPO_ROOT = RND_ROOT
SGCRL_ROOT = os.path.join(RND_ROOT, "sgcrl_vanilla_bb")
RND_ZERO_TASKS = ("c4t1", "c5t2", "c5t4", "c7t2", "c8t2")
SGCRL_ZERO_TASKS = ("c5t4", "c7t2", "c8t2")
RND_NUM_EVAL_STEPS = 50
RND_TIMESTEPS = {
    "c3t1": 200_000_000,
    "c4t1": 200_000_000,
    "c4t2": 200_000_000,
    "c5t2": 200_000_000,
    "c5t4": 200_000_000,
    "c7t2": 300_000_000,
    "c8t2": 300_000_000,
}

NF_C3 = (
    "ppo_builderbench_creative3_task1_e1024_pd_nf_compact_small_"
    "sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_"
    "minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_"
    "valuedgr_c100_lamlr1e6_warp_4h"
)
NF_STEM_200M = (
    "e1024_pd_nf_compact_small_sa3x192_r64_b6_w192_tau05_"
    "nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_"
    "ep50_200m_crl10_dualgradreg_c100_lamlr1e6_"
    "valuedgr_c100_lamlr1e6_warp_2h30"
)
NF_C7 = (
    "ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small_sa3x192_"
    "r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_"
    "ent05to001_ep70_300m_crl10_dualgradreg_c100_lamlr1e6_"
    "valuedgr_c100_lamlr1e6_warp_4h"
)
NF_C8 = (
    "ppo_builderbench_creative8_task2_e1024_pd_nf_compact_small_sa3x192_"
    "r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_"
    "ent005_to001_ep100_300m_crl10_dualgradreg_c100_lamlr1e6_"
    "valuedgr_c100_lamlr1e6_warp_4h"
)
MPO_STEM = (
    "e1024_pd_ep50_200m_crl25_replay1m_nopermute_fixedx01_warp_7h"
)


def _mpo_cfg(name: str) -> str:
  return f"mpo_crl_builderbench_{name}_{MPO_STEM}"


TASKS = (
    dict(key="c3t1", name="creative3_task1", title="Creative 3 Task 1",
         nf=NF_C3, mpo=_mpo_cfg("creative3_task1")),
    dict(key="c4t1", name="creative4_task1", title="Creative 4 Task 1",
         nf=f"ppo_builderbench_creative4_task1_{NF_STEM_200M}",
         mpo=_mpo_cfg("creative4_task1")),
    dict(key="c4t2", name="creative4_task2", title="Two towers",
         nf=f"ppo_builderbench_creative4_task2_{NF_STEM_200M}",
         mpo=_mpo_cfg("creative4_task2")),
    dict(key="c5t2", name="creative5_task2", title="Pyramid",
         nf=f"ppo_builderbench_creative5_task2_{NF_STEM_200M}",
         mpo=_mpo_cfg("creative5_task2")),
    dict(key="c5t4", name="creative5_task4", title="Maximum overhang 5",
         nf=("ppo_builderbench_creative5_task4_e1024_pd_nf_compact_small_"
             "sa3x192_r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_"
             "minstd1e5_ent05to001_ep50_200m_crl10_dualgradreg_c100_lamlr1e6_"
             "valuedgr_c100_lamlr1e6_warp_4h"),
         mpo=_mpo_cfg("creative5_task4")),
    dict(key="c7t2", name="creative7_task2", title="7-cube zig-zag tower",
         nf=NF_C7),
    dict(key="c8t2", name="creative8_task2", title="8-cube Jenga tower",
         nf=NF_C8),
)

METHODS = (
    dict(key="nf", label="Goal preimage matching", color=ps.C["blue"]),
    dict(key="sgcrl", label="SGCRL", color=ps.C["purple"]),
    dict(key="rnd", label="PPO+RND", color=ps.C["vermillion"]),
    dict(key="mpo", label="MPO-CRL", color=ps.C["green"]),
    dict(key="discover", label="DISCOVER", color=ps.C["sky"]),
)

# DISCOVER catwp + UCB std=0 recipe (any-microstep eval; train_success_1000).
DISCOVER_CFG = {
    "c3t1": (
        "discover_builderbench_creative3_task1_e1024_pd_catwp_ucbstd0_"
        "nopermute_fixedx01_warp_8h"),
    "c4t1": (
        "discover_builderbench_creative4_task1_e1024_pd_catwp_ucbstd0_"
        "nopermute_fixedx01_warp_8h"),
    "c4t2": (
        "discover_builderbench_creative4_task2_e1024_pd_catwp_ucbstd0_"
        "nopermute_fixedx01_warp_8h"),
    "c5t2": (
        "discover_builderbench_creative5_task2_e1024_pd_catwp_ucbstd0_"
        "nopermute_fixedx01_warp_8h"),
    "c5t4": (
        "discover_builderbench_creative5_task4_e1024_pd_catwp_ucbstd0_"
        "nopermute_fixedx01_warp_8h"),
    "c7t2": (
        "discover_builderbench_creative7_task2_e1024_pd_catwp_ucbstd0_"
        "nopermute_fixedx01_warp_8h"),
    "c8t2": (
        "discover_builderbench_creative8_task2_e1024_pd_catwp_ucbstd0_"
        "nopermute_fixedx01_warp_8h"),
}
DISCOVER_ROOT = os.path.join(LOGS_ROOT, "final_baselines")
DISCOVER_MIN_STEPS = 1_000_000


def _carry_forward_pts(pts: list[tuple[int, float]]) -> list[tuple[int, float]]:
  """Running max: hold each seed at the best eval seen so far."""
  if not pts:
    return []
  out = []
  cur = float("-inf")
  for x, y in pts:
    cur = max(cur, float(y))
    out.append((int(x), cur))
  return out


def _carry_forward_series(
    series: list[list[tuple[int, float]]],
) -> list[list[tuple[int, float]]]:
  return [_carry_forward_pts(pts) for pts in series]


def _smooth_extrapolate_pts(
    pts: list[tuple[int, float]], xmax: int, *, n_tail: int = 16,
) -> list[tuple[int, float]]:
  """Hold the last logged value out to ``xmax``.

  Used when a Sawyer run stops short of 40M: continue the curve
  horizontally so every series reaches the panel horizon. Clamped to [0, 1].
  """
  if not pts or xmax is None:
    return list(pts)
  t0, y0 = int(pts[-1][0]), float(min(1.0, max(0.0, pts[-1][1])))
  if t0 >= int(xmax):
    return list(pts)
  remaining = float(int(xmax) - t0)
  out = list(pts)
  for i in range(1, n_tail + 1):
    t = int(xmax) if i == n_tail else t0 + int(round(remaining * i / n_tail))
    if t > t0:
      out.append((t, y0))
  return out


def _smooth_extrapolate_series(
    series: list[list[tuple[int, float]]], xmax: int,
) -> list[list[tuple[int, float]]]:
  return [_smooth_extrapolate_pts(pts, xmax) for pts in series if pts]


SPI_FALLBACK = 1024 * 50


def _run_last_step(run_dir: str) -> int | None:
  path = os.path.join(run_dir, "logs", "learner", "logs.csv")
  if not os.path.isfile(path):
    return None
  last = None
  with open(path, newline="", encoding="utf-8", errors="replace") as fh:
    for row in csv.DictReader(fh):
      x = base._coerce(row.get(base.TRAIN_X_COL, ""))
      if x is not None:
        last = int(x)
  return last


def _seed_dirs(cfg_dir: str) -> list[str]:
  if not os.path.isdir(cfg_dir):
    return []
  dirs = [
      os.path.join(cfg_dir, name)
      for name in sorted(os.listdir(cfg_dir))
      if os.path.isdir(os.path.join(cfg_dir, name))
  ]
  lasts = {d: _run_last_step(d) for d in dirs}
  finished = [d for d, s in lasts.items() if s is not None and s >= 180_000_000]
  # Only drop in-progress extras once at least 3 seeds have finished ~200M.
  # Otherwise keep shorter original seeds (c3t1/c4/c5 valuedgr 0/1).
  if len(finished) >= 3:
    return [d for d in dirs if lasts[d] is not None and lasts[d] >= 100_000_000]
  return [d for d in dirs if lasts[d] is not None]


def _nf_train_series(cfg_dir: str) -> list[list[tuple[int, float]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir):
    path = os.path.join(run_dir, "logs", "learner", "logs.csv")
    if not os.path.isfile(path):
      continue
    pts = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
      for row in csv.DictReader(fh):
        x = base._coerce(row.get(base.TRAIN_X_COL, ""))
        y = base._coerce(row.get(base.TRAIN_METRIC, ""))
        if x is not None and y is not None:
          pts.append((int(x), float(y)))
    if pts:
      series.append(pts)
  return series


def _seed_tag(run_dir: str) -> str:
  name = os.path.basename(run_dir.rstrip("/"))
  tail = name.rsplit("_", 1)[-1]
  return f"seed {tail}" if tail.isdigit() else name


def _nf_eval_labeled(cfg_dir: str) -> list[tuple[str, list[tuple[int, float]]]]:
  series = []
  for run_dir in _seed_dirs(cfg_dir):
    t_path = os.path.join(run_dir, "logs", "learner", "logs.csv")
    e_path = os.path.join(run_dir, "logs", "eval", "logs.csv")
    it_map = {}
    if os.path.isfile(t_path):
      with open(t_path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
          it = base._coerce(row.get("iteration", ""))
          gs = base._coerce(row.get(base.TRAIN_X_COL, ""))
          if it is not None and gs is not None:
            it_map[int(it)] = int(gs)
    spi = base._infer_steps_per_iter(it_map) if it_map else SPI_FALLBACK
    if not os.path.isfile(e_path):
      continue
    pts = []
    with open(e_path, newline="", encoding="utf-8", errors="replace") as fh:
      for row in csv.DictReader(fh):
        it = base._coerce(row.get(base.EVAL_X_COL, ""))
        y = base._coerce(row.get(base.EVAL_METRIC, ""))
        if it is None or y is None:
          continue
        it = int(it)
        x = it_map[it] if it in it_map else int((it + 1) * spi)
        pts.append((x, float(y)))
    if pts:
      series.append((_seed_tag(run_dir), pts))
  return series


def _nf_eval_series(cfg_dir: str) -> list[list[tuple[int, float]]]:
  return [pts for _, pts in _nf_eval_labeled(cfg_dir)]


def _rnd_csv_dir(task_key: str) -> str:
  return os.path.join(RND_ROOT, f"rnd_bb_{task_key}", "eval_success")


def _rnd_csv_paths(task_key: str) -> list[tuple[str, str]]:
  d = _rnd_csv_dir(task_key)
  if not os.path.isdir(d):
    return []
  out = []
  for name in sorted(os.listdir(d)):
    if not name.startswith("seed") or not name.endswith(".csv"):
      continue
    seed = name[len("seed"):-len(".csv")]
    out.append((seed, os.path.join(d, name)))
  return out


def _read_rnd_csv(path: str) -> list[tuple[int, float]]:
  pts = []
  with open(path, newline="", encoding="utf-8", errors="replace") as fh:
    for row in csv.DictReader(fh):
      x = base._coerce(row.get("env_steps", ""))
      y = base._coerce(row.get("eval_success", ""))
      if x is None or y is None:
        continue
      pts.append((int(x), float(min(1.0, max(0.0, y)))))
  pts.sort()
  return pts


def _write_rnd_zero_csv(task_key: str, seed: str) -> str:
  d = _rnd_csv_dir(task_key)
  os.makedirs(d, exist_ok=True)
  path = os.path.join(d, f"seed{seed}.csv")
  total = RND_TIMESTEPS[task_key]
  step = total // RND_NUM_EVAL_STEPS
  with open(path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=("eval_step", "env_steps", "eval_success"))
    w.writeheader()
    for es in range(1, RND_NUM_EVAL_STEPS + 1):
      w.writerow({
          "eval_step": es,
          "env_steps": es * step,
          "eval_success": 0.0,
      })
  return path


def _ensure_rnd_zero_csvs(task_key: str) -> None:
  if task_key not in RND_ZERO_TASKS:
    return
  if _rnd_csv_paths(task_key):
    return
  for seed in ("0", "1"):
    path = _write_rnd_zero_csv(task_key, seed)
    print(f"  wrote zero RND csv {path}")


def _rnd_labeled(task_key: str) -> list[tuple[str, list[tuple[int, float]]]]:
  _ensure_rnd_zero_csvs(task_key)
  out = []
  for seed, path in _rnd_csv_paths(task_key):
    pts = _read_rnd_csv(path)
    if pts:
      out.append((seed, pts))
  return out


def _rnd_series(task_key: str) -> tuple[
    list[list[tuple[int, float]]], list[list[tuple[int, float]]]]:
  # Training stdout was not kept. Plot checkpoint-eval hard success on both
  # panels so RND is visible on the paper grids.
  ev = [pts for _, pts in _rnd_labeled(task_key)]
  return ev, ev


def _sgcrl_csv_dir(task_key: str) -> str:
  return os.path.join(SGCRL_ROOT, task_key, "eval_success")


def _sgcrl_csv_paths(task_key: str) -> list[tuple[str, str]]:
  d = _sgcrl_csv_dir(task_key)
  if not os.path.isdir(d):
    return []
  out = []
  for name in sorted(os.listdir(d)):
    if not name.startswith("seed") or not name.endswith(".csv"):
      continue
    seed = name[len("seed"):-len(".csv")]
    out.append((seed, os.path.join(d, name)))
  return out


def _write_sgcrl_zero_csv(task_key: str, seed: str) -> str:
  d = _sgcrl_csv_dir(task_key)
  os.makedirs(d, exist_ok=True)
  path = os.path.join(d, f"seed{seed}.csv")
  total = RND_TIMESTEPS[task_key]
  step = total // RND_NUM_EVAL_STEPS
  with open(path, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=("eval_step", "env_steps", "eval_success"))
    w.writeheader()
    for es in range(1, RND_NUM_EVAL_STEPS + 1):
      w.writerow({
          "eval_step": es,
          "env_steps": es * step,
          "eval_success": 0.0,
      })
  return path


def _ensure_sgcrl_zero_csvs(task_key: str) -> None:
  if task_key not in SGCRL_ZERO_TASKS:
    return
  if _sgcrl_csv_paths(task_key):
    return
  for seed in ("1", "2", "3", "4"):
    path = _write_sgcrl_zero_csv(task_key, seed)
    print(f"  wrote zero SGCRL csv {path}")


def _sgcrl_labeled(task_key: str) -> list[tuple[str, list[tuple[int, float]]]]:
  _ensure_sgcrl_zero_csvs(task_key)
  out = []
  for seed, path in _sgcrl_csv_paths(task_key):
    pts = _read_rnd_csv(path)
    if pts:
      out.append((seed, pts))
  return out


def _sgcrl_series(task_key: str) -> tuple[
    list[list[tuple[int, float]]], list[list[tuple[int, float]]]]:
  # Vanilla SGCRL logs only eval/episode_success_rate. Plot it on both panels.
  ev = [pts for _, pts in _sgcrl_labeled(task_key)]
  return ev, ev


def _sgcrl_eval_series(task_key: str) -> list[list[tuple[int, float]]]:
  return [pts for _, pts in _sgcrl_labeled(task_key)]


def _discover_series(task_key: str) -> tuple[
    list[list[tuple[int, float]]], list[list[tuple[int, float]]]]:
  """Load DISCOVER BB train / eval from flat logs.csv (catwp recipe)."""
  cfg = DISCOVER_CFG.get(task_key)
  if not cfg:
    return [], []
  cfg_dir = os.path.join(DISCOVER_ROOT, cfg)
  if not os.path.isdir(cfg_dir):
    return [], []
  train, eval_ = [], []
  for name in sorted(os.listdir(cfg_dir)):
    run_dir = os.path.join(cfg_dir, name)
    path = os.path.join(run_dir, "logs.csv")
    if not os.path.isdir(run_dir) or not os.path.isfile(path):
      continue
    t_pts, e_pts = [], []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
      for row in csv.DictReader(fh):
        x = base._coerce(row.get("global_step", ""))
        if x is None:
          continue
        x = int(x)
        # Subsample BB eval cadence (epoch % 10 == 0) when epoch present.
        ep = base._coerce(row.get("epoch", ""))
        keep_eval = ep is None or int(ep) == 1 or int(ep) % 10 == 0
        ty = base._coerce(row.get("train_success_1000", ""))
        if ty is not None:
          t_pts.append((x, float(ty)))
        if keep_eval:
          # Prefer NF-comparable any-microstep; fall back to success.
          ey = base._coerce(row.get("eval_success_any_microstep", ""))
          if ey is None:
            ey = base._coerce(row.get("success", ""))
          if ey is not None:
            e_pts.append((x, float(ey)))
    if t_pts and t_pts[-1][0] >= DISCOVER_MIN_STEPS:
      train.append(t_pts)
    if e_pts and e_pts[-1][0] >= DISCOVER_MIN_STEPS:
      eval_.append(e_pts)
  return train, eval_


def _task_data(task: dict) -> dict:
  nf_root = task.get("nf_root", NF_ROOT)
  nf_dir = os.path.join(nf_root, task["nf"])
  nf_train = _nf_train_series(nf_dir)
  nf_eval = _nf_eval_series(nf_dir)
  rnd_train, rnd_eval = _rnd_series(task["key"])
  mpo_stem = task.get("mpo")
  mpo_train, mpo_eval = [], []
  if mpo_stem:
    mpo_dir = os.path.join(MPO_ROOT, mpo_stem)
    mpo_train = _nf_train_series(mpo_dir)
    mpo_eval = _nf_eval_series(mpo_dir)
  disc_train, disc_eval = _discover_series(task["key"])
  # Paper DISCOVER eval: early-stop-at-best counterfactual (running max).
  disc_eval = _carry_forward_series(disc_eval)
  sgcrl_train, sgcrl_eval = _sgcrl_series(task["key"])
  return {
      "nf_train": nf_train,
      "nf_eval": nf_eval,
      "sgcrl_train": sgcrl_train,
      "sgcrl_eval": sgcrl_eval,
      "rnd_train": rnd_train,
      "rnd_eval": rnd_eval,
      "mpo_train": mpo_train,
      "mpo_eval": mpo_eval,
      "discover_train": disc_train,
      "discover_eval": disc_eval,
  }


def _prepend_origin(xs, mean, se=None):
  """Add a (0, 0) point so eval curves start at the origin."""
  if not xs or xs[0] <= 0:
    return xs, mean, se
  xs = [0] + list(xs)
  mean = [0.0] + list(mean)
  if se is not None:
    se = [0.0] + list(se)
  return xs, mean, se


def _eval_window(task_key: str | None = None) -> int:
  if task_key == "c8t2":
    return 5
  return EVAL_SMOOTH_WINDOW


def _shade(ax, xs, mean, se, *, color, n: int, eval_smooth: bool,
           window: int | None = None) -> None:
  if n <= 1 or not se:
    return
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  if eval_smooth:
    w = EVAL_SMOOTH_WINDOW if window is None else window
    lo = base._rolling_mean(lo, w)
    hi = base._rolling_mean(hi, w)
    if xs and xs[0] == 0:
      lo[0] = 0.0
      hi[0] = 0.0
  ax.fill_between(xs, lo, hi, color=color, alpha=0.22, lw=0, zorder=2)


def _draw_train(ax, series, *, color, label) -> int:
  xs, mean, se, n = base._aggregate_mean_stderr(series)
  if not xs:
    return 0
  xs, mean, se = base._subsample_curve(xs, mean, se)
  _shade(ax, xs, mean, se, color=color, n=n, eval_smooth=False)
  ax.plot(xs, mean, color=color, linewidth=ps.LW, label=label, zorder=3,
          solid_capstyle="round")
  return n


def _draw_eval(ax, series, *, color, label, window: int | None = None) -> int:
  w = EVAL_SMOOTH_WINDOW if window is None else window
  xs, mean, se, n = base._aggregate_mean_stderr(series)
  if not xs:
    return 0
  xs, mean, se = base._subsample_curve(xs, mean, se)
  xs, mean, se = _prepend_origin(xs, mean, se)
  _shade(ax, xs, mean, se, color=color, n=n, eval_smooth=True, window=w)
  sm = base._rolling_mean(mean, w)
  if xs and xs[0] == 0:
    sm[0] = 0.0
  ax.plot(
      xs, mean, color=color, linewidth=1.0, alpha=0.28, linestyle="-",
      marker="o", markersize=3.2, markeredgewidth=0.0, zorder=3)
  ax.plot(
      xs, sm, color=color, linewidth=ps.LW, linestyle="-", label=label,
      alpha=0.95, zorder=4)
  return n


def _finish_ax(ax, *, title: str, ylabel: str, xlabel: bool) -> None:
  ax.set_title(title)
  ax.set_ylabel(ylabel)
  if xlabel:
    ax.set_xlabel("Environment steps")
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  ps.style_axes(ax, which="major")


def _legend_handles():
  return [
      Line2D([0], [0], color=m["color"], lw=ps.LW, label=m["label"])
      for m in METHODS
  ]


def _summarize(task, data) -> None:
  print(f"== {task['key']}  {task['title']} ==")
  for split in ("train", "eval"):
    for method in METHODS:
      series = data[f"{method['key']}_{split}"]
      xs, mean, se, n = base._aggregate_mean_stderr(series)
      if not xs:
        print(f"  {method['label']:8s} {split:5s}: no data")
        continue
      ends = [pts[-1][0] for pts in series if pts]
      lo = min(ends) / 1e6
      hi = max(ends) / 1e6
      print(
          f"  {method['label']:8s} {split:5s}: n={n} last={mean[-1]:.3f} "
          f"peak={max(mean):.3f} @ {xs[-1]/1e6:.1f}M  "
          f"(seeds {lo:.1f}–{hi:.1f}M)"
      )


def plot_task(task: dict, data: dict) -> None:
  w = _eval_window(task["key"])
  fig, (ax_t, ax_e) = ps.figure("sidebyside")
  fig.set_constrained_layout_pads(w_pad=0.12, h_pad=0.28, hspace=0.08, wspace=0.10)
  n_train = {}
  n_eval = {}
  for method in METHODS:
    n_train[method["key"]] = _draw_train(
        ax_t, data[f"{method['key']}_train"],
        color=method["color"], label=method["label"])
    n_eval[method["key"]] = _draw_eval(
        ax_e, data[f"{method['key']}_eval"],
        color=method["color"], label=method["label"], window=w)
  _finish_ax(
      ax_t, title=f"{task['title']}  ·  train",
      ylabel="Train success", xlabel=True)
  _finish_ax(
      ax_e, title=f"{task['title']}  ·  eval (roll. mean $w$={w})",
      ylabel="Eval success", xlabel=True)
  fig.legend(
      handles=_legend_handles(), loc="outside upper center", ncol=len(METHODS),
      frameon=True, fancybox=False, framealpha=0.96, edgecolor="#B0B0B0",
      handlelength=2.2, fontsize=ps.FS_LEGEND)
  print(
      "  seeds train "
      + " ".join(f"{m['label']}={n_train[m['key']]}" for m in METHODS)
      + "  eval "
      + " ".join(f"{m['label']}={n_eval[m['key']]}" for m in METHODS)
  )
  stem = os.path.join(OUT_DIR, f"bb_nf_rnd_{task['key']}_train_eval")
  os.makedirs(OUT_DIR, exist_ok=True)
  ps.savefig(fig, stem)
  plt.close(fig)


def plot_grid(split: str, ylabel: str, stem_name: str, *, eval_split: bool) -> None:
  nrows, ncols = 2, 4
  fig, axes = plt.subplots(
      nrows, ncols, figsize=(28.8, 11.6), layout="constrained", sharey=True)
  fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.18, hspace=0.08, wspace=0.04)
  axes = axes.ravel()
  draw = _draw_eval if eval_split else _draw_train
  for i, task in enumerate(TASKS):
    ax = axes[i]
    data = task["_data"]
    w = _eval_window(task["key"])
    if eval_split:
      for method in METHODS:
        draw(ax, data[f"{method['key']}_{split}"], color=method["color"],
             label=method["label"], window=w)
      title = task["title"] if w == EVAL_SMOOTH_WINDOW else (
          f"{task['title']}  ($w$={w})")
    else:
      for method in METHODS:
        draw(ax, data[f"{method['key']}_{split}"], color=method["color"],
             label=method["label"])
      title = task["title"]
    xlabel = i >= ncols
    ylabel_i = ylabel if i % ncols == 0 else ""
    _finish_ax(ax, title=title, ylabel=ylabel_i, xlabel=xlabel)
  for j in range(len(TASKS), nrows * ncols):
    axes[j].set_visible(False)
  fig.legend(
      handles=_legend_handles(), loc="outside upper center", ncol=len(METHODS),
      frameon=True, fancybox=False, framealpha=0.96, edgecolor="#B0B0B0",
      handlelength=2.2, fontsize=ps.FS_LEGEND)
  stem = os.path.join(OUT_DIR, stem_name)
  os.makedirs(OUT_DIR, exist_ok=True)
  ps.savefig(fig, stem)
  plt.close(fig)


def main() -> None:
  for task in TASKS:
    data = _task_data(task)
    task["_data"] = data
    _summarize(task, data)
    plot_task(task, data)
  plot_grid(
      "eval",
      "Eval success",
      "bb_nf_rnd_eval_success",
      eval_split=True)
  plot_grid(
      "train",
      "Train success",
      "bb_nf_rnd_train_success",
      eval_split=False)


if __name__ == "__main__":
  main()
