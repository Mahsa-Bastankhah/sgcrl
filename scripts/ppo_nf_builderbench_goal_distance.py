#!/usr/bin/env python3
"""Stochastic Hungarian goal-distance for PPO+NF checkpoints.

Default: final-run c7t2 seed 4 (success at ~184M). 5 trajs per ckpt.
Marks first nonzero eval and first eval success >= 0.1.

c3t1 overlay (dense probe + long-run tail vs RND):

  python scripts/ppo_nf_builderbench_goal_distance.py \\
    --run_dir=.../ppo_builderbench_creative_3_task1_0 --seed=0 \\
    --overlay_rnd_csv=logs/final_baselines/rnd_bb_c3t1/goal_distance/seed1.csv \\
    --overlay_tail_csv=.../seed2.csv

  python scripts/ppo_nf_builderbench_goal_distance.py --plot_only
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

NF_C7 = (
    "ppo_builderbench_creative7_task2_e1024_pd_nf_compact_small_sa3x192_"
    "r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_"
    "ent05to001_ep70_300m_crl10_dualgradreg_c100_lamlr1e6_"
    "valuedgr_c100_lamlr1e6_warp_4h"
)
SPI_FALLBACK = 1024 * 70
CSV_FIELDS = (
    "iteration",
    "env_steps",
    "mean_term_dist",
    "se_term_dist",
    "traj0",
    "traj1",
    "traj2",
    "traj3",
    "traj4",
)


def _load_bb_video():
  path = _REPO / "scripts" / "ppo_builderbench_rollout_video.py"
  spec = importlib.util.spec_from_file_location("bb_video_gdist", path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _default_run_dir(seed: int) -> Path:
  return (
      _REPO / "logs" / "final_runs" / NF_C7
      / f"ppo_builderbench_creative_7_task2_{seed}"
  )


def _dist_csv(run_dir: Path, seed: int) -> Path:
  return run_dir / "goal_distance" / f"seed{seed}.csv"


def _plot_path() -> Path:
  return (
      _REPO / "paper material" / "paper plots"
      / "bb_nf_c7t2_seed4_goal_dist"
  )


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
      x = it_map.get(it, (it + 1) * SPI_FALLBACK)
      rows.append((it, x, y))
  rows.sort()
  return rows


def _success_markers(rows, threshold: float):
  first_nz = next(((it, x, y) for it, x, y in rows if y > 0.0), None)
  first_ok = next(((it, x, y) for it, x, y in rows if y >= threshold), None)
  if first_ok is None:
    raise SystemExit(f"no eval success >= {threshold}")
  return first_nz, first_ok


def _read_dist_csv(path: Path) -> list[dict]:
  if not path.is_file():
    return []
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      it_key = "iteration" if "iteration" in row and row["iteration"] else "eval_step"
      rows.append({
          "iteration": int(row[it_key]),
          "env_steps": int(float(row["env_steps"])),
          "mean_term_dist": float(row["mean_term_dist"]),
          "se_term_dist": float(row["se_term_dist"]),
          "trajs": [float(row[f"traj{i}"]) for i in range(5)],
      })
  rows.sort(key=lambda r: r["iteration"])
  return rows


def _write_dist_csv(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    w.writeheader()
    for row in rows:
      out = {
          "iteration": row["iteration"],
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


def _early_stretch_fns(split: float, xmax: float, frac: float = 0.11):
  """Piecewise-linear x: [0, split] occupies `frac` of the axis (not split/xmax).

  Used to enlarge 0 → NF first success; after that the scale is linear.
  """
  import numpy as np
  split = float(split)
  xmax = float(max(xmax, split + 1.0))
  span = xmax - split

  def _fwd(x):
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    y = np.empty_like(x)
    early = x <= split
    y[early] = (x[early] / split) * frac
    y[~early] = frac + (x[~early] - split) / span * (1.0 - frac)
    return y[0] if scalar else y

  def _inv(y):
    y = np.asarray(y, dtype=float)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    x = np.empty_like(y)
    early = y <= frac
    x[early] = (y[early] / frac) * split
    x[~early] = split + (y[~early] - frac) / (1.0 - frac) * span
    return x[0] if scalar else x

  return _fwd, _inv


def _merge_tail(probe: list[dict], tail: list[dict]) -> list[dict]:
  if not tail:
    return list(probe)
  if not probe:
    return list(tail)
  max_x = max(int(r["env_steps"]) for r in probe)
  extra = [r for r in tail if int(r["env_steps"]) > max_x]
  return list(probe) + extra


def _read_rnd_success_csv(path: Path, threshold: float):
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      es = int(row["eval_step"])
      x = int(float(row["env_steps"]))
      y = float(row["eval_success"])
      rows.append((es, x, y))
  rows.sort()
  first_nz = next(((es, x, y) for es, x, y in rows if y > 0.0), None)
  first_ok = next(((es, x, y) for es, x, y in rows if y >= threshold), None)
  if first_ok is None:
    raise SystemExit(f"no eval_success >= {threshold} in {path}")
  return first_nz, first_ok


def _draw_series(ax, rows, color, z_line=3, jump_at: int | None = None):
  import paper_style as ps
  if not rows:
    return [], []

  def _line(xs, mean, se):
    if not xs:
      return
    lo = [m - s for m, s in zip(mean, se)]
    hi = [m + s for m, s in zip(mean, se)]
    ax.fill_between(xs, lo, hi, color=color, alpha=0.22, lw=0, zorder=z_line - 1)
    ax.plot(
        xs, mean, color=color, linewidth=ps.LW,
        solid_capstyle="butt", solid_joinstyle="miter", zorder=z_line)

  if jump_at is None:
    _line(
        [r["env_steps"] for r in rows],
        [r["mean_term_dist"] for r in rows],
        [r["se_term_dist"] for r in rows])
  else:
    # One polyline: hold the last pre-success y until jump_at, then drop.
    # Do not diagonally blend across the success checkpoint.
    jx = int(jump_at)
    before = [r for r in rows if int(r["env_steps"]) < jx]
    at = next((r for r in rows if int(r["env_steps"]) == jx), None)
    after = [r for r in rows if int(r["env_steps"]) > jx]
    xs, mean, se = [], [], []
    for r in before:
      xs.append(int(r["env_steps"]))
      mean.append(float(r["mean_term_dist"]))
      se.append(float(r["se_term_dist"]))
    if mean:
      y0, se0 = mean[-1], se[-1]
      if at is not None:
        y1, se1 = float(at["mean_term_dist"]), float(at["se_term_dist"])
      elif after:
        y1 = float(after[0]["mean_term_dist"])
        se1 = float(after[0]["se_term_dist"])
      else:
        y1, se1 = y0, se0
      xs.extend([jx, jx + 1])
      mean.extend([y0, y1])
      se.extend([se0, se1])
    for r in after:
      xs.append(int(r["env_steps"]))
      mean.append(float(r["mean_term_dist"]))
      se.append(float(r["se_term_dist"]))
    _line(xs, mean, se)
  for r in rows:
    ax.plot(
        [r["env_steps"]] * 5, r["trajs"], linestyle="None", marker="o",
        markersize=4.0, markeredgewidth=0.0, color=color, alpha=0.22,
        zorder=z_line)
  return [r["env_steps"] for r in rows], [r["mean_term_dist"] for r in rows]


def _hold_from_zero(rows: list[dict]) -> list[dict]:
  if not rows or int(rows[0]["env_steps"]) <= 0:
    return list(rows)
  head = dict(rows[0])
  head["env_steps"] = 0
  return [head] + list(rows)


def _mean_at(rows: list[dict], env_steps: int) -> float | None:
  for r in rows:
    if int(r["env_steps"]) == int(env_steps):
      return float(r["mean_term_dist"])
  # nearest
  if not rows:
    return None
  best = min(rows, key=lambda r: abs(int(r["env_steps"]) - int(env_steps)))
  if abs(int(best["env_steps"]) - int(env_steps)) > 2_000_000:
    return None
  return float(best["mean_term_dist"])


DASH_LW = 2.2
# Overlay-only (this figure is a paper exception; do not change paper_style).
FS_STILL = 32          # Goal / PIM / RND panel labels
FS_OVERLAY_LEGEND = 26
FS_OVERLAY_LABEL = 32
FS_OVERLAY_TICK = 32
STILL_DIR = _REPO / "videos" / "builderbench" / "c3t1_goal_dist_stills"
NF_STILL_VIDEO = STILL_DIR / "nf_c3t1_presuccess_iter_0000110.mp4"
RND_STILL_VIDEO = STILL_DIR / "params_39_stoch_ep0_seed0.mp4"
# Probe ckpt iter 110; RND still is pre-success; arrow lands at 120M.
NF_STILL_STEPS = 5_683_200
RND_STILL_STEPS = 120_000_000
STILL_CACHE = _REPO / "figs" / "builderbench" / "c3t1_goal_dist_stills"
FINAL_PLOT_DIR = _REPO / "paper material" / "paper plots" / "final final plots"


def _ffmpeg_bin() -> str:
  return "/usr/bin/ffmpeg"


def _extract_last_frame(video: Path, out_png: Path) -> None:
  import subprocess
  out_png.parent.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env.pop("LD_LIBRARY_PATH", None)
  env.pop("LD_PRELOAD", None)
  subprocess.check_call(
      [
          _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
          "-sseof", "-0.1", "-i", str(video), "-frames:v", "1", str(out_png),
      ],
      env=env,
  )


def _render_c3t1_goal(out_png: Path) -> None:
  """Solid cubes at the c3t1 target; mocap parked (same as c7t2 Goal panel)."""
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("BUILDERBENCH_MJX_IMPL", "jax")
  import numpy as np
  from PIL import Image
  from builderbench.creative_cube import CreativeCube, default_config
  from envs.builderbench_utils import (
      default_fixed_target_goal, parse_bb_env_id, sgcrl_env_name_to_bb_env_id)

  env_id = sgcrl_env_name_to_bb_env_id("builderbench_creative_3_task1")
  num_cubes, task_id = parse_bb_env_id(env_id)
  cfg = default_config()
  cfg.num_cubes = num_cubes
  cfg.task_id = task_id
  cfg.permute_start_boxes = False
  cfg.impl = os.environ.get("BUILDERBENCH_MJX_IMPL", "jax")
  env = CreativeCube(config=cfg)
  xyz = default_fixed_target_goal(num_cubes, task_id).reshape(-1, 3)
  # Cube 0 coral/pink, 2 mustard/yellow. Put yellow on the table, pink on top.
  xyz[[0, 2]] = xyz[[2, 0]]
  qpos = np.array(env._init_q, copy=True)
  qpos[np.asarray(env._objs_pos_qpos_idxs)] = xyz.astype(np.float32).reshape(-1)
  qpos[np.asarray(env._objs_quat_qpos_idxs)] = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), int(num_cubes))
  n_task = int(env._num_task_cubes)
  mocap_pos = np.tile(np.array([10.0, 10.0, 10.0], dtype=np.float32), (n_task, 1))
  mocap_quat = np.tile(
      np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_task, 1))
  qvel = np.zeros(int(env._mj_model.nv), dtype=np.float32)
  img = np.asarray(env.render_from_info(
      qpos, qvel, mocap_pos, mocap_quat,
      height=480, width=640, camera=-1,
  ))
  out_png.parent.mkdir(parents=True, exist_ok=True)
  Image.fromarray(img).save(out_png)


def _zoom_goal(img):
  from PIL import Image
  import numpy as np
  h, w = img.shape[:2]
  m = max(4, int(0.06 * min(h, w)))
  cropped = img[m:h - m, m:w - m]
  return np.asarray(
      Image.fromarray(cropped).resize((w, h), Image.Resampling.LANCZOS))


def _load_goal_still():
  from PIL import Image
  import numpy as np
  png = STILL_CACHE / "c3t1_goal_yellow_bottom.png"
  if not png.is_file():
    _render_c3t1_goal(png)
  img = np.asarray(Image.open(png).convert("RGB"))
  return _zoom_goal(_crop_cubes(img))


def _crop_cubes(img):
  import numpy as np
  h, w = img.shape[:2]
  arr = img.astype(np.float32)
  chroma = arr.max(axis=2) - arr.min(axis=2)
  bright = arr.max(axis=2)
  # Skip the sky / success banner; keep bright saturated cubes on the table.
  y0 = int(0.32 * h)
  cube = (chroma[y0:] > 78.0) & (bright[y0:] > 110.0)
  ys, xs = np.where(cube)
  if len(xs) < 15:
    return img[int(0.48 * h):int(0.82 * h), int(0.38 * w):int(0.62 * w)]
  pad = 10
  xa = max(0, int(xs.min()) - pad)
  xb = min(w, int(xs.max()) + pad)
  ya = max(0, int(ys.min() + y0) - pad)
  yb = min(h, int(ys.max() + y0) + pad)
  side = max(xb - xa, yb - ya, 96)
  cx, cy = (xa + xb) // 2, (ya + yb) // 2
  xa = max(0, min(w - side, cx - side // 2))
  ya = max(0, min(h - side, cy - side // 2))
  xb, yb = min(w, xa + side), min(h, ya + side)
  return img[ya:yb, xa:xb]


def _load_still(video: Path, cache_name: str):
  from PIL import Image
  import numpy as np
  if not video.is_file():
    return None
  png = STILL_CACHE / cache_name
  if not png.is_file() or png.stat().st_mtime < video.stat().st_mtime:
    _extract_last_frame(video, png)
  img = np.asarray(Image.open(png).convert("RGB"))
  return _crop_cubes(img)


def _decorate_goal_panel(img):
  """Gold rounded border as in bb_c7t2_s3_policy_progression (label is outside)."""
  from PIL import Image, ImageDraw
  import numpy as np
  scale = 4
  panel = Image.fromarray(img).resize(
      (img.shape[1] * scale, img.shape[0] * scale), Image.Resampling.LANCZOS)
  w, h = panel.size
  draw = ImageDraw.Draw(panel)
  gold = (245, 176, 24)
  inset = max(8, int(round(10 * min(w, h) / 660)))
  radius = max(16, int(round(26 * min(w, h) / 660)))
  width = max(12, int(round(20 * min(w, h) / 660)))
  draw.rounded_rectangle(
      (inset, inset, w - inset - 1, h - inset - 1),
      radius=radius, outline=gold, width=width)
  return np.asarray(panel)


def _show_still(
    ax, img, *, title: str, color: str, hide_spines: bool = False,
    title_side: str = "left",
) -> None:
  ax.imshow(img)
  if title:
    ax.set_ylabel(
        title, color=color, fontsize=FS_STILL, rotation=0, labelpad=8)
    ax.yaxis.label.set_verticalalignment("center")
    if title_side == "right":
      ax.yaxis.set_label_position("right")
      ax.yaxis.label.set_horizontalalignment("left")
    else:
      ax.yaxis.label.set_horizontalalignment("right")
  ax.set_xticks([])
  ax.set_yticks([])
  for name in ("top", "right", "bottom", "left"):
    if hide_spines:
      ax.spines[name].set_visible(False)
    else:
      ax.spines[name].set_visible(True)
      ax.spines[name].set_color(color)
      ax.spines[name].set_linewidth(4.8)
  ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
  ax.set_aspect("equal")
  ax.set_adjustable("box")


def _fit_square_axes(
    ax, *, halign: str = "center", valign: str = "bottom", dx: float = 0.0,
) -> None:
  """Shrink a still axes to a square so the colored frame hugs the crop."""
  fig = ax.figure
  fig.canvas.draw()
  bbox = ax.get_position()
  fw, fh = fig.get_size_inches()
  w_in, h_in = bbox.width * fw, bbox.height * fh
  side = min(w_in, h_in)
  nw, nh = side / fw, side / fh
  if halign == "left":
    x0 = bbox.x0
  elif halign == "right":
    x0 = bbox.x1 - nw
  else:
    x0 = bbox.x0 + 0.5 * (bbox.width - nw)
  if valign == "top":
    y0 = bbox.y1 - nh
  elif valign == "center":
    y0 = bbox.y0 + 0.5 * (bbox.height - nh)
  else:
    y0 = bbox.y0
  ax.set_position([x0 + dx, y0, nw, nh])


def _link_still(
    fig, ax_img, ax, rows, steps, color, *, xyA=(0.5, 0.0),
) -> None:
  from matplotlib.patches import ConnectionPatch
  y = _mean_at(rows, steps)
  if y is None:
    return
  ax.plot(
      [steps], [y], marker="s", markersize=11,
      markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.6,
      zorder=7, clip_on=False)
  fig.add_artist(ConnectionPatch(
      xyA=xyA, coordsA=ax_img.transAxes,
      xyB=(steps, y), coordsB=ax.transData,
      color=color, lw=1.6, arrowstyle="-|>", mutation_scale=16,
      shrinkA=2, shrinkB=8, zorder=8, clip_on=False))


def draw_nf_rnd_overlay(
    ax,
    nf_rows: list[dict],
    rnd_rows: list[dict],
    *,
    nf_ok: tuple,
    rnd_ok: tuple,
    title: str,
    dash_lw: float = DASH_LW,
) -> None:
  import matplotlib.ticker as mticker
  from matplotlib.lines import Line2D
  import paper_style as ps

  nf_c = ps.C["blue"]
  rnd_c = ps.C["vermillion"]
  nf_ok = (nf_ok[0], 8_000_000, nf_ok[2])
  rnd_rows = _hold_from_zero(rnd_rows)

  # One polyline of measured means. RND holds until first success, then
  # drops — no diagonal blend across that checkpoint.
  _draw_series(ax, nf_rows, nf_c, z_line=4)
  _draw_series(ax, rnd_rows, rnd_c, z_line=3, jump_at=int(rnd_ok[1]))

  def _mark(ok, color, rows):
    ok_x = int(ok[1])
    ax.axvline(ok_x, color=color, linestyle=ps.DASH, linewidth=dash_lw, zorder=1)
    ok_y = _mean_at(rows, ok_x)
    if ok_y is not None:
      ax.plot(
          [ok_x], [ok_y], marker="o", markersize=ps.MS,
          markerfacecolor="white", markeredgewidth=ps.MEW,
          markeredgecolor=color, zorder=6)

  _mark(nf_ok, nf_c, nf_rows)
  _mark(rnd_ok, rnd_c, rnd_rows)

  x_full = 175_000_000
  ax.set_xscale(
      "function", functions=_early_stretch_fns(int(nf_ok[1]), x_full))
  ax.set_xlim(0, x_full)
  ax.set_ylim(0.0, 1.05)
  ax.set_ylabel("Distance to goal", fontsize=FS_OVERLAY_LABEL)
  ax.set_xlabel("Environment steps", fontsize=FS_OVERLAY_LABEL)
  if title:
    ax.set_title(title, fontsize=ps.FS_TITLE, pad=ps.TITLE_PAD)
  ps.style_axes(ax, which="major")
  ax.tick_params(axis="both", which="major", labelsize=FS_OVERLAY_TICK)
  ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
  ax.xaxis.set_minor_locator(mticker.NullLocator())
  ax.set_xticks([100_000_000])
  ax.set_xticklabels(["100M"])

  handles = [
      Line2D([0], [0], color=nf_c, lw=ps.LW, label="PIM (ours)"),
      Line2D([0], [0], color=rnd_c, lw=ps.LW, label="RND"),
      Line2D(
          [0], [0], color=ps.C["gray"], linestyle=ps.DASH, linewidth=dash_lw,
          label="first success"),
  ]
  ps.nice_legend(
      ax, handles=handles, loc="upper left", overlay=True,
      bbox_to_anchor=(0.16, 1.0), fontsize=FS_OVERLAY_LEGEND)


def plot_nf_rnd_overlay(
    nf_rows: list[dict],
    rnd_rows: list[dict],
    *,
    nf_ok: tuple,
    rnd_ok: tuple,
    out_path: Path,
    title: str,
) -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import paper_style as ps

  ps.apply()
  nf_img = _load_still(NF_STILL_VIDEO, "nf_iter110_last.png")
  rnd_img = _load_still(RND_STILL_VIDEO, "rnd_params39_last.png")
  has_stills = nf_img is not None and rnd_img is not None
  if nf_img is None:
    print(f"[nf_dist] missing NF still {NF_STILL_VIDEO}")
  if rnd_img is None:
    print(f"[nf_dist] missing RND still {RND_STILL_VIDEO}")

  if has_stills:
    # Still row height ≈ column width so the squares fill the strip
    # (no nested 3x3 padding). Plot keeps most of the figure.
    fig = plt.figure(figsize=(10.2, 6.90))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.38, 2.42],
        width_ratios=[1.0, 1.0, 1.0],
        left=0.128, right=0.972, top=0.992, bottom=0.138,
        hspace=0.045, wspace=0.05)
    ax_goal = fig.add_subplot(gs[0, 0])
    ax_nf = fig.add_subplot(gs[0, 1])
    ax_rnd = fig.add_subplot(gs[0, 2])
    ax = fig.add_subplot(gs[1, :])
    goal_img = _decorate_goal_panel(_load_goal_still())
    _show_still(
        ax_goal, goal_img, title="Goal", color=ps.C["black"], hide_spines=True)
    _show_still(ax_nf, nf_img, title="PIM", color=ps.C["blue"])
    _show_still(
        ax_rnd, rnd_img, title="RND", color=ps.C["vermillion"],
        title_side="right")
  else:
    fig, ax = plt.subplots(figsize=(10.2, 6.2), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=ps.LAYOUT_PADS["w_pad"], h_pad=ps.LAYOUT_PADS["h_pad"])
    ax_nf = ax_rnd = None

  draw_nf_rnd_overlay(
      ax, nf_rows, rnd_rows, nf_ok=nf_ok, rnd_ok=rnd_ok,
      title="" if has_stills else title, dash_lw=DASH_LW)
  if has_stills:
    _fit_square_axes(ax_goal, halign="right", valign="bottom", dx=-0.038)
    _fit_square_axes(ax_nf, halign="center", valign="bottom")
    _fit_square_axes(ax_rnd, halign="left", valign="bottom")
    _link_still(
        fig, ax_nf, ax, nf_rows, NF_STILL_STEPS, ps.C["blue"],
        xyA=(0.22, 0.0))
    _link_still(
        fig, ax_rnd, ax, rnd_rows, RND_STILL_STEPS, ps.C["vermillion"],
        xyA=(0.28, 0.0))

  out_path.parent.mkdir(parents=True, exist_ok=True)
  pdf = Path(ps.savefig(fig, out_path, pad_inches=0.06))
  final = FINAL_PLOT_DIR / pdf.name
  if final.resolve() != pdf.resolve():
    import shutil
    FINAL_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, final)
    print("Saved:", final)
  plt.close(fig)


def plot_distance(
    rows: list[dict],
    *,
    first_nz: tuple | None,
    first_ok: tuple,
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
  mean = [r["mean_term_dist"] for r in rows]
  se = [r["se_term_dist"] for r in rows]
  color = ps.C["blue"]
  lo = [m - s for m, s in zip(mean, se)]
  hi = [m + s for m, s in zip(mean, se)]
  ax.fill_between(xs, lo, hi, color=color, alpha=0.22, lw=0, zorder=2)
  ax.plot(xs, mean, color=color, linewidth=ps.LW, solid_capstyle="round", zorder=3)
  for r in rows:
    ax.plot(
        [r["env_steps"]] * 5, r["trajs"], linestyle="None", marker="o",
        markersize=4.0, markeredgewidth=0.0, color=color, alpha=0.22, zorder=3)

  handles = [
      Line2D([0], [0], color=color, lw=ps.LW, label="NF (ours)"),
  ]
  ok_x = first_ok[1]
  ax.axvline(ok_x, color=color, linestyle=ps.DASH, linewidth=2.2, zorder=1)
  ok_y = _mean_at(rows, ok_x)
  if ok_y is not None:
    ax.plot(
        [ok_x], [ok_y], marker="o", markersize=ps.MS,
        markerfacecolor="white", markeredgewidth=ps.MEW,
        markeredgecolor=color, zorder=6)
  handles.append(Line2D(
      [0], [0], color=ps.C["gray"], linestyle=ps.DASH, linewidth=2.2,
      label="First success"))

  ax.set_xlabel("Environment steps")
  ax.set_ylabel("Hungarian distance to goal")
  ax.set_title(title)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  if xs:
    ax.set_xlim(0, xs[-1] * 1.06)
  ax.set_ylim(bottom=0)
  ps.style_axes(ax, which="major")
  ps.nice_legend(ax, handles=handles, loc="upper right")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  ps.savefig(fig, out_path)
  plt.close(fig)


def _parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--seed", type=int, default=4)
  p.add_argument("--min_iter", type=int, default=0)
  p.add_argument("--max_iter", type=int, default=10**9)
  p.add_argument("--num_traj", type=int, default=5)
  p.add_argument("--success_threshold", type=float, default=0.1)
  p.add_argument("--run_dir", default="")
  p.add_argument("--csv_output", default="")
  p.add_argument("--plot_output", default="")
  p.add_argument("--title", default="")
  p.add_argument(
      "--overlay_rnd_csv", default="",
      help="If set, overlay this RND distance CSV on the NF plot.")
  p.add_argument(
      "--overlay_rnd_success_csv", default="",
      help="RND eval_success CSV (eval_step,env_steps,eval_success).")
  p.add_argument(
      "--overlay_tail_csv", default="",
      help="Long-run NF distance CSV stitched after the probe's last step.")
  p.add_argument("--plot_only", action="store_true")
  return p.parse_args()


def main() -> None:
  args = _parse_args()
  if args.num_traj != 5:
    raise SystemExit("CSV columns are fixed at 5 trajs")
  run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir(args.seed)
  dist_path = (
      Path(args.csv_output) if args.csv_output else _dist_csv(run_dir, args.seed))
  plot_path = Path(args.plot_output) if args.plot_output else _plot_path()
  cfg_path = run_dir / "run_config.json"
  with cfg_path.open(encoding="utf-8") as fh:
    run_cfg = json.load(fh)
  env_name = str(run_cfg.get("env", "builderbench_creative_7_task2"))
  flags = run_cfg.get("flags", {})
  mj_ep_len = flags.get("builderbench_mj_episode_length")
  it_map = _iter_to_steps(run_dir)
  eval_rows = _eval_success_rows(run_dir, it_map)
  first_nz, first_ok = _success_markers(eval_rows, args.success_threshold)
  print(f"[nf_dist] run_dir={run_dir}")
  print(f"[nf_dist] env={env_name}")
  if first_nz is None:
    print("[nf_dist] first nonzero: none")
  else:
    print(
        f"[nf_dist] first nonzero: iter={first_nz[0]}  "
        f"steps={first_nz[1]/1e6:.1f}M  success={first_nz[2]:.4f}")
  print(
      f"[nf_dist] first >= {args.success_threshold}: iter={first_ok[0]}  "
      f"steps={first_ok[1]/1e6:.1f}M  success={first_ok[2]:.4f}")

  plot_title = args.title or (
      "PPO+NF c7t2 seed 4" if not args.run_dir else f"PPO+NF seed {args.seed}")
  overlay_rnd_path = Path(args.overlay_rnd_csv) if args.overlay_rnd_csv else None
  overlay_tail_path = (
      Path(args.overlay_tail_csv) if args.overlay_tail_csv else None)

  def _maybe_overlay(rows: list[dict]) -> None:
    if overlay_rnd_path is None:
      plot_distance(
          rows, first_nz=first_nz, first_ok=first_ok, out_path=plot_path,
          title=plot_title)
      return
    rnd_rows = _read_dist_csv(overlay_rnd_path)
    if not rnd_rows:
      raise SystemExit(f"no RND distance CSV at {overlay_rnd_path}")
    nf_rows = list(rows)
    if overlay_tail_path is not None:
      tail = _read_dist_csv(overlay_tail_path)
      nf_rows = _merge_tail(nf_rows, tail)
      stitched = dist_path.with_name(f"seed{args.seed}_stitched.csv")
      _write_dist_csv(stitched, nf_rows)
      print(
          f"[nf_dist] stitched {len(rows)} probe + "
          f"{len(nf_rows) - len(rows)} tail rows -> {stitched}")
    rnd_succ = (
        Path(args.overlay_rnd_success_csv) if args.overlay_rnd_success_csv
        else overlay_rnd_path.parent.parent / "eval_success" / overlay_rnd_path.name)
    _, rnd_ok = _read_rnd_success_csv(rnd_succ, args.success_threshold)
    ov_title = args.title or "Distance to goal during training"
    plot_nf_rnd_overlay(
        nf_rows, rnd_rows, nf_ok=first_ok, rnd_ok=rnd_ok,
        out_path=plot_path, title=ov_title)
    print(f"[nf_dist] overlay plot {plot_path}")

  if args.plot_only:
    rows = _read_dist_csv(dist_path)
    if not rows:
      raise SystemExit(f"no distance CSV at {dist_path}")
    _maybe_overlay(rows)
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
  ckpt_dir = run_dir / "checkpoints"
  ctx = bb_video._load_train_ctx(env_name, str(ckpt_dir))
  pending = []
  for path in sorted(ckpt_dir.glob("ckpt_iter_*.pkl")):
    it = int(path.stem.split("_")[-1])
    if args.min_iter <= it <= args.max_iter:
      pending.append(it)
  if not pending:
    raise SystemExit(f"no ckpt_iter_*.pkl in {ckpt_dir} for "
                     f"iters {args.min_iter}-{args.max_iter}")
  done = {r["iteration"]: r for r in _read_dist_csv(dist_path)}
  todo = [it for it in pending if it not in done]
  print(
      f"[nf_dist] iters {pending[0]}-{pending[-1]}  "
      f"{len(pending)} ckpts, {len(done)} done, {len(todo)} pending")
  print(f"[nf_dist] jax={jax.default_backend()} {jax.devices()}")

  if todo:
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
    n_task = int(np.asarray(fixed_goal).size // 3)
    n_traj = args.num_traj
    filter_obs = bool(ctx.filter_policy_obs)
    print(
        f"[nf_dist] MJX impl={cfg.impl} n_task={n_task} "
        f"macro_ep_len={episode_length} filter_obs={filter_obs}")

    def generate_unroll(policy_params, key):
      reset_key, unroll_key = jax.random.split(key)
      state = env.reset(jax.random.split(reset_key, n_traj))
      state = bb_video._maybe_fix_target(
          state, fixed_goal, mocap_targets, num_cubes)

      def body(_i, carry):
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
        return (next_state, key)

      final_state, _ = jax.lax.fori_loop(
          0, episode_length, body, (state, unroll_key))
      return final_state.metrics["obj_goal_dist"] / float(n_task)

    run_unroll = jax.jit(generate_unroll)
    key = jax.random.PRNGKey(args.seed + 40_011)

    for i, it in enumerate(todo):
      param_file = ckpt_dir / f"ckpt_iter_{it:07d}.pkl"
      t0 = time.time()
      ckpt = ppo_learner.load_checkpoint(str(param_file))
      env_steps = int(ckpt.get("global_step", it_map.get(it, it * SPI_FALLBACK)))
      key, unroll_key = jax.random.split(key)
      dists = np.asarray(run_unroll(ckpt["policy_params"], unroll_key))
      dists = np.reshape(dists, (n_traj,))
      mean = float(dists.mean())
      se = float(dists.std(ddof=1) / np.sqrt(n_traj)) if n_traj > 1 else 0.0
      done[it] = {
          "iteration": it,
          "env_steps": env_steps,
          "mean_term_dist": mean,
          "se_term_dist": se,
          "trajs": [float(x) for x in dists.tolist()],
      }
      _write_dist_csv(dist_path, [done[k] for k in sorted(done)])
      print(
          f"[nf_dist] {param_file.name}  step={env_steps/1e6:.1f}M  "
          f"dist={mean:.4f}±{se:.4f}  {time.time()-t0:.1f}s  "
          f"({i+1}/{len(todo)})",
          flush=True,
      )

  rows = [done[k] for k in sorted(done)]
  if not rows:
    raise SystemExit("no distance rows to plot")
  _maybe_overlay(rows)
  print(f"[nf_dist] wrote {dist_path}")


if __name__ == "__main__":
  main()
