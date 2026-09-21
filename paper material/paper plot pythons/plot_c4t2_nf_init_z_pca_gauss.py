#!/usr/bin/env python3
"""Paper-ready C4T2 NF init-z PCA on N(0,I) density surface.

Reuses NF/PCA helpers from ``scripts/plot_nf_init_z_pca3d.py``. Prefers a
cached scores CSV written by that script; otherwise probes checkpoints.

  CUDA_VISIBLE_DEVICES=0 BUILDERBENCH_MJX_IMPL=warp \\
  python "paper material/paper plot pythons/plot_c4t2_nf_init_z_pca_gauss.py"

Outputs:
  paper material/paper plots/bb_c4t2_nf_init_z_pca_gauss_density.{png,pdf}
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import paper_style as ps  # noqa: E402

ps.apply()

RUN_DIR = os.path.join(
    REPO,
    "logs",
    "final_runs",
    "visualizations",
    "ppo_builderbench_creative4_task2_e1024_pd_nf_compact_small_sa3x192_"
    "r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_"
    "ent05to001_ep50_20m_ckpt20_crl10_dualgradreg_c100_lamlr1e6_"
    "valuedgr_c100_lamlr1e6_warp_logp_s0",
    "ppo_builderbench_creative_4_task2_0",
)
OUT_STEM = os.path.join(
    PAPER_DIR, "paper plots", "bb_c4t2_nf_init_z_pca_gauss_density")
MAX_ITER = 280
STRIDE = 40


def _iso_gauss2d_density(x, y):
  x = np.asarray(x, dtype=np.float64)
  y = np.asarray(y, dtype=np.float64)
  return (1.0 / (2.0 * np.pi)) * np.exp(-0.5 * (x * x + y * y))


def _load_cache(path: str) -> dict:
  meta = {}
  rows = []
  with open(path, newline="") as f:
    for line in f:
      if line.startswith("#"):
        for part in line.lstrip("#").split():
          if "=" in part:
            k, v = part.split("=", 1)
            meta[k] = float(v)
        continue
      rows.append(line)
  reader = csv.DictReader(rows)
  iters, pc1, pc2 = [], [], []
  for r in reader:
    iters.append(int(float(r["iter"])))
    pc1.append(float(r["pc1"]))
    pc2.append(float(r["pc2"]))
  return {
      "iters": np.asarray(iters, dtype=np.int64),
      "xy": np.column_stack([pc1, pc2]).astype(np.float64),
      "var_ratio": np.asarray(
          [meta["var_ratio_pc1"], meta["var_ratio_pc2"]], dtype=np.float64),
      "success_mean_iter": int(meta["success_mean_iter"]),
      "success_mean_step": int(meta["success_mean_step"]),
      "success_1000_iter": int(meta["success_1000_iter"]),
  }


def _probe_src():
  src_path = os.path.join(REPO, "scripts", "plot_nf_init_z_pca3d.py")
  spec = importlib.util.spec_from_file_location("plot_nf_init_z_pca3d", src_path)
  src = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(src)
  return src


def _load_or_collect(run_dir: str, *, force_reprobe: bool = False) -> dict:
  cache = os.path.join(run_dir, f"init_z_pca2d_scores_to{MAX_ITER}.csv")
  if (not force_reprobe) and os.path.isfile(cache):
    print(f"loading cache {cache}", flush=True)
    return _load_cache(cache)

  print(f"cache missing or --reprobe; collecting from {run_dir}", flush=True)
  src = _probe_src()
  data = src.collect_init_z_pca(
      run_dir,
      stride=STRIDE,
      max_iter=MAX_ITER,
      include_final=True,
      n_components=2,
  )
  src.save_pca2d_scores_csv(
      cache,
      data["iters"],
      data["scores"][:, :2],
      data["var_ratio"],
      success_mean_iter=data["success_mean_iter"],
      success_mean_step=data["success_mean_step"],
      success_1000_iter=data["success_1000_iter"],
  )
  print(f"wrote {cache}", flush=True)
  return {
      "iters": data["iters"],
      "xy": np.asarray(data["scores"][:, :2], dtype=np.float64),
      "var_ratio": data["var_ratio"],
      "success_mean_iter": data["success_mean_iter"],
      "success_mean_step": data["success_mean_step"],
      "success_1000_iter": data["success_1000_iter"],
  }


def _draw_surface_arrows(ax, xy: np.ndarray, *, z_eps: float,
                         marker_r: float = 0.14, zorder: float = 8) -> None:
  """Centerline arrows on the density surface, tip at the next marker edge."""
  dens = _iso_gauss2d_density
  for i in range(xy.shape[0] - 1):
    p0 = np.asarray(xy[i], dtype=np.float64)
    p1 = np.asarray(xy[i + 1], dtype=np.float64)
    dxy = p1 - p0
    dist = float(np.linalg.norm(dxy))
    if dist < 0.18:
      continue
    u = dxy / dist
    pad = min(float(marker_r), 0.28 * dist)
    a = p0 + u * pad
    b = p1 - u * pad
    span = float(np.linalg.norm(b - a))
    if span < 0.08:
      continue
    t = np.linspace(0.0, 1.0, 24)
    sx = a[0] + (b[0] - a[0]) * t
    sy = a[1] + (b[1] - a[1]) * t
    sz = dens(sx, sy) + float(z_eps)
    for lw, color, zo in ((3.4, "white", zorder), (2.15, "0.05", zorder + 0.2)):
      ax.plot(sx, sy, sz, color=color, lw=lw, zorder=zo,
              solid_capstyle="butt")
    head_len = min(0.16, 0.28 * span)
    ang = float(np.arctan2(u[1], u[0]))
    z_tip = float(dens(b[0], b[1]) + z_eps)
    for da in (0.42, -0.42):
      hx = b[0] - head_len * np.cos(ang + da)
      hy = b[1] - head_len * np.sin(ang + da)
      for lw, color, zo in (
          (3.4, "white", zorder + 1), (2.15, "0.05", zorder + 1.2)):
        ax.plot(
            [b[0], hx], [b[1], hy],
            [z_tip, float(dens(hx, hy) + z_eps)],
            color=color, lw=lw, zorder=zo, solid_capstyle="round")


def _plot_paper(
    iters: np.ndarray,
    xy: np.ndarray,
    var_ratio: np.ndarray,
    *,
    success_mean_iter: int,
    out_stem: str,
    sigma_clip: float = 2.0,
    grid_pad: float = 0.12,
) -> int:
  dens_pts = _iso_gauss2d_density(xy[:, 0], xy[:, 1])
  z_eps = 0.003 * max(float(dens_pts.max()), 1e-3)
  xyz = np.column_stack([xy[:, 0], xy[:, 1], dens_pts + z_eps])

  lim = float(sigma_clip) + float(grid_pad)
  # Polar mesh stops at the orange 2σ ring so the unused tail is gone.
  rr = np.linspace(0.0, float(sigma_clip), 16)
  th = np.linspace(0.0, 2.0 * np.pi, 48)
  RR, TH = np.meshgrid(rr, th)
  XX = RR * np.cos(TH)
  YY = RR * np.sin(TH)
  ZZ = _iso_gauss2d_density(XX, YY)

  fig = plt.figure(figsize=(7.6, 6.2))
  ax = fig.add_subplot(111, projection="3d")
  if hasattr(ax, "computed_zorder"):
    ax.computed_zorder = False
  ax.view_init(elev=28, azim=-125)
  try:
    ax.set_box_aspect((1.0, 1.0, 0.55), zoom=1.48)
  except TypeError:
    try:
      ax.set_box_aspect((1.0, 1.0, 0.55))
    except Exception:
      pass
    ax.dist = 6.4

  ax.plot_surface(
      XX, YY, ZZ,
      color="#7EC8F5", alpha=0.38, linewidth=0, antialiased=True,
      shade=False, zorder=1)
  ax.plot_wireframe(
      XX, YY, ZZ,
      color="#4A9BC7", linewidth=0.45, rstride=1, cstride=2,
      alpha=0.55, zorder=2)

  theta = np.linspace(0.0, 2.0 * np.pi, 180)
  for r, color in ((1.0, ps.C["blue"]), (2.0, ps.C["orange"])):
    cx = r * np.cos(theta)
    cy = r * np.sin(theta)
    cz = _iso_gauss2d_density(cx, cy)
    ax.plot(cx, cy, cz, color=color, lw=2.1, alpha=0.95, zorder=3)

  _draw_surface_arrows(ax, xy, z_eps=z_eps)

  sc = ax.scatter(
      xyz[:, 0], xyz[:, 1], xyz[:, 2],
      c=iters, cmap="viridis", s=150, depthshade=False,
      edgecolors="k", linewidths=1.0, zorder=10)
  cbar = fig.colorbar(sc, ax=ax, pad=0.20, shrink=0.58, aspect=18)
  cbar.set_label("training iteration", fontsize=ps.FS_LABEL, labelpad=10)
  cbar.ax.tick_params(labelsize=max(ps.FS_TICK - 8, 16))

  success_color = "#C41E3A"
  mark_it = -1
  for i, it in enumerate(iters):
    if int(it) >= int(success_mean_iter):
      mark_it = int(it)
      ax.scatter(
          [xyz[i, 0]], [xyz[i, 1]], [xyz[i, 2]],
          s=440, facecolors="none", edgecolors=success_color,
          linewidths=2.6, depthshade=False, zorder=12)
      ax.text(
          xyz[i, 0] + 0.16, xyz[i, 1] + 0.46, xyz[i, 2] + 0.016,
          "first success",
          color=success_color, fontsize=26, fontweight="normal",
          ha="left", va="bottom", zorder=13)
      break

  ax.set_xlabel("")
  ax.set_ylabel("")
  ax.set_zlabel("")
  ax.set_xlim(-lim, lim)
  ax.set_ylim(-lim, lim)
  ax.set_zlim(0.0, float(np.nanmax(ZZ)) * 1.05)
  ax.set_xticks([-2, 0, 2])
  ax.set_yticks([-2, 0, 2])
  ax.set_zticks([])
  ax.tick_params(axis="x", labelsize=max(ps.FS_TICK - 6, 16), pad=1)
  ax.tick_params(axis="y", labelsize=max(ps.FS_TICK - 6, 16), pad=1)
  ax.set_title(
      r"goal $z$ evolution during training",
      fontsize=ps.FS_TITLE, pad=6)

  for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis.pane.set_edgecolor("0.82")
  ax.grid(True, linestyle=":", alpha=0.28)
  fig.subplots_adjust(left=0.0, right=0.80, bottom=0.0, top=0.93)

  ev = ", ".join(f"{100.0 * r:.0f}%" for r in var_ratio[:2])
  print(f"PCA expl. var (PC1, PC2): [{ev}]; marked first success @ {mark_it}",
        flush=True)

  os.makedirs(os.path.dirname(out_stem) or ".", exist_ok=True)
  ps.savefig(fig, out_stem, dpi=300, pad_inches=0.08)
  plt.close(fig)
  return mark_it


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run_dir", type=str, default=RUN_DIR)
  p.add_argument("--out_stem", type=str, default=OUT_STEM)
  p.add_argument("--reprobe", action="store_true",
                 help="Ignore scores CSV and re-probe checkpoints")
  args = p.parse_args(argv)

  data = _load_or_collect(args.run_dir, force_reprobe=bool(args.reprobe))
  mark_it = _plot_paper(
      data["iters"],
      data["xy"],
      data["var_ratio"],
      success_mean_iter=int(data["success_mean_iter"]),
      out_stem=args.out_stem,
  )
  print(f"marked first success iter={mark_it}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
