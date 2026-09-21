#!/usr/bin/env python3
"""PIM (ours) vs MPO-CRL, eval only, baseline-row formatting.

Same loaders, smoothing, and fonts as plot_baseline_env_row.py.
7-cube and 8-cube have no final MPO run; MPO-CRL is a flat zero.
Allegro MPO is the NVIDIA-init 500M seeds only (the 300M seed 0
is the superseded run).

  python "paper material/paper plot pythons/plot_pim_vs_mpo_crl.py"
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import paper_style as ps  # noqa: E402
import plot_baseline_env_row as row  # noqa: E402

ps.apply()

OUT_NAME = "pim_vs_mpo_crl_eval"

METHODS = (
    dict(key="nf", label="Goal preimage state matching", color=ps.C["blue"],
         ls="-", z=6),
    dict(key="mpo", label="Goal preimage action matching (MPO-CRL)",
         color=ps.C["orange"], ls="-", z=3),
)

# Paper baseline order, then 5-cube overhang (MPO task the baseline row omits).
PANELS = (
    dict(key="peg", kind="sawyer", title="Sawyer peg", horizon=40_000_000),
    dict(key="c3t1", kind="bb", title="3-cube stacking", horizon=200_000_000),
    dict(key="c4t1", kind="bb", title="4-cube stacking", horizon=200_000_000),
    dict(key="c4t2", kind="bb", title="4-cube parallel towers",
         horizon=200_000_000),
    dict(key="c5t2", kind="bb", title="5-cube pyramid", horizon=200_000_000),
    dict(key="c5t4", kind="bb", title="5-cube overhang", horizon=200_000_000),
    dict(key="c7t2", kind="bb", title="7-cube zig-zag tower",
         horizon=300_000_000, mpo_zero=True),
    dict(key="c8t2", kind="bb", title="8-cube Jenga tower",
         horizon=300_000_000, mpo_zero=True),
    dict(key="allegro", kind="allegro", title="Allegro grasp and throw",
         horizon=500_000_000),
)

ALLEGRO_MPO_500 = (
    "new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed0_14h",
    "new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed1_14h",
    "new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed2_14h",
)


def _allegro_mpo_500() -> list:
  mpo = []
  for d in ALLEGRO_MPO_500:
    mpo.extend(row.base._read_eval_seed_series(row.base.LOG_ROOT, d))
  return mpo


def _prepare(panel: dict) -> dict:
  data = row._prepare(panel)
  if panel["kind"] == "allegro":
    data["mpo"] = row._clip_series(_allegro_mpo_500(), data["_xmax"])
  if panel.get("mpo_zero"):
    xmax = int(data["_xmax"])
    data["mpo"] = [[(0, 0.0), (xmax, 0.0)]]
  return data


def _legend_handles():
  return [
      Line2D([0], [0], color=m["color"], lw=ps.LW, ls=m["ls"], label=m["label"])
      for m in METHODS
  ]


# Two rows (5 + 4). Panel width is set from the longest title at FS_TASK
# ("Allegro grasp and throw", ~16in) so neighbors do not collide.
N_COLS = 5
FIG_W = 90.0
FIG_H = 39.0
FS_TASK = 96
FS_LEG = 92
FS_AX = 84
FS_LABEL = 88
_LEFT_IN = 4.3
_RIGHT_IN = 0.7
_GAP_IN = 0.55
_AX_W_IN = (FIG_W - _LEFT_IN - _RIGHT_IN - (N_COLS - 1) * _GAP_IN) / N_COLS
_AX_H_IN = 11.4


def _row_positions(n: int, y_in: float) -> list[tuple[float, float, float, float]]:
  """Equal-width panels, in figure fractions. A short row is centered."""
  pw = _AX_W_IN / FIG_W
  gap = _GAP_IN / FIG_W
  h = _AX_H_IN / FIG_H
  y = y_in / FIG_H
  row_w = n * pw + (n - 1) * gap
  content = 1.0 - _LEFT_IN / FIG_W - _RIGHT_IN / FIG_W
  x0 = _LEFT_IN / FIG_W + (content - row_w) / 2
  return [(x0 + i * (pw + gap), y, pw, h) for i in range(n)]


def main() -> None:
  panels = PANELS
  n_top = N_COLS
  fig = plt.figure(figsize=(FIG_W, FIG_H))
  positions = _row_positions(n_top, 21.0) + _row_positions(len(panels) - n_top, 4.2)
  axes = [fig.add_axes(rect) for rect in positions]
  for ax in axes[1:]:
    ax.sharey(axes[0])
  fig.legend(
      handles=_legend_handles(), loc="upper center",
      bbox_to_anchor=(0.5, 0.985), ncol=len(METHODS),
      frameon=True, fancybox=False, framealpha=0.96, edgecolor="#B0B0B0",
      handlelength=2.8, handleheight=1.4, columnspacing=1.6,
      fontsize=FS_LEG)

  for i, (panel, ax) in enumerate(zip(panels, axes)):
    data = _prepare(panel)
    w = row._eval_window(panel)
    print(f"== {panel['key']}  xmax={data['_xmax']/1e6:.1f}M  w={w} ==")
    for method in METHODS:
      series = data.get(method["key"], [])
      if panel.get("mpo_zero") and method["key"] == "mpo":
        n = row._draw_dirac(
            ax, series, color=method["color"], ls=method["ls"], z=method["z"])
        print(f"  {method['label']:22s} n=0  flat zero")
        continue
      n = row._draw_eval(
          ax, series, color=method["color"], ls=method["ls"],
          z=method["z"], window=w,
          extrapolate_to=(data["_xmax"] if panel["kind"] == "sawyer"
                          else None))
      last = ""
      xs, mean, _se, _n = row.base._aggregate_mean_stderr(series)
      if xs:
        last = f" last={mean[-1]:.3f} peak={max(mean):.3f}"
      print(f"  {method['label']:22s} n={n}{last}")
    leftmost = i == 0 or i == n_top
    row._finish_ax(
        ax, title=panel["title"], ylabel="Eval success", leftmost=leftmost,
        xmax=data["_xmax"] if data["_xmax"] else 1,
        fs_task=FS_TASK, fs_label=FS_LABEL, fs_ax=FS_AX,
        title_pad=20, yticks=row.LAYOUT_MAIN.get("yticks"))

  fig.text(
      0.5, 0.018, "Environment steps", ha="center", va="bottom",
      fontsize=FS_LABEL, clip_on=False)
  stem, stem_final = row._stems(OUT_NAME)
  os.makedirs(os.path.dirname(stem), exist_ok=True)
  os.makedirs(os.path.dirname(stem_final), exist_ok=True)
  ps.savefig(fig, stem, png=False, pad_inches=0.12, bbox_inches=None)
  ps.savefig(fig, stem_final, png=False, pad_inches=0.12, bbox_inches=None)
  plt.close(fig)
  print(f"→ {stem}.pdf")
  print(f"→ {stem_final}.pdf")


if __name__ == "__main__":
  main()
