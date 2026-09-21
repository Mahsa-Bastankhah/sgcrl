#!/usr/bin/env python3
"""Overlay NF and DISCOVER c3t1 unique discretized cube states.

Reads the existing novel-states CSVs (same 2cm bins, 5 stochastic trajs).
Cumulative unique only — per-ckpt bars overlap too much across methods.

  python scripts/plot_bb_nf_discover_novel_states.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO = Path(__file__).resolve().parents[1]
_PAPER_PY = _REPO / "paper material" / "paper plot pythons"
if str(_PAPER_PY) not in sys.path:
  sys.path.insert(0, str(_PAPER_PY))

DASH_LW = 2.2
FIG_H = 6.2
FIG_W = 8.6

NF_CSV = (
    _REPO / "logs"
    / "ppo_builderbench_creative3_task1_e1024_pd_nf_compact_small_sa3x192_"
      "r64_b6_w192_tau05_nopermute_fixedx01_catwp_extrew1_minstd1e5_ent05to001_"
      "ep50_ckpt10_iter200_crl10_dualgradreg_c100_lamlr1e6_valuedgr_c100_"
      "lamlr1e6_warp"
    / "ppo_builderbench_creative_3_task1_0"
    / "novel_states" / "seed0_eps0.02.csv"
)
NF_RUN = NF_CSV.parents[1]
DISC_CSV = (
    _REPO / "logs"
    / "discover_builderbench_creative3_task1_e1024_pd_catwp_ucbstd0_"
      "nopermute_fixedx01_warp_8h"
    / "discover_builderbench_creative_3_task1_1"
    / "novel_states" / "seed1_eps0.02.csv"
)
DISC_LOGS = DISC_CSV.parents[1] / "logs.csv"
OUTS = (
    _REPO / "paper material" / "paper plots" / "bb_nf_discover_c3t1_novel_states",
    _REPO / "paper material" / "paper plots" / "final final plots"
    / "bb_nf_discover_c3t1_novel_states",
)


def _read_novel(path: Path) -> list[dict]:
  rows = []
  with path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      rows.append({
          "env_steps": int(float(row["env_steps"])),
          "n_cumulative": int(float(row["n_cumulative"])),
      })
  rows.sort(key=lambda r: r["env_steps"])
  return rows


def _nf_first_ok(threshold: float = 0.1) -> tuple[int, int, float]:
  it_map: dict[int, int] = {}
  learner = NF_RUN / "logs" / "learner" / "logs.csv"
  if learner.is_file():
    with learner.open(newline="", encoding="utf-8") as fh:
      for row in csv.DictReader(fh):
        it_map[int(float(row["iteration"]))] = int(float(row["global_step"]))
  eval_path = NF_RUN / "logs" / "eval" / "logs.csv"
  with eval_path.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      it = int(float(row["iteration"]))
      y = float(row["success"])
      x = it_map.get(it, it * 1024 * 50)
      if y >= threshold:
        return it, x, y
  raise SystemExit("no NF eval success >= 0.1")


def _disc_first_ok(threshold: float = 0.1) -> tuple[int, int, float]:
  with DISC_LOGS.open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
      ep = int(float(row["epoch"]))
      x = int(float(row["global_step"]))
      y = float(row["eval_success_any_microstep"])
      if y >= threshold:
        return ep, x, y
  raise SystemExit("no DISCOVER eval success >= 0.1")


def _fmt_steps(v, _=None):
  if v == 0:
    return "0"
  if v >= 1e6:
    s = f"{v / 1e6:.1f}M"
    return s.replace(".0M", "M")
  if v >= 1e3:
    return f"{v / 1e3:.0f}K"
  return str(int(v))


def _savefig_fixed_height(fig, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fig.set_size_inches(FIG_W, FIG_H, forward=True)
  fig.canvas.draw()
  from matplotlib.transforms import Bbox
  box = Bbox.from_bounds(0, 0, FIG_W, FIG_H)
  p = path.with_suffix(".pdf")
  fig.savefig(p, bbox_inches=box, pad_inches=0, facecolor="white")
  print("Saved:", p)


def main() -> None:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import matplotlib.ticker as mticker
  from matplotlib.lines import Line2D
  import paper_style as ps

  nf_rows = _read_novel(NF_CSV)
  disc_rows = _read_novel(DISC_CSV)
  if not nf_rows or not disc_rows:
    raise SystemExit("missing novel-states CSV")
  nf_ok = _nf_first_ok()
  disc_ok = _disc_first_ok()
  nf_xmax = int(nf_rows[-1]["env_steps"])
  kept = [r for r in disc_rows if int(r["env_steps"]) <= nf_xmax]
  rest = [r for r in disc_rows if int(r["env_steps"]) > nf_xmax]
  if rest and int(rest[0]["env_steps"]) - nf_xmax <= 3_000_000:
    kept.append(rest[0])
  disc_rows = kept

  ps.apply()
  fig, ax = ps.figure("single")
  nf_c = ps.C["blue"]
  disc_c = ps.C["vermillion"]
  ax.plot(
      [r["env_steps"] for r in nf_rows],
      [r["n_cumulative"] for r in nf_rows],
      color=nf_c, linewidth=ps.LW, solid_capstyle="round",
      zorder=4, marker="o", markersize=7)
  ax.plot(
      [r["env_steps"] for r in disc_rows],
      [r["n_cumulative"] for r in disc_rows],
      color=disc_c, linewidth=ps.LW, solid_capstyle="round",
      zorder=3, marker="o", markersize=7)
  ax.axvline(
      int(nf_ok[1]), color=nf_c, linestyle=ps.DASH, linewidth=DASH_LW,
      zorder=1)
  ax.axvline(
      int(disc_ok[1]), color=disc_c, linestyle=ps.DASH, linewidth=DASH_LW,
      zorder=1)
  ax.set_xlabel("Environment steps", fontsize=ps.FS_LABEL)
  ax.set_ylabel("cumulative # of\nvisited states", fontsize=ps.FS_LABEL)
  ax.set_title("DISCOVER vs PIM exploration rate", fontsize=ps.FS_TITLE, pad=ps.TITLE_PAD)
  ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_steps))
  ax.set_xlim(0, nf_xmax * 1.08)
  ax.set_ylim(bottom=0)
  ps.style_axes(ax, which="major")
  ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=False))
  handles = [
      Line2D([0], [0], color=nf_c, lw=ps.LW, marker="o", label="PIM (Ours)"),
      Line2D([0], [0], color=disc_c, lw=ps.LW, marker="o", label="DISCOVER"),
      Line2D(
          [0], [0], color=ps.C["gray"], linestyle=ps.DASH, linewidth=DASH_LW,
          label="first success"),
  ]
  ps.nice_legend(ax, handles=handles, loc="upper left")
  for out in OUTS:
    _savefig_fixed_height(fig, out)
  plt.close(fig)


if __name__ == "__main__":
  main()
