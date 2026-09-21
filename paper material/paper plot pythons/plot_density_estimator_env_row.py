#!/usr/bin/env python3
"""Main-paper eval rows: PIM density-estimator ablations.

Two figures (eval only). BuilderBench-only (no Sawyer / ManiSkill).

  1. Highlight row: 4-cube stacking, 5-cube pyramid, parallel towers,
     underspecified towers.
  2. Remaining: 3-cube stack, maximum overhang, zig-zag, Jenga.

PIM (NF) uses the same blue as plot_baseline_env_row.py.
Missing estimator runs (c5t4, c8t2 CRL/TD3/TD-InfoNCE) are plotted as zeros.
Layout matches the baseline env rows (fonts, legend row, panel gaps).

  python "paper material/paper plot pythons/plot_density_estimator_env_row.py"
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
_PAPER = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PAPER)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

import paper_style as ps  # noqa: E402
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_bb_nf_rnd_success as bb  # noqa: E402
import plot_baseline_env_row as env_row  # noqa: E402

ps.apply()

OUT_DIR = os.path.join(_PAPER, "paper plots")
OUT_DIR_FINAL = os.path.join(_PAPER, "paper plots", "final final plots")
LOG_ROOT = os.path.join(_REPO, "logs")
# Match baseline vertical layout and fonts. 4-column rows use a wider
# gap; the more row is wider still so long titles do not collide.
LAYOUT_MAIN = env_row.LAYOUT_MORE
LAYOUT_MORE = {**env_row.LAYOUT_MORE, "wspace": 1.05}
C5T3_SMOOTH_WINDOW = 11

PANELS_MAIN = (
    dict(key="c4t1", kind="bb", title="4-cube stacking", horizon=200_000_000),
    dict(key="c5t2", kind="bb", title="5-cube pyramid", horizon=200_000_000),
    dict(key="c4t2", kind="bb", title="4-cube parallel towers",
         horizon=200_000_000),
    dict(key="c5t3", kind="bb", title="Underspecified towers",
         horizon=200_000_000),
)

PANELS_MORE = (
    dict(key="c3t1", kind="bb", title="3-cube stacking", horizon=200_000_000),
    dict(key="c5t4", kind="bb", title="5-cube maximum overhang",
         horizon=200_000_000),
    dict(key="c7t2", kind="bb", title="7-cube zig-zag tower",
         horizon=300_000_000),
    dict(key="c8t2", kind="bb", title="8-cube Jenga tower",
         horizon=300_000_000),
)

FIGURES = (
    dict(name="density_estimator_env_row_eval", panels=PANELS_MAIN,
         layout=LAYOUT_MAIN),
    dict(name="density_estimator_env_row_eval_more", panels=PANELS_MORE,
         layout=LAYOUT_MORE),
)

NF_C5T3 = (
    "ppo_builderbench_creative5_task3_e1024_pd_nf_compact_"
    "sa3x256_r64_b6_w256_tau05_actorreset_nopermute_norand_"
    "minstd1e5_entanneal_evalvid_catselect_extrew1"
)

METHODS = (
    dict(key="nf", label="PIM (NF)", color=ps.C["blue"], ls="-", z=6),
    dict(key="crl", label="PIM (CRL)", color=ps.C["orange"], ls="-", z=4),
    dict(key="td3", label="PIM (TD3)", color=ps.C["vermillion"], ls="-", z=3),
    dict(key="tdinfo", label="PIM (TD-InfoNCE)", color=ps.C["green"], ls="-", z=5),
)

# Paper-selected recipes. Multiple paths are pooled (e.g. 10- and 25-update).
ESTIMATOR_RUNS = {
    "c3t1": dict(
        crl=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_"
            "catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30",
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_"
            "catselect_extrew1_ep50_200m_crl25_eval10_warp_4h",
        ),
        td3=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_"
            "catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30",
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative3_task1_e1024_pd_td3_logq_tau05_"
            "catselect_extrew1_ep50_200m_crl25_eval10_warp_4h",
        ),
        tdinfo=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative3_task1_e1024_pd_tdinfonce_tau05_"
            "catselect_ep50_200m_crl10_eval10_warp_2h30",
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative3_task1_e1024_pd_tdinfonce_tau05_"
            "catselect_ep50_200m_crl25_eval10_warp_4h",
        ),
    ),
    "c4t1": dict(
        crl=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task1_e1024_pd_crl_tau05_actorreset_"
            "nopermute_norand_normobs_catselect_minstd1e4_extrew1_ep50_200m_"
            "crl25_eval10_warp_4h",
        ),
        td3=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task1_e1024_pd_td3_logq_tau05_"
            "actorreset_nopermute_catselect_minstd1e4_fixedx01_extrew1_ep50_"
            "200m_crl25_eval10_warp_4h",
        ),
        tdinfo=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task1_e1024_pd_tdinfonce_tau05_"
            "actorreset_nopermute_catselect_minstd1e4_extrew1_ep50_200m_"
            "crl10_eval10_warp_2h30",
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task1_e1024_pd_tdinfonce_tau05_"
            "actorreset_nopermute_catselect_minstd1e4_extrew1_ep50_200m_"
            "crl25_eval10_warp_4h",
        ),
    ),
    "c4t2": dict(
        crl=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_"
            "catselect_stateonly_extrew1_ep50_200m_crl10_eval10_warp_2h30",
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task2_e1024_pd_crl_tau05_"
            "catselect_stateonly_extrew1_ep50_200m_crl25_eval10_warp_4h",
        ),
        td3=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_"
            "catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30",
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task2_e1024_pd_td3_logq_tau05_"
            "catselect_extrew1_ep50_200m_crl25_eval10_warp_4h",
        ),
        tdinfo=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_"
            "catselect_extrew1_ep50_200m_crl10_eval10_warp_2h30",
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative4_task2_e1024_pd_tdinfonce_tau05_"
            "catselect_extrew1_ep50_200m_crl25_eval10_warp_4h",
        ),
    ),
    "c5t2": dict(
        crl=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task2_e1024_pd_crl_tau05_"
            "actorreset_evalvid_catselect_extrew1_ep70_200m_crl10_"
            "eval10_warp_2h30",
        ),
        td3=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task2_e1024_pd_td3_logq_tau05_"
            "actorreset_nopermute_normobs_evalvid_catselect_extrew1_"
            "ep70_200m_crl25_eval10_warp_4h",
        ),
        tdinfo=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task2_e1024_pd_tdinfonce_tau05_"
            "catselect_extrew1_ep70_200m_crl25_eval10_warp_4h",
        ),
    ),
    "c5t3": dict(
        crl=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task3_e1024_pd_crl_tau05_"
            "actorreset_nopermute_norand_normobs_evalvid_catselect_extrew1_"
            "ep70_200m_crl10_eval10_warp_2h30",
        ),
        td3=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task3_e1024_pd_td3_logq_tau05_"
            "actorreset_permute_rand_evalvid_catselect_extrew1_ep70_200m_"
            "crl10_eval10_warp_2h30",
        ),
        tdinfo=(
            "final_runs/other_density_estimators/"
            "ppo_builderbench_creative5_task3_e1024_pd_tdinfonce_tau05_"
            "catselect_extrew1_ep70_200m_crl10_eval10_warp_2h30",
        ),
    ),
    "c7t2": dict(
        crl=(
            "final_runs/"
            "ppo_builderbench_creative7_task2_e1024_pd_crl_tau05_"
            "nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_"
            "200m_crl10_dualgradreg_c100_lamlr1e2_valuedgr_c100_"
            "lamlr1e6_warp_2h30",
        ),
        td3=(
            "ppo_builderbench_creative7_task2_e1024_pd_td3_logq_tau05_"
            "actorreset_nopermute_normobs_evalvid_catwp_tol0013",
        ),
        tdinfo=(
            "final_runs/"
            "ppo_builderbench_creative7_task2_e1024_pd_tdinfonce_tau05_"
            "nopermute_fixedx01_catwp_extrew1_minstd1e5_entanneal_ep60_"
            "200m_crl10_warp_2h30",
        ),
    ),
}


def _eval_runs(run_rels: tuple[str, ...]) -> list[list[tuple[int, float]]]:
  series = []
  for rel in run_rels:
    series.extend(base._read_eval_seed_series(LOG_ROOT, rel))
  return series


def _zeros(xmax: int) -> list[list[tuple[int, float]]]:
  return [[(0, 0.0), (int(xmax), 0.0)]]


def _nf_eval(panel: dict) -> list[list[tuple[int, float]]]:
  if panel["key"] == "c5t3":
    return bb._nf_eval_series(os.path.join(LOG_ROOT, NF_C5T3))
  task = env_row.BB_BY_KEY[panel["key"]]
  nf_dir = os.path.join(task.get("nf_root", bb.NF_ROOT), task["nf"])
  return bb._nf_eval_series(nf_dir)


def _prepare(panel: dict) -> dict:
  xmax = int(panel["horizon"] or 0)
  nf = _nf_eval(panel)
  xmax = max(xmax, env_row._series_xmax(nf))
  specs = ESTIMATOR_RUNS.get(panel["key"], {})
  data = {
      "nf": nf,
      "crl": _eval_runs(specs.get("crl", ())),
      "td3": _eval_runs(specs.get("td3", ())),
      "tdinfo": _eval_runs(specs.get("tdinfo", ())),
  }
  for series in data.values():
    xmax = max(xmax, env_row._series_xmax(series))
  if panel["key"] == "c8t2":
    data["nf"] = [
        env_row._extend_last(env_row._carry_forward_pts(pts), xmax)
        for pts in data["nf"]
    ]
  # TD-InfoNCE on zig-zag is a 200M run; hold last eval to the 300M axis.
  if panel["key"] == "c7t2":
    data["tdinfo"] = [
        env_row._extend_last(pts, xmax) for pts in data["tdinfo"]
    ]
  for key in ("crl", "td3", "tdinfo"):
    if not data[key]:
      data[key] = _zeros(xmax)
  data["_xmax"] = xmax
  return data


def _legend_handles():
  return [
      Line2D([0], [0], color=m["color"], lw=ps.LW, ls=m["ls"], label=m["label"])
      for m in METHODS
  ]


def _stems(name: str) -> tuple[str, str]:
  return (
      os.path.join(OUT_DIR, name),
      os.path.join(OUT_DIR_FINAL, name),
  )


def _eval_window(panel: dict) -> int:
  if panel["key"] == "c5t3":
    return C5T3_SMOOTH_WINDOW
  return env_row._eval_window(panel)


def _draw_figure(name: str, panels: tuple, layout: dict) -> None:
  n_panels = len(panels)
  fig = plt.figure(figsize=(env_row.PANEL_W * n_panels, env_row.FIG_H))
  gs = fig.add_gridspec(
      2, n_panels, height_ratios=[layout["legend_row"], 1.0],
      wspace=layout["wspace"], hspace=layout["hspace"],
      **layout["margins"])
  leg_ax = fig.add_subplot(gs[0, :])
  leg_ax.set_axis_off()
  leg_ax.legend(
      handles=_legend_handles(), loc="center", ncol=len(METHODS),
      frameon=True, fancybox=False, framealpha=0.96, edgecolor="#B0B0B0",
      handlelength=2.8, handleheight=1.4, columnspacing=1.4,
      fontsize=env_row.FS_LEG)
  axes = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]
  for ax in axes[1:]:
    ax.sharey(axes[0])

  for i, (panel, ax) in enumerate(zip(panels, axes)):
    data = _prepare(panel)
    w = _eval_window(panel)
    show_raw = panel["key"] != "c5t3"
    print(f"== {name} / {panel['key']}  "
          f"xmax={data['_xmax']/1e6:.1f}M  w={w}  raw={show_raw} ==")
    for method in METHODS:
      series = data.get(method["key"], [])
      n = env_row._draw_eval(
          ax, series, color=method["color"], ls=method["ls"],
          z=method["z"], window=w, show_raw=show_raw)
      last = ""
      xs, mean, _se, _n = base._aggregate_mean_stderr(series)
      if xs:
        last = f" last={mean[-1]:.3f} peak={max(mean):.3f}"
      print(f"  {method['label']:24s} n={n}{last}")
    env_row._finish_ax(
        ax, title=panel["title"], ylabel="Eval success", leftmost=(i == 0),
        xmax=data["_xmax"] if data["_xmax"] else 1,
        fs_task=env_row.FS_TASK, fs_label=env_row.FS_LABEL,
        fs_ax=layout.get("fs_ax"), title_pad=env_row.TITLE_PAD,
        yticks=layout.get("yticks"))

  fig.text(
      0.5, layout["xlabel_y"], "Environment steps", ha="center", va="bottom",
      fontsize=env_row.FS_LABEL, clip_on=False)
  stem, stem_final = _stems(name)
  os.makedirs(os.path.dirname(stem), exist_ok=True)
  os.makedirs(os.path.dirname(stem_final), exist_ok=True)
  ps.savefig(
      fig, stem, png=False, pad_inches=layout["save_pad"],
      bbox_inches=layout.get("bbox_inches", "tight"))
  ps.savefig(
      fig, stem_final, png=False, pad_inches=layout["save_pad"],
      bbox_inches=layout.get("bbox_inches", "tight"))
  plt.close(fig)
  print(f"→ {stem}.pdf")
  print(f"→ {stem_final}.pdf")


def main() -> None:
  for fig in FIGURES:
    _draw_figure(fig["name"], fig["panels"], fig["layout"])


if __name__ == "__main__":
  main()
