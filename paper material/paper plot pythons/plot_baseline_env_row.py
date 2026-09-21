#!/usr/bin/env python3
"""Main-paper eval rows: PIM (ours) vs baselines.

Two figures (eval only). Sawyer bin is omitted.

  1. Highlight row: peg, 4-cube stack, 7-cube zig-zag, 8-cube Jenga,
     Allegro NVIDIA-init throw, plus empty ManiSkill.
  2. Remaining BuilderBench: 3-cube stack, 5-cube pyramid, parallel
     towers.
  3. Standalone 8-cube Jenga tower.

Baselines: PPO+RND, DISCOVER, SGCRL, Dirac state matching (all-zero).
MPO-CRL is omitted for now (data still listed under ALLEGRO_MPO).
DISCOVER is carry-forward peak. Sawyer curves that stop early are smoothly
extrapolated to 40M.

  python "paper material/paper plot pythons/plot_baseline_env_row.py"
"""
from __future__ import annotations

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
import plot_allegro_keeparm_nf_mpo_rnd as rndplot  # noqa: E402
import plot_allegro_new_baselines as akt  # noqa: E402
import plot_builderbench_train_success1000 as base  # noqa: E402
import plot_bb_nf_rnd_success as bb  # noqa: E402
import plot_sawyer_bin_peg_success as sw  # noqa: E402

ps.apply()

OUT_DIR = os.path.join(_PAPER, "paper plots")
OUT_DIR_FINAL = os.path.join(_PAPER, "paper plots", "final final plots")
# Shared by both rows (main + more). Drawn large; shrinks in the paper.
FS_TASK = 72
FS_LEG = 76
FS_AX = 50
FS_LABEL = 66
TITLE_PAD = 12
PANEL_W = 11.6
FIG_H = 13.4
# Vertical layout is shared: more copies this dict and only overrides wspace.
LAYOUT_MAIN = dict(
    wspace=0.28,
    hspace=0.36,
    legend_row=0.14,
    margins=dict(left=0.042, right=0.992, top=0.99, bottom=0.24),
    xlabel_y=0.05,
    save_pad=0.16,
    bbox_inches=None,
    fs_ax=66,
    yticks=(0.0, 0.5, 1.0),
)
# More row: identical vertical layout and fonts; only panel gap differs.
LAYOUT_MORE = {**LAYOUT_MAIN, "wspace": 0.32}
# One-task figure: in-axes legend, larger fonts than the shrunken row.
LAYOUT_SINGLE = {
    **LAYOUT_MAIN,
    "panel_w": 16.0,
    "legend_inside": True,
    "legend_ncol": 1,
    "fs_leg": 46,
    "fs_ax": 64,
    "fs_task": 88,
    "fs_label": 64,
    "margins": dict(left=0.14, right=0.93, top=0.90, bottom=0.16),
    "xlabel_y": 0.03,
}

METHODS = (
    dict(key="nf", label="PIM (ours)", color=ps.C["blue"], ls="-", z=6),
    dict(key="rnd", label="PPO+RND", color=ps.C["vermillion"], ls="-", z=3),
    dict(key="discover", label="DISCOVER", color=ps.C["green"], ls="-", z=5),
    dict(key="sgcrl", label="SGCRL", color=ps.C["purple"], ls="-", z=4),
    dict(key="dirac", label="Dirac state matching", color=ps.C["black"],
         ls=ps.DASH, z=1),
)

PANELS_MAIN = (
    dict(key="peg", kind="sawyer", title="Sawyer peg", horizon=40_000_000),
    dict(key="c4t1", kind="bb", title="4-cube stacking", horizon=200_000_000),
    dict(key="c7t2", kind="bb", title="7-cube zig-zag tower",
         horizon=300_000_000),
    dict(key="c8t2", kind="bb", title="8-cube Jenga tower",
         horizon=300_000_000),
    dict(key="allegro", kind="allegro", title="Allegro grasp and throw",
         horizon=500_000_000),
    dict(key="maniskill", kind="placeholder", title="ManiSkill", horizon=None),
)

PANELS_MORE = (
    dict(key="c3t1", kind="bb", title="3-cube stacking", horizon=200_000_000),
    dict(key="c5t2", kind="bb", title="5-cube pyramid", horizon=200_000_000),
    dict(key="c4t2", kind="bb", title="4-cube parallel towers",
         horizon=200_000_000),
)

PANELS_JENGA = (
    dict(key="c8t2", kind="bb", title="8-cube Jenga tower",
         horizon=200_000_000, clip_horizon=True),
)

FIGURES = (
    dict(name="baseline_env_row_eval", panels=PANELS_MAIN, layout=LAYOUT_MAIN),
    dict(name="baseline_env_row_eval_more", panels=PANELS_MORE,
         layout=LAYOUT_MORE),
    dict(name="baseline_env_row_eval_jenga", panels=PANELS_JENGA,
         layout=LAYOUT_SINGLE),
)

BB_BY_KEY = {t["key"]: t for t in bb.TASKS}

# NVIDIA-init Allegro throw (working NF + baselines).
# Old 16G seed0/1 OOM ~400M; 32G seed0/1/2 continue past that.
ALLEGRO_NF = (
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed0_8h',
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed1_8h',
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed2_8h',
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed0_32G_8h',
    'ppo_allegro_kuka_throw_e1024_nf_compactsmall_nvidiainit'
    '_nvbucket_xy050030_noshape_ep150_fut80_500m_q05shift_noreplace_inbucket_norand_z055_seed1_32G_8h',
)
ALLEGRO_SGCRL = (
    'sac_crl_allegro_kuka_throw_e1024_nvidiainit_z055_ep150_400m_fut80_seed0_10h',
    'sac_crl_allegro_kuka_throw_e1024_nvidiainit_z055_ep150_500m_fut80_seed1_12h',
)
ALLEGRO_MPO = (
    'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_300m_seed0_14h',
    'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed0_14h',
    'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed1_14h',
    'new_allegrobaselines/mpo_nvidiainit_z055_ep150_fut80_500m_seed2_14h',
)
ALLEGRO_RND = (
    'new_allegrobaselines/rnd_nvidiainit_z055_ep150_300m_seed0_5h',
    'new_allegrobaselines/rnd_nvidiainit_z055_ep150_500m_seed0_6h',
    'new_allegrobaselines/rnd_nvidiainit_z055_ep150_500m_seed1_6h',
    'new_allegrobaselines/rnd_nvidiainit_z055_ep150_500m_seed2_6h',
)
ALLEGRO_EVAL_WINDOW = 5


def _carry_forward_pts(pts):
  """Running max: hold each seed at the best eval seen so far."""
  return bb._carry_forward_pts(pts)


def _extend_last(pts, xmax):
  if not pts or xmax is None:
    return list(pts)
  if pts[-1][0] < int(xmax):
    return list(pts) + [(int(xmax), pts[-1][1])]
  return list(pts)


def _series_xmax(seed_series):
  m = 0
  for pts in seed_series:
    if pts:
      m = max(m, pts[-1][0])
  return m


def _clip_series(seed_series, xmax):
  """Drop points past xmax so baselines cannot stretch the NF horizon."""
  out = []
  for pts in seed_series:
    clipped = [(x, y) for x, y in pts if x <= xmax]
    if clipped:
      out.append(clipped)
  return out


def _allegro_eval() -> dict[str, list]:
  nf, sgcrl, mpo, rnd = [], [], [], []
  for d in ALLEGRO_NF:
    nf.extend(base._read_eval_seed_series(base.LOG_ROOT, d))
  for d in ALLEGRO_SGCRL:
    sgcrl.extend(akt._read_sac_crl_eval(d))
  for d in ALLEGRO_MPO:
    mpo.extend(base._read_eval_seed_series(base.LOG_ROOT, d))
  for d in ALLEGRO_RND:
    pts = rndplot._read_rnd_eval(d)
    if pts:
      rnd.append(pts)
  return {
      "nf": nf,
      "sgcrl": sgcrl,
      "mpo": mpo,
      "discover": [],
      "rnd": rnd,
  }


def _sawyer_eval(env_key: str) -> dict[str, list]:
  if env_key == "bin":
    nf_dir = os.path.join(sw.LOG_ROOT, sw.PPO_DIR_BIN_FLOORGRASP)
  else:
    nf_dir = os.path.join(sw.LOG_ROOT, sw.PPO_DIR_PEG)
  return {
      "nf": sw._ppo_eval_series(nf_dir),
      "sgcrl": sw._lp_series(env_key, "evaluator"),
      "mpo": sw._ppo_eval_series(sw.MPO_DIR_BIN if env_key == "bin"
                                else sw.MPO_DIR_PEG),
      "discover": sw._discover_series(env_key, "eval"),
      "rnd": sw._rnd_eval_series(
          sw.RND_DIR_BIN if env_key == "bin" else sw.RND_DIR_PEG),
  }


def _bb_eval(task_key: str) -> dict[str, list]:
  task = BB_BY_KEY[task_key]
  nf_dir = os.path.join(task.get("nf_root", bb.NF_ROOT), task["nf"])
  mpo_stem = task.get("mpo")
  mpo = []
  if mpo_stem:
    mpo = bb._nf_eval_series(os.path.join(bb.MPO_ROOT, mpo_stem))
  _train, disc = bb._discover_series(task_key)
  _rnd_t, rnd = bb._rnd_series(task_key)
  return {
      "nf": bb._nf_eval_series(nf_dir),
      "sgcrl": bb._sgcrl_eval_series(task_key),
      "mpo": mpo,
      "discover": disc,
      "rnd": rnd,
  }


def _prepare(panel: dict) -> dict[str, list]:
  if panel["kind"] == "sawyer":
    data = _sawyer_eval(panel["key"])
  elif panel["kind"] == "allegro":
    data = _allegro_eval()
  else:
    data = _bb_eval(panel["key"])
  if panel["kind"] == "allegro" or panel.get("clip_horizon"):
    # Allegro: 500M. Standalone Jenga: 200M (PIM plateaus ~180M).
    xmax = int(panel["horizon"] or 0)
    for key in list(data):
      data[key] = _clip_series(data[key], xmax)
  else:
    xmax = int(panel["horizon"] or 0)
    for key, series in data.items():
      xmax = max(xmax, _series_xmax(series))
  sawyer = panel["kind"] == "sawyer"
  raw_bin_discover = sawyer and panel["key"] == "bin"
  disc = []
  for pts in data["discover"]:
    if not raw_bin_discover:
      pts = _carry_forward_pts(pts)
    if not sawyer:
      pts = _extend_last(pts, xmax)
    disc.append(pts)
  data["discover"] = disc
  if panel["key"] == "c5t4" and not data["discover"]:
    data["discover"] = [[(0, 0.0), (int(xmax), 0.0)]]
  # Jenga NF: 4/5 seeds reach success; seed 2 never does and ends early.
  # Running-max per seed then hold to xmax so the tail is 0.8, not 1.0
  # from the single longest seed, and not 0.68 from a late 0.4 eval.
  if panel["key"] == "c8t2":
    data["nf"] = [
        _extend_last(_carry_forward_pts(pts), xmax)
        for pts in data.get("nf", [])
    ]
    for key in ("sgcrl", "mpo", "discover", "rnd"):
      data[key] = [_extend_last(pts, xmax) for pts in data.get(key, [])]
  data["dirac"] = [[(0, 0.0), (xmax, 0.0)]]
  data["_xmax"] = xmax
  return data


def _eval_window(panel: dict) -> int:
  if panel["kind"] == "sawyer":
    return sw.EVAL_SMOOTH_WINDOW
  if panel["kind"] == "allegro":
    return ALLEGRO_EVAL_WINDOW
  return bb._eval_window(panel["key"])


def _draw_eval(ax, series, *, color, ls, z, window: int,
               extrapolate_to: int | None = None,
               show_raw: bool = True) -> int:
  xs, mean, se, n = base._aggregate_mean_stderr(series)
  if not xs:
    return 0
  xs, mean, se = base._subsample_curve(xs, mean, se)
  xs, mean, se = bb._prepend_origin(xs, mean, se)
  bb._shade(ax, xs, mean, se, color=color, n=n, eval_smooth=True, window=window)
  sm = base._rolling_mean(mean, window)
  if xs and xs[0] == 0:
    sm[0] = 0.0
  if show_raw:
    ax.plot(
        xs, mean, color=color, linewidth=1.0, alpha=0.28, linestyle=ls,
        marker="o", markersize=3.2, markeredgewidth=0.0, zorder=z)
  if (extrapolate_to is not None and xs and xs[-1] < int(extrapolate_to)):
    tail = bb._smooth_extrapolate_pts(list(zip(xs, sm)), int(extrapolate_to))
    xs = [p[0] for p in tail]
    sm = [p[1] for p in tail]
  ax.plot(
      xs, sm, color=color, linewidth=ps.LW, linestyle=ls, alpha=0.95,
      zorder=z + 1)
  return n


def _draw_dirac(ax, series, *, color, ls, z) -> int:
  if not series or not series[0]:
    return 0
  xs = [p[0] for p in series[0]]
  ys = [p[1] for p in series[0]]
  ax.plot(xs, ys, color=color, linewidth=ps.LW * 0.75, linestyle=ls,
          zorder=z, solid_capstyle="round")
  return 1


def _xtick_locs(xmax: float) -> list[float]:
  """Nice ticks with the last one on the right spine (keeps panel gaps even)."""
  if xmax <= 50_000_000:
    return [0.0, 20_000_000.0, 40_000_000.0]
  if xmax <= 220_000_000:
    return [0.0, 200_000_000.0]
  if xmax <= 350_000_000:
    return [0.0, 300_000_000.0]
  if xmax <= 450_000_000:
    return [0.0, 200_000_000.0, 400_000_000.0]
  return [0.0, 250_000_000.0, 500_000_000.0]


def _finish_ax(ax, *, title: str, ylabel: str, leftmost: bool = True,
               xmax: float | None = None, fs_task: float | None = None,
               fs_label: float | None = None, fs_ax: float | None = None,
               title_pad: float | None = None, yticks=None) -> None:
  ax.set_title(title, fontsize=FS_TASK if fs_task is None else fs_task,
               pad=TITLE_PAD if title_pad is None else title_pad)
  ax.set_ylabel(
      ylabel if leftmost else "",
      fontsize=FS_LABEL if fs_label is None else fs_label)
  ax.set_xlabel("")
  if xmax:
    ax.set_xlim(0, xmax)
    xticks = _xtick_locs(xmax)
    ax.set_xticks(xticks)
    ax.set_xticklabels([base._fmt_steps(t, None) for t in xticks])
  else:
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  ax.set_ylim(-0.05, 1.05)
  if yticks is not None:
    ax.set_yticks(list(yticks))
  ps.style_axes(ax, which="major")
  ax.tick_params(
      axis="both", which="major",
      labelsize=FS_AX if fs_ax is None else fs_ax)
  if xmax:
    labels = ax.get_xticklabels()
    if labels:
      labels[-1].set_horizontalalignment("right")
  if not leftmost:
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylabel("")


def _placeholder(ax, title: str) -> None:
  ax.set_title(title, fontsize=FS_TASK, pad=TITLE_PAD)
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  for name, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linestyle((0, (4, 3)))
    spine.set_color("#B0B0B0")
    spine.set_linewidth(ps.SPINE)
  ax.set_facecolor("#F4F4F4")
  ax.text(
      0.5, 0.5, "placeholder", transform=ax.transAxes,
      ha="center", va="center", color="#888888", fontsize=FS_AX)


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


def _draw_figure(name: str, panels: tuple, layout: dict) -> None:
  n_panels = len(panels)
  panel_w = float(layout.get("panel_w", PANEL_W))
  fig = plt.figure(figsize=(panel_w * n_panels, FIG_H))
  inside = bool(layout.get("legend_inside"))
  if inside:
    gs = fig.add_gridspec(1, n_panels, **layout["margins"])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
  else:
    gs = fig.add_gridspec(
        2, n_panels, height_ratios=[layout["legend_row"], 1.0],
        wspace=layout["wspace"], hspace=layout["hspace"],
        **layout["margins"])
    leg_ax = fig.add_subplot(gs[0, :])
    leg_ax.set_axis_off()
    leg_ax.legend(
        handles=_legend_handles(), loc="center",
        ncol=int(layout.get("legend_ncol", len(METHODS))),
        frameon=True, fancybox=False, framealpha=0.96, edgecolor="#B0B0B0",
        handlelength=2.8, handleheight=1.4, columnspacing=1.4,
        fontsize=layout.get("fs_leg", FS_LEG))
    axes = [fig.add_subplot(gs[1, i]) for i in range(n_panels)]
  data_axes = [ax for ax, p in zip(axes, panels) if p["kind"] != "placeholder"]
  for ax in data_axes[1:]:
    ax.sharey(data_axes[0])

  for i, (panel, ax) in enumerate(zip(panels, axes)):
    if panel["kind"] == "placeholder":
      _placeholder(ax, panel["title"])
      continue
    data = _prepare(panel)
    w = _eval_window(panel)
    print(f"== {name} / {panel['key']}  "
          f"xmax={data['_xmax']/1e6:.1f}M  w={w} ==")
    for method in METHODS:
      series = data.get(method["key"], [])
      if method["key"] == "dirac":
        n = _draw_dirac(
            ax, series, color=method["color"], ls=method["ls"], z=method["z"])
      else:
        n = _draw_eval(
            ax, series, color=method["color"], ls=method["ls"],
            z=method["z"], window=w,
            extrapolate_to=(data["_xmax"] if panel["kind"] == "sawyer"
                            else None))
      last = ""
      xs, mean, _se, _n = base._aggregate_mean_stderr(series)
      if xs:
        last = f" last={mean[-1]:.3f} peak={max(mean):.3f}"
      print(f"  {method['label']:22s} n={n}{last}")
    _finish_ax(
        ax, title=panel["title"], ylabel="Eval success", leftmost=(i == 0),
        xmax=data["_xmax"] if data["_xmax"] else 1,
        fs_ax=layout.get("fs_ax"), fs_task=layout.get("fs_task"),
        fs_label=layout.get("fs_label"), yticks=layout.get("yticks"))
    if inside and i == 0:
      ax.legend(
          handles=_legend_handles(), loc="upper left",
          ncol=int(layout.get("legend_ncol", 1)),
          frameon=True, fancybox=False, framealpha=0.94, edgecolor="#B0B0B0",
          handlelength=2.2, handleheight=1.2, borderpad=0.5,
          labelspacing=0.35, fontsize=layout.get("fs_leg", FS_LEG))

  fig.text(
      0.5, layout["xlabel_y"], "Environment steps", ha="center", va="bottom",
      fontsize=layout.get("fs_label", FS_LABEL), clip_on=False)
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
