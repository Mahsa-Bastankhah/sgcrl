#!/usr/bin/env python3
"""c5t2 nfbwd iter 1000: log p(g|s,a) vs ‖∇_s log p‖ (twin-y paper figure).

  python "paper material/paper plot pythons/plot_c5t2_nfbwd_iter1000_ppo_r_grad_s.py"
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
  sys.path.insert(0, _here)

import paper_style as ps
ps.apply()

_paper = os.path.dirname(_here)
_repo = os.path.dirname(_paper)
OUT_STEM = os.path.join(
    _paper, "paper plots",
    "c5t2_nfbwd_iter1000_ppo_r_grad_s")

CSV_PATH = os.path.join(
    _repo, "figs", "builderbench", "nf_logp_reward_probe",
    "c5t2_nfbwd_succ_s0_online_all",
    "c5t2_nfbwd_succ_s0_online_iter_0001000.csv")


def main() -> None:
  d = np.genfromtxt(CSV_PATH, delimiter=",", names=True)
  t = d["t"]
  logp = d["reward_logp_online"]
  success = d["success"] >= 0.5
  grad_s = d["grad_s_norm"]
  pos_sum = d["pos_delta_sum"]
  cube_cols = [n for n in d.dtype.names if n.startswith("pos_delta_c")]

  t_succ = int(np.argmax(success))
  move_incl = float(pos_sum[t_succ:].sum())
  move_after = float(pos_sum[t_succ + 1:].sum())
  print(f"first success t = {t_succ} / {int(t[-1])}")
  print(f"sum_i ||Δxyz_i||  success→end (incl success step) = {move_incl:.4g} m")
  print(f"sum_i ||Δxyz_i||  after success (excl success step) = {move_after:.4g} m")
  for name in cube_cols:
    print(f"  {name} success→end = {float(d[name][t_succ:].sum()):.4g} m")

  fig, ax = ps.figure("single")
  fig.set_size_inches(11.0, 6.2)
  ax.plot(
      t, logp, color=ps.C["blue"], label=r"$\log p(g\mid s,a)$")
  ax.axvline(t_succ, color=ps.C["green"], linestyle=ps.DASH, zorder=2)
  ax.set_xlabel(r"$t$")
  ax.set_ylabel(r"$\log p(g\mid s,a)$")
  ax.set_title("NF log likelihood after success")
  ax.set_xlim(float(t[0]), float(t[-1]))
  ps.style_axes(ax, which="major")

  ax2 = ax.twinx()
  ax2.plot(
      t, grad_s, color=ps.C["vermillion"],
      label=r"$\Vert\nabla_s \log p(g\mid s,a)\Vert$")
  ax2.set_ylabel(r"$\Vert\nabla_s \log p(g\mid s,a)\Vert$")
  ps.style_twin(ax, ax2, left_dashed=False)
  ax.spines["left"].set_linestyle("solid")
  # Twin-y: hide ax2's white patch so the left curve shows through. An axes
  # legend cannot occlude the other axes, so the legend is a figure artist.
  ax2.patch.set_visible(False)
  ax.set_zorder(1)
  ax2.set_zorder(2)

  ax.annotate(
      "success", xy=(t_succ, logp[t_succ]),
      xytext=(6, -22), textcoords="offset points",
      color=ps.C["green"], va="top", ha="left", zorder=10)

  h1, l1 = ax.get_legend_handles_labels()
  h2, l2 = ax2.get_legend_handles_labels()
  # Right side, slightly above center, in the gap after success.
  leg = ax2.legend(
      h1 + h2, l1 + l2,
      loc="center right",
      bbox_to_anchor=(1.0, 0.58),
      frameon=True,
      fancybox=False,
      framealpha=1.0,
      facecolor="white",
      edgecolor="#B0B0B0",
      handlelength=1.0,
      borderpad=0.3,
      labelspacing=0.28,
      handletextpad=0.4,
      fontsize=26)
  leg.get_frame().set_linewidth(1.1)
  leg.get_frame().set_alpha(1.0)
  try:
    leg.get_frame().set_boxstyle("square")
  except Exception:
    pass
  leg.set_zorder(20)
  os.makedirs(os.path.dirname(OUT_STEM), exist_ok=True)
  ps.savefig(fig, OUT_STEM)
  plt.close(fig)


if __name__ == "__main__":
  main()
