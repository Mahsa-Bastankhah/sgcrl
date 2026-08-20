#!/usr/bin/env python3
"""Normalized hard-goal coords for BuilderBench creative-5 task2 NF.

Same diagnostic as peg/bin: z_i = (g_task_i - nf_goal_mean_i) / nf_goal_std_i.
g_task is the 15-D packed cube-xyz pyramid printed at run start
(hard_goal in the slurm log). NF always z-scores this vector before the flow
(unless --nonf_normalize_goals).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / (
    "logs/ppo_builderbench_creative5_task2_e1024_pd_nf_compact_small_"
    "sa3x192_r64_b6_w192_tau05_nopermute_catselect_sfpert_spert001")
SEEDS = [("seed 0", RUN / "ppo_builderbench_creative_5_task2_0")]

# From slurm log hard_goal (cubes), 5 cubes × xyz.
G_TASK = np.array([
    0.27, -0.035, 0.02,
    0.27,  0.035, 0.02,
    0.27, -0.02,  0.06,
    0.27,  0.02,  0.06,
    0.27,  0.0,   0.10,
], dtype=np.float64)
N = 15
DIM_LABELS = [f"c{c}{ax}" for c in range(5) for ax in "xyz"]
CUBE_COLORS = ["#d1495b", "#2d6a4f", "#1d4e89", "#b08900", "#6a4c93"]


def main():
    fig, axes = plt.subplots(len(SEEDS), 3, figsize=(19, 4.4 * len(SEEDS)), squeeze=False)

    for r, (sname, sdir) in enumerate(SEEDS):
        df = pd.read_csv(sdir / "logs" / "learner" / "logs.csv")
        x = df["global_step"].to_numpy(dtype=float) / 1e6
        mean = np.column_stack(
            [pd.to_numeric(df[f"nf/goal_mean_{i}"], errors="coerce") for i in range(N)])
        std = np.column_stack(
            [pd.to_numeric(df[f"nf/goal_std_{i}"], errors="coerce") for i in range(N)])
        z = (G_TASK[None, :] - mean) / std
        znorm = np.linalg.norm(z, axis=1)
        prior_logp = -0.5 * znorm ** 2
        raw = pd.to_numeric(df["reward_repr_raw_mean"], errors="coerce").to_numpy(dtype=float)

        ax = axes[r][0]
        for i in range(N):
            cube, ax_i = divmod(i, 3)
            ls = ["-", "--", ":"][ax_i]
            ax.plot(x, z[:, i], lw=1.1, color=CUBE_COLORS[cube], ls=ls,
                    label=DIM_LABELS[i])
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"BB c5-t2 {sname}: normalized hard goal\n"
                     r"$z_i=(g_{\mathrm{task}}-\mu)/\sigma$  (solid=x, dash=y, dot=z)")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("z (std units)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=6, ncol=5, loc="best")

        ax = axes[r][1]
        ax.plot(x, znorm, color="#d1495b", lw=1.6, label=r"$\|z\|$ (task goal)")
        ax.set_title(f"BB c5-t2 {sname}: $\|z\|$ (how OOD the pyramid is)")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel(r"$\|z\|$")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")

        ax = axes[r][2]
        ax.plot(x, prior_logp, color="#5b8c5a", lw=1.5, label=r"$-0.5\|z\|^2$ (base prior)")
        ax.plot(x, raw, color="#d1495b", lw=1.0, alpha=0.8, label="reward_repr_raw_mean")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(f"BB c5-t2 {sname}: prior term vs actual raw NF reward")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("log p (symlog)")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")

        finite = np.isfinite(z).all(axis=1)
        z0 = z[finite][0]
        zlast = z[finite][-1]
        print(f"{sname} n={finite.sum()} steps last_M={x[finite][-1]:.2f}")
        print(f"  first ||z||={np.linalg.norm(z0):.2f}  last ||z||={np.linalg.norm(zlast):.2f}")
        print("  last z per cube xyz:")
        for c in range(5):
            print(f"    cube{c}:", np.round(zlast[3 * c:3 * c + 3], 2),
                  "  g_task=", G_TASK[3 * c:3 * c + 3],
                  "  mean=", np.round(mean[finite][-1][3 * c:3 * c + 3], 3),
                  "  std=", np.round(std[finite][-1][3 * c:3 * c + 3], 3))
        print("  last raw NF reward mean:", float(raw[finite][-1]))

    fig.suptitle(
        "BuilderBench creative-5 task2 NF: the 15-D pyramid goal is z-scored "
        "by replay cube-xyz stats before the flow",
        fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = REPO / "figs" / "builderbench_c5t2_normalized_hard_goal.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
