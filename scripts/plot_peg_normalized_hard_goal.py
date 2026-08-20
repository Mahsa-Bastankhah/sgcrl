#!/usr/bin/env python3
"""How out-of-distribution is the peg task goal, over training?

The NF reward evaluates log p(g_task | s, a) with the goal normalized by the
running replay-goal stats: z_i = (g_task_i - nf_goal_mean_i) / nf_goal_std_i.
The Gaussian base prior contributes -0.5 * ||z||^2 to log p, so a large ||z||
(a coordinate deep in the tail) directly explains the huge-negative NF reward.

g_task is derived analytically from SawyerPeg._get_obs (env_utils.py:968) with
the fixed norand goal _goal_pos = [-0.3, 0.6, 0.0]:
    ideal_hand = _goal_pos + [0.13, 0, 0.03] = [-0.17, 0.60, 0.03]
    gripper    = 0.40
    peg_head   = _goal_pos               = [-0.30, 0.60, 0.00]
    g_task = [ -0.17, 0.60, 0.03, 0.40, -0.30, 0.60, 0.00 ]

Uses logged nf/goal_mean_{0..6}, nf/goal_std_{0..6}. Read-only; writes one PNG.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PEG_DIR = REPO / "logs" / "ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_norand_minstd1e5_evalvid"
SEEDS = [("seed 0", PEG_DIR / "ppo_sawyer_peg_0"),
         ("seed 1", PEG_DIR / "ppo_sawyer_peg_1")]

G_TASK = np.array([-0.17, 0.60, 0.03, 0.40, -0.30, 0.60, 0.00], dtype=np.float64)
DIM_LABELS = ["hand_x", "hand_y", "hand_z", "gripper", "peg_x", "peg_y", "peg_z"]


def main():
    fig, axes = plt.subplots(len(SEEDS), 3, figsize=(19, 4 * len(SEEDS)), squeeze=False)

    for r, (sname, sdir) in enumerate(SEEDS):
        df = pd.read_csv(sdir / "logs" / "learner" / "logs.csv")
        x = df["global_step"].to_numpy(dtype=float) / 1e6
        mean = np.column_stack([pd.to_numeric(df[f"nf/goal_mean_{i}"], errors="coerce") for i in range(7)])
        std = np.column_stack([pd.to_numeric(df[f"nf/goal_std_{i}"], errors="coerce") for i in range(7)])
        z = (G_TASK[None, :] - mean) / std  # (T, 7) normalized hard-goal coords
        znorm = np.linalg.norm(z, axis=1)
        prior_logp = -0.5 * znorm ** 2  # Gaussian base-density contribution
        raw = pd.to_numeric(df["reward_repr_raw_mean"], errors="coerce").to_numpy(dtype=float)

        # Col 1: per-dim normalized hard-goal coordinate
        ax = axes[r][0]
        for i in range(7):
            ax.plot(x, z[:, i], lw=1.2, label=f"{i}:{DIM_LABELS[i]}")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"peg {sname}: normalized hard goal per dim\nz_i=(g_task-mean)/std")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("z (std units)")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=7, ncol=2, loc="best")

        # Col 2: ||z|| and the std of the OOD dims
        ax = axes[r][1]
        ax.plot(x, znorm, color="#d1495b", lw=1.6, label="||z|| (task goal)")
        ax.set_title(f"peg {sname}: ||z|| (how OOD the task goal is)")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("||z||")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")

        # Col 3: Gaussian-prior logp -0.5||z||^2 vs actual raw NF reward
        ax = axes[r][2]
        ax.plot(x, prior_logp, color="#5b8c5a", lw=1.5, label="-0.5||z||^2 (base prior)")
        ax.plot(x, raw, color="#d1495b", lw=1.0, alpha=0.8, label="reward_repr_raw_mean")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(f"peg {sname}: prior term vs actual raw NF reward")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("log p (symlog)")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Peg: the fixed task goal sits deep in the flow's tail (large ||z||) -> huge-negative NF reward",
        fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPO / "figs" / "peg_normalized_hard_goal.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    # quick numeric summary
    for sname, sdir in SEEDS:
        df = pd.read_csv(sdir / "logs" / "learner" / "logs.csv")
        mean = np.column_stack([pd.to_numeric(df[f"nf/goal_mean_{i}"], errors="coerce") for i in range(7)])
        std = np.column_stack([pd.to_numeric(df[f"nf/goal_std_{i}"], errors="coerce") for i in range(7)])
        z = (G_TASK[None, :] - mean) / std
        zlast = z[-1]
        print(f"peg {sname} last z per dim:", np.round(zlast, 2), " ||z||=", round(float(np.linalg.norm(zlast)), 2))


if __name__ == "__main__":
    main()
