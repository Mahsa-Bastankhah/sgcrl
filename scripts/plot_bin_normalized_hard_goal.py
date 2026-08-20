#!/usr/bin/env python3
"""Normalized hard-goal coords for the successful tiny-NF bin run.

Same diagnostic as plot_peg_normalized_hard_goal.py: z_i = (g_task_i - μ_i)/σ_i
from logged nf/goal_mean_*, nf/goal_std_*.

g_task from SawyerBin._get_obs (env_utils.py:460) with norand
_goal = [0.12, 0.7, 0.02] (fixed_goal_dict['sawyer_bin']):
    ideal_hand = _goal + [0, 0, 0.03] = [0.12, 0.70, 0.05]
    gripper    = 0.0  (closed)
    object     = _goal                   = [0.12, 0.70, 0.02]
    g_task = [ 0.12, 0.70, 0.05, 0.00, 0.12, 0.70, 0.02 ]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
BIN_DIR = REPO / (
    "logs/ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_"
    "minstd1e5_extrew1_norand_noobs")
SEEDS = [("seed 0", BIN_DIR / "ppo_sawyer_bin_0"),
         ("seed 1", BIN_DIR / "ppo_sawyer_bin_1")]

G_TASK = np.array([0.12, 0.70, 0.05, 0.00, 0.12, 0.70, 0.02], dtype=np.float64)
DIM_LABELS = ["hand_x", "hand_y", "hand_z", "gripper", "obj_x", "obj_y", "obj_z"]


def main():
    fig, axes = plt.subplots(len(SEEDS), 3, figsize=(19, 4 * len(SEEDS)), squeeze=False)

    for r, (sname, sdir) in enumerate(SEEDS):
        df = pd.read_csv(sdir / "logs" / "learner" / "logs.csv")
        x = df["global_step"].to_numpy(dtype=float) / 1e6
        mean = np.column_stack([pd.to_numeric(df[f"nf/goal_mean_{i}"], errors="coerce") for i in range(7)])
        std = np.column_stack([pd.to_numeric(df[f"nf/goal_std_{i}"], errors="coerce") for i in range(7)])
        z = (G_TASK[None, :] - mean) / std
        znorm = np.linalg.norm(z, axis=1)
        prior_logp = -0.5 * znorm ** 2
        raw = pd.to_numeric(df["reward_repr_raw_mean"], errors="coerce").to_numpy(dtype=float)

        ax = axes[r][0]
        for i in range(7):
            ax.plot(x, z[:, i], lw=1.2, label=f"{i}:{DIM_LABELS[i]}")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"bin {sname}: normalized hard goal per dim\nz_i=(g_task-mean)/std")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("z (std units)")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=7, ncol=2, loc="best")

        ax = axes[r][1]
        ax.plot(x, znorm, color="#d1495b", lw=1.6, label="||z|| (task goal)")
        ax.set_title(f"bin {sname}: ||z|| (how OOD the task goal is)")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("||z||")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")

        ax = axes[r][2]
        ax.plot(x, prior_logp, color="#5b8c5a", lw=1.5, label="-0.5||z||^2 (base prior)")
        ax.plot(x, raw, color="#d1495b", lw=1.0, alpha=0.8, label="reward_repr_raw_mean")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_title(f"bin {sname}: prior term vs actual raw NF reward")
        ax.set_xlabel("env steps (M)"); ax.set_ylabel("log p (symlog)")
        ax.grid(True, alpha=0.25); ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Bin (successful tiny NF): normalized hard-goal vs raw NF reward",
        fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPO / "figs" / "bin_normalized_hard_goal.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)
    for sname, sdir in SEEDS:
        df = pd.read_csv(sdir / "logs" / "learner" / "logs.csv")
        mean = np.column_stack([pd.to_numeric(df[f"nf/goal_mean_{i}"], errors="coerce") for i in range(7)])
        std = np.column_stack([pd.to_numeric(df[f"nf/goal_std_{i}"], errors="coerce") for i in range(7)])
        z = (G_TASK[None, :] - mean) / std
        zlast = z[-1]
        z0 = z[np.isfinite(z).all(axis=1)][0]
        print(f"bin {sname} first z:", np.round(z0, 2), " ||z||=", round(float(np.linalg.norm(z0)), 2))
        print(f"bin {sname} last  z:", np.round(zlast, 2), " ||z||=", round(float(np.linalg.norm(zlast)), 2))
        print(f"bin {sname} last mean:", np.round(mean[-1], 4))
        print(f"bin {sname} last std: ", np.round(std[-1], 4))


if __name__ == "__main__":
    main()
