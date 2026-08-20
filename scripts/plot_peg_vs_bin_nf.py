#!/usr/bin/env python3
"""Compare a FAILING sawyer-peg NF run against a SUCCESSFUL sawyer-bin NF run.

Both use the identical tiny RealNVP (sa2x128 r32 b4 w128, tau=0.85, crl10, min_std=1e-5,
extrew=1, no obs-norm). The peg run never reaches the goal; the bin run solves it. This
script overlays per-metric training curves (mean across the two seeds, min-max band) so we
can see *where* the two diverge.

Read-only over the run CSVs; writes a single PNG under figs/.

Eval success is smoothed (centered rolling mean, window=5) per the smooth-eval-plots rule,
reusing _plot_eval_smoothed / _rolling_mean from plot_builderbench_train_success1000.py.
All other (train) curves are shown raw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from plot_builderbench_train_success1000 import _rolling_mean, EVAL_SMOOTH_WINDOW  # noqa: E402

LOGS = REPO / "logs"
PEG_DIR = LOGS / "ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_norand_minstd1e5_evalvid"
BIN_DIR = LOGS / "ppo_bin_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_minstd1e5_extrew1_norand_noobs"

STEPS_PER_ITER = 1024  # num_envs(4) * rollout_length(256)

GROUPS = {
    "peg (fails)": {
        "color": "#d1495b",
        "seeds": [PEG_DIR / "ppo_sawyer_peg_0", PEG_DIR / "ppo_sawyer_peg_1"],
    },
    "bin (succeeds)": {
        "color": "#1f77b4",
        "seeds": [BIN_DIR / "ppo_sawyer_bin_0", BIN_DIR / "ppo_sawyer_bin_1"],
    },
}


def _load(group):
    """Return list of (learner_df, eval_df) per seed; eval gets a global_step column."""
    out = []
    for sdir in group["seeds"]:
        learn = pd.read_csv(sdir / "logs" / "learner" / "logs.csv")
        # Intrinsic NF reward AFTER return-normalization, in isolation (no external
        # bonus): raw NF reward divided by the running return-normalizer std. Shows
        # how much of the NF signal survives the sigma_R "volume knob".
        raw = pd.to_numeric(learn.get("reward_repr_raw_mean"), errors="coerce")
        sig = pd.to_numeric(learn.get("reward_return_norm_std"), errors="coerce")
        learn["norm_nf_reward"] = raw / sig.replace(0.0, np.nan)
        ev = pd.read_csv(sdir / "logs" / "eval" / "logs.csv")
        ev = ev.copy()
        ev["global_step"] = (ev["iteration"].astype(float) + 1.0) * STEPS_PER_ITER
        out.append((learn, ev))
    return out


def _mean_band(dfs, xcol, ycol, npts=1500):
    """Interpolate each seed onto a shared step grid; return grid, mean, lo, hi."""
    xmax = min(float(df[xcol].max()) for df in dfs)
    grid = np.linspace(0.0, xmax, npts)
    stacked = []
    for df in dfs:
        x = df[xcol].to_numpy(dtype=float)
        y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            stacked.append(np.full_like(grid, np.nan))
            continue
        order = np.argsort(x[m])
        stacked.append(np.interp(grid, x[m][order], y[m][order]))
    arr = np.vstack(stacked)
    return grid, np.nanmean(arr, 0), np.nanmin(arr, 0), np.nanmax(arr, 0)


# (title, source, ycol, yscale, smoothed, note)
PANELS = [
    ("Eval success", "eval", "success", "linear", True, "rolling mean w=5"),
    ("External sparse-bonus firing rate\nreward_env_mean", "learner", "reward_env_mean", "linear", False, ""),
    ("Eval final_dist (lower=better)", "eval", "final_dist", "linear", False, ""),
    ("Eval min_dist (lower=better)", "eval", "min_dist", "linear", False, ""),
    ("NF log p mean", "learner", "nf/log_p_mean", "linear", False, "density fit"),
    ("NF log p min (overfit tail)", "learner", "nf/log_p_min", "linear", False, ""),
    ("Raw NF reward mean\nreward_repr_raw_mean", "learner", "reward_repr_raw_mean", "symlog", False, ""),
    ("Raw NF reward std\nreward_repr_raw_std", "learner", "reward_repr_raw_std", "log", False, ""),
    ("Return-normalizer std\nreward_return_norm_std", "learner", "reward_return_norm_std", "log", False, "poisoned if huge"),
    ("Total normalized reward policy sees\nreward_repr_mean (incl. ext bonus)", "learner", "reward_repr_mean", "linear", False, ""),
    ("Normalized NF reward (intrinsic)\nraw / return-norm std", "learner", "norm_nf_reward", "symlog", False, "NF signal after sigma_R"),
    ("Advantage std (PPO signal)", "learner", "advantage_std", "log", False, ""),
    ("PPO clipfrac", "learner", "ppo/clipfrac", "linear", False, ""),
]


def main():
    data = {name: _load(g) for name, g in GROUPS.items()}

    ncol = 4
    nrow = 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(22, 12))
    axes = axes.ravel()

    for ax, (title, src, ycol, yscale, smoothed, note) in zip(axes, PANELS):
        for name, group in GROUPS.items():
            color = group["color"]
            dfs = [d[0] if src == "learner" else d[1] for d in data[name]]
            if any(ycol not in df.columns for df in dfs):
                continue
            grid, mean, lo, hi = _mean_band(dfs, "global_step", ycol)
            xg = grid / 1e6  # millions of env steps

            if smoothed:
                sm_mean = np.asarray(_rolling_mean(list(mean), EVAL_SMOOTH_WINDOW))
                sm_lo = np.asarray(_rolling_mean(list(lo), EVAL_SMOOTH_WINDOW))
                sm_hi = np.asarray(_rolling_mean(list(hi), EVAL_SMOOTH_WINDOW))
                ax.plot(xg, mean, color=color, alpha=0.28, lw=1.0)  # faint raw mean
                ax.fill_between(xg, sm_lo, sm_hi, color=color, alpha=0.15, lw=0)
                ax.plot(xg, sm_mean, color=color, lw=2.4, label=f"{name} (smoothed)")
            else:
                ax.fill_between(xg, lo, hi, color=color, alpha=0.15, lw=0)
                ax.plot(xg, mean, color=color, lw=2.0, label=name)

        ax.set_yscale(yscale)
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("env steps (M)", fontsize=9)
        if note:
            ax.text(0.98, 0.03, note, transform=ax.transAxes, fontsize=8,
                    ha="right", va="bottom", style="italic", color="0.35")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Tiny-NF sawyer: PEG (fails, red) vs BIN (succeeds, blue)  —  mean of 2 seeds, min-max band",
        fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = REPO / "figs" / "peg_vs_bin_nf_tiny_diagnostics.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
