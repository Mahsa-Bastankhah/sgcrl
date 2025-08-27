#!/usr/bin/env python
"""
Plot evaluator success_1000 vs evaluator_episodes
for two groups of runs (e.g. Q-max vs Actor).

Each seed’s curve uses a different colour (tab10 palette);
the group mean appears as a black dashed line.
"""

import argparse, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import EmptyDataError


# ------------------------------------------------------------------ #
# 1. Load a single run, return (steps, success) or None if unusable. #
# ------------------------------------------------------------------ #
def load_run(env: str, seed: int, root: str):
    csv = (
        pathlib.Path(root)
        / f"contrastive_cpc_{env}_{seed}"
        / "logs/evaluator/logs.csv"
    )
    if not csv.exists() or csv.stat().st_size == 0:
        print(f"[skip] {csv} is missing or empty")
        return None

    try:
        df = pd.read_csv(csv)
    except EmptyDataError:
        print(f"[skip] {csv} has no header/rows yet")
        return None

    df.columns = df.columns.str.strip()            # trim stray spaces/tabs
    if "evaluator_episodes" not in df.columns:
        print(f"[skip] {csv}: 'evaluator_episodes' column not found")
        return None

    steps = pd.to_numeric(df["evaluator_episodes"], errors="coerce").values
    succ  = pd.to_numeric(df["success_1000"],       errors="coerce").values
    return steps, succ


# ------------------------------------------------------------------ #
# 2. Plot helper for one colour group                                #
# ------------------------------------------------------------------ #
def lines_plus_mean(
    ax,
    runs,                       # list of (seed, steps, succ)
    label,
    lw_seed: float  = 1.0,
    lw_mean: float  = 3.0,
    alpha_seed: float = 0.65,
    mean_style: str = "--",
):
    """Draw every seed (distinct colour) + a black dashed mean curve."""
    if not runs:
        return

    palette = plt.get_cmap("tab20").colors   # 10 qualitative colours

    # ── 1. individual seed curves ──────────────────────────────────
    for idx, (seed, steps, succ) in enumerate(runs):
        col = palette[idx % len(palette)]
        ax.plot(
            steps,
            succ,
            color     = col,
            linewidth = lw_seed,
            alpha     = alpha_seed,
            label     = f"{label} seed {seed}",
        )

    # ── 2. mean curve on common x-axis ─────────────────────────────
    max_common = min(steps.max() for _, steps, _ in runs)
    x   = np.linspace(0, max_common, 200)
    mat = np.vstack([
        np.interp(x, steps, succ)           # ← list comprehension wrapped […]
        for _, steps, succ in runs
    ])
    mean = np.nanmean(mat, axis=0)

    ax.plot(
        x, mean,
        color     = "black",
        linewidth = lw_mean,
        linestyle = mean_style,
        label     = f"{label} (mean)",
    )


# ------------------------------------------------------------------ #
# 3. CLI                                                             #
# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env",          required=True)
    p.add_argument("--q_seeds",      nargs="+", type=int, default=[])
    p.add_argument("--actor_seeds",  nargs="+", type=int, default=[])
    p.add_argument("--log_root",     default="./logs")
    p.add_argument("--label_qmax",   default="Q-max")
    p.add_argument("--label_actor",  default="Actor")
    args = p.parse_args()

    # ────────────────────────── plotting ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"success_1000 vs evaluator_episodes • {args.env}")
    ax.set_xlabel("evaluator_episodes")
    ax.set_ylabel("success_1000")

    # ─── group 1: Q-max ────────────────────────────────────────────
    if not args.q_seeds:
        print("[warn] no Q-max seeds provided, skipping this group")
    else:
        q_runs = [
            (s, *r)
            for s in args.q_seeds
            if (r := load_run(args.env, s, args.log_root)) is not None
        ]
        lines_plus_mean(ax, q_runs, label=args.label_qmax)

    # ─── group 2: Actor ────────────────────────────────────────────
    a_runs = [
        (s, *r)
        for s in args.actor_seeds
        if (r := load_run(args.env, s, args.log_root)) is not None
    ]
    lines_plus_mean(ax, a_runs, label=args.label_actor)

    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = f"success_1000_{args.env}_{args.actor_seeds}.png"
    fig.savefig(out, dpi=120)
    print("saved →", out)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # silence NaN slices
    main()