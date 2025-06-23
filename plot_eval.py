#!/usr/bin/env python
"""
Plot evaluator success_1000 vs actor_steps
for two groups (Q-max = blue, Actor = red).
"""

import argparse, pathlib, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pandas.errors import EmptyDataError
#python plot_eval.py     --env point_Wall11x11     --q_seeds 400 401 402 404 405 406 407     --actor_seeds 110 111 112 114 115 116 117     --log_root ./logs
# python plot_eval.py     --env point_Wall11x11     --q_seeds 400    --actor_seeds 34 800 801 802 803 804 805 806 807     --log_root ./logs

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

    # ── always use actor_steps ───────────────────────────────────
    df.columns = df.columns.str.strip()        # trim stray spaces/tabs
    if "actor_steps" not in df.columns:
        print(f"[skip] {csv}: 'actor_steps' column not found")
        return None

    steps = pd.to_numeric(df["actor_steps"], errors="coerce").values
    succ  = pd.to_numeric(df["success_1000"], errors="coerce").values
    return steps, succ



# ------------------------------------------------------------------ #
# 2. Plot helper for one colour group                                #
# ------------------------------------------------------------------ #
def plot_group(ax, runs, color, label, alpha=.25):
    if not runs:
        print(f"[warn] group '{label}' has 0 usable runs – skipped")
        return

    # 2-a individual faint lines
    for s, v in runs:
        ax.plot(s, v, color=color, alpha=alpha, linewidth=1)

    # 2-b mean ± std on common interval
    max_common = min(s.max() for s, _ in runs)
    x = np.linspace(0, max_common, 200)
    mat = np.vstack([np.interp(x, s, v) for s, v in runs])

    if np.isnan(mat).all():
        print(f"[warn] group '{label}' only NaNs – skipped")
        return

    mean, std = np.nanmean(mat, 0), np.nanstd(mat, 0)
    ax.plot(x, mean, color=color, linewidth=2.5, label=f"{label} (mean)")
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


# ------------------------------------------------------------------ #
# 3. CLI                                                             #
# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--q_seeds",     nargs="+", type=int, default=[])
    p.add_argument("--actor_seeds", nargs="+", type=int, default=[])
    p.add_argument("--log_root",    default="./logs")
    p.add_argument("--label_qmax",  default="Q-max")
    p.add_argument("--label_actor", default="Actor")
    args = p.parse_args()

    # ────────────────────────── PLOTTING ──────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"success_1000 vs actor_steps • {args.env}")
    ax.set_xlabel("actor_steps")
    ax.set_ylabel("success_1000")

    def lines_plus_mean(runs, color, label,
                        lw_seed=1.0, lw_mean=3.0, alpha_seed=0.65,
                        mean_style="--"):
        """Plot every seed (thin) and the group mean (thick, dashed)."""
        if not runs:        # nothing to draw
            return

        # individual curves
        for s, v in runs:
            ax.plot(s, v, color=color, linewidth=lw_seed, alpha=alpha_seed)

        # interpolate to common x-axis and compute mean
        max_common = min(s.max() for s, _ in runs)
        x = np.linspace(0, max_common, 200)
        y = np.vstack([np.interp(x, s, v) for s, v in runs])
        mean = np.nanmean(y, axis=0)

        ax.plot(x, mean, color=color, linewidth=lw_mean,
                linestyle=mean_style, label=f"{label} (mean)")

    # ─── Blue: Q-max ───────────────────────────────────────────────
    if args.q_seeds == []:
        print("[warn] no Q-max seeds provided, skipping this group")
    else:
        q_runs = [r for s in args.q_seeds
                if (r := load_run(args.env, s, args.log_root)) is not None]
        lines_plus_mean(q_runs, color="tab:blue", label=args.label_qmax)

    # ─── Red: Actor ────────────────────────────────────────────────
    a_runs = [r for s in args.actor_seeds
            if (r := load_run(args.env, s, args.log_root)) is not None]
    lines_plus_mean(a_runs, color="tab:red", label=args.label_actor)

    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    out = f"success_1000_{args.env}_{args.actor_seeds}.png"
    fig.savefig(out, dpi=120)
    print("saved →", out)


if __name__ == "__main__":
    # Silence NumPy “mean of empty slice” if they ever sneak through
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
