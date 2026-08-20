#!/usr/bin/env python3
"""Per-seed reward/normalizer trace for the failing sawyer-peg NF run.

Shows the causal chain for each peg seed separately:
  raw NF reward goes very negative  ->  return-normalizer std jumps and stays high
  ->  the normalized reward the policy sees is crushed toward zero.

2 rows (peg seed 0, seed 1) x 3 cols (raw NF reward, total normalized reward,
return-normalizer std). Read-only over the run CSVs; writes one PNG under figs/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN = "logs/ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_extrew1_norand_minstd1e5_evalvid"


def _discover_seeds(run_dir: Path):
    """Return [(label, seed_dir), ...] for every seed subdir with a learner CSV."""
    seeds = []
    for sd in sorted(run_dir.glob("ppo_*")):
        if (sd / "logs" / "learner" / "logs.csv").exists():
            # label = trailing seed index if present, else dir name
            label = f"seed {sd.name.split('_')[-1]}"
            seeds.append((label, sd))
    return seeds

# (col, title, yscale, color)
METRICS = [
    ("reward_repr_raw_mean", "Raw NF reward (pre-norm)\nreward_repr_raw_mean", "symlog", "#d1495b"),
    ("reward_repr_mean", "Normalized reward policy sees\nreward_repr_mean", "linear", "#2a9d8f"),
    ("reward_return_norm_std", "Return-normalizer std\nreward_return_norm_std", "log", "#e29578"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="run directory (relative to repo or absolute)")
    args = ap.parse_args()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    seeds = _discover_seeds(run_dir)
    if not seeds:
        raise SystemExit(f"no seed subdirs with learner CSV under {run_dir}")

    fig, axes = plt.subplots(len(seeds), len(METRICS),
                             figsize=(18, 4 * len(seeds)), squeeze=False)

    for r, (sname, sdir) in enumerate(seeds):
        df = pd.read_csv(sdir / "logs" / "learner" / "logs.csv")
        x = df["global_step"].to_numpy(dtype=float) / 1e6  # env steps (M)
        # step at which raw NF reward is most negative (the poisoning event)
        raw = pd.to_numeric(df["reward_repr_raw_mean"], errors="coerce").to_numpy(dtype=float)
        imin = int(np.nanargmin(raw))
        x_event = x[imin]

        for c, (col, title, yscale, color) in enumerate(METRICS):
            ax = axes[r][c]
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y, color=color, lw=1.4)
            ax.axvline(x_event, color="0.4", ls="--", lw=1.0,
                       label=f"raw min @ {x_event:.1f}M\n(={raw[imin]:.0f})")
            ax.set_yscale(yscale)
            if yscale == "symlog":
                ax.set_yscale("symlog", linthresh=10)
            ax.set_title(f"peg {sname}: {title}", fontsize=10)
            ax.set_xlabel("env steps (M)", fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"{run_dir.name} (per seed): raw reward -> return-normalizer std -> normalized reward",
        fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = REPO / "figs" / f"seed_reward_norm__{run_dir.name}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
