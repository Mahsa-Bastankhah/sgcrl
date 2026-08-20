#!/usr/bin/env python3
"""Peg NF reward diagnostics for tanh / nonormg ablation runs.

One run at a time. Focus panels:
  - pre-tanh: nf/log_p_mean (density batch) + inverted atanh from rollout raw
  - post-tanh / raw NF reward: reward_repr_raw_mean
  - return-normalizer std
  - normalized reward the policy sees

  python scripts/plot_peg_nf_tanh_reward_diag.py \\
      --run logs/ppo_peg_nf_tiny_..._ent005_tanhr
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RUN = (
    "logs/ppo_peg_nf_tiny_sa2x128_r32_b4_w128_tau085_crl10_40m_"
    "extrew1_rand_minstd1e5_ent005_tanhr"
)


def _discover_seed(run_dir: Path) -> Path:
    seeds = sorted(
        sd for sd in run_dir.glob("ppo_*")
        if (sd / "logs" / "learner" / "logs.csv").exists()
    )
    if not seeds:
        raise SystemExit(f"no seed subdir with learner CSV under {run_dir}")
    if len(seeds) > 1:
        print(f"note: {len(seeds)} seeds found; plotting first: {seeds[0].name}")
    return seeds[0]


def _tanh_scale(seed_dir: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    cfg_path = seed_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        # flags may live under nested keys or flat; try common spots
        for key in ("nf_reward_tanh_scale", "ppo_nf_reward_tanh_scale"):
            if key in cfg:
                return float(cfg[key])
        flags = cfg.get("flags") or cfg.get("absl_flags") or {}
        if isinstance(flags, dict) and "nf_reward_tanh_scale" in flags:
            return float(flags["nf_reward_tanh_scale"])
    return 50.0  # job default


def _invert_tanh_mean(raw: np.ndarray, scale: float) -> np.ndarray:
    """Recover pre-tanh log_p from post-tanh reward mean (saturates near ±scale)."""
    eps = 1e-6
    r = np.clip(raw / scale, -1.0 + eps, 1.0 - eps)
    return scale * np.arctanh(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--tanh_scale", type=float, default=None,
                    help="override nf_reward_tanh_scale (default: from run_config or 50)")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default under figs/sawyer_peg/)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    seed_dir = _discover_seed(run_dir)
    scale = _tanh_scale(seed_dir, args.tanh_scale)

    df = pd.read_csv(seed_dir / "logs" / "learner" / "logs.csv")
    x = df["global_step"].to_numpy(dtype=float) / 1e6
    raw = pd.to_numeric(df["reward_repr_raw_mean"], errors="coerce").to_numpy(dtype=float)
    norm = pd.to_numeric(df["reward_repr_mean"], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(df["reward_return_norm_std"], errors="coerce").to_numpy(dtype=float)
    logp = pd.to_numeric(df["nf/log_p_mean"], errors="coerce").to_numpy(dtype=float)
    logp_min = pd.to_numeric(df["nf/log_p_min"], errors="coerce").to_numpy(dtype=float)
    pretanh_inv = _invert_tanh_mean(raw, scale)
    # what density mean would look like after the same tanh
    dens_post = scale * np.tanh(logp / scale)

    sat_frac = float(np.mean(np.abs(raw) >= 0.998 * scale))
    imin = int(np.nanargmin(raw))
    isig = int(np.nanargmax(sig))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.plot(x, logp, color="#264653", lw=1.5, label="nf/log_p_mean (density)")
    ax.plot(x, logp_min, color="#264653", lw=1.0, alpha=0.35, label="nf/log_p_min")
    ax.plot(x, pretanh_inv, color="#d1495b", lw=1.2, alpha=0.85,
            label=f"atanh invert of rollout raw (scale={scale:g})")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title("Pre-tanh log p\n(density batch vs inverted rollout reward)")
    ax.set_xlabel("env steps (M)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[0, 1]
    ax.plot(x, raw, color="#d1495b", lw=1.6, label="reward_repr_raw_mean (post-tanh)")
    ax.plot(x, dens_post, color="#264653", lw=1.2, alpha=0.7,
            label=f"{scale:g}*tanh(log_p_mean/{scale:g}) from density")
    ax.axhline(-scale, color="0.4", ls="--", lw=1.0, label=f"±{scale:g} floor/ceil")
    ax.axhline(scale, color="0.4", ls="--", lw=1.0)
    ax.axvline(x[imin], color="0.45", ls=":", lw=1.0,
               label=f"raw min @ {x[imin]:.2f}M")
    ax.set_title(f"Post-tanh / raw NF reward\n(|raw|≥{0.998*scale:.1f} on {sat_frac:.1%} of iters)")
    ax.set_xlabel("env steps (M)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1, 0]
    ax.plot(x, sig, color="#e29578", lw=1.6)
    ax.axvline(x[isig], color="0.45", ls="--", lw=1.0,
               label=f"σ max={sig[isig]:.0f} @ {x[isig]:.2f}M")
    ax.set_yscale("log")
    ax.set_title("Return-normalizer std\nreward_return_norm_std")
    ax.set_xlabel("env steps (M)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1, 1]
    ax.plot(x, norm, color="#2a9d8f", lw=1.6)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_title("Normalized reward policy sees\nreward_repr_mean (incl. ext bonus)")
    ax.set_xlabel("env steps (M)")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{run_dir.name} / {seed_dir.name}\n"
        f"NF reward tanh (scale={scale:g}): pre-tanh vs post-tanh vs normalizer",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
    else:
        out = REPO / "figs" / "sawyer_peg" / f"reward_diag__{run_dir.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print(
        f"summary: post-tanh raw min={np.nanmin(raw):.4g} max={np.nanmax(raw):.4g} "
        f"sat_frac={sat_frac:.3%}  σ_R max={np.nanmax(sig):.4g} last={sig[-1]:.4g}  "
        f"log_p_mean last={logp[-1]:.4g}"
    )


if __name__ == "__main__":
    main()
