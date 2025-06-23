#!/usr/bin/env python
"""
psi_stats.py  —  Batch-level ψ-similarity analysis.

Usage (example):
  python psi_stats.py \
      --env point_Spiral11x11 \
      --log_dir logs \
      --ckpt 10 \
      --seeds 11 12 13 14 \
      --success 1 1 0 1 \
      --projected   # <- include this flag for projected ψ

Dependencies:
  assumes get_psi_norms() and get_cell_projected_psi_norms()
  are import-able exactly as in your current repo layout.
"""
import argparse, os, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

# ---- import your helper functions ------------------------
from experiments.mahsa_utils import (
    get_psi_norms,
    get_cell_projected_psi_norms,
    get_point_map,
)


# ----------------------------------------------------------

def load_psi_similarity(env, log_dir, seed, ckpt, projected, metric):
    """Return ψ-similarity vector for one (env,seed,ckpt) run."""
    if projected:
        point_map = get_point_map(env) 
        goal_locations, psi_norms, critic_sf, critic_g, psi_sim, *_ = get_cell_projected_psi_norms(
            env_name=env,
            log_dir=log_dir,
            seed=seed,
            ckpt_num=ckpt,
            project=True,            # call expects True
            point_map=point_map,
            cell_size=2,          # cell size for point map
        )
    else:
        goal_locations, psi_norms, critic_sf, critic_g, psi_sim, _, _, _, _,_ , phi_psi_similarity, *_ = get_psi_norms(
            env_name=env,
            log_dir=log_dir,
            seed=seed,
            ckpt_num=ckpt,
            project=False,
        )
    if metric == "Q_value":
        return phi_psi_similarity
    elif metric == "psi_similarity":
        return psi_sim    # shape (N_samples,)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--ckpt", type=int, required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--success", type=int, nargs="+", required=True,
                   help="1 = successful rollout, 0 = unsuccessful")
    p.add_argument("--projected", action="store_true",
                   help="use projected ψ similarity")
    p.add_argument(
    "--metric",
    type=str,
    default="psi_similarity",
    choices=["psi_similarity" , "Q_value"],
)
    args = p.parse_args()

    assert len(args.seeds) == len(args.success), \
        "Provide one success flag per seed"

    means_succ, means_fail = [], []
    neg_counts_succ, neg_counts_fail   = [], []
    neg_means_succ,  neg_means_fail    = [], []

    # NEW -- non-negative ψ containers
    pos_means_succ, pos_means_fail     = [], []

    # --- 1. choose common bins between -1 and 1  ----------------------
    n_bins  = 40
    bins    = np.linspace(-1.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])   # for plotting curves

    # holders for per-seed histograms
    hists_succ, hists_fail = [], []




    print("Loading metric …")
    for seed, succ in tqdm(zip(args.seeds, args.success),
                           total=len(args.seeds)):
        metric = load_psi_similarity(
            args.env, args.log_dir, seed, args.ckpt, args.projected, args.metric
        )
        run_mean = np.mean(metric)
        if succ:
            means_succ.append(run_mean)
        else:
            means_fail.append(run_mean)
        
        neg_vals = metric[metric < 0]
        neg_count = len(neg_vals)
        neg_mean  = neg_vals.mean() if neg_count else 0.0

        # --- NEW: positive (≥0) mean ---
        pos_vals = metric[metric >= 0]
        pos_mean = pos_vals.mean() if len(pos_vals) else 0.0
        # --------------------------------

        if succ:
            neg_counts_succ.append(neg_count)
            neg_means_succ.append(neg_mean)
            # NEW
            pos_means_succ.append(pos_mean)
        else:
            neg_counts_fail.append(neg_count)
            neg_means_fail.append(neg_mean)
            # NEW
            pos_means_fail.append(pos_mean)



        counts, _ = np.histogram(metric, bins=bins, density=False)
        hist =counts / counts.sum()          # each bar is P(bin), sums to 1
        if succ:
            hists_succ.append(hist)
        else:
            hists_fail.append(hist)

    
    # --- 2. averaged histograms --------------------------------------
    mean_hist_s   = np.mean(hists_succ, axis=0) if hists_succ else np.zeros_like(centers)
    mean_hist_f   = np.mean(hists_fail, axis=0) if hists_fail else np.zeros_like(centers)



    # -------------------- Stats ----------------------------
    print("\nSummary:")
    print(f"  Successful runs   : {len(means_succ)}")
    print(f"    mean {args.metric}     : {np.mean(means_succ):.4f}" if means_succ else
          "    (none)")
    print(f"  Unsuccessful runs : {len(means_fail)}")
    print(f"    mean {args.metric}      : {np.mean(means_fail):.4f}" if means_fail else
          "    (none)")



    # --- 3. plot average histogram ------------------------------------------------------
    fig_hist, axh = plt.subplots(figsize=(7,4))

    # shaded average curves
    axh.fill_between(centers, mean_hist_s, color="mediumseagreen", alpha=0.3,
                    label="Success  (avg)")
    axh.fill_between(centers, mean_hist_f, color="salmon", alpha=0.3,
                    label="Fail     (avg)")

    # per-run lines (thin)
    for h in hists_succ:
        axh.plot(centers, h, color="forestgreen", alpha=0.8, lw=1)
    for h in hists_fail:
        axh.plot(centers, h, color="firebrick",  alpha=0.8, lw=1)

    axh.set_xlim(-1, 1)
    axh.set_xlabel(f"{args.metric} value")
    axh.set_ylabel("Probability density")
    axh.set_title(f"{args.env} · ckpt {args.ckpt} · histograms of {args.metric}\n"
                f"({ 'projected' if args.projected else 'R^64' })")
    axh.legend()
    axh.grid(ls=":", alpha=0.4)
    out_dir = os.path.join("experiments/plots", f"{args.env}_stats")
    fig_hist.tight_layout()
    out_hist = os.path.join(out_dir,
        f"26_{args.metric}_hist_{'proj' if args.projected else 'R64'}_ckpt{args.ckpt}.png")
    fig_hist.savefig(out_hist, dpi=300)
    print(f"Plot saved to {out_hist}")

if __name__ == "__main__":
    main()
