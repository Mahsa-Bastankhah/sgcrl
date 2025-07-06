#!/usr/bin/env python
"""
pca_stats.py — batch PCA visualisation of φ(s,a), ψ(s) and ψ(g).

Example:
  python pca_stats.py \
      --env point_Spiral11x11 \
      --log_dir logs \
      --ckpts 10 11 12 \
      --seeds  31 32 33 \
      --action_mode actor_max \
      --num_episodes 10
"""
import argparse, os, sys, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------------------------------------
# import the helper you added previously (same file or dotted path)
# ----------------------------------------------------------------------
try:
    from experiments.mahsa_utils import run_pca_on_visited_cells
except ImportError as e:
    sys.exit(
        "❌  Could not import run_pca_on_visited_cells. "
        "Make sure it is on PYTHONPATH or adjust the import."
    )

# ----------------------------------------------------------------------
def ensure_lists_match(seeds, ckpts):
    """
    Convenience: allow either one ckpt for all seeds or one-to-one lists.
    """
    if len(ckpts) == 1 and len(seeds) > 1:
        ckpts = ckpts * len(seeds)
    if len(seeds) != len(ckpts):
        raise ValueError("Number of --seeds must equal number of --ckpts "
                         "(or pass a single ckpt to apply to all seeds).")
    return ckpts


def save_current_fig(out_path: str):
    """Grab the current Matplotlib figure & write it to disk (then close)."""
    fig = plt.gcf()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env",      required=True)
    ap.add_argument("--log_dir",  default="logs")
    ap.add_argument("--ckpts",    type=int, nargs="+", required=True)
    ap.add_argument("--seed",    type=int, required=True)
    ap.add_argument("--action_mode", default="actor_max",
                    choices=["actor_max", "actor_sample", "q_max"])
    ap.add_argument("--grid_width",   type=float, default=0.01)
    ap.add_argument("--num_episodes", type=int,   default=10)
    args = ap.parse_args()

    #ckpts = ensure_lists_match(args.seeds, args.ckpts)
    # seed_dir = os.path.join(out_dir, str(seed))
    # os.makedirs(seed_dir, exist_ok=True)

    # name the file by checkpoint only
    # fname = f"vis_ckpt{ckpt}.png"
    # save_current_fig(os.path.join(seed_dir, fname))
    out_dir = os.path.join("experiments", "plots", f"{args.env}_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)


    print("Generating PCA plots …")
    for ckpt in tqdm(list(args.ckpts)):
        print(f"  seed={args.seed}, ckpt={ckpt} ...")
        run_pca_on_visited_cells(
            env_name     = args.env,
            log_dir      = args.log_dir,
            seed         = args.seed,
            ckpt_num     = ckpt,
            action_mode  = args.action_mode,
            grid_width   = args.grid_width,
            num_episodes = args.num_episodes,
        )

        fname = f"vis_ckpt{ckpt}.png"
        save_current_fig(os.path.join(out_dir, fname))

    print(f"✓ All plots written to  {out_dir}/")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
