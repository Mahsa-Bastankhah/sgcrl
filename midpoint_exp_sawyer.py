# ---------------------------------------------------------------------------
# Sawyer‑Bin:  find the state w* whose ψ(w*) is closest to a target rep
# ---------------------------------------------------------------------------
import itertools
import jax.numpy as jnp
import numpy as np
from lp_contrastive import fixed_goal_dict
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import trange
import jax
import functools
import tensorflow as tf
from exp_utils import load_checkpoint
from point_env import WALLS
from collections import defaultdict
import numpy as np
import os
from point_env import WALLS
import os
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA   # only for optional arrow plot
# add once near your other imports
import jax
import os, csv, math
import jax.numpy as jnp
from typing import Sequence
from typing import Tuple
from typing import Union      # add to the existing typing imports
from lp_contrastive import fixed_goal_dict
import jax.numpy as jnp
from functools import partial
import os
import csv
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math
# helpers -----------------------------------------------------------
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
os.environ["CUDA_VISIBLE_DEVICES"] = ""
Action_dim = 4

def _phi_state_action(params_q, nets, s, a):
    s = jnp.atleast_1d(s)
    zeros = jnp.zeros_like(s)
    obs   = jnp.concatenate([s, zeros], axis=-1)[None]
    _, sa_repr, _ = nets.q_network.apply(params_q, obs, a)
    return sa_repr[0]


def _psi_goal_external(params_q, nets, g):
    """ψ(g) from g_repr; state part is dummy."""
    g = jnp.atleast_1d(g)                        # ← guarantee rank‑1
    zeros = jnp.zeros_like(g)
    obs   = jnp.concatenate([zeros, g], axis=-1)[None]
    a = np.zeros((1, Action_dim), dtype=np.float32)
    _, _, g_repr = nets.q_network.apply(params_q, obs, a)
    return g_repr[0]


def closest_state_on_grid(rep: np.ndarray,
                          params_q,
                          nets,
                          *,
                          grid_axes: list[np.ndarray]):
    """
    Parameters
    ----------
    rep        : (D,) target representation.
    params_q   : q‑network params (from the learner state).
    nets       : namespace with q_network.
    grid_axes  : list of 1‑D numpy arrays, one per *state* dimension.

    Returns
    -------
    best_state : (dim,) numpy array.
    best_dist  : float  ‖ψ(w*)‑rep‖₂.
    """
    # Cartesian product of the axes -> all candidate states (N, dim)
    mesh = np.stack(np.meshgrid(*grid_axes, indexing='ij'), axis=-1)
    states = mesh.reshape(-1, mesh.shape[-1]).astype(np.float32)  # [N,dim]

    # ψ for each state
    psi_grid = jax.vmap(lambda s: _psi_goal_external(params_q, nets, s))(states)

    d2 = np.sum((np.array(psi_grid) - rep[None, :])**2, axis=1)
    best_i = int(np.argmin(d2))
    return states[best_i], float(np.sqrt(d2[best_i]))



def compute_and_save_interp_points(env_name: str,
                                    ckpt_num: int,
                                    seed_num: int,
                                    goal: Sequence[np.ndarray],
                                    alphas: Sequence[float],
                                    grid_bounds: dict[str, tuple[float,float]],
                                    n_per_axis: int = 10,
                                    save_root: str = "plots") -> str:
    """
    Computes interpolated points and saves to CSV.
    Returns the path to the saved CSV file.
    """

    # ---- load learner and env -------------------------------------------
    state, env, nets = load_checkpoint(
        alpha='0.1',
        misc_params='0.1_None',
        env_name=env_name,
        base_log_dir='./logs',
        seed=seed_num,
        fix_goals=True,
        ckpt_num=ckpt_num,
        ckpt_dir=f'/home/mahsa/sgcrl/logs/contrastive_cpc_{env_name}_{seed_num}/checkpoints/learner',
        goal=goal,
    )
    params_q = state.q_params

    # ---- initial state --------------------------------------------------
    timestep = env.reset()
    obs0 = timestep.observation
    s0 = np.asarray(obs0[:7], dtype=np.float32)

    # ---- goal prep ------------------------------------------------------
    g = np.concatenate([goal + np.array([0.0, 0.0, 0.03]), [0.4], goal])
    a_sg  = nets.policy_network.apply(state.policy_params,
                                      np.concatenate([s0, g])[None]).mode()
    psi_s = _phi_state_action(params_q, nets, s0, a_sg)
    psi_g = _psi_goal_external(params_q, nets, g)

    # ---- grid for search -----------------------------------------------
    grid_axes = [np.linspace(lo, hi, n_per_axis) for lo, hi in grid_bounds.values()]

    # ---- gather all points ---------------------------------------------
    rows = []
    def append_row(alpha, hand, obj, label):
        rows.append({
            "ckpt_num": ckpt_num,
            "seed_num": seed_num,
            "alpha": alpha,
            "hand_x": hand[0],
            "hand_y": hand[1],
            "hand_z": hand[2],
            "obj_x": obj[0],
            "obj_y": obj[1],
            "obj_z": obj[2],
            "label": label
        })

    # Start and Goal
    append_row(0.0, s0[:3], s0[4:7], "start")
    append_row(1.0, g[:3], g[4:7], "goal")

    # Interpolated Points
    for α in alphas:
        rep = (1-α) * psi_s + α * psi_g
        w_star, dist = closest_state_on_grid(rep, params_q, nets, grid_axes=grid_axes)
        append_row(α, w_star[:3], w_star[4:7], "interp")

    # ---- Save CSV ------------------------------------------------------
    out_dir = os.path.join(save_root, env_name, str(seed_num))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"hand_obj_ckpt{ckpt_num}.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"[compute_and_save_interp_points] saved CSV → {csv_path}")
    return csv_path


def plot_interpolations(csv_path: str):
    df = pd.read_csv(csv_path)
    fig_path = csv_path.replace(".csv", ".png")

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    Z0 = bounds["z"][0]          # lowest z bound – where helper lines start


    # Optional: adjust bounds if needed
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(*bounds["x"])   # X‑axis
    ax.set_ylim(*bounds["y"])   # Y‑axis
    ax.set_zlim(*bounds["z"])   # Z‑axis

    ckpt_num = df['ckpt_num'].iloc[0]
    env_name = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
    ax.set_title(f"{env_name}  checkpoint {ckpt_num}")

    # Plot
    palette = ["green", "blue", "orange"]
    size_start = 300
    size_goal = 50
    size_base = 250
    decay = 50

    interp_df = df[df["label"] == "interp"].sort_values("alpha")
    for i, row in interp_df.iterrows():
        col = palette[i % len(palette)]
        size = size_base - decay * i
        ax.scatter(row["hand_x"], row["hand_y"], row["hand_z"],
                   marker="x", c=col, s=size, alpha=0.6,
                   label=f"hand α={row['alpha']:.2f}")
        # ↓ add this helper line
        ax.plot([row["hand_x"], row["hand_x"]],
                [row["hand_y"], row["hand_y"]],
                [Z0,            row["hand_z"]],
                color=col, alpha=0.3, linewidth=1)
        ax.scatter(row["obj_x"], row["obj_y"], row["obj_z"],
                   marker="o", c=col, s=size, alpha=0.6,
                   label=f"obj α={row['alpha']:.2f}")
        # ↓ and another helper line (same colour)
        ax.plot([row["obj_x"], row["obj_x"]],
                [row["obj_y"], row["obj_y"]],
                [Z0,            row["obj_z"]],
                color=col, alpha=0.3, linewidth=1)

    # Start and goal
    for _, row in df[df["label"] == "start"].iterrows():
        ax.scatter(row["hand_x"], row["hand_y"], row["hand_z"],
                   marker="x", c="black", s=size_start, alpha=0.8, label="hand start")
        # ↓ add this helper line
        ax.plot([row["hand_x"], row["hand_x"]],
                [row["hand_y"], row["hand_y"]],
                [Z0,            row["hand_z"]],
                color=col, alpha=0.3, linewidth=1)
        ax.scatter(row["obj_x"], row["obj_y"], row["obj_z"],
                   marker="o", c="black", s=size_start, alpha=0.8, label="obj start")
        ax.plot([row["obj_x"], row["obj_x"]],
                [row["obj_y"], row["obj_y"]],
                [Z0,            row["obj_z"]],
                color=col, alpha=0.3, linewidth=1)
    for _, row in df[df["label"] == "goal"].iterrows():
        ax.scatter(row["hand_x"], row["hand_y"], row["hand_z"],
                   marker="x", c="red", s=size_goal, alpha=0.8, label="hand goal")
        # ↓ add this helper line
        ax.plot([row["hand_x"], row["hand_x"]],
                [row["hand_y"], row["hand_y"]],
                [Z0,            row["hand_z"]],
                color=col, alpha=0.3, linewidth=1)
        ax.scatter(row["obj_x"], row["obj_y"], row["obj_z"],
                   marker="o", c="red", s=size_goal, alpha=0.8, label="obj goal")
        ax.plot([row["obj_x"], row["obj_x"]],
                [row["obj_y"], row["obj_y"]],
                [Z0,            row["obj_z"]],
                color=col, alpha=0.3, linewidth=1)

    ax.legend(fontsize=7, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[plot_from_interp_csv] saved PNG → {fig_path}")

def compare_q_distances(env_name: str,
                              ckpt_nums: Sequence[int],
                              seed_num: int,
                              alphas: Sequence[float],
                              save_root: str = "plots"):
    """
    Uses the CSVs saved by `compute_and_save_interp_points`.

    For each checkpoint:
        • reloads the learner (to access Q‑network)
        • computes   d(s0,g)        = Q(g,g) – Q(s0,g)
        • computes   d(s0,wα)+d(wα,g)   for every α in `alphas`
    Saves:
        • PNG: plots/<env>/<seed>/q_distances.png
        • CSV: plots/<env>/<seed>/q_distances.csv
    """

    # ---------- colour palette (bright & distinct) -----------------------
    palette = ["red", "orange", "yellow", "green",
               "cyan", "blue", "purple", "magenta"]

    # ---------- holders --------------------------------------------------
    d_sg_vals      = []
    d_sum_by_alpha = {α: [] for α in alphas}
    csv_rows       = [["ckpt", "alpha", "d_sg", "d_sum"]]

    # ---------- iterate checkpoints -------------------------------------
    for ck in ckpt_nums:
        csv_path = os.path.join(
            save_root, env_name, str(seed_num),
            f"hand_obj_ckpt{ck}.csv"
        )
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        # ----- reload learner to get Q‑network ---------------------------
        state, env, nets = load_checkpoint(
            alpha='0.1',
            misc_params='0.1_None',
            env_name=env_name,
            base_log_dir='./logs',
            seed=seed_num,
            fix_goals=True,
            ckpt_num=ck,
            ckpt_dir=(f'/home/mahsa/sgcrl/logs/contrastive_cpc_{env_name}_{seed_num}'
                      f'/checkpoints/learner'),
            goal=None,
        )
        params_q = state.q_params
        policy   = lambda st, go: nets.policy_network.apply(
            state.policy_params, jnp.concatenate([st, go])[None]).mode()

        def q_val(st, act, go):
            obs = jnp.concatenate([st, go])[None]
            q, _, _ = nets.q_network.apply(params_q, obs, act)
            return float(q[0])

        # ----- start / goal ---------------------------------------------
        s_row = df[df["label"] == "start"].iloc[0]
        g_row = df[df["label"] == "goal" ].iloc[0]

        s0 = jnp.asarray([s_row["hand_x"], s_row["hand_y"], s_row["hand_z"],
                          0.4,
                          s_row["obj_x"],  s_row["obj_y"],  s_row["obj_z"]],
                         dtype=jnp.float32)
        g  = jnp.asarray([g_row["hand_x"], g_row["hand_y"], g_row["hand_z"],
                          0.4,
                          g_row["obj_x"],  g_row["obj_y"],  g_row["obj_z"]],
                         dtype=jnp.float32)

        a_sg = policy(s0, g)
        a_gg = policy(g , g)
        d_sg = q_val(g, a_gg, g) - q_val(s0, a_sg, g)
        d_sg_vals.append(d_sg)

        # ----- loop α ----------------------------------------------------
        for α in alphas:
            row = df[(df["label"] == "interp") & (np.isclose(df["alpha"], α))].iloc[0]
            w = jnp.asarray([row["hand_x"], row["hand_y"], row["hand_z"],
                             0.4,
                             row["obj_x"],  row["obj_y"],  row["obj_z"]],
                            dtype=jnp.float32)

            a_sw = policy(s0, w)
            a_wg = policy(w , g)
            a_ww = policy(w , w)

            d_s0w = q_val(w, a_ww, w) - q_val(s0, a_sw, w)
            d_wg  = q_val(g, a_gg, g) - q_val(w, a_wg, g)
            d_sum = d_s0w + d_wg

            d_sum_by_alpha[α].append(d_sum)
            csv_rows.append([ck, α, d_sg, d_sum])

    # ---------- save CSV -------------------------------------------------
    out_dir = os.path.join(save_root, env_name, str(seed_num))
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "q_distances.csv")
    with open(csv_out, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"[plot_q_distances_from_csv] data saved → {csv_out}")

    # ---------- plot -----------------------------------------------------
    fig_path = os.path.join(out_dir, "q_distances.png")
    plt.figure(figsize=(7, 4))
    plt.plot(ckpt_nums, d_sg_vals, "k--", marker="s", label="d(s0,g)")

    for k, α in enumerate(alphas):
        plt.plot(ckpt_nums,
                 d_sum_by_alpha[α],
                 color=palette[k % len(palette)],
                 marker="o",
                 label=f"d(s0,w_α)+d(w_α,g)  α={α:.2f}")

    plt.xlabel("Checkpoint")
    plt.ylabel("Distance (Q values)")
    plt.title(f"{env_name} – Q‑distance decomposition vs checkpoint")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[plot_q_distances_from_csv] figure saved → {fig_path}")

# define coarse bounds for Sawyer‑Bin workspace
# s0=[ 0.00615235  0.6001898   0.19430117  1.         -0.19185776  0.6628703
  #0.02999509]
#g [0.12 0.7  0.05 0.4  0.12 0.7  0.02]
# s0=[ 0.00615235  0.6001898   0.19430117  1.         -0.12103546  0.72738045
#  0.02999509]

bounds = dict(
    x      = (-0.30, 0.25),
    y      = (0.60, 0.90),
    z      = ( 0.01, 0.2),
    g      = ( 0.40, 0.40),   # gripper fixed at 0.4
    obj_x  = (-0.30, 0.25),
    obj_y  = (0.60, 0.90),
    obj_z  = ( 0.01, 0.2),
)
env_name = "sawyer_bin"
seed_num = 42
ckpt_nums = [1 , 2, 3,4 , 5, 6, 7, 8, 9, 10, 20, 30] # for more checkpoints
# ckpt_nums = [1 , 5, 10, 20]  # for more checkpoints
# [0.12 0.7  0.05 0.4  0.12 0.7  0.02]

goal = fixed_goal_dict[env_name]
# for ckpt_num in ckpt_nums:

    # compute_and_save_interp_points(env_name,
    #                                 ckpt_num,
    #                                 seed_num,
    #                                 goal,
    #                                 alphas = [0.3, 0.5, 0.7],
    #                                 grid_bounds = bounds)
    # plot_interpolations(
    #     os.path.join("plots", env_name, str(seed_num), f"hand_obj_ckpt{ckpt_num}.csv")
    # )


compare_q_distances(
    env_name  = env_name,
    ckpt_nums = ckpt_nums,
    seed_num  = seed_num,
    alphas    = [0.3, 0.5, 0.7],
    save_root = "plots",
)