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

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import math
# helpers -----------------------------------------------------------
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


# Hide all CUDA devices from every library
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Force JAX to use CPU
import jax
jax.config.update("jax_platform_name", "cpu")
# ‑‑‑ adjustable constants ----------------------------------------------------
RESOLUTION = 1.0                     # 1.0 ⇒ one grid‑cell per integer (same as _discretize_state)
ACTIONS = jnp.stack([
    jnp.array([-0.1,  0.], dtype=jnp.float32),   # up
    jnp.array([ 0.1,  0.], dtype=jnp.float32),   # down
    jnp.array([ 0.,  0.1], dtype=jnp.float32),   # right
    jnp.array([ 0., -0.1], dtype=jnp.float32),   # left
    jnp.array([ -0.1, -0.1], dtype=jnp.float32), 
    jnp.array([ -0.1, 0.1], dtype=jnp.float32),   
    jnp.array([ 0.1, -0.1], dtype=jnp.float32), 
    jnp.array([ 0.1, 0.1], dtype=jnp.float32), 
])



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
    _, _, g_repr = nets.q_network.apply(params_q, obs, jnp.zeros_like(g)[None])
    return g_repr[0]

def compare_q_distances(env_name: str,
                        ckpt_nums: Sequence[int],
                        seed_num: int,
                        goal,
                        alphas: Sequence[float]):
    """
    Same interface as before, but:
      • plots log‑distances  log(max(d,1))
      • writes CSV with columns:
            ckpt, alpha, w_i, w_j, d_sg, d_grid_avg, d_sum_alpha
    """

    # ─────────────── helper for ONE checkpoint ────────────────
    def _metrics_for_ckpt(ckpt_num: int):
        trained_state, env, nets = load_checkpoint(
            alpha='0.1',
            misc_params='0.1_None',
            env_name=env_name,
            base_log_dir='./logs',
            seed=seed_num,
            fix_goals=True,
            ckpt_num=ckpt_num,
            ckpt_dir=(f'/home/mahsa/sgcrl/logs/contrastive_cpc_{env_name}_{seed_num}'
                      f'/checkpoints/learner'),
            goal=goal,
        )
        params_q = trained_state.q_params
        policy = lambda st, go: nets.policy_network.apply(
            trained_state.policy_params,
            jnp.concatenate([st, go])[None]).mode()

        def q_val(st, act, go):
            obs = jnp.concatenate([st, go])[None]
            q, _, _ = nets.q_network.apply(params_q, obs, act)
            return float(q[0])

        # start / goal
        py_s0, py_g = goal
        s0 = jnp.asarray(py_s0, dtype=jnp.float32)
        g  = jnp.asarray(py_g , dtype=jnp.float32)

        a_sg = policy(s0, g)
        a_gg = policy(g , g)

        # baseline d(s,g)
        d_sg = -q_val(s0, a_sg, g) + q_val(g, a_gg, g)

        # ψ grid pre‑compute
        maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
        H, W = WALLS[maze_key].shape
        idxs = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'),
                         axis=-1).reshape(-1, 2)
        psi_grid = jax.vmap(
            lambda idx: _psi_goal_external(params_q, nets,
                                           jnp.array([idx[0], idx[1]], jnp.float32)))(idxs)
        idxs_np, psi_grid_np = np.array(idxs), np.array(psi_grid)

        # grid‑average d(s,w)+d(w,g)
        q_gg = q_val(g, a_gg, g)
        d_sum_vals = []
        for w_idx in idxs_np:
            w = jnp.asarray(w_idx, dtype=jnp.float32)
            a_sw = policy(s0, w)
            a_wg = policy(w , g)
            a_ww = policy(w , w)
            d_sum_vals.append(
                q_val(w, a_wg, w)
              - q_val(s0, a_sg, w)
              - q_val(w , a_wg, g)
              + q_gg
            )
        d_grid_avg = float(np.mean(d_sum_vals))

        # representations for α interpolation
        psi_s = _phi_state_action(params_q, nets, s0, a_sg)
        psi_g = _psi_goal_external(params_q, nets, g)
        print("norm 2 of phi(s)", np.linalg.norm(psi_s))
        print("norm 2 of psi(g)", np.linalg.norm(psi_g))

        def d_sum_and_w(alpha: float):
            rep = (1-alpha) * psi_s + alpha * psi_g
            best_i = np.argmin(np.sum((psi_grid_np - rep[None, :])**2, axis=1))
            w_idx  = idxs_np[best_i]                      # (i,j)
            w      = jnp.asarray(w_idx, dtype=jnp.float32)

            a_sw = policy(s0, w)
            a_wg = policy(w , g)
            a_ww = policy(w , w)
            d_sum = (
                q_val(w, a_wg, w)
              - q_val(s0, a_sg, w)
              - q_val(w , a_wg, g)
              + q_gg
            )
            return w_idx, d_sum

        return d_sg, d_grid_avg, d_sum_and_w

    # ───────────── gather & store results ──────────────
    rows_for_csv = []                      # each row: [ckpt, α, w_i, w_j, d_sg, d_avg, d_sum]
    d_sg_plot, d_avg_plot = [], []
    dsum_plot_by_alpha = {α: [] for α in alphas}

    for ck in ckpt_nums:
        d_sg, d_avg, d_sum_fn = _metrics_for_ckpt(ck)
        d_sg_plot.append(d_sg)
        d_avg_plot.append(d_avg)

        for α in alphas:
            w_idx, d_sum = d_sum_fn(α)
            dsum_plot_by_alpha[α].append(d_sum)

            rows_for_csv.append([
                ck, α, int(w_idx[0]), int(w_idx[1]),
                d_sg, d_avg, d_sum
            ])

    # ───────────── save CSV ──────────────
    goal_str = "_".join(str(int(x)) for x in goal[1]) if isinstance(goal, Sequence) else "goal"
    out_dir  = f"plots/{env_name}/{seed_num}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "distances_ckpts_alphas.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ckpt", "alpha", "w_i", "w_j",
                         "d_sg", "d_grid_avg", "d_sum_alpha"])
        writer.writerows(rows_for_csv)
    print(f"[compare_q_distances] CSV saved → {csv_path}")



    plt.figure(figsize=(8,4))
    plt.plot(ckpt_nums,
             [x for x in d_sg_plot],
             marker='^', linestyle='--', linewidth=2,
             label=r"$d(s,g)$")

    plt.plot(ckpt_nums,
             [x for x in d_avg_plot],
             marker='v', linestyle='--', linewidth=2,
             label=r"$\overline{d(s,w)+d(w,g)}$")

    for α in alphas:
        plt.plot(ckpt_nums,
                 [x for x in dsum_plot_by_alpha[α]],
                 marker='o',
                 label=rf"$d(s,w_α)+d(w_α,g)$  (α={α})")

    plt.xlabel("Checkpoint")
    plt.ylabel(r"$\log$ Distance (clipped)")
    plt.title(f"Distance metrics vs. checkpoint ({env_name})")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(out_dir, "distances_ckpts_alphas.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"[compare_q_distances] Plot saved → {png_path}")



def plot_psi_interpolation(env_name: str,
                                  trained_state,
                                  nets,
                                  goal: Sequence[np.ndarray],
                                  n_steps: int = 10,
                                  save_path: str = "closest_states.png"):
    """
    On the discrete maze grid, for each of n_steps interpolants between ψ(s₀)→ψ(g),
    find the grid cell whose ψ(w) is closest in L2, and plot those cells with
    a blue→red gradient.
    """
    # 1. Maze geometry
    maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
    walls = WALLS[maze_key]
    H, W   = walls.shape

    # 2. Unpack & cast s₀ and g
    py_s0, py_g = goal
    s0 = jnp.asarray(py_s0, dtype=jnp.float32)
    g  = jnp.asarray(py_g,  dtype=jnp.float32)

    # 3. Compute ψ(s₀), ψ(g)
    psi_s0 = _psi_goal_external(trained_state.q_params, nets, s0)
    @jax.jit
    def select_action(params, obs):
        return nets.policy_network.apply(params.policy_params, obs).mode()
    a_0 = select_action(trained_state, obs=jnp.concatenate([s0, g])[None])
    psi_s0 = _phi_state_action(trained_state.q_params, nets, s0, a_0)  # ψ(s₀)
    psi_g  = _psi_goal_external(trained_state.q_params, nets, g)

    # 4. Pre‑compute ψ(w) at every grid cell, shape [H*W, D]
    idxs = jnp.stack(
        jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'),
        axis=-1).reshape(-1, 2)  # [H*W,2]
    psi_grid = jax.vmap(
        lambda idx: _psi_goal_external(trained_state.q_params, nets,
                                       jnp.array([idx[0], idx[1]], jnp.float32)))(idxs)  # → [H*W, D]

    # 5. Build the n_steps interpolations
    alphas = np.linspace(0.0, 1.0, n_steps)
    
    reps   = np.stack([np.array((1-a)*psi_s0 + a*psi_g) for a in alphas])  # [n_steps, D]

    # 6. For each rep, find the closest w in L2 norm
    psi_grid_np = np.array(psi_grid)           # [H*W, D]
    idxs_np     = np.array(idxs)               # [H*W, 2]
    chosen      = []
    for rep in reps:
        # compute squared distances to every ψ(w)
        d2 = np.sum((psi_grid_np - rep[None, :])**2, axis=1)
        best = np.argmin(d2)
        chosen.append(idxs_np[best])           # (i, j)

    # 7. Plot
    wall_img = np.ones_like(walls, dtype=float)
    wall_img[walls==1] = 0.5
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(wall_img, origin="upper", cmap="gray", extent=[0,W,0,H])

    # Scatter the chosen cells
    chosen = np.array(chosen)  # [n_steps,2]
    xs = chosen[:,1] + 0.5
    ys = H - (chosen[:,0] + 0.5)
    sc = ax.scatter(xs, ys,
                    c=alphas, cmap="coolwarm",
                    s=120, edgecolors="black", zorder=3)

    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_title("Closest ψ‑states along interpolation s₀→g")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("α (0=s₀ → 1=g)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved closest‑states plot to: {save_path}")
    plt.close()



def run_interp_and_closest_states(env_name: str,
                                  ckpt_num: int,
                                  seed_num: int,
                                  goal: Sequence[np.ndarray],
                                  save_root: str = "plots"):
    """
    1) Loads trained_state, env, nets via load_checkpoint
    2) Calls plot_closest_states_along_psi
    3) (Optionally) Calls plot_psi_interpolation or any other helper
    """
    # 1) Load checkpoint
    trained_state, env, nets = load_checkpoint(
        alpha='0.1',
        misc_params='0.1_None',
        env_name=env_name,
        base_log_dir='./logs',
        seed=seed_num,
        fix_goals=True,
        ckpt_num=ckpt_num,
        ckpt_dir=f'/home/mahsa/sgcrl/logs/'
                 f'contrastive_cpc_{env_name}_{seed_num}/'
                 f'checkpoints/learner',
        goal=goal,
    )

    # Prepare output directory
    out_dir = os.path.join(save_root, env_name, str(seed_num))
    os.makedirs(out_dir, exist_ok=True)

    # 2) Plot closest states along ψ‐interpolation
    closest_path = os.path.join(out_dir, f"closest_states_ckpt{ckpt_num}_phi.png")
    plot_psi_interpolation(
        env_name=env_name,
        trained_state=trained_state,
        nets=nets,
        goal=goal,
        n_steps=10,
        save_path=closest_path,
    )

    

    print(f"Done: figures saved to {out_dir}")






def posterior_grid(trained_learner_state,
                   nets,
                   goal,
                   env_name: str,
                   resolution: float = RESOLUTION):
    """
    Returns an H×W array where each entry i,j =
      exp[ φ(s0, a₁) · ψ(w) + φ(w, a₂) · ψ(g) - φ(s0, a₀)·ψ(g) ]
    where:
      a₀ = argmax_a Q(s0, a, g)
      a₁ = argmax_a Q(s0, a, w)
      a₂ = argmax_a Q(w, a, g)
    """
    maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
    H, W     = WALLS[maze_key].shape

    params_q = trained_learner_state.q_params

    # unpack & cast start & goal
    py_s0, py_g = goal
    s0 = jnp.asarray(py_s0, dtype=jnp.float32)
    g  = jnp.asarray(py_g, dtype=jnp.float32)

    # pre‑compute ψ(g)
    psi_g = _psi_goal_external(params_q, nets, g)

    # 1) find a₀ = argmax_a Q(s0,a,g)
    # a0 = maximize_q_action(
    #     q_network=nets.q_network,
    #     q_params=params_q,
    #     obs=jnp.concatenate([s0, g])[None]  # note: obs must include goal
    # )[0]  # shape (action_dim,)

    # # 2) compute baseline term φ(s0,a₀)·ψ(g)
    # phi_s0_a0 = _phi_state_action(params_q, nets, s0, a0)
    # denom_term = jnp.dot(phi_s0_a0, psi_g)

    idxs = jnp.stack(
        jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'),
        axis=-1).reshape(-1, 2)

    def _score_one(idx):
        i, j = idx
        w = jnp.array([i, j], dtype=jnp.float32) / resolution


        @jax.jit
        def select_action(params, obs):
            return nets.policy_network.apply(params.policy_params, obs).mode()
        # a₁ = argmax_a Q(s0, a, w)
        # a1 = maximize_q_action(
        #     q_network=nets.q_network,
        #     q_params=params_q,
        #     obs=jnp.concatenate([s0, g])[None]
        # )[0]

        a1 = select_action(trained_learner_state, obs=jnp.concatenate([s0, w])[None])





        # term1 = φ(s0, a₁)·ψ(w)
        psi_w = _psi_goal_external(params_q, nets, w)
        phi_s0_a1 = _phi_state_action(params_q, nets, s0, a1)
        term1 = jnp.dot(phi_s0_a1, psi_w)

        # a₂ = argmax_a Q(w, a, g)
        
        # a2 = maximize_q_action(
        #     q_network=nets.q_network,
        #     q_params=params_q,
        #     obs=jnp.concatenate([w, g])[None]
        # )[0]
        a2 = select_action(trained_learner_state, obs=jnp.concatenate([w, g])[None])




        # term2 = φ(w, a₂)·ψ(g)
        phi_w_a2 = _phi_state_action(params_q, nets, w, a2)
        term2 = jnp.dot(phi_w_a2, psi_g)
        # scale = 0.0001
        scale = 1
        # final score
        score = jnp.exp( scale * (term1 + term2))

        # optional debug
        jax.debug.print("w=({},{}): term1={}, term2={}, score={}",
                        i, j, term1, term2, score, ordered=True)
        return score

    scores = jax.vmap(_score_one)(idxs)       # [H*W]
    return np.asarray(scores.reshape(H, W))


def plot_heatmap_on_maze(env_name: str,
                         heatmap: np.ndarray,
                         save_path: str = "heatmap.png",
                         cmap: str = "inferno"):
    """
    Overlay a heat‑map on the maze, and draw a red border around each wall cell.
    """
    maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
    walls = WALLS[maze_key]
    H, W = walls.shape

    # Background: white floor, gray walls
    wall_img = np.ones_like(walls, dtype=float)
    # wall_img[walls == 1] = 0.5

    fig, ax = plt.subplots(figsize=(6, 6))
    #ax.imshow(wall_img, origin="upper", cmap="gray", extent=[0, W, 0, H])

    # Heatmap over all cells
    im = ax.imshow(
        heatmap,
        origin="upper",
        cmap=cmap,
        extent=[0, W, 0, H],
        alpha=0.9,
        interpolation="nearest",
    )

        # mask out infinities for scaling
    # finite_mask = np.isfinite(heatmap)
    # if finite_mask.any():
    #     vmin, vmax = heatmap[finite_mask].min(), heatmap[finite_mask].max()
    # else:
    #     vmin, vmax = 0.0, 1.0
    # im = ax.imshow(
    #     heatmap,
    #     origin="upper",
    #     cmap=cmap,
    #     extent=[0, W, 0, H],
    #     alpha=0.9,
    #     interpolation="nearest",
    #     vmin=vmin, vmax=vmax,
    # )

    # # overlay stars on any infinities
    # inf_idxs = np.argwhere(np.isinf(heatmap))
    # for i, j in inf_idxs:
    #     # convert grid (i,j) to plot coords
    #     x = j + 0.5
    #     y = H - (i + 0.5)
    #     ax.scatter([x], [y],
    #                marker="*", c="yellow", edgecolors="black",
    #                s=120, zorder=3, label="_nolegend_")

    # Draw red border around each wall cell
    for (i, j), val in np.ndenumerate(walls):
        if val == 1:
            # Rectangle: lower-left corner at (j, H-i-1), width=1, height=1
            ax.add_patch(
                Rectangle(
                    (j, H - i - 1),
                    1, 1,
                    fill=False,
                    edgecolor="red",
                    linewidth=2
                )
            )

    # Cosmetics
    ax.set_title(f"Posterior distribution {maze_key}")
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_aspect("equal")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved heat‑map to: {save_path}")
    plt.close()

def posterior_heatmap_for_checkpoint(env_name: str,
                           ckpt_num: int,
                           seed_num: int,
                           goal: np.ndarray,
                           save_root: str = "plots",
                           resolution: float = 1.0,
                           cmap: str = "inferno"):
    """
    • Loads the checkpoint (via your existing load_checkpoint)
    • Builds the exp[φᵀψ] grid for action “right”
    • Saves a heat‑map PNG next to your trajectory plots
    """
    # ---- 1. load learner state, env & networks ------------------------------
    trained_state, env, nets = load_checkpoint(
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

        # ---- 1.5. load & average prior marginals over checkpoints up to ckpt_num ---
    import glob, os
    data_dir = f"data/{env_name}/{seed_num}/"
    files = sorted(glob.glob(os.path.join(data_dir, "ckpt_*.npz")))
    sel = [f for f in files if int(os.path.basename(f).split('_')[1]) <= ckpt_num]
    mats = [np.load(f)['state_prior_dist'] for f in sel]
    avg_prior = np.mean(mats, axis=0) if mats else None

    # ---- 2. make the score grid --------------------------------------------
    grid = posterior_grid(
        trained_learner_state=trained_state,
        nets=nets,
        goal=goal,
        env_name=env_name,
        resolution=resolution,
    )
    maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
    walls = WALLS[maze_key]
    H, W = walls.shape
    print(avg_prior)

        # ---- 2.5. Apply the prior weighting -------------------------------------
    if avg_prior is not None:
        print("================Applying prior weighting================")
        grid = grid.copy()
        #grid *= avg_prior
        grid = np.where(np.isnan(grid), 0.0, grid)
    # --- Debug: scan wall cells in the output ---
    for i in range(H):
        for j in range(W):
            if walls[i, j] == 1 and grid[i, j] != 0:
                print(f"[BUG] wall cell ({i},{j}) has GRID score {grid[i,j]:.4g}")
            if walls[i, j] == 1 and avg_prior[i, j] != 0:
                print(f"[BUG] wall cell ({i},{j}) has PRIOR score {avg_prior[i,j]:.4g}")

    # ---- 3. plot & save -----------------------------------------------------
    goal_str = "_".join([f"{int(g)}" for g in (goal[1] if isinstance(goal, Sequence) else goal)])
    heat_path = os.path.join(
        save_root, env_name, str(seed_num),
        f"no_prior_heatmap_ckpt_{ckpt_num}_{goal_str}.png"
    )
    os.makedirs(os.path.dirname(heat_path), exist_ok=True)
    plot_heatmap_on_maze(env_name, grid, save_path=heat_path, cmap=cmap)




def run_episode(trained_learner_state, env, networks):
    positions = []
    actions = []
    rewards = []

    timestep = env.reset()
    print("timestep ", timestep)

    @jax.jit
    def select_action(params, obs):
        return networks.policy_network.apply(params.policy_params, obs).mode()
        # return maximize_q_action(
        #     q_network=networks.q_network,
        #     q_params=params.q_params,
        #     obs=obs,
        # )[0]
    
    i = 0
    while not timestep.last():
        obs = timestep.observation
        action = select_action(trained_learner_state, obs)
        
        positions.append(obs[:2]) 
        print("new step:" , i)
        print("state", obs[:2])
        print("action", action)         
        i = i + 1       
        # Assuming obs[:2] = (x, y)
        actions.append(action)
        rewards.append(timestep.reward)

        timestep = env.step(np.array(action))

    return {
        "positions": np.array(positions),
        "actions": np.array(actions),
        "rewards": np.array(rewards)
    }


def generate_and_save_episodes(env_name,
                               ckpt_num,
                               seed_num=42,
                               num_episodes=10,
                               plot=True,
                               goal=None):
    """
    Run episodes, track state visit frequencies (discretely over first two obs dims),
    propagate visits to neighbors, and save trajectories + marginal state distribution.
    """
    # Load trained agent
    trained_learner_state, env, networks = load_checkpoint(
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

    all_positions, all_actions, all_rewards = [], [], []

    # Get maze size from WALLS
    maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
    H, W = WALLS[maze_key].shape

    # Initialize the full state space marginal distribution
    marg_dist = np.zeros((H, W), dtype=np.float32)
    true_dist = np.zeros((H, W), dtype=np.float32)

    total_visits = 0
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (-1,1), (1,-1), (-1, -1)]

    save_dir = f"data/{env_name}/{seed_num}/"
    os.makedirs(save_dir, exist_ok=True)

    # Run episodes
    for ep in trange(num_episodes, desc="Running episodes"):
        data = run_episode(trained_learner_state, env, networks)
        positions = np.array(data['positions'])  # shape (T, obs_dim)
        actions = np.array(data['actions'])
        rewards = np.array(data['rewards'])

        all_positions.append(positions)
        all_actions.append(actions)
        all_rewards.append(rewards)

        # Update state counts and propagate to neighbors
        for obs in positions:
            x, y = min(np.floor(obs[0]).astype(int) , H - 1), min(np.floor(obs[1]).astype(int), W - 1)
            # Check if it's inside the wall
            # if not (0 <= x < H and 0 <= y < W):
            #     print(f"[ERROR] State out of bounds: (x={x}, y={y}) - Skipping update.")
            if WALLS[maze_key][x, y] == 1:
                print(f"[ERROR] Agent detected in a wall! is ", WALLS[maze_key][x, y] == 1)
                x, y = min(np.floor(obs[0]-0.01).astype(int) , H - 1), min(np.floor(obs[1]-0.01).astype(int), W - 1)
                
                # if WALLS[maze_key][x, y] == 1:
                #     raise ValueError(f"Agent detected in a wall! is {WALLS[maze_key][x, y] == 1}")
           
            if 0 <= x < H and 0 <= y < W:
                # Mark the visited state
                marg_dist[x, y] += 1
                total_visits += 1
                true_dist[x,y] +=1

                # Propagate to neighbors
                for dx, dy in neighbor_offsets:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and WALLS[maze_key][nx, ny] == 0:
                        # Only set to 1 if it was previously zero
                        if marg_dist[nx, ny] == 0:
                            marg_dist[nx, ny] = 1
                            total_visits += 1

        # Optional plotting
        if plot:
            goal_str = "_".join(str(int(g)) for g in (goal[1] if goal is not None else []))
            plot_path = f"plots/{env_name}/{seed_num}/trajectory_ckpt_{ckpt_num}_{ep}_{goal_str}.png"
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plot_trajectory_on_maze_with_time(env_name, positions, save_path=plot_path)

    # Normalize the distribution
    if total_visits > 0:
        marg_dist /= total_visits
    print(marg_dist)
    print("true dist", true_dist)

    # Save trajectories and marginal distribution
    out_path = os.path.join(save_dir, f"ckpt_{ckpt_num}_{goal_str}.npz")
    np.savez_compressed(
        out_path,
        positions=np.array(all_positions, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object),
        state_prior_dist=marg_dist
    )
    print(f"Saved {num_episodes} episodes and marginal state distribution to {out_path}")
def plot_trajectory_on_maze_with_time(env_name='Spiral11x11', positions=None, save_path='agent_trajectory.png'):
    """
    Plots agent trajectory, colors by time, and marks 'jumps' with a star if consecutive movement is too large.
    
    Args:
        jump_threshold: distance (in grid units) considered a "jump" (default 2.0 units).
    """
    if "stochastic" in env_name:
        maze_name = env_name.split("_")[2]
    else:   
        maze_name = env_name.split("_")[1]
    walls = WALLS[maze_name]
    H, W = walls.shape

    wall_img = np.ones_like(walls, dtype=float)
    wall_img[walls == 1] = 0.5

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the maze
    ax.imshow(wall_img, origin='upper', cmap='gray', extent=[0, W, 0, H])

    # Extract positions
    xs = np.array([pos[1] for pos in positions])
    ys = np.array([H - pos[0] for pos in positions])

    # Create a time array normalized between 0 and 1
    timesteps = np.linspace(0, 1, len(xs))

    # Define red → blue colormap
    red_to_blue = LinearSegmentedColormap.from_list("red_blue", ["red", "blue"])

    # Scatter plot colored by timestep
    sc = ax.scatter(xs, ys, c=timesteps, cmap=red_to_blue, s=25, label='Trajectory')

    # Connect points with lines colored by time
    for i in range(1, len(xs)):
        seg_x = xs[i-1:i+1]
        seg_y = ys[i-1:i+1]
        color = red_to_blue(timesteps[i])



        ax.plot(seg_x, seg_y, color=color, linewidth=2)

    # Start and End markers
    ax.scatter(xs[0], ys[0], color='green', s=80, label='Start')
    ax.scatter(xs[-1], ys[-1], color='black', s=80, label='End')

    # Title, axes, and colorbar
    ax.set_title(f"Agent Trajectory in {maze_name} (time: red → blue)")
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_aspect('equal')

    # Unique legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Add colorbar to indicate progress over time
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Progress (0 = start, 1 = end)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved trajectory to: {save_path}")
    plt.close()

def plot_policy_kl_divergence(env_name: str,
                              ckpt_nums: Sequence[int],
                              seed_num: int,
                              goal,
                              alphas: Sequence[float],
                              save_root: str = "plots"):
    """
    Computes and plots  KL( π(·|s₀,g)  ||  π(·|s₀,w_α) )  for every
    α ∈ alphas and checkpoint ∈ ckpt_nums.

    Steps per checkpoint
    --------------------
    1.   Load learner state, env, nets (via load_checkpoint).
    2.   Compute action‑probabilities p = π(·|s₀,g).
    3.   For each α
         • find w_α whose ψ(w) is nearest to (1‑α)ψ(s₀)+αψ(g);
         • compute q = π(·|s₀,w_α);
         • KL = Σ_a  p_a · log(p_a / q_a).
    4.   Plot one curve per α (linear scale) and save:
         • PNG  →  <save_root>/<env>/<seed>/kl_ckpts_alphas.png
         • CSV  →  …/kl_ckpts_alphas.csv   (ckpt,α,w_i,w_j,kl)
    """

    # ───── path prep ───────────────────────────────────────────────────────
    out_dir = os.path.join(save_root, env_name, str(seed_num))
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "kl_ckpts_alphas.png")
    csv_path = os.path.join(out_dir, "kl_ckpts_alphas.csv")

    # ───── holders for plotting / csv ──────────────────────────────────────
    kl_by_alpha = {α: [] for α in alphas}
    csv_rows = [["ckpt", "alpha", "w_i", "w_j", "kl"]]

    # ───── iterate checkpoints ────────────────────────────────────────────
    for ckpt in ckpt_nums:
        # 1) checkpoint -----------------------------------------------------
        state, env, nets = load_checkpoint(
            alpha='0.1',
            misc_params='0.1_None',
            env_name=env_name,
            base_log_dir='./logs',
            seed=seed_num,
            fix_goals=True,
            ckpt_num=ckpt,
            ckpt_dir=(f'/home/mahsa/sgcrl/logs/contrastive_cpc_{env_name}_{seed_num}'
                      f'/checkpoints/learner'),
            goal=goal,
        )
        params_pol = state.policy_params
        params_q   = state.q_params



        # ------------------------------------------------------------------
        # 1.  Extract the *base* Normal (loc, scale) from the policy output
        # ------------------------------------------------------------------
        def gaussian_from_policy(dist):
            """
            dist = tfp.Independent(tfd.TransformedDistribution(
                    distribution=tfd.Normal(loc, scale), bijector=Tanh))
            Return the underlying Normal.
            """
            if isinstance(dist, tfd.Independent):
                base = dist.distribution          # TransformedDistribution
                if isinstance(base, tfd.TransformedDistribution):
                    return base.distribution      # Normal(loc,scale)
            raise TypeError("Unexpected policy distribution type")

        # ------------------------------------------------------------------
        # 2.  Compute KL between two multivariate diagonal Normals
        # ------------------------------------------------------------------
        def kl_gaussians(norm_p: tfd.Normal,
                 norm_q: tfd.Normal,
                 reduce: str = "sum") -> float:
            """
            KL between two diagonal multivariate Normals.

            If 'norm_p' and 'norm_q' have event_shape=[d] but batch_shape=[dims],
            tfp.kl_divergence returns a tensor with shape=batch_shape.
            Reduce that tensor to a scalar before casting to float.
            """
            kl_vec = tfd.kl_divergence(norm_p, norm_q)   # e.g. shape (2,)

            if reduce == "mean":
                kl_scalar = jnp.mean(kl_vec)
            else:                      # "sum" (default)
                kl_scalar = jnp.sum(kl_vec)
                print(f"kl_vec = {kl_vec}")
                kl_scalar = kl_vec[0][1]

            return float(kl_scalar)
        # put this near gaussian_from_policy / kl_gaussians helpers
        def tanh_mean(dist):
            """Return tanh(loc) after un‑wrapping Independent+Tanh."""
            if isinstance(dist, tfd.Independent):
                dist = dist.distribution          # TransformedDistribution
            if isinstance(dist, tfd.TransformedDistribution):
                dist = dist.distribution          # Normal(loc, scale)
            return jnp.tanh(dist.loc)             # shape (action_dim,)
        

        def actor_mean_and_std(params_pol, obs, squash=True):
            """
            Returns:
                mu_tanh   : tanh(μ)
                mu_plus   : tanh(μ + σ)
                mu_minus  : tanh(μ − σ)

            Works with Haiku policies that output
            Independent( TransformedDistribution( Normal, bijector=tanh ) ).
            """
            dist = nets.policy_network.apply(params_pol, obs)

            # --- unwrap until we see Normal(loc, scale) ---------------------------
            if isinstance(dist, tfd.Independent):
                dist = dist.distribution                    # TransformedDistribution
            if isinstance(dist, tfd.TransformedDistribution):
                dist = dist.distribution                    # underlying Normal

            if not hasattr(dist, "loc") or not hasattr(dist, "scale"):
                raise TypeError("Could not locate mean / std in policy distribution")

            mu = dist.loc           # shape (action_dim,)
            sigma = dist.scale      # std‑dev (positive)

            if squash:
                mu_tanh   = jnp.tanh(mu)
                mu_plus   = jnp.tanh(mu + sigma)
                mu_minus  = jnp.tanh(mu - sigma)
            else:
                mu_tanh   = mu
                mu_plus   = mu + sigma
                mu_minus  = mu - sigma

            return mu_tanh, mu_plus, mu_minus

        # 2) p(a|s0,g) ------------------------------------------------------
        s0 = jnp.asarray(goal[0], dtype=jnp.float32)
        g  = jnp.asarray(goal[1], dtype=jnp.float32)
        dist_p = nets.policy_network.apply(params_pol,  jnp.concatenate([s0, g])[None])

        # ψ(s0) & ψ(g) for interpolation -----------------------------------
        a_sg = nets.policy_network.apply(params_pol,
                                         jnp.concatenate([s0, g])[None]).mode()
        psi_s = _phi_state_action(params_q, nets, s0, a_sg)
        psi_g = _psi_goal_external(params_q, nets, g)

        # pre‑compute ψ grid ----------------------------------------------
        maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
        H, W = WALLS[maze_key].shape
        idxs = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'),
                         axis=-1).reshape(-1, 2)
        psi_grid = jax.vmap(lambda idx: _psi_goal_external(
            params_q, nets,
            jnp.array([idx[0], idx[1]], jnp.float32)))(idxs)
        idxs_np, psi_grid_np = np.array(idxs), np.array(psi_grid)

        # 3) loop over α ----------------------------------------------------
        for α in alphas:
            rep = (1-α) * psi_s + α * psi_g
            best = np.argmin(np.sum((psi_grid_np - rep[None, :])**2, axis=1))
            w_idx = idxs_np[best]
            w = jnp.asarray(w_idx, dtype=jnp.float32)

            dist_q = nets.policy_network.apply(params_pol, jnp.concatenate([s0, w])[None] )
            # try:
            #     print("KL implemnted for this distribution")
            #     kl_exact = tfd.kl_divergence(dist_p, dist_q)         # may already work
            #     kl_val = float(jnp.sum(kl_exact))  
            #    # reduce batch dims
            #except NotImplementedError:
            # norm_p = gaussian_from_policy(dist_p)
            # norm_q = gaussian_from_policy(dist_q)
            # p_mu , p_mu_plus , p_mu_minus = actor_mean_and_std(params_pol, jnp.concatenate([s0, g])[None])
            # q_mu , q_mu_plus , q_mu_minus = actor_mean_and_std(params_pol, jnp.concatenate([s0, w])[None])
            # print("=====================")
            # print("p_mu", p_mu, "p_mu_plus", p_mu_plus, "p_mu_minus", p_mu_minus)
            # print("q_mu", q_mu, "q_mu_plus", q_mu_plus, "q_mu_minus", q_mu_minus)
            # kl_val = kl_gaussians(norm_p, norm_q)
            mu_p = tanh_mean(dist_p)                 # shape (2,)
            mu_q = tanh_mean(dist_q)
            abs_diff = jnp.abs(mu_p - mu_q)          # (2,)
            kl_val = abs_diff[0][1]
                

            kl_by_alpha[α].append(kl_val)
            csv_rows.append([ckpt, α, int(w_idx[0]), int(w_idx[1]), kl_val])

            print(f"[ckpt {ckpt}] α={α:.2f} KL={kl_val:.6f}  w={tuple(w_idx)}")

    # ───── save CSV --------------------------------------------------------
    # with open(csv_path, "w", newline="") as f:
    #     csv.writer(f).writerows(csv_rows)
    # print(f"CSV saved → {csv_path}")

    # ───── plot ------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    for α in alphas:
        plt.plot(ckpt_nums,
                 kl_by_alpha[α],
                 marker='o',
                 label=rf"$\|\mu_{{\pi}}(s_0, g) - \mu_{{\pi}}(s_0, w_{{\alpha}})\|_1$, $\alpha={α}$")
    plt.xlabel("Checkpoint")
    plt.ylabel("Mean difference")
    plt.title(f"Policy mean difference. checkpoint  ({env_name})")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Plot saved → {png_path}")

def report_representation_stats(env_name: str,
                                     ckpt_num: int,
                                     seed_num: int,
                                     goal):
    """
    Reports statistics (mean, variance) of L2 norms for ψ(s), φ(s,a), and norm of ψ(g) on the grid.
    """

    # ─── Load model and env ────────────────────────
    trained_state, env, nets = load_checkpoint(
        alpha='0.1',
        misc_params='0.1_None',
        env_name=env_name,
        base_log_dir='./logs',
        seed=seed_num,
        fix_goals=True,
        ckpt_num=ckpt_num,
        ckpt_dir=(f'/home/mahsa/sgcrl/logs/contrastive_cpc_{env_name}_{seed_num}'
                  f'/checkpoints/learner'),
        goal=goal,
    )
    params_q = trained_state.q_params

    # ─── Setup for grid points ─────────────────────
    maze_key = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
    H, W = WALLS[maze_key].shape
    idxs = jnp.stack(jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij'),
                     axis=-1).reshape(-1, 2)

    py_s0, py_g = goal
    g = jnp.asarray(py_g, dtype=jnp.float32)

    # ─── ψ(g) norm ────────────────────────────────
    psi_g = _psi_goal_external(params_q, nets, g)
    norm_psi_g = jnp.linalg.norm(psi_g)
    print(f"‣ ‖ψ(g)‖ = {float(norm_psi_g):.4f}")

    # ─── L2 norms of ψ(s) over the grid ───────────
    psi_s_vals = jax.vmap(
        lambda s: jnp.linalg.norm(_psi_goal_external(params_q, nets,
                                  jnp.array([s[0], s[1]], jnp.float32))))(idxs)
    psi_s_np = np.array(psi_s_vals)
    print(f"‣ ‖ψ(s)‖: mean = {np.mean(psi_s_np):.4f}, var = {np.var(psi_s_np):.4f}")

    # ─── L2 norms of φ(s, π(s, g)) over the grid ──
    def phi_norm_from_idx(idx):
        s = jnp.array([idx[0], idx[1]], jnp.float32)
        a = nets.policy_network.apply(trained_state.policy_params,
                                      jnp.concatenate([s, g])[None]).mode()
        return jnp.linalg.norm(_phi_state_action(params_q, nets, s, a))

    phi_norms = jax.vmap(phi_norm_from_idx)(idxs)
    phi_np = np.array(phi_norms)
    print(f"‣ ‖φ(s, π(s,g))‖: mean = {np.mean(phi_np):.4f}, var = {np.var(phi_np):.4f}")
# === Main ===
if __name__ == "__main__":

    num_episodes = 1

    
    # ckpt_num_list = [13, 14, 15]
    #ckpt_num_list = [1 ,3 , 5, 7, 10, 15,  20, 25,30]
    
    #goals = [None]
    ckpt_num_list = [15 , 20, 25, 30, 35, 40, 45, 50]
    ckpt_num_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #ckpt_num_list = [1 ,2 , 3, 4, 5, 6, 7, 8, 9, 10, 11 , 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    #ckpt_num_list = [ 20, 30, 40, 50, 60, 70, 80,90, 100, 110, 120, 130, 140, 150, 160, 170, 180,190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]
    #ckpt_num_list = [ 1 , 2, 3, 4, 5, 6, 7, 8,9, 10, 15, 20, 30, 40, 50, 60, 70, 80,90, 100, 110, 120, 130, 140, 150, 160, 170]
    #ckpt_num_list = [1 , 5,  10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150, 170, 200, 250]
    #ckpt_num_list = [220, 240]
    #ckpt_num_list = [2 , 3 , 4, 5, 6, 7, 8,9, 10]
    #ckpt_num_list = [42, 45, 47, 49]
    #ckpt_num_list = [75, 78]
    #ckpt_num_list = [20, 21, 22, 23]
    #env_name = "point_Maze11x11"
    env_name = "point_Impossible"
    #env_name = "point_Spiral11x11"
    env_name = "point_Spiral11x11"
    #env_name = "point_FourRooms"
    # if env_name == "point_FourRooms":
    env_name = "point_Wall11x11"

    #     seed_num_list = [11]
    # else:
    #     seed_num_list = [10]
    seed_num_list = [1]
    goal = fixed_goal_dict[env_name]
    print("Goals: ", goal)
    #goals = [None]


    for seed_num in seed_num_list:
        for ckpt_num in ckpt_num_list:
            # first this dhould be called to create the prior over the states
            generate_and_save_episodes(
                env_name=env_name,
                ckpt_num=ckpt_num,
                seed_num=seed_num,
                num_episodes=num_episodes,
                plot=True,
                goal = goal
            )



    # for seed_num in seed_num_list:
    #     for ckpt_num in ckpt_num_list:


    #         # posterior_heatmap_for_checkpoint(
    #         # env_name=env_name,
    #         # ckpt_num=ckpt_num,
    #         # seed_num=seed_num,
    #         # goal=goal,              # same goal you pass elsewhere
    #         # )
    #         print("====================")
    #         print("ckpt num", ckpt_num)
            

    #         # run_interp_and_closest_states(
    #         # env_name=env_name,
    #         # ckpt_num=ckpt_num,
    #         # seed_num=seed_num,
    #         # goal=fixed_goal_dict[env_name],
    #         # save_root="plots",
    #         # )

            # report_representation_stats(
            #     env_name=env_name,
            #     ckpt_num=ckpt_num,
            #     seed_num=seed_num,
            #     goal=fixed_goal_dict[env_name],
            # )

        
    # compare_q_distances(
    #     env_name=env_name,
    #     ckpt_nums=ckpt_num_list,
    #     seed_num=seed_num_list[0],
    #     goal=fixed_goal_dict[env_name],
    #     alphas = [0.3 , 0.5, 0.7],
    # )

    # plot_policy_kl_divergence(
    # env_name = env_name,
    # ckpt_nums = ckpt_num_list,
    # seed_num = seed_num_list[0],
    # goal = fixed_goal_dict[env_name],
    # alphas = [0.3 , 0.5 , 0.7]
    # )