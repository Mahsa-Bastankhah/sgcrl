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
import jax.numpy as jnp
from typing import Sequence
from typing import Tuple
from typing import Union      # add to the existing typing imports

def _discretize_state(state, resolution=1.0, walls=None):
    ij = np.floor(resolution * state).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(walls.shape) - 1)
    return tuple(ij)
# Make sure your config, make_environment, make_networks, ContrastiveLearner, etc., are properly imported



def run_episode(trained_learner_state, env, networks):
    positions = []
    actions = []
    rewards = []

    timestep = env.reset()
    print("timestep ", timestep)

    @jax.jit
    def select_action(params, obs):
        return networks.policy_network.apply(params, obs).mode()
    i = 0
    while not timestep.last():
        obs = timestep.observation
        action = select_action(trained_learner_state.policy_params, obs)
        
        positions.append(obs[:3]) 
        print("new step:" , i)
        print("state", obs[:3])
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


def generate_and_save_episodes(env_name, ckpt_num, seed_num=42, num_episodes=10, plot=True, goal=None):
    trained_learner_state, env, networks = load_checkpoint(
        alpha='0.1',
        misc_params='0.1_None',
        env_name=env_name,
        base_log_dir='./logs',
        seed=seed_num,
        fix_goals=True,
        ckpt_num=ckpt_num,
        ckpt_dir=f'/home/mahsa/sgcrl/logs/contrastive_cpc_{env_name}_{seed_num}/checkpoints/learner',
        goal = goal,
    )

    all_positions, all_actions, all_rewards = [], [], []
    save_dir = f"data/{env_name}/{seed_num}/"
    os.makedirs(save_dir, exist_ok=True)

    for i in trange(num_episodes, desc="Running episodes"):
        data = run_episode(trained_learner_state, env, networks)
        all_positions.append(data['positions'])
        all_actions.append(data['actions'])
        all_rewards.append(data['rewards'])

        if plot:
            goal_str = "_".join([f"{int(g)}" for g in goal[1]])
            plot_path = f"plots/{env_name}/{seed_num}/trajectory_ckpt_{ckpt_num}_{i}_{goal_str}.png"

            #plot_trajectory_on_maze_with_time(env_name, data['positions'], plot_path)
            plot_trajectory_on_maze_by_action(
                env_name=env_name,
                positions=data['positions'],
                actions=data['actions'],
                save_path=plot_path,
                jump_band=(0.50, 0.55)  # tweak if the threshold changes
            )

    np.savez_compressed(
        os.path.join(save_dir, f"ckpt_{ckpt_num}_{goal_str}.npz"),
        positions=np.array(all_positions, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object)
    )
    print(f"Saved all {num_episodes} episodes to {save_dir}/ckpt_{ckpt_num}_{goal_str}.npz")

def plot_trajectory_on_maze_with_depth(env_name='Spiral11x11', positions=None, save_path='agent_trajectory.png'):
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

    # Extract positions and depths
    xs = [pos[1] for pos in positions]
    ys = [H - pos[0] for pos in positions]
    depths = [pos[2] if len(pos) > 2 else 0 for pos in positions]  # Depth in [0, 1]

    # Define blue → orange colormap
    blue_to_orange = LinearSegmentedColormap.from_list("blue_orange", ["blue", "orange"])

    # Create scatter plot colored by depth
    sc = ax.scatter(xs, ys, c=depths, cmap=blue_to_orange, s=25, label='Trajectory')

    # Connect trajectory lines using color mapping per segment
    for i in range(1, len(xs)):
        seg_x = xs[i-1:i+1]
        seg_y = ys[i-1:i+1]
        ax.plot(seg_x, seg_y, color=blue_to_orange(depths[i]), linewidth=2)

    # Start and End markers
    ax.scatter(xs[0], ys[0], color='green', s=80, label='Start')
    ax.scatter(xs[-1], ys[-1], color='red', s=80, label='End')

    # Title, axes, and colorbar
    ax.set_title(f"Agent Trajectory in {maze_name} (depth: blue → orange)")
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_aspect('equal')
    ax.legend()

    # Add colorbar to indicate depth
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Depth (0 = blue, 1 = orange)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved trajectory to: {save_path}")
    plt.close()

def plot_trajectory_on_maze_with_time(env_name='Spiral11x11', positions=None, save_path='agent_trajectory.png', jump_threshold=1.4):
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

        # Calculate Euclidean distance between points
        distance = np.linalg.norm(np.array([seg_x[1] - seg_x[0], seg_y[1] - seg_y[0]]))

        if distance > jump_threshold:
            # If jump detected, plot a star at the starting point
            ax.scatter(seg_x[0], seg_y[0], marker='*', s=120, color='gold', edgecolor='black', label='Jump' if i == 1 else "")
            ax.scatter(seg_x[1], seg_y[1], marker='*', s=120, color='gold', edgecolor='black')
        else:
            # Normal connection
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


def plot_trajectory_on_maze_by_action(
        env_name: str = 'Spiral11x11',
        positions: Union[Sequence[Tuple[int, int]], np.ndarray, None] = None,
        actions: Union[Sequence[np.ndarray], None] = None,
        save_path: str = 'agent_trajectory.png',
        jump_band: Tuple[float, float] = (0.50, 0.55),
    ):
    """
    Plot an agent trajectory on the maze, colouring by time and marking
    *jump* actions with gold stars.

    A jump is defined as any timestep t whose action satisfies
    jump_band[0] ≤ action[t][2] ≤ jump_band[1].
    Stars are placed on the position *before* and *after* the jump.

    Args
    ----
    env_name   : full environment name (e.g. 'contrastive_Spiral11x11').
    positions  : list/array of (row, col) grid positions, length = T.
    actions    : list/array of action vectors, length = T‑1 (one per step).
    save_path  : output PNG path.
    jump_band  : closed interval for a[2] that signals a jump.
    """
    assert positions is not None and actions is not None, \
        "Need both positions and actions for jump detection."
    # assert len(actions) == len(positions) - 1, \
    #     "actions should have one fewer element than positions."

    # --- Maze geometry -----------------------------------------------------
    maze_name = env_name.split("_")[2] if "stochastic" in env_name else env_name.split("_")[1]
    walls = WALLS[maze_name]
    H, W = walls.shape
    wall_img = np.ones_like(walls, dtype=float)
    wall_img[walls == 1] = 0.5

    # --- Figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(wall_img, origin='upper', cmap='gray', extent=[0, W, 0, H])

    xs = np.array([p[1] for p in positions])
    ys = np.array([H - p[0] for p in positions])
    timesteps = np.linspace(0, 1, len(xs))
    cmap = LinearSegmentedColormap.from_list("red_blue", ["red", "blue"])
    sc = ax.scatter(xs, ys, c=timesteps, cmap=cmap, s=25, label='Trajectory')

    # --- Draw path segments and stars --------------------------------------
    for t in range(len(actions[:-1])):
        seg_x, seg_y = xs[t:t+2], ys[t:t+2]
        ax.plot(seg_x, seg_y, color=cmap(timesteps[t+1]), linewidth=2)

        a2 = actions[t][2]
        if jump_band[0] <= a2 <= jump_band[1]:
            # star at start (t) and end (t+1)
            ax.scatter(seg_x[0], seg_y[0], marker='*', s=120,
                       color='gold', edgecolor='black',
                       label='Jump' if t == 0 else "")
            ax.scatter(seg_x[1], seg_y[1], marker='*', s=120,
                       color='gold', edgecolor='black')

    # --- Start / End markers & cosmetics -----------------------------------
    ax.scatter(xs[0], ys[0], color='green', s=80, label='Start')
    ax.scatter(xs[-1], ys[-1], color='black', s=80, label='End')
    ax.set_title(f"Agent Trajectory in {maze_name} – time: red → blue")
    ax.set_xlim([0, W]); ax.set_ylim([0, H]); ax.set_aspect('equal')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(),
              dict(zip(labels, handles)).keys())
    cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Progress (0 = start, 1 = end)")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Saved trajectory plot to {save_path}")

def plot_depth_action_histogram(env_name, seed_num, ckpt_num, data_dir="data", plot_dir="plots", goal=None):
    """Reads saved episode data and saves a histogram of z-axis (depth) actions with fixed x-axis."""
    if goal is None:
        file_path = os.path.join(data_dir, env_name, str(seed_num), f"ckpt_{ckpt_num}.npz")
    else:
        goal_str = "_".join([f"{int(g)}" for g in goal[1]])
        file_path = os.path.join(data_dir, env_name, str(seed_num), f"ckpt_{ckpt_num}_{goal_str}.npz")
    
    if not os.path.exists(file_path):
        print(f"Data file not found at: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    actions = data["actions"]

    # Flatten list of episodes and steps into one big array
    flat_actions = np.concatenate(actions, axis=0)
    
    if flat_actions.shape[1] < 3:
        print("No depth component found in actions.")
        return

    depth_actions = flat_actions[:, 2]

    # Prepare save path
    save_dir = os.path.join(plot_dir, env_name, str(seed_num))
    os.makedirs(save_dir, exist_ok=True)
    if goal is None:
        save_path = os.path.join(save_dir, f"depth_hist_ckpt_{ckpt_num}.png")
    else:
        save_path = os.path.join(save_dir, f"depth_hist_ckpt_{ckpt_num}_{goal_str}.png")

    # Plot and save
    plt.figure(figsize=(6, 4))
    plt.hist(depth_actions, bins=30, color='teal', edgecolor='black', range=(-1, 1))
    plt.xlabel("Depth action value (action[2])")
    plt.ylabel("Frequency")
    plt.title(f"Depth Action Histogram\n{env_name}, Seed {seed_num}, Checkpoint {ckpt_num}")
    plt.xlim(-1, 1)  # Force x-axis to show full action range
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved depth action histogram to: {save_path}")

def plot_action_dim2_over_time(env_name, seed_num, ckpt_num, data_dir="data", plot_dir="plots", goal=None):
    """Plots action[2] (third dimension) over time for each episode."""
    if goal is None:
        file_path = os.path.join(data_dir, env_name, str(seed_num), f"ckpt_{ckpt_num}.npz")
    else:
        goal_str = "_".join([f"{int(g)}" for g in goal[1]])
        file_path = os.path.join(data_dir, env_name, str(seed_num), f"ckpt_{ckpt_num}_{goal_str}.npz")
    
    if not os.path.exists(file_path):
        print(f"Data file not found at: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    actions = data["actions"]  # shape: (num_episodes, steps, action_dim)

    save_dir = os.path.join(plot_dir, env_name, str(seed_num))
    os.makedirs(save_dir, exist_ok=True)
    if goal is None:
        save_path = os.path.join(save_dir, f"action2_time_ckpt_{ckpt_num}.png")
    else:
        save_path = os.path.join(save_dir, f"action2_time_ckpt_{ckpt_num}_{goal_str}.png")

    plt.figure(figsize=(8, 5))

    for ep_idx, ep_actions in enumerate(actions):
        ep_actions = np.array(ep_actions)
        if ep_actions.shape[1] < 3:
            print(f"Episode {ep_idx} has actions with fewer than 3 dimensions.")
            continue
        action_dim2 = ep_actions[:, 2]  # Third dimension
        plt.plot(action_dim2, label=f"Episode {ep_idx}")

    plt.xlabel("Timestep")
    plt.ylabel("Action[2] (third dimension)")
    plt.title(f"Action[2] over Time\n{env_name}, Seed {seed_num}, Ckpt {ckpt_num}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved action[2] over time plot to: {save_path}")


# ────────────────────────────────────────────────────────────────────────────
def arrow_plot_phi_psi_ckpt(env_name, seed_num, ckpt_num,
                         goal=None,
                         grid_size=10,                    # 10 × 10 lattice
                         plot_dir="plots",
                         save_name="sa_g_arrows.png"):
    """
    Draw arrows  sa_repr(s,a)  →  g_repr_state(s′)
    for every east transition  s = (x,y,0)  →  s′ = s + (1,0,0).

    sa_repr  == φ(s,a)   and   g_repr_state == ψ(s′).
    """
    if "stochastic" in env_name:
        maze_name = env_name.split("_")[2]
    else:   
        maze_name = env_name.split("_")[1]
    walls = WALLS[maze_name]
    H, W = walls.shape
    # 1) Load checkpoint (same call as before)
    trained_learner_state, _, networks = load_checkpoint(
        alpha='0.1',
        misc_params='0.1_None',
        env_name=env_name,
        base_log_dir='./logs',
        seed=seed_num,
        fix_goals=True,
        ckpt_num=ckpt_num,
        ckpt_dir=(f'/home/mahsa/sgcrl/logs/'
                  f'contrastive_cpc_{env_name}_{seed_num}/checkpoints/learner'),
        goal=goal,
    )

    # 2) Your new JIT helpers -------------------------------------------------
    @jax.jit
    def get_q_and_repr(q_params, obs_batch, action_batch):
        return networks.q_network.apply(q_params, obs_batch, action_batch)



    # 3) Enumerate grid states and collect representations --------------------
    east_action   = np.array([1., 0., 0.], dtype=float)
    action_batch  = jnp.expand_dims(jnp.array(east_action), axis=0)

    sa_vecs, g_vecs = [], []
        # … previous code …



    # NEW ──────────────────────────────────────────────────────────────────
    arrows   = []      # L2  distance  ‖sa_repr − g_repr‖
    coords  = []      # (x,y)  of the corresponding state
    # ──────────────────────────────────────────────────────────────────────

    for x in range(grid_size - 1):
        for y in range(grid_size):
            s      = np.array([x,     y, 0.], dtype=float)
            s_next = np.array([x + 1, y, 0.], dtype=float)

            # (state ‖ goal)  where goal := s′
            obs_concat   = np.concatenate([s, s_next])
            obs_batch    = jnp.expand_dims(jnp.array(obs_concat), axis=0)
            action_batch = jnp.expand_dims(jnp.array(east_action), axis=0)

            # one call gives q_value, sa_repr, g_repr
            _, sa_repr, g_repr = get_q_and_repr(
                                    trained_learner_state.q_params,
                                    obs_batch,
                                    action_batch)

            sa_vecs.append(np.asarray(sa_repr)[0])   # φ(s,a)
            g_vecs.append(np.asarray(g_repr)[0])     # ψ(s′)


            # NEW ───────────────────────────────────────────────────────────
              # NEW – store arrow and its origin's coords
            arrows.append(g_vecs[-1] - sa_vecs[-1])
            coords.append((int(s[0]), int(s[1])))

            # ───────────────────────────────────────────────────────────────

        # NEW ──────────────────────────────────────────────────────────────────
        # NEW – find angular outliers
    U = np.stack([v / (np.linalg.norm(v) + 1e-8) for v in arrows])   # unit dirs
    mean_u = U.mean(axis=0)
    mean_u /= np.linalg.norm(mean_u) + 1e-8

    cos_sims = (U @ mean_u)          # cosine similarity to mean direction
    thresh   = 0.3                   # flag anything with cos < 0.80  (~> 36°)

    print("\nStates whose arrow direction differs a lot (cos < 0.80):")
    for (x, y), c in zip(coords, cos_sims):
        if c < thresh:
            ang_deg = np.degrees(np.arccos(np.clip(c, -1, 1)))
            print(f"  (x={x}, y={H - y})   cos={c:.3f}   angle={ang_deg:.1f}°")

    # ──────────────────────────────────────────────────────────────────────


    # 4) PCA projection + arrow plot -----------------------------------------
    V  = np.vstack([sa_vecs, g_vecs])
    xy = PCA(n_components=2).fit_transform(V)
    n  = len(sa_vecs)
    sa_xy, g_xy = xy[:n], xy[n:]

    save_dir = os.path.join(plot_dir, env_name, str(seed_num))
    os.makedirs(save_dir, exist_ok=True)

    base, ext = save_name.rsplit('.', 1)
    save_path = os.path.join(save_dir, f"{base}_{ckpt_num}")

    plt.figure(figsize=(6, 6))
    for p, q in zip(sa_xy, g_xy):
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1],
                  head_width=0.02, length_includes_head=True, alpha=0.6)
    plt.scatter(sa_xy[:, 0], sa_xy[:, 1], c='blue',   s=12, label='sa_repr φ')
    plt.scatter(g_xy[:, 0],  g_xy[:, 1],  c='orange', s=12, label='g_repr ψ')
    plt.legend(); plt.axis('equal')
    plt.title("sa_repr(s,a) → g_repr_state(s′) arrows (east moves)")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Saved arrow plot to {save_path}")


# === Main ===
if __name__ == "__main__":

    num_episodes = 1

    
    # ckpt_num_list = [13, 14, 15]
    #ckpt_num_list = [1 ,3 , 5, 7, 10, 15,  20, 25,30]
    seed_num_list = [72]
    #goals = [None]
    ckpt_num_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    goals =  [[np.array([0,0, 0], dtype=float), np.array([8,10,0], dtype=float)]]
    #goals = [None]
    env_name = 'stochastic_point_FourRooms'
    for goal in goals:
        for seed_num in seed_num_list:
            for ckpt_num in ckpt_num_list:
                generate_and_save_episodes(
                    env_name=env_name,
                    ckpt_num=ckpt_num,
                    seed_num=seed_num,
                    num_episodes=num_episodes,
                    plot=True,
                    goal = goal
                )
                # plot_depth_action_histogram(
                # env_name=env_name,
                # seed_num=seed_num,
                # ckpt_num=ckpt_num,
                # goal = goal
                # )
                # plot_action_dim2_over_time(
                # env_name=env_name,
                # seed_num=seed_num,
                # ckpt_num=ckpt_num,
                # goal=goal
                # )

                # arrow_plot_phi_psi_ckpt(
                #     env_name=env_name,
                #     seed_num=seed_num,
                #     ckpt_num=ckpt_num,
                #     goal=goal   # or None
                # )



    #ckpt_nums = [1, 2, 3, 4, 5, 6, 7, 8]
    # T, state_to_index, index_to_state = create_transition_matrix_from_checkpoints(
    # env_name='point_FourRooms',
    # seed_num=seed_num,
    # ckpt_nums=ckpt_nums,
    # resolution=1.0,
    # )

    # nonzero_indices = np.argwhere(T > 0)

    # for i, j in nonzero_indices:
    #     from_state = index_to_state[i]  # (row, col)
    #     to_state = index_to_state[j]    # (row, col)
    #     print(f"T[{i} → {j}] ({from_state} → {to_state}) = {T[i, j]}")


    # print(f"Total states: {len(index_to_state)}")

    # plot_gradient_policy_from_eigenvector(
    # env_name='point_FourRooms',
    # seed_num=seed_num,
    # ckpt_nums=ckpt_nums,
    # eigvec_index=0  # usually skip eigvec 0 (all-ones, eigenvalue 0)
    # )

    # plot_gradient_policies_combined(
    # env_name='point_FourRooms',
    # seed_num=seed_num,
    # ckpt_nums=ckpt_nums,
    # max_vectors=16
    # )

