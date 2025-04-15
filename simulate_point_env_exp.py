import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import trange
import jax
import functools
import tensorflow as tf
from construct_T_matrix import load_checkpoint
from point_env import WALLS
from collections import defaultdict
import numpy as np
import os
from point_env import WALLS
import numpy as np
import matplotlib.pyplot as plt
from point_env import WALLS
import os
import matplotlib.cm as cm

def _discretize_state(state, resolution=1.0, walls=None):
    ij = np.floor(resolution * state).astype(int)
    ij = np.clip(ij, np.zeros(2), np.array(walls.shape) - 1)
    return tuple(ij)
# Make sure your config, make_environment, make_networks, ContrastiveLearner, etc., are properly imported





# def run_episode(trained_learner_state, env, networks):
#     positions = []
#     timestep = env.reset()

#     # Compile the policy
#     @jax.jit
#     def select_action(params, obs):
#         return networks.policy_network.apply(params, obs).mode()

#     while not timestep.last():
#         obs = timestep.observation
#         action = select_action(trained_learner_state.policy_params, obs)
#         positions.append(obs[:2])  # Assuming obs[:2] = (x, y)
#         print(f"Action: {action}, Position: {obs[:2]}")
#         timestep = env.step(np.array(action))

#     return positions

def run_episode(trained_learner_state, env, networks):
    positions = []
    actions = []
    rewards = []

    timestep = env.reset()

    @jax.jit
    def select_action(params, obs):
        return networks.policy_network.apply(params, obs).mode()

    while not timestep.last():
        obs = timestep.observation
        action = select_action(trained_learner_state.policy_params, obs)
        
        positions.append(obs[:2])                  # Assuming obs[:2] = (x, y)
        actions.append(action)
        rewards.append(timestep.reward)

        timestep = env.step(np.array(action))

    return {
        "positions": np.array(positions),
        "actions": np.array(actions),
        "rewards": np.array(rewards)
    }


def generate_and_save_episodes(env_name, ckpt_num, seed_num=42, num_episodes=10, plot=True):
    trained_learner_state, env, networks = load_checkpoint(
        alpha='0.1',
        misc_params='0.1_None',
        env_name=env_name,
        base_log_dir='./logs',
        seed=seed_num,
        fix_goals=True,
        ckpt_num=ckpt_num,
        ckpt_dir=f'/home/mahsa/sgcrl/logs/contrastive_cpc_{env_name}_{seed_num}/checkpoints/learner'
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
            plot_path = f"plots/{env_name}/{seed_num}/trajectory_ckpt_{ckpt_num}_{i}.png"
            plot_trajectory_on_maze(env_name, data['positions'], plot_path)

    np.savez_compressed(
        os.path.join(save_dir, f"ckpt_{ckpt_num}.npz"),
        positions=np.array(all_positions, dtype=object),
        actions=np.array(all_actions, dtype=object),
        rewards=np.array(all_rewards, dtype=object)
    )
    print(f"Saved all {num_episodes} episodes to {save_dir}/ckpt_{ckpt_num}.npz")



def plot_trajectory_on_maze(env_name='Spiral11x11', positions=None, save_path='agent_trajectory.png'):
    maze_name = env_name.split("_")[1]
    walls = WALLS[maze_name]
    H, W = walls.shape

    wall_img = np.ones_like(walls, dtype=float)
    wall_img[walls == 1] = 0.5

    fig, ax = plt.subplots(figsize=(6, 6))

    # origin='lower' keeps bottom-left as (0,0) in the plot
    ax.imshow(wall_img, origin='upper', cmap='gray', extent=[0, W, 0, H])

    # # Flip Y axis to match matrix top-down indexing
    xs = [pos[1] for pos in positions]            # x = column
    ys = [H - pos[0] for pos in positions]        # y = flipped row



    ax.plot(xs, ys, color='blue', linewidth=2, label='Agent Path')
    ax.scatter(xs[0], ys[0], color='green', s=80, label='Start')
    ax.scatter(xs[-1], ys[-1], color='red', s=80, label='End')

    ax.set_title(f"Agent Trajectory in {maze_name}")
    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved trajectory to: {save_path}")
    plt.close()


def create_state_to_index_mapping(walls):
    H, W = walls.shape
    state_to_index = {}
    index_to_state = []
    idx = 0
    for i in range(H):
        for j in range(W):
            if walls[i, j] == 0:  # not a wall
                state_to_index[(i, j)] = idx
                index_to_state.append((i, j))
                idx += 1
    return state_to_index, index_to_state


def create_transition_matrix_from_checkpoints(env_name, seed_num, ckpt_nums, resolution=1.0):
    walls = WALLS[env_name.split("_")[1]]
    state_to_index, index_to_state = create_state_to_index_mapping(walls)
    num_states = len(index_to_state)

    T = np.zeros((num_states, num_states), dtype=int)

    for ckpt_num in ckpt_nums:
        file_path = f"data/{env_name}/{seed_num}/ckpt_{ckpt_num}.npz"
        if not os.path.exists(file_path):
            print(f"Checkpoint {ckpt_num} not found at {file_path}, skipping.")
            continue

        data = np.load(file_path, allow_pickle=True)
        positions_all = data['positions']

        for episode in positions_all:
            for t in range(len(episode) - 1):
                s = _discretize_state(episode[t], resolution, walls=walls)
                s_next = _discretize_state(episode[t + 1], resolution, walls=walls)

                if s not in state_to_index or s_next not in state_to_index:
                    continue  # skip invalid positions (e.g., walls or out-of-bounds)

                i = state_to_index[s]
                j = state_to_index[s_next]
                T[i, j] = 1
    # At the end of create_transition_matrix_from_checkpoints:
    save_path = f"data/{env_name}/{seed_num}/T_ckpts_{'_'.join(map(str, ckpt_nums))}.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez_compressed(
        save_path,
        T=T,
        index_to_state=np.array(index_to_state, dtype=object)
    )

    print(f"Saved transition matrix to {save_path}")

        # Compute graph Laplacian
    A = T.copy().astype(float)  # adjacency matrix
    D = np.diag(A.sum(axis=1))  # degree matrix
    L = D - A                   # Laplacian

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)  # since L is symmetric

    # Print them
    print("Eigenvalues of Laplacian:")
    print(eigvals)

    # Save them
    # Save Laplacian and eigen decomposition
    lap_path = f"data/{env_name}/{seed_num}/laplacian_ckpts_{'_'.join(map(str, ckpt_nums))}.npz"
    np.savez_compressed(
        lap_path,
        L=L,
        eigvals=eigvals,
        eigvecs=eigvecs,
        index_to_state=np.array(index_to_state, dtype=object)  # ✅ include this
    )
    print(f"Saved Laplacian and eigendecomposition to {lap_path}")

    




    return T, state_to_index, index_to_state

def plot_gradient_policy_from_eigenvector(env_name, seed_num, ckpt_nums, eigvec_index=1):
    # Load Laplacian data
    lap_path = f"data/{env_name}/{seed_num}/laplacian_ckpts_{'_'.join(map(str, ckpt_nums))}.npz"
    data = np.load(lap_path, allow_pickle=True)
    eigvecs = data['eigvecs']
    eigenvalues = data['eigvals']
    index_to_state = data['index_to_state']

    walls = WALLS[env_name.split("_")[1]]
    H, W = walls.shape

    # Convert state lookup
    state_to_index = {tuple(s): i for i, s in enumerate(index_to_state)}
    eig = eigvecs[:, eigvec_index]  # use the requested eigenvector
    print("eigen vector ", eig)
    print("eigenvalue", eigenvalues[eigvec_index])

    # Prepare grid for arrows
    X, Y, U, V = [], [], [], []

    directions = {
        'up':    (-1, 0),
        'down':  (1, 0),
        'left':  (0, -1),
        'right': (0, 1),
    }

    for eigvec_index, eigval in enumerate(eigenvalues):
        if np.isclose(eigval, 0.0):  # skip zero eigenvalue (trivial eigenvector)
            continue

        eig = eigvecs[:, eigvec_index]

        # Find max state
        max_index = np.argmax(eig)
        max_position = tuple(index_to_state[max_index])
        max_value = eig[max_index]
        print(f"Eigenvector {eigvec_index} (λ = {eigval:.4f}) → max = {max_value:.4f} at position {max_position}")

        # Prepare grid for arrows
        X, Y, U, V = [], [], [], []

        for idx, (i, j) in enumerate(index_to_state):
            best_reward = -np.inf
            best_direction = None

            for di, dj in directions.values():
                ni, nj = i + di, j + dj
                if (ni, nj) in state_to_index:
                    s = idx
                    s_next = state_to_index[(ni, nj)]
                    reward = eig[s_next] - eig[s]
                    if reward > best_reward:
                        best_reward = reward
                        best_direction = (di, dj)

            if best_direction and best_reward > 0:
                print(f"State {idx} ({i}, {j}) → {i +best_direction[0], j + best_direction[1]} with reward {best_reward:.4f}")
                X.append(j)
                Y.append(H - i)  # to align with 'upper' origin
                U.append(best_direction[1])
                V.append(best_direction[0])

        # Plot maze and vector field
        wall_img = np.ones_like(walls, dtype=float)
        wall_img[walls == 1] = 0.5

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(wall_img, origin='upper', cmap='gray', extent=[0, W, 0, H])
        ax.quiver(X, Y, U, -np.array(V), color='blue', scale=1.5, scale_units='xy')

        ax.set_xlim([0, W])
        ax.set_ylim([0, H])
        ax.set_aspect('equal')
        ax.set_title(f"Policy from Eigenvector {eigvec_index} (λ = {eigval:.4f})")
        plt.tight_layout()

        save_path = f"plots/{env_name}/{seed_num}/policy_from_eig_ckpts_{'_'.join(map(str, ckpt_nums))}_{eigvec_index}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved policy plot to {save_path}")



def plot_gradient_policies_combined(env_name, seed_num, ckpt_nums, max_vectors=20):
    # Load Laplacian eigendecomposition
    lap_path = f"data/{env_name}/{seed_num}/laplacian_ckpts_{'_'.join(map(str, ckpt_nums))}.npz"
    data = np.load(lap_path, allow_pickle=True)
    eigvecs = data['eigvecs']
    eigenvalues = data['eigvals']
    index_to_state = data['index_to_state']

    walls = WALLS[env_name.split("_")[1]]
    H, W = walls.shape
    state_to_index = {tuple(s): i for i, s in enumerate(index_to_state)}

    directions = {
        'up':    (-1, 0),
        'down':  (1, 0),
        'left':  (0, -1),
        'right': (0, 1),
    }

    wall_img = np.ones_like(walls, dtype=float)
    wall_img[walls == 1] = 0.5

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(wall_img, origin='upper', cmap='gray', extent=[0, W, 0, H])

    colormap = cm.get_cmap('nipy_spectral', max_vectors)  # or 'turbo', 'hsv', 'tab20', etc.

    color_count = 0

    for eigvec_index, eigval in enumerate(eigenvalues):
        if np.isclose(eigval, 0.0):
            continue
        if color_count >= max_vectors:
            break

        eig = eigvecs[:, eigvec_index]

        # Find max state
        max_index = np.argmax(eig)
        max_position = tuple(index_to_state[max_index])
        max_value = eig[max_index]
        print(f"Eigenvector {eigvec_index} (λ = {eigval:.4f}) → max = {max_value:.4f} at position {max_position}")

        X, Y, U, V = [], [], [], []

        for idx, (i, j) in enumerate(index_to_state):
            best_reward = -np.inf
            best_direction = None

            for di, dj in directions.values():
                ni, nj = i + di, j + dj
                if (ni, nj) in state_to_index:
                    s = idx
                    s_next = state_to_index[(ni, nj)]
                    reward = eig[s_next] - eig[s]
                    if reward > best_reward:
                        best_reward = reward
                        best_direction = (di, dj)

            if best_direction and best_reward > 0:
                X.append(j)
                Y.append(H - i)
                U.append(best_direction[1])
                V.append(best_direction[0])

        style_idx = color_count  # or eigvec_index

        arrow = ax.quiver(
            X, Y, U, -np.array(V),
            color=colormap(style_idx / max_vectors),
            scale=1.5,
            scale_units='xy',
            width=0.005 + 0.001 * (style_idx % 4),         # shaft width variation
            headwidth=3 + style_idx % 3,                   # head shape
            headlength=4 + style_idx % 2,
            alpha=0.7 + 0.1 * (style_idx % 3),             # slight transparency variation
            label=f'Eigenvec {eigvec_index} (λ={eigval:.2f})'
        )

        color_count += 1

    ax.set_xlim([0, W])
    ax.set_ylim([0, H])
    ax.set_aspect('equal')
    ax.set_title("Gradient Policy from Multiple Eigenvectors")
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small', borderaxespad=0.)

    plt.tight_layout()

    save_path = f"plots/{env_name}/{seed_num}/combined_policy_ckpts_{'_'.join(map(str, ckpt_nums))}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved combined policy plot to {save_path}")


# === Main ===
if __name__ == "__main__":
    seed_num = 32
    #seed_num = 52
    num_episodes = 1
    

    env_name = 'point_Spiral11x11'
    ckpt_num = 77
    
    env_name = 'point_FourRooms' # 'point_Spiral11x11'
    ckpt_num = 250
    


    generate_and_save_episodes(
        env_name=env_name,
        ckpt_num=ckpt_num,
        seed_num=seed_num,
        num_episodes=num_episodes,
        plot=True
    )

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

