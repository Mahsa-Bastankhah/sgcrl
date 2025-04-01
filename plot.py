import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re

# Paths
DATA_DIR = "./data"
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_simulation_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def compute_q_mean_std(sim_data):
    max_len = max(len(ep) for ep in sim_data["episodes"])
    all_qs = [[] for _ in range(max_len)]

    for episode in sim_data["episodes"]:
        for t, step in enumerate(episode):
            all_qs[t].append(step["q_value"])

    q_means = np.array([np.mean(qs) if qs else np.nan for qs in all_qs])
    q_stds = np.array([np.std(qs) if qs else np.nan for qs in all_qs])

    return q_means, q_stds

def plot_q_single_checkpoint(ckpt_num, data_dir=DATA_DIR, save_dir=PLOT_DIR):
    filename = f"checkpoint_{ckpt_num}_simulation_data.pkl"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Checkpoint file '{filename}' not found in {data_dir}")
        return

    sim_data = load_simulation_data(filepath)
    q_means, q_stds = compute_q_mean_std(sim_data)

    timesteps = np.arange(len(q_means))
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, q_means, label=f"Checkpoint {ckpt_num}")
    plt.fill_between(timesteps, q_means - q_stds, q_means + q_stds, alpha=0.3)

    plt.xlabel("Timestep")
    plt.ylabel("Average Q-value")
    plt.title(f"Average Q-value Over Time (Checkpoint {ckpt_num})")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    save_path = os.path.join(save_dir, f"avg_q_ckpt_{ckpt_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")



def compute_grepr_magnitude_stats(sim_data):
    max_len = max(len(ep) for ep in sim_data["episodes"])
    grepr_norms = [[] for _ in range(max_len)]
    gstate_norms = [[] for _ in range(max_len)]

    for episode in sim_data["episodes"]:
        for t, step in enumerate(episode):
            grepr = step["g_repr"]
            gstate = step["g_repr_state"]
            grepr_norms[t].append(np.linalg.norm(grepr))
            gstate_norms[t].append(np.linalg.norm(gstate))

    grepr_means = np.array([np.mean(n) if n else np.nan for n in grepr_norms])
    grepr_stds = np.array([np.std(n) if n else np.nan for n in grepr_norms])
    gstate_means = np.array([np.mean(n) if n else np.nan for n in gstate_norms])
    gstate_stds = np.array([np.std(n) if n else np.nan for n in gstate_norms])

    return grepr_means, grepr_stds, gstate_means, gstate_stds



def compute_grepr_dot_stats(sim_data):
    max_len = max(len(ep) for ep in sim_data["episodes"])
    dot_products = [[] for _ in range(max_len)]

    for episode in sim_data["episodes"]:
        for t, step in enumerate(episode):
            grepr = step["g_repr"]
            gstate = step["g_repr_state"]
            dot = np.dot(grepr, gstate)
            dot_products[t].append(dot)

    dot_means = np.array([np.mean(dots) if dots else np.nan for dots in dot_products])
    dot_stds = np.array([np.std(dots) if dots else np.nan for dots in dot_products])

    return dot_means, dot_stds


def plot_grepr_dot_product(ckpt_num, data_dir=DATA_DIR, save_dir=PLOT_DIR):
    filename = f"checkpoint_{ckpt_num}_simulation_data.pkl"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Checkpoint file '{filename}' not found in {data_dir}")
        return

    sim_data = load_simulation_data(filepath)
    dot_means, dot_stds = compute_grepr_dot_stats(sim_data)

    timesteps = np.arange(len(dot_means))
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, dot_means, label="Dot(g_repr, g_repr_state)")
    plt.fill_between(timesteps, dot_means - dot_stds, dot_means + dot_stds, alpha=0.3)

    plt.xlabel("Timestep")
    plt.ylabel("Dot Product")
    plt.title(f"Dot Product of Goal vs State-as-Goal Representations (Ckpt {ckpt_num})")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    save_path = os.path.join(save_dir, f"dot_grepr_ckpt_{ckpt_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

# if __name__ == "__main__":
#     # Example usage: change to your desired checkpoint
#     plot_q_single_checkpoint(ckpt_num=4)
#     plot_grepr_dot_product(ckpt_num=4)

def compute_reward_mean_std(sim_data):
    max_len = max(len(ep) for ep in sim_data["episodes"])
    all_rewards = [[] for _ in range(max_len)]

    for episode in sim_data["episodes"]:
        for t, step in enumerate(episode):
            if step["reward"] is not None:
                all_rewards[t].append(step["reward"])
                if step["reward"] < 0:
                    print(step["reward"])

    reward_means = np.array([np.mean(r) if r else np.nan for r in all_rewards])
    reward_stds = np.array([np.std(r) if r else np.nan for r in all_rewards])

    return reward_means, reward_stds

def plot_reward_single_checkpoint(ckpt_num, data_dir=DATA_DIR, save_dir=PLOT_DIR):
    filename = f"checkpoint_{ckpt_num}_simulation_data.pkl"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Checkpoint file '{filename}' not found in {data_dir}")
        return

    sim_data = load_simulation_data(filepath)
    reward_means, reward_stds = compute_reward_mean_std(sim_data)

    timesteps = np.arange(len(reward_means))
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, reward_means, label="Average Reward")
    #plt.fill_between(timesteps, reward_means - reward_stds, reward_means + reward_stds, alpha=0.3)

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.ylim(0, 1)
    plt.title(f"Average Reward Over Time (Checkpoint {ckpt_num})")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    save_path = os.path.join(save_dir, f"avg_reward_ckpt_{ckpt_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Reward plot saved to {save_path}")





def check_grepr_consistency(ckpt_num, data_dir="./data"):
    filename = f"checkpoint_{ckpt_num}_simulation_data.pkl"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Checkpoint file '{filename}' not found.")
        return False

    sim_data = load_simulation_data(filepath)

    reference_grepr = None
    all_equal = True

    for epi_idx, episode in enumerate(sim_data["episodes"]):
        for step_idx, step in enumerate(episode):
            grepr = step["g_repr"]
            if reference_grepr is None:
                reference_grepr = grepr
            else:
                if not np.allclose(grepr, reference_grepr):
                    print(f"Inconsistency found at episode {epi_idx}, step {step_idx}")
                    all_equal = False
                    break
        if not all_equal:
            break

    if all_equal:
        print(f"✅ All g_repr values are identical in checkpoint {ckpt_num}.")
    else:
        print(f"❌ Inconsistent g_repr values found in checkpoint {ckpt_num}.")

    return all_equal



def compute_observation_variance(sim_data, num_coords=7):
    max_len = max(len(ep) for ep in sim_data["episodes"])
    coord_values = [ [[] for _ in range(max_len)] for _ in range(num_coords) ]

    for episode in sim_data["episodes"]:
        for t, step in enumerate(episode):
            obs = step["observation"][:num_coords]
            for i in range(num_coords):
                coord_values[i][t].append(obs[i])

    variances = np.full((num_coords, max_len), np.nan)
    for i in range(num_coords):
        for t in range(max_len):
            values = coord_values[i][t]
            if values:
                variances[i, t] = np.var(values)

    return variances

def plot_observation_variance(ckpt_num, data_dir=DATA_DIR, save_dir=PLOT_DIR, num_coords=7):
    filename = f"checkpoint_{ckpt_num}_simulation_data.pkl"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Checkpoint file '{filename}' not found.")
        return

    sim_data = load_simulation_data(filepath)
    variances = compute_observation_variance(sim_data, num_coords)

    timesteps = np.arange(variances.shape[1])
    plt.figure(figsize=(12, 6))
    for i in range(num_coords):
        plt.plot(timesteps, variances[i], label=f"Obs Coord {i}")

    plt.xlabel("Timestep")
    plt.ylabel("Variance")
    plt.title(f"Variance of First {num_coords} Observation Coordinates (Ckpt {ckpt_num})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"obs_var_ckpt_{ckpt_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved observation variance plot to {save_path}")


def compute_state_goal_distance(sim_data, num_coords=7):
    max_len = max(len(ep) for ep in sim_data["episodes"])
    all_distances = [[] for _ in range(max_len)]

    for episode in sim_data["episodes"]:
        for t, step in enumerate(episode):
            obs = step["observation"]
            state = obs[:num_coords]
            goal = obs[-num_coords:]
            distance = np.linalg.norm(np.array(state) - np.array(goal))
            all_distances[t].append(distance)

    distance_means = np.array([np.mean(d) if d else np.nan for d in all_distances])
    distance_stds = np.array([np.std(d) if d else np.nan for d in all_distances])

    return distance_means, distance_stds

def plot_state_goal_distance(ckpt_num, data_dir=DATA_DIR, save_dir=PLOT_DIR, num_coords=7):
    filename = f"checkpoint_{ckpt_num}_simulation_data.pkl"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Checkpoint file '{filename}' not found.")
        return

    sim_data = load_simulation_data(filepath)
    distance_means, distance_stds = compute_state_goal_distance(sim_data, num_coords)

    timesteps = np.arange(len(distance_means))
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, distance_means, label="Mean Distance to Goal")
    plt.fill_between(timesteps, distance_means - distance_stds,
                     distance_means + distance_stds, alpha=0.3)

    plt.xlabel("Timestep")
    plt.ylabel("Distance to Goal")
    plt.title(f"Mean Distance Between State and Goal (Ckpt {ckpt_num})")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()

    save_path = os.path.join(save_dir, f"state_goal_dist_ckpt_{ckpt_num}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved distance-to-goal plot to {save_path}")



def extract_goal_repr(sim_data):
    """
    Given simulation data, returns the goal representation (g_repr)
    from the first step of the first episode.
    """
    try:
        return np.array(sim_data["episodes"][0][0]["g_repr"])
    except (KeyError, IndexError) as e:
        print("Error extracting g_repr:", e)
        return None

def plot_goal_repr_evolution(data_dir="./data", save_path="./plots/goal_repr_all_dims_over_checkpoints.png"):
    """
    Loads simulation files from data_dir, extracts the goal representation from the first step
    of the first episode in each file, and plots each dimension over checkpoint numbers.
    
    Assumes filenames like 'checkpoint_{ckpt}_simulation_data.pkl'
    """
    # List simulation files that match our pattern
    file_pattern = re.compile(r"checkpoint_(\d+)_simulation_data\.pkl")
    files = [f for f in os.listdir(data_dir) if file_pattern.match(f)]
    
    if not files:
        raise FileNotFoundError(f"No simulation files found in {data_dir} matching the pattern.")
    
    # Extract checkpoint numbers and corresponding goal representations
    ckpt_goal_list = []
    for fname in files:
        match = file_pattern.match(fname)
        if match:
            ckpt_num = int(match.group(1))
            filepath = os.path.join(data_dir, fname)
            sim_data = load_simulation_data(filepath)
            g_repr = extract_goal_repr(sim_data)
            if g_repr is not None:
                ckpt_goal_list.append((ckpt_num, g_repr))
            else:
                print(f"Warning: No g_repr found in {fname}.")
    
    if not ckpt_goal_list:
        print("No valid goal representations found.")
        return
    
    # Sort by checkpoint number
    ckpt_goal_list.sort(key=lambda x: x[0])
    checkpoint_numbers = [item[0] for item in ckpt_goal_list]
    # Assume all g_repr vectors have the same dimension
    goal_matrix = np.array([item[1] for item in ckpt_goal_list])  # shape: (num_ckpts, repr_dim)

    # Create the plots
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(14, 6))
    for dim in range(goal_matrix.shape[1]):
        plt.plot(checkpoint_numbers, goal_matrix[:, dim], label=f"Dim {dim}", alpha=0.6)
    
    plt.xlabel("Checkpoint")
    plt.ylabel("Goal Representation Value")
    plt.title("Evolution of Each Goal Representation Dimension Over Checkpoints")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Plot saved to '{save_path}'")

def load_all_goal_reprs(data_dir=DATA_DIR, file_pattern=r"checkpoint_(\d+)_simulation_data\.pkl"):
    """
    Loads simulation files from data_dir matching the file_pattern,
    extracts the goal representation from the first step of the first episode in each file,
    and returns sorted checkpoint numbers and a matrix of goal representations.
    """
    pattern = re.compile(file_pattern)
    ckpt_goal_list = []
    
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            ckpt_num = int(match.group(1))
            filepath = os.path.join(data_dir, fname)
            sim_data = load_simulation_data(filepath)
            g_repr = extract_goal_repr(sim_data)
            if g_repr is not None:
                ckpt_goal_list.append((ckpt_num, g_repr))
            else:
                print(f"Warning: No g_repr found in {fname}.")
                
    if not ckpt_goal_list:
        raise ValueError("No valid goal representations found.")
        
    # Sort by checkpoint number
    ckpt_goal_list.sort(key=lambda x: x[0])
    ckpt_nums = [item[0] for item in ckpt_goal_list]
    goal_matrix = np.array([item[1] for item in ckpt_goal_list])  # shape: (num_ckpts, repr_dim)
    return ckpt_nums, goal_matrix

def plot_adjacent_goal_distance(data_dir=DATA_DIR, 
                                save_path=os.path.join(PLOT_DIR, "goal_repr_adjacent_distance.png")):
    """
    Computes the Euclidean distance between adjacent goal representation vectors across checkpoints
    and plots the distance over time. The x-axis is set to the checkpoint number corresponding
    to the latter of each pair.
    """
    ckpt_nums, goal_matrix = load_all_goal_reprs(data_dir=data_dir)
    
    # Compute Euclidean distances between adjacent goal representations
    adjacent_distances = []
    x_axis = []
    for i in range(len(goal_matrix) - 1):
        dist = np.linalg.norm(goal_matrix[i+1] - goal_matrix[i])
        adjacent_distances.append(dist)
        # x-axis: we can use the checkpoint number of the later one in the pair
        x_axis.append(ckpt_nums[i+1])
    
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, adjacent_distances, marker='o', label="Adjacent Goal Distances")
    plt.xlabel("Checkpoint")
    plt.ylabel("Euclidean Distance")
    plt.title("Distance Between Adjacent Goal Representations Over Checkpoints")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"✅ Plot saved to '{save_path}'")


def compute_q_reward1_mean_std(sim_data):
    """
    For the given simulation data, computes the average and standard deviation
    of Q-values for steps where reward equals 1.
    
    Args:
        sim_data: Dictionary containing simulation data with a key "episodes",
                  where each episode is a list of step dictionaries.
                  
    Returns:
        q_mean: Average Q-value over all steps with reward == 1.
        q_std: Standard deviation of Q-values over those steps.
    """
    q_vals = []
    for episode in sim_data["episodes"]:
        for step in episode:
            if step.get("reward", 0) == 1:
                q_vals.append(step["q_value"])
    if q_vals:
        q_mean = np.mean(q_vals)
        q_std = np.std(q_vals)
    else:
        q_mean, q_std = np.nan, np.nan
    return q_mean, q_std


def plot_avg_q_reward1_over_checkpoints(ckpt_nums, data_dir=DATA_DIR, save_dir=PLOT_DIR):
    """
    For each checkpoint in ckpt_nums, load the simulation data and compute the average
    Q-value (and std) for steps with reward 1. Then plot these values over the checkpoints.
    
    Args:
        ckpt_nums: List of checkpoint numbers (e.g. [3, 6, 9, ...])
        data_dir: Directory where simulation pickle files are stored.
        save_dir: Directory where the resulting plot image will be saved.
    """
    means = []
    stds = []
    valid_ckpts = []
    for ckpt_num in ckpt_nums:
        filename = f"checkpoint_{ckpt_num}_simulation_data.pkl"
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Checkpoint file '{filename}' not found in {data_dir}")
            continue
        sim_data = load_simulation_data(filepath)
        q_mean, q_std = compute_q_reward1_mean_std(sim_data)
        valid_ckpts.append(ckpt_num)
        means.append(q_mean)
        stds.append(q_std)
    
    if not valid_ckpts:
        print("No valid checkpoints found to plot.")
        return
    
    valid_ckpts = np.array(valid_ckpts)
    means = np.array(means)
    stds = np.array(stds)
    
    plt.figure(figsize=(10, 5))
    plt.errorbar(valid_ckpts, means, yerr=stds, fmt='-o', capsize=5, label="Avg Q (reward==1)")
    plt.xlabel("Checkpoint")
    plt.ylabel("Average Q-value (reward==1)")
    plt.title("Average Q-value for Steps with Reward==1 Over Checkpoints")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "avg_q_reward1_over_checkpoints.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    from pathlib import Path

    DATA_DIR = "./data"
    existing_files = [f.name for f in Path(DATA_DIR).glob("checkpoint_*.pkl")]
    existing_ckpts = {
        int(f.split("_")[1]) for f in existing_files if f.split("_")[1].isdigit()
    }

    for ckpt_num in sorted(existing_ckpts):
        if ckpt_num % 3 != 0  :
           continue
        print(f"Plotting for checkpoint {ckpt_num}...")
        try:
            #plot_q_single_checkpoint(ckpt_num=ckpt_num)
            #plot_grepr_dot_product(ckpt_num=ckpt_num)
            #plot_reward_single_checkpoint(ckpt_num=ckpt_num)
            #check_grepr_consistency(ckpt_num=3)
            #plot_observation_variance(ckpt_num=ckpt_num)
            #plot_state_goal_distance(ckpt_num=ckpt_num)
            #plot_state_repr_dimensions(ckpt_num=ckpt_num)
            x = 1

        except Exception as e:
            print(f"Error in checkpoint {ckpt_num}: {e}")

    #plot_goal_repr_evolution(data_dir="./data", 
    #                         save_path="./plots/goal_repr_all_dims_over_checkpoints.png")
    #plot_adjacent_goal_distance()
    # Example: plot for checkpoints that are multiples of 3
    ckpt_nums = [ckpt for ckpt in range(1, 160) if ckpt % 3 == 0]
    plot_avg_q_reward1_over_checkpoints(ckpt_nums)
