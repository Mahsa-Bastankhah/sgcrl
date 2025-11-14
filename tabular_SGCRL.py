"""
Tabular Contrastive RL in a Maze Environment
==========================================
Author: Mahsa Bastankhah

A tabular implementation of contrastive RL for maze navigation environments.
The agent learns state representations (psi) and state-action representations (phi)
through contrastive learning objectives.

Key Features:
- Tabular representations (no neural networks)
- Multiple maze environments (FourRooms, Spiral, etc.)
- Contrastive learning with configurable loss modes
- Trajectory cutting and psi filtering options
- Comprehensive visualization and logging
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import json
import time
import os
import re
from pathlib import Path
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import trange
from collections import deque 
import pandas as pd
import matplotlib as mpl

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================
mpl.rcParams.update({
    "text.usetex": False,                      # No LaTeX rendering
    "axes.labelsize": 24,
    "axes.titlesize": 18,
    "legend.fontsize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.dpi": 200,
    "pdf.fonttype": 42,                        # Embed TrueType fonts in PDFs
    "ps.fonttype": 42,
})

# ============================================================================
# ENVIRONMENT STEP FUNCTION
# ============================================================================
def step(state: int, action: int) -> int:
    """
    Execute one step in the maze environment.
    
    Args:
        state: Current state (flattened index)
        action: Action to take (0=stay, 1=down, 2=up, 3=right, 4=left)
    
    Returns:
        New state after taking the action (blocked moves return current state)
    """
    di, dj = A_TO_DELTA[action]
    i, j = np.unravel_index(state, walls.shape)
    ni, nj = i + di, j + dj
    
    # Check bounds and walls
    if 0 <= ni < HEIGHT and 0 <= nj < WIDTH and walls[ni, nj] == 0:
        return np.ravel_multi_index((ni, nj), walls.shape)
    return state  # Blocked: stay in place

# ============================================================================
# ARGUMENT PARSING
# ============================================================================






def parse_args():
    """Parse command line arguments for the tabular SGCRL experiment."""
    p = argparse.ArgumentParser(description="Tabular Contrastive RL for Maze Navigation")
    
    # Core hyperparameters
    p.add_argument("--rep-dim",          type=int,   default=16,      help="Representation dimension")
    p.add_argument("--epsilon",          type=float, default=0.01,    help="Epsilon for epsilon-greedy exploration")
    p.add_argument("--episodes-per-upd", type=int,   default=5,       help="Episodes between representation updates")
    p.add_argument("--max-steps",        type=int,   default=100,     help="Maximum steps per episode")
    p.add_argument("--continuous-episode", type=bool, default=False,
                   help="If True, keep rolling out a single continuous episode and only chunk it for replay storage")
    p.add_argument("--batch-size",       type=int,   default=128,     help="Batch size for contrastive updates")
    p.add_argument("--lr-phi-psi",       type=float, default=1e-3,    help="Learning rate for phi and psi updates")
    p.add_argument("--num-episodes",     type=int,   default=50_000,  help="Total number of episodes to run")
    p.add_argument("--gamma",            type=float, default=0.99,    help="Discount factor for future state sampling")
    p.add_argument("--seed",             type=int,   default=1,       help="Random seed")
    p.add_argument("--replay-capacity",  type=int,   default=1000,    help="Number of trajectories to keep in replay buffer")
    
    # Representation initialization
    p.add_argument("--near-var",         type=float, default=0.5,     help="Variance for states near goal during initialization")
    p.add_argument("--far-var",          type=float, default=10.0,    help="Variance for states far from goal during initialization")
    p.add_argument("--alpha",            type=float, default=5.0,     help="Temperature parameter for softmax action selection")
    p.add_argument("--init_mode",        type=str,   default="uniform_around_goal", 
                   choices=["distance_aware", "uniform_around_goal", "fully_random"],
                   help="Method for initializing psi representations")
    
    # Environment and goal settings
    p.add_argument("--env",              type=str,   default="fourRooms10", 
                   choices=["fourRooms10", "spiral", "impossible_maze", "fourRooms20", "box"],
                   help="Environment type")
    p.add_argument("--two_goals",        type=bool,  default=False,   help="Use two goals instead of one")
    p.add_argument("--random_exploration", type=bool, default=False,  help="Use a random goal representation for exploration, the goal remains consistent throughout training")
    
    # Training settings
    p.add_argument("--loss_mode",        type=str,   default="backward", help="Loss mode for contrastive learning")
    p.add_argument("--entropy_coeff",    type=float, default=0.1,     help="Entropy coefficient for action selection")
    p.add_argument("--verbose",          type=bool,  default=False,   help="Enable verbose logging")
    
    # Experimental features
    p.add_argument("--relative_sim_factor", type=bool, default=False, 
                   help="Use relative similarity factor in softmax action selection")
    p.add_argument("--optimality_exp_mode", type=str,  default="deactive", 
                   choices=["sgcrl", "random_goal", "deactive"], help="Optimality experiment mode")
    
    # Optimality Experiment: Tests data collection efficiency when states are initialized 
    # orthogonal to the goal representation. This experiment evaluates whether SGCRL 
    # maintains optimal data collection under challenging initialization conditions.
    #
    # Modes:
    #   - "deactive": Experiment disabled (default)
    #   - "sgcrl": Run experiment using SGCRL algorithm for data collection
    #   - "random_goal": Run experiment using random goal exploration as baseline
    #
    # Results from this experiment are reported in Appendix D5 of the paper.
    
    # Psi similarity filtering parameters (don't update the representations if both the anchor and future state are too similar to the initial state)
    p.add_argument("--psi_sim_threshold",   type=float, default=0.5, 
                   help="Maximum psi similarity with initial state - filter pairs where both states exceed this")
    p.add_argument("--enable_psi_filtering", type=bool, default=False, 
                   help="Enable filtering states based on psi similarity with initial state")
    
    return p.parse_args()

# ============================================================================
# HYPERPARAMETER SETUP
# ============================================================================
args = parse_args()

# Core training parameters
rep_dim          = args.rep_dim
epsilon          = args.epsilon
episodes_per_upd = args.episodes_per_upd
max_steps        = args.max_steps
batch_size       = args.batch_size
lr_phi_psi       = args.lr_phi_psi
num_episodes     = args.num_episodes
gamma            = args.gamma
seed             = args.seed
replay_capacity  = args.replay_capacity
continuous_episode = args.continuous_episode

# Representation initialization parameters
near_var         = args.near_var
far_var          = args.far_var
init_mode        = args.init_mode
alpha            = args.alpha

# Environment and goal parameters
env              = args.env
two_goals        = args.two_goals
random_exploration = args.random_exploration

# Training configuration
entropy_coeff    = args.entropy_coeff
loss_mode        = args.loss_mode
verbose          = args.verbose

# Experimental features
relavce_sim_factor = args.relative_sim_factor
optimality_exp_mode = args.optimality_exp_mode

# Psi similarity filtering parameters
psi_sim_threshold    = args.psi_sim_threshold
enable_psi_filtering = args.enable_psi_filtering

# Derived parameters
plot_mult = max(num_episodes // 100, 1)  # Plot ~100 times during training

# Fixed configuration (can be made arguments if needed)
norm = True                               # Whether to normalize representations
rer_order_exp = False                     # Reachability order experiment

# Set random seed
np.random.seed(seed)

# ============================================================================
# EXPERIMENT CONFIGURATION AND LOGGING SETUP
# ============================================================================
# Create configuration dictionary for saving and logging
config = {
    # Core parameters
    "rep_dim": rep_dim,
    "episodes_per_upd": episodes_per_upd,
    "max_steps": max_steps,
    "batch_size": batch_size,
    "lr_phi_psi": lr_phi_psi,
    "num_episodes": num_episodes,
    "gamma": gamma,
    "seed": seed,
    "replay_capacity": replay_capacity,
    "continuous_episode": continuous_episode,
    
    # Representation parameters
    "near_var": near_var,
    "far_var": far_var,
    "alpha": alpha,
    "init_mode": init_mode,
    "norm": norm,
    
    # Environment parameters
    "env": env,
    "two_goals": two_goals,
    "random_exploration": random_exploration,
    
    # Training parameters
    "loss_mode": loss_mode,
    "entropy_coeff": entropy_coeff,
    "plot_mult": plot_mult,
    
    # Experimental features
    "relative_sim_factor": relavce_sim_factor,
    "optimality_exp_mode": optimality_exp_mode,
    "rer_order_exp": rer_order_exp,
    
    # Psi filtering
    "psi_sim_threshold": psi_sim_threshold,
    "enable_psi_filtering": enable_psi_filtering,
}


print(f"[INFO] Using environment: {env}")
print(f"[INFO] Using seed: {seed}")

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
if "fourRooms" in env:
    # Parse size from environment name (e.g., "fourRooms10" -> size=10)
    match = re.search(r"fourRooms(\d+)", env)
    if match:
        size = int(match.group(1))
    else:
        raise ValueError(f"Could not determine size from env name: {env}")
    
    # Four Rooms Environment Construction
    HEIGHT, WIDTH = size, size
    GOAL_COORD = (size - 1, size - 1)  # Bottom-right corner
    walls = np.zeros((HEIGHT, WIDTH), dtype=int)
    DOOR_LEN = 2   
    
    # Create horizontal wall with doors
    walls[HEIGHT // 2, :] = 1
    doors_h = np.concatenate([
        WIDTH // 4 + np.arange(DOOR_LEN),
        WIDTH * 3 // 4 + np.arange(DOOR_LEN)
    ])
    walls[HEIGHT // 2, doors_h] = 0

    # Create vertical wall with doors
    walls[:, WIDTH // 2] = 1
    doors_v = np.concatenate([
        HEIGHT // 4 + np.arange(DOOR_LEN),
        HEIGHT * 3 // 4 + np.arange(DOOR_LEN)
    ])
    walls[doors_v, WIDTH // 2] = 0

    empty_states = np.where(walls.flatten() == 0)[0]



    NUM_STATES     = HEIGHT * WIDTH
    NUM_ACTIONS    = 5           # stay, down, up, right, left
    A_TO_DELTA     = np.array([[0, 0],
                            [1, 0], [-1, 0],
                            [0, 1], [0, -1]])
    # NUM_ACTIONS    = 4           # stay, down, up, right, left
    # A_TO_DELTA     = np.array([[1, 0], [-1, 0],
    #                            [0, 1], [0, -1]])
    START_STATE = np.ravel_multi_index((0, 0), walls.shape) 

elif env == "box":
    HEIGHT, WIDTH = 5, 5
    GOAL_COORD = (4, 4)  # zero-indexed bottom-right corner
    walls = np.zeros((HEIGHT, WIDTH), dtype=int)  # no walls

    empty_states = np.where(walls.flatten() == 0)[0]

    NUM_STATES     = HEIGHT * WIDTH
    NUM_ACTIONS    = 5           # stay, down, up, right, left
    A_TO_DELTA     = np.array([[0, 0],
                            [1, 0], [-1, 0],
                            [0, 1], [0, -1]])
    # NUM_ACTIONS    = 4           # stay, down, up, right, left
    # A_TO_DELTA     = np.array([[1, 0], [-1, 0],
    #                            [0, 1], [0, -1]])
    START_STATE = np.ravel_multi_index((0, 0), walls.shape) 

elif env == "spiral":
    # -------- Maze construction: spiral pattern -------------------------------
    HEIGHT = WIDTH = 11
    GOAL_COORD = (10, 10)               # bottom-right (zero-indexed)

    # 1 = wall, 0 = free space
    walls = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                      [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
                      [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
                      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                      [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                      [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
                      [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    # ---------------------------------------------------------------------------
    empty_states = np.where(walls.flatten() == 0)[0]
    NUM_STATES   = HEIGHT * WIDTH
    NUM_ACTIONS  = 5                     # stay, down, up, right, left
    A_TO_DELTA   = np.array([[0, 0],
                             [1, 0], [-1, 0],
                             [0, 1], [0, -1]])
    START_STATE = np.ravel_multi_index((5, 5), walls.shape) 

elif env == "impossible_maze":
    walls = np.array([
    [0,1,0,0,0,0,0,0,0],
    [0,1,0,1,1,1,1,1,0],
    [0,1,0,0,0,0,1,0,0],
    [0,1,1,1,1,0,1,0,1],
    [0,1,0,0,0,0,1,0,0],
    [0,1,0,1,1,1,1,1,0],
    [0,0,0,1,0,0,0,1,0],
    [0,1,0,1,0,1,0,1,1],
    [0,1,0,0,0,1,0,1,0]
    ], dtype=np.int8)

    HEIGHT = WIDTH = 9
    GOAL_COORD = (6, 8)  # (row, col)

    k = 3  # scale factor

    wide_walls = np.repeat(np.repeat(walls, k, axis=0), k, axis=1)



    # Update goal to the center of the expanded block corresponding to the old goal
    GOAL_COORD = (GOAL_COORD[0]*k + k//2, GOAL_COORD[1]*k + k//2)
    walls = wide_walls
    HEIGHT, WIDTH = walls.shape



    # ---------------------------------------------------------------------------
    empty_states = np.where(walls.flatten() == 0)[0]
    NUM_STATES   = HEIGHT * WIDTH
    NUM_ACTIONS  = 5                     # stay, down, up, right, left
    A_TO_DELTA   = np.array([[0,0],
                             [1, 0], [-1, 0],
                             [0, 1], [0, -1]])
    START_STATE = np.ravel_multi_index((0, 0), walls.shape) 

else:
    raise ValueError(f"Unknown environment: {env}. Choose from fourRooms10, spiral, impossible_maze, fourRooms20, box.")



current_env_state = START_STATE  # used when rolling out continuous episodes


# NEW: helper to make a unit vector orthogonal to a given vector
def orthogonal_unit(vec, rng=np.random):
    v = vec.astype(float)
    n2 = float(v @ v)
    if n2 < 1e-12:
        u = np.zeros_like(v); u[0] = 1.0
        return u
    for _ in range(8):
        r = rng.randn(v.size)
        u = r - (r @ v) / n2 * v
        nu = np.linalg.norm(u)
        if nu > 1e-8:
            return u / nu
    # deterministic fallback
    i = int(np.argmax(np.abs(v)))
    e = np.zeros_like(v); e[(i + 1) % v.size] = 1.0
    u = e - (e @ v) / n2 * v
    return u / (np.linalg.norm(u) + 1e-12)


# Create a stable UID *excluding the seed* so same hyperparams with different seeds
#    land in parallel subfolders under their own seed directory.
hash_basis = {k: v for k, v in config.items() if k != "seed"}
hash_json  = json.dumps(hash_basis, sort_keys=True).encode()
uid = hashlib.sha1(hash_json).hexdigest()[:10]      # first 10 hex chars

# Build run directory: runs/seed{seed}/{uid}/
run_dir = Path(f"runs_final/seed{seed}") / uid
run_dir.mkdir(parents=True, exist_ok=True)

# Add timestamp then save full config (including seed & uid & hash_basis copy)
config["uid"] = uid
config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

with open(run_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"[INFO] Run directory: {run_dir}")
print(f"[INFO] Config UID: {uid}")

with open(f"{run_dir}/index.csv", "a") as idx:
    if idx.tell() == 0:
        idx.write("uid,seed,rep_dim,entropy_coeff,episodes_per_upd,lr_phi_psi,near_var,far_var,alpha,plot_mult,timestamp\n")
    idx.write(f"{uid},{seed},{rep_dim},{entropy_coeff},{episodes_per_upd},{lr_phi_psi},{near_var},{far_var},{alpha},{plot_mult},{config['timestamp']}\n")


# -------- Tabular parameters -------------------------------------------------

# two goal experimebt is only suported for fourRooms10 env
SECOND_GOAL_COORD = (9, 6)

goal1 = np.ravel_multi_index(GOAL_COORD, walls.shape)
goal2 = np.ravel_multi_index(SECOND_GOAL_COORD, walls.shape) if two_goals else goal1

# keep `goal` for backward compatibility
goal = goal1


# --- NEW: distance-aware ψ initialisation -----------------------------------
coords      = np.column_stack(np.unravel_index(np.arange(NUM_STATES), walls.shape))
goal_coord  = coords[goal]                     # (row, col) of the goal state
max_dist    = np.linalg.norm([HEIGHT - 1, WIDTH - 1])  # farthest possible corner

psi         = np.empty((NUM_STATES, rep_dim))
psi_goal    = np.random.randn(rep_dim) * 0.1             # anchor embedding
psi[goal]   = psi_goal  
if norm :
    psi_norm = np.linalg.norm(psi[goal]) + 1e-8
    psi[goal]  /= psi_norm
    psi_goal /= psi_norm

if two_goals:
    psi_goal2 = np.random.randn(rep_dim) * 0.1             # anchor embedding for second goal
    psi_norm = np.linalg.norm(psi_goal2) + 1e-8
    psi[goal2] = psi_goal2 / psi_norm
    psi_goal2 /= psi_norm
    twin_goal_psi = 0.5 * psi_goal + 0.5 * psi_goal2




                                 # set goal’s ψ once
initialization_for_other_states = np.random.randn(rep_dim) * 0.1             # anchor embedding
## initialize representations in a distance-aware manner
if two_goals is False and init_mode == "distance_aware":
    print("[INFO] Initializing ψ(s) distance-aware...")
    for s, c in enumerate(coords):
        if s == goal:
            continue
        d         = np.linalg.norm(c - goal_coord)          # L2 to goal
        frac      = (d / max_dist) ** alpha                # 0 near goal → 1 far away
        var_scale = near_var + (far_var - near_var) * frac # extreme scaling
        #print(f"State {s} at {c}: dist={d:.2f}, frac={frac:.2f}, var_scale={var_scale:.2f}")
        psi[s]    = psi_goal + np.random.randn(rep_dim) * 0.1 * var_scale



elif two_goals is False and init_mode == "uniform_around_goal":
    print("[INFO] Initializing ψ(s) uniformly around ψ(goal)...")
    for s, c in enumerate(coords):
        if s == goal:
            continue
        psi[s]    =  psi_goal + np.random.randn(rep_dim) * 0.1 *  near_var
    

    # --- SPECIAL OVERRIDE: 2x2 room centers orthogonal to ψ(goal) ---
    if rer_order_exp and env == "fourRooms10":
        # centers of the four rooms for size=10:
        # rooms are split by row=5 and col=5 (0-indexed). Quarter centers near (2,2), (2,7), (7,2), (7,7).
        centers = [(2, 2), (2, 7), (7, 2), (7, 7)]
        # build 2x2 squares starting at each center (i,j) -> cells {(i,j),(i+1,j),(i,j+1),(i+1,j+1)}
        squares = []
        for (ci, cj) in centers:
            cells = [(ci, cj), (ci+1, cj), (ci, cj+1), (ci+1, cj+1)]
            # keep only free cells inside the maze
            cells = [(i, j) for (i, j) in cells
                     if 0 <= i < HEIGHT and 0 <= j < WIDTH and walls[i, j] == 0]
            squares.extend(cells)

        # a unit vector orthogonal to ψ(goal)
        u_perp = orthogonal_unit(psi_goal)

        # override those states with u_perp + small noise
        for (i, j) in squares:
            s = np.ravel_multi_index((i, j), walls.shape)
            if s == goal:
                continue
            vec = -1 * psi_goal + np.random.randn(rep_dim) * 0.1 * near_var
            if norm:
                vec = vec / (np.linalg.norm(vec) + 1e-8)
            psi[s] = vec


elif two_goals is False and init_mode == "fully_random":
    print("[INFO] Initializing ψ(s) uniformly at random...")
    for s, c in enumerate(coords):
        if s == goal:
            continue
        psi[s] = np.random.randn(rep_dim) * 0.1





elif two_goals is True:
    print("[INFO] Initializing ψ(s) distance-aware with two goals...")
    for s, c in enumerate(coords):
        if s == goal:
            continue
        psi[s]   =  psi_goal + np.random.randn(rep_dim) * 0.1 * near_var



else:
    print("specification of `two_goals` and `distance_aware_init` and `random init` is not supported for initialization")
    raise ValueError("Unsupported combination of two_goals and distance_aware_init")


# Normalize ψ to unit vectors (L2 norm)
if norm :
    psi_norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-8
    psi /= psi_norms
    



# NEW: room split and orthogonal anchor
mid_r, mid_c = HEIGHT // 2, WIDTH // 2


if random_exploration:
    random_goal_psi = np.random.randn(rep_dim) * 0.1
    random_goal_psi /= np.linalg.norm(random_goal_psi) + 1e-8
    perp_anchor = orthogonal_unit(random_goal_psi) 
    print(f"[INFO] Using random exploration with ψ(goal) = {random_goal_psi[:5]}...")
    if init_mode == "fully_random":
        print("[INFO] Initializing ψ(s) uniformly at random...")
        for s, c in enumerate(coords):
            psi[s] = np.random.randn(rep_dim) * 0.1 
            
    elif init_mode == "uniform_around_goal":
        for s, (r ,c) in enumerate(coords):
            var_scale = near_var 
            print(f"State {s} var_scale={var_scale:.2f}")
            psi[s]    =  random_goal_psi + np.random.randn(rep_dim) * 0.1 * var_scale
            if optimality_exp_mode !="deactive" and env == "fourRooms10":
                # special case for the fourRooms10 experiment where I initialize some states to be orthogonal to the goal
                in_top_left = (7 <= r <= 9) and (0 <= c <= 3)
                in_bottom_right = (0 <= r <= 2) and (6 <= c <= 9)


                base = perp_anchor if (in_top_left or in_bottom_right) else random_goal_psi
                psi[s] = base + np.random.randn(rep_dim) * 0.1 * var_scale

            
    else:
        raise ValueError("Unsupported combination of random_exploration and initialization mode")




def select_action(s: int, g: int) -> int:
    """
    Softmax action selection based on ψ(s') · ψ(g) similarity,
    scaled by 1 / entropy_coeff (i.e., inverse temperature)
    """
    goal_vec = psi[g]  # (rep_dim,)
    #goal_norm = np.linalg.norm(goal_vec) + 1e-8
    if random_exploration:
        goal_vec = random_goal_psi
    if two_goals:
        goal_vec = 0.5 * psi[goal] + 0.5 * psi[goal2]  # average of two goals

    similarities = []
    for a in range(NUM_ACTIONS):
        ns = step(s, a)
        sim = psi[ns] @ goal_vec  # dot similarity (or cosine if normalized)
        if optimality_exp_mode == "random_goal":
            sim = psi[ns] @ np.random.randn(rep_dim) * 0.1
        similarities.append(sim)

    # Apply entropy regularization (inverse temperature)
    logits = np.array(similarities)
    ## the smaller the max(similarity), the larger the inverse temperature, the more uniform
    inverse_temp = 1.0 / entropy_coeff  
    if relavce_sim_factor:
        inverse_temp *= (max(similarities) + 1) / 2
    logits *= inverse_temp

    # Softmax over scaled logits
    exp_logits = np.exp(logits - np.max(logits))  # stability trick
    probs = exp_logits / np.sum(exp_logits)

    return np.random.choice(NUM_ACTIONS, p=probs)

# -------- Replay buffer & data collection -----------------------------------
replay = []       # list of trajectories (each: list[int])
visited_states = set()
visited_counts = [] 

def is_near_goal(s):
    si, sj = np.unravel_index(s, walls.shape)
    gi, gj = np.unravel_index(goal, walls.shape)
    return abs(si - gi) + abs(sj - gj) <= 1


def collect_episode(episode_num=None) -> None:
    """Roll out a trajectory segment and push it into the replay buffer."""
    global current_env_state

    if continuous_episode:
        # keep rolling the long episode without resetting between segments
        start_state = current_env_state
    else:
        start_state = START_STATE
        current_env_state = START_STATE

    step_success = []
    traj = [start_state]
    for _ in range(max_steps):
        a  = select_action(traj[-1], goal)
        ns = step(traj[-1], a)
        traj.append(ns)
        success = (ns == goal) or is_near_goal(ns)
        step_success.append(1 if success else 0)

    segment_final_state = traj[-1]

    replay.append(traj)
    if len(replay) > replay_capacity:      # manual eviction
        replay.pop(0)  

    if continuous_episode:
        current_env_state = segment_final_state

    return step_success   


def eval_action(s: int, g: int) -> int:
    """
    Deterministic evaluation policy: chooses the action with max cosine similarity
    between ψ(s') and ψ(g). No exploration or sampling.
    """
    goal_vec = psi[g]
    goal_norm = np.linalg.norm(goal_vec) + 1e-8

    best_a = 0
    best_val = -np.inf

    for a in range(NUM_ACTIONS):
        ns = step(s, a)
        ns_vec = psi[ns]
        val = ns_vec @ goal_vec 
        if val > best_val:
            best_val = val
            best_a = a

    return best_a

def run_eval_episode() -> None:
    """Generate one trajectory and push into replay buffer."""
    step_success = []
    traj = [START_STATE]
    for _ in range(max_steps):
        a  = eval_action(traj[-1], goal)
        ns = step(traj[-1], a)
        traj.append(ns)
        success = (ns == goal) or is_near_goal(ns)
        step_success.append(1 if success else 0)

    return step_success   


update_cos = []
past_cos = []
update_size = []
anchor_update_size = []
anchor_update_cos = []
pos_update_size = []
pos_update_cos = []




def update_representations(verbose: bool = False) -> float:
    """
    Vectorised ψ-only contrastive update (same objective, much faster).

    For B (anchor, positive) pairs (s_k , sp_k) in the minibatch, let
        D      = ψ(s_j)ᵀ ψ(sp_k)                    # (B×B) dot-product matrix
        P      = softmax_j D                        # column-wise soft-max
        coeff  = I - P                              # (B×B) update coefficients

    Then
        Δψ(s_j)   = η Σ_k coeff[j,k] ψ(sp_k)
        Δψ(sp_k)  = η (ψ(s_k) − Σ_j P[j,k] ψ(s_j))

    The mean negative log-likelihood is  −B⁻¹ Σ_k log P[k,k].
    """

    
    global psi
    global psi_goal

    if verbose:
        psi_0_before = psi[0].copy()

    # ----- 1. Sample minibatch exactly as before ----------------------------
    if len(replay) < 2:
        return 0.0

    s_list, sp_list = [], []
    attempts = 0
    max_attempts = batch_size * 10  # Prevent infinite loops
    
    # Continue sampling until we have a full batch or exceed max attempts
    while len(s_list) < batch_size and attempts < max_attempts:
        # Sample a random trajectory
        idx = np.random.choice(len(replay))
        traj = replay[idx]
        
        if len(traj) < 2:
            attempts += 1
            continue
            
        i = np.random.randint(0, len(traj) - 1)
        remaining = len(traj) - i

        w = gamma ** np.arange(1, remaining)
        w /= w.sum()
        j = i + np.random.choice(np.arange(1, remaining), p=w)

        anchor_state = traj[i]
        future_state = traj[j]
        
        # Apply psi similarity filtering if enabled
        if enable_psi_filtering:
            # Compute psi similarities with initial state
            anchor_sim = np.dot(psi[anchor_state], psi[START_STATE])
            future_sim = np.dot(psi[future_state], psi[START_STATE])
            
            # Skip this pair if both similarities are above threshold (too similar to start)
            if anchor_sim > psi_sim_threshold and future_sim > psi_sim_threshold:
                attempts += 1
                if verbose and attempts <= 5:  # Only show first few rejections to avoid spam
                    print(f"[PSI_FILTER] Rejected pair: anchor_sim={anchor_sim:.3f}, future_sim={future_sim:.3f} > {psi_sim_threshold}")
                continue

        s_list.append(anchor_state)
        sp_list.append(future_state)
        attempts += 1
    
    # Log sampling effectiveness
    if enable_psi_filtering and verbose and attempts > batch_size:
        rejection_rate = 1.0 - (len(s_list) / attempts)
        print(f"[PSI_FILTER] Sampled {len(s_list)}/{batch_size} pairs in {attempts} attempts, rejection rate: {rejection_rate:.3f}")
        print(f"[PSI_FILTER] Filtering pairs where both states have similarity > {psi_sim_threshold} with initial state")

    if not s_list:                        # batch could be empty (very rare with new approach)
        if enable_psi_filtering and verbose:
            print(f"[PSI_FILTER] Warning: Could not find any valid samples after {max_attempts} attempts")
            print(f"[PSI_FILTER] All pairs had both states too similar to initial state (threshold: {psi_sim_threshold})")
        return 0.0
    
    # Log if we couldn't get a full batch
    if len(s_list) < batch_size and enable_psi_filtering and verbose:
        print(f"[PSI_FILTER] Warning: Only got {len(s_list)}/{batch_size} samples after {max_attempts} attempts")

    s_batch  = np.asarray(s_list,  dtype=np.int32)   # anchors (B,)
    sp_batch = np.asarray(sp_list, dtype=np.int32)   # positives (B,)
    B        = len(s_batch)




    # ----- 2. Gather current embeddings -------------------------------------
    psi_s = psi[s_batch]                  # ψ(s_j)      – shape (B,d)
    psi_p = psi[sp_batch]                 # ψ(sp_k)     – shape (B,d)

    # ----- 3. Similarities ---------------------------------------------------
    D = psi_s @ psi_p.T                   # (B,B), rows=j (anchors), cols=k (positives)

    # Backward (columns) softmax: P = softmax_j D[:,k]
    D_col = D - D.max(axis=0, keepdims=True)       # numerically stable per column
    exp_col = np.exp(D_col)
    P = exp_col / (exp_col.sum(axis=0, keepdims=True) + 1e-12)   # (B,B), sum over rows per column = 1

    # Forward (rows) softmax: Q = softmax_k D[j,:]
    D_row = D - D.max(axis=1, keepdims=True)       # numerically stable per row
    exp_row = np.exp(D_row)
    Q = exp_row / (exp_row.sum(axis=1, keepdims=True) + 1e-12)   # (B,B), sum over cols per row = 1

    # Diagonal probs and losses
    diag_P = np.diag(P)                             # backward: p_k(k)
    diag_Q = np.diag(Q)                             # forward:  q_j(j)
    nll_bwd = -np.mean(np.log(diag_P + 1e-12))      # column-CE
    nll_fwd = -np.mean(np.log(diag_Q + 1e-12))      # row-CE

    # Choose which loss to optimize
    #   "backward" -> only columns (your current behavior)
    #   "forward"  -> only rows
    #   "both"     -> symmetric (CLIP-style)


    if loss_mode == "backward":
        nll = nll_bwd
    elif loss_mode == "forward":
        nll = nll_fwd
    else:  # "both"
        nll = 0.5 * (nll_bwd + nll_fwd)

    # ----- 4/5. Vectorized updates ------------------------------------------
    # Backward (column) updates (your existing formulas):
    #   Δψ(s_j)     = η * (I - P) @ ψ(sp)                    (B,d)
    #   Δψ(sp_k)    = η * (ψ(s_k) - P^T @ ψ(s))              (B,d)
    I_B = np.eye(B, dtype=psi.dtype)
    coeff_col = I_B - P
    anchor_update_b = lr_phi_psi * (coeff_col @ psi_p)          # (B,d)
    pos_update_b    = lr_phi_psi * (psi_s - (P.T @ psi_s))      # (B,d)

    # Forward (row) updates (symmetric):
    #   Δψ(s_j)     = η * (I - Q) @ ψ(sp)
    #   Δψ(sp_k)    = η * (ψ(s_k) - Q^T @ ψ(s))
    coeff_row = I_B - Q
    anchor_update_f = lr_phi_psi * (coeff_row @ psi_p)          # (B,d)
    pos_update_f    = lr_phi_psi * (psi_s - (Q.T @ psi_s))      # (B,d)

    # Combine according to loss_mode
    if loss_mode == "backward":
        anchor_update = anchor_update_b
        pos_update    = pos_update_b
    elif loss_mode == "forward":
        anchor_update = anchor_update_f
        pos_update    = pos_update_f
    else:  # "both" -> average to keep effective step size comparable
        anchor_update = 0.5 * (anchor_update_b + anchor_update_f)
        pos_update    = 0.5 * (pos_update_b    + pos_update_f)

    # Scatter-add into the table
    np.add.at(psi, s_batch,  anchor_update)        # anchors s_j
    np.add.at(psi, sp_batch, pos_update)           # positives sp_k

    
     # After applying the updates (anchor and positive)
    if verbose:  # only print for state 0
        anchor_count = np.sum(s_batch == 0)
        positive_count = np.sum(sp_batch == 0)

        # Compute net updates for state 0
        anchor_net = np.zeros_like(psi[0])
        for idx, s in enumerate(s_batch):
            if s == 0:
                anchor_net += anchor_update[idx]

        positive_net = np.zeros_like(psi[0])
        for idx, sp in enumerate(sp_batch):
            if sp == 0:
                positive_net += pos_update[idx]

        psi_0_after = psi[0]  # already updated at this point
        if random_exploration:
            psi_goal = random_goal_psi
        else:
            psi_goal = psi[goal]


        update_cos.append(np.dot(positive_net + anchor_net,psi_goal )/(np.linalg.norm(positive_net + anchor_net) * np.linalg.norm(psi_goal)))
        past_cos.append(np.dot(psi_0_before, psi_goal)/(np.linalg.norm(psi_0_before) * np.linalg.norm(psi_goal)))
        update_size.append(np.linalg.norm(positive_net + anchor_net))
        anchor_update_cos.append(np.dot(anchor_net,psi_goal )/(np.linalg.norm(anchor_net) * np.linalg.norm(psi_goal) + 1e-8))
        anchor_update_size.append(np.linalg.norm(anchor_net))
        pos_update_cos.append(np.dot(positive_net,psi_goal )/(np.linalg.norm(positive_net) * np.linalg.norm(psi_goal) + 1e-8))
        pos_update_size.append(np.linalg.norm(positive_net))
    # Normalize ψ to unit vectors (L2 norm)
    if norm:
        psi_norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-8
        psi /= psi_norms

    return nll

# === NEW: generic "near goal" helper + per-episode trackers ===
def is_near_state_to_goal(s, g, radius=1):
    si, sj = np.unravel_index(s, walls.shape)
    gi, gj = np.unravel_index(g, walls.shape)
    return abs(si - gi) + abs(sj - gj) <= radius

def frac_time_at_goals(traj, g1, g2, radius=1):
    """
    Fraction of steps spent within 'radius' of goal1/goal2.
    Uses visited states after each step (traj[1:]) as the time base.
    """
    if len(traj) <= 1:
        return 0.0, 0.0
    states = traj[1:]  # post-action states
    n = len(states)
    c1 = sum(is_near_state_to_goal(s, g1, radius) for s in states)
    c2 = sum(is_near_state_to_goal(s, g2, radius) for s in states)
    return c1 / n, c2 / n


# === NEW: "ever reached goal?" helpers and per-episode storage ===
def reached_goal(traj, g, radius=0):
    """Return True iff any state in the trajectory is within `radius` of goal g.
       Use radius=0 for exact-goal cell; set to 1 to allow 'near goal'."""
    if len(traj) == 0:
        return False
    for s in traj:  # includes start as well; that's fine
        si, sj = np.unravel_index(s, walls.shape)
        gi, gj = np.unravel_index(g, walls.shape)
        if abs(si - gi) + abs(sj - gj) <= radius:
            return True
    return False


# tracking goal reaching rates for the two goal exprriment
# per-episode indicators: 1 if reached, else 0
reached_g1 = []
reached_g2 = []
# also track episodes that reached BOTH goals at least once
reached_both = []






# -------- Add just after the hyper-parameters section -----------------------

# Book-keeping
loss_history   = []                    # track −log pₖ losses
visit_counter  = np.zeros_like(walls)  # reused heat-map buffer
success_list = []
eval_success_list = []

plt.ion()                              # interactive plotting

import os

os.makedirs(run_dir , exist_ok=True)
def save_heat(counts, ep):
    fig, ax = plt.subplots(figsize=(6,6))
    im=ax.imshow(counts, cmap="inferno", origin="lower")
    ax.set_title(f"State visits up to ep {ep}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{run_dir}/visits_ep{ep:05d}.png", dpi=150)
    plt.close(fig)

# -------- Add at the very end of the file ----------------------------------

# -------- Training loop -----------------------------------------------------
for ep in trange(1, num_episodes + 1, desc="episodes"):
    # initial plot just to make sure how the initial state look like
    if ep == 1 :
        goal_vec   = psi[goal]  
        if random_exploration:
            goal_vec = random_goal_psi        
        elif two_goals:
            goal_vec = 0.5 * psi[goal] + 0.5 * psi[goal2]  # ψ(g)
        goal_norm  = np.linalg.norm(goal_vec) + 1e-8             # avoid /0
        sim        = (psi @ goal_vec) / (np.linalg.norm(psi, axis=1) * goal_norm + 1e-8)
        
                                    # shape (NUM_STATES,)
        sim_map = sim.reshape(walls.shape)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Background: similarity heatmap
        im = ax.imshow(
            sim_map, cmap="viridis", origin="lower", interpolation="nearest"
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\psi$-similarity")

        # Walls: overlay ONLY the wall cells in black, leave others transparent
        wall_mask = np.ma.masked_where(walls == 0, np.ones_like(walls))  # 1 where wall, masked elsewhere
        ax.imshow(
            wall_mask,
            cmap=mpl.colors.ListedColormap(["black"]),  # force walls to black
            origin="lower",
            interpolation="nearest"
        )

        ax.legend(loc="upper left", frameon=False)

        ax.set_title(f"Episodes {episodes_per_upd + 1}–{ep}")
        ax.axis("off")
        if psi.shape[1] == 2:
            X, Y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
            U = np.zeros_like(X, dtype=float)
            V = np.zeros_like(Y, dtype=float)

            for idx in range(NUM_STATES):
                if walls.flat[idx] == 0:
                    i, j = np.unravel_index(idx, walls.shape)
                    U[i, j], V[i, j] = psi[idx]

            ax.quiver(X, Y, U, V, color="white", scale=1.5, scale_units="xy", angles='xy', width=0.005)
        fig.tight_layout()
        #fig.savefig(f"{run_dir}/trajs_ep{ep:05d}_cos.png", dpi=150)
        fig.savefig(f"{run_dir}/trajs_ep{ep:05d}_cos.pdf", dpi=300, bbox_inches="tight")

        plt.close(fig)




    ep_success = collect_episode(episode_num=ep)
    # === NEW: per-episode "reach any time?" flags for goal1/goal2 ===
    traj = replay[-1]
    hit1 = 1 if reached_goal(traj, goal1, radius=0) else 0   # radius=0 => exact cell
    hit2 = 1 if reached_goal(traj, goal2, radius=0) else 0
    reached_g1.append(hit1)
    reached_g2.append(hit2)
    # reached both goals sometime during this episode?
    reached_both.append(1 if (hit1 and hit2) else 0)



    latest_trajs = replay[-episodes_per_upd:]
    success_list.append(np.mean(ep_success))
    for state in replay[-1]:     # the latest trajectory
        visited_states.add(state)
    visited_counts.append(len(visited_states))

    # perform an SGD update every N episodes
    if ep %  episodes_per_upd == 0:

        loss = update_representations(verbose=verbose)
        loss_history.append(loss)

                # ------- visualise ψ(s)·ψ(g) + trajectories ------------------------
        
    if ep %  plot_mult  == 0 or ep == episodes_per_upd :

        eval_ep_success = run_eval_episode()
        eval_success_list.append(np.mean(eval_ep_success))
       
        #1. similarity heat-map  ψ(s)·ψ(g)  over the whole maze
        goal_vec   = psi[goal]  

        if random_exploration:
            goal_vec = random_goal_psi     
        if two_goals:
            goal_vec = 0.5 * psi[goal] + 0.5 * psi[goal2]
                                      # ψ(g)
        goal_norm  = np.linalg.norm(goal_vec) + 1e-8             # avoid /0
        sim        = (psi @ goal_vec) / (np.linalg.norm(psi, axis=1) * goal_norm + 1e-8)
        
                                    # shape (NUM_STATES,)
        sim_map = sim.reshape(walls.shape)

                # --- PLOT: similarity map + colored trajectories, paper-ready ---
                # --- PLOT: similarity map + single-color trajectories, paper-ready ---
                # --- PLOT: similarity map + single-color trajectories, paper-ready ---
        fig, ax = plt.subplots(figsize=(8, 8))

        # Background: similarity heatmap
        im = ax.imshow(
            sim_map, cmap="viridis", origin="lower", interpolation="nearest"
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\psi$-similarity")
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        # Walls: overlay ONLY the wall cells in black, leave others transparent
        wall_mask = np.ma.masked_where(walls == 0, np.ones_like(walls))  # 1 where wall, masked elsewhere
        ax.imshow(
            wall_mask,
            cmap=mpl.colors.ListedColormap(["black"]),  # force walls to black
            origin="lower",
            interpolation="nearest"
        )

        traj_color = "black"
        for t in latest_trajs:
            coords = np.array([np.unravel_index(s, walls.shape) for s in t])
            ax.plot(
                coords[:, 1], coords[:, 0],
                color=traj_color, linewidth=1.8, alpha=0.95, zorder=3
            )

        # Start marker (bigger)
        si, sj = np.unravel_index(START_STATE, walls.shape)
        h_start = ax.scatter(
            sj, si, marker="o", s=160, c="lime",
            edgecolors="black", linewidths=0.9, zorder=5
        )

        # Goal markers (stars) + handles for legend
        g1i, g1j = np.unravel_index(goal1, walls.shape)
        h_g1 = ax.scatter(
            g1j, g1i, marker="*", s=300, c="crimson",
            edgecolors="black", linewidths=0.9, zorder=6
        )

        h_g2 = None
        if two_goals:
            g2i, g2j = np.unravel_index(goal2, walls.shape)
            h_g2 = ax.scatter(
                g2j, g2i, marker="*", s=300, c="purple",   # purple for Goal 2
                edgecolors="black", linewidths=0.9, zorder=6
            )

        # Title and styling
        ax.set_title(f"Episode {ep}", fontsize=26, pad=10)
        ax.tick_params(labelsize=20)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Legend (Start, Goal 1, Goal 2 if present)
        handles = [(h_start, "Start"), (h_g1, "Goal 1")]
        if h_g2 is not None:
            handles.append((h_g2, "Goal 2"))
        leg = ax.legend(
            [h for h, _ in handles], [lbl for _, lbl in handles],
            loc="upper left", frameon=True, fontsize=20
        )
        leg.get_frame().set_alpha(0.98)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")

        fig.tight_layout()
        fig.savefig(f"{run_dir}/trajs_ep{ep:05d}_cos.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)




        def plot_goal_aligned_3d(psi_mat, goal_vec, out_path, state_indices=None):
            """
            3D scatter of ψ in a goal-aligned basis, using different markers
            for each of the four rooms. If state_indices is provided, it must
            be a 1D array of flattened grid indices aligned with rows of psi_mat.
            """
            eps = 1e-12
            N, d = psi_mat.shape
            if d < 2:
                print("[viz] rep_dim < 2; skipping goal-aligned 3D.")
                return

            # ----- map rows of psi_mat to grid (i, j) -----
            # If user passed only non-wall states, provide their flat indices here.
            if state_indices is None:
                # assume psi_mat corresponds to ALL states in row-major order
                flat_idx = np.arange(walls.size)
            else:
                flat_idx = np.asarray(state_indices)

            ij = np.column_stack(np.unravel_index(flat_idx, walls.shape))
            ii, jj = ij[:, 0], ij[:, 1]

            # If any wall cells slipped in, mask them out to stay safe.
            mask_nonwall = (walls.flatten()[flat_idx] == 0)
            psi_use = psi_mat[mask_nonwall]
            ii = ii[mask_nonwall]
            jj = jj[mask_nonwall]
            N = psi_use.shape[0]

            # ----- normalize and build goal-aligned basis -----
            U  = psi_use / (np.linalg.norm(psi_use, axis=1, keepdims=True) + eps)
            u1 = goal_vec / (np.linalg.norm(goal_vec) + eps)

            R = U - (U @ u1)[:, None] * u1
            try:
                _, _, Vt = np.linalg.svd(R, full_matrices=False)
            except np.linalg.LinAlgError:
                print("[viz] SVD failed; skipping goal-aligned 3D.")
                return

            u2 = Vt[0]
            u2 = u2 - (u1 @ u2) * u1; u2 /= (np.linalg.norm(u2) + eps)

            if d >= 3:
                u3 = Vt[1]
                u3 = u3 - (u1 @ u3) * u1 - (u2 @ u3) * u2
                if np.linalg.norm(u3) < eps:
                    e = np.zeros_like(u1); e[0] = 1.0
                    u3 = e - (u1 @ e) * u1 - (u2 @ e) * u2
                u3 /= (np.linalg.norm(u3) + eps)
            else:
                e = np.zeros_like(u1); e[0] = 1.0
                u3 = e - (u1 @ e) * u1 - (u2 @ e) * u2
                u3 /= (np.linalg.norm(u3) + eps)

            X = U @ u2
            Y = U @ u3
            Z = U @ u1
            c = np.clip(Z, -1, 1)  # color by cos(ψ(s), ψ(goal))

            # ----- room masks (four quadrants) -----
            mid_i = walls.shape[0] // 2
            mid_j = walls.shape[1] // 2
            m_TL = (ii <  mid_i) & (jj <  mid_j)
            m_TR = (ii <  mid_i) & (jj >= mid_j)
            m_BL = (ii >= mid_i) & (jj <  mid_j)
            m_BR = (ii >= mid_i) & (jj >= mid_j)

            # ----- figure & shared colormap -----
            fig = plt.figure(figsize=(12, 10))
            ax  = fig.add_subplot(111, projection='3d')

            # Use a single ScalarMappable to keep one colorbar across groups
            import matplotlib as mpl
            norm = mpl.colors.Normalize(vmin=c.min() if N else 0.0, vmax=c.max() if N else 1.0)
            cmap = plt.get_cmap("viridis")
            sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

            # (marker, label, mask) for the four rooms
            groups = [
                ('o', 'Bottom-Left',     m_TL),
                ('^', 'Bottom-Right',    m_TR),
                ('s', 'Top-Left',  m_BL),
                ('D', 'Top-Right', m_BR),
            ]
            dot_size = 60        # try 36, 60, 100; larger = bigger dots
            rasterize_flag = (N >= 30000)   # keep rasterization for very large clouds
            # plot each room with a different marker (color still encodes cos angle)
            for marker, label, m in groups:
                if np.any(m):
                    ax.scatter(
                    X[m], Y[m], Z[m],
                    c=c[m], cmap=cmap, norm=norm,
                    s=dot_size,
                    alpha=0.95,
                    marker=marker,
                    label=label,
                    edgecolor='k', linewidth=0.18,
                    rasterized=rasterize_flag
                )
            cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03)
            cb.set_label("cos(ψ(s), ψ(goal))")

            # unit sphere wireframe (reference)
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.3, alpha=0.5)

            # mark goal at (0,0,1)
            ax.scatter([0], [0], [1], c='red', s=100, marker='*', label='Goal (z)')


            ax.set_title("3D PCA plot of ψ - FourRooms", fontsize=24)

            # equal aspect ratio
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() if N else 1.0
            cx, cy, cz = (X.mean() if N else 0.0), (Y.mean() if N else 0.0), (Z.mean() if N else 0.0)

            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-0.15, 1.1)   # <-- focus on the top half of the unit sphere

            leg = ax.legend(loc="upper left",
                markerscale=2.5, # multiplies the marker size shown in legend
                handletextpad=0.6,
                labelspacing=0.4,
                borderpad=0.4)

            fig.tight_layout()
            #fig.savefig(out_path, dpi=200)
            fig.savefig(out_path, format="pdf", bbox_inches="tight")


            plt.close(fig)

            # --- choose base path from out_path ---
            base, _ = os.path.splitext(out_path)
            npz_path = base + "_goal_aligned.npz"
            csv_path = base + "_goal_aligned.csv"

            # --- build per-point room label array (string) ---
            room_labels = np.full(X.shape[0], "None", dtype=object)
            room_labels[m_TL] = "Bottom-Left"
            room_labels[m_TR] = "Bottom-Right"
            room_labels[m_BL] = "Top-Left"
            room_labels[m_BR] = "Top-Right"

            # --- compute the flat indices corresponding to the rows saved (only when you masked nonwalls) ---
            # If you used `flat_idx` and `mask_nonwall` above, save the actual flat indices for each saved point:
            if 'flat_idx' in locals() and 'mask_nonwall' in locals():
                saved_flat_idx = flat_idx[mask_nonwall]
            else:
                # fallback: use a simple 0..N-1 index when no walls mapping
                saved_flat_idx = np.arange(X.shape[0])

            # --- Save a compact binary .npz with full fidelity (recommended) ---
            np.savez(
                npz_path,
                flat_idx=saved_flat_idx,
                i=ii,
                j=jj,
                X=X,
                Y=Y,
                Z=Z,
                c=c,
                room=room_labels,
                u1=u1,
                u2=u2,
                u3=u3,
                psi_use=psi_use
            )

            # --- Save a CSV for easy inspection / sharing ---
            df = pd.DataFrame({
                "flat_idx": saved_flat_idx,
                "i": ii,
                "j": jj,
                "X": X,
                "Y": Y,
                "Z": Z,
                "c": c,
                "room": room_labels
            })
            df.to_csv(csv_path, index=False)

            print(f"[viz] saved projection data: {npz_path} (binary), {csv_path} (csv)")

        def plot_pca_3d(psi_mat, out_path):
            eps = 1e-12
            N, d = psi_mat.shape
            if d < 3:
                print("[viz] rep_dim < 3; skipping PCA-3D.")
                return
            U = psi_mat / (np.linalg.norm(psi_mat, axis=1, keepdims=True) + eps)
            X = U - U.mean(axis=0, keepdims=True)
            try:
                U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)
            except np.linalg.LinAlgError:
                print("[viz] SVD failed; skipping PCA-3D.")
                return
            PCs = Vt[:3].T
            Z = X @ PCs
            var_ratio = (S[:3]**2) / (S**2).sum()

            r = np.linalg.norm(Z, axis=1)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=r, cmap="viridis", s=12, alpha=0.9)
            fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="‖PC coords‖")

            ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
            ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
            ax.set_title("PCA 3D of ψ (unit-normalized)")

            max_range = np.array([Z[:,0].ptp(), Z[:,1].ptp(), Z[:,2].ptp()]).max()
            cx, cy, cz = Z[:,0].mean(), Z[:,1].mean(), Z[:,2].mean()
            ax.set_xlim(cx - max_range/2, cx + max_range/2)
            ax.set_ylim(cy - max_range/2, cy + max_range/2)
            ax.set_zlim(cz - max_range/2, cz + max_range/2)

            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

        # Build abstract goal vector exactly like in your selectors
        

        # Save figures
        ## uncomment this to get the pca plots
        # plot_goal_aligned_3d(psi, goal_vec, os.path.join(run_dir, f"psi_goal_frame_3d_{ep:05d}.pdf"))




# ---------- Plot and Save Success Curve -------------------------
if success_list:  # make sure we recorded some
    fig_s, ax_s = plt.subplots(figsize=(6, 3))
    episodes = np.arange(1, len(success_list) + 1)
    ax_s.plot(episodes, success_list, marker=".", linewidth=1)
    ax_s.set_xlabel("episode")
    ax_s.set_ylabel("avg step-success")
    ax_s.set_title("Per-episode success rate")
    ax_s.set_ylim(0, 1.05)
    fig_s.tight_layout()
    fig_s.savefig(f"{run_dir}/success_curve.png", dpi=150)
    plt.close(fig_s)

    # Save success list
    np.save(os.path.join(run_dir, "success_list.npy"), np.array(success_list))
    # Optionally: save as text
    # np.savetxt(os.path.join(run_dir, "success_list.txt"), np.array(success_list), fmt="%.4f")


if eval_success_list:  # make sure we recorded some
    fig_s, ax_s = plt.subplots(figsize=(6, 3))
    episodes = np.arange(1, len(eval_success_list) + 1)
    ax_s.plot(episodes, eval_success_list, marker=".", linewidth=1)
    ax_s.set_xlabel("episode")
    ax_s.set_ylabel("avg eval step-success")
    ax_s.set_title("Per-episode eval success rate")
    ax_s.set_ylim(0, 1.05)
    fig_s.tight_layout()
    fig_s.savefig(f"{run_dir}/eval_success_curve.png", dpi=150)
    plt.close(fig_s)

    # Save success list
    np.save(os.path.join(run_dir, "eval_success_list.npy"), np.array(eval_success_list))
    # Optionally: save as text
    # np.savetxt(os.path.join(run_dir, "success_list.txt"), np.array(success_list), fmt="%.4f")


def moving_avg(x, w=100):
    """Simple moving average with window w (valid mode)."""
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return np.array([]), np.array([])  # not enough points
    kernel = np.ones(w) / w
    smoothed = np.convolve(x, kernel, mode="valid")
    episodes = np.arange(w, len(x) + 1)
    return episodes, smoothed

# === NEW: save per-episode goal reach indicators ===
# === Save per-episode goal reach indicators (including BOTH) ===
if reached_g1 and reached_g2 and reached_both:
    df_reach = pd.DataFrame({
        "episode": np.arange(1, len(reached_g1) + 1),
        "reached_goal1": reached_g1,
        "reached_goal2": reached_g2,
        "reached_both":  reached_both,
    })
    df_reach.to_csv(os.path.join(run_dir, "goal_reach_rates.csv"), index=False)
def ema_series(x, alpha=0.02):
    """
    Exponential moving average (weighted average; more weight on recent episodes).
    alpha in (0,1]: higher -> reacts faster. alpha≈2/(N+1) mimics N-episode SMA.
    Returns np.array of same length as x.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([])
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for t in range(1, x.size):
        y[t] = alpha * x[t] + (1 - alpha) * y[t-1]
    return y

# === NEW: plot smoothed reach rates (moving average of 0/1) ===
# === Plot weighted (EMA) reach rates for Goal1, Goal2, and BOTH ===
if reached_g1 and reached_g2 and reached_both:
    # choose your smoothing strength here:
    # e.g., to mimic ~100-episode SMA: alpha ≈ 2/(100+1) ≈ 0.0198
    alpha = 0.02

    g1_ema   = ema_series(reached_g1,   alpha=alpha)
    g2_ema   = ema_series(reached_g2,   alpha=alpha)
    both_ema = ema_series(reached_both, alpha=alpha)
    episodes = np.arange(1, len(reached_g1) + 1)

    # save the EMA series as CSV too
    df_ema = pd.DataFrame({
        "episode": episodes,
        "goal1_ema": g1_ema,
        "goal2_ema": g2_ema,
        "both_ema":  both_ema,
        "alpha":     np.full_like(g1_ema, alpha, dtype=float),
    })
    df_ema.to_csv(os.path.join(run_dir, "goal_reach_rates_weighted_ema.csv"), index=False)

    # figure
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(episodes, g1_ema, label="Goal1 (EMA)")
    ax.plot(episodes, g2_ema, label="Goal2 (EMA)")
    ax.plot(episodes, both_ema, label="Both (EMA)")
    ax.set_xlabel("episode")
    ax.set_ylabel("weighted reach rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Weighted (EMA, α={alpha:.3f}) reach rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "goal_reach_rates_weighted_ema.png"), dpi=150)
    plt.close(fig)







    

