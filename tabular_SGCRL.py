"""
Tabular Contrastive RL in a Maze
================================
Author: Mahsa Bastankhah and ChatGPT (July 2025)

* Maze world: 30×30 grid with a vertical and horizontal wall, each with 5-cell doors.
* Policy: ε-greedy to maximise  φ(s,a) · ψ(g).
* Replay: every completed trajectory is stored; every `episodes_per_upd`
  we run a simple hinge contrastive update on a random batch of trajectories.
* No neural networks – purely tabular φ and ψ.

Run:
  $ pip install numpy matplotlib tqdm
  $ python tabular_contrastive_maze.py
"""
import argparse, json, time, os
from pathlib import Path
import hashlib


               # lightweight optimiser library for JAX
# -------- Imports ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from collections import deque 

# -------- Maze construction --------------------------------------------------
HEIGHT, WIDTH = 10, 10
GOAL_COORD = (9, 9)                           # row 29, col 29 (zero-indexed)
walls = np.zeros((HEIGHT, WIDTH), dtype=int)
DOOR_LEN = 2   
# horizontal wall
walls[HEIGHT // 2, :] = 1
doors_h = np.concatenate([
    WIDTH // 4 + np.arange(DOOR_LEN),
    WIDTH * 3 // 4 + np.arange(DOOR_LEN)
])
walls[HEIGHT // 2, doors_h] = 0

# vertical wall
walls[:, WIDTH // 2] = 1
doors_v = np.concatenate([
    HEIGHT // 4 + np.arange(DOOR_LEN),
    HEIGHT * 3 // 4 + np.arange(DOOR_LEN)
])
walls[doors_v, WIDTH // 2] = 0

empty_states = np.where(walls.flatten() == 0)[0]


# HEIGHT, WIDTH = 5, 5
# GOAL_COORD = (4, 4)  # zero-indexed bottom-right corner
# walls = np.zeros((HEIGHT, WIDTH), dtype=int)  # no walls

# empty_states = np.where(walls.flatten() == 0)[0]

NUM_STATES     = HEIGHT * WIDTH
NUM_ACTIONS    = 5           # stay, down, up, right, left
A_TO_DELTA     = np.array([[0, 0],
                           [1, 0], [-1, 0],
                           [0, 1], [0, -1]])
# NUM_ACTIONS    = 4           # stay, down, up, right, left
# A_TO_DELTA     = np.array([[1, 0], [-1, 0],
#                            [0, 1], [0, -1]])
START_STATE = np.ravel_multi_index((0, 0), walls.shape) 

#### spiral
# -------- Maze construction: spiral pattern -------------------------------
# HEIGHT = WIDTH = 11
# GOAL_COORD = (10, 10)               # bottom-right (zero-indexed)

# # 1 = wall, 0 = free space
# walls = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                   [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
#                   [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
#                   [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
#                   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
#                   [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
#                   [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
#                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

# # ---------------------------------------------------------------------------
# empty_states = np.where(walls.flatten() == 0)[0]
# NUM_STATES   = HEIGHT * WIDTH
# NUM_ACTIONS  = 5                     # stay, down, up, right, left
# A_TO_DELTA   = np.array([[0, 0],
#                          [1, 0], [-1, 0],
#                          [0, 1], [0, -1]])
# START_STATE = np.ravel_multi_index((5, 5), walls.shape) 






### impossible goal
# walls = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 1, 0, 1, 1, 1, 1, 1, 0],
#                   [0, 1, 0, 0, 0, 0, 1, 0, 0],
#                   [0, 1, 1, 1, 1, 0, 1, 0, 1],
#                   [0, 1, 0, 0, 0, 0, 1, 0, 0],
#                   [0, 1, 0, 1, 1, 1, 1, 1, 0],
#                   [0, 0, 0, 1, 0, 0, 0, 1, 0],
#                   [0, 1, 0, 1, 0, 1, 0, 1, 1],
#                   [0, 1, 0, 0, 0, 1, 0, 1, 0]])

# HEIGHT = WIDTH = 9
# GOAL_COORD = (6, 8)               # bottom-right (zero-indexed)



# # ---------------------------------------------------------------------------
# empty_states = np.where(walls.flatten() == 0)[0]
# NUM_STATES   = HEIGHT * WIDTH
# NUM_ACTIONS  = 5                     # stay, down, up, right, left
# A_TO_DELTA   = np.array([[0, 0],
#                          [1, 0], [-1, 0],
#                          [0, 1], [0, -1]])
# START_STATE = np.ravel_multi_index((0, 0), walls.shape) 


# -------- Environment helpers ------------------------------------------------
def step(state: int, action: int) -> int:
    """Deterministic step function for the maze."""
    di, dj = A_TO_DELTA[action]
    i, j   = np.unravel_index(state, walls.shape)
    ni, nj = i + di, j + dj
    if 0 <= ni < HEIGHT and 0 <= nj < WIDTH and walls[ni, nj] == 0:
        return np.ravel_multi_index((ni, nj), walls.shape)
    return state  # blocked: stay in place


# -------- Environment helpers ------------------------------------------------
def step(state: int, action: int) -> int:
    """Deterministic step function for the maze."""
    di, dj = A_TO_DELTA[action]
    i, j   = np.unravel_index(state, walls.shape)
    ni, nj = i + di, j + dj
    if 0 <= ni < HEIGHT and 0 <= nj < WIDTH and walls[ni, nj] == 0:
        return np.ravel_multi_index((ni, nj), walls.shape)
    return state  # blocked: stay in place




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rep-dim",          type=int,   default=16)
    p.add_argument("--epsilon",          type=float, default=0.01)
    p.add_argument("--episodes-per-upd", type=int,   default=5)
    p.add_argument("--max-steps",        type=int,   default=100)
    p.add_argument("--batch-size",       type=int,   default=128)
    p.add_argument("--lr-phi-psi",       type=float, default=1e-3)
    p.add_argument("--num-episodes",     type=int,   default=50_000)
    p.add_argument("--gamma",            type=float, default=0.99)
    p.add_argument("--seed",             type=int,   default=1)
    p.add_argument("--replay-capacity",  type=int,   default=1000)
    # distance-aware ψ init params
    p.add_argument("--near-var",         type=float, default=0.5)
    p.add_argument("--far-var",          type=float, default=10.0)
    p.add_argument("--alpha",            type=float, default=5.0)
    p.add_argument("--entropy_coeff",            type=float, default=0.1)
    p.add_argument("--random_exploration",         type=bool, default=False)
    p.add_argument("--two_goals",         type=bool, default=False)
    p.add_argument("--distance_aware_init",         type=bool, default=False)
    p.add_argument("--verbose",         type=bool, default=False)
    p.add_argument("--fully_random_init",         type=bool, default=False)
    p.add_argument("--loss",         type=str, default="backward")
    return p.parse_args()

args = parse_args()

# Bind to simple variable names (optional, keeps rest of code readable)
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
near_var         = args.near_var
far_var          = args.far_var
fully_random_init = args.fully_random_init
verbose = args.verbose
alpha            = args.alpha
# plot_mult        = args.plot_mult
plot_mult        = num_episodes // 50
if verbose:
    plot_mult = max(num_episodes // 50 , 50 )# <-- for testing, plot every 5 episodes
entropy_coeff   = args.entropy_coeff
norm = True
random_exploration = args.random_exploration
two_goals = args.two_goals
plain_goal = False
distance_aware_init = args.distance_aware_init

np.random.seed(seed)

# -------- Run directory & config save ---------------------------------------
# 1. Assemble full config dict (no timestamp yet for hashing)
config = {
    "rep_dim": rep_dim,
    "episodes_per_upd": episodes_per_upd,
    "max_steps": max_steps,
    "batch_size": batch_size,
    "lr_phi_psi": lr_phi_psi,
    "num_episodes": num_episodes,
    "gamma": gamma,
    "seed": seed,
    "replay_capacity": replay_capacity,
    "near_var": near_var,
    "far_var": far_var,
    "alpha": alpha,
    "plot_mult": plot_mult,
    "entropy_coeff": entropy_coeff,
    "random_exploration": random_exploration,
    "norm": norm,
    "two_goals": two_goals,
    "distance_aware_init": distance_aware_init,
    "fully_random_init": fully_random_init,
    "loss": args.loss,
}

# 2. Create a stable UID *excluding the seed* so same hyperparams with different seeds
#    land in parallel subfolders under their own seed directory.
hash_basis = {k: v for k, v in config.items() if k != "seed"}
hash_json  = json.dumps(hash_basis, sort_keys=True).encode()
uid = hashlib.sha1(hash_json).hexdigest()[:10]      # first 10 hex chars

# 3. Build run directory: runs/seed{seed}/{uid}/
run_dir = Path(f"runs_final/seed{seed}") / uid
run_dir.mkdir(parents=True, exist_ok=True)

# 4. Add timestamp then save full config (including seed & uid & hash_basis copy)
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
SECOND_GOAL_COORD = (0, 9)

goal1 = np.ravel_multi_index(GOAL_COORD, walls.shape)
goal2 = np.ravel_multi_index(SECOND_GOAL_COORD, walls.shape) if two_goals else goal1

# keep `goal` for backward compatibility
goal = goal1
# goal = np.ravel_multi_index(GOAL_COORD, walls.shape)
  # <-- add once, near other hyper-params
# phi = np.random.randn(NUM_STATES, NUM_ACTIONS, rep_dim) * 0.1  # φ(s,a)
# psi = np.random.randn(NUM_STATES,                 rep_dim) * 0.1  # ψ(s)
# -------- Tabular parameters -------------------------------------------------
# phi = np.random.randn(NUM_STATES, NUM_ACTIONS, rep_dim) * 0.1  # φ(s,a)

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



# def plane_projector(v1: np.ndarray, v2: np.ndarray):
#     """Return orthogonal projector P onto span{v1, v2}."""
#     eps = 1e-8
#     u1 = v1 / (np.linalg.norm(v1) + eps)

#     v2_orth = v2 - (u1 @ v2) * u1
#     if np.linalg.norm(v2_orth) < eps:
#         # fall back to any orthogonal direction
#         e = np.zeros_like(u1); e[0] = 1.0
#         if abs(e @ u1) > 0.9:
#             e = np.zeros_like(u1); e[1] = 1.0
#         v2_orth = e - (e @ u1) * u1

#     u2 = v2_orth / (np.linalg.norm(v2_orth) + eps)
#     U  = np.stack([u1, u2], axis=1)   # (d x 2)
#     P  = U @ U.T                      # (d x d)
#     return P

# P_plane = None
# if plain_goal:
#     psi_goal2 = np.random.randn(rep_dim) * 0.1             # anchor embedding for second goal
#     psi[goal2] = psi_goal2
#     P_plane = plane_projector(psi_goal, psi_goal2)



                                 # set goal’s ψ once
initialization_for_other_states = np.random.randn(rep_dim) * 0.1             # anchor embedding
## initialize representations in a distance-aware manner
if two_goals is False and distance_aware_init:
    print("[INFO] Initializing ψ(s) distance-aware...")
    for s, c in enumerate(coords):
        if s == goal:
            continue
        d         = np.linalg.norm(c - goal_coord)          # L2 to goal
        frac      = (d / max_dist) ** alpha                # 0 near goal → 1 far away
        var_scale = near_var + (far_var - near_var) * frac # extreme scaling
        print(f"State {s} at {c}: dist={d:.2f}, frac={frac:.2f}, var_scale={var_scale:.2f}")
        psi[s]    = psi_goal + np.random.randn(rep_dim) * 0.1 * var_scale



elif two_goals is False and not distance_aware_init:
    print("[INFO] Initializing ψ(s) uniformly around ψ(goal)...")
    for s, c in enumerate(coords):
        if s == goal:
            continue
        psi[s]    =  psi_goal + np.random.randn(rep_dim) * 0.1 *  near_var


elif two_goals is True and distance_aware_init:
    print("[INFO] Initializing ψ(s) distance-aware with two goals...")
    for s, c in enumerate(coords):
        if s == goal or s == goal2:
            continue
        d         = np.linalg.norm(c - goal_coord)          # L2 to goal
        frac      = (d / max_dist) ** alpha                # 0 near goal → 1 far away
        var_scale = near_var + (far_var - near_var) * frac # extreme scaling
        print(f"State {s} at {c}: dist={d:.2f}, frac={frac:.2f}, var_scale={var_scale:.2f}")
        psi[s]    = twin_goal_psi + np.random.randn(rep_dim) * 0.1 * var_scale
else:
    print("specification of `two_goals` and `distance_aware_init` and `random init` is not supported for initialization")
    raise ValueError("Unsupported combination of two_goals and distance_aware_init")
    ## each element's is around 0.1 so the norm for a 16 d vec is around 0.1 * 4 = 0.4
    #psi[s] = psi_goal
    
    ## all randomly initialized
    #psi[s] = np.random.randn(rep_dim) * 0.1

# Normalize ψ to unit vectors (L2 norm)
if norm :
    psi_norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-8
    psi /= psi_norms
    
#### if random exploration is True, initialization is done from scratch 
if random_exploration:
    random_goal_psi = np.random.randn(rep_dim) * 0.1
    random_goal_psi /= np.linalg.norm(random_goal_psi) + 1e-8
    print(f"[INFO] Using random exploration with ψ(goal) = {random_goal_psi[:5]}...")
    if fully_random_init:
        print("[INFO] Initializing ψ(s) uniformly at random...")
        for s, c in enumerate(coords):
            psi[s] = np.random.randn(rep_dim) * 0.1 
    else:
        for s, c in enumerate(coords):
            var_scale = near_var 
            print(f"State {s} var_scale={var_scale:.2f}")
            psi[s]    = random_goal_psi + np.random.randn(rep_dim) * 0.1 * var_scale




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
        if plain_goal:
            proj = P_plane @ psi[ns]
            sim = proj @ proj  # squared norm in the plane
        similarities.append(sim)

    # Apply entropy regularization (inverse temperature)
    logits = np.array(similarities)
    inverse_temp = 1.0 / entropy_coeff
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


def collect_episode() -> None:
    """Generate one trajectory and push into replay buffer."""
    step_success = []
    traj = [START_STATE]
    for _ in range(max_steps):
        a  = select_action(traj[-1], goal)
        ns = step(traj[-1], a)
        traj.append(ns)
        success = (ns == goal) or is_near_goal(ns)
        step_success.append(1 if success else 0)

    replay.append(traj)
    if len(replay) > replay_capacity:      # manual eviction
        replay.pop(0)  

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

    if verbose:
        psi_0_before = psi[0].copy()

    # ----- 1. Sample minibatch exactly as before ----------------------------
    if len(replay) < 2:
        return 0.0

    traj_ids = np.random.choice(len(replay), batch_size, replace=True)
    s_list, sp_list = [], []
    for idx in traj_ids:
        traj = replay[idx]
        if len(traj) < 2:
            continue
        i = np.random.randint(0, len(traj) - 1)
        remaining = len(traj) - i
        # w = gamma ** np.arange(remaining)
        # w /= w.sum()
        # j = i + np.random.choice(remaining, p=w)

        w = gamma ** np.arange(1, remaining)
        w /= w.sum()
        j = i + np.random.choice(np.arange(1, remaining), p=w)

        s_list.append(traj[i])
        sp_list.append(traj[j])

    if not s_list:                        # batch could be empty
        return 0.0

    s_batch  = np.asarray(s_list,  dtype=np.int32)   # anchors (B,)
    sp_batch = np.asarray(sp_list, dtype=np.int32)   # positives (B,)
    B        = len(s_batch)

    # if verbose:
    #     from collections import Counter

    #     pair_counter = Counter(zip(s_batch, sp_batch))

    #     print("\n=== [Anchor–Positive Pair Histogram] ===")
    #     for (s, sp), count in pair_counter.most_common():
    #         print(f"(s={s:4d}, sp={sp:4d})  | Count: {count:2d}")

    #     print(f"\nTotal unique pairs: {len(pair_counter)}")
    #     top_pair = pair_counter.most_common(1)[0]
    #     print(f"Most common pair: (s={top_pair[0][0]}, sp={top_pair[0][1]})  | Count: {top_pair[1]}")


    # ----- 2. Gather current embeddings -------------------------------------
    psi_s = psi[s_batch]                  # ψ(s_j)      – shape (B,d)
    psi_p = psi[sp_batch]                 # ψ(sp_k)     – shape (B,d)

    # ----- 3. Column-wise soft-max probabilities P --------------------------
    dots        = psi_s @ psi_p.T                         # (B,B)
    dots       -= dots.max(axis=0, keepdims=True)         # stabilise
    exp_logits  = np.exp(dots)
    P           = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    diag_P = np.diag(P)                                   # p_k(k)
    nll    = -np.mean(np.log(diag_P + 1e-12))             # mean-NLL

    # ----- 4. Anchor-state updates  Δψ(s_j) -------------------------------
    coeff         = np.eye(B) - P                         # (B,B)
    anchor_update = lr_phi_psi * (coeff @ psi_p)          # (B,d)
    np.add.at(psi, s_batch, anchor_update)                # scatter-add

    # if verbose and goal is not None:
    #     psi_goal = psi[goal]
    #     for idx, s in enumerate(s_batch):
    #         ip = np.dot(anchor_update[idx], psi_goal)
    #         print(f"[Anchor] s={s}, ⟨Δψ(s), ψ(goal)⟩ = {ip:.10f}")

    # ----- 5. Positive-state updates  Δψ(sp_k) ----------------------------
    expected_anchor = (P.T @ psi_s)                       # (B,d)
    pos_update      = lr_phi_psi * (psi_s - expected_anchor)
    np.add.at(psi, sp_batch, pos_update)                  # scatter-add

    
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

        # Print update summary
        # print("\n=== [State 0 Summary] ===")
        # print(f"Anchor count:   {anchor_count}")
        # print(f"Positive count: {positive_count}")
        # print(f"cosine sim of ⟨ψ₀_before, ψ(goal)⟩ = {np.dot(psi_0_before,psi_goal )/(np.linalg.norm(psi_0_before) * np.linalg.norm(psi_goal)):.10f}")
        # print(f"cos sim of ⟨ψ₀_after,  ψ(goal)⟩ = {np.dot(psi_0_after, psi_goal)/(np.linalg.norm(psi_0_after) * np.linalg.norm(psi_goal)):.10f}")
        
        # print(f" Δψ₀_anchor , psi(goal) = {np.dot(anchor_net,psi_goal )}")
        # print(f" Δψ₀_positive , psi(goal) = {np.dot(positive_net,psi_goal )}")
        # print(f" Δψ₀_sum , psi(goal) = {np.dot(positive_net + anchor_net,psi_goal )}")

        # print(f" cos sim Δψ₀_sum , psi(goal) = {np.dot(positive_net + anchor_net,psi_goal )/(np.linalg.norm(positive_net + anchor_net) * np.linalg.norm(psi_goal))}")
        # print(f"‖Δψ₀_total‖ = {np.linalg.norm(psi_0_after - psi_0_before):.6f}")

        #print(f"‖Δψ₀_total_recalc‖ = {np.linalg.norm(anchor_net + positive_net):.6f}")
        #print(f"‖ψ₀_after‖ = {np.linalg.norm(anchor_net + positive_net + psi_0_before):.6f}")
        #print(f"‖ψ₀_before‖ = {np.linalg.norm(psi_0_before):.6f}")
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
            goal_vec = random_goal_psi                                 # ψ(g)
        goal_norm  = np.linalg.norm(goal_vec) + 1e-8             # avoid /0
        sim        = (psi @ goal_vec) / (np.linalg.norm(psi, axis=1) * goal_norm + 1e-8)
        
                                    # shape (NUM_STATES,)
        sim_map = sim.reshape(walls.shape)

        fig, ax = plt.subplots(figsize=(6, 6))

        # background: similarity heat-map
        im = ax.imshow(sim_map,
                    cmap="viridis",          # pick any perceptual colormap
                    origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                    label=r"$cos \psi(s)\!,\!\psi(g)$")

        # optional: overlay walls in light gray
        ax.imshow(walls,
                cmap=plt.cm.binary, vmin=0, vmax=1,
                alpha=0.5, origin="lower")

        ax.legend(loc="upper left", frameon=False)

        ax.set_title(f"Episodes {ep - episodes_per_upd + 1}–{ep}, cos similarity")
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
        fig.savefig(f"{run_dir}/trajs_ep{ep:05d}_cos.png", dpi=150)
        plt.close(fig)




    ep_success = collect_episode()
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

        fig, ax = plt.subplots(figsize=(6, 6))

        # background: similarity heat-map
        im = ax.imshow(sim_map,
                    cmap="viridis",          # pick any perceptual colormap
                    origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                    label=r"$cos \psi(s)\!,\!\psi(g)$")

        # optional: overlay walls in light gray
        ax.imshow(walls,
                cmap=plt.cm.binary, vmin=0, vmax=1,
                alpha=0.5, origin="lower")

        # 2. draw the most-recent trajectories
        for t in latest_trajs:
            coords = np.array([np.unravel_index(s, walls.shape) for s in t])
            ax.plot(coords[:, 1], coords[:, 0], color="black", linewidth=1.0)

        # 3. highlight the fixed start and goal
        ax.scatter(*np.unravel_index(START_STATE, walls.shape)[::-1],
                marker="o", s=60, c="lime", label="start")
        ax.scatter(*np.unravel_index(goal, walls.shape)[::-1],
                marker="*", s=90, c="red",  label="goal")
        ax.legend(loc="upper left", frameon=False)

        ax.set_title(f"Episodes {ep - episodes_per_upd + 1}–{ep}, cos similarity")
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
        fig.savefig(f"{run_dir}/trajs_ep{ep:05d}_cos.png", dpi=150)
        plt.close(fig)


        sim1        = (psi @ goal_vec) 
        sim1_map = sim1.reshape(walls.shape)

        fig, ax = plt.subplots(figsize=(6, 6))

        # background: similarity heat-map
        im1 = ax.imshow(sim1_map,
                    cmap="viridis",          # pick any perceptual colormap
                    origin="lower")
        fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04,
                    label=r"$\psi(s)\!\cdot\!\psi(g)$")

        # optional: overlay walls in light gray
        ax.imshow(walls,
                cmap=plt.cm.binary,
                alpha=0.5, origin="lower")

        # 2. draw the most-recent trajectories
        for t in latest_trajs:
            coords = np.array([np.unravel_index(s, walls.shape) for s in t])
            ax.plot(coords[:, 1], coords[:, 0], color="black", linewidth=1.0)

        # 3. highlight the fixed start and goal
        ax.scatter(*np.unravel_index(START_STATE, walls.shape)[::-1],
                marker="o", s=60, c="lime", label="start")
        ax.scatter(*np.unravel_index(goal, walls.shape)[::-1],
                marker="*", s=90, c="red",  label="goal")
        ax.legend(loc="upper left", frameon=False)

        ax.set_title(f"Episodes {ep - episodes_per_upd + 1}–{ep}, inner product")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(f"{run_dir}/trajs_ep{ep:05d}.png", dpi=150)
        plt.close(fig)

             # ---------- NEW: save loss curve up to this episode --------------
if loss_history:                                     # nothing to plot on very first call
    fig_l, ax_l = plt.subplots(figsize=(4, 2.5))
    xs = np.arange(len(loss_history)) * episodes_per_upd
    ax_l.plot(xs, loss_history, marker=".", linewidth=1)
    ax_l.set_xlabel("episode")
    ax_l.set_ylabel("contrastive NLL")
    ax_l.set_title("training loss")
    fig_l.tight_layout()
    fig_l.savefig(f"{run_dir}/loss_ep{ep:05d}.png", dpi=150)
    plt.close(fig_l)


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



# ---------- Plot and Save Unique Visited States -------------------------
if visited_counts:
    fig_v, ax_v = plt.subplots(figsize=(6, 3))
    ax_v.plot(np.arange(1, len(visited_counts) + 1), visited_counts, marker=".", linewidth=1)
    ax_v.set_xlabel("episode")
    ax_v.set_ylabel("# unique visited states")
    ax_v.set_title("Exploration: unique states visited")
    fig_v.tight_layout()
    fig_v.savefig(f"{run_dir}/visited_states.png", dpi=150)
    plt.close(fig_v)

    # Save visited counts
    np.save(os.path.join(run_dir, "visited_counts.npy"), np.array(visited_counts))
    # Optionally: save as text
    # np.savetxt(os.path.join(run_dir, "visited_counts.txt"), np.array(visited_counts), fmt="%d")

if verbose:
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    def moving_avg_valid(x, w=100):
        x = np.asarray(x, dtype=float)
        if x.size < w:
            return np.array([])  # not enough points
        kernel = np.ones(w, dtype=float) / w
        return np.convolve(x, kernel, mode="valid")

    window = 100
    os.makedirs(run_dir, exist_ok=True)

    # --- compute rolling means ---
    past_cos_ma   = moving_avg_valid(past_cos,   window)
    update_cos_ma = moving_avg_valid(update_cos, window)
    update_size_ma = moving_avg_valid(update_size, window)
    anchor_update_cos_ma = moving_avg_valid(anchor_update_cos, window)
    anchor_update_size_ma = moving_avg_valid(anchor_update_size, window)
    pos_update_cos_ma = moving_avg_valid(pos_update_cos, window)
    pos_update_size_ma = moving_avg_valid(pos_update_size, window)

    # steps for 'valid' window: first averaged point corresponds to index window-1
    steps_ma = np.arange(window - 1, (window - 1) + len(past_cos_ma))

    # Warn if not enough points yet
    if len(past_cos_ma) == 0:
        print(f"[warn] fewer than {window} updates; skipping moving-average plots.")
    else:
        # -------- Figure 1: cosine similarities (smoothed) --------
        fname = f"state_0_cosine_diagnostics_w{window}.png"
        plt.figure()
        # optional: faint raw curves for reference
        plt.plot(np.arange(len(past_cos)), past_cos, alpha=0.2, label="cos(ψ0_before, ψ_goal) [raw]")
        plt.plot(np.arange(len(update_cos)), update_cos, alpha=0.2, label="cos(Δψ0_total, ψ_goal) [raw]")
        # moving averages
        plt.plot(steps_ma, past_cos_ma,   label=f"cos(ψ0_before, ψ_goal) [MA{window}]")
        plt.plot(steps_ma, update_cos_ma, label=f"cos(Δψ0_total, ψ_goal) [MA{window}]")
        plt.xlabel("Update step")
        plt.ylabel("Cosine similarity")
        plt.title("State 0 cosine diagnostics (100-step moving average)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname), dpi=200)
        plt.close()

        # -------- Figure 2: update size (smoothed) --------
        fname = f"state_0_cosine_diagnostics_update_size_w{window}.png"
        plt.figure()
        plt.plot(np.arange(len(update_size)), update_size, alpha=0.2, label="‖Δψ0_total‖ [raw]")
        plt.plot(steps_ma, update_size_ma, label=f"‖Δψ0_total‖ [MA{window}]")
        plt.xlabel("Update step")
        plt.ylabel("Update norm")
        plt.title("State 0 update magnitude (100-step moving average)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname), dpi=200)
        plt.close()

        fname = f"anchor-and-pos-update-cos-{window}.png"
        plt.figure()
        print(f"anchor_update_cos_ma: {len(anchor_update_cos_ma)}, pos_update_cos_ma: {len(pos_update_cos_ma)}, steps_ma: {len(steps_ma)}")
        plt.plot(steps_ma, anchor_update_cos_ma, alpha=0.2, label="anchor update cos")
        plt.plot(steps_ma, pos_update_cos_ma, label=f"pos update cos")
        plt.xlabel("Update step")
        plt.ylabel("Update cos")
        plt.title("")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname), dpi=200)
        plt.close()

        fname = f"anchor-and-pos-update-size-{window}.png"
        plt.figure()
        plt.plot(steps_ma, anchor_update_size_ma, alpha=0.2, label="anchor update size")
        plt.plot(steps_ma, pos_update_size_ma, label=f"pos update size")
        plt.xlabel("Update step")
        plt.ylabel("Update size")
        plt.title("")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, fname), dpi=200)
        plt.close()



        def spherical_stats(psi, eps=1e-8):
            psi_unit = psi / (np.linalg.norm(psi, axis=1, keepdims=True) + eps)
            N, d = psi_unit.shape
            mean_dir = psi_unit.mean(axis=0)
            R = np.linalg.norm(mean_dir)               # mean resultant length ∈ [0,1]
            mean_dir = mean_dir / (R + eps) if R > eps else np.zeros_like(mean_dir)

            # rough vMF κ estimate (valid for d>=3 and not too small/large R)
            # κ ≈ R*(d - R**2) / (1 - R**2)
            if R < 1 - 1e-6:
                kappa = R * (d - R**2) / (1 - R**2)
            else:
                kappa = np.inf

            return R, kappa, mean_dir

        R, kappa, mean_dir = spherical_stats(psi)
        print(f"[spherical] mean resultant length R = {R:.4f}  (0≈uniform, 1≈perfectly aligned)")
        print(f"[spherical] vMF κ (rough) = {kappa:.3f}")
        print(f"[spherical] angle(mean_dir, goal) = "
            f"{np.degrees(np.arccos(np.clip((mean_dir @ (psi[goal]/(np.linalg.norm(psi[goal])+1e-8))), -1, 1))):.2f}°")
        
        def plot_goal_aligned_3d(psi_mat, goal_vec, out_path):
            eps = 1e-12
            N, d = psi_mat.shape
            if d < 2:
                print("[viz] rep_dim < 2; skipping goal-aligned 3D.")
                return
            # unit-normalize all
            U = psi_mat / (np.linalg.norm(psi_mat, axis=1, keepdims=True) + eps)

            # u1 = goal direction (unit)
            u1 = goal_vec / (np.linalg.norm(goal_vec) + eps)

            # remove along-goal part to find dominant orth directions
            R = U - (U @ u1)[:, None] * u1
            # SVD to get top orth directions
            try:
                _, _, Vt = np.linalg.svd(R, full_matrices=False)
            except np.linalg.LinAlgError:
                print("[viz] SVD failed; skipping goal-aligned 3D.")
                return

            u2 = Vt[0]
            u2 = u2 - (u1 @ u2) * u1; u2 /= (np.linalg.norm(u2) + eps)

            # third axis (if d>=3), else fabricate an orthonormal axis
            if d >= 3:
                u3 = Vt[1]
                u3 = u3 - (u1 @ u3) * u1 - (u2 @ u3) * u2
                if np.linalg.norm(u3) < eps:
                    # fabricate a perpendicular axis
                    e = np.zeros_like(u1); e[0] = 1.0
                    u3 = e - (u1 @ e) * u1 - (u2 @ e) * u2
                u3 /= (np.linalg.norm(u3) + eps)
            else:
                # fabricate u3 in the plane orthogonal to u1, u2
                e = np.zeros_like(u1); e[0] = 1.0
                u3 = e - (u1 @ e) * u1 - (u2 @ e) * u2
                u3 /= (np.linalg.norm(u3) + eps)

            # coordinates: X=u2·ψ, Y=u3·ψ, Z=u1·ψ (Z is cos to goal since U is unit)
            X = U @ u2
            Y = U @ u3
            Z = U @ u1
            c = np.clip(Z, -1, 1)

            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(X, Y, Z, c=c, cmap="viridis", s=12, alpha=0.9)
            cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("cos(ψ(s), ψ(goal))")

            # unit sphere wireframe for reference
            u = np.linspace(0, 2*np.pi, 60)
            v = np.linspace(0, np.pi, 30)
            xs = np.outer(np.cos(u), np.sin(v))
            ys = np.outer(np.sin(u), np.sin(v))
            zs = np.outer(np.ones_like(u), np.cos(v))
            ax.plot_wireframe(xs, ys, zs, color='lightgray', linewidth=0.3, alpha=0.5)

            # mark goal at (0,0,1)
            ax.scatter([0], [0], [1], c='red', s=80, marker='*', label='goal (u1)')

            ax.set_xlabel("u2 (orthogonal to goal)")
            ax.set_ylabel("u3 (orthogonal to goal)")
            ax.set_zlabel("u1 (along goal)")
            ax.set_title("ψ projected to goal-aligned 3D frame")

            # equal aspect ratio
            max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
            cx, cy, cz = X.mean(), Y.mean(), Z.mean()
            ax.set_xlim(cx - max_range/2, cx + max_range/2)
            ax.set_ylim(cy - max_range/2, cy + max_range/2)
            ax.set_zlim(cz - max_range/2, cz + max_range/2)

            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)

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
        plot_goal_aligned_3d(psi, goal_vec, os.path.join(run_dir, "psi_goal_frame_3d.png"))
        # optional:
        plot_pca_3d(psi, os.path.join(run_dir, "psi_pca_3d.png"))

