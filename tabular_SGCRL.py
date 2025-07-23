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
HEIGHT, WIDTH = 20, 20
GOAL_COORD = (19, 19)                           # row 29, col 29 (zero-indexed)
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
NUM_STATES     = HEIGHT * WIDTH
NUM_ACTIONS    = 5           # stay, down, up, right, left
A_TO_DELTA     = np.array([[0, 0],
                           [1, 0], [-1, 0],
                           [0, 1], [0, -1]])
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
    # plotting frequency (in *update* intervals)
    p.add_argument("--plot-mult",        type=int,   default=1_000,
                   help="save plots every plot-mult * episodes_per_upd episodes")
    p.add_argument("--random_exploration",         type=bool, default=True)
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
alpha            = args.alpha
# plot_mult        = args.plot_mult
plot_mult        = num_episodes // 30
entropy_coeff   = args.entropy_coeff
norm = True
random_exploration = args.random_exploration
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

goal = np.ravel_multi_index(GOAL_COORD, walls.shape)
  # <-- add once, near other hyper-params
# phi = np.random.randn(NUM_STATES, NUM_ACTIONS, rep_dim) * 0.1  # φ(s,a)
# psi = np.random.randn(NUM_STATES,                 rep_dim) * 0.1  # ψ(s)
# -------- Tabular parameters -------------------------------------------------
phi = np.random.randn(NUM_STATES, NUM_ACTIONS, rep_dim) * 0.1  # φ(s,a)

# --- NEW: distance-aware ψ initialisation -----------------------------------
coords      = np.column_stack(np.unravel_index(np.arange(NUM_STATES), walls.shape))
goal_coord  = coords[goal]                     # (row, col) of the goal state
max_dist    = np.linalg.norm([HEIGHT - 1, WIDTH - 1])  # farthest possible corner

psi         = np.empty((NUM_STATES, rep_dim))
psi_goal    = np.random.randn(rep_dim) * 0.1             # anchor embedding
psi[goal]   = psi_goal                                   # set goal’s ψ once

## initialize representations in a distance-aware manner
for s, c in enumerate(coords):
    if s == goal:
        continue
    d         = np.linalg.norm(c - goal_coord)          # L2 to goal
    frac      = (d / max_dist) ** alpha                # 0 near goal → 1 far away
    var_scale = near_var + (far_var - near_var) * frac # extreme scaling
    print(f"State {s} at {c}: dist={d:.2f}, frac={frac:.2f}, var_scale={var_scale:.2f}")
    psi[s]    = psi_goal + np.random.randn(rep_dim) * 0.1 * var_scale
    ## each element's is around 0.1 so the norm for a 16 d vec is around 0.1 * 4 = 0.4
    #psi[s] = psi_goal
    
    ## all randomly initialized
    #psi[s] = np.random.randn(rep_dim) * 0.1

# Normalize ψ to unit vectors (L2 norm)
if norm :
    psi_norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-8
    psi /= psi_norms
# # -------- ε-greedy goal-directed policy -------------------------------------
# def select_action(s: int, g: int) -> int:
#     if np.random.rand() < epsilon:
#         return np.random.randint(NUM_ACTIONS)
#     dots = phi[s] @ psi[g]                    # shape (5,)
#     return int(dots.argmax())


# def select_action(s: int, g: int) -> int:
#     # ε-greedy over next-state ψ similarity to ψ(goal)
#     if np.random.rand() < epsilon:
#         return np.random.randint(NUM_ACTIONS)

#     goal_vec = psi[g]                 # (rep_dim,)
#     best_a   = 0
#     best_val = -np.inf

#     for a in range(NUM_ACTIONS):
#         ns   = step(s, a)             # next state (could be same if wall)
#         #val  = psi[ns] @ goal_vec     # dot similarity
#         goal_norm  = np.linalg.norm(goal_vec) + 1e-8     
#         val  = psi[ns] @ goal_vec / (np.linalg.norm(psi[ns]) * goal_norm + 1e-8)
#         if val > best_val:
#             best_val = val
#             best_a   = a
#     return best_a

if random_exploration:
    random_goal_psi = np.random.randn(rep_dim) * 0.1
    print(f"[INFO] Using random exploration with ψ(goal) = {random_goal_psi[:5]}...")

def select_action(s: int, g: int) -> int:
    """
    Softmax action selection based on ψ(s') · ψ(g) similarity,
    scaled by 1 / entropy_coeff (i.e., inverse temperature)
    """
    goal_vec = psi[g]  # (rep_dim,)
    #goal_norm = np.linalg.norm(goal_vec) + 1e-8
    if random_exploration:
        goal_vec = random_goal_psi

    similarities = []
    for a in range(NUM_ACTIONS):
        ns = step(s, a)
        sim = psi[ns] @ goal_vec  # dot similarity (or cosine if normalized)
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





def update_representations() -> float:
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
        w = gamma ** np.arange(remaining)
        w /= w.sum()
        j = i + np.random.choice(remaining, p=w)

        s_list.append(traj[i])
        sp_list.append(traj[j])

    if not s_list:                        # batch could be empty
        return 0.0

    s_batch  = np.asarray(s_list,  dtype=np.int32)   # anchors (B,)
    sp_batch = np.asarray(sp_list, dtype=np.int32)   # positives (B,)
    B        = len(s_batch)

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

    # ----- 5. Positive-state updates  Δψ(sp_k) ----------------------------
    expected_anchor = (P.T @ psi_s)                       # (B,d)
    pos_update      = lr_phi_psi * (psi_s - expected_anchor)
    np.add.at(psi, sp_batch, pos_update)                  # scatter-add

    # Normalize ψ to unit vectors (L2 norm)
    if norm:
        psi_norms = np.linalg.norm(psi, axis=1, keepdims=True) + 1e-8
        psi /= psi_norms

    return nll





# def update_representations() -> float:
#     """
#     Sequential ψ-only contrastive update.
#     For a minibatch of B (anchor_state s_k, positive_state sp_k) pairs, apply:

#         p_j(k) = softmax_j( ψ(s_j)·ψ(sp_k) )

#     Updates:
#         Δψ(s_k)      =  η (1 - p_k(k)) ψ(sp_k)
#         Δψ(s_j)      = -η p_j(k) ψ(sp_k)                  for j ≠ k
#         Δψ(sp_k)     =  η (1 - p_k(k)) ψ(s_k) - η Σ_{j≠k} p_j(k) ψ(s_j)

#     Returns mean NLL:  (1/B) Σ_k -log p_k(k)
#     """
#     global psi

#     if len(replay) < 2:
#         return 0.0

#     # -------- 1. Sample minibatch of (s, sp) pairs (same logic as before) ----
#     traj_ids = np.random.choice(len(replay), batch_size, replace=True)
#     s_list, sp_list = [], []
#     for idx in traj_ids:
#         traj = replay[idx]
#         if len(traj) < 2:
#             continue
#         i = np.random.randint(0, len(traj) - 1)

#         remaining = len(traj) - i
#         weights = gamma ** np.arange(remaining)
#         weights /= weights.sum()
#         j = i + np.random.choice(remaining, p=weights)

#         s_list.append(traj[i])   # anchor state
#         sp_list.append(traj[j])  # positive future state

#     if not s_list:
#         return 0.0

#     s_batch  = np.asarray(s_list,  dtype=np.int32)   # (B,)
#     sp_batch = np.asarray(sp_list, dtype=np.int32)   # (B,)
#     B = len(s_batch)

#     # Snapshot of anchor embeddings (we refresh after each column update
#     # to keep "sequential dependency" semantics like your original loop).
#     anchor_snapshot = psi[s_batch].copy()

#     nll = 0.0

#     # -------- 2. Per-positive column updates --------------------------------
#     for k in range(B):
#         pos_state = sp_batch[k]
#         psi_pos   = psi[pos_state]                   # ψ(sp_k): current value

#         # logits over anchors j for this column k
#         dots = anchor_snapshot @ psi_pos             # (B,)
#         m = dots.max()
#         exp_logits = np.exp(dots - m)
#         probs = exp_logits / exp_logits.sum()        # p_j(k)
#         p_k = probs[k]

#         nll -= np.log(p_k + 1e-12)

#         # --- Anchor updates -------------------------------------------------
#         # Positive anchor j = k:
#         psi[s_batch[k]] += lr_phi_psi * (1.0 - p_k) * psi_pos

#         # Negative anchors j ≠ k:
#         if B > 1:
#             neg_mask = np.ones(B, dtype=bool)
#             neg_mask[k] = False
#             neg_ids = s_batch[neg_mask]
#             neg_coeffs = probs[neg_mask]            # p_j(k)
#             # For each negative: Δψ(s_j) = -η p_j(k) ψ(sp_k)
#             # Use add.at to handle duplicates safely
#             np.add.at(psi, neg_ids,
#                       -lr_phi_psi * neg_coeffs[:, None] * psi_pos[None, :])

#         # --- Positive state update -----------------------------------------
#         # sum_{j≠k} p_j(k) ψ(s_j)
#         if B > 1:
#             sum_pj_psi_neg = (probs[neg_mask, None] *
#                               anchor_snapshot[neg_mask]).sum(axis=0)
#         else:
#             sum_pj_psi_neg = 0.0

#         # Δψ(sp_k) = η(1 - p_k) ψ(s_k) - η Σ_{j≠k} p_j(k) ψ(s_j)
#         psi[pos_state] += lr_phi_psi * (
#             (1.0 - p_k) * anchor_snapshot[k] - sum_pj_psi_neg
#         )

#         # Refresh snapshot so later columns see the changed anchors
#         anchor_snapshot = psi[s_batch]

#     return nll / B
#-------- Sequential contrastive update -------------------------------------




## sequential version of the phi psi
# def update_representations() -> float:
#     """
#     Soft-max contrastive update.

#     For a minibatch of B triples (s_i , a_i , s⁺_i):

#         p_j(k) = exp( φ(s_j,a_j)·ψ(s⁺_k) ) / Σ_m exp( φ(s_m,a_m)·ψ(s⁺_k) )

#     Updates for anchor k (index k in the minibatch):

#         Δφ(s_k,a_k)      =  η (1 - p_k(k)) ψ(s⁺_k)
#         Δψ(s⁺_k)         =  η (1 - p_k(k)) φ(s_k,a_k)  -  η Σ_{j≠k} p_j(k) φ(s_j,a_j)
#         Δφ(s_j,a_j)      = -η p_j(k) ψ(s⁺_k)            for every j ≠ k
#     """
#     global phi, psi

#     if len(replay) < 2:
#         return 0.0

#     # ---------- 1. build minibatch -----------------------------------------
#     traj_ids = np.random.choice(len(replay), batch_size, replace=True)

#     s_batch, a_batch, sp_batch = [], [], []   # anchor s, anchor a, positive s⁺
#     for idx in traj_ids:
#         traj = replay[idx]
#         if len(traj) < 2:
#             continue

#         # anchor index i
#         i = np.random.randint(0, len(traj) - 1)

#         # positive index j ≥ i chosen with geometric weights γ^k
#         remaining = len(traj) - i
#         weights   = gamma ** np.arange(remaining)
#         weights  /= weights.sum()
#         offset    = np.random.choice(remaining, p=weights)
#         j = i + offset

#         s_batch.append(traj[i])
#         sp_batch.append(traj[j])
#         a_batch.append(select_action(traj[i], traj[j]))

#     B = len(s_batch)
#     if B == 0:
#         return 0.0

#     s_batch  = np.array(s_batch,  dtype=int)
#     a_batch  = np.array(a_batch,  dtype=int)
#     sp_batch = np.array(sp_batch, dtype=int)

#      # ---------- 2. gather embeddings ---------------------------------------
#     phi_batch = phi[s_batch, a_batch]              # shape (B, rep_dim)

#     nll = 0.0                                      # negative-log-likelihood accumulator

#     # ---------- 3. per-anchor soft-max update ------------------------------
#     for k in range(B):
#         psi_pos = psi[sp_batch[k]]                 # ψ(s⁺_k)     (rep_dim,)

#         # soft-max probs p_j(k)
#         dots       = phi_batch @ psi_pos           # (B,)
#         dots_max   = dots.max()                    # for numerical stability
#         exp_logits = np.exp(dots - dots_max)
#         probs      = exp_logits / exp_logits.sum() # p_j(k)
#         p_k        = probs[k]

#         nll -= np.log(p_k + 1e-12)                # accumulate −log p_k

#         # mask for negatives j ≠ k
#         neg_mask = np.ones(B, dtype=bool)
#         neg_mask[k] = False

#         # Σ_{j≠k} p_j φ_j   (needed for Δψ)
#         sum_pj_phi_neg = (probs[neg_mask, None] * phi_batch[neg_mask]).sum(axis=0)

#         # ------ Δφ updates -------------------------------------------------
#         phi[s_batch[k], a_batch[k]]        += lr_phi_psi * (1.0 - p_k) * psi_pos
#         phi[s_batch[neg_mask], a_batch[neg_mask]] += (
#             -lr_phi_psi * probs[neg_mask, None] * psi_pos)

#         # ------ Δψ update (only for s⁺_k) ----------------------------------
#         psi[sp_batch[k]] += lr_phi_psi * ((1.0 - p_k) * phi_batch[k] - sum_pj_phi_neg)

#     # ---------- 4. return mean −log p_k ------------------------------------
#     return nll / B

# Precompute geometric weight tables up to max_steps+1 (trajectory length upper bound)
# _geometric_tables = {}
# def geom_weights(length):
#     w = _geometric_tables.get(length)
#     if w is None:
#         arr = gamma ** np.arange(length)
#         _geometric_tables[length] = arr / arr.sum()
#         w = _geometric_tables[length]
#     return w


# def update_representations() -> float:
#     global phi, psi
#     if len(replay) < 2:
#         return 0.0

#     # ---- 1. Sample minibatch (anchors & positives) ----
#     traj_ids = np.random.choice(len(replay), batch_size, replace=True)

#     s_list, sp_list = [], []
#     for idx in traj_ids:
#         traj = replay[idx]
#         if len(traj) < 2:
#             continue
#         i = np.random.randint(0, len(traj) - 1)
#         remaining = len(traj) - i
#         weights = geom_weights(remaining)
#         j = i + np.random.choice(remaining, p=weights)
#         s_list.append(traj[i])
#         sp_list.append(traj[j])

#     if not s_list:
#         return 0.0

#     s_batch  = np.array(s_list, dtype=int)
#     sp_batch = np.array(sp_list, dtype=int)

#     # Vectorized ε-greedy action selection relative to positive state
#     psi_pos_batch = psi[sp_batch]                                 # (B, D)
#     phi_all = phi[s_batch]                                        # (B, A, D)
#     scores  = np.einsum('bad,bd->ba', phi_all, psi_pos_batch)     # (B, A)
#     a_batch = scores.argmax(axis=1)
#     if epsilon > 0:
#         mask = (np.random.rand(len(a_batch)) < epsilon)
#         a_batch[mask] = np.random.randint(NUM_ACTIONS, size=mask.sum())

#     phi_batch = phi[s_batch, a_batch]                             # (B, D)
#     S = psi_pos_batch                                             # rename for formulas
#     B = phi_batch.shape[0]

#     # ---- 2. Compute logits & softmax matrix L[j,k] = φ_j · ψ_pos_k ----
#     # L shape (B, B)
#     L = phi_batch @ S.T
#     L -= L.max(axis=0, keepdims=True)       # stability per column
#     np.exp(L, out=L)
#     L /= L.sum(axis=0, keepdims=True)
#     P = L                                    # now P[j,k] = p_j(k)

#     # Negative log-likelihood: mean -log p_k(k)
#     diag = np.diag(P)
#     nll = -np.log(diag + 1e-12).mean()

#     # ---- 3. Vectorized gradients ----
#     # Δφ_j = η ( ψ_pos_j - Σ_k P[j,k] ψ_pos_k )
#     sum_P_row = P @ S                        # (B, D)
#     dphi_batch = lr_phi_psi * (S - sum_P_row)

#     # Δψ_k = η ( φ_k - Σ_j P[j,k] φ_j )
#     sum_P_col = P.T @ phi_batch              # (B, D)
#     dpsi_batch = lr_phi_psi * (phi_batch - sum_P_col)

#     # ---- 4. Scatter-add into global tables (handle duplicates) ----
#     # (If duplicates appear, accumulate with np.add.at)
#     np.add.at(phi, (s_batch, a_batch), dphi_batch)
#     np.add.at(psi, sp_batch, dpsi_batch)

#     return nll


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

        loss = update_representations()
        loss_history.append(loss)

                # ------- visualise ψ(s)·ψ(g) + trajectories ------------------------
        
    if ep %  plot_mult  == 0 or ep == episodes_per_upd :

        eval_ep_success = run_eval_episode()
        eval_success_list.append(np.mean(eval_ep_success))
       
        #1. similarity heat-map  ψ(s)·ψ(g)  over the whole maze
        goal_vec   = psi[goal]  

        if random_exploration:
            goal_vec = random_goal_psi     
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


