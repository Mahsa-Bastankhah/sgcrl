# (don't modify)
#from utils_exploration_psi_norms import eval_and_get_cells_visited, get_psi_norms, load_checkpoint
import sys
sys.path.append("/home/gliu2/contrastive_rl")
import os
import copy
import functools

from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import jax
import optax
import numpy as np
from acme import specs
import tensorflow as tf

from acme.tf.savers import SaveableAdapter

from contrastive.config import ContrastiveConfig
from contrastive import utils as contrastive_utils
from contrastive import make_networks
from contrastive.utils import make_environment
from contrastive import ContrastiveLearner

# disable tensorflow_probability warning: The use of `check_types` is deprecated and does not have any effect.
import logging
logger = logging.getLogger("root")

class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

logger.addFilter(CheckTypesFilter())

# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler
# import seaborn as sn
# import pandas as pd
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.mahsa_utils import eval_and_get_cells_visited, get_psi_norms, load_checkpoint, subspace_transferability, get_cell_projected_psi_norms, get_sa_repr_subspace_basis

import pickle
# CHANGE ME
from tqdm import tqdm
alpha = "0.1"
env_name = 'point_Spiral11x11'
#env_name = 'point_Impossible'
seed = 4
log_dir = f'logs'
alg = 'contrastive_cpc'
ckpt_list = [1 , 40, 80, 120, 160, 200, 240, 250]
ckpt_list =  [1,  2, 3, 4, 5, 6 , 7, 8, 9, 10]


NUM_AXES = 2
NUM_EPISODES = 5
project = False

if env_name == 'point_Spiral11x11':
    point_map = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
    axes_lims = {
        0: (0, 11),
        1: (0, 11),
    }
    ct = 11
    EPISODE_LENGTH = 100
elif env_name == 'point_Impossible':
    point_map = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 1, 0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0, 1, 0, 1, 1],
                  [0, 1, 0, 0, 0, 1, 0, 1, 0]])
    axes_lims = {
        0: (0, 9),
        1: (0, 9),
    }
    ct = 9
    EPISODE_LENGTH = 50


for ckpt_num in tqdm(ckpt_list):
    print("-----------------------------------------------------------")
    axes_names = ['x', 'y']
    

    # roll out + compute sample fixed goal trajectories
    print(log_dir)
    returns_dict, success_rates_dict, positions_dict, goals_dict, cells = eval_and_get_cells_visited(env_name, log_dir, seed, ckpt_num, grid_width=0.01, NUM_EPISODES=NUM_EPISODES)

    # compute psi norms
    ### guide: if you wanna plot psi similarity and ranks uncomment the below line and the subspace line
    ## if you wanna plot the projected psi use the second function get cell projected psi 
    if project is False:
        goal_locations, psi_norms, critic_sf, critic_g, psi_similarity, waypoint_similarity,actions,env, trained_learner_state, networks = get_psi_norms(env_name, log_dir,seed, ckpt_num, project= project)
        get_sa_repr_subspace_basis(env, trained_learner_state, networks, goal_locations)
    else:
        goal_locations, psi_norms, critic_sf, critic_g, psi_similarity, waypoint_similarity,actions,env, trained_learner_state, networks = get_cell_projected_psi_norms(env_name, log_dir,seed, ckpt_num, project= project, point_map = point_map)

    #subspace_transferability(env=env, trained_learner_state=trained_learner_state, networks=networks, point_map=point_map, env_name=env_name, seed=seed, ckpt_num=ckpt_num)
    pos_arr = np.array(positions_dict[seed]).reshape(-1, NUM_AXES)
    pos_id_arr = np.broadcast_to(np.expand_dims(np.arange(NUM_EPISODES), 1), (NUM_EPISODES, EPISODE_LENGTH)).flatten()
    goals_arr = np.array(goals_dict[seed])
    starts_arr = np.array(positions_dict[seed])[:, 0, :]

    # Create figure with three subplots
    # fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    max_y = ct
    
    # plot_data = [
    #     (psi_similarity, "Psi Similarity"),
    #     (waypoint_similarity, "Waypoint Similarity"),
    #     (critic_sf, "Value of S_f"),
    # ]


    ## only plotting the psi similarity
    plot_data = [(psi_similarity, "Psi Similarity")]
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    data , title = plot_data[0]


    arrow_scale = 0.2  # scale for visualization
    dx = actions[:, 1] * arrow_scale
    dy = -actions[:, 0] * arrow_scale  # NEGATE for correct display (invert y-axis)

    #### if you wanna plot heatmpas that have negative values too

    # Step 1: Create a custom colormap
    coolwarm = plt.get_cmap("coolwarm", 128)  # Only for positive values
    neg_shades = plt.get_cmap("Greens_r", 128)  # For negative values (reversed so darker = more neg)

    # Combine: negative colormap (0→-1) and coolwarm (0→+1)
    colors = np.vstack([
        neg_shades(np.linspace(0.1, 0.5, 128)),  # customize start to make it darker
        coolwarm(np.linspace(0, 1, 128)),
    ])
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("negpos_cmap", colors)

    # Step 2: Normalize so that 0 is the center
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    # for ax, (data, title) in zip(axes, plot_data):
        # Plot map and heatmap
    ax.imshow(point_map, cmap='binary', extent=[0, ct, 0, ct])
    heatmap = ax.scatter(goal_locations[:, 1], max_y - goal_locations[:, 0], c=data, s=4, alpha=0.8, cmap=custom_cmap, norm=norm)
    plt.colorbar(heatmap, ax=ax, label=title.lower())
    
    # Plot trajectories
    for epi in range(NUM_EPISODES):
        epi_slice = slice(epi*EPISODE_LENGTH, (epi+1)*EPISODE_LENGTH)
        ax.plot(pos_arr[epi_slice, 1], max_y - pos_arr[epi_slice, 0], label=epi, alpha= 0.5)
    
    # Plot start and goal points
    ax.scatter(starts_arr[:, 1], max_y - starts_arr[:, 0], c='red', marker='D', s=30)
    ax.scatter(goals_arr[:, 1], max_y - goals_arr[:, 0], c='orange', marker='*', s=80)
    # --- Overlay arrows for policy directions
    # Choose 1/10 of the points at random
    N = len(goal_locations)
    fraction = 0.15
    sample_size = int(fraction * N)
    subset_idx = np.random.choice(N, size=sample_size, replace=False)

    ## plot the optimal action
    # ax.quiver(
    #     goal_locations[subset_idx, 1],
    #     max_y - goal_locations[subset_idx, 0],
    #     dx[subset_idx],
    #     dy[subset_idx],
    #     angles='xy',
    #     scale_units='xy',
    #     scale=1,  # keep the arrow length the same
    #     color='green',
    #     width=0.003,
    #     alpha=0.3,
    #     headwidth=5,         # default is 3 — increase this
    #     headlength=7,        # default is 5 — increase this
    #     headaxislength=6     # default is 4.5 — increase this
    # )


    # Set labels and limits
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(axes_lims[0])
    ax.set_ylim(axes_lims[1])
    ax.set_title(f"{title} - checkpoint {ckpt_num}")

    plt.tight_layout()
    # Create directory path
    plot_dir = os.path.join("experiments/plots", f"{env_name}_{seed}")
    os.makedirs(plot_dir, exist_ok=True)
    if project is True:
        name = "psi_projected"
    else:
        name = "psi"
    # Save the figure as PDF or PNG — use whichever you prefer
    save_path = os.path.join(plot_dir, f"{name}_{ckpt_num}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to: {save_path}")