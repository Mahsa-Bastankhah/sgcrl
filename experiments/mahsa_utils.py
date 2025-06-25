import numpy as np
import jax
import optax
from copy import deepcopy
from acme import specs
import tensorflow as tf
import functools
from acme.tf.savers import SaveableAdapter
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import HTML
from tqdm import tqdm
import os
from itertools import product
import numpy as np
import jax.numpy as jnp
from contrastive.config import ContrastiveConfig
from contrastive import utils as contrastive_utils
from contrastive import make_networks
from contrastive.utils import make_environment
from contrastive import ContrastiveLearner
from sklearn.cluster import AgglomerativeClustering
from matplotlib import colors
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.linalg import subspace_angles
import json, os
from types import SimpleNamespace


# disable tensorflow_probability warning: The use of `check_types` is deprecated and does not have any effect.
import logging
logger = logging.getLogger("root")

class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()

# hyperparams (mostly unimportant)

# NOTE: uses cpc and L2 critic
config = ContrastiveConfig()



obs_repr_shape = 4
state_repr_shape = 2
action_repr_shape = 2
action_list = [[1,0], [1,1], [0,1], [0,0], [-1,0], [-1,-1], [0,-1], [-1,1], [1,-1]]
# observation = np.array([5,5,10,10], dtype=float)

logger.addFilter(CheckTypesFilter())

axes_names = ['x', 'y']
def get_point_map(env_name):
    """Return the 0/1 occupancy grid (point_map) for each point-nav env."""
    if env_name == "point_Spiral11x11":
        return np.array([
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0],
            [1,0,1,1,1,1,1,1,1,1,0],
            [1,0,1,0,0,0,0,0,0,1,0],
            [1,0,1,0,1,1,1,1,0,1,0],
            [1,0,1,0,1,0,0,1,0,1,0],
            [1,0,1,0,1,1,0,1,0,1,0],
            [1,0,1,0,0,0,0,1,0,1,0],
            [1,0,1,1,1,1,1,1,0,1,0],
            [1,0,0,0,0,0,0,0,0,1,0],
            [1,1,1,1,1,1,1,1,1,1,0],
        ], dtype=int)

    elif env_name == "point_Impossible":
        return np.array([
            [0,1,0,0,0,0,0,0,0],
            [0,1,0,1,1,1,1,1,0],
            [0,1,0,0,0,0,1,0,0],
            [0,1,1,1,1,0,1,0,1],
            [0,1,0,0,0,0,1,0,0],
            [0,1,0,1,1,1,1,1,0],
            [0,0,0,1,0,0,0,1,0],
            [0,1,0,1,0,1,0,1,1],
            [0,1,0,0,0,1,0,1,0],
        ], dtype=int)

    elif env_name == "point_Wall11x11":
        return np.array([
            [0,0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0],
        ], dtype=int)

    else:
        raise ValueError(f"No point_map defined for env '{env_name}'")
# ----------------------------------------------------------


def load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals = False, ckpt_num = None, NUM_EPISODES = 10, alg = 'contrastive_cpc', uid = None):
    state_entropy_coefficient = alpha
    config = ContrastiveConfig()
    ckpt_dir = '{}/{}_{}_{}/checkpoints/learner'.format(log_dir, alg, env_name,seed)
    if uid is not None:
        ckpt_dir = '{}/{}_{}_{}/{}/checkpoints/learner'.format(log_dir, alg, env_name, seed, uid)
    #ckpt_dir = '{}/{}_{}_{}_{}/checkpoints/learner'.format(log_dir, alg, env_name, misc_params, seed)
    
    fixed_start_end = None
    fixed_goal = False
        
    if fix_goals:
        if env_name == 'point_Spiral11x11':
            fixed_start_end = [np.array([5,5], dtype=float), np.array([10,10], dtype=float)]
        elif env_name == 'point_FourRooms':
            fixed_start_end = [np.array([0.5,0.5], dtype=float), np.array([8.5,8.5], dtype=float)]
        elif env_name == 'point_Impossible':
            fixed_start_end = [np.array([9,0], dtype=float), np.array([0,9], dtype=float)]
        elif env_name == 'point_Wall11x11':
            fixed_start_end = [np.array([2,8], dtype=float), np.array([0,10], dtype=float)]
            #fixed_start_end = [np.array([5,5], dtype=float), np.array([0,10], dtype=float)]
    else:
        fixed_start_end = None
    


    run_root = os.path.dirname(os.path.dirname(ckpt_dir))      # “…/ALG_ENV_SEED[/uid]”
    cfg_path = os.path.join(run_root, "config.json")

    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            override = json.load(f)
        print(f"[config] overriding from {cfg_path}", flush=True)

        # ensure dot-access even for new keys
        if not isinstance(config, SimpleNamespace):
            config = SimpleNamespace(**vars(config))

        # overwrite only the keys present in the JSON
        for k, v in override.items():
            setattr(config, k, v)
    else:
        print(f"[config] {cfg_path} not found – using code defaults", flush=True)
    
    env_factory = lambda seed: make_environment(env_name, config.start_index, 
                                                config.end_index, seed=np.random.randint(1e6), fixed_start_end=fixed_start_end)[0] 
    dummy_seed = 1
    environment_spec = specs.make_environment_spec(env_factory(np.random.randint(1e6)))

    obs_dim = make_environment(env_name, config.start_index, config.end_index, seed=np.random.randint(1e6), fixed_start_end=fixed_start_end)[1]
                                   
    network_factory = functools.partial(
        make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
        repr_norm=config.repr_norm, twin_q=False,
        use_image_obs=config.use_image_obs,
        hidden_layer_sizes=config.hidden_layer_sizes, config=config
    )

    random_key = jax.random.PRNGKey(np.random.choice(int(1e6)))
    networks = network_factory(environment_spec)
    policy_optimizer = optax.adam(
        learning_rate=config.actor_learning_rate)
    q_optimizer = optax.adam(.001)#learning_rate=config.critic_learning_rate
    print(config)

    trained_learner = ContrastiveLearner(
        networks=networks,
        rng=random_key,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        iterator=None,
        counter=None,
        logger=None,
        obs_to_goal=functools.partial(contrastive_utils.obs_to_goal_2d,
                                      start_index=config.start_index,
                                      end_index=config.end_index),
        config=config)

    returns_list = []
    success_rate_list = []
    
    env = env_factory(np.random.randint(1e6))
    obs_dim = env.observation_spec().shape[0] // 2
    episode_returns = np.zeros([NUM_EPISODES, ])

    ckpt = tf.train.Checkpoint(learner=SaveableAdapter(trained_learner))
    ckpt_mgr = tf.train.CheckpointManager(ckpt, ckpt_dir, 1)
    if ckpt_num is not None:
        ckpt_path = ckpt_dir + '/ckpt-' + str(ckpt_num)
        ckpt.restore(ckpt_path).assert_consumed()
        print(ckpt_path)
    else:
        ckpt.restore(ckpt_mgr.latest_checkpoint).assert_consumed()
        print(ckpt_mgr.latest_checkpoint)

    trained_learner_state = trained_learner._state
    
    print("Model loaded from: {}".format(ckpt_dir))
    return trained_learner_state, env, networks

def eval_fixed_goal_exploration(alpha, misc_params, env_name, log_dir, seeds_list, ckpt_num, NUM_EPISODES=10, action_mode="actor_max", uid = None):
    success_rates_dict = {}
    returns_dict = {}
    goals_dict = {}
    
    positions_dict = {}
    for seed in seeds_list:
        trained_learner_state, env, networks = load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals = True, ckpt_num = ckpt_num, uid = uid)
        print("I am evaluating inside the eval fixed goal function and my action mode is: ", action_mode)
        episode_returns = np.zeros(NUM_EPISODES)
        positions_dict[seed] = []
        goals_dict[seed] = []
        
        for epi in tqdm(range(NUM_EPISODES)):
            t = 0
            timestep = env.reset()
            episode_return = 0
            positions_dict[seed].append([])

            while not timestep.last():
                obs = timestep.observation

                if action_mode == "q_max":
                    # Discretize action space: 10 values per dimension between -1 and 1
                    action_dim = env.action_spec().shape[0]
                    action_vals = np.linspace(-1, 1, 10)
                    grid = np.meshgrid(*([action_vals] * action_dim))
                    actions = np.stack([g.ravel() for g in grid], axis=1)  # shape: (num_actions, action_dim)

                    obs_batch = np.tile(obs, (actions.shape[0], 1))
                    
                    # Evaluate Q-values
                    q_values, _, _ = networks.q_network.apply(
                        trained_learner_state.q_params,
                        obs_batch,
                        actions
                    )
                    q_values = np.asarray(q_values)
                    q_values = np.diag(q_values)  # shape: (100,)
                    # print("q_values shape: ", q_values.shape)
                    q_values = np.asarray(q_values).flatten()
                    q_values = np.asarray(q_values).squeeze()
                    # print("q_values shape: ", q_values.shape)

                    best_action = actions[np.argmax(q_values)]
                    action = best_action
                    action = best_action.astype(np.float32)
                    # print("Best action shape: ", action)

                    
                if action_mode == "actor_max":
                    dist = networks.policy_network.apply(
                    trained_learner_state.policy_params,
                    obs
                    )
                    action = np.array(dist.mode())
                elif action_mode == "actor_sample":
                    print("obs", obs)
                    # 1. Create or update a PRNG key
                    rng_key = jax.random.PRNGKey(seed)  # or split from an existing key

                    # 2. Apply the policy to get the distribution
                    rng_key, subkey1, subkey2 = jax.random.split(rng_key, 3)
                    dist = networks.policy_network.apply(
                        trained_learner_state.policy_params,
                        # subkey1,  # Pass PRNGKey to .apply() if randomness is used inside
                        obs
                    )

                    # 3. Sample from the distribution
                    action = np.array(dist.sample(seed=subkey2))


                    import inspect

                    def unwrap_normal(d):
                        """
                        Follow the .distribution chain until we reach something that
                        *has* a loc / scale (or mean / stddev) attribute.
                        Works for TFP-JAX, Distrax, or Haiku convenience wrappers.
                        """
                        while hasattr(d, "distribution"):
                            # Stop if the next layer is identical (safety against infinite loops)
                            nxt = d.distribution
                            if nxt is d or nxt is None:
                                break
                            d = nxt
                        return d
                    # ------------------------------------------------------------
                    # unwrap until we land on the Normal
                    base = unwrap_normal(dist)

                    # access parameters robustly
                    if hasattr(base, "loc") and hasattr(base, "scale"):    # TFP / Distrax Normal
                        mean = np.asarray(base.loc)
                        std  = np.asarray(base.scale)
                    else:                                                  # fall back on methods
                        mean = np.asarray(base.mean())
                        std  = np.asarray(base.stddev())

                    var   = std ** 2

                    print(f"[policy] mean={mean}, std={std}, var={var}")
                    # === Already have `mean` and `std` from the base Normal ===
                    low  = np.tanh(mean - std)   # lower edge (mean − 1 σ)
                    high = np.tanh(mean + std)   # upper edge (mean + 1 σ)

                    print(f"[tanh-range] 1σ band maps to ≈ [{high-low}] in tanh space")



                timestep = env.step(action)
                positions_dict[seed][-1].append(deepcopy(env.state))

                t += 1
                episode_return += timestep.reward

            # print("episode return = {}".format(episode_return))
            episode_returns[epi] = episode_return
            goals_dict[seed].append(deepcopy(env.goal))

        returns_dict[seed] = episode_returns
        success_rates_dict[seed] = episode_returns >= 1
        print("avg episode return: {}".format(np.mean(returns_dict[seed])))
        print("success rate: {}".format(np.mean(success_rates_dict[seed])))
        print()

    return returns_dict, success_rates_dict, positions_dict, goals_dict
  
def get_cells_visited(positions, grid_width=0.005, grid_min=0, grid_max=11, num_dims=2, uid = None):
    '''
    positions: agent (x,y,z)-positions per episode per batch; (batch_size, episodes_length, num_dims=2)
    returns average unique gridcells visited per episode, at grid_width resolution
    '''
    assert grid_min <= np.min(positions) and np.max(positions) <= grid_max
    grid_bins = np.arange(grid_min, grid_max, grid_width)
    
    pos_ids_by_dim = []
    for dim in range(num_dims):
        pos_ids_by_dim.append(np.digitize(positions[:,:,dim], grid_bins))
    pos_ids = np.stack(pos_ids_by_dim, axis=-1)
    
    num_episodes, episode_len, _ = pos_ids.shape
    cells_visited = []
    for epi in range(num_episodes):
        cells_visited.append(len(set(np.array_str(pos) for pos in pos_ids[epi])))
    return cells_visited

def eval_and_get_cells_visited(env_name, log_dir, seed, ckpt_num=None, grid_width=0.01,
                               alpha='0.1', render_images=False,uid=None, **kwargs):
    misc_params = '{}_None'.format(alpha)

    returns_dict, success_rates_dict, positions_dict, goals_dict = eval_fixed_goal_exploration(alpha, misc_params, env_name=env_name, log_dir=log_dir, seeds_list=[seed], ckpt_num=ckpt_num, uid=uid, **kwargs)
    positions = np.array(positions_dict[seed])
    return returns_dict, success_rates_dict, positions_dict, goals_dict, get_cells_visited(positions, grid_width=grid_width, uid=uid)
      
      
def get_psi_norms(env_name, log_dir,  seed, ckpt_num=None, alg = 'contrastive_cpc', alpha = '0.1',
                  state_repr_shape=2, N_SAMPLES = 10_000, axes_lims=None, beta = 0.1, project = False, uid= None):
    '''
    Assumes axes are 0, 1, 2. Only supports sawyer bin and box.
    '''
    # TODO currently only supports obj = agent (tying agent and obj positions together)
    
    if axes_lims is None:
        axes_lims = {
            0: (0, 11),
            1: (0, 11),
        }
    
    misc_params = '{}_None'.format(alpha)
    
    obs_dim = 2

    # load weights + init environment
    trained_learner_state, env, networks = load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals = True, ckpt_num = ckpt_num, uid=uid)

    timestep = env.reset()
    initial_state = np.zeros((state_repr_shape))

    # init dummy obs + action
    obs = timestep.observation
    obs_dim = obs.shape[0] // 2

    dist0 = networks.policy_network.apply(trained_learner_state.policy_params, obs)
    a0 = np.array(dist0.mode())      
    obs0 = obs[None, :]         # shape (1, 2*obs_dim)
    a0   = a0[None, :]          # shape (1, action_dim)                      # a₀
    _, phi_s0_a0, _ = networks.q_network.apply(
            trained_learner_state.q_params, obs0, a0)
    phi_s0_a0 = phi_s0_a0.squeeze(0)      # shape (repr_dim,)
    
    
    # dist = networks.policy_network.apply(
    #   trained_learner_state.policy_params,
    #   obs
    # )
    # action = np.array(dist.mode())
    # action_batch = np.broadcast_to(action, (N_SAMPLES, action.shape[0]))
    
    # uniformly sample states within axes_lims
    random_goals = np.random.rand(N_SAMPLES, 2*obs_dim)
    random_goals[:, :obs_dim] = 0
    # transforms U([0,1)) -> U(axis_min, axis_max) for each axis (0, 1, 2)
    for axis in (0, 1):
        axis_min, axis_max = axes_lims[axis]
        random_goals[:, obs_dim+axis] *= axis_max - axis_min
        random_goals[:, obs_dim+axis] += axis_min

    goal_locations = random_goals[:, obs_dim:].copy()
    # obs_batch = np.broadcast_to(obs, (N_SAMPLES, 2*obs_dim)).copy()
    # obs_batch[:, obs_dim:] = 0
    # obs_batch = obs_batch + random_goals
    # Set both state and goal parts to the same random goal

    ## both parts are random states we just sampld, we wanna calculate both their psi similarity with the goal and their phi psi similarity
    obs_batch = np.zeros((N_SAMPLES, 2 * obs_dim), dtype=np.float32)
    obs_batch[:, :obs_dim] = goal_locations 
    obs_batch[:, obs_dim:] = goal_locations 

    # INSERT NEW CODE HERE
    dist = networks.policy_network.apply(
        trained_learner_state.policy_params,
        obs_batch
    )
    action_batch = np.array(dist.mode())


    # compute psi norms and critic_sf
    use_phi_critic = False
    use_l2_critic = True # doesn't actually matter if we don't use first return value
    q_action, sa_repr, g_repr_sa = networks.q_network.apply(trained_learner_state.q_params, obs_batch, action_batch,
                                                        #  use_l2_critic=use_l2_critic, 
                                                        #  use_phi_critic=use_phi_critic, 
                                                        #  use_sa_reg=False, 
                                                        #  alpha=alpha
                                                        )


    critic_sf = np.diag(q_action)
    psi_norms = np.diag(np.einsum('ik,jk->ij', g_repr_sa, g_repr_sa))

    # uniformly sample states within axes_lims
    random_states = np.random.rand(N_SAMPLES, 2*obs_dim)
    random_states[:, obs_dim:] = 0
    # transforms U([0,1)) -> U(axis_min, axis_max) for each axis (0, 1, 2)
    for axis in (0, 1):
        axis_min, axis_max = axes_lims[axis]
        random_states[:, axis] *= axis_max - axis_min
        random_states[:, axis] += axis_min


    fixed_goal_obs_batch = np.broadcast_to(obs, (N_SAMPLES, 2*obs_dim)).copy()
    fixed_goal_obs_batch[:, :obs_dim] = 0
    fixed_goal_obs_batch = fixed_goal_obs_batch + random_states
    dist = networks.policy_network.apply(trained_learner_state.policy_params, fixed_goal_obs_batch)
    
    action_batch = np.array(dist.mode())

    q_action, _, g_repr_g = networks.q_network.apply(trained_learner_state.q_params, fixed_goal_obs_batch, action_batch,
                                                        #  use_l2_critic=use_l2_critic, 
                                                        #  use_phi_critic=use_phi_critic, 
                                                        #  use_sa_reg=False, 
                                                        #  alpha=alpha
                                                        )
    critic_g = np.diag(q_action)
    waypoint_repr = (1 - beta) * sa_repr + beta * g_repr_g
    
    #psi_similarity = np.diag(np.einsum('ik,jk->ij', 
    #                                     g_repr_sa / np.linalg.norm(g_repr_sa, axis=1, keepdims=True), 
    #                                     g_repr_g / np.linalg.norm(g_repr_g, axis=1, keepdims=True)))
    # Project g_repr_sa and g_repr_g onto the subspace
    



    # Compute cosine similarity on the projected representations
    psi_similarity = np.diag(np.einsum(
        'ik,jk->ij',
        g_repr_sa / np.linalg.norm(g_repr_sa, axis=1, keepdims=True),
        g_repr_g / np.linalg.norm(g_repr_g, axis=1, keepdims=True)
    ))
    psi_inner_product = np.diag(np.einsum(
        'ik,jk->ij',
        g_repr_sa,
        g_repr_g
    ))

    if (g_repr_sa < 0).any() or (g_repr_g < 0).any():
        print("❌ Error: There are negative values in g_repr_sa or g_repr_g")


    phi_psi_similarity = np.diag(np.einsum(
        'ik,jk->ij',
        sa_repr / np.linalg.norm(sa_repr, axis=1, keepdims=True),
        g_repr_g / np.linalg.norm(g_repr_g, axis=1, keepdims=True)
    ))
    phi_psi_inner_product = np.diag(np.einsum(
        'ik,jk->ij',
        sa_repr ,
        g_repr_g 
    ))

    local_phi_psi_similarity = np.diag(np.einsum(
        'ik,jk->ij',
        sa_repr / np.linalg.norm(sa_repr, axis=1, keepdims=True),
        g_repr_sa / np.linalg.norm(g_repr_sa, axis=1, keepdims=True)
    ))
    waypoint_similarity = np.diag(np.einsum('ik,jk->ij', g_repr_sa / np.linalg.norm(g_repr_sa, axis=1, keepdims=True), waypoint_repr / np.linalg.norm(waypoint_repr, axis=1, keepdims=True)))

    # 2. φ(s₀,a₀)·ψ(s)   (vector length N_SAMPLES)
    phi_s0_dot_psi_s = np.einsum('i,ji->j', phi_s0_a0, g_repr_sa)

    # 3. φ(s,a)·ψ(g)     (vector length N_SAMPLES)
    phi_s_dot_psi_g = np.einsum('ij,ij->i', sa_repr, g_repr_g)

    # 4. Combined metric
    posterior = phi_s0_dot_psi_s + phi_s_dot_psi_g

    ### getting the optimal actions at random locations but using the fixed goal 
    # --- Use fixed goal positions from before
    fixed_goal_positions = fixed_goal_obs_batch[:, obs_dim:]  # shape (N_SAMPLES, obs_dim)
    ## eventhough it is called goal locations because we are assessing its psi but it is actually states
    cross_obs_batch = np.concatenate([goal_locations, fixed_goal_positions], axis=1)

    # --- Query policy at these states
    dist = networks.policy_network.apply(trained_learner_state.policy_params, cross_obs_batch)
    actions_at_random_states_fixed_goal = np.array(dist.mode())
    return goal_locations, psi_norms, critic_sf, critic_g, psi_similarity, waypoint_similarity, actions_at_random_states_fixed_goal, env, trained_learner_state, networks, phi_psi_similarity, local_phi_psi_similarity, posterior, phi_s0_dot_psi_s, phi_s_dot_psi_g, psi_inner_product, phi_psi_inner_product





def get_cell_projected_psi_norms(env_name, log_dir,  seed, ckpt_num=None, alg = 'contrastive_cpc', alpha = '0.1',
                  state_repr_shape=2, N_SAMPLES = 10_000, axes_lims=None, beta = 0.1, project = False, point_map=None, cell_size=1, uid=None):
    '''
    Assumes axes are 0, 1, 2. Only supports sawyer bin and box.
    '''
    # TODO currently only supports obj = agent (tying agent and obj positions together)
    
    if axes_lims is None:
        axes_lims = {
            0: (0, 11),
            1: (0, 11),
        }
    
    misc_params = '{}_None'.format(alpha)
    
    obs_dim = 2

    # load weights + init environment
    trained_learner_state, env, networks = load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals = True, ckpt_num = ckpt_num, uid=uid)

    timestep = env.reset()
    initial_state = np.zeros((state_repr_shape))

    # init dummy obs + action
    obs = timestep.observation
    
    obs_dim = obs.shape[0] // 2
    dist = networks.policy_network.apply(
      trained_learner_state.policy_params,
      obs
    )
    action = np.array(dist.mode())
    action_batch = np.broadcast_to(action, (N_SAMPLES, action.shape[0]))

    fit_scores, cells , proj_matrixes = subspace_transferability(env, trained_learner_state, networks, point_map=point_map, env_name=env_name, seed =seed, ckpt_num=ckpt_num, cell_size=cell_size)
    

    def goal_to_cell(goal, cell_size=1):
        """Given a goal (x,y), return its corresponding cell as (row,col)."""
        row, col = (goal // cell_size).astype(int)
        return (row, col)

    # uniformly sample states within axes_lims
    random_goals = np.random.rand(N_SAMPLES, 2*obs_dim)
    random_goals[:, :obs_dim] = 0
    # transforms U([0,1)) -> U(axis_min, axis_max) for each axis (0, 1, 2)
    for axis in (0, 1):
        axis_min, axis_max = axes_lims[axis]
        random_goals[:, obs_dim+axis] *= axis_max - axis_min
        random_goals[:, obs_dim+axis] += axis_min

    goal_locations = random_goals[:, obs_dim:].copy()
    obs_batch = np.broadcast_to(obs, (N_SAMPLES, 2*obs_dim)).copy()
    obs_batch[:, obs_dim:] = 0
    obs_batch = obs_batch + random_goals


    
    q_action, sa_repr, g_repr_sa = networks.q_network.apply(trained_learner_state.q_params, obs_batch, action_batch,
                                                        )


    critic_sf = np.diag(q_action)
    psi_norms = np.diag(np.einsum('ik,jk->ij', g_repr_sa, g_repr_sa))

    # uniformly sample states within axes_lims
    random_states = np.random.rand(N_SAMPLES, 2*obs_dim)
    random_states[:, obs_dim:] = 0
    # transforms U([0,1)) -> U(axis_min, axis_max) for each axis (0, 1, 2)
    for axis in (0, 1):
        axis_min, axis_max = axes_lims[axis]
        random_states[:, axis] *= axis_max - axis_min
        random_states[:, axis] += axis_min


    fixed_goal_obs_batch = np.broadcast_to(obs, (N_SAMPLES, 2*obs_dim)).copy()
    fixed_goal_obs_batch[:, :obs_dim] = 0
    fixed_goal_obs_batch = fixed_goal_obs_batch + random_states
    dist = networks.policy_network.apply(trained_learner_state.policy_params, fixed_goal_obs_batch)
    
    action_batch = np.array(dist.mode())

    q_action, _, g_repr_g = networks.q_network.apply(trained_learner_state.q_params, fixed_goal_obs_batch, action_batch,
                                                        )
    critic_g = np.diag(q_action)
    waypoint_repr = (1 - beta) * sa_repr + beta * g_repr_g
    

    psi_similarity = np.zeros(N_SAMPLES)
    waypoint_similarity = np.zeros(N_SAMPLES)

    for idx in range(N_SAMPLES):
        goal_loc = goal_locations[idx, :2]
        # print(f"Processing goal location {goal_loc} for sample {idx+1}/{N_SAMPLES}")
        cell = goal_to_cell(goal_loc)
        # print(f"Processing cell {cell} for sample {idx+1}/{N_SAMPLES}")
        if cell not in proj_matrixes:
            psi_similarity[idx] = 0.0
            waypoint_similarity[idx] = 0.0
            continue  # Skip if the cell is not in the projection matrixes
        ## it means the cell is a wall

        proj_mat = proj_matrixes[cell]

        # Project psi vectors onto subspace of current cell
        g_sa_proj = g_repr_sa[idx] @ proj_mat
        g_g_proj = g_repr_g[idx] @ proj_mat
        waypoint_proj = waypoint_repr[idx] @ proj_mat

        # Compute similarity on projected vectors
        psi_similarity[idx] = np.dot(
            g_sa_proj / (np.linalg.norm(g_sa_proj) + 1e-8),
            g_g_proj / (np.linalg.norm(g_g_proj) + 1e-8)
        )

        waypoint_similarity[idx] = np.dot(
            g_sa_proj / (np.linalg.norm(g_sa_proj) + 1e-8),
            waypoint_proj / (np.linalg.norm(waypoint_proj) + 1e-8)
        )

    ### getting the optimal actions at random locations but using the fixed goal 
    # --- Use fixed goal positions from before
    fixed_goal_positions = fixed_goal_obs_batch[:, obs_dim:]  # shape (N_SAMPLES, obs_dim)
    ## eventhough it is called goal locations because we are assessing its psi but it is actually states
    cross_obs_batch = np.concatenate([goal_locations, fixed_goal_positions], axis=1)

    # --- Query policy at these states
    dist = networks.policy_network.apply(trained_learner_state.policy_params, cross_obs_batch)
    actions_at_random_states_fixed_goal = np.array(dist.mode())
    return goal_locations, psi_norms, critic_sf, critic_g, psi_similarity, waypoint_similarity, actions_at_random_states_fixed_goal, env, trained_learner_state, networks






def get_sa_repr_subspace_basis(env, trained_learner_state, networks, goal_locations):
    obs_dim   = goal_locations.shape[1]
    N_SAMPLES = goal_locations.shape[0]

    # one random action per sample, uniform in [‑1,1]^dim
    action_batch = np.random.uniform(-1.0, 1.0,
                                    size=(N_SAMPLES, obs_dim)
                                    ).astype(np.float32)

    # build (s,a) batch: agent at goal_locations, goal at the origin
    obs_batch = np.zeros((N_SAMPLES, 2 * obs_dim), dtype=np.float32)
    obs_batch[:, :obs_dim] = goal_locations          # first half = agent position

    # compute ϕ(s,a)
    _, sa_repr_matrix, _ = networks.q_network.apply(
        trained_learner_state.q_params,
        obs_batch,
        action_batch,
    )   # sa_repr_mat

    # Compute basis via SVD
    # Compute rank directly
    rank = np.linalg.matrix_rank(sa_repr_matrix)
    #print(f"True rank from data span = {rank}")

    # SVD of the stacked matrix  (M × d)
    # full‑matrices=False is fine
    U, S, Vt = np.linalg.svd(sa_repr_matrix, full_matrices=False)

    # --- use numpy’s default tolerance ---------------------------------
    tol = S.max() * max(sa_repr_matrix.shape) * np.finfo(S.dtype).eps
    rank = np.sum(S > tol)                         # ← now matches matrix_rank
    # --------------------------------------------------------------------

    # right‑singular vectors (Vt) give the basis in 64‑D representation space
    basis = Vt[:rank, :]                           # shape (rank, 64)

    # projection matrix (64 × 64)
    projection_matrix = basis.T @ basis
    print("Consistent rank:", rank, "projection‑matrix shape:", projection_matrix.shape)

    return basis, projection_matrix  # Each row is a basis vector (can also return Vt or U if needed)

# ────────────────────────────────────────────────────────────────────────────────
# REPLACE the previous `subspace_transferability` (and any PCA helper it used)
# with the version below ─ nothing else in your file needs to change.
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
def subspace_transferability(env,
                             trained_learner_state,
                             networks,
                             point_map=None,            # ← NEW
                             start_cell=(5, 5),         # grid (row, col)
                             goal_cell=(10, 10),
                             path_cells=None,
                             cell_size=1,
                             verbose=False,
                             env_name="point_Spiral11x11",
                             seed=5,
                             ckpt_num=None, 
                             use_all_free_cells=True):
    """
    Follow the true walkable corridor in point_Spiral11x11 (or any binary map)
    and measure how well each square’s ϕ(s,a) vectors fit the previous square’s
    sub‑space (basis computed via SVD, identical to your earlier code).

    • If `path_cells` is given, that exact sequence is used.
    • Otherwise we expect a `point_map` (0 = free, 1 = wall) and run BFS to
      find the shortest 4‑connected path from `start_cell` to `goal_cell`.
    """
    import numpy as np
    from collections import deque

    # -------------------------------------------------------------------------
    # 1. Figure out which cells to visit
    # -------------------------------------------------------------------------
    if path_cells is None:
        if point_map is None:
            raise ValueError("Either path_cells or point_map must be provided.")

        R, C = point_map.shape
        # assume `cell_size` is an int defined in the outer scope
        if use_all_free_cells:
            path_cells = [
                (r, c)
                for r in range(0, R, cell_size)      # ← step by cell_size
                for c in range(0, C, cell_size)      # ← step by cell_size
                # if point_map[r, c] == 0            # keep commented or restore
            ]


    # -------------------------------------------------------------------------
    # 2. Helpers to sample and compute ϕ(s,a)
    # -------------------------------------------------------------------------
    # def _sa_repr(states, actions):
    #     _, phi, _ = networks.q_network.apply(
    #         trained_learner_state.q_params, states, actions
    #     )
    #     return np.asarray(phi)

    def _sa_repr(states, actions, *, batch_size: int = 1024):
        """
        Return ϕ(s,a) for a (possibly large) batch of (states, actions),
        processing in smaller chunks to avoid OOM.

        Args:
            states:  np.ndarray / jnp.ndarray with shape (N, …)
            actions: np.ndarray / jnp.ndarray with shape (N, …)
            batch_size: maximum #pairs to feed into the network at once
        """
        num = states.shape[0]
        chunks = []
        for start in range(0, num, batch_size):
            stop = start + batch_size
            # slice the current mini-batch
            s_batch = states[start:stop]
            a_batch = actions[start:stop]
            _, phi_batch, _ = networks.q_network.apply(
                trained_learner_state.q_params, s_batch, a_batch
            )
            chunks.append(phi_batch)

        # jnp.concatenate keeps everything on device; np.asarray() pulls to host
        phi_full = jnp.concatenate(chunks, axis=0)
        return np.asarray(phi_full)          # keep original return type


    # def _sample_in_cell(cell_xy):
    #     low  = np.asarray(cell_xy, dtype=np.float32)
    #     high = low + cell_size
    #     r, c = cell_xy
    #     # if point_map is not None and point_map[r, c] == 1:
    #     #     raise ValueError(f"Cell {cell_xy} is a wall; cannot sample here.")


    #     xs = np.random.uniform(low=low, high=high, size=(n_samples, 2))
    #     obs_tpl = env.reset().observation
    #     obs     = np.tile(obs_tpl, (n_samples, 1)).astype(np.float32)
    #     obs[:, :2] = xs                                 # assumes obs[:2] = (row,col)

    #     act_dim = env.action_spec().shape[0]
    #     acts = np.random.uniform(-1., 1., size=(n_samples, act_dim)).astype(np.float32)



    #     return obs, acts


    def _sample_in_cell(cell_xy, *, grid_res=0.1):
        """Return exhaustive (state, action) pairs for one cell.

        - States:  (row,col) grid inside the cell with spacing = grid_res.
        - Actions: all Cartesian products of (-1,0,1) per action dim
                (currently drops the all-zero vector).

        Returns
        -------
        obs  : (n_states * n_actions, obs_dim) float32
        acts : (n_states * n_actions, act_dim) float32
        """
        low   = np.asarray(cell_xy, dtype=np.float32)
        high  = low + cell_size              # cell_size defined elsewhere
        r, c  = cell_xy

        # ----- 1. deterministic state grid inside the cell -----------------
        rows = np.arange(low[0], high[0], grid_res, dtype=np.float32)
        cols = np.arange(low[1], high[1], grid_res, dtype=np.float32)
        xs   = np.stack(np.meshgrid(rows, cols, indexing="ij"), axis=-1).reshape(-1, 2)
        n_states = xs.shape[0]

        # ----- 2. exhaustive discrete actions ------------------------------
        act_dim = env.action_spec().shape[0]
        action_list = list(product([-1., 0., 1.], repeat=act_dim))
        # optionally drop the zero-action
        action_list = [a for a in action_list if any(v != 0. for v in a)]
        acts_grid   = np.asarray(action_list, dtype=np.float32)            # (n_act, act_dim)
        n_actions   = acts_grid.shape[0]

        # ----- 3. tile / repeat to pair every state with every action ------
        obs_tpl = env.reset().observation
        obs = np.tile(obs_tpl, (n_states * n_actions, 1)).astype(np.float32)
        obs[:, :2] = np.repeat(xs, n_actions, axis=0)

        acts = np.tile(acts_grid, (n_states, 1))

        return obs, acts

    def _svd_basis(vecs):
        true_rank = np.linalg.matrix_rank(vecs)
        print(f"True rank from data span = {true_rank}")


        U, S, Vt = np.linalg.svd(vecs, full_matrices=False)
        
        tol  = S.max() * max(vecs.shape) * np.finfo(S.dtype).eps
        rank = int(np.sum(S > tol))
        basis = Vt[:rank, :]                           # (rank, d)
        proj  = basis.T @ basis                         # (d, d)
        return proj, true_rank, basis.T
    
    def diagnose_rank(X, pca_cut=0.99, eps=None):
        U,S,Vt = np.linalg.svd(X, full_matrices=False)
        
        print("all σ:", S[:10], "…")
        # (i) algebraic rank
        print("exact rank:", np.linalg.matrix_rank(X, tol=0.0))
        # (ii) relative-tol rank
        if eps is None:
            eps = np.finfo(S.dtype).eps
        tol = eps * S[0] * max(X.shape)
        print(f"rank tol={tol:.1e} :", (S>tol).sum())
        # (iii) PCA variance rank
        r = np.searchsorted(np.cumsum(S**2)/S.sum()**2, pca_cut) + 1
        print(f"components for {pca_cut*100:.1f}% variance:", r)
        return S


    def _avg_residual(vecs, proj, relative=True, eps=1e-12):
        """
        Parameters
        ----------
        vecs : ndarray  (N, d)
        proj : ndarray  (d, d)   projection matrix  B Bᵀ
        relative : bool
            • True  → return ⟨‖ϕ − Pϕ‖ / ‖ϕ‖⟩  (unit‑free, in [0,1]).
            • False → return ⟨‖ϕ − Pϕ‖⟩        (absolute L2 length).
        """
        diff = vecs - vecs @ proj                  # (N, d)
        num  = np.linalg.norm(diff, axis=1)        # residual norms
        if relative:
            den = np.linalg.norm(vecs, axis=1) + eps
            return float(np.mean(num / den))
        else:
            return float(np.mean(num))


    # -------------------------------------------------------------------------
    # 3. Walk along the cells, build sub‑spaces, measure residuals
    # -------------------------------------------------------------------------
    prev_prev_proj = None        # two steps back
    prev_proj      = None        # one step back
    fit_scores     = []          # [(cell, r1, r2), …]
    ranks_along_path = []
    proj_by_cell = {}          # (row,col)  ->  projection matrix  (d×d)
    basis_by_cell = {}          # (row,col)  ->  basis vectors  (rank,d)


    for cell in path_cells:
        #print(f"\nSampling cell {cell}", end=' ')
        obs, acts = _sample_in_cell(cell)
        phi       = _sa_repr(obs, acts)
        #diagnose_rank(phi, pca_cut=0.99)  # print the rank diagnostics
        #diagnose_rank(phi, pca_cut=0.99, eps=None)  # print the rank diagnostics
        if prev_proj is not None:
            r1 = _avg_residual(phi, prev_proj)              # ← parent
            r2 = (_avg_residual(phi, prev_prev_proj)
                  if prev_prev_proj is not None else None)  # ← grand‑parent
            fit_scores.append((cell, r1, r2))

            if verbose:
                msg = f"res→prev={r1:.4f}"
                if r2 is not None:
                    msg += f",  res→prevprev={r2:.4f}"
                #print(msg)

        # slide the window: current basis becomes 'prev', old 'prev' → 'prev_prev'
        prev_prev_proj = prev_proj
        prev_proj, rank , basis     = _svd_basis(phi)
        
        proj_by_cell[cell] = prev_proj     
        basis_by_cell[cell] = basis
        ranks_along_path.append((cell, rank))

    if verbose:
        print("\n=== Sub‑space transferability (start→goal) ===")
        hdr = "cell           resid→prev    resid→prevprev"
        print(hdr)
        for cell, r1, r2 in fit_scores:
            print(f"{cell!s:<14}{r1:12.4f}{'' if r2 is None else f'{r2:16.4f}'}")
    
    
    ############# Now cluster the similar cells
    cells      = list(proj_by_cell.keys())
    proj_list  = [proj_by_cell[c] for c in cells]
    bases     = [basis_by_cell[c] for c in cells]
    K          = len(cells)

    # -------------------------------------------------------------------------
    # 4. Build 2D heatmaps for the full point_map
    # -------------------------------------------------------------------------
    R, C = point_map.shape
    rank_grid = np.full((R, C), np.nan, dtype=np.float32)
    resid_prev_grid = np.full((R, C), np.nan, dtype=np.float32)
    resid_prevprev_grid = np.full((R, C), np.nan, dtype=np.float32)

    for (r, c), rank in ranks_along_path:
        rank_grid[r, c] = rank

    for (r, c), r1, r2 in fit_scores:
        resid_prev_grid[r, c] = r1
        if r2 is not None:
            resid_prevprev_grid[r, c] = r2

    # Mask walls for a gray background
    mask = (point_map == 1)
    gray_background = np.ones_like(rank_grid) * np.nan
    gray_background[mask] = 1.0

    # -------------------------------------------------------------------------
    # 5. Plot heatmaps
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    vmin_rank, vmax_rank = np.nanmin(rank_grid), np.nanmax(rank_grid)
    vmin_r1, vmax_r1 = np.nanmin(resid_prev_grid), np.nanmax(resid_prev_grid)
    vmin_r2, vmax_r2 = np.nanmin(resid_prevprev_grid), np.nanmax(resid_prevprev_grid)

    def show(ax, data, title, cmap, vmin, vmax):
        ax.imshow(gray_background, cmap='gray', vmin=0, vmax=1)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        return im

    im0 = show(axes[0], rank_grid, "Local rank", "viridis", vmin_rank, vmax_rank)
    im1 = show(axes[1], resid_prev_grid, "Residual vs prev", "coolwarm", vmin_r1, vmax_r1)
    im2 = show(axes[2], resid_prevprev_grid, "Residual vs prev‑prev", "coolwarm", vmin_r2, vmax_r2)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # -------------------------------------------------------------------------
    # 6. Save the plot
    # -------------------------------------------------------------------------
    plot_dir = os.path.join("experiments/plots", f"{env_name}_{seed}")
    os.makedirs(plot_dir, exist_ok=True)
    fname = f"rank_{ckpt_num}.png" if ckpt_num is not None else "rank.png"
    path = os.path.join(plot_dir, fname)
    ## uncomment it if you wanna save the rank plot
    #plt.savefig(path, dpi=200)
    plt.close(fig)

    if verbose:
        print(f"✓ Saved heatmap plot to {path}")

    # -------------------------------------------------------------------------
    # 4. Report average residuals
    # -------------------------------------------------------------------------
    prev_residuals = [r1 for _, r1, _ in fit_scores if r1 is not None]
    prevprev_residuals = [r2 for _, _, r2 in fit_scores if r2 is not None]

    avg_prev = np.mean(prev_residuals) if prev_residuals else float("nan")
    avg_prevprev = np.mean(prevprev_residuals) if prevprev_residuals else float("nan")

    if verbose:
        print("\n=== Average Residuals Across Path ===")
        print(f"Avg residual → prev      = {avg_prev:.6f}")
        print(f"Avg residual → prev-prev = {avg_prevprev:.6f}")




    return fit_scores, cells , proj_by_cell


