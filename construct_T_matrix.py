

import os
import numpy as np
import jax
import optax
import tensorflow as tf
import functools
import pickle
import re
import jax.numpy as jnp
from copy import deepcopy
from tqdm import tqdm
from acme import specs
from acme.tf.savers import SaveableAdapter
from contrastive.config import ContrastiveConfig
from contrastive import utils as contrastive_utils
from contrastive import make_networks
from contrastive.utils import make_environment
from contrastive import ContrastiveLearner
from acme.jax import utils
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


seed_num = 52
# disable tensorflow_probability warning: The use of `check_types` is deprecated and does not have any effect.
import logging
logger = logging.getLogger("root")
logger.addFilter(lambda record: "check_types" not in record.getMessage())

# hyperparams (mostly unimportant)
config = ContrastiveConfig()
config.use_goal_action = True
config.use_quasimetric_logit = True
config.use_goal_potential = True
config.twin_q = True

# For sawyer environments, use these fixed start/end positions.
def get_fixed_start_end(env_name):
    if env_name in ['sawyer_box', 'sawyer_peg']:
        return np.array([0.0, 0.75, 0.133])
    else:
        # assumed to be 'sawyer_bin'
        return np.array([0.12, 0.7, 0.02])

# ---------------------------
# Updated load_checkpoint function.
# ---------------------------
def load_checkpoint(alpha, misc_params, env_name, base_log_dir, seed, fix_goals=False, ckpt_num=None, NUM_EPISODES=10, alg='contrastive_cpc'):
    state_entropy_coefficient = alpha

    # Here we override the ckpt_dir to be the folder provided.
    ckpt_dir = f'/home/mahsa/sgcrl/logs/contrastive_cpc_sawyer_bin_{seed_num}/checkpoints/learner'
    
    fixed_start_end = get_fixed_start_end(env_name) if fix_goals else None

    # Create the environment using your provided factory.
    # seed should be diverse to create diverse episodes otherwise the episodes
    #  always remain the same across different runs

    env_factory = lambda seed: make_environment(env_name, config.start_index, 
                                                config.end_index, seed=np.random.randint(1e6), 
                                                fixed_start_end=fixed_start_end)[0]
    dummy_seed = 1
    environment_spec = specs.make_environment_spec(env_factory(np.random.randint(1e6)))

    # Get observation dimension (for non-image observations).
    obs_dim = make_environment(env_name, config.start_index, config.end_index, seed=np.random.randint(1e6),
                               fixed_start_end=fixed_start_end)[1]

    network_factory = functools.partial(
        make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
        repr_norm=config.repr_norm, twin_q=False,
        use_image_obs=config.use_image_obs,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    #random_key = jax.random.PRNGKey(np.random.choice(int(1e6)))
    networks = network_factory(environment_spec)
    # policy_optimizer = optax.adam(learning_rate=config.actor_learning_rate)
    # q_optimizer = optax.adam(0.001)  # learning_rate=config.critic_learning_rate
    policy_optimizer = optax.adam(
        learning_rate=config.actor_learning_rate, eps=1e-7)
    q_optimizer = optax.adam(learning_rate=config.learning_rate, eps=1e-7)

    key = jax.random.PRNGKey(seed_num)
    learner_key, key = jax.random.split(key)
    actor_key, key = jax.random.split(key)

    trained_learner = ContrastiveLearner(
        networks=networks,
        rng=learner_key,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        iterator=None,
        counter=None,
        logger=None,
        obs_to_goal=functools.partial(contrastive_utils.obs_to_goal_2d,
                                      start_index=config.start_index,
                                      end_index=config.end_index),
        config=config)

    #env = env_factory(np.random.randint(1e6))
    #episode_returns = np.zeros([NUM_EPISODES, ])

    environment_key, actor_key = jax.random.split(actor_key)
    # Create environment and policy core.

    # Environments normally require uint32 as a seed.
    #env = env_factory(
    #    utils.sample_uint32(environment_key))
    env = env_factory(np.random.randint(1e6))

    # Create the checkpoint object.
    ckpt = tf.train.Checkpoint(learner=SaveableAdapter(trained_learner))
    ckpt_mgr = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1)

    if ckpt_num is not None:
        ckpt_path = os.path.join(ckpt_dir, 'ckpt-' + str(ckpt_num))
        ckpt.restore(ckpt_path).assert_consumed()
        print("Restored checkpoint:", ckpt_path)
    else:
        latest = ckpt_mgr.latest_checkpoint
        ckpt.restore(latest).assert_consumed()
        print("Restored latest checkpoint:", latest)

    trained_learner_state = trained_learner._state

    print("Model loaded from: {}".format(ckpt_dir))
    return trained_learner_state, env, networks

# ---------------------------
# Helper to list all checkpoint numbers in a folder.
# ---------------------------


def list_all_checkpoints(ckpt_dir):
    """
    List all unique checkpoint numbers in the folder by parsing filenames like:
    ckpt-1.index, ckpt-2.data-00000-of-00002, etc.
    """
    ckpt_files = os.listdir(ckpt_dir)
    pattern = re.compile(r"ckpt-(\d+)\.")
    
    ckpt_nums = set()
    for fname in ckpt_files:
        match = pattern.match(fname)
        if match:
            ckpt_nums.add(int(match.group(1)))
    
    return sorted(list(ckpt_nums))


import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
import os

def collect_unique_transitions_single_ckpt(ckpt_num,
                                           num_transitions=25000,
                                           alpha='0.1',
                                           misc_params='0.1_None',
                                           env_name='sawyer_bin',
                                           seed=seed_num,
                                           save_path='./T_25k_unique.npy',
                                           policy_type='policy'):
    """
    Collects unique transitions using either the learned policy or a random one.
    
    Args:
        ckpt_num (int): Checkpoint number to load.
        num_transitions (int): Number of unique transitions desired.
        policy_type (str): 'policy' to use the trained agent, 'random' for uniform random actions.
        Other args are for environment and checkpoint loading.
    """
    # Load environment and policy
    trained_learner_state, env, networks = load_checkpoint(
        alpha, misc_params, env_name, base_log_dir='',
        seed=np.random.randint(1e6), fix_goals=True, ckpt_num=ckpt_num)

    # JAX action function
    @jax.jit
    def select_action(params, obs):
        dist = networks.policy_network.apply(params, obs)
        return dist.mode()

    unique_deltas = set()
    delta_list = []

    pbar = tqdm(total=num_transitions, desc=f"Collecting ({policy_type}) from ckpt-{ckpt_num}")
    while len(delta_list) < num_transitions:
        env.seed(np.random.randint(1e6))
        timestep = env.reset()
        prev_obs = timestep.observation[:7]

        while not timestep.last() and len(delta_list) < num_transitions:
            if policy_type == 'random':
                action = env.action_space.sample()
            else:
                obs_batch = jnp.expand_dims(jnp.array(timestep.observation), axis=0)
                action = np.array(select_action(trained_learner_state.policy_params, obs_batch)).squeeze()

            timestep = env.step(action)
            curr_obs = timestep.observation[:7]

            delta = curr_obs - prev_obs
            delta_tuple = tuple(np.round(delta, decimals=5))  # deduplicate

            if delta_tuple not in unique_deltas:
                unique_deltas.add(delta_tuple)
                delta_list.append(delta)
                pbar.update(1)

            prev_obs = curr_obs

    pbar.close()
    T = np.stack(delta_list)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, T)
    print(f"Saved {T.shape[0]} unique deltas to {save_path}")




def compute_eigenpurposes_from_T(T_path='./logs/contrastive_cpc_sawyer_bin_52/T_25k_unique.npy',
                                  save_path='eigenpurposes.npy',
                                  eigenvalues_path='eigenvalues.npy',
                                  top_k=None):
    """
    Loads a transition matrix T, performs SVD, and returns & prints top eigenpurposes (V columns)
    and their corresponding singular values (used as eigenvalues).
    
    Args:
        T_path (str): Path to the saved .npy file containing the transition matrix.
        save_path (str): Path to save the eigenpurposes (.npy format).
        eigenvalues_path (str): Path to save the eigenvalues (.npy format).
        top_k (int or None): If provided, only return and print the top_k eigenpurposes.
        
    Returns:
        eigenpurposes (np.ndarray): Matrix of shape (7, top_k) or (7, 7).
        eigenvalues (np.ndarray): Vector of corresponding singular values.
    """
    # Load the transition matrix
    T = np.load(T_path)
    print(f"Loaded T with shape {T.shape}")

    # Perform SVD
    U, S, Vh = np.linalg.svd(T, full_matrices=False)
    eigenpurposes = Vh.T  # shape (7, 7)
    eigenvalues = S       # shape (7,)

    # Optionally limit to top_k
    if top_k is not None:
        eigenpurposes = eigenpurposes[:, :top_k]
        eigenvalues = eigenvalues[:top_k]

    # Print each eigenvalue and its corresponding eigenvector
    print("\n--- Eigenpurposes and Corresponding Eigenvalues ---")
    for i in range(eigenpurposes.shape[1]):
        print(f"\nEigenvalue {i+1}: {eigenvalues[i]:.6f}")
        print(f"Eigenvector {i+1}: {eigenpurposes[:, i]}")

    # Save to files
    np.save(save_path, eigenpurposes)
    np.save(eigenvalues_path, eigenvalues)
    print(f"\nSaved eigenpurposes to: {save_path}")
    print(f"Saved eigenvalues to: {eigenvalues_path}")

    return eigenpurposes, eigenvalues





def plot_intrinsic_metrics_multiple_episodes(ckpt_num,
                                             num_transitions,
                                             eigenpurposes_path='eigenpurposes.npy',
                                             eigenvalues_path='eigenvalues.npy',
                                             env_name='sawyer_bin',
                                             seed=seed_num,
                                             alpha='0.1',
                                             misc_params='0.1_None',
                                             plot_dir=f'plots/{seed_num}',
                                             policy='policy',
                                             num_episodes=30):
    """
    Runs multiple episodes and saves 2 plots per episode:
    - Intrinsic rewards per option + Q + object distance
    - Cosine similarities per option + Q + object distance
    """

    eigenpurposes = np.load(eigenpurposes_path)
    eigenvalues = np.load(eigenvalues_path)

    os.makedirs(plot_dir, exist_ok=True)
    trained_learner_state, env, networks = load_checkpoint(
            alpha=alpha, misc_params=misc_params, env_name=env_name,
            base_log_dir='', seed=np.random.randint(1e6), fix_goals=True, ckpt_num=ckpt_num)

    @jax.jit
    def select_action(params, obs):
        dist = networks.policy_network.apply(params, obs)
        return dist.mode()

    @jax.jit
    def get_q_value(q_params, obs, action):
        obs = jnp.expand_dims(jnp.array(obs), axis=0)
        action = jnp.expand_dims(jnp.array(action), axis=0)
        q_val, _, _ = networks.q_network.apply(q_params, obs, action)
        return q_val.squeeze()

    for episode in range(num_episodes):
        

        timestep = env.reset()

        rewards_over_time = [[] for _ in range(eigenpurposes.shape[1])]
        cos_sims_over_time = [[] for _ in range(eigenpurposes.shape[1])]
        q_values_over_time = []
        env_rewards = []
        obj_goal_dists = []

        obs_t = timestep.observation[:7]
        goal_obj_pos = timestep.observation[11:14]

        while not timestep.last():
            obs_batch = jnp.expand_dims(jnp.array(timestep.observation), axis=0)

            if policy == 'random':
                action = env.action_space.sample()
            else:
                action = np.array(select_action(trained_learner_state.policy_params, obs_batch)).squeeze()

            q_val = float(get_q_value(trained_learner_state.q_params, timestep.observation, action))
            q_values_over_time.append(q_val)
            env_rewards.append(timestep.reward)

            obs = timestep.observation[:7]
            obj_pos = timestep.observation[4:7]
            obj_goal_dists.append(np.linalg.norm(obj_pos - goal_obj_pos))

            next_timestep = env.step(action)

            obs_norm = np.linalg.norm(obs) + 1e-8

            for i, v_k in enumerate(eigenpurposes.T):
                reward = np.dot(obs, v_k)
                rewards_over_time[i].append(reward)

                cos_sim = reward / (obs_norm * (np.linalg.norm(v_k) + 1e-8))
                cos_sims_over_time[i].append(cos_sim)

            timestep = next_timestep


        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Left y-axis: intrinsic rewards, object distance, and env reward
        for i, rewards in enumerate(rewards_over_time):
            ax1.plot(rewards, label=f'Option {i+1}')
        ax1.plot(obj_goal_dists, label='||object - goal||', color='red', linestyle='--')
        ax1.plot(env_rewards, label='Env Reward', color='green', linestyle=':')
        ax1.set_ylabel("Intrinsic Reward / Object Distance / Env Reward")
        ax1.set_xlabel("Timestep")
        ax1.grid(True)

        # Right y-axis: Q-value
        ax2.plot(q_values_over_time, label='Q-value', color='black', linewidth=2)
        ax2.set_ylabel("Q-value")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title(f'Episode {episode+1} - Intrinsic Reward + Q-value + Distance + EnvReward (ckpt {ckpt_num})')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'episode_{episode+1}_intrinsic_ckpt_{ckpt_num}_{num_transitions}_{policy}.png'))
        plt.close()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Left y-axis: cosine similarity, object distance, env reward
        for i, cos_vals in enumerate(cos_sims_over_time):
            ax1.plot(cos_vals, label=f'Option {i+1} (eig={eigenvalues[i]:.2f})')
        ax1.plot(obj_goal_dists, label='||object - goal||', color='red', linestyle='--')
        ax1.plot(env_rewards, label='Env Reward', color='green', linestyle=':')
        ax1.set_ylabel("CosSim / Object Distance / Env Reward")
        ax1.set_xlabel("Timestep")
        ax1.grid(True)

        # Right y-axis: Q-value
        ax2.plot(q_values_over_time, label='Q-value', color='black', linewidth=2)
        ax2.set_ylabel("Q-value")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title(f'Episode {episode+1} - CosSim + Q-value + Distance + EnvReward (ckpt {ckpt_num})')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'episode_{episode+1}_cosine_ckpt_{ckpt_num}_{num_transitions}_{policy}.png'))
        plt.close()




# import numpy as np
# import jax
# import jax.numpy as jnp

# def debug_episode_print_states(ckpt_num,
#                                 eigenpurposes_path='eigenpurposes.npy',
#                                 eigenvalues_path='eigenvalues.npy',
#                                 env_name='sawyer_bin',
#                                 seed=None,
#                                 alpha='0.1',
#                                 misc_params='0.1_None',
#                                 print_every=20,
#                                 policy='policy'):
#     """
#     Simulates one episode and prints full state every N steps for debugging.
#     Useful for checking if episodes are identical across runs.
    
#     Args:
#         ckpt_num (int): Which checkpoint to load.
#         seed (int or None): If provided, sets fixed random seed for env.
#         print_every (int): How often to print the state.
#         policy (str): If 'random', uses random actions; otherwise uses learned policy.
#     """
#     # Load checkpoint and networks
#     trained_learner_state, env, networks = load_checkpoint(
#         alpha=alpha, misc_params=misc_params, env_name=env_name,
#         base_log_dir='', seed=np.random.randint(1e6),
#         fix_goals=True, ckpt_num=ckpt_num)

#     @jax.jit
#     def select_action(params, obs):
#         dist = networks.policy_network.apply(params, obs)
#         return dist.mode()

#     # Set fresh seed or use provided one
#     runtime_seed = seed if seed is not None else np.random.randint(1e6)
#     print(f"\n🔁 Running episode with seed: {runtime_seed}")
    

    
#     for episode in range(3):
#         env.seed(np.random.randint(1e6))
#         timestep = env.reset()
#         step = 0
#         obs_t = timestep.observation[:7]  # can print more if needed
#         while not timestep.last():
#             if step % print_every == 0:
#                 print(f"Step {step}: observation = {np.round(timestep.observation, 4)}")

#             if policy == 'random':
#                 action = env.action_space.sample()
#             else:
#                 obs_batch = jnp.expand_dims(jnp.array(timestep.observation), axis=0)
#                 action = np.array(select_action(trained_learner_state.policy_params, obs_batch)).squeeze()

#             timestep = env.step(action)
#             step += 1

#         print(f"\n✅ Episode finished in {step} steps.")

if __name__ == "__main__":
    num_transitions=240000
    policy = 'policy'  # or 'random'
    #collect_unique_transitions_single_ckpt(ckpt_num=160, num_transitions=num_transitions, save_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/T_{num_transitions}_{policy}_unique.npy', policy_type = policy)
    
    # eigenpurposes, eigenvalues = compute_eigenpurposes_from_T(
    #     T_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/T_{num_transitions}_{policy}_unique.npy',
    #     save_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/eigenpurposes_{num_transitions}_{policy}.npy',
    #     eigenvalues_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/eigenvalues_{num_transitions}_{policy}.npy',
    #     top_k=None  # optional
    # )
    # plot_intrinsic_metrics_from_episode(
    #     ckpt_num=160,
    #     eigenpurposes_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/eigenpurposes_{num_transitions}_{policy}.npy',
    #     eigenvalues_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/eigenvalues_{num_transitions}_{policy}.npy',
    #     num_transitions=num_transitions,
    #     policy = policy,
    # )
    #debug_episode_print_states(ckpt_num=160)

    plot_intrinsic_metrics_multiple_episodes(
    ckpt_num=160,
    num_transitions=num_transitions,
    eigenpurposes_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/eigenpurposes_{num_transitions}_{policy}.npy',
    eigenvalues_path=f'./logs/contrastive_cpc_sawyer_bin_{seed_num}/eigenvalues_{num_transitions}_{policy}.npy',
    policy=policy,
    plot_dir=f'plots/{seed_num}',
    )


'''
seed = 52

Eigenvalue 1: 11.557190
Eigenvector 1: [ 0.06526066 -0.05166191  0.04509981  0.99544305 -0.01038855  0.002323
 -0.00423566]

Eigenvalue 2: 3.200727
Eigenvector 2: [ 0.8923977  -0.2872677   0.2619923  -0.08287542  0.20765103 -0.02600793
  0.04242622]

Eigenvalue 3: 2.021045
Eigenvector 3: [ 0.29655915  0.8316962  -0.24635     0.03654864  0.25154954  0.3024678
 -0.05954798]

Eigenvalue 4: 1.603939
Eigenvector 4: [ 0.11463037 -0.31692943 -0.84845847  0.01547118  0.19201106 -0.15723236
 -0.32358035]

Eigenvalue 5: 1.133106
Eigenvector 5: [-0.29675597 -0.07139707  0.17947371  0.01793841  0.92247134 -0.11639956
  0.09901281]

Eigenvalue 6: 0.868058
Eigenvector 6: [ 0.10106368  0.28200048 -0.12584552  0.01678095 -0.07428756 -0.82884437
  0.44904512]

Eigenvalue 7: 0.720649
Eigenvector 7: [-0.00141736  0.19499642  0.31734532 -0.00680279 -0.01247828 -0.42729092
 -0.8237031 ]

 


 32


 Eigenvalue 1: 11.884020
Eigenvector 1: [ 5.73255413e-04 -1.30123645e-02 -2.35855002e-02  9.98919010e-01
  9.98940971e-03 -3.17249745e-02 -1.81285925e-02]

Eigenvalue 2: 4.061943
Eigenvector 2: [ 0.86782897 -0.23273319  0.16798113 -0.00615653  0.39291385 -0.0974872
  0.02382195]

Eigenvalue 3: 3.094311
Eigenvector 3: [ 0.19858573  0.7632358  -0.22943483  0.01700507  0.24281226  0.5114808
 -0.06734165]

Eigenvalue 4: 2.150633
Eigenvector 4: [ 0.05398392  0.09894654  0.7946061   0.04122834 -0.32420778  0.3825654
  0.3205147 ]

Eigenvalue 5: 1.602991
Eigenvector 5: [ 0.4359141   0.03196549 -0.25707945 -0.00247416 -0.81268185  0.01588035
 -0.28663072]

Eigenvalue 6: 1.213178
Eigenvector 6: [-0.0928048  -0.5397359  -0.05199671  0.00551199  0.10954174  0.70983374
 -0.4259989 ]

Eigenvalue 7: 0.943507
Eigenvector 7: [ 0.07673412 -0.24700205 -0.46732572  0.0098732  -0.0948644   0.27821708
  0.79259515]


'''