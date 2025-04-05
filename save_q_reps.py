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
seed_num = 32
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
    env_factory = lambda seed: make_environment(env_name, config.start_index, 
                                                config.end_index, seed=seed_num, 
                                                fixed_start_end=fixed_start_end)[0]
    dummy_seed = 1
    environment_spec = specs.make_environment_spec(env_factory(dummy_seed))

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
    env = env_factory(
        utils.sample_uint32(environment_key))

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


# ---------------------------
# Simulation function for one checkpoint.
# ---------------------------


def simulate_checkpoint(ckpt_num, num_episodes=100, alpha='0.1', misc_params='0.1_None',
                        env_name='sawyer_bin', base_log_dir='/dummy/unused', seed=seed_num):
    """
    For the given checkpoint number, load the checkpoint,
    simulate `num_episodes` episodes, and record per-step Q value,
    state-action representation, goal representation, and g_repr_state.
    """
    trained_learner_state, env, networks = load_checkpoint(
        alpha, misc_params, env_name, base_log_dir, seed, fix_goals=True, ckpt_num=ckpt_num)

    simulation_data = {"episodes": []}
    @jax.jit
    def get_q_and_repr(q_params, obs_batch, action_batch):
        return networks.q_network.apply(q_params, obs_batch, action_batch)

    @jax.jit
    def get_g_repr(q_params, obs_state_as_goal, action_batch):
        _, g_repr_state, _ = networks.repr_fn(q_params, obs_state_as_goal, action_batch)
        return g_repr_state
    
    @jax.jit
    def select_action(params, obs):
        dist = networks.policy_network.apply(params, obs)
        return dist.mode()


    
    for epi in tqdm(range(num_episodes), desc=f"Checkpoint {ckpt_num} Simulation"):
        episode_data = []
        env.seed(np.random.randint(1e6))
        timestep = env.reset()

        

        while not timestep.last():
            obs = timestep.observation
            # [ 0.00796224  0.6000598   0.20067754  0.92119074 -0.12937918  0.7408595
            # 0.02999509  0.12        0.7         0.05        0.4         0.12
            # 0.7         0.02      ]

            #print(obs)
            obs_dim = obs.shape[0] // 2  # move outside the loop
            # dist = networks.policy_network.apply(trained_learner_state.policy_params, obs)
            # action = dist.mode()
            action = np.array(select_action(trained_learner_state.policy_params, obs))


            obs_batch = jnp.expand_dims(jnp.array(obs), axis=0)
            action_batch = jnp.expand_dims(jnp.array(action), axis=0)

            #q_value, sa_repr, g_repr = networks.q_network.apply(
            #    trained_learner_state.q_params, obs_batch, action_batch)
            q_value, sa_repr, g_repr = get_q_and_repr(trained_learner_state.q_params, obs_batch, action_batch)


            obs_state_as_goal = jnp.array(obs)
            obs_state_as_goal = obs_state_as_goal.at[obs_dim:].set(obs_state_as_goal[:obs_dim])
            obs_state_as_goal = jnp.expand_dims(obs_state_as_goal, axis=0)

            #_, g_repr_state, _ = networks.repr_fn(
            #    trained_learner_state.q_params, obs_state_as_goal, action_batch)
            g_repr_state = get_g_repr(trained_learner_state.q_params, obs_state_as_goal, action_batch)

            step_data = {
                "observation": np.array(obs),
                "action": np.array(action),
                "q_value": np.squeeze(np.array(q_value)),
                "sa_repr": np.squeeze(np.array(sa_repr)),
                "g_repr": np.squeeze(np.array(g_repr)),
                "g_repr_state": np.squeeze(np.array(g_repr_state)),
                "reward": timestep.reward,
            }
            episode_data.append(step_data)

            timestep = env.step(np.array(action))



        simulation_data["episodes"].append(episode_data)

    return simulation_data


# ---------------------------
# Loop over all checkpoints in the folder and save simulation data.
# ---------------------------
def simulate_all_checkpoints(num_episodes=100, alpha='0.1', misc_params='0.1_None',
                             env_name='sawyer_bin', seed=seed_num):
    ckpt_dir = f'/home/mahsa/sgcrl/logs/contrastive_cpc_sawyer_bin_{seed_num}/checkpoints/learner'
    data_dir = f"./data/{seed_num}"
    os.makedirs(data_dir, exist_ok=True)

    ckpt_nums = list_all_checkpoints(ckpt_dir)
    print("Found checkpoint numbers:", ckpt_nums)

    for ckpt in ckpt_nums:
        if ckpt % 3 != 0:
            continue

        filename = f"checkpoint_{ckpt}_simulation_data.pkl"
        filepath = os.path.join(data_dir, filename)

        if os.path.exists(filepath):
            print(f"Skipping checkpoint {ckpt} — already exists.")
            continue

        print(f"Simulating for checkpoint {ckpt} ...")
        sim_data = simulate_checkpoint(ckpt, num_episodes, alpha, misc_params, env_name,
                                       base_log_dir=None, seed=seed)

        with open(filepath, 'wb') as f:
            pickle.dump(sim_data, f)
        print(f"Saved simulation data for checkpoint {ckpt} to {filepath}")


# ---------------------------
# Example usage.
# ---------------------------
if __name__ == "__main__":
    # Simulate for all found checkpoints.
    simulate_all_checkpoints(num_episodes=100, alpha='0.1', misc_params='0.1_None',
                             env_name='sawyer_bin', seed=1)
