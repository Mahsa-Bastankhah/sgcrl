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
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import jax
import jax.numpy as jnp
import pickle
import json
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
from lp_contrastive import fixed_goal_dict


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
    return fixed_goal_dict[env_name]

# ---------------------------
# Updated load_checkpoint function.
# ---------------------------
def load_checkpoint(alpha, misc_params, env_name, base_log_dir, seed, fix_goals=False, ckpt_num=None, NUM_EPISODES=10, alg='contrastive_cpc', ckpt_dir=None, goal=None):
    state_entropy_coefficient = alpha

    # Here we override the ckpt_dir to be the folder provided.
    #ckpt_dir = f'/home/mahsa/sgcrl/logs/contrastive_cpc_sawyer_bin_{seed_num}/checkpoints/learner'
    if goal is None:
        fixed_start_end = get_fixed_start_end(env_name) if fix_goals else None
    else:
        fixed_start_end = goal
        print("Using provided goal:", goal)

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

