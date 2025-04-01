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

# hyperparams (mostly unimportant)

# NOTE: uses cpc and L2 critic
config = ContrastiveConfig()
# config.use_cpc_symm = True
config.use_goal_action = True
config.use_quasimetric_logit = True
config.use_goal_potential = True
config.twin_q = True

obs_repr_shape = 14
state_repr_shape = 7
action_repr_shape = 4
action_list = [[1,0], [1,1], [0,1], [0,0], [-1,0], [-1,-1], [0,-1], [-1,1], [1,-1]]
# observation = np.array([5,5,10,10], dtype=float)

logger.addFilter(CheckTypesFilter())

axes_names = ['x', 'y', 'z']

def load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals = False, ckpt_num = None, NUM_EPISODES = 10, alg = 'contrastive_cpc'):
    state_entropy_coefficient = alpha
    
    ckpt_dir = '{}/{}_{}_{}_{}/checkpoints/learner'.format(log_dir, alg, env_name, misc_params, seed)
    
    fixed_start_end = None
    fixed_goal = False
        
    if fix_goals:
        if env_name == 'sawyer_box' or env_name == 'sawyer_peg':
            fixed_start_end = np.array([0.0, 0.75, 0.133])
        else:
            assert env_name == 'sawyer_bin'
            fixed_start_end = np.array([0.12, 0.7, 0.02])
    else:
        fixed_start_end = None
    
    
    env_factory = lambda seed: make_environment(env_name, config.start_index, 
                                                config.end_index, seed=np.random.randint(1e6), fixed_start_end=fixed_start_end)[0] 
    dummy_seed = 1
    environment_spec = specs.make_environment_spec(env_factory(np.random.randint(1e6)))

    obs_dim = make_environment(env_name, config.start_index, config.end_index, seed=np.random.randint(1e6), fixed_start_end=fixed_start_end)[1]
                                   
    network_factory = functools.partial(
        make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
        repr_norm=config.repr_norm, twin_q=False,
        use_image_obs=config.use_image_obs,
        hidden_layer_sizes=config.hidden_layer_sizes,
    )

    random_key = jax.random.PRNGKey(np.random.choice(int(1e6)))
    networks = network_factory(environment_spec)
    policy_optimizer = optax.adam(
        learning_rate=config.actor_learning_rate)
    q_optimizer = optax.adam(.001)#learning_rate=config.critic_learning_rate

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

def eval_fixed_goal_exploration(alpha, misc_params, env_name, log_dir, seeds_list, ckpt_num, NUM_EPISODES=10, IMAGE_HEIGHT = 480, IMAGE_WIDTH = 640, render_images=False):
    success_rates_dict = {}
    returns_dict = {}
    goals_dict = {}
    images_dict = {}
    
    # mean_sr_list_ent = []
    # std_sr_list_ent = []
    # mean_returns_list_ent = []
    # std_returns_list_ent = []
    
    # returns_list = []
    # success_rate_list = []
    positions_dict = {}
    for seed in seeds_list:
        trained_learner_state, env, networks = load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals = True, ckpt_num = ckpt_num)
        
        episode_returns = np.zeros(NUM_EPISODES)
        positions_dict[seed] = []
        goals_dict[seed] = []
        imgs = np.zeros([NUM_EPISODES, env._step_limit, IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=np.uint8)
        
        for epi in tqdm(range(NUM_EPISODES)):
            t = 0
            env.seed(np.random.randint(1e6)) 
            timestep = env.reset()
            episode_return = 0
            positions_dict[seed].append([])

            while not timestep.last():
                if render_images:
                    imgs[epi, t] = env.render("gripperPOV") #mode='rgb_array', height=IMAGE_HEIGHT, width=IMAGE_WIDTH

                dist = networks.policy_network.apply(
                    trained_learner_state.policy_params,
                    timestep.observation
                )
                action = np.array(dist.mode())
                timestep = env.step(action)
                positions_dict[seed][-1].append(deepcopy(env.get_info())['obj_pos'])

                t += 1
                episode_return += timestep.reward

            # print("episode return = {}".format(episode_return))
            episode_returns[epi] = episode_return
            if env_name == 'sawyer_bin':
                goals_dict[seed].append(deepcopy(env._goal))
            else:
                assert env_name == 'sawyer_box'
                goals_dict[seed].append(deepcopy(env._goal_pos))

        returns_dict[seed] = episode_returns
        success_rates_dict[seed] = episode_returns >= 1
        imgs = imgs.reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        images_dict[seed] = imgs

        print("avg episode return: {}".format(np.mean(returns_dict[seed])))
        print("success rate: {}".format(np.mean(success_rates_dict[seed])))
        print()

        # mean_returns_list_ent.append(np.mean(returns_list))
        # std_returns_list_ent.append(np.std(returns_list))
        # mean_sr_list_ent.append(np.mean(success_rate_list))
        # std_sr_list_ent.append(np.std(success_rate_list))
    # print("overall avg episode return: {}".format(np.mean(returns_list)))
    # print("overall avg sucess rate: {}".format(np.mean(success_rate_list)))
    # print()

    # return mean_sr_list_ent, std_sr_list_ent, mean_returns_list_ent, std_returns_list_ent, positions_dict
    if render_images:
        return returns_dict, success_rates_dict, positions_dict, goals_dict, images_dict
    return returns_dict, success_rates_dict, positions_dict, goals_dict
  
def get_cells_visited(positions, grid_width=0.005, grid_min=-50, grid_max=50, num_dims=3):
    '''
    positions: agent (x,y,z)-positions per episode per batch; (batch_size, episodes_length, num_dims=3)
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
  
# def get_example_log_dir_seed(env_name, goal_type):
#     '''
#     Chosen to be settings + seeds where both FG and RG gets significant nonzero success
#     '''
#     # env + L2 / dot product
#     if env_name == 'sawyer_box':
#         log_dir = '/home/graceliu/crl/cpc_l2/'
#         seed = 6
#     else:
#         assert env_name == 'sawyer_bin'
#         log_dir = '/home/graceliu/crl/cpc/'
#         seed = 9

#     # FG vs. RG
#     if goal_type == 'fixed':
#         log_dir += 'fixed'
#     else:
#         assert goal_type == 'random'
#         log_dir += 'fixed_eval'

#     return log_dir, seed

def eval_and_get_cells_visited(env_name, log_dir, goal_type, seed, ckpt_num=None, grid_width=0.01,
                               alpha='0.1', render_images=False, **kwargs):
    misc_params = '{}_None'.format(alpha)
    # log_dir = '/home/graceliu/crl_final/cpc/'
    if goal_type == 'fixed':
        log_dir += 'fixed'
    else:
        assert goal_type == 'random'
        log_dir += 'fixed_eval'
    # log_dir, seed = get_example_log_dir_seed(env_name, goal_type)

    # generate
    if render_images:
        returns_dict, success_rates_dict, positions_dict, goals_dict, images_dict = eval_fixed_goal_exploration(alpha, misc_params, env_name, log_dir, [seed], ckpt_num, render_images=True, **kwargs)
        positions = np.array(positions_dict[seed])
        return returns_dict, success_rates_dict, positions_dict, goals_dict, images_dict, get_cells_visited(positions, grid_width=grid_width)
    else:
        returns_dict, success_rates_dict, positions_dict, goals_dict = eval_fixed_goal_exploration(alpha, misc_params, env_name, log_dir, [seed], ckpt_num, **kwargs)
        positions = np.array(positions_dict[seed])
        return returns_dict, success_rates_dict, positions_dict, goals_dict, get_cells_visited(positions, grid_width=grid_width)
      
def gen_images(alpha, misc_params, env_name, log_dir, seed, ckpt_num, NUM_EPISODES=10, IMAGE_HEIGHT = 480, IMAGE_WIDTH = 640):    
    
    imgs_list = []

    trained_learner_state, env, networks = load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals=True, ckpt_num=ckpt_num)
    episode_returns = np.zeros([NUM_EPISODES, ])
    imgs = np.zeros([NUM_EPISODES, env._step_limit, IMAGE_HEIGHT, IMAGE_WIDTH, 3], dtype=np.uint8)
    for epi in range(NUM_EPISODES):
        t = 0
        env.seed(epi)  # use fixed seed for different methods
        timestep = env.reset()
        episode_return = 0

        while not timestep.last():
            # render images
            img = env.render("gripperPOV") #mode='rgb_array', height=IMAGE_HEIGHT, width=IMAGE_WIDTH
            # # add a border if success
            # img = Image.fromarray(img)
            # orig_size = img.size
            # border = (8, 8, 8, 8)
            # if episode_return >= 1:
            #   img = ImageOps.expand(img, border=border, fill="#008000")
            # else:
            #   img = ImageOps.expand(img, border=border, fill="#FF3131")
            # img = img.resize(orig_size)
            # img = np.array(img)
            imgs[epi, t] = img

            dist = networks.policy_network.apply(
              trained_learner_state.policy_params,
              timestep.observation
            )
            action = np.array(dist.mode())
            timestep = env.step(action)

            # Book-keeping.
            t += 1
            episode_return += timestep.reward

        # assert t == env._step_limit
        print("episode return = {}".format(episode_return))
        episode_returns[epi] = episode_return

    imgs = imgs.reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    imgs_list.append(imgs)
    print("avg episode return: {}".format(np.mean(episode_returns)))
    print("success rate: {}".format(np.mean(episode_returns >= 1)))
    print()

    return imgs_list    
  
def play_video(imgs_list, fname=None):
    '''
    If fname is None, plays it as HTML. If it's provided and not none, saves it at fname
    and doesn't play it.
    '''
    for imgs in imgs_list:
        fig = plt.figure()
        im = plt.imshow(imgs[0, :, :, :])

        plt.close() # this is required to not display the generated image

        def init():
            im.set_data(imgs[0, :, :, :])

        def animate(i):
            im.set_data(imgs[i, :, :, :])
            return im

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=imgs.shape[0],
                                       interval=100)
        if fname is not None:
            FFwriter = animation.FFMpegWriter(fps=10)
            anim.save(fname, writer = FFwriter)
        else:
            video = HTML(anim.to_html5_video())
            display(video)
  
def save_and_play_video(env_name, goal_type, seed, ckpt_num=None, alpha='0.1'):
    misc_params = '{}_None'.format(alpha)
    # log_dir = '/home/graceliu/crl_final/cpc/'
    log_dir = '/home/gliu2/rebuttal/pcpg/'
    if goal_type == 'fixed':
        log_dir += 'fixed'
    else:
        assert goal_type == 'random'
        log_dir += 'fixed_eval'
    # log_dir, seed = get_example_log_dir_seed(env_name, goal_type)
    
    # generate
    imgs_list = gen_images(alpha, misc_params, env_name, log_dir, seed, ckpt_num=ckpt_num)
    
    # save + show video
    fname = '_'.join(['imgs_list', env_name, goal_type, str(ckpt_num), misc_params, log_dir.replace('/', '_'), str(seed)])
    with open(fname, 'wb') as f:
        pickle.dump(imgs_list, f)
    print('Saving to', fname)
    play_video(imgs_list)
      
def get_psi_norms(env_name, log_dir, goal_type, seed, ckpt_num=None, alg = 'contrastive_cpc', alpha = '0.1',
                  state_repr_shape=7, N_SAMPLES = 10_000, axes_lims=None):
    '''
    Assumes axes are 0, 1, 2. Only supports sawyer bin and box.
    '''
    # TODO currently only supports obj = agent (tying agent and obj positions together)
    
    if axes_lims is None:
        if env_name == 'sawyer_bin':
            axes_lims = {
                0: (-0.3, 0.25),
                1: (0.6, 0.9),
                2: (0.01, 0.15),
            }
        else:
            assert env_name == 'sawyer_box'
            axes_lims = {
                0: (-0.3, 0.25),
                1: (0.45, 0.9),
                2: (0.01, 0.16),
            }
    
    misc_params = '{}_None'.format(alpha)
    # log_dir = '/home/graceliu/crl_final/cpc/'
    if goal_type == 'fixed':
        log_dir += 'fixed'
    else:
        assert goal_type == 'random'
        log_dir += 'fixed_eval'
    # log_dir, seed = get_example_log_dir_seed(env_name, goal_type)
    
    if env_name == 'sawyer_bin':
        obs_dim = 7
    else:
        assert env_name == 'sawyer_box'
        obs_dim = 11

    # load weights + init environment
    trained_learner_state, env, networks = load_checkpoint(alpha, misc_params, env_name, log_dir, seed, fix_goals = True, ckpt_num = ckpt_num)
    env.seed(np.random.randint(1e6))
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
    
    # uniformly sample states within axes_lims
    random_goals = np.random.rand(N_SAMPLES, 2*obs_dim)
    random_goals[:, :obs_dim] = 0
    # transforms U([0,1)) -> U(axis_min, axis_max) for each axis (0, 1, 2)
    for axis in (0, 1, 2):
        axis_min, axis_max = axes_lims[axis]
        random_goals[:, obs_dim+axis] *= axis_max - axis_min
        random_goals[:, obs_dim+axis] += axis_min
    random_goals[:, obs_dim+3] = 0.4 # goal gripper dist
    if env_name == 'sawyer_box':
        random_goals[:, obs_dim+7:] = np.array([[0.707, 0, 0, 0.707]]) # goal quaternions
    random_goals[:, obs_dim+4:obs_dim+7] = random_goals[:, obs_dim+0:obs_dim+3] # obj = agent
    goal_locations = random_goals[:, obs_dim:].copy()
    obs_batch = np.broadcast_to(obs, (N_SAMPLES, 2*obs_dim)).copy()
    obs_batch[:, obs_dim:] = 0
    obs_batch = obs_batch + random_goals
    # assert np.all(obs_batch2[:,:obs_dim] == obs_batch[:,:obs_dim])

    # compute psi norms
    use_phi_critic = False
    use_l2_critic = True # doesn't actually matter if we don't use first return value
    q_action, sa_repr, g_repr = networks.q_network.apply(trained_learner_state.q_params, obs_batch, action_batch,
                                                         use_l2_critic=use_l2_critic, 
                                                         use_phi_critic=use_phi_critic, 
                                                         use_sa_reg=False, 
                                                         alpha=alpha)
    critic = np.diag(q_action)
    psi_norms = np.diag(np.einsum('ik,jk->ij', g_repr, g_repr))
    return goal_locations, psi_norms, critic
