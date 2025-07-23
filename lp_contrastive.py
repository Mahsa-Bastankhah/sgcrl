
r"""Example running contrastive RL in JAX.

Run using multi-threading
  python lp_contrastive.py --lp_launch_type=local_mt


"""
import functools
from typing import Any, Dict
import json
import pathlib  # ← add near the other imports
import json, pathlib, uuid
from absl import app
from absl import flags
import contrastive
from contrastive import utils as contrastive_utils
import launchpad as lp
import numpy as np
import os
from new_point_env import PointEnvExtras   # the 20-dim env you just wrote
from absl import flags
import jax.numpy as jnp   # used later to build arrays

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/', 'Where to log metrics')
flags.DEFINE_integer('time_delta_minutes', 5, 'how often to save checkpoints')
flags.DEFINE_integer('seed', 12, 'Specify seed, only used if use_slurm_array is false')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('alg', 'contrastive_cpc', 'Algorithm type, e.g. default is contrastive_cpc with no entropy or KL losses')
flags.DEFINE_string('env', 'sawyer_bin', 'Environment type, e.g. default is sawyer bin')
flags.DEFINE_integer('num_steps', 2_000_000, 'Number of steps to run', lower_bound=0)
flags.DEFINE_bool('sample_goals', False, 'sample the goal position uniformly according to the environment (corresponds to the original contrastive_rl algorithm)')
flags.DEFINE_string(
    'init_weight',          # flag name
    None,                 # default → no warm-start
    'Path to a pickled/npz checkpoint containing "policy_params" and '
    '"q_params" to use as initial weights.')
flags.DEFINE_bool('Q_max',  False, 'Wether using the actor or the arg max Q policy ')
# e.g. --hidden_layer_sizes=512 --hidden_layer_sizes=512 --hidden_layer_sizes=256
flags.DEFINE_multi_integer(
    'hidden_layer_sizes',
    [256, 256],                # default
    'Sizes of each hidden layer in the policy/Q MLP. '
    'Repeat the flag for each layer, e.g. '
    '"--hidden_layer_sizes=512 --hidden_layer_sizes=256".')
flags.DEFINE_integer('goal_neg_actor_steps', 0, 'Number of actor steps to use goal as a negative example', lower_bound=0)
flags.DEFINE_integer('goal_pos_actor_steps', 0, 'Number of actor steps to use goal as a positive example', lower_bound=0)
flags.DEFINE_bool('softmax_repr', False, 'Whether to do softmax normalization on the representation. ')
flags.DEFINE_bool('cold_q_init', False, 'Whether to do cold initialization for the Q network. ')
flags.DEFINE_float('cold_q_scale', 1e-12 , 'Cold initialization scale for the Q network. ')
flags.DEFINE_integer('perturbed_negatives_num', 0, 'Number of purturbed negatives to sample. If 0, no perturbation is done.')
flags.DEFINE_integer('perturbed_negatives_goal_num', 0, 'Number of purturbed negatives to sample for the goal. If 0, no perturbation is done.')
flags.DEFINE_string('fixed_goal', None, 'Override the fixed goal with a custom goal coordinate as a comma-separated string, e.g., "0.1,0.2,0.3"')
flags.DEFINE_bool('use_residual_mlp', False, 'whether to use residual MLP for representation learning')
# --- NEW: random-feature point env ------------------------------------------
flags.DEFINE_integer(
    'extra_dim', 8,
    'How many additional coordinates to add to state/goal in PointEnvExtras.')
# ---------------------------------------------------------------------------

flags.DEFINE_float('goal_pos_frac', 0.05, 'fraction of fake goal positive sampling in the critic loss')
flags.DEFINE_integer('weight_reset_interval', 0, 'Interval for resetting weights, 0 means no reset')
flags.DEFINE_bool('backward_loss', False, 'Whether to use backward loss')

flags.DEFINE_string(
    'region_bounds',
    None,
    'Axis-aligned box for masking, formatted '
    '"x_lo,y_lo:x_hi,y_hi"  (no spaces). '
    'Omit to disable masking.')
flags.DEFINE_bool(
    'stop_grad_fixed',
    True,
    'Freeze gradients flowing through the substituted goal representation.')

flags.DEFINE_bool(
    'negative_goal_repr',
    True,
    'Using negative goal representation in the designated area, ')


# fixed goal coordinates for supported environments
fixed_goal_dict={'point_Spiral11x11': [np.array([5,5], dtype=float), np.array([10,10], dtype=float)],
                 'point_FourRooms': [np.array([0,0], dtype=float), np.array([10,8], dtype=float)], #[10,8] 
                 'point_Impossible' :  [np.array([9,0], dtype=float), np.array([7 , 9], dtype=float)], # hardest right before the final wall [7,9]
                 'point_Maze11x11' : [np.array([0,0], dtype=float), np.array([5,4], dtype=float)], # hardest [11,11] , [5,4] doable using 1024 network
                 'point_Wall11x11' : [np.array([2,0], dtype=float), np.array([0,0], dtype=float)], # hardest [2,0] [0,0] easier [2,8] [0,10]
                 'random_point_Impossible' :  [np.array([9,0], dtype=float), np.array([7 , 9], dtype=float)], # hardest right before the final wall [7,9]
                     #note: sawyer fixed goal positions vary slightly with each episode
                      'sawyer_bin': np.array([0.12, 0.7, 0.02]),
                      'sawyer_box': np.array([0.0, 0.75, 0.133]),
                      'sawyer_peg': np.array([-0.3, 0.6, 0.0])}

@functools.lru_cache
def get_env(env_name, start_index, end_index, seed, fix_goals = False, fix_goals_actor = False, use_naive_sampling=False, clock_period=None):
  if fix_goals:
    fixed_start_end = fixed_goal_dict[env_name]
  else:
    fixed_start_end = None
  
  if FLAGS.fixed_goal:
    try:
        # Parse string input like "0.1,0.2,0.3"
        goal_coords = np.array([float(x) for x in FLAGS.fixed_goal.split(',')])
        fixed_start_end[1] = goal_coords
        print(f"Overriding fixed goal with custom input: {fixed_start_end}")
    except Exception as e:
        raise ValueError(f"Invalid format for --fixed_goal: {FLAGS.fixed_goal}") from e
    
  return contrastive_utils.make_environment(env_name, start_index, end_index, seed=seed, fixed_start_end = fixed_start_end, extra_dim=FLAGS.extra_dim)


def get_program(params):
  """Constructs the program."""

  env_name = params['env_name']
  seed = params['seed']

  config = contrastive.ContrastiveConfig(**params)
  
  fix_goals = params['fix_goals']
  print('Using fixed goals: {}...'.format(fix_goals))

  if fix_goals:
    fixed_start_end = fixed_goal_dict[env_name]
  else:
    fixed_start_end = None


  if FLAGS.fixed_goal:
    try:
        # Parse string input like "0.1,0.2,0.3"
        goal_coords = np.array([float(x) for x in FLAGS.fixed_goal.split(',')])
        fixed_start_end[1] = goal_coords
        print(f"Overriding fixed goal with custom input: {fixed_start_end}")
    except Exception as e:
        raise ValueError(f"Invalid format for --fixed_goal: {FLAGS.fixed_goal}") from e


  print('Using fixed start and end: {}...'.format(fixed_start_end))
    
  env_factory = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_start_end, extra_dim=FLAGS.extra_dim)

  env_factory_no_extra = lambda seed: env_factory(seed)[0]  # Remove obs_dim.
    
  environment, obs_dim = get_env(env_name, config.start_index,
                                 config.end_index, seed, fix_goals = fix_goals)

  assert (environment.action_spec().minimum == -1).all()
  assert (environment.action_spec().maximum == 1).all()
  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(environment, '_step_limit') + 1
  network_factory = functools.partial(
      contrastive.make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
      repr_norm=config.repr_norm, twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes, config=config)
    
  env_factory_fixed_goals = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_goal_dict[env_name], extra_dim=FLAGS.extra_dim)
  env_factory_no_extra_fixed_goals = lambda seed: env_factory_fixed_goals(seed)[0]  # Remove obs_dim.
    
  agent = contrastive.DistributedContrastive(
      seed=seed,
      environment_factory=env_factory_no_extra,
      environment_factory_fixed_goals=env_factory_no_extra_fixed_goals,
      network_factory=network_factory,
      config=config,
      num_actors=config.num_actors,
      log_to_bigtable=True,
      max_number_of_steps=config.max_number_of_steps)
  return agent.build()


def main(_):
  # Create experiment description.

  # 1. Select an environment.
  # Supported environments:
  #   Metaworld: sawyer_{bin,box,peg}
  #   2D nav: point_{Spiral11x11}
  env_name = FLAGS.env
  print('Using env {}...'.format(env_name))
  print(" fixed goal being an instance of ", type(fixed_goal_dict.get(env_name, None)))
  if isinstance(fixed_goal_dict.get(env_name, None), (list, tuple, dict)):
    goal_coords = fixed_goal_dict.get(env_name, None)[1]
  else:
    obj_coords = fixed_goal_dict.get(env_name, None)
    goal_coords = np.concatenate([obj_coords + np.array([0.0, 0.0, 0.03]),
                           [0.4], obj_coords])


  if FLAGS.fixed_goal:
    print(f"Overriding fixed goal with custom input: {FLAGS.fixed_goal}")
    try:
        # Parse string input like "0.1,0.2,0.3"
        goal_coords = np.array([float(x) for x in FLAGS.fixed_goal.split(',')])
        print(f"Overriding fixed goal with custom input: {goal_coords}")
    except Exception as e:
        raise ValueError(f"Invalid format for --fixed_goal: {FLAGS.fixed_goal}") from e
  
  seed_idx = FLAGS.seed
  print('Using random seed {}...'.format(seed_idx))
  params = {
      'seed': seed_idx,
      'use_random_actor': True,
      # entropy_coefficient = None will use adaptive; if setting to a number, note this is log alpha
      'entropy_coefficient': 0.0,
      'env_name': env_name,
      # the number of environment steps
      'max_number_of_steps': FLAGS.num_steps,
  }
  # 2. Select an algorithm. The currently-supported algorithms are:
  # contrastive_nce, contrastive_cpc, c_learning, nce+c_learning
  # Many other algorithms can be implemented by passing other parameters
  # or adding a few lines of code.
  # By default, do contrastive CPC
  alg = FLAGS.alg
  print('Using alg {}...'.format(alg))
  params['alg_name'] = alg
  params['fix_goals'] = not FLAGS.sample_goals
  params['hidden_layer_sizes'] = tuple(FLAGS.hidden_layer_sizes)
  params['goal_neg_actor_steps'] = FLAGS.goal_neg_actor_steps
  params['use_residual_mlp'] = FLAGS.use_residual_mlp
  params['goal_pos_actor_steps'] = FLAGS.goal_pos_actor_steps
  add_uid = FLAGS.add_uid
  params['softmax_repr'] = FLAGS.softmax_repr
  params['cold_q_init'] = FLAGS.cold_q_init
  params['perturbed_negatives_num'] = FLAGS.perturbed_negatives_num
  params['perturbed_negatives_goal_num'] = FLAGS.perturbed_negatives_goal_num
  params['cold_q_scale'] = FLAGS.cold_q_scale
  if FLAGS.sample_goals:
    params['fixed_goal'] = None
  else:
    params['fixed_goal'] = tuple(goal_coords.astype(float))
    print('Using fixed goal: {}...'.format(params['fixed_goal']))
  params['add_uid'] = add_uid
  params['Q_max'] = FLAGS.Q_max
  params['init_weight'] = FLAGS.init_weight
  print('Adding uid: {}...'.format(params['add_uid']))

  params['goal_pos_frac'] = FLAGS.goal_pos_frac  # whether to use naive sampling for the goal
  
  params['log_dir'] = FLAGS.log_dir_path
  params['time_delta_minutes'] = FLAGS.time_delta_minutes
  params['weight_reset_interval'] = FLAGS.weight_reset_interval
  params['backward_loss'] = FLAGS.backward_loss
  if FLAGS.region_bounds is None:
    params['region_bounds'] = None
  else:
      try:
          # 1. split on ';'  → list of "lo:hi" strings
          boxes = [b.strip() for b in FLAGS.region_bounds.split(';') if b.strip()]
          lowers, uppers = [], []

          for box in boxes:
              lower_str, upper_str = box.split(':')
              lower = list(map(float, lower_str.split(',')))
              upper = list(map(float, upper_str.split(',')))
              if len(lower) != len(upper):
                  raise ValueError(f"Mismatched dims in '{box}'")
              lowers.append(lower)
              uppers.append(upper)

          # 2. store as list-of-lists so JAX helper can broadcast
          params['region_bounds'] = (lowers, uppers)

      except Exception as e:
          raise ValueError(
              f"Bad --region_bounds '{FLAGS.region_bounds}'. "
              "Use 'x_lo,y_lo:x_hi,y_hi' -- or multiple boxes separated by ';'."
          ) from e

  params['stop_grad_fixed'] = FLAGS.stop_grad_fixed
  params['negative_goal_repr'] = FLAGS.negative_goal_repr
  # ---- sanity-check ----------------------------------------------------
  rb = params.get('region_bounds')
  if rb is None:
      print("[mask] region_bounds = None  → masking DISABLED")
  else:
      lowers, uppers = rb
      for i, (lo, hi) in enumerate(zip(lowers, uppers)):
          print(f"[mask] box {i}: lower={lo}  upper={hi}")
      print(f"stop_grad_fixed = {params['stop_grad_fixed']}")
# stop-grad flag
  
  if alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
  elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
  else:
    raise NotImplementedError('Unknown method: %s' % alg)


  
  # === NEW BLOCK: persist the run configuration =============
  run_dir = pathlib.Path(params['log_dir']) / f"{params['alg_name']}_{params['env_name']}_{params['seed']}"

  # If you add a UID elsewhere, replicate it here:
  if params.get('add_uid'):               # True/False in FLAGS
      run_dir = run_dir.with_name(run_dir.name + f"_{uuid.uuid4().hex[:6]}")

  run_dir.mkdir(parents=True, exist_ok=True)

  # --------------------------------------------------------------
  # Persist the configuration *inside* that run folder.
  config_file = run_dir / 'config.json'
  with config_file.open('w') as f:
      json.dump(params, f, indent=2, sort_keys=True)

  print(f"Saved run config to {config_file}")
    # ==========================================================



  program = get_program(params)
  # Set terminal='tmux' if you want different components in different windows.
  
  print(params)
  
  lp.launch(program, terminal='current_terminal')

if __name__ == '__main__':
  app.run(main)