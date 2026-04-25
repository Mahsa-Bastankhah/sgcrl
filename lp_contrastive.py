
r"""Example running contrastive RL in JAX.

Run using multi-threading
  python lp_contrastive.py --lp_launch_type=local_mt


"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports
import functools
from typing import Any, Dict

from absl import app
from absl import flags
import contrastive
from contrastive import utils as contrastive_utils
import launchpad as lp
import numpy as np
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/', 'Where to log metrics')
flags.DEFINE_integer('time_delta_minutes', 5, 'how often to save checkpoints')
flags.DEFINE_integer('seed', 42, 'Specify seed, only used if use_slurm_array is false')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('alg', 'contrastive_cpc', 'Algorithm type: contrastive_cpc | c_learning | nce+c_learning | kappa_sac | q_sac')
flags.DEFINE_string('env', 'sawyer_bin', 'Environment type, e.g. default is sawyer bin')
flags.DEFINE_integer('num_steps', 8_000_000, 'Number of steps to run', lower_bound=0)
flags.DEFINE_bool('sample_goals', False, 'sample the goal position uniformly according to the environment (corresponds to the original contrastive_rl algorithm)')
# Rate-limiter knob: samples-per-insert (SPI) for the reverb table.  This is
# the single biggest wall-clock knob in the distributed setup — dropping it
# from 256 to 32-64 typically gives 4-8× speedup on `point_*` envs because
# the learner is much faster than the actors can feed it at high SPI.
# Defaults to whatever ContrastiveConfig.samples_per_insert is (256).
flags.DEFINE_float('samples_per_insert', -1.0,
                   'Reverb samples-per-insert rate limit.  <0 keeps the '
                   'ContrastiveConfig default (256).')
# HER auxiliary coefficient for reward_shaping_mode=q (overrides default 2.5).
# Negative = keep whatever the alg block sets.
flags.DEFINE_float('q_actor_her_aux_coef', -1.0,
                   'Weight on the HER-relabeled auxiliary loss when '
                   'reward_shaping_mode=q (Launchpad --alg=q_sac).  <0 keeps the '
                   'alg-block default (2.5).')
# Trust-region KL on the actor for reward_shaping_mode q or kappa (see learning.py).
flags.DEFINE_float(
    'q_actor_kl_to_prev_coef', -1.0,
    'β on mean(log π_new - log π_prev) at the same action.  '
    '<0 keeps ContrastiveConfig default (0.0).')
# κ Bellman discount γ_κ (only used when training κ, e.g. --alg=kappa_sac).
# Negative keeps ContrastiveConfig default (0.95).
flags.DEFINE_float(
    'discount_kappa', -1.0,
    'κ bootstrap discount γ_κ.  <0 keeps ContrastiveConfig default (0.95).')
# κ Adam LR (only when training κ, e.g. --alg=kappa_sac).  <0 keeps default.
flags.DEFINE_float(
    'learning_rate_kappa', -1.0,
    'Adam learning rate for κ (and twin κ).  <0 keeps ContrastiveConfig '
    'default (1e-4).')
# Global grad-norm clip for κ; 0 disables clipping (see learning.py).  <0
# keeps ContrastiveConfig default (1.0).
flags.DEFINE_float(
    'kappa_max_grad_norm', -1.0,
    'optax global norm clip for κ gradients.  0 disables.  <0 keeps '
    'ContrastiveConfig default (1.0).')
# Twin κ (TD3-style min in the actor).  Only used for --alg=kappa_sac.
# Default True matches historical runs; pass --notwin_kappa for a single κ.
flags.DEFINE_bool(
    'twin_kappa', True,
    'If True, train κ₁ and κ₂ and use min(κ₁·ψ, κ₂·ψ) in the actor.  '
    'If False, one κ only.  Applies to --alg=kappa_sac only.')
# κ actor only: auxiliary on ψ(hard_goal) when set in config.  0 = off (default).
flags.DEFINE_float(
    'kappa_actor_hard_goal_coef', 0.0,
    'Adds -coef·mean(κ·ψ(s,a,g_hard)) to the κ actor loss (--alg=kappa_sac).')
# If True, L2-normalize φ/ψ in the critic (see ContrastiveConfig.repr_norm).
flags.DEFINE_bool('repr_norm', False,
                  'If True, L2-normalize critic φ and ψ before the dot product.')
flags.DEFINE_bool(
    'kappa_actor_match_phi_norm', False,
    'If True (kappa_sac only): in the κ actor loss, rescale each κ row so '
    '‖κ‖=‖φ‖ (φ from the same critic forward) before κ·ψ.  Orthogonal to '
    'repr_norm.')
# Comma-separated trunk widths for policy/critic/κ/etc. (see make_networks).
# Example: --hidden_layer_sizes=256,256,256  or  --hidden_layer_sizes="(256, 256, 256)"
# Empty string keeps ContrastiveConfig default (256, 256).
flags.DEFINE_string(
    'hidden_layer_sizes', '',
    'Comma-separated hidden widths, e.g. 256,256,256. Empty = config default.')

# fixed goal coordinates for supported environments
fixed_goal_dict={'point_Spiral11x11': [np.array([5, 5], dtype=float),
                                       np.array([10, 10], dtype=float)],
                 'point_FourRooms':   [np.array([0, 0], dtype=float),
                                       np.array([10, 8],  dtype=float)],
                 'point_Impossible': [np.array([9, 0], dtype=float),
                                      np.array([7, 9], dtype=float)],
                 #note: sawyer fixed goal positions vary slightly with each episode
                      'sawyer_bin': np.array([0.12, 0.7, 0.02]),
                      'sawyer_box': np.array([0.0, 0.75, 0.133]),
                      'sawyer_peg': np.array([-0.3, 0.6, 0.0]),
                      # One-hot goal; length must match RIVERSWIM_LEN (default 6).
                      'riverswim': np.array(
                          [0., 0., 0., 0., 0., 1.], dtype=float)}


def _extract_hard_goal(env_name):
  """Returns goal slice in the same layout as env observations.

  `hard_goal` must match the goal-part dimensionality of observations used by
  `make_environment` (i.e. obs[:, obs_dim:]).  For sawyer envs this is NOT the
  raw 3D xyz goal from `fixed_goal_dict`; it is the full goal template emitted
  by each env's `_get_obs()` in `env_utils.py`.
  """
  if env_name not in fixed_goal_dict:
    return None
  value = fixed_goal_dict[env_name]
  # 2D point envs store [start, goal].
  if env_name.startswith('point_'):
    if isinstance(value, (list, tuple)):
      if len(value) == 0:
        return None
      value = value[-1]
    arr = np.asarray(value, dtype=float).reshape(-1)
    return tuple(float(x) for x in arr)

  # Sawyer envs in fixed_goal_dict hold raw xyz goals. Expand to the full
  # goal vectors that env observations actually append.
  xyz = np.asarray(value, dtype=float).reshape(-1)
  if xyz.shape[0] != 3:
    return tuple(float(x) for x in xyz)

  if env_name == 'sawyer_bin':
    # Matches env_utils.SawyerBin._get_obs goal construction.
    goal = np.concatenate([xyz + np.array([0.0, 0.0, 0.03]),
                           np.array([0.4]), xyz])
    return tuple(float(x) for x in goal)
  if env_name == 'sawyer_peg':
    # Matches env_utils.SawyerPeg._get_obs goal construction.
    goal = np.concatenate([xyz + np.array([0.13, 0.0, 0.03]),
                           np.array([0.4]), xyz])
    return tuple(float(x) for x in goal)
  if env_name == 'sawyer_box':
    # Matches env_utils.SawyerBox._get_obs goal construction.
    goal_quat = np.array([0.707, 0.0, 0.0, 0.707], dtype=float)
    goal = np.concatenate([xyz + np.array([0.0, 0.0, 0.03]),
                           np.array([0.4]), xyz, goal_quat])
    return tuple(float(x) for x in goal)

  return tuple(float(x) for x in xyz)

@functools.lru_cache
def get_env(env_name, start_index, end_index, seed, fix_goals = False, fix_goals_actor = False, use_naive_sampling=False, clock_period=None):
  if fix_goals:
    fixed_start_end = fixed_goal_dict[env_name]
  else:
    fixed_start_end = None
    
  return contrastive_utils.make_environment(env_name, start_index, end_index, seed=seed, fixed_start_end = fixed_start_end)


def get_program(params):
  """Constructs the program."""

  env_name = params['env_name']
  seed = params['seed']

  config = contrastive.ContrastiveConfig(**params)
  
  fix_goals = params['fix_goals']

  if fix_goals:
    fixed_start_end = fixed_goal_dict[env_name]
  else:
    fixed_start_end = None
    
  env_factory = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_start_end)

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
      hidden_layer_sizes=config.hidden_layer_sizes)
    
  env_factory_fixed_goals = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed, fixed_start_end = fixed_goal_dict[env_name])
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
  add_uid = FLAGS.add_uid
  params['add_uid'] = add_uid
  print('Adding uid: {}...'.format(params['add_uid']))
  
  params['log_dir'] = FLAGS.log_dir_path
  params['time_delta_minutes'] = FLAGS.time_delta_minutes
  hard_goal = _extract_hard_goal(env_name)
  if hard_goal is not None:
    params['hard_goal'] = hard_goal
    print('Using hard_goal={}...'.format(hard_goal))

  if FLAGS.samples_per_insert > 0:
    params['samples_per_insert'] = FLAGS.samples_per_insert
    print('Using samples_per_insert={}...'.format(FLAGS.samples_per_insert))
  
  if alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
  elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
  elif alg == 'kappa_sac':
    # Contrastive CPC critic + κ Bellman network(s) + SAC actor on κ·ψ.
    # Uses the shared adaptive-α pipeline with `config.target_entropy`
    # (default 0.0 — same as stock CRL).  Bump `target_entropy` via the
    # main config / CLI if you want a more entropic κ-actor.
    params['use_cpc'] = True
    params['use_kappa'] = True
    params['twin_kappa'] = bool(FLAGS.twin_kappa)
    params['reward_shaping_mode'] = 'kappa'
    params['kappa_actor_hard_goal_coef'] = float(
        FLAGS.kappa_actor_hard_goal_coef)
    print('Using twin_kappa={}...'.format(params['twin_kappa']))
    print('Using kappa_actor_hard_goal_coef={}...'.format(
        FLAGS.kappa_actor_hard_goal_coef))
  elif alg == 'q_sac':
    # Contrastive CPC critic + twin scalar Q(s,a,g) trained TD3-style on
    # r = sg(φ·ψ) + SAC actor on min(Q1, Q2).  Much simpler bootstrap
    # target than κ (scalar vs. R^repr_dim) so the value network stays
    # in [0, 1/(1-γ_q)] with repr_norm=True.  Shares the adaptive-α
    # pipeline with every other SAC-flavoured actor here.
    params['use_cpc'] = True
    params['use_q_repr'] = True
    params['twin_q_repr'] = True
    params['reward_shaping_mode'] = 'q'
    params['q_actor_her_aux_coef'] = 0
    if FLAGS.q_actor_her_aux_coef >= 0:
      params['q_actor_her_aux_coef'] = FLAGS.q_actor_her_aux_coef
      print('Using q_actor_her_aux_coef={}...'.format(
          FLAGS.q_actor_her_aux_coef))
  else:
    raise NotImplementedError('Unknown method: %s' % alg)

  if FLAGS.discount_kappa >= 0:
    params['discount_kappa'] = float(FLAGS.discount_kappa)
    print('Using discount_kappa={}...'.format(FLAGS.discount_kappa))
  if FLAGS.learning_rate_kappa >= 0:
    params['learning_rate_kappa'] = float(FLAGS.learning_rate_kappa)
    print('Using learning_rate_kappa={}...'.format(FLAGS.learning_rate_kappa))
  if FLAGS.kappa_max_grad_norm >= 0:
    params['kappa_max_grad_norm'] = float(FLAGS.kappa_max_grad_norm)
    print('Using kappa_max_grad_norm={}...'.format(FLAGS.kappa_max_grad_norm))
  if FLAGS.q_actor_kl_to_prev_coef >= 0:
    params['q_actor_kl_to_prev_coef'] = float(FLAGS.q_actor_kl_to_prev_coef)
    print('Using q_actor_kl_to_prev_coef={}...'.format(
        FLAGS.q_actor_kl_to_prev_coef))
  params['repr_norm'] = bool(FLAGS.repr_norm)
  if FLAGS.repr_norm:
    print('Using repr_norm=True...')
  params['kappa_actor_match_phi_norm'] = bool(
      FLAGS.kappa_actor_match_phi_norm)
  if FLAGS.kappa_actor_match_phi_norm:
    print('Using kappa_actor_match_phi_norm=True...')

  hls = FLAGS.hidden_layer_sizes.strip()
  if hls:
    hls = hls.strip('()[]')
    parts = [int(x.strip()) for x in hls.split(',') if x.strip()]
    if not parts:
      raise ValueError(
          f'--hidden_layer_sizes={FLAGS.hidden_layer_sizes!r} parsed to no integers')
    params['hidden_layer_sizes'] = tuple(parts)
    print('Using hidden_layer_sizes={}...'.format(params['hidden_layer_sizes']))

  program = get_program(params)
  # Set terminal='tmux' if you want different components in different windows.
  
  print(params)
  
  lp.launch(program, terminal='current_terminal')

if __name__ == '__main__':
  app.run(main)
