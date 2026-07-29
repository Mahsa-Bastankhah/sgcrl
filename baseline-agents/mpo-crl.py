"""Online MPO-CRL baseline.

Example:
  python baseline-agents/mpo-crl.py \
      --env=point_FourRooms \
      --num_steps=8000000 \
      --seed=0
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

import sgcrl_jax_acme_compat  # noqa: E402,F401

from absl import app, flags  # noqa: E402
import numpy as np  # noqa: E402

import contrastive  # noqa: E402
from contrastive import utils as contrastive_utils  # noqa: E402
# Reuse the established environment flags/default tables without running PPO.
import ppo_contrastive as ppo_entry  # noqa: E402
import mpo_crl_learner  # noqa: E402


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'mpo_log_dir_path', 'logs/mpo/', 'Directory for MPO-CRL runs.')
flags.DEFINE_integer(
    'mpo_num_envs', -1, 'Parallel environments; <0 uses environment defaults.')
flags.DEFINE_integer(
    'mpo_rollout_length', -1,
    'Environment steps per actor and iteration; <0 uses environment defaults.')
flags.DEFINE_integer(
    'mpo_policy_batch_size', 256, 'Original-goal replay batch size for MPO.')
flags.DEFINE_integer(
    'mpo_policy_updates_per_iter', 32, 'MPO SGD updates per collection iteration.')
flags.DEFINE_integer(
    'mpo_crl_updates_per_iter', -1,
    'Geometric CRL updates per iteration; <0 uses environment defaults.')
flags.DEFINE_integer(
    'mpo_crl_batch_size', -1,
    'Geometric CRL batch size; <0 keeps ContrastiveConfig.batch_size.')
flags.DEFINE_integer(
    'mpo_min_replay_size', 1000,
    'Minimum original/geometric replay transitions before updates.')
flags.DEFINE_integer(
    'mpo_num_action_samples', 20,
    'Target-policy action samples per state in the MPO E-step.')

flags.DEFINE_float('mpo_policy_learning_rate', 1e-4, 'MPO policy Adam rate.')
flags.DEFINE_float('mpo_dual_learning_rate', 1e-2, 'MPO dual Adam rate.')
flags.DEFINE_float('mpo_policy_grad_norm_clip', 40.0, 'Policy/dual grad clip.')
flags.DEFINE_string(
    'mpo_policy_hidden_sizes', '256,256,256',
    'Comma-separated hidden sizes for the unsquashed Gaussian policy.')
flags.DEFINE_float(
    'mpo_policy_init_scale', 0.7, 'Initial Gaussian policy scale.')
flags.DEFINE_bool(
    'mpo_use_td_critic', False,
    'Train fixed-goal scalar G(s,a) on target CRL rewards and use target G '
    'for MPO instead of directly using phi(s,a) dot psi(g).')
flags.DEFINE_float(
    'mpo_critic_learning_rate', 1e-4, 'Scalar G critic Adam rate.')
flags.DEFINE_float(
    'mpo_critic_grad_norm_clip', 40.0, 'Scalar G global gradient-norm clip.')
flags.DEFINE_string(
    'mpo_critic_hidden_sizes', '512,512,256',
    'Comma-separated hidden sizes for fixed-goal scalar G(s,a).')
flags.DEFINE_integer(
    'mpo_critic_target_update_period', 100,
    'Hard online-to-target G copy period; <=0 enables Polyak updates.')
flags.DEFINE_float(
    'mpo_critic_target_update_rate', 0.005,
    'Polyak online-G weight when critic_target_update_period <=0.')
flags.DEFINE_integer(
    'mpo_bootstrap_action_samples', 20,
    'Target-policy actions averaged in the expected-SARSA G target.')
flags.DEFINE_bool(
    'mpo_normalize_critic_reward', False,
    'Standardize CRL rewards using running EMA mean and variance.')
flags.DEFINE_float(
    'mpo_critic_reward_norm_rate', 0.001,
    'EMA update rate for CRL reward normalization moments.')

flags.DEFINE_float('mpo_epsilon', 0.1, 'Non-parametric MPO KL bound.')
flags.DEFINE_float('mpo_epsilon_mean', 0.0025, 'Policy mean KL bound.')
flags.DEFINE_float('mpo_epsilon_stddev', 1e-6, 'Policy stddev KL bound.')
flags.DEFINE_float(
    'mpo_epsilon_penalty', 0.001, 'Out-of-action-bounds MO-MPO KL bound.')
flags.DEFINE_float(
    'mpo_init_log_temperature', 10.0, 'Initial MPO log temperature.')
flags.DEFINE_float(
    'mpo_init_log_alpha_mean', 10.0, 'Initial mean-KL log dual.')
flags.DEFINE_float(
    'mpo_init_log_alpha_stddev', 1000.0, 'Initial stddev-KL log dual.')
flags.DEFINE_bool(
    'mpo_per_dim_constraining', True, 'Constrain policy KL per action dimension.')
flags.DEFINE_bool(
    'mpo_action_penalization', True, 'Enable MO-MPO out-of-bounds penalty.')

flags.DEFINE_integer(
    'mpo_target_update_period', 100,
    'Hard target-policy copy period; <=0 enables Polyak updates.')
flags.DEFINE_float(
    'mpo_target_update_rate', 0.005,
    'Polyak online weight when target_update_period <=0.')
flags.DEFINE_float(
    'mpo_repr_target_update_rate', 0.005,
    'MPO-style online weight for incremental phi/psi target updates: '
    'target = (1-rate)*target + rate*online.')
flags.DEFINE_integer(
    'mpo_eval_interval', -1, 'Evaluation interval; <0 uses PPO env default.')
flags.DEFINE_integer(
    'mpo_eval_episodes', -1, 'Evaluation episodes; <0 keeps config default.')
flags.DEFINE_integer(
    'mpo_checkpoint_interval', -1,
    'Checkpoint interval; <0 uses PPO env/config default.')
flags.DEFINE_integer(
    'mpo_checkpoint_keep_last', 0,
    'Milestone checkpoints to retain; 0 retains all.')


def _comma_ints(value: str):
  return tuple(int(item) for item in value.split(',') if item.strip())


def main(_):
  env_name = FLAGS.env
  seed = int(FLAGS.seed)
  env_defaults = ppo_entry.ppo_env_defaults_for_env(
      env_name,
      use_pd=(bool(FLAGS.builderbench_use_pd)
              if env_name.startswith('builderbench_') else False))

  config = contrastive.ContrastiveConfig(
      seed=seed,
      env_name=env_name,
      alg_name='mpo_crl',
      reward_shaping_mode='',
      use_cpc=True,
      max_number_of_steps=int(FLAGS.num_steps),
      log_dir=FLAGS.mpo_log_dir_path,
      add_uid=bool(FLAGS.add_uid),
      fix_goals=not bool(FLAGS.sample_goals))
  config.repr_norm = bool(FLAGS.repr_norm)
  config.uniform_sampling = bool(FLAGS.uniform_sampling)
  if FLAGS.discount >= 0.0:
    config.discount = float(FLAGS.discount)
  if FLAGS.max_replay_size >= 0:
    config.max_replay_size = int(FLAGS.max_replay_size)
  if FLAGS.mpo_crl_batch_size > 0:
    config.batch_size = int(FLAGS.mpo_crl_batch_size)
  if FLAGS.ppo_crl_loss_direction.strip():
    config.ppo_crl_loss_direction = (
        FLAGS.ppo_crl_loss_direction.strip().lower())
  if FLAGS.hidden_layer_sizes.strip():
    config.hidden_layer_sizes = _comma_ints(FLAGS.hidden_layer_sizes)

  default_rollout = int(config.ppo_rollout_length)
  default_crl_updates = int(config.ppo_crl_steps_per_iter)
  default_num_envs = int(config.ppo_num_envs)
  default_eval_interval = int(config.ppo_eval_interval)
  default_checkpoint_interval = int(config.ppo_checkpoint_interval)
  if env_defaults is not None:
    default_rollout = int(env_defaults.get('rollout_length', default_rollout))
    default_crl_updates = int(
        env_defaults.get('crl_steps_per_iter', default_crl_updates))
    default_num_envs = int(env_defaults.get('num_envs', default_num_envs))
    default_eval_interval = int(
        env_defaults.get('eval_interval', default_eval_interval))
    default_checkpoint_interval = int(
        env_defaults.get('checkpoint_interval', default_checkpoint_interval))
    if 'start_index' in env_defaults:
      config.start_index = int(env_defaults['start_index'])
    if 'end_index' in env_defaults:
      config.end_index = int(env_defaults['end_index'])

  mpo_config = mpo_crl_learner.MPOCRLConfig(
      num_envs=(int(FLAGS.mpo_num_envs)
                if FLAGS.mpo_num_envs > 0 else default_num_envs),
      rollout_length=(
          int(FLAGS.mpo_rollout_length)
          if FLAGS.mpo_rollout_length > 0 else default_rollout),
      policy_batch_size=int(FLAGS.mpo_policy_batch_size),
      policy_updates_per_iter=int(FLAGS.mpo_policy_updates_per_iter),
      crl_updates_per_iter=(
          int(FLAGS.mpo_crl_updates_per_iter)
          if FLAGS.mpo_crl_updates_per_iter >= 0 else default_crl_updates),
      min_replay_size=int(FLAGS.mpo_min_replay_size),
      num_action_samples=int(FLAGS.mpo_num_action_samples),
      policy_learning_rate=float(FLAGS.mpo_policy_learning_rate),
      dual_learning_rate=float(FLAGS.mpo_dual_learning_rate),
      policy_grad_norm_clip=float(FLAGS.mpo_policy_grad_norm_clip),
      policy_hidden_sizes=_comma_ints(FLAGS.mpo_policy_hidden_sizes),
      policy_init_scale=float(FLAGS.mpo_policy_init_scale),
      use_td_critic=bool(FLAGS.mpo_use_td_critic),
      critic_learning_rate=float(FLAGS.mpo_critic_learning_rate),
      critic_grad_norm_clip=float(FLAGS.mpo_critic_grad_norm_clip),
      critic_hidden_sizes=_comma_ints(FLAGS.mpo_critic_hidden_sizes),
      critic_target_update_period=int(
          FLAGS.mpo_critic_target_update_period),
      critic_target_update_rate=float(FLAGS.mpo_critic_target_update_rate),
      bootstrap_action_samples=int(FLAGS.mpo_bootstrap_action_samples),
      normalize_critic_reward=bool(FLAGS.mpo_normalize_critic_reward),
      critic_reward_norm_rate=float(FLAGS.mpo_critic_reward_norm_rate),
      epsilon=float(FLAGS.mpo_epsilon),
      epsilon_mean=float(FLAGS.mpo_epsilon_mean),
      epsilon_stddev=float(FLAGS.mpo_epsilon_stddev),
      epsilon_penalty=float(FLAGS.mpo_epsilon_penalty),
      init_log_temperature=float(FLAGS.mpo_init_log_temperature),
      init_log_alpha_mean=float(FLAGS.mpo_init_log_alpha_mean),
      init_log_alpha_stddev=float(FLAGS.mpo_init_log_alpha_stddev),
      per_dim_constraining=bool(FLAGS.mpo_per_dim_constraining),
      action_penalization=bool(FLAGS.mpo_action_penalization),
      target_update_period=int(FLAGS.mpo_target_update_period),
      target_update_rate=float(FLAGS.mpo_target_update_rate),
      repr_target_update_rate=float(FLAGS.mpo_repr_target_update_rate),
      eval_interval=(
          int(FLAGS.mpo_eval_interval)
          if FLAGS.mpo_eval_interval >= 0 else default_eval_interval),
      eval_episodes=(
          int(FLAGS.mpo_eval_episodes)
          if FLAGS.mpo_eval_episodes >= 0 else int(config.ppo_eval_episodes)),
      checkpoint_interval=(
          int(FLAGS.mpo_checkpoint_interval)
          if FLAGS.mpo_checkpoint_interval >= 0
          else default_checkpoint_interval),
      checkpoint_keep_last=int(FLAGS.mpo_checkpoint_keep_last))

  total_steps = int(FLAGS.num_steps)
  if env_name.startswith('builderbench_'):
    from envs.builderbench_utils import (
        BUILDERBENCH_NUM_STEPS,
        builderbench_replay_size,
    )
    if FLAGS.max_replay_size < 0:
      config.max_replay_size = builderbench_replay_size(mpo_config.num_envs)
    if FLAGS.num_steps == 8_000_000:
      total_steps = BUILDERBENCH_NUM_STEPS
      config.max_number_of_steps = total_steps

  fixed_start_end = (
      ppo_entry.fixed_goal_for_env(env_name) if config.fix_goals else None)
  if mpo_config.use_td_critic and fixed_start_end is None:
    raise ValueError(
        '--mpo_use_td_critic trains G(s,a) without a goal input and therefore '
        'requires a fixed goal; remove --sample_goals and use an environment '
        'with a configured fixed goal')
  env_kwargs = {}
  if env_name == 'sawyer_bin' and FLAGS.bin_randomize_gripper_init:
    env_kwargs['randomize_gripper_init'] = True
  if env_name.startswith('builderbench_'):
    env_kwargs.update(
        builderbench_use_pd=bool(FLAGS.builderbench_use_pd),
        builderbench_pd_duration=int(FLAGS.builderbench_pd_duration),
        builderbench_permute_start_boxes=bool(
            FLAGS.builderbench_permute_start_boxes))

  def env_factory(factory_seed):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, factory_seed,
        fixed_start_end=fixed_start_end, **env_kwargs)
    return env

  def eval_env_factory(factory_seed):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, factory_seed,
        fixed_start_end=ppo_entry.fixed_goal_for_env(env_name), **env_kwargs)
    return env

  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index, seed,
      fixed_start_end=fixed_start_end, **env_kwargs)
  config.obs_dim = int(obs_dim)
  config.max_episode_steps = int(getattr(probe_env, '_step_limit')) + 1
  del probe_env

  network_factory = functools.partial(
      contrastive.make_networks,
      obs_dim=config.obs_dim,
      repr_dim=config.repr_dim,
      repr_norm=config.repr_norm,
      twin_q=False,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes)

  run_dir = os.path.join(
      config.log_dir, f'{config.alg_name}_{config.env_name}_{seed}')
  os.makedirs(run_dir, exist_ok=True)
  payload = {
      'entrypoint': 'baseline-agents/mpo-crl.py',
      'env': env_name,
      'seed': seed,
      'flags': {
          key: ppo_entry._json_safe(value)  # pylint: disable=protected-access
          for key, value in FLAGS.flag_values_dict().items()
      },
      'resolved_config': {
          key: ppo_entry._json_safe(value)  # pylint: disable=protected-access
          for key, value in config.__dict__.items()
      },
      'mpo_config': dataclasses.asdict(mpo_config),
      'fixed_start_end': ppo_entry._json_safe(  # pylint: disable=protected-access
          fixed_start_end),
      'env_defaults': ppo_entry._json_safe(  # pylint: disable=protected-access
          env_defaults),
  }
  with open(
      os.path.join(run_dir, 'run_config.json'), 'w', encoding='utf-8'
  ) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)

  from default import make_default_logger
  logger_fn = functools.partial(
      make_default_logger,
      save_dir=run_dir,
      add_uid=config.add_uid,
      steps_key='learner_steps')
  print(
      f'[mpo-crl] env={env_name} seed={seed} E={mpo_config.num_envs} '
      f'T={mpo_config.rollout_length} policy_updates='
      f'{mpo_config.policy_updates_per_iter} crl_updates='
      f'{mpo_config.crl_updates_per_iter} target_period='
      f'{mpo_config.target_update_period} repr_target_rate='
      f'{mpo_config.repr_target_update_rate} td_critic='
      f'{mpo_config.use_td_critic} critic_target_period='
      f'{mpo_config.critic_target_update_period} critic_target_rate='
      f'{mpo_config.critic_target_update_rate} reward_norm='
      f'{mpo_config.normalize_critic_reward} reward_norm_rate='
      f'{mpo_config.critic_reward_norm_rate} bootstrap_actions='
      f'{mpo_config.bootstrap_action_samples} run_dir={run_dir}')

  _bb_kwargs = None
  if env_name.startswith('builderbench_'):
    _bb_kwargs = dict(env_kwargs)
    if fixed_start_end is not None:
      _bb_kwargs['fixed_target_goal'] = np.asarray(
          fixed_start_end, dtype=np.float32)

  mpo_crl_learner.run_mpo_crl_training(
      config=config,
      mpo_config=mpo_config,
      env_factory=env_factory,
      eval_env_factory=eval_env_factory,
      network_factory=network_factory,
      logger_fn=logger_fn,
      total_steps=total_steps,
      seed=seed,
      checkpoint_dir=os.path.join(run_dir, 'checkpoints'),
      builderbench_kwargs=_bb_kwargs)


if __name__ == '__main__':
  app.run(main)
