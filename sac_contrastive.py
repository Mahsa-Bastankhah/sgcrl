"""Standalone GPU-batched SAC + Contrastive RL (SGCRL) entry point.

Run with:
  python sac_contrastive.py \
      --env=maniskill_close_subtask_train \
      --seed=0 \
      --num_steps=70000000 \
      --hidden_layer_sizes="256,256,256,256,256,256" \
      --ppo_num_envs=64 \
      --ppo_crl_steps_per_iter=256 \
      --maniskill_native_vec \
      --uniform_sampling \
      --log_dir_path=maniskill_close_subtask_train_sac/
"""
import sgcrl_jax_acme_compat  # noqa: F401 -- must precede all acme/jax imports
import functools
import json
import os

from absl import app
from absl import flags

import contrastive
from contrastive import sac_crl_learner
from contrastive import utils as contrastive_utils
from ppo_env_defaults import fixed_goal_dict, PPO_ENV_DEFAULTS

FLAGS = flags.FLAGS

SCRATCH_LOG_ROOT = '/network/scratch/m/mohammad-sami-nur.islam/sgcrl_logs'


def _resolve_log_dir(log_dir_path: str) -> str:
  """Resolves relative log paths to SCRATCH_LOG_ROOT."""
  if os.path.isabs(log_dir_path):
    return log_dir_path
  return os.path.join(SCRATCH_LOG_ROOT, log_dir_path)


flags.DEFINE_string('log_dir_path', 'logs/sac_crl/', 'Where to log metrics')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('env', 'maniskill_close_subtask_train', 'Environment type')
flags.DEFINE_integer('num_steps', 70_000_000, 'Total env steps', lower_bound=0)
flags.DEFINE_bool('sample_goals', False,
                  'Sample goal uniformly (else use fixed goal dict)')
flags.DEFINE_bool(
    'repr_norm', False,
    'If True, L2-normalize critic phi and psi before dot products.')
flags.DEFINE_integer('ppo_rollout_length', -1,
                     'If >=0, overrides the per-env rollout length default.')
flags.DEFINE_integer('ppo_crl_steps_per_iter', -1,
                     'If >=0, overrides the per-env CRL-steps default.')
flags.DEFINE_integer(
    'ppo_num_envs', -1,
    'If >=0, overrides ContrastiveConfig.ppo_num_envs (default 64).')
flags.DEFINE_float(
    'discount', -1.0,
    'If >=0, overrides ContrastiveConfig.discount (CRL discount).')
flags.DEFINE_float('actor_learning_rate', 3e-4, 'Adam LR for SAC actor.')
flags.DEFINE_float('learning_rate', 3e-4, 'Adam LR for CRL critic & alpha.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for CRL & actor updates.')
flags.DEFINE_float('entropy_coefficient', -1.0,
                   'Fixed entropy coefficient alpha; negative uses adaptive alpha.')
flags.DEFINE_float('target_entropy', 0.0,
                   'Target entropy for adaptive temperature alpha.')
flags.DEFINE_bool(
    'uniform_sampling', False,
    'If True, mix uniformly sampled goals into each CRL replay batch.')
flags.DEFINE_integer(
    'ppo_checkpoint_interval', -1,
    'If >=0, overrides checkpoint interval (iterations).')
flags.DEFINE_bool(
    'maniskill_native_vec', False,
    'ManiSkill envs only: collect rollouts from native GPU-batched ManiSkill simulation.')
flags.DEFINE_string(
    'hidden_layer_sizes', '256,256,256,256,256,256',
    'Comma-separated MLP widths for policy & critic networks.')
flags.DEFINE_string(
    'wandb_project', '',
    'WandB project to log to. Empty string disables wandb.')
flags.DEFINE_string(
    'wandb_entity', '',
    'WandB entity/org.')
flags.DEFINE_string(
    'wandb_run_name', '',
    'Optional explicit WandB run name.')
flags.DEFINE_string(
    'wandb_group', '',
    'WandB run group.')
flags.DEFINE_bool(
    'render_video', True,
    'If True, render rollout video at the end of training.')
flags.DEFINE_integer('video_fps', 30, 'FPS for rollout mp4.')
flags.DEFINE_integer(
    'video_max_steps', -1,
    'Override rollout length for video; -1 uses max_episode_steps.')
flags.DEFINE_bool(
    'video_stochastic', False,
    'If True, sample actions for video instead of deterministic mode.')
flags.DEFINE_integer(
    'video_every_steps', 0,
    'If > 0, render rollout video periodically during training.')


def _json_safe(value):
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if hasattr(value, 'tolist'):
    try:
      return value.tolist()
    except Exception:
      pass
  return str(value)


def main(_):
  env_name = FLAGS.env
  seed = FLAGS.seed
  print(f'[sac_contrastive] env={env_name} seed={seed}')

  # 1. Build Config
  params = dict(
      seed=seed,
      env_name=env_name,
      alg_name='sac_crl',
      reward_shaping_mode='sac_crl',
      use_cpc=True,
      max_number_of_steps=FLAGS.num_steps,
      log_dir=_resolve_log_dir(FLAGS.log_dir_path),
      add_uid=FLAGS.add_uid,
      fix_goals=not FLAGS.sample_goals,
      batch_size=FLAGS.batch_size,
      actor_learning_rate=FLAGS.actor_learning_rate,
      learning_rate=FLAGS.learning_rate,
      target_entropy=FLAGS.target_entropy,
  )
  if FLAGS.entropy_coefficient >= 0.0:
    params['entropy_coefficient'] = float(FLAGS.entropy_coefficient)
  else:
    params['entropy_coefficient'] = None

  config = contrastive.ContrastiveConfig(**params)
  config.repr_norm = bool(FLAGS.repr_norm)
  config.hidden_layer_sizes = tuple(
      int(x) for x in FLAGS.hidden_layer_sizes.split(',')
  )

  # 2. Per-env defaults
  env_defaults = PPO_ENV_DEFAULTS.get(env_name, {})
  if 'rollout_length' in env_defaults:
    config.ppo_rollout_length = int(env_defaults['rollout_length'])
  if 'crl_steps_per_iter' in env_defaults:
    config.ppo_crl_steps_per_iter = int(env_defaults['crl_steps_per_iter'])
  if 'start_index' in env_defaults:
    config.start_index = int(env_defaults['start_index'])
  if 'end_index' in env_defaults:
    config.end_index = int(env_defaults['end_index'])

  if FLAGS.ppo_rollout_length >= 0:
    config.ppo_rollout_length = int(FLAGS.ppo_rollout_length)
  if FLAGS.ppo_crl_steps_per_iter >= 0:
    config.ppo_crl_steps_per_iter = int(FLAGS.ppo_crl_steps_per_iter)
  if FLAGS.ppo_num_envs >= 0:
    config.ppo_num_envs = int(FLAGS.ppo_num_envs)
  if FLAGS.discount >= 0.0:
    config.discount = float(FLAGS.discount)
  config.uniform_sampling = bool(FLAGS.uniform_sampling)
  if FLAGS.ppo_checkpoint_interval >= 0:
    config.ppo_checkpoint_interval = int(FLAGS.ppo_checkpoint_interval)
  config.ppo_maniskill_native_vec = bool(FLAGS.maniskill_native_vec)

  if config.ppo_maniskill_native_vec and not env_name.startswith('maniskill_'):
    print(f'[sac_contrastive] WARNING: --maniskill_native_vec ignored for {env_name!r}.')
    config.ppo_maniskill_native_vec = False

  # 3. Environment Factories
  fixed_start_end = (
      fixed_goal_dict[env_name] if config.fix_goals and env_name in fixed_goal_dict else None
  )

  def env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=fixed_start_end,
    )
    return env

  def eval_env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=(fixed_goal_dict[env_name] if env_name in fixed_goal_dict else None),
    )
    return env

  if config.ppo_maniskill_native_vec:
    import env_utils as _env_utils
    obs_dim, _max_steps = _env_utils.maniskill_static_obs_info(env_name)
    config.obs_dim = obs_dim
    config.max_episode_steps = _max_steps + 1
  else:
    probe_env, obs_dim = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, seed,
        fixed_start_end=fixed_start_end,
    )
    config.obs_dim = obs_dim
    config.max_episode_steps = getattr(probe_env, '_step_limit') + 1
    del probe_env

  # 4. Network Factory
  network_factory = functools.partial(
      contrastive.make_networks,
      obs_dim=obs_dim,
      repr_dim=config.repr_dim,
      repr_norm=config.repr_norm,
      twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
  )

  # 5. Logger & WandB Setup
  run_dir = os.path.join(
      config.log_dir,
      f'{config.alg_name}_{config.env_name}_{seed}'
  )
  os.makedirs(run_dir, exist_ok=True)
  run_config_path = os.path.join(run_dir, 'run_config.json')
  with open(run_config_path, 'w', encoding='utf-8') as fh:
    json.dump({
        'entrypoint': 'sac_contrastive.py',
        'env': env_name,
        'seed': int(seed),
        'flags': {k: _json_safe(v) for k, v in FLAGS.flag_values_dict().items()},
        'resolved_config': {k: _json_safe(v) for k, v in config.__dict__.items()},
    }, fh, indent=2, sort_keys=True)

  wandb_run = None
  if FLAGS.wandb_project:
    if not os.environ.get('WANDB_DIR'):
      os.environ['WANDB_DIR'] = '/network/scratch/m/mohammad-sami-nur.islam/wandb_cache'
    import wandb
    wandb_run_id_path = os.path.join(run_dir, 'wandb_run_id.txt')
    wandb_run_id = None
    if os.path.exists(wandb_run_id_path):
      with open(wandb_run_id_path, 'r', encoding='utf-8') as fh:
        wandb_run_id = fh.read().strip() or None
    wandb_run = wandb.init(
        project=FLAGS.wandb_project,
        entity=(FLAGS.wandb_entity or None),
        name=(FLAGS.wandb_run_name or f'{config.alg_name}_{env_name}_{seed}'),
        group=(FLAGS.wandb_group or config.alg_name),
        job_type='train',
        config={k: _json_safe(v) for k, v in config.__dict__.items()},
        dir=run_dir,
        id=wandb_run_id,
        resume='allow' if wandb_run_id else None,
    )
    with open(wandb_run_id_path, 'w', encoding='utf-8') as fh:
      fh.write(wandb_run.id)
    print(f'[sac_contrastive] WandB run URL: {wandb_run.url}')

  from default import make_default_logger
  logger_fn = functools.partial(
      make_default_logger,
      save_dir=run_dir,
      add_uid=config.add_uid,
      steps_key='learner_steps',
      wandb_run=wandb_run,
  )

  # 6. Periodic Video Hook
  video_fn = None
  if FLAGS.render_video and FLAGS.video_every_steps > 0:
    def video_fn(policy_params, obs_normalizer_, global_step_, iteration_):
      try:
        import ppo_video_utils
        video_path = os.path.join(run_dir, 'rollout_periodic.mp4')
        ppo_video_utils.record_rollout_video(
            env_name=env_name,
            seed=seed,
            policy_params=policy_params,
            hidden_layer_sizes=config.hidden_layer_sizes,
            output_path=video_path,
            fixed_start_end=(fixed_goal_dict[env_name] if env_name in fixed_goal_dict else None),
            max_steps=(None if FLAGS.video_max_steps < 0 else FLAGS.video_max_steps),
            fps=FLAGS.video_fps,
            config=config,
            stochastic=FLAGS.video_stochastic,
            obs_normalizer=None,
        )
        if wandb_run is not None:
          wandb_run.log(
              {
                  'video/periodic_rollout': wandb.Video(
                      video_path, fps=FLAGS.video_fps, format='mp4'
                  ),
                  'video/periodic_rollout_global_step': global_step_,
              },
              step=iteration_,
          )
        print(f'[sac_contrastive] Periodic video recorded at step={global_step_}: {video_path}')
      except Exception as exc:
        print(f'[sac_contrastive] WARNING: periodic video failed: {type(exc).__name__}: {exc}')

  # 7. Start Training
  checkpoint_dir = os.path.join(run_dir, 'checkpoints')
  final_state = sac_crl_learner.run_sac_crl_training(
      config=config,
      env_factory=env_factory,
      eval_env_factory=eval_env_factory,
      network_factory=network_factory,
      logger_fn=logger_fn,
      total_steps=FLAGS.num_steps,
      seed=seed,
      video_fn=video_fn,
      video_every_steps=FLAGS.video_every_steps,
      checkpoint_dir=checkpoint_dir,
  )

  # 8. End-of-Training Video
  if FLAGS.render_video:
    try:
      import ppo_video_utils
      video_path = os.path.join(run_dir, 'rollout_final.mp4')
      ppo_video_utils.record_rollout_video(
          env_name=env_name,
          seed=seed,
          policy_params=final_state.policy_params,
          hidden_layer_sizes=config.hidden_layer_sizes,
          output_path=video_path,
          fixed_start_end=(fixed_goal_dict[env_name] if env_name in fixed_goal_dict else None),
          max_steps=(None if FLAGS.video_max_steps < 0 else FLAGS.video_max_steps),
          fps=FLAGS.video_fps,
          config=config,
          stochastic=FLAGS.video_stochastic,
          obs_normalizer=None,
      )
      if wandb_run is not None:
        wandb_run.log({
            'video/final_rollout': wandb.Video(
                video_path, fps=FLAGS.video_fps, format='mp4'
            )
        })
      print(f'[sac_contrastive] Final rollout video saved: {video_path}')
    except Exception as exc:
      print(f'[sac_contrastive] WARNING: final video failed: {type(exc).__name__}: {exc}')

  if wandb_run is not None:
    wandb_run.finish()


if __name__ == '__main__':
  app.run(main)
