"""Standalone PPO+RND entry point.

Sibling to `ppo_contrastive.py` (PPO-on-φ·ψ): same CLI conventions, same env
infra (including `--maniskill_native_vec`), same wandb/video/checkpoint
plumbing -- but no CRL critic/replay. Reward is sparse extrinsic (evaluator
success -- see `contrastive.utils.SuccessObserver` /
`DrawerClosedSuccessObserver`, NOT ManiSkill's own dense reward) plus RND
intrinsic exploration bonus. See `contrastive/ppo_rnd_learner.py` for the
algorithm.

Run with:
  python ppo_rnd.py \
      --env=maniskill_open_cabinet_drawer \
      --seed=0 \
      --num_steps=600000000 \
      --maniskill_native_vec \
      --log_dir_path=logs/ppo_rnd/
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports
import functools
import json
import os

from absl import app
from absl import flags

import contrastive
from contrastive import ppo_rnd_learner
from contrastive import utils as contrastive_utils
from ppo_env_defaults import fixed_goal_dict, PPO_ENV_DEFAULTS

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/ppo_rnd/', 'Where to log metrics')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('env', 'point_FourRooms', 'Environment type')
flags.DEFINE_integer('num_steps', 8_000_000, 'Total env steps', lower_bound=0)
flags.DEFINE_bool('sample_goals', False,
                  'Sample goal uniformly (else use the fixed goal dict)')
flags.DEFINE_integer('ppo_rollout_length', -1,
                     'If >=0, overrides the per-env rollout length default.')
flags.DEFINE_integer(
    'ppo_num_envs', -1,
    'If >=0, overrides ContrastiveConfig.ppo_num_envs (default 8). With '
    '--maniskill_native_vec, all envs are simulated in one GPU-batched '
    'ManiSkill scene, so this can be pushed well past the default without '
    'the per-env Python-loop overhead that would otherwise scale with it.')
flags.DEFINE_float(
    'ppo_discount', -1.0,
    'If >0, overrides the extrinsic-reward PPO discount for GAE/returns; '
    '<=0 falls back to config.discount.')
flags.DEFINE_float('ppo_clip_coef', -1.0, 'If >0, overrides PPO clip coefficient.')
flags.DEFINE_float('ppo_actor_min_std', -1.0, 'If >0, overrides PPO actor min std.')
flags.DEFINE_float(
    'ppo_ent_coef', -1.0,
    'If >=0, overrides PPO entropy bonus coefficient; '
    '<0 keeps ContrastiveConfig default.')
flags.DEFINE_bool(
    'ppo_anneal_lr', True,
    'If True, linearly decay PPO Adam learning rate to 0 over training; '
    'if False, use a fixed learning_rate.  Pass --noppo_anneal_lr to disable.')
flags.DEFINE_float(
    'rnd_int_coef', -1.0,
    'If >=0, overrides the weight on the intrinsic advantage in the '
    'combined PPO advantage (rnd_ext_coef*ext_adv + rnd_int_coef*int_adv).')
flags.DEFINE_float(
    'rnd_ext_coef', -1.0,
    'If >=0, overrides the weight on the extrinsic (sparse-success) '
    'advantage in the combined PPO advantage.')
flags.DEFINE_float(
    'rnd_int_discount', -1.0,
    'If >0, overrides the discount used for the intrinsic-reward GAE/return '
    'stream (independent of the extrinsic --ppo_discount).')
flags.DEFINE_integer(
    'rnd_output_size', -1,
    'If >0, overrides the RND predictor/target networks\' output width.')
flags.DEFINE_integer(
    'ppo_checkpoint_interval', -1,
    'If >=0, overrides ContrastiveConfig.ppo_checkpoint_interval (default '
    '500 iterations). Useful to lower for long/GPU-crash-prone runs so a '
    'mid-run crash loses less progress before the next `sbatch` resubmit '
    'auto-resumes from checkpoint_dir/latest.pkl.')
flags.DEFINE_bool(
    'maniskill_native_vec', False,
    'ManiSkill env_names only: collect PPO rollouts from one native '
    'num_envs=ppo_num_envs GPU-batched ManiSkill simulation instead of '
    'ppo_num_envs separate single-env sims stepped one at a time in a '
    'Python loop. Pure throughput optimization -- same rollouts, same '
    'learning dynamics. Requires a CUDA GPU.')
flags.DEFINE_bool(
    'bin_randomize_gripper_init', False,
    'SawyerBin: randomize the initial gripper TCP offset around the '
    'object at reset, instead of the fixed +0.03m-above-object default.')
flags.DEFINE_string(
    'hidden_layer_sizes', '256,256',
    'Comma-separated MLP widths for the policy/value/RND networks, '
    'e.g. "256,256,256,256,256,256". Overrides '
    'ContrastiveConfig.hidden_layer_sizes; more than 2 entries '
    'automatically switches contrastive/networks.py to its ResidualMLP path.')
flags.DEFINE_string(
    'wandb_project', '',
    'wandb project to log to. Empty string (default) disables wandb '
    'entirely -- no wandb.init() call is made.')
flags.DEFINE_string(
    'wandb_entity', '',
    'wandb entity/org. Empty string defers to the WANDB_ENTITY env var '
    '(wandb resolves this itself when entity=None).')
flags.DEFINE_string(
    'wandb_run_name', '',
    'Optional explicit wandb run name. Empty string auto-generates '
    '"{alg_name}_{env_name}_{seed}".')
flags.DEFINE_string(
    'wandb_group', '',
    'wandb run group, used to cluster e.g. the different --seed runs of '
    'the same experiment together in the UI. Empty string (default) '
    'auto-generates "{alg_name}".')
flags.DEFINE_bool(
    'render_video', True,
    'If True, render one rollout with the final policy at the end of '
    'training and save it as an mp4 under the run directory (and log it '
    'to wandb if --wandb_project is set). Best-effort: rendering failures '
    'are caught and logged as warnings, never fatal to a completed run.')
flags.DEFINE_integer('video_fps', 30, 'FPS for the end-of-training mp4.')
flags.DEFINE_integer(
    'video_max_steps', -1,
    'Override rollout length for the end-of-training video; -1 = use '
    'config.max_episode_steps.')
flags.DEFINE_bool(
    'video_stochastic', False,
    'If True, sample actions for the end-of-training video instead of '
    'using the policy mode.')
flags.DEFINE_integer(
    'video_every_steps', 0,
    'If > 0 (and --render_video), also render/log a rollout video every '
    'this many global steps during training (in addition to the '
    'end-of-training video). The mp4 is written to a single fixed path '
    'under the run directory and overwritten each time, so local disk '
    'usage does not grow with training length. 0 (default) disables '
    'periodic video logging.')


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
    except Exception:  # pragma: no cover
      pass
  return str(value)


def main(_):
  env_name = FLAGS.env
  seed = FLAGS.seed
  print(f'[ppo_rnd] env={env_name} seed={seed}')

  # ---- Build config ------------------------------------------------------
  params = dict(
      seed=seed,
      env_name=env_name,
      alg_name='ppo_rnd',
      reward_shaping_mode='ppo',
      max_number_of_steps=FLAGS.num_steps,
      log_dir=FLAGS.log_dir_path,
      add_uid=FLAGS.add_uid,
      fix_goals=not FLAGS.sample_goals,
  )
  config = contrastive.ContrastiveConfig(**params)
  config.hidden_layer_sizes = tuple(
      int(x) for x in FLAGS.hidden_layer_sizes.split(','))

  # ---- Per-env PPO defaults (CLI flags still override) -------------------
  env_defaults = PPO_ENV_DEFAULTS.get(env_name)
  if env_defaults is None:
    print(f'[ppo_rnd] WARNING: no PPO_ENV_DEFAULTS entry for {env_name!r}; '
          f'falling back to ContrastiveConfig defaults '
          f'(T={config.ppo_rollout_length}).')
  else:
    config.ppo_rollout_length = int(env_defaults['rollout_length'])
    if 'start_index' in env_defaults:
      config.start_index = int(env_defaults['start_index'])
    if 'end_index' in env_defaults:
      config.end_index = int(env_defaults['end_index'])

  if FLAGS.ppo_rollout_length >= 0:
    config.ppo_rollout_length = int(FLAGS.ppo_rollout_length)
  if FLAGS.ppo_num_envs >= 0:
    config.ppo_num_envs = int(FLAGS.ppo_num_envs)
  if FLAGS.ppo_discount > 0.0:
    config.ppo_discount = float(FLAGS.ppo_discount)
  if FLAGS.ppo_clip_coef > 0.0:
    config.ppo_clip_coef = float(FLAGS.ppo_clip_coef)
  if FLAGS.ppo_actor_min_std > 0.0:
    config.ppo_actor_min_std = float(FLAGS.ppo_actor_min_std)
  if FLAGS.ppo_ent_coef >= 0.0:
    config.ppo_ent_coef = float(FLAGS.ppo_ent_coef)
  config.ppo_anneal_lr = bool(FLAGS.ppo_anneal_lr)
  if FLAGS.rnd_int_coef >= 0.0:
    config.rnd_int_coef = float(FLAGS.rnd_int_coef)
  if FLAGS.rnd_ext_coef >= 0.0:
    config.rnd_ext_coef = float(FLAGS.rnd_ext_coef)
  if FLAGS.rnd_int_discount > 0.0:
    config.rnd_int_discount = float(FLAGS.rnd_int_discount)
  if FLAGS.rnd_output_size > 0:
    config.rnd_output_size = int(FLAGS.rnd_output_size)
  if FLAGS.ppo_checkpoint_interval >= 0:
    config.ppo_checkpoint_interval = int(FLAGS.ppo_checkpoint_interval)
  config.ppo_maniskill_native_vec = bool(FLAGS.maniskill_native_vec)

  print(f'[ppo_rnd] PPO knobs: '
        f'rollout_length={config.ppo_rollout_length}, '
        f'num_envs={config.ppo_num_envs}, '
        f'clip_coef={config.ppo_clip_coef}, '
        f'actor_min_std={config.ppo_actor_min_std}, '
        f'ent_coef={config.ppo_ent_coef}, '
        f'discount_ext={config.ppo_discount if config.ppo_discount > 0 else config.discount}, '
        f'rnd_int_coef={config.rnd_int_coef}, '
        f'rnd_ext_coef={config.rnd_ext_coef}, '
        f'rnd_int_discount={config.rnd_int_discount}, '
        f'rnd_output_size={config.rnd_output_size}, '
        f'norm_obs={config.ppo_norm_obs}, '
        f'ppo_anneal_lr={config.ppo_anneal_lr}')

  # ---- Env-specific extra kwargs (forwarded to env_utils.load) ----------
  _env_kwargs = {}
  if env_name == 'sawyer_bin' and FLAGS.bin_randomize_gripper_init:
    _env_kwargs['randomize_gripper_init'] = True
    print('[ppo_rnd] sawyer_bin init: randomized gripper position at reset')

  if config.ppo_maniskill_native_vec and not env_name.startswith('maniskill_'):
    print(f'[ppo_rnd] WARNING: --maniskill_native_vec has no effect on '
          f'env={env_name!r} (not a maniskill_* env); ignoring.')
    config.ppo_maniskill_native_vec = False

  # ---- Build env factories ----------------------------------------------
  fixed_start_end = (fixed_goal_dict[env_name]
                     if config.fix_goals else None)

  def env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=fixed_start_end, **_env_kwargs)
    return env

  def eval_env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=fixed_goal_dict[env_name], **_env_kwargs)
    return env

  # obs_dim / max_episode_steps: see `ppo_contrastive.py`'s matching comment
  # on why --maniskill_native_vec must not probe via a throwaway CPU env.
  if config.ppo_maniskill_native_vec:
    import env_utils as _env_utils
    obs_dim, _max_steps = _env_utils.maniskill_static_obs_info(env_name)
    config.obs_dim = obs_dim
    config.max_episode_steps = _max_steps + 1
  else:
    probe_env, obs_dim = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, seed,
        fixed_start_end=fixed_start_end, **_env_kwargs)
    config.obs_dim = obs_dim
    config.max_episode_steps = getattr(probe_env, '_step_limit') + 1
    del probe_env

  # ---- Network factory ----------------------------------------------------
  network_factory = functools.partial(
      contrastive.make_networks,
      obs_dim=obs_dim,
      repr_dim=config.repr_dim,
      repr_norm=config.repr_norm,
      twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
      actor_min_std=float(config.ppo_actor_min_std))

  # ---- Logger ------------------------------------------------------------
  run_dir = os.path.join(
      config.log_dir,
      f'{config.alg_name}_{config.env_name}_{seed}')
  os.makedirs(run_dir, exist_ok=True)
  run_config_path = os.path.join(run_dir, 'run_config.json')
  run_cfg_payload = {
      'entrypoint': 'ppo_rnd.py',
      'env': env_name,
      'seed': int(seed),
      'flags': {k: _json_safe(v) for k, v in FLAGS.flag_values_dict().items()},
      'resolved_config': {
          k: _json_safe(v) for k, v in config.__dict__.items()
      },
      'fixed_start_end': _json_safe(fixed_start_end),
      'ppo_env_defaults': _json_safe(env_defaults),
  }
  with open(run_config_path, 'w', encoding='utf-8') as fh:
    json.dump(run_cfg_payload, fh, indent=2, sort_keys=True)
  print(f'[ppo_rnd] wrote run config: {run_config_path}')

  # ---- Optional live wandb logging ---------------------------------------
  wandb_run = None
  if FLAGS.wandb_project:
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
    print(f'[ppo_rnd] wandb run: {wandb_run.url}')

  from default import make_default_logger
  logger_fn = functools.partial(
      make_default_logger,
      save_dir=run_dir,
      add_uid=config.add_uid,
      steps_key='learner_steps',
      wandb_run=wandb_run)

  # ---- Optional periodic mid-training rollout video -----------------------
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
            fixed_start_end=fixed_goal_dict[env_name],
            max_steps=(None if FLAGS.video_max_steps < 0
                      else FLAGS.video_max_steps),
            fps=FLAGS.video_fps,
            config=config,
            stochastic=FLAGS.video_stochastic,
            obs_normalizer=obs_normalizer_)
        if wandb_run is not None:
          wandb_run.log(
              {'video/periodic_rollout': wandb.Video(
                  video_path, fps=FLAGS.video_fps, format='mp4'),
               'video/periodic_rollout_global_step': global_step_},
              step=iteration_)
        print(f'[ppo_rnd] wrote periodic rollout video at '
              f'global_step={global_step_}: {video_path}')
      except Exception as exc:  # noqa: BLE001 -- never fail training over video
        print(f'[ppo_rnd] WARNING: periodic video rendering failed '
              f'(training is unaffected): {type(exc).__name__}: {exc}')

  # ---- Go ----------------------------------------------------------------
  checkpoint_dir = os.path.join(run_dir, 'checkpoints')
  final_state = ppo_rnd_learner.run_ppo_rnd_training(
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

  # ---- Optional end-of-training rollout video ----------------------------
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
          fixed_start_end=fixed_goal_dict[env_name],
          max_steps=(None if FLAGS.video_max_steps < 0
                    else FLAGS.video_max_steps),
          fps=FLAGS.video_fps,
          config=config,
          stochastic=FLAGS.video_stochastic,
          obs_normalizer=final_state.obs_normalizer)
      if wandb_run is not None:
        wandb_run.log({'video/final_rollout': wandb.Video(
            video_path, fps=FLAGS.video_fps, format='mp4')})
      print(f'[ppo_rnd] wrote end-of-training video: {video_path}')
    except Exception as exc:  # noqa: BLE001 -- never fail a completed run over video
      print(f'[ppo_rnd] WARNING: end-of-training video rendering failed '
            f'(training results are unaffected): '
            f'{type(exc).__name__}: {exc}')

  if wandb_run is not None:
    wandb_run.finish()


if __name__ == '__main__':
  app.run(main)
