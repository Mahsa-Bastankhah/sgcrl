"""Standalone SAC + contrastive CPC on Isaac Gym (no Launchpad).

PhysX must be created before JAX (Preview 4 GPU kernels).  Then a single
process runs ``ContrastiveLearner`` against ``EpisodeReplay`` and a native
batched AllegroKukaThrow env — same critic/actor as ``lp_contrastive.py
--alg=contrastive_cpc``, at E=1024.

  python sac_contrastive.py \
      --env=allegro_kuka_throw \
      --alg=contrastive_cpc \
      --sac_num_envs=1024 \
      --num_steps=300000000 \
      --log_dir_path=logs/sac_allegro_kuka_throw/
"""
from envs.isaacgym_physx_bootstrap import maybe_create_from_argv as _ig_boot
_ig_boot()

import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports
import functools
import json
import os

from absl import app
from absl import flags

import contrastive

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/sac/', 'Where to log metrics')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_bool('add_uid', False,
                  'Whether to add a unique id to the log directory name')
flags.DEFINE_string('env', 'allegro_kuka_throw', 'Environment type')
flags.DEFINE_string(
    'alg', 'contrastive_cpc',
    'Algorithm: contrastive_cpc (stock, actor max φ·ψ) or q_sac '
    '(twin Q on r=sg(φ·ψ), actor max min Q).')
flags.DEFINE_integer('num_steps', 8_000_000, 'Total env steps', lower_bound=0)
flags.DEFINE_integer(
    'sac_num_envs', 8,
    'Parallel Isaac Gym envs. Bootstrap reads this from argv before JAX.')
flags.DEFINE_integer(
    'batch_size', -1,
    'SGD batch size. <0 keeps ContrastiveConfig.batch_size (256).')
flags.DEFINE_integer(
    'num_sgd_steps_per_step', 1,
    'SGD updates inside one ContrastiveLearner.step(). Keep 1 for vec SAC; '
    'do not copy Launchpad\'s 64.')
flags.DEFINE_integer(
    'sac_updates_per_env_step', -1,
    'ContrastiveLearner.step() calls per PhysX step after warmup. '
    '<0 maps Launchpad SPI onto the vec env: '
    'round(E * samples_per_insert / (T * batch_size)).')
flags.DEFINE_float(
    'samples_per_insert', -1.0,
    'Only used when sac_updates_per_env_step<0 (LP SPI mapping). '
    '<0 keeps ContrastiveConfig.samples_per_insert (256).')
flags.DEFINE_integer(
    'min_replay_size', -1,
    '<0 keeps ContrastiveConfig.min_replay_size (10000).')
flags.DEFINE_integer(
    'max_replay_size', -1,
    '<0 keeps ContrastiveConfig.max_replay_size (1e6).')
flags.DEFINE_integer(
    'log_interval_sim_steps', -1,
    'Log every this many PhysX steps. <0 uses isaacgym_episode_length.')
flags.DEFINE_integer(
    'checkpoint_interval', 100,
    'Save ckpt_iter_*.pkl every this many log ticks (PPO-style). 0 disables.')
flags.DEFINE_integer(
    'checkpoint_keep_last', 0,
    'Keep this many milestone ckpts (0 = keep all).')
flags.DEFINE_integer(
    'checkpoint_replay_max', 0,
    'Serialize this many recent replay transitions into latest.pkl. 0 disables.')
flags.DEFINE_float(
    'discount', -1.0,
    'If >=0, overrides ContrastiveConfig.discount.')
flags.DEFINE_float(
    'tau', -1.0,
    'If >=0, overrides ContrastiveConfig.tau (target critic / Q).')
flags.DEFINE_float(
    'entropy_coefficient', -1.0,
    'Fixed SAC temperature α. <0 uses adaptive α (stock).')
flags.DEFINE_float(
    'target_entropy', 0.0,
    'Adaptive-α target entropy (stock CRL default 0.0).')
flags.DEFINE_bool(
    'use_random_actor', True,
    'Uniform[-1,1] actions until the first learner update.')
flags.DEFINE_bool(
    'repr_norm', False,
    'If True, L2-normalize critic φ and ψ before dot products.')
flags.DEFINE_float(
    'actor_min_std', 1e-6,
    'Tanh-Gaussian policy std floor (stock SAC / make_networks default).')
flags.DEFINE_string(
    'hidden_layer_sizes', '',
    'Comma-separated hidden widths. Empty keeps ContrastiveConfig default '
    '(256x6 ResidualMLP).')
flags.DEFINE_integer(
    'isaacgym_episode_length', 300,
    'Isaac Gym episode length in env steps.')
flags.DEFINE_string(
    'isaacgym_pipeline', 'gpu',
    'Isaac Gym PhysX pipeline: "gpu" or "cpu".')
flags.DEFINE_list(
    'isaacgym_fixed_target_xyz', ['0.5', '-0.3', '0.4'],
    'Fixed bucket/goal world-frame xyz.')
flags.DEFINE_bool(
    'isaacgym_randomize_init', True,
    'NVIDIA reset randomization of object pose/joints.')
flags.DEFINE_bool(
    'isaacgym_randomize_object_shape', True,
    'Mix object dimensions on reset. Off keeps a single default cube.')
flags.DEFINE_bool(
    'isaacgym_palm_goal', False,
    '6-D palm+object goal packing. Off = 49+3 throw packing.')
flags.DEFINE_list(
    'isaacgym_palm_goal_xyz', ['0.17', '0.08', '0.57'],
    'Commanded palm xyz when --isaacgym_palm_goal is on.')
flags.DEFINE_bool(
    'isaacgym_table_push', False,
    'Easy on-desk push: object goal 10 cm toward the robot on the table '
    '(0, 0.10, 0.555). Parks the throw bucket off the desk. Use with '
    '--noisaacgym_randomize_init so spawn is not already in the success ball.')


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


def _apply_alg(config, alg: str):
  alg = str(alg).strip().lower()
  if alg == 'contrastive_cpc':
    config.use_cpc = True
    config.use_td = False
    config.use_q_repr = False
    config.use_kappa = False
    config.reward_shaping_mode = ''
  elif alg == 'q_sac':
    config.use_cpc = True
    config.use_q_repr = True
    config.twin_q_repr = True
    config.reward_shaping_mode = 'q'
    config.q_actor_her_aux_coef = 0
  else:
    raise ValueError(
        f'Unknown --alg={alg!r} (want contrastive_cpc or q_sac)')
  return alg


def main(_):
  env_name = FLAGS.env
  seed = int(FLAGS.seed)
  print(f'[sac_contrastive] env={env_name} seed={seed} alg={FLAGS.alg}')

  if not str(env_name).startswith('allegro_kuka'):
    raise ValueError(
        'sac_contrastive.py currently supports allegro_kuka_* only '
        f'(got {env_name!r})')

  params = dict(
      seed=seed,
      env_name=env_name,
      alg_name=str(FLAGS.alg).strip().lower(),
      reward_shaping_mode='',
      use_cpc=True,
      max_number_of_steps=int(FLAGS.num_steps),
      log_dir=FLAGS.log_dir_path,
      add_uid=bool(FLAGS.add_uid),
      ppo_num_envs=int(FLAGS.sac_num_envs),
      num_sgd_steps_per_step=int(FLAGS.num_sgd_steps_per_step),
      use_random_actor=bool(FLAGS.use_random_actor),
      target_entropy=float(FLAGS.target_entropy),
      repr_norm=bool(FLAGS.repr_norm),
  )
  config = contrastive.ContrastiveConfig(**params)
  _apply_alg(config, FLAGS.alg)

  if int(FLAGS.batch_size) > 0:
    config.batch_size = int(FLAGS.batch_size)
  if float(FLAGS.samples_per_insert) >= 0:
    config.samples_per_insert = float(FLAGS.samples_per_insert)
  if int(FLAGS.min_replay_size) >= 0:
    config.min_replay_size = int(FLAGS.min_replay_size)
  if int(FLAGS.max_replay_size) >= 0:
    config.max_replay_size = int(FLAGS.max_replay_size)
  if float(FLAGS.discount) >= 0:
    config.discount = float(FLAGS.discount)
  if float(FLAGS.tau) >= 0:
    config.tau = float(FLAGS.tau)
  if float(FLAGS.entropy_coefficient) >= 0:
    config.entropy_coefficient = float(FLAGS.entropy_coefficient)
  if FLAGS.hidden_layer_sizes.strip():
    config.hidden_layer_sizes = tuple(
        int(x) for x in FLAGS.hidden_layer_sizes.split(',') if x.strip())

  config.ppo_checkpoint_interval = int(FLAGS.checkpoint_interval)
  config.ppo_checkpoint_keep_last = int(FLAGS.checkpoint_keep_last)
  config.ppo_checkpoint_replay_max = int(FLAGS.checkpoint_replay_max)

  from envs.allegro_kuka_throw_env import GOAL_DIM as _IG_GOAL_THROW
  from envs.allegro_kuka_throw_env import GOAL_DIM_PALM as _IG_GOAL_PALM
  from envs.allegro_kuka_throw_env import STATE_DIM as _IG_STATE_THROW
  from envs.allegro_kuka_throw_env import STATE_DIM_PALM as _IG_STATE_PALM
  palm_goal = bool(FLAGS.isaacgym_palm_goal)
  state_dim = _IG_STATE_PALM if palm_goal else _IG_STATE_THROW
  goal_dim = _IG_GOAL_PALM if palm_goal else _IG_GOAL_THROW
  config.obs_dim = int(state_dim)
  config.goal_dim = int(goal_dim)
  config.max_episode_steps = int(FLAGS.isaacgym_episode_length)
  config.start_index = int(state_dim - goal_dim)
  config.end_index = int(state_dim)
  print(f'[sac_contrastive] isaacgym: palm_goal={palm_goal} '
        f'obs_dim={config.obs_dim} goal_dim={config.goal_dim} '
        f'goal_slice=state[{config.start_index}:{config.end_index}] '
        f'max_episode_steps={config.max_episode_steps}')

  if int(config.num_sgd_steps_per_step) != 1:
    print('[sac_contrastive] WARNING: num_sgd_steps_per_step='
          f'{config.num_sgd_steps_per_step} (LP uses 64; vec SAC wants 1)')

  from contrastive.sac_learner_isaacgym import _default_updates_per_env_step
  if int(FLAGS.sac_updates_per_env_step) > 0:
    n_updates = int(FLAGS.sac_updates_per_env_step)
  else:
    n_updates = _default_updates_per_env_step(
        num_envs=int(config.ppo_num_envs),
        max_episode_steps=int(config.max_episode_steps),
        batch_size=int(config.batch_size),
        samples_per_insert=float(config.samples_per_insert))
  log_every = int(FLAGS.log_interval_sim_steps)
  if log_every <= 0:
    log_every = int(config.max_episode_steps)

  actor_min_std = float(FLAGS.actor_min_std)
  network_factory = functools.partial(
      contrastive.make_networks,
      obs_dim=config.obs_dim,
      repr_dim=config.repr_dim,
      repr_norm=config.repr_norm,
      twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
      actor_min_std=actor_min_std,
  )

  run_dir = os.path.join(
      config.log_dir,
      f'{config.alg_name}_{config.env_name}_{seed}')
  os.makedirs(run_dir, exist_ok=True)
  run_config_path = os.path.join(run_dir, 'run_config.json')
  payload = {
      'entrypoint': 'sac_contrastive.py',
      'env': env_name,
      'seed': int(seed),
      'flags': {k: _json_safe(v) for k, v in FLAGS.flag_values_dict().items()},
      'resolved_config': {
          k: _json_safe(v) for k, v in config.__dict__.items()
      },
      'sac_updates_per_env_step': int(n_updates),
      'log_interval_sim_steps': int(log_every),
  }
  with open(run_config_path, 'w', encoding='utf-8') as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
  print(f'[sac_contrastive] wrote run config: {run_config_path}')
  print(f'[sac_contrastive] SAC knobs: alg={config.alg_name} '
        f'E={config.ppo_num_envs} T={config.max_episode_steps} '
        f'batch={config.batch_size} n_sgd={config.num_sgd_steps_per_step} '
        f'updates/step={n_updates} SPI={config.samples_per_insert} '
        f'min_replay={config.min_replay_size} '
        f'max_replay={config.max_replay_size} '
        f'discount={config.discount} tau={config.tau} '
        f'repr_norm={config.repr_norm} '
        f'entropy_coefficient={config.entropy_coefficient} '
        f'target_entropy={config.target_entropy} '
        f'hidden={config.hidden_layer_sizes} '
        f'actor_min_std={actor_min_std} '
        f'shaping={config.reward_shaping_mode!r}')

  from default import make_default_logger
  logger_fn = functools.partial(
      make_default_logger,
      save_dir=run_dir,
      add_uid=config.add_uid,
      steps_key='learner_steps',
      time_delta=0.0)

  from contrastive import sac_learner_isaacgym
  ig_kwargs = {
      'isaacgym_episode_length': int(FLAGS.isaacgym_episode_length),
      'isaacgym_fixed_target_xyz': tuple(
          float(v) for v in FLAGS.isaacgym_fixed_target_xyz),
      'isaacgym_pipeline': str(FLAGS.isaacgym_pipeline).strip().lower(),
      'isaacgym_randomize_init': bool(FLAGS.isaacgym_randomize_init),
      'isaacgym_randomize_object_shape': bool(
          FLAGS.isaacgym_randomize_object_shape),
      'isaacgym_palm_goal': bool(FLAGS.isaacgym_palm_goal),
      'isaacgym_palm_goal_xyz': tuple(
          float(v) for v in FLAGS.isaacgym_palm_goal_xyz),
      'isaacgym_table_push': bool(FLAGS.isaacgym_table_push),
  }
  sac_learner_isaacgym.run_sac_training(
      config=config,
      network_factory=network_factory,
      logger_fn=logger_fn,
      total_steps=int(FLAGS.num_steps),
      seed=seed,
      checkpoint_dir=os.path.join(run_dir, 'checkpoints'),
      isaacgym_kwargs=ig_kwargs,
      updates_per_env_step=n_updates,
      log_interval_sim_steps=log_every,
      actor_min_std=actor_min_std,
  )


if __name__ == '__main__':
  app.run(main)
