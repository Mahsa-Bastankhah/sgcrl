"""Weighted Behavioural Cloning on φ·ψ — entry point.

Replaces the PPO update with a weighted maximum-likelihood objective:

    max_π  E_{(s,a,g) ~ π_old} [  φ(s,a)·ψ(g)  ·  log π(a|s)  ]

CRL networks (φ, ψ) are trained off-policy from the same replay buffer.

Run with:
  python wbc_contrastive.py \\
      --env=point_EightRooms \\
      --seed=0 \\
      --num_steps=10000000 \\
      --log_dir_path=logs/wbc_eightrooms/
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports
import functools
import json
import os

from absl import app
from absl import flags
import numpy as np

import contrastive
from contrastive import wbc_learner
from contrastive import utils as contrastive_utils


# ---------------------------------------------------------------------------
# Shared helpers (duplicated here to avoid importing ppo_contrastive, which
# would re-register its absl flags and cause DuplicateFlagError).
# ---------------------------------------------------------------------------
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


fixed_goal_dict = {
    'point_Spiral7x7':   [np.array([3, 3], dtype=float),
                          np.array([6, 6], dtype=float)],
    'point_Spiral9x9':   [np.array([5, 5], dtype=float),
                          np.array([8, 8], dtype=float)],
    'point_Spiral11x11': [np.array([5, 5], dtype=float),
                          np.array([10, 10], dtype=float)],
    'point_FourRooms':   [np.array([0, 0], dtype=float),
                          np.array([2, 8],  dtype=float)],
    'point_EightRooms':  [np.array([0, 0],   dtype=float),
                          np.array([10, 20], dtype=float)],
    'point_SixteenRooms': [np.array([0, 0],   dtype=float),
                           np.array([20, 20], dtype=float)],
    'point_SixteenRooms4D': [np.array([0, 0, 0],    dtype=float),
                              np.array([20, 20, 0], dtype=float)],
    'point_SixteenRoomsActual4D': [np.array([0, 0, 0, 0],    dtype=float),
                                   np.array([20, 20, 0, 0], dtype=float)],
    'point_Impossible':  [np.array([0, 0], dtype=float),
                          np.array([6, 8], dtype=float)],
    'point_Maze11x11':   [np.array([0, 0], dtype=float),
                          np.array([0, 10], dtype=float)],
    'point_Wall11x11':   [np.array([0, 0], dtype=float),
                          np.array([10, 10], dtype=float)],
    'sawyer_bin':  np.array([0.12, 0.7, 0.02]),
    'sawyer_box':  np.array([0.0, 0.75, 0.133]),
    'sawyer_peg':  np.array([-0.3, 0.6, 0.0]),
    'riverswim':   np.array([0., 0., 0., 0., 0., 1.], dtype=float),
}

PPO_ENV_DEFAULTS = {
    'point_FourRooms':            dict(rollout_length=128,  crl_steps_per_iter=64),
    'point_EightRooms':           dict(rollout_length=256,  crl_steps_per_iter=128),
    'point_SixteenRooms':         dict(rollout_length=512,  crl_steps_per_iter=256),
    'point_SixteenRooms4D':       dict(rollout_length=512,  crl_steps_per_iter=256),
    'point_SixteenRoomsActual4D': dict(rollout_length=512,  crl_steps_per_iter=256),
    'point_Spiral7x7':            dict(rollout_length=128,  crl_steps_per_iter=64),
    'point_Spiral9x9':            dict(rollout_length=128,  crl_steps_per_iter=64),
    'point_Spiral11x11':          dict(rollout_length=128,  crl_steps_per_iter=64),
    'point_Maze11x11':            dict(rollout_length=128,  crl_steps_per_iter=64),
    'point_Wall11x11':            dict(rollout_length=128,  crl_steps_per_iter=64),
    'point_Impossible':           dict(rollout_length=128,  crl_steps_per_iter=64),
    'riverswim':                  dict(rollout_length=128,  crl_steps_per_iter=64),
    'sawyer_bin':                 dict(rollout_length=256,  crl_steps_per_iter=10),
    'sawyer_box':                 dict(rollout_length=256,  crl_steps_per_iter=10),
    'sawyer_peg':                 dict(rollout_length=256,  crl_steps_per_iter=10),
}

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/wbc/', 'Where to log metrics.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('add_uid', False, 'Add a UID suffix to the log dir.')
flags.DEFINE_string('env', 'point_EightRooms', 'Environment name.')
flags.DEFINE_integer('num_steps', 10_000_000, 'Total env steps.')
flags.DEFINE_bool('sample_goals', False,
                  'Sample goals uniformly (else use fixed_goal_dict).')
flags.DEFINE_bool('repr_norm', False,
                  'L2-normalise CRL representations φ and ψ.')

# Rollout / CRL budget (same semantics as ppo_contrastive)
flags.DEFINE_integer('ppo_rollout_length', -1,
                     'Rollout horizon T.  -1 uses PPO_ENV_DEFAULTS.')
flags.DEFINE_integer('ppo_crl_steps_per_iter', -1,
                     'Off-policy CRL steps per iteration.  -1 uses defaults.')
flags.DEFINE_integer('ppo_num_epochs', -1,
                     'WBC policy update epochs per rollout.  -1 uses config default (10).')

# Optimiser
flags.DEFINE_float('ppo_actor_min_std', -1.0,
                   'Policy minimum std.  -1 uses config default.')
flags.DEFINE_float('ppo_ent_coef', -1.0,
                   'Entropy bonus coefficient.  -1 uses config default.')
flags.DEFINE_bool('ppo_anneal_lr', True,
                  'Linearly decay Adam LR to 0.')
flags.DEFINE_float('discount', -1.0,
                   'CRL discount.  -1 uses config default.')

# WBC-specific
flags.DEFINE_float(
    'wbc_weight_offset', 0.0,
    'Additive shift applied to normalised weights after z-scoring.  '
    '0 = centred (negative weights push policy away from bad actions).  '
    '>0 biases towards imitating all actions.')
flags.DEFINE_float(
    'wbc_kl_coef', 0.0,
    'Coefficient for KL trust-region penalty: coef * mean(log π_new - log π_old).  '
    '0 = disabled.  Positive values penalise the policy drifting far from '
    'the rollout policy, preventing tanh-saturation collapse.')
flags.DEFINE_float(
    'wbc_loc_clip', 0.0,
    'WBC only: clamp pre-tanh Gaussian loc (policy mean μ) to '
    '[-wbc_loc_clip, wbc_loc_clip] during rollout and policy updates.  '
    '0 = disabled.  e.g. 4.0 keeps tanh(loc) away from saturation rails.')

# Misc
flags.DEFINE_bool('uniform_sampling', False,
                  'Mix uniform negatives into CRL replay batches.')
flags.DEFINE_integer('ppo_checkpoint_interval', -1,
                     'Save checkpoint every N iters.  <0 uses config default.')
flags.DEFINE_integer('ppo_checkpoint_keep_last', -1,
                     'Max milestone checkpoints to retain.  0 = all.')
flags.DEFINE_string('hidden_layer_sizes', '',
                    'Comma-separated hidden widths, e.g. "256,256,256,256,256,256".')


def main(_):
    env_name = FLAGS.env
    seed = FLAGS.seed
    print(f'[wbc_contrastive] env={env_name}  seed={seed}')

    # ---- Config ------------------------------------------------------------
    params = dict(
        seed=seed,
        env_name=env_name,
        alg_name='wbc',
        reward_shaping_mode='ppo',   # reuse the same CRL+policy network setup
        use_cpc=True,
        max_number_of_steps=FLAGS.num_steps,
        log_dir=FLAGS.log_dir_path,
        add_uid=FLAGS.add_uid,
        fix_goals=not FLAGS.sample_goals,
    )
    config = contrastive.ContrastiveConfig(**params)
    config.repr_norm = bool(FLAGS.repr_norm)

    # ---- Per-env defaults --------------------------------------------------
    env_defaults = PPO_ENV_DEFAULTS.get(env_name)
    if env_defaults is None:
        print(f'[wbc_contrastive] WARNING: no PPO_ENV_DEFAULTS entry for '
              f'{env_name!r}; using ContrastiveConfig defaults.')
    else:
        config.ppo_rollout_length = int(env_defaults['rollout_length'])
        config.ppo_crl_steps_per_iter = int(env_defaults['crl_steps_per_iter'])

    if FLAGS.ppo_rollout_length >= 0:
        config.ppo_rollout_length = int(FLAGS.ppo_rollout_length)
    if FLAGS.ppo_crl_steps_per_iter >= 0:
        config.ppo_crl_steps_per_iter = int(FLAGS.ppo_crl_steps_per_iter)
    if FLAGS.ppo_num_epochs >= 0:
        config.ppo_num_epochs = int(FLAGS.ppo_num_epochs)
    if FLAGS.discount >= 0.0:
        config.discount = float(FLAGS.discount)
    if FLAGS.ppo_actor_min_std > 0.0:
        config.ppo_actor_min_std = float(FLAGS.ppo_actor_min_std)
    if FLAGS.ppo_ent_coef >= 0.0:
        config.ppo_ent_coef = float(FLAGS.ppo_ent_coef)
    config.ppo_anneal_lr = bool(FLAGS.ppo_anneal_lr)
    config.uniform_sampling = bool(FLAGS.uniform_sampling)
    config.wbc_weight_offset = float(FLAGS.wbc_weight_offset)
    config.wbc_kl_coef = float(FLAGS.wbc_kl_coef)
    config.wbc_loc_clip = float(FLAGS.wbc_loc_clip)
    if FLAGS.ppo_checkpoint_interval >= 0:
        config.ppo_checkpoint_interval = int(FLAGS.ppo_checkpoint_interval)
    if FLAGS.ppo_checkpoint_keep_last >= 0:
        config.ppo_checkpoint_keep_last = int(FLAGS.ppo_checkpoint_keep_last)
    if FLAGS.hidden_layer_sizes.strip():
        config.hidden_layer_sizes = tuple(
            int(x) for x in FLAGS.hidden_layer_sizes.split(',') if x.strip())

    print(f'[wbc_contrastive] knobs: '
          f'T={config.ppo_rollout_length}  '
          f'epochs={config.ppo_num_epochs}  '
          f'crl_steps={config.ppo_crl_steps_per_iter}  '
          f'num_envs={config.ppo_num_envs}  '
          f'ent_coef={config.ppo_ent_coef}  '
          f'actor_min_std={config.ppo_actor_min_std}  '
          f'discount={config.discount}  '
          f'anneal_lr={config.ppo_anneal_lr}  '
          f'wbc_weight_offset={config.wbc_weight_offset}  '
          f'wbc_kl_coef={config.wbc_kl_coef}  '
          f'wbc_loc_clip={config.wbc_loc_clip}  '
          f'hidden_layers={config.hidden_layer_sizes}')

    # ---- Env factories -----------------------------------------------------
    fixed_start_end = (fixed_goal_dict[env_name]
                       if config.fix_goals else None)

    def env_factory(s):
        env, _ = contrastive_utils.make_environment(
            env_name, config.start_index, config.end_index, s,
            fixed_start_end=fixed_start_end)
        return env

    def eval_env_factory(s):
        env, _ = contrastive_utils.make_environment(
            env_name, config.start_index, config.end_index, s,
            fixed_start_end=fixed_goal_dict[env_name])
        return env

    probe_env, obs_dim = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, seed,
        fixed_start_end=fixed_start_end)
    config.obs_dim = obs_dim
    config.max_episode_steps = getattr(probe_env, '_step_limit') + 1
    del probe_env

    # ---- Network factory ---------------------------------------------------
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
    with open(run_config_path, 'w', encoding='utf-8') as fh:
        json.dump({
            'entrypoint':       'wbc_contrastive.py',
            'env':              env_name,
            'seed':             int(seed),
            'flags':            {k: _json_safe(v)
                                 for k, v in FLAGS.flag_values_dict().items()},
            'resolved_config':  {k: _json_safe(v)
                                 for k, v in config.__dict__.items()},
            'fixed_start_end':  _json_safe(fixed_start_end),
            'env_defaults':     _json_safe(env_defaults),
        }, fh, indent=2, sort_keys=True)
    print(f'[wbc_contrastive] wrote run config: {run_config_path}')

    from default import make_default_logger
    logger_fn = functools.partial(
        make_default_logger,
        save_dir=run_dir,
        add_uid=config.add_uid,
        steps_key='learner_steps',
        float_precision=8)

    # ---- Train -------------------------------------------------------------
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    wbc_learner.run_wbc_training(
        config=config,
        env_factory=env_factory,
        eval_env_factory=eval_env_factory,
        network_factory=network_factory,
        logger_fn=logger_fn,
        total_steps=FLAGS.num_steps,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == '__main__':
    app.run(main)
