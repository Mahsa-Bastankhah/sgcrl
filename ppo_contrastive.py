"""Standalone PPO-on-φ·ψ entry point.

Run with:
  python ppo_contrastive.py \
      --env=point_FourRooms \
      --seed=0 \
      --num_steps=8000000 \
      --log_dir_path=logs/ppo/

Sawyer bin (fixed goal, vanilla CRL without task-goal extra negatives):
  python ppo_contrastive.py \
      --env=sawyer_bin \
      --seed=0 \
      --num_steps=8000000 \
      --log_dir_path=logs/ppo/

PPO is on-policy and single-process; this script does NOT go through
Launchpad.  The SAC-based kappa actor stays untouched and is still
launched via lp_contrastive.py --alg=kappa_sac.
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports
import functools
import json
import os

from absl import app
from absl import flags

import contrastive
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils

FLAGS = flags.FLAGS

flags.DEFINE_string('log_dir_path', 'logs/ppo/', 'Where to log metrics')
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_bool('add_uid', False, 'Whether to add a unique id to the log directory name')
flags.DEFINE_string('env', 'point_FourRooms', 'Environment type')
flags.DEFINE_integer('num_steps', 8_000_000, 'Total env steps', lower_bound=0)
flags.DEFINE_bool('sample_goals', False,
                  'Sample goal uniformly (else use the fixed goal dict)')
flags.DEFINE_bool(
    'repr_norm', False,
    'If True, L2-normalize critic φ and ψ before dot products.')
# Optional explicit overrides for the two per-env-defaulted knobs.
# If left at -1 (the default), the per-env lookup in PPO_ENV_DEFAULTS wins;
# any non-negative value supplied on the CLI overrides that table.  This lets
# sweeps pin T / crl_steps without editing the source.
flags.DEFINE_integer('ppo_rollout_length', -1,
                     'If >=0, overrides the per-env rollout length default.')
flags.DEFINE_integer('ppo_crl_steps_per_iter', -1,
                     'If >=0, overrides the per-env CRL-steps default.')
flags.DEFINE_integer(
    'ppo_num_envs', -1,
    'If >=0, overrides ContrastiveConfig.ppo_num_envs (default 8). With '
    '--maniskill_native_vec, all envs are simulated in one GPU-batched '
    'ManiSkill scene, so this can be pushed well past the default without '
    'the per-env Python-loop overhead that would otherwise scale with it.')
flags.DEFINE_float(
    'discount', -1.0,
    'If >=0, overrides ContrastiveConfig.discount (CRL discount).')
flags.DEFINE_float(
    'ppo_discount', -1.0,
    'If >0, overrides PPO discount for GAE/returns only; '
    '<=0 falls back to config.discount.')
flags.DEFINE_float(
    'ppo_clip_coef', -1.0,
    'If >0, overrides PPO clip coefficient.')
flags.DEFINE_float(
    'ppo_actor_min_std', -1.0,
    'If >0, overrides PPO actor min std.')
flags.DEFINE_float(
    'ppo_ent_coef', -1.0,
    'If >=0, overrides PPO entropy bonus coefficient; '
    '<0 keeps ContrastiveConfig default.')
flags.DEFINE_bool(
    'ppo_anneal_lr', True,
    'If True, linearly decay PPO Adam learning rate to 0 over training; '
    'if False, use a fixed learning_rate.  Pass --noppo_anneal_lr to disable.')
flags.DEFINE_bool(
    'uniform_sampling', False,
    'If True, mix uniformly sampled goals into each CRL replay batch as '
    'extra off-diagonal negatives (never used as positives).')
flags.DEFINE_integer(
    'uniform_num_negatives', -1,
    'Number of uniform negative goals to add per CRL batch when '
    'uniform_sampling is True.  -1 (default) uses batch_size // 2.')
flags.DEFINE_float(
    'ppo_crl_repr_tau', -1.0,
    'If in (0, 1), use an EMA "slow" copy of q_params '
    '(ema <- tau*ema + (1-tau)*online after each CRL step) to compute the '
    'PPO reward, while the online q_params keeps training on the CRL '
    'loss. Target-network-style stabilization for PPO on a non-stationary '
    'learned reward. <=0 or >=1 (default -1) disables this: reward reads '
    'the live q_params directly.')
flags.DEFINE_bool(
    'ppo_crl_add_extrinsic_reward', False,
    'If True, add the sparse extrinsic success reward (1 if the goal has '
    'been reached, else 0) on top of the φ·ψ representation reward before '
    'GAE. Only affects maniskill_close_subtask_train/'
    'maniskill_open_subtask_train (uses success_key=\'drawer_closed\'/'
    '\'drawer_open\' for the extra term); a no-op elsewhere.')
flags.DEFINE_integer(
    'ppo_checkpoint_interval', -1,
    'If >=0, overrides ContrastiveConfig.ppo_checkpoint_interval (default '
    '500 iterations). Useful to lower for long/GPU-crash-prone runs '
    '(e.g. maniskill_close_subtask_train) so a mid-run crash loses less '
    'progress before the next `sbatch` resubmit auto-resumes from '
    'checkpoint_dir/latest.pkl.')
flags.DEFINE_string(
    'ppo_repr_mode', 'crl',
    'Density estimator for the PPO shaped reward / off-policy critic '
    'update: "crl" (default, InfoNCE on φ,ψ) or "nf" (conditional RealNVP '
    'log p_NF(g|s,a), see contrastive/nf_density.py).')
flags.DEFINE_integer('nf_rep_size', 256,
                     'NF mode: SA encoder output dim (conditioning vector).')
flags.DEFINE_integer('nf_num_blocks', 12,
                     'NF mode: number of RealNVP coupling blocks.')
flags.DEFINE_integer(
    'nf_coupling_width', 512,
    'NF mode: width of each coupling block\'s s/t sub-networks.')
flags.DEFINE_integer('nf_sa_hidden', 1024, 'NF mode: SA encoder hidden width.')
flags.DEFINE_integer('nf_sa_num_layers', 4, 'NF mode: SA encoder depth.')
flags.DEFINE_integer(
    'nf_goal_enc_size', 0,
    'NF mode: >0 enables a goal encoder mapping goal to this latent size; '
    '0 = raw goal fed to the flow.')
flags.DEFINE_float('nf_encoder_lr', 3e-4,
                   'NF mode: Adam lr for the SA (+ goal) encoder(s).')
flags.DEFINE_float('nf_critic_lr', 1e-4, 'NF mode: AdamW lr for the RealNVP flow.')
flags.DEFINE_float('nf_critic_weight_decay', 1e-6,
                   'NF mode: AdamW weight decay for the flow.')
flags.DEFINE_float(
    'nf_grad_clip', 1.0,
    'NF mode: global-norm grad clip for NF optimizers (0 disables).')
flags.DEFINE_float(
    'nf_noise_std', 0.05,
    'NF mode: Gaussian noise added to normalized goals during NF training.')
flags.DEFINE_float(
    'nf_goal_std_min', 0.1,
    'NF mode: floor on the per-dim replay-goal std used to normalize goals '
    'before the flow.')
flags.DEFINE_boolean(
    'nf_mix_task_goal_stats', False,
    'NF mode: compute goal normalization mean/std from batch_size replay '
    'hindsight goals plus batch_size copies of the current rollout\'s '
    'actual env task goal, so the task goal is in-distribution for the '
    'normalizer instead of only hindsight-relabeled goals. Helps prevent '
    'the flow density (and thus the PPO reward) from exploding early in '
    'training when the task goal is far out-of-distribution relative to '
    'replay.')
flags.DEFINE_float(
    'ppo_nf_reward_tau', 0.0,
    'NF mode: if in (0, 1), use an EMA "slow" copy of the NF params '
    '(mirrors --ppo_crl_repr_tau, CRL-only) to compute the PPO reward, '
    'while the online NF params keep training on the NLL loss. '
    '<=0 or >=1 (default) disables this: reward reads live NF params.')
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
    'Comma-separated MLP widths for the policy/value/critic networks, '
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
# ---------------------------------------------------------------------------
# Fixed-goal lookup + per-env PPO defaults, shared with `ppo_rnd.py`. Factored
# out to `ppo_env_defaults.py` (no `absl.flags` side effects there) so that
# module can be imported without pulling in this module's flag definitions --
# see that module's docstring. Re-exported here so existing
# `from ppo_contrastive import fixed_goal_dict` call sites keep working.
# ---------------------------------------------------------------------------
from ppo_env_defaults import fixed_goal_dict, PPO_ENV_DEFAULTS  # noqa: E402


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
  print(f'[ppo_contrastive] env={env_name} seed={seed}')

  # ---- Build config ------------------------------------------------------
  # Only parameters we explicitly want to override are set here; everything
  # else inherits the ContrastiveConfig defaults (including the new PPO_*
  # fields in contrastive/config.py).
  params = dict(
      seed=seed,
      env_name=env_name,
      alg_name='ppo',
      reward_shaping_mode='ppo',
      use_cpc=True,                   # CRL loss: InfoNCE / CPC (matches kappa_sac)
      max_number_of_steps=FLAGS.num_steps,
      log_dir=FLAGS.log_dir_path,
      add_uid=FLAGS.add_uid,
      fix_goals=not FLAGS.sample_goals,
  )
  config = contrastive.ContrastiveConfig(**params)
  config.repr_norm = bool(FLAGS.repr_norm)
  config.hidden_layer_sizes = tuple(
      int(x) for x in FLAGS.hidden_layer_sizes.split(','))

  # ---- Per-env PPO defaults (CLI flags still override) -------------------
  env_defaults = PPO_ENV_DEFAULTS.get(env_name)
  if env_defaults is None:
    print(f'[ppo_contrastive] WARNING: no PPO_ENV_DEFAULTS entry for '
          f'{env_name!r}; falling back to ContrastiveConfig defaults '
          f'(T={config.ppo_rollout_length}, '
          f'crl_steps={config.ppo_crl_steps_per_iter}).')
  else:
    config.ppo_rollout_length = int(env_defaults['rollout_length'])
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
  if FLAGS.ppo_discount > 0.0:
    config.ppo_discount = float(FLAGS.ppo_discount)
  if FLAGS.ppo_clip_coef > 0.0:
    config.ppo_clip_coef = float(FLAGS.ppo_clip_coef)
  if FLAGS.ppo_actor_min_std > 0.0:
    config.ppo_actor_min_std = float(FLAGS.ppo_actor_min_std)
  if FLAGS.ppo_ent_coef >= 0.0:
    config.ppo_ent_coef = float(FLAGS.ppo_ent_coef)
  config.ppo_anneal_lr = bool(FLAGS.ppo_anneal_lr)
  config.uniform_sampling = bool(FLAGS.uniform_sampling)
  if FLAGS.uniform_num_negatives >= 0:
    config.uniform_num_negatives = int(FLAGS.uniform_num_negatives)
  if 0.0 < FLAGS.ppo_crl_repr_tau < 1.0:
    config.ppo_crl_repr_tau = float(FLAGS.ppo_crl_repr_tau)
  config.ppo_crl_add_extrinsic_reward = bool(FLAGS.ppo_crl_add_extrinsic_reward)
  if FLAGS.ppo_checkpoint_interval >= 0:
    config.ppo_checkpoint_interval = int(FLAGS.ppo_checkpoint_interval)
  config.ppo_maniskill_native_vec = bool(FLAGS.maniskill_native_vec)

  config.ppo_repr_mode = FLAGS.ppo_repr_mode.strip().lower()
  if config.ppo_repr_mode not in ('crl', 'nf'):
    raise ValueError(
        f'Unknown --ppo_repr_mode={FLAGS.ppo_repr_mode!r}; expected "crl" or "nf"')
  config.nf_rep_size = int(FLAGS.nf_rep_size)
  config.nf_num_blocks = int(FLAGS.nf_num_blocks)
  config.nf_coupling_width = int(FLAGS.nf_coupling_width)
  config.nf_sa_hidden = int(FLAGS.nf_sa_hidden)
  config.nf_sa_num_layers = int(FLAGS.nf_sa_num_layers)
  config.nf_goal_enc_size = int(FLAGS.nf_goal_enc_size)
  config.nf_encoder_lr = float(FLAGS.nf_encoder_lr)
  config.nf_critic_lr = float(FLAGS.nf_critic_lr)
  config.nf_critic_weight_decay = float(FLAGS.nf_critic_weight_decay)
  config.nf_grad_clip = float(FLAGS.nf_grad_clip)
  config.nf_noise_std = float(FLAGS.nf_noise_std)
  config.nf_goal_std_min = float(FLAGS.nf_goal_std_min)
  config.nf_mix_task_goal_stats = bool(FLAGS.nf_mix_task_goal_stats)
  if 0.0 < FLAGS.ppo_nf_reward_tau < 1.0:
    config.ppo_nf_reward_tau = float(FLAGS.ppo_nf_reward_tau)
  if config.ppo_repr_mode == 'nf' and config.uniform_sampling:
    print('[ppo_contrastive] WARNING: --uniform_sampling has no effect '
          'under --ppo_repr_mode=nf (the NF density loss never reads the '
          'extra_goals CRL adds); proceeding without uniform negatives.')

  print(f'[ppo_contrastive] PPO knobs: '
        f'rollout_length={config.ppo_rollout_length}, '
        f'crl_steps_per_iter={config.ppo_crl_steps_per_iter}, '
        f'num_envs={config.ppo_num_envs}, '
        f'clip_coef={config.ppo_clip_coef}, '
        f'actor_min_std={config.ppo_actor_min_std}, '
        f'ent_coef={config.ppo_ent_coef}, '
        f'discount_crl={config.discount}, '
        f'discount_ppo={config.ppo_discount if config.ppo_discount > 0 else config.discount}, '
        f'norm_reward={config.ppo_norm_reward}, '
        f'norm_obs={config.ppo_norm_obs}, '
        f'repr_norm={config.repr_norm}, '
        f'ppo_anneal_lr={config.ppo_anneal_lr}')
  if config.ppo_crl_repr_tau > 0.0:
    print(f'[ppo_contrastive] CRL reward repr EMA: '
          f'tau={config.ppo_crl_repr_tau} (reward uses a slow copy of '
          f'q_params; CRL loss still trains the live q_params)')

  # ---- Env-specific extra kwargs (forwarded to env_utils.load) ----------
  _env_kwargs = {}
  if env_name == 'sawyer_bin' and FLAGS.bin_randomize_gripper_init:
    _env_kwargs['randomize_gripper_init'] = True
    print('[ppo_contrastive] sawyer_bin init: randomized gripper position '
          'at reset')

  if config.ppo_maniskill_native_vec and not env_name.startswith('maniskill_'):
    print(f'[ppo_contrastive] WARNING: --maniskill_native_vec has no '
          f'effect on env={env_name!r} (not a maniskill_* env); ignoring.')
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

  # obs_dim / max_episode_steps inferred from one sample env -- EXCEPT for
  # --maniskill_native_vec, where probing via a real (num_envs=1, CPU)
  # ManiSkill env here would permanently block the later GPU vec env from
  # initializing (SAPIEN requires physx.enable_gpu() -- triggered when
  # ppo_learner.run_ppo_training builds the real num_envs>1 vec env -- to
  # be the very first PhysX-touching call in the process; see
  # env_utils.maniskill_static_obs_info and the "vec env" comment atop
  # run_ppo_training). Both values are static per env_name for ManiSkill
  # envs, so a plain lookup avoids constructing anything.
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

  # ---- Network factory (adds value_network via networks.py changes) -----
  # NOTE: `actor_min_std` is raised from the shared default (1e-6) to the
  # PPO-specific floor (0.1 by default).  Without this, the tanh-squashed
  # Gaussian policy collapses to a near-point-mass within a handful of
  # PPO updates — SAC gets away with a 1e-6 floor because adaptive-α
  # actively regulates entropy; PPO has no such control loop and relies
  # on (a) an entropy bonus and (b) a hard std floor to stay exploratory.
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
      'entrypoint': 'ppo_contrastive.py',
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
  print(f'[ppo_contrastive] wrote run config: {run_config_path}')

  # ---- Optional live wandb logging ---------------------------------------
  # The wandb run id is persisted to `run_dir` so that a later invocation
  # against the same log dir (e.g. re-running with a larger --num_steps to
  # continue training past a checkpoint) resumes logging into the SAME
  # wandb run instead of starting a new one -- `run_ppo_training` already
  # restores `global_step`/`iteration` from the checkpoint and WandbLogger
  # logs with those true step values (default.py), so resuming the run here
  # is the last piece needed for a single continuous wandb timeline.
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
    print(f'[ppo_contrastive] wandb run: {wandb_run.url}')

  from default import make_default_logger
  logger_fn = functools.partial(
      make_default_logger,
      save_dir=run_dir,
      add_uid=config.add_uid,
      steps_key='learner_steps',
      wandb_run=wandb_run)

  # ---- Optional periodic mid-training rollout video -----------------------
  # Writes to a single fixed local path, overwritten every call, so local
  # disk usage doesn't grow with training length -- unlike the (separate)
  # end-of-training video below, which is kept at its own path.
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
          # step=iteration_ matches the 'learner'/'eval' WandbLoggers'
          # 'learner_steps' convention (see default.WandbLogger) so this
          # never regresses wandb's monotonic step counter relative to the
          # per-iteration metric writes happening in the same loop.
          wandb_run.log(
              {'video/periodic_rollout': wandb.Video(
                  video_path, fps=FLAGS.video_fps, format='mp4'),
               'video/periodic_rollout_global_step': global_step_},
              step=iteration_)
        print(f'[ppo_contrastive] wrote periodic rollout video at '
              f'global_step={global_step_}: {video_path}')
      except Exception as exc:  # noqa: BLE001 -- never fail training over video
        print(f'[ppo_contrastive] WARNING: periodic video rendering failed '
              f'(training is unaffected): {type(exc).__name__}: {exc}')

  # ---- Go ----------------------------------------------------------------
  checkpoint_dir = os.path.join(run_dir, 'checkpoints')
  final_state = ppo_learner.run_ppo_training(
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
  # Best-effort: a rendering bug must never lose a completed run's results.
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
      print(f'[ppo_contrastive] wrote end-of-training video: {video_path}')
    except Exception as exc:  # noqa: BLE001 -- never fail a completed run over video
      print(f'[ppo_contrastive] WARNING: end-of-training video rendering '
            f'failed (training results are unaffected): '
            f'{type(exc).__name__}: {exc}')

  if wandb_run is not None:
    wandb_run.finish()


if __name__ == '__main__':
  app.run(main)
