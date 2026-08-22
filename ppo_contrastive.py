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
import numpy as np

import contrastive
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
import env_utils

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
    'ppo_crl_batch_size', -1,
    'If >0, overrides the replay batch size used by each PPO-CRL update '
    '(ContrastiveConfig.batch_size default=256).')
flags.DEFINE_integer('ppo_num_envs', -1,
                     'If >=0, overrides the number of parallel env rollouts.')
flags.DEFINE_integer(
    'ppo_num_epochs', -1,
    'If >=0, overrides PPO update epochs over each rollout batch '
    '(ContrastiveConfig.ppo_num_epochs default=10).')
flags.DEFINE_integer(
    'ppo_num_minibatches', -1,
    'If >=0, overrides minibatches per PPO epoch '
    '(ContrastiveConfig.ppo_num_minibatches default=4). '
    'Must divide rollout_length * num_envs evenly.')
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
flags.DEFINE_bool(
    'ppo_deterministic_select_dim', False,
    'If True, deactivate policy std on the last action dim (BuilderBench PD '
    'select_action): that dim always uses μ/mode; other dims stay stochastic.')
flags.DEFINE_bool(
    'ppo_categorical_select', True,
    'If True (default), use hybrid actor: shared trunk, tanh-Gaussian on '
    'continuous action dims, categorical logits over num_cubes classes for '
    'select (BuilderBench creative only; no-op elsewhere). '
    'Pass --noppo_categorical_select to disable.')
flags.DEFINE_float(
    'ppo_ent_coef', -1.0,
    'If >=0, overrides PPO entropy bonus coefficient; '
    '<0 keeps ContrastiveConfig default.')
flags.DEFINE_bool(
    'ppo_anneal_lr', True,
    'If True, linearly decay PPO Adam learning rate to 0 over training; '
    'if False, use a fixed learning_rate.  Pass --noppo_anneal_lr to disable.')
flags.DEFINE_bool(
    'ppo_anneal_ent_coef', False,
    'If True, linearly decay the PPO entropy-bonus coefficient from '
    'ppo_ent_coef down to ppo_ent_coef_final over training (mirrors '
    'ppo_anneal_lr). Off by default (fixed ent_coef).')
flags.DEFINE_float(
    'ppo_ent_coef_final', 0.0,
    'Final entropy coefficient when --ppo_anneal_ent_coef is set.')
flags.DEFINE_bool(
    'ppo_use_good_buffer', False,
    'If True, enable Good Experience Replay buffer for PPO bootstrapping.')
flags.DEFINE_string(
    'ppo_good_buffer_mode', 'sil',
    'Good experience buffer mode: "sil" (Self-Imitation Loss) or "mixed" (Mixed minibatches).')
flags.DEFINE_integer(
    'ppo_good_buffer_min_cubes', 2,
    'Minimum stacked cubes (2, 3, or 4) required to save rollout trajectory.')
flags.DEFINE_float(
    'ppo_good_buffer_coef', 0.1,
    'Loss coefficient for SIL loss (or mix fraction for mixed mode).')
flags.DEFINE_integer(
    'ppo_good_buffer_max_size', 50000,
    'Maximum transitions stored in Good Experience Buffer.')
flags.DEFINE_bool(
    'ppo_norm_obs', False,
    'Normalize state observations using running per-dimension statistics. '
    'Goals reuse the corresponding state statistics. Off by default.')
flags.DEFINE_float(
    'ppo_obs_norm_clip', 10.0,
    'Absolute clipping bound after observation normalization.')
flags.DEFINE_bool(
    'uniform_sampling', False,
    'If True, mix 50% uniformly sampled goals into each CRL replay batch. '
    'Half the in-batch InfoNCE negatives come from the uniform goal '
    'distribution, half from the replay future-state distribution.')
flags.DEFINE_string(
    'ppo_reward_mode', '',
    "PPO rollout reward baseline. '' = φ(s,a)·ψ(g); "
    "'dirac_target' = log(eps)−φ(s0,a)·ψ(s) off-goal, −φ(s0,a)·ψ(g) at goal; "
    "'kde_dirac' = same formula but densities estimated via Gaussian KDE on "
    "the replay buffer instead of CRL dot products.")
flags.DEFINE_string(
    'ppo_repr_mode', 'crl',
    "Density estimator for the PPO shaped reward. "
    "'crl' (default) = contrastive φ(s,a)·ψ(g) representations; "
    "'gaussian' = diagonal Gaussian p_θ(g|s), reward = log p_θ(g|s_t); "
    "'nf' = conditional RealNVP log p_NF(g|s,a), reward = log p_NF; "
    "'fm' = OT flow-matching log p_FM(g|s,a) via reverse ODE, "
    "reward = log p_FM (uses --ppo_fm_reward_tau for reward EMA); "
    "'td3' = twin Q(s,a,s_f) TD3-style on r=1{s≈s_f}, reward = Q1(s,a,g) "
    "(or log((1−γ)Q1) with --ppo_td3_log_reward); "
    "'crl_td3_switch' = train both critics, initially reward PPO with CRL, "
    "then switch to TD3 after the configured number of hard-goal visits; "
    "'tdinfonce' = TD InfoNCE φ(s,a)·ψ(s_f) critic, reward still φ·ψ "
    "(uses --ppo_crl_repr_tau for reward EMA).")
flags.DEFINE_float(
    'ppo_td_infonce_target_tau', -1.0,
    'TD-InfoNCE mode: keep-rate for target critic '
    'target ← τ·target + (1−τ)·online. Default/config is 0.995. '
    '<0 keeps ContrastiveConfig.ppo_td_infonce_target_tau. '
    'Independent of ppo_crl_repr_tau (reward EMA).')
flags.DEFINE_float(
    'ppo_td_infonce_discount', -1.0,
    'TD-InfoNCE mode: mix γ in L=(1-γ)·L_InfoNCE + γ·L_TD. '
    '<0 uses --discount / config.discount (also HER γ). '
    'Set ~0 to isolate term 1 for sanity checks without changing HER.')
flags.DEFINE_float(
    'ppo_td_infonce_logsumexp_coef', -1.0,
    'TD-InfoNCE mode: coef on logsumexp(logits)^2 added to softmax CE '
    '(CRL/CPC default is 0.01). <0 keeps config default. 0 disables.')
flags.DEFINE_integer(
    'ppo_tdinfonce_crl_warmup_iters', 0,
    'TD-InfoNCE mode: if >0, train φ,ψ with standard CRL InfoNCE for this '
    'many PPO iterations, then switch critic updates to TD-InfoNCE. '
    'Reward stays φ·ψ throughout. 0 = TD-InfoNCE from the start.')
flags.DEFINE_integer(
    'ppo_reward_switch_goal_visits', 5,
    "crl_td3_switch mode: completed successful training episodes required "
    "before PPO changes from CRL reward to TD3 reward on the next iteration.")
flags.DEFINE_integer(
    'ppo_reward_switch_blend_iters', 0,
    "crl_td3_switch mode: after the visit threshold, linearly blend "
    "r=(1-w)·CRL + w·TD3 over this many PPO iterations "
    "(w=0 at switch_iter, w=1 after blend_iters). "
    "0 = hard switch to TD3 (legacy).")
flags.DEFINE_float(
    'ppo_td3_tau', -1.0,
    'TD3 mode: Polyak τ for target Q networks. '
    'Independent of ppo_crl_repr_tau. <0 keeps config default '
    '(falls back to ContrastiveConfig.tau=0.005).')
flags.DEFINE_float(
    'ppo_td3_goal_tol', -1.0,
    'TD3 mode: L2 tolerance for indicator 1{s≈s_f}. '
    '<0 keeps config default (1e-2).')
flags.DEFINE_boolean(
    'ppo_td3_use_target_policy', False,
    'TD3 mode: if True, sample a\' for Q(s\',a\',s_f) from a Polyak '
    'target policy (same τ as Q targets); if False (default), use online '
    'PPO policy.  PPO reward always uses online Q1.')
flags.DEFINE_boolean(
    'ppo_td3_cross_batch_goals', False,
    'TD3 mode: if True, train Q(s_i,a_i,g_j) for every batch '
    'goal g_j (B² backups); if False (default), only the paired g_i.')
flags.DEFINE_boolean(
    'ppo_td3_bilinear', False,
    'TD3 mode: if True, Q(s,a,g)=x(s,a)·y(g) with x/y matching CRL φ/ψ '
    '(repr_dim); if False (default), MLP on concat([s;g;a]).')
flags.DEFINE_float(
    'ppo_td3_reward_tau', -1.0,
    'TD3 mode: EMA decay τ for Q params used in PPO reward r=Q1(s,a,g). '
    'ema ← τ·ema + (1−τ)·online after each TD3 critic step. '
    '0 = use online Q1 (default). Independent of ppo_td3_tau (Polyak). '
    '<0 keeps config default.')
flags.DEFINE_boolean(
    'ppo_td3_log_reward', False,
    'TD3 mode: if True, PPO reward is log((1−γ)·max(Q1, ε)) instead of '
    'raw Q1 (log-occupancy / log-density scale).')
flags.DEFINE_float(
    'ppo_gaussian_reward_tau', -1.0,
    'Gaussian mode: EMA decay τ for density params used in PPO reward '
    'r=log p_θ(g|s,a). ema ← τ·ema + (1−τ)·online after each density step. '
    '0 = use online params (default). <0 keeps config default.')
flags.DEFINE_float(
    'ppo_fm_reward_tau', -1.0,
    'FM mode: EMA decay τ for velocity-field params used in PPO reward '
    'r=log p_FM(g|s,a). ema ← τ·ema + (1−τ)·online after each density step. '
    '0 = use online params (default). <0 keeps config default.')
flags.DEFINE_integer(
    'fm_flow_steps', -1,
    'FM mode: Euler steps for reverse-ODE log-density / sampling. '
    '<0 keeps config default (10).')
flags.DEFINE_string(
    'fm_logp_mode', '',
    "FM mode: divergence for log p — 'exact', 'hutch-rade', or 'hutch-gaus'. "
    'Empty keeps config default (exact).')
flags.DEFINE_integer(
    'fm_hutch_probes', -1,
    'FM mode: Hutchinson probes when fm_logp_mode is hutch-*. '
    '<0 keeps config default (8).')
flags.DEFINE_boolean(
    'fm_layer_norm', False,
    'FM mode: apply LayerNorm inside the velocity MLP (FAC-style).')
flags.DEFINE_boolean(
    'fm_time_embedding', False,
    'FM mode: apply Fourier sinusoidal embeddings to scalar time t.')
flags.DEFINE_integer(
    'fm_time_embed_dim', 32,
    'FM mode: output dimension for Fourier time embedding.')
flags.DEFINE_string(
    'fm_ode_solver', 'euler',
    "FM mode: ODE solver for reverse-ODE log-density & sampling ('euler' | 'heun').")
flags.DEFINE_string(
    'fm_t_sample_mode', 'uniform',
    "FM mode: timestep sampling distribution for training ('uniform' | 'logit_normal').")
flags.DEFINE_float(
    'fm_t_logit_loc', 0.0,
    'FM mode: location (mean) for Logit-Normal timestep sampling.')
flags.DEFINE_float(
    'fm_t_logit_scale', 1.0,
    'FM mode: scale (std) for Logit-Normal timestep sampling.')
flags.DEFINE_float(
    'fm_goal_noise_std', 0.0,
    'FM mode: Gaussian noise std added to future goals s_f during training updates (0 = disabled).')
flags.DEFINE_boolean(
    'fm_norm_goals', False,
    'FM mode: normalize goal vectors s_f to unit variance before flow density updates.')
flags.DEFINE_float(
    'fm_goal_std_min', 0.02,
    'FM mode: floor on per-dim replay std (avoids division by zero on static dims).')
flags.DEFINE_float(
    'fm_cond_dropout', 0.0,
    'FM mode: probability of condition dropout, replacing (s, a) with (0, 0) during flow training.')
flags.DEFINE_float(
    'fm_reward_clip', 0.0,
    'FM mode: reward clipping bound for reverse ODE logp (0 = disabled, e.g. 20.0 clips to [-20, 20]).')
flags.DEFINE_string(
    'fm_cat_acc_mode', 'midpoint',
    "FM mode: categorical accuracy mode ('midpoint' | 'logp').")
flags.DEFINE_integer(
    'fm_cat_acc_subbatch', 128,
    'FM mode: sub-batch size for categorical accuracy retrieval matrix.')
flags.DEFINE_integer(
    'fm_cat_acc_flow_steps', -1,
    'FM mode: reverse ODE steps for categorical accuracy logp (-1 uses fm_flow_steps).')
flags.DEFINE_boolean(
    'fm_td_mode', False,
    'FM mode: use TD-Flow (Bellman probability path targets) for density updates.')
flags.DEFINE_float(
    'fm_td_gamma', 0.99,
    'FM mode: discount factor gamma for TD-Flow Bellman probability path targets.')
flags.DEFINE_float(
    'fm_td_target_tau', 0.005,
    'FM mode: Polyak soft-update rate for TD-Flow target vector field v_phi.')
flags.DEFINE_integer(
    'fm_td_boot_steps', 1,
    'FM mode: ODE integration steps for target goal bootstrapping during training.')
flags.DEFINE_integer(
    'fm_logp_diag_interval', 50,
    'FM mode: PPO iteration interval for seen vs unseen FM log-prob diagnostics (0 = disabled).')
flags.DEFINE_integer(
    'fm_logp_diag_batch_size', 64,
    'FM mode: batch size of transitions for FM log-prob diagnostics.')
flags.DEFINE_integer(
    'fm_logp_diag_flow_steps', 5,
    'FM mode: ODE integration steps for FM log-prob diagnostics.')
flags.DEFINE_string(
    'fm_logp_diag_ode_solver', 'euler',
    'FM mode: ODE solver for FM log-prob diagnostics (euler | heun).')
flags.DEFINE_integer(
    'nf_rep_size', 64,
    'NF mode: SA encoder output dim (conditioning vector size).')
flags.DEFINE_integer(
    'nf_num_blocks', 8,
    'NF mode: number of affine coupling blocks in RealNVP.')
flags.DEFINE_integer(
    'nf_coupling_width', 256,
    'NF mode: width of s/t sub-networks inside each coupling block.')
flags.DEFINE_integer(
    'nf_sa_hidden', 1024,
    'NF mode: SA encoder hidden width (reference 1024; compact runs use 256).')
flags.DEFINE_integer(
    'nf_sa_num_layers', 4,
    'NF mode: SA encoder depth (reference 4; compact runs use 3).')
flags.DEFINE_float(
    'nf_encoder_lr', 3e-4,
    'NF mode: Adam learning rate for SA encoder (ref actor_lr).')
flags.DEFINE_float(
    'nf_critic_lr', 1e-4,
    'NF mode: AdamW learning rate for RealNVP flow (ref critic_lr).')
flags.DEFINE_float(
    'nf_critic_weight_decay', 1e-6,
    'NF mode: AdamW weight decay for RealNVP flow (ref critic_weight_decay).')
flags.DEFINE_float(
    'nf_grad_clip', 1.0,
    'NF mode: global-norm gradient clipping for both SA encoder and flow (0 = disabled).')
flags.DEFINE_float(
    'nf_noise_std', 0.05,
    'NF mode: std of Gaussian noise added to goals during training (0 = disabled).')
flags.DEFINE_float(
    'nf_goal_std_min', 0.02,
    'NF mode: minimum per-dim goal std from replay stats (prevents div-by-tiny-std on static dims).')
flags.DEFINE_string(
    'nf_no_norm_goal_dims', '',
    'Deprecated; ignored. NF uses running replay mean/std with nf_goal_std_min floor.')
flags.DEFINE_boolean(
    'ppo_skip_first_eval', False,
    'Skip logging the iteration-0 eval (avoids logging the checkpoint result as the first data point when resuming).')
flags.DEFINE_string(
    'ppo_frozen_reward_ckpt', '',
    'Path to a pretrained PPO/CRL checkpoint (.pkl). On a fresh run, load only '
    'φ/ψ (prefer q_params_ema) as a stationary r=φ(s,a)·ψ(g) reward; policy/'
    'value start fresh. Forces --ppo_crl_steps_per_iter=0. Empty = disabled.')
flags.DEFINE_string(
    'ppo_actor_reset_iters', '',
    'Comma-separated PPO iterations at which to force an actor reinit '
    '(in addition to the short-episode guard). Empty disables the schedule.')
flags.DEFINE_integer(
    'ppo_eval_interval', -1,
    'Run eval every N PPO iterations. <0 keeps config/env default (10). 0 disables eval.')
flags.DEFINE_integer(
    'ppo_eval_episodes', -1,
    'Number of eval episodes per eval round. <0 keeps config default (5).')
flags.DEFINE_integer(
    'ppo_video_interval', -1,
    'Render a deterministic BuilderBench video every N PPO iterations. '
    '<0 keeps config default (0=disabled). 0 disables. Uses live obs_rms.')
flags.DEFINE_integer(
    'ppo_video_fps', -1,
    'FPS for in-train BuilderBench videos. <0 keeps config default (10).')
flags.DEFINE_boolean(
    'ppo_skip_first_video', True,
    'Skip the iteration-0 in-train video (random init policy).')
flags.DEFINE_boolean(
    'ppo_save_success_checkpoint', True,
    'Save dedicated checkpoint (ckpt_eval_success_iter_*.pkl, best_eval_success.pkl) '
    'whenever success is achieved.')
flags.DEFINE_boolean(
    'ppo_video_include_reward_plot', True,
    'Include dual-curve synchronized lockstep reward timeline plot above MuJoCo frames.')
flags.DEFINE_integer(
    'ppo_video_max_train_success_videos', 1,
    'Max train-success video renders per iteration.')
flags.DEFINE_boolean(
    'ppo_save_train_success_video', True,
    'Whether to render and log train success videos when on-policy training envs succeed.')
flags.DEFINE_integer(
    'ppo_train_success_min_interval', 10,
    'Minimum PPO iterations between train success video/checkpoint logs to prevent overwhelming.')


flags.DEFINE_boolean(
    'use_wandb', True,
    'Whether to log metrics and videos online to Weights & Biases.')
flags.DEFINE_string(
    'wandb_project', 'dist-matching',
    'Weights & Biases project name.')
flags.DEFINE_string(
    'wandb_entity', 'doina-precup',
    'Weights & Biases entity/team name.')
flags.DEFINE_string(
    'wandb_mode', 'online',
    'Weights & Biases mode: online, offline, or disabled.')
flags.DEFINE_string(
    'wandb_group', '',
    'Weights & Biases group name (defaults to exp_name).')
flags.DEFINE_boolean(
    'ppo_norm_reward', True,
    'Normalize the repr reward by the running std of discounted returns. Set False to pass raw reward directly to PPO.')
flags.DEFINE_boolean(
    'nf_mix_env_goal_stats', False,
    'NF mode: when computing normalisation stats, also include the actual env goals '
    '(obs[obs_dim:] from current rollout) so the normaliser covers reward goals too.')
flags.DEFINE_integer(
    'nf_goal_enc_size', 0,
    'Goal encoder output dim for NF mode. 0 = disabled (raw normalized goal fed to flow). '
    '>0 adds a compact 2×(Dense256+LN+swish)→Dense(N) encoder trained end-to-end with the flow.')
flags.DEFINE_integer(
    'ppo_return_norm_window', 0,
    'Soft sliding-window size for the return normalizer. 0 = infinite (standard Welford). '
    '>0 caps the effective sample count so the variance stays responsive to recent reward shifts.')
flags.DEFINE_float(
    'ppo_dirac_eps', 1e-6,
    'Epsilon in dirac_target / kde_dirac reward: log(eps) − log_p(s).')
flags.DEFINE_integer(
    'kde_max_points', 2000,
    'Number of replay-buffer states to fit the Gaussian KDE on (kde_dirac mode).')
flags.DEFINE_integer(
    'kde_refit_interval', 1,
    'Refit the KDE every N PPO iterations (kde_dirac mode). 1 = every iteration.')
flags.DEFINE_float(
    'kde_bandwidth', 0.0,
    'KDE bandwidth (kde_dirac mode). 0.0 = Scott\'s rule automatically.')
flags.DEFINE_integer(
    'max_replay_size', -1,
    'Max transitions in the PPO episode replay buffer. <0 keeps default (1e6).')
flags.DEFINE_integer(
    'ppo_min_replay_size', -1,
    'Min replay transitions before density/CRL updates start. <0 keeps default (1e4).')
flags.DEFINE_float(
    'ppo_success_sample_weight', -1.0,
    'CRL replay: sampling weight for successful episodes (others weight 1). '
    'Episode indices are drawn proportional to w / sum(w). '
    '1.0 is uniform. <0 keeps config default (1.0).')
flags.DEFINE_boolean(
    'ppo_use_external_reward', False,
    'If True, add ppo_external_reward_scale to the PPO reward on hard-success '
    'steps (BuilderBench metrics["success"], or Sawyer sparse env reward). '
    'Off by default.')
flags.DEFINE_float(
    'ppo_external_reward_scale', 100.0,
    'Bonus added to the PPO reward when ppo_use_external_reward is True and '
    'the step is a hard success. Default +100.')
flags.DEFINE_boolean(
    'ppo_external_reward_before_norm', False,
    'If True with ppo_use_external_reward, add the extrinsic bonus to the raw '
    'shaped reward BEFORE return-norm. Default False (add after norm).')
flags.DEFINE_integer(
    'ppo_checkpoint_interval', -1,
    'Save checkpoints every N PPO iterations. <0 keeps config default (500).')
flags.DEFINE_integer(
    'ppo_checkpoint_keep_last', -1,
    'Max milestone ckpt_iter_*.pkl files to retain (FIFO). '
    '0 = keep all. <0 keeps config default (0 = keep all).')
flags.DEFINE_string(
    'ppo_crl_loss_direction', 'forward',
    "InfoNCE loss direction for PPO-CRL: 'forward' (fix anchor, vary goal) "
    "or 'backward' (fix goal, vary anchor = transpose logits).")
flags.DEFINE_float(
    'ppo_crl_repr_tau', -1.0,
    'CRL mode: EMA decay τ for φ, ψ used in PPO reward r=φ·ψ. '
    'ema ← τ·ema + (1−τ)·online after each CRL step. '
    '0 = use online params (default). Higher τ = slower reward tracking. '
    '<0 keeps config default.')
flags.DEFINE_float(
    'ppo_nf_reward_tau', -1.0,
    'NF mode: EMA decay τ for NF params used in PPO reward r=log p_NF(g|s,a). '
    'ema ← τ·ema + (1−τ)·online after each NF density step. '
    '0 = use online params (default). Higher τ = slower / smoother reward. '
    'Independent of ppo_crl_repr_tau. <0 keeps config default.')
flags.DEFINE_boolean(
    'ppo_log_dormancy', True,
    'Whether to log network dormancy (DNR, GMA) and vector field diagnostics.')
flags.DEFINE_integer(
    'ppo_dormancy_interval', 50,
    'Log network dormancy and vector field diagnostics every N PPO iterations.')
flags.DEFINE_float(
    'ppo_dormancy_tau', 0.01,
    'Dormancy threshold relative to layer mean activation (default 0.01).')
flags.DEFINE_boolean(
    'bin_randomize_gripper_init', False,
    'SawyerBin: randomize initial gripper TCP offset around the object at reset.')
flags.DEFINE_boolean(
    'sawyer_randomize_init', True,
    'Sawyer bin/peg: if False, freeze MetaWorld object (and peg hole) spawn '
    'to the default init pose every reset. Also disables bin gripper-init '
    'noise even when --bin_randomize_gripper_init is set.')
flags.DEFINE_boolean(
    'builderbench_use_pd', False,
    'BuilderBench: wrap env in PDWrapper (short horizon). Default False = raw control.')
flags.DEFINE_integer(
    'builderbench_pd_duration', 5,
    'BuilderBench PDWrapper: low-level MuJoCo steps per RL step when use_pd=True.')
flags.DEFINE_boolean(
    'builderbench_permute_start_boxes', True,
    'BuilderBench: if True (default), randomly permute which cube gets which '
    'start-box lane (y assignment) at reset. Set False to freeze lane order '
    'from the task file (still samples x within each lane box).')
flags.DEFINE_float(
    'builderbench_fixed_start_x', -1.0,
    'BuilderBench: if >=0, collapse every start-box x low/high to this value '
    '(fixed cube init x; y lanes unchanged). <0 keeps the task-file x range.')
flags.DEFINE_integer(
    'builderbench_mj_episode_length', -1,
    'BuilderBench: if >0, override MuJoCo episode_length (before PD macro '
    'division). E.g. 300 with pd_duration=5 → 60 PD macro steps. '
    '<=0 keeps creative_cube_mj_episode_length default.')
flags.DEFINE_string(
    'hidden_layer_sizes', '',
    'Comma-separated hidden layer widths, e.g. "256,256,256,256,256,256". '
    'Empty string keeps the ContrastiveConfig default. '
    'Stacks with >2 layers automatically use ResidualMLP (LayerNorm + Swish, '
    'skip every 2 layers).')

flags.DEFINE_string('exp_name', 'ppo_contrastive.py', 'Experiment name for logging')
flags.DEFINE_string('obs_space', 'xy,select', 'Comma-separated obs components')

flags.DEFINE_boolean(
      'ppo_cleanrl_actor', False,
      'If True, uses Tanh activations, Orthogonal init, and state-independent std for the Actor (Trick 2).')

flags.DEFINE_enum(
    'obs_norm_mode', 'none', ['none', 'z_scale', 'tied_rsnorm'],
    'Observation preprocessing / normalization mode. '
    '\'none\' (default) = raw observations; '
    '\'z_scale\' = static scaling of z-coordinates (z_scaled = z * z_scale_multiplier); '
    '\'tied_rsnorm\' = tied per-dimension Running Statistics Normalization.')
flags.DEFINE_float(
    'z_scale_multiplier', 3.0,
    'Scaling multiplier k for z_scale mode (z_scaled = z * k).')
flags.DEFINE_float(
    'rsnorm_clip', 10.0,
    'Clipping bound for tied_rsnorm mode.')


flags.DEFINE_float(
    'ppo_warmup_percent', 0.0,
    'Percentage of total iterations to wait before starting PPO actor updates '
    '(CRL trains during this time).')

flags.DEFINE_boolean(
    'staggered_resets', False,
    'If True, randomly staggers the parallel environments before training begins to maximize batch diversity.')
flags.DEFINE_boolean(
    'crl_on_policy', False,
    'If True, disables the replay buffer and trains CRL strictly on the current (T, E) rollout tensor.')
flags.DEFINE_float(
      'builderbench_episode_length_multiplier', 1.0,
      'BuilderBench: scale factor on the base episode length '
        '(100 + num_cubes*50 raw MuJoCo steps), applied before optional '
        'PD-duration division. 1.0 = default; 2.0 = double, etc. Result must '
        'be divisible by --builderbench_pd_duration when '
        '--builderbench_use_pd=true.')
# ---------------------------------------------------------------------------
# Fixed-goal lookup reused from lp_contrastive.py.
# ---------------------------------------------------------------------------
fixed_goal_dict = {
    'point_Spiral7x7':   [np.array([3, 3], dtype=float),
                          np.array([6, 6], dtype=float)],
    'point_Spiral9x9':   [np.array([5, 5], dtype=float),
                          np.array([8, 8], dtype=float)],
    'point_Spiral11x11': [np.array([5, 5], dtype=float),
                          np.array([10, 10], dtype=float)],
    'point_FourRooms':   [np.array([0, 0], dtype=float),
                          np.array([2, 8],  dtype=float)],
    # point_EightRooms: start top-left corner, goal bottom-right corner.
    # Must traverse all 8 rooms (11×21 grid, 100-step episodes).
    'point_EightRooms':  [np.array([0, 0],   dtype=float),
                          np.array([10, 20], dtype=float)],
    # point_SixteenRooms: start top-left corner, goal bottom-right corner.
    # Must traverse all 16 rooms (21×21 grid, 200-step episodes).
    'point_SixteenRooms': [np.array([0, 0],   dtype=float),
                           np.array([20, 20], dtype=float)],
    # point_SixteenRooms4D: same maze + 2 free extra dims (all start/goal at 0).
    'point_SixteenRooms4D': [np.array([0, 0, 0],    dtype=float),
                              np.array([20, 20, 0], dtype=float)],
    # point_SixteenRoomsActual4D: same maze + 2 free extra dims.
    'point_SixteenRoomsActual4D': [np.array([0, 0, 0, 0],    dtype=float),
                                   np.array([20, 20, 0, 0], dtype=float)],
    # point_Impossible: start top-left (0,0), goal row 6 col 8 (reachable via
    # the long winding path through the maze).
    'point_Impossible': [np.array([0, 0], dtype=float),
                         np.array([6, 8], dtype=float)],
    # point_Maze11x11: start top-left (0,0), goal top-right (0,10).
    'point_Maze11x11':  [np.array([0, 0], dtype=float),
                         np.array([0, 10], dtype=float)],
    # point_Wall11x11: start top-left (0,0), goal bottom-right (10,10);
    # must go along the top corridor, down the right gap, then along the bottom.
    'point_Wall11x11':  [np.array([0, 0], dtype=float),
                         np.array([10, 10], dtype=float)],
    'sawyer_bin':   np.array([0.12, 0.7, 0.02]),
    'sawyer_box':   np.array([0.0, 0.75, 0.133]),
    'sawyer_peg':   np.array([-0.3, 0.6, 0.0]),
    # Reach: fixed goal = centre of the goal cube (x=0, y=0.85, z=0.2).
    'sawyer_reach': np.array([0.0, 0.85, 0.2]),
    # Push: fixed goal = puck target (z≈0.02); ψ hand = target − 8 cm y, +3 cm z.
    'sawyer_push':  np.array([0.0, 0.85, 0.02]),
    # Drawer-open: fixed goal = open handle (_target_pos); ψ also places
    # ideal_hand at closed-handle first-contact (+0.2 y, −0.02 z).
    'sawyer_drawer_open':  np.array([0.0, 0.54, 0.09]),
    # Button-press: fixed goal = button depressed to the hole site (y≈0.78).
    'sawyer_button_press': np.array([0, 0.8, 0.115]),
    # One-hot goal over river cells (length must match RIVERSWIM_LEN, default 6).
    'riverswim': np.array([0., 0., 0., 0., 0., 1.], dtype=float),
    # Flow figureeight variants: goal = all N vehicles at target_velocity
    # (20 m/s), normalized by the network max_speed (30 m/s).
    'flow_figureeight':       np.full(14, 20.0 / 30.0, dtype=float),
    'flow_figureeight_7rl':   np.full(14, 20.0 / 30.0, dtype=float),
    'flow_figureeight_14rl':  np.full(14, 20.0 / 30.0, dtype=float),
    'flow_figureeight_4v2rl': np.full(4,  20.0 / 30.0, dtype=float),
    'flow_figureeight_8v4rl': np.full(8,  20.0 / 30.0, dtype=float),
    'flow_figureeight_1v1rl': np.full(1,  20.0 / 30.0, dtype=float),
    'flow_figureeight_2v1rl': np.full(2,  20.0 / 30.0, dtype=float),
    'flow_figureeight_2v2rl': np.full(2,  20.0 / 30.0, dtype=float),
}

# ---------------------------------------------------------------------------
# Per-env PPO defaults.
#
# The single knob that really needs to scale with the environment is the
# rollout length T, because it interacts with episode length: if T < ep_len,
# most rollouts complete zero episodes and GAE must bootstrap the return off
# V(s_T), which hurts sample efficiency early in training.  The CRL step
# count scales proportionally so that the CRL-to-env-step ratio stays
# roughly constant (CRL-steps ≈ T / 2).
#
# point_FourRooms / point_Spiral11x11: 50/100-step episodes -> T=128 keeps
#   ~1-2 completed episodes per env per rollout.  These are the values the
#   tuned point-env runs use, so we keep them to avoid regressing.
# sawyer_{bin,box,peg,...}: 150-step episodes -> T=256 gives one full episode
#   per env per rollout, matching CleanRL's MuJoCo convention. Default CRL
#   updates/iter=10 (sparse relative to env steps; override with
#   --ppo_crl_steps_per_iter).
# ---------------------------------------------------------------------------
PPO_ENV_DEFAULTS = {
    'point_FourRooms':   dict(rollout_length=128, crl_steps_per_iter=64),
    # EightRooms: 100-step episodes (2× FourRooms) → T=256 keeps ~2 episodes/rollout.
    'point_EightRooms':   dict(rollout_length=256, crl_steps_per_iter=128),
    # SixteenRooms: 200-step episodes (2× EightRooms) → T=512 keeps ~2 episodes/rollout.
    'point_SixteenRooms': dict(rollout_length=512, crl_steps_per_iter=256),
    # SixteenRooms4D / SixteenRoomsActual4D: same episode length, same rollout budget.
    'point_SixteenRooms4D':       dict(rollout_length=512, crl_steps_per_iter=256),
    'point_SixteenRoomsActual4D': dict(rollout_length=512, crl_steps_per_iter=256),
    'point_Spiral7x7':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Spiral9x9':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Spiral11x11': dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Maze11x11':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Wall11x11':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Impossible':  dict(rollout_length=128, crl_steps_per_iter=64),
    'riverswim':         dict(rollout_length=128, crl_steps_per_iter=64),
    'sawyer_bin':        dict(rollout_length=256, crl_steps_per_iter=10),
    'sawyer_box':        dict(rollout_length=256, crl_steps_per_iter=10),
    'sawyer_peg':        dict(rollout_length=256, crl_steps_per_iter=10),
    # Reach: very short episodes (150 steps), tiny obs → fast.
    'sawyer_reach':      dict(rollout_length=256, crl_steps_per_iter=10),
    # Push: same budget as bin (same episode length, similar obs structure).
    'sawyer_push':       dict(rollout_length=256, crl_steps_per_iter=10),
    # Drawer-open: same episode length / obs structure as push.
    'sawyer_drawer_open':  dict(rollout_length=256, crl_steps_per_iter=10),
    # Button-press: same episode length / obs structure as push.
    'sawyer_button_press': dict(rollout_length=256, crl_steps_per_iter=10),
    # flow_figureeight: 1500-step episodes. T=1500 gives one complete episode
    # per env per rollout (important for GAE accuracy on a long-horizon env).
    # φ: state = speeds+positions; ψ: goal speeds only (end_index=14 on state).
    'flow_figureeight':  dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=14),
    # 7/14 RL variant: same horizon and indexing, wider action space (7-D).
    'flow_figureeight_7rl':  dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=14),
    # 14/14 RL variant: all vehicles RL-controlled, 14-D action space.
    'flow_figureeight_14rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=14),
    'flow_figureeight_4v2rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=4),
    'flow_figureeight_8v4rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=8),
    # 1-vehicle sanity check: 2-D state, 1-D goal. ψ sees only the 1 goal speed.
    'flow_figureeight_1v1rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=1),
    # 2-vehicle variants: 4-D state (speeds+positions), 2-D goal (target speeds).
    # end_index=2 so ψ only sees the 2 goal speeds.
    'flow_figureeight_2v1rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=2),
    'flow_figureeight_2v2rl': dict(
        rollout_length=1500, crl_steps_per_iter=750,
        start_index=0, end_index=2),
}


def fixed_goal_for_env(env_name: str) -> np.ndarray:
  """Return the fixed goal vector for ``env_name`` (incl. all BuilderBench creative tasks)."""
  if env_name in fixed_goal_dict:
    return fixed_goal_dict[env_name]
  from envs.builderbench_utils import (
      default_fixed_target_goal,
      is_builderbench_creative_env,
      parse_sgcrl_builderbench_env_name,
  )
  if is_builderbench_creative_env(env_name):
    _, num_cubes, task_index = parse_sgcrl_builderbench_env_name(env_name)
    return default_fixed_target_goal(num_cubes, task_index)
  raise KeyError(f'No fixed goal configured for env {env_name!r}')


def ppo_env_defaults_for_env(env_name: str, use_pd: bool = False,
                             episode_length_multiplier: float = 1.0):
  """Return per-env PPO defaults, including dynamic BuilderBench creative tasks."""
  if env_name in PPO_ENV_DEFAULTS:
    return PPO_ENV_DEFAULTS[env_name]
  from envs.builderbench_utils import (
      is_builderbench_creative_env,
      parse_sgcrl_builderbench_env_name,
      ppo_env_defaults as builderbench_ppo_defaults,
  )
  if is_builderbench_creative_env(env_name):
    _, num_cubes, task_index = parse_sgcrl_builderbench_env_name(env_name)
    return builderbench_ppo_defaults(
        num_cubes, use_pd=use_pd, task_index=task_index, episode_length_multiplier=episode_length_multiplier)
  return None


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
      obs_norm_mode=FLAGS.obs_norm_mode,
      z_scale_multiplier=float(FLAGS.z_scale_multiplier),
      rsnorm_clip=float(FLAGS.rsnorm_clip),
  )
  config = contrastive.ContrastiveConfig(**params)
  config.repr_norm = bool(FLAGS.repr_norm)

  # ---- Per-env PPO defaults (CLI flags still override) -------------------
  _use_pd = (bool(FLAGS.builderbench_use_pd)
             if env_name.startswith('builderbench_') else False)
  _ep_mult = float(FLAGS.builderbench_episode_length_multiplier)
  env_defaults = ppo_env_defaults_for_env(env_name, use_pd=_use_pd, episode_length_multiplier=_ep_mult)
  if env_defaults is None:
    print(f'[ppo_contrastive] WARNING: no PPO_ENV_DEFAULTS entry for '
          f'{env_name!r}; falling back to ContrastiveConfig defaults '
          f'(T={config.ppo_rollout_length}, '
          f'crl_steps={config.ppo_crl_steps_per_iter}).')
  else:
    # BuilderBench PD mode deliberately omits these two keys now (see
    # envs/builderbench_utils.ppo_env_defaults docstring): rollout_length
    # and crl_steps_per_iter must always come from an explicit
    # --ppo_rollout_length / --ppo_crl_steps_per_iter flag for PD jobs, not
    # a hidden per-env formula. Non-PD envs still get them from
    # PPO_ENV_DEFAULTS above.
    if 'rollout_length' in env_defaults:
      config.ppo_rollout_length = int(env_defaults['rollout_length'])
    if 'crl_steps_per_iter' in env_defaults:
      config.ppo_crl_steps_per_iter = int(env_defaults['crl_steps_per_iter'])
    if 'start_index' in env_defaults:
      config.start_index = int(env_defaults['start_index'])
    if 'end_index' in env_defaults:
      config.end_index = int(env_defaults['end_index'])
    if 'goal_state_indices' in env_defaults:
      config.goal_state_indices = tuple(
          int(x) for x in env_defaults['goal_state_indices'])
      print(f'[ppo_contrastive] goal_state_indices='
            f'{config.goal_state_indices} '
            f'(masked LHER goal_dim='
            f'{len(config.goal_state_indices)})')
    if 'num_envs' in env_defaults and FLAGS.ppo_num_envs < 0:
      config.ppo_num_envs = int(env_defaults['num_envs'])
    if ('eval_interval' in env_defaults
        and FLAGS.ppo_eval_interval < 0):
      config.ppo_eval_interval = int(env_defaults['eval_interval'])
    if ('checkpoint_interval' in env_defaults
        and FLAGS.ppo_checkpoint_interval < 0):
      config.ppo_checkpoint_interval = int(env_defaults['checkpoint_interval'])

  if FLAGS.ppo_rollout_length >= 0:
    config.ppo_rollout_length = int(FLAGS.ppo_rollout_length)
  if FLAGS.ppo_crl_steps_per_iter >= 0:
    config.ppo_crl_steps_per_iter = int(FLAGS.ppo_crl_steps_per_iter)
  if FLAGS.ppo_crl_batch_size > 0:
    config.batch_size = int(FLAGS.ppo_crl_batch_size)
  if FLAGS.ppo_num_envs >= 0:
    config.ppo_num_envs = int(FLAGS.ppo_num_envs)
  if FLAGS.ppo_num_epochs >= 0:
    config.ppo_num_epochs = int(FLAGS.ppo_num_epochs)
  if FLAGS.ppo_num_minibatches >= 0:
    config.ppo_num_minibatches = int(FLAGS.ppo_num_minibatches)

  total_steps = int(FLAGS.num_steps)
  if env_name.startswith('builderbench_'):
    from envs.builderbench_utils import (
        BUILDERBENCH_NUM_STEPS,
        builderbench_replay_size,
    )
    if FLAGS.max_replay_size < 0:
      config.max_replay_size = builderbench_replay_size(config.ppo_num_envs)
    if FLAGS.num_steps == 8_000_000:
      total_steps = BUILDERBENCH_NUM_STEPS
      config.max_number_of_steps = total_steps
    print(f'[ppo] builderbench scale: num_steps={total_steps} '
          f'max_replay_size={config.max_replay_size} '
          f'(E={config.ppo_num_envs})')

  if FLAGS.discount >= 0.0:
    config.discount = float(FLAGS.discount)
  if FLAGS.ppo_discount > 0.0:
    config.ppo_discount = float(FLAGS.ppo_discount)
  if FLAGS.ppo_clip_coef > 0.0:
    config.ppo_clip_coef = float(FLAGS.ppo_clip_coef)
  if FLAGS.ppo_actor_min_std > 0.0:
    config.ppo_actor_min_std = float(FLAGS.ppo_actor_min_std)
  config.ppo_deterministic_select_dim = bool(
      FLAGS.ppo_deterministic_select_dim)
  config.ppo_categorical_select = bool(FLAGS.ppo_categorical_select)
  if FLAGS.ppo_ent_coef >= 0.0:
    config.ppo_ent_coef = float(FLAGS.ppo_ent_coef)
  config.ppo_anneal_lr = bool(FLAGS.ppo_anneal_lr)
  config.ppo_anneal_ent_coef = bool(FLAGS.ppo_anneal_ent_coef)
  config.ppo_ent_coef_final = float(FLAGS.ppo_ent_coef_final)
  config.ppo_use_good_buffer = bool(FLAGS.ppo_use_good_buffer)
  config.ppo_good_buffer_mode = str(FLAGS.ppo_good_buffer_mode).strip().lower()
  config.ppo_good_buffer_min_cubes = int(FLAGS.ppo_good_buffer_min_cubes)
  config.ppo_good_buffer_coef = float(FLAGS.ppo_good_buffer_coef)
  config.ppo_good_buffer_max_size = int(FLAGS.ppo_good_buffer_max_size)
  config.uniform_sampling = bool(FLAGS.uniform_sampling)
  config.staggered_resets = bool(FLAGS.staggered_resets)
  config.crl_on_policy = bool(FLAGS.crl_on_policy)
  config.ppo_reward_mode = str(FLAGS.ppo_reward_mode).strip()
  config.ppo_repr_mode = str(FLAGS.ppo_repr_mode).strip()
  if FLAGS.ppo_td_infonce_target_tau >= 0.0:
    config.ppo_td_infonce_target_tau = float(FLAGS.ppo_td_infonce_target_tau)
  # Allow 0.0 (term-1-only sanity); <0 keeps config default (−1 → use discount).
  if FLAGS.ppo_td_infonce_discount >= 0.0:
    config.ppo_td_infonce_discount = float(FLAGS.ppo_td_infonce_discount)
  # Allow 0.0 (disable); <0 keeps config default (0.01).
  if FLAGS.ppo_td_infonce_logsumexp_coef >= 0.0:
    config.ppo_td_infonce_logsumexp_coef = float(
        FLAGS.ppo_td_infonce_logsumexp_coef)
  config.ppo_tdinfonce_crl_warmup_iters = int(
      FLAGS.ppo_tdinfonce_crl_warmup_iters)
  if str(config.ppo_repr_mode).strip().lower() in ('tdinfonce', 'td_infonce'):
    if bool(config.twin_q):
      print('[ppo_contrastive] tdinfonce requires twin_q=False; '
            'forcing twin_q=False')
    config.twin_q = False
  config.ppo_reward_switch_goal_visits = int(
      FLAGS.ppo_reward_switch_goal_visits)
  config.ppo_reward_switch_blend_iters = int(
      FLAGS.ppo_reward_switch_blend_iters)
  if FLAGS.ppo_td3_tau >= 0.0:
    config.ppo_td3_tau = float(FLAGS.ppo_td3_tau)
  if FLAGS.ppo_td3_goal_tol >= 0.0:
    config.ppo_td3_goal_tol = float(FLAGS.ppo_td3_goal_tol)
  config.ppo_td3_use_target_policy = bool(FLAGS.ppo_td3_use_target_policy)
  config.ppo_td3_cross_batch_goals = bool(FLAGS.ppo_td3_cross_batch_goals)
  config.ppo_td3_bilinear = bool(FLAGS.ppo_td3_bilinear)
  if FLAGS.ppo_td3_reward_tau >= 0.0:
    config.ppo_td3_reward_tau = float(FLAGS.ppo_td3_reward_tau)
  config.ppo_td3_log_reward = bool(FLAGS.ppo_td3_log_reward)
  if FLAGS.ppo_gaussian_reward_tau >= 0.0:
    config.ppo_gaussian_reward_tau = float(FLAGS.ppo_gaussian_reward_tau)
  if FLAGS.ppo_fm_reward_tau >= 0.0:
    config.ppo_fm_reward_tau = float(FLAGS.ppo_fm_reward_tau)
  if FLAGS.fm_flow_steps >= 0:
    config.fm_flow_steps = int(FLAGS.fm_flow_steps)
  if str(FLAGS.fm_logp_mode or '').strip():
    config.fm_logp_mode = str(FLAGS.fm_logp_mode).strip().lower()
  if FLAGS.fm_hutch_probes >= 0:
    config.fm_hutch_probes = int(FLAGS.fm_hutch_probes)
  config.fm_layer_norm = bool(FLAGS.fm_layer_norm)
  config.fm_time_embedding = bool(FLAGS.fm_time_embedding)
  if FLAGS.fm_time_embed_dim > 0:
    config.fm_time_embed_dim = int(FLAGS.fm_time_embed_dim)
  if str(FLAGS.fm_ode_solver or '').strip():
    config.fm_ode_solver = str(FLAGS.fm_ode_solver).strip().lower()
  if str(FLAGS.fm_t_sample_mode or '').strip():
    config.fm_t_sample_mode = str(FLAGS.fm_t_sample_mode).strip().lower()
  config.fm_t_logit_loc = float(FLAGS.fm_t_logit_loc)
  config.fm_t_logit_scale = float(FLAGS.fm_t_logit_scale)
  config.fm_goal_noise_std = float(FLAGS.fm_goal_noise_std)
  config.fm_norm_goals = bool(FLAGS.fm_norm_goals)
  config.fm_goal_std_min = float(FLAGS.fm_goal_std_min)
  config.fm_cond_dropout = float(FLAGS.fm_cond_dropout)
  config.fm_reward_clip = float(FLAGS.fm_reward_clip)
  if str(FLAGS.fm_cat_acc_mode or '').strip():
    config.fm_cat_acc_mode = str(FLAGS.fm_cat_acc_mode).strip().lower()
  if FLAGS.fm_cat_acc_subbatch > 0:
    config.fm_cat_acc_subbatch = int(FLAGS.fm_cat_acc_subbatch)
  config.fm_cat_acc_flow_steps = int(FLAGS.fm_cat_acc_flow_steps)
  config.fm_td_mode = bool(FLAGS.fm_td_mode)
  config.fm_td_gamma = float(FLAGS.fm_td_gamma)
  config.fm_td_target_tau = float(FLAGS.fm_td_target_tau)
  config.fm_td_boot_steps = int(FLAGS.fm_td_boot_steps)
  config.fm_logp_diag_interval = int(FLAGS.fm_logp_diag_interval)
  config.fm_logp_diag_batch_size = int(FLAGS.fm_logp_diag_batch_size)
  config.fm_logp_diag_flow_steps = int(FLAGS.fm_logp_diag_flow_steps)
  if str(FLAGS.fm_logp_diag_ode_solver or '').strip():
    config.fm_logp_diag_ode_solver = str(FLAGS.fm_logp_diag_ode_solver).strip().lower()
  config.ppo_dirac_eps = float(FLAGS.ppo_dirac_eps)
  config.nf_rep_size = int(FLAGS.nf_rep_size)
  config.nf_num_blocks = int(FLAGS.nf_num_blocks)
  config.nf_coupling_width = int(FLAGS.nf_coupling_width)
  config.nf_sa_hidden = int(FLAGS.nf_sa_hidden)
  config.nf_sa_num_layers = int(FLAGS.nf_sa_num_layers)
  config.nf_encoder_lr = float(FLAGS.nf_encoder_lr)
  config.nf_critic_lr = float(FLAGS.nf_critic_lr)
  config.nf_critic_weight_decay = float(FLAGS.nf_critic_weight_decay)
  config.nf_grad_clip = float(FLAGS.nf_grad_clip)
  config.nf_noise_std = float(FLAGS.nf_noise_std)
  config.nf_goal_std_min = float(FLAGS.nf_goal_std_min)
  config.nf_mix_env_goal_stats = bool(FLAGS.nf_mix_env_goal_stats)
  config.ppo_skip_first_eval = bool(FLAGS.ppo_skip_first_eval)
  if str(FLAGS.ppo_frozen_reward_ckpt or '').strip():
    config.ppo_frozen_reward_ckpt = str(FLAGS.ppo_frozen_reward_ckpt).strip()
    # Stationary reward: never train CRL / density after loading φ/ψ.
    if int(config.ppo_crl_steps_per_iter) != 0:
      print(f'[ppo_contrastive] frozen reward ckpt set: forcing '
            f'ppo_crl_steps_per_iter {config.ppo_crl_steps_per_iter} -> 0')
      config.ppo_crl_steps_per_iter = 0
  if str(FLAGS.ppo_actor_reset_iters or '').strip():
    config.ppo_actor_reset_iters = str(FLAGS.ppo_actor_reset_iters).strip()
  if FLAGS.ppo_eval_interval >= 0:
    config.ppo_eval_interval = int(FLAGS.ppo_eval_interval)
  if FLAGS.ppo_eval_episodes >= 0:
    config.ppo_eval_episodes = int(FLAGS.ppo_eval_episodes)
  if FLAGS.ppo_video_interval >= 0:
    config.ppo_video_interval = int(FLAGS.ppo_video_interval)
  else:
    ckpt_iv = int(getattr(config, 'ppo_checkpoint_interval', 0) or 0)
    eval_iv = int(getattr(config, 'ppo_eval_interval', 0) or 0)
    config.ppo_video_interval = ckpt_iv if ckpt_iv > 0 else eval_iv
  if FLAGS.ppo_video_fps >= 0:
    config.ppo_video_fps = int(FLAGS.ppo_video_fps)
  config.ppo_skip_first_video = bool(FLAGS.ppo_skip_first_video)
  config.ppo_save_success_checkpoint = bool(FLAGS.ppo_save_success_checkpoint)
  config.ppo_video_include_reward_plot = bool(FLAGS.ppo_video_include_reward_plot)
  config.ppo_video_max_train_success_videos_per_iter = int(FLAGS.ppo_video_max_train_success_videos)
  config.ppo_save_train_success_video = bool(FLAGS.ppo_save_train_success_video)
  config.ppo_train_success_min_interval = int(FLAGS.ppo_train_success_min_interval)


  config.use_wandb = bool(FLAGS.use_wandb)
  config.wandb_project = str(FLAGS.wandb_project)
  config.wandb_entity = str(FLAGS.wandb_entity)
  config.wandb_mode = str(FLAGS.wandb_mode)
  config.wandb_group = str(FLAGS.wandb_group)
  config.ppo_norm_reward = bool(FLAGS.ppo_norm_reward)
  config.ppo_norm_obs = bool(FLAGS.ppo_norm_obs)
  config.ppo_obs_norm_clip = float(FLAGS.ppo_obs_norm_clip)
  config.nf_goal_enc_size = int(FLAGS.nf_goal_enc_size)
  config.ppo_return_norm_window = int(FLAGS.ppo_return_norm_window)
  config.ppo_warmup_percent = float(FLAGS.ppo_warmup_percent)
  config.kde_max_points = int(FLAGS.kde_max_points)
  config.kde_refit_interval = int(FLAGS.kde_refit_interval)
  config.kde_bandwidth = float(FLAGS.kde_bandwidth)
  config.builderbench_episode_length_multiplier = float(FLAGS.builderbench_episode_length_multiplier)
  if FLAGS.max_replay_size >= 0:
    config.max_replay_size = int(FLAGS.max_replay_size)
  if FLAGS.ppo_min_replay_size >= 0:
    config.ppo_min_replay_size = int(FLAGS.ppo_min_replay_size)
  if FLAGS.ppo_success_sample_weight >= 0.0:
    config.ppo_success_sample_weight = float(
        FLAGS.ppo_success_sample_weight)
  config.ppo_use_external_reward = bool(FLAGS.ppo_use_external_reward)
  config.ppo_external_reward_scale = float(FLAGS.ppo_external_reward_scale)
  config.ppo_external_reward_before_norm = bool(
      FLAGS.ppo_external_reward_before_norm)
  if FLAGS.ppo_checkpoint_interval >= 0:
    config.ppo_checkpoint_interval = int(FLAGS.ppo_checkpoint_interval)
  if FLAGS.ppo_checkpoint_keep_last >= 0:
    config.ppo_checkpoint_keep_last = int(FLAGS.ppo_checkpoint_keep_last)
  if FLAGS.ppo_crl_loss_direction.strip():
    config.ppo_crl_loss_direction = FLAGS.ppo_crl_loss_direction.strip().lower()
  if FLAGS.ppo_crl_repr_tau >= 0.0:
    config.ppo_crl_repr_tau = float(FLAGS.ppo_crl_repr_tau)
  if FLAGS.ppo_nf_reward_tau >= 0.0:
    config.ppo_nf_reward_tau = float(FLAGS.ppo_nf_reward_tau)
  config.ppo_log_dormancy = bool(FLAGS.ppo_log_dormancy)
  config.ppo_dormancy_interval = int(FLAGS.ppo_dormancy_interval)
  config.ppo_dormancy_tau = float(FLAGS.ppo_dormancy_tau)
  if FLAGS.hidden_layer_sizes.strip():
    config.hidden_layer_sizes = tuple(
        int(x) for x in FLAGS.hidden_layer_sizes.split(',') if x.strip())

  print(f'[ppo_contrastive] PPO knobs: '
        f'rollout_length={config.ppo_rollout_length}, '
        f'crl_steps_per_iter={config.ppo_crl_steps_per_iter}, '
        f'num_envs={config.ppo_num_envs}, '
        f'num_epochs={config.ppo_num_epochs}, '
        f'num_minibatches={config.ppo_num_minibatches}, '
        f'clip_coef={config.ppo_clip_coef}, '
        f'actor_min_std={config.ppo_actor_min_std}, '
        f'deterministic_select_dim={config.ppo_deterministic_select_dim}, '
        f'categorical_select={config.ppo_categorical_select}, '
        f'ent_coef={config.ppo_ent_coef}, '
        f'anneal_ent_coef={config.ppo_anneal_ent_coef}'
        f'{f"->{config.ppo_ent_coef_final}" if config.ppo_anneal_ent_coef else ""}, '
        f'discount_crl={config.discount}, '
        f'discount_ppo={config.ppo_discount if config.ppo_discount > 0 else config.discount}, '
        f'norm_reward={config.ppo_norm_reward}, '
        f'norm_obs={config.ppo_norm_obs}'
        f'{f"(clip={config.ppo_obs_norm_clip})" if config.ppo_norm_obs else ""}, '
        f'eval_interval={config.ppo_eval_interval}, '
        f'video_interval={config.ppo_video_interval}, '
        f'repr_norm={config.repr_norm}, '
        f'ppo_anneal_lr={config.ppo_anneal_lr}  '
        f'ppo_repr_mode={config.ppo_repr_mode!r}  '
        f'ppo_tdinfonce_crl_warmup_iters={config.ppo_tdinfonce_crl_warmup_iters}  '
        f'ppo_td_infonce_discount={config.ppo_td_infonce_discount}  '
        f'ppo_reward_switch_goal_visits={config.ppo_reward_switch_goal_visits}  '
        f'ppo_reward_switch_blend_iters={config.ppo_reward_switch_blend_iters}  '
        f'ppo_reward_mode={config.ppo_reward_mode!r}  '
        f'ppo_frozen_reward_ckpt={config.ppo_frozen_reward_ckpt!r}  '
        f'ppo_crl_repr_tau={config.ppo_crl_repr_tau}  '
        f'ppo_nf_reward_tau={config.ppo_nf_reward_tau}  '
        f'ppo_gaussian_reward_tau={config.ppo_gaussian_reward_tau}  '
        f'ppo_fm_reward_tau={config.ppo_fm_reward_tau}  '
        f'fm_flow_steps={config.fm_flow_steps}  '
        f'fm_logp_mode={config.fm_logp_mode!r}  '
        f'ppo_td3_tau={config.ppo_td3_tau if config.ppo_td3_tau >= 0 else config.tau}  '
        f'ppo_td3_goal_tol={config.ppo_td3_goal_tol}  '
        f'ppo_td3_use_target_policy={config.ppo_td3_use_target_policy}  '
        f'ppo_td3_cross_batch_goals={config.ppo_td3_cross_batch_goals}  '
        f'ppo_td3_bilinear={config.ppo_td3_bilinear}  '
        f'ppo_td3_reward_tau={config.ppo_td3_reward_tau}  '
        f'ppo_td3_log_reward={config.ppo_td3_log_reward}  '
        f'ppo_dirac_eps={config.ppo_dirac_eps}  '
        f'max_replay_size={config.max_replay_size}  '
        f'ppo_min_replay_size={config.ppo_min_replay_size}  '
        f'ppo_success_sample_weight={config.ppo_success_sample_weight}  '
        f'ppo_use_external_reward={config.ppo_use_external_reward}  '
        f'ppo_external_reward_scale={config.ppo_external_reward_scale}  '
        f'ppo_external_reward_before_norm='
        f'{config.ppo_external_reward_before_norm}  '
        f'kde_max_points={config.kde_max_points}  '
        f'kde_refit_interval={config.kde_refit_interval}  '
        f'kde_bandwidth={config.kde_bandwidth}  '
        f'ckpt_interval={config.ppo_checkpoint_interval}  '
        f'ckpt_keep_last={config.ppo_checkpoint_keep_last} '
        f'({"all milestones" if config.ppo_checkpoint_keep_last <= 0 else "FIFO prune"})  '
        f'hidden_layers={config.hidden_layer_sizes}')

  # ---- Build env factories ----------------------------------------------
  fixed_start_end = (fixed_goal_for_env(env_name)
                     if config.fix_goals else None)
  _hard_goal_print = (
      fixed_start_end if fixed_start_end is not None
      else fixed_goal_for_env(env_name))
  print(f'[ppo_contrastive] hard_goal=\n{_hard_goal_print}')

  # NF push: start episodes with gripper closed (does not affect CRL/Gaussian).
  _env_kwargs = {}
  if (str(config.ppo_repr_mode).strip().lower() == 'nf'
      and env_name == 'sawyer_push'):
    _env_kwargs['nf_closed_gripper_init'] = True
    print('[ppo] sawyer_push NF init: closed gripper at reset')
  if env_name in ('sawyer_bin', 'sawyer_peg'):
    _env_kwargs['randomize_init'] = bool(FLAGS.sawyer_randomize_init)
    if not FLAGS.sawyer_randomize_init:
      print(f'[ppo] {env_name} init: frozen (no MetaWorld object/hole randomness)')
  if env_name == 'sawyer_bin' and FLAGS.bin_randomize_gripper_init:
    if FLAGS.sawyer_randomize_init:
      _env_kwargs['randomize_gripper_init'] = True
      print('[ppo] sawyer_bin init: randomized gripper position at reset')
    else:
      print('[ppo] sawyer_bin init: ignoring --bin_randomize_gripper_init '
            '(frozen by --sawyer_randomize_init=false)')
  if env_name.startswith('builderbench_'):
    _env_kwargs['builderbench_use_pd'] = bool(FLAGS.builderbench_use_pd)
    _env_kwargs['builderbench_pd_duration'] = int(FLAGS.builderbench_pd_duration)
    _env_kwargs['builderbench_episode_length_multiplier'] = _ep_mult
    if FLAGS.builderbench_use_pd and FLAGS.ppo_rollout_length < 0:
      pd_defaults = ppo_env_defaults_for_env(
          env_name, use_pd=True, episode_length_multiplier=_ep_mult)
    _env_kwargs['builderbench_permute_start_boxes'] = bool(
        FLAGS.builderbench_permute_start_boxes)
    if float(FLAGS.builderbench_fixed_start_x) >= 0:
      _env_kwargs['builderbench_fixed_start_x'] = float(
          FLAGS.builderbench_fixed_start_x)
    if int(FLAGS.builderbench_mj_episode_length) > 0:
      _env_kwargs['builderbench_mj_episode_length'] = int(
          FLAGS.builderbench_mj_episode_length)
    # NOTE: this used to silently recompute ppo_rollout_length /
    # ppo_crl_steps_per_iter a second time (duplicating the block above at
    # ~line 476) whenever the flags were left unset. Removed: for PD mode
    # these must always be explicit --ppo_rollout_length /
    # --ppo_crl_steps_per_iter flags now.
    _mj_ep = int(FLAGS.builderbench_mj_episode_length)
    _mj_msg = (
        f' mj_episode_length={_mj_ep}'
        if _mj_ep > 0 else ' mj_episode_length=default')
    _fx = float(FLAGS.builderbench_fixed_start_x)
    _fx_msg = f' fixed_start_x={_fx}' if _fx >= 0 else ''
    print(f'[ppo] builderbench: use_pd={FLAGS.builderbench_use_pd} '
          f'pd_duration={FLAGS.builderbench_pd_duration} '
          f'permute_start_boxes={FLAGS.builderbench_permute_start_boxes}'
          f' episode_length_multiplier={_ep_mult} {_mj_msg}'
          f'{_fx_msg}{_mj_msg}'
          + (' pd_policy_obs=pos+select' if FLAGS.builderbench_use_pd else ''))


  _env_kwargs['obs_space_list'] = [s.strip() for s in FLAGS.obs_space.split(',')]

  def env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=fixed_start_end, **_env_kwargs)
    return env

  def eval_env_factory(s):
    env, _ = contrastive_utils.make_environment(
        env_name, config.start_index, config.end_index, s,
        fixed_start_end=fixed_goal_for_env(env_name), **_env_kwargs)
    return env

  # obs_dim / max_episode_steps inferred from one sample env.
  probe_env, obs_dim = contrastive_utils.make_environment(
      env_name, config.start_index, config.end_index, seed,
      fixed_start_end=fixed_start_end, **_env_kwargs)
  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(probe_env, '_step_limit') + 1
  del probe_env

  # ---- Network factory (adds value_network via networks.py changes) -----
  # NOTE: `actor_min_std` is raised from the shared default (1e-6) to the
  # PPO-specific floor (1e-5 by default; was 0.01 — revisit if collapse /
  # bad exploration reappears).  Without a floor, the tanh-squashed
  # Gaussian policy can collapse to a near-point-mass within a handful of
  # PPO updates — SAC gets away with a 1e-6 floor because adaptive-α
  # actively regulates entropy; PPO has no such control loop and relies
  # on (a) an entropy bonus and (b) a hard std floor to stay exploratory.
  _cat_select_classes = None
  if bool(config.ppo_categorical_select):
    from envs.builderbench_utils import (
        is_builderbench_creative_env,
        parse_sgcrl_builderbench_env_name,
    )
    if not is_builderbench_creative_env(env_name):
      print(f'[ppo_contrastive] ppo_categorical_select ignored for non-'
            f'BuilderBench-creative env {env_name!r} '
            f'(use a creative_* env to enable)')
      config.ppo_categorical_select = False
    else:
      _, _num_cubes, _ = parse_sgcrl_builderbench_env_name(env_name)
      _cat_select_classes = int(_num_cubes)
      print(f'[ppo_contrastive] categorical select actor: '
            f'{_cat_select_classes} cube classes, '
            f'Gaussian on {5 - 1} continuous dims')
  network_factory = functools.partial(
      contrastive.make_networks,
      obs_dim=obs_dim,
      repr_dim=config.repr_dim,
      repr_norm=config.repr_norm,
      twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
      actor_min_std=float(config.ppo_actor_min_std),
      ppo_cleanrl_actor=bool(FLAGS.ppo_cleanrl_actor),
      categorical_select_classes=_cat_select_classes,
  )

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

  # ---- WandB -------------------------------------------------------------
  if FLAGS.use_wandb:
    try:
      import time
      import wandb
      parent_name = os.path.basename(os.path.normpath(config.log_dir))
      folder_name = f'{config.alg_name}_{config.env_name}_{seed}'
      wandb_run_name = f'{parent_name}--{folder_name}' if parent_name else folder_name
      wandb_group = FLAGS.wandb_group or FLAGS.exp_name or parent_name
      wandb_run_id = f'{parent_name}_{folder_name}_{int(time.time())}'.replace('/', '_')
      wandb.init(
          project=FLAGS.wandb_project,
          entity=FLAGS.wandb_entity or None,
          mode=FLAGS.wandb_mode,
          group=wandb_group,
          name=wandb_run_name,
          id=wandb_run_id,
          resume="allow",
          config=run_cfg_payload,
      )
      print(f'[ppo_contrastive] initialized wandb run: {wandb_run_name} (id={wandb_run_id}, mode={FLAGS.wandb_mode})')
    except Exception as _wb_err:
      print(f'[ppo_contrastive] WARNING: Failed to initialize WandB: {_wb_err}')


  from default import make_default_logger
  logger_fn = functools.partial(
      make_default_logger,
      save_dir=run_dir,
      add_uid=config.add_uid,
      steps_key='learner_steps')

  # ---- Go ----------------------------------------------------------------
  checkpoint_dir = os.path.join(run_dir, 'checkpoints')
  _bb_kwargs = None
  if env_name.startswith('builderbench_'):
    _bb_kwargs = dict(_env_kwargs)
    if fixed_start_end is not None:
      _bb_kwargs['fixed_target_goal'] = np.asarray(
          fixed_start_end, dtype=np.float32)

  ppo_learner.run_ppo_training(
      config=config,
      env_factory=env_factory,
      eval_env_factory=eval_env_factory,
      network_factory=network_factory,
      logger_fn=logger_fn,
      total_steps=total_steps,
      seed=seed,
      checkpoint_dir=checkpoint_dir,
      builderbench_kwargs=_bb_kwargs,
      fixed_start_end=fixed_start_end,
  )


if __name__ == '__main__':
  app.run(main)
