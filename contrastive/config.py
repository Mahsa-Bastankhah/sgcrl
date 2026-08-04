"""Contrastive RL config."""
import dataclasses
from typing import Any, Optional, Union, Tuple

from acme import specs
import numpy as onp

_DEFAULT_PRIORITY_TABLE = 'priority_table'


@dataclasses.dataclass
class ContrastiveConfig:
  """Configuration options for contrastive RL."""
  add_uid: bool = True
  time_delta_minutes: int = 5
  log_dir: str = 'logs/'
  env_name: str = ''
  alg_name: str = ''
  seed: int = 0
  max_number_of_steps: int = 8_000_000
  num_actors: int = 4

  # env options
  fix_goals: bool = False
    
  # Loss options
  batch_size: int = 256
  actor_learning_rate: float = 3e-4
  learning_rate: float = 3e-4
  reward_scale: float = 1
  discount: float = 0.99
  n_step: int = 1
  # Target smoothing coefficient.
  tau: float = 0.005
  hidden_layer_sizes: Tuple[int, Ellipsis] = (256, 256, 256, 256, 256, 256)
  
  # Loss options - entropy
  # Coefficient applied to the entropy bonus. If None, an adaptative
  # coefficient will be used.
  use_action_entropy: bool = True
  entropy_coefficient: Optional[float] = None # note this is alpha
  target_entropy: float = 0.0

  # Replay options
  min_replay_size: int = 10000
  max_replay_size: int = 1000000
  replay_table_name: str = _DEFAULT_PRIORITY_TABLE
  prefetch_size: int = 4
  num_parallel_calls: Optional[int] = 4
  samples_per_insert: float = 256
  # Rate to be used for the SampleToInsertRatio rate limitter tolerance.
  # See a formula in make_replay_tables for more details.
  samples_per_insert_tolerance_rate: float = 0.1
  num_sgd_steps_per_step: int = 64  # Gradient updates to perform per step.
  
  # training options
  no_repr: bool = False
  repr_dim: Union[int, str] = 64  # Size of representation.
  use_random_actor: bool = True  # Initial with uniform random policy.
  repr_norm: bool = False
  use_cpc: bool = False
  local: bool = False  # Whether running locally. Disables eval.
  use_td: bool = False
  twin_q: bool = False
  use_kappa: bool = False   # Train κ(s,a): discounted-sum-of-φ value network.
  twin_kappa: bool = False  # Two independent κ networks + targets; actor uses min(κ1,κ2)·ψ.
  # If True (reward_shaping_mode='kappa' only): in the actor loss, rescale each κ
  # row so ‖κ‖_2 = ‖φ‖_2 using φ = sa_repr from the same critic forward as ψ,
  # before κ·ψ.  Independent of repr_norm (which L2-normalizes φ, ψ inside the
  # critic).  κ Bellman regression still uses raw κ vs raw φ.
  kappa_actor_match_phi_norm: bool = False
  # Separate discount and LR for κ.  Rationale:
  #   - κ* has ‖κ*‖ ≤ (max ‖φ‖) / (1 - γ_κ); at γ=0.99 that's 100·‖φ‖ which
  #     is a huge distance for a bootstrapped net to travel while φ itself
  #     is still moving.  γ_κ = 0.95 caps the fixed-point norm at 20·‖φ‖.
  #   - κ bootstraps off itself (non-stationary target) and its loss is
  #     averaged over a repr_dim=64 vector; a smaller LR than the critic's
  #     is standard for any bootstrapped value network in this regime.
  discount_kappa: float = 0.95
  learning_rate_kappa: float = 1e-4
  # Global-norm clip on κ (0 disables); same role as q_repr_max_grad_norm.
  kappa_max_grad_norm: float = 1.0
  # Goal-conditioned scalar Q(s,a,g) trained TD3-style on r = sg(φ(s,a)·ψ(g)).
  # Enabled by the `q_sac` launchpad alg (see lp_contrastive.py) and set by
  # reward_shaping_mode='q'.  The reward signal is the same φ·ψ product
  # used by the κ route, but here it enters directly as a scalar reward and
  # Q is a plain scalar (not a vector in R^repr_dim), so there's no
  # high-dim bootstrap like κ — much easier to stabilize.
  use_q_repr: bool = False
  twin_q_repr: bool = True
  # Rationale for the γ_q / LR defaults mirrors κ: with repr_norm=True the
  # reward is bounded in [-1, 1], so Q* ≤ 1/(1 - γ_q); γ_q=0.95 caps it at
  # 20.  A smaller LR than the critic's is standard for any bootstrapped
  # target.  Raise γ_q toward 0.99 once you see stable training.
  discount_q: float = 0.95
  learning_rate_q: float = 3e-4
  # Gradient-norm clip on the Q optimizer (stabilises TD3-style training
  # against occasional large targets early on).  0 or negative disables.
  q_repr_max_grad_norm: float = 1.0
  # Deprecated toggle (kept for backward compatibility). Hard-goal logic is
  # now automatically applied whenever use_q_repr or use_kappa is enabled and
  # `hard_goal` is provided.
  q_repr_reward_use_hard_goal: bool = False
  # Flat goal vector from lp_contrastive fixed_goal_dict; used for hard-goal
  # conditioning in reward_shaping_mode `q` / `kappa` paths and repr observer logging.
  hard_goal: Any = None
  # reward_shaping_mode='kappa': add -coef·(κ·ψ)(s, a, hard_goal) to the actor
  # loss (same HER state `s`, sampled `a`).  Ignored if hard_goal is None; 0 disables.
  kappa_actor_hard_goal_coef: float = 0.0
  # Trust-region on the actor for reward_shaping_mode in {'q', 'kappa'}:
  #     loss += β · mean[log π_new(a|·) - log π_prev(a|·)]
  # at the same sampled a; `π_prev` is stop-gradiented.  For `q`, conditioning
  # matches `actor_loss_q` (replay obs, possibly hard-goal); for `kappa` it
  # matches the HER-shaped `new_obs` used for κ·ψ.  β = 0 disables.
  q_actor_kl_to_prev_coef: float = 0.0
  # HER-relabeled auxiliary when reward_shaping_mode='q'.  Mirrors the stock
  # `actor_loss` pattern: doubles the policy batch via `config.random_goals`
  # (same shuffling scheme) and adds
  #     loss += λ · mean_{i}[ α·log π(a_i | s_i, g'_i)  -  diag(CRL)(s_i, a_i, g'_i) ]
  # where CRL is the contrastive critic with `q_params` stop-gradiented
  # (policy gradients only).  g'_i are either the original goals or
  # random future-state goals from other trajectories in the batch,
  # matching `random_goals` ∈ {0, 0.5, 1}.  0 disables.
  #
  # Intuition: the bootstrapped Q is specialized to the goals it was
  # trained on, but π(a|s, g) needs to generalize across all goals.
  # The CRL critic already scores any (s,a,g) pair, so this gives the
  # actor a free auxiliary loss on relabeled goals without changing the
  # bootstrapped Q objective.
  #
  # Scale note: the bootstrapped Q saturates near 1/(1-γ_q) ≈ 20, while
  # the CRL diagonal logit is typically O(1).  λ = 1.0 therefore weighs
  # the auxiliary ~20× less than the main term in absolute scale.
  # Raise λ (e.g. 5-20) if you want the auxiliary to carry more weight.
  q_actor_her_aux_coef: float = 0.0
  # Repr-based reward shaping (which actor objective branch runs):
  #   ''       = disabled (stock CRL actor on the critic's logits).
  #   'kappa'  = actor maximises min(κ1,κ2)·ψ (needs use_kappa=True); same
  #              adaptive-α pipeline as stock CRL (log_alpha / target_entropy).
  #   'q'      = actor maximises min(Q1,Q2); Q nets trained TD3-style on
  #              r = sg(φ·ψ) (needs use_q_repr=True).
  #   'ppo'    = only used by standalone PPO (ppo_contrastive.py), not Launchpad.
  reward_shaping_mode: str = ''
  # For reward_shaping_mode='kappa', the actor still uses the shared adaptive-α
  # pipeline (log_alpha, alpha_loss, alpha_optimizer) driven by `target_entropy`
  # — same machinery as stock CRL.  Raise `target_entropy` if you want a more
  # entropic policy on κ·ψ.

  # -------------------------------------------------------------------------
  # PPO-on-φ·ψ actor (reward_shaping_mode='ppo').  These are only read by the
  # standalone PPO training loop in ppo_contrastive.py / ppo_learner.py.
  # -------------------------------------------------------------------------
  ppo_num_envs: int = 8
  ppo_rollout_length: int = 128       # T: steps per env per iteration
  ppo_num_epochs: int = 10            # PPO update epochs over each rollout batch
  ppo_num_minibatches: int = 4        # Minibatches per epoch
  ppo_clip_coef: float = 0.2
  ppo_vf_coef: float = 0.5
  ppo_ent_coef: float = 0.05
  # PPO-only discount used for GAE/returns in standalone PPO.  If <=0,
  # `config.discount` is used (backward-compatible behavior).
  ppo_discount: float = -1.0
  ppo_gae_lambda: float = 0.95
  ppo_max_grad_norm: float = 0.5
  ppo_clip_vloss: bool = True
  ppo_norm_adv: bool = True
  # Optional observation normalization for standalone PPO. Running statistics
  # cover raw state dimensions only; goal coordinates reuse the corresponding
  # state statistics selected by start_index:end_index.
  ppo_norm_obs: bool = False
  ppo_obs_norm_clip: float = 10.0
  # Normalize the φ·ψ reward by the running std of discounted returns
  # (CleanRL's `NormalizeReward` wrapper).  Strongly recommended when the
  # reward comes from a non-stationary learned critic: unscaled φ·ψ values
  # can be O(10) early in training, producing huge advantages and causing
  # the policy to collapse after a few PPO epochs.
  ppo_norm_reward: bool = True
  ppo_anneal_lr: bool = True
  # If True, linearly anneal ppo_ent_coef -> ppo_ent_coef_final over all PPO
  # SGD updates (mirrors ppo_anneal_lr's schedule). Off by default so existing
  # runs keep a fixed entropy bonus unless explicitly opted in.
  ppo_anneal_ent_coef: bool = False
  ppo_ent_coef_final: float = 0.0
  ppo_target_kl: Optional[float] = None
  # Good Experience Replay & Bootstrapping
  ppo_use_good_buffer: bool = False
  ppo_good_buffer_mode: str = 'sil'  # 'sil' (Approach 1) | 'mixed' (Approach 2)
  ppo_good_buffer_min_cubes: int = 2  # Min cubes stacked (2, 3, 4) to store trajectory
  ppo_good_buffer_coef: float = 0.1  # SIL loss weight (or mix ratio for 'mixed' mode)
  ppo_good_buffer_max_size: int = 50_000  # Max transitions in Good Experience Buffer
  # Minimum policy std for the PPO actor.  The shared `make_networks`
  # floor is 1e-6 (fine for SAC where adaptive-α controls entropy); PPO
  # has no such mechanism and historically used a larger floor (~0.01)
  # to prevent tanh-Gaussian collapse.  Currently 1e-5 — revisit if
  # policy collapse / bad exploration shows up again.  Passed through
  # `network_factory` in `ppo_contrastive.py`.
  ppo_actor_min_std: float = 1e-5
  # ppo_actor_min_std: float = 0.01 [OLD: ASK why removed?]
  # If True, force the last action dim (BuilderBench PD `select_action`) to
  # use μ / mode only: policy std on that dim is deactivated for sampling,
  # log-prob, and entropy. Other action dims keep their stochastic policy.
  ppo_deterministic_select_dim: bool = False
  # If True (default), use a hybrid actor: shared policy trunk, tanh-Gaussian
  # mean/std on continuous action dims, and a categorical logits head over
  # num_cubes classes for the select dim (cube ids 0..n-1).  BuilderBench
  # creative only; ignored on other envs.  Pass --noppo_categorical_select.
  ppo_categorical_select: bool = True
  # CRL updates per PPO iteration (InfoNCE on φ, ψ over replay).
  ppo_crl_steps_per_iter: int = 64
  # InfoNCE direction for PPO-CRL. 'forward': fix anchor sᵢ, vary goal gⱼ
  # (standard). 'backward': fix goal gᵢ, vary anchor sⱼ (logits transposed).
  ppo_crl_loss_direction: str = 'forward'  # 'forward' | 'backward'
  # EMA decay τ for φ, ψ used in the PPO reward r = φ·ψ (CRL mode only).
  # Reward uses EMA params: ema ← τ·ema + (1−τ)·online after each CRL step.
  # τ=0 uses online params directly (no EMA).  Higher τ = slower / smoother reward.
  ppo_crl_repr_tau: float = 0.0
  # EMA decay τ for NF params used in the PPO reward r = log p_NF(g|s,a)
  # (NF mode only).  Same update: ema ← τ·ema + (1−τ)·online after each NF
  # density step.  τ=0 uses online NF params (default).  Independent of
  # ppo_crl_repr_tau.
  ppo_nf_reward_tau: float = 0.0
  # EMA decay τ for Gaussian density params used in PPO reward
  # r = log p_θ(g|s,a) (gaussian mode only).  Same update as NF:
  # ema ← τ·ema + (1−τ)·online after each density step.  τ=0 uses online
  # params (default).  Independent of ppo_crl_repr_tau / ppo_nf_reward_tau.
  ppo_gaussian_reward_tau: float = 0.0
  # EMA decay τ for flow-matching density params used in PPO reward
  # r = log p_FM(g|s,a) (fm mode only).  Same update as Gaussian/NF:
  # ema ← τ·ema + (1−τ)·online after each density step.  τ=0 uses online
  # params (default).
  ppo_fm_reward_tau: float = 0.0
  # Minimum replay size before CRL updates start.
  ppo_min_replay_size: int = 10_000
  # CRL replay episode sampling weight for successful trajectories.
  # Unsuccessful episodes always have weight 1. Episode indices are drawn
  # with probability w_k / sum_i w_i. Default 1.0 recovers uniform sampling.
  ppo_success_sample_weight: float = 1.0
  # If True, add `ppo_external_reward_scale` to the PPO reward on steps where
  # hard success fires: BuilderBench metrics['success'], or Sawyer MetaWorld
  # sparse env reward (>=0.5). By default applied after shaped-reward
  # computation (CRL/NF/…) and after reward normalisation so the bonus is not
  # washed out by return-std scaling. Set
  # `ppo_external_reward_before_norm=True` to add the bonus to the raw shaped
  # reward before return-norm instead.
  ppo_use_external_reward: bool = False
  ppo_external_reward_scale: float = 100.0
  ppo_external_reward_before_norm: bool = False
  # Checkpointing: save policy/value/CRL params every N PPO iterations.
  # At default settings (8 envs × 128 steps = 1024 env-steps/iter), 100
  # iterations ≈ 100k env steps — light enough not to bottleneck training.
  # Set to 0 or a negative number to disable.
  ppo_checkpoint_interval: int = 200
  # How many milestone ckpt_iter_*.pkl files to keep (FIFO prune of oldest).
  # 0 = keep all milestones (no pruning).  `latest.pkl` is always overwritten.
  ppo_checkpoint_keep_last: int = 0
  # If True, mix 50% uniformly sampled goals into each CRL replay batch so
  # that half the in-batch negatives come from the uniform goal distribution
  # rather than the replay future-state distribution.
  uniform_sampling: bool = False
  # PPO reward baseline (standalone PPO only).  '' = r = φ(s,a)·ψ(g).
  # 'dirac_target': for s ≠ g, r = log(eps) − φ(s0,a)·ψ(s); at s = g,
  # r = −φ(s0,a)·ψ(g).  s0 is the episode initial state; a ~ π(·|s).
  # 'kde_dirac': same formula but CRL dot products replaced by Gaussian KDE
  # log-densities estimated from the replay buffer.
  ppo_reward_mode: str = ''
  ppo_dirac_eps: float = 1e-6
  # Density estimator used for the PPO shaped reward.
  #   'crl'      (default) — φ(s,a)·ψ(g) contrastive representations.
  #   'gaussian' — diagonal Gaussian  p_θ(g|s); reward = log p_θ(g|s_t).
  #   'nf'       — conditional RealNVP  log p_NF(g|s,a); reward = log p_NF.
  #   'fm'       — OT flow-matching velocity field; reward = log p_FM(g|s,a)
  #                via reverse ODE (FAC-style); train on CRL future goals.
  #   'td3'      — twin Q(s,a,s_f) with TD3 backup on r=1{s≈s_f};
  #                PPO reward = Q1(s,a,g) (or log((1−γ)Q) if
  #                ppo_td3_log_reward).  Target Polyak uses ppo_td3_tau
  #                (falls back to `tau`); NOT ppo_crl_repr_tau.
  #   'crl_td3_switch' — opt-in hybrid: train CRL and TD3 together, use the
  #                CRL reward until `ppo_reward_switch_goal_visits` completed
  #                training episodes have visited the hard goal, then use only
  #                the TD3 reward starting on the following PPO iteration.
  #   'tdinfonce' — TD InfoNCE bilinear φ(s,a)·ψ(s_f) critic (Zheng et al.);
  #                PPO reward is still r = φ·ψ with ppo_crl_repr_tau EMA.
  #                Target-critic EMA keep-rate uses ppo_td_infonce_target_tau
  #                (independent of the reward EMA).
  ppo_repr_mode: str = 'crl'
  # Keep-rate for TD-InfoNCE target critic:
  #   target ← τ·target + (1−τ)·online.  Default 0.995 (slow target).
  ppo_td_infonce_target_tau: float = 0.995
  # Mix weight for TD-InfoNCE critic loss L = (1-γ) L_InfoNCE + γ L_TD.
  # <0 → use config.discount (shared with HER geometric sampling).
  # Set ≈0 to sanity-check term 1 only without changing HER γ.
  ppo_td_infonce_discount: float = -1.0
  # Coefficient on logsumexp(logits)^2 added to TD-InfoNCE softmax CE
  # (same default 0.01 as CPC/CRL).  Set 0 to disable.
  ppo_td_infonce_logsumexp_coef: float = 0.01
  # If >0 with ppo_repr_mode=tdinfonce: train φ,ψ with standard CRL InfoNCE
  # for this many PPO iterations, then switch critic updates to TD-InfoNCE
  # (same params).  PPO reward stays r=φ·ψ throughout.  0 = TD-InfoNCE from
  # the first critic update (default).
  ppo_tdinfonce_crl_warmup_iters: int = 0
  # Goal-visit threshold for ppo_repr_mode='crl_td3_switch'. Values <= 0 are
  # invalid in that mode. This does not affect any other representation mode.
  ppo_reward_switch_goal_visits: int = 5
  # After the visit threshold, blend PPO rewards for this many iterations:
  #   r = (1-w)·r_CRL + w·r_TD3,  w = clip((iter - switch_iter) / blend_iters, 0, 1)
  # At switch_iter, w=0 (pure CRL); after blend_iters, w=1 (pure TD3).
  # 0 = hard switch to TD3 on the first post-threshold iteration (legacy).
  ppo_reward_switch_blend_iters: int = 0
  # Polyak τ for TD3 target Q networks when ppo_repr_mode='td3'.
  # Independent of ppo_crl_repr_tau (CRL reward EMA).  <0 → use `tau`.
  ppo_td3_tau: float = -1.0
  # Goal-hit tolerance for the sparse indicator 1{‖obs_to_goal(s')−g‖ < tol}.
  ppo_td3_goal_tol: float = 1e-2
  # If True, TD3 backup samples a' from a Polyak target policy π̄
  # (same τ as Q targets) instead of the online PPO policy.  Default off.
  ppo_td3_use_target_policy: bool = False
  # If True, each (s_i,a_i,s'_i) is trained vs every batch goal g_j
  # (B² TD backups).  If False (default), only the paired g_i is used.
  ppo_td3_cross_batch_goals: bool = False
  # If True, Q(s,a,g)=x(s,a)·y(g) with x/y matching CRL φ/ψ; else MLP([s;g;a]).
  ppo_td3_bilinear: bool = False
  # EMA decay τ for TD3 Q params used in the PPO reward r = Q1(s,a,g).
  # Same update as NF: ema ← τ·ema + (1−τ)·online after each critic step.
  # τ=0 uses online Q1 (default).  Independent of ppo_td3_tau (Polyak targets).
  ppo_td3_reward_tau: float = 0.0
  # If True, PPO reward is log((1−γ)·max(Q1, ε)) instead of raw Q1
  # (log-occupancy scale, comparable to Gaussian/NF log p).
  ppo_td3_log_reward: bool = False
  # Flow-matching options (only used when ppo_repr_mode == 'fm').
  fm_flow_steps: int = 10          # Euler steps for sample / reverse-ODE logp
  fm_logp_mode: str = 'exact'      # 'exact' | 'hutch-rade' | 'hutch-gaus'
  fm_hutch_probes: int = 8         # Hutchinson probes when using hutch-* modes
  fm_layer_norm: bool = False      # LayerNorm inside velocity MLP (FAC-style)
  fm_time_embedding: bool = False  # Fourier sinusoidal time embedding for t
  fm_time_embed_dim: int = 32      # Dim for Fourier time embedding
  fm_ode_solver: str = 'euler'     # ODE solver: 'euler' | 'heun'
  fm_t_sample_mode: str = 'uniform'# Timestep sampling: 'uniform' | 'logit_normal'
  fm_t_logit_loc: float = 0.0      # Logit-Normal mean
  fm_t_logit_scale: float = 1.0    # Logit-Normal std scale
  fm_goal_noise_std: float = 0.0   # Gaussian noise added to future goals s_f during training (0 = disabled)
  fm_norm_goals: bool = False      # Normalize goal vectors s_f to unit variance before flow estimation
  fm_td_mode: bool = False         # If True, use TD-Flow (Bellman probability path targets)
  fm_td_gamma: float = 0.99        # Discount factor gamma for TD-Flow Bellman target mixture
  fm_td_target_tau: float = 0.005  # Polyak EMA soft update rate for target vector field v_phi
  fm_td_boot_steps: int = 1        # ODE steps for train-time bootstrapped target state generation
  # NF-specific options (only used when ppo_repr_mode == 'nf').
  nf_rep_size: int = 256       # SA encoder output dim (conditioning vector)
  nf_num_blocks: int = 12      # number of affine coupling blocks
  nf_coupling_width: int = 512  # width of s/t sub-networks inside each block
  nf_sa_hidden: int = 1024     # SA encoder hidden width (reference: 1024)
  nf_sa_num_layers: int = 4    # SA encoder depth (reference: 4)
  nf_encoder_lr: float = 3e-4   # Adam lr for SA encoder (ref: actor_lr)
  nf_critic_lr: float = 1e-4    # AdamW lr for RealNVP flow (ref: critic_lr)
  nf_critic_weight_decay: float = 1e-6  # AdamW wd for RealNVP (ref)
  nf_grad_clip: float = 1.0     # global-norm gradient clipping for NF (0 = disabled)
  nf_noise_std: float = 0.05   # Gaussian noise added to goals during NF training (0 = disabled)
  nf_goal_std_min: float = 0.1  # floor on per-dim replay std (avoids blow-ups on static dims)
  nf_mix_env_goal_stats: bool = False  # also include rollout env goals when computing NF normalisation stats
  nf_goal_enc_size: int = 0     # >0 enables goal encoder (maps goal → goal_enc_size-dim latent)
  ppo_return_norm_window: int = 0  # >0 caps effective count in return normalizer (soft sliding window)
  nf_no_norm_goal_dims: tuple = ()  # deprecated; unused (running stats + std floor only)
  nf_goal_norm_low: Optional[Any] = None   # deprecated; unused
  nf_goal_norm_high: Optional[Any] = None  # deprecated; unused
  ppo_skip_first_eval: bool = False  # skip logging the iteration-0 eval (avoids artificially high checkpoint result)
  # Path to a pretrained PPO/CRL checkpoint (.pkl). On a fresh run (no resume),
  # load only φ/ψ (preferring q_params_ema when present) and use r=φ(s,a)·ψ(g)
  # as a *stationary* reward. Policy/value are freshly initialized. Forces
  # ppo_crl_steps_per_iter=0 so representations never update. Empty = disabled.
  ppo_frozen_reward_ckpt: str = ''
  # Comma-separated PPO iterations at which to force an actor reinit
  # (in addition to the short-episode guard). Empty = schedule disabled.
  ppo_actor_reset_iters: str = ''
  ppo_eval_interval: int = 10  # run eval every N PPO iterations (0 = disabled)
  ppo_eval_episodes: int = 5  # number of eval episodes per eval round
  # In-train BuilderBench video (deterministic policy + live obs_rms).
  # 0 = disabled. Videos written under <run_dir>/videos/.
  ppo_video_interval: int = 0
  ppo_video_fps: int = 10
  ppo_skip_first_video: bool = True
  # KDE options (only used when ppo_reward_mode == 'kde_dirac').
  # kde_max_points: number of replay states to fit the KDE on.
  # kde_refit_interval: refit the KDE every N PPO iterations (1 = every iter).
  # kde_bandwidth: fixed bandwidth; 0.0 = use Scott's rule automatically.
  kde_max_points: int = 2000
  kde_refit_interval: int = 1
  kde_bandwidth: float = 0.0


  use_image_obs: bool = False
  random_goals: float = 0.5
  jit: bool = True
  add_mc_to_td: bool = False
  resample_neg_actions: bool = False

  # Observation Normalization & Scaling options
  obs_norm_mode: str = 'none'         # Options: 'none', 'z_scale', 'tied_rsnorm'
  z_scale_multiplier: float = 3.0     # Multiplier k for z_scale mode (z_scaled = z * k)
  rsnorm_clip: float = 10.0           # Clipping bound for tied_rsnorm mode
  
  # Parameters that should be overwritten, based on each environment.
  obs_dim: int = -1
  max_episode_steps: int = -1
  start_index: int = 0
  end_index: int = -1

  def __post_init__(self):
    # Map legacy mode strings to current vocabulary.
    legacy = {'q_sac': 'q', 'sac': 'kappa'}
    m = self.reward_shaping_mode
    if m in legacy:
      object.__setattr__(self, 'reward_shaping_mode', legacy[m])


def target_entropy_from_env_spec(
    spec,
    target_entropy_per_dimension = None,
):
  """A heuristic to determine a target entropy.

  If target_entropy_per_dimension is not specified, the target entropy is
  computed as "-num_actions", otherwise it is
  "target_entropy_per_dimension * num_actions".

  Args:
    spec: environment spec
    target_entropy_per_dimension: None or target entropy per action dimension

  Returns:
    target entropy
  """

  def get_num_actions(action_spec):
    """Returns a number of actions in the spec."""
    if isinstance(action_spec, specs.BoundedArray):
      return onp.prod(action_spec.shape, dtype=int)
    elif isinstance(action_spec, tuple):
      return sum(get_num_actions(subspace) for subspace in action_spec)
    else:
      raise ValueError('Unknown action space type.')

  num_actions = get_num_actions(spec.actions)
  if target_entropy_per_dimension is None:
    if not isinstance(spec.actions, specs.BoundedArray) or isinstance(
        spec.actions, specs.DiscreteArray):
      raise ValueError('Only accept BoundedArrays for automatic '
                       f'target_entropy, got: {spec.actions}')
    if not onp.all(spec.actions.minimum == -1.):
      raise ValueError(
          f'Minimum expected to be -1, got: {spec.actions.minimum}')
    if not onp.all(spec.actions.maximum == 1.):
      raise ValueError(
          f'Maximum expected to be 1, got: {spec.actions.maximum}')

    return -num_actions
  else:
    return target_entropy_per_dimension * num_actions
