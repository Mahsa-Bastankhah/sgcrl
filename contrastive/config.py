"""Contrastive RL config."""
import dataclasses
from typing import Any, Optional, Union, Tuple

from acme import specs
from acme.adders import reverb as adders_reverb
import numpy as onp


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
  hidden_layer_sizes: Tuple[int, Ellipsis] = (256, 256)
  
  # Loss options - entropy
  # Coefficient applied to the entropy bonus. If None, an adaptative
  # coefficient will be used.
  use_action_entropy: bool = True
  entropy_coefficient: Optional[float] = None # note this is alpha
  target_entropy: float = 0.0

  # Replay options
  min_replay_size: int = 10000
  max_replay_size: int = 1000000
  replay_table_name: str = adders_reverb.DEFAULT_PRIORITY_TABLE
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
  # Normalize the φ·ψ reward by the running std of discounted returns
  # (CleanRL's `NormalizeReward` wrapper).  Strongly recommended when the
  # reward comes from a non-stationary learned critic: unscaled φ·ψ values
  # can be O(10) early in training, producing huge advantages and causing
  # the policy to collapse after a few PPO epochs.
  ppo_norm_reward: bool = True
  ppo_anneal_lr: bool = True
  ppo_target_kl: Optional[float] = None
  # Minimum policy std for the PPO actor.  The shared `make_networks`
  # floor is 1e-6 (fine for SAC where adaptive-α controls entropy); PPO
  # has no such mechanism and needs a larger floor (~0.05-0.1) to prevent
  # the tanh-squashed Gaussian from collapsing to a point mass.  Passed
  # through `network_factory` in `ppo_contrastive.py`.
  ppo_actor_min_std: float = 0.01
  # CRL updates per PPO iteration (InfoNCE on φ, ψ over replay).
  ppo_crl_steps_per_iter: int = 64
  # Minimum replay size before CRL updates start.
  ppo_min_replay_size: int = 10_000
  # Checkpointing: save policy/value/CRL params every N PPO iterations.
  # At default settings (8 envs × 128 steps = 1024 env-steps/iter), 100
  # iterations ≈ 100k env steps — light enough not to bottleneck training.
  # Set to 0 or a negative number to disable.
  ppo_checkpoint_interval: int = 500
  # How many milestone checkpoints to keep on disk (older ones are
  # deleted in FIFO order).  `latest.pkl` is always overwritten in place.
  ppo_checkpoint_keep_last: int = 5
  # If True, mix uniformly sampled goals into each CRL replay batch as extra
  # off-diagonal negatives (they are never used as positives).
  uniform_sampling: bool = False
  # Number of uniform negative goals to add per CRL batch.  -1 means
  # batch_size // 2 (the default when uniform_sampling is True).
  uniform_num_negatives: int = -1
  # When 0 < tau < 1, a separate EMA-averaged "slow" copy of q_params
  # (ema <- tau*ema + (1-tau)*online after each CRL step) is used only to
  # compute the PPO reward, while the "fast"/online q_params keeps training
  # on the CRL loss every step. This is target-network-style stabilization
  # for PPO on top of a non-stationary learned reward. 0 (default) disables
  # this: reward reads the live q_params directly.
  ppo_crl_repr_tau: float = 0.0
  # If True, add the sparse extrinsic success reward (1 if the goal has
  # been reached, else 0) on top of the φ·ψ representation reward before
  # GAE. Only has an effect for maniskill_close_subtask_train /
  # maniskill_open_subtask_train (uses success_key='drawer_closed'/
  # 'drawer_open' instead of the default 'success' for the extra reward
  # term -- see ManiskillVecEnv). False (default) leaves the reward purely
  # representation-based, unchanged from prior behavior.
  ppo_crl_add_extrinsic_reward: bool = False
  # If True (maniskill_* env_names only), collect PPO rollouts from a
  # single ManiSkill env simulating all ppo_num_envs copies at once on the
  # GPU (PhysX GPU backend), instead of ppo_num_envs independent single-env
  # SAPIEN scenes stepped one at a time in a Python loop. Pure wall-clock
  # throughput optimization -- same rollouts, same learning dynamics.
  # Requires a CUDA GPU; ManiSkill raises at construction time otherwise.
  ppo_maniskill_native_vec: bool = False
  # Normalize observations (state + goal) per-dimension using an online
  # running mean/std (Welford, see `ppo_learner.ObsNormalizer`), so raw
  # feature dims with very different natural scales -- e.g. position in
  # meters vs. a [0, 1] boolean grasp flag -- don't dominate/vanish in the
  # policy/value/CRL encoders purely due to magnitude. Goal columns reuse
  # the SAME per-column stats as their corresponding state column (goal
  # column j <-> state column start_index+j), so HER-relabeled training
  # goals (literal future-state slices) and env-provided rollout goals
  # (e.g. ManiskillOpenCabinetDrawer's chased handle_target_pos) stay on a
  # consistent scale instead of drifting apart under independently-fit
  # statistics. Mirrors `ppo_norm_reward`'s always-on-by-default precedent.
  ppo_norm_obs: bool = True

  # -------------------------------------------------------------------------
  # Density estimator selector for the PPO shaped reward + off-policy
  # critic/density update (standalone PPO only, reward_shaping_mode='ppo').
  #   'crl' (default) — φ(s,a)·ψ(g) contrastive representations (InfoNCE);
  #                     100% unchanged behavior from before this field existed.
  #   'nf'            — conditional RealNVP density log p_NF(g|s,a), ported
  #                     from contrastive/nf_density.py (origin/new-builderbench).
  # -------------------------------------------------------------------------
  ppo_repr_mode: str = 'crl'

  # NF-specific options (only read when ppo_repr_mode == 'nf'). Minimal
  # subset of contrastive/nf_density.py's knobs; nf_density.py supports
  # other features (TD-NF, gradient regularizer, coupling-scale tanh,
  # reward-mode variants, mask_prob, s-perturbation, goal encoder)
  # intentionally not exposed here -- they keep nf_density.py's function
  # defaults (effectively off) since we never pass overrides.
  nf_rep_size: int = 256
  nf_num_blocks: int = 12
  nf_coupling_width: int = 512
  nf_sa_hidden: int = 1024
  nf_sa_num_layers: int = 4
  nf_goal_enc_size: int = 0
  nf_encoder_lr: float = 3e-4
  nf_critic_lr: float = 1e-4
  nf_critic_weight_decay: float = 1e-6
  nf_grad_clip: float = 1.0
  nf_noise_std: float = 0.05
  nf_goal_std_min: float = 0.1
  # If True, the goal-normalization stats (nf_goal_mean/nf_goal_std) are
  # computed from batch_size replay hindsight goals PLUS batch_size copies
  # of the current rollout's actual env task goal, so the task goal is
  # in-distribution for the normalizer instead of relying solely on
  # hindsight-relabeled goals. Stabilizes r = log p_NF(g_task|s,a) early in
  # training, when the task goal can otherwise be far out-of-distribution
  # relative to the replay's hindsight goals and blow up the flow's density.
  nf_mix_task_goal_stats: bool = False
  # EMA decay tau for NF params used in the PPO reward r = log p_NF(g|s,a).
  # Mirrors ppo_crl_repr_tau (CRL-only); the two are never both active.
  ppo_nf_reward_tau: float = 0.0

  # -------------------------------------------------------------------------
  # PPO+RND agent (`ppo_rnd.py` / `contrastive/ppo_rnd_learner.py`). Separate
  # standalone entrypoint from `ppo_contrastive.py` -- no CRL critic/replay;
  # reward is sparse extrinsic (evaluator success) + RND intrinsic. Reuses
  # the `ppo_*` fields above (num_envs, rollout_length, clip_coef, ent_coef,
  # actor_min_std, discount, gae_lambda, anneal_lr, max_grad_norm, norm_obs,
  # checkpoint_interval, maniskill_native_vec) and `hidden_layer_sizes`.
  # -------------------------------------------------------------------------
  # Weight on the (normalized) intrinsic advantage in the combined PPO
  # advantage: advantages = rnd_ext_coef*ext_adv + rnd_int_coef*int_adv.
  rnd_int_coef: float = 1.0
  rnd_ext_coef: float = 1.0
  # Discount used for the intrinsic-reward GAE/return stream only; the
  # extrinsic stream uses `ppo_discount` (falls back to `discount`), same as
  # the CRL agent.
  rnd_int_discount: float = 0.99
  # Output width of the RND predictor/target networks' final layer.
  rnd_output_size: int = 256

  use_image_obs: bool = False
  random_goals: float = 0.5
  jit: bool = True
  add_mc_to_td: bool = False
  resample_neg_actions: bool = False
  
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
