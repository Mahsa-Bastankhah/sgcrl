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
