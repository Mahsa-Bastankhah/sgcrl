"""Weighted Behavioural Cloning (WBC) learner on CRL representations.

Objective
---------
    max_π  E_{(s,a,g) ~ π_old}[  φ(s,a)·ψ(g)  ·  log π(a|s)  ]

The policy is updated via weighted maximum-likelihood over fresh on-policy
rollouts: actions with high CRL reward φ(s,a)·ψ(g) are reinforced
proportionally.  The CRL networks φ/ψ are trained off-policy from the same
replay buffer as in ppo_learner.py using the same InfoNCE loss.

Compared to PPO
  – No value function or GAE.  Weights come directly from CRL.
  – No importance-ratio clipping.
  – Simple update: weighted-MLE gradient step + optional entropy bonus.
  – Multiple epochs over each rollout are allowed (no IS ratio needed,
    but the data is slightly off-policy after the first epoch – works fine
    empirically for the short rollouts used here).
"""
import os
import time
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from acme.jax import networks as networks_lib

from contrastive import config as contrastive_config
from contrastive import networks as contrastive_networks

# Reuse utilities that don't depend on the PPO objective.
from contrastive.ppo_learner import (
    EpisodeReplay,
    VecEnv,
    _prune_old_checkpoints,
    _save_checkpoint,
    _truncate_csv_to_iteration,
    load_checkpoint,
    make_crl_update_fn,
)


# ---------------------------------------------------------------------------
# WBC policy update
# ---------------------------------------------------------------------------
def make_wbc_update_fn(
    networks: contrastive_networks.ContrastiveNetworks,
    config: contrastive_config.ContrastiveConfig,
    policy_optimizer: optax.GradientTransformation,
):
    """Returns a jitted one-minibatch WBC policy update.

    Loss:
        L = -mean( w_norm * log π(a|s) )  -  ent_coef * H(π)

    where w_norm = w / (std(w) + eps) centres and scales the raw CRL weights
    φ(s,a)·ψ(g) so that their magnitude is independent of the CRL network
    scale.  A small positive shift can be applied via config.wbc_weight_offset.
    """
    ent_coef = float(getattr(config, 'ppo_ent_coef', 0.0))
    weight_offset = float(getattr(config, 'wbc_weight_offset', 0.0))
    # KL trust-region: penalise KL(π_new || π_old) = mean(log π_new - log π_old)
    # which discourages the policy from drifting far from the rollout policy.
    kl_coef = float(getattr(config, 'wbc_kl_coef', 0.0))

    def wbc_loss(policy_params, batch, key):
        dist = networks.policy_network.apply(policy_params, batch['obs'])
        log_pi = networks.log_prob(dist, batch['actions'])          # (B,)

        # Entropy bonus: MC estimate H ≈ -log π(ã|s)
        fresh_a = networks.sample(dist, key)
        entropy_est = -networks.log_prob(dist, fresh_a)             # (B,)
        entropy_mean = jnp.mean(entropy_est)

        # Normalise weights: divide by std so scale is ~1 regardless of
        # how large φ·ψ happens to be.  Optional positive shift to bias
        # towards imitating all actions rather than suppressing bad ones.
        w = batch['weights']                                        # (B,)
        w_norm = (w - w.mean()) / (w.std() + 1e-8) + weight_offset

        wbc = -jnp.mean(w_norm * log_pi)

        # KL trust-region penalty: mean(log π_new - log π_old)
        # ≈ forward KL divergence (positive when policy moves away from old).
        kl_penalty = jnp.mean(log_pi - batch['log_pi_old'])

        wbc_term     = wbc
        ent_term     = -ent_coef * entropy_mean
        kl_term      = kl_coef * kl_penalty
        total        = wbc_term + ent_term + kl_term

        # Fractional contribution of each term to the total loss magnitude.
        abs_total    = jnp.abs(total) + 1e-8
        metrics = {
            'wbc_loss':           wbc,
            'entropy':            entropy_mean,
            'kl_penalty':         kl_penalty,
            'total_loss':         total,
            # Share of each term: signed_contribution / |total|
            'share_wbc':          wbc_term  / abs_total,
            'share_ent':          ent_term  / abs_total,
            'share_kl':           kl_term   / abs_total,
            'weight_mean':        jnp.mean(batch['weights']),
            'weight_std':         jnp.std(batch['weights']),
            'weight_norm_mean':   jnp.mean(w_norm),
            'log_pi_mean':        jnp.mean(log_pi),
        }
        return total, metrics

    grad_fn = jax.value_and_grad(wbc_loss, has_aux=True)

    @jax.jit
    def update(policy_params, opt_state, batch, key):
        (_, metrics), grads = grad_fn(policy_params, batch, key)

        # Gradient norm: L2 norm of the flattened policy gradient.
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in grad_leaves))
        metrics['grad_norm'] = grad_norm

        updates, new_opt_state = policy_optimizer.update(
            grads, opt_state, policy_params)
        new_params = optax.apply_updates(policy_params, updates)
        return new_params, new_opt_state, metrics

    return update


# ---------------------------------------------------------------------------
# Weight computation: φ(s,a)·ψ(g) over a flat batch
# ---------------------------------------------------------------------------
def make_weight_fn(networks: contrastive_networks.ContrastiveNetworks):
    """Returns a jitted function that computes φ(s,a)·ψ(g) for a batch."""
    @jax.jit
    def weight_fn(q_params, obs, action):
        _, phi, psi_g = networks.q_network.apply(q_params, obs, action)
        return jnp.sum(phi * psi_g, axis=-1)   # (B,)
    return weight_fn


# ---------------------------------------------------------------------------
# Checkpoint helpers (WBC has no value params)
# ---------------------------------------------------------------------------
def _save_wbc_checkpoint(path, policy_params, q_params,
                         policy_opt_state, q_opt_state,
                         iteration, global_step, key):
    import pickle as _pkl
    ckpt = {
        'policy_params':       policy_params,
        'value_params':        None,            # kept for compat with plot scripts
        'q_params':            q_params,
        'ppo_optimizer_state': policy_opt_state,
        'q_optimizer_state':   q_opt_state,
        'iteration':           int(iteration),
        'global_step':         int(global_step),
        'key':                 key,
    }
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        _pkl.dump(ckpt, fh, protocol=_pkl.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Top-level training loop
# ---------------------------------------------------------------------------
def run_wbc_training(
    config: contrastive_config.ContrastiveConfig,
    env_factory: Callable,
    eval_env_factory: Callable,
    network_factory: Callable,
    logger_fn: Callable,
    total_steps: int,
    seed: int = 0,
    checkpoint_dir: Optional[str] = None,
):
    """Top-level WBC-on-φ·ψ training loop.

    Per iteration:
      1. Collect T×E on-policy steps.
      2. Compute weights w = φ(s,a)·ψ(g) with frozen CRL params.
      3. For num_epochs × num_minibatches: WBC policy update.
      4. For crl_steps: off-policy CRL update from replay buffer.
      5. Log + eval + checkpoint.
    """
    from acme import specs as _specs
    import contrastive.utils as _cu

    probe_env = env_factory(seed)
    spec = _specs.make_environment_spec(probe_env)
    networks = network_factory(spec=spec)
    del probe_env

    vec_env = VecEnv(env_factory, config.ppo_num_envs, seed=seed * 31)
    E = vec_env.num_envs
    obs_shape = vec_env.observation_shape
    act_shape = vec_env.action_shape
    T = int(config.ppo_rollout_length)
    batch_per_iter = T * E
    mb_size = batch_per_iter // int(config.ppo_num_minibatches)
    assert mb_size * int(config.ppo_num_minibatches) == batch_per_iter
    num_iterations = int(total_steps) // (T * E)

    # ---- init params -------------------------------------------------------
    key = jax.random.PRNGKey(seed)
    k_pol, k_q, key = jax.random.split(key, 3)
    policy_params = networks.policy_network.init(k_pol)
    q_params = networks.q_network.init(k_q)

    # ---- optimizers --------------------------------------------------------
    if getattr(config, 'ppo_anneal_lr', True):
        total_updates = (num_iterations
                         * int(config.ppo_num_epochs)
                         * int(config.ppo_num_minibatches))
        lr_schedule = optax.linear_schedule(
            init_value=float(config.learning_rate),
            end_value=0.0,
            transition_steps=max(1, total_updates))
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
            optax.scale_by_adam(eps=1e-5),
            optax.scale_by_schedule(lambda c: -lr_schedule(c)))
    else:
        lr_schedule = None
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.ppo_max_grad_norm)),
            optax.adam(float(config.learning_rate), eps=1e-5))
    policy_opt_state = policy_optimizer.init(policy_params)

    q_optimizer = optax.adam(float(config.learning_rate))
    q_opt_state = q_optimizer.init(q_params)

    # ---- resume from checkpoint if present ---------------------------------
    start_iteration = 0
    global_step = 0
    wbc_sgd_step = 0
    if checkpoint_dir is not None:
        _latest = os.path.join(checkpoint_dir, 'latest.pkl')
        if os.path.exists(_latest):
            _ckpt = load_checkpoint(_latest)
            policy_params   = _ckpt['policy_params']
            q_params        = _ckpt['q_params']
            policy_opt_state = _ckpt['ppo_optimizer_state']
            q_opt_state     = _ckpt['q_optimizer_state']
            start_iteration = int(_ckpt['iteration']) + 1
            global_step     = int(_ckpt['global_step'])
            key             = _ckpt['key']
            wbc_sgd_step    = (start_iteration
                               * int(config.ppo_num_epochs)
                               * int(config.ppo_num_minibatches))
            print(f'[wbc] resumed: start_iteration={start_iteration}, '
                  f'global_step={global_step}')
            _run_dir = os.path.dirname(checkpoint_dir)
            for _label in ('learner', 'eval'):
                _csv = os.path.join(_run_dir, 'logs', _label, 'logs.csv')
                _truncate_csv_to_iteration(_csv, int(_ckpt['iteration']))

    # ---- jitted helpers ----------------------------------------------------
    weight_fn = make_weight_fn(networks)
    wbc_update = make_wbc_update_fn(networks, config, policy_optimizer)
    crl_update = make_crl_update_fn(networks, q_optimizer)

    @jax.jit
    def sample_action(policy_p, obs, rng):
        dist = networks.policy_network.apply(policy_p, obs)
        action = networks.sample(dist, rng)
        return action

    @jax.jit
    def compute_log_prob(policy_p, obs, actions):
        dist = networks.policy_network.apply(policy_p, obs)
        return networks.log_prob(dist, actions)

    # ---- uniform-goal bounds -----------------------------------------------
    uniform_sampling = bool(getattr(config, 'uniform_sampling', False))
    goal_low = goal_high = None
    if uniform_sampling:
        import env_utils as _eu
        goal_low, goal_high = _eu.resolve_uniform_goal_bounds(
            spec, vec_env._envs[0], int(config.obs_dim),
            int(config.start_index), int(config.end_index))
        print(f'[wbc] uniform_sampling: goal_low={goal_low}, '
              f'goal_high={goal_high}')

    # ---- replay buffer -----------------------------------------------------
    replay = EpisodeReplay(
        capacity=int(config.max_replay_size),
        obs_dim=int(config.obs_dim),
        discount=float(config.discount),
        start_index=int(config.start_index),
        end_index=int(config.end_index))
    np_rng = np.random.default_rng(seed + 12345)

    # ---- episode accounting ------------------------------------------------
    ep_obs: list = [[] for _ in range(E)]
    ep_act: list = [[] for _ in range(E)]
    ep_return = np.zeros(E, dtype=np.float32)
    ep_len = np.zeros(E, dtype=np.int32)
    recent_returns: list = []
    recent_lengths: list = []

    obs = vec_env.reset()
    next_done = np.zeros(E, dtype=np.float32)
    for i in range(E):
        ep_obs[i].append(obs[i].copy())

    # No per-step reward normalizer needed — weight z-scoring happens
    # inside make_wbc_update_fn on each minibatch.

    # ---- loggers -----------------------------------------------------------
    learner_logger = logger_fn(label='learner')
    eval_logger = logger_fn(label='eval')

    _env_name = str(getattr(config, 'env_name', '') or '').lower()
    if _env_name == 'riverswim':
        eval_success_obs = _cu.RiverSwimGoalVisitSuccessObserver(
            obs_dim=int(config.obs_dim))
    else:
        eval_success_obs = _cu.SuccessObserver()
    eval_dist_obs = _cu.DistanceObserver(
        obs_dim=int(config.obs_dim),
        start_index=int(config.start_index),
        end_index=int(config.end_index))

    # ---- rollout storage ---------------------------------------------------
    roll_obs = np.zeros((T, E) + obs_shape, dtype=np.float32)
    roll_acts = np.zeros((T, E) + act_shape, dtype=np.float32)
    roll_env_rew = np.zeros((T, E), dtype=np.float32)
    roll_dones = np.zeros((T, E), dtype=np.float32)
    # Weights computed after rollout; stored flat for minibatch indexing.
    roll_weights = np.zeros((T, E), dtype=np.float32)
    # log π_old(a|s) from the rollout policy, used for KL trust-region penalty.
    roll_log_pi_old = np.zeros((T, E), dtype=np.float32)

    # ---- checkpointing -----------------------------------------------------
    ckpt_interval = int(getattr(config, 'ppo_checkpoint_interval', 0))
    ckpt_keep_last = int(getattr(config, 'ppo_checkpoint_keep_last', 0))
    if ckpt_interval > 0 and checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        keep_msg = ('keep all'
                    if ckpt_keep_last <= 0
                    else f'keep last {ckpt_keep_last}')
        print(f'[wbc] checkpoints → {checkpoint_dir} '
              f'(every {ckpt_interval} iters, {keep_msg})')

    start_time = time.time()

    for iteration in range(start_iteration, num_iterations):
        # =================================================================
        # 1. Rollout (on-policy)
        # =================================================================
        for t in range(T):
            roll_obs[t] = obs
            roll_dones[t] = next_done

            key, k_act = jax.random.split(key)
            action_j = sample_action(
                policy_params, jnp.asarray(obs), k_act)
            action = np.asarray(action_j)
            roll_acts[t] = action
            roll_log_pi_old[t] = np.asarray(compute_log_prob(
                policy_params, jnp.asarray(obs), action_j))

            next_obs, env_rew, dones, terminal_obs = vec_env.step(action)
            roll_env_rew[t] = env_rew

            for i in range(E):
                ep_act[i].append(action[i].copy())
                ep_return[i] += float(env_rew[i])
                ep_len[i] += 1
                if dones[i]:
                    ep_obs[i].append(terminal_obs[i].copy())
                    try:
                        replay.add_episode(
                            np.stack(ep_obs[i], axis=0),
                            np.stack(ep_act[i], axis=0))
                    except AssertionError:
                        pass
                    ep_obs[i] = [next_obs[i].copy()]
                    ep_act[i] = []
                    recent_returns.append(float(ep_return[i]))
                    recent_lengths.append(int(ep_len[i]))
                    ep_return[i] = 0.0
                    ep_len[i] = 0
                    if len(recent_returns) > 100:
                        recent_returns.pop(0)
                        recent_lengths.pop(0)
                else:
                    ep_obs[i].append(next_obs[i].copy())

            obs = next_obs
            next_done = dones.astype(np.float32)
            global_step += E

        # =================================================================
        # 2. Compute CRL weights φ(s,a)·ψ(g) over the entire rollout
        #    (frozen q_params — same convention as PPO reward computation)
        # =================================================================
        flat_obs_all = roll_obs.reshape((batch_per_iter,) + obs_shape)
        flat_acts_all = roll_acts.reshape((batch_per_iter,) + act_shape)
        flat_log_pi_old_all = roll_log_pi_old.reshape(batch_per_iter)
        weights_j = weight_fn(
            q_params,
            jnp.asarray(flat_obs_all),
            jnp.asarray(flat_acts_all))
        flat_weights = np.asarray(weights_j)

        flat_weights_norm = flat_weights   # z-score done per-minibatch inside wbc_update

        # =================================================================
        # 3. WBC policy updates (epochs × minibatches)
        # =================================================================
        wbc_metrics_agg: Dict[str, list] = {}
        for _epoch in range(int(config.ppo_num_epochs)):
            perm = np_rng.permutation(batch_per_iter)
            for start in range(0, batch_per_iter, mb_size):
                mb = perm[start:start + mb_size]
                batch = {
                    'obs':         jnp.asarray(flat_obs_all[mb]),
                    'actions':     jnp.asarray(flat_acts_all[mb]),
                    'weights':     jnp.asarray(flat_weights_norm[mb]),
                    'log_pi_old':  jnp.asarray(flat_log_pi_old_all[mb]),
                }
                key, k_mb = jax.random.split(key)
                policy_params, policy_opt_state, m = wbc_update(
                    policy_params, policy_opt_state, batch, k_mb)
                wbc_sgd_step += 1
                for k_, v in m.items():
                    wbc_metrics_agg.setdefault(k_, []).append(float(v))

        # =================================================================
        # 4. CRL updates (off-policy, replay buffer)
        # =================================================================
        crl_metrics_agg: Dict[str, list] = {}
        if replay.size >= int(config.ppo_min_replay_size):
            for _ in range(int(config.ppo_crl_steps_per_iter)):
                if uniform_sampling:
                    crl_batch_np = replay.sample_with_uniform_negatives(
                        int(config.batch_size), np_rng, goal_low, goal_high)
                else:
                    crl_batch_np = replay.sample(
                        int(config.batch_size), np_rng)
                crl_batch = {k_: jnp.asarray(v)
                             for k_, v in crl_batch_np.items()}
                key, k_crl = jax.random.split(key)
                q_params, q_opt_state, m = crl_update(
                    q_params, q_opt_state, crl_batch, k_crl)
                for k_, v in m.items():
                    crl_metrics_agg.setdefault(k_, []).append(float(v))

        # =================================================================
        # 5. Logging
        # =================================================================
        elapsed = time.time() - start_time
        if lr_schedule is not None:
            lr_log = float(lr_schedule(max(0, wbc_sgd_step - 1)))
        else:
            lr_log = float(config.learning_rate)

        log = {
            'iteration':              iteration,
            'learner_steps':          iteration,
            'global_step':            global_step,
            'sps':                    global_step / max(1e-6, elapsed),
            'replay_size':            int(replay.size),
            'weight_raw_mean':        float(flat_weights.mean()),
            'weight_raw_std':         float(flat_weights.std()),
            'reward_env_mean':        float(roll_env_rew.mean()),
            'ep_return_mean':  (float(np.mean(recent_returns))
                                if recent_returns else float('nan')),
            'ep_length_mean':  (float(np.mean(recent_lengths))
                                if recent_lengths else float('nan')),
            'wbc/learning_rate':      lr_log,
            'crl/crl_loss':           float('nan'),
            'crl/categorical_accuracy': float('nan'),
            'crl/binary_accuracy':    float('nan'),
            'crl/logits_pos':         float('nan'),
            'crl/logits_neg':         float('nan'),
            'crl/logsumexp':          float('nan'),
        }
        for k_, vs in wbc_metrics_agg.items():
            log[f'wbc/{k_}'] = float(np.mean(vs))
        for k_, vs in crl_metrics_agg.items():
            log[f'crl/{k_}'] = float(np.mean(vs))
        learner_logger.write(log)

        # =================================================================
        # 6. Evaluation (5 episodes every 10 iters)
        # =================================================================
        if iteration % 10 == 0:
            ep_metrics_list = []
            for e_i in range(5):
                env = eval_env_factory(
                    seed + 900_000 + iteration * 100 + e_i)
                ts = env.reset()
                eval_success_obs.observe_first(env, ts)
                eval_dist_obs.observe_first(env, ts)
                ret_e, n_e = 0.0, 0
                while not ts.last():
                    key, k_eval = jax.random.split(key)
                    a_j = sample_action(
                        policy_params,
                        jnp.asarray(ts.observation)[None], k_eval)
                    action = np.asarray(a_j)[0].astype(np.float32)
                    action = np.nan_to_num(action, nan=0.0,
                                           posinf=1.0, neginf=-1.0)
                    action = np.clip(action, -1.0, 1.0)
                    ts = env.step(action)
                    eval_success_obs.observe(env, ts, action)
                    eval_dist_obs.observe(env, ts, action)
                    ret_e += float(ts.reward or 0.0)
                    n_e += 1
                ep_m = {'episode_return': ret_e, 'episode_length': n_e}
                ep_m.update(eval_success_obs.get_metrics())
                ep_m.update(eval_dist_obs.get_metrics())
                ep_metrics_list.append(ep_m)

            agg = {'iteration': iteration, 'learner_steps': iteration}
            for k_ in ep_metrics_list[0].keys():
                agg[k_] = float(
                    np.nanmean([m[k_] for m in ep_metrics_list]))
            eval_logger.write(agg)

        # =================================================================
        # 7. Checkpointing
        # =================================================================
        if (ckpt_interval > 0
                and checkpoint_dir is not None
                and (iteration % ckpt_interval == 0
                     or iteration == num_iterations - 1)):
            ckpt_kw = dict(
                policy_params=policy_params,
                q_params=q_params,
                policy_opt_state=policy_opt_state,
                q_opt_state=q_opt_state,
                iteration=iteration,
                global_step=global_step,
                key=key)
            milestone = os.path.join(
                checkpoint_dir, f'ckpt_iter_{iteration:07d}.pkl')
            _save_wbc_checkpoint(milestone, **ckpt_kw)
            _save_wbc_checkpoint(
                os.path.join(checkpoint_dir, 'latest.pkl'), **ckpt_kw)
            _prune_old_checkpoints(checkpoint_dir, ckpt_keep_last)
