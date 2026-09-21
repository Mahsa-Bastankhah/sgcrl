"""DISCOVER on BuilderBench.

TD3 + HER + UCB follow Diaz-Bone et al. / JaxGCRL
``src/baselines/td3/td3_train.py`` (scan SGD, trajectory replay, HER flatten,
UCB, policy delay, target EMA). Env wiring is BuilderBench-only: CreativeCube
+ PD, curriculum goal written into ``info['target_goal']`` + mocap.

Not copied (unused by the paper DISCOVER recipe): MEGA/KDE, SAC, dynamics,
achievement net, pmap.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Tuple

_REPO = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

os.environ.setdefault('MUJOCO_GL', 'egl')
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
  os.environ['XLA_FLAGS'] = xla_flags + ' --xla_gpu_triton_gemm_any=True'

import sgcrl_jax_acme_compat  # noqa: E402,F401

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState

from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv
from envs.builderbench_utils import creative_cube_pd_macro_episode_length

import bb_env
import networks as nets
import discover_video


def _soft_update(target, online, tau: float):
  return jax.tree_util.tree_map(
      lambda t, o: (1.0 - tau) * t + tau * o, target, online)


def _horizon_transform(q, gamma: float):
  q = jnp.maximum(q + 0.01, 0.001)
  return -jnp.log((1.0 - gamma) * q) / jnp.log(gamma)


@dataclass
class Args:
  env: str = 'builderbench_creative_2_task1'
  seed: int = 0
  log_dir: str = 'logs/discover_builderbench_creative2_task1_e128_pd_tanh_sgdE_warp_1h/'
  num_timesteps: int = 20_000_000
  num_envs: int = 128
  num_eval_envs: int = 128
  batch_size: int = 128
  # Unused when equal to num_envs (their sample always uses num_envs streams).
  sgd_num_envs: int = 0
  ensemble_size: int = 6
  ucb_mean_ini: float = 0.0
  ucb_mean_tar: float = 1.0
  ucb_std_ini: float = 1.0
  ucb_std_tar: float = 0.0
  transform_ucb: bool = True
  activate_final: bool = True
  relabel_prob_future: float = 0.7
  relabel_prob_uniform: float = 0.0
  relabel_prob_geometric: float = 0.0
  num_goals: int = 5000
  train_step_multiplier: int = 32
  critic_lr: float = 3e-4
  actor_lr: float = 3e-4
  tau: float = 0.005
  tau_policy: float = 5e-7
  discount: float = 0.99
  policy_delay: int = 2
  exploration_noise: float = 0.4
  noise_clip: float = 0.5
  smoothing_noise: float = 0.2
  target_computation: str = 'min_random'
  min_replay_size: int = 1000
  # Time axis of the JaxGCRL buffer (not total transitions). Their default 25000.
  max_replay_size: int = 25000
  adaptation_strategy: str = 'simple'
  goal_achievement_target: float = 0.5
  adaptation_rate: int = 100
  # On each fresh eval with endpoint success > 0, reduce the initial-state
  # uncertainty/novelty coefficient by this amount (clipped at zero).
  eval_std_ini_decay: float = 0.005
  continue_strategy: str = 'uniform'
  pd_duration: int = 5
  permute_start_boxes: bool = False
  fixed_start_x: float = -1.0
  goal_reach_thresh: float = 0.1
  bb_success_thresh: float = 0.02
  eval_interval: int = 10
  checkpoint_interval: int = 50
  video_interval: int = 500
  log_interval: int = 1
  # Stop training once a fresh endpoint eval reaches this success rate
  # (>=).  Set <0 to disable.
  early_stop_eval_success: float = 0.9
  hidden: str = '256,256'
  categorical_select: bool = False
  categorical_select_waypoint: bool = False


@flax.struct.dataclass
class Buf:
  obs: jax.Array
  next_obs: jax.Array
  action: jax.Array
  reward: jax.Array
  discount: jax.Array
  truncation: jax.Array
  traj_id: jax.Array
  insert_pos: jax.Array
  size: jax.Array
  key: jax.Array


@flax.struct.dataclass
class Learner:
  actor: TrainState
  q: TrainState
  target_actor: Any
  target_q: Any
  slow_target_actor: Any
  grad_steps: jax.Array


def _csv_write(path: Path, row: Dict[str, Any], header_written: list):
  path.parent.mkdir(parents=True, exist_ok=True)
  write_header = not path.exists() or not header_written
  with path.open('a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
      w.writeheader()
      header_written.append(True)
    w.writerow(row)


def main(args: Args) -> None:
  hidden = tuple(int(x) for x in args.hidden.split(',') if x.strip())
  env_name = args.env
  (bb_env_id, num_cubes, task_index, state_dim, goal_dim,
   goal_idx) = bb_env.env_layout(env_name, pd_filter=True)
  goal_idx_j = jnp.asarray(goal_idx, dtype=jnp.int32)
  g_star = jnp.asarray(bb_env.task_goal(num_cubes, task_index), dtype=jnp.float32)
  ep_len = creative_cube_pd_macro_episode_length(
      num_cubes, task_index, args.pd_duration)
  unroll = ep_len
  n_env = int(args.num_envs)
  n_eval = int(args.num_eval_envs)
  batch = int(args.batch_size)
  n_cubes = int(num_cubes)
  use_wp = bool(args.categorical_select_waypoint)
  use_cat = bool(args.categorical_select) or use_wp
  n_goals = int(args.num_goals)
  n_critics = int(args.ensemble_size)
  n_inner = int(args.train_step_multiplier)
  gamma = float(args.discount)
  thresh = float(args.goal_reach_thresh)
  bb_thresh = float(args.bb_success_thresh)
  p_future = float(args.relabel_prob_future)
  p_unif = float(args.relabel_prob_uniform)
  p_geom = float(args.relabel_prob_geometric)
  p_none = 1.0 - (p_future + p_unif + p_geom)
  if p_none < -1e-6:
    raise ValueError('HER relabel probabilities must sum to <= 1')
  p_none = max(p_none, 0.0)

  # Their buffer is (time, E). Cap time so E=1024 fits in 16G (~their 25000x128 cells).
  their_cells = 25000 * 128
  time_cap = int(args.max_replay_size)
  if time_cap * n_env > their_cells:
    time_cap = max(2 * ep_len, their_cells // n_env)
  if time_cap < 2 * ep_len:
    raise ValueError(f'replay time_cap={time_cap} < 2*ep_len={2 * ep_len}')
  if (ep_len * n_env) % batch != 0:
    raise ValueError(
        f'episode_length*num_envs = {ep_len * n_env} must divide batch_size={batch}')

  fx = None if args.fixed_start_x < 0 else float(args.fixed_start_x)
  vec = JaxBuilderBenchVecEnv(
      env_name=env_name,
      num_envs=n_env,
      seed=args.seed,
      use_pd=True,
      pd_duration=args.pd_duration,
      pd_filter_policy_obs=True,
      fixed_target_goal=None,
      permute_start_boxes=args.permute_start_boxes,
      fixed_start_x=fx,
  )
  mocap_targets = vec._mocap_targets
  n_task = int(vec._num_task_cubes)
  reset_fn = vec._reset_fn
  step_fn = vec._step_fn
  obs_dim = int(vec._obs_dim_total)
  act_dim = int(vec._action_dim)
  assert act_dim == 5, f'PD action_dim expected 5, got {act_dim}'
  assert obs_dim == state_dim + goal_dim

  if use_wp:
    actor_def = nets.CatWpActor(n_cubes=n_cubes, hidden=hidden)
  elif use_cat:
    actor_def = nets.Actor(n_cubes=n_cubes, n_continuous=4, hidden=hidden)
  else:
    actor_def = nets.TanhActor(action_dim=act_dim, hidden=hidden)
  q_def = nets.EnsembleQ(
      n_critics=n_critics, hidden=hidden, activate_final=args.activate_final)

  rng = jax.random.PRNGKey(args.seed)
  rng, k_a, k_q, k_env, k_eval, k_buf = jax.random.split(rng, 6)
  dummy_obs = jnp.zeros((1, obs_dim), jnp.float32)
  dummy_act = jnp.zeros((1, act_dim), jnp.float32)
  actor_params = actor_def.init(k_a, dummy_obs)
  q_params = q_def.init(k_q, dummy_obs, dummy_act)
  learner = Learner(
      actor=TrainState.create(
          apply_fn=actor_def.apply, params=actor_params,
          tx=optax.adam(args.actor_lr)),
      q=TrainState.create(
          apply_fn=q_def.apply, params=q_params,
          tx=optax.adam(args.critic_lr)),
      target_actor=actor_params,
      target_q=q_params,
      slow_target_actor=actor_params,
      grad_steps=jnp.zeros((), jnp.int32),
  )

  def apply_actor(params, obs):
    return actor_def.apply(params, obs)

  def apply_q(params, obs, action):
    return q_def.apply(params, obs, action)

  def pack_from_state(state):
    return bb_env.pack_obs(
        state.obs, state.info['target_goal'], num_cubes=n_cubes, pd_filter=True)

  def achieved_of(obs):
    return obs[..., goal_idx_j]

  def policy_act(params, obs, key, explore):
    if use_wp:
      wp, yaw, logits = apply_actor(params, obs)
      return nets.catwp_action(
          wp, yaw, logits, key, n_cubes, explore,
          args.exploration_noise, args.noise_clip)
    if use_cat:
      mean, logits = apply_actor(params, obs)
      return nets.actor_action(
          mean, logits, key, n_cubes, explore,
          args.exploration_noise, args.noise_clip)
    mean = apply_actor(params, obs)
    return nets.tanh_actor_action(
        mean, key, explore, args.exploration_noise, args.noise_clip)

  def det_act(params, obs):
    if use_wp:
      wp, yaw, logits = apply_actor(params, obs)
      return nets.catwp_det_action(wp, yaw, logits, n_cubes)
    if use_cat:
      mean, logits = apply_actor(params, obs)
      return nets.actor_det_action(mean, logits, n_cubes)
    return apply_actor(params, obs)

  def compute_value(q_params, actor_params, observations):
    actions = det_act(actor_params, observations)
    q = apply_q(q_params, observations, actions)
    if args.transform_ucb:
      qt = _horizon_transform(q, gamma)
      return jnp.mean(qt, axis=-1), jnp.nan_to_num(jnp.std(qt, axis=-1), nan=0.0)
    qt = _horizon_transform(q, gamma)
    return jnp.mean(q, axis=-1), jnp.nan_to_num(jnp.std(qt, axis=-1), nan=0.0)

  def empty_buf(key):
    z = lambda *sh: jnp.zeros(sh, jnp.float32)
    return Buf(
        obs=z(time_cap, n_env, obs_dim),
        next_obs=z(time_cap, n_env, obs_dim),
        action=z(time_cap, n_env, act_dim),
        reward=z(time_cap, n_env),
        discount=z(time_cap, n_env),
        truncation=z(time_cap, n_env),
        traj_id=z(time_cap, n_env),
        insert_pos=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
        key=key,
    )

  def insert_unroll(buf: Buf, data) -> Buf:
    t = data['obs'].shape[0]
    pos = buf.insert_pos
    idx = (pos + jnp.arange(t, dtype=jnp.int32)) % time_cap
    return buf.replace(
        obs=buf.obs.at[idx].set(data['obs']),
        next_obs=buf.next_obs.at[idx].set(data['next_obs']),
        action=buf.action.at[idx].set(data['action']),
        reward=buf.reward.at[idx].set(data['reward']),
        discount=buf.discount.at[idx].set(data['discount']),
        truncation=buf.truncation.at[idx].set(data['truncation']),
        traj_id=buf.traj_id.at[idx].set(data['traj_id']),
        insert_pos=(pos + t) % time_cap,
        size=jnp.minimum(buf.size + t, time_cap),
    )

  def sample_trajs(buf: Buf, key):
    """(E, T, ...) windows, JaxGCRL ``sample_internal``."""
    key, k_e, k_s = jax.random.split(key, 3)
    env_idx = jax.random.permutation(k_e, n_env)
    max_start = jnp.maximum(buf.size - ep_len, 1)
    starts = jax.random.randint(k_s, (n_env,), 0, max_start)
    t_ix = (starts[:, None] + jnp.arange(ep_len)[None, :]) % jnp.maximum(buf.size, 1)
    e_ix = env_idx[:, None]
    return buf.replace(key=key), {
        'obs': buf.obs[t_ix, e_ix],
        'next_obs': buf.next_obs[t_ix, e_ix],
        'action': buf.action[t_ix, e_ix],
        'reward': buf.reward[t_ix, e_ix],
        'discount': buf.discount[t_ix, e_ix],
        'truncation': buf.truncation[t_ix, e_ix],
        'traj_id': buf.traj_id[t_ix, e_ix],
    }

  def flatten_her(obs, nxt, act, rew, disc, trunc, traj, key):
    """One trajectory (T, ...). ``td3_train.TrajectoryUniformSamplingQueue.flatten_crl_fn``."""
    seq = obs.shape[0]
    key, k_strat, k_u, k_g = jax.random.split(key, 4)
    strat = jax.random.choice(
        k_strat, jnp.array([0, 1, 2, 3]),
        p=jnp.array([p_future, p_unif, p_geom, p_none]))
    t_lin = jnp.linspace(0.0, seq - 1.0, seq)
    u = jax.random.uniform(k_u, (seq,))
    fut_idx = ((seq - jnp.linspace(1.0, seq, seq)) * u + t_lin).astype(jnp.int32)
    unif_idx = (seq * jax.random.uniform(k_u, (seq,))).astype(jnp.int32)
    p = 0.2
    geom = jnp.ceil(jnp.log(1.0 - jax.random.uniform(k_g, (seq,))) / jnp.log(1.0 - p))
    geom_idx = jnp.minimum(geom + t_lin, seq - 1.0).astype(jnp.int32)
    g_fut = achieved_of(obs[fut_idx])
    g_unif = achieved_of(obs[unif_idx])
    g_geom = achieved_of(obs[geom_idx])
    g_none = obs[:, state_dim:]
    new_g = (
        (strat == 0).astype(jnp.float32) * g_fut
        + (strat == 1).astype(jnp.float32) * g_unif
        + (strat == 2).astype(jnp.float32) * g_geom
        + (strat == 3).astype(jnp.float32) * g_none)
    state = obs[:, :state_dim]
    nstate = nxt[:, :state_dim]
    new_obs = jnp.concatenate([state, new_g], axis=-1)
    new_next = jnp.concatenate([nstate, new_g], axis=-1)
    r = bb_env.sparse_ball_reward(achieved_of(new_obs), new_g, thresh)
    return new_obs, new_next, act, r, disc * gamma, trunc, traj

  def critic_loss_fn(q_params, target_q, target_actor, obs, nxt, act, rew, disc, key):
    q = apply_q(q_params, obs, act)
    next_a = det_act(target_actor, nxt)
    key, k_t = jax.random.split(key)
    noise = (jax.random.normal(key, next_a.shape) * args.smoothing_noise).clip(
        -args.noise_clip, args.noise_clip)
    next_a = jnp.clip(next_a + noise, -1.0, 1.0)
    nq = apply_q(target_q, nxt, next_a)
    if args.target_computation == 'mean':
      target = jnp.mean(nq, axis=-1)
    elif args.target_computation == 'min':
      target = jnp.min(nq, axis=-1)
    elif args.target_computation == 'single':
      target = nq
    else:
      rdm = jax.random.choice(
          k_t, n_critics, shape=(2,), replace=False)
      target = jnp.min(nq[:, rdm], axis=-1)
    if args.target_computation == 'single':
      y = jax.lax.stop_gradient(rew[:, None] + disc[:, None] * target)
      err = q - y
    else:
      y = jax.lax.stop_gradient(rew + disc * target)
      err = q - y[:, None]
    return 0.5 * jnp.mean(jnp.square(err))

  def actor_loss_fn(actor_params, q_params, obs):
    if use_wp:
      wp, yaw, logits = apply_actor(actor_params, obs)
      pi = jax.nn.softmax(logits, axis=-1)
      acts = nets.pack_all_catwp(wp, yaw, n_cubes)
      b = wp.shape[0]
      q = apply_q(
          q_params,
          jnp.repeat(obs, n_cubes, axis=0),
          acts.reshape((b * n_cubes, act_dim)))
      q_half, _ = jnp.split(q, 2, axis=-1)
      q_i = jnp.mean(q_half, axis=-1).reshape((b, n_cubes))
      return -jnp.mean(jnp.sum(pi * q_i, axis=-1))
    if use_cat:
      # Exact E_i[Q(s, [xyz/yaw, center_i])] so logits get ∇ and xyz/yaw is
      # shared across selects (argmax is not differentiable).
      mean, logits = apply_actor(actor_params, obs)
      pi = jax.nn.softmax(logits, axis=-1)
      acts = nets.pack_all_selects(mean, n_cubes)
      b = mean.shape[0]
      q = apply_q(
          q_params,
          jnp.repeat(obs, n_cubes, axis=0),
          acts.reshape((b * n_cubes, act_dim)))
      q_half, _ = jnp.split(q, 2, axis=-1)
      q_i = jnp.mean(q_half, axis=-1).reshape((b, n_cubes))
      return -jnp.mean(jnp.sum(pi * q_i, axis=-1))
    a = det_act(actor_params, obs)
    q = apply_q(q_params, obs, a)
    q_half, _ = jnp.split(q, 2, axis=-1)
    return -jnp.mean(q_half)

  def update_step(carry, mb):
    learner, key = carry
    obs, nxt, act, rew, disc = mb
    key, k_c, k_a = jax.random.split(key, 3)

    def c_loss(qp):
      return critic_loss_fn(
          qp, learner.target_q, learner.target_actor, obs, nxt, act, rew, disc, k_c)

    c_val, c_grad = jax.value_and_grad(c_loss)(learner.q.params)
    q_state = learner.q.apply_gradients(grads=c_grad)

    def do_pi(_):
      def a_loss(ap):
        return actor_loss_fn(ap, q_state.params, obs)
      a_val, a_grad = jax.value_and_grad(a_loss)(learner.actor.params)
      actor = learner.actor.apply_gradients(grads=a_grad)
      # Their policy target: EMA from *previous online* to new online.
      new_ta = _soft_update(learner.actor.params, actor.params, args.tau)
      new_tq = _soft_update(learner.target_q, q_state.params, args.tau)
      new_slow = _soft_update(
          learner.slow_target_actor, actor.params, args.tau_policy)
      return a_val, actor, new_ta, new_tq, new_slow

    def skip_pi(_):
      return (
          jnp.zeros((), jnp.float32), learner.actor, learner.target_actor,
          learner.target_q, learner.slow_target_actor)

    a_val, actor, new_ta, new_tq, new_slow = jax.lax.cond(
        learner.grad_steps % int(args.policy_delay) == 0, do_pi, skip_pi, 0)
    new_learner = learner.replace(
        actor=actor, q=q_state, target_actor=new_ta, target_q=new_tq,
        slow_target_actor=new_slow, grad_steps=learner.grad_steps + 1)
    return (new_learner, key), (c_val, a_val)

  def train_steps(learner, buf, key):
    key, k_samp, k_her, k_perm, k_upd = jax.random.split(key, 5)
    buf, traj = sample_trajs(buf, k_samp)
    her_keys = jax.random.split(k_her, n_env)
    obs, nxt, act, rew, disc, _, _ = jax.vmap(flatten_her)(
        traj['obs'], traj['next_obs'], traj['action'], traj['reward'],
        traj['discount'], traj['truncation'], traj['traj_id'], her_keys)
    # (E, T, ...) → (E*T, ...) Fortran, then (n_mb, B, ...) as in td3_train.
    def flat_f(x):
      return jnp.reshape(x, (-1,) + x.shape[2:], order='F')
    obs, nxt, act, rew, disc = map(flat_f, (obs, nxt, act, rew, disc))
    perm = jax.random.permutation(k_perm, obs.shape[0])
    obs, nxt, act, rew, disc = obs[perm], nxt[perm], act[perm], rew[perm], disc[perm]
    n_mb = obs.shape[0] // batch
    obs = obs.reshape((n_mb, batch, -1))
    nxt = nxt.reshape((n_mb, batch, -1))
    act = act.reshape((n_mb, batch, -1))
    rew = rew.reshape((n_mb, batch))
    disc = disc.reshape((n_mb, batch))
    (learner, _), (c_loss, a_loss) = jax.lax.scan(
        update_step, (learner, k_upd), (obs, nxt, act, rew, disc))
    return learner, buf, jnp.mean(c_loss), jnp.mean(a_loss)

  def scan_train_steps(learner, buf, key):
    def body(carry, _):
      learner, buf, key = carry
      key, k = jax.random.split(key)
      learner, buf, c_loss, a_loss = train_steps(learner, buf, k)
      return (learner, buf, key), (c_loss, a_loss)
    (learner, buf, key), (c_loss, a_loss) = jax.lax.scan(
        body, (learner, buf, key), (), length=n_inner)
    return learner, buf, key, jnp.mean(c_loss), jnp.mean(a_loss)

  def ucb_select(q_params, actor_params, cand_obs, s0_state, g_star_vec, ucb_w):
    n = cand_obs.shape[0]
    achieved = achieved_of(cand_obs)
    tile = n // s0_state.shape[0] + 1
    s0_rep = jnp.tile(s0_state, (tile, 1))[:n]
    obs_ini = jnp.concatenate([s0_rep, achieved], axis=-1)
    mean_ini, std_ini = compute_value(q_params, actor_params, obs_ini)
    gstar_b = jnp.broadcast_to(g_star_vec, (n, g_star_vec.shape[0]))
    obs_tar = jnp.concatenate([cand_obs[:, :state_dim], gstar_b], axis=-1)
    mean_tar, std_tar = compute_value(q_params, actor_params, obs_tar)
    ucb = (ucb_w[0] * mean_ini + ucb_w[1] * mean_tar
           + ucb_w[2] * std_ini + ucb_w[3] * std_tar)
    idx = jnp.argpartition(ucb, -n_env)[-n_env:]
    return achieved[idx]

  def sample_cand(buf: Buf, key):
    key, k_t, k_e = jax.random.split(key, 3)
    t_ix = jax.random.randint(k_t, (n_goals,), 0, jnp.maximum(buf.size, 1))
    e_ix = jax.random.randint(k_e, (n_goals,), 0, n_env)
    return buf.obs[t_ix, e_ix], key

  def collect_unroll(env_state, actor_params, goals, key, g_star_vec):
    goals_unach = jnp.ones((n_env,), jnp.float32)
    traj_id = jnp.arange(n_env, dtype=jnp.float32)

    def body(carry, _):
      state, unach, key = carry
      key, k_act, k_rnd = jax.random.split(key, 3)
      obs = pack_from_state(state)
      pi_a = policy_act(actor_params, obs, k_act, True)
      rnd_a = jax.random.uniform(k_rnd, pi_a.shape, minval=-1.0, maxval=1.0)
      if args.continue_strategy == 'uniform':
        a = unach[:, None] * pi_a + (1.0 - unach)[:, None] * rnd_a
      else:
        a = pi_a
      nstate = step_fn(state, a)
      nobs = pack_from_state(nstate)
      r = bb_env.sparse_ball_reward(achieved_of(obs), goals, thresh)
      unach = unach * (1.0 - r)
      done = nstate.done.astype(jnp.float32)
      trunc = nstate.info['truncation'].astype(jnp.float32)
      task_s = bb_env.cube_success(
          nstate.info['achieved_goal'], g_star_vec, bb_thresh)
      return (nstate, unach, key), {
          'obs': obs, 'next_obs': nobs, 'action': a, 'reward': r,
          'discount': 1.0 - done, 'truncation': trunc, 'traj_id': traj_id,
          'task_success': task_s,
      }

    (nstate, unach, key), data = jax.lax.scan(
        body, (env_state, goals_unach, key), (), length=unroll)
    return nstate, data, 1.0 - unach, key

  def adapt_ucb(ucb_w, hit_rate):
    if args.adaptation_strategy != 'simple':
      return ucb_w
    mid = 0.5 * (args.ucb_mean_ini + args.ucb_mean_tar)
    if abs(mid) < 1e-8:
      mid = 0.5 * (args.ucb_std_ini + 1e-6)
    step = float(mid) / float(args.adaptation_rate)
    mt = ucb_w[1]
    tgt = args.goal_achievement_target
    mt = jnp.where(hit_rate > tgt + 0.2, jnp.minimum(mt + step, 2.0 * mid), mt)
    mt = jnp.where(hit_rate < tgt - 0.2, jnp.maximum(mt - step, 0.0), mt)
    mi = 2.0 * mid - mt
    return ucb_w.at[0].set(mi).at[1].set(mt)

  def eval_unroll(env_state, actor_params, g_star_vec):
    env_state = bb_env.set_shared_goal(
        env_state, g_star_vec, mocap_targets, n_task)

    def body(state, _):
      obs = pack_from_state(state)
      a = det_act(actor_params, obs)
      nstate = step_fn(state, a)
      return nstate, {
          # Match train_success_1000 exactly: all task cubes must be within
          # bb_thresh (2 cm by default) at a PD macro-action endpoint.
          'endpoint_success': bb_env.cube_success(
              nstate.info['achieved_goal'], g_star_vec, bb_thresh),
          # PDWrapper sums metrics over its microsteps, so this separately
          # records transient success anywhere inside a macro-action.
          'any_microstep_success': nstate.metrics['success'],
      }

    _, data = jax.lax.scan(body, env_state, (), length=unroll)
    ep_endpoint = jnp.max(data['endpoint_success'], axis=0)
    ep_any_microstep = jnp.max(
        (data['any_microstep_success'] >= 0.5).astype(jnp.float32), axis=0)
    return jnp.mean(ep_endpoint), jnp.mean(ep_any_microstep)

  collect_unroll = jax.jit(collect_unroll)
  insert_unroll = jax.jit(insert_unroll)
  scan_train_steps = jax.jit(scan_train_steps)
  ucb_select = jax.jit(ucb_select)
  sample_cand = jax.jit(sample_cand)
  eval_unroll = jax.jit(eval_unroll)
  adapt_ucb = jax.jit(adapt_ucb)

  def s0_from_reset(keys):
    return pack_from_state(reset_fn(keys))[:, :state_dim]

  s0_from_reset = jax.jit(s0_from_reset)

  def training_step(learner, env_state, buf, key, ucb_w, g_star_vec):
    key, k_ucb, k_col, k_sgd, k_reset = jax.random.split(key, 5)
    cand, k_ucb = sample_cand(buf, k_ucb)
    s0 = s0_from_reset(jax.random.split(k_reset, n_env))
    goals = ucb_select(
        learner.q.params, learner.target_actor, cand, s0, g_star_vec, ucb_w)
    # Mean concatenated L2 distance from commanded UCB goals to final g*.
    # This matches the goal-vector distance convention used by DISCOVER.
    goal_dist_to_final = jnp.mean(
        jnp.linalg.norm(goals - g_star_vec[None, :], axis=-1))
    env_state = bb_env.set_per_env_goals(env_state, goals, mocap_targets, n_task)
    env_state, data, hit, k_col = collect_unroll(
        env_state, learner.actor.params, goals, k_col, g_star_vec)
    buf = insert_unroll(buf, data)
    ucb_w = adapt_ucb(ucb_w, jnp.mean(hit))
    learner, buf, k_sgd, c_loss, a_loss = scan_train_steps(learner, buf, k_sgd)
    return (
        learner, env_state, buf, k_col, ucb_w, data, hit,
        goal_dist_to_final, c_loss, a_loss)

  log_dir = Path(args.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  run_dir = log_dir / f'discover_{env_name}_{args.seed}'
  run_dir.mkdir(parents=True, exist_ok=True)
  csv_path = run_dir / 'logs.csv'
  ckpt_dir = run_dir / 'checkpoints'
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  header_written: list = []
  run_cfg = {
      'env': env_name,
      'seed': int(args.seed),
      'num_cubes': n_cubes,
      'hidden': list(hidden),
      'builderbench_use_pd': True,
      'builderbench_pd_duration': int(args.pd_duration),
      'builderbench_permute_start_boxes': bool(args.permute_start_boxes),
      'builderbench_fixed_start_x': (
          None if args.fixed_start_x < 0 else float(args.fixed_start_x)),
      'categorical_select': bool(use_cat),
      'categorical_select_waypoint': bool(use_wp),
      'goal_reach_thresh': float(args.goal_reach_thresh),
      'eval_std_ini_decay': float(args.eval_std_ini_decay),
      'eval_interval': int(args.eval_interval),
      'checkpoint_interval': int(args.checkpoint_interval),
      'video_interval': int(args.video_interval),
      'sgd_num_envs': n_env,
      'num_envs': int(args.num_envs),
      'replay_time_cap': int(time_cap),
      'trainer': 'jaxgcrl_scan',
  }
  (run_dir / 'run_config.json').write_text(json.dumps(run_cfg, indent=2))

  def _save_ckpt(epoch_i: int, steps_i: int) -> None:
    payload = {
        'actor_params': jax.device_get(learner.actor.params),
        'epoch': int(epoch_i),
        'env_steps': int(steps_i),
        'n_cubes': n_cubes,
        'hidden': list(hidden),
        'env': env_name,
        'categorical_select': use_cat,
        'categorical_select_waypoint': use_wp,
    }
    blob = pickle.dumps(payload)
    (ckpt_dir / 'latest.pkl').write_bytes(blob)
    named = ckpt_dir / f'ckpt_epoch_{int(epoch_i):07d}.pkl'
    named.write_bytes(blob)
    print(f'[discover] wrote {named} steps={steps_i}', flush=True)

  env_keys = jax.random.split(k_env, n_env)
  env_state = reset_fn(env_keys)
  env_state = bb_env.set_shared_goal(env_state, g_star, mocap_targets, n_task)
  buf = empty_buf(k_buf)
  ucb_w = jnp.array([
      args.ucb_mean_ini, args.ucb_mean_tar, args.ucb_std_ini, args.ucb_std_tar
  ], dtype=jnp.float32)

  recent_task_success: Deque[float] = deque(maxlen=1000)
  recent_goal_hit: Deque[float] = deque(maxlen=1000)
  env_steps = 0
  epoch = 0
  t0 = time.time()

  updates_per_epoch = n_inner * (ep_len * n_env) // batch
  print('[discover] env='
        f'{env_name} bb={bb_env_id} cubes={n_cubes} task={task_index+1} '
        f'E={n_env} ep_len={ep_len} obs={obs_dim} act={act_dim} '
        f'state_dim={state_dim} goal_dim={goal_dim} '
        f'updates/epoch={updates_per_epoch} replay_T={time_cap} '
        f'trainer=jaxgcrl_scan', flush=True)
  steps_per_epoch = n_env * unroll
  video_dir = run_dir / 'videos'
  video_dir.mkdir(parents=True, exist_ok=True)
  vid_env = None
  vid_run = None
  video_enabled = args.video_interval > 0
  print(
      f'[discover] eval every {args.eval_interval} epochs; '
      f'ckpt/video every {args.checkpoint_interval}/{args.video_interval} '
      f'epochs = {steps_per_epoch * max(args.video_interval, 1)} env steps '
      f'(1 epoch = {n_env} envs × {unroll} macros = {steps_per_epoch} steps); '
      f'early_stop_eval_success={args.early_stop_eval_success}; '
      f'videos → {video_dir}',
      flush=True)
  print('[discover] UCB '
        f'mean_ini={args.ucb_mean_ini} mean_tar={args.ucb_mean_tar} '
        f'std_ini={args.ucb_std_ini} std_tar={args.ucb_std_tar} '
        f'ensemble={args.ensemble_size} her_future={args.relabel_prob_future} '
        f'transform_ucb={args.transform_ucb} catselect={use_cat} '
        f'catwp={use_wp} '
        f'target_computation={args.target_computation}', flush=True)
  print('[discover] compile collect/eval (first unroll is slow)...', flush=True)

  n_prefill = max(int(args.min_replay_size) // max(n_env * unroll, 1) + 1, 2)
  while True:
    rng, k_c = jax.random.split(rng)
    goals = jnp.broadcast_to(g_star, (n_env, goal_dim))
    env_state = bb_env.set_per_env_goals(env_state, goals, mocap_targets, n_task)
    env_state, data, hit, k_c = collect_unroll(
        env_state, learner.actor.params, goals, k_c, g_star)
    buf = insert_unroll(buf, data)
    env_steps += n_env * unroll
    print(f'[discover] prefill size_T={int(buf.size)} env_steps={env_steps}',
          flush=True)
    n_prefill -= 1
    if int(buf.size) >= 2 * ep_len and n_prefill <= 0:
      break

  last_eval_s = float('nan')
  last_eval_endpoint = float('nan')
  early_stop_thresh = float(args.early_stop_eval_success)

  while env_steps < args.num_timesteps:
    epoch += 1
    t_epoch = time.time()
    rng, k_step = jax.random.split(rng)
    (learner, env_state, buf, _, ucb_w, data, hit, goal_dist_to_final,
     c_loss, a_loss) = training_step(
         learner, env_state, buf, k_step, ucb_w, g_star)
    jax.block_until_ready(c_loss)
    env_steps += n_env * unroll
    hit_np = np.asarray(hit)
    hit_rate = float(np.mean(hit_np))
    ep_task = np.max(np.asarray(data['task_success']), axis=0)
    for x in ep_task.tolist():
      recent_task_success.append(float(x))
    for x in hit_np.tolist():
      recent_goal_hit.append(float(x))

    do_eval = (epoch % args.eval_interval == 0) or epoch == 1
    do_ckpt = (epoch % args.checkpoint_interval == 0) or epoch == 1
    do_video = video_enabled and (epoch % args.video_interval == 0)
    early_stop = False
    if do_ckpt:
      _save_ckpt(epoch, env_steps)
    if do_eval:
      eval_keys = jax.random.split(k_eval, n_eval)
      if n_eval != n_env:
        raise ValueError('num_eval_envs must equal num_envs for this 1-env JIT')
      est = reset_fn(eval_keys)
      # Primary eval matches NF/PPO BB: PD metrics['success'] (any microstep
      # in a macro). Endpoint 2cm check is logged separately.
      last_eval_endpoint, last_eval_s = eval_unroll(
          est, learner.actor.params, g_star)
      last_eval_endpoint = float(last_eval_endpoint)
      last_eval_s = float(last_eval_s)
      if last_eval_s > 0.0:
        # Decay exactly once per fresh positive evaluation.  Do not
        # reuse a cached positive eval on intervening training iterations.
        ucb_w = ucb_w.at[2].set(jnp.maximum(
            0.0, ucb_w[2] - float(args.eval_std_ini_decay)))
      if early_stop_thresh >= 0.0 and last_eval_s >= early_stop_thresh:
        early_stop = True
        if not do_ckpt:
          _save_ckpt(epoch, env_steps)
    if do_video and not early_stop:
      t_vid = time.time()
      try:
        if vid_env is None:
          vid_env, vid_mocap, vid_ep_len, _ = discover_video.init_video_env(
              env_name,
              pd_duration=args.pd_duration,
              permute_start_boxes=args.permute_start_boxes,
              fixed_start_x=fx,
          )
          vid_run = discover_video.make_rollout_fn(
              vid_env, vid_mocap, actor_def, n_cubes, g_star, vid_ep_len,
              categorical_select=use_cat, categorical_select_waypoint=use_wp)
          print('[discover] compiled video env (first clip is slow)...',
                flush=True)
        rng, k_vid = jax.random.split(rng)
        states = vid_run(learner.actor.params, k_vid)
        jax.block_until_ready(states.data.qpos)
        out_mp4 = str(video_dir / f'epoch_{int(epoch):07d}_s{int(env_steps)}.mp4')
        n_frames, n_on, ep_succ = discover_video.write_rollout_mp4(
            vid_env, states, out_mp4, fps=10)
        print(
            f'[discover] video {out_mp4} frames={n_frames} overlay_on={n_on} '
            f'ep_success={ep_succ:.3f} wall={time.time()-t_vid:.1f}s',
            flush=True)
      except Exception:
        traceback.print_exc()
        print('[discover] video failed; disabling further in-train video',
              flush=True)
        video_enabled = False

    sps = (n_env * unroll) / max(time.time() - t_epoch, 1e-6)
    train_s = float(np.mean(recent_task_success)) if recent_task_success else float('nan')
    row = {
        'global_step': int(env_steps),
        'epoch': int(epoch),
        'train_success_1000': train_s,
        'train_goal_hit_1000': (
            float(np.mean(recent_goal_hit)) if recent_goal_hit else float('nan')),
        # Primary success matches NF/PPO BB (PD any-microstep).
        'success': last_eval_s,
        'eval_success_any_microstep': last_eval_s,
        'eval_success_endpoint': last_eval_endpoint,
        'critic_loss': float(c_loss),
        'actor_loss': float(a_loss),
        'ucb_mean_ini': float(ucb_w[0]),
        'ucb_mean_tar': float(ucb_w[1]),
        'ucb_std_ini': float(ucb_w[2]),
        'ucb_std_tar': float(ucb_w[3]),
        'goal_dist_to_final': float(goal_dist_to_final),
        'goal_hit_rate': hit_rate,
        'sps': float(sps),
        'grad_steps': int(learner.grad_steps),
    }
    if epoch % args.log_interval == 0:
      _csv_write(csv_path, row, header_written)
      print(
          f'[discover] step={env_steps} epoch={epoch} '
          f'train_success_1000={train_s:.4f} '
          f'eval_success={last_eval_s:.4f} '
          f'eval_endpoint={last_eval_endpoint:.4f} '
          f'goal_dist_to_final={float(goal_dist_to_final):.4f} '
          f'goal_hit={hit_rate:.3f} '
          f'ucb_w=['
          f'{float(ucb_w[0]):.3f},{float(ucb_w[1]):.3f},'
          f'{float(ucb_w[2]):.3f}]',
          flush=True)
    if early_stop:
      print(
          f'[discover] early_stop eval_success={last_eval_s:.4f} '
          f'>= {early_stop_thresh:.4f} step={env_steps} epoch={epoch}',
          flush=True)
      break

  print(f'[discover] done env_steps={env_steps} wall={time.time()-t0:.0f}s',
        flush=True)


if __name__ == '__main__':
  main(tyro.cli(Args))
