"""DISCOVER on Sawyer bin / peg.

Same TD3 + HER + UCB SGD as ``train.py`` (JaxGCRL scan). Collect is gym /
MetaWorld (CPU, 4 envs) instead of BuilderBench MJX.

Goal representation: object xyz (3D), matching the original DISCOVER repo's
``goal_indices_2`` (task-relevant dims only).  Actor input is [state_7D|goal_3D]=10D.
UTD matches DISCOVER ARM: ``train_step_multiplier / batch_size ≈ 32/128 = 0.25``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Deque, Dict

_REPO = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
  sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))

os.environ.setdefault('MUJOCO_GL', 'osmesa')
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
  os.environ['XLA_FLAGS'] = xla_flags + ' --xla_gpu_triton_gemm_any=True'

import sgcrl_jax_acme_compat  # noqa: E402,F401

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import networks as nets
import sawyer_env as saw


def _soft_update(target, online, tau: float):
  return jax.tree_util.tree_map(
      lambda t, o: (1.0 - tau) * t + tau * o, target, online)


def _horizon_transform(q, gamma: float):
  q = jnp.maximum(q + 0.01, 0.001)
  return -jnp.log((1.0 - gamma) * q) / jnp.log(gamma)


@dataclass
class Args:
  env: str = 'sawyer_bin'
  seed: int = 0
  log_dir: str = 'logs/discover_sawyer_bin_e4_40m_rand_24h/'
  num_timesteps: int = 40_000_000
  num_envs: int = 4
  num_eval_envs: int = 4
  batch_size: int = 150
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
  # ARM: E=128, batch=128, multiplier=32 → UTD=0.25. Here batch=150
  # so (ep_len * 4) divides; 38/150 ≈ 0.253.
  train_step_multiplier: int = 38
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
  max_replay_size: int = 25000
  adaptation_strategy: str = 'simple'
  goal_achievement_target: float = 0.5
  adaptation_rate: int = 100
  continue_strategy: str = 'uniform'
  goal_reach_thresh: float = 0.1
  randomize_init: bool = True
  bin_safe_grasp_reset: bool = False
  bin_randomize_tcp_z: bool = False
  # ~10k env steps, matching Sawyer PPO eval_interval=10 at T=256, E=4.
  eval_interval: int = 17
  checkpoint_interval: int = 200
  log_interval: int = 1
  # Stop once a fresh eval reaches this success rate (>=). Set <0 to disable.
  early_stop_eval_success: float = 0.9
  hidden: str = '256,256'


def _add_bool(p: argparse.ArgumentParser, name: str, default: bool) -> None:
  dest = name.replace('-', '_')
  grp = p.add_mutually_exclusive_group()
  grp.add_argument(f'--{name}', dest=dest, action='store_true')
  grp.add_argument(f'--no-{name}', dest=dest, action='store_false')
  p.set_defaults(**{dest: default})


def parse_args(argv=None) -> Args:
  """Argparse instead of tyro: ``sgcrl_flow`` does not ship tyro."""
  d = Args()
  p = argparse.ArgumentParser(description='DISCOVER on Sawyer bin/peg')
  p.add_argument('--env', default=d.env)
  p.add_argument('--seed', type=int, default=d.seed)
  p.add_argument('--log-dir', dest='log_dir', default=d.log_dir)
  p.add_argument('--num-timesteps', dest='num_timesteps', type=int,
                 default=d.num_timesteps)
  p.add_argument('--num-envs', dest='num_envs', type=int, default=d.num_envs)
  p.add_argument('--num-eval-envs', dest='num_eval_envs', type=int,
                 default=d.num_eval_envs)
  p.add_argument('--batch-size', dest='batch_size', type=int,
                 default=d.batch_size)
  p.add_argument('--ensemble-size', dest='ensemble_size', type=int,
                 default=d.ensemble_size)
  p.add_argument('--ucb-mean-ini', dest='ucb_mean_ini', type=float,
                 default=d.ucb_mean_ini)
  p.add_argument('--ucb-mean-tar', dest='ucb_mean_tar', type=float,
                 default=d.ucb_mean_tar)
  p.add_argument('--ucb-std-ini', dest='ucb_std_ini', type=float,
                 default=d.ucb_std_ini)
  p.add_argument('--ucb-std-tar', dest='ucb_std_tar', type=float,
                 default=d.ucb_std_tar)
  _add_bool(p, 'transform-ucb', d.transform_ucb)
  _add_bool(p, 'activate-final', d.activate_final)
  p.add_argument('--relabel-prob-future', dest='relabel_prob_future',
                 type=float, default=d.relabel_prob_future)
  p.add_argument('--relabel-prob-uniform', dest='relabel_prob_uniform',
                 type=float, default=d.relabel_prob_uniform)
  p.add_argument('--relabel-prob-geometric', dest='relabel_prob_geometric',
                 type=float, default=d.relabel_prob_geometric)
  p.add_argument('--num-goals', dest='num_goals', type=int, default=d.num_goals)
  p.add_argument('--train-step-multiplier', dest='train_step_multiplier',
                 type=int, default=d.train_step_multiplier)
  p.add_argument('--critic-lr', dest='critic_lr', type=float, default=d.critic_lr)
  p.add_argument('--actor-lr', dest='actor_lr', type=float, default=d.actor_lr)
  p.add_argument('--tau', type=float, default=d.tau)
  p.add_argument('--tau-policy', dest='tau_policy', type=float,
                 default=d.tau_policy)
  p.add_argument('--discount', type=float, default=d.discount)
  p.add_argument('--policy-delay', dest='policy_delay', type=int,
                 default=d.policy_delay)
  p.add_argument('--exploration-noise', dest='exploration_noise', type=float,
                 default=d.exploration_noise)
  p.add_argument('--noise-clip', dest='noise_clip', type=float,
                 default=d.noise_clip)
  p.add_argument('--smoothing-noise', dest='smoothing_noise', type=float,
                 default=d.smoothing_noise)
  p.add_argument('--target-computation', dest='target_computation',
                 default=d.target_computation)
  p.add_argument('--min-replay-size', dest='min_replay_size', type=int,
                 default=d.min_replay_size)
  p.add_argument('--max-replay-size', dest='max_replay_size', type=int,
                 default=d.max_replay_size)
  p.add_argument('--adaptation-strategy', dest='adaptation_strategy',
                 default=d.adaptation_strategy)
  p.add_argument('--goal-achievement-target', dest='goal_achievement_target',
                 type=float, default=d.goal_achievement_target)
  p.add_argument('--adaptation-rate', dest='adaptation_rate', type=int,
                 default=d.adaptation_rate)
  p.add_argument('--continue-strategy', dest='continue_strategy',
                 default=d.continue_strategy)
  p.add_argument('--goal-reach-thresh', dest='goal_reach_thresh', type=float,
                 default=d.goal_reach_thresh)
  _add_bool(p, 'randomize-init', d.randomize_init)
  _add_bool(p, 'bin-safe-grasp-reset', d.bin_safe_grasp_reset)
  _add_bool(p, 'bin-randomize-tcp-z', d.bin_randomize_tcp_z)
  p.add_argument('--eval-interval', dest='eval_interval', type=int,
                 default=d.eval_interval)
  p.add_argument('--checkpoint-interval', dest='checkpoint_interval', type=int,
                 default=d.checkpoint_interval)
  p.add_argument('--log-interval', dest='log_interval', type=int,
                 default=d.log_interval)
  p.add_argument('--early-stop-eval-success', dest='early_stop_eval_success',
                 type=float, default=d.early_stop_eval_success)
  p.add_argument('--hidden', default=d.hidden)
  ns = p.parse_args(argv)
  allowed = {f.name for f in fields(Args)}
  return Args(**{k: v for k, v in vars(ns).items() if k in allowed})


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
  state_dim, goal_dim, act_dim, ep_len, task_xyz = saw.env_layout(env_name)
  # g_star is the 3-D object-xyz task target (DISCOVER goal space = obj xyz only).
  # Bug-fix: was task_goal_7d (7D ψ), which made HER sparse-ball reward near-zero.
  g_star_np = saw.task_goal_3d(env_name)
  g_star = jnp.asarray(g_star_np, dtype=jnp.float32)
  unroll = ep_len
  n_env = int(args.num_envs)
  n_eval = int(args.num_eval_envs)
  batch = int(args.batch_size)
  n_goals = int(args.num_goals)
  n_critics = int(args.ensemble_size)
  n_inner = int(args.train_step_multiplier)
  gamma = float(args.discount)
  thresh = float(args.goal_reach_thresh)
  p_future = float(args.relabel_prob_future)
  p_unif = float(args.relabel_prob_uniform)
  p_geom = float(args.relabel_prob_geometric)
  p_none = 1.0 - (p_future + p_unif + p_geom)
  if p_none < -1e-6:
    raise ValueError('HER relabel probabilities must sum to <= 1')
  p_none = max(p_none, 0.0)

  time_cap = int(args.max_replay_size)
  if time_cap < 2 * ep_len:
    raise ValueError(f'replay time_cap={time_cap} < 2*ep_len={2 * ep_len}')
  if (ep_len * n_env) % batch != 0:
    raise ValueError(
        f'episode_length*num_envs = {ep_len * n_env} must divide batch_size={batch}')

  obs_dim = state_dim + goal_dim
  actor_def = nets.TanhActor(action_dim=act_dim, hidden=hidden)
  q_def = nets.EnsembleQ(
      n_critics=n_critics, hidden=hidden, activate_final=args.activate_final)

  rng = jax.random.PRNGKey(args.seed)
  rng, k_a, k_q, k_buf = jax.random.split(rng, 4)
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

  def achieved_of(obs):
    # Object xyz = dims 4:7 of the 7D state (= dims 4:7 of the packed obs too).
    # Bug-fix: was obs[..., :state_dim] (full 7D), making sparse-ball reward
    # require all of hand+gripper+obj to match within 0.1 L2 — near-zero probability.
    return obs[..., 4:7]

  def pack_obs(state, goals):
    return np.concatenate(
        [np.asarray(state, dtype=np.float32), np.asarray(goals, dtype=np.float32)],
        axis=-1)

  def policy_act(params, obs, key, explore):
    mean = apply_actor(params, obs)
    return nets.tanh_actor_action(
        mean, key, explore, args.exploration_noise, args.noise_clip)

  def det_act(params, obs):
    return apply_actor(params, obs)

  def compute_value(q_params, actor_params, observations):
    actions = det_act(actor_params, observations)
    q = apply_q(q_params, observations, actions)
    qt = _horizon_transform(q, gamma)
    if args.transform_ucb:
      return jnp.mean(qt, axis=-1), jnp.nan_to_num(jnp.std(qt, axis=-1), nan=0.0)
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
    r = saw.sparse_ball_reward(achieved_of(new_obs), new_g, thresh)
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

  insert_unroll = jax.jit(insert_unroll)
  scan_train_steps = jax.jit(scan_train_steps)
  ucb_select = jax.jit(ucb_select)
  sample_cand = jax.jit(sample_cand)
  adapt_ucb = jax.jit(adapt_ucb)
  policy_act = jax.jit(policy_act, static_argnames=('explore',))
  det_act = jax.jit(det_act)

  vec = saw.SawyerVec(
      env_name, n_env, seed=args.seed, randomize_init=args.randomize_init,
      safe_grasp_reset=args.bin_safe_grasp_reset,
      randomize_tcp_z=args.bin_randomize_tcp_z)
  eval_vec = saw.SawyerVec(
      env_name, n_eval, seed=args.seed + 10_000,
      randomize_init=args.randomize_init,
      safe_grasp_reset=args.bin_safe_grasp_reset,
      randomize_tcp_z=args.bin_randomize_tcp_z)

  def collect_unroll(actor_params, goals, key, raw_obs):
    """One 150-step episode per env.

    ``raw_obs``  – just-reset 14-D gym obs (state_7D | env_internal_goal_7D).
    ``goals``    – (n_env, 3) object-xyz curriculum goals from UCB.
    Actor input  – pack_obs(state_7D, goals_3D) = 10-D.

    Returns (data, hit, key, coll_stats) where coll_stats is a dict of
    per-episode diagnostic scalars for logging.
    """
    goals_np = np.asarray(goals, dtype=np.float32)   # (n_env, 3)
    task_np  = task_xyz.reshape(1, 3)                 # (1, 3) broadcast target
    state = np.asarray(raw_obs[:, :state_dim], dtype=np.float32)
    unach = np.ones((n_env,), dtype=np.float32)
    traj_id = np.arange(n_env, dtype=np.float32)
    obs_l, nobs_l, act_l, rew_l, disc_l, trunc_l, task_l = (
        [], [], [], [], [], [], [])
    # Diagnostic accumulators (updated each step).
    min_dist_ucb  = np.full((n_env,), np.inf, dtype=np.float32)
    min_dist_task = np.full((n_env,), np.inf, dtype=np.float32)
    end_dist_task = np.zeros((n_env,), dtype=np.float32)
    total_rew_steps = 0.0   # sum of r across all (step, env) for raw_rew_frac
    for _ in range(unroll):
      key, k_act, k_rnd = jax.random.split(key, 3)
      packed = pack_obs(state, goals_np)
      pi_a = np.asarray(policy_act(actor_params, packed, k_act, True))
      rnd_a = np.asarray(
          jax.random.uniform(k_rnd, pi_a.shape, minval=-1.0, maxval=1.0))
      if args.continue_strategy == 'uniform':
        a = unach[:, None] * pi_a + (1.0 - unach)[:, None] * rnd_a
      else:
        a = pi_a
      nraw, _env_r, done = vec.step(a)
      nstate = nraw[:, :state_dim]
      npacked = pack_obs(nstate, goals_np)
      obj = state[:, 4:7]  # object xyz, (n_env, 3)
      # Use object xyz (dims 4:7) to match the 3D goal space.
      # Bug-fix: was full 7D `state` vs 3D goal → shape mismatch / wrong reward.
      r = np.asarray(
          saw.sparse_ball_reward(
              jnp.asarray(obj), jnp.asarray(goals_np), thresh))
      unach = unach * (1.0 - r)
      trunc = done.astype(np.float32)
      task_s = saw.mw_task_success(nraw, env_name, task_xyz)
      obs_l.append(packed)
      nobs_l.append(npacked)
      act_l.append(a.astype(np.float32))
      rew_l.append(r.astype(np.float32))
      disc_l.append(1.0 - trunc)
      trunc_l.append(trunc)
      task_l.append(task_s)
      # --- diagnostic tracking ---
      d_ucb  = np.linalg.norm(obj - goals_np,  axis=-1)  # (n_env,)
      d_task = np.linalg.norm(obj - task_np,   axis=-1)  # (n_env,)
      min_dist_ucb  = np.minimum(min_dist_ucb,  d_ucb)
      min_dist_task = np.minimum(min_dist_task, d_task)
      end_dist_task = d_task
      total_rew_steps += float(np.sum(r))
      # ---------------------------
      state = nstate
    data = {
        'obs': jnp.asarray(np.stack(obs_l, axis=0)),
        'next_obs': jnp.asarray(np.stack(nobs_l, axis=0)),
        'action': jnp.asarray(np.stack(act_l, axis=0)),
        'reward': jnp.asarray(np.stack(rew_l, axis=0)),
        'discount': jnp.asarray(np.stack(disc_l, axis=0)),
        'truncation': jnp.asarray(np.stack(trunc_l, axis=0)),
        'traj_id': jnp.broadcast_to(
            jnp.asarray(traj_id)[None, :], (unroll, n_env)),
        'task_success': jnp.asarray(np.stack(task_l, axis=0)),
    }
    coll_stats = {
        # min distance obj→UCB goal over the episode, per env then mean.
        # If goal_hit_coll=1, this must be < thresh.
        'min_dist_ucb': float(np.mean(min_dist_ucb)),
        # min distance obj→task_xyz over the episode (key convergence signal).
        'min_dist_task': float(np.mean(min_dist_task)),
        # distance obj→task_xyz at the final step.
        'end_dist_task': float(np.mean(end_dist_task)),
        # fraction of (step, env) pairs where UCB goal was hit (per-step rate).
        'raw_rew_frac': total_rew_steps / max(unroll * n_env, 1),
    }
    return data, 1.0 - unach, key, coll_stats

  def eval_unroll(actor_params) -> float:
    raw = eval_vec.reset_with_task_goal(task_xyz)
    g = np.broadcast_to(g_star_np, (n_eval, goal_dim))
    state = raw[:, :state_dim]
    ep_succ = np.zeros((n_eval,), dtype=np.float32)
    for _ in range(unroll):
      packed = pack_obs(state, g)
      a = np.asarray(det_act(actor_params, packed))
      nraw, env_r, _done = eval_vec.step(a)
      ep_succ = np.maximum(ep_succ, env_r.astype(np.float32))
      ep_succ = np.maximum(
          ep_succ, saw.mw_task_success(nraw, env_name, task_xyz))
      state = nraw[:, :state_dim]
    return float(np.mean(ep_succ))

  log_dir = Path(args.log_dir)
  log_dir.mkdir(parents=True, exist_ok=True)
  run_dir = log_dir / f'discover_{env_name}_{args.seed}'
  run_dir.mkdir(parents=True, exist_ok=True)
  csv_path = run_dir / 'logs.csv'
  ckpt_dir = run_dir / 'checkpoints'
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  header_written: list = []
  utd = float(n_inner) / float(batch)
  run_cfg = {
      'env': env_name,
      'seed': int(args.seed),
      'hidden': list(hidden),
      'goal_dim': goal_dim,
      'state_dim': state_dim,
      'randomize_init': bool(args.randomize_init),
      'bin_safe_grasp_reset': bool(args.bin_safe_grasp_reset),
      'bin_randomize_tcp_z': bool(args.bin_randomize_tcp_z),
      'goal_reach_thresh': float(args.goal_reach_thresh),
      'eval_interval': int(args.eval_interval),
      'checkpoint_interval': int(args.checkpoint_interval),
      'num_envs': n_env,
      'batch_size': batch,
      'train_step_multiplier': n_inner,
      'utd': utd,
      'replay_time_cap': int(time_cap),
      'trainer': 'jaxgcrl_scan_sawyer_gym',
      'g_star_3d': g_star_np.tolist(),   # object xyz (3D) task target
      'task_xyz': task_xyz.tolist(),
  }
  (run_dir / 'run_config.json').write_text(json.dumps(run_cfg, indent=2))

  def _save_ckpt(epoch_i: int, steps_i: int) -> None:
    payload = {
        'actor_params': jax.device_get(learner.actor.params),
        'epoch': int(epoch_i),
        'env_steps': int(steps_i),
        'hidden': list(hidden),
        'env': env_name,
        'action_dim': act_dim,
    }
    blob = pickle.dumps(payload)
    (ckpt_dir / 'latest.pkl').write_bytes(blob)
    named = ckpt_dir / f'ckpt_epoch_{int(epoch_i):07d}.pkl'
    named.write_bytes(blob)
    print(f'[discover] wrote {named} steps={steps_i}', flush=True)

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
  steps_per_epoch = n_env * unroll
  print(
      f'[discover] env={env_name} E={n_env} ep_len={ep_len} obs={obs_dim} '
      f'act={act_dim} state_dim={state_dim} goal_dim={goal_dim}(obj_xyz_3D) '
      f'g_star={g_star_np.tolist()} '
      f'updates/epoch={updates_per_epoch} UTD={utd:.3f} replay_T={time_cap} '
      f'trainer=jaxgcrl_scan_sawyer_gym randomize_init={args.randomize_init} '
      f'safe_grasp={args.bin_safe_grasp_reset} '
      f'tcp_z={args.bin_randomize_tcp_z}',
      flush=True)
  print(
      f'[discover] eval every {args.eval_interval} epochs '
      f'(~{steps_per_epoch * args.eval_interval} env steps); '
      f'ckpt every {args.checkpoint_interval} epochs; '
      f'1 epoch = {n_env} envs × {unroll} = {steps_per_epoch} steps; '
      f'40M @ this UTD needs ~{args.num_timesteps / max(steps_per_epoch, 1):.0f} epochs',
      flush=True)
  print('[discover] UCB '
        f'mean_ini={args.ucb_mean_ini} mean_tar={args.ucb_mean_tar} '
        f'std_ini={args.ucb_std_ini} std_tar={args.ucb_std_tar} '
        f'ensemble={args.ensemble_size} her_future={args.relabel_prob_future} '
        f'transform_ucb={args.transform_ucb} '
        f'target_computation={args.target_computation}', flush=True)
  print('[discover] compile SGD (first scan is slow)...', flush=True)

  n_prefill = max(int(args.min_replay_size) // max(n_env * unroll, 1) + 1, 2)
  goals_star = np.broadcast_to(g_star_np, (n_env, goal_dim))  # (n_env, 3)
  while True:
    rng, k_c = jax.random.split(rng)
    raw = vec.reset()
    data, hit, k_c, _cs = collect_unroll(learner.actor.params, goals_star, k_c, raw)
    buf = insert_unroll(buf, data)
    env_steps += n_env * unroll
    print(f'[discover] prefill size_T={int(buf.size)} env_steps={env_steps}',
          flush=True)
    n_prefill -= 1
    if int(buf.size) >= 2 * ep_len and n_prefill <= 0:
      break

  last_eval_s = float('nan')
  early_stop_thresh = float(args.early_stop_eval_success)

  while env_steps < args.num_timesteps:
    epoch += 1
    t_epoch = time.time()
    rng, k_ucb, k_col, k_sgd = jax.random.split(rng, 4)
    raw = vec.reset()
    s0 = jnp.asarray(raw[:, :state_dim], dtype=jnp.float32)
    cand, k_ucb = sample_cand(buf, k_ucb)
    goals = ucb_select(
        learner.q.params, learner.target_actor, cand, s0, g_star, ucb_w)
    jax.block_until_ready(goals)
    # Distance from UCB-selected goals (3D obj xyz) to the task goal.
    # Should shrink over training if the curriculum is converging toward the task.
    goals_np_arr = np.asarray(goals, dtype=np.float32)
    goal_dist_to_task = float(np.mean(
        np.linalg.norm(goals_np_arr - task_xyz.reshape(1, 3), axis=-1)))
    data, hit, k_col, coll_stats = collect_unroll(
        learner.actor.params, goals, k_col, raw)
    buf = insert_unroll(buf, data)
    ucb_w = adapt_ucb(ucb_w, jnp.mean(jnp.asarray(hit)))
    learner, buf, k_sgd, c_loss, a_loss = scan_train_steps(learner, buf, k_sgd)
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
    early_stop = False
    if do_ckpt:
      _save_ckpt(epoch, env_steps)
    if do_eval:
      last_eval_s = float(eval_unroll(learner.actor.params))
      if early_stop_thresh >= 0.0 and last_eval_s >= early_stop_thresh:
        early_stop = True
        if not do_ckpt:
          _save_ckpt(epoch, env_steps)

    sps = (n_env * unroll) / max(time.time() - t_epoch, 1e-6)
    eta_h = (args.num_timesteps - env_steps) / max(sps, 1e-6) / 3600.0
    train_s = float(np.mean(recent_task_success)) if recent_task_success else float('nan')
    buf_fill = float(int(buf.size)) / float(time_cap)
    row = {
        'global_step': int(env_steps),
        'epoch': int(epoch),
        'train_success_1000': train_s,
        'train_goal_hit_1000': (
            float(np.mean(recent_goal_hit)) if recent_goal_hit else float('nan')),
        'success': last_eval_s,
        'critic_loss': float(c_loss),
        'actor_loss': float(a_loss),
        'ucb_mean_ini': float(ucb_w[0]),
        'ucb_mean_tar': float(ucb_w[1]),
        # goal_hit_coll: fraction of training envs that achieved the UCB
        # curriculum subgoal (obj xyz) at least once during COLLECTION.
        # This is NOT the HER critic reward rate.
        'goal_hit_coll': hit_rate,
        'goal_dist_to_task': goal_dist_to_task,
        # Diagnostics from collect_unroll:
        # min_dist_ucb: how close obj got to UCB goal (mean over envs).
        #   Should be < thresh when goal_hit_coll > 0 — sanity check.
        'min_dist_ucb': coll_stats['min_dist_ucb'],
        # min_dist_task: closest obj got to task_xyz in this episode.
        #   Key convergence signal: should decrease over training.
        'min_dist_task': coll_stats['min_dist_task'],
        # end_dist_task: obj distance to task_xyz at episode end.
        'end_dist_task': coll_stats['end_dist_task'],
        # raw_rew_frac: fraction of (step, env) pairs with UCB reward=1.
        #   Per-step density; goal_hit_coll is per-episode binary.
        'raw_rew_frac': coll_stats['raw_rew_frac'],
        'buf_fill': buf_fill,
        'sps': float(sps),
        'grad_steps': int(learner.grad_steps),
    }
    if epoch % args.log_interval == 0:
      _csv_write(csv_path, row, header_written)
      # UCB goal coords (4 envs × 3D) — lets you see WHERE the curriculum sends the agent.
      ucb_goals_str = ' '.join(
          f'[{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}]' for g in goals_np_arr)
      print(
          f'[discover] step={env_steps} epoch={epoch} '
          f'train_succ={train_s:.4f} '
          f'eval_succ={last_eval_s:.4f} '
          # goal_hit_coll = fraction of envs hitting the UCB subgoal (per-episode)
          f'goal_hit_coll={hit_rate:.3f} '
          f'raw_rew_frac={coll_stats["raw_rew_frac"]:.4f} '
          f'min_d_ucb={coll_stats["min_dist_ucb"]:.3f} '
          f'min_d_task={coll_stats["min_dist_task"]:.3f} '
          f'end_d_task={coll_stats["end_dist_task"]:.3f} '
          f'goal_dist={goal_dist_to_task:.3f} '
          f'ucb_w=[{float(ucb_w[0]):.2f},{float(ucb_w[1]):.2f}] '
          f'buf={buf_fill:.2f} '
          f'critic={row["critic_loss"]:.4f} actor={row["actor_loss"]:.4f} '
          f'sps={sps:.0f} eta_40M={eta_h:.1f}h wall={time.time()-t0:.0f}s',
          flush=True)
      print(f'[discover]   ucb_goals={ucb_goals_str}', flush=True)
    if early_stop:
      print(
          f'[discover] early_stop eval_success={last_eval_s:.4f} '
          f'>= {early_stop_thresh:.4f} step={env_steps} epoch={epoch}',
          flush=True)
      break

  _save_ckpt(epoch, env_steps)
  print(f'[discover] done env_steps={env_steps} wall={time.time()-t0:.0f}s',
        flush=True)
  vec.close()
  eval_vec.close()


if __name__ == '__main__':
  main(parse_args())
