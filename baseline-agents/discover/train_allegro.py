"""DISCOVER on Isaac Gym AllegroKukaThrow.

Same TD3 + HER + UCB SGD as ``train.py`` (JaxGCRL scan / BB recipe). Collect
is one PhysX sim (typically E=1024) instead of MJX.

Goal representation: object xyz (3D) = last 3 dims of the 49-D throw state.
Actor input is [state_49 | goal_3] = 52D. UTD matches BB: 32/128 = 0.25.
Replay time axis default 25000, same as JaxGCRL / BB DISCOVER.
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

xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
  os.environ['XLA_FLAGS'] = xla_flags + ' --xla_gpu_triton_gemm_any=True'

# PhysX must own CUDA before JAX.
from envs.isaacgym_physx_bootstrap import (  # noqa: E402
    maybe_create_from_argv as _ig_boot,
    take_prebuilt_env as _ig_take,
)
_ig_boot()

import sgcrl_jax_acme_compat  # noqa: E402,F401

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.training.train_state import TrainState

import networks as nets


def _soft_update(target, online, tau: float):
  return jax.tree_util.tree_map(
      lambda t, o: (1.0 - tau) * t + tau * o, target, online)


def _horizon_transform(q, gamma: float):
  q = jnp.maximum(q + 0.01, 0.001)
  return -jnp.log((1.0 - gamma) * q) / jnp.log(gamma)


def _sparse_ball_reward(achieved, goal, thresh: float):
  dist = jnp.linalg.norm(achieved - goal, axis=-1)
  return (dist < thresh).astype(jnp.float32)


def _torch_to_np(tensor) -> np.ndarray:
  return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


@dataclass
class Args:
  env: str = 'allegro_kuka_throw'
  seed: int = 0
  log_dir: str = (
      'logs/new_allegrobaselines/discover_nvidiainit_z055_ep150_300m_seed0_14h/')
  num_timesteps: int = 300_000_000
  num_envs: int = 1024
  batch_size: int = 128
  ensemble_size: int = 6
  ucb_mean_ini: float = 0.0
  ucb_mean_tar: float = 1.0
  ucb_std_ini: float = 0.0
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
  # Time axis of the JaxGCRL / BB DISCOVER buffer (not total transitions).
  max_replay_size: int = 25000
  adaptation_strategy: str = 'simple'
  goal_achievement_target: float = 0.5
  adaptation_rate: int = 100
  eval_std_ini_decay: float = 0.005
  continue_strategy: str = 'uniform'
  goal_reach_thresh: float = 0.1
  eval_interval: int = 10
  checkpoint_interval: int = 50
  log_interval: int = 1
  early_stop_eval_success: float = 0.9
  hidden: str = '256,256'


def _add_bool(p: argparse.ArgumentParser, name: str, default: bool) -> None:
  dest = name.replace('-', '_')
  grp = p.add_mutually_exclusive_group()
  grp.add_argument(f'--{name}', dest=dest, action='store_true')
  grp.add_argument(f'--no-{name}', dest=dest, action='store_false')
  p.set_defaults(**{dest: default})


def parse_args(argv=None) -> Args:
  """Argparse + parse_known_args so ``--isaacgym_*`` reaches the bootstrap."""
  d = Args()
  p = argparse.ArgumentParser(description='DISCOVER on AllegroKukaThrow')
  p.add_argument('--env', default=d.env)
  p.add_argument('--seed', type=int, default=d.seed)
  p.add_argument('--log-dir', dest='log_dir', default=d.log_dir)
  p.add_argument('--num-timesteps', dest='num_timesteps', type=int,
                 default=d.num_timesteps)
  p.add_argument('--num-envs', dest='num_envs', type=int, default=d.num_envs)
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
  p.add_argument('--eval-std-ini-decay', dest='eval_std_ini_decay',
                 type=float, default=d.eval_std_ini_decay)
  p.add_argument('--continue-strategy', dest='continue_strategy',
                 default=d.continue_strategy)
  p.add_argument('--goal-reach-thresh', dest='goal_reach_thresh', type=float,
                 default=d.goal_reach_thresh)
  p.add_argument('--eval-interval', dest='eval_interval', type=int,
                 default=d.eval_interval)
  p.add_argument('--checkpoint-interval', dest='checkpoint_interval', type=int,
                 default=d.checkpoint_interval)
  p.add_argument('--log-interval', dest='log_interval', type=int,
                 default=d.log_interval)
  p.add_argument('--early-stop-eval-success', dest='early_stop_eval_success',
                 type=float, default=d.early_stop_eval_success)
  p.add_argument('--hidden', default=d.hidden)
  ns, _unknown = p.parse_known_args(argv)
  allowed = {f.name for f in fields(Args)}
  return Args(**{k: v for k, v in vars(ns).items() if k in allowed})


class HostBuf:
  """JaxGCRL-shaped replay on host RAM.

  Same time axis as BB DISCOVER (default 25000). Kept off the GPU so Isaac
  Gym PhysX + TD3 SGD can share the device. Sampling / insert match
  ``train.py``; only the device changes.
  """

  def __init__(self, time_cap: int, n_env: int, obs_dim: int, act_dim: int):
    self.time_cap = int(time_cap)
    self.n_env = int(n_env)
    self.obs = np.zeros((time_cap, n_env, obs_dim), np.float32)
    self.next_obs = np.zeros((time_cap, n_env, obs_dim), np.float32)
    self.action = np.zeros((time_cap, n_env, act_dim), np.float32)
    self.reward = np.zeros((time_cap, n_env), np.float32)
    self.discount = np.zeros((time_cap, n_env), np.float32)
    self.truncation = np.zeros((time_cap, n_env), np.float32)
    self.traj_id = np.zeros((time_cap, n_env), np.float32)
    self.insert_pos = 0
    self.size = 0

  def insert(self, data) -> None:
    t = int(np.asarray(data['obs']).shape[0])
    pos = self.insert_pos
    idx = (pos + np.arange(t, dtype=np.int32)) % self.time_cap
    self.obs[idx] = np.asarray(data['obs'], dtype=np.float32)
    self.next_obs[idx] = np.asarray(data['next_obs'], dtype=np.float32)
    self.action[idx] = np.asarray(data['action'], dtype=np.float32)
    self.reward[idx] = np.asarray(data['reward'], dtype=np.float32)
    self.discount[idx] = np.asarray(data['discount'], dtype=np.float32)
    self.truncation[idx] = np.asarray(data['truncation'], dtype=np.float32)
    self.traj_id[idx] = np.asarray(data['traj_id'], dtype=np.float32)
    self.insert_pos = (pos + t) % self.time_cap
    self.size = min(self.size + t, self.time_cap)

  def sample_trajs(self, key, ep_len: int):
    key, k_e, k_s = jax.random.split(key, 3)
    env_idx = np.asarray(jax.random.permutation(k_e, self.n_env))
    max_start = max(self.size - ep_len, 1)
    starts = np.asarray(jax.random.randint(k_s, (self.n_env,), 0, max_start))
    t_ix = (starts[:, None] + np.arange(ep_len)[None, :]) % max(self.size, 1)
    e_ix = env_idx[:, None]
    return key, {
        'obs': jnp.asarray(self.obs[t_ix, e_ix]),
        'next_obs': jnp.asarray(self.next_obs[t_ix, e_ix]),
        'action': jnp.asarray(self.action[t_ix, e_ix]),
        'reward': jnp.asarray(self.reward[t_ix, e_ix]),
        'discount': jnp.asarray(self.discount[t_ix, e_ix]),
        'truncation': jnp.asarray(self.truncation[t_ix, e_ix]),
        'traj_id': jnp.asarray(self.traj_id[t_ix, e_ix]),
    }

  def sample_cand(self, key, n_goals: int):
    key, k_t, k_e = jax.random.split(key, 3)
    t_ix = np.asarray(jax.random.randint(k_t, (n_goals,), 0, max(self.size, 1)))
    e_ix = np.asarray(jax.random.randint(k_e, (n_goals,), 0, self.n_env))
    return jnp.asarray(self.obs[t_ix, e_ix]), key


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
  env = _ig_take()
  if env is None:
    raise RuntimeError(
        'Pass --env=allegro_kuka_throw and --num-envs=... so '
        'isaacgym_physx_bootstrap can create PhysX before JAX.')

  hidden = tuple(int(x) for x in args.hidden.split(',') if x.strip())
  env_name = args.env
  n_env = int(env.num_envs)
  state_dim = int(env.obs_dim)
  goal_dim = int(env.goal_dim)
  act_dim = int(env.action_dim)
  ep_len = int(getattr(env, 'max_episode_steps', 150))
  if state_dim != 49 or goal_dim != 3:
    raise ValueError(
        'DISCOVER Allegro adapter expects default throw packing '
        f'(state=49 goal=3), got state={state_dim} goal={goal_dim}. '
        'Do not pass palm_goal / joint_goal / control_sanity.')
  if args.num_envs != n_env:
    print(f'[discover] num_envs arg={args.num_envs} -> PhysX E={n_env}',
          flush=True)

  # g_star is the packed object-xyz task target (bucket xy, goal_z).
  packed0 = _torch_to_np(env.reset())
  g_star_np = np.asarray(packed0[0, state_dim:], dtype=np.float32).copy()
  g_star = jnp.asarray(g_star_np, dtype=jnp.float32)
  obj_idx = slice(state_dim - goal_dim, state_dim)
  unroll = ep_len
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
        f'episode_length*num_envs = {ep_len * n_env} must divide '
        f'batch_size={batch}')

  obs_dim = state_dim + goal_dim
  actor_def = nets.TanhActor(action_dim=act_dim, hidden=hidden)
  q_def = nets.EnsembleQ(
      n_critics=n_critics, hidden=hidden, activate_final=args.activate_final)

  rng = jax.random.PRNGKey(args.seed)
  rng, k_a, k_q, _k_buf = jax.random.split(rng, 4)
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
    return obs[..., obj_idx]

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
    r = _sparse_ball_reward(achieved_of(new_obs), new_g, thresh)
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

  def train_steps(learner, traj, key):
    key, k_her, k_perm, k_upd = jax.random.split(key, 4)
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
    return learner, jnp.mean(c_loss), jnp.mean(a_loss)

  def scan_train_steps(learner, buf, key):
    c_losses = []
    a_losses = []
    for _ in range(n_inner):
      key, k_samp, k = jax.random.split(key, 3)
      _, traj = buf.sample_trajs(k_samp, ep_len)
      learner, c_loss, a_loss = train_steps(learner, traj, k)
      c_losses.append(c_loss)
      a_losses.append(a_loss)
    return (
        learner, buf, key,
        jnp.mean(jnp.stack(c_losses)), jnp.mean(jnp.stack(a_losses)))

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

  train_steps = jax.jit(train_steps)
  ucb_select = jax.jit(ucb_select)
  adapt_ucb = jax.jit(adapt_ucb)
  policy_act = jax.jit(policy_act, static_argnames=('explore',))
  det_act = jax.jit(det_act)

  device = env.device

  def collect_unroll(actor_params, goals, key, raw_obs):
    """One episode per env. ``goals`` are (E, 3) UCB object-xyz targets."""
    goals_np = np.asarray(goals, dtype=np.float32)
    task_np = g_star_np.reshape(1, 3)
    state = np.asarray(raw_obs[:, :state_dim], dtype=np.float32)
    unach = np.ones((n_env,), dtype=np.float32)
    traj_id = np.arange(n_env, dtype=np.float32)
    obs_l, nobs_l, act_l, rew_l, disc_l, trunc_l, task_l = (
        [], [], [], [], [], [], [])
    min_dist_ucb = np.full((n_env,), np.inf, dtype=np.float32)
    min_dist_task = np.full((n_env,), np.inf, dtype=np.float32)
    end_dist_task = np.zeros((n_env,), dtype=np.float32)
    total_rew_steps = 0.0
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
      a = np.clip(np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
      act_t = torch.from_numpy(a).to(device)
      nraw_t, _env_r, done_t = env.step(act_t)
      nraw = _torch_to_np(nraw_t)
      done = _torch_to_np(done_t).reshape(-1)
      task_s = _torch_to_np(env.success()).reshape(-1)
      nstate = nraw[:, :state_dim]
      npacked = pack_obs(nstate, goals_np)
      obj = state[:, obj_idx]
      r = np.asarray(
          _sparse_ball_reward(
              jnp.asarray(obj), jnp.asarray(goals_np), thresh))
      unach = unach * (1.0 - r)
      trunc = done.astype(np.float32)
      obs_l.append(packed)
      nobs_l.append(npacked)
      act_l.append(a.astype(np.float32))
      rew_l.append(r.astype(np.float32))
      disc_l.append(1.0 - trunc)
      trunc_l.append(trunc)
      task_l.append(task_s)
      d_ucb = np.linalg.norm(obj - goals_np, axis=-1)
      d_task = np.linalg.norm(obj - task_np, axis=-1)
      min_dist_ucb = np.minimum(min_dist_ucb, d_ucb)
      min_dist_task = np.minimum(min_dist_task, d_task)
      end_dist_task = d_task
      total_rew_steps += float(np.sum(r))
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
        'min_dist_ucb': float(np.mean(min_dist_ucb)),
        'min_dist_task': float(np.mean(min_dist_task)),
        'end_dist_task': float(np.mean(end_dist_task)),
        'raw_rew_frac': total_rew_steps / max(unroll * n_env, 1),
    }
    return data, 1.0 - unach, key, coll_stats

  def eval_unroll(actor_params) -> float:
    raw = _torch_to_np(env.reset())
    g = np.broadcast_to(g_star_np, (n_env, goal_dim))
    state = raw[:, :state_dim]
    ep_succ = np.zeros((n_env,), dtype=np.float32)
    for _ in range(unroll):
      packed = pack_obs(state, g)
      a = np.clip(np.asarray(det_act(actor_params, packed)), -1.0, 1.0)
      act_t = torch.from_numpy(a.astype(np.float32)).to(device)
      nraw_t, _env_r, _done = env.step(act_t)
      ep_succ = np.maximum(ep_succ, _torch_to_np(env.success()).reshape(-1))
      state = _torch_to_np(nraw_t)[:, :state_dim]
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
      'action_dim': act_dim,
      'g_star_3d': g_star_np.tolist(),
      'goal_reach_thresh': float(args.goal_reach_thresh),
      'eval_interval': int(args.eval_interval),
      'checkpoint_interval': int(args.checkpoint_interval),
      'num_envs': n_env,
      'batch_size': batch,
      'train_step_multiplier': n_inner,
      'utd': utd,
      'replay_time_cap': int(time_cap),
      'trainer': 'jaxgcrl_scan_allegro_isaacgym_hostreplay',
      'ucb': [args.ucb_mean_ini, args.ucb_mean_tar,
              args.ucb_std_ini, args.ucb_std_tar],
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

  print('[discover] allocate host replay '
        f'T={time_cap} E={n_env} obs={obs_dim} act={act_dim} ...', flush=True)
  buf = HostBuf(time_cap, n_env, obs_dim, act_dim)
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
      f'trainer=jaxgcrl_scan_allegro_isaacgym_hostreplay',
      flush=True)
  print(
      f'[discover] eval every {args.eval_interval} epochs '
      f'(~{steps_per_epoch * args.eval_interval} env steps); '
      f'ckpt every {args.checkpoint_interval} epochs; '
      f'1 epoch = {n_env} envs × {unroll} = {steps_per_epoch} steps',
      flush=True)
  print('[discover] UCB '
        f'mean_ini={args.ucb_mean_ini} mean_tar={args.ucb_mean_tar} '
        f'std_ini={args.ucb_std_ini} std_tar={args.ucb_std_tar} '
        f'ensemble={args.ensemble_size} her_future={args.relabel_prob_future} '
        f'transform_ucb={args.transform_ucb} '
        f'target_computation={args.target_computation}', flush=True)
  print('[discover] compile SGD (first scan is slow)...', flush=True)

  n_prefill = max(int(args.min_replay_size) // max(n_env * unroll, 1) + 1, 2)
  goals_star = np.broadcast_to(g_star_np, (n_env, goal_dim))
  while True:
    rng, k_c = jax.random.split(rng)
    raw = _torch_to_np(env.reset())
    data, hit, k_c, _cs = collect_unroll(learner.actor.params, goals_star, k_c, raw)
    buf.insert(data)
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
    raw = _torch_to_np(env.reset())
    s0 = jnp.asarray(raw[:, :state_dim], dtype=jnp.float32)
    cand, k_ucb = buf.sample_cand(k_ucb, n_goals)
    goals = ucb_select(
        learner.q.params, learner.target_actor, cand, s0, g_star, ucb_w)
    jax.block_until_ready(goals)
    goals_np_arr = np.asarray(goals, dtype=np.float32)
    goal_dist_to_task = float(np.mean(
        np.linalg.norm(goals_np_arr - g_star_np.reshape(1, 3), axis=-1)))
    data, hit, k_col, coll_stats = collect_unroll(
        learner.actor.params, goals, k_col, raw)
    buf.insert(data)
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
      if last_eval_s > 0.0:
        ucb_w = ucb_w.at[2].set(jnp.maximum(
            0.0, ucb_w[2] - float(args.eval_std_ini_decay)))
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
        'ucb_std_ini': float(ucb_w[2]),
        'ucb_std_tar': float(ucb_w[3]),
        'goal_hit_coll': hit_rate,
        'goal_dist_to_task': goal_dist_to_task,
        'min_dist_ucb': coll_stats['min_dist_ucb'],
        'min_dist_task': coll_stats['min_dist_task'],
        'end_dist_task': coll_stats['end_dist_task'],
        'raw_rew_frac': coll_stats['raw_rew_frac'],
        'buf_fill': buf_fill,
        'sps': float(sps),
        'grad_steps': int(learner.grad_steps),
    }
    if epoch % args.log_interval == 0:
      _csv_write(csv_path, row, header_written)
      ucb_goals_str = ' '.join(
          f'[{g[0]:.2f},{g[1]:.2f},{g[2]:.2f}]' for g in goals_np_arr[:8])
      print(
          f'[discover] step={env_steps} epoch={epoch} '
          f'train_succ={train_s:.4f} '
          f'eval_succ={last_eval_s:.4f} '
          f'goal_hit_coll={hit_rate:.3f} '
          f'raw_rew_frac={coll_stats["raw_rew_frac"]:.4f} '
          f'min_d_ucb={coll_stats["min_dist_ucb"]:.3f} '
          f'min_d_task={coll_stats["min_dist_task"]:.3f} '
          f'end_d_task={coll_stats["end_dist_task"]:.3f} '
          f'goal_dist={goal_dist_to_task:.3f} '
          f'ucb_w=[{float(ucb_w[0]):.2f},{float(ucb_w[1]):.2f}] '
          f'buf={buf_fill:.2f} '
          f'critic={row["critic_loss"]:.4f} actor={row["actor_loss"]:.4f} '
          f'sps={sps:.0f} eta_300M={eta_h:.1f}h wall={time.time()-t0:.0f}s',
          flush=True)
      print(f'[discover]   ucb_goals[:8]={ucb_goals_str}', flush=True)
    if early_stop:
      print(
          f'[discover] early_stop eval_success={last_eval_s:.4f} '
          f'>= {early_stop_thresh:.4f} step={env_steps} epoch={epoch}',
          flush=True)
      break

  _save_ckpt(epoch, env_steps)
  print(f'[discover] done env_steps={env_steps} wall={time.time()-t0:.0f}s',
        flush=True)


if __name__ == '__main__':
  main(parse_args())
