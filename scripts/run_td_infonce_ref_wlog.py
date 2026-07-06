#!/usr/bin/env python3
"""Run the upstream TD-InfoNCE learner locally and log w / logits_w vectors.

Uses the cloned reference repo at ``external/td_infonce`` (CDPC network +
paper critic loss) with this project's env stack and ``sgcrl_jax_acme_compat``.

Outputs (compatible with ``scripts/watch_w_diag_distribution.py``):
  ``<log_dir>/logs/w_diag/vectors.csv``      — softmax IS weights diagonal
  ``<log_dir>/logs/logits_w/vectors.csv``    — min-Q logits_w diagonal pre-softmax

Example::

  python scripts/run_td_infonce_ref_wlog.py \\
      --env=sawyer_drawer_open --seed=0 \\
      --log_dir=logs/td_infonce_ref_drawer \\
      --learner_steps=512 --batch_size=256
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import sgcrl_jax_acme_compat  # noqa: F401 — must precede acme/jax imports

ROOT = Path(__file__).resolve().parent.parent
TD_ROOT = ROOT / 'external' / 'td_infonce'
sys.path.insert(0, str(TD_ROOT))
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import reverb
from acme import specs, types

import contrastive.utils as contrastive_utils
from td_infonce.config import TDInfoNCEConfig
from td_infonce.learning import TDInfoNCELearner
from td_infonce.networks import make_networks


def _obs_to_goal_np(obs, start_index, end_index):
  return np.asarray(
      contrastive_utils.obs_to_goal_2d(obs, start_index, end_index),
      dtype=np.float32)


def _sample_batch(episodes, batch_size, obs_dim, discount, start_index, end_index):
  """Sample a flat batch with HER + in-batch random goals (td_infonce semantics)."""
  obs_rows, next_rows, actions, future_goals = [], [], [], []
  for _ in range(batch_size):
    ep = episodes[np.random.randint(len(episodes))]
    t = np.random.randint(0, len(ep['observation']))
    obs_t = ep['observation'][t]
    next_t = ep['next_observation'][t]
    act_t = ep['action'][t]
    future_idx = np.arange(t, len(ep['observation']))
    weights = discount ** (future_idx - t).astype(np.float64)
    weights /= np.maximum(weights.sum(), 1e-8)
    g_i = int(np.random.choice(future_idx, p=weights))
    goal = _obs_to_goal_np(
        ep['observation'][g_i:g_i + 1, :obs_dim], start_index, end_index)[0]
    obs_rows.append(np.concatenate([obs_t[:obs_dim], goal]))
    next_rows.append(np.concatenate([next_t[:obs_dim], goal]))
    actions.append(act_t)
    future_goals.append(goal)

  obs = np.stack(obs_rows).astype(np.float32)
  next_obs = np.stack(next_rows).astype(np.float32)
  action = np.stack(actions).astype(np.float32)
  future_goal = np.stack(future_goals).astype(np.float32)
  states = obs[:, :obs_dim]
  shift = np.random.randint(0, batch_size)
  random_goal = _obs_to_goal_np(np.roll(states, shift, axis=0), start_index, end_index)
  return types.Transition(
      observation=obs,
      action=action,
      reward=np.zeros(batch_size, dtype=np.float32),
      discount=np.ones(batch_size, dtype=np.float32),
      next_observation=next_obs,
      extras={
          'future_goal': future_goal,
          'random_goal': random_goal.astype(np.float32),
      },
  )


class ReplayBatchIterator:
  """Infinite iterator yielding ``reverb.ReplaySample`` batches."""

  def __init__(self, episodes, config: TDInfoNCEConfig):
    self._episodes = episodes
    self._config = config
    self._batch_size = config.batch_size * config.num_sgd_steps_per_step

  def __iter__(self):
    return self

  def __next__(self):
    batch = _sample_batch(
        self._episodes,
        self._batch_size,
        self._config.obs_dim,
        self._config.discount,
        self._config.start_index,
        self._config.end_index,
    )
    info = reverb.SampleInfo(key=0, probability=0.0, table_size=0, priority=0.0)
    return reverb.ReplaySample(info=info, data=batch)


def _collect_episodes(env, num_transitions, obs_dim, seed):
  rng = np.random.RandomState(seed)
  episodes = deque(maxlen=5000)
  total = 0
  while total < num_transitions:
    ts = env.reset()
    obs_list, act_list, rew_list, disc_list, next_obs_list = [], [], [], [], []
    while not ts.last():
      obs = np.asarray(ts.observation, dtype=np.float32)
      action = rng.uniform(-1.0, 1.0, size=env.action_spec().shape).astype(np.float32)
      ts = env.step(action)
      next_obs = np.asarray(ts.observation, dtype=np.float32)
      obs_list.append(obs)
      act_list.append(action)
      rew_list.append(float(ts.reward))
      disc_list.append(float(ts.discount))
      next_obs_list.append(next_obs)
      total += 1
      if total >= num_transitions:
        break
    if obs_list:
      episodes.append({
          'observation': np.stack(obs_list, axis=0),
          'action': np.stack(act_list, axis=0),
          'reward': np.asarray(rew_list, dtype=np.float32),
          'discount': np.asarray(disc_list, dtype=np.float32),
          'next_observation': np.stack(next_obs_list, axis=0),
      })
  return list(episodes)


def _run_eval(
    *,
    learner: TDInfoNCELearner,
    networks,
    env_name: str,
    seed: int,
    start_index: int,
    end_index: int,
    learner_step: int,
    num_episodes: int,
    eval_logger,
) -> None:
  """Evaluate deterministic policy; log success_1000 like PPO eval."""
  from ppo_contrastive import fixed_goal_dict

  fixed_start_end = fixed_goal_dict.get(env_name)
  if fixed_start_end is None:
    print(f'[ref_td_infonce] skip eval: no fixed goal for env={env_name!r}')
    return

  success_obs = contrastive_utils.SuccessObserver()
  ep_metrics_list = []
  policy_params = learner.get_variables(['policy'])[0]

  @jax.jit
  def _policy_action(obs):
    dist = networks.policy_network.apply(
        policy_params, jnp.asarray(obs, dtype=jnp.float32)[None])
    return dist.mode()[0]

  for e_i in range(num_episodes):
    env, _ = contrastive_utils.make_environment(
        env_name, start_index, end_index,
        seed + 900_000 + learner_step * 100 + e_i,
        fixed_start_end=fixed_start_end)
    ts = env.reset()
    success_obs.observe_first(env, ts)
    ret_e, n_e = 0.0, 0
    while not ts.last():
      obs = np.asarray(ts.observation, dtype=np.float32)
      action = np.asarray(_policy_action(obs), dtype=np.float32)
      action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
      action = np.clip(action, -1.0, 1.0)
      ts = env.step(action)
      success_obs.observe(env, ts, action)
      ret_e += float(ts.reward or 0.0)
      n_e += 1
    ep_metrics = {
        'episode_return': ret_e,
        'episode_length': float(n_e),
    }
    ep_metrics.update(success_obs.get_metrics())
    ep_metrics_list.append(ep_metrics)

  agg = contrastive_utils.aggregate_eval_metrics(ep_metrics_list, learner_step)
  eval_logger.write(agg)
  print(f'[ref_td_infonce] eval step={learner_step} '
        f'success_1000={agg.get("success_1000", float("nan")):.3f}')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', default='sawyer_drawer_open')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--log_dir', default='logs/td_infonce_ref_wlog')
  parser.add_argument('--min_replay', type=int, default=10_000)
  parser.add_argument('--learner_steps', type=int, default=512)
  parser.add_argument('--batch_size', type=int, default=256)
  parser.add_argument('--num_sgd_steps_per_step', type=int, default=1,
                      help='Set 1 so each CSV row is one critic batch.')
  parser.add_argument('--repr_dim', type=int, default=16)
  parser.add_argument('--repr_norm', action='store_true', default=True)
  parser.add_argument('--target_tau', type=float, default=0.005,
                      help='Polyak τ for target net (paper default 0.005). '
                           'Use 0.995 to match your PPO TD-InfoNCE runs.')
  parser.add_argument('--hidden_layer_sizes', default='512,512,512,512')
  parser.add_argument('--start_index', type=int, default=0)
  parser.add_argument('--end_index', type=int, default=-1)
  parser.add_argument('--eval_interval', type=int, default=10,
                      help='Run eval every N learner steps (0=disable).')
  parser.add_argument('--num_eval_episodes', type=int, default=5)
  args = parser.parse_args()

  os.environ.setdefault('MUJOCO_GL', 'osmesa')
  log_dir = Path(args.log_dir)
  wlog_dir = log_dir / 'logs'
  wlog_dir.mkdir(parents=True, exist_ok=True)

  env, obs_dim = contrastive_utils.make_environment(
      args.env, args.start_index, args.end_index, args.seed)
  environment_spec = specs.make_environment_spec(env)
  assert (environment_spec.actions.minimum == -1).all()
  assert (environment_spec.actions.maximum == 1).all()

  hidden = tuple(int(x) for x in args.hidden_layer_sizes.split(',') if x.strip())
  config = TDInfoNCEConfig(
      env_name=args.env,
      batch_size=args.batch_size,
      num_sgd_steps_per_step=args.num_sgd_steps_per_step,
      min_replay_size=args.min_replay,
      repr_dim=args.repr_dim,
      repr_norm=args.repr_norm,
      twin_q=True,
      tau=args.target_tau,
      hidden_layer_sizes=hidden,
      obs_dim=obs_dim,
      max_episode_steps=getattr(env, '_step_limit', 150),
      start_index=args.start_index,
      end_index=args.end_index,
      local=True,
      jit=True,
  )

  run_cfg = vars(args)
  run_cfg['obs_dim'] = obs_dim
  run_cfg['hidden_layer_sizes'] = hidden
  with open(log_dir / 'run_config.json', 'w', encoding='utf-8') as fh:
    json.dump(run_cfg, fh, indent=2)

  print(f'[ref_td_infonce] env={args.env} obs_dim={obs_dim} '
        f'batch={args.batch_size} repr_dim={args.repr_dim} tau={args.target_tau}')
  print(f'[ref_td_infonce] collecting >= {args.min_replay} random transitions...')
  episodes = _collect_episodes(env, args.min_replay, obs_dim, args.seed)
  print(f'[ref_td_infonce] replay episodes={len(episodes)}')

  network_factory = functools.partial(
      make_networks,
      obs_dim=obs_dim,
      repr_dim=config.repr_dim,
      repr_norm=config.repr_norm,
      repr_norm_temp=config.repr_norm_temp,
      twin_q=config.twin_q,
      hidden_layer_sizes=config.hidden_layer_sizes,
  )
  networks = network_factory(environment_spec)

  import optax
  policy_optimizer = optax.adam(learning_rate=config.actor_learning_rate)
  q_optimizer = optax.adam(learning_rate=config.critic_learning_rate)
  iterator = ReplayBatchIterator(episodes, config)
  obs_to_goal = functools.partial(
      contrastive_utils.obs_to_goal_2d,
      start_index=config.start_index,
      end_index=config.end_index)

  learner = TDInfoNCELearner(
      networks=networks,
      rng=jax.random.PRNGKey(args.seed),
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
      iterator=iterator,
      counter=None,
      logger=None,
      obs_to_goal=obs_to_goal,
      config=config,
      wlog_dir=str(wlog_dir),
  )

  eval_logger = None
  if args.eval_interval > 0:
    from contrastive.resumable_csv_logger import ResumableCSVLogger
    eval_log_dir = wlog_dir / 'eval'
    eval_log_dir.mkdir(parents=True, exist_ok=True)
    eval_logger = ResumableCSVLogger(
        directory_or_file=str(eval_log_dir),
        label='',
        add_uid=False,
    )
    print(f'[ref_td_infonce] eval every {args.eval_interval} steps → '
          f'{eval_log_dir / "logs.csv"}')

  print(f'[ref_td_infonce] training for {args.learner_steps} learner steps...')
  t0 = time.time()
  for step in range(args.learner_steps):
    learner.step()
    if (args.eval_interval > 0 and eval_logger is not None
        and step % args.eval_interval == 0):
      _run_eval(
          learner=learner,
          networks=networks,
          env_name=args.env,
          seed=args.seed,
          start_index=args.start_index,
          end_index=args.end_index,
          learner_step=step,
          num_episodes=args.num_eval_episodes,
          eval_logger=eval_logger,
      )
    if step == 0 or (step + 1) % 50 == 0:
      print(f'  learner step {step + 1}/{args.learner_steps} '
            f'({time.time() - t0:.1f}s elapsed)')

  print(f'[ref_td_infonce] done. w_diag CSV: {wlog_dir / "w_diag" / "vectors.csv"}')
  print(f'[ref_td_infonce] logits_w CSV: {wlog_dir / "logits_w" / "vectors.csv"}')
  if eval_logger is not None:
    print(f'[ref_td_infonce] eval CSV: {wlog_dir / "eval" / "logs.csv"}')
  print('Live compare plot:')
  print('  python scripts/watch_w_diag_compare_user_vs_ref.py --watch')


if __name__ == '__main__':
  main()
