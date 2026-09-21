#!/usr/bin/env python3
"""NF future-goal binary rank accuracy before vs after first success.

Stochastic policy rollouts.  For an anchor (s_t, a_t) the positive goal is a
truncated-geometric future achieved goal from the same episode (γ=0.99).  The
negative is an achieved goal from a random timestep of another env.  Report

    P[log p(g_pos | s_t, a_t) > log p(g_neg | s_t, a_t)]

split by whether the anchor is before vs at/after first success.  Pre/post
splits use only episodes that succeed at least once.

Example::

  python -u scripts/eval_builderbench_nf_binary_acc_prepost.py \\
      --run-dir=logs/.../ppo_builderbench_creative_5_task2_0 \\
      --iterations=300,400,1000,1400 \\
      --output-csv=figs/builderbench/out.csv \\
      --output-plot=figs/builderbench/out.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import pickle
import re
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('MUJOCO_GL', 'egl')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from contrastive import nf_density as _nf
from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv


CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')


def _load_bb_helpers():
  path = os.path.join(REPO, 'scripts', 'builderbench_nf_goal_preimage_viz.py')
  spec = importlib.util.spec_from_file_location('bb_nf_preimage', path)
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


def _iteration(path: str) -> int:
  match = CKPT_RE.search(os.path.basename(path))
  if match is None:
    raise ValueError(f'not a checkpoint filename: {path}')
  return int(match.group(1))


def _load_ckpt(path: str):
  with open(path, 'rb') as fh:
    return pickle.load(fh)


def _env_name_from_run(run_dir: str) -> str:
  cfg_path = os.path.join(run_dir, 'run_config.json')
  with open(cfg_path, 'r', encoding='utf-8') as fh:
    cfg = json.load(fh)
  env = cfg.get('env') or cfg.get('flags', {}).get('env')
  if not env:
    env = cfg.get('resolved_config', {}).get('env_name')
  if not env:
    raise KeyError(f'no env name in {cfg_path}')
  return str(env)


def _mj_episode_length(run_cfg: dict, ctx) -> int:
  flags = run_cfg.get('flags', {})
  resolved = run_cfg.get('resolved_config', {})
  mj = flags.get(
      'builderbench_mj_episode_length',
      resolved.get('builderbench_mj_episode_length'))
  if mj not in (None, '', False):
    return int(mj)
  if ctx.use_pd:
    return int(ctx.episode_length) * int(ctx.pd_duration)
  return int(ctx.episode_length)


def _sample_delta(max_delta: int, discount: float, rng: np.random.Generator) -> int:
  if max_delta < 1:
    raise ValueError('max_delta must be >= 1')
  if 0.0 < discount < 1.0:
    trunc_cdf = 1.0 - float(discount) ** max_delta
    u = float(rng.random()) * trunc_cdf
    delta = 1 + int(np.floor(np.log1p(-u) / np.log(float(discount))))
    return int(np.clip(delta, 1, max_delta))
  return int(rng.integers(1, max_delta + 1))


def _first_success(success: np.ndarray) -> np.ndarray:
  """Per-env first t with success>=0.5, or -1 if none. success is (T, E)."""
  hit = np.asarray(success, dtype=np.float32) >= 0.5
  has = hit.any(axis=0)
  first = np.argmax(hit, axis=0).astype(np.int32)
  return np.where(has, first, np.int32(-1))


def _make_pairs(
    packed: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    obs_dim: int,
    goal_state_indices: np.ndarray,
    discount: float,
    rng: np.random.Generator,
):
  """Pairs from N stochastic rollouts. Returns arrays + env index and t."""
  if packed.ndim != 3 or packed.shape[1] < 2:
    raise ValueError(f'expected packed shape (T,E>=2,D), got {packed.shape}')
  n_t, n_e = packed.shape[:2]
  states = packed[:, :, :obs_dim]
  achieved = states[:, :, goal_state_indices]
  other = [j for j in range(n_e)]

  anchor_obs, anchor_actions = [], []
  pos_goals, neg_goals, deltas = [], [], []
  env_idx, times = [], []

  for src in range(n_e):
    others = [j for j in other if j != src]
    for t in range(n_t - 1):
      max_delta = 0
      for delta_i in range(1, n_t - t):
        if np.any(dones[t:t + delta_i, src]):
          break
        max_delta = delta_i
      if max_delta < 1:
        continue
      delta = _sample_delta(max_delta, discount, rng)
      neg_src = int(others[int(rng.integers(0, len(others)))])
      neg_t = int(rng.integers(0, n_t))
      anchor_obs.append(states[t, src])
      anchor_actions.append(actions[t, src])
      pos_goals.append(achieved[t + delta, src])
      neg_goals.append(achieved[neg_t, neg_src])
      deltas.append(delta)
      env_idx.append(src)
      times.append(t)

  return (
      np.asarray(anchor_obs, dtype=np.float32),
      np.asarray(anchor_actions, dtype=np.float32),
      np.asarray(pos_goals, dtype=np.float32),
      np.asarray(neg_goals, dtype=np.float32),
      np.asarray(deltas, dtype=np.int32),
      np.asarray(env_idx, dtype=np.int32),
      np.asarray(times, dtype=np.int32),
  )


def _score_pairs(
    reward_fn,
    nf_params,
    anchors: np.ndarray,
    actions: np.ndarray,
    pos_goals: np.ndarray,
    neg_goals: np.ndarray,
    goal_mean: np.ndarray,
    goal_std: np.ndarray,
):
  pos_obs = np.concatenate([anchors, pos_goals], axis=-1)
  neg_obs = np.concatenate([anchors, neg_goals], axis=-1)
  pos = np.asarray(
      reward_fn(
          nf_params, jnp.asarray(pos_obs), jnp.asarray(actions),
          jnp.asarray(goal_mean), jnp.asarray(goal_std)),
      dtype=np.float32,
  )
  neg = np.asarray(
      reward_fn(
          nf_params, jnp.asarray(neg_obs), jnp.asarray(actions),
          jnp.asarray(goal_mean), jnp.asarray(goal_std)),
      dtype=np.float32,
  )
  return pos, neg


def _split_metrics(pos, neg, deltas, mask: np.ndarray) -> dict:
  n = int(np.sum(mask))
  if n == 0:
    nan = float('nan')
    return {
        'num_samples': 0,
        'rank_accuracy': nan,
        'rank_accuracy_se': nan,
        'tie_fraction': nan,
        'positive_logp_mean': nan,
        'negative_logp_mean': nan,
        'margin_mean': nan,
        'future_delta_mean': nan,
    }
  p = pos[mask]
  q = neg[mask]
  correct = p > q
  acc = float(np.mean(correct))
  se = float(np.sqrt(acc * (1.0 - acc) / n))
  return {
      'num_samples': n,
      'rank_accuracy': acc,
      'rank_accuracy_se': se,
      'tie_fraction': float(np.mean(p == q)),
      'positive_logp_mean': float(np.mean(p)),
      'negative_logp_mean': float(np.mean(q)),
      'margin_mean': float(np.mean(p - q)),
      'future_delta_mean': float(np.mean(deltas[mask])),
  }


def _write_csv(path: str, rows: list[dict]) -> None:
  os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
  tmp = path + '.tmp'
  with open(tmp, 'w', newline='', encoding='utf-8') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  os.replace(tmp, path)


def _plot(path: str, rows: list[dict], title: str) -> None:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt

  iters = sorted({int(r['iteration']) for r in rows})
  by = {(int(r['iteration']), r['split']): r for r in rows}

  fig, ax = plt.subplots(figsize=(8.8, 4.6))
  splits = (
      ('overall', '#6C757D', 'overall'),
      ('before', '#4C9BE8', 'before success'),
      ('after', '#E07A5F', 'after success'),
  )
  present = [s for s, _, _ in splits if any(
      int(by[k]['num_samples']) > 0 for k in by if k[1] == s)]
  width = 0.24 if len(present) > 2 else 0.36
  xs = np.arange(len(iters), dtype=float)
  offsets = np.linspace(-(len(present) - 1) / 2.0, (len(present) - 1) / 2.0,
                        num=len(present)) * width
  colors = {s: c for s, c, _ in splits}
  labels = {s: lab for s, _, lab in splits}
  for offset, split in zip(offsets, present):
    acc, se, nn = [], [], []
    for it in iters:
      row = by.get((it, split))
      if row is None or int(row['num_samples']) == 0:
        acc.append(np.nan)
        se.append(0.0)
        nn.append(0)
      else:
        acc.append(float(row['rank_accuracy']))
        se.append(float(row['rank_accuracy_se']))
        nn.append(int(row['num_samples']))
    acc = np.asarray(acc, dtype=float)
    se = np.asarray(se, dtype=float)
    ax.bar(
        xs + offset, acc, width=width, color=colors[split],
        yerr=1.96 * se, capsize=3, label=f'{labels[split]} (n≈{max(nn)})',
        zorder=3)
  ax.axhline(0.5, color='0.45', lw=1.0, ls='--', label='chance = 50%')
  ax.set_xticks(xs)
  ax.set_xticklabels([str(it) for it in iters])
  ax.set_ylim(0.0, 1.0)
  ax.set_xlabel('checkpoint iteration')
  ax.set_ylabel(r'$P[\log p(g^+) > \log p(g^-)]$')
  ax.set_title(title, fontsize=11, fontweight='bold')
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.35)
  ax.legend(fontsize=8, loc='lower right', framealpha=0.95)
  os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
  tmp = path + '.tmp.png'
  fig.tight_layout()
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, path)
  plt.close(fig)


def _rollout(env, act_fn, policy_params, key, n_steps: int):
  packed_obs = env.reset()
  packed, actions, dones, success = [], [], [], []
  for _ in range(int(n_steps)):
    key, subkey = jax.random.split(key)
    action = act_fn(policy_params, jnp.asarray(packed_obs), subkey)
    action_np = np.asarray(action, dtype=np.float32)
    packed.append(np.asarray(packed_obs, dtype=np.float32).copy())
    actions.append(action_np.copy())
    packed_obs, _, done_t, _, _ = env.step(action_np)
    dones.append(np.asarray(done_t, dtype=bool).copy())
    success.append(np.asarray(env.last_success, dtype=np.float32).copy())
  return (
      np.asarray(packed, dtype=np.float32),
      np.asarray(actions, dtype=np.float32),
      np.asarray(dones, dtype=bool),
      np.asarray(success, dtype=np.float32),
      key,
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--run-dir', required=True)
  parser.add_argument('--output-csv', required=True)
  parser.add_argument('--output-plot', required=True)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num-envs', type=int, default=32)
  parser.add_argument('--min-samples', type=int, default=32)
  parser.add_argument('--iterations', default='')
  parser.add_argument('--future-discount', type=float, default=0.99)
  parser.add_argument(
      '--num-steps', type=int, default=0,
      help='0 → one macro episode (clean pre/post split)')
  parser.add_argument('--env-name', default='')
  parser.add_argument(
      '--policy', choices=('checkpoint', 'random'), default='checkpoint',
      help='checkpoint: stochastic actor from the ckpt. '
           'random: uniform actions in [-1, 1] (NF still from the ckpt).')
  args = parser.parse_args()
  if int(args.num_envs) < 2:
    raise ValueError('--num-envs must be >= 2')

  run_dir = os.path.abspath(args.run_dir)
  ckpt_dir = os.path.join(run_dir, 'checkpoints')
  paths = sorted(
      glob.glob(os.path.join(ckpt_dir, 'ckpt_iter_*.pkl')),
      key=_iteration,
  )
  if args.iterations.strip():
    requested = {
        int(value.strip()) for value in args.iterations.split(',')
        if value.strip()
    }
    paths = [path for path in paths if _iteration(path) in requested]
    found = {_iteration(path) for path in paths}
    if found != requested:
      raise FileNotFoundError(
          f'missing requested checkpoint iterations: {sorted(requested - found)}')
  if not paths:
    raise FileNotFoundError(f'no ckpt_iter_*.pkl in {ckpt_dir}')

  bb = _load_bb_helpers()
  env_name = args.env_name.strip() or _env_name_from_run(run_dir)
  ctx, run_cfg = bb._load_train_ctx(env_name, run_dir)
  flags = run_cfg.get('flags', {})
  resolved = run_cfg.get('resolved_config', {})

  print(
      f'[nf-prepost] env={env_name} permute={ctx.permute_start_boxes} '
      f'fixed_start_x={ctx.fixed_start_x} use_pd={ctx.use_pd} '
      f'macro_ep={ctx.episode_length} num_envs={int(args.num_envs)} '
      f'policy={args.policy} '
      f'impl={os.environ.get("BUILDERBENCH_MJX_IMPL", "jax")}',
      flush=True,
  )

  networks, nf_nets, act_dim, _ = bb._build_networks(
      env_name, seed=int(args.seed), ctx=ctx, repr_mode='nf')
  del act_dim
  if nf_nets is None:
    raise RuntimeError('NF networks were not built (repr_mode=nf expected)')

  mj_ep = _mj_episode_length(run_cfg, ctx)
  vec_env = JaxBuilderBenchVecEnv(
      env_name=env_name,
      num_envs=int(args.num_envs),
      seed=int(args.seed),
      use_pd=bool(ctx.use_pd),
      pd_duration=int(ctx.pd_duration),
      pd_filter_policy_obs=bool(ctx.filter_policy_obs),
      fixed_target_goal=ctx.fixed_target_goal,
      permute_start_boxes=bool(ctx.permute_start_boxes),
      mj_episode_length=int(mj_ep),
      fixed_start_x=ctx.fixed_start_x,
  )
  obs_dim = int(ctx.obs_dim)
  goal_dim = int(ctx.goal_dim)
  si, ei = int(ctx.start_index), int(ctx.end_index)
  if ei < 0:
    goal_state_indices = np.arange(si, obs_dim, dtype=np.int32)
  else:
    goal_state_indices = np.arange(si, ei, dtype=np.int32)
  if len(goal_state_indices) != goal_dim:
    raise ValueError(
        f'achieved-goal index count {len(goal_state_indices)} != goal_dim '
        f'{goal_dim}')

  reward_fn = _nf.make_nf_reward_fn(
      nf_nets, obs_dim=obs_dim, tanh_scale=0.0, reward_mode='forward')

  act_dim = int(vec_env.action_shape[0])
  if args.policy == 'random':
    @jax.jit
    def act_fn(policy_params, packed_obs, key):
      del policy_params
      n = packed_obs.shape[0]
      return jax.random.uniform(
          key, (n, act_dim), minval=-1.0, maxval=1.0)
  else:
    @jax.jit
    def act_fn(policy_params, packed_obs, key):
      dist = networks.policy_network.apply(policy_params, packed_obs)
      return networks.sample(dist, key)

  n_steps = int(args.num_steps)
  if n_steps <= 0:
    n_steps = int(vec_env.episode_length)
  print(f'[nf-prepost] num_steps={n_steps} (macro_ep={vec_env.episode_length})',
        flush=True)

  device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices()[0]
  results = []
  for path in paths:
    ckpt = _load_ckpt(path)
    iteration = int(ckpt.get('iteration', _iteration(path)))
    policy_params = (
        None if args.policy == 'random'
        else jax.device_put(ckpt['policy_params'], device))
    nf_key = 'q_params_ema' if 'q_params_ema' in ckpt else 'q_params'
    nf_params = jax.device_put(ckpt[nf_key], device)
    goal_mean, goal_std = bb._load_goal_stats(run_dir, iteration, goal_dim)

    key = jax.random.PRNGKey(int(args.seed) + iteration)
    packed, actions, dones, success, _ = _rollout(
        vec_env, act_fn, policy_params, key, n_steps)
    first = _first_success(success)
    n_succ_eps = int(np.sum(first >= 0))
    print(
        f'[nf-prepost] iter={iteration:04d} success_envs={n_succ_eps}/'
        f'{packed.shape[1]} first_succ_med='
        f'{np.median(first[first >= 0]) if n_succ_eps else float("nan"):.1f}',
        flush=True,
    )

    rng = np.random.default_rng(int(args.seed) + 1_000_003 + iteration)
    (anchors, anchor_actions, pos_goals, neg_goals, deltas,
     env_idx, times) = _make_pairs(
        packed, actions, dones, obs_dim, goal_state_indices,
        float(args.future_discount), rng)
    if len(anchors) < int(args.min_samples):
      raise RuntimeError(
          f'iter={iteration}: only {len(anchors)} valid comparisons; '
          f'need at least {args.min_samples}')

    pos, neg = _score_pairs(
        reward_fn, nf_params, anchors, anchor_actions, pos_goals, neg_goals,
        goal_mean, goal_std)
    ep_first = first[env_idx]
    succeeded = ep_first >= 0
    before = succeeded & (times < ep_first)
    after = succeeded & (times >= ep_first)

    for split, mask in (
        ('overall', np.ones(len(times), dtype=bool)),
        ('before', before),
        ('after', after),
    ):
      metrics = _split_metrics(pos, neg, deltas, mask)
      row = {
          'iteration': iteration,
          'split': split,
          'n_success_envs': n_succ_eps,
          'n_envs': int(packed.shape[1]),
          **metrics,
          'nf_params_source': nf_key,
          'policy': args.policy,
      }
      results.append(row)
      print(
          f'[nf-prepost] iter={iteration:04d} split={split:7s} '
          f'n={metrics["num_samples"]} acc={metrics["rank_accuracy"]:.3f} '
          f'pos={metrics["positive_logp_mean"]:+.3f} '
          f'neg={metrics["negative_logp_mean"]:+.3f} '
          f'margin={metrics["margin_mean"]:+.3f}',
          flush=True,
      )

  _write_csv(args.output_csv, results)
  policy_tag = (
      'random uniform[-1,1] actions' if args.policy == 'random'
      else 'stochastic checkpoint policy')
  _plot(
      args.output_plot, results,
      f'BB NF future-goal rank acc ({policy_tag}; {env_name})')
  meta_path = os.path.splitext(args.output_csv)[0] + '_meta.json'
  with open(meta_path, 'w', encoding='utf-8') as fh:
    json.dump({
        'definition': 'P[log p(g_pos|s,a) > log p(g_neg|s,a)]',
        'positive': 'same-rollout truncated-geometric future achieved goal',
        'negative': 'random achieved goal timestep from another env',
        'split': (
            'before/after use only episodes that succeed; '
            'before = anchor t < first success, after = t >= first success'
        ),
        'policy': (
            'random uniform[-1,1]' if args.policy == 'random'
            else 'stochastic (networks.sample)'),
        'future_discount': float(args.future_discount),
        'goal_state_indices': goal_state_indices.tolist(),
        'seed': int(args.seed),
        'num_envs': int(args.num_envs),
        'num_steps': int(n_steps),
        'run_dir': run_dir,
        'env_name': env_name,
        'permute_start_boxes': bool(ctx.permute_start_boxes),
        'fixed_start_x': ctx.fixed_start_x,
        'mjx_impl': os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax'),
        'flags_repr_mode': flags.get('ppo_repr_mode'),
        'resolved_obs_dim': resolved.get('obs_dim'),
    }, fh, indent=2)
  print(f'[nf-prepost] wrote {args.output_csv}', flush=True)
  print(f'[nf-prepost] wrote {args.output_plot}', flush=True)
  print(f'[nf-prepost] wrote {meta_path}', flush=True)


if __name__ == '__main__':
  main()
