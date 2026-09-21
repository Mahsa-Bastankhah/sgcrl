#!/usr/bin/env python3
"""Evaluate NF positive-vs-negative ranking on BuilderBench checkpoints.

Metric matches ``scripts/eval_allegro_nf_rank_accuracy.py``:

  Two independent stochastic rollouts (num_envs=2).  For an anchor (s_t, a_t),
  the positive goal is a truncated-geometric future achieved goal from the
  same rollout (γ=0.99, EpisodeReplay semantics).  The negative goal is an
  achieved goal from a random timestep of the other rollout.  Report

    P[log p(g_pos | s_t, a_t) > log p(g_neg | s_t, a_t)]

  Positive pairs that would cross an environment reset are excluded.

Init / env settings are taken from each run's ``run_config.json`` (do not
force nopermute/fixedx unless that run used them). Successful NF BB recipes
are typically nopermute+fixedx01; the two envs still differ via independent
reset RNG and policy noise.

Examples::

  python -u scripts/eval_builderbench_nf_rank_accuracy.py \\
      --run-dir=logs/.../ppo_builderbench_creative_5_task2_0 \\
      --iterations=100,200,400 \\
      --output-csv=figs/builderbench/c5t2_nf_rank_accuracy.csv \\
      --output-plot=figs/builderbench/c5t2_nf_rank_accuracy.png
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

# Prefer GPU; must be set before importing modules that setdefault JAX_PLATFORMS.
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

import contrastive
from contrastive import nf_density as _nf
from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv


CKPT_RE = re.compile(r'ckpt_iter_(\d+)\.pkl$')


def _load_bb_helpers():
  path = os.path.join(REPO, 'scripts', 'builderbench_nf_goal_preimage_viz.py')
  spec = importlib.util.spec_from_file_location('bb_nf_preimage', path)
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  # Register before exec so @dataclass can resolve the module namespace.
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
  # Macro episode length * PD duration recovers MJ steps when PD is on.
  if ctx.use_pd:
    return int(ctx.episode_length) * int(ctx.pd_duration)
  return int(ctx.episode_length)


def _make_pairs(
    packed: np.ndarray,
    actions: np.ndarray,
    dones: np.ndarray,
    obs_dim: int,
    goal_state_indices: np.ndarray,
    discount: float,
    rng: np.random.Generator,
):
  """Create symmetric positive/negative comparisons from two rollouts."""
  if packed.ndim != 3 or packed.shape[1] != 2:
    raise ValueError(f'expected packed shape (T,2,D), got {packed.shape}')
  states = packed[:, :, :obs_dim]
  achieved = states[:, :, goal_state_indices]
  anchor_obs = []
  anchor_actions = []
  pos_goals = []
  neg_goals = []
  deltas = []

  for src, neg_src in ((0, 1), (1, 0)):
    for t in range(packed.shape[0] - 1):
      max_delta = 0
      for delta_i in range(1, packed.shape[0] - t):
        if np.any(dones[t:t + delta_i, src]):
          break
        max_delta = delta_i
      if max_delta < 1:
        continue
      if 0.0 < discount < 1.0:
        trunc_cdf = 1.0 - float(discount) ** max_delta
        u = float(rng.random()) * trunc_cdf
        delta = 1 + int(np.floor(
            np.log1p(-u) / np.log(float(discount))))
        delta = int(np.clip(delta, 1, max_delta))
      else:
        delta = int(rng.integers(1, max_delta + 1))
      neg_t = int(rng.integers(0, packed.shape[0]))
      anchor_obs.append(states[t, src])
      anchor_actions.append(actions[t, src])
      pos_goals.append(achieved[t + delta, src])
      neg_goals.append(achieved[neg_t, neg_src])
      deltas.append(delta)

  return tuple(
      np.asarray(x, dtype=np.float32)
      for x in (anchor_obs, anchor_actions, pos_goals, neg_goals, deltas)
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

  x = np.asarray([r['iteration'] for r in rows], dtype=float)
  acc = np.asarray([r['rank_accuracy'] for r in rows], dtype=float)
  se = np.asarray([r['rank_accuracy_se'] for r in rows], dtype=float)
  pos = np.asarray([r['positive_logp_mean'] for r in rows], dtype=float)
  neg = np.asarray([r['negative_logp_mean'] for r in rows], dtype=float)
  margin = np.asarray([r['margin_mean'] for r in rows], dtype=float)

  fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.2), sharex=True)
  axes[0].plot(x, acc, color='#4C9BE8', lw=2.0, marker='o',
               label=r'$P[\log p(g^+) > \log p(g^-)]$')
  axes[0].fill_between(x, acc - 1.96 * se, acc + 1.96 * se,
                       color='#4C9BE8', alpha=0.2, label='approx. 95% CI')
  axes[0].axhline(0.5, color='0.45', lw=1.0, ls='--', label='chance = 50%')
  axes[0].set_ylim(0.0, 1.0)
  axes[0].set_ylabel('rank accuracy')
  axes[0].legend(fontsize=8)

  axes[1].plot(x, pos, color='#2A9D8F', lw=1.8, marker='o',
               label=r'positive $\log p$')
  axes[1].plot(x, neg, color='#E8834C', lw=1.8, marker='o',
               label=r'negative $\log p$')
  axes[1].set_ylabel('mean raw log p')
  axes[1].legend(fontsize=8)

  axes[2].plot(x, margin, color='#A84CE8', lw=1.8, marker='o',
               label=r'mean $\log p(g^+) - \log p(g^-)$')
  axes[2].axhline(0.0, color='0.45', lw=1.0, ls='--')
  axes[2].set_ylabel('mean log-p margin')
  axes[2].set_xlabel('checkpoint iteration')
  axes[2].legend(fontsize=8)

  for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
  fig.suptitle(title, fontsize=11, fontweight='bold')
  fig.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
  tmp = path + '.tmp.png'
  fig.savefig(tmp, dpi=150, bbox_inches='tight')
  os.replace(tmp, path)
  plt.close(fig)


def _rollout_two_envs(env, act_fn, policy_params, key, n_steps: int):
  """Return packed obs (T,2,D), actions (T,2,A), dones (T,2), and new key."""
  packed_obs = env.reset()
  packed = []
  actions = []
  dones = []
  for _ in range(int(n_steps)):
    key, subkey = jax.random.split(key)
    action = act_fn(policy_params, jnp.asarray(packed_obs), subkey)
    action_np = np.asarray(action, dtype=np.float32)
    packed.append(np.asarray(packed_obs, dtype=np.float32).copy())
    actions.append(action_np.copy())
    packed_obs, _, done_t, _, _ = env.step(action_np)
    dones.append(np.asarray(done_t, dtype=bool).copy())
  return (
      np.asarray(packed, dtype=np.float32),
      np.asarray(actions, dtype=np.float32),
      np.asarray(dones, dtype=bool),
      key,
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--run-dir', required=True,
                      help='Run directory containing run_config.json + checkpoints/')
  parser.add_argument('--output-csv', required=True)
  parser.add_argument('--output-plot', required=True)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--min-samples', type=int, default=256)
  parser.add_argument(
      '--iterations', default='',
      help='comma-separated checkpoint iterations; empty evaluates all')
  parser.add_argument(
      '--future-discount', type=float, default=0.99,
      help='truncated-geometric future-goal discount (training default 0.99)')
  parser.add_argument(
      '--num-steps', type=int, default=0,
      help='0 → 3× macro episode length (≈Allegro 296 pairs for ep=50)')
  parser.add_argument('--env-name', default='',
                      help='override env name; default from run_config.json')
  args = parser.parse_args()

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
      f'[nf-rank] env={env_name} permute={ctx.permute_start_boxes} '
      f'fixed_start_x={ctx.fixed_start_x} use_pd={ctx.use_pd} '
      f'macro_ep={ctx.episode_length} obs_dim={ctx.obs_dim} '
      f'goal=[{ctx.start_index}:{ctx.end_index}] '
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
      num_envs=2,
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
        f'{goal_dim} (start={si}, end={ei}, obs_dim={obs_dim})')
  print(
      f'[nf-rank] achieved goal = state indices '
      f'{goal_state_indices.tolist()} (obs_dim={obs_dim}, goal_dim={goal_dim})',
      flush=True,
  )

  reward_fn = _nf.make_nf_reward_fn(
      nf_nets, obs_dim=obs_dim, tanh_scale=0.0, reward_mode='forward')

  @jax.jit
  def act_fn(policy_params, packed_obs, key):
    dist = networks.policy_network.apply(policy_params, packed_obs)
    return networks.sample(dist, key)

  n_steps = int(args.num_steps)
  if n_steps <= 0:
    # 3 macro episodes → ~2*(3*49)=294 pairs for ep=50, near Allegro's 296.
    n_steps = 3 * int(vec_env.episode_length)
  print(f'[nf-rank] num_steps={n_steps} (macro_ep={vec_env.episode_length})',
        flush=True)

  device = jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices()[0]
  results = []
  for path in paths:
    ckpt = _load_ckpt(path)
    iteration = int(ckpt.get('iteration', _iteration(path)))
    policy_params = jax.device_put(ckpt['policy_params'], device)
    nf_key = 'q_params_ema' if 'q_params_ema' in ckpt else 'q_params'
    nf_params = jax.device_put(ckpt[nf_key], device)
    goal_mean, goal_std = bb._load_goal_stats(run_dir, iteration, goal_dim)

    key = jax.random.PRNGKey(int(args.seed) + iteration)
    packed, actions, dones, _ = _rollout_two_envs(
        vec_env, act_fn, policy_params, key, n_steps)
    rng = np.random.default_rng(int(args.seed) + 1_000_003 + iteration)
    anchors, anchor_actions, pos_goals, neg_goals, deltas = _make_pairs(
        packed, actions, dones, obs_dim, goal_state_indices,
        float(args.future_discount), rng)
    if len(anchors) < int(args.min_samples):
      raise RuntimeError(
          f'iter={iteration}: only {len(anchors)} valid comparisons; '
          f'need at least {args.min_samples}')

    pos, neg = _score_pairs(
        reward_fn, nf_params, anchors, anchor_actions, pos_goals, neg_goals,
        goal_mean, goal_std)
    correct = pos > neg
    ties = pos == neg
    accuracy = float(np.mean(correct))
    se = float(np.sqrt(accuracy * (1.0 - accuracy) / len(correct)))
    row = {
        'iteration': iteration,
        'num_samples': len(correct),
        'rank_accuracy': accuracy,
        'rank_accuracy_se': se,
        'tie_fraction': float(np.mean(ties)),
        'positive_logp_mean': float(np.mean(pos)),
        'positive_logp_std': float(np.std(pos)),
        'negative_logp_mean': float(np.mean(neg)),
        'negative_logp_std': float(np.std(neg)),
        'margin_mean': float(np.mean(pos - neg)),
        'margin_median': float(np.median(pos - neg)),
        'delta1_fraction': float(np.mean(deltas == 1)),
        'future_delta_mean': float(np.mean(deltas)),
        'future_delta_median': float(np.median(deltas)),
        'nf_params_source': nf_key,
        'permute_start_boxes': bool(ctx.permute_start_boxes),
        'fixed_start_x': (
            '' if ctx.fixed_start_x is None else float(ctx.fixed_start_x)),
    }
    results.append(row)
    print(
        f'[nf-rank] iter={iteration:04d} n={len(correct)} '
        f'acc={accuracy:.3f} pos={np.mean(pos):+.3f} '
        f'neg={np.mean(neg):+.3f} margin={np.mean(pos - neg):+.3f}',
        flush=True,
    )

  _write_csv(args.output_csv, results)
  title = (
      f'BB NF ranking ({env_name}): same-rollout future vs other-rollout goal'
  )
  _plot(args.output_plot, results, title)
  meta_path = os.path.splitext(args.output_csv)[0] + '_meta.json'
  with open(meta_path, 'w', encoding='utf-8') as fh:
    json.dump({
        'definition': 'P[log p(g_pos|s,a) > log p(g_neg|s,a)]',
        'positive': 'same-rollout truncated-geometric future achieved goal',
        'negative': 'random achieved goal timestep from other rollout',
        'future_discount': float(args.future_discount),
        'goal_state_indices': goal_state_indices.tolist(),
        'symmetric_rollouts': True,
        'exclude_positive_reset_crossings': True,
        'seed': int(args.seed),
        'num_steps': int(n_steps),
        'run_dir': run_dir,
        'env_name': env_name,
        'permute_start_boxes': bool(ctx.permute_start_boxes),
        'fixed_start_x': ctx.fixed_start_x,
        'init_note': (
            'Two envs use independent reset RNG + policy noise. '
            'If permute=false and fixed_start_x is set, box xy init diversity '
            'is limited to residual RNG (match training recipe).'
        ),
        'nf_arch': {
            'rep_size': ctx.nf_rep_size,
            'num_blocks': ctx.nf_num_blocks,
            'channels': ctx.nf_coupling_width,
            'sa_hidden': ctx.nf_sa_hidden,
            'sa_num_layers': ctx.nf_sa_num_layers,
            'state_only': bool(ctx.nf_state_only),
        },
        'mjx_impl': os.environ.get('BUILDERBENCH_MJX_IMPL', 'jax'),
        'flags_repr_mode': flags.get('ppo_repr_mode'),
        'resolved_obs_dim': resolved.get('obs_dim'),
    }, fh, indent=2)
  print(f'[nf-rank] wrote {args.output_csv}', flush=True)
  print(f'[nf-rank] wrote {args.output_plot}', flush=True)
  print(f'[nf-rank] wrote {meta_path}', flush=True)


if __name__ == '__main__':
  main()
