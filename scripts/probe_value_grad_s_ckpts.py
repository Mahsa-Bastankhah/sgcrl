#!/usr/bin/env python3
"""Roll out each PPO ckpt and measure ||∇_s V(s)|| on those states.

V is the PPO value net (same packed obs as the policy). ∇_s is w.r.t. the
state slice of packed obs (goal slice is held fixed).

  python scripts/probe_value_grad_s_ckpts.py \\
      --run_dir=logs/final_runs/.../ppo_builderbench_creative_7_task2_1 \\
      --env=builderbench_creative_7_task2 \\
      --stride=200 --num_eval_episodes=5

5 stochastic (policy.sample) rollouts per checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import jax
import jax.numpy as jnp
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
sys.path.insert(0, _SCRIPTS)

from contrastive import ppo_learner  # noqa: E402


def _load_mod(name: str, filename: str):
  path = os.path.join(_SCRIPTS, filename)
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


ev = _load_mod('ppo_builderbench_checkpoint_eval', 'ppo_builderbench_checkpoint_eval.py')
base = _load_mod('plot_builderbench_train_success1000', 'plot_builderbench_train_success1000.py')

CSV_FIELDS = [
    'iteration', 'global_step', 'success_mean',
    'v_mean', 'v_std',
    'grad_s_mean', 'grad_s_p50', 'grad_s_p90', 'grad_s_max',
    'n_states',
]


def _make_grad_fn(value_apply, obs_dim: int):
  def _v_of_s(s, g, params):
    packed = jnp.concatenate([s, g], axis=-1)
    return value_apply(params, packed[None])[0]

  grad_s = jax.jit(jax.vmap(jax.grad(_v_of_s, argnums=0), in_axes=(0, 0, None)))
  val_fn = jax.jit(jax.vmap(_v_of_s, in_axes=(0, 0, None)))
  return grad_s, val_fn, int(obs_dim)


def _summarize(x: np.ndarray) -> tuple[float, float, float, float]:
  x = np.asarray(x, dtype=np.float64).reshape(-1)
  x = x[np.isfinite(x)]
  if x.size == 0:
    return (float('nan'),) * 4
  return (
      float(np.mean(x)),
      float(np.median(x)),
      float(np.quantile(x, 0.90)),
      float(np.max(x)),
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument('--run_dir', required=True)
  p.add_argument('--env', required=True)
  p.add_argument('--num_eval_episodes', type=int, default=5)
  p.add_argument('--eval_seed', type=int, default=0)
  p.add_argument('--stride', type=int, default=200,
                 help='Keep ckpts whose iteration %% stride == 0 (plus first/last).')
  p.add_argument('--csv_output', default='')
  p.add_argument('--output', default='')
  args = p.parse_args()

  run_dir, ckpt_dir = ev.resolve_paths(args.run_dir, None)
  run_cfg = ev.read_run_config(run_dir)
  files = ev.list_checkpoint_files(ckpt_dir, include_latest=False)
  if not files:
    raise SystemExit(f'no checkpoints in {ckpt_dir}')
  ctx = ev._load_train_ctx(args.env, files[0][1])
  stride = max(int(args.stride), 1)
  picked = []
  for i, item in enumerate(files):
    it = ev._iteration_from_label(item[0])
    if i == 0 or i == len(files) - 1 or it % stride == 0:
      picked.append(item)
  print(f'[gradV] {len(picked)}/{len(files)} ckpts stride={stride} '
        f'E={args.num_eval_episodes} stochastic obs_dim={ctx.obs_dim}',
        flush=True)

  session = ev.CheckpointEvalSession(
      args.env, ctx, run_cfg,
      num_eval_episodes=int(args.num_eval_episodes),
      eval_seed=int(args.eval_seed),
  )
  policy_network = session.networks.policy_network
  sample_fn = session.networks.sample

  def _stoch_action(policy_p, obs, key):
    return sample_fn(policy_network.apply(policy_p, obs), key)

  stoch_unroll = session.vec_env.compile_mpo_unroll(
      _stoch_action, unroll_length=session.vec_env.episode_length)
  value_apply = session.networks.value_network.apply
  grad_fn, val_fn, obs_dim = _make_grad_fn(value_apply, int(ctx.obs_dim))

  rows = []
  for label, path, _mtime in picked:
    ckpt = ppo_learner.load_checkpoint(path)
    iteration = int(ckpt.get('iteration', ev._iteration_from_label(label)))
    global_step = int(ckpt.get('global_step', iteration))
    print(f'[gradV] stochastic unroll {label} …', flush=True)
    eval_state = session.vec_env.reset_state()
    key = jax.random.PRNGKey(int(args.eval_seed) + int(iteration) * 1009)
    _, steps = stoch_unroll(eval_state, ckpt['policy_params'], key)
    packed = np.asarray(steps['obs'], dtype=np.float32)
    flat = packed.reshape(-1, packed.shape[-1])
    s = jnp.asarray(flat[:, :obs_dim])
    g = jnp.asarray(flat[:, obs_dim:])
    v = np.asarray(val_fn(s, g, ckpt['value_params']), dtype=np.float32)
    gs = np.asarray(grad_fn(s, g, ckpt['value_params']), dtype=np.float32)
    gnorm = np.linalg.norm(gs.reshape(gs.shape[0], -1), axis=-1)
    succ = ev.episode_successes_from_steps(steps)
    g_mean, g_p50, g_p90, g_max = _summarize(gnorm)
    rows.append({
        'iteration': iteration,
        'global_step': global_step,
        'success_mean': float(succ.mean()),
        'v_mean': float(np.mean(v)),
        'v_std': float(np.std(v)),
        'grad_s_mean': g_mean,
        'grad_s_p50': g_p50,
        'grad_s_p90': g_p90,
        'grad_s_max': g_max,
        'n_states': int(gnorm.size),
    })
    print(
        f'  it={iteration} succ={succ.mean():.2f} '
        f'||∇_s V|| mean={g_mean:.4g} p90={g_p90:.4g} max={g_max:.4g} '
        f'Vmean={float(np.mean(v)):.4g}',
        flush=True)

  tag = os.path.basename(run_dir.rstrip('/'))
  csv_path = args.csv_output or os.path.join(
      _REPO, 'figs', 'builderbench', 'final_runs',
      f'{tag}_value_grad_s.csv')
  png_path = args.output or os.path.join(
      _REPO, 'figs', 'builderbench', 'final_runs',
      f'{tag}_value_grad_s.png')
  os.makedirs(os.path.dirname(csv_path), exist_ok=True)
  with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    w.writeheader()
    w.writerows(rows)
  print(f'→ {csv_path}', flush=True)

  xs = [r['global_step'] for r in rows]
  fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.2), sharex=True)
  fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.08, hspace=0.18)
  axes[0].plot(
      xs, [r['success_mean'] for r in rows],
      color='#C44E52', lw=2.0, marker='o', ms=3.5)
  axes[0].set_ylabel('Eval success (this probe)')
  axes[0].set_ylim(-0.05, 1.05)
  axes[1].plot(
      xs, [r['grad_s_mean'] for r in rows],
      color='#4C72B0', lw=2.0, label='mean')
  axes[1].plot(
      xs, [r['grad_s_p90'] for r in rows],
      color='#4C72B0', lw=1.3, ls='--', alpha=0.85, label='p90')
  axes[1].plot(
      xs, [r['grad_s_max'] for r in rows],
      color='#4C72B0', lw=1.0, ls=':', alpha=0.7, label='max')
  axes[1].legend(frameon=False, fontsize=8)
  axes[1].set_ylabel(r'$\|\nabla_s V(s)\|$')
  axes[2].plot(
      xs, [r['v_mean'] for r in rows],
      color='#55A868', lw=2.0)
  axes[2].set_ylabel('V(s) mean')
  axes[2].set_xlabel('Env steps')
  axes[2].xaxis.set_major_formatter(mticker.FuncFormatter(base._fmt_steps))
  for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
  axes[0].set_title(
      f'{tag}  ·  ‖∇_s V‖ on ckpt rollouts',
      fontsize=11, fontweight='bold')
  fig.savefig(png_path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'→ {png_path}', flush=True)


if __name__ == '__main__':
  main()
