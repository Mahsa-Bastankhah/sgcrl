"""Roll out a successful creative-3-task1 traj and report frozen φ·ψ rewards.

Uses the successful joint PPO+CRL policy + q_params_ema that the frozen-reward
job loads as its stationary reward.
"""
import os
import sys

# Prefer GPU if present; fall back to CPU.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BB = os.environ.get('BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BB not in sys.path:
  sys.path.insert(0, _BB)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from contrastive import ppo_learner
from contrastive import ContrastiveConfig
from envs.builderbench_utils import (
    filter_pd_policy_state_obs,
    parse_bb_env_id,
    sgcrl_env_name_to_bb_env_id,
)
import importlib.util as _ilu
_bbv_path = os.path.join(_REPO, 'scripts', 'ppo_builderbench_rollout_video.py')
_bbv_spec = _ilu.spec_from_file_location('ppo_builderbench_rollout_video', _bbv_path)
_bbv = _ilu.module_from_spec(_bbv_spec)
assert _bbv_spec.loader is not None
_bbv_spec.loader.exec_module(_bbv)
_build_networks = _bbv._build_networks
_load_train_ctx = _bbv._load_train_ctx
_make_bb_env = _bbv._make_bb_env
_make_policy_fn = _bbv._make_policy_fn
_maybe_fix_target = _bbv._maybe_fix_target


ENV = 'builderbench_creative_3_task1'
CKPT = (
    'logs/ppo_builderbench_creative3_task1_e1024_pd_crl_tau05_catselect_extrew1/'
    'ppo_builderbench_creative_3_task1_0/checkpoints/latest.pkl')
OUT_DIR = 'figs/builderbench/frozen_crl_reward_probe'
MAX_TRIES = 40
SEED = 0


def _compile_rollout(policy, env, ep_len, fixed_goal, mocap_targets, num_cubes,
                     filter_policy_obs, obs_dim):
  """One XLA-compiled episode: returns packed_obs, actions, success, dist."""

  @jax.jit
  def _run(key):
    env_key, key = jax.random.split(key)
    state = env.reset(jax.random.split(env_key, 1))
    state = _maybe_fix_target(state, fixed_goal, mocap_targets, num_cubes)

    def step(carry, _):
      state, key = carry
      key, act_key = jax.random.split(key)
      goals = state.info['target_goal']
      action, _ = policy(state.obs, goals, act_key)
      if filter_policy_obs:
        policy_obs = filter_pd_policy_state_obs(state.obs, num_cubes)
      else:
        policy_obs = state.obs
      packed = jnp.concatenate([policy_obs, goals], axis=-1)
      next_state = env.step(state, action)
      next_state = _maybe_fix_target(
          next_state, fixed_goal, mocap_targets, num_cubes)
      succ = jnp.asarray(next_state.metrics['success']).reshape(-1)[0]
      # Optional distance metric (not all builds expose the same key).
      metrics = next_state.metrics
      if 'dist' in metrics:
        dist = jnp.asarray(metrics['dist']).reshape(-1)[0]
      else:
        dist = jnp.asarray(jnp.nan, dtype=jnp.float32)
      out = {
          'packed': packed[0],
          'action': action[0],
          'success': succ,
          'dist': dist,
      }
      return (next_state, key), out

    _, traj = jax.lax.scan(step, (state, key), (), length=ep_len)
    return traj

  return _run


def main():
  os.makedirs(OUT_DIR, exist_ok=True)
  print(f'[probe] jax backend={jax.default_backend()} devices={jax.devices()}')
  env_id = sgcrl_env_name_to_bb_env_id(ENV)
  num_cubes, _ = parse_bb_env_id(env_id)

  print(f'[probe] ckpt={CKPT}')
  ctx = _load_train_ctx(ENV, CKPT)
  print(f'[probe] use_pd={ctx.use_pd} filter={ctx.filter_policy_obs} '
        f'obs_dim={ctx.obs_dim} ep_len={ctx.episode_length} '
        f'cat_classes={ctx.categorical_select_classes}')

  networks = _build_networks(ENV, seed=SEED, ctx=ctx)
  env, _base, mocap_targets, ep_len = _make_bb_env(env_id, ctx)

  ckpt = ppo_learner.load_checkpoint(CKPT)
  policy_params = ckpt['policy_params']
  q_params = ckpt.get('q_params_ema')
  q_src = 'q_params_ema'
  if q_params is None:
    q_params = ckpt['q_params']
    q_src = 'q_params'
  print(f'[probe] iteration={ckpt.get("iteration")} reward_src={q_src}')

  policy = _make_policy_fn(
      networks, policy_params, stochastic=False,
      filter_policy_obs=ctx.filter_policy_obs,
      num_cubes=num_cubes,
      normalize_obs=False,
      obs_dim=ctx.obs_dim,
      start_index=ctx.start_index,
      end_index=ctx.end_index,
  )

  cfg = ContrastiveConfig()
  cfg.obs_dim = int(ctx.obs_dim)
  cfg.start_index = int(ctx.start_index)
  cfg.end_index = int(ctx.end_index)
  cfg.ppo_norm_obs = False
  reward_fn = ppo_learner.make_reward_fn(networks, cfg)

  print('[probe] compiling rollout...', flush=True)
  run = _compile_rollout(
      policy, env, ep_len, ctx.fixed_target_goal, mocap_targets, num_cubes,
      ctx.filter_policy_obs, ctx.obs_dim)

  key = jax.random.PRNGKey(SEED)
  # Warm compile
  key, warm_key = jax.random.split(key)
  _ = jax.block_until_ready(run(warm_key))
  print('[probe] compile done', flush=True)

  mean0 = jnp.zeros((ctx.obs_dim,), dtype=jnp.float32)
  var1 = jnp.ones((ctx.obs_dim,), dtype=jnp.float32)

  best = None
  for attempt in range(MAX_TRIES):
    key, roll_key = jax.random.split(key)
    traj = run(roll_key)
    packed = np.asarray(traj['packed'], dtype=np.float32)
    actions = np.asarray(traj['action'], dtype=np.float32)
    succ = np.asarray(traj['success'], dtype=np.float32)
    dist = np.asarray(traj['dist'], dtype=np.float32)
    rewards = np.asarray(
        reward_fn(q_params, jnp.asarray(packed), jnp.asarray(actions),
                  mean0, var1),
        dtype=np.float32)
    reached = bool(np.any(succ >= 0.5))
    print(f'[probe] try={attempt} success={reached} '
          f'rew_sum={rewards.sum():.3f} '
          f'rew_min={rewards.min():.3f} '
          f'rew_max={rewards.max():.3f} '
          f'final_succ={succ[-1]:.0f}', flush=True)
    cur = {
        'rewards': rewards, 'success': succ, 'dist': dist,
        'actions': actions, 'obs': packed, 'attempt': attempt,
    }
    if reached:
      best = cur
      break
    if best is None or rewards.sum() > best['rewards'].sum():
      best = cur

  r = best['rewards']
  succ = best['success']
  first_succ = int(np.argmax(succ >= 0.5)) if np.any(succ >= 0.5) else -1

  print('\n=== successful trajectory reward report ===' if first_succ >= 0
        else '\n=== best (non-success) trajectory reward report ===')
  print(f'attempt={best["attempt"]}  T={len(r)}  first_success_t={first_succ}')
  print(f'r=φ(s,a)·ψ(g)  sum={r.sum():.4f}  mean={r.mean():.4f}  '
        f'std={r.std():.4f}  min={r.min():.4f}  max={r.max():.4f}')
  print('t  reward_raw  success')
  for t, (ri, si) in enumerate(zip(r, succ)):
    mark = '  <-- success' if si >= 0.5 else ''
    print(f'{t:02d}  {ri:+8.4f}    {si:.0f}{mark}')

  csv_path = os.path.join(OUT_DIR, 'c3t1_frozen_phi_psi_successful_traj.csv')
  np.savetxt(
      csv_path,
      np.stack([np.arange(len(r)), r, succ, best['dist']], axis=1),
      delimiter=',',
      header='t,reward_phi_dot_psi,success,dist',
      comments='')
  print(f'[probe] wrote {csv_path}')

  fig, ax = plt.subplots(figsize=(8, 3.2))
  ax.plot(np.arange(len(r)), r, marker='o', ms=3.5, lw=1.5, color='#1f4e79')
  if first_succ >= 0:
    ax.axvline(first_succ, color='#c45c26', ls='--', lw=1.2,
               label=f'first success t={first_succ}')
    ax.legend(frameon=False)
  ax.set_xlabel('macro step t')
  ax.set_ylabel(r'raw reward $r=\varphi(s,a)\cdot\psi(g)$')
  ax.set_title('creative-3-task1 · frozen CRL reward on successful traj\n'
               'policy+q from pd_crl_tau05_catselect_extrew1 latest.pkl')
  ax.grid(True, alpha=0.3)
  fig.tight_layout()
  png_path = os.path.join(OUT_DIR, 'c3t1_frozen_phi_psi_successful_traj.png')
  fig.savefig(png_path, dpi=160)
  print(f'[probe] wrote {png_path}')


if __name__ == '__main__':
  main()
