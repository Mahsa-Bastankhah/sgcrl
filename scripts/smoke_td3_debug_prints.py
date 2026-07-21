"""Minimal smoke test for TD3 debug prints (s, s', g).

Builds a tiny EpisodeReplay, samples a batch, runs a few TD3 updates with a
stub policy so the [td3 debug] prints fire without a full PPO train loop.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from contrastive.ppo_learner import EpisodeReplay
from contrastive import td3_density as td3


class _StubDist:
  def sample(self, seed):
    del seed
    # Match batch size dynamically via closure — set below.
    return self._actions

  def __init__(self, actions):
    self._actions = actions


class _StubPolicy:
  def apply(self, params, obs):
    del params
    B = obs.shape[0]
    act_dim = 2
    return _StubDist(jnp.zeros((B, act_dim), dtype=jnp.float32))


def main():
  print('[smoke] devices:', jax.devices())
  obs_dim, act_dim, goal_dim = 4, 2, 4
  gamma = 0.99

  # Synthetic episodes: linear walk so s' and s_f are easy to read.
  replay = EpisodeReplay(
      capacity=10_000, obs_dim=obs_dim, discount=gamma,
      start_index=0, end_index=-1)
  rng = np.random.default_rng(0)
  for ep in range(8):
    T = 20
    # obs has shape (T+1, 2*obs_dim) packed as [state; goal_placeholder]
    # EpisodeReplay only uses [:obs_dim] as state.
    states = np.cumsum(
        rng.normal(0, 0.3, size=(T + 1, obs_dim)).astype(np.float32), axis=0)
    # Fake full obs = [state; zeros goal slot] — sampler rebuilds goals from s_j.
    full_obs = np.concatenate(
        [states, np.zeros_like(states)], axis=-1)
    actions = rng.normal(0, 0.1, size=(T, act_dim)).astype(np.float32)
    replay.add_episode(full_obs, actions)
  print(f'[smoke] replay size={replay.size}')

  nets = td3.make_td3_density_networks(
      obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
      hidden_layer_sizes=(256,) * 6)
  q_params = td3.init_td3_params(nets, jax.random.PRNGKey(0))
  opt = optax.adam(3e-4)
  opt_state = opt.init(td3.online_td3_params(q_params))

  update = td3.make_td3_density_update_fn(
      nets,
      policy_network=_StubPolicy(),
      sample_fn=lambda dist, key: dist.sample(seed=key),
      optimizer=opt,
      obs_dim=obs_dim,
      start_index=0,
      end_index=-1,
      discount=gamma,
      tau=0.005,
      goal_tol=1e-2,
      use_target_policy=False,
  )

  key = jax.random.PRNGKey(1)
  # First 5 updates trigger the debug prints.
  policy_params = {}
  policy_target = {}
  for i in range(5):
    batch_np = replay.sample(8, rng)
    batch = {k: jnp.asarray(v) for k, v in batch_np.items()}
    key, k = jax.random.split(key)
    q_params, opt_state, metrics, policy_target = update(
        q_params, opt_state, batch, k, policy_params, policy_target)
    print(f'[smoke] step {i}: loss={float(metrics["td3_qf_loss"]):.4f}  '
          f'hit_frac={float(metrics["td3_goal_hit_frac"]):.4f}')
  print('[smoke] done')


if __name__ == '__main__':
  main()
