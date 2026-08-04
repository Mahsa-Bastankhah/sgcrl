"""Unit test for TD-Flow density estimator and updates."""
import jax
import jax.numpy as jnp
import optax
from contrastive.fm_density import (
    make_fm_density_networks,
    init_fm_params,
    generate_target_goals_bootstrapped,
    make_td_fm_density_update_fn,
)

def test_td_fm_density():
  print("Starting test_td_fm_density...")
  obs_dim = 10
  act_dim = 4
  goal_dim = 5
  batch_size = 16

  key = jax.random.PRNGKey(42)
  key_net, key_init, key_batch, key_update = jax.random.split(key, 4)

  # 1. Build networks
  fm_nets = make_fm_density_networks(
      obs_dim=obs_dim,
      act_dim=act_dim,
      goal_dim=goal_dim,
      hidden_layer_sizes=(64, 64),
      time_embedding=True,
      time_embed_dim=16,
  )

  # 2. Init parameters
  online_params = init_fm_params(fm_nets, key_init)
  target_params = online_params

  optimizer = optax.adam(1e-3)
  opt_state = optimizer.init(online_params)

  # 3. Create synthetic batch
  key_obs, key_act, key_next_obs = jax.random.split(key_batch, 3)
  s_0 = jax.random.normal(key_obs, (batch_size, obs_dim))
  g_0 = jax.random.normal(key_obs, (batch_size, goal_dim))
  obs = jnp.concatenate([s_0, g_0], axis=-1)

  action = jax.random.normal(key_act, (batch_size, act_dim))

  s_1 = jax.random.normal(key_next_obs, (batch_size, obs_dim))
  g_1 = jax.random.normal(key_next_obs, (batch_size, goal_dim))
  next_obs = jnp.concatenate([s_1, g_1], axis=-1)

  batch = {
      'obs': obs,
      'action': action,
      'next_obs': next_obs,
  }

  # 4. Test target goal generation
  key_boot = jax.random.split(key_update, 1)[0]
  x_1_boot = generate_target_goals_bootstrapped(
      fm_nets, target_params, s_1, action, key_boot, boot_steps=2, ode_solver='euler', goal_dim=goal_dim
  )
  assert x_1_boot.shape == (batch_size, goal_dim)
  assert jnp.all(jnp.isfinite(x_1_boot))
  print(f"Target goal generation successful! x_1_boot shape: {x_1_boot.shape}")

  # 5. Build TD-Flow update function
  update_fn = make_td_fm_density_update_fn(
      fm_nets, optimizer, obs_dim=obs_dim, gamma=0.99, target_tau=0.05, boot_steps=2
  )

  # 6. Perform 5 update steps
  cur_online = online_params
  cur_target = target_params
  cur_opt_state = opt_state

  for i in range(5):
    key_step = jax.random.fold_in(key_update, i)
    cur_online, cur_target, cur_opt_state, metrics = update_fn(
        cur_online, cur_target, cur_opt_state, batch, key_step
    )
    print(f"Step {i+1}: loss = {metrics['density_loss']:.6f}, "
          f"vel_pred_norm = {metrics['vel_pred_norm']:.4f}, "
          f"ratio_1step = {metrics['ratio_1step']:.2f}")

    assert jnp.isfinite(metrics['density_loss'])
    assert metrics['update_skipped_nonfinite'] == 0.0

  print("ALL TD-Flow tests passed successfully!")

if __name__ == '__main__':
  test_td_fm_density()
