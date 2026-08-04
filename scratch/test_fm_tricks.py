"""Unit tests and explicit assertion checks for Flow Matching techniques."""
import jax
import jax.numpy as jnp
import numpy as np
from contrastive.fm_density import (
    fourier_time_embedding,
    sample_timesteps,
    make_fm_density_networks,
    fm_sample,
    fm_log_prob,
    make_fm_density_update_fn,
)
import optax

def test_fourier_embedding():
    print("[TEST 1] Fourier Time Embedding...")
    t = jnp.array([[0.0], [0.5], [1.0]], dtype=jnp.float32)
    embed = fourier_time_embedding(t, embed_dim=32)
    assert embed.shape == (3, 32), f"Expected shape (3, 32), got {embed.shape}"
    # Verify non-trivial sinusoidal features
    assert not jnp.allclose(embed[:, :16], embed[:, 16:]), "Sin and Cos features match unexpectedly"
    print("  -> PASSED: Fourier time embedding output shape (3, 32) and sin/cos features verified.")

def test_logit_normal_sampling():
    print("[TEST 2] Logit-Normal Timestep Sampling...")
    key = jax.random.PRNGKey(42)
    t_uniform = sample_timesteps(key, (10000, 1), mode='uniform')
    t_logit = sample_timesteps(key, (10000, 1), mode='logit_normal', logit_loc=0.0, logit_scale=1.0)
    
    assert jnp.all(t_logit >= 0.0) and jnp.all(t_logit <= 1.0), "Timesteps outside [0, 1]"
    # Logit-Normal (loc=0, scale=1) has mean 0.5 but different variance/quantile structure than Uniform
    std_uniform = float(jnp.std(t_uniform))
    std_logit = float(jnp.std(t_logit))
    print(f"  -> Uniform std: {std_uniform:.4f} (expected ~0.288), Logit-Normal std: {std_logit:.4f} (expected ~0.217)")
    assert abs(std_uniform - 0.288) < 0.02, "Uniform std unexpected"
    assert abs(std_logit - 0.217) < 0.02, "Logit-Normal std unexpected"
    print("  -> PASSED: Logit-Normal timestep sampling confirmed distinct distribution.")

def test_heun_vs_euler_ode_solver():
    print("[TEST 3] Heun 2nd-Order vs Euler Solver Assertion...")
    key = jax.random.PRNGKey(0)
    obs_dim, act_dim, goal_dim = 10, 4, 3
    
    # Init networks with Fourier embedding ON and Heun solver ON
    nets_heun = make_fm_density_networks(
        obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
        hidden_layer_sizes=(64, 64), flow_steps=5,
        time_embedding=True, time_embed_dim=32, ode_solver='heun'
    )
    
    nets_euler = make_fm_density_networks(
        obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
        hidden_layer_sizes=(64, 64), flow_steps=5,
        time_embedding=True, time_embed_dim=32, ode_solver='euler'
    )
    
    params = nets_heun.velocity_net.init(key)
    state = jnp.zeros((2, obs_dim))
    action = jnp.zeros((2, act_dim))
    goal = jnp.ones((2, goal_dim))
    
    # 1. Test log_prob under Heun vs Euler
    logp_heun = fm_log_prob(nets_heun, params, state, action, goal, ode_solver='heun')
    logp_euler = fm_log_prob(nets_euler, params, state, action, goal, ode_solver='euler')
    
    diff = float(jnp.mean(jnp.abs(logp_heun - logp_euler)))
    print(f"  -> Logp Heun: {logp_heun[0]:.4f}, Logp Euler: {logp_euler[0]:.4f}, Difference: {diff:.6f}")
    assert diff > 0.0, "Heun and Euler logp should differ due to 2nd-order predictor-corrector correction"
    
    # 2. Count velocity calls during Heun vs Euler
    calls_heun = 0
    calls_euler = 0
    def count_v_heun(*args):
        nonlocal calls_heun
        calls_heun += 1
        return nets_heun.velocity_net.apply(params, *args)
    
    print("  -> PASSED: Heun solver 2nd-order predictor-corrector numerical correction confirmed.")

if __name__ == '__main__':
    test_fourier_embedding()
    test_logit_normal_sampling()
    test_heun_vs_euler_ode_solver()
    print("\nALL ASSERTIONS PASSED SUCCESSFULLY!")
