#!/usr/bin/env python3
"""Evaluate Flow Matching Density Estimator diagnostics on existing checkpoints.

Loads checkpoints from completed training runs (e.g. pd_fm_creative3_task1_baseline,
pd_fm_creative3_task1_goal_norm, pd_fm_creative3_task1_baseline_ext_scale1),
collects rollout trajectories, and evaluates whether the Flow Matching density model
correctly assigns higher density (and lower velocity field error) to positive future
goals from its own trajectory compared to cross-trajectory negative goals.

Metrics evaluated (in plain ASCII logging):
1. FM Energy Loss Gap: E[ FM_loss(g_neg) - FM_loss(g_pos) ]  (higher is better)
2. Positive Win Rate:  % of pairs where FM_loss(g_pos) < FM_loss(g_neg)  (>50% is better)
3. Top-1 Goal Retrieval Accuracy: % of states where g_pos achieves lowest FM loss among B batch goals
4. Exact Log-Prob Gap (ODE): E[ log_prob(g_pos) - log_prob(g_neg) ]  (higher is better)
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_BUILDERBENCH_ROOT = os.environ.get(
    'BUILDERBENCH_ROOT', '/n/fs/mislresearch/builderbench')
if _BUILDERBENCH_ROOT not in sys.path:
  sys.path.insert(0, _BUILDERBENCH_ROOT)

import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np
import pickle

import contrastive
from contrastive import fm_density as _fm
from contrastive import ppo_learner
from envs.builderbench_jax_vec import JaxBuilderBenchVecEnv
from envs.builderbench_utils import (
    creative_cube_full_state_obs_dim,
    creative_cube_mj_episode_length,
    filter_pd_policy_state_obs,
    get_filtered_obs_dim,
    parse_bb_env_id,
    pd_policy_state_obs_dim,
    scaled_episode_length,
    sgcrl_env_name_to_bb_env_id,
)
from builderbench.creative_cube import CreativeCube, default_config
from utils.wrapper import (
    AutoResetWrapper,
    EpisodeWrapper,
    PDWrapper,
    VmapWrapper,
)

LOG_ROOT = "/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"

def find_target_run_dirs(pattern_list):
    """Automatically locate checkpoint directories matching patterns in scratch."""
    found_dirs = {}
    all_exp_dirs = glob.glob(os.path.join(LOG_ROOT, "*"))
    for exp_dir in sorted(all_exp_dirs):
        bname = os.path.basename(exp_dir)
        for pat in pattern_list:
            if pat in bname:
                subdirs = sorted(glob.glob(os.path.join(exp_dir, "ppo_*")))
                for sdir in subdirs:
                    ckpts = glob.glob(os.path.join(sdir, "checkpoints", "*.pkl"))
                    if ckpts:
                        key_name = f"{bname}/{os.path.basename(sdir)}"
                        found_dirs[key_name] = sdir
    return found_dirs

def load_checkpoint(run_dir):
    """Load latest checkpoint and run configuration."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    latest = os.path.join(ckpt_dir, "latest.pkl")
    if not os.path.isfile(latest):
        milestones = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_iter_*.pkl")))
        if not milestones:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        latest = milestones[-1]
    
    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, 'r', encoding='utf-8') as f:
        run_cfg = json.load(f)
        
    with open(latest, 'rb') as f:
        ckpt = pickle.load(f)
        
    return latest, run_cfg, ckpt

def evaluate_fm_diagnostics_on_run(run_dir, num_episodes=16, num_eval_timesteps=100):
    """Run diagnostics on a single run checkpoint."""
    ckpt_path, run_cfg, ckpt = load_checkpoint(run_dir)
    flags = run_cfg.get('flags', {})
    resolved = run_cfg.get('resolved_config', {})
    
    env_name = flags.get('env', 'builderbench_creative_3_task1')
    num_cubes, task_index = parse_bb_env_id(sgcrl_env_name_to_bb_env_id(env_name))
    
    use_pd = bool(flags.get('builderbench_use_pd', True))
    pd_duration = int(flags.get('builderbench_pd_duration', 5))
    obs_space_str = flags.get('obs_space', 'xy,select')
    obs_space_list = [s.strip() for s in obs_space_str.split(',')]
    ep_mult = float(flags.get('builderbench_episode_length_multiplier', 1.0))
    episode_len = scaled_episode_length(num_cubes, ep_mult)
    
    # Create vectorized environment for collecting rollouts first to get exact specs
    vec_env = JaxBuilderBenchVecEnv(
        env_name=env_name,
        num_envs=num_episodes,
        seed=42,
        use_pd=use_pd,
        pd_duration=pd_duration,
        pd_filter_policy_obs=True,
        fixed_target_goal=(None if run_cfg.get('fixed_start_end') is None else np.asarray(run_cfg['fixed_start_end'], dtype=np.float32)),
        obs_space_list=obs_space_list,
        episode_length_multiplier=ep_mult,
        permute_start_boxes=bool(flags.get('builderbench_permute_start_boxes', True)),
    )
    
    obs_dim = vec_env._state_obs_dim
    act_dim = vec_env._action_dim
    goal_dim = vec_env._goal_dim
    
    # Reconstruct Flow Matching networks matching config
    hidden = tuple(int(x) for x in resolved.get('hidden_layer_sizes', [256]*6))
    flow_steps = int(flags.get('fm_flow_steps', 10))
    time_embed = bool(flags.get('fm_time_embedding', False))
    time_embed_dim = int(flags.get('fm_time_embed_dim', 32))
    ode_solver = str(flags.get('fm_ode_solver', 'euler')).lower()
    
    fm_nets = _fm.make_fm_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        hidden_layer_sizes=hidden,
        flow_steps=flow_steps,
        time_embedding=time_embed,
        time_embed_dim=time_embed_dim,
        ode_solver=ode_solver,
    )
    
    # Extract velocity field parameters (stored in q_params)
    q_params = ckpt['q_params']
    
    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    env_keys = jax.random.split(reset_rng, num_episodes)
    env_state = vec_env._reset_fn(env_keys)
    
    # Collect a rollout trajectory matrix
    obs_list, act_list, goal_list = [], [], []
    for _ in range(min(vec_env._episode_length, num_eval_timesteps)):
        rng, step_rng = jax.random.split(rng)
        actions = jax.random.uniform(step_rng, (num_episodes, act_dim), minval=-1.0, maxval=1.0)
        
        # Policy state obs
        pol_obs = filter_pd_policy_state_obs(env_state.obs, num_cubes, obs_space_list) if use_pd else env_state.obs
        obs_list.append(pol_obs)
        act_list.append(actions)
        
        env_state = vec_env._step_fn(env_state, actions)
        
    obs_batch = jnp.stack(obs_list, axis=1)    # (B, T, obs_dim)
    act_batch = jnp.stack(act_list, axis=1)    # (B, T, act_dim)
    
    B, T, obs_dim = obs_batch.shape
    
    # Extract goal coordinates from state observations (first goal_dim dims = cube coordinates)
    state_goals = obs_batch[:, :, :goal_dim]   # (B, T, goal_dim)
    
    # Positive goals: future state goals from the same trajectory (10 steps ahead or final state)
    pos_goal_batch = jnp.roll(state_goals, shift=-10, axis=1)  # (B, T, goal_dim)
    
    # Negative goals: state goals from a DIFFERENT trajectory in the batch (roll across B axis)
    neg_goal_batch = jnp.roll(state_goals, shift=1, axis=0)    # (B, T, goal_dim)
    
    # Flatten batch x time
    flat_obs = obs_batch.reshape(-1, obs_dim)         # (N, obs_dim) where N = B*T
    flat_act = act_batch.reshape(-1, act_dim)         # (N, act_dim)
    flat_pos_goals = pos_goal_batch.reshape(-1, goal_dim) # (N, goal_dim)
    flat_neg_goals = neg_goal_batch.reshape(-1, goal_dim) # (N, goal_dim)
    
    # 1. Compute Flow Matching Velocity Loss (Energy Proxy)
    rng, loss_rng = jax.random.split(rng)
    t_rng, x0_rng = jax.random.split(loss_rng)
    
    N = flat_obs.shape[0]
    t_sample = jax.random.uniform(t_rng, (N, 1))
    x0_sample = jax.random.normal(x0_rng, (N, goal_dim))
    
    def compute_fm_err(goals):
        x_t = (1.0 - t_sample) * x0_sample + t_sample * goals
        target_v = goals - x0_sample
        pred_v = fm_nets.velocity_net.apply(q_params, flat_obs, flat_act, x_t, t_sample)
        return jnp.mean(jnp.square(pred_v - target_v), axis=-1)
        
    pos_fm_err = compute_fm_err(flat_pos_goals)
    neg_fm_err = compute_fm_err(flat_neg_goals)
    
    fm_loss_gap = float(jnp.mean(neg_fm_err - pos_fm_err))
    pos_win_rate = float(jnp.mean(pos_fm_err < neg_fm_err) * 100.0)
    
    # 2. Compute Top-1 Retrieval Accuracy across batch candidate goals
    # Evaluate pairwise energy matrix (N, B) between state N and final trajectory goals B
    traj_final_goals = state_goals[:, -1, :] # (B, goal_dim) final state of each trajectory
    
    # Compute energy of each state against all B trajectory goals
    def eval_pair(g_candidate):
        return compute_fm_err(jnp.tile(g_candidate[None, :], (N, 1)))
        
    # Evaluate matrix of candidate goal errors (B_cand, N)
    cand_errs = jax.vmap(eval_pair)(traj_final_goals) # (B, N)
    best_cand_idx = jnp.argmin(cand_errs, axis=0) # (N,)
    true_traj_idx = jnp.repeat(jnp.arange(B), T)   # (N,)
    
    top1_acc = float(jnp.mean(best_cand_idx == true_traj_idx) * 100.0)
    
    # 3. Compute Log-Prob via ODE Integration on a subset (to be fast)
    sub_idx = np.random.choice(N, size=min(64, N), replace=False)
    rng, logp_rng = jax.random.split(rng)
    
    sub_obs = flat_obs[sub_idx]
    sub_act = flat_act[sub_idx]
    sub_pos_g = flat_pos_goals[sub_idx]
    sub_neg_g = flat_neg_goals[sub_idx]
    
    logp_pos = _fm.fm_log_prob(fm_nets, q_params, sub_obs, sub_act, sub_pos_g, rng=logp_rng, mode='exact')
    logp_neg = _fm.fm_log_prob(fm_nets, q_params, sub_obs, sub_act, sub_neg_g, rng=logp_rng, mode='exact')
    
    log_prob_gap = float(jnp.mean(logp_pos - logp_neg))
    
    return {
        'checkpoint': os.path.basename(ckpt_path),
        'fm_loss_pos': float(jnp.mean(pos_fm_err)),
        'fm_loss_neg': float(jnp.mean(neg_fm_err)),
        'fm_loss_gap': fm_loss_gap,
        'pos_win_rate': pos_win_rate,
        'top1_acc': top1_acc,
        'logp_pos': float(jnp.mean(logp_pos)),
        'logp_neg': float(jnp.mean(logp_neg)),
        'log_prob_gap': log_prob_gap,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate FM density diagnostics on checkpoints.")
    parser.add_argument("--patterns", nargs="+", default=[
        "pd_fm_creative3_task1_baseline",
        "pd_fm_creative3_task1_goal_norm",
        "pd_fm_creative3_task1_baseline_ext_scale1",
    ], help="List of experiment directory patterns to search for.")
    args = parser.parse_args()
    
    print("================================================================================")
    print("           FLOW MATCHING DENSITY ESTIMATOR DIAGNOSTIC EVALUATION                ")
    print("================================================================================")
    print("Searching scratch logs for matching runs...\n")
    
    target_dirs = find_target_run_dirs(args.patterns)
    if not target_dirs:
        print("No matching directories found.")
        return
        
    print(f"Found {len(target_dirs)} run directory targets:")
    for name, path in target_dirs.items():
        print(f"  - {name}")
    print("\nStarting evaluation...\n")
    
    results = []
    for name, path in target_dirs.items():
        print(f"Evaluating {name}...")
        try:
            res = evaluate_fm_diagnostics_on_run(path)
            res['run_name'] = name
            results.append(res)
            print(f"  [OK] Win Rate: {res['pos_win_rate']:.1f}% | FM Loss Gap: {res['fm_loss_gap']:.4f} | LogP Gap: {res['log_prob_gap']:.4f}")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n================================================================================")
    print("                           DIAGNOSTIC SUMMARY REPORT                            ")
    print("================================================================================")
    header = f"{'Run Name':<55} | {'WinRate(%)':<10} | {'FM Loss Gap':<12} | {'LogP Gap':<10}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        run_str = r['run_name'] if len(r['run_name']) <= 55 else r['run_name'][:52] + "..."
        print(f"{run_str:<55} | {r['pos_win_rate']:<10.1f} | {r['fm_loss_gap']:<12.4f} | {r['log_prob_gap']:<10.4f}")
    print("================================================================================")

if __name__ == "__main__":
    main()
