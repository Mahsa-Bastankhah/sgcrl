#!/usr/bin/env python3
"""Side-by-side CRL vs NF reward landscapes + policy rollouts on a point maze.

For a given checkpoint iteration, plots:
  - Left:  CRL PPO reward  φ(s,a)·ψ(g)  (policy mode action)
  - Right: NF PPO reward   log p_NF(g | s, a)

Each panel overlays the deterministic policy rollout on the maze.

Example::

  python scripts/plot_crl_nf_maze_compare.py \\
      --env point_SixteenRoomsActual4D \\
      --crl_ckpt logs/ppo_sixteenroomsactual4d/ppo_point_SixteenRoomsActual4D_120/checkpoints/ckpt_iter_0000000.pkl \\
      --nf_ckpt logs/ppo_nf_sixteenroomsactual4d/ppo_point_SixteenRoomsActual4D_0/checkpoints/ckpt_iter_0000000.pkl \\
      --output figs/sixteenrooms_crl_nf_iter0.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from acme import specs

import contrastive
from acme.jax import networks as networks_lib
import haiku as hk

from contrastive import nf_density as _nf
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from ppo_contrastive import fixed_goal_dict

# Reuse maze helpers from ppo_rollout_maze.py
from scripts.ppo_rollout_maze import (  # noqa: E402
    _build_networks,
    _draw_maze_panel,
    _get_raw_point_env,
    _infer_run_config_from_checkpoint,
    _make_maze_repr_field_fn,
    _point_state_goal_dim,
    _rollout_one,
)


def _load_run_config(ckpt_path: str) -> dict:
    cfg = _infer_run_config_from_checkpoint(ckpt_path)
    return cfg if isinstance(cfg, dict) else {}


def _is_legacy_flat_nf_params(q_params: dict) -> bool:
    return ('sa_encoder' not in q_params
            and any(k.startswith('sa_encoder/linear_') for k in q_params))


def _normalize_nf_q_params(q_params: dict) -> dict:
    """Flat merged checkpoints → nested ``{sa_encoder, nf_flow}``."""
    if 'sa_encoder' in q_params and 'nf_flow' in q_params:
        return q_params
    sa_enc, nf_flow = {}, {}
    for k, v in q_params.items():
        if k.startswith('sa_encoder/'):
            sa_enc[k[len('sa_encoder/'):]] = v
        elif k == 'sa_proj':
            sa_enc['sa_proj'] = v
        else:
            nf_flow[k] = v
    return {'sa_encoder': sa_enc, 'nf_flow': nf_flow}


def _infer_flow_l1_width(q_params: dict, rep_size: int, channels: int) -> int:
    for root in (q_params, q_params.get('nf_flow', {})):
        if not isinstance(root, dict):
            continue
        layer = root.get('s_0_l1')
        if isinstance(layer, dict) and 'w' in layer:
            return int(layer['w'].shape[1])
    return channels + rep_size


def _make_legacy_nf_density_networks(
    obs_dim: int,
    act_dim: int,
    goal_dim: int,
    hidden_layer_sizes: Sequence[int],
    rep_size: int = 64,
    num_blocks: int = 8,
    channels: int = 256,
    l1_width: Optional[int] = None,
) -> _nf.NFDensityNetworks:
    l1_width = int(l1_width if l1_width is not None else channels + rep_size)
    """MLP SA encoder + RealNVP without PLU (matches early NF PPO checkpoints)."""
    split_cond = (goal_dim + 1) // 2
    split_trans = goal_dim // 2
    w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    zero_init = hk.initializers.Constant(0.)

    def _sa_fn(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, action], axis=-1)
        for i, width in enumerate(hidden_layer_sizes):
            x = hk.Linear(width, w_init=w_init, name=f'linear_{i}')(x)
            if i < len(hidden_layer_sizes) - 1:
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                 name=f'ln_{i}')(x)
            x = jax.nn.swish(x)
        return hk.Linear(rep_size, w_init=w_init, name='sa_proj')(x)

    def _flow_log_prob(goal: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        x = goal
        log_dets = jnp.zeros(x.shape[0], dtype=x.dtype)
        for i in range(num_blocks):
            x_cond = x[:, :split_cond]
            x_trans = x[:, split_cond:]
            cond_in = jnp.concatenate([x_cond, y], axis=-1)

            s_h = hk.Linear(l1_width, w_init=w_init, name=f's_{i}_l1')(cond_in)
            s_h = jax.nn.leaky_relu(s_h)
            s_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name=f's_{i}_ln1')(s_h)
            s_h = hk.Linear(channels, w_init=w_init, name=f's_{i}_l2')(s_h)
            s_h = jax.nn.leaky_relu(s_h)
            s_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name=f's_{i}_ln2')(s_h)
            s = hk.Linear(split_trans, w_init=zero_init, b_init=zero_init,
                          name=f's_{i}_out')(s_h)

            t_h = hk.Linear(l1_width, w_init=w_init, name=f't_{i}_l1')(cond_in)
            t_h = jax.nn.leaky_relu(t_h)
            t_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name=f't_{i}_ln1')(t_h)
            t_h = hk.Linear(channels, w_init=w_init, name=f't_{i}_l2')(t_h)
            t_h = jax.nn.leaky_relu(t_h)
            t_h = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                               name=f't_{i}_ln2')(t_h)
            t = hk.Linear(split_trans, w_init=zero_init, b_init=zero_init,
                          name=f't_{i}_out')(t_h)

            x_trans_new = (x_trans - t) * jnp.exp(-s)
            log_dets = log_dets - jnp.sum(s, axis=-1)
            x = jnp.concatenate([x_cond, x_trans_new], axis=-1)
            if i < num_blocks - 1:
                x = x[:, ::-1]

        log_norm = -0.5 * goal_dim * jnp.log(2.0 * jnp.pi)
        log_prior = -0.5 * jnp.sum(x ** 2, axis=-1) + log_norm
        return log_prior + log_dets

    sa_transformed = hk.without_apply_rng(hk.transform(_sa_fn))
    flow_transformed = hk.without_apply_rng(hk.transform(_flow_log_prob))
    dummy_state = np.zeros((1, obs_dim), dtype=np.float32)
    dummy_action = np.zeros((1, act_dim), dtype=np.float32)
    dummy_goal = np.zeros((1, goal_dim), dtype=np.float32)
    dummy_y = np.zeros((1, rep_size), dtype=np.float32)
    sa_encoder_net = networks_lib.FeedForwardNetwork(
        init=lambda key: sa_transformed.init(key, dummy_state, dummy_action),
        apply=sa_transformed.apply,
    )
    flow_net = networks_lib.FeedForwardNetwork(
        init=lambda key: flow_transformed.init(key, dummy_goal, dummy_y),
        apply=flow_transformed.apply,
    )
    return _nf.NFDensityNetworks(
        sa_encoder_net=sa_encoder_net,
        flow_net=flow_net,
        goal_encoder_net=None,
        flow_dim=goal_dim,
    )


def _build_nf_networks(env_name: str, seed: int, run_cfg: dict, q_params: dict):
    probe_env, obs_dim = contrastive_utils.make_environment(
        env_name, start_index=0, end_index=-1, seed=seed,
        fixed_start_end=fixed_goal_dict[env_name])
    env_spec = specs.make_environment_spec(probe_env)
    act_dim = int(np.prod(env_spec.actions.shape))
    del probe_env
    goal_dim = obs_dim
    hidden = tuple(run_cfg.get('hidden_layer_sizes') or [256] * 6)
    rep_size = int(run_cfg.get('nf_rep_size', 64))
    num_blocks = int(run_cfg.get('nf_num_blocks', 8))
    channels = int(run_cfg.get('nf_coupling_width', 256))
    if _is_legacy_flat_nf_params(q_params):
        l1_w = _infer_flow_l1_width(q_params, rep_size, channels)
        print(f'[compare] legacy NF arch (MLP encoder, flow_l1={l1_w})',
              flush=True)
        nf_nets = _make_legacy_nf_density_networks(
            obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
            hidden_layer_sizes=hidden, rep_size=rep_size,
            num_blocks=num_blocks, channels=channels, l1_width=l1_w)
    else:
        nf_nets = _nf.make_nf_density_networks(
            obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
            hidden_layer_sizes=hidden, rep_size=rep_size,
            num_blocks=num_blocks, channels=channels)
    return nf_nets, obs_dim


def _make_nf_reward_field_fn(nf_networks, networks, state_dim: int):
    """log p_NF(g | s, a_π(s,g)) on a batch of maze positions."""

    @jax.jit
    def _eval(policy_p, nf_p, pos_batch: jnp.ndarray, goal_vec: jnp.ndarray):
        g = jnp.broadcast_to(goal_vec[None, :], (pos_batch.shape[0], state_dim))
        obs_pg = jnp.concatenate([pos_batch, g], axis=-1)
        dist = networks.policy_network.apply(policy_p, obs_pg)
        act = networks.sample_eval(dist, jax.random.PRNGKey(0))
        return _nf.nf_log_prob(nf_networks, nf_p, pos_batch, act, g)

    return _eval


def _scalar_grid_over_maze(
    walls: np.ndarray,
    goal: np.ndarray,
    state_dim: int,
    eval_batch_fn,
    subcells: int = 5,
    extra_fill: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluate a scalar field on free maze cells (NaN on walls)."""
    H, W = walls.shape
    subcells = max(1, int(subcells))
    rows, cols = np.where(walls == 0)
    rr = (np.arange(subcells, dtype=np.float32) + 0.5) / float(subcells)
    cc = (np.arange(subcells, dtype=np.float32) + 0.5) / float(subcells)
    offsets = np.stack(np.meshgrid(rr, cc, indexing='ij'), axis=-1).reshape(-1, 2)
    base = np.stack([rows.astype(np.float32), cols.astype(np.float32)], axis=-1)
    pos_2d = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
    if state_dim > 2:
        n_extra = state_dim - 2
        if extra_fill is None:
            fill = np.zeros(n_extra, dtype=np.float32)
        else:
            fill = np.asarray(extra_fill, dtype=np.float32).reshape(n_extra)
        pos_full = np.concatenate(
            [pos_2d, np.broadcast_to(fill, (len(pos_2d), n_extra))],
            axis=-1)
    else:
        pos_full = pos_2d
    gvec = jnp.asarray(goal, dtype=jnp.float32)
    pos_j = jnp.asarray(pos_full, dtype=jnp.float32)
    values = np.asarray(eval_batch_fn(pos_j, gvec), dtype=np.float32)

    HH, WW = H * subcells, W * subcells
    grid = np.full((HH, WW), np.nan, dtype=np.float32)
    i0 = np.repeat(rows * subcells, subcells * subcells)
    j0 = np.repeat(cols * subcells, subcells * subcells)
    di = np.tile(np.repeat(np.arange(subcells), subcells), rows.shape[0])
    dj = np.tile(np.tile(np.arange(subcells), subcells), rows.shape[0])
    grid[i0 + di, j0 + dj] = values
    return grid


def _crl_reward_grid(
    walls, goal, policy_p, q_p, value_p, networks, state_dim, subcells,
    repr_norm: bool,
    extra_fill: Optional[np.ndarray] = None,
):
    eval_fields = _make_maze_repr_field_fn(
        networks, state_dim, normalize_repr=repr_norm)

    def _batch(pos_j, gvec):
        s0 = jnp.zeros((pos_j.shape[0], state_dim), dtype=jnp.float32)
        r, _ = eval_fields(policy_p, q_p, value_p, pos_j, gvec, s0)
        return r

    return _scalar_grid_over_maze(
        walls, goal, state_dim, _batch, subcells, extra_fill=extra_fill)


def _nf_reward_grid(
    walls, goal, policy_p, nf_p, nf_networks, networks, state_dim, subcells,
    extra_fill: Optional[np.ndarray] = None,
):
    eval_nf = _make_nf_reward_field_fn(nf_networks, networks, state_dim)

    def _batch(pos_j, gvec):
        return eval_nf(policy_p, nf_p, pos_j, gvec)

    return _scalar_grid_over_maze(
        walls, goal, state_dim, _batch, subcells, extra_fill=extra_fill)


def _extra_dim_grid_values(
    n_extra: int,
    n_per_dim: int = 3,
    lo: float = 0.0,
    hi: float = 20.0,
) -> list[np.ndarray]:
    """Return n_per_dim^n_extra combinations of extra-dim values."""
    if n_extra <= 0:
        return [np.zeros(0, dtype=np.float32)]
    import itertools
    vals = [float(v) for v in np.linspace(lo, hi, n_per_dim, dtype=np.float32)]
    return [
        np.asarray(t, dtype=np.float32)
        for t in itertools.product(*([vals] * n_extra))
    ]


def _plot_compare(
    walls: np.ndarray,
    goal: np.ndarray,
    crl_states: np.ndarray,
    nf_states: np.ndarray,
    crl_heat: np.ndarray,
    nf_heat: np.ndarray,
    title: str,
    out_path: str,
    crl_success: bool,
    nf_success: bool,
    crl_reward: float,
    nf_reward: float,
    fig_scale: float = 1.5,
):
    H, W = walls.shape
    goal_2d = goal[:2]
    h_in = (5 * H / max(W, 1)) * fig_scale
    fig, axes = plt.subplots(
        1, 2, figsize=(14 * fig_scale, h_in), layout='constrained')
    fig.suptitle(title, fontsize=12)

    crl_traj = [crl_states[:, :2]]
    nf_traj = [nf_states[:, :2]]
    n_extra = crl_states.shape[1] - 2
    crl_extra = [crl_states[:, 2:]] if n_extra > 0 else None
    nf_extra = [nf_states[:, 2:]] if n_extra > 0 else None

    _draw_maze_panel(
        axes[0], walls, None, crl_traj, goal_2d, crl_heat,
        r'CRL reward $\phi(s,a)\!\cdot\!\psi(g)$', extra_dims_trajs=crl_extra)
    axes[0].set_title(
        f'CRL  len={len(crl_states)}  env_reward={crl_reward:.2f}  '
        f'success={crl_success}')

    _draw_maze_panel(
        axes[1], walls, None, nf_traj, goal_2d, nf_heat,
        r'NF reward $\log p(g|s,a)$', extra_dims_trajs=nf_extra)
    axes[1].set_title(
        f'NF  len={len(nf_states)}  env_reward={nf_reward:.2f}  '
        f'success={nf_success}')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[compare] wrote {out_path}')


def _plot_extra_dim_grid(
    walls: np.ndarray,
    goal: np.ndarray,
    crl_states: np.ndarray,
    nf_states: np.ndarray,
    crl_heats: list[np.ndarray],
    nf_heats: list[np.ndarray],
    extra_combos: list[np.ndarray],
    title: str,
    out_path: str,
    crl_success: bool,
    nf_success: bool,
    crl_reward: float,
    nf_reward: float,
    grid_size: int = 3,
    fig_scale: float = 1.2,
):
    """3×3 (or n×n) grid of CRL|NF heatmap pairs at fixed extra-dim values."""
    n_cells = len(extra_combos)
    grid_n = int(np.ceil(np.sqrt(n_cells)))
    H, W = walls.shape
    goal_2d = goal[:2]
    crl_traj = [crl_states[:, :2]]
    nf_traj = [nf_states[:, :2]]
    n_extra = crl_states.shape[1] - 2
    crl_extra = [crl_states[:, 2:]] if n_extra > 0 else None
    nf_extra = [nf_states[:, 2:]] if n_extra > 0 else None

    cell_h = (4.5 * H / max(W, 1)) * fig_scale
    cell_w = 4.8 * fig_scale
    fig = plt.figure(
        figsize=(cell_w * grid_n * 2, cell_h * grid_n), layout='constrained')
    fig.suptitle(
        (f'{title}\n'
         f'CRL rollout: len={len(crl_states)} reward={crl_reward:.2f} '
         f'success={crl_success}  |  '
         f'NF rollout: len={len(nf_states)} reward={nf_reward:.2f} '
         f'success={nf_success}'),
        fontsize=11)

    outer = fig.add_gridspec(grid_n, grid_n, wspace=0.15, hspace=0.35)
    for idx, extra_fill in enumerate(extra_combos):
        r, c = divmod(idx, grid_n)
        inner = outer[r, c].subgridspec(1, 2, wspace=0.08)
        ax_crl = fig.add_subplot(inner[0, 0])
        ax_nf = fig.add_subplot(inner[0, 1])
        if n_extra == 1:
            extra_lbl = f'extra dim = {extra_fill[0]:.1f}'
        else:
            extra_lbl = ', '.join(
                f'd{i + 3}={extra_fill[i]:.1f}' for i in range(n_extra))
        ax_crl.set_title(extra_lbl, fontsize=9, pad=4)
        _draw_maze_panel(
            ax_crl, walls, None, crl_traj, goal_2d, crl_heats[idx],
            r'CRL $\phi\!\cdot\!\psi$', extra_dims_trajs=crl_extra)
        _draw_maze_panel(
            ax_nf, walls, None, nf_traj, goal_2d, nf_heats[idx],
            r'NF $\log p(g|s,a)$', extra_dims_trajs=nf_extra)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'[compare] wrote extra-dim grid {out_path}')


def _list_ckpt_iters(ckpt_dir: str) -> list[int]:
    import glob
    import re
    iters = []
    for path in glob.glob(os.path.join(ckpt_dir, 'ckpt_iter_*.pkl')):
        m = re.search(r'ckpt_iter_(\d+)\.pkl$', path)
        if m:
            iters.append(int(m.group(1)))
    return sorted(iters)


def _snap_iter(target: int, available: list[int]) -> int:
    """Largest value in *available* that is <= *target*."""
    candidates = [i for i in available if i <= target]
    if not candidates:
        raise ValueError(f'no checkpoint at or before iter {target} in {available}')
    return candidates[-1]


def _run_one_compare(
    env: str,
    crl_ckpt_path: str,
    nf_ckpt_path: str,
    output: str,
    seed: int,
    crl_seed: int,
    nf_seed: int,
    heatmap_subcells: int,
    fig_scale: float,
    max_steps: int,
    label_iter: Optional[int] = None,
) -> None:
    crl_ckpt = ppo_learner.load_checkpoint(crl_ckpt_path)
    nf_ckpt = ppo_learner.load_checkpoint(nf_ckpt_path)
    crl_iter = int(crl_ckpt.get('iteration', -1))
    nf_iter = int(nf_ckpt.get('iteration', -1))

    crl_cfg = _load_run_config(crl_ckpt_path)
    nf_cfg = _load_run_config(nf_ckpt_path)
    crl_seed = crl_seed if crl_seed >= 0 else int(crl_cfg.get('seed', seed))
    nf_seed = nf_seed if nf_seed >= 0 else int(nf_cfg.get('seed', seed))

    tag = f'label={label_iter}' if label_iter is not None else 'single'
    print(f'[compare] {tag}  env={env}  CRL iter={crl_iter}  NF iter={nf_iter}',
          flush=True)

    networks, _ = _build_networks(env, seed=seed)
    nf_q_raw = nf_ckpt['q_params']
    nf_networks, obs_dim = _build_nf_networks(
        env, seed=seed, run_cfg=nf_cfg, q_params=nf_q_raw)
    gym_env, _, env_max_steps, walls = _get_raw_point_env(env)
    state_dim = _point_state_goal_dim(gym_env)
    max_steps = env_max_steps if max_steps < 0 else int(max_steps)

    crl_policy = crl_ckpt['policy_params']
    crl_q = crl_ckpt['q_params']
    crl_value = crl_ckpt['value_params']
    nf_policy = nf_ckpt['policy_params']
    nf_q = _normalize_nf_q_params(nf_q_raw)

    crl_states, goal, crl_rew, crl_ok = _rollout_one(
        crl_policy, gym_env, networks, max_steps,
        stochastic=False, seed=crl_seed, state_dim=state_dim)
    gym_env.reset()
    nf_states, _, nf_rew, nf_ok = _rollout_one(
        nf_policy, gym_env, networks, max_steps,
        stochastic=False, seed=nf_seed, state_dim=state_dim)
    goal = goal.astype(np.float32)

    repr_norm = bool(crl_cfg.get('repr_norm', False))
    crl_heat = _crl_reward_grid(
        walls, goal, crl_policy, crl_q, crl_value, networks, state_dim,
        heatmap_subcells, repr_norm)
    nf_heat = _nf_reward_grid(
        walls, goal, nf_policy, nf_q, nf_networks, networks, state_dim,
        heatmap_subcells)

    mismatch = ''
    if label_iter is not None and (crl_iter != label_iter or nf_iter != label_iter):
        mismatch = f'  (snap: CRL@{crl_iter} NF@{nf_iter})'
    title = (f'{env}  label iter={label_iter if label_iter is not None else crl_iter}'
             f'{mismatch}\n'
             f'goal=(row={goal[0]:.1f}, col={goal[1]:.1f}'
             f'{", extra="+", ".join(f"{goal[i]:.2f}" for i in range(2, len(goal))) if len(goal)>2 else ""})')
    _plot_compare(
        walls, goal, crl_states, nf_states, crl_heat, nf_heat,
        title=title, out_path=output,
        crl_success=crl_ok, nf_success=nf_ok,
        crl_reward=crl_rew, nf_reward=nf_rew,
        fig_scale=fig_scale)


def _run_one_extra_dim_grid(
    env: str,
    crl_ckpt_path: str,
    nf_ckpt_path: str,
    output: str,
    seed: int,
    crl_seed: int,
    nf_seed: int,
    heatmap_subcells: int,
    fig_scale: float,
    max_steps: int,
    extra_grid_points: int,
    extra_dim_lo: float,
    extra_dim_hi: float,
    label_iter: Optional[int] = None,
) -> None:
    crl_ckpt = ppo_learner.load_checkpoint(crl_ckpt_path)
    nf_ckpt = ppo_learner.load_checkpoint(nf_ckpt_path)
    crl_iter = int(crl_ckpt.get('iteration', -1))
    nf_iter = int(nf_ckpt.get('iteration', -1))

    crl_cfg = _load_run_config(crl_ckpt_path)
    nf_cfg = _load_run_config(nf_ckpt_path)
    crl_seed = crl_seed if crl_seed >= 0 else int(crl_cfg.get('seed', seed))
    nf_seed = nf_seed if nf_seed >= 0 else int(nf_cfg.get('seed', seed))

    tag = f'label={label_iter}' if label_iter is not None else 'single'
    print(f'[compare] extra-grid {tag}  env={env}  CRL iter={crl_iter}  '
          f'NF iter={nf_iter}', flush=True)

    networks, _ = _build_networks(env, seed=seed)
    nf_q_raw = nf_ckpt['q_params']
    nf_networks, obs_dim = _build_nf_networks(
        env, seed=seed, run_cfg=nf_cfg, q_params=nf_q_raw)
    gym_env, _, env_max_steps, walls = _get_raw_point_env(env)
    state_dim = _point_state_goal_dim(gym_env)
    n_extra = state_dim - 2
    if n_extra <= 0:
        raise ValueError(
            f'extra_dim_grid requires state_dim > 2 (got {state_dim} for {env})')

    max_steps = env_max_steps if max_steps < 0 else int(max_steps)
    crl_policy = crl_ckpt['policy_params']
    crl_q = crl_ckpt['q_params']
    crl_value = crl_ckpt['value_params']
    nf_policy = nf_ckpt['policy_params']
    nf_q = _normalize_nf_q_params(nf_q_raw)

    crl_states, goal, crl_rew, crl_ok = _rollout_one(
        crl_policy, gym_env, networks, max_steps,
        stochastic=False, seed=crl_seed, state_dim=state_dim)
    gym_env.reset()
    nf_states, _, nf_rew, nf_ok = _rollout_one(
        nf_policy, gym_env, networks, max_steps,
        stochastic=False, seed=nf_seed, state_dim=state_dim)
    goal = goal.astype(np.float32)

    extra_combos = _extra_dim_grid_values(
        n_extra, n_per_dim=extra_grid_points,
        lo=extra_dim_lo, hi=extra_dim_hi)
    repr_norm = bool(crl_cfg.get('repr_norm', False))
    crl_heats, nf_heats = [], []
    for extra_fill in extra_combos:
        crl_heats.append(_crl_reward_grid(
            walls, goal, crl_policy, crl_q, crl_value, networks, state_dim,
            heatmap_subcells, repr_norm, extra_fill=extra_fill))
        nf_heats.append(_nf_reward_grid(
            walls, goal, nf_policy, nf_q, nf_networks, networks, state_dim,
            heatmap_subcells, extra_fill=extra_fill))

    mismatch = ''
    if label_iter is not None and (crl_iter != label_iter or nf_iter != label_iter):
        mismatch = f'  (snap: CRL@{crl_iter} NF@{nf_iter})'
    title = (f'{env}  label iter={label_iter if label_iter is not None else crl_iter}'
             f'{mismatch}\n'
             f'goal maze=({goal[0]:.1f}, {goal[1]:.1f})'
             f'{", extra="+", ".join(f"{goal[i]:.1f}" for i in range(2, len(goal))) if len(goal)>2 else ""}'
             f'  |  extra grid {extra_grid_points}×{extra_grid_points} '
             f'in [{extra_dim_lo:.1f}, {extra_dim_hi:.1f}]')
    _plot_extra_dim_grid(
        walls, goal, crl_states, nf_states, crl_heats, nf_heats,
        extra_combos, title=title, out_path=output,
        crl_success=crl_ok, nf_success=nf_ok,
        crl_reward=crl_rew, nf_reward=nf_rew,
        grid_size=extra_grid_points, fig_scale=fig_scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--env', default='point_SixteenRoomsActual4D')
    ap.add_argument('--crl_ckpt', default=None,
                    help='Single CRL checkpoint .pkl (mutually exclusive with --crl_ckpt_dir).')
    ap.add_argument('--nf_ckpt', default=None,
                    help='Single NF checkpoint .pkl (mutually exclusive with --nf_ckpt_dir).')
    ap.add_argument('--crl_ckpt_dir', default=None,
                    help='Directory of CRL ckpt_iter_*.pkl for batch mode.')
    ap.add_argument('--nf_ckpt_dir', default=None,
                    help='Directory of NF ckpt_iter_*.pkl for batch mode.')
    ap.add_argument('--output', default=None,
                    help='Output PNG (single mode) or directory ending in / (batch mode).')
    ap.add_argument('--seed', type=int, default=0,
                    help='Rollout seed (use each run\'s training seed if paths differ).')
    ap.add_argument('--crl_seed', type=int, default=-1,
                    help='Override CRL rollout seed (-1: use --seed).')
    ap.add_argument('--nf_seed', type=int, default=-1,
                    help='Override NF rollout seed (-1: use --seed).')
    ap.add_argument('--heatmap_subcells', type=int, default=5,
                    help='Subcells per maze cell (5 -> 25 samples/cell).')
    ap.add_argument('--fig_scale', type=float, default=1.4)
    ap.add_argument('--max_steps', type=int, default=-1)
    ap.add_argument('--extra_dim_grid', action='store_true',
                    help='3×3 grid over extra dims (requires state_dim > 2).')
    ap.add_argument('--extra_grid_points', type=int, default=3,
                    help='Values per extra dim (3 → 3×3 grid for 2 extra dims).')
    ap.add_argument('--extra_dim_lo', type=float, default=0.0)
    ap.add_argument('--extra_dim_hi', type=float, default=20.0,
                    help='Extra-dim sampling range (PointEnv clamps to [0, 20]).')
    ap.add_argument('--checkpoint_limit', type=int, default=0,
                    help='Batch mode: only plot first N checkpoint labels (0 = all).')
    args = ap.parse_args()

    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('MPLBACKEND', 'Agg')

    batch = args.crl_ckpt_dir is not None and args.nf_ckpt_dir is not None
    if batch:
        if not args.output:
            ap.error('batch mode requires --output=<directory/>')
        out_dir = args.output
        if not out_dir.endswith(os.sep):
            out_dir = out_dir + os.sep
        os.makedirs(out_dir, exist_ok=True)
        crl_iters = _list_ckpt_iters(args.crl_ckpt_dir)
        nf_iters = _list_ckpt_iters(args.nf_ckpt_dir)
        if not crl_iters or not nf_iters:
            raise SystemExit('no checkpoints found in one or both dirs')
        label_iters = sorted(set(crl_iters) | set(nf_iters))
        if args.checkpoint_limit > 0:
            label_iters = label_iters[:args.checkpoint_limit]
        mode = 'extra-dim grid' if args.extra_dim_grid else 'standard'
        print(f'[compare] batch ({mode}): {len(label_iters)} label iters  '
              f'({len(crl_iters)} CRL, {len(nf_iters)} NF ckpts)', flush=True)
        failed = 0
        for label in label_iters:
            crl_i = label if label in crl_iters else _snap_iter(label, crl_iters)
            nf_i = label if label in nf_iters else _snap_iter(label, nf_iters)
            crl_path = os.path.join(
                args.crl_ckpt_dir, f'ckpt_iter_{crl_i:07d}.pkl')
            nf_path = os.path.join(
                args.nf_ckpt_dir, f'ckpt_iter_{nf_i:07d}.pkl')
            suffix = '_extra_grid' if args.extra_dim_grid else ''
            out_path = os.path.join(out_dir, f'iter_{label:07d}{suffix}.png')
            try:
                if args.extra_dim_grid:
                    _run_one_extra_dim_grid(
                        env=args.env,
                        crl_ckpt_path=crl_path,
                        nf_ckpt_path=nf_path,
                        output=out_path,
                        seed=args.seed,
                        crl_seed=args.crl_seed,
                        nf_seed=args.nf_seed,
                        heatmap_subcells=args.heatmap_subcells,
                        fig_scale=args.fig_scale,
                        max_steps=args.max_steps,
                        extra_grid_points=args.extra_grid_points,
                        extra_dim_lo=args.extra_dim_lo,
                        extra_dim_hi=args.extra_dim_hi,
                        label_iter=label,
                    )
                else:
                    _run_one_compare(
                        env=args.env,
                        crl_ckpt_path=crl_path,
                        nf_ckpt_path=nf_path,
                        output=out_path,
                        seed=args.seed,
                        crl_seed=args.crl_seed,
                        nf_seed=args.nf_seed,
                        heatmap_subcells=args.heatmap_subcells,
                        fig_scale=args.fig_scale,
                        max_steps=args.max_steps,
                        label_iter=label,
                    )
            except Exception as exc:
                failed += 1
                print(f'[compare] FAILED iter {label}: {exc}', flush=True)
        print(f'[compare] batch done: {len(label_iters) - failed}/{len(label_iters)} ok',
              flush=True)
        if failed:
            raise SystemExit(1)
        return

    if not args.crl_ckpt or not args.nf_ckpt or not args.output:
        ap.error('single mode requires --crl_ckpt, --nf_ckpt, and --output')
    if args.extra_dim_grid:
        _run_one_extra_dim_grid(
            env=args.env,
            crl_ckpt_path=args.crl_ckpt,
            nf_ckpt_path=args.nf_ckpt,
            output=args.output,
            seed=args.seed,
            crl_seed=args.crl_seed,
            nf_seed=args.nf_seed,
            heatmap_subcells=args.heatmap_subcells,
            fig_scale=args.fig_scale,
            max_steps=args.max_steps,
            extra_grid_points=args.extra_grid_points,
            extra_dim_lo=args.extra_dim_lo,
            extra_dim_hi=args.extra_dim_hi,
        )
    else:
        _run_one_compare(
            env=args.env,
            crl_ckpt_path=args.crl_ckpt,
            nf_ckpt_path=args.nf_ckpt,
            output=args.output,
            seed=args.seed,
            crl_seed=args.crl_seed,
            nf_seed=args.nf_seed,
            heatmap_subcells=args.heatmap_subcells,
            fig_scale=args.fig_scale,
            max_steps=args.max_steps,
        )


if __name__ == '__main__':
    main()
