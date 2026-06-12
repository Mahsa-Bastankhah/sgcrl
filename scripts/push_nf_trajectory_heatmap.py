"""Push NF trajectory visualization: scene previews + log p(ψ_f | s₀, π(s₀)).

Along the line from initial puck position to target, construct hindsight goals
ψ_f with the puck at each grid point and the hand slightly behind (+3 cm z,
−8 cm y).  Evaluate NF log density conditioned on reset state s₀ and the
policy action a₀ = π(s₀).

Example::

  # 1) Preview trajectory scenes (no checkpoint needed)
  python scripts/push_nf_trajectory_heatmap.py --mode preview

  # 2) Heatmaps for every checkpoint in a run
  python scripts/push_nf_trajectory_heatmap.py --mode heatmaps \\
      --checkpoint=logs/ppo_nf_push_v4/ppo_sawyer_push_0/checkpoints \\
      --output=figs/push_nf_traj_hm/seed0/

  # 3) Both
  python scripts/push_nf_trajectory_heatmap.py --mode all \\
      --checkpoint=logs/ppo_nf_push_v4/ppo_sawyer_push_0/checkpoints \\
      --output=figs/push_nf_traj_hm/seed0/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import List, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
from acme import specs

import contrastive
import env_utils
from contrastive import nf_density as _nf
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from ppo_contrastive import fixed_goal_dict

# Hand offset behind puck (matches SawyerPush ψ construction).
_HAND_OFFSET = np.array([0.0, -0.08, 0.03], dtype=np.float32)
_GRIP_CLOSED = 0.0


def _infer_run_config(ckpt_path: str) -> dict:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    run_cfg = os.path.join(os.path.dirname(ckpt_dir), 'run_config.json')
    if not os.path.isfile(run_cfg):
        return {}
    try:
        with open(run_cfg, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        resolved = payload.get('resolved_config', {})
        return dict(resolved) if isinstance(resolved, dict) else {}
    except Exception:
        return {}


def _enumerate_checkpoints(path: str):
    if os.path.isfile(path):
        base = os.path.splitext(os.path.basename(path))[0]
        label = base[len('ckpt_'):] if base.startswith('ckpt_') else base
        return [(label, path)]
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Checkpoint path not found: {path}')
    iter_files = glob.glob(os.path.join(path, 'ckpt_iter_*.pkl'))

    def _it(fname):
        m = re.search(r'ckpt_iter_(\d+)\.pkl$', fname)
        return int(m.group(1)) if m else -1

    iter_files.sort(key=_it)
    entries = [(f'iter_{_it(f):07d}', f) for f in iter_files]
    latest = os.path.join(path, 'latest.pkl')
    if os.path.isfile(latest):
        entries.append(('latest', latest))
    return entries


def _set_scene_static(env, obj_xyz: np.ndarray, hand_xyz: np.ndarray) -> None:
    """Teleport puck and gripper to exact positions without running physics.

    Uses mocap weld + _set_obj_xyz + sim.forward() so nothing moves due to
    contact forces, giving pixel-accurate intermediate states along the trajectory.
    """
    obj = np.asarray(obj_xyz, dtype=np.float64)
    hand = np.asarray(hand_xyz, dtype=np.float64)
    # Teleport the gripper mocap body to exactly hand_xyz.
    env.data.set_mocap_pos('mocap', hand)
    env.data.set_mocap_quat('mocap', np.array([1.0, 0.0, 1.0, 0.0]))
    # Teleport the puck to exactly obj_xyz.
    env._set_obj_xyz(obj)
    # Propagate kinematics (no physics step) so rendering is consistent.
    env.sim.forward()


def _render(env, camera: str, width: int, height: int) -> np.ndarray:
    return np.asarray(
        env.render(offscreen=True, camera_name=camera, resolution=(width, height)),
        dtype=np.uint8)


def _psi_from_obj(obj_xyz: np.ndarray) -> np.ndarray:
    """Hindsight ψ: hand behind puck, gripper closed, puck at ``obj_xyz``."""
    obj = np.asarray(obj_xyz, dtype=np.float32).reshape(3)
    hand = obj + _HAND_OFFSET
    return np.concatenate([hand, [_GRIP_CLOSED], obj]).astype(np.float32)


def _state_from_obj(obj_xyz: np.ndarray) -> np.ndarray:
    """φ state matching the ideal push pose at ``obj_xyz``."""
    return _psi_from_obj(obj_xyz)


def _interp_obj(puck_init: np.ndarray, puck_target: np.ndarray,
                t: float) -> np.ndarray:
    return ((1.0 - t) * puck_init + t * puck_target).astype(np.float32)


def _trajectory_xy(puck_init: np.ndarray, puck_target: np.ndarray,
                   n: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([_interp_obj(puck_init, puck_target, t) for t in ts], axis=0)


def _grid_limits(puck_init: np.ndarray, puck_target: np.ndarray,
                 margin: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xs = [float(puck_init[0]), float(puck_target[0])]
    ys = [float(puck_init[1]), float(puck_target[1])]
    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)
    return xlim, ylim


def _is_legacy_nf_params(q_params: dict) -> bool:
    if any(k.startswith('sa_encoder/linear_') for k in q_params):
        return True
    sa = q_params.get('sa_encoder', {})
    return any(str(k).startswith('linear_') for k in sa)


def _normalize_nf_q_params(q_params: dict) -> dict:
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
    obs_dim: int, act_dim: int, goal_dim: int,
    hidden_layer_sizes: Sequence[int],
    rep_size: int, num_blocks: int, channels: int, l1_width: int,
):
    import haiku as hk
    from acme.jax import networks as networks_lib

    split_cond = (goal_dim + 1) // 2
    split_trans = goal_dim // 2
    w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
    zero_init = hk.initializers.Constant(0.)

    def _sa_fn(state, action):
        x = jnp.concatenate([state, action], axis=-1)
        for i, width in enumerate(hidden_layer_sizes):
            x = hk.Linear(width, w_init=w_init, name=f'linear_{i}')(x)
            if i < len(hidden_layer_sizes) - 1:
                x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                 name=f'ln_{i}')(x)
            x = jax.nn.swish(x)
        return hk.Linear(rep_size, w_init=w_init, name='sa_proj')(x)

    def _flow_log_prob(goal, y):
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
    return _nf.NFDensityNetworks(sa_encoder_net=sa_encoder_net, flow_net=flow_net)


def _build_nf_networks(env_name: str, seed: int, run_cfg: dict, q_params: dict):
    probe_env, obs_dim = contrastive_utils.make_environment(
        env_name, start_index=0, end_index=-1, seed=seed,
        fixed_start_end=fixed_goal_dict[env_name])
    env_spec = specs.make_environment_spec(probe_env)
    act_dim = int(np.prod(env_spec.actions.shape))
    del probe_env
    hidden = tuple(run_cfg.get('hidden_layer_sizes') or [256] * 6)
    rep_size = int(run_cfg.get('nf_rep_size', 64))
    num_blocks = int(run_cfg.get('nf_num_blocks', 8))
    channels = int(run_cfg.get('nf_coupling_width', 256))
    if _is_legacy_nf_params(q_params):
        l1_w = _infer_flow_l1_width(q_params, rep_size, channels)
        nf_nets = _make_legacy_nf_density_networks(
            obs_dim=obs_dim, act_dim=act_dim, goal_dim=obs_dim,
            hidden_layer_sizes=hidden, rep_size=rep_size,
            num_blocks=num_blocks, channels=channels, l1_width=l1_w)
    else:
        nf_nets = _nf.make_nf_density_networks(
            obs_dim=obs_dim, act_dim=act_dim, goal_dim=obs_dim,
            hidden_layer_sizes=hidden, rep_size=rep_size,
            num_blocks=num_blocks, channels=channels)
    return nf_nets, obs_dim, act_dim


def _build_policy_networks(env_name: str, seed: int, run_cfg: dict):
    probe_env, obs_dim = contrastive_utils.make_environment(
        env_name, start_index=0, end_index=-1, seed=seed,
        fixed_start_end=fixed_goal_dict[env_name])
    env_spec = specs.make_environment_spec(probe_env)
    del probe_env
    cfg = contrastive.ContrastiveConfig()
    hidden = run_cfg.get('hidden_layer_sizes') or cfg.hidden_layer_sizes
    networks = contrastive.make_networks(
        spec=env_spec,
        obs_dim=obs_dim,
        repr_dim=int(run_cfg.get('repr_dim', cfg.repr_dim)),
        repr_norm=bool(run_cfg.get('repr_norm', cfg.repr_norm)),
        twin_q=bool(run_cfg.get('twin_q', cfg.twin_q)),
        use_image_obs=bool(run_cfg.get('use_image_obs', cfg.use_image_obs)),
        hidden_layer_sizes=tuple(hidden),
        actor_min_std=float(run_cfg.get('ppo_actor_min_std', cfg.ppo_actor_min_std)),
    )
    return networks, obs_dim


def _goal_norm_stats(psi_target: np.ndarray, std_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate replay goal stats for fixed-target push (mean → ψ*, std → floor)."""
    mean = np.asarray(psi_target, dtype=np.float32)
    std = np.full_like(mean, float(std_min), dtype=np.float32)
    return mean, std


def _make_logp_fn(nf_networks, goal_mean: np.ndarray, goal_std: np.ndarray):
    gm = jnp.asarray(goal_mean)
    gs = jnp.asarray(goal_std)

    @jax.jit
    def logp_from_s0(nf_params, s0, a0, psi_batch):
        """log p(ψ_f | s₀, a₀) for a batch of hindsight goals ψ_f."""
        s0_b = jnp.broadcast_to(s0[None, :], (psi_batch.shape[0], s0.shape[0]))
        a0_b = jnp.broadcast_to(a0[None, :], (psi_batch.shape[0], a0.shape[0]))
        goal = (psi_batch - gm) / (gs + 1e-8)
        y = nf_networks.sa_encoder_net.apply(
            nf_params['sa_encoder'], s0_b, a0_b)
        if nf_networks.goal_encoder_net is not None:
            goal = nf_networks.goal_encoder_net.apply(
                nf_params['goal_encoder'], goal)
        return nf_networks.flow_net.apply(nf_params['nf_flow'], goal, y)

    return logp_from_s0


def _compute_xy_heatmap(
    logp_fn,
    nf_params,
    s0: np.ndarray,
    a0: np.ndarray,
    puck_init: np.ndarray,
    puck_target: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xy_grid: int,
    chunk: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.linspace(xlim[0], xlim[1], xy_grid, dtype=np.float32)
    y_vals = np.linspace(ylim[0], ylim[1], xy_grid, dtype=np.float32)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
    z = float(0.5 * (puck_init[2] + puck_target[2]))
    objs = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z, np.float32)], axis=-1)
    psi = np.stack([_psi_from_obj(o) for o in objs], axis=0)

    s0_j = jnp.asarray(s0.astype(np.float32))
    a0_j = jnp.asarray(a0.astype(np.float32))
    flat: List[np.ndarray] = []
    for start in range(0, len(psi), chunk):
        batch = jnp.asarray(psi[start:start + chunk])
        flat.append(np.asarray(logp_fn(nf_params, s0_j, a0_j, batch), dtype=np.float32))
    grid = np.concatenate(flat).reshape(len(y_vals), len(x_vals))
    return grid, x_vals, y_vals


def _line_profile(
    logp_fn, nf_params, s0, a0, traj_objs: np.ndarray, chunk: int = 256,
) -> np.ndarray:
    psi = np.stack([_psi_from_obj(o) for o in traj_objs], axis=0)
    s0_j = jnp.asarray(s0.astype(np.float32))
    a0_j = jnp.asarray(a0.astype(np.float32))
    vals: List[np.ndarray] = []
    for start in range(0, len(psi), chunk):
        batch = jnp.asarray(psi[start:start + chunk])
        vals.append(np.asarray(logp_fn(nf_params, s0_j, a0_j, batch), dtype=np.float32))
    return np.concatenate(vals)


def _draw_heatmap_ax(
    ax,
    grid: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    puck_init: np.ndarray,
    puck_target: np.ndarray,
    traj_objs: np.ndarray,
    vmin: float,
    vmax: float,
    title: str,
):
    im = ax.imshow(
        grid, origin='lower', aspect='auto',
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        cmap='plasma', vmin=vmin, vmax=vmax)
    ax.plot(traj_objs[:, 0], traj_objs[:, 1], 'w-', lw=1.5, alpha=0.9)
    ax.scatter([puck_init[0]], [puck_init[1]], c='lime', s=40,
               edgecolors='k', zorder=5, label='puck init')
    ax.scatter([puck_target[0]], [puck_target[1]], c='red', s=40,
               edgecolors='k', zorder=5, label='puck target')
    ax.set_xlabel('obj x (m)')
    ax.set_ylabel('obj y (m)')
    ax.set_title(title, fontsize=9)
    return im


def render_trajectory_preview(
    output: str,
    seed: int,
    n_along: int,
    width: int,
    height: int,
    camera: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render scene at each ideal pose along puck init→target.

    Uses static teleport (no physics) so the puck stays exactly where placed.
    Produces two camera rows (corner + topview) plus an xy schematic.
    """
    cams = [camera, 'topview']
    fixed = fixed_goal_dict['sawyer_push']
    env, _, _ = env_utils.load('sawyer_push', fixed, seed=seed)
    obs0 = env.reset()
    puck_init = obs0[4:7].copy()
    puck_target = np.asarray(fixed, dtype=np.float32)
    traj_objs = _trajectory_xy(puck_init, puck_target, n_along)

    # Render each trajectory state with all cameras (static placement)
    all_frames: List[List[np.ndarray]] = []   # [cam_idx][traj_idx]
    for _ in cams:
        all_frames.append([])

    ts = np.linspace(0.0, 1.0, n_along, dtype=np.float32)
    for i, (obj, t) in enumerate(zip(traj_objs, ts)):
        env.reset()
        hand = (obj + _HAND_OFFSET).astype(np.float64)
        _set_scene_static(env, obj, hand)
        for ci, cam in enumerate(cams):
            all_frames[ci].append(_render(env, cam, width, height))

    xlim, ylim = _grid_limits(puck_init, puck_target, margin=0.05)

    # Layout: top schematic + one render row per camera
    n_cam = len(cams)
    fig = plt.figure(figsize=(2.4 * n_along, 3.5 + 2.6 * n_cam),
                     facecolor='#1a1a2e')
    gs = fig.add_gridspec(
        1 + n_cam, n_along,
        height_ratios=[1.6] + [1.0] * n_cam,
        hspace=0.06, wspace=0.04,
        left=0.04, right=0.97, top=0.93, bottom=0.04)

    # ── xy schematic spanning all columns ────────────────────────────────
    ax_map = fig.add_subplot(gs[0, :])
    ax_map.set_facecolor('#12122a')
    ax_map.plot(traj_objs[:, 0], traj_objs[:, 1],
                '-', color='#4C9BE8', lw=2, zorder=2)
    ax_map.scatter(traj_objs[:, 0], traj_objs[:, 1],
                   c=ts, cmap='plasma', s=60, zorder=3,
                   edgecolors='white', linewidths=0.5)
    for i, (obj, t) in enumerate(zip(traj_objs, ts)):
        ax_map.annotate(
            f'{i}', (float(obj[0]), float(obj[1])),
            xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=7.5, color='white', fontweight='bold')
    ax_map.scatter([puck_init[0]], [puck_init[1]], c='#00ff88', s=100,
                   edgecolors='white', zorder=5, label='puck init')
    ax_map.scatter([puck_target[0]], [puck_target[1]], c='#ff4444', s=100,
                   edgecolors='white', zorder=5, label='puck target')
    # Draw gripper offset as arrows
    for i, obj in enumerate(traj_objs[::max(1, n_along // 5)]):
        hand = obj + _HAND_OFFSET
        ax_map.annotate('', xy=(float(obj[0]), float(obj[1])),
                        xytext=(float(hand[0]), float(hand[1])),
                        arrowprops=dict(arrowstyle='->', color='#aaaaff',
                                        lw=0.8, alpha=0.7))
    ax_map.set_xlim(xlim)
    ax_map.set_ylim(ylim)
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xlabel('puck x (m)', color='white', fontsize=9)
    ax_map.set_ylabel('puck y (m)', color='white', fontsize=9)
    ax_map.tick_params(colors='white', labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor('#444466')
    ax_map.set_title(
        'SawyerPush — ideal trajectory states  '
        '(puck on line, hand = puck + [0, −0.08, +0.03], gripper closed)',
        color='white', fontsize=10, pad=6)
    ax_map.legend(loc='upper left', fontsize=8,
                  facecolor='#22223a', labelcolor='white', edgecolor='#555577')

    # ── render rows ──────────────────────────────────────────────────────
    for ci, cam in enumerate(cams):
        for i, (img, t) in enumerate(zip(all_frames[ci], ts)):
            ax = fig.add_subplot(gs[1 + ci, i])
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_ylabel(cam, color='white', fontsize=8, rotation=90,
                              labelpad=2)
            # progress label beneath first camera row only
            if ci == 0:
                ax.set_title(f't={t:.2f}', color='white', fontsize=7, pad=2)
            # puck y label in last camera row
            if ci == n_cam - 1:
                obj = traj_objs[i]
                ax.set_xlabel(f'y={obj[1]:.3f}', color='#aaaacc',
                              fontsize=6.5, labelpad=1)

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f'[push_traj] saved trajectory preview → {output}', flush=True)
    return puck_init, puck_target, traj_objs, obs0


def run_heatmaps(
    checkpoint: str,
    output_dir: str,
    seed: int,
    xy_grid: int,
    margin: float,
    puck_init: np.ndarray | None,
    puck_target: np.ndarray | None,
    obs0: np.ndarray | None,
):
    ckpt_entries = _enumerate_checkpoints(checkpoint)
    if not ckpt_entries:
        raise FileNotFoundError(f'No checkpoints under {checkpoint}')

    run_cfg = _infer_run_config(ckpt_entries[0][1])
    std_min = float(run_cfg.get('nf_goal_std_min', 0.1))

    fixed = fixed_goal_dict['sawyer_push']
    env, _, _ = env_utils.load('sawyer_push', fixed, seed=seed)
    if obs0 is None:
        obs0 = env.reset()
    else:
        env.reset()
    if puck_init is None:
        puck_init = obs0[4:7].copy()
    if puck_target is None:
        puck_target = np.asarray(fixed, dtype=np.float32)

    obs_dim = 7
    s0 = np.asarray(obs0[:obs_dim], dtype=np.float32)
    psi_target = np.asarray(obs0[obs_dim:], dtype=np.float32)
    goal_mean, goal_std = _goal_norm_stats(psi_target, std_min)

    networks, _ = _build_policy_networks('sawyer_push', seed, run_cfg)
    q0 = ckpt_entries[0][1]
    ckpt0 = ppo_learner.load_checkpoint(q0)
    nf_networks, _, _ = _build_nf_networks('sawyer_push', seed, run_cfg, ckpt0['q_params'])

    @jax.jit
    def policy_mode(params, obs):
        dist = networks.policy_network.apply(params, obs)
        return networks.sample_eval(dist, jax.random.PRNGKey(0))

    obs_j = jnp.asarray(obs0[None].astype(np.float32))
    a0 = np.asarray(policy_mode(ckpt0['policy_params'], obs_j)[0], dtype=np.float32)

    xlim, ylim = _grid_limits(puck_init, puck_target, margin=margin)
    traj_objs = _trajectory_xy(puck_init, puck_target, n=21)
    logp_fn = _make_logp_fn(nf_networks, goal_mean, goal_std)

    os.makedirs(output_dir, exist_ok=True)
    grids = []
    labels = []
    line_profiles = []

    for label, path in ckpt_entries:
        ckpt = ppo_learner.load_checkpoint(path)
        nf_params = _normalize_nf_q_params(ckpt['q_params'])
        grid, x_vals, y_vals = _compute_xy_heatmap(
            logp_fn, nf_params, s0, a0, puck_init, puck_target,
            xlim, ylim, xy_grid)
        profile = _line_profile(logp_fn, nf_params, s0, a0, traj_objs)
        grids.append(grid)
        labels.append(label)
        line_profiles.append(profile)
        step = ckpt.get('global_step', '?')
        print(f'[push_traj] {label}: step={step}  '
              f'line logp [{profile.min():.2f}, {profile.max():.2f}]', flush=True)

    combined = np.concatenate([g.ravel() for g in grids])
    vmin = float(np.percentile(combined, 2))
    vmax = float(np.percentile(combined, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0

    n_ckpt = len(grids)
    fig, axes = plt.subplots(n_ckpt, 1, figsize=(8, 3.2 * n_ckpt), squeeze=False)
    for i, (grid, label) in enumerate(zip(grids, labels)):
        im = _draw_heatmap_ax(
            axes[i, 0], grid, x_vals, y_vals,
            puck_init, puck_target, traj_objs,
            vmin, vmax, f'{label}  log p(ψ_f | s₀, π(s₀))')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='log p')
    fig.suptitle(
        'Push NF: log p(hindsight ψ_f | reset s₀, a₀=π(s₀))\n'
        f'ψ_f: puck on grid, hand at obj+[0,−0.08,+0.03]  seed={seed}',
        fontsize=11)
    fig.tight_layout()
    hm_path = os.path.join(output_dir, 'push_traj_heatmaps_all_ckpts.png')
    fig.savefig(hm_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[push_traj] saved → {hm_path}', flush=True)

    # Line profile comparison
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ts = np.linspace(0, 1, len(traj_objs))
    for profile, label in zip(line_profiles, labels):
        ax2.plot(ts, profile, label=label, lw=1.5)
    ax2.set_xlabel('progress along puck init → target')
    ax2.set_ylabel('log p(ψ_f | s₀, π(s₀))')
    ax2.set_title('Log density along ideal push line (all checkpoints)')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    line_path = os.path.join(output_dir, 'push_traj_line_profile.png')
    fig2.tight_layout()
    fig2.savefig(line_path, dpi=130, bbox_inches='tight')
    plt.close(fig2)
    print(f'[push_traj] saved → {line_path}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=['preview', 'heatmaps', 'all'], default='all')
    ap.add_argument('--checkpoint', default='',
                    help='Checkpoint dir or .pkl (required for heatmaps/all)')
    ap.add_argument('--output', default='figs/push_nf_traj_hm',
                    help='Output directory')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_along', type=int, default=9,
                    help='Scene images along trajectory')
    ap.add_argument('--xy_grid', type=int, default=36)
    ap.add_argument('--margin', type=float, default=0.05,
                    help='Heatmap margin around puck init/target (m)')
    ap.add_argument('--width', type=int, default=480)
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--camera', default='corner')
    args = ap.parse_args()

    out_dir = args.output.rstrip('/')
    os.makedirs(out_dir, exist_ok=True)

    puck_init = puck_target = obs0 = None
    if args.mode in ('preview', 'all'):
        preview_path = os.path.join(
            out_dir, f'push_traj_preview_seed{args.seed}.png')
        puck_init, puck_target, _, obs0 = render_trajectory_preview(
            preview_path, args.seed, args.n_along,
            args.width, args.height, args.camera)

    if args.mode in ('heatmaps', 'all'):
        if not args.checkpoint:
            raise SystemExit('--checkpoint required for heatmaps mode')
        run_heatmaps(
            args.checkpoint, out_dir, args.seed,
            args.xy_grid, args.margin, puck_init, puck_target, obs0)


if __name__ == '__main__':
    main()
