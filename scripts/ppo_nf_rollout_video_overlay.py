"""NF PPO rollout videos with NF-density visualisation.

**sawyer_reach** (3-D tcp + 3-D goal): MuJoCo render on top, five fixed-z
xy heatmaps of ``log p(g|x,y,z)`` below (policy action at each grid cell).
Markers show init/start, current end-effector, goal, and trajectory end.

**sawyer_drawer_open**: same z-slice xy heatmaps, but over handle (x,y) with
hand/gripper held at the rollout state (init for the static grid).

Example::

  python scripts/ppo_nf_rollout_video_overlay.py \\
      --checkpoint=logs/ppo_nf_reach_debug/ppo_sawyer_reach_0/checkpoints \\
      --env=sawyer_reach \\
      --output=videos/ppo_nf_reach_debug_overlay_sawyer_reach_0/
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcrl_jax_acme_compat  # noqa: F401

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import numpy as np
from acme import specs

import haiku as hk
from acme.jax import networks as networks_lib

import contrastive
import env_utils
from contrastive import nf_density as _nf
from contrastive import ppo_learner
from contrastive import utils as contrastive_utils
from ppo_contrastive import fixed_goal_dict

_DEFAULT_CAMERA = {
    'sawyer_bin': 'corner',
    'sawyer_box': 'corner',
    'sawyer_peg': 'corner',
    'sawyer_reach': 'corner',
    'sawyer_drawer_open': 'corner',
}

# Reach xy heatmap workspace (m); z clamped to MW goal-space z range.
_REACH_XLIM = (-0.15, 0.15)
_REACH_YLIM = (0.75, 0.95)
_REACH_Z_CLAMP = (0.05, 0.30)

# Drawer handle workspace (m); z clamped to MW handle height range.
_DRAWER_XLIM = (-0.15, 0.15)
_DRAWER_YLIM = (0.42, 0.62)
_DRAWER_Z_CLAMP = (0.05, 0.20)


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
    """Read coupling MLP input width from checkpoint (often channels+rep_size)."""
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
    return _nf.NFDensityNetworks(sa_encoder_net=sa_encoder_net, flow_net=flow_net)


def _build_ppo_networks(env_name: str, seed: int, run_cfg: dict):
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
    return networks, int(obs_dim)


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
        print(f'[nf_video] legacy NF arch (MLP encoder, flow_l1={l1_w})',
              flush=True)
        nf_nets = _make_legacy_nf_density_networks(
            obs_dim=obs_dim, act_dim=act_dim, goal_dim=obs_dim,
            hidden_layer_sizes=hidden, rep_size=rep_size,
            num_blocks=num_blocks, channels=channels, l1_width=l1_w)
    else:
        nf_nets = _nf.make_nf_density_networks(
            obs_dim=obs_dim, act_dim=act_dim, goal_dim=obs_dim,
            hidden_layer_sizes=hidden, rep_size=rep_size,
            num_blocks=num_blocks, channels=channels,
            state_only=bool(run_cfg.get('nf_state_only', False)),
            scale_tanh=bool(run_cfg.get('nf_scale_tanh', False)),
            scale_tanh_c=float(run_cfg.get('nf_scale_tanh_c', 2.0)))
    return nf_nets, obs_dim


def _get_render_fn(gym_env, width, height, camera_name, rotate_deg=180):
    from mujoco_py import MjRenderContextOffscreen

    sim = gym_env.sim
    camera_id = sim.model.camera_name2id(camera_name)
    ctx = None
    for c in getattr(sim, 'render_contexts', []) or []:
        if getattr(c, 'offscreen', False):
            ctx = c
            break
    if ctx is None:
        ctx = MjRenderContextOffscreen(sim, device_id=-1)

    k_rot = int(round((rotate_deg % 360) / 90)) % 4

    def _render():
        ctx.render(width, height, camera_id)
        rgb = ctx.read_pixels(width, height, depth=False)
        frame = rgb[::-1]
        if k_rot:
            frame = np.rot90(frame, k=k_rot)
        return np.ascontiguousarray(frame)

    return _render, camera_id, k_rot


def _make_projector(sim, camera_id: int, width: int, height: int, k_rot: int):
    fovy_rad = float(sim.model.cam_fovy[camera_id]) * np.pi / 180.0
    aspect = width / float(height)
    scale_y = height / (2.0 * np.tan(fovy_rad / 2.0))
    scale_x = scale_y * aspect

    def _rot_screen(x: float, y: float) -> Tuple[int, int]:
        xi, yi = int(round(x)), int(round(y))
        if k_rot == 1:
            xi, yi = yi, width - 1 - xi
        elif k_rot == 2:
            xi, yi = width - 1 - xi, height - 1 - yi
        elif k_rot == 3:
            xi, yi = height - 1 - yi, xi
        return xi, yi

    def project(world_pt: np.ndarray) -> Optional[Tuple[int, int]]:
        cam_pos = sim.data.cam_xpos[camera_id].copy()
        cam_mat = sim.data.cam_xmat[camera_id].reshape(3, 3).copy()
        rel = np.asarray(world_pt, dtype=np.float64) - cam_pos
        cam = cam_mat.T @ rel
        if cam[2] >= -1e-5:
            return None
        x = width / 2.0 + scale_x * cam[0] / (-cam[2])
        y = height / 2.0 - scale_y * cam[1] / (-cam[2])
        xi, yi = _rot_screen(x, y)
        if 0 <= xi < width and 0 <= yi < height:
            return xi, yi
        return None

    return project


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


def _world_xyz(env_name: str, gym_env, obs: np.ndarray) -> np.ndarray:
    if env_name == 'sawyer_reach':
        return np.asarray(gym_env.tcp_center, dtype=np.float32)
    if env_name == 'sawyer_drawer_open':
        return np.asarray(obs[4:7], dtype=np.float32)
    return np.asarray(obs[:3], dtype=np.float32)


def _goal_world_xyz(env_name: str, obs: np.ndarray, obs_dim: int) -> np.ndarray:
    if env_name == 'sawyer_reach':
        return np.asarray(obs[obs_dim:obs_dim + 3], dtype=np.float32)
    return np.asarray(obs[obs_dim + 4:obs_dim + 7], dtype=np.float32)


def _state_from_world(
    env_name: str,
    base_obs: np.ndarray,
    obs_dim: int,
    world_xyz: np.ndarray,
    near_goal: bool,
) -> np.ndarray:
    state = np.asarray(base_obs[:obs_dim], dtype=np.float32).copy()
    if env_name == 'sawyer_reach':
        state[:3] = world_xyz
    elif near_goal:
        state[4:7] = world_xyz
    else:
        state[:3] = world_xyz
    return state


def _sample_offsets(span: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros((1, 3), dtype=np.float32)
    vals = np.linspace(-span, span, n, dtype=np.float32)
    gx, gy, gz = np.meshgrid(vals, vals, vals, indexing='ij')
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def _reward_colour(r: float, vmin: float, vmax: float) -> Tuple[int, int, int]:
    if vmax <= vmin:
        t = 0.5
    else:
        t = float(np.clip((r - vmin) / (vmax - vmin), 0.0, 1.0))
    rgb = cm.plasma(t)[:3]
    return tuple(int(255 * c) for c in rgb)


def _draw_overlay(
    frame: np.ndarray,
    points: Sequence[Tuple[np.ndarray, float, str, int]],
    project,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Draw projected reward dots.  point = (world_xyz, reward, label, radius)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for world_pt, rew, label, dot_r in points:
        xy = project(world_pt)
        if xy is None:
            continue
        colour = _reward_colour(rew, vmin, vmax)
        x, y = xy
        draw.ellipse((x - dot_r, y - dot_r, x + dot_r, y + dot_r),
                     fill=colour, outline=(0, 0, 0))
        if label and font is not None:
            draw.text((x + dot_r + 2, y - 4), label, fill=(255, 255, 255),
                      font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def _draw_colorbar(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    h, w = frame.shape[:2]
    bar_w, bar_h = 18, min(140, h - 20)
    x0, y0 = w - bar_w - 8, 10
    for i in range(bar_h):
        t = 1.0 - i / max(bar_h - 1, 1)
        colour = _reward_colour(vmin + t * (vmax - vmin), vmin, vmax)
        draw.line((x0, y0 + i, x0 + bar_w, y0 + i), fill=colour)
    draw.rectangle((x0 - 1, y0 - 1, x0 + bar_w + 1, y0 + bar_h + 1),
                   outline=(255, 255, 255))
    if font is not None:
        draw.text((x0 - 2, y0), f'{vmax:.1f}', fill=(255, 255, 255), font=font)
        draw.text((x0 - 2, y0 + bar_h - 10), f'{vmin:.1f}',
                  fill=(255, 255, 255), font=font)
        draw.text((x0 - 70, y0 + bar_h + 2), 'log p(g|s,a)',
                  fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _pick_z_slices(
    init_z: float,
    goal_z: float,
    n: int,
    z_clamp: Tuple[float, float],
) -> np.ndarray:
    lo = max(z_clamp[0], min(init_z, goal_z) - 0.04)
    hi = min(z_clamp[1], max(init_z, goal_z) + 0.04)
    if hi - lo < 0.05:
        mid = 0.5 * (init_z + goal_z)
        lo = max(z_clamp[0], mid - 0.08)
        hi = min(z_clamp[1], mid + 0.08)
    return np.linspace(lo, hi, n, dtype=np.float32)


def _xy_world_to_panel_px(
    x: float,
    y: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    panel_w: int,
    panel_h: int,
) -> Tuple[int, int]:
    """Map world xy to image pixels (origin lower-left in data space)."""
    px = int(round((x - xlim[0]) / max(xlim[1] - xlim[0], 1e-6) * (panel_w - 1)))
    py = int(round((y - ylim[0]) / max(ylim[1] - ylim[0], 1e-6) * (panel_h - 1)))
    py = panel_h - 1 - py
    return px, py


def _grid_to_rgb(grid: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        t = np.full(grid.shape, 0.5, dtype=np.float32)
    else:
        t = np.clip((grid - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cm.plasma(t)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _compute_reach_z_slice_grids(
    policy_params,
    reward_params,
    goal_tail: np.ndarray,
    z_values: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xy_grid: int,
    policy_mode,
    batch_reward,
    chunk: int = 512,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float, float]:
    x_vals = np.linspace(xlim[0], xlim[1], xy_grid, dtype=np.float32)
    y_vals = np.linspace(ylim[0], ylim[1], xy_grid, dtype=np.float32)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
    goal_broadcast = np.broadcast_to(
        goal_tail[None, :], (xx.size, goal_tail.shape[0])).copy()

    grids: List[np.ndarray] = []
    all_vals: List[np.ndarray] = []
    for z in z_values:
        zz = np.full(xx.shape, z, dtype=np.float32)
        states = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        obs_batch = np.concatenate([states, goal_broadcast], axis=-1).astype(np.float32)
        rews: List[np.ndarray] = []
        for start in range(0, len(obs_batch), chunk):
            batch = obs_batch[start:start + chunk]
            obs_j = jnp.asarray(batch)
            acts = policy_mode(policy_params, obs_j)
            rews.append(np.asarray(
                batch_reward(reward_params, obs_j, acts), dtype=np.float32))
        flat = np.concatenate(rews)
        grids.append(flat.reshape(len(y_vals), len(x_vals)))
        all_vals.append(flat)

    combined = np.concatenate(all_vals)
    vmin = float(np.percentile(combined, 2))
    vmax = float(np.percentile(combined, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return grids, x_vals, y_vals, vmin, vmax


def _compute_drawer_z_slice_grids(
    policy_params,
    reward_params,
    base_state: np.ndarray,
    goal_tail: np.ndarray,
    z_values: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xy_grid: int,
    policy_mode,
    batch_reward,
    chunk: int = 512,
) -> Tuple[List[np.ndarray], float, float]:
    """Heatmaps over handle (x,y); hand+gripper fixed at ``base_state``."""
    x_vals = np.linspace(xlim[0], xlim[1], xy_grid, dtype=np.float32)
    y_vals = np.linspace(ylim[0], ylim[1], xy_grid, dtype=np.float32)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='xy')
    hand_grip = np.asarray(base_state[:4], dtype=np.float32)
    goal_broadcast = np.broadcast_to(
        goal_tail[None, :], (xx.size, goal_tail.shape[0])).copy()

    grids: List[np.ndarray] = []
    all_vals: List[np.ndarray] = []
    for z in z_values:
        zz = np.full(xx.shape, z, dtype=np.float32)
        handle = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        hand_grip_tile = np.broadcast_to(
            hand_grip[None, :], (xx.size, 4)).copy()
        states = np.concatenate([hand_grip_tile, handle], axis=-1)
        obs_batch = np.concatenate([states, goal_broadcast], axis=-1).astype(np.float32)
        rews: List[np.ndarray] = []
        for start in range(0, len(obs_batch), chunk):
            batch = obs_batch[start:start + chunk]
            obs_j = jnp.asarray(batch)
            acts = policy_mode(policy_params, obs_j)
            rews.append(np.asarray(
                batch_reward(reward_params, obs_j, acts), dtype=np.float32))
        flat = np.concatenate(rews)
        grids.append(flat.reshape(len(y_vals), len(x_vals)))
        all_vals.append(flat)

    combined = np.concatenate(all_vals)
    vmin = float(np.percentile(combined, 2))
    vmax = float(np.percentile(combined, 98))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return grids, vmin, vmax


def _draw_xy_slice_panel(
    grid: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    vmin: float,
    vmax: float,
    panel_w: int,
    panel_h: int,
    z_val: float,
    traj_world: Sequence[np.ndarray],
    traj_idx: int,
    goal_xy: np.ndarray,
    at_end: bool,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    rgb_small = _grid_to_rgb(grid, vmin, vmax)
    img = Image.fromarray(rgb_small).resize((panel_w, panel_h), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    traj_so_far = traj_world[:traj_idx + 1]
    if len(traj_so_far) >= 2:
        pts = [
            _xy_world_to_panel_px(p[0], p[1], xlim, ylim, panel_w, panel_h)
            for p in traj_so_far
        ]
        draw.line(pts, fill=(220, 220, 220), width=1)

    init = traj_world[0]
    ix, iy = _xy_world_to_panel_px(
        float(init[0]), float(init[1]), xlim, ylim, panel_w, panel_h)
    draw.ellipse((ix - 4, iy - 4, ix + 4, iy + 4),
                 fill=(0, 210, 0), outline=(0, 0, 0))

    gx, gy = _xy_world_to_panel_px(
        float(goal_xy[0]), float(goal_xy[1]), xlim, ylim, panel_w, panel_h)
    draw.ellipse((gx - 4, gy - 4, gx + 4, gy + 4),
                 fill=(220, 40, 40), outline=(0, 0, 0))

    cur = traj_world[traj_idx]
    cx, cy = _xy_world_to_panel_px(
        float(cur[0]), float(cur[1]), xlim, ylim, panel_w, panel_h)
    if at_end and traj_idx == len(traj_world) - 1:
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5),
                     fill=(60, 120, 255), outline=(0, 0, 0))
    elif traj_idx > 0:
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3),
                     fill=(0, 220, 220), outline=(0, 0, 0))

    try:
        font = ImageFont.load_default()
        draw.text((3, 2), f'{z_val:.2f}', fill=(255, 255, 255), font=font,
                  stroke_width=1, stroke_fill=(0, 0, 0))
    except Exception:
        pass
    return np.asarray(img)


def _draw_slice_colorbar(
    strip: np.ndarray,
    vmin: float,
    vmax: float,
    bar_w: int = 14,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(strip)
    draw = ImageDraw.Draw(img)
    h, w = strip.shape[:2]
    bar_h = h - 8
    x0, y0 = w - bar_w - 4, 4
    for i in range(bar_h):
        t = 1.0 - i / max(bar_h - 1, 1)
        colour = _reward_colour(vmin + t * (vmax - vmin), vmin, vmax)
        draw.line((x0, y0 + i, x0 + bar_w, y0 + i), fill=colour)
    draw.rectangle((x0 - 1, y0 - 1, x0 + bar_w + 1, y0 + bar_h + 1),
                   outline=(200, 200, 200))
    return np.asarray(img)


def _compose_slice_frame(
    render_frame: np.ndarray,
    panels: Sequence[np.ndarray],
    vmin: float,
    vmax: float,
    hud_text: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    render_h, render_w = render_frame.shape[:2]
    n_panels = len(panels)
    slice_h = panels[0].shape[0]
    slice_w = render_w // n_panels
    out_h = render_h + slice_h
    out = np.zeros((out_h, render_w, 3), dtype=np.uint8)
    out[:render_h] = render_frame

    for i, panel in enumerate(panels):
        if panel.shape[0] != slice_h or panel.shape[1] != slice_w:
            panel = np.asarray(
                Image.fromarray(panel).resize((slice_w, slice_h), Image.NEAREST))
        x0 = i * slice_w
        out[render_h:, x0:x0 + slice_w] = panel

    out[render_h:] = _draw_slice_colorbar(out[render_h:], vmin, vmax)

    hud = Image.fromarray(out)
    hud_draw = ImageDraw.Draw(hud)
    try:
        hud_font = ImageFont.load_default()
        hud_draw.text((8, 8), hud_text, fill=(255, 255, 255), font=hud_font,
                      stroke_width=1, stroke_fill=(0, 0, 0))
    except Exception:
        pass
    return np.asarray(hud)


def _rollout_xy_slice_video(
    gym_env,
    env_name: str,
    obs_dim: int,
    policy_params,
    reward_params,
    networks,
    reward_fn,
    render,
    max_steps: int,
    stochastic: bool,
    seed: int,
    num_z_slices: int,
    xy_grid: int,
    slice_row_h: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    z_clamp: Tuple[float, float],
    repr_tag: str = 'NF',
    grid_builder=None,
) -> Tuple[List[np.ndarray], dict]:
    @jax.jit
    def policy_mode(params, obs):
        dist = networks.policy_network.apply(params, obs)
        return networks.sample_eval(dist, jax.random.PRNGKey(0))

    @jax.jit
    def policy_sample(params, obs, rng):
        dist = networks.policy_network.apply(params, obs)
        return networks.sample(dist, rng)

    @jax.jit
    def batch_reward(rp, obs_batch, act_batch):
        return reward_fn(rp, obs_batch, act_batch)

    obs = np.asarray(gym_env.reset(), dtype=np.float32)
    goal_tail = obs[obs_dim:].copy()
    goal_world = _goal_world_xyz(env_name, obs, obs_dim)
    init_world = _world_xyz(env_name, gym_env, obs)
    base_state = np.asarray(obs[:obs_dim], dtype=np.float32)

    z_values = _pick_z_slices(
        float(init_world[2]), float(goal_world[2]), num_z_slices, z_clamp)
    print(f'[{repr_tag}_video] {env_name} z-slices: {z_values}', flush=True)
    if grid_builder is None:
        grids, _, _, vmin, vmax = _compute_reach_z_slice_grids(
            policy_params, reward_params, goal_tail, z_values,
            xlim, ylim, xy_grid, policy_mode, batch_reward)
    else:
        grids, vmin, vmax = grid_builder(
            policy_params, reward_params, base_state, goal_tail, z_values,
            xlim, ylim, xy_grid, policy_mode, batch_reward)

    render_w = None
    traj_world: List[np.ndarray] = []
    traj_rew: List[float] = []
    frames: List[np.ndarray] = []
    total_env_reward = 0.0
    success = False
    rng = jax.random.PRNGKey(seed)
    done = False

    for t in range(max_steps + 1):
        world = _world_xyz(env_name, gym_env, obs)
        obs_j = jnp.asarray(obs[None])
        if stochastic:
            rng, k = jax.random.split(rng)
            act_j = policy_sample(policy_params, obs_j, k)
        else:
            act_j = policy_mode(policy_params, obs_j)
        repr_r = float(np.asarray(batch_reward(reward_params, obs_j, act_j))[0])
        traj_world.append(world.copy())
        traj_rew.append(repr_r)

        base = render()
        if render_w is None:
            render_w = base.shape[1]
        panel_w = max(1, render_w // num_z_slices)
        panels = [
            _draw_xy_slice_panel(
                grid, xlim, ylim, vmin, vmax,
                panel_w, slice_row_h, float(z_val),
                traj_world, t, goal_world[:2],
                at_end=(t >= max_steps),
            )
            for grid, z_val in zip(grids, z_values)
        ]
        hud = f'step={t}  {repr_tag} r={repr_r:.2f}'
        frames.append(_compose_slice_frame(base, panels, vmin, vmax, hud))

        if t >= max_steps:
            break

        action = np.asarray(act_j)[0].astype(np.float32)
        obs_next, r, done, _ = gym_env.step(action)
        obs = np.asarray(obs_next, dtype=np.float32)
        total_env_reward += float(r)
        if float(r) > 0.0:
            success = True

    return frames, dict(
        total_reward=total_env_reward,
        success=success,
        length=len(frames),
        repr_reward_final=traj_rew[-1] if traj_rew else float('nan'),
    )


def _rollout_with_overlay(
    env_name: str,
    gym_env,
    obs_dim: int,
    policy_params,
    nf_params,
    networks,
    nf_networks,
    nf_reward_fn,
    render,
    project,
    max_steps: int,
    stochastic: bool,
    seed: int,
    sample_span: float,
    sample_grid: int,
    label_samples: bool,
) -> Tuple[List[np.ndarray], dict]:
    @jax.jit
    def policy_mode(params, obs):
        dist = networks.policy_network.apply(params, obs)
        return networks.sample_eval(dist, jax.random.PRNGKey(0))

    @jax.jit
    def policy_sample(params, obs, rng):
        dist = networks.policy_network.apply(params, obs)
        return networks.sample(dist, rng)

    @jax.jit
    def batch_nf_reward(nf_p, obs_batch, act_batch):
        return nf_reward_fn(nf_p, obs_batch, act_batch)

    obs = np.asarray(gym_env.reset(), dtype=np.float32)
    goal_tail = obs[obs_dim:].copy()
    goal_world = _goal_world_xyz(env_name, obs, obs_dim)

    traj_world: List[np.ndarray] = []
    traj_rew: List[float] = []
    all_rewards: List[float] = []
    frames: List[np.ndarray] = []
    total_env_reward = 0.0
    success = False
    rng = jax.random.PRNGKey(seed)
    offsets = _sample_offsets(sample_span, sample_grid)

    def _eval_points(state_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_batch = np.concatenate(
            [state_rows, np.broadcast_to(goal_tail[None, :], (len(state_rows), goal_tail.shape[0]))],
            axis=-1).astype(np.float32)
        obs_j = jnp.asarray(obs_batch)
        acts = policy_mode(policy_params, obs_j)
        rews = np.asarray(batch_nf_reward(nf_params, obs_j, acts), dtype=np.float32)
        return obs_batch, rews

    for t in range(max_steps + 1):
        world = _world_xyz(env_name, gym_env, obs)
        obs_j = jnp.asarray(obs[None])
        if stochastic:
            rng, k = jax.random.split(rng)
            act_j = policy_sample(policy_params, obs_j, k)
        else:
            act_j = policy_mode(policy_params, obs_j)
        nf_r = float(np.asarray(batch_nf_reward(nf_params, obs_j, act_j))[0])
        traj_world.append(world.copy())
        traj_rew.append(nf_r)
        all_rewards.append(nf_r)

        # Sample cloud near current EE / handle.
        near_states = np.stack([
            _state_from_world(env_name, obs, obs_dim, world + off, near_goal=False)
            for off in offsets
        ], axis=0)
        _, near_rews = _eval_points(near_states)

        # Sample cloud near goal.
        goal_states = np.stack([
            _state_from_world(env_name, obs, obs_dim, goal_world + off, near_goal=True)
            for off in offsets
        ], axis=0)
        _, goal_rews = _eval_points(goal_states)
        all_rewards.extend(near_rews.tolist())
        all_rewards.extend(goal_rews.tolist())

        vmin = float(np.percentile(all_rewards, 5))
        vmax = float(np.percentile(all_rewards, 95))
        if vmax <= vmin:
            vmax = vmin + 1.0

        base = render()
        overlay_pts = []
        for i, (wpt, rew) in enumerate(zip(traj_world, traj_rew)):
            label = f'{rew:.1f}' if (i == len(traj_world) - 1 or i % 8 == 0) else ''
            radius = 5 if i == len(traj_world) - 1 else 3
            overlay_pts.append((wpt, rew, label, radius))

        for off, rew in zip(offsets, near_rews):
            wpt = world + off
            label = f'{rew:.1f}' if label_samples else ''
            overlay_pts.append((wpt, float(rew), label, 2))

        for off, rew in zip(offsets, goal_rews):
            wpt = goal_world + off
            label = f'{rew:.1f}' if label_samples else ''
            overlay_pts.append((wpt, float(rew), label, 2))

        frame = _draw_overlay(base, overlay_pts, project, vmin, vmax)
        frame = _draw_colorbar(frame, vmin, vmax)
        from PIL import Image, ImageDraw, ImageFont
        hud = Image.fromarray(frame)
        hud_draw = ImageDraw.Draw(hud)
        try:
            hud_font = ImageFont.load_default()
        except Exception:
            hud_font = None
        hud_text = f'step={t}  NF r={nf_r:.2f}'
        if hud_font is not None:
            hud_draw.text((8, 8), hud_text, fill=(255, 255, 255), font=hud_font,
                          stroke_width=1, stroke_fill=(0, 0, 0))
        frames.append(np.asarray(hud))

        if t >= max_steps:
            break

        action = np.asarray(act_j)[0].astype(np.float32)
        obs_next, r, done, _ = gym_env.step(action)
        obs = np.asarray(obs_next, dtype=np.float32)
        total_env_reward += float(r)
        if float(r) > 0.0:
            success = True
        if done:
            break

    return frames, dict(
        total_reward=total_env_reward,
        success=success,
        length=len(frames),
        nf_reward_final=traj_rew[-1] if traj_rew else float('nan'),
    )


def _write_video(frames, path: str, fps: int):
    import imageio.v2 as imageio
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)


def _resolve_output_path(
    output_arg: str,
    env: str,
    label: str,
    multi: bool,
    suffix: str = 'nf_overlay',
) -> str:
    is_dir_like = output_arg.endswith(os.sep) or os.path.isdir(output_arg)
    if is_dir_like:
        return os.path.join(output_arg, f'{env}_{label}_{suffix}.mp4')
    if not multi:
        return output_arg
    stem, ext = os.path.splitext(output_arg)
    return f'{stem}_{label}_{suffix}{ext or ".mp4"}'


def _build_crl_config(run_cfg: dict, obs_dim: int) -> contrastive.ContrastiveConfig:
    cfg = contrastive.ContrastiveConfig()
    cfg.obs_dim = int(obs_dim)
    if run_cfg.get('repr_dim') is not None:
        cfg.repr_dim = int(run_cfg['repr_dim'])
    if run_cfg.get('repr_norm') is not None:
        cfg.repr_norm = bool(run_cfg['repr_norm'])
    if run_cfg.get('twin_q') is not None:
        cfg.twin_q = bool(run_cfg['twin_q'])
    mode = run_cfg.get('ppo_reward_mode')
    if mode is not None:
        cfg.ppo_reward_mode = str(mode)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--env', default='sawyer_reach',
                    choices=['sawyer_reach', 'sawyer_drawer_open'])
    ap.add_argument('--output', required=True)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--camera', default=None)
    ap.add_argument('--rotate', type=int, default=180)
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--max_steps', type=int, default=-1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sample_span', type=float, default=0.06,
                    help='(drawer) Half-width (m) of local / goal sample cubes.')
    ap.add_argument('--sample_grid', type=int, default=3,
                    help='(drawer) Samples per axis in each cube (3 -> 27 points).')
    ap.add_argument('--label_samples', action='store_true',
                    help='(drawer) Print reward text on sample dots.')
    ap.add_argument('--num_z_slices', type=int, default=5,
                    help='(reach) Number of fixed-z xy heatmap panels.')
    ap.add_argument('--xy_grid', type=int, default=28,
                    help='(reach) Grid resolution per axis for xy heatmaps.')
    ap.add_argument('--slice_row_h', type=int, default=160,
                    help='(reach) Height (px) of the heatmap strip below MuJoCo.')
    ap.add_argument('--repr_mode', choices=['nf', 'crl'], default='nf',
                    help='Reach heatmaps: NF log p(g|s,a) or CRL φ·ψ.')
    args = ap.parse_args()

    tag = args.repr_mode.upper()
    out_suffix = f'{args.repr_mode}_overlay'

    ckpt_entries = _enumerate_checkpoints(args.checkpoint)
    if not ckpt_entries:
        print(f'[{tag}_video] no checkpoints found')
        return
    print(f'[{tag}_video] {len(ckpt_entries)} checkpoint(s)')

    run_cfg = _infer_run_config(ckpt_entries[0][1])
    networks, obs_dim = _build_ppo_networks(args.env, args.seed, run_cfg)

    gym_env, _, env_max_steps = env_utils.load(
        args.env, fixed_start_end=fixed_goal_dict[args.env], seed=args.seed)
    max_steps = env_max_steps if args.max_steps < 0 else int(args.max_steps)
    camera = args.camera or _DEFAULT_CAMERA.get(args.env, 'corner')
    render, camera_id, k_rot = _get_render_fn(
        gym_env, args.width, args.height, camera, rotate_deg=args.rotate)
    multi = len(ckpt_entries) > 1
    for label, path in ckpt_entries:
        print(f'[{tag}_video] === {label} ===')
        ckpt = ppo_learner.load_checkpoint(path)
        ckpt_cfg = _infer_run_config(path) or run_cfg
        policy_params = ckpt['policy_params']
        q_raw = ckpt['q_params']

        if args.repr_mode == 'crl':
            if args.env != 'sawyer_reach':
                raise ValueError('repr_mode=crl is only supported for sawyer_reach')
            crl_cfg = _build_crl_config(ckpt_cfg, obs_dim)
            reward_params = q_raw
            reward_fn = ppo_learner.make_reward_fn(networks, crl_cfg)
            nf_networks, nf_params, nf_reward_fn = None, None, None
            reach_repr_tag = 'CRL'
        else:
            nf_networks, nf_obs_dim = _build_nf_networks(
                args.env, args.seed, ckpt_cfg, q_raw)
            nf_params = _normalize_nf_q_params(q_raw)
            assert nf_obs_dim == obs_dim
            reward_params = nf_params
            reward_fn = _nf.make_nf_reward_fn(nf_networks, obs_dim=obs_dim)
            reach_repr_tag = 'NF'

        if args.env == 'sawyer_reach':
            slice_kwargs = dict(
                xlim=_REACH_XLIM, ylim=_REACH_YLIM, z_clamp=_REACH_Z_CLAMP,
                grid_builder=None,
            )
        elif args.env == 'sawyer_drawer_open':
            if args.repr_mode != 'nf':
                raise ValueError('repr_mode=crl is only supported for sawyer_reach')
            slice_kwargs = dict(
                xlim=_DRAWER_XLIM, ylim=_DRAWER_YLIM, z_clamp=_DRAWER_Z_CLAMP,
                grid_builder=_compute_drawer_z_slice_grids,
            )
        else:
            raise ValueError(f'unsupported env {args.env}')

        frames, stats = _rollout_xy_slice_video(
            gym_env=gym_env,
            env_name=args.env,
            obs_dim=obs_dim,
            policy_params=policy_params,
            reward_params=reward_params,
            networks=networks,
            reward_fn=reward_fn,
            render=render,
            max_steps=max_steps,
            stochastic=args.stochastic,
            seed=args.seed,
            num_z_slices=args.num_z_slices,
            xy_grid=args.xy_grid,
            slice_row_h=args.slice_row_h,
            repr_tag=reach_repr_tag,
            **slice_kwargs,
        )
        print(f'[{tag}_video]   len={stats["length"]}  env_reward={stats["total_reward"]:.3f}  '
              f'success={stats["success"]}  repr_final={stats["repr_reward_final"]:.3f}')

        out_path = _resolve_output_path(
            args.output, args.env, label, multi, suffix=out_suffix)
        _write_video(frames, out_path, args.fps)
        print(f'[{tag}_video]   wrote {out_path}')


if __name__ == '__main__':
    main()
