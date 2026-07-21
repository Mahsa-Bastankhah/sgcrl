"""Visualise one PPO rollout on the figureeight0 benchmark.

Produces a two-panel animation:
  - Left:  top-down view of the figure-eight road.  Each vehicle is a
           coloured disk; colour encodes current speed (blue=0, red=max).
           The RL vehicle is shown with a white border.  A colour-bar on
           the right shows the speed scale; a horizontal dashed line marks
           the target velocity.
  - Right: real-time speed traces for all 14 vehicles.  Human vehicles
           are thin blue lines; the RL vehicle is a thick red line.  A
           dashed line at target_velocity = 20 m/s shows the goal.

Usage:
  python flow_figureeight_video.py \
      --checkpoint logs/ppo_flow_test2/ppo_flow_figureeight_0/checkpoints/latest.pkl \
      --steps 300 --output videos/figureeight/figureeight_ppo.gif

  # No checkpoint → random (untrained) policy
  python flow_figureeight_video.py --steps 1500 --output videos/figureeight/figureeight_random_full.mp4

  # Full episode with iter-0 checkpoint (if saved):
  python flow_figureeight_video.py \\
      --checkpoint logs/ppo/ppo_flow_figureeight_0/checkpoints/ckpt_iter_0000000.pkl \\
      --steps 1500 --output videos/figureeight/figureeight_iter0_full.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless: no display needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# sgcrl imports ---------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sgcrl_jax_acme_compat  # noqa: F401
import jax
import jax.numpy as jnp
from acme import specs
import contrastive
from contrastive import ppo_learner as _ppo_learner
from contrastive import utils as contrastive_utils
from ppo_contrastive import fixed_goal_dict

# Flow env import (sets SUMO_HOME / LD_LIBRARY_PATH automatically) ------------
from flow_env import (FlowFigureEightEnv, FlowFigureEight7RL, FlowFigureEight14RL,
                      FlowFigureEight2V1RL, FlowFigureEight2V2RL,
                      FlowFigureEight4V2RL, FlowFigureEight8V4RL)

TARGET_VELOCITY = FlowFigureEightEnv.TARGET_VELOCITY  # 20 m/s
MAX_SPEED       = FlowFigureEightEnv.MAX_SPEED        # 30 m/s
N_VEH           = FlowFigureEightEnv.N_VEHICLES       # 14

_ENV_CLASSES = {
    'flow_figureeight':        FlowFigureEightEnv,
    'flow_figureeight_7rl':    FlowFigureEight7RL,
    'flow_figureeight_14rl':   FlowFigureEight14RL,
    'flow_figureeight_2v1rl':  FlowFigureEight2V1RL,
    'flow_figureeight_2v2rl':  FlowFigureEight2V2RL,
    'flow_figureeight_4v2rl':  FlowFigureEight4V2RL,
    'flow_figureeight_8v4rl':  FlowFigureEight8V4RL,
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _get_lane_polys(flow_env):
    """Return list of (x,y) polyline arrays for all lanes from the SUMO kernel."""
    polys = []
    for lane_id in flow_env.k.kernel_api.lane.getIDList():
        pts = flow_env.k.kernel_api.lane.getShape(lane_id)
        if pts:
            xs, ys = zip(*pts)
            polys.append((np.array(xs), np.array(ys)))
    return polys


def _get_vehicle_xy(flow_env, veh_id):
    """Return (x, y) from SUMO for a given vehicle id."""
    try:
        pos = flow_env.k.kernel_api.vehicle.getPosition(veh_id)
        return pos[0], pos[1]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Network / policy builders
# ---------------------------------------------------------------------------

_END_INDEX = {
    'flow_figureeight':       14,
    'flow_figureeight_7rl':   14,
    'flow_figureeight_14rl':  14,
    'flow_figureeight_2v1rl':  2,
    'flow_figureeight_2v2rl':  2,
    'flow_figureeight_4v2rl':  4,
    'flow_figureeight_8v4rl':  8,
}

def _build_networks(env_name='flow_figureeight', seed=0):
    end_index = _END_INDEX.get(env_name, 14)
    probe_env, obs_dim = contrastive_utils.make_environment(
        env_name, start_index=0, end_index=end_index, seed=seed,
        fixed_start_end=fixed_goal_dict[env_name])
    env_spec = specs.make_environment_spec(probe_env)
    probe_env.close()

    cfg = contrastive.ContrastiveConfig()
    networks = contrastive.make_networks(
        spec=env_spec,
        obs_dim=obs_dim,
        repr_dim=cfg.repr_dim,
        repr_norm=cfg.repr_norm,
        twin_q=cfg.twin_q,
        use_image_obs=cfg.use_image_obs,
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        actor_min_std=float(cfg.ppo_actor_min_std),
    )
    return networks, obs_dim


# ---------------------------------------------------------------------------
# Rollout data collection
# ---------------------------------------------------------------------------

def collect_rollout(policy_params, networks, n_steps: int, seed: int = 0,
                    env_name: str = 'flow_figureeight'):
    """Run n_steps of the policy and return trajectory data.

    Returns
    -------
    dict with keys:
        speeds        : (T, N_VEH) actual speeds in m/s
        rl_speeds     : (T,)       mean RL vehicle speed in m/s each step
        actions       : (T,)       mean RL action in [-1, 1]
        actions_ms2   : (T,)       mean RL acceleration applied in m/s²
        rewards       : (T,)       sparse goal reward
        xy            : (T, N_VEH, 2)
        veh_ids, rl_indices, lane_polys, done_step
    """
    print(f'[video] collecting {n_steps}-step rollout (env={env_name})...')

    env_cls = _ENV_CLASSES.get(env_name, FlowFigureEightEnv)

    @jax.jit
    def _policy(params, obs, key):
        dist = networks.policy_network.apply(params, obs)
        return networks.sample(dist, key)

    from flow.benchmarks.figureeight0 import flow_params as _base_fp
    from flow.core.params import (InitialConfig, TrafficLightParams,
                                  VehicleParams, SumoCarFollowingParams)
    from flow.controllers import IDMController, ContinuousRouter, RLController

    fp = deepcopy(_base_fp)
    fp['sim'].render = False
    fp['sim'].port = None

    # Rebuild vehicle mix matching the chosen env class.
    n_rl    = env_cls.NUM_RL
    n_human = env_cls.N_VEHICLES - n_rl
    vehicles = VehicleParams()
    if n_human > 0:
        vehicles.add(
            veh_id='human',
            acceleration_controller=(IDMController, {'noise': 0.2}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode='obey_safe_speed', decel=1.5),
            num_vehicles=n_human)
    vehicles.add(
        veh_id='rl',
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode='obey_safe_speed'),
        num_vehicles=n_rl)
    fp['veh'] = vehicles

    env_class = fp['env_name']
    net_class = fp['network']
    network = net_class(
        name=fp['exp_tag'],
        vehicles=fp['veh'],
        net_params=fp['net'],
        initial_config=fp.get('initial', InitialConfig()),
        traffic_lights=fp.get('tls', TrafficLightParams()),
    )
    flow_env = env_class(
        env_params=fp['env'],
        sim_params=fp['sim'],
        network=network,
        simulator=fp['simulator'],
    )

    n_veh = env_cls.N_VEHICLES
    goal = np.full(env_cls.GOAL_OBS_DIM, TARGET_VELOCITY / MAX_SPEED, dtype=np.float32)

    def _make_obs(raw_obs):
        state = np.asarray(raw_obs[:env_cls.STATE_OBS_DIM], dtype=np.float32)
        return np.concatenate([state, goal])

    raw_obs = flow_env.reset()
    obs = _make_obs(raw_obs)

    veh_ids    = list(flow_env.k.vehicle.get_ids())
    rl_ids     = list(flow_env.k.vehicle.get_rl_ids())
    rl_indices = [veh_ids.index(rid) for rid in rl_ids if rid in veh_ids]
    if not rl_indices:
        rl_indices = [0]

    speeds_history    = []
    rl_speed_history  = []
    action_history    = []
    action_ms2_history = []
    reward_history    = []
    xy_history        = []
    done_step = n_steps
    key = jax.random.PRNGKey(seed)

    for t in range(n_steps):
        speeds_ms = np.array([
            flow_env.k.vehicle.get_speed(vid) for vid in veh_ids
        ], dtype=np.float32)
        xy_t = []
        for vid in veh_ids:
            pos = _get_vehicle_xy(flow_env, vid)
            xy_t.append(pos if pos is not None else (0.0, 0.0))

        speeds_history.append(speeds_ms)
        rl_speed_history.append(float(speeds_ms[rl_indices].mean()))
        xy_history.append(xy_t)

        key, subkey = jax.random.split(key)
        action_j  = _policy(policy_params, obs[None], subkey)
        action_np = np.asarray(action_j)[0].astype(np.float32)  # shape (n_rl,)
        flow_action = action_np * FlowFigureEightEnv.MAX_ACCEL

        raw_obs_next, flow_reward, done, _ = flow_env.step(flow_action)
        obs = _make_obs(raw_obs_next)

        reward = float(np.all(np.abs(speeds_ms - TARGET_VELOCITY) < 1.0))
        action_history.append(float(np.mean(action_np)))
        action_ms2_history.append(float(np.mean(flow_action)))
        reward_history.append(reward)

        if done:
            done_step = t + 1
            print(f'[video] episode ended at step {done_step} (crash or horizon)')
            break

    lane_polys = _get_lane_polys(flow_env)
    flow_env.terminate()

    return {
        'speeds':      np.array(speeds_history),
        'rl_speeds':   np.array(rl_speed_history),
        'actions':     np.array(action_history),
        'actions_ms2': np.array(action_ms2_history),
        'rewards':     np.array(reward_history),
        'xy':          np.array(xy_history),
        'veh_ids':     veh_ids,
        'rl_indices':  rl_indices,
        'lane_polys':  lane_polys,
        'done_step':   done_step,
    }


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def build_animation(data: dict, fps: int = 10):
    speeds      = data['speeds']
    rl_speeds   = data['rl_speeds']
    xy          = data['xy']
    actions     = data['actions']
    actions_ms2 = data['actions_ms2']
    rl_indices  = data.get('rl_indices', [data.get('rl_idx', 0)])
    if isinstance(rl_indices, (int, np.integer)):
        rl_indices = [int(rl_indices)]
    polys       = data['lane_polys']
    T, N    = speeds.shape
    max_accel = FlowFigureEightEnv.MAX_ACCEL

    cmap  = cm.RdYlGn_r

    fig = plt.figure(figsize=(15, 9), facecolor='#1a1a2e')
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.2, 1.0],
                           wspace=0.10, hspace=0.28)
    ax_net   = fig.add_subplot(gs[0, 0])
    ax_rlspd = fig.add_subplot(gs[0, 1])
    ax_rlacc = fig.add_subplot(gs[1, 1])
    ax_all   = fig.add_subplot(gs[1, 0])

    for ax in (ax_net, ax_rlspd, ax_rlacc, ax_all):
        ax.set_facecolor('#0f0f23')
        ax.tick_params(colors='#cccccc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # ---- network view ----
    n_rl_agents = len(rl_indices)
    n_total_veh = N   # actual vehicle count from speeds.shape
    ax_net.set_aspect('equal')
    ax_net.set_title(
        f'Figure-eight (white ring = RL agent, {n_rl_agents}/{n_total_veh} RL)',
        color='#eeeeff', fontsize=10)
    ax_net.set_xticks([]); ax_net.set_yticks([])
    for (xs, ys) in polys:
        ax_net.plot(xs, ys, color='#404060', lw=1.5, zorder=1)

    veh_xy0 = xy[0]
    scat = ax_net.scatter(
        veh_xy0[:, 0], veh_xy0[:, 1],
        c=speeds[0], cmap=cmap, vmin=0, vmax=MAX_SPEED,
        s=120, zorder=3, edgecolors='none')
    rl_ring = ax_net.scatter(
        veh_xy0[rl_indices, 0], veh_xy0[rl_indices, 1],
        s=220, zorder=4, facecolors='none', edgecolors='white', linewidths=2.0)

    sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=MAX_SPEED))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_net, fraction=0.046, pad=0.03)
    cb.set_label('speed (m/s)', color='#cccccc', fontsize=8)
    cb.ax.yaxis.set_tick_params(color='#cccccc')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#cccccc', fontsize=7)

    hud = ax_net.text(
        0.02, 0.98, '', transform=ax_net.transAxes,
        color='#ffe066', fontsize=9, va='top', ha='left', family='monospace')

    # ---- RL speed panel ----
    ax_rlspd.set_xlim(0, max(T - 1, 1))
    ax_rlspd.set_ylim(-0.5, MAX_SPEED + 1)
    ax_rlspd.set_xlabel('step', color='#cccccc', fontsize=9)
    ax_rlspd.set_ylabel('RL speed (m/s)', color='#cccccc', fontsize=9)
    ax_rlspd.set_title('RL vehicle speed', color='#eeeeff', fontsize=10)
    ax_rlspd.axhline(TARGET_VELOCITY, color='lime', lw=1.2, ls='--', alpha=0.7)
    rl_spd_ln, = ax_rlspd.plot([], [], color='tomato', lw=1.8)
    rl_spd_dot, = ax_rlspd.plot([], [], 'o', color='white', ms=5)

    # ---- RL action panel ----
    ax_rlacc.set_xlim(0, max(T - 1, 1))
    ax_rlacc.set_ylim(-max_accel - 0.5, max_accel + 0.5)
    ax_rlacc.set_xlabel('step', color='#cccccc', fontsize=9)
    ax_rlacc.set_ylabel('RL accel (m/s²)', color='#cccccc', fontsize=9)
    ax_rlacc.set_title('RL vehicle acceleration (policy × 3)', color='#eeeeff', fontsize=10)
    ax_rlacc.axhline(0.0, color='#888899', lw=0.8)
    rl_acc_ln, = ax_rlacc.plot([], [], color='#66ccff', lw=1.8)
    rl_acc_dot, = ax_rlacc.plot([], [], 'o', color='white', ms=5)
    ax_rlacc_twin = ax_rlacc.twinx()
    ax_rlacc_twin.set_ylim(-1.05, 1.05)
    ax_rlacc_twin.set_ylabel('action in [-1, 1]', color='#cc9966', fontsize=8)
    ax_rlacc_twin.tick_params(colors='#cc9966')
    act_ln, = ax_rlacc_twin.plot([], [], color='#cc9966', lw=1.0, alpha=0.7, ls=':')

    # ---- all vehicles (faint) ----
    ax_all.set_xlim(0, max(T - 1, 1))
    ax_all.set_ylim(-0.5, MAX_SPEED + 1)
    ax_all.set_xlabel('step', color='#cccccc', fontsize=9)
    ax_all.set_ylabel('speed (m/s)', color='#cccccc', fontsize=9)
    ax_all.set_title('All vehicles (RL = red)', color='#eeeeff', fontsize=10)
    ax_all.axhline(TARGET_VELOCITY, color='lime', lw=1.0, ls='--', alpha=0.5)
    rl_set = set(rl_indices)
    lines_all = []
    for i in range(N):
        col = 'tomato' if i in rl_set else '#5599cc'
        lw = 1.6 if i in rl_set else 0.6
        al = 1.0 if i in rl_set else 0.35
        ln, = ax_all.plot([], [], color=col, lw=lw, alpha=al)
        lines_all.append(ln)

    def _update(frame):
        t = frame
        xs = np.arange(t + 1)

        xy_t = xy[t]
        scat.set_offsets(xy_t)
        scat.set_array(speeds[t])
        rl_ring.set_offsets(xy_t[rl_indices])

        hud.set_text(
            f't={t}/{T-1}\n'
            f'RL speed (mean) = {rl_speeds[t]:5.2f} m/s  (goal {TARGET_VELOCITY:.0f})\n'
            f'RL accel (mean) = {actions_ms2[t]:+5.2f} m/s²\n'
            f'action  (mean)  = {actions[t]:+5.3f}  (×{max_accel:.0f} → m/s²)')

        rl_spd_ln.set_data(xs, rl_speeds[:t + 1])
        rl_spd_dot.set_data([t], [rl_speeds[t]])
        rl_acc_ln.set_data(xs, actions_ms2[:t + 1])
        rl_acc_dot.set_data([t], [actions_ms2[t]])
        act_ln.set_data(xs, actions[:t + 1])

        for i, ln in enumerate(lines_all):
            ln.set_data(xs, speeds[:t + 1, i])

        artists = [scat, rl_ring, hud, rl_spd_ln, rl_spd_dot,
                   rl_acc_ln, rl_acc_dot, act_ln] + lines_all
        return artists

    ani = animation.FuncAnimation(
        fig, _update, frames=T, interval=int(1000 / fps),
        blit=True, repeat=False)
    return ani, fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None,
                        help='Path to a .pkl PPO checkpoint.  '
                             'Omit to use a freshly-initialized (random) policy.')
    parser.add_argument('--env', default='flow_figureeight',
                        choices=list(_ENV_CLASSES.keys()),
                        help='Which figure-eight variant to render.')
    parser.add_argument('--steps', type=int, default=1500,
                        help='Simulation steps (1500 = full episode).')
    parser.add_argument('--fps', type=int, default=20,
                        help='Animation frames per second (default 20).')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Keep every Nth frame (default 1 = all frames). '
                             'Use 2+ to reduce memory when saving long GIFs.')
    parser.add_argument('--output', default='videos/figureeight/figureeight_random_full.mp4',
                        help='Output file path (.gif or .mp4).')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # 1. Build networks.
    print('[video] building networks...')
    networks, obs_dim = _build_networks(env_name=args.env, seed=args.seed)

    # 2. Load or initialise policy params.
    if args.checkpoint:
        print(f'[video] loading checkpoint: {args.checkpoint}')
        ckpt = _ppo_learner.load_checkpoint(args.checkpoint)
        policy_params = ckpt['policy_params']
        print(f'        iter={ckpt.get("iteration")}  '
              f'global_step={ckpt.get("global_step")}')
    else:
        print('[video] no checkpoint — using random initial policy.')
        import jax
        rng = jax.random.PRNGKey(args.seed)
        policy_params = networks.policy_network.init(rng)

    # 3. Collect rollout.
    data = collect_rollout(policy_params, networks,
                           n_steps=args.steps, seed=args.seed,
                           env_name=args.env)

    T = len(data['speeds'])
    rl_final = data['rl_speeds'][-10:].mean()
    print(f'[video] rollout done: {T} steps (done_step={data.get("done_step", T)})')
    print(f'        RL speed last-10 mean = {rl_final:.2f} m/s (target {TARGET_VELOCITY})')
    print(f'        RL accel range = [{data["actions_ms2"].min():+.2f}, '
          f'{data["actions_ms2"].max():+.2f}] m/s²')

    # Subsample frames to reduce peak memory for long GIFs.
    skip = max(1, int(args.frame_skip))
    if skip > 1:
        for k in ('speeds', 'rl_speeds', 'actions', 'actions_ms2', 'rewards', 'xy'):
            if k in data:
                data[k] = data[k][::skip]
        T = len(data['speeds'])
        print(f'[video] frame_skip={skip}: {T} frames to render')

    print(f'[video] building animation ({T} frames at {args.fps} fps)...')
    ani, fig = build_animation(data, fps=args.fps)

    ext = os.path.splitext(args.output)[1].lower()
    if ext == '.gif':
        print(f'[video] saving GIF → {args.output}')
        writer = animation.PillowWriter(fps=args.fps)
        ani.save(args.output, writer=writer, dpi=100)
    elif ext in ('.mp4', '.m4v', '.mov'):
        print(f'[video] saving MP4 → {args.output}')
        try:
            writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
            ani.save(args.output, writer=writer, dpi=110)
        except Exception as exc:
            gif_path = os.path.splitext(args.output)[0] + '.gif'
            print(f'[video] FFMpegWriter failed ({exc}); saving GIF → {gif_path}')
            writer = animation.PillowWriter(fps=min(args.fps, 15))
            ani.save(gif_path, writer=writer, dpi=90)
            args.output = gif_path
    else:
        raise ValueError(f'Unsupported output extension: {ext}')
    print(f'[video] saved → {args.output}')

    plt.close(fig)
    print('[video] done.')


if __name__ == '__main__':
    main()
