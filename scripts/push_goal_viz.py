"""Render MetaWorld SawyerPush: initial state and ψ goal state side-by-side.

ψ goal: puck at target, hand behind/above puck with gripper closed.

Usage:
    python scripts/push_goal_viz.py
    python scripts/push_goal_viz.py --output figs/push_goal_render.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_utils


def _place_object(env, pos: np.ndarray) -> None:
    env._set_obj_xyz(np.asarray(pos, dtype=np.float64))
    env.sim.forward()


def _place_gripper(env, target_pos: np.ndarray,
                   grip: float = -1.0, steps: int = 160) -> None:
    mocap_quat = np.array([1.0, 0.0, 1.0, 0.0])
    mocap_pos = np.asarray(target_pos, dtype=np.float64).copy()
    ctrl = [grip, -grip]
    for _ in range(steps):
        mocap_pos += target_pos - env.tcp_center
        env.data.set_mocap_pos('mocap', mocap_pos)
        env.data.set_mocap_quat('mocap', mocap_quat)
        env.do_simulation(ctrl, env.frame_skip)
    env.sim.forward()


def _render(env, camera: str, width: int, height: int) -> np.ndarray:
    return np.asarray(
        env.render(offscreen=True, camera_name=camera, resolution=(width, height)),
        dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='figs/push_goal_render.png')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    fixed_goal = np.array([0.0, 0.85, 0.02], dtype=np.float32)
    env, _, _ = env_utils.load('sawyer_push', fixed_goal, seed=0)

    # ── INIT state ────────────────────────────────────────────────────────────
    obs_init = env.reset()
    hand_init = obs_init[0:3]
    puck_init = obs_init[4:7]
    gripper_init = float(obs_init[3])

    cam_list = ['corner2', 'corner3', 'topview']
    init_frames = [_render(env, c, args.width, args.height) for c in cam_list]

    # ── GOAL state ────────────────────────────────────────────────────────────
    env.reset()
    puck_goal_pos = env._goal.copy()
    ideal_hand = puck_goal_pos + np.array([0.0, -0.08, 0.03], dtype=np.float32)

    _place_object(env, puck_goal_pos)
    _place_gripper(env, ideal_hand, grip=-1.0, steps=160)

    obs_goal = env._get_obs()
    hand_goal = obs_goal[0:3]
    puck_goal = obs_goal[4:7]
    gripper_goal = float(obs_goal[3])

    goal_frames = [_render(env, c, args.width, args.height) for c in cam_list]

    # ── Panel ─────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'SawyerPush — TOP: init state   BOTTOM: ψ goal state',
        fontsize=14, fontweight='bold')

    init_label = [
        f'hand:   [{hand_init[0]:.3f}  {hand_init[1]:.3f}  {hand_init[2]:.3f}]',
        f'puck:   [{puck_init[0]:.3f}  {puck_init[1]:.3f}  {puck_init[2]:.3f}]',
        f'gripper: {gripper_init:.2f}',
        f'puck-goal dist: {np.linalg.norm(puck_init - fixed_goal):.3f} m',
    ]
    goal_label = [
        f'hand:   [{hand_goal[0]:.3f}  {hand_goal[1]:.3f}  {hand_goal[2]:.3f}]',
        f'puck:   [{puck_goal[0]:.3f}  {puck_goal[1]:.3f}  {puck_goal[2]:.3f}]',
        f'gripper: {gripper_goal:.2f}  (closed≈0)',
        f'target: [{fixed_goal[0]:.3f}  {fixed_goal[1]:.3f}  {fixed_goal[2]:.3f}]',
        f'ideal hand: [{ideal_hand[0]:.3f}  {ideal_hand[1]:.3f}  {ideal_hand[2]:.3f}]',
    ]

    for col, (cam, init_img, goal_img) in enumerate(
        zip(cam_list, init_frames, goal_frames)):
        for row, (ax, img, label) in enumerate(
            [(axes[0][col], init_img, init_label),
             (axes[1][col], goal_img, goal_label)]):
            ax.imshow(img)
            ax.set_title(f'{"init" if row == 0 else "goal"} — {cam}', fontsize=10)
            ax.axis('off')
            ax.text(
                6, args.height - 6,
                '\n'.join(label),
                fontsize=7.5, family='monospace', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))

    out_panel = args.output.replace('.png', '_panel.png')
    plt.tight_layout()
    plt.savefig(out_panel, dpi=130, bbox_inches='tight')
    print(f'Saved: {out_panel}')
    print(f'\nINIT:  hand={hand_init}  puck={puck_init}  gripper={gripper_init:.2f}')
    print(f'GOAL:  hand={hand_goal}  puck={puck_goal}  gripper={gripper_goal:.2f}')
    print(f'  fixed puck target = {fixed_goal}  ideal_hand = {ideal_hand}')


if __name__ == '__main__':
    main()
