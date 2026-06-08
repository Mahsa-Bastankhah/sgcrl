"""Render MetaWorld SawyerBin: initial state and ψ goal state side-by-side.

ψ goal: gripper closed around the object, object in the target bin.

Usage:
    python scripts/bin_goal_viz.py
    python scripts/bin_goal_viz.py --output figs/bin_goal_render.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_utils


def _place_object(env, pos: np.ndarray) -> None:
  """Teleport the cube to ``pos``."""
  env._set_obj_xyz(pos)
  env.sim.forward()


def _place_gripper(env, target_pos: np.ndarray,
                   grip: float = -1.0, steps: int = 160) -> None:
  """Weld mocap to ``target_pos`` and hold gripper at ``grip`` effort."""
  mocap_quat = np.array([1.0, 0.0, 1.0, 0.0])
  mocap_pos = np.asarray(target_pos, dtype=np.float64).copy()
  ctrl = [grip, -grip]
  for _ in range(steps):
    mocap_pos += target_pos - env.get_endeff_pos()
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
  parser.add_argument('--output', default='figs/bin_goal_render.png')
  parser.add_argument('--width',  type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  args = parser.parse_args()

  fixed_goal = np.array([0.12, 0.7, 0.02], dtype=np.float32)
  env, _, _ = env_utils.load('sawyer_bin', fixed_goal, seed=0)

  # ── INIT state ────────────────────────────────────────────────────────────
  obs_init     = env.reset()
  hand_init    = obs_init[0:3]
  obj_init     = obs_init[4:7]
  gripper_init = float(obs_init[3])

  cam_list = ['corner2', 'corner3', 'topview']
  init_frames = [_render(env, c, args.width, args.height) for c in cam_list]

  # ── GOAL state ────────────────────────────────────────────────────────────
  # Object init pos is ~[-0.12, 0.7, 0.02]; target bin is [0.12, 0.7, 0.02].
  # Expert grips at cube + [0, 0, 0.03].  We close the gripper (ctrl +1).
  env.reset()
  goal_pos    = env._goal.copy()              # [0.12, 0.7, 0.02]
  # hand grips the cube: TCP at cube center + 3cm above
  grip_pos    = goal_pos + np.array([0.0, 0.0, 0.03])

  _place_object(env, goal_pos)
  _place_gripper(env, grip_pos, grip=1.0)     # grip=+1 → close fingers
  # Re-lock object in case physics moved it slightly
  _place_object(env, goal_pos)

  obs_goal     = env._get_obs()
  hand_goal    = obs_goal[0:3]
  obj_goal     = obs_goal[4:7]
  gripper_goal = float(obs_goal[3])

  goal_frames = [_render(env, c, args.width, args.height) for c in cam_list]

  # ── Panel ─────────────────────────────────────────────────────────────────
  import matplotlib.pyplot as plt
  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

  fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  fig.suptitle(
      'SawyerBin — TOP: init state   BOTTOM: ψ goal state\n'
      'ψ goal: object in target bin, gripper closed (holding)',
      fontsize=12, fontweight='bold')

  init_label = [
      f'hand:    [{hand_init[0]:.3f}  {hand_init[1]:.3f}  {hand_init[2]:.3f}]',
      f'object:  [{obj_init[0]:.3f}  {obj_init[1]:.3f}  {obj_init[2]:.3f}]  LEFT bin',
      f'gripper: {gripper_init:.2f}  (open=1)',
  ]
  goal_label = [
      f'hand:    [{hand_goal[0]:.3f}  {hand_goal[1]:.3f}  {hand_goal[2]:.3f}]',
      f'object:  [{obj_goal[0]:.3f}  {obj_goal[1]:.3f}  {obj_goal[2]:.3f}]  RIGHT bin',
      f'gripper: {gripper_goal:.2f}  (closed≈0)',
  ]

  for col, (cam, init_img, goal_img) in enumerate(
      zip(cam_list, init_frames, goal_frames)):
    for row, (ax, img, label) in enumerate(
        [(axes[0][col], init_img, init_label),
         (axes[1][col], goal_img, goal_label)]):
      ax.imshow(img)
      ax.set_title(f'{"init" if row==0 else "goal"} — {cam}', fontsize=10)
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
  print(f'\nINIT:  hand={hand_init}  obj={obj_init}  gripper={gripper_init:.2f}')
  print(f'GOAL:  hand={hand_goal}  obj={obj_goal}  gripper={gripper_goal:.2f}')
  print(f'  ideal grip_pos = {grip_pos}')


if __name__ == '__main__':
  main()
