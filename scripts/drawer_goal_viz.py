"""Render MetaWorld Sawyer drawer: initial state and ψ goal state side-by-side.

Usage:
    python scripts/drawer_goal_viz.py
    python scripts/drawer_goal_viz.py --output figs/drawer_goal_render.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_utils


def _set_drawer_open(env, goal_handle: np.ndarray) -> None:
  try:
    jid = env.model.joint_name2id('goal_slidey')
    qpos_addr = env.model.jnt_qposadr[jid]
    env.data.qpos[qpos_addr] = float(env.model.jnt_range[jid][0])  # fully open
    env.sim.forward()
  except Exception:
    pass

  delta = goal_handle - env._get_pos_objects()
  if np.linalg.norm(delta) > 1e-3:
    env.sim.model.body_pos[env.model.body_name2id('drawer')] += delta
    env.sim.forward()


def _restore_drawer_closed(env) -> None:
  """Reset drawer body pos and slide joint to the closed default."""
  try:
    jid = env.model.joint_name2id('goal_slidey')
    env.data.qpos[env.model.jnt_qposadr[jid]] = 0.0
    env.sim.forward()
  except Exception:
    pass
  env.sim.model.body_pos[env.model.body_name2id('drawer')] = (
      env.init_config['obj_init_pos'].copy())
  env.sim.forward()


def _place_gripper(env, target_pos: np.ndarray, steps: int = 160) -> None:
  mocap_quat = np.array([1.0, 0.0, 1.0, 0.0])
  mocap_pos = np.asarray(target_pos, dtype=np.float64).copy()
  for _ in range(steps):
    mocap_pos += target_pos - env.get_endeff_pos()
    env.data.set_mocap_pos('mocap', mocap_pos)
    env.data.set_mocap_quat('mocap', mocap_quat)
    env.do_simulation([-1.0, 1.0], env.frame_skip)
  env.sim.forward()


def _render(env, camera: str, width: int, height: int) -> np.ndarray:
  return np.asarray(
      env.render(offscreen=True, camera_name=camera, resolution=(width, height)),
      dtype=np.uint8)


def _annotate(ax, lines: list[str]) -> None:
  """Add a white-bg text box in the lower-left of an imshow axis."""
  ax.text(
      8, ax.get_ylim()[0] - 8,
      '\n'.join(lines),
      fontsize=8, family='monospace', va='bottom',
      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', default='figs/drawer_goal_render.png')
  parser.add_argument('--width',  type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  args = parser.parse_args()

  fixed_goal = np.array([0.0, 0.54, 0.09], dtype=np.float32)
  env, _, _ = env_utils.load('sawyer_drawer_open', fixed_goal, seed=0)

  # ── INIT state ────────────────────────────────────────────────────────────
  # Hard-reset drawer to closed before capturing init.
  env.reset()
  _restore_drawer_closed(env)
  obs_init    = env._get_obs()
  hand_init   = obs_init[0:3]
  handle_init = obs_init[4:7]

  # Render from corner2 (angled), corner3 (more side-on), topview
  cam_list = ['corner2', 'corner3', 'topview']
  init_frames = [_render(env, c, args.width, args.height) for c in cam_list]

  # ── GOAL state ────────────────────────────────────────────────────────────
  env.reset()
  goal_handle = env._goal.copy()
  ideal_hand  = env._ideal_pull_hand()

  _set_drawer_open(env, goal_handle)
  _place_gripper(env, ideal_hand)
  _set_drawer_open(env, goal_handle)

  obs_goal    = env._get_obs()
  hand_goal   = obs_goal[0:3]
  handle_goal = obs_goal[4:7]
  gripper_val = float(obs_goal[3])

  goal_frames = [_render(env, c, args.width, args.height) for c in cam_list]

  # ── Panel ─────────────────────────────────────────────────────────────────
  import matplotlib.pyplot as plt

  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

  fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  fig.suptitle(
      'SawyerDrawerOpen — TOP: init state   BOTTOM: ψ goal state',
      fontsize=14, fontweight='bold')

  init_label = [
      f'hand:   [{hand_init[0]:.3f}  {hand_init[1]:.3f}  {hand_init[2]:.3f}]',
      f'handle: [{handle_init[0]:.3f}  {handle_init[1]:.3f}  {handle_init[2]:.3f}]  CLOSED',
      f'gripper: open (1.0)',
  ]
  goal_label = [
      f'hand:   [{hand_goal[0]:.3f}  {hand_goal[1]:.3f}  {hand_goal[2]:.3f}]',
      f'handle: [{handle_goal[0]:.3f}  {handle_goal[1]:.3f}  {handle_goal[2]:.3f}]  OPEN',
      f'gripper: {gripper_val:.1f}  (open=1)',
  ]

  for col, (cam, init_img, goal_img) in enumerate(
      zip(cam_list, init_frames, goal_frames)):
    for row, (ax, img, label) in enumerate(
        [(axes[0][col], init_img, init_label),
         (axes[1][col], goal_img, goal_label)]):
      ax.imshow(img)
      ax.set_title(f'{"init" if row==0 else "goal"} — {cam}', fontsize=10)
      ax.axis('off')
      # text overlay
      ax.text(
          6, args.height - 6,
          '\n'.join(label),
          fontsize=7.5, family='monospace', va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))

  out_panel = args.output.replace('.png', '_panel.png')
  plt.tight_layout()
  plt.savefig(out_panel, dpi=130, bbox_inches='tight')
  print(f'Saved: {out_panel}')

  print(f'\nINIT:  hand={hand_init}  handle={handle_init}  (CLOSED, y≈0.73)')
  print(f'GOAL:  hand={hand_goal}  handle={handle_goal}  (OPEN,   y≈0.54)  gripper={gripper_val:.2f}')


if __name__ == '__main__':
  main()
