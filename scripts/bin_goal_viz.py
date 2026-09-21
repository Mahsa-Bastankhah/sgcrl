"""Render MetaWorld SawyerBin: init vs floor-hold ψ goal, plus gripper zoom.

ψ goal: closed gripper around the cube sitting on the floor at ``_goal``.
Success still only checks object distance to ``_goal`` (unchanged).

Usage:
    python scripts/bin_goal_viz.py
    python scripts/bin_goal_viz.py --output figs/sawyer_bin/bin_goal_render.png
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


def _move_tcp(env, target_pos: np.ndarray, *, grip_ctrl: float,
              steps: int = 200) -> None:
  target = np.asarray(target_pos, dtype=np.float64)
  mocap_quat = np.array([1.0, 0.0, 1.0, 0.0])
  mocap_pos = env.data.get_mocap_pos('mocap').copy()
  ctrl = [float(grip_ctrl), -float(grip_ctrl)]
  for _ in range(steps):
    mocap_pos = mocap_pos + (target - env.get_endeff_pos())
    env.data.set_mocap_pos('mocap', mocap_pos)
    env.data.set_mocap_quat('mocap', mocap_quat)
    env.do_simulation(ctrl, env.frame_skip)
  env.sim.forward()


def _place_psi_goal(env, held_obj: np.ndarray, hand_pos: np.ndarray,
                    grip: float = 0.4) -> None:
  """Place floor-hold matching the ψ vector (grip≈cube width)."""
  held_obj = np.asarray(held_obj, dtype=np.float64)
  hand_pos = np.asarray(hand_pos, dtype=np.float64)
  approach = hand_pos.copy()
  approach[2] = max(float(hand_pos[2]) + 0.12, 0.18)

  _place_object(env, held_obj)
  _move_tcp(env, approach, grip_ctrl=-1.0, steps=320)
  _place_object(env, held_obj)
  _move_tcp(env, hand_pos, grip_ctrl=-1.0, steps=320)
  _place_object(env, held_obj)
  # Calibrate the actual finger-site aperture after positioning. Additional
  # dynamics here can make contacts push the arm away from the visualized pose.
  env._set_gripper_obs_width(grip)
  _place_object(env, held_obj)
  env.data.set_mocap_pos('mocap', hand_pos.copy())
  env.sim.forward()


def _render(env, camera: str, width: int, height: int) -> np.ndarray:
  return np.asarray(
      env.render(offscreen=True, camera_name=camera, resolution=(width, height)),
      dtype=np.uint8)


def _zoom_crop(img: np.ndarray, *, frac: float = 0.32,
               cy: float = 0.42, cx: float = 0.58) -> np.ndarray:
  """Crop around the blue-bin / grasp region, then upsample."""
  h, w = img.shape[:2]
  ch, cw = int(h * frac), int(w * frac)
  y0 = max(0, min(h - ch, int(h * cy) - ch // 2))
  x0 = max(0, min(w - cw, int(w * cx) - cw // 2))
  crop = img[y0:y0 + ch, x0:x0 + cw]
  try:
    import cv2
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
  except Exception:
    sy, sx = max(h // max(ch, 1), 1), max(w // max(cw, 1), 1)
    return np.repeat(np.repeat(crop, sy, axis=0), sx, axis=1)[:h, :w]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--output', default='figs/sawyer_bin/bin_goal_render.png')
  parser.add_argument('--width', type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  args = parser.parse_args()

  fixed_goal = np.array([0.12, 0.7, 0.02], dtype=np.float32)
  env, _, _ = env_utils.load('sawyer_bin', fixed_goal, seed=0)

  obs_init = env.reset()
  hand_init = obs_init[0:3]
  obj_init = obs_init[4:7]
  gripper_init = float(obs_init[3])

  cam_list = ['corner2', 'corner3', 'topview']
  init_frames = [_render(env, c, args.width, args.height) for c in cam_list]

  env.reset()
  success_target = env._goal.copy()
  held_obj = env._psi_held_object().copy()
  psi_hand = env._ideal_grasp_hand().copy()
  psi_grip = float(env.PSI_GRIPPER)
  psi_vec = np.concatenate([psi_hand, [psi_grip], held_obj]).astype(np.float32)
  _place_psi_goal(env, held_obj, psi_hand, grip=psi_grip)

  obs_goal = env._get_obs()
  hand_goal = obs_goal[0:3]
  obj_goal = obs_goal[4:7]
  gripper_goal = float(obs_goal[3])
  goal_frames = [_render(env, c, args.width, args.height) for c in cam_list]
  zoom_frames = [_zoom_crop(f) for f in goal_frames]

  import matplotlib.pyplot as plt
  os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

  fig, axes = plt.subplots(3, 3, figsize=(18, 14))
  fig.suptitle(
      'SawyerBin — row1 init · row2 floor-hold ψ goal · row3 zoom on gripper\n'
      f'ψ: grip={float(env_utils.SawyerBin.PSI_GRIPPER)} around cube on floor  |  '
      f'success ‖obj−_goal‖ only  (_goal={success_target})',
      fontsize=12, fontweight='bold')

  init_label = [
      f'hand: [{hand_init[0]:.3f} {hand_init[1]:.3f} {hand_init[2]:.3f}]',
      f'obj:  [{obj_init[0]:.3f} {obj_init[1]:.3f} {obj_init[2]:.3f}] LEFT',
      f'gripper: {gripper_init:.2f} (open≈1)',
      f'hand-obj dist: {np.linalg.norm(hand_init - obj_init):.3f} m',
  ]
  goal_label = [
      f'ψ hand: [{psi_vec[0]:.3f} {psi_vec[1]:.3f} {psi_vec[2]:.3f}]',
      f'ψ obj:  [{psi_vec[4]:.3f} {psi_vec[5]:.3f} {psi_vec[6]:.3f}]  '
      f'grip={psi_vec[3]:.1f}',
      f'rendered hand-obj dist: '
      f'{np.linalg.norm(hand_goal - obj_goal):.3f} m',
      f'success _goal (floor): '
      f'[{success_target[0]:.3f} {success_target[1]:.3f} '
      f'{success_target[2]:.3f}]',
  ]
  zoom_label = ['ZOOM: floor hold, grip≈0.4 (cube width)']

  rows = [
      (0, init_frames, init_label, 'init'),
      (1, goal_frames, goal_label, 'held ψ goal'),
      (2, zoom_frames, zoom_label, 'zoom'),
  ]
  for row_i, frames, label, name in rows:
    for col, (cam, img) in enumerate(zip(cam_list, frames)):
      ax = axes[row_i, col]
      ax.imshow(img)
      ax.set_title(f'{name} — {cam}', fontsize=10)
      ax.axis('off')
      ax.text(
          6, args.height - 6,
          '\n'.join(label),
          fontsize=7.5, family='monospace', va='bottom',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.88))

  out_panel = args.output.replace('.png', '_panel.png')
  plt.tight_layout()
  plt.savefig(out_panel, dpi=130, bbox_inches='tight')
  # Also save a dedicated zoom strip.
  out_zoom = args.output.replace('.png', '_zoom.png')
  fig_z, ax_z = plt.subplots(1, 3, figsize=(15, 4.5))
  fig_z.suptitle('Floor-hold ψ goal — zoomed gripper', fontsize=12, fontweight='bold')
  for ax, cam, img in zip(ax_z, cam_list, zoom_frames):
    ax.imshow(img)
    ax.set_title(cam)
    ax.axis('off')
  plt.tight_layout()
  fig_z.savefig(out_zoom, dpi=140, bbox_inches='tight')
  plt.close(fig_z)
  plt.close(fig)

  print(f'Saved: {out_panel}')
  print(f'Saved: {out_zoom}')
  print(f'success _goal={success_target}')
  print(f'ψ vec={psi_vec}')
  print(f'RENDERED: hand={hand_goal} obj={obj_goal} grip={gripper_goal:.2f}')
  print(f'  hand err={np.linalg.norm(hand_goal - psi_hand):.4f} m'
        f'  obj err={np.linalg.norm(obj_goal - held_obj):.4f} m')


if __name__ == '__main__':
  main()
