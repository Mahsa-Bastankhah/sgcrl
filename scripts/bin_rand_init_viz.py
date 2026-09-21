#!/usr/bin/env python3
"""Render a few Sawyer-bin default-rand inits and report gripper–object contact.

Default rand = --sawyer_randomize_init=true (MetaWorld object XY) with the
wrapper TCP parked at object + (0, 0, 0.03 m) and gripper width ≈ 0.4.

  python scripts/bin_rand_init_viz.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_utils  # noqa: E402

N_EXAMPLES = 4
CAMS = ("corner2", "corner3")
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figs", "sawyer_bin", "bin_default_rand_init.png")


def _render(env, camera: str, width: int = 640, height: int = 480) -> np.ndarray:
  return np.asarray(
      env.render(offscreen=True, camera_name=camera, resolution=(width, height)),
      dtype=np.uint8)


def _body_name(env, geom_id: int) -> str:
  bid = int(env.model.geom_bodyid[int(geom_id)])
  try:
    name = env.model.body_id2name(bid)
  except Exception:
    name = None
  return str(name) if name else f"body{bid}"


def _contacts(env) -> list[tuple[str, str, float]]:
  out = []
  ncon = int(getattr(env.data, "ncon", 0))
  for i in range(ncon):
    c = env.data.contact[i]
    dist = float(getattr(c, "dist", 0.0))
    out.append((_body_name(env, c.geom1), _body_name(env, c.geom2), dist))
  return out


def _gripper_obj_contacts(pairs: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
  grip_keys = ("claw", "pad", "hand", "finger", "grip")
  obj_keys = ("obja", "obj")
  hits = []
  for a, b, d in pairs:
    al, bl = a.lower(), b.lower()
    a_g = any(k in al for k in grip_keys)
    b_g = any(k in bl for k in grip_keys)
    a_o = any(k in al for k in obj_keys)
    b_o = any(k in bl for k in obj_keys)
    if (a_g and b_o) or (b_g and a_o):
      hits.append((a, b, d))
  return hits


def main() -> None:
  env, _, _ = env_utils.load("sawyer_bin", seed=0, randomize_init=True)
  print("randomize_init", env._randomize_init,
        "randomize_gripper_init", env._randomize_gripper_init,
        "mw.random_init", env.random_init)
  print("PSI_TCP_ABOVE_OBJ", env.PSI_TCP_ABOVE_OBJ, "PSI_GRIPPER", env.PSI_GRIPPER)
  if hasattr(env, "obj_init_pos"):
    print("obj_init_pos", np.asarray(env.obj_init_pos))
  if hasattr(env, "_random_reset_space"):
    sp = env._random_reset_space
    print("random_reset_space low", np.asarray(sp.low), "high", np.asarray(sp.high))

  rows = []
  stats = []
  for i, seed in enumerate((0, 1, 2, 3)[:N_EXAMPLES]):
    np.random.seed(seed)
    obs = env.reset()
    hand = np.asarray(obs[:3], dtype=np.float64)
    grip = float(obs[3])
    obj = np.asarray(obs[4:7], dtype=np.float64)
    d = float(np.linalg.norm(hand - obj))
    dz = float(hand[2] - obj[2])
    dxy = float(np.linalg.norm(hand[:2] - obj[:2]))
    pairs = _contacts(env)
    hits = _gripper_obj_contacts(pairs)
    stats.append(dict(seed=seed, hand=hand, obj=obj, grip=grip, d=d, dz=dz,
                      dxy=dxy, ncon=len(pairs), n_grip_obj=len(hits), hits=hits))
    print(f"\nseed {seed}")
    print(f"  obj  {obj}")
    print(f"  hand {hand}")
    print(f"  grip {grip:.3f}  hand-obj {d:.4f} m  xy {dxy:.4f}  dz {dz:.4f}")
    print(f"  ncon={len(pairs)} gripper-obj contacts={len(hits)}")
    for a, b, dist in hits[:8]:
      print(f"    {a} <-> {b}  dist={dist:.5f}")
    if not hits:
      print("    (no named gripper–object contacts)")
    rows.append([_render(env, c) for c in CAMS])

  import matplotlib.pyplot as plt
  fig, axes = plt.subplots(N_EXAMPLES, len(CAMS), figsize=(11.5, 3.4 * N_EXAMPLES))
  if N_EXAMPLES == 1:
    axes = np.array([axes])
  fig.suptitle(
      "Sawyer bin · default rand init  (object XY randomized, gripper parked on cube)\n"
      "TCP target = obj + (0,0,0.03 m), gripper width ≈ 0.4  ·  "
      f"randomize_gripper_init={env._randomize_gripper_init}",
      fontsize=12, fontweight="bold")
  for r, (frames, st) in enumerate(zip(rows, stats)):
    hit = "YES contact" if st["n_grip_obj"] else "NO named contact"
    label = (
        f"seed {st['seed']}  {hit}\n"
        f"hand-obj {st['d']:.3f} m  xy {st['dxy']:.3f}  dz {st['dz']:.3f}\n"
        f"grip={st['grip']:.2f}  ncon={st['ncon']}"
    )
    for c, (cam, img) in enumerate(zip(CAMS, frames)):
      ax = axes[r, c]
      ax.imshow(img)
      ax.set_title(f"seed {st['seed']} · {cam}", fontsize=10)
      ax.axis("off")
      if c == 0:
        ax.text(
            8, img.shape[0] - 8, label, fontsize=8, family="monospace",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
  os.makedirs(os.path.dirname(OUT), exist_ok=True)
  fig.tight_layout()
  fig.savefig(OUT, dpi=140, bbox_inches="tight")
  plt.close(fig)
  print(f"\n→ {OUT}")


if __name__ == "__main__":
  main()
