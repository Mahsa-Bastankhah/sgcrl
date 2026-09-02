#!/usr/bin/env python3
"""One-env GPU check: cube on desk, palm IK'd onto it.

Exits 0 only if |palm−obj| < TABLE_SPAWN_MAX_PALM_OBJ after reset noise.

  python scripts/probe_allegro_table_spawn_touch.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from envs.allegro_kuka_throw_env import (  # noqa: E402
    TABLE_SIDE_GOAL_XYZ,
    TABLE_SIDE_PALM_XYZ,
    TABLE_SPAWN_MAX_PALM_OBJ,
    AllegroKukaThrowVecEnv,
)


def main() -> int:
    env = AllegroKukaThrowVecEnv(
        num_envs=1,
        seed=0,
        episode_length=300,
        table_spawn=True,
        table_push=True,
        table_push_xyz=TABLE_SIDE_GOAL_XYZ,
        palm_goal=True,
        palm_goal_xyz=TABLE_SIDE_PALM_XYZ,
        randomize_init=True,
        randomize_object_shape=False,
    )
    env.reset()
    palm = env._palm_xyz()[0]
    obj = env._object_xyz()[0]
    dist = float((palm - obj).norm().item())
    print(
        f'PROBE palm={tuple(round(float(v), 3) for v in palm.tolist())} '
        f'obj={tuple(round(float(v), 3) for v in obj.tolist())} '
        f'|palm-obj|={dist:.3f}m  limit={TABLE_SPAWN_MAX_PALM_OBJ:.2f}m',
        flush=True,
    )
    if dist > TABLE_SPAWN_MAX_PALM_OBJ:
        print('PROBE FAIL', flush=True)
        return 1
    print('PROBE OK', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
