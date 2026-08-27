"""Shared per-env defaults for the standalone PPO entrypoints.

Factored out of `ppo_contrastive.py` so `ppo_rnd.py` (PPO+RND) can import
`fixed_goal_dict`/`PPO_ENV_DEFAULTS` without also importing `ppo_contrastive`
itself -- that module registers its own `absl.flags` at import time, and
`ppo_rnd.py` registers an overlapping-but-not-identical set of flag names
(e.g. `--seed`, `--env`, `--hidden_layer_sizes`), which would collide with a
`flags.DuplicateFlagError` if both modules defined them in the same process.

This module has no side effects (no flag registration) and is safe to import
from anywhere. `ppo_contrastive.py` re-exports these two names (`from
ppo_env_defaults import fixed_goal_dict, PPO_ENV_DEFAULTS`) so existing code
that does `from ppo_contrastive import fixed_goal_dict` keeps working
unchanged.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Fixed-goal lookup reused from lp_contrastive.py.
# ---------------------------------------------------------------------------
fixed_goal_dict = {
    'point_Spiral7x7':   [np.array([3, 3], dtype=float),
                          np.array([6, 6], dtype=float)],
    'point_Spiral9x9':   [np.array([5, 5], dtype=float),
                          np.array([8, 8], dtype=float)],
    'point_Spiral11x11': [np.array([5, 5], dtype=float),
                          np.array([10, 10], dtype=float)],
    'point_FourRooms':   [np.array([0, 0], dtype=float),
                          np.array([10, 8],  dtype=float)],
    # point_Impossible: start top-left (0,0), goal row 6 col 8 (reachable via
    # the long winding path through the maze).
    'point_Impossible': [np.array([0, 0], dtype=float),
                         np.array([6, 8], dtype=float)],
    # point_Maze11x11: start top-left (0,0), goal top-right (0,10).
    'point_Maze11x11':  [np.array([0, 0], dtype=float),
                         np.array([0, 10], dtype=float)],
    'sawyer_bin':  np.array([0.12, 0.7, 0.02]),
    'sawyer_box':  np.array([0.0, 0.75, 0.133]),
    'sawyer_peg':  np.array([-0.3, 0.6, 0.0]),
    # One-hot goal over river cells (length must match RIVERSWIM_LEN, default 6).
    'riverswim': np.array([0., 0., 0., 0., 0., 1.], dtype=float),
    # PushCube-v1 randomizes both cube start and goal region every reset
    # inside ManiSkill itself; there's no single canonical fixed goal.
    'maniskill_pushcube': None,
    # PickCube-v1: same reasoning as maniskill_pushcube.
    'maniskill_pickcube': None,
    # OpenCabinetDrawer-v1 randomizes the cabinet model, robot start pose,
    # and which drawer to open every reset; same reasoning as above.
    'maniskill_open_cabinet_drawer': None,
    # CloseCabinetDrawer-v1 (this repo's own close-direction sibling of
    # OpenCabinetDrawer-v1, registered in env_utils.py): same per-episode
    # randomization as open, just starting each drawer open instead of
    # closed; same reasoning as above.
    'maniskill_close_cabinet_drawer': None,
    # CloseSubtaskTrain-v0 randomizes the scene/spawn/articulation instance
    # every reset (sampled from its premade task-plan/spawn-data assets);
    # same reasoning as above.
    'maniskill_close_subtask_train': None,
    # OpenSubtaskTrain-v0: same reasoning as maniskill_close_subtask_train
    # (scene/spawn/articulation instance randomized every reset from the
    # premade task-plan/spawn-data assets).
    'maniskill_open_subtask_train': None,
}

# ---------------------------------------------------------------------------
# Per-env PPO defaults.
#
# The single knob that really needs to scale with the environment is the
# rollout length T, because it interacts with episode length: if T < ep_len,
# most rollouts complete zero episodes and GAE must bootstrap the return off
# V(s_T), which hurts sample efficiency early in training.  The CRL step
# count scales proportionally so that the CRL-to-env-step ratio stays
# roughly constant (CRL-steps ≈ T / 2). `crl_steps_per_iter` is only
# consumed by `ppo_contrastive.py` (PPO+CRL) -- `ppo_rnd.py` (PPO+RND, no
# CRL critic) reads `rollout_length`/`end_index` only.
#
# point_FourRooms / point_Spiral11x11: 50/100-step episodes -> T=128 keeps
#   ~1-2 completed episodes per env per rollout.  These are the values the
#   tuned point-env runs use, so we keep them to avoid regressing.
# sawyer_{bin,box,peg}: 150-step episodes -> T=256 gives one full episode
#   per env per rollout, matching CleanRL's MuJoCo convention.
# ---------------------------------------------------------------------------
PPO_ENV_DEFAULTS = {
    'point_FourRooms':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Spiral7x7':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Spiral9x9':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Spiral11x11': dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Maze11x11':   dict(rollout_length=128, crl_steps_per_iter=64),
    'point_Impossible':  dict(rollout_length=128, crl_steps_per_iter=64),
    'riverswim':         dict(rollout_length=128, crl_steps_per_iter=64),
    'sawyer_bin':        dict(rollout_length=256, crl_steps_per_iter=128),
    'sawyer_box':        dict(rollout_length=256, crl_steps_per_iter=128),
    'sawyer_peg':        dict(rollout_length=256, crl_steps_per_iter=128),
    # PushCube-v1 has 50-step episodes, same order of magnitude as
    # point_FourRooms/point_Spiral* -- reuse their defaults.
    'maniskill_pushcube': dict(rollout_length=128, crl_steps_per_iter=64),
    # PickCube-v1: same 50-step episode length as PushCube.
    'maniskill_pickcube': dict(rollout_length=128, crl_steps_per_iter=64),
    # OpenCabinetDrawer-v1 has 100-step episodes (2x PushCube/PickCube) --
    # double T so ~1-2 episodes still complete per env per rollout.
    # end_index=6 (=ManiskillOpenCabinetDrawer.GOAL_DIM) is required, not
    # optional: this env's state (10 dims: tcp_pos, drawer_qpos,
    # gripper_aperture, is_grasping, sin/cos heading, base_x, base_y) is
    # longer than its goal (6 dims: handle_target_pos, drawer_target_qpos,
    # gripper_target, is_grasping_target) -- heading and base position are
    # state-only, with no goal counterpart (see ManiskillOpenCabinetDrawer's
    # docstring for why). Leaving end_index at its default (-1, "to the end
    # of the state") would make obs_to_goal read past the end of the raw
    # per-step goal vector emitted by the wrapper.
    'maniskill_open_cabinet_drawer': dict(
        rollout_length=256, crl_steps_per_iter=128, end_index=6),
    # CloseCabinetDrawer-v1: sibling of maniskill_open_cabinet_drawer with
    # the goal reversed (drawer starts open, success means closing it) --
    # same 100-step episodes, same STATE_DIM=10/GOAL_DIM=6 asymmetric split
    # (see ManiskillCloseCabinetDrawer's docstring), so identical defaults.
    'maniskill_close_cabinet_drawer': dict(
        rollout_length=256, crl_steps_per_iter=128, end_index=6),
    # CloseSubtaskTrain-v0 has 400-step episodes (spawn now requires
    # navigating to an adjacent room before manipulating the drawer, up from
    # the earlier 200-step/~3m-same-room-only version) -- T and CRL steps
    # scale 2x along with the episode length, keeping the same T/ep_len and
    # CRL-to-env-step ratios as before. State/goal are now an *asymmetric*
    # 10/8 split (base_xy/heading appended to state; base_xy has a goal
    # counterpart (target_base_xy), heading doesn't -- see
    # ManiskillCloseSubtaskTrain's docstring), same reason
    # maniskill_open_cabinet_drawer needs end_index=6 above: without
    # end_index=8 here, obs_to_goal would read past the end of the raw
    # 8-dim per-step goal.
    'maniskill_close_subtask_train': dict(
        rollout_length=512, crl_steps_per_iter=256, end_index=8),
    # OpenSubtaskTrain-v0: sibling of maniskill_close_subtask_train with the
    # goal reversed (drawer starts closed, success means opening it) --
    # same scene, same adjacent-room spawn, same episode length, same
    # asymmetric 10/8 state/goal split (see ManiskillOpenSubtaskTrain's
    # docstring), so identical defaults.
    'maniskill_open_subtask_train': dict(
        rollout_length=512, crl_steps_per_iter=256, end_index=8),
}
