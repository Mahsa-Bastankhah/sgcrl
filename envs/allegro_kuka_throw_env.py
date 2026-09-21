"""Isaac Gym AllegroKukaThrow adapter for the sgcrl PPO+NF pipeline.

This wraps NVIDIA IsaacGymEnvs' ``AllegroKukaThrow`` (Kuka iiwa7 arm + Allegro
16-DOF hand throwing an object toward a bucket) into a *batched* env that the
dedicated Isaac Gym learner (``contrastive/ppo_learner_isaacgym.py``) can drive.

Key differences from the upstream task:

* **Fixed goal.** Upstream randomizes the bucket / goal position every reset.
  Here we consider the bucket position *fixed* (a constant of the environment),
  so it never appears in the observation. ``_reset_target`` is overridden to
  place the bucket + goal at a single constant world-frame location.

* **Optional frozen init.** ``randomize_init=False`` zeros NVIDIA reset noise
  (object xyz, joints, velocities), keeps the authored object quaternion,
  uses a single default cube (no per-env shape mix), and sets ``forceScale=0``.
  Orientation must be overridden in code: upstream ``get_random_quat`` ignores
  ``resetRotationNoise`` and always samples a uniform quaternion.
  ``randomize_object_xyz=True`` with frozen init keeps NVIDIA
  ``resetPositionNoise{X,Y,Z}`` (±0.1 m xy, ±0.02 m z) but still freezes
  shape / joints / quat / forces.
  ``randomize_object_shape=False`` with full init keeps joints / quat /
  xyz / forces but uses a single default cube (no dimension mix).

* **Compact goal-conditioned observation.** Default throw packing
  (``palm_goal=False``, ``joint_goal=False``):

    - ``state[0:23]``  = joint positions normalized to [-1, 1]
      (arm 7 + hand 16; Isaac Gym ``unscale`` convention)
    - ``state[23:46]`` = joint velocities
    - ``state[46:49]`` = object absolute xyz
    - ``goal[0:3]``    = fixed bucket xyz

  With ``joint_goal=True`` (``--isaacgym_joint_goal``): same 49-D state (no
  palm xyz). Goal is the tight-10 *finger* paddle plus the object target
  (arm joints are not in the goal — they vary along the push):

    - ``goal[0:16]`` = ``TABLE_SIDE_JOINT_GOAL_Q[7:23]`` (hand only)
    - ``goal[16:19]`` = object xyz (table-push target, or bucket)
    - hindsight gathers ``state[7:23]`` (hand q) and ``state[46:49]`` (object)

  With ``palm_goal=True`` (``--isaacgym_palm_goal``), Sawyer-bin analog:

    - ``state[0:23]``  = joint positions
    - ``state[23:46]`` = joint velocities
    - ``state[46:49]`` = palm xyz (``palm_center_pos``)
    - ``state[49:52]`` = object xyz
    - ``goal[0:3]``    = commanded palm (default: on-desk slide pose)
    - ``goal[3:6]``    = object in bucket (bucket xy, origin + 5 cm z)

  ``table_push`` + ``palm_goal`` is allowed: bucket stays parked off-table,
  success / ``goal[3:6]`` use the on-desk object target, and ``goal[0:3]``
  is the commanded palm (typically a few cm above that object target).
  ``joint_goal`` wins over ``palm_goal`` (palm is dropped from the state).

  ``table_spawn`` (``--isaacgym_table_spawn``, default off): cube on the
  robot-near side of the desk (``TABLE_SPAWN_OBJECT_XY``, z =
  ``TABLE_OBJECT_Z``) with small xy noise, yaw-only quat, and a one-shot
  IK reach so the palm sits on the cube (touching). Arm / finger reset
  noise is small so hand and object stay nearby. Startup raises if
  ``|palm−obj|`` is large. Does not flip ``randomize_init`` globally.

  ``--isaacgym_table_spawn_toward_bucket`` (needs ``table_spawn``): IK the
  palm onto the *anti-bucket* side of the cube (standoff18 defaults:
  18 cm back / 8 cm up, paddle facing the throw bucket). Contact-free
  stand-off like ``table_spawn_behind``, but oriented for a throw toward
  ``DEFAULT_FIXED_TARGET_XYZ``. Wins over ``table_spawn_behind``.

  ``--isaacgym_table_spawn_in_hand`` (needs ``table_spawn``): after the
  table-spawn IK + paddle curl, teleport the cube into the palm
  (``palm + offset·palm_z``) and keep it there on reset. Intended for
  palm-down hover (``table_spawn_behind`` with ``dy=0``) so the episode
  starts already holding. ``tableSpawnCorrelatedXY`` jitters the IK
  target (hand xyz); ``tableSpawnInHandObjNoise`` jitters the cube in
  the palm tangent plane. Does not re-run IK on reset.

  ``--isaacgym_table_spawn_in_hand_keep_arm``: skip IK. Keep NVIDIA
  throw pose v1, add ``tableSpawnInHandWristOffset`` to A7, curl, snap.

  ``--isaacgym_lock_arm_base``: policy action is 19-D (A5–A7 + 16
  fingers). A1–A4 stay at the reset pose (relative arm action 0).
  State/goal stay 49/3. Incompatible with ``control_sanity_trim_sa``.

  ``--isaacgym_reset_z_above`` (default 0 = off): extra episode reset when
  object z exceeds the threshold, via NVIDIA ``_extra_reset_rules``. Same
  path as the built-in fall reset (``z < 0.1``).

  ``--isaacgym_hide_table``: keep NVIDIA's table actor but swap in a 1 cm
  box parked far away so the workspace is table-free. Mutually exclusive
  with ``large_table``. Use with a throw-bucket goal so a miss falls to
  the ground (NVIDIA resets at object ``z < 0.1``).

  ``throw_success`` (``--isaacgym_throw_success``, default ``in_bucket``):
  binary ``success()`` for throw. ``in_bucket`` = cube center in the
  physical cylinder. ``nvidia_goal`` = one-frame NVIDIA ball around
  (bucket xy, floor + 5 cm), radius ``success_tolerance * keypoint_scale``
  (0.075 × 1.5 = 0.1125 m). ``goal_ball`` = the same ball with radius
  ``success_tolerance`` only (7.5 cm). Neither mode resets or respawns
  on success.

  Hindsight is the last ``goal_dim`` entries of the state, except
  ``joint_goal`` which uses ``JOINT_GOAL_STATE_INDICES`` (hand q + object).
  The default throw bucket is swapped for the slide lip when ``palm_goal``
  is on (and ``table_push`` is off) and the caller left ``fixed_target_xyz``
  at the throw default.

Everything is kept on-GPU as torch CUDA tensors; the learner converts them to
JAX via DLPack (zero-copy).  We deliberately do NOT go through numpy per step.

IMPORTANT: ``import isaacgym`` (and this module) must be imported *before*
``torch`` anywhere in the process, per Isaac Gym's requirement.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import isaacgym  # noqa: F401  — must precede torch import
from isaacgym import gymapi  # noqa: F401
import torch
from torch import Tensor

from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_throw import AllegroKukaThrow


# Fixed world-frame target (bucket) location.  Within the upstream
# randomization ranges (x in [-0.9, 0.9], y in [-1.0, 0.7], z in [0.0, 1.0]).
# Chosen as a moderate, reliably-throwable target off to one side of the table.
DEFAULT_FIXED_TARGET_XYZ: Tuple[float, float, float] = (0.5, -0.3, 0.4)
# Slide lip: bucket against table +x / robot-near edge, rim 2 cm below top.
# Matches scripts/allegro_kuka_throw_palm_goal_viz.py --scene=slide.
SLIDE_BUCKET_XYZ: Tuple[float, float, float] = (0.37, 0.08, 0.312)
SLIDE_PALM_XYZ: Tuple[float, float, float] = (0.17, 0.08, 0.57)
OBJECT_ABOVE_BUCKET = 0.05
# NVIDIA bucket.obj: origin at the floor, height ≈ 0.198 m, xy radius ≈ 0.12 m.
# Default env.success() is this cylinder. ``nvidia_goal`` uses the 3D ball
# around (xy, floor + 0.05) with radius success_tolerance * keypoint_scale.
# ``goal_ball`` is that ball at success_tolerance only (7.5 cm).
BUCKET_HEIGHT = 0.198
BUCKET_INNER_RADIUS = 0.12
THROW_SUCCESS_IN_BUCKET = "in_bucket"
THROW_SUCCESS_NVIDIA_GOAL = "nvidia_goal"
THROW_SUCCESS_GOAL_BALL = "goal_ball"
THROW_SUCCESS_MODES = (
    THROW_SUCCESS_IN_BUCKET,
    THROW_SUCCESS_NVIDIA_GOAL,
    THROW_SUCCESS_GOAL_BALL,
)
# NVIDIA table_narrow: 0.475 x 0.4 x 0.3 box at (0, 0, 0.38) → top z = 0.53.
# Spawn is table center (0, 0); robot base is (0, 0.8, 0).
LARGE_TABLE_ASSET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "urdf", "allegro_table_1p5x1p5_away.urdf")
HIDDEN_TABLE_ASSET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "urdf", "allegro_table_hidden.urdf")
# Park the throw bucket with the hidden table (out of camera / workspace).
HIDDEN_BUCKET_XYZ: Tuple[float, float, float] = (10.0, 10.0, -2.0)
# keep_arm palm-up pad-hold start (no noise). Used to frame videos.
KEEP_ARM_PALM_APPROX: Tuple[float, float, float] = (0.00, 0.08, 0.72)
LARGE_TABLE_SIZE_XY: Tuple[float, float] = (1.5, 1.5)
# Keep the original robot-facing edge at y=+0.2; expand toward -y.
LARGE_TABLE_CENTER_XY: Tuple[float, float] = (0.0, -0.55)
LARGE_TABLE_BOUNDS_XY: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (-0.75, 0.75), (-1.30, 0.20))
# Table-push tasks do not use the bucket. The default parked location becomes
# part of the enlarged tabletop, so move the bucket beyond its +x edge.
LARGE_TABLE_PARKED_BUCKET_XYZ: Tuple[float, float, float] = (1.10, -0.30, 0.40)
# Nudge-only goal: 10 cm toward the robot. success_tolerance is 7.5 cm, so
# the cube is *not* already successful at spawn, but a short shove is enough.
TABLE_TOP_Z = 0.38 + 0.15
TABLE_OBJECT_Z = TABLE_TOP_Z + 0.025  # ~5 cm cube sitting on the desk
# Optional fly-away reset (``--isaacgym_reset_z_above``). 0 = off. NVIDIA
# already ends the episode when the cube falls (object z < 0.1).
OBJECT_FLY_RESET_Z = 1.0
TABLE_PUSH_GOAL_XYZ: Tuple[float, float, float] = (0.0, 0.10, TABLE_OBJECT_Z)
# On-table, toward the throw bucket (0.5, −0.3). Table half-extents are
# 0.2375 × 0.2; this sits ~3 cm inside the +x/−y edge. Spawn (0,0) is 25 cm
# away. With ±0.1 m xy init noise the closest spawn is still ~11 cm (> 7.5 cm).
TABLE_SIDE_GOAL_XYZ: Tuple[float, float, float] = (0.20, -0.15, TABLE_OBJECT_Z)
# Palm hover above the table-side object target (~6.5 cm, above the cube).
TABLE_SIDE_PALM_XYZ: Tuple[float, float, float] = (0.20, -0.15, TABLE_OBJECT_Z + 0.065)
# table_spawn: cube on the desk at the slide-palm xy (proven reachable by
# the palm-goal viz IK). Table center is at the iiwa7 reach limit. Palm is
# IK'd onto the cube, then small reset noise is applied to object xy / joints.
TABLE_SPAWN_OBJECT_XY: Tuple[float, float] = (0.17, 0.08)
TABLE_SPAWN_XY = 0.02
TABLE_SPAWN_ARM_NOISE = 0.008
TABLE_SPAWN_FINGER_NOISE = 0.02
TABLE_SPAWN_PALM_ABOVE = 0.025  # palm on cube top (center z + half-height)
TABLE_SPAWN_MAX_PALM_OBJ = 0.12
TABLE_SPAWN_PALM_Z = TABLE_OBJECT_Z + TABLE_SPAWN_PALM_ABOVE
# Palm behind the cube along +y (robot side) so a −y shove is a straight push.
# Cube x matches the table-side goal. Off unless table_spawn_behind=True.
TABLE_SPAWN_BEHIND_OBJECT_XY: Tuple[float, float] = (0.20, 0.08)
# 14 cm behind + 8 cm above: Allegro fingers are ~10 cm, so 8 cm of
# palm-center gap still spears the cube.
TABLE_SPAWN_BEHIND_PALM_DY = 0.14
TABLE_SPAWN_BEHIND_PALM_ABOVE = 0.08
TABLE_SPAWN_BEHIND_MAX_PALM_OBJ = 0.20
# Palm faces the cube and slightly down (paddle, not a finger-poke).
TABLE_SPAWN_BEHIND_Z_DES: Tuple[float, float, float] = (0.0, -0.75, -0.66)
# table_spawn_toward_bucket (standoff18): palm on the anti-bucket side of
# TABLE_SPAWN_OBJECT_XY so a shove/throw goes toward DEFAULT_FIXED_TARGET.
TABLE_SPAWN_BUCKET_STANDOFF = 0.18
TABLE_SPAWN_BUCKET_ABOVE = 0.08
TABLE_SPAWN_BUCKET_MAX_PALM_OBJ = 0.25
# in_hand (IK hover): cube center along palm +z (into the fingers).
TABLE_SPAWN_IN_HAND_OFFSET = 0.042
TABLE_SPAWN_IN_HAND_OBJ_NOISE = 0.012  # ±1.2 cm in the palm tangent plane
TABLE_SPAWN_IN_HAND_MAX_PALM_OBJ = 0.10
TABLE_SPAWN_IN_HAND_ABOVE = 0.07
# keep_arm palm-up: sit ON the pad (world +Z), not along finger-axis
# palm_z. Cube half is ~2.5 cm; 3.4 cm clears the palm mesh. A short
# slide toward the fingers puts it in the cup. Re-snap for N settle
# steps so contacts do not explode.
TABLE_SPAWN_IN_HAND_PAD_UP = 0.034
TABLE_SPAWN_IN_HAND_PAD_ALONG = 0.012
TABLE_SPAWN_IN_HAND_SETTLE_STEPS = 8
# keep_arm: NVIDIA pose v1 is palm-down. −π rolls A7 to palm-up so the
# cube can sit in the hand. A7 stays inside ±3.05.
TABLE_SPAWN_IN_HAND_WRIST_OFFSET = -math.pi
TABLE_SPAWN_IN_HAND_WRIST_NOISE = 0.10
# NVIDIA throw pose v1 (allegro_kuka_base desired_kuka_pos).
THROW_DEFAULT_ARM_Q: Tuple[float, ...] = (
    -1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 2.358,
)
# Palm-up hold sits near (0.00, 0.04, 0.72). Goal is on the desk, ~26 cm
# in front (−y): far enough to throw, still inside iiwa reach (same
# band as TABLE_SIDE_GOAL y=−0.15).
IN_HAND_KEEP_ARM_GOAL_XYZ: Tuple[float, float, float] = (
    0.00, -0.22, TABLE_OBJECT_Z)
# Curled paddle (4 joints × index/mid/ring + thumb). Zeros spear the cube.
TABLE_SPAWN_BEHIND_FINGER_Q: Tuple[float, ...] = (
    0.55, 0.75, 0.65, 0.35,
    0.55, 0.75, 0.65, 0.35,
    0.55, 0.75, 0.65, 0.35,
    0.40, 0.70, 0.55, 0.35,
)
# tight10 IK (2026-09-02): palm 10 cm +y of TABLE_SIDE_GOAL, 5 cm above cube
# center, paddle toward −y. Dump from goal_joint_tight10_q.txt.
TABLE_SIDE_JOINT_GOAL_Q: Tuple[float, ...] = (
    -1.303683, 1.059191, -0.176880, 0.176791, -0.070335, 1.156637, 2.209153,
    0.542279, 0.761501, 0.649947, 0.349950,
    0.548979, 0.726916, 0.784531, 0.348627,
    0.540559, 0.799928, 0.654862, 0.360575,
    0.415613, 0.702510, 0.541933, 0.348866,
)
CONTROL_SANITY_FINGER_GOAL_Q: Tuple[float, ...] = (
    TABLE_SIDE_JOINT_GOAL_Q[7:23])
# Mild all-hand (16-D) pose for a simple full-finger control sanity.
CONTROL_SANITY_HAND16_GOAL_Q: Tuple[float, ...] = (
    0.15, 0.40, 0.25, 0.12,  # index
    0.12, 0.38, 0.25, 0.12,  # middle
    0.10, 0.35, 0.22, 0.10,  # ring
    0.30, 0.35, 0.25, 0.12,  # thumb (base >=~0.28 URDF)
)
# Hard 16-DOF anti-synergy zigzag: each finger a different pattern,
# far from the open-hand init. Values sit inside Allegro URDF limits.
CONTROL_SANITY_HAND16_FIGURE_GOAL_Q: Tuple[float, ...] = (
    0.48, 1.55, 0.12, 1.48,   # index: proximal+tip curl, mid almost straight
    -0.50, 0.10, 1.58, 0.10,  # middle: abducted out, mid-only curl
    0.45, 1.52, 1.58, 0.12,   # ring: almost-fist, tip open
    1.50, 0.12, 1.55, 0.15,   # thumb: max oppose, proximal open, mid curl
)
# Loose OK-circle (index+thumb) with middle+ring open. NF 3858696 reached
# real train success on this pose before we cancelled it.
CONTROL_SANITY_HAND16_OK_GOAL_Q: Tuple[float, ...] = (
    0.18, 1.05, 1.15, 0.85,  # index
    -0.38, 0.06, 0.02, 0.00,  # middle open
    0.22, 0.08, 0.04, 0.00,  # ring open
    1.00, 0.55, 0.80, 0.65,  # thumb
)
# V-sign / peace: index+middle open and spread, ring curled, thumb tucked.
# Mid-range (not zigzag limits). No index-thumb meet (not OK).
CONTROL_SANITY_HAND16_PEACE_GOAL_Q: Tuple[float, ...] = (
    0.28, 0.10, 0.05, 0.02,   # index open, +abduct
    -0.32, 0.10, 0.05, 0.02,  # middle open, abduct away
    0.18, 1.15, 1.25, 1.05,   # ring curled
    0.65, 0.95, 0.80, 0.55,   # thumb tucked
)
# Point: only index open; middle+ring curled; thumb mid (no OK meet).
# Harder than peace (two curled fingers instead of one). Mid-range.
CONTROL_SANITY_HAND16_POINT_GOAL_Q: Tuple[float, ...] = (
    0.10, 0.08, 0.04, 0.02,   # index open
    0.20, 1.20, 1.30, 1.10,   # middle curled
    0.18, 1.18, 1.28, 1.08,   # ring curled
    0.50, 0.75, 0.65, 0.45,   # thumb mid
)
# Finger-gun: index out, thumb out (high oppose, other joints open),
# middle+ring curled harder than point. Three independent roles.
# Mid-range (not zigzag limits). No OK-style fingertip meet.
CONTROL_SANITY_HAND16_GUN_GOAL_Q: Tuple[float, ...] = (
    0.08, 0.08, 0.04, 0.02,   # index open (barrel)
    0.22, 1.28, 1.38, 1.18,   # middle curled
    0.20, 1.26, 1.36, 1.16,   # ring curled
    1.15, 0.22, 0.18, 0.12,   # thumb out
)
# Whole-arm greeting wave: 7 iiwa + 16 Allegro. Init is NVIDIA pose v1
# (arm folded over the table: A2=1.571, A4=1.376). Goal stands the arm
# up (A2→0, elbow almost straight) with an open "hi" hand.
CONTROL_SANITY_ARM23_WAVE_GOAL_Q: Tuple[float, ...] = (
    -1.571, 0.20, 0.00, -0.35, 0.00, 0.80, 1.57,  # arm standing up
    0.08, 0.45, 0.35, 0.20,   # index
    -0.12, 0.42, 0.32, 0.18,  # middle
    0.10, 0.40, 0.30, 0.16,   # ring
    0.70, 0.40, 0.30, 0.18,   # thumb
)
CONTROL_SANITY_INDEX_GOAL_Q: Tuple[float, ...] = (0.20, 0.50, 0.30, 0.15)
CONTROL_SANITY_TWO_FINGER_GOAL_Q: Tuple[float, ...] = (
    0.20, 0.50, 0.30, 0.15,
    0.15, 0.55, 0.35, 0.20,
)
# Intermediate between index (4-D) and two_finger (8-D): full index +
# middle base/proximal only.
CONTROL_SANITY_SIX_GOAL_Q: Tuple[float, ...] = (
    0.20, 0.50, 0.30, 0.15,
    0.15, 0.55,
)
# Two joints per finger (base + proximal). Hand-local indices:
# index 0,1 / middle 4,5 / ring 8,9 / thumb 12,13.
CONTROL_SANITY_THREE2_HAND_INDICES: Tuple[int, ...] = (0, 1, 4, 5, 8, 9)
CONTROL_SANITY_THREE2_GOAL_Q: Tuple[float, ...] = (
    0.25, 0.55,  # index
    0.20, 0.50,  # middle
    0.15, 0.45,  # ring
)
CONTROL_SANITY_FOUR2_HAND_INDICES: Tuple[int, ...] = (
    0, 1, 4, 5, 8, 9, 12, 13)
CONTROL_SANITY_FOUR2_GOAL_Q: Tuple[float, ...] = (
    0.25, 0.55,  # index
    0.20, 0.50,  # middle
    0.15, 0.45,  # ring
    0.30, 0.40,  # thumb
)
# Same 8 DOFs as four2, much stronger curls (RND crushed mild four2 ~40M).
CONTROL_SANITY_FOUR2H_GOAL_Q: Tuple[float, ...] = (
    0.45, 1.10,  # index
    0.40, 1.00,  # middle
    0.35, 0.95,  # ring
    0.55, 0.90,  # thumb
)
# Between mild four2 (both solvable; RND too easy) and four2h (both fail).
CONTROL_SANITY_FOUR2M_GOAL_Q: Tuple[float, ...] = (
    0.35, 0.82,  # index
    0.30, 0.75,  # middle
    0.25, 0.70,  # ring
    0.42, 0.65,  # thumb
)
# Between four2m (RND too easy @60M) and four2h (both fail).
CONTROL_SANITY_FOUR2MH_GOAL_Q: Tuple[float, ...] = (
    0.42, 1.00,  # index
    0.37, 0.92,  # middle
    0.32, 0.87,  # ring
    0.50, 0.82,  # thumb
)
# Midpoint between four2m (RND easy) and four2mh (both fail @200M).
CONTROL_SANITY_FOUR2MM_GOAL_Q: Tuple[float, ...] = (
    0.39, 0.91,  # index
    0.34, 0.84,  # middle
    0.29, 0.79,  # ring
    0.46, 0.74,  # thumb
)
# Between four2mm (RND too easy @40M) and four2mh (both fail @200M).
CONTROL_SANITY_FOUR2MMH_GOAL_Q: Tuple[float, ...] = (
    0.41, 0.96,  # index
    0.36, 0.88,  # middle
    0.30, 0.83,  # ring
    0.48, 0.78,  # thumb
)
# Midpoint of four2mm (RND easy@40M) and four2mmh (both fail@200M).
CONTROL_SANITY_FOUR2MMX_GOAL_Q: Tuple[float, ...] = (
    0.40, 0.935,  # index
    0.35, 0.86,   # middle
    0.295, 0.81,  # ring
    0.47, 0.76,   # thumb
)
# Straight index + thumb task. The thumb base stays just above its ≈0.279-rad
# lower limit; middle and ring are not controlled.
CONTROL_SANITY_INDEX_THUMB_HAND_INDICES: Tuple[int, ...] = (
    0, 1, 2, 3, 12, 13, 14, 15)
CONTROL_SANITY_INDEX_THUMB_STRAIGHT_GOAL_Q: Tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0,  # index
    0.30, 0.0, 0.0, 0.0,  # thumb
)
# Existing moderately curled table-side pose, restricted to index + thumb.
CONTROL_SANITY_INDEX_THUMB_CURLED_RESET_Q: Tuple[float, ...] = tuple(
    CONTROL_SANITY_FINGER_GOAL_Q[i]
    for i in CONTROL_SANITY_INDEX_THUMB_HAND_INDICES)
# Asymmetric "point": index open, middle+ring curled, thumb as open as
# URDF allows (thumb base lower limit ≈0.279).
CONTROL_SANITY_FOUR2W_GOAL_Q: Tuple[float, ...] = (
    0.05, 0.10,  # index open (point)
    0.50, 1.15,  # middle curled
    0.45, 1.10,  # ring curled
    0.30, 0.20,  # thumb near-open (base cannot go below ~0.28)
)
CONTROL_SANITY_PALM_GOAL_XYZ: Tuple[float, float, float] = (0.0, 0.0, 0.80)
# The upstream task requires an object actor for its fixed tensor layout. In
# control-only sanity modes it is parked outside the robot/camera workspace.
CONTROL_SANITY_PARKED_OBJECT_XYZ: Tuple[float, float, float] = (
    -0.65, -1.15, TABLE_OBJECT_Z)
# Randomized trim-SA starts vary only controlled joint positions by this
# fraction of each joint's full URDF range (``trim_init_mode=curled``).
CONTROL_SANITY_TRIM_INIT_RANGE_FRAC = 0.10
# ``curled``: center ± range_frac·(hi−lo). ``full_range``: independent
# Uniform[lo, hi] per controlled joint (no curled-center bias).
CONTROL_SANITY_TRIM_INIT_MODES = ("curled", "full_range")
CONTROL_SANITY_BALANCED_MEAN_TOL = 0.15
CONTROL_SANITY_BALANCED_MAX_TOL = 0.30
CONTROL_SANITY_RESET_REJECTION_MAX_TRIES = 32


def _skew(v: Tensor) -> Tensor:
    z = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    x, y, zz = v[:, 0], v[:, 1], v[:, 2]
    row0 = torch.stack([z, -zz, y], dim=-1)
    row1 = torch.stack([zz, z, -x], dim=-1)
    row2 = torch.stack([-y, x, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

# Observation layout constants.
_NUM_ARM_DOFS = 7
_NUM_HAND_DOFS = 16
_NUM_ARM_HAND_DOFS = _NUM_ARM_DOFS + _NUM_HAND_DOFS  # 23
STATE_DIM = 2 * _NUM_ARM_HAND_DOFS + 3   # q + qd + object xyz = 49
GOAL_DIM = 3                              # bucket xyz
PACKED_OBS_DIM = STATE_DIM + GOAL_DIM     # 52
STATE_DIM_CONTROL = 2 * _NUM_ARM_HAND_DOFS       # q + qd = 46
STATE_DIM_PALM_CONTROL = STATE_DIM_CONTROL + 3   # q + qd + palm xyz = 49
GOAL_DIM_FINGER_CONTROL = _NUM_HAND_DOFS         # finger q* = 16
GOAL_DIM_INDEX_CONTROL = 4                       # index-finger q* = 4
GOAL_DIM_SIX_CONTROL = 6                         # index 4 + middle base/prox
GOAL_DIM_THREE2_CONTROL = 6                      # 3 fingers × base+prox
GOAL_DIM_TWO_FINGER_CONTROL = 8                  # index + middle q* = 8
GOAL_DIM_FOUR2_CONTROL = 8                       # 4 fingers × base+prox
GOAL_DIM_INDEX_THUMB_CONTROL = 8                 # full index + thumb q* = 8
GOAL_DIM_PALM_CONTROL = 3                        # palm xyz = 3
CONTROL_SANITY_LEVEL_MODES = (
    "index", "six", "two_finger", "three2", "four2", "four2h", "four2m",
    "four2mh", "four2mm", "four2mmh", "four2mmx", "four2w",
    "index_thumb_straight", "hand16", "hand16fig", "hand16ok",
    "hand16peace", "hand16point", "hand16gun", "arm23wave")
CONTROL_SANITY_ARM23_MODES = ("arm23wave",)
GOAL_DIM_ARM23_CONTROL = _NUM_ARM_HAND_DOFS  # all 23 q*
STATE_DIM_PALM = 2 * _NUM_ARM_HAND_DOFS + 6  # q + qd + palm + object = 52
GOAL_DIM_PALM = 6                             # ideal palm + object in bucket
PACKED_OBS_DIM_PALM = STATE_DIM_PALM + GOAL_DIM_PALM  # 58
GOAL_DIM_JOINT = _NUM_HAND_DOFS + 3           # hand q* + object xyz = 19
PACKED_OBS_DIM_JOINT = STATE_DIM + GOAL_DIM_JOINT  # 68
# Hindsight for joint_goal: achieved hand q and achieved object (skip arm / qd).
JOINT_GOAL_STATE_INDICES: Tuple[int, ...] = tuple(range(7, 23)) + (46, 47, 48)
ACTION_DIM = _NUM_ARM_HAND_DOFS           # 23
# Option A: freeze iiwa A1–A4, free wrist A5–A7 + 16 fingers.
LOCK_ARM_BASE_N_PROXIMAL = 4
LOCK_ARM_BASE_ACTION_DIM = ACTION_DIM - LOCK_ARM_BASE_N_PROXIMAL  # 19
if len(JOINT_GOAL_STATE_INDICES) != GOAL_DIM_JOINT:
    raise RuntimeError(
        'JOINT_GOAL_STATE_INDICES length '
        f'{len(JOINT_GOAL_STATE_INDICES)} != GOAL_DIM_JOINT {GOAL_DIM_JOINT}')
if len(TABLE_SIDE_JOINT_GOAL_Q) != _NUM_ARM_HAND_DOFS:
    raise RuntimeError(
        'TABLE_SIDE_JOINT_GOAL_Q length '
        f'{len(TABLE_SIDE_JOINT_GOAL_Q)} != {_NUM_ARM_HAND_DOFS}')
if len(CONTROL_SANITY_ARM23_WAVE_GOAL_Q) != _NUM_ARM_HAND_DOFS:
    raise RuntimeError(
        'CONTROL_SANITY_ARM23_WAVE_GOAL_Q length '
        f'{len(CONTROL_SANITY_ARM23_WAVE_GOAL_Q)} != {_NUM_ARM_HAND_DOFS}')


def resolve_bucket_xyz(
    palm_goal: bool,
    fixed_target_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """If palm-goal is on and the throw default bucket was left in place, use the slide lip."""
    tgt = tuple(float(v) for v in fixed_target_xyz)
    if bool(palm_goal) and tgt == DEFAULT_FIXED_TARGET_XYZ:
        return SLIDE_BUCKET_XYZ
    return tgt


def frame_keep_arm_camera(
    points: List[Tuple[float, float, float]],
    hfov_deg: float = 48.0,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Side/elevated camera that keeps the start palm, arm, and goals in frame.

    The old keep-arm cam looks at (0, -0.08) with HFOV ~42–48°, so a bucket
    at y=−0.80 sits ~29° off-axis and is cropped.
    """
    pts = [tuple(float(v) for v in p) for p in points]
    pts.append(KEEP_ARM_PALM_APPROX)
    # Toward the base so the iiwa links stay in view, not just the hand.
    pts.append((0.00, 0.55, 0.55))
    cx = sum(p[0] for p in pts) / float(len(pts))
    cy = sum(p[1] for p in pts) / float(len(pts))
    cz = sum(p[2] for p in pts) / float(len(pts))
    span = 0.0
    for p in pts:
        span = max(
            span,
            math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2))
    span = max(span, 0.35) + 0.28
    half = math.radians(float(hfov_deg) * 0.5)
    dist = max(1.65, span / max(math.tan(half), 1e-3) * 1.28)
    # From +x / +y / up: −y goals stay left of frame, robot on the right.
    ox, oy, oz = 0.82, 0.42, 0.40
    norm = math.sqrt(ox * ox + oy * oy + oz * oz)
    eye = (cx + ox / norm * dist, cy + oy / norm * dist, cz + oz / norm * dist)
    return eye, (cx, cy, cz)


def object_in_bucket_xyz(
    bucket_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    b = tuple(float(v) for v in bucket_xyz)
    return (b[0], b[1], b[2] + OBJECT_ABOVE_BUCKET)


class FixedGoalAllegroKukaThrow(AllegroKukaThrow):
    """AllegroKukaThrow with a single fixed bucket / goal position.

    The fixed target is read from ``cfg['env']['fixedTargetXYZ']`` if present,
    else falls back to :data:`DEFAULT_FIXED_TARGET_XYZ`.
    """

    def _hand_cube_contact_pairs(self, env_idx: int = 0):
        object_bodies = {int(v) for v in self.object_rb_handles.tolist()}
        hand_bodies = set(range(int(self.num_hand_arm_bodies)))
        pairs = []
        for contact in self.gym.get_env_rigid_contacts(self.envs[env_idx]):
            try:
                b0, b1 = int(contact["body0"]), int(contact["body1"])
            except (KeyError, TypeError):
                b0, b1 = int(contact.body0), int(contact.body1)
            if ((b0 in object_bodies and b1 in hand_bodies)
                    or (b1 in object_bodies and b0 in hand_bodies)):
                pairs.append((b0, b1))
        return pairs

    def __init__(self, cfg, *args, **kwargs):
        env_cfg = cfg.get("env", {})
        tgt = env_cfg.get("fixedTargetXYZ", DEFAULT_FIXED_TARGET_XYZ)
        # Store as a plain tuple; tensors are built lazily (device not ready yet).
        self._fixed_target_xyz = tuple(float(v) for v in tgt)
        self._fixed_target_tensor: Optional[Tensor] = None
        goal = env_cfg.get("fixedGoalXYZ", None)
        if goal is None:
            self._fixed_goal_xyz: Optional[Tuple[float, float, float]] = None
        else:
            self._fixed_goal_xyz = tuple(float(v) for v in goal)
            if len(self._fixed_goal_xyz) != 3:
                raise ValueError(
                    f'fixedGoalXYZ must be 3 floats, got {goal!r}')
        self._fixed_goal_tensor: Optional[Tensor] = None
        # Must be set before super().__init__ (reset_idx runs during construction).
        self._freeze_init = not bool(env_cfg.get("randomizeInit", True))
        self._control_sanity_mode = str(
            env_cfg.get("controlSanityMode", "") or "").strip().lower()
        self._control_sanity_trim_sa = bool(
            env_cfg.get("controlSanityTrimSA", False))
        self._control_sanity_trim_init_range_frac = float(
            env_cfg.get(
                "controlSanityTrimInitRangeFrac",
                CONTROL_SANITY_TRIM_INIT_RANGE_FRAC))
        if not 0.0 <= self._control_sanity_trim_init_range_frac <= 1.0:
            raise ValueError(
                "controlSanityTrimInitRangeFrac must be in [0, 1], got "
                f"{self._control_sanity_trim_init_range_frac}")
        self._control_sanity_trim_init_mode = str(
            env_cfg.get("controlSanityTrimInitMode", "curled")
            or "curled").strip().lower()
        if self._control_sanity_trim_init_mode not in (
                CONTROL_SANITY_TRIM_INIT_MODES):
            raise ValueError(
                "controlSanityTrimInitMode must be one of "
                f"{CONTROL_SANITY_TRIM_INIT_MODES}, got "
                f"{self._control_sanity_trim_init_mode!r}")
        self._trim_init_rejected_total = 0
        self._trim_init_sampled_total = 0
        _fullrange_hand = bool(
            self._control_sanity_mode in (
                "finger", "hand16", "hand16fig", "hand16ok", "hand16peace",
                "hand16point", "hand16gun", "arm23wave")
            and self._control_sanity_trim_init_mode == "full_range"
            and not self._freeze_init)
        self._control_sanity_random_trim_init = bool(
            (self._control_sanity_mode
             and self._control_sanity_trim_sa
             and not self._freeze_init)
            or _fullrange_hand)
        if self._control_sanity_mode not in (
                "", "finger", "index", "six", "two_finger",
                "three2", "four2", "four2h", "four2m", "four2mh", "four2mm",
                "four2mmh", "four2mmx", "four2w", "index_thumb_straight",
                "hand16", "hand16fig", "hand16ok", "hand16peace",
                "hand16point", "hand16gun", "arm23wave", "palm"):
            raise ValueError(
                "controlSanityMode must be '', 'finger', 'index', "
                "'six', 'two_finger', 'three2', 'four2', 'four2h', 'four2m', "
                "'four2mh', 'four2mm', 'four2mmh', 'four2mmx', 'four2w', "
                "'index_thumb_straight', 'hand16', 'hand16fig', "
                "'hand16ok', 'hand16peace', 'hand16point', 'hand16gun', "
                "'arm23wave', or 'palm'")
        self._table_spawn = bool(env_cfg.get("tableSpawn", False))
        self._table_spawn_behind = bool(env_cfg.get("tableSpawnBehind", False))
        self._table_spawn_toward_bucket = bool(
            env_cfg.get("tableSpawnTowardBucket", False))
        spawn_xy = env_cfg.get("tableSpawnObjectXY", TABLE_SPAWN_OBJECT_XY)
        self._table_spawn_object_xy = tuple(float(v) for v in spawn_xy)
        if len(self._table_spawn_object_xy) != 2:
            raise ValueError(
                f"tableSpawnObjectXY must have 2 values, got {spawn_xy!r}")
        self._table_spawn_behind_dy = float(
            env_cfg.get("tableSpawnBehindDY", TABLE_SPAWN_BEHIND_PALM_DY))
        self._table_spawn_behind_above = float(
            env_cfg.get(
                "tableSpawnBehindAbove", TABLE_SPAWN_BEHIND_PALM_ABOVE))
        self._table_spawn_correlated_xy = float(
            env_cfg.get("tableSpawnCorrelatedXY", 0.0) or 0.0)
        if self._table_spawn_correlated_xy < 0.0:
            raise ValueError("tableSpawnCorrelatedXY must be non-negative")
        self._table_spawn_finger_curl_scale = float(
            env_cfg.get("tableSpawnFingerCurlScale", 1.0))
        if not 0.0 <= self._table_spawn_finger_curl_scale <= 1.0:
            raise ValueError("tableSpawnFingerCurlScale must be in [0, 1]")
        self._table_spawn_finger_noise = float(
            env_cfg.get("tableSpawnFingerNoise", 0.0) or 0.0)
        if not 0.0 <= self._table_spawn_finger_noise <= 1.0:
            raise ValueError("tableSpawnFingerNoise must be in [0, 1]")
        self._table_spawn_arm_noise = float(
            env_cfg.get("tableSpawnArmNoise", 0.0) or 0.0)
        if not 0.0 <= self._table_spawn_arm_noise <= 1.0:
            raise ValueError("tableSpawnArmNoise must be in [0, 1]")
        self._in_hand_center: Optional[Tensor] = None
        self._in_hand_x: Optional[Tensor] = None
        self._in_hand_y: Optional[Tensor] = None
        self._table_spawn_in_hand = bool(env_cfg.get("tableSpawnInHand", False))
        self._table_spawn_in_hand_offset = float(
            env_cfg.get("tableSpawnInHandOffset", TABLE_SPAWN_IN_HAND_OFFSET))
        if self._table_spawn_in_hand_offset <= 0.0:
            raise ValueError("tableSpawnInHandOffset must be positive")
        self._table_spawn_in_hand_obj_noise = float(
            env_cfg.get("tableSpawnInHandObjNoise", TABLE_SPAWN_IN_HAND_OBJ_NOISE)
            or 0.0)
        if self._table_spawn_in_hand_obj_noise < 0.0:
            raise ValueError("tableSpawnInHandObjNoise must be non-negative")
        if self._table_spawn_in_hand and not self._table_spawn:
            raise ValueError("tableSpawnInHand requires tableSpawn")
        self._table_spawn_in_hand_keep_arm = bool(
            env_cfg.get("tableSpawnInHandKeepArm", False))
        self._table_spawn_in_hand_wrist_offset = float(
            env_cfg.get(
                "tableSpawnInHandWristOffset",
                TABLE_SPAWN_IN_HAND_WRIST_OFFSET))
        self._table_spawn_in_hand_wrist_noise = float(
            env_cfg.get(
                "tableSpawnInHandWristNoise",
                TABLE_SPAWN_IN_HAND_WRIST_NOISE) or 0.0)
        if self._table_spawn_in_hand_wrist_noise < 0.0:
            raise ValueError("tableSpawnInHandWristNoise must be non-negative")
        if self._table_spawn_in_hand_keep_arm and not self._table_spawn_in_hand:
            raise ValueError("tableSpawnInHandKeepArm requires tableSpawnInHand")
        # Behind-pose paddle downward tilt: z_des = normalize((0,-1,-tilt)).
        # tilt~0.88 reproduces the legacy (0,-0.75,-0.66); lower is more
        # vertical so fingertips press the cube's back face at table height.
        _tilt_default = (
            abs(TABLE_SPAWN_BEHIND_Z_DES[2]) /
            (abs(TABLE_SPAWN_BEHIND_Z_DES[1]) + 1e-12))
        self._table_spawn_behind_tilt = float(
            env_cfg.get("tableSpawnBehindTilt", _tilt_default))
        if self._table_spawn_behind_tilt < 0.0:
            raise ValueError("tableSpawnBehindTilt must be non-negative")
        self._large_table = bool(env_cfg.get("largeTable", False))
        self._hide_table = bool(env_cfg.get("hideTable", False))
        if self._hide_table and self._large_table:
            raise ValueError("hideTable and largeTable are mutually exclusive")
        if self._table_spawn_toward_bucket:
            # Oriented throw stand-off wins over the desk-push behind pose.
            self._table_spawn_behind = False
        self._reset_z_above = float(env_cfg.get("resetObjectZAbove", 0.0) or 0.0)
        self._last_fall = None
        super().__init__(cfg, *args, **kwargs)
        if self._table_spawn:
            from isaacgym import gymtorch

            keep_arm = bool(
                getattr(self, "_table_spawn_in_hand_keep_arm", False))
            if keep_arm:
                self._init_in_hand_keep_arm()
            else:
                # One fixed offset per vectorized environment. Both the cube's
                # reset state and the batched IK target use this same position,
                # so the hand follows the randomized cube rather than missing it.
                if self._table_spawn_correlated_xy > 0.0:
                    offsets = (
                        2.0 * torch.rand(
                            (self.num_envs, 2), device=self.device) - 1.0
                    ) * self._table_spawn_correlated_xy
                    self.object_init_state[:, 0:2] += offsets
                    print(
                        '[allegro_kuka_throw] correlated table-spawn xy '
                        f'range=±{self._table_spawn_correlated_xy:g} '
                        f'actual_min={tuple(round(float(v), 4) for v in offsets.min(dim=0).values.tolist())} '
                        f'actual_max={tuple(round(float(v), 4) for v in offsets.max(dim=0).values.tolist())}',
                        flush=True,
                    )
                self._ik_palm_onto_cube()
                self.arm_hand_dof_pos[:, :] = self.hand_arm_default_dof_pos
                self.cur_targets[:, :] = self.hand_arm_default_dof_pos
                self.prev_targets[:, :] = self.hand_arm_default_dof_pos
                self._set_object_root(None)
                self.gym.set_dof_position_target_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.cur_targets))
                self.gym.set_dof_state_tensor(
                    self.sim, gymtorch.unwrap_tensor(self.dof_state))
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.compute_observations()
            self._table_spawn_initial_hand_contacts = (
                self._hand_cube_contact_pairs())
            from isaacgym.torch_utils import quat_rotate
            palm = self.palm_center_pos[0]
            obj = self.object_pos[0]
            dist = float(torch.norm(palm - obj).item())
            z_axis = torch.zeros((1, 3), device=self.device, dtype=torch.float)
            z_axis[:, 2] = 1.0
            z_w = quat_rotate(self._palm_rot[:1], z_axis)[0]
            print(
                f'[allegro_kuka_throw] table_spawn confirm (noiseless) '
                f'palm={tuple(round(float(v), 3) for v in palm.tolist())} '
                f'obj={tuple(round(float(v), 3) for v in obj.tolist())} '
                f'|palm-obj|={dist:.3f}m '
                f'hand_contacts={self._table_spawn_initial_hand_contacts} '
                f'palm_z={tuple(round(float(v), 3) for v in z_w.tolist())} '
                f'arm_q={tuple(round(float(v), 3) for v in self.hand_arm_default_dof_pos[:7].tolist())}',
                flush=True,
            )
            if getattr(self, "_table_spawn_in_hand", False):
                _palm_lim = TABLE_SPAWN_IN_HAND_MAX_PALM_OBJ
            elif getattr(self, "_table_spawn_toward_bucket", False):
                _palm_lim = TABLE_SPAWN_BUCKET_MAX_PALM_OBJ
            elif getattr(self, "_table_spawn_behind", False):
                _palm_lim = TABLE_SPAWN_BEHIND_MAX_PALM_OBJ
            else:
                _palm_lim = TABLE_SPAWN_MAX_PALM_OBJ
            if dist > _palm_lim:
                raise RuntimeError(
                    f'table_spawn palm is {dist:.3f}m from the cube '
                    f'(limit {_palm_lim:.2f}m); '
                    'refusing to train with a bad hover')
            all_ids = torch.arange(self.num_envs, device=self.device)
            self.reset_idx(all_ids)
            self.set_actor_root_state_tensor_indexed()
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.compute_observations()
            palm = self.palm_center_pos[0]
            obj = self.object_pos[0]
            dist_n = float(torch.norm(palm - obj).item())
            print(
                f'[allegro_kuka_throw] table_spawn confirm (reset noise) '
                f'palm={tuple(round(float(v), 3) for v in palm.tolist())} '
                f'obj={tuple(round(float(v), 3) for v in obj.tolist())} '
                f'|palm-obj|={dist_n:.3f}m',
                flush=True,
            )
            if (getattr(self, "_table_spawn_in_hand_keep_arm", False)
                    and dist_n > 0.08):
                raise RuntimeError(
                    f'keep_arm cube left the palm after reset+1 step '
                    f'(|palm-obj|={dist_n:.3f}m); init is not a stable hold')

    def _install_override_table_asset(self, source_path, installed_rel):
        """Copy a tracked table URDF into IsaacGymEnvs assets (no ``..`` paths)."""
        import inspect
        import shutil

        nvidia_task_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(AllegroKukaThrow)))
        nvidia_asset_root = os.path.abspath(
            os.path.join(nvidia_task_dir, "../../../assets"))
        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"Allegro table asset missing: {source_path}")
        installed_path = os.path.join(nvidia_asset_root, installed_rel)
        with open(source_path, "rb") as src_f:
            source_bytes = src_f.read()
        try:
            with open(installed_path, "rb") as dst_f:
                installed_bytes = dst_f.read()
        except OSError:
            installed_bytes = b""
        if installed_bytes != source_bytes:
            tmp_path = f"{installed_path}.{os.getpid()}.tmp"
            shutil.copyfile(source_path, tmp_path)
            os.replace(tmp_path, installed_path)
        self.asset_files_dict["table"] = installed_rel
        return installed_path

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Select the tracked table asset before NVIDIA loads actors."""
        if self._hide_table:
            installed_path = self._install_override_table_asset(
                HIDDEN_TABLE_ASSET, "urdf/sgcrl_allegro_table_hidden.urdf")
            print(
                '[allegro_kuka_throw] hide_table=true '
                f'asset={HIDDEN_TABLE_ASSET} installed={installed_path} '
                '(1cm box parked at +10,+10,−2; workspace has no table)',
                flush=True,
            )
        elif self._large_table:
            installed_path = self._install_override_table_asset(
                LARGE_TABLE_ASSET,
                "urdf/sgcrl_allegro_table_1p5x1p5_away.urdf")
            print(
                '[allegro_kuka_throw] large_table=true '
                f'asset={LARGE_TABLE_ASSET} installed={installed_path} '
                f'size_xy={LARGE_TABLE_SIZE_XY} '
                f'center_xy={LARGE_TABLE_CENTER_XY} '
                f'bounds_xy={LARGE_TABLE_BOUNDS_XY} top_z={TABLE_TOP_Z:.3f}',
                flush=True,
            )
        return super()._create_envs(num_envs, spacing, num_per_row)

    def _object_start_pose(self, allegro_pose, table_pose_dy, table_pose_dz):
        if getattr(self, "_control_sanity_mode", ""):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*CONTROL_SANITY_PARKED_OBJECT_XYZ)
            pose.r = gymapi.Quat(0, 0, 0, 1)
            return pose
        if not getattr(self, "_table_spawn", False):
            return super()._object_start_pose(
                allegro_pose, table_pose_dy, table_pose_dz)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3()
        pose.p.x = self._table_spawn_object_xy[0]
        pose.p.y = self._table_spawn_object_xy[1]
        pose.p.z = TABLE_OBJECT_Z
        pose.r = gymapi.Quat(0, 0, 0, 1)
        return pose

    def _set_object_root(self, xyz: Optional[Tensor] = None) -> None:
        """Teleport the cube (and zero its velocity). ``xyz`` is (E, 3) or None=init."""
        from isaacgym import gymtorch

        idx = self.object_indices
        if xyz is None:
            self.root_state_tensor[idx, 0:7] = self.object_init_state[:, 0:7]
        else:
            self.root_state_tensor[idx, 0:3] = xyz
            self.root_state_tensor[idx, 3:7] = self.object_init_state[:, 3:7]
        self.root_state_tensor[idx, 7:13] = 0.0
        i32 = idx.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(i32),
            int(self.num_envs),
        )

    def _init_in_hand_keep_arm(self) -> None:
        """NVIDIA pose v1 arm, roll the wrist, curl fingers, snap cube in."""
        from isaacgym import gymtorch

        fq = torch.tensor(
            TABLE_SPAWN_BEHIND_FINGER_Q, dtype=torch.float,
            device=self.device) * self._table_spawn_finger_curl_scale
        q = self.hand_arm_default_dof_pos.clone()
        q[:7] = torch.tensor(
            THROW_DEFAULT_ARM_Q, dtype=torch.float, device=self.device)
        lo7 = self.arm_hand_dof_lower_limits[6]
        hi7 = self.arm_hand_dof_upper_limits[6]
        q[6] = torch.clamp(
            q[6] + float(self._table_spawn_in_hand_wrist_offset), lo7, hi7)
        q[7:23] = fq
        self.hand_arm_default_dof_pos[:] = q
        per = q.unsqueeze(0).expand(self.num_envs, -1).clone()
        wrist_j = float(self._table_spawn_in_hand_wrist_noise)
        if wrist_j > 0.0:
            per[:, 6] = torch.clamp(
                per[:, 6] + (
                    2.0 * torch.rand((self.num_envs,), device=self.device)
                    - 1.0
                ) * wrist_j,
                lo7, hi7,
            )
        self._table_spawn_per_env_dof_pos = per
        self.arm_hand_dof_pos[:, :] = per
        self.arm_hand_dof_vel[:, :] = 0.0
        self.cur_targets[:, :] = per
        self.prev_targets[:, :] = per
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        self.gym.set_dof_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_state))
        for _ in range(6):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self.compute_observations()
        self._place_cube_in_hand(write_init=True, use_stored=False)
        self._settle_cube_in_hand()
        self.compute_observations()
        palm0 = self.palm_center_pos[0]
        obj0 = self.object_pos[0]
        print(
            '[allegro_kuka_throw] in_hand keep_arm '
            f'wrist_offset={self._table_spawn_in_hand_wrist_offset:g} '
            f'wrist_noise=±{wrist_j:g} '
            f'curl_scale={self._table_spawn_finger_curl_scale:g} '
            f'arm_q={tuple(round(float(v), 3) for v in q[:7].tolist())} '
            f'palm0={tuple(round(float(v), 3) for v in palm0.tolist())} '
            f'obj0={tuple(round(float(v), 3) for v in obj0.tolist())} '
            f'|palm-obj|={float(torch.norm(palm0 - obj0).item()):.3f}m '
            'settle=construction+VecEnv.reset only (skip on mid-episode reset)',
            flush=True,
        )

    def _place_cube_in_hand(
        self,
        env_ids: Optional[Tensor] = None,
        *,
        write_init: bool = False,
        use_stored: bool = False,
    ) -> None:
        """Teleport the cube into the curled palm.

        ``write_init=True`` stores the noiseless grasp center so later
        resets can jitter around it without re-running IK.
        ``use_stored=True`` places from that stored center (reset path).
        """
        from isaacgym import gymtorch
        from isaacgym.torch_utils import quat_rotate

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = int(env_ids.numel())
        noise = float(self._table_spawn_in_hand_obj_noise)
        if use_stored and getattr(self, "_in_hand_center", None) is not None:
            xyz = self._in_hand_center[env_ids].clone()
            x_w = self._in_hand_x[env_ids]
            y_w = self._in_hand_y[env_ids]
        else:
            self.compute_observations()
            palm = self.palm_center_pos[env_ids]
            z_axis = torch.zeros((n, 3), device=self.device, dtype=torch.float)
            z_axis[:, 2] = 1.0
            x_axis = torch.zeros((n, 3), device=self.device, dtype=torch.float)
            x_axis[:, 0] = 1.0
            y_axis = torch.zeros((n, 3), device=self.device, dtype=torch.float)
            y_axis[:, 1] = 1.0
            rot = self._palm_rot[env_ids]
            z_w = quat_rotate(rot, z_axis)
            x_w = quat_rotate(rot, x_axis)
            y_w = quat_rotate(rot, y_axis)
            off = float(self._table_spawn_in_hand_offset)
            if getattr(self, "_table_spawn_in_hand_keep_arm", False):
                # Palm local +z is along the fingers, not the pad normal.
                # Sitting the cube there intersects the proximal joints and
                # the first PhysX step launches it. Rest it on the pad.
                up = torch.zeros((n, 3), device=self.device, dtype=torch.float)
                up[:, 2] = 1.0
                xyz = (
                    palm
                    + float(TABLE_SPAWN_IN_HAND_PAD_UP) * up
                    + float(TABLE_SPAWN_IN_HAND_PAD_ALONG) * z_w)
            else:
                xyz = palm + off * z_w
            if write_init:
                if getattr(self, "_in_hand_center", None) is None:
                    self._in_hand_center = torch.zeros(
                        (self.num_envs, 3), device=self.device,
                        dtype=torch.float)
                    self._in_hand_x = torch.zeros_like(self._in_hand_center)
                    self._in_hand_y = torch.zeros_like(self._in_hand_center)
                self._in_hand_center[env_ids] = xyz
                self._in_hand_x[env_ids] = x_w
                self._in_hand_y[env_ids] = y_w
        if noise > 0.0:
            u = (2.0 * torch.rand((n, 2), device=self.device) - 1.0) * noise
            xyz = xyz + u[:, 0:1] * x_w + u[:, 1:2] * y_w
        xyz = xyz.clone()
        xyz[:, 2] = torch.clamp(xyz[:, 2], min=TABLE_OBJECT_Z)
        idx = self.object_indices[env_ids]
        if write_init:
            self.object_init_state[env_ids, 0:3] = xyz
        self.root_state_tensor[idx, 0:3] = xyz
        self.root_state_tensor[idx, 3:7] = self.object_init_state[env_ids, 3:7]
        self.root_state_tensor[idx, 7:13] = 0.0
        i32 = idx.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(i32),
            n,
        )

    def _settle_cube_in_hand(
        self,
        env_ids: Optional[Tensor] = None,
        steps: int = TABLE_SPAWN_IN_HAND_SETTLE_STEPS,
    ) -> None:
        """Hold the cube at the stored rest pose while PhysX contacts settle."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n_steps = max(int(steps), 0)
        for _ in range(n_steps):
            self._place_cube_in_hand(
                env_ids, write_init=False, use_stored=True)
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
        self._place_cube_in_hand(env_ids, write_init=False, use_stored=True)

    def _stow_object_for_ik(self) -> None:
        """Park the cube off-table so the palm can IK without collisions."""
        xyz = self.object_init_state[:, 0:3].clone()
        xyz[:, 0] = 3.0
        xyz[:, 1] = 0.0
        xyz[:, 2] = 1.0
        self._set_object_root(xyz)

    def _ik_apply_q(self, q: Tensor) -> None:
        from isaacgym import gymtorch

        self.cur_targets[:, :7] = q
        self.prev_targets[:, :7] = q
        self.arm_hand_dof_pos[:, :7] = q
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        self.gym.set_dof_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._stow_object_for_ik()

    def _ik_palm_stage(
        self,
        jac,
        goal: Tensor,
        n_iters: int,
        step: float,
        lam: float,
        name: str,
        stop_at: float,
        use_orn: bool = False,
        orn_w: float = 0.25,
        stop_ang: float = 0.25,
        z_des_xyz: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        """DLS IK toward a frozen ``goal`` (E, 3). Palm-down if ``use_orn``."""
        from isaacgym.torch_utils import quat_rotate, tensor_clamp

        device = self.device
        palm_off = torch.as_tensor(
            self.palm_offset, dtype=torch.float, device=device).view(1, 3)
        body = int(self.allegro_palm_handle)
        if body >= int(jac.shape[1]):
            body = int(jac.shape[1]) - 1
        lo = self.arm_hand_dof_lower_limits[:7]
        hi = self.arm_hand_dof_upper_limits[:7]
        z_xyz = (0.0, 0.0, -1.0) if z_des_xyz is None else tuple(
            float(v) for v in z_des_xyz)
        z_des = torch.tensor([list(z_xyz)], dtype=torch.float, device=device)
        z_des = z_des / (torch.norm(z_des, dim=-1, keepdim=True) + 1e-8)
        x_raw = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float, device=device)
        x_des = x_raw - (x_raw * z_des).sum(dim=-1, keepdim=True) * z_des
        x_des = x_des / (torch.norm(x_des, dim=-1, keepdim=True) + 1e-8)
        z_axis = torch.zeros((1, 3), device=device, dtype=torch.float)
        z_axis[:, 2] = 1.0
        x_axis = torch.zeros((1, 3), device=device, dtype=torch.float)
        x_axis[:, 0] = 1.0
        last_err = 1e9
        last_ang = 1e9
        best_q = self.arm_hand_dof_pos[:, :7].detach().clone()
        best_score = 1e9
        for i in range(int(n_iters)):
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self._stow_object_for_ik()
            self.compute_observations()
            pos_err = goal - self.palm_center_pos
            last_err = float(torch.norm(pos_err, dim=-1).mean().item())
            z_cur = quat_rotate(
                self._palm_rot, z_axis.expand(self.num_envs, 3))
            x_cur = quat_rotate(
                self._palm_rot, x_axis.expand(self.num_envs, 3))
            cosang = torch.clamp((z_cur * z_des).sum(dim=-1), -1.0, 1.0)
            last_ang = float(torch.acos(cosang).mean().item())
            score = last_err + (0.15 * last_ang if use_orn else 0.0)
            if score < best_score and last_err < 0.15:
                best_score = score
                best_q = self.arm_hand_dof_pos[:, :7].detach().clone()
            done = last_err < stop_at
            if use_orn:
                done = done and last_ang < stop_ang
            if done:
                print(
                    f'[allegro_kuka_throw] table_spawn ik {name} {i:3d} '
                    f'|palm-target|={last_err:.3f}m  '
                    f'palm_down_err={last_ang * 180.0 / math.pi:.1f}deg (stop)',
                    flush=True,
                )
                break
            j_full = jac[:, body, :, :7]
            j_lin = j_full[:, 0:3, :]
            j_ang = j_full[:, 3:6, :]
            off_w = quat_rotate(
                self._palm_rot, palm_off.expand(self.num_envs, 3))
            j_palm = j_lin + torch.bmm(-_skew(off_w), j_ang)
            if not use_orn:
                jjt = torch.bmm(j_palm, j_palm.transpose(1, 2))
                damp = (lam ** 2) * torch.eye(3, device=device, dtype=j_palm.dtype)
                dq = torch.bmm(
                    j_palm.transpose(1, 2),
                    torch.linalg.solve(jjt + damp, pos_err.unsqueeze(-1)),
                ).squeeze(-1)
            else:
                orn_err = torch.cross(z_cur, z_des.expand_as(z_cur), dim=-1)
                orn_err = orn_err + 0.5 * torch.cross(
                    x_cur, x_des.expand_as(x_cur), dim=-1)
                err6 = torch.cat([pos_err, float(orn_w) * orn_err], dim=-1)
                j6 = torch.cat([j_palm, j_ang], dim=1)
                jjt = torch.bmm(j6, j6.transpose(1, 2))
                damp = (lam ** 2) * torch.eye(6, device=device, dtype=j6.dtype)
                dq = torch.bmm(
                    j6.transpose(1, 2),
                    torch.linalg.solve(jjt + damp, err6.unsqueeze(-1)),
                ).squeeze(-1)
            q = tensor_clamp(self.arm_hand_dof_pos[:, :7] + step * dq, lo, hi)
            self._ik_apply_q(q)
            if i % 40 == 0 or i == n_iters - 1:
                extra = ''
                if use_orn:
                    extra = f'  palm_down_err={last_ang * 180.0 / math.pi:.1f}deg'
                print(
                    f'[allegro_kuka_throw] table_spawn ik {name} {i:3d} '
                    f'|palm-target|={last_err:.3f}m{extra}',
                    flush=True,
                )
        self._ik_apply_q(best_q)
        return last_err

    def _ik_palm_onto_cube(self) -> None:
        """Waypoint DLS IK: hover, flip palm-down, then lower onto the cube.

        ``table_spawn_behind`` instead stands the palm on the +y face
        (robot side), facing −y, so a shove toward the table-side goal
        does not start from on-top contact.

        ``table_spawn_toward_bucket`` stands the palm on the anti-bucket
        side of the cube (standoff18), paddle facing the throw bucket.
        """
        from isaacgym import gymtorch
        from isaacgym.torch_utils import quat_rotate

        try:
            raw = self.gym.acquire_jacobian_tensor(self.sim, "allegro")
        except Exception as exc:
            raise RuntimeError(f"table_spawn jacobian acquire failed: {exc}") from exc
        jac = gymtorch.wrap_tensor(raw)
        self._stow_object_for_ik()
        self.compute_observations()
        obj = self.object_init_state[:, 0:3]
        behind = bool(getattr(self, "_table_spawn_behind", False))
        toward_bucket = bool(getattr(self, "_table_spawn_toward_bucket", False))
        if toward_bucket:
            cube_xy = (
                float(self._table_spawn_object_xy[0]),
                float(self._table_spawn_object_xy[1]))
            buck_xy = (
                float(DEFAULT_FIXED_TARGET_XYZ[0]),
                float(DEFAULT_FIXED_TARGET_XYZ[1]))
            dxy = (buck_xy[0] - cube_xy[0], buck_xy[1] - cube_xy[1])
            n = math.sqrt(dxy[0] * dxy[0] + dxy[1] * dxy[1]) + 1e-12
            ux, uy = dxy[0] / n, dxy[1] / n
            dist = float(TABLE_SPAWN_BUCKET_STANDOFF)
            above = float(TABLE_SPAWN_BUCKET_ABOVE)
            # Palm opposite the bucket; paddle faces cube/bucket + slight down.
            stand = obj.clone()
            stand[:, 0] = cube_xy[0] - dist * ux
            stand[:, 1] = cube_xy[1] - dist * uy
            stand[:, 2] = TABLE_OBJECT_Z + above
            hover = stand.clone()
            hover[:, 2] = TABLE_OBJECT_Z + 0.18
            z_raw = (ux, uy, -0.85)
            zn = math.sqrt(z_raw[0] ** 2 + z_raw[1] ** 2 + z_raw[2] ** 2)
            z_des = (z_raw[0] / zn, z_raw[1] / zn, z_raw[2] / zn)
            self._ik_palm_stage(
                jac, hover, n_iters=220, step=0.18, lam=0.12,
                name="hover", stop_at=0.04)
            self._ik_palm_stage(
                jac, hover, n_iters=220, step=0.12, lam=0.12,
                name="face-bucket", stop_at=0.05, use_orn=True, orn_w=0.40,
                stop_ang=0.35, z_des_xyz=z_des)
            last_err = self._ik_palm_stage(
                jac, stand, n_iters=220, step=0.10, lam=0.10,
                name="standoff-bucket", stop_at=0.03, use_orn=True, orn_w=0.30,
                stop_ang=0.40, z_des_xyz=z_des)
            want = 'want paddle toward bucket'
        elif behind:
            stand = obj.clone()
            stand[:, 1] = stand[:, 1] + self._table_spawn_behind_dy
            stand[:, 2] = TABLE_OBJECT_Z + self._table_spawn_behind_above
            hover = stand.clone()
            hover[:, 2] = max(
                TABLE_OBJECT_Z + 0.18,
                TABLE_OBJECT_Z + float(self._table_spawn_behind_above) + 0.06)
            # dy≈0 is a hover-above init: palm-down over the cube, not a
            # +y paddle standoff (those fingers would still spear the cube).
            if abs(float(self._table_spawn_behind_dy)) < 1e-6:
                z_des = (0.0, 0.0, -1.0)
                face_name = "palm-down"
                stand_name = "above"
                want = 'want ~ (0,0,-1) = palm-down above the cube'
            else:
                _tilt = float(getattr(self, "_table_spawn_behind_tilt",
                                      abs(TABLE_SPAWN_BEHIND_Z_DES[2])))
                _zn = math.sqrt(1.0 + _tilt * _tilt)
                z_des = (0.0, -1.0 / _zn, -_tilt / _zn)
                face_name = "face-goal"
                stand_name = "behind"
                want = 'want ~ (0,-1,0) = facing the cube / goal'
            self._ik_palm_stage(
                jac, hover, n_iters=220, step=0.18, lam=0.12,
                name="hover", stop_at=0.04)
            self._ik_palm_stage(
                jac, hover, n_iters=220, step=0.12, lam=0.12,
                name=face_name, stop_at=0.05, use_orn=True, orn_w=0.40,
                stop_ang=0.35, z_des_xyz=z_des)
            last_err = self._ik_palm_stage(
                jac, stand, n_iters=220, step=0.10, lam=0.10,
                name=stand_name, stop_at=0.03, use_orn=True, orn_w=0.30,
                stop_ang=0.40, z_des_xyz=z_des)
        else:
            hover = obj.clone()
            hover[:, 2] = TABLE_OBJECT_Z + 0.18
            touch = obj.clone()
            touch[:, 2] = TABLE_OBJECT_Z + TABLE_SPAWN_PALM_ABOVE
            z_des = (0.0, 0.0, -1.0)
            self._ik_palm_stage(
                jac, hover, n_iters=220, step=0.18, lam=0.12,
                name="hover", stop_at=0.04)
            self._ik_palm_stage(
                jac, hover, n_iters=200, step=0.12, lam=0.12,
                name="palm-down", stop_at=0.05, use_orn=True, orn_w=0.35,
                stop_ang=0.30)
            last_err = self._ik_palm_stage(
                jac, touch, n_iters=220, step=0.10, lam=0.10,
                name="touch", stop_at=0.03, use_orn=True, orn_w=0.25,
                stop_ang=0.35)
            want = 'want ~ (0,0,-1) = facing the cube'
        self._set_object_root(None)
        self.compute_observations()
        self.hand_arm_default_dof_pos[:7] = self.arm_hand_dof_pos[0, :7].detach()
        in_hand = bool(getattr(self, "_table_spawn_in_hand", False))
        if behind or toward_bucket or in_hand:
            fq = torch.tensor(
                TABLE_SPAWN_BEHIND_FINGER_Q, dtype=torch.float,
                device=self.device) * self._table_spawn_finger_curl_scale
            self.hand_arm_default_dof_pos[7:23] = fq
            self.arm_hand_dof_pos[:, 7:23] = fq
            self.cur_targets[:, 7:23] = fq
            self.prev_targets[:, 7:23] = fq
            self._ik_apply_q(self.arm_hand_dof_pos[:, :7])
            if in_hand:
                self.compute_observations()
                self._place_cube_in_hand(write_init=True, use_stored=False)
                self.compute_observations()
                palm0 = self.palm_center_pos[0]
                obj0 = self.object_pos[0]
                print(
                    '[allegro_kuka_throw] in_hand snap '
                    f'offset={self._table_spawn_in_hand_offset:g} '
                    f'obj_noise=±{self._table_spawn_in_hand_obj_noise:g} '
                    f'palm0={tuple(round(float(v), 3) for v in palm0.tolist())} '
                    f'obj0={tuple(round(float(v), 3) for v in obj0.tolist())} '
                    f'|palm-obj|={float(torch.norm(palm0 - obj0).item()):.3f}m',
                    flush=True,
                )
            else:
                self._set_object_root(None)
            self._table_spawn_per_env_dof_pos = (
                self.arm_hand_dof_pos.detach().clone())
        z_axis = torch.zeros((1, 3), device=self.device, dtype=torch.float)
        z_axis[:, 2] = 1.0
        z_w = quat_rotate(self._palm_rot[:1], z_axis)
        print(
            f'[allegro_kuka_throw] table_spawn ik done mean_err={last_err:.3f}m '
            f'palm_z={tuple(round(float(v), 3) for v in z_w[0].tolist())} '
            f'({want}) '
            f'arm_q={tuple(round(float(v), 3) for v in self.hand_arm_default_dof_pos[:7].tolist())}',
            flush=True,
        )

    def reset_target_pose(self, env_ids: Tensor) -> None:
        """Park the bucket/goal on a *full* reset only.

        NVIDIA sets ``reset_goal_buf`` on keypoint success and then calls this
        from ``pre_physics_step``, which used to ``reset_object_pose`` (cube
        snapped back to the palm). Swallow that path. ``reset_idx`` still
        respawns the cube via the ``_in_full_reset`` flag.
        """
        if env_ids is None or len(env_ids) == 0:
            return
        if not getattr(self, "_in_full_reset", False):
            self.reset_goal_buf[env_ids] = 0
            self.near_goal_steps[env_ids] = 0
            self.closest_keypoint_max_dist[env_ids] = -1
            return
        super().reset_target_pose(env_ids)

    def reset_idx(self, env_ids: Tensor) -> None:
        """Apply specialized table-spawn or randomized trim-SA joint poses."""
        self._in_full_reset = True
        try:
            super().reset_idx(env_ids)
        finally:
            self._in_full_reset = False
        per_env_q = getattr(self, "_table_spawn_per_env_dof_pos", None)
        mode = self._control_sanity_mode
        index_thumb_straight = mode == "index_thumb_straight"
        random_trim = getattr(
            self, "_control_sanity_random_trim_init", False)
        random_controlled = (
            random_trim
            or (index_thumb_straight and not getattr(self, "_freeze_init", False)))
        if per_env_q is None and not random_trim and not index_thumb_straight:
            return
        from isaacgym import gymtorch

        if per_env_q is not None:
            q = per_env_q[env_ids].clone()
            finger_noise = float(
                getattr(self, "_table_spawn_finger_noise", 0.0) or 0.0)
            arm_noise = float(
                getattr(self, "_table_spawn_arm_noise", 0.0) or 0.0)
            if finger_noise > 0.0:
                lo = self.arm_hand_dof_lower_limits[7:23]
                hi = self.arm_hand_dof_upper_limits[7:23]
                u = 2.0 * torch.rand(
                    (len(env_ids), 16), device=self.device) - 1.0
                q[:, 7:23] = torch.clamp(
                    q[:, 7:23] + u * (finger_noise * (hi - lo)).unsqueeze(0),
                    lo, hi)
            if arm_noise > 0.0:
                lo = self.arm_hand_dof_lower_limits[:7]
                hi = self.arm_hand_dof_upper_limits[:7]
                u = 2.0 * torch.rand(
                    (len(env_ids), 7), device=self.device) - 1.0
                q[:, :7] = torch.clamp(
                    q[:, :7] + u * (arm_noise * (hi - lo)).unsqueeze(0),
                    lo, hi)
        else:
            # Keep arm and non-goal fingers fixed. Only controlled joints vary,
            # so every randomized coordinate is observed and controllable.
            q = self.hand_arm_default_dof_pos.unsqueeze(0).expand(
                len(env_ids), -1).clone()
            if index_thumb_straight:
                hand_idx = CONTROL_SANITY_INDEX_THUMB_HAND_INDICES
            elif mode == "three2":
                hand_idx = CONTROL_SANITY_THREE2_HAND_INDICES
            elif mode in (
                    "four2", "four2h", "four2m", "four2mh", "four2mm",
                    "four2mmh", "four2mmx", "four2w"):
                hand_idx = CONTROL_SANITY_FOUR2_HAND_INDICES
            elif mode == "index":
                hand_idx = tuple(range(GOAL_DIM_INDEX_CONTROL))
            elif mode == "six":
                hand_idx = tuple(range(GOAL_DIM_SIX_CONTROL))
            elif mode == "two_finger":
                hand_idx = tuple(range(GOAL_DIM_TWO_FINGER_CONTROL))
            elif mode in ("finger", "hand16", "hand16fig", "hand16ok",
                          "hand16peace", "hand16point", "hand16gun"):
                hand_idx = tuple(range(_NUM_HAND_DOFS))
            elif mode in CONTROL_SANITY_ARM23_MODES:
                raise ValueError(
                    f"{mode} does not support randomized trim-SA init")
            else:
                raise ValueError(
                    "randomized trim-SA init requires a finger control mode; "
                    f"got {mode!r}")
            full_idx = torch.tensor(
                [_NUM_ARM_DOFS + int(i) for i in hand_idx],
                dtype=torch.long, device=self.device)
            lo = self.arm_hand_dof_lower_limits[full_idx]
            hi = self.arm_hand_dof_upper_limits[full_idx]
            if index_thumb_straight:
                center = torch.tensor(
                    CONTROL_SANITY_INDEX_THUMB_CURLED_RESET_Q,
                    dtype=torch.float, device=self.device)
            else:
                center = self.hand_arm_default_dof_pos[full_idx]
            center = torch.clamp(center, lo, hi)
            if random_controlled:
                init_mode = getattr(
                    self, "_control_sanity_trim_init_mode", "curled")
                range_frac = self._control_sanity_trim_init_range_frac
                full_range = init_mode == "full_range"
                if full_range:
                    def _sample(n: int) -> Tensor:
                        u = torch.rand(
                            (n, len(hand_idx)), device=self.device)
                        return lo.unsqueeze(0) + u * (hi - lo).unsqueeze(0)
                else:
                    radius = range_frac * (hi - lo)

                    def _sample(n: int) -> Tensor:
                        noise = 2.0 * torch.rand(
                            (n, len(hand_idx)), device=self.device) - 1.0
                        return torch.clamp(
                            center.unsqueeze(0) + noise * radius.unsqueeze(0),
                            lo, hi)

                controlled_q = _sample(len(env_ids))
                # Broad / full-range resets can land in the balanced success
                # set. Reject those for index_thumb_straight only; legacy
                # curled±10% stays byte-for-byte intact.
                reject_success = (
                    index_thumb_straight
                    and (full_range
                         or range_frac > CONTROL_SANITY_TRIM_INIT_RANGE_FRAC))
                if reject_success:
                    goal = torch.tensor(
                        CONTROL_SANITY_INDEX_THUMB_STRAIGHT_GOAL_Q,
                        dtype=torch.float, device=self.device)
                    rejected = 0
                    for _ in range(CONTROL_SANITY_RESET_REJECTION_MAX_TRIES):
                        err = (controlled_q - goal.unsqueeze(0)).abs()
                        successful = (
                            (err.mean(dim=-1)
                             <= CONTROL_SANITY_BALANCED_MEAN_TOL)
                            & (err.max(dim=-1).values
                               <= CONTROL_SANITY_BALANCED_MAX_TOL))
                        n_bad = int(successful.sum().item())
                        if n_bad == 0:
                            break
                        rejected += n_bad
                        controlled_q[successful] = _sample(n_bad)
                    err = (controlled_q - goal.unsqueeze(0)).abs()
                    successful = (
                        (err.mean(dim=-1)
                         <= CONTROL_SANITY_BALANCED_MEAN_TOL)
                        & (err.max(dim=-1).values
                           <= CONTROL_SANITY_BALANCED_MAX_TOL))
                    if bool(successful.any().item()):
                        raise RuntimeError(
                            "bounded index_thumb_straight reset rejection "
                            f"failed after "
                            f"{CONTROL_SANITY_RESET_REJECTION_MAX_TRIES} tries "
                            f"(init_mode={init_mode})")
                    self._trim_init_rejected_total += rejected
                    self._trim_init_sampled_total += len(env_ids) + rejected
            else:
                controlled_q = center.unsqueeze(0)
            q[:, full_idx] = torch.clamp(controlled_q, lo, hi)
        self.arm_hand_dof_pos[env_ids, :] = q
        self.arm_hand_dof_vel[env_ids, :] = 0.0
        self.prev_targets[env_ids, :] = q
        self.cur_targets[env_ids, :] = q
        hand_indices = self.allegro_hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        if (getattr(self, "_table_spawn_in_hand", False)
                and getattr(self, "_in_hand_center", None) is not None):
            # Reuse the pad pose saved at construction. Do not settle
            # here: gym.simulate on the whole world on every partial
            # reset is what made keep-arm ~3× slower than desk spawn.
            self._place_cube_in_hand(
                env_ids, write_init=False, use_stored=True)

    def get_random_quat(self, env_ids: Tensor) -> Tensor:
        """Upstream always samples a uniform quat; freeze uses the spawn pose."""
        # table_spawn's yaw jitter is itself a form of init noise. Honor
        # ``randomize_init=False`` so keep-arm / frozen-spawn stays fixed.
        if (getattr(self, "_table_spawn", False)
                and not getattr(self, "_freeze_init", False)):
            n = len(env_ids)
            half = 0.5 * (
                torch.rand(n, device=self.device) * (2.0 * math.pi) - math.pi)
            q = torch.zeros((n, 4), dtype=torch.float, device=self.device)
            q[:, 2] = torch.sin(half)
            q[:, 3] = torch.cos(half)
            return q
        # The parked object is irrelevant in control-sanity tasks. Even when
        # joint initialization is randomized, its pose remains deterministic.
        if getattr(self, "_control_sanity_mode", ""):
            init = getattr(self, "object_init_state", None)
            if init is not None:
                return init[env_ids, 3:7].clone()
            q = torch.zeros(
                (len(env_ids), 4), dtype=torch.float, device=self.device)
            q[:, 3] = 1.0
            return q
        if not getattr(self, "_freeze_init", False):
            return super().get_random_quat(env_ids)
        init = getattr(self, "object_init_state", None)
        if init is not None:
            return init[env_ids, 3:7].clone()
        q = torch.zeros((len(env_ids), 4), dtype=torch.float, device=self.device)
        q[:, 3] = 1.0
        return q

    def _fixed_target(self) -> Tensor:
        if self._fixed_target_tensor is None:
            self._fixed_target_tensor = torch.tensor(
                self._fixed_target_xyz, dtype=torch.float, device=self.device
            )
        return self._fixed_target_tensor

    def _fixed_goal(self) -> Optional[Tensor]:
        if self._fixed_goal_xyz is None:
            return None
        if self._fixed_goal_tensor is None:
            self._fixed_goal_tensor = torch.tensor(
                self._fixed_goal_xyz, dtype=torch.float, device=self.device
            )
        return self._fixed_goal_tensor

    def _reset_target(self, env_ids: Tensor) -> None:
        """Place the bucket at the parked pose; goal may be a separate table point."""
        tgt = self._fixed_target()
        n = len(env_ids)
        # Bucket actor position (x, y, z).
        self.root_state_tensor[self.bucket_object_indices[env_ids], 0:3] = tgt.expand(n, 3)

        goal = self._fixed_goal()
        if goal is not None:
            # Table-push (or any explicit object target): do not add the
            # throw-task +0.05 z offset.
            self.goal_states[env_ids, 0:3] = goal.expand(n, 3)
        else:
            # Goal position: bucket location, lifted slightly (matches upstream +0.05 z).
            self.goal_states[env_ids, 0:3] = tgt.expand(n, 3)
            self.goal_states[env_ids, 2:3] = tgt[2] + 0.05

        # Reset the object back onto the table and clear the lifting reward.
        self.reset_object_pose(env_ids)
        self.lifted_object[env_ids] = False

        object_indices_to_reset = [
            self.bucket_object_indices[env_ids],
            self.object_indices[env_ids],
        ]
        self.deferred_set_actor_root_state_tensor_indexed(object_indices_to_reset)

    def _extra_reset_rules(self, resets):
        """End the episode if the cube flies above ``resetObjectZAbove``.

        NVIDIA already resets on fall (``object_pos z < 0.1``) in
        ``_compute_resets``.  ``<= 0`` disables this extra ceiling.
        Object z here is still pre-reset, so stash the fall mask for
        the learner's optional fall penalty.
        """
        fall = (self.object_pos[:, 2] < 0.1).to(resets.dtype)
        self._last_fall = fall
        z_max = float(getattr(self, "_reset_z_above", 0.0) or 0.0)
        if z_max <= 0.0:
            return resets
        return torch.where(
            self.object_pos[:, 2] > z_max,
            torch.ones_like(resets),
            resets,
        )


class AllegroKukaThrowVecEnv:
    """Batched (E parallel envs) Isaac Gym adapter for the sgcrl learner.

    Presents packed ``[state | goal]`` torch CUDA tensors of shape
    ``(E, PACKED_OBS_DIM)`` and standard reward / done tensors of shape ``(E,)``.

    Attributes exposed for the learner / ``make_environment`` convention:
      - ``num_envs``          int
      - ``obs_dim``           49 or 52 (state length, what phi encodes)
      - ``goal_dim``          3 or 6  (packed goal length, what psi encodes)
      - ``action_dim``        23
      - ``max_episode_steps`` episode length (from cfg)
    """

    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        sim_device: Optional[str] = None,
        rl_device: Optional[str] = None,
        graphics_device_id: int = 0,
        headless: bool = True,
        episode_length: int = 300,
        fixed_target_xyz: Tuple[float, float, float] = DEFAULT_FIXED_TARGET_XYZ,
        cfg_dir: Optional[str] = None,
        pipeline: str = "gpu",
        enable_cameras: bool = False,
        camera_width: int = 640,
        camera_height: int = 480,
        randomize_init: bool = True,
        randomize_object_xyz: bool = False,
        randomize_object_shape: bool = True,
        palm_goal: bool = False,
        palm_goal_xyz: Optional[Tuple[float, float, float]] = None,
        joint_goal: bool = False,
        control_sanity_mode: str = "",
        control_sanity_palm_xyz: Tuple[float, float, float] = (
            CONTROL_SANITY_PALM_GOAL_XYZ),
        control_sanity_finger_tol: float = 0.15,
        control_sanity_palm_tol: float = 0.05,
        control_sanity_trim_sa: bool = False,
        control_sanity_trim_init_range_frac: float = (
            CONTROL_SANITY_TRIM_INIT_RANGE_FRAC),
        control_sanity_trim_init_mode: str = "curled",
        control_sanity_q_only: bool = False,
        control_sanity_goal_include_qd: bool = False,
        coordinate_mode: str = "mixed",
        table_push: bool = False,
        table_push_xyz: Optional[Tuple[float, float, float]] = None,
        table_spawn: bool = False,
        table_spawn_object_xy: Optional[Tuple[float, float]] = None,
        table_spawn_behind: bool = False,
        table_spawn_behind_dy: float = TABLE_SPAWN_BEHIND_PALM_DY,
        table_spawn_behind_above: float = TABLE_SPAWN_BEHIND_PALM_ABOVE,
        table_spawn_correlated_xy: float = 0.0,
        table_spawn_finger_curl_scale: float = 1.0,
        table_spawn_finger_noise: float = 0.0,
        table_spawn_arm_noise: float = 0.0,
        table_spawn_behind_tilt: Optional[float] = None,
        table_spawn_toward_bucket: bool = False,
        table_spawn_in_hand: bool = False,
        table_spawn_in_hand_offset: float = TABLE_SPAWN_IN_HAND_OFFSET,
        table_spawn_in_hand_obj_noise: float = TABLE_SPAWN_IN_HAND_OBJ_NOISE,
        table_spawn_in_hand_keep_arm: bool = False,
        table_spawn_in_hand_wrist_offset: float = (
            TABLE_SPAWN_IN_HAND_WRIST_OFFSET),
        table_spawn_in_hand_wrist_noise: float = (
            TABLE_SPAWN_IN_HAND_WRIST_NOISE),
        large_table: bool = False,
        hide_table: bool = False,
        palm_and_object_success: bool = False,
        throw_success: str = THROW_SUCCESS_IN_BUCKET,
        goal_z: Optional[float] = None,
        reset_z_above: float = 0.0,
        lock_arm_base: bool = False,
        camera_eye: Optional[Tuple[float, float, float]] = None,
        camera_tgt: Optional[Tuple[float, float, float]] = None,
        camera_hfov: Optional[float] = None,
    ):
        # GPU PhysX by default.  CPU is a fallback for cluster nodes where
        # Preview 4 GPU kernels fail to register (CUDA 13 / driver 590).
        pipeline = str(pipeline).strip().lower()
        if pipeline not in ("cpu", "gpu"):
            raise ValueError(f"pipeline must be 'cpu' or 'gpu', got {pipeline!r}")
        if sim_device is None:
            # AllegroKukaBase requires "device:id" (it splits on ':').
            sim_device = "cuda:0" if pipeline == "gpu" else "cpu:0"
        if rl_device is None:
            rl_device = sim_device
        self.pipeline = pipeline
        self._cam_handle = None
        self._cam_wh = (int(camera_width), int(camera_height))
        self.control_sanity_mode = str(
            control_sanity_mode or "").strip().lower()
        if self.control_sanity_mode == "off":
            self.control_sanity_mode = ""
        if self.control_sanity_mode not in (
                "", "finger", "index", "six", "two_finger",
                "three2", "four2", "four2h", "four2m", "four2mh", "four2mm",
                "four2mmh", "four2mmx", "four2w", "index_thumb_straight",
                "hand16", "hand16fig", "hand16ok", "hand16peace",
                "hand16point", "hand16gun", "arm23wave", "palm"):
            raise ValueError(
                "control_sanity_mode must be '', 'finger', 'index', "
                "'six', 'two_finger', 'three2', 'four2', 'four2h', 'four2m', "
                "'four2mh', 'four2mm', 'four2mmh', 'four2mmx', 'four2w', "
                "'index_thumb_straight', 'hand16', 'hand16fig', "
                "'hand16ok', 'hand16peace', 'hand16point', 'hand16gun', "
                "'arm23wave', or 'palm'")
        self.control_sanity_hand_indices: Optional[Tuple[int, ...]] = None
        self.control_sanity_full_q = (
            self.control_sanity_mode in CONTROL_SANITY_ARM23_MODES)
        self.control_sanity_palm_xyz = tuple(
            float(v) for v in control_sanity_palm_xyz)
        if len(self.control_sanity_palm_xyz) != 3:
            raise ValueError("control_sanity_palm_xyz must contain 3 values")
        self.control_sanity_finger_tol = float(control_sanity_finger_tol)
        self.control_sanity_palm_tol = float(control_sanity_palm_tol)
        # This task's semantics are intrinsically index+thumb-only: expose
        # only those eight state/action coordinates and hold middle/ring fixed.
        self.control_sanity_trim_sa = (
            bool(control_sanity_trim_sa)
            or self.control_sanity_mode == "index_thumb_straight")
        self.control_sanity_trim_init_range_frac = float(
            control_sanity_trim_init_range_frac)
        if not 0.0 <= self.control_sanity_trim_init_range_frac <= 1.0:
            raise ValueError(
                "control_sanity_trim_init_range_frac must be in [0, 1], got "
                f"{self.control_sanity_trim_init_range_frac}")
        self.control_sanity_trim_init_mode = str(
            control_sanity_trim_init_mode or "curled").strip().lower()
        if self.control_sanity_trim_init_mode not in (
                CONTROL_SANITY_TRIM_INIT_MODES):
            raise ValueError(
                "control_sanity_trim_init_mode must be one of "
                f"{CONTROL_SANITY_TRIM_INIT_MODES}, got "
                f"{self.control_sanity_trim_init_mode!r}")
        self.control_sanity_q_only = bool(control_sanity_q_only)
        if self.control_sanity_q_only and not self.control_sanity_trim_sa:
            raise ValueError(
                "control_sanity_q_only requires control_sanity_trim_sa")
        self.control_sanity_goal_include_qd = bool(
            control_sanity_goal_include_qd)
        if (self.control_sanity_goal_include_qd
                and (not self.control_sanity_trim_sa
                     or self.control_sanity_q_only
                     or not self.control_sanity_mode
                     or self.control_sanity_mode == "palm")):
            raise ValueError(
                "control_sanity_goal_include_qd requires a joint-control "
                "trim-SA mode with q,qd state")
        self.coordinate_mode = str(coordinate_mode or "mixed").strip().lower()
        if self.coordinate_mode not in ("mixed", "physical", "fully_scaled"):
            raise ValueError(
                "coordinate_mode must be mixed, physical, or fully_scaled, "
                f"got {self.coordinate_mode!r}")
        self.randomize_init = bool(randomize_init)
        self._logged_random_trim_init = False
        self._logged_coordinate_ranges = False
        self._logged_success_coordinates = False
        self._trim_hand_idx: Optional[list] = None
        self._trim_full_action_idx: Optional[list] = None
        self.joint_goal = bool(joint_goal) and not self.control_sanity_mode
        self.palm_goal = (
            bool(palm_goal) and not self.joint_goal
            and not self.control_sanity_mode)
        self.table_push = bool(table_push)
        self.palm_and_object_success = bool(palm_and_object_success)
        self.throw_success = str(throw_success or THROW_SUCCESS_IN_BUCKET).strip().lower()
        if self.throw_success not in THROW_SUCCESS_MODES:
            raise ValueError(
                "throw_success must be 'in_bucket', 'nvidia_goal', or "
                f"'goal_ball', got {self.throw_success!r}")
        self.table_spawn = bool(table_spawn)
        if table_spawn_in_hand_keep_arm:
            default_spawn_xy = (0.0, 0.0)
        elif table_spawn_behind:
            default_spawn_xy = TABLE_SPAWN_BEHIND_OBJECT_XY
        else:
            default_spawn_xy = TABLE_SPAWN_OBJECT_XY
        self.table_spawn_object_xy = (
            default_spawn_xy if table_spawn_object_xy is None
            else tuple(float(v) for v in table_spawn_object_xy))
        if len(self.table_spawn_object_xy) != 2:
            raise ValueError(
                "table_spawn_object_xy must contain exactly two values")
        self.table_spawn_behind = bool(table_spawn_behind)
        self.table_spawn_behind_dy = float(table_spawn_behind_dy)
        self.table_spawn_behind_above = float(table_spawn_behind_above)
        self.table_spawn_correlated_xy = float(table_spawn_correlated_xy)
        self.table_spawn_finger_curl_scale = float(
            table_spawn_finger_curl_scale)
        self.table_spawn_finger_noise = float(table_spawn_finger_noise)
        if not 0.0 <= self.table_spawn_finger_noise <= 1.0:
            raise ValueError(
                "table_spawn_finger_noise must be in [0, 1], got "
                f"{self.table_spawn_finger_noise}")
        self.table_spawn_arm_noise = float(table_spawn_arm_noise)
        if not 0.0 <= self.table_spawn_arm_noise <= 1.0:
            raise ValueError(
                "table_spawn_arm_noise must be in [0, 1], got "
                f"{self.table_spawn_arm_noise}")
        self.table_spawn_behind_tilt = (
            None if table_spawn_behind_tilt is None
            else float(table_spawn_behind_tilt))
        self.table_spawn_toward_bucket = bool(table_spawn_toward_bucket)
        self.table_spawn_in_hand = bool(table_spawn_in_hand)
        self.table_spawn_in_hand_offset = float(table_spawn_in_hand_offset)
        if self.table_spawn_in_hand_offset <= 0.0:
            raise ValueError(
                "table_spawn_in_hand_offset must be positive, got "
                f"{self.table_spawn_in_hand_offset}")
        self.table_spawn_in_hand_obj_noise = float(
            table_spawn_in_hand_obj_noise)
        if self.table_spawn_in_hand_obj_noise < 0.0:
            raise ValueError(
                "table_spawn_in_hand_obj_noise must be non-negative, got "
                f"{self.table_spawn_in_hand_obj_noise}")
        if self.table_spawn_in_hand and not table_spawn:
            raise ValueError("table_spawn_in_hand requires table_spawn")
        self.table_spawn_in_hand_keep_arm = bool(table_spawn_in_hand_keep_arm)
        self.table_spawn_in_hand_wrist_offset = float(
            table_spawn_in_hand_wrist_offset)
        self.table_spawn_in_hand_wrist_noise = float(
            table_spawn_in_hand_wrist_noise)
        if self.table_spawn_in_hand_wrist_noise < 0.0:
            raise ValueError(
                "table_spawn_in_hand_wrist_noise must be non-negative, got "
                f"{self.table_spawn_in_hand_wrist_noise}")
        if self.table_spawn_in_hand_keep_arm and not self.table_spawn_in_hand:
            raise ValueError(
                "table_spawn_in_hand_keep_arm requires table_spawn_in_hand")
        self.lock_arm_base = bool(lock_arm_base)
        if self.lock_arm_base and self.control_sanity_trim_sa:
            raise ValueError(
                "lock_arm_base is incompatible with control_sanity_trim_sa")
        self.large_table = bool(large_table)
        self.hide_table = bool(hide_table)
        if self.hide_table and self.large_table:
            raise ValueError("hide_table and large_table are mutually exclusive")
        if self.table_spawn_toward_bucket:
            self.table_spawn_behind = False
        self.reset_z_above = float(reset_z_above or 0.0)
        self._cam_eye_override = camera_eye
        self._cam_tgt_override = camera_tgt
        self._cam_hfov_override = camera_hfov
        self.randomize_object_xyz = bool(randomize_object_xyz)
        self.randomize_object_shape = bool(randomize_object_shape)
        bucket_xyz = resolve_bucket_xyz(
            self.palm_goal and not self.table_push, fixed_target_xyz)
        object_goal_xyz: Optional[Tuple[float, float, float]] = None
        self._object_goal_xyz: Optional[Tuple[float, float, float]] = None
        self._palm_goal_xyz_cmd: Optional[Tuple[float, float, float]] = None
        if self.table_push:
            # Park the bucket off the workspace. hide_table also hides the
            # bucket actor (same far pose as the 1 cm dummy table).
            if self.hide_table:
                bucket_xyz = HIDDEN_BUCKET_XYZ
            else:
                bucket_xyz = (
                    LARGE_TABLE_PARKED_BUCKET_XYZ
                    if self.large_table else DEFAULT_FIXED_TARGET_XYZ)
            if table_push_xyz is None:
                object_goal_xyz = (
                    IN_HAND_KEEP_ARM_GOAL_XYZ
                    if self.table_spawn_in_hand_keep_arm
                    else TABLE_PUSH_GOAL_XYZ)
            else:
                object_goal_xyz = tuple(float(v) for v in table_push_xyz)
                if len(object_goal_xyz) != 3:
                    raise ValueError(
                        f'table_push_xyz must be 3 floats, got {table_push_xyz!r}')
        self._object_goal_xyz = object_goal_xyz
        self._bucket_xyz = tuple(float(v) for v in bucket_xyz)
        self.goal_z = None if goal_z is None else float(goal_z)
        if self.goal_z is not None and object_goal_xyz is None:
            cfg_fixed_goal = (
                self._bucket_xyz[0], self._bucket_xyz[1], self.goal_z)
        else:
            cfg_fixed_goal = object_goal_xyz
        cfg_dict = self._build_cfg(
            num_envs=num_envs,
            episode_length=episode_length,
            fixed_target_xyz=bucket_xyz,
            cfg_dir=cfg_dir,
            pipeline=pipeline,
            enable_cameras=bool(enable_cameras),
            randomize_init=bool(randomize_init),
            randomize_object_xyz=bool(randomize_object_xyz),
            randomize_object_shape=bool(randomize_object_shape),
            fixed_goal_xyz=cfg_fixed_goal,
            table_spawn=bool(table_spawn),
            table_spawn_object_xy=self.table_spawn_object_xy,
            table_spawn_behind=bool(self.table_spawn_behind),
            table_spawn_behind_dy=self.table_spawn_behind_dy,
            table_spawn_behind_above=self.table_spawn_behind_above,
            table_spawn_correlated_xy=self.table_spawn_correlated_xy,
            table_spawn_finger_curl_scale=self.table_spawn_finger_curl_scale,
            table_spawn_finger_noise=self.table_spawn_finger_noise,
            table_spawn_arm_noise=self.table_spawn_arm_noise,
            table_spawn_behind_tilt=self.table_spawn_behind_tilt,
            table_spawn_toward_bucket=bool(self.table_spawn_toward_bucket),
            table_spawn_in_hand=bool(self.table_spawn_in_hand),
            table_spawn_in_hand_offset=self.table_spawn_in_hand_offset,
            table_spawn_in_hand_obj_noise=self.table_spawn_in_hand_obj_noise,
            table_spawn_in_hand_keep_arm=bool(
                self.table_spawn_in_hand_keep_arm),
            table_spawn_in_hand_wrist_offset=(
                self.table_spawn_in_hand_wrist_offset),
            table_spawn_in_hand_wrist_noise=(
                self.table_spawn_in_hand_wrist_noise),
            large_table=bool(self.large_table),
            hide_table=bool(self.hide_table),
            reset_z_above=float(reset_z_above or 0.0),
            control_sanity_mode=self.control_sanity_mode,
            control_sanity_trim_sa=self.control_sanity_trim_sa,
            control_sanity_trim_init_range_frac=(
                self.control_sanity_trim_init_range_frac),
            control_sanity_trim_init_mode=self.control_sanity_trim_init_mode,
        )
        _pos_noise = (
            cfg_dict.get('env', {}).get('resetPositionNoiseX'),
            cfg_dict.get('env', {}).get('resetPositionNoiseY'),
            cfg_dict.get('env', {}).get('resetPositionNoiseZ'),
        )
        _fin_noise = cfg_dict.get('env', {}).get(
            'resetDofPosRandomIntervalFingers')
        print(f'[allegro_kuka_throw] pipeline={pipeline} '
              f'sim_device={sim_device} rl_device={rl_device} E={int(num_envs)} '
              f'cameras={bool(enable_cameras)} '
              f'randomize_init={bool(randomize_init)} '
              f'finger_reset_noise={_fin_noise} '
              f'randomize_object_xyz={bool(randomize_object_xyz)} '
              f'randomize_object_shape={bool(randomize_object_shape)} '
              f'pos_noise_xyz={_pos_noise} '
              f'palm_goal={self.palm_goal} joint_goal={self.joint_goal} '
              f'control_sanity_mode={self.control_sanity_mode or "off"} '
              f'goal_include_qd={self.control_sanity_goal_include_qd} '
              f'coordinate_mode={self.coordinate_mode} '
              f'table_push={self.table_push} '
              f'table_spawn={self.table_spawn} '
              f'goal_z={self.goal_z} '
              f'table_spawn_object_xy={self.table_spawn_object_xy} '
              f'table_spawn_behind={self.table_spawn_behind} '
              f'table_spawn_behind_dy={self.table_spawn_behind_dy:g} '
              f'table_spawn_behind_above={self.table_spawn_behind_above:g} '
              f'table_spawn_correlated_xy={self.table_spawn_correlated_xy:g} '
              f'table_spawn_finger_curl_scale='
              f'{self.table_spawn_finger_curl_scale:g} '
              f'table_spawn_finger_noise='
              f'{self.table_spawn_finger_noise:g} '
              f'table_spawn_arm_noise='
              f'{self.table_spawn_arm_noise:g} '
              f'table_spawn_behind_tilt={self.table_spawn_behind_tilt} '
              f'table_spawn_toward_bucket={self.table_spawn_toward_bucket} '
              f'table_spawn_in_hand={self.table_spawn_in_hand} '
              f'table_spawn_in_hand_offset='
              f'{self.table_spawn_in_hand_offset:g} '
              f'table_spawn_in_hand_obj_noise='
              f'{self.table_spawn_in_hand_obj_noise:g} '
              f'table_spawn_in_hand_keep_arm='
              f'{self.table_spawn_in_hand_keep_arm} '
              f'table_spawn_in_hand_wrist_offset='
              f'{self.table_spawn_in_hand_wrist_offset:g} '
              f'table_spawn_in_hand_wrist_noise='
              f'{self.table_spawn_in_hand_wrist_noise:g} '
              f'large_table={self.large_table} '
              f'hide_table={self.hide_table} '
              f'palm_and_object_success={self.palm_and_object_success} '
              f'throw_success={self.throw_success} '
              f'reset_z_above={self.reset_z_above:g} '
              f'lock_arm_base={self.lock_arm_base} '
              f'bucket={bucket_xyz}'
              f'{f" object_goal={object_goal_xyz}" if object_goal_xyz else ""}')
        self._env = FixedGoalAllegroKukaThrow(
            cfg=cfg_dict,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=False,
            force_render=False,
        )
        self.num_envs = int(self._env.num_envs)
        self.last_fall = torch.zeros(
            (self.num_envs,), dtype=torch.float, device=self._env.device)
        self.action_dim = ACTION_DIM
        self.max_episode_steps = int(episode_length)
        self.device = self._env.device
        actor = self._env.gym.find_actor_handle(
            self._env.envs[0], "allegro")
        dof_props = self._env.gym.get_actor_dof_properties(
            self._env.envs[0], actor)
        self._joint_velocity_limits = torch.as_tensor(
            dof_props["velocity"], dtype=torch.float, device=self.device)
        if (self._joint_velocity_limits.numel() != _NUM_ARM_HAND_DOFS
                or not torch.all(torch.isfinite(self._joint_velocity_limits))
                or not torch.all(self._joint_velocity_limits > 0)):
            raise RuntimeError(
                "invalid Allegro per-joint physical velocity limits: "
                f"{self._joint_velocity_limits.tolist()}")
        if self.control_sanity_mode == "finger":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_FINGER_GOAL_Q
            print(
                '[allegro_kuka_throw] finger-control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                '(finger q* only; dummy object parked)')
        elif self.control_sanity_mode == "hand16":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_HAND16_GOAL_Q
            print(
                '[allegro_kuka_throw] hand16 control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(all 16 hand q* mild={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "hand16fig":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_HAND16_FIGURE_GOAL_Q
            print(
                '[allegro_kuka_throw] hand16fig control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(all 16 hand q* hard-zigzag={goal_vec}; '
                f'init_mode={self.control_sanity_trim_init_mode}; '
                'dummy object parked)')
        elif self.control_sanity_mode == "hand16ok":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_HAND16_OK_GOAL_Q
            print(
                '[allegro_kuka_throw] hand16ok control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(all 16 hand q* OK-circle={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "hand16peace":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_HAND16_PEACE_GOAL_Q
            print(
                '[allegro_kuka_throw] hand16peace control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(all 16 hand q* V-sign={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "hand16point":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_HAND16_POINT_GOAL_Q
            print(
                '[allegro_kuka_throw] hand16point control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(all 16 hand q* point={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "hand16gun":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_HAND16_GUN_GOAL_Q
            print(
                '[allegro_kuka_throw] hand16gun control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(all 16 hand q* gun={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "arm23wave":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_ARM23_CONTROL
            goal_vec = CONTROL_SANITY_ARM23_WAVE_GOAL_Q
            print(
                '[allegro_kuka_throw] arm23wave control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(all 23 arm+hand q* wave arm={goal_vec[:7]} '
                f'hand={goal_vec[7:]}; dummy object parked)')
        elif self.control_sanity_mode == "index":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_INDEX_CONTROL
            goal_vec = CONTROL_SANITY_INDEX_GOAL_Q
            print(
                '[allegro_kuka_throw] index-control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(index q*={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "six":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_SIX_CONTROL
            goal_vec = CONTROL_SANITY_SIX_GOAL_Q
            print(
                '[allegro_kuka_throw] six-dof control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(index4+middle2 q*={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "two_finger":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_TWO_FINGER_CONTROL
            goal_vec = CONTROL_SANITY_TWO_FINGER_GOAL_Q
            print(
                '[allegro_kuka_throw] two-finger control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(index+middle q*={goal_vec}; dummy object parked)')
        elif self.control_sanity_mode == "three2":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_THREE2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_THREE2_HAND_INDICES
            goal_vec = CONTROL_SANITY_THREE2_GOAL_Q
            print(
                '[allegro_kuka_throw] three2 control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(idx/mid/ring base+prox q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "four2":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2_GOAL_Q
            print(
                '[allegro_kuka_throw] four2 control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers base+prox q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "four2h":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2H_GOAL_Q
            print(
                '[allegro_kuka_throw] four2h control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers base+prox HARD q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "four2m":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2M_GOAL_Q
            print(
                '[allegro_kuka_throw] four2m control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers base+prox MED q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "four2mh":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2MH_GOAL_Q
            print(
                '[allegro_kuka_throw] four2mh control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers base+prox MED-HARD q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "four2mm":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2MM_GOAL_Q
            print(
                '[allegro_kuka_throw] four2mm control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers base+prox m–mh mid q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "four2mmh":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2MMH_GOAL_Q
            print(
                '[allegro_kuka_throw] four2mmh control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers base+prox mm–mh mid q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "four2mmx":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2MMX_GOAL_Q
            print(
                '[allegro_kuka_throw] four2mmx control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers base+prox mm↔mmh mid q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "index_thumb_straight":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_INDEX_THUMB_CONTROL
            self.control_sanity_hand_indices = (
                CONTROL_SANITY_INDEX_THUMB_HAND_INDICES)
            goal_vec = CONTROL_SANITY_INDEX_THUMB_STRAIGHT_GOAL_Q
            print(
                '[allegro_kuka_throw] index_thumb_straight control packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(index+thumb straight q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; '
                f'init_mode={self.control_sanity_trim_init_mode}; '
                'dummy object parked)')
        elif self.control_sanity_mode == "four2w":
            self.obs_dim = STATE_DIM_CONTROL
            self.goal_dim = GOAL_DIM_FOUR2_CONTROL
            self.control_sanity_hand_indices = CONTROL_SANITY_FOUR2_HAND_INDICES
            goal_vec = CONTROL_SANITY_FOUR2W_GOAL_Q
            print(
                '[allegro_kuka_throw] four2w control sanity packing: '
                f'state={self.obs_dim} (q,qd) goal={self.goal_dim} '
                f'(4fingers WEIRD point q*={goal_vec}; hand_idx='
                f'{self.control_sanity_hand_indices}; dummy object parked)')
        elif self.control_sanity_mode == "palm":
            self.obs_dim = STATE_DIM_PALM_CONTROL
            self.goal_dim = GOAL_DIM_PALM_CONTROL
            goal_vec = self.control_sanity_palm_xyz
            print(
                '[allegro_kuka_throw] palm-control sanity packing: '
                f'state={self.obs_dim} (q,qd,palm) goal={self.goal_dim} '
                f'(palm xyz={goal_vec}; dummy object parked)')
        elif self.joint_goal:
            if len(TABLE_SIDE_JOINT_GOAL_Q) != _NUM_ARM_HAND_DOFS:
                raise ValueError(
                    'TABLE_SIDE_JOINT_GOAL_Q must have '
                    f'{_NUM_ARM_HAND_DOFS} values, got '
                    f'{len(TABLE_SIDE_JOINT_GOAL_Q)}')
            self.obs_dim = STATE_DIM
            self.goal_dim = GOAL_DIM_JOINT
            hand_star = tuple(
                float(v) for v in TABLE_SIDE_JOINT_GOAL_Q[_NUM_ARM_DOFS:])
            if object_goal_xyz is not None:
                obj_g = tuple(float(v) for v in object_goal_xyz)
                _obj_src = 'object_goal (table_push)'
            else:
                obj_g = object_in_bucket_xyz(bucket_xyz)
                _obj_src = 'object_in_bucket'
            goal_vec = hand_star + obj_g
            print(f'[allegro_kuka_throw] joint-goal packing: state={self.obs_dim} '
                  f'(q,qd,obj; no palm) goal={self.goal_dim} '
                  f'(hand q* 16-D + {_obj_src}={obj_g}) '
                  f'HER=state[7:23]+state[46:49]')
        elif self.palm_goal:
            self.obs_dim = STATE_DIM_PALM
            self.goal_dim = GOAL_DIM_PALM
            palm = SLIDE_PALM_XYZ if palm_goal_xyz is None else tuple(
                float(v) for v in palm_goal_xyz)
            if len(palm) != 3:
                raise ValueError(f'palm_goal_xyz must be 3 floats, got {palm!r}')
            self._palm_goal_xyz_cmd = palm
            if object_goal_xyz is not None:
                obj_g = tuple(float(v) for v in object_goal_xyz)
                _obj_src = 'object_goal (table_push)'
            else:
                obj_g = object_in_bucket_xyz(bucket_xyz)
                _obj_src = 'object_in_bucket'
            goal_vec = palm + obj_g
            print(f'[allegro_kuka_throw] palm-goal packing: state={self.obs_dim} '
                  f'(q,qd,palm,obj) goal={self.goal_dim} '
                  f'palm={palm} {_obj_src}={obj_g}')
        elif object_goal_xyz is not None:
            self.obs_dim = STATE_DIM
            self.goal_dim = GOAL_DIM
            goal_vec = tuple(float(v) for v in object_goal_xyz)
            print(f'[allegro_kuka_throw] table-push packing: state={self.obs_dim} '
                  f'goal={goal_vec} (object xyz on desk; bucket parked)')
        else:
            self.obs_dim = STATE_DIM
            self.goal_dim = GOAL_DIM
            if self.goal_z is not None:
                goal_vec = (
                    float(bucket_xyz[0]), float(bucket_xyz[1]),
                    float(self.goal_z))
            else:
                goal_vec = tuple(float(v) for v in bucket_xyz)
            if self.throw_success == THROW_SUCCESS_NVIDIA_GOAL:
                _succ = (
                    f'success=nvidia_goal ball around z+{OBJECT_ABOVE_BUCKET:g} '
                    f'(r=success_tolerance*keypoint_scale, one frame; no hold)')
            elif self.throw_success == THROW_SUCCESS_GOAL_BALL:
                _succ = (
                    f'success=goal_ball around z+{OBJECT_ABOVE_BUCKET:g} '
                    f'(r=success_tolerance 7.5cm, one frame; no hold)')
            else:
                _succ = (
                    f'success=in-bucket cylinder '
                    f'r={BUCKET_INNER_RADIUS:g} h={BUCKET_HEIGHT:g}')
            print(
                f'[allegro_kuka_throw] throw packing: state={self.obs_dim} '
                f'goal={goal_vec} '
                f'({"near-rim goal z" if self.goal_z is not None else "bucket xyz"}; '
                f'{_succ}; '
                f'NVIDIA success-respawn off'
                f'{"; table hidden" if self.hide_table else ""})')
        self._control_sanity_goal_q_batch: Optional[Tensor] = None
        self._control_sanity_goal_action_batch: Optional[Tensor] = None
        self._normalized_goal_action_batch: Optional[Tensor] = None
        joint_control_goal = bool(
            self.control_sanity_mode
            and self.control_sanity_mode != "palm")
        if joint_control_goal:
            if self.control_sanity_full_q:
                # Goal is the full 23-D arm+hand q. Do not offset by 7.
                full_goal_idx = list(range(_NUM_ARM_HAND_DOFS))
            else:
                if not self.control_sanity_hand_indices:
                    # Consecutive first ``goal_dim`` hand joints for finger,
                    # hand16, index, six, and two_finger.
                    self.control_sanity_hand_indices = tuple(
                        range(int(self.goal_dim)))
                full_goal_idx = [
                    _NUM_ARM_DOFS + int(i)
                    for i in self.control_sanity_hand_indices]
            raw_goal = torch.tensor(
                list(goal_vec), dtype=torch.float, device=self.device)
            position_goal_dim = int(raw_goal.numel())
            expected_n = len(full_goal_idx)
            if position_goal_dim != expected_n:
                raise RuntimeError(
                    "joint-control position goal width does not match "
                    f"controlled joints ({position_goal_dim} != {expected_n})")
            lo = self._env.arm_hand_dof_lower_limits[full_goal_idx]
            hi = self._env.arm_hand_dof_upper_limits[full_goal_idx]
            normalized_goal_action = (
                2.0 * (raw_goal - lo) / (hi - lo + 1e-8) - 1.0)
            roundtrip = lo + 0.5 * (
                normalized_goal_action + 1.0) * (hi - lo)
            if not torch.allclose(
                    roundtrip, raw_goal, atol=1e-5, rtol=1e-5):
                raise RuntimeError(
                    "control-sanity goal normalization round-trip failed: "
                    f"raw={raw_goal.tolist()} "
                    f"action={normalized_goal_action.tolist()} "
                    f"roundtrip={roundtrip.tolist()}")
            self._control_sanity_goal_q_batch = raw_goal.unsqueeze(0).expand(
                self.num_envs, position_goal_dim).contiguous()
            self._control_sanity_goal_action_batch = (
                normalized_goal_action.unsqueeze(0).expand(
                    self.num_envs, position_goal_dim).contiguous())
            # Backward-compatible alias used by existing diagnostics.
            self._normalized_goal_action_batch = (
                self._control_sanity_goal_action_batch)
            packed_q_goal = (
                raw_goal
                if self.coordinate_mode == "physical"
                else normalized_goal_action)
            if self.control_sanity_goal_include_qd:
                packed_goal = torch.cat(
                    [packed_q_goal, torch.zeros_like(packed_q_goal)], dim=0)
                self.goal_dim = 2 * position_goal_dim
            else:
                packed_goal = packed_q_goal
                self.goal_dim = position_goal_dim
            self._goal_batch = packed_goal.unsqueeze(0).expand(
                self.num_envs, self.goal_dim).contiguous()
            print(
                "[allegro_kuka_throw] control goal coordinates: "
                f"raw_rad={tuple(round(float(v), 6) for v in raw_goal.tolist())} "
                f"packed_{self.coordinate_mode}="
                f"{tuple(round(float(v), 6) for v in packed_goal.tolist())} "
                f"normalized_action={tuple(round(float(v), 6) for v in normalized_goal_action.tolist())} "
                "physical→normalized-action roundtrip=ok",
                flush=True)
        else:
            self._goal_batch = torch.tensor(
                list(goal_vec), dtype=torch.float, device=self.device
            ).unsqueeze(0).expand(
                self.num_envs, self.goal_dim).contiguous()
        if self.control_sanity_trim_sa:
            if not self.control_sanity_hand_indices:
                raise ValueError(
                    "control_sanity_trim_sa requires a joint control goal")
            n = len(self.control_sanity_hand_indices)
            self._trim_hand_idx = list(self.control_sanity_hand_indices)
            self._trim_full_action_idx = [
                _NUM_ARM_DOFS + int(i) for i in self._trim_hand_idx]
            self.obs_dim = n if self.control_sanity_q_only else 2 * n
            self.action_dim = n
            state_desc = "q only" if self.control_sanity_q_only else "q,qd"
            print(
                '[allegro_kuka_throw] control_sanity_trim_sa: '
                f'state={self.obs_dim} ({state_desc} of hand_idx='
                f'{self._trim_hand_idx}) action={self.action_dim}; '
                f'goal={self.goal_dim} '
                f'(include_qd={self.control_sanity_goal_include_qd}); '
                f'coordinate_mode={self.coordinate_mode}; '
                'arm + other hand joints frozen at default targets',
                flush=True)
        if self.lock_arm_base:
            if self._trim_hand_idx is not None:
                raise ValueError(
                    "lock_arm_base is incompatible with control_sanity_trim_sa")
            self.action_dim = int(LOCK_ARM_BASE_ACTION_DIM)
            print(
                '[allegro_kuka_throw] lock_arm_base: A1–A4 held '
                '(relative 0); policy action='
                f'{self.action_dim} (A5–A7 + 16 fingers)',
                flush=True)
        if bool(enable_cameras):
            self._setup_camera()
        if self.table_spawn:
            palm0 = self._palm_xyz()[0]
            obj0 = self._object_xyz()[0]
            dist = float(torch.norm(palm0 - obj0).item())
            print(
                f'[allegro_kuka_throw] table_spawn reset '
                f'palm={tuple(round(float(v), 3) for v in palm0.tolist())} '
                f'obj={tuple(round(float(v), 3) for v in obj0.tolist())} '
                f'|palm-obj|={dist:.3f}m',
                flush=True,
            )

    @staticmethod
    def _build_cfg(num_envs, episode_length, fixed_target_xyz, cfg_dir,
                   pipeline="gpu", enable_cameras=False, randomize_init=True,
                   randomize_object_xyz=False, randomize_object_shape=True,
                   fixed_goal_xyz=None, table_spawn=False,
                   table_spawn_object_xy=TABLE_SPAWN_OBJECT_XY,
                   table_spawn_behind=False,
                   table_spawn_behind_dy=TABLE_SPAWN_BEHIND_PALM_DY,
                   table_spawn_behind_above=TABLE_SPAWN_BEHIND_PALM_ABOVE,
                   table_spawn_correlated_xy=0.0,
                   table_spawn_finger_curl_scale=1.0,
                   table_spawn_finger_noise=0.0,
                   table_spawn_arm_noise=0.0,
                   table_spawn_behind_tilt=None,
                   table_spawn_toward_bucket=False,
                   table_spawn_in_hand=False,
                   table_spawn_in_hand_offset=TABLE_SPAWN_IN_HAND_OFFSET,
                   table_spawn_in_hand_obj_noise=TABLE_SPAWN_IN_HAND_OBJ_NOISE,
                   table_spawn_in_hand_keep_arm=False,
                   table_spawn_in_hand_wrist_offset=(
                       TABLE_SPAWN_IN_HAND_WRIST_OFFSET),
                   table_spawn_in_hand_wrist_noise=(
                       TABLE_SPAWN_IN_HAND_WRIST_NOISE),
                   large_table=False, hide_table=False,
                   reset_z_above=0.0,
                   control_sanity_mode="", control_sanity_trim_sa=False,
                   control_sanity_trim_init_range_frac=(
                       CONTROL_SANITY_TRIM_INIT_RANGE_FRAC),
                   control_sanity_trim_init_mode="curled"):
        """Compose the IsaacGymEnvs Hydra config for the throw subtask."""
        from hydra import compose, initialize_config_dir
        from isaacgymenvs.utils.reformat import omegaconf_to_dict

        if cfg_dir is None:
            import isaacgymenvs
            cfg_dir = os.path.join(os.path.dirname(isaacgymenvs.__file__), "cfg")

        with initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "task=AllegroKuka",
                    "task.env.subtask=throw",
                    f"num_envs={int(num_envs)}",
                    "headless=True",
                    f"pipeline={pipeline}",
                ],
            )
            cfg_dict = omegaconf_to_dict(cfg.task)

        cfg_dict["env"]["numEnvs"] = int(num_envs)
        cfg_dict["env"]["episodeLength"] = int(episode_length)
        cfg_dict["env"]["fixedTargetXYZ"] = [float(v) for v in fixed_target_xyz]
        if fixed_goal_xyz is not None:
            cfg_dict["env"]["fixedGoalXYZ"] = [float(v) for v in fixed_goal_xyz]
        cfg_dict["env"]["randomizeInit"] = bool(randomize_init)
        if not randomize_init:
            # Freeze joints / quat / forces. Quat freeze is in
            # get_random_quat. Object xyz stays at NVIDIA defaults when
            # randomize_object_xyz is on (0.1, 0.1, 0.02).
            if not randomize_object_xyz:
                cfg_dict["env"]["resetPositionNoiseX"] = 0.0
                cfg_dict["env"]["resetPositionNoiseY"] = 0.0
                cfg_dict["env"]["resetPositionNoiseZ"] = 0.0
            cfg_dict["env"]["resetRotationNoise"] = 0.0
            cfg_dict["env"]["resetDofPosRandomIntervalFingers"] = 0.0
            cfg_dict["env"]["resetDofPosRandomIntervalArm"] = 0.0
            cfg_dict["env"]["resetDofVelRandomInterval"] = 0.0
            cfg_dict["env"]["forceScale"] = 0.0
            cfg_dict["task"]["randomize"] = False
        # Shape mix is on only for full NVIDIA init. Frozen init, or an
        # explicit --noisaacgym_randomize_object_shape, keeps one cube.
        if (not randomize_init) or (not randomize_object_shape):
            cfg_dict["env"]["randomizeObjectDimensions"] = False
            cfg_dict["env"]["withSmallCuboids"] = False
            cfg_dict["env"]["withBigCuboids"] = False
            cfg_dict["env"]["withSticks"] = False
        if enable_cameras:
            # Keep graphics_device_id >= 0 in VecTask even when headless.
            cfg_dict["env"]["enableCameraSensors"] = True
        toward_bucket = bool(table_spawn_toward_bucket)
        behind = bool(table_spawn_behind) and not toward_bucket
        cfg_dict["env"]["tableSpawn"] = bool(table_spawn)
        cfg_dict["env"]["tableSpawnObjectXY"] = [
            float(v) for v in table_spawn_object_xy]
        cfg_dict["env"]["tableSpawnBehind"] = behind
        cfg_dict["env"]["tableSpawnBehindDY"] = float(table_spawn_behind_dy)
        cfg_dict["env"]["tableSpawnBehindAbove"] = float(
            table_spawn_behind_above)
        cfg_dict["env"]["tableSpawnCorrelatedXY"] = float(
            table_spawn_correlated_xy)
        cfg_dict["env"]["tableSpawnFingerCurlScale"] = float(
            table_spawn_finger_curl_scale)
        if not 0.0 <= float(table_spawn_finger_noise) <= 1.0:
            raise ValueError(
                "table_spawn_finger_noise must be in [0, 1], got "
                f"{table_spawn_finger_noise}")
        cfg_dict["env"]["tableSpawnFingerNoise"] = float(
            table_spawn_finger_noise)
        if not 0.0 <= float(table_spawn_arm_noise) <= 1.0:
            raise ValueError(
                "table_spawn_arm_noise must be in [0, 1], got "
                f"{table_spawn_arm_noise}")
        cfg_dict["env"]["tableSpawnArmNoise"] = float(
            table_spawn_arm_noise)
        if table_spawn_behind_tilt is not None:
            cfg_dict["env"]["tableSpawnBehindTilt"] = float(
                table_spawn_behind_tilt)
        cfg_dict["env"]["tableSpawnTowardBucket"] = toward_bucket
        cfg_dict["env"]["tableSpawnInHand"] = bool(table_spawn_in_hand)
        cfg_dict["env"]["tableSpawnInHandOffset"] = float(
            table_spawn_in_hand_offset)
        cfg_dict["env"]["tableSpawnInHandObjNoise"] = float(
            table_spawn_in_hand_obj_noise)
        cfg_dict["env"]["tableSpawnInHandKeepArm"] = bool(
            table_spawn_in_hand_keep_arm)
        cfg_dict["env"]["tableSpawnInHandWristOffset"] = float(
            table_spawn_in_hand_wrist_offset)
        cfg_dict["env"]["tableSpawnInHandWristNoise"] = float(
            table_spawn_in_hand_wrist_noise)
        cfg_dict["env"]["largeTable"] = bool(large_table)
        cfg_dict["env"]["hideTable"] = bool(hide_table)
        cfg_dict["env"]["resetObjectZAbove"] = float(reset_z_above or 0.0)
        cfg_dict["env"]["controlSanityMode"] = str(
            control_sanity_mode or "")
        cfg_dict["env"]["controlSanityTrimSA"] = bool(
            control_sanity_trim_sa)
        cfg_dict["env"]["controlSanityTrimInitRangeFrac"] = float(
            control_sanity_trim_init_range_frac)
        mode = str(control_sanity_trim_init_mode or "curled").strip().lower()
        if mode not in CONTROL_SANITY_TRIM_INIT_MODES:
            raise ValueError(
                "control_sanity_trim_init_mode must be one of "
                f"{CONTROL_SANITY_TRIM_INIT_MODES}, got {mode!r}")
        cfg_dict["env"]["controlSanityTrimInitMode"] = mode
        if control_sanity_mode:
            cfg_dict["env"]["randomizeObjectDimensions"] = False
            cfg_dict["env"]["withSmallCuboids"] = False
            cfg_dict["env"]["withBigCuboids"] = False
            cfg_dict["env"]["withSticks"] = False
            cfg_dict["env"]["resetPositionNoiseX"] = 0.0
            cfg_dict["env"]["resetPositionNoiseY"] = 0.0
            cfg_dict["env"]["resetPositionNoiseZ"] = 0.0
            cfg_dict["env"]["resetRotationNoise"] = 0.0
            cfg_dict["env"]["resetDofPosRandomIntervalArm"] = 0.0
            cfg_dict["env"]["resetDofVelRandomInterval"] = 0.0
            cfg_dict["env"]["forceScale"] = 0.0
            cfg_dict["task"]["randomize"] = False
            # Finger init: NVIDIA default is 0.1 * Uniform(lo-q0, hi-q0).
            # Only freeze it when randomize_init is off. Arm stays frozen
            # so the figure goal is not mixed with arm drift.
            if not randomize_init:
                cfg_dict["env"]["resetDofPosRandomIntervalFingers"] = 0.0
        if table_spawn:
            # Cube on desk near the robot, ±2 cm xy, yaw quat, small joint
            # noise so palm and object stay nearby. No z / force noise.
            # Contact-free stand-offs (behind / toward-bucket) are noiseless.
            _contact_free = (
                behind or toward_bucket
                or (bool(table_spawn_in_hand)
                    and bool(table_spawn_in_hand_keep_arm)))
            _xy = 0.0 if _contact_free else TABLE_SPAWN_XY
            _arm = 0.0 if _contact_free else TABLE_SPAWN_ARM_NOISE
            _fin = 0.0 if _contact_free else TABLE_SPAWN_FINGER_NOISE
            cfg_dict["env"]["resetPositionNoiseX"] = _xy
            cfg_dict["env"]["resetPositionNoiseY"] = _xy
            cfg_dict["env"]["resetPositionNoiseZ"] = 0.0
            cfg_dict["env"]["resetRotationNoise"] = 0.0
            cfg_dict["env"]["resetDofPosRandomIntervalFingers"] = _fin
            cfg_dict["env"]["resetDofPosRandomIntervalArm"] = _arm
            cfg_dict["env"]["resetDofVelRandomInterval"] = 0.0
            cfg_dict["env"]["forceScale"] = 0.0
            cfg_dict["task"]["randomize"] = False
        return cfg_dict

    # -- observation packing -------------------------------------------------
    def _object_xyz(self) -> Tensor:
        env = self._env
        if getattr(env, "object_pos", None) is not None:
            return env.object_pos
        # Fallback before compute_observations has ever run.
        return env.root_state_tensor[env.object_indices, 0:3]

    def _palm_xyz(self) -> Tensor:
        env = self._env
        palm = getattr(env, "palm_center_pos", None)
        if palm is not None:
            return palm
        return torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

    def _pack_obs(self) -> Tensor:
        """Build ``[state | goal]`` from the current env tensors.

        ``mixed`` (default) uses normalized q and physical qd, preserving old
        checkpoints. ``physical`` uses raw radians and radians/sec.
        ``fully_scaled`` uses normalized q and qd divided by the actor's
        per-joint physical velocity limits, clipped to [-1, 1].
        Palm and object xyz are live positions after ``compute_observations``.
        With ``control_sanity_trim_sa``, state is q,qd of goal hand joints;
        ``control_sanity_q_only`` drops qd for the explicit partial-state
        ablation.
        """
        env = self._env
        q_norm = env.obs_buf[:, :_NUM_ARM_HAND_DOFS]
        q_raw = env.arm_hand_dof_pos[:, :_NUM_ARM_HAND_DOFS]
        qd_raw = env.arm_hand_dof_vel[:, :_NUM_ARM_HAND_DOFS]
        if self.coordinate_mode == "physical":
            packed_q, packed_qd = q_raw, qd_raw
        elif self.coordinate_mode == "fully_scaled":
            # PhysX can overshoot position limits slightly (observed q_norm
            # ~=1.012), so bound the fully-scaled representation explicitly.
            packed_q = torch.clamp(q_norm, -1.0, 1.0)
            packed_qd = torch.clamp(
                qd_raw / self._joint_velocity_limits.unsqueeze(0),
                -1.0, 1.0)
        else:
            packed_q, packed_qd = q_norm, qd_raw
        joint_pos_vel = torch.cat([packed_q, packed_qd], dim=-1)
        object_xyz = self._object_xyz()                             # (E, 3)
        if self.control_sanity_mode in (
                "finger", "hand16", "hand16fig", "hand16ok", "hand16peace",
                "hand16point", "hand16gun", "arm23wave", "index", "six",
                "two_finger",
                "three2", "four2",
                "four2h", "four2m", "four2mh", "four2mm", "four2mmh", "four2mmx",
                "four2w", "index_thumb_straight"):
            if self._trim_hand_idx is not None:
                hand_q = packed_q[:, _NUM_ARM_DOFS:]
                hand_qd = packed_qd[:, _NUM_ARM_DOFS:]
                idx = self._trim_hand_idx
                if self.control_sanity_q_only:
                    state = hand_q[:, idx]
                else:
                    state = torch.cat(
                        [hand_q[:, idx], hand_qd[:, idx]], dim=-1)
                achieved = (
                    state if self.control_sanity_goal_include_qd
                    else hand_q[:, idx])
                if not torch.equal(
                        state[:, :achieved.shape[1]], achieved):
                    raise RuntimeError(
                        "HER state coordinates differ from packed achieved "
                        f"goal in coordinate_mode={self.coordinate_mode}")
            else:
                state = joint_pos_vel
        elif self.control_sanity_mode == "palm":
            state = torch.cat(
                [joint_pos_vel, self._palm_xyz()], dim=1)  # (E, 49)
        elif self.palm_goal:
            state = torch.cat(
                [joint_pos_vel, self._palm_xyz(), object_xyz], dim=1)  # (E, 52)
        else:
            state = torch.cat([joint_pos_vel, object_xyz], dim=1)      # (E, 49)
        if self.coordinate_mode == "fully_scaled":
            q_width = (
                len(self._trim_hand_idx)
                if self._trim_hand_idx is not None
                else _NUM_ARM_HAND_DOFS)
            if (torch.any(state[:, :q_width] < -1.0001)
                    or torch.any(state[:, :q_width] > 1.0001)):
                raise RuntimeError("fully_scaled q escaped [-1, 1]")
            if not self.control_sanity_q_only:
                qd_part = state[:, q_width:2 * q_width]
                if (torch.any(qd_part < -1.0001)
                        or torch.any(qd_part > 1.0001)):
                    raise RuntimeError("fully_scaled qd escaped [-1, 1]")
        return torch.cat([state, self._goal_batch], dim=1)

    def normalized_goal_action(self) -> Tensor:
        """Normalized absolute-target oracle action, independent of goal packing."""
        if self._control_sanity_goal_action_batch is None:
            raise RuntimeError(
                "normalized_goal_action is only defined for joint-control goals")
        action = self._control_sanity_goal_action_batch
        if torch.any(action < -1.0001) or torch.any(action > 1.0001):
            raise RuntimeError("normalized oracle action escaped [-1, 1]")
        return action

    def _unscale_joint_q(self, q: Tensor, dof_idx: int) -> Tensor:
        """Map joint angle → action in [-1, 1] (inverse of NVIDIA ``scale``)."""
        env = self._env
        lo = env.arm_hand_dof_lower_limits[dof_idx]
        hi = env.arm_hand_dof_upper_limits[dof_idx]
        return (2.0 * (q - lo) / (hi - lo + 1e-8) - 1.0).clamp(-1.0, 1.0)

    def _hold_pose_actions(self) -> Tensor:
        """Stay at the keep-arm / table-spawn pose.

        Arm actions are relative (0 = hold). Finger actions are absolute
        targets; 0 is mid-range and opens the hand, so we send the
        authored curled q.
        """
        env = self._env
        a = torch.zeros(
            (self.num_envs, ACTION_DIM), dtype=torch.float, device=self.device)
        per = getattr(env, "_table_spawn_per_env_dof_pos", None)
        if per is not None:
            q = per
            for j in range(_NUM_ARM_DOFS, ACTION_DIM):
                a[:, j] = self._unscale_joint_q(q[:, j], j)
        else:
            q = env.hand_arm_default_dof_pos
            for j in range(_NUM_ARM_DOFS, ACTION_DIM):
                a[:, j] = self._unscale_joint_q(q[j], j)
        if getattr(self, 'lock_arm_base', False):
            return a[:, LOCK_ARM_BASE_N_PROXIMAL:]
        return a

    def _expand_lock_arm_base_actions(self, actions: Tensor) -> Tensor:
        """Map 19-D ``[A5,A6,A7,fingers]`` → 23-D PhysX. A1–A4 stay 0."""
        e = int(actions.shape[0])
        if int(actions.shape[-1]) != int(LOCK_ARM_BASE_ACTION_DIM):
            raise ValueError(
                f'lock_arm_base expects actions (E, {LOCK_ARM_BASE_ACTION_DIM}), '
                f'got {tuple(actions.shape)}')
        full = torch.zeros(
            (e, ACTION_DIM), dtype=torch.float, device=actions.device)
        full[:, LOCK_ARM_BASE_N_PROXIMAL:] = actions
        return full

    def _expand_trimmed_actions(self, actions: Tensor) -> Tensor:
        """Map trim policy actions ``(E, n)`` → full PhysX actions ``(E, 23)``.

        Controlled hand joints use ``actions``. Arm deltas stay 0 (hold prior
        target). Other hand joints are driven to the default pose via the
        absolute-position action encoding.
        """
        e = int(actions.shape[0])
        full = torch.zeros(
            (e, ACTION_DIM), dtype=torch.float, device=actions.device)
        ctrl = self._trim_full_action_idx
        assert ctrl is not None and self._trim_hand_idx is not None
        full[:, ctrl] = actions
        default_q = self._env.hand_arm_default_dof_pos  # (23,)
        ctrl_set = set(ctrl)
        for j in range(_NUM_ARM_DOFS, ACTION_DIM):
            if j in ctrl_set:
                continue
            full[:, j] = self._unscale_joint_q(default_q[j], j)
        return full

    def _default_trim_actions(self) -> Tensor:
        """Trim actions that hold the default pose on controlled joints."""
        assert self._trim_full_action_idx is not None
        if self.control_sanity_mode == "index_thumb_straight":
            # Preserve each environment's independently randomized curled
            # reset through the initialization step.
            return torch.stack([
                self._unscale_joint_q(
                    self._env.arm_hand_dof_pos[:, j], j)
                for j in self._trim_full_action_idx
            ], dim=-1)
        default_q = self._env.hand_arm_default_dof_pos
        parts = [
            self._unscale_joint_q(default_q[j], j).expand(self.num_envs)
            for j in self._trim_full_action_idx
        ]
        return torch.stack(parts, dim=-1)

    # -- gym-like API (torch CUDA tensors) ----------------------------------
    def reset(self) -> Tensor:
        """Reset all envs; returns packed obs ``(E, obs_dim+goal_dim)``.

        The base ``VecTask.reset()`` does NOT reset physics or compute
        observations, so we force every env to reset and step once with zero
        actions to initialize object/goal/dof state and populate ``obs_buf``.
        """
        self._env.reset_buf[:] = 1
        self._env.progress_buf[:] = 0
        if self._trim_hand_idx is not None:
            hold = self._default_trim_actions()
            self._env.step(self._expand_trimmed_actions(hold))
        elif (getattr(self, 'table_spawn_in_hand', False)
              or getattr(self, 'table_spawn_in_hand_keep_arm', False)):
            # Zero finger actions map to mid-range joints and open the
            # hand, launching a palm-up cube. Hold the authored pose.
            hold = self._hold_pose_actions()
            if getattr(self, 'lock_arm_base', False):
                hold = self._expand_lock_arm_base_actions(hold)
            self._env.step(hold)
            if getattr(self, 'table_spawn_in_hand_keep_arm', False):
                self._env._settle_cube_in_hand()
                self._env.compute_observations()
        else:
            zero = torch.zeros(
                (self.num_envs, self.action_dim), dtype=torch.float,
                device=self.device)
            self._env.step(zero)
        obs = self._pack_obs()
        if (self._trim_hand_idx is not None and self.randomize_init
                and not self._logged_random_trim_init):
            n = len(self._trim_hand_idx)
            q0_raw = self._env.arm_hand_dof_pos[
                :, self._trim_full_action_idx]
            lo = self._env.arm_hand_dof_lower_limits[
                self._trim_full_action_idx]
            hi = self._env.arm_hand_dof_upper_limits[
                self._trim_full_action_idx]
            q0_norm = 2.0 * (q0_raw - lo) / (hi - lo + 1e-8) - 1.0
            packed_q = obs[:, :n]
            packed_qd = obs[:, n:2 * n]
            q0_norm_min = tuple(round(float(v), 4) for v in
                                q0_norm.min(dim=0).values.tolist())
            q0_norm_max = tuple(round(float(v), 4) for v in
                                q0_norm.max(dim=0).values.tolist())
            q0_raw_min = tuple(round(float(v), 4) for v in
                               q0_raw.min(dim=0).values.tolist())
            q0_raw_max = tuple(round(float(v), 4) for v in
                               q0_raw.max(dim=0).values.tolist())
            packed_q_range = (
                float(packed_q.min().item()), float(packed_q.max().item()))
            packed_qd_range = (
                float(packed_qd.min().item()), float(packed_qd.max().item()))
            velocity_limits = self._joint_velocity_limits[
                self._trim_full_action_idx]
            initial_success = self.success()
            initial_success_frac = float(initial_success.mean().item())
            broad_index_reset = (
                self.control_sanity_mode == "index_thumb_straight"
                and (self.control_sanity_trim_init_mode == "full_range"
                     or self.control_sanity_trim_init_range_frac
                     > CONTROL_SANITY_TRIM_INIT_RANGE_FRAC))
            if broad_index_reset and bool(initial_success.any().item()):
                raise RuntimeError(
                    "index_thumb_straight broad reset produced successful "
                    f"initial states: fraction={initial_success_frac:.8f}")
            rejected = int(getattr(
                self._env, "_trim_init_rejected_total", 0))
            sampled = int(getattr(
                self._env, "_trim_init_sampled_total", 0))
            print(
                '[allegro_kuka_throw] randomized trim reset verified: '
                f'init_mode={self.control_sanity_trim_init_mode} '
                f'range_frac={self.control_sanity_trim_init_range_frac:g} '
                f'initial_success_frac={initial_success_frac:.8f} '
                f'rejected={rejected}/{sampled} '
                f'q_raw_rad_min={q0_raw_min} q_raw_rad_max={q0_raw_max} '
                f'q_normalized_min={q0_norm_min} '
                f'q_normalized_max={q0_norm_max} '
                f'packed_q_range=({packed_q_range[0]:.6g},'
                f'{packed_q_range[1]:.6g}) '
                f'packed_qd_range=({packed_qd_range[0]:.6g},'
                f'{packed_qd_range[1]:.6g}) '
                f'velocity_limits_rad_s='
                f'{tuple(round(float(v), 6) for v in velocity_limits.tolist())}; '
                'uncontrolled DOFs=default',
                flush=True)
            self._logged_random_trim_init = True
        return obs

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Step all envs with ``actions`` ``(E, action_dim)`` torch CUDA tensor.

        Returns ``(next_obs, reward, done)`` — all torch CUDA.
        Isaac Gym auto-resets envs that terminate; ``next_obs`` already reflects
        the post-reset state for those envs.
        """
        if (not torch.all(torch.isfinite(actions))
                or torch.any(actions < -1.0001)
                or torch.any(actions > 1.0001)):
            raise ValueError(
                "Allegro policy actions must be finite normalized values in "
                f"[-1, 1], got range=({actions.min().item():.6g}, "
                f"{actions.max().item():.6g})")
        if self._trim_hand_idx is not None:
            if int(actions.shape[-1]) != int(self.action_dim):
                raise ValueError(
                    f'trim_sa expects actions (E, {self.action_dim}), '
                    f'got {tuple(actions.shape)}')
            actions = self._expand_trimmed_actions(actions)
        elif getattr(self, 'lock_arm_base', False):
            actions = self._expand_lock_arm_base_actions(actions)
        self._env.step(actions)
        next_obs = self._pack_obs()
        reward = self._env.rew_buf.detach()
        done = self._env.reset_buf.detach().clone()
        fall = getattr(self._env, "_last_fall", None)
        if fall is None:
            self.last_fall = torch.zeros(
                (self.num_envs,), dtype=torch.float, device=self.device)
        else:
            self.last_fall = fall.detach()
        return next_obs, reward, done

    # -- success signal for logging -----------------------------------------
    def success(self) -> Tensor:
        """Per-env sparse success for the configured goal."""
        if self.control_sanity_mode in CONTROL_SANITY_LEVEL_MODES:
            if not self._logged_success_coordinates:
                print(
                    "[allegro_kuka_throw] success coordinates: physical raw "
                    "q radians versus physical raw goal radians",
                    flush=True)
                self._logged_success_coordinates = True
            return self.success_levels()["hard"]
        if self.control_sanity_mode == "finger":
            q = self._env.arm_hand_dof_pos[:, _NUM_ARM_DOFS:]
            assert self._control_sanity_goal_q_batch is not None
            err = torch.max(
                torch.abs(q - self._control_sanity_goal_q_batch),
                dim=-1).values
            return (err <= self.control_sanity_finger_tol).float()
        if self.control_sanity_mode == "palm":
            err = torch.norm(self._palm_xyz() - self._goal_batch, dim=-1)
            return (err <= self.control_sanity_palm_tol).float()
        if self.table_push:
            env = self._env
            dist = torch.norm(env.object_pos - env.goal_pos, dim=-1)
            tol = float(getattr(env, "success_tolerance", 0.075))
            ok = dist <= tol
            if self.palm_and_object_success and self.palm_goal:
                palm = self._palm_xyz()
                palm_g = self._goal_batch[:, :3]
                ok = ok & (torch.norm(palm - palm_g, dim=-1) <= tol)
            return ok.float()
        if self.throw_success == THROW_SUCCESS_NVIDIA_GOAL:
            return self.nvidia_goal()
        if self.throw_success == THROW_SUCCESS_GOAL_BALL:
            return self.goal_ball()
        return self.in_bucket()

    def _goal_ball(self, radius: float) -> Tensor:
        env = self._env
        dist = torch.norm(env.object_pos - env.goal_pos, dim=-1)
        return (dist <= radius).float()

    def nvidia_goal(self) -> Tensor:
        """Cube center in NVIDIA's one-frame throw goal ball.

        Goal is bucket xy, floor + 0.05 m (``env.goal_pos``). Radius is
        ``success_tolerance * keypoint_scale`` (0.075 × 1.5 = 0.1125 m).
        """
        env = self._env
        tol = float(getattr(env, "success_tolerance", 0.075))
        scale = float(getattr(env, "keypoint_scale", 1.5))
        return self._goal_ball(tol * scale)

    def goal_ball(self) -> Tensor:
        """Cube center in the 7.5 cm one-frame throw goal ball.

        Same center as ``nvidia_goal`` (bucket xy, floor + 0.05 m) but
        radius is ``success_tolerance`` only (no keypoint_scale).
        """
        env = self._env
        tol = float(getattr(env, "success_tolerance", 0.075))
        return self._goal_ball(tol)

    def in_bucket(self) -> Tensor:
        """Cube center inside the physical bucket cylinder.

        NVIDIA ``bucket.obj``: floor at the actor origin, height 0.198 m,
        inner xy radius 0.12 m. Default throw ``success()``.
        """
        env = self._env
        obj = env.object_pos
        bidx = env.bucket_object_indices
        bucket = env.root_state_tensor[bidx, 0:3]
        xy = torch.norm(obj[:, :2] - bucket[:, :2], dim=-1)
        z_in = obj[:, 2] - bucket[:, 2]
        return (
            (xy <= BUCKET_INNER_RADIUS)
            & (z_in >= 0.0)
            & (z_in <= BUCKET_HEIGHT)
        ).float()

    def success_levels(self):
        """Nested index-goal success tensors for diagnostics.

        Also returns continuous diagnostics for joint-goal modes:
          joint_frac_010 / joint_frac_020 — fraction of goal joints within
            0.10 / 0.20 rad (per env, in [0, 1])
          mean_abs_joint_err — mean |q - q*| over goal joints (radians)
        """
        if self.control_sanity_mode not in CONTROL_SANITY_LEVEL_MODES:
            success = self.success()
            z = torch.zeros_like(success)
            return {
                "hard": success,
                "easy": success,
                "very_easy": success,
                "joint_frac_010": z,
                "joint_frac_020": z,
                "mean_abs_joint_err": z,
            }
        if self.control_sanity_full_q:
            q = self._env.arm_hand_dof_pos[:, :_NUM_ARM_HAND_DOFS]
        else:
            hand_q = self._env.arm_hand_dof_pos[:, _NUM_ARM_DOFS:]
            if self.control_sanity_hand_indices is not None:
                idx = list(self.control_sanity_hand_indices)
                q = hand_q[:, idx]
            else:
                width = int(self._control_sanity_goal_q_batch.shape[1])
                q = hand_q[:, :width]
        assert self._control_sanity_goal_q_batch is not None
        abs_err = torch.abs(q - self._control_sanity_goal_q_batch)
        max_err = torch.max(abs_err, dim=-1).values
        if self.control_sanity_mode == "index_thumb_straight":
            balanced = (
                (abs_err.mean(dim=-1) <= 0.15) & (max_err <= 0.30)).float()
            hard = easy = very_easy = balanced
        else:
            hard = (max_err <= 0.10).float()
            easy = (max_err <= 0.20).float()
            very_easy = (max_err <= 0.30).float()
        return {
            "hard": hard,
            "easy": easy,
            "very_easy": very_easy,
            "joint_frac_010": (abs_err <= 0.10).float().mean(dim=-1),
            "joint_frac_020": (abs_err <= 0.20).float().mean(dim=-1),
            "mean_abs_joint_err": abs_err.mean(dim=-1),
        }

    def _setup_camera(self) -> None:
        """Attach an RGB camera to env 0 (headless GPU rendering)."""
        env = self._env
        w, h = self._cam_wh
        props = gymapi.CameraProperties()
        props.width = int(w)
        props.height = int(h)
        props.enable_tensors = False
        # Tighter FOV so a 7.5 cm success disk is actually visible.
        keep_arm = bool(getattr(self, 'table_spawn_in_hand_keep_arm', False))
        table_spawn = bool(getattr(self, 'table_spawn', False))
        hfov = (
            42.0 if keep_arm
            else (50.0 if (table_spawn or self.table_push) else 75.0))
        if self._cam_hfov_override is not None:
            hfov = float(self._cam_hfov_override)
        if hasattr(props, 'horizontal_fov'):
            props.horizontal_fov = float(hfov)
        env_ptr = env.envs[0]
        self._cam_handle = env.gym.create_camera_sensor(env_ptr, props)
        # Hardcoded table_spawn eye (0.72, −0.62) sits on a far −y bucket
        # (e.g. 0.45, −0.60) and crops it. Frame from bucket + spawn + arm.
        if keep_arm or table_spawn:
            pts: List[Tuple[float, float, float]] = []
            if self._object_goal_xyz is not None:
                pts.append(self._object_goal_xyz)
            if keep_arm and self._palm_goal_xyz_cmd is not None:
                pts.append(self._palm_goal_xyz_cmd)
            pts.append(self._bucket_xyz)
            if table_spawn:
                sx, sy = self.table_spawn_object_xy
                pts.append((float(sx), float(sy), float(TABLE_OBJECT_Z)))
                if bool(getattr(self, 'table_spawn_behind', False)):
                    dy = float(getattr(self, 'table_spawn_behind_dy', 0.10))
                    above = float(
                        getattr(self, 'table_spawn_behind_above', 0.10))
                    pts.append((
                        float(sx), float(sy) + dy,
                        float(TABLE_OBJECT_Z) + above))
            eye, tgt = frame_keep_arm_camera(pts, hfov_deg=hfov)
            cam_pos = gymapi.Vec3(*eye)
            cam_tgt = gymapi.Vec3(*tgt)
        elif self.table_push:
            # Over the desk: spawn (0.17, 0.08) and object goal (0.20, -0.15).
            cam_pos = gymapi.Vec3(0.72, -0.62, 0.98)
            cam_tgt = gymapi.Vec3(0.14, -0.02, 0.54)
        else:
            cam_pos = gymapi.Vec3(1.4, -1.6, 1.2)
            cam_tgt = gymapi.Vec3(0.0, 0.0, 0.35)
        if self._cam_eye_override is not None:
            cam_pos = gymapi.Vec3(*[float(v) for v in self._cam_eye_override])
        if self._cam_tgt_override is not None:
            cam_tgt = gymapi.Vec3(*[float(v) for v in self._cam_tgt_override])
        self._cam_eye = (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z))
        self._cam_tgt = (float(cam_tgt.x), float(cam_tgt.y), float(cam_tgt.z))
        self._cam_hfov = float(hfov)
        env.gym.set_camera_location(self._cam_handle, env_ptr, cam_pos, cam_tgt)
        print(f'[allegro_kuka_throw] camera {w}x{h} fov={hfov} '
              f'eye={self._cam_eye} tgt={self._cam_tgt}', flush=True)

    def render_rgb(self):
        """Return one RGB frame (H, W, 3) uint8 from env 0, or None."""
        import numpy as np
        if self._cam_handle is None:
            return None
        env = self._env
        env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(
            env.sim, env.envs[0], self._cam_handle, gymapi.IMAGE_COLOR)
        w, h = self._cam_wh
        img = np.asarray(img, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return img
