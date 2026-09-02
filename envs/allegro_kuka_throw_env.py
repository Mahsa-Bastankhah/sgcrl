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
  (``palm_goal=False``):

    - ``state[0:23]``  = joint positions   (arm 7 + hand 16), unscaled
    - ``state[23:46]`` = joint velocities
    - ``state[46:49]`` = object absolute xyz
    - ``goal[0:3]``    = fixed bucket xyz

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

  ``table_spawn`` (``--isaacgym_table_spawn``, default off): cube on the
  robot-near side of the desk (``TABLE_SPAWN_OBJECT_XY``, z =
  ``TABLE_OBJECT_Z``) with small xy noise, yaw-only quat, and a one-shot
  IK reach so the palm sits on the cube (touching). Arm / finger reset
  noise is small so hand and object stay nearby. Startup raises if
  ``|palm−obj|`` is large. Does not flip ``randomize_init`` globally.

  Hindsight is the last ``goal_dim`` entries of the state. The default
  throw bucket is swapped for the slide lip when ``palm_goal`` is on (and
  ``table_push`` is off) and the caller left ``fixed_target_xyz`` at the
  throw default.

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
# NVIDIA table_narrow: 0.475 x 0.4 x 0.3 box at (0, 0, 0.38) → top z = 0.53.
# Spawn is table center (0, 0); robot base is (0, 0.8, 0).
# Nudge-only goal: 10 cm toward the robot. success_tolerance is 7.5 cm, so
# the cube is *not* already successful at spawn, but a short shove is enough.
TABLE_TOP_Z = 0.38 + 0.15
TABLE_OBJECT_Z = TABLE_TOP_Z + 0.025  # ~5 cm cube sitting on the desk
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
# Curled paddle (4 joints × index/mid/ring + thumb). Zeros spear the cube.
TABLE_SPAWN_BEHIND_FINGER_Q: Tuple[float, ...] = (
    0.55, 0.75, 0.65, 0.35,
    0.55, 0.75, 0.65, 0.35,
    0.55, 0.75, 0.65, 0.35,
    0.40, 0.70, 0.55, 0.35,
)


def _skew(v: Tensor) -> Tensor:
    z = torch.zeros(v.shape[0], device=v.device, dtype=v.dtype)
    x, y, zz = v[:, 0], v[:, 1], v[:, 2]
    row0 = torch.stack([z, -zz, y], dim=-1)
    row1 = torch.stack([zz, z, -x], dim=-1)
    row2 = torch.stack([-y, x, z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

# Observation layout constants.
_NUM_ARM_HAND_DOFS = 23           # 7 arm + 16 hand
STATE_DIM = 2 * _NUM_ARM_HAND_DOFS + 3   # q + qd + object xyz = 49
GOAL_DIM = 3                              # bucket xyz
PACKED_OBS_DIM = STATE_DIM + GOAL_DIM     # 52
STATE_DIM_PALM = 2 * _NUM_ARM_HAND_DOFS + 6  # q + qd + palm + object = 52
GOAL_DIM_PALM = 6                             # ideal palm + object in bucket
PACKED_OBS_DIM_PALM = STATE_DIM_PALM + GOAL_DIM_PALM  # 58
ACTION_DIM = _NUM_ARM_HAND_DOFS           # 23


def resolve_bucket_xyz(
    palm_goal: bool,
    fixed_target_xyz: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """If palm-goal is on and the throw default bucket was left in place, use the slide lip."""
    tgt = tuple(float(v) for v in fixed_target_xyz)
    if bool(palm_goal) and tgt == DEFAULT_FIXED_TARGET_XYZ:
        return SLIDE_BUCKET_XYZ
    return tgt


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
        self._table_spawn = bool(env_cfg.get("tableSpawn", False))
        self._table_spawn_behind = bool(env_cfg.get("tableSpawnBehind", False))
        super().__init__(cfg, *args, **kwargs)
        if self._table_spawn:
            self._ik_palm_onto_cube()
            from isaacgym import gymtorch
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
                f'palm_z={tuple(round(float(v), 3) for v in z_w.tolist())} '
                f'arm_q={tuple(round(float(v), 3) for v in self.hand_arm_default_dof_pos[:7].tolist())}',
                flush=True,
            )
            _palm_lim = (
                TABLE_SPAWN_BEHIND_MAX_PALM_OBJ
                if getattr(self, "_table_spawn_behind", False)
                else TABLE_SPAWN_MAX_PALM_OBJ)
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

    def _object_start_pose(self, allegro_pose, table_pose_dy, table_pose_dz):
        if not getattr(self, "_table_spawn", False):
            return super()._object_start_pose(
                allegro_pose, table_pose_dy, table_pose_dz)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3()
        if getattr(self, "_table_spawn_behind", False):
            pose.p.x = TABLE_SPAWN_BEHIND_OBJECT_XY[0]
            pose.p.y = TABLE_SPAWN_BEHIND_OBJECT_XY[1]
        else:
            pose.p.x = TABLE_SPAWN_OBJECT_XY[0]
            pose.p.y = TABLE_SPAWN_OBJECT_XY[1]
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
        if behind:
            stand = obj.clone()
            stand[:, 1] = stand[:, 1] + TABLE_SPAWN_BEHIND_PALM_DY
            stand[:, 2] = TABLE_OBJECT_Z + TABLE_SPAWN_BEHIND_PALM_ABOVE
            hover = stand.clone()
            hover[:, 2] = TABLE_OBJECT_Z + 0.18
            z_des = TABLE_SPAWN_BEHIND_Z_DES
            self._ik_palm_stage(
                jac, hover, n_iters=220, step=0.18, lam=0.12,
                name="hover", stop_at=0.04)
            self._ik_palm_stage(
                jac, hover, n_iters=220, step=0.12, lam=0.12,
                name="face-goal", stop_at=0.05, use_orn=True, orn_w=0.40,
                stop_ang=0.35, z_des_xyz=z_des)
            last_err = self._ik_palm_stage(
                jac, stand, n_iters=220, step=0.10, lam=0.10,
                name="behind", stop_at=0.03, use_orn=True, orn_w=0.30,
                stop_ang=0.40, z_des_xyz=z_des)
            want = 'want ~ (0,-1,0) = facing the cube / goal'
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
        if behind:
            fq = torch.tensor(
                TABLE_SPAWN_BEHIND_FINGER_Q, dtype=torch.float,
                device=self.device)
            self.hand_arm_default_dof_pos[7:23] = fq
            self.arm_hand_dof_pos[:, 7:23] = fq
            self.cur_targets[:, 7:23] = fq
            self.prev_targets[:, 7:23] = fq
            self._ik_apply_q(self.arm_hand_dof_pos[:, :7])
            self._set_object_root(None)
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

    def get_random_quat(self, env_ids: Tensor) -> Tensor:
        """Upstream always samples a uniform quat; freeze uses the spawn pose."""
        if getattr(self, "_table_spawn", False):
            n = len(env_ids)
            half = 0.5 * (
                torch.rand(n, device=self.device) * (2.0 * math.pi) - math.pi)
            q = torch.zeros((n, 4), dtype=torch.float, device=self.device)
            q[:, 2] = torch.sin(half)
            q[:, 3] = torch.cos(half)
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
        table_push: bool = False,
        table_push_xyz: Optional[Tuple[float, float, float]] = None,
        table_spawn: bool = False,
        table_spawn_behind: bool = False,
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
        self.palm_goal = bool(palm_goal)
        self.table_push = bool(table_push)
        self.table_spawn = bool(table_spawn)
        self.table_spawn_behind = bool(table_spawn_behind)
        self._cam_eye_override = camera_eye
        self._cam_tgt_override = camera_tgt
        self._cam_hfov_override = camera_hfov
        self.randomize_object_xyz = bool(randomize_object_xyz)
        self.randomize_object_shape = bool(randomize_object_shape)
        bucket_xyz = resolve_bucket_xyz(
            self.palm_goal and not self.table_push, fixed_target_xyz)
        object_goal_xyz: Optional[Tuple[float, float, float]] = None
        if self.table_push:
            # Park the bucket at the throw location so it is not sitting on
            # the desk. Success / packed goal use the on-table push target.
            bucket_xyz = DEFAULT_FIXED_TARGET_XYZ
            if table_push_xyz is None:
                object_goal_xyz = TABLE_PUSH_GOAL_XYZ
            else:
                object_goal_xyz = tuple(float(v) for v in table_push_xyz)
                if len(object_goal_xyz) != 3:
                    raise ValueError(
                        f'table_push_xyz must be 3 floats, got {table_push_xyz!r}')
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
            fixed_goal_xyz=object_goal_xyz,
            table_spawn=bool(table_spawn),
            table_spawn_behind=bool(table_spawn_behind),
        )
        _pos_noise = (
            cfg_dict.get('env', {}).get('resetPositionNoiseX'),
            cfg_dict.get('env', {}).get('resetPositionNoiseY'),
            cfg_dict.get('env', {}).get('resetPositionNoiseZ'),
        )
        print(f'[allegro_kuka_throw] pipeline={pipeline} '
              f'sim_device={sim_device} rl_device={rl_device} E={int(num_envs)} '
              f'cameras={bool(enable_cameras)} '
              f'randomize_init={bool(randomize_init)} '
              f'randomize_object_xyz={bool(randomize_object_xyz)} '
              f'randomize_object_shape={bool(randomize_object_shape)} '
              f'pos_noise_xyz={_pos_noise} '
              f'palm_goal={self.palm_goal} table_push={self.table_push} '
              f'table_spawn={self.table_spawn} '
              f'table_spawn_behind={self.table_spawn_behind} '
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
        self.action_dim = ACTION_DIM
        self.max_episode_steps = int(episode_length)
        self.device = self._env.device
        if self.palm_goal:
            self.obs_dim = STATE_DIM_PALM
            self.goal_dim = GOAL_DIM_PALM
            palm = SLIDE_PALM_XYZ if palm_goal_xyz is None else tuple(
                float(v) for v in palm_goal_xyz)
            if len(palm) != 3:
                raise ValueError(f'palm_goal_xyz must be 3 floats, got {palm!r}')
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
            goal_vec = tuple(float(v) for v in bucket_xyz)
        self._goal_batch = torch.tensor(
            list(goal_vec), dtype=torch.float, device=self.device
        ).unsqueeze(0).expand(self.num_envs, self.goal_dim).contiguous()
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
                   table_spawn_behind=False):
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
        cfg_dict["env"]["tableSpawn"] = bool(table_spawn)
        cfg_dict["env"]["tableSpawnBehind"] = bool(table_spawn_behind)
        if table_spawn:
            # Cube on desk near the robot, ±2 cm xy, yaw quat, small joint
            # noise so palm and object stay nearby. No z / force noise.
            # Behind-the-cube push spawn is noiseless (contact-free stand-off).
            _xy = 0.0 if table_spawn_behind else TABLE_SPAWN_XY
            _arm = 0.0 if table_spawn_behind else TABLE_SPAWN_ARM_NOISE
            _fin = 0.0 if table_spawn_behind else TABLE_SPAWN_FINGER_NOISE
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

        Joint pos/vel come from the 99-D obs buffer (slices 0:46). Palm and
        object xyz are live body positions after ``compute_observations``.
        """
        env = self._env
        joint_pos_vel = env.obs_buf[:, 0:2 * _NUM_ARM_HAND_DOFS]     # (E, 46)
        object_xyz = self._object_xyz()                             # (E, 3)
        if self.palm_goal:
            state = torch.cat(
                [joint_pos_vel, self._palm_xyz(), object_xyz], dim=1)  # (E, 52)
        else:
            state = torch.cat([joint_pos_vel, object_xyz], dim=1)      # (E, 49)
        return torch.cat([state, self._goal_batch], dim=1)

    # -- gym-like API (torch CUDA tensors) ----------------------------------
    def reset(self) -> Tensor:
        """Reset all envs; returns packed obs ``(E, obs_dim+goal_dim)``.

        The base ``VecTask.reset()`` does NOT reset physics or compute
        observations, so we force every env to reset and step once with zero
        actions to initialize object/goal/dof state and populate ``obs_buf``.
        """
        self._env.reset_buf[:] = 1
        self._env.progress_buf[:] = 0
        zero = torch.zeros(
            (self.num_envs, self.action_dim), dtype=torch.float, device=self.device
        )
        self._env.step(zero)
        return self._pack_obs()

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Step all envs with ``actions`` ``(E, 23)`` torch CUDA tensor.

        Returns ``(next_obs, reward, done)`` — all torch CUDA.
        Isaac Gym auto-resets envs that terminate; ``next_obs`` already reflects
        the post-reset state for those envs.
        """
        self._env.step(actions)
        next_obs = self._pack_obs()
        reward = self._env.rew_buf.detach()
        done = self._env.reset_buf.detach().clone()
        return next_obs, reward, done

    # -- success signal for logging -----------------------------------------
    def success(self) -> Tensor:
        """Per-env sparse success: object within success tolerance of the goal."""
        env = self._env
        dist = torch.norm(env.object_pos - env.goal_pos, dim=-1)
        tol = float(getattr(env, "success_tolerance", 0.075))
        return (dist <= tol).float()

    def _setup_camera(self) -> None:
        """Attach an RGB camera to env 0 (headless GPU rendering)."""
        env = self._env
        w, h = self._cam_wh
        props = gymapi.CameraProperties()
        props.width = int(w)
        props.height = int(h)
        props.enable_tensors = False
        # Tighter FOV so a 7.5 cm success disk is actually visible.
        hfov = 50.0 if (self.table_spawn or self.table_push) else 75.0
        if self._cam_hfov_override is not None:
            hfov = float(self._cam_hfov_override)
        if hasattr(props, 'horizontal_fov'):
            props.horizontal_fov = float(hfov)
        env_ptr = env.envs[0]
        self._cam_handle = env.gym.create_camera_sensor(env_ptr, props)
        if self.table_spawn or self.table_push:
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
