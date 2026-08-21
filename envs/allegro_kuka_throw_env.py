"""Isaac Gym AllegroKukaThrow adapter for the sgcrl PPO+NF pipeline.

This wraps NVIDIA IsaacGymEnvs' ``AllegroKukaThrow`` (Kuka iiwa7 arm + Allegro
16-DOF hand throwing an object toward a bucket) into a *batched* env that the
dedicated Isaac Gym learner (``contrastive/ppo_learner_isaacgym.py``) can drive.

Key differences from the upstream task:

* **Fixed goal.** Upstream randomizes the bucket / goal position every reset.
  Here we consider the bucket position *fixed* (a constant of the environment),
  so it never appears in the observation. ``_reset_target`` is overridden to
  place the bucket + goal at a single constant world-frame location.

* **Compact goal-conditioned observation.** The sgcrl NF density models
  ``log p(phi(state) | psi(goal))``. We repack the raw 99-D full-state buffer
  into a compact ``[state (49) | goal (3)]`` layout:

    - ``state[0:23]``  = joint positions   (arm 7 + hand 16), unscaled
    - ``state[23:46]`` = joint velocities
    - ``state[46:49]`` = object absolute xyz
    - ``goal[0:3]``    = fixed target xyz  (= the baked-in bucket location)

  ``obs_dim = 49`` (what phi encodes), ``goal_dim = 3`` (what psi encodes),
  packed obs length = 52.  ``goal_dim`` is exposed so ``make_environment`` /
  the learner keep the packed goal intact (same convention as BuilderBench).

Everything is kept on-GPU as torch CUDA tensors; the learner converts them to
JAX via DLPack (zero-copy).  We deliberately do NOT go through numpy per step.

IMPORTANT: ``import isaacgym`` (and this module) must be imported *before*
``torch`` anywhere in the process, per Isaac Gym's requirement.
"""
from __future__ import annotations

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

# Observation layout constants.
_NUM_ARM_HAND_DOFS = 23           # 7 arm + 16 hand
STATE_DIM = 2 * _NUM_ARM_HAND_DOFS + 3   # joint pos + joint vel + object xyz = 49
GOAL_DIM = 3                              # fixed target xyz
PACKED_OBS_DIM = STATE_DIM + GOAL_DIM     # 52
ACTION_DIM = _NUM_ARM_HAND_DOFS           # 23


class FixedGoalAllegroKukaThrow(AllegroKukaThrow):
    """AllegroKukaThrow with a single fixed bucket / goal position.

    The fixed target is read from ``cfg['env']['fixedTargetXYZ']`` if present,
    else falls back to :data:`DEFAULT_FIXED_TARGET_XYZ`.
    """

    def __init__(self, cfg, *args, **kwargs):
        tgt = cfg.get("env", {}).get("fixedTargetXYZ", DEFAULT_FIXED_TARGET_XYZ)
        # Store as a plain tuple; tensors are built lazily (device not ready yet).
        self._fixed_target_xyz = tuple(float(v) for v in tgt)
        self._fixed_target_tensor: Optional[Tensor] = None
        super().__init__(cfg, *args, **kwargs)

    def _fixed_target(self) -> Tensor:
        if self._fixed_target_tensor is None:
            self._fixed_target_tensor = torch.tensor(
                self._fixed_target_xyz, dtype=torch.float, device=self.device
            )
        return self._fixed_target_tensor

    def _reset_target(self, env_ids: Tensor) -> None:
        """Place the bucket + goal at the fixed constant location (no randomness)."""
        tgt = self._fixed_target()
        n = len(env_ids)
        # Bucket actor position (x, y, z).
        self.root_state_tensor[self.bucket_object_indices[env_ids], 0:3] = tgt.expand(n, 3)

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
      - ``obs_dim``           49  (state length, what phi encodes)
      - ``goal_dim``          3   (packed goal length, what psi encodes)
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
        cfg_dict = self._build_cfg(
            num_envs=num_envs,
            episode_length=episode_length,
            fixed_target_xyz=fixed_target_xyz,
            cfg_dir=cfg_dir,
            pipeline=pipeline,
            enable_cameras=bool(enable_cameras),
        )
        print(f'[allegro_kuka_throw] pipeline={pipeline} '
              f'sim_device={sim_device} rl_device={rl_device} E={int(num_envs)} '
              f'cameras={bool(enable_cameras)}')
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
        self.obs_dim = STATE_DIM
        self.goal_dim = GOAL_DIM
        self.action_dim = ACTION_DIM
        self.max_episode_steps = int(episode_length)
        self.device = self._env.device
        # Fixed goal tensor broadcast to (E, 3), reused every step.
        self._goal_batch = torch.tensor(
            [float(v) for v in fixed_target_xyz], dtype=torch.float, device=self.device
        ).unsqueeze(0).expand(self.num_envs, GOAL_DIM).contiguous()
        if bool(enable_cameras):
            self._setup_camera()

    @staticmethod
    def _build_cfg(num_envs, episode_length, fixed_target_xyz, cfg_dir,
                   pipeline="gpu", enable_cameras=False):
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
        if enable_cameras:
            # Keep graphics_device_id >= 0 in VecTask even when headless.
            cfg_dict["env"]["enableCameraSensors"] = True
        return cfg_dict

    # -- observation packing -------------------------------------------------
    def _object_xyz(self) -> Tensor:
        env = self._env
        if getattr(env, "object_pos", None) is not None:
            return env.object_pos
        # Fallback before compute_observations has ever run.
        return env.root_state_tensor[env.object_indices, 0:3]

    def _pack_obs(self) -> Tensor:
        """Build ``[state(49) | goal(3)]`` from the current env tensors.

        Uses the already-computed 99-D obs buffer for joint pos/vel (slices
        0:46, unscaled + clamped by the base task) and the live object position.
        """
        env = self._env
        joint_pos_vel = env.obs_buf[:, 0:2 * _NUM_ARM_HAND_DOFS]     # (E, 46)
        object_xyz = self._object_xyz()                             # (E, 3)
        state = torch.cat([joint_pos_vel, object_xyz], dim=1)        # (E, 49)
        return torch.cat([state, self._goal_batch], dim=1)          # (E, 52)

    # -- gym-like API (torch CUDA tensors) ----------------------------------
    def reset(self) -> Tensor:
        """Reset all envs; returns packed obs ``(E, 52)`` torch CUDA tensor.

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

        Returns ``(next_obs (E,52), reward (E,), done (E,))`` — all torch CUDA.
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
        """Attach a side-view RGB camera to env 0 (headless GPU rendering)."""
        env = self._env
        w, h = self._cam_wh
        props = gymapi.CameraProperties()
        props.width = int(w)
        props.height = int(h)
        props.enable_tensors = False
        env_ptr = env.envs[0]
        self._cam_handle = env.gym.create_camera_sensor(env_ptr, props)
        # Look from the front-right at the table / fixed bucket.
        cam_pos = gymapi.Vec3(1.4, -1.6, 1.2)
        cam_tgt = gymapi.Vec3(0.0, 0.0, 0.35)
        env.gym.set_camera_location(self._cam_handle, env_ptr, cam_pos, cam_tgt)
        print(f'[allegro_kuka_throw] camera {w}x{h} on env 0')

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
