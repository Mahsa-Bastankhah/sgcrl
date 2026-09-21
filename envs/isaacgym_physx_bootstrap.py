"""Create Isaac Gym PhysX *before* JAX/TF initialize a CUDA context.

Isaac Gym Preview 4 GPU kernels fail to register
(``mergeChangedAABBMgrHandlesLaunch``) if JAX or TensorFlow already own the
device.  ``ppo_contrastive.py`` must call :func:`maybe_create_from_argv`
before ``import sgcrl_jax_acme_compat``.
"""
from __future__ import annotations

import sys
from typing import Any, Optional

_PREBUILT: Optional[Any] = None


def _flag_aliases(name: str):
  """absl ``foo_bar`` and tyro ``foo-bar`` spellings."""
  under = name.replace('-', '_')
  hyphen = under.replace('_', '-')
  return (under, hyphen) if under != hyphen else (under,)


def _argv_has_flag(argv, name: str) -> bool:
  for alias in _flag_aliases(name):
    eq, opt = f'--{alias}=', f'--{alias}'
    no_absl, no_tyro = f'--no{alias}', f'--no-{alias}'
    for a in argv:
      if a.startswith(eq) or a in (opt, no_absl, no_tyro):
        return True
  return False


def _argv_flag(argv, name: str, default: Optional[str] = None) -> Optional[str]:
  for alias in _flag_aliases(name):
    eq, opt = f'--{alias}=', f'--{alias}'
    for i, a in enumerate(argv):
      if a.startswith(eq):
        return a[len(eq):]
      if a == opt and i + 1 < len(argv) and not argv[i + 1].startswith('-'):
        return argv[i + 1]
  return default


def _argv_bool(argv, name: str, default: bool = True) -> bool:
  """Parse absl ``DEFINE_bool`` or tyro ``--foo-bar`` / ``--no-foo-bar``."""
  falsey = {'0', 'false', 'f', 'no', 'n', 'off'}
  truthy = {'1', 'true', 't', 'yes', 'y', 'on'}
  for alias in _flag_aliases(name):
    no_flags = (f'--no{alias}', f'--no-{alias}')
    eq, opt = f'--{alias}=', f'--{alias}'
    for i, a in enumerate(argv):
      if a in no_flags:
        return False
      if a.startswith(eq):
        v = a[len(eq):].strip().lower()
        if v in falsey:
          return False
        if v in truthy:
          return True
        raise ValueError(f'invalid bool for --{name}: {a!r}')
      if a == opt:
        if i + 1 < len(argv) and not argv[i + 1].startswith('-'):
          v = argv[i + 1].strip().lower()
          if v in falsey:
            return False
          if v in truthy:
            return True
        return True
  return bool(default)


def take_prebuilt_env() -> Optional[Any]:
  """Return and clear the process-wide prebuilt Allegro env (or None)."""
  global _PREBUILT
  env, _PREBUILT = _PREBUILT, None
  return env


def maybe_create_from_argv(argv=None) -> Optional[Any]:
  """If ``--env`` is allegro_kuka_*, construct GPU/CPU PhysX now.

  No-op for every other env so importing ``ppo_contrastive`` stays cheap.
  """
  global _PREBUILT
  if _PREBUILT is not None:
    return _PREBUILT
  argv = list(sys.argv if argv is None else argv)
  env_name = (
      _argv_flag(argv, 'env')
      or _argv_flag(argv, 'env_id')
      or _argv_flag(argv, 'env-id'))
  if not env_name or not str(env_name).startswith('allegro_kuka'):
    return None

  already = [m for m in ('jax', 'tensorflow', 'torch') if m in sys.modules]
  if already:
    print('[isaacgym_bootstrap] WARNING: already imported '
          f'{already} before PhysX create_sim; GPU kernels may fail',
          flush=True)

  # SAC: --sac_num_envs; contrastive PPO: --ppo_num_envs; flax PPO+RND: --num-envs.
  raw_envs = (
      _argv_flag(argv, 'sac_num_envs')
      or _argv_flag(argv, 'ppo_num_envs')
      or _argv_flag(argv, 'mpo_num_envs')
      or _argv_flag(argv, 'num_envs'))
  if raw_envs is None or int(raw_envs) < 0:
    # env-id path is flax ppo-rnd (Allegro default 1024); --env is contrastive PPO.
    used_env_id = _argv_flag(argv, 'env_id') or _argv_flag(argv, 'env-id')
    num_envs = 1024 if used_env_id else 8
  else:
    num_envs = int(raw_envs)
  pipeline = str(_argv_flag(argv, 'isaacgym_pipeline', 'gpu') or 'gpu').strip().lower()
  ep_len = int(_argv_flag(argv, 'isaacgym_episode_length', '300') or '300')
  tgt_raw = _argv_flag(argv, 'isaacgym_fixed_target_xyz', '0.5,-0.3,0.4')
  xyz = tuple(float(x) for x in str(tgt_raw).split(','))
  if len(xyz) != 3:
    raise ValueError(f'isaacgym_fixed_target_xyz must be 3 floats, got {tgt_raw!r}')
  goal_z_raw = _argv_flag(argv, 'isaacgym_goal_z', '-1')
  goal_z = float(goal_z_raw) if goal_z_raw not in (None, '') else -1.0
  goal_z = None if goal_z < 0.0 else goal_z
  seed = int(_argv_flag(argv, 'seed', '0') or '0')
  used_env_id = bool(
      _argv_flag(argv, 'env_id') or _argv_flag(argv, 'env-id'))
  # flax ppo-rnd Allegro: xyz-only rand. contrastive PPO (--env): NVIDIA defaults.
  randomize_init = _argv_bool(
      argv, 'isaacgym_randomize_init', False if used_env_id else True)
  randomize_object_xyz = _argv_bool(
      argv, 'isaacgym_randomize_object_xyz', True if used_env_id else False)
  randomize_object_shape = _argv_bool(
      argv, 'isaacgym_randomize_object_shape', False if used_env_id else True)
  palm_goal = _argv_bool(argv, 'isaacgym_palm_goal', False)
  joint_goal = _argv_bool(argv, 'isaacgym_joint_goal', False)
  control_sanity_mode = str(
      _argv_flag(argv, 'isaacgym_control_sanity_mode', 'off')
      or 'off').strip().lower()
  if control_sanity_mode == 'off':
    control_sanity_mode = ''
  if control_sanity_mode not in (
      '', 'finger', 'index', 'six', 'two_finger', 'three2', 'four2', 'four2h',
      'four2m', 'four2mh', 'four2mm', 'four2mmh', 'four2mmx', 'four2w',
      'index_thumb_straight', 'hand16', 'hand16fig', 'hand16ok',
      'hand16peace', 'hand16point', 'hand16gun', 'arm23wave', 'palm'):
    raise ValueError(
        'isaacgym_control_sanity_mode must be off, finger, index, '
        'six, two_finger, three2, four2, four2h, four2m, four2mh, four2mm, '
        'four2mmh, four2mmx, four2w, index_thumb_straight, hand16, '
        'hand16fig, hand16ok, hand16peace, hand16point, hand16gun, '
        'arm23wave, or palm')
  sanity_palm_raw = _argv_flag(
      argv, 'isaacgym_control_sanity_palm_xyz', '0.0,0.0,0.80')
  sanity_palm_xyz = tuple(
      float(x) for x in str(sanity_palm_raw).split(','))
  if len(sanity_palm_xyz) != 3:
    raise ValueError('isaacgym_control_sanity_palm_xyz must be 3 floats')
  sanity_finger_tol = float(_argv_flag(
      argv, 'isaacgym_control_sanity_finger_tol', '0.15') or 0.15)
  sanity_palm_tol = float(_argv_flag(
      argv, 'isaacgym_control_sanity_palm_tol', '0.05') or 0.05)
  control_sanity_trim_sa = _argv_bool(
      argv, 'isaacgym_control_sanity_trim_sa', False)
  if control_sanity_mode == 'index_thumb_straight':
    control_sanity_trim_sa = True
  control_sanity_trim_init_range_frac = float(_argv_flag(
      argv, 'isaacgym_control_sanity_trim_init_range_frac', '0.10') or 0.10)
  if not 0.0 <= control_sanity_trim_init_range_frac <= 1.0:
    raise ValueError(
        'isaacgym_control_sanity_trim_init_range_frac must be in [0, 1], got '
        f'{control_sanity_trim_init_range_frac}')
  control_sanity_trim_init_mode = str(_argv_flag(
      argv, 'isaacgym_control_sanity_trim_init_mode', 'curled')
      or 'curled').strip().lower()
  if control_sanity_trim_init_mode not in ('curled', 'full_range'):
    raise ValueError(
        'isaacgym_control_sanity_trim_init_mode must be curled or '
        f'full_range, got {control_sanity_trim_init_mode!r}')
  control_sanity_q_only = _argv_bool(
      argv, 'isaacgym_control_sanity_q_only', False)
  control_sanity_goal_include_qd = _argv_bool(
      argv, 'isaacgym_control_sanity_goal_include_qd', False)
  coordinate_mode = str(_argv_flag(
      argv, 'isaacgym_coordinate_mode', 'mixed') or 'mixed').strip().lower()
  if coordinate_mode not in ('mixed', 'physical', 'fully_scaled'):
    raise ValueError(
        'isaacgym_coordinate_mode must be mixed, physical, or fully_scaled, '
        f'got {coordinate_mode!r}')
  if control_sanity_trim_sa and not control_sanity_mode:
    raise ValueError(
        'isaacgym_control_sanity_trim_sa requires '
        'isaacgym_control_sanity_mode != off')
  if control_sanity_q_only and not control_sanity_trim_sa:
    raise ValueError(
        'isaacgym_control_sanity_q_only requires '
        'isaacgym_control_sanity_trim_sa')
  if control_sanity_goal_include_qd and (
      not control_sanity_trim_sa or control_sanity_q_only):
    raise ValueError(
        'isaacgym_control_sanity_goal_include_qd requires trim-SA q,qd state')
  if joint_goal:
    palm_goal = False
  if control_sanity_mode:
    palm_goal = False
    joint_goal = False
  table_push = _argv_bool(argv, 'isaacgym_table_push', False)
  table_spawn = _argv_bool(argv, 'isaacgym_table_spawn', False)
  table_spawn_behind = _argv_bool(
      argv, 'isaacgym_table_spawn_behind', False)
  table_spawn_behind_dy = float(
      _argv_flag(argv, 'isaacgym_table_spawn_behind_dy', '0.14') or 0.14)
  table_spawn_behind_above = float(
      _argv_flag(argv, 'isaacgym_table_spawn_behind_above', '0.08') or 0.08)
  table_spawn_correlated_xy = float(
      _argv_flag(argv, 'isaacgym_table_spawn_correlated_xy', '0') or 0)
  table_spawn_finger_curl_scale = float(
      _argv_flag(argv, 'isaacgym_table_spawn_finger_curl_scale', '1') or 1)
  table_spawn_finger_noise = float(
      _argv_flag(argv, 'isaacgym_table_spawn_finger_noise', '0') or 0)
  if not 0.0 <= table_spawn_finger_noise <= 1.0:
    raise ValueError(
        'isaacgym_table_spawn_finger_noise must be in [0, 1], got '
        f'{table_spawn_finger_noise}')
  table_spawn_arm_noise = float(
      _argv_flag(argv, 'isaacgym_table_spawn_arm_noise', '0') or 0)
  if not 0.0 <= table_spawn_arm_noise <= 1.0:
    raise ValueError(
        'isaacgym_table_spawn_arm_noise must be in [0, 1], got '
        f'{table_spawn_arm_noise}')
  table_spawn_toward_bucket = _argv_bool(
      argv, 'isaacgym_table_spawn_toward_bucket', False)
  table_spawn_in_hand = _argv_bool(
      argv, 'isaacgym_table_spawn_in_hand', False)
  table_spawn_in_hand_offset = float(
      _argv_flag(argv, 'isaacgym_table_spawn_in_hand_offset', '0.042')
      or 0.042)
  table_spawn_in_hand_obj_noise = float(
      _argv_flag(argv, 'isaacgym_table_spawn_in_hand_obj_noise', '0.012')
      or 0.012)
  if table_spawn_in_hand_offset <= 0.0:
    raise ValueError(
        'isaacgym_table_spawn_in_hand_offset must be positive, got '
        f'{table_spawn_in_hand_offset}')
  if table_spawn_in_hand_obj_noise < 0.0:
    raise ValueError(
        'isaacgym_table_spawn_in_hand_obj_noise must be non-negative, got '
        f'{table_spawn_in_hand_obj_noise}')
  if table_spawn_in_hand and not table_spawn:
    raise ValueError(
        'isaacgym_table_spawn_in_hand requires isaacgym_table_spawn')
  table_spawn_in_hand_keep_arm = _argv_bool(
      argv, 'isaacgym_table_spawn_in_hand_keep_arm', False)
  table_spawn_in_hand_wrist_offset = float(
      _argv_flag(
          argv, 'isaacgym_table_spawn_in_hand_wrist_offset',
          '-3.141592653589793')
      or -3.141592653589793)
  table_spawn_in_hand_wrist_noise = float(
      _argv_flag(argv, 'isaacgym_table_spawn_in_hand_wrist_noise', '0.10')
      or 0.10)
  if table_spawn_in_hand_keep_arm and not table_spawn_in_hand:
    raise ValueError(
        'isaacgym_table_spawn_in_hand_keep_arm requires '
        'isaacgym_table_spawn_in_hand')
  if table_spawn_in_hand_wrist_noise < 0.0:
    raise ValueError(
        'isaacgym_table_spawn_in_hand_wrist_noise must be non-negative, got '
        f'{table_spawn_in_hand_wrist_noise}')
  lock_arm_base = _argv_bool(argv, 'isaacgym_lock_arm_base', False)
  if lock_arm_base and control_sanity_trim_sa:
    raise ValueError(
        'isaacgym_lock_arm_base is incompatible with '
        'isaacgym_control_sanity_trim_sa')
  large_table = _argv_bool(argv, 'isaacgym_large_table', False)
  hide_table = _argv_bool(argv, 'isaacgym_hide_table', False)
  palm_and_object_success = _argv_bool(
      argv, 'isaacgym_palm_and_object_success', False)
  throw_success = str(
      _argv_flag(argv, 'isaacgym_throw_success', 'in_bucket')
      or 'in_bucket').strip().lower()
  if throw_success not in ('in_bucket', 'nvidia_goal', 'goal_ball'):
    raise ValueError(
        'isaacgym_throw_success must be in_bucket, nvidia_goal, or '
        f'goal_ball, got {throw_success!r}')
  if hide_table and large_table:
    raise ValueError(
        'isaacgym_hide_table and isaacgym_large_table are mutually exclusive')
  reset_z_above = float(
      _argv_flag(argv, 'isaacgym_reset_z_above', '0') or 0)
  if 'slide' in str(env_name) and not _argv_has_flag(argv, 'isaacgym_table_push'):
    table_push = True
  palm_raw = _argv_flag(argv, 'isaacgym_palm_goal_xyz')
  palm_xyz = None
  if palm_raw:
    palm_xyz = tuple(float(x) for x in str(palm_raw).split(','))
    if len(palm_xyz) != 3:
      raise ValueError(
          f'isaacgym_palm_goal_xyz must be 3 floats, got {palm_raw!r}')
  push_raw = _argv_flag(argv, 'isaacgym_table_push_xyz')
  push_xyz = None
  if push_raw:
    push_xyz = tuple(float(x) for x in str(push_raw).split(','))
    if len(push_xyz) != 3:
      raise ValueError(
          f'isaacgym_table_push_xyz must be 3 floats, got {push_raw!r}')
  spawn_raw = _argv_flag(argv, 'isaacgym_table_spawn_object_xy')
  spawn_xy = None
  if spawn_raw:
    spawn_xy = tuple(float(x) for x in str(spawn_raw).split(','))
    if len(spawn_xy) != 2:
      raise ValueError(
          'isaacgym_table_spawn_object_xy must be 2 floats, '
          f'got {spawn_raw!r}')

  print('[isaacgym_bootstrap] creating PhysX sim before JAX/TF '
        f'(env={env_name} E={num_envs} pipeline={pipeline} ep_len={ep_len} '
        f'randomize_init={randomize_init} '
        f'randomize_object_xyz={randomize_object_xyz} '
        f'randomize_object_shape={randomize_object_shape} '
        f'palm_goal={palm_goal} joint_goal={joint_goal} '
        f'control_sanity={control_sanity_mode or "off"} '
        f'trim_sa={control_sanity_trim_sa} '
        f'trim_init_mode={control_sanity_trim_init_mode} '
        f'trim_init_range_frac={control_sanity_trim_init_range_frac:g} '
        f'q_only={control_sanity_q_only} '
        f'goal_include_qd={control_sanity_goal_include_qd} '
        f'coordinate_mode={coordinate_mode} '
        f'table_push={table_push} '
        f'table_spawn={table_spawn}'
        f'{f" spawn_xy={spawn_xy}" if spawn_xy else ""}'
        f'{f" behind=dy{table_spawn_behind_dy:g},z{table_spawn_behind_above:g}" if table_spawn_behind else ""}'
        f'{f" correlated_xy=±{table_spawn_correlated_xy:g}" if table_spawn_correlated_xy > 0 else ""}'
        f' finger_curl_scale={table_spawn_finger_curl_scale:g}'
        f'{f" finger_noise={table_spawn_finger_noise:g}" if table_spawn_finger_noise > 0 else ""}'
        f'{f" arm_noise={table_spawn_arm_noise:g}" if table_spawn_arm_noise > 0 else ""}'
        f'{f" toward_bucket={table_spawn_toward_bucket}" if table_spawn_toward_bucket else ""}'
        f'{f" in_hand=off{table_spawn_in_hand_offset:g},n{table_spawn_in_hand_obj_noise:g}" if table_spawn_in_hand else ""}'
        f'{f" keep_arm wrist={table_spawn_in_hand_wrist_offset:g}" if table_spawn_in_hand_keep_arm else ""}'
        f'{f" lock_arm_base={lock_arm_base}" if lock_arm_base else ""}'
        f' large_table={large_table}'
        f' hide_table={hide_table}'
        f' palm_and_object_success={palm_and_object_success}'
        f' throw_success={throw_success}'
        f'{f" goal_z={goal_z:g}" if goal_z is not None else ""}'
        f'{f" table_push_xyz={push_xyz}" if push_xyz else ""}'
        f'{f" reset_z_above={reset_z_above:g}" if reset_z_above > 0 else ""})',
        flush=True)

  # isaacgym must precede torch; the env module enforces that.
  from envs.allegro_kuka_throw_env import AllegroKukaThrowVecEnv
  _PREBUILT = AllegroKukaThrowVecEnv(
      num_envs=int(num_envs),
      seed=int(seed * 31),
      episode_length=int(ep_len),
      fixed_target_xyz=xyz,
      pipeline=pipeline,
      headless=True,
      randomize_init=bool(randomize_init),
      randomize_object_xyz=bool(randomize_object_xyz),
      randomize_object_shape=bool(randomize_object_shape),
      palm_goal=bool(palm_goal),
      palm_goal_xyz=palm_xyz,
      joint_goal=bool(joint_goal),
      control_sanity_mode=control_sanity_mode,
      control_sanity_palm_xyz=sanity_palm_xyz,
      control_sanity_finger_tol=sanity_finger_tol,
      control_sanity_palm_tol=sanity_palm_tol,
      control_sanity_trim_sa=bool(control_sanity_trim_sa),
      control_sanity_trim_init_range_frac=(
          control_sanity_trim_init_range_frac),
      control_sanity_trim_init_mode=control_sanity_trim_init_mode,
      control_sanity_q_only=bool(control_sanity_q_only),
      control_sanity_goal_include_qd=bool(
          control_sanity_goal_include_qd),
      coordinate_mode=coordinate_mode,
      table_push=bool(table_push),
      table_push_xyz=push_xyz,
      table_spawn=bool(table_spawn),
      table_spawn_object_xy=spawn_xy,
      table_spawn_behind=bool(table_spawn_behind),
      table_spawn_behind_dy=float(table_spawn_behind_dy),
      table_spawn_behind_above=float(table_spawn_behind_above),
      table_spawn_correlated_xy=float(table_spawn_correlated_xy),
      table_spawn_finger_curl_scale=float(table_spawn_finger_curl_scale),
      table_spawn_finger_noise=float(table_spawn_finger_noise),
      table_spawn_arm_noise=float(table_spawn_arm_noise),
      table_spawn_toward_bucket=bool(table_spawn_toward_bucket),
      table_spawn_in_hand=bool(table_spawn_in_hand),
      table_spawn_in_hand_offset=float(table_spawn_in_hand_offset),
      table_spawn_in_hand_obj_noise=float(table_spawn_in_hand_obj_noise),
      table_spawn_in_hand_keep_arm=bool(table_spawn_in_hand_keep_arm),
      table_spawn_in_hand_wrist_offset=float(
          table_spawn_in_hand_wrist_offset),
      table_spawn_in_hand_wrist_noise=float(
          table_spawn_in_hand_wrist_noise),
      large_table=bool(large_table),
      hide_table=bool(hide_table),
      palm_and_object_success=bool(palm_and_object_success),
      throw_success=str(throw_success),
      goal_z=goal_z,
      lock_arm_base=bool(lock_arm_base),
      reset_z_above=float(reset_z_above),
  )
  print('[isaacgym_bootstrap] PhysX sim ready '
        f'(device={_PREBUILT.device}, E={_PREBUILT.num_envs})',
        flush=True)
  return _PREBUILT
