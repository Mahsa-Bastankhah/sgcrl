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
  table_push = _argv_bool(argv, 'isaacgym_table_push', False)
  table_spawn = _argv_bool(argv, 'isaacgym_table_spawn', False)
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

  print('[isaacgym_bootstrap] creating PhysX sim before JAX/TF '
        f'(env={env_name} E={num_envs} pipeline={pipeline} ep_len={ep_len} '
        f'randomize_init={randomize_init} '
        f'randomize_object_xyz={randomize_object_xyz} '
        f'randomize_object_shape={randomize_object_shape} '
        f'palm_goal={palm_goal} table_push={table_push} '
        f'table_spawn={table_spawn}'
        f'{f" table_push_xyz={push_xyz}" if push_xyz else ""})',
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
      table_push=bool(table_push),
      table_push_xyz=push_xyz,
      table_spawn=bool(table_spawn),
  )
  print('[isaacgym_bootstrap] PhysX sim ready '
        f'(device={_PREBUILT.device}, E={_PREBUILT.num_envs})',
        flush=True)
  return _PREBUILT
