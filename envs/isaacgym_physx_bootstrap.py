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


def _argv_flag(argv, name: str, default: Optional[str] = None) -> Optional[str]:
  eq = f'--{name}='
  opt = f'--{name}'
  for i, a in enumerate(argv):
    if a.startswith(eq):
      return a[len(eq):]
    if a == opt and i + 1 < len(argv) and not argv[i + 1].startswith('-'):
      return argv[i + 1]
  return default


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
  env_name = _argv_flag(argv, 'env')
  if not env_name or not str(env_name).startswith('allegro_kuka'):
    return None

  already = [m for m in ('jax', 'tensorflow', 'torch') if m in sys.modules]
  if already:
    print('[isaacgym_bootstrap] WARNING: already imported '
          f'{already} before PhysX create_sim; GPU kernels may fail',
          flush=True)

  raw_envs = _argv_flag(argv, 'ppo_num_envs')
  if raw_envs is None or int(raw_envs) < 0:
    num_envs = 8  # ContrastiveConfig.ppo_num_envs default
  else:
    num_envs = int(raw_envs)
  pipeline = str(_argv_flag(argv, 'isaacgym_pipeline', 'gpu') or 'gpu').strip().lower()
  ep_len = int(_argv_flag(argv, 'isaacgym_episode_length', '300') or '300')
  tgt_raw = _argv_flag(argv, 'isaacgym_fixed_target_xyz', '0.5,-0.3,0.4')
  xyz = tuple(float(x) for x in str(tgt_raw).split(','))
  if len(xyz) != 3:
    raise ValueError(f'isaacgym_fixed_target_xyz must be 3 floats, got {tgt_raw!r}')
  seed = int(_argv_flag(argv, 'seed', '0') or '0')

  print('[isaacgym_bootstrap] creating PhysX sim before JAX/TF '
        f'(env={env_name} E={num_envs} pipeline={pipeline} ep_len={ep_len})',
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
  )
  print('[isaacgym_bootstrap] PhysX sim ready '
        f'(device={_PREBUILT.device}, E={_PREBUILT.num_envs})',
        flush=True)
  return _PREBUILT
