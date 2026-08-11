"""Compatibility shim: patch deprecated JAX attributes for older acme.

Newer JAX (>= 0.4.x) removed several attributes that older acme still
references in ``acme/jax/utils.py``.  Importing this module *before* any
acme/jax import patches all known gaps in one place.

Removed attributes patched here (all found in acme/jax/utils.py):
  - jax.xla          (module)       → stub pointing to jax.Device / jax.Array
  - jax.xla.Device                  → jax.Device
  - jax.xla.DeviceArray             → jax.Array
  - jax.pxla         (module)       → stub; ShardedDeviceArray → jax.Array
  - jnp.DeviceArray                 → jax.Array

Also stubs out cv2 (OpenCV) if its native library cannot be loaded.  The flow
renderer (flow/renderer/pyglet_renderer.py) imports cv2 at the module level,
but it is never called during PPO training.  The cv2 binary in .venv requires
libavif-cbf1e83c.so.16.3.0, which may be missing on some cluster nodes; the
stub allows the renderer module to import without crashing.

Note: TF/Reverb/TFP require NumPy < 2.  The .venv pins numpy==1.26.4 for
this reason (numpy 2.x breaks TF 2.8 C extensions with _ARRAY_API not found).

Usage — must be the very first import in any entry-point script:
    import sgcrl_jax_acme_compat  # noqa: F401
"""
import ctypes
import importlib
import pathlib
import sys
import types


def _preload_nvidia_cuda_libs() -> None:
  """Preload pip NVIDIA CUDA libs before JAX initializes the CUDA plugin.

  JAX 0.10 + jax-cuda12-plugin calls cusparseGetProperty during plugin init.
  If libcusparse from the pip ``nvidia-*`` packages is not already loaded with
  global scope, JAX fails version checks and falls back to CPU even when a GPU
  is allocated by Slurm.
  """
  _mods = (
      'cuda_runtime', 'nvjitlink', 'cublas', 'cusparse', 'cusolver',
      'cufft', 'cudnn', 'cuda_nvrtc', 'nccl',
  )
  for name in _mods:
    try:
      mod = importlib.import_module(f'nvidia.{name}')
    except ImportError:
      continue
    lib_dir = pathlib.Path(mod.__path__[0]) / 'lib'
    if not lib_dir.is_dir():
      continue
    for so in sorted(lib_dir.glob('*.so*')):
      if so.is_symlink() and not so.name.endswith('.so'):
        continue
      try:
        ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
      except OSError:
        pass


_preload_nvidia_cuda_libs()

import jax
import jax.numpy as jnp
import jax.tree_util as _jax_tree_util

# --- jax.tree_* (removed from top-level jax in >= 0.4) ---
if not hasattr(jax, 'tree_map'):
    jax.tree_map = _jax_tree_util.tree_map
if not hasattr(jax, 'tree_flatten'):
    jax.tree_flatten = _jax_tree_util.tree_flatten
if not hasattr(jax, 'tree_multimap'):
    jax.tree_multimap = _jax_tree_util.tree_map

# ---------------------------------------------------------------------------
# launchpad stub — dm-launchpad is not pip-installable on Python 3.11, but acme
# imports it in acme/utils/signals.py.  PPO does not use launchpad workers;
# register_stop_handler / unregister_stop_handler are no-ops.
#
# Prefer the real package when present (e.g. sgcrl_flow + lp_contrastive).
# Only stub if import fails — otherwise we shadow dm-launchpad and break
# --lp_launch_type / Launchpad program startup.
# ---------------------------------------------------------------------------
if 'launchpad' not in sys.modules:
    try:
        import launchpad as _real_launchpad  # noqa: F401
    except ImportError:
        _lp = types.ModuleType('launchpad')
        _lp._stop_handlers = []

        def register_stop_handler(handler):
            _lp._stop_handlers.append(handler)

        def unregister_stop_handler(handler):
            try:
                _lp._stop_handlers.remove(handler)
            except ValueError:
                pass

        _lp.register_stop_handler = register_stop_handler
        _lp.unregister_stop_handler = unregister_stop_handler
        sys.modules['launchpad'] = _lp
        del register_stop_handler, unregister_stop_handler, _lp

# ---------------------------------------------------------------------------
# cv2 stub — only installed when the real cv2 cannot be loaded.
# Flow's renderer imports cv2 at module level; PPO never renders, so a
# no-op stub is sufficient.
# ---------------------------------------------------------------------------
if 'cv2' not in sys.modules:
    try:
        import cv2  # noqa: F401
    except (ImportError, OSError):
        class _AutoAttrModule(types.ModuleType):
            """Module stub that returns 0 for any missing attribute (cv2 constants)."""
            def __getattr__(self, name):
                if name in ('__file__', '__spec__', '__path__'):
                    return super().__getattribute__(name)
                return 0
        _cv2_stub = _AutoAttrModule('cv2')
        _cv2_stub.__file__ = __file__
        _cv2_stub.__version__ = '0.0.0-stub'
        sys.modules['cv2'] = _cv2_stub
        del _AutoAttrModule, _cv2_stub

_Array = getattr(jax, 'Array', object)
_Device = getattr(jax, 'Device', object)

# --- jax.xla (module) ---
if not hasattr(jax, 'xla'):
    _xla = types.ModuleType('jax.xla')
    _xla.Device = _Device
    _xla.DeviceArray = _Array
    jax.xla = _xla
else:
    # Module exists but individual attrs may still be missing.
    if not hasattr(jax.xla, 'Device'):
        jax.xla.Device = _Device
    if not hasattr(jax.xla, 'DeviceArray'):
        jax.xla.DeviceArray = _Array

# --- jax.pxla (module) ---
if not hasattr(jax, 'pxla'):
    _pxla = types.ModuleType('jax.pxla')
    _pxla.ShardedDeviceArray = _Array
    jax.pxla = _pxla
else:
    if not hasattr(jax.pxla, 'ShardedDeviceArray'):
        jax.pxla.ShardedDeviceArray = _Array

# --- jnp.DeviceArray ---
if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = _Array

# --- jax.interpreters.xla.pytype_aval_mappings ---
# TFP 0.24–0.25 still writes to jax.interpreters.xla.pytype_aval_mappings, but
# JAX >= 0.4 moved it to jax.core.pytype_aval_mappings (deprecated alias).
import jax.core as _jax_core
import jax.interpreters.xla as _jax_xla
if not hasattr(_jax_xla, 'pytype_aval_mappings'):
    _jax_xla.pytype_aval_mappings = _jax_core.pytype_aval_mappings

# ---------------------------------------------------------------------------
# TFP / TensorFlow — acme.jax.networks.distributional does
#   `tfp = tensorflow_probability.substrates.jax` at import time.
# TFP 0.25 only attaches `.substrates` after the jax submodule is imported,
# and its jax backend expects TF + tf_keras to be present.
#
# tf_keras is only a separate pip package (needed for Keras-3 TF releases,
# e.g. the sgcrl_builderbench env's TF 2.21). Older envs (e.g. sgcrl_flow's
# TF 2.8) ship Keras built into `tf.keras` and never had tf_keras installed,
# nor do they need it — so this import is optional.
# ---------------------------------------------------------------------------
import tensorflow as _tf  # noqa: F401
try:
    import tf_keras as _tf_keras  # noqa: F401
except ImportError:
    pass
import tensorflow_probability.substrates.jax as _tfp_jax  # noqa: F401
