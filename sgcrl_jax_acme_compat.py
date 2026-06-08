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
import sys
import types
import jax
import jax.numpy as jnp

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
                return 0
        _cv2_stub = _AutoAttrModule('cv2')
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
