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

Usage — must be the very first import in any entry-point script:
    import sgcrl_jax_acme_compat  # noqa: F401
"""
import types
import jax
import jax.numpy as jnp

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
