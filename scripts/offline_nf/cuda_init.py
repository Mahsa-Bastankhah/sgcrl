"""Torch-first CUDA initialization required by this cluster's JAX build."""
from __future__ import annotations

import os
import sys


def initialize_torch_first() -> None:
  import torch
  import torch.nn.functional as torch_f

  if not torch.cuda.is_available():
    raise RuntimeError("offline NF training requires a CUDA GPU")
  device = torch.device("cuda:0")
  x = torch.ones((1, 8, 16, 16), dtype=torch.float32, device=device)
  kernel = torch.ones((16, 8, 3, 3), dtype=torch.float32, device=device)
  checksum = torch_f.conv2d(x, kernel, padding=1).sum()
  torch.cuda.synchronize(device)
  cudnn_version = torch.backends.cudnn.version()
  print(
      "[cuda-init] "
      f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
      f"cudnn={cudnn_version} device={torch.cuda.get_device_name(device)} "
      f"conv_checksum={float(checksum):.1f}", flush=True)
  try:
    with open("/proc/self/maps", encoding="utf-8") as handle:
      paths = sorted({
          line.split()[-1] for line in handle
          if "libcudnn" in line and "/" in line
      })
    for path in paths:
      print(f"[cuda-init] mapped_cudnn={path}", flush=True)
  except OSError:
    pass
  if cudnn_version is None or int(cudnn_version) // 1000 != 8:
    raise RuntimeError(
        "This JAX build requires cuDNN major 8, but Torch initialized "
        f"cuDNN version {cudnn_version!r}")
  del x, kernel, checksum
  torch.cuda.empty_cache()

  repo = os.path.dirname(os.path.dirname(os.path.dirname(
      os.path.abspath(__file__))))
  if repo not in sys.path:
    sys.path.insert(0, repo)
  import sgcrl_jax_acme_compat  # noqa: F401
  import jax
  import jax.numpy as jnp
  import jaxlib

  gpu_devices = jax.devices("gpu")
  if not gpu_devices:
    raise RuntimeError(f"JAX found no GPU devices; all devices={jax.devices()}")
  ready = jnp.sum(
      jnp.ones((64, 64), dtype=jnp.float32)
      @ jnp.ones((64, 64), dtype=jnp.float32)).block_until_ready()
  print(
      "[jax-ready] "
      f"python={sys.version.split()[0]} jax={jax.__version__} "
      f"jaxlib={jaxlib.__version__} backend={jax.default_backend()} "
      f"device={gpu_devices[0]} matmul_checksum={float(ready):.1f} "
      f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}", flush=True)
