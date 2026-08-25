# Source after: conda activate sgcrl_builderbench && module load cudatoolkit/12.6
# JAX 0.10 + jax-cuda12-plugin needs NVIDIA pip CUDA libs (cusparse, cublas, …)
# on LD_LIBRARY_PATH *before* the system CUDA toolkit, or JAX falls back to CPU.
# Default MJX backend is Warp (BUILDERBENCH_MJX_IMPL=warp). Export jax to override.

unset LD_PRELOAD
# Never inherit a submitter-forced CPU platform into GPU training jobs.
unset JAX_PLATFORMS
unset JAX_PLATFORM_NAME

_CONDA_ENV="${CONDA_PREFIX:-/n/fs/mislresearch/miniconda3/envs/sgcrl_builderbench}"
_NVIDIA_LIB_ROOT="${_CONDA_ENV}/lib/python3.11/site-packages/nvidia"
if [ -d "${_NVIDIA_LIB_ROOT}" ]; then
  _NVLIBS="$(
    find "${_NVIDIA_LIB_ROOT}" -name lib -type d 2>/dev/null | paste -sd: -
  )"
  if [ -n "${_NVLIBS}" ]; then
    export LD_LIBRARY_PATH="${_NVLIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export LD_LIBRARY_PATH="${_CONDA_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/nvidia"

export BUILDERBENCH_MJX_IMPL="${BUILDERBENCH_MJX_IMPL:-warp}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
