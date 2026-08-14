#!/bin/bash
# In-job 10-ckpt reward/grad probe (same Slurm allocation as training).
# Uses CPU JAX so it does not fight the trainer GPU.
#
# Required env:
#   PROBE_CKPT_DIR  PROBE_OUT_DIR  PROBE_TAG_PREFIX  PROBE_ENV
#   PROBE_MODE=nf|crl
# Optional:
#   PROBE_ITERS  PROBE_TRAIN_PID  PROBE_SEED (default 0)

PROBE_ITERS="${PROBE_ITERS:-300 900 1200 1500 1800 2400 2700 3000 3600 3900}"
PROBE_SEED="${PROBE_SEED:-0}"

bb_probe_render_one() {
  local iter="$1"
  local ckpt
  ckpt=$(printf '%s/ckpt_iter_%07d.pkl' "${PROBE_CKPT_DIR}" "${iter}")
  if [[ ! -f "${ckpt}" ]]; then
    return 0
  fi
  mkdir -p "${PROBE_OUT_DIR}"
  echo "[probe] rendering iter=${iter}  ckpt=${ckpt}"
  if [[ "${PROBE_MODE}" == "crl" ]]; then
    JAX_PLATFORMS=cpu python -u scripts/render_frozen_crl_traj_reward_video.py \
      --checkpoint="${ckpt}" \
      --env="${PROBE_ENV}" \
      --out_dir="${PROBE_OUT_DIR}" \
      --tag_prefix="${PROBE_TAG_PREFIX}" \
      --allow_no_success \
      --max_tries=1 \
      --seed="${PROBE_SEED}" \
      --fps=8 \
      --normalize_reward \
      --show_phi_psi_grad \
      --skip_existing
  else
    JAX_PLATFORMS=cpu python -u scripts/render_nf_traj_reward_video.py \
      --checkpoint="${ckpt}" \
      --env="${PROBE_ENV}" \
      --out_dir="${PROBE_OUT_DIR}" \
      --tag_prefix="${PROBE_TAG_PREFIX}" \
      --allow_no_success \
      --max_tries=1 \
      --seed="${PROBE_SEED}" \
      --fps=8 \
      --normalize_reward \
      --show_logp_grad \
      --skip_existing
  fi
}

bb_probe_sidecar() {
  mkdir -p "${PROBE_OUT_DIR}"
  echo "[probe] sidecar start mode=${PROBE_MODE} ckpt_dir=${PROBE_CKPT_DIR}"
  echo "[probe] out_dir=${PROBE_OUT_DIR} targets=${PROBE_ITERS}"
  local iter ckpt
  for iter in ${PROBE_ITERS}; do
    ckpt=$(printf '%s/ckpt_iter_%07d.pkl' "${PROBE_CKPT_DIR}" "${iter}")
    echo "[probe] waiting for iter=${iter}"
    while [[ ! -f "${ckpt}" ]]; do
      if [[ -n "${PROBE_TRAIN_PID:-}" ]] && ! kill -0 "${PROBE_TRAIN_PID}" 2>/dev/null; then
        echo "[probe] trainer exited before iter=${iter}"
        break
      fi
      sleep 60
    done
    if [[ -f "${ckpt}" ]]; then
      bb_probe_render_one "${iter}" || echo "[probe] render failed iter=${iter}"
    fi
  done
  echo "[probe] sidecar done"
  ls -lh "${PROBE_OUT_DIR}"/*.mp4 2>/dev/null | tail -20 || true
}

bb_probe_sweep() {
  local iter
  for iter in ${PROBE_ITERS}; do
    bb_probe_render_one "${iter}" || true
  done
}
