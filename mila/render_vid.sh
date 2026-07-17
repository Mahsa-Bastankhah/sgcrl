#!/bin/bash
# ==============================================================================
# Interactive video rendering for BuilderBench PPO-CRL checkpoints.
#
# Mirrors the EXPERIMENTS / SEEDS / BASE_FLAGS / LOG_ROOT definitions in
# mila/scratch_mila.sh so it can recompute the same SAFE_NAME hash and find
# already-trained run directories on disk. For every matching run, renders a
# video for EVERY checkpoint found in that run's checkpoints/ directory
# (ckpt_iter_*.pkl + latest.pkl, via rollout_video.py's own directory
# handling), writing to <run_dir>/videos/.
#
# IMPORTANT: Keep the "KEEP IN SYNC" block below identical to the
# corresponding values in mila/scratch_mila.sh. SAFE_NAME is an md5 hash of
# these exact strings, so any drift there will make this script look in the
# wrong (or a nonexistent) run directory.
#
# Usage:
#   ./mila/render_videos.sh
#   ./mila/render_videos.sh --exp_name=reproduce
#   ./mila/render_videos.sh --exp_name=reproduce,reproduce_longer_rollout \
#                            --env=builderbench_creative_4_task1 --seed=0
#   ./mila/render_videos.sh --seed=1 --fps=15 --dry_run
#
# Run this from an interactive session that already has the
# sgcrl_builderbench conda env activated and (if needed) BUILDERBENCH_ROOT
# set, exactly like the training/eval SLURM jobs do.
# ==============================================================================

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# ---------- KEEP IN SYNC WITH mila/scratch_mila.sh ---------------------------
SEEDS=( 0 1 )

LOG_ROOT="/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"

BASE_FLAGS="--num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=0 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes=\"256,256,256,256,256,256\" --env=builderbench_creative_3_task1"

EXPERIMENTS=(
    "reproduce|--env=builderbench_creative_4_task1"
    "reproduce|--env=builderbench_creative_4_task6 --obs_space="xy,quaternions,select""
    "reproduce|--env=builderbench_creative_3_task2 -obs_space="xy,quaternions,select""
    # "reproduce2|"
    "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --ppo_rollout_length=128" # Roughly double which is automatically calculated to 60.
    "reproduce_longer_rollout|--env=builderbench_creative_4_task1 --ppo_rollout_length=256" # Roughly quadruple which is automatically calculated to 60.
)
# ------------------------------------------------------------------------------

# ---------- CLI filters -------------------------------------------------------
FILTER_EXP_NAME=""
FILTER_ENV=""
FILTER_SEED=""
FPS=10
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --exp_name=*) FILTER_EXP_NAME="${arg#*=}" ;;
    --env=*)      FILTER_ENV="${arg#*=}" ;;
    --seed=*)     FILTER_SEED="${arg#*=}" ;;
    --fps=*)      FPS="${arg#*=}" ;;
    --dry_run)    DRY_RUN=true ;;
    -h|--help)
      echo "Usage: $0 [--exp_name=a,b] [--env=x,y] [--seed=0,1] [--fps=10] [--dry_run]"
      exit 0
      ;;
    *)
      echo "[render_videos] Unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

# _list_contains <comma_separated_list_or_empty> <value>
# Empty list = wildcard (always matches).
_list_contains() {
  local list="$1" value="$2"
  [[ -z "$list" ]] && return 0
  local IFS=','
  local items=($list)
  local item
  for item in "${items[@]}"; do
    [[ "$item" == "$value" ]] && return 0
  done
  return 1
}

echo "[render_videos] REPO_DIR=$REPO_DIR"
echo "[render_videos] LOG_ROOT=$LOG_ROOT"
echo "[render_videos] filters: exp_name='${FILTER_EXP_NAME:-*}' env='${FILTER_ENV:-*}' seed='${FILTER_SEED:-*}'"
echo "[render_videos] fps=$FPS dry_run=$DRY_RUN"
echo ""

n_matched=0
n_rendered=0
n_skipped=0
n_failed=0

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    EXP_NAME="${EXPERIMENT%%|*}"
    EXP_FLAGS="${EXPERIMENT##*|}"

    if [[ "$EXP_FLAGS" =~ --env=([^[:space:]]+) ]]; then
        ENV_ID="${BASH_REMATCH[1]}"
    else
        ENV_ID="builderbench_creative_3_task1"
    fi

    _list_contains "$FILTER_EXP_NAME" "$EXP_NAME" || continue
    _list_contains "$FILTER_ENV" "$ENV_ID" || continue

    for seed in "${SEEDS[@]}"; do
        _list_contains "$FILTER_SEED" "$seed" || continue

        # Must match the exact SAFE_NAME construction in scratch_mila.sh.
        SIG_STR="env=${ENV_ID}_seed=${seed}_base=${BASE_FLAGS}_exp=${EXP_FLAGS}"
        CMD_HASH=$(echo "$SIG_STR" | md5sum | cut -c1-8)
        SAFE_NAME="${EXP_NAME//+/_plus_}_${CMD_HASH}"

        RUN_DIR="${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}"
        CKPT_DIR="${RUN_DIR}/checkpoints"
        VIDEO_DIR="${RUN_DIR}/videos"
        RUN_TAG="${SAFE_NAME}_${ENV_ID}_s${seed}"

        n_matched=$((n_matched + 1))

        if [[ ! -d "$CKPT_DIR" ]]; then
            echo "[skip] no checkpoints/ dir: $CKPT_DIR"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        n_ckpts=$(find "$CKPT_DIR" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l)
        if [[ "$n_ckpts" -eq 0 ]]; then
            echo "[skip] checkpoints/ dir is empty: $CKPT_DIR"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        echo "[render] exp=$EXP_NAME env=$ENV_ID seed=$seed"
        echo "         run_dir=$RUN_DIR  ($n_ckpts checkpoint file(s))"
        echo "         -> $VIDEO_DIR/"

        if [[ "$DRY_RUN" == true ]]; then
            n_rendered=$((n_rendered + 1))
            echo ""
            continue
        fi

        mkdir -p "$VIDEO_DIR"
        (
          cd "$REPO_DIR" && \
          python scripts/ppo_builderbench_rollout_video.py \
              --checkpoint="${CKPT_DIR}" \
              --env="${ENV_ID}" \
              --output="${VIDEO_DIR}/" \
              --fps="${FPS}" \
              --run_tag="${RUN_TAG}"
        )
        status=$?
        if [[ $status -ne 0 ]]; then
            echo "[error] rollout_video.py failed for $RUN_DIR (exit $status)" >&2
            n_failed=$((n_failed + 1))
        else
            n_rendered=$((n_rendered + 1))
        fi
        echo ""
    done
done

echo "======================================================================"
echo "[render_videos] matched=$n_matched  rendered=$n_rendered  skipped=$n_skipped  failed=$n_failed"

if [[ "$n_failed" -gt 0 ]]; then
  exit 1
fi
