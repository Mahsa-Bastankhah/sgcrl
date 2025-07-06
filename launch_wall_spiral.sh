#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

###############################################################################
# Conda
###############################################################################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate contrastive_rl_nn
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

###############################################################################
# Experiment definitions
###############################################################################
wall_seeds=({6000..6007})

spiral_clusters=(
  "50000:5000 5001 5002 5003 5004 5005 5006 5007"      # A
  "100000:5100 5101 5102 5103 5104 5105 5106 5107"     # B
  "200000:5200 5201 5202 5203 5204 5205 5206 5207"     # C
)

spiral_hidden="--hidden_layer_sizes 256 --hidden_layer_sizes 256 \
--hidden_layer_sizes 256 --hidden_layer_sizes 256 --hidden_layer_sizes 256 \
--hidden_layer_sizes 256"

###############################################################################
# Robust GPU picker  (uses lockfiles + double-check)
###############################################################################
gpu_lock_dir="/tmp/gpu_locks"
mkdir -p "$gpu_lock_dir"

pick_gpu () {
  while true; do
    mapfile -t gpus < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    for line in "${gpus[@]}"; do
      read -r idx mem <<< "$line"
      lock="$gpu_lock_dir/lock_$idx"
      # try to grab lock non-blocking
      if ( set -o noclobber; echo $$ >"$lock" ) 2>/dev/null; then
        # we hold the lock; verify GPU really idle
        num_proc=$(nvidia-smi --query-compute-apps=pid --id="$idx" --format=csv,noheader | wc -l)
        if [[ $mem -lt 100 && $num_proc -eq 0 ]]; then
          echo "$idx"
          return
        fi
        # not idle → release lock and continue searching
        rm -f "$lock"
      fi
    done
    sleep 30    # nothing free yet
  done
}

# ensure locks are released on exit
cleanup() { rm -f "$gpu_lock_dir"/lock_* 2>/dev/null || true; }
trap cleanup EXIT

###############################################################################
launch () {
  local env=$1 seed=$2 gneg=$3 extra=$4
  local gpu=$(pick_gpu)
  local lock="$gpu_lock_dir/lock_$gpu"
  local log_dir="logs/${env}/neg${gneg}"
  mkdir -p "$log_dir"
  local log="${log_dir}/seed${seed}.out"

  echo "▶ ${env} seed=${seed} gneg=${gneg} gpu=${gpu} → ${log}"

  # run in subshell so we can release lock when process finishes
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
      nohup python lp_contrastive.py \
        --env "$env" \
        --seed "$seed" \
        --goal_neg_actor_steps "$gneg" \
        $extra \
        > "$log" 2>&1
    rm -f "$lock"
  ) &

  sleep 5   # slight stagger
}

###############################################################################
# 1) Wall
###############################################################################
for s in "${wall_seeds[@]}"; do
  launch point_Wall11x11 "$s" 50000 ""
done

###############################################################################
# 2) Spiral (3 clusters)
###############################################################################
for cluster in "${spiral_clusters[@]}"; do
  IFS=':' read -r gneg seeds_str <<< "$cluster"
  read -r -a seeds <<< "$seeds_str"
  for s in "${seeds[@]}"; do
    launch point_Spiral11x11 "$s" "$gneg" "$spiral_hidden"
  done
done

echo "✅  All jobs submitted.  Tail with:  tail -F logs/*/*.out"


##SBATCH --array=0-4              # job array with index values 0, 1, 2, 3, 4
#SBATCH --partition=mig          