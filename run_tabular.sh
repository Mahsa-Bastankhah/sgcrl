#!/bin/bash

# Activate your environment
module purge
module load anaconda3/2023.3
source /home/mb6458/miniconda3/bin/activate contrastive_rl
mkdir -p logs

# ---------------- Hyperparameter grids ----------------
REP_DIMS=(16)
EPISODES_PER_UPD=(5)
LR_PHI_PSI=(1e-3)
NEAR_VARS=(0.2)
FAR_VARS=(5.0)
ALPHAS=(100)
REPLAY_CAPS=(1000)
MAX_STEPS_LIST=(100)
BATCH_SIZES=(128)
NUM_EPISODES_LIST=(50001)
GAMMAS=(0.99)
SEEDS=(205)
entropy_coeff=(0.1)
# -------------------------------------------------------

for REP in "${REP_DIMS[@]}"; do
  for UPD in "${EPISODES_PER_UPD[@]}"; do
    for LR in "${LR_PHI_PSI[@]}"; do
      for NVAR in "${NEAR_VARS[@]}"; do
        for FVAR in "${FAR_VARS[@]}"; do
          for ALPHA in "${ALPHAS[@]}"; do
            for RCAP in "${REPLAY_CAPS[@]}"; do
              for MSTEP in "${MAX_STEPS_LIST[@]}"; do
                for BS in "${BATCH_SIZES[@]}"; do
                  for NEPI in "${NUM_EPISODES_LIST[@]}"; do
                    for GAM in "${GAMMAS[@]}"; do
                      for ECO in "${entropy_coeff[@]}"; do
                        for SEED in "${SEEDS[@]}"; do

                          echo "------------------------------------------------------------"
                          echo "rep=$REP upd=$UPD lr=$LR near=$NVAR far=$FVAR alpha=$ALPHA"
                          echo "replay_cap=$RCAP max_steps=$MSTEP batch_size=$BS num_episodes=$NEPI gamma=$GAM entropy_coeff=$ECO seed=$SEED"
                          echo "------------------------------------------------------------"

                          nohup python -u tabular_SGCRL.py \
                            --rep-dim "$REP" \
                            --episodes-per-upd "$UPD" \
                            --lr-phi-psi "$LR" \
                            --near-var "$NVAR" \
                            --far-var "$FVAR" \
                            --alpha "$ALPHA" \
                            --replay-capacity "$RCAP" \
                            --max-steps "$MSTEP" \
                            --batch-size "$BS" \
                            --num-episodes "$NEPI" \
                            --gamma "$GAM" \
                            --seed "$SEED" \
                            --entropy_coeff "$ECO" \
                            --loss_mode "backward" \
                            --env "fourRooms10" \
                            --verbose True \
                            --random_exploration True \
                            --optimality_exp_mode "sgcrl" \

                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
