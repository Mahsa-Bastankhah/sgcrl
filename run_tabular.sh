#!/bin/bash

# Activate your environment
module purge
module load anaconda3/2023.3
source /home/mb6458/miniconda3/bin/activate contrastive_rl
mkdir -p logs

# ---------------- Hyperparameter grids ----------------
REP_DIMS=(64)
EPISODES_PER_UPD=(5)
LR_PHI_PSI=(0.05)
## These three parameyers are not important if you use the default initi_mode which is "uniform_around_goal".
NEAR_VARS=(0.2)
FAR_VARS=(5.0)
ALPHAS=(100)


REPLAY_CAPS=(1000)
MAX_STEPS_LIST=(100)
BATCH_SIZES=(128)
NUM_EPISODES_LIST=(10000)
GAMMAS=(0.99)
SEEDS=(501)
entropy_coeff=(0.1)

# Trajectory cutting parameters
TRAJ_CUT_START_FRACS=(0)
TRAJ_CUT_END_FRACS=(0)
TRAJ_CUT_RATES=(0)

# Psi similarity filtering parameters
PSI_SIM_THRESHOLDS=(0.98)
ENABLE_PSI_FILTERING=(True)
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
                        for TCSF in "${TRAJ_CUT_START_FRACS[@]}"; do
                          for TCEF in "${TRAJ_CUT_END_FRACS[@]}"; do
                            for TCR in "${TRAJ_CUT_RATES[@]}"; do
                              for PST in "${PSI_SIM_THRESHOLDS[@]}"; do
                                for EPF in "${ENABLE_PSI_FILTERING[@]}"; do
                                  for SEED in "${SEEDS[@]}"; do

                                echo "------------------------------------------------------------"
                                echo "rep=$REP upd=$UPD lr=$LR near=$NVAR far=$FVAR alpha=$ALPHA"
                                echo "replay_cap=$RCAP max_steps=$MSTEP batch_size=$BS num_episodes=$NEPI gamma=$GAM entropy_coeff=$ECO"
                                echo "traj_cut_start=$TCSF traj_cut_end=$TCEF traj_cut_rate=$TCR"
                                echo "psi_sim_thresh=$PST enable_psi_filter=$EPF seed=$SEED"
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
                            --env "spiral" \
                            --verbose True \
                            --continuous-episode True \

                            > "tabular_${REP}_${UPD}_${LR}_${NVAR}_${FVAR}_${ALPHA}_${RCAP}_${MSTEP}_${BS}_${NEPI}_${GAM}_${ECO}_${TCSF}_${TCEF}_${TCR}_${PST}_${EPF}_${SEED}.log" 2>&1 &

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
        done
      done
    done
  done
done

echo "All jobs submitted to background. Check logs directory for output."
echo "Use 'ps aux | grep tabular_SGCRL.py' to see running processes."
exit 0