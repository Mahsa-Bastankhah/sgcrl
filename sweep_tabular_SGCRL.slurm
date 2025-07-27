#!/bin/bash
#SBATCH --job-name=maze_ms
#SBATCH --array=0-3%4             # (NUM_CONFIGS * NUM_SEEDS) - 1
#SBATCH --time=01:30:00
#SBATCH --account=eysenbach
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=mb6458@princeton.edu


module purge
module load anaconda3/2023.3
source /home/mb6458/miniconda3/bin/activate contrastive_rl
mkdir -p logs

# ---------------- Hyperparameter grids (EDIT BELOW) ----------------
REP_DIMS=(16)
## 50 is too high
## seems 3 is almost the same as 5
EPISODES_PER_UPD=(5)
## any lr less than 1e-3 is too low
## it seems increasing the lr to more than 1e-3 doesn't help much, 5e-3 is almost similar to 1e-3
LR_PHI_PSI=(1e-3)
NEAR_VARS=(0.2)
FAR_VARS=(5.0)  
## 3 is the bnest for alpha        # keep single if fixed
ALPHAS=(100)
## should I try larger? probably too small right now?
## it seems increasing the replay buffer size never helps, 10k is probably too high, keeping the last 100 episodes in the buffer
REPLAY_CAPS=(1000)      # new
MAX_STEPS_LIST=(100)    # new
## I think 64 is too small, 128 is better
BATCH_SIZES=(128)       # new
NUM_EPISODES_LIST=(100000)  # new
GAMMAS=(0.99)           # new
SEEDS=(112 113 114)
entropy_coeff=(0.1)
# -------------------------------------------------------------------

# ----- Lengths
L_REP=${#REP_DIMS[@]}
L_UPD=${#EPISODES_PER_UPD[@]}
L_LR=${#LR_PHI_PSI[@]}
L_NVAR=${#NEAR_VARS[@]}
L_FVAR=${#FAR_VARS[@]}
L_ALPHA=${#ALPHAS[@]}
L_RCAP=${#REPLAY_CAPS[@]}
L_MSTEP=${#MAX_STEPS_LIST[@]}
L_BS=${#BATCH_SIZES[@]}
L_NEPI=${#NUM_EPISODES_LIST[@]}
L_GAM=${#GAMMAS[@]}
L_SEED=${#SEEDS[@]}
L_ECO=${#entropy_coeff[@]}

NUM_CONFIGS=$(( L_REP * L_UPD * L_LR * L_NVAR * L_FVAR * L_ALPHA * L_RCAP * L_MSTEP * L_BS * L_NEPI * L_GAM * L_ECO))
TOTAL=$(( NUM_CONFIGS * L_SEED ))

# Safety vs header
if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
  echo "Index $SLURM_ARRAY_TASK_ID out of range (TOTAL=$TOTAL). Adjust --array."
  exit 1
fi

CONFIG_INDEX=$(( SLURM_ARRAY_TASK_ID / L_SEED ))
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID % L_SEED ))

# Factor CONFIG_INDEX across all config axes (order must match multiplication order above)
tmp=$CONFIG_INDEX

rep_i=$(( tmp % L_REP ));    tmp=$(( tmp / L_REP ))
# eps_i=$(( tmp % L_EPS ));    tmp=$(( tmp / L_EPS ))
upd_i=$(( tmp % L_UPD ));    tmp=$(( tmp / L_UPD ))
lr_i=$(( tmp % L_LR ));      tmp=$(( tmp / L_LR ))
nvar_i=$(( tmp % L_NVAR ));  tmp=$(( tmp / L_NVAR ))
fvar_i=$(( tmp % L_FVAR ));  tmp=$(( tmp / L_FVAR ))
alpha_i=$(( tmp % L_ALPHA ));tmp=$(( tmp / L_ALPHA ))
rcap_i=$(( tmp % L_RCAP ));  tmp=$(( tmp / L_RCAP ))
mstep_i=$(( tmp % L_MSTEP ));tmp=$(( tmp / L_MSTEP ))
bs_i=$(( tmp % L_BS ));      tmp=$(( tmp / L_BS ))
nepi_i=$(( tmp % L_NEPI ));  tmp=$(( tmp / L_NEPI ))
gam_i=$(( tmp % L_GAM ));    tmp=$(( tmp / L_GAM ))
eco_i=$(( tmp % L_ECO ));  tmp=$(( tmp / L_ECO ))

REP=${REP_DIMS[$rep_i]}
# EPS=${EPSILONS[$eps_i]}
UPD=${EPISODES_PER_UPD[$upd_i]}
LR=${LR_PHI_PSI[$lr_i]}
NVAR=${NEAR_VARS[$nvar_i]}
FVAR=${FAR_VARS[$fvar_i]}
ALPHA=${ALPHAS[$alpha_i]}
RCAP=${REPLAY_CAPS[$rcap_i]}
MSTEP=${MAX_STEPS_LIST[$mstep_i]}
BS=${BATCH_SIZES[$bs_i]}
NEPI=${NUM_EPISODES_LIST[$nepi_i]}
GAM=${GAMMAS[$gam_i]}
SEED=${SEEDS[$SEED_INDEX]}
ECO=${entropy_coeff[$eco_i]}

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  echo "[INFO] NUM_CONFIGS=$NUM_CONFIGS L_SEED=$L_SEED TOTAL=$TOTAL"
fi

echo "------------------------------------------------------------"
echo "Global $SLURM_ARRAY_TASK_ID / $TOTAL   ConfigIndex=$CONFIG_INDEX  Seed=$SEED"
echo "rep=$REP eps=$EPS upd=$UPD lr=$LR near=$NVAR far=$FVAR alpha=$ALPHA"
echo "replay_cap=$RCAP max_steps=$MSTEP batch_size=$BS num_episodes=$NEPI gamma=$GAM, entropy_coeff=$ECO"
echo "------------------------------------------------------------"

python -u tabular_SGCRL.py \
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
  --random_exploration False \
  --two_goals True \