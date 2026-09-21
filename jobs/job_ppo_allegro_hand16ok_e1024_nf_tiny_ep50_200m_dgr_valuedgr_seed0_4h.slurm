#!/bin/bash
#SBATCH --job-name=akt_nf_hand16ok_tiny_vdgr_s0
#SBATCH --output=slurm/ppo_allegro_hand16ok_e1024_nf_tiny_ep50_200m_dgr_valuedgr_seed0_%j.log
#SBATCH --error=slurm/ppo_allegro_hand16ok_e1024_nf_tiny_ep50_200m_dgr_valuedgr_seed0_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=seas

# Same OK-hand valuedgr recipe as 3861945, but NF tiny (sa2x128 r32 b4 w128)
# instead of compact-small.
unset LD_PRELOAD
module purge
cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_isaacgym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR=/n/fs/mislresearch/mb6458-tmp
export MPLCONFIGDIR=/n/fs/mislresearch/mb6458-tmp/mpl-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

EP_LEN=50
GRAD_REG_COEF=5e-4
GRAD_REG_C=100.0
GRAD_REG_REL=0.1
GRAD_REG_LAM_LR=1e-6
LOG_DIR="logs/ppo_allegro_hand16ok_e1024_nf_tiny_fully_scaled_ep50_200m_dgr_valuedgr_seed0_4h/"
mkdir -p slurm "${LOG_DIR}"
echo "seed=0 E=1024 ep=${EP_LEN} steps=200000000"
echo "task=hand16ok object_free=true state=46 (all 23q+23qd) goal=16"
echo "figure: index+thumb loose OK circle, middle+ring open"
echo "coordinate_mode=fully_scaled  no trim_sa  frozen init  large_table"
echo "nf tiny sa2x128 r32 b4 w128 r(s,a) mix50 extrew1 ent=0.01 fixed minstd=1e-4"
echo "nf_grad_reg: dual, coef=${GRAD_REG_COEF} c=${GRAD_REG_C} rel=${GRAD_REG_REL} lam_lr=${GRAD_REG_LAM_LR}"
echo "value_grad_reg: dual, c=${GRAD_REG_C} lam_lr=${GRAD_REG_LAM_LR}"
echo "ppo_eval_interval=10"
echo "LOG_DIR=${LOG_DIR}"
nvidia-smi || true

python -u ppo_contrastive.py \
  --env=allegro_kuka_throw \
  --seed=0 \
  --num_steps=200000000 \
  --log_dir_path="${LOG_DIR}" \
  --ppo_num_envs=1024 \
  --isaacgym_episode_length="${EP_LEN}" \
  --ppo_rollout_length="${EP_LEN}" \
  --isaacgym_pipeline=gpu \
  --noisaacgym_randomize_init \
  --noisaacgym_randomize_object_shape \
  --isaacgym_control_sanity_mode=hand16ok \
  --isaacgym_coordinate_mode=fully_scaled \
  --isaacgym_large_table \
  --ppo_crl_steps_per_iter=10 \
  --ppo_ent_coef=0.01 \
  --ppo_anneal_ent_coef=false \
  --ppo_actor_min_std=1e-4 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_eval_interval=10 \
  --ppo_video_interval=100 \
  --ppo_checkpoint_interval=100 \
  --max_replay_size=10000000 \
  --ppo_checkpoint_replay_max=10000000 \
  --ppo_repr_mode=nf \
  --ppo_nf_reward_tau=0.5 \
  --ppo_use_external_reward \
  --ppo_external_reward_scale=1 \
  --nf_mix_task_goal_stats=true \
  --nf_mix_task_goal_frac=0.50 \
  --nf_rep_size=32 \
  --nf_num_blocks=4 \
  --nf_coupling_width=128 \
  --nf_sa_hidden=128 \
  --nf_sa_num_layers=2 \
  --nf_grad_clip=1.0 \
  --nf_noise_std=0.05 \
  --nf_goal_std_min=0.02 \
  --nf_goal_enc_size=0 \
  --ppo_nf_grad_reg \
  --ppo_crl_grad_reg_coef="${GRAD_REG_COEF}" \
  --ppo_crl_grad_reg_c="${GRAD_REG_C}" \
  --ppo_crl_grad_reg_rel="${GRAD_REG_REL}" \
  --ppo_nf_grad_reg_lam_lr="${GRAD_REG_LAM_LR}" \
  --ppo_value_grad_reg \
  --ppo_value_grad_reg_c="${GRAD_REG_C}" \
  --ppo_value_grad_reg_lam_lr="${GRAD_REG_LAM_LR}" \
  --hidden_layer_sizes="256,256,256,256,256,256"
PY_EXIT=$?
pkill -P $$ 2>/dev/null || true
exit $PY_EXIT
