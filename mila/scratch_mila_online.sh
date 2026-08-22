#!/bin/bash
# ==============================================================================
# SGCRL PPO Contrastive Baseline + Online WandB Evaluation & In-Train Video Rendering
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ---------- USER CONFIGURATION ------------------------------------------------
SEEDS=( 0 1 )

LOG_ROOT="/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs"
# ------------------------------------------------------------------------------

BASE_FLAGS="--num_steps=200000000 --ppo_num_envs=1024 --ppo_ent_coef=0.05 --ppo_actor_min_std=0.01 --ppo_discount=0.99 --ppo_clip_coef=0.2 --ppo_checkpoint_interval=150 --builderbench_use_pd=true --builderbench_pd_duration=5 --ppo_skip_first_eval=true --ppo_eval_interval=150 --ppo_video_interval=150 --ppo_video_fps=10 --max_replay_size=10000000 --ppo_crl_repr_tau=0 --hidden_layer_sizes=\"256,256,256,256,256,256\" --env=builderbench_creative_3_task1 --ppo_rollout_length=50 --ppo_crl_steps_per_iter=25 --use_wandb=true --wandb_project=dist-matching --wandb_entity=doina-precup --wandb_mode=online --ppo_categorical_select --ppo_use_external_reward "

EXPERIMENTS=( 

    "pd_fm_creative4_task2_all_tricks_diff_ext_reward_scales_50_anneal_goal_noise|--env=builderbench_creative_4_task2 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --fm_goal_noise_std=0.01 --ppo_use_external_reward --ppo_external_reward_scale=50 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"

    # "pd_crl_creative3_task1_catselect_extrew1|--env=builderbench_creative_3_task1 --ppo_repr_mode=crl --ppo_crl_repr_tau=0.5 --ppo_external_reward_scale=1 --ppo_eval_interval=200"
    #
    #
    #
    ## FUTURE  (With best of ACTIVE)

    #
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_0_anneal|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=0 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"



    ### ACTIVE 


    # "pd_fm_creative4_task2_all_tricks_diff_ext_reward_scales_50_0|--env=builderbench_creative_4_task2 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=50 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"

    # "pd_fm_creative4_task1_all_tricks_diff_ext_reward_scales_50_0|--env=builderbench_creative_4_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=50 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"
    #
    # "pd_fm_creative4_task2_all_tricks_diff_ext_reward_scales_5_anneal|--env=builderbench_creative_4_task2 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=5 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"

    # "pd_fm_creative4_task1_all_tricks_diff_ext_reward_scales_5_anneal|--env=builderbench_creative_4_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=5 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"

    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_1_anneal|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=1 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"
    #
    #
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_5_anneal|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=5 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"

    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_100_0|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=100 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"

    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_50_0|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=50 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"
    #
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_50_0|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=50 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --num_steps=400_000_000 --ppo_ent_coef=0.0005  --ppo_ent_coef_final=0.0001 --ppo_anneal_ent_coef=True"
    #
    #
    #
    #
    #
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_100_0|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=100 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm"
    #
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_50_0|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=50 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm"
    #
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_1_smaller_net|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_external_reward_scale=1 --hidden_layer_sizes=\"256,256,256\""

    # "pd_fm_creative3_task1_all_tricks_longer_rollout|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=400000000 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_actor_min_std=0.01 --ppo_external_reward_scale=1"
    #
    # "pd_fm_creative3_task1_all_tricks_synergy_600m|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --fm_goal_noise_std=0.02 --fm_norm_goals=True --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_actor_min_std=0.01 --ppo_external_reward_scale=1 --num_steps=600000000"
    #
    # "pd_fm_creative3_task1_all_tricks_exact_cond_drop15_reward_clip|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --fm_norm_goals=True --fm_cond_dropout=0.15 --fm_reward_clip=20.0 --ppo_external_reward_scale=1 --num_steps=400000000 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_actor_min_std=0.01" 

    # "ext_ent_anneal_05_precision_online|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.01 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=600000000"
    # "ext_ent_anneal_05_precision_online_cat_select|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.01 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=600000000 --ppo_categorical_select"
    # "ext_ent_anneal_05_precision_online_cat_select_ext_reward|--env=builderbench_creative_4_task1 --ppo_actor_min_std=0.01 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_rollout_length=100 --ppo_crl_steps_per_iter=50 --num_steps=400000000 --ppo_categorical_select --ppo_external_reward_scale=1"
    # --- Active Flow Matching Experiments ---
    # Baseline
    # "pd_fm_creative3_task1_baseline|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select"
    # "pd_fm_creative3_task1_baseline_ext_reward_cat_select_anneal_ent|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1 --num_steps=400000000 --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_actor_min_std=0.01"
    # "pd_fm_creative3_task1_baseline_ext_reward_cat_select_no_permute_anneal_ent|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1 --num_steps=400000000 --builderbench_permute_start_boxes=false --builderbench_fixed_start_x=0.1 ---ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_actor_min_std=0.01"
    # "pd_fm_creative3_task1_all_tricks|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0"
    #
    
    # LOG p for cat acc computation.
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=1 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --fm_cat_acc_mode=logp --fm_cat_acc_subbatch=128"
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_0_25|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=0.25 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --fm_cat_acc_mode=logp --fm_cat_acc_subbatch=128"
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_0_5|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=0.5 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --fm_cat_acc_mode=logp --fm_cat_acc_subbatch=128"
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_1_5|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=1.5 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --fm_cat_acc_mode=logp --fm_cat_acc_subbatch=128"
    # "pd_fm_creative3_task1_all_tricks_diff_ext_reward_scales_2_0|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --ppo_use_external_reward --ppo_external_reward_scale=2 --ppo_eval_interval=200 --ppo_save_train_success_video=true --ppo_repr_mode=fm --fm_cat_acc_mode=logp --fm_cat_acc_subbatch=128"

    #
    # # Goal Normalization
    # "pd_fm_creative3_task1_goal_norm|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_norm_goals=True"
    #
    # # All Tricks + Goal Noise + Goal Norm
    # "pd_fm_creative3_task1_all_tricks_noise_norm|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --ppo_fm_reward_tau=0.5 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_t_sample_mode=logit_normal --fm_t_logit_loc=0.0 --fm_t_logit_scale=1.0 --fm_goal_noise_std=0.02 --fm_norm_goals=True"

    # --- TD-Flow (Bellman Probability Path) Experiments ---
    # Baseline TD-Flow (best guestimate defaults: gamma=0.99, target_tau=0.005, boot_steps=1)
    # "pd_fm_td_creative3_task1_baseline_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.99 --fm_td_target_tau=0.005 --fm_td_boot_steps=1 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1"

    # Baseline TD-Flow with Fixed Start Flags (--builderbench_permute_start_boxes=false --builderbench_fixed_start_x=0.1)
    # "pd_fm_td_creative3_task1_fixed_start_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.99 --fm_td_target_tau=0.005 --fm_td_boot_steps=1 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1 --builderbench_permute_start_boxes=false --builderbench_fixed_start_x=0.1"

    # Hyperparameter Variation 1: Discount factor gamma = 0.95 (shorter horizon)
    # "pd_fm_td_creative3_task1_gamma095_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.95 --fm_td_target_tau=0.005 --fm_td_boot_steps=1 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1"

    # Hyperparameter Variation 2: Target Polyak tau = 0.01 (faster target updates)
    # "pd_fm_td_creative3_task1_tau001_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.99 --fm_td_target_tau=0.01 --fm_td_boot_steps=1 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1"

    # Hyperparameter Variation 3: Target Polyak tau = 0.001 (slower, smoother target updates)
    # "pd_fm_td_creative3_task1_tau0001_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.99 --fm_td_target_tau=0.001 --fm_td_boot_steps=1 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1"

    # # High-precision TD-Flow (Fourier time embedding + Heun solver)
    # "pd_fm_td_creative3_task1_time_embed_heun_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.99 --fm_td_target_tau=0.005 --fm_td_boot_steps=1 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1 --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun"
    # "pd_fm_td_creative3_task1_gamma095_tricks_ext_scale1|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.95 --fm_td_target_tau=0.001 --fm_td_boot_steps=3 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1 --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_norm_goals=True"
    # "pd_fm_td_creative3_task1_gamma095_tricks_ext_scale1_anneal_ent|--env=builderbench_creative_3_task1 --ppo_repr_mode=fm --fm_td_mode=True --fm_td_gamma=0.95 --fm_td_target_tau=0.001 --fm_td_boot_steps=3 --fm_flow_steps=10 --fm_logp_mode=exact --ppo_categorical_select --ppo_external_reward_scale=1 --fm_time_embedding=True --fm_time_embed_dim=32 --fm_ode_solver=heun --fm_norm_goals=True --ppo_anneal_ent_coef=True --ppo_ent_coef=0.05 --ppo_ent_coef_final=0.01 --ppo_actor_min_std=0.01"
)

mkdir -p "$SCRIPT_DIR/slurm_logs"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    EXP_NAME="${EXPERIMENT%%|*}"
    EXP_FLAGS="${EXPERIMENT##*|}"

    if [[ "$EXP_FLAGS" =~ --env=([^ ]+) ]]; then
        ENV_ID="${BASH_REMATCH[1]}"
    else
        ENV_ID="builderbench_creative_3_task1" 
    fi
    echo "Running experiment: $EXP_NAME with env: $ENV_ID and flags: $EXP_FLAGS"

    for seed in "${SEEDS[@]}"; do

        # 1. Create a unique signature of the parameters and hash it
        SIG_STR="env=${ENV_ID}_seed=${seed}_base=${BASE_FLAGS}_exp=${EXP_FLAGS}"
        CMD_HASH=$(echo "$SIG_STR" | md5sum | cut -c1-8)
        SAFE_NAME="${EXP_NAME//+/_plus_}_${CMD_HASH}"

        # 2. Build the command using SAFE_NAME for the log_dir_path
        CMD="python -u ppo_contrastive.py \
            --seed=${seed} \
            --log_dir_path=${LOG_ROOT}/${SAFE_NAME} \
            --exp_name=${EXP_NAME} \
            ${BASE_FLAGS} \
            ${EXP_FLAGS}"

        echo "CMD: $CMD for SAFE_NAME: $SAFE_NAME"
        SLURM_SCRIPT="$SCRIPT_DIR/slurm_logs/${SAFE_NAME}_${ENV_ID}_s${seed}.slurm"

        # 3. Create Training Script (with Online Eval, Video Rendering, and WandB Logging)
        cat <<EOT > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${SAFE_NAME}_s${seed}_${ENV_ID}
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=23:00:00
#SBATCH --mem=256G
#SBATCH --output=%j.out

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench
export MUJOCO_GL=egl

export WANDB_DIR=\$SLURM_TMPDIR/wandb
export WANDB_CACHE_DIR=\$SLURM_TMPDIR/.cache/wandb
export WANDB_CONFIG_DIR=\$SLURM_TMPDIR/.config/wandb
export WANDB_DATA_DIR=\$SLURM_TMPDIR/.data/wandb
export CHECKPOINT_BASE_DIR=\$SCRATCH/jaxgcrl/checkpoints

if [ -d "${LOG_ROOT}/${SAFE_NAME}" ]; then
    echo "Warning: Directory exists, deleting to ensure clean restart."
    rm -rf "${LOG_ROOT}/${SAFE_NAME}"
fi

${CMD}
EOT

        echo "Submitting ${EXP_NAME} | env=${ENV_ID} | seed=${seed}..."
        TRAIN_OUTPUT=$(sbatch "$SLURM_SCRIPT")
        TRAIN_JOB_ID=$(echo "$TRAIN_OUTPUT" | awk '{print $4}')
        echo "  Training Job ID: $TRAIN_JOB_ID"
        echo "  Run dir is ${LOG_ROOT}/${SAFE_NAME}/ppo_${ENV_ID}_${seed}"

    done  # seeds
done  # experiments

echo ""
echo "All jobs submitted cleanly. Training, evaluation, video rendering, and WandB logging will execute live."
