

```
module unload python; module load anaconda/3
conda create -n sgcrl_builderbench python=3.11 -y
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install https://github.com/google-deepmind/acme/archive/refs/tags/0.4.0.tar.gz --no-deps
```


On node `cn-c021` on the Mila cluster.


```
python -u ppo_contrastive.py \
  --env=builderbench_creative_3_task1\
  --seed=0 \
  --num_steps=200000000 \
  --log_dir_path=/network/scratch/m/mohammad-sami-nur.islam/dist_matching/logs/debug \
  --ppo_num_envs=1024 \
  --ppo_ent_coef=0.05 \
  --ppo_actor_min_std=0.01 \
  --ppo_discount=0.99 \
  --ppo_clip_coef=0.2 \
  --ppo_checkpoint_interval=150 \
  --builderbench_use_pd=true \
  --builderbench_pd_duration=5 \
  --ppo_skip_first_eval=true \
  --ppo_eval_interval=0 \
  --max_replay_size=10000000 \
  --ppo_crl_repr_tau=0 \
  --hidden_layer_sizes="256,256,256,256,256,256"



```
