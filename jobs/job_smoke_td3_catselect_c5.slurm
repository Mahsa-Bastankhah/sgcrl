#!/bin/bash
#SBATCH --job-name=smoke_td3_cat_c5
#SBATCH --output=slurm/smoke_td3_catselect_c5_%j.log
#SBATCH --error=slurm/smoke_td3_catselect_c5_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --account=seas
#SBATCH --exclude=neu306

# Minimal TD3 + categorical-select init/compile smoke for c5t2 dims.
unset LD_PRELOAD
module purge
try_cuda() {
  if module load "$@" 2>/dev/null; then
    export CUDA_HOME="${CUDA_HOME:-${CUDA_ROOT:-}}"
    [ -n "${CUDA_HOME:-}" ] && return 0
    for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
      [ -d "${_cand}/lib64" ] && export CUDA_HOME="${_cand}" && return 0
    done
    return 1
  fi
  return 1
}
try_cuda cudatoolkit/12.6 || try_cuda cudatoolkit/12.5 || try_cuda cudatoolkit/12.4 || true
if [ -z "${CUDA_HOME:-}" ]; then
  for _cand in /usr/local/cuda-12.6 /usr/local/cuda-12.5 /usr/local/cuda; do
    if [ -d "${_cand}/lib64" ]; then
      export CUDA_HOME="${_cand}"
      export PATH="${CUDA_HOME}/bin${PATH:+:${PATH}}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      break
    fi
  done
fi

cd /n/fs/mislresearch/sgcrl
source /n/fs/mislresearch/miniconda3/etc/profile.d/conda.sh
conda activate sgcrl_builderbench
source scripts/sgcrl_builderbench_cuda_env.sh
export BUILDERBENCH_ROOT=/n/fs/mislresearch/builderbench
export BUILDERBENCH_MJX_IMPL=jax
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
mkdir -p slurm
nvidia-smi || true

python -u <<'PY'
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import sgcrl_jax_acme_compat  # noqa: F401
from acme import specs

from contrastive import networks as cnets
from contrastive import td3_density as td3
from contrastive import config as cconfig
from contrastive.ppo_learner import make_ppo_update_fn

print('[smoke] devices', jax.devices(), flush=True)
# Match c5t2 PD policy-obs packing: state_dim=16, goal_dim=15, act=5, 5 cubes.
obs_dim, goal_dim, act_dim, n_cubes = 16, 15, 5, 5
obs_shape = (obs_dim + goal_dim,)
act_shape = (act_dim,)
spec = SimpleNamespace(
    observations=specs.Array(shape=obs_shape, dtype=np.float32),
    actions=specs.BoundedArray(
        shape=act_shape, dtype=np.float32, minimum=-1.0, maximum=1.0),
)

t0 = time.time()
print('[smoke] make_networks(catselect)...', flush=True)
nets = cnets.make_networks(
    spec, obs_dim=obs_dim, repr_dim=64,
    hidden_layer_sizes=(256,) * 6,
    actor_min_std=0.01,
    categorical_select_classes=n_cubes,
)
print(f'[smoke] networks ok in {time.time()-t0:.1f}s', flush=True)

t0 = time.time()
key = jax.random.PRNGKey(0)
k_pol, k_val, k_q = jax.random.split(key, 3)
print('[smoke] init policy...', flush=True)
pol = nets.policy_network.init(k_pol)
print(f'[smoke] policy ok {time.time()-t0:.1f}s', flush=True)
print('[smoke] init value...', flush=True)
val = nets.value_network.init(k_val)
print(f'[smoke] value ok {time.time()-t0:.1f}s', flush=True)
print('[smoke] init td3...', flush=True)
td3_nets = td3.make_td3_density_networks(
    obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim,
    hidden_layer_sizes=(256,) * 6)
q = td3.init_td3_params(td3_nets, k_q)
print(f'[smoke] td3 ok {time.time()-t0:.1f}s', flush=True)

print('[smoke] sample hybrid action...', flush=True)
obs = jnp.zeros((8, obs_dim + goal_dim), dtype=jnp.float32)
dist = nets.policy_network.apply(pol, obs)
act = nets.sample(dist, jax.random.PRNGKey(1))
lp = nets.log_prob(dist, act)
print('[smoke] action', act.shape, 'logp', float(lp.mean()), flush=True)

print('[smoke] make_td3_update...', flush=True)
opt = optax.adam(3e-4)
update = td3.make_td3_density_update_fn(
    td3_nets,
    policy_network=nets.policy_network,
    sample_fn=nets.sample,
    optimizer=opt,
    obs_dim=obs_dim,
    start_index=0,
    end_index=obs_dim,
    discount=0.99,
    tau=0.005,
    goal_tol=1e-2,
    use_target_policy=False,
    cross_batch_goals=False,
    normalize_obs=True,
    obs_norm_clip=10.0,
)
print('[smoke] make_td3_reward...', flush=True)
rew_fn = td3.make_td3_reward_fn(
    td3_nets, obs_dim=obs_dim, discount=0.99, log_reward=True,
    start_index=0, end_index=obs_dim, normalize_obs=True, obs_norm_clip=10.0)

B = 64
batch = {
    'obs': jnp.zeros((B, obs_dim + goal_dim), jnp.float32),
    'action': jnp.zeros((B, act_dim), jnp.float32),
    'next_obs': jnp.zeros((B, obs_dim + goal_dim), jnp.float32),
}
obs_mean = jnp.zeros((obs_dim,), jnp.float32)
obs_var = jnp.ones((obs_dim,), jnp.float32)
opt_state = opt.init(td3.online_td3_params(q))
print('[smoke] first TD3 update (JIT compile)...', flush=True)
t0 = time.time()
q, opt_state, metrics, _ = update(
    q, opt_state, batch, jax.random.PRNGKey(2), pol, pol, obs_mean, obs_var)
jax.block_until_ready(metrics['td3_qf_loss'])
print(f'[smoke] TD3 update done in {time.time()-t0:.1f}s '
      f'loss={float(metrics["td3_qf_loss"]):.4f}', flush=True)

print('[smoke] first TD3 reward (JIT)...', flush=True)
t0 = time.time()
r = rew_fn(q, batch['obs'], batch['action'], obs_mean, obs_var)
jax.block_until_ready(r)
print(f'[smoke] reward done in {time.time()-t0:.1f}s mean={float(r.mean()):.4f}',
      flush=True)

cfg = cconfig.ContrastiveConfig()
cfg.obs_dim = obs_dim
cfg.start_index = 0
cfg.end_index = obs_dim
cfg.ppo_norm_obs = True
cfg.ppo_categorical_select = True
cfg.hidden_layer_sizes = (256,) * 6
ppo_opt = optax.adam(3e-4)
ppo_params = {'policy': pol, 'value': val}
ppo_opt_state = ppo_opt.init(ppo_params)
print('[smoke] make_ppo_update_fn...', flush=True)
ppo_update = make_ppo_update_fn(nets, cfg, ppo_opt)
# Fresh batch-sized actions for PPO
dist_b = nets.policy_network.apply(pol, batch['obs'])
act_b = nets.sample(dist_b, jax.random.PRNGKey(4))
lp_b = nets.log_prob(dist_b, act_b)
mb = {
    'obs': batch['obs'],
    'actions': act_b,
    'old_logprobs': lp_b,
    'advantages': jnp.zeros((B,), jnp.float32),
    'returns': jnp.zeros((B,), jnp.float32),
    'old_values': jnp.zeros((B,), jnp.float32),
}
print('[smoke] first PPO update (JIT)...', flush=True)
t0 = time.time()
ppo_params, ppo_opt_state, ppo_m = ppo_update(
    ppo_params, ppo_opt_state, mb, jax.random.PRNGKey(3),
    obs_mean, obs_var)
jax.block_until_ready(next(iter(ppo_m.values())))
print(f'[smoke] PPO update done in {time.time()-t0:.1f}s', flush=True)
print('[smoke] SUCCESS', flush=True)
PY
