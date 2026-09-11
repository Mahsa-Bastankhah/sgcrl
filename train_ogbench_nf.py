"""Pre-train Normalizing Flow density estimator on OGBench dataset."""
import os
import pickle
import sys
import time
from typing import Sequence

from absl import app, flags
import jax
import jax.numpy as jnp
import numpy as np

# Ensure sgcrl is in python path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from contrastive import nf_density as nf
import ogbench_data

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cube-double-play-v0', 'OGBench dataset name.')
flags.DEFINE_string(
    'dataset_dir', '', 'Path to ogbench datasets directory. Empty string uses $SCRATCH/.ogbench/data.'
)
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('train_steps', 500000, 'Total training gradient steps.')
flags.DEFINE_integer('batch_size', 256, 'Training batch size.')
flags.DEFINE_integer('eval_batch_size', 256, 'Validation evaluation batch size.')
flags.DEFINE_integer('log_interval', 100, 'Logging interval in gradient steps.')
flags.DEFINE_integer('eval_interval', 2500, 'Evaluation interval in gradient steps.')
flags.DEFINE_integer('save_interval', 50000, 'Checkpoint saving interval.')
flags.DEFINE_string(
    'save_dir', '', 'Directory to save checkpoints and logs. Empty string defaults to $SCRATCH/ogbench_nf_exp/.'
)
flags.DEFINE_float('gamma', 0.99, 'Geometric discount for future goal sampling.')

# Normalizing Flow Architecture knobs
flags.DEFINE_bool('state_only', False, 'If True, learn p(g|s); if False, learn p(g|s,a).')
flags.DEFINE_integer('nf_rep_size', 256, 'Representation dimension of the SA/S encoder.')
flags.DEFINE_integer('nf_num_blocks', 8, 'Number of RealNVP affine coupling blocks.')
flags.DEFINE_integer('nf_coupling_width', 512, 'Hidden layer width for RealNVP coupling networks.')
flags.DEFINE_integer('nf_sa_hidden', 1024, 'Hidden width for SA conditioning encoder.')
flags.DEFINE_integer('nf_sa_num_layers', 4, 'Number of layers for SA conditioning encoder.')
flags.DEFINE_integer('nf_goal_enc_size', 0, '0 = raw normalized goal to flow; >0 = goal encoder MLP.')
flags.DEFINE_float('nf_critic_lr', 1e-4, 'Learning rate for RealNVP flow (AdamW).')
flags.DEFINE_float('nf_encoder_lr', 3e-4, 'Learning rate for conditioning and goal encoders (Adam).')
flags.DEFINE_float('nf_critic_weight_decay', 1e-6, 'Weight decay for RealNVP flow.')
flags.DEFINE_float('nf_grad_clip', 1.0, 'Global gradient clip norm.')
flags.DEFINE_float('nf_noise_std', 0.0, 'Gaussian noise std added to goals during training.')

# WandB logging
flags.DEFINE_string('wandb_project', '', 'WandB project name. Empty string disables WandB.')
flags.DEFINE_string('wandb_group', '', 'WandB group name.')
flags.DEFINE_string('wandb_entity', '', 'WandB entity.')
flags.DEFINE_string('wandb_run_name', '', 'WandB run name.')


def main(_):
    # Resolve save directory
    if FLAGS.save_dir:
        save_dir = os.path.expanduser(FLAGS.save_dir)
    else:
        scratch = os.environ.get('SCRATCH', os.path.expanduser('~'))
        save_dir = os.path.join(scratch, 'ogbench_nf_exp', f'{FLAGS.env_name}_s{FLAGS.seed}')
    os.makedirs(save_dir, exist_ok=True)

    dataset_dir = FLAGS.dataset_dir or None

    print('=================================================================')
    print(f'[train_ogbench_nf] Initializing NF Pre-Training for {FLAGS.env_name}')
    print(f'[train_ogbench_nf] Output directory: {save_dir}')
    print('=================================================================')

    # 1. Load Datasets & Goal Statistics
    train_dataset, val_dataset, goal_mean_np, goal_std_np = ogbench_data.make_ogbench_datasets(
        dataset_name=FLAGS.env_name,
        dataset_dir=dataset_dir,
        gamma=FLAGS.gamma,
        seed=FLAGS.seed,
    )
    obs_dim = train_dataset.obs_dim
    act_dim = train_dataset.act_dim
    goal_dim = train_dataset.goal_dim
    goal_mean = jnp.asarray(goal_mean_np)
    goal_std = jnp.asarray(goal_std_np)

    print(f'[train_ogbench_nf] Obs dim: {obs_dim}, Act dim: {act_dim}, Goal dim: {goal_dim}')
    print(f'[train_ogbench_nf] Train size: {train_dataset.size}, Val size: {val_dataset.size}')
    print(f'[train_ogbench_nf] Flow: {FLAGS.nf_num_blocks} blocks, width {FLAGS.nf_coupling_width}')
    print(f'[train_ogbench_nf] SA encoder: {FLAGS.nf_sa_num_layers}x{FLAGS.nf_sa_hidden} -> {FLAGS.nf_rep_size}')

    # 2. Build NF Density Networks & Optimizers
    nf_networks = nf.make_nf_density_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        goal_dim=goal_dim,
        rep_size=FLAGS.nf_rep_size,
        num_blocks=FLAGS.nf_num_blocks,
        channels=FLAGS.nf_coupling_width,
        goal_enc_size=FLAGS.nf_goal_enc_size,
        sa_hidden=FLAGS.nf_sa_hidden,
        sa_num_layers=FLAGS.nf_sa_num_layers,
        state_only=FLAGS.state_only,
    )

    optimizer = nf.make_nf_optimizers(
        encoder_lr=FLAGS.nf_encoder_lr,
        critic_lr=FLAGS.nf_critic_lr,
        critic_weight_decay=FLAGS.nf_critic_weight_decay,
        grad_clip=FLAGS.nf_grad_clip,
        has_goal_encoder=(FLAGS.nf_goal_enc_size > 0),
    )

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, k_init = jax.random.split(rng)
    params = nf.init_nf_params(nf_networks, k_init)
    opt_state = optimizer.init(params)

    # 3. Build Update and Eval Functions
    update_fn = nf.make_nf_density_update_fn(
        nf_networks=nf_networks,
        optimizer=optimizer,
        obs_dim=obs_dim,
        noise_std=FLAGS.nf_noise_std,
    )

    @jax.jit
    def eval_step(params, states, actions, goals):
        # Normalize goals
        goals_norm = (goals - goal_mean) / (goal_std + 1e-8)
        # Compute B x B pairwise log-density matrix
        pairwise_logp = nf.compute_pairwise_log_prob(
            nf_networks, params, states, actions, goals_norm
        )
        metrics = nf.compute_categorical_metrics(pairwise_logp)
        # Also compute paired validation NLL
        log_p = nf.nf_log_prob(nf_networks, params, states, actions, goals_norm)
        metrics['val_density_loss'] = -jnp.mean(log_p)
        metrics['val_log_p_mean'] = jnp.mean(log_p)
        return metrics

    # Optional WandB setup
    use_wandb = bool(FLAGS.wandb_project)
    if use_wandb:
        import wandb
        wandb_run_name = FLAGS.wandb_run_name or f'nf_{FLAGS.env_name}_s{FLAGS.seed}'
        wandb.init(
            project=FLAGS.wandb_project,
            group=FLAGS.wandb_group or f'nf_{FLAGS.env_name}',
            entity=FLAGS.wandb_entity or None,
            name=wandb_run_name,
            config=FLAGS.flag_values_dict(),
        )

    # 4. Training Loop
    csv_path = os.path.join(save_dir, 'metrics.csv')
    csv_file = open(csv_path, 'w', buffering=1)
    csv_header_written = False

    t0 = time.time()
    last_log_time = t0
    lam_val = None

    print('[train_ogbench_nf] Starting training loop...')
    for step in range(1, FLAGS.train_steps + 1):
        # Sample training batch
        train_batch_np = train_dataset.sample(FLAGS.batch_size)
        train_batch = {
            'obs': jnp.asarray(train_batch_np['obs']),
            'action': jnp.asarray(train_batch_np['action']),
        }

        rng, k_step = jax.random.split(rng)
        params, opt_state, lam_val, train_metrics = update_fn(
            params, opt_state, train_batch, k_step, goal_mean, goal_std, lam_val
        )

        # Logging
        if step % FLAGS.log_interval == 0 or step == 1:
            step_time = (time.time() - last_log_time) / (
                FLAGS.log_interval if step > 1 else 1
            )
            sps = (FLAGS.batch_size * FLAGS.log_interval) / max(time.time() - last_log_time, 1e-6)
            last_log_time = time.time()

            log_dict = {
                'step': step,
                'train/density_loss': float(train_metrics['density_loss']),
                'train/log_p_mean': float(train_metrics['log_p_mean']),
                'train/repr_norm': float(train_metrics['repr_norm']),
                'train/flow_grad_norm': float(train_metrics['flow_grad_norm']),
                'train/encoder_grad_norm': float(train_metrics['encoder_grad_norm']),
                'train/sps': sps,
            }

            # Periodic Validation Evaluation
            if step % FLAGS.eval_interval == 0 or step == FLAGS.train_steps:
                val_batch_np = val_dataset.sample(FLAGS.eval_batch_size)
                val_states = jnp.asarray(val_batch_np['state'])
                val_actions = jnp.asarray(val_batch_np['action'])
                val_goals = jnp.asarray(val_batch_np['goal'])

                val_metrics = eval_step(params, val_states, val_actions, val_goals)
                log_dict.update({
                    'val/categorical_accuracy': float(val_metrics['categorical_accuracy']),
                    'val/top_5_accuracy': float(val_metrics['top_5_accuracy']),
                    'val/pos_logp': float(val_metrics['pos_logp']),
                    'val/neg_logp': float(val_metrics['neg_logp']),
                    'val/margin': float(val_metrics['margin']),
                    'val/implicit_infonce': float(val_metrics['implicit_infonce']),
                    'val/density_loss': float(val_metrics['val_density_loss']),
                })

                print(
                    f'[Step {step:7d}/{FLAGS.train_steps}] '
                    f'Train NLL: {log_dict["train/density_loss"]:.4f} | '
                    f'Val NLL: {log_dict["val/density_loss"]:.4f} | '
                    f'Val CatAcc: {log_dict["val/categorical_accuracy"] * 100:.2f}% | '
                    f'Val Top5Acc: {log_dict["val/top_5_accuracy"] * 100:.2f}% | '
                    f'Margin: {log_dict["val/margin"]:.3f}'
                )
            else:
                print(
                    f'[Step {step:7d}/{FLAGS.train_steps}] '
                    f'Train NLL: {log_dict["train/density_loss"]:.4f} | '
                    f'LogP Mean: {log_dict["train/log_p_mean"]:.4f} | '
                    f'SPS: {sps:.1f}'
                )

            # Write to CSV
            if not csv_header_written:
                csv_file.write(','.join(log_dict.keys()) + '\n')
                csv_header_written = True
            csv_file.write(','.join(str(v) for v in log_dict.values()) + '\n')

            if use_wandb:
                wandb.log(log_dict, step=step)

        # Periodic Checkpointing
        if step % FLAGS.save_interval == 0 or step == FLAGS.train_steps:
            ckpt_path = os.path.join(save_dir, f'checkpoint_step{step}.pkl')
            ckpt_data = {
                'params': jax.device_get(params),
                'goal_mean': np.asarray(goal_mean_np),
                'goal_std': np.asarray(goal_std_np),
                'step': step,
                'env_name': FLAGS.env_name,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
                'goal_dim': goal_dim,
                'config': FLAGS.flag_values_dict(),
            }
            with open(ckpt_path, 'wb') as f:
                pickle.dump(ckpt_data, f)
            print(f'[train_ogbench_nf] Checkpoint saved: {ckpt_path}')

    csv_file.close()
    if use_wandb:
        wandb.finish()
    print('[train_ogbench_nf] Pre-training completed successfully.')


if __name__ == '__main__':
    app.run(main)
