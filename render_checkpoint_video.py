"""Render one rollout from a saved PPO checkpoint and log it to wandb.

Ad-hoc "what is the robot actually doing" check for a run you stopped
mid-training (e.g. because it looked stalled) -- unlike the end-of-training
video hook in `ppo_contrastive.py`, this loads params from a checkpoint
file on disk instead of an in-memory training state, and starts its own
short-lived wandb run instead of resuming the (finished/killed) training
run's.

Example (matches close_subtask_train.slurm's --array=0-2 run 0):
  python render_checkpoint_video.py \
      --checkpoint=logs/ppo_maniskill_close_subtask_train_1M/ppo_maniskill_close_subtask_train_0/checkpoints/latest.pkl \
      --env=maniskill_close_subtask_train \
      --wandb_project=sgcrl-maniskill-close-subtask-train \
      --wandb_group=ppo_goal_fast
"""
import sgcrl_jax_acme_compat  # noqa: F401 — must precede all acme/jax imports

import argparse
import os

from contrastive import ppo_learner
from ppo_contrastive import fixed_goal_dict
import ppo_video_utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint',
      default='logs/ppo_maniskill_close_subtask_train_1M/'
              'ppo_maniskill_close_subtask_train_0/checkpoints/latest.pkl',
      help='Path to a checkpoint .pkl (default: run 0\'s latest.pkl under '
           'the close_subtask_train.slurm log dir).')
  parser.add_argument('--env', default='maniskill_close_subtask_train',
                      choices=list(fixed_goal_dict.keys()))
  parser.add_argument('--output', default='',
                      help='Where to write the .mp4. Defaults to '
                           '<checkpoint_dir>/../rollout_<iter>.mp4.')
  parser.add_argument('--width', type=int, default=640)
  parser.add_argument('--height', type=int, default=480)
  parser.add_argument('--fps', type=int, default=30)
  parser.add_argument('--max_steps', type=int, default=-1,
                      help='Override rollout length; -1 = env default.')
  parser.add_argument('--stochastic', action='store_true',
                      help='Sample from the policy instead of using mode().')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--hidden_layer_sizes', default='',
                      help='Comma-separated widths. If empty (default), '
                           'read from the checkpoint\'s stored '
                           '`hidden_layer_sizes` field.')
  parser.add_argument('--wandb_project', default='',
                      help='wandb project to log the video to. Empty '
                           '(default) disables wandb -- video is only '
                           'written to --output.')
  parser.add_argument('--wandb_entity', default='',
                      help='Empty defers to the WANDB_ENTITY env var.')
  parser.add_argument('--wandb_group', default='',
                      help='e.g. the training run\'s wandb_group, so this '
                           'shows up clustered with it in the UI.')
  parser.add_argument('--wandb_run_name', default='',
                      help='Empty auto-generates from the checkpoint path.')
  args = parser.parse_args()

  ckpt = ppo_learner.load_checkpoint(args.checkpoint)
  iteration = ckpt.get('iteration')
  global_step = ckpt.get('global_step')
  print(f'[render] loaded {args.checkpoint} '
        f'(iteration={iteration}, global_step={global_step})')

  if args.hidden_layer_sizes:
    hidden_layer_sizes = tuple(
        int(x) for x in args.hidden_layer_sizes.split(','))
  else:
    stored = ckpt.get('hidden_layer_sizes')
    if stored is None:
      raise ValueError(
          'Checkpoint has no stored hidden_layer_sizes; pass '
          '--hidden_layer_sizes explicitly.')
    hidden_layer_sizes = tuple(stored)
  print(f'[render] hidden_layer_sizes={hidden_layer_sizes}')

  _obs_norm_state = ckpt.get('obs_normalizer_state')
  obs_normalizer = (
      ppo_learner.ObsNormalizer.from_state_dict(_obs_norm_state)
      if _obs_norm_state is not None else None)
  if obs_normalizer is None:
    print('[render] WARNING: checkpoint has no obs_normalizer_state -- '
          'feeding the policy raw observations.')

  if args.output:
    output_path = args.output
  else:
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    output_path = os.path.join(
        os.path.dirname(ckpt_dir), f'rollout_iter{iteration}.mp4')

  video_path = ppo_video_utils.record_rollout_video(
      env_name=args.env,
      seed=args.seed,
      policy_params=ckpt['policy_params'],
      hidden_layer_sizes=hidden_layer_sizes,
      output_path=output_path,
      fixed_start_end=fixed_goal_dict[args.env],
      max_steps=(None if args.max_steps < 0 else args.max_steps),
      fps=args.fps,
      stochastic=args.stochastic,
      obs_normalizer=obs_normalizer,
  )

  if args.wandb_project:
    import wandb
    run_name = (args.wandb_run_name or
               f'render_{os.path.splitext(os.path.basename(args.checkpoint))[0]}'
               f'_iter{iteration}')
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=(args.wandb_entity or None),
        name=run_name,
        group=(args.wandb_group or None),
        job_type='render_video',
        config={
            'checkpoint': args.checkpoint,
            'env': args.env,
            'iteration': iteration,
            'global_step': global_step,
            'stochastic': args.stochastic,
        },
    )
    print(f'[render] wandb run: {wandb_run.url}')
    wandb_run.log({'video/rollout': wandb.Video(
        video_path, fps=args.fps, format='mp4')})
    wandb_run.finish()
    print('[render] logged video to wandb')


if __name__ == '__main__':
  main()
