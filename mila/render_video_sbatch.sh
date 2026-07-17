#!/bin/bash
#SBATCH --job-name=render_videos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%j.out
# ==============================================================================
# Runs mila/render_videos.sh as a batch job (instead of interactively).
#
# Any arguments passed to sbatch after the script name are forwarded as-is
# to render_videos.sh, e.g.:
#
#   sbatch mila/render_videos.slurm --exp_name=reproduce --seed=0
#   sbatch mila/render_videos.slurm --exp_name=reproduce,reproduce_longer_rollout
#   sbatch mila/render_videos.slurm --dry_run
#   sbatch mila/render_videos.slurm            # no filters = render everything
# ==============================================================================

set -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

module unload python; module load anaconda/3
conda activate sgcrl_builderbench

export BUILDERBENCH_ROOT=/home/mila/m/mohammad-sami-nur.islam/sgcrl/builderbench

cd "$REPO_DIR"
bash /home/mila/m/mohammad-sami-nur.islam/sgcrl/mila/render_vid.sh "$@"
