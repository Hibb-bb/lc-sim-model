#!/bin/bash
# Survey-pair experiment, one array task per seed on a Quest GPU node:
#   sbatch run_pairs.sbatch                                   # all methods, AB/AC, probes on A-D
#   sbatch run_pairs.sbatch --variants lejepa_pred            # extra args go to run_pairs.py
# Each task writes results/pairs_unseen/seed<k>; merge the seeds afterwards.

#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lcsim-pairs
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pairs_unseen_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../.venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python run_pairs.py --seeds "$SLURM_ARRAY_TASK_ID" --out "${OUT_ROOT:-results/pairs_unseen}/seed$SLURM_ARRAY_TASK_ID" "$@"
