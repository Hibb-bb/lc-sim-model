#!/bin/bash
# One lcsim experiment on a Quest GPU node. Arguments are passed to run_experiment.py:
#   sbatch run_gpu.sbatch --out results/base
#   sbatch run_gpu.sbatch --variants lejepa,lejepa_aa,lejepa_aab --epochs 100 --out results/lejepa_family
# SSL training runs on the GPU; the sklearn probes run on the CPUs, hence cpus-per-task.

#SBATCH --account=p32593
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=lcsim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lcsim_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../.venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python run_experiment.py "$@"
