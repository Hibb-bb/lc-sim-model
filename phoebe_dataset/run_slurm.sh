#!/bin/bash
# Three-stage SLURM workflow for phoebe_eb_dataset.py on QUEST (Northwestern).
#
#   1. on a login node (needs internet for the passband tables):
#        source ../.venv/bin/activate
#        python phoebe_eb_dataset.py filters --bands "Gaia:G,LSST:g,LSST:r,LSST:i,TESS:T" --out data/filters.json
#   2. sbatch run_slurm.sh                      # the array below
#   3. after the array finishes, on a login node (needs internet + HF token):
#        source .env                            # HF_TOKEN, HF_REPO
#        python phoebe_eb_dataset.py assemble --shards data/shards --filters data/filters.json \
#               --repo "$HF_REPO" --push --save_dir data/hf_local
#
# Budget: ~5-15 CPU-seconds per system per band-set (contact ~3 s, detached
# ~12 s with 5 bands, 201+80 phases, 800 triangles). 5k systems ~ 13-20 CPU-h.
# 25 shards x 8 cores -> a few minutes of wall time per task.

#SBATCH --account=p32593
#SBATCH --partition=short
#SBATCH --job-name=phoebe-eb
#SBATCH --array=0-24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/phoebe-eb_%A_%a.out
#SBATCH --error=logs/phoebe-eb_%A_%a.err

# If b1094/ciera-std is busy, the general-access alternative is one of your
# allocation accounts on a general partition, e.g.:
#   #SBATCH --account=p32234
#   #SBATCH --partition=short        # 4 h limit; 'normal' = 2 d, 'long' = 7 d
# (the b1094 account carries the 'buyin' QOS, which short/normal/long deny,
#  so account and partition have to be switched together.)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs data/shards

# --- environment ---
source ../.venv/bin/activate                  # repo-root venv: uv venv --python 3.11 .venv
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PHOEBE_ENABLE_ONLINE_PASSBANDS=FALSE   # compute nodes have no internet; tables were installed in step 1

NSHARDS=${SLURM_ARRAY_TASK_COUNT:-25}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

python phoebe_eb_dataset.py generate \
    --filters data/filters.json \
    --obs_config obs_config_example.json \
    --out data/shards \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --n_total 5000 --n_val 625 --n_test 625 \
    --morph_fracs 0.70,0.20,0.10 \
    --n_phases 201 --ntriangles 800 --irrad_method none \
    --seed 0
