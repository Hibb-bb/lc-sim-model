#!/bin/bash
# Three-stage SLURM workflow for phoebe_eb_dataset.py.
#
#   1. on a login node (needs internet for the passband tables):
#        python phoebe_eb_dataset.py filters --bands "Gaia:G,LSST:g,LSST:r,LSST:i,TESS:T" --out data/filters.json
#   2. sbatch run_slurm.sh                      # the array below
#   3. after the array finishes, on a login node (needs internet + HF token):
#        export HF_TOKEN=hf_...
#        python phoebe_eb_dataset.py assemble --shards data/shards --filters data/filters.json \
#               --repo <hf-user>/phoebe-eb-multiband --push --save_dir data/hf_local
#
# Budget: ~5-15 CPU-seconds per system per band-set (contact ~3 s, detached
# ~12 s with 5 bands, 201+80 phases, 800 triangles). 20k systems ~ 50-80 CPU-h.
# 100 shards x 8 cores -> well under an hour of wall time.

#SBATCH --job-name=phoebe-eb
#SBATCH --array=0-99
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/phoebe-eb_%A_%a.out

set -euo pipefail
mkdir -p logs data/shards

# --- environment: adapt to your cluster ---
# module load python/3.11
# source ~/venvs/phoebe/bin/activate        # pip install -r requirements.txt
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PHOEBE_ENABLE_ONLINE_PASSBANDS=FALSE   # compute nodes usually have no internet; tables were installed in step 1

NSHARDS=${SLURM_ARRAY_TASK_COUNT:-100}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

python phoebe_eb_dataset.py generate \
    --filters data/filters.json \
    --obs_config obs_config_example.json \
    --out data/shards \
    --shard "$SHARD" --nshards "$NSHARDS" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --n_total 20000 --n_val 2500 --n_test 2500 \
    --morph_fracs 0.70,0.20,0.10 \
    --n_phases 201 --ntriangles 800 --irrad_method none \
    --seed 0
