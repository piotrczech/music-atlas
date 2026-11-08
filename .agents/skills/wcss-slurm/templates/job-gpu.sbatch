#!/bin/bash
# GPU job. Only the Lem cluster has GPUs: 4 x NVIDIA H100 (GRES type "hopper")
# per node. Bem2 has none.
#
# Three rules the submit filter enforces, all measured:
#   * --gres=gpu:hopper:N is the ONLY accepted form (--gpus=N is rejected)
#   * --cpus-per-task must be stated, and at most 16 cores per GPU
#   * at most 250 GB of memory per GPU
# GPU hours come out of a separate pool from CPU hours; scripts/discover.sh
# shows both, and preflight.sh checks both.

#SBATCH --job-name=EXP-001-train
#SBATCH --account=YOUR_SERVICE_ACCOUNT
#SBATCH --partition=lem-gpu-short          # short=3d; lem-gpu-normal=7d, not 21
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=8                  # dataloader workers; <= 16 per GPU; billed as CPU-hours
#SBATCH --mem=64gb                         # <= 250 GB per GPU
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_EMAIL
##SBATCH --gres=gpu:hopper:1,storage:local:200g   # add node-local NVMe for a large working set

set -euo pipefail

source /usr/local/sbin/modules.sh
module load Python/3.13.5-GCCcore-14.3.0
module load CUDA/12.8

# Dataloader workers are processes; keep BLAS to one thread each.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1                   # ignore ~/.local, which shadows modules

# The Lem GPU nodes carry a different module set from the Lem CPU nodes and
# from Bem2. Verify names in an interactive session on the partition you will
# actually use: sub-interactive-lem-gpu

PD="/lustre/pd03/${SLURM_JOB_ACCOUNT}"      # PD of the service that pays
OUT="$PD/runs/$SLURM_JOB_ID"
source "$PD/envs/myproject/bin/activate"

copy_out() {
    mkdir -p "$OUT"
    if [ -d "$TMPDIR/results" ]; then cp -r "$TMPDIR/results/." "$OUT/"; fi
    echo "finished  $(date -Is)   results $OUT"
}
trap copy_out EXIT

echo "job       ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "gpus      ${SLURM_GPUS_ON_NODE:-?}  visible=${CUDA_VISIBLE_DEVICES:-unset}"
echo "commit    $(git rev-parse --short HEAD 2>/dev/null || echo no-repo)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Compute nodes have outbound internet, so a first-run model download works.
# Point caches at scratch: faster than PD, and $HOME's file quota stays intact.
export HF_HOME="$TMPDIR/hf"
export TORCH_HOME="$TMPDIR/torch"

mkdir -p "$TMPDIR/results"
python train.py --out "$TMPDIR/results"

# Alternative to a venv: run inside a prebuilt container. --nv exposes the GPUs.
#   apptainer exec --nv /lustre/software-data/container-images/pytorch_latest.sif \
#       python train.py --out "$TMPDIR/results"
