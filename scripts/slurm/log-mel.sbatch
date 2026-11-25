#!/usr/bin/env bash
#SBATCH --job-name=atlas-log-mel
#SBATCH --partition=bem2-cpu-short
#SBATCH --dependency=singleton
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm-%j.log

# One worker caches one shard at a time and failed shards are retried within the run. Extra
# arguments reach the command and override the defaults, e.g. `--track --limit 2` for a pilot,
# `--resume RUN_ID` to finish it or `--profile quick` for the quick sample.
set -euo pipefail
source "$HOME/.atlas-env.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${ATLAS_DATA_DIR:?Set ATLAS_DATA_DIR to project storage}"
: "${ATLAS_ARTIFACTS_DIR:?Set ATLAS_ARTIFACTS_DIR to project storage}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

uv run --offline --frozen --no-sync atlas log-mel --profile full \
  --workers "${SLURM_CPUS_PER_TASK}" "$@"
