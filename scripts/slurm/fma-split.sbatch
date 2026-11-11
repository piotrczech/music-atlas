#!/usr/bin/env bash
#SBATCH --job-name=atlas-fma-split
#SBATCH --partition=bem2-cpu-short
#SBATCH --dependency=singleton
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail
source "$HOME/.atlas-env.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${ATLAS_DATA_DIR:?Set ATLAS_DATA_DIR to project storage}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec uv run --offline --frozen --no-sync atlas fma-split "$@"
