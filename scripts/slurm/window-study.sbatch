#!/usr/bin/env bash
#SBATCH --job-name=atlas-window-study
#SBATCH --partition=bem2-cpu-short
#SBATCH --dependency=singleton
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=outputs/slurm-%j.log

# The quick sample only: 128 tracks at 5, 10 and 20 s, timed on one core.
set -euo pipefail
source "$HOME/.atlas-env.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${ATLAS_DATA_DIR:?Set ATLAS_DATA_DIR to project storage}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec uv run --offline --frozen --no-sync atlas window-study "$@"
