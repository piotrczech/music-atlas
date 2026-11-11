#!/usr/bin/env bash
#SBATCH --job-name=atlas-fma-metadata
#SBATCH --partition=bem2-cpu-short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${ATLAS_DATA_DIR:?Source ~/.atlas-env.sh before submitting}"
: "${ATLAS_OUTPUTS_DIR:?Source ~/.atlas-env.sh before submitting}"
export MPLCONFIGDIR="${TMPDIR:?}/atlas-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

git rev-parse HEAD
uv run --frozen --no-sync atlas fma-metadata \
  --output "$ATLAS_OUTPUTS_DIR/fma-metadata-${SLURM_JOB_ID}" "$@"
