#!/usr/bin/env bash
#SBATCH --job-name=atlas-smoke
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=outputs/slurm-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${ATLAS_DATA_DIR:?Set ATLAS_DATA_DIR to project storage}"
: "${ATLAS_ARTIFACTS_DIR:?Set ATLAS_ARTIFACTS_DIR to project storage}"
: "${ATLAS_OUTPUTS_DIR:?Set ATLAS_OUTPUTS_DIR to project storage}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ATLAS_ARTIFACTS_DIR}/uv-cache}"
export HF_HOME="${HF_HOME:-${ATLAS_ARTIFACTS_DIR}/hf-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${ATLAS_ARTIFACTS_DIR}/venv-${USER}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# Sync the environment before submitting; compute nodes may have no internet.
uv run --frozen --no-sync atlas doctor --profile quick
uv run --frozen --no-sync atlas demo --output "${ATLAS_OUTPUTS_DIR}/smoke-${SLURM_JOB_ID}.wav"
uv run --frozen --no-sync atlas analyze "${ATLAS_OUTPUTS_DIR}/smoke-${SLURM_JOB_ID}.wav" \
  > "${ATLAS_OUTPUTS_DIR}/smoke-${SLURM_JOB_ID}.json"
