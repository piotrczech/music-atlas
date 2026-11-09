#!/usr/bin/env bash
#SBATCH --job-name=atlas-fma-download
#SBATCH --partition=bem2-cpu-short
#SBATCH --dependency=singleton
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=outputs/slurm-%j.log

# Extra arguments reach the command, e.g. `--limit 2` for a pilot. The singleton
# dependency stops a second submission from writing the same shard concurrently.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${ATLAS_DATA_DIR:?Set ATLAS_DATA_DIR to project storage}"
: "${ATLAS_ARTIFACTS_DIR:?Set ATLAS_ARTIFACTS_DIR to project storage}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${ATLAS_ARTIFACTS_DIR}/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${ATLAS_ARTIFACTS_DIR}/venv-${USER}}"
export PYTHONNOUSERSITE=1

# Each pass skips finished shards, so a retry resumes rather than starts over.
for pass in 1 2 3 4 5; do
  uv run --frozen --no-sync atlas fma-download --workers "${SLURM_CPUS_PER_TASK}" "$@" && exit 0
  echo "Pass ${pass} failed; retrying in 5 minutes." >&2
  sleep 300
done
exit 1
