#!/usr/bin/env bash
#SBATCH --job-name=atlas-fma-audit
#SBATCH --partition=bem2-cpu-short
#SBATCH --dependency=singleton
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm-%j.log

# One worker decodes one shard at a time; extra arguments reach the command, e.g. `--limit 2`.
set -euo pipefail
source "$HOME/.atlas-env.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${ATLAS_DATA_DIR:?Set ATLAS_DATA_DIR to project storage}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Each pass skips measured shards, so a retry resumes rather than starts over.
for pass in 1 2 3; do
  uv run --offline --frozen --no-sync atlas fma-audit --workers "${SLURM_CPUS_PER_TASK}" "$@" \
    && exit 0
  echo "Pass ${pass} failed; retrying." >&2
done
exit 1
