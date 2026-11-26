#!/usr/bin/env bash
#SBATCH --job-name=atlas-embeddings
#SBATCH --partition=lem-gpu-short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:hopper:1,storage:local:50g
#SBATCH --output=outputs/clap-%j.log

set -euo pipefail
source "$HOME/.atlas-env.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${HF_HOME:?Set HF_HOME on project storage}"
if [[ "${1:-}" == prepare-musicfm && "${SLURM_JOB_PARTITION:-}" != lem-cpu-* ]]; then
    echo "Submit prepare-musicfm on Lem CPU with --gres=storage:local:50g." >&2
    exit 2
fi
export UV_PROJECT_ENVIRONMENT="${TMPDIR:?}/venv" UV_LINK_MODE=copy
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
uv sync --locked --extra training --extra research --extra tracking
atlas="$UV_PROJECT_ENVIRONMENT/bin/atlas"
case "${1:-}" in
    prepare-musicfm)
        exec "$UV_PROJECT_ENVIRONMENT/bin/python" -m music_atlas.musicfm
        ;;
    pilot)
        shift
        set -- --profile quick --encoder both "$@"
        ;;
    benchmark)
        shift
        for batch in 16 32; do
            "$atlas" embeddings --device cuda --track "$@" \
                --profile full --benchmark-shards 3 --batch-size "$batch"
        done
        exit 0
        ;;
esac
exec "$atlas" embeddings --device cuda --track "$@"
