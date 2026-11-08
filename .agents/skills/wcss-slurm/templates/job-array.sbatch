#!/bin/bash
# Parameter sweep as a SLURM array job.
#
# Cost is per task: 8 cores x 2 h x 200 tasks = 3200 CPU-hours. preflight.sh
# multiplies by the array size for you; check before submitting.

#SBATCH --job-name=EXP-001-sweep
#SBATCH --account=YOUR_SERVICE_ACCOUNT
#SBATCH --partition=bem2-cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=2:00:00                     # per task, not for the whole array
#SBATCH --array=1-200%20                   # 200 tasks, at most 20 at once
#SBATCH --output=%x_%A_%a.out              # %A = array id, %a = task index
#SBATCH --error=%x_%A_%a.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_EMAIL

# The %20 throttle keeps you inside the 150-running-job association limit and
# leaves the backfill scheduler room. MaxArraySize is 1001: indices stay below it.

set -euo pipefail

source /usr/local/sbin/modules.sh
module load Python/3.13.5-GCCcore-14.3.0

# One process per task using all its cores through threads. For a Pool inside
# each task use OMP_NUM_THREADS=1 instead. See references/python.md.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONNOUSERSITE=1                   # ignore ~/.local, which shadows modules

PD="/lustre/pd03/${SLURM_JOB_ACCOUNT}"      # PD of the service that pays
source "$PD/envs/myproject/bin/activate"

# Key everything on %A_%a. $SLURM_JOB_ID is NOT unique across tasks: the last
# task reuses the array's own job id.
TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
OUT="$PD/runs/${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"

# Many small files across many tasks exhaust the PD inode quota faster than
# the byte quota: one archive per task, copied out on every exit.
copy_out() {
    mkdir -p "$OUT"
    if [ -d "$TMPDIR/results" ]; then tar -C "$TMPDIR" -cf "$OUT/results.tar" results; fi
    echo "finished  $(date -Is)   results $OUT/results.tar"
}
trap copy_out EXIT

echo "task      ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}  (tag $TAG)"
echo "commit    $(git rev-parse --short HEAD 2>/dev/null || echo no-repo)"

# Pick this task's parameters from a manifest, one line per task.
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PD/sweep/params.txt")
[ -z "$PARAMS" ] && { echo "no parameters on line ${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
echo "params    $PARAMS"

mkdir -p "$TMPDIR/results"
python train.py --params "$PARAMS" --out "$TMPDIR/results"
