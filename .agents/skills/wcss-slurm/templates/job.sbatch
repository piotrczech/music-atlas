#!/bin/bash
# Single-node CPU job on WCSS. Replace the CAPITALISED placeholders; preflight
# refuses to pass a script that still contains one.
#   check:   scripts/preflight.sh templates/job.sbatch
#   submit:  JOBID=$(sbatch --parsable templates/job.sbatch)
#   record:  scripts/run-record.sh "$JOBID"

#SBATCH --job-name=EXP-001-baseline
#SBATCH --account=YOUR_SERVICE_ACCOUNT     # explicit: it decides whose grant pays
#SBATCH --partition=bem2-cpu-short         # short=3d, normal=21d; see references/partitions-and-hardware.md
#SBATCH --cpus-per-task=16                 # billed for these whether busy or idle
#SBATCH --mem=64gb                         # default is only 4000 MB per core
#SBATCH --time=4:00:00                     # MANDATORY here; a bare number means MINUTES
#SBATCH --output=%x_%j.out                 # lands in the submit directory; if $HOME is near its quota, write a fixed
#SBATCH --error=%x_%j.err                  # PD path here (sbatch does not expand $variables in directives)
##SBATCH --mail-type=END,FAIL              # remove one '#' and fill in to get mail
##SBATCH --mail-user=YOUR_EMAIL

# Stop on the first error. Without this a failed `module load` lets the script
# run on and burn the whole allocation producing nothing.
set -euo pipefail

# Project storage of the service that PAYS for this job. $PDDIR would also
# work inside the job, but deriving it from the account makes the choice
# visible in the script and identical on the login node.
PD="/lustre/pd03/${SLURM_JOB_ACCOUNT}"
OUT="$PD/runs/$SLURM_JOB_ID"

# This file exists on compute nodes but NOT on the login node.
source /usr/local/sbin/modules.sh

# --- software: pick ONE ---------------------------------------------------
# (a) numpy / scipy / pandas and nothing else: the cluster's own build, no venv.
#     2024.05 is the one version present on both Bem2 and Lem.
module load SciPy-bundle/2024.05-gfbf-2024a
# (b) anything pip-installed: the bare module Python (it has no numpy) plus a
#     venv kept on PD; see references/python.md for building it.
# module load Python/3.13.5-GCCcore-14.3.0
# source "$PD/envs/myproject/bin/activate"

# --- threads vs processes: pick ONE of the two lines -----------------------
# The libraries default to the NODE's core count, not your allocation. Either
# let one process use all your cores through BLAS/OpenMP threads ...
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# ... or, if train.py fans out with multiprocessing.Pool, give each worker one
# thread instead (processes x threads must not exceed the allocation):
# export OMP_NUM_THREADS=1
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONNOUSERSITE=1                   # ignore ~/.local, which shadows module packages

# Copy results out on EVERY exit: success, error, and the 30 s grace period
# after a wall-clock kill. $TMPDIR also survives 14 days in finished_jobs/,
# but that is a safety net, not a plan.
copy_out() {
    mkdir -p "$OUT"
    if [ -d "$TMPDIR/results" ]; then cp -r "$TMPDIR/results/." "$OUT/"; fi
    echo "finished  $(date -Is)   results $OUT"
}
trap copy_out EXIT

# Provenance. Capture it now: sacct forgets this job in about three months.
echo "job       ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "account   ${SLURM_JOB_ACCOUNT}   partition ${SLURM_JOB_PARTITION}"
echo "commit    $(git rev-parse --short HEAD 2>/dev/null || echo no-repo)"
echo "tmpdir    ${TMPDIR}"
echo "started   $(date -Is)"

# Compute against fast scratch, not against $HOME or PD.
mkdir -p "$TMPDIR/results"
python train.py --out "$TMPDIR/results"
