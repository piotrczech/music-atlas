#!/bin/bash
# A job longer than the partition allows, run as a chain of legs. Each leg
# checkpoints to PD, gets a signal before its wall limit, saves, and requeues
# itself; the next leg resumes from the newest checkpoint. See
# references/long-jobs.md. Replace the CAPITALISED placeholders.

#SBATCH --job-name=EXP-001-long
#SBATCH --account=YOUR_SERVICE_ACCOUNT
#SBATCH --partition=bem2-cpu-short          # 3 d per leg; the chain continues past it
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@600                 # USR1 to this shell ~10 min before the limit (may be up to 60 s earlier)
#SBATCH --requeue                           # allow scontrol requeue from inside the job
#SBATCH --open-mode=append                  # every leg appends to the same log
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source /usr/local/sbin/modules.sh
module load SciPy-bundle/2024.05-gfbf-2024a
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

PD="/lustre/pd03/${SLURM_JOB_ACCOUNT}"
OUT="$PD/runs/$SLURM_JOB_ID"                # same job id on every leg, so this persists
CKPT="$OUT/checkpoint"
LEG="${SLURM_RESTART_COUNT:-0}"
mkdir -p "$OUT" "$CKPT" "$TMPDIR/results" "$TMPDIR/ckpt"
echo "leg $LEG started $(date -Is) on $SLURM_JOB_NODELIST  commit $(git rev-parse --short HEAD 2>/dev/null || echo no-repo)"

save() {
    cp -r "$TMPDIR/results/." "$OUT/"
    if [ -n "$(ls -A "$TMPDIR/ckpt" 2>/dev/null)" ]; then cp -r "$TMPDIR/ckpt/." "$CKPT/"; fi
}
on_usr1() {
    echo "leg $LEG: wall limit near at $(date -Is); saving and requeueing"
    # If train.py checkpoints on SIGUSR1, uncomment the next two lines; a
    # program without a handler is killed by the signal, so the default is to
    # rely on its periodic checkpoints.
    # kill -USR1 "$PID" 2>/dev/null || true
    # wait "$PID" || true
    save
    scontrol requeue "$SLURM_JOB_ID"        # kills this leg at once; the next starts with SLURM_RESTART_COUNT+1
}
trap on_usr1 USR1
trap save EXIT

# Resume from the newest checkpoint on PD if a previous leg left one.
RESUME=()
if [ -e "$CKPT/latest" ]; then RESUME=(--resume "$CKPT/latest"); echo "leg $LEG resumes from $CKPT/latest"; fi

# train.py must write a checkpoint to --ckpt-dir every so often and name the
# newest one "latest"; that is the contract the chain depends on.
python train.py --out "$TMPDIR/results" --ckpt-dir "$TMPDIR/ckpt" --ckpt-every 30m "${RESUME[@]}" &
PID=$!
wait "$PID"
echo "leg $LEG finished the whole job at $(date -Is)"
