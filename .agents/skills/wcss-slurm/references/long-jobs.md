# Jobs longer than a partition allows

**TL;DR**
- Limits per leg: 3 d on `*-short`, 21 d on `*-cpu-normal`, 7 d on `lem-gpu-normal`.
  Anything longer runs as a chain of legs that checkpoint to PD and continue.
- `--signal=B:USR1@600` delivers `USR1` to the batch shell about ten minutes
  before the limit; it may arrive up to a minute earlier than asked (SUB-15).
- To continue: `scontrol requeue $SLURM_JOB_ID` from inside a job submitted
  with `--requeue` kills the running leg at once and restarts it under the same
  job ID with `SLURM_RESTART_COUNT` incremented; `--open-mode=append` keeps one
  log (SUB-17). Or submit the next leg yourself: `sbatch` works on compute
  nodes, and `--dependency=afterany:$SLURM_JOB_ID` runs it after this one (SUB-16).
- The program must write checkpoints on its own schedule and resume from the
  newest one on PD; the job script only moves files and requeues. Each leg is
  billed separately, and a requeued job goes to the back of the queue.

[templates/job-restartable.sbatch](../templates/job-restartable.sbatch) is the
whole pattern with placeholders.

## The three moving parts

**The signal.** `#SBATCH --signal=B:USR1@600` asks SLURM to send `SIGUSR1` 600 s
before the wall limit. The `B:` sends it to the batch shell only; without it
every `srun` step gets it too. Measured with `@60` on a 3-minute job: the
signal arrived at 107 s, 73 s before the limit, and a trap that saved a file
and called `exit 0` ended the job as `COMPLETED` (SUB-15). SLURM documents the
slack as up to 60 s early, so budget the copy-out for the worse case.

**The trap.** A signal is delivered only while the shell is waiting, so run the
program in the background and `wait` on it:

```bash
trap on_usr1 USR1
python train.py ... &
PID=$!
wait "$PID"
```

`on_usr1` copies the newest checkpoint from `$TMPDIR` to PD, then requeues.
If the program can checkpoint on demand, send it the signal too
(`kill -USR1 "$PID"; wait "$PID"`); a program without a handler dies on it, so
by default rely on its periodic checkpoints.

**The continuation.** Two measured routes:

| Route | Measured | Use when |
|---|---|---|
| `scontrol requeue "$SLURM_JOB_ID"` with `#SBATCH --requeue` | the leg was cancelled "DUE TO JOB REQUEUE" within a second; the job restarted 2 min later with `SLURM_RESTART_COUNT=1` and `sacct` `Restarts=1` (SUB-17) | the same script and shape continue; one job ID, one log with `--open-mode=append` |
| `sbatch --dependency=afterany:$SLURM_JOB_ID next.sbatch` from inside the job | `sbatch` is `/usr/local/sbin/sbatch` on compute nodes; the child started 21 s after the parent ended (SUB-16) | the next leg needs a different shape, partition or script |

`JobRequeue=0` in the site config only switches off automatic requeue on node
failure; an explicit requeue works. Call it as the last thing after the
checkpoint is safely on PD, because the running leg is killed immediately.

## Resuming

At the start of every leg, look for the newest checkpoint on PD and pass it to
the program; `SLURM_RESTART_COUNT` (unset on the first leg) tells you which leg
this is. Keep checkpoints on PD, not `$TMPDIR`: the next leg may land on another
node, and Lem scratch is node-local. `finished_jobs` keeps the previous leg's
scratch 14 days as a fallback ([storage-and-data.md](storage-and-data.md)).

## What it costs

Each leg is billed on its own elapsed time, so a job that checkpoints every
30 minutes and restarts loses at most 30 minutes of work per leg plus the
copy time. A requeued or dependent leg queues like any new job; on a busy
partition that is the real cost, not the CPU-hours. `preflight.sh` costs one
leg; multiply by the number of legs yourself.
