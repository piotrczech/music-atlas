# Submitting jobs

**TL;DR**
- Defaults without flags: 1 core, 4000 MB per core, 1 node (SUB-01). `$SLURM_MEM_PER_NODE`
  is empty unless you passed `--mem`.
- `--time` is mandatory (SUB-02); a bare number is **minutes** (SUB-03); above
  the partition limit is refused at submit (SUB-06).
- `sbatch` stops reading `#SBATCH` at the first command (SUB-12); it does not
  expand `$variables` in directives, and a log path in a missing directory
  passes `--test-only` and fails at launch (SUB-14).
- `sbatch` on `*-interactive` is refused by a Lua filter (SUB-04): use
  `srun --pty` or `sub-interactive`.
- Arrays: `--array=1-200%20`, indices below 1001 (SUB-07), key outputs on `%A_%a`
  (SUB-08). The `%20` is a limit, not a habit: an association runs at most **150
  jobs at once** (QOS-03), and tasks past that pend with
  `QOSMaxJobsPerUserLimit` or `AssocMaxJobsLimit`. Cost is unaffected - the
  array still bills tasks × cores × wall time, so check it against the grant.
  Dependencies work (SUB-09). No automatic requeue (SUB-10).
- Order: `preflight.sh job.sbatch` → `sbatch --parsable job.sbatch` → `squeue`.

Official: [uruchamianie zadań](https://man.e-science.pl/pl/kdm/slurm/uruchamianie-zadan),
[tablice zadań](https://man.e-science.pl/pl/kdm/slurm/tablica-zadan),
[TL;DR](https://man.e-science.pl/pl/kdm/slurm/tldr).

## What a "here is your job script" answer has to contain

Handing over a script is not the whole answer. Four things, every time:

1. **What it will cost**, as `cores × hours` (GPU jobs: both pools, since the
   allocated cores are billed too), checked against what the paying account has
   left — not against `sbatch --test-only`, which reports a start time for a job
   the grant cannot afford (QOS-06).
2. **`$HOME` headroom**, stated. The job's own `.out`/`.err` land in the submit
   directory by default, and a full `$HOME` kills the next job through its own
   log file. `quota -s -f ~` is one command; `discover.sh` reports it too. Say
   the figure, so the reader knows it was looked at rather than skipped.
3. **The `--time` in `H:MM:SS` or `D-HH:MM:SS`**, never a bare number.
4. **That nothing was submitted**, and the exact command the user runs to submit
   it themselves.

## Template

```bash
#!/bin/bash
#SBATCH --job-name=EXP-001-baseline
#SBATCH --account=YOUR_SERVICE_ACCOUNT      # explicit: it decides whose grant pays
#SBATCH --partition=bem2-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=4:00:00                      # mandatory; H:MM:SS or D-HH:MM:SS
#SBATCH --output=%x_%j.out                  # a fixed path; $PDDIR here would be literal
# every #SBATCH line stays above the first command

set -euo pipefail                           # a failed module load must not burn the allocation
source /usr/local/sbin/modules.sh
module load SciPy-bundle/2024.05-gfbf-2024a
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" PYTHONNOUSERSITE=1

PD="/lustre/pd03/${SLURM_JOB_ACCOUNT}"      # the paying service's PD
OUT="$PD/runs/$SLURM_JOB_ID"
copy_out() { mkdir -p "$OUT"; cp -r "$TMPDIR/results/." "$OUT/"; }
trap copy_out EXIT

echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST  account $SLURM_JOB_ACCOUNT  commit $(git rev-parse --short HEAD 2>/dev/null || echo no-repo)"
mkdir -p "$TMPDIR/results"
python train.py --out "$TMPDIR/results"
```

Ready-made: [job.sbatch](../templates/job.sbatch), [job-array.sbatch](../templates/job-array.sbatch),
[job-gpu.sbatch](../templates/job-gpu.sbatch), [job-mpi.sbatch](../templates/job-mpi.sbatch),
and [job-restartable.sbatch](../templates/job-restartable.sbatch) for anything
longer than a partition allows ([long-jobs.md](long-jobs.md)). The first four
run to completion as shipped; `verify-jobs.sh` submits them at minimal size
(TPL-01, TPL-02, TPL-03, TPL-04).
Optional directives in them are `##SBATCH`; `preflight.sh` refuses a copy that
still holds a `YOUR_*` placeholder, since `sbatch` accepts `--mail-user=YOUR_EMAIL`
without a word.

## Errors at submit

```
Time limit specification required, but not provided          # no --time
Requested time limit is invalid (missing or exceeds some limit) # above the partition limit
sbatch: lua: Only `srun` interactive jobs allowed for partition  # sbatch on *-interactive
```

GPU-related and node-shape messages are in
[partitions-and-hardware.md](partitions-and-hardware.md); the full list with
meanings is in [monitoring-and-triage.md](monitoring-and-triage.md).

## Interactive, arrays, chains

```bash
sub-interactive -c 4 -m 12 -t 2
srun -p lem-gpu-interactive --gres=gpu:hopper:1 -c 4 --mem=16gb -t 60 --pty bash
```

Interactive QOS allows 2 sessions (1 on GPU).

Arrays: `MaxArraySize=1001`, `MaxJobCount=6000` (SUB-07). Inside a task
`$SLURM_ARRAY_JOB_ID`, `$SLURM_ARRAY_TASK_ID`, `$SLURM_ARRAY_TASK_COUNT`; the last
task reuses the array's own job ID, so `$SLURM_JOB_ID` is not unique (SUB-08).
The `%20` throttle keeps you inside the 150 running jobs per association.

```bash
J1=$(sbatch --parsable stage.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 compute.sbatch)   # (SUB-09)
```

`JobRequeue=0` (SUB-10): a node failure does not resubmit your job; add
`--requeue` only for idempotent work.

## Flags worth knowing

`--parsable` (just the ID), `--gres=storage:local:100g` (Lem NVMe),
`--gres=gpu:hopper:N`, `--mail-type=END,FAIL --mail-user=...`, `--test-only`
(syntax, partition, wall limit and node shape, **not** the grant, not the log path).
