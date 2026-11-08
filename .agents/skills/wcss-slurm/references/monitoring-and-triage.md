# Monitoring, accounting and triage

**TL;DR**
- `squeue -u $USER -o "%.10i %.20P %.18j %.10T %.10M %R"`: read the **reason**.
  `QOSGrpCPUMinutesLimit` means the account cannot afford the job and it will
  never start; `Priority` or `Resources` means wait; `RUNNING` with no output
  for minutes means the node is booting (PART-05).
- `run-record.sh <jobid>` prints state, cost, CPU efficiency and memory use
  from `sacct`, since `seff` is not installed (ACT-03) and `MaxRSS` lives only
  on the `.batch` step (ACT-04).
- `sacct` keeps jobs about three months (ACT-01) and refuses wide date ranges
  (ACT-02); `squeue` drops a job five minutes after it ends (ACT-05). Record runs.
- `sacct -j <id> -X -n -P -o SubmitLine` recovers the exact `sbatch` command (ACT-06).
  It drops the original quoting, so a `--wrap` with semicolons reads like
  separate commands; the job's log settles what ran.
- `MaxRSS` on an OOM-killed job equals the request: a floor, not a measurement.
- Before resubmitting anything, `quota -s -f ~`: a nearly full `$HOME` kills the
  next job through its own log file. `run-record.sh` prints a note above 90 %.

Official: [zarządzanie zadaniami](https://man.e-science.pl/pl/kdm/slurm/zarzadzanie-zadaniami),
[FAQ](https://man.e-science.pl/pl/kdm/faq).

## What a triage answer has to contain

Knowing these facts is not the same as putting them in the answer. Four things,
every time:

1. **The state and exit code of each job**, read from `sacct`, not inferred from
   the symptom.
2. **What the job actually asked for**: `SubmitLine`, `Timelimit`, `ReqMem`
   (ACT-06). The user's memory of what they submitted is not evidence.
3. **The replacement request written out in full**, and `--time` in `H:MM:SS` or
   `D-HH:MM:SS` — never a bare number. `-t 10` is a correct way to ask for ten
   minutes and an ambiguous thing to hand someone whose job just died because
   `-t 5` meant five minutes, not five hours. Write `0:10:00` and the mistake
   cannot repeat.
4. **`$HOME` headroom before the resubmit**, and say what it was.
   `run-record.sh` reads it and prints a note only above 90 %, so a silent run
   means it is fine — report that it was checked, rather than leaving the reader
   unsure whether it was. `quota -s -f ~` gets the same number directly.

## Pending reasons

| Reason | Meaning | Action |
|---|---|---|
| `QOSGrpCPUMinutesLimit`, `QOSGrpGRESRunMinutes` | the account cannot afford it; never starts | shrink, top up, or move it (human decision) |
| `Priority`, `Resources` | queued behind other work | wait |
| `Dependency` | waiting on `--dependency` | check the parent did not fail |
| `QOSMaxJobsPerUserLimit`, `AssocMaxJobsLimit` | at the 150 running-job cap | wait, or throttle the array |
| `ReqNodeNotAvail`, `Reserved…` | maintenance reservation | check the announced window |
| `(null)` for weeks | look at the balance | `discover.sh` |

States: `PD` pending, `R` running, `CG` completing, `CD` completed, `F` failed,
`TO` timeout, `OOM` out of memory, `CA` cancelled, `NF` node failure.
`squeue -j <id> --start` estimates a start; `scontrol show job <id>` shows
everything for five minutes after the end.

## After the job

```bash
sacct -j <jobid> -o JobID,JobName%18,State%20,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,TotalCPU
sacct -j <jobid> -X -n -P -o SubmitLine
```

CPU efficiency is `TotalCPU / (Elapsed × AllocCPUS)`, memory headroom
`MaxRSS / ReqMem`; `run-record.sh` computes both and emits a Markdown, YAML or
TSV row for the experiment note. Right-size the next run from them; a job that
used 4 GB of 64 GB paid for cores, not memory, but a job that used 4 of 32
cores paid for 32. Retention was probed in 30-day windows over two years on a
heavily used account: nothing older than about three months (ACT-01).

## Failure modes, most frequent first

1. **`module: command not found`**: no `source /usr/local/sbin/modules.sh`, or a
   script written against the login node where the file does not exist.
2. **Parked forever with `QOSGrpCPUMinutesLimit`**: `cores × hours` exceeds the
   grant; `--test-only` said nothing. `preflight.sh` is the only check.
3. **Out of memory** (FAIL-01): state `OUT_OF_MEMORY`, exit `0:125`, log says
   `Killed` and `Detected 1 oom_kill event`. `MaxRSS` equals the request, so
   raise `--mem` well above it or measure interactively. Default is 4000 MB per core.
4. **Wall-clock kill** (FAIL-02): state `TIMEOUT`, `.batch` step `CANCELLED 0:15`,
   log says `CANCELLED ... DUE TO TIME LIMIT`; 30 s grace, no more. The template's
   exit trap runs in the grace period; for anything that needs more than one
   partition limit, see [long-jobs.md](long-jobs.md).
5. **`Only 'srun' interactive jobs allowed`**: `sbatch` on an interactive partition.
6. **`Disk quota exceeded`**: `$HOME` hit 50 GB or a million files. Write to
   `$TMPDIR`, copy to PD, and keep logs off a nearly full home.
7. **"The results are gone"**: `$TMPDIR` was moved to
   `/lustre/tmp/slurm/finished_jobs/<jobid>/<TYPE>/` for 14 days; look from
   inside a job, `/lustre/tmp` is an empty stub on `ui`.
8. **"It worked on the other cluster"**: AlmaLinux 8 vs 9, different modules. Rebuild.
9. **Rejected at submit** with a site message:

   | Message | Meaning |
   |---|---|
   | `does not meet the minimum required resources for partition 'lem-gpu-*'` | no GPU requested, or `--gpus=N` |
   | `exceeds the maximum allowed resources for partition 'lem-gpu-*'` | over 16 cores or 250 GB per GPU, or no explicit `-c` |
   | `Invalid generic resource (gres) specification` | `--gres=gpu:N` without `hopper` |
   | `Requested node configuration is not available` | more than any node has, or a GPU on a CPU partition |
   | `Required partition not available (inactive or drain)` | a bare alias with `--time` above every variant |
   | `Time limit specification required` | `--time` missing |

   `preflight.sh` prints the explanation next to each.
10. **Died at launch, no log**: `--output` in a missing directory or containing
    a `$variable`; both pass `--test-only` (SUB-14). `preflight.sh` checks.
11. **`RUNNING` and nothing happens**: the node was powered down; about five
    minutes, unbilled (PART-05).

## Cancelling

```bash
scancel <jobid>
scancel -u "$USER" --state=PENDING
scancel <arrayid>_[5-9]
```

Never cancel a job you did not submit. Classify a failure before resubmitting;
a deterministic failure resubmitted spends the grant twice.
