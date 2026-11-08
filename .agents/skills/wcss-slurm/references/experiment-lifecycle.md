# Running a whole experiment

**TL;DR**: `sacct` forgets jobs after about three months, so anything not
written down at submission time is unreconstructible, and the annual service
report asks exactly those questions. The loop below exists for that; the
scripts do the bookkeeping.

1. **Frame it.** One-sentence question, the account to bill (a project's own
   service, never a colleague's by default), the scale as cores × hours × runs,
   and where results land under PD. [templates/experiment.md](../templates/experiment.md)
   is the shape.
2. **Prototype interactively.** `sub-interactive -c 4 -m 12 -t 2`, then
   `source /usr/local/sbin/modules.sh; module avail Python`. Interactive jobs
   start fast; batch debugging costs an allocation per attempt. Confirm on the
   cluster you will use, the module sets differ.
3. **Size from evidence.** Run once small; `run-record.sh <id>` gives CPU
   efficiency and memory headroom. Elapsed × margin is the next `--time`; cores
   are what you pay for, memory is free.
4. **Preflight every new shape.** `preflight.sh job.sbatch`: cost against both
   pools, GPU rules, placeholders, log directory, `$HOME` headroom, and
   `--test-only`, with the site's messages explained. Re-run when the shape
   changes, not for every submission.
5. **Submit and capture the ID.** `JOBID=$(sbatch --parsable job.sbatch)` and
   write it into the note in the same breath. Chain stages with
   `--dependency=afterok:$J1`; use one array with a `%20` throttle for a sweep.
6. **Monitor by exception.** `squeue -u $USER -o "%.10i %.20P %.18j %.10T %.10M %R"`,
   reason column; `--mail-type=END,FAIL` for anything longer than a coffee.
7. **Collect.** The templates copy `$TMPDIR/results` to
   `/lustre/pd03/$SLURM_JOB_ACCOUNT/runs/$SLURM_JOB_ID` on every exit, including
   the 30 s after a wall-clock kill. Forgot? `finished_jobs` keeps scratch 14 days.
8. **Record.** `run-record.sh <jobid> >> experiment.md`: account, partition,
   allocation, elapsed, `MaxRSS`, CPU-hours billed, exit code, submit line, and
   the commit if the job echoed it (`git rev-parse --short HEAD` in the script;
   a commit read afterwards from the working directory is labelled as such).
9. **Close the loop.** Outcome, cost, what to change; keep the WCSS
   acknowledgement with the results.

One line per run, from the moment of submission:

| Job ID | Date | Account | Partition | Cores | Mem | Wall req/used | MaxRSS | CPUh | Exit | Output | Commit |
|---|---|---|---|---|---|---|---|---|---|---|---|

Everything in that row is available for five minutes in `squeue`, three months
in `sacct`, and forever only if you wrote it down.
