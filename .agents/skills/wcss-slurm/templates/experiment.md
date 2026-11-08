# EXP-001: <one sentence saying what this experiment asks>

- **Status:** planned | running | done | abandoned
- **Account:** `YOUR_SERVICE_ACCOUNT`
- **Started:** YYYY-MM-DD
- **Code:** <repo URL and branch, or the path if it is not in git yet>

## Question

What are you actually asking, and what result would answer it? One paragraph.
If you cannot say what outcome would change your mind, the experiment is not
ready to run.

## Method

- Inputs, and where they live (a path under `$PDDIR`, not `$HOME`).
- What the job does.
- Which parameters vary, and over what range.
- What counts as success.

## Budget

Worked out **before** submitting, because the pool is shared and does not refill:

| | |
|---|---|
| cores per job | |
| wall time per job | |
| number of jobs / array tasks | |
| **estimated CPU-hours** | cores × hours × tasks |
| **estimated GPU-hours** | |
| account balance at planning time | from `scripts/discover.sh` |

`scripts/preflight.sh job.sbatch` does this arithmetic and refuses a job the
account cannot afford — which `sbatch --test-only` will not do.

## Runs

One row per submission, filled in at the moment of submission, not afterwards.
`scripts/run-record.sh <jobid>` generates these; `--format tsv` gives a row.

| Job ID | Date | Partition | Cores | Mem | Wall req/used | MaxRSS | CPUh | Exit | Output | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |

> `sacct` keeps per-job records for roughly three months. After that this table
> is the only record that the run happened, what it cost, and which code
> produced it.

## Results

What came out. Link to the output paths under `$PDDIR`. Include the failures:
a job that died at hour 19 of 20 is a finding about sizing, and it cost real
hours from a shared grant.

## What I would change

Right-sizing evidence for the next round: `MaxRSS` against the memory requested,
CPU efficiency, wall time used against the limit. All three come from
`scripts/run-record.sh`.

## Acknowledgement

Any publication or thesis using these results must carry:

> Created using resources provided by Wroclaw Centre for Networking and
> Supercomputing (http://wcss.pl)

The service owner also files an annual report listing research done,
publications and next year's plan. The Runs table above is what makes that
writable.
