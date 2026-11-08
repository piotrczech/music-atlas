# Accounts, grants and billing

**TL;DR**
- A *service* is one grant: a SLURM account (`-A`), a pool of CPU-hours and
  GPU-hours shared with everyone on it, and a PD directory.
- You pay allocated **cores × wall time**, busy or idle. Memory is free, GPUs
  are metered into their own pool (QOS-08). Pools never refill (QOS-01).
- A job whose `cores × hours` exceeds what is left is accepted, parks in
  `PENDING` with reason `QOSGrpCPUMinutesLimit`, and never starts. `sbatch
  --test-only` does not check this (QOS-06); `preflight.sh` does.
- Moving a parked job to another `-A` spends someone else's grant: the human decides.

Official: [rozliczanie usług](https://man.e-science.pl/pl/kdm/rozliczanie_uslug),
[limity](https://man.e-science.pl/pl/kdm/limity).

## Where the limit lives

Each service has a QOS named after the account carrying `GrpTRESMins` with the
flags `DenyOnLimit,NoDecay` (QOS-01). Limit and usage together, in
TRES-minutes (QOS-02):

```bash
scontrol show assoc_mgr qos=<account> flags=qos | grep GrpTRESMins
# cpu=1800000(1599986): a 30 000 CPU-hour grant with 26 666 spent; gres/gpu=... likewise
```

`service-balance` shows the same numbers as a table and agreed to the minute.
No partition sets `TRESBillingWeights`, so the `billing` TRES equals the CPU
count: a GPU job's record reads `billing=2,cpu=2,gres/gpu=1,mem=8G` (QOS-08).
Be generous with `--mem`, stingy with `-c` and `--time`; `--time` itself is not
billed, elapsed time is, but a long request backfills worse and must fit the
remaining pool.

## The trap `--test-only` does not catch

Measured (QOS-06): 48 cores × 72 h against an account with about 3 334 CPU-hours
left, submitted with `--test-only`, returned a normal start time. The job would
have queued forever. `preflight.sh job.sbatch` does the arithmetic for both
pools and refuses; it also refuses a GPU job on a service that has no GPU pool.

## Per-user and per-account ceilings

| Scope | Limit | Value |
|---|---|---|
| association | running / submitted jobs | 150 / 500 (QOS-03); the manual says 50 and 250 |
| QOS `cpu-bem2` | cores / RAM per user | 1000 / 10 000 GB (QOS-04) |
| QOS `cpu-lem` | cores / RAM per user | 4000 / 60 000 GB |
| QOS `cpu-interactive`, `cpu-lem-interactive` | jobs, cores, RAM | 2, 24, 45 or 300 GB |
| QOS `gpu-interactive` | jobs, cores, GPU, RAM | 1, 16, 1, 250 GB (QOS-05) |
| QOS `gpu` | GPUs | none at QOS level; the per-service `GrpTRES gres/gpu` is the ceiling (QOS-07), the manual's flat "8" is not |

Re-check with `sacctmgr -nP show assoc user=$USER format=Account,MaxJobs,MaxSubmit`.

## Obligations

Publications and theses must carry *"Created using resources provided by
Wroclaw Centre for Networking and Supercomputing (http://wcss.pl)"* (BILL-02).
The service owner files an annual report of work done and publications
(BILL-01); users report to their owner. That report is the practical reason to
keep your own run records, since `sacct` will not.
