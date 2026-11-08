# Discovering your own account

**TL;DR**
- `discover.sh` prints all of this in one call: services by name, CPU- and
  GPU-hours left, the concurrent GPU ceiling, job-count limits, `$HOME` and PD
  quotas, partitions with powered-down node counts, scheduler defaults, stale
  profile lines, recent jobs.
- Nothing in this skill hardcodes a username, service ID, balance or PD path;
  derive them. Account IDs look like `hpc-<owner>-<epoch>`; only `PD-info`
  prints the human name behind each.

## Who am I, which grants can I spend

```bash
sacctmgr -nP show user "$USER" format=User,DefaultAccount
sacctmgr -nP show user "$USER" format=Account -s        # every account you may use
PD-info                                                 # names, PD paths, quotas
```

The default account is what `sbatch` uses without `-A`; `service-balance` marks
it `(D)`, `set-default-service -l` lists accounts without marking it. Change it
only on the human's say-so: `set-default-service <account>`.

## Hours left

```bash
service-balance                        # friendly table; --check-cpu / --check-gpu
scontrol show assoc_mgr qos=<account> flags=qos | grep -E 'GrpTRES'
```

`GrpTRESMins=cpu=1800000(1599986)` reads *limit 1 800 000 CPU-minutes,
1 599 986 used*; divide by 60. `GrpTRES=...gres/gpu=32(3)` is a different limit:
at most 32 GPUs held at once by everyone on the service, 3 in use now. See
[accounts-and-billing.md](accounts-and-billing.md).

## Limits, storage, partitions, defaults

```bash
sacctmgr -nP show assoc user="$USER" format=Account,MaxJobs,MaxSubmit
quota -s -f ~                                  # space AND file quota of $HOME
echo "$PDDIR"; echo "$PDDIRS"                  # default PD; every PD, colon-separated
sinfo -o "%20P %10l %6D %10T"                  # partitions, wall limits, node states
sinfo -h -N -o "%n|%c|%m|%G" | sort -u -t'|' -k2   # distinct node shapes
scontrol show config | grep -E "DefMemPerCPU|MaxArraySize|MinJobAge|JobRequeue|JobSubmitPlugins|CliFilterPlugins|SuspendTime"
```

That last line explains most surprises: 4000 MB per core by default, arrays
capped at 1001, jobs gone from `squeue` five minutes after they end, no
automatic requeue, `--time` mandatory, `sbatch` refused on interactive
partitions, nodes powered down when idle.

## What is running, what ran

```bash
squeue -u "$USER" -o "%.10i %.20P %.18j %.10T %.10M %R"
sacct -X -u "$USER" -S "$(date -d '-7 days' +%F)" -o JobID,JobName%18,State%20,Elapsed,AllocCPUS
sacct -j <jobid> -X -n -P -o SubmitLine        # the exact sbatch command
```

Keep `sacct` windows narrow; nothing older than about three months survives
([monitoring-and-triage.md](monitoring-and-triage.md)).

## What software exists

Not answerable from the login node. Take an interactive session on the
partition you will use, then `source /usr/local/sbin/modules.sh; module avail`
([software-environment.md](software-environment.md)).
