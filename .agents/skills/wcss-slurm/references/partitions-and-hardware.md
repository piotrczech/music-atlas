# Partitions and hardware

**TL;DR**
- Real partitions: `bem2-cpu-{short,normal,interactive}`, `lem-cpu-{...}`,
  `lem-gpu-{...}`; short = 3 d, normal = 21 d except `lem-gpu-normal` = 7 d,
  interactive = 6 h (PART-01, PART-02). `bem2-cpu`, `lem-cpu`, `lem-gpu` are
  aliases routed by `--time` (SUB-05).
- Bem2 node: 48 cores, 375 GB, AlmaLinux 8. Lem CPU node: 128 cores (256
  threads), 1.5 TB, AlmaLinux 9. Lem GPU node: 64 cores, 1 TB, 4 × H100 (HW-01..04).
- **GPU rules**: `--gres=gpu:hopper:N` only (GPU-05); a GPU partition needs a
  GPU (SUB-11); at most 16 cores and 250 GB per GPU with `-c` explicit (GPU-04).
- Idle nodes power down after 30 min and take about 5 min to resume, unbilled (PART-05).
- More cores, memory or GRES than any node has is refused at submit (SUB-13).

Official: [partycje SLURM](https://man.e-science.pl/pl/kdm/slurm/partycje-slurm).

## Partitions

| Partition | Wall limit | Nodes | Notes |
|---|---|---|---|
| `bem2-cpu-short` | 3 d | 96 | default choice for short CPU work |
| `bem2-cpu-normal` | 21 d | 48 | |
| `lem-cpu-short` | 3 d | 186 | |
| `lem-cpu-normal` | 21 d | 140 | |
| `lem-gpu-short` | 3 d | 74 | |
| `lem-gpu-normal` | **7 d** | 52 | |
| `*-interactive` | 6 h | 1 to 2 | `srun --pty` or `sub-interactive`; `sbatch` is refused (SUB-04) |
| `k8s-build` | 4 h | 2 | container building |

Node counts drift; `sinfo -o "%20P %10l %6D %10T"` is current. Interactive
partitions carry `PriorityJobFactor=1000` against 1 elsewhere (PART-03), which
is why a shell starts fast. `EnforcePartLimits=ANY` (PART-04): a `--time` above
the partition's limit is refused at submit.

**Aliases.** A submit plugin rewrites `-p lem-gpu` to `-short` or `-normal` from
`--time` and prints `[AUTOMATIC PARTITION CHOICE]`. The boundary is exactly
three days inclusive (SUB-05); a `--time` above every variant fails with the
misleading `Required partition not available (inactive or drain)`.

**Power saving** (PART-05). `SuspendTime` 1800 s, `ResumeTimeout` 500 s; `sinfo`
shows powered-down nodes as `idle~`. A job placed on one shows `RUNNING` in
`squeue` while the node boots (5 min 28 s measured on a GPU node) and `sacct`
starts the clock when it is up, so nothing is billed for the wait. `discover.sh`
prints the powered-down count per partition.

## Node hardware

| | Bem2 CPU | Lem CPU | Lem GPU |
|---|---|---|---|
| CPU | Xeon Platinum 8268, 2 × 24 cores | EPYC 9554, 2 × 64 cores, 256 threads | Xeon Platinum 8462Y+, 2 × 32 cores |
| RAM | 375 GB (some nodes 190 GB) | 1510 GB | 1006 GB |
| GPU | none | none | 4 × H100 95 830 MiB, NVLink, driver 595.71, cc 9.0 |
| OS | **AlmaLinux 8.10** | AlmaLinux 9.8 | AlmaLinux 9.8 |

(HW-01, HW-02, HW-03, HW-04.) The manual counts cores, `lscpu` in a job counts
threads (HW-05); SLURM allocates by core. Different libc and module sets
across the two clusters: rebuild compiled code on the target; a venv built with
the module Python does carry over ([python.md](python.md)). `sinfo -o %m` prints
`190464+` for Bem2's mixed memory group, the `+` meaning "at least"; ask per
node with `sinfo -N -o %m` and strip the sign (HW-06). Asking
for more than any node has is refused as `Requested node configuration is not
available` (SUB-13), which also greets a GPU request on a CPU partition.

## GPUs

```bash
#SBATCH -p lem-gpu-short
#SBATCH --gres=gpu:hopper:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
```

The submit filter enforces, all measured with `--test-only`:

- **Only `--gres=gpu:hopper:N`** (GPU-05). `--gpus=N` and `--gpus-per-node` are
  rejected as below the partition minimum; `--gres=gpu:N` is "Invalid generic
  resource".
- **A GPU partition needs a GPU** (SUB-11): "does not meet the minimum required
  resources".
- **16 cores and 250 GB per GPU, `-c` stated** (GPU-04): 17 cores or 251 GB with
  one GPU, 33 cores with two, or no `-c` at all are "exceeds the maximum allowed
  resources". `--mem-per-cpu × cores` counts. Node-local scratch has no
  per-GPU cap; a whole node is 4 GPUs, 64 cores and a little under 1006 GB.

Verified on a two-GPU job: both devices visible, `nvcc -arch=sm_90` compiles
and computes correctly, about 92 GiB free per GPU (GPU-01, GPU-03), and `NV6`
NVLink between them with per-GPU NUMA affinity from `nvidia-smi topo -m`
(GPU-02). `CUDA/12.8`, `12.9.1` and `13.0.0` modules exist; load them in the
current shell, never in a pipeline ([python.md](python.md)).

## Choosing

| Want | Go |
|---|---|
| ≤ 3 days, plain CPU | `bem2-cpu-short`, usually the emptiest |
| lots of RAM per core, fast cores | `lem-cpu-short` |
| more than 3 days | `*-cpu-normal` (21 d) or `lem-gpu-normal` (7 d) |
| a GPU | `lem-gpu-short`, `--gres=gpu:hopper:N`, `-c` ≤ 16 per GPU |
| a shell to test in | `sub-interactive`, never `sbatch` |
| node-local NVMe scratch | Lem, `--gres=storage:local:Ng` |
| scratch shared across nodes | Bem2 Lustre `$TMPDIR` |
