# MPI and multi-node jobs

**TL;DR**
- **Launch with `mpirun`, never `srun`.** `srun --mpi=list` offers only `none`,
  `cray_shasta`, `pmi2`; no `pmix`, no `MpiDefault` (MPI-01). OpenMPI 5 under
  `srun` starts N single-rank copies that exit 0 with wrong answers (MPI-02);
  OpenMPI 4 at least aborts in `MPI_Init` (MPI-03). `mpirun` works for both (MPI-04).
- Print `MPI_Comm_size` at rank 0 and compare with `$SLURM_NTASKS` on every run.
- OpenMPI 5 puts its session directory on Lustre and warns; the fix is
  `export PRTE_MCA_prte_tmpdir_base=/dev/shm/...`, not `PRTE_TMPDIR`, which
  it ignores (MPI-06).
- Nodes have InfiniBand (`mlx5_0`, `ib0`); UCX picks it up unaided (MPI-05).
- Cost is every core on every node for the whole wall time: 2 × 48 × 2 h = 192 CPU-hours.

Official: [MPI](https://man.e-science.pl/pl/kdm/slurm/mpi). Measured on
`bem2-cpu-short`, 2 nodes, OpenMPI 4.1.6 and 5.0.3.

## The silent failure

```
No PMIx server was reachable, but a PMI1/2 was detected.
If srun is being used to launch application,  8 singletons will be started.
   >>> size=1 allreduce=1 *** BROKEN ***      (printed 8 times)
```

An 8-rank job became eight 1-rank jobs; a collective that should sum to 8
returned 1; exit code 0; full price paid (MPI-02).

## A job that works

[templates/job-mpi.sbatch](../templates/job-mpi.sbatch): whole nodes
(`--nodes=2 --ntasks-per-node=48`), `module load OpenMPI/5.0.3-GCC-13.3.0` in
the current shell, the `PRTE_MCA_prte_tmpdir_base` export, a placement check
(`mpirun hostname | sort | uniq -c`), then `mpirun ./program`. `mpirun` inherits
the allocation; no `-np`, no hostfile. Do not write `mpirun bash -c '...'`:
`prterun` takes the `-c` for itself and reports "No executable was specified".

Smoke test, compiled with `mpicc -O2 -o smoke smoke.c`:

```c
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    int rank, size, one = 1, total = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Allreduce(&one, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("size=%d allreduce=%d %s\n", size, total,
                          (size == total && size > 1) ? "OK" : "BROKEN");
    MPI_Finalize();
    return 0;
}
```

OpenMPI modules on Bem2: `4.1.4`, `4.1.5`, `4.1.6`, `5.0.3`; the set differs on
Lem. `mpicc`, `mpirun` and friends exist only inside jobs.

## Sizing and hybrid runs

Measure 1, 2 and 4 nodes on a short problem before a long run; idle cores on a
badly scaling job cost exactly as much as busy ones. For MPI + OpenMP, tasks ×
threads must fit the cores and ranks need pinning:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6          # 8 × 6 = 48 cores per node
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mpirun --map-by socket:PE="$SLURM_CPUS_PER_TASK" ./hybrid
```
