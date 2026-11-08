#!/bin/bash
# Multi-node MPI job.
#
# The one rule on this site: launch with `mpirun`, NOT `srun`. There is no pmix
# plugin here, so OpenMPI 5 under srun starts N single-rank jobs that exit 0 with
# wrong answers, and OpenMPI 4 fails at MPI_Init. See references/mpi-and-multinode.md
#
# Check the cost first - you pay for every core on every node for the whole wall
# time:  scripts/preflight.sh templates/job-mpi.sbatch

#SBATCH --job-name=EXP-001-mpi
#SBATCH --account=YOUR_SERVICE_ACCOUNT
#SBATCH --partition=bem2-cpu-short
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48        # whole Bem2 nodes (128 on Lem)
#SBATCH --time=2:00:00              # 2 x 48 x 2 h = 192 CPU-hours
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_EMAIL
# Hybrid MPI + OpenMP instead? Replace --ntasks-per-node above with, e.g.:
##SBATCH --ntasks-per-node=8
##SBATCH --cpus-per-task=6         # 8 x 6 = 48 cores per node
# and launch with:  mpirun --map-by socket:PE="$SLURM_CPUS_PER_TASK" ./hybrid
# (sbatch stops reading #SBATCH lines at the first command, so keep them up here.)

set -euo pipefail

source /usr/local/sbin/modules.sh
module load OpenMPI/5.0.3-GCC-13.3.0    # never inside a pipeline: it would be lost

# OpenMPI 5 keeps its PRRTE session directory under $TMPDIR, which is Lustre on
# Bem2, and warns that this costs performance. Point it at node-local memory.
# (PRTE_TMPDIR is NOT honoured by this version; the MCA parameter is.)
export PRTE_MCA_prte_tmpdir_base="/dev/shm/${USER}-${SLURM_JOB_ID}"
mkdir -p "$PRTE_MCA_prte_tmpdir_base"
trap 'rm -rf "$PRTE_MCA_prte_tmpdir_base"' EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "allocation  $SLURM_NNODES nodes, $SLURM_NTASKS tasks"
echo "nodelist    $SLURM_JOB_NODELIST"
echo "commit      $(git rev-parse --short HEAD 2>/dev/null || echo no-repo)"

# Smoke test: prove the launcher places one rank per task across the nodes
# before spending the allocation. (Not `mpirun bash -c ...`: prterun takes the
# -c for itself.) Then have the program print MPI_Comm_size and compare it
# with $SLURM_NTASKS; if it is 1 you are running singletons - stop and read
# references/mpi-and-multinode.md rather than trusting the results.
echo "expected: $SLURM_NTASKS ranks on $SLURM_NNODES nodes; launched:"
mpirun hostname | sort | uniq -c

mpirun ./my_mpi_program
