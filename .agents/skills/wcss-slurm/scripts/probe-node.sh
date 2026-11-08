#!/usr/bin/env bash
# probe-node.sh - measure the facts that can only be seen from inside a job.
#
# This is the payload behind the claims verify.sh has to report as SKIP.
# Submit it once per partition you care about:
#
#   sbatch -A <account> -p bem2-cpu-short -c 4 --mem=8gb -t 10 \
#          -o probe_%j.txt --wrap="bash scripts/probe-node.sh"
#   sbatch -A <account> -p lem-cpu-short  -c 4 --mem=8gb -t 10 \
#          --gres=storage:local:10g -o probe_%j.txt --wrap="bash scripts/probe-node.sh"
#   sbatch -A <account> -p lem-gpu-short  -c 4 --mem=8gb -t 10 \
#          --gres=gpu:hopper:1 -o probe_%j.txt --wrap="bash scripts/probe-node.sh"
#
# Cost is a few core-minutes per partition. Run preflight.sh first if unsure.
# It reads and reports; it writes only one small file into $TMPDIR.

set +e

echo "=== [N01] node identity ==="
echo "host=$(hostname)  partition=${SLURM_JOB_PARTITION:-n/a}  job=${SLURM_JOB_ID:-n/a}"
cat /etc/redhat-release 2>/dev/null
uname -r

echo
echo "=== [N02] cpu  (claims HW-01, HW-02, HW-03) ==="
lscpu 2>/dev/null | grep -E "^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)):" | sed 's/  */ /g'
echo "allocated_cpus=${SLURM_CPUS_ON_NODE:-?}  cpus_per_task=${SLURM_CPUS_PER_TASK:-?}"

echo
echo "=== [N03] memory ==="
free -g 2>/dev/null | head -2
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} MB   (empty when --mem was not given)"

echo
echo "=== [N04] scratch tiers  (claims STO-05..STO-09) ==="
echo "TMPDIR        = ${TMPDIR:-unset}"
echo "TMPDIR_LUSTRE = ${TMPDIR_LUSTRE:-unset}"
echo "TMPDIR_LOCAL  = ${TMPDIR_LOCAL:-unset}"
echo "TMPDIR_SHM    = ${TMPDIR_SHM:-unset}"
if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ]; then
    echo "-- \$TMPDIR filesystem: $(stat -f -c '%T' "$TMPDIR" 2>&1)"
    df -h "$TMPDIR" 2>&1 | tail -1
    if echo probe > "$TMPDIR/.probe.$$" 2>/dev/null; then
        echo "-- write test: ok"; rm -f "$TMPDIR/.probe.$$"
    else
        echo "-- write test: FAILED"
    fi
else
    echo "-- \$TMPDIR is unset or missing"
fi
echo "-- /tmp:"     ; df -h /tmp     2>&1 | tail -1
echo "-- /dev/shm:" ; df -h /dev/shm 2>&1 | tail -1

echo
echo "=== [N05] module system  (claims MOD-03, MOD-04, MOD-05) ==="
if [ -f /usr/local/sbin/modules.sh ]; then
    echo "modules.sh: present  ($(stat -c '%s bytes, %y' /usr/local/sbin/modules.sh 2>/dev/null | cut -d. -f1))"
    # shellcheck disable=SC1091
    source /usr/local/sbin/modules.sh
    echo "module is now a: $(type -t module 2>/dev/null || echo MISSING)"
    echo "MODULEPATH=$MODULEPATH"
    echo "module count: $(module avail --terse 2>&1 | grep -cvE '^/|^$|:$')"
    echo "-- newest version of some families (exact prefix match):"
    for fam in Python CUDA cuDNN NCCL GCC OpenMPI R Julia MATLAB netCDF ecCodes CDO NCO; do
        v=$(module avail --terse "$fam" 2>&1 | grep -E "^${fam}/" | tail -1)
        printf "   %-10s %s\n" "$fam" "${v:-none}"
    done
    echo "-- every version of the Python stacks (claims PY-05, PY-08):"
    module avail --terse SciPy-bundle PyTorch TensorFlow scikit-learn numba dask 2>&1 | grep -E '^[A-Za-z]' | sed 's/^/   /'
    echo "-- module load in a pipeline is discarded (claim PY-02):"
    module load Python/3.13.5-GCCcore-14.3.0 2>/dev/null | cat >/dev/null
    echo "   pipeline-load python: $(python3 -V 2>&1 | awk '{print $2}')"
    module purge >/dev/null 2>&1
else
    echo "modules.sh: ABSENT on this node"
fi

echo
echo "=== [N06] containers  (claims CON-01, CON-02) ==="
for t in apptainer singularity; do
    printf "%-12s %s\n" "$t" "$(command -v "$t" 2>/dev/null || echo ABSENT)"
done
command -v apptainer >/dev/null 2>&1 && apptainer --version 2>&1
echo "-- prebuilt images:"
for f in /lustre/software-data/container-images/*.sif; do
    [ -e "$f" ] || continue
    if [ -r "$f" ]; then echo "   readable:     $(basename "$f")"
    else                 echo "   NOT readable: $(basename "$f")"; fi
done

echo
echo "=== [N07] gpu  (claim HW-03) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>&1
else
    echo "nvidia-smi ABSENT"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"

echo
echo "=== [N08] outbound network  (claims NET-01, NET-02) ==="
for u in https://pypi.org/simple/ https://github.com https://huggingface.co \
         https://api.open-meteo.com/v1/forecast https://man.e-science.pl; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 12 "$u" 2>/dev/null)
    printf "   %-42s %s\n" "$u" "${code:-TIMEOUT}"
done
echo "   egress address: $(curl -s --max-time 12 https://api.ipify.org 2>/dev/null || echo unavailable)"
echo "   (an address matters if a data provider filters by IP; it differs per cluster)"

echo
echo "=== [N09] TMPDIR retention  (claim STO-10) ==="
echo "Scratch from finished jobs is moved to finished_jobs/<jobid>/<TYPE>/ and kept 14 days."
if [ -d /lustre/tmp/slurm/finished_jobs ]; then
    echo "finished_jobs is present. Recover an earlier job's scratch with:"
    echo "  ls -l /lustre/tmp/slurm/finished_jobs/<jobid>/"
    echo "(from inside a job - /lustre/tmp is an empty stub on the login node)"
else
    echo "finished_jobs not visible from this node."
fi

echo
echo "=== [N11] python: what each interpreter believes about the cores  (claims PY-01, PY-04) ==="
pyprobe() { "$1" - <<'PY' 2>/dev/null || echo "   $1: failed"
import sys, os, multiprocessing as mp
pc = getattr(os, "process_cpu_count", None)
with mp.Pool() as p: n = p._processes
print(f"   python {sys.version.split()[0]:8s} cpu_count={mp.cpu_count():3d} sched_getaffinity={len(os.sched_getaffinity(0))} process_cpu_count={pc() if pc else 'n/a':>3} default Pool()={n}")
PY
}
echo "-- system /usr/bin/python3:"; pyprobe /usr/bin/python3
if [ -f /usr/local/sbin/modules.sh ]; then
    for m in SciPy-bundle/2024.05-gfbf-2024a Python/3.13.5-GCCcore-14.3.0; do
        module purge >/dev/null 2>&1
        if module load "$m" >/dev/null 2>&1; then echo "-- $m:"; pyprobe python; else echo "-- $m: not on this cluster"; fi
    done
    module purge >/dev/null 2>&1
fi

echo
echo "=== [N10] slurm environment ==="
env | grep -E "^SLURM_(JOB_ID|JOB_NAME|JOB_PARTITION|JOB_ACCOUNT|CPUS_PER_TASK|MEM_PER_NODE|NNODES|JOB_NODELIST|SUBMIT_DIR|ARRAY_JOB_ID|ARRAY_TASK_ID|ARRAY_TASK_COUNT)=" | sort
echo "PDDIR=${PDDIR:-unset}"

echo
echo "=== PROBE COMPLETE ==="
