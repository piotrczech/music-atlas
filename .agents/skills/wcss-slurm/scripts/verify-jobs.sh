#!/usr/bin/env bash
# verify-jobs.sh - measure the claims verify.sh has to skip, by running jobs.
#
#   scripts/verify-jobs.sh <account> [--keep] [--timeout MINUTES]
#
# verify.sh is read-only and therefore cannot see inside a compute node. This
# script submits a dozen small jobs on <account> (a few CPU-minutes and about
# one GPU-minute in total), waits for them, and grades their output with the
# same PASS/FAIL/SKIP table. Run it from the repository root on the login node.
#
# It writes only under $HOME/wcss-verify-<timestamp>/ (job scripts and logs)
# and $PD/wcss-verify-<timestamp>/ (a venv archive), plus the runs/<jobid>/
# directories the templates create on PD; all are removed at the end unless
# --keep is given. Exit: 0 if nothing FAILED, 1 otherwise, 2 on usage or a
# job that never finished.

set -uo pipefail

ACCT="${1:-}"; shift || true
[ -z "$ACCT" ] && { echo "usage: $0 <account> [--keep] [--timeout MINUTES]" >&2; exit 2; }
KEEP=0; TIMEOUT_MIN=45
while [ $# -gt 0 ]; do case "$1" in --keep) KEEP=1 ;; --timeout) TIMEOUT_MIN="${2:-45}"; shift ;; *) echo "unknown option $1" >&2; exit 2 ;; esac; shift; done
command -v sbatch >/dev/null 2>&1 || { echo "sbatch not found; run on the WCSS login node." >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sacctmgr -nP show user "$USER" format=Account -s 2>/dev/null | grep -qx "$ACCT" || { echo "$USER cannot use account $ACCT" >&2; exit 2; }
PD="/lustre/pd03/$ACCT"; [ -w "$PD" ] || { echo "PD $PD is not writable" >&2; exit 2; }

TS="$(date +%Y%m%d-%H%M%S)"
WD="$HOME/wcss-verify-$TS"; PDW="$PD/wcss-verify-$TS"
mkdir -p "$WD" "$PDW"

PASS=0; FAIL=0; SKIP=0; NOTE=0; FAILED_IDS=(); NOTES=()
c_pass=$'\033[32m'; c_fail=$'\033[31m'; c_skip=$'\033[33m'; c_note=$'\033[36m'; c_off=$'\033[0m'
[ -t 1 ] || { c_pass=""; c_fail=""; c_skip=""; c_note=""; c_off=""; }
report() {
    local id="$1" status="$2" detail="$3"
    case "$status" in
        PASS) PASS=$((PASS+1)); printf "  %s%-4s%s %-12s %s\n" "$c_pass" PASS "$c_off" "$id" "$detail" ;;
        FAIL) FAIL=$((FAIL+1)); FAILED_IDS+=("$id"); printf "  %s%-4s%s %-12s %s\n" "$c_fail" FAIL "$c_off" "$id" "$detail" ;;
        SKIP) SKIP=$((SKIP+1)); printf "  %s%-4s%s %-12s %s\n" "$c_skip" SKIP "$c_off" "$id" "$detail" ;;
        NOTE) NOTE=$((NOTE+1)); NOTES+=("$id: $detail"); printf "  %s%-4s%s %-12s %s\n" "$c_note" NOTE "$c_off" "$id" "$detail" ;;
    esac
}
check()    { local id="$1" desc="$2"; shift 2; if "$@" >/dev/null 2>&1; then report "$id" PASS "$desc"; else report "$id" FAIL "$desc"; fi; }
checkval() { local id="$1" desc="$2" want="$3" got="$4"; if [ "$want" = "$got" ]; then report "$id" PASS "$desc"; else report "$id" FAIL "$desc (expected '$want', got '$got')"; fi; }
skip()     { report "$1" SKIP "$2"; }
# grepf FILE REGEX : the file exists and contains the pattern
grepf() { [ -f "$1" ] && grep -qE "$2" "$1"; }
# field FILE KEY : value after "KEY=" or "KEY " on the first matching line
field() { grep -m1 -oE "$2[= ][^ ]*" "$1" 2>/dev/null | sed -E "s/^$2[= ]//"; }
export -f grepf field   # the checks run inside bash -c

echo "verify-jobs.sh - measuring in-job claims on account $ACCT (work dir $WD)"
echo "host: $(hostname)   date: $(date -u +%Y-%m-%dT%H:%MZ)"
echo

# ------------------------------------------------------------------ payloads
cat > "$WD/payload.py" <<'PY'
import os, sys, multiprocessing, subprocess
print("python     ", sys.version.split()[0], sys.executable)
print("affinity   ", len(os.sched_getaffinity(0)), " cpu_count", multiprocessing.cpu_count(), " SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_PER_TASK"))
try:
    import numpy; print("numpy      ", numpy.__version__, numpy.__file__)
except Exception as e:
    print("numpy      import failed:", e)
with multiprocessing.Pool() as p: print("default Pool workers:", p._processes)
t = os.environ["TMPDIR"]; os.makedirs(t + "/results", exist_ok=True)
open(t + "/results/hello.txt", "w").write("ok " + os.environ.get("SLURM_JOB_ID", "?") + "\n")
print("TMPDIR     ", t)
print("PDDIR      ", os.environ.get("PDDIR"), " job account", os.environ.get("SLURM_JOB_ACCOUNT"), " SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID"))
PY
cat > "$WD/smoke.c" <<'C'
#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    int rank, size, one = 1, total = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Allreduce(&one, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("size=%d allreduce=%d %s\n", size, total, (size == total && size > 1) ? "OK" : "BROKEN");
    MPI_Finalize();
    return 0;
}
C
cat > "$WD/vec.cu" <<'CU'
#include <cstdio>
__global__ void add(const float *a, const float *b, float *c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }
int main() {
    int n = 1 << 20, devs = 0; cudaGetDeviceCount(&devs);
    for (int d = 0; d < devs; d++) {
        cudaSetDevice(d); cudaDeviceProp p; cudaGetDeviceProperties(&p, d);
        float *a, *b, *c; cudaMallocManaged(&a, n * 4); cudaMallocManaged(&b, n * 4); cudaMallocManaged(&c, n * 4);
        for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }
        add<<<(n + 255) / 256, 256>>>(a, b, c, n); cudaDeviceSynchronize();
        double s = 0; for (int i = 0; i < n; i++) s += c[i];
        size_t fr, tot; cudaMemGetInfo(&fr, &tot);
        printf("device %d (%s, sm_%d%d): %s, checksum %s, %.1f GiB free of %.1f\n", d, p.name, p.major, p.minor,
               cudaGetErrorString(cudaGetLastError()), (s == 3.0 * n) ? "correct" : "WRONG", fr / 1073741824.0, tot / 1073741824.0);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }
    return 0;
}
CU
# extra work for the GPU probe: NVLink topology and a CUDA kernel on each device
cat > "$WD/gpu-extra.sh" <<'SH'
echo "=== [V-GPU] uuids ==="; nvidia-smi -L
echo "=== [V-GPU] topo ==="; nvidia-smi topo -m 2>&1 | head -6
echo "=== [V-GPU] cuda ==="
source /usr/local/sbin/modules.sh
module load CUDA/12.8 2>/dev/null || module load CUDA 2>/dev/null
if command -v nvcc >/dev/null; then nvcc -arch=sm_90 -o "$TMPDIR/vec" "$WD/vec.cu" && "$TMPDIR/vec"; else echo "nvcc ABSENT"; fi
SH
sed -i "s|\$WD|$WD|g" "$WD/gpu-extra.sh"
# everything the MPI claims need, run inside the MPI template
cat > "$WD/mpitest.sh" <<'SH'
echo "=== [V-MPI] ib ==="; ls /sys/class/infiniband 2>/dev/null | tr '\n' ' '; echo; ip -o link show 2>/dev/null | grep -oE 'ib[0-9]+' | head -1
echo "=== [V-MPI] ompi5 mpirun ==="
mpicc -O2 -o "$TMPDIR/smoke5" "$WD/smoke.c" && mpirun "$TMPDIR/smoke5"
echo "=== [V-MPI] ompi5 srun pmi2 ==="; timeout 120 srun --mpi=pmi2 "$TMPDIR/smoke5" 2>&1 | grep -E 'size=|singletons' | sort | uniq -c; echo "srun exit=${PIPESTATUS[0]}"
echo "=== [V-MPI] prte tmpdir ==="
# the template already exports PRTE_MCA_prte_tmpdir_base; drop it for the baseline runs
env -u PRTE_MCA_prte_tmpdir_base mpirun "$TMPDIR/smoke5" 2>&1 | grep -q "shared file system" && echo "default: warning present" || echo "default: no warning"
D="/dev/shm/$USER-$SLURM_JOB_ID"; mkdir -p "$D"
PRTE_MCA_prte_tmpdir_base="$D" mpirun "$TMPDIR/smoke5" 2>&1 | grep -q "shared file system" && echo "mca: warning present" || echo "mca: warning gone"
env -u PRTE_MCA_prte_tmpdir_base PRTE_TMPDIR="$D" mpirun "$TMPDIR/smoke5" 2>&1 | grep -q "shared file system" && echo "PRTE_TMPDIR: warning present" || echo "PRTE_TMPDIR: warning gone"
rm -rf "$D"
echo "=== [V-MPI] ompi4 srun ==="
module purge >/dev/null 2>&1; module load OpenMPI/4.1.6-GCC-13.2.0 2>/dev/null || { echo "ompi4 module absent"; exit 0; }
mpicc -O2 -o "$TMPDIR/smoke4" "$WD/smoke.c" && { timeout 120 srun --mpi=pmi2 "$TMPDIR/smoke4" 2>&1 | grep -m1 -E 'MPI_Init|error occurred' || echo "ompi4 srun: no error text"; }
echo "=== [V-MPI] ompi4 mpirun ==="; mpirun "$TMPDIR/smoke4"
SH
sed -i "s|\$WD|$WD|g" "$WD/mpitest.sh"

# ------------------------------------------------------------------ job scripts
# the templates, as shipped, with placeholders filled and sizes shrunk
tpl() { # tpl NAME SRC sed-expressions...
    local name="$1" src="$2"; shift 2
    sed -e "s/YOUR_SERVICE_ACCOUNT/$ACCT/" -e "s|^#SBATCH --output=.*|#SBATCH --output=$WD/${name}_%j.out|" \
        -e "s|^#SBATCH --error=.*|#SBATCH --error=$WD/${name}_%j.err|" -e "s/--job-name=EXP-001-[a-z]*/--job-name=vj-$name/" "$@" \
        "$ROOT/templates/$src" > "$WD/$name.sbatch"
    # only active lines count: directives and commands, not ##SBATCH or # comments
    if grep -E '^[[:space:]]*#SBATCH |^[^#]' "$WD/$name.sbatch" | grep -qE 'YOUR_|train\.py|my_mpi_program'; then
        echo "template $src no longer matches the substitutions in $0" >&2; exit 2; fi
}
tpl single job.sbatch -e 's/--cpus-per-task=16/--cpus-per-task=2/' -e 's/--mem=64gb/--mem=4gb/' -e 's/--time=4:00:00/--time=5/' \
    -e "s|^python train.py.*|python \"$WD/payload.py\"|"
tpl array job-array.sbatch -e 's/--cpus-per-task=8/--cpus-per-task=1/' -e 's/--mem=32gb/--mem=2gb/' -e 's/--time=2:00:00/--time=5/' \
    -e 's/--array=1-200%20/--array=1-3/' -e 's|^#SBATCH --output=.*|#SBATCH --output='"$WD"'/array_%A_%a.out|' \
    -e 's|^#SBATCH --error=.*|#SBATCH --error='"$WD"'/array_%A_%a.err|' \
    -e 's|^PARAMS=.*|PARAMS="p${SLURM_ARRAY_TASK_ID}"|' -e '/envs\/myproject\/bin\/activate/d' -e "s|^python train.py.*|python \"$WD/payload.py\"|"
tpl gpu job-gpu.sbatch -e 's/--cpus-per-task=8/--cpus-per-task=2/' -e 's/--mem=64gb/--mem=8gb/' -e 's/--time=12:00:00/--time=5/' \
    -e '/envs\/myproject\/bin\/activate/d' -e "s|^python train.py.*|python \"$WD/payload.py\"|"
tpl mpi job-mpi.sbatch -e 's/--ntasks-per-node=48.*/--ntasks-per-node=2/' -e 's/--time=2:00:00/--time=5/' \
    -e "s|^mpirun ./my_mpi_program.*|bash \"$WD/mpitest.sh\"|"

cat > "$WD/oom.sbatch" <<SB
#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH -c 1
#SBATCH --mem=1gb
#SBATCH -t 5
#SBATCH -J vj-oom
#SBATCH -o $WD/oom_%j.out
python3 -c "x=bytearray(2*1024**3); print(len(x))"
echo never-reached
SB
cat > "$WD/timeout.sbatch" <<SB
#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH -c 1
#SBATCH --mem=1gb
#SBATCH -t 1
#SBATCH -J vj-timeout
#SBATCH -o $WD/timeout_%j.out
echo start; sleep 150; echo never-reached
SB
cat > "$WD/signal.sbatch" <<SB
#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH -c 1
#SBATCH --mem=1gb
#SBATCH -t 3
#SBATCH --signal=B:USR1@60
#SBATCH -J vj-signal
#SBATCH -o $WD/signal_%j.out
T0=\$(date +%s); echo "start \$T0"
echo "sbatch-in-job: \$(command -v sbatch || echo ABSENT)"
trap 'echo "USR1 at +\$(( \$(date +%s) - T0 ))s"; exit 0' USR1
CHILD=\$(sbatch --parsable --dependency=afterany:\$SLURM_JOB_ID -A $ACCT -p bem2-cpu-short -c 1 --mem=1gb -t 2 -J vj-chain-child -o "$WD/chain_child_%j.out" --wrap="echo child ran after \\\$(date +%s)" 2>&1)
echo "child \$CHILD"
sleep 600 & wait
SB
cat > "$WD/requeue.sbatch" <<SB
#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH -c 1
#SBATCH --mem=1gb
#SBATCH -t 2
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -J vj-requeue
#SBATCH -o $WD/requeue_%j.out
echo "run restart_count=\${SLURM_RESTART_COUNT:-unset}"
if [ "\${SLURM_RESTART_COUNT:-0}" -ge 1 ]; then echo "second run done"; exit 0; fi
scontrol requeue "\$SLURM_JOB_ID" && echo "requeue accepted" || echo "requeue REFUSED"
sleep 45
SB
cat > "$WD/venv-build.sbatch" <<SB
#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH -c 2
#SBATCH --mem=4gb
#SBATCH -t 10
#SBATCH -J vj-venv-build
#SBATCH -o $WD/venv_build_%j.out
set -e
source /usr/local/sbin/modules.sh
module load Python/3.13.5-GCCcore-14.3.0
echo "build on \$(cat /etc/redhat-release) python \$(python -V)"
python -m venv "\$TMPDIR/venv" && "\$TMPDIR/venv/bin/pip" -q install numpy
"\$TMPDIR/venv/bin/python" -c "import numpy,sys; print('numpy', numpy.__version__, 'abi', sys.implementation.cache_tag)"
echo "venv files: \$(find "\$TMPDIR/venv" -type f | wc -l)"
T=\$(date +%s.%N); tar -C "\$TMPDIR" -cf "$PDW/venv.tar" venv; echo "tar to PD: \$(echo "\$(date +%s.%N) - \$T" | bc) s, \$(du -m "$PDW/venv.tar" | cut -f1) MB"
SB
cat > "$WD/venv-use.sbatch" <<SB
#!/bin/bash
#SBATCH -p lem-cpu-short
#SBATCH -c 2
#SBATCH --mem=4gb
#SBATCH -t 5
#SBATCH -J vj-venv-use
#SBATCH -o $WD/venv_use_%j.out
set -e
source /usr/local/sbin/modules.sh
module load Python/3.13.5-GCCcore-14.3.0      # the venv's interpreter is the module's; load it on the target too
echo "use on \$(cat /etc/redhat-release)"
T=\$(date +%s.%N); tar -xf "$PDW/venv.tar" -C "\$TMPDIR"; echo "untar from PD: \$(echo "\$(date +%s.%N) - \$T" | bc) s"
source "\$TMPDIR/venv/bin/activate"
python -c "import numpy; a=numpy.ones((200,200)); print('OK numpy', numpy.__version__, 'matmul', float((a@a).sum()))"
SB
cat > "$WD/retention.sbatch" <<SB
#!/bin/bash
#SBATCH -p bem2-cpu-short
#SBATCH -c 1
#SBATCH --mem=1gb
#SBATCH -t 2
#SBATCH -J vj-retention
#SBATCH -o $WD/retention_%j.out
echo "finished_jobs of \$PARENT: \$(ls /lustre/tmp/slurm/finished_jobs/\$PARENT/ 2>&1 | tr '\n' ' ')"
cat /lustre/tmp/slurm/finished_jobs/\$PARENT/LUSTRE/results/hello.txt 2>&1
SB

# ------------------------------------------------------------------ submit
declare -A J
sub() { local name="$1"; shift; local id; id="$(sbatch --parsable -A "$ACCT" "$@" 2>&1)" || { echo "submit $name failed: $id" >&2; exit 2; }; J[$name]="${id%%;*}"; printf "  submitted %-12s %s\n" "$name" "${J[$name]}"; }
echo "Submitting"
sub probe-bem2   -p bem2-cpu-short -c 4 --mem=8gb -t 10 -J vj-probe-bem2 -o "$WD/probe_bem2_%j.out" --wrap="bash $ROOT/scripts/probe-node.sh"
sub probe-lemcpu -p lem-cpu-short  -c 4 --mem=8gb -t 10 --gres=storage:local:10g -J vj-probe-lemcpu -o "$WD/probe_lemcpu_%j.out" --wrap="bash $ROOT/scripts/probe-node.sh"
sub probe-lemgpu -p lem-gpu-short  -c 4 --mem=8gb -t 10 --gres=gpu:hopper:2 -J vj-probe-lemgpu -o "$WD/probe_lemgpu_%j.out" --wrap="bash $ROOT/scripts/probe-node.sh; bash $WD/gpu-extra.sh"
sub single   "$WD/single.sbatch"
sub array    "$WD/array.sbatch"
sub gpu      "$WD/gpu.sbatch"
sub mpi      "$WD/mpi.sbatch"
sub oom      "$WD/oom.sbatch"
sub timeout  "$WD/timeout.sbatch"
sub signal   "$WD/signal.sbatch"
sub requeue  "$WD/requeue.sbatch"
sub venv-build "$WD/venv-build.sbatch"
sub venv-use --dependency=afterok:"${J[venv-build]}" "$WD/venv-use.sbatch"
sub retention --dependency=afterok:"${J[single]}" --export=ALL,PARENT="${J[single]}" "$WD/retention.sbatch"

# power-up timing needs a powered-down GPU node right now; measured opportunistically
PDN="$(sinfo -h -p lem-gpu-short -o '%T %n' | awk '$1=="idle~"{print $2}' | head -1)"
if [ -n "$PDN" ]; then sub powerup -p lem-gpu-short -w "$PDN" --gres=gpu:hopper:1 -c 1 --mem=2gb -t 3 -J vj-powerup -o "$WD/powerup_%j.out" --wrap="uptime"; fi

# ------------------------------------------------------------------ login-node measurements meanwhile
echo
echo "Measured from the login node"
# QOS-06: a job the account cannot afford passes --test-only. Needs a shape
# within partition limits whose cores x hours exceeds what is left.
LEFT_H="$(scontrol show assoc_mgr "qos=$ACCT" flags=qos 2>/dev/null | grep -oE 'GrpTRESMins=cpu=[0-9]+\([0-9]+\)' | head -1 | awk -F'[=()]' '{printf "%d", ($3-$4)/60}')"
if [ -n "$LEFT_H" ] && [ "$LEFT_H" -lt 500000 ]; then
    CORES=$(( LEFT_H / 504 + 2 ))     # 21 days = 504 h on bem2-cpu-normal
    check QOS-06 "a job of ${CORES} cores x 21 d (> ${LEFT_H} CPU-h left) passes sbatch --test-only anyway" \
        bash -c "sbatch --test-only -A '$ACCT' -p bem2-cpu-normal -c $CORES --mem=1gb -t 21-00:00:00 --wrap=true 2>&1 | grep -q 'to start at'"
else
    skip QOS-06 "account has ${LEFT_H:-?} CPU-h left; no shape within partition limits exceeds it"
fi
# DT-03: write throughput to PD from the login node
MBS="$(dd if=/dev/zero of="$PDW/dd.tmp" bs=1M count=100 oflag=direct 2>&1 | grep -oE '[0-9.]+ MB/s' | cut -d' ' -f1)"; rm -f "$PDW/dd.tmp"
check DT-03 "100 MB direct write to PD from the login node ran at ${MBS:-?} MB/s (claim: tens of MB/s, ~92 measured)" \
    awk -v m="${MBS:-0}" 'BEGIN{exit !(m > 20)}'

# ------------------------------------------------------------------ wait
echo
echo "Waiting for ${#J[@]} jobs (timeout ${TIMEOUT_MIN} min); GPU jobs may wait ~5 min for a node to power up"
ids="$(printf '%s,' "${J[@]}")"; ids="${ids%,}"
deadline=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
while :; do
    left="$(squeue -h -j "$ids" -o '%i' 2>/dev/null | wc -l)"
    [ "$left" -eq 0 ] && break
    [ "$(date +%s)" -gt "$deadline" ] && { echo "timeout: $left job(s) still queued: $(squeue -h -j "$ids" -o '%i:%T:%r' | tr '\n' ' ')" >&2; echo "cancel them with: scancel $ids" >&2; exit 2; }
    sleep 20
done
sleep 10   # let sacct catch up
o() { ls "$WD/${1}_${J[$2]}"*.out 2>/dev/null | head -1; }   # o PREFIX NAME -> output file
st() { sacct -j "${J[$1]}" -X -n -P -o State 2>/dev/null | head -1; }
export WD; export -f o st; for k in "${!J[@]}"; do export "J_${k//-/_}=${J[$k]}"; done
# bash -c cannot see the J array; st and o look it up through the exported copies
st() { local v="J_${1//-/_}"; sacct -j "${!v}" -X -n -P -o State 2>/dev/null | head -1; }
o()  { local v="J_${2//-/_}"; ls "$WD/${1}_${!v}"*.out 2>/dev/null | head -1; }
export -f o st

# ------------------------------------------------------------------ grade: hardware and OS
echo
echo "Hardware and OS (probe-node.sh on three partitions)"
PB="$(o probe_bem2 probe-bem2)"; PL="$(o probe_lemcpu probe-lemcpu)"; PG="$(o probe_lemgpu probe-lemgpu)"
check HW-01  "Bem2 node: Xeon Platinum 8268, 48 CPUs, ~375 GB, AlmaLinux 8.10" bash -c "grepf '$PB' '8268' && grepf '$PB' '^CPU\(s\): 48' && grepf '$PB' 'AlmaLinux release 8.10' && grepf '$PB' '^Mem: +37[0-9] '"
check HW-02  "Lem CPU node: EPYC 9554, 256 threads, ~1510 GB, AlmaLinux 9" bash -c "grepf '$PL' '9554' && grepf '$PL' '^CPU\(s\): 256' && grepf '$PL' 'AlmaLinux release 9' && grepf '$PL' '^Mem: +15[0-9][0-9] '"
check HW-03c "Lem GPU node: Xeon 8462Y+, 64 CPUs, ~1006 GB, H100 95830 MiB" bash -c "grepf '$PG' '8462Y' && grepf '$PG' '^CPU\(s\): 64' && grepf '$PG' 'H100, 95830 MiB'"
check HW-04  "Bem2 is AlmaLinux 8 and Lem is AlmaLinux 9" bash -c "grepf '$PB' 'release 8\.' && grepf '$PL' 'release 9\.' && grepf '$PG' 'release 9\.'"
check HW-05  "lscpu counts threads on Lem (256) where the manual counts cores (128)" bash -c "grepf '$PL' '^CPU\(s\): 256' && grepf '$PL' 'Core\(s\) per socket: 64'"
check MOD-03 "/usr/local/sbin/modules.sh exists in jobs and defines module as a function" bash -c "grepf '$PB' 'modules.sh: present' && grepf '$PB' 'module is now a: function'"
mb="$(field "$PB" 'module count:')"; ml="$(field "$PL" 'module count:')"; mg="$(field "$PG" 'module count:')"
check MOD-04 "module counts differ per cluster (bem2 $mb, lem-cpu $ml, lem-gpu $mg)" bash -c "[ -n '$mb' ] && [ '$mb' != '$ml' ] && [ '$ml' != '$mg' ]"
check MOD-05 "module contents differ: ecCodes on Bem2, none on Lem" bash -c "grepf '$PB' '^ +ecCodes +ecCodes/' && grepf '$PL' '^ +ecCodes +none'"
check STO-05 "Bem2 TMPDIR is /lustre/tmp/slurm/<jobid> on Lustre" bash -c "grepf '$PB' 'TMPDIR += /lustre/tmp/slurm/${J[probe-bem2]}' && grepf '$PB' 'filesystem: lustre'"
check STO-06 "Lem TMPDIR is /mnt/lscratch/slurm/<jobid> on ZFS, sized by storage:local" bash -c "grepf '$PL' 'TMPDIR += /mnt/lscratch/slurm/${J[probe-lemcpu]}' && grepf '$PL' 'filesystem: zfs'"
check STO-07 "scratch variables: Bem2 LUSTRE only; Lem CPU with storage:local LOCAL; Lem GPU both" \
    bash -c "grepf '$PB' 'TMPDIR_LUSTRE = /lustre' && grepf '$PB' 'TMPDIR_LOCAL += unset' && grepf '$PL' 'TMPDIR_LOCAL += /mnt/lscratch' && grepf '$PG' 'TMPDIR_LUSTRE = /lustre' && grepf '$PG' 'TMPDIR_LOCAL += /mnt/lscratch'"
tb="$(grep -A1 -- '-- /tmp:' "$PB" | tail -1 | awk '{print $2}')"; tl="$(grep -A1 -- '-- /tmp:' "$PL" | tail -1 | awk '{print $2}')"
check STO-08 "/tmp is 200 MB on Bem2 ($tb) and 1 GB on Lem ($tl)" bash -c "[ '$tb' = '200M' ] && [ '$tl' = '1000M' -o '$tl' = '1.0G' ]"
sb="$(grep -A1 -- '-- /dev/shm:' "$PB" | tail -1 | awk '{print $2}')"; sl="$(grep -A1 -- '-- /dev/shm:' "$PL" | tail -1 | awk '{print $2}')"; sg="$(grep -A1 -- '-- /dev/shm:' "$PG" | tail -1 | awk '{print $2}')"
check STO-09 "/dev/shm is 357 GB / 1.5 TB / 956 GB ($sb / $sl / $sg)" bash -c "[ '$sb' = '357G' ] && [ '$sl' = '1.5T' ] && [ '$sg' = '956G' ]"
check NET-01 "compute nodes reach pypi.org, github.com and huggingface.co (HTTP 200)" bash -c "grepf '$PB' 'pypi.org/simple/ +200' && grepf '$PL' 'github.com +200' && grepf '$PG' 'huggingface.co +200'"

echo
echo "Python (probe [N11] and the single-job template)"
check PY-01 "system python3 is 3.6.8 on Bem2 and 3.9.25 on Lem" bash -c "grep -A1 'system /usr/bin/python3' '$PB' | grep -q 'python 3.6.8' && grep -A1 'system /usr/bin/python3' '$PL' | grep -q 'python 3.9.25'"
check PY-02 "module load inside a pipeline is discarded" bash -c "grepf '$PB' 'pipeline-load python: 3\.[69]\.' "
check PY-04 "cpu_count() reports the node; Pool() follows it on 3.12 and the affinity on 3.13" \
    bash -c "grep -A1 'SciPy-bundle' '$PB' | grep -q 'cpu_count= 48 .*default Pool()=48' && grep -A1 'Python/3.13' '$PB' | grep -q 'default Pool()=4' && grep -A1 'SciPy-bundle' '$PL' | grep -q 'cpu_count=128 .*default Pool()=128'"
check PY-05 "SciPy-bundle/2024.05-gfbf-2024a is on both clusters; the other versions differ" \
    bash -c "grepf '$PB' 'SciPy-bundle/2024.05-gfbf-2024a' && grepf '$PL' 'SciPy-bundle/2024.05-gfbf-2024a' && [ \"\$(grep -c 'SciPy-bundle/' '$PB')\" != \"\$(grep -c 'SciPy-bundle/' '$PL')\" ]"
check PY-08 "no PyTorch, TensorFlow, scikit-learn, numba or dask modules" bash -c "! grep -E '^(PyTorch|TensorFlow|scikit-learn|numba|dask)/' '$PB' '$PL' '$PG' | grep -q ."
VB="$(o venv_build venv-build)"; VU="$(o venv_use venv-use)"
check PY-06 "a venv built with the module Python on Bem2 imports and computes on Lem, with the same module loaded there" bash -c "grepf '$VB' 'release 8' && grepf '$VU' 'release 9' && grepf '$VU' 'OK numpy .* matmul'"
check PY-12 "tar the venv to PD and untar into TMPDIR works in seconds" bash -c "grepf '$VB' 'tar to PD: [0-9.]+ s' && grepf '$VU' 'untar from PD: [0-9.]+ s'"
report PY-09 NOTE "venv files on disk: $(field "$VB" 'venv files:' ) (the Lustre small-file argument; the speed ratio is not re-measured here)"
skip PY-10 "uv: installing it modifies the account's profiles; measured by hand on 2026-09-10"
skip PY-11 "uv: see PY-10"

echo
echo "Submission semantics (templates, OOM, timeout, dependency, retention)"
SO="$(o single single)"
check SUB-03 "a bare --time=5 means five minutes (TimelimitRaw=5)" bash -c "[ \"\$(sacct -j ${J[single]} -X -n -P -o TimelimitRaw)\" = 5 ]"
checkval TPL-01 "templates/job.sbatch runs to completion as shipped" "COMPLETED" "$(st single)"
check STO-13 "inside a job PDDIR follows --account (pd01 alias) and the template's results land in that service's PD" \
    bash -c "grepf '$SO' 'PDDIR +/lustre/pd0[13]/$ACCT' && test -f '$PD/runs/${J[single]}/hello.txt'"
check ACT-04 "MaxRSS appears only on the .batch step" bash -c "[ -z \"\$(sacct -j ${J[single]} -X -n -P -o MaxRSS)\" ] && [ -n \"\$(sacct -j ${J[single]}.batch -n -P -o MaxRSS)\" ]"
checkval TPL-02 "templates/job-array.sbatch: all three tasks complete" "COMPLETED COMPLETED COMPLETED" "$(sacct -j "${J[array]}" -X -n -P -o State | sort | tr '\n' ' ' | sed 's/ $//')"
A3="$(ls "$WD"/array_"${J[array]}"_3.out 2>/dev/null)"
check SUB-08 "array tasks get %A_%a outputs; the last task reuses the array's own job id" bash -c "test -f '$A3' && grepf '$A3' 'SLURM_JOB_ID ${J[array]}\$'"
checkval TPL-03 "templates/job-gpu.sbatch runs to completion as shipped" "COMPLETED" "$(st gpu)"
checkval TPL-04 "templates/job-mpi.sbatch runs to completion as shipped" "COMPLETED" "$(st mpi)"
check FAIL-01 "OOM gives OUT_OF_MEMORY, exit 0:125, oom_kill in the log, MaxRSS pinned at the request" \
    bash -c "[ \"\$(sacct -j ${J[oom]} -X -n -P -o State,ExitCode)\" = 'OUT_OF_MEMORY|0:125' ] && grepf '$(o oom oom)' 'oom_kill event'"
check FAIL-02 "wall-clock kill gives TIMEOUT, .batch CANCELLED 0:15, 'DUE TO TIME LIMIT' in the log" \
    bash -c "[ \"\$(sacct -j ${J[timeout]} -X -n -P -o State)\" = TIMEOUT ] && [ \"\$(sacct -j ${J[timeout]}.batch -n -P -o State,ExitCode)\" = 'CANCELLED|0:15' ] && grepf '$(o timeout timeout)' 'DUE TO TIME LIMIT'"
RO="$(o retention retention)"
check SUB-09 "an afterok dependency runs after its parent completes" bash -c "[ \"\$(st retention)\" = COMPLETED ]"
check STO-10 "a finished job's TMPDIR is readable under finished_jobs/<jobid>/LUSTRE" bash -c "grepf '$RO' 'finished_jobs of ${J[single]}: .*LUSTRE' && grepf '$RO' '^ok ${J[single]}'"

echo
echo "Long jobs (signal, self-submission, requeue)"
SG="$(o signal signal)"
check SUB-15 "--signal=B:USR1@60 reaches the batch shell about 60 s before the limit and the trap can exit 0" \
    bash -c "grepf '$SG' 'USR1 at \+([6-9][0-9]|1[0-2][0-9])s' && [ \"\$(st signal)\" = COMPLETED ]"
check SUB-16 "sbatch works inside a job; a child submitted with afterany runs after the parent" \
    bash -c "grepf '$SG' 'sbatch-in-job: /' && ls '$WD'/chain_child_*.out >/dev/null 2>&1 && grep -q 'child ran after' '$WD'/chain_child_*.out"
RQ="$(o requeue requeue)"
check SUB-17 "scontrol requeue from inside a --requeue job restarts it with SLURM_RESTART_COUNT=1" \
    bash -c "grepf '$RQ' 'requeue accepted' && grepf '$RQ' 'restart_count=1' && grepf '$RQ' 'second run done'"

echo
echo "MPI (2 nodes x 2 tasks, inside the MPI template)"
MO="$(o mpi mpi)"
check MPI-02 "OpenMPI 5 under srun starts singletons: size=1, exit 0" bash -c "grep -A3 'ompi5 srun pmi2' '$MO' | grep -q 'size=1 allreduce=1 BROKEN' && grep -A4 'ompi5 srun pmi2' '$MO' | grep -q 'srun exit=0'"
check MPI-03 "OpenMPI 4 under srun aborts in MPI_Init" bash -c "grep -A2 'ompi4 srun' '$MO' | grep -qE 'MPI_Init|error occurred'"
check MPI-04 "mpirun works for both generations: size=4 allreduce=4" bash -c "grep -A2 'ompi5 mpirun' '$MO' | grep -q 'size=4 allreduce=4 OK' && grep -A2 'ompi4 mpirun' '$MO' | grep -q 'size=4 allreduce=4 OK'"
check MPI-05 "compute nodes carry InfiniBand (mlx5_0, ib0)" bash -c "grep -A2 'V-MPI. ib' '$MO' | grep -q 'mlx5_0' && grep -A2 'V-MPI. ib' '$MO' | grep -q 'ib0'"
check MPI-06 "the PRRTE session-dir warning is silenced by prte_tmpdir_base, not by PRTE_TMPDIR" bash -c "grepf '$MO' 'default: warning present' && grepf '$MO' 'mca: warning gone' && grepf '$MO' 'PRTE_TMPDIR: warning present'"

echo
echo "GPU (two H100s)"
check GPU-01 "a --gres=gpu:hopper:2 job sees two H100s with distinct UUIDs" bash -c "[ \"\$(grep -c 'UUID: GPU-' '$PG')\" -eq 2 ] && grepf '$PG' 'SLURM_GPUS_ON_NODE=2'"
check GPU-02 "nvidia-smi topo -m reports NVLink (NV) between the GPUs" bash -c "grep -A5 'V-GPU. topo' '$PG' | grep -qE 'GPU0.*NV[0-9]|GPU1.*NV[0-9]'"
check GPU-03 "nvcc -arch=sm_90 compiles and both devices compute correctly" bash -c "[ \"\$(grep -c 'checksum correct' '$PG')\" -eq 2 ]"
if [ -n "${J[powerup]:-}" ]; then
    S="$(sacct -j "${J[powerup]}" -X -n -P -o Submit,Start | head -1)"; d=$(( $(date -d "${S#*|}" +%s) - $(date -d "${S%|*}" +%s) ))
    check PART-05 "a job placed on a powered-down node waited ${d}s for it to resume, unbilled" bash -c "[ $d -gt 60 ] && [ \"\$(sacct -j ${J[powerup]} -X -n -P -o ElapsedRaw)\" -lt 30 ]"
else
    skip PART-05 "no powered-down GPU node at submit time; resume timing not re-measured"
fi
skip OOD-02 "needs an Open OnDemand session started in a browser"
for id in DOC-01 DOC-02 DOC-03; do skip "$id" "a statement about the manual's text; read the linked page"; done

# ------------------------------------------------------------------ cost and cleanup
echo
CPUS="$(sacct -j "$ids" -X -n -P -o AllocCPUS,ElapsedRaw 2>/dev/null | awk -F'|' '{s+=$1*$2} END{printf "%.1f", s/60}')"
echo "  billed: about ${CPUS:-?} CPU-minutes on $ACCT across ${#J[@]} jobs"
if [ "$KEEP" -eq 0 ]; then
    rm -rf "$PDW"; for n in single array gpu; do rm -rf "$PD/runs/${J[$n]}"; done; rm -rf "$WD"
    echo "  cleaned up $WD and $PDW (use --keep to inspect the outputs)"
else
    echo "  outputs kept in $WD and $PDW"
fi
echo "-------------------------------------------------------------"
printf "  %spassed %d%s   %sfailed %d%s   %sskipped %d%s   %snotes %d%s\n" "$c_pass" "$PASS" "$c_off" "$c_fail" "$FAIL" "$c_off" "$c_skip" "$SKIP" "$c_off" "$c_note" "$NOTE" "$c_off"
[ "$NOTE" -gt 0 ] && printf '    %s\n' "${NOTES[@]}"
if [ "$FAIL" -gt 0 ]; then echo "  Failed claims: ${FAILED_IDS[*]}"; echo "  A failure means this repository is out of date; open an issue with the ID and the output (re-run with --keep)."; exit 1; fi
exit 0
