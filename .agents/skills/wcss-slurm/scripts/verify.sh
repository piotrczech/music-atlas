#!/usr/bin/env bash
# verify.sh - re-check every measured claim in this skill against the live system.
#
#   scripts/verify.sh [ID-prefix]        e.g. scripts/verify.sh GPU
#
# READ-ONLY. Submits no jobs (sbatch is used only with --test-only, which asks
# the scheduler what it would do), writes nothing, changes nothing.
# Run it on the WCSS login node (ui.wcss.pl).
#
# Claims that can only be measured inside a job are reported SKIP with a reason,
# together with the command that would establish them (scripts/probe-node.sh).
#
# Exit: 0 if nothing FAILED, 1 otherwise. NOTE lines never affect the exit code.

set -uo pipefail

PASS=0; FAIL=0; SKIP=0; NOTE=0
FAILED_IDS=()
NOTES=()
ONLY="${1:-}"

c_pass=$'\033[32m'; c_fail=$'\033[31m'; c_skip=$'\033[33m'; c_note=$'\033[36m'; c_off=$'\033[0m'
[ -t 1 ] || { c_pass=""; c_fail=""; c_skip=""; c_note=""; c_off=""; }

# report ID STATUS "detail"
#
# FAIL means THIS REPOSITORY is wrong or out of date -- it is the signal to open
# an issue. NOTE means the claim holds and something about the reader's own
# account is worth their attention; it must not affect the exit code, or a
# perfectly accurate repository would report failure on a full home directory.
report() {
    local id="$1" status="$2" detail="$3"
    case "$status" in
        PASS) PASS=$((PASS+1)); printf "  %s%-4s%s %-12s %s\n" "$c_pass" PASS "$c_off" "$id" "$detail" ;;
        FAIL) FAIL=$((FAIL+1)); FAILED_IDS+=("$id")
              printf "  %s%-4s%s %-12s %s\n" "$c_fail" FAIL "$c_off" "$id" "$detail" ;;
        SKIP) SKIP=$((SKIP+1)); printf "  %s%-4s%s %-12s %s\n" "$c_skip" SKIP "$c_off" "$id" "$detail" ;;
        NOTE) NOTE=$((NOTE+1)); NOTES+=("$id: $detail")
              printf "  %s%-4s%s %-12s %s\n" "$c_note" NOTE "$c_off" "$id" "$detail" ;;
    esac
}

# check ID "expectation" <command...>  - PASS if command exits 0
check() {
    local id="$1" desc="$2"; shift 2
    if [ -n "$ONLY" ] && [[ "$id" != $ONLY* ]]; then return 0; fi
    if "$@" >/dev/null 2>&1; then report "$id" PASS "$desc"
    else report "$id" FAIL "$desc"; fi
}

# checkval ID "expectation" "expected" "actual"
checkval() {
    local id="$1" desc="$2" want="$3" got="$4"
    if [ -n "$ONLY" ] && [[ "$id" != $ONLY* ]]; then return 0; fi
    if [ "$want" = "$got" ]; then report "$id" PASS "$desc"
    else report "$id" FAIL "$desc (expected '$want', got '$got')"; fi
}

skip() {
    local id="$1" desc="$2"
    if [ -n "$ONLY" ] && [[ "$id" != $ONLY* ]]; then return 0; fi
    report "$id" SKIP "$desc"
}

have() { command -v "$1" >/dev/null 2>&1; }

echo "verify.sh - re-checking measured claims (baseline: 2026-09-10, SLURM 26.05.1)"
echo "host: $(hostname 2>/dev/null)   date: $(date -u +%Y-%m-%dT%H:%MZ)"
echo

if ! have sinfo; then
    echo "SLURM client commands not found. Run this on the WCSS login node." >&2
    exit 2
fi

CONF="$(scontrol show config 2>/dev/null)"
conf() { echo "$CONF" | grep -E "^$1 " | head -1 | sed 's/.*= *//' | tr -d ' '; }

# dry ID "expectation" <sbatch options...>  - what sbatch --test-only says; the
# result is matched by the caller. Nothing is ever submitted.
dry() { sbatch --test-only --wrap=true "$@" 2>&1; }
ACCTS="$(sacctmgr -nP show user "$USER" format=Account -s 2>/dev/null | sort -u)"
A1="$(echo "$ACCTS" | head -1)"
DRYACCT=(); [ -n "$A1" ] && DRYACCT=(-A "$A1")

# ---------------------------------------------------------------- access
echo "Access and login node"
checkval ACC-01 "login node is AlmaLinux 9" "9" \
    "$(sed -n 's/.*release \([0-9]*\).*/\1/p' /etc/redhat-release 2>/dev/null | head -1)"
# The login banner is assembled by /etc/motd_{upper,lower}.sh, not /etc/motd,
# which is empty here.
MOTD_FILES=(/etc/motd /etc/motd_upper.sh /etc/motd_lower.sh)
checkval ACC-03 "each user is capped at 1 GiB of RAM on the login node (cgroup memory.max)" "1073741824" \
    "$(cat "/sys/fs/cgroup/user.slice/user-$(id -u).slice/memory.max" 2>/dev/null)"
check    ACC-04 "login banner points hpc-info at the KDM VPN" \
    grep -qiE "hpc-info.*VPN" "${MOTD_FILES[@]}"
check    ACC-07 "set-default-service is a shell function, not a binary" \
    test ! -x /usr/local/bin/set-default-service
# bash sources ~/.bashrc when it is started by sshd for a command (SSH_CLIENT is
# set), and the stock ~/.bashrc pulls in /etc/bashrc, which runs /etc/profile.d.
# So `ssh host 'cmd'` does see PDDIR - the login-shell advice is a belt, not a need.
check    ACC-08 "an ssh command shell (non-login) still gets \$PDDIR through ~/.bashrc -> /etc/bashrc -> profile.d" \
    env -i HOME="$HOME" USER="$USER" PATH=/usr/bin:/bin SSH_CLIENT=x bash -c 'test -n "$PDDIR"'
if ! grep -qE '^\s*\.?\s*(source\s+)?/etc/bashrc' "$HOME/.bashrc" 2>/dev/null; then
    report ACC-08b NOTE "your ~/.bashrc does not source /etc/bashrc; use ssh host 'bash -lc ...' to get \$PDDIR in commands"
fi

# ---------------------------------------------------------------- scheduler config
echo
echo "Scheduler configuration"
checkval SUB-01 "DefMemPerCPU is 4000 MB" "4000" "$(conf DefMemPerCPU)"
check    SUB-02 "require_timelimit plugin is loaded (--time mandatory)" \
    bash -c 'scontrol show config | grep -q "JobSubmitPlugins.*require_timelimit"'
checkval SUB-07 "MaxArraySize is 1001" "1001" "$(conf MaxArraySize)"
checkval SUB-10 "JobRequeue is off" "0" "$(conf JobRequeue)"
check    SUB-04 "a Lua cli_filter is installed (rejects sbatch on *-interactive)" \
    bash -c 'scontrol show config | grep -q "CliFilterPlugins.*lua"'
checkval PART-04 "EnforcePartLimits is ANY" "ANY" "$(conf EnforcePartLimits)"
checkval ACT-05 "MinJobAge is 300 s" "300sec" "$(conf MinJobAge)"
checkval PART-05a "idle nodes are powered down after SuspendTime=1800 s" "1800sec" "$(conf SuspendTime)"
checkval PART-05b "a powered-down node gets ResumeTimeout=500 s to come back" "500sec" "$(conf ResumeTimeout)"
check    ACT-06 "sacct exposes the SubmitLine field (the exact sbatch command)" \
    bash -c 'sacct --helpformat 2>/dev/null | grep -qw SubmitLine'

# ---------------------------------------------------------------- partitions
echo
echo "Partitions and hardware"
PARTS="$(sinfo -h -o "%P" 2>/dev/null | tr -d '*' | sort -u | tr '\n' ' ')"
for p in bem2-cpu-short bem2-cpu-normal bem2-cpu-interactive \
         lem-cpu-short lem-cpu-normal lem-cpu-interactive \
         lem-gpu-short lem-gpu-normal lem-gpu-interactive; do
    check PART-01 "partition $p exists" bash -c "echo '$PARTS' | grep -qw '$p'"
done
# grep -w treats '-' as a word separator, so 'lem-gpu' would match inside
# 'lem-gpu-short'. Compare whole lines instead.
check PART-01b "no partition is literally named lem-gpu / bem2-cpu / lem-cpu" \
    bash -c "! printf '%s\n' $PARTS | grep -qxE 'lem-gpu|bem2-cpu|lem-cpu'"

wall() { scontrol show partition "$1" 2>/dev/null | grep -oE 'MaxTime=[^ ]+' | head -1 | cut -d= -f2; }
checkval PART-02a "bem2-cpu-short wall limit is 3 days"  "3-00:00:00"  "$(wall bem2-cpu-short)"
checkval PART-02b "bem2-cpu-normal wall limit is 21 days" "21-00:00:00" "$(wall bem2-cpu-normal)"
checkval PART-02c "lem-gpu-normal wall limit is 7 days"   "7-00:00:00"  "$(wall lem-gpu-normal)"
checkval PART-02d "lem-gpu-short wall limit is 3 days"    "3-00:00:00"  "$(wall lem-gpu-short)"
checkval PART-02e "interactive wall limit is 6 hours"     "06:00:00"    "$(wall bem2-cpu-interactive)"
checkval PART-03 "interactive partitions carry PriorityJobFactor=1000" "1000" \
    "$(scontrol show partition bem2-cpu-interactive 2>/dev/null | grep -oE 'PriorityJobFactor=[0-9]+' | cut -d= -f2)"

check HW-03a "lem-gpu nodes advertise gpu:hopper GRES" \
    bash -c "sinfo -h -p lem-gpu-short -o '%G' | grep -q 'gpu:hopper'"
check HW-03b "bem2 nodes advertise no GPU GRES" \
    bash -c "! sinfo -h -p bem2-cpu-short -o '%G' | grep -q 'gpu:'"
check HW-06 "sinfo groups Bem2's mixed-memory nodes and prints a '+' suffix; per node (-N) the largest is 384000 MB" \
    bash -c "sinfo -h -p bem2-cpu-short -o '%m' | grep -q '+' && [ \"\$(sinfo -h -N -p bem2-cpu-short -o '%m' | tr -d '+' | sort -n | tail -1)\" -ge 384000 ]"
check STO-06a "lem nodes advertise storage:local GRES" \
    bash -c "sinfo -h -p lem-cpu-short -o '%G' | grep -q 'storage:local'"

# ---------------------------------------------------------------- submit-time rules (all --test-only)
echo
echo "Submit-time rules (sbatch --test-only; nothing is submitted)"
if [ -z "$A1" ]; then
    for id in SUB-05 SUB-06 SUB-11 SUB-12 SUB-13 SUB-14 GPU-04 GPU-05; do skip "$id" "no SLURM account for $USER; cannot dry-run"; done
else
    check SUB-06 "a walltime above the partition limit is refused at submit time" \
        bash -c "$(declare -f dry); ! dry ${DRYACCT[*]} -p bem2-cpu-short -c 1 --mem=1gb -t 5-00:00:00 | grep -q 'to start at'"
    check SUB-05a "the bare alias lem-cpu with --time of exactly 3 days routes to lem-cpu-short" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-cpu -c 1 --mem=1gb -t 3-00:00:00 | grep -q \"selecting partition 'lem-cpu-short'\""
    check SUB-05b "one second more routes to lem-cpu-normal" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-cpu -c 1 --mem=1gb -t 3-00:00:01 | grep -q \"selecting partition 'lem-cpu-normal'\""
    check SUB-05c "a --time above every variant's limit fails as 'Required partition not available'" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p bem2-cpu -c 1 --mem=1gb -t 22-00:00:00 | grep -q 'Required partition not available'"
    check SUB-11 "a job on a GPU partition with no GPU request is rejected" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-gpu-short -c 1 --mem=1gb -t 5 | grep -q 'minimum required resources'"
    TD="$(mktemp)"; printf '#!/bin/bash\n#SBATCH -c 1\n#SBATCH -t 2\n#SBATCH --mem=1gb\ntrue\n#SBATCH -c 4\n' > "$TD"
    check SUB-12 "sbatch ignores #SBATCH lines after the first command (a trailing -c 4 still gives 1 processor)" \
        bash -c "sbatch --test-only ${DRYACCT[*]} -p bem2-cpu-short '$TD' 2>&1 | grep -q 'using 1 processors'"
    rm -f "$TD"
    check SUB-14 "an --output path in a missing directory passes --test-only (the job then fails at launch)" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p bem2-cpu-short -c 1 --mem=1gb -t 1 -o /nonexistent_dir_xyz/%j.out | grep -q 'to start at'"
    check SUB-13 "asking more memory than any node has is refused as 'Requested node configuration is not available'" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p bem2-cpu-short -c 1 --mem=400gb -t 5 | grep -q 'Requested node configuration is not available'"
    check GPU-04a "16 cores per GPU are accepted on lem-gpu-short" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-gpu-short --gres=gpu:hopper:1 -c 16 --mem=8gb -t 5 | grep -q 'to start at'"
    check GPU-04b "17 cores per GPU are rejected as 'exceeds the maximum allowed resources'" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-gpu-short --gres=gpu:hopper:1 -c 17 --mem=8gb -t 5 | grep -q 'exceeds the maximum allowed resources'"
    check GPU-04c "250 GB per GPU are accepted, 251 GB are rejected" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-gpu-short --gres=gpu:hopper:1 -c 4 --mem=250gb -t 5 | grep -q 'to start at' \
                 && dry ${DRYACCT[*]} -p lem-gpu-short --gres=gpu:hopper:1 -c 4 --mem=251gb -t 5 | grep -q 'exceeds the maximum'"
    check GPU-04d "a GPU job without an explicit --cpus-per-task is rejected" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-gpu-short --gres=gpu:hopper:1 --mem=8gb -t 5 | grep -q 'exceeds the maximum'"
    check GPU-05a "--gpus=1 is rejected on lem-gpu-short" \
        bash -c "$(declare -f dry); ! dry ${DRYACCT[*]} -p lem-gpu-short --gpus=1 -c 1 --mem=1gb -t 5 | grep -q 'to start at'"
    check GPU-05b "--gres=gpu:1 without the hopper type is 'Invalid generic resource'" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-gpu-short --gres=gpu:1 -c 1 --mem=1gb -t 5 | grep -q 'Invalid generic resource'"
    check GPU-05c "--gres=gpu:hopper:1 is accepted" \
        bash -c "$(declare -f dry); dry ${DRYACCT[*]} -p lem-gpu-short --gres=gpu:hopper:1 -c 1 --mem=1gb -t 5 | grep -q 'to start at'"
fi

# ---------------------------------------------------------------- accounts
echo
echo "Accounts, grants and limits"
if [ -z "$ACCTS" ]; then
    skip QOS-01 "no SLURM accounts found for $USER"
    skip QOS-02 "no SLURM accounts found for $USER"
    skip QOS-03 "no SLURM accounts found for $USER"
else
    check QOS-01 "per-service QOS carries flags DenyOnLimit,NoDecay" \
        bash -c "sacctmgr -nP show qos '$A1' format=Flags | grep -q 'NoDecay'"
    check QOS-02 "GrpTRESMins exposes limit(used) via assoc_mgr" \
        bash -c "scontrol show assoc_mgr qos='$A1' flags=qos 2>/dev/null | grep -qE 'GrpTRESMins=cpu=[0-9]+\([0-9]+\)'"
    MAXJ="$(sacctmgr -nP show assoc user="$USER" format=MaxJobs 2>/dev/null | grep -v '^$' | sort -u | head -1)"
    checkval QOS-03 "association running-job limit is 150 (docs say 50/250)" "150" "$MAXJ"
fi
check QOS-04 "QOS cpu-bem2 caps per user at 1000 cores" \
    bash -c "sacctmgr -nP show qos cpu-bem2 format=MaxTRESPU | grep -q 'cpu=1000'"
check QOS-05 "QOS gpu-interactive allows 1 job with 1 hopper GPU" \
    bash -c "sacctmgr -nP show qos gpu-interactive format=MaxJobsPU,MaxTRESPerJob | grep -q 'gpu:hopper=1'"
# The manual gives a flat "8 GPU" for lem-gpu; the QOS itself sets no ceiling,
# so the real limit is the per-service GrpTRES.
check QOS-07 "QOS gpu sets no GPU ceiling of its own (docs say 8)" \
    bash -c "test -z \"\$(sacctmgr -nP show qos gpu format=MaxTRESPU,GrpTRES | tr -d '|')\""
# With no TRESBillingWeights on any partition, the billing TRES is the CPU count:
# memory and GPUs do not enter it (GPU-hours are metered separately as gres/gpu).
check QOS-08 "no partition sets TRESBillingWeights, so billing = allocated CPUs" \
    bash -c "! scontrol show partition | grep -q TRESBillingWeights"

# ---------------------------------------------------------------- storage
echo
echo "Storage"
check STO-01 "quota reports both a space and a file limit for \$HOME" \
    bash -c "quota -s -f ~ 2>/dev/null | grep -qi files"
check STO-02 "home snapshots are exposed under ~/.snapshot" test -d "$HOME/.snapshot"
if [ -n "${PDDIR:-}" ]; then
    check STO-03 "\$PDDIR is exported by /etc/profile.d and points into /lustre" \
        bash -c "echo '${PDDIR:-}' | grep -q '^/lustre/' && test -f /etc/profile.d/pddir-exports.sh"
else
    skip STO-03 "\$PDDIR unset in this shell - see ACC-08; run in a login shell"
fi
checkval STO-04 "login-node TMPDIR points into \$HOME" "$HOME/tmp" "${TMPDIR:-unset}"
check STO-11 "/lustre/tmp is an empty stub on the login node" \
    bash -c "test -d /lustre/tmp && ! test -d /lustre/tmp/slurm"

# ---------------------------------------------------------------- modules
echo
echo "Modules and software"
check MOD-01 "module is not a usable command on the login node" \
    bash -c "! command -v module >/dev/null 2>&1"
check MOD-02 "/usr/local/sbin/modules.sh does not exist on the login node" \
    test ! -f /usr/local/sbin/modules.sh
check MOD-06 "the sub-* wrapper directory exists" test -d /usr/local/bin/bem2
check MOD-07 "sub-interactive wrappers are present for each cluster" \
    bash -c "test -x /usr/local/bin/bem2/sub-interactive-lem-gpu"
check CON-01 "apptainer is absent from the login node" \
    bash -c "! command -v apptainer >/dev/null 2>&1"
check CON-02 "the shared container image directory is readable" \
    bash -c "ls /lustre/software-data/container-images/*.sif >/dev/null 2>&1"

# ---------------------------------------------------------------- mpi
echo
echo "MPI"
MPILIST="$(srun --mpi=list 2>&1 | tr -d '\t' | grep -vE 'plugin types' | tr '\n' ' ')"
check MPI-01a "srun offers no pmix plugin (so OpenMPI 5 + srun is broken)" \
    bash -c "! echo '$MPILIST' | grep -qw pmix"
check MPI-01b "srun does offer pmi2" bash -c "echo '$MPILIST' | grep -qw pmi2"
checkval MPI-01c "no site-wide MpiDefault is set" "(null)" "$(conf MpiDefault)"

# ---------------------------------------------------------------- python
echo
echo "Python"
check PY-03a "pip --user site directory is inside \$HOME" \
    bash -c "python3 -c 'import site,sys; sys.exit(0 if site.getusersitepackages().startswith(\"$HOME\") else 1)'"
if [ -d "$HOME/.local/lib" ]; then
    sz="$(du -sm "$HOME/.local" 2>/dev/null | cut -f1)"
    if [ "${sz:-0}" -gt 1024 ]; then
        # shellcheck disable=SC2088  # display text, not a path to expand
        report PY-03b NOTE "\$HOME/.local is ${sz} MB and is on sys.path in every job, shadowing module packages. See references/python.md"
    else
        # shellcheck disable=SC2088
        report PY-03b PASS "\$HOME/.local is ${sz:-0} MB"
    fi
else
    skip PY-03b "no ~/.local yet - nothing shadowing module packages"
fi
for f in "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile"; do
    [ -f "$f" ] || continue
    p="$(grep -E '^\s*(\.|source)\s' "$f" 2>/dev/null | grep -oE '/mnt/lscratch/[^" ]+|/lustre/tmp/[^" ]+' | head -1)"
    if [ -n "$p" ] && [ ! -e "$p" ]; then
        report PY-11b NOTE "${f/#$HOME/~} sources $p, which vanished with its job (the uv installer trap, PY-11); every login prints an error until you remove that line"
    fi
done

# ---------------------------------------------------------------- transfer
echo
echo "Data transfer"
check DT-01a "rsync, rclone, zstd and pigz are on the login node" \
    bash -c "command -v rsync >/dev/null && command -v rclone >/dev/null \
             && command -v zstd >/dev/null && command -v pigz >/dev/null"
check DT-01b "lftp, sshfs and globus-url-copy are absent" \
    bash -c "! command -v lftp >/dev/null 2>&1 && ! command -v sshfs >/dev/null 2>&1 \
             && ! command -v globus-url-copy >/dev/null 2>&1"
check DT-02 "there is one login host and no separate transfer node" \
    bash -c "test \"\$(getent hosts ui.wcss.pl | awk '{print \$1}' | sort -u | wc -l)\" -eq 1"

# ---------------------------------------------------------------- ood
echo
echo "Open OnDemand"
check OOD-01 "ood.e-science.pl redirects to the e-science SSO, not to a key prompt" \
    bash -c "curl -s -o /dev/null -w '%{redirect_url}' --max-time 15 \
             https://ood.e-science.pl/pun/sys/dashboard 2>/dev/null | grep -q 'sso.e-science.pl'"
if [ -d "$HOME/ondemand" ]; then
    check OOD-03 "OOD session artefacts live under \$HOME/ondemand" test -d "$HOME/ondemand/data"
else
    skip OOD-03 "no ~/ondemand yet - start a session at https://ood.e-science.pl first"
fi

# ---------------------------------------------------------------- accounting
echo
echo "Accounting and tooling"
check ACT-03 "seff and reportseff are not installed" \
    bash -c "! command -v seff >/dev/null 2>&1 && ! command -v reportseff >/dev/null 2>&1"
check ACT-02 "sacct refuses an over-wide date range" \
    bash -c "sacct -X -n -u '$USER' -S 2020-01-01 -o JobID 2>&1 | grep -q 'Too wide'"
OLD="$(sacct -X -n -u "$USER" -S "$(date -d '-330 days' +%F)" -E "$(date -d '-300 days' +%F)" -o JobID 2>/dev/null | wc -l)"
checkval ACT-01 "no job records survive ~11 months back" "0" "$OLD"

# ---------------------------------------------------------------- in-job only
echo
echo "Claims measurable only from inside a job (scripts/verify-jobs.sh <account> runs them)"
skip HW-01   "Bem2 node: Xeon 8268 / 375 GB / AlmaLinux 8.10 - run scripts/probe-node.sh on bem2-cpu-short"
skip HW-02   "Lem CPU node: EPYC 9554 / 1510 GB / AlmaLinux 9.8 - probe-node.sh on lem-cpu-short"
skip HW-03c  "Lem GPU node: 8462Y+ / 1006 GB / 4x H100 95830 MiB - probe-node.sh on lem-gpu-short"
skip HW-04   "Bem2 is AlmaLinux 8 and Lem AlmaLinux 9 - compare two probe-node.sh runs"
skip HW-05   "the manual counts cores per node, lscpu counts threads - probe-node.sh [N02]"
skip MOD-03  "modules.sh exists in jobs - probe-node.sh"
skip MOD-04  "module counts differ per cluster (2164/1735/1018) - probe-node.sh"
skip MOD-05  "module contents differ per cluster - probe-node.sh"
skip STO-05  "Bem2 TMPDIR is /lustre/tmp/slurm/\$JOBID - probe-node.sh"
skip STO-06  "Lem TMPDIR is /mnt/lscratch, sized by --gres=storage:local - probe-node.sh"
skip STO-07  "TMPDIR_LUSTRE / _LOCAL / _SHM: the partition's default tiers plus what you request - probe-node.sh [N04]"
skip STO-08  "/tmp is 200 MB on Bem2, 1 GB on Lem - probe-node.sh"
skip STO-09  "/dev/shm is 357 GB / 1.5 TB / 956 GB - probe-node.sh"
skip STO-10  "TMPDIR is kept 14 d under finished_jobs/<id>/<TYPE> - ls it from inside a job"
skip STO-13  "in a job \$PDDIR follows --account (and uses the pd01 alias); on the login node it follows the default service - probe-node.sh [N10]"
skip NET-01  "compute nodes have outbound internet - probe-node.sh"
skip QOS-06  "--test-only passes an unaffordable job - needs an account near its limit"
skip SUB-03  "a bare --time value means minutes - submit -t 5 and read TimelimitRaw"
skip SUB-08  "array %A/%a semantics - submit a 3-task array"
skip SUB-09  "afterok dependencies work - submit a two-job chain"
skip FAIL-01 "OOM gives OUT_OF_MEMORY and exit 0:125, with MaxRSS pinned at the request"
skip FAIL-02 "wall-clock kill gives TIMEOUT and 'DUE TO TIME LIMIT'"
skip ACT-04  "MaxRSS appears only on the .batch step - sacct any finished job"
skip PART-05 "a job placed on a powered-down node waits for the resume - see references/monitoring-and-triage.md"
skip PY-01   "system python3 is 3.6.8 on Bem2 and 3.9.25 on Lem - run 'python3 -V' in a job on each"
skip PY-02   "module load inside a pipeline is discarded (subshell) - compare 'module load X' with 'module load X | cat'"
skip PY-04   "cpu_count() reports 48/128 in a -c 4 job; Pool() follows it on Python <= 3.12 and the affinity on 3.13 - probe-node.sh [N11]"
skip PY-05   "SciPy-bundle versions differ per cluster; only 2024.05-gfbf-2024a is on both"
skip PY-06   "a venv built with the MODULE python on Bem2 imports and computes on Lem"
skip PY-08   "no PyTorch, TensorFlow, scikit-learn, numba or dask modules on the partitions measured"
skip DT-03   "PD write throughput was ~92 MB/s from the login node (dd, 100 MB, oflag=direct)"
skip OOD-02  "an OOD session is a SLURM job billed to a chosen service - read ~/ondemand/.../job_script_options.json"
skip PY-09   "a venv on Lustre is ~1.6x slower than on node scratch, and ~2200 small files against the shared MDS"
skip PY-10   "uv is not installed but works without root; 946 files vs 2199 for the same env"
skip PY-11   "the uv installer appends a source line to every profile it finds (.bash_profile, .profile, .zshrc) unless UV_NO_MODIFY_PATH=1"
skip PY-12   "tar the venv to PD and untar into \$TMPDIR: one sequential file instead of thousands"
skip MPI-02  "OpenMPI 5.0.3 + srun starts singletons: size=1, allreduce wrong, exit 0"
skip MPI-03  "OpenMPI 4.1.6 + srun aborts in MPI_Init"
skip MPI-04  "mpirun works for both OpenMPI generations: size=8 allreduce=8 on 2 nodes"
skip MPI-05  "compute nodes carry InfiniBand (mlx5_0, ib0)"
skip MPI-06  "OpenMPI 5 warns that its PRRTE session dir sits on Lustre; PRTE_TMPDIR is ignored, the MCA parameter prte_tmpdir_base is honoured"
skip GPU-01  "a --gres=gpu:hopper:2 job sees 2 H100s with distinct UUIDs"
skip GPU-02  "nvidia-smi topo -m reports NV6 NVLink between the GPUs, plus per-GPU NUMA affinity"
skip GPU-03  "nvcc -arch=sm_90 compiles and both devices return correct results"
skip SUB-15  "--signal=B:USR1@N reaches the batch shell before the wall limit, up to 60 s early"
skip SUB-16  "sbatch works inside a job; an afterany child runs after its parent"
skip SUB-17  "scontrol requeue from inside a --requeue job restarts it with SLURM_RESTART_COUNT=1"
skip TPL-01  "templates/job.sbatch runs to completion as shipped"
skip TPL-02  "templates/job-array.sbatch runs to completion as shipped"
skip TPL-03  "templates/job-gpu.sbatch runs to completion as shipped"
skip TPL-04  "templates/job-mpi.sbatch runs to completion as shipped"
skip LLM-01  "pip installs a vLLM that knows current architectures; the site container v0.9.0.1 does not"
skip LLM-02  "all-reduce bus bandwidth is 337 GB/s on NVLink and 152 GB/s over InfiniBand"
skip LLM-03  "an 8 GB, 82k-file venv unpacks from PD to node-local NVMe in about 10 s"
skip LLM-04  "a Lem node has no ib0: InfiniBand is ibp92s0 etc on 10.43.x, and the hostname resolves to management ethernet"
skip LLM-05  "a Ray cluster started without --block dies when the srun step ends"
skip LLM-06  "an authenticated HF download runs ~20x faster, and Xet must be disabled for long ones"
skip LLM-07  "the official DeepSeek V4 Pro checkpoint has FP4 experts and needs Blackwell; the FP8 repack is the Hopper path"
skip LLM-08  "DeepSeek-V4-Flash serves on one node: 37.8 GiB per GPU, 103k-token KV cache"
skip LLM-09  "a venv's console scripts carry the absolute shebang of the job that built them"
skip LLM-10  "multi-node vLLM hangs in symmetric-memory rendezvous unless VLLM_ALLREDUCE_USE_SYMM_MEM=0"
skip LLM-11  "vLLM shells out to ninja, so the venv's bin must be on PATH"
skip LLM-12  "expert parallel beats pipeline parallel by ~46x aggregate on the same checkpoint, with correct output at TP 24"
skip LLM-13  "FP8 weight_block_size 128 caps tensor parallelism at 8 for DeepSeek V4 Pro, below what the weights need"
skip LLM-14  "SGLANG_SHARED_EXPERT_TP1=1 makes the server start and emit token salad; pick a TP dividing 24 instead"
skip LLM-15  "32 H100s reach 2250 tokens/s aggregate on V4 Pro, on a server whose text is token salad"
skip LLM-16  "aggregate throughput peaks at 512 concurrent requests and falls back beyond it"
skip LLM-17  "one node with CUDA graphs on serves V4 Flash at 144.5 tokens/s, 10x the eager floor"
skip LLM-18  "the expert-parallel layout serves at full speed and produces token salad; pipeline parallel is correct"
skip LLM-19  "schema-constrained decoding returns valid JSON from a model emitting noise, hiding the failure"
skip LLM-20  "chunked_prefill_size and max_running_requests are divided by dp_size before validation"
skip LLM-21  "deep_gemm asserts on import unless CUDA_HOME resolves; GPU nodes set neither it nor nvcc"
skip LLM-22  "at TP 24 the weights need 95.6% of HBM, so long prompts, not throughput, are the binding limit"
skip LLM-23  "the correct TP 24 shape peaks at 2077 tokens/s at 128 concurrent, half the salad shape's figure"
skip LLM-24  "beyond ~128 concurrent the server dies rather than queues: Triton kernels load lazily into 0.1 GiB free"

echo
echo "Claims about the manual's own text (read the linked page to confirm)"
skip DOC-01  "limity heads a GPU block 'lem-cpu-interactive'; the figures are lem-gpu-interactive's"
skip DOC-02  "SLURM_module shows 'module avail' at a ui prompt, where module does not work"
skip DOC-03  "katalog-domowy and the TMPDIR page disagree on how long \$TMPDIR survives"

# ---------------------------------------------------------------- summary
echo
echo "-------------------------------------------------------------"
printf "  %spassed %d%s   %sfailed %d%s   %sskipped %d%s   %snotes %d%s\n" \
    "$c_pass" "$PASS" "$c_off" "$c_fail" "$FAIL" "$c_off" \
    "$c_skip" "$SKIP" "$c_off" "$c_note" "$NOTE" "$c_off"
if [ "$NOTE" -gt 0 ]; then
    echo
    echo "  Notes are about your account, not about this repository:"
    printf '    %s\n' "${NOTES[@]}"
fi
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "  Failed claims: ${FAILED_IDS[*]}"
    echo "  A failure means this repository is out of date, not that you did"
    echo "  something wrong. Please open an issue quoting the ID, the command"
    echo "  you ran and its output. See CONTRIBUTING.md."
    exit 1
fi
echo "  Skipped claims need a job: scripts/verify-jobs.sh <account> measures them."
exit 0
