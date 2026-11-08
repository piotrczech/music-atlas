#!/usr/bin/env bash
# discover.sh - print what THIS account has: services, grant balances, storage,
# quotas, partitions and the scheduler defaults that surprise people. Nothing
# is hardcoded; everything is read from the system.
#
# READ-ONLY. Submits nothing, writes nothing.
# Run it on the WCSS login node:
#   ssh wcss 'bash -lc "~/wcss-slurm/scripts/discover.sh"'
# (bash -lc is belt and braces: a stock ~/.bashrc already sources
#  /etc/profile.d in an ssh command shell, but yours may not.)

set -uo pipefail

have() { command -v "$1" >/dev/null 2>&1; }
hr()   { printf '%s\n' "----------------------------------------------------------------"; }
strip_ansi() { sed 's/\x1b\[[0-9;]*m//g'; }

if ! have sacctmgr; then
    echo "SLURM client commands not found. Run this on the WCSS login node." >&2
    exit 2
fi

echo "WCSS account overview for ${USER} on $(hostname), $(date -u +%Y-%m-%dT%H:%MZ)"
hr

# ------------------------------------------------------------------ services
DEFAULT_ACCT="$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null | head -1)"
mapfile -t ACCTS < <(sacctmgr -nP show user "$USER" format=Account -s 2>/dev/null | sort -u)

# Human names come from PD-info, which prints "SERVICE NAME:" then "SERVICE ID:"
# blocks. Parsed once here so every table below can show the name next to the ID.
declare -A NAME=() PDPATH=() PDUSED=() PDQUOTA=()
if have PD-info; then
    while IFS=$'\t' read -r id name path used quota; do
        [ -z "$id" ] && continue
        NAME["$id"]="$name"; PDPATH["$id"]="$path"; PDUSED["$id"]="$used"; PDQUOTA["$id"]="$quota"
    done < <(PD-info 2>/dev/null | strip_ansi | awk '
        /SERVICE NAME:/ { sub(/.*SERVICE NAME:[[:space:]]*/, ""); name=$0 }
        /SERVICE ID:/   { sub(/.*SERVICE ID:[[:space:]]*/, "");   id=$0 }
        /QUOTA \(FILES\):/ { sub(/.*QUOTA \(FILES\):[[:space:]]*/, ""); quota=$0 }
        /USED \(FILES\):/  { sub(/.*USED \(FILES\):[[:space:]]*/, "");  used=$0 }
        /PD DIRECTORY:/ { sub(/.*PD DIRECTORY:[[:space:]]*/, ""); path=$0
                          if (id != "") printf "%s\t%s\t%s\t%s\t%s\n", id, name, path, used, quota
                          id=""; name=""; path=""; used=""; quota="" }')
fi
label() { local id="$1" n="${NAME[$1]:-}"; [ -n "$n" ] && printf '%s' "${n:0:38}" || printf '%s' "$id"; }

echo "SERVICES (SLURM accounts; * = default, used when a job omits -A)"
if [ "${#ACCTS[@]}" -eq 0 ]; then
    echo "  none - this account belongs to no service, so it cannot submit jobs."
else
    for a in "${ACCTS[@]}"; do
        mark="   "; [ "$a" = "$DEFAULT_ACCT" ] && mark=" * "
        printf "%s%-40s %s\n" "$mark" "$a" "${NAME[$a]:-}"
    done
fi
hr

# ------------------------------------------------------------------ balances
echo "GRANT BALANCE  (hours; source: the per-service QOS GrpTRESMins, limit(used))"
printf "  %-38s %-5s %10s %10s %10s\n" "service" "pool" "granted" "used" "left"
for a in "${ACCTS[@]}"; do
    line="$(scontrol show assoc_mgr "qos=$a" flags=qos 2>/dev/null | grep -m1 -oE 'GrpTRESMins=.*')"
    [ -z "$line" ] && { printf "  %-38s %s\n" "$(label "$a")" "(no QOS limits found)"; continue; }
    shown=0
    for res in cpu gres/gpu; do
        # The first TRES follows "GrpTRESMins=", the rest follow a comma, so the
        # separator may be either. "gres/gpu=" cannot match "gres/gpu:hopper=".
        pair="$(echo "$line" | grep -oE "[=,]${res}=[0-9]+\([0-9]+\)" | head -1 | sed 's/^[=,]//')"
        [ -z "$pair" ] && continue
        lim="${pair#*=}"; lim="${lim%%(*}"
        use="${pair#*\(}"; use="${use%)*}"
        printf "  %-38s %-5s %10.0f %10.1f %10.1f\n" "$( [ "$shown" -eq 0 ] && label "$a" )" "${res#gres/}" \
               "$(awk -v l="$lim" 'BEGIN{printf "%.0f", l/60}')" \
               "$(awk -v u="$use" 'BEGIN{printf "%.1f", u/60}')" \
               "$(awk -v l="$lim" -v u="$use" 'BEGIN{printf "%.1f", (l-u)/60}')"
        shown=1
    done
    echo "$line" | grep -q 'gres/gpu=' || printf "  %-38s %-5s %10s\n" "" "gpu" "(no GPU pool: this service cannot run GPU jobs)"
    # GrpTRES is the concurrent ceiling, not the hours: gres/gpu=32(3) means at
    # most 32 GPUs held at once by everyone on the service, 3 in use right now.
    cap="$(scontrol show assoc_mgr "qos=$a" flags=qos 2>/dev/null | grep -m1 -oE 'GrpTRES=[^ ]*' | grep -oE '[=,]gres/gpu=[0-9]+\([0-9]+\)' | head -1 | sed 's/^[=,]gres\/gpu=//')"
    if [ -n "$cap" ]; then
        printf "  %-38s %-5s %10s\n" "$(label "$a")" "cap" "at most ${cap%%(*} GPUs at once, $(echo "$cap" | grep -oE '\([0-9]+\)' | tr -d '()') in use now"
    elif echo "$line" | grep -q 'gres/gpu='; then
        printf "  %-38s %-5s %10s\n" "$(label "$a")" "cap" "no concurrent-GPU ceiling on this service"
    fi
done
echo
echo "  Pools do not refill (QOS flag NoDecay) and are shared with everyone on"
echo "  the service. A job needing more hours than remain is accepted and then"
echo "  parks in PENDING with reason QOSGrpCPUMinutesLimit, forever; only the"
echo "  arithmetic in scripts/preflight.sh catches that, sbatch --test-only does not."
hr

# ------------------------------------------------------------------ job limits
echo "JOB-COUNT LIMITS PER SERVICE"
sacctmgr -nP show assoc user="$USER" format=Account,MaxJobs,MaxSubmit 2>/dev/null \
  | awk -F'|' 'NF>=3 && $1!="" {printf "  %-40s running<=%-6s submitted<=%s\n",$1,($2==""?"-":$2),($3==""?"-":$3)}' \
  | sort -u
hr

# ------------------------------------------------------------------ storage
echo "STORAGE"
if have quota; then
    q="$(quota -s -f ~ 2>/dev/null | tail -1)"
    used="$(awk '{print $1}' <<<"$q")"; lim="$(awk '{print $3}' <<<"$q")"
    fused="$(awk '{print (NF==6)?$4:$5}' <<<"$q")"; flim="$(awk '{print (NF==6)?$6:$7}' <<<"$q")"
    printf "  \$HOME  %s of %s used, %s of %s files  (50 GB and 1M files; weekly snapshots in ~/.snapshot)\n" "$used" "$lim" "$fused" "$flim"
    if [ -d "$HOME/.local/lib" ]; then
        printf "         ~/.local is %s - on sys.path in every job unless PYTHONNOUSERSITE=1 is set\n" "$(du -sh "$HOME/.local" 2>/dev/null | cut -f1)"
    fi
else
    echo "  \$HOME quota: 'quota' not available"
fi
echo
echo "  PD (project storage, per service; not snapshotted, not backed up):"
if [ "${#PDPATH[@]}" -gt 0 ]; then
    printf "  %-38s %-22s %-24s %s\n" "service" "space used/quota" "files used/quota" "path"
    for a in "${ACCTS[@]}"; do
        [ -n "${PDPATH[$a]:-}" ] || continue
        u="${PDUSED[$a]}"; qv="${PDQUOTA[$a]}"
        printf "  %-38s %-22s %-24s %s\n" "$(label "$a")" \
            "${u%% (*} / ${qv%% (*}" "$(echo "$u" | grep -oE '\([0-9]+\)' | tr -d '()') / $(echo "$qv" | grep -oE '\([0-9]+\)' | tr -d '()')" \
            "${PDPATH[$a]}"
    done
elif [ -n "${PDDIRS:-}" ]; then
    echo "$PDDIRS" | tr ':' '\n' | sed 's/^/    /'
else
    echo "    \$PDDIRS is unset and PD-info gave nothing - re-run as: ssh <host> 'bash -lc \"...discover.sh\"'"
fi
echo "  \$PDDIR on this login node = ${PDDIR:-unset}  (the DEFAULT service's PD; inside a job it follows -A)"
echo "  Login-node \$TMPDIR = ${TMPDIR:-unset}  <- inside \$HOME; never use it for scratch"
hr

# ------------------------------------------------------------------ partitions
echo "PARTITIONS  (nodes: total, and how many are powered down right now)"
printf "  %-22s %-12s %6s %8s  %s\n" "partition" "wall limit" "nodes" "pwr-down" "GRES"
sinfo -h -o "%P|%l|%D|%T|%G" 2>/dev/null \
  | awk -F'|' '{gsub(/\*/,"",$1); n[$1]+=$3; if($4 ~ /~$/) pd[$1]+=$3; if(!($1 in l)){l[$1]=$2}; if($5!="(null)"){g[$1]=$5}}
       END{for(p in n) printf "  %-22s %-12s %6d %8d  %s\n", p, l[p], n[p], pd[p]+0, (p in g? g[p] : "-")}' \
  | sort
echo
echo "  bem2-cpu, lem-cpu and lem-gpu are not partitions but accepted as -p aliases;"
echo "  a submit plugin picks -short (<= 3 days) or -normal from your --time."
echo "  Powered-down nodes resume on demand when a job lands on them (a few minutes)."
echo "  GPU jobs: --gres=gpu:hopper:N is the only accepted form (--gpus=N is refused);"
echo "  at most 16 cores and 250 GB per GPU, and --cpus-per-task must be stated."
hr

# ------------------------------------------------------------------ scheduler
echo "SCHEDULER DEFAULTS THAT SURPRISE PEOPLE"
scontrol show config 2>/dev/null \
  | grep -E "^(DefMemPerCPU|MaxArraySize|MaxJobCount|MinJobAge|JobRequeue|EnforcePartLimits|JobSubmitPlugins|CliFilterPlugins|SuspendTime|ResumeTimeout)" \
  | sed 's/^/  /'
echo
echo "  require_timelimit in JobSubmitPlugins  -> --time is MANDATORY, and a bare number is MINUTES"
echo "  cli_filter/lua                         -> sbatch is refused on *-interactive; GPU caps enforced"
echo "  DefMemPerCPU                           -> memory default is per CORE, not per job"
echo "  JobRequeue=0                           -> failed jobs are NOT resubmitted"
hr

# ------------------------------------------------------------------ dotfiles
# A tool installed inside a job (uv, conda installers, ...) may have written a
# `source /mnt/lscratch/...` line into a profile. The path dies with the job and
# every later login prints an error. Worth knowing; nothing is changed here.
found=0
for f in "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile" "$HOME/.zshrc"; do
    [ -f "$f" ] || continue
    while IFS= read -r l; do
        p="$(echo "$l" | grep -oE '(/mnt/lscratch|/lustre/tmp|/tmp)/[^" ]+' | head -1)"
        [ -n "$p" ] && [ ! -e "$p" ] && { [ "$found" -eq 0 ] && echo "DOTFILES"; found=1
            echo "  ${f/#$HOME/~} sources $p, which no longer exists (left by a job-scratch install)"; }
    done < <(grep -E '^\s*(\.|source)\s' "$f" 2>/dev/null)
done
[ "$found" -eq 1 ] && hr

# ------------------------------------------------------------------ activity
echo "RECENT ACTIVITY (sacct keeps roughly three months)"
sacct -X -n -u "$USER" -S "$(date -d '-14 days' +%F 2>/dev/null || date -v-14d +%F)" \
      -o JobID%12,JobName%18,Partition%20,State%16,Elapsed 2>/dev/null | sed 's/^/  /' | tail -15
echo
squeue -h -u "$USER" -o "  QUEUED %.10i %.20P %.18j %.10T %R" 2>/dev/null
hr
echo "Next: scripts/preflight.sh <job.sbatch> before submitting a new job shape."
