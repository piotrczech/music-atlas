#!/usr/bin/env bash
# preflight.sh - check a job script before you submit it.
#
#   scripts/preflight.sh job.sbatch [extra sbatch args...]
#
# Runs on the WCSS login node and never submits. It does the checks that are
# free, in the order that matters:
#   1. cores x wall time (x array tasks) against the grant's remaining CPU-hours,
#      and GPUs x wall time against its GPU-hours -- the check that
#      `sbatch --test-only` does NOT do
#   2. the site's GPU rules: --gres=gpu:hopper:N is the only accepted form, a
#      GPU partition needs a GPU and an explicit -c, at most 16 cores and
#      250 GB per GPU
#   3. leftover placeholders such as YOUR_SERVICE_ACCOUNT or YOUR_EMAIL, which
#      sbatch accepts silently
#   4. $HOME space and inode headroom
#   5. sbatch --test-only, for syntax, partition, wall-limit and node-shape
#      errors, with the site's error messages explained
#   6. a plain statement of which account will be billed
#
# Exit 0 = looks submittable, 1 = a blocking problem, 2 = usage error.

set -uo pipefail

SCRIPT="${1:-}"
[ -z "$SCRIPT" ] && { echo "usage: $0 <job.sbatch> [extra sbatch args...]" >&2; exit 2; }
[ -r "$SCRIPT" ] || { echo "preflight: cannot read '$SCRIPT'" >&2; exit 2; }
shift
command -v sbatch >/dev/null 2>&1 || { echo "preflight: sbatch not found; run this on the WCSS login node." >&2; exit 2; }

c_bad=$'\033[31m'; c_warn=$'\033[33m'; c_ok=$'\033[32m'; c_off=$'\033[0m'
[ -t 1 ] || { c_bad=""; c_warn=""; c_ok=""; c_off=""; }
PROBLEMS=0
bad()  { printf "  %sBLOCK%s %s\n" "$c_bad"  "$c_off" "$*"; PROBLEMS=$((PROBLEMS+1)); }
warn() { printf "  %sWARN %s %s\n" "$c_warn" "$c_off" "$*"; }
ok()   { printf "  %sOK   %s %s\n" "$c_ok"   "$c_off" "$*"; }
note() { printf "        %s\n" "$*"; }

# ---- option parsing --------------------------------------------------------
# sbatch reads #SBATCH directives only until the first line that is neither a
# comment nor blank, and the command line overrides the file. Both facts
# matter for costing: a directive placed after the first command is ignored by
# sbatch but would be costed here if we read the whole file, and
# `sbatch -c 48 job.sbatch` costs twelve times a file that says -c 4.

declare -A OPT=()

declare -A SHORT_TO_LONG=(
    [A]=account [p]=partition [c]=cpus-per-task [n]=ntasks [N]=nodes
    [t]=time    [a]=array     [J]=job-name      [o]=output [e]=error
    [D]=chdir   [d]=dependency [C]=constraint   [q]=qos    [w]=nodelist
    [m]=distribution [S]=core-spec [G]=gpus
)
VALUED_LONG=" account partition cpus-per-task ntasks ntasks-per-node ntasks-per-core \
nodes mem mem-per-cpu mem-per-gpu time time-min array job-name output error chdir \
dependency constraint qos gres gpus gpus-per-node gpus-per-task nodelist wrap \
mail-type mail-user distribution core-spec threads-per-core reservation \
signal licenses clusters comment export cpus-per-gpu "

takes_value() { case "$VALUED_LONG" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

parse_tokens() {
    local -a toks=("$@")
    local i=0 n=${#toks[@]} tok name val
    while [ "$i" -lt "$n" ]; do
        tok="${toks[$i]}"
        case "$tok" in
            --*=*)
                name="${tok%%=*}"; name="${name#--}"; val="${tok#*=}"
                OPT["$name"]="$val" ;;
            --*)
                name="${tok#--}"
                if takes_value "$name" && [ $((i+1)) -lt "$n" ]; then
                    i=$((i+1)); OPT["$name"]="${toks[$i]}"
                else
                    OPT["$name"]=1        # a flag such as --exclusive
                fi ;;
            -[A-Za-z]*)
                local s="${tok:1:1}" rest="${tok:2}"
                name="${SHORT_TO_LONG[$s]:-}"
                [ -z "$name" ] && { i=$((i+1)); continue; }
                if [ -n "$rest" ]; then
                    OPT["$name"]="${rest#=}"
                elif [ $((i+1)) -lt "$n" ]; then
                    i=$((i+1)); OPT["$name"]="${toks[$i]}"
                fi ;;
        esac
        i=$((i+1))
    done
}

# 1. the directives sbatch will actually read: stop at the first command
mapfile -t SBATCH_TOKENS < <(
    awk '
        /^[[:space:]]*#SBATCH[[:space:]]/ { sub(/^[[:space:]]*#SBATCH[[:space:]]+/, ""); sub(/[[:space:]]+#.*$/, ""); print; next }
        /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
        { exit }
    ' "$SCRIPT" | tr -s '[:space:]' '\n' | grep -v '^$'
)
[ "${#SBATCH_TOKENS[@]}" -gt 0 ] && parse_tokens "${SBATCH_TOKENS[@]}"
IGNORED_DIRECTIVES="$(awk '
    BEGIN { body = 0; n = 0 }
    /^[[:space:]]*#SBATCH[[:space:]]/ { if (body) n++; next }
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    { body = 1 }
    END { print n }' "$SCRIPT")"

# 2. the command line, which overrides the file
CLI_ARGS=("$@")
OVERRIDDEN=""
if [ "${#CLI_ARGS[@]}" -gt 0 ]; then
    declare -A BEFORE=()
    for k in "${!OPT[@]}"; do BEFORE["$k"]="${OPT[$k]}"; done
    parse_tokens "${CLI_ARGS[@]}"
    for k in "${!OPT[@]}"; do
        if [ "${BEFORE[$k]-__unset__}" != "${OPT[$k]}" ] && [ -n "${BEFORE[$k]-}" ]; then
            OVERRIDDEN+=" $k(${BEFORE[$k]}->${OPT[$k]})"
        fi
    done
fi

opt() { printf '%s' "${OPT[$1]-}"; }

# ---- helpers ---------------------------------------------------------------
# SLURM time string -> minutes. Accepts MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS
time_to_min() {
    local t="$1"
    local d=0
    local rest="$t"
    case "$t" in *-*) d="${t%%-*}"; rest="${t#*-}" ;; esac
    local h=0 m=0 s=0
    case "$(printf '%s' "$rest" | tr -cd ':' | wc -c)" in
        0) if [ "$d" != 0 ]; then h="$rest"; else m="$rest"; fi ;;
        1) if [ "$d" != 0 ]; then h="${rest%%:*}"; m="${rest#*:}"
           else m="${rest%%:*}"; s="${rest#*:}"; fi ;;
        2) h="${rest%%:*}"; local hm="${rest#*:}"; m="${hm%%:*}"; s="${hm#*:}" ;;
        *) printf '%s' ""; return 1 ;;
    esac
    [[ "$d$h$m$s" =~ ^[0-9]+$ ]] || { printf '%s' ""; return 1; }
    awk -v d="${d:-0}" -v h="${h:-0}" -v m="${m:-0}" -v s="${s:-0}" \
        'BEGIN{ printf "%.4f", d*1440 + h*60 + m + s/60 }'
}

# array spec -> number of tasks, honouring 1-200%20 and 1,5,9:2
array_tasks() {
    local spec="${1%%%*}"
    [ -z "$spec" ] && { echo 1; return; }
    local total=0 part a b
    IFS=',' read -ra parts <<< "$spec"
    for part in "${parts[@]}"; do
        part="${part%%:*}"
        if [[ "$part" == *-* ]]; then
            a="${part%%-*}"; b="${part#*-}"
            [[ "$a" =~ ^[0-9]+$ && "$b" =~ ^[0-9]+$ ]] && total=$((total + b - a + 1))
        elif [[ "$part" =~ ^[0-9]+$ ]]; then
            total=$((total + 1))
        fi
    done
    [ "$total" -lt 1 ] && total=1
    echo "$total"
}

# memory string -> MB. Accepts sbatch forms (64gb, 4000M, 2T) and quota forms (50589M).
to_mb() {
    local v="${1:-}"; [ -z "$v" ] && { echo 0; return; }
    awk -v s="$v" 'BEGIN{
        n = s + 0
        if      (s ~ /[tT][bB]?$/) printf "%.0f", n*1048576
        else if (s ~ /[gG][bB]?$/) printf "%.0f", n*1024
        else if (s ~ /[kK][bB]?$/) printf "%.0f", n/1024
        else                       printf "%.0f", n
    }'
}

echo "preflight: $SCRIPT"
echo "----------------------------------------------------------------"

ACCT="$(opt account)"
PART="$(opt partition)"
CPUS="$(opt cpus-per-task)"
NTASKS="$(opt ntasks)"
TPN="$(opt ntasks-per-node)"
NODES="$(opt nodes)"
MEM="$(opt mem)"
MEMPERCPU="$(opt mem-per-cpu)"
TIME="$(opt time)"
ARRAY="$(opt array)"
GRES="$(opt gres)"
GPUS_OPT="$(opt gpus)$(opt gpus-per-node)$(opt gpus-per-task)"

[ -z "$ACCT" ] && ACCT="$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null | head -1)"
CPUS_GIVEN="$CPUS"
CPUS="${CPUS:-1}"
NODES="${NODES:-1}"; NODES="${NODES##*-}"      # a range such as 2-4 is costed at its upper bound
if [ -n "$TPN" ]; then NTASKS=$(( NODES * TPN )); else NTASKS="${NTASKS:-1}"; fi
TASKS="$(array_tasks "$ARRAY")"
CORES=$(( CPUS * NTASKS ))
[ "$CORES" -lt "$NODES" ] && CORES="$NODES"

GPUS=0
if [[ "$GRES" =~ gpu(:[a-zA-Z0-9_]+)?:([0-9]+) ]]; then GPUS="${BASH_REMATCH[2]}"; fi
GPU_TYPED=0
[[ "$GRES" =~ gpu:[a-zA-Z_][a-zA-Z0-9_]*:[0-9]+ ]] && GPU_TYPED=1

TOTAL_MEM_MB=0
if   [ -n "$MEM" ];       then TOTAL_MEM_MB=$(( $(to_mb "$MEM") * NODES ))
elif [ -n "$MEMPERCPU" ]; then TOTAL_MEM_MB=$(( $(to_mb "$MEMPERCPU") * CORES ))
fi

echo "REQUEST"
printf "  %-16s %s\n" "account"    "${ACCT:-<none - and no default set>}"
printf "  %-16s %s\n" "partition"  "${PART:-<cluster default>}"
shape="cpus-per-task=$CPUS ntasks=$NTASKS nodes=$NODES"
[ -n "$TPN" ] && shape="$shape ntasks-per-node=$TPN"
printf "  %-16s %s\n" "cores/job"  "$CORES  ($shape)"
if   [ -n "$MEM" ];       then printf "  %-16s %s\n" "memory" "$MEM per node"
elif [ -n "$MEMPERCPU" ]; then printf "  %-16s %s\n" "memory" "$MEMPERCPU per core x $CORES cores = ${TOTAL_MEM_MB} MB"
else                           printf "  %-16s %s\n" "memory" "<default: 4000 MB per core = $((CORES * 4000)) MB>"; fi
printf "  %-16s %s\n" "time"       "${TIME:-<MISSING>}"
[ "$TASKS" -gt 1 ] && printf "  %-16s %s\n" "array tasks" "$TASKS  (from --array=$ARRAY)"
[ -n "$GRES" ]     && printf "  %-16s %s\n" "gres"        "$GRES"
[ "$GPUS" -gt 0 ]  && printf "  %-16s %s\n" "gpus/job"    "$GPUS"
if [ -n "$OVERRIDDEN" ]; then
    printf "  %-16s %s\n" "cli overrides" "$OVERRIDDEN"
    printf "  %-16s %s\n" "" "(command-line options beat #SBATCH directives, and are costed)"
fi
echo

echo "CHECKS"

# ---------------------------------------------------------------- 0. the file itself
if [ "$IGNORED_DIRECTIVES" -gt 0 ]; then
    warn "$IGNORED_DIRECTIVES #SBATCH line(s) come after the first command. sbatch stops reading directives there, so they do nothing; write notes as ##SBATCH."
fi
for k in "${!OPT[@]}"; do
    if [[ "${OPT[$k]}" =~ YOUR_[A-Z_]+ ]] || [[ "${OPT[$k]}" =~ ^\<[^\>]+\>$ ]]; then
        bad "--$k=${OPT[$k]} is a placeholder. sbatch would accept it as written (mail to YOUR_EMAIL goes nowhere; an unknown account is refused only at submit)."
    fi
done

# Log files. sbatch does not expand $variables in directives, --test-only does
# not look at the path, and a job whose log directory is missing is accepted
# and then fails at launch [M:SUB-14]. Also the only sensible trigger for the
# "$HOME is nearly full" warning is a log that will actually land there.
CHDIR="$(opt chdir)"; [ -z "$CHDIR" ] && CHDIR="$PWD"
LOGS_IN_HOME=0
for k in output error; do
    v="$(opt "$k")"; [ -z "$v" ] && { [[ "$CHDIR" == "$HOME"* ]] && LOGS_IN_HOME=1; continue; }
    if [[ "$v" == *'$'* ]]; then
        bad "--$k=$v contains a \$variable; sbatch does not expand variables in #SBATCH directives, so the log would be written to a literal '\$...' path. Use a fixed path."
        continue
    fi
    [[ "$v" != /* ]] && v="$CHDIR/$v"
    d="$(dirname "$v")"
    d="${d//%x/x}"; d="${d//%j/1}"; d="${d//%A/1}"; d="${d//%a/1}"; d="${d//%u/$USER}"
    if [ ! -d "$d" ]; then
        bad "--$k directory $d does not exist (a relative path is taken from the directory you run sbatch in; this check used $CHDIR). sbatch and --test-only accept it; the job then fails at launch without writing a log. mkdir -p it first."
    fi
    [[ "$d" == "$HOME"* ]] && LOGS_IN_HOME=1
done

# ---------------------------------------------------------------- 1. --time
MINUTES=""
if [ -z "$TIME" ]; then
    bad "--time is missing. This site requires it (require_timelimit); sbatch will refuse the job."
else
    MINUTES="$(time_to_min "$TIME")"
    if [ -z "$MINUTES" ]; then
        warn "could not parse --time='$TIME'; skipping the cost arithmetic"
    else
        if [[ "$TIME" =~ ^[0-9]+$ ]]; then
            warn "--time=$TIME is a bare number, so it means $TIME MINUTES, not hours"
        fi
        ok "--time=$TIME parses as $(awk -v m="$MINUTES" 'BEGIN{printf "%.2f", m/60}') h"
    fi
fi

# ---------------------------------------------------------------- 2. GPU rules
# All measured with sbatch --test-only; see references/partitions-and-hardware.md.
if [ -n "$GPUS_OPT" ]; then
    bad "--gpus / --gpus-per-node / --gpus-per-task are rejected here. The only accepted form is --gres=gpu:hopper:N."
fi
if [[ "$GRES" == *gpu* ]] && [ "$GPU_TYPED" -eq 0 ]; then
    bad "--gres=$GRES has no GPU type. This site accepts only --gres=gpu:hopper:N ('Invalid generic resource' otherwise)."
fi
case "$PART" in
    lem-gpu*)
        if [ "$GPUS" -eq 0 ]; then
            [ -z "$GPUS_OPT" ] && bad "$PART requires a GPU: a job without --gres=gpu:hopper:N is rejected ('does not meet the minimum required resources')."
        else
            if [ -z "$CPUS_GIVEN" ]; then
                bad "GPU jobs must state --cpus-per-task explicitly; without it the submit filter rejects the job."
            fi
            if [ "$CORES" -gt $(( 16 * GPUS * NODES )) ]; then
                bad "$CORES cores for $GPUS GPU(s): the limit is 16 cores per GPU ('exceeds the maximum allowed resources'). Ask for more GPUs or fewer cores."
            fi
            if [ "$TOTAL_MEM_MB" -gt $(( 250 * 1024 * GPUS * NODES )) ]; then
                bad "$TOTAL_MEM_MB MB for $GPUS GPU(s): the limit is 250 GB per GPU, counted over --mem or --mem-per-cpu x cores."
            fi
            [ "$PROBLEMS" -eq 0 ] && ok "--gres=gpu:hopper:$GPUS is the accepted form (--gpus=N would be refused) and fits the per-GPU caps (16 cores, 250 GB per GPU)"
        fi ;;
    "")
        [ "$GPUS" -gt 0 ] && warn "GPUs requested with no partition; name lem-gpu-short, lem-gpu-normal or the lem-gpu alias. Only Lem has GPUs." ;;
    *)
        [ "$GPUS" -gt 0 ] && bad "$PART has no GPUs; only lem-gpu-* do ('Requested node configuration is not available')." ;;
esac

# ---------------------------------------------------------------- 3. the cost
if [ -n "$MINUTES" ] && [ -n "$ACCT" ]; then
    COST_CPU_MIN="$(awk -v c="$CORES" -v m="$MINUTES" -v t="$TASKS" 'BEGIN{printf "%.0f", c*m*t}')"
    COST_GPU_MIN="$(awk -v g="$GPUS" -v m="$MINUTES" -v t="$TASKS" 'BEGIN{printf "%.0f", g*m*t}')"

    LINE="$(scontrol show assoc_mgr "qos=$ACCT" flags=qos 2>/dev/null | grep -m1 -oE 'GrpTRESMins=.*')"
    if [ -z "$LINE" ]; then
        warn "no QOS limits found for account '$ACCT'; cannot check affordability (is the account name right?)"
    else
        for res in cpu gres/gpu; do
            # The first TRES follows "GrpTRESMins=", the rest follow a comma.
            # "gres/gpu=" cannot match "gres/gpu:hopper=".
            pair="$(echo "$LINE" | grep -oE "[=,]${res}=[0-9]+\([0-9]+\)" | head -1 | sed 's/^[=,]//')"
            if [ "$res" = "cpu" ]; then cost="$COST_CPU_MIN"; label="CPU"; else cost="$COST_GPU_MIN"; label="GPU"; fi
            if [ -z "$pair" ]; then
                if [ "$res" = "gres/gpu" ] && [ "$cost" -gt 0 ]; then
                    bad "this job needs $GPUS GPU(s) but account $ACCT has no GPU-hours pool at all (service-balance --check-gpu lists the ones that do)."
                fi
                continue
            fi
            lim="${pair#*=}"; lim="${lim%%(*}"
            use="${pair#*\(}"; use="${use%)*}"
            left=$(( lim - use ))
            [ "$cost" -eq 0 ] && continue
            ch="$(awk -v v="$cost" 'BEGIN{printf "%.1f", v/60}')"
            lh="$(awk -v v="$left" 'BEGIN{printf "%.1f", v/60}')"
            if [ "$cost" -gt "$left" ]; then
                bad "$label cost ${ch} h EXCEEDS the ${lh} h left on $ACCT."
                note "This job would be accepted and then sit in PENDING with reason"
                note "QOSGrp${label}MinutesLimit forever. sbatch --test-only will NOT warn you."
                note "Shrink it, use another service, or request more hours."
            elif awk -v c="$cost" -v l="$left" 'BEGIN{exit !(c > l*0.5)}'; then
                warn "$label cost ${ch} h uses over half of the ${lh} h left on $ACCT"
            else
                ok "$label cost ${ch} h fits within the ${lh} h left on $ACCT"
            fi
        done
    fi
elif [ -z "$ACCT" ]; then
    bad "no --account and no default account; cannot tell whose grant would pay"
fi

# ---------------------------------------------------------------- 4. quota
if command -v quota >/dev/null 2>&1; then
    q="$(quota -s -f ~ 2>/dev/null | tail -1)"
    used_s="$(awk '{print $1}' <<<"$q")"; lim_s="$(awk '{print $3}' <<<"$q")"
    um="$(to_mb "$used_s")"; lm="$(to_mb "$lim_s")"
    if awk -v l="$lm" 'BEGIN{exit !(l>0)}'; then
        free_mb="$(awk -v u="$um" -v l="$lm" 'BEGIN{printf "%.0f", l-u}')"
        if [ "$free_mb" -lt 1024 ] && [ "$LOGS_IN_HOME" -eq 1 ]; then
            warn "\$HOME has only ${free_mb} MB free of ${lm} MB and this job's logs land there; a growing log hits EDQUOT. Point --output at PD, and write data to \$TMPDIR."
        elif [ "$free_mb" -lt 1024 ]; then
            ok "\$HOME has only ${free_mb} MB free, but this job writes its logs elsewhere"
        else
            ok "\$HOME has ${free_mb} MB free"
        fi
    fi
    # No grace column while under the soft limit, so the file fields shift left.
    files_used="$(awk '{print (NF==6)?$4:$5}' <<<"$q")"; files_lim="$(awk '{print (NF==6)?$6:$7}' <<<"$q")"
    [ -n "$files_lim" ] && ok "\$HOME file count ${files_used:-?} of ${files_lim} (inodes run out before bytes do)"
fi

# ---------------------------------------------------------------- 4b. node shape
# sbatch --test-only refuses a job no node can hold, but its message does not
# say what the largest node has. Ask per node: the grouped form (sinfo -o %m
# without -N) prints "190464+" for a partition whose nodes differ in memory,
# which is not a number. The scheduler keeps a little of each node back, so a
# request just under the figure can still be refused; --test-only below is
# the last word. Aliases such as lem-gpu resolve to their hidden partition's
# nodes, or to none, and the check is safe either way.
if [ "$TOTAL_MEM_MB" -gt 0 ] && [ -n "$PART" ]; then
    biggest="$(sinfo -h -N -p "$PART" -o "%m" 2>/dev/null | tr -d '+' | sort -n | tail -1)"
    if [ -n "$biggest" ] && [ "$biggest" -gt 0 ] 2>/dev/null; then
        pernode=$(( TOTAL_MEM_MB / NODES ))
        if [ "$pernode" -gt "$biggest" ]; then
            bad "${pernode} MB per node requested, but the largest node in $PART has ${biggest} MB. No node can hold this job; sbatch refuses it as 'Requested node configuration is not available'."
        else
            ok "memory nominally fits a node: ${pernode} MB per node requested, largest node in $PART has ${biggest} MB"
        fi
    fi
fi

# ---------------------------------------------------------------- 5. test-only
echo
echo "sbatch --test-only (syntax, partition, wall limits, node shapes - NOT the grant):"
if out="$(sbatch --test-only "$@" "$SCRIPT" 2>&1)"; then
    printf '  %s\n' "$out"
    ok "accepted by the scheduler's own dry run"
else
    printf '%s\n' "$out" | grep -vE '^sbatch: error: *$' | sed 's/^/  /'
    bad "sbatch --test-only rejected this job (see above)"
    case "$out" in
        *"Requested node configuration is not available"*)
            note "No single node in the partition has this many cores, this much memory and these GRES at once."
            note "Node shapes: sinfo -N -p $PART -o '%n %c %m %G' | sort -u -k2" ;;
        *"minimum required resources"*)
            note "GPU partitions need --gres=gpu:hopper:N." ;;
        *"exceeds the maximum allowed resources"*)
            note "Per-GPU caps: 16 cores and 250 GB per GPU, and -c must be given." ;;
        *"Required partition not available"*)
            note "With a bare alias (bem2-cpu, lem-cpu, lem-gpu) this means --time exceeds every variant's wall limit." ;;
        *"Time limit specification required"*)
            note "--time is mandatory here." ;;
        *"Invalid account"*)
            note "Accounts you may use: sacctmgr -nP show user \$USER format=Account -s" ;;
        *"Only \`srun\` interactive"*|*"Only 'srun' interactive"*)
            note "*-interactive partitions take srun --pty or sub-interactive, never sbatch." ;;
    esac
fi

echo "----------------------------------------------------------------"
if [ "$PROBLEMS" -gt 0 ]; then
    echo "  ${c_bad}$PROBLEMS blocking problem(s). Not ready to submit.${c_off}"
    exit 1
fi
echo "  ${c_ok}No blocking problems.${c_off}  Billed to: $ACCT"
echo "  Submit with:  JOBID=\$(sbatch --parsable $SCRIPT) && echo \"\$JOBID\""
echo "  Then record it:  scripts/run-record.sh \"\$JOBID\""
exit 0
