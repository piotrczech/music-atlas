#!/usr/bin/env bash
# run-record.sh - emit a provenance record for a finished job.
#
#   scripts/run-record.sh <jobid> [--format md|yaml|tsv] [--commit <sha>]
#
# Why this exists: sacct on this site keeps per-job records for roughly three
# months, and then they are gone -- while aggregate grant usage survives. If you
# want to know later which job produced which result, and what it cost, you have
# to write it down at the time. This does that.
#
# It also fills two gaps in the local tooling:
#   * MaxRSS is reported only on the .batch step, never the job row, so a naive
#     "sacct | head -1" reads it as empty.
#   * seff and reportseff are not installed here, so CPU efficiency and memory
#     headroom are computed from sacct fields directly.
#
# READ-ONLY with respect to the cluster: only reads sacct.

set -uo pipefail

JOBID="${1:-}"
[ -z "$JOBID" ] && { echo "usage: $0 <jobid> [--format md|yaml|tsv] [--commit <sha>]" >&2; exit 2; }
shift

FORMAT=md
COMMIT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --format) FORMAT="${2:-md}"; shift 2 ;;
        --commit) COMMIT="${2:-}"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

command -v sacct >/dev/null 2>&1 || { echo "sacct not found; run on the WCSS login node." >&2; exit 2; }

FIELDS=JobID,JobName,Account,Partition,State,ExitCode,Submit,Start,End,Elapsed,ElapsedRaw,TimelimitRaw,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList,WorkDir,SubmitLine
RAW="$(sacct -j "$JOBID" -n -P -o "$FIELDS" 2>/dev/null)"
[ -z "$RAW" ] && {
    echo "No sacct record for job $JOBID." >&2
    echo "Records here survive about three months; older jobs are unrecoverable." >&2
    exit 1
}

# The job row is the one whose JobID has no dot; MaxRSS lives on .batch.
job_row="$(awk -F'|' '$1 !~ /\./ {print; exit}' <<<"$RAW")"
batch_row="$(awk -F'|' '$1 ~ /\.batch$/ {print; exit}' <<<"$RAW")"

f() { awk -F'|' -v n="$1" '{print $n}' <<<"$2"; }

JID=$(f 1 "$job_row");        NAME=$(f 2 "$job_row")
ACCT=$(f 3 "$job_row");       PART=$(f 4 "$job_row")
STATE=$(f 5 "$job_row");      EXIT=$(f 6 "$job_row")
SUBMIT=$(f 7 "$job_row");     START=$(f 8 "$job_row");   END=$(f 9 "$job_row")
ELAPSED=$(f 10 "$job_row");   ELRAW=$(f 11 "$job_row");  TLRAW=$(f 12 "$job_row")
CPUS=$(f 13 "$job_row");      REQMEM=$(f 14 "$job_row")
TOTALCPU=$(f 16 "$job_row");  NODES=$(f 17 "$job_row");  WORKDIR=$(f 18 "$job_row")
SUBMITLINE=$(f 19 "$job_row")
MAXRSS=$(f 15 "$batch_row")
[ -z "$MAXRSS" ] && MAXRSS=$(f 15 "$job_row")

# ---- derived numbers -------------------------------------------------------
to_kb() { # 1049636K / 51820K / 1.5G / 2M -> KiB
    local v="${1:-}"; [ -z "$v" ] && { echo 0; return; }
    case "$v" in
        *K) awk -v n="${v%K}" 'BEGIN{printf "%.0f", n}' ;;
        *M) awk -v n="${v%M}" 'BEGIN{printf "%.0f", n*1024}' ;;
        *G) awk -v n="${v%G}" 'BEGIN{printf "%.0f", n*1048576}' ;;
        *T) awk -v n="${v%T}" 'BEGIN{printf "%.0f", n*1073741824}' ;;
        *)  awk -v n="$v" 'BEGIN{printf "%.0f", n}' 2>/dev/null || echo 0 ;;
    esac
}
# SLURM TotalCPU: [DD-]HH:MM:SS[.mmm] or MM:SS.mmm
cputime_sec() {
    local t="${1:-0}" d=0
    case "$t" in *-*) d="${t%%-*}"; t="${t#*-}" ;; esac
    awk -v d="$d" -v t="$t" 'BEGIN{
        n=split(t,p,":"); s=0
        if(n==3) s=p[1]*3600+p[2]*60+p[3]
        else if(n==2) s=p[1]*60+p[2]
        else s=p[1]
        printf "%.2f", d*86400+s }'
}

CPU_SEC="$(cputime_sec "$TOTALCPU")"
CPUH="$(awk -v c="${CPUS:-0}" -v e="${ELRAW:-0}" 'BEGIN{printf "%.4f", c*e/3600}')"
EFF="n/a"
if [ "${ELRAW:-0}" -gt 0 ] && [ "${CPUS:-0}" -gt 0 ]; then
    EFF="$(awk -v u="$CPU_SEC" -v c="$CPUS" -v e="$ELRAW" 'BEGIN{printf "%.1f%%", 100*u/(c*e)}')"
fi
MEM_USE="n/a"
rss_kb="$(to_kb "$MAXRSS")"; req_kb="$(to_kb "$REQMEM")"
if [ "${req_kb:-0}" -gt 0 ] && [ "${rss_kb:-0}" -gt 0 ]; then
    MEM_USE="$(awk -v r="$rss_kb" -v q="$req_kb" 'BEGIN{printf "%.1f%%", 100*r/q}')"
fi
WALL_USE="n/a"
if [ "${TLRAW:-0}" -gt 0 ]; then
    WALL_USE="$(awk -v e="$ELRAW" -v t="$TLRAW" 'BEGIN{printf "%.1f%%", 100*e/(t*60)}')"
fi
# Without --commit the best we can do is the workdir's HEAD *now*, which may not
# be what the job ran. It is labelled as such; the job's own log line is better.
COMMIT_SOURCE="given"
if [ -z "$COMMIT" ] && [ -n "${WORKDIR:-}" ] && [ -d "$WORKDIR/.git" ]; then
    COMMIT="$(git -C "$WORKDIR" rev-parse --short HEAD 2>/dev/null || true)"
    [ -n "$COMMIT" ] && COMMIT_SOURCE="workdir HEAD at record time"
fi
[ -z "$COMMIT" ] && { COMMIT="unknown"; COMMIT_SOURCE="none"; }

# Account health that decides whether the NEXT run survives: a nearly full
# $HOME kills a job through its own log file (EDQUOT), and nothing on the
# triage path looks at the quota otherwise.
HOME_NOTE=""
if command -v quota >/dev/null 2>&1; then
    q="$(quota -s -f ~ 2>/dev/null | tail -1)"
    hu="$(awk '{print $1}' <<<"$q")"; hl="$(awk '{print $3}' <<<"$q")"
    to_mb_q() { awk -v s="${1:-0}" 'BEGIN{n=s+0; if(s~/[tT]$/)n*=1048576; else if(s~/[gG]$/)n*=1024; else if(s~/[kK]$/)n/=1024; printf "%.0f", n}'; }
    hum="$(to_mb_q "$hu")"; hlm="$(to_mb_q "$hl")"
    if [ "${hlm:-0}" -gt 0 ]; then
        pct=$(( 100 * hum / hlm )); free_mb=$(( hlm - hum ))
        [ "$pct" -ge 90 ] && HOME_NOTE="\$HOME is ${pct}% full (${free_mb} MB free of ${hlm} MB). A job whose log or output lands there dies with 'Disk quota exceeded'; point --output at PD and write data to \$TMPDIR."
    fi
fi

advice=""
case "$STATE" in
    OUT_OF_MEMORY*) advice="Killed by the OOM killer. MaxRSS on an OOM-killed job equals the request, so it is a floor, not a measurement: raise --mem well above it, or measure the real peak in an interactive session." ;;
    TIMEOUT*)       advice="Hit the wall limit. Raise --time, or checkpoint and restart." ;;
    FAILED*)        advice="Non-zero exit from the program itself; read the job output." ;;
    CANCELLED*)     advice="Cancelled by a user or an administrator." ;;
    COMPLETED*)
        if [ "$MEM_USE" != "n/a" ] && awk -v m="${MEM_USE%\%}" 'BEGIN{exit !(m<25)}'; then
            advice="Used only $MEM_USE of the memory requested; you are paying for idle reservation."
        elif [ "$EFF" != "n/a" ] && awk -v e="${EFF%\%}" 'BEGIN{exit !(e<50)}'; then
            advice="CPU efficiency $EFF; consider fewer cores, since billing is allocation x wall time."
        fi ;;
esac

case "$FORMAT" in
  tsv)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$JID" "$NAME" "$ACCT" "$PART" "$CPUS" "$REQMEM" "$ELAPSED" "$MAXRSS" "$CPUH" "$EXIT" "$COMMIT"
    ;;
  yaml)
    cat <<YAML
- job_id: "$JID"
  name: "$NAME"
  account: "$ACCT"
  partition: "$PART"
  state: "$STATE"
  exit_code: "$EXIT"
  submitted: "$SUBMIT"
  started: "$START"
  ended: "$END"
  elapsed: "$ELAPSED"
  cores: $CPUS
  mem_requested: "$REQMEM"
  max_rss: "$MAXRSS"
  cpu_hours_billed: $CPUH
  cpu_efficiency: "$EFF"
  memory_used_of_request: "$MEM_USE"
  walltime_used_of_limit: "$WALL_USE"
  nodes: "$NODES"
  workdir: "$WORKDIR"
  commit: "$COMMIT"
  commit_source: "$COMMIT_SOURCE"
  submit_line: "$SUBMITLINE"
  home_note: "$HOME_NOTE"
  recorded: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
YAML
    ;;
  md|*)
    cat <<MD

### Run $JID — $NAME

| field | value |
|---|---|
| state | \`$STATE\` (exit $EXIT) |
| account | \`$ACCT\` |
| partition | \`$PART\` |
| cores | $CPUS |
| memory requested | $REQMEM |
| MaxRSS | ${MAXRSS:-n/a} ($MEM_USE of request) |
| elapsed | $ELAPSED ($WALL_USE of limit) |
| CPU efficiency | $EFF |
| **CPU-hours billed** | **$CPUH** |
| nodes | $NODES |
| workdir | \`$WORKDIR\` |
| commit | \`$COMMIT\` ($COMMIT_SOURCE) |
| submit line | \`${SUBMITLINE:-n/a}\` |
| submitted / started / ended | $SUBMIT / $START / $END |
| recorded | $(date -u +%Y-%m-%dT%H:%M:%SZ) |
MD
    [ -n "$advice" ] && printf '\n> %s\n' "$advice"
    [ -n "$HOME_NOTE" ] && printf '\n> Account note: %s\n' "$HOME_NOTE"
    if [ "$COMMIT_SOURCE" != "given" ]; then
        printf '\n> Commit is %s. Pass --commit <sha> from the job log, or echo\n' "$COMMIT_SOURCE"
        printf '> `git rev-parse --short HEAD` inside the job script so the log carries it.\n'
    fi
    ;;
esac
