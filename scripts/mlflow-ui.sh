#!/usr/bin/env bash
# Open the MLflow UI on the runs recorded by `atlas ... --track`. Jobs write to the store
# themselves, so the UI runs only while someone looks at it.
#
#   scripts/mlflow-ui.sh                      # local runs in outputs/mlruns on 127.0.0.1:5000
#   scripts/mlflow-ui.sh --wcss               # copy the WCSS runs from PD first, then show them
#   scripts/mlflow-ui.sh --ood                # on a WCSS compute node, behind Open OnDemand
#   MLFLOW_PORT=5001 scripts/mlflow-ui.sh     # another port
#   MLFLOW_TRACKING_URI=file:/path/to/mlruns scripts/mlflow-ui.sh   # any other store
#   scripts/mlflow-ui.sh --workers 1          # extra options go to `mlflow ui`
#
# The store defaults to $MLFLOW_TRACKING_URI, then $ATLAS_OUTPUTS_DIR/mlruns, as in
# music_atlas.tracking. MLFLOW_HOST and MLFLOW_PORT override 127.0.0.1 and 5000.
# --wcss mirrors the PD store into $ATLAS_OUTPUTS_DIR/mlruns-wcss over SSH; ATLAS_WCSS_HOST
# (an SSH alias, default `wcss`) and ATLAS_WCSS_MLRUNS (the store on PD) override the source.
# --ood serves the PD store at https://ood.e-science.pl/node/<host>/<port>/ from a job or an
# OOD session (scripts/slurm/mlflow-ui.sbatch starts one); never on the login node.

set -euo pipefail
cd "$(dirname "$0")/.."
outputs=${ATLAS_OUTPUTS_DIR:-outputs}
store=${MLFLOW_TRACKING_URI:-file:$outputs/mlruns}
host=${MLFLOW_HOST:-127.0.0.1}
port=${MLFLOW_PORT:-5000}
uv_args=(--extra tracking)

if [[ ${1:-} == --wcss ]]; then
  shift
  remote=${ATLAS_WCSS_MLRUNS:-/lustre/pd03/hpc-piotrczech-1792095018/music-atlas/outputs/mlruns}
  copy=$(mkdir -p "$outputs/mlruns-wcss" && cd "$outputs/mlruns-wcss" && pwd)
  # Never fall back to the account password: three failures lock it for 24 h.
  rsync -a --delete -e "ssh -o NumberOfPasswordPrompts=0" \
    "${ATLAS_WCSS_HOST:-wcss}:$remote/" "$copy/"
  # The file store keeps absolute artifact paths; point the copy at itself, not at PD.
  find "$copy" -name meta.yaml -exec sed -i.orig "s#file://$remote#file://$copy#g" {} +
  find "$copy" -name meta.yaml.orig -delete
  store=file:$copy
elif [[ ${1:-} == --ood ]]; then
  shift
  : "${SLURM_JOB_ID:?Run --ood in a job or an OOD session, not on the login node}"
  # OOD proxies /node/<host>/<port>/ to the node with the path kept, as for its Jupyter
  # (base_url /node/${host}/${port}/), so the UI must listen on the node's interfaces under
  # that prefix. --allowed-hosts refuses requests that did not come through OOD.
  host=0.0.0.0
  port=${MLFLOW_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')}
  set -- --static-prefix "/node/$(hostname)/$port" --allowed-hosts ood.e-science.pl \
    --workers 1 "$@"
  # Jobs use the environment as synced on the login node; nothing is installed here.
  uv_args=(--offline --frozen --no-sync)
  echo "Open https://ood.e-science.pl/node/$(hostname)/$port/ after the SSO login." >&2
fi

# MLflow 3.17 refuses the file store unless opted in; SQLite breaks on Lustre (E2-01).
export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1
echo "MLflow UI for $store on http://$host:$port" >&2
exec uv run "${uv_args[@]}" mlflow ui --backend-store-uri "$store" \
  --host "$host" --port "$port" "$@"
