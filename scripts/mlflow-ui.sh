#!/usr/bin/env bash
# Open the MLflow UI on the runs recorded by `atlas ... --track`. Jobs write to the store
# themselves, so the UI runs only while someone looks at it.
#
#   scripts/mlflow-ui.sh                      # local runs in outputs/mlruns on 127.0.0.1:5000
#   MLFLOW_PORT=5001 scripts/mlflow-ui.sh     # another port
#   MLFLOW_TRACKING_URI=file:/path/to/mlruns scripts/mlflow-ui.sh   # e.g. a copy from PD
#   scripts/mlflow-ui.sh --workers 1          # extra options go to `mlflow ui`
#
# The store defaults to $MLFLOW_TRACKING_URI, then $ATLAS_OUTPUTS_DIR/mlruns, as in
# music_atlas.tracking. MLFLOW_HOST and MLFLOW_PORT override 127.0.0.1 and 5000.

set -euo pipefail
cd "$(dirname "$0")/.."
store=${MLFLOW_TRACKING_URI:-file:${ATLAS_OUTPUTS_DIR:-outputs}/mlruns}
host=${MLFLOW_HOST:-127.0.0.1}
port=${MLFLOW_PORT:-5000}
# MLflow 3.17 refuses the file store unless opted in; SQLite breaks on Lustre (E2-01).
export MLFLOW_ALLOW_FILE_STORE=true MLFLOW_DISABLE_AGENT_HINT=1
echo "MLflow UI for $store on http://$host:$port" >&2
exec uv run --extra tracking mlflow ui --backend-store-uri "$store" \
  --host "$host" --port "$port" "$@"
