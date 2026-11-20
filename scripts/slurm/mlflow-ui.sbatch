#!/usr/bin/env bash
#SBATCH --job-name=atlas-mlflow-ui
# Lem, not Bem2: the MLflow server loads a cryptography build that needs the newer glibc of
# AlmaLinux 9 (Lem and the login node, where the venv is synced); Bem2 runs AlmaLinux 8.
#SBATCH --partition=lem-cpu-short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=outputs/slurm-%j.log

# Serves the MLflow UI on the PD store through Open OnDemand; the link is in the job log.
# Billed per core while it runs, busy or idle: `scancel JOB_ID` when finished.

set -euo pipefail
source "$HOME/.atlas-env.sh"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI in ~/.atlas-env.sh}"
exec scripts/mlflow-ui.sh --ood "$@"
