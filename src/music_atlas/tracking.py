"""Record a CLI step as an MLflow run with a run directory that serving can read without MLflow."""

import json
import os
import platform
import secrets
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from music_atlas.settings import ROOT, profile


def git_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def dataset_identity(data_dir: Path, version: str) -> dict:
    """Pin the run to the split's manifest hash; a local checkout may only have the sample."""
    for source in (data_dir / "ready" / version / "summary.json",
                   data_dir / "samples" / version / "sample.json"):
        if source.is_file():
            pinned = json.loads(source.read_text(encoding="utf-8"))
            return {"version": pinned["version"], "manifest_sha256": pinned["manifest_sha256"]}
    raise ValueError(f"No summary.json or sample.json for {version} under {data_dir}.")


def tracking_uri(outputs_dir: Path) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{outputs_dir / 'mlruns'}"
    if uri.startswith("file:"):
        # MLflow 3.17 refuses the file store unless opted in; SQLite breaks on Lustre (E2-01).
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


class Run:
    def __init__(self, run_id: str, directory: Path):
        self.run_id, self.directory = run_id, directory
        self.metrics, self.results = {}, {}

    def log(self, result: dict):
        """Numbers become metrics, everything else is kept as a result field."""
        for key, value in result.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                self.metrics[key] = value
            elif value is not None:
                self.results[key] = str(value)


@contextmanager
def track(step: str, profile_name: str, storage: dict[str, Path], version: str | None = None,
          representation: str | None = None):
    """Run one step as an MLflow run; failures keep their run, marked FAILED or KILLED."""
    import mlflow

    config = profile(profile_name)
    version = version or config["dataset"]["split"]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # The suffix keeps parallel array tasks started in the same second apart.
    run_id = f"{step}-{profile_name}-{stamp}-{secrets.token_hex(2)}"
    directory = (storage["artifacts"] / representation / run_id if representation
                 else storage["outputs"] / run_id)
    directory.mkdir(parents=True)
    dataset = dataset_identity(storage["data"], version)
    params = {
        **{key: value for key, value in config.items() if not isinstance(value, dict)},
        "profile": profile_name, "dataset_version": dataset["version"],
        "manifest_sha256": dataset["manifest_sha256"],
    }
    # Library versions follow from the Git revision through uv.lock.
    tags = {
        "atlas.step": step, "atlas.run_id": run_id, "atlas.run_dir": str(directory),
        "mlflow.source.git.commit": git_revision(), "host": platform.node(),
        "python": platform.python_version(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }

    mlflow.set_tracking_uri(tracking_uri(storage["outputs"]))
    mlflow.set_experiment(config["experiment"]["name"])
    run = Run(run_id, directory)
    started = time.perf_counter()
    status = "FAILED"
    # Not a `with` block: its exit would record an interrupted run as FAILED, not KILLED.
    active = mlflow.start_run(run_name=run_id, tags=tags)
    try:
        mlflow.log_params(params)
        yield run
        status = "FINISHED"
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    finally:
        run.metrics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        mlflow.log_metrics(run.metrics)
        if run.results:
            mlflow.log_params(run.results)
        record = {
            "run_id": run_id, "status": status, "mlflow_run_id": active.info.run_id,
            "experiment": config["experiment"]["name"], "params": params, "tags": tags,
            "metrics": run.metrics, "results": run.results,
        }
        (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        mlflow.end_run(status)
