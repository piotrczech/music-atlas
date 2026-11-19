"""Record a CLI step as an MLflow run with a run directory that serving can read without MLflow."""

import json
import os
import platform
import secrets
import subprocess
import sys
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from music_atlas.settings import ROOT, profile

try:
    import resource
except ImportError:  # Windows
    resource = None

# MLflow copies logged files into its store; vectors and checkpoints stay in the run directory.
MAX_LOGGED_FILE_MIB = 50
# Every run files its MLflow artifacts the same way: what it read, how it scored, what it
# fitted and how it ran.
ARTIFACT_FOLDERS = ("dataset", "evaluation", "model", "run")


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
    for source, kind in ((data_dir / "ready" / version / "summary.json", "split"),
                         (data_dir / "samples" / version / "sample.json", "quick-sample")):
        if source.is_file():
            pinned = json.loads(source.read_text(encoding="utf-8"))
            return {"alias": version, "version": pinned["version"],
                    "manifest_sha256": pinned["manifest_sha256"], "source": source.parent,
                    "kind": kind}
    raise ValueError(f"No summary.json or sample.json for {version} under {data_dir}.")


def tracking_uri(outputs_dir: Path) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{outputs_dir / 'mlruns'}"
    if uri.startswith("file:"):
        # MLflow 3.17 refuses the file store unless opted in; SQLite breaks on Lustre (E2-01).
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    return uri


def usage() -> dict:
    """CPU seconds and peak RSS of this process and its finished workers, not the whole node."""
    if resource is None:
        return {}
    own, workers = (resource.getrusage(who)
                    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    scale = 2**20 if sys.platform == "darwin" else 2**10
    return {"cpu_seconds": own.ru_utime + own.ru_stime + workers.ru_utime + workers.ru_stime,
            "peak_rss_mib": round(max(own.ru_maxrss, workers.ru_maxrss) / scale)}


def dataset_input(dataset: dict, name: str, summary: dict | None = None, tracks=None):
    """An MLflow dataset whose digest is the manifest hash and whose profile is our summary."""
    from mlflow.data.dataset_source_registry import resolve_dataset_source
    from mlflow.data.meta_dataset import MetaDataset
    from mlflow.data.pandas_dataset import PandasDataset

    class TrackTable(PandasDataset):
        # MLflow's own profile only counts rows and cells; the UI shows ours instead.
        @property
        def profile(self):
            return summary

    with warnings.catch_warnings():
        # MLflow 3.17 registers the local source class twice and warns that it is ambiguous.
        warnings.filterwarnings("ignore", "The specified dataset source", UserWarning)
        source = resolve_dataset_source(str(dataset["source"]))
    # 16 hex digits of the manifest hash tell versions apart; the full hash is a run param.
    digest = dataset["manifest_sha256"][:16]
    if tracks is None:
        return MetaDataset(source, name=name, digest=digest)
    return TrackTable(df=tracks, source=source, name=name, digest=digest)


class Run:
    def __init__(self, run_id: str, directory: Path, profile_name: str, dataset: dict):
        self.run_id, self.directory = run_id, directory
        self.profile_name, self.dataset = profile_name, dataset
        self.metrics, self.results, self.subsets = {}, {}, []

    def log(self, result: dict, prefix: str = ""):
        """Numbers become metrics and other values results; nested dicts become `key/name`."""
        for key, value in result.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                self.log(value, f"{name}/")
            elif isinstance(value, int | float) and not isinstance(value, bool):
                self.metrics[name] = value
            elif value is not None:
                self.results[name] = str(value)

    def log_dataset(self, tracks, subset: str, genres: dict | None = None,
                    summary: dict | None = None):
        """Record the tracks this run read as one MLflow dataset, e.g. `train` or `quick-sample`.

        Counts describing the data belong here, not in metrics. `genres` maps the scored genre
        IDs to names for per-genre positives; `summary` adds step-specific fields.
        """
        import mlflow

        profile = {
            "profile": self.profile_name, "subset": subset,
            "dataset_version": self.dataset["version"],
            "manifest_sha256": self.dataset["manifest_sha256"], "tracks": len(tracks),
        }
        if "artist_id" in tracks:
            profile["artists"] = int(tracks["artist_id"].nunique())
        if genres and "genre_ids" in tracks:
            counts = Counter(genre for ids in tracks["genre_ids"] for genre in ids)
            profile["genre_positives"] = {name: counts[genre] for genre, name in genres.items()}
        profile |= summary or {}

        name = f"{self.dataset['alias']}/{subset}"
        with warnings.catch_warnings():
            # Schema inference warns that integer columns cannot hold missing values.
            warnings.filterwarnings("ignore", "Hint: Inferred schema", UserWarning)
            mlflow.log_input(dataset_input(self.dataset, name, profile, tracks), context=subset,
                             tags={"profile": self.profile_name})
        path = self.directory / "dataset" / f"{subset}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(path), "dataset")
        self.subsets.append(subset)

    def log_file(self, path: Path, folder: str):
        """Attach a small file to one of ARTIFACT_FOLDERS of the MLflow run."""
        import mlflow

        if folder.split("/")[0] not in ARTIFACT_FOLDERS:
            raise ValueError(f"Artifact folder must start with one of {ARTIFACT_FOLDERS}.")
        if path.stat().st_size > MAX_LOGGED_FILE_MIB * 2**20:
            raise ValueError(f"{path} is too large for MLflow; keep it in {self.directory}.")
        mlflow.log_artifact(str(path), folder)


@contextmanager
def track(step: str, profile_name: str, storage: dict[str, Path], version: str | None = None,
          representation: str | None = None, description: str | None = None,
          resume: str | None = None):
    """Run one step as an MLflow run; failures keep their run, marked FAILED or KILLED.

    `resume` continues an unfinished run in its directory as a new MLflow run that points
    back to the earlier attempt, so attempts stay apart from completed results.
    """
    import mlflow

    config = profile(profile_name)
    version = version or config["dataset"]["split"]
    base = storage["artifacts"] / representation if representation else storage["outputs"]
    tags = {}
    if resume:
        run_id, directory = resume, base / resume
        previous = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if previous["status"] == "FINISHED":
            raise ValueError(f"{resume} already finished; start a new run instead.")
        tags["atlas.resumed_from"] = previous["mlflow_run_id"]
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        # The suffix keeps parallel array tasks started in the same second apart.
        run_id = f"{step}-{profile_name}-{stamp}-{secrets.token_hex(2)}"
        directory = base / run_id
        directory.mkdir(parents=True)
    dataset = dataset_identity(storage["data"], version)
    params = {
        **{key: value for key, value in config.items() if not isinstance(value, dict)},
        "profile": profile_name, "dataset_version": dataset["version"],
        "manifest_sha256": dataset["manifest_sha256"],
    }
    # Library versions follow from the Git revision through uv.lock.
    tags |= {
        "atlas.step": step, "atlas.run_id": run_id, "atlas.run_dir": str(directory),
        "atlas.representation": representation or "", "atlas.profile": profile_name,
        "mlflow.source.git.commit": git_revision(), "host": platform.node(),
        "cpus": str(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                    else os.cpu_count()),
        "python": platform.python_version(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }

    mlflow.set_tracking_uri(tracking_uri(storage["outputs"]))
    mlflow.set_experiment(config["experiment"]["name"])
    run = Run(run_id, directory, profile_name, dataset)
    started, before = time.perf_counter(), usage()
    status = "FAILED"
    # Not a `with` block: its exit would record an interrupted run as FAILED, not KILLED.
    active = mlflow.start_run(run_name=run_id, tags=tags, description=description)
    try:
        mlflow.log_params(params)
        yield run
        status = "FINISHED"
    except KeyboardInterrupt:
        status = "KILLED"
        raise
    finally:
        if not run.subsets:
            # A step that did not describe its tracks still names the dataset version it used.
            mlflow.log_input(
                dataset_input(dataset, f"{dataset['alias']}/{dataset['kind']}"),
                context=dataset["kind"], tags={"profile": profile_name},
            )
        after = usage()
        run.metrics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        if after:
            run.metrics["cpu_seconds"] = round(after["cpu_seconds"] - before["cpu_seconds"], 1)
            run.metrics["peak_rss_mib"] = after["peak_rss_mib"]
        mlflow.log_metrics(run.metrics)
        if run.results:
            mlflow.log_params(run.results)
        record = {
            "run_id": run_id, "status": status, "mlflow_run_id": active.info.run_id,
            "experiment": config["experiment"]["name"], "description": description,
            "datasets": run.subsets, "params": params, "tags": tags, "metrics": run.metrics,
            "results": run.results,
        }
        (directory / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        for name in ("run.json", "metadata.json"):
            if (directory / name).is_file():
                mlflow.log_artifact(str(directory / name), "run")
        mlflow.end_run(status)
