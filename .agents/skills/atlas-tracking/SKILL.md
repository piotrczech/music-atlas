---
name: atlas-tracking
description: Music Atlas convention for recording and reading experiment runs in MLflow. Use when adding a data or model step that should be tracked (feature extraction, log-mel, embeddings, baselines, studies), when logging metrics, figures or models, resuming a failed run, comparing runs, opening the MLflow UI, or finding a run's vectors. Project-specific; for WCSS jobs and storage also read wcss-slurm.
---

# Tracking runs in Music Atlas

One step = one MLflow run + one run directory. The code is `src/music_atlas/tracking.py`
(about 250 lines); read it instead of the MLflow docs.

## Wrap a step

The step function takes an optional `record` and logs into it; the CLI opens the run and
exposes it as `--track`. `sample.run` + `atlas quick-sample --track` is the reference.

```python
from music_atlas.settings import paths
from music_atlas.tracking import track

with track("acoustic-features", "quick", paths(), representation="acoustic",
           description="MFCC, chroma, spectral stats; mean+std over windows.") as run:
    run.log_dataset(train, "train", genres)           # once per subset read
    np.save(run.directory / "vectors.npy", vectors)   # + track_ids.npy, metadata.json
    run.log({"failed": k, "tracks_per_second": rate}) # outcome and speed of the step
    run.log({"recall": per_genre, "macro_f1": f1})    # nested dict -> recall/Rock
    run.log_file(report_csv, "evaluation")            # small files only (<= 50 MiB)
```

`track()` records by itself: Git revision (`-dirty` if modified), profile params (seed,
windows, sample rate), full manifest hash, host, CPUs, SLURM job ID, elapsed and CPU
seconds, peak RSS, status, and `run/run.json` (+ `run/metadata.json` if present).

## Dataset, metrics, artifacts: what goes where

| What | Call | Shows up as |
|---|---|---|
| Tracks the run read: counts, artists, genre positives, seed, window settings | `run.log_dataset(tracks, subset, genres, summary)` | MLflow Dataset (name `<split>/<subset>`, digest, schema, profile) + `dataset/<subset>.json` |
| What the step measured: loaded/failed counts, speed, scores | `run.log({...})` | Metrics |
| Text values of the outcome | `run.log({...})` | Params |
| Files | `run.log_file(path, folder)` | Artifacts in a fixed folder |

- `subset` is the part read: `train`, `validation`, `test` or `quick-sample`; call
  `log_dataset` once per subset. Its DataFrame needs `track_id`, ideally `artist_id` and
  `genre_ids`. `genres` maps scored genre IDs to names (`labels.json`).
- Dataset digest = first 16 hex of `manifest_sha256`: two runs with the same digest read
  the same split file. The version suffix (`fma-cap10-v1-46a8…`) names the split options.
- A step that never calls `log_dataset` still gets a dataset entry with the version only.
- Artifact folders, enforced by `log_file`:
  - `dataset/`: subset summaries, exclusions/rejects, data figures;
  - `evaluation/`: per-genre reports, confusion matrices, result figures;
  - `model/`: small fitted objects (scaler, class mapping, linear head);
  - `run/`: `run.json`, `metadata.json`, configs.

## Rules

- Run directory: `artifacts/<representation>/<run_id>/` with `representation=`, otherwise
  `outputs/<run_id>/`. Vectors, checkpoints and shards stay there; MLflow gets paths,
  numbers and small files. `run.json` in the directory mirrors the MLflow record, so the API
  and notebooks read results without MLflow.
- `run_id` = `<step>-<profile>-<UTC time>-<4 hex>`; also the MLflow run name.
- Experiment = the profile's `[experiment] name` (`music-atlas-quick`, `music-atlas-full`).
- Status: `FINISHED`, `FAILED` (exception) or `KILLED` (interrupt). Continue an unfinished
  run with `track(..., resume=run_id)`: same directory, new MLflow run tagged
  `atlas.resumed_from`. A finished run cannot be resumed.
- Pass numbers to `run.log` as numbers, not strings; strings become params. Do not log
  dataset descriptors (tracks, artists, genres covered) as metrics.
- Fit on train, select on validation; log test metrics only for the final evaluation.
- Quick profile first; size the full job from the quick run's `tracks_per_second`.

## Store and UI

- Store = `MLFLOW_TRACKING_URI`, else `file:$ATLAS_OUTPUTS_DIR/mlruns` (`outputs/mlruns`).
  Use the file store, never SQLite on Lustre. MLflow 3.17 needs
  `MLFLOW_ALLOW_FILE_STORE=true` for it; `tracking.py` and the script set it.
- UI: `scripts/mlflow-ui.sh` (overrides: `MLFLOW_TRACKING_URI`, `MLFLOW_PORT`,
  `MLFLOW_HOST`, extra args go to `mlflow ui`).
- Prefer querying over opening the UI when an agent needs numbers:

```python
import os, mlflow
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("file:outputs/mlruns")
mlflow.search_runs(experiment_names=["music-atlas-quick"],
                   filter_string="tags.`atlas.step` = 'quick-sample'",
                   order_by=["attributes.start_time DESC"])[
    ["tags.mlflow.runName", "status", "metrics.tracks_per_second", "tags.atlas.run_dir"]]
```

## WCSS

Read `wcss-slurm` for jobs, partitions, storage and costs. Agreed in E2-01, not yet set up or
verified on WCSS: `~/.atlas-env.sh` sets `MLFLOW_TRACKING_URI=file:$ATLAS_ARTIFACTS_DIR/mlruns`
on PD; jobs log while they run and no server is kept up. View runs in an OOD session or after
`rsync` of `mlruns/` to a laptop (no grant hours); a copied store keeps PD paths for artifacts.
