import json

import pandas as pd
import pytest

from music_atlas.tracking import track, tracking_uri

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    sample = tmp_path / "data" / "samples" / "v1"
    sample.mkdir(parents=True)
    pinned = {"version": "v1", "manifest_sha256": "abc"}
    (sample / "sample.json").write_text(json.dumps(pinned), encoding="utf-8")
    return {name: tmp_path / name for name in ("data", "artifacts", "outputs")}


def finished_run(storage, run):
    mlflow.set_tracking_uri(f"file:{storage['outputs'] / 'mlruns'}")
    record = json.loads((run.directory / "run.json").read_text(encoding="utf-8"))
    return record, mlflow.get_run(record["mlflow_run_id"])


def test_completed_run_records_dataset_metrics_and_files(storage, tmp_path):
    report = tmp_path / "report.csv"
    report.write_text("genre,recall\n", encoding="utf-8")
    tracks = pd.DataFrame({"track_id": [1, 2, 3], "artist_id": [7, 7, 8],
                           "genre_ids": [[2, 30], [2], [3]]})
    with track("probe", "quick", storage, "v1", description="A probe.") as run:
        run.log_dataset(tracks, "train", {2: "International", 3: "Blues"}, {"seed": 42})
        run.log({"loaded": 3, "source": "archive", "flag": True, "peak": None,
                 "recall": {"Rock": 0.5}})
        run.log_file(report, "evaluation")
        with pytest.raises(ValueError, match="folder"):
            run.log_file(report, "reports")
    record, logged = finished_run(storage, run)
    assert run.directory == storage["outputs"] / run.run_id
    assert record["status"] == logged.info.status == "FINISHED"
    assert logged.data.params["manifest_sha256"] == "abc"
    assert logged.data.params["seed"] == "42" and logged.data.params["source"] == "archive"
    assert logged.data.metrics["loaded"] == 3 and logged.data.metrics["recall/Rock"] == 0.5
    assert {"elapsed_seconds", "cpu_seconds", "peak_rss_mib"} <= set(logged.data.metrics)
    assert logged.data.tags["mlflow.note.content"] == "A probe."

    (dataset,) = logged.inputs.dataset_inputs
    assert dataset.dataset.name == "v1/train"
    profile = json.loads(dataset.dataset.profile)
    assert profile["manifest_sha256"] == "abc" and len(profile["tracks_sha256"]) == 64
    assert profile["tracks"] == 3 and profile["artists"] == 2 and profile["seed"] == 42
    assert profile["genre_positives"] == {"International": 2, "Blues": 1}
    assert "genre_ids" in dataset.dataset.schema
    assert json.loads((run.directory / "dataset" / "train.json").read_text()) == profile
    files = {item.path for item in mlflow.artifacts.list_artifacts(run_id=logged.info.run_id)}
    assert files == {"dataset", "evaluation", "run"}

    # MLflow reuses a dataset with the same name and digest, so a changed description of the
    # same tracks must get its own digest rather than show the first run's profile.
    with track("probe", "quick", storage, "v1") as again:
        again.log_dataset(tracks, "train", {2: "International", 3: "Blues"}, {"seed": 7})
    _, relogged = finished_run(storage, again)
    (redataset,) = relogged.inputs.dataset_inputs
    assert redataset.dataset.digest != dataset.dataset.digest
    assert json.loads(redataset.dataset.profile)["seed"] == 7


def test_store_under_an_artifacts_folder_is_refused(storage, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'artifacts' / 'mlruns'}")
    with pytest.raises(ValueError, match="artifacts folder"):
        tracking_uri(storage["outputs"])


def test_failed_run_can_be_resumed_as_a_separate_attempt(storage):
    with pytest.raises(ValueError), track("probe", "quick", storage, "v1", "features") as run:
        raise ValueError("broken")
    record, failed = finished_run(storage, run)
    assert run.directory == storage["artifacts"] / "features" / run.run_id
    assert record["status"] == failed.info.status == "FAILED"
    # A step that logged no tracks still names the dataset version it read.
    (dataset,) = failed.inputs.dataset_inputs
    assert dataset.dataset.name == "v1/quick-sample"

    with track("probe", "quick", storage, "v1", "features", resume=run.run_id) as again:
        pass
    record, resumed = finished_run(storage, again)
    assert again.directory == run.directory and record["status"] == "FINISHED"
    assert resumed.data.tags["atlas.resumed_from"] == failed.info.run_id
    with pytest.raises(ValueError, match="already finished"), \
            track("probe", "quick", storage, "v1", "features", resume=run.run_id):
        pass


def test_overridden_setting_replaces_the_profile_param_in_its_step_experiment(storage):
    with track("probe", "quick", storage, "v1", overrides={"windows_per_track": 5}) as run:
        pass
    record, logged = finished_run(storage, run)
    assert logged.data.params["windows_per_track"] == "5"
    assert record["experiment"] == "probe"
    assert mlflow.get_experiment(logged.info.experiment_id).name == "probe"
