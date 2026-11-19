import json

import pytest

from music_atlas.tracking import track

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


def test_completed_run_records_dataset_counts_and_status(storage):
    with track("probe", "quick", storage, "v1") as run:
        run.log({"tracks": 4, "seconds": 1.5, "source": "archive", "flag": True, "peak": None})
    record, logged = finished_run(storage, run)
    assert run.directory == storage["outputs"] / run.run_id
    assert record["status"] == logged.info.status == "FINISHED"
    assert logged.data.params["manifest_sha256"] == "abc"
    assert logged.data.params["seed"] == "42" and logged.data.params["source"] == "archive"
    assert logged.data.metrics["tracks"] == 4 and "elapsed_seconds" in logged.data.metrics
    assert logged.data.tags["atlas.step"] == "probe"


def test_failed_run_stays_distinguishable(storage):
    with pytest.raises(ValueError), track("probe", "quick", storage, "v1", "features") as run:
        raise ValueError("broken")
    record, logged = finished_run(storage, run)
    assert run.directory == storage["artifacts"] / "features" / run.run_id
    assert record["status"] == logged.info.status == "FAILED"
