import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from music_atlas.logmel import log_mel
from music_atlas.window_study import CHUNK, run, track_log_mel

RATE = 22050


def wav(y):
    buffer = io.BytesIO()
    sf.write(buffer, y, RATE, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def quick_sample(root):
    """A 40 s track of noise and clicks with a tone in its middle 4 s, and a 3 s track."""
    t = np.arange(40 * RATE) / RATE
    rng = np.random.default_rng(0)
    y = 0.01 * rng.standard_normal(len(t))
    y[(t % 1) < 0.02] += 0.5 * rng.standard_normal(int(((t % 1) < 0.02).sum()))  # a click a second
    y[(t >= 18) & (t < 22)] += 0.5 * np.sin(2 * np.pi * 440 * t[(t >= 18) & (t < 22)])
    sample = root / "samples" / "fma-cap10-v1"
    sample.mkdir(parents=True)
    with tarfile.open(sample / "audio.tar", "w") as tar:
        for name, audio in (("000/000001.wav", y), ("000/000002.wav", y[: 3 * RATE])):
            data = wav(audio.astype(np.float32))
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tracks = [{"track_id": i, "artist_id": i, "genre_ids": [2], "audio_path": f"000/00000{i}.wav"}
              for i in (1, 2)]
    (sample / "sample.json").write_text(json.dumps({
        "version": "fma-cap10-v1-abc", "manifest_sha256": "abc", "scored_genres": [2],
        "genre_names": {"2": "International"}, "tracks": tracks,
    }))
    (sample / "rejects.csv").write_text("track_id,error\n")


def test_chunked_track_transform_matches_one_call():
    y = 0.1 * np.random.default_rng(1).standard_normal(2 * CHUNK + 12345).astype(np.float32)
    np.testing.assert_allclose(track_log_mel(y), log_mel(y), atol=1e-4)


def test_longer_windows_see_more_events_and_more_of_the_track(tmp_path):
    quick_sample(tmp_path)
    result = run(tmp_path, tmp_path / "outputs")
    assert (result["tracks"], result["rejected"]) == (2, 0)
    table = pd.read_csv(tmp_path / "outputs" / "window-study" / "results.csv")
    assert table["frames"].tolist() == [216, 431, 862]
    assert table["tensor_kib"].tolist() == [108, 215.5, 431]
    # The 3 s track is padded at every length; neither window is silent.
    assert table["padded_tracks"].tolist() == [1, 1, 1] and table["silent_windows"].eq(0).all()

    windows = pd.read_csv(tmp_path / "outputs" / "window-study" / "tracks.csv")
    long = windows[windows["track_id"].eq(1)].set_index("window_seconds")
    # Centred windows share the 20 s mark and catch one click a second; the tone fills most of
    # the 5 s window and a fifth of the 20 s one, so the longer window is nearer the whole track.
    assert ((long["start_seconds"] + long.index / 2).round(2) == 20).all()
    assert long["onsets"].tolist() == [5, 10, 20]
    assert long["gap_db"].is_monotonic_decreasing
    assert long.loc[5, "gap_db"] > 2 * long.loc[20, "gap_db"]
    short = windows[windows["track_id"].eq(2)]
    assert (short["gap_db"] < 0.01).all()  # every window holds the whole 3 s track
    assert (tmp_path / "outputs" / "window-study" / "examples.png").stat().st_size > 0


def test_tracked_study_logs_results_by_length(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from music_atlas.tracking import track

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    quick_sample(tmp_path)
    storage = {"data": tmp_path, "artifacts": tmp_path / "artifacts",
               "outputs": tmp_path / "outputs"}
    with track("window-study", "quick", storage) as record:
        run(tmp_path, storage["outputs"], record)
    saved = json.loads((record.directory / "run.json").read_text())
    assert set(saved["variants"]) == {"w5", "w10", "w20"}
    logged = mlflow.get_run(saved["mlflow_run_id"])
    assert {"between_tracks_gap_db", "rejected"} <= set(logged.data.metrics)
    assert logged.data.params["window_seconds"] == "10"  # the profile's, unless overridden
    files = {item.path for item in mlflow.artifacts.list_artifacts(
        run_id=logged.info.run_id, artifact_path="evaluation")}
    assert files == {"evaluation/results.csv", "evaluation/examples.png"}

    # One child run per length, comparable column by column in the MLflow UI.
    children = mlflow.search_runs(
        experiment_ids=[logged.info.experiment_id], output_format="list",
        filter_string=f"tags.mlflow.parentRunId = '{logged.info.run_id}'")
    assert sorted(int(child.data.params["window_seconds"]) for child in children) == [5, 10, 20]
    assert all({"own_track_share", "decode_ms", "gap_db"} <= set(child.data.metrics)
               for child in children)
    assert {child.data.params["profile"] for child in children} == {"quick"}
