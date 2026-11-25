import io
import json
import tarfile

import librosa
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from music_atlas.loader import Window
from music_atlas.logmel import (
    FLOOR_DB,
    HOP,
    N_MELS,
    log_mel,
    mel_filters,
    normalise,
    track_tensor,
    valid_frames,
)
from music_atlas.logmel_cache import read, run
from music_atlas.split_views import digest

RATE = 22050


def window(samples, decoded_seconds=None, rate=RATE):
    seconds = len(samples) / rate if decoded_seconds is None else decoded_seconds
    return Window(1, 0, 0.0, seconds, rate, np.asarray(samples, np.float32))


def test_transform_matches_librosa():
    t = np.arange(10 * RATE) / RATE
    rng = np.random.default_rng(0)
    y = (0.3 * np.sin(2 * np.pi * (100 + 400 * t) * t) + 0.01 * rng.standard_normal(len(t)))
    y = y.astype(np.float32)
    ours = log_mel(y)
    power = librosa.feature.melspectrogram(
        y=y, sr=RATE, n_fft=2048, hop_length=HOP, n_mels=N_MELS, fmin=0, fmax=RATE / 2,
        power=2.0, center=True, pad_mode="constant",
    )
    theirs = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)
    assert ours.shape == theirs.shape == (N_MELS, 431) and ours.dtype == np.float32
    np.testing.assert_allclose(ours, theirs, atol=0.02)
    reference = librosa.filters.mel(sr=RATE, n_fft=2048, n_mels=N_MELS, fmin=0, fmax=RATE / 2)
    np.testing.assert_allclose(mel_filters(), reference, atol=1e-7)


def test_tone_lands_in_its_band_and_padding_sits_at_the_floor():
    t = np.arange(4 * RATE) / RATE
    samples = np.zeros(10 * RATE, np.float32)
    samples[: len(t)] = 0.5 * np.sin(2 * np.pi * 440 * t)
    short = window(samples, decoded_seconds=4)
    x, valid = track_tensor([short])
    assert x.shape == (1, 1, N_MELS, 431)
    assert valid.tolist() == [valid_frames(short)] == [-(-4 * RATE // HOP)]
    band = mel_filters()[:, round(440 * 2048 / RATE)].argmax()
    assert abs(int(x[0, 0, :, 50].argmax()) - band) <= 1
    # Frames that only see zero padding are exactly the clipping floor, like digital silence.
    assert (x[0, 0, :, valid[0] + 2 :] == FLOOR_DB).all()
    assert (log_mel(np.zeros(RATE, np.float32)) == FLOOR_DB).all()
    with pytest.raises(ValueError, match="22050"):
        track_tensor([window(samples, rate=16000)])


def write_tar(path, members):
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def wav(amplitude, seconds=12):
    buffer = io.BytesIO()
    noise = np.random.default_rng(1).uniform(-amplitude, amplitude, seconds * RATE)
    sf.write(buffer, noise, RATE, format="WAV")
    return buffer.getvalue()


def make_split(root):
    """Five tracks in two shards; track 5 is not an audio file."""
    ready = root / "ready" / "fma-cap10-v1"
    ready.mkdir(parents=True)
    (root / "fma").mkdir()
    rows = pd.DataFrame({
        "track_id": [1, 2, 3, 4, 5], "split": ["train", "train", "validation", "test", "train"],
        "shard": ["fma/000.tar"] * 3 + ["fma/001.tar"] * 2, "valid": True,
        "artist_id": [1, 2, 3, 4, 5], "genre_ids": [[2], [3], [2], [3], [2]],
    })
    rows["audio_path"] = [f"{shard[4:7]}/{track:06d}.wav"
                          for shard, track in zip(rows["shard"], rows["track_id"], strict=True)]
    write_tar(root / "fma/000.tar", {"000/000001.wav": wav(0.1), "000/000002.wav": wav(0.1),
                                     "000/000003.wav": wav(0.8)})
    write_tar(root / "fma/001.tar", {"001/000004.wav": wav(0.8), "001/000005.wav": b"\0" * 64})
    rows.to_parquet(ready / "tracks.parquet")
    for split, group in rows.groupby("split"):
        group.to_parquet(ready / f"{split}.parquet")
    summary = {"version": "fma-cap10-v1", "manifest_sha256": digest(ready / "tracks.parquet")}
    (ready / "summary.json").write_text(json.dumps(summary))
    (ready / "labels.json").write_text(json.dumps([
        {"genre_id": 2, "name": "International", "scored": True},
        {"genre_id": 3, "name": "Blues", "scored": True},
    ]))
    return ready, summary


def test_cache_covers_the_three_splits_resumes_and_fits_the_normaliser_on_train(tmp_path):
    ready, summary = make_split(tmp_path)
    artifacts = tmp_path / "artifacts"
    pilot = run(tmp_path, artifacts, "full", workers=1, limit=1)
    assert (pilot["parts"], pilot["parts_cached"], pilot["failed"]) == (2, 1, {})
    result = run(tmp_path, artifacts, "full", workers=1)
    assert (result["tracks"], result["windows"], result["rejected"]) == (5, 12, 1)

    directory = artifacts / "logmel" / "fma-cap10-v1-full"
    index, arrays, metadata = read(directory)
    assert metadata["counts"]["train"] == {"tracks": 2, "windows": 6, "short_windows": 0}
    assert index.groupby("split")["track_id"].nunique().to_dict() == {
        "test": 1, "train": 2, "validation": 1,
    }
    assert index["window"].tolist()[:3] == [0, 1, 2] and (index["valid_frames"] == 431).all()
    assert (index["floor_share"] == 0).all() and index["mean_db"].between(-60, 20).all()
    assert arrays["000"].shape == (9, N_MELS, 431) and arrays["000"].dtype == np.float16
    assert (directory / "examples.png").stat().st_size > 0
    assert pd.read_csv(directory / "rejects.csv")["track_id"].tolist() == [5]
    selection = pd.read_parquet(directory / "filter.parquet")
    assert selection.set_index("track_id")["included"].to_dict() == {
        1: True, 2: True, 3: True, 4: True, 5: False,
    }

    # Louder validation and test tracks do not move the training statistics.
    train = np.concatenate([arrays["000"][i] for i in range(6)], axis=1).astype(np.float64)
    # The statistics come from float32 values; the cache holds float16.
    np.testing.assert_allclose(metadata["normaliser"]["mean"], train.mean(axis=1), atol=0.05)
    np.testing.assert_allclose(metadata["normaliser"]["std"], train.std(axis=1), atol=0.05)
    z = normalise(arrays["000"][:6], metadata["normaliser"])
    assert abs(z.mean()) < 0.01

    (ready / "summary.json").write_text(json.dumps({**summary, "manifest_sha256": "changed"}))
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path, artifacts, "full", workers=1)


def test_unfinished_tracked_cache_fails_its_run_and_resumes_in_place(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    from music_atlas.tracking import track

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    make_split(tmp_path)
    storage = {"data": tmp_path, "artifacts": tmp_path / "artifacts",
               "outputs": tmp_path / "outputs"}
    mlflow.set_tracking_uri(f"file:{storage['outputs'] / 'mlruns'}")
    with pytest.raises(ValueError, match="--resume"), \
            track("log-mel", "full", storage, representation="logmel") as pilot:
        run(tmp_path, storage["artifacts"], "full", workers=1, limit=1, record=pilot)
    assert json.loads((pilot.directory / "run.json").read_text())["status"] == "FAILED"

    with track("log-mel", "full", storage, representation="logmel",
               resume=pilot.run_id) as record:
        result = run(tmp_path, storage["artifacts"], "full", workers=1, record=record)
    assert record.directory == pilot.directory and result["windows"] == 12
    logged = mlflow.get_run(json.loads((record.directory / "run.json").read_text())
                            ["mlflow_run_id"])
    assert logged.info.status == "FINISHED"
    assert {"rejected", "seconds_per_window/decode", "seconds_per_window/transform",
            "storage_kib_per_track"} <= set(logged.data.metrics)
    names = {entry.dataset.name for entry in logged.inputs.dataset_inputs}
    assert names == {f"fma-cap10-v1/{split}" for split in ("train", "validation", "test")}
    files = {item.path for folder in ("dataset", "run")
             for item in mlflow.artifacts.list_artifacts(run_id=logged.info.run_id,
                                                         artifact_path=folder)}
    assert {"dataset/examples.png", "dataset/rejects.csv", "run/metadata.json"} <= files


def test_quick_profile_caches_the_quick_sample(tmp_path):
    sample = tmp_path / "samples" / "fma-cap10-v1"
    sample.mkdir(parents=True)
    write_tar(sample / "audio.tar", {"000/000001.wav": wav(0.1)})
    tracks = [{"track_id": 1, "artist_id": 1, "genre_ids": [2], "audio_path": "000/000001.wav"},
              {"track_id": 2, "artist_id": 2, "genre_ids": [3], "audio_path": "000/000002.wav"}]
    pinned = {"version": "fma-cap10-v1-abc", "manifest_sha256": "abc", "scored_genres": [2, 3],
              "tracks": tracks}
    (sample / "sample.json").write_text(json.dumps(pinned))
    (sample / "rejects.csv").write_text("track_id,error\n2,decode_error\n")
    result = run(tmp_path, tmp_path / "artifacts", "quick", workers=1)
    assert (result["tracks"], result["windows"], result["rejected"]) == (1, 1, 0)
    metadata = read(tmp_path / "artifacts" / "logmel" / "fma-cap10-v1-quick")[2]
    assert (metadata["version"], metadata["windows"]["windows_per_track"]) == (
        "fma-cap10-v1-abc", 1,
    )
    other = {**pinned, "version": "fma-cap10-v1-abd", "manifest_sha256": "abd"}
    (sample / "sample.json").write_text(json.dumps(other))
    with pytest.raises(ValueError, match="other settings"):
        run(tmp_path, tmp_path / "artifacts", "quick", workers=1)
