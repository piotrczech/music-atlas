import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from music_atlas.representation import extract, read_tracks

SETTINGS = {"sample_rate": 16000, "window_seconds": 1, "windows_per_track": 3}


def add(tar, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, io.BytesIO(data))


def wav(seconds, level):
    buffer = io.BytesIO()
    sf.write(buffer, np.full(seconds * 8000, level), 8000, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def level(samples):
    """A stand-in encoder: two numbers per window."""
    return np.stack([samples.mean(axis=1), samples.std(axis=1)], axis=1)


@pytest.fixture
def data_dir(tmp_path):
    for shard, tracks in {"a": {3: 0.25, 1: 0.5}, "b": {2: 0.1}}.items():
        with tarfile.open(tmp_path / f"{shard}.tar", "w") as tar:
            for track_id, value in tracks.items():
                add(tar, f"{track_id}.wav", wav(5, value))
            add(tar, "9.wav", b"not audio")
    return tmp_path


def subsets():
    rows = [(1, "a.tar"), (2, "b.tar"), (3, "a.tar"), (9, "b.tar")]
    tracks = pd.DataFrame(rows, columns=["track_id", "shard"])
    tracks["audio_path"] = tracks["track_id"].map("{}.wav".format)
    return {"train": tracks.iloc[:3], "test": tracks.iloc[3:]}


def test_tracks_of_all_subsets_become_aligned_vectors_and_failures(data_dir, tmp_path):
    out = tmp_path / "run"
    result = extract(data_dir, subsets(), out, level, SETTINGS, {"encoder": "level"})

    assert result["encoded"] == 3 and result["failed"] == 1
    assert np.load(out / "track_ids.npy").tolist() == [1, 2, 3]
    np.testing.assert_allclose(np.load(out / "vectors.npy")[:, 0], [0.5, 0.1, 0.25], rtol=1e-4)
    assert pd.read_csv(out / "failed.csv")["track_id"].tolist() == [9]
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["encoder"] == "level" and meta["dimensions"] == 2
    assert meta["subsets"] == ["train", "test"] and meta["settings"] == SETTINGS


def test_resumed_extraction_skips_finished_shards(data_dir, tmp_path):
    out = tmp_path / "run"
    calls = []

    def crash_on_second_shard(samples):
        calls.append(len(samples))
        if len(calls) == 2:
            raise RuntimeError("job killed")
        return level(samples)

    with pytest.raises(RuntimeError):
        extract(data_dir, subsets(), out, crash_on_second_shard, SETTINGS, {})
    assert [part.name for part in (out / "parts").iterdir()] == ["a.npz"]

    calls.clear()
    result = extract(data_dir, subsets(), out, crash_on_second_shard, SETTINGS, {})
    # Shard a is kept, so only shard b's one readable track (three windows) is encoded now.
    assert calls == [3]
    assert np.load(out / "track_ids.npy").tolist() == [1, 2, 3]
    assert result["encoded"] == 3


def test_quick_profile_reads_the_sample_archive(tmp_path):
    sample = tmp_path / "samples" / "v1"
    sample.mkdir(parents=True)
    tracks = [{"track_id": 5, "artist_id": 1, "genre_ids": [2], "audio_path": "5.mp3"}]
    (sample / "sample.json").write_text(json.dumps({"tracks": tracks}), encoding="utf-8")

    [(name, rows)] = read_tracks(tmp_path, "v1", "quick").items()
    assert name == "quick-sample"
    assert rows[["track_id", "shard"]].values.tolist() == [[5, "samples/v1/audio.tar"]]
