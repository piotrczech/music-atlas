import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from music_atlas.representation import extract
from music_atlas.settings import profile

VERSION = profile("quick")["dataset"]["split"]


def wav(seconds, level):
    buffer = io.BytesIO()
    sf.write(buffer, np.full(seconds * 8000, level), 8000, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def level(samples):
    """A stand-in encoder: two numbers per window."""
    return np.stack([samples.mean(axis=1), samples.std(axis=1)], axis=1)


@pytest.fixture
def data_dir(tmp_path):
    """A quick sample of three readable tracks and one that cannot be decoded."""
    sample = tmp_path / "samples" / VERSION
    sample.mkdir(parents=True)
    files = {3: wav(30, 0.25), 1: wav(30, 0.5), 2: wav(5, 0.1), 9: b"not audio"}
    with tarfile.open(sample / "audio.tar", "w") as tar:
        for track_id, data in files.items():
            member = tarfile.TarInfo(f"{track_id}.wav")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    tracks = [{"track_id": track_id, "artist_id": track_id, "genre_ids": [2],
               "audio_path": f"{track_id}.wav"} for track_id in files]
    pinned = {"version": f"{VERSION}-abc", "manifest_sha256": "abc", "scored_genres": [2],
              "tracks": tracks}
    (sample / "sample.json").write_text(json.dumps(pinned), encoding="utf-8")
    (sample / "rejects.csv").write_text("track_id,error\n", encoding="utf-8")
    return tmp_path


def test_tracks_become_aligned_mean_vectors_with_every_window_kept(data_dir, tmp_path):
    out = tmp_path / "run"
    result = extract(data_dir, out, level, 16000, {"encoder": "level"}, windows_per_track=3)

    assert result["encoded"] == 3 and result["failed"] == 1
    assert np.load(out / "track_ids.npy").tolist() == [1, 2, 3]
    # The 5 s track gets one 10 s window, half of it zero padding, so its mean halves.
    np.testing.assert_allclose(np.load(out / "vectors.npy")[:, 0], [0.5, 0.05, 0.25], rtol=1e-4)
    windows = pd.read_parquet(out / "windows.parquet")
    assert windows.groupby("track_id").size().to_dict() == {1: 3, 2: 1, 3: 3}
    assert np.load(out / "window_vectors.npy").shape == (7, 2)
    assert pd.read_csv(out / "failed.csv")["track_id"].tolist() == [9]
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["encoder"] == "level" and meta["dimensions"] == 2
    assert meta["settings"]["windows_per_track"] == 3 and meta["settings"]["sample_rate"] == 16000


def test_a_failed_job_resumes_and_refuses_other_settings(data_dir, tmp_path):
    out = tmp_path / "run"

    def crash(samples):
        raise RuntimeError("job killed")

    result = extract(data_dir, out, crash, 16000, {"encoder": "level"})
    assert result["parts_done"] == 0 and "RuntimeError" in result["failed"]["audio"]

    assert extract(data_dir, out, level, 16000, {"encoder": "level"})["encoded"] == 3
    with pytest.raises(ValueError, match="other settings"):
        extract(data_dir, out, level, 16000, {"encoder": "level"}, windows_per_track=5)
