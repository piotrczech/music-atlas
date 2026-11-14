import hashlib
import io
import tarfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from music_atlas import audit
from music_atlas.manifest import build

RATE = 22050


def mp3(seconds, seed=1, gain=0.5, **options):
    """A 440 Hz tone whose loudness changes every second, so contours can be compared."""
    t = np.arange(round(seconds * RATE)) / RATE
    steps = np.random.default_rng(seed).uniform(0.05, 1, int(seconds) + 1)
    buffer = io.BytesIO()
    sf.write(buffer, gain * steps[t.astype(int)] * np.sin(2 * np.pi * 440 * t), RATE,
             format="MP3", **options)
    return buffer.getvalue()


@pytest.mark.skipif("MP3" not in sf.available_formats(), reason="libsndfile lacks MP3 support")
def test_audit_explains_every_track_and_feeds_the_split(tmp_path):
    original = mp3(20)
    audio = {
        1: original,
        2: original,  # byte-identical: confirmed duplicate
        3: mp3(20, compression_level=0.9),  # same audio, other bitrate: candidate only
        4: b"ID3\3\0\0\0\0\0\0TT2\0\0\4\0Song" + mp3(8, seed=2),  # leftover tag libsndfile rejects
        5: b"not audio",
        6: mp3(5, gain=0),
        7: mp3(0.5),
        9: mp3(12, seed=3, gain=0.005),
    }
    shard = tmp_path / "fma/fma_full/000.tar"
    shard.parent.mkdir(parents=True)
    with tarfile.open(shard, "w") as tar:
        for track, data in audio.items():
            entry = tarfile.TarInfo(f"000/{track:06d}.mp3")
            entry.size = len(data)
            tar.addfile(entry, io.BytesIO(data))
    ids = pd.Index(range(1, 10), name="track_id")
    (tmp_path / "manifests").mkdir()
    pd.DataFrame({
        "shard": "fma/fma_full/000.tar", "audio_path": [f"000/{i:06d}.mp3" for i in ids],
        "sha1": [hashlib.sha1(audio.get(i, b"lost")).hexdigest() for i in ids],
        "status": ["missing" if i == 8 else "present" for i in ids],
    }, index=ids).to_csv(tmp_path / "manifests/fma_full_inventory.csv")
    metadata = tmp_path / "fma/fma_metadata"
    metadata.mkdir(parents=True)
    pd.DataFrame({"genre_id": [1, 2], "parent": 0, "title": ["Rock", "Pop"]}).to_csv(
        metadata / "genres.csv", index=False)
    pd.DataFrame({
        ("artist", "id"): [1, 1, 2, 3, 4, 5, 6, 7, 8], ("album", "id"): [1] * 8 + [-1],
        ("track", "duration"): [20, 20, 20, 8, 3, 5, 0, 30, 100],
        ("track", "genre_top"): ["Rock"] * 3 + [None, "Pop", "Pop", None, "Rock", None],
        ("track", "genres"): ["[1]"] * 3 + ["[1, 2]", "[2]", "[2]", "[]", "[1]", "[]"],
        ("track", "genres_all"): ["[1]"] * 3 + ["[1, 2]", "[2]", "[2]", "[]", "[1]", "[]"],
        ("track", "license"): ["Attribution-ShareAlike", "CC0 1.0 Universal",
                               "Attribution-NonCommercial", "Attribution 3.0 United States",
                               *["Attribution"] * 4, "Attribution-NoDerivatives"],
        ("set", "split"): "training",
    }, index=ids).to_csv(metadata / "tracks.csv")

    summary = audit.run(tmp_path, workers=1)
    table = pd.read_parquet(summary["audit"]).set_index("track_id")
    assert table["error"].fillna("").tolist() == [
        "", "", "", "", "decode_error", "silent", "too_short", "inventory_missing", "",
    ]
    assert table["duplicate_group"].fillna(0).tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert table["reencoding_candidate"].fillna(0).tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert table["warning"].fillna("").tolist() == [""] * 3 + ["short"] + [""] * 4 + [
        "duration_mismatch;quiet"]
    assert table.index[table["listening_example"]].tolist() == [1, 2, 4]
    assert table.at[4, "duration_seconds"] == pytest.approx(8, abs=0.1)
    assert summary["annotations"]["multiple_top_genres"] == 1
    assert summary["genres"]["Rock"] == {"tracks": 4, "valid": 4, "artists": 3,
                                         "valid_artists": 3, "listening_examples": 3}
    assert build(tmp_path, tmp_path / "manifests/fma_audio_audit.parquet")["eligible"] == 5
