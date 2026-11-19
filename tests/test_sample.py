import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from music_atlas.sample import run, select
from music_atlas.split_views import digest


def test_selection_is_seeded_and_spreads_over_genres_and_artists():
    train = pd.DataFrame({
        "track_id": range(1, 41),
        "artist_id": [i % 10 for i in range(40)],
        "genre_ids": [[2]] * 30 + [[3]] * 8 + [[], [2]],
        "valid": [True] * 39 + [False],
    })
    picked = select(train, [2, 3], seed=42, limit=8)
    assert len(picked) == 8 and picked["artist_id"].nunique() == 8
    assert picked["genre_ids"].map(lambda genres: 3 in genres).sum() == 4
    shuffled = train.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(picked, select(shuffled, [2, 3], seed=42, limit=8))
    # Invalid and unlabelled tracks never enter the sample.
    assert set(select(train, [2, 3], seed=42, limit=100)["track_id"]) == set(range(1, 39))


def test_sample_is_read_from_shards_once_then_from_its_archive(tmp_path):
    ready = tmp_path / "ready" / "v1"
    ready.mkdir(parents=True)
    shard = tmp_path / "fma" / "000.tar"
    shard.parent.mkdir()
    train = pd.DataFrame({
        "track_id": [1, 2, 3, 4], "shard": "fma/000.tar",
        "audio_path": [f"000/{i:06d}.wav" for i in range(1, 5)],
        "artist_id": [1, 2, 3, 4], "genre_ids": [[2], [3], [2], [3]], "valid": True,
    })
    sf.write(tmp_path / "track.wav", np.full(12 * 22050, 0.1), 22050)
    with tarfile.open(shard, "w") as tar:
        for track in (1, 2):
            tar.add(tmp_path / "track.wav", arcname=f"000/00000{track}.wav")
        broken = tarfile.TarInfo("000/000003.wav")
        broken.size = 64
        tar.addfile(broken, io.BytesIO(b"\0" * 64))
    # Track 3 cannot be decoded and track 4 is missing from its shard.
    train.to_parquet(ready / "train.parquet")
    train.to_parquet(ready / "tracks.parquet")
    summary = {"version": "v1", "manifest_sha256": digest(ready / "tracks.parquet")}
    (ready / "summary.json").write_text(json.dumps(summary))
    (ready / "labels.json").write_text(json.dumps([
        {"genre_id": 2, "name": "International", "scored": True},
        {"genre_id": 3, "name": "Blues", "scored": True},
    ]))

    first = run(tmp_path, "v1")
    assert (first["source"], first["tracks"], first["loaded"], first["rejected"]) == (
        "shards", 4, 2, 2,
    )
    assert first["scored_genres_covered"] == 2
    target = tmp_path / "samples" / "v1"
    rejects = pd.read_csv(target / "rejects.csv")
    assert list(rejects.columns) == ["track_id", "error"]
    assert sorted(rejects["track_id"]) == [3, 4]
    with tarfile.open(target / "audio.tar") as tar:
        assert sorted(tar.getnames()) == ["000/000001.wav", "000/000002.wav"]
    sample = json.loads((target / "sample.json").read_text())
    assert (sample["version"], sample["profile"], sample["seed"]) == ("v1", "quick", 42)
    assert sample["genre_names"] == {"2": "International", "3": "Blues"}

    shard.unlink()  # Local runs need only the sample directory.
    again = run(tmp_path, "v1")
    assert (again["source"], again["loaded"], again["rejected"]) == ("archive", 2, 2)

    (ready / "summary.json").write_text(json.dumps({**summary, "manifest_sha256": "changed"}))
    with pytest.raises(ValueError, match="drawn from v1"):
        run(tmp_path, "v1")
