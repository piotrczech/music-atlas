import numpy as np
import pandas as pd
import pytest

from music_atlas.shards import process_shards, save


def count_tracks(tar, rows, part):
    """A stand-in step: one number per shard, failing on the shard named `bad`."""
    if tar.endswith("bad.tar"):
        raise OSError("cannot open")
    save(part.with_suffix(".npz"), lambda file: np.savez(file, tracks=len(rows)))


ROWS = pd.DataFrame({"track_id": [1, 2, 3, 4], "tar": ["a.tar", "a.tar", "b.tar", "bad.tar"]})


@pytest.mark.parametrize("workers", [1, 2])
def test_every_shard_gets_one_part_and_failures_are_reported(tmp_path, workers):
    finished, failed = process_shards(ROWS, tmp_path, count_tracks, workers)
    assert finished == ["a", "b"] and failed == {"bad": "OSError: cannot open"}
    assert int(np.load(tmp_path / "a.npz")["tracks"]) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_rerun_skips_finished_shards_and_limit_stops_early(tmp_path):
    assert process_shards(ROWS, tmp_path, count_tracks, limit=1) == (["a"], {})
    assert process_shards(ROWS, tmp_path, count_tracks)[0] == ["b"]
