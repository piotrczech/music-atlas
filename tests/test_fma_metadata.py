import json

import pytest

pd = pytest.importorskip("pandas")
metadata = pytest.importorskip("music_atlas.fma_metadata")


def test_metadata_report_counts_tracks_once_and_reruns_after_audit(tmp_path):
    data = tmp_path / "data"
    fma = data / "fma/fma_metadata"
    manifests = data / "manifests"
    fma.mkdir(parents=True)
    manifests.mkdir()
    pd.DataFrame({"track_id": [4, 3, 2, 1, 5], "status": ["present"] * 4 + ["missing"]}).to_csv(
        manifests / "fma_full_inventory.csv", index=False
    )
    pd.DataFrame(
        {
            ("track", "genres"): ["[1, 2, 2]", "[2, 3]", "[]", "[3]"],
            ("track", "genre_top"): ["Rock", None, None, "Jazz"],
            ("track", "duration"): [30, 60, 0, 100],
            ("artist", "id"): [10, 10, 20, 50],
            ("album", "id"): [-1, 22, 23, 25],
            ("set", "split"): ["training", "test", "test", "training"],
        },
        index=pd.Index([1, 2, 3, 5], name="track_id"),
    ).to_csv(fma / "tracks.csv")
    (fma / "genres.csv").write_text(
        "genre_id,parent,title\n1,0,Rock\n2,1,Punk\n3,0,Jazz\n4,2,Unused\n",
        encoding="utf-8",
    )

    baseline = tmp_path / "baseline"
    result = metadata.explore(data, baseline)
    assert result["analysed_tracks"] == 4
    assert result["selected_without_metadata"] == 1
    assert result["coverage"]["No direct genre"] == 2
    assert result["coverage"]["Missing album ID"] == 2
    assert result["coverage"]["Multiple top-level branches"] == 1
    assert result["duration_seconds"]["median"] == 45
    assert result["published_split"]["artists_in_multiple_splits"] == 1

    support = pd.read_csv(baseline / "genre_support.csv", index_col="genre_id")
    assert support.loc[1, ["direct_tracks", "inclusive_tracks", "inclusive_artists"]].tolist() == [
        1,
        2,
        1,
    ]
    assert support.loc[4, "path"] == "Rock / Punk / Unused"
    assert support.loc[4, "inclusive_tracks"] == 0
    pairs = pd.read_csv(baseline / "genre_cooccurrence.csv")
    assert pairs[["genre_id_a", "genre_id_b", "tracks"]].values.tolist() == [[1, 2, 1], [2, 3, 1]]
    assert pairs.jaccard.tolist() == [0.5, 0.5]
    assert len(list(baseline.glob("*.png"))) == 5

    audit = manifests / "tracks.parquet"
    pd.DataFrame({"track_id": [1, 2, 4], "valid": [False, True, False]}).to_parquet(audit)
    after = tmp_path / "after"
    metadata.explore(data, after, audit)
    summary = json.loads((after / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysed_tracks"] == 1
    assert summary["present_rejected"] == 2
    assert summary["present_unlisted_in_manifest"] == 1
    assert summary["duration_seconds"]["median"] == 60
    assert len(summary["input_sha256"]["manifest"]) == 64
    assert (after / "interpretation.md").is_file()
    with pytest.raises(FileExistsError):
        metadata.explore(data, after, audit)
