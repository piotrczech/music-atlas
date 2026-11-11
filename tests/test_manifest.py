import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from music_atlas.manifest import SPLITS, assign_groups, build, prepare
from music_atlas.split_views import record_filter, write_balanced_eval


@pytest.fixture
def catalogue():
    ids = pd.Index(range(1, 19), name="track_id")
    inventory = pd.DataFrame({
        "shard": "fma/fma_full/000.tar",
        "audio_path": [f"000/{i:06d}.mp3" for i in ids],
        "sha1": [f"{i:040x}" for i in ids],
        "status": "present",
    }, index=ids)
    inventory.loc[2, "sha1"] = inventory.loc[1, "sha1"]
    metadata = pd.DataFrame({
        "artist_id": [1, 2, 3, 3, *range(4, 17), None],
        "album_id": 1,
        "genres": ["[12, 2]", "[12]", "[2]", "[12, 3]", *(["[12]", "[2]"] * 7)],
        "duration_seconds": 90,
        "published_split": ["training", "validation", "test"] * 6,
    }, index=ids)
    genres = pd.DataFrame({"title": ["Rock", "Pop", "Jazz", "Punk", "Post-Punk"],
                           "parent": [0, 0, 0, 1, 11]},
                          index=pd.Index([1, 2, 3, 11, 12], name="genre_id"))
    audit = pd.DataFrame({"valid": True, "error": None, "duration_seconds": 91.0,
                          "duplicate_group": None}, index=ids)
    audit.loc[[2, 3], "duplicate_group"] = "confirmed"
    audit.loc[2, ["valid", "error"]] = [False, "decode_error"]
    return inventory, metadata, genres, audit


def test_groups_are_transitive_and_labels_are_frozen(catalogue):
    manifest, mapping, support, report = prepare(*catalogue)
    by_id = manifest.set_index("track_id")
    assert by_id.loc[[1, 2, 3], "duplicate_group"].tolist() == [1, 1, 1]
    assert by_id.loc[[1, 2, 3, 4], "split_group"].tolist() == [1, 1, 1, 1]
    assert by_id.loc[[1, 3, 4], "split"].nunique() == 1
    assert by_id.at[2, "error"] == "decode_error"
    assert pd.isna(by_id.at[2, "split"]) and pd.isna(by_id.at[18, "split"])
    assert by_id.at[1, "genre_ids"] == [1, 2, 11, 12]
    assert by_id.at[1, "duration_seconds"] == 91
    assert [label["genre_id"] for label in mapping] == [1, 2, 3]
    assert not mapping[2]["scored"]
    assert not any(report["overlap"]["canonical"].values())
    assert report["overlap"]["published"]["duplicate_groups"] == 1
    assert set(by_id["split"].dropna()) == set(SPLITS)
    assert set(support["scheme"]) == {"canonical", "group_random", "published"}
    shuffled = tuple(frame.sample(frac=1, random_state=4) for frame in catalogue)
    pd.testing.assert_frame_equal(manifest, prepare(*shuffled)[0])
    if report["baselines"]:
        test = report["baselines"]["test"]["all_negative"]
        assert test["macro_recall"] == 0 and test["macro_balanced_accuracy"] == 0.5


def test_prepare_rejects_an_unsafe_inventory_path(catalogue):
    inventory, *metadata = catalogue
    inventory.loc[1, "audio_path"] = "../outside.mp3"
    with pytest.raises(ValueError, match="safe relative paths in audio_path"):
        prepare(inventory, *metadata)


@pytest.fixture
def catalogue_files(tmp_path, catalogue):
    inventory, metadata, genres, audit = catalogue
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    source = tmp_path / "fma/fma_metadata"
    source.mkdir(parents=True)
    inventory.to_csv(manifests / "fma_full_inventory.csv")
    genres.to_csv(source / "genres.csv")
    metadata.columns = pd.MultiIndex.from_tuples([
        ("artist", "id"), ("album", "id"), ("track", "genres"), ("track", "duration"),
        ("set", "split"),
    ])
    metadata.to_csv(source / "tracks.csv")
    return manifests / "audit.parquet", audit


def test_pre_audit_and_audited_publication_are_distinct(tmp_path, catalogue_files):
    audit_path, audit = catalogue_files
    audit.iloc[:-1].reset_index().to_parquet(audit_path)
    canonical = audit_path.with_name("tracks.parquet")
    with pytest.raises(ValueError, match="missing present track IDs"):
        build(tmp_path, audit_path)

    provisional = build(tmp_path)
    provisional_dir = tmp_path / "ready" / provisional["version"]
    assert provisional["version"].startswith("pre-audit-") and not provisional["audio_audited"]
    assert not canonical.exists()
    provisional_table = pq.read_table(provisional_dir / "tracks.parquet")
    assert provisional_table.schema.metadata[b"audio_audited"] == b"false"
    assert provisional_table["valid"][1].as_py()
    assert provisional_table["duration_seconds"][0].as_py() == 90
    assert sum(pq.read_table(provisional_dir / f"{part}.parquet").num_rows for part in SPLITS) == 17

    alternative = build(tmp_path, strategy="rare-support")
    alternative_dir = tmp_path / "ready" / alternative["version"]
    assert alternative["strategy"] == "rare-support"
    assert alternative_dir != provisional_dir and not canonical.exists()
    groups = [pq.read_table(alternative_dir / f"{part}.parquet") for part in SPLITS]
    assert sorted(track for group in groups for track in group["track_id"].to_pylist()) == list(
        range(1, 18)
    )

    audit.reset_index().to_parquet(audit_path)
    ready = build(tmp_path, audit_path)
    table = pq.read_table(canonical)
    assert table.schema.metadata[b"dataset_version"] == ready["version"].encode()
    assert table.schema.metadata[b"audio_audited"] == b"true"
    assert provisional["version"] != ready["version"]
    assert (provisional_dir / "tracks.parquet").exists()
    assert table["genre_ids"][0].as_py() == [1, 2, 11, 12]
    seen = set()
    for part in SPLITS:
        split = pq.read_table(tmp_path / "ready" / ready["version"] / f"{part}.parquet")
        ids = set(split["track_id"].to_pylist())
        assert ids and not seen.intersection(ids)
        assert np.all(split["split"].to_numpy() == part)
        seen.update(ids)
    assert seen == set(range(1, 19)) - {2, 18}
    alternate_audited = build(tmp_path, audit_path, strategy="rare-support")
    assert alternate_audited["audio_audited"] and alternate_audited["version"] != ready["version"]
    assert pq.read_metadata(canonical).metadata[b"dataset_version"] == ready["version"].encode()
    assert build(tmp_path, audit_path) == ready

    exclusions = tmp_path / "exclusions.csv"
    exclusions.write_text("track_id,error\n1,representation_failure\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    record = record_filter(tmp_path, artifacts, ready["version"], "mfcc", "pilot", exclusions)
    assert record["excluded"] == 1 and record["included"] == len(seen) - 1
    view = pq.read_table(record["filter"])
    assert view.schema.metadata[b"manifest_sha256"].decode() == ready["manifest_sha256"]
    excluded = view.to_pandas().set_index("track_id").loc[1]
    assert excluded["error"] == "representation_failure" and not excluded["included"]
    assert excluded["split"] == table["split"][0].as_py()
    repeated = record_filter(tmp_path, artifacts, ready["version"], "mfcc", "pilot", exclusions)
    assert repeated == record
    if os.name == "posix":
        assert ready["short_version"] == "fma-v1"
        assert record_filter(tmp_path, artifacts, "fma-v1", "mfcc", "pilot", exclusions) == record
    exclusions.write_text("track_id,error\n1,changed_reason\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Existing view differs"):
        record_filter(tmp_path, artifacts, ready["version"], "mfcc", "pilot", exclusions)


def test_rerun_rejects_incomplete_or_changed_version(tmp_path, catalogue_files):
    version = build(tmp_path)
    directory = Path(version["directory"])
    validation = directory / "validation.parquet"
    original = validation.read_bytes()
    validation.unlink()
    with pytest.raises(ValueError, match=r"Missing or changed .*validation\.parquet"):
        build(tmp_path)

    validation.write_bytes(original)
    (directory / "labels.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Missing or changed .*labels\.json"):
        build(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="Short version links are published on POSIX")
def test_short_version_stays_pinned_to_original_inputs(tmp_path, catalogue_files):
    first = build(tmp_path)
    assert first["short_version"] == "pre-audit-v1"
    alias = tmp_path / "ready" / first["short_version"]
    tracks = tmp_path / "fma/fma_metadata/tracks.csv"
    with tracks.open("a", encoding="utf-8") as stream:
        stream.write("\n")

    second = build(tmp_path)
    assert second["version"] != first["version"]
    assert second["short_version"] is None
    assert alias.resolve() == Path(first["directory"]).resolve()


def test_balanced_eval_preserves_splits_and_uses_independent_groups(tmp_path):
    frame = pd.DataFrame({
        "track_id": range(1, 13),
        "split": ["validation"] * 6 + ["test"] * 6,
        "split_group": [1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10],
        "genre_ids": [[1, 2], [1], [1], [2], [2], []] * 2,
    })
    mapping = [{"genre_id": genre, "scored": True} for genre in (1, 2)]
    assert write_balanced_eval(frame, mapping, tmp_path, "fixture-v1", "fixture", 42) == 2
    table = pq.read_table(tmp_path / "balanced_eval.parquet")
    assert table.schema.metadata[b"dataset_version"] == b"fixture-v1"
    view = table.to_pandas()
    rows = view.merge(frame, on=["track_id", "split"], validate="many_to_one")
    assert len(rows) == len(view) == 16
    assert rows.groupby(["split", "genre_id", "label_present"])["split_group"].nunique().eq(2).all()
    assert all((row.genre_id in row.genre_ids) == row.label_present for row in rows.itertuples())


def test_rare_support_enriches_both_held_out_splits():
    sizes = np.ones(4000, dtype=int)
    positives = np.zeros((4000, 2), dtype=int)
    positives[:400, 0] = 1
    positives[100:3100, 1] = 1
    proportional, _ = assign_groups(sizes, positives)
    rare_support, _ = assign_groups(sizes, positives, strategy="rare-support")
    for split in (1, 2):
        assert positives[rare_support == split, 0].sum() > positives[proportional == split, 0].sum()
    assert np.bincount(rare_support).min() > 350
