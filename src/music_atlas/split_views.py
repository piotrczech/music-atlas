"""Derived views tied to one immutable FMA split version."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_once(table, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".view-", dir=target.parent) as temp:
        source = Path(temp) / target.name
        pq.write_table(table, source)
        try:
            # Windows rename refuses to replace an existing file; POSIX rename does not.
            if os.name == "nt":
                source.rename(target)
            else:
                os.link(source, target)
        except FileExistsError:
            if not pq.read_table(target).equals(table, check_metadata=True):
                raise ValueError(f"Existing view differs: {target}") from None


def write_balanced_eval(manifest, mapping, directory, version, manifest_hash, seed):
    """Sample equal positive and untagged-by-class rows within each existing holdout."""
    genres = [row["genre_id"] for row in mapping if row["scored"]]
    if not genres:
        return None

    pools = []
    annotated = manifest["genre_ids"].map(len).gt(0)
    for split in ("validation", "test"):
        tracks = manifest.loc[annotated & manifest["split"].eq(split)]
        for genre in genres:
            positive = tracks["genre_ids"].map(lambda tags, genre=genre: genre in tags)
            for present, mask in ((True, positive), (False, ~positive)):
                pools.append((split, genre, present, tracks.loc[mask, ["track_id", "split_group"]]))
    quota = min(len(tracks) for _, _, _, tracks in pools)
    if not quota:
        raise ValueError("A scored genre has an empty held-out evaluation pool.")

    rng = np.random.default_rng(seed)
    rows = []
    for split, genre, present, pool in pools:
        shuffled = pool.iloc[rng.permutation(len(pool))]
        # Sample each independent group once before drawing a second track from it.
        rank = shuffled.groupby("split_group", sort=False).cumcount().to_numpy()
        ids = shuffled.iloc[np.argsort(rank, kind="stable")[:quota]]["track_id"]
        rows.extend((split, genre, track, present) for track in ids)
    frame = pd.DataFrame(rows, columns=["split", "genre_id", "track_id", "label_present"])
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata({
        b"dataset_version": version.encode(), b"manifest_sha256": manifest_hash.encode(),
        b"rows_per_label_per_side": str(quota).encode(), b"seed": str(seed).encode(),
    })
    pq.write_table(table, directory / "balanced_eval.parquet")
    return quota


def record_filter(data_dir, artifacts_dir, version, representation, run_id, exclusions_path):
    """Record representation-specific exclusions while inheriting each track's split."""
    for value in (version, representation, run_id):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"Invalid path component: {value!r}.")

    ready = (data_dir / "ready").resolve()
    directory = (ready / version).resolve()
    if directory.parent != ready:
        raise ValueError(f"Split version must resolve within {ready}.")
    version = directory.name
    manifest_path = directory / "tracks.parquet"
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    manifest_hash = digest(manifest_path)
    if report["version"] != version or report["manifest_sha256"] != manifest_hash:
        raise ValueError(f"Manifest version or hash mismatch: {directory}.")
    manifest = pd.read_parquet(manifest_path, columns=["track_id", "split", "valid"])
    assigned = manifest["valid"] & manifest["split"].notna()
    selection = manifest.loc[assigned, ["track_id", "split"]].copy()
    exclusions = pd.read_csv(exclusions_path, usecols=["track_id", "error"], dtype={
        "track_id": "Int64", "error": "string",
    })
    reasons = exclusions["error"].str.strip().replace("", pd.NA)
    if exclusions["track_id"].duplicated().any() or reasons.isna().any():
        raise ValueError("Exclusions require unique track IDs and nonempty reasons.")
    if not exclusions["track_id"].isin(selection["track_id"]).all():
        raise ValueError("Exclusions contain track IDs outside the selected split version.")
    errors = reasons.set_axis(exclusions["track_id"])
    selection["error"] = selection["track_id"].map(errors).astype("string")
    selection["included"] = selection["error"].isna()
    table = pa.Table.from_pandas(selection, preserve_index=False).replace_schema_metadata({
        b"dataset_version": version.encode(), b"manifest_sha256": manifest_hash.encode(),
    })
    target = artifacts_dir / representation / run_id / "filter.parquet"
    write_once(table, target)
    return {
        "filter": str(target), "dataset_version": version, "manifest_sha256": manifest_hash,
        "included": int(selection["included"].sum()), "excluded": len(exclusions),
    }
