"""Audit FMA-full audio, annotations and licences before the final split (E1-03)."""

import json
import sys
import tarfile
import time
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from music_atlas.loader import load_windows, open_audio
from music_atlas.settings import profile

# Exclusions cover technical failures only; unusual but decodable music stays valid.
MIN_SECONDS = 1.0  # less audio than this is a broken upload, not a musical example
SILENT_PEAK = 1e-3  # no sample above -60 dBFS: digital silence
# Warnings keep the track valid.
SHORT_SECONDS = 10.0  # shorter than one loader window, so zero-padded
QUIET_DBFS = -40.0
DURATION_TOLERANCE = 2.0  # published durations are whole seconds
# Re-encoding candidates: near-equal length and loudness contour over the first two minutes.
ENVELOPE_SECONDS = 120
MATCH_SECONDS = 1.0
MATCH_CORRELATION = 0.98
# Free to share and adapt, also commercially: CC0, PDM, CC BY and CC BY-SA in any version.
OPEN_LICENCE = (
    r"CC0 1\.0 Universal|Public Domain Mark 1\.0"
    r"|(Creative Commons )?Attribution(-Share ?Alike)?( \d.*)?"
)
FIELDS = {
    ("artist", "id"): "artist_id", ("album", "id"): "album_id", ("track", "duration"): "duration",
    ("track", "genre_top"): "genre_top", ("track", "genres_all"): "genres_all",
    ("track", "license"): "license",
}


def measure(file, track_id, windows):
    """Decode a whole track in 60 s blocks, then read it through the shared loader."""
    row = {"track_id": track_id, "seconds": np.nan, "peak": np.nan, "rms_dbfs": np.nan,
           "envelope": np.empty(0, np.float32), "error": None}
    stage = "decode_error"
    try:
        with open_audio(file) as audio:
            rate = audio.samplerate
            frames, energy, peak, envelope = 0, 0.0, 0.0, [np.empty(0)]
            while len(y := audio.read(60 * rate, dtype="float32", always_2d=True)):
                y = y.mean(axis=1)
                if not np.isfinite(y).all():
                    raise ValueError("non-finite samples")
                frames += len(y)
                energy += float(y @ y)
                peak = max(peak, float(np.abs(y).max()))
                seconds = len(y) // rate
                envelope.append((y[: seconds * rate].reshape(seconds, rate) ** 2).mean(axis=1))
        envelope = 10 * np.log10(np.concatenate(envelope)[:ENVELOPE_SECONDS] + 1e-10)
        row.update(seconds=frames / rate, peak=peak,
                   rms_dbfs=10 * np.log10(energy / max(frames, 1) + 1e-10),
                   envelope=np.maximum(envelope, -60).astype(np.float32))
        if frames:
            stage = "loader_error"
            file.seek(0)
            load_windows(file, track_id, **windows)
    except (RuntimeError, ValueError, OSError) as error:
        row["error"] = f"{stage}: {error}"
    return row


def measure_shard(shard: Path, target: Path) -> str | None:
    """Measure every track of one tar; the Parquet file appears only when the shard is done."""
    settings = profile("full")
    windows = {key: settings[key] for key in ("sample_rate", "window_seconds", "windows_per_track")}
    started = time.monotonic()
    try:
        with tarfile.open(shard) as tar:
            rows = [measure(tar.extractfile(member), int(Path(member.name).stem), windows)
                    for member in tar if member.isfile()]
        part = target.with_name(target.name + ".part")
        pd.DataFrame(rows).to_parquet(part, index=False)
        part.replace(target)
    except Exception as error:  # report the shard and continue; a rerun retries it
        return f"{type(error).__name__}: {error}"
    print(f"{shard.name} {len(rows)} tracks in {time.monotonic() - started:.0f} s", file=sys.stderr)


def reencodings(seconds, envelopes, sha1):
    """Link tracks of near-equal length whose loudness contours correlate but bytes differ."""
    lengths = envelopes.map(len, na_action="ignore")
    order = seconds[seconds.notna() & lengths.ge(10)].sort_values(kind="stable")
    ids, span = order.index.to_numpy(), order.to_numpy()
    sizes, keys = lengths[order.index].to_numpy(int), sha1[order.index].to_numpy()
    contours = np.zeros((len(ids), ENVELOPE_SECONDS), np.float32)
    for row, envelope in enumerate(envelopes[order.index]):
        contours[row, : len(envelope)] = envelope
    pairs = []
    for i, n in enumerate(sizes):
        stop = np.searchsorted(span, span[i] + MATCH_SECONDS, side="right")
        x = contours[i, :n] - contours[i, :n].mean()
        others = contours[i + 1 : stop, :n]
        others = others - others.mean(axis=1, keepdims=True)
        r = others @ x / (np.linalg.norm(others, axis=1) * np.linalg.norm(x) + 1e-9)
        matches = i + 1 + np.flatnonzero(r >= MATCH_CORRELATION)
        pairs += [(i, j) for j in matches if keys[j] != keys[i]]
    a, b = np.array(pairs, dtype=int).reshape(-1, 2).T
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ids), len(ids)))
    labels = connected_components(graph, directed=False)[1]
    group = pd.Series(ids, index=ids).groupby(labels).transform("min")
    return group.where(np.bincount(labels)[labels] > 1).reindex(seconds.index).astype("Int64")


def audit(inventory, metadata, genres, measured):
    """Return one row per inventory track with validity, reasons and groups, plus a summary."""
    ids, present = inventory.index, inventory["status"].eq("present")
    if len(ids[present].difference(measured.index)):
        raise ValueError("Present tracks lack audio measurements; rerun the audit.")
    m, tracks = measured.reindex(ids), metadata.reindex(ids)
    seconds, failure = m["seconds"], m["error"].astype("string")
    error = (
        pd.Series(pd.NA, index=ids, dtype="string")
        .mask(m["peak"].lt(SILENT_PEAK), "silent")
        .mask(seconds.lt(MIN_SECONDS), "too_short")
        .mask(failure.notna(), failure.str.split(":").str[0])
        .mask(~present, "inventory_missing")
    )
    valid = error.isna()
    # Published SHA-1s give stable groups even before every shard is present.
    sha1 = inventory["sha1"].str.lower()
    first = pd.Series(ids, index=ids).groupby(sha1).transform("min")
    duplicate_group = first.where(sha1.map(sha1.value_counts()).gt(1)).astype("Int64")
    candidate = reencodings(seconds.where(valid), m["envelope"], sha1)
    published = pd.to_numeric(tracks["duration"], errors="coerce")
    flags = pd.DataFrame({name: flag & valid for name, flag in {
        "duration_mismatch": (seconds - published).abs().gt(DURATION_TOLERANCE),
        "short": seconds.lt(SHORT_SECONDS),
        "quiet": m["rms_dbfs"].lt(QUIET_DBFS),
    }.items()})
    warning = flags.dot(flags.columns + ";").str.rstrip(";").replace("", pd.NA).astype("string")
    licence = tracks["license"].astype("string").str.strip()
    example = valid & licence.str.fullmatch(OPEN_LICENCE).fillna(False).astype(bool)
    table = pd.DataFrame({
        "valid": valid, "error": error, "duration_seconds": seconds,
        "duplicate_group": duplicate_group, "reencoding_candidate": candidate,
        "warning": warning, "listening_example": example,
    }).rename_axis("track_id").reset_index()

    roots = set(genres.index[genres["parent"].eq(0)])
    top = tracks["genres_all"].fillna("[]").map(lambda tags: [g for g in literal_eval(tags)
                                                             if g in roots])
    unique_top = top.map(lambda tags: genres.at[tags[0], "title"] if len(tags) == 1 else "")
    pairs = top[present].explode().dropna()
    rows = pd.DataFrame({
        "genre": genres.loc[pairs.astype(int), "title"].to_numpy(),
        "artist": tracks.loc[pairs.index, "artist_id"].to_numpy(),
        "valid": valid.loc[pairs.index].to_numpy(), "example": example.loc[pairs.index].to_numpy(),
    })
    by = rows.groupby("genre")
    coverage = pd.DataFrame({
        "tracks": by.size(), "valid": by["valid"].sum(), "artists": by["artist"].nunique(),
        "valid_artists": rows[rows["valid"]].groupby("genre")["artist"].nunique(),
        "listening_examples": by["example"].sum(),
    }).fillna(0).astype(int)
    summary = {
        "tracks": len(ids), "present": int(present.sum()), "valid": int(valid.sum()),
        "excluded": error.value_counts().to_dict(), "warnings": flags.sum().to_dict(),
        "exact_duplicates": {"groups": duplicate_group.nunique(),
                             "tracks": int(duplicate_group.notna().sum())},
        "reencoding_candidates": {"groups": candidate.nunique(),
                                  "tracks": int(candidate.notna().sum())},
        "annotations": {
            "untagged": int(top.map(len).eq(0).sum()),
            "multiple_top_genres": int(top.map(len).gt(1).sum()),
            "genre_top_inconsistent": int(unique_top.ne(tracks["genre_top"].fillna("")).sum()),
            "missing_artist": int((~tracks["artist_id"].gt(0)).sum()),
            "missing_album": int((~tracks["album_id"].gt(0)).sum()),
            "published_duration_not_positive": int((~published.gt(0)).sum()),
        },
        "licences": {"missing": int(licence.isna().sum()), "distinct": licence.nunique(),
                     "listening_examples": int(example.sum())},
        "genres": coverage.to_dict("index"),
    }
    return table, summary


def run(data_dir: Path, workers: int = 4, limit: int = 0) -> dict:
    """Measure the present shards not yet measured, then write the audit and its summary."""
    manifests = data_dir / "manifests"
    inventory = pd.read_csv(manifests / "fma_full_inventory.csv", index_col="track_id",
                            dtype={"sha1": "string"})
    shards = sorted(inventory.loc[inventory["status"].eq("present"), "shard"].unique())
    if not shards:
        raise ValueError("The inventory lists no present shards.")
    cache = manifests / "fma_audio_audit"
    cache.mkdir(parents=True, exist_ok=True)
    targets = {shard: cache / f"{Path(shard).stem}.parquet" for shard in shards}
    todo = [shard for shard, target in targets.items() if not target.exists()][: limit or None]
    with ProcessPoolExecutor(workers) as pool:
        errors = pool.map(measure_shard, [data_dir / s for s in todo], [targets[s] for s in todo])
        failed = {shard: error for shard, error in zip(todo, errors, strict=True) if error}
    remaining = sum(not target.exists() for target in targets.values())
    if remaining:
        return {"shards": len(shards), "remaining": remaining, "failed": failed}

    measured = pd.concat(pd.read_parquet(target) for target in targets.values())
    source = data_dir / "fma/fma_metadata"
    metadata = pd.read_csv(source / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
    metadata = metadata[list(FIELDS)].set_axis(list(FIELDS.values()), axis=1)
    genres = pd.read_csv(source / "genres.csv", index_col="genre_id")
    table, summary = audit(inventory, metadata, genres, measured.set_index("track_id"))
    target = manifests / "fma_audio_audit.parquet"
    part = target.with_name(target.name + ".part")
    table.to_parquet(part, index=False)
    part.replace(target)
    text = json.dumps(summary, indent=2, default=int) + "\n"
    target.with_suffix(".json").write_text(text, encoding="utf-8")
    return {**summary, "audit": str(target), "failed": failed}
