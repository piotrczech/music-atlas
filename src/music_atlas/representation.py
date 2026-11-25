"""Turn every track into one vector and save it in the shared representation format.

A representation only supplies `encode`: windows in, one vector per window out. This module
reads the tracks shard by shard, averages each track's windows and writes `vectors.npy`,
`track_ids.npy`, `metadata.json` and `failed.csv` (docs/EXPERIMENTS.md). Each finished shard
is kept as a part, so a resumed run (`track(..., resume=run_id)`) skips it.
"""

import json
import tarfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from music_atlas.loader import load_windows

PARTITIONS = ("train", "validation", "test")
# Windows per `encode` call; a few tracks at a time keep a GPU busy without holding a shard.
BATCH_WINDOWS = 64

# (n_windows, n_samples) float32 mono at settings["sample_rate"] -> (n_windows, d)
Encoder = Callable[[np.ndarray], np.ndarray]


def read_tracks(data_dir: Path, version: str, profile_name: str) -> dict[str, pd.DataFrame]:
    """Tracks per subset with `track_id`, `shard` (relative to data_dir) and `audio_path`.

    The quick profile reads its sample from `audio.tar`, the full profile all three partitions.
    """
    if profile_name == "quick":
        sample = json.loads(
            (data_dir / "samples" / version / "sample.json").read_text(encoding="utf-8")
        )
        tracks = pd.DataFrame(sample["tracks"]).assign(shard=f"samples/{version}/audio.tar")
        return {"quick-sample": tracks}
    ready = data_dir / "ready" / version
    return {name: pd.read_parquet(ready / f"{name}.parquet") for name in PARTITIONS}


def encode_tracks(tracks, encode: Encoder) -> dict:
    """Encode [(track_id, windows)] in one batch and average each track's windows."""
    samples = np.stack([window.samples for _, windows in tracks for window in windows])
    vectors = np.asarray(encode(samples), dtype=np.float32)
    if len(vectors) != len(samples):
        raise ValueError(f"encode returned {len(vectors)} vectors for {len(samples)} windows.")
    result, start = {}, 0
    for track_id, windows in tracks:
        result[track_id] = vectors[start : start + len(windows)].mean(axis=0)
        start += len(windows)
    return result


def extract_shard(archive: Path, rows: pd.DataFrame, encode: Encoder, settings: dict) -> dict:
    """Vectors and errors of the tracks in one shard, which is opened once."""
    vectors, errors, batch = {}, {}, []
    with tarfile.open(archive) as tar:
        for row in rows.itertuples():
            try:
                windows = load_windows(tar.extractfile(row.audio_path), row.track_id, **settings)
            except (KeyError, ValueError, OSError) as error:
                errors[row.track_id] = str(error)
                continue
            batch.append((row.track_id, windows))
            if sum(len(windows) for _, windows in batch) >= BATCH_WINDOWS:
                vectors |= encode_tracks(batch, encode)
                batch = []
    if batch:
        vectors |= encode_tracks(batch, encode)
    return {"track_ids": np.array(list(vectors), dtype=np.int64),
            "vectors": np.array(list(vectors.values())),
            "failed_ids": np.array(list(errors), dtype=np.int64),
            "errors": np.array(list(errors.values()), dtype=str)}


def extract(data_dir: Path, subsets: dict[str, pd.DataFrame], directory: Path,
            encode: Encoder, settings: dict, metadata: dict, record=None) -> dict:
    """Encode every track of `subsets` into `directory`, resuming from finished shards.

    `settings` are the loader's (`sample_rate` is the encoder's own); `metadata` names the
    encoder, checkpoint and revision. With `record` (a `music_atlas.tracking.Run`) each subset
    is logged as a dataset and the step's outcome and speed as metrics.
    """
    tracks = pd.concat(subsets.values(), ignore_index=True)
    parts = directory / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    started, encoded = time.perf_counter(), 0
    # Partitions share shards, so all of them go in one pass and each shard is read once.
    for shard, rows in tracks.groupby("shard", sort=True):
        part = parts / f"{Path(shard).stem}.npz"
        if part.is_file():
            continue
        result = extract_shard(data_dir / shard, rows, encode, settings)
        # Written under another name first: a job killed mid-write leaves no half part.
        partial = part.with_suffix(".partial")
        with partial.open("wb") as file:
            np.savez(file, **result)
        partial.rename(part)
        encoded += len(rows)
    seconds = time.perf_counter() - started

    loaded = [np.load(part) for part in sorted(parts.glob("*.npz"))]
    track_ids = np.concatenate([part["track_ids"] for part in loaded])
    vectors = np.concatenate([part["vectors"] for part in loaded if len(part["vectors"])])
    order = np.argsort(track_ids)
    np.save(directory / "track_ids.npy", track_ids[order])
    np.save(directory / "vectors.npy", vectors[order])
    # The input of `atlas fma-record-filter` for tracks this representation cannot use.
    failed = pd.DataFrame({
        "track_id": np.concatenate([part["failed_ids"] for part in loaded]),
        "error": np.concatenate([part["errors"] for part in loaded]),
    }).sort_values("track_id")
    failed.to_csv(directory / "failed.csv", index=False)
    meta = {**metadata, "settings": settings, "subsets": list(subsets),
            "aggregation": "mean of window vectors", "tracks": len(track_ids),
            "dimensions": int(vectors.shape[1])}
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    measured = {"encoded": len(track_ids), "failed": len(failed), "seconds": round(seconds, 1),
                # Only this attempt's shards, so a resumed run still measures real throughput.
                "tracks_per_second": round(encoded / seconds, 2) if encoded else None}
    if record is not None:
        for name, rows in subsets.items():
            record.log_dataset(rows, name, summary={"settings": settings})
        record.log_file(directory / "failed.csv", "dataset")
        record.log(measured)
    return {"run_dir": str(directory), **measured}
