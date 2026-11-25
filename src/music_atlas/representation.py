"""Turn every track into vectors with any encoder and save them in the shared format.

A representation supplies only `encode`: windows in, one vector per window out. `extract`
runs it shard by shard through music_atlas.shards, so a killed job resumes from its finished
shards, and writes (docs/EXPERIMENTS.md, Representations and windows):

- `vectors.npy` (tracks, d) and `track_ids.npy`: the mean of each track's window vectors;
- `window_vectors.npy` (windows, d) and `windows.parquet` (track_id, window, start_seconds):
  every window, so another aggregation needs no second extraction;
- `metadata.json` and `failed.csv` (the input of `atlas fma-record-filter`).
"""

import json
import tarfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from music_atlas.loader import load_windows
from music_atlas.settings import profile
from music_atlas.shards import process_shards, read_tracks, save

# Windows per `encode` call; a few tracks at a time keep a GPU busy without holding a shard.
BATCH_WINDOWS = 64

# (n_windows, n_samples) float32 mono at the encoder's sample rate -> (n_windows, d)
Encoder = Callable[[np.ndarray], np.ndarray]


def encode_batch(windows: list, encode: Encoder) -> np.ndarray:
    samples = np.stack([window.samples for window in windows])
    vectors = np.asarray(encode(samples), dtype=np.float32)
    if len(vectors) != len(samples):
        raise ValueError(f"encode returned {len(vectors)} vectors for {len(samples)} windows.")
    return vectors


def encode_shard(tar_path: str, rows: pd.DataFrame, part: Path, encode: Encoder,
                 settings: dict) -> None:
    """Encode every window of one shard and save them with their tracks to `<part>.npz`."""
    windows, vectors, batch, errors = [], [], [], {}
    with tarfile.open(tar_path) as tar:
        for row in rows.itertuples():
            try:
                track = load_windows(tar.extractfile(row.audio_path), row.track_id, **settings)
            except (KeyError, ValueError, OSError) as error:
                errors[row.track_id] = str(error)
                continue
            windows += track
            batch += track
            if len(batch) >= BATCH_WINDOWS:
                vectors.append(encode_batch(batch, encode))
                batch = []
    if batch:
        vectors.append(encode_batch(batch, encode))

    save(part.with_suffix(".npz"), lambda file: np.savez(
        file, track_ids=np.array([w.track_id for w in windows], np.int64),
        window=np.array([w.index for w in windows], np.int16),
        start_seconds=np.array([w.start_seconds for w in windows], np.float32),
        vectors=np.concatenate(vectors) if vectors else np.zeros((0, 0), np.float32),
        error_ids=np.array(list(errors), np.int64), errors=np.array(list(errors.values()), str),
    ))


def extract(data_dir: Path, directory: Path, encode: Encoder, sample_rate: int, metadata: dict,
            profile_name: str = "quick", windows_per_track: int | None = None, limit: int = 0,
            record=None) -> dict:
    """Encode the profile's tracks into `directory`, resuming from finished shards.

    `sample_rate` is the encoder's own; window positions do not depend on it. `metadata` names
    the encoder, checkpoint and revision. `windows_per_track` replaces the profile's count
    (5 windows contain the default 3 as windows 0, 2 and 4). The encoder runs in this process,
    so a GPU model is loaded once. With `record` (a `music_atlas.tracking.Run`) the partitions
    become its datasets and the outcome and speed its metrics; a run with shards still missing
    fails, so `--resume` continues it.
    """
    config = profile(profile_name)
    settings = {"sample_rate": sample_rate, "window_seconds": config["window_seconds"],
                "windows_per_track": windows_per_track or config["windows_per_track"]}
    rows, dataset, genres = read_tracks(data_dir, profile_name)
    directory.mkdir(parents=True, exist_ok=True)

    # A rerun must continue with the same encoder and windows, not mix in other parts.
    identity = {**metadata, "version": dataset["version"],
                "manifest_sha256": dataset["manifest_sha256"], "profile": profile_name,
                "settings": settings, "tracks": len(rows)}
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in identity} != identity:
            raise ValueError(f"{directory} holds vectors with other settings; start a new run.")
    else:
        metadata_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    parts = directory / "parts"
    started = time.perf_counter()
    work = partial(encode_shard, encode=encode, settings=settings)
    finished, failed = process_shards(rows, parts, work, limit=limit)
    seconds = time.perf_counter() - started
    new_tracks = int(rows["tar"].map(lambda tar: Path(tar).stem).isin(finished).sum())
    # Only this attempt's shards, so a resumed run still measures real throughput.
    speed = {"seconds": round(seconds, 1),
             "tracks_per_second": round(new_tracks / seconds, 2) if new_tracks else None}

    shards = sorted({Path(tar).stem for tar in rows["tar"]})
    done = [name for name in shards if (parts / f"{name}.npz").exists()]
    if len(done) < len(shards):
        progress = {"parts": len(shards), "parts_done": len(done),
                    "failed_parts": len(failed), **speed}
        if record is None:
            return {"run_dir": str(directory), **progress, "failed": failed}
        record.log(progress)
        raise ValueError(f"{len(shards) - len(done)} of {len(shards)} shards are not encoded "
                         f"yet; continue with --resume {record.run_id}.")

    loaded = [dict(np.load(parts / f"{name}.npz")) for name in done]
    windows = pd.DataFrame({key: np.concatenate([part[key] for part in loaded])
                            for key in ("track_ids", "window", "start_seconds")})
    window_vectors = np.concatenate([part["vectors"] for part in loaded if part["vectors"].size])
    failed_tracks = pd.DataFrame({
        "track_id": np.concatenate([part["error_ids"] for part in loaded]),
        "error": np.concatenate([part["errors"] for part in loaded]),
    }).sort_values("track_id")

    # Windows sorted by track, then each track's mean: the explicit aggregation rule.
    order = np.lexsort((windows["window"], windows["track_ids"]))
    windows, window_vectors = windows.iloc[order].reset_index(drop=True), window_vectors[order]
    track_ids, starts, counts = np.unique(windows["track_ids"], return_index=True,
                                          return_counts=True)
    vectors = np.add.reduceat(window_vectors, starts) / counts[:, None]

    np.save(directory / "vectors.npy", vectors.astype(np.float32))
    np.save(directory / "track_ids.npy", track_ids)
    np.save(directory / "window_vectors.npy", window_vectors)
    windows.rename(columns={"track_ids": "track_id"}).to_parquet(
        directory / "windows.parquet", index=False)
    failed_tracks.to_csv(directory / "failed.csv", index=False)
    meta = {**identity, "aggregation": "mean of window vectors", "encoded_tracks": len(track_ids),
            "windows": len(windows), "dimensions": int(vectors.shape[1])}
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    measured = {"encoded": len(track_ids), "failed": len(failed_tracks), **speed}
    if record is not None:
        for split, group in rows.groupby("split"):
            subset = "quick-sample" if profile_name == "quick" else split
            record.log_dataset(group, subset, genres, {"settings": settings})
        record.log_file(directory / "failed.csv", "dataset")
        record.log(measured)
    return {"run_dir": str(directory), **measured}
