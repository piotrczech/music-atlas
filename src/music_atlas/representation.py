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
from hashlib import sha256
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from music_atlas.loader import load_windows
from music_atlas.settings import profile
from music_atlas.shards import SPLITS, process_shards, read_tracks, save
from music_atlas.tracking import write_json

# Windows per `encode` call; a few tracks at a time keep a GPU busy without holding a shard.
BATCH_WINDOWS = 64
MEAN = "mean of window vectors"
UNIT_MEAN = "L2 per window, arithmetic mean, L2 per track"

# (n_windows, n_samples) float32 mono at the encoder's sample rate -> (n_windows, d)
Encoder = Callable[[np.ndarray], np.ndarray]


def aggregate(windows, rule):
    if rule == MEAN:
        return windows.mean(axis=0)
    lengths = np.linalg.norm(windows, axis=1)
    if not np.isfinite(lengths).all() or (lengths <= 1e-12).any():
        raise ValueError("Zero or non-finite window projection.")
    mean = (windows / lengths[:, None]).mean(axis=0)
    length = np.linalg.norm(mean)
    if not np.isfinite(length) or length <= 1e-12:
        raise ValueError("Zero or non-finite track mean.")
    return mean / length


def encode_batch(windows: list, encode: Encoder, prepare=None, dimensions=0) -> np.ndarray:
    samples = np.stack([prepare(w) if prepare else w.samples for w in windows])
    vectors = np.asarray(encode(samples), dtype=np.float32)
    if (vectors.ndim != 2 or len(vectors) != len(samples) or not vectors.shape[1]
            or dimensions and vectors.shape[1] != dimensions):
        raise ValueError(f"Encoder shape {vectors.shape} for {len(samples)} windows.")
    return vectors


def encode_shard(tar_path: str, rows: pd.DataFrame, part: Path, encode: Encoder,
                 settings: dict, batch_size=BATCH_WINDOWS, prepare=None,
                 dimensions=0, aggregation=MEAN) -> None:
    """Encode every window of one shard and save them with their tracks to `<part>.npz`."""
    windows, vectors, batch, errors = [], [], [], {}
    with tarfile.open(tar_path) as tar:
        for row in rows.itertuples():
            try:
                with tar.extractfile(row.audio_path) as audio:
                    track = load_windows(audio, row.track_id, **settings)
            except (KeyError, ValueError, OSError) as error:
                errors[row.track_id] = str(error)
                continue
            windows.extend((w.track_id, w.index, w.start_seconds) for w in track)
            batch += track
            while len(batch) >= batch_size:
                vectors.append(encode_batch(batch[:batch_size], encode, prepare, dimensions))
                batch = batch[batch_size:]
    if batch:
        vectors.append(encode_batch(batch, encode, prepare, dimensions))

    ids, indices, starts = (np.asarray(col) for col in zip(*windows, strict=True)) if windows else (
        np.array([]), np.array([]), np.array([]))
    projected = np.concatenate(vectors) if vectors else np.empty((0, dimensions), np.float32)
    norms = np.linalg.norm(projected, axis=1) if aggregation == UNIT_MEAN else []
    if aggregation == UNIT_MEAN and len(ids):
        unique, offsets = np.unique(ids, return_index=True)
        for track_id, start, end in zip(unique, offsets, [*offsets[1:], len(ids)], strict=True):
            try:
                aggregate(projected[start:end], aggregation)
            except ValueError as error:
                errors[track_id] = str(error)
        keep = ~np.isin(ids, list(errors))
        ids, indices, starts, projected = ids[keep], indices[keep], starts[keep], projected[keep]
    save(part.with_suffix(".npz"), lambda file: np.savez(
        file, track_ids=ids.astype(np.int64), window=indices.astype(np.int16),
        start_seconds=starts.astype(np.float32), vectors=projected, raw_norms=norms,
        error_ids=np.array(list(errors), np.int64), errors=np.array(list(errors.values()), str),
    ))


def complete_part(path, rows, *, dimensions):
    try:
        with np.load(path, allow_pickle=False) as part:
            ids = part["track_ids"]
            tracks, starts, counts = np.unique(ids, return_index=True, return_counts=True)
            vectors = part["vectors"]
            indices = np.arange(len(ids)) - np.repeat(starts, counts)
            return (not len(part["error_ids"]) and not len(part["errors"])
                    and np.array_equal(tracks, np.sort(rows.track_id))
                    and np.array_equal(ids, np.repeat(tracks, counts))
                    and np.array_equal(part["window"], indices)
                    and len(part["start_seconds"]) == len(ids)
                    and vectors.shape == (len(ids), dimensions)
                    and np.isfinite(vectors).all()
                    and len(part["raw_norms"]) == len(ids))
    except (OSError, EOFError, BadZipFile, KeyError, ValueError):
        return False


def extract(data_dir: Path, directory: Path, encode: Encoder, sample_rate: int, metadata: dict,
            profile_name: str = "quick", windows_per_track: int | None = None, limit: int = 0,
            record=None, *, inputs=None, batch_size=BATCH_WINDOWS, aggregation=MEAN,
            prepare=None) -> dict:
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
    if batch_size < 1 or aggregation not in (MEAN, UNIT_MEAN):
        raise ValueError("Invalid batch size or aggregation rule.")
    rows, dataset, genres = inputs if inputs is not None else read_tracks(data_dir, profile_name)
    rows = rows.sort_values("track_id")
    if rows.empty or rows.track_id.isna().any() or not rows.track_id.is_unique:
        raise ValueError("Expected nonempty, unique input track IDs.")
    tars = sorted(rows.tar.unique())
    if len({Path(t).stem for t in tars}) != len(tars):
        raise ValueError("Shard basenames must be unique.")
    directory.mkdir(parents=True, exist_ok=True)

    # A rerun must continue with the same encoder and windows, not mix in other parts.
    identity = {**metadata, "version": dataset["version"],
                "manifest_sha256": dataset["manifest_sha256"], "profile": profile_name,
                "settings": settings, "tracks": len(rows)}
    if aggregation == UNIT_MEAN:
        inputs_hash = rows[["track_id", "split", "audio_path"]].assign(
            shard=rows.tar.map(lambda t: Path(t).relative_to(data_dir).as_posix()))
        tracks_csv = inputs_hash.to_csv(index=False, lineterminator="\n")
        identity.update(
            aggregation=aggregation, batch_size=batch_size,
            tracks_sha256=sha256(tracks_csv.encode()).hexdigest(),
            subsets=["quick-sample"] if profile_name == "quick" else
            [p for p in SPLITS if p in set(rows.split)],
        )
        config_json = json.dumps(identity, sort_keys=True)
        identity["config_sha256"] = sha256(config_json.encode()).hexdigest()
    metadata_path = directory / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in identity} != identity:
            raise ValueError(f"{directory} holds vectors with other settings or configuration.")
    else:
        if any((directory / "parts").glob("*.npz")):
            raise ValueError("Saved parts have no extraction configuration.")
        write_json(metadata_path, identity)

    if record is not None:
        for split, group in rows.groupby("split"):
            subset = "quick-sample" if profile_name == "quick" else split
            if subset not in record.subsets:
                record.log_dataset(group, subset, genres, {"settings": settings})
    parts = directory / "parts"
    started = time.perf_counter()
    dimensions = metadata.get("dimensions", 0)
    work = partial(encode_shard, encode=encode, settings=settings, prepare=prepare,
                   dimensions=dimensions, batch_size=batch_size, aggregation=aggregation)
    complete = partial(complete_part, dimensions=dimensions) if aggregation == UNIT_MEAN else None
    finished, failed = process_shards(rows, parts, work, limit=limit, complete=complete)
    seconds = time.perf_counter() - started
    new_tracks = int(rows["tar"].map(lambda tar: Path(tar).stem).isin(finished).sum())
    # Only this attempt's shards, so a resumed run still measures real throughput.
    speed = {"seconds": round(seconds, 1),
             "tracks_per_second": round(new_tracks / seconds, 2) if new_tracks else None}

    shards = [Path(tar).stem for tar in tars]
    done = [name for name in shards if (parts / f"{name}.npz").exists() and name not in failed]
    if len(done) < len(shards):
        progress = {"parts": len(shards), "parts_done": len(done),
                    "failed_parts": len(failed), **speed}
        if record is None:
            return {"run_dir": str(directory), **progress, "failed": failed}
        record.log(progress)
        raise ValueError(f"{len(shards) - len(done)} of {len(shards)} shards are not encoded "
                         f"yet ({failed}); continue with --resume {record.run_id}.")

    loaded = [dict(np.load(parts / f"{name}.npz")) for name in done]
    windows = pd.DataFrame({key: np.concatenate([part[key] for part in loaded])
                            for key in ("track_ids", "window", "start_seconds")})
    nonempty = [part["vectors"] for part in loaded if part["vectors"].size]
    window_vectors = (np.concatenate(nonempty) if nonempty else
                      np.empty((0, dimensions), dtype=np.float32))
    failed_tracks = pd.DataFrame({
        "track_id": np.concatenate([part["error_ids"] for part in loaded]),
        "error": np.concatenate([part["errors"] for part in loaded]),
    }).sort_values("track_id")

    # Windows sorted by track, then each track's mean: the explicit aggregation rule.
    order = np.lexsort((windows["window"], windows["track_ids"]))
    windows, window_vectors = windows.iloc[order].reset_index(drop=True), window_vectors[order]
    track_ids, starts, counts = np.unique(windows["track_ids"], return_index=True,
                                          return_counts=True)
    if aggregation == UNIT_MEAN:
        vectors = np.array([aggregate(window_vectors[s:s+n], aggregation)
                            for s, n in zip(starts, counts, strict=True)],
                           dtype=np.float32).reshape(len(track_ids), dimensions)
        n_windows = sum(len(part["raw_norms"]) for name, part in zip(done, loaded, strict=True)
                        if name in finished)
        speed["windows_per_second"] = n_windows / seconds if n_windows else None
        speed["tracks_per_second"] = new_tracks / seconds if new_tracks else None
    else:
        vectors = (np.add.reduceat(window_vectors, starts) / counts[:, None]
                   if len(starts) else window_vectors)

    np.save(directory / "vectors.npy", vectors.astype(np.float32))
    np.save(directory / "track_ids.npy", track_ids)
    np.save(directory / "window_vectors.npy", window_vectors)
    windows.rename(columns={"track_ids": "track_id"}).to_parquet(
        directory / "windows.parquet", index=False)
    failed_tracks.to_csv(directory / "failed.csv", index=False)
    meta = {**identity, "aggregation": aggregation, "encoded_tracks": len(track_ids),
            "windows": len(windows), "dimensions": int(vectors.shape[1]), "dtype": "float32"}
    write_json(metadata_path, meta)

    measured = {"encoded": len(track_ids), "failed": len(failed_tracks), **speed}
    if record is not None:
        record.log_file(directory / "failed.csv", "dataset")
        record.log(measured)
    return {"run_dir": str(directory), **measured}
