"""Fixed quick-profile sample of training tracks, kept as one tar for local runs."""

import io
import json
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from music_atlas.loader import load_windows
from music_atlas.settings import profile
from music_atlas.split_views import digest

try:
    import resource
except ImportError:  # Windows
    resource = None


def select(train: pd.DataFrame, genres: list[int], seed: int, limit: int) -> pd.DataFrame:
    """Take the scored genres in turn, preferring artists that are not in the sample yet."""
    rows = train[train["valid"]].sort_values("track_id")
    rows = rows.iloc[np.random.default_rng(seed).permutation(len(rows))]
    queues = [[row for row in rows.itertuples() if genre in row.genre_ids] for genre in genres]
    picked, artists = {}, set()
    while len(picked) < limit:
        size = len(picked)
        for queue in queues:
            unseen = (row for row in queue if row.track_id not in picked)
            row = next((row for row in unseen if row.artist_id not in artists), None)
            row = row or next((row for row in queue if row.track_id not in picked), None)
            if row is not None:
                picked[row.track_id] = row
                artists.add(row.artist_id)
            if len(picked) == limit:
                break
        if len(picked) == size:
            break
    return rows.set_index("track_id").loc[list(picked)].reset_index()


def check(track_id, file, settings, errors) -> bool:
    """Run the shared loader on one track and record the reason if it is rejected."""
    try:
        load_windows(file, track_id, **settings)
    except (ValueError, OSError) as error:
        errors[track_id] = str(error)
        return False
    return True


def build(data_dir: Path, ready: Path, target: Path, settings: dict, seed: int, limit: int):
    """Select the sample, read it from the shards and write sample.json, rejects and audio.tar."""
    summary = json.loads((ready / "summary.json").read_text(encoding="utf-8"))
    if digest(ready / "tracks.parquet") != summary["manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch: {ready}.")
    labels = json.loads((ready / "labels.json").read_text(encoding="utf-8"))
    genres = [label["genre_id"] for label in labels if label["scored"]]
    chosen = select(pd.read_parquet(ready / "train.parquet"), genres, seed, limit)

    errors = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sample-", dir=target.parent) as temp:
        stage = Path(temp) / target.name
        stage.mkdir()
        with tarfile.open(stage / "audio.tar", "w") as archive:
            for shard, rows in chosen.groupby("shard", sort=True):
                # Members are read in place; nothing is unpacked on Lustre.
                with tarfile.open(data_dir / shard) as tar:
                    for row in rows.itertuples():
                        try:
                            member = tar.getmember(row.audio_path)
                        except KeyError:
                            errors[row.track_id] = f"Missing from {shard}: {row.audio_path}."
                            continue
                        file = tar.extractfile(member)
                        if check(row.track_id, file, settings, errors):
                            file.seek(0)
                            # One read: tarfile copies in 16 KiB chunks, which is slow on Lustre.
                            archive.addfile(member, io.BytesIO(file.read()))
        tracks = [
            {"track_id": int(row.track_id), "artist_id": int(row.artist_id),
             "genre_ids": [int(genre) for genre in row.genre_ids], "audio_path": row.audio_path}
            for row in chosen.itertuples()
        ]
        sample = {
            "version": summary["version"], "manifest_sha256": summary["manifest_sha256"],
            "profile": "quick", "seed": seed, "settings": settings, "scored_genres": genres,
            "tracks": tracks,
        }
        (stage / "sample.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        rejects = pd.DataFrame({"track_id": list(errors), "error": list(errors.values())})
        rejects.to_csv(stage / "rejects.csv", index=False)
        stage.rename(target)
    return sample, errors


def run(data_dir: Path, version: str = "pre-audit-v1") -> dict:
    """Build the sample on first use, otherwise reload it from audio.tar; time the loader."""
    quick = profile("quick")
    settings = {key: quick[key] for key in ("sample_rate", "window_seconds", "windows_per_track")}
    ready = data_dir / "ready" / version
    target = data_dir / "samples" / version
    started = time.perf_counter()
    if (target / "sample.json").is_file():
        source = "archive"
        sample = json.loads((target / "sample.json").read_text(encoding="utf-8"))
        # Locally only the sample exists; on WCSS a regenerated dataset must not reuse it.
        if ready.is_dir():
            summary = json.loads((ready / "summary.json").read_text(encoding="utf-8"))
            if summary["manifest_sha256"] != sample["manifest_sha256"]:
                raise ValueError(f"{target} was drawn from {sample['version']}, not {ready}.")
        ids = {track["audio_path"]: track["track_id"] for track in sample["tracks"]}
        errors = {}
        with tarfile.open(target / "audio.tar") as tar:
            for member in tar:
                check(ids[member.name], tar.extractfile(member), settings, errors)
        errors |= pd.read_csv(target / "rejects.csv").set_index("track_id")["error"].to_dict()
    else:
        source = "shards"
        sample, errors = build(
            data_dir, ready, target, settings, quick["seed"], quick["max_tracks"],
        )
    seconds = time.perf_counter() - started

    tracks = sample["tracks"]
    tagged = {genre for track in tracks for genre in track["genre_ids"]}
    peak = None
    if resource is not None:
        # ru_maxrss is in bytes on macOS and in KiB on Linux.
        scale = 2**20 if sys.platform == "darwin" else 2**10
        peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale)
    return {
        "sample": str(target), "version": sample["version"], "source": source,
        "tracks": len(tracks), "loaded": len(tracks) - len(errors), "rejected": len(errors),
        "artists": len({track["artist_id"] for track in tracks}),
        "scored_genres_covered": len(tagged & set(sample["scored_genres"])),
        "seconds": round(seconds, 1), "tracks_per_second": round(len(tracks) / seconds, 2),
        "peak_rss_mib": peak, "archive_mib": round((target / "audio.tar").stat().st_size / 2**20),
    }
