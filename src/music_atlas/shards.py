"""Read the profile's tracks and process them shard by shard, one part file per shard.

Every catalogue-wide step (acoustic features, log-mel, embeddings, CNN vectors) reads the same
tracks from the same tar shards. Opening a shard takes seconds on Lustre and a job can be
killed, so a step handles one shard at a time and writes `parts/<shard>.npz` last; a rerun
skips every shard whose part exists. The step itself is one function, `work`.
"""

import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from music_atlas.settings import profile
from music_atlas.split_views import digest

SPLITS = ("train", "validation", "test")
# A shard that fails on a Lustre hiccup usually passes on the next try.
PASSES = 2

# work(tar_path, rows, part) writes `part` + ".npz" (and any other files named after `part`)
Work = Callable[[str, pd.DataFrame, Path], None]


def read_tracks(data_dir: Path, profile_name: str) -> tuple[pd.DataFrame, dict, dict]:
    """Tracks with their `split` and `tar`, the dataset identity and the scored genre names.

    The quick profile reads the quick sample from its `audio.tar`, the full profile every
    track of the three partitions.
    """
    version = profile(profile_name)["dataset"]["split"]
    if profile_name == "quick":
        sample_dir = data_dir / "samples" / version
        sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
        rejected = pd.read_csv(sample_dir / "rejects.csv")["track_id"]
        rows = pd.DataFrame(sample["tracks"])
        rows = rows[~rows["track_id"].isin(rejected)]
        rows = rows.assign(split="train", tar=str(sample_dir / "audio.tar"))
        names = sample.get("genre_names", {})
        genres = {genre: names.get(str(genre), str(genre)) for genre in sample["scored_genres"]}
        return rows, sample, genres

    ready = data_dir / "ready" / version
    summary = json.loads((ready / "summary.json").read_text(encoding="utf-8"))
    if digest(ready / "tracks.parquet") != summary["manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch: {ready}.")
    columns = ["track_id", "artist_id", "genre_ids", "shard", "audio_path"]
    rows = pd.concat(
        pd.read_parquet(ready / f"{split}.parquet", columns=columns).assign(split=split)
        for split in SPLITS
    )
    rows["tar"] = [str(data_dir / shard) for shard in rows.pop("shard")]
    labels = json.loads((ready / "labels.json").read_text(encoding="utf-8"))
    genres = {label["genre_id"]: label["name"] for label in labels if label["scored"]}
    return rows, summary, genres


def save(path: Path, write: Callable) -> None:
    """Write beside `path` and rename, so a killed job never leaves a half-written file."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        write(file)
    temporary.replace(path)


def safe_work(work: Work, tar: str, rows: pd.DataFrame, part: Path) -> str | None:
    """Run one shard; return its error instead of stopping the other shards."""
    try:
        work(tar, rows, part)
    except Exception as error:  # reported per shard; the next pass or a rerun retries it
        return f"{type(error).__name__}: {error}"
    return None


def process_shards(rows: pd.DataFrame, parts: Path, work: Work, workers: int = 1,
                   limit: int = 0, *,
                   complete: Callable[[Path, pd.DataFrame], bool] | None = None
                   ) -> tuple[list[str], dict[str, str]]:
    """Run `work` on every shard without a part yet, at most `limit` shards if set.

    Returns the shards finished now and the errors of those that failed every pass.
    """
    parts.mkdir(parents=True, exist_ok=True)
    shards = {Path(tar).stem: (tar, group) for tar, group in rows.groupby("tar", sort=True)}
    def reusable(name):
        path = parts / f"{name}.npz"
        return path.is_file() and (complete is None or complete(path, shards[name][1]))

    todo = [name for name in shards if not reusable(name)][: limit or None]
    finished, failed = [], {}
    for _ in range(PASSES):
        if not todo:
            break
        jobs = [(work, *shards[name], parts / name) for name in todo]
        if workers == 1:  # no subprocesses: simpler to debug and to test
            errors = [safe_work(*job) for job in jobs]
        else:
            # Not fork: a tracked run has MLflow threads, and forking a threaded process can
            # deadlock the child. forkserver starts workers from a clean process instead.
            methods = multiprocessing.get_all_start_methods()
            method = "forkserver" if "forkserver" in methods else "spawn"
            context = multiprocessing.get_context(method)
            with ProcessPoolExecutor(workers, mp_context=context) as pool:
                errors = list(pool.map(safe_work, *zip(*jobs, strict=True)))
        failed = {name: error for name, error in zip(todo, errors, strict=True) if error}
        finished += [name for name in todo if name not in failed]
        todo = list(failed)
    return finished, failed
