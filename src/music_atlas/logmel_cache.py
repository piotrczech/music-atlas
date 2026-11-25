"""Bounded log-mel cache for CNN training: one float16 array per tar shard.

Training reads every window once per epoch and decoding the MP3 costs several times more than the
transform, so windows are decoded once here. The API applies music_atlas.logmel on demand."""

import json
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from music_atlas.loader import load_windows
from music_atlas.logmel import (
    FLOOR_DB,
    FMAX,
    FMIN,
    HOP,
    N_MELS,
    PARAMETERS,
    SAMPLE_RATE,
    hz_to_mel,
    log_mel,
    mel_to_hz,
    track_tensor,
)
from music_atlas.settings import profile
from music_atlas.split_views import digest, record_filter
from music_atlas.tracking import git_revision, usage

SPLITS = ("train", "validation", "test")
# One row per cached window, in the order of the part's array; level and floor share are
# measured over the decoded frames only.
COLUMNS = {"track_id": np.int64, "split": str, "window": np.int16, "start_seconds": np.float32,
           "valid_frames": np.int32, "mean_db": np.float32, "floor_share": np.float32}
PASSES = 3  # a shard that fails on a Lustre hiccup usually passes on the next try


def tracks(data_dir: Path, profile_name: str) -> tuple[pd.DataFrame, dict, dict]:
    """Tracks to cache with their tar, the dataset identity and the scored genres' names."""
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


def save(path: Path, write) -> None:
    """Write beside the target and rename, so a part never exists half-written."""
    part = path.with_name(path.name + ".part")
    with part.open("wb") as file:
        write(file)
    part.replace(path)


def cache_part(tar_path: str, rows: pd.DataFrame, settings: dict, target: Path) -> str | None:
    """Cache one tar's windows; its .npz index is written last, so a rerun skips finished parts."""
    blocks, index, errors = [], [], {}
    sums, frames, seconds = np.zeros((2, N_MELS)), 0, np.zeros(2)
    try:
        with tarfile.open(tar_path) as tar:
            for row in rows.itertuples():
                started = time.perf_counter()
                try:
                    file = tar.extractfile(row.audio_path)
                    windows = load_windows(file, row.track_id, **settings)
                except (KeyError, ValueError, OSError) as error:
                    errors[row.track_id] = str(error)
                    continue
                decoded = time.perf_counter()
                x, valid = track_tensor(windows)
                seconds += decoded - started, time.perf_counter() - decoded
                blocks.append(x[:, 0].astype(np.float16))
                for window, spectrogram, n in zip(windows, x[:, 0], valid, strict=True):
                    audio = spectrogram[:, :n]
                    index.append((row.track_id, row.split, window.index, window.start_seconds, n,
                                  audio.mean() if n else FLOOR_DB,
                                  (audio <= FLOOR_DB).mean() if n else 1.0))
                    if row.split == "train":  # the normaliser sees training audio, no padding
                        values = audio.astype(np.float64)
                        sums += values.sum(axis=1), (values**2).sum(axis=1)
                        frames += int(n)
        array = np.concatenate(blocks) if blocks else np.zeros((0, N_MELS, 0), np.float16)
        table = pd.DataFrame(index, columns=list(COLUMNS))
        arrays = {column: table[column].to_numpy(dtype) for column, dtype in COLUMNS.items()}
        arrays |= {"error_ids": np.array(list(errors), np.int64),
                   "errors": np.array(list(errors.values()), str),
                   "sums": sums, "frames": frames, "seconds": seconds}
        save(target.with_suffix(".npy"), lambda file: np.save(file, array))
        save(target.with_suffix(".npz"), lambda file: np.savez(file, **arrays))
    except Exception as error:  # report the part and continue; a rerun retries it
        return f"{type(error).__name__}: {error}"
    print(f"{Path(tar_path).name}: {len(index)} windows in {seconds.sum():.0f} s", file=sys.stderr)
    return None


def draw_examples(index: pd.DataFrame, arrays: dict, window_seconds: float, path: Path) -> Path:
    """Windows to look at before trusting the cache: level extremes, padding, floor, silence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full = index[index["valid_frames"].eq(index["valid_frames"].max())]
    typical = full.iloc[int((full["mean_db"] - full["mean_db"].median()).abs().argmin())]
    picks = [(typical, "median level")]
    for column, pick, note in (("mean_db", "idxmax", "loudest"), ("mean_db", "idxmin", "quietest"),
                               ("valid_frames", "idxmin", "shortest; padding after the line"),
                               ("floor_share", "idxmax", "most values at the floor")):
        # Each panel shows another window, e.g. the shortest track is not repeated for the floor.
        rest = index.drop([row.name for row, _ in picks])
        if rest.empty:
            break
        picks.append((rest.loc[getattr(rest[column], pick)()], note))
    silence = log_mel(np.zeros(round(window_seconds * SAMPLE_RATE), np.float32))
    centres = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))[1:-1]
    ticks = [250, 1000, 4000, 8000]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True,
                             layout="constrained")
    panels = [*picks, (None, "digital silence")]
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (row, note) in zip(axes.flat, panels, strict=False):
        x = silence if row is None else arrays[row["part"]][row["row"]].astype(np.float32)
        image = ax.imshow(x, origin="lower", aspect="auto", cmap="magma", vmin=FLOOR_DB, vmax=40,
                          extent=(0, x.shape[1] * HOP / SAMPLE_RATE, -0.5, N_MELS - 0.5))
        if row is None:
            ax.set_title(f"{note}: the value of padding", fontsize=9, loc="left")
            continue
        if row["valid_frames"] < x.shape[1]:
            ax.axvline(row["valid_frames"] * HOP / SAMPLE_RATE, color="white", ls="--", lw=1)
        ax.set_title(f"{row['track_id']} ({row['split']}): {note}\nmean {row['mean_db']:.0f} dB, "
                     f"{row['floor_share']:.0%} at the floor", fontsize=9, loc="left")
    axes[0, 0].set_yticks(np.interp(ticks, centres, np.arange(N_MELS)), [f"{t:,}" for t in ticks])
    for ax in axes[:, 0]:
        ax.set_ylabel("mel band (Hz)")
    for ax in axes[1]:
        ax.set_xlabel("time in the window (s)")
    label = f"power, dB re full scale (floor {FLOOR_DB:.0f})"
    fig.colorbar(image, ax=axes, shrink=0.8, label=label)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run(data_dir: Path, artifacts_dir: Path, profile_name: str = "quick", workers: int = 4,
        limit: int = 0, record=None) -> dict:
    """Cache the profile's windows part by part, then write the index, normaliser and metadata.

    With `record` (a `music_atlas.tracking.Run`) the cache lives in its run directory, the
    partitions become its datasets and cost and storage its metrics. A cache with parts still
    missing fails the run, so `--resume` continues it instead of it looking complete.
    """
    settings = profile(profile_name)
    windows = {key: settings[key] for key in ("sample_rate", "window_seconds", "windows_per_track")}
    rows, identity, genres = tracks(data_dir, profile_name)
    short_version = settings["dataset"]["split"]
    target = (record.directory if record is not None
              else artifacts_dir / "logmel" / f"{short_version}-{profile_name}")
    parts = target / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    config = {"representation": "logmel", "version": identity["version"],
              "manifest_sha256": identity["manifest_sha256"], "profile": profile_name,
              "windows": windows, "transform": PARAMETERS, "tracks": len(rows)}
    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in config} != config:
            raise ValueError(f"{target} holds a cache with other settings; start a new run.")
    else:
        metadata_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    groups = {Path(tar).stem: (tar, group) for tar, group in rows.groupby("tar", sort=True)}
    todo = [stem for stem in groups if not (parts / f"{stem}.npz").exists()][: limit or None]
    started, failed, fresh = time.perf_counter(), {}, []
    for _ in range(PASSES):
        if not todo:
            break
        with ProcessPoolExecutor(workers) as pool:
            results = pool.map(
                cache_part, [groups[stem][0] for stem in todo], [groups[stem][1] for stem in todo],
                [windows] * len(todo), [parts / stem for stem in todo],
            )
            failed = {stem: error for stem, error in zip(todo, results, strict=True) if error}
        fresh += [stem for stem in todo if stem not in failed]
        todo = list(failed)
    elapsed = time.perf_counter() - started

    done = [stem for stem in groups if (parts / f"{stem}.npz").exists()]
    index, errors, sums, frames, seconds = [], {}, np.zeros((2, N_MELS)), 0, np.zeros(2)
    for stem in done:
        with np.load(parts / f"{stem}.npz") as part:
            index.append(pd.DataFrame({
                "part": stem, "row": np.arange(len(part["track_id"])),
                **{column: part[column] for column in COLUMNS},
            }))
            errors |= dict(zip(part["error_ids"].tolist(), part["errors"].tolist(), strict=True))
            sums += part["sums"]
            frames += int(part["frames"])
            seconds += part["seconds"]
    index = pd.concat(index, ignore_index=True) if index else pd.DataFrame(columns=list(COLUMNS))
    count = max(len(index), 1)
    fresh_tracks = sum(len(groups[stem][1]) for stem in fresh)
    measured = {
        "rejected": len(errors), "failed_parts": len(failed),
        "seconds_per_window": {"decode": round(seconds[0] / count, 4),
                               "transform": round(seconds[1] / count, 4)},
        "tracks_per_second": round(fresh_tracks / elapsed, 2) if fresh_tracks else None,
    }
    if len(done) < len(groups):
        progress = {"parts": len(groups), "parts_cached": len(done), **measured}
        if record is None:
            return {"run": str(target), **progress, "failed": failed}
        record.log(progress)
        raise ValueError(f"{len(groups) - len(done)} of {len(groups)} parts are not cached yet; "
                         f"continue with --resume {record.run_id}.")

    index.to_parquet(target / "index.parquet", index=False)
    rejects = target / "rejects.csv"
    pd.DataFrame({"track_id": list(errors), "error": list(errors.values())}).to_csv(
        rejects, index=False)
    if profile_name == "full":  # the split's filter.parquet beside the run, as for every model
        record_filter(data_dir, target.parent.parent, short_version, "logmel", target.name,
                      rejects)
    mean = sums[0] / max(frames, 1)
    std = np.sqrt(np.maximum(sums[1] / max(frames, 1) - mean**2, 0))
    storage = sum(path.stat().st_size for path in parts.glob("*.npy"))
    arrays = {stem: np.load(parts / f"{stem}.npy", mmap_mode="r") for stem in done}
    examples = draw_examples(index, arrays, windows["window_seconds"], target / "examples.png")
    window_frames = 1 + round(windows["window_seconds"] * SAMPLE_RATE) // HOP
    counts = {split: {"tracks": int(group["track_id"].nunique()), "windows": len(group),
                      "short_windows": int(group["valid_frames"].lt(window_frames).sum())}
              for split, group in index.groupby("split")}
    metadata = {
        **config, "git_revision": git_revision(), "counts": counts, "rejected": len(errors),
        "storage": {"bytes": storage, "layout": "parts/<shard>.npy holds (windows, n_mels, "
                    "frames) float16 dB; index.parquet maps its rows to tracks and windows"},
        "normaliser": {"fitted_on": "train", "frames": frames,
                       "mean": mean.round(4).tolist(), "std": std.round(4).tolist()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    measured["storage_kib_per_track"] = round(storage / 2**10 / index["track_id"].nunique(), 1)
    if record is not None:
        rejected = set(errors)
        for split, group in rows.groupby("split"):
            subset = "quick-sample" if profile_name == "quick" else split
            record.log_dataset(group, subset, genres, {
                "settings": windows, **counts.get(split, {}),
                "rejected": len(rejected & set(group["track_id"])),
            })
        record.log_file(rejects, "dataset")
        record.log_file(examples, "dataset")
        record.log(measured)
    return {"run": str(target), "version": identity["version"], "tracks": len(rows),
            "windows": len(index), **measured, "storage_mib": round(storage / 2**20),
            "seconds": round(elapsed, 1), "peak_rss_mib": usage().get("peak_rss_mib"),
            "failed": failed}


def read(run_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    """Return the window index, one memory-mapped array per part and the run metadata."""
    index = pd.read_parquet(run_dir / "index.parquet")
    arrays = {part: np.load(run_dir / "parts" / f"{part}.npy", mmap_mode="r")
              for part in index["part"].unique()}
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return index, arrays, metadata
