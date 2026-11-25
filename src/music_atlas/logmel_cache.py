"""Log-mel cache for CNN training: every window of every track, one float16 array per shard.

Training reads every window in every epoch and decoding an MP3 costs about nine times the
transform, so windows are decoded once here (docs/EXPERIMENTS.md, Log-mel spectrograms).

A run directory holds:
- `parts/<shard>.npy`: (windows, n_mels, frames) float16 dB, memory-mapped by `read`;
- `index.parquet`: one row per window, pointing at its part and row;
- `metadata.json`: transform parameters, counts and the normaliser fitted on train;
- `rejects.csv`, `examples.png` and, in the full profile, the split's `filter.parquet`.
"""

import json
import sys
import tarfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from music_atlas.loader import load_windows
from music_atlas.logmel import FLOOR_DB, HOP, N_MELS, PARAMETERS, SAMPLE_RATE, track_tensor
from music_atlas.settings import profile
from music_atlas.shards import process_shards, read_tracks, save
from music_atlas.split_views import record_filter
from music_atlas.tracking import git_revision, usage

# One row per window. Level and floor share are measured on decoded frames, without padding.
COLUMNS = {"track_id": np.int64, "split": str, "window": np.int16, "start_seconds": np.float32,
           "valid_frames": np.int32, "mean_db": np.float32, "floor_share": np.float32}
# A window this much at the floor is digital silence, mostly a track's silent ending. It stays
# in the cache; training skips it through index.parquet.
SILENT_SHARE = 0.99


def cache_shard(tar_path: str, rows: pd.DataFrame, part: Path, settings: dict) -> None:
    """Write the shard's spectrograms to `<part>.npy`, then its window index to `<part>.npz`."""
    spectrograms, windows, errors = [], [], {}
    # Sums of training values and their squares per band, for the normaliser.
    sums, squares, frames = np.zeros(N_MELS), np.zeros(N_MELS), 0
    decode_seconds = transform_seconds = 0.0
    with tarfile.open(tar_path) as tar:
        for row in rows.itertuples():
            started = time.perf_counter()
            try:
                track = load_windows(tar.extractfile(row.audio_path), row.track_id, **settings)
            except (KeyError, ValueError, OSError) as error:
                errors[row.track_id] = str(error)
                continue
            decoded = time.perf_counter()
            x, valid = track_tensor(track)
            decode_seconds += decoded - started
            transform_seconds += time.perf_counter() - decoded

            for window, spectrogram, n in zip(track, x[:, 0], valid, strict=True):
                spectrograms.append(spectrogram.astype(np.float16))
                audio = spectrogram[:, :n].astype(np.float64)
                windows.append((row.track_id, row.split, window.index, window.start_seconds, n,
                                audio.mean() if n else FLOOR_DB,
                                (audio <= FLOOR_DB).mean() if n else 1.0))
                if row.split == "train":
                    sums += audio.sum(axis=1)
                    squares += (audio**2).sum(axis=1)
                    frames += int(n)

    array = np.stack(spectrograms) if spectrograms else np.zeros((0, N_MELS, 0), np.float16)
    table = pd.DataFrame(windows, columns=list(COLUMNS))
    index = {column: table[column].to_numpy(dtype) for column, dtype in COLUMNS.items()}
    save(part.with_suffix(".npy"), lambda file: np.save(file, array))
    save(part.with_suffix(".npz"), lambda file: np.savez(
        file, **index, error_ids=np.array(list(errors), np.int64),
        errors=np.array(list(errors.values()), str), sums=sums, squares=squares, frames=frames,
        seconds=np.array([decode_seconds, transform_seconds]),
    ))
    print(f"{Path(tar_path).name}: {len(windows)} windows", file=sys.stderr)


def combine(parts: Path, names: list[str]) -> tuple[pd.DataFrame, dict, dict]:
    """Join the parts' window indexes, rejected tracks and sums into one table and two dicts."""
    tables, errors = [], {}
    totals = {"sums": np.zeros(N_MELS), "squares": np.zeros(N_MELS), "frames": 0,
              "seconds": np.zeros(2)}
    for name in names:
        with np.load(parts / f"{name}.npz") as part:
            table = pd.DataFrame({column: part[column] for column in COLUMNS})
            tables.append(table.assign(part=name, row=np.arange(len(table))))
            errors |= dict(zip(part["error_ids"].tolist(), part["errors"].tolist(), strict=True))
            for key in totals:
                totals[key] = totals[key] + part[key]
    index = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=[*COLUMNS])
    return index, errors, totals


def draw_examples(index: pd.DataFrame, arrays: dict, genres: dict, path: Path) -> Path:
    """Windows to look at before trusting the cache, each titled with its track's genres."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audible = index[index["floor_share"] < SILENT_SHARE]
    if audible.empty:
        audible = index
    choices = [
        ("median level", (audible["mean_db"] - audible["mean_db"].median()).abs(), "idxmin"),
        ("loudest", audible["mean_db"], "idxmax"),
        ("quietest", audible["mean_db"], "idxmin"),
        ("shortest; padding after the line", index["valid_frames"], "idxmin"),
        ("most values at the floor", index["floor_share"], "idxmax"),
    ]
    picks = {}
    for note, values, how in choices:
        values = values.drop(list(picks.values()))  # five different windows
        if len(values):
            picks[note] = getattr(values, how)()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True,
                             layout="constrained")
    for ax, (note, i) in zip(axes.flat, picks.items(), strict=False):
        row = index.loc[i]
        x = arrays[row["part"]][row["row"]].astype(np.float32)
        image = ax.imshow(x, origin="lower", aspect="auto", cmap="magma", vmin=FLOOR_DB, vmax=40,
                          extent=(0, x.shape[1] * HOP / SAMPLE_RATE, 0, N_MELS))
        if row["valid_frames"] < x.shape[1]:
            ax.axvline(row["valid_frames"] * HOP / SAMPLE_RATE, color="white", ls="--", lw=1)
        ax.set_title(f"{row['track_id']} ({row['split']}): {note}\n"
                     f"{genres.get(row['track_id'], 'no scored genre')}, "
                     f"mean {row['mean_db']:.0f} dB", fontsize=9, loc="left")
    axes.flat[-1].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("mel band")
    for ax in axes[1]:
        ax.set_xlabel("time in the window (s)")
    fig.colorbar(image, ax=axes, shrink=0.8, label=f"dB re full scale (floor {FLOOR_DB:.0f})")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run(data_dir: Path, artifacts_dir: Path, profile_name: str = "quick", workers: int = 4,
        limit: int = 0, record=None, windows_per_track: int | None = None) -> dict:
    """Cache the profile's windows shard by shard, then write the index, normaliser and metadata.

    With `record` (a `music_atlas.tracking.Run`) the cache lives in its run directory, the
    partitions become its datasets and cost and storage its metrics. A cache with shards still
    missing fails the run, so `--resume` continues it instead of it looking complete.
    `windows_per_track` replaces the profile's count, e.g. 5 windows (50 s) against 3 (30 s).
    """
    config = profile(profile_name)
    settings = {key: config[key] for key in ("sample_rate", "window_seconds", "windows_per_track")}
    if windows_per_track:
        settings["windows_per_track"] = windows_per_track
    rows, dataset, genre_names = read_tracks(data_dir, profile_name)
    version = config["dataset"]["split"]
    name = f"{version}-{profile_name}" + (f"-w{windows_per_track}" if windows_per_track else "")
    target = record.directory if record is not None else artifacts_dir / "logmel" / name
    target.mkdir(parents=True, exist_ok=True)

    # A rerun must continue the same cache, not mix in parts made with other settings.
    identity = {"representation": "logmel", "version": dataset["version"],
                "manifest_sha256": dataset["manifest_sha256"], "profile": profile_name,
                "windows": settings, "transform": PARAMETERS, "tracks": len(rows)}
    metadata_path = target / "metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in identity} != identity:
            raise ValueError(f"{target} holds a cache with other settings; start a new run.")
    else:
        metadata_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    parts = target / "parts"
    started = time.perf_counter()
    work = partial(cache_shard, settings=settings)
    finished, failed = process_shards(rows, parts, work, workers, limit)
    seconds = time.perf_counter() - started

    shards = sorted({Path(tar).stem for tar in rows["tar"]})
    done = [name for name in shards if (parts / f"{name}.npz").exists()]
    index, errors, totals = combine(parts, done)
    new_tracks = rows["tar"].map(lambda tar: Path(tar).stem).isin(finished).sum()
    windows = max(len(index), 1)
    measured = {
        "rejected": len(errors), "failed_parts": len(failed),
        "seconds_per_window": {"decode": round(totals["seconds"][0] / windows, 4),
                               "transform": round(totals["seconds"][1] / windows, 4)},
        # Only this attempt's shards, so a resumed run still measures real throughput.
        "tracks_per_second": round(new_tracks / seconds, 2) if new_tracks else None,
    }
    if len(done) < len(shards):
        progress = {"parts": len(shards), "parts_cached": len(done), **measured}
        if record is None:
            return {"run": str(target), **progress, "failed": failed}
        record.log(progress)
        raise ValueError(f"{len(shards) - len(done)} of {len(shards)} parts are not cached yet; "
                         f"continue with --resume {record.run_id}.")

    index.to_parquet(target / "index.parquet", index=False)
    rejects = target / "rejects.csv"
    pd.DataFrame({"track_id": list(errors), "error": list(errors.values())}).to_csv(
        rejects, index=False)
    if profile_name == "full":  # the split's filter.parquet beside the run, as for every model
        record_filter(data_dir, target.parent.parent, version, "logmel", target.name, rejects)

    frames = max(int(totals["frames"]), 1)
    mean = totals["sums"] / frames
    std = np.sqrt(np.maximum(totals["squares"] / frames - mean**2, 0))
    full_window = 1 + round(settings["window_seconds"] * SAMPLE_RATE) // HOP
    counts = {split: {"tracks": int(group["track_id"].nunique()), "windows": len(group),
                      "short_windows": int((group["valid_frames"] < full_window).sum()),
                      "silent_windows": int((group["floor_share"] >= SILENT_SHARE).sum())}
              for split, group in index.groupby("split")}
    storage = sum(path.stat().st_size for path in parts.glob("*.npy"))
    metadata = {
        **identity, "git_revision": git_revision(), "counts": counts, "rejected": len(errors),
        "storage": {"bytes": storage, "layout": "parts/<shard>.npy holds (windows, n_mels, "
                    "frames) float16 dB; index.parquet maps its rows to tracks and windows"},
        "normaliser": {"fitted_on": "train", "frames": int(totals["frames"]),
                       "mean": mean.round(4).tolist(), "std": std.round(4).tolist()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    arrays = {name: np.load(parts / f"{name}.npy", mmap_mode="r") for name in done}
    track_genres = {row.track_id: ", ".join(genre_names[g] for g in row.genre_ids
                                            if g in genre_names) or "no scored genre"
                    for row in rows.itertuples()}
    examples = draw_examples(index, arrays, track_genres, target / "examples.png")
    measured["storage_kib_per_track"] = round(storage / 2**10 / index["track_id"].nunique(), 1)

    if record is not None:
        for split, group in rows.groupby("split"):
            subset = "quick-sample" if profile_name == "quick" else split
            record.log_dataset(group, subset, genre_names, {
                "settings": settings, **counts.get(split, {}),
                "rejected": int(group["track_id"].isin(errors).sum()),
            })
        record.log_file(rejects, "dataset")
        record.log_file(examples, "dataset")
        record.log(measured)
    return {"run": str(target), "version": dataset["version"], "tracks": len(rows),
            "windows": len(index), **measured, "storage_mib": round(storage / 2**20),
            "seconds": round(seconds, 1), "peak_rss_mib": usage().get("peak_rss_mib"),
            "failed": failed}


def read(run_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    """Return the window index, one memory-mapped array per part and the run metadata."""
    index = pd.read_parquet(run_dir / "index.parquet")
    arrays = {part: np.load(run_dir / "parts" / f"{part}.npy", mmap_mode="r")
              for part in index["part"].unique()}
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    return index, arrays, metadata
