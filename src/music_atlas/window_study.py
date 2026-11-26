"""Window length study: log-mel of the same quick-sample tracks at 5, 10 and 20 s.

Only `window_seconds` changes; the sample rate, the transform and the window's centre stay
fixed, so each shorter window lies inside the longer ones. Every length decodes its own window
from the same bytes, which times decoding and the transform without the tar read. The whole
track's average spectrum is the reference for how much of the track a window represents.
"""

import io
import tarfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from music_atlas.loader import load_windows, open_audio
from music_atlas.logmel import FLOOR_DB, HOP, N_FFT, N_MELS, SAMPLE_RATE, log_mel, valid_frames
from music_atlas.logmel_cache import SILENT_SHARE
from music_atlas.shards import read_tracks

LENGTHS = (5, 10, 20)
# About a minute in whole hops: the whole track is transformed in chunks of this size.
CHUNK = 2560 * HOP


def track_log_mel(samples: np.ndarray) -> np.ndarray:
    """`log_mel` of a whole track in chunks, frame for frame as one call, in far less memory.

    Each chunk carries half an FFT of real audio on both sides, so no frame sees chunk padding.
    """
    edge = N_FFT // 2
    padded = np.pad(samples, edge)
    frames = 1 + len(samples) // HOP
    chunks = [log_mel(padded[start : start + CHUNK + 2 * edge])[:, edge // HOP :][:, : CHUNK // HOP]
              for start in range(0, len(samples), CHUNK)]
    return np.concatenate(chunks, axis=1)[:, :frames]


def onset_times(x: np.ndarray) -> np.ndarray:
    """Onsets librosa finds in a whole track's log-mel, in seconds.

    Detected once per track and counted inside each window: per-window detection rescales its
    threshold to the window's loudest change, so a single click hides every other onset.
    """
    import librosa

    envelope = librosa.onset.onset_strength(S=x, sr=SAMPLE_RATE, hop_length=HOP)
    return librosa.onset.onset_detect(onset_envelope=envelope, sr=SAMPLE_RATE, hop_length=HOP,
                                      units="time")


def spectrum(x: np.ndarray) -> np.ndarray:
    """Mean dB per band over frames with any sound, so digital silence does not pull it down."""
    sound = x[:, (x > FLOOR_DB).any(axis=0)].astype(np.float64)
    return sound.mean(axis=1) if sound.shape[1] else np.full(N_MELS, FLOOR_DB)


def gaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean absolute dB difference between every spectrum in `a` and every one in `b`."""
    return np.abs(a[:, None] - b[None]).mean(axis=2)


def window(data: bytes, track_id: int, seconds: float):
    settings = {"sample_rate": SAMPLE_RATE, "window_seconds": seconds, "windows_per_track": 1}
    [result] = load_windows(io.BytesIO(data), track_id, **settings)
    return result


def measure(data: bytes, track_id: int) -> tuple[list[dict], dict, dict]:
    """Rows, spectrograms and average spectra of one track's windows; spectra key 0 = whole."""
    with open_audio(io.BytesIO(data)) as file:
        track = window(data, track_id, file.frames / file.samplerate)
    whole = track_log_mel(track.samples)
    spectra, onsets = {0: spectrum(whole)}, onset_times(whole)
    rows, spectrograms = [], {}
    for length in LENGTHS:
        started = time.perf_counter()
        cut = window(data, track_id, length)
        decoded = time.perf_counter()
        x = log_mel(cut.samples)
        finished = time.perf_counter()
        # Again under tracemalloc, which would slow the timed pass; numpy reports its arrays.
        tracemalloc.start()
        log_mel(window(data, track_id, length).samples)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        n = valid_frames(cut)
        spectra[length] = spectrum(x[:, :n])
        rows.append({
            "track_id": track_id, "window_seconds": length, "track_seconds": track.end_seconds,
            "start_seconds": cut.start_seconds, "frames": x.shape[1], "valid_frames": n,
            "decode_seconds": decoded - started, "transform_seconds": finished - decoded,
            "peak_mib": peak / 2**20,
            "onsets": int(((onsets >= cut.start_seconds) & (onsets < cut.end_seconds)).sum()),
            "floor_share": float((x[:, :n] <= FLOOR_DB).mean()) if n else 1.0,
            "gap_db": float(np.abs(spectra[length] - spectra[0]).mean()),
        })
        spectrograms[length] = x.astype(np.float16)
    return rows, spectrograms, spectra


def summarise(windows: pd.DataFrame, spectra: dict) -> pd.DataFrame:
    """The results table, one row per window length.

    `own_track_share` is how often a window's spectrum is nearer its own track's than any
    other sample track's: a relative measure, lower with more candidates.
    """
    ids = list(spectra)
    tracks = np.array([spectra[track_id][0] for track_id in ids])
    rows = []
    for length, group in windows.groupby("window_seconds"):
        frames = int(group["frames"].iloc[0])
        nearest = gaps(np.array([spectra[track_id][length] for track_id in ids]), tracks)
        rows.append({
            "window_seconds": length, "frames": frames,
            "tensor_kib": N_MELS * frames * 4 / 2**10,  # one float32 window, as the CNN gets it
            "decode_ms": 1000 * group["decode_seconds"].mean(),
            "transform_ms": 1000 * group["transform_seconds"].mean(),
            "peak_mib": group["peak_mib"].median(), "onsets": group["onsets"].median(),
            "gap_db": group["gap_db"].median(),
            "own_track_share": (nearest.argmin(axis=1) == np.arange(len(ids))).mean(),
            "padded_tracks": int((group["valid_frames"] < frames).sum()),
            "silent_windows": int((group["floor_share"] >= SILENT_SHARE).sum()),
        })
    return pd.DataFrame(rows).round(3)


def draw_examples(windows: pd.DataFrame, spectrograms: dict, titles: dict, path: Path) -> Path:
    """The same tracks at every length on the track's time axis; grey is what a window misses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = windows[windows["window_seconds"].eq(LENGTHS[0])].set_index("track_id")
    choices = [
        ("typical gap to the whole track", -(short["gap_db"] - short["gap_db"].median()).abs()),
        ("largest gap to the whole track", short["gap_db"]),
        ("most onsets", short["onsets"]),
        ("shortest track", -short["track_seconds"]),
    ]
    picks = {}
    for note, values in choices:
        values = values.drop(list(picks.values()))  # four different tracks
        if len(values):
            picks[note] = values.idxmax()
    by_window = windows.set_index(["track_id", "window_seconds"])
    fig, axes = plt.subplots(len(picks), len(LENGTHS), figsize=(13, 2.4 * len(picks)),
                             sharey=True, squeeze=False, layout="constrained")
    for row_axes, (note, track_id) in zip(axes, picks.items(), strict=True):
        start = by_window.loc[(track_id, LENGTHS[-1]), "start_seconds"]
        for ax, length in zip(row_axes, LENGTHS, strict=True):
            row = by_window.loc[(track_id, length)]
            image = ax.imshow(
                spectrograms[track_id, length].astype(np.float32), origin="lower", aspect="auto",
                cmap="magma", vmin=FLOOR_DB, vmax=40,
                extent=(row["start_seconds"], row["start_seconds"] + length, 0, N_MELS),
            )
            if row["valid_frames"] < row["frames"]:
                ax.axvline(row["start_seconds"] + row["valid_frames"] * HOP / SAMPLE_RATE,
                           color="white", ls="--", lw=1)
            ax.set_xlim(start, start + LENGTHS[-1])
            ax.set_facecolor("0.85")
            ax.set_title(f"{length} s: {row['onsets']:.0f} onsets, {row['gap_db']:.1f} dB from "
                         "the whole track", fontsize=9, loc="left")
        row_axes[0].set_ylabel(f"{track_id}: {note}\n{titles.get(track_id, '')}\nmel band",
                               fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time in the track (s)")
    fig.colorbar(image, ax=axes, shrink=0.6, label=f"dB re full scale (floor {FLOOR_DB:.0f})")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run(data_dir: Path, outputs_dir: Path, record=None) -> dict:
    """Measure every length on every quick-sample track and write the table and examples.

    With `record` (a `music_atlas.tracking.Run`) the files go to its run directory, the
    results by length to its metrics and the table and figure to `evaluation/`.
    """
    rows, dataset, genres = read_tracks(data_dir, "quick")
    target = record.directory if record is not None else outputs_dir / "window-study"
    target.mkdir(parents=True, exist_ok=True)
    table, spectrograms, spectra, errors = [], {}, {}, {}
    for tar_path, group in rows.groupby("tar"):
        with tarfile.open(tar_path) as tar:
            for row in group.itertuples():
                try:
                    data = tar.extractfile(row.audio_path).read()
                    track_rows, track_spectrograms, spectra[row.track_id] = measure(
                        data, row.track_id)
                except (KeyError, ValueError, OSError) as error:
                    errors[row.track_id] = str(error)
                    continue
                table += track_rows
                spectrograms |= {(row.track_id, length): x
                                 for length, x in track_spectrograms.items()}
    if not table:
        raise ValueError(f"No quick-sample track could be measured: {errors}")

    windows = pd.DataFrame(table)
    windows.to_csv(target / "tracks.csv", index=False)
    results = summarise(windows, spectra)
    results.to_csv(target / "results.csv", index=False)
    titles = {row.track_id: ", ".join(genres[genre] for genre in row.genre_ids if genre in genres)
              for row in rows.itertuples()}
    examples = draw_examples(windows, spectrograms, titles, target / "examples.png")
    by_length = {f"w{row.pop('window_seconds')}": row for row in results.to_dict("records")}
    whole = np.array([track[0] for track in spectra.values()])
    # How far apart two different tracks are, to read the windows' gaps against.
    between = float(np.median(gaps(whole, whole)[np.triu_indices(len(whole), 1)]))
    measured = {"rejected": len(errors), "between_tracks_gap_db": round(between, 3), **by_length}

    if record is not None:
        record.log_dataset(rows, "quick-sample", genres, {
            "window_seconds": list(LENGTHS), "sample_rate": SAMPLE_RATE, "windows_per_track": 1,
        })
        record.log(measured)
        record.log_file(target / "results.csv", "evaluation")
        record.log_file(examples, "evaluation")
    return {"run": str(target), "version": dataset["version"], "tracks": len(rows), **measured}
