import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# libsndfile garbles the start of each MP3 read at low bitrates, so windows are read with pre-roll.
PREROLL_SECONDS = 1.0


@dataclass(frozen=True)
class Window:
    track_id: int | str
    index: int
    start_seconds: float
    end_seconds: float
    sample_rate: int
    # Mono float32 of exactly window_seconds; zeros after end_seconds for short tracks.
    samples: np.ndarray


def window_starts(frames: int, length: int, count: int) -> list[int]:
    """Centre one window, or spread `count` from start to end, overlapping in short tracks."""
    if frames <= length:
        return [0]
    if count == 1:
        return [(frames - length) // 2]
    return [round(i * (frames - length) / (count - 1)) for i in range(count)]


def load_windows(
    path: str | Path,
    track_id: int | str,
    *,
    sample_rate: int,
    window_seconds: float,
    windows_per_track: int,
) -> list[Window]:
    """Decode only the selected windows of one track, mixed to mono and resampled."""
    if sample_rate <= 0 or window_seconds <= 0 or windows_per_track < 1:
        raise ValueError("Sample rate, window length and window count must be positive.")
    size = round(window_seconds * sample_rate)
    try:
        with sf.SoundFile(path) as file:
            # Positions use native frames, so every target rate receives the same time spans.
            native = file.samplerate
            divisor = math.gcd(native, sample_rate)
            length = round(window_seconds * native)
            preroll = round(PREROLL_SECONDS * native)
            frames = file.frames
            while True:
                windows = []
                for index, start in enumerate(window_starts(frames, length, windows_per_track)):
                    skip = min(start, preroll)
                    file.seek(start - skip)
                    y = file.read(skip + length, dtype="float32", always_2d=True)[skip:]
                    y = y.mean(axis=1)
                    if len(y) < min(length, frames - start):
                        break
                    if not len(y):
                        raise ValueError(f"No audio decoded from {path}.")
                    if not np.isfinite(y).all():
                        raise ValueError(f"Audio contains non-finite samples: {path}.")
                    end = start + len(y)
                    y = resample_poly(y, sample_rate // divisor, native // divisor)
                    samples = np.zeros(size, dtype=np.float32)
                    samples[: min(size, len(y))] = y[:size]
                    windows.append(
                        Window(track_id, index, start / native, end / native, sample_rate, samples)
                    )
                else:
                    return windows
                # MP3 headers can overstate the length; refit the windows to the decoded audio.
                frames = start + len(y)
    except RuntimeError as error:  # sf.LibsndfileError, including missing files
        raise ValueError(f"Cannot decode {path}: {error}") from error


def iter_windows(
    tracks: Iterable[tuple[int | str, str | Path]], **settings
) -> Iterator[tuple[int | str, list[Window], str | None]]:
    """Yield (track_id, windows, error) for each track; a bad track does not stop the loop."""
    for track_id, path in tracks:
        try:
            yield track_id, load_windows(path, track_id, **settings), None
        except (ValueError, OSError) as error:
            yield track_id, [], str(error)
