import math
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, welch


def analyze(source: str | Path | BinaryIO, sample_rate: int = 22050) -> dict:
    """Inspect at most the first 30 seconds; keep decoding memory bounded."""
    try:
        with sf.SoundFile(source) as file:
            if not 8000 <= file.samplerate <= 192000 or not 1 <= file.channels <= 8:
                raise ValueError("Use audio with 1–8 channels and a sample rate of 8–192 kHz.")
            duration = file.frames / file.samplerate
            native_rate = file.samplerate
            y = file.read(frames=30 * native_rate, dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, RuntimeError) as error:
        raise ValueError("Cannot decode this audio. Try a WAV, FLAC or MP3 file.") from error
    if len(y) < native_rate // 10 or not np.isfinite(y).all() or np.abs(y).max() > 8:
        raise ValueError("Audio must contain at least 0.1 seconds of finite, normalized samples.")
    divisor = math.gcd(native_rate, sample_rate)
    y = resample_poly(y, sample_rate // divisor, native_rate // divisor)
    frequencies, power = welch(y, fs=sample_rate, nperseg=min(2048, len(y)))
    total = float(power.sum())
    centroid = float(np.dot(frequencies, power) / total) if total else 0.0
    chunks = np.array_split(y, min(256, len(y)))
    return {
        "duration_seconds": round(duration, 3),
        "analyzed_seconds": round(len(y) / sample_rate, 3),
        "sample_rate": sample_rate,
        "rms": round(float(np.sqrt(np.mean(y**2))), 6),
        "zero_crossing_rate": round(float(np.mean(np.diff(np.signbit(y)))), 6),
        "spectral_centroid_hz": round(centroid, 2),
        "waveform_min": [round(float(chunk.min()), 5) for chunk in chunks],
        "waveform_max": [round(float(chunk.max()), 5) for chunk in chunks],
        "frequencies_hz": frequencies[::4].round(1).tolist(),
        "spectrum_db": (10 * np.log10(np.maximum(power[::4], 1e-12))).round(2).tolist(),
    }
