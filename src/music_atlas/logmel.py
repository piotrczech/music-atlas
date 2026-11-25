"""Log-mel spectrograms for the CNN, in numpy and scipy only so the API image applies the same
transform as training without the research extra."""

from functools import cache

import numpy as np
from scipy.signal import get_window

from music_atlas.loader import Window

SAMPLE_RATE = 22050
N_FFT = 2048  # 93 ms
HOP = 512  # 23 ms; a 10 s window gives 431 frames
N_MELS = 128
FMIN, FMAX = 0.0, SAMPLE_RATE / 2
# Power is in units of a full-scale signal, not of each recording's peak, so a quiet track stays
# quiet. Power below AMIN, including the zero padding of short tracks, is clipped to -100 dB.
REFERENCE = 1.0
AMIN = 1e-10
FLOOR_DB = 10 * np.log10(AMIN / REFERENCE)

PARAMETERS = {
    "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop_length": HOP, "window": "hann",
    "center": True, "pad_mode": "constant", "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
    "mel_scale": "slaney", "mel_norm": "slaney", "power": 2.0, "reference": REFERENCE,
    "amin": AMIN, "floor_db": FLOOR_DB, "layout": "(windows, 1, n_mels, frames) float32",
}


def hz_to_mel(hz):
    """Slaney's scale: linear below 1 kHz, logarithmic above (librosa's default)."""
    hz = np.asarray(hz, dtype=np.float64)
    return np.where(hz < 1000, hz * 3 / 200, 15 + np.log(np.maximum(hz, 1000) / 1000) * 27
                    / np.log(6.4))


def mel_to_hz(mel):
    mel = np.asarray(mel, dtype=np.float64)
    return np.where(mel < 15, mel * 200 / 3, 1000 * np.exp((mel - 15) * np.log(6.4) / 27))


@cache
def mel_filters() -> np.ndarray:
    """Triangular filters of unit area, shape (n_mels, n_fft // 2 + 1), as librosa builds them."""
    bins = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    edges = mel_to_hz(np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2))
    ramps = edges[:, None] - bins[None, :]
    widths = np.diff(edges)
    rising = -ramps[:-2] / widths[:-1, None]
    falling = ramps[2:] / widths[1:, None]
    weights = np.maximum(0, np.minimum(rising, falling))
    weights *= (2 / (edges[2:] - edges[:-2]))[:, None]
    return weights.astype(np.float32)


def log_mel(samples: np.ndarray) -> np.ndarray:
    """Return the log-mel power of mono samples at SAMPLE_RATE in dB, shape (n_mels, frames)."""
    if samples.ndim != 1 or not len(samples):
        raise ValueError("Log-mel needs a nonempty mono signal.")
    # Centred frames: frame t covers samples t * HOP ± N_FFT / 2, zeros beyond the signal.
    padded = np.pad(samples.astype(np.float32, copy=False), N_FFT // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP]
    window = get_window("hann", N_FFT).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = mel_filters() @ power.T
    return (10 * np.log10(np.maximum(mel, AMIN) / REFERENCE)).astype(np.float32)


def valid_frames(window: Window) -> int:
    """Frames centred on decoded audio; later frames of a short track only see zero padding."""
    samples = len(window.samples)
    decoded = min(samples, round((window.end_seconds - window.start_seconds) * window.sample_rate))
    return min(1 + samples // HOP, -(-decoded // HOP))


def track_tensor(windows: list[Window]) -> tuple[np.ndarray, np.ndarray]:
    """Stack a track's windows as (windows, 1, n_mels, frames) with each window's valid frames."""
    if not windows:
        raise ValueError("A track needs at least one window.")
    if any(window.sample_rate != SAMPLE_RATE for window in windows):
        raise ValueError(f"Log-mel windows must be sampled at {SAMPLE_RATE} Hz.")
    x = np.stack([log_mel(window.samples) for window in windows])[:, None]
    return x, np.array([valid_frames(window) for window in windows], dtype=np.int32)


def normalise(x: np.ndarray, normaliser: dict) -> np.ndarray:
    """Standardise each mel band with the training mean and std recorded beside the cache."""
    mean = np.asarray(normaliser["mean"], np.float32)[:, None]
    std = np.maximum(np.asarray(normaliser["std"], np.float32), 1e-3)[:, None]
    return (np.asarray(x, np.float32) - mean) / std
