import librosa
import numpy as np
import pytest

from music_atlas.loader import Window
from music_atlas.logmel import (
    FLOOR_DB,
    HOP,
    N_MELS,
    log_mel,
    mel_filters,
    track_tensor,
    valid_frames,
)

RATE = 22050


def window(samples, decoded_seconds=None, rate=RATE):
    seconds = len(samples) / rate if decoded_seconds is None else decoded_seconds
    return Window(1, 0, 0.0, seconds, rate, np.asarray(samples, np.float32))


def test_transform_matches_librosa():
    t = np.arange(10 * RATE) / RATE
    rng = np.random.default_rng(0)
    y = (0.3 * np.sin(2 * np.pi * (100 + 400 * t) * t) + 0.01 * rng.standard_normal(len(t)))
    y = y.astype(np.float32)
    ours = log_mel(y)
    power = librosa.feature.melspectrogram(
        y=y, sr=RATE, n_fft=2048, hop_length=HOP, n_mels=N_MELS, fmin=0, fmax=RATE / 2,
        power=2.0, center=True, pad_mode="constant",
    )
    theirs = librosa.power_to_db(power, ref=1.0, amin=1e-10, top_db=None)
    assert ours.shape == theirs.shape == (N_MELS, 431) and ours.dtype == np.float32
    np.testing.assert_allclose(ours, theirs, atol=0.02)
    reference = librosa.filters.mel(sr=RATE, n_fft=2048, n_mels=N_MELS, fmin=0, fmax=RATE / 2)
    np.testing.assert_allclose(mel_filters(), reference, atol=1e-7)


def test_tone_lands_in_its_band_and_padding_sits_at_the_floor():
    t = np.arange(4 * RATE) / RATE
    samples = np.zeros(10 * RATE, np.float32)
    samples[: len(t)] = 0.5 * np.sin(2 * np.pi * 440 * t)
    short = window(samples, decoded_seconds=4)
    x, valid = track_tensor([short])
    assert x.shape == (1, 1, N_MELS, 431)
    assert valid.tolist() == [valid_frames(short)] == [-(-4 * RATE // HOP)]
    band = mel_filters()[:, round(440 * 2048 / RATE)].argmax()
    assert abs(int(x[0, 0, :, 50].argmax()) - band) <= 1
    # Frames that only see zero padding are exactly the clipping floor, like digital silence.
    assert (x[0, 0, :, valid[0] + 2 :] == FLOOR_DB).all()
    assert (log_mel(np.zeros(RATE, np.float32)) == FLOOR_DB).all()
    with pytest.raises(ValueError, match="22050"):
        track_tensor([window(samples, rate=16000)])
