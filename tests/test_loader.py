import numpy as np
import pytest
import soundfile as sf

from music_atlas.loader import iter_windows, load_windows, window_starts

QUICK = {"sample_rate": 22050, "window_seconds": 10, "windows_per_track": 1}
FULL = {**QUICK, "windows_per_track": 3}


def tone(seconds, rate=44100):
    t = np.arange(round(seconds * rate)) / rate
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def write(path, y, rate=44100, **options):
    sf.write(path, y, rate, **options)
    return path


def bounds(windows):
    return [(window.start_seconds, window.end_seconds) for window in windows]


def test_windows_cover_beginning_middle_and_end_or_overlap_when_short():
    assert window_starts(100, 10, 1) == [45]
    assert window_starts(100, 10, 3) == [0, 45, 90]
    assert window_starts(25, 10, 3) == [0, 8, 15]
    assert window_starts(10, 10, 3) == window_starts(4, 10, 3) == [0]


def test_stereo_track_gives_aligned_mono_windows_at_target_rate(tmp_path):
    y = tone(60)
    path = write(tmp_path / "stereo.wav", np.stack([y, np.zeros_like(y)], axis=1), subtype="FLOAT")
    windows = load_windows(path, 7, **FULL)
    assert [(window.track_id, window.index) for window in windows] == [(7, 0), (7, 1), (7, 2)]
    assert bounds(windows) == [(0, 10), (25, 35), (50, 60)]
    for window in windows:
        assert window.sample_rate == 22050
        assert window.samples.shape == (220500,) and window.samples.dtype == np.float32
        rms = np.sqrt(np.mean(window.samples[100:-100] ** 2))
        assert rms == pytest.approx(0.25 / np.sqrt(2), rel=0.01)
        peak = np.argmax(np.abs(np.fft.rfft(window.samples)))
        assert peak * 22050 / 220500 == pytest.approx(440, abs=0.2)


def test_window_positions_do_not_depend_on_target_rate(tmp_path):
    path = write(tmp_path / "track.wav", tone(37))
    by_rate = {
        rate: load_windows(path, 1, **{**FULL, "sample_rate": rate}) for rate in (16000, 48000)
    }
    assert bounds(by_rate[16000]) == bounds(by_rate[48000]) == [(0, 10), (13.5, 23.5), (27, 37)]
    assert by_rate[48000][1].samples.shape == (480000,)


def test_short_track_gives_one_zero_padded_window(tmp_path):
    path = write(tmp_path / "short.wav", tone(4), subtype="FLOAT")
    [window] = load_windows(path, 1, **FULL)
    assert (window.start_seconds, window.end_seconds) == (0, 4)
    assert window.samples.shape == (220500,)
    assert np.abs(window.samples[100 : 4 * 22050 - 100]).max() > 0.45
    assert not window.samples[4 * 22050 :].any()


def test_silent_track_stays_finite(tmp_path):
    path = write(tmp_path / "silence.wav", np.zeros(30 * 8000), rate=8000)
    windows = load_windows(path, 1, **FULL)
    assert len(windows) == 3
    assert all(np.isfinite(window.samples).all() and not window.samples.any() for window in windows)


def test_non_finite_samples_are_rejected(tmp_path):
    y = tone(12)
    y[6 * 44100] = np.nan
    path = write(tmp_path / "nan.wav", y, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        load_windows(path, 1, **QUICK)


def test_bad_tracks_are_reported_and_the_catalogue_continues(tmp_path):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio")
    tracks = [
        (1, broken),
        (2, tmp_path / "missing.wav"),
        (3, write(tmp_path / "empty.wav", np.zeros(0))),
        (4, write(tmp_path / "good.wav", tone(3))),
    ]
    results = list(iter_windows(tracks, **QUICK))
    assert [track_id for track_id, _, _ in results] == [1, 2, 3, 4]
    assert all(error and not windows for _, windows, error in results[:3])
    assert results[3][2] is None and len(results[3][1]) == 1


def chirp_mp3(path, seconds, rate=44100, **options):
    t = np.arange(seconds * rate) / rate
    y = 0.3 * np.sin(2 * np.pi * (220 + 10 * t) * t)
    return write(path, y, rate, format="MP3", subtype="MPEG_LAYER_III", **options)


needs_mp3 = pytest.mark.skipif(
    "MP3" not in sf.available_formats(), reason="libsndfile lacks MP3 support"
)


@needs_mp3
def test_low_bitrate_mp3_windows_match_a_continuous_decode(tmp_path):
    # Without pre-roll errors reach 0.3; atol allows float32 rounding by decoder start.
    path = chirp_mp3(tmp_path / "chirp.mp3", 30, compression_level=0.99, bitrate_mode="CONSTANT")
    continuous, rate = sf.read(path, dtype="float32")
    windows = load_windows(path, 1, **{**FULL, "sample_rate": rate})
    assert windows[-1].end_seconds == len(continuous) / rate
    for window in windows:
        start = round(window.start_seconds * rate)
        expected = continuous[start : start + 10 * rate]
        np.testing.assert_allclose(window.samples, expected, rtol=0, atol=1e-6)


@needs_mp3
def test_truncated_mp3_windows_stay_within_the_decoded_audio(tmp_path):
    path = chirp_mp3(tmp_path / "full.mp3", 60, bitrate_mode="VARIABLE")
    truncated = tmp_path / "truncated.mp3"
    truncated.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 3])
    assert sf.info(truncated).duration == pytest.approx(60, abs=0.1)
    decoded = len(sf.read(truncated)[0]) / 44100
    windows = load_windows(truncated, 1, **FULL)
    assert len(windows) == 3 and 15 < decoded < 25
    assert windows[-1].end_seconds == pytest.approx(decoded)
    assert all(np.abs(window.samples).max() > 0.2 for window in windows)
