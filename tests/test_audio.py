import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from music_atlas.api import app
from music_atlas.audio import analyze


def wav(y, rate=22050):
    file = io.BytesIO()
    sf.write(file, y, rate, format="WAV", subtype="FLOAT")
    file.seek(0)
    return file


def test_tone_preserves_time_level_and_frequency_after_resampling():
    rate = 44100
    t = np.arange(rate) / rate
    result = analyze(wav(0.5 * np.sin(2 * np.pi * 440 * t), rate))
    assert result["analyzed_seconds"] == 1
    assert result["rms"] == pytest.approx(0.5 / np.sqrt(2), abs=0.002)
    assert result["spectral_centroid_hz"] == pytest.approx(440, abs=20)
    assert len(result["waveform_min"]) == 256


def test_silence_and_long_input_stay_finite_and_bounded():
    result = analyze(wav(np.zeros(31 * 8000), 8000))
    assert result["duration_seconds"] == 31
    assert result["analyzed_seconds"] == 30
    assert result["rms"] == result["spectral_centroid_hz"] == 0
    assert np.isfinite(result["spectrum_db"]).all()


def test_api_handles_real_and_invalid_audio():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        invalid = client.post("/api/analyze", files={"file": ("bad.wav", b"broken")})
        assert invalid.status_code == 422
        data = wav(np.zeros(22050)).getvalue()
        response = client.post("/api/analyze", files={"file": ("silence.wav", data)})
        assert response.status_code == 200
        assert response.json()["analyzed_seconds"] == 1


@pytest.mark.parametrize("data", [np.array([]), np.array([float("nan")] * 22050)])
def test_empty_or_non_finite_input_is_rejected(data):
    with pytest.raises(ValueError):
        analyze(wav(data))
