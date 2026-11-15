import pandas as pd
import pytest

from music_atlas.data_report import concentration, select_version


def test_concentration_counts_unknown_artists_as_tracks_only():
    rows = pd.DataFrame({
        "split": "test", "genre_id": 1, "artist_id": [1, 1, 1, 2, None],
    })
    stats = concentration(rows).loc[("test", 1)]
    assert stats["Tracks"] == 5 and stats["Artists"] == 2
    assert stats["Top artist (%)"] == pytest.approx(60)
    assert stats["Effective artists"] == pytest.approx(16 / 10)


def test_select_version_prefers_diversity_among_versions_within_limits():
    comparison = pd.DataFrame({
        "Largest artist share (%)": [77, 10, 12, 19],
        "Smallest held-out share (%)": [9.7, 9.6, 8.1, 9.5],
        "Fewest effective artists": [1.7, 14.4, 14.9, 8.5],
        "Max prevalence shift (pp)": [0.2, 0.4, 0.5, 0.3],
    }, index=["fma-v1", "fma-cap10-v1", "fma-penalty4-v1", "fma-cap20-v1"])
    assert select_version(comparison, comparison.index, 20, 9.5) == "fma-cap10-v1"
    assert select_version(comparison, ["fma-v1"], 20, 9.5) is None
