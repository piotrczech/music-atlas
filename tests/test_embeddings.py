import io
import json
import multiprocessing
import os
import signal
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from music_atlas import embeddings, musicfm
from music_atlas.embedding_pilot import pilot_report
from music_atlas.embeddings import CHECKPOINTS, DIMENSIONS, RATE, load_encoder, verify_vectors
from music_atlas.loader import Window
from music_atlas.representation import UNIT_MEAN, aggregate, extract
from music_atlas.settings import profile
from music_atlas.split_views import digest
from music_atlas.tracking import track


def test_checkpoint_preprocessing_matches_transformers_without_downloading(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers")
    import huggingface_hub

    processor = hf.ClapFeatureExtractor(truncation="rand_trunc", padding="repeatpad")
    captured = []

    class Projection(torch.nn.Module):
        config = SimpleNamespace(projection_dim=512, enable_fusion=False)

        def forward(self, input_features):
            assert not self.training and torch.is_inference_mode_enabled()
            captured.append(input_features.numpy())
            return SimpleNamespace(audio_embeds=torch.ones(len(input_features), 512))

    def snapshot(repo_id, revision, **kwargs):
        assert (repo_id, revision) == CHECKPOINTS["clap-general"]
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(hf.ClapFeatureExtractor, "from_pretrained", lambda *a, **k: processor)
    info = {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}
    monkeypatch.setattr(
        hf.ClapAudioModelWithProjection, "from_pretrained", lambda *a, **k: (Projection(), info)
    )
    encoder, _ = load_encoder(*CHECKPOINTS["clap-general"], torch.device("cpu"))
    wave = np.zeros((2, RATE * 10), dtype=np.float32)
    wave[1, 100:200] = 0.5
    reference = processor(wave, sampling_rate=RATE, return_tensors="np")["input_features"]
    assert encoder(wave).shape == (2, 512)
    np.testing.assert_allclose(captured[0], reference, atol=2e-4, rtol=1e-6)
    short = wave[1, : int(3.5 * RATE)]
    window = Window(1, 0, 0, 3.5, RATE, wave[1])
    padded = embeddings.repeatpad(window)
    expected = processor(short, sampling_rate=RATE, return_tensors="np")["input_features"]
    encoder(padded[None, :])
    np.testing.assert_allclose(captured[-1], expected, atol=2e-4, rtol=1e-6)
    with pytest.raises(ValueError, match="10-second"):
        encoder(wave[:, :-1])


def test_pilot_neighbours_exclude_artist_and_split_group(tmp_path):
    (tmp_path / "parts").mkdir()
    ids = np.arange(1, 7)
    np.save(tmp_path / "track_ids.npy", ids)
    np.save(tmp_path / "vectors.npy", [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
    np.savez(tmp_path / "parts" / "audio.npz", raw_norms=np.arange(1, 7))
    rows = pd.DataFrame(
        {
            "track_id": ids,
            "artist_id": [1, 1, 2, 3, 4, 5],
            "split_group": [1, 1, 1, 3, 4, 5],
            "genre_ids": [[10], [10], [10], [11], [11], []],
        }
    )
    report = pilot_report(tmp_path, rows, {10: "Rock", 11: "Blues"})
    query = report["examples"][0]
    assert query["track_id"] == 1 and [r["track_id"] for r in query["neighbours"]] == [4, 5]
    assert report["queries"] == 5 and report["raw_window_norm"]["median"] == 3.5
    assert report["shared_genre_cosine"]["median"] == 1
    assert report["no_shared_tag_cosine"]["median"] == 0
    assert report["random_neighbour_tag_match"] == pytest.approx(0.1)
    assert report["constant_score_macro_ap"] == 0.5


def test_musicfm_loads_pinned_weights_and_pools_frames(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from torch import nn
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerEncoder,
    )

    config = transformers.Wav2Vec2ConformerConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=12,
        num_conv_pos_embedding_groups=1,
        position_embeddings_type="rotary",
    )
    config_path = tmp_path / "config.json"
    config.to_json_file(config_path)
    stats_path = tmp_path / "msd_stats.json"
    stats_path.write_text('{"melspec_2048_mean":0,"melspec_2048_std":1}', encoding="utf-8")
    state = {
        f"model.conformer.{k}": v for k, v in Wav2Vec2ConformerEncoder(config).state_dict().items()
    }
    for block in range(2):
        for conv_id in (1, 2, 3):
            conv = nn.Conv2d(1, 1, 3)
            norm = nn.BatchNorm2d(1)
            prefix = f"model.conv.conv.{block}"
            state.update({f"{prefix}.conv{conv_id}.{k}": v for k, v in conv.state_dict().items()})
            state.update({f"{prefix}.bn{conv_id}.{k}": v for k, v in norm.state_dict().items()})
    state.update({f"model.conv.linear.{k}": v for k, v in nn.Linear(32, 8).state_dict().items()})
    frontend = "model.preprocessor_melspec_2048.mel_stft"
    state[f"{frontend}.spectrogram.window"] = torch.hann_window(2048)
    state[f"{frontend}.mel_scale.fb"] = torch.rand(1025, 128)
    weights = tmp_path / "pretrained_msd.pt"
    torch.save({"state_dict": state}, weights)
    monkeypatch.setattr(musicfm, "DIMENSIONS", 8)

    def cached_files(*, offline):
        assert offline
        return weights, stats_path, config_path

    monkeypatch.setattr(musicfm, "checkpoint_files", cached_files)
    encode, _ = musicfm.load_encoder("cpu")
    wave = np.random.default_rng(42).uniform(-0.2, 0.2, (2, 240000)).astype(np.float32)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        vectors = encode(wave)
        assert vectors.shape == (2, 8)
        np.testing.assert_allclose(encode(wave[:1])[0], vectors[0], atol=1e-5)
    finally:
        torch.set_num_threads(old_threads)
    assert np.isfinite(vectors).all()
    with pytest.raises(ValueError, match="24 kHz"):
        encode(wave[:, ::2])


def test_unit_mean_gives_equal_weight_to_each_window():
    windows = np.array([[3, 0], [0, 7]], dtype=np.float32)
    np.testing.assert_allclose(aggregate(windows, UNIT_MEAN), np.full(2, 2**-0.5), rtol=1e-6)


@pytest.mark.parametrize("windows", [[[0, 0]], [[1, 0], [-1, 0]], [[np.nan, 1]], [[1, np.inf]]])
def test_invalid_embeddings_have_a_reason(windows):
    with pytest.raises(ValueError, match="(Zero|finite)"):
        aggregate(np.array(windows), UNIT_MEAN)


def input_data(root, members=((11, "b", 0.25), (24, "a", 0.75))):
    data = root / "d"
    ready = data / "ready" / "v1"
    ready.mkdir(parents=True)
    (ready / "summary.json").write_text(json.dumps({"version": "v1", "manifest_sha256": "abc"}))
    tracks = []
    for track_id, shard, level in members:
        audio = io.BytesIO()
        sf.write(audio, np.full(25 * 8000, level), 8000, format="WAV", subtype="FLOAT")
        content = audio.getvalue()
        member = tarfile.TarInfo(f"{track_id}.wav")
        member.size = len(content)
        with tarfile.open(data / f"{shard}.tar", "w") as archive:
            archive.addfile(member, io.BytesIO(content))
        tracks.append((track_id, f"{shard}.tar", member.name))
    return {"train": pd.DataFrame(tracks, columns=["track_id", "shard", "audio_path"])}


def encode(samples):
    values = np.ones((len(samples), DIMENSIONS), dtype="float32")
    values[:, 0] = samples.mean(axis=1)
    return values


@pytest.mark.parametrize("encoder", ["clap-general", "musicfm-msd"])
def test_full_extraction_and_shard_estimate_on_synthetic_manifest(tmp_path, monkeypatch, encoder):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "runs").as_uri())
    rows = input_data(tmp_path, [(11, "b", 0.25), (24, "a", 0.75), (6, "c", 0.4)])["train"]
    rows["split"] = ["train", "validation", "test"]
    rows["artist_id"] = rows.track_id
    rows["genre_ids"] = [[1]] * len(rows)
    storage = {"data": tmp_path / "d", "outputs": tmp_path / "o", "artifacts": tmp_path / "a"}
    alias = profile("full")["dataset"]["split"]
    ready = storage["data"] / "ready" / alias
    ready.mkdir()
    (ready / "labels.json").write_text('[{"genre_id":1,"name":"Rock","scored":true}]')
    rows.to_parquet(ready / "tracks.parquet")
    for name, subset in rows.groupby("split"):
        subset.to_parquet(ready / f"{name}.parquet")
    summary = {"version": "synthetic", "manifest_sha256": digest(ready / "tracks.parquet")}
    (ready / "summary.json").write_text(json.dumps(summary))
    if encoder == "musicfm-msd":
        rate, dimensions = musicfm.RATE, musicfm.DIMENSIONS

        def musicfm_encode(samples):
            assert samples.shape[1] == 240000
            return np.tile(encode(samples), (1, 2))

        monkeypatch.setattr(musicfm, "load_encoder", lambda _: (musicfm_encode, {}))
    else:
        rate, dimensions = RATE, DIMENSIONS
        monkeypatch.setattr(embeddings, "load_encoder", lambda *args: (encode, {}))
    result = embeddings.run(storage, encoder, "full", "cpu", 2, True, windows_per_track=5)
    full = storage["artifacts"] / encoder / result["run_id"]
    assert np.load(full / "track_ids.npy").tolist() == [6, 11, 24]
    meta = json.loads((full / "metadata.json").read_text())
    assert meta["manifest_sha256"] == summary["manifest_sha256"] and meta["tracks"] == 3
    assert meta["subsets"] == ["train", "validation", "test"]
    assert meta["sample_rate"] == rate and meta["dimensions"] == dimensions
    window_rows = pd.read_parquet(full / "windows.parquet")
    assert window_rows.groupby("track_id").size().tolist() == [5, 5, 5]
    raw = np.load(full / "window_vectors.npy")
    saved = np.load(full / "vectors.npy")
    for block, vector in zip(raw.reshape(3, 5, dimensions), saved, strict=True):
        np.testing.assert_allclose(aggregate(block, UNIT_MEAN), vector)
    coverage = json.loads((full / "evaluation" / "coverage.json").read_text())
    assert coverage["shape"] == [3, dimensions] and coverage["failed"] == 0
    assert coverage["window_shape"] == [15, dimensions]
    assert all(p == {"expected": 1, "encoded": 1} for p in coverage["partitions"].values())
    assert coverage["sha256"]["vectors.npy"] == digest(full / "vectors.npy")
    bench = embeddings.run(storage, encoder, "full", "cpu", 2, False, benchmark_shards=1)
    root = storage["artifacts"] / encoder / bench["run_id"]
    estimate = json.loads((root / "evaluation" / "benchmark.json").read_text())
    assert estimate["catalogue_tracks"] == 3 and estimate["benchmark_tracks"] == 1
    assert estimate["source_shards"] == ["b.tar"] and estimate["estimated_gpu_hours"] == 0
    assert estimate["vectors_and_ids_bytes"] == 3 * (dimensions * 4 + 8)


def extract_test(root, rows, directory, encoder=encode, record=None, batch_size=2):
    data = root / "d"
    inputs = pd.concat(
        [r.assign(split=k, tar=[str(data / s) for s in r.shard]) for k, r in rows.items()]
    )
    return extract(
        data,
        directory,
        encoder,
        8000,
        {"dimensions": DIMENSIONS},
        profile_name="full",
        windows_per_track=3,
        record=record,
        inputs=(inputs, {"version": "v1", "manifest_sha256": "abc"}, {}),
        aggregation=UNIT_MEAN,
        batch_size=batch_size,
    )


def test_parts_alignment_repair_and_configuration(tmp_path):
    rows = input_data(tmp_path)
    directory = tmp_path / "vectors"
    extract_test(tmp_path, rows, directory)
    verify_vectors(directory, rows)
    expected = np.load(directory / "vectors.npy")
    np.testing.assert_allclose(expected[:, 0] / expected[:, 1], [0.25, 0.75])
    corrupt = directory / "parts" / "a.npz"
    corrupt.write_bytes(b"interrupted")
    calls = []

    def counted(samples):
        calls.append(len(samples))
        return encode(samples)

    extract_test(tmp_path, rows, directory, counted)
    assert calls == [2, 1]
    np.testing.assert_array_equal(np.load(directory / "vectors.npy"), expected)
    with pytest.raises(ValueError, match="configuration"):
        extract_test(tmp_path, rows, directory, batch_size=1)
    with pytest.raises(ValueError, match="unique"):
        extract_test(tmp_path, {**rows, "test": rows["train"]}, directory)
    wrong = extract_test(tmp_path, rows, tmp_path / "wrong", lambda x: np.ones((len(x), 3)))
    assert all("Encoder shape" in reason for reason in wrong["failed"].values())


def test_retry_track_failure(tmp_path):
    rows = input_data(tmp_path)
    directory = tmp_path / "out"
    result = extract_test(
        tmp_path, rows, directory, lambda x: encode(x) * (x.mean(axis=1, keepdims=True) > 0.5)
    )
    assert result["failed"] == 1
    failed = pd.read_csv(directory / "failed.csv")
    assert failed.track_id.tolist() == [11] and "Zero" in failed.error[0]
    assert extract_test(tmp_path, rows, directory)["failed"] == 0
    verify_vectors(directory, rows)


def tracked_worker(root, rows, kill=False, resume=None):
    storage = {"data": root / "d", "artifacts": root / "a", "outputs": root / "o"}
    with track("embeddings", "quick", storage, "v1", "synthetic", resume=resume) as run:

        def worker(samples):
            if kill and (run.directory / "parts" / "a.npz").exists():
                if hasattr(signal, "SIGKILL"):
                    os.kill(os.getpid(), signal.SIGKILL)
                os._exit(137)
            return encode(samples)

        extract_test(root, rows, run.directory, worker, run)
        return run.directory


def test_hard_kill_and_tracked_resume(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "runs").as_uri())
    rows = input_data(tmp_path)
    child = multiprocessing.get_context("spawn").Process(
        target=tracked_worker, args=(tmp_path, rows, True)
    )
    child.start()
    child.join(90)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("Timed out before the first shard was published")
    assert child.exitcode in (-9, 137)
    (directory,) = (tmp_path / "a" / "synthetic").iterdir()
    first = json.loads((directory / "run.json").read_text())
    assert first["status"] == "RUNNING"
    completed = directory / "parts" / "a.npz"
    before = completed.stat().st_mtime_ns
    assert tracked_worker(tmp_path, rows, resume=directory.name) == directory
    assert completed.stat().st_mtime_ns == before
    clean = tracked_worker(tmp_path, rows)
    for name in ("vectors.npy", "track_ids.npy", "window_vectors.npy"):
        np.testing.assert_array_equal(np.load(clean / name), np.load(directory / name))
    final = json.loads((directory / "run.json").read_text())
    assert final["status"] == "FINISHED" and final["datasets"] == ["train"]
    assert mlflow.get_run(first["mlflow_run_id"]).info.status == "KILLED"
    assert final["tags"]["atlas.resumed_from"] == first["mlflow_run_id"]
    assert len(mlflow.get_run(first["mlflow_run_id"]).inputs.dataset_inputs) == 1
    assert not mlflow.get_run(final["mlflow_run_id"]).inputs.dataset_inputs
