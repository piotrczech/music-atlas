"""Frozen LAION-CLAP audio projections on the shared FMA windows."""

import math
import os
import tempfile
import time
from contextlib import nullcontext
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

from music_atlas.representation import UNIT_MEAN, extract
from music_atlas.settings import profile
from music_atlas.shards import SPLITS, read_tracks
from music_atlas.split_views import digest
from music_atlas.tracking import dataset_identity, track, usage, write_json

CHECKPOINTS = {
    "clap-general": ("laion/clap-htsat-unfused", "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"),
    "clap-music": ("laion/larger_clap_music", "a0b4534a14f58e20944452dff00a22a06ce629d1"),
}
RATE = 48_000
DIMENSIONS = 512


def repeatpad(window):
    """Apply CLAP repeatpad to the decoded interval, preserving the loader's window positions."""
    samples = window.samples
    length = math.ceil(round((window.end_seconds - window.start_seconds) * window.sample_rate, 7))
    if length <= 0:
        raise ValueError("Empty CLAP window.")
    if length >= len(samples):
        return samples
    repeated = np.tile(samples[:length], len(samples) // length)
    return np.pad(repeated, (0, len(samples) - len(repeated)))


def load_encoder(checkpoint, revision, device):
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import ClapAudioModelWithProjection, ClapFeatureExtractor

    started = time.perf_counter()
    files = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
    kwargs = {"repo_id": checkpoint, "revision": revision, "allow_patterns": files}
    try:
        snapshot = Path(snapshot_download(**kwargs, local_files_only=True))
        cached = all((snapshot / name).is_file() for name in files)
    except LocalEntryNotFoundError:
        cached = False
    if not cached:
        snapshot = Path(snapshot_download(**kwargs))
    fetched = time.perf_counter()
    processor = ClapFeatureExtractor.from_pretrained(snapshot, local_files_only=True)
    model, loading = ClapAudioModelWithProjection.from_pretrained(
        snapshot,
        local_files_only=True,
        output_loading_info=True,
    )
    if (
        loading["missing_keys"]
        or loading["mismatched_keys"]
        or loading["error_msgs"]
        or model.config.projection_dim != DIMENSIONS
        or model.config.enable_fusion
        or processor.truncation != "rand_trunc"
        or processor.padding != "repeatpad"
        or processor.sampling_rate != RATE
        or processor.nb_max_samples != 10 * RATE
    ):
        raise ValueError(f"Checkpoint differs from the verified unfused CLAP recipe: {checkpoint}")
    model.to(device).eval()
    hann = torch.hann_window(processor.fft_window_size, dtype=torch.float64, device=device)
    filters = torch.as_tensor(processor.mel_filters_slaney, device=device)

    def features(samples):
        if samples.ndim != 2 or samples.shape[1] != processor.nb_max_samples:
            raise ValueError("CLAP expects shared 10-second mono windows at 48 kHz.")
        waveform = torch.as_tensor(samples, dtype=torch.float64, device=device)
        spectrum = torch.stft(
            waveform,
            processor.fft_window_size,
            processor.hop_length,
            window=hann,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        # The reference stores FFT bins as complex64 before calculating power in float64.
        power = spectrum.to(torch.complex64).abs().double().square().transpose(1, 2)
        return (10 * torch.log10((power @ filters).clamp_min(1e-10))).float().unsqueeze(1)

    # Verify the batched implementation against this exact checkpoint's CPU processor.
    probe = np.random.default_rng(42).normal(0, 0.1, (1, 10 * RATE)).astype("float32")
    reference = processor(probe, sampling_rate=RATE, return_tensors="np")["input_features"]
    np.testing.assert_allclose(features(probe).cpu().numpy(), reference, atol=2e-4, rtol=1e-6)

    @torch.inference_mode()
    def encode(samples):
        return model(input_features=features(samples)).audio_embeds.float().cpu().numpy()

    return encode, {
        "checkpoint_cache_hit": cached,
        "checkpoint_fetch_seconds": 0.0 if cached else fetched - started,
        "snapshot_seconds": fetched - started,
        "model_init_seconds": time.perf_counter() - fetched,
    }


def verify_vectors(directory, subsets):
    """Reopen published arrays and verify alignment and unit lengths in bounded blocks."""
    ids = np.load(directory / "track_ids.npy", mmap_mode="r")
    vectors = np.load(directory / "vectors.npy", mmap_mode="r")
    windows = pd.read_parquet(directory / "windows.parquet")
    window_vectors = np.load(directory / "window_vectors.npy", mmap_mode="r")
    expected = np.sort(np.concatenate([rows.track_id for rows in subsets.values()]))
    failures = pd.read_csv(directory / "failed.csv", usecols=["track_id"])
    if (
        not np.array_equal(
            np.sort(np.r_[ids, failures.track_id.to_numpy(dtype=np.int64)]), expected
        )
        or np.any(ids[1:] <= ids[:-1])
        or ids.dtype != np.int64
        or vectors.shape != (len(ids), DIMENSIONS)
        or vectors.dtype != np.float32
    ):
        raise ValueError("Published embedding IDs, dtype or shape do not match the manifest.")
    distinct, starts, counts = np.unique(windows.track_id, return_index=True, return_counts=True)
    if (
        not np.array_equal(distinct, ids)
        or not np.array_equal(windows.track_id, np.repeat(distinct, counts))
        or not np.array_equal(windows.window, np.arange(len(windows)) - np.repeat(starts, counts))
        or window_vectors.shape != (len(windows), DIMENSIONS)
        or window_vectors.dtype != np.float32
    ):
        raise ValueError(
            "Published window vectors are not aligned with their track and window IDs."
        )
    for start in range(0, len(windows), 4096):
        if not np.isfinite(window_vectors[start : start + 4096]).all():
            raise ValueError("Published window vectors contain non-finite values.")
    max_norm_error = 0.0
    for start in range(0, len(ids), 4096):
        lengths = np.linalg.norm(vectors[start : start + 4096], axis=1)
        if not np.allclose(lengths, 1.0, rtol=0, atol=1e-5):
            raise ValueError("Published vectors have non-finite or non-unit L2 norms.")
        max_norm_error = max(max_norm_error, float(np.max(np.abs(lengths - 1))))
    return {
        "shape": list(vectors.shape),
        "window_shape": list(window_vectors.shape),
        "max_unit_norm_error": max_norm_error,
        "failed": len(failures),
        "partitions": {
            name: {"expected": len(rows), "encoded": int(np.isin(rows.track_id, ids).sum())}
            for name, rows in subsets.items()
        },
    }


def run(
    storage,
    encoder,
    profile_name,
    device,
    batch_size,
    tracked,
    resume=None,
    benchmark_shards=0,
    windows_per_track=None,
):
    import torch

    from music_atlas import loader, representation, shards

    started = time.perf_counter()
    cfg = profile(profile_name)
    alias = cfg["dataset"]["split"]
    identity = dataset_identity(storage["data"], alias)
    ready = storage["data"] / "ready" / alias
    quick = profile_name == "quick"
    window_count = cfg["windows_per_track"] if windows_per_track is None else windows_per_track
    if window_count < 1:
        raise ValueError("Expected at least one window.")
    if benchmark_shards < 0 or (quick and benchmark_shards):
        raise ValueError("Benchmark shards are a positive subset of the full profile.")
    if batch_size < 1 or cfg["window_seconds"] * RATE != 480000:
        raise ValueError("Expected positive batch size and shared 10-second windows.")
    if quick and ready.exists() and digest(ready / "tracks.parquet") != identity["manifest_sha256"]:
        raise ValueError("The canonical manifest does not match its published hash.")
    rows, dataset, labels = read_tracks(storage["data"], profile_name)
    if (dataset["version"], dataset["manifest_sha256"]) != (
        identity["version"],
        identity["manifest_sha256"],
    ):
        raise ValueError("The input tracks and canonical manifest refer to different versions.")
    if quick:
        if ready.exists():
            training = pd.read_parquet(ready / "train.parquet", columns=["track_id", "split_group"])
            rows = rows.merge(training, on="track_id", how="left", validate="1:1", indicator=True)
            if not rows.pop("_merge").eq("both").all():
                raise ValueError("A quick sample ID is outside the training partition.")
    else:
        canonical = pd.read_parquet(ready / "tracks.parquet", columns=["track_id", "split"])
        expected = canonical.dropna(subset=["split"]).set_index("track_id").split
        for name in SPLITS:
            partition_ids = expected[expected == name].index
            if not np.array_equal(
                np.sort(rows.loc[rows.split.eq(name), "track_id"]), np.sort(partition_ids)
            ):
                raise ValueError(f"{name} IDs differ from the canonical manifest.")
    if benchmark_shards:
        tars = sorted(rows.tar.unique())
        if benchmark_shards >= len(tars):
            raise ValueError("Benchmark must use fewer shards than the entire catalogue.")
        selected = {
            tars[(2 * i + 1) * len(tars) // (2 * benchmark_shards)] for i in range(benchmark_shards)
        }
        rows = rows[rows.tar.isin(selected)]
    subsets = (
        {"quick-sample": rows} if quick else {name: rows[rows.split.eq(name)] for name in SPLITS}
    )

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no GPU is visible.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    checkpoint, revision = CHECKPOINTS[encoder]
    params = {
        "encoder": encoder,
        "checkpoint": checkpoint,
        "revision": revision,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "aggregation": UNIT_MEAN,
        "sample_rate": RATE,
        "batch_size": batch_size,
        "dimensions": DIMENSIONS,
        "benchmark_shards": benchmark_shards,
        "preprocessing": "CLAP repeatpad, Slaney log-mel",
        "windows_per_track": window_count,
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "scipy_version": package_version("scipy"),
        "libsndfile_version": loader.sf.__libsndfile_version__,
        "transformers_version": package_version("transformers"),
        "source_sha256": digest(Path(__file__)),
        "loader_sha256": digest(Path(loader.__file__)),
        "representation_sha256": digest(Path(representation.__file__)),
        "shards_sha256": digest(Path(shards.__file__)),
    }
    context = (
        track(
            "embeddings",
            profile_name,
            storage,
            alias,
            encoder,
            description="Frozen audio encoder on shared FMA windows.",
            resume=resume,
            overrides=params,
        )
        if tracked or resume
        else nullcontext()
    )
    with context as record:
        if record:
            directory = record.directory
        else:
            base = storage["artifacts"] / encoder
            base.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix=f"embeddings-{profile_name}-", dir=base))
        print(f"Run: {directory.name}", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        encode, timings = load_encoder(checkpoint, revision, device)
        metadata = {
            **params,
            "run_id": directory.name,
            "dataset": cfg["dataset"]["name"],
            "split_alias": alias,
            "dataset_version": identity["version"],
            "profile": profile_name,
            "manifest_sha256": identity["manifest_sha256"],
            "window_selection": "native-frame positions from music_atlas.loader",
            "normalisation": "unit L2",
            "seed": cfg["seed"],
        }
        result = extract(
            storage["data"],
            directory,
            encode,
            RATE,
            metadata,
            profile_name,
            windows_per_track=window_count,
            record=record,
            prepare=repeatpad,
            inputs=(rows, dataset, labels),
            batch_size=batch_size,
            aggregation=UNIT_MEAN,
        )
        if "encoded" not in result:
            raise ValueError(f"Incomplete shard extraction: {result['failed']}")
        coverage = verify_vectors(directory, subsets)
        metrics = {**timings, **usage(), "total_seconds": time.perf_counter() - started}
        if device.type == "cuda":
            metrics["gpu_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            metrics["gpu_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        evaluation = directory / "evaluation"
        evaluation.mkdir(exist_ok=True)
        if quick:
            from music_atlas.embedding_pilot import pilot_report

            report_path = evaluation / "pilot.json"
            report = pilot_report(directory, subsets["quick-sample"], labels)
            write_json(report_path, report)
            if record:
                record.log_file(report_path, "evaluation")
                record.log({k: v for k, v in report.items() if k not in ("examples", "queries")})
        elif benchmark_shards and result["tracks_per_second"] and not result["failed"]:
            seconds = len(expected) / result["tracks_per_second"]
            minutes = math.ceil((seconds * 1.3 + 300) / 60)
            estimate = {
                "source_shards": [
                    Path(tar).relative_to(storage["data"]).as_posix() for tar in sorted(selected)
                ],
                "benchmark_tracks": result["encoded"],
                "catalogue_tracks": len(expected),
                "estimated_extraction_hours": seconds / 3600,
                "estimated_gpu_hours": seconds / 3600 if device.type == "cuda" else 0,
                "suggested_walltime": f"{minutes // 60:02}:{minutes % 60:02}:00",
                "vectors_and_ids_bytes": len(expected) * (4 * DIMENSIONS + 8),
                "estimated_window_files_bytes": round(
                    sum(
                        (directory / name).stat().st_size
                        for name in ("window_vectors.npy", "windows.parquet")
                    )
                    * len(expected)
                    / result["encoded"]
                ),
                "estimated_parts_bytes": round(
                    sum(p.stat().st_size for p in (directory / "parts").glob("*.npz"))
                    * len(expected)
                    / result["encoded"]
                ),
            }
            write_json(evaluation / "benchmark.json", estimate)
            if record:
                record.log_file(evaluation / "benchmark.json", "evaluation")
        elif not benchmark_shards:
            files = ("vectors.npy", "track_ids.npy", "window_vectors.npy", "windows.parquet")
            coverage["sha256"] = {name: digest(directory / name) for name in files}
            coverage["artifact_bytes"] = {name: (directory / name).stat().st_size for name in files}
            write_json(evaluation / "coverage.json", coverage)
            if record:
                record.log_file(evaluation / "coverage.json", "evaluation")
        write_json(evaluation / "performance.json", {**result, **metrics})
        if record:
            record.log(metrics)
            record.log_file(evaluation / "performance.json", "evaluation")
        if result["failed"]:
            raise ValueError(
                f"{result['failed']} failed tracks in {directory / 'failed.csv'}; "
                f"retry with --resume {directory.name}."
            )
        return {**result, **metrics, "run_id": directory.name}
