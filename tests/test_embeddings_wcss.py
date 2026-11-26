"""Opt-in WCSS recovery check on four real quick-sample tracks in two original FMA shards."""

import json
import multiprocessing
import os
import signal

import numpy as np
import pandas as pd
import pytest

from music_atlas.embeddings import (
    CHECKPOINTS,
    DIMENSIONS,
    RATE,
    load_encoder,
    repeatpad,
    verify_vectors,
)
from music_atlas.representation import UNIT_MEAN, extract
from music_atlas.settings import paths, profile
from music_atlas.shards import read_tracks
from music_atlas.split_views import digest
from music_atlas.tracking import track, tracking_uri, write_json


def attempt(encoder, rows, marker, *, interrupt=False, resume=None):
    import torch

    storage = paths()
    alias = profile("quick")["dataset"]["split"]
    cfg = profile("full")
    settings = {k: cfg[k] for k in ("window_seconds", "windows_per_track")} | {"sample_rate": RATE}
    checkpoint, revision = CHECKPOINTS[encoder]
    params = {
        **settings,
        "encoder": encoder,
        "checkpoint": checkpoint,
        "revision": revision,
        "batch_size": 6,
        "dimensions": DIMENSIONS,
        "aggregation": UNIT_MEAN,
        "device": "cpu",
    }
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    with track(
        "embedding-recovery",
        "quick",
        storage,
        alias,
        encoder,
        description="Hard-kill recovery on four training examples from two FMA shards.",
        overrides=params,
        resume=resume,
    ) as record:
        marker.write_text(record.run_id, encoding="utf-8")
        encode, timings = load_encoder(checkpoint, revision, torch.device("cpu"))
        record.log(timings)

        def encode_batch(samples):
            if interrupt and any((record.directory / "parts").glob("*.npz")):
                os.kill(os.getpid(), signal.SIGKILL)
            return encode(samples)

        subsets = {"quick-sample": rows}
        meta = {
            **params,
            "run_id": record.run_id,
            "dataset_version": record.dataset["version"],
            "manifest_sha256": record.dataset["manifest_sha256"],
        }
        result = extract(
            storage["data"],
            record.directory,
            encode_batch,
            RATE,
            meta,
            record=record,
            inputs=(rows, record.dataset, {}),
            windows_per_track=cfg["windows_per_track"],
            prepare=repeatpad,
            batch_size=6,
            aggregation=UNIT_MEAN,
        )
        assert result["failed"] == 0
        verify_vectors(record.directory, subsets)
        return record.directory


@pytest.mark.skipif(
    os.environ.get("ATLAS_VERIFY_CLAP") != "1",
    reason="Requires the WCSS FMA shards and both cached checkpoints",
)
@pytest.mark.parametrize("encoder", CHECKPOINTS)
def test_recovery_on_fma(tmp_path, encoder):
    from mlflow import MlflowClient

    storage = paths()
    alias = profile("quick")["dataset"]["split"]
    sample, _, _ = read_tracks(storage["data"], "quick")
    train = pd.read_parquet(storage["data"] / "ready" / alias / "train.parquet")
    candidates = train[train.track_id.isin(sample.track_id)].sort_values("track_id")
    counts = candidates.groupby("shard").size()
    shards = counts[counts >= 2].index[:2]
    rows = candidates[candidates.shard.isin(shards)].groupby("shard").head(2)
    rows = rows.assign(tar=[str(storage["data"] / shard) for shard in rows.shard])
    assert len(rows) == 4

    marker = tmp_path / "attempt-id"
    clean = attempt(encoder, rows, marker)
    proc = multiprocessing.get_context("spawn").Process(
        target=attempt, args=(encoder, rows, marker), kwargs={"interrupt": True}
    )
    proc.start()
    proc.join(300)
    if proc.is_alive():
        proc.kill()
        proc.join()
        pytest.fail("Timed out before the second shard")
    assert proc.exitcode == -signal.SIGKILL
    run_id = marker.read_text(encoding="utf-8")
    interrupted = storage["artifacts"] / encoder / run_id
    (kept,) = (interrupted / "parts").glob("*.npz")
    before = (kept.stat().st_mtime_ns, digest(kept))
    first = json.loads((interrupted / "run.json").read_text())
    resumed = attempt(encoder, rows, marker, resume=run_id)
    assert before == (kept.stat().st_mtime_ns, digest(kept))
    np.testing.assert_array_equal(
        np.load(clean / "track_ids.npy"), np.load(resumed / "track_ids.npy")
    )
    vectors, reference = (np.load(d / "vectors.npy") for d in (resumed, clean))
    np.testing.assert_allclose(vectors, reference, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(
        np.load(resumed / "window_vectors.npy"),
        np.load(clean / "window_vectors.npy"),
        atol=1e-6,
        rtol=1e-5,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(clean / "windows.parquet"), pd.read_parquet(resumed / "windows.parquet")
    )
    state = json.loads((resumed / "run.json").read_text())
    assert state["tags"]["atlas.resumed_from"] == first["mlflow_run_id"]
    client = MlflowClient(tracking_uri=tracking_uri(storage["outputs"]))
    assert client.get_run(first["mlflow_run_id"]).info.status == "KILLED"
    proof = resumed / "evaluation" / "recovery.json"
    proof.parent.mkdir(exist_ok=True)
    write_json(
        proof,
        {
            "reference_run": clean.name,
            "resumed_run": run_id,
            "device": "cpu",
            "tracks": len(rows),
            "shards": list(shards),
            "retained_part": kept.name,
            "retained_sha256": before[1],
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "signal": "SIGKILL",
            "max_abs_error": float(np.max(abs(vectors - reference))),
        },
    )
    client.log_artifact(state["mlflow_run_id"], str(proof), "evaluation")
    print(f"Recovery proof: {proof}", flush=True)
