"""Compare the fixed training sample, with a fresh process per candidate for peak RAM/VRAM."""

import json
import multiprocessing
from pathlib import Path

import numpy as np

from music_atlas.tracking import tracking_uri, write_json


def _distribution(values):
    if not len(values):
        return {}
    return dict(
        zip(
            ("min", "p05", "median", "p95", "max"),
            np.percentile(values, [0, 5, 50, 95, 100]).tolist(),
            strict=True,
        )
    )


def pilot_report(directory, rows, labels):
    """Leave-one-out diagnostics within the quick training sample, with group exclusions."""
    from sklearn.metrics import average_precision_score

    ids = np.load(directory / "track_ids.npy")
    vectors = np.load(directory / "vectors.npy")
    table = rows.set_index("track_id").loc[ids]
    roots = list(labels)
    tags = np.array(
        [[genre in values for genre in roots] for values in table.genre_ids], dtype=bool
    ).reshape(len(ids), len(roots))
    artists = table.artist_id.to_numpy()
    allowed = (artists[:, None] != artists) & tags.any(1)[:, None] & tags.any(1)
    if "split_group" in table:
        groups = table.split_group.to_numpy()
        allowed &= groups[:, None] != groups
    cosine = np.clip(vectors @ vectors.T, -1, 1)
    shared = tags.astype(int) @ tags.T > 0
    pairs = np.triu(allowed, 1)
    hits, random_hits, scores, targets, examples = [], [], [], [], []
    wanted = {"Rock", "Blues", "Hip-Hop", "Classical"}
    for i, query in enumerate(ids):
        eligible = np.flatnonzero(allowed[i])
        nearest = eligible[np.argsort(-cosine[i, eligible], kind="stable")[:5]]
        if not len(nearest):
            continue
        hits.append(shared[i, nearest].mean())
        random_hits.append(shared[i, eligible].mean())
        scores.append(tags[nearest].mean(axis=0))
        targets.append(tags[i])
        names = [labels[g] for g, tagged in zip(roots, tags[i], strict=True) if tagged]
        if wanted.intersection(names):
            wanted.difference_update(names)
            examples.append(
                {
                    "track_id": int(query),
                    "genres": names,
                    "neighbours": [
                        {
                            "track_id": int(ids[j]),
                            "cosine": float(cosine[i, j]),
                            "genres": [
                                labels[g] for g, tag in zip(roots, tags[j], strict=True) if tag
                            ],
                        }
                        for j in nearest[:3]
                    ],
                }
            )
    y = np.array(targets, dtype=bool).reshape(-1, len(roots))
    supported = y.any(axis=0) & (~y).any(axis=0)
    ap = (
        average_precision_score(y[:, supported], np.array(scores)[:, supported], average="macro")
        if supported.any()
        else None
    )
    with np.load(directory / "parts" / "audio.npz") as part:
        norms = part["raw_norms"]
    return {
        "raw_window_norm": _distribution(norms[np.isfinite(norms)]),
        "zero_windows": int((norms <= 1e-12).sum()),
        "nan_windows": int(np.isnan(norms).sum()),
        "infinite_windows": int(np.isinf(norms).sum()),
        "pair_cosine": _distribution(cosine[pairs]),
        "shared_genre_cosine": _distribution(cosine[pairs & shared]),
        "no_shared_tag_cosine": _distribution(cosine[pairs & ~shared]),
        "retrieval_precision_at_5": float(np.mean(hits)) if hits else None,
        "random_neighbour_tag_match": float(np.mean(random_hits)) if random_hits else None,
        "genre_vote_macro_ap": ap,
        "constant_score_macro_ap": float(y[:, supported].mean()) if supported.any() else None,
        "queries": len(hits),
        "examples": examples,
    }


def compare(storage, device, batch_size, tracked, windows_per_track=None):
    from music_atlas.embeddings import CHECKPOINTS, run

    arguments = [
        (storage, name, "quick", device, batch_size, tracked, None, 0, windows_per_track)
        for name in CHECKPOINTS
    ]
    with multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        runs = pool.starmap(run, arguments, chunksize=1)
    directories = [Path(result["run_dir"]) for result in runs]
    reports, inputs = {}, []
    for directory in directories:
        meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        inputs.append(
            [
                meta[key]
                for key in (
                    "tracks_sha256",
                    "manifest_sha256",
                    "settings",
                    "aggregation",
                    "hardware",
                    "batch_size",
                )
            ]
        )
        evaluation = directory / "evaluation"
        reports[meta["encoder"]] = {
            "run_id": directory.name,
            "revision": meta["revision"],
            "dimensions": meta["dimensions"],
            "performance": json.loads((evaluation / "performance.json").read_text()),
            "diagnostics": json.loads((evaluation / "pilot.json").read_text()),
        }
    if inputs[0] != inputs[1]:
        raise ValueError("Cannot compare candidates with different inputs or inference settings.")
    for directory in directories:
        output = directory / "evaluation" / "comparison.json"
        write_json(output, reports)
        if tracked:
            from mlflow import MlflowClient

            state = json.loads((directory / "run.json").read_text())
            client = MlflowClient(tracking_uri=tracking_uri(storage["outputs"]))
            client.log_artifact(state["mlflow_run_id"], str(output), "evaluation")
    return {"comparison": str(output), "runs": [d.name for d in directories]}
