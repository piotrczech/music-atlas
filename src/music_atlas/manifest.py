"""Build the shared FMA manifest and measure split quality before model fitting."""

import hashlib
import json
import os
import shutil
import tempfile
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from music_atlas.split_views import digest, write_balanced_eval

SPLITS = ("train", "validation", "test")
FRACTIONS = np.array([0.8, 0.1, 0.1])
SEED = 42
VERSION = 1
MIN_EVAL_GROUPS = 2
STRATEGIES = ("proportional", "rare-support")
RARE_EVAL_TARGET = 250
RARE_EVAL_MAX_FRACTION = 0.20


def indexed(frame, name):
    ids = frame.index
    if not pd.api.types.is_integer_dtype(ids) or not ids.is_unique or (ids <= 0).any():
        raise ValueError(f"{name} requires unique positive integer IDs.")
    return frame.sort_index()


def taxonomy(genres):
    parents = genres["parent"].to_dict()
    ancestors = {}
    for genre in parents:
        chain = []
        node = genre
        while node:
            if node not in parents or node in chain:
                raise ValueError(f"Invalid genre hierarchy at ID {node}.")
            chain.append(node)
            node = parents[node]
        ancestors[genre] = chain
    roots = genres.index[genres["parent"].eq(0)].tolist()
    if not roots:
        raise ValueError("The genre hierarchy has no top-level labels.")
    return roots, ancestors


def components(*keys):
    """Return the smallest row position in each connected component."""
    parent = list(range(len(keys[0])))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in keys:
        for positions in key.groupby(key, sort=False).indices.values():
            if len(positions) < 2:
                continue
            root = find(positions[0])
            for position in positions[1:]:
                other = find(position)
                low, high = sorted((root, other))
                parent[high] = low
                root = low
    return np.fromiter((find(i) for i in range(len(parent))), dtype=np.int64)


def assign_groups(sizes, positives, strategy="proportional"):
    """Seed with a size-matched random split, then improve per-label balance."""
    order = np.random.default_rng(SEED).permutation(len(sizes))
    track_targets = FRACTIONS * sizes.sum()
    tracks = np.zeros(len(SPLITS), dtype=np.int64)
    random_split = np.empty(len(sizes), dtype=np.int8)
    for step, group in enumerate(order):
        deficit = track_targets - tracks
        empty = tracks == 0
        if len(order) - step == empty.sum():
            deficit[~empty] = -np.inf
        chosen = int(np.argmax(deficit))
        random_split[group] = chosen
        tracks[chosen] += sizes[group]

    active = positives > 0
    n_labels = positives.shape[1]
    vectors = np.column_stack((sizes, positives, active))
    targets = FRACTIONS[:, None] * vectors.sum(axis=0)
    if strategy == "rare-support":
        totals = positives.sum(axis=0)
        held_out = np.clip(RARE_EVAL_TARGET / np.maximum(totals, 1),
                           FRACTIONS[1], RARE_EVAL_MAX_FRACTION)
        label_fractions = np.stack((1 - 2 * held_out, held_out, held_out))
        targets[:, 1 : 1 + n_labels] = label_fractions * totals
        targets[:, 1 + n_labels :] = label_fractions * active.sum(axis=0)
    # Compare relative deviations within each split so validation and test count too.
    importance = np.r_[n_labels, np.ones(n_labels), np.full(n_labels, 0.2)]
    weights = importance / np.maximum(targets, 1) ** 2
    canonical = random_split.copy()
    counts = np.zeros_like(targets)
    np.add.at(counts, canonical, vectors)
    groups = np.bincount(canonical, minlength=len(SPLITS))
    order = order[np.argsort(-sizes[order], kind="stable")]
    can_cover = active.sum(axis=0) >= len(SPLITS)
    for _ in range(6):
        moved = False
        for group in order:
            previous = canonical[group]
            if groups[previous] == 1:
                continue
            vector = vectors[group]
            residual = counts - targets
            add = ((2 * residual * vector + vector**2) * weights).sum(axis=1)
            remove = ((-2 * residual[previous] * vector + vector**2) * weights[previous]).sum()
            coverage = counts[:, 1 + n_labels :]
            required = active[group] & can_cover
            gain = ((coverage == 0) & required).sum(axis=1)
            loss = ((coverage[previous] == 1) & required).sum()
            cost = add + remove + 0.5 * (loss - gain)
            cost[previous] = 0
            chosen = int(np.argmin(cost))
            if cost[chosen] < -1e-12:
                counts[previous] -= vector
                counts[chosen] += vector
                groups[previous] -= 1
                groups[chosen] += 1
                canonical[group] = chosen
                moved = True
        if not moved:
            break
    return canonical, random_split


def overlaps(split, **keys):
    return {name: int(split.groupby(key).nunique().gt(1).sum()) for name, key in keys.items()}


def support_report(manifest, labels, names, splits):
    rows = []
    annotated = labels.any(axis=1)
    for scheme, assignment in splits.items():
        for split in SPLITS:
            included = assignment.eq(split).fillna(False).to_numpy(dtype=bool)
            n_annotated = int((included & annotated).sum())
            for column, (genre_id, name) in enumerate(names.items()):
                tracks = manifest.loc[included & labels[:, column]]
                rows.append({
                    "scheme": scheme, "split": split, "genre_id": genre_id, "genre": name,
                    "tracks": len(tracks), "artists": tracks["artist_id"].nunique(),
                    "groups": tracks["split_group"].nunique(), "annotated_tracks": n_annotated,
                    "prevalence": len(tracks) / n_annotated if n_annotated else None,
                })
    return pd.DataFrame(rows)


def baseline_report(labels, assignment, scored):
    annotated = labels.any(axis=1)
    training = assignment.eq("train").fillna(False).to_numpy(dtype=bool) & annotated
    if not training.any() or not scored.any():
        return {}
    prevalence = labels[training][:, scored].mean(axis=0)
    majority = int(prevalence.argmax())
    report = {}
    for split in ("validation", "test"):
        included = assignment.eq(split).fillna(False).to_numpy(dtype=bool) & annotated
        truth = labels[included][:, scored]
        positive = truth.sum(axis=0)
        negative = len(truth) - positive
        scores = {}
        for name in ("all_negative", "most_frequent_genre"):
            prediction = np.zeros(len(prevalence), dtype=bool)
            if name == "most_frequent_genre":
                prediction[majority] = True
            tp = positive * prediction
            fp = negative * prediction
            fn = positive - tp
            recall = tp / positive
            specificity = (negative - fp) / negative
            scores[name] = {
                "label_accuracy": float((tp + negative - fp).sum() / truth.size),
                "macro_recall": float(recall.mean()),
                "macro_balanced_accuracy": float(((recall + specificity) / 2).mean()),
                "macro_f1": float((2 * tp / (2 * tp + fp + fn)).mean()),
                "micro_f1": float(2 * tp.sum() / (2 * tp + fp + fn).sum()),
            }
        # A constant score per genre ranks no examples within that genre.
        scores["constant_score_map"] = float((positive / len(truth)).mean())
        report[split] = scores
    return report


def prepare(inventory, metadata, genres, audit, strategy="proportional"):
    inventory = indexed(inventory, "Inventory")
    metadata = indexed(metadata, "Track metadata")
    genres = indexed(genres, "Genre metadata")
    roots, ancestors = taxonomy(genres)
    columns = {genre_id: i for i, genre_id in enumerate(roots)}
    present = inventory["status"].eq("present")
    if not inventory["status"].isin(["present", "missing"]).all():
        raise ValueError("Unexpected inventory status.")
    for field in ("shard", "audio_path"):
        values = inventory[field].astype("string").str.strip()
        unsafe = values.str.contains(r"^/|(?:^|/)\.\.?(?:/|$)|[\\:]", regex=True, na=True)
        if unsafe.any() or values.eq("").any():
            raise ValueError(f"Inventory requires safe relative paths in {field}.")
    if inventory.duplicated(["shard", "audio_path"]).any():
        raise ValueError("More than one track refers to the same inventory member.")
    if "size_bytes" in inventory and not inventory.loc[present, "size_bytes"].gt(0).all():
        raise ValueError("Present inventory entries require a positive file size.")
    if not inventory.loc[present, "sha1"].str.fullmatch(r"[0-9a-fA-F]{40}", na=False).all():
        raise ValueError("Every present track needs a verified SHA-1 in the inventory.")
    tracks = metadata.reindex(inventory.index)
    manifest = inventory[["shard", "audio_path"]].copy()
    for field in ("artist_id", "album_id"):
        manifest[field] = tracks[field].where(tracks[field].gt(0)).astype("Int64")

    labels = np.zeros((len(tracks), len(roots)), dtype=bool)
    genre_ids = []
    for i, raw in enumerate(tracks["genres"].fillna("[]")):
        direct = literal_eval(raw)
        if not isinstance(direct, list) or any(genre not in ancestors for genre in direct):
            raise ValueError(f"Unknown genre annotation for track {tracks.index[i]}.")
        expanded = sorted({parent for genre in direct for parent in ancestors[genre]})
        genre_ids.append(expanded)
        labels[i, [columns[genre] for genre in expanded if genre in columns]] = True
    manifest["genre_ids"] = genre_ids

    valid = present.copy()
    errors = pd.Series(pd.NA, index=inventory.index, dtype="string")
    audit_groups = errors.copy()
    duration = tracks["duration_seconds"]
    # E1-03 integration point: replace inventory eligibility with decoded-audio decisions.
    if audit is not None:
        audit = indexed(audit, "Audio audit")
        if not pd.api.types.is_bool_dtype(audit["valid"]) or audit["valid"].isna().any():
            raise ValueError("Audio audit validity must contain non-null booleans.")
        if len(inventory.index[present].difference(audit.index)):
            raise ValueError("Audio audit is missing present track IDs.")
        if len(audit.index.difference(inventory.index)):
            raise ValueError("Audio audit contains track IDs outside the inventory.")
        audit = audit.reindex(inventory.index)
        valid &= audit["valid"].eq(True).fillna(False)
        errors = audit["error"].astype("string").str.strip().replace("", pd.NA)
        if errors[present & ~valid].isna().any():
            raise ValueError("Every rejected present track needs an audit error.")
        audit_groups = audit["duplicate_group"].astype("string").str.strip().replace("", pd.NA)
        duration = pd.to_numeric(audit["duration_seconds"], errors="raise")
    usable_duration = duration.gt(0) & np.isfinite(duration)
    if audit is not None and not usable_duration[valid].fillna(False).all():
        raise ValueError("Every valid track needs a positive audited duration.")
    manifest["duration_seconds"] = duration.where(usable_duration)
    manifest["valid"] = valid
    manifest["error"] = errors.mask(valid).mask(~present, "inventory_missing")

    # Invalid present rows can still connect duplicate recordings and their artists.
    dup = components(inventory["sha1"].str.lower().where(present), audit_groups.where(present))
    ids = inventory.index.to_numpy()
    duplicate_ids = pd.Series(ids[dup], index=inventory.index, dtype="Int64")
    manifest["duplicate_group"] = duplicate_ids.where(np.bincount(dup)[dup] > 1)
    artist = manifest["artist_id"].where(present)
    group = components(artist, manifest["duplicate_group"])
    manifest["split_group"] = ids[group]
    eligible = valid & artist.notna()
    group_ids, inverse = np.unique(group[eligible], return_inverse=True)
    if len(group_ids) < len(SPLITS):
        raise ValueError("At least three independent components are needed to form the splits.")
    sizes = np.bincount(inverse)
    positives = np.zeros((len(sizes), len(roots)), dtype=np.int64)
    np.add.at(positives, inverse, labels[eligible])
    assignments = {}
    allocations = assign_groups(sizes, positives, strategy)
    for name, allocation in zip(("canonical", "group_random"), allocations, strict=True):
        if len(set(allocation)) != len(SPLITS):
            raise ValueError(f"{name} could not populate every split with independent groups.")
        split = pd.Series(pd.NA, index=inventory.index, dtype="string")
        split.loc[eligible] = np.array(SPLITS)[allocation[inverse]]
        assignments[name] = split
    manifest["split"] = assignments["canonical"]
    published = tracks["published_split"].replace("training", "train")
    assignments["published"] = published.where(eligible & published.isin(SPLITS))
    keys = {"artists": artist, "duplicate_groups": manifest["duplicate_group"],
            "components": ids[group]}
    leakage = {name: overlaps(split, **keys) for name, split in assignments.items()}
    if any(leakage["canonical"].values()):
        raise ValueError("The canonical split contains an overlapping group.")

    names = genres.loc[roots, "title"]
    support = support_report(manifest, labels, names, assignments)
    canonical = support[support["scheme"].eq("canonical")]
    has_both = canonical["tracks"].gt(0) & canonical["tracks"].lt(canonical["annotated_tracks"])
    group_support = canonical["split"].eq("train") | canonical["groups"].ge(MIN_EVAL_GROUPS)
    scored = (has_both & group_support).groupby(canonical["genre_id"]).all()
    scored = scored.reindex(roots).to_numpy()
    mapping = [
        {"column": i, "genre_id": genre, "name": names.loc[genre], "scored": bool(scored[i])}
        for i, genre in enumerate(roots)
    ]
    labelled = labels[eligible & labels.any(axis=1)]
    reference = labelled.mean(axis=0) if len(labelled) else np.zeros(len(roots))
    support["prevalence_shift"] = support["prevalence"] - support["genre_id"].map(dict(zip(
        roots, reference, strict=True,
    )))
    summary = {
        "tracks": len(manifest), "present": int(present.sum()), "eligible": int(eligible.sum()),
        "invalid": int((~valid).sum()) if audit is not None else None,
        "missing_artist": int((valid & artist.isna()).sum()),
        "unlabelled": int((eligible & ~labels.any(axis=1)).sum()),
        "multilabel": int((eligible & (labels.sum(axis=1) > 1)).sum()),
        "splits": {name: {part: int(s.eq(part).sum()) for part in SPLITS}
                   for name, s in assignments.items()},
        "overlap": leakage,
        "baselines": baseline_report(labels, assignments["canonical"], scored),
    }
    return manifest.reset_index(), mapping, support, summary


def publish_version(destination, manifest, mapping, support, summary):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fma-", dir=destination.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        table = pa.Table.from_pandas(manifest, preserve_index=False)
        column = table.schema.get_field_index("genre_ids")
        tag_array = pa.array(manifest["genre_ids"], pa.list_(pa.int64()))
        table = table.set_column(column, "genre_ids", tag_array)
        table = table.replace_schema_metadata({
            **(table.schema.metadata or {}), b"dataset_version": destination.name.encode(),
            b"audio_audited": str(summary["audio_audited"]).lower().encode(),
            b"duration_source": summary["duration_source"].encode(),
        })
        pq.write_table(table, stage / "tracks.parquet")
        for split in SPLITS:
            rows = manifest["split"].eq(split).fillna(False)
            pq.write_table(table.filter(pa.array(rows)), stage / f"{split}.parquet")
        summary["manifest_sha256"] = digest(stage / "tracks.parquet")
        (stage / "labels.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        support.to_csv(stage / "class_support.csv", index=False)
        quota = write_balanced_eval(
            manifest, mapping, stage, destination.name, summary["manifest_sha256"], SEED,
        )

        try:
            stage.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            for expected in sorted(stage.iterdir()):
                stored = destination / expected.name
                if not stored.is_file() or digest(stored) != digest(expected):
                    raise ValueError(
                        f"Missing or changed file in existing version: {stored}"
                    ) from None
    return quota


def build(data_dir: Path, audit_path: Path | None = None, strategy: str = "proportional") -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown split strategy: {strategy}.")
    canonical = data_dir / "manifests/tracks.parquet"
    if audit_path is not None and audit_path.resolve() == canonical.resolve():
        raise ValueError("The E1-03 audit must be separate from the canonical manifest.")
    audited = audit_path is not None
    metadata_dir = data_dir / "fma/fma_metadata"
    sources = {
        "inventory": data_dir / "manifests/fma_full_inventory.csv",
        "tracks": metadata_dir / "tracks.csv", "genres": metadata_dir / "genres.csv",
    }
    if audit_path is not None:
        sources["audit"] = audit_path
    hashes = {name: digest(path) for name, path in sources.items()}
    recipe = {
        "schema": VERSION, "seed": SEED,
        "fractions": dict(zip(SPLITS, FRACTIONS.tolist(), strict=True)),
        "min_eval_groups": MIN_EVAL_GROUPS,
        "allocator": "group-refinement-v1",
        # Preserve the identities and summaries of the published v1 datasets.
        "inputs": hashes, "preview": False, "audio_audited": audited,
        "duration_source": "decoded_audio_audit" if audited else "published_fma_metadata",
    }
    prefix = "fma" if audited else "pre-audit"
    if strategy == "rare-support":
        prefix = f"{prefix}-rare-support"
        recipe.update(strategy=strategy, rare_eval_target=RARE_EVAL_TARGET,
                      rare_eval_max_fraction=RARE_EVAL_MAX_FRACTION)
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    short_version = f"{prefix}-v{VERSION}"
    version = f"{short_version}-{identity[:16]}"

    inventory = pd.read_csv(sources["inventory"], index_col="track_id", dtype={"sha1": "string"})
    genres = pd.read_csv(sources["genres"], index_col="genre_id")
    fields = {
        ("artist", "id"): "artist_id", ("album", "id"): "album_id", ("track", "genres"): "genres",
        ("track", "duration"): "duration_seconds", ("set", "split"): "published_split",
    }
    metadata = pd.read_csv(sources["tracks"], index_col=0, header=[0, 1], low_memory=False)
    metadata = metadata[list(fields)]
    metadata.columns = list(fields.values())
    audit = None
    if audit_path is not None:
        audit = pd.read_parquet(audit_path, columns=[
            "track_id", "valid", "error", "duplicate_group", "duration_seconds",
        ]).set_index("track_id")
    manifest, mapping, support, summary = prepare(inventory, metadata, genres, audit, strategy)
    if hashes != {name: digest(path) for name, path in sources.items()}:
        raise ValueError("Inputs changed during manifest preparation. Retry with stable inputs.")

    destination = data_dir / "ready" / version
    summary.update(recipe, version=version, version_sha256=identity)
    balanced_quota = publish_version(destination, manifest, mapping, support, summary)
    if audited and strategy == "proportional":
        with tempfile.TemporaryDirectory(prefix=".fma-", dir=canonical.parent) as temp:
            copy = Path(temp) / canonical.name
            shutil.copyfile(destination / "tracks.parquet", copy)
            os.replace(copy, canonical)

    # Keep an existing short name pinned to its original dataset.
    alias = destination.with_name(short_version)
    if os.name == "posix":
        try:
            alias.symlink_to(destination.name, target_is_directory=True)
        except FileExistsError:
            pass
    has_alias = alias.is_symlink() and alias.readlink() == Path(version)
    return {
        **summary, "directory": str(destination), "balanced_eval_quota": balanced_quota,
        "short_version": short_version if has_alias else None,
    }
