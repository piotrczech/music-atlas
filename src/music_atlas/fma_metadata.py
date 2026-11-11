"""Reproducible metadata summaries for the pre-audit or usable FMA-full catalogue."""

import hashlib
import json
from ast import literal_eval
from collections import Counter
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pandas.api.types import is_bool_dtype

TRACK_FIELDS = {
    ("artist", "id"): "artist_id",
    ("album", "id"): "album_id",
    ("track", "genres"): "genres",
    ("track", "genre_top"): "genre_top",
    ("track", "duration"): "duration_seconds",
    ("set", "split"): "published_split",
}


def _id_index(values, source):
    ids = pd.Index(values, dtype="Int64")
    if ids.hasnans or not ids.is_unique or (ids <= 0).any():
        raise ValueError(f"{source} must have unique positive IDs.")
    return ids


def _catalogue(inventory_path, manifest_path):
    inventory = pd.read_csv(inventory_path, usecols=["track_id", "status"])
    inventory.index = _id_index(inventory.track_id, inventory_path.name)
    if not inventory.status.isin(["present", "missing"]).all():
        raise ValueError("Inventory status must be present or missing.")
    present = inventory.index[inventory.status == "present"]
    counts = {"inventory_tracks": len(inventory), "present_tracks": len(present)}
    if manifest_path is None:
        return present, counts

    columns = ["track_id", "valid"]
    match manifest_path.suffix.lower():
        case ".parquet":
            manifest = pd.read_parquet(manifest_path, columns=columns)
        case ".csv":
            manifest = pd.read_csv(manifest_path, usecols=columns, dtype={"valid": "boolean"})
        case _:
            raise ValueError("The audited manifest must be CSV or Parquet.")
    manifest.index = _id_index(manifest.track_id, manifest_path.name)
    if not is_bool_dtype(manifest.valid) or manifest.valid.isna().any():
        raise ValueError("Manifest valid must contain a Boolean for every row.")
    usable = manifest.index[manifest.valid]
    unexpected = usable.difference(present)
    if len(unexpected):
        raise ValueError(f"{len(unexpected)} valid manifest IDs are absent from present inventory.")
    counts["present_rejected"] = len(present.intersection(manifest.index[~manifest.valid]))
    counts["present_unlisted_in_manifest"] = len(present.difference(manifest.index))
    return usable, counts


def _hierarchy(genres):
    paths = {}
    parents = genres.parent.to_dict()
    for genre_id in genres.index:
        chain = [int(genre_id)]
        while (parent := parents[chain[-1]]) != 0:
            if parent not in parents or parent in chain:
                raise ValueError(f"Invalid genre ancestry for genre {genre_id}.")
            chain.append(int(parent))
        paths[int(genre_id)] = tuple(reversed(chain))
    return paths


def _parse_labels(value):
    if pd.isna(value):
        return ()
    try:
        labels = literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid genre annotation: {value!r}.") from error
    if not isinstance(labels, list) or any(type(label) is not int for label in labels):
        raise ValueError(f"Expected a list of genre IDs, got {value!r}.")
    return tuple(sorted(set(labels)))


def _genre_counts(tracks, genres, ancestry):
    direct = tracks.genres.map(_parse_labels)
    unknown = {label for labels in direct for label in labels}.difference(genres.index)
    if unknown:
        raise ValueError(f"Track metadata uses unknown genre IDs: {sorted(unknown)}.")
    inclusive = direct.map(
        lambda ids: sorted({parent for label in ids for parent in ancestry[label]})
    )
    genre_table = genres[["title", "parent"]].rename(columns={"parent": "parent_id"}).copy()
    genre_table["root_id"] = [ancestry[i][0] for i in genres.index]
    genre_table["depth"] = [len(ancestry[i]) - 1 for i in genres.index]
    genre_table["path"] = [" / ".join(genres.loc[list(ancestry[i]), "title"]) for i in genres.index]
    for prefix, labels in [("direct", direct), ("inclusive", inclusive)]:
        tagged = (
            tracks[["artist_id"]]
            .assign(genre_id=labels)
            .explode("genre_id")
            .dropna(subset=["genre_id"])
        )
        by_genre = tagged.groupby("genre_id")
        by_artist = tagged.groupby(["genre_id", "artist_id"]).size()
        counts = by_genre.size().reindex(genres.index, fill_value=0)
        largest = by_artist.groupby(level=0).max().reindex(genres.index, fill_value=0)
        genre_table[f"{prefix}_tracks"] = counts
        genre_table[f"{prefix}_artists"] = by_genre.artist_id.nunique().reindex(
            genres.index, fill_value=0
        )
        genre_table[f"{prefix}_top_artist_share"] = largest / counts.where(counts > 0)
    return direct, inclusive, genre_table


def _cooccurrence(direct, genres):
    pairs = Counter(pair for labels in direct for pair in combinations(labels, 2))
    counts = genres.direct_tracks
    return pd.DataFrame(
        [
            (a, b, count, count / (counts[a] + counts[b] - count))
            for (a, b), count in sorted(pairs.items())
        ],
        columns=["genre_id_a", "genre_id_b", "tracks", "jaccard"],
    )


def _concentration(tracks, field):
    return (
        tracks.groupby(field)
        .size()
        .rename("tracks")
        .sort_values(ascending=False, kind="stable")
        .to_frame()
        .assign(catalogue_share=lambda table: table.tracks / len(tracks))
    )


def _figures(output, genres, pairs, artists, albums, durations, coverage):
    def save(figure, name):
        FigureCanvasAgg(figure).print_figure(output / f"{name}.png", dpi=150)

    figure = Figure(figsize=(12.5, 6.7), layout="constrained")
    panels = figure.subplots(1, 2)
    roots = genres[genres.depth == 0].nlargest(18, "inclusive_tracks")
    children = genres[genres.depth > 0].nlargest(12, "inclusive_tracks")
    for ax, table, labels, title in [
        (panels[0], roots, roots.title, "Top-level genres"),
        (panels[1], children, children.path, "Most frequent subgenres"),
    ]:
        positions = range(len(table))
        ax.barh(positions, table.inclusive_tracks, color="#b3d2de", label="Including descendants")
        ax.barh(positions, table.direct_tracks, color="#246a81", label="Direct annotation")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set(xlabel="Unique tracks", title=title)
    panels[0].legend(loc="lower right", fontsize=8)
    save(figure, "01-genres")

    figure = Figure(figsize=(8.7, 7.6), layout="constrained")
    ax = figure.subplots()
    common = genres[genres.direct_tracks > 0].nlargest(12, "direct_tracks")
    if not common.empty:
        size = len(common)
        locations = {genre_id: i for i, genre_id in enumerate(common.index)}
        matrix = np.zeros((size, size))
        for a, b, _, jaccard in pairs.itertuples(index=False, name=None):
            if a in locations and b in locations:
                matrix[locations[a], locations[b]] = matrix[locations[b], locations[a]] = jaccard
        upper = max(0.05, np.ceil(matrix.max() * 20) / 20)
        cmap = colormaps["Blues"].with_extremes(bad="#e5e7eb")
        np.fill_diagonal(matrix, np.nan)
        plotted = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=upper)
        ax.set_xticks(range(size), common.title, rotation=50, ha="right")
        ax.set_yticks(range(size), common.title)
        figure.colorbar(plotted, ax=ax, label="Shared / union of tracks")
    else:
        ax.text(
            0.5, 0.5, "No direct genre labels", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_axis_off()
    ax.set_title("Co-occurrence of direct genre labels")
    save(figure, "02-cooccurrence")

    figure = Figure(figsize=(8.5, 4.7), layout="constrained")
    ax = figure.subplots()
    for label, table, color in [("Artists", artists, "#246a81"), ("Albums", albums, "#c1663d")]:
        if not table.empty:
            rank = np.arange(1, len(table) + 1)
            ax.plot(rank, table.catalogue_share.cumsum() * 100, label=label, color=color)
    if not (artists.empty and albums.empty):
        ax.legend()
    ax.set(
        xscale="log",
        ylim=(0, 100),
        xlabel="Entities ranked by number of tracks",
        ylabel="Cumulative share of catalogue (%)",
        title="Artist and album concentration",
    )
    save(figure, "03-concentration")

    figure = Figure(figsize=(8.5, 4.7), layout="constrained")
    ax = figure.subplots()
    if len(durations):
        lower, upper = durations.min(), durations.max()
        bins = np.geomspace(lower * 0.99, upper * 1.01, 41)
        ax.hist(durations, bins=bins, color="#246a81")
        ax.axvline(durations.median(), color="#c1663d", label=f"Median {durations.median():.0f} s")
        ax.legend()
    ax.set(
        xscale="log",
        xlabel="Published duration (seconds, log scale)",
        ylabel="Tracks per bin",
        title="Track duration distribution",
    )
    save(figure, "04-durations")

    figure = Figure(figsize=(9, 4.5), layout="constrained")
    ax = figure.subplots()
    percentages = coverage.share * 100
    ax.barh(coverage.metric, percentages, color="#246a81")
    ax.invert_yaxis()
    ax.set(xlim=(0, 114), xlabel="Share of analysed catalogue (%)", title="Annotation coverage")
    ax.set_xticks(range(0, 101, 20))
    for i, (value, count) in enumerate(zip(percentages, coverage.tracks, strict=True)):
        ax.text(value + 1, i, f"{value:.2f}% ({count:,})", va="center", fontsize=9)
    save(figure, "05-coverage")


def _interpretation(summary, genres):
    coverage = summary["coverage"]
    concentrations = summary["concentration"]
    published = summary["published_split"]
    observed = genres[genres.direct_tracks > 0]
    sparse = observed[observed.direct_artists < 3]
    examples = sparse.sort_values(["direct_artists", "direct_tracks"]).head(4)
    rare = ", ".join(
        f"{row.title} ({row.direct_tracks:,} / {row.direct_artists:,})"
        for row in examples.itertuples()
    )
    roots = genres[genres.depth == 0]
    gap = roots.inclusive_tracks - roots.direct_tracks
    hierarchy = ""
    if not roots.empty:
        root = roots.loc[gap.idxmax()]
        hierarchy = (
            f"{root.title} has {root.direct_tracks:,} direct annotations and "
            f"{root.inclusive_tracks:,} tracks including descendants. "
        )
    concentrated = observed[observed.direct_tracks >= 20].nlargest(1, "direct_top_artist_share")
    concentration_example = ""
    if not concentrated.empty:
        label = concentrated.iloc[0]
        concentration_example = (
            f"In {label.title}, one artist accounts for {label.direct_top_artist_share:.1%} "
            f"of {label.direct_tracks:,} directly labelled tracks. "
        )
    return (
        "# FMA metadata exploration\n\n"
        f"Analysed **{summary['analysed_tracks']:,}** tracks from the "
        f"**{summary['catalogue']}** catalogue. Durations are published metadata.\n\n"
        f"- **Label set.** {hierarchy}{len(sparse):,} directly observed genres have fewer "
        "than three known artists, limiting representation across three artist-disjoint splits."
        + (f" Examples in tracks / known artists are {rare}." if rare else "")
        + " Check class support in tracks and independent split groups after the audit. "
        "Consider mapping undersupported children to a supported ancestor or keeping them out "
        "of the scored label set.\n"
        f"- **Missing annotations.** {coverage['No direct genre']:,} tracks have no direct "
        f"genre and {coverage['Multiple top-level branches']:,} span multiple top-level "
        f"branches. Missing metadata affects {coverage['Missing artist ID']:,} artist IDs, "
        f"{coverage['Missing album ID']:,} album IDs and "
        f"{coverage['Missing or invalid duration']:,} durations. Treat missing genre annotations "
        "as unknown and keep records without usable labels out of supervised evaluation. "
        "Use multilabel targets for supported top-level branches to retain cross-branch tracks.\n"
        f"- **Split strategy.** The ten largest artists account for "
        f"{concentrations['artist']['top_10_share']:.1%} of all analysed tracks and the "
        f"ten largest albums for {concentrations['album']['top_10_share']:.1%}. "
        f"The published split has {published['artists_in_multiple_splits']:,} known artists and "
        f"{published['albums_in_multiple_splits']:,} known albums across multiple partitions. "
        f"{concentration_example}Build one artist-disjoint split, link artists joined by "
        "confirmed duplicates, "
        "resolve unknown artists explicitly and inspect album overlap. Check per-class support "
        "and duration distributions in each partition before freezing the split.\n"
    )


def explore(data: Path, output: Path, manifest: Path | None = None) -> dict:
    """Write a compact report from the inventory, optionally restricted by audit validity."""
    metadata = data / "fma/fma_metadata"
    sources = {
        "inventory": data / "manifests/fma_full_inventory.csv",
        "tracks": metadata / "tracks.csv",
        "genres": metadata / "genres.csv",
    }
    if manifest is not None:
        sources["manifest"] = manifest
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    ids, inventory = _catalogue(sources["inventory"], manifest)
    if ids.empty:
        raise ValueError("The selected catalogue has no tracks.")
    raw = pd.read_csv(sources["tracks"], header=[0, 1], index_col=0, low_memory=False)
    raw.index = _id_index(raw.index, "tracks.csv")
    without_metadata = len(ids.difference(raw.index))
    tracks = raw.loc[:, list(TRACK_FIELDS)].reindex(ids)
    del raw
    tracks.columns = list(TRACK_FIELDS.values())
    for field in ["artist_id", "album_id"]:
        tracks[field] = tracks[field].astype("Int64").where(lambda values: values > 0)
    genres = pd.read_csv(sources["genres"], index_col="genre_id", dtype={"parent": "int64"})
    genres.index = _id_index(genres.index, "genres.csv")
    ancestry = _hierarchy(genres)
    direct, inclusive, genre_table = _genre_counts(tracks, genres, ancestry)
    pairs = _cooccurrence(direct, genre_table)
    artists = _concentration(tracks, "artist_id")
    albums = _concentration(tracks, "album_id")
    duration = pd.to_numeric(tracks.duration_seconds, errors="coerce")
    durations = duration[duration.gt(0) & np.isfinite(duration)]
    n_labels = direct.map(len)
    roots = {i for i, chain in ancestry.items() if len(chain) == 1}
    n_branches = inclusive.map(lambda ids: len(roots.intersection(ids)))
    coverage = {
        "No direct genre": int(n_labels.eq(0).sum()),
        "One direct genre": int(n_labels.eq(1).sum()),
        "Multiple direct genres": int(n_labels.gt(1).sum()),
        "Multiple top-level branches": int(n_branches.gt(1).sum()),
        "Missing published genre_top": int(tracks.genre_top.isna().sum()),
        "Missing artist ID": int(tracks.artist_id.isna().sum()),
        "Missing album ID": int(tracks.album_id.isna().sum()),
        "Missing or invalid duration": len(tracks) - len(durations),
    }
    summary = {
        "catalogue": "audited" if manifest else "present inventory (pre-audit)",
        "analysed_tracks": len(tracks),
        **inventory,
        "selected_without_metadata": without_metadata,
        "coverage": coverage,
        "duration_seconds": {
            name: float(durations.quantile(q)) if len(durations) else None
            for name, q in [("min", 0), ("p05", 0.05), ("median", 0.5), ("p95", 0.95), ("max", 1)]
        },
        "concentration": {
            key: {
                "entities": len(table),
                "top_10_share": float(table.catalogue_share.head(10).sum()),
            }
            for key, table in [("artist", artists), ("album", albums)]
        },
        "published_split": {
            "tracks": {str(k): int(v) for k, v in tracks.published_split.value_counts().items()},
            **{
                f"{field}s_in_multiple_splits": int(
                    tracks.groupby(f"{field}_id").published_split.nunique().gt(1).sum()
                )
                for field in ["artist", "album"]
            },
        },
        "input_sha256": {},
    }
    for name, path in sources.items():
        with path.open("rb") as file:
            summary["input_sha256"][name] = hashlib.file_digest(file, "sha256").hexdigest()
    coverage_table = pd.Series(coverage, name="tracks").rename_axis("metric").reset_index()
    coverage_table["share"] = coverage_table.tracks / len(tracks)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".fma-metadata-") as work:
        stage = Path(work) / "report"
        stage.mkdir()
        genre_table.to_csv(stage / "genre_support.csv", index_label="genre_id")
        pairs.to_csv(stage / "genre_cooccurrence.csv", index=False)
        artists.to_csv(stage / "artists.csv")
        albums.to_csv(stage / "albums.csv")
        coverage_table.to_csv(stage / "coverage.csv", index=False)
        _figures(stage, genre_table, pairs, artists, albums, durations, coverage_table)
        (stage / "interpretation.md").write_text(
            _interpretation(summary, genre_table), encoding="utf-8"
        )
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    return {"output": output, **summary}
