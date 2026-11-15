"""Tables behind notebooks/01-data-exploration.ipynb, so the notebook reads as a report."""

import json
from ast import literal_eval
from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from music_atlas import audit as rules

# tracks.csv columns used in the report, with the name and meaning shown in its field table.
FIELDS = {
    ("track", "title"): ("title", "Track title"),
    ("artist", "id"): ("artist_id", "Artist ID; the split keeps each artist in one partition"),
    ("artist", "name"): ("artist", "Artist name"),
    ("album", "id"): ("album_id", "Album ID; missing when the track has no album"),
    ("track", "duration"): ("duration", "Published duration in whole seconds"),
    ("track", "bit_rate"): ("bit_rate", "MP3 bit rate (bit/s)"),
    ("track", "date_created"): ("date_created", "Upload date to the Free Music Archive"),
    ("track", "genres"): ("genres", "Genre IDs the artist tagged directly"),
    ("track", "genres_all"): ("genres_all", "Direct genres and all their ancestors: our labels"),
    ("track", "genre_top"): ("genre_top", "Single top-level genre, set only when it is unique"),
    ("track", "language_code"): ("language", "Language of the lyrics"),
    ("track", "license"): ("license", "Licence; open ones allow listening examples"),
    ("track", "listens"): ("listens", "Play count on the Free Music Archive"),
    ("track", "favorites"): ("favorites", "Times marked as a favourite"),
    ("track", "interest"): ("interest", "FMA's popularity score"),
    ("set", "subset"): ("subset", "Smallest FMA subset with the track: small, medium or large"),
    ("set", "split"): ("published_split", "Split published with FMA; kept for comparison only"),
}
METADATA_FILES = {
    "tracks.csv": "One row per track: artist, album, track fields, published split and subset",
    "genres.csv": "Genre hierarchy: ID, parent, title and top-level ancestor",
    "features.csv": "518 precomputed librosa statistics per track (not used: we extract our own)",
    "echonest.csv": "Echonest audio features for a 13k-track subset (not used)",
}
VERSION_FILES = {
    "train.parquet": "Training rows of the manifest",
    "validation.parquet": "Validation rows: model and threshold selection",
    "test.parquet": "Test rows: final evaluation only",
    "tracks.parquet": "Every catalogue row, with its partition or exclusion reason",
    "labels.json": "Genre column order and which genres are scored",
    "class_support.csv": "Tracks, artists and groups per genre and partition",
    "summary.json": "Version ID, input hashes, recipe, overlap checks and baselines",
    "balanced_eval.parquet": "Diagnostic per-genre view of validation and test",
}
WARNINGS = {"duration_mismatch": "Duration mismatch", "short": "Under one window", "quiet": "Quiet"}
HELD_OUT = ["validation", "test"]


@dataclass(frozen=True)
class Catalogue:
    inventory: pd.DataFrame  # every published track, indexed by track_id
    tracks: pd.DataFrame  # FMA metadata of the present tracks
    genres: pd.DataFrame  # with Direct and With descendants track counts
    direct_tags: pd.DataFrame  # (track_id, genre_id) pairs as tagged by the artists
    root_tags: pd.DataFrame  # (track_id, genre_id) pairs for top-level genres

    @property
    def roots(self) -> pd.DataFrame:
        return self.genres[self.genres["parent"].eq(0)]


def load_catalogue(data: Path) -> Catalogue:
    inventory = pd.read_csv(
        data / "manifests/fma_full_inventory.csv",
        usecols=["track_id", "shard", "audio_path", "size_bytes", "status"], index_col="track_id",
    )
    ids = inventory.index[inventory["status"].eq("present")]
    source = data / "fma/fma_metadata"
    tracks = pd.read_csv(source / "tracks.csv", index_col=0, header=[0, 1], low_memory=False)
    names = [name for name, _ in FIELDS.values()]
    tracks = tracks[list(FIELDS)].set_axis(names, axis=1).reindex(ids).rename_axis("track_id")
    for field in ("artist_id", "album_id"):
        tracks[field] = tracks[field].where(tracks[field] > 0)
    tracks["date_created"] = pd.to_datetime(tracks["date_created"], errors="coerce")
    tracks["published_split"] = tracks["published_split"].replace("training", "train")
    genres = pd.read_csv(source / "genres.csv", index_col="genre_id")
    tags = {}
    for field in ("genres", "genres_all"):
        lists = tracks[field].fillna("[]").map(literal_eval)
        tags[field] = (
            lists.explode().dropna().astype(int).rename("genre_id").reset_index().drop_duplicates()
        )
        tracks[field] = lists
    genres["Direct"] = tags["genres"]["genre_id"].value_counts()
    genres["With descendants"] = tags["genres_all"]["genre_id"].value_counts()
    genres[["Direct", "With descendants"]] = genres[["Direct", "With descendants"]].fillna(0)
    roots = genres.index[genres["parent"].eq(0)]
    root_tags = tags["genres_all"][tags["genres_all"]["genre_id"].isin(roots)]
    return Catalogue(inventory, tracks, genres.astype({"Direct": int, "With descendants": int}),
                     tags["genres"].reset_index(drop=True), root_tags.reset_index(drop=True))


def source_files(data: Path, catalogue: Catalogue) -> pd.DataFrame:
    """The downloaded files: audio shards and the metadata tables the report reads."""
    inventory = catalogue.inventory
    rows = {"fma_full/NNN.tar": {
        "Size (GiB)": inventory["size_bytes"].sum() / 2**30,
        "Contents": f"{inventory['shard'].nunique()} verified tar shards with "
                    f"{len(inventory):,} original MP3s, one shard per FMA folder",
    }}
    for name, contents in METADATA_FILES.items():
        path = data / "fma/fma_metadata" / name
        if path.is_file():
            rows[f"fma_metadata/{name}"] = {"Size (GiB)": path.stat().st_size / 2**30,
                                            "Contents": contents}
    rows["manifests/fma_full_inventory.csv"] = {
        "Size (GiB)": (data / "manifests/fma_full_inventory.csv").stat().st_size / 2**30,
        "Contents": "Our index: track_id, shard, path inside the shard, size, SHA-1, status",
    }
    return pd.DataFrame(rows).T.rename_axis("File")


def field_table(catalogue: Catalogue) -> pd.DataFrame:
    """Meaning, type, share missing and an example value of every field the report uses."""
    tracks = catalogue.tracks
    example = tracks.loc[tracks["genres"].map(len).gt(1) & tracks["album_id"].notna()].iloc[0]
    rows = {}
    for name, description in FIELDS.values():
        values = tracks[name]
        if isinstance(example[name], list):
            missing, kind = values.map(len).eq(0), "list of genre IDs"
        else:
            missing = values.isna() | values.astype("string").str.strip().eq("")
            kind = "date" if name == "date_created" else (
                "number" if pd.api.types.is_numeric_dtype(values) else "text")
        rows[name] = {"Meaning": description, "Type": kind,
                      "Missing (%)": round(100 * missing.mean(), 1),
                      "Example": str(example[name])[:40]}
    return pd.DataFrame(rows).T.rename_axis(f"Field (example: track {example.name})")


def genre_tree(genres: pd.DataFrame) -> str:
    """Collapsible HTML list of the FMA hierarchy with track counts."""
    children = genres.sort_values("title").groupby("parent").groups

    def node(genre_id):
        row = genres.loc[genre_id]
        label = f"{escape(row['title'])} · {row['With descendants']:,}"
        below = children.get(genre_id, ())
        if not len(below):
            return label
        items = "".join(f"<li>{node(child)}</li>" for child in below)
        return f"<details><summary><strong>{label}</strong></summary><ul>{items}</ul></details>"

    roots = genres[genres["parent"].eq(0)].sort_values("With descendants", ascending=False)
    return "<ul>" + "".join(f"<li>{node(root)}</li>" for root in roots.index) + "</ul>"


def concentration(rows: pd.DataFrame, keys=("split", "genre_id")) -> pd.DataFrame:
    """Tracks, artists, largest artist share and effective artists (inverse Simpson) per key."""
    keys = list(keys)
    per_artist = rows.groupby([*keys, "artist_id"]).size()  # unknown artists drop out here
    grouped = per_artist.groupby(level=keys)
    stats = pd.DataFrame({"Tracks": rows.groupby(keys).size(), "Artists": grouped.size()})
    stats["Top artist (%)"] = 100 * grouped.max() / stats["Tracks"]
    stats["Effective artists"] = grouped.sum() ** 2 / (per_artist**2).groupby(level=keys).sum()
    return stats


def genre_overview(catalogue: Catalogue) -> pd.DataFrame:
    """Per top-level genre: size, artist concentration and size in FMA's smaller subsets."""
    tracks = catalogue.tracks
    rows = catalogue.root_tags.join(tracks["artist_id"], on="track_id")
    table = concentration(rows, keys=["genre_id"])
    table.insert(1, "Catalogue (%)", 100 * table["Tracks"] / len(tracks))
    table.index = table.index.map(catalogue.genres["title"]).rename("Genre")
    # FMA's single-label subsets use genre_top, so they leave out every multi-genre track.
    table["fma-small"] = tracks.loc[tracks["subset"].eq("small"), "genre_top"].value_counts()
    table["fma-medium"] = tracks.loc[
        tracks["subset"].isin(["small", "medium"]), "genre_top"
    ].value_counts()
    table[["fma-small", "fma-medium"]] = table[["fma-small", "fma-medium"]].fillna(0).astype(int)
    return table.sort_values("Tracks", ascending=False)


def label_overlap(catalogue: Catalogue, count: int = 10) -> pd.DataFrame:
    """Jaccard overlap between the most frequent direct labels."""
    genres = catalogue.genres
    common = genres.nlargest(count, "Direct").index
    tags = catalogue.direct_tags[catalogue.direct_tags["genre_id"].isin(common)]
    flags = pd.crosstab(tags["track_id"], tags["genre_id"]).reindex(columns=common)
    shared = flags.T @ flags
    totals = pd.Series(shared.to_numpy().diagonal(), index=common)
    overlap = shared / (totals.to_numpy()[:, None] + totals.to_numpy() - shared)
    titles = genres.loc[common, "title"]
    return overlap.set_axis(titles, axis=0).set_axis(titles, axis=1)


def numeric_correlations(catalogue: Catalogue) -> pd.DataFrame:
    """Spearman correlations between numeric metadata, robust to their heavy tails."""
    tracks = catalogue.tracks
    roots = catalogue.root_tags.groupby("track_id").size().reindex(tracks.index, fill_value=0)
    values = pd.DataFrame({
        "Duration": tracks["duration"], "Bit rate": tracks["bit_rate"],
        "Upload year": tracks["date_created"].dt.year, "Listens": tracks["listens"],
        "Favourites": tracks["favorites"], "Interest": tracks["interest"],
        "Direct tags": tracks["genres"].map(len), "Top-level genres": roots,
    })
    return values.corr(method="spearman")


def concentration_curves(catalogue: Catalogue) -> dict[str, pd.Series]:
    """Cumulative catalogue share of artists and albums ranked by track count."""
    tracks = catalogue.tracks
    return {
        label: (tracks[field].value_counts().cumsum() / len(tracks)).reset_index(drop=True)
        for field, label in {"artist_id": "Artists", "album_id": "Albums"}.items()
    }


def sparse_genres(catalogue: Catalogue, min_tracks: int = 20) -> pd.DataFrame:
    """Direct labels resting on the fewest artists or dominated by one."""
    genres = catalogue.genres
    tagged = catalogue.direct_tags.join(catalogue.tracks["artist_id"], on="track_id")
    table = concentration(tagged, keys=["genre_id"])
    table = table.join(genres["title"]).set_index("title")
    fewest = table.nsmallest(5, ["Artists", "Tracks"]).assign(Why="Fewest artists")
    dominated = table[table["Tracks"].ge(min_tracks)].nlargest(5, "Top artist (%)")
    return pd.concat([fewest, dominated.assign(Why="One artist dominates")]).set_index(
        "Why", append=True).swaplevel().rename_axis(["", "Genre"])


def annotation_coverage(catalogue: Catalogue) -> pd.Series:
    tracks = catalogue.tracks
    direct, roots = tracks["genres"].map(len), catalogue.root_tags.groupby("track_id").size()
    roots = roots.reindex(tracks.index, fill_value=0)
    return pd.Series({
        "No genre tag": direct.eq(0).sum(),
        "One direct tag": direct.eq(1).sum(),
        "Several direct tags": direct.gt(1).sum(),
        "Several top-level genres": roots.gt(1).sum(),
        "Unknown artist": tracks["artist_id"].isna().sum(),
        "Unknown album": tracks["album_id"].isna().sum(),
        "No published split": tracks["published_split"].isna().sum(),
    }, name="Tracks")


def load_audit(data: Path, catalogue: Catalogue) -> pd.DataFrame:
    """The audit table with whole-track levels, warning flags and track metadata."""
    audit = pd.read_parquet(data / "manifests/fma_audio_audit.parquet").set_index("track_id")
    levels = pd.concat(
        pd.read_parquet(path, columns=["track_id", "peak", "rms_dbfs", "error"])
        for path in sorted((data / "manifests/fma_audio_audit").glob("*.parquet"))
    ).set_index("track_id").rename(columns={"error": "decoder_message"})
    flags = audit["warning"].str.get_dummies(";").astype(bool)
    flags = flags.reindex(columns=list(WARNINGS), fill_value=False)
    metadata = catalogue.tracks[["artist", "title", "artist_id", "duration", "published_split"]]
    return audit.join(levels).join(flags).join(metadata)


def audit_thresholds() -> pd.Series:
    return pd.Series({
        "Excluded: shorter than (s)": rules.MIN_SECONDS,
        "Excluded: digitally silent, peak below": rules.SILENT_PEAK,
        "Warning: differs from the published duration by over (s)": rules.DURATION_TOLERANCE,
        "Warning: shorter than one window (s)": rules.SHORT_SECONDS,
        "Warning: RMS below (dBFS)": rules.QUIET_DBFS,
        "Re-encoding candidate: durations within (s)": rules.MATCH_SECONDS,
        "Re-encoding candidate: loudness contour correlation from": rules.MATCH_CORRELATION,
    }, name="Threshold")


def audit_outcome(audit: pd.DataFrame, catalogue: Catalogue) -> pd.Series:
    valid = audit["valid"]
    untagged = ~audit.index.isin(catalogue.root_tags["track_id"])
    return pd.Series({
        "Catalogue": len(audit),
        "Valid": valid.sum(),
        **{f"Excluded: {reason}": n for reason, n in audit["error"].value_counts().items()},
        "Valid with a warning": (valid & audit[list(WARNINGS)].any(axis=1)).sum(),
        "Valid in a confirmed duplicate group": (valid & audit["duplicate_group"].notna()).sum(),
        "Valid, unresolved re-encoding candidate": (
            valid & audit["reencoding_candidate"].notna()
        ).sum(),
        "Valid without a top-level genre": (valid & untagged).sum(),
        "Open licence (listening example)": audit["listening_example"].sum(),
    }, name="Tracks")


def decoder_messages(audit: pd.DataFrame, count: int = 5) -> pd.Series:
    """The most frequent decoder errors, with file paths masked."""
    messages = audit["decoder_message"].dropna().str.replace(r"<[^>]*>", "<file>", regex=True)
    return messages.value_counts().head(count).rename("Tracks").rename_axis("Decoder message")


def problem_examples(audit: pd.DataFrame, per_problem: int = 2) -> pd.DataFrame:
    ratio = audit["duration_seconds"] / audit["duration"].where(audit["duration"] > 0)
    problems = {
        "Cannot be decoded": audit[audit["error"].isin(["decode_error", "loader_error"])],
        "Digitally silent": audit[audit["error"].eq("silent")],
        "Too short": audit[audit["error"].eq("too_short")],
        "Truncated (warning)": audit[audit["duration_mismatch"]].assign(ratio=ratio)
        .sort_values("ratio"),
        "Quiet (warning)": audit[audit["quiet"]].sort_values("rms_dbfs"),
    }
    columns = ["artist", "title", "duration", "duration_seconds", "rms_dbfs"]
    return pd.concat(
        rows[columns].head(per_problem).assign(Problem=name) for name, rows in problems.items()
    ).set_index("Problem", append=True).swaplevel().rename(columns={
        "duration": "Published (s)", "duration_seconds": "Decoded (s)", "rms_dbfs": "RMS (dBFS)",
    })


def duplicate_report(audit: pd.DataFrame) -> pd.DataFrame:
    """Confirmed duplicates and re-encoding candidates, and which of them a split could break."""

    def report(column):
        groups = audit.dropna(subset=[column]).groupby(column)
        return {
            "Groups": groups.ngroups,
            "Tracks": int(groups.size().sum()),
            "Groups with several artists": int(groups["artist_id"].nunique().gt(1).sum()),
            "Groups across published splits": int(groups["published_split"].nunique().gt(1).sum()),
        }

    return pd.DataFrame({
        "Confirmed duplicates (same SHA-1)": report("duplicate_group"),
        "Re-encoding candidates": report("reencoding_candidate"),
    }).T


def genre_effect(audit: pd.DataFrame, catalogue: Catalogue) -> pd.DataFrame:
    rows = catalogue.root_tags.join(audit[["valid", "artist_id", *WARNINGS]], on="track_id")
    every, kept = rows.groupby("genre_id"), rows[rows["valid"]].groupby("genre_id")
    table = pd.DataFrame({
        "Tracks": every.size(),
        "Excluded (%)": 100 * (1 - kept.size() / every.size()),
        "Artists lost": every["artist_id"].nunique() - kept["artist_id"].nunique(),
        **{f"{label} (%)": 100 * kept[flag].mean() for flag, label in WARNINGS.items()},
    })
    table.index = table.index.map(catalogue.genres["title"]).rename("Genre")
    return table.sort_values("Tracks", ascending=False)


def held_out_profile(directory: Path, root_ids) -> pd.DataFrame:
    """Concentration of each top-level genre in validation and test of one dataset version."""
    assigned = pd.read_parquet(
        directory / "tracks.parquet", columns=["track_id", "split", "artist_id", "genre_ids"]
    ).dropna(subset=["split"])
    rows = assigned.explode("genre_ids").rename(columns={"genre_ids": "genre_id"})
    rows = rows[rows["genre_id"].isin(root_ids)]
    held = concentration(rows).loc[HELD_OUT]
    genre_size = rows.groupby("genre_id").size()
    held["Share of genre (%)"] = (
        100 * held["Tracks"] / held.index.get_level_values("genre_id").map(genre_size).to_numpy()
    )
    return held


def compare_versions(ready: Path, candidates: dict, root_ids, audit: pd.DataFrame):
    """One row per built candidate version with the criteria used to select the split."""
    profiles, rows = {}, {}
    for version, description in candidates.items():
        directory = ready / version
        if not (directory / "tracks.parquet").is_file():
            continue
        held = profiles[version] = held_out_profile(directory, root_ids)
        assigned = pd.read_parquet(directory / "tracks.parquet", columns=["track_id", "split"])
        assigned = assigned.dropna(subset=["split"]).set_index("track_id")
        support = pd.read_csv(directory / "class_support.csv").query("scheme == 'canonical'")
        candidate_splits = assigned.join(audit["reencoding_candidate"]).dropna()
        rows[version] = {
            "Allocation": description,
            "Train tracks": assigned["split"].eq("train").sum(),
            "Largest artist share (%)": held["Top artist (%)"].max(),
            "Smallest held-out share (%)": held["Share of genre (%)"].min(),
            "Fewest effective artists": held["Effective artists"].min(),
            "Max prevalence shift (pp)": 100 * support["prevalence_shift"].abs().max(),
            "Re-encoding groups across partitions": candidate_splits
            .groupby("reencoding_candidate")["split"].nunique().gt(1).sum(),
        }
    return pd.DataFrame(rows).T.infer_objects(), profiles


def select_version(comparison: pd.DataFrame, versions, max_artist_share: float,
                   min_held_out_share: float) -> str | None:
    """Among versions meeting both limits, prefer artist diversity, then the smallest shift."""
    eligible = comparison.loc[list(versions)]
    eligible = eligible[eligible["Largest artist share (%)"].le(max_artist_share)
                        & eligible["Smallest held-out share (%)"].ge(min_held_out_share)]
    if eligible.empty:
        return None
    ranked = eligible.sort_values(
        ["Fewest effective artists", "Max prevalence shift (pp)"], ascending=[False, True]
    )
    return ranked.index[0]


def by_genre(profiles: dict, versions, column: str, reducer: str, genres) -> pd.DataFrame:
    """One column per version: a held-out statistic reduced over validation and test."""
    table = pd.DataFrame({
        version: profiles[version][column].groupby(level="genre_id").agg(reducer)
        for version in versions
    })
    return table.rename(index=genres["title"])


def split_support(directory: Path, profile: pd.DataFrame, genres: pd.DataFrame) -> pd.DataFrame:
    """Validation and test tracks, artists and largest artist share per top-level genre."""
    columns = ["Tracks", "Artists", "Top artist (%)"]
    table = profile[columns].unstack("split").swaplevel(axis=1)
    table = table[[(split, column) for split in HELD_OUT for column in columns]]
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    scored = {label["genre_id"] for label in labels if label["scored"]}
    table.insert(0, ("", "Scored"), table.index.isin(scored))
    table.index = table.index.map(genres["title"]).rename("Genre")
    return table.sort_values(("validation", "Tracks"), ascending=False)


def held_out_versions(ready: Path, versions) -> pd.DataFrame:
    """Held-out tracks / independent groups per genre, side by side for several versions."""
    rows = pd.concat(
        pd.read_csv(ready / version / "class_support.csv").assign(version=version)
        for version in versions
    ).query("scheme == 'canonical' and split in @HELD_OUT")
    rows["Tracks / groups"] = rows["tracks"].astype(str) + " / " + rows["groups"].astype(str)
    return rows.pivot(index="genre", columns=["version", "split"], values="Tracks / groups")


def version_contents(ready: Path, versions, catalogue: Catalogue) -> pd.DataFrame:
    """What each version holds, and whether its labels match FMA's published genres_all."""
    published = catalogue.tracks["genres_all"].map(frozenset)
    roots = frozenset(catalogue.roots.index)
    rows = {}
    for version in versions:
        manifest = pd.read_parquet(
            ready / version / "tracks.parquet", columns=["track_id", "split", "genre_ids"]
        ).set_index("track_id")
        assigned = manifest.dropna(subset=["split"])
        ours = assigned["genre_ids"].map(frozenset)
        theirs = published.reindex(assigned.index)
        rows[version] = {
            "Catalogue rows": len(manifest),
            "Assigned tracks": len(assigned),
            "Not assigned": len(manifest) - len(assigned),
            "No genre annotation": assigned["genre_ids"].map(len).eq(0).sum(),
            "Labels differ from FMA genres_all": ours.ne(theirs).sum(),
            "Top-level labels differ": ours.map(roots.intersection)
            .ne(theirs.map(roots.intersection)).sum(),
        }
    return pd.DataFrame(rows).T


def version_files(directory: Path) -> pd.DataFrame:
    """The files of one dataset version, with their size and row count."""
    rows = {}
    for name, contents in VERSION_FILES.items():
        path = directory / name
        if path.is_file():
            rows[name] = {
                "Size (MB)": path.stat().st_size / 1e6,
                "Rows": pq.read_metadata(path).num_rows if path.suffix == ".parquet" else None,
                "Contents": contents,
            }
    return pd.DataFrame(rows).T.rename_axis("File")


def storage_sizes(data: Path, catalogue: Catalogue, directory: Path) -> pd.Series:
    """Bytes of the audio next to the metadata that splits and audits add on top of it."""
    ready = sorted(path for path in (data / "ready").iterdir() if not path.is_symlink())
    return pd.Series({
        "Audio shards (MP3)": catalogue.inventory["size_bytes"].sum(),
        "Audio audit": sum(path.stat().st_size for path in
                           (data / "manifests").glob("fma_audio_audit*") if path.is_file()),
        f"All {len(ready)} split versions": sum(
            path.stat().st_size for version in ready for path in version.iterdir()),
        "One split version": sum(path.stat().st_size for path in directory.iterdir()),
    }, name="Bytes")


def published_version(data: Path) -> str | None:
    """The version that manifests/tracks.parquet was published from, if any."""
    path = data / "manifests/tracks.parquet"
    if not path.is_file():
        return None
    return (pq.read_schema(path).metadata or {}).get(b"dataset_version", b"").decode() or None


def partition_shares(directory: Path, root_ids, genres: pd.DataFrame) -> pd.DataFrame:
    """Share of each top-level genre's assigned tracks in train, validation and test."""
    assigned = pd.read_parquet(directory / "tracks.parquet", columns=["split", "genre_ids"])
    rows = assigned.dropna(subset=["split"]).explode("genre_ids")
    rows = rows[rows["genre_ids"].isin(root_ids)]
    counts = pd.crosstab(rows["genre_ids"], rows["split"])[["train", *HELD_OUT]]
    shares = 100 * counts.div(counts.sum(axis=1), axis=0)
    return shares.rename(index=genres["title"]).sort_values("train")


def split_leakage(directory: Path, audit: pd.DataFrame) -> pd.DataFrame:
    """Groups spread over several partitions, in our split and in the published FMA split."""
    manifest = pd.read_parquet(
        directory / "tracks.parquet",
        columns=["track_id", "split", "artist_id", "album_id", "duplicate_group"],
    ).dropna(subset=["split"]).set_index("track_id")
    rows = manifest.join(audit[["reencoding_candidate", "published_split"]])
    fields = {"artist_id": "Artists", "album_id": "Albums", "duplicate_group": "Duplicate groups",
              "reencoding_candidate": "Re-encoding candidate groups"}
    return pd.DataFrame({
        name: {label: rows.groupby(field)[column].nunique().gt(1).sum()
               for field, label in fields.items()}
        for name, column in {"Published FMA split": "published_split", "Ours": "split"}.items()
    }).T
