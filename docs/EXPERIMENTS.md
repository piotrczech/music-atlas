# Experimental protocol

`notebooks/01-data-exploration.ipynb` shows the data, the audit and the choice of split with
saved results; this document defines the rules and formats behind them.

## Dataset

The project uses FMA-full. FMA labels are hierarchical, imbalanced and incomplete, so a
missing tag is unknown, not a verified absence of a genre. Every representation shares one
versioned, artist-disjoint train/validation/test split. Keep artists, duplicate groups,
recordings, windows and augmentations within one split. Fit scaling, imputation and
dimensionality reduction on training data only. Select models and thresholds on validation,
then evaluate once on test.

**Primary benchmark: `fma-cap10-v1`**, with `fma-rare-support-cap10-v1` as its supplementary
partition. The criteria that selected it are in section 3 of the data notebook. Code reads the
version from `[dataset] split` in `configs/*.toml` rather than repeating it.

## Data pipeline

Each step is a CPU job that writes under `$ATLAS_DATA_DIR` (commands in
[DEVELOPMENT.md](DEVELOPMENT.md#data-pipeline)).

| Step | Command | Output |
|---|---|---|
| Download | `atlas fma-download` | `fma/fma_full/NNN.tar` (156 shards), `fma/fma_metadata/`, `manifests/fma_full_inventory.csv` |
| Audit | `atlas fma-audit` | `manifests/fma_audio_audit.parquet` and `.json` |
| Split | `atlas fma-split --audit …` | `ready/<version>/`; with `--publish`, `manifests/tracks.parquet` |
| Quick sample | `atlas quick-sample` | `samples/<version>/`: `sample.json`, `rejects.csv`, `audio.tar` |
| Exclusions | `atlas fma-record-filter` | `filter.parquet` beside a representation run |

### Inventory

`manifests/fma_full_inventory.csv` has one row per published track: `track_id`, `shard` (a
tar relative to the data directory), `audio_path` (the member inside it), `size_bytes`,
`sha1` and `status` (`present` or `missing`). Jobs read MP3s straight from the shards.

### Audit

`atlas fma-audit` decodes every present track from its shard and reads it through the shared
loader, so `valid` means the experiments can load it. The table has one row per inventory track:

- `error` excludes a track for a technical failure only: `inventory_missing`, `decode_error`,
  `loader_error`, `too_short` (under 1 s) or `silent` (peak under −60 dBFS).
- `warning` marks unusual but valid audio, which stays in the dataset: `duration_mismatch`
  (over 2 s from FMA's duration), `short` (under one window) or `quiet` (RMS under −40 dBFS).
- `duplicate_group` is the lowest track ID sharing a SHA-1; the split keeps the group together.
- `reencoding_candidate` links tracks within 1 s of each other whose per-second loudness over
  the first two minutes correlates at 0.98 or more. Candidates are reported, not resolved.
- `listening_example` marks valid tracks under CC0, Public Domain Mark, CC BY or CC BY-SA.

Thresholds live in `music_atlas.audit`. `fma_audio_audit.json` counts exclusions, annotation
gaps and per-genre coverage.

### Splits

`atlas fma-split` assigns valid tracks with a known artist to train, validation and test in an
80/10/10 ratio. It moves whole groups: connected components of artists and confirmed duplicates.
Starting from a seeded random assignment, it moves groups until each top-level genre has its
catalogue share of tracks and of independent groups in every partition, following the
proportional objective of iterative multi-label stratification. Unlabelled valid tracks keep a
partition for retrieval but are not scored. Excluded rows stay in the manifest with their
reason; the audio stays in the shards.

Options change the allocation and are part of the version name:

- `--strategy rare-support` raises sparse genres to up to 250 positives per held-out split,
  capped at 20% of the genre, taking them from training. It is a separate dataset.
- `--max-artist-share 0.1` keeps an artist in training when it would supply more than 10% of a
  genre's validation or test quota (`fma-cap10-v1`).
- `--concentration-penalty 4` makes such artists costlier to hold out (`fma-penalty4-v1`).
- `--publish` copies the version to `manifests/tracks.parquet`; only an audited proportional
  version can be published.

Without `--audit` the job builds a pre-audit version (`audio_audited=false`), where `valid`
only means present and durations are FMA's. Pre-audit versions are kept for reference.

`ready/<version>/` holds `tracks.parquet`, `train.parquet`, `validation.parquet`,
`test.parquet`, `labels.json`, `class_support.csv`, `summary.json` and
`balanced_eval.parquet`. Manifests carry `track_id`, `shard`, `audio_path`, `artist_id`,
`album_id`, `genre_ids` (a list), `split`, `duplicate_group`, `duration_seconds`, `valid`
and `error`. A short link such as `ready/fma-cap10-v1` points to the full version
directory; pin runs to the full version ID and `manifest_sha256` from `summary.json`. Keep the
three split files of one version together and compare models within one version.

`balanced_eval.parquet` is a diagnostic one-vs-rest view of the existing validation and test
tracks: for each scored genre it samples equally many tagged and untagged tracks, preferring
distinct groups. Keep its results apart from the primary metrics.

### Quick sample

`atlas quick-sample` fixes the quick profile's input: up to `max_tracks` valid training tracks
of the profile's dataset version in its seeded order, taking the scored genres in turn and
preferring artists not yet sampled. `samples/<version>/` holds `sample.json` (dataset version,
manifest hash, seed, profile, scored genres with their names and tracks), `rejects.csv` (`track_id,error` for
`atlas fma-record-filter`) and `audio.tar` with the accepted MP3s. The first run reads them from
the shards in place; later runs, including local ones, read only `audio.tar` and stop if the
version's manifest hash has changed.

### Model-specific exclusions

Record tracks a representation cannot use as `track_id,error` rows. `atlas fma-record-filter
VERSION REPRESENTATION RUN_ID exclusions.csv` writes `filter.parquet` beside the run with
inclusion, reason, dataset version and manifest hash, without moving tracks between
partitions. A header-only CSV records a run that keeps every track.

## Labels and metrics

The manifest propagates each direct tag to all its ancestors. `labels.json` fixes the
top-level genre columns in genre-ID order. A column is scored only when every split has
positive and negative support and each held-out split has at least two independent positive
groups. Validation and test keep natural prevalence.

Report macro-F1, per-genre recall and mAP alongside micro-F1, with track, artist and group
support, next to the train-frequency and all-negative baselines in `summary.json`. Any
reweighting or resampling happens within training.

## Prior practice

| Practice | Source | Our choice |
|---|---|---|
| 80/10/10 split stratified by genre, each artist in one split, unlabelled tracks in training | FMA paper, §2.7 [1] | **Adopted** fractions and the artist constraint. Unlabelled valid tracks keep a partition for retrieval but are not scored. |
| The published split, exactly stratified only for fma-small | FMA paper, §2.7 [1] | **Kept for comparison.** On the audited fma-full catalogue it puts 11 duplicate groups in two partitions and shifts validation prevalence by up to 5 points. |
| Single top-genre labels on fma-medium, scored by accuracy; the WWW 2018 challenge scored log loss and F1 | FMA paper, §3.2 [1]; challenge [2] | **Rejected as our task.** We score the 16 top-level genres as multiple labels on fma-full. |
| Artist filter, because classifiers otherwise learn artists | Flexer [3]; FMA paper [1] | **Adopted** as a hard constraint, extended to confirmed duplicates. |
| Repetition faults and classes dominated by few artists, e.g. GTZAN Blues from nine artists | Sturm [4] | **Adopted and extended.** Duplicates are grouped, re-encodings reported, and `--max-artist-share` limits one artist's share of a held-out genre. |
| Known fma-full issues: wrong lengths, 0 s clips, 251 tracks with bad ID3 tags, 937 surplus duplicate copies | FMA wiki [5] | **Adopted.** Decoded durations replace metadata, the loader reads past unreadable ID3 tags, and duplicates are grouped (the audit finds the same 595 groups). |
| Iterative label-wise stratification; a second-order variant balances label pairs | Sechidis et al. [6]; Szymański and Kajdanowicz [7] | **Adapted** to whole artist groups. Label pairs are not balanced: grouped units leave too little freedom. |
| Tags kept only with at least 50 artists; five alternative splits | MTG-Jamendo [8] | **Adapted.** A genre is scored with at least two independent groups per held-out split, and effective artists are reported. One split is frozen; other seeds remain a robustness check. |

Only results on fma-small or fma-medium with the published split and single top-genre accuracy
compare directly with the FMA baselines. Report ours as our own protocol.

[1] [FMA: A Dataset for Music Analysis](https://arxiv.org/abs/1612.01840), ISMIR 2017.
[2] [Learning to Recognize Musical Genre from Audio](https://arxiv.org/abs/1803.05337), WWW 2018.
[3] [A closer look on artist filters for musical genre classification](https://archives.ismir.net/ismir2007/paper/000341.pdf), ISMIR 2007.
[4] [The GTZAN dataset: its contents, its faults, their effects on evaluation](https://arxiv.org/abs/1306.1461), 2013.
[5] [FMA known issues](https://github.com/mdeff/fma/wiki).
[6] [On the stratification of multi-label data](https://doi.org/10.1007/978-3-642-23808-6_10), ECML PKDD 2011.
[7] [A network perspective on stratification of multi-label data](https://proceedings.mlr.press/v74/szyma%C5%84ski17a.html), LIDTA 2017.
[8] [The MTG-Jamendo dataset for automatic music tagging](https://github.com/MTG/mtg-jamendo-dataset), ML4MD 2019.

## Representations and windows

Representations are stored under `artifacts/<representation>/<run_id>/` as `vectors.npy`
with shape `(n_tracks, d)`, aligned `track_ids.npy` and `metadata.json` (manifest and
configuration hashes, extractor or checkpoint version, sample rate, window selection,
aggregation and normalisation). Export fitted scalers and class mappings with the model;
queries use the same settings.

`music_atlas.loader` cuts deterministic 10-second windows: three per track (beginning, middle,
end) in the full profile and one centred window in the quick profile. Windows overlap in short
tracks, a track shorter than one window gets one zero-padded window, and an MP3 libsndfile
rejects for its ID3 tag is read from its first frame.

## Comparisons

Compare acoustic features with logistic regression and one tree-based model, a compact CNN
on log-mel spectrograms (its penultimate layer for retrieval), and a frozen pretrained encoder
with a linear head, chosen after checking its model card, licence and pilot throughput.

For retrieval, report P@10 and nDCG@10 with the documented genre-based relevance rule and a
separate blinded listening comparison. Validation and test queries search a training-only pool;
exclude self matches and duplicates, and report the effect of excluding the query artist. The
full-catalogue demonstration index is separate from evaluation.

Use PCA/UMAP maps and clustering to examine structure and outliers; a two-dimensional layout is
an exploratory view. Window, preprocessing, augmentation and aggregation experiments change one
factor at a time on the same split and metrics.

## Tracking and resources

Record the Git revision, seed, library versions, manifest hash, configuration, sample counts,
hardware, runtime, CPU/GPU hours, RAM/VRAM and metrics for each run in MLflow, and write
reports and figures to `outputs/<run_id>/`. Start with the 128-track quick sample and use its
throughput to schedule full runs, keeping about 20% of the 4,000 CPU-hour and 300 GPU-hour
allocations for reruns and integration.
