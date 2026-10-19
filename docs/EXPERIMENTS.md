# Experimental protocol

## Dataset and splits

Use the complete validated FMA-full catalogue for exploration and the demonstration
index. Record unreadable files and exclusions. Genre classification uses recordings
with usable annotations; the label set and hierarchy mapping are fixed after the audit.
FMA labels are hierarchical, imbalanced and potentially incomplete, so missing tags
must not be interpreted as verified absence of a genre.

Audit the published splits and create one versioned, artist-disjoint train/validation/test
manifest. Keep duplicate groups, recordings, windows and augmentations within one split.
Choose labels and report class support before fitting models. Fit scaling, imputation
and dimensionality reduction on training data only. Select models and thresholds on
validation data, then freeze them before the final test evaluation.

## Shared formats

`data/manifests/tracks.parquet` contains `track_id`, `audio_path`, `artist_id`,
`album_id`, `genre_ids`, `split`, `duplicate_group`, `duration_seconds`, `valid`
and `error`. Paths are relative to the data directory; `genre_ids` is a list.

Representations are stored under `artifacts/<representation>/<run_id>/` as
`vectors.npy` with shape `(n_tracks, d)`, aligned `track_ids.npy` and `metadata.json`.
Metadata records the manifest and configuration hashes, extractor/checkpoint version,
sample rate, window selection, aggregation and normalisation. Export fitted scalers
and class mappings alongside the model. Query processing uses the same settings.

The initial full profile samples three deterministic 10-second windows per track,
covering its beginning, middle and end; the quick profile samples one window.
Define the short-track policy in the shared loader and use the same windows across
methods. Full catalogue coverage does not imply processing every second of each track.

## Comparisons

Compare acoustic features with logistic regression and one tree-based model, a compact
CNN trained on log-mel spectrograms, and a frozen pretrained encoder with a linear head.
Select the encoder after checking its model card, licence, processing requirements and
pilot throughput. Use the CNN's penultimate layer for its retrieval representation.

Report macro-F1, micro-F1 and mAP with per-class support and a frequency baseline.
For retrieval, report P@10 and nDCG@10 using the documented genre-based relevance rule,
alongside a separate blinded listening comparison. Validation/test queries search a
training-only candidate pool. Exclude self matches and duplicates, and report the
effect of excluding the query artist. The full-catalogue demonstration index is
separate from this evaluation setup.

Use PCA/UMAP maps and clustering to examine structure, outliers and ambiguous examples.
Interpret cluster metrics in the representation space; a two-dimensional layout is
an exploratory view. Window length, preprocessing, augmentation and aggregation
experiments change one factor at a time while retaining the same split and metrics.

## Tracking and resources

Record the Git revision, seed, library versions, manifest hash, configuration, sample
counts, hardware, runtime, CPU/GPU hours, RAM/VRAM and metrics for each run in MLflow.
Write reports and figures to `outputs/<run_id>/`. Begin with the 128-track pilot and
use measured throughput to schedule full runs, reserving approximately 20% of the
4,000 CPU-hour and 300 GPU-hour allocations for reruns and integration.

Primary references: [FMA](https://github.com/mdeff/fma),
[librosa](https://librosa.org/doc/latest/), [CLAP](https://github.com/LAION-AI/CLAP).
[GTZAN's documented faults](https://github.com/boblsturm/GTZAN) provide context for
the treatment of repeated recordings and artist leakage in music benchmarks.
