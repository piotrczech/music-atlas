# Experimental protocol

## Dataset and splits

Use the FMA-full inventory for initial exploration and preliminary experiments. Freeze
the audited catalogue for the final evaluation and demonstration index. Record unreadable
files and exclusions. Genre classification uses recordings with usable annotations.
FMA labels are hierarchical, imbalanced and potentially incomplete, so missing tags
must not be interpreted as verified absence of a genre.

Audit the published splits and create one versioned, artist-disjoint train/validation/test
manifest. Keep duplicate groups, recordings, windows and augmentations within one split.
Choose labels and report class support before fitting models. Fit scaling, imputation
and dimensionality reduction on training data only. Select models and thresholds on
validation data, then freeze them before the final test evaluation.

## Shared formats

`data/manifests/fma_full_inventory.csv` lists each published FMA-full track with `track_id`,
`shard`, `audio_path`, `size_bytes`, `sha1` and `status` (`present` or `missing`).
`data/manifests/tracks.parquet` contains `track_id`, `shard`, `audio_path`, `artist_id`,
`album_id`, `genre_ids`, `split`, `duplicate_group`, `duration_seconds`, `valid`
and `error`. `shard` is a tar relative to the data directory and `audio_path` the file
inside it; `genre_ids` is a list.

### Preparing FMA splits

Run the CPU job with no arguments to prepare a version from the verified inventory and
published FMA metadata. The resulting Parquet files carry `audio_audited=false`.
In this version `valid` means the inventory marks the file present, and duration is
the published FMA value.

```bash
source ~/.atlas-env.sh
cd ~/music-atlas
uv sync --locked --extra research --extra notebook
sbatch --output="$ATLAS_OUTPUTS_DIR/fma-split-%j.log" scripts/slurm/fma-split.sbatch
```

Run the same job with `--strategy rare-support` to produce a second complete partition
of the same eligible tracks. Both target 80/10/10 overall. The default preserves genre
prevalence. Rare-support targets up to 250 positives per genre in each held-out split
when the proportional target is smaller, capped at 20% of that genre's tracks per split.
Independent group targets follow the same fractions. Whole groups make those counts
approximate. Compare models within the same dataset version and use the proportional
version for the primary prevalence-preserving benchmark. Keep all three split files
from one version together in each experiment.

**E1-03 integration.** Add `--audit "$ATLAS_DATA_DIR/manifests/fma_audio_audit.parquet"`
after E1-03 is complete. The audit needs one row per present `track_id`, with Boolean
`valid`, an `error` for rejected audio, positive decoded `duration_seconds` for valid
audio and a nullable `duplicate_group` for confirmed duplicates. Keep re-encoding
candidates out of the confirmed group column.

The job compares the published split, random assignment of groups and proportionate
assignment of groups. Open `notebooks/fma_splits.ipynb` in OOD with the **Music Atlas**
kernel to inspect the version, class counts and baselines. The E1-04 follow-up compares
expanded labels with the published FMA annotations, checks for shared albums and shows
artist concentration and duration cutoff sensitivity by genre and split.

The canonical 80/10/10 split assigns connected components of artists and confirmed or
exact duplicates. The allocation uses every original top-level label on a track and
balances both track and independent group counts approximately. Known artist IDs are
required for a split assignment. Valid unlabelled tracks keep a partition for retrieval
and stay out of supervised scoring. Rejected rows remain in the full manifest with the
audit reason. The audio shards stay in place.

`$ATLAS_DATA_DIR/ready/<version>/` contains `tracks.parquet`, `train.parquet`, `validation.parquet`,
`test.parquet`, `labels.json`, `class_support.csv` and `summary.json`. The split files
are metadata only and belong on the shared music service Lustre storage on WCSS.
The audited proportional run also publishes `$ATLAS_DATA_DIR/manifests/tracks.parquet`. The
pre-audit version stays under `ready/` and is available for preliminary experiments.
The version derives from the input hashes and split recipe. Use the versioned files
for experiments. `balanced_eval.parquet` is a supplementary one-vs-rest view of the
existing validation and test tracks. For each scored genre it samples equally many
positives and annotated tracks without that tag, using the smallest pool as the common
quota and prioritising distinct groups. Keep results from this diagnostic separate
from the primary test metrics, which use the original prevalence.

Record model-specific exclusions as `track_id,error` rows. `atlas fma-record-filter
VERSION REPRESENTATION RUN_ID exclusions.csv` writes `filter.parquet` beside the
representation. It preserves the inherited split and records inclusion, reason,
dataset version and manifest hash. A header-only CSV records a run with all split
tracks included. Use a new run ID for a different filter and retain the same
manifest hash in the representation's `metadata.json`.

The manifest propagates each direct tag to all its ancestors once. `labels.json` fixes
the original top-level columns in genre-ID order. A column is scored only when every
split has positive and negative support among annotated tracks and each evaluation
split has at least two independent positive components. Report track, artist and group
counts to expose sparse support. Keep natural prevalence in validation and
test and report macro-F1, per-genre recall and mAP alongside micro-F1. Compare to the
train-frequency and all-negative baselines. Macro averaging gives each scored genre
equal weight, while any reweighting or resampling of tracks happens within training.
The allocator uses per-label counts on grouped units, following the proportional
objective of [iterative multilabel stratification](https://doi.org/10.1007/978-3-642-23808-6_10).

Representations are stored under `artifacts/<representation>/<run_id>/` as
`vectors.npy` with shape `(n_tracks, d)`, aligned `track_ids.npy` and `metadata.json`.
Metadata records the manifest and configuration hashes, extractor/checkpoint version,
sample rate, window selection, aggregation and normalisation. Export fitted scalers
and class mappings alongside the model. Query processing uses the same settings.

The initial full profile samples three deterministic 10-second windows per track,
covering its beginning, middle and end; the quick profile samples one centred window.
`music_atlas.loader` applies this to every method: windows overlap in short tracks, and a
track shorter than one window gets a single zero-padded window. Full catalogue coverage
does not imply processing every second of each track.

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
