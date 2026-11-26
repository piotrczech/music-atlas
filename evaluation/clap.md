# Frozen CLAP embeddings

## Provisional encoder selection

The leading candidate is **General CLAP**, `laion/clap-htsat-unfused`, at revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`. In WCSS CPU job `6443546`, an average of
49.7% of its five nearest neighbours shared a genre label with the query, compared with
30.3% for Music CLAP. Both candidates used the same 128 training tracks in the earlier
extractor. The music checkpoint also has a clearer risk of exposure to FMA because the
published LAION music collection lists FMA. Track-level overlap remains unknown for both
checkpoints.

As of 26 November 2026 at 12:00 CET, the paired Hopper pilot (`6446080`) is pending
in the WCSS queue. It will compare both checkpoints with the integrated extractor, measure
throughput and peak memory, and provide the evidence for the final choice. The three-shard
benchmark and full catalogue extraction are pending.

## Dataset

Both candidates use Music Atlas's artist-disjoint FMA split selected by `[dataset] split`. A quick run reads
the same 128 training tracks from the existing sample tar. A full run reads the three
partitions together, opens each source tar once per attempt and visits each assigned track,
including tracks without genre labels. The audited manifest used for preparation contains
85,023 training, 10,628 validation and 10,628 test tracks (106,279 in total).

## Checkpoints and training data

| CLI name | Public Hugging Face weights | Training sources and overlap risk |
|---|---|---|
| `clap-general` | [LAION unfused HTS-AT, pinned revision](https://huggingface.co/laion/clap-htsat-unfused/tree/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a) | The card cites the original CLAP paper and LAION-Audio-630K. The authors report AudioSet in the general model family. Reposts can put the same recording in multiple source collections. |
| `clap-music` | [LAION larger music, pinned revision](https://huggingface.co/laion/larger_clap_music/tree/a0b4534a14f58e20944452dff00a22a06ce629d1) | LAION describes training the music checkpoint on its music collection, AudioSet and LAION-Audio-630K. Its collection inventory explicitly marks Free Music Archive as processed. This creates an additional direct route for benchmark recordings to enter pretraining. |

The two Hugging Face cards declare Apache-2.0 for the distributed models. The upstream
code repository displays CC0. Individual recordings retain their source licences. Both exact
revisions resolved publicly on WCSS, including the model configuration, feature extractor
configuration and PyTorch weights. The music checkpoint has a larger HTS-AT backbone. Both
project audio into 512 dimensions. [General model card](https://huggingface.co/laion/clap-htsat-unfused)
and [music model card](https://huggingface.co/laion/larger_clap_music).

The published training-source table and the release index identify the component collections.
The index links URL lists for AudioStock and Epidemic Sound, a Freesound release with its
own identifiers and a withdrawn BBC list. Other source collections are unavailable. AudioSet
lists YouTube IDs and segment times. FMA track IDs and MP3 checksums cannot identify a
re-upload under another URL or encoding. The music inventory links the original FMA corpus,
but does not specify which individual tracks entered the pinned weights. No checkpoint-specific
list establishes the intersection with our test partition. Any score using either encoder must
carry this limitation, especially for the music model. An artist-disjoint downstream split
controls leakage within FMA and cannot undo exposure during pretraining.
[CLAP paper, Appendix D](https://arxiv.org/html/2211.06687),
[upstream checkpoint recipes](https://github.com/LAION-AI/CLAP#pretrained-models),
[music collection inventory](https://github.com/LAION-AI/audio-dataset/blob/main/data_collection/README.md#music-dataset),
[630K URL-list release](https://github.com/LAION-AI/audio-dataset/tree/main/laion-audio-630k)
and [AudioSet segment lists](https://research.google.com/audioset/download.html).

WCSS job `6443535` found no explicit FMA references in the released AudioStock (10k) and
Epidemic Sound lists or the eight published CLAP experiment scripts. Job `6443572` parsed
555,644 records in the Freesound licence export. It found 19 descriptions mentioning FMA.
Their URLs matched 14 artist pages and three album pages in FMA metadata. Exact track URLs
and normalized uploader/title pairs produced zero matches. These references indicate a
possible route through Freesound. The export does not identify the recordings used to train
a particular checkpoint. Renamed or re-encoded recordings can also evade these checks.
The extent of overlap remains unknown.

## Measurements on WCSS

WCSS CPU job `6441334` resolved both exact revisions on Hugging Face and fetched their weights
into the shared cache. Fetching the three required files took 7.00 seconds for the general
model and 8.36 seconds for the music model.

CPU job `6443546` ran the earlier extractor with zero padding. The same 128 training
tracks and batch size of 32 gave the following results on four CPU cores. The GPU
comparison will use the integrated extractor and the checkpoints' short-audio padding.

| Measurement | General CLAP | Music CLAP |
|---|---:|---:|
| Successfully encoded tracks | 128 / 128 | 128 / 128 |
| Extraction time | 18.51 s | 26.44 s |
| Process peak RAM | 2,034 MiB | 2,397 MiB |
| Fraction of five neighbours sharing a genre | 49.69% | 30.31% |
| Macro average precision of neighbour genre votes | 0.275 | 0.165 |
| Median cosine, shared genre / no shared tag | 0.467 / 0.392 | 0.951 / 0.951 |
| Pair cosine, 5th to 95th percentile | 0.107–0.708 | 0.837–0.989 |
| Raw window L2 norm, minimum / median / maximum | 3.57 / 4.63 / 5.75 | 0.214 / 0.413 / 0.484 |

Neither run returned zero, NaN or infinite window vectors. The music model assigned high
similarity to most pairs, even across genre tags. For example, its closest match was the
same electronic/experimental recording (99235) for both a blues track (12099) and a
rock/folk track (15552). General CLAP retrieved a rock-tagged recording for 15552 and
another hip-hop recording for 19726. Music CLAP retrieved a classical-tagged neighbour for
30130, whose closest neighbour under general CLAP was tagged international. These are label
comparisons on training recordings. The Hopper pilot measures GPU performance separately.

WCSS CPU job `6443532` tested the earlier extractor with each real checkpoint on four training recordings
from FMA shards 012 and 014. It killed the extraction process with SIGKILL after its first
shard, then resumed it. Both runs retained the first shard byte for byte and produced the
same ordered IDs and vectors as uninterrupted extraction (maximum absolute error 0.0).
Each resumed run has an `evaluation/recovery.json` report linked to its earlier MLflow attempt.

## From audio to a reusable vector

The shared loader selects intervals in the original sample coordinates. It returns one
centred window in the quick profile and up to three in the full profile by default. For CLAP
it mixes to mono and resamples to 48 kHz. Before feature extraction, short windows use
`repeatpad`: repeat the unpadded audio as many whole times as fit and zero-pad the remainder
to 480,000 samples. This preserves the selected time interval. Both
checkpoint processors use `rand_trunc` for overlong input. The extractor requires the exact
window size, so no random crop or shift is possible.

The batched log-mel calculation uses the checkpoint's Slaney filters, Hann window of
1,024 samples, hop of 480, reflected edges, power spectrum and decibel scale. On startup
the extractor compares its output with the pinned Transformers processor on the same probe.
Inference uses the audio projection in evaluation mode. Raw vectors are kept in
`window_vectors.npy` with matching rows in `windows.parquet`. For aggregation, divide each
window vector by its L2 norm, average the resulting vectors within each track and normalize
that mean to unit length. A zero or non-finite result records the track ID and reason in
`failed.csv`.
[Reference feature extractor](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/clap/feature_extraction_clap.py).

## Run on WCSS

Source the project environment and run from a checkout containing these changes. The GPU
script requests one Hopper on Lem, uses node-local scratch for the virtual environment and
keeps weights in `$HF_HOME`. Each candidate runs as a separate process on the same GPU, so
peak memory counters start afresh. Use the music service account and preflight each new
SLURM allocation.

```bash
source ~/.atlas-env.sh
bash .agents/skills/wcss-slurm/scripts/preflight.sh scripts/slurm/embeddings.sbatch -A "$ATLAS_ACCOUNT"
sbatch -A "$ATLAS_ACCOUNT" scripts/slurm/embeddings.sbatch pilot --batch-size 32
```

Each quick run writes `evaluation/pilot.json` and `evaluation/performance.json` under
`$ATLAS_ARTIFACTS_DIR/<encoder>/<run_id>/` and logs them to MLflow. Inspect the two runs
together in the MLflow `embeddings` experiment. The paired pilot also attaches `evaluation/comparison.json`
with both candidates to each run. The report gives raw window norms, cosine similarity
percentiles, counts of invalid projections and the nearest neighbours of selected example
tracks. A pair shares a genre when at least one scored root tag matches. The comparison uses
other artists and excludes members of the same split group on WCSS. Precision at five is the
fraction of the five neighbours sharing a tag, averaged over queries. The genre diagnostic
uses the fraction of neighbours carrying each tag as its score, then averages precision across
the ranked examples for each supported genre (macro average precision). The report compares
these scores with choosing an eligible neighbour uniformly at random and assigning every
track the same score for a given genre. Both baselines reflect this sample's tag frequencies.
Missing FMA tags make apparent mismatches uncertain.

After reviewing coverage, neighbour behaviour, overlap risk and measured cost, set
`ENCODER` to the chosen CLI name. Run the baseline benchmark on three evenly spaced source
shards. The benchmark compares batches of 16 and 32 windows in separate processes on the same
GPU. Each writes to its own run directory using the full window configuration.
Supply `--windows 5` to benchmark five windows. In that variant, windows 0, 2 and 4 are
the usual three, so the saved raw vectors allow quality comparisons without re-encoding.

```bash
sbatch -A "$ATLAS_ACCOUNT" --time="$BENCH_WALLTIME" \
  scripts/slurm/embeddings.sbatch benchmark --encoder "$ENCODER"
```

Use the measured track rate and the assigned track count to calculate GPU hours
(`tracks / tracks_per_second / 3600` on one GPU). Allow time for setup, final assembly and
variation across shards. Each benchmark writes `evaluation/benchmark.json` with its shard
names, measured sample size, expected file sizes and suggested wall time (30% headroom
plus five minutes). Set `WALLTIME` and `BATCH_SIZE` from the benchmark before submitting.

```bash
bash .agents/skills/wcss-slurm/scripts/preflight.sh scripts/slurm/embeddings.sbatch \
  -A "$ATLAS_ACCOUNT" --time="$WALLTIME"
sbatch -A "$ATLAS_ACCOUNT" --time="$WALLTIME" scripts/slurm/embeddings.sbatch \
  --encoder "$ENCODER" --profile full --batch-size "$BATCH_SIZE"
```

The job prints its run ID at startup. To resume, use the same checkout, checkpoint,
profile, window count, benchmark shard count and batch size. Only one process can own a run. A hard kill
releases the OS lock and leaves the attempt ID on disk. A resumed attempt links to it in
MLflow, checks the extraction configuration and exact input track list, then reuses complete
shards. A partially written shard or a shard containing a failed track is processed again.

```bash
sbatch -A "$ATLAS_ACCOUNT" --time="$WALLTIME" scripts/slurm/embeddings.sbatch \
  --encoder "$ENCODER" --profile full --batch-size "$BATCH_SIZE" --resume "$RUN_ID"
```

A complete run reopens the final arrays and checks sorted unique IDs, partition coverage,
shape, float32 values and unit norms. Row `i` of `vectors.npy` describes `track_ids.npy[i]`.
`metadata.json` records the manifest and configuration hashes, pinned weights, window rule,
aggregation, software and dimensions. Both the frozen classifier and cosine retrieval can
read the same arrays with `np.load(path, mmap_mode="r")`. Only a new audio query needs encoding.
The expected track vector payload for 106,279 rows is 207.6 MiB plus IDs. The benchmark
also estimates the size of the window arrays and the per-shard recovery files.

Use `scripts/mlflow-ui.sh --ood` from a compute allocation or
`sbatch scripts/slurm/mlflow-ui.sbatch` for the OOD link. The existing
`scripts/mlflow-ui.sh --wcss` command mirrors the MLflow store for local inspection.
