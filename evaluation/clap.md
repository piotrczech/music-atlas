# Frozen audio embeddings

## Encoder selection

Select **General CLAP** (`clap-general`), `laion/clap-htsat-unfused` at revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`. It led both neighbour metrics on the
identical 128-track training sample and used the least GPU memory. Music CLAP had the
highest throughput and lists FMA as a training source. Exact recording overlap with
the benchmark remains uncertain. The three-shard benchmark is WCSS job `6451500`.

## Dataset

All candidates use Music Atlas's artist-disjoint FMA split selected by `[dataset] split`. A quick run reads
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

### MusicFM-MSD

The additional candidate uses [the authors' MusicFM-MSD release](https://huggingface.co/minzwon/MusicFM/tree/4513b38bc25ad1d227b1980819b9691ba97f4d87)
(`pretrained_msd.pt` and `msd_stats.json`, MIT). The authors list Million Song Dataset
for these weights and FMA-large for their separate MusicFM-FMA release. There is no
recording-level overlap check between the Million Song Dataset audio and our FMA split.
[MusicFM checkpoint and inference notes](https://github.com/minzwon/musicfm#quick-start).

MusicFM reads the same 10-second intervals as CLAP. The shared loader supplies 24 kHz
mono audio and zero padding. MusicFM uses its saved mel filters and MSD normalization
statistics. We average the 25 Hz frames from hidden-state index 7 into one 1,024-component
vector per window and apply the same window-to-track aggregation as CLAP. The adapter
reuses the Transformers Conformer and records the pinned architecture configuration
revision and weights checksum. It loads the release tensors on CPU before moving the
inference model to the chosen device. WCSS CPU job `6448847` verified the release
checkpoint SHA-256 and encoded two synthetic signals. It completed with exit code
`0:0` in 2 min 12 s. Hopper job `6448858` then ran MusicFM-MSD on the FMA quick sample.

## Measurements on WCSS

WCSS CPU job `6441334` resolved both exact revisions on Hugging Face and fetched their weights
into the shared cache. Fetching the three required files took 7.00 seconds for the general
model and 8.36 seconds for the music model.

CPU job `6443546` ran the earlier extractor with zero padding. The same 128 training
tracks and batch size of 32 gave the following results on four CPU cores.

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

### Hopper pilots

CLAP job `6446080` completed with exit code `0:0` in 2 min 49 s. MusicFM-MSD job
`6448858` completed with exit code `0:0` in 2 min 12 s. Each run used a batch size of
32 and one centred 10-second window per track. The SHA-256 sums of `track_ids.npy` and
`windows.parquet` match for all three runs. Their manifest and input-track hashes also
match. MLflow recorded the CLAP checkout as `267c036-dirty`. The recorded extraction
source matches `53783ae`. The loader, representation and shard reader hashes match
`4ba4ab5`, the revision used for the MusicFM-MSD pilot.

| Encoder | Tracks | Failed | Precision at 5 | Genre vote macro AP | Tracks/s | Peak allocated GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| General CLAP | 128 | 0 | 49.69% | 0.2755 | 20.02 | 974.2 |
| Music CLAP | 128 | 0 | 30.63% | 0.1662 | 30.36 | 1399.2 |
| MusicFM-MSD | 128 | 0 | 37.81% | 0.2146 | 19.55 | 9622.9 |

| Encoder | Run | MLflow run ID |
|---|---|---|
| General CLAP | `embeddings-quick-20261126T130219Z-54f6` | `138b08a2c7c9498b9037c5a7887adf63` |
| Music CLAP | `embeddings-quick-20261126T130236Z-c92f` | `a6abaff000714c2ba93e7ebf1bdbe7f1` |
| MusicFM-MSD | `embeddings-quick-20261127T005600Z-3532` | `0161f18436304517ae74dd95c6aa4308` |

The random-neighbour precision was 27.66% and the constant-score macro AP was 0.1230 for
the common sample. General CLAP retrieved a Rock neighbour for Rock/Folk track 15552
and a Hip-Hop neighbour for Hip-Hop track 19726. Music CLAP retrieved the same
Electronic/Experimental track for the Blues and Rock/Folk queries and found a
Classical neighbour for Classical track 30130. MusicFM-MSD found Hip-Hop and Classical
neighbours for tracks 19726 and 30130, respectively, and shared General CLAP's closest
neighbour for the Blues query 12099. All three runs produced 128 finite, nonzero window
vectors. The examples and scores describe retrieval on the training sample.

### Three-shard benchmark

Job `6451500` completed with exit code `0:0` in 9 min 38 s on one H100. Each batch
size encoded all 1,642 tracks from `026.tar`, `078.tar` and `130.tar` with three windows
per track and zero failures. The output track IDs and window positions match byte for byte.

| Windows per batch | Tracks/s | Peak allocated GPU MiB | Peak reserved GPU MiB | Estimated GPU hours | Suggested wall time |
|---:|---:|---:|---:|---:|---:|
| 16 | 5.79 | 558.8 | 652.0 | 5.10 | 06:43:00 |
| 32 | 7.40 | 974.2 | 1110.0 | 3.99 | 05:17:00 |

| Windows per batch | Run | MLflow run ID |
|---:|---|---|
| 16 | `embeddings-full-20261127T113302Z-c1f1` | `95b22cfce5fd4d93a02c34cc9f91d13b` |
| 32 | `embeddings-full-20261127T113757Z-a71a` | `5aa1a6d34a3e4b01b8c34f0ebf27133a` |

Batch 32 ran second, so warm filesystem caches could contribute to its measured gain.
It had ample memory headroom. The production run uses batch 32, three windows per track
and the suggested wall time of 5 hr 17 min. The projected PD footprint is 1.43 GiB for
track vectors and IDs, window files and resumable shard parts combined. Full extraction
was submitted as WCSS job `6453884` from checkout `4ba4ab5`.

### Full extraction attempt

Job `6453884` ended with status `FAILED` and exit code `1:0` after 4 hr 48 min 43 s.
Run `embeddings-full-20261128T014136Z-06a4` (MLflow `f252aa66eee649d792dc30b45b7ae1b4`)
saved 106,277 512-dimensional track vectors and 317,791 window vectors. The maximum
absolute track norm error was `1.19e-7`.

| Partition | Expected | Encoded |
|---|---:|---:|
| Train | 85,023 | 85,021 |
| Validation | 10,628 | 10,628 |
| Test | 10,628 | 10,628 |

Both failures were in training. libsndfile reported `Format not recognised` for tracks
`40546` and `92123`. All 156 shard part files remain on PD. The attempt's
`evaluation/coverage.json`, checksums, and `failed.csv` are under
`$ATLAS_ARTIFACTS_DIR/clap-general/embeddings-full-20261128T014136Z-06a4/`.
Complete coverage requires decoding both failed tracks.

### Decoder fix and recovery

CPU probe `6459520` reproduced both failures. In each file, the first strict MPEG
header candidate after the ID3 tag led to an unrecognised-format error. Starting the
decoder at the earlier MPEG sync candidate restored the decoded duration recorded in
the FMA audit. The shared loader now tries that earlier offset after the strict
candidate fails with this error. Successful reads follow their existing path.

CPU validation job `6459585` verified the source MP3s against the inventory SHA-1,
decoded all three windows from each recovered track, and matched their final endpoints
to the audited durations. All 128 quick-sample windows matched the previous loader
bit for bit at 48 kHz. GPU recovery job `6459668` uses a separate run directory and
the same encoder, manifest, batch size and window settings. It reuses the 154
successful shard files and re-encodes shards `040` and `092`. The original failed
run remains available in MLflow and on PD.

### Verified full embeddings

Recovery job `6459668` completed with exit code `0:0` in 7 min 47 s. The completed
run is `embeddings-full-20261128T131619Z-7a2b`, with MLflow ID
`0dcb7f354cc14448bbc11c8ad30cc700`. It uses General CLAP at revision
`8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`, split
`fma-cap10-v1-46a89930d45ea6fb`, and manifest SHA-256
`28fac223066f19c8a04a11322c257a019ae998ee975856c40a7441f189c2aaa1`.

| Partition | Expected | Encoded |
|---|---:|---:|
| Train | 85,023 | 85,023 |
| Validation | 10,628 | 10,628 |
| Test | 10,628 | 10,628 |
| **Total** | **106,279** | **106,279** |

The run has zero failures, 106,279 track vectors and 317,797 window vectors, each
with 512 float32 components. The maximum absolute track norm error is `1.19e-7`.
The recovery job independently verified the final file sizes and all four SHA-256
sums against `evaluation/coverage.json`. It also verified byte-for-byte equality
for the 154 reused shard files. `evaluation/recovery.json` records the loader
hashes and links to the failed attempt.

```text
/lustre/pd03/hpc-piotrczech-1792095018/music-atlas/artifacts/clap-general/embeddings-full-20261128T131619Z-7a2b/
```

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

The paired `pilot` command runs the two CLAP checkpoints. To prepare MusicFM separately,
request CPU resources and wait for the job to finish before submitting the MusicFM pilot.
The CPU step checks the official checkpoint SHA-256 and tests inference on synthetic
audio. The GPU step reads the cached files.

```bash
cpu_args=(-A "$ATLAS_ACCOUNT" -p lem-cpu-short --cpus-per-task=2 --mem=12G
          --gres=storage:local:50g --time=00:20:00)
bash .agents/skills/wcss-slurm/scripts/preflight.sh \
  scripts/slurm/embeddings.sbatch "${cpu_args[@]}"
sbatch --parsable "${cpu_args[@]}" scripts/slurm/embeddings.sbatch prepare-musicfm
```

Once preparation returns `COMPLETED` with exit code `0:0`, run on the same Hopper
type, quick profile and batch size as the paired CLAP pilot:

```bash
bash .agents/skills/wcss-slurm/scripts/preflight.sh \
  scripts/slurm/embeddings.sbatch -A "$ATLAS_ACCOUNT"
sbatch --parsable -A "$ATLAS_ACCOUNT" scripts/slurm/embeddings.sbatch \
  --encoder musicfm-msd --profile quick --batch-size 32
```

Each quick run writes `evaluation/pilot.json` and `evaluation/performance.json` under
`$ATLAS_ARTIFACTS_DIR/<encoder>/<run_id>/` and logs them to MLflow. Inspect the three runs
together in the MLflow `embeddings` experiment. Confirm that each run encoded 128 tracks
with zero failures and compare the SHA-256 sums of `track_ids.npy` and `windows.parquet`
across all three runs. The paired CLAP pilot also attaches `evaluation/comparison.json`
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
bash .agents/skills/wcss-slurm/scripts/preflight.sh scripts/slurm/embeddings.sbatch \
  -A "$ATLAS_ACCOUNT" --time="$BENCH_WALLTIME"
sbatch -A "$ATLAS_ACCOUNT" --time="$BENCH_WALLTIME" \
  scripts/slurm/embeddings.sbatch benchmark --encoder "$ENCODER" --windows 3
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
In `evaluation/coverage.json`, verify that each partition's encoded count equals its
expected count and that the failure count is zero. The report records the SHA-256 sums of
the four final representation files.
`metadata.json` records the manifest and configuration hashes, pinned weights, window rule,
aggregation, software and dimensions. Both the frozen classifier and cosine retrieval can
read the same arrays with `np.load(path, mmap_mode="r")`. Only a new audio query needs encoding.
The expected track vector payload for 106,279 rows is 207.6 MiB for CLAP or 415.2 MiB for
MusicFM-MSD, plus IDs. The benchmark also estimates the size of the window arrays and the
per-shard recovery files.

Use `scripts/mlflow-ui.sh --ood` from a compute allocation or
`sbatch scripts/slurm/mlflow-ui.sbatch` for the OOD link. The existing
`scripts/mlflow-ui.sh --wcss` command mirrors the MLflow store for local inspection.
