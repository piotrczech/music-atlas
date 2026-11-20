# Stage 2 · Representations · weeks 3–4

Build the three representations the project compares (acoustic features, log-mel
spectrograms for the CNN, pretrained embeddings), the first genre baseline and a shared way
to run and record experiments. Task descriptions use frontmatter labels for stage, priority,
size, package and owner.

## What stage 1 provides

- **Data:** FMA-full in 156 verified tar shards on PD, audited (106,279 valid tracks of
  106,574). The data notebook `notebooks/01-data-exploration.ipynb` explains the catalogue,
  the audit and the split; section 4 shows how to read a partition and its audio.
- **Split:** the primary benchmark `fma-cap10-v1` (85,023 / 10,628 / 10,628 tracks, 16 scored
  top-level genres) under `$ATLAS_DATA_DIR/ready/fma-cap10-v1/`. Code reads the version from
  `[dataset] split` in `configs/*.toml`.
- **Loader:** `music_atlas.loader.load_windows` reads windows straight from a tar member;
  the quick profile takes one centred 10 s window at 22,050 Hz, the full profile three.
- **Quick sample:** 128 training tracks (128 artists, all 16 genres) in
  `samples/fma-cap10-v1/`, copied locally as the README shows (*Quick sample first, full data
  second*). `atlas quick-sample` checks that it loads.
- **Throughput:** opening a shard takes about 4 s on Lustre and one quick window about 0.06 s,
  so catalogue jobs go shard by shard; reading every track once in the quick profile is about
  2 CPU-hours.

## Working rules for every task

1. Develop on the quick sample locally, then run a short quick job on WCSS, then the full
   profile on WCSS sized from the measured throughput.
2. Process every track of the three partitions: 106,279 tracks, exactly the audited valid
   catalogue (the 295 broken files are already out, and the quick sample loads 128 of 128).
   Write representations as `artifacts/<representation>/<run_id>/` with `vectors.npy`,
   `track_ids.npy` and `metadata.json` (`docs/EXPERIMENTS.md`). If a track still fails in one
   representation, which should be rare, record it with `atlas fma-record-filter`; the split
   never changes.
3. Record runs with `music_atlas.tracking.track()` and a `--track` flag (E2-01; the
   `atlas-tracking` skill). Each task's *Tracking* section says what it logs where. On WCSS,
   add the `tracking` extra to your environment and the MLflow lines of
   `scripts/slurm/atlas-env.example.sh` to `~/.atlas-env.sh` once.
4. Fit any preprocessing on train only; select on validation; leave test untouched.

## E2. Representations · weeks 3–4

| Task | Package | Priority | Size | Needs |
|---|---|---|---|---|
| [[E2-01] Experiment runner and MLflow integration](E2-01.md) | lider | wyzszy | wieksze | — |
| [[E2-02] Catalogue-wide acoustic feature extraction](E2-02.md) | a | wyzszy | wieksze | E2-01 conventions |
| [[E2-06] Pretrained encoder selection and embedding extraction](E2-06.md) | c | wyzszy | wieksze | E2-01 conventions |
| [[E2-03] Feature analysis and initial genre baseline](E2-03.md) | a | sredni | male | E2-02 |
| [[E2-04] Log-mel spectrogram processing](E2-04.md) | b | sredni | wieksze | — |
| [[E2-05] Window length and preprocessing study](E2-05.md) | b | niski | male | E2-04 transform |
| [[E2-07] Representation maps and cluster exploration](E2-07.md) | c | niski | male | E2-02, E2-06 |

Priorities set the order of work:

- **wyzszy:** E2-01 fixes the run naming and logging every other task uses, so a minimal
  version lands first. E2-02 and E2-06 are the long catalogue-wide extractions (CPU and GPU)
  that E2-03, E2-07 and stage 3 wait for, so they start as soon as the quick pilot works.
- **sredni:** E2-03 turns the features into the first genre baseline. E2-04 fixes the
  log-mel transform the stage 3 CNN needs; computed on demand, it needs no catalogue-wide run.
- **niski:** E2-05 and E2-07 are studies that inform later choices without blocking anything.

## Completed: E1. Data foundations · weeks 1–2

[E1-01](E1-01.md) repository and WCSS setup · [E1-02](E1-02.md) FMA-full download ·
[E1-03](E1-03.md) audio, annotation and licence audit · [E1-04](E1-04.md) metadata
exploration · [E1-05](E1-05.md) versioned splits · [E1-06](E1-06.md) shared audio loader ·
[E1-07](E1-07.md) quick profile and sample.
