![Music Atlas demo](docs/assets/demo.png)

# Music Atlas

Music Atlas explores how different representations of audio capture musical structure.
The project combines genre classification, exploratory analysis and similarity search
to study what can be learned from a large collection of music recordings. Its central
question is how the choice of representation changes the patterns we discover, the
genres we recognise and the tracks we consider similar.

**Start here:** [`notebooks/01-data-exploration.ipynb`](notebooks/01-data-exploration.ipynb)
introduces the data, what is wrong with it, how we clean it and how we split it.

## Research approach

We compare three approaches on a shared dataset and evaluation protocol:

| Representation | Classification | Analysis and retrieval |
|---|---|---|
| Acoustic features: MFCCs, chroma, rhythm and spectral statistics | Logistic regression and a tree-based model | Feature interpretation, clustering and nearest neighbours |
| Log-mel spectrograms | A compact CNN trained from scratch | Learned features and similarities between recordings |
| Embeddings from a frozen pretrained audio encoder | A linear classification head | Embedding maps, clustering and sounds-like search |

All three use one artist-disjoint train/validation/test split and the same deterministic audio
windows. Classification is compared with macro-F1, micro-F1 and mAP; retrieval with
precision@10 and nDCG@10, alongside listening examples and computational cost. These are
planned experiments, not completed model results.

## Data and computation

The project uses [FMA-full](https://github.com/mdeff/fma): 106,574 full-length tracks in a
hierarchy of 163 genres under 16 top-level genres. An audio audit excludes the 0.3% of files
that are broken, and the primary benchmark `fma-cap10-v1` limits any one artist to 10% of a
genre's validation or test tracks. The [experimental protocol](docs/EXPERIMENTS.md) defines
the rules and formats.

Processing and training run on [WCSS](https://man.e-science.pl/pl/kdm/bem2) with
**4,000 CPU hours, 300 GPU hours and 2 TB of storage**. Local development uses uv and a
128-track sample; the same code runs on WCSS with the full profile. Data, caches and artifacts
live outside Git, and MLflow will record configurations, metrics and artifacts.

## Application

A Next.js interface backed by FastAPI will bring together dataset exploration, audio
inspection, genre predictions and nearest-neighbour search, comparing representations for
the same recording. It currently supports audio upload, playback, waveform and power-spectrum
plots and basic acoustic measurements. The final application runs through Docker Compose with
exported models and indexes.

## Development

Python 3.12, [uv](https://docs.astral.sh/uv/) and Node.js 22+ are required.

```bash
uv sync --locked
uv run uvicorn music_atlas.api:app --reload
```

In a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open http://localhost:3000; API documentation is at http://localhost:8000/docs.

### Commands

The data commands need `uv sync --locked --extra research` and run on WCSS as jobs through
`scripts/slurm/*.sbatch`, which pass extra arguments on.

| Command | What it does |
|---|---|
| `atlas doctor --profile quick` | Show the resolved `ATLAS_*` paths and a profile |
| `atlas demo`, `atlas analyze FILE` | Write a synthetic test tone; analyse up to 30 s of an audio file |
| `atlas fma-download [--probe] [--limit N]` | Stream FMA-full into verified tar shards and write the inventory |
| `atlas fma-audit [--limit N]` | Decode every track and write the audio audit |
| `atlas fma-split --audit FILE [--max-artist-share 0.1] [--strategy rare-support] [--publish]` | Build a versioned train/validation/test split; `--publish` makes it the shared manifest |
| `atlas fma-record-filter VERSION REPRESENTATION RUN_ID exclusions.csv` | Record the tracks a representation run could not use |
| `uv run ruff check .`, `uv run pytest` | Lint and test; `npm run typecheck` and `npm run build` in `web/` |

- [Development guide](docs/DEVELOPMENT.md): layout, WCSS, data pipeline, notebooks, MLflow and Docker
- [Experimental protocol](docs/EXPERIMENTS.md): data, splits, metrics and representations
- [Tasks](tasks/README.md)
