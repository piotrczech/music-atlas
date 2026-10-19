# Development

## Repository layout

`src/music_atlas/` contains shared audio processing, configuration, CLI and API code.
`web/` contains the Next.js application. Profiles live in `configs/`, cluster entry
points in `scripts/slurm/`, and the roadmap in `tasks/`. New processing and model
modules belong in the Python package as their tasks are implemented.

```bash
uv run atlas doctor --profile quick
uv run atlas demo
uv run atlas analyze outputs/demo.wav
```

`demo` produces a synthetic 440 Hz signal for checking the processing path.
The inspection endpoint reads up to the first 30 seconds of an uploaded file, with
a 32 MiB upload limit. Research windowing and artifact formats are defined in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Local configuration

`configs/quick.toml` defines a 128-track development sample; `configs/full.toml`
selects the full validated manifest. Dataset selection and batch processing are
implemented through the data and representation tasks.

`ATLAS_DATA_DIR`, `ATLAS_ARTIFACTS_DIR` and `ATLAS_OUTPUTS_DIR` override the default
`data/`, `artifacts/` and `outputs/` directories. Export these variables in the shell
or configure them in the process environment; the CLI does not load `.env` files.

Install the research and tracking dependencies when needed:

```bash
uv sync --locked --extra research --extra tracking
mkdir -p outputs artifacts/mlflow
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///outputs/mlflow.db \
  --default-artifact-root ./artifacts/mlflow
```

## WCSS

Clone the repository into the home directory. WCSS home storage is limited to 50 GB;
dataset files, environments and caches belong on the allocated project storage (PD).
Run `PD-info` to obtain the service directory, then substitute it below:

```bash
export ATLAS_STORAGE="/lustre/pd01/YOUR_SERVICE/music-atlas"
mkdir -p "$ATLAS_STORAGE"/{data,artifacts,outputs}
export ATLAS_DATA_DIR="$ATLAS_STORAGE/data"
export ATLAS_ARTIFACTS_DIR="$ATLAS_STORAGE/artifacts"
export ATLAS_OUTPUTS_DIR="$ATLAS_STORAGE/outputs"
export UV_CACHE_DIR="$ATLAS_ARTIFACTS_DIR/uv-cache"
export HF_HOME="$ATLAS_ARTIFACTS_DIR/hf-cache"
export UV_PROJECT_ENVIRONMENT="$ATLAS_ARTIFACTS_DIR/venv-$USER"
```

Keep account-specific exports in a private shell file. Optional repository symlinks
provide access through the same paths used locally; create them only when those
paths do not already contain files:

```bash
ln -s "$ATLAS_DATA_DIR" data
ln -s "$ATLAS_ARTIFACTS_DIR" artifacts
ln -s "$ATLAS_OUTPUTS_DIR" outputs
```

Prepare the environment on an appropriate node before submitting a job. From the
repository root, with the correct account and partition:

```bash
uv sync --locked
mkdir -p outputs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION scripts/slurm/smoke.sbatch
```

The supplied CPU job checks the environment and writes a synthetic audio analysis.
Compute jobs use `--frozen --no-sync` with the prepared environment. Confirm the GPU
type, driver stack and compatible PyTorch build before configuring training jobs;
the `training` extra is not a cluster-specific GPU installation profile.

The FMA-full archive is approximately 879 GiB. Allow for both download and extraction
within the storage quota, remove the archive after verified extraction when appropriate,
and retain compressed source audio. Bound spectrogram caches and use sharded outputs.
Job-local `TMPDIR` is temporary and must not hold the only copy of results.

WCSS references: [home storage](https://man.e-science.pl/i/150),
[project storage and symlinks](https://man.e-science.pl/i/218),
[SLURM](https://man.e-science.pl/pl/kdm/slurm).

## Application and checks

```bash
docker compose up --build
docker compose --profile tracking up --build
```

Compose serves the interface on localhost:3000 and FastAPI on localhost:8000.
The optional tracking profile serves MLflow on localhost:5000. Next.js proxies
`/api/*` to FastAPI; `API_INTERNAL_URL` is resolved at build time. The serving
application consumes exported results and does not require a live tracking server.
Cluster execution and container builds still require validation on the target hosts.

```bash
uv run ruff check .
uv run pytest
cd web
npm run typecheck
npm run build
```

`AGENTS.md` provides shared project instructions for Codex and Cursor; `CLAUDE.md`
imports it for Claude Code. Skills live in `.agents/skills/`, with `.claude/skills`
linking to that directory. Preserve this symlink when cloning or extracting the project.
