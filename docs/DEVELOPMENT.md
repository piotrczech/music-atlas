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
dataset files, environments and caches belong on the music service's project storage
(PD), shared by the whole team. Every team member must be a member of that service;
an account with several services bills jobs to its default service unless told otherwise.
`PD-info` lists the services and their SLURM accounts.

Run the setup once per person on the login node `ui.wcss.pl`:

```bash
git clone https://github.com/piotrczech/music-atlas.git ~/music-atlas
cd ~/music-atlas
cp scripts/slurm/atlas-env.example.sh ~/.atlas-env.sh  # then set ATLAS_ACCOUNT in it
source ~/.atlas-env.sh
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
mkdir -p "$ATLAS_STORAGE"/{data,artifacts,outputs}
ln -s "$ATLAS_DATA_DIR" data
ln -s "$ATLAS_ARTIFACTS_DIR" artifacts
ln -s "$ATLAS_OUTPUTS_DIR" outputs
uv sync --locked
```

`~/.atlas-env.sh` stays outside Git. It sets `SBATCH_ACCOUNT`, so `sbatch` bills the
music service without `-A`; `PYTHONNOUSERSITE=1`, so packages in `~/.local` cannot
shadow the environment; and `umask 0002`, so files written on PD stay writable for the
team. Source it after every login. The symlinks keep local paths valid and send job
logs (`outputs/slurm-*.log`) to PD. `uv sync` takes seconds and does not compute, so the
login node is fine for it; everything else runs as a job.

Check the setup with the CPU smoke job, which writes a synthetic audio analysis:

```bash
sbatch --partition=bem2-cpu-short scripts/slurm/smoke.sbatch
squeue -u $USER
sacct -j JOB_ID -o JobID,Account,State,Elapsed,MaxRSS
cat "$ATLAS_OUTPUTS_DIR"/smoke-JOB_ID.json
```

Compute jobs use `--frozen --no-sync` with the prepared environment. The grant's CPU
and GPU hours are shared and do not refill: request few cores and short `--time`, and
cancel what you do not need. Confirm the GPU
type, driver stack and compatible PyTorch build before configuring training jobs;
the `training` extra is not a cluster-specific GPU installation profile.

The FMA-full archive is approximately 879 GiB. `scripts/slurm/fma-download.sbatch` runs
`atlas fma-download`, which streams it without storing the archive and writes one verified
tar of source MP3s per FMA folder (156 shards, up to 11 GiB, 930 GiB of the 2,000 GiB
quota), then the inventory. Its summary lists archived files absent from `checksums` as
`unexpected`. Reruns skip finished shards; `--probe` reports sizes and `--limit 2` runs a
pilot. Lustre suits few large files, so jobs read MP3s straight from the shards; Bem2's
`TMPDIR` is Lustre too, so unpack only to node-local storage such as Lem's NVMe.
Bound spectrogram caches and use sharded outputs.
Job-local `TMPDIR` is temporary and must not hold the only copy of results.

### Notebooks on Open OnDemand

[ood.e-science.pl](https://ood.e-science.pl) opens a shell, a file browser over home and
PD, the job list and JupyterLab, through the e-science login. Use notebooks to inspect
manifests, shards and results; run processing and training as `sbatch` jobs. Register
the project environment as a kernel once, after `uv sync`:

```bash
source ~/.atlas-env.sh && cd ~/music-atlas
uv sync --locked --extra notebook --extra research
uv run --frozen --no-sync python -m ipykernel install --user \
  --name music-atlas --display-name "Music Atlas" \
  --env ATLAS_DATA_DIR "$ATLAS_DATA_DIR" --env ATLAS_ARTIFACTS_DIR "$ATLAS_ARTIFACTS_DIR" \
  --env ATLAS_OUTPUTS_DIR "$ATLAS_OUTPUTS_DIR" --env HF_HOME "$HF_HOME" \
  --env PYTHONNOUSERSITE 1
```

Start JupyterLab from Interactive Apps, choose the music service and one or two
cores, and pick the "Music Atlas" kernel. A session is a SLURM job billed for as long
as it stays open, idle or not: delete it from My Interactive Sessions when finished.
Keep notebooks in your clone and commit only the ones that reproduce a result.

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
