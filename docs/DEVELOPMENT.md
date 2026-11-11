# Development

## Repository layout

`src/music_atlas/` contains shared audio processing, configuration, CLI and API code.
`web/` contains the Next.js application. Profiles live in `configs/`, cluster entry
points in `scripts/slurm/`, exploration notebooks in `notebooks/`, and the roadmap in
`tasks/`. New processing and model
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
cancel what you do not need.

GPUs are only on Lem: `lem-gpu-short` (up to 3 days) with `--gres=gpu:hopper:N` and an
explicit `--cpus-per-task` (at most 16 cores per GPU). In November 2026 a node reported an
NVIDIA H100 with 94 GiB, driver 595.112.07 and CUDA 13.2. The locked `training` extra
(PyTorch 2.14.1+cu130 from PyPI) ran on it; sync it into the job's node-local `TMPDIR`
(`--gres=gpu:hopper:1,storage:local:50g`), which took 14 s, rather than onto Lustre. The
site container `pytorch_latest.sif` also works but ships PyTorch 2.2.1.

FMA-full is downloaded once for the team and shared on PD. The archive is approximately 879 GiB. `scripts/slurm/fma-download.sbatch` runs
`atlas fma-download`, which streams it without storing the archive and writes one verified
tar of source MP3s per FMA folder (156 shards, up to 11 GiB, 930 GiB of the 2,000 GiB
quota), then the inventory. Its summary lists archived files absent from `checksums` as
`unexpected`. Reruns skip finished shards; `--probe` reports sizes and `--limit 2` runs a
pilot. The November 2026 pilot on `bem2-cpu-short` fetched about 10.7 MiB/s per shard;
the job downloads one shard per allocated core. SLURM's `MaxRSS` includes the page cache
of the tars being written, so it reaches `--mem` without the job running out of memory.
The full run (8 cores) finished in 3 h 10 min for 25 CPU-hours: 156 shards and 106,574
tracks under `$ATLAS_DATA_DIR/fma/`, with the inventory in `$ATLAS_DATA_DIR/manifests/`.
Lustre suits few large files, so jobs read MP3s straight from the shards; Bem2's
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

Then open Interactive Apps → Servers → **Jupyter Notebook** (the CPU app; the GPU,
TensorFlow and PyTorch variants reserve an H100) and fill in the form:

| Field | Value |
|---|---|
| Account | the music service; the list starts with the account's default service |
| Choose partition | `lem-cpu-interactive` (the `bem2` nodes are often powered down) |
| Duration (hours) | 1–2 |
| Number of CPU cores | 1–2 |
| RAM Memory | 8 GB; memory is not billed |

After Launch the session is Queued, then Running within a minute or two; My Interactive
Sessions shows its job ID, node and remaining time. Connect to Jupyter normally logs in
by itself. If Jupyter asks for a password, it is a one-off session password, not the
WCSS one; copy it from the session's `connection.yml`:

```bash
grep '^password:' ~/ondemand/data/sys/dashboard/batch_connect/sys/*/output/SESSION_ID/connection.yml
```

JupyterLab opens in `$HOME`. Choose the "Music Atlas" kernel and start with
`notebooks/fma_shards.ipynb`, which counts the shards, lists the tracks of one shard and
plays a track through the shared loader. A session is a SLURM job billed for as long as it
stays open, idle or not: Delete it in My Interactive Sessions when finished. The session's
settings and log are in the same `output/SESSION_ID/` directory (`job_script_options.json`,
`output.log`). Keep notebooks in your clone and commit only the ones that reproduce a
result, without outputs.

WCSS references: [home storage](https://man.e-science.pl/i/150),
[project storage and symlinks](https://man.e-science.pl/i/218),
[SLURM](https://man.e-science.pl/pl/kdm/slurm).

### FMA metadata report

Use the metadata and inventory already on PD. The CPU job reads CSV metadata and does
not open any audio shards. Run these commands in the OOD shell from the repository root
after configuring the music service as above.

```bash
source ~/.atlas-env.sh
cd ~/music-atlas
uv sync --locked --extra research --extra notebook
bash .agents/skills/wcss-slurm/scripts/preflight.sh \
  scripts/slurm/fma-metadata.sbatch --account="$ATLAS_ACCOUNT"
```

Once preflight passes, submit the report and check its state.

```bash
submission=$(sbatch --parsable --account="$ATLAS_ACCOUNT" \
  scripts/slurm/fma-metadata.sbatch)
job_id=${submission%%;*}
sacct -j "$job_id" -o JobID,Account,State,Elapsed,MaxRSS
printf 'Job ID: %s\n' "$job_id"
```

After `COMPLETED`, open `notebooks/fma_metadata.ipynb` in OOD, select **Music Atlas**
and set `JOB_ID` to the printed ID. The report in
`$ATLAS_OUTPUTS_DIR/fma-metadata-<job_id>/` contains five figures, the underlying
tables, an interpretation and input hashes. The SLURM log is in `outputs/slurm-<job_id>.log`.

Before the audio audit, the report uses every inventory row marked `present`.
After the audit, pass the canonical manifest to the same job.

```bash
sbatch --parsable --account="$ATLAS_ACCOUNT" scripts/slurm/fma-metadata.sbatch \
  --manifest "$ATLAS_DATA_DIR/manifests/tracks.parquet"
```

The manifest may be CSV or Parquet with unique `track_id` and Boolean `valid` columns.
Only valid IDs present in the inventory enter the report. The summary counts rejected
and unlisted present IDs separately. Both runs use the published durations so their
distributions can be compared on the same basis.

`genre_support.csv` contains the full taxonomy and counts each track once per genre.
Inclusive counts add ancestor labels using `genres.csv`. Co-occurrence uses direct
annotations. Concentration shares use the entire selected catalogue as the denominator,
with missing IDs excluded from entity counts. Read the rare-class examples together with
per-class artist support before fixing the scored labels in E1-05.

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
