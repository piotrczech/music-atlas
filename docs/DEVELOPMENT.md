# Development

## Repository layout

| Path | Contents |
|---|---|
| `src/music_atlas/` | Python package: CLI (`cli.py`), API (`api.py`), download (`fma.py`), audit (`audit.py`), splits (`manifest.py`, `split_views.py`), audio loader (`loader.py`), the shard loop (`shards.py`), log-mel (`logmel.py`, `logmel_cache.py`) and the tables behind the data notebook (`data_report.py`) |
| `web/` | Next.js interface |
| `configs/` | `quick.toml` (128-track sample) and `full.toml` profiles |
| `scripts/slurm/` | WCSS job scripts |
| `notebooks/` | `01-data-exploration.ipynb`: the data, audit and split |
| `tasks/` | Roadmap |

The CLI and API call the same functions. Research windows and file formats are defined in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Local setup

```bash
uv sync --locked --extra research
uv run atlas doctor --profile quick
uv run atlas demo                      # synthetic 440 Hz signal
uv run atlas analyze outputs/demo.wav
```

`ATLAS_DATA_DIR`, `ATLAS_ARTIFACTS_DIR` and `ATLAS_OUTPUTS_DIR` override the default `data/`,
`artifacts/` and `outputs/` directories. Export them in the shell; the CLI does not read `.env`
files. The upload endpoint reads up to the first 30 seconds of a file, with a 32 MiB limit.

### MLflow

`music_atlas.tracking.track()` records one step as one MLflow run, and CLI steps expose it as
`--track`. The run gets the Git revision, profile parameters, elapsed and CPU seconds, peak
memory and its status (`FINISHED`, `FAILED` or `KILLED`; a resumed run points to the earlier
attempt). Each step is one MLflow experiment (`quick-sample`, `log-mel`, ...), and each
subset it reads is a dataset named `<split>/<subset>` describing the tracks; metrics hold only
what the step measured. MLflow artifacts sit in four folders: `dataset/`, `evaluation/`, `model/` and `run/`. Large
outputs stay in the run directory (`artifacts/<representation>/<run_id>/` or
`outputs/<run_id>/`) beside a `run.json` copy of the record. The application reads those
files and never needs MLflow. Conventions for new steps are in
`.agents/skills/atlas-tracking/SKILL.md`.

Runs are written to `MLFLOW_TRACKING_URI`, by default the file store `outputs/mlruns`; no
server runs while they are recorded. MLflow 3.17 refuses file stores unless
`MLFLOW_ALLOW_FILE_STORE=true`, which the runner and `scripts/mlflow-ui.sh` set. The file
store is kept because SQLite breaks with concurrent writers on Lustre. View runs with:

```bash
uv sync --locked --extra research --extra tracking
scripts/mlflow-ui.sh                                   # outputs/mlruns on 127.0.0.1:5000
scripts/mlflow-ui.sh --wcss                            # copy the WCSS runs first
MLFLOW_PORT=5001 MLFLOW_TRACKING_URI=file:/path/to/mlruns scripts/mlflow-ui.sh
```

On WCSS, `~/.atlas-env.sh` sets `MLFLOW_TRACKING_URI=file:$ATLAS_OUTPUTS_DIR/mlruns`, so
`sbatch scripts/slurm/quick-sample.sbatch --track` logs to PD while the job runs.
`--wcss` mirrors that store into `outputs/mlruns-wcss` with `rsync` and points its artifact
paths at the copy, which costs no grant hours. It connects to the SSH alias `wcss` (the
`wcss-slurm` skill's configuration) unless `ATLAS_WCSS_HOST` names another.

To read the PD store in place, open a Jupyter session on `lem-cpu-interactive` ([Notebooks on
Open OnDemand](#notebooks-on-open-ondemand)) and run in its terminal:

```bash
source ~/.atlas-env.sh && cd ~/music-atlas && scripts/mlflow-ui.sh --ood
```

Open the printed `https://ood.e-science.pl/node/<node>/<port>/` link once the terminal shows
`Uvicorn running`; before that OOD answers "Failed to connect". The UI opens in GenAI mode:
switch to *Model training* to see runs. `sbatch scripts/slurm/mlflow-ui.sbatch` starts the
same server as a job and prints the link in its log. Use Lem, not Bem2: the UI's
`cryptography` build needs AlmaLinux 9. Stop the server and delete the session when finished.

**Comparing runs.** In the UI, tick runs of one experiment and choose *Compare*; compare only
runs whose dataset digest matches, since the digest changes with the tracks read. The same
query from Python (for notebooks and agents):

```python
import mlflow  # with MLFLOW_ALLOW_FILE_STORE=true

mlflow.set_tracking_uri("file:outputs/mlruns-wcss")  # or file:outputs/mlruns
runs = mlflow.search_runs(
    experiment_names=["quick-sample"],  # one experiment per step
    filter_string="tags.`atlas.profile` = 'quick' and attributes.status = 'FINISHED'",
    order_by=["metrics.tracks_per_second DESC"],
)
```

**Results without MLflow.** MLflow is the lab notebook, not a dependency of the results. A
run directory holds everything the next step or the application needs: `run.json` (status,
params, metrics, dataset profile names), `metadata.json`, and for representations
`vectors.npy` with aligned `track_ids.npy`. Read only directories whose `run.json` says
`FINISHED`; a resumed run keeps its directory and its last attempt's status.

```python
import json
from pathlib import Path

import numpy as np

run = Path("artifacts/acoustic/RUN_ID")
assert json.loads((run / "run.json").read_text())["status"] == "FINISHED"
vectors, track_ids = np.load(run / "vectors.npy"), np.load(run / "track_ids.npy")
```

The application is pointed at such directories (exported models and indexes) and never
queries MLflow, so it runs with no tracking server or store.

A tracking server (`docker compose --profile tracking`, see
[Application and checks](#application-and-checks)) also works:
set `MLFLOW_TRACKING_URI=http://127.0.0.1:5000` in the shell that runs experiments.

## WCSS

Clone the repository into `$HOME` (50 GB). Data, environments and caches go to the music
service's project storage (PD), shared by the team. Every member must belong to that service;
`PD-info` lists services and their SLURM accounts. Set up once on the login node `ui.wcss.pl`:

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
uv sync --locked --extra research --extra notebook --extra tracking
```

Source `~/.atlas-env.sh` after every login. It stays outside Git and sets `SBATCH_ACCOUNT`
(jobs bill the music service without `-A`), `PYTHONNOUSERSITE=1` (`~/.local` cannot shadow
the environment) and `umask 0002` (files on PD stay writable for the team). The symlinks keep
local paths valid and send job logs (`outputs/slurm-*.log`) to PD. `uv sync` is fine on the
login node; keep the same extras on every sync, since `uv sync` removes the ones not listed.
Everything else runs as a job. Check the setup with the CPU smoke job:

```bash
sbatch --partition=bem2-cpu-short scripts/slurm/smoke.sbatch
sacct -j JOB_ID -o JobID,Account,State,Elapsed,MaxRSS
cat "$ATLAS_OUTPUTS_DIR"/smoke-JOB_ID.json
```

Jobs run with `--frozen --no-sync`. CPU and GPU hours are shared and do not refill: request few
cores and a short `--time`, and cancel what you do not need.

**Storage.** PD is Lustre, which suits few large files. Read MP3s straight from the shards and
write sharded outputs. Bem2's `TMPDIR` is also Lustre, so unpack only to node-local storage
such as Lem's NVMe, and never keep the only copy of a result in `TMPDIR`.

**GPUs** are only on Lem: `lem-gpu-short` (up to 3 days) with `--gres=gpu:hopper:N` and an
explicit `--cpus-per-task` (at most 16 per GPU), H100 with 94 GiB. Sync the locked `training`
extra into node-local `TMPDIR` with `--gres=gpu:hopper:1,storage:local:50g` rather than onto
Lustre.

### Data pipeline

FMA-full is downloaded once and shared on PD. Run each step from the repository root; reruns
skip finished shards.

```bash
sbatch scripts/slurm/fma-download.sbatch          # --probe: sizes only, --limit 2: pilot
sbatch scripts/slurm/fma-audit.sbatch
sbatch scripts/slurm/fma-split.sbatch --audit "$ATLAS_DATA_DIR/manifests/fma_audio_audit.parquet" \
  --max-artist-share 0.1 --publish
sbatch scripts/slurm/quick-sample.sbatch
sbatch scripts/slurm/log-mel.sbatch --track        # --limit 16: pilot, --resume RUN_ID: finish,
                                                   # --windows 5: 50 s per track instead of 30 s
```

Costs for sizing jobs (full profile unless stated):

- **Download**: 156 tar shards of up to 11 GiB, 930 GiB of the 2,000 GiB quota, about
  25 CPU-hours.
- **Audit**: one shard per core, about 19 CPU-hours for 8,200 hours of audio.
- **Split**: minutes on one core.
- **Quick sample**: 128 training tracks copied from about 85 shards into
  `samples/fma-cap10-v1/` (1 GiB), mostly waiting on Lustre. Opening a shard takes about 4 s,
  so batch jobs go shard by shard.
- **Log-mel cache**: one shard per core; decoding a 10 s window takes about 0.08 s and the
  transform 0.01 s. Three windows per track take about 9 CPU-hours and 33 GiB, five about
  5/3 of that. SLURM's `MaxRSS` counts the page cache of written files, so it can reach
  `--mem` safely.

For local work, copy the sample as the README shows (*Quick sample first, full data second*);
a rerun reads only `audio.tar` in seconds.

### Notebooks on Open OnDemand

[ood.e-science.pl](https://ood.e-science.pl) offers a shell, a file browser, the job list and
JupyterLab. Use notebooks to inspect data and results; run processing as `sbatch` jobs.
Register the project environment as a kernel once:

```bash
source ~/.atlas-env.sh && cd ~/music-atlas
uv run --frozen --no-sync python -m ipykernel install --user \
  --name music-atlas --display-name "Music Atlas" \
  --env ATLAS_DATA_DIR "$ATLAS_DATA_DIR" --env ATLAS_ARTIFACTS_DIR "$ATLAS_ARTIFACTS_DIR" \
  --env ATLAS_OUTPUTS_DIR "$ATLAS_OUTPUTS_DIR" --env HF_HOME "$HF_HOME" \
  --env PYTHONNOUSERSITE 1
```

Open Interactive Apps → Servers → **Jupyter Notebook** (the CPU app; GPU variants reserve an
H100) with the music service account, partition `lem-cpu-interactive` (`bem2` nodes are often
powered down), 1–2 hours, 1–2 cores and 8 GB RAM (memory is not billed). If Jupyter asks for a
password, take the one-off one from the session's `connection.yml`:

```bash
grep '^password:' ~/ondemand/data/sys/dashboard/batch_connect/sys/*/output/SESSION_ID/connection.yml
```

Choose the "Music Atlas" kernel. A session is billed while it is open, idle or not: delete it in
My Interactive Sessions when finished. Commit notebooks with aggregate outputs and no embedded
audio, so they read on GitHub without the cluster.

WCSS references: [home storage](https://man.e-science.pl/i/150),
[project storage and symlinks](https://man.e-science.pl/i/218),
[SLURM](https://man.e-science.pl/pl/kdm/slurm).

## Application and checks

```bash
docker compose up --build
docker compose --profile tracking up --build   # adds MLflow on localhost:5000
```

Compose serves the interface on localhost:3000 and FastAPI on localhost:8000; Next.js proxies
`/api/*` to FastAPI (`API_INTERNAL_URL` is resolved at build time).

```bash
uv run ruff check .
uv run pytest
cd web && npm run typecheck && npm run build
```

`AGENTS.md` holds the shared agent instructions and `CLAUDE.md` imports it. Skills live in
`.agents/skills/`, with `.claude/skills` linking to it; keep that symlink when copying the project.
