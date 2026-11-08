# Python on WCSS

**TL;DR**
```bash
source /usr/local/sbin/modules.sh              # exists only inside jobs
module load SciPy-bundle/2024.05-gfbf-2024a    # numpy 1.26 on both clusters; NOT in a pipeline
export PYTHONNOUSERSITE=1                      # ~/.local shadows every module package
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"   # or =1 under a multiprocessing pool
```
- `cpu_count()` is the node, not your allocation: 48 or 128 in a `-c 4` job.
  Python 3.12 and older size `Pool()` from it; size it yourself (PY-04).
- System Python is 3.6.8 on Bem2 and 3.9.25 on Lem (PY-01): always load a module.
- No PyTorch, TensorFlow, scikit-learn, numba or dask modules (PY-08): use a
  container or a venv, kept as one tar on PD and unpacked into `$TMPDIR`.
- A venv built with the *module* Python runs on both clusters (PY-06).

## The four traps

**1. `module load` in a pipeline does nothing** (PY-02). `module load X | tail -2`
runs the load in a subshell; you see its output and `python3` stays the system
one. Redirect instead: `module load X 2>/dev/null`.

**2. The system Python is ancient** (PY-01), 3.6.8 on Bem2 and 3.9.25 on Lem. A
venv built on it gets 2021-era packages.

**3. `~/.local` overrides everything** (PY-03). `pip install --user`, or plain
`pip install` with no venv, writes to `~/.local/lib/python3.X/site-packages`,
which is on `sys.path` for every job on both clusters. On the account
measured it held 22 GB and shadowed the module numpy. `python3 -s` diagnoses,
`export PYTHONNOUSERSITE=1` in the job script cures, `du -sh ~/.local` audits.

**4. Thread and process oversubscription** (PY-04), measured in a `-c 4` job:

| Interpreter | `cpu_count()` Bem2 / Lem | `sched_getaffinity` | default `Pool()` |
|---|---|---|---|
| system 3.6.8 / 3.9.25 | 48 / 128 | 4 | **48 / 128 workers** |
| `SciPy-bundle/2024.05` (3.12.3) | 48 / 128 | 4 | **48 / 128 workers** |
| `Python/3.13.5` module | 48 / 128 | 4 | 4 |

```python
n = len(os.sched_getaffinity(0))           # correct everywhere
with multiprocessing.Pool(n) as pool: ...
```

Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` explicitly:
`$SLURM_CPUS_PER_TASK` for one threaded process, `1` under a pool. Processes ×
threads must not exceed `-c`; a 32-worker pool with 32 BLAS threads each is
1024 threads on 32 cores, and it runs, just slowly, while you pay.

## Getting packages

**Modules.** `SciPy-bundle` versions differ per cluster; only
`2024.05-gfbf-2024a` (numpy 1.26.4) is on both (PY-05). Bem2 also has 2023.02
and 2023.07; Lem has 2025.06 and 2025.07 (numpy 2.3). `matplotlib`, `h5py`,
`IPython` exist; no ML frameworks (PY-08).

**A venv, stored as one file.** A bare venv is about 2200 small files and
every import is a metadata burst against Lustre (PY-09); a 200-task array of
them hurts everyone. Build once, tar to PD, unpack per job (PY-12): measured
0.3 s each way for a small archive.

```bash
# once, in an interactive job
module load Python/3.13.5-GCCcore-14.3.0
python -m venv "$TMPDIR/venv" && "$TMPDIR/venv/bin/pip" install -r requirements.txt
tar -C "$TMPDIR" -cf "$PD/envs/myproject.tar" venv
# in every job, after loading the same Python module
tar -xf "$PD/envs/myproject.tar" -C "$TMPDIR" && source "$TMPDIR/venv/bin/activate"
```

Pin with `pip freeze`. Built with the module Python the venv moved from Bem2
to Lem and computed correctly (PY-06), with the same module loaded on Lem: the
venv's `python` is a symlink into the module's tree, so without the module the
activated `python` is not found. Built with the system Python it cannot move,
the ABIs differ.

**`uv`** is not installed but works without root and makes 946 files where
`venv` + `pip` make 2199 (PY-10). Two things to control (PY-11): its installer
appends a `source` line to every shell profile it finds (`.bash_profile`,
`.profile`, `.zshrc`), and if you installed into job scratch every later login
prints `No such file or directory` for a path that died with the job; set
`UV_NO_MODIFY_PATH=1` and install somewhere permanent. Its cache defaults to
`~/.cache/uv`; set `UV_CACHE_DIR="$TMPDIR/uv-cache"`.

**Containers** for PyTorch and similar: [software-environment.md](software-environment.md).

## A job script that gets it right

[templates/job.sbatch](../templates/job.sbatch) is this, with placeholders:
`set -euo pipefail`, `modules.sh`, one of the two software routes, the threads
or pool choice, `PYTHONNOUSERSITE=1`, results in `$TMPDIR` copied to the paying
service's PD by an exit trap, and provenance echoed at the top of the log.
Notebooks: Open OnDemand runs Jupyter as a normal job, billed while open
([access.md](access.md)).
