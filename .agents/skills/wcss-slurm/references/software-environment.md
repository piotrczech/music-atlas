# Modules, environments and containers

**TL;DR**
- `module` works only inside jobs, after `source /usr/local/sbin/modules.sh`
  (MOD-03). On the login node that file does not exist (MOD-02) and `module`
  is an alias that says so (MOD-01). This is the most common one-second job death.
- Module sets differ per cluster in size and content (MOD-04, MOD-05): 2164 on
  Bem2, 1735 on Lem CPU, 1018 on Lem GPU nodes. Check on the target partition.
- Python: `SciPy-bundle/2024.05-gfbf-2024a` (numpy, scipy, pandas) is the one
  version on both clusters; the bare `Python/3.13.5` module has no numpy.
  Details and traps in [python.md](python.md).
- Apptainer 1.5.2 on compute nodes only (CON-01); prebuilt `pytorch_latest.sif`
  and vLLM images in `/lustre/software-data/container-images/` (CON-02).
- 75 `sub-*` wrappers in `/usr/local/bin/bem2/` build and submit jobs for common
  packages (MOD-06); `sub-interactive-lem-gpu` is the documented GPU shell (MOD-07).

Official: [module](https://man.e-science.pl/pl/kdm/slurm/SLURM_module),
[sub-skrypty](https://man.e-science.pl/pl/kdm/slurm/skrypty-sub),
[kontenery](https://man.e-science.pl/pl/kdm/slurm/kontenery),
[mamba](https://man.e-science.pl/pl/kdm/mamba).

## Modules

```bash
source /usr/local/sbin/modules.sh       # in every batch script
module load Python/3.13.5-GCCcore-14.3.0
```

Never `module load X | tail` or `$(module load X)`: the load happens in a
subshell and is discarded. To explore, `sub-interactive -c 2 -m 4 -t 1` then
`module avail Python`. Names present on Bem2 (`ecCodes`, `CDO`, `JupyterLab`)
were absent on Lem; assume nothing across clusters.

## Containers

```bash
apptainer exec --nv /lustre/software-data/container-images/pytorch_latest.sif python train.py
```

`--nv` exposes GPUs. A `.sif` is one file, the friendliest thing you can put on
Lustre. No root and no `--fakeroot`: build elsewhere or through the Buildtainer
route on `k8s-build`. Keep the pull cache off shared filesystems:
`export APPTAINER_TMPDIR="$TMPDIR" APPTAINER_CACHEDIR="$TMPDIR"`.

## `sub-*` wrappers

Run on the login node: `sub-interactive -c 4 -m 12 -t 6`,
`sub-interactive-bem2-cpu`, `sub-interactive-lem-cpu`, `sub-interactive-lem-gpu`,
and about 60 packages (VASP, ORCA, Gaussian, GROMACS, MATLAB, R, ...). If your
software is on the list, start from the wrapper. For conda-style work the
manual documents Mamba; put the environment on PD.
