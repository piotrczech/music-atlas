# Official documentation, and where it disagrees with the machine

The WCSS manual at <https://man.e-science.pl/pl/kdm/> is published all rights
reserved (`contentLicense: "alr"`). This repository therefore **links** to it and
describes behaviour in its own words; it copies no text. Read the official pages
for anything this skill does not cover — especially the ~50 per-application
pages under `kdm/oprogramowanie/`, which this repository does not duplicate.

The manual is in Polish.

## Map

### Getting started

| Page | What is there |
|---|---|
| [kdm](https://man.e-science.pl/pl/kdm/) | manual index |
| [kdm/uzytkownik](https://man.e-science.pl/pl/kdm/uzytkownik) | how to become a KDM user |
| [kdm/wnioski_uslugi](https://man.e-science.pl/pl/kdm/wnioski_uslugi) | applying for a service |
| [kdm/dostep](https://man.e-science.pl/pl/kdm/dostep) | SSH and Open OnDemand access; the 3-strike lockout |
| [kdm/ssh](https://man.e-science.pl/pl/kdm/ssh) | SSH client configuration |
| [kdm/szkolenia](https://man.e-science.pl/pl/kdm/szkolenia) | beginners' training |
| [kdm/faq](https://man.e-science.pl/pl/kdm/faq) | job problems FAQ |
| [kdm/regulaminy](https://man.e-science.pl/pl/kdm/regulaminy) | terms of use |

### SLURM

| Page | What is there |
|---|---|
| [kdm/slurm](https://man.e-science.pl/pl/kdm/slurm) | SLURM index |
| [kdm/slurm/tldr](https://man.e-science.pl/pl/kdm/slurm/tldr) | condensed cheat sheet |
| [kdm/slurm/partycje-slurm](https://man.e-science.pl/pl/kdm/slurm/partycje-slurm) | the real partition table, node models, TMPDIR per partition, and the bare-name routing mechanism |
| [kdm/slurm/uruchamianie-zadan](https://man.e-science.pl/pl/kdm/slurm/uruchamianie-zadan) | submitting jobs |
| [kdm/slurm/alokacja-zasobow](https://man.e-science.pl/pl/kdm/slurm/alokacja-zasobow) | requesting cores, memory, nodes |
| [kdm/slurm/zarzadzanie-zadaniami](https://man.e-science.pl/pl/kdm/slurm/zarzadzanie-zadaniami) | monitoring and cancelling |
| [kdm/slurm/tablica-zadan](https://man.e-science.pl/pl/kdm/slurm/tablica-zadan) | array jobs |
| [kdm/slurm/SLURM_module](https://man.e-science.pl/pl/kdm/slurm/SLURM_module) | the `module` command |
| [kdm/slurm/skrypty-sub](https://man.e-science.pl/pl/kdm/slurm/skrypty-sub) | the `sub-*` wrappers |
| [kdm/slurm/kontenery](https://man.e-science.pl/pl/kdm/slurm/kontenery) | Apptainer |
| [kdm/slurm/gpu](https://man.e-science.pl/pl/kdm/slurm/gpu) · [gpu/nodes](https://man.e-science.pl/pl/kdm/slurm/gpu/nodes) | GPU jobs and GPU node inventory |
| [kdm/slurm/mpi](https://man.e-science.pl/pl/kdm/slurm/mpi) | MPI |
| [kdm/slurm/zmienne-srodowiskowe](https://man.e-science.pl/pl/kdm/slurm/zmienne-srodowiskowe) | SLURM environment variables |
| [kdm/slurm/sinfo](https://man.e-science.pl/pl/kdm/slurm/sinfo) | reading `sinfo` |
| [kdm/slurm/przyklady](https://man.e-science.pl/pl/kdm/slurm/przyklady) | worked examples |
| [kdm/pbs-i-slurm](https://man.e-science.pl/pl/kdm/pbs-i-slurm) | migrating from the old PBS queue |

### Storage

| Page | What is there |
|---|---|
| [kdm/katalog-domowy](https://man.e-science.pl/pl/kdm/katalog-domowy) | `$HOME`, the 50 GB cap, snapshots |
| [kdm/przestrzen-dyskowa](https://man.e-science.pl/pl/kdm/przestrzen-dyskowa) | PD directories |
| [kdm/tymczasowa-przestrzen-dyskowa-tmp](https://man.e-science.pl/pl/kdm/tymczasowa-przestrzen-dyskowa-tmp) | `$TMPDIR`, its three types, and the 14-day retention |
| [kdm/lustre](https://man.e-science.pl/pl/kdm/lustre) | the Lustre filesystem |
| [kdm/odzyskiwanie-danych-i-backupy](https://man.e-science.pl/pl/kdm/odzyskiwanie-danych-i-backupy) | backups and recovery |
| [kdm/archiwum](https://man.e-science.pl/pl/kdm/archiwum) | archiving |
| [kdm/bezpieczenstwo-danych](https://man.e-science.pl/pl/kdm/bezpieczenstwo-danych) | data security |
| [kdm/nieaktywni-uzytkownicy](https://man.e-science.pl/pl/kdm/nieaktywni-uzytkownicy) | what happens to inactive users' data |

### Accounting, machines, tooling

| Page | What is there |
|---|---|
| [kdm/limity](https://man.e-science.pl/pl/kdm/limity) | per-service QOS limits — see the disagreements below |
| [kdm/rozliczanie_uslug](https://man.e-science.pl/pl/kdm/rozliczanie_uslug) | annual reporting duty |
| [kdm/rejestr_zuzycia_zasobow](https://man.e-science.pl/pl/kdm/rejestr_zuzycia_zasobow) | usage register |
| [kdm/bem2/parametry](https://man.e-science.pl/pl/kdm/bem2/parametry) · [kdm/lem-parametry](https://man.e-science.pl/pl/kdm/lem-parametry) | machine specifications |
| [kdm/bem2/build-containers](https://man.e-science.pl/pl/kdm/bem2/build-containers) | building containers (Buildtainer) |
| [kdm/kompilacja](https://man.e-science.pl/pl/kdm/kompilacja) | compiling on the cluster |
| [kdm/mamba](https://man.e-science.pl/pl/kdm/mamba) | Mamba / Conda |
| [kdm/hpc-info](https://man.e-science.pl/pl/kdm/hpc-info) | the job statistics portal |
| [kdm/k8s](https://man.e-science.pl/pl/kdm/k8s) | Kubernetes service |
| [kdm/nomachine](https://man.e-science.pl/pl/kdm/nomachine) · [kdm/przekierowanie-portow](https://man.e-science.pl/pl/kdm/przekierowanie-portow) | remote desktop, port forwarding |
| [kdm/plgrid](https://man.e-science.pl/pl/kdm/plgrid) | PLGrid users |
| [kdm/news](https://man.e-science.pl/pl/kdm/news) | changes announcements — check this first when something breaks |

## Documented claims

These IDs rest on the manual rather than on a measurement. Each one is tied to
its page here; `scripts/check-claim-ids.sh` fails if a cited ID has neither a
check in `verify.sh` nor a row in this table.

| ID | Page | What it says |
|---|---|---|
| `D:ACC-02` | [kdm/dostep](https://man.e-science.pl/pl/kdm/dostep) | three failed password logins lock the account for 24 h |
| `D:ACC-05` | [kdm/dostep](https://man.e-science.pl/pl/kdm/dostep) | the login host is `ui.wcss.pl` |
| `D:ACC-06` | [kdm/dostep](https://man.e-science.pl/pl/kdm/dostep) | Open OnDemand is at `ood.e-science.pl` |
| `D:STO-12` | [kdm/tymczasowa-przestrzen-dyskowa-tmp](https://man.e-science.pl/pl/kdm/tymczasowa-przestrzen-dyskowa-tmp) | finished jobs' scratch is kept 14 days |
| `D:BILL-01` | [kdm/rozliczanie_uslug](https://man.e-science.pl/pl/kdm/rozliczanie_uslug) | the service owner files an annual report |
| `D:BILL-02` | [kdm/rozliczanie_uslug](https://man.e-science.pl/pl/kdm/rozliczanie_uslug) | publications must acknowledge WCSS |

## Disagreements

Each is re-checked by `scripts/verify.sh`. Where the machine and the manual differ, the machine is authoritative; these are listed so you can see *why* a number here may not match a page you have read.

| ID | The manual | The machine |
|---|---|---|
| `M≠D:QOS-03` | [limity](https://man.e-science.pl/pl/kdm/limity): 50 running jobs on `bem2-cpu`, 250 on `lem-cpu` | associations carry `MaxJobs=150` for both, `MaxSubmit=500` |
| `M≠D:QOS-07` | [limity](https://man.e-science.pl/pl/kdm/limity): a flat "8 GPU" for `lem-gpu` | the GPU ceiling is a per-service `GrpTRES gres/gpu` and differed from 8 on the service tested; QOS `gpu` sets no limit itself |
| `M≠D:DOC-01` | [limity](https://man.e-science.pl/pl/kdm/limity) heads a block "Partycja GPU **lem-cpu-interactive**" | the figures under it (1 job, 16 cores, 250 GB, 1 GPU) match QOS `gpu-interactive`, i.e. `lem-gpu-interactive`. Looks like a typo in the heading |
| `M≠D:DOC-02` | [SLURM_module](https://man.e-science.pl/pl/kdm/slurm/SLURM_module) shows `module avail` examples at a `ui` prompt | `module` is unavailable on `ui`; it is an alias that says so (MOD-01), and `/usr/local/sbin/modules.sh` does not exist there (MOD-02). The same page also states modules should not be loaded on `ui`, so the examples appear to be stale rather than wrong in intent |
| `M≠D:DOC-03` | [katalog-domowy](https://man.e-science.pl/pl/kdm/katalog-domowy) says `$TMPDIR` exists only for the duration of the job | [TMPDIR](https://man.e-science.pl/pl/kdm/tymczasowa-przestrzen-dyskowa-tmp) says it is kept 14 days in `finished_jobs`, and measurement confirms the 14-day version (STO-10). Two official pages disagree with each other |
| `M≠D:HW-05` | partition tables give cores per node (128 for a Lem CPU node) | `lscpu` in a job reports threads (256). Both true; know which you are sizing against |

Node counts on [partycje-slurm](https://man.e-science.pl/pl/kdm/slurm/partycje-slurm)
also differ slightly from `sinfo` (102 vs 96 for `bem2-cpu-short`, 184 vs 186 for
`lem-cpu-short`). That is drift from drained nodes, not an error — always trust
`sinfo` for what is up right now.

## Claims corrected in this repository

For anyone arriving from an older private version of this material, these were
checked and found wrong:

- "The documented partitions `bem2-cpu`, `lem-cpu`, `lem-gpu` do not exist."
  They are not SLURM partitions, but they work as submit-time aliases, and the
  manual documents that mechanism accurately (SUB-05).
- "`source /usr/local/sbin/modules.sh` is required first, every time, everywhere."
  The file does not exist on the login node (MOD-02); it exists inside jobs
  (MOD-03), which is where the line belongs.
- "`sbatch --test-only` catches limit errors." It does not catch the one that
  matters — an unaffordable job passes (QOS-06).
- "`reportseff` / `seff` give you efficiency." Neither is installed (ACT-03).
- "Defaults are 2 cores and 4 GB." Measured: 1 core, 4000 MB (SUB-01).
- "`$TMPDIR` is deleted when the job ends; results vanish." It is moved and kept
  14 days (STO-10).
- "`/tmp` is 1 GB." 200 MB on Bem2, 1 GB on Lem (STO-08).

And these were wrong in version 0.1 of this repository itself, found by
re-measuring on 2026-09-10:

- "`ssh host 'cmd'` loses `$PDDIR` and the WCSS shell functions." A stock
  `~/.bashrc` brings `/etc/profile.d` in; only the `module` alias is missing
  (ACC-08).
- "A default `Pool()` starts 48 or 128 workers." True up to Python 3.12; the
  3.13 module honours the allocation (PY-04).
- "`export PRTE_TMPDIR=...` moves OpenMPI 5's session directory." It is
  ignored; the MCA parameter `prte_tmpdir_base` works (MPI-06).
- "The `uv` installer appends to `~/.bashrc`." It appends to every profile it
  finds; on the account measured that was `.bash_profile`, `.profile` and
  `.zshrc` (PY-11).
