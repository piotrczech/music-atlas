# Storage, scratch and results

**TL;DR**
- `$HOME`: 50 GB **and 1 000 000 files** (STO-01), snapshotted in `~/.snapshot`
  (STO-02). Code and small inputs only.
- PD `/lustre/pd03/<service>`: terabytes, not backed up. Data, environments as
  one archive, results.
- `$TMPDIR` in a job: Lustre on Bem2 (STO-05), node-local NVMe on Lem sized by
  `--gres=storage:local:Ng` (STO-06). Moved to
  `/lustre/tmp/slurm/finished_jobs/<jobid>/<TYPE>/` at job end and kept 14 days
  (STO-10, STO-12): "the results vanished" is usually recoverable, from inside a
  job, since `/lustre/tmp` is an empty stub on `ui` (STO-11).
- `$PDDIR` is the default service's PD on the login node and the **job's
  account's** PD inside a job (STO-13). Templates use
  `/lustre/pd03/${SLURM_JOB_ACCOUNT}` so the two agree.
- Lustre hates small files: never run a venv from PD; tar it, unpack into `$TMPDIR`.
- Compute nodes reach the internet (NET-01): `pip install` and downloads work in jobs.

Official: [katalog domowy](https://man.e-science.pl/pl/kdm/katalog-domowy),
[PD](https://man.e-science.pl/pl/kdm/przestrzen-dyskowa),
[TMPDIR](https://man.e-science.pl/pl/kdm/tymczasowa-przestrzen-dyskowa-tmp),
[Lustre](https://man.e-science.pl/pl/kdm/lustre).

## Tiers

| Where | Size | Lifetime | Use for |
|---|---|---|---|
| `$HOME` | 50 GB, 1 M files | permanent, snapshots | scripts, small inputs, small logs |
| PD | 200 GB to 20 TB per service | permanent, no backup | datasets, weights, results |
| `$TMPDIR` | per job | job + 14 days | scratch while computing |
| `/tmp` | 200 MB Bem2, 1 GB Lem (STO-08) | job | nothing that matters |
| `/dev/shm` | 357 GB to 1.5 TB (STO-09) | job | fastest scratch, costs memory |

`quota -s -f ~` shows both `$HOME` quotas; the file one bites first, a Python
environment can exhaust it long before 50 GB. Login-node `$TMPDIR` is
`$HOME/tmp` (STO-04), a trap for scratch. On Lem `--gres=storage:local:10g`
gave a 10 GB `/mnt/lscratch`; a GPU job with no storage GRES got 40 GB.

## `$PDDIR` and the three scratch variables

`/etc/profile.d/pddir-exports.sh` exports `$PDDIR` (default service) and
`$PDDIRS` (all, colon-separated) (STO-03), also in `ssh host 'cmd'` on a stock
account. Inside a job `$PDDIR` follows `--account`, and the same directory
appears as `/lustre/pd01/...` there and `/lustre/pd03/...` on `ui`; `pd01` links
to `pd03` (STO-13). A job billed to another service writes results where
`ls $PDDIR/runs` on `ui` will not find them, which is why the templates derive
the path from the account.

`TMPDIR_LUSTRE`, `TMPDIR_LOCAL`, `TMPDIR_SHM` are set for the partition's
default tiers plus what you requested with `--gres=storage:lustre:1`,
`storage:local:Ng`, `storage:shm:Ng`; a Lem GPU job with only a GPU GRES had
both `LUSTRE` and `LOCAL` (STO-07). `$TMPDIR` points at the default.

## Results

Copy out inside the job, on every exit:

```bash
OUT="/lustre/pd03/${SLURM_JOB_ACCOUNT}/runs/$SLURM_JOB_ID"
copy_out() { mkdir -p "$OUT"; cp -r "$TMPDIR/results/." "$OUT/"; }
trap copy_out EXIT      # success, error, and the 30 s after a wall-clock kill
```

If you forgot: `ls /lustre/tmp/slurm/finished_jobs/<jobid>/` from inside a job
(`<TYPE>` is `LUSTRE`, `LOCAL` or `SHM`), for 14 days. Never write bulk output
to `$HOME`; a job that overruns the quota dies with `EDQUOT` after you paid for
the allocation.

## Lustre and small files

PD and the Lustre `$TMPDIR` share a metadata server. The manual asks for large
sequential reads, few small files, small directories, and no `ls -l` in big
ones, and warns misuse can degrade the system for everyone. Consequences: keep
environments as one archive on PD and unpack per job ([python.md](python.md)),
tar per array task, use `find` or `lfs find` for listings. Compute nodes reach
`pypi.org`, `github.com` and `huggingface.co` (NET-01); egress addresses differ
per cluster, check `curl -s https://api.ipify.org` in a job if a provider
allowlists by IP.

Staging pattern: download or rsync → PD → `$TMPDIR` per job → PD for results.
