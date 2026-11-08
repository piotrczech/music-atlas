# Getting data in and out

**TL;DR**
- One door: `ui.wcss.pl`, no transfer node (DT-02), 15 shared cores and 1 GiB
  of RAM per user. `rsync` and `scp` stream and are fine; `tar`, `zip -r` or a
  conversion script belong in `sub-interactive`.
- Land bulk data in PD, never `$HOME`. About 92 MB/s to PD from the login node (DT-03).
- `rsync -avh --partial --inplace`, and tar many small files before copying:
  both `$HOME` and PD have file-count quotas and Lustre is slow with small files.
- `rsync`, `rclone`, `zstd`, `pigz` are present; `lftp`, `sshfs`,
  `globus-url-copy` are not (DT-01). Compute nodes can download directly (NET-01).
- PD has **no backup**; `$HOME` has snapshots; `$TMPDIR` lives 14 days in `finished_jobs`.

Official: [PD](https://man.e-science.pl/pl/kdm/przestrzen-dyskowa),
[archiwum](https://man.e-science.pl/pl/kdm/archiwum),
[backupy](https://man.e-science.pl/pl/kdm/odzyskiwanie-danych-i-backupy),
[FileSender](https://man.e-science.pl/pl/filesender).

## In

```bash
PD=$(ssh wcss 'echo $PDDIR')
rsync -avh --partial --inplace --info=progress2 ./dataset/ "wcss:$PD/dataset/"
```

`--partial` resumes after a drop, `--inplace` avoids a second copy near a
quota, `-z` is a mistake for already-compressed data, `--dry-run` first when
the destination is not empty. Run long transfers under `tmux` on your own
machine. For many small files, archive first: `tar -I 'zstd -T0 -3' -cf
dataset.tar.zst dataset/`, copy the archive, unpack in a job into `$TMPDIR`,
keep the archive on PD as the canonical copy.

Inside a job, `curl -L -o "$TMPDIR/data.tar" https://...` works; stage once to
PD rather than re-downloading in every job. `rclone` covers S3, Drive, Dropbox,
WebDAV. FileSender is the manual's route for one-off files from outsiders.

## Out

```bash
ssh wcss 'cd $PDDIR/runs && tar -I "zstd -T0" -cf - 5855074' > run-5855074.tar.zst
```

streams without a second copy on the cluster. For anything big, tar in a job
and `rsync` the archive.

## Backup

| | Backup |
|---|---|
| `$HOME` | `~/.snapshot/` |
| PD | **none**: keep an off-site copy of anything irreplaceable, or use the archive service |
| `$TMPDIR` | none, but 14 days in `finished_jobs/` |

Data of inactive users has [its own policy](https://man.e-science.pl/pl/kdm/nieaktywni-uzytkownicy).
