"""Download FMA into verified tar shards on project storage."""

import csv
import hashlib
import io
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

FMA_URL = "https://os.unil.cloud.switch.ch/fma/{}.zip"
AUDIO = re.compile(r"fma_full/(\d{3})/(\d{6})\.mp3")
BLOCK = 16 << 20
GIB = 2**30


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file; each read is one Range request."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urlopen(Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = max(0, offset + (0, self.pos, self.size)[whence])
        return self.pos

    def readinto(self, buffer):
        end = min(self.pos + len(buffer), self.size)
        if self.pos >= end:
            return 0
        request = Request(self.url, headers={"Range": f"bytes={self.pos}-{end - 1}"})
        with urlopen(request, timeout=120) as r:
            # A 200 reply would stream the whole archive; stop before reading it.
            if r.status != 206:
                raise OSError(f"Server ignored the Range header (HTTP {r.status}).")
            count = r.readinto(buffer)
        self.pos += count
        return count


def open_remote(url: str) -> io.BufferedReader:
    # Members are read in archive order, so a 16 MiB buffer keeps Range requests few.
    return io.BufferedReader(HTTPRangeFile(url), BLOCK)


def parse_checksums(text: str) -> dict[str, str]:
    lines = (line.split(maxsplit=1) for line in text.splitlines() if line.strip())
    return {path.lstrip("*"): sha1 for sha1, path in lines}


def fetch_metadata(fma_dir: Path) -> None:
    """Extract and verify the metadata; the folder gets its final name only when complete."""
    target, part = fma_dir / "fma_metadata", fma_dir / "fma_metadata.part"
    if target.exists():
        return
    shutil.rmtree(part, ignore_errors=True)
    with zipfile.ZipFile(open_remote(FMA_URL.format("fma_metadata"))) as archive:
        archive.extractall(part)
    folder = part / "fma_metadata"
    for name, sha1 in parse_checksums((folder / "checksums").read_text()).items():
        with (folder / name).open("rb") as file:
            if hashlib.file_digest(file, "sha1").hexdigest() != sha1:
                raise ValueError(f"SHA-1 mismatch for fma_metadata/{name}")
    folder.replace(target)
    part.rmdir()


def write_shard(url: str, members: list[str], sha1s: dict[str, str], target: Path) -> None:
    """Pack one archive folder into a tar that gets its final name only if every file verifies."""
    part = target.with_name(target.name + ".part")
    try:
        with zipfile.ZipFile(open_remote(url)) as archive, tarfile.open(part, "w") as tar:
            for name in members:
                data = archive.read(name)  # zipfile checks the CRC-32
                if hashlib.sha1(data).hexdigest() != sha1s[name]:
                    raise ValueError(f"SHA-1 mismatch for {name}")
                entry = tarfile.TarInfo(name.removeprefix("fma_full/"))
                entry.size = len(data)
                tar.addfile(entry, io.BytesIO(data))
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)


def download(data_dir: Path, probe: bool = False, limit: int = 0, workers: int = 4) -> dict:
    """Fetch the metadata and one tar per FMA-full folder, then write the inventory."""
    fma_dir, url = data_dir / "fma", FMA_URL.format("fma_full")
    remote = open_remote(url)
    with zipfile.ZipFile(remote) as archive:
        listing = parse_checksums(archive.read("fma_full/checksums").decode())
        sizes = {info.filename: info.file_size for info in archive.infolist()}
    sha1s = {f"fma_full/{path}": sha1 for path, sha1 in listing.items()}
    # Unlisted files cannot be verified, so they are reported and not packed.
    unexpected = sorted(name for name in sizes if AUDIO.fullmatch(name) and name not in sha1s)
    shards = defaultdict(list)
    for name in sha1s:
        if (match := AUDIO.fullmatch(name)) and name in sizes:
            shards[match[1]].append(name)
    shard_bytes = {key: sum(sizes[name] for name in names) for key, names in shards.items()}
    todo = [key for key in sorted(shards) if not (fma_dir / f"fma_full/{key}.tar").exists()]
    todo = todo[: limit or None]
    if probe:
        return {
            "archive_gib": round(remote.raw.size / GIB, 1),
            "audio_gib": round(sum(shard_bytes.values()) / GIB, 1),
            "shards": len(shards),
            "largest_shard_gib": round(max(shard_bytes.values()) / GIB, 1),
            "shards_to_download": len(todo),
            "unexpected": unexpected,
        }

    fetch_metadata(fma_dir)
    (fma_dir / "fma_full").mkdir(parents=True, exist_ok=True)

    def run(key: str) -> str | None:
        started = time.monotonic()
        try:
            write_shard(url, shards[key], sha1s, fma_dir / f"fma_full/{key}.tar")
        except Exception as error:  # report the shard and continue; a rerun retries it
            return f"{type(error).__name__}: {error}"
        seconds = time.monotonic() - started
        print(f"{key}.tar {shard_bytes[key] / GIB:.1f} GiB in {seconds:.0f} s", file=sys.stderr)

    pool = ThreadPoolExecutor(workers)
    try:
        errors = dict(zip(todo, pool.map(run, todo), strict=True))
    finally:
        pool.shutdown(cancel_futures=True)  # Ctrl+C would otherwise wait for every queued shard

    done = {key for key in shards if (fma_dir / f"fma_full/{key}.tar").exists()}
    rows = []
    for name, sha1 in sorted(sha1s.items()):
        if match := AUDIO.fullmatch(name):
            shard = f"fma/fma_full/{match[1]}.tar"
            present = name in sizes and match[1] in done
            row = (int(match[2]), shard, name.removeprefix("fma_full/"), sizes.get(name, ""), sha1)
            rows.append((*row, "present" if present else "missing"))
    inventory = data_dir / "manifests" / "fma_full_inventory.csv"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("track_id", "shard", "audio_path", "size_bytes", "sha1", "status"))
        writer.writerows(rows)
    present = sum(row[-1] == "present" for row in rows)
    return {
        "shards": len(shards),
        "complete": len(done),
        "tracks_present": present,
        "tracks_missing": len(rows) - present,
        "unexpected": unexpected,
        "failed": {key: error for key, error in errors.items() if error},
    }
