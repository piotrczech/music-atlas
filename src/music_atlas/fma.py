"""Access to the published FMA archives."""

import collections
import csv
import hashlib
import io
import json
import struct
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

FMA_URL = "https://os.unil.cloud.switch.ch/fma/{}.zip"
ARCHIVES = ("fma_full", "fma_metadata")
BLOCK = 1 << 20
COMPRESSION = {0: "stored", 8: "deflate", 12: "bzip2", 14: "lzma"}
GIB = 2**30
INVENTORY_FIELDS = ("track_id", "audio_path", "status", "size_bytes", "sha1")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched in 1 MiB blocks with Range requests."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.requests = 0
        self.cache: dict[int, bytes] = {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            start = index * BLOCK
            end = min(start + BLOCK, self.size) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as r:
                # A 200 reply would stream the whole archive; stop before reading it.
                if r.status != 206:
                    raise OSError(f"Server ignored the Range header (HTTP {r.status}).")
                self.cache[index] = r.read()
            self.requests += 1
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[index]

    def readinto(self, buffer):
        view = memoryview(buffer).cast("B")
        filled = 0
        while filled < len(view) and self.pos < self.size:
            index, offset = divmod(self.pos, BLOCK)
            chunk = self._block(index)[offset : offset + len(view) - filled]
            view[filled : filled + len(chunk)] = chunk
            filled += len(chunk)
            self.pos += len(chunk)
        return filled


def archive_name(url: str) -> str:
    return PurePosixPath(urllib.parse.urlparse(url).path).stem


def parse_checksums(listing: bytes, root: PurePosixPath) -> dict[str, str]:
    """Map each path in a `sha1sum` listing, prefixed with `root`, to its SHA-1."""
    listed = {}
    for line in listing.decode().splitlines():
        if line.strip():
            sha1, path = line.split(maxsplit=1)
            listed[str(root / path.lstrip("*"))] = sha1
    return listed


def read_index(url: str) -> tuple[HTTPRangeFile, list[zipfile.ZipInfo], dict[str, str], bytes]:
    """Read a remote archive's members and its `checksums` file as {member name: SHA-1}."""
    remote = HTTPRangeFile(url)
    with zipfile.ZipFile(remote) as archive:
        files = [m for m in archive.infolist() if not m.is_dir()]
        checksums = next((m for m in files if PurePosixPath(m.filename).name == "checksums"), None)
        listing = archive.read(checksums) if checksums else b""
    # Checksum paths are relative to the directory holding the checksum file.
    root = PurePosixPath(checksums.filename).parent if checksums else PurePosixPath()
    return remote, files, parse_checksums(listing, root), listing


def probe(url: str, output_dir: Path) -> dict:
    """Plan disk use from a remote archive's central directory, without downloading audio.

    Writes `<archive>_probe.json` and the archive's own checksum list to `output_dir`.
    """
    remote, files, listed, listing = read_index(url)
    names = {m.filename for m in files}
    extracted = sum(m.file_size for m in files)
    both = remote.size + extracted
    summary = {
        "url": url,
        "files": len(files),
        "extensions": dict(
            collections.Counter(PurePosixPath(m.filename).suffix.lower() or "(none)" for m in files)
        ),
        "compression_methods": dict(
            collections.Counter(COMPRESSION.get(m.compress_type, m.compress_type) for m in files)
        ),
        "top_level": sorted({m.filename.split("/")[0] for m in files}),
        "examples": [m.filename for m in files[:5]],
        "archive_bytes": remote.size,
        "extracted_bytes": extracted,
        "archive_gib": round(remote.size / GIB, 2),
        "extracted_gib": round(extracted / GIB, 2),
        "archive_and_extracted_gib": round(both / GIB, 2),
        "empty_files": sum(m.file_size == 0 for m in files),
        "largest_file_mib": round(max((m.file_size for m in files), default=0) / 2**20, 1),
        "checksum_entries": len(listed),
        "missing_from_archive": sorted(listed.keys() - names),
        "unlisted_in_checksums": sorted(names - listed.keys()),
        "range_requests": remote.requests,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    # Bytes, not text: keeps the LF endings `sha1sum -c` expects when written on Windows.
    (output_dir / f"{archive_name(url)}_checksums.txt").write_bytes(listing)
    (output_dir / f"{archive_name(url)}_probe.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def fetch_member(url: str, info: zipfile.ZipInfo, end: int, root: Path, sha1: str | None) -> bool:
    """Write one member under `root` unless it is already there; return whether it was fetched."""
    target = root.joinpath(info.filename)
    # Lexical on purpose: resolve() races with other workers creating the same directory.
    if ".." in Path(info.filename).parts or not target.is_relative_to(root):
        raise ValueError("Member path leaves the destination directory.")
    # Files reach their final name only after verification, so a matching size means done.
    if target.is_file() and target.stat().st_size == info.file_size:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha1()
    request = urllib.request.Request(url, headers={"Range": f"bytes={info.header_offset}-{end}"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
            if response.status != 206:
                raise OSError(f"Server ignored the Range header (HTTP {response.status}).")
            header = response.read(zipfile.sizeFileHeader)
            if len(header) < zipfile.sizeFileHeader or header[:4] != b"PK\x03\x04":
                raise zipfile.BadZipFile("Missing local file header.")
            response.read(sum(struct.unpack("<2H", header[26:])))  # file name and extra field
            # ZipExtFile is undocumented but stable; it decompresses and checks the CRC-32.
            with zipfile.ZipExtFile(response, "r", info) as member:
                while chunk := member.read(BLOCK):
                    digest.update(chunk)
                    out.write(chunk)
        if sha1 and digest.hexdigest() != sha1:
            raise ValueError("SHA-1 does not match the published checksum.")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    return True


def download(url: str, dest: Path, report_dir: Path, limit: int = 0, workers: int = 4) -> dict:
    """Stream archive members into `dest` one Range request each, keeping verified files only.

    The archive itself is never stored. A rerun skips completed members and retries failures;
    `limit` takes the first members in archive order for a throughput pilot.
    """
    remote, files, listed, _ = read_index(url)
    files.sort(key=lambda m: m.header_offset)
    # A member's bytes end where the next one starts; the last one ends before the index.
    ends = [m.header_offset - 1 for m in files[1:]] + [remote.size - 1]
    selected = list(zip(files, ends, strict=True))[: limit or None]
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()

    def fetch(item):
        info, end = item
        try:
            return fetch_member(url, info, end, root, listed.get(info.filename)), None
        except Exception as error:  # record the member and continue; a rerun retries it
            return False, f"{type(error).__name__}: {error}"

    started = time.monotonic()
    downloaded = skipped = downloaded_bytes = 0
    failures = []
    pool = ThreadPoolExecutor(workers)
    try:
        results = zip(selected, pool.map(fetch, selected), strict=True)
        for done, ((info, _), (fetched, error)) in enumerate(results, 1):
            if error:
                failures.append({"path": info.filename, "error": error})
            elif fetched:
                downloaded += 1
                downloaded_bytes += info.file_size
            else:
                skipped += 1
            if done % 1000 == 0:
                rate = downloaded_bytes / 2**20 / (time.monotonic() - started)
                print(
                    f"{done}/{len(selected)} members, {downloaded_bytes / GIB:.1f} GiB new, "
                    f"{rate:.1f} MiB/s, {len(failures)} failed",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        # Without cancel_futures, Ctrl+C would wait for every queued member.
        pool.shutdown(cancel_futures=True)

    elapsed = time.monotonic() - started
    report = {
        "url": url,
        "selected_members": len(selected),
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": len(failures),
        "downloaded_gib": round(downloaded_bytes / GIB, 3),
        "elapsed_seconds": round(elapsed, 1),
        "mib_per_second": round(downloaded_bytes / 2**20 / max(elapsed, 1e-6), 1),
        "workers": workers,
        "unverified": sorted(m.filename for m, _ in selected if m.filename not in listed),
        "missing_from_archive": sorted(listed.keys() - {m.filename for m in files}),
        "failures": failures,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{archive_name(url)}_download.json").write_text(json.dumps(report, indent=2))
    return report


def inventory(data_dir: Path, report_dir: Path) -> dict:
    """Link every track in `fma_full/checksums` to its audio file and flag unlisted files.

    Writes `manifests/fma_full_inventory.csv` with paths relative to `data_dir`. Files reach
    their final name only after `download` verifies them, so this only lists them;
    `sha1sum -c checksums` in the archive directory re-verifies their content.
    """
    archive = PurePosixPath("fma", "fma_full")
    checksums = data_dir / archive / "checksums"
    if not checksums.is_file():
        raise OSError(f"{checksums} not found; run `atlas fma-download` first.")
    listed = parse_checksums(checksums.read_bytes(), archive)
    on_disk = {}
    for folder, _, names in (data_dir / archive).walk():
        for name in names:
            path = folder / name
            on_disk[path.relative_to(data_dir).as_posix()] = path.stat().st_size
    known = {str(archive / "README.txt"), str(archive / "checksums")}

    rows = []
    for path in sorted((listed.keys() | on_disk.keys()) - known):
        if path not in listed:
            status = "unexpected"
        elif path in on_disk:
            status = "present"
        else:
            status = "missing"
        stem = PurePosixPath(path).stem
        track_id = int(stem) if stem.isdigit() else ""
        values = (track_id, path, status, on_disk.get(path, ""), listed.get(path, ""))
        rows.append(dict(zip(INVENTORY_FIELDS, values, strict=True)))
    output = data_dir / "manifests" / "fma_full_inventory.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    with part.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    part.replace(output)

    counts = collections.Counter(row["status"] for row in rows)
    summary = {
        "inventory": output.relative_to(data_dir).as_posix(),
        "listed_tracks": len(listed),
        "present": counts["present"],
        "missing": counts["missing"],
        "unexpected": counts["unexpected"],
        "present_gib": round(sum(on_disk[p] for p in listed.keys() & on_disk.keys()) / GIB, 2),
        "unexpected_paths": [row["audio_path"] for row in rows if row["status"] == "unexpected"],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "fma_full_inventory.json").write_text(json.dumps(summary, indent=2))
    return summary
