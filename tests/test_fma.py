import csv
import hashlib
import io
import sys
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from music_atlas import cli, fma

AUDIO = {"000/000002.mp3": b"ID3" + bytes(500), "000/000003.mp3": b"", "001/001000.mp3": b"ID3"}
CHECKSUMS = {path: hashlib.sha1(data).hexdigest() for path, data in AUDIO.items()}
CHECKSUMS["001/001000.mp3"] = "0" * 40  # corrupt, so shard 001 must fail
CHECKSUMS["000/000005.mp3"] = "1" * 40  # listed but not archived
UNLISTED = {"002/002000.mp3": b"ID3"}  # archived but not listed
TRACKS = b"track_id\n2\n"


def archive(folder: str, files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_BZIP2) as zf:
        for name, data in files.items():
            zf.writestr(f"{folder}/{name}", data)
    return buffer.getvalue()


def listing(checksums: dict[str, str]) -> bytes:
    return "".join(f"{sha1}  {path}\n" for path, sha1 in checksums.items()).encode()


class RangeHandler(BaseHTTPRequestHandler):
    """Serves the two archives and honours single `bytes=a-b` ranges, like the FMA store."""

    files = {
        "/fma_full.zip": archive(
            "fma_full", {"checksums": listing(CHECKSUMS), **AUDIO, **UNLISTED}
        ),
        "/fma_metadata.zip": archive(
            "fma_metadata",
            {
                "checksums": listing({"tracks.csv": hashlib.sha1(TRACKS).hexdigest()}),
                "tracks.csv": TRACKS,
            },
        ),
    }

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.files[self.path])))
        self.end_headers()

    def do_GET(self):
        start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
        body = self.files[self.path][start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_download_packs_verified_shards_writes_inventory_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(fma, "BLOCK", 64)  # force reads across many Range requests
    shard = tmp_path / "fma/fma_full/000.tar"
    with ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setattr(fma, "FMA_URL", f"http://127.0.0.1:{server.server_port}/{{}}.zip")
        plan = fma.download(tmp_path, probe=True)
        assert not (tmp_path / "fma").exists()
        first = fma.download(tmp_path, workers=2)
        built = shard.stat().st_mtime_ns
        second = fma.download(tmp_path, workers=2)
        server.shutdown()

    assert (plan["shards"], plan["shards_to_download"]) == (2, 2)
    assert plan["unexpected"] == first["unexpected"] == ["fma_full/002/002000.mp3"]
    assert list(first["failed"]) == ["001"] and "SHA-1" in first["failed"]["001"]
    with tarfile.open(shard) as tar:
        assert {m.name: tar.extractfile(m).read() for m in tar} == {
            path: data for path, data in AUDIO.items() if path.startswith("000/")
        }
    assert not list(tmp_path.rglob("*.part"))
    assert (tmp_path / "fma/fma_metadata/tracks.csv").read_bytes() == TRACKS
    with (tmp_path / "manifests/fma_full_inventory.csv").open(newline="") as file:
        rows = [(row["track_id"], row["shard"], row["status"]) for row in csv.DictReader(file)]
    assert rows == [
        ("2", "fma/fma_full/000.tar", "present"),
        ("3", "fma/fma_full/000.tar", "present"),
        ("5", "fma/fma_full/000.tar", "missing"),
        ("1000", "fma/fma_full/001.tar", "missing"),
    ]
    assert (second["complete"], list(second["failed"])) == (1, ["001"])
    assert shard.stat().st_mtime_ns == built


def test_full_download_with_missing_tracks_exits_non_zero(monkeypatch, capsys):
    monkeypatch.setattr(cli, "download", lambda *args: {"tracks_missing": 1, "failed": {}})
    monkeypatch.setattr(sys, "argv", ["atlas", "fma-download"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 1
    monkeypatch.setattr(sys, "argv", ["atlas", "fma-download", "--limit", "2"])
    cli.main()  # a pilot leaves tracks missing by design
