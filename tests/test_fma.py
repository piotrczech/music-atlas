import contextlib
import csv
import hashlib
import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from music_atlas import fma

AUDIO = {"fma_full/000/000002.mp3": b"ID3" + bytes(500), "fma_full/000/000003.mp3": b""}
CHECKSUMS = {
    "000/000002.mp3": hashlib.sha1(AUDIO["fma_full/000/000002.mp3"]).hexdigest(),
    "000/000003.mp3": "0" * 40,  # deliberately wrong
    "000/000005.mp3": "1" * 40,  # listed but not archived
}


def fma_like_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_BZIP2) as archive:
        archive.writestr("fma_full/README.txt", "readme")
        archive.writestr("fma_full/checksums", "".join(f"{s}  {p}\n" for p, s in CHECKSUMS.items()))
        for name, data in AUDIO.items():
            archive.writestr(name, data)
        archive.writestr("../escape.txt", "must not leave the destination")
    return buffer.getvalue()


class RangeHandler(BaseHTTPRequestHandler):
    """Serves one file and honours single `bytes=a-b` ranges, like the FMA object store."""

    data = b""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.data)))
        self.end_headers()

    def do_GET(self):
        start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
        body = self.data[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def served(data: bytes):
    handler = type("Handler", (RangeHandler,), {"data": data})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_port}/fma_full.zip"
        server.shutdown()


def test_probe_lists_remote_archive_across_range_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(fma, "BLOCK", 64)  # force reads to span many Range requests
    data = fma_like_archive()
    with served(data) as url:
        summary = fma.probe(url, tmp_path)

    assert summary["archive_bytes"] == len(data)
    assert summary["compression_methods"] == {"bzip2": 5}
    assert summary["empty_files"] == 1
    assert summary["checksum_entries"] == 3
    assert summary["missing_from_archive"] == ["fma_full/000/000005.mp3"]
    assert summary["unlisted_in_checksums"] == [
        "../escape.txt",
        "fma_full/README.txt",
        "fma_full/checksums",
    ]
    assert summary["range_requests"] > 2
    assert b"\r" not in (tmp_path / "fma_full_checksums.txt").read_bytes()
    assert (tmp_path / "fma_full_probe.json").exists()


def test_download_keeps_only_verified_members_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(fma, "BLOCK", 64)  # decompress and hash in several chunks
    dest, reports = tmp_path / "data", tmp_path / "outputs"
    with served(fma_like_archive()) as url:
        first = fma.download(url, dest, reports, workers=2)
        second = fma.download(url, dest, reports, workers=2)

    assert (first["downloaded"], first["skipped_existing"], first["failed"]) == (3, 0, 2)
    errors = {failure["path"]: failure["error"] for failure in first["failures"]}
    assert "SHA-1" in errors["fma_full/000/000003.mp3"]
    assert "leaves the destination" in errors["../escape.txt"]
    assert first["missing_from_archive"] == ["fma_full/000/000005.mp3"]
    good = dest / "fma_full/000/000002.mp3"
    assert good.read_bytes() == AUDIO["fma_full/000/000002.mp3"]
    assert not (dest / "fma_full/000/000003.mp3").exists()
    assert not (tmp_path / "escape.txt").exists()
    assert not list(tmp_path.rglob("*.part"))
    assert (second["downloaded"], second["skipped_existing"], second["failed"]) == (0, 3, 2)
    assert (reports / "fma_full_download.json").exists()


def test_inventory_links_listed_tracks_and_flags_unlisted_files(tmp_path):
    archive = tmp_path / "fma" / "fma_full"
    (archive / "000").mkdir(parents=True)
    (archive / "checksums").write_text("".join(f"{s}  {p}\n" for p, s in CHECKSUMS.items()))
    (archive / "README.txt").write_text("readme")
    (archive / "000/000002.mp3").write_bytes(AUDIO["fma_full/000/000002.mp3"])
    (archive / "000/000099.mp3.part").write_bytes(b"interrupted")

    summary = fma.inventory(tmp_path, tmp_path / "outputs")

    assert (summary["present"], summary["missing"], summary["unexpected"]) == (1, 2, 1)
    assert summary["unexpected_paths"] == ["fma/fma_full/000/000099.mp3.part"]
    output = tmp_path / summary["inventory"]
    assert b"\r" not in output.read_bytes()
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(r["track_id"], r["status"], r["size_bytes"]) for r in rows] == [
        ("2", "present", "503"),
        ("3", "missing", ""),
        ("5", "missing", ""),
        ("", "unexpected", "11"),
    ]
    assert rows[0]["audio_path"] == "fma/fma_full/000/000002.mp3"
    assert rows[0]["sha1"] == CHECKSUMS["000/000002.mp3"]
