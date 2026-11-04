import argparse
import json
import wave
from pathlib import Path

import numpy as np

from music_atlas.audio import analyze
from music_atlas.fma import ARCHIVES, FMA_URL, download, probe
from music_atlas.settings import paths, profile


def main():
    parser = argparse.ArgumentParser(prog="atlas")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Show resolved paths and the selected profile.")
    doctor.add_argument("--profile", choices=["quick", "full"], default="quick")
    inspect = commands.add_parser("analyze", help="Analyze an audio file, at most 30 seconds.")
    inspect.add_argument("input", type=Path)
    demo = commands.add_parser("demo", help="Generate a synthetic signal, not a music example.")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo.wav"))
    fma_probe = commands.add_parser(
        "fma-probe", help="Report an FMA archive's size and file list without downloading it."
    )
    fma_probe.add_argument("archive", nargs="?", choices=ARCHIVES, default="fma_full")
    fma_probe.add_argument("--output-dir", type=Path, help="Defaults to ATLAS_OUTPUTS_DIR.")
    fma_download = commands.add_parser(
        "fma-download",
        help="Stream and verify an FMA archive file by file into ATLAS_DATA_DIR/fma.",
    )
    fma_download.add_argument("archive", nargs="?", choices=ARCHIVES, default="fma_full")
    fma_download.add_argument("--limit", type=int, default=0, help="First N members; 0 is all.")
    fma_download.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "doctor":
        result = {"paths": paths(), "profile": profile(args.profile)}
    elif args.command in ("fma-probe", "fma-download"):
        url = FMA_URL.format(args.archive)
        try:
            if args.command == "fma-probe":
                result = probe(url, args.output_dir or paths()["outputs"])
            else:
                dirs = paths()
                result = download(
                    url, dirs["data"] / "fma", dirs["outputs"], args.limit, args.workers
                )
        except (OSError, ValueError) as error:
            parser.exit(1, f"Cannot read {url}: {error}\n")
    elif args.command == "demo":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(3 * 22050) / 22050
        y = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
        with wave.open(str(args.output), "wb") as file:
            file.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
            file.writeframes(y.tobytes())
        result = {"output": args.output, "signal": "440 Hz sine; synthetic, not music"}
    else:
        try:
            result = analyze(args.input)
        except (ValueError, OSError) as error:
            parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2, default=str))
    if result.get("failed"):  # a non-zero exit lets scripts rerun the download until it is complete
        parser.exit(1)
