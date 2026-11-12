import argparse
import json
import wave
from pathlib import Path

import numpy as np

from music_atlas.audio import analyze
from music_atlas.fma import download
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
    fma = commands.add_parser("fma-download", help="Download FMA into verified tar shards.")
    fma.add_argument("--probe", action="store_true", help="Only report sizes and shard counts.")
    fma.add_argument("--limit", type=int, default=0, help="Download at most N shards.")
    fma.add_argument("--workers", type=int, default=4)
    manifest = commands.add_parser("fma-split", help="Prepare versioned FMA dataset splits.")
    manifest.add_argument("--audit", type=Path, help="Apply the completed E1-03 audio audit.")
    manifest.add_argument(
        "--strategy", choices=("proportional", "rare-support"), default="proportional",
        help="Set label targets for the held-out splits.",
    )
    record_filter = commands.add_parser("fma-record-filter", help="Record exclusions for a run.")
    record_filter.add_argument("version", help="Version of the existing FMA split manifest.")
    record_filter.add_argument("representation")
    record_filter.add_argument("run_id")
    record_filter.add_argument("exclusions", type=Path, help="CSV with track_id and error columns.")
    args = parser.parse_args()
    if args.command == "doctor":
        result = {"paths": paths(), "profile": profile(args.profile)}
    elif args.command == "fma-download":
        try:
            result = download(paths()["data"], args.probe, args.limit, args.workers)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
    elif args.command in {"fma-split", "fma-record-filter"}:
        try:
            storage = paths()
            if args.command == "fma-split":
                from music_atlas.manifest import build

                result = build(storage["data"], args.audit, strategy=args.strategy)
            else:
                from music_atlas.split_views import record_filter

                result = record_filter(
                    storage["data"], storage["artifacts"], args.version,
                    args.representation, args.run_id, args.exclusions,
                )
        except ImportError as error:
            parser.exit(1, f"{error}. Install with uv sync --locked --extra research.\n")
        except (OSError, ValueError, KeyError) as error:
            parser.exit(1, f"{error}\n")
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
    # Lets the SLURM script retry failed shards. Missing tracks also fail a full run, since
    # every shard can succeed while the catalogue stays incomplete; a --limit pilot cannot.
    if result.get("failed") or (result.get("tracks_missing") and not args.limit):
        parser.exit(1)
