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
    audit = commands.add_parser("fma-audit", help="Audit FMA-full audio, labels and licences.")
    audit.add_argument("--limit", type=int, default=0, help="Measure at most N shards.")
    audit.add_argument("--workers", type=int, default=4)
    manifest = commands.add_parser("fma-split", help="Prepare versioned FMA dataset splits.")
    manifest.add_argument("--audit", type=Path, help="Apply the completed E1-03 audio audit.")
    manifest.add_argument(
        "--strategy", choices=("proportional", "rare-support"), default="proportional",
        help="Set label targets for the held-out splits.",
    )
    manifest.add_argument(
        "--max-artist-share", type=float,
        help="Keep an artist in training if it would supply more of a genre's held-out quota.",
    )
    manifest.add_argument(
        "--concentration-penalty", type=float, default=0.0,
        help="Make artists that fill much of a held-out genre costlier to place there.",
    )
    manifest.add_argument(
        "--publish", action="store_true",
        help="Make this version the shared manifests/tracks.parquet.",
    )
    record_filter = commands.add_parser("fma-record-filter", help="Record exclusions for a run.")
    record_filter.add_argument("version", help="Version of the existing FMA split manifest.")
    record_filter.add_argument("representation")
    record_filter.add_argument("run_id")
    record_filter.add_argument("exclusions", type=Path, help="CSV with track_id and error columns.")
    sample = commands.add_parser(
        "quick-sample", help="Build or reload the quick-profile sample and time the loader.",
    )
    sample.add_argument("version", nargs="?", help="Dataset version; the profile's by default.")
    sample.add_argument("--track", action="store_true", help="Record the run in MLflow.")
    logmel = commands.add_parser("log-mel", help="Cache log-mel windows for CNN training.")
    logmel.add_argument("--profile", choices=["quick", "full"], default="quick",
                        help="quick: the quick sample; full: every track of the three splits.")
    logmel.add_argument("--limit", type=int, default=0, help="Cache at most N shards.")
    logmel.add_argument("--workers", type=int, default=4)
    logmel.add_argument("--track", action="store_true", help="Record the run in MLflow.")
    logmel.add_argument("--resume", metavar="RUN_ID", help="Continue an unfinished tracked run.")
    logmel.add_argument(
        "--windows", type=int, help="Windows per track instead of the profile's; 5 contains the "
        "three default windows as windows 0, 2 and 4.",
    )
    commands.add_parser(
        "window-study", help="Compare 5, 10 and 20 s log-mel windows on the quick sample.",
    ).add_argument("--track", action="store_true", help="Record the run in MLflow.")
    args = parser.parse_args()
    if args.command == "doctor":
        result = {"paths": paths(), "profile": profile(args.profile)}
    elif args.command == "fma-download":
        try:
            result = download(paths()["data"], args.probe, args.limit, args.workers)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
    elif args.command in {
        "fma-audit", "fma-split", "fma-record-filter", "quick-sample", "log-mel", "window-study",
    }:
        try:
            storage = paths()
            if args.command == "fma-audit":
                from music_atlas.audit import run

                result = run(storage["data"], args.workers, args.limit)
            elif args.command == "fma-split":
                from music_atlas.manifest import build

                result = build(
                    storage["data"], args.audit, strategy=args.strategy,
                    max_artist_share=args.max_artist_share,
                    concentration_penalty=args.concentration_penalty, publish=args.publish,
                )
            elif args.command == "quick-sample":
                from music_atlas.sample import run

                if args.track:
                    from music_atlas.tracking import track

                    with track(
                        "quick-sample", "quick", storage, args.version,
                        description="Load every quick-sample track through the shared loader "
                        "and time it; the reference throughput for sizing full jobs.",
                    ) as record:
                        result = run(storage["data"], args.version, record)
                    result |={"run_id": record.run_id, "run_dir": record.directory}
                else:
                    result = run(storage["data"], args.version)
            elif args.command == "log-mel":
                from music_atlas.logmel_cache import run

                options = {"profile_name": args.profile, "workers": args.workers,
                           "limit": args.limit, "windows_per_track": args.windows}
                if args.track or args.resume:
                    from music_atlas.tracking import track

                    windows = args.windows or profile(args.profile)["windows_per_track"]
                    with track(
                        "log-mel", args.profile, storage, representation="logmel",
                        resume=args.resume,
                        overrides={"windows_per_track": args.windows} if args.windows else None,
                        description=f"Log-mel of {windows} windows per track (128 bands, 23 ms "
                        "hop, dB re full scale) cached as float16, one file per shard, for CNN "
                        "training.",
                    ) as record:
                        result = run(storage["data"], storage["artifacts"], **options,
                                     record=record)
                    result |= {"run_id": record.run_id, "run_dir": record.directory}
                else:
                    result = run(storage["data"], storage["artifacts"], **options)
            elif args.command == "window-study":
                from music_atlas.window_study import run

                if args.track:
                    from music_atlas.tracking import track

                    with track(
                        "window-study", "quick", storage,
                        description="Window length study: the log-mel of 5, 10 and 20 s centred "
                        "windows of the quick sample. Only the window length changes; the sample "
                        "rate, transform and window centre stay fixed.",
                    ) as record:
                        result = run(storage["data"], storage["outputs"], record)
                    result |= {"run_id": record.run_id, "run_dir": record.directory}
                else:
                    result = run(storage["data"], storage["outputs"])
            else:
                from music_atlas.split_views import record_filter

                result = record_filter(
                    storage["data"], storage["artifacts"], args.version,
                    args.representation, args.run_id, args.exclusions,
                )
        except ImportError as error:
            parser.exit(
                1, f"{error}. Install with uv sync --locked --extra research --extra tracking.\n",
            )
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
