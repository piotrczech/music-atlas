import argparse
import json
import wave
from pathlib import Path

import numpy as np

from music_atlas.audio import analyze
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
    args = parser.parse_args()
    if args.command == "doctor":
        result = {"paths": paths(), "profile": profile(args.profile)}
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
