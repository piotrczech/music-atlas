import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def profile(name: str) -> dict:
    if name not in {"quick", "full"}:
        raise ValueError("Profile must be quick or full.")
    with (ROOT / "configs" / f"{name}.toml").open("rb") as file:
        return tomllib.load(file)


def paths() -> dict[str, Path]:
    return {
        name: Path(os.environ.get(f"ATLAS_{name.upper()}_DIR", ROOT / name)).expanduser().resolve()
        for name in ("data", "artifacts", "outputs")
    }
