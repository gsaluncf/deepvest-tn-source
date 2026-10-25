"""Install with Python 3.12: python scripts/install.py --tn-wheel PATH.

The native TN wheel is supplied separately and checked against its pinned digest.
Unity, OPA and the governed model server are external services or executables.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from dependency_paths import resolve_dependency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tn-wheel", type=Path)
    parser.add_argument("--dependency-dir", type=Path, default=Path(__file__).resolve().parents[1] / ".runtime/dependencies")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = json.loads((root / "config/sdk-build.json").read_text())
    try:
        wheel = resolve_dependency("tn_wheel", args.tn_wheel, directory=args.dependency_dir)
    except ValueError as error:
        parser.error(str(error))
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != build["wheel_sha256"]:
        raise SystemExit("TN wheel does not match the pinned SHA-256 digest")
    target = root / "vendor/wheels" / build["wheel_sha256"] / wheel.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target != wheel:
        shutil.copy2(wheel, target)
    subprocess.run([sys.executable, "-m", "pip", "install", str(target)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(root / "dependencies/tn-polars"),
                    "-e", str(root / "dependencies/tn-llama-client"), "-e", str(root) + "[dev]"], check=True)


if __name__ == "__main__":
    main()
