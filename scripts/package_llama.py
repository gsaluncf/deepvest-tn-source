"""Install the complete local server build, including its shared libraries."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def package(source, destination, backend):
    source, destination = Path(source), Path(destination)
    if not (source / "llama-server").is_file():
        raise FileNotFoundError(source / "llama-server")
    files = [source / "llama-server", *sorted(source.glob("*.so*"))]
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for path in files:
        target = destination / path.name
        shutil.copy2(path, target)
        records.append({"file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (destination / "build.json").write_text(json.dumps({
        "backend": backend,
        "llama_cpp_revision": "3f545beccee69d9975f466ec7e45fd9aacd8ba90",
        "source_manifest_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "native/tn-llama/source-manifest.json").read_bytes()
        ).hexdigest(),
        "files": records,
    }, indent=2) + "\n")
    print(f"Installed TN-enabled server and libraries: {destination}")


if __name__ == "__main__":
    package(*sys.argv[1:])
