"""Set up and run SQL -> dataframe -> native LLM -> accepted report -> review queue.

Run with Python 3.12 on Windows. Docker Desktop is required unless --unity-url
names an existing catalog. The TN-enabled Linux model server runs through WSL.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from uuid import uuid4

from install_unity import docker, install, wait_ready
from dependency_paths import resolve_dependency

ROOT = Path(__file__).resolve().parents[1]


def case_command(python, args, directory, unity_url):
    return [str(python), "-m", "experiments.pvldb_evaluation.run", "--mode", "primitive",
            "--rows", str(args.rows), "--numeric-type", args.numeric_type, "--with-model",
            "--unity-url", unity_url, "--binary", str(args.binary.resolve()),
            "--model-path", str(args.model_path.resolve()), "--port", str(args.model_port),
            "--run-directory", str(directory)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tn-wheel", type=Path, help="Override dependency discovery")
    parser.add_argument("--binary", type=Path, help="Override dependency discovery")
    parser.add_argument("--build-server", choices=["cpu", "cuda"],
                        help="Build the included TN server source through WSL before running")
    parser.add_argument("--model-path", type=Path, help="Override dependency discovery")
    parser.add_argument("--opa", type=Path, help="Override dependency discovery")
    parser.add_argument("--dependency-dir", type=Path, default=ROOT / ".runtime/dependencies",
                        help="Directory containing the wheel, OPA, TN model server and GGUF")
    parser.add_argument("--unity-url", help="Use this existing catalog instead of installing one")
    parser.add_argument("--unity-port", type=int, default=18080)
    parser.add_argument("--model-port", type=int, default=18325)
    parser.add_argument("--numeric-type", choices=["int", "float", "Decimal"], default="Decimal")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--python", type=Path, help="Use an existing Python 3.12 environment")
    parser.add_argument("--skip-install", action="store_true", help="Use dependencies already installed in --python")
    args = parser.parse_args()
    for required in [ROOT / "uv.lock", ROOT / "config/sdk-build.json"]:
        if not required.is_file():
            parser.error(f"Required dependency record is missing: {required}")
    for name in ("tn_wheel", "model_path", "opa"):
        try:
            path = resolve_dependency(name, getattr(args, name), directory=args.dependency_dir)
        except ValueError as error:
            parser.error(str(error))
        setattr(args, name, path)
    directory = (args.run_directory or ROOT / "experiments/results" / ("submission-" + uuid4().hex[:12])).resolve()
    if directory.exists():
        parser.error("Run directory must be new")
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.build_server:
        if args.binary:
            parser.error("Use --build-server or --binary, not both")
        distribution = os.environ.get("TN_WSL_DISTRO")
        script = (ROOT / "scripts/build_llama.sh").resolve()
        if len(script.drive) != 2:
            parser.error("The server build requires a drive-backed Windows checkout")
        linux_script = "/mnt/" + script.drive[0].lower() + script.as_posix()[2:]
        subprocess.run(["wsl", *(["-d", distribution] if distribution else []),
                        "--", "bash", "-lc", 'exec bash "$1" "$2"', "build-tn",
                        linux_script, args.build_server], check=True)
        args.binary = ROOT / ".runtime/dependencies/tn-llama/llama-server"
    try:
        args.binary = resolve_dependency("binary", args.binary, directory=args.dependency_dir)
    except ValueError as error:
        parser.error(str(error))
    python = args.python.resolve() if args.python else ROOT / ".venv/Scripts/python.exe"
    if not args.python and not python.exists():
        subprocess.run(["py", "-3.12", "-m", "venv", str(ROOT / ".venv")], check=True)
    subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 12), 'Python 3.12 required'"], check=True)
    if not args.skip_install:
        subprocess.run([str(python), str(ROOT / "scripts/install.py"), "--tn-wheel", str(args.tn_wheel)], check=True, cwd=ROOT)
    environment = os.environ.copy()
    environment["TN_POLARS_OPA"] = str(args.opa)
    environment["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in
        ["src", "dependencies/tn-polars", "dependencies/tn-llama-client"])
    service = None
    try:
        if args.unity_url:
            url = args.unity_url.rstrip("/")
            wait_ready(url, timeout=15)
        else:
            service = install("tn-submission-" + uuid4().hex[:10], args.unity_port)
            url = service["url"]
        result = subprocess.run(case_command(python, args, directory, url), cwd=ROOT, env=environment)
        summary_path = directory / "case-summary.json"
        if result.returncode or not summary_path.is_file():
            raise RuntimeError(f"Workflow did not complete. Inspect {summary_path}")
        summary = json.loads(summary_path.read_text())
        if not (summary.get("completed") and summary.get("verified") and summary.get("with_model")):
            raise RuntimeError(f"Workflow or verification failed. Inspect {summary_path}")
        for filename in ["report.html", "review.json"]:
            if not (directory / filename).is_file():
                raise RuntimeError(f"Missing completed model output: {filename}")
        print(f"Accepted LLM report: {directory / 'report.html'}")
        print(f"Pending human review: {directory / 'review.json'}")
    finally:
        if service:
            docker("stop", service["container"])
            print(f"Unity stopped; container {service['name']} and its data volume retained.")


if __name__ == "__main__":
    main()
