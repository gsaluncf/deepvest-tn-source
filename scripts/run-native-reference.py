"""Run the native governed application after installing the pinned project environment."""

import argparse
import os
from pathlib import Path

from deepvest_tn.native_run import run_native_reference

parser = argparse.ArgumentParser()
parser.add_argument("--run-directory", required=True, type=Path)
parser.add_argument("--unity-url", default="http://127.0.0.1:8080")
parser.add_argument("--binary", required=True, type=Path)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--port", default=18280, type=int)
parser.add_argument("--wsl", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--check-refusals", action="store_true")
args = parser.parse_args()
api_key = os.environ.get("TN_LLAMA_API_KEY", "")
if not api_key.strip():
    parser.error("TN_LLAMA_API_KEY must be set")
report, summary = run_native_reference(
    directory=args.run_directory,
    unity_url=args.unity_url,
    binary=args.binary,
    model=args.model,
    port=args.port,
    api_key=api_key,
    wsl=args.wsl,
    resume=args.resume,
    check_refusals=args.check_refusals,
)
print(f"Verified report: {report}")
print(f"Execution record: {summary}")
