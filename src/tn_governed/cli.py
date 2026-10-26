"""Command-line entry points for governed workflows."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

_CONTAINMENT_ENVIRONMENT = (
    "PYTHONDONTWRITEBYTECODE",
    "TN_NO_LINK",
    "TN_NO_STDOUT",
    "TN_AUTOINIT_QUIET",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tn-governed")
    commands = parser.add_subparsers(dest="command", required=True)
    native = commands.add_parser(
        "governed-path",
        help="run signed editions through native TN, Polars, calculations and tn-llama",
    )
    native.add_argument("--run-directory", required=True, type=Path)
    native.add_argument("--unity-url", default="http://127.0.0.1:8080")
    native.add_argument("--server-binary", required=True, type=Path)
    native.add_argument("--model", required=True, type=Path)
    native.add_argument("--llama-port", default=18280, type=int)
    native.add_argument("--wsl", action="store_true")
    native.add_argument("--resume", action="store_true")
    native.add_argument("--check-refusals", action="store_true")
    portfolio_demo = commands.add_parser(
        "portfolio-demo",
        help="run the fixed local Unity-to-governed-tn-llama composition",
    )
    portfolio_demo.add_argument("--run-directory", required=True, type=Path)
    portfolio_demo.add_argument("--unity-url", default="http://127.0.0.1:18080")
    portfolio_demo.add_argument("--feed-url", default="http://127.0.0.1:18082")
    portfolio_demo.add_argument("--llama-port", required=True, type=int)
    reference = commands.add_parser(
        "reference-path",
        help="run the TN-object path through local Unity and governed tn-llama",
    )
    reference.add_argument("--run-directory", required=True, type=Path)
    reference.add_argument("--unity-url", default="http://127.0.0.1:18080")
    reference.add_argument("--llama-port", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected command without accepting secrets as arguments."""
    parser = _parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "first-pass":
        tokens[0] = "portfolio-demo"
    arguments = parser.parse_args(tokens)
    api_key = os.environ.get("TN_LLAMA_API_KEY", "")
    if not api_key.strip():
        parser.error("TN_LLAMA_API_KEY must be set to a nonempty local API key")

    previous = {name: os.environ.get(name) for name in _CONTAINMENT_ENVIRONMENT}
    try:
        for name in _CONTAINMENT_ENVIRONMENT:
            os.environ[name] = "1"

        if arguments.command == "governed-path":
            from .native_run import run_native_reference

            report_path, summary_path = run_native_reference(
                directory=arguments.run_directory,
                unity_url=arguments.unity_url,
                binary=arguments.server_binary,
                model=arguments.model,
                port=arguments.llama_port,
                api_key=api_key,
                wsl=arguments.wsl,
                resume=arguments.resume,
                check_refusals=arguments.check_refusals,
            )
            result = SimpleNamespace(report_path=report_path, summary_path=summary_path)
        elif arguments.command == "reference-path":
            from .run import ReferenceRunConfig, run_reference_path

            result = run_reference_path(
                ReferenceRunConfig(
                    run_directory=arguments.run_directory,
                    unity_url=arguments.unity_url,
                    llama_port=arguments.llama_port,
                    api_key=api_key,
                )
            )
        else:
            from .run import PortfolioDemoConfig, run_portfolio_demo

            result = run_portfolio_demo(
                PortfolioDemoConfig(
                    run_directory=arguments.run_directory,
                    unity_url=arguments.unity_url,
                    feed_url=arguments.feed_url,
                    llama_port=arguments.llama_port,
                    api_key=api_key,
                )
            )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print(f"Verified report: {result.report_path}")
    print(f"Factual run summary: {result.summary_path}")
    witness_path = getattr(result, "witness_path", None)
    if witness_path is not None:
        print(f"Governance verification report: {witness_path}")
    return 0
