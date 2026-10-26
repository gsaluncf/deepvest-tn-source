"""Internal use followed by an LLM report with a pending human-review workflow.

The SQL rows are synthetic Bloomberg-style demonstration data, not a Bloomberg
feed or an interpretation of a vendor license.
"""

import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
from uuid import uuid4

import tn
from deepvest_tn.native_run import governed_server, wsl_path
from deepvest_tn.publications import PublicationStore, write_once

from .catalog import AssignedRules, UnityGovernance
from .demo import handle
from .evidence import EvidenceRecorder
from .model import NativeModelEndpoint, digest, prepare_native_model
from .primitives import Context
from .provision import provision
from .sql import GovernedSQL, source_identity

NOTICE = "REVIEW REQUIRED: This AI-generated report uses synthetic Bloomberg-style market data. Human review is pending. Internal use only."
REVIEW_ACTION = {"action": "human_review", "assignee": "internal-reviewer",
                 "reason": "LLM use of governed market data requires a human to review the report."}
APPLICATION, PURPOSE = "bloomberg-example", "internal-analytics"
MODEL_USE = {"application": "primitive-model", "purpose": PURPOSE, "operation": "generate_report"}
RECIPIENT_USE = {"application": "primitive-recipient", "purpose": PURPOSE, "operation": "read_report"}


def policy(column):
    return f"""## experiment.primitive
### instruction
Synthetic Bloomberg-style {column} data for internal analytics.
### use_for
Internal calculations and LLM reports. Notify the LLM to mark its report with: {NOTICE}
Generate the report now and request human review; review may remain pending.
### do_not_use_for
Claiming a pending report was reviewed or these synthetic values are market quotes.
### consequences
Carry these conditions through derivatives. Send the exact report to internal-reviewer.
### on_violation_or_error
Record outstanding actions and retain the report's source lineage.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity-url", default="http://127.0.0.1:8080")
    parser.add_argument("--project-directory", type=Path, default=Path.cwd(),
                        help="Project containing config/sdk-build.json and the pinned dependencies")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--port", type=int, default=18325)
    parser.add_argument("--native-windows", action="store_true", help="use a compatible Windows native binary instead of WSL")
    args = parser.parse_args()
    repository = args.project_directory.resolve()
    args.binary = args.binary or repository / ".runtime/dependencies/tn-llama/llama-server"
    args.model = args.model or repository / ".artifacts/models/Qwen3-4B-Q4_K_M.gguf"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    root = (args.run_directory or repository / "experiments/results" / ("primitive-review-workflow-" + stamp)).resolve()
    sdk = json.loads((repository / "config/sdk-build.json").read_text())
    native_module = Path(tn.__file__).parent / "_native.pyd"
    if importlib.metadata.version("tn-proto") != sdk["version"] or digest(native_module) != sdk["install"]["native_module_sha256"]:
        raise RuntimeError("installed TN runtime differs from the pinned build")
    if not args.binary.is_file() or not args.model.is_file():
        raise FileNotFoundError("the configured native model binary and GGUF must exist")
    root.mkdir(parents=True, exist_ok=False)
    print(f"Run artifacts: {root}", flush=True)
    database = root / "market.sqlite"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE market_data (name TEXT NOT NULL, value INTEGER NOT NULL)")
        connection.execute("INSERT INTO market_data VALUES (?, ?)", ("ACME", 1250))
        connection.commit()
    paths = provision(root / "sessions", roles=["catalog", "gateway", "worker", "model-reader", "observer"],
                      read_only_roles=("model-reader", "observer"))
    output_paths = provision(root / "output-sessions", roles=["model-writer", "recipient"],
                             groups=["tn.agents", "default"], read_only_roles=("recipient",))
    store = PublicationStore(root / "publications")
    with ExitStack() as stack:
        sessions = {role: stack.enter_context(tn.Session.from_config(str(path)))
                    for role, path in {**paths, **output_paths}.items() if role != "observer"}
        authority = sessions["catalog"].did
        admin = sessions["catalog"].policy("experiment.administration")
        audit_policy = sessions["catalog"].policy("experiment.evidence")
        writers = [sessions[r].did for r in ("gateway", "worker")]
        catalog_name = "disclosure_" + uuid4().hex[:12]
        def resolver(role):
            return UnityGovernance(args.unity_url, catalog_name, session=sessions[role],
                                   authority=authority, administration=admin, store=store)
        publisher = resolver("catalog")
        model_conditions = {"use": MODEL_USE, "recipient_use": RECIPIENT_USE,
                            "destinations": ["internal-model"], "report_destination": "internal-report",
                            "required_disclosure": NOTICE, "required_tool": "log_compliance_creation",
                            "review": REVIEW_ACTION}
        columns = {column: {"kind": kind, "markdown": policy(column), "rule": {
            "destinations": ["internal-console"], "model": model_conditions,
            "additional_uses": [MODEL_USE, RECIPIENT_USE],
        }} for column, kind in (("name", "str"), ("value", "int"))}
        assignment_id = publisher.publish_assignment(root / "assignment", table="market_data",
            source_identity=source_identity(database), application=APPLICATION, actors=["analyst"],
            purposes=[PURPOSE], columns=columns)
        contexts = {}
        for role in ("gateway", "worker"):
            resolution = resolver(role)
            rules = AssignedRules(resolution, actor="analyst", application=APPLICATION, purpose=PURPOSE, writers=writers)
            contexts[role] = Context(sessions[role], EvidenceRecorder(sessions[role], store, audit_policy), rules,
                application=APPLICATION, purpose=PURPOSE, actor="analyst", resolver=resolution)
        context = contexts["worker"]
        db = GovernedSQL(database, gateway=contexts["gateway"], consumer=context,
                         edition_publisher=publisher, directory=root / "editions")
        name, value = db.execute("SELECT name, value FROM market_data LIMIT 1").fetchone()
        with context:
            amount = value + 250
            brief = name.upper() + ": " + amount.as_string() + " demonstration cents"
            internal = brief.reveal(context, destination="internal-console")
        print(f"Internal computation: {internal}; no model notice is required for this calculation.", flush=True)
        native_dir = root / "native-model"
        config_path, reader, requirements = prepare_native_model(brief, context=context,
            input_session=sessions["model-reader"], output_session=sessions["model-writer"],
            recipient_session=sessions["recipient"], input_config=paths["model-reader"],
            output_config=output_paths["model-writer"], directory=native_dir,
            model=args.model.resolve(), binary=args.binary.resolve(), translate=str if args.native_windows else wsl_path)
        # Authenticate requests to the local model process.
        api_key = uuid4().hex
        report = None
        try:
            print("Generating a governed report marked REVIEW REQUIRED, followed by a signed pending-review item.", flush=True)
            with governed_server(binary=args.binary.resolve(), model=args.model.resolve(), config=config_path,
                                 port=args.port, api_key=api_key, wsl=not args.native_windows) as base_url:
                endpoint = NativeModelEndpoint(context=context, value=brief, base_url=base_url, api_key=api_key,
                    output_reader=reader, directory=native_dir, requirements=requirements, runtime_config=config_path)
                with context:
                    report = endpoint.generate(brief, prompt=(
                        "Prepare a short internal HTML draft about the supplied synthetic calculation for a human reviewer. "
                        "Select the values section. State the supplied demonstration amount without treating it as a market quote. "
                        "Mark the report as requiring human review, using the exact carried review notice in a visible paragraph. "
                        "Generate the report now; do not claim a human has reviewed it. Use no CSS, hidden content, or external resources."))
            write_once(root / "report.html", report.report_html.encode())
            write_once(root / "review.json", json.dumps(report.review.to_dict(), indent=2).encode())
            print(f"Report generated. needs_review={report.needs_review}; status={report.review.status}; assignee={report.review.assignee}", flush=True)
        finally:
            tapes = {role: {**ctx.recorder.export(root / "tapes" / role), "path": "tapes/" + role}
                     for role, ctx in contexts.items()}
            manifest = {
                "schema": "tn.primitive-run.v1", "created_utc": stamp,
                "execution": "live Java Unity, SQLite, native TN, governed model report, and pending review workflow",
                "unity": {"url": args.unity_url, "catalog": catalog_name, "mode": "local development; authorization disabled"},
                "request": {"actor": "analyst", "application": APPLICATION, "purpose": PURPOSE,
                            "identity_source": "trusted local demonstration configuration"},
                "authority": authority, "writers": writers, "audit_policy": audit_policy.fields,
                "administration_policy": admin.fields, "assignment_id": assignment_id,
                "source": source_identity(database), "observer_config": "sessions/observer/tn.yaml", "tapes": tapes,
                "source_editions": db.editions, "sources": [handle(v) for v in db.published_values],
                "outputs": {"model_input": handle(brief), "amount": handle(amount)},
                "console_disclosures": {"internal": internal},
                "sql_executions": db.executions, "unity_resolutions": {r: c.resolver.requests for r, c in contexts.items()},
                "model": {"completed": report is not None, "required_disclosure": NOTICE,
                    "input_object_id": brief.reference.object_id, "output_object_id": report.object.id if report else None,
                    "writer": sessions["model-writer"].did, "use": MODEL_USE, "recipient_use": RECIPIENT_USE,
                    "destination": "internal-report", "recipient_config": "output-sessions/recipient/tn.yaml",
                    "model_reader_config": "sessions/model-reader/tn.yaml", "runtime_config": "native-model/runtime.json",
                    "runtime_config_sha256": digest(config_path), "model_path": str(args.model.resolve()),
                    "model_sha256": digest(args.model), "binary_path": str(args.binary.resolve()),
                    "binary_sha256": digest(args.binary), "implementation_sha256": digest(args.binary.parent / ("llama-server-impl.dll" if args.native_windows else "libllama-server-impl.so")),
                    "tool_object_ids": [o.id for o in report.release.tool_objects] if report else [],
                    "result_record_id": report.result_record_id if report else None,
                    "review": report.review.to_dict() if report and report.review else None},
                "runtime": {"python": platform.python_version(), "platform": platform.platform(), "tn_proto": sdk["version"],
                    "native_module_sha256": digest(native_module), "wheel_sha256": sdk["wheel_sha256"],
                    "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                                         text=True, capture_output=True, check=True).stdout.strip(),
                    "experiment_source_sha256": {"deepvest_tn/governed/" + p.name: digest(p)
                        for p in sorted(Path(__file__).parent.glob("*.py"))}},
                "scope": "Synthetic market data and demonstration policy.",
            }
            write_once(root / "run.json", json.dumps(manifest, indent=2).encode())
    print(f"Report: {root / 'report.html'}", flush=True)
    return root


if __name__ == "__main__":
    main()
