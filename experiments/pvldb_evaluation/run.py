"""Run one measured SQL-to-report case without retrying a native request.

Both modes share SQL assignment, the governed dataframe with real OPA, native
publications, model service, recipient, and pending-review workflow. They differ
only in downstream scalar dispatch: overloaded primitives or explicit callbacks.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import time
from uuid import uuid4

import tn
import tn_llama_client
import tn_polars

import deepvest_tn.governed as governed
from deepvest_tn.native_run import governed_server, wsl_path
from deepvest_tn.publications import open_publication_store, write_once
from deepvest_tn.governed.market_report import (
    APPLICATION, MODEL_USE, NOTICE, PURPOSE, RECIPIENT_USE, REVIEW_ACTION, policy,
)
from deepvest_tn.governed.catalog import AssignedRules
from deepvest_tn.governed.demo import handle
from deepvest_tn.governed.evidence import EvidenceRecorder
from deepvest_tn.governed.model import NativeModelEndpoint, digest, prepare_native_model
from deepvest_tn.governed.provision import provision
from deepvest_tn.governed.sql import GovernedSQL, source_identity
from .frames import DataframeContext, DataframeGovernance, dataframe_calculate
from .metrics import stage as profile_stage


CLAIM = "Suitable for identifying the synthetic company in this internal demonstration report."
PROMPT = (
    "Prepare a short internal HTML draft about the supplied synthetic calculation for a human reviewer. "
    "Select the values section. State the supplied demonstration amount without treating it as a market quote. "
    "Mark the report as requiring human review, using the exact carried review notice in a visible paragraph. "
    "Generate the report now; do not claim a human has reviewed it. Use no CSS, hidden content, or external resources."
)


@contextmanager
def _timed(timings, name):
    with profile_stage(name):
        started = time.perf_counter()
        try:
            yield
        finally:
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started


def _error(exc, stage):
    # Exception strings can contain credentials or configuration data.
    return {"stage": stage, "type": type(exc).__name__, "native_code": getattr(exc, "code", None)}


def _adapter_sources(repository):
    """Identify the actual editable adapter and installed transport sources."""
    roots = {
        "deepvest_tn": repository / "src/deepvest_tn",
        "tn_polars": Path(tn_polars.__file__).resolve().parent,
        "tn_llama_client": Path(tn_llama_client.__file__).resolve().parent,
    }
    return {f"{name}/{path.relative_to(folder).as_posix()}": digest(path)
            for name, folder in roots.items() for path in sorted(folder.rglob("*.py"))}


class _Operations:
    """Keep the operation schedule equal while varying developer dispatch."""

    def __init__(self, context, mode):
        self.explicit = None
        if mode == "explicit":
            from .explicit import ExplicitOperations
            self.explicit = ExplicitOperations(context)

    def upper(self, value):
        return (self.explicit.execute("upper", (value,), lambda text: text.upper())
                if self.explicit else value.upper())

    def prefix(self, value):
        return (self.explicit.execute("slice", (value,), lambda text: text[:1],
                                      parameters=(None, 1, None))
                if self.explicit else value[:1])

    def add(self, left, right):
        return (self.explicit.execute("add", (left, right), lambda a, b: a + b)
                if self.explicit else left + right)

    def add_literal(self, value, literal):
        return (self.explicit.execute("add", (value,), lambda number: number + literal,
                    parameters=(literal, "right"),
                    literals=({"parameter_index": 0, "position": "right", "claims": []},))
                if self.explicit else value + literal)

    def concat(self, left, right):
        return (self.explicit.execute("concat", (left, right), lambda a, b: a + b)
                if self.explicit else left + right)

    def concat_literal(self, value, literal):
        return (self.explicit.execute("concat", (value,), lambda text: text + literal,
                    parameters=(literal, "right"),
                    literals=({"parameter_index": 0, "position": "right", "claims": []},))
                if self.explicit else value + literal)

    def as_string(self, value):
        return (self.explicit.execute("to_string", (value,), str)
                if self.explicit else value.as_string())


def run_case(root, mode, rows=1, depth=1, branches=1, model=False, claim=True,
             unity_url="http://127.0.0.1:8080", binary=None, model_path=None,
             port=18325, numeric_type="int", decimal_precision=38, decimal_scale=4, storage="sqlite",
             sdk_build_record=None):
    """Return a case summary and preserve the verifier's run manifest.

    A model attempt uses one fresh server and one native request. Failures remain
    failures; no retry, fallback, or fabricated report is performed here.
    """
    if mode not in {"primitive", "explicit"}:
        raise ValueError("mode must be primitive or explicit")
    if any(type(n) is not int or n < 1 for n in (rows, branches)):
        raise ValueError("rows and branches must be positive integers")
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    repository = Path(__file__).resolve().parents[2]
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    binary = Path(binary or repository / ".runtime/dependencies/tn-llama/llama-server").resolve()
    model_path = Path(model_path or repository / ".artifacts/models/Qwen3-4B-Q4_K_M.gguf").resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timings = {}
    summary = {
        "schema": "tn.pvldb-case.v1", "run_directory": str(root), "created_utc": stamp,
        "mode": mode, "rows": rows, "depth": depth, "branches": branches,
        "with_model": bool(model), "with_claim": bool(claim), "completed": False,
        "verified": False, "timings_seconds": timings, "error": None,
        "publication_storage": storage,
    }
    if numeric_type not in {"int", "float", "Decimal"}:
        raise ValueError("unsupported source numeric type")
    raw_values = [1250 + i for i in range(rows)]
    if numeric_type == "float":
        raw_values = [float(value) + 0.125 for value in raw_values]
    elif numeric_type == "Decimal":
        raw_values = [Decimal(f"{value}.1250") for value in raw_values]
    expected_amount = sum(value + 250 for value in raw_values) + depth
    summary["numeric_type"] = numeric_type
    expected_label = "ACME" + " / A" * (branches - 1)
    summary["expected_content"] = f"{expected_label}: {expected_amount} demonstration cents"
    write_once(root / "case-request.json", json.dumps({k: v for k, v in summary.items()
               if k not in {"timings_seconds", "error"}}, indent=2).encode())
    total_started = time.perf_counter()
    stage = "preflight"
    try:
        build_record = Path(sdk_build_record or repository / "config/sdk-build.json")
        build_bytes = build_record.read_bytes()
        sdk = json.loads(build_bytes)
        native_module = Path(tn.__file__).parent / "_native.pyd"
        if (importlib.metadata.version("tn-proto") != sdk["version"]
                or digest(native_module) != sdk["install"]["native_module_sha256"]):
            raise RuntimeError("installed TN runtime differs from the selected build record")
        write_once(root / "sdk-build.json", build_bytes)
        if model and (not binary.is_file() or not model_path.is_file()):
            raise FileNotFoundError("configured native model binary or GGUF is missing")
        if mode == "explicit":
            from .explicit import ExplicitOperations  # noqa: F401 - Validate the adapter before provisioning.

        stage = "setup"
        with _timed(timings, stage):
            database = root / "market.sqlite"
            sqlite3.register_converter("DECIMAL_TEXT", lambda raw: Decimal(raw.decode("ascii")))
            column_type = {"int": "INTEGER", "float": "REAL", "Decimal": "DECIMAL_TEXT"}[numeric_type]
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(f"CREATE TABLE market_data (name TEXT NOT NULL, value {column_type} NOT NULL)")
                connection.executemany("INSERT INTO market_data VALUES (?, ?)",
                                       [("ACME" if i == 0 else f"ACME-{i}", str(value) if numeric_type == "Decimal" else value) for i, value in enumerate(raw_values)])
                connection.commit()
            paths = provision(root / "sessions", roles=["catalog", "gateway", "worker", "model-reader", "observer"],
                              read_only_roles=("model-reader", "observer"))
            output_paths = provision(root / "output-sessions", roles=["model-writer", "recipient"],
                                     groups=["tn.agents", "default"], read_only_roles=("recipient",))
            store = open_publication_store(root / "publications", backend=storage)
        with ExitStack() as stack:
            stack.enter_context(store)
            sessions = {role: stack.enter_context(tn.Session.from_config(str(path)))
                        for role, path in {**paths, **output_paths}.items() if role != "observer"}
            authority = sessions["catalog"].did
            admin = sessions["catalog"].policy("experiment.administration")
            audit_policy = sessions["catalog"].policy("experiment.evidence")
            writers = [sessions[r].did for r in ("gateway", "worker")]
            catalog_name = "pvldb_" + uuid4().hex[:12]

            def resolver(role):
                return DataframeGovernance(unity_url, catalog_name, session=sessions[role],
                                       authority=authority, administration=admin, store=store)

            with _timed(timings, "assignment"):
                publisher = resolver("catalog")
                model_conditions = {"use": MODEL_USE, "recipient_use": RECIPIENT_USE,
                    "destinations": ["internal-model"], "report_destination": "internal-report",
                    "required_disclosure": NOTICE, "required_tool": "log_compliance_creation", "review": REVIEW_ACTION}
                columns = {column: {"kind": kind, "markdown": policy(column), "rule": {
                    "destinations": ["internal-console"], "model": model_conditions,
                    "additional_uses": [MODEL_USE, RECIPIENT_USE],
                }} for column, kind in (("name", "str"), ("value", numeric_type))}
                assignment_id = publisher.publish_assignment(root / "assignment", table="market_data",
                    source_identity=source_identity(database), application=APPLICATION,
                    actors=["analyst"], purposes=[PURPOSE], columns=columns,
                    decimal_precision=decimal_precision, decimal_scale=decimal_scale)
                contexts = {}
                for role in ("gateway", "worker"):
                    resolution = resolver(role)
                    rules = AssignedRules(resolution, actor="analyst", application=APPLICATION,
                                          purpose=PURPOSE, writers=writers)
                    contexts[role] = DataframeContext(sessions[role], EvidenceRecorder(sessions[role], store, audit_policy),
                        rules, application=APPLICATION, purpose=PURPOSE, actor="analyst", resolver=resolution)
            context = contexts["worker"]
            db = GovernedSQL(database, gateway=contexts["gateway"], consumer=context,
                             edition_publisher=publisher, directory=root / "editions", detect_types=sqlite3.PARSE_DECLTYPES)
            outputs, internal, report, config_path = {}, None, None, None
            requirements = None
            try:
                stage = "query"
                with _timed(timings, stage):
                    fetched = db.execute("SELECT name, value FROM market_data LIMIT ?", (rows,)).fetchall()
                    if len(fetched) != rows:
                        raise RuntimeError("SQL returned a different row count")
                stage = "claim"
                with _timed(timings, stage), context:
                    name = fetched[0][0]
                    if claim:
                        name = name.attest(CLAIM, context)
                        outputs["attested_name"] = handle(name)
                frame_rows = [(name, fetched[0][1]), *fetched[1:]]
                stage = "dataframe"
                with _timed(timings, stage), context:
                    amount = dataframe_calculate(context, frame_rows)
                    outputs["dataframe_result"] = handle(amount)
                    summary["dataframe"] = context.frame_runs
                stage = "compute"
                with _timed(timings, stage), context:
                    ops = _Operations(context, mode)
                    for _ in range(depth):
                        amount = ops.add_literal(amount, 1)
                    labels = [ops.upper(name)] + [ops.prefix(name) for _ in range(branches - 1)]
                    label = labels[0]
                    for part in labels[1:]:
                        label = ops.concat(ops.concat_literal(label, " / "), part)
                    brief = ops.concat_literal(ops.concat(ops.concat_literal(label, ": "),
                                               ops.as_string(amount)), " demonstration cents")
                    outputs.update(model_input=handle(brief), amount=handle(amount))
                stage = "console_disclosure"
                with _timed(timings, stage):
                    internal = brief.reveal(context, destination="internal-console")
                    if internal != summary["expected_content"]:
                        raise RuntimeError("application calculation differs from its declared workload")
                if model:
                    native_dir = root / "native-model"
                    stage = "native_prepare"
                    with _timed(timings, stage):
                        config_path, reader, requirements = prepare_native_model(brief, context=context,
                            input_session=sessions["model-reader"], output_session=sessions["model-writer"],
                            recipient_session=sessions["recipient"], input_config=paths["model-reader"],
                            output_config=output_paths["model-writer"], directory=native_dir,
                            model=model_path, binary=binary, translate=wsl_path)
                    api_key = uuid4().hex
                    stage = "server_start"
                    with ExitStack() as model_stack:
                        with _timed(timings, stage):
                            base_url = model_stack.enter_context(governed_server(
                                binary=binary, model=model_path, config=config_path,
                                port=port, api_key=api_key, wsl=True))
                        endpoint = NativeModelEndpoint(context=context, value=brief, base_url=base_url,
                            api_key=api_key, output_reader=reader, directory=native_dir,
                            requirements=requirements, runtime_config=config_path)
                        stage = "generation"
                        with _timed(timings, stage), context:
                            report = endpoint.generate(brief, prompt=PROMPT)
                        write_once(root / "report.html", report.report_html.encode())
                        write_once(root / "review.json", json.dumps(report.review.to_dict(), indent=2).encode())
                summary["completed"] = True
            except Exception as exc:
                summary["error"] = _error(exc, stage)
            finally:
                records = [context.recorder.read(identity) for identity in context.recorder._ids]
                handoffs = [r for r in records if r.body["kind"] == "ModelHandoff"]
                tapes = {role: {**ctx.recorder.export(root / "tapes" / role), "path": "tapes/" + role}
                         for role, ctx in contexts.items()}
                source_hashes = {
                    "experiments/pvldb_evaluation/" + path.name: digest(path)
                    for path in sorted(Path(__file__).parent.glob("*.py"))
                }
                source_hashes.update({
                    "deepvest_tn/governed/" + path.name: digest(path)
                    for path in sorted(Path(governed.__file__).parent.glob("*.py"))
                })
                commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                        text=True, capture_output=True, check=True).stdout.strip()
                manifest = {
                    "schema": "tn.primitive-run.v1", "created_utc": stamp,
                    "execution": "live Java Unity, SQLite, governed Polars with OPA and native TN" + (
                        ", governed model report and pending review" if model else ", scalar computation"),
                    "unity": {"url": unity_url, "catalog": catalog_name,
                              "mode": "local development; authorization disabled"},
                    "request": {"actor": "analyst", "application": APPLICATION, "purpose": PURPOSE,
                                "identity_source": "trusted local demonstration configuration"},
                    "authority": authority, "writers": writers, "audit_policy": audit_policy.fields,
                    "administration_policy": admin.fields, "assignment_id": assignment_id,
                    "source": source_identity(database), "observer_config": "sessions/observer/tn.yaml",
                    "tapes": tapes, "source_editions": db.editions,
                    "sources": [handle(v) for v in db.published_values], "outputs": outputs,
                    "console_disclosures": {"internal": internal}, "sql_executions": db.executions,
                    "unity_resolutions": {role: ctx.resolver.requests for role, ctx in contexts.items()},
                    "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "tn_proto": sdk["version"], "native_module_sha256": digest(native_module),
                        "wheel_sha256": sdk["wheel_sha256"], "repository_commit": commit,
                        "dependency_versions": {name: importlib.metadata.version(name) for name in
                            ("polars", "tn-polars", "tn-llama-client", "wasmtime")},
                        "sdk_build_record": "sdk-build.json",
                        "sdk_build_record_sha256": digest(root / "sdk-build.json"),
                        "uv_lock_sha256": digest(repository / "uv.lock"),
                        "pyproject_sha256": digest(repository / "pyproject.toml"),
                        "adapter_source_sha256": _adapter_sources(repository),
                        "experiment_source_sha256": source_hashes},
                    "evaluation": {"mode": mode, "rows": rows, "depth": depth, "branches": branches,
                                   "publication_storage": storage,
                                   "claim": bool(claim), "requested_model": bool(model),
                                   "error": summary["error"], "expected_content": summary["expected_content"],
                                   "comparison_scope": "Shared governed dataframe; automatic versus explicit downstream scalar propagation",
                                   "dataframe": context.frame_runs},
                }
                if handoffs:
                    manifest["model"] = {
                        "completed": report is not None, "required_disclosure": NOTICE,
                        "input_object_id": brief.reference.object_id,
                        "output_object_id": report.object.id if report else None,
                        "writer": sessions["model-writer"].did, "use": MODEL_USE,
                        "recipient_use": RECIPIENT_USE, "destination": "internal-report",
                        "recipient_config": "output-sessions/recipient/tn.yaml",
                        "model_reader_config": "sessions/model-reader/tn.yaml",
                        "runtime_config": "native-model/runtime.json", "runtime_config_sha256": digest(config_path),
                        "model_path": str(model_path), "model_sha256": digest(model_path),
                        "binary_path": str(binary), "binary_sha256": digest(binary),
                        "implementation_sha256": digest(binary.parent / "libllama-server-impl.so"),
                        "tool_object_ids": [obj.id for obj in report.release.tool_objects] if report else [],
                        "result_record_id": report.result_record_id if report else None,
                        "review": report.review.to_dict() if report and report.review else None,
                    }
                write_once(root / "run.json", json.dumps(manifest, indent=2).encode())
        stage = "verification"
        from deepvest_tn.governed.verify import verify
        from .verify_frame import verify_frame
        try:
            with _timed(timings, stage):
                verification = verify(root)
                if outputs.get("dataframe_result"):
                    summary["dataframe_verification"] = verify_frame(root)
                if claim and outputs.get("model_input"):
                    graph = json.loads((root / "dag.json").read_text())
                    final_node = next(n for n in graph["nodes"] if n["id"] == outputs["model_input"]["reference"]["object_id"])
                    if not final_node["claims"]:
                        raise RuntimeError("required original claim is absent from the model input")
                    if verification.get("model", {}).get("completed") and not verification["model"]["source_claim_ids"]:
                        raise RuntimeError("required original claim is absent from model result evidence")
            summary["verified"] = True
            summary["verification"] = {k: verification[k] for k in
                ("record_count", "record_kinds", "incarnations", "derivation_edges", "source_editions", "policy_revisions")}
            if "model" in verification:
                summary["model"] = verification["model"]
        except Exception as exc:
            summary["verification_error"] = _error(exc, stage)
            summary["completed"] = False
    except Exception as exc:
        summary["error"] = _error(exc, stage)
        summary["completed"] = False
    finally:
        timings["total"] = time.perf_counter() - total_started
        write_once(root / "case-summary.json", json.dumps(summary, indent=2).encode())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-type", choices=["int", "float", "Decimal"], default="int")
    parser.add_argument("--storage", choices=["files", "sqlite"], default="sqlite",
                        help="Publication storage (default: sqlite; files retains separate-file storage)")
    parser.add_argument("--decimal-precision", type=int, default=38)
    parser.add_argument("--decimal-scale", type=int, default=4)
    parser.add_argument("--mode", choices=("primitive", "explicit"), required=True)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--branches", type=int, default=1)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--without-claim", action="store_true")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--unity-url", default="http://127.0.0.1:8080")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--port", type=int, default=18325)
    parser.add_argument("--sdk-build-record", type=Path,
                        help="Expected native build record (default: repository's pinned build)")
    args = parser.parse_args()
    summary = run_case(args.run_directory, args.mode, rows=args.rows, depth=args.depth,
        branches=args.branches, model=args.with_model, claim=not args.without_claim,
        unity_url=args.unity_url, binary=args.binary, model_path=args.model_path, port=args.port, numeric_type=args.numeric_type,
        decimal_precision=args.decimal_precision, decimal_scale=args.decimal_scale, storage=args.storage,
        sdk_build_record=args.sdk_build_record)
    print(json.dumps(summary, indent=2))
    return 0 if summary["completed"] and summary["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
