"""Live Unity + SQLite example. Run with the repository's pinned Python environment."""

import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
from uuid import uuid4

import tn
from tn.governed import NotEntitled
from deepvest_tn.publications import PublicationStore, write_once

from .catalog import AssignedRules, UnityGovernance
from .evidence import EvidenceRecorder
from .primitives import Context
from .numeric import TNNumeric
from .numeric_codec import record_value
from decimal import Decimal
from .catalog import markdown
from .provision import provision
from .sql import GovernedSQL, source_identity
from .string import OperationDenied


def handle(value):
    return {"reference": value.reference.to_dict(), "decision_id": value.decision_id,
            "result_id": value.result_id, "kind": type(value).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unity-url", default="http://127.0.0.1:8080")
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--numeric-type", choices=["int", "float", "Decimal"], default="int")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    sdk_record = repository / "config/sdk-build.json"
    sdk = json.loads(sdk_record.read_text())
    native_module = Path(tn.__file__).parent / "_native.pyd"
    native_digest = hashlib.sha256(native_module.read_bytes()).hexdigest()
    version = importlib.metadata.version("tn-proto")
    if version != sdk["version"] or native_digest != sdk["install"]["native_module_sha256"]:
        raise RuntimeError("the installed TN runtime does not match this repository's pinned build")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    root = (args.run_directory or Path("experiments/results") / ("governed-primitives-" + stamp)).resolve()
    root.mkdir(parents=True, exist_ok=False)
    print(f"Run artifacts: {root}", flush=True)
    database = root / "example.sqlite"
    column_type, samples = {
        "int": ("INTEGER", [21, 34]),
        "float": ("REAL", [21.0, 34.0]),
        "Decimal": ("DECIMAL_TEXT", ["21.00", "34.00"]),
    }[args.numeric_type]
    sqlite3.register_converter("DECIMAL_TEXT", lambda raw: Decimal(raw.decode("ascii")))
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(f"CREATE TABLE customer_data (name TEXT NOT NULL, value {column_type} NOT NULL)")
        connection.executemany("INSERT INTO customer_data VALUES (?, ?)", [("Ada", samples[0]), ("Grace", samples[1])])
        connection.commit()
    configs = provision(root / "sessions")
    store = PublicationStore(root / "publications")
    catalog_name = "primitives_" + uuid4().hex[:12]
    contexts = {}
    with ExitStack() as stack:
        sessions = {role: stack.enter_context(tn.Session.from_config(str(config)))
                    for role, config in configs.items() if role != "observer"}
        authority = sessions["catalog"].did
        administration = sessions["catalog"].policy("experiment.administration")
        audit_policy = sessions["catalog"].policy("experiment.evidence")
        writers = [sessions[role].did for role in ("gateway", "worker", "metadata_only")]
        def resolver(session):
            return UnityGovernance(args.unity_url, catalog_name, session=session,
                authority=authority, administration=administration, store=store)
        publisher = resolver(sessions["catalog"])
        assignment = publisher.publish_assignment(root / "assignment", table="customer_data",
            source_identity=source_identity(database), application="sql-demo",
            actors=["analyst"], purposes=["customer-support"], columns={
                "name": {"kind": "str", "markdown": markdown("name")},
                "value": {"kind": args.numeric_type, "markdown": markdown("value"),
                    "extend_markdown": markdown("value").replace("with monitoring", "with a signed pre-operation monitoring event")},
            })
        print(f"Java Unity catalog: {catalog_name}; signed column assignment: {assignment}", flush=True)

        def context(role, purpose="customer-support", label=None):
            session = sessions[role]
            resolution = resolver(session)
            rules = AssignedRules(resolution, actor="analyst", application="sql-demo",
                                  purpose=purpose, writers=writers)
            value = Context(session, EvidenceRecorder(session, store, audit_policy), rules,
                            actor="analyst", application="sql-demo", purpose=purpose, resolver=resolution)
            contexts[label or role] = value
            return value

        gateway, worker = context("gateway"), context("worker")
        driver = GovernedSQL(database, gateway=gateway, consumer=worker,
                             edition_publisher=publisher, directory=root / "editions", detect_types=sqlite3.PARSE_DECLTYPES)
        rows = driver.execute("SELECT name, value FROM customer_data LIMIT ?", (2,)).fetchall()
        name, value = rows[0]
        other_name, other_value = rows[1]
        print("SQL returned TNString / TNNumeric pairs; four source editions resolved through Unity.", flush=True)
        # These observations run without any ambient content-operation context.
        metadata = {"name_length": len(name), "name_equals_Ada": name == "Ada",
                    "name_hash_matches": hash(name) == hash("Ada"),
                    "value_equals_21": value == 21, "value_hash_matches": hash(value) == hash(21),
                    "string_subclass": isinstance(name, str), "numeric_handle": isinstance(value, TNNumeric), "numeric_type": value.dtype.__name__,
                    "name_fingerprint": name.content_fingerprint}
        with worker:
            annotated = name.attest("Suitable for the internal customer support greeting.", worker)
            upper = annotated.upper()
            prefix = annotated[:1]
            label = upper + " / " + prefix
            incremented = value + 4
            combined = incremented + other_value
            total = combined * 2
            report = label + " = " + total.as_string()
            copied = report.copy()
            metadata.update(copy_equal=copied == report, copy_same_incarnation=copied.same_incarnation(report),
                            protected_order=value < other_value)
            exposed = report.reveal(worker, destination="support-console")
            with worker.exposure("support-console"):
                exposed_total = total.reveal(worker, destination="support-console")
            refused_destination = None
            try:
                report.reveal(worker, destination="public-web")
            except OperationDenied as exc:
                refused_destination = str(exc)
        print(f"Authorized output: {exposed!r}; number: {exposed_total!r}", flush=True)

        # A different enrolled recipient can inspect declared facts but has no payload reader.
        metadata_only = context("metadata_only")
        metadata_only.resolver.resolve("customer_data", ["name", "value"], actor="analyst",
            application="sql-demo", purpose="customer-support", expected_source=source_identity(database))
        for edition in driver.editions:
            metadata_only.resolver.admit_edition(edition, metadata_only)
        capability_failure = None
        with metadata_only:
            readable_length = len(name)
            try:
                name.upper()
            except NotEntitled as exc:
                capability_failure = type(exc).__name__

        marketing = context("gateway", "marketing", "marketing")
        denied_driver = GovernedSQL(database, gateway=marketing, consumer=worker,
                                   edition_publisher=publisher, directory=root / "denied-editions")
        refused_source = None
        try:
            denied_driver.table("customer_data").select("name,value").limit(1).execute()
        except OperationDenied as exc:
            refused_source = str(exc)
        tapes = {}
        for label_, ctx in contexts.items():
            tapes[label_] = {**ctx.recorder.export(root / "tapes" / label_),
                             "path": "tapes/" + label_}
        objects = {"report": handle(report), "total": handle(total), "copy": handle(copied)}
        source_hashes = {p.relative_to(repository).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(Path(__file__).parent.glob("*.py"))}
        manifest = {
            "schema": "tn.primitive-run.v1", "created_utc": stamp,
            "execution": "live Java Unity and SQLite; native TN BTN encryption and signatures",
            "unity": {"url": args.unity_url, "catalog": catalog_name, "mode": "local development; authorization disabled"},
            "request": {"actor": "analyst", "application": "sql-demo", "purpose": "customer-support",
                        "identity_source": "explicit trusted local demonstration configuration"},
            "authority": authority, "writers": writers,
            "audit_policy": audit_policy.fields, "administration_policy": administration.fields,
            "assignment_id": assignment, "source": source_identity(database),
            "observer_config": "sessions/observer/tn.yaml", "tapes": tapes,
            "source_editions": driver.editions, "sources": [handle(v) for v in driver.published_values],
            "outputs": objects, "metadata": metadata,
            "console_disclosures": {"report": exposed, "total": exposed_total},
            "refusals": {"public_destination": refused_destination, "marketing_query": refused_source},
            "metadata_only_recipient": {"declared_length": readable_length,
                                        "compute_failure": capability_failure},
            "sql_executions": driver.executions, "marketing_sql_executions": denied_driver.executions,
            "unity_resolutions": {name_: ctx.resolver.requests for name_, ctx in contexts.items()},
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "tn_proto": version, "native_module_sha256": native_digest,
                        "wheel_sha256": sdk["wheel_sha256"], "sdk_source": sdk["source"],
                        "sdk_build_record_sha256": hashlib.sha256(sdk_record.read_bytes()).hexdigest(),
                        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                            check=True, text=True, capture_output=True).stdout.strip(),
                        "experiment_source_sha256": source_hashes},
            "trust_note": "Keep this manifest's endpoint identities and tape heads independently. It is an index, not a self-authenticating trust root.",
        }
        write_once(root / "run.json", json.dumps(record_value(manifest), indent=2).encode())
    print(f"Evidence ready for an independent, offline recipient: {root / 'run.json'}", flush=True)
    return root


if __name__ == "__main__":
    main()
