"""Connected native TN workflow with explicit service sessions and exact releases."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import tn
import yaml
from tn._native.btn import PublisherState
from tn.signing import DeviceKey
from tn_llama_client.clients import NativeOutputReader
from tn_polars.policy import CompiledPolicy

from .calculation_client import EngineCalculationPort
from .feed_server import FeedServerConfig, create_feed_server
from .getdata import GetDataConfig
from .llama import _LLAMA_SOURCE_REVISION, submit_governed_report
from .pipeline import run_governed_calculation_path
from .publications import PublicationStore, write_once
from .unity import (
    CatalogTrust,
    FinanceEditionRequest,
    UnityCatalogConfig,
    originate_policy_revision,
    publish_dataset_edition,
    resolve_finance_editions,
    unity_catalog_registry,
)

POLARS_USE = tn.UseContext("analytics", "portfolio_analysis", "prepare_wealth")
POLARS_RELEASE = tn.UseContext("analytics", "portfolio_analysis", "release_calculation")
CALCULATION_USE = tn.UseContext("calculation", "portfolio_analysis", "calculate")
CALCULATION_RELEASE = tn.UseContext("calculation", "portfolio_analysis", "release_calculation")
MODEL_USE = tn.UseContext("tn-llama", "portfolio_analysis", "generate_report")
RECIPIENT_USE = tn.UseContext("reporting", "portfolio_analysis", "read_report")
METADATA_USE = tn.UseContext("catalog", "metadata_resolution", "read")
DESTINATION = "reporting"
POLICY = """## portfolio.feed.data.v1
### instruction
Prepare approved portfolio analysis from supplied observations.
### use_for
Calculate maximum drawdown and prepare an internal report of the supplied result.
### do_not_use_for
Individual disclosure, public aggregate publication, or investment recommendations.
### consequences
Contract review.
### on_violation_or_error
Refuse release.
"""
REPORT_PROMPT = (
    "Select the finance section. Write a short HTML report of its calculation. "
    "State the function, numerical result, units, and method from the supplied result. "
    "Explain that a negative maximum drawdown records the fall from an earlier peak. "
    "Use the supplied calculation; do not calculate a different metric or recommend a trade."
)


def _use_dict(use):
    return {name: getattr(use, name) for name in ("application", "purpose", "operation")}


def _digest_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def provision_reference_sessions(root: Path) -> dict[str, Path]:
    """Provision the local example once; processing receives existing sessions."""
    root.mkdir(parents=True, exist_ok=False)
    paths = {}
    for namespace, names, roles in [
        (
            "source",
            ["finance", "policy_revision", "dataset_edition", "tn.agents"],
            [("source", True), ("polars", True), ("calculation", True), ("model-reader", False)],
        ),
        ("report", ["default", "tn.agents"], [("model-writer", True), ("recipient", False)]),
    ]:
        index = os.urandom(32)
        materials = {}
        for group in names:
            state = PublisherState()
            materials[group] = (state.mint(), state.to_bytes())
        ceremony = f"cer_native_{namespace}_{uuid.uuid4().hex}"
        for role, publish in roles:
            directory = root / role
            keys = directory / "keys"
            kits = directory / "readers"
            keys.mkdir(parents=True)
            kits.mkdir()
            policy_dir = directory / ".tn" / "config"
            policy_dir.mkdir(parents=True)
            (policy_dir / "agents.md").write_text(POLICY, encoding="utf-8")
            device = DeviceKey.generate()
            (keys / "local.private").write_bytes(device.private_bytes)
            (keys / "index_master.key").write_bytes(index)
            for group, (kit, state) in materials.items():
                (keys / f"{group}.btn.mykit").write_bytes(kit)
                (kits / f"{group}.btn.mykit").write_bytes(kit)
                if publish:
                    (keys / f"{group}.btn.state").write_bytes(state)
            config = {
                "ceremony": {"id": ceremony, "cipher": "btn", "mode": "local"},
                "keystore": {"path": "keys"},
                "device": {"device_identity": device.did},
                "logs": {"path": "register.ndjson"},
                "groups": {
                    group: {"cipher": "btn", "policy": "private", "index_epoch": 0}
                    for group in names
                },
            }
            path = directory / "tn.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            paths[role] = path
    return paths


@dataclass
class PreparedReference:
    directory: Path
    sessions: dict
    paths: dict
    stack: ExitStack
    catalog: tn.DatasetCatalog
    policy_dag: tn.PolicyDag
    store: PublicationStore
    editions: tuple
    calculation: object
    metadata_path: Path
    preparation_path: Path

    def close(self):
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def prepare_governed_reference(directory: Path, *, unity_url: str, registry=None):
    """Resolve two signed editions and run OPA, Polars, and the calculation engine."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    paths = provision_reference_sessions(directory / "sessions")
    stack = ExitStack()
    try:
        sessions = {
            role: stack.enter_context(tn.Session.from_config(path)) for role, path in paths.items()
        }
        source = sessions["source"]
        trust = CatalogTrust(
            lambda ctx: ctx.writer == source.did,
            lambda revision, parent: revision.writer == source.did,
            lambda edition: edition.writer == source.did,
            METADATA_USE,
        )
        origin_dag, dag = tn.PolicyDag(), tn.PolicyDag()
        store = PublicationStore(directory / "publications")
        catalog_name = "native_" + uuid.uuid4().hex[:12]
        if registry is None:
            registry = unity_catalog_registry(
                UnityCatalogConfig(unity_url, catalog_name, "finance")
            )
        administration = source.policy("portfolio.feed.data.v1")
        revisions, published = [], []
        rows_by_dataset = {
            "wealth.observations": [
                {"observation_id": i, "wealth": value}
                for i, value in [(4, 110.0), (1, 100.0), (3, 90.0), (2, 120.0)]
            ],
            "valuation.calendar": [
                {"observation_id": day, "observed_at": f"2026-09-0{day}T16:00:00+00:00"}
                for day in range(1, 5)
            ],
        }
        for dataset, rows in rows_by_dataset.items():
            revision, policy = originate_policy_revision(
                session=source,
                policy_dag=origin_dag,
                markdown=POLICY.replace("supplied observations", dataset),
                policy_id=dataset,
                object_type="portfolio.feed.data.v1",
                scope=dataset,
                administration_contract=administration,
                trust=trust,
            )
            revisions.append(revision)
            published.append(
                publish_dataset_edition(
                    dataset=dataset,
                    edition="close-2026-09-04",
                    rows=rows,
                    compiled_policy=CompiledPolicy.from_dict({
                        "schema": "tn.polars-policy.v1",
                        "policy_ref": policy.policy_ref,
                        "fields": {name: "confidential" for name in rows[0]},
                        "release": {
                            "public": {"allow_aggregates": False, "min_group_size": 2},
                        },
                    }),
                    session=source,
                    policy=policy,
                    policy_dag=origin_dag,
                    revisions=[revision],
                    eligible_uses=[POLARS_USE, CALCULATION_USE, MODEL_USE, RECIPIENT_USE],
                    grant_ref=f"internal-analysis:{dataset}",
                    administration_contract=administration,
                    trust=trust,
                    publish_use=tn.UseContext("source", "portfolio_analysis", "publish"),
                    publish_decide=lambda _: True,
                    destination="analytics",
                    publication_store=store,
                    output_dir=directory / "editions" / dataset,
                    registry=registry,
                    allowed_operations=["load", "join", "select", "sort", "release"],
                    audiences={
                        "calculation": "confidential",
                        "tn-llama": "confidential",
                        "reporting": "confidential",
                    },
                )
            )
        editions = resolve_finance_editions(
            [item.request(POLARS_USE) for item in published],
            catalog_session=source,
            policy_dag=dag,
            trust=trust,
            registry=registry,
            metadata_store=store,
        )
        with (directory / "source-http.ndjson").open("w", encoding="utf-8") as events:
            server = create_feed_server(
                FeedServerConfig(directory, events, 0, publication_store=store)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address[:2]
                calculation = run_governed_calculation_path(
                    wealth_edition=editions[0],
                    calendar_edition=editions[1],
                    getdata=GetDataConfig(f"http://{host}:{port}"),
                    polars_session=sessions["polars"],
                    calculation_session=sessions["calculation"],
                    policy_dag=dag,
                    publication_store=store,
                    calculation=EngineCalculationPort(),
                    polars_use=POLARS_USE,
                    polars_release_use=POLARS_RELEASE,
                    calculation_use=CALCULATION_USE,
                    calculation_release_use=CALCULATION_RELEASE,
                    decide_source=lambda ctx: ctx.writer == source.did,
                    decide_polars_release=lambda _: True,
                    decide_calculation_receive=lambda _: True,
                    decide_calculation_release=lambda _: True,
                    to="tn-llama",
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
        metadata = {
            "policy_revisions": [item.object.wire for item in revisions],
            "dataset_editions": [item.record.object.wire for item in published],
            "publications": [
                *(item.source.wire for item in published),
                calculation.polars_object.wire,
                calculation.calculation_object.wire,
            ],
        }
        metadata_path = directory / "metadata.json"
        write_once(metadata_path, json.dumps(metadata, ensure_ascii=False).encode())
        write_once(directory / "model-input.tn", calculation.calculation_object.forward())
        facts = {
            "profile": "native-governed-reference.v1",
            "unity_url": unity_url,
            "registry": type(registry).__name__,
            "unity_catalog": catalog_name,
            "editions": [
                {
                    "dataset": item.selection.dataset,
                    "edition": item.selection.edition,
                    "record_id": item.selection.edition_record_id,
                    "source_object_id": item.selection.source_object_id,
                }
                for item in editions
            ],
            "polars_object_id": calculation.polars_object.id,
            "calculation_object_id": calculation.calculation_object.id,
            "model_input_wire_sha256": hashlib.sha256(
                calculation.calculation_object.forward()
            ).hexdigest(),
            "calculation": asdict(calculation.execution.result),
            "sessions": {role: session.did for role, session in sessions.items()},
            "source_http_requests": [asdict(fact) for fact in calculation.feed_fetch_facts],
        }
        preparation = directory / "preparation.json"
        write_once(preparation, json.dumps(facts, indent=2).encode())
        return PreparedReference(
            directory,
            sessions,
            paths,
            stack,
            editions[0].catalog,
            dag,
            store,
            editions,
            calculation,
            metadata_path,
            preparation,
        )
    except BaseException:
        stack.close()
        raise


def write_governed_runtime_config(
    prepared,
    *,
    model: Path,
    binary: Path,
    path_for_runtime=str,
    llama_source_revision=_LLAMA_SOURCE_REVISION,
):
    """Configure the native gate from the exact metadata admitted during preparation."""
    directory = prepared.directory
    (directory / "model-traces").mkdir(exist_ok=True)
    source_path = prepared.paths["model-reader"]
    output_path = prepared.paths["model-writer"]
    source_config = yaml.safe_load(source_path.read_text())
    output_config = yaml.safe_load(output_path.read_text())
    policies = (
        prepared.sessions["calculation"]
        .governance(prepared.calculation.calculation_object)
        .governance.policies
    )
    config = {
        "version": "tn.llama.runtime.v2",
        "writer_authority": "group_capability",
        "expected_event_type": "portfolio.calculation.calculated",
        "allowed_input_groups": ["finance"],
        "governed_objects": {
            "native_reader_config_path": path_for_runtime(source_path),
            "metadata_store_path": path_for_runtime(prepared.metadata_path),
            "use_context": _use_dict(MODEL_USE),
            "destinations": [DESTINATION],
            "trusted_writers": [
                prepared.sessions[role].did for role in ["source", "polars", "calculation"]
            ],
            "authority_writers": [prepared.sessions["source"].did],
            "contracts": [
                {
                    "revision_id": policy.revision_id,
                    "scope": prepared.policy_dag.get(policy.revision_id).scope,
                    "allowed_uses": [_use_dict(MODEL_USE)],
                }
                for policy in policies
            ],
        },
        "report_contract": {
            "allowed_input_sections": ["finance"],
            "output_event_type": "report.generated",
            "output_public_section": "default",
            "operation": MODEL_USE.operation,
            "section_rules": {
                "finance": {
                    "propagate_source_governance": True,
                    "required_attribution": None,
                    "required_tool": None,
                }
            },
            # Required by the shared version 1 configuration fields. Version 2
            # inherits native policies; it never constructs governance from this block.
            "output_policy": {
                name: policies[0].fields[name]
                for name in [
                    "instruction",
                    "use_for",
                    "do_not_use_for",
                    "consequences",
                    "on_violation_or_error",
                ]
            },
        },
        "reader_key_dir": path_for_runtime(source_path.parent / "readers"),
        "input_index_key_path": path_for_runtime(source_path.parent / "keys" / "index_master.key"),
        "ceremony_id": source_config["ceremony"]["id"],
        "governance_group": "tn.agents",
        "input_index_epoch": 0,
        "output_tn_config_path": path_for_runtime(output_path),
        "output_reader_key_dir": path_for_runtime(output_path.parent / "readers"),
        "output_index_key_path": path_for_runtime(output_path.parent / "keys" / "index_master.key"),
        "output_ceremony_id": output_config["ceremony"]["id"],
        "output_governance_group": "tn.agents",
        "output_index_epoch": 0,
        "audit_path": path_for_runtime(directory / "model-audit.ndjson"),
        "raw_trace_dir": path_for_runtime(directory / "model-traces"),
        "build_id": "native-governed-reference",
        "llama_source_revision": llama_source_revision,
        "model_sha256": _digest_file(model),
        "binary_sha256": _digest_file(binary),
        "max_request_bytes": 1024 * 1024,
        "max_message_bytes": 16 * 1024,
        "max_active_sessions": 1,
        "max_chain_namespaces": 8,
        "echo_max_tokens": 8192,
        "final_max_tokens": 2048,
    }
    path = directory / "runtime.json"
    write_once(path, json.dumps(config, indent=2).encode())
    return path


def load_governed_reference(directory: Path, *, registry=None):
    """Reopen retained publications and resolve the same editions after a restart."""
    directory = Path(directory).resolve()
    facts = json.loads((directory / "preparation.json").read_bytes())
    paths = {role: directory / "sessions" / role / "tn.yaml" for role in facts["sessions"]}
    stack = ExitStack()
    try:
        sessions = {
            role: stack.enter_context(tn.Session.from_config(path)) for role, path in paths.items()
        }
        if {role: session.did for role, session in sessions.items()} != facts["sessions"]:
            raise ValueError("reference session identity changed after preparation")
        source = sessions["source"]
        trust = CatalogTrust(
            lambda ctx: ctx.writer == source.did,
            lambda revision, parent: revision.writer == source.did,
            lambda edition: edition.writer == source.did,
            METADATA_USE,
        )
        dag = tn.PolicyDag()
        store = PublicationStore(directory / "publications")
        if registry is None:
            registry = unity_catalog_registry(
                UnityCatalogConfig(facts["unity_url"], facts["unity_catalog"], "finance")
            )
        editions = resolve_finance_editions(
            [
                FinanceEditionRequest(
                    item["dataset"], item["edition"], item["record_id"], POLARS_USE
                )
                for item in facts["editions"]
            ],
            catalog_session=source,
            policy_dag=dag,
            trust=trust,
            registry=registry,
            metadata_store=store,
        )
        calculation = tn.GovernedObject.read(io.BytesIO(store.get(facts["calculation_object_id"])))
        polars = tn.GovernedObject.read(io.BytesIO(store.get(facts["polars_object_id"])))
        if (directory / "model-input.tn").read_bytes() != calculation.forward() or hashlib.sha256(
            calculation.forward()
        ).hexdigest() != facts["model_input_wire_sha256"]:
            raise ValueError("retained model input differs from the prepared release")
        return PreparedReference(
            directory,
            sessions,
            paths,
            stack,
            editions[0].catalog,
            dag,
            store,
            editions,
            SimpleNamespace(calculation_object=calculation, polars_object=polars),
            directory / "metadata.json",
            directory / "preparation.json",
        )
    except BaseException:
        stack.close()
        raise


def finish_governed_reference(prepared, *, base_url: str, api_key: str, prompt=REPORT_PROMPT):
    """Submit the exact calculation release and save only a recipient-admitted report."""
    source = prepared.calculation.calculation_object
    ancestor_reader = prepared.sessions["model-reader"]
    accepted_writers = {prepared.sessions[role].did for role in ["source", "polars", "calculation"]}

    def resolve(identity):
        view = ancestor_reader.governance(
            tn.GovernedObject.read(io.BytesIO(prepared.store.get(identity)))
        )
        view.accept(
            use=RECIPIENT_USE, groups=["finance"], decide=lambda ctx: ctx.writer in accepted_writers
        )
        return view

    recipient = NativeOutputReader(
        prepared.sessions["recipient"],
        expected_source=source,
        expected_writer=prepared.sessions["model-writer"].did,
        use=RECIPIENT_USE,
        destination=DESTINATION,
        release_use=MODEL_USE,
        expected_tool_name="log_compliance_creation",
        expected_tool_sections=("finance",),
        catalog=prepared.catalog,
        policy_dag=prepared.policy_dag,
        resolve_ancestor=resolve,
        decide=lambda ctx: ctx.use_context == RECIPIENT_USE,
    )
    release, report = submit_governed_report(
        source,
        prompt=prompt,
        use=MODEL_USE,
        destination=DESTINATION,
        base_url=base_url,
        api_key=api_key,
        transcript_path=prepared.directory / "model-http.ndjson",
        output_reader=recipient,
        publication_store=prepared.store,
    )
    html = report.data["report_html"]
    if not isinstance(html, str) or not html.strip():
        raise ValueError("model report has no HTML content")
    report_path = prepared.directory / "report.html"
    write_once(report_path, html.encode())
    summary = json.loads(prepared.preparation_path.read_bytes())
    summary.update(
        {
            "model_source_object_id": release.source_object_id,
            "report_object_id": release.output_object_id,
            "report_wire_sha256": hashlib.sha256(release.output_object.forward()).hexdigest(),
            "report_writer": release.output_object.writer,
            "tool_object_ids": [item.id for item in release.tool_objects],
            "recipient_contract_count": len(report.policies),
            "recipient_edition_count": len(report.dataset_bindings),
            "trace": release.trace,
        }
    )
    summary_path = prepared.directory / "summary.json"
    write_once(summary_path, json.dumps(summary, indent=2).encode())
    return report_path, summary_path
