import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType

import polars as pl
import pytest
import tn
from tn_llama_client.clients import GovernedLlamaClient, GovernedRequestDenied
from tn_llama_client.protocol import ReportRelease, SourceRecord
from tn_llama_client.transcript import HttpTranscript

from tn_governed.calculation import StubCalculationPort, execute_portfolio_calculation
from tn_governed.getdata import FeedFetchFact
from tn_governed.llama import (
    adapt_finance_source_record,
    build_finance_governed_request,
    submit_finance_report,
    write_finance_runtime_config,
)
from tn_governed.polars_stage import GovernedPortfolioResult
from tn_governed.tn_flow import ModelBoundTNRow, TNFlowResult, provision_tn_flow


@pytest.fixture(autouse=True)
def isolated_tn(monkeypatch):
    monkeypatch.setenv("TN_NO_LINK", "1")
    monkeypatch.setenv("TN_NO_STDOUT", "1")
    monkeypatch.setenv("TN_AUTOINIT_QUIET", "1")
    with suppress(Exception):
        tn.flush_and_close()
    yield
    with suppress(Exception):
        tn.flush_and_close()


@pytest.fixture(scope="module")
def governed_flow(tmp_path_factory) -> TNFlowResult:
    root = tmp_path_factory.mktemp("llama-governed-flow")
    schema = (
        ("account_id", pl.String),
        ("portfolio_id", pl.String),
        ("symbol", pl.String),
        ("quantity", pl.Int64),
        ("cost_basis", pl.Float64),
        ("currency", pl.String),
        ("as_of", pl.String),
        ("last_price", pl.Float64),
        ("market_source", pl.String),
        ("position_market_value", pl.Float64),
    )
    portfolio = GovernedPortfolioResult(
        _rows=((
            "acct-001",
            "portfolio-001",
            "AAPL",
            25,
            175.50,
            "USD",
            "2026-08-31",
            228.15,
            "deterministic-market-fixture",
            5703.75,
        ),),
        _schema=schema,
        sources=MappingProxyType(
            {name: frozenset({name}) for name, _dtype in schema}
        ),
        row_sources=frozenset({"account_id", "symbol", "currency"}),
        policy_ref="tn://policy/calculation-portfolio-report@1",
        tn_governance=MappingProxyType(
            {"policy": "tn://policy/calculation-portfolio-report@1"}
        ),
        feed_names=("portfolio.internal.positions", "portfolio.market.prices"),
        feed_fetch_facts=(
            FeedFetchFact(
                "getdata.feed_fetched", "portfolio.internal.positions", "/positions", 200,
                "application/x-ndjson",
            ),
            FeedFetchFact(
                "getdata.feed_fetched", "portfolio.market.prices", "/prices", 200,
                "application/x-ndjson",
            ),
        ),
    )
    execution = execute_portfolio_calculation(StubCalculationPort(), portfolio)
    return provision_tn_flow(
        portfolio,
        execution,
        source_project_dir=root / "source",
        source_identity_dir=root / "source-identity",
        source_agents_path=Path("fixtures/portfolio/agents.policy"),
        input_recipient_config_path=root / "input-recipient" / "tn.yaml",
        input_recipient_identity_path=root / "input-recipient-identity" / "identity.json",
        input_package_path=root / "packages" / "input-runtime.tnpkg",
        input_reader_dir=root / "input-reader",
        input_index_key_path=root / "input.index.key",
        output_writer_config_path=root / "output-writer" / "tn.yaml",
        output_writer_identity_path=root / "output-writer-identity" / "identity.json",
        output_agents_path=Path("fixtures/portfolio/report-agents.policy"),
        output_recipient_config_path=root / "output-recipient" / "tn.yaml",
        output_recipient_identity_path=root / "output-recipient-identity" / "identity.json",
        output_package_path=root / "packages" / "output-runtime.tnpkg",
        output_reader_dir=root / "output-reader",
        output_index_key_path=root / "output.index.key",
    )


@pytest.fixture(scope="module")
def encrypted_release_body(governed_flow: TNFlowResult) -> bytes:
    writer = tn.init(governed_flow.output_writer.config_path, link=False, stdout=False)
    policy_document = tn.agents.policy()
    assert policy_document is not None
    template = policy_document.templates["report.generated"]
    policy_reference = (
        f"{template.path}#{template.event_type}@{template.version}#{template.content_hash}"
    )
    emitted = writer.log(
        "report.generated",
        level="info",
        aad={"governed_by": writer.cfg.device.device_identity, "policy": policy_reference},
        default={
            "report_html": (
                "<p>The calculations in this report are deterministic "
                "stub outputs, not live calculation results.</p>"
            )
        },
        source_lineage={
            "source_device_identity": governed_flow.source_row.source_writer_did,
            "source_event_id": governed_flow.source_row.event_id,
            "source_event_type": "portfolio.calculation.calculated",
            "derived_from_row_hash": governed_flow.source_row.row_hash,
            "source_policy": governed_flow.source_row.policy_reference,
            "selected_sections": ["finance"],
            "operation": "derived_html_report",
        },
    )
    assert emitted is not None
    matches = []
    for raw in governed_flow.output_writer.log_path.read_bytes().splitlines(keepends=True):
        parsed = json.loads(raw)
        if (
            parsed.get("event_id") == emitted["event_id"]
            and parsed.get("row_hash") == emitted["row_hash"]
        ):
            matches.append(raw)
    assert len(matches) == 1
    output_ndjson = matches[0].decode("utf-8")
    return json.dumps(
        {
            "protocol": "tn.llama.release.v1",
            "run_id": "run-client-test",
            "source_event_id": governed_flow.source_row.event_id,
            "source_row_hash": governed_flow.source_row.row_hash,
            "output_event_id": emitted["event_id"],
            "output_row_hash": emitted["row_hash"],
            "output_ndjson": output_ndjson,
            "trace": [],
        },
        separators=(",", ":"),
    ).encode()


@contextmanager
def _serve_response(
    body: bytes,
    *,
    status: int = 200,
) -> Iterator[tuple[str, list[dict[str, object]]]]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": self.rfile.read(length),
                }
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _model_bound_row(tmp_path: Path) -> ModelBoundTNRow:
    envelope = {
        "device_identity": "did:key:zFinanceWriter",
        "event_id": "evt-finance-1",
        "event_type": "portfolio.calculation.calculated",
        "finance": {"ciphertext": "finance-ciphertext", "field_hashes": {}},
        "level": "info",
        "prev_hash": "genesis",
        "row_hash": "sha256:finance-row",
        "sequence": 0,
        "signature": "signed-row",
        "timestamp": "2026-09-02T00:00:00Z",
        "tn.agents": {"ciphertext": "governance-ciphertext", "field_hashes": {}},
    }
    raw = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    return ModelBoundTNRow(
        raw_ndjson=raw,
        envelope=MappingProxyType(envelope),
        event_id="evt-finance-1",
        row_hash="sha256:finance-row",
        encrypted_groups=("finance", "tn.agents"),
        policy_reference=".tn/config/agents.md#portfolio.calculation.calculated@1#sha256:policy",
        source_writer_did="did:key:zFinanceWriter",
        source_ceremony_id="cer-finance",
        source_log_path=(tmp_path / "source.ndjson").resolve(),
    )


def _replace_wire_envelope(
    row: ModelBoundTNRow,
    envelope: dict[str, object],
    *,
    encrypted_groups: tuple[str, ...] | None = None,
) -> ModelBoundTNRow:
    raw = json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    return replace(
        row,
        raw_ndjson=raw,
        envelope=MappingProxyType(envelope),
        encrypted_groups=encrypted_groups or row.encrypted_groups,
    )


def _clone_governed_flow(flow: TNFlowResult, destination: Path) -> TNFlowResult:
    paths = (
        flow.source_row.source_log_path,
        flow.marker_free_row.source_log_path,
        flow.input_reader.reader_key_dir,
        flow.input_reader.index_key_path,
        flow.input_reader.package_path,
        flow.output_writer.config_path,
        flow.output_writer.log_path,
        flow.output_writer.reader.reader_key_dir,
        flow.output_writer.reader.index_key_path,
        flow.output_writer.reader.package_path,
    )
    source_root = Path(os.path.commonpath([str(path) for path in paths]))
    shutil.copytree(source_root, destination)

    def cloned(path: Path) -> Path:
        return (destination / path.relative_to(source_root)).resolve()

    input_reader = replace(
        flow.input_reader,
        reader_key_dir=cloned(flow.input_reader.reader_key_dir),
        index_key_path=cloned(flow.input_reader.index_key_path),
        package_path=cloned(flow.input_reader.package_path),
    )
    output_reader = replace(
        flow.output_writer.reader,
        reader_key_dir=cloned(flow.output_writer.reader.reader_key_dir),
        index_key_path=cloned(flow.output_writer.reader.index_key_path),
        package_path=cloned(flow.output_writer.reader.package_path),
    )
    output_writer = replace(
        flow.output_writer,
        config_path=cloned(flow.output_writer.config_path),
        log_path=cloned(flow.output_writer.log_path),
        reader=output_reader,
    )
    return replace(
        flow,
        source_row=replace(
            flow.source_row,
            source_log_path=cloned(flow.source_row.source_log_path),
        ),
        marker_free_row=replace(
            flow.marker_free_row,
            source_log_path=cloned(flow.marker_free_row.source_log_path),
        ),
        input_reader=input_reader,
        output_writer=output_writer,
    )


def _snapshot_tree(root: Path) -> dict[str, tuple[str, bytes]]:
    if not root.exists():
        return {}
    snapshot: dict[str, tuple[str, bytes]] = {".": ("directory", b"")}
    for path in sorted(root.rglob("*"), key=lambda candidate: str(candidate)):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path).encode())
        elif path.is_dir():
            snapshot[relative] = ("directory", b"")
        else:
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


_PROTECTED_TN_CATEGORIES = (
    "input-reader-kit",
    "input-index-key",
    "input-package",
    "output-reader-kit",
    "output-index-key",
    "output-package",
    "output-writer-config",
    "output-writer-keystore",
    "output-writer-log",
    "output-writer-root-child",
    "source-config",
    "source-keystore",
    "source-log",
    "source-root-child",
)


def _protected_tn_target(flow: TNFlowResult, category: str) -> Path:
    output_writer_root = flow.output_writer.config_path.parent
    source_root = flow.source_row.source_log_path.parent.parent
    output_keystore = tn.config.load(flow.output_writer.config_path).keystore
    source_keystore = tn.config.load(source_root / "tn.yaml").keystore
    targets = {
        "input-reader-kit": flow.input_reader.reader_key_dir / "finance.btn.mykit",
        "input-index-key": flow.input_reader.index_key_path,
        "input-package": flow.input_reader.package_path,
        "output-reader-kit": (
            flow.output_writer.reader.reader_key_dir / "default.btn.mykit"
        ),
        "output-index-key": flow.output_writer.reader.index_key_path,
        "output-package": flow.output_writer.reader.package_path,
        "output-writer-config": flow.output_writer.config_path,
        "output-writer-keystore": output_keystore / "index_master.key",
        "output-writer-log": flow.output_writer.log_path,
        "output-writer-root-child": output_writer_root / "runtime-output-injection",
        "source-config": source_root / "tn.yaml",
        "source-keystore": source_keystore / "index_master.key",
        "source-log": flow.source_row.source_log_path,
        "source-root-child": source_root / "runtime-output-injection",
    }
    return targets[category].resolve(strict=False)


def test_finance_adapter_preserves_the_exact_source_row(tmp_path: Path) -> None:
    model_row = _model_bound_row(tmp_path)

    record = adapt_finance_source_record(model_row)

    assert isinstance(record, SourceRecord)
    assert record.ndjson == model_row.raw_ndjson
    assert record.event_id == "evt-finance-1"
    assert record.row_hash == "sha256:finance-row"
    assert record.encrypted_groups == ("finance", "tn.agents")


def test_finance_adapter_rejects_more_or_less_than_one_ndjson_row(
    tmp_path: Path,
) -> None:
    model_row = _model_bound_row(tmp_path)

    for raw in (
        model_row.raw_ndjson.removesuffix(b"\n"),
        model_row.raw_ndjson + model_row.raw_ndjson,
        b"\n",
    ):
        try:
            adapt_finance_source_record(replace(model_row, raw_ndjson=raw))
        except ValueError:
            pass
        else:
            raise AssertionError("malformed NDJSON framing reached the sibling source type")


def test_finance_adapter_rejects_rows_outside_the_finance_profile(
    tmp_path: Path,
) -> None:
    model_row = _model_bound_row(tmp_path)

    wrong_event = dict(model_row.envelope)
    wrong_event["event_type"] = "precisely.risk.enriched"
    missing_signature = dict(model_row.envelope)
    del missing_signature["signature"]
    wrong_group = dict(model_row.envelope)
    wrong_group["other"] = wrong_group.pop("finance")

    cases = (
        _replace_wire_envelope(model_row, wrong_event),
        _replace_wire_envelope(model_row, missing_signature),
        _replace_wire_envelope(
            model_row,
            wrong_group,
            encrypted_groups=("other", "tn.agents"),
        ),
    )
    for candidate in cases:
        with pytest.raises(ValueError):
            adapt_finance_source_record(candidate)


def test_finance_adapter_rejects_source_metadata_or_envelope_mismatch(
    tmp_path: Path,
) -> None:
    model_row = _model_bound_row(tmp_path)
    altered = dict(model_row.envelope)
    altered["unexpected"] = "different-row"
    altered_raw = json.dumps(
        altered, separators=(",", ":"), sort_keys=True
    ).encode() + b"\n"

    cases = (
        replace(model_row, raw_ndjson=altered_raw),
        replace(model_row, event_id="evt-other"),
        replace(model_row, row_hash="sha256:other"),
        replace(model_row, source_writer_did="did:key:zOtherWriter"),
        replace(model_row, encrypted_groups=("tn.agents", "finance")),
    )
    for candidate in cases:
        with pytest.raises(ValueError):
            adapt_finance_source_record(candidate)


def test_finance_request_uses_the_sibling_closed_wrapper_and_exact_row(
    tmp_path: Path,
) -> None:
    model_row = _model_bound_row(tmp_path)

    body = build_finance_governed_request(
        model_row,
        "Generate the governed portfolio report.",
    )

    expected_prefix = (
        b'{"messages":[{"content":"Generate the governed portfolio report.",'
        b'"role":"user"}],"protocol":"tn.llama.request.v1",'
        b'"purpose":"derived_html_report","stream":false,"tn_row":'
    )
    assert body == expected_prefix + model_row.raw_ndjson.removesuffix(b"\n") + b"}"
    parsed = json.loads(body)
    assert set(parsed) == {"messages", "protocol", "purpose", "stream", "tn_row"}
    assert parsed["tn_row"] == dict(model_row.envelope)


def test_runtime_config_is_the_closed_finance_profile_with_live_material(
    tmp_path: Path,
    governed_flow: TNFlowResult,
) -> None:
    model_path = tmp_path / "model.gguf"
    server_path = tmp_path / "llama-server.exe"
    model_path.write_bytes(b"gguf-model")
    server_path.write_bytes(b"governed-server")
    config_path = (tmp_path / "runtime" / "runtime-config.json").resolve()
    audit_path = (tmp_path / "runtime" / "decision-events.ndjson").resolve()
    raw_trace_dir = (tmp_path / "runtime" / "raw-trace").resolve()

    written = write_finance_runtime_config(
        governed_flow,
        config_path=config_path,
        audit_path=audit_path,
        raw_trace_dir=raw_trace_dir,
        model_path=model_path.resolve(),
        server_path=server_path.resolve(),
        build_id="tn-governed-portfolio-demo",
    )

    assert written == config_path
    document = json.loads(config_path.read_text(encoding="utf-8"))
    assert document == {
        "version": "tn.llama.runtime.v2",
        "writer_authority": "group_capability",
        "expected_event_type": "portfolio.calculation.calculated",
        "allowed_input_groups": ["finance"],
        "report_contract": {
            "allowed_input_sections": ["finance"],
            "output_event_type": "report.generated",
            "output_public_section": "default",
            "operation": "derived_html_report",
            "section_rules": {
                "finance": {
                    "propagate_source_governance": True,
                    "required_attribution": (
                        "The calculations in this report are deterministic "
                        "stub outputs, not live calculation results."
                    ),
                    "required_tool": None,
                }
            },
            "output_policy": {
                "instruction": "Generate only the configured derived report.",
                "use_for": "Creating the governed HTML report requested by the user.",
                "do_not_use_for": "Any ungoverned output or unrelated purpose.",
                "consequences": "Reject the report and release no output when validation fails.",
                "on_violation_or_error": (
                    "Abort the governed session and record the failing gate."
                ),
            },
        },
        "reader_key_dir": str(governed_flow.input_reader.reader_key_dir),
        "input_index_key_path": str(governed_flow.input_reader.index_key_path),
        "ceremony_id": governed_flow.input_reader.ceremony_id,
        "governance_group": "tn.agents",
        "input_index_epoch": 0,
        "output_tn_config_path": str(governed_flow.output_writer.config_path),
        "output_reader_key_dir": str(governed_flow.output_writer.reader.reader_key_dir),
        "output_index_key_path": str(governed_flow.output_writer.reader.index_key_path),
        "output_ceremony_id": governed_flow.output_writer.ceremony_id,
        "output_governance_group": "tn.agents",
        "output_index_epoch": 0,
        "audit_path": str(audit_path),
        "raw_trace_dir": str(raw_trace_dir),
        "build_id": "tn-governed-portfolio-demo",
        "llama_source_revision": "3f545beccee69d9975f466ec7e45fd9aacd8ba90",
        "model_sha256": "2cbdd631a15e89b1e57204890d9515dc9091da693880f9cde3b32e7bd0f04597",
        "binary_sha256": "537227e448ebb9a8c0274f14f180f77a3a37e02b6343ce9ca9a03787c36fc8e0",
        "max_request_bytes": 1048576,
        "max_message_bytes": 16384,
        "max_active_sessions": 1,
        "max_chain_namespaces": 8,
        "echo_max_tokens": 1536,
        "final_max_tokens": 2048,
    }


def test_runtime_config_reads_the_active_output_tn_policy(
    tmp_path: Path,
    governed_flow: TNFlowResult,
) -> None:
    cloned_writer_root = tmp_path / "output-writer"
    shutil.copytree(governed_flow.output_writer.config_path.parent, cloned_writer_root)
    cloned_config = cloned_writer_root / "tn.yaml"
    policy_path = cloned_writer_root / ".tn" / "config" / "agents.md"
    policy_path.write_text(
        policy_path.read_text(encoding="utf-8").replace(
            "Creating the governed HTML report requested by the user.",
            "Creating one mutation-tested governed finance report.",
        ),
        encoding="utf-8",
    )
    cloned_output_writer = replace(
        governed_flow.output_writer,
        config_path=cloned_config.resolve(),
        log_path=(cloned_writer_root / ".tn" / "logs" / "tn.ndjson").resolve(),
    )
    cloned_flow = replace(governed_flow, output_writer=cloned_output_writer)
    model_path = (tmp_path / "model.gguf").resolve()
    server_path = (tmp_path / "llama-server.exe").resolve()
    model_path.write_bytes(b"gguf-model")
    server_path.write_bytes(b"governed-server")
    config_path = (tmp_path / "runtime-config.json").resolve()

    write_finance_runtime_config(
        cloned_flow,
        config_path=config_path,
        audit_path=(tmp_path / "decision-events.ndjson").resolve(),
        raw_trace_dir=(tmp_path / "raw-trace").resolve(),
        model_path=model_path,
        server_path=server_path,
        build_id="policy-mutation",
    )

    document = json.loads(config_path.read_text(encoding="utf-8"))
    assert document["report_contract"]["output_policy"]["use_for"] == (
        "Creating one mutation-tested governed finance report."
    )


def test_runtime_config_refuses_missing_tn_reader_material_before_write(
    tmp_path: Path,
    governed_flow: TNFlowResult,
) -> None:
    reader_dir = tmp_path / "input-reader"
    shutil.copytree(governed_flow.input_reader.reader_key_dir, reader_dir)
    (reader_dir / "finance.btn.mykit").unlink()
    incomplete_input = replace(governed_flow.input_reader, reader_key_dir=reader_dir.resolve())
    incomplete_flow = replace(governed_flow, input_reader=incomplete_input)
    model_path = (tmp_path / "model.gguf").resolve()
    server_path = (tmp_path / "llama-server.exe").resolve()
    model_path.write_bytes(b"gguf-model")
    server_path.write_bytes(b"governed-server")
    config_path = (tmp_path / "runtime-config.json").resolve()

    with pytest.raises(ValueError, match="exactly"):
        write_finance_runtime_config(
            incomplete_flow,
            config_path=config_path,
            audit_path=(tmp_path / "decision-events.ndjson").resolve(),
            raw_trace_dir=(tmp_path / "raw-trace").resolve(),
            model_path=model_path,
            server_path=server_path,
            build_id="missing-kit",
        )

    assert not config_path.exists()


@pytest.mark.parametrize(
    "collision",
    (
        "config-in-input-reader",
        "decision-event-is-input-index-key",
        "decision-event-is-input-kit",
        "raw-trace-contains-reader-state",
        "config-is-decision-event-by-alias",
    ),
)
def test_runtime_config_rejects_state_aliases_before_any_filesystem_change(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    collision: str,
) -> None:
    flow = _clone_governed_flow(governed_flow, tmp_path / "flow")
    model_path = (tmp_path / "model.gguf").resolve()
    server_path = (tmp_path / "llama-server.exe").resolve()
    model_path.write_bytes(b"gguf-model")
    server_path.write_bytes(b"governed-server")
    runtime_dir = tmp_path / "runtime"
    config_path = (runtime_dir / "runtime-config.json").resolve()
    decision_event_path = (runtime_dir / "decision-events.ndjson").resolve()
    raw_trace_dir = (runtime_dir / "raw-trace").resolve()

    if collision == "config-in-input-reader":
        config_path = flow.input_reader.reader_key_dir / "runtime-config.json"
    elif collision == "decision-event-is-input-index-key":
        decision_event_path = flow.input_reader.index_key_path
    elif collision == "decision-event-is-input-kit":
        decision_event_path = flow.input_reader.reader_key_dir / "finance.btn.mykit"
    elif collision == "raw-trace-contains-reader-state":
        raw_trace_dir = (tmp_path / "flow").resolve()
    else:
        decision_event_path = runtime_dir / "aliases" / ".." / "runtime-config.json"

    before = _snapshot_tree(tmp_path)
    with pytest.raises(ValueError):
        write_finance_runtime_config(
            flow,
            config_path=config_path,
            audit_path=decision_event_path,
            raw_trace_dir=raw_trace_dir,
            model_path=model_path,
            server_path=server_path,
            build_id="state-alias-denial",
        )

    assert _snapshot_tree(tmp_path) == before


@pytest.mark.parametrize("protected_category", _PROTECTED_TN_CATEGORIES)
@pytest.mark.parametrize(
    "generated_field",
    ("runtime-config", "decision-event", "raw-trace"),
)
def test_every_protected_tn_category_rejects_every_runtime_output(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    protected_category: str,
    generated_field: str,
) -> None:
    flow = _clone_governed_flow(governed_flow, tmp_path / "flow")
    model_path = (tmp_path / "model.gguf").resolve()
    server_path = (tmp_path / "llama-server.exe").resolve()
    model_path.write_bytes(b"gguf-model")
    server_path.write_bytes(b"governed-server")
    config_path = (tmp_path / "runtime" / "runtime-config.json").resolve()
    decision_event_path = (
        tmp_path / "runtime" / "decision-events.ndjson"
    ).resolve()
    raw_trace_dir = (tmp_path / "runtime" / "raw-trace").resolve()
    protected_target = _protected_tn_target(flow, protected_category)

    if generated_field == "runtime-config":
        config_path = protected_target
    elif generated_field == "decision-event":
        decision_event_path = protected_target
    else:
        raw_trace_dir = protected_target

    before = _snapshot_tree(tmp_path)
    with pytest.raises(ValueError):
        write_finance_runtime_config(
            flow,
            config_path=config_path,
            audit_path=decision_event_path,
            raw_trace_dir=raw_trace_dir,
            model_path=model_path,
            server_path=server_path,
            build_id="protected-state-cross-product",
        )

    assert _snapshot_tree(tmp_path) == before


@pytest.mark.parametrize(
    "collision",
    (
        "transcript-is-output-kit",
        "transcript-is-source-log",
        "transcript-is-received-output",
        "received-output-is-output-kit",
        "received-output-is-source-log",
        "paths-match-by-lexical-alias",
    ),
)
def test_submit_rejects_state_aliases_without_http_or_filesystem_change(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    collision: str,
) -> None:
    flow = _clone_governed_flow(governed_flow, tmp_path / "flow")
    transcript_path = (tmp_path / "client-http.ndjson").resolve()
    received_path = (tmp_path / "received-output.ndjson").resolve()
    output_kit = flow.output_writer.reader.reader_key_dir / "default.btn.mykit"

    if collision == "transcript-is-output-kit":
        transcript_path = output_kit
    elif collision == "transcript-is-source-log":
        transcript_path = flow.source_row.source_log_path
    elif collision == "transcript-is-received-output":
        transcript_path = received_path
    elif collision == "received-output-is-output-kit":
        received_path = output_kit
    elif collision == "received-output-is-source-log":
        received_path = flow.source_row.source_log_path
    else:
        transcript_path = tmp_path / "aliases" / ".." / "shared.ndjson"
        received_path = (tmp_path / "shared.ndjson").resolve()

    before = _snapshot_tree(tmp_path)
    with _serve_response(b"{}") as (base_url, requests), pytest.raises(ValueError):
        submit_finance_report(
            flow,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=transcript_path,
            received_output_path=received_path,
        )

    assert requests == []
    assert _snapshot_tree(tmp_path) == before


@pytest.mark.parametrize("relative_field", ("transcript", "received-output"))
def test_submit_requires_absolute_output_paths_before_http_or_write(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    monkeypatch: pytest.MonkeyPatch,
    relative_field: str,
) -> None:
    flow = _clone_governed_flow(governed_flow, tmp_path / "flow")
    transcript_path = (tmp_path / "client-http.ndjson").resolve()
    received_path = (tmp_path / "received-output.ndjson").resolve()
    if relative_field == "transcript":
        transcript_path = Path("relative-client-http.ndjson")
    else:
        received_path = Path("relative-received-output.ndjson")
    monkeypatch.chdir(tmp_path)

    before = _snapshot_tree(tmp_path)
    with _serve_response(b"{}") as (base_url, requests), pytest.raises(ValueError):
        submit_finance_report(
            flow,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=transcript_path,
            received_output_path=received_path,
        )

    assert requests == []
    assert _snapshot_tree(tmp_path) == before


@pytest.mark.parametrize("protected_category", _PROTECTED_TN_CATEGORIES)
@pytest.mark.parametrize("generated_field", ("transcript", "received-output"))
def test_every_protected_tn_category_rejects_every_submit_output(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    protected_category: str,
    generated_field: str,
) -> None:
    flow = _clone_governed_flow(governed_flow, tmp_path / "flow")
    transcript_path = (tmp_path / "client-http.ndjson").resolve()
    received_path = (tmp_path / "received-output.ndjson").resolve()
    protected_target = _protected_tn_target(flow, protected_category)
    if generated_field == "transcript":
        transcript_path = protected_target
    else:
        received_path = protected_target

    before = _snapshot_tree(tmp_path)
    with _serve_response(b"{}") as (base_url, requests), pytest.raises(ValueError):
        submit_finance_report(
            flow,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=transcript_path,
            received_output_path=received_path,
        )

    assert requests == []
    assert _snapshot_tree(tmp_path) == before


def test_submit_rejects_a_row_not_selected_from_the_exact_governed_handoff(
    tmp_path: Path,
    governed_flow: TNFlowResult,
) -> None:
    flow = _clone_governed_flow(governed_flow, tmp_path / "flow")
    reconstructed_row = replace(flow.source_row)
    before = _snapshot_tree(tmp_path)

    with _serve_response(b"{}") as (base_url, requests), pytest.raises(ValueError):
        submit_finance_report(
            flow,
            selected_row=reconstructed_row,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=(tmp_path / "client-http.ndjson").resolve(),
            received_output_path=(tmp_path / "received-output.ndjson").resolve(),
        )

    assert requests == []
    assert _snapshot_tree(tmp_path) == before


def test_normal_sibling_runtime_and_client_paths_share_one_run_root(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    encrypted_release_body: bytes,
) -> None:
    run_root = tmp_path / "run"
    flow = _clone_governed_flow(governed_flow, run_root / "governed-state")
    model_path = (run_root / "artifacts" / "model.gguf").resolve()
    server_path = (run_root / "artifacts" / "llama-server.exe").resolve()
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"gguf-model")
    server_path.write_bytes(b"governed-server")
    config_path = (run_root / "runtime" / "runtime-config.json").resolve()

    written = write_finance_runtime_config(
        flow,
        config_path=config_path,
        audit_path=(run_root / "runtime" / "decision-events.ndjson").resolve(),
        raw_trace_dir=(run_root / "runtime" / "raw-trace").resolve(),
        model_path=model_path,
        server_path=server_path,
        build_id="one-run-root",
    )
    with _serve_response(encrypted_release_body) as (base_url, requests):
        release = submit_finance_report(
            flow,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=(run_root / "client" / "http.ndjson").resolve(),
            received_output_path=(run_root / "client" / "received.ndjson").resolve(),
        )

    assert written == config_path
    assert len(requests) == 1
    assert release.source_event_id == flow.source_row.event_id
    assert release.source_row_hash == flow.source_row.row_hash


def test_submit_uses_only_the_governed_route_and_returns_verified_release(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    encrypted_release_body: bytes,
) -> None:
    transcript_path = tmp_path / "client-http.ndjson"
    received_path = tmp_path / "received-output.ndjson"
    with _serve_response(encrypted_release_body) as (base_url, requests):
        release = submit_finance_report(
            governed_flow,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=transcript_path,
            received_output_path=received_path,
        )

    assert isinstance(release, ReportRelease)
    assert release.report_html == (
        "<p>The calculations in this report are deterministic "
        "stub outputs, not live calculation results.</p>"
    )
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/tn/chat/completions"
    assert requests[0]["body"] == build_finance_governed_request(
        governed_flow.source_row,
        "Generate the governed portfolio report.",
    )
    assert requests[0]["headers"]["Authorization"] == "Bearer local-test-key"
    transcript = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    request_event = next(item for item in transcript if item.get("direction") == "request")
    assert all(name != "authorization" for name, _value in request_event["headers"])
    assert received_path.read_text(encoding="utf-8") == json.loads(
        encrypted_release_body
    )["output_ndjson"]


def test_malformed_release_is_rejected_by_the_sibling_parser(
    tmp_path: Path,
    governed_flow: TNFlowResult,
) -> None:
    with (
        _serve_response(b'{"protocol":"tn.llama.release.v1"}') as (base_url, _requests),
        pytest.raises(ValueError, match="exact release shape"),
    ):
        submit_finance_report(
            governed_flow,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=tmp_path / "client-http.ndjson",
            received_output_path=tmp_path / "received-output.ndjson",
        )

    assert not (tmp_path / "received-output.ndjson").exists()
    assert list(tmp_path.rglob("*.html")) == []


def test_governed_denial_is_returned_once_without_retry_or_substitute(
    tmp_path: Path,
    governed_flow: TNFlowResult,
) -> None:
    denial = b'{"error":{"code":"tn_aad_shape"}}'
    with (
        _serve_response(denial, status=400) as (base_url, requests),
        pytest.raises(GovernedRequestDenied) as raised,
    ):
        submit_finance_report(
            governed_flow,
            selected_row=governed_flow.marker_free_row,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=tmp_path / "client-http.ndjson",
            received_output_path=tmp_path / "received-output.ndjson",
        )

    assert raised.value.status == 400
    assert raised.value.code == "tn_aad_shape"
    assert len(requests) == 1
    assert requests[0]["body"] == build_finance_governed_request(
        governed_flow.marker_free_row,
        "Generate the governed portfolio report.",
    )
    assert not (tmp_path / "received-output.ndjson").exists()
    assert list(tmp_path.rglob("*.html")) == []


def test_sibling_client_refuses_an_encrypted_release_without_an_output_reader(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    encrypted_release_body: bytes,
) -> None:
    transcript = HttpTranscript(tmp_path / "client-http.ndjson")
    with _serve_response(encrypted_release_body) as (base_url, _requests):
        client = GovernedLlamaClient(
            base_url,
            "local-test-key",
            transcript,
            output_reader=None,
        )
        with pytest.raises(ValueError, match="requires a TN output reader"):
            client.submit_request(
                build_finance_governed_request(
                    governed_flow.source_row,
                    "Generate the governed portfolio report.",
                )
            )

    assert list(tmp_path.rglob("*.html")) == []


@pytest.mark.parametrize("kit_state", ["missing", "wrong"])
def test_missing_or_wrong_output_kit_rolls_back_and_returns_no_release(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    encrypted_release_body: bytes,
    kit_state: str,
) -> None:
    reader_dir = tmp_path / "output-reader"
    shutil.copytree(governed_flow.output_writer.reader.reader_key_dir, reader_dir)
    default_kit = reader_dir / "default.btn.mykit"
    if kit_state == "missing":
        default_kit.unlink()
    else:
        default_kit.write_bytes(
            (governed_flow.input_reader.reader_key_dir / "finance.btn.mykit").read_bytes()
        )
    output_reader = replace(
        governed_flow.output_writer.reader,
        reader_key_dir=reader_dir.resolve(),
    )
    flow = replace(
        governed_flow,
        output_writer=replace(governed_flow.output_writer, reader=output_reader),
    )
    received_path = tmp_path / "received-output.ndjson"
    transcript_path = tmp_path / "client-http.ndjson"
    expected_error = FileNotFoundError if kit_state == "missing" else (ValueError, RuntimeError)

    with (
        _serve_response(encrypted_release_body) as (base_url, _requests),
        pytest.raises(expected_error),
    ):
        submit_finance_report(
            flow,
            prompt="Generate the governed portfolio report.",
            base_url=base_url,
            api_key="local-test-key",
            transcript_path=transcript_path,
            received_output_path=received_path,
        )

    assert received_path.read_bytes() == b""
    transitions = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    rejected = [item for item in transitions if item.get("event") == "encrypted_release_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["ledger_rolled_back"] is True
    assert list(tmp_path.rglob("*.html")) == []


def test_fresh_process_opens_the_exact_output_with_only_recipient_kits(
    tmp_path: Path,
    governed_flow: TNFlowResult,
    encrypted_release_body: bytes,
) -> None:
    response_path = tmp_path / "release.json"
    response_path.write_bytes(encrypted_release_body)
    received_path = tmp_path / "fresh-received.ndjson"
    transcript_path = tmp_path / "fresh-http.ndjson"
    script = """
import json
import sys
from pathlib import Path
from tn_llama_client.clients import TnOutputReader
from tn_llama_client.protocol import EncryptedReportRelease
from tn_llama_client.transcript import HttpTranscript

response_path, reader_dir, received_path, transcript_path = map(Path, sys.argv[1:])
encrypted = EncryptedReportRelease.from_http_json(response_path.read_bytes())
release = TnOutputReader(reader_dir, received_path).open(
    encrypted,
    HttpTranscript(transcript_path),
)
print(json.dumps({
    "output_event_id": release.output_event_id,
    "output_row_hash": release.output_row_hash,
    "report_html": release.report_html,
}))
"""
    environment = os.environ.copy()
    python_path = [
        str(Path("src").resolve()),
        str(Path("../tn-llama").resolve()),
        str(Path("../tn-polars").resolve()),
    ]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(response_path),
            str(governed_flow.output_writer.reader.reader_key_dir),
            str(received_path),
            str(transcript_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    opened = json.loads(completed.stdout)
    wrapper = json.loads(encrypted_release_body)
    assert opened["output_event_id"] == wrapper["output_event_id"]
    assert opened["output_row_hash"] == wrapper["output_row_hash"]
    assert opened["report_html"] == (
        "<p>The calculations in this report are deterministic "
        "stub outputs, not live calculation results.</p>"
    )
    assert received_path.read_text(encoding="utf-8") == wrapper["output_ndjson"]
