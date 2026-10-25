"""Thin application adapter for the governed tn-llama client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tn
from tn_llama_client.clients import GovernedLlamaClient, TnOutputReader
from tn_llama_client.protocol import ReportRelease, SourceRecord, build_governed_request
from tn_llama_client.transcript import HttpTranscript

from .tn_flow import (
    ModelBoundTNRow,
    TNFlowResult,
    validate_runtime_reader_material,
)

_FINANCE_EVENT_TYPE = "portfolio.deepvest.calculated"
_FINANCE_GROUP = "finance"
_GOVERNANCE_GROUP = "tn.agents"
_OUTPUT_EVENT_TYPE = "report.generated"
_LLAMA_SOURCE_REVISION = "3f545beccee69d9975f466ec7e45fd9aacd8ba90"
_STUB_ATTRIBUTION = (
    "DeepVest calculations in this first-pass report are deterministic stub outputs, "
    "not live DeepVest results."
)
_POLICY_FIELDS = (
    "instruction",
    "use_for",
    "do_not_use_for",
    "consequences",
    "on_violation_or_error",
)


def submit_governed_report(
    object: tn.GovernedObject, *, prompt: str, use: tn.UseContext, destination: str,
    base_url: str, api_key: str, transcript_path: Path, output_reader, publication_store,
):
    """Send the released publication, then admit the native report at its recipient."""
    client = GovernedLlamaClient(base_url, api_key, HttpTranscript(transcript_path))
    release = client.submit_model(object, prompt, use, destination=destination)
    report = output_reader.open(release)
    for tool in getattr(release, "tool_objects", ()):
        publication_store.put(tool)
    publication_store.put(release.output_object)
    return release, report


def adapt_finance_source_record(row: ModelBoundTNRow) -> SourceRecord:
    """Adapt a governed model-input row to tn-llama's source type."""
    raw = row.raw_ndjson
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1 or not raw[:-1]:
        raise ValueError("finance source must be exactly one newline-terminated NDJSON row")
    parsed = json.loads(raw[:-1])
    if not isinstance(parsed, dict):
        raise ValueError("finance source TN row must be a JSON object")
    if parsed.get("event_type") != _FINANCE_EVENT_TYPE:
        raise ValueError("finance source has the wrong TN event type")
    for name in ("event_id", "row_hash", "signature"):
        if not isinstance(parsed.get(name), str) or not parsed[name]:
            raise ValueError(f"finance source requires a nonempty {name}")
    groups = tuple(
        sorted(
            name
            for name, value in parsed.items()
            if isinstance(value, dict) and "ciphertext" in value
        )
    )
    if groups != (_FINANCE_GROUP, _GOVERNANCE_GROUP):
        raise ValueError("finance source must encrypt exactly finance and tn.agents")
    if parsed != dict(row.envelope):
        raise ValueError("finance source bytes differ from the Task 5 envelope")
    if (
        parsed["event_id"] != row.event_id
        or parsed["row_hash"] != row.row_hash
        or parsed.get("device_identity") != row.source_writer_did
        or groups != row.encrypted_groups
    ):
        raise ValueError("finance source headers differ from the Task 5 metadata")
    return SourceRecord(
        ndjson=raw,
        event_id=row.event_id,
        row_hash=row.row_hash,
        encrypted_groups=row.encrypted_groups,
    )


def build_finance_governed_request(row: ModelBoundTNRow, prompt: str) -> bytes:
    """Build tn-llama's closed request around the exact governed source row."""
    return build_governed_request(adapt_finance_source_record(row), prompt)


def submit_finance_report(
    flow: TNFlowResult,
    *,
    selected_row: ModelBoundTNRow | None = None,
    prompt: str,
    base_url: str,
    api_key: str,
    transcript_path: Path,
    received_output_path: Path,
) -> ReportRelease:
    """Submit one exact governed request and return only a recipient-opened release."""
    row = flow.source_row if selected_row is None else selected_row
    if row is not flow.source_row and row is not flow.marker_free_row:
        raise ValueError("selected TN row is not one of the exact Task 5 handoff rows")
    transcript_path, received_output_path, output_reader_key_dir = (
        _preflight_submit_paths(
            flow,
            transcript_path=transcript_path,
            received_output_path=received_output_path,
        )
    )
    body = build_finance_governed_request(row, prompt)
    transcript = HttpTranscript(transcript_path)
    output_reader = TnOutputReader(
        output_reader_key_dir,
        received_output_path,
    )
    client = GovernedLlamaClient(
        base_url,
        api_key,
        transcript,
        output_reader,
    )
    return client.submit_request(body)


def _require_absolute_file(path: Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() or not resolved.is_file():
        raise FileNotFoundError(f"{label} must be an existing absolute file: {resolved}")
    return resolved


def _require_absolute_output(path: Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        raise ValueError(f"{label} must be absolute: {resolved}")
    return resolved.resolve(strict=False)


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or path.is_relative_to(directory)


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _source_ceremony_root(source_log_path: Path) -> Path:
    source_log = _require_absolute_output(source_log_path, "TN source log path")
    for ancestor in source_log.parents:
        if (ancestor / "tn.yaml").is_file():
            return ancestor
    raise FileNotFoundError(f"TN source ceremony config not found above: {source_log}")


def _task5_protected_state(flow: TNFlowResult) -> tuple[tuple[str, Path], ...]:
    output_writer_config = _require_absolute_output(
        flow.output_writer.config_path,
        "TN output writer config",
    )
    return (
        (
            "TN input reader directory",
            _require_absolute_output(
                flow.input_reader.reader_key_dir,
                "TN input reader directory",
            ),
        ),
        (
            "TN input index key",
            _require_absolute_output(flow.input_reader.index_key_path, "TN input index key"),
        ),
        (
            "TN input package",
            _require_absolute_output(flow.input_reader.package_path, "TN input package"),
        ),
        (
            "TN output reader directory",
            _require_absolute_output(
                flow.output_writer.reader.reader_key_dir,
                "TN output reader directory",
            ),
        ),
        (
            "TN output index key",
            _require_absolute_output(
                flow.output_writer.reader.index_key_path,
                "TN output index key",
            ),
        ),
        (
            "TN output package",
            _require_absolute_output(
                flow.output_writer.reader.package_path,
                "TN output package",
            ),
        ),
        ("TN output writer root", output_writer_config.parent),
        (
            "TN source ceremony root",
            _source_ceremony_root(flow.source_row.source_log_path),
        ),
    )


def _preflight_submit_paths(
    flow: TNFlowResult,
    *,
    transcript_path: Path,
    received_output_path: Path,
) -> tuple[Path, Path, Path]:
    transcript = _require_absolute_output(transcript_path, "HTTP transcript path")
    received = _require_absolute_output(received_output_path, "received-output path")
    protected = _task5_protected_state(flow)
    reader_dir = dict(protected)["TN output reader directory"]

    if _paths_overlap(transcript, received):
        raise ValueError("HTTP transcript and received-output paths must be separate")
    for label, path in (
        ("HTTP transcript", transcript),
        ("received output", received),
    ):
        for protected_label, protected_path in protected:
            if _paths_overlap(path, protected_path):
                raise ValueError(f"{label} path overlaps {protected_label}")
    return transcript, received, reader_dir


def _preflight_runtime_output_paths(
    flow: TNFlowResult,
    *,
    config_path: Path,
    audit_path: Path,
    raw_trace_dir: Path,
) -> tuple[Path, Path, Path]:
    generated = (
        ("runtime config", _require_absolute_output(config_path, "runtime config path")),
        (
            "decision-event file",
            _require_absolute_output(audit_path, "decision-event path"),
        ),
        (
            "raw-trace directory",
            _require_absolute_output(raw_trace_dir, "raw-trace directory"),
        ),
    )
    for index, (left_label, left_path) in enumerate(generated):
        for right_label, right_path in generated[index + 1 :]:
            if _paths_overlap(left_path, right_path):
                raise ValueError(f"{left_label} overlaps {right_label}")

    protected = _task5_protected_state(flow)
    for generated_label, generated_path in generated:
        for protected_label, protected_path in protected:
            if _paths_overlap(generated_path, protected_path):
                raise ValueError(f"{generated_label} overlaps {protected_label}")

    return generated[0][1], generated[1][1], generated[2][1]


def _sha256(path: Path) -> str:
    with path.open("rb") as artifact:
        return hashlib.file_digest(artifact, "sha256").hexdigest()


def _active_output_policy(flow: TNFlowResult) -> dict[str, str]:
    writer = tn.init(flow.output_writer.config_path, link=False, stdout=False)
    if writer.cfg.ceremony_id != flow.output_writer.ceremony_id:
        raise ValueError("active TN output ceremony differs from Task 5 material")
    document = tn.agents.policy()
    template = document.templates.get(_OUTPUT_EVENT_TYPE) if document is not None else None
    if template is None:
        raise ValueError("active TN output policy does not cover report.generated")
    values = {name: getattr(template, name) for name in _POLICY_FIELDS}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError("active TN output policy has an empty foundational field")
    return values


def _validate_task5_material(flow: TNFlowResult) -> None:
    adapt_finance_source_record(flow.source_row)
    validate_runtime_reader_material(
        flow.input_reader,
        expected_groups=(_FINANCE_GROUP, _GOVERNANCE_GROUP),
    )
    validate_runtime_reader_material(
        flow.output_writer.reader,
        expected_groups=("default", _GOVERNANCE_GROUP),
    )
    _require_absolute_file(flow.output_writer.config_path, "TN output writer config")
    if (
        flow.input_reader.ceremony_id != flow.source_row.source_ceremony_id
        or flow.input_reader.index_epoch != 0
        or flow.output_writer.reader.ceremony_id != flow.output_writer.ceremony_id
        or flow.output_writer.reader.index_epoch != 0
        or flow.output_writer.index_epoch != 0
        or flow.input_reader.ceremony_id == flow.output_writer.ceremony_id
    ):
        raise ValueError("Task 5 TN ceremony or index provenance is inconsistent")


def write_finance_runtime_config(
    flow: TNFlowResult,
    *,
    config_path: Path,
    audit_path: Path,
    raw_trace_dir: Path,
    model_path: Path,
    server_path: Path,
    build_id: str,
) -> Path:
    """Write tn-llama's closed v2 runtime config for the finance workflow."""
    destination, decision_events, raw_traces = _preflight_runtime_output_paths(
        flow,
        config_path=config_path,
        audit_path=audit_path,
        raw_trace_dir=raw_trace_dir,
    )
    if destination.exists():
        raise FileExistsError(f"refusing to replace runtime config: {destination}")
    _validate_task5_material(flow)
    model = _require_absolute_file(model_path, "GGUF model")
    server = _require_absolute_file(server_path, "governed llama-server")
    if model.suffix.lower() != ".gguf":
        raise ValueError(f"configured model is not a GGUF file: {model}")
    if not build_id.strip():
        raise ValueError("tn-llama build id must be nonempty")

    document = {
        "version": "tn.llama.runtime.v2",
        "writer_authority": "group_capability",
        "expected_event_type": _FINANCE_EVENT_TYPE,
        "allowed_input_groups": [_FINANCE_GROUP],
        "report_contract": {
            "allowed_input_sections": [_FINANCE_GROUP],
            "output_event_type": _OUTPUT_EVENT_TYPE,
            "output_public_section": "default",
            "operation": "derived_html_report",
            "section_rules": {
                _FINANCE_GROUP: {
                    "propagate_source_governance": True,
                    "required_attribution": _STUB_ATTRIBUTION,
                    "required_tool": None,
                }
            },
            "output_policy": _active_output_policy(flow),
        },
        "reader_key_dir": str(flow.input_reader.reader_key_dir),
        "input_index_key_path": str(flow.input_reader.index_key_path),
        "ceremony_id": flow.input_reader.ceremony_id,
        "governance_group": _GOVERNANCE_GROUP,
        "input_index_epoch": flow.input_reader.index_epoch,
        "output_tn_config_path": str(flow.output_writer.config_path),
        "output_reader_key_dir": str(flow.output_writer.reader.reader_key_dir),
        "output_index_key_path": str(flow.output_writer.reader.index_key_path),
        "output_ceremony_id": flow.output_writer.ceremony_id,
        "output_governance_group": _GOVERNANCE_GROUP,
        "output_index_epoch": flow.output_writer.index_epoch,
        "audit_path": str(decision_events),
        "raw_trace_dir": str(raw_traces),
        "build_id": build_id,
        "llama_source_revision": _LLAMA_SOURCE_REVISION,
        "model_sha256": _sha256(model),
        "binary_sha256": _sha256(server),
        "max_request_bytes": 1024 * 1024,
        "max_message_bytes": 16 * 1024,
        "max_active_sessions": 1,
        "max_chain_namespaces": 8,
        "echo_max_tokens": 1536,
        "final_max_tokens": 2048,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    decision_events.parent.mkdir(parents=True, exist_ok=True)
    raw_traces.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return destination
