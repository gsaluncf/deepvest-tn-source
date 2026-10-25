"""Independent nine-question evaluation of a completed reference-path run."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class QuestionResult:
    number: int
    question: str
    status: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WitnessResult:
    run_directory: str
    questions: tuple[QuestionResult, ...]

    @property
    def all_full(self) -> bool:
        return all(item.status == "full" for item in self.questions)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_directory": self.run_directory,
            "all_full": self.all_full,
            "questions": [asdict(item) for item in self.questions],
        }


def _read_ndjson(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _body(entry: dict[str, object]) -> dict[str, object]:
    encoded = entry.get("body_b64")
    if not isinstance(encoded, str):
        return {}
    return json.loads(base64.b64decode(encoded).decode("utf-8"))


def _status(condition: bool) -> str:
    return "full" if condition else "not_shown"


def verify_reference_run(run_directory: Path) -> WitnessResult:
    """Derive nine answers from native artifacts rather than producer assertions."""
    root = Path(run_directory).resolve()
    summary = json.loads((root / "run-summary.json").read_text(encoding="utf-8"))
    runtime = _read_ndjson(root / "runtime" / "decision-events.ndjson")
    client = _read_ndjson(root / "client" / "http.ndjson")
    source_rows = _read_ndjson(root / "tn" / "deepvest" / ".tn" / "tn" / "logs" / "tn.ndjson")
    output_rows = _read_ndjson(root / "client" / "received-output.ndjson")

    request_entries = [
        item
        for item in client
        if item.get("peer") == "llama" and item.get("direction") == "request"
    ]
    response_entries = [
        item
        for item in client
        if item.get("peer") == "llama" and item.get("direction") == "response"
    ]
    request = _body(request_entries[0]) if len(request_entries) == 1 else {}
    response = _body(response_entries[0]) if len(response_entries) == 1 else {}
    source_id = summary["tn_handoffs"]["model_source_event_id"]
    source_hash = summary["tn_handoffs"]["model_source_row_hash"]
    output_id = summary["verified_output"]["event_id"]
    output_hash = summary["verified_output"]["row_hash"]
    source = next((row for row in source_rows if row.get("event_id") == source_id), {})
    output = next((row for row in output_rows if row.get("event_id") == output_id), {})
    kinds = {str(item.get("kind")): item for item in runtime}
    client_events = {str(item.get("event")): item for item in client if item.get("event")}
    opened = [
        item
        for item in client
        if item.get("event") == "release_group_opened"
    ]
    policy = summary["tn_handoffs"].get("policy_reference")
    aad = json.loads(source.get("tn_aad", "{}")) if source else {}
    governed_by = source.get("device_identity")
    marker_ok = bool(aad) and all(
        marker.get("policy") == policy and marker.get("governed_by") == governed_by
        for marker in aad.values()
        if isinstance(marker, dict)
    )
    row_verified = kinds.get("row_verified", {})
    row_facts = json.loads(str(row_verified.get("facts_json", "{}")))
    commitments = kinds.get("commitments_matched", {})
    commitment_facts = json.loads(str(commitments.get("facts_json", "{}")))
    packages = root / "tn" / "packages"
    required_packages = {
        "source-to-polars.tnpkg",
        "polars-to-deepvest.tnpkg",
        "deepvest-to-llama.tnpkg",
        "llama-to-report.tnpkg",
    }
    package_ok = required_packages.issubset(
        {path.name for path in packages.glob("*.tnpkg")}
    )
    exact_route = (
        len(request_entries) == 1
        and len(response_entries) == 1
        and request_entries[0].get("url", "").endswith("/v1/tn/chat/completions")
        and request.get("protocol") == "tn.llama.request.v1"
        and response_entries[0].get("status") == 200
        and "output_emitted_and_read_back" in kinds
    )
    release_verified = client_events.get("encrypted_release_verified", {})

    checks = (
        (
            "What governance basis applies?",
            isinstance(policy, str)
            and "#sha256:" in policy
            and marker_ok
            and "governance_opened_and_verified" in kinds,
            ("content-addressed policy", "authenticated TN AAD", "governance opened first"),
        ),
        (
            "Which authorized rule permits this transition?",
            kinds.get("sections_selected", {}).get("outcome") == "allow"
            and request.get("purpose") == "derived_html_report",
            ("allowed section selection", "derived_html_report purpose"),
        ),
        (
            "Is the rule bound to this exact financial output?",
            commitment_facts.get("hashes_matched") is True
            and response.get("source_row_hash") == source_hash
            and response.get("output_row_hash") == output_hash,
            ("commitments matched", "source row joined to exact output row"),
        ),
        (
            "Is the claim authentic?",
            row_facts.get("signature_verified") is True
            and len(opened) == 2
            and all(item.get("signature") is True for item in opened),
            ("source signature verified", "recipient verified output signature"),
        ),
        (
            "Is the claimant authorized for this role?",
            package_ok and len(opened) == 2,
            ("four pre-run TN packages", "recipient capability opened output"),
        ),
        (
            "Can this receiver interpret the rule?",
            "governance_opened_and_verified" in kinds
            and commitment_facts.get("governance_commitment_count") == 6,
            ("six governance fields opened", "six commitments interpreted"),
        ),
        (
            "Were recipient and purpose evaluated at decision time?",
            request.get("purpose") == "derived_html_report"
            and [item.get("group") for item in opened] == ["tn.agents", "default"],
            ("purpose in governed request", "configured recipient opened release"),
        ),
        (
            "Did every declared output route pass through the gate?",
            exact_route and len(response_entries) == 1,
            ("single governed route", "single gated output emission and response"),
        ),
        (
            "Does the evidence link calculation through continuation?",
            summary["deepvest"].get("invocation_count") == 1
            and source.get("row_hash") == source_hash
            and output.get("row_hash") == output_hash
            and response.get("source_event_id") == source_id
            and response.get("output_event_id") == output_id
            and release_verified.get("output_row_hash") == output_hash
            and "source_chain_committed" in kinds,
            ("one DeepVest invocation", "source/output IDs joined", "source chain committed"),
        ),
    )
    return WitnessResult(
        run_directory=str(root),
        questions=tuple(
            QuestionResult(number, question, _status(ok), evidence)
            for number, (question, ok, evidence) in enumerate(checks, start=1)
        ),
    )


def write_witness(run_directory: Path, output_path: Path | None = None) -> Path:
    result = verify_reference_run(run_directory)
    target = output_path or Path(run_directory) / "nine-question-witness.json"
    target.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target
