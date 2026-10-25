"""Closed wire-format helpers for governed tn-llama requests."""

import json
import re
from dataclasses import dataclass
from typing import Any

_RELEASE_KEYS = {
    "protocol",
    "run_id",
    "source_event_id",
    "source_row_hash",
    "output_event_id",
    "output_row_hash",
    "output_ndjson",
    "trace",
}


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, member in pairs:
        if key in value:
            raise ValueError("release response must not contain duplicate JSON keys")
        value[key] = member
    return value


@dataclass(frozen=True)
class SourceRecord:
    ndjson: bytes
    event_id: str
    row_hash: str
    encrypted_groups: tuple[str, ...]

    @classmethod
    def from_ndjson(cls, raw: bytes) -> "SourceRecord":
        line = raw.removesuffix(b"\n")
        value = json.loads(line)
        if (
            not line
            or b"\n" in line
            or not isinstance(value, dict)
            or value.get("event_type") != "precisely.risk.enriched"
            or not isinstance(value.get("event_id"), str)
            or not isinstance(value.get("row_hash"), str)
        ):
            raise ValueError(
                "source response must be one precisely.risk.enriched NDJSON row"
            )
        encrypted_groups = tuple(
            sorted(
                name
                for name, member in value.items()
                if isinstance(member, dict) and "ciphertext" in member
            )
        )
        return cls(raw, value["event_id"], value["row_hash"], encrypted_groups)


def build_governed_request(source: SourceRecord, prompt: str) -> bytes:
    prefix = json.dumps(
        {
            "messages": [{"content": prompt, "role": "user"}],
            "protocol": "tn.llama.request.v1",
            "purpose": "derived_html_report",
            "stream": False,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return prefix[:-1] + b',"tn_row":' + source.ndjson.removesuffix(b"\n") + b"}"


def build_model_request(object, prompt: str, use, *, destination: str) -> bytes:
    """Transport an exact native publication and complete use to the v2 endpoint."""
    import tn
    if not isinstance(object, tn.GovernedObject) or not isinstance(use, tn.UseContext):
        raise TypeError("model requests require a native GovernedObject and UseContext")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("model prompt must be a nonempty string")
    if (not isinstance(destination, str) or not destination.strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in destination)):
        raise ValueError("model destination must be a nonempty string without controls")
    return json.dumps({
        "protocol": "tn.llama.request.v2", "tn_object": object.wire,
        "use": {"application": use.application, "purpose": use.purpose, "operation": use.operation},
        "destination": destination,
        "messages": [{"role": "user", "content": prompt}], "stream": False,
    }, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True)
class NativeModelRelease:
    """Closed v2 transport; its native object remains encrypted until admission."""

    run_id: str
    source_object_id: str
    output_object_id: str
    output_object: object
    trace: list[Any]
    tool_objects: tuple[object, ...] = ()

    @classmethod
    def from_http_json(cls, raw: bytes) -> "NativeModelRelease":
        import tn
        value = json.loads(raw, object_pairs_hook=_without_duplicate_keys)
        required = {
            "protocol", "run_id", "source_object_id", "output_object_id",
            "output_object", "trace",
        }
        allowed_shapes = (required, required | {"tool_objects"})
        if not isinstance(value, dict) or set(value) not in allowed_shapes:
            raise ValueError("native release response must have the exact v2 shape")
        if value["protocol"] != "tn.llama.release.v2":
            raise ValueError("native release response has an unsupported protocol")
        if (any(not isinstance(value[key], str) or not value[key] for key in required - {"trace"})
                or not isinstance(value["trace"], list)):
            raise ValueError("native release response fields have unexpected types")
        if any(not re.fullmatch(r"sha256:[0-9a-f]{64}", value[key])
               for key in ["source_object_id", "output_object_id"]):
            raise ValueError("native release response needs canonical object identities")
        output = tn.GovernedObject.parse(value["output_object"])
        if output.id != value["output_object_id"]:
            raise ValueError("native output identity differs from its response wrapper")
        raw_tools = value.get("tool_objects", [])
        if not isinstance(raw_tools, list) or any(
            not isinstance(item, str) for item in raw_tools
        ):
            raise ValueError("native tool objects must be a list of exact wire strings")
        tools = tuple(tn.GovernedObject.parse(item) for item in raw_tools)
        tool_ids = [item.id for item in tools]
        if len(set(tool_ids)) != len(tool_ids):
            raise ValueError("native tool objects must have unique identities")
        return cls(
            value["run_id"], value["source_object_id"], output.id, output,
            value["trace"], tools,
        )


@dataclass(frozen=True)
class ReportRelease:
    run_id: str
    source_event_id: str
    source_row_hash: str
    output_event_id: str
    output_row_hash: str
    output_ndjson: str
    trace: list[Any]
    report_html: str
    governance: dict[str, Any] | None = None


@dataclass(frozen=True)
class EncryptedReportRelease:
    """Strict HTTP wrapper around an exact, still-encrypted TN output row."""

    run_id: str
    source_event_id: str
    source_row_hash: str
    output_event_id: str
    output_row_hash: str
    output_ndjson: str
    trace: list[Any]
    row: dict[str, Any]

    @classmethod
    def from_http_json(cls, raw: bytes) -> "EncryptedReportRelease":
        value = json.loads(raw, object_pairs_hook=_without_duplicate_keys)
        if not isinstance(value, dict) or set(value) != _RELEASE_KEYS:
            raise ValueError("release response must have the exact release shape")
        if value["protocol"] != "tn.llama.release.v1":
            raise ValueError("release response has an unsupported protocol")

        string_keys = (
            "run_id",
            "source_event_id",
            "source_row_hash",
            "output_event_id",
            "output_row_hash",
            "output_ndjson",
        )
        if any(not isinstance(value[key], str) for key in string_keys):
            raise ValueError("release response fields must have the expected types")
        if not isinstance(value["trace"], list):
            raise ValueError("release trace must be a list")

        output_ndjson = value["output_ndjson"]
        if not output_ndjson.endswith("\n") or output_ndjson.count("\n") != 1:
            raise ValueError(
                "release output must be exactly one newline-terminated NDJSON row"
            )
        row = json.loads(output_ndjson[:-1], object_pairs_hook=_without_duplicate_keys)
        if not isinstance(row, dict):
            raise ValueError("release output row must be an object")
        if (
            row.get("event_type") != "report.generated"
            or row.get("event_id") != value["output_event_id"]
            or row.get("row_hash") != value["output_row_hash"]
        ):
            raise ValueError("release output row does not match its wrapper")
        if not isinstance(row.get("signature"), str) or not row["signature"]:
            raise ValueError("release output row must carry a signature")
        if not isinstance(row.get("tn_aad"), str) or not row["tn_aad"]:
            raise ValueError("release output row must carry canonical TN AAD")
        encrypted_names = {
            name
            for name, member in row.items()
            if isinstance(member, dict) and "ciphertext" in member
        }
        if encrypted_names != {"default", "tn.agents"}:
            raise ValueError(
                "release output row must encrypt exactly default and tn.agents"
            )
        for name in sorted(encrypted_names):
            block = row[name]
            if (
                set(block) != {"ciphertext", "field_hashes"}
                or not isinstance(block["ciphertext"], str)
                or not block["ciphertext"]
                or not isinstance(block["field_hashes"], dict)
            ):
                raise ValueError(f"release output group {name} is malformed")

        return cls(
            run_id=value["run_id"],
            source_event_id=value["source_event_id"],
            source_row_hash=value["source_row_hash"],
            output_event_id=value["output_event_id"],
            output_row_hash=value["output_row_hash"],
            output_ndjson=output_ndjson,
            trace=value["trace"],
            row=row,
        )
