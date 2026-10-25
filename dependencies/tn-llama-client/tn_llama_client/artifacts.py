"""Persist released report and exact local HTTP evidence artifacts."""

import base64
import json
from pathlib import Path

from .protocol import ReportRelease


def save_report_html(release: ReportRelease, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{release.output_event_id}.html"
    target.write_text(release.report_html, encoding="utf-8", newline="")
    return target


def save_http_body(body: bytes, path: Path) -> Path:
    """Persist an HTTP body without parsing or reserializing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def materialize_llama_bodies(
    transcript_path: Path, directory: Path
) -> tuple[Path, Path]:
    return (
        _save_transcript_body(transcript_path, directory, "request"),
        _save_transcript_body(transcript_path, directory, "response"),
    )


def existing_llama_artifacts(directory: Path) -> tuple[Path | None, Path | None]:
    request_path = directory / "request.json"
    response_path = directory / "response.json"
    return (
        request_path if request_path.exists() else None,
        response_path if response_path.exists() else None,
    )


def save_server_trace(release: ReportRelease, directory: Path) -> Path:
    target = directory / "trace.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(release.trace, indent=2) + "\n", encoding="utf-8")
    return target


def _save_transcript_body(
    transcript_path: Path, directory: Path, direction: str
) -> Path:
    entries = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entry = next(
        (
            value
            for value in reversed(entries)
            if isinstance(value, dict)
            and value.get("peer") == "llama"
            and value.get("direction") == direction
        ),
        None,
    )
    if not isinstance(entry, dict) or not isinstance(entry.get("body_b64"), str):
        raise ValueError(f"missing llama {direction} transcript entry")
    try:
        body = base64.b64decode(entry["body_b64"], validate=True)
    except ValueError as error:
        raise ValueError(f"invalid llama {direction} transcript body") from error
    return save_http_body(body, directory / f"{direction}.json")
