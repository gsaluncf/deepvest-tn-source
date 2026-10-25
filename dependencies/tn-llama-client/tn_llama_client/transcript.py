"""Append-only local HTTP body transcript storage."""

import base64
import hashlib
import json
import urllib.request
from collections.abc import Iterable
from pathlib import Path


class HttpTranscript:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append_exchange(
        self,
        *,
        peer: str,
        direction: str,
        method: str,
        url: str,
        status: int | None,
        headers: Iterable[tuple[str, str]],
        body: bytes,
    ) -> None:
        entry = {
            "peer": peer,
            "direction": direction,
            "method": method,
            "url": url,
            "status": status,
            "headers": list(headers),
            "body_b64": base64.b64encode(body).decode("ascii"),
            "body_sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as transcript:
            transcript.write(json.dumps(entry, separators=(",", ":")) + "\n")
            transcript.flush()

    def log_request(
        self, peer: str, request: urllib.request.Request, body: bytes
    ) -> None:
        self.append_exchange(
            peer=peer,
            direction="request",
            method=request.get_method(),
            url=request.full_url,
            status=None,
            headers=_safe_headers(request.header_items()),
            body=body,
        )

    def log_response(
        self,
        peer: str,
        url: str,
        response: object,
        body: bytes,
        *,
        method: str = "GET",
    ) -> None:
        headers = getattr(response, "headers")
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        self.append_exchange(
            peer=peer,
            direction="response",
            method=method,
            url=url,
            status=status,
            headers=_safe_headers(headers.items()),
            body=body,
        )

    def log_state(self, event: str, **facts: object) -> None:
        """Record a non-secret client transition beside its HTTP evidence."""

        entry = {"kind": "client_state", "event": event, **facts}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as transcript:
            transcript.write(json.dumps(entry, separators=(",", ":")) + "\n")
            transcript.flush()


def _safe_headers(headers: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    return [
        (name.lower(), value)
        for name, value in headers
        if name.lower() != "authorization"
    ]
