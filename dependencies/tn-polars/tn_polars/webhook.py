"""Compliance webhook client with a file fallback so tests need no sockets."""
import hashlib
import json
import urllib.request
from pathlib import Path


class ComplianceWebhook:
    """POST violation payloads to a compliance endpoint, or append to a file."""

    def __init__(self, url, fallback_path):
        self.url = url
        self.fallback_path = Path(fallback_path)

    def post(self, payload):
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        delivery_id = hashlib.sha256(body).hexdigest()[:16]
        try:
            request = urllib.request.Request(
                self.url, data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=2):
                pass
            return {"delivery": "http", "delivery_id": delivery_id, "endpoint": self.url}
        except OSError:
            return self._fallback(payload, delivery_id)

    def _fallback(self, payload, delivery_id):
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.fallback_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({"delivery_id": delivery_id, "payload": payload}) + "\n")
        return {"delivery": "file_fallback", "delivery_id": delivery_id,
                "endpoint": str(self.fallback_path)}
