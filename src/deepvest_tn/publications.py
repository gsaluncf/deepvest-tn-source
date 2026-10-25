"""Retain exact canonical TN publications; transport never creates a new version."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import tn


def object_digest(object_id: str) -> str:
    """Validate an identity before using its digest as a local path component."""
    if not isinstance(object_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", object_id):
        raise ValueError("publication identity requires sha256 and 64 lowercase hexadecimal digits")
    return object_id[7:]


def write_once(path: Path, content: bytes) -> None:
    """Atomically expose complete bytes without replacing any existing representation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".publication-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(
                    f"immutable publication representation differs: {path.name}"
                ) from None
    finally:
        Path(temporary).unlink(missing_ok=True)


class PublicationStore:
    """Append-only signed wire storage with an immutable, explicitly verified index."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def put(self, object: tn.GovernedObject) -> str:
        if not isinstance(object, tn.GovernedObject):
            raise TypeError("publication storage requires a native GovernedObject")
        wire = object.forward()
        verified = tn.GovernedObject.read(io.BytesIO(wire))
        digest = object_digest(verified.id)
        relative = f"objects/{digest}.tn"
        write_once(self.root / relative, wire)
        index = {
            "object_id": verified.id,
            "file": relative,
            "wire_sha256": hashlib.sha256(wire).hexdigest(),
        }
        write_once(
            self.root / "index" / f"{digest}.json",
            json.dumps(index, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        return verified.id

    def get(self, object_id: str) -> bytes:
        digest = object_digest(object_id)
        index = json.loads((self.root / "index" / f"{digest}.json").read_bytes())
        relative = f"objects/{digest}.tn"
        if index.get("object_id") != object_id or index.get("file") != relative:
            raise ValueError("publication index does not name the requested identity and file")
        path = self.root / relative
        wire = path.read_bytes()
        if hashlib.sha256(wire).hexdigest() != index.get("wire_sha256"):
            raise ValueError("retained publication wire digest differs from its immutable index")
        publication = tn.GovernedObject.read(path)
        if publication.id != object_id or publication.forward() != wire:
            raise ValueError("retained publication has the wrong native identity")
        return wire
