"""Governance-bound sealing for released artifacts.

Uses the tn-proto btn broadcast-tree primitives. The governance marker
(policy reference plus audience) is authenticated data on the ciphertext:
opening requires both the ceremony's key material and the exact marker. An
agent operating outside the governed path holds bytes it cannot read, and a
swapped marker refuses to open even for a keyed reader.
"""
from pathlib import Path

from .release import PolicyDenied

CIPHER_NAME = "tn-btn"


def governance_marker(policy_ref, audience):
    return f"{policy_ref}#audience={audience}"


class SealedBox:
    """One persistent btn producer whose state lives under the ceremony."""

    def __init__(self, producer):
        self._producer = producer

    @classmethod
    def open_dir(cls, key_dir):
        from tn import btn
        key_dir = Path(key_dir)
        key_dir.mkdir(parents=True, exist_ok=True)
        state = key_dir / "producer.btn"
        if state.exists():
            return cls(btn.Producer.from_bytes(state.read_bytes()))
        producer = btn.setup()
        state.write_bytes(producer.to_bytes())
        return cls(producer)

    def seal(self, data, marker):
        return self._producer.encrypt(data, aad=marker.encode("utf-8"))

    def open(self, data, marker):
        try:
            return self._producer.decrypt(data, aad=marker.encode("utf-8"))
        except Exception as error:
            raise PolicyDenied(
                "sealed artifact refused to open: governance marker mismatch "
                "or missing entitlement") from error
