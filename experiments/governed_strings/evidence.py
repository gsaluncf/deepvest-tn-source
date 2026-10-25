"""Context-owned evidence carried in the native TN format."""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from uuid import uuid4

import tn

from deepvest_tn.publications import PublicationStore, write_once

EVIDENCE_TYPE = "experiment.evidence"
EVIDENCE_SCHEMA = "tn.string-evidence.v1"
KINDS = {"Decision", "Result", "Refusal", "Disclosure", "Failure", "Carry", "Claim",
         "Monitor", "Observation", "SQLRead", "ModelHandoff", "ModelResult", "ReviewRequested"}


class EvidenceError(ValueError):
    """Evidence failed verification or did not support the claimed association."""


@dataclass(frozen=True)
class Record:
    id: str
    writer: str
    body: dict


def read_record(wire, *, session, policy, expected_writer=None):
    """Verify native TN before opening only the evidence group.

    Accepting this read contract permits inspection, not endorsement of its claims.
    """
    try:
        obj = tn.GovernedObject.read(io.BytesIO(wire))
        if obj.object_type != EVIDENCE_TYPE:
            raise EvidenceError("wrong evidence object type")
        if expected_writer is not None and obj.writer != expected_writer:
            raise EvidenceError("unexpected evidence writer")
        working = session.receive(
            obj, use=tn.UseContext("string-evidence-reader", "inspect-evidence", "read"),
            groups=["evidence"],
            decide=lambda ctx: len(ctx.policies) == 1
            and ctx.governance.matches_contract(policy),
        )
        body = working.get("record", group="evidence")
        required = {
            "schema", "context_id", "sequence", "previous", "kind", "call_id",
            "application", "purpose", "operation", "inputs", "destination", "details",
        }
        if not isinstance(body, dict) or set(body) != required:
            raise EvidenceError("invalid evidence record shape")
        if body["schema"] != EVIDENCE_SCHEMA or body["kind"] not in KINDS:
            raise EvidenceError("unknown evidence schema or kind")
        if type(body["sequence"]) is not int or body["sequence"] < 1:
            raise EvidenceError("invalid evidence sequence")
        for field in ("context_id", "call_id", "application", "purpose", "operation"):
            if not isinstance(body[field], str) or not body[field]:
                raise EvidenceError(f"invalid evidence {field}")
        if not isinstance(body["inputs"], list) or not isinstance(body["details"], dict):
            raise EvidenceError("invalid evidence inputs or details")
        return Record(obj.id, obj.writer, body)
    except EvidenceError:
        raise
    except Exception as exc:
        raise EvidenceError("TN evidence verification failed") from exc


def verify_tape(wires, *, session, policy, expected_writer, expected_context, expected_head):
    """Verify a bounded record chain against an independently retained head/context/writer.

    Without a trusted head, a valid prefix cannot be distinguished from a complete record chain.
    Signatures establish attribution, not the truth of the endpoint's observations.
    """
    records = []
    previous = None
    for sequence, wire in enumerate(wires, 1):
        record = read_record(
            wire, session=session, policy=policy, expected_writer=expected_writer,
        )
        body = record.body
        if (body["context_id"] != expected_context or body["sequence"] != sequence
                or body["previous"] != previous):
            raise EvidenceError("evidence tape has a gap, reordering, or context substitution")
        records.append(record)
        previous = record.id
    if previous != expected_head:
        raise EvidenceError("evidence tape does not reach the expected head")
    return tuple(records)


class EvidenceRecorder:
    """Single-process append-only recorder. PublicationStore retains exact native wire.

    This is an application record schema within TN, not a replacement envelope,
    signer, key service, or permit token. Each context starts a fresh record chain.
    """

    def __init__(self, session, store: PublicationStore, policy):
        self.session = session
        self.store = store
        self.policy = policy
        self.context_id = uuid4().hex
        self._ids = []
        self._lock = threading.RLock()

    @property
    def head(self):
        return self._ids[-1] if self._ids else None

    def append(self, kind, *, call_id, application, purpose, operation,
               inputs=(), destination=None, details=None):
        if kind not in KINDS:
            raise ValueError("unknown evidence kind")
        with self._lock:
            body = {
                "schema": EVIDENCE_SCHEMA,
                "context_id": self.context_id,
                "sequence": len(self._ids) + 1,
                "previous": self.head,
                "kind": kind,
                "call_id": call_id,
                "application": application,
                "purpose": purpose,
                "operation": operation,
                "inputs": [value.to_dict() for value in inputs],
                "destination": destination,
                "details": details or {},
            }
            data = self.session.create(
                {"record": body}, self.policy, object_type=EVIDENCE_TYPE, group="evidence",
            )
            self.store.put(data.snapshot)
            publication = self.session.release(
                data, use=tn.UseContext(application, purpose, "record-evidence"),
                to="context-recorder", object_type=EVIDENCE_TYPE,
                decide=lambda ctx: ctx.writer == self.session.did
                and len(ctx.policies) == 1 and ctx.policies[0].matches_contract(self.policy),
            )
            self.store.put(publication)
            # A complete index entry is exposed only after its exact TN bytes exist.
            path = self.store.root / "tapes" / self.context_id / f"{body['sequence']:08d}.ref"
            write_once(path, publication.id.encode("ascii"))
            self._ids.append(publication.id)
            return publication.id

    def records(self):
        with self._lock:
            return tuple(self.store.get(identity) for identity in self._ids)

    def read(self, identity):
        return read_record(self.store.get(identity), session=self.session, policy=self.policy)

    def export(self, directory):
        """Export exact records plus a head to retain over a separately trusted channel.

        The JSON checkpoint is an index, not a new authority or self-authenticating trust root.
        Copy the PublicationStore too when result/input publications must be inspected.
        """
        from pathlib import Path

        directory = Path(directory)
        with self._lock:
            for sequence, wire in enumerate(self.records(), 1):
                write_once(directory / f"{sequence:08d}.tn", wire)
            checkpoint = {
                "writer": self.session.did, "context_id": self.context_id,
                "head": self.head, "count": len(self._ids),
            }
            write_once(directory / "checkpoint.json", json.dumps(checkpoint, indent=2).encode())
            return checkpoint
