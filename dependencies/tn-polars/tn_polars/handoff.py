"""Policy-directed egress boundary.

``governed_handoff`` evaluates the compiled policy like ``release`` does,
then follows the ``on_violation_or_error`` directive from the tn.agents
policy instead of always blocking. The warn-and-allow directive fires the
compliance webhook with the full lineage graph, attaches a restricted-data
marker to the release manifest, and emits the envelope; nothing is silently
dropped and nothing hard-blocks the pipeline.
"""
import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .release import PolicyDenied
from .sealing import governance_marker

WARNING_MARKER = "WARNING: RESTRICTED DATA"
WARN_DIRECTIVE = "log_to_compliance_webhook_and_allow"


@dataclass(frozen=True)
class HandoffResult:
    artifact: object
    decision: object
    warning: object
    webhook_receipt: object


def governed_handoff(frame, path, *, audience, directive=None, ledger=None,
                     webhook=None, cipher=None):
    """Release ``frame`` for ``audience``, honoring the violation directive."""
    if frame.tn_governance.get("policy") != frame.policy.policy_ref:
        raise PolicyDenied("TN governance body and compiled policy reference differ")
    marker = governance_marker(frame.policy.policy_ref, audience) if cipher else None
    materialized = frame._support.frame.collect()
    decision = frame.evaluator.decide(frame._policy_input(audience, materialized))
    released = materialized.select(list(frame.sources))
    if decision.allow:
        artifact = frame._write_release(path, decision, cipher=cipher, seal_marker=marker, materialized=released)
        _log_release(ledger, audience, artifact, decision, warning=None)
        return HandoffResult(artifact, decision, None, None)
    if directive != WARN_DIRECTIVE:
        raise PolicyDenied(decision.reason)
    reasons = decision.reason.split("; ")
    receipt = _notify_compliance(frame, path, audience, reasons, ledger, webhook)
    artifact = frame._write_release(path, decision, cipher=cipher, seal_marker=marker, materialized=released, extra={
        "warning": WARNING_MARKER,
        "on_violation_or_error": WARN_DIRECTIVE,
        "violation_reasons": reasons,
        "webhook_delivery": receipt,
    })
    _log_release(ledger, audience, artifact, decision, warning=WARNING_MARKER)
    return HandoffResult(artifact, decision, WARNING_MARKER, receipt)


def _notify_compliance(frame, path, audience, reasons, ledger, webhook):
    payload = {
        "schema": "tn.polars-compliance.v1",
        "policy_ref": frame.policy.policy_ref,
        "audience": audience,
        "artifact": Path(path).name,
        "violation_reasons": reasons,
        "lineage": {name: sorted(roots) for name, roots in frame.sources.items()},
        "row_selection_sources": sorted(frame.row_sources),
        "tn_governance": frame.tn_governance,
    }
    receipt = webhook.post(payload) if webhook else None
    if ledger is not None:
        ledger.log(
            "desk.compliance.violation", audience=audience, artifact=Path(path).name,
            violation_reasons="; ".join(reasons),
            delivery=receipt["delivery"] if receipt else "none",
            delivery_id=receipt["delivery_id"] if receipt else "")
    return receipt


def _log_release(ledger, audience, artifact, decision, *, warning):
    if ledger is None:
        return
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    body = artifact.path.read_bytes()
    fields = {
        "audience": audience,
        "artifact": artifact.path.name,
        "file_sha256": hashlib.sha256(body).hexdigest(),
        "decision_id": decision.decision_id,
        "allowed": "true" if decision.allow else "false",
        "lineage_json": json.dumps(manifest["lineage"], sort_keys=True),
        "row_selection_sources": ",".join(manifest["row_selection_sources"]),
    }
    if manifest.get("sealed"):
        fields["payload_b64"] = base64.b64encode(body).decode("ascii")
        fields["payload_cipher"] = manifest["sealed"]["cipher"]
    if warning is not None:
        fields["warning"] = warning
    ledger.log("desk.release", **fields)
