"""Behavioral probes for the experimental governed string boundary."""

from dataclasses import replace

import pytest
import tn

from deepvest_tn.publications import PublicationStore


POLICY = """## experiment.string
### instruction
Strings supplied for a local governed-runtime experiment.
### use_for
Context-authorized string operations and internal support display.
### do_not_use_for
Exposure without an explicit permitted destination.
### consequences
Source claims remain claims about their original subjects.
### on_violation_or_error
Record the decision and stop the protected operation.

## experiment.evidence
### instruction
Attributable experimental endpoint records, not independent proof of compliance.
### use_for
Inspection by enrolled evidence readers.
### do_not_use_for
Treating a signature as automatic acceptance of a claim.
### consequences
Recipients evaluate authority in their own context.
### on_violation_or_error
Refuse unverified evidence.
"""


@pytest.fixture
def env(tmp_path):
    from experiments.governed_strings import Context, EvidenceRecorder, Rules, publish_source

    with tn.Session(POLICY, groups=["strings", "evidence"]) as session:
        store = PublicationStore(tmp_path / "publications")
        recorder = EvidenceRecorder(session, store, session.policy("experiment.evidence"))
        rules = Rules(
            writers=frozenset({session.did}),
            policies=(session.policy("experiment.string"),),
            operations=frozenset({"copy", "upper", "lower", "slice", "concat", "attest", "reveal"}),
            destinations=frozenset({"support-console"}),
        )
        context = Context(session, recorder, rules, application="support", purpose="customer-support")
        values = publish_source(session, store, {"name": "Ada", "value": "Lovelace"})
        yield context, values, rules


def records(context):
    from experiments.governed_strings import verify_tape

    return verify_tape(
        context.recorder.records(), session=context.session,
        expected_writer=context.session.did, expected_head=context.recorder.head,
        expected_context=context.recorder.context_id,
        policy=context.recorder.policy,
    )


def test_compute_can_succeed_when_exposure_is_refused(env):
    from experiments.governed_strings import OperationDenied, TNString

    context, values, rules = env
    context.evaluator = replace(rules, destinations=frozenset())
    with context:
        upper = values["name"].upper()
        assert isinstance(upper, TNString)
        assert "Ada" not in repr(upper)
        with pytest.raises(OperationDenied):
            upper.reveal(context, destination="support-console")
    tape = records(context)
    assert [r.body["kind"] for r in tape] == ["Decision", "Result", "Decision", "Refusal"]
    assert tape[1].body["details"]["output"] == upper.reference.to_dict()
    assert upper.decision_id == tape[0].id
    assert upper.result_id == tape[1].id
    context.evaluator = rules
    assert upper.reveal(context, destination="support-console") == "ADA"
    assert [r.body["kind"] for r in records(context)[-2:]] == ["Decision", "Disclosure"]


def test_denial_and_failed_obligation_never_open_business_data(env, monkeypatch):
    from experiments.governed_strings import Assessment, ObligationResult, OperationDenied

    context, values, _ = env
    opened = []
    monkeypatch.setattr(context, "_open_values", lambda *args: opened.append(args))
    context.evaluator = lambda request: Assessment(False, "operation not permitted")
    with context, pytest.raises(OperationDenied):
        values["name"].upper()
    context.evaluator = lambda request: Assessment(True, "conditional", obligations=("monitor",))
    context.obligations["monitor"] = lambda request: ObligationResult(False, "unavailable")
    with context, pytest.raises(OperationDenied):
        values["name"].upper()
    assert opened == []
    assert [r.body["kind"] for r in records(context)] == [
        "Decision", "Refusal", "Decision", "Refusal",
    ]


def test_obligation_completes_and_decision_is_durable_before_open(env, monkeypatch):
    from experiments.governed_strings import Assessment, ObligationResult

    context, values, _ = env
    order = []
    context.evaluator = lambda request: Assessment(True, "allowed", obligations=("monitor",))
    context.obligations["monitor"] = lambda request: (
        order.append("monitor") or ObligationResult(True, "monitoring:started")
    )
    original = context._open_values

    def checked_open(*args):
        assert order == ["monitor"]
        assert records(context)[-1].body["kind"] == "Decision"
        assert records(context)[-1].body["details"]["permitted"] is True
        order.append("open")
        return original(*args)

    monkeypatch.setattr(context, "_open_values", checked_open)
    with context:
        values["name"].upper()
    assert order == ["monitor", "open"]


def test_claims_follow_concat_without_becoming_binding(env):
    context, values, _ = env
    with context:
        claimed = values["name"].attest("This name is suitable for display.", context)
        combined = claimed.concat(values["value"])
    request = context.inspect(combined)
    assert len(request.claims) == 1
    claim = request.claims[0]
    assert claim.issuer == context.session.did
    assert claim.statement == "This name is suitable for display."
    assert claim.subjects == (values["name"].reference,)
    assert combined.reveal(context, destination="support-console") == "AdaLovelace"
    result = next(r for r in records(context) if r.id == combined.result_id)
    assert result.body["inputs"] == [claimed.reference.to_dict(), values["value"].reference.to_dict()]
    assert result.body["details"]["claims"] == [claim.id]
    governance = context.session.governance(context.store_object(combined.reference.object_id)).governance
    assert governance.matches_contract(context.session.policy("experiment.string"))


def test_copy_preserves_identity_and_gated_str_uses_destination(env):
    from experiments.governed_strings import OperationDenied

    context, values, _ = env
    with pytest.raises(OperationDenied):
        str(values["name"])
    with context:
        copied = values["name"].copy()
        assert copied.reference == values["name"].reference
        with pytest.raises(OperationDenied):
            str(copied)
        with context.exposure("support-console"):
            assert str(copied) == "Ada"
    assert "Carry" in [r.body["kind"] for r in records(context)]
    assert records(context)[-1].body["kind"] == "Disclosure"


def test_recording_failure_never_returns_plaintext_or_runs_compute(env, monkeypatch):
    context, values, _ = env
    opened = []
    monkeypatch.setattr(context, "_open_values", lambda *args: opened.append(args))

    def disk_failed(*args, **kwargs):
        raise OSError("recorder unavailable")

    monkeypatch.setattr(context.recorder, "append", disk_failed)
    with context, pytest.raises(OSError):
        values["name"].upper()
    assert opened == []


def test_tape_rejects_tampering_truncation_and_wrong_writer(env):
    from experiments.governed_strings import EvidenceError, verify_tape

    context, values, _ = env
    with context:
        values["name"].upper()
    wires = list(context.recorder.records())
    kwargs = dict(
        session=context.session, expected_writer=context.session.did,
        expected_head=context.recorder.head, expected_context=context.recorder.context_id,
        policy=context.recorder.policy,
    )
    assert len(verify_tape(wires, **kwargs)) == 2
    with pytest.raises(EvidenceError):
        verify_tape(wires[:-1], **kwargs)
    with pytest.raises(EvidenceError):
        verify_tape(wires[::-1], **kwargs)
    with pytest.raises(EvidenceError):
        verify_tape(wires, **(kwargs | {"expected_writer": "untrusted"}))
    import json

    changed = json.loads(wires[0])
    changed["timestamp"] = "2026-01-01T00:00:00Z"
    with pytest.raises(EvidenceError):
        verify_tape([json.dumps(changed).encode(), *wires[1:]], **kwargs)


def test_forged_result_citation_and_invalid_outcomes_are_refused(env):
    from experiments.governed_strings import EvidenceError, OperationDenied

    context, values, _ = env
    with context:
        upper = values["name"].upper()
        other = values["value"].lower()
        forged = replace(upper, result_id=other.result_id)
        with pytest.raises(EvidenceError):
            context.inspect(forged)
        context.evaluator = lambda request: "not refused"
        with pytest.raises(OperationDenied):
            values["name"].upper()


def test_scalar_conversions_preserve_exact_operation_and_content(env):
    context, values, _ = env
    with context:
        result = values["value"].upper().slice(1, 5).lower()
    assert result.reveal(context, destination="support-console") == "ovel"
    operations = [r.body["operation"] for r in records(context) if r.body["kind"] == "Result"]
    assert operations == ["upper", "slice", "lower"]
