"""Native source groups and field attribution survive primitive input merging."""

from uuid import uuid4

import pytest
import tn

from deepvest_tn.publications import PublicationStore
from deepvest_tn.governed.evidence import EvidenceRecorder
from deepvest_tn.governed.metadata import describe
from deepvest_tn.governed.primitives import Context, SCHEMA, VALUE_TYPE
from deepvest_tn.governed.runtime import Assessment, Request
from deepvest_tn.governed.string import OperationDenied, ValueReference
from experiments.pvldb_evaluation.explicit import ExplicitOperations


POLICY = """## experiment.primitive
### instruction
Retain the original claims and exact inputs through calculations.
### use_for
Internal calculations and authorized display.
### do_not_use_for
Unapproved disclosure.
### consequences
Record decisions and results.
### on_violation_or_error
Stop and record the failure.

## experiment.evidence
### instruction
Keep signed operation records for enrolled readers.
### use_for
Inspect original statements and operations.
### do_not_use_for
Treating a source claim as an endorsement of a derived result.
### consequences
Recipients evaluate the claimant and requested use.
### on_violation_or_error
Refuse invalid evidence.
"""


@pytest.fixture
def context(tmp_path):
    with tn.Session(POLICY, groups=["values", "strings", "evidence"]) as session:
        policy = session.policy(VALUE_TYPE)
        recorder = EvidenceRecorder(
            session, PublicationStore(tmp_path / "publications"),
            session.policy("experiment.evidence"),
        )
        yield Context(
            session, recorder,
            lambda request: Assessment(
                request.operation != "reveal" or request.destination == "console",
                "configured receiver",
            ),
            application="analytics", purpose="internal", actor="analyst",
            metadata_accept=lambda arrived: arrived.writer == session.did
            and arrived.object_type == VALUE_TYPE
            and len(arrived.policies) == 1
            and arrived.governance.matches_contract(policy),
        )


def _batch(context):
    policy = context.session.policy(VALUE_TYPE)
    request = Request(context.application, context.purpose, "select", (), (), (policy,), ())
    call_id = uuid4().hex
    decision = context._decision(request, call_id)
    cells = [
        (content, {"table": "market", "column": "value", "row": row,
                   "query_id": call_id, "assignment_id": "assignment",
                   "snapshot_sha256": "a" * 64})
        for row, content in enumerate((7, 8))
    ]
    return context.originate_batch(
        cells, policy, request=request, decision_id=decision, call_id=call_id,
    )


def _two_group_publication(context):
    """Create real signed bindings for two fields in different encrypted groups."""
    policy = context.session.policy(VALUE_TYPE)
    request = Request(context.application, context.purpose, "select", (), (), (policy,), ())
    call_id = uuid4().hex
    decision = context._decision(request, call_id)
    fields = (("strings", "left"), ("values", "right"))
    data = context.session.create(
        {"right": "same"}, policy, object_type=VALUE_TYPE, group="values",
    )
    context.recorder.store.put(data.snapshot)
    data.set(None, {"left": "same"}, group="strings")
    bindings = {
        field: {"claims": [], "parents": [], "decision_id": decision,
                "declared": describe("same"), "origins": [], "source": None}
        for _group, field in fields
    }
    data.set(None, {"manifest": {"schema": SCHEMA, "bindings": bindings}}, group="evidence")
    publication = context._release_value(data, request)
    handles = []
    for group, field in fields:
        reference = ValueReference(publication.id, field, group)
        result = context._append("Result", request, call_id, {
            "decision_id": decision, "output": reference.to_dict(), "claims": [],
            "field_commitment": publication.envelope[group]["field_hashes"][field],
            "transition": {"governed_inputs": [], "retained_claims": [], "literal_inputs": []},
            "source": None,
        })
        handles.append(context.hydrate(reference, decision, result, kind="str"))
    return publication, handles


@pytest.mark.parametrize("reverse", [False, True])
def test_same_publication_different_groups_preserve_both_native_references(context, reverse):
    source, inputs = _two_group_publication(context)
    retained_wire = source.forward()
    if reverse:
        inputs.reverse()
    with context:
        output = inputs[0] + inputs[1]

    assert output.reveal(context, destination="console") == "samesame"
    publication = context.store_object(output.reference.object_id)
    governance = context.session.governance(publication).governance
    parents = governance.sources
    assert len(parents) == 2
    assert {parent.object_id for parent in parents} == {source.id}
    assert {frozenset(parent.groups) for parent in parents} == {
        frozenset({"strings", "evidence"}), frozenset({"values", "evidence"}),
    }
    assert {parent.operation for parent in parents} == {"concat"}
    expected_inputs = [value.reference.to_dict() for value in inputs]
    metadata, binding, claims = context._metadata(output)
    assert binding["parents"] == expected_inputs
    assert claims == ()
    assert len(metadata.policies) == 1
    assert metadata.policies[0].matches_contract(context.session.policy(VALUE_TYPE))
    result = context.recorder.read(output.result_id).body
    assert result["inputs"] == expected_inputs
    assert result["details"]["transition"]["governed_inputs"] == expected_inputs
    assert context.store_object(source.id).forward() == retained_wire


def test_explicit_sibling_arithmetic_preserves_results_and_each_source_row(context, monkeypatch):
    left, right = _batch(context)
    assert left.reference.object_id == right.reference.object_id
    calls = []
    native_include = tn.DataObject.include

    def include(data, other):
        calls.append((data, other))
        return native_include(data, other)

    monkeypatch.setattr(tn.DataObject, "include", include)
    with context:
        output = ExplicitOperations(context).execute("add", (left, right), lambda a, b: a + b)

    assert output.reveal(context, destination="console") == 15
    metadata, binding, claims = context._metadata(output)
    assert binding["parents"] == [left.reference.to_dict(), right.reference.to_dict()]
    assert [origin["row"] for origin in binding["origins"]] == [0, 1]
    assert claims == ()
    assert len(metadata.policies) == 1
    assert metadata.policies[0].matches_contract(context.session.policy(VALUE_TYPE))
    parents = context.session.governance(context.store_object(output.reference.object_id)).governance.sources
    assert len(parents) == 1
    assert parents[0].object_id == left.reference.object_id
    assert frozenset(parents[0].groups) == frozenset({"values", "evidence"})
    assert calls == []


def test_explicit_arithmetic_preserves_original_claim_subject_and_author(context):
    original, right = _batch(context)
    statement = "The first observation is suitable for this internal example."
    claimed = original.attest(statement, context)
    original_claim = context._metadata(claimed)[2][0]
    with context:
        output = ExplicitOperations(context).execute("add", (claimed, right), lambda a, b: a + b)

    assert output.reveal(context, destination="console") == 15
    _, binding, claims = context._metadata(output)
    assert binding["parents"] == [claimed.reference.to_dict(), right.reference.to_dict()]
    assert binding["claims"] == [original_claim.id]
    assert len(claims) == 1
    assert claims[0].id == original_claim.id
    assert claims[0].issuer == context.session.did
    assert claims[0].subjects == (original.reference,)
    assert claims[0].statement == statement


def test_refused_explicit_arithmetic_opens_no_values_and_does_not_call_compute(context, monkeypatch):
    left, right = _batch(context)
    context.evaluator = lambda request: Assessment(False, "current operation refused")
    opens, computes = [], []

    def open_values(*args):
        opens.append(args)
        raise AssertionError("a refused operation must not open values")

    def compute(*args):
        computes.append(args)
        raise AssertionError("a refused operation must not invoke computation")

    monkeypatch.setattr(context, "_open_values", open_values)
    with context, pytest.raises(OperationDenied):
        ExplicitOperations(context).execute("add", (left, right), compute)
    assert opens == []
    assert computes == []
    refusal = context.recorder.read(context.recorder.head)
    assert refusal.body["kind"] == "Refusal"
    decision = context.recorder.read(refusal.body["details"]["decision_id"])
    assert decision.body["kind"] == "Decision"
    assert decision.body["details"]["permitted"] is False
    assert decision.body["inputs"] == [left.reference.to_dict(), right.reference.to_dict()]
