from decimal import Decimal

import pytest
import tn

from deepvest_tn.publications import PublicationStore
from deepvest_tn.governed.evidence import EvidenceError, EvidenceRecorder
from deepvest_tn.governed.primitives import Context, VALUE_TYPE
from deepvest_tn.governed.runtime import Assessment, Request
from deepvest_tn.governed.string import OperationDenied, TNString
from deepvest_tn.governed.numeric import TNNumeric


POLICY = """## experiment.primitive
### instruction
Retain source attribution through internal calculations.
### use_for
Internal calculation and authorized display.
### do_not_use_for
Unapproved disclosure.
### consequences
Record each decision and its result.
### on_violation_or_error
Stop the operation and retain its failure.

## experiment.evidence
### instruction
Signed records for enrolled readers.
### use_for
Inspect attributed operation records.
### do_not_use_for
Treating a statement as another endpoint's endorsement.
### consequences
Recipients evaluate authority.
### on_violation_or_error
Refuse invalid evidence.
"""


@pytest.fixture
def context(tmp_path):
    with tn.Session(POLICY, groups=["values", "evidence"]) as session:
        store = PublicationStore(tmp_path / "publications")
        policy = session.policy(VALUE_TYPE)
        recorder = EvidenceRecorder(session, store, session.policy("experiment.evidence"))
        yield Context(session, recorder,
            lambda request: Assessment(request.operation != "reveal" or request.destination == "console",
                                       "configured receiver", obligations=("monitor",)),
            application="analytics", purpose="internal", actor="analyst",
            metadata_accept=lambda ctx: ctx.writer == session.did and ctx.object_type == VALUE_TYPE
                and len(ctx.policies) == 1 and ctx.governance.matches_contract(policy))


def source(row):
    return {"table": "market", "column": "value", "row": row, "query_id": "query",
            "assignment_id": "assignment", "snapshot_sha256": "a" * 64}


def publish(context, contents):
    policy = context.session.policy(VALUE_TYPE)
    request = Request(context.application, context.purpose, "select", (), (), (policy,), ())
    decision = context._decision(request, "query")
    return context.originate_batch(
        tuple((content, source(i)) for i, content in enumerate(contents)), policy,
        request=request, decision_id=decision, call_id="query")


@pytest.mark.parametrize("contents", [
    ["Ada", "Grace", "Ada"], [7, 8, 7], [0.125, -0.0, 1.25],
    [Decimal("1.2500"), Decimal("-0.00"), Decimal("2.3750")],
])
def test_cells_share_publication_but_retain_exact_identity_and_type(context, contents):
    values = publish(context, contents)
    assert len({value.reference.object_id for value in values}) == 1
    assert len({value.reference for value in values}) == len(contents)
    assert len({value.result_id for value in values}) == len(contents)
    for index, (value, expected) in enumerate(zip(values, contents, strict=True)):
        assert isinstance(value, TNString if type(expected) is str else TNNumeric)
        _, binding, _ = context._metadata(value)
        assert binding["source"] == source(index)
        assert binding["origins"] == [source(index)]
        actual = value.reveal(context, destination="console")
        assert type(actual) is type(expected) and actual == expected
        if isinstance(expected, Decimal):
            assert actual.as_tuple() == expected.as_tuple()
    wire = context.store_object(values[0].reference.object_id).forward()
    assert b"Retain source attribution" not in wire
    assert b'"snapshot_sha256"' not in wire
    assert b'"Ada"' not in wire


def test_changed_field_cannot_reuse_another_cells_result(context):
    first, second = publish(context, ["Ada", "Grace"])
    forged = TNString(second.reference, first.decision_id, first.result_id,
                      metadata_provider=context.string_metadata)
    with pytest.raises(EvidenceError, match="authorized result"):
        context._metadata(forged)


def test_copy_and_attest_preserve_selected_field_without_opening_values(context, monkeypatch):
    first, second = publish(context, ["Ada", "Grace"])
    original = context._open_values
    def no_open(*args):
        pytest.fail("copy/attest must retain encrypted values without opening them")
    monkeypatch.setattr(context, "_open_values", no_open)
    with context:
        copied = second.copy()
        claimed = first.attest("Suitable for this display", context)
    assert copied.reference.object_id != second.reference.object_id
    assert copied.reference.field == second.reference.field
    assert claimed.reference.field == first.reference.field
    _, copied_binding, _ = context._metadata(copied)
    _, claimed_binding, claims = context._metadata(claimed)
    assert copied_binding["parents"] == [second.reference.to_dict()]
    assert claimed_binding["parents"] == [first.reference.to_dict()]
    assert claims[0].subjects == (first.reference,)
    assert claims[0].issuer == context.session.did
    monkeypatch.setattr(context, "_open_values", original)
    assert copied.reveal(context, destination="console") == "Grace"
    assert claimed.reveal(context, destination="console") == "Ada"
    with context:
        joined = claimed + second
    assert joined.reveal(context, destination="console") == "AdaGrace"
    assert context._metadata(joined)[2][0].subjects == (first.reference,)


def test_changed_request_is_evaluated_before_batch_value_opens(context, monkeypatch):
    value = publish(context, [5, 6])[0]
    context.evaluator = lambda request: Assessment(False, "current request refused")
    opened = []
    monkeypatch.setattr(context, "_open_values", lambda *args: opened.append(args))
    with context, pytest.raises(OperationDenied):
        value + 1
    assert opened == []


def test_shared_contract_is_evaluated_once_with_all_input_references(context):
    values = publish(context, ["Ada", "Grace", "Linus"])
    requests = []
    def evaluate(request):
        requests.append(request)
        return Assessment(True, "accepted")
    context.evaluator = evaluate
    with context:
        result = values[0] + values[1]
    assert len(requests) == 1
    assert len(requests[0].policies) == 1
    assert requests[0].inputs == (values[0].reference, values[1].reference)
    assert result.reveal(context, destination="console") == "AdaGrace"


@pytest.mark.parametrize("change", [
    {"column": "other"}, {"assignment_id": "other"}, {"query_id": "other"},
    {"snapshot_sha256": "b" * 64}, {"row": 0},
])
def test_batch_rejects_mixed_source_scope_and_duplicate_rows(context, change):
    policy = context.session.policy(VALUE_TYPE)
    request = Request(context.application, context.purpose, "select", (), (), (policy,), ())
    decision = context._decision(request, "query")
    before = list(context.recorder.records())
    with pytest.raises(ValueError):
        context.originate_batch([(1, source(0)), (2, {**source(1), **change})], policy,
            request=request, decision_id=decision, call_id="query")
    assert list(context.recorder.records()) == before


def test_batch_refuses_mixed_types_or_unassigned_policy(context):
    policy = context.session.policy(VALUE_TYPE)
    request = Request(context.application, context.purpose, "select", (), (), (policy,), ())
    decision = context._decision(request, "query")
    with pytest.raises(TypeError):
        context.originate_batch([(1, source(0)), (1.0, source(1))], policy,
            request=request, decision_id=decision, call_id="query")
    with pytest.raises(ValueError):
        context.originate_batch([(1, source(0))], context.recorder.policy,
            request=request, decision_id=decision, call_id="query")


@pytest.fixture
def enrolled_batch_contexts(tmp_path):
    from contextlib import ExitStack
    from deepvest_tn.governed.provision import provision

    paths = provision(tmp_path / "sessions", roles=["gateway", "metadata_only"])
    primitive_policy = POLICY.split("## experiment.evidence", 1)[0]
    for path in paths.values():
        policy_file = path.parent / ".tn" / "config" / "agents.md"
        with policy_file.open("a", encoding="utf-8") as stream:
            stream.write("\n" + primitive_policy)
    assert not (paths["metadata_only"].parent / "keys" / "values.btn.mykit").exists()
    assert (paths["gateway"].parent / "keys" / "values.btn.mykit").is_file()
    with ExitStack() as stack:
        sessions = {role: stack.enter_context(tn.Session.from_config(str(path)))
                    for role, path in paths.items()}
        writer = sessions["gateway"]
        policy = writer.policy(VALUE_TYPE)
        audit = writer.policy("experiment.evidence")
        store = PublicationStore(tmp_path / "publications")
        contexts = {}
        for role, session in sessions.items():
            contexts[role] = Context(session, EvidenceRecorder(session, store, audit),
                lambda request: Assessment(True, "current receiver permits the operation"),
                application="analytics", purpose="internal", actor=role,
                metadata_accept=lambda ctx: ctx.writer == writer.did
                    and ctx.object_type == VALUE_TYPE and len(ctx.policies) == 1
                    and ctx.governance.matches_contract(policy))
        yield contexts


@pytest.mark.parametrize("contents", [
    ["Ada", "Grace"], [7, 8], [0.125, 1.25], [Decimal("1.2500"), Decimal("2.3750")],
])
def test_metadata_reader_needs_real_values_key_for_compute_and_reveal(enrolled_batch_contexts, contents):
    from tn.governed import NotEntitled
    from deepvest_tn.governed.evidence import read_record

    gateway, receiver = (enrolled_batch_contexts[role] for role in ("gateway", "metadata_only"))
    sources = publish(gateway, contents)
    values = receiver.hydrate_many([
        (value.reference, value.decision_id, value.result_id, type(content).__name__)
        for value, content in zip(sources, contents, strict=True)])
    first = values[0]
    # Reading signed declared facts does not need the separate values reader kit.
    _, binding, _ = receiver._metadata(first)
    assert binding["source"] == source(0)
    assert first == contents[0]
    if isinstance(first, TNString):
        assert len(first) == 3
    else:
        assert first.dtype is type(contents[0])
    with receiver, pytest.raises(NotEntitled):
        first.upper() if isinstance(first, TNString) else first + 1
    with pytest.raises(NotEntitled):
        first.reveal(receiver, destination="console")
    records = [read_record(wire, session=receiver.session, policy=receiver.recorder.policy)
               for wire in receiver.recorder.records()]
    assert [record.body["kind"] for record in records] == ["Decision", "Failure", "Decision", "Failure"]
    assert all(record.body["details"]["permitted"] is True
               for record in records if record.body["kind"] == "Decision")
    assert all(record.body["details"]["error_type"] == "NotEntitled"
               for record in records if record.body["kind"] == "Failure")
    assert all(record.writer == receiver.session.did for record in records)


@pytest.mark.parametrize("changed", ["application", "purpose", "writer", "policy"])
def test_changed_admission_is_checked_before_batch_content_access(context, monkeypatch, changed):
    first, second = publish(context, ["Ada", "Grace"])
    context._metadata_many((first, second))
    reads = []
    original = context._read
    def track_read(value, groups, operation):
        reads.append((tuple(groups), operation))
        return original(value, groups, operation)
    monkeypatch.setattr(context, "_read", track_read)
    if changed in {"application", "purpose"}:
        setattr(context, changed, "external")
    elif changed == "writer":
        context.metadata_accept = lambda ctx: ctx.writer == "did:unexpected-writer"
    else:
        other_policy = context.recorder.policy
        context.metadata_accept = lambda ctx: ctx.governance.matches_contract(other_policy)
    with context, pytest.raises(OperationDenied, match="input admission failed"):
        first + second
    assert reads == [(("evidence",), "inspect-claims")]
    assert context.recorder.read(context.recorder.head).body["kind"] == "Refusal"


def test_metadata_batch_reuses_reads_only_within_current_call(context, monkeypatch):
    from collections import Counter
    from tn.governed import UseDenied

    first, second, third = publish(context, ["Ada", "Grace", "Linus"])
    values = (first, second, third, first)
    reads, records = [], []
    original_read, original_record = context._read, context.recorder.read
    def track_read(value, groups, operation):
        reads.append((value.reference.object_id, tuple(groups), operation))
        return original_read(value, groups, operation)
    def track_record(identity):
        records.append(identity)
        return original_record(identity)
    monkeypatch.setattr(context, "_read", track_read)
    monkeypatch.setattr(context.recorder, "read", track_record)
    expected = Counter({first.decision_id: 1, first.result_id: 1, second.result_id: 1, third.result_id: 1})
    for repetition in range(1, 3):
        items = context._metadata_many(values)
        assert len(items) == 4
        assert len(reads) == repetition
        assert all(groups == ("evidence",) and operation == "inspect-claims"
                   for _, groups, operation in reads)
        assert Counter(records) == Counter({identity: count * repetition for identity, count in expected.items()})
    context.metadata_accept = lambda ctx: False
    with pytest.raises(UseDenied):
        context._metadata_many(values)
    assert len(reads) == 3
    assert len(records) == 8


def test_open_batch_reuses_one_receive_then_readmits_on_next_call(context, monkeypatch):
    from tn.governed import UseDenied

    values = publish(context, [7, 8, 9])
    request = Request(context.application, context.purpose, "add",
        tuple(value.reference for value in values), (), (context.session.policy(VALUE_TYPE),), ())
    reads = []
    original = context._read
    def track_read(value, groups, operation):
        reads.append((value.reference.object_id, tuple(groups), operation))
        return original(value, groups, operation)
    monkeypatch.setattr(context, "_read", track_read)
    for repetition in range(1, 3):
        opened = context._open_values(request, values)
        assert [content for _, content in opened] == [7, 8, 9]
        assert len(reads) == repetition
        assert all(groups == ("values", "evidence") and operation == "add"
                   for _, groups, operation in reads)
    context.metadata_accept = lambda ctx: False
    with pytest.raises(UseDenied):
        context._open_values(request, values)
    assert len(reads) == 3


def test_shared_publication_is_copied_and_merged_once(context, monkeypatch):
    values = publish(context, [7, 8, 9])
    request = Request(context.application, context.purpose, "add",
        tuple(value.reference for value in values), (), (context.session.policy(VALUE_TYPE),), ())
    copies, includes = [], []
    native_copy, native_include = tn.DataObject.copy, tn.DataObject.include

    def track_copy(data):
        copies.append(data)
        return native_copy(data)

    def track_include(data, other):
        includes.append((data, other))
        return native_include(data, other)

    monkeypatch.setattr(tn.DataObject, "copy", track_copy)
    monkeypatch.setattr(tn.DataObject, "include", track_include)
    opened = context._open_values(request, values)
    assert len(copies) == 1
    assert len({id(data) for data, _content in opened}) == 1

    merged = context._merge_opened(values, opened)
    assert merged is opened[0][0]
    assert includes == []


@pytest.mark.parametrize("contents", [["Ada", "DO-NOT-INCLUDE-SIBLING"], [7, 818181]])
def test_computed_singleton_does_not_contain_other_batch_values(context, contents):
    first, sibling = publish(context, contents)
    with context:
        derived = first.upper() if isinstance(first, TNString) else first + 1
    expected = "ADA" if isinstance(first, TNString) else 8
    assert derived.reveal(context, destination="console") == expected
    publication = context.store_object(derived.reference.object_id)
    assert set(publication.envelope["values"]["field_hashes"]) == {"value"}
    opened = context._read(derived, ["values", "evidence"], "reveal")
    assert opened.get(None, group="values") == {"value": "ADA" if isinstance(first, TNString) else "8"}
    manifest = opened.get("manifest", group="evidence")
    assert set(manifest["bindings"]) == {"value"}
    assert manifest["bindings"]["value"]["parents"] == [first.reference.to_dict()]
    assert manifest["bindings"]["value"]["origins"] == [source(0)]
    assert sibling.reference.to_dict() not in manifest["bindings"]["value"]["parents"]
