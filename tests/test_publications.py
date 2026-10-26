import hashlib
import io
import json
from types import SimpleNamespace

import pytest
import tn

from tn_governed.publications import PublicationStore, open_publication_store


@pytest.fixture(params=["files", "sqlite"])
def governed_store(tmp_path, request):
    with open_publication_store(tmp_path, backend=request.param) as store:
        yield store

POLICY = """---
version: 1
schema: tn-agents-policy@v1
---
## portfolio.feed.data.v1
### instruction
Prepare approved portfolio analysis.
### use_for
Portfolio analysis.
### do_not_use_for
Individual disclosure.
### consequences
Contract review.
### on_violation_or_error
Refuse release.
"""


def publication():
    with tn.Session(POLICY, groups=["finance"]) as session:
        data = session.create_obj(
            {"rows": [{"amount": 42}]}, session.policy("portfolio.feed.data.v1"),
            object_type="portfolio.feed.data.v1", group="finance",
        )
        return data.release(
            use=tn.UseContext("source", "portfolio_analysis", "publish"),
            to="analytics", decide=lambda context: True,
        )


def test_retained_publication_is_exact_and_survives_store_restart(tmp_path):
    source = publication()
    store = PublicationStore(tmp_path)
    assert store.put(source) == source.id
    assert store.put(source) == source.id
    assert store.get(source.id) == source.forward()
    assert PublicationStore(tmp_path).get(source.id) == source.forward()


def test_identical_retained_bytes_are_not_written_again(tmp_path, monkeypatch):
    from tn_governed import publications

    source = publication()
    store = PublicationStore(tmp_path)
    store.put(source)

    def no_rewrite(*_args, **_kwargs):
        raise AssertionError("identical immutable bytes must use the retained representation")

    monkeypatch.setattr(publications.tempfile, "mkstemp", no_rewrite)
    assert store.put(source) == source.id
    assert store.get(source.id) == source.forward()


def test_same_native_identity_cannot_replace_retained_wire_representation(tmp_path):
    source = publication()
    reformatted = tn.GovernedObject.parse(json.dumps(source.envelope, indent=2))
    assert reformatted.id == source.id
    assert reformatted.wire != source.wire
    store = PublicationStore(tmp_path)
    store.put(source)
    with pytest.raises(ValueError, match="immutable"):
        store.put(reformatted)
    assert store.get(source.id) == source.forward()


def test_missing_publication_and_unsafe_identity_fail_closed(tmp_path):
    store = PublicationStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("sha256:" + "a" * 64)
    for identity in ["../outside", "sha256:" + "A" * 64, "sha256:short"]:
        with pytest.raises(ValueError):
            store.get(identity)


def test_store_detects_content_and_route_index_substitution(tmp_path):
    first, second = publication(), publication()
    store = PublicationStore(tmp_path)
    store.put(first)
    store.put(second)
    first_path = tmp_path / "objects" / f"{first.id.removeprefix('sha256:')}.tn"
    first_path.write_bytes(second.forward())
    with pytest.raises(ValueError, match="identity|digest"):
        store.get(first.id)
    first_path.write_bytes(first.forward())
    index = tmp_path / "index" / f"{first.id.removeprefix('sha256:')}.json"
    value = json.loads(index.read_text())
    value["object_id"] = second.id
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="index"):
        store.get(first.id)


def test_get_verifies_the_bytes_already_read_without_reopening_the_file(tmp_path, monkeypatch):
    from tn_governed import publications

    source = publication()
    store = PublicationStore(tmp_path)
    store.put(source)
    native_read = tn.GovernedObject.read
    inputs = []

    def observed_read(value):
        inputs.append(value)
        return native_read(value)

    monkeypatch.setattr(publications, "tn", SimpleNamespace(
        GovernedObject=SimpleNamespace(read=observed_read)))
    assert store.get(source.id) == source.forward()
    assert len(inputs) == 1
    assert isinstance(inputs[0], io.BytesIO)


@pytest.mark.parametrize("method", ["get", "read"])
def test_each_read_detects_changed_wire_even_after_a_successful_read(tmp_path, method):
    source = publication()
    store = PublicationStore(tmp_path)
    store.put(source)
    load = getattr(store, method)
    load(source.id)
    digest = source.id.removeprefix("sha256:")
    path = tmp_path / "objects" / f"{digest}.tn"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest"):
        load(source.id)


@pytest.mark.parametrize("method", ["get", "read"])
def test_matching_index_hash_does_not_accept_a_changed_signed_publication(tmp_path, method):
    source = publication()
    store = PublicationStore(tmp_path)
    store.put(source)
    digest = source.id.removeprefix("sha256:")
    changed = dict(source.envelope)
    changed["timestamp"] = "2026-01-01T00:00:00Z"
    wire = json.dumps(changed).encode()
    (tmp_path / "objects" / f"{digest}.tn").write_bytes(wire)
    index_path = tmp_path / "index" / f"{digest}.json"
    index = json.loads(index_path.read_bytes())
    index["wire_sha256"] = hashlib.sha256(wire).hexdigest()
    index_path.write_text(json.dumps(index))
    load = getattr(store, method)
    with pytest.raises(tn.governed.VerificationError):
        load(source.id)


def test_native_read_preserves_wire_and_requires_a_new_use_decision(governed_store):
    with tn.Session(POLICY, groups=["finance"]) as session:
        data = session.create(
            {"amount": 42}, session.policy("portfolio.feed.data.v1"),
            object_type="portfolio.feed.data.v1", group="finance")
        source = session.release(data, use=tn.UseContext("source", "analysis", "publish"),
                                 to="analytics", decide=lambda context: True)
        store = governed_store
        store.put(source)
        loaded = store.read(source.id)
        assert isinstance(loaded, tn.GovernedObject)
        assert loaded.forward() == source.forward()
        seen = []

        def decide(context):
            seen.append(context.use_context.operation)
            return context.use_context.operation == "calculate"

        allowed = session.receive(loaded, use=tn.UseContext("analytics", "analysis", "calculate"),
                                  groups=["finance"], decide=decide)
        assert allowed.get("amount", group="finance") == 42
        with pytest.raises(tn.governed.UseDenied):
            session.receive(loaded, use=tn.UseContext("analytics", "analysis", "expose"),
                            groups=["finance"], decide=decide)
        assert seen == ["calculate", "expose"]


def test_record_read_rechecks_authority_after_loading_the_native_object(governed_store):
    from tn_governed.values.evidence import EvidenceError, EvidenceRecorder, read_record

    other_policy = POLICY.replace("portfolio.feed.data.v1", "another.policy")
    with tn.Session(POLICY + "\n" + other_policy, groups=["evidence"]) as session:
        policy = session.policy("portfolio.feed.data.v1")
        recorder = EvidenceRecorder(session, governed_store, policy)
        identity = recorder.append("Observation", call_id="test", application="test",
                                   purpose="verification", operation="read")
        assert recorder.read(identity).body["kind"] == "Observation"
        native = recorder.store.read(identity)
        with pytest.raises(EvidenceError, match="writer"):
            read_record(native, session=session, policy=policy, expected_writer="unrecognized")
        recorder.policy = session.policy("another.policy")
        with pytest.raises(EvidenceError, match="verification"):
            recorder.read(identity)


def test_string_operations_preserve_claims_and_recheck_permission(governed_store, monkeypatch):
    from uuid import uuid4
    from tn_governed.values import (
        Assessment, Context, EvidenceRecorder, OperationDenied, Request, TNString,
    )

    with tn.Session(POLICY, groups=["values", "evidence"]) as session:
        policy = session.policy("portfolio.feed.data.v1")
        recorder = EvidenceRecorder(session, governed_store, policy)
        context = Context(session, recorder, lambda r: Assessment(True, "internal use"),
                          application="test", purpose="analysis", actor="worker",
                          metadata_accept=lambda c: c.writer == session.did)
        request = Request("test", "analysis", "select", (), (), (policy,), ())
        call = uuid4().hex
        decision = context._decision(request, call)
        name = context.originate("Ada", policy, source={"column": "name"},
                                 request=request, decision_id=decision, call_id=call)
        with context:
            annotated = name.attest("Name for internal identification", context)
            result = (annotated.upper() + " report").copy()
            assert isinstance(result, TNString)
            assert result.reveal(context, destination="internal") == "ADA report"
            claims = context._metadata(result)[2]
            assert len(claims) == 1 and claims[0].subjects == (name.reference,)
            context.evaluator = lambda r: Assessment(False, "use changed")
            monkeypatch.setattr(context, "_open_values", lambda *a: pytest.fail("opened after refusal"))
            with pytest.raises(OperationDenied):
                result.upper()
