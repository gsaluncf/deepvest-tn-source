"""Section-based model transport must not include uncited sibling values."""

from types import SimpleNamespace

import pytest
import tn

from deepvest_tn.publications import PublicationStore
from experiments.governed_strings.evidence import EvidenceRecorder, read_record
from experiments.governed_strings.model import NativeModelEndpoint, prepare_native_model
from experiments.governed_strings.primitives import Context, VALUE_TYPE
from experiments.governed_strings.runtime import Assessment, Request
from experiments.governed_strings.string import OperationDenied, TNString, ValueReference


POLICY = """## experiment.primitive
### instruction
Preserve the source conditions when producing an internal report.
### use_for
Internal reports with their required notice.
### do_not_use_for
Unapproved disclosure.
### consequences
Record the decision and outcome.
### on_violation_or_error
Stop and record the failure.

## experiment.evidence
### instruction
Retain attributable operation records.
### use_for
Inspect original decisions and outcomes.
### do_not_use_for
Claiming another writer endorsed a result.
### consequences
Recipients evaluate each record.
### on_violation_or_error
Refuse invalid evidence.
"""

MODEL = {
    "use": {"application": "tn-llama", "purpose": "internal", "operation": "generate_report"},
    "recipient_use": {"application": "reports", "purpose": "internal", "operation": "read"},
    "destinations": ["internal-model"], "report_destination": "reports",
    "required_tool": "log_compliance_creation", "required_disclosure": "Internal report; review required.",
    "review": None,
}


@pytest.fixture
def source_values(tmp_path):
    with tn.Session(POLICY, groups=["values", "evidence"]) as session:
        policy = session.policy(VALUE_TYPE)
        store = PublicationStore(tmp_path / "publications")
        recorder = EvidenceRecorder(session, store, session.policy("experiment.evidence"))
        context = Context(session, recorder, lambda request: Assessment(True, "authorized internal operation"),
            application="analytics", purpose="internal", actor="analyst",
            metadata_accept=lambda ctx: ctx.writer == session.did
                and ctx.object_type == VALUE_TYPE and ctx.governance.matches_contract(policy))
        context.resolver = SimpleNamespace(rules={policy.revision_id: {"model": MODEL}})
        request = Request("analytics", "internal", "select", (), (), (policy,), ())
        decision = context._decision(request, "query")
        sources = [{"table": "people", "column": "name", "row": row, "query_id": "query",
                    "assignment_id": "assignment", "snapshot_sha256": "a" * 64}
                   for row in range(2)]
        values = context.originate_batch(list(zip(["Ada", "Grace"], sources, strict=True)), policy,
            request=request, decision_id=decision, call_id="query")
        yield context, values


def records_since(context, count):
    return [read_record(wire, session=context.session, policy=context.recorder.policy)
            for wire in context.recorder.records()[count:]]


def selected_input(context, first, form):
    with context:
        if form == "copy":
            return first.copy()
        if form == "attest":
            return first.attest("Suitable for this internal report", context)
        return first


@pytest.mark.parametrize("form", ["source", "copy", "attest", "missing-field"])
def test_model_preparation_rejects_siblings_or_missing_field_before_writing_files(source_values, tmp_path, form):
    context, (first, _) = source_values
    value = selected_input(context, first, form)
    if form == "missing-field":
        value = TNString(ValueReference(first.reference.object_id, "not-present", "values"),
                         first.decision_id, first.result_id, metadata_provider=context.string_metadata)
    output = tmp_path / "model-config"
    with pytest.raises(OperationDenied, match="standalone governed value"):
        prepare_native_model(value, context=context, input_session=None, output_session=None,
            recipient_session=None, input_config=tmp_path / "unused-input.yaml",
            output_config=tmp_path / "unused-output.yaml", directory=output,
            model=tmp_path / "unused-model.gguf", binary=tmp_path / "unused-server")
    assert not output.exists()


class TransportReached(Exception):
    """Stop the test at the real transport boundary without making a model call."""


def endpoint(context, value, tmp_path):
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    return NativeModelEndpoint(context=context, value=value, base_url="http://unused", api_key="test",
        output_reader=None, directory=tmp_path, requirements=MODEL, runtime_config=config)


@pytest.mark.parametrize("form", ["source", "copy", "attest"])
def test_protected_handoff_records_failure_without_submitting_siblings(source_values, tmp_path, monkeypatch, form):
    context, (first, _) = source_values
    value = selected_input(context, first, form)
    target = endpoint(context, value, tmp_path)
    submissions = []
    def reached(source, **kwargs):
        submissions.append(source)
        raise TransportReached()
    monkeypatch.setattr("experiments.governed_strings.model.submit_governed_report", reached)
    count = len(context.recorder.records())
    with context, pytest.raises(OperationDenied, match="standalone governed value"):
        target.generate(value, prompt="Summarize the selected value.")
    assert submissions == []
    records = records_since(context, count)
    assert [record.body["kind"] for record in records] == ["Decision", "Failure"]
    decision, failure = records
    assert decision.body["details"]["permitted"] is True
    assert failure.body["details"]["decision_id"] == decision.id
    assert failure.body["details"]["error_type"] == "OperationDenied"
    assert all(record.body["inputs"] == [value.reference.to_dict()] for record in records)
    assert all(record.writer == context.session.did for record in records)


@pytest.mark.parametrize("field", ["value", "value_0"])
def test_sole_cited_field_continues_to_native_transport(source_values, tmp_path, monkeypatch, field):
    context, (first, _) = source_values
    if field == "value":
        with context:
            value = first.upper()
    else:
        request = Request("analytics", "internal", "select", (), (), (context.session.policy(VALUE_TYPE),), ())
        decision = context._decision(request, "one-row-query")
        source = {"table": "people", "column": "name", "row": 0, "query_id": "one-row-query",
                  "assignment_id": "assignment", "snapshot_sha256": "a" * 64}
        value, = context.originate_batch([("Ada", source)], context.session.policy(VALUE_TYPE),
            request=request, decision_id=decision, call_id="one-row-query")
    target = endpoint(context, value, tmp_path)
    submissions = []
    def reached(source, **kwargs):
        submissions.append(source)
        raise TransportReached()
    monkeypatch.setattr("experiments.governed_strings.model.submit_governed_report", reached)
    count = len(context.recorder.records())
    with context, pytest.raises(TransportReached):
        target.generate(value, prompt="Summarize the selected value.")
    assert len(submissions) == 1
    assert submissions[0].forward() == context.store_object(value.reference.object_id).forward()
    assert set(submissions[0].envelope["values"]["field_hashes"]) == {field}
    records = records_since(context, count)
    assert [record.body["kind"] for record in records] == ["Decision", "ModelHandoff", "Failure"]
    assert records[1].body["details"]["subject"] == value.reference.to_dict()
