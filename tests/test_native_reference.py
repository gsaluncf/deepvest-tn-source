"""Verify the connected application's preparation and its single model handoff."""

import json
from types import SimpleNamespace

import pytest
from tn_polars.registry import LocalRegistry

from tn_governed.llama import submit_governed_report
from tn_governed.native_reference import (
    DESTINATION,
    MODEL_USE,
    finish_governed_reference,
    load_governed_reference,
    prepare_governed_reference,
    write_governed_runtime_config,
)


@pytest.fixture
def prepared(tmp_path):
    with prepare_governed_reference(
        tmp_path / "run",
        unity_url="local-registry",
        registry=LocalRegistry(tmp_path / "registry"),
    ) as value:
        yield value


def test_preparation_pins_real_calculation_and_all_native_service_identities(prepared):
    facts = json.loads(prepared.preparation_path.read_bytes())
    assert len(set(facts["sessions"].values())) == 6
    assert facts["calculation"]["value"] == -0.25
    assert len(facts["editions"]) == 2
    assert len(facts["source_http_requests"]) == 2
    source = prepared.calculation.calculation_object
    assert (prepared.directory / "model-input.tn").read_bytes() == source.forward()
    metadata = json.loads(prepared.metadata_path.read_bytes())
    assert len(metadata["policy_revisions"]) == len(metadata["dataset_editions"]) == 2
    assert source.wire in metadata["publications"]
    view = prepared.sessions["model-reader"].governance(source)
    assert len(view.governance.dataset_bindings) == len(view.governance.policies) == 2


def test_preparation_publishes_explicit_output_requirements(prepared):
    policies = list((prepared.directory / "editions").rglob("compiled_policy.json"))
    assert len(policies) == 2
    for path in policies:
        compiled = json.loads(path.read_bytes())
        assert compiled["release"] == {
            "public": {"allow_aggregates": False, "min_group_size": 2},
        }


def test_runtime_configuration_uses_exact_edition_metadata_and_model_use(prepared):
    model = prepared.directory / "test-model.gguf"
    binary = prepared.directory / "test-server"
    model.write_bytes(b"test model identity")
    binary.write_bytes(b"test server identity")
    path = write_governed_runtime_config(prepared, model=model, binary=binary)
    config = json.loads(path.read_bytes())
    assert config["governed_objects"]["use_context"]["application"] == MODEL_USE.application
    assert len(config["governed_objects"]["contracts"]) == 2
    assert config["ceremony_id"] != config["output_ceremony_id"]
    assert config["governed_objects"]["metadata_store_path"] == str(prepared.metadata_path)


def test_reopen_resolves_the_same_editions_without_republishing(prepared):
    registry = LocalRegistry(prepared.directory.parent / "registry")
    before = {path.name for path in (prepared.store.root / "objects").iterdir()}
    with load_governed_reference(prepared.directory, registry=registry) as reopened:
        assert (
            reopened.calculation.calculation_object.wire == prepared.calculation.calculation_object.wire
        )
        assert [item.selection.edition_record_id for item in reopened.editions] == [
            item.selection.edition_record_id for item in prepared.editions
        ]
        assert {path.name for path in (reopened.store.root / "objects").iterdir()} == before


def test_application_passes_the_existing_release_and_waits_for_recipient(prepared, monkeypatch):
    calls = []
    source = prepared.calculation.calculation_object
    release = SimpleNamespace(output_object=source)

    class Client:
        def __init__(self, *_):
            pass

        def submit_model(self, object, prompt, use, *, destination):
            calls.append((object, prompt, use, destination))
            return release

    def open_report(value):
        assert value is release
        calls.append("recipient")
        return "accepted-report"

    def retain(value):
        assert calls[-1] == "recipient"
        calls.append(value)

    monkeypatch.setattr("tn_governed.llama.GovernedLlamaClient", Client)
    actual, report = submit_governed_report(
        source,
        prompt="Explain the result.",
        use=MODEL_USE,
        destination=DESTINATION,
        base_url="http://unused",
        api_key="test",
        transcript_path=prepared.directory / "http.json",
        output_reader=SimpleNamespace(open=open_report),
        publication_store=SimpleNamespace(put=retain),
    )
    assert actual is release and report == "accepted-report"
    assert calls[0][0] is source
    assert calls[0][2:] == (MODEL_USE, DESTINATION)
    assert calls[1:] == ["recipient", source]


def test_recipient_refusal_does_not_retain_model_release(prepared, monkeypatch):
    class Client:
        def __init__(self, *_):
            pass

        def submit_model(self, *_args, **_kwargs):
            return SimpleNamespace(output_object=prepared.calculation.calculation_object)

    def refuse(_):
        raise PermissionError("recipient use denied")

    monkeypatch.setattr("tn_governed.llama.GovernedLlamaClient", Client)
    with pytest.raises(PermissionError, match="recipient use denied"):
        submit_governed_report(
            prepared.calculation.calculation_object,
            prompt="Explain.",
            use=MODEL_USE,
            destination=DESTINATION,
            base_url="http://unused",
            api_key="test",
            transcript_path=prepared.directory / "http.json",
            output_reader=SimpleNamespace(open=refuse),
            publication_store=SimpleNamespace(
                put=lambda _: pytest.fail("unaccepted report retained")
            ),
        )


@pytest.mark.parametrize("with_tool", [False, True])
def test_complete_recipient_path_accepts_a_separate_native_publisher(
    prepared, monkeypatch, with_tool,
):
    from tn_llama_client.protocol import NativeModelRelease

    source = prepared.calculation.calculation_object
    data = prepared.sessions["model-reader"].receive(
        source,
        use=MODEL_USE,
        groups=["finance"],
        decide=lambda ctx: prepared.catalog.accepts(ctx, prepared.policy_dag),
    )
    tools = ()
    if with_tool:
        tool_data = prepared.sessions["model-reader"].receive(
            source, use=MODEL_USE, groups=["finance"],
            decide=lambda ctx: prepared.catalog.accepts(ctx, prepared.policy_dag),
        )
        tool_data.groups["default"] = {
            "run_id": "test-native-recipient",
            "request_digest": "sha256:" + "a" * 64,
            "tool_name": "log_compliance_creation",
            "model_tool_call_id": "call-compliance",
            "source_object_id": source.id,
            "source_writer": source.writer,
            "source_type": source.object_type,
            "selected_sections": ["finance"],
            "action": MODEL_USE.operation,
        }
        tool_data.retain_groups(["default"])
        tool = prepared.sessions["model-writer"].release(
            tool_data, use=MODEL_USE, to=DESTINATION, decide=lambda _: True,
            object_type="compliance.creation.logged",
        )
        data.include(prepared.sessions["model-writer"].receive(
            tool, use=MODEL_USE, groups=["default"], decide=lambda _: True,
        ))
        tools = (tool,)
    data.groups["default"] = {"report_html": "<p>Maximum drawdown: -0.25.</p>"}
    data.retain_groups(["default"])
    output = prepared.sessions["model-writer"].release(
        data,
        use=MODEL_USE,
        to=DESTINATION,
        decide=lambda _: True,
        object_type="report.generated",
    )
    release = NativeModelRelease(
        "test-native-recipient", source.id, output.id, output, [], tool_objects=tools,
    )

    class Client:
        def __init__(self, *_):
            pass

        def submit_model(self, object, _prompt, use, *, destination):
            assert object.wire == source.wire
            assert use == MODEL_USE and destination == DESTINATION
            return release

    monkeypatch.setattr("tn_governed.llama.GovernedLlamaClient", Client)
    report_path, summary_path = finish_governed_reference(
        prepared,
        base_url="http://unused",
        api_key="test",
    )
    assert report_path.read_text() == "<p>Maximum drawdown: -0.25.</p>"
    facts = json.loads(summary_path.read_bytes())
    assert facts["recipient_contract_count"] == facts["recipient_edition_count"] == 2
    assert facts["model_source_object_id"] == facts["calculation_object_id"]
    assert prepared.store.get(output.id) == output.forward()
    for tool in tools:
        assert prepared.store.get(tool.id) == tool.forward()
