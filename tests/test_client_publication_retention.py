from types import SimpleNamespace

import pytest

from tn_governed import llama


def test_submit_retains_tools_then_report_only_after_recipient_acceptance(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    tool_one = SimpleNamespace(id="tool-1")
    tool_two = SimpleNamespace(id="tool-2")
    output = SimpleNamespace(id="report")
    release = SimpleNamespace(
        tool_objects=(tool_one, tool_two),
        output_object=output,
    )
    events = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def submit_model(self, *args, **kwargs):
            events.append("submitted")
            return release

    class Reader:
        def open(self, actual):
            assert actual is release
            events.append("accepted")
            return "opened-report"

    class Store:
        def put(self, object):
            events.append("retained:" + object.id)

    monkeypatch.setattr(llama, "GovernedLlamaClient", Client)

    actual_release, report = llama.submit_governed_report(
        object(),
        prompt="report",
        use=object(),
        destination="recipient",
        base_url="http://127.0.0.1:1",
        api_key="test",
        transcript_path=tmp_path / "http.ndjson",
        output_reader=Reader(),
        publication_store=Store(),
    )

    assert actual_release is release
    assert report == "opened-report"
    assert events == [
        "submitted",
        "accepted",
        "retained:tool-1",
        "retained:tool-2",
        "retained:report",
    ]


def test_submit_retains_nothing_when_recipient_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    release = SimpleNamespace(
        tool_objects=(SimpleNamespace(id="tool-1"),),
        output_object=SimpleNamespace(id="report"),
    )
    retained = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def submit_model(self, *args, **kwargs):
            return release

    class Reader:
        def open(self, actual):
            raise PermissionError("recipient refused")

    monkeypatch.setattr(llama, "GovernedLlamaClient", Client)

    with pytest.raises(PermissionError, match="recipient refused"):
        llama.submit_governed_report(
            object(),
            prompt="report",
            use=object(),
            destination="recipient",
            base_url="http://127.0.0.1:1",
            api_key="test",
            transcript_path=tmp_path / "http.ndjson",
            output_reader=Reader(),
            publication_store=SimpleNamespace(put=retained.append),
        )

    assert retained == []
