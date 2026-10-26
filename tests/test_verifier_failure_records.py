"""A failed call can retain partial results but has only one final Failure."""

from contextlib import ExitStack
import json

import pytest
import tn

from deepvest_tn.publications import PublicationStore
from deepvest_tn.governed.evidence import EvidenceError, EvidenceRecorder
from deepvest_tn.governed.verify import verify
from test_primitive_graph_identity import batch_run as batch_run


def append_failures(root, run, count):
    checkpoint = run["tapes"]["gateway"]
    with ExitStack() as stack:
        session = stack.enter_context(
            tn.Session.from_config(str(root / "sessions/gateway/tn.yaml"))
        )
        catalog = stack.enter_context(
            tn.Session.from_config(str(root / "sessions/catalog/tn.yaml"))
        )
        store = stack.enter_context(PublicationStore(root / "publications"))
        recorder = EvidenceRecorder(session, store, catalog.policy("experiment.evidence"))
        recorder.context_id = checkpoint["context_id"]
        recorder._ids = list(store.record_ids(checkpoint["context_id"]))
        decision = run["outputs"]["0"]["decision_id"]
        for _ in range(count):
            recorder.append(
                "Failure",
                call_id="query",
                application="test",
                purpose="internal",
                operation="select",
                details={"decision_id": decision, "error_type": "PublicationError"},
            )
        retained = recorder.export(root / "records-with-failure")
    run["tapes"]["gateway"] = {**retained, "path": "records-with-failure"}
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")


def test_duplicate_failure_records_are_rejected(batch_run):
    root, run = batch_run
    append_failures(root, run, count=2)

    with pytest.raises(EvidenceError, match="follows a failure"):
        verify(root)


def test_partial_results_before_one_final_failure_remain_verifiable(batch_run):
    root, run = batch_run
    append_failures(root, run, count=1)

    summary = verify(root)

    assert summary["record_kinds"]["Result"] == 2
    assert summary["record_kinds"]["Failure"] == 1
    assert summary["incarnations"] == 2
    graph = json.loads((root / "dag.json").read_text(encoding="utf-8"))
    assert {node["result_id"] for node in graph["nodes"]} == {
        output["result_id"] for output in run["outputs"].values()
    }
