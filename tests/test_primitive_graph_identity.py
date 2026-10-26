"""Offline verification distinguishes fields sharing one signed publication."""

from contextlib import ExitStack
from collections import Counter
from functools import lru_cache
import json
import sqlite3

import pytest
import tn
from tn.governed import GovernedError

from tn_governed.publications import PublicationStore
from tn_governed.values.catalog import AssignedRules, UnityGovernance, markdown
from tn_governed.values.evidence import EvidenceError, EvidenceRecorder, Record
from tn_governed.values.metadata import describe
from tn_governed.values.primitives import Context, SCHEMA, VALUE_TYPE
from tn_governed.values.provision import provision
from tn_governed.values.runtime import Request
from tn_governed.values.sql import source_identity
from tn_governed.values.string import TNString, ValueReference
from tn_governed.values.verify import verify


@pytest.fixture
def batch_run(tmp_path, monkeypatch, request):
    """Real native keys, policies, signatures and records; catalog transport is local."""
    options = getattr(request, "param", {})
    row_count = options.get("rows", 2)
    database = tmp_path / "source.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE people (name TEXT NOT NULL)")
        connection.executemany("INSERT INTO people VALUES (?)", [("Ada",)] * row_count)
    identity = source_identity(database)
    paths = provision(tmp_path / "sessions", roles=["catalog", "gateway", "observer"])
    locations = {}

    def transport(method, path, payload=None):
        if method == "POST" and path == "/volumes":
            key = f"{payload['catalog_name']}.{payload['schema_name']}.{payload['name']}"
            locations[key] = payload["storage_location"]
        if method == "GET":
            return {"storage_location": locations[path.split("/")[-1]]}
        return {}

    with ExitStack() as stack:
        sessions = {role: stack.enter_context(tn.Session.from_config(str(paths[role])))
                    for role in ("catalog", "gateway")}
        authority = sessions["catalog"].did
        admin = sessions["catalog"].policy("experiment.administration")
        audit = sessions["catalog"].policy("experiment.evidence")
        store = PublicationStore(tmp_path / "publications")
        resolvers = {}
        for role, session in sessions.items():
            resolver = UnityGovernance("http://unused", "test", session=session,
                authority=authority, administration=admin, store=store)
            monkeypatch.setattr(resolver.registry, "_request", transport)
            resolvers[role] = resolver
        assignment_id = resolvers["catalog"].publish_assignment(tmp_path / "assignment",
            table="people", source_identity=identity, application="test", actors=["analyst"],
            purposes=["internal"], columns={"name": {"kind": "str", "markdown": markdown("name")}})
        resolution = resolvers["gateway"].resolve("people", ["name"], actor="analyst",
            application="test", purpose="internal", expected_source=identity)
        session = sessions["gateway"]
        rules = AssignedRules(resolvers["gateway"], actor="analyst", application="test",
                              purpose="internal", writers=[session.did])
        recorder = EvidenceRecorder(session, store, audit)
        context = Context(session, recorder, rules, application="test", purpose="internal",
                          actor="analyst", resolver=resolvers["gateway"])
        policy = resolution.policies["name"]
        operation = Request("test", "internal", "select", (), (), (policy,), ())
        decision = context._decision(operation, "query")
        for _ in range(options.get("sql_read_count", 1)):
            context._append("SQLRead", operation, "query", {"decision_id": decision,
                "query_id": "query", "row_count": row_count, "source": identity, "assignment_id": assignment_id})
        sources = [{"table": "people", "column": "name", "row": row, "query_id": "query",
                    "assignment_id": assignment_id, "snapshot_sha256": identity["snapshot_sha256"]}
                   for row in range(row_count)]
        sources[1].update(options.get("second_source", {}))
        fields = {f"value_{row}": "Ada" for row in range(row_count)}
        data = session.create(fields, policy, object_type=VALUE_TYPE, group="values")
        store.put(data.snapshot)
        bindings = {field: {"claims": [], "parents": [], "decision_id": decision,
                           "declared": describe("Ada"), "origins": [sources[row]], "source": sources[row]}
                    for row, field in enumerate(fields)}
        data.set(None, {"manifest": {"schema": SCHEMA, "bindings": bindings}}, group="evidence")
        publication = session.release(data, use=tn.UseContext("test", "internal", "select"),
                                      to="governed-runtime", decide=lambda ctx: ctx.writer == session.did)
        store.put(publication)
        outputs, values = {}, []
        for row, field in enumerate(fields):
            reference = ValueReference(publication.id, field, "values")
            result = context._append("Result", operation, "query", {
                "decision_id": decision, "output": reference.to_dict(), "claims": [],
                "field_commitment": publication.envelope["values"]["field_hashes"][field],
                "transition": {"governed_inputs": [], "retained_claims": [], "literal_inputs": []},
                "source": sources[row]})
            values.append(TNString(reference, decision, result, metadata_provider=context.string_metadata))
            outputs[str(row)] = {"reference": reference.to_dict(), "decision_id": decision, "result_id": result}
        edition = resolvers["catalog"].publish_edition(values[0], policy=policy, source=sources[0],
            directory=tmp_path / "edition", application="test", purpose="internal")
        if options.get("wrong_edition_dataset"):
            publisher = resolvers["catalog"]
            draft = tn.DatasetEditionDraft("people.other", "query-0", publication,
                ["values", "evidence"], [tn.ContractBinding(policy.revision_id, "people.name")],
                [tn.UseContext("test", "internal", "inspect-claims")], assignment_id, [])
            record_object = publisher.session.seal(draft.into_draft(admin))
            record = tn.DatasetEdition.from_opened(publisher._open(record_object, "dataset_edition"))
            store.put(record_object)
            edition = {"dataset": record.dataset, "edition": record.edition,
                       "record_id": record.id, "source_object_id": publication.id}
        checkpoint = {**recorder.export(tmp_path / "tapes"), "path": "tapes"}
        run = {"schema": "tn.primitive-run.v1", "authority": authority, "writers": [session.did],
               "administration_policy": admin.fields, "audit_policy": audit.fields,
               "assignment_id": assignment_id, "source": identity,
               "request": {"actor": "analyst", "application": "test", "purpose": "internal"},
               "observer_config": "sessions/observer/tn.yaml", "source_editions": [edition],
               "tapes": {"gateway": checkpoint}, "outputs": outputs}
        (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return tmp_path, run


def test_equal_sibling_values_keep_distinct_nodes_and_evidence(batch_run):
    root, run = batch_run
    summary = verify(root)
    graph = json.loads((root / "dag.json").read_text())
    assert summary["incarnations"] == 2
    assert len(graph["nodes"]) == 2
    assert len({node["id"] for node in graph["nodes"]}) == 2
    assert {node["field"] for node in graph["nodes"]} == {"value_0", "value_1"}
    assert {node["source"]["row"] for node in graph["nodes"]} == {0, 1}
    assert {node["result_id"] for node in graph["nodes"]} == {
        value["result_id"] for value in run["outputs"].values()}
    assert len({node["object_id"] for node in graph["nodes"]}) == 1


@pytest.mark.parametrize("batch_run", [{"sql_read_count": 0}, {"sql_read_count": 2}], indirect=True)
def test_source_batch_requires_exactly_one_signed_sql_read(batch_run):
    root, _ = batch_run
    with pytest.raises(EvidenceError, match="authorized SQL read"):
        verify(root)


@pytest.mark.parametrize("batch_run", [{"rows": 64}], indirect=True)
def test_record_selection_work_is_linear_in_batch_size(batch_run, monkeypatch):
    from tn_governed.values import verify as module

    root, _ = batch_run
    original = module.verify_tape
    counts = Counter()

    class CountedBody(dict):
        def __getitem__(self, key):
            if key == "kind":
                counts["kind_checks"] += 1
            return super().__getitem__(key)

    def counted_records(*args, **kwargs):
        records = original(*args, **kwargs)
        counts["records"] += len(records)
        return tuple(Record(record.id, record.writer, CountedBody(record.body)) for record in records)

    monkeypatch.setattr(module, "verify_tape", counted_records)
    assert verify(root)["incarnations"] == 64
    assert counts["kind_checks"] <= 20 * counts["records"]


@pytest.mark.parametrize("part", ["result_id", "field"])
def test_sibling_handle_cannot_substitute_another_fields_evidence(batch_run, part):
    root, run = batch_run
    if part == "field":
        run["outputs"]["0"]["reference"]["field"] = "value_1"
    else:
        run["outputs"]["0"]["result_id"] = run["outputs"]["1"]["result_id"]
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(EvidenceError, match="exported handle"):
        verify(root)


@pytest.mark.parametrize("batch_run", [
    {"second_source": {"row": 2}}, {"second_source": {"row": -1}},
    {"second_source": {"row": True}}, {"second_source": {"row": 0}},
], indirect=True)
def test_source_row_must_match_signed_query_range_and_field(batch_run):
    root, _ = batch_run
    with pytest.raises(EvidenceError, match="SQL row|SQL field"):
        verify(root)


@pytest.mark.parametrize("batch_run", [{"wrong_edition_dataset": True}], indirect=True)
def test_source_edition_must_name_the_assigned_table_and_column(batch_run):
    root, _ = batch_run
    with pytest.raises(EvidenceError, match="SQL column"):
        verify(root)


def test_shared_publication_is_read_and_opened_once_per_verification(batch_run, monkeypatch):
    root, run = batch_run
    source_id = run["outputs"]["0"]["reference"]["object_id"]
    reads, receives = Counter(), Counter()
    original_read, original_receive = PublicationStore.read, tn.Session.receive

    def read(self, identity):
        reads[identity] += 1
        return original_read(self, identity)

    def receive(self, obj, **kwargs):
        receives[obj.id, tuple(kwargs.get("groups", []))] += 1
        return original_receive(self, obj, **kwargs)

    monkeypatch.setattr(PublicationStore, "read", read)
    monkeypatch.setattr(tn.Session, "receive", receive)
    for repetition in (1, 2):
        assert verify(root)["incarnations"] == 2
        assert reads[source_id] == repetition
        assert receives[source_id, ("evidence",)] == repetition


@pytest.mark.parametrize("fail", [False, True])
def test_verification_cache_is_bounded_and_cleared_on_exit(batch_run, monkeypatch, fail):
    from tn_governed.values import verify as module
    root, run = batch_run
    caches = []

    def tracked_cache(*args, **kwargs):
        def decorate(function):
            cached = lru_cache(*args, **kwargs)(function)
            caches.append(cached)
            return cached
        return decorate

    monkeypatch.setattr(module, "lru_cache", tracked_cache)
    if fail:
        run["outputs"]["0"]["result_id"] = run["outputs"]["1"]["result_id"]
        (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
        with pytest.raises(EvidenceError, match="exported handle"):
            verify(root)
    else:
        verify(root)
    assert caches
    assert all(cache.cache_info().maxsize == 128 and cache.cache_info().currsize == 0
               for cache in caches)


@pytest.mark.parametrize("change", ["authority", "use", "wire"])
def test_later_verification_rechecks_changed_inputs(batch_run, change):
    root, run = batch_run
    verify(root)
    if change == "authority":
        run["authority"] = run["writers"][0]
    elif change == "use":
        run["request"]["purpose"] = "unassigned-purpose"
    else:
        identity = run["outputs"]["0"]["reference"]["object_id"]
        path = root / "publications" / "objects" / (identity[7:] + ".tn")
        path.write_bytes(path.read_bytes() + b" ")
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises((ValueError, GovernedError)):
        verify(root)


def test_successful_terminal_record_cannot_follow_failure(batch_run):
    root, run = batch_run
    checkpoint = run["tapes"]["gateway"]
    with ExitStack() as stack:
        session = stack.enter_context(tn.Session.from_config(str(root / "sessions/gateway/tn.yaml")))
        catalog = stack.enter_context(tn.Session.from_config(str(root / "sessions/catalog/tn.yaml")))
        store = PublicationStore(root / "publications")
        recorder = EvidenceRecorder(session, store, catalog.policy("experiment.evidence"))
        recorder.context_id = checkpoint["context_id"]
        recorder._ids = list(store.record_ids(checkpoint["context_id"]))
        common = {
            "call_id": "query",
            "application": "test",
            "purpose": "internal",
            "operation": "select",
        }
        decision = run["outputs"]["0"]["decision_id"]
        recorder.append("Failure", **common,
            details={"decision_id": decision, "error_type": "TestFailure"})
        recorder.append("Observation", **common,
            details={"decision_id": decision, "observation": "late-success", "value": True})
        retained = recorder.export(root / "tapes-with-late-success")
    run["tapes"]["gateway"] = {**retained, "path": "tapes-with-late-success"}
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(EvidenceError, match="follows a failure"):
        verify(root)
