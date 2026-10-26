"""A verification run shares publication work while checking every sibling field."""

from collections import Counter
import functools
import json
import shutil
import sqlite3
from types import SimpleNamespace

import pytest
import tn
from tn.governed import GovernedError

from tn_governed.sqlite_publications import SQLitePublicationStore
from tn_governed.values.evidence import EvidenceError
from experiments.pvldb_evaluation import verify_frame as module
from experiments.pvldb_evaluation.run import run_case
from tn_polars.registry import UnityCatalogRegistry


@pytest.fixture(scope="module")
def retained_frame(tmp_path_factory):
    root = tmp_path_factory.mktemp("frame-cache") / "case"
    volumes = {}

    def request(self, method, path, payload=None):
        if path in {"/catalogs", "/schemas"}:
            return {}
        if method == "POST" and path == "/volumes":
            full = f"{payload['catalog_name']}.{payload['schema_name']}.{payload['name']}"
            volumes[full] = dict(payload)
            return volumes[full]
        if method == "GET" and path.startswith("/volumes/"):
            return volumes[path.removeprefix("/volumes/")]
        raise AssertionError((method, path))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(UnityCatalogRegistry, "_request", request)
        result = run_case(root, "primitive", rows=5)
    assert result["completed"] and result["verified"], result
    return root


@pytest.fixture
def saved_frame(retained_frame, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(retained_frame, root)
    return root, json.loads((root / "run.json").read_text())


@pytest.fixture
def observation(monkeypatch):
    reads, receives, views, lineage = Counter(), Counter(), Counter(), Counter()
    decoded, manifests, caches, clears = [], [], [], []
    native_read = SQLitePublicationStore.read
    native_receive, native_governance = tn.Session.receive, tn.Session.governance
    native_lineage, native_decode = tn.LineageVerifier, module.decode
    native_get = tn.DataObject.get

    def read(self, identity):
        reads[identity] += 1
        return native_read(self, identity)

    def receive(self, obj, *args, **kwargs):
        receives[obj.id] += 1
        return native_receive(self, obj, *args, **kwargs)

    def governance(self, obj):
        views[obj.id] += 1
        return native_governance(self, obj)

    def get(self, field, *args, **kwargs):
        if field == "manifest" and kwargs.get("group") == "evidence":
            manifests.append(field)
        return native_get(self, field, *args, **kwargs)

    class Lineage:
        def __init__(self, *args, **kwargs):
            self.native = native_lineage(*args, **kwargs)

        def verify(self, view, *args, **kwargs):
            lineage[view.object.id] += 1
            return self.native.verify(view, *args, **kwargs)

    class Native:
        LineageVerifier = Lineage

        def __getattr__(self, name):
            return getattr(tn, name)

    def decode(payload, kind):
        decoded.append((payload, kind))
        return native_decode(payload, kind)

    def cache_factory(*args, **kwargs):
        decorator = functools.lru_cache(*args, **kwargs)

        def decorate(function):
            cached = decorator(function)
            clear = cached.cache_clear

            def clear_and_observe():
                clears.append((cached.__name__, cached.cache_info().currsize))
                clear()

            cached.cache_clear = clear_and_observe
            caches.append(cached)
            return cached

        return decorate

    monkeypatch.setattr(SQLitePublicationStore, "read", read)
    monkeypatch.setattr(tn.Session, "receive", receive)
    monkeypatch.setattr(tn.Session, "governance", governance)
    monkeypatch.setattr(tn.DataObject, "get", get)
    monkeypatch.setattr(module, "tn", Native())
    monkeypatch.setattr(module, "decode", decode)
    monkeypatch.setattr(module, "lru_cache", cache_factory, raising=False)
    return SimpleNamespace(reads=reads, receives=receives, views=views, lineage=lineage,
                           decoded=decoded, manifests=manifests, caches=caches, clears=clears)


def assert_cleared(observation, *, minimum=3):
    assert len(observation.caches) >= minimum
    assert all(cache.cache_info().maxsize == 128 for cache in observation.caches)
    assert all(cache.cache_info().currsize == 0 for cache in observation.caches)
    assert len(observation.clears) == len(observation.caches)


def test_shared_publications_are_verified_and_opened_once(saved_frame, observation):
    root, run = saved_frame
    result = module.verify_frame(root)
    sources = {item["source_object_id"] for item in run["source_editions"]}
    assert len(sources) == 2
    for identity in sources:
        assert observation.reads[identity] == 1
        assert observation.receives[identity] == 1
        assert observation.views[identity] == 1
        assert observation.lineage[identity] == 1
    assert len(observation.decoded) == 12  # Ten source fields and two numeric outputs.
    references = run["evaluation"]["dataframe"][0]["input_references"] + [
        run["outputs"][label]["reference"] for label in ("dataframe_result", "amount")]
    assert len(observation.manifests) == len({reference["object_id"] for reference in references})
    assert {payload for payload, kind in observation.decoded if kind == "int"} >= {
        "1250", "1251", "1252", "1253", "1254"}
    assert result["opa_decisions_replayed"] == 12
    assert_cleared(observation)


@pytest.mark.parametrize("changed", ["authority", "use", "wire"])
def test_next_verification_rechecks_changed_inputs(saved_frame, observation, changed):
    root, run = saved_frame
    module.verify_frame(root)
    assert_cleared(observation)
    first_cache_ids = {id(cache) for cache in observation.caches}
    source = run["source_editions"][0]["source_object_id"]
    if changed == "authority":
        run["authority"] = run["writers"][0]
        (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    elif changed == "use":
        run["request"]["purpose"] = "another-purpose"
        (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    else:
        with sqlite3.connect(root / "publications/publications.sqlite3") as database:
            wire = database.execute("SELECT wire FROM publications WHERE object_id=?", (source,)).fetchone()[0]
            database.execute("UPDATE publications SET wire=? WHERE object_id=?", (wire + b" ", source))
    with pytest.raises((ValueError, GovernedError), match="wire digest differs" if changed == "wire" else None):
        module.verify_frame(root)
    assert len({id(cache) for cache in observation.caches} - first_cache_ids) >= 1
    assert_cleared(observation)
    if changed == "wire":
        assert observation.reads[source] == 2
        assert observation.receives[source] == 1


def test_duplicate_records_across_chains_are_rejected(saved_frame, observation):
    root, run = saved_frame
    checkpoint = next(iter(run["tapes"].values()))
    run["tapes"]["duplicate"] = dict(checkpoint)
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(EvidenceError, match="duplicate evidence record"):
        module.verify_frame(root)
    assert_cleared(observation, minimum=2)


def test_cached_publication_still_checks_each_siblings_declared_facts(saved_frame, observation, monkeypatch):
    root, run = saved_frame
    describe = module.describe

    def wrong_second_field(value):
        facts = describe(value)
        if type(value) is int and value == 1251:
            return {**facts, "kind": "float"}
        return facts

    monkeypatch.setattr(module, "describe", wrong_second_field)
    with pytest.raises(EvidenceError, match="declared facts"):
        module.verify_frame(root)
    source = next(item["source_object_id"] for item in run["source_editions"]
                  if item["dataset"].endswith(".value"))
    assert observation.receives[source] == 1
    assert ("1250", "int") in observation.decoded and ("1251", "int") in observation.decoded
    assert_cleared(observation)
