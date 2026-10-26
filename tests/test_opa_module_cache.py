import json

import pytest
import tn
import wasmtime

from deepvest_tn.unity import resolve_finance_editions
from governed_edition_fixtures import signed_edition
from test_unity_editions import register, request, trust
from tn_polars import opa
from tn_polars.registry import LocalRegistry
from tn_polars.tn_boundary import PolicySet


@pytest.fixture
def source(tmp_path):
    return signed_edition(tmp_path / "source")


@pytest.fixture
def compilations(monkeypatch):
    original = wasmtime.Module
    calls = []

    def compile_module(engine, raw):
        calls.append(bytes(raw))
        return original(engine, raw)

    monkeypatch.setattr(wasmtime, "Module", compile_module)
    return calls


def evaluator(source, cache, *, data=None, revision=None, wasm=None):
    return opa.OpaWasmEvaluator.from_bytes(
        source.files["policy.wasm"] if wasm is None else wasm,
        source.files["data.json"] if data is None else data,
        module_cache=cache,
        policy_revision=source.revision.id if revision is None else revision,
    )


def permitted_request(source):
    return {"operation": "load", "audience": "internal",
            "policy_ref": json.loads(source.files["compiled_policy.json"])["policy_ref"],
            "columns": [], "row_selection_sources": []}


def test_compiled_code_is_reused_but_state_and_decisions_are_independent(source, compilations):
    cache = opa.OpaModuleCache()
    first = evaluator(source, cache)
    second = evaluator(source, cache)
    assert len(compilations) == 1
    assert first.store is not second.store
    assert first.instance is not second.instance
    assert first.memory is not second.memory
    allowed = permitted_request(source)
    assert first.decide(allowed).allow
    assert second.decide(allowed).allow
    assert not second.decide({**allowed, "operation": "unassigned"}).allow
    assert not second.decide({**allowed, "operation": "release", "audience": "unassigned"}).allow
    assert first.decide(allowed).allow


def test_changed_policy_data_is_loaded_even_when_compiled_code_is_reused(source, compilations):
    cache = opa.OpaModuleCache()
    first = evaluator(source, cache)
    data = json.loads(source.files["data.json"])
    data["tn_polars"]["policy"]["operations"] = []
    second = evaluator(source, cache, data=json.dumps(data).encode())
    assert len(compilations) == 1
    assert first.decide(permitted_request(source)).allow
    assert not second.decide(permitted_request(source)).allow
    assert first.decide(permitted_request(source)).allow


def test_different_revision_and_new_cache_compile_independently(source, compilations):
    cache = opa.OpaModuleCache()
    evaluator(source, cache)
    evaluator(source, cache, revision="another-revision")
    evaluator(source, opa.OpaModuleCache())
    assert len(compilations) == 3


def test_invalid_changed_wasm_never_uses_the_previous_module(source, compilations):
    cache = opa.OpaModuleCache()
    evaluator(source, cache)
    with pytest.raises(wasmtime.WasmtimeError):
        evaluator(source, cache, wasm=b"invalid wasm")
    assert len(compilations) == 2
    assert evaluator(source, cache).decide(permitted_request(source)).allow
    assert len(compilations) == 2


@pytest.mark.parametrize("bound", ["entries", "bytes"])
def test_least_recently_used_modules_are_evicted_without_invalidating_evaluators(source, compilations, bound):
    cache = (opa.OpaModuleCache(max_entries=1) if bound == "entries" else
             opa.OpaModuleCache(max_source_bytes=len(source.files["policy.wasm"])))
    first = evaluator(source, cache)
    evaluator(source, cache, revision="another-revision")
    evaluator(source, cache)
    assert len(compilations) == 3
    assert first.decide(permitted_request(source)).allow


def test_oversized_module_is_used_but_not_retained(source, compilations):
    cache = opa.OpaModuleCache(max_source_bytes=1)
    assert evaluator(source, cache).decide(permitted_request(source)).allow
    assert evaluator(source, cache).decide(permitted_request(source)).allow
    assert len(compilations) == 2


class FakeMemory:
    def __init__(self, raw):
        self.raw = raw
        self.reads = []

    def data_len(self, _store):
        return len(self.raw)

    def read(self, _store, start, end):
        self.reads.append((start, end))
        return self.raw[start:end]


def test_opa_string_read_stops_at_the_terminator():
    evaluator = opa.OpaWasmEvaluator.__new__(opa.OpaWasmEvaluator)
    evaluator.store = object()
    evaluator.memory = FakeMemory(b"xx{\"allow\":true}\0" + b"unused" * 20_000)
    assert evaluator._read_c_string(2) == '{"allow":true}'
    assert evaluator.memory.reads == [(2, 2 + 64 * 1024)]


def test_opa_string_read_rejects_missing_terminator():
    evaluator = opa.OpaWasmEvaluator.__new__(opa.OpaWasmEvaluator)
    evaluator.store = object()
    evaluator.memory = FakeMemory(b"unterminated")
    with pytest.raises(ValueError, match="unterminated"):
        evaluator._read_c_string(0)


def test_cache_requires_policy_revision(source, compilations):
    with pytest.raises(ValueError, match="revision"):
        opa.OpaWasmEvaluator.from_bytes(source.files["policy.wasm"], source.files["data.json"],
                                        module_cache=opa.OpaModuleCache())
    assert compilations == []


@pytest.mark.parametrize("mutation", ["wasm", "data", "policy", "profile"])
def test_warm_cache_does_not_bypass_current_signed_artifact_checks(source, tmp_path, compilations, monkeypatch, mutation):
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, source)
    resolved, = resolve_finance_editions(
        [request(source)], registry=registry, catalog_session=source.session,
        policy_dag=tn.PolicyDag(), trust=trust(source.session.did),
    )
    cache = opa.OpaModuleCache()
    first = PolicySet.for_selection(resolved.entries, resolved.selection, module_cache=cache)
    initialized = []
    initialize = opa.OpaWasmEvaluator._initialize

    def observe_initialize(*args, **kwargs):
        initialized.append(True)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(opa.OpaWasmEvaluator, "_initialize", observe_initialize)
    entry = resolved.entries[source.revision.id]
    if mutation == "policy":
        entry.policy.fields["amount"] = "public"
    elif mutation == "profile":
        entry.dtypes["amount"] = "String"
    else:
        path = entry.wasm_path if mutation == "wasm" else entry.data_path
        path.write_bytes(path.read_bytes() + b" altered")
    with pytest.raises(Exception, match="digest|changed"):
        PolicySet.for_selection(resolved.entries, resolved.selection, module_cache=cache)
    assert initialized == []
    assert len(compilations) == 1
    assert first._entries[source.revision.id].evaluator.decide(permitted_request(source)).allow
