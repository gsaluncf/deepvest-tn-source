"""One catalog registration and lookup for an immutable group of source editions."""
import hashlib
import json
import shutil

import pytest
import tn

from deepvest_tn.unity import resolve_finance_editions
from governed_edition_fixtures import signed_edition
from test_unity_editions import request, trust
from test_unity_registration import VolumeTransport
from tn_polars.registry import LocalRegistry, UnityCatalogRegistry
from tn_polars.release import PolicyDenied


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("signed-query-editions")
    first = signed_edition(root / "first")
    second = signed_edition(root / "second", session=first.session, edition="new")
    yield first, second
    first.session.close()


@pytest.fixture
def batch(tmp_path, sources, monkeypatch):
    directory = tmp_path / "query"
    entries = []
    for index, source in enumerate(sources):
        bundle = directory / str(index)
        shutil.copytree(source.bundle, bundle)
        entries.append((source.record.dataset, source.record.edition, source.record.id, bundle))
    registry = UnityCatalogRegistry("http://catalog", "finance", "sources")
    transport = VolumeTransport()
    monkeypatch.setattr(registry, "_request", transport.request)
    return registry, transport, directory, entries


def requested(entries):
    return [entry[:3] for entry in entries]


def volume(batch_id="query-1"):
    return "/volumes/finance.sources.tn_query_" + hashlib.sha256(batch_id.encode()).hexdigest()


@pytest.mark.parametrize("create_namespace", [True, False])
def test_batch_has_one_registration_and_one_readback(batch, sources, create_namespace):
    registry, transport, directory, entries = batch
    loaded = registry.register_editions("query-1", entries, directory,
                                       create_namespace=create_namespace)
    setup = [("POST", "/catalogs"), ("POST", "/schemas")] if create_namespace else []
    assert transport.calls == [*setup, ("POST", "/volumes"), ("GET", volume())]
    assert [item.record_id for item in loaded] == [source.record.id for source in sources]
    assert [item.edition_wire for item in loaded] == [source.record.object.forward() for source in sources]
    transport.calls.clear()
    reloaded = registry.get_editions(requested(entries), batch_id="query-1")
    assert transport.calls == [("GET", volume())]
    assert reloaded == loaded


def test_exact_retry_retains_index_and_still_checks_catalog(batch):
    registry, transport, directory, entries = batch
    registry.register_editions("query-1", entries, directory, create_namespace=False)
    before = (directory / "editions.json").read_bytes()
    transport.calls.clear()
    registry.register_editions("query-1", entries, directory, create_namespace=False)
    assert (directory / "editions.json").read_bytes() == before
    assert transport.calls == [("POST", "/volumes"), ("GET", volume())]


def test_changed_or_duplicate_membership_cannot_replace_index(batch):
    registry, transport, directory, entries = batch
    registry.register_editions("query-1", entries, directory, create_namespace=False)
    transport.calls.clear()
    with pytest.raises(PolicyDenied, match="differs"):
        registry.register_editions("query-1", entries[:1], directory, create_namespace=False)
    with pytest.raises(PolicyDenied, match="unique|duplicate"):
        registry.register_editions("query-2", [entries[0], entries[0]], directory,
                                   create_namespace=False)
    assert transport.calls == []


def test_external_bundle_cannot_enter_batch(batch, tmp_path):
    registry, transport, directory, entries = batch
    external = tmp_path / "external"
    shutil.copytree(entries[0][3], external)
    with pytest.raises(PolicyDenied, match="within|outside|descendant"):
        registry.register_editions("query-1", [(*entries[0][:3], external)], directory,
                                   create_namespace=False)
    assert transport.calls == []


@pytest.mark.parametrize("path", ["../0", "/absolute", "C:/outside", "a/../../0", "\\server\\share"])
def test_index_path_escape_is_rejected(batch, path):
    registry, _, directory, entries = batch
    registry.register_editions("query-1", entries, directory, create_namespace=False)
    manifest_path = directory / "editions.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["editions"][0]["bundle"] = path
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(PolicyDenied):
        registry.get_editions(requested(entries), batch_id="query-1")


def test_storage_location_and_requested_batch_are_checked(batch, tmp_path):
    registry, transport, directory, entries = batch
    transport.return_location = (tmp_path / "other").as_uri()
    with pytest.raises(PolicyDenied, match="another"):
        registry.register_editions("query-1", entries, directory, create_namespace=False)
    transport.return_location = directory.as_uri()
    with pytest.raises(PolicyDenied, match="identity|batch"):
        registry.get_editions(requested(entries), batch_id="other-query")


@pytest.mark.parametrize("change", ["dataset", "edition", "record_id"])
def test_every_requested_identity_must_match(batch, change):
    registry, _, directory, entries = batch
    registry.register_editions("query-1", entries, directory, create_namespace=False)
    values = list(entries[0][:3])
    values[{"dataset": 0, "edition": 1, "record_id": 2}[change]] = (
        "sha256:" + "0" * 64 if change == "record_id" else "other")
    with pytest.raises(PolicyDenied, match="identity|requested|match"):
        registry.get_editions([values], batch_id="query-1")


def test_fresh_lookup_cannot_bypass_revoked_catalog_credentials(batch, monkeypatch):
    registry, _, directory, entries = batch
    registry.register_editions("query-1", entries, directory, create_namespace=False)
    registry.token = "new-credential"
    def refuse(*args):
        raise PolicyDenied("HTTP 403: revoked")
    monkeypatch.setattr(registry, "_request", refuse)
    with pytest.raises(PolicyDenied, match="403"):
        registry.get_editions(requested(entries), batch_id="query-1")


class ReturnedBatch:
    def __init__(self, entries):
        self.entries = {(item.dataset, item.edition, item.record_id): item for item in entries}

    def get_edition(self, dataset, edition, record_id):
        return self.entries[(dataset, edition, record_id)]


def test_returned_transports_still_require_native_authority_and_use_admission(batch, sources):
    registry, _, directory, entries = batch
    loaded = registry.register_editions("query-1", entries, directory, create_namespace=False)
    arguments = dict(registry=ReturnedBatch(loaded), catalog_session=sources[0].session,
                     policy_dag=tn.PolicyDag(), trust=trust(sources[0].session.did))
    resolved = resolve_finance_editions([request(item) for item in sources], **arguments)
    assert [item.selection.source_object_id for item in resolved] == [item.source.id for item in sources]
    for result, source in zip(resolved, sources, strict=True):
        result.selection.verify_source(source.session.governance(source.source), request(source).use,
                                       ["finance"])
    with pytest.raises(Exception):
        resolve_finance_editions([request(sources[0], tn.UseContext("other", "portfolio_analysis", "join_prices"))],
                                 **arguments)
    with pytest.raises(Exception):
        resolve_finance_editions([request(sources[0])], **(arguments | {"trust": trust("unaccepted")}))


@pytest.mark.parametrize("tamper", ["edition", "artifact"])
def test_substituted_signed_material_is_rejected_by_native_admission(batch, sources, tamper):
    registry, _, directory, entries = batch
    registry.register_editions("query-1", entries, directory, create_namespace=False)
    if tamper == "edition":
        (entries[0][3] / "edition.tn").write_bytes(sources[1].record.object.forward())
    else:
        target = entries[0][3] / "evaluators" / sources[0].revision.id[7:] / "data.json"
        target.write_bytes(b'{"changed":true}')
    loaded = registry.get_editions(requested(entries), batch_id="query-1")
    with pytest.raises(Exception, match="identity|digest"):
        resolve_finance_editions([request(sources[0])], registry=ReturnedBatch(loaded),
            catalog_session=sources[0].session, policy_dag=tn.PolicyDag(), trust=trust(sources[0].session.did))


def test_local_registry_has_same_explicit_batch_lookup(batch, tmp_path):
    _, _, directory, entries = batch
    registry = LocalRegistry(tmp_path / "registry")
    returned = registry.register_editions("query-1", entries, directory)
    assert registry.get_editions(requested(entries), batch_id="query-1") == returned
    with pytest.raises(PolicyDenied, match="unique|duplicate"):
        registry.get_editions([entries[0][:3], entries[0][:3]], batch_id="query-1")
