from contextlib import ExitStack
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
import time

import pytest
import tn

from deepvest_tn.publications import PublicationStore
from experiments.governed_strings.catalog import SignedMetadataCache, UnityGovernance
from experiments.governed_strings.provision import provision
from experiments.governed_strings.string import OperationDenied


class CatalogTransport:
    catalog, schema = "query_test", "governance"
    def __init__(self):
        self.locations, self.lookups = {}, []
    def _ensure(self, path, body):
        if path == "/volumes":
            self.locations[body["name"]] = body["storage_location"]
    def _request(self, method, path):
        self.lookups.append((method, path))
        name = path.rsplit(".", 1)[-1]
        return {"storage_location": self.locations[name], "volume_id": name}


@pytest.fixture
def setup(tmp_path):
    configs = provision(tmp_path / "sessions", roles=["catalog", "gateway", "consumer"], read_only_roles=())
    with ExitStack() as stack:
        sessions = {name: stack.enter_context(tn.Session.from_config(str(path))) for name, path in configs.items()}
        store = PublicationStore(tmp_path / "publications")
        transport = CatalogTransport()
        resolvers = {}
        for name, session in sessions.items():
            resolver = UnityGovernance("http://unused", transport.catalog, session=session,
                authority=sessions["catalog"].did,
                administration=sessions["catalog"].policy("experiment.administration"), store=store)
            resolver.registry = transport
            resolvers[name] = resolver
        source = {"kind": "sqlite", "location": "file:///market.sqlite", "snapshot_sha256": "a" * 64}
        directory = tmp_path / "assignment"
        resolvers["catalog"].publish_assignment(directory, table="market", source_identity=source,
            application="analytics", actors=["analyst"], purposes=["internal"])
        transport.lookups.clear()
        yield resolvers, transport, source, directory


def resolve(resolver, source, **changes):
    return resolver.resolve("market", ["name", "value"], **{
        "actor": "analyst", "application": "analytics", "purpose": "internal",
        "expected_source": source, **changes})


def test_one_lookup_and_independent_endpoint_admission(setup):
    resolvers, transport, source, _ = setup
    arrived = resolve(resolvers["gateway"], source)
    received = resolve(resolvers["consumer"], source, packet=arrived.packet)
    assert len(transport.lookups) == 1
    assert received.assignment_id == arrived.assignment_id
    assert set(received.policies) == {"name", "value"}
    assert all(received.policies[k].matches_contract(arrived.policies[k]) for k in arrived.policies)
    assert resolvers["gateway"].policy_dag is not resolvers["consumer"].policy_dag


@pytest.mark.parametrize("change", [{"actor": "outsider"}, {"purpose": "marketing"},
    {"application": "other"}, {"operation": "unassigned"},
    {"expected_source": {"snapshot_sha256": "different"}}])
def test_carried_packet_never_caches_request_authorization(setup, change):
    resolvers, _, source, _ = setup
    packet = resolve(resolvers["gateway"], source).packet
    with pytest.raises(OperationDenied):
        resolve(resolvers["consumer"], source, packet=packet, **change)


def test_fresh_lookup_and_integrity_cache_on_repeated_query(setup):
    resolvers, transport, source, _ = setup
    resolver = resolvers["gateway"]
    first = resolve(resolver, source)
    before = resolver.metadata_cache.hits
    second = resolve(resolver, source)
    assert len(transport.lookups) == 2
    assert resolver.metadata_cache.hits > before
    assert first.assignment_id == second.assignment_id
    assert resolver.metadata_cache.bytes_used <= resolver.metadata_cache.max_bytes


def test_changed_authority_callback_cannot_reuse_prior_admission(setup):
    resolvers, _, source, _ = setup
    resolver = resolvers["gateway"]
    packet = resolve(resolver, source).packet
    resolver.trust = replace(resolver.trust, policy_revision=lambda revision, parent: False)
    with pytest.raises(Exception):
        resolve(resolver, source, packet=packet)


def test_changed_signed_assignment_is_seen_on_next_query(setup, tmp_path):
    resolvers, _, source, _ = setup
    first = resolve(resolvers["gateway"], source)
    resolvers["catalog"].publish_assignment(tmp_path / "replacement", table="market",
        source_identity=source, application="analytics", actors=["other-actor"], purposes=["internal"])
    with pytest.raises(OperationDenied):
        resolve(resolvers["gateway"], source)
    replacement = resolve(resolvers["gateway"], source, actor="other-actor")
    assert replacement.assignment_id != first.assignment_id


def test_tampered_wire_and_substituted_revision_fail_after_cache_hit(setup):
    resolvers, _, source, directory = setup
    resolver = resolvers["gateway"]
    packet = resolve(resolver, source).packet
    changed = replace(packet, assignment_wire=packet.assignment_wire.replace(b'"timestamp":', b'"altered":', 1))
    assert changed.assignment_wire != packet.assignment_wire
    with pytest.raises(Exception):
        resolve(resolver, source, packet=changed)
    revisions = packet.revision_wires
    with pytest.raises(OperationDenied):
        resolve(resolver, source, packet=replace(packet,
            revision_wires=((revisions[0][0], revisions[-1][1]), *revisions[1:])))
    original = (directory / "assignment.tn").read_bytes()
    (directory / "assignment.tn").write_bytes(original[:-4])
    with pytest.raises(Exception):
        resolve(resolver, source)


def packet_wires(setup):
    resolvers, _, source, _ = setup
    packet = resolve(resolvers["gateway"], source).packet
    return [packet.assignment_wire, *(wire for _identity, wire in packet.revision_wires)]


def test_integrity_cache_evicts_least_recently_used_entry(setup):
    first, second, third, *_ = packet_wires(setup)
    cache = SignedMetadataCache(max_entries=2, max_bytes=10 * 1024 * 1024)
    first_object = cache.parse(first)
    cache.parse(second)
    assert cache.parse(first) is first_object
    cache.parse(third)
    assert len(cache.entries) == 2
    assert cache.bytes_used == len(first) + len(third)
    assert cache.parse(first) is first_object
    before = cache.misses
    cache.parse(second)
    assert cache.misses == before + 1
    assert len(cache.entries) == 2
    assert cache.bytes_used == len(first) + len(second)


def test_integrity_cache_byte_limit_evicts_without_losing_verified_result(setup):
    wires = sorted(packet_wires(setup), key=len)
    small, large = wires[0], wires[-1]
    cache = SignedMetadataCache(max_entries=20, max_bytes=len(large))
    cache.parse(small)
    actual = cache.parse(large)
    assert actual.id == tn.GovernedObject.parse(large).id
    assert len(cache.entries) == 1
    assert cache.bytes_used == len(large)
    cache.parse(small)
    assert len(cache.entries) == 1
    assert cache.bytes_used == len(small)


def test_oversized_metadata_is_verified_without_entering_or_flushing_cache(setup):
    wires = sorted(packet_wires(setup), key=len)
    small, large = wires[0], wires[-1]
    assert len(small) < len(large)
    cache = SignedMetadataCache(max_entries=2, max_bytes=len(small))
    first = cache.parse(small)
    before = dict(cache.entries)
    actual = cache.parse(large)
    assert actual.id == tn.GovernedObject.parse(large).id
    assert cache.entries == before and cache.bytes_used == len(small)
    assert cache.parse(small) is first


@pytest.mark.parametrize("limits", [{"max_entries": 0}, {"max_bytes": 0}])
def test_disabled_cache_retains_no_metadata(setup, limits):
    wire = packet_wires(setup)[0]
    cache = SignedMetadataCache(**limits)
    first, second = cache.parse(wire), cache.parse(wire)
    assert first.id == second.id
    assert not cache.entries and cache.bytes_used == 0
    assert cache.hits == 0 and cache.misses == 2


def test_changed_metadata_authority_is_checked_before_cached_policy_admission(setup):
    resolvers, _, source, _ = setup
    resolver = resolvers["gateway"]
    packet = resolve(resolver, source).packet
    admitted_queries, hits = len(resolver.requests), resolver.metadata_cache.hits
    metadata_calls, revision_calls = [], []
    resolver.trust = replace(resolver.trust,
        metadata=lambda context: metadata_calls.append(context.object.id) or False,
        policy_revision=lambda revision, parent: revision_calls.append(revision.id) or True)
    with pytest.raises(Exception):
        resolve(resolver, source, packet=packet)
    assert resolver.metadata_cache.hits == hits + 1
    assert len(metadata_calls) == 1
    assert revision_calls == []
    assert len(resolver.requests) == admitted_queries


def test_consumer_rejects_previously_accepted_packet_when_its_metadata_authority_changes(setup):
    resolvers, transport, source, _ = setup
    packet = resolve(resolvers["gateway"], source).packet
    consumer = resolvers["consumer"]
    resolve(consumer, source, packet=packet)
    consumer.trust = replace(consumer.trust, metadata=lambda context: False)
    with pytest.raises(Exception):
        resolve(consumer, source, packet=packet)
    assert len(transport.lookups) == 1
    assert len(consumer.requests) == 1


def test_concurrent_integrity_cache_misses_do_not_double_count_retained_wire(setup, monkeypatch):
    from experiments.governed_strings import catalog as module

    wire = packet_wires(setup)[0]
    native_read = tn.GovernedObject.read
    def scheduled_parse(raw):
        # Make simultaneous misses overlap without depending on native GIL behavior.
        time.sleep(0.02)
        return native_read(raw)
    monkeypatch.setattr(module, "tn", SimpleNamespace(
        GovernedObject=SimpleNamespace(read=scheduled_parse)))
    cache = SignedMetadataCache(max_entries=8, max_bytes=8 * len(wire))
    start = Barrier(8)
    def parse(_index):
        start.wait(timeout=5)
        return cache.parse(wire)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(parse, range(8)))
    assert len({item.id for item in results}) == 1
    assert len(cache.entries) == 1
    assert cache.bytes_used == len(wire)
    assert cache.bytes_used == sum(size for _object, size in cache.entries.values())
