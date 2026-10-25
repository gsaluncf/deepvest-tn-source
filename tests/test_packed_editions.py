import json
from concurrent.futures import ThreadPoolExecutor
import zipfile

import pytest
import tn

from deepvest_tn.unity import resolve_finance_editions
from governed_edition_fixtures import BUSINESS_USE, signed_edition
from test_unity_editions import request, trust
from tn_polars import registry as module
from tn_polars.opa import OpaModuleCache
from tn_polars.release import PolicyDenied


@pytest.fixture
def source(tmp_path):
    return signed_edition(tmp_path / "source")


def arguments(source):
    manifest = json.loads((source.bundle / "bundle.json").read_bytes())
    return dict(dataset=source.record.dataset, edition=source.record.edition,
                record_id=source.record.id, edition_wire=source.record.object.forward(),
                revision_wires=[(identity, (source.bundle / "revisions" / f"{identity[7:]}.tn").read_bytes())
                                for identity in manifest["revisions"]],
                evaluators={source.revision.id: source.files})


def pack(source, tmp_path):
    return module.write_edition_bundle(tmp_path / "packed", **arguments(source), packed=True)


def resolve(source, root, tmp_path, *, use=BUSINESS_USE, decisions=None):
    registry = module.LocalRegistry(tmp_path / "registry")
    registry.register_edition(source.record.dataset, source.record.edition, source.record.id, root)
    return resolve_finance_editions(
        [request(source, use)], registry=registry, catalog_session=source.session,
        policy_dag=tn.PolicyDag(), trust=decisions or trust(source.session.did),
    )[0]


def replace_member(archive, name, value):
    with zipfile.ZipFile(archive) as bundle:
        members = {info.filename: bundle.read(info) for info in bundle.infolist()}
    members[name] = value
    with zipfile.ZipFile(archive, "w") as bundle:
        for member, raw in members.items():
            bundle.writestr(member, raw)


def test_packed_bundle_publishes_once_and_retains_every_exact_file(source, tmp_path, monkeypatch):
    writes = []
    write = module._immutable_write

    def observed(path, content):
        writes.append(path)
        return write(path, content)

    monkeypatch.setattr(module, "_immutable_write", observed)
    root = pack(source, tmp_path)
    assert writes == [root / "bundle.zip"]
    with zipfile.ZipFile(root / "bundle.zip") as bundle:
        actual = {info.filename: bundle.read(info) for info in bundle.infolist()}
    expected = {p.relative_to(source.bundle).as_posix(): p.read_bytes()
                for p in source.bundle.rglob("*") if p.is_file()}
    assert actual == expected
    admitted = resolve(source, root, tmp_path)
    assert admitted.selection.binding == source.selection.binding
    entry = admitted.entries[source.revision.id]
    assert entry.evaluator().decide({"operation": "load", "audience": "internal",
        "policy_ref": entry.policy.policy_ref, "columns": [], "row_selection_sources": []}).allow


def test_exact_retry_is_deterministic_and_different_content_cannot_replace_it(source, tmp_path):
    root = pack(source, tmp_path)
    before = (root / "bundle.zip").read_bytes()
    pack(source, tmp_path)
    assert (root / "bundle.zip").read_bytes() == before
    args = arguments(source)
    args["edition_wire"] += b" "
    with pytest.raises(PolicyDenied, match="differs"):
        module.write_edition_bundle(root, **args, packed=True)
    assert (root / "bundle.zip").read_bytes() == before


def test_failed_atomic_publication_leaves_no_readable_partial_bundle(source, tmp_path, monkeypatch):
    link = module.os.link

    def fail(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(module.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        pack(source, tmp_path)
    assert not (tmp_path / "packed" / "bundle.zip").exists()
    assert list((tmp_path / "packed").iterdir()) == []
    monkeypatch.setattr(module.os, "link", link)
    assert resolve(source, pack(source, tmp_path), tmp_path).selection.binding == source.selection.binding


def test_concurrent_conflicting_publications_never_replace_the_winner(source, tmp_path):
    args = arguments(source)
    altered = {**args, "edition_wire": args["edition_wire"] + b" "}

    def publish(kwargs):
        try:
            module.write_edition_bundle(tmp_path / "packed", **kwargs, packed=True)
            return kwargs["edition_wire"]
        except PolicyDenied:
            return None

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(publish, [args, altered]))
    winners = [wire for wire in results if wire is not None]
    assert len(winners) == 1
    with zipfile.ZipFile(tmp_path / "packed" / "bundle.zip") as bundle:
        assert bundle.read("edition.tn") == winners[0]


@pytest.mark.parametrize("filename", ["compiled_policy.json", "profile.json", "policy.wasm", "data.json"])
def test_packed_artifact_tampering_is_rejected_before_evaluator_loading(source, tmp_path, monkeypatch, filename):
    root = pack(source, tmp_path)
    replace_member(root / "bundle.zip", f"evaluators/{source.revision.id[7:]}/{filename}", b"altered")
    prepared = []
    monkeypatch.setattr("tn_polars.opa.OpaWasmEvaluator._initialize", lambda *a, **k: prepared.append(True))
    with pytest.raises(PolicyDenied, match="digest"):
        resolve(source, root, tmp_path)
    assert prepared == []


@pytest.mark.parametrize("filename", ["policy.wasm", "data.json"])
def test_current_archive_is_rechecked_after_a_warm_module_cache(source, tmp_path, monkeypatch, filename):
    root = pack(source, tmp_path)
    entry = resolve(source, root, tmp_path).entries[source.revision.id]
    cache = OpaModuleCache()
    entry.evaluator(module_cache=cache)
    replace_member(root / "bundle.zip", f"evaluators/{source.revision.id[7:]}/{filename}", b"altered")
    prepared = []
    monkeypatch.setattr("tn_polars.opa.OpaWasmEvaluator._initialize", lambda *a, **k: prepared.append(True))
    with pytest.raises(PolicyDenied, match="digest"):
        entry.evaluator(module_cache=cache)
    assert prepared == []


@pytest.mark.parametrize("mismatch", ["use", "writer"])
def test_packed_transport_does_not_change_native_authority_or_use(source, tmp_path, monkeypatch, mismatch):
    root = pack(source, tmp_path)
    loaded = []
    monkeypatch.setattr("deepvest_tn.unity.verified_edition_entry", lambda *a, **kw: loaded.append(True))
    kwargs = ({"use": tn.UseContext("other", "portfolio_analysis", "join_prices")} if mismatch == "use"
              else {"decisions": trust("did:untrusted")})
    with pytest.raises(tn.governed.UseDenied):
        resolve(source, root, tmp_path, **kwargs)
    assert loaded == []


@pytest.mark.parametrize("damage", ["duplicate", "truncated", "missing"])
def test_malformed_archive_is_not_accepted(source, tmp_path, damage):
    root = pack(source, tmp_path)
    path = root / "bundle.zip"
    if damage == "duplicate":
        with zipfile.ZipFile(path, "a") as bundle, pytest.warns(UserWarning, match="Duplicate"):
            bundle.writestr("bundle.json", bundle.read("bundle.json"))
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:100])
    else:
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("bundle.json", b"{}")
    with pytest.raises(PolicyDenied):
        module._load_edition(source.record.dataset, source.record.edition, source.record.id, root)
