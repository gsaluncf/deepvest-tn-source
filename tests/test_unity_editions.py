import pytest
import tn
from tn_polars.registry import LocalRegistry

from deepvest_tn.unity import CatalogTrust, FinanceEditionRequest, resolve_finance_editions
from deepvest_tn.publications import PublicationStore
from governed_edition_fixtures import BUSINESS_USE, METADATA_USE, signed_edition


def trust(writer):
    return CatalogTrust(
        metadata=lambda context: context.writer == writer,
        policy_revision=lambda revision, parent: revision.writer == writer,
        edition=lambda record: record.writer == writer,
        metadata_use=METADATA_USE,
    )


def request(fixture, use=BUSINESS_USE):
    return FinanceEditionRequest(fixture.record.dataset, fixture.record.edition, fixture.record.id, use)


def register(registry, fixture):
    registry.register_edition(fixture.record.dataset, fixture.record.edition, fixture.record.id, fixture.bundle)


def test_native_resolver_accepts_both_explicit_editions_and_distinct_contracts(tmp_path):
    first = signed_edition(tmp_path / "first")
    second = signed_edition(tmp_path / "second", session=first.session, edition="new")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, first)
    register(registry, second)
    resolved = resolve_finance_editions(
        [request(first), request(second)], registry=registry, catalog_session=first.session,
        policy_dag=tn.PolicyDag(), trust=trust(first.session.did),
    )
    assert [item.selection.source_object_id for item in resolved] == [first.source.id, second.source.id]
    assert first.source.id != second.source.id
    assert first.revision.id != second.revision.id
    registry.register_feed("market.prices", second.bundle)
    assert resolved[0].selection.edition_record_id == first.record.id
    assert resolved[0].catalog is resolved[1].catalog
    entry = resolved[0].entries[first.revision.id]
    assert entry.revision_id == first.revision.id
    assert entry.verified_artifacts == first.record.evaluator_artifacts[0]
    assert entry.evaluator().decide({
        "operation": "load", "audience": "internal", "policy_ref": entry.policy.policy_ref,
        "columns": [], "row_selection_sources": [],
    }).allow


@pytest.mark.parametrize("filename", ["compiled_policy.json", "profile.json", "policy.wasm", "data.json"])
def test_signed_artifact_digest_failure_prevents_evaluator_preparation(tmp_path, filename):
    fixture = signed_edition(tmp_path / "source")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, fixture)
    artifact = fixture.bundle / "evaluators" / fixture.revision.id.removeprefix("sha256:") / filename
    artifact.write_bytes(artifact.read_bytes() + b" altered")
    with pytest.raises(Exception, match="digest"):
        resolve_finance_editions(
            [request(fixture)], registry=registry, catalog_session=fixture.session,
            policy_dag=tn.PolicyDag(), trust=trust(fixture.session.did),
        )


def test_metadata_writer_refusal_occurs_before_any_metadata_group_open(tmp_path):
    fixture = signed_edition(tmp_path / "source")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, fixture)
    opened = []

    class ObservedSession:
        def governance(self, object):
            return fixture.session.governance(object)

        def open(self, admitted, groups):
            opened.append(groups)
            return fixture.session.open(admitted, groups)

    with pytest.raises(tn.governed.UseDenied):
        resolve_finance_editions(
            [request(fixture)], registry=registry, catalog_session=ObservedSession(),
            policy_dag=tn.PolicyDag(), trust=trust("did:untrusted"),
        )
    assert opened == []


def test_native_selection_rejects_wrong_complete_use(tmp_path):
    fixture = signed_edition(tmp_path / "source")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, fixture)
    with pytest.raises(tn.governed.UseDenied):
        resolve_finance_editions(
            [request(fixture, tn.UseContext("other", "portfolio_analysis", "join_prices"))],
            registry=registry, catalog_session=fixture.session, policy_dag=tn.PolicyDag(),
            trust=trust(fixture.session.did),
        )


@pytest.mark.parametrize("authority", ["revision", "edition"])
def test_metadata_authority_refusal_precedes_evaluator_preparation(tmp_path, monkeypatch, authority):
    fixture = signed_edition(tmp_path / "source")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, fixture)
    decisions = trust(fixture.session.did)
    if authority == "revision":
        decisions = CatalogTrust(decisions.metadata, lambda revision, parent: False,
                                 decisions.edition, METADATA_USE)
    else:
        decisions = CatalogTrust(decisions.metadata, decisions.policy_revision,
                                 lambda record: False, METADATA_USE)
    prepared = []
    monkeypatch.setattr("deepvest_tn.unity.verified_edition_entry", lambda *args: prepared.append(args))
    with pytest.raises(tn.governed.GovernedError):
        resolve_finance_editions(
            [request(fixture)], registry=registry, catalog_session=fixture.session,
            policy_dag=tn.PolicyDag(), trust=decisions,
        )
    assert prepared == []


def test_revision_wire_substitution_is_rejected_before_metadata_admission(tmp_path):
    fixture = signed_edition(tmp_path / "source")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, fixture)
    path = fixture.bundle / "revisions" / f"{fixture.revision.id[7:]}.tn"
    path.write_bytes(fixture.source.wire.encode())
    decisions = []
    checked = CatalogTrust(lambda context: decisions.append(context), lambda *args: True,
                           lambda record: True, METADATA_USE)
    with pytest.raises(Exception, match="identity"):
        resolve_finance_editions(
            [request(fixture)], registry=registry, catalog_session=fixture.session,
            policy_dag=tn.PolicyDag(), trust=checked,
        )
    assert decisions == []


@pytest.mark.parametrize("mutation", ["wasm", "data", "policy", "profile"])
def test_evaluator_rechecks_artifacts_and_parsed_metadata_at_loading_time(tmp_path, mutation):
    fixture = signed_edition(tmp_path / "source")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, fixture)
    resolved, = resolve_finance_editions(
        [request(fixture)], registry=registry, catalog_session=fixture.session,
        policy_dag=tn.PolicyDag(), trust=trust(fixture.session.did),
    )
    entry = resolved.entries[fixture.revision.id]
    if mutation == "policy":
        entry.policy.fields["amount"] = "public"
    elif mutation == "profile":
        entry.dtypes["amount"] = "String"
    else:
        path = entry.wasm_path if mutation == "wasm" else entry.data_path
        path.write_bytes(path.read_bytes() + b" altered")
    with pytest.raises(Exception, match="digest|changed"):
        entry.evaluator()


def test_resolver_reuses_exact_retained_metadata_and_pins_its_representation(tmp_path):
    fixture = signed_edition(tmp_path / "source")
    registry = LocalRegistry(tmp_path / "registry")
    register(registry, fixture)
    store = PublicationStore(tmp_path / "retained")
    kwargs = dict(registry=registry, catalog_session=fixture.session, policy_dag=tn.PolicyDag(),
                  trust=trust(fixture.session.did), metadata_store=store)
    first, = resolve_finance_editions([request(fixture)], **kwargs)
    second, = resolve_finance_editions([request(fixture)], catalog=first.catalog, **kwargs)
    assert first.selection.binding == second.selection.binding
    assert store.get(fixture.record.id) == fixture.record.object.wire.encode()
    assert store.get(fixture.revision.id) == fixture.revision.object.wire.encode()
    import json
    (fixture.bundle / "edition.tn").write_text(json.dumps(fixture.record.object.envelope, indent=2))
    with pytest.raises(Exception, match="retained"):
        resolve_finance_editions([request(fixture)], **kwargs)
