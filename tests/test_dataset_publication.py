import pytest
import tn
import json
import polars as pl
from tn_polars import CompiledPolicy
from tn_polars.builder import GovernedPolars
from tn_polars.release import PolicyDenied
from tn_polars.registry import LocalRegistry

from deepvest_tn.publications import PublicationStore
from deepvest_tn.unity import (
    CatalogTrust, originate_policy_revision, publish_dataset_edition, resolve_finance_editions,
)
from governed_edition_fixtures import BUSINESS_USE, METADATA_USE
from test_publications import POLICY


def setup(tmp_path, scope="finance"):
    session = tn.Session(POLICY, groups=["finance", "policy_revision", "dataset_edition"])
    trust = CatalogTrust(
        lambda context: context.writer == session.did,
        lambda revision, parent: revision.writer == session.did,
        lambda edition: edition.writer == session.did,
        METADATA_USE,
    )
    dag = tn.PolicyDag()
    administration = session.policy("portfolio.feed.data.v1")
    revision, policy = originate_policy_revision(
        session=session, policy_dag=dag, markdown=POLICY, policy_id="agents.md",
        object_type="portfolio.feed.data.v1", scope=scope,
        administration_contract=administration, trust=trust,
    )
    return {
        "dataset": "wealth.measurements", "edition": "2026-09-09",
        "rows": [{"observation_id": 1, "wealth": 100.0}],
        "compiled_policy": CompiledPolicy.from_dict({
            "schema": "tn.polars-policy.v1", "policy_ref": policy.policy_ref,
            "fields": {"observation_id": "internal", "wealth": "confidential"},
            "release": {"public": {"allow_aggregates": True, "min_group_size": 2}},
        }),
        "session": session, "policy": policy, "policy_dag": dag, "revisions": [revision],
        "eligible_uses": [BUSINESS_USE], "grant_ref": "grant:wealth",
        "administration_contract": administration, "trust": trust,
        "publish_use": tn.UseContext("wealth_source", "portfolio_analysis", "publish"),
        "publish_decide": lambda context: context.writer == session.did,
        "destination": "analytics", "publication_store": PublicationStore(tmp_path / "store"),
        "output_dir": tmp_path / "bundle", "registry": LocalRegistry(tmp_path / "registry"),
        "allowed_operations": ["load", "join", "release"],
    }


def test_publication_pins_final_release_and_retains_native_creation_and_metadata(tmp_path):
    kwargs = setup(tmp_path)
    published = publish_dataset_edition(**kwargs)
    assert published.creation.id != published.source.id
    assert published.record.source_object_id == published.source.id
    for object in [published.creation, published.source, published.record.object, kwargs["revisions"][0].object]:
        assert kwargs["publication_store"].get(object.id) == object.wire.encode()
    resolved, = resolve_finance_editions(
        [published.request(BUSINESS_USE)], registry=kwargs["registry"],
        catalog_session=kwargs["session"], policy_dag=tn.PolicyDag(), trust=kwargs["trust"],
    )
    data = kwargs["session"].receive(
        published.source, use=BUSINESS_USE, groups=["finance"],
        selection=resolved.selection, decide=lambda context: True,
    )
    assert list(data.data["rows"])[0]["wealth"] == 100.0
    assert data.dataset_bindings == [resolved.selection.binding]


def test_refused_source_release_does_not_publish_a_dataset_record(tmp_path):
    kwargs = setup(tmp_path)
    kwargs["publish_decide"] = lambda context: False
    with pytest.raises(tn.governed.UseDenied):
        publish_dataset_edition(**kwargs)
    assert not (tmp_path / "bundle" / "bundle.json").exists()
    assert not (tmp_path / "store" / "objects").exists()


def test_business_group_and_contract_scope_are_independent(tmp_path):
    kwargs = setup(tmp_path, scope="wealth.measurements")
    published = publish_dataset_edition(**kwargs)
    assert published.record.source_groups == ["finance"]
    assert published.record.contracts[0].scope == "wealth.measurements"
    resolved, = resolve_finance_editions(
        [published.request(BUSINESS_USE)], registry=kwargs["registry"],
        catalog_session=kwargs["session"], policy_dag=tn.PolicyDag(), trust=kwargs["trust"],
    )
    received = kwargs["session"].receive(
        published.source, use=BUSINESS_USE, groups=["finance"],
        selection=resolved.selection, decide=lambda context: True,
    )
    assert received.dataset_bindings[0].contracts[0].scope == "wealth.measurements"


def test_two_editions_can_reuse_one_exact_accepted_revision(tmp_path):
    kwargs = setup(tmp_path)
    first = publish_dataset_edition(**kwargs)
    kwargs["edition"] = "later"
    kwargs["output_dir"] = tmp_path / "later"
    second = publish_dataset_edition(**kwargs)
    resolved = resolve_finance_editions(
        [first.request(BUSINESS_USE), second.request(BUSINESS_USE)],
        registry=kwargs["registry"], catalog_session=kwargs["session"],
        policy_dag=kwargs["policy_dag"], trust=kwargs["trust"],
    )
    assert resolved[0].selection.record.contracts == resolved[1].selection.record.contracts
    assert resolved[0].selection.source_object_id != resolved[1].selection.source_object_id


def test_evaluator_artifact_explicitly_permits_the_named_release_destination(tmp_path):
    kwargs = setup(tmp_path)
    kwargs["audiences"] = {"deepvest": "confidential"}
    published = publish_dataset_edition(**kwargs)
    resolved, = resolve_finance_editions(
        [published.request(BUSINESS_USE)], registry=kwargs["registry"],
        catalog_session=kwargs["session"], policy_dag=tn.PolicyDag(), trust=kwargs["trust"],
    )
    entry = resolved.entries[kwargs["policy"].revision_id]
    assert entry.evaluator().decide({
        "operation": "release", "audience": "deepvest", "policy_ref": entry.policy.policy_ref,
        "columns": [], "row_selection_sources": [],
    }).allow


@pytest.mark.parametrize("mismatch", ["reference", "columns", "row_schema"])
def test_invalid_compiled_policy_refuses_before_source_creation(tmp_path, monkeypatch, mismatch):
    kwargs = setup(tmp_path)
    document = kwargs["compiled_policy"].to_dict()
    if mismatch == "reference":
        document["policy_ref"] = "tn://unaccepted"
    elif mismatch == "columns":
        document["fields"]["undeclared"] = "public"
    else:
        kwargs["rows"].append({"observation_id": 2, "wealth": 42.0, "extra": "hidden"})
    kwargs["compiled_policy"] = CompiledPolicy.from_dict(document)
    with pytest.raises(ValueError, match="reference|fields"):
        publish_dataset_edition(**kwargs)
    assert not (tmp_path / "store" / "objects").exists()
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("allow,minimum,accepted", [(True, 2, True), (False, 1, False), (True, 3, False)])
def test_nonempty_rules_survive_native_publication_resolution_and_polars_release(tmp_path, allow, minimum, accepted):
    kwargs = setup(tmp_path)
    kwargs["rows"] = [{"observation_id": 1, "wealth": 100.0}, {"observation_id": 2, "wealth": 120.0}]
    document = kwargs["compiled_policy"].to_dict()
    document["fields"] = {name: "public" for name in document["fields"]}
    document["release"] = {"public": {"allow_aggregates": allow, "min_group_size": minimum}}
    kwargs["compiled_policy"] = CompiledPolicy.from_dict(document)
    kwargs["allowed_operations"] = ["load", "select", "release"]
    published = publish_dataset_edition(**kwargs)
    resolved, = resolve_finance_editions(
        [published.request(BUSINESS_USE)], registry=kwargs["registry"],
        catalog_session=kwargs["session"], policy_dag=tn.PolicyDag(), trust=kwargs["trust"],
    )
    entry = resolved.entries[kwargs["policy"].revision_id]
    assert json.loads(entry.compiled_policy_bytes)["release"] == document["release"]
    frame = GovernedPolars.load_from_tn(published.source.wire, session=kwargs["session"],
        use=BUSINESS_USE, selection=resolved.selection, policies=resolved.entries, decide=lambda _: True)
    result = frame.select(pl.col("wealth").mean())
    calls = []
    def release():
        return result.release_tn(use=BUSINESS_USE, to="public", decide=lambda _: calls.append(True) or True)
    if accepted:
        output = release()
        data = kwargs["session"].receive(output, use=BUSINESS_USE, groups=["finance"], decide=lambda _: True)
        assert data.state.groups["finance"]["rows"] == [{"wealth": 110.0}]
        assert len(data.dataset_bindings) == 1
    else:
        with pytest.raises(PolicyDenied, match="aggregate"):
            release()
        assert calls == []
