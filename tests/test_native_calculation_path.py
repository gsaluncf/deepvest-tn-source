"""Deterministic native TN, real OPA, Polars and actual-engine composition."""

import io
import os
import threading
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
import tn
import yaml
from test_publications import POLICY
from tn._native.btn import PublisherState
from tn.signing import DeviceKey
from tn_polars import CompiledPolicy
from tn_polars.registry import LocalRegistry
from tn_polars.release import PolicyDenied

from deepvest_tn.deepvest_engine import EngineDeepVestPort
from deepvest_tn.feed_server import FeedServerConfig, create_feed_server
from deepvest_tn.getdata import FeedFetchResponseError, GetDataConfig
from deepvest_tn.pipeline import run_governed_calculation_path
from deepvest_tn.polars_stage import prepare_wealth_calculation
from deepvest_tn.publications import PublicationStore
from deepvest_tn.unity import (
    CatalogTrust,
    originate_policy_revision,
    publish_dataset_edition,
    resolve_finance_editions,
)

POLARS_USE = tn.UseContext("analytics", "portfolio_analysis", "prepare_wealth")
POLARS_RELEASE = tn.UseContext("analytics", "portfolio_analysis", "release_calculation")
DEEPVEST_USE = tn.UseContext("deepvest", "portfolio_analysis", "calculate")
DEEPVEST_RELEASE = tn.UseContext("deepvest", "portfolio_analysis", "release_calculation")
METADATA_USE = tn.UseContext("catalog", "metadata_resolution", "read")
OPERATIONS = ["load", "join", "select", "sort", "release"]


class CountingPort:
    def __init__(self):
        self.requests = []

    def calculate(self, request):
        self.requests.append(request)
        return EngineDeepVestPort().calculate(request)


def configured_sessions(root):
    """Preprovision real test BTN material for three distinct native identities."""
    materials = {}
    for group in ["finance", "policy_revision", "dataset_edition", "tn.agents"]:
        state = PublisherState()
        materials[group] = state.mint(), state.to_bytes()
    index = os.urandom(32)
    paths = []
    for name in ["source", "polars", "deepvest"]:
        directory = root / name
        keys = directory / ".tn" / "keys"
        keys.mkdir(parents=True)
        policy_dir = directory / ".tn" / "config"
        policy_dir.mkdir()
        (policy_dir / "agents.md").write_text(POLICY)
        device = DeviceKey.generate()
        (keys / "local.private").write_bytes(device.private_bytes)
        (keys / "index_master.key").write_bytes(index)
        groups = {}
        for group, (kit, state) in materials.items():
            (keys / f"{group}.btn.mykit").write_bytes(kit)
            (keys / f"{group}.btn.state").write_bytes(state)
            groups[group] = {"policy": "private", "cipher": "btn", "index_epoch": 0,
                             "recipients": [{"recipient_identity": device.did}]}
        config = {
            "ceremony": {"id": "cer_calculation_test", "mode": "local", "cipher": "btn",
                         "protocol_events_location": "main_log"},
            "keystore": {"path": "./.tn/keys"},
            "device": {"device_identity": device.did}, "logs": {"path": "./test.log"},
            "groups": groups, "public_fields": [], "default_policy": "private", "fields": {},
            "llm_classifier": {"enabled": False, "provider": "", "model": ""},
        }
        path = directory / "tn.yaml"
        path.write_text(yaml.safe_dump(config))
        paths.append(path)
    return paths


@contextmanager
def serving(store, root):
    server = create_feed_server(FeedServerConfig(root, io.StringIO(), 0, publication_store=store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield GetDataConfig(f"http://{host}:{port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def setup(tmp_path):
    with ExitStack() as stack:
        source, polars, deepvest = [stack.enter_context(tn.Session.from_config(path))
                                   for path in configured_sessions(tmp_path / "sessions")]
        trust = CatalogTrust(
            lambda ctx: ctx.writer == source.did,
            lambda revision, parent: revision.writer == source.did,
            lambda record: record.writer == source.did,
            METADATA_USE,
        )
        origin_dag = tn.PolicyDag()
        dag = tn.PolicyDag()
        store = PublicationStore(tmp_path / "store")
        registry = LocalRegistry(tmp_path / "registry")
        administration = source.policy("portfolio.feed.data.v1")
        built = []

        def publish(name, rows, *, operations=OPERATIONS):
            text = POLICY.replace("Prepare approved portfolio analysis.", f"Analyze {name}.")
            revision, policy = originate_policy_revision(
                session=source, policy_dag=origin_dag, markdown=text, policy_id=name,
                object_type="portfolio.feed.data.v1", scope="finance",
                administration_contract=administration, trust=trust,
            )
            result = publish_dataset_edition(
                dataset=name, edition="2026-09-09", rows=rows,
                compiled_policy=CompiledPolicy.from_dict({
                    "schema": "tn.polars-policy.v1", "policy_ref": policy.policy_ref,
                    "fields": {key: "confidential" for key in rows[0]},
                    "release": {"public": {"allow_aggregates": False, "min_group_size": 2}},
                }),
                session=source, policy=policy, policy_dag=origin_dag, revisions=[revision],
                eligible_uses=[POLARS_USE, DEEPVEST_USE], grant_ref=f"grant:{name}",
                administration_contract=administration, trust=trust,
                publish_use=tn.UseContext("source", "portfolio_analysis", "publish"),
                publish_decide=lambda _: True, destination="analytics", publication_store=store,
                output_dir=tmp_path / name, registry=registry, allowed_operations=operations,
                audiences={"deepvest": "confidential", "reporting": "confidential"},
            )
            built.append(result)
            return result

        def editions(*, deny_side=None, deny_operation=None):
            def ops(side):
                return [item for item in OPERATIONS if side != deny_side or item != deny_operation]
            wealth = publish("wealth.observations", [
                {"observation_id": 4, "wealth": 110.0}, {"observation_id": 1, "wealth": 100.0},
                {"observation_id": 3, "wealth": 90.0}, {"observation_id": 2, "wealth": 120.0},
            ], operations=ops("wealth"))
            calendar = publish("valuation.calendar", [
                {"observation_id": day, "observed_at": f"2026-09-0{day}T16:00:00+00:00"}
                for day in range(1, 5)
            ], operations=ops("calendar"))
            return resolve_finance_editions(
                [wealth.request(POLARS_USE), calendar.request(POLARS_USE)],
                catalog_session=source, policy_dag=dag, trust=trust, registry=registry,
            )

        def run(resolved, *, fetch_store=None, **changes):
            port = changes.pop("deepvest", CountingPort())
            with serving(fetch_store or store, tmp_path) as getdata:
                kwargs = dict(
                    wealth_edition=resolved[0], calendar_edition=resolved[1], getdata=getdata,
                    polars_session=polars, deepvest_session=deepvest, policy_dag=dag,
                    publication_store=store, deepvest=port,
                    polars_use=POLARS_USE, polars_release_use=POLARS_RELEASE,
                    deepvest_use=DEEPVEST_USE, deepvest_release_use=DEEPVEST_RELEASE,
                    decide_source=lambda ctx: ctx.writer == source.did,
                    decide_polars_release=lambda _: True,
                    decide_deepvest_receive=lambda _: True,
                    decide_deepvest_release=lambda _: True, to="reporting",
                )
                kwargs.update(changes)
                return run_governed_calculation_path(**kwargs), port

        yield SimpleNamespace(source=source, polars=polars, deepvest=deepvest, dag=dag,
                              store=store, editions=editions, run=run, built=built)


def test_real_calculation_keeps_two_contracts_exact_parents_and_fetches(setup, monkeypatch):
    editions = setup.editions()
    monkeypatch.setattr(tn, "init", lambda *_a, **_k: pytest.fail("global session switch"))
    result, port = setup.run(editions)
    assert result.execution.result.value == -0.25
    assert port.requests[0].arguments == {"wealth_path": [100.0, 120.0, 90.0, 110.0]}
    assert len(port.requests) == 1
    assert result.execution.source_object_id == result.polars_object.id
    assert not hasattr(result, "model_row")
    assert setup.source.did != setup.polars.did != setup.deepvest.did
    assert result.polars_object.writer == setup.polars.did
    assert result.deepvest_object.writer == setup.deepvest.did
    for publication in [result.polars_object, result.deepvest_object]:
        assert setup.store.get(publication.id) == publication.forward()
    view = setup.deepvest.governance(result.deepvest_object)
    assert len(view.governance.policies) == len(view.governance.dataset_bindings) == 2
    assert [parent.object_id for parent in view.governance.sources] == [result.polars_object.id]
    frame = setup.polars.receive(result.polars_object, use=POLARS_USE, groups=["finance"],
                                  decide=lambda _: True)
    assert {parent.object_id for parent in frame.governance.sources} == {
        item.selection.source_object_id for item in editions}
    assert frame.data["rows"][0] == {"observed_at": "2026-09-01T16:00:00+00:00", "wealth": 100.0}
    lineage = frame.data["tn_polars_lineage"]
    assert {item.selection.source_object_id + "::observation_id" for item in editions} <= set(
        lineage["row_selection_sources"])
    assert [fact.request_path for fact in result.feed_fetch_facts] == [
        "/v1/publications/" + item.selection.source_object_id.removeprefix("sha256:")
        for item in editions]
    setup.source.close()
    assert setup.deepvest.governance(result.deepvest_object).object.id == result.deepvest_object.id


@pytest.mark.parametrize("side", ["wealth", "calendar"])
@pytest.mark.parametrize("operation", ["join", "release"])
def test_either_real_opa_policy_can_stop_before_engine(setup, side, operation):
    editions = setup.editions(deny_side=side, deny_operation=operation)
    port = CountingPort()
    with pytest.raises(PolicyDenied, match="operation is not enabled"):
        setup.run(editions, deepvest=port)
    assert port.requests == []


@pytest.mark.parametrize("stage", ["polars", "deepvest"])
def test_wrong_use_refuses_before_engine(setup, stage):
    editions = setup.editions()
    port = CountingPort()
    with pytest.raises(tn.governed.UseDenied):
        setup.run(editions, deepvest=port,
                  **{f"{stage}_use": tn.UseContext(stage, "other_purpose", "calculate")})
    assert port.requests == []


def test_exact_source_route_substitution_refuses_before_engine(setup):
    editions = setup.editions()
    port = CountingPort()

    class SubstitutedStore:
        def get(self, _identity):
            return setup.built[1].source.forward()

    with pytest.raises(FeedFetchResponseError, match="identity"):
        setup.run(editions, deepvest=port, fetch_store=SubstitutedStore())
    assert port.requests == []


def test_ancestor_application_refusal_stops_before_engine(setup):
    editions = setup.editions()
    port = CountingPort()
    with pytest.raises(ValueError, match="ancestor"):
        setup.run(editions, deepvest=port, decide_ancestor=lambda _: False)
    assert port.requests == []


def test_direct_stage_cannot_substitute_source_bytes(setup, monkeypatch):
    wealth, calendar = setup.editions()
    monkeypatch.setattr("tn_polars.tn_boundary.pl.DataFrame",
                        lambda *_a, **_k: pytest.fail("source opened before selection acceptance"))
    with pytest.raises(ValueError, match="selected source identity"):
        prepare_wealth_calculation(
            setup.built[1].source.wire, setup.built[0].source.wire,
            wealth_edition=wealth, calendar_edition=calendar,
            session=setup.polars, use=POLARS_USE, decide_receive=lambda _: True,
        )


def test_missing_retained_ancestor_stops_before_engine(setup):
    editions = setup.editions()
    port = CountingPort()

    class UnavailableAncestorStore:
        def put(self, publication):
            return setup.store.put(publication)

        def get(self, _identity):
            raise LookupError("retained ancestor unavailable")

    with pytest.raises(ValueError, match="ancestor unavailable"):
        setup.run(editions, deepvest=port, publication_store=UnavailableAncestorStore())
    assert port.requests == []
