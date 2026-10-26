from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import polars as pl
import pytest
from tn_polars import PolicyDenied
from tn_polars.admission import SchemaPinViolation
from tn_polars.builder import GovernedPolars
from tn_polars.frame import GovernedLazyFrame
from tn_polars.registry import _load_entry

from deepvest_tn.feed_server import FeedServerConfig, create_feed_server, seed_finance_feed_data
from deepvest_tn.getdata import FeedFetchFact
from deepvest_tn.polars_stage import (
    EmptyPortfolioSelection,
    GovernedPortfolioResult,
    run_governed_polars_stage,
    transform_position_market_values,
)
from deepvest_tn.unity import (
    FEED_NAMES,
    MARKET_PRICES_FEED,
    POLICY_REF,
    POSITIONS_FEED,
    UnityCatalogConfig,
    build_finance_feed_bundles,
)

FIXTURE_ROOT = Path("fixtures/portfolio")
POSITIONS_PATH = FIXTURE_ROOT / "internal_positions.ndjson"
MARKET_PRICES_PATH = FIXTURE_ROOT / "market_prices.ndjson"


@contextmanager
def _running_feed_server(server) -> Iterator[str]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def finance_feed_service(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    data_dir = seed_finance_feed_data(tmp_path / "getdata")
    server = create_feed_server(
        FeedServerConfig(data_dir=data_dir, event_stream=io.StringIO(), port=0)
    )
    with _running_feed_server(server) as feed_url:
        yield feed_url, data_dir


class RecordingUnityTransport:
    def __init__(self, bundles: dict[str, Path], config: UnityCatalogConfig) -> None:
        self.get_paths: list[str] = []
        self.volumes = {
            f"{config.catalog}.{config.schema}.{name.replace('.', '_')}": bundle.resolve().as_uri()
            for name, bundle in bundles.items()
        }

    def request(self, method: str, path: str, payload=None):
        if method == "GET" and path.startswith("/volumes/"):
            self.get_paths.append(path)
            return {"storage_location": self.volumes[path.removeprefix("/volumes/")]}
        raise AssertionError(f"unexpected Unity request: {method} {path} {payload}")


@pytest.fixture(scope="module")
def finance_bundles(tmp_path_factory) -> dict[str, Path]:
    return build_finance_feed_bundles(tmp_path_factory.mktemp("polars-finance-bundles"))


@pytest.fixture()
def unity_transport(monkeypatch, finance_bundles) -> RecordingUnityTransport:
    config = UnityCatalogConfig()
    transport = RecordingUnityTransport(finance_bundles, config)

    def request(_registry, method: str, path: str, payload=None):
        return transport.request(method, path, payload)

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", request)
    return transport


@pytest.fixture()
def admitted_frames(finance_bundles):
    positions_entry = _load_entry(
        POSITIONS_FEED,
        finance_bundles[POSITIONS_FEED],
        described_as="Positions bundle",
    )
    prices_entry = _load_entry(
        MARKET_PRICES_FEED,
        finance_bundles[MARKET_PRICES_FEED],
        described_as="Market-prices bundle",
    )
    positions = GovernedPolars.prepare(positions_entry).load(pl.scan_ndjson(POSITIONS_PATH))
    prices = GovernedPolars.prepare(prices_entry).load(pl.scan_ndjson(MARKET_PRICES_PATH))
    return positions_entry, prices_entry, positions, prices


def test_production_stage_resolves_admits_and_materializes_exact_values(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, _data_dir = finance_feed_service
    result = run_governed_polars_stage(
        UnityCatalogConfig(),
        account_id="acct-001",
        feed_url=feed_url,
    )

    assert isinstance(result, GovernedPortfolioResult)
    assert result.feed_names == FEED_NAMES
    assert result.policy_ref == POLICY_REF
    assert result.tn_governance["policy"] == POLICY_REF
    assert result.feed_fetch_facts == (
        FeedFetchFact(
            event_name="getdata.feed_fetched",
            feed_name=POSITIONS_FEED,
            request_path=f"/v1/feeds/{POSITIONS_FEED}/data",
            http_status=200,
            content_type="application/x-ndjson; charset=utf-8",
        ),
        FeedFetchFact(
            event_name="getdata.feed_fetched",
            feed_name=MARKET_PRICES_FEED,
            request_path=f"/v1/feeds/{MARKET_PRICES_FEED}/data",
            http_status=200,
            content_type="application/x-ndjson; charset=utf-8",
        ),
    )
    assert result.values.columns == [
        "account_id",
        "portfolio_id",
        "symbol",
        "quantity",
        "cost_basis",
        "currency",
        "as_of",
        "last_price",
        "market_source",
        "position_market_value",
    ]
    assert result.values.select("symbol", "position_market_value").to_dicts() == [
        {"symbol": "AAPL", "position_market_value": 28518.75},
        {"symbol": "MSFT", "position_market_value": 40792.0},
        {"symbol": "VTI", "position_market_value": 65398.2},
    ]
    assert unity_transport.get_paths == [
        "/volumes/deepvest.finance.deepvest_internal_positions",
        "/volumes/deepvest.finance.deepvest_market_prices",
    ]


def test_transformation_returns_real_unmaterialized_governed_frame(
    admitted_frames, monkeypatch
) -> None:
    _, _, positions, prices = admitted_frames

    def forbid_collect(*_args, **_kwargs):
        raise AssertionError("the transformation-only function collected its lazy frame")

    monkeypatch.setattr(pl.LazyFrame, "collect", forbid_collect)

    transformed = transform_position_market_values(
        positions,
        prices,
        account_id="acct-001",
    )

    assert isinstance(transformed, GovernedLazyFrame)


def test_native_lineage_and_row_selection_survive_the_full_transformation(
    admitted_frames,
) -> None:
    _, _, positions, prices = admitted_frames

    transformed = transform_position_market_values(
        positions,
        prices,
        account_id="acct-001",
    )

    assert transformed.sources["position_market_value"] == {"quantity", "last_price"}
    assert transformed.row_sources == {"account_id", "symbol", "currency"}
    assert set(transformed.sources) == {
        "account_id",
        "portfolio_id",
        "symbol",
        "quantity",
        "cost_basis",
        "currency",
        "as_of",
        "last_price",
        "market_source",
        "position_market_value",
    }
    assert "currency_right" not in transformed.frame.collect_schema().names()


def test_production_boundary_collects_exactly_once(
    unity_transport: RecordingUnityTransport, finance_feed_service: tuple[str, Path], monkeypatch
) -> None:
    feed_url, _data_dir = finance_feed_service
    original_collect = pl.LazyFrame.collect
    collect_calls = 0

    def recording_collect(frame, *args, **kwargs):
        nonlocal collect_calls
        collect_calls += 1
        return original_collect(frame, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", recording_collect)

    run_governed_polars_stage(
        UnityCatalogConfig(),
        account_id="acct-001",
        feed_url=feed_url,
    )

    assert collect_calls == 1


def test_stage_resolves_both_governance_entries_before_the_first_data_request(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepvest_tn.polars_stage as polars_stage

    feed_url, _data_dir = finance_feed_service
    events: list[str] = []
    real_resolve = polars_stage.resolve_finance_feeds
    real_fetch = polars_stage.fetch_feed

    def resolve(config: UnityCatalogConfig):
        entries = real_resolve(config)
        events.append("unity_resolved")
        return entries

    def fetch(config, feed_name: str):
        assert events[0] == "unity_resolved"
        events.append(feed_name)
        return real_fetch(config, feed_name)

    monkeypatch.setattr(polars_stage, "resolve_finance_feeds", resolve)
    monkeypatch.setattr(polars_stage, "fetch_feed", fetch)

    run_governed_polars_stage(
        UnityCatalogConfig(), account_id="acct-001", feed_url=feed_url
    )

    assert events == ["unity_resolved", POSITIONS_FEED, MARKET_PRICES_FEED]


def test_unity_resolution_failure_makes_zero_data_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deepvest_tn.polars_stage as polars_stage

    calls: list[str] = []

    def unavailable(_config: UnityCatalogConfig):
        raise ConnectionError("Unity unavailable")

    def fetch(_config, feed_name: str):
        calls.append(feed_name)
        raise AssertionError("data request must follow successful Unity resolution")

    monkeypatch.setattr(polars_stage, "resolve_finance_feeds", unavailable)
    monkeypatch.setattr(polars_stage, "fetch_feed", fetch)

    with pytest.raises(ConnectionError, match="Unity unavailable"):
        run_governed_polars_stage(
            UnityCatalogConfig(), account_id="acct-001", feed_url="http://127.0.0.1:18082"
        )

    assert calls == []


def test_positions_route_serving_prices_is_rejected_by_matching_entry_admission(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, data_dir = finance_feed_service
    (data_dir / "internal_positions.ndjson").write_bytes(MARKET_PRICES_PATH.read_bytes())

    with pytest.raises(SchemaPinViolation, match="symbol|account_id"):
        run_governed_polars_stage(
            UnityCatalogConfig(), account_id="acct-001", feed_url=feed_url
        )


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_schema_shape_mutation_fails_during_admission(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
    mutation: str,
) -> None:
    feed_url, data_dir = finance_feed_service
    positions = pl.read_ndjson(POSITIONS_PATH)
    if mutation == "extra":
        positions = positions.with_columns(pl.lit("unexpected").alias("new_vendor_field"))
        expected = "new_vendor_field"
    else:
        positions = positions.drop("cost_basis")
        expected = "cost_basis"
    mutated_path = data_dir / "internal_positions.ndjson"
    positions.write_ndjson(mutated_path)

    with pytest.raises(SchemaPinViolation, match=expected):
        run_governed_polars_stage(
            UnityCatalogConfig(),
            account_id="acct-001",
            feed_url=feed_url,
        )


def test_dtype_mutation_fails_during_admission(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, data_dir = finance_feed_service
    mutated_path = data_dir / "internal_positions.ndjson"
    (
        pl.read_ndjson(POSITIONS_PATH)
        .with_columns(pl.col("quantity").cast(pl.String))
        .write_ndjson(mutated_path)
    )

    with pytest.raises(SchemaPinViolation, match="quantity arrived as String"):
        run_governed_polars_stage(
            UnityCatalogConfig(),
            account_id="acct-001",
            feed_url=feed_url,
        )


def test_cross_policy_join_is_refused_by_existing_governed_frame(admitted_frames) -> None:
    _, prices_entry, positions, _ = admitted_frames
    other_policy = replace(prices_entry.policy, policy_ref="tn://policy/unrelated-market-feed@1")
    other_governance = dict(prices_entry.tn_governance)
    other_governance["policy"] = other_policy.policy_ref
    other_entry = replace(
        prices_entry,
        policy=other_policy,
        tn_governance=other_governance,
    )
    other_prices = GovernedPolars.prepare(other_entry).load(pl.scan_ndjson(MARKET_PRICES_PATH))

    with pytest.raises(PolicyDenied, match="different TN policies"):
        transform_position_market_values(
            positions,
            other_prices,
            account_id="acct-001",
        )


def test_nonexistent_account_fails_instead_of_returning_empty_values(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, _data_dir = finance_feed_service
    with pytest.raises(EmptyPortfolioSelection, match="acct-missing"):
        run_governed_polars_stage(
            UnityCatalogConfig(),
            account_id="acct-missing",
            feed_url=feed_url,
        )


def test_result_is_immutable_and_owns_copies_of_native_facts(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, _data_dir = finance_feed_service
    result = run_governed_polars_stage(
        UnityCatalogConfig(),
        account_id="acct-001",
        feed_url=feed_url,
    )

    with pytest.raises(FrozenInstanceError):
        result.policy_ref = "tn://policy/replaced@1"
    with pytest.raises(FrozenInstanceError):
        result.feed_fetch_facts = ()
    assert all(not hasattr(fact, "body") for fact in result.feed_fetch_facts)
    with pytest.raises(TypeError):
        result.sources["position_market_value"] = frozenset({"replacement"})
    with pytest.raises(AttributeError):
        result.sources["position_market_value"].add("replacement")
    with pytest.raises(AttributeError):
        result.row_sources.add("replacement")
    with pytest.raises(TypeError):
        result.tn_governance["policy"] = "tn://policy/replaced@1"

    external_values = result.values
    market_value_index = external_values.get_column_index("position_market_value")
    external_values.replace_column(
        market_value_index,
        pl.Series("position_market_value", [0.0, 0.0, 0.0]),
    )

    assert result.sources["position_market_value"] == {"quantity", "last_price"}
    assert result.row_sources == {"account_id", "symbol", "currency"}
    assert result.values.get_column("position_market_value").to_list() == [
        28518.75,
        40792.0,
        65398.2,
    ]


@pytest.mark.parametrize(
    "facts",
    [
        (),
        (
            FeedFetchFact(
                "getdata.feed_fetched", MARKET_PRICES_FEED, "/wrong", 200,
                "application/x-ndjson",
            ),
            FeedFetchFact(
                "getdata.feed_fetched", POSITIONS_FEED, "/wrong", 200,
                "application/x-ndjson",
            ),
        ),
    ],
)
def test_result_rejects_fetch_facts_that_are_not_the_exact_ordered_feed_pair(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
    facts: tuple[FeedFetchFact, ...],
) -> None:
    feed_url, _data_dir = finance_feed_service
    result = run_governed_polars_stage(
        UnityCatalogConfig(), account_id="acct-001", feed_url=feed_url
    )

    with pytest.raises(ValueError, match="fetch facts"):
        replace(result, feed_fetch_facts=facts)


def test_nested_governance_is_accepted_and_recursively_read_only(
    tmp_path: Path,
    monkeypatch,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, _data_dir = finance_feed_service
    bundles = build_finance_feed_bundles(tmp_path / "nested-governance-bundles")
    nested = {
        "limits": {"maximum_positions": 50},
        "audiences": [
            "portfolio-desk",
            {"name": "risk-review", "readers": ["finance-agent", "auditor"]},
        ],
    }
    for bundle in bundles.values():
        governance_path = bundle / "tn_governance.json"
        governance = json.loads(governance_path.read_text(encoding="utf-8"))
        governance["approval_context"] = nested
        governance_path.write_text(json.dumps(governance), encoding="utf-8")

    config = UnityCatalogConfig()
    transport = RecordingUnityTransport(bundles, config)

    def request(_registry, method: str, path: str, payload=None):
        return transport.request(method, path, payload)

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", request)
    result = run_governed_polars_stage(
        config,
        account_id="acct-001",
        feed_url=feed_url,
    )
    approval_context = result.tn_governance["approval_context"]

    assert approval_context["audiences"][1]["readers"] == (
        "finance-agent",
        "auditor",
    )
    with pytest.raises(TypeError):
        approval_context["limits"]["maximum_positions"] = 0
    with pytest.raises(AttributeError):
        approval_context["audiences"].append("replacement")
    with pytest.raises(TypeError):
        approval_context["audiences"][1]["name"] = "replacement"


def test_mutable_dataframe_backing_is_not_exposed(
    unity_transport: RecordingUnityTransport,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, _data_dir = finance_feed_service
    result = run_governed_polars_stage(
        UnityCatalogConfig(),
        account_id="acct-001",
        feed_url=feed_url,
    )
    replacement = pl.Series("position_market_value", [0.0, 0.0, 0.0])

    with pytest.raises(AttributeError):
        result._values.replace_column(
            result._values.get_column_index("position_market_value"),
            replacement,
        )

    first = result.values
    second = result.values
    first.replace_column(
        first.get_column_index("position_market_value"),
        replacement,
    )

    assert first is not second
    assert second.get_column("position_market_value").to_list() == [
        28518.75,
        40792.0,
        65398.2,
    ]


def test_production_stage_does_not_cross_forbidden_boundaries(
    unity_transport: RecordingUnityTransport,
    monkeypatch,
    finance_feed_service: tuple[str, Path],
) -> None:
    feed_url, _data_dir = finance_feed_service
    import tn
    from tn_llama_client.clients import GovernedLlamaClient, TnOutputReader
    from tn_polars.registry import LocalRegistry
    from tn_polars.tn_envelopes import DeskLedger

    import deepvest_tn.deepvest as deepvest_boundary
    import deepvest_tn.polars_stage as polars_stage

    calls: list[str] = []

    def deny(name: str):
        def denied(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"Polars preparation invoked forbidden boundary {name}")

        return denied

    monkeypatch.setattr(LocalRegistry, "__init__", deny("LocalRegistry.__init__"))
    monkeypatch.setattr(LocalRegistry, "get", deny("LocalRegistry.get"))
    monkeypatch.setattr(GovernedLazyFrame, "release", deny("GovernedLazyFrame.release"))
    monkeypatch.setattr(
        GovernedLazyFrame,
        "release_delta",
        deny("GovernedLazyFrame.release_delta"),
    )
    monkeypatch.setattr(DeskLedger, "open", deny("DeskLedger.open"))
    monkeypatch.setattr(DeskLedger, "log", deny("DeskLedger.log"))
    monkeypatch.setattr(tn, "init", deny("tn.init"))
    monkeypatch.setattr(tn, "log", deny("tn.log"))
    monkeypatch.setattr(tn.TN, "log", deny("TN.log"))
    monkeypatch.setattr(
        deepvest_boundary,
        "build_deepvest_request",
        deny("build_deepvest_request"),
    )
    monkeypatch.setattr(
        deepvest_boundary,
        "execute_deepvest",
        deny("execute_deepvest"),
    )
    monkeypatch.setattr(
        deepvest_boundary.StubDeepVest,
        "calculate",
        deny("StubDeepVest.calculate"),
    )
    monkeypatch.setattr(
        polars_stage,
        "execute_deepvest",
        deny("polars_stage.execute_deepvest"),
        raising=False,
    )
    monkeypatch.setattr(
        GovernedLlamaClient,
        "submit",
        deny("GovernedLlamaClient.submit"),
    )
    monkeypatch.setattr(
        GovernedLlamaClient,
        "submit_request",
        deny("GovernedLlamaClient.submit_request"),
    )
    monkeypatch.setattr(TnOutputReader, "open", deny("TnOutputReader.open"))

    result = run_governed_polars_stage(
        UnityCatalogConfig(),
        account_id="acct-001",
        feed_url=feed_url,
    )

    assert result.values.height == 3
    assert calls == []
