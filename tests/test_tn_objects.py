from __future__ import annotations

import io
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest
import tn
from tn.identity import Identity
from tn_polars.registry import LocalRegistry

from tn_governed.calculation import StubCalculationPort, execute_sealed_calculation as execute_calculation
from tn_governed.feed_server import FeedServerConfig, create_feed_server, seed_finance_feed_data
from tn_governed.getdata import GetDataConfig, fetch_governed_feed
from tn_governed.llama import adapt_finance_source_record
from tn_governed.pipeline import run_processing_path
from tn_governed.polars_stage import run_governed_polars_tn_stage
from tn_governed.tn_flow import (
    build_reference_flow,
    provision_reference_path,
    publish_calculation_object_for_model,
)
from tn_governed.tn_objects import open_governed_object, seal_governed_object
from tn_governed.unity import (
    FEED_NAMES,
    MARKET_PRICES_FEED,
    POSITIONS_FEED,
    build_finance_feed_bundles,
)

GOVERNANCE = {
    "instruction": "Calculate the governed internal portfolio report.",
    "use_for": "Internal portfolio analysis and reporting.",
    "do_not_use_for": "Trade execution or external redistribution.",
    "consequences": "Unauthorized disclosure of confidential account information.",
    "on_violation_or_error": "stop",
    "policy": "tn://policy/calculation-portfolio-report@1",
}


@contextmanager
def _running_server(server) -> Iterator[str]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def isolated_tn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TN_NO_LINK", "1")
    monkeypatch.setenv("TN_NO_STDOUT", "1")
    monkeypatch.setenv("TN_AUTOINIT_QUIET", "1")
    with suppress(Exception):
        tn.flush_and_close()
    yield
    with suppress(Exception):
        tn.flush_and_close()


def _endpoint(root: Path, name: str, business_fields: list[str]) -> tn.TN:
    identity_path = root / f"{name}-identity" / "identity.json"
    identity_path.parent.mkdir(parents=True)
    identity = Identity.create_new()
    identity.ensure_written(identity_path)
    endpoint = tn.init(
        root / name / "tn.yaml",
        profile="transaction",
        identity=identity,
        cipher="btn",
        link=False,
        stdout=False,
    )
    tn.ensure_group(endpoint.cfg, "finance", cipher="btn", fields=business_fields)
    return endpoint


def _enrol(publisher: tn.TN, recipient: tn.TN, package_path: Path) -> None:
    publisher = tn.init(publisher.yaml_path, link=False, stdout=False)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package = tn.admin.add_agent_runtime(
        recipient.cfg.device.device_identity,
        groups=["finance"],
        out_path=package_path,
        label=f"{publisher.name}-to-{recipient.name}",
    )
    recipient = tn.init(recipient.yaml_path, link=False, stdout=False)
    receipt = recipient.absorb(package)
    assert receipt.status == "enrolment_applied"


def test_governed_tn_object_crosses_two_keyed_processing_boundaries(tmp_path: Path) -> None:
    source = _endpoint(tmp_path, "source", ["feed_name", "rows"])
    polars = _endpoint(tmp_path, "polars", ["portfolio_values"])
    calculation = _endpoint(tmp_path, "calculation", ["calculation_result"])
    _enrol(source, polars, tmp_path / "packages" / "source-to-polars.tnpkg")
    _enrol(polars, calculation, tmp_path / "packages" / "polars-to-calculation.tnpkg")

    source_object = seal_governed_object(
        source.yaml_path,
        "portfolio.source.v1",
        business_group="finance",
        business_fields={
            "feed_name": "portfolio.internal.positions",
            "rows": [{"account_id": "acct-001", "quantity": 125}],
        },
        governance=GOVERNANCE,
    )
    opened_source = open_governed_object(
        polars.yaml_path,
        source_object,
        expected_type="portfolio.source.v1",
        business_group="finance",
    )
    assert opened_source.business == {
        "feed_name": "portfolio.internal.positions",
        "rows": [{"account_id": "acct-001", "quantity": 125}],
    }
    assert opened_source.governance == GOVERNANCE

    polars_object = seal_governed_object(
        polars.yaml_path,
        "portfolio.polars.derived.v1",
        business_group="finance",
        business_fields={
            "portfolio_values": [
                {
                    "account_id": "acct-001",
                    "position_market_value": 28518.75,
                }
            ]
        },
        governance=opened_source.governance,
    )
    opened_polars = open_governed_object(
        calculation.yaml_path,
        polars_object,
        expected_type="portfolio.polars.derived.v1",
        business_group="finance",
    )

    assert opened_polars.business == {
        "portfolio_values": [
            {"account_id": "acct-001", "position_market_value": 28518.75}
        ]
    }
    assert opened_polars.governance == GOVERNANCE
    assert set(source_object) >= {"finance", "tn.agents", "signature", "row_hash"}
    assert set(polars_object) >= {"finance", "tn.agents", "signature", "row_hash"}
    assert "acct-001" not in str(source_object)
    assert "Internal portfolio" not in str(source_object)


def test_calculation_opens_polars_tn_object_and_seals_its_result_for_llama(
    tmp_path: Path,
) -> None:
    polars = _endpoint(tmp_path, "polars", ["portfolio_values"])
    calculation = _endpoint(
        tmp_path,
        "calculation",
        ["portfolio_values", "calculation_result"],
    )
    llama = _endpoint(tmp_path, "llama", ["unused"])
    _enrol(polars, calculation, tmp_path / "packages" / "polars-to-calculation.tnpkg")
    _enrol(calculation, llama, tmp_path / "packages" / "calculation-to-llama.tnpkg")

    portfolio_values = [
        {
            "portfolio_id": "portfolio-001",
            "symbol": "AAPL",
            "quantity": 125,
            "cost_basis": 182.4,
            "last_price": 228.15,
            "position_market_value": 28518.75,
            "currency": "USD",
            "as_of": "2026-09-01T16:00:00Z",
            "market_source": "market-feed",
        }
    ]
    polars_object = seal_governed_object(
        polars.yaml_path,
        "portfolio.polars.derived.v1",
        business_group="finance",
        business_fields={"portfolio_values": portfolio_values},
        governance=GOVERNANCE,
    )

    stub = StubCalculationPort()
    execution, calculation_object = execute_calculation(
        stub,
        polars_object,
        recipient_config_path=calculation.yaml_path,
        publisher_config_path=calculation.yaml_path,
    )
    opened = open_governed_object(
        llama.yaml_path,
        calculation_object,
        expected_type="portfolio.calculation.calculated",
        business_group="finance",
    )

    assert stub.invocations == (execution.request,)
    assert execution.result.provider == "stub"
    assert execution.result.live is False
    assert opened.business["portfolio_values"] == portfolio_values
    assert opened.business["calculation_result"] == {
        "calculation_id": "stub-portfolio-risk-001",
        "analysis_type": "portfolio_risk",
        "metrics": [
            {"name": "value_at_risk_95", "value": 12500.0, "unit": "USD"},
            {"name": "expected_shortfall_95", "value": 16800.0, "unit": "USD"},
            {"name": "maximum_drawdown", "value": -0.18, "unit": "ratio"},
        ],
        "provider": "stub",
        "live": False,
    }
    assert opened.governance == GOVERNANCE


def test_polars_opens_source_tn_objects_and_reseals_real_derived_values(
    tmp_path: Path,
) -> None:
    bundles = build_finance_feed_bundles(tmp_path / "bundles")
    registry = LocalRegistry(tmp_path / "registry")
    for feed_name, bundle in bundles.items():
        registry.register_feed(feed_name, bundle)
    entries = {feed_name: registry.get(feed_name) for feed_name in FEED_NAMES}
    governance = entries[POSITIONS_FEED].tn_governance

    source = _endpoint(tmp_path, "source", ["feed_name", "rows"])
    polars = _endpoint(
        tmp_path,
        "polars",
        [
            "portfolio_values",
            "native_feed_names",
            "native_column_sources",
            "native_row_sources",
        ],
    )
    calculation = _endpoint(tmp_path, "calculation", ["unused"])
    _enrol(source, polars, tmp_path / "packages" / "source-to-polars.tnpkg")
    _enrol(polars, calculation, tmp_path / "packages" / "polars-to-calculation.tnpkg")

    source_objects = {}
    fixture_files = {
        POSITIONS_FEED: Path("fixtures/portfolio/internal_positions.ndjson"),
        MARKET_PRICES_FEED: Path("fixtures/portfolio/market_prices.ndjson"),
    }
    for feed_name in FEED_NAMES:
        rows = [json.loads(line) for line in fixture_files[feed_name].read_text().splitlines()]
        source_objects[feed_name] = seal_governed_object(
            source.yaml_path,
            "portfolio.feed.data.v1",
            business_group="finance",
            business_fields={"feed_name": feed_name, "rows": rows},
            governance=governance,
        )

    portfolio = run_governed_polars_tn_stage(
        entries,
        source_objects,
        endpoint_config_path=polars.yaml_path,
        account_id="acct-001",
    )
    opened = open_governed_object(
        calculation.yaml_path,
        portfolio.sealed_object,
        expected_type="portfolio.polars.derived.v1",
        business_group="finance",
    )

    assert portfolio.values["position_market_value"].to_list() == [
        28518.75,
        40792.0,
        65398.2,
    ]
    assert opened.business["portfolio_values"] == portfolio.values.to_dicts()
    assert opened.business["native_feed_names"] == list(FEED_NAMES)
    assert opened.business["native_column_sources"]["position_market_value"] == [
        "last_price",
        "quantity",
    ]
    assert opened.business["native_row_sources"] == ["account_id", "currency", "symbol"]
    assert opened.governance == governance


def test_http_data_service_returns_a_tn_object_with_unity_governance(tmp_path: Path) -> None:
    source = _endpoint(tmp_path, "source", ["feed_name", "rows"])
    polars = _endpoint(tmp_path, "polars", ["portfolio_values"])
    _enrol(source, polars, tmp_path / "packages" / "source-to-polars.tnpkg")
    data_dir = seed_finance_feed_data(tmp_path / "data")
    server = create_feed_server(
        FeedServerConfig(
            data_dir=data_dir,
            event_stream=io.StringIO(),
            port=0,
            publisher_config_path=source.yaml_path,
            governance=GOVERNANCE,
        )
    )

    with _running_server(server) as base_url:
        fetched = fetch_governed_feed(
            GetDataConfig(base_url=base_url),
            POSITIONS_FEED,
        )
    opened = open_governed_object(
        polars.yaml_path,
        fetched.body,
        expected_type="portfolio.feed.data.v1",
        business_group="finance",
    )

    assert fetched.fact.content_type == "application/vnd.tn+json; charset=utf-8"
    assert opened.business["feed_name"] == POSITIONS_FEED
    assert opened.business["rows"][0]["account_id"] == "acct-001"
    assert opened.governance == GOVERNANCE
    assert b"acct-001" not in fetched.body


def test_processing_path_runs_unity_http_polars_and_calculation_as_tn_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = build_finance_feed_bundles(tmp_path / "bundles")
    unity_locations = {
        f"portfolio.finance.{name.replace('.', '_')}": bundle.resolve().as_uri()
        for name, bundle in bundles.items()
    }

    def unity_request(_registry, method: str, path: str, payload=None):
        assert method == "GET"
        assert payload is None
        return {"storage_location": unity_locations[path.removeprefix("/volumes/")]}

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", unity_request)
    governance = json.loads(Path("fixtures/portfolio/tn_governance.json").read_text())
    source = _endpoint(tmp_path, "source", ["feed_name", "rows"])
    polars = _endpoint(
        tmp_path,
        "polars",
        [
            "portfolio_values",
            "native_feed_names",
            "native_column_sources",
            "native_row_sources",
        ],
    )
    calculation = _endpoint(
        tmp_path,
        "calculation",
        ["portfolio_values", "calculation_result"],
    )
    llama = _endpoint(tmp_path, "llama", ["unused"])
    _enrol(source, polars, tmp_path / "packages" / "source-to-polars.tnpkg")
    _enrol(polars, calculation, tmp_path / "packages" / "polars-to-calculation.tnpkg")
    _enrol(calculation, llama, tmp_path / "packages" / "calculation-to-llama.tnpkg")
    server = create_feed_server(
        FeedServerConfig(
            data_dir=seed_finance_feed_data(tmp_path / "data"),
            event_stream=io.StringIO(),
            port=0,
            publisher_config_path=source.yaml_path,
            governance=governance,
        )
    )

    with _running_server(server) as feed_url:
        result = run_processing_path(
            feed_url=feed_url,
            account_id="acct-001",
            polars_config_path=polars.yaml_path,
            calculation_config_path=calculation.yaml_path,
            calculation=StubCalculationPort(),
        )
    opened = open_governed_object(
        llama.yaml_path,
        result.calculation_object,
        expected_type="portfolio.calculation.calculated",
        business_group="finance",
    )

    assert result.polars.values.height == 3
    assert result.calculation.result.provider == "stub"
    assert adapt_finance_source_record(result.model_row).ndjson == result.model_row.raw_ndjson
    assert opened.business["calculation_result"]["calculation_id"] == (
        "stub-portfolio-risk-001"
    )
    assert opened.governance == governance


def test_tn_governed_object_is_published_as_tn_llamas_existing_chained_row(
    tmp_path: Path,
) -> None:
    calculation = _endpoint(
        tmp_path,
        "calculation",
        ["portfolio_values", "calculation_result", "run_id"],
    )
    calculation_object = seal_governed_object(
        calculation.yaml_path,
        "portfolio.calculation.calculated",
        business_group="finance",
        business_fields={
            "portfolio_values": [{"portfolio_id": "portfolio-001"}],
            "calculation_result": {
                "calculation_id": "stub-portfolio-risk-001",
                "provider": "stub",
                "live": False,
            },
        },
        governance={
            **GOVERNANCE,
            "policy": (
                ".tn/config/agents.md#portfolio.calculation.calculated@1#"
                "sha256:398b235838f609ff6dbb7d5c66847935dccb27cc0fc61b857c9727988884a9d7"
            ),
        },
    )

    row = publish_calculation_object_for_model(
        calculation_object,
        publisher_config_path=calculation.yaml_path,
    )
    source = adapt_finance_source_record(row)

    assert row.envelope["sequence"] == 1
    assert row.envelope["prev_hash"] == "sha256:" + "0" * 64
    assert row.encrypted_groups == ("finance", "tn.agents")
    assert json.loads(row.envelope["tn_aad"]) == {
        "finance": {
            "governed_by": row.source_writer_did,
            "policy": row.policy_reference,
        },
        "tn.agents": {
            "governed_by": row.source_writer_did,
            "policy": row.policy_reference,
        },
    }
    assert source.ndjson == row.raw_ndjson


def test_reference_path_provisions_each_endpoint_before_the_happy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = build_finance_feed_bundles(tmp_path / "bundles")
    unity_locations = {
        f"portfolio.finance.{name.replace('.', '_')}": bundle.resolve().as_uri()
        for name, bundle in bundles.items()
    }

    def unity_request(_registry, method: str, path: str, payload=None):
        assert method == "GET"
        assert payload is None
        return {"storage_location": unity_locations[path.removeprefix("/volumes/")]}

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", unity_request)
    material = provision_reference_path(
        tmp_path / "tn",
        source_agents_path=Path("fixtures/portfolio/agents.policy"),
        output_agents_path=Path("fixtures/portfolio/report-agents.policy"),
    )
    governance = json.loads(Path("fixtures/portfolio/tn_governance.json").read_text())
    server = create_feed_server(
        FeedServerConfig(
            data_dir=seed_finance_feed_data(tmp_path / "data"),
            event_stream=io.StringIO(),
            port=0,
            publisher_config_path=material.source_config_path,
            governance=governance,
        )
    )

    with _running_server(server) as feed_url:
        result = run_processing_path(
            feed_url=feed_url,
            account_id="acct-001",
            polars_config_path=material.polars_config_path,
            calculation_config_path=material.calculation_config_path,
            calculation=StubCalculationPort(),
        )
    opened = open_governed_object(
        material.input_recipient_config_path,
        result.model_row.raw_ndjson,
        expected_type="portfolio.calculation.calculated",
        business_group="finance",
    )
    flow = build_reference_flow(material, result.model_row)

    assert opened.business["calculation_result"]["provider"] == "stub"
    assert opened.governance == governance
    assert material.input_reader.reader_key_dir.is_dir()
    assert material.output_writer.reader.reader_key_dir.is_dir()
    assert flow.source_row is result.model_row
    assert flow.marker_free_row is None
