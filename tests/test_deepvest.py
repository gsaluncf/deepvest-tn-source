from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import polars as pl
import pytest

from deepvest_tn.deepvest import (
    DeepVestExecution,
    DeepVestMetric,
    DeepVestPort,
    DeepVestPosition,
    DeepVestRequest,
    DeepVestResult,
    StubDeepVest,
    build_deepvest_request,
    execute_deepvest,
)
from deepvest_tn.feed_server import FeedServerConfig, create_feed_server, seed_finance_feed_data
from deepvest_tn.polars_stage import GovernedPortfolioResult, run_governed_polars_stage
from deepvest_tn.unity import UnityCatalogConfig, build_finance_feed_bundles

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
def feed_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    data_dir = seed_finance_feed_data(tmp_path_factory.mktemp("deepvest-getdata") / "data")
    server = create_feed_server(
        FeedServerConfig(data_dir=data_dir, event_stream=io.StringIO(), port=0)
    )
    with _running_feed_server(server) as base_url:
        yield base_url


class RecordingUnityTransport:
    def __init__(self, bundles: dict[str, Path], config: UnityCatalogConfig) -> None:
        self.volumes = {
            f"{config.catalog}.{config.schema}.{name.replace('.', '_')}": bundle.resolve().as_uri()
            for name, bundle in bundles.items()
        }

    def request(self, method: str, path: str, payload=None):
        if method == "GET" and path.startswith("/volumes/"):
            return {"storage_location": self.volumes[path.removeprefix("/volumes/")]}
        raise AssertionError(f"unexpected Unity request: {method} {path} {payload}")


@pytest.fixture(scope="module")
def finance_bundles(tmp_path_factory) -> dict[str, Path]:
    return build_finance_feed_bundles(tmp_path_factory.mktemp("deepvest-finance-bundles"))


@pytest.fixture()
def governed_portfolio(monkeypatch, finance_bundles, feed_url: str) -> GovernedPortfolioResult:
    config = UnityCatalogConfig()
    transport = RecordingUnityTransport(finance_bundles, config)

    def request(_registry, method: str, path: str, payload=None):
        return transport.request(method, path, payload)

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", request)
    return run_governed_polars_stage(
        config,
        account_id="acct-001",
        feed_url=feed_url,
    )


def expected_stub_result() -> DeepVestResult:
    return DeepVestResult(
        calculation_id="stub-portfolio-risk-001",
        analysis_type="portfolio_risk",
        metrics=(
            DeepVestMetric("value_at_risk_95", 12500.0, "USD"),
            DeepVestMetric("expected_shortfall_95", 16800.0, "USD"),
            DeepVestMetric("maximum_drawdown", -0.18, "ratio"),
        ),
        provider="stub",
        live=False,
    )


def test_real_governed_result_becomes_exact_ordered_allowlisted_request(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    request = build_deepvest_request(governed_portfolio)

    assert request == DeepVestRequest(
        positions=(
            DeepVestPosition(
                portfolio_id="portfolio-001",
                symbol="AAPL",
                quantity=125,
                cost_basis=182.4,
                last_price=228.15,
                position_market_value=28518.75,
                currency="USD",
                as_of="2026-09-01T16:00:00Z",
                market_source="deterministic-market-fixture",
            ),
            DeepVestPosition(
                portfolio_id="portfolio-001",
                symbol="MSFT",
                quantity=80,
                cost_basis=412.15,
                last_price=509.9,
                position_market_value=40792.0,
                currency="USD",
                as_of="2026-09-01T16:00:00Z",
                market_source="deterministic-market-fixture",
            ),
            DeepVestPosition(
                portfolio_id="portfolio-001",
                symbol="VTI",
                quantity=210,
                cost_basis=265.75,
                last_price=311.42,
                position_market_value=65398.2,
                currency="USD",
                as_of="2026-09-01T16:00:00Z",
                market_source="deterministic-market-fixture",
            ),
        )
    )
    assert not hasattr(request, "account_id")
    assert not hasattr(request, "policy_ref")
    assert not hasattr(request.positions[0], "account_id")


def test_execute_calls_substitutable_port_once_and_retains_exact_exchange(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    result = expected_stub_result()

    class RecordingPort:
        def __init__(self) -> None:
            self.requests: list[DeepVestRequest] = []

        def calculate(self, request: DeepVestRequest) -> DeepVestResult:
            self.requests.append(request)
            return result

    port = RecordingPort()
    assert isinstance(port, DeepVestPort)

    execution = execute_deepvest(port, governed_portfolio)

    assert len(port.requests) == 1
    assert execution == DeepVestExecution(request=port.requests[0], result=result)


def test_stub_records_exact_request_and_returns_fixed_non_live_result(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    request = build_deepvest_request(governed_portfolio)
    stub = StubDeepVest()

    result = stub.calculate(request)

    assert stub.invocations == (request,)
    assert result == expected_stub_result()
    assert result.provider == "stub"
    assert result.live is False


def test_semantic_input_change_is_recorded_but_does_not_change_stub_result(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    original = build_deepvest_request(governed_portfolio)
    changed_position = replace(original.positions[0], position_market_value=1.0)
    changed = replace(original, positions=(changed_position, *original.positions[1:]))
    stub = StubDeepVest()

    original_result = stub.calculate(original)
    changed_result = stub.calculate(changed)

    assert stub.invocations == (original, changed)
    assert stub.invocations[0] != stub.invocations[1]
    assert original_result == changed_result == expected_stub_result()


def test_two_direct_stub_calls_both_succeed_and_are_recorded(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    request = build_deepvest_request(governed_portfolio)
    stub = StubDeepVest()

    first = stub.calculate(request)
    second = stub.calculate(request)

    assert first == second == expected_stub_result()
    assert stub.invocations == (request, request)


def test_raising_port_is_attempted_once_without_retry_or_fallback(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    failure = RuntimeError("DeepVest boundary unavailable")

    class RaisingPort:
        def __init__(self) -> None:
            self.attempts = 0

        def calculate(self, request: DeepVestRequest) -> DeepVestResult:
            self.attempts += 1
            raise failure

    port = RaisingPort()

    with pytest.raises(RuntimeError) as caught:
        execute_deepvest(port, governed_portfolio)

    assert caught.value is failure
    assert port.attempts == 1


def test_contract_and_exposed_invocation_history_are_deeply_immutable(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    request = build_deepvest_request(governed_portfolio)
    stub = StubDeepVest()
    result = stub.calculate(request)
    execution = DeepVestExecution(request, result)
    history = stub.invocations

    with pytest.raises(FrozenInstanceError):
        request.positions[0].symbol = "REPLACED"
    with pytest.raises(FrozenInstanceError):
        request.positions = ()
    with pytest.raises(AttributeError):
        request.positions.append(request.positions[0])
    with pytest.raises(FrozenInstanceError):
        result.metrics[0].value = 0.0
    with pytest.raises(AttributeError):
        result.metrics.append(result.metrics[0])
    with pytest.raises(FrozenInstanceError):
        result.provider = "live"
    with pytest.raises(FrozenInstanceError):
        execution.result = expected_stub_result()
    with pytest.raises(AttributeError):
        history.append(request)

    assert stub.invocations == (request,)


class CountingPort:
    def __init__(self) -> None:
        self.calls = 0

    def calculate(self, request: DeepVestRequest) -> DeepVestResult:
        self.calls += 1
        return expected_stub_result()


def replace_portfolio_values(
    portfolio: GovernedPortfolioResult,
    values: pl.DataFrame,
) -> GovernedPortfolioResult:
    """Keep adversarial fixtures on the immutable calculation result type."""
    return replace(
        portfolio,
        _rows=tuple(values.iter_rows()),
        _schema=tuple(values.schema.items()),
    )


def test_empty_table_fails_before_the_port_is_called(
    governed_portfolio: GovernedPortfolioResult,
) -> None:
    empty = replace_portfolio_values(
        governed_portfolio,
        governed_portfolio.values.head(0),
    )
    port = CountingPort()

    with pytest.raises(ValueError, match="empty"):
        execute_deepvest(port, empty)

    assert port.calls == 0


@pytest.mark.parametrize(
    ("column", "replacement_value", "message"),
    [
        ("portfolio_id", "portfolio-002", "portfolio_id"),
        ("currency", "EUR", "currency"),
        ("as_of", "2026-09-02T16:00:00Z", "as_of"),
    ],
)
def test_mixed_context_fails_before_the_port_is_called(
    governed_portfolio: GovernedPortfolioResult,
    column: str,
    replacement_value: str,
    message: str,
) -> None:
    rows = governed_portfolio.values.to_dicts()
    rows[1][column] = replacement_value
    mixed = replace_portfolio_values(governed_portfolio, pl.DataFrame(rows))
    port = CountingPort()

    with pytest.raises(ValueError, match=message):
        execute_deepvest(port, mixed)

    assert port.calls == 0


def test_stub_execution_creates_no_local_artifacts(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    execution = execute_deepvest(StubDeepVest(), governed_portfolio)

    assert execution.result == expected_stub_result()
    assert list(tmp_path.iterdir()) == []
