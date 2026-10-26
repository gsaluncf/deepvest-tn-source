"""Calculation interface for governed portfolio data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import tn

from .calculation_client import CalculationRequest, CalculationResult


@dataclass(frozen=True)
class CalculationExecution:
    """The invocation and result retained with their exact governed input."""

    source_object_id: str
    request: CalculationRequest
    result: CalculationResult


def execute_calculation(
    port,
    source: tn.GovernedObject | str | bytes,
    *,
    work: tn.Workflow,
) -> tuple[CalculationExecution, tn.GovernedObject]:
    """Accept, calculate, and release the exact model-facing TN object once."""
    working = work.receive(source)
    input_id = working.snapshot.id
    rows = working.get("rows", group="finance")
    if rows is None:
        raise ValueError("calculation input requires ordered wealth rows")
    observations = []
    previous = None
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"observed_at", "wealth"}:
            raise ValueError("calculation rows require observed_at and wealth")
        try:
            timestamp = datetime.fromisoformat(row["observed_at"])
        except (TypeError, ValueError) as error:
            raise ValueError("wealth rows require an ISO timestamp") from error
        if timestamp.tzinfo is None:
            raise ValueError("wealth rows require a timezone-aware timestamp")
        if previous is not None and timestamp <= previous:
            raise ValueError("wealth rows must be strictly ordered by timestamp")
        previous = timestamp
        observations.append(row["wealth"])
    request = CalculationRequest("max_drawdown", {"wealth_path": observations})
    result = port.calculate(request)
    execution = CalculationExecution(input_id, request, result)
    working.set(None, {
        "calculation": {
            "input_object_id": input_id,
            "request": {"function": request.function, "arguments": request.arguments},
            "result": asdict(result),
        },
    }, group="finance")
    working.select(["finance"])
    released = work.release(working)
    return execution, released

if TYPE_CHECKING:
    from .polars_stage import GovernedPortfolioResult


@dataclass(frozen=True)
class PortfolioPosition:
    """One allowlisted governed position supplied to the calculation boundary."""

    portfolio_id: str
    symbol: str
    quantity: int
    cost_basis: float
    last_price: float
    position_market_value: float
    currency: str
    as_of: str
    market_source: str


@dataclass(frozen=True)
class PortfolioRequest:
    """Ordered immutable portfolio input for a calculation adapter."""

    positions: tuple[PortfolioPosition, ...]


@dataclass(frozen=True)
class CalculationMetric:
    """One normalized metric returned by the calculation boundary."""

    name: str
    value: float
    unit: str


@dataclass(frozen=True)
class PortfolioResult:
    """Normalized application-domain result from a calculation port."""

    calculation_id: str
    analysis_type: str
    metrics: tuple[CalculationMetric, ...]
    provider: str
    live: bool


@dataclass(frozen=True)
class PortfolioExecution:
    """Exact immutable request/result exchange at the calculation boundary."""

    request: PortfolioRequest
    result: PortfolioResult


@runtime_checkable
class CalculationPort(Protocol):
    """Calculation interface shared by fixed-output and engine-backed adapters."""

    def calculate(self, request: PortfolioRequest) -> PortfolioResult:
        """Calculate the typed portfolio request."""
        ...


_POSITION_FIELDS = (
    "portfolio_id",
    "symbol",
    "quantity",
    "cost_basis",
    "last_price",
    "position_market_value",
    "currency",
    "as_of",
    "market_source",
)
_COMMON_CONTEXT_FIELDS = ("portfolio_id", "currency", "as_of")
_STUB_RESULT = PortfolioResult(
    calculation_id="stub-portfolio-risk-001",
    analysis_type="portfolio_risk",
    metrics=(
        CalculationMetric("value_at_risk_95", 12500.0, "USD"),
        CalculationMetric("expected_shortfall_95", 16800.0, "USD"),
        CalculationMetric("maximum_drawdown", -0.18, "ratio"),
    ),
    provider="stub",
    live=False,
)


def build_portfolio_request(portfolio: GovernedPortfolioResult) -> PortfolioRequest:
    """Copy the allowlisted governed values into the immutable port contract."""
    values = portfolio.values
    if values.is_empty():
        raise ValueError("cannot build a calculation request from an empty portfolio")

    for field in _COMMON_CONTEXT_FIELDS:
        if values.get_column(field).n_unique() != 1:
            raise ValueError(f"Calculation request contains mixed {field} values")

    positions = tuple(
        PortfolioPosition(
            portfolio_id=str(row["portfolio_id"]),
            symbol=str(row["symbol"]),
            quantity=int(row["quantity"]),
            cost_basis=float(row["cost_basis"]),
            last_price=float(row["last_price"]),
            position_market_value=float(row["position_market_value"]),
            currency=str(row["currency"]),
            as_of=str(row["as_of"]),
            market_source=str(row["market_source"]),
        )
        for row in values.select(*_POSITION_FIELDS).iter_rows(named=True)
    )
    return PortfolioRequest(positions=positions)


def execute_portfolio_calculation(
    port: CalculationPort,
    portfolio: GovernedPortfolioResult,
) -> PortfolioExecution:
    """Invoke the injected calculation port once with no retry or fallback."""
    request = build_portfolio_request(portfolio)
    result = port.calculate(request)
    return PortfolioExecution(request=request, result=result)


class StubCalculationPort:
    """Calculation adapter returning synthetic fixed metrics."""

    def __init__(self) -> None:
        self._invocations: list[PortfolioRequest] = []

    @property
    def invocations(self) -> tuple[PortfolioRequest, ...]:
        """Return an immutable record of every exact request received."""
        return tuple(self._invocations)

    def calculate(self, request: PortfolioRequest) -> PortfolioResult:
        """Record the request and return the fixed non-live transport fixture."""
        self._invocations.append(request)
        return _STUB_RESULT


def execute_sealed_calculation(
    port: CalculationPort,
    source: tn.SealedObject | Mapping[str, object] | str | bytes | Path,
    *,
    recipient_config_path: Path,
    publisher_config_path: Path,
) -> tuple[PortfolioExecution, tn.SealedObject]:
    """Open the Polars TN object, calculate once, and seal the governed result."""
    from .tn_objects import seal_governed_object

    opened, execution = _open_and_execute_sealed_calculation(
        port,
        source,
        recipient_config_path=recipient_config_path,
    )
    sealed = seal_governed_object(
        publisher_config_path,
        "portfolio.calculation.calculated",
        business_group="finance",
        business_fields={
            "portfolio_values": opened.business["portfolio_values"],
            "calculation_result": asdict(execution.result),
        },
        governance=opened.governance,
    )
    return execution, sealed


def _open_and_execute_sealed_calculation(
    port: CalculationPort,
    source: tn.SealedObject | Mapping[str, object] | str | bytes | Path,
    *,
    recipient_config_path: Path,
):
    from .tn_objects import open_governed_object

    opened = open_governed_object(
        recipient_config_path,
        source,
        expected_type="portfolio.polars.derived.v1",
        business_group="finance",
    )
    rows = opened.business.get("portfolio_values")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("Polars TN object must contain nonempty portfolio_values")

    request = PortfolioRequest(
        positions=tuple(
            PortfolioPosition(
                portfolio_id=str(row["portfolio_id"]),
                symbol=str(row["symbol"]),
                quantity=int(row["quantity"]),
                cost_basis=float(row["cost_basis"]),
                last_price=float(row["last_price"]),
                position_market_value=float(row["position_market_value"]),
                currency=str(row["currency"]),
                as_of=str(row["as_of"]),
                market_source=str(row["market_source"]),
            )
            for row in rows
        )
    )
    result = port.calculate(request)
    execution = PortfolioExecution(request=request, result=result)
    return opened, execution


def execute_calculation_for_model(
    port: CalculationPort,
    source: tn.SealedObject | Mapping[str, object] | str | bytes | Path,
    *,
    recipient_config_path: Path,
    publisher_config_path: Path,
):
    """Run the calculation and publish both canonical TN forms before leaving the endpoint."""
    from .tn_flow import publish_calculation_values_for_model
    from .tn_objects import seal_governed_object

    opened, execution = _open_and_execute_sealed_calculation(
        port,
        source,
        recipient_config_path=recipient_config_path,
    )
    portfolio_values = opened.business["portfolio_values"]
    calculation_result = asdict(execution.result)
    sealed = seal_governed_object(
        publisher_config_path,
        "portfolio.calculation.calculated",
        business_group="finance",
        business_fields={
            "portfolio_values": portfolio_values,
            "calculation_result": calculation_result,
        },
        governance=opened.governance,
    )
    model_row = publish_calculation_values_for_model(
        portfolio_values=portfolio_values,
        calculation_result=calculation_result,
        governance=opened.governance,
        publisher_config_path=publisher_config_path,
    )
    return execution, sealed, model_row
