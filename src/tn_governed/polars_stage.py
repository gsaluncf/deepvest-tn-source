"""User-authored governed Polars transformations for the portfolio workflow."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import polars as pl
import tn
from tn_polars.builder import GovernedPolars
from tn_polars.frame import GovernedLazyFrame

from .getdata import FeedFetchFact, GetDataConfig, fetch_feed
from .unity import (
    FEED_NAMES,
    MARKET_PRICES_FEED,
    POSITIONS_FEED,
    UnityCatalogConfig,
    resolve_finance_feeds,
)


def prepare_wealth_calculation(
    wealth_wire,
    calendar_wire,
    *,
    wealth_edition,
    calendar_edition,
    session: tn.Session,
    use: tn.UseContext,
    decide_receive,
) -> GovernedLazyFrame:
    """Admit two exact editions and prepare ordered wealth rows for calculation."""
    if wealth_edition.catalog is not calendar_edition.catalog:
        raise ValueError("calculation editions must share one accepted native catalog")
    wealth = GovernedPolars.load_from_tn(
        wealth_wire, session=session, use=use, selection=wealth_edition.selection,
        policies=wealth_edition.entries, decide=decide_receive,
    )
    calendar = GovernedPolars.load_from_tn(
        calendar_wire, session=session, use=use, selection=calendar_edition.selection,
        policies=calendar_edition.entries, decide=decide_receive,
    )
    return (
        wealth.join(calendar, on="observation_id")
        .select("observed_at", "wealth")
        .sort("observed_at")
    )

_POSITION_COLUMNS = (
    "account_id",
    "portfolio_id",
    "symbol",
    "quantity",
    "cost_basis",
    "currency",
)
_MARKET_COLUMNS = (
    "symbol",
    "currency",
    "as_of",
    "last_price",
    "market_source",
)
_DOWNSTREAM_COLUMNS = (
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
)


def _freeze_json_like(value: object) -> object:
    """Recursively copy JSON-like containers into read-only equivalents."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_json_like(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_like(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_json_like(item) for item in value)
    return value


class EmptyPortfolioSelection(ValueError):
    """The requested account produced no governed positions."""


@dataclass(frozen=True, slots=True)
class GovernedPortfolioResult:
    """Materialized values and direct copies of their native governance facts."""

    _rows: tuple[tuple[object, ...], ...]
    _schema: tuple[tuple[str, pl.DataType], ...]
    sources: Mapping[str, frozenset[str]]
    row_sources: frozenset[str]
    policy_ref: str
    tn_governance: Mapping[str, object]
    feed_names: tuple[str, str]
    feed_fetch_facts: tuple[FeedFetchFact, FeedFetchFact]

    def __post_init__(self) -> None:
        if tuple(fact.feed_name for fact in self.feed_fetch_facts) != self.feed_names:
            raise ValueError("fetch facts must match the exact ordered feed names")

    @property
    def values(self) -> pl.DataFrame:
        """Materialize an independent DataFrame from the immutable stored payload."""
        return pl.DataFrame(self._rows, schema=list(self._schema), orient="row")


@dataclass(frozen=True, slots=True)
class GovernedPolarsTNResult:
    """Real Polars result plus the canonical TN object emitted for calculation."""

    _rows: tuple[tuple[object, ...], ...]
    _schema: tuple[tuple[str, pl.DataType], ...]
    sources: Mapping[str, frozenset[str]]
    row_sources: frozenset[str]
    policy_ref: str
    tn_governance: Mapping[str, object]
    feed_names: tuple[str, str]
    sealed_object: tn.SealedObject

    @property
    def values(self) -> pl.DataFrame:
        return pl.DataFrame(self._rows, schema=list(self._schema), orient="row")


def transform_position_market_values(
    positions: GovernedLazyFrame,
    market_prices: GovernedLazyFrame,
    *,
    account_id: str,
) -> GovernedLazyFrame:
    """Build the requested position-level values without materializing the lazy frame."""
    selected_positions = positions.filter(
        pl.col("account_id") == account_id
    ).select(*_POSITION_COLUMNS)
    selected_prices = market_prices.select(*_MARKET_COLUMNS)

    return (
        selected_positions.join(
            selected_prices,
            on=("symbol", "currency"),
        )
        .with_columns(
            (pl.col("quantity") * pl.col("last_price")).round(2).alias(
                "position_market_value"
            )
        )
        .select(*_DOWNSTREAM_COLUMNS)
        .sort("symbol")
    )


def run_governed_polars_stage(
    config: UnityCatalogConfig,
    *,
    account_id: str,
    feed_url: str,
) -> GovernedPortfolioResult:
    """Resolve, fetch, admit, transform, and materialize governed finance feeds."""
    entries = resolve_finance_feeds(config)
    positions_entry = entries[POSITIONS_FEED]
    prices_entry = entries[MARKET_PRICES_FEED]
    getdata = GetDataConfig(base_url=feed_url)

    fetched_positions = fetch_feed(getdata, POSITIONS_FEED)
    positions = GovernedPolars.prepare(positions_entry).load(
        pl.scan_ndjson(io.BytesIO(fetched_positions.body))
    )
    fetched_prices = fetch_feed(getdata, MARKET_PRICES_FEED)
    prices = GovernedPolars.prepare(prices_entry).load(
        pl.scan_ndjson(io.BytesIO(fetched_prices.body))
    )
    governed = transform_position_market_values(
        positions,
        prices,
        account_id=account_id,
    )

    values = governed.frame.collect()
    if values.is_empty():
        raise EmptyPortfolioSelection(
            f"governed position selection for account {account_id!r} was empty"
        )

    return GovernedPortfolioResult(
        _rows=tuple(
            tuple(_freeze_json_like(item) for item in row)
            for row in values.iter_rows()
        ),
        _schema=tuple(values.schema.items()),
        sources=MappingProxyType(
            {name: frozenset(roots) for name, roots in governed.sources.items()}
        ),
        row_sources=frozenset(governed.row_sources),
        policy_ref=positions_entry.policy.policy_ref,
        tn_governance=_freeze_json_like(positions_entry.tn_governance),
        feed_names=FEED_NAMES,
        feed_fetch_facts=(fetched_positions.fact, fetched_prices.fact),
    )


def run_governed_polars_tn_stage_legacy(
    entries: Mapping[str, object],
    source_objects: Mapping[str, tn.SealedObject | Mapping[str, object] | str | bytes],
    *,
    endpoint_config_path: Path,
    account_id: str,
) -> GovernedPolarsTNResult:
    """Open governed feed objects, run ordinary Polars, and reseal the derivation."""
    from .tn_objects import open_governed_object, seal_governed_object

    opened = {}
    frames = {}
    for feed_name in FEED_NAMES:
        item = open_governed_object(
            endpoint_config_path,
            source_objects[feed_name],
            expected_type="portfolio.feed.data.v1",
            business_group="finance",
        )
        if item.business.get("feed_name") != feed_name:
            raise ValueError(f"TN source object does not contain feed {feed_name!r}")
        rows = item.business.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"TN source object for {feed_name!r} contains no rows")
        entry = entries[feed_name]
        if item.governance != entry.tn_governance:
            raise ValueError(f"TN source governance differs from Unity for {feed_name!r}")
        opened[feed_name] = item
        frames[feed_name] = GovernedPolars.prepare(entry).load(pl.LazyFrame(rows))

    if opened[POSITIONS_FEED].governance != opened[MARKET_PRICES_FEED].governance:
        raise ValueError("TN source feeds do not carry the same governance")
    governed = transform_position_market_values(
        frames[POSITIONS_FEED],
        frames[MARKET_PRICES_FEED],
        account_id=account_id,
    )
    values = governed.frame.collect()
    if values.is_empty():
        raise EmptyPortfolioSelection(
            f"governed position selection for account {account_id!r} was empty"
        )

    governance = opened[POSITIONS_FEED].governance
    business_fields = {
        "portfolio_values": values.to_dicts(),
        "native_feed_names": list(FEED_NAMES),
        "native_column_sources": {
            name: sorted(sources) for name, sources in sorted(governed.sources.items())
        },
        "native_row_sources": sorted(governed.row_sources),
    }
    sealed = seal_governed_object(
        endpoint_config_path,
        "portfolio.polars.derived.v1",
        business_group="finance",
        business_fields=business_fields,
        governance=governance,
    )
    return GovernedPolarsTNResult(
        _rows=tuple(
            tuple(_freeze_json_like(item) for item in row) for row in values.iter_rows()
        ),
        _schema=tuple(values.schema.items()),
        sources=MappingProxyType(
            {name: frozenset(roots) for name, roots in governed.sources.items()}
        ),
        row_sources=frozenset(governed.row_sources),
        policy_ref=entries[POSITIONS_FEED].policy.policy_ref,
        tn_governance=_freeze_json_like(governance),
        feed_names=FEED_NAMES,
        sealed_object=sealed,
    )


run_governed_polars_tn_stage = run_governed_polars_tn_stage_legacy
