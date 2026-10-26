"""Executable TN-object processing path from Unity resolution through calculation."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import tn

from .calculation import (
    CalculationExecution,
    PortfolioExecution,
    CalculationPort,
    execute_calculation,
    execute_calculation_for_model,
)
from .getdata import FeedFetchFact, GetDataConfig, fetch_governed_feed
from .polars_stage import (
    GovernedPolarsTNResult,
    prepare_wealth_calculation,
    run_governed_polars_tn_stage,
)
from .tn_flow import ModelBoundTNRow
from .unity import FEED_NAMES, UnityCatalogConfig, resolve_finance_feeds
from .workflow_setup import bind_workflow


@dataclass(frozen=True, slots=True)
class GovernedCalculationPathResult:
    """Exact signed boundaries and the calculation performed between them."""

    polars_object: tn.GovernedObject
    calculation_object: tn.GovernedObject
    execution: CalculationExecution
    feed_fetch_facts: tuple[FeedFetchFact, ...]


def run_governed_calculation_path(
    *,
    wealth_edition,
    calendar_edition,
    getdata: GetDataConfig,
    polars_session: tn.Session,
    calculation_session: tn.Session,
    policy_dag: tn.PolicyDag,
    publication_store,
    calculation,
    polars_use: tn.UseContext,
    polars_release_use: tn.UseContext,
    calculation_use: tn.UseContext,
    calculation_release_use: tn.UseContext,
    decide_source,
    decide_polars_release,
    decide_calculation_receive,
    decide_calculation_release,
    to: str,
    decide_ancestor=None,
) -> GovernedCalculationPathResult:
    """Fetch selected publications, prepare wealth, and publish one calculation result.

    Sessions and use contexts belong to the configured receiving endpoints.
    Application callbacks supplement native selection, catalog and ancestry checks.
    """
    catalog = wealth_edition.catalog
    if catalog is not calendar_edition.catalog:
        raise ValueError("calculation editions must share one accepted native catalog")
    editions = (wealth_edition, calendar_edition)
    fetched = tuple(fetch_governed_feed(getdata, item.selection) for item in editions)
    frame = prepare_wealth_calculation(
        fetched[0].body, fetched[1].body,
        wealth_edition=wealth_edition, calendar_edition=calendar_edition,
        session=polars_session, use=polars_use, decide_receive=decide_source,
    )
    for item in fetched:
        publication_store.put(tn.GovernedObject.read(io.BytesIO(item.body)))
    polars_object = frame.release_tn(
        use=polars_release_use, to="calculation", decide=decide_polars_release,
        group="finance", rows_field="rows", object_type="portfolio.calculation.input",
    )
    publication_store.put(polars_object)
    accepted_writers = {
        wealth_edition.selection.record.source_writer,
        calendar_edition.selection.record.source_writer,
        polars_session.did,
        calculation_session.did,
    }

    def resolve_ancestor(identity):
        object = tn.GovernedObject.read(io.BytesIO(publication_store.get(identity)))
        view = calculation_session.governance(object)
        if object.writer not in accepted_writers:
            raise PermissionError("retained ancestor writer is not accepted by this calculation")
        if decide_ancestor is not None and decide_ancestor(view) is not True:
            raise PermissionError("application refused the retained ancestor governance")
        return view

    def accept_calculation(context):
        if (context.writer != polars_session.did
                or context.object.id != polars_object.id
                or context.object_type != "portfolio.calculation.input"
                or context.use_context != calculation_use):
            return False
        if not catalog.accepts(context, policy_dag):
            return False
        tn.LineageVerifier().verify(
            calculation_session.governance(context.object), catalog, policy_dag, resolve_ancestor,
        )
        return decide_calculation_receive(context)

    expected_view = calculation_session.governance(polars_object)
    expected_contracts = expected_view.governance.policies

    def accept_calculation_release(context):
        policies = context.policies
        exact_contracts = (
            len(policies) == len(expected_contracts)
            and all(
                sum(policy.matches_contract(expected) for expected in expected_contracts) == 1
                for policy in policies
            )
            and all(
                sum(expected.matches_contract(policy) for policy in policies) == 1
                for expected in expected_contracts
            )
        )
        return (
            context.use_context == calculation_release_use
            and context.destination == to
            and context.object_type == "portfolio.calculation.calculated"
            and exact_contracts
            and len(context.sources) == 1
            and context.sources[0].references(polars_object)
            and decide_calculation_release(context)
        )

    work = bind_workflow(
        calculation_session,
        application="calculation",
        receive_use=calculation_use,
        release_use=calculation_release_use,
        input_type="portfolio.calculation.input",
        output_type="portfolio.calculation.calculated",
        groups=["finance"],
        to=to,
        decide_receive=accept_calculation,
        decide_release=accept_calculation_release,
    )
    execution, calculation_object = execute_calculation(
        calculation, polars_object, work=work,
    )
    publication_store.put(calculation_object)
    return GovernedCalculationPathResult(
        polars_object, calculation_object, execution, tuple(item.fact for item in fetched),
    )


@dataclass(frozen=True, slots=True)
class ProcessingPathResult:
    """The two governed application-stage results produced before tn-llama."""

    polars: GovernedPolarsTNResult
    calculation: PortfolioExecution
    calculation_object: tn.SealedObject
    model_row: ModelBoundTNRow
    feed_fetch_facts: tuple[FeedFetchFact, ...]


def run_processing_path_legacy(
    *,
    feed_url: str,
    account_id: str,
    polars_config_path: Path,
    calculation_config_path: Path,
    calculation: CalculationPort,
    unity: UnityCatalogConfig | None = None,
) -> ProcessingPathResult:
    """Snapshot workflow with a separate logged model publication."""
    entries = resolve_finance_feeds(unity)
    getdata = GetDataConfig(base_url=feed_url)
    fetched = tuple(fetch_governed_feed(getdata, feed_name) for feed_name in FEED_NAMES)
    source_objects = {item.feed_name: item.body for item in fetched}
    polars = run_governed_polars_tn_stage(
        entries,
        source_objects,
        endpoint_config_path=polars_config_path,
        account_id=account_id,
    )
    execution, calculation_object, model_row = execute_calculation_for_model(
        calculation,
        polars.sealed_object,
        recipient_config_path=calculation_config_path,
        publisher_config_path=calculation_config_path,
    )
    return ProcessingPathResult(
        polars=polars,
        calculation=execution,
        calculation_object=calculation_object,
        model_row=model_row,
        feed_fetch_facts=tuple(item.fact for item in fetched),
    )


# This alias selects the snapshot workflow; native execution has a separate entry point.
run_processing_path = run_processing_path_legacy
