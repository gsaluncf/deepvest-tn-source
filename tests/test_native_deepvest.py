"""One accepted input produces one calculated and signed model-facing object."""

from dataclasses import asdict

import pytest
import tn
from test_native_tn_objects import POLICY

from deepvest_tn.deepvest import execute_deepvest_tn
from deepvest_tn.deepvest_engine import EngineDeepVestPort
from deepvest_tn.workflow_setup import bind_workflow

ROWS = [
    {"observed_at": "2026-09-01T16:00:00+00:00", "wealth": 100.0},
    {"observed_at": "2026-09-02T16:00:00+00:00", "wealth": 120.0},
    {"observed_at": "2026-09-03T16:00:00+00:00", "wealth": 90.0},
    {"observed_at": "2026-09-04T16:00:00+00:00", "wealth": 110.0},
]


class CountingPort:
    def __init__(self):
        self.requests = []

    def calculate(self, request):
        self.requests.append(request)
        return EngineDeepVestPort().calculate(request)


def source(session, rows=ROWS):
    return session.create_obj_with_groups(
        {"finance": {"rows": rows}, "audit": {"source_note": "retain at origin"}},
        session.policy("finance.account"), object_type="portfolio.calculation.input",
        primary_group="finance",
    ).snapshot


def workflow(session, *, accept=lambda _: True, release=lambda _: True):
    return bind_workflow(
        session, application="deepvest",
        receive_use=tn.UseContext("deepvest", "portfolio_analysis", "calculate"),
        release_use=tn.UseContext("deepvest", "portfolio_analysis", "release_calculation"),
        input_type="portfolio.calculation.input",
        output_type="portfolio.deepvest.calculated",
        groups=["finance"], to="tn-llama",
        decide_receive=accept, decide_release=release,
    )


def execute(session, publication, port, *, accept=lambda _: True, release=lambda _: True):
    return execute_deepvest_tn(
        port, publication, work=workflow(session, accept=accept, release=release),
    )


def test_native_calculation_preserves_contract_and_exact_parent_and_projects_output():
    with tn.Session(POLICY, groups=["finance", "audit"]) as session:
        publication = source(session)
        port = CountingPort()
        releases = []
        execution, result = execute(
            session, publication, port,
            accept=lambda ctx: ctx.object.id == publication.id and ctx.writer == session.did,
            release=lambda ctx: releases.append(ctx) or True,
        )
        assert len(port.requests) == len(releases) == 1
        assert execution.result.value == -0.25
        assert execution.source_object_id == publication.id
        assert execution.request.arguments["wealth_path"] == [100, 120, 90, 110]
        output = session.receive(
            result, use=tn.UseContext("tn-llama", "portfolio_analysis", "generate_report"),
            groups=["finance"], decide=lambda _: True,
        )
        assert output.governance.sources[0].object_id == publication.id
        assert output.data["calculation"]["input_object_id"] == publication.id
        assert output.data["calculation"]["result"] == asdict(execution.result)
        assert "rows" not in output.data
        assert "audit" not in result.envelope
        assert len(output.policies) == 1
        assert output.governance.fields["release_context"] == {
            "application": "deepvest", "purpose": "portfolio_analysis",
            "operation": "release_calculation", "destination": "tn-llama",
        }


def test_native_refusal_makes_zero_calculation_calls():
    with tn.Session(POLICY, groups=["finance", "audit"]) as session:
        port = CountingPort()
        with pytest.raises(tn.governed.UseDenied):
            execute(session, source(session), port, accept=lambda _: False)
        assert port.requests == []


def test_exact_source_pin_refuses_before_calculation():
    with tn.Session(POLICY, groups=["finance", "audit"]) as session:
        expected = source(session)
        replacement = source(session)
        port = CountingPort()
        work = workflow(session, accept=lambda context: context.object.id == expected.id)
        with pytest.raises(tn.governed.UseDenied):
            execute_deepvest_tn(port, replacement, work=work)
        assert port.requests == []


def test_configured_release_refusal_runs_one_calculation_and_publishes_nothing():
    with tn.Session(POLICY, groups=["finance", "audit"]) as session:
        port = CountingPort()
        with pytest.raises(tn.governed.UseDenied):
            execute(session, source(session), port, release=lambda _: False)
        assert len(port.requests) == 1


@pytest.mark.parametrize("rows", [list(reversed(ROWS)), [ROWS[0], ROWS[0]],
                                  [{"observed_at": "2026-09-01", "wealth": 100}, ROWS[1]]])
def test_source_order_is_validated_before_engine_invocation(rows):
    with tn.Session(POLICY, groups=["finance", "audit"]) as session:
        port = CountingPort()
        with pytest.raises(ValueError, match="ordered|timestamp"):
            execute(session, source(session, rows), port)
        assert port.requests == []
