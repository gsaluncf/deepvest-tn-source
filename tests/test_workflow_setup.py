"""Application configuration binds native endpoint routes without replacing enrollment."""

from contextlib import ExitStack

import pytest
import tn

from deepvest_tn.native_reference import provision_reference_sessions
from deepvest_tn.workflow_setup import bind_workflow

POLICY = """## finance.input
### instruction
Calculate an approved result.
### use_for
Portfolio analysis.
### do_not_use_for
Other uses.
### consequences
Review.
### on_violation_or_error
Refuse.

## finance.other
### instruction
Retain an unrelated input.
### use_for
Portfolio analysis.
### do_not_use_for
Other uses.
### consequences
Review.
### on_violation_or_error
Refuse.
"""


def uses(application="deepvest"):
    return (
        tn.UseContext(application, "portfolio_analysis", "calculate"),
        tn.UseContext(application, "model_handoff", "release_calculation"),
    )


def configured(session, *, decide_receive=lambda _: True, decide_release=lambda _: True):
    receive_use, release_use = uses()
    return bind_workflow(
        session,
        application="deepvest",
        receive_use=receive_use,
        release_use=release_use,
        input_type="finance.input",
        output_type="finance.calculated",
        groups=["finance"],
        to="tn-llama",
        decide_receive=decide_receive,
        decide_release=decide_release,
    )


def test_workflow_rejects_mismatched_application_before_registering_routes():
    with tn.Session(POLICY, groups=["finance"]) as session:
        receive_use, release_use = uses()
        with pytest.raises(ValueError, match="uses must match"):
            bind_workflow(
                session,
                application="analytics",
                receive_use=receive_use,
                release_use=release_use,
                input_type="finance.input",
                output_type="finance.calculated",
                groups=["finance"],
                to="tn-llama",
                decide_receive=lambda _: pytest.fail("mismatched route was invoked"),
                decide_release=lambda _: pytest.fail("mismatched route was invoked"),
            )
        source = session.create(
            {"rows": [{"wealth": 100}]},
            session.policy("finance.input"),
            object_type="finance.input",
            group="finance",
        )
        with pytest.raises(ValueError, match="not configured"):
            session.receive(source, purpose=receive_use.purpose)


def test_bound_workflow_refuses_unregistered_input_type_before_admission_callback():
    with tn.Session(POLICY, groups=["finance"]) as session:
        admissions = []
        work = configured(
            session, decide_receive=lambda context: admissions.append(context) or True
        )
        other = session.create(
            {"rows": [{"wealth": 100}]},
            session.policy("finance.other"),
            object_type="finance.other",
            group="finance",
        )
        with pytest.raises(ValueError, match="configured|route|type"):
            work.receive(other)
        assert admissions == []


def test_configured_release_refusal_cannot_be_overridden_per_call():
    with tn.Session(POLICY, groups=["finance"]) as session:
        work = configured(session, decide_release=lambda _: False)
        source = session.create(
            {"rows": [{"wealth": 100}]},
            session.policy("finance.input"),
            object_type="finance.input",
            group="finance",
        )
        data = work.receive(source)
        data.set(None, {"calculation": {"value": -0.25}}, group="finance")
        data.select(["finance"])
        with pytest.raises(tn.governed.UseDenied):
            work.release(data, decide=lambda _: True)


def test_enrolled_endpoint_identities_and_reader_capabilities_survive_reopen(tmp_path):
    paths = provision_reference_sessions(tmp_path / "sessions")
    roles = ("source", "polars", "deepvest", "model-reader", "model-writer", "recipient")
    source_use = tn.UseContext("tn-llama", "portfolio_analysis", "read_source")
    report_use = tn.UseContext("reporting", "portfolio_analysis", "read_report")
    with ExitStack() as stack:
        first = {
            role: stack.enter_context(tn.Session.from_config(paths[role])) for role in roles
        }
        identities = {role: session.did for role, session in first.items()}
        assert len(set(identities.values())) == len(roles)
        for role in ("source", "polars", "deepvest"):
            assert first[role].check_groups(["finance"]).is_ready
        assert first["model-writer"].check_groups(["default"]).is_ready
        source = first["source"].create(
            {"rows": [{"wealth": 100}]},
            first["source"].policy("portfolio.feed.data.v1"),
            object_type="portfolio.feed.data.v1",
            group="finance",
        ).release(
            use=tn.UseContext("source", "portfolio_analysis", "publish"),
            to="tn-llama",
            decide=lambda _: True,
        )
        report = first["model-writer"].create(
            {"report_html": "<p>ok</p>"},
            first["model-writer"].policy("portfolio.feed.data.v1"),
            object_type="report.generated",
            group="default",
        ).release(
            use=tn.UseContext("tn-llama", "portfolio_analysis", "generate_report"),
            to="reporting",
            decide=lambda _: True,
        )
        assert first["model-reader"].receive(
            source, use=source_use, groups=["finance"], decide=lambda _: True,
        ).get("rows", group="finance") == [{"wealth": 100}]
        assert first["recipient"].receive(
            report, use=report_use, groups=["default"], decide=lambda _: True,
        ).get("report_html", group="default") == "<p>ok</p>"
    with ExitStack() as stack:
        reopened = {
            role: stack.enter_context(tn.Session.from_config(paths[role])) for role in roles
        }
        assert {role: session.did for role, session in reopened.items()} == identities
        assert reopened["model-reader"].receive(
            source, use=source_use, groups=["finance"], decide=lambda _: True,
        ).get("rows", group="finance") == [{"wealth": 100}]
        assert reopened["recipient"].receive(
            report, use=report_use, groups=["default"], decide=lambda _: True,
        ).get("report_html", group="default") == "<p>ok</p>"
