"""Application helpers carry the native working object through a service."""

import pytest
import tn

from tn_governed import tn_objects

POLICY = """## finance.account
### instruction
Calculate the requested account total.
### use_for
Portfolio analysis.
### do_not_use_for
Trade execution.
### consequences
Review.
### on_violation_or_error
Refuse release.
"""


def test_origin_and_receipt_use_native_objects_without_global_identity_switch(monkeypatch):
    def global_init_forbidden(*args, **kwargs):
        pytest.fail("processing must use its own Session")

    monkeypatch.setattr(tn, "init", global_init_forbidden)
    with tn.Session(POLICY, groups=["finance"]) as session:
        source = tn_objects.originate_governed_object(
            session, {"rows": [{"amount": 12}, {"amount": 18}]},
            session.policy("finance.account"), object_type="finance.account", group="finance",
        )
        use = tn.UseContext("calculation", "portfolio_analysis", "calculate")
        seen = []

        def accept(context):
            seen.append(context)
            return context.writer == session.did and context.use_context == use

        received = tn_objects.receive_governed_object(
            session, source.snapshot.wire, use=use, groups=["finance"], decide=accept,
        )
        assert isinstance(source, tn._native.governed.DataObject)
        assert isinstance(received, tn._native.governed.DataObject)
        assert received.data["rows"] == [{"amount": 12}, {"amount": 18}]
        assert seen[0].object.id == source.snapshot.id
        assert received.sources[0].object_id == source.snapshot.id


@pytest.mark.parametrize("reason", ["writer", "contract", "use"])
def test_refused_receipt_never_invokes_calculation(reason):
    with tn.Session(POLICY, groups=["finance"]) as session:
        source = session.create_obj(
            {"rows": [{"amount": 12}]}, session.policy("finance.account"),
            object_type="finance.account", group="finance",
        )
        called = []

        def receive_and_calculate():
            working = tn_objects.receive_governed_object(
                session, source.snapshot.wire,
                use=tn.UseContext("calculation", "portfolio_analysis", "calculate"),
                groups=["finance"],
                decide=lambda context: {
                    "writer": context.writer == "unaccepted writer",
                    "contract": len(context.policies) == 2,
                    "use": context.use_context.application == "some other application",
                }[reason],
            )
            called.append(working.data["rows"])

        with pytest.raises(tn.governed.UseDenied):
            receive_and_calculate()
        assert called == []
