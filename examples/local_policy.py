"""Read a local policy, govern a string and a Decimal, and verify signed records."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import tn

from tn_governed.publications import open_publication_store
from tn_governed.values import (
    Assessment,
    Context,
    EvidenceRecorder,
    Request,
    Rules,
    verify_tape,
)
from tn_governed.values.primitives import VALUE_TYPE


def originate(context, raw, policy, label):
    """Authorize local origination before attaching the supplied policy."""
    request = Request(
        context.application, context.purpose, "select",
        (), (context.session.did,), (policy,), (),
    )
    call_id = uuid4().hex
    decision_id = context._decision(request, call_id)
    return context.originate(
        raw, policy, source={"label": label}, request=request,
        decision_id=decision_id, call_id=call_id,
    )


def main():
    policy_text = Path(__file__).with_name("internal_report.policy").read_text(encoding="utf-8")
    with (
        TemporaryDirectory(prefix="tn-local-policy-") as directory,
        tn.Session(
            policy_text, groups=["values", "evidence"], policy_id="internal_report.policy",
        ) as session,
        open_publication_store(Path(directory) / "publications") as store,
    ):
        policy = session.policy("internal.report")
        recorder = EvidenceRecorder(session, store, session.policy("operation.records"))
        rules = Rules(
            writers=frozenset({session.did}),
            policies=(policy,),
            operations=frozenset({
                "select", "attest", "upper", "add", "concat", "to_string", "reveal",
            }),
            destinations=frozenset({"internal-console"}),
        )

        def evaluate(request):
            if request.application != "local-report" or request.purpose != "internal-report":
                return Assessment(False, "Application or purpose is not permitted.")
            decision = rules(request)
            return replace(decision, obligations=("monitor",)) if decision.permit else decision

        def accept_metadata(arrived):
            return (
                arrived.writer in rules.writers
                and arrived.object_type == VALUE_TYPE
                and bool(arrived.policies)
                and all(item.matches_contract(policy) for item in arrived.policies)
            )

        context = Context(
            session, recorder, evaluate,
            application="local-report", purpose="internal-report", actor="analyst",
            metadata_accept=accept_metadata,
        )
        name = originate(context, "ACME", policy, "company name")
        amount = originate(context, Decimal("12.50"), policy, "amount")
        with context:
            name = name.attest("This name identifies the example company.", context)
            total = amount + Decimal("0.25")
            report = name.upper() + ": " + total.as_string()
        text = report.reveal(context, destination="internal-console")
        records = verify_tape(
            recorder.records(), session=session, policy=recorder.policy,
            expected_writer=session.did, expected_context=recorder.context_id,
            expected_head=recorder.head,
        )
        print(text)
        print(f"Verified {len(records)} signed records.")


if __name__ == "__main__":
    main()
