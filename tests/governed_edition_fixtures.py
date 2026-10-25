"""Ordinary signed native records and real compiled evaluator artifacts for tests."""

import hashlib
import json
from dataclasses import dataclass

import tn
from tn_polars.build import build_opa_wasm
from tn_polars.policy import CompiledPolicy
from tn_polars.registry import write_edition_bundle

from test_publications import POLICY

BUSINESS_USE = tn.UseContext("analytics", "portfolio_analysis", "join_prices")
METADATA_USE = tn.UseContext("catalog", "metadata_resolution", "read")


@dataclass
class SignedFixture:
    session: object
    source: object
    revision: object
    record: object
    selection: object
    bundle: object
    files: dict


def signed_edition(root, *, session=None, dataset="market.prices", edition="old", uses=None):
    session = session or tn.Session(POLICY, groups=["finance", "policy_revision", "dataset_edition"])
    policy_text = POLICY.replace("Portfolio analysis.", f"Portfolio analysis edition {edition}.")
    revision_object = session.seal(
        tn.PolicyRevisionDraft.from_markdown(
            session.did, policy_text, "agents.md", "portfolio.feed.data.v1", "finance",
        ).into_draft(session.policy("portfolio.feed.data.v1"))
    )
    admitted = session.governance(revision_object).accept(
        use=METADATA_USE, groups=["policy_revision"], decide=lambda context: True,
    )
    revision = tn.PolicyRevision.from_opened(session.open(admitted, ["policy_revision"]))
    dag = tn.PolicyDag()
    dag.admit(revision, lambda record, parent: True)
    policy = dag.select(revision.id, "finance", lambda record: True)
    data = session.create_obj(
        {"feed_name": dataset, "rows": [{"amount": 42}]}, policy,
        object_type="portfolio.feed.data.v1", group="finance",
    )
    source = data.release(
        use=tn.UseContext("source", "portfolio_analysis", "publish"),
        to="analytics", decide=lambda context: True,
    )
    compiled = CompiledPolicy.from_dict({
        "schema": "tn.polars-policy.v1", "policy_ref": policy.policy_ref,
        "fields": {"amount": "internal"}, "release": {},
    })
    build = root / "build"
    build_opa_wasm(compiled, build, operations=["load", "select", "join", "release"])
    files = {
        "compiled_policy.json": json.dumps(compiled.to_dict()).encode(),
        "profile.json": json.dumps({"dtypes": {"amount": "Int64"}}).encode(),
        "policy.wasm": (build / "policy.wasm").read_bytes(),
        "data.json": (build / "data.json").read_bytes(),
    }
    artifacts = tn.EvaluatorArtifactSet(
        revision.id, *(hashlib.sha256(files[name]).hexdigest() for name in (
            "compiled_policy.json", "profile.json", "policy.wasm", "data.json",
        )),
    )
    record_object = session.seal(tn.DatasetEditionDraft(
        dataset, edition, source, ["finance"], [tn.ContractBinding(revision.id, "finance")],
        uses or [BUSINESS_USE], "grant:test", [artifacts],
    ).into_draft(session.policy("portfolio.feed.data.v1")))
    admitted = session.governance(record_object).accept(
        use=METADATA_USE, groups=["dataset_edition"], decide=lambda context: True,
    )
    record = tn.DatasetEdition.from_opened(session.open(admitted, ["dataset_edition"]))
    catalog = tn.DatasetCatalog()
    catalog.admit(record, dag, lambda record: True)
    selection = catalog.select(dataset, edition, record.id, (uses or [BUSINESS_USE])[0])
    bundle = write_edition_bundle(
        root / "bundle", dataset=dataset, edition=edition, record_id=record.id,
        edition_wire=record_object.wire.encode(),
        revision_wires=[(revision.id, revision_object.wire.encode())],
        evaluators={revision.id: files},
    )
    return SignedFixture(session, source, revision, record, selection, bundle, files)
