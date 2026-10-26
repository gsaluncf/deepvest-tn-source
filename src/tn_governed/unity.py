"""Build and publish the two governed portfolio input feeds."""

from __future__ import annotations

import json
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Callable, Mapping, Sequence

import polars as pl
import tn
from tn_polars.build import build_opa_wasm
from tn_polars.compiler import candidate_to_policy
from tn_polars.policy import CompiledPolicy
from tn_polars.registry import UnityCatalogRegistry, verified_edition_entry, write_edition_bundle
from tn_polars.release import PolicyDenied

POSITIONS_FEED = "portfolio.internal.positions"
MARKET_PRICES_FEED = "portfolio.market.prices"
FEED_NAMES = (POSITIONS_FEED, MARKET_PRICES_FEED)
POLICY_REF = (
    ".tn/config/agents.md#portfolio.calculation.calculated@1#"
    "sha256:398b235838f609ff6dbb7d5c66847935dccb27cc0fc61b857c9727988884a9d7"
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_ROOT = _PROJECT_ROOT / "fixtures" / "portfolio"
_REQUIRED_BUNDLE_FILES = {
    "compiled_policy.json",
    "tn_governance.json",
    "profile.json",
    "policy.wasm",
    "data.json",
}
_FEEDS = {
    POSITIONS_FEED: {
        "source": _FIXTURE_ROOT / "internal_positions.ndjson",
        "fields": {
            "account_id": "confidential", "portfolio_id": "confidential", "symbol": "internal",
            "quantity": "confidential", "cost_basis": "confidential", "currency": "internal",
        },
    },
    MARKET_PRICES_FEED: {
        "source": _FIXTURE_ROOT / "market_prices.ndjson",
        "fields": {
            "symbol": "internal", "as_of": "internal", "last_price": "internal",
            "currency": "internal", "market_source": "internal",
        },
    },
}


@dataclass(frozen=True)
class UnityCatalogConfig:
    """Connection settings for the local Unity Catalog service."""

    base_url: str = "http://127.0.0.1:8080"
    catalog: str = "portfolio"
    schema: str = "finance"
    token: str | None = None


@dataclass(frozen=True)
class PublishedFinanceFeeds:
    """The catalog feed names and local artifacts published during setup."""

    feed_names: tuple[str, str]
    positions_bundle: Path
    market_prices_bundle: Path


@dataclass(frozen=True)
class CatalogTrust:
    """Application decisions applied to native metadata and authority snapshots."""

    metadata: Callable
    policy_revision: Callable
    edition: Callable
    metadata_use: tn.UseContext


@dataclass(frozen=True)
class FinanceEditionRequest:
    dataset: str
    edition: str
    record_id: str
    use: tn.UseContext


@dataclass(frozen=True)
class ResolvedFinanceEdition:
    selection: tn.DatasetSelection
    catalog: tn.DatasetCatalog
    entries: Mapping[str, object]


@dataclass(frozen=True)
class PublishedDatasetEdition:
    creation: tn.GovernedObject
    source: tn.GovernedObject
    record: tn.DatasetEdition
    bundle: Path

    def request(self, use: tn.UseContext) -> FinanceEditionRequest:
        return FinanceEditionRequest(self.record.dataset, self.record.edition, self.record.id, use)


def originate_policy_revision(
    *, session, policy_dag, markdown, policy_id, object_type, scope,
    administration_contract, trust: CatalogTrust,
):
    """Originate and admit a native revision using the application's authority decisions."""
    object = session.seal(tn.PolicyRevisionDraft.from_markdown(
        session.did, markdown, policy_id, object_type, scope,
    ).into_draft(administration_contract))
    admitted = session.governance(object).accept(
        use=trust.metadata_use, groups=["policy_revision"], decide=trust.metadata,
    )
    revision = tn.PolicyRevision.from_opened(session.open(admitted, ["policy_revision"]))
    policy_dag.admit(revision, trust.policy_revision)
    return revision, policy_dag.select(revision.id, scope, lambda accepted: True)


def publish_dataset_edition(
    *, dataset, edition, rows, compiled_policy: CompiledPolicy, session, policy, policy_dag, revisions,
    eligible_uses, grant_ref, administration_contract, trust: CatalogTrust,
    publish_use, publish_decide, destination, publication_store, output_dir,
    registry=None, allowed_operations=None, audiences=None,
    group="finance", object_type="portfolio.feed.data.v1",
) -> PublishedDatasetEdition:
    """Release one native source, retain it, and publish its signed edition.

    The policy and revisions are accepted native inputs. The edition names the
    final release, and every evaluator file is hashed before its metadata seal.
    """
    if not isinstance(compiled_policy, CompiledPolicy):
        raise TypeError("publication requires an approved CompiledPolicy")
    compiled = CompiledPolicy.from_dict(compiled_policy.to_dict())
    if compiled.policy_ref != policy.policy_ref:
        raise ValueError("compiled policy reference differs from accepted native policy")
    revision = policy_dag.get(policy.revision_id) if policy.revision_id else None
    if revision is None:
        raise ValueError("publication policy must name an accepted native revision")
    selected_revision = policy_dag.resolve(policy, revision.scope)
    revisions = tuple(revisions)
    if selected_revision.id not in {revision.id for revision in revisions}:
        raise ValueError("publication needs the selected revision in its retained metadata")
    retained_ids = set()
    for revision in revisions:
        accepted = policy_dag.get(revision.id)
        if accepted is None:
            raise ValueError("publication metadata contains an unaccepted policy revision")
        if revision.id in retained_ids or revision.object.wire != accepted.object.wire:
            raise ValueError("publication revision metadata must contain each exact accepted wire once")
        if any(parent.revision_id not in retained_ids for parent in revision.parents):
            raise ValueError("publication revision metadata requires every parent before its child")
        retained_ids.add(revision.id)
    if isinstance(rows, Sequence) and any(
        not isinstance(row, Mapping) or set(row) != set(compiled.fields) for row in rows
    ):
        raise ValueError("evaluator fields must describe every source row exactly")
    frame = pl.DataFrame(rows)
    if set(frame.columns) != set(compiled.fields):
        raise ValueError("evaluator fields must describe every source column exactly")
    data = session.create_obj(
        {"feed_name": dataset, "rows": frame.to_dicts()}, policy,
        object_type=object_type, group=group,
    )
    creation = data.snapshot
    source = data.release(use=publish_use, to=destination, decide=publish_decide)
    with tempfile.TemporaryDirectory(prefix="tn-edition-evaluator-") as temporary:
        build = Path(temporary)
        build_opa_wasm(compiled, build, operations=allowed_operations, audiences=audiences)
        files = {
            "compiled_policy.json": json.dumps(compiled.to_dict(), sort_keys=True).encode(),
            "profile.json": json.dumps({"dtypes": {
                name: str(dtype) for name, dtype in frame.schema.items()
            }}, sort_keys=True).encode(),
            "policy.wasm": (build / "policy.wasm").read_bytes(),
            "data.json": (build / "data.json").read_bytes(),
        }
    artifacts = tn.EvaluatorArtifactSet(
        selected_revision.id, *(hashlib.sha256(files[name]).hexdigest() for name in (
            "compiled_policy.json", "profile.json", "policy.wasm", "data.json",
        )),
    )
    object = session.seal(tn.DatasetEditionDraft(
        dataset, edition, source, [group], [tn.ContractBinding(selected_revision.id, selected_revision.scope)],
        eligible_uses, grant_ref, [artifacts],
    ).into_draft(administration_contract))
    admitted = session.governance(object).accept(
        use=trust.metadata_use, groups=["dataset_edition"], decide=trust.metadata,
    )
    record = tn.DatasetEdition.from_opened(session.open(admitted, ["dataset_edition"]))
    catalog = tn.DatasetCatalog()
    catalog.admit(record, policy_dag, trust.edition)
    for accepted in [*(revision.object for revision in revisions), creation, source, object]:
        publication_store.put(accepted)
    bundle = write_edition_bundle(
        output_dir, dataset=dataset, edition=edition, record_id=record.id,
        edition_wire=object.wire.encode(),
        revision_wires=[(revision.id, revision.object.wire.encode()) for revision in revisions],
        evaluators={selected_revision.id: files},
    )
    if registry is not None:
        registry.register_edition(dataset, edition, record.id, bundle)
    return PublishedDatasetEdition(creation, source, record, bundle)


def resolve_finance_editions(
    requests: Sequence[FinanceEditionRequest], *, catalog_session, policy_dag,
    trust: CatalogTrust, registry=None, config=None, catalog=None, metadata_store=None,
) -> tuple[ResolvedFinanceEdition, ...]:
    """Resolve exact signed records, with native admission before metadata parsing.

    Unity locates bytes. The native policy DAG and dataset catalog determine
    accepted authority, complete contracts, and eligibility for the requested use.
    """
    registry = registry if registry is not None else unity_catalog_registry(config)
    catalog = catalog if catalog is not None else tn.DatasetCatalog()
    results = []
    for request in requests:
        transport = registry.get_edition(request.dataset, request.edition, request.record_id)
        retained = []
        for identity, wire in transport.revision_wires:
            object = _retained_metadata(identity, wire, metadata_store)
            if object.id != identity:
                raise PolicyDenied("policy revision wire differs from its requested identity")
            admitted = catalog_session.governance(object).accept(
                use=trust.metadata_use, groups=["policy_revision"], decide=trust.metadata,
            )
            revision = policy_dag.get(identity)
            if revision is None:
                revision = tn.PolicyRevision.from_opened(
                    catalog_session.open(admitted, ["policy_revision"])
                )
                policy_dag.admit(revision, trust.policy_revision)
            elif revision.object.wire != object.wire:
                raise PolicyDenied("revision differs from its exact accepted publication")
            retained.append(object)
        object = _retained_metadata(request.record_id, transport.edition_wire, metadata_store)
        if object.id != request.record_id:
            raise PolicyDenied("dataset edition wire differs from its requested identity")
        admitted = catalog_session.governance(object).accept(
            use=trust.metadata_use, groups=["dataset_edition"], decide=trust.metadata,
        )
        record = catalog.get(request.record_id)
        if record is None:
            record = tn.DatasetEdition.from_opened(
                catalog_session.open(admitted, ["dataset_edition"])
            )
            catalog.admit(record, policy_dag, trust.edition)
        elif record.object.wire != object.wire:
            raise PolicyDenied("edition differs from its exact accepted publication")
        selection = catalog.select(
            request.dataset, request.edition, request.record_id, request.use,
        )
        expected = {artifact.policy_revision: artifact for artifact in record.evaluator_artifacts}
        if set(transport.evaluator_paths) != set(expected):
            raise PolicyDenied("edition evaluator artifacts differ from its complete signed set")
        scopes = {binding.revision_id: binding.scope for binding in record.contracts}
        entries = {}
        for identity, artifact in expected.items():
            policy = policy_dag.select(identity, scopes[identity], lambda revision: True)
            entry = verified_edition_entry(
                request.dataset, policy.fields, identity, artifact,
                transport.evaluator_paths[identity],
            )
            if entry.policy.policy_ref != policy.policy_ref:
                raise PolicyDenied("compiled evaluator policy differs from its native revision")
            entries[identity] = entry
        if metadata_store is not None:
            for accepted in [*retained, object]:
                metadata_store.put(accepted)
        results.append(ResolvedFinanceEdition(selection, catalog, MappingProxyType(entries)))
    return tuple(results)


def _retained_metadata(identity, wire, store):
    if store is not None:
        try:
            retained = store.get(identity)
        except FileNotFoundError:
            pass
        else:
            if retained != wire:
                raise PolicyDenied("catalog metadata differs from its exact retained publication")
            wire = retained
    return tn.GovernedObject.parse(wire)


def unity_catalog_registry(config: UnityCatalogConfig | None = None) -> UnityCatalogRegistry:
    """Create the sole production registry adapter for this application."""
    settings = config or UnityCatalogConfig()
    return UnityCatalogRegistry(
        settings.base_url,
        settings.catalog,
        settings.schema,
        token=settings.token,
    )


def build_finance_feed_bundles(
    output_dir: Path, *, governance_path: Path | None = None
) -> dict[str, Path]:
    """Create the exact five-artifact governed bundle for each fixed feed."""
    governance = _load_governance(governance_path or _FIXTURE_ROOT / "tn_governance.json")
    if governance.get("policy") != POLICY_REF:
        raise ValueError("both finance feeds require the common policy reference")
    bundles: dict[str, Path] = {}
    for feed_name in FEED_NAMES:
        feed = _FEEDS[feed_name]
        bundle = Path(output_dir) / feed_name
        bundle.mkdir(parents=True, exist_ok=True)
        _write_bundle(
            bundle,
            _compiled_policy(governance, feed["fields"]),
            governance,
            feed["source"],
        )
        bundles[feed_name] = bundle
    return bundles


def publish_finance_feeds(
    output_dir: Path, *, config: UnityCatalogConfig | None = None
) -> PublishedFinanceFeeds:
    """Build and register both fixed feeds through Unity Catalog."""
    bundles = build_finance_feed_bundles(output_dir)
    catalog = unity_catalog_registry(config)
    for feed_name in FEED_NAMES:
        catalog.register_feed(feed_name, bundles[feed_name])
    return PublishedFinanceFeeds(FEED_NAMES, bundles[POSITIONS_FEED], bundles[MARKET_PRICES_FEED])


def resolve_finance_feeds(config: UnityCatalogConfig | None = None) -> dict[str, object]:
    """Resolve both input feeds from Unity for the portfolio workflow."""
    catalog = unity_catalog_registry(config)
    entries = {feed_name: catalog.get(feed_name) for feed_name in FEED_NAMES}
    _validate_resolved_feeds(entries)
    return entries


def _load_governance(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compiled_policy(governance: dict, fields: dict[str, str]):
    candidate = {
        "schema": "tn.polars-policy-candidate.v1", "policy_ref": governance["policy"],
        "fields": fields, "release": {"public": {"allow_aggregates": False, "min_group_size": 3}},
    }
    return candidate_to_policy(candidate, governance, set(fields))


def _validate_resolved_feeds(entries: dict[str, object]) -> None:
    """Reject catalog locations that do not resolve to their declared feed bundle."""
    positions = entries[POSITIONS_FEED]
    prices = entries[MARKET_PRICES_FEED]
    if positions.policy.fields != _FEEDS[POSITIONS_FEED]["fields"]:
        raise ValueError(f"{POSITIONS_FEED} resolved with unexpected fields")
    if prices.policy.fields != _FEEDS[MARKET_PRICES_FEED]["fields"]:
        raise ValueError(f"{MARKET_PRICES_FEED} resolved with unexpected fields")
    if positions.policy.policy_ref != POLICY_REF or prices.policy.policy_ref != POLICY_REF:
        raise ValueError("resolved finance feeds do not share the required policy reference")
    if positions.tn_governance != prices.tn_governance:
        raise ValueError("resolved finance feeds do not share one governance body")


def _write_bundle(bundle: Path, policy, governance: dict, source: Path) -> None:
    build_opa_wasm(policy, bundle)
    schema = pl.scan_ndjson(source).collect_schema()
    (bundle / "compiled_policy.json").write_text(
        json.dumps(policy.to_dict(), indent=2), encoding="utf-8"
    )
    (bundle / "tn_governance.json").write_text(
        json.dumps(governance, indent=2), encoding="utf-8"
    )
    (bundle / "profile.json").write_text(
        json.dumps(
            {"dtypes": {name: str(dtype) for name, dtype in schema.items()}}, indent=2
        ),
        encoding="utf-8",
    )
    for generated_name in ("tn_polars.rego", "bundle.tar.gz"):
        (bundle / generated_name).unlink(missing_ok=True)
    if {path.name for path in bundle.iterdir()} != _REQUIRED_BUNDLE_FILES:
        raise RuntimeError(
            f"bundle {bundle} did not contain exactly the required registry artifacts"
        )
