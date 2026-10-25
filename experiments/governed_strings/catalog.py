"""Live Unity locations plus native signed, source/request-specific assignments.

The Java catalog stores locations. This adapter verifies its signed assignment,
admits revisions into native PolicyDag, and evaluates the declared request rules.
Unity itself does not acquire a TN signing key or decrypt business content.
"""

from dataclasses import dataclass
import io
import json
from pathlib import Path
from urllib.request import url2pathname
from urllib.parse import urlparse

import tn
from tn_polars.registry import UnityCatalogRegistry, write_edition_bundle

from deepvest_tn.publications import write_once
from deepvest_tn.unity import CatalogTrust, FinanceEditionRequest, resolve_finance_editions

from .primitives import OPERATIONS, VALUE_TYPE
from .runtime import Assessment
from .string import OperationDenied


def markdown(column):
    return f"""## {VALUE_TYPE}
### instruction
The {column} field of customer_data for the configured local SQL demonstration.
### use_for
Customer support computations by the assigned application and actor, with monitoring.
### do_not_use_for
Marketing, public disclosure, or removal of contributing claims and source lineage.
### consequences
Retain the writer, exact incarnation, assigned rules, and operation evidence.
### on_violation_or_error
Record refusal or failure before releasing any unapproved result.
"""


@dataclass(frozen=True)
class Resolution:
    assignment_id: str
    table: str
    policies: dict
    rules: dict
    columns: dict


class AssignedRules:
    def __init__(self, resolver, *, actor, application, purpose, writers):
        self.resolver, self.actor = resolver, actor
        self.application, self.purpose, self.writers = application, purpose, frozenset(writers)

    def accepts_metadata(self, ctx):
        return (ctx.object_type == VALUE_TYPE and ctx.writer in self.writers
                and bool(ctx.policies) and all(
                    p.revision_id in self.resolver.policies
                    and p.matches_contract(self.resolver.policies[p.revision_id])
                    for p in ctx.policies))

    def __call__(self, request):
        reasons, obligations = [], []
        if any(writer not in self.writers for writer in request.writers):
            reasons.append("unrecognized publishing endpoint")
        if not request.policies:
            reasons.append("no assigned source governance")
        for policy in request.policies:
            rule = self.resolver.rules.get(policy.revision_id)
            if rule is None or not policy.matches_contract(self.resolver.policies[policy.revision_id]):
                reasons.append("policy is not an admitted Unity assignment")
                continue
            if (self.actor not in rule["actors"] or self.application != rule["application"]
                    or self.purpose not in rule["purposes"]):
                reasons.append("requester/application/purpose is outside source assignment")
            if request.operation not in rule["operations"]:
                reasons.append(f"source assignment does not permit {request.operation}")
            if request.operation == "reveal" and request.destination not in rule["destinations"]:
                reasons.append("destination is outside source assignment")
            if request.operation == "model_handoff":
                model = rule.get("model")
                if (not isinstance(model, dict) or request.destination not in model.get("destinations", [])
                        or not model.get("required_disclosure")):
                    reasons.append("source assignment does not grant this model handoff with disclosure")
            obligations.extend(rule["obligations"])
        return Assessment(not reasons, "; ".join(dict.fromkeys(reasons)) or "all source assignments permit this request",
                          tuple(dict.fromkeys(obligations)),
                          tuple((c.id, "informative") for c in request.claims))


class UnityGovernance:
    def __init__(self, url, catalog, *, session, authority, administration, store):
        self.registry = UnityCatalogRegistry(url, catalog, "governance")
        self.session, self.authority, self.administration, self.store = session, authority, administration, store
        self.policy_dag = tn.PolicyDag()
        self.policies, self.rules = {}, {}
        self.assignments = {}
        self.requests = []
        self.trust = CatalogTrust(
            lambda ctx: ctx.writer == authority and ctx.governance.matches_contract(administration),
            lambda revision, parent: revision.writer == authority,
            lambda edition: edition.writer == authority,
            tn.UseContext("catalog", "governance-resolution", "read"),
        )

    def _open(self, obj, group):
        admitted = self.session.governance(obj).accept(
            use=self.trust.metadata_use, groups=[group], decide=self.trust.metadata)
        return self.session.open(admitted, [group])

    def publish_assignment(self, directory, *, table, source_identity, application, actors, purposes,
                           columns=None):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=False)
        fields, revisions = {}, []
        columns = columns or {
            "name": {"kind": "str", "markdown": markdown("name")},
            "value": {"kind": "int", "markdown": markdown("value"),
                      "extend_markdown": markdown("value").replace("with monitoring", "with a signed pre-operation monitoring event")},
        }
        for column, definition in columns.items():
            kind = definition["kind"]
            if kind not in {"str", "int", "float", "Decimal", "complex"}:
                raise TypeError("source assignment requires a supported primitive type")
            scope = table + "." + column
            draft = tn.PolicyRevisionDraft.from_markdown(
                self.authority, definition["markdown"], scope, VALUE_TYPE, scope)
            obj = self.session.seal(draft.into_draft(self.administration))
            revision = tn.PolicyRevision.from_opened(self._open(obj, "policy_revision"))
            self.policy_dag.admit(revision, self.trust.policy_revision)
            revisions.append(revision)
            if definition.get("extend_markdown"):
                # A real signed policy-history edge, resolved before data origination.
                extended = tn.PolicyRevisionDraft.from_markdown(
                    self.authority, definition["extend_markdown"],
                    scope, VALUE_TYPE, scope).parent(revision.id, tn.PolicyRelation.Extend)
                obj = self.session.seal(extended.into_draft(self.administration))
                revision = tn.PolicyRevision.from_opened(self._open(obj, "policy_revision"))
                self.policy_dag.admit(revision, self.trust.policy_revision)
                revisions.append(revision)
            rule = {"actors": actors, "application": application, "purposes": purposes,
                    "operations": sorted(OPERATIONS), "destinations": ["support-console"], "obligations": ["monitor"]}
            rule.update(definition.get("rule", {}))
            fields[column] = {"kind": kind, "revision": revision.id, "scope": scope, "rule": rule}
        body = {"schema": "tn.sql-assignment.v1", "table": table, "source": source_identity,
                "columns": fields, "revisions": [r.id for r in revisions]}
        data = self.session.create({"assignment": body}, self.administration,
                                   object_type="experiment.assignment", group="catalog_cfg")
        self.store.put(data.snapshot)
        obj = self.session.release(data, use=tn.UseContext("catalog", "governance-resolution", "assign"),
                                   to="sql-gateway", decide=lambda ctx: ctx.writer == self.authority)
        self.store.put(obj)
        write_once(directory / "assignment.tn", obj.forward())
        self.assignments[obj.id] = body
        for revision in revisions:
            self.store.put(revision.object)
            write_once(directory / (revision.id[7:] + ".tn"), revision.object.forward())
        self.registry._ensure("/catalogs", {"name": self.registry.catalog})
        self.registry._ensure("/schemas", {"name": self.registry.schema, "catalog_name": self.registry.catalog})
        self.registry._ensure("/volumes", {"catalog_name": self.registry.catalog,
            "schema_name": self.registry.schema, "name": "source_" + table,
            "volume_type": "EXTERNAL", "storage_location": directory.as_uri(),
            "comment": "Signed TN source/request assignments for governed Python primitives"})
        return obj.id

    def resolve(self, table, columns, *, actor, application, purpose, expected_source):
        full = f"{self.registry.catalog}.{self.registry.schema}.source_{table}"
        volume = self.registry._request("GET", "/volumes/" + full)
        uri = urlparse(volume["storage_location"])
        if uri.scheme != "file" or uri.netloc not in ("", "localhost"):
            raise OperationDenied("this local resolver requires an enrolled local bundle")
        directory = Path(url2pathname(uri.path)).resolve()
        obj = tn.GovernedObject.read(directory / "assignment.tn")
        opened = self.session.receive(obj, use=self.trust.metadata_use,
                                      groups=["catalog_cfg"], decide=self.trust.metadata)
        body = opened.get("assignment", group="catalog_cfg")
        if body["schema"] != "tn.sql-assignment.v1" or body["table"] != table or body["source"] != expected_source:
            raise OperationDenied("Unity assignment does not match the requested SQL source")
        self.store.put(obj)
        for identity in body["revisions"]:
            rev_obj = tn.GovernedObject.read(directory / (identity[7:] + ".tn"))
            if rev_obj.id != identity:
                raise OperationDenied("catalog revision identity mismatch")
            revision = tn.PolicyRevision.from_opened(self._open(rev_obj, "policy_revision"))
            if self.policy_dag.get(identity) is None:
                self.policy_dag.admit(revision, self.trust.policy_revision)
            self.store.put(rev_obj)
        chosen = {}
        for column in columns:
            if column not in body["columns"]:
                raise OperationDenied("requested column has no source governance assignment")
            field = body["columns"][column]
            policy = self.policy_dag.select(field["revision"], field["scope"], lambda rev: rev.writer == self.authority)
            self.policies[policy.revision_id] = policy
            self.rules[policy.revision_id] = field["rule"]
            chosen[column] = policy
        self.assignments[obj.id] = body
        self.requests.append({"source": expected_source, "table": table, "columns": list(columns),
                              "actor": actor, "application": application, "purpose": purpose,
                              "assignment_id": obj.id, "unity_volume_id": volume.get("volume_id")})
        return Resolution(obj.id, table, chosen, self.rules, body["columns"])

    def publish_edition(self, value, *, policy, source, directory, application, purpose):
        obj = tn.GovernedObject.read(io.BytesIO(self.store.get(value.reference.object_id)))
        dataset = source["table"] + "." + source["column"]
        edition = source["query_id"] + "-" + str(source["row"])
        uses = [tn.UseContext(application, purpose, op) for op in [*sorted(OPERATIONS), "inspect-claims", "verify"]]
        assignment = self.assignments[source["assignment_id"]]
        extra_uses = assignment["columns"][source["column"]]["rule"].get("additional_uses", [])
        uses.extend(tn.UseContext(**use) for use in extra_uses)
        draft = tn.DatasetEditionDraft(dataset, edition, obj, ["values", "evidence"],
                    [tn.ContractBinding(policy.revision_id, dataset)], uses, source["assignment_id"], [])
        record_obj = self.session.seal(draft.into_draft(self.administration))
        record = tn.DatasetEdition.from_opened(self._open(record_obj, "dataset_edition"))
        self.store.put(record_obj)
        bundle = write_edition_bundle(directory, dataset=dataset, edition=edition, record_id=record.id,
            edition_wire=record_obj.forward(),
            revision_wires=[(r, self.store.get(r)) for r in assignment["revisions"]], evaluators={})
        self.registry.register_edition(dataset, edition, record.id, bundle)
        return {"dataset": dataset, "edition": edition, "record_id": record.id,
                "source_object_id": obj.id}

    def admit_edition(self, info, context):
        request = FinanceEditionRequest(info["dataset"], info["edition"], info["record_id"],
                                        tn.UseContext(context.application, context.purpose, "inspect-claims"))
        resolved = resolve_finance_editions([request], catalog_session=context.session,
            policy_dag=self.policy_dag, trust=self.trust, registry=self.registry,
            catalog=context.catalog, metadata_store=self.store)[0]
        context.editions[info["source_object_id"]] = resolved.selection.record
        return resolved.selection
