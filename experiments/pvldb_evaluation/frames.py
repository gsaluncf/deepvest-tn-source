"""SQL cells through governed Polars to one governed numeric result.

This experiment adapter implements one protected ``dataframe_calculate`` call.
Its intermediate lazy frames retain the same native working object and complete
contracts. Only the final numeric result is published. Native TN,
catalog admission, OPA evaluation, Polars lineage, and Context evidence perform
their usual work; this module supplies the cell-to-frame application glue.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
from uuid import uuid4

import polars as pl
import tn
from tn_polars.build import build_opa_wasm
from tn_polars.frame import GovernedLazyFrame
from tn_polars.policy import CompiledPolicy
from tn_polars.registry import write_edition_bundle
from tn_polars.support import Support
from tn_polars.tn_boundary import PolicySet, TnBoundary, root_name

from deepvest_tn.unity import FinanceEditionRequest, resolve_finance_editions
from experiments.governed_strings.catalog import UnityGovernance, markdown
from experiments.governed_strings.evidence import EvidenceError
from experiments.governed_strings.numeric import TNNumeric
from experiments.governed_strings.numeric_codec import record_value
from .numeric_profile import numeric_profile, polars_dtype, build_frame, check_fixed_plan_range
from experiments.governed_strings.primitives import Context, OPERATIONS
from experiments.governed_strings.string import OperationDenied, TNString, _active


OPERATION = "dataframe_calculate"
FRAME_OPERATIONS = ("load", "filter", "sort", "with_columns", "select", "release")
ARTIFACT_NAMES = ("compiled_policy.json", "profile.json", "policy.wasm", "data.json")
PLAN = (
    {"operation": "filter", "column": "value", "comparison": ">=", "literal": 0},
    {"operation": "sort", "columns": ["name"], "descending": False},
    {"operation": "with_columns", "column": "value", "operator": "+",
     "literal": 250, "output": "adjusted"},
    {"operation": "select", "aggregate": "sum", "column": "adjusted", "output": "amount"},
)


class DataframeGovernance(UnityGovernance):
    """Issue the experiment's source assignments and signed OPA artifact sets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_artifacts = {}

    def publish_assignment(self, directory, *, columns=None, decimal_precision=38, decimal_scale=None, **kwargs):
        columns = deepcopy(columns) if columns is not None else {
            "name": {"kind": "str", "markdown": markdown("name")},
            "value": {"kind": "int", "markdown": markdown("value")},
        }
        if set(columns) != {"name", "value"}:
            raise ValueError("this dataframe workload requires the name/value source schema")
        if columns["name"]["kind"] != "str":
            raise TypeError("name must be a string column")
        profile = numeric_profile(columns["value"]["kind"], decimal_precision, decimal_scale)
        for definition in columns.values():
            rule = definition.setdefault("rule", {})
            rule["dataframe_numeric"] = profile
            rule["operations"] = sorted(set(rule.get("operations", OPERATIONS)) | {OPERATION})
        return super().publish_assignment(directory, columns=columns, **kwargs)

    def _evaluator_artifacts(self, policy, application, profile):
        key = (policy.revision_id, application, json.dumps(profile, sort_keys=True))
        if key not in self._frame_artifacts:
            compiled = CompiledPolicy.from_dict({
                "schema": "tn.polars-policy.v1", "policy_ref": policy.policy_ref,
                "fields": {"name": "internal", "value": "internal"},
                "release": {"public": {"allow_aggregates": False, "min_group_size": 2}},
            })
            with tempfile.TemporaryDirectory(prefix="tn-pvldb-evaluator-") as temporary:
                build = Path(temporary)
                build_opa_wasm(compiled, build, operations=FRAME_OPERATIONS,
                               audiences={application: "internal", "governed-runtime": "internal"})
                files = {
                    "compiled_policy.json": json.dumps(compiled.to_dict(), sort_keys=True).encode(),
                    "profile.json": json.dumps({"dtypes": {"name": "String", "value": profile["dtype"]}},
                                               sort_keys=True).encode(),
                    "policy.wasm": (build / "policy.wasm").read_bytes(),
                    "data.json": (build / "data.json").read_bytes(),
                }
            artifacts = tn.EvaluatorArtifactSet(policy.revision_id, *(
                hashlib.sha256(files[name]).hexdigest() for name in ARTIFACT_NAMES))
            self._frame_artifacts[key] = artifacts, files
        return self._frame_artifacts[key]

    def publish_edition(self, value, *, policy, source, directory, application, purpose):
        obj = tn.GovernedObject.read(io.BytesIO(self.store.get(value.reference.object_id)))
        dataset = source["table"] + "." + source["column"]
        edition = source["query_id"] + "-" + str(source["row"])
        uses = [tn.UseContext(application, purpose, op)
                for op in sorted(OPERATIONS | {OPERATION, "inspect-claims", "verify"})]
        assignment = self.assignments[source["assignment_id"]]
        rule = assignment["columns"][source["column"]]["rule"]
        uses.extend(tn.UseContext(**use) for use in rule.get("additional_uses", []))
        artifacts, files = self._evaluator_artifacts(policy, application, rule["dataframe_numeric"])
        draft = tn.DatasetEditionDraft(dataset, edition, obj, ["values", "evidence"],
            [tn.ContractBinding(policy.revision_id, dataset)], uses, source["assignment_id"], [artifacts])
        record_obj = self.session.seal(draft.into_draft(self.administration))
        record = tn.DatasetEdition.from_opened(self._open(record_obj, "dataset_edition"))
        self.store.put(record_obj)
        bundle = write_edition_bundle(directory, dataset=dataset, edition=edition, record_id=record.id,
            edition_wire=record_obj.forward(),
            revision_wires=[(r, self.store.get(r)) for r in assignment["revisions"]],
            evaluators={policy.revision_id: files})
        # Source assignment publication and resolution have established this catalog and schema.
        self.registry.register_edition(dataset, edition, record.id, bundle, create_namespace=False)
        return {"dataset": dataset, "edition": edition, "record_id": record.id,
                "source_object_id": obj.id}

    def admit_edition(self, info, context):
        request = FinanceEditionRequest(info["dataset"], info["edition"], info["record_id"],
            tn.UseContext(context.application, context.purpose, "inspect-claims"))
        resolved = resolve_finance_editions([request], catalog_session=context.session,
            policy_dag=self.policy_dag, trust=self.trust, registry=self.registry,
            catalog=context.catalog, metadata_store=self.store)[0]
        context.editions[info["source_object_id"]] = resolved.selection.record
        context.frame_editions[info["source_object_id"]] = resolved
        return resolved.selection


class DataframeContext(Context):
    """Attach dataframe facts to the signed operation Result record."""

    def __init__(self, *args, **kwargs):
        self.frame_editions = {}
        self.frame_runs = []
        self._frame_pending = {}
        super().__init__(*args, **kwargs)

    def _append(self, kind, request, call_id, details):
        facts = self._frame_pending.get(call_id)
        if facts is not None and kind in {"Result", "Refusal", "Failure"}:
            details = {**details, "dataframe": deepcopy(facts)}
        return super()._append(kind, request, call_id, details)


class _RecordingEvaluator:
    """Record the request and actual OPA answer without changing evaluation."""

    def __init__(self, evaluator, records):
        self.evaluator, self.records = evaluator, records

    def decide(self, request):
        decision = self.evaluator.decide(request)
        self.records.append({"request": deepcopy(request), "decision_id": decision.decision_id,
                             "allow": decision.allow, "reason": decision.reason})
        return decision


def _artifact_facts(artifacts):
    return {name: getattr(artifacts, name) for name in (
        "policy_revision", "compiled_policy_sha256", "profile_sha256", "wasm_sha256", "data_sha256")}


def _select_policies(context, metadata, values, use):
    """Resolve each cell's original edition, including attested descendants."""
    policy_set = None
    sources = {"name": set(), "value": set()}
    editions, inputs = {}, []
    for index, ((data, primitive_binding, _), value) in enumerate(zip(metadata, values)):
        column = "name" if index % 2 == 0 else "value"
        bindings = data.dataset_bindings
        if len(bindings) != 1:
            raise EvidenceError("each input cell must retain exactly one original SQL edition")
        binding = bindings[0]
        context.catalog.verify_binding(binding, context.policy_dag)
        resolved = context.frame_editions.get(binding.source_object_id)
        if resolved is None:
            raise EvidenceError("the cell's original evaluator edition was not resolved")
        record = resolved.selection.record
        if (record.id != binding.edition_record_id or record.dataset != binding.dataset
                or record.edition != binding.edition or not record.dataset.endswith("." + column)):
            raise EvidenceError("cell position differs from its signed source edition")
        if len(primitive_binding["origins"]) != 1 or primitive_binding["origins"][0]["column"] != column:
            raise EvidenceError("cell position differs from its retained SQL origin")
        selection = context.catalog.select(record.dataset, record.edition, record.id, use)
        current = PolicySet.for_selection(resolved.entries, selection)
        policy_set = current if policy_set is None else policy_set.include(current)
        root = root_name(binding.source_object_id, column)
        sources[column].add(root)
        editions[record.id] = {"dataset": record.dataset, "edition": record.edition,
            "record_id": record.id, "source_object_id": record.source_object_id,
            "contracts": [item.revision_id for item in record.contracts]}
        inputs.append({"row": index // 2, "column": column, "reference": value.reference.to_dict(),
                       "source_object_id": binding.source_object_id, "edition_record_id": record.id,
                       "root": root})
    return policy_set, sources, list(editions.values()), inputs


def _row_support(lazy, cells, policies, row_count):
    """Bind row support to each cell publication, rather than one arbitrary input."""
    columns, identities = [], {}
    for cell in cells:
        source = cell["source_object_id"]
        name = "__tn_support_" + uuid4().hex
        # Each source publication contains one cell. Its local row ordinal is 0.
        rows = [[source + "::0"] if row == cell["row"] else [] for row in range(row_count)]
        columns.append(pl.Series(name, rows, dtype=pl.List(pl.String)))
        identities[name] = source
    return Support(lazy.with_columns(columns), identities, {},
                   {revision: entry.source_ids for revision, entry in policies._entries.items()})


def dataframe_calculate(context, rows):
    """Filter, sort, add 250, and sum SQL cells inside one governed operation.

    ``rows`` contains (TNString, TNNumeric) pairs. Payloads open only after source
    assignments, signed evaluator artifacts, and OPA load admission succeed.
    The result retains every input's contracts, original claims, and ancestry.
    """
    if not isinstance(context, DataframeContext) or _active.get() is not context:
        raise OperationDenied("dataframe calculation requires its active DataframeContext")
    rows = tuple(rows)
    if not rows or any(len(row) != 2 or not isinstance(row[0], TNString)
                       or not isinstance(row[1], TNNumeric) for row in rows):
        raise TypeError("dataframe calculation requires nonempty governed name/value rows")
    values = tuple(value for row in rows for value in row)
    call_id = uuid4().hex
    request, metadata, decision = context._prepare(OPERATION, values, None, (deepcopy(PLAN),), call_id)
    facts = {"schema": "tn.pvldb-dataframe.v1", "plan": deepcopy(list(PLAN)),
             "input_references": [value.reference.to_dict() for value in values],
             "source_row_count": len(rows), "opa_decisions": []}
    context._frame_pending[call_id] = facts
    try:
        use = tn.UseContext(context.application, context.purpose, OPERATION)
        policies, sources, editions, cells = _select_policies(context, metadata, values, use)
        policies.validate_contracts(request)
        policies = PolicySet({revision: replace(entry,
            evaluator=_RecordingEvaluator(entry.evaluator, facts["opa_decisions"]))
            for revision, entry in policies._entries.items()})
        facts.update(source_editions=editions, source_cells=cells,
            input_columns={column: sorted(roots) for column, roots in sources.items()},
            policy_artifacts=[_artifact_facts(entry.artifacts) for entry in policies._entries.values()])
        profiles = [context.resolver.rules[p.revision_id]["dataframe_numeric"] for p in request.policies]
        profile = profiles[0]
        polars_dtype(profile)
        if any(p != profile for p in profiles):
            raise EvidenceError("contributing contracts declare conflicting numeric profiles")
        expected_schema = {"name": "String", "value": profile["dtype"]}
        if any(entry.dtypes != expected_schema for entry in policies._entries.values()):
            raise EvidenceError("signed evaluator schema differs from the source assignment")
        if any(binding["declared"]["kind"] != ("str" if i % 2 == 0 else profile["kind"])
               for i, (_, binding, _) in enumerate(metadata)):
            raise EvidenceError("input primitive kind differs from the signed dataframe profile")
        facts["numeric_profile"] = deepcopy(profile)
        # All artifact verification and the OPA load decision precede this open.
        policies.decide("load", context.application, sources, set(), use)
        opened = context._open_values(request, values)
        data = opened[0][0]
        for other_data, _ in opened[1:]:
            data.include(other_data)
        frame_rows = [{"name": opened[i][1], "value": opened[i + 1][1]}
                      for i in range(0, len(opened), 2)]
        loaded = build_frame(frame_rows, profile)
        check_fixed_plan_range(loaded, profile)
        lazy = loaded.lazy()
        facts["input_schema"] = {name: str(dtype) for name, dtype in loaded.schema.items()}
        policies.pin(lazy)
        first = next(iter(policies._entries.values()))
        frame = GovernedLazyFrame(lazy, first.policy, {}, first.evaluator, sources=sources,
            tn_boundary=TnBoundary(data, policies, use, "values", "rows"),
            _support=_row_support(lazy, cells, policies, len(rows)))
        result = (frame.filter(pl.col("value") >= 0).sort("name")
                  .with_columns((pl.col("value") + 250).alias("adjusted")))
        result = result.select(pl.col("adjusted").sum().alias("amount"))
        materialized = result._support.frame.collect()
        result._tn.policies.decide("release", "governed-runtime", result.sources,
            result.row_sources, use, support=result._support, materialized=materialized)
        amount = materialized.select("amount").item()
        if type(amount).__name__ != profile["kind"]:
            raise TypeError("dataframe result changed the signed numeric kind")
        if str(materialized.schema["amount"]) != profile["dtype"]:
            raise TypeError("dataframe result changed the signed numeric storage type")
        facts["output_schema"] = {"amount": str(materialized.schema["amount"])}
        facts.update(output_columns={name: sorted(roots) for name, roots in result.sources.items()},
                     row_selection_sources=sorted(result.row_sources),
                     output={"column": "amount", "value": record_value(amount)})
        origins = []
        for _, binding, _ in metadata:
            for origin in binding["origins"]:
                if origin not in origins:
                    origins.append(origin)
        output = context._publish(result._tn.data.copy(), content=amount, declared=None,
            claims=list(dict.fromkeys(claim.id for claim in request.claims)), origins=origins,
            values=values, request=request, decision_id=decision, call_id=call_id,
            literals=[{"parameter_index": 0, "role": "dataframe-plan", "claims": []}])
        context.frame_runs.append({**deepcopy(facts), "decision_id": decision,
            "result_id": output.result_id, "output_reference": output.reference.to_dict()})
        return output
    except Exception as exc:
        context._append("Failure", request, call_id,
            {"decision_id": decision, "error_type": type(exc).__name__})
        raise
    finally:
        context._frame_pending.pop(call_id, None)
