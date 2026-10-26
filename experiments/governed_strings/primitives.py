"""Application execution for governed Python primitives over the pinned native TN SDK."""

from dataclasses import replace
import operator
import math
from decimal import getcontext
from uuid import uuid4

import tn

from .evidence import EvidenceError
from .numeric import TNNumeric
from .numeric_codec import KINDS, NUMERIC_TYPES, encode, decode, record_value
from .metadata import DeclaredMetadata, describe
from .runtime import Assessment, Context as StringContext, ObligationResult, Request
from .string import OperationDenied, TNString, ValueReference, _active

VALUE_TYPE = "experiment.primitive"
SCHEMA = "tn.primitive-bindings.v1"
BINARY = {
    "add": operator.add, "subtract": operator.sub, "multiply": operator.mul,
    "true_divide": operator.truediv, "divmod": divmod,
    "floor_divide": operator.floordiv, "modulo": operator.mod, "power": operator.pow,
    "bit_and": operator.and_, "bit_or": operator.or_, "bit_xor": operator.xor,
    "left_shift": operator.lshift, "right_shift": operator.rshift,
}
OPERATIONS = frozenset({
    "select", "model_handoff", "copy", "attest", "upper", "lower", "strip", "replace", "slice", "index",
    "repeat", "concat", "reveal", "compare", "to_int", "to_string", "negate",
    "positive", "absolute", "invert", "round", "trunc", "floor", "ceil",
    "conjugate", "real", "imag", "numerator", "denominator", "as_integer_ratio",
    "is_integer", "truth", "power_mod", "quantize", "sqrt", "normalize", "exp", "ln", "log10", *BINARY,
})


class Context(StringContext):
    def __init__(self, session, recorder, evaluator, *, application, purpose, actor,
                 metadata_accept=None, resolver=None):
        super().__init__(session, recorder, evaluator, application=application, purpose=purpose,
                         metadata_accept=metadata_accept)
        self.actor = actor
        self.resolver = resolver
        self.string_metadata = DeclaredMetadata(self)
        self.obligations["monitor"] = self._monitor
        self.catalog = tn.DatasetCatalog()
        self.policy_dag = resolver.policy_dag if resolver is not None else tn.PolicyDag()
        self.editions = {}
        # Bind the stable metadata/attestation handoff to this enrolled endpoint.
        # Content operations retain explicit per-call use and per-input selection.
        session.configure_receive(use=tn.UseContext(application, purpose, "inspect-claims"),
                                  groups=["evidence"], object_type=VALUE_TYPE, decide=self._admit)
        session.configure_release(use=tn.UseContext(application, purpose, "attest"),
                                  to="governed-runtime", object_type=VALUE_TYPE,
                                  decide=lambda ctx: ctx.writer == session.did)
        self.workflow = session.workflow(receive=purpose, release=purpose)

    def _monitor(self, request):
        identity = self._append("Monitor", request, uuid4().hex, {
            "event": "protected-operation-observed", "actor": self.actor,
        })
        return ObligationResult(True, identity)

    def _append(self, kind, request, call_id, details):
        return super()._append(kind, request, call_id, record_value({**details, "actor": self.actor}))

    def _read(self, value, groups, operation):
        obj = self.store_object(value.reference.object_id)
        use = tn.UseContext(self.application, self.purpose, operation)
        record = self.editions.get(obj.id)
        selection = None if record is None else self.catalog.select(
            record.dataset, record.edition, record.id, use)
        if operation == "inspect-claims":
            return self.workflow.receive(obj, selection=selection)
        return self.session.receive(obj, use=use, groups=groups, selection=selection, decide=self._admit)

    def _admit(self, ctx):
        if (not self.metadata_accept(ctx) or ctx.use_context.application != self.application
                or ctx.use_context.purpose != self.purpose):
            return False
        if ctx.governance.dataset_bindings:
            if not self.catalog.accepts(ctx, self.policy_dag):
                return False
            def ancestor(identity):
                parent = self.store_object(identity)
                view = self.session.governance(parent)
                view.accept(use=ctx.use_context, groups=["evidence"], decide=self.metadata_accept)
                return view
            tn.LineageVerifier(max_objects=65536, max_depth=256).verify(self.session.governance(ctx.object), self.catalog, self.policy_dag, ancestor)
        return True

    def hydrate(self, reference, decision_id, result_id, *, kind):
        """Verify a received handle after its edition/authority has been resolved."""
        return self.hydrate_many([(reference, decision_id, result_id, kind)])[0]

    def hydrate_many(self, handles):
        handles = tuple(handles)
        values = []
        for reference, decision_id, result_id, kind in handles:
            if kind not in {"str", *KINDS}:
                raise TypeError("unsupported primitive kind")
            cls = TNString if kind == "str" else TNNumeric
            values.append(cls(reference, decision_id, result_id, metadata_provider=self.string_metadata))
        for (_, binding, _), handle in zip(self._metadata_many(values), handles, strict=True):
            if binding["declared"]["kind"] != handle[3]:
                raise EvidenceError("requested type differs from signed primitive kind")
        return tuple(values)

    def _metadata(self, value, *, _reads=None, _records=None):
        # These maps live only for this operation. Every later operation admits
        # the publication again under its current endpoint and requested use.
        reads = {} if _reads is None else _reads
        records = {} if _records is None else _records
        identity = value.reference.object_id
        if identity not in reads:
            data = self._read(value, ["evidence"], "inspect-claims")
            reads[identity] = data, data.get("manifest", group="evidence")
        data, manifest = reads[identity]
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise EvidenceError("missing signed primitive manifest")
        binding = manifest.get("bindings", {}).get(value.reference.field)
        if not isinstance(binding, dict):
            raise EvidenceError("missing signed primitive binding")
        expected_kinds = KINDS if isinstance(value, TNNumeric) else {"str"}
        if binding.get("declared", {}).get("kind") not in expected_kinds:
            raise EvidenceError("primitive class differs from signed logical kind")
        if binding.get("decision_id") != value.decision_id or value.result_id is None:
            raise EvidenceError("primitive requires its exact signed decision and result")
        def record(identity):
            if identity not in records:
                records[identity] = self.recorder.read(identity)
            return records[identity]
        result, decision = record(value.result_id), record(value.decision_id)
        rb, db = result.body, decision.body
        if (rb["kind"] != "Result" or db["kind"] != "Decision"
                or result.writer != data.snapshot.writer or decision.writer != result.writer
                or rb["details"].get("output") != value.reference.to_dict()
                or rb["details"].get("decision_id") != decision.id
                or rb["inputs"] != binding["parents"]
                or rb["details"].get("claims") != binding["claims"]
                or rb["details"].get("field_commitment") !=
                    data.snapshot.envelope[value.reference.group]["field_hashes"][value.reference.field]
                or db["details"].get("permitted") is not True
                or rb["sequence"] <= db["sequence"]
                or any(rb[key] != db[key] for key in (
                    "context_id", "call_id", "operation", "inputs", "application", "purpose"))):
            raise EvidenceError("primitive is not bound to its authorized result")
        from .runtime import Claim
        claims = []
        for identity in binding["claims"]:
            claim = record(identity)
            if claim.body["kind"] != "Claim":
                raise EvidenceError("claim citation names another record kind")
            claims.append(Claim(identity, claim.writer,
                                tuple(ValueReference(**r) for r in claim.body["inputs"]),
                                claim.body["details"]["statement"]))
        return data, binding, tuple(claims)

    def _metadata_many(self, values):
        reads, records = {}, {}
        return [self._metadata(value, _reads=reads, _records=records) for value in values]

    def _decision(self, request, call_id):
        discharged = []
        try:
            assessment = self.evaluator(request)
            if not isinstance(assessment, Assessment) or type(assessment.permit) is not bool:
                raise ValueError("an explicit assessment is required")
            permitted = assessment.permit
            if request.operation == "reveal" and not request.destination:
                assessment = Assessment(False, "exposure requires a named destination")
                permitted = False
            if permitted:
                for name in assessment.obligations:
                    handler = self.obligations.get(name)
                    item = handler(request) if handler else None
                    ok = isinstance(item, ObligationResult) and item.satisfied is True and bool(item.evidence)
                    discharged.append({"obligation": name, "satisfied": bool(ok),
                                       "evidence": item.evidence if ok else None})
                    if not ok:
                        permitted = False
                        break
            details = {"permitted": permitted, "requested_permit": assessment.permit,
                       "reason": assessment.reason, "obligations": list(assessment.obligations),
                       "discharged": discharged, "parameters": list(request.parameters),
                       "decimal_context": {"precision": getcontext().prec, "rounding": getcontext().rounding,
                           "Emin": getcontext().Emin, "Emax": getcontext().Emax,
                           "clamp": getcontext().clamp, "capitals": getcontext().capitals,
                           "traps": sorted(k.__name__ for k,v in getcontext().traps.items() if v)},
                       "policies": [{"authority": p.governed_by, "policy": p.policy_ref,
                                     "revision": p.revision_id} for p in request.policies],
                       "considered_claims": [c.id for c in request.claims],
                       "claim_judgments": [list(j) for j in assessment.claim_judgments]}
        except Exception as exc:
            permitted = False
            details = {"permitted": False, "reason": "decision could not be completed",
                       "error_type": type(exc).__name__, "discharged": discharged}
        identity = self._append("Decision", request, call_id, details)
        if not permitted:
            self._append("Refusal", request, call_id, {"decision_id": identity, "reason": details["reason"]})
            raise OperationDenied(f"{request.operation} refused; decision {identity}")
        return identity

    def _prepare(self, operation, values, destination, parameters, call_id):
        base = Request(self.application, self.purpose, operation,
                       tuple(v.reference for v in values), (), (), (), destination, parameters)
        try:
            items = self._metadata_many(values)
        except Exception as exc:
            decision = self._append("Decision", base, call_id, {
                "permitted": False, "reason": "input admission failed", "error_type": type(exc).__name__})
            self._append("Refusal", base, call_id, {"decision_id": decision, "reason": "input admission failed"})
            raise OperationDenied(f"input admission failed: {type(exc).__name__}") from exc
        claims = {c.id: c for _, _, cs in items for c in cs}
        policies = []
        for data, _, _ in items:
            for policy in data.policies:
                if not any(policy.matches_contract(existing) for existing in policies):
                    policies.append(policy)
        request = replace(base, writers=tuple(d.snapshot.writer for d, _, _ in items),
                          policies=tuple(policies),
                          claims=tuple(claims.values()))
        return request, items, self._decision(request, call_id)

    def _open_values(self, request, values):
        result, opened, copies = [], {}, {}
        for value in values:
            key = value.reference.object_id, value.reference.group
            if key not in opened:
                data = self._read(value, [value.reference.group, "evidence"], request.operation)
                opened[key] = data, data.get("manifest", group="evidence")
                copies[key] = data.copy()
            data, manifest = opened[key]
            payload = data.get(value.reference.field, group=value.reference.group)
            facts = manifest["bindings"][value.reference.field]["declared"]
            content = decode(payload, facts["kind"])
            if describe(content) != facts:
                raise EvidenceError("declared facts do not match the opened content")
            result.append((copies[key], content))
        return result

    def _merge_opened(self, values, opened):
        """Combine each contributing publication/group once, preserving field-level inputs."""
        if not values or len(values) != len(opened):
            raise ValueError("opened values must match nonempty governed inputs")
        data = opened[0][0]
        merged = {(values[0].reference.object_id, values[0].reference.group)}
        for value, (other_data, _) in zip(values[1:], opened[1:], strict=True):
            identity = value.reference.object_id, value.reference.group
            if identity not in merged:
                data.include(other_data)
                merged.add(identity)
        return data

    def _release_value(self, data, request):
        use = tn.UseContext(self.application, self.purpose, request.operation)
        def accept_release(ctx):
            return (ctx.writer == self.session.did and ctx.use_context == use
                    and all(any(p.matches_contract(q) for q in request.policies) for p in ctx.policies)
                    and all(any(p.matches_contract(q) for q in ctx.policies) for p in request.policies))
        publication = (self.workflow.release(data, decide=accept_release) if request.operation == "attest"
                       else self.session.release(data, use=use, to="governed-runtime",
                                                 object_type=VALUE_TYPE, decide=accept_release))
        self.recorder.store.put(publication)
        return publication

    def _publish(self, data, *, content, declared, claims, origins, values, request,
                 decision_id, call_id, literals=(), source=None):
        field = values[0].reference.field if content is None else "value"
        if content is not None:
            if type(content) not in (str, *NUMERIC_TYPES):
                raise NotImplementedError("unsupported governed result type")
            declared = describe(content)
            data.set(None, {"value": encode(content)}, group="values")
        parents = [v.reference.to_dict() for v in values]
        binding = {"claims": claims, "parents": parents, "decision_id": decision_id,
                   "declared": declared, "origins": origins, "source": source}
        data.set(None, {"manifest": {"schema": SCHEMA, "bindings": {field: binding}}}, group="evidence")
        publication = self._release_value(data, request)
        reference = ValueReference(publication.id, field, "values")
        result = self._append("Result", request, call_id, {
            "decision_id": decision_id, "output": reference.to_dict(), "claims": claims,
            "field_commitment": publication.envelope["values"]["field_hashes"][field],
            "transition": {"governed_inputs": parents, "retained_claims": claims,
                           "literal_inputs": list(literals)}, "source": source,
        })
        cls = TNString if declared["kind"] == "str" else TNNumeric
        return cls(reference, decision_id, result, metadata_provider=self.string_metadata)

    def originate_batch(self, cells, policy, *, request, decision_id, call_id):
        """Publish one column batch; retain a signed Result for each exact field.

        All cells belong to one query, assignment, source snapshot and column,
        and use this endpoint's same provisioned group readers and policy.
        """
        cells = tuple(cells)
        if not cells or len(cells) > 64:
            raise ValueError("source publication requires between 1 and 64 cells")
        if request.operation != "select" or not any(policy.matches_contract(p) for p in request.policies):
            raise ValueError("source policy must be covered by the query request")
        first_content, first_source = cells[0]
        scope = {key: value for key, value in first_source.items() if key != "row"}
        fields, bindings = {}, {}
        for content, source in cells:
            if type(content) not in (str, *NUMERIC_TYPES) or type(content) is not type(first_content):
                raise TypeError("a source batch requires one supported scalar type")
            row = source.get("row")
            if (type(row) is not int or row < 0
                    or {key: value for key, value in source.items() if key != "row"} != scope):
                raise ValueError("source cells must share a column, query, assignment and snapshot")
            field = f"value_{row}"
            if field in fields:
                raise ValueError("duplicate row in source publication")
            fields[field] = encode(content)
            bindings[field] = {"claims": [], "parents": [], "decision_id": decision_id,
                               "declared": describe(content), "origins": [source], "source": source}
        data = self.session.create(fields, policy, object_type=VALUE_TYPE, group="values")
        self.recorder.store.put(data.snapshot)
        data.set(None, {"manifest": {"schema": SCHEMA, "bindings": bindings}}, group="evidence")
        publication = self._release_value(data, replace(request, policies=(policy,)))
        commitments = publication.envelope["values"]["field_hashes"]
        values = []
        for field, binding in bindings.items():
            reference = ValueReference(publication.id, field, "values")
            result = self._append("Result", request, call_id, {
                "decision_id": decision_id, "output": reference.to_dict(), "claims": [],
                "field_commitment": commitments[field],
                "transition": {"governed_inputs": [], "retained_claims": [], "literal_inputs": []},
                "source": binding["source"],
            })
            cls = TNString if binding["declared"]["kind"] == "str" else TNNumeric
            values.append(cls(reference, decision_id, result, metadata_provider=self.string_metadata))
        return tuple(values)

    def originate(self, content, policy, *, source, request, decision_id, call_id):
        data = self.session.create({"value": encode(content)},
                                   policy, object_type=VALUE_TYPE, group="values")
        self.recorder.store.put(data.snapshot)
        # Each cell carries only its assigned policy. The query Decision can cover several columns.
        cell_request = replace(request, policies=(policy,))
        return self._publish(data, content=content, declared=None, claims=[], origins=[source],
                             values=(), request=cell_request, decision_id=decision_id,
                             call_id=call_id, source=source)

    def compare(self, left, right, *, operation):
        if _active.get() is not self:
            raise OperationDenied("protected comparison requires the active context")
        if isinstance(right, (TNNumeric, TNString)):
            return self.perform("compare", (left, right), parameters=(operation,))
        if type(right) not in (str, bool, *NUMERIC_TYPES):
            raise TypeError("unsupported comparison operand")
        return self.perform("compare", (left,), parameters=(operation, right))

    def perform(self, operation, values, *, destination=None, parameters=()):
        if operation not in OPERATIONS - {"select", "model_handoff"}:
            raise NotImplementedError(f"unsupported governed primitive operation: {operation}")
        if operation not in {"reveal", "attest"} and _active.get() is not self:
            raise OperationDenied("content operations require this active context")
        call_id = uuid4().hex
        request, metadata, decision = self._prepare(operation, values, destination, parameters, call_id)
        try:
            claims = list(dict.fromkeys(c.id for c in request.claims))
            origins = []
            for _, binding, _ in metadata:
                for origin in binding["origins"]:
                    if origin not in origins:
                        origins.append(origin)
            literals = []
            if operation in {"copy", "attest"}:
                data, binding, _ = metadata[0]
                data = data.copy()
                if operation == "attest":
                    claims.append(self._append("Claim", request, call_id, {"statement": parameters[0]}))
                content, declared = None, binding["declared"]
            else:
                opened = self._open_values(request, values)
                data = self._merge_opened(values, opened)
                content = opened[0][1]
                declared = None
                if operation == "reveal":
                    self._append("Disclosure", request, call_id, {
                        "decision_id": decision, "subject": values[0].reference.to_dict(),
                        "delivery": "prepared-for-return", "kind": type(content).__name__,
                    })
                    return content
                if operation == "compare":
                    other = opened[1][1] if len(opened) == 2 else parameters[1]
                    answer = {"eq": operator.eq, "lt": operator.lt, "le": operator.le,
                              "gt": operator.gt, "ge": operator.ge}[parameters[0]](content, other)
                    self._append("Observation", request, call_id, {
                        "decision_id": decision, "observation": parameters[0], "value": answer,
                    })
                    return answer
                if operation in {"concat", *BINARY}:
                    if len(opened) == 2:
                        left, right = content, opened[1][1]
                    else:
                        literal, side = parameters
                        left, right = (literal, content) if side == "left" else (content, literal)
                        literals = [{"parameter_index": 0, "position": side, "claims": []}]
                    content = (operator.add if operation == "concat" else BINARY[operation])(left, right)
                elif operation == "upper": content = content.upper()
                elif operation == "lower": content = content.lower()
                elif operation == "strip": content = content.strip(*parameters)
                elif operation == "replace": content = content.replace(*parameters)
                elif operation == "slice": content = content[slice(*parameters)]
                elif operation == "index": content = content[parameters[0]]
                elif operation == "repeat": content *= opened[1][1] if len(opened) == 2 else parameters[0]
                elif operation == "to_int": content = int(content, *parameters)
                elif operation == "to_string": content = str(content)
                elif operation == "negate": content = -content
                elif operation == "positive": content = +content
                elif operation == "absolute": content = abs(content)
                elif operation == "invert": content = ~content
                elif operation == "round": content = round(content, *parameters)
                elif operation in {"trunc", "floor", "ceil"}: content = getattr(math, operation)(content)
                elif operation in {"real", "imag", "numerator", "denominator"}: content = getattr(content, operation)
                elif operation in {"conjugate", "as_integer_ratio", "sqrt", "normalize", "exp", "ln", "log10"}:
                    content = getattr(content, operation)()
                elif operation == "power_mod":
                    args = [opened[arg["input"]][1] if "input" in arg else arg["literal"] for arg in parameters]
                    content = pow(content, *args)
                elif operation == "quantize":
                    exponent = opened[1][1] if len(opened) == 2 else parameters[1]
                    content = content.quantize(exponent, rounding=parameters[0])
                elif operation in {"is_integer", "truth"}:
                    answer = bool(content) if operation == "truth" else content.is_integer()
                    self._append("Observation", request, call_id, {
                        "decision_id": decision, "observation": operation, "value": answer})
                    return answer
                if parameters and not literals:
                    literals = [{"parameter_index": i, "role": "operation-parameter", "claims": []}
                                for i in range(len(parameters))]
            if type(content) is tuple:
                return tuple(self._publish(data.copy(), content=item, declared=None, claims=claims,
                    origins=origins, values=values, request=request, decision_id=decision,
                    call_id=call_id, literals=literals) for item in content)
            return self._publish(data, content=content, declared=declared, claims=claims,
                                 origins=origins, values=values, request=request, decision_id=decision,
                                 call_id=call_id, literals=literals)
        except Exception as exc:
            self._append("Failure", request, call_id, {"decision_id": decision,
                                                       "error_type": type(exc).__name__})
            raise
