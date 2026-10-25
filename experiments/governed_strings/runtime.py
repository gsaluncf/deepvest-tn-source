"""String operations over native TN publications.

The application process is trusted to enforce authorization. Callers with control
over the process or independent reader keys can access plaintext outside this API.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from uuid import uuid4

import tn

from .evidence import EvidenceError, EvidenceRecorder
from .string import (
    OperationDenied, StringMetadataProvider, TNString, ValueReference, _active, _exposure,
)

STRING_TYPE = "experiment.string"
STRING_SCHEMA = "tn.string-bindings.v1"


@dataclass(frozen=True)
class Claim:
    id: str
    issuer: str
    subjects: tuple[ValueReference, ...]
    statement: str


@dataclass(frozen=True)
class Request:
    application: str
    purpose: str
    operation: str
    inputs: tuple[ValueReference, ...]
    writers: tuple[str, ...]
    policies: tuple
    claims: tuple[Claim, ...]
    destination: str | None = None
    parameters: tuple = ()


@dataclass(frozen=True)
class Assessment:
    permit: bool
    reason: str
    obligations: tuple[str, ...] = ()
    claim_judgments: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ObligationResult:
    satisfied: bool
    evidence: str


@dataclass(frozen=True)
class Rules:
    """Explicit local demonstration evaluator, not an interpreter of arbitrary prose.

    It accepts configured native contracts. Extra claims are considered informative;
    another receiver may supply a different evaluator and reach a different decision.
    """

    writers: frozenset[str]
    policies: tuple
    operations: frozenset[str]
    destinations: frozenset[str]

    def accepts_metadata(self, ctx):
        return (ctx.writer in self.writers and ctx.object_type == STRING_TYPE
                and bool(ctx.policies)
                and all(any(p.matches_contract(expected) for expected in self.policies)
                        for p in ctx.policies))

    def __call__(self, request):
        judgments = tuple((claim.id, "informative") for claim in request.claims)
        trusted = all(writer in self.writers for writer in request.writers)
        contracts = bool(request.policies) and all(
            any(policy.matches_contract(expected) for expected in self.policies)
            for policy in request.policies
        )
        allowed = trusted and contracts and request.operation in self.operations
        if request.operation == "reveal":
            allowed = allowed and request.destination in self.destinations
        return Assessment(bool(allowed), "configured receiver rules", claim_judgments=judgments)


def publish_source(session, store, fields, *, policy=None, string_metadata=None):
    """Local origination helper. A real driver must authorize retrieval before this step."""
    if not fields or any(not isinstance(k, str) or not k or type(v) is not str
                         for k, v in fields.items()):
        raise TypeError("source requires nonempty string fields")
    policy = policy or session.policy(STRING_TYPE)
    data = session.create(fields, policy, object_type=STRING_TYPE, group="strings")
    store.put(data.snapshot)
    data.set(None, {"manifest": {
        "schema": STRING_SCHEMA,
        "bindings": {name: {"claims": [], "parents": [], "decision_id": None}
                     for name in fields},
    }}, group="evidence")
    publication = session.release(
        data, use=tn.UseContext("local-source", "string-experiment", "originate"),
        to="governed-runtime", object_type=STRING_TYPE,
        decide=lambda ctx: len(ctx.policies) == 1 and ctx.policies[0].matches_contract(policy),
    )
    store.put(publication)
    return {name: TNString(ValueReference(publication.id, name), metadata_provider=string_metadata)
            for name in fields}


class Context:
    """Trusted string application around an explicitly supplied native TN Session."""

    def __init__(self, session, recorder: EvidenceRecorder, evaluator: Callable, *,
                 application, purpose, metadata_accept=None, obligations=None,
                 string_metadata: StringMetadataProvider | None = None):
        if recorder.session is not session:
            raise ValueError("recorder must belong to this endpoint session")
        tn.UseContext(application, purpose, "inspect")  # Native nonempty validation.
        self.session = session
        self.recorder = recorder
        self.evaluator = evaluator
        self.application = application
        self.purpose = purpose
        self.string_metadata = string_metadata
        self.metadata_accept = metadata_accept or getattr(evaluator, "accepts_metadata", None)
        if self.metadata_accept is None:
            raise TypeError("explicit metadata admission callback required")
        self.obligations = dict(obligations or {})
        self._tokens = []

    def __enter__(self):
        self._tokens.append(_active.set(self))
        return self

    def __exit__(self, *_):
        _active.reset(self._tokens.pop())

    @contextmanager
    def exposure(self, destination):
        if not isinstance(destination, str) or not destination:
            raise ValueError("exposure requires a named destination")
        token = _exposure.set((self.recorder.context_id, destination))
        try:
            yield self
        finally:
            _exposure.reset(token)

    def store_object(self, identity):
        return tn.GovernedObject.read(io.BytesIO(self.recorder.store.get(identity)))

    def _metadata(self, value):
        obj = self.store_object(value.reference.object_id)
        data = self.session.receive(
            obj, use=tn.UseContext(self.application, self.purpose, "inspect-claims"),
            groups=["evidence"], decide=self.metadata_accept,
        )
        manifest = data.get("manifest", group="evidence")
        if not isinstance(manifest, dict) or manifest.get("schema") != STRING_SCHEMA:
            raise EvidenceError("missing signed string manifest")
        binding = manifest.get("bindings", {}).get(value.reference.field)
        if not isinstance(binding, dict) or set(binding) != {"claims", "parents", "decision_id"}:
            raise EvidenceError("missing signed field binding")
        if not isinstance(binding["claims"], list) or not isinstance(binding["parents"], list):
            raise EvidenceError("invalid signed string lineage")
        decision_id = binding["decision_id"]
        if decision_id is not None:
            if decision_id != value.decision_id or value.result_id is None:
                raise EvidenceError("derived string requires its signed decision/result citations")
            result = self.recorder.read(value.result_id)
            decision = self.recorder.read(decision_id)
            rb, db = result.body, decision.body
            if (rb["kind"] != "Result" or db["kind"] != "Decision"
                    or result.writer != obj.writer or decision.writer != obj.writer
                    or rb["details"].get("output") != value.reference.to_dict()
                    or rb["details"].get("decision_id") != decision_id
                    or rb["inputs"] != binding["parents"]
                    or rb["details"].get("claims") != binding["claims"]
                    or db["details"].get("permitted") is not True
                    or any(rb[key] != db[key] for key in (
                        "context_id", "call_id", "operation", "inputs", "application", "purpose",
                    ))):
                raise EvidenceError("result citation does not bind this string to its decision")
        elif value.decision_id is not None or value.result_id is not None:
            raise EvidenceError("origin has unexpected derived citations")
        claims = []
        for identity in binding["claims"]:
            record = self.recorder.read(identity)
            if record.body["kind"] != "Claim":
                raise EvidenceError("claim reference does not identify a claim")
            statement = record.body["details"].get("statement")
            if not isinstance(statement, str) or not statement.strip():
                raise EvidenceError("invalid signed claim statement")
            claims.append(Claim(record.id, record.writer,
                                tuple(ValueReference(**r) for r in record.body["inputs"]), statement))
        return data, binding, tuple(claims)

    def inspect(self, value):
        data, _, claims = self._metadata(value)
        return Request(self.application, self.purpose, "inspect", (value.reference,),
                       (data.snapshot.writer,), tuple(data.policies), claims)

    def _append(self, kind, request, call_id, details):
        return self.recorder.append(
            kind, call_id=call_id, application=self.application, purpose=self.purpose,
            operation=request.operation, inputs=request.inputs,
            destination=request.destination, details=details,
        )

    def _prepare(self, operation, values, destination, parameters, call_id):
        base = Request(self.application, self.purpose, operation,
                       tuple(v.reference for v in values), (), (), (), destination, parameters)
        try:
            metadata = [self._metadata(value) for value in values]
            claims = {claim.id: claim for _, _, cs in metadata for claim in cs}
            request = replace(base,
                writers=tuple(data.snapshot.writer for data, _, _ in metadata),
                policies=tuple(policy for data, _, _ in metadata for policy in data.policies),
                claims=tuple(claims.values()))
            assessment = self.evaluator(request)
            if (not isinstance(assessment, Assessment) or type(assessment.permit) is not bool
                    or not isinstance(assessment.reason, str)
                    or not isinstance(assessment.obligations, tuple)
                    or any(not isinstance(name, str) or not name for name in assessment.obligations)):
                raise ValueError("evaluator did not return a valid explicit assessment")
            if operation == "reveal" and (not isinstance(destination, str) or not destination):
                assessment = Assessment(False, "exposure requires a named destination")
            discharged = []
            permitted = assessment.permit
            if permitted:
                for name in assessment.obligations:
                    handler = self.obligations.get(name)
                    result = handler(request) if handler else None
                    ok = (isinstance(result, ObligationResult) and result.satisfied is True
                          and isinstance(result.evidence, str) and bool(result.evidence))
                    discharged.append({"obligation": name, "satisfied": ok,
                                       "evidence": result.evidence if ok else None})
                    if not ok:
                        permitted = False
                        break
            details = {
                "permitted": permitted, "requested_permit": assessment.permit,
                "reason": assessment.reason, "obligations": list(assessment.obligations),
                "discharged": discharged, "parameters": list(parameters),
                "policies": [{"authority": p.governed_by, "policy": p.policy_ref,
                              "revision": p.revision_id} for p in request.policies],
                "considered_claims": [claim.id for claim in request.claims],
                "claim_judgments": [list(j) for j in assessment.claim_judgments],
            }
        except Exception as exc:
            # A failed decision is not an authorization; avoid logging arbitrary exception text.
            request, metadata, permitted = base, [], False
            details = {"permitted": False, "reason": "decision could not be completed",
                       "error_type": type(exc).__name__}
        decision_id = self._append("Decision", request, call_id, details)
        if not permitted:
            self._append("Refusal", request, call_id, {"decision_id": decision_id,
                                                       "reason": details["reason"]})
            raise OperationDenied(f"{operation} was not permitted; decision {decision_id}")
        return request, metadata, decision_id

    def _open_values(self, request, values):
        """Native admission and business decrypt; called only after a recorded permit."""
        working = []
        for value in values:
            obj = self.store_object(value.reference.object_id)
            use = tn.UseContext(self.application, self.purpose, request.operation)
            data = self.session.receive(
                obj, use=use, groups=["strings", "evidence"],
                decide=lambda ctx, obj=obj, use=use: (
                    ctx.object.id == obj.id and ctx.writer == obj.writer
                    and ctx.use_context == use and set(ctx.groups) == {"strings", "evidence"}
                    and self.metadata_accept(ctx)
                ),
            )
            content = data.get(value.reference.field, group="strings")
            if type(content) is not str:
                raise TypeError("selected value is not a string")
            working.append((data, content))
        return working

    def compare(self, left: TNString, right: str, *, operation: str) -> bool:
        """Reject comparisons that require opening protected string content.

        Metadata comparisons are handled separately. This method requires an
        active context and raises NotImplementedError before opening content.
        """
        if _active.get() is not self:
            raise OperationDenied("protected comparison requires this active context")
        raise NotImplementedError("protected comparison execution is not implemented yet")

    def perform(self, operation, values, *, destination=None, parameters=()):
        if operation not in {
            "copy", "upper", "lower", "slice", "index", "repeat", "concat", "attest", "reveal",
        }:
            raise ValueError("unsupported string operation")
        literal_concat = operation == "concat" and bool(parameters)
        if literal_concat and (
            len(parameters) != 2 or type(parameters[0]) is not str
            or parameters[1] not in {"left", "right"}
        ):
            raise TypeError("literal concat requires text and its operand position")
        if len(values) != (2 if operation == "concat" and not literal_concat else 1):
            raise ValueError("wrong number of governed inputs")
        call_id = uuid4().hex
        request, metadata, decision_id = self._prepare(
            operation, values, destination, parameters, call_id,
        )
        try:
            claims = list(dict.fromkeys(claim.id for claim in request.claims))
            if operation in {"copy", "attest"}:
                if operation == "attest":
                    claim_id = self._append("Claim", request, call_id, {"statement": parameters[0]})
                    claims.append(claim_id)
                # Both operations publish a new incarnation with the same text.
                # Retaining ciphertext does not require opening the business group.
                data = metadata[0][0].copy()  # Keep unopened strings as native retained ciphertext.
                field = values[0].reference.field
            else:
                opened = self._open_values(request, values)
                data, content = opened[0]
                if operation == "reveal":
                    # Persist before returning control/plaintext to the caller.
                    self._append("Disclosure", request, call_id, {
                        "decision_id": decision_id, "subject": values[0].reference.to_dict(),
                        "delivery": "prepared-for-return",  # Not proof a downstream sink received it.
                    })
                    return content
                if operation == "upper":
                    content = content.upper()
                elif operation == "lower":
                    content = content.lower()
                elif operation == "slice":
                    content = content[slice(*parameters)]
                elif operation == "index":
                    content = content[parameters[0]]
                elif operation == "repeat":
                    content = content * parameters[0]
                elif operation == "concat":
                    if literal_concat:
                        # Plain operands are caller-supplied parameters in the signed
                        # decision; they are not attributed to an earlier endpoint.
                        literal, side = parameters
                        content = literal + content if side == "left" else content + literal
                    else:
                        data.include(opened[1][0])
                        content += opened[1][1]
                field = "value"
                data.set(None, {field: content}, group="strings")
            data.set(None, {"manifest": {
                "schema": STRING_SCHEMA,
                "bindings": {field: {"claims": claims,
                                     "parents": [v.reference.to_dict() for v in values],
                                     "decision_id": decision_id}},
            }}, group="evidence")
            use = tn.UseContext(self.application, self.purpose, operation)
            publication = self.session.release(
                data, use=use, to="governed-runtime", object_type=STRING_TYPE,
                decide=lambda ctx: ctx.writer == self.session.did and ctx.use_context == use
                and all(any(p.matches_contract(parent) for parent in request.policies)
                        for p in ctx.policies),
            )
            self.recorder.store.put(publication)
            reference = ValueReference(publication.id, field)
            result_id = self._append("Result", request, call_id, {
                "decision_id": decision_id, "output": reference.to_dict(), "claims": claims,
                "field_commitment": publication.envelope["strings"]["field_hashes"][field],
                "transition": {
                    "governed_inputs": [v.reference.to_dict() for v in values],
                    "governance": "retain-all-contributing-governance",
                    "literal_inputs": ([{
                        "parameter_index": 0, "position": parameters[1], "claims": [],
                    }] if literal_concat else []),
                },
            })
            return TNString(reference, decision_id, result_id,
                            metadata_provider=self.string_metadata)
        except Exception as exc:
            # Execution/key/storage failures are not represented as successful results.
            self._append("Failure", request, call_id,
                         {"decision_id": decision_id, "error_type": type(exc).__name__})
            raise
