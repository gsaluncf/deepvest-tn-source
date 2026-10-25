"""Polars and OPA orchestration around Rust-owned TN working objects.

Revision resolution and artifact verification belong to the dataset resolver.
This boundary never reconstructs governance or creates an accepted selection.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

import polars as pl

from .admission import SchemaPinViolation, _pin_dtypes, _pin_message
from .policy import CompiledPolicy
from .release import PolicyDenied


def _check_use(use):
    from tn.governed import UseContext
    if not isinstance(use, UseContext):
        raise TypeError("the TN boundary requires a native UseContext for use")


def root_name(source_id, column):
    return f"{source_id}::{column}"


def _refusal(message):
    return PolicyDenied(message + ". Data governance must review the edition's evaluator artifacts; "
                        "platform ops must correct routing or the requested operation.")


@dataclass(frozen=True)
class _PolicyEntry:
    policy: CompiledPolicy
    evaluator: object
    artifacts: object
    source_ids: frozenset
    dtypes: dict | None


class PolicySet:
    """All contributing evaluators, keyed by exact native policy revision."""

    def __init__(self, entries):
        self._entries = MappingProxyType(dict(entries))

    @classmethod
    def for_selection(cls, entries, selection, *, module_cache=None):
        from tn.governed import DatasetSelection, EvaluatorArtifactSet
        if not isinstance(selection, DatasetSelection):
            raise TypeError("TN frame loading requires an accepted native DatasetSelection")
        required = {binding.revision_id for binding in selection.record.contracts}
        expected = {item.policy_revision: item for item in selection.record.evaluator_artifacts}
        if set(expected) != required or set(entries) != required:
            raise _refusal("Every selected contract needs exactly one verified evaluator")
        accepted = {}
        for revision, entry in entries.items():
            artifacts = getattr(entry, "verified_artifacts", None)
            if (getattr(entry, "revision_id", None) != revision
                    or not isinstance(artifacts, EvaluatorArtifactSet)
                    or artifacts != expected[revision]):
                raise _refusal("Evaluator digests do not match the selected signed edition")
            # The resolver's factory rechecks parsed policy/profile consistency
            # and loads precisely the artifact bytes it has verified.
            compiled_bytes = entry.compiled_policy_bytes
            profile_bytes = entry.profile_bytes
            if (hashlib.sha256(compiled_bytes).hexdigest() != artifacts.compiled_policy_sha256
                    or hashlib.sha256(profile_bytes).hexdigest() != artifacts.profile_sha256):
                raise _refusal("Compiled policy or profile bytes differ from the approved edition")
            policy = CompiledPolicy.from_dict(json.loads(compiled_bytes))
            profile = json.loads(profile_bytes)
            evaluator = (entry.evaluator() if module_cache is None
                         else entry.evaluator(module_cache=module_cache))
            accepted[revision] = _PolicyEntry(policy, evaluator, artifacts,
                frozenset({selection.source_object_id}), deepcopy(profile.get("dtypes")))
        return cls(accepted)

    @property
    def revision_ids(self):
        return tuple(sorted(self._entries))

    def copy(self):
        return PolicySet(self._entries)

    def include(self, other):
        merged = dict(self._entries)
        for revision, entry in other._entries.items():
            if revision in merged:
                current = merged[revision]
                if (current.artifacts != entry.artifacts
                        or current.policy.to_dict() != entry.policy.to_dict()):
                    raise _refusal("Conflicting evaluator artifacts for the same signed revision")
                entry = _PolicyEntry(current.policy, current.evaluator, current.artifacts,
                    current.source_ids | entry.source_ids, current.dtypes)
            merged[revision] = entry
        return PolicySet(merged)

    def validate_contracts(self, context):
        actual = {policy.revision_id: policy for policy in context.policies}
        if set(actual) != set(self._entries):
            raise _refusal("The source's complete contract set differs from its evaluators")
        for revision, entry in self._entries.items():
            if actual[revision].policy_ref != entry.policy.policy_ref:
                raise _refusal("Compiled evaluator and native accepted policy reference differ")

    def pin(self, frame):
        schema = frame.collect_schema()
        arrived = set(schema.names())
        for entry in self._entries.values():
            declared = set(entry.policy.fields)
            if arrived != declared:
                raise SchemaPinViolation(_pin_message(sorted(arrived - declared), sorted(declared - arrived)))
            if entry.dtypes:
                _pin_dtypes(schema, entry.dtypes)

    @staticmethod
    def _local_roots(entry, roots):
        return {column for root in roots for source, separator, column in [root.partition("::")]
                if separator and source in entry.source_ids}

    @staticmethod
    def _level(entry, roots):
        levels = {entry.policy.level(name) for name in roots}
        if levels - {"public", "internal"}:
            return "confidential"
        return "internal" if "internal" in levels else "public"

    def classification(self, roots):
        levels = [self._level(entry, self._local_roots(entry, roots))
                  for entry in self._entries.values()]
        return "confidential" if "confidential" in levels else "internal" if "internal" in levels else "public"

    def decide(self, operation, audience, sources, row_sources, use, *, support=None, materialized=None):
        decisions = []
        for revision, entry in self._entries.items():
            columns = []
            for name, roots in sources.items():
                local = self._local_roots(entry, roots)
                columns.append({"name": name, "sources": sorted(local),
                                "classification": self._level(entry, local)})
            request = {
                "policy_ref": entry.policy.policy_ref, "policy_revision": revision,
                "operation": operation, "audience": audience, "columns": columns,
                "row_selection_sources": sorted(self._local_roots(entry, row_sources)),
                "aggregate_support": support.facts(materialized, {revision}) if support is not None else {"known": False},
                "use": {"application": use.application, "purpose": use.purpose,
                        "operation": use.operation},
            }
            decision = entry.evaluator.decide(request)
            if type(decision.allow) is not bool or not decision.allow:
                raise _refusal(f"Contract {revision} refused {operation}: {decision.reason}")
            decisions.append(decision)
        return decisions


@dataclass(frozen=True)
class TnBoundary:
    data: object
    policies: PolicySet
    use: object
    group: str
    rows_field: str

    def copy(self):
        return TnBoundary(self.data.copy(), self.policies.copy(), self.use, self.group, self.rows_field)

    def include(self, other):
        if self.use != other.use:
            raise _refusal("Joined TN inputs require the same admitted application use")
        data = self.data.copy()
        data.include(other.data)
        return TnBoundary(data, self.policies.include(other.policies), self.use, self.group, self.rows_field)


def load_from_tn(wire, *, session, use, selection, policies, decide, group="finance",
                 rows_field="rows"):
    from .frame import GovernedLazyFrame
    _check_use(use)
    policy_set = PolicySet.for_selection(policies, selection)
    # Schema is known from verified evaluator profiles before any business open.
    declared = set().union(*(set(entry.policy.fields) for entry in policy_set._entries.values()))
    sources = {name: {root_name(selection.source_object_id, name)} for name in declared}

    def accept(context):
        policy_set.validate_contracts(context)
        policy_set.decide("load", use.application, sources, set(), use)
        return decide(context)

    data = session.receive(wire, use=use, groups=[group], selection=selection, decide=accept)
    # Native mutable containers are projected only after receive succeeded.
    rows = deepcopy(data.get(rows_field, group=group))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise _refusal(f"The admitted {group}.{rows_field} field must contain row objects")
    frame = pl.DataFrame(rows).lazy()
    policy_set.pin(frame)
    first = next(iter(policy_set._entries.values()))
    return GovernedLazyFrame(frame, first.policy, {}, first.evaluator, sources=sources,
        tn_boundary=TnBoundary(data, policy_set, use, group, rows_field))


def release_tn(frame, *, use, to, decide, object_type=None, group=None, rows_field=None):
    _check_use(use)
    boundary = frame._tn
    if boundary is None:
        raise _refusal("A TN release requires a frame loaded through native TN admission")
    group = group or boundary.group
    rows_field = rows_field or boundary.rows_field
    if rows_field == "tn_polars_lineage":
        raise _refusal("The output rows field is reserved for signed Polars lineage")
    materialized = frame._support.frame.collect()
    decisions = boundary.policies.decide("release", to, frame.sources, frame.row_sources, use,
                                        support=frame._support, materialized=materialized)
    # Work on an independent native object so a refused release cannot change
    # this frame, its siblings, or their retained source publications.
    data = boundary.data.copy()
    rows = materialized.select(list(frame.sources)).to_dicts()
    lineage = {
        "schema": "tn.polars-lineage.v1",
        "columns": {name: sorted(roots) for name, roots in frame.sources.items()},
        "row_selection_sources": sorted(frame.row_sources),
        "policy_revisions": list(boundary.policies.revision_ids),
        "decisions": [item.decision_id for item in decisions],
    }
    data.set(None, {rows_field: rows, "tn_polars_lineage": lineage}, group=group)
    data.select([group])
    return data.release(use=use, to=to, decide=decide,
                        object_type=object_type or data.object_type)
