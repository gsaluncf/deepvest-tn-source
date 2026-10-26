"""Transport a governed primitive to the native model service.

No payload reveal, prompt-side governance loop, or client-side report gate lives
here. The native service owns model access, compliance, and visible attribution.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

import tn
import yaml
from tn_llama_client.clients import NativeOutputReader

from deepvest_tn.llama import submit_governed_report
from deepvest_tn.publications import write_once

from .string import OperationDenied, _active


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def use_dict(use):
    return {name: getattr(use, name) for name in ("application", "purpose", "operation")}


def _require_standalone_value(source, reference):
    # The native endpoint selects whole sections. Sending a shared values
    # section would include siblings absent from this primitive's handoff.
    fields = source.envelope.get("values", {}).get("field_hashes", {})
    if (reference.object_id != source.id or reference.group != "values"
            or set(fields) != {reference.field}):
        raise OperationDenied(
            "The selected primitive must be materialized as a standalone governed value before model handoff.")


def model_requirements(context, policies):
    """Resolve complete model conditions from the already verified assignments."""
    declarations = []
    for policy in policies:
        rule = context.resolver.rules.get(policy.revision_id, {})
        model = rule.get("model")
        if not isinstance(model, dict) or not isinstance(model.get("required_disclosure"), str):
            raise OperationDenied("a contributing contract has no disclosure-conditioned model grant")
        if not model["required_disclosure"].strip():
            raise OperationDenied("model grant requires a nonempty disclosure")
        declarations.append(model)
    if not declarations:
        raise OperationDenied("model handoff has no contributing contracts")
    first = declarations[0]
    keys = ("use", "recipient_use", "destinations", "report_destination", "required_tool")
    if any(any(m[key] != first[key] for key in keys) for m in declarations):
        raise OperationDenied("contributing model-use conditions need a compatible endpoint")
    reviews = [m["review"] for m in declarations if m.get("review") is not None]
    if any(not isinstance(r, dict) or set(r) != {"action", "assignee", "reason"}
           or r["action"] != "human_review" or not all(isinstance(v, str) and v.strip() for v in r.values())
           for r in reviews):
        raise ValueError("review action requires an assignee and reason")
    if reviews and any(r != reviews[0] for r in reviews):
        raise ValueError("this endpoint needs a common review route for its contributing contracts")
    return {**first, "required_disclosure": " ".join(dict.fromkeys(m["required_disclosure"] for m in declarations)),
            "review": reviews[0] if reviews else None}


@dataclass(frozen=True)
class ReviewRequest:
    record_id: str
    report_object_id: str
    status: str
    assignee: str
    reason: str
    queue_entry: str

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class NativeReport:
    object: object
    report_html: str
    release: object
    decision_id: str
    handoff_record_id: str
    result_record_id: str
    review: ReviewRequest | None = None

    @property
    def needs_review(self):
        return self.review is not None and self.review.status == "pending"


class NativeModelEndpoint:
    def __init__(self, *, context, value, base_url, api_key, output_reader, directory,
                 requirements, runtime_config):
        self.context, self.reference = context, value.reference
        self.base_url, self.api_key, self.output_reader = base_url, api_key, output_reader
        self.directory, self.requirements = Path(directory), requirements
        self.runtime_config = Path(runtime_config)
        self.runtime_digest = digest(runtime_config)
        self.use = tn.UseContext(**requirements["use"])

    def generate(self, value, *, prompt, destination="internal-model"):
        context = self.context
        if _active.get() is not context:
            raise OperationDenied("model handoff requires the active governed context")
        if value.reference != self.reference:
            raise OperationDenied("this native endpoint was prepared for another exact input")
        if type(prompt) is not str or not prompt.strip():
            raise ValueError("prompt must be an ordinary nonempty instruction")
        call_id = uuid4().hex
        request, metadata, decision = context._prepare(
            "model_handoff", (value,), destination, (prompt,), call_id)
        try:
            requirements = model_requirements(context, request.policies)
            if requirements != self.requirements or digest(self.runtime_config) != self.runtime_digest:
                raise OperationDenied("model endpoint configuration changed after preparation")
            source = context.store_object(value.reference.object_id)
            _require_standalone_value(source, value.reference)
            handoff = context._append("ModelHandoff", request, call_id, {
                "decision_id": decision, "subject": value.reference.to_dict(),
                "delivery": "prepared-for-submit", "native_use": use_dict(self.use),
                "runtime_config_sha256": self.runtime_digest,
                "required_disclosure": requirements["required_disclosure"],
                "review_action": requirements["review"],
                "input_wire_sha256": hashlib.sha256(source.forward()).hexdigest(),
            })
            release, opened = submit_governed_report(source, prompt=prompt, use=self.use,
                destination=requirements["report_destination"], base_url=self.base_url, api_key=self.api_key,
                transcript_path=self.directory / "model-http.ndjson", output_reader=self.output_reader,
                publication_store=context.recorder.store)
            # NativeOutputReader has already accepted the report's intended internal use.
            html = opened.get("report_html", group="default")
            review = None
            if requirements["review"] is not None:
                action = requirements["review"]
                review_id = context._append("ReviewRequested", request, call_id, {
                    "decision_id": decision, "handoff_record_id": handoff,
                    "input_object_id": source.id, "report_object_id": release.output_object.id,
                    "status": "pending", "review_action": action,
                    "required_disclosure": requirements["required_disclosure"],
                    "policy_revisions": sorted(p.revision_id for p in request.policies),
                    "report_html_sha256": hashlib.sha256(html.encode()).hexdigest(),
                })
                # Materialize the follow-up action as exact signed TN wire in a
                # local queue. This requests review; it does not claim review occurred.
                entry = "review-queue/" + review_id[7:] + ".tn"
                write_once(self.directory.parent / entry, context.recorder.store.get(review_id))
                review = ReviewRequest(review_id, release.output_object.id, "pending",
                                       action["assignee"], action["reason"], entry)
            result = context._append("ModelResult", request, call_id, {
                "decision_id": decision, "handoff_record_id": handoff,
                "input_object_id": source.id, "output_object_id": release.output_object.id,
                "tool_object_ids": [obj.id for obj in release.tool_objects],
                "required_disclosure": requirements["required_disclosure"],
                "claims": [claim.id for claim in request.claims],
                "claim_resolution": "original subjects and statements remain in the exact retained input ancestry",
                "report_html_sha256": hashlib.sha256(html.encode()).hexdigest(),
                "review_request_id": review.record_id if review else None,
            })
            return NativeReport(release.output_object, html, release, decision, handoff, result, review)
        except Exception as exc:
            context._append("Failure", request, call_id, {"decision_id": decision,
                "error_type": type(exc).__name__, "native_code": getattr(exc, "code", None)})
            raise


def prepare_native_model(value, *, context, input_session, output_session, recipient_session,
                         input_config, output_config, directory, model, binary, translate=str):
    """Compile carried requirements into the native endpoint's configuration."""
    source = context.store_object(value.reference.object_id)
    _require_standalone_value(source, value.reference)
    metadata, _, _ = context._metadata(value)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    input_config, output_config = Path(input_config), Path(output_config)
    requirements = model_requirements(context, metadata.policies)
    model_use = tn.UseContext(**requirements["use"])
    recipient_use = tn.UseContext(**requirements["recipient_use"])
    def resolve(identity):
        obj = context.store_object(identity)
        if obj.writer not in context.evaluator.writers:
            raise OperationDenied("unrecognized primitive ancestor writer")
        return input_session.governance(obj)
    lineage = tn.LineageVerifier(max_objects=65536, max_depth=256).verify(input_session.governance(source), context.catalog, context.policy_dag, resolve)
    revision_ids = list(dict.fromkeys(identity for a in context.resolver.assignments.values() for identity in a["revisions"]))
    edition_ids = [r.id for r in context.editions.values()]
    metadata_path = directory / "native-metadata.json"
    wire = lambda identity: context.recorder.store.get(identity).decode("utf-8")
    write_once(metadata_path, json.dumps({
        "policy_revisions": [wire(i) for i in revision_ids],
        "dataset_editions": [wire(i) for i in edition_ids],
        "publications": [wire(i) for i in lineage.object_ids],
    }).encode())
    for config in (input_config, output_config):
        readers = config.parent / "readers"
        readers.mkdir(exist_ok=True)
        for kit in (config.parent / "keys").glob("*.btn.mykit"):
            shutil.copyfile(kit, readers / kit.name)
    source_cfg, output_cfg = yaml.safe_load(input_config.read_text()), yaml.safe_load(output_config.read_text())
    if source_cfg["ceremony"]["id"] == output_cfg["ceremony"]["id"]:
        raise ValueError("native model input and output require independent ceremonies")
    (directory / "model-traces").mkdir(exist_ok=True)
    config = {
        "version": "tn.llama.runtime.v2", "writer_authority": "group_capability",
        # Evidence travels encrypted with the exact primitive publication. Only
        # values is selectable as model content under the report contract below.
        "expected_event_type": source.object_type, "allowed_input_groups": ["values", "evidence"],
        "governed_objects": {
            "native_reader_config_path": translate(input_config), "metadata_store_path": translate(metadata_path),
            "use_context": use_dict(model_use), "destinations": [requirements["report_destination"]],
            "trusted_writers": sorted(context.evaluator.writers), "authority_writers": [context.resolver.authority],
            "contracts": [{"revision_id": p.revision_id, "scope": context.policy_dag.get(p.revision_id).scope,
                           "allowed_uses": [use_dict(model_use)]} for p in metadata.policies],
        },
        "report_contract": {
            "allowed_input_sections": ["values"], "output_event_type": "report.generated",
            "output_public_section": "default", "operation": model_use.operation,
            "section_rules": {"values": {"propagate_source_governance": True,
                "required_attribution": requirements["required_disclosure"],
                "required_tool": requirements["required_tool"]}},
            "output_policy": {name: metadata.policies[0].fields[name] for name in
                ("instruction", "use_for", "do_not_use_for", "consequences", "on_violation_or_error")},
        },
        "reader_key_dir": translate(input_config.parent / "readers"),
        "input_index_key_path": translate(input_config.parent / "keys/index_master.key"),
        "ceremony_id": source_cfg["ceremony"]["id"], "governance_group": "tn.agents", "input_index_epoch": 0,
        "output_tn_config_path": translate(output_config),
        "output_reader_key_dir": translate(output_config.parent / "readers"),
        "output_index_key_path": translate(output_config.parent / "keys/index_master.key"),
        "output_ceremony_id": output_cfg["ceremony"]["id"], "output_governance_group": "tn.agents", "output_index_epoch": 0,
        "audit_path": translate(directory / "model-audit.ndjson"), "raw_trace_dir": translate(directory / "model-traces"),
        "build_id": "primitive-model-disclosure", "llama_source_revision": "3f545beccee69d9975f466ec7e45fd9aacd8ba90",
        "model_sha256": digest(model), "binary_sha256": digest(binary),
        "max_request_bytes": 1024 * 1024, "max_message_bytes": 16 * 1024,
        "max_active_sessions": 1, "max_chain_namespaces": 8, "echo_max_tokens": 8192, "final_max_tokens": 2048,
    }
    path = directory / "runtime.json"
    write_once(path, json.dumps(config, indent=2).encode())
    reader = NativeOutputReader(recipient_session, expected_source=source, expected_writer=output_session.did,
        use=recipient_use, destination=requirements["report_destination"], release_use=model_use,
        catalog=context.catalog, policy_dag=context.policy_dag, resolve_ancestor=resolve,
        decide=lambda ctx: ctx.use_context == recipient_use,
        expected_tool_name=requirements["required_tool"], expected_tool_sections=("values",))
    return path, reader, requirements
