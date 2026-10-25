"""Offline evidence inspection of a real native model handoff, never a model gate."""

import base64
import hashlib
from html.parser import HTMLParser
import json

import tn
from tn.canonical import _canonical_bytes
from tn_llama_client.clients import NativeOutputReader
from tn_llama_client.protocol import NativeModelRelease

from .evidence import EvidenceError


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class PlainReportText(HTMLParser):
    """Inspect the intentionally plain report artifact, without evaluating CSS.

    This verifier refuses styling or hidden/script content rather than claiming
    to implement the native runtime's general human-visible HTML gate.
    """
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        require(tag in {"html", "body", "p", "div", "span", "br", "h1", "h2", "h3", "h4",
                        "ul", "ol", "li", "strong", "b", "em", "i", "code", "pre",
                        "table", "thead", "tbody", "tr", "th", "td"} and not attrs,
                "report is outside the unstyled, visible-element inspection profile")
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "td"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "li", "h1", "h2", "h3", "td"}:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def verify_model(root, run, records, by_id, rules, nodes, bindings, store, catalog, dag, resolve):
    model = run["model"]
    handoffs = [r for r in records if r.body["kind"] == "ModelHandoff"]
    results = [r for r in records if r.body["kind"] == "ModelResult"]
    require(len(handoffs) == 1, "demonstration must retain one attempted model submission")
    handoff = handoffs[0]
    hb, hd = handoff.body, handoff.body["details"]
    decision = by_id[hd["decision_id"]]
    subject = hd["subject"]
    key = subject["object_id"], subject["group"], subject["field"]
    require(key in bindings and hb["inputs"] == [subject] and hd["delivery"] == "prepared-for-submit",
            "handoff does not identify its exact primitive")
    source = resolve(key[0]).object
    require(source.id == model["input_object_id"] and sha(source.forward()) == hd["input_wire_sha256"],
            "handoff input differs from the retained wire")
    revisions = nodes[key]["policy_revisions"]
    dd = decision.body["details"]
    require(set(revisions) == {p["revision"] for p in dd["policies"]}
            and set(bindings[key]["claims"]) == set(dd["considered_claims"]),
            "model Decision omitted contributing governance or claims")
    declarations = [rules[r]["model"] for r in revisions]
    review_actions = [m["review"] for m in declarations if m.get("review") is not None]
    require(not review_actions or all(r == review_actions[0] for r in review_actions),
            "incompatible contributing review actions")
    review_action = review_actions[0] if review_actions else None
    require(hd.get("review_action") == review_action, "handoff dropped the required compensatory action")
    notices = set(m["required_disclosure"] for m in declarations)
    # Each original requirement must be present; joining does not rewrite it.
    notice = hd["required_disclosure"]
    require(all(n in notice for n in notices) and notice == model["required_disclosure"],
            "handoff omitted an original disclosure requirement")
    for grant in declarations:
        require(hb["destination"] in grant["destinations"] and grant["use"] == hd["native_use"] == model["use"]
                and grant["recipient_use"] == model["recipient_use"]
                and grant["report_destination"] == model["destination"], "model use differs from signed source grant")
    config_bytes = (root / model["runtime_config"]).read_bytes()
    require(sha(config_bytes) == model["runtime_config_sha256"] == hd["runtime_config_sha256"],
            "native configuration differs from recorded handoff")
    config = json.loads(config_bytes)
    section = config["report_contract"]["section_rules"]["values"]
    require(section["required_attribution"] == notice and section["propagate_source_governance"] is True
            and all(section["required_tool"] == m["required_tool"] for m in declarations),
            "native gate does not carry the complete disclosure/compliance requirement")
    exchanges = [json.loads(line) for line in (root / "native-model/model-http.ndjson").read_text().splitlines()]
    http = [e for e in exchanges if "body_b64" in e]
    for entry in http:
        require(entry["body_sha256"] == "sha256:" + sha(base64.b64decode(entry["body_b64"], validate=True)),
                "HTTP transcript digest mismatch")
    requests = [e for e in http if e["direction"] == "request"]
    responses = [e for e in http if e["direction"] == "response"]
    require(len(requests) == len(responses) == 1, "expected one actual model exchange")
    request = json.loads(base64.b64decode(requests[0]["body_b64"]))
    require(request["tn_object"].encode() == source.forward() and request["use"] == model["use"]
            and request["destination"] == model["destination"], "model transport changed native wire or use")
    require(len(dd["parameters"]) == 1 and request["messages"] == [{"role": "user", "content": dd["parameters"][0]}],
            "model prompt differs from the signed Decision")
    request_digest = "sha256:" + sha(b"tn.llama.request.v2\0" + _canonical_bytes({
        "object_id": source.id, "use": request["use"], "destination": request["destination"],
        "messages": request["messages"],
    }))
    audit = [json.loads(line) for line in (root / "native-model/model-audit.ndjson").read_text().splitlines()]
    observations = {"record_kinds": [r["kind"] for r in audit],
                    "terminal_errors": [r["error_code"] for r in audit if r["kind"] == "terminal_error"],
                    "assurance": "local process observations; this audit tape is not signed remote attestation"}
    if not model["completed"]:
        require(not results and responses[0]["status"] != 200, "failed model run has success evidence")
        error = json.loads(base64.b64decode(responses[0]["body_b64"]))["error"]["code"]
        failures = [r for r in records if r.body["kind"] == "Failure" and r.body["details"]["decision_id"] == decision.id]
        require(len(failures) == 1 and failures[0].body["details"]["native_code"] == error
                and model["output_object_id"] is None and not model["tool_object_ids"],
                "failed native request differs from signed endpoint failure")
        return {"completed": False, "http_status": responses[0]["status"], "input_object_id": source.id,
                "native_error": error, "native_observations": observations}
    require(len(results) == 1 and responses[0]["status"] == 200, "completed model run lacks exact result/response")
    result = results[0]
    rd = result.body["details"]
    require(result.id == model["result_record_id"] and rd["handoff_record_id"] == handoff.id
            and result.body["sequence"] > hb["sequence"] and rd["input_object_id"] == source.id
            and rd["required_disclosure"] == notice and rd["claims"] == bindings[key]["claims"],
            "model result changed its source, conditions, or claim subjects")
    release = NativeModelRelease.from_http_json(base64.b64decode(responses[0]["body_b64"]))
    require(release.output_object.id == model["output_object_id"] == rd["output_object_id"]
            and [o.id for o in release.tool_objects] == model["tool_object_ids"] == rd["tool_object_ids"]
            and len(release.tool_objects) == 1, "report or compliance identity differs from signed result")
    for obj in (release.output_object, *release.tool_objects):
        require(store.get(obj.id) == obj.forward(), "retained native release bytes differ from response")
    review = None
    if review_action is not None:
        review_id = rd.get("review_request_id")
        review_record = by_id.get(review_id)
        require(review_record is not None and review_record.body["kind"] == "ReviewRequested",
                "generated report has no signed review request")
        vb, vd = review_record.body, review_record.body["details"]
        require(review_record.writer == result.writer and vb["call_id"] == hb["call_id"]
                and vb["context_id"] == hb["context_id"]
                and hb["sequence"] < vb["sequence"] < result.body["sequence"]
                and vd["decision_id"] == decision.id and vd["handoff_record_id"] == handoff.id,
                "review request is disconnected from the authorized handoff")
        require(vd["status"] == "pending" and vd["review_action"] == review_action
                and vd["input_object_id"] == source.id and vd["report_object_id"] == release.output_object.id
                and vd["required_disclosure"] == notice and vd["policy_revisions"] == sorted(revisions)
                and vd["report_html_sha256"] == rd["report_html_sha256"],
                "pending review changed its report, action, or source basis")
        review = {"record_id": review_id, "report_object_id": release.output_object.id,
                  "status": "pending", "assignee": review_action["assignee"], "reason": review_action["reason"],
                  "queue_entry": "review-queue/" + review_id[7:] + ".tn"}
        require(model.get("review") == review and json.loads((root / "review.json").read_bytes()) == review,
                "review display differs from its signed pending request")
        require((root / review["queue_entry"]).read_bytes() == store.get(review_id),
                "pending queue does not contain the exact signed review request")
        trace = [json.loads(line) for path in (root / "native-model/model-traces").glob("*.ndjson")
                 for line in path.read_text().splitlines()]
        authorized = {(r["run_id"], r["llama_task_id"], r["llama_slot_id"], json.loads(r["facts_json"]).get("phase"))
                      for r in audit if r["kind"] == "turn_begin_authorized"}
        def turn_key(r):
            return r["run_id"], r["task_id"], r["slot_id"], r["phase"]
        completed = {turn_key(r) for r in trace if r["kind"] == "model_result" and r.get("raw_model_output_b64")}
        notified = [r["task_id"] for r in trace
                    if r["run_id"] == release.run_id and r["kind"] == "turn_started"
                    and turn_key(r) in authorized and turn_key(r) in completed and r.get("rendered_prompt_b64")
                    and notice in base64.b64decode(r["rendered_prompt_b64"]).decode("utf-8")]
        observations["review_notification_task_ids"] = notified
        observations["review_notice_in_model_prompt"] = bool(notified)
        require(observations["review_notice_in_model_prompt"], "no native evidence that the model saw its review instruction")
    use = tn.UseContext(**model["recipient_use"])
    with tn.Session.from_config(str(root / model["recipient_config"])) as recipient:
        reader = NativeOutputReader(recipient, expected_source=source, expected_writer=model["writer"],
            use=use, destination=model["destination"], release_use=tn.UseContext(**model["use"]),
            catalog=catalog, policy_dag=dag, resolve_ancestor=resolve,
            decide=lambda ctx: ctx.use_context == use,
            expected_tool_name=section["required_tool"], expected_tool_sections=("values",))
        opened = reader.open(release)
        for tool in release.tool_objects:
            tool_data = recipient.receive(tool, use=use, groups=["default"],
                decide=lambda ctx, tool=tool: reader._accept(ctx, tool))
            require(tool_data.get("request_digest", group="default") == request_digest,
                    "signed native compliance action belongs to another request/prompt")
        # This is independent artifact inspection after recipient admission. The
        # application release decision already happened in the native service.
        html = opened.get("report_html", group="default")
        require(sha(html.encode()) == rd["report_html_sha256"] and (root / "report.html").read_bytes() == html.encode(),
                "display artifact differs from signed native report")
        parser = PlainReportText()
        parser.feed(html)
        parser.close()
        visible = " ".join("".join(parser.parts).split())
        require(all(" ".join(n.split()) in visible for n in notices), "required notice is absent from plain report text")
    source_node_id = nodes[key]["id"]
    edges = [{"parent": source_node_id, "child": t.id, "operation": section["required_tool"]} for t in release.tool_objects]
    edges += [{"parent": source_node_id if obj.id == source.id else obj.id,
               "child": release.output_object.id, "operation": "generate_report"}
              for obj in (source, *release.tool_objects)]
    result = {"completed": True, "input_object_id": source.id, "output_object_id": release.output_object.id,
              "tool_object_ids": [o.id for o in release.tool_objects], "required_disclosure": notice,
              "visible_report_text": visible, "source_policy_revisions": revisions,
              "source_claim_ids": bindings[key]["claims"], "native_edges": edges,
              "request_digest": request_digest,
              "review": review,
              "native_observations": observations,
              "scope": "Independent native recipient admission and lineage verification; exact transport and signed evidence bindings; notice found in unstyled HTML. Native process logs are supporting observations, not signed remote attestation."}
    (root / "model-lineage.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
