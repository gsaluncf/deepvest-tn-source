"""Offline verification of retained native TN publications and evidence.

The configured observer checks governance without opening primitive payloads.
A retained model report is opened under its recipient use. The run manifest
supplies trusted identities and record-chain checkpoints. No publisher or
Java connection is needed. Cryptography verifies attribution and bindings;
it does not establish the truth of a computation.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import tn
from tn_governed.publications import open_publication_store

from .evidence import EvidenceError, verify_tape
from .primitives import SCHEMA, VALUE_TYPE
from .provision import POLICY


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def ref_key(reference):
    return reference["object_id"], reference["group"], reference["field"]


def reference_node_id(key):
    """Identify a field without conflating siblings in the same publication."""
    if key[1:] == ("values", "value"):
        return key[0]
    return key[0] + "#" + json.dumps(key[1:], ensure_ascii=True, separators=(",", ":"))


def verify(root):
    root = Path(root).resolve()
    run = json.loads((root / "run.json").read_text())
    require(run["schema"] == "tn.primitive-run.v1", "unknown run manifest")
    store = open_publication_store(root / "publications", read_only=True)
    authority, writers = run["authority"], set(run["writers"])
    with store, tn.Session.from_config(str(root / run["observer_config"])) as session, ExitStack() as caches:
        def cached(function):
            # All entries belong to this reader and this fixed verification run.
            result = lru_cache(maxsize=128)(function)
            caches.callback(result.cache_clear)
            return result

        @cached
        def object_(identity):
            return store.read(identity)

        admin = tn.Governance.from_markdown(authority, POLICY, ".tn/config/agents.md", "experiment.administration")
        audit = tn.Governance.from_markdown(authority, POLICY, ".tn/config/agents.md", "experiment.evidence")
        require(admin.fields == run["administration_policy"] and audit.fields == run["audit_policy"],
                "checkpoint governance differs from the configured reader contract")
        def accept_admin(ctx):
            return ctx.writer == authority and len(ctx.policies) == 1 and ctx.governance.matches_contract(admin)
        def open_admin(obj, group):
            admitted = session.governance(obj).accept(
                use=tn.UseContext("offline-verifier", "inspect-evidence", "read"),
                groups=[group], decide=accept_admin)
            return session.open(admitted, [group])
        assignment_obj = object_(run["assignment_id"])
        assignment_data = session.receive(assignment_obj,
            use=tn.UseContext("offline-verifier", "inspect-evidence", "read"),
            groups=["catalog_cfg"], decide=accept_admin)
        assignment = assignment_data.get("assignment", group="catalog_cfg")
        require(assignment["schema"] == "tn.sql-assignment.v1" and assignment["source"] == run["source"],
                "source assignment does not match the retained run")
        dag, catalog = tn.PolicyDag(), tn.DatasetCatalog()
        policy_edges = []
        for identity in assignment["revisions"]:
            rev = tn.PolicyRevision.from_opened(open_admin(object_(identity), "policy_revision"))
            dag.admit(rev, lambda revision, parent: revision.writer == authority)
            policy_edges.extend({"parent": p.revision_id, "child": rev.id, "relation": str(p.relation)}
                                for p in rev.parents)
        policies = {f["revision"]: dag.select(f["revision"], f["scope"], lambda rev: rev.writer == authority)
                    for f in assignment["columns"].values()}
        rules = {f["revision"]: f["rule"] for f in assignment["columns"].values()}
        source_editions = {}
        for info in run["source_editions"]:
            edition = tn.DatasetEdition.from_opened(open_admin(object_(info["record_id"]), "dataset_edition"))
            require((edition.dataset, edition.edition, edition.source_object_id) ==
                    (info["dataset"], info["edition"], info["source_object_id"]), "edition index substitution")
            require(edition.grant_ref == assignment_obj.id and edition.source_writer in writers,
                    "source edition has an unrecognized assignment or writer")
            catalog.admit(edition, dag, lambda record: record.writer == authority)
            source_use = tn.UseContext(run["request"]["application"], run["request"]["purpose"], "inspect-claims")
            selected = catalog.select(edition.dataset, edition.edition, edition.id, source_use)
            selected.verify_source(session.governance(object_(edition.source_object_id)), source_use, ["values", "evidence"])
            source_editions[edition.source_object_id] = edition

        records, by_id, calls = [], {}, defaultdict(list)
        for checkpoint in run["tapes"].values():
            require(checkpoint["writer"] in writers, "unrecognized recorder writer")
            files = sorted((root / checkpoint["path"]).glob("*.tn"))
            require(len(files) == checkpoint["count"], "record count differs from retained checkpoint")
            tape = verify_tape([f.read_bytes() for f in files], session=session, policy=audit,
                expected_writer=checkpoint["writer"], expected_context=checkpoint["context_id"],
                expected_head=checkpoint["head"])
            for record in tape:
                require(record.id not in by_id, "duplicate evidence record across record chains")
                by_id[record.id] = record
                records.append(record)
                if record.body["kind"] != "Monitor":
                    calls[record.body["context_id"], record.body["call_id"]].append(record)

        refusals, sql_reads_by_call = [], {}
        for call_key, call in calls.items():
            decisions = [r for r in call if r.body["kind"] == "Decision"]
            require(len(decisions) == 1, "protected call must have exactly one Decision")
            decision = decisions[0]
            db, dd = decision.body, decision.body["details"]
            for record in call:
                body = record.body
                require(record.writer == decision.writer and all(body[k] == db[k] for k in
                        ("inputs", "operation", "application", "purpose", "destination")),
                        "call evidence changed identity, input, or requested use")
                if body["kind"] != "Decision":
                    require(body["sequence"] > db["sequence"], "operation evidence precedes Decision")
                    if body["kind"] != "Claim":
                        require(body["details"].get("decision_id") == decision.id, "completion cites another Decision")
            terminal = [r.body["kind"] for r in call if r.body["kind"] in
                        ("Result", "Refusal", "Disclosure", "Observation", "Failure", "ModelResult")]
            sql_reads = [r for r in call if r.body["kind"] == "SQLRead"]
            sql_reads_by_call[call_key] = sql_reads
            empty_select = (db["operation"] == "select" and len(sql_reads) == 1
                            and sql_reads[0].body["details"]["row_count"] == 0)
            require(terminal or empty_select, "protected call lacks terminal evidence")
            if dd.get("permitted") is not True:
                require(terminal == ["Refusal"] and all(r.body["kind"] in ("Decision", "Refusal") for r in call),
                        "refused call has protected-stage evidence")
                refusals.append({"decision_id": decision.id, "operation": db["operation"],
                                 "purpose": db["purpose"], "destination": db["destination"],
                                 "following_record_kinds": terminal})
                continue
            require("Refusal" not in terminal, "permitted call was reported as refused")
            require(dd.get("policies"), "permit lacks source policies")
            obligations = set()
            for item in dd["policies"]:
                rule = rules.get(item["revision"])
                require(rule is not None and item["authority"] == authority
                        and item["policy"] == policies[item["revision"]].policy_ref, "permit cites unknown governance")
                require(dd["actor"] in rule["actors"] and db["application"] == rule["application"]
                        and db["purpose"] in rule["purposes"] and db["operation"] in rule["operations"],
                        "permit is outside the signed source assignment")
                if db["operation"] == "reveal":
                    require(db["destination"] in rule["destinations"], "disclosure destination was not permitted")
                if db["operation"] == "model_handoff":
                    grant = rule.get("model", {})
                    require(db["destination"] in grant.get("destinations", []) and grant.get("required_disclosure"),
                            "model handoff lacks a disclosure-conditioned destination grant")
                obligations.update(rule["obligations"])
            require(set(dd["obligations"]) == obligations, "required source obligation omitted")
            require({d["obligation"] for d in dd["discharged"]} == obligations, "obligation was not discharged")
            for discharge in dd["discharged"]:
                monitor = by_id.get(discharge["evidence"])
                require(discharge["satisfied"] is True and monitor is not None, "missing obligation evidence")
                mb = monitor.body
                require(monitor.writer == decision.writer and mb["kind"] == "Monitor"
                        and mb["sequence"] < db["sequence"] and all(mb[k] == db[k] for k in
                        ("context_id", "application", "purpose", "operation", "inputs", "destination"))
                        and mb["details"]["actor"] == dd["actor"], "monitoring did not precede this permit")
            if "Failure" in terminal:
                require(terminal.count("Failure") == 1 and terminal[-1] == "Failure",
                        "successful operation evidence follows a failure")
            else:
                expected = {"reveal": "Disclosure", "compare": "Observation", "truth": "Observation", "is_integer": "Observation", "model_handoff": "ModelResult"}.get(db["operation"], "Result")
                # A SELECT yielding zero rows has SQLRead but no Result.
                require(all(t == expected for t in terminal), "wrong completion kind for operation")

        results = {}
        for record in records:
            if record.body["kind"] == "Result":
                key = ref_key(record.body["details"]["output"])
                require(key not in results, "more than one Result for an incarnation")
                results[key] = record
        visited, visiting, nodes, edges, bindings_by_key = set(), set(), {}, [], {}
        def accepted_value(ctx):
            return (ctx.writer in writers and ctx.object_type == VALUE_TYPE and bool(ctx.policies)
                    and all(p.revision_id in policies and p.matches_contract(policies[p.revision_id]) for p in ctx.policies))
        @cached
        def view(identity):
            obj = object_(identity)
            require(obj.writer in writers and obj.object_type == VALUE_TYPE, "unrecognized ancestor publisher/type")
            result = session.governance(obj)
            require(all(p.revision_id in policies and p.matches_contract(policies[p.revision_id])
                        for p in result.governance.policies), "ancestor substituted a contract")
            return result
        @cached
        def evidence(identity):
            obj = object_(identity)
            working = session.receive(obj, use=tn.UseContext("offline-verifier", "inspect-evidence", "read"),
                                      groups=["evidence"], decide=accepted_value)
            manifest = working.get("manifest", group="evidence")
            require(manifest["schema"] == SCHEMA, "unknown primitive manifest")
            return working, manifest

        @cached
        def lineage(identity):
            return tn.LineageVerifier(max_objects=65536, max_depth=256).verify(view(identity), catalog, dag, view)

        def visit(key):
            if key in visited:
                return
            require(key not in visiting, "cycle in primitive incarnation graph")
            visiting.add(key)
            record = results.get(key)
            require(record is not None, "missing signed Result for a primitive parent")
            rb, rd = record.body, record.body["details"]
            decision = by_id[rd["decision_id"]]
            obj = object_(key[0])
            require(record.writer == obj.writer, "Result is not from the incarnation's publisher")
            working, manifest = evidence(key[0])
            binding = manifest["bindings"][key[2]]
            require(binding["decision_id"] == decision.id and binding["parents"] == rb["inputs"]
                    and binding["claims"] == rd["claims"], "signed binding disagrees with Result")
            require(rd["field_commitment"] == obj.envelope[key[1]]["field_hashes"][key[2]],
                    "Result does not bind the exact native field commitment")
            transition = rd["transition"]
            require(transition["governed_inputs"] == binding["parents"] and transition["retained_claims"] == binding["claims"],
                    "result transition changed its contributing references")
            for literal in transition["literal_inputs"]:
                require(literal["claims"] == [], "literal invents a claimant")
            parent_policies, parent_claims, parent_origins = set(), set(), []
            for parent in binding["parents"]:
                pk = ref_key(parent)
                visit(pk)
                parent_policies.update(nodes[pk]["policy_revisions"])
                pb = bindings_by_key[pk]
                parent_claims.update(pb["claims"])
                for origin in pb["origins"]:
                    if origin not in parent_origins:
                        parent_origins.append(origin)
                edges.append({"parent": reference_node_id(pk), "child": reference_node_id(key),
                              "operation": rb["operation"]})
            actual_policies = {p.revision_id for p in working.policies}
            require(parent_claims <= set(binding["claims"]), "a derivative dropped a contributing claim")
            require(parent_claims == set(decision.body["details"]["considered_claims"]),
                    "Decision did not consider the complete set of contributing claims")
            if binding["parents"]:
                require(parent_policies == actual_policies and parent_origins == binding["origins"],
                        "a derivative changed contributing governance or source origins")
                require(parent_policies == {p["revision"] for p in decision.body["details"]["policies"]},
                        "Decision did not evaluate the complete contributing governance")
            else:
                source = binding["source"]
                require(source is not None and rd["source"] == source and key[0] in source_editions,
                        "source primitive lacks its signed SQL origin or edition")
                require(source_editions[key[0]].dataset == source["table"] + "." + source["column"],
                        "source edition does not identify the assigned SQL column")
                field = assignment["columns"][source["column"]]
                require(actual_policies == {field["revision"]} and source["assignment_id"] == assignment_obj.id
                        and source["snapshot_sha256"] == assignment["source"]["snapshot_sha256"]
                        and source["table"] == assignment["table"] and binding["declared"]["kind"] == field["kind"],
                        "SQL source was hydrated under the wrong column assignment")
                require(actual_policies <= {p["revision"] for p in decision.body["details"]["policies"]},
                        "SQL cell's policy was not included in its query Decision")
                sql_records = sql_reads_by_call[rb["context_id"], rb["call_id"]]
                require(len(sql_records) == 1 and sql_records[0].body["sequence"] < rb["sequence"]
                        and sql_records[0].body["details"]["query_id"] == source["query_id"]
                        and sql_records[0].body["details"]["source"] == assignment["source"]
                        and sql_records[0].body["details"]["assignment_id"] == assignment_obj.id,
                        "source publication does not follow its authorized SQL read")
                row_count = sql_records[0].body["details"]["row_count"]
                require(type(row_count) is int and type(source["row"]) is int
                        and 0 <= source["row"] < row_count,
                        "source primitive has an invalid SQL row ordinal")
                require(key[1] == "values" and key[2] in {"value", "value_" + str(source["row"])},
                        "source primitive SQL field differs from its signed row ordinal")
            for identity in binding["claims"]:
                claim = by_id.get(identity)
                require(claim is not None and claim.body["kind"] == "Claim", "unverifiable contributing claim")
                if identity not in parent_claims:
                    require(rb["operation"] == "attest" and claim.body["call_id"] == rb["call_id"]
                            and claim.writer == record.writer and claim.body["inputs"] == binding["parents"],
                            "new claim does not identify its original subjects and issuer")
            nodes[key] = {"id": reference_node_id(key), "object_id": key[0], "group": key[1], "field": key[2],
                            "reference": {"object_id": key[0], "group": key[1], "field": key[2]},
                            "kind": binding["declared"]["kind"], "operation": rb["operation"],
                            "writer": obj.writer, "decision_id": decision.id, "result_id": record.id,
                            "policy_revisions": sorted(actual_policies), "claims": binding["claims"],
                            "source": binding["source"]}
            bindings_by_key[key] = binding
            visiting.remove(key)
            visited.add(key)
        for key in results:
            visit(key)
        native_lineage = {}
        for name, h in run["outputs"].items():
            key = ref_key(h["reference"])
            require(results[key].id == h["result_id"] and nodes[key]["decision_id"] == h["decision_id"],
                    "exported handle cites different operation evidence")
            checked_lineage = lineage(key[0])
            native_lineage[name] = {"object_ids": checked_lineage.object_ids,
                                    "source_object_ids": checked_lineage.source_object_ids}
        failures = [{"decision_id": r.body["details"]["decision_id"], "operation": r.body["operation"],
                     "error_type": r.body["details"]["error_type"], "writer": r.writer}
                    for r in records if r.body["kind"] == "Failure"]
        summary = {"schema": "tn.primitive-verification.v1", "network_used": False,
                   "observer": session.did, "record_count": len(records),
                   "record_kinds": dict(Counter(r.body["kind"] for r in records)),
                   "incarnations": len(nodes), "derivation_edges": len(edges),
                   "source_editions": len(source_editions), "policy_revisions": len(dag),
                   "policy_edges": policy_edges, "refusals": refusals, "failures": failures,
                   "native_lineage": native_lineage,
                   "verifier_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                              for name in ("verify.py", "verify_model.py") if Path(__file__).with_name(name).exists()},
                   "scope": "Native signatures/integrity; record ordering; source assignments; discharged monitoring; claim and contract retention; native edition and lineage verification. No source primitive payload opened, and no assertion that a signature proves computation truth."}
        if run.get("model"):
            from .verify_model import verify_model
            summary["model"] = verify_model(root, run, records, by_id, rules, nodes, bindings_by_key,
                                            store, catalog, dag, view)
            summary["scope"] += " Model verification additionally admits and opens the signed report under the enrolled recipient's declared use; source primitive payloads remain closed."
        graph = {"nodes": list(nodes.values()), "edges": edges, "policy_edges": policy_edges}
    (root / "verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "dag.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    names = {node["id"]: "n" + str(i) for i, node in enumerate(nodes.values())}
    mermaid = ["flowchart LR"]
    for node in nodes.values():
        identity = node["id"]
        source = node["source"]
        label = (source["column"] + " row " + str(source["row"])) if source else node["operation"]
        mermaid.append(f'  {names[identity]}["{node["kind"]}: {label}<br/>{identity[7:17]}"]')
    mermaid.extend(f'  {names[e["parent"]]} --> {names[e["child"]]}' for e in edges)
    (root / "dag.mmd").write_text("\n".join(mermaid) + "\n", encoding="utf-8")
    model = summary.get("model", {})
    if model.get("completed"):
        native_nodes = [{"id": identity, "kind": "native-compliance", "operation": "log_compliance_creation"}
                        for identity in model["tool_object_ids"]]
        native_nodes.append({"id": model["output_object_id"], "kind": "native-report", "operation": "generate_report"})
        review_edges = []
        if model.get("review"):
            review = model["review"]
            native_nodes.append({"id": review["record_id"], "kind": "pending-review-evidence", "operation": "request_review"})
            review_edges.append({"parent": model["output_object_id"], "child": review["record_id"],
                                 "operation": "request_review", "relation": "signed-subject-reference"})
        model_graph = {"nodes": graph["nodes"] + native_nodes,
                       "edges": edges + model["native_edges"] + review_edges, "policy_edges": policy_edges}
        for node in native_nodes:
            names[node["id"]] = "n" + str(len(names))
            mermaid.append(f'  {names[node["id"]]}["{node["kind"]}<br/>{node["id"][7:17]}"]')
        mermaid.extend(f'  {names[e["parent"]]} --> {names[e["child"]]}' for e in model["native_edges"])
        mermaid.extend(f'  {names[e["parent"]]} -. "review requested" .-> {names[e["child"]]}' for e in review_edges)
        (root / "model-dag.json").write_text(json.dumps(model_graph, indent=2), encoding="utf-8")
        (root / "model-dag.mmd").write_text("\n".join(mermaid) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("network_used", "record_count", "record_kinds", "incarnations",
                                              "derivation_edges", "source_editions", "policy_revisions")}, indent=2))
    print(f"Offline evidence and DAG: {root / 'verification.json'}")
    if summary.get("model", {}).get("completed"):
        print(f"Native report and visible disclosure verified: {summary['model']['output_object_id']}")
        if summary["model"].get("review"):
            print(f"Compensatory action: report generated, model notified, signed review request pending for {summary['model']['review']['assignee']}.")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    verify(parser.parse_args().run_directory)
