"""Offline verification of the fixed SQL/dataframe calculation and OPA evidence.

The original primitive verifier checks the complete record/claim/publication DAG.
This additional reader checks the dataframe-specific signed observations, loads
the exact evaluator bytes bound by source editions, and independently calculates
the expected result from the retained SQLite snapshot.
"""

from collections import Counter
from contextlib import ExitStack, closing
from functools import lru_cache
import hashlib
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sqlite3
import zipfile

import tn
from tn_polars.opa import OpaWasmEvaluator

from deepvest_tn.publications import open_publication_store, write_once
from experiments.governed_strings.evidence import EvidenceError, verify_tape
from experiments.governed_strings.primitives import SCHEMA, VALUE_TYPE
from experiments.governed_strings.numeric_codec import decode, encode, record_value
from experiments.governed_strings.metadata import describe
from experiments.governed_strings.provision import POLICY


_STAGES = ("load", "filter", "sort", "with_columns", "select", "release")
_PLAN = [
    {"operation": "filter", "column": "value", "comparison": ">=", "literal": 0},
    {"operation": "sort", "columns": ["name"], "descending": False},
    {"operation": "with_columns", "column": "value", "operator": "+", "literal": 250, "output": "adjusted"},
    {"operation": "select", "aggregate": "sum", "column": "adjusted", "output": "amount"},
]
_FILES = {
    "compiled_policy.json": "compiled_policy_sha256", "profile.json": "profile_sha256",
    "policy.wasm": "wasm_sha256", "data.json": "data_sha256",
}


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _artifact_facts(artifact):
    return {name: getattr(artifact, name) for name in ("policy_revision", *_FILES.values())}


def _bundle_members(marker, names):
    """Read transport files independently of the application's bundle adapter."""
    if marker.name == "bundle.zip":
        _require(not (marker.parent / "bundle.json").exists(), "conflicting edition bundle formats")
        with zipfile.ZipFile(marker) as archive:
            infos = archive.infolist()
            _require(len(infos) == len({item.filename for item in infos}), "duplicate archive members")
            _require(all(item.compress_type == zipfile.ZIP_STORED for item in infos),
                     "edition archive contains compressed members")
            return {name: archive.read(name) for name in names}
    _require(not (marker.parent / "bundle.zip").exists(), "conflicting edition bundle formats")
    return {name: (marker.parent / name).read_bytes() for name in names}


def verify_frame(root):
    """Verify a completed fixed-plan dataframe result without accessing Unity."""
    root = Path(root).resolve()
    run = json.loads((root / "run.json").read_text())
    _require(run["schema"] == "tn.primitive-run.v1", "unknown run schema")
    store = open_publication_store(root / "publications", read_only=True)
    authority, writers = run["authority"], set(run["writers"])
    use = tn.UseContext(run["request"]["application"], run["request"]["purpose"], "verify")

    @lru_cache(maxsize=128)
    def object_(identity):
        obj = store.read(identity)
        _require(obj.id == identity, "retained native publication identity differs")
        return obj

    with store, tn.Session.from_config(str(root / run["observer_config"])) as session, ExitStack() as caches:
        caches.callback(object_.cache_clear)

        @lru_cache(maxsize=128)
        def governance(identity):
            return session.governance(object_(identity))

        caches.callback(governance.cache_clear)
        admin = tn.Governance.from_markdown(authority, POLICY, ".tn/config/agents.md", "experiment.administration")
        audit = tn.Governance.from_markdown(authority, POLICY, ".tn/config/agents.md", "experiment.evidence")
        _require(admin.fields == run["administration_policy"] and audit.fields == run["audit_policy"],
                 "verification contracts differ from checkpoint")

        def accept_admin(ctx):
            return ctx.writer == authority and len(ctx.policies) == 1 and ctx.governance.matches_contract(admin)

        def open_admin(obj, group):
            admitted = governance(obj.id).accept(
                use=tn.UseContext("offline-verifier", "inspect-evidence", "read"), groups=[group], decide=accept_admin)
            return session.open(admitted, [group])

        assignment_obj = object_(run["assignment_id"])
        assignment = session.receive(assignment_obj,
            use=tn.UseContext("offline-verifier", "inspect-evidence", "read"),
            groups=["catalog_cfg"], decide=accept_admin).get("assignment", group="catalog_cfg")
        _require(assignment["source"] == run["source"] and assignment["table"] == "market_data",
                 "dataframe assignment refers to another SQL snapshot")
        numeric_kind = assignment["columns"]["value"]["kind"]
        _require(numeric_kind in {"int", "float", "Decimal"}, "unsupported SQL numeric type")
        numeric_type = {"int": int, "float": float, "Decimal": Decimal}[numeric_kind]
        numeric_profile = assignment["columns"]["value"]["rule"].get(
            "dataframe_numeric", {"kind": "int", "dtype": "Int64"})
        if numeric_kind == "Decimal":
            precision, scale = numeric_profile.get("precision"), numeric_profile.get("scale")
            _require(type(precision) is int and type(scale) is int and 0 <= scale <= precision <= 38
                     and precision > 0, "invalid signed Decimal precision or scale")
            expected_profile = {"kind": "Decimal", "precision": precision, "scale": scale,
                                "dtype": f"Decimal(precision={precision}, scale={scale})"}
        else:
            expected_profile = {"kind": numeric_kind, "dtype": {"int": "Int64", "float": "Float64"}[numeric_kind]}
        _require(numeric_profile == expected_profile, "numeric profile contradicts the signed SQL kind")
        for field in assignment["columns"].values():
            _require(field["rule"].get("dataframe_numeric", {"kind": "int", "dtype": "Int64"}) == numeric_profile,
                     "source contracts disagree about dataframe numeric type")
        expected_schema = {"name": "String", "value": numeric_profile["dtype"]}
        dag, catalog = tn.PolicyDag(), tn.DatasetCatalog()
        for identity in assignment["revisions"]:
            revision = tn.PolicyRevision.from_opened(open_admin(object_(identity), "policy_revision"))
            dag.admit(revision, lambda item, parent: item.writer == authority)
        policies = {}
        policy_columns = {}
        for column, field in assignment["columns"].items():
            _require(column in {"name", "value"} and "dataframe_calculate" in field["rule"]["operations"],
                     "dataframe operation is absent from a contributing assignment")
            policies[field["revision"]] = dag.select(field["revision"], field["scope"], lambda item: item.writer == authority)
            policy_columns[field["revision"]] = column
        _require(set(policy_columns.values()) == {"name", "value"}, "fixed calculation needs both source contracts")

        editions, source_editions, artifacts = {}, {}, {}
        for info in run["source_editions"]:
            edition = tn.DatasetEdition.from_opened(open_admin(object_(info["record_id"]), "dataset_edition"))
            _require((edition.dataset, edition.edition, edition.source_object_id) ==
                     (info["dataset"], info["edition"], info["source_object_id"]), "source edition index differs")
            _require(edition.grant_ref == assignment_obj.id and edition.source_writer in writers,
                     "source edition authority differs")
            catalog.admit(edition, dag, lambda item: item.writer == authority)
            selected = catalog.select(edition.dataset, edition.edition, edition.id, use)
            selected.verify_source(governance(edition.source_object_id), use, ["values", "evidence"])
            required = {item.revision_id for item in edition.contracts}
            _require({item.policy_revision for item in edition.evaluator_artifacts} == required,
                     "edition lacks a complete signed evaluator set")
            for artifact in edition.evaluator_artifacts:
                value = _artifact_facts(artifact)
                _require(artifact.policy_revision not in artifacts or artifacts[artifact.policy_revision] == value,
                         "one policy has conflicting signed evaluator sets")
                artifacts[artifact.policy_revision] = value
            editions[edition.id] = edition
            source_editions[edition.source_object_id] = edition

        records = []
        for checkpoint in run["tapes"].values():
            _require(checkpoint["writer"] in writers, "unrecognized recorder")
            files = sorted((root / checkpoint["path"]).glob("*.tn"))
            _require(len(files) == checkpoint["count"], "record checkpoint count differs")
            records.extend(verify_tape([path.read_bytes() for path in files], session=session, policy=audit,
                expected_writer=checkpoint["writer"], expected_context=checkpoint["context_id"], expected_head=checkpoint["head"]))
        by_id = {}
        for record in records:
            _require(record.id not in by_id, "duplicate evidence record across tapes")
            by_id[record.id] = record
        frame_results = [record for record in records if record.body["kind"] == "Result"
                         and record.body["operation"] == "dataframe_calculate"]
        _require(len(frame_results) == 1, "expected one completed dataframe Result")
        result = frame_results[0]
        facts = result.body["details"]["dataframe"]
        decision = by_id[result.body["details"]["decision_id"]]
        _require(decision.body["kind"] == "Decision" and decision.body["details"]["permitted"] is True,
                 "dataframe Result has no completed permit")
        _require(facts["schema"] == "tn.pvldb-dataframe.v1" and facts["plan"] == _PLAN,
                 "dataframe evidence is not the fixed evaluated plan")
        _require(decision.body["details"]["parameters"] == [_PLAN],
                 "dataframe Result plan differs from the authorized operation parameters")
        _require(facts["input_references"] == result.body["inputs"] == decision.body["inputs"],
                 "dataframe facts changed the protected inputs")
        _require(result.body["application"] == use.application and result.body["purpose"] == use.purpose,
                 "dataframe evidence refers to another request context")
        if "dataframe_numeric" in assignment["columns"]["value"]["rule"]:
            _require(facts.get("numeric_profile") == numeric_profile
                     and facts.get("input_schema") == expected_schema
                     and facts.get("output_schema") == {"amount": numeric_profile["dtype"]},
                     "signed execution schema differs from the assigned numeric profile")
        expected_export = {**facts, "decision_id": decision.id, "result_id": result.id,
                           "output_reference": result.body["details"]["output"]}
        _require(run["evaluation"]["dataframe"] == [expected_export],
                 "exported dataframe observations differ from the signed Result")
        _require(run["outputs"]["dataframe_result"]["result_id"] == result.id
                 and run["outputs"]["dataframe_result"]["reference"] == result.body["details"]["output"],
                 "dataframe output handle differs from its signed Result")

        def accepted_value(ctx):
            return (ctx.writer in writers and ctx.object_type == VALUE_TYPE and bool(ctx.policies)
                and all(policy.revision_id in policies and policy.matches_contract(policies[policy.revision_id])
                        for policy in ctx.policies)
                and ctx.use_context == use
                and (not ctx.governance.dataset_bindings or catalog.accepts(ctx, dag)))

        def ancestor(identity):
            obj = object_(identity)
            view = governance(identity)
            _require(obj.writer in writers and obj.object_type == VALUE_TYPE
                     and all(policy.revision_id in policies
                             and policy.matches_contract(policies[policy.revision_id])
                             for policy in view.governance.policies),
                     "native ancestor has an unrecognized writer or contract")
            return view

        @lru_cache(maxsize=128)
        def open_publication(identity):
            obj = object_(identity)
            tn.LineageVerifier(max_objects=65536, max_depth=256).verify(governance(identity), catalog, dag, ancestor)
            edition = source_editions.get(obj.id)
            selection = (catalog.select(edition.dataset, edition.edition, edition.id, use)
                         if edition is not None else None)
            data = session.receive(obj, use=use, groups=["values", "evidence"], selection=selection, decide=accepted_value)
            manifest = data.get("manifest", group="evidence")
            _require(manifest["schema"] == SCHEMA, "unknown primitive binding schema")
            return obj, data, manifest

        caches.callback(open_publication.cache_clear)

        def open_value(reference):
            _require(reference["group"] == "values", "unexpected business group")
            obj, data, manifest = open_publication(reference["object_id"])
            binding = manifest["bindings"][reference["field"]]
            payload = data.get(reference["field"], group="values")
            _require(type(payload) is str, "primitive wire payload is not a string")
            kind = binding["declared"]["kind"]
            _require(kind in {"str", numeric_kind}, "unsupported primitive kind")
            value = decode(payload, kind)
            _require(encode(value) == payload and describe(value) == binding["declared"],
                     "numeric encoding or declared facts differ from the opened value")
            return obj, data, binding, value

        database = root / "market.sqlite"
        snapshot_hash = _digest(database.read_bytes())
        _require(snapshot_hash == assignment["source"]["snapshot_sha256"], "retained SQLite snapshot differs from signed assignment")
        count = facts["source_row_count"]
        _require(type(count) is int and count > 0 and count == run["evaluation"]["rows"], "source row count differs")
        sqlite3.register_converter("DECIMAL_TEXT", lambda raw: Decimal(raw.decode("ascii")))
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, detect_types=sqlite3.PARSE_DECLTYPES)) as connection:
            connection.execute("PRAGMA query_only = ON")
            selected_rows = connection.execute("SELECT name, value FROM market_data LIMIT ?", (count,)).fetchall()
        _require(_digest(database.read_bytes()) == snapshot_hash and len(selected_rows) == count,
                 "SQL reference calculation changed the snapshot or row count")
        filtered_rows = sorted((row for row in selected_rows if row[1] >= 0), key=lambda row: row[0])
        with localcontext() as decimal_context:
            decimal_context.prec = max(50, 40 + len(str(count)))
            expected_amount = sum((row[1] + 250 for row in filtered_rows), numeric_type(0))
            if numeric_kind == "Decimal":
                expected_amount = expected_amount.quantize(Decimal(1).scaleb(-numeric_profile["scale"]))
        input_roots = {"name": set(), "value": set()}
        input_commitments, retained_editions = [], {}
        _require(len(facts["source_cells"]) == 2 * count and len(facts["input_references"]) == 2 * count,
                 "dataframe omitted or duplicated an input cell")
        for index, cell in enumerate(facts["source_cells"]):
            column, row = ("name" if index % 2 == 0 else "value"), index // 2
            reference = facts["input_references"][index]
            _require(cell["row"] == row and cell["column"] == column and cell["reference"] == reference,
                     "dataframe cell position differs from signed inputs")
            obj, data, binding, value = open_value(reference)
            _require(value == selected_rows[row][index % 2] and type(value) is (str if column == "name" else numeric_type),
                     "governed cell differs from the retained SQL value")
            _require(len(data.dataset_bindings) == 1 and len(binding["origins"]) == 1,
                     "input cell has ambiguous original source")
            source = binding["origins"][0]
            native_binding = data.dataset_bindings[0]
            edition = editions[native_binding.edition_record_id]
            _require(source["column"] == column and source["row"] == row and source["table"] == "market_data"
                     and source["snapshot_sha256"] == snapshot_hash and source["assignment_id"] == assignment_obj.id,
                     "input origin differs from its signed SQL assignment")
            _require(cell["source_object_id"] == edition.source_object_id == native_binding.source_object_id
                     and cell["edition_record_id"] == edition.id, "dataframe cell substituted an edition")
            expected_root = edition.source_object_id + "::" + column
            _require(cell["root"] == expected_root, "column lineage root differs from exact source publication")
            input_roots[column].add(expected_root)
            retained_editions[edition.id] = {"dataset": edition.dataset, "edition": edition.edition,
                "record_id": edition.id, "source_object_id": edition.source_object_id,
                "contracts": [contract.revision_id for contract in edition.contracts]}
            input_commitments.append({"reference": reference,
                "field_commitment": obj.envelope[reference["group"]]["field_hashes"][reference["field"]]})
        _require(facts["source_editions"] == list(retained_editions.values()), "frame edition observations differ from native records")
        _require(facts["input_columns"] == {name: sorted(roots) for name, roots in input_roots.items()},
                 "input column lineage omits source roots")
        _require(facts["output_columns"] == {"amount": sorted(input_roots["value"])}
                 and facts["row_selection_sources"] == sorted(input_roots["name"] | input_roots["value"]),
                 "aggregate or row-selection lineage differs from filter/sort dependencies")

        # Locate immutable bundles by their indexed native edition identities.
        bundles = {}
        markers = [*(root / "editions").glob("*/*/bundle.json"),
                   *(root / "editions").glob("*/*/bundle.zip")]
        for path in markers:
            locator = json.loads(_bundle_members(path, ["bundle.json"])["bundle.json"])
            _require(locator["record_id"] not in bundles, "duplicate edition bundle locator")
            bundles[locator["record_id"]] = path
        evaluators, artifact_hashes = {}, {}
        _require({item["policy_revision"]: item for item in facts["policy_artifacts"]} == artifacts,
                 "signed dataframe artifact observations differ from source editions")
        for edition in editions.values():
            marker = bundles[edition.id]
            names = [f"evaluators/{artifact.policy_revision[7:]}/{name}"
                     for artifact in edition.evaluator_artifacts for name in _FILES]
            members = _bundle_members(marker, ["edition.tn", *names])
            _require(members["edition.tn"] == store.get(edition.id), "bundle edition differs from retained native wire")
            for artifact in edition.evaluator_artifacts:
                revision = artifact.policy_revision
                member_prefix = f"evaluators/{revision[7:]}/"
                payload = {name: members[member_prefix + name] for name in _FILES}
                for name, attribute in _FILES.items():
                    actual = _digest(payload[name])
                    _require(actual == getattr(artifact, attribute), "evaluator artifact differs from its signed edition")
                    location = (marker.relative_to(root).as_posix() + "!/" + member_prefix + name
                                if marker.name == "bundle.zip" else
                                (marker.parent / member_prefix / name).relative_to(root).as_posix())
                    artifact_hashes[location] = actual
                compiled, profile = json.loads(payload["compiled_policy.json"]), json.loads(payload["profile.json"])
                _require(compiled["policy_ref"] == policies[revision].policy_ref
                         and compiled["fields"] == {"name": "internal", "value": "internal"}
                         and profile["dtypes"] == expected_schema,
                         "evaluator metadata differs from the fixed source schema or contract")
                if revision not in evaluators:
                    evaluators[revision] = OpaWasmEvaluator.from_bytes(payload["policy.wasm"], payload["data.json"])

        expected_pairs = Counter((stage, revision) for stage in _STAGES for revision in policies)
        observed_pairs = Counter()
        _require(len(facts["opa_decisions"]) == len(_STAGES) * len(policies), "missing or additional OPA decisions")
        for index, recorded in enumerate(facts["opa_decisions"]):
            request = recorded["request"]
            stage = _STAGES[index // len(policies)]
            revision = request["policy_revision"]
            _require(revision in policies and request["operation"] == stage, "OPA stage order or contributing contract differs")
            observed_pairs[stage, revision] += 1
            local_column = policy_columns[revision]
            fields = {"name": "name", "value": "value"}
            if stage == "with_columns":
                fields["adjusted"] = "value"
            elif stage in {"select", "release"}:
                fields = {"amount": "value"}
            expected_columns = [{"name": name, "sources": [source] if source == local_column else [],
                                 "classification": "internal" if source == local_column else "public"}
                                for name, source in fields.items()]
            row_dependencies = set() if stage == "load" else {"value"} if stage == "filter" else {"name", "value"}
            _require(request["columns"] == expected_columns
                     and request["row_selection_sources"] == ([local_column] if local_column in row_dependencies else []),
                     "OPA request omitted or changed its column/row dependencies")
            _require(request["policy_ref"] == policies[revision].policy_ref
                     and request["audience"] == ("governed-runtime" if stage == "release" else use.application)
                     and request["use"] == {"application": use.application, "purpose": use.purpose, "operation": "dataframe_calculate"},
                     "OPA request changed the accepted application use or destination")
            expected_support = ({"known": True, "aggregated": True, "group_sizes": [len(filtered_rows)], "output_rows": 1}
                                if stage == "release" else {"known": False})
            _require(request["aggregate_support"] == expected_support, "OPA aggregate support differs from the retained source rows")
            actual = evaluators[revision].decide(request)
            _require(recorded["allow"] is True and actual.allow is True
                     and actual.decision_id == recorded["decision_id"] and actual.reason == recorded["reason"],
                     "offline OPA replay differs from the signed operation observation")
        _require(observed_pairs == expected_pairs, "not every contributing contract covered every dataframe stage")

        output_commitments = {}
        for label, expected in (("dataframe_result", expected_amount), ("amount", expected_amount + run["evaluation"]["depth"])):
            reference = run["outputs"][label]["reference"]
            obj, _data, _binding, value = open_value(reference)
            _require(type(value) is numeric_type and value == expected, "encrypted numeric result differs from independent reference calculation")
            output_commitments[label] = {"reference": reference, "value": value,
                "field_commitment": obj.envelope[reference["group"]]["field_hashes"][reference["field"]]}
        _require(facts["output"] == {"column": "amount", "value": record_value(expected_amount)}, "signed dataframe output differs from its encrypted value")
        _require(result.body["details"]["field_commitment"] == output_commitments["dataframe_result"]["field_commitment"],
                 "dataframe Result field commitment differs from native output")
        summary = {"schema": "tn.pvldb-frame-verification.v1", "network_used": False,
            "result_record_id": result.id, "decision_record_id": decision.id, "observer": session.did,
            "verify_use": {"application": use.application, "purpose": use.purpose, "operation": use.operation},
            "source_snapshot_sha256": snapshot_hash, "source_rows": count, "filtered_rows": len(filtered_rows),
            "opa_decisions_replayed": len(facts["opa_decisions"]), "policy_revisions": sorted(policies),
            "stages_per_policy": list(_STAGES), "input_commitments": input_commitments,
            "output_commitments": output_commitments, "artifact_sha256": artifact_hashes,
            "verifier_source_sha256": _digest(Path(__file__).read_bytes())}
    write_once(root / "frame-verification.json", json.dumps(record_value(summary), indent=2).encode())
    return record_value(summary)
