"""Repeated, interleaved measurements of the governed SQL-to-report path."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from .metrics import Profiler
from .run import run_case

WORKLOADS = [
    ("base", 1, 1, 1),
    ("rows4", 4, 1, 1),
    ("rows8", 8, 1, 1),
    ("depth4", 1, 4, 1),
    ("depth8", 1, 8, 1),
    ("branches2", 1, 1, 2),
    ("branches4", 1, 1, 4),
]


def graph_signature(root):
    """Normalize random identities while retaining operation and source structure."""
    path = root / "dag.json"
    if not path.is_file():
        return None
    graph = json.loads(path.read_text())
    by_id = {n["id"]: n for n in graph["nodes"]}
    parents = defaultdict(list)
    for edge in graph["edges"]:
        parents[edge["child"]].append(edge["parent"])
    cache = {}

    def signature(identity):
        if identity not in cache:
            n = by_id[identity]
            source = n["source"]
            semantic = {
                "kind": n["kind"], "operation": n["operation"],
                "source": ({k: source[k] for k in ("table", "column", "row")}
                           if source else None),
                "policies": len(n["policy_revisions"]), "claims": len(n["claims"]),
                "parents": [signature(p) for p in parents[identity]],
            }
            cache[identity] = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
        return cache[identity]

    return sorted(signature(identity) for identity in by_id)


def storage(root):
    from tn_governed.publications import open_publication_store
    with open_publication_store(root / "publications", read_only=True) as store:
        objects = store.publication_sizes()
    graph = json.loads((root / "dag.json").read_text()) if (root / "dag.json").exists() else {"nodes": []}
    values = {n["id"] for n in graph["nodes"]}
    record_files = list((root / "tapes").glob("*/*.tn"))
    return {
        "retained_native_objects": len(objects),
        "retained_native_bytes": sum(objects.values()),
        "value_publication_bytes": sum(size for identity, size in objects.items() if identity in values),
        "physical_store_bytes": sum(p.stat().st_size for p in (root / "publications").rglob("*") if p.is_file()),
        "exported_record_bytes": sum(p.stat().st_size for p in record_files),
        "note": "native bytes include metadata, records and creation snapshots; exported records duplicate store objects",
    }


def summarize(root):
    cases = []
    for path in sorted(root.glob("*/case-summary.json")):
        case = json.loads(path.read_text())
        case["directory"] = path.parent.name
        case["storage"] = storage(path.parent)
        case["graph_signature"] = graph_signature(path.parent)
        cases.append(case)
    groups = defaultdict(list)
    pairs = defaultdict(dict)
    for case in cases:
        identity = (case["rows"], case["depth"], case["branches"], case["with_model"],
                    case.get("publication_storage", "files"))
        groups[(*identity, case["mode"])].append(case)
        repetition = case["directory"].split("-r")[-1].split("-")[0]
        pairs[(*identity, repetition)][case["mode"]] = case
    comparisons = []
    for key, modes in pairs.items():
        if set(modes) != {"primitive", "explicit"}:
            continue
        a, b = modes["primitive"], modes["explicit"]
        comparisons.append({
            "workload": list(key),
            "both_verified": a["verified"] and b["verified"],
            "both_completed": a["completed"] and b["completed"],
            "same_expected_content": a["expected_content"] == b["expected_content"],
            "same_operation_source_structure": (a["graph_signature"] is not None
                                                and a["graph_signature"] == b["graph_signature"]),
            "same_record_counts": a.get("verification", {}).get("record_kinds") == b.get("verification", {}).get("record_kinds"),
        })
    aggregates = []
    for key, runs in sorted(groups.items()):
        successful = [run for run in runs if run["completed"] and run["verified"]]
        stage_names = sorted({name for run in successful for name in run["timings_seconds"]})
        timing = {}
        for name in stage_names:
            values = [run["timings_seconds"][name] for run in successful if name in run["timings_seconds"]]
            timing[name] = {"n": len(values), "median_s": statistics.median(values),
                           "min_s": min(values), "max_s": max(values)}
        aggregates.append({"rows": key[0], "depth": key[1], "branches": key[2],
                           "model_requested": key[3], "publication_storage": key[4],
                           "mode": key[5], "attempts": len(runs),
                           "completed": sum(bool(r["completed"]) for r in runs),
                           "verified": sum(bool(r["verified"]) for r in runs),
                           "timings_successful_verified_only": timing,
                           "unsuccessful_attempts": [{"directory": r["directory"],
                               "error": r.get("error"), "verification_error": r.get("verification_error"),
                               "timings_seconds": r["timings_seconds"]}
                               for r in runs if not (r["completed"] and r["verified"])],
                           "median_native_bytes_successful": (statistics.median(
                               r["storage"]["retained_native_bytes"] for r in successful)
                               if successful else None)})
    result = {"schema": "tn.pvldb-measurements.v1", "cases": cases,
              "paired_comparisons": comparisons, "aggregates": aggregates,
              "scope": "paired scalar APIs sharing SQL/Unity, governed Polars/OPA, native TN and model controls; local wall-time observations",
              "graph_comparison_scope": "operation/source topology plus policy and claim counts; each case separately verifies actual policies, claims and business content"}
    (root / "measurements.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--storage", choices=["files", "sqlite"], default="sqlite")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workloads", nargs="+", choices=[w[0] for w in WORKLOADS], default=[w[0] for w in WORKLOADS])
    parser.add_argument("--model-repetitions", type=int, default=0)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.summarize_only:
        result = summarize(root)
        print(json.dumps({"attempts": len(result["cases"]), "aggregates": result["aggregates"]}, indent=2))
        return
    if args.repetitions < 0 or args.model_repetitions < 0:
        raise ValueError("repetition counts must be nonnegative")
    root.mkdir(parents=True, exist_ok=False)
    schedule = []
    for repetition in range(args.repetitions):
        for name, rows, depth, branches in WORKLOADS:
            if name in args.workloads:
                order = ("primitive", "explicit") if repetition % 2 == 0 else ("explicit", "primitive")
                for mode in order:
                    schedule.append((f"{name}-r{repetition+1}-{mode}", mode, rows, depth, branches, False))
    for repetition in range(args.model_repetitions):
        for mode in (("primitive", "explicit") if repetition % 2 == 0 else ("explicit", "primitive")):
            schedule.append((f"model-r{repetition+1}-{mode}", mode, 4, 1, 1, True))
    (root / "schedule.json").write_text(json.dumps({"created": datetime.now(timezone.utc).isoformat(),
                                                   "schedule": schedule, "storage": args.storage,
                                                   "hidden_retries": False}, indent=2))
    for number, (name, mode, rows, depth, branches, model) in enumerate(schedule, 1):
        print(f"[{number}/{len(schedule)}] {name}: rows={rows} depth={depth} branches={branches} model={model}", flush=True)
        with (root / (name + ".log")).open("w", encoding="utf-8") as log, redirect_stdout(log):
            with Profiler() as profile:
                case = run_case(root / name, mode, rows=rows, depth=depth, branches=branches,
                                model=model, storage=args.storage)
        (root / name / "profile.json").write_text(json.dumps(profile.export(), indent=2))
        print(json.dumps({"case": name, "completed": case["completed"], "verified": case["verified"],
                          "seconds": case["timings_seconds"]["total"], "error": case.get("error"),
                          "verification_error": case.get("verification_error")}), flush=True)
        summarize(root)
        if not case["verified"]:
            raise RuntimeError("Evidence verification failed; inspect the saved case before continuing")
    print(f"Measurements: {root / 'measurements.json'}", flush=True)


if __name__ == "__main__":
    main()
