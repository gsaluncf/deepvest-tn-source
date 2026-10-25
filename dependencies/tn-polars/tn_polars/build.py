import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


REGO = '''package tn.polars

default decision := {"allow": false, "reason": "no matching release rule"}

rank := {"public": 0, "internal": 1, "confidential": 2}

clearance := data.tn_polars.policy.audiences[input.audience]

deny contains "policy reference does not match compiled policy" if {
    input.policy_ref != data.tn_polars.policy.policy_ref
}

deny contains "operation is not enabled by this policy artifact" if {
    not input.operation in data.tn_polars.policy.operations
}

deny contains "release audience is not enabled by this policy" if {
    input.operation == "release"
    not data.tn_polars.policy.audiences[input.audience]
}

deny contains reason if {
    input.operation == "release"
    some column in input.columns
    rank[column.classification] > rank[clearance]
    reason := concat("", [input.audience, " release has non-", clearance, " output"])
}

deny contains reason if {
    input.operation == "release"
    some source in input.row_selection_sources
    rank[data.tn_polars.policy.fields[source]] > rank[clearance]
    reason := concat("", [input.audience, " release uses a non-", clearance,
                          " row-selection source"])
}

deny contains reason if {
    input.operation == "release"
    some source in input.row_selection_sources
    not data.tn_polars.policy.fields[source]
    reason := concat("", ["row-selection source ", source,
                          " is not classified by this policy"])
}

deny contains "aggregate support is unknown or missing" if {
    input.operation == "release"
    data.tn_polars.policy.release[input.audience]
    not input.aggregate_support.known == true
}

deny contains "aggregate release is disabled" if {
    input.operation == "release"
    rule := data.tn_polars.policy.release[input.audience]
    input.aggregate_support.aggregated == true
    rule.allow_aggregates == false
}

deny contains "aggregate support facts are incomplete" if {
    input.operation == "release"
    data.tn_polars.policy.release[input.audience]
    not is_boolean(input.aggregate_support.aggregated)
}

deny contains "aggregate support facts are incomplete" if {
    input.operation == "release"
    data.tn_polars.policy.release[input.audience]
    input.aggregate_support.aggregated == true
    not is_array(input.aggregate_support.group_sizes)
}

deny contains "aggregate support facts are incomplete" if {
    input.operation == "release"
    data.tn_polars.policy.release[input.audience]
    input.aggregate_support.aggregated == true
    not is_number(input.aggregate_support.output_rows)
}

deny contains "aggregate support facts are missing for released groups" if {
    input.operation == "release"
    data.tn_polars.policy.release[input.audience]
    input.aggregate_support.aggregated == true
    input.aggregate_support.output_rows > 0
    count(input.aggregate_support.group_sizes) == 0
}

deny contains "aggregate group has too few distinct contributing source rows" if {
    input.operation == "release"
    rule := data.tn_polars.policy.release[input.audience]
    input.aggregate_support.aggregated == true
    some size in input.aggregate_support.group_sizes
    size < rule.min_group_size
}

decision := {"allow": true, "reason": "release satisfies compiled TN policy"} if {
    count(deny) == 0
}

decision := {"allow": false, "reason": concat("; ", deny)} if {
    count(deny) > 0
}
'''

DEFAULT_AUDIENCES = {"public": "public", "internal": "confidential"}


def build_opa_wasm(policy, output_dir, audiences=None, *, operations=None):
    """Compile the fixed tn-polars Rego evaluator and policy data to OPA/Wasm.

    ``audiences`` maps an audience name to its clearance level (public,
    internal, or confidential). The default preserves the original
    public/internal pair. ``operations`` explicitly enables additional adapter
    operations in this artifact; omitted means release only.
    """
    from .policy import CompiledPolicy
    policy = CompiledPolicy.from_dict(policy.to_dict())
    audiences = dict(DEFAULT_AUDIENCES if audiences is None else audiences)
    if any(not isinstance(name, str) or not name or not isinstance(level, str)
           or level not in {"public", "internal", "confidential"}
           for name, level in audiences.items()):
        raise ValueError("OPA audiences require nonempty names and supported clearance levels")
    output_dir = Path(output_dir)
    operations = ["release"] if operations is None else list(operations)
    if not operations or any(not isinstance(item, str) or not item for item in operations):
        raise ValueError("OPA artifact operations must be nonempty operation names")
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_rego = output_dir / "tn_polars.rego"
    policy_data = output_dir / "data.json"
    bundle = output_dir / "bundle.tar.gz"
    policy_wasm = output_dir / "policy.wasm"
    policy_rego.write_text(REGO, encoding="utf-8")
    policy_data.write_text(json.dumps({"tn_polars": {"policy": {
        "policy_ref": policy.policy_ref,
        "fields": policy.fields,
        "release": policy.release,
        "audiences": audiences,
        "operations": sorted(set(operations)),
    }}}, indent=2), encoding="utf-8")
    opa = _opa_binary()
    command = [opa, "build", "-t", "wasm", "-e", "tn/polars/decision",
               "-o", str(bundle), str(policy_rego), str(policy_data)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with tarfile.open(bundle, "r:gz") as archive:
        member = next(item for item in archive.getmembers() if item.name.endswith("policy.wasm"))
        with archive.extractfile(member) as source:
            policy_wasm.write_bytes(source.read())
    return policy_wasm, policy_data


def _opa_binary():
    configured = os.environ.get("TN_POLARS_OPA")
    if configured:
        return configured
    bundled = Path(__file__).resolve().parents[1] / ".tools" / "opa.exe"
    if bundled.exists():
        return str(bundled)
    found = shutil.which("opa")
    if found:
        return found
    raise RuntimeError("OPA binary not found; set TN_POLARS_OPA or install .tools/opa.exe")
