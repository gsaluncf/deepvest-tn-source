"""Delta Lake release sink: the governance basis rides the table's own commit log."""

import json
from pathlib import Path

from deltalake import CommitProperties, DeltaTable

from .policy import policy_sha256
from .release import GovernedRelease, PolicyDenied

COMMIT_SCHEMA = "tn.polars-delta-release.v1"


def write_delta_release(governed, path, decision, covered_columns, *, materialized=None):
    """Write the collected frame as a Delta table with governance commit metadata."""
    path = Path(path)
    metadata = {
        "tn_schema": COMMIT_SCHEMA,
        "policy_ref": governed.policy.policy_ref,
        "compiled_policy_sha256": policy_sha256(governed.policy),
        "opa_decision_id": decision.decision_id,
        "opa_reason": decision.reason,
        "covered_columns": json.dumps(sorted(covered_columns)),
        "tn_governance_policy": governed.tn_governance.get("policy", ""),
    }
    (materialized if materialized is not None else governed.frame.collect()).write_delta(
        path,
        delta_write_options={
            "commit_properties": CommitProperties(custom_metadata=metadata)
        },
    )
    return DeltaTable(path).version()


def read_governed_delta(path, *, current_policy=None):
    """Verify a Delta release before a later governed loop consumes it.

    The basis is read from the latest commit's custom metadata, so the next
    hop needs the table alone. ``current_policy`` applies the same policy-skew
    fail-closed rule as file releases.
    """
    table = DeltaTable(Path(path))
    commit = table.history(1)[0]
    if commit.get("tn_schema") != COMMIT_SCHEMA:
        raise PolicyDenied(
            f"delta table {Path(path).name} carries no tn-polars governance commit. "
            "Its latest commit was not written by release_delta, so the governance "
            "basis for the current table state is unknown. This also fires when "
            "someone appends to a governed table with a plain writer; the table "
            "must be re-released through the governed path."
        )
    if current_policy is not None:
        if commit.get("compiled_policy_sha256") != policy_sha256(current_policy):
            raise PolicyDenied(
                f"delta release {Path(path).name} was compiled under a deprecated "
                "policy; failing closed until the release is re-issued"
            )
    manifest = {
        "schema": COMMIT_SCHEMA,
        "policy_ref": commit.get("policy_ref"),
        "compiled_policy_sha256": commit.get("compiled_policy_sha256"),
        "covered_columns": json.loads(commit.get("covered_columns", "[]")),
        "delta_version": table.version(),
    }
    return GovernedRelease(path=Path(path), manifest=manifest)
