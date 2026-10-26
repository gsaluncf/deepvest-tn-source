"""Admission: schema pinning between a Polars frame and its compiled policy."""

from .frame import GovernedLazyFrame
from .release import PolicyDenied


class SchemaPinViolation(PolicyDenied):
    """The arriving frame does not match the schema the policy was compiled for."""


def admit(frame, policy, tn_governance, evaluator, *, dtypes=None, label_map=None):
    """Return a governed frame only when the frame schema equals the policy schema.

    Both schemas must contain the same columns. Admission fails if an input
    column has no policy or a column required by the policy is missing.
    """
    schema = frame.collect_schema()
    arrived = set(schema.names())
    declared = set(policy.fields)
    unmapped = sorted(arrived - declared)
    missing = sorted(declared - arrived)
    if unmapped or missing:
        raise SchemaPinViolation(_pin_message(unmapped, missing))
    if dtypes:
        _pin_dtypes(schema, dtypes)
    return GovernedLazyFrame(frame, policy, tn_governance, evaluator,
                             label_map=label_map)


def _pin_dtypes(schema, dtypes):
    changed = sorted(
        f"{name} arrived as {schema[name]} but the profile declares {expected}"
        for name, expected in dtypes.items()
        if str(schema[name]) != str(expected)
    )
    if changed:
        raise SchemaPinViolation(
            "column types differ from the declared profile: "
            + "; ".join(changed)
            + ". A type change can silently alter what a rule covers, so "
            "admission halts. Platform ops updates the profile if the change "
            "is expected; otherwise the feed itself changed and data "
            "governance must re-review the reading."
        )


def _pin_message(unmapped, missing):
    parts = []
    if unmapped:
        parts.append(
            "columns arrived that the compiled policy does not classify: "
            + ", ".join(unmapped)
            + ". The vendor changed the record shape, so the generated reading "
            "cannot cover these columns. Admission halts until data governance "
            "regenerates and reviews the reading against the current licence text."
        )
    if missing:
        parts.append(
            "columns the compiled policy classifies are absent: "
            + ", ".join(missing)
            + ". Either this record belongs to a different profile (a routing "
            "problem for platform ops) or the reading is stale for this feed."
        )
    return " ".join(parts)
