"""Canonical TN object handoffs between application processing stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import tn


def originate_governed_object(
    session: tn.Session,
    fields: dict[str, object],
    policy: tn.Governance,
    *,
    object_type: str,
    group: str,
) -> tn.DataObject:
    """Create the service's native working object with its accepted contract."""
    return session.create(fields, policy, object_type=object_type, group=group)


def receive_governed_object(
    session: tn.Session,
    wire: tn.GovernedObject | str | bytes,
    *,
    use: tn.UseContext,
    groups: Sequence[str],
    decide: Callable[[tn.AdmissionContext], bool],
    selection: tn.DatasetSelection | None = None,
) -> tn.DataObject:
    """Delegate governance admission and selected opening to the Rust session."""
    return session.receive(
        wire, use=use, groups=groups, decide=decide, selection=selection,
    )


# Seal/unseal adapters support the version 1 chained-model execution path.

TN_AGENTS_FIELDS = (
    "instruction",
    "use_for",
    "do_not_use_for",
    "consequences",
    "on_violation_or_error",
    "policy",
)


@dataclass(frozen=True, slots=True)
class OpenedGovernedObject:
    """The readable portions of a verified TN object inside one endpoint."""

    event_type: str
    business: dict[str, object]
    governance: dict[str, object]


def seal_governed_object(
    publisher_config_path: Path,
    object_type: str,
    *,
    business_group: str,
    business_fields: Mapping[str, object],
    governance: Mapping[str, object],
) -> tn.SealedObject:
    """Seal business fields and the standard ``tn.agents`` fields together."""
    missing = [name for name in TN_AGENTS_FIELDS if not governance.get(name)]
    if missing:
        raise ValueError("TN governance is missing: " + ", ".join(missing))

    tn.init(Path(publisher_config_path).resolve(), link=False, stdout=False)
    sealed = tn.seal(
        object_type,
        **dict(business_fields),
        **{name: governance[name] for name in TN_AGENTS_FIELDS},
    )
    encrypted_groups = {
        name
        for name, value in sealed.items()
        if isinstance(value, Mapping) and "ciphertext" in value
    }
    required_groups = {business_group, "tn.agents"}
    if not required_groups.issubset(encrypted_groups):
        raise ValueError(
            "TN object did not contain the required encrypted groups: "
            + ", ".join(sorted(required_groups - encrypted_groups))
        )
    return sealed


def open_governed_object(
    recipient_config_path: Path,
    source: tn.SealedObject | Mapping[str, object] | str | bytes | Path,
    *,
    expected_type: str,
    business_group: str,
) -> OpenedGovernedObject:
    """Verify a TN object and require readable business and governance groups."""
    tn.init(Path(recipient_config_path).resolve(), link=False, stdout=False)
    opened = tn.unseal(source, raw=True)
    envelope = opened["envelope"]
    if envelope.get("event_type") != expected_type:
        raise ValueError(
            f"expected TN object type {expected_type!r}, got {envelope.get('event_type')!r}"
        )
    valid = opened.get("valid")
    if not isinstance(valid, Mapping) or not all(valid.values()):
        raise ValueError("TN object did not pass canonical verification")

    plaintext = opened.get("plaintext")
    if not isinstance(plaintext, Mapping):
        raise ValueError("TN object did not produce a plaintext group map")
    business = _required_plaintext_group(plaintext, business_group)
    governance = _required_plaintext_group(plaintext, "tn.agents")
    missing = [name for name in TN_AGENTS_FIELDS if not governance.get(name)]
    if missing:
        raise ValueError("opened tn.agents is missing: " + ", ".join(missing))
    return OpenedGovernedObject(
        event_type=expected_type,
        business=business,
        governance=governance,
    )


def _required_plaintext_group(
    plaintext: Mapping[str, object], group: str
) -> dict[str, object]:
    body = plaintext.get(group)
    if not isinstance(body, Mapping) or body.get("$no_read_key") is True:
        raise PermissionError(f"TN endpoint has no reader material for {group}")
    if body.get("$decrypt_error") is True:
        raise ValueError(f"TN endpoint could not open {group}")
    return dict(body)
