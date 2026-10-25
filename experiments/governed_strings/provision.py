"""One-time native BTN enrollment for the local demonstration, outside processing."""

import os
from pathlib import Path
from uuid import uuid4

import yaml
from tn._native.btn import PublisherState
from tn.signing import DeviceKey

POLICY = """## experiment.administration
### instruction
Signed catalog configuration, native policy revisions, and dataset edition records.
### use_for
Resolve source and request governance, retain evidence, and verify native lineage.
### do_not_use_for
Treating catalog locations as unsigned authorization or issuing request-time reader keys.
### consequences
Decisions remain attributable to each endpoint.
### on_violation_or_error
Refuse before opening business content.

## experiment.evidence
### instruction
Attributable decisions, operation observations, results, and disclosures.
### use_for
Offline verification and monitoring by enrolled evidence readers.
### do_not_use_for
Treating a writer's statement as automatic endorsement by an earlier source.
### consequences
Recipients evaluate the evidence in their own context.
### on_violation_or_error
Retain the failure and refuse the next protected step.
"""


def provision(root, *, roles=None, groups=None, read_only_roles=("observer",)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    groups = list(groups or ["tn.agents", "values", "evidence", "catalog_cfg", "policy_revision", "dataset_edition"])
    roles = list(roles or ["catalog", "gateway", "worker", "metadata_only", "observer"])
    if len(set(roles)) != len(roles) or len(set(groups)) != len(groups):
        raise ValueError("enrollment roles and groups must be unique")
    material = {}
    for group in groups:
        state = PublisherState()
        kits = {role: state.mint() for role in roles}
        material[group] = kits, state.to_bytes()
    index_key = os.urandom(32)
    ceremony = "cer_primitives_" + uuid4().hex
    configs = {}
    for role in roles:
        directory = root / role
        keys = directory / "keys"
        keys.mkdir(parents=True)
        device = DeviceKey.generate()
        (keys / "local.private").write_bytes(device.private_bytes)
        (keys / "index_master.key").write_bytes(index_key)
        for group, (kits, state) in material.items():
            if role == "metadata_only" and group == "values":
                continue
            (keys / f"{group}.btn.mykit").write_bytes(kits[role])
            if role not in read_only_roles:
                (keys / f"{group}.btn.state").write_bytes(state)
        policy_dir = directory / ".tn" / "config"
        policy_dir.mkdir(parents=True)
        (policy_dir / "agents.md").write_text(POLICY, encoding="utf-8")
        config = {
            "ceremony": {"id": ceremony, "cipher": "btn", "mode": "local"},
            "keystore": {"path": "keys"}, "device": {"device_identity": device.did},
            "logs": {"path": "register.ndjson"},
            "groups": {group: {"cipher": "btn", "policy": "private", "index_epoch": 0}
                       for group in groups if not (role == "metadata_only" and group == "values")},
        }
        path = directory / "tn.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        configs[role] = path
    return configs
