"""Canonical TN source-row minting and tn-llama endpoint provisioning."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import tn
import yaml
from tn.identity import Identity
from tn_polars.tn_envelopes import DeskLedger

from .deepvest import DeepVestExecution
from .polars_stage import GovernedPortfolioResult

_SOURCE_EVENT_TYPE = "portfolio.deepvest.calculated"
_SOURCE_GROUP = "finance"
_GOVERNANCE_GROUP = "tn.agents"
_SOURCE_FIELDS = ("portfolio_values", "deepvest_result", "run_id")
_GOVERNANCE_FIELDS = (
    "instruction",
    "use_for",
    "do_not_use_for",
    "consequences",
    "on_violation_or_error",
    "policy",
)
_OUTPUT_GOVERNANCE_FIELDS = (*_GOVERNANCE_FIELDS, "source_lineage", "compliance_evidence")
_KIT_SUFFIX = ".btn.mykit"


@dataclass(frozen=True, slots=True)
class ModelBoundTNRow:
    """One exact canonical TN row selected from its writer's log."""

    raw_ndjson: bytes
    envelope: Mapping[str, object]
    event_id: str
    row_hash: str
    encrypted_groups: tuple[str, ...]
    policy_reference: str
    source_writer_did: str
    source_ceremony_id: str
    source_log_path: Path


@dataclass(frozen=True, slots=True)
class RuntimeReaderMaterial:
    """Clean recipient capabilities and commitment-verification provenance."""

    reader_key_dir: Path
    index_key_path: Path
    ceremony_id: str
    index_epoch: int
    recipient_did: str
    recipient_ceremony_id: str
    package_path: Path
    absorb_status: str
    absorb_reason: str
    absorb_peer_did: str | None
    absorb_offer_digest: str | None
    absorb_artifact_digest: str | None
    absorb_reader_did: str | None


@dataclass(frozen=True, slots=True)
class OutputWriterMaterial:
    """Separate canonical TN writer and recipient material for report rows."""

    config_path: Path
    log_path: Path
    writer_did: str
    ceremony_id: str
    index_epoch: int
    reader: RuntimeReaderMaterial


@dataclass(frozen=True, slots=True)
class ReferencePathMaterial:
    """Pre-provisioned TN endpoints used by the local reference path."""

    source_config_path: Path
    polars_config_path: Path
    deepvest_config_path: Path
    input_recipient_config_path: Path
    input_reader: RuntimeReaderMaterial
    output_writer: OutputWriterMaterial


@dataclass(frozen=True, slots=True)
class TNFlowResult:
    """Governed input publication consumed by the governed tn-llama client stage."""

    source_row: ModelBoundTNRow
    marker_free_row: ModelBoundTNRow | None
    input_reader: RuntimeReaderMaterial
    output_writer: OutputWriterMaterial


def build_reference_flow(
    material: ReferencePathMaterial,
    source_row: ModelBoundTNRow,
) -> TNFlowResult:
    """Bind a produced model row to the already-provisioned tn-llama endpoints."""
    return TNFlowResult(
        source_row=source_row,
        marker_free_row=None,
        input_reader=material.input_reader,
        output_writer=material.output_writer,
    )


def _create_endpoint(
    config_path: Path,
    identity_path: Path,
    *,
    fields: list[str],
) -> tn.TN:
    endpoint = _open_recipient(config_path, identity_path)
    tn.ensure_group(endpoint.cfg, _SOURCE_GROUP, cipher="btn", fields=fields)
    return _activate(config_path)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _json_copy(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_copy(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_json_copy(item) for item in value)
    return value


def _absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _prepare_output_file(path: Path) -> Path:
    resolved = _absolute(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(f"refusing to replace existing runtime material: {resolved}")
    return resolved


def _prepare_clean_directory(path: Path) -> Path:
    resolved = _absolute(path)
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        if any(resolved.iterdir()):
            raise ValueError(f"runtime reader directory must be empty: {resolved}")
    else:
        resolved.mkdir(parents=True)
    return resolved


def _load_or_create_identity(path: Path) -> Identity:
    identity_path = _absolute(path)
    if identity_path.exists():
        return Identity.load(identity_path)
    identity = Identity.create_new()
    identity.ensure_written(identity_path)
    return identity


def _open_recipient(config_path: Path, identity_path: Path) -> tn.TN:
    resolved_config = _absolute(config_path)
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    identity = _load_or_create_identity(identity_path)
    recipient = tn.init(
        resolved_config,
        profile="transaction",
        identity=identity,
        cipher="btn",
        link=False,
        stdout=False,
    )
    if recipient.cfg.device.device_identity != identity.did:
        raise ValueError("recipient ceremony DID does not match its explicit TN identity")
    return recipient


def _activate(config_path: Path) -> tn.TN:
    return tn.init(_absolute(config_path), link=False, stdout=False)


def _policy_reference(writer: tn.TN) -> str:
    document = tn.agents.policy()
    if document is None:
        raise ValueError("source ceremony has no TN agents-policy document")
    template = document.templates.get(_SOURCE_EVENT_TYPE)
    if template is None:
        raise ValueError(f"TN agents policy does not cover {_SOURCE_EVENT_TYPE!r}")
    return (
        f"{template.path}#{template.event_type}@{template.version}#{template.content_hash}"
    )


def _encrypted_groups(envelope: Mapping[str, object]) -> tuple[str, ...]:
    groups = []
    for name, value in envelope.items():
        if isinstance(value, Mapping) and set(value) == {"ciphertext", "field_hashes"}:
            groups.append(name)
    return tuple(sorted(groups))


def _select_exact_row(
    log_path: Path,
    emitted: Mapping[str, object],
    *,
    policy_reference: str,
    writer_did: str,
    ceremony_id: str,
) -> ModelBoundTNRow:
    event_id = emitted.get("event_id")
    row_hash = emitted.get("row_hash")
    if not isinstance(event_id, str) or not isinstance(row_hash, str):
        raise ValueError("TN.log() returned an envelope without string event_id and row_hash")

    matches: list[tuple[bytes, dict[str, object]]] = []
    for raw_line in log_path.read_bytes().splitlines(keepends=True):
        if not raw_line.strip():
            continue
        if not raw_line.endswith(b"\n"):
            raise ValueError(f"TN log contains a non-newline-terminated row: {log_path}")
        parsed = json.loads(raw_line)
        if parsed.get("event_id") == event_id and parsed.get("row_hash") == row_hash:
            matches.append((raw_line, parsed))
    if len(matches) != 1:
        raise ValueError(
            "TN log must contain exactly one row matching the emitted (event_id, row_hash)"
        )
    raw_ndjson, parsed = matches[0]
    if parsed != dict(emitted):
        raise ValueError("selected TN log row differs from the dictionary returned by TN.log()")

    return ModelBoundTNRow(
        raw_ndjson=bytes(raw_ndjson),
        envelope=_freeze(parsed),
        event_id=event_id,
        row_hash=row_hash,
        encrypted_groups=_encrypted_groups(parsed),
        policy_reference=policy_reference,
        source_writer_did=writer_did,
        source_ceremony_id=ceremony_id,
        source_log_path=_absolute(log_path),
    )


def _require_exact_source_groups(row: ModelBoundTNRow) -> None:
    if row.encrypted_groups != (_SOURCE_GROUP, _GOVERNANCE_GROUP):
        raise ValueError(
            "model-bound TN row must contain exactly finance and tn.agents encrypted groups"
        )


def _portfolio_payload(portfolio: GovernedPortfolioResult) -> dict[str, object]:
    return {
        "rows": portfolio.values.to_dicts(),
        "native_feed_names": list(portfolio.feed_names),
        "native_column_sources": {
            name: sorted(sources) for name, sources in sorted(portfolio.sources.items())
        },
        "native_row_sources": sorted(portfolio.row_sources),
        "admission_policy": portfolio.policy_ref,
        "unity_governance": _json_copy(portfolio.tn_governance),
    }


def _deepvest_payload(execution: DeepVestExecution) -> dict[str, object]:
    return asdict(execution.result)


def _validate_package(package_path: Path, expected_groups: tuple[str, ...]) -> None:
    import zipfile

    expected = {"manifest.json", *(f"body/{group}{_KIT_SUFFIX}" for group in expected_groups)}
    with zipfile.ZipFile(package_path) as package:
        names = package.namelist()
    if len(names) != len(set(names)):
        raise ValueError("TN runtime package contains duplicate entries")
    if set(names) != expected:
        raise ValueError(
            f"TN runtime package entries differ from the required kit bundle: {sorted(names)!r}"
        )


def _validated_v2_index_epoch(
    config: tn.config.LoadedConfig,
    groups: tuple[str, ...],
) -> int:
    try:
        epochs = [config.groups[group].index_epoch for group in groups]
    except KeyError as exc:
        raise ValueError(f"TN ceremony omitted required group {exc.args[0]!r}") from exc
    if not epochs or len(set(epochs)) != 1 or epochs[0] != 0:
        raise ValueError(
            f"TN v2 requires index epoch 0 for all groups {groups!r}; got {epochs!r}"
        )
    return epochs[0]


def validate_runtime_reader_material(
    material: RuntimeReaderMaterial,
    *,
    expected_groups: tuple[str, ...],
) -> None:
    """Fail closed unless a runtime directory contains exactly the named TN kits."""
    if len(expected_groups) != len(set(expected_groups)):
        raise ValueError("expected runtime kit names contain duplicates")
    directory = _absolute(material.reader_key_dir)
    if directory != material.reader_key_dir or not directory.is_dir():
        raise ValueError("runtime reader key directory must be an existing absolute directory")
    entries = list(directory.iterdir())
    if any(not entry.is_file() for entry in entries):
        raise ValueError("runtime reader directory may contain only regular TN kit files")
    names = [entry.name for entry in entries]
    expected = [f"{group}{_KIT_SUFFIX}" for group in expected_groups]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(
            f"runtime reader directory must contain exactly {sorted(expected)!r}"
        )
    if not material.index_key_path.is_absolute() or len(material.index_key_path.read_bytes()) != 32:
        raise ValueError("runtime index-key path must contain exactly 32 bytes")


def _absorb_and_materialize(
    *,
    publisher_config_path: Path,
    recipient: tn.TN,
    groups: tuple[str, ...],
    package_path: Path,
    reader_dir: Path,
    index_key_path: Path,
    expected_index_epoch: int,
    label: str,
) -> RuntimeReaderMaterial:
    recipient = _activate(recipient.yaml_path)
    recipient_cfg = recipient.cfg
    recipient_did = recipient_cfg.device.device_identity
    recipient_ceremony_id = recipient_cfg.ceremony_id
    recipient_keystore = _absolute(recipient_cfg.keystore)

    writer = _activate(publisher_config_path)
    writer_cfg = writer.cfg
    writer_ceremony_id = writer_cfg.ceremony_id
    if (
        _absolute(recipient.yaml_path) == _absolute(publisher_config_path)
        or recipient_ceremony_id == writer_ceremony_id
        or recipient_did == writer_cfg.device.device_identity
    ):
        raise ValueError(
            "TN publisher and recipient must use distinct configs, ceremonies, and DIDs"
        )
    expected_groups = tuple(dict.fromkeys((*groups, _GOVERNANCE_GROUP)))
    writer_index_epoch = _validated_v2_index_epoch(writer_cfg, expected_groups)
    if writer_index_epoch != expected_index_epoch:
        raise ValueError("TN publisher index epoch changed after profile validation")
    writer_master_path = writer_cfg.keystore / "index_master.key"
    writer_master = writer_master_path.read_bytes()
    if len(writer_master) != 32:
        raise ValueError("TN writer index master must contain exactly 32 bytes")

    package = tn.admin.add_agent_runtime(
        recipient_did,
        groups=list(groups),
        out_path=package_path,
        label=label,
    )
    package = _absolute(package)
    _validate_package(package, expected_groups)

    receipt = recipient.absorb(package)
    if receipt.status != "enrolment_applied":
        raise ValueError(
            f"TN recipient did not canonically absorb runtime package: {receipt.status!r}"
        )
    clean_reader_dir = _prepare_clean_directory(reader_dir)
    for group in expected_groups:
        source_kit = recipient_keystore / f"{group}{_KIT_SUFFIX}"
        if not source_kit.is_file():
            raise FileNotFoundError(f"canonical TN absorption omitted {source_kit.name}")
        shutil.copyfile(source_kit, clean_reader_dir / source_kit.name)

    copied_index = _prepare_output_file(index_key_path)
    shutil.copyfile(writer_master_path, copied_index)
    material = RuntimeReaderMaterial(
        reader_key_dir=clean_reader_dir,
        index_key_path=copied_index,
        ceremony_id=writer_ceremony_id,
        index_epoch=writer_index_epoch,
        recipient_did=recipient_did,
        recipient_ceremony_id=recipient_ceremony_id,
        package_path=package,
        absorb_status=receipt.status,
        absorb_reason=receipt.reason,
        absorb_peer_did=receipt.peer_did,
        absorb_offer_digest=receipt.offer_digest,
        absorb_artifact_digest=receipt.artifact_digest,
        absorb_reader_did=receipt.reader_did,
    )
    validate_runtime_reader_material(material, expected_groups=expected_groups)
    return material


def _install_agents_policy(config_path: Path, policy_text: str) -> None:
    policy_path = config_path.parent / ".tn" / "config" / "agents.md"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(policy_text, encoding="utf-8")


def _set_literal_empty_public_fields(config_path: Path) -> None:
    before_text = config_path.read_text(encoding="utf-8")
    before = yaml.safe_load(before_text)
    if not isinstance(before, dict) or not isinstance(before.get("public_fields"), list):
        raise ValueError("TN output config has no public_fields list to normalize")

    lines = before_text.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.rstrip("\r\n") == "public_fields:"),
        None,
    )
    if start is None:
        raise ValueError("TN output config has no top-level public_fields block")
    end = start + 1
    while end < len(lines):
        stripped = lines[end].rstrip("\r\n")
        if stripped and not stripped.startswith((" ", "-")):
            break
        end += 1
    after_text = "".join((*lines[:start], "public_fields: []\n", *lines[end:]))
    after = yaml.safe_load(after_text)
    if not isinstance(after, dict) or after.get("public_fields") != []:
        raise ValueError("TN output public_fields normalization failed")
    before_without = dict(before)
    after_without = dict(after)
    before_without.pop("public_fields")
    after_without.pop("public_fields")
    if before_without != after_without:
        raise ValueError("TN output normalization attempted to alter another config value")
    config_path.write_text(after_text, encoding="utf-8")


def _validate_source_writer_profile(config_path: Path) -> int:
    resolved = _absolute(config_path)
    document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    loaded = tn.config.load(resolved)
    if not isinstance(document, dict):
        raise ValueError("source writer profile must be a TN YAML mapping")
    if not loaded.sign or not loaded.chain:
        raise ValueError("source writer profile must enable signing and chaining")

    expected_groups = {"default", _SOURCE_GROUP, _GOVERNANCE_GROUP}
    group_documents = document.get("groups")
    if not isinstance(group_documents, dict) or set(loaded.groups) != expected_groups:
        raise ValueError(
            "source writer profile must contain exactly default, finance, and tn.agents"
        )
    for group_name in (_SOURCE_GROUP, _GOVERNANCE_GROUP):
        group_document = group_documents.get(group_name)
        group = loaded.groups[group_name]
        if not isinstance(group_document, dict) or group_document.get("policy") != "private":
            raise ValueError("source writer profile protected groups must be private")
        if group.cipher.name != "btn" or group.aad_default:
            raise ValueError(
                "source writer profile protected groups must use BTN without default AAD"
            )

    index_epoch = _validated_v2_index_epoch(
        loaded,
        (_SOURCE_GROUP, _GOVERNANCE_GROUP),
    )
    expected_routes = {
        **{field: [_SOURCE_GROUP] for field in _SOURCE_FIELDS},
        **{field: [_GOVERNANCE_GROUP] for field in _GOVERNANCE_FIELDS},
    }
    if loaded.field_to_groups != expected_routes:
        raise ValueError("source writer profile field routing is not exact")
    return index_epoch


def _validate_output_writer_profile(config_path: Path) -> int:
    resolved = _absolute(config_path)
    document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    loaded = tn.config.load(resolved)
    if not isinstance(document, dict):
        raise ValueError("output writer profile must be a TN YAML mapping")
    if not loaded.sign or not loaded.chain:
        raise ValueError("output writer profile must enable signing and chaining")
    if document.get("public_fields") != []:
        raise ValueError("output writer profile must contain literal public_fields: []")

    expected_groups = {"default", _GOVERNANCE_GROUP}
    group_documents = document.get("groups")
    if not isinstance(group_documents, dict) or set(loaded.groups) != expected_groups:
        raise ValueError("output writer profile must contain exactly default and tn.agents")
    for group_name in expected_groups:
        group_document = group_documents.get(group_name)
        group = loaded.groups[group_name]
        if not isinstance(group_document, dict) or group_document.get("policy") != "private":
            raise ValueError("output writer profile groups must be private")
        if (
            group.cipher.name != "btn"
            or group.aad_default
        ):
            raise ValueError(
                "output writer profile groups must use BTN at epoch 0 without default AAD"
            )
    index_epoch = _validated_v2_index_epoch(
        loaded,
        ("default", _GOVERNANCE_GROUP),
    )

    expected_routes = {
        "default": ["default"],
        **{field: [_GOVERNANCE_GROUP] for field in _OUTPUT_GOVERNANCE_FIELDS},
    }
    if loaded.field_to_groups != expected_routes:
        raise ValueError("output writer profile field routing is not exact")
    return index_epoch


def _prepare_output_writer(
    *,
    config_path: Path,
    identity_path: Path,
    policy_text: str,
) -> tuple[tn.TN, int]:
    resolved_config = _absolute(config_path)
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    _install_agents_policy(resolved_config, policy_text)
    identity = _load_or_create_identity(identity_path)
    writer = tn.init(
        resolved_config,
        profile="transaction",
        identity=identity,
        cipher="btn",
        link=False,
        stdout=False,
    )
    if writer.cfg.device.device_identity != identity.did:
        raise ValueError("output writer ceremony DID does not match its explicit TN identity")
    tn.ensure_group(writer.cfg, "default", cipher="btn", fields=["default"])
    tn.ensure_group(
        writer.cfg,
        _GOVERNANCE_GROUP,
        cipher="btn",
        fields=list(_OUTPUT_GOVERNANCE_FIELDS),
    )
    _set_literal_empty_public_fields(resolved_config)
    index_epoch = _validate_output_writer_profile(resolved_config)
    return _activate(resolved_config), index_epoch


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _absolute(left)
    right = _absolute(right)
    return left == right or left in right.parents or right in left.parents


def _require_separate_state(
    source_reader: RuntimeReaderMaterial,
    output_writer: OutputWriterMaterial,
    output_keystore: Path,
) -> None:
    output_paths = (
        output_keystore,
        output_writer.log_path,
        output_writer.reader.reader_key_dir,
        output_writer.reader.index_key_path,
    )
    for source_path in (source_reader.reader_key_dir, source_reader.index_key_path):
        if any(_paths_overlap(source_path, output_path) for output_path in output_paths):
            raise ValueError("source reader/index material overlaps output TN state")


def provision_reference_path(
    root: Path,
    *,
    source_agents_path: Path,
    output_agents_path: Path,
) -> ReferencePathMaterial:
    """Provision the source, Polars, DeepVest, tn-llama, and report recipients."""
    state = _absolute(root)
    if state.exists():
        raise FileExistsError(f"reference-path TN state already exists: {state}")
    state.mkdir(parents=True)

    source = _create_endpoint(
        state / "source" / "tn.yaml",
        state / "source-identity" / "identity.json",
        fields=["feed_name", "rows"],
    )
    polars = _create_endpoint(
        state / "polars" / "tn.yaml",
        state / "polars-identity" / "identity.json",
        fields=[
            "portfolio_values",
            "native_feed_names",
            "native_column_sources",
            "native_row_sources",
        ],
    )
    deepvest_config_path = state / "deepvest" / "tn.yaml"
    _install_agents_policy(
        deepvest_config_path,
        _absolute(source_agents_path).read_text(encoding="utf-8"),
    )
    deepvest = _create_endpoint(
        deepvest_config_path,
        state / "deepvest-identity" / "identity.json",
        fields=list(_SOURCE_FIELDS),
    )

    def enrol(
        publisher: tn.TN,
        recipient: tn.TN,
        name: str,
    ) -> RuntimeReaderMaterial:
        writer = _activate(publisher.yaml_path)
        index_epoch = _validated_v2_index_epoch(
            writer.cfg,
            (_SOURCE_GROUP, _GOVERNANCE_GROUP),
        )
        return _absorb_and_materialize(
            publisher_config_path=writer.yaml_path,
            recipient=recipient,
            groups=(_SOURCE_GROUP,),
            package_path=_prepare_output_file(state / "packages" / f"{name}.tnpkg"),
            reader_dir=state / "readers" / name,
            index_key_path=state / "indexes" / f"{name}.key",
            expected_index_epoch=index_epoch,
            label=name,
        )

    enrol(source, polars, "source-to-polars")
    enrol(polars, deepvest, "polars-to-deepvest")

    input_recipient_config_path = state / "llama-input" / "tn.yaml"
    input_recipient = _open_recipient(
        input_recipient_config_path,
        state / "llama-input-identity" / "identity.json",
    )
    input_reader = enrol(deepvest, input_recipient, "deepvest-to-llama")

    output_writer, output_index_epoch = _prepare_output_writer(
        config_path=state / "llama-output" / "tn.yaml",
        identity_path=state / "llama-output-identity" / "identity.json",
        policy_text=_absolute(output_agents_path).read_text(encoding="utf-8"),
    )
    output_config_path = _absolute(output_writer.yaml_path)
    output_cfg = output_writer.cfg
    output_log_path = _absolute(output_cfg.resolve_log_path())
    output_writer_did = output_cfg.device.device_identity
    output_ceremony_id = output_cfg.ceremony_id
    output_recipient = _open_recipient(
        state / "report-recipient" / "tn.yaml",
        state / "report-recipient-identity" / "identity.json",
    )
    output_reader = _absorb_and_materialize(
        publisher_config_path=output_config_path,
        recipient=output_recipient,
        groups=("default",),
        package_path=_prepare_output_file(state / "packages" / "llama-to-report.tnpkg"),
        reader_dir=state / "readers" / "llama-to-report",
        index_key_path=state / "indexes" / "llama-to-report.key",
        expected_index_epoch=output_index_epoch,
        label="llama-to-report",
    )
    output = OutputWriterMaterial(
        config_path=output_config_path,
        log_path=output_log_path,
        writer_did=output_writer_did,
        ceremony_id=output_ceremony_id,
        index_epoch=output_reader.index_epoch,
        reader=output_reader,
    )
    return ReferencePathMaterial(
        source_config_path=_absolute(source.yaml_path),
        polars_config_path=_absolute(polars.yaml_path),
        deepvest_config_path=_absolute(deepvest.yaml_path),
        input_recipient_config_path=_absolute(input_recipient_config_path),
        input_reader=input_reader,
        output_writer=output,
    )


def publish_deepvest_object_for_llama(
    source: tn.SealedObject | Mapping[str, object] | str | bytes | Path,
    *,
    publisher_config_path: Path,
) -> ModelBoundTNRow:
    """Publish a portable DeepVest TN object as tn-llama's chained TN input row."""
    from .tn_objects import open_governed_object

    opened = open_governed_object(
        publisher_config_path,
        source,
        expected_type=_SOURCE_EVENT_TYPE,
        business_group=_SOURCE_GROUP,
    )
    required = {"portfolio_values", "deepvest_result"}
    if not required.issubset(opened.business):
        raise ValueError("DeepVest TN object omitted portfolio_values or deepvest_result")
    policy_reference = opened.governance.get("policy")
    if not isinstance(policy_reference, str) or "#sha256:" not in policy_reference:
        raise ValueError("DeepVest TN object policy is not content addressed")

    return publish_deepvest_values_for_llama(
        portfolio_values=opened.business["portfolio_values"],
        deepvest_result=opened.business["deepvest_result"],
        governance=opened.governance,
        publisher_config_path=publisher_config_path,
    )


def publish_deepvest_values_for_llama(
    *,
    portfolio_values: object,
    deepvest_result: object,
    governance: Mapping[str, object],
    publisher_config_path: Path,
) -> ModelBoundTNRow:
    """Log DeepVest's in-process result using tn-llama's canonical TN row contract."""
    policy_reference = governance.get("policy")
    if not isinstance(policy_reference, str) or "#sha256:" not in policy_reference:
        raise ValueError("DeepVest governance policy is not content addressed")

    writer = _activate(publisher_config_path)
    tn.ensure_group(
        writer.cfg,
        _SOURCE_GROUP,
        cipher="btn",
        fields=list(_SOURCE_FIELDS),
    )
    writer = _activate(publisher_config_path)
    cfg = writer.cfg
    marker = {
        "governed_by": cfg.device.device_identity,
        "policy": policy_reference,
    }
    emitted = writer.log(
        _SOURCE_EVENT_TYPE,
        level="info",
        aad=marker,
        portfolio_values=portfolio_values,
        deepvest_result=deepvest_result,
        **dict(governance),
    )
    if emitted is None:
        raise RuntimeError("TN filtered the DeepVest model-bound row")
    row = _select_exact_row(
        _absolute(cfg.resolve_log_path()),
        emitted,
        policy_reference=policy_reference,
        writer_did=cfg.device.device_identity,
        ceremony_id=cfg.ceremony_id,
    )
    _require_exact_source_groups(row)
    return row


def provision_tn_flow(
    portfolio: GovernedPortfolioResult,
    execution: DeepVestExecution,
    *,
    source_project_dir: Path,
    source_identity_dir: Path,
    source_agents_path: Path,
    input_recipient_config_path: Path,
    input_recipient_identity_path: Path,
    input_package_path: Path,
    input_reader_dir: Path,
    input_index_key_path: Path,
    output_writer_config_path: Path,
    output_writer_identity_path: Path,
    output_agents_path: Path,
    output_recipient_config_path: Path,
    output_recipient_identity_path: Path,
    output_package_path: Path,
    output_reader_dir: Path,
    output_index_key_path: Path,
) -> TNFlowResult:
    """Mint the governed source rows and provision separate tn-llama TN state."""
    source_agents_text = _absolute(source_agents_path).read_text(encoding="utf-8")
    output_agents_text = _absolute(output_agents_path).read_text(encoding="utf-8")

    source_ledger = DeskLedger.open(
        _absolute(source_project_dir),
        ceremony="desk",
        agents_md=source_agents_text,
        identity_dir=_absolute(source_identity_dir),
    )
    source_config_path = _absolute(source_ledger.ceremony_dir / "tn.yaml")
    source_writer = _activate(source_config_path)
    tn.ensure_group(
        source_writer.cfg,
        _SOURCE_GROUP,
        cipher="btn",
        fields=list(_SOURCE_FIELDS),
    )
    source_index_epoch = _validate_source_writer_profile(source_config_path)

    input_package = _prepare_output_file(input_package_path)
    input_recipient = _open_recipient(
        input_recipient_config_path,
        input_recipient_identity_path,
    )
    input_material = _absorb_and_materialize(
        publisher_config_path=source_config_path,
        recipient=input_recipient,
        groups=(_SOURCE_GROUP,),
        package_path=input_package,
        reader_dir=input_reader_dir,
        index_key_path=input_index_key_path,
        expected_index_epoch=source_index_epoch,
        label="deepvest-input-runtime",
    )

    source_writer = _activate(source_config_path)
    source_cfg = source_writer.cfg
    policy_reference = _policy_reference(source_writer)
    marker = {
        "governed_by": source_cfg.device.device_identity,
        "policy": policy_reference,
    }
    fields = {
        "portfolio_values": _portfolio_payload(portfolio),
        "deepvest_result": _deepvest_payload(execution),
    }
    emitted = source_writer.log(
        _SOURCE_EVENT_TYPE,
        level="info",
        aad=marker,
        **fields,
    )
    if emitted is None:
        raise RuntimeError("TN filtered the model-bound source row")
    source_log_path = _absolute(source_cfg.resolve_log_path())
    source_row = _select_exact_row(
        source_log_path,
        emitted,
        policy_reference=policy_reference,
        writer_did=source_cfg.device.device_identity,
        ceremony_id=source_cfg.ceremony_id,
    )
    _require_exact_source_groups(source_row)
    marker_free_emitted = source_writer.log(
        _SOURCE_EVENT_TYPE,
        level="info",
        **fields,
    )
    if marker_free_emitted is None:
        raise RuntimeError("TN filtered the marker-free source row")
    marker_free_row = _select_exact_row(
        source_log_path,
        marker_free_emitted,
        policy_reference=policy_reference,
        writer_did=source_cfg.device.device_identity,
        ceremony_id=source_cfg.ceremony_id,
    )
    _require_exact_source_groups(marker_free_row)

    output_writer, output_index_epoch = _prepare_output_writer(
        config_path=output_writer_config_path,
        identity_path=output_writer_identity_path,
        policy_text=output_agents_text,
    )
    output_cfg = output_writer.cfg
    output_writer_did = output_cfg.device.device_identity
    output_ceremony_id = output_cfg.ceremony_id
    output_log_path = _absolute(output_cfg.resolve_log_path())
    output_keystore = _absolute(output_cfg.keystore)
    output_master = (output_keystore / "index_master.key").read_bytes()

    output_package = _prepare_output_file(output_package_path)
    output_recipient = _open_recipient(
        output_recipient_config_path,
        output_recipient_identity_path,
    )
    output_reader = _absorb_and_materialize(
        publisher_config_path=_absolute(output_writer_config_path),
        recipient=output_recipient,
        groups=("default",),
        package_path=output_package,
        reader_dir=output_reader_dir,
        index_key_path=output_index_key_path,
        expected_index_epoch=output_index_epoch,
        label="deepvest-output-runtime",
    )
    output_material = OutputWriterMaterial(
        config_path=_absolute(output_writer_config_path),
        log_path=output_log_path,
        writer_did=output_writer_did,
        ceremony_id=output_ceremony_id,
        index_epoch=output_reader.index_epoch,
        reader=output_reader,
    )

    if source_row.source_ceremony_id == output_material.ceremony_id:
        raise ValueError("source and output TN ceremonies must be distinct")
    if input_material.index_key_path.read_bytes() == output_master:
        raise ValueError("source and output TN index masters must be distinct")
    _require_separate_state(input_material, output_material, output_keystore)

    return TNFlowResult(
        source_row=source_row,
        marker_free_row=marker_free_row,
        input_reader=input_material,
        output_writer=output_material,
    )
