from __future__ import annotations

import base64
import io
import json
import shutil
import socket
import subprocess
import threading
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import tn
import yaml
from tn.identity import Identity
from tn_polars.tn_envelopes import DeskLedger

from deepvest_tn.deepvest import DeepVestExecution, StubDeepVest, execute_deepvest
from deepvest_tn.feed_server import FeedServerConfig, create_feed_server, seed_finance_feed_data
from deepvest_tn.polars_stage import GovernedPortfolioResult, run_governed_polars_stage
from deepvest_tn.tn_flow import (
    ModelBoundTNRow,
    OutputWriterMaterial,
    RuntimeReaderMaterial,
    TNFlowResult,
    _validate_output_writer_profile,
    _validate_package,
    _validate_source_writer_profile,
    provision_tn_flow,
    validate_runtime_reader_material,
)
from deepvest_tn.unity import UnityCatalogConfig, build_finance_feed_bundles

FIXTURE_ROOT = Path("fixtures/portfolio")
POSITIONS_PATH = FIXTURE_ROOT / "internal_positions.ndjson"
MARKET_PRICES_PATH = FIXTURE_ROOT / "market_prices.ndjson"
SOURCE_AGENTS_PATH = FIXTURE_ROOT / "agents.policy"
OUTPUT_AGENTS_PATH = FIXTURE_ROOT / "report-agents.policy"


@contextmanager
def _running_feed_server(server) -> Iterator[str]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture()
def feed_url(tmp_path: Path) -> Iterator[str]:
    data_dir = seed_finance_feed_data(tmp_path / "getdata")
    server = create_feed_server(
        FeedServerConfig(data_dir=data_dir, event_stream=io.StringIO(), port=0)
    )
    with _running_feed_server(server) as base_url:
        yield base_url


class RecordingUnityTransport:
    def __init__(self, bundles: dict[str, Path], config: UnityCatalogConfig) -> None:
        self.volumes = {
            f"{config.catalog}.{config.schema}.{name.replace('.', '_')}": bundle.resolve().as_uri()
            for name, bundle in bundles.items()
        }

    def request(self, method: str, path: str, payload=None):
        if method == "GET" and path.startswith("/volumes/"):
            return {"storage_location": self.volumes[path.removeprefix("/volumes/")]}
        raise AssertionError(f"unexpected Unity request: {method} {path} {payload}")


@pytest.fixture(autouse=True)
def isolated_tn(monkeypatch):
    monkeypatch.setenv("TN_NO_LINK", "1")
    monkeypatch.setenv("TN_NO_STDOUT", "1")
    monkeypatch.setenv("TN_AUTOINIT_QUIET", "1")
    with suppress(Exception):
        tn.flush_and_close()
    yield
    with suppress(Exception):
        tn.flush_and_close()


@pytest.fixture(scope="module")
def finance_bundles(tmp_path_factory) -> dict[str, Path]:
    return build_finance_feed_bundles(tmp_path_factory.mktemp("task5-finance-bundles"))


@pytest.fixture()
def governed_portfolio(monkeypatch, finance_bundles, feed_url: str) -> GovernedPortfolioResult:
    config = UnityCatalogConfig()
    transport = RecordingUnityTransport(finance_bundles, config)

    def request(_registry, method: str, path: str, payload=None):
        return transport.request(method, path, payload)

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", request)
    return run_governed_polars_stage(
        config,
        account_id="acct-001",
        feed_url=feed_url,
    )


@pytest.fixture()
def tn_flow(tmp_path: Path, governed_portfolio: GovernedPortfolioResult) -> TNFlowResult:
    return _provision(tmp_path, governed_portfolio)


def _provision(
    root: Path,
    governed_portfolio: GovernedPortfolioResult,
    execution: DeepVestExecution | None = None,
    *,
    alias_output_recipient: bool = False,
) -> TNFlowResult:
    if execution is None:
        execution = execute_deepvest(StubDeepVest(), governed_portfolio)
    return provision_tn_flow(
        governed_portfolio,
        execution,
        source_project_dir=root / "source",
        source_identity_dir=root / "source-identity",
        source_agents_path=SOURCE_AGENTS_PATH,
        input_recipient_config_path=root / "input-recipient" / "tn.yaml",
        input_recipient_identity_path=root / "input-recipient-identity" / "identity.json",
        input_package_path=root / "packages" / "input-runtime.tnpkg",
        input_reader_dir=root / "input-reader",
        input_index_key_path=root / "input.index.key",
        output_writer_config_path=root / "output-writer" / "tn.yaml",
        output_writer_identity_path=root / "output-writer-identity" / "identity.json",
        output_agents_path=OUTPUT_AGENTS_PATH,
        output_recipient_config_path=(
            root / "output-writer" / "tn.yaml"
            if alias_output_recipient
            else root / "output-recipient" / "tn.yaml"
        ),
        output_recipient_identity_path=(
            root / "output-writer-identity" / "identity.json"
            if alias_output_recipient
            else root / "output-recipient-identity" / "identity.json"
        ),
        output_package_path=root / "packages" / "output-runtime.tnpkg",
        output_reader_dir=root / "output-reader",
        output_index_key_path=root / "output.index.key",
    )


def _opened_target(row: ModelBoundTNRow, material: RuntimeReaderMaterial, group: str) -> dict:
    opened = list(
        tn.reader.read_as_recipient(
            row.source_log_path,
            material.reader_key_dir,
            group=group,
        )
    )
    matches = [entry for entry in opened if entry["envelope"].get("event_id") == row.event_id]
    assert len(matches) == 1
    return matches[0]


def _precreate_output_writer_config(
    root: Path,
    *,
    ceremony_override: tuple[str, object],
) -> None:
    config_path = (root / "output-writer" / "tn.yaml").resolve()
    identity_path = (root / "output-writer-identity" / "identity.json").resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity = Identity.create_new()
    identity.ensure_written(identity_path)
    tn.init(
        config_path,
        profile="transaction",
        identity=identity,
        cipher="btn",
        link=False,
        stdout=False,
    )
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    field, value = ceremony_override
    document["ceremony"][field] = value
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def _precreate_source_writer_config(
    root: Path,
    *,
    ceremony_override: tuple[str, object],
) -> Path:
    ledger = DeskLedger.open(
        (root / "source").resolve(),
        ceremony="desk",
        agents_md=SOURCE_AGENTS_PATH.read_text(encoding="utf-8"),
        identity_dir=(root / "source-identity").resolve(),
    )
    config_path = (ledger.ceremony_dir / "tn.yaml").resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    field, value = ceremony_override
    document["ceremony"][field] = value
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_canonical_source_row_round_trips_exact_values_and_governance(
    tn_flow: TNFlowResult,
) -> None:
    row = tn_flow.source_row

    assert isinstance(row, ModelBoundTNRow)
    assert row.raw_ndjson.endswith(b"\n")
    assert json.loads(row.raw_ndjson) == dict(row.envelope)
    assert row.event_id == row.envelope["event_id"]
    assert row.row_hash == row.envelope["row_hash"]
    assert row.envelope["event_type"] == "portfolio.deepvest.calculated"
    assert row.envelope["level"] == "info"
    assert row.encrypted_groups == ("finance", "tn.agents")
    assert b"acct-001" not in row.raw_ndjson
    assert b"stub-portfolio-risk-001" not in row.raw_ndjson
    assert b"Trade execution" not in row.raw_ndjson
    assert b"Perform governed portfolio analysis" not in row.raw_ndjson

    matching_raw_rows = []
    for raw_line in row.source_log_path.read_bytes().splitlines(keepends=True):
        parsed = json.loads(raw_line)
        if (
            parsed.get("event_id") == row.event_id
            and parsed.get("row_hash") == row.row_hash
        ):
            matching_raw_rows.append(raw_line)
    assert matching_raw_rows == [row.raw_ndjson]

    aad = json.loads(row.envelope["tn_aad"])
    marker = {"governed_by": row.source_writer_did, "policy": row.policy_reference}
    assert aad == {"finance": marker, "tn.agents": marker}

    finance = _opened_target(row, tn_flow.input_reader, "finance")
    governance = _opened_target(row, tn_flow.input_reader, "tn.agents")
    assert all(finance["valid"].values())
    assert all(governance["valid"].values())
    assert finance["envelope"] == dict(row.envelope)
    assert finance["plaintext"]["finance"]["portfolio_values"] == {
        "rows": [
            {
                "account_id": "acct-001",
                "portfolio_id": "portfolio-001",
                "symbol": "AAPL",
                "quantity": 125,
                "cost_basis": 182.4,
                "currency": "USD",
                "as_of": "2026-09-01T16:00:00Z",
                "last_price": 228.15,
                "market_source": "deterministic-market-fixture",
                "position_market_value": 28518.75,
            },
            {
                "account_id": "acct-001",
                "portfolio_id": "portfolio-001",
                "symbol": "MSFT",
                "quantity": 80,
                "cost_basis": 412.15,
                "currency": "USD",
                "as_of": "2026-09-01T16:00:00Z",
                "last_price": 509.9,
                "market_source": "deterministic-market-fixture",
                "position_market_value": 40792.0,
            },
            {
                "account_id": "acct-001",
                "portfolio_id": "portfolio-001",
                "symbol": "VTI",
                "quantity": 210,
                "cost_basis": 265.75,
                "currency": "USD",
                "as_of": "2026-09-01T16:00:00Z",
                "last_price": 311.42,
                "market_source": "deterministic-market-fixture",
                "position_market_value": 65398.2,
            },
        ],
        "native_feed_names": [
            "deepvest.internal.positions",
            "deepvest.market.prices",
        ],
        "native_column_sources": {
            "account_id": ["account_id"],
            "as_of": ["as_of"],
            "cost_basis": ["cost_basis"],
            "currency": ["currency"],
            "last_price": ["last_price"],
            "market_source": ["market_source"],
            "portfolio_id": ["portfolio_id"],
            "position_market_value": ["last_price", "quantity"],
            "quantity": ["quantity"],
            "symbol": ["symbol"],
        },
        "native_row_sources": ["account_id", "currency", "symbol"],
        "admission_policy": row.policy_reference,
        "unity_governance": {
            "instruction": (
                "Perform governed portfolio analysis and generate the approved internal "
                "portfolio report."
            ),
            "use_for": "Portfolio analysis and reporting for the authorized internal workflow.",
            "do_not_use_for": (
                "Trade execution or external redistribution of account-level data."
            ),
            "consequences": (
                "Unauthorized use can expose confidential account information and violate "
                "portfolio-data governance."
            ),
            "on_violation_or_error": "log_and_continue",
            "policy": row.policy_reference,
        },
    }
    assert finance["plaintext"]["finance"]["deepvest_result"] == {
        "calculation_id": "stub-portfolio-risk-001",
        "analysis_type": "portfolio_risk",
        "metrics": [
            {"name": "value_at_risk_95", "value": 12500.0, "unit": "USD"},
            {"name": "expected_shortfall_95", "value": 16800.0, "unit": "USD"},
            {"name": "maximum_drawdown", "value": -0.18, "unit": "ratio"},
        ],
        "provider": "stub",
        "live": False,
    }
    assert governance["plaintext"]["tn.agents"] == {
        "instruction": (
            "Use the governed portfolio inputs only to create the approved portfolio "
            "analysis and report row."
        ),
        "use_for": (
            "Internal portfolio analysis and reporting for the authorized DeepVest workflow."
        ),
        "do_not_use_for": (
            "Trade execution, investment order placement, or external redistribution of "
            "account-level data."
        ),
        "consequences": (
            "Unauthorized use can expose confidential account information and violate "
            "portfolio-data governance."
        ),
        "on_violation_or_error": "log_and_continue",
        "policy": row.policy_reference,
    }

    source_config = row.source_log_path.parent.parent / "tn.yaml"
    _source_writer = tn.init(source_config, link=False, stdout=False)
    source_policy = tn.agents.policy()
    assert source_policy is not None
    template = source_policy.templates["portfolio.deepvest.calculated"]
    parsed_reference = (
        f"{template.path}#{template.event_type}@{template.version}#"
        f"{template.content_hash}"
    )
    assert row.policy_reference == parsed_reference
    assert _source_writer.cfg.device.device_identity == row.source_writer_did


def test_runtime_packages_are_canonically_absorbed_into_clean_reader_material(
    tn_flow: TNFlowResult,
) -> None:
    material = tn_flow.input_reader

    assert isinstance(material, RuntimeReaderMaterial)
    assert material.absorb_status == "enrolment_applied"
    assert material.reader_key_dir.is_absolute()
    assert material.index_key_path.is_absolute()
    assert material.package_path.is_absolute()
    assert {path.name for path in material.reader_key_dir.iterdir()} == {
        "finance.btn.mykit",
        "tn.agents.btn.mykit",
    }
    with zipfile.ZipFile(material.package_path) as package:
        names = set(package.namelist())
        manifest = json.loads(package.read("manifest.json"))
    assert names == {
        "manifest.json",
        "body/finance.btn.mykit",
        "body/tn.agents.btn.mykit",
    }
    assert not any(name.endswith("local.private") for name in names)
    assert not any(name.endswith("index_master.key") for name in names)
    assert manifest["recipient_identity"] == material.recipient_did
    assert manifest["publisher_identity"] == tn_flow.source_row.source_writer_did
    source_master = (
        tn_flow.source_row.source_log_path.parent.parent / "keys" / "index_master.key"
    ).read_bytes()
    assert material.index_key_path.read_bytes() == source_master
    assert len(source_master) == 32
    assert not hasattr(material, "source_index_master")


def test_output_writer_and_recipient_material_are_distinct_and_exact(
    tn_flow: TNFlowResult,
) -> None:
    output = tn_flow.output_writer

    assert isinstance(output, OutputWriterMaterial)
    assert output.config_path.is_absolute()
    assert output.log_path.is_absolute()
    assert output.ceremony_id != tn_flow.source_row.source_ceremony_id
    assert output.reader.ceremony_id == output.ceremony_id
    assert (
        output.reader.index_key_path.read_bytes()
        != tn_flow.input_reader.index_key_path.read_bytes()
    )
    assert {path.name for path in output.reader.reader_key_dir.iterdir()} == {
        "default.btn.mykit",
        "tn.agents.btn.mykit",
    }
    config_text = output.config_path.read_text(encoding="utf-8")
    assert "public_fields: []" in config_text
    assert output.index_epoch == 0

    document = yaml.safe_load(config_text)
    loaded = tn.config.load(output.config_path)
    assert loaded.sign is True
    assert loaded.chain is True
    assert document["public_fields"] == []
    assert set(loaded.groups) == {"default", "tn.agents"}
    assert loaded.groups["default"].cipher.name == "btn"
    assert loaded.groups["default"].index_epoch == 0
    assert loaded.groups["default"].aad_default == {}
    assert loaded.groups["tn.agents"].cipher.name == "btn"
    assert loaded.groups["tn.agents"].index_epoch == 0
    assert loaded.groups["tn.agents"].aad_default == {}
    assert loaded.field_to_groups == {
        "default": ["default"],
        "instruction": ["tn.agents"],
        "use_for": ["tn.agents"],
        "do_not_use_for": ["tn.agents"],
        "consequences": ["tn.agents"],
        "on_violation_or_error": ["tn.agents"],
        "policy": ["tn.agents"],
        "source_lineage": ["tn.agents"],
        "compliance_evidence": ["tn.agents"],
    }

    assert output.reader.package_path.is_absolute()
    with zipfile.ZipFile(output.reader.package_path) as package:
        package_names = set(package.namelist())
        manifest = json.loads(package.read("manifest.json"))
    assert package_names == {
        "manifest.json",
        "body/default.btn.mykit",
        "body/tn.agents.btn.mykit",
    }
    assert not any(name.endswith(("local.private", "index_master.key")) for name in package_names)
    assert manifest["recipient_identity"] == output.reader.recipient_did
    assert manifest["publisher_identity"] == output.writer_did
    assert output.reader.absorb_status == "enrolment_applied"
    assert output.writer_did != output.reader.recipient_did
    assert output.reader.recipient_did != tn_flow.input_reader.recipient_did

    _activate_output = tn.init(output.config_path, link=False, stdout=False)
    policy = tn.agents.policy()
    assert policy is not None
    assert set(policy.templates) == {"report.generated"}
    template = policy.templates["report.generated"]
    assert template.instruction == "Generate only the configured derived report."
    assert template.use_for == "Creating the governed HTML report requested by the user."
    assert template.do_not_use_for == "Any ungoverned output or unrelated purpose."
    assert template.consequences == (
        "Reject the report and release no output when validation fails."
    )
    assert template.on_violation_or_error == (
        "Abort the governed session and record the failing gate."
    )
    assert _activate_output.cfg.device.device_identity == output.writer_did


@pytest.mark.parametrize("flag", ["sign", "chain"])
def test_preexisting_unsigned_or_unchained_output_writer_is_rejected(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    flag: str,
) -> None:
    root = tmp_path / f"invalid-output-{flag}"
    _precreate_output_writer_config(root, ceremony_override=(flag, False))

    with pytest.raises(ValueError, match="output writer profile"):
        _provision(root, governed_portfolio)


@pytest.mark.parametrize("flag", ["sign", "chain"])
def test_reused_unsigned_or_unchained_source_writer_fails_before_package(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    monkeypatch,
    flag: str,
) -> None:
    root = tmp_path / f"invalid-source-{flag}"
    _precreate_source_writer_config(root, ceremony_override=(flag, False))
    original_add_runtime = tn.admin.add_agent_runtime
    packages_issued: list[Path] = []

    def record_package_issuance(*args, **kwargs):
        package = original_add_runtime(*args, **kwargs)
        packages_issued.append(Path(package))
        return package

    monkeypatch.setattr(tn.admin, "add_agent_runtime", record_package_issuance)

    with pytest.raises(ValueError, match="source writer profile"):
        _provision(root, governed_portfolio)
    assert packages_issued == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cross_route", "field routing"),
        ("public_policy", "must be private"),
        ("group_aad", "without default AAD"),
    ],
)
def test_source_profile_validator_rejects_load_bearing_config_mutations(
    tn_flow: TNFlowResult,
    mutation: str,
    message: str,
) -> None:
    config_path = tn_flow.source_row.source_log_path.parent.parent / "tn.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if mutation == "cross_route":
        document["groups"]["tn.agents"]["fields"].append("deepvest_result")
    elif mutation == "public_policy":
        document["groups"]["finance"]["policy"] = "public"
    else:
        document["groups"]["finance"]["aad"] = {"purpose": "unexpected"}
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _validate_source_writer_profile(config_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("cross_route", "field routing"),
        ("public_policy", "must be private"),
        ("group_aad", "without default AAD"),
        ("public_field", "literal public_fields"),
        ("extra_group", "exactly default and tn.agents"),
    ],
)
def test_output_profile_validator_rejects_load_bearing_config_mutations(
    tn_flow: TNFlowResult,
    mutation: str,
    message: str,
) -> None:
    config_path = tn_flow.output_writer.config_path
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if mutation == "cross_route":
        document["groups"]["tn.agents"]["fields"].append("default")
    elif mutation == "public_policy":
        document["groups"]["default"]["policy"] = "public"
    elif mutation == "group_aad":
        document["groups"]["default"]["aad"] = {"purpose": "unexpected"}
    elif mutation == "public_field":
        document["public_fields"] = ["report_html"]
    else:
        document["groups"]["extra"] = dict(document["groups"]["default"])
        document["groups"]["extra"]["fields"] = []
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _validate_output_writer_profile(config_path)


def test_marker_free_canonical_row_is_retained_without_tn_aad(tn_flow: TNFlowResult) -> None:
    marker_free = tn_flow.marker_free_row

    assert marker_free.raw_ndjson.endswith(b"\n")
    assert "tn_aad" not in marker_free.envelope
    assert marker_free.encrypted_groups == ("finance", "tn.agents")


def test_exposed_task5_result_is_deeply_immutable_and_contains_no_writer_key(
    tn_flow: TNFlowResult,
) -> None:
    with pytest.raises(FrozenInstanceError):
        tn_flow.source_row = tn_flow.marker_free_row
    with pytest.raises(FrozenInstanceError):
        tn_flow.source_row.event_id = "replaced"
    with pytest.raises(TypeError):
        tn_flow.source_row.envelope["event_id"] = "replaced"
    with pytest.raises(TypeError):
        tn_flow.source_row.envelope["finance"]["ciphertext"] = "replaced"
    with pytest.raises(FrozenInstanceError):
        tn_flow.input_reader.recipient_did = "did:key:replaced"
    with pytest.raises(FrozenInstanceError):
        tn_flow.output_writer.reader = tn_flow.input_reader

    for exposed in (
        tn_flow,
        tn_flow.source_row,
        tn_flow.marker_free_row,
        tn_flow.input_reader,
        tn_flow.output_writer,
        tn_flow.output_writer.reader,
    ):
        assert not hasattr(exposed, "local_private")
        assert not hasattr(exposed, "index_master")
        assert not hasattr(exposed, "keystore")
    assert not {"permit", "nonce", "authorization"} & set(
        tn_flow.source_row.envelope
    )


def test_fresh_process_official_reader_opens_both_exact_target_groups(
    tn_flow: TNFlowResult,
) -> None:
    script = """
import json
import sys
import tn

log_path, reader_dir, event_id = sys.argv[1:]
result = {}
for group in ("finance", "tn.agents"):
    matches = [
        entry
        for entry in tn.reader.read_as_recipient(log_path, reader_dir, group=group)
        if entry["envelope"].get("event_id") == event_id
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one target for {group}, got {len(matches)}")
    result[group] = matches[0]
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        [
            str(Path(tn.__file__).parents[1].parent.parent / "Scripts" / "python.exe"),
            "-c",
            script,
            str(tn_flow.source_row.source_log_path),
            str(tn_flow.input_reader.reader_key_dir),
            tn_flow.source_row.event_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    opened = json.loads(completed.stdout)

    for group in ("finance", "tn.agents"):
        assert opened[group]["envelope"] == dict(tn_flow.source_row.envelope)
        assert opened[group]["valid"] == {
            "record": True,
            "row_hash": True,
            "signature": True,
            "chain": True,
            "aad": True,
        }


def test_missing_and_wrong_finance_kits_cannot_open_finance(
    tmp_path: Path,
    tn_flow: TNFlowResult,
) -> None:
    missing_dir = tmp_path / "missing-finance"
    missing_dir.mkdir()
    shutil.copyfile(
        tn_flow.input_reader.reader_key_dir / "tn.agents.btn.mykit",
        missing_dir / "tn.agents.btn.mykit",
    )
    with pytest.raises(FileNotFoundError, match="finance"):
        list(
            tn.reader.read_as_recipient(
                tn_flow.source_row.source_log_path,
                missing_dir,
                group="finance",
            )
        )

    wrong_dir = tmp_path / "wrong-finance"
    wrong_dir.mkdir()
    shutil.copyfile(
        tn_flow.output_writer.reader.reader_key_dir / "default.btn.mykit",
        wrong_dir / "finance.btn.mykit",
    )
    wrong = list(
        tn.reader.read_as_recipient(
            tn_flow.source_row.source_log_path,
            wrong_dir,
            group="finance",
        )
    )
    target = [
        entry
        for entry in wrong
        if entry["envelope"].get("event_id") == tn_flow.source_row.event_id
    ]
    assert len(target) == 1
    assert target[0]["plaintext"]["finance"] == {"$no_read_key": True}
    assert "portfolio_values" not in target[0]["plaintext"]["finance"]


@pytest.mark.parametrize("mutation", ["missing", "non_kit"])
def test_reader_material_validation_rejects_missing_or_non_kit_entries(
    tmp_path: Path,
    tn_flow: TNFlowResult,
    mutation: str,
) -> None:
    invalid_dir = tmp_path / f"invalid-reader-{mutation}"
    invalid_dir.mkdir()
    for kit in tn_flow.input_reader.reader_key_dir.iterdir():
        shutil.copyfile(kit, invalid_dir / kit.name)
    if mutation == "missing":
        (invalid_dir / "finance.btn.mykit").unlink()
    else:
        (invalid_dir / "index_master.key").write_bytes(b"not allowed here")
    invalid = replace(tn_flow.input_reader, reader_key_dir=invalid_dir.resolve())

    with pytest.raises(ValueError, match="exactly|only"):
        validate_runtime_reader_material(
            invalid,
            expected_groups=("finance", "tn.agents"),
        )


def test_reader_material_validation_rejects_duplicate_expected_kit_names(
    tn_flow: TNFlowResult,
) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        validate_runtime_reader_material(
            tn_flow.input_reader,
            expected_groups=("finance", "finance", "tn.agents"),
        )


def test_runtime_package_validation_rejects_duplicate_kit_entries(
    tmp_path: Path,
    tn_flow: TNFlowResult,
) -> None:
    duplicate = tmp_path / "duplicate-kit.tnpkg"
    with zipfile.ZipFile(tn_flow.input_reader.package_path) as original:
        entries = [(name, original.read(name)) for name in original.namelist()]
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(duplicate, "w") as package,
    ):
        for name, body in entries:
            package.writestr(name, body)
        package.writestr("body/finance.btn.mykit", entries[1][1])

    with pytest.raises(ValueError, match="duplicate"):
        _validate_package(duplicate, ("finance", "tn.agents"))


def test_runtime_reader_validation_rejects_truncated_index_master(
    tmp_path: Path,
    tn_flow: TNFlowResult,
) -> None:
    truncated = tmp_path / "truncated.index.key"
    truncated.write_bytes(tn_flow.input_reader.index_key_path.read_bytes()[:-1])
    invalid = replace(tn_flow.input_reader, index_key_path=truncated.resolve())

    with pytest.raises(ValueError, match="exactly 32 bytes"):
        validate_runtime_reader_material(
            invalid,
            expected_groups=("finance", "tn.agents"),
        )


@pytest.mark.parametrize("missing_group", ["default", "tn.agents"])
def test_output_reader_validation_rejects_each_missing_required_kit(
    tmp_path: Path,
    tn_flow: TNFlowResult,
    missing_group: str,
) -> None:
    invalid_dir = tmp_path / f"output-missing-{missing_group}"
    invalid_dir.mkdir()
    for kit in tn_flow.output_writer.reader.reader_key_dir.iterdir():
        if kit.name != f"{missing_group}.btn.mykit":
            shutil.copyfile(kit, invalid_dir / kit.name)
    invalid = replace(tn_flow.output_writer.reader, reader_key_dir=invalid_dir.resolve())

    with pytest.raises(ValueError, match="exactly"):
        validate_runtime_reader_material(
            invalid,
            expected_groups=("default", "tn.agents"),
        )


def _flip_terminal_character(value: str) -> str:
    return value[:-1] + ("A" if value[-1] != "A" else "B")


@pytest.mark.parametrize(
    ("mutation", "expected_valid", "plaintext_must_be_hidden"),
    [
        (
            "ciphertext",
            {
                "record": True,
                "signature": True,
                "row_hash": False,
                "chain": True,
                "aad": True,
            },
            True,
        ),
        (
            "commitment",
            {
                "record": True,
                "signature": True,
                "row_hash": False,
                "chain": True,
                "aad": True,
            },
            False,
        ),
        (
            "signature",
            {
                "record": True,
                "signature": False,
                "row_hash": True,
                "chain": True,
                "aad": True,
            },
            False,
        ),
        (
            "aad_governed_by",
            {
                "record": True,
                "signature": True,
                "row_hash": False,
                "chain": True,
                "aad": True,
            },
            True,
        ),
        (
            "aad_policy",
            {
                "record": True,
                "signature": True,
                "row_hash": False,
                "chain": True,
                "aad": True,
            },
            True,
        ),
    ],
)
def test_official_reader_marks_wire_mutations_invalid(
    tmp_path: Path,
    tn_flow: TNFlowResult,
    mutation: str,
    expected_valid: dict[str, bool],
    plaintext_must_be_hidden: bool,
) -> None:
    envelope = json.loads(tn_flow.source_row.raw_ndjson)
    if mutation == "ciphertext":
        ciphertext = envelope["finance"]["ciphertext"]
        decoded = bytearray(base64.b64decode(ciphertext, validate=True))
        decoded[0] ^= 0x01
        envelope["finance"]["ciphertext"] = base64.b64encode(decoded).decode()
    elif mutation == "commitment":
        field_hashes = envelope["finance"]["field_hashes"]
        field = next(iter(field_hashes))
        field_hashes[field] = _flip_terminal_character(field_hashes[field])
    elif mutation == "signature":
        signature = envelope["signature"]
        decoded = bytearray(base64.urlsafe_b64decode(signature + "=="))
        decoded[0] ^= 0x01
        envelope["signature"] = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode()
    else:
        aad = json.loads(envelope["tn_aad"])
        field = "governed_by" if mutation == "aad_governed_by" else "policy"
        aad["finance"][field] += "-mutated"
        envelope["tn_aad"] = json.dumps(aad, separators=(",", ":"), sort_keys=True)
    mutated_log = tmp_path / f"{mutation}.ndjson"
    mutated_log.write_text(
        json.dumps(envelope, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    opened = list(
        tn.reader.read_as_recipient(
            mutated_log,
            tn_flow.input_reader.reader_key_dir,
            group="finance",
        )
    )

    assert len(opened) == 1
    assert opened[0]["envelope"]["event_id"] == tn_flow.source_row.event_id
    assert opened[0]["valid"] == expected_valid
    if mutation == "signature":
        assert opened[0]["valid"]["signature"] is False
    if plaintext_must_be_hidden:
        assert opened[0]["plaintext"]["finance"] == {"$no_read_key": True}
        plaintext_text = json.dumps(opened[0]["plaintext"], sort_keys=True)
        assert "portfolio_values" not in plaintext_text
        assert "deepvest_result" not in plaintext_text


def test_unrouted_automatic_run_id_fails_source_profile_before_package(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    monkeypatch,
) -> None:
    original = tn.ensure_group

    def omit_run_id(cfg, group, *, pool_size=4, fields=None, cipher=None):
        if group == "finance" and fields is not None:
            fields = [field for field in fields if field != "run_id"]
        return original(
            cfg,
            group,
            pool_size=pool_size,
            fields=fields,
            cipher=cipher,
        )

    monkeypatch.setattr(tn, "ensure_group", omit_run_id)

    root = tmp_path / "unrouted"
    with pytest.raises(ValueError, match="source writer profile field routing"):
        _provision(root, governed_portfolio)
    assert not (root / "packages" / "input-runtime.tnpkg").exists()


@pytest.mark.parametrize("finance_epoch,agents_epoch", [(1, 1), (1, 0)])
def test_source_rotated_or_mismatched_index_epochs_fail_before_package_issuance(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    monkeypatch,
    finance_epoch: int,
    agents_epoch: int,
) -> None:
    original_ensure_group = tn.ensure_group
    original_add_runtime = tn.admin.add_agent_runtime
    packages_issued: list[Path] = []

    def rotate_source_epochs(cfg, group, *, pool_size=4, fields=None, cipher=None):
        result = original_ensure_group(
            cfg,
            group,
            pool_size=pool_size,
            fields=fields,
            cipher=cipher,
        )
        if group == "finance":
            document = yaml.safe_load(cfg.yaml_path.read_text(encoding="utf-8"))
            document["groups"]["finance"]["index_epoch"] = finance_epoch
            document["groups"]["tn.agents"]["index_epoch"] = agents_epoch
            cfg.yaml_path.write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
        return result

    def record_package_issuance(*args, **kwargs):
        package = original_add_runtime(*args, **kwargs)
        packages_issued.append(Path(package))
        return package

    monkeypatch.setattr(tn, "ensure_group", rotate_source_epochs)
    monkeypatch.setattr(tn.admin, "add_agent_runtime", record_package_issuance)

    with pytest.raises(ValueError, match="index epoch 0"):
        _provision(tmp_path / "invalid-source-epoch", governed_portfolio)
    assert packages_issued == []


@pytest.mark.parametrize("default_epoch,agents_epoch", [(1, 1), (1, 0)])
def test_output_rotated_or_mismatched_index_epochs_fail_before_output_package(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    default_epoch: int,
    agents_epoch: int,
) -> None:
    root = tmp_path / "invalid-output-epoch"
    _precreate_output_writer_config(root, ceremony_override=("sign", True))
    config_path = root / "output-writer" / "tn.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    document["groups"]["default"]["index_epoch"] = default_epoch
    document["groups"]["tn.agents"]["index_epoch"] = agents_epoch
    config_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index epoch 0"):
        _provision(root, governed_portfolio)
    assert not (root / "packages" / "output-runtime.tnpkg").exists()


def test_output_writer_and_recipient_alias_fails_before_package_or_absorption(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    monkeypatch,
) -> None:
    root = tmp_path / "aliased-output-recipient"
    original_absorb = tn.TN.absorb
    absorbed_packages: list[str] = []

    def record_absorb(recipient, package_path):
        absorbed_packages.append(Path(package_path).name)
        return original_absorb(recipient, package_path)

    monkeypatch.setattr(tn.TN, "absorb", record_absorb)

    with pytest.raises(ValueError, match="publisher and recipient"):
        _provision(root, governed_portfolio, alias_output_recipient=True)

    assert absorbed_packages == ["input-runtime.tnpkg"]
    assert not (root / "packages" / "output-runtime.tnpkg").exists()


def test_tn_flow_consumes_prepared_values_without_recalling_sibling_boundaries(
    tmp_path: Path,
    governed_portfolio: GovernedPortfolioResult,
    monkeypatch,
) -> None:
    execution = execute_deepvest(StubDeepVest(), governed_portfolio)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Task 5 recalled a forbidden sibling or network boundary")

    monkeypatch.setattr(
        "deepvest_tn.polars_stage.run_governed_polars_stage",
        forbidden,
    )
    monkeypatch.setattr("deepvest_tn.deepvest.execute_deepvest", forbidden)
    monkeypatch.setattr(StubDeepVest, "calculate", forbidden)
    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    result = _provision(
        tmp_path / "no-sibling-recall",
        governed_portfolio,
        execution,
    )

    assert result.source_row.event_id
