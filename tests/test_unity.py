from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path

import polars as pl
import pytest
from tn_polars.registry import LocalRegistry, _load_entry

from tn_governed.unity import (
    FEED_NAMES,
    MARKET_PRICES_FEED,
    POLICY_REF,
    POSITIONS_FEED,
    UnityCatalogConfig,
    build_finance_feed_bundles,
    publish_finance_feeds,
    resolve_finance_feeds,
    unity_catalog_registry,
)


class RecordingUnityTransport:
    def __init__(self) -> None:
        self.volume_requests: list[dict] = []
        self.volumes: dict[str, str] = {}

    def request(self, method: str, path: str, payload=None):
        if method == "POST" and path == "/volumes":
            self.volume_requests.append(payload)
            full_name = f"{payload['catalog_name']}.{payload['schema_name']}.{payload['name']}"
            self.volumes[full_name] = payload["storage_location"]
        if method == "GET" and path.startswith("/volumes/"):
            return {"storage_location": self.volumes[path.removeprefix("/volumes/")]}
        return {}


def install_recording_transport(monkeypatch) -> RecordingUnityTransport:
    transport = RecordingUnityTransport()

    def request(_registry, method: str, path: str, payload=None):
        return transport.request(method, path, payload)

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", request)
    return transport


def reserve_consecutive_port_pair() -> tuple[int, socket.socket]:
    """Reserve the second port while returning a currently free public port."""
    for _ in range(50):
        public_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        public_socket.bind(("127.0.0.1", 0))
        public_port = public_socket.getsockname()[1]
        if public_port >= 65534:
            public_socket.close()
            continue
        internal_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            internal_socket.bind(("127.0.0.1", public_port + 1))
        except OSError:
            public_socket.close()
            internal_socket.close()
            continue
        public_socket.close()
        return public_port, internal_socket
    raise RuntimeError("could not reserve a free consecutive loopback port pair")


@pytest.mark.parametrize("occupied_port", ["public", "internal"])
def test_launcher_rejects_each_unity_port_before_java_launch(
    occupied_port: str, tmp_path: Path,
) -> None:
    unity_root = tmp_path / "unity"
    server_args = unity_root / "server" / "target" / "server.args"
    server_args.parent.mkdir(parents=True)
    server_args.write_text("", encoding="utf-8")
    launch_marker = tmp_path / "java-started.txt"
    java_command = tmp_path / "java.cmd"
    java_command.write_text(
        f'@echo off\n> "{launch_marker}" echo started\nexit /b 9\n', encoding="utf-8"
    )
    public_port, internal_socket = reserve_consecutive_port_pair()
    if occupied_port == "public":
        internal_socket.close()
        occupied_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied_socket.bind(("127.0.0.1", public_port))
        expected_port = public_port
    else:
        occupied_socket = internal_socket
        expected_port = public_port + 1
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(Path("scripts/start-local-unity.ps1").resolve()),
                "-UnityRoot",
                str(unity_root),
                "-JavaExe",
                str(java_command),
                "-Port",
                str(public_port),
                "-TimeoutSeconds",
                "5",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        occupied_socket.close()

    assert result.returncode != 0
    assert not launch_marker.exists()
    assert f"{occupied_port} loopback port {expected_port} is already occupied" in (
        result.stdout + result.stderr
    )


def test_builds_two_complete_real_tn_polars_bundles(tmp_path: Path) -> None:
    bundles = build_finance_feed_bundles(tmp_path)

    assert tuple(bundles) == FEED_NAMES
    expected_files = {
        "compiled_policy.json",
        "tn_governance.json",
        "profile.json",
        "policy.wasm",
        "data.json",
    }
    expected_fields = {
        POSITIONS_FEED: {
            "account_id": "confidential",
            "portfolio_id": "confidential",
            "symbol": "internal",
            "quantity": "confidential",
            "cost_basis": "confidential",
            "currency": "internal",
        },
        MARKET_PRICES_FEED: {
            "symbol": "internal",
            "as_of": "internal",
            "last_price": "internal",
            "currency": "internal",
            "market_source": "internal",
        },
    }

    for name, bundle in bundles.items():
        assert {path.name for path in bundle.iterdir()} == expected_files
        compiled = json.loads((bundle / "compiled_policy.json").read_text(encoding="utf-8"))
        profile = json.loads((bundle / "profile.json").read_text(encoding="utf-8"))
        source = pl.scan_ndjson(
            Path("fixtures/portfolio")
            / ("internal_positions.ndjson" if name == POSITIONS_FEED else "market_prices.ndjson")
        )
        entry = _load_entry(name, bundle, described_as="test bundle")

        assert compiled["policy_ref"] == POLICY_REF
        assert compiled["fields"] == expected_fields[name]
        assert profile["dtypes"] == {
            name: str(dtype) for name, dtype in source.collect_schema().items()
        }
        assert entry.policy.policy_ref == POLICY_REF
        assert entry.evaluator()

    entries = {
        name: _load_entry(name, bundle, described_as="test bundle")
        for name, bundle in bundles.items()
    }
    assert entries[POSITIONS_FEED].tn_governance == entries[MARKET_PRICES_FEED].tn_governance
    assert {entry.policy.policy_ref for entry in entries.values()} == {POLICY_REF}


def test_bundle_load_contract_rejects_missing_required_file(tmp_path: Path) -> None:
    bundle = build_finance_feed_bundles(tmp_path)[POSITIONS_FEED]
    (bundle / "policy.wasm").unlink()
    registry = LocalRegistry(tmp_path / "deterministic-registry")
    registry.register_feed(POSITIONS_FEED, bundle)

    with pytest.raises(Exception, match="policy.wasm"):
        registry.get(POSITIONS_FEED)


def test_policy_and_dtype_pins_reject_drift(tmp_path: Path) -> None:
    bundle = build_finance_feed_bundles(tmp_path)[POSITIONS_FEED]
    compiled_path = bundle / "compiled_policy.json"
    compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
    compiled["fields"].pop("cost_basis")
    compiled_path.write_text(json.dumps(compiled), encoding="utf-8")

    entry = _load_entry(POSITIONS_FEED, bundle, described_as="tampered bundle")
    with pytest.raises(
        Exception, match="columns arrived that the compiled policy does not classify"
    ):
        from tn_polars.builder import GovernedPolars

        GovernedPolars.prepare(entry).load(pl.scan_ndjson("fixtures/portfolio/internal_positions.ndjson"))

    bundle = build_finance_feed_bundles(tmp_path / "dtype")[POSITIONS_FEED]
    profile_path = bundle / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["dtypes"]["quantity"] = "String"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    entry = _load_entry(POSITIONS_FEED, bundle, described_as="dtype bundle")

    with pytest.raises(Exception, match="column types differ"):
        from tn_polars.builder import GovernedPolars

        GovernedPolars.prepare(entry).load(pl.scan_ndjson("fixtures/portfolio/internal_positions.ndjson"))


def test_rejects_different_policy_references_between_feeds(tmp_path: Path) -> None:
    governance_path = Path("fixtures/portfolio/tn_governance.json")
    governance = json.loads(governance_path.read_text(encoding="utf-8"))
    governance["policy"] = "tn://policy/other@1"
    altered = tmp_path / "governance.json"
    altered.write_text(json.dumps(governance), encoding="utf-8")

    with pytest.raises(ValueError, match="common policy reference"):
        build_finance_feed_bundles(tmp_path / "bundles", governance_path=altered)


def test_publish_and_resolve_both_exact_unity_feeds(tmp_path: Path, monkeypatch) -> None:
    transport = install_recording_transport(monkeypatch)
    config = UnityCatalogConfig()
    result = publish_finance_feeds(tmp_path, config=config)
    resolved = resolve_finance_feeds(config)

    assert result.feed_names == FEED_NAMES
    assert [request["name"] for request in transport.volume_requests] == [
        "portfolio_internal_positions",
        "portfolio_market_prices",
    ]
    assert transport.volumes == {
        "portfolio.finance.portfolio_internal_positions": result.positions_bundle.resolve().as_uri(),
        "portfolio.finance.portfolio_market_prices": result.market_prices_bundle.resolve().as_uri(),
    }
    assert resolved[POSITIONS_FEED].policy.fields["cost_basis"] == "confidential"
    assert resolved[MARKET_PRICES_FEED].policy.fields["last_price"] == "internal"
    assert resolved[POSITIONS_FEED].tn_governance == resolved[MARKET_PRICES_FEED].tn_governance


@pytest.mark.parametrize("bad_location", ["swapped", "duplicate", "missing"])
def test_resolve_rejects_wrong_registered_bundle_locations(
    tmp_path: Path, monkeypatch, bad_location: str
) -> None:
    transport = install_recording_transport(monkeypatch)
    config = UnityCatalogConfig()
    published = publish_finance_feeds(tmp_path, config=config)
    market_volume = "portfolio.finance.portfolio_market_prices"
    if bad_location in {"swapped", "duplicate"}:
        transport.volumes[market_volume] = published.positions_bundle.resolve().as_uri()
    else:
        transport.volumes[market_volume] = (tmp_path / "missing").resolve().as_uri()

    with pytest.raises(Exception, match="expected fields|incomplete"):
        resolve_finance_feeds(config)


def test_unity_configuration_defaults() -> None:
    config = UnityCatalogConfig()
    registry = unity_catalog_registry(config)

    assert (registry.base_url, registry.catalog, registry.schema, registry.token) == (
        "http://127.0.0.1:8080", "portfolio", "finance", None
    )
