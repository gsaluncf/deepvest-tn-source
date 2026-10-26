from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
import uuid
import venv
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
import tn
from tn_llama_client.artifacts import save_report_html as sibling_save_report_html
from tn_llama_client.clients import GovernedRequestDenied
from tn_llama_client.protocol import ReportRelease
from tn_polars.admission import SchemaPinViolation
from tn_polars.release import PolicyDenied

from deepvest_tn import cli as cli_module
from deepvest_tn import run as run_module
from deepvest_tn.feed_server import FeedServerConfig, create_feed_server, seed_finance_feed_data
from deepvest_tn.getdata import FeedFetchHTTPError, FeedFetchResponseError, FetchedFeed
from deepvest_tn.run import (
    FirstPassConfig,
    FirstPassResult,
    GovernedServerSpec,
    ReferenceRunConfig,
    run_first_pass,
    run_reference_path,
)
from deepvest_tn.tn_flow import TNFlowResult
from deepvest_tn.unity import (
    FEED_NAMES,
    MARKET_PRICES_FEED,
    POSITIONS_FEED,
    UnityCatalogConfig,
    build_finance_feed_bundles,
)

_ACTIVE_FEED_URL = ""


def test_report_prompt_pins_the_mandatory_disclosure_as_the_final_paragraph() -> None:
    fixed_narrative = (
        "This first-pass report demonstrates governed transport of supplied portfolio "
        "positions and provided DeepVest stub risk metrics without recalculation; it does "
        "not claim data-specific narrative generation."
    )
    disclosure_paragraph = f"<p>{run_module._STUB_DISCLOSURE}</p>"

    assert fixed_narrative == run_module._FIXED_NARRATIVE_PARAGRAPH
    assert run_module._REPORT_PROMPT.count(run_module._STUB_DISCLOSURE) == 1
    assert run_module._REPORT_PROMPT.count(fixed_narrative) == 1
    assert (
        f"<p>{fixed_narrative}</p>\n{disclosure_paragraph}\n</body></html>"
        in run_module._REPORT_PROMPT
    )
    assert "Write the concise portfolio narrative here" not in run_module._REPORT_PROMPT
    assert "Place the narrative next" not in run_module._REPORT_PROMPT
    assert (
        "Do not call emit_governed_record until report_html ends with that exact final "
        "disclosure paragraph."
    ) in run_module._REPORT_PROMPT


@pytest.fixture(scope="module", autouse=True)
def local_getdata_service(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, Path]]:
    global _ACTIVE_FEED_URL
    data_dir = seed_finance_feed_data(tmp_path_factory.mktemp("getdata") / "data")
    server = create_feed_server(
        FeedServerConfig(data_dir=data_dir, event_stream=io.StringIO(), port=0)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    _ACTIVE_FEED_URL = f"http://{host}:{port}"
    try:
        yield _ACTIVE_FEED_URL, data_dir
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        _ACTIVE_FEED_URL = ""


@pytest.fixture(autouse=True)
def isolated_tn(monkeypatch: pytest.MonkeyPatch):
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
    return build_finance_feed_bundles(tmp_path_factory.mktemp("finance-bundles"))


class RecordingUnityTransport:
    def __init__(self, bundles: dict[str, Path], config: UnityCatalogConfig) -> None:
        self.get_paths: list[str] = []
        self.volumes = {
            f"{config.catalog}.{config.schema}.{name.replace('.', '_')}": bundle.resolve().as_uri()
            for name, bundle in bundles.items()
        }

    def request(self, method: str, path: str, payload=None):
        if method == "GET" and path.startswith("/volumes/"):
            self.get_paths.append(path)
            return {"storage_location": self.volumes[path.removeprefix("/volumes/")]}
        raise AssertionError(f"unexpected Unity request: {method} {path} {payload}")


def _install_unity_transport(
    monkeypatch: pytest.MonkeyPatch,
    bundles: dict[str, Path],
    *,
    base_url: str,
) -> RecordingUnityTransport:
    config = UnityCatalogConfig(base_url=base_url)
    transport = RecordingUnityTransport(bundles, config)

    def request(_registry, method: str, path: str, payload=None):
        return transport.request(method, path, payload)

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", request)
    return transport


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _canonical_release_body(flow: TNFlowResult) -> bytes:
    writer = tn.init(flow.output_writer.config_path, link=False, stdout=False)
    policy_document = tn.agents.policy()
    assert policy_document is not None
    template = policy_document.templates["report.generated"]
    policy_reference = (
        f"{template.path}#{template.event_type}@{template.version}#{template.content_hash}"
    )
    report_html = (
        "<p>DeepVest calculations in this first-pass report are deterministic "
        "stub outputs, not live DeepVest results.</p>"
    )
    emitted = writer.log(
        "report.generated",
        level="info",
        aad={"governed_by": writer.cfg.device.device_identity, "policy": policy_reference},
        default={"report_html": report_html},
        source_lineage={
            "source_device_identity": flow.source_row.source_writer_did,
            "source_event_id": flow.source_row.event_id,
            "source_event_type": "portfolio.deepvest.calculated",
            "derived_from_row_hash": flow.source_row.row_hash,
            "source_policy": flow.source_row.policy_reference,
            "selected_sections": ["finance"],
            "operation": "derived_html_report",
        },
    )
    assert emitted is not None
    matches = [
        raw
        for raw in flow.output_writer.log_path.read_bytes().splitlines(keepends=True)
        if (
            (parsed := json.loads(raw)).get("event_id") == emitted["event_id"]
            and parsed.get("row_hash") == emitted["row_hash"]
        )
    ]
    assert len(matches) == 1
    output_ndjson = matches[0].decode("utf-8")
    return json.dumps(
        {
            "protocol": "tn.llama.release.v1",
            "run_id": "test-deterministic-run",
            "source_event_id": flow.source_row.event_id,
            "source_row_hash": flow.source_row.row_hash,
            "output_event_id": emitted["event_id"],
            "output_row_hash": emitted["row_hash"],
            "output_ndjson": output_ndjson,
            "trace": [],
        },
        separators=(",", ":"),
    ).encode()


@contextmanager
def _serve_response(
    *,
    port: int,
    body: bytes,
    status: int = 200,
) -> Iterator[list[dict[str, object]]]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            length = int(self.headers.get("Content-Length", "0"))
            request_body = self.rfile.read(length)
            requests.append(
                {
                    "path": self.path,
                    "body": request_body,
                    "authorization": self.headers.get("Authorization"),
                }
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _install_fake_local_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    project_root: Path,
) -> tuple[Path, Path, Path]:
    project_root.mkdir(parents=True, exist_ok=True)
    dependency_root = project_root / "read-only-dependencies"
    build_directory = dependency_root / "governed-build"
    bin_directory = build_directory / "bin"
    bin_directory.mkdir(parents=True)
    server_path = bin_directory / "llama-server.exe"
    server_path.write_bytes(b"test-governed-server")
    (bin_directory / "llama.dll").write_bytes(b"test-companion")
    model_path = dependency_root / "model.gguf"
    model_path.write_bytes(b"test-model")
    launcher_path = dependency_root / "run-tn-server.ps1"
    launcher_path.write_text("# deterministic process double", encoding="utf-8")
    monkeypatch.setattr(run_module, "_PROJECT_ROOT", project_root.resolve())
    monkeypatch.setattr(run_module, "_LLAMA_BUILD_DIRECTORY", build_directory.resolve())
    monkeypatch.setattr(run_module, "_GOVERNED_SERVER_PATH", server_path.resolve())
    monkeypatch.setattr(run_module, "_LLAMA_MODEL_PATH", model_path.resolve())
    monkeypatch.setattr(run_module, "_LLAMA_LAUNCHER_PATH", launcher_path.resolve())
    return build_directory, server_path, model_path


def _first_pass_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
) -> tuple[Path, FirstPassConfig]:
    project_root = tmp_path / "project"
    _install_fake_local_dependencies(monkeypatch, project_root)
    run_directory = (project_root / "experiments" / "results" / name).resolve()
    return project_root, FirstPassConfig(
        run_directory=run_directory,
        unity_url="http://127.0.0.1:18080",
        llama_port=_free_port(),
        api_key="fixture-api-key",
        feed_url=_ACTIVE_FEED_URL,
    )


def _assert_no_released_outputs(run_directory: Path) -> None:
    assert not (run_directory / "run-summary.json").exists()
    assert list((run_directory / "reports").glob("*.html")) == []


def test_reference_run_executes_the_tn_object_path_through_the_governed_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
) -> None:
    project_root = tmp_path / "project"
    _install_fake_local_dependencies(monkeypatch, project_root)
    run_directory = (project_root / "experiments" / "results" / "reference").resolve()
    unity_url = "http://127.0.0.1:18080"
    unity_config = run_module._reference_unity_config(run_directory, unity_url)
    unity_transport = RecordingUnityTransport(finance_bundles, unity_config)
    monkeypatch.setattr(
        "tn_polars.registry.UnityCatalogRegistry._request",
        lambda _registry, method, path, payload=None: unity_transport.request(
            method, path, payload
        ),
    )
    monkeypatch.setattr(
        run_module,
        "publish_finance_feeds",
        lambda output_dir, *, config: SimpleNamespace(
            feed_names=FEED_NAMES,
            positions_bundle=output_dir / POSITIONS_FEED,
            market_prices_bundle=output_dir / MARKET_PRICES_FEED,
        ),
    )
    port = _free_port()
    captured: dict[str, object] = {}
    real_build = run_module.build_reference_flow

    def observed_build(*args, **kwargs):
        flow = real_build(*args, **kwargs)
        captured["flow"] = flow
        return flow

    monkeypatch.setattr(run_module, "build_reference_flow", observed_build)

    @contextmanager
    def governed_transport(spec: GovernedServerSpec) -> Iterator[None]:
        spec.stdout_path.touch()
        spec.stderr_path.touch()
        body = _canonical_release_body(captured["flow"])
        with _serve_response(port=spec.port, body=body) as requests:
            captured["requests"] = requests
            yield

    result = run_reference_path(
        ReferenceRunConfig(
            run_directory=run_directory,
            unity_url=unity_url,
            llama_port=port,
            api_key="reference-test-key",
        ),
        server_lifecycle_factory=governed_transport,
        witness_writer=lambda run: run / "nine-question-witness.json",
    )

    assert result.report_path.is_file()
    assert result.facts.stub_invocation_count == 1
    assert result.facts.response_status == 200
    assert len(captured["requests"]) == 1
    request = json.loads(captured["requests"][0]["body"])
    assert request["tn_row"] == dict(captured["flow"].source_row.envelope)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["getdata"]["representation"] == "application/vnd.tn+json"
    assert summary["deepvest"]["live"] is False
    assert summary["tn_llama"]["route"] == "/v1/tn/chat/completions"


def test_first_pass_composes_real_stages_and_saves_only_recipient_opened_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
) -> None:
    project_root = tmp_path / "project"
    _install_fake_local_dependencies(monkeypatch, project_root)
    run_directory = (project_root / "experiments" / "results" / "permitted").resolve()
    unity_url = "http://127.0.0.1:18080"
    unity = _install_unity_transport(
        monkeypatch,
        finance_bundles,
        base_url=unity_url,
    )
    api_key = "test-sentinel-api-key"
    port = _free_port()
    events: list[str] = []
    handoff: dict[str, object] = {}

    real_polars = run_module.run_governed_polars_stage
    real_deepvest = run_module.execute_deepvest
    real_tn = run_module.provision_tn_flow
    real_config = run_module.write_finance_runtime_config
    real_submit = run_module.submit_finance_report
    real_summary = run_module._write_summary

    def observed_polars(*args, **kwargs):
        events.append("polars")
        return real_polars(*args, **kwargs)

    def observed_deepvest(*args, **kwargs):
        events.append("deepvest")
        return real_deepvest(*args, **kwargs)

    def observed_tn(*args, **kwargs):
        events.append("tn")
        flow = real_tn(*args, **kwargs)
        handoff["flow"] = flow
        return flow

    def observed_config(*args, **kwargs):
        events.append("runtime_config")
        return real_config(*args, **kwargs)

    def observed_submit(*args, **kwargs):
        events.append("submit")
        return real_submit(*args, **kwargs)

    def observed_save(release: ReportRelease, directory: Path) -> Path:
        events.append("save_report")
        return sibling_save_report_html(release, directory)

    def observed_summary(path: Path, document: dict[str, object]) -> Path:
        events.append("save_summary")
        return real_summary(path, document)

    monkeypatch.setattr(run_module, "run_governed_polars_stage", observed_polars)
    monkeypatch.setattr(run_module, "execute_deepvest", observed_deepvest)
    monkeypatch.setattr(run_module, "provision_tn_flow", observed_tn)
    monkeypatch.setattr(run_module, "write_finance_runtime_config", observed_config)
    monkeypatch.setattr(run_module, "submit_finance_report", observed_submit)
    monkeypatch.setattr(run_module, "save_report_html", observed_save)
    monkeypatch.setattr(run_module, "_write_summary", observed_summary)

    @contextmanager
    def governed_transport(spec: GovernedServerSpec) -> Iterator[None]:
        events.append("server_enter")
        handoff["server_spec"] = spec
        spec.stdout_path.touch()
        spec.stderr_path.touch()
        body = _canonical_release_body(handoff["flow"])
        with _serve_response(port=spec.port, body=body) as requests:
            handoff["requests"] = requests
            yield
        events.append("server_exit")

    before = {path.relative_to(project_root) for path in project_root.rglob("*")}
    result = run_first_pass(
        FirstPassConfig(
            run_directory=run_directory,
            unity_url=unity_url,
            llama_port=port,
            api_key=api_key,
            feed_url=_ACTIVE_FEED_URL,
        ),
        server_lifecycle_factory=governed_transport,
    )

    assert isinstance(result, FirstPassResult)
    assert isinstance(result.release, ReportRelease)
    assert result.report_path.read_text(encoding="utf-8") == result.release.report_html
    assert result.summary_path == run_directory / "run-summary.json"
    assert events == [
        "polars",
        "deepvest",
        "tn",
        "runtime_config",
        "server_enter",
        "submit",
        "save_report",
        "save_summary",
        "server_exit",
    ]
    assert unity.get_paths == [
        "/volumes/deepvest.finance.deepvest_internal_positions",
        "/volumes/deepvest.finance.deepvest_market_prices",
    ]

    requests = handoff["requests"]
    assert len(requests) == 1
    assert requests[0]["path"] == "/v1/tn/chat/completions"
    assert requests[0]["authorization"] == f"Bearer {api_key}"
    request_body = json.loads(requests[0]["body"])
    flow = handoff["flow"]
    assert request_body["tn_row"] == dict(flow.source_row.envelope)
    assert "only the finance section" in request_body["messages"][0]["content"]
    assert (
        "DeepVest calculations in this first-pass report are deterministic stub outputs, "
        "not live DeepVest results."
    ) in request_body["messages"][0]["content"]

    server_spec = handoff["server_spec"]
    assert server_spec.build_directory == run_directory / "execution" / "governed-build"
    assert (server_spec.build_directory / "bin" / "llama.dll").read_bytes() == (
        b"test-companion"
    )
    runtime_document = json.loads(server_spec.runtime_config_path.read_text(encoding="utf-8"))
    assert runtime_document["binary_sha256"] == _sha256(
        server_spec.build_directory / "bin" / "llama-server.exe"
    )

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary == {
        "profile": "first_pass",
        "account_id": "acct-001",
        "unity": {
            "role": "governance_source",
            "base_url": unity_url,
            "resolved_feeds": list(FEED_NAMES),
        },
        "getdata": {
            "role": "business_data_source",
            "boundary": "local_http_seeded",
            "base_url": _ACTIVE_FEED_URL,
            "fetch_facts": [
                {
                    "event_name": "getdata.feed_fetched",
                    "feed_name": POSITIONS_FEED,
                    "request_path": f"/v1/feeds/{POSITIONS_FEED}/data",
                    "http_status": 200,
                    "content_type": "application/x-ndjson; charset=utf-8",
                },
                {
                    "event_name": "getdata.feed_fetched",
                    "feed_name": MARKET_PRICES_FEED,
                    "request_path": f"/v1/feeds/{MARKET_PRICES_FEED}/data",
                    "http_status": 200,
                    "content_type": "application/x-ndjson; charset=utf-8",
                },
            ],
        },
        "polars": {
            "admitted_schema": {
                "account_id": "String",
                "portfolio_id": "String",
                "symbol": "String",
                "quantity": "Int64",
                "cost_basis": "Float64",
                "currency": "String",
                "as_of": "String",
                "last_price": "Float64",
                "market_source": "String",
                "position_market_value": "Float64",
            },
            "column_sources": {
                "account_id": ["account_id"],
                "portfolio_id": ["portfolio_id"],
                "symbol": ["symbol"],
                "quantity": ["quantity"],
                "cost_basis": ["cost_basis"],
                "currency": ["currency"],
                "as_of": ["as_of"],
                "last_price": ["last_price"],
                "market_source": ["market_source"],
                "position_market_value": ["last_price", "quantity"],
            },
            "row_selection_sources": ["account_id", "currency", "symbol"],
        },
        "deepvest": {
            "provider": "stub",
            "live": False,
            "invocation_count": 1,
            "calculation_id": "stub-portfolio-risk-001",
            "analysis_type": "portfolio_risk",
            "metrics": [
                {"name": "value_at_risk_95", "value": 12500.0, "unit": "USD"},
                {"name": "expected_shortfall_95", "value": 16800.0, "unit": "USD"},
                {"name": "maximum_drawdown", "value": -0.18, "unit": "ratio"},
            ],
        },
        "narrative": {
            "mode": "fixed_template",
            "model_authored": False,
            "data_specific_generation_claimed": False,
        },
        "tn_source": {
            "event_type": "portfolio.deepvest.calculated",
            "event_id": flow.source_row.event_id,
            "row_hash": flow.source_row.row_hash,
            "encrypted_groups": ["finance", "tn.agents"],
        },
        "tn_llama": {
            "route": "/v1/tn/chat/completions",
            "response_status": 200,
            "recipient_open_groups": ["tn.agents", "default"],
        },
        "verified_output": {
            "event_type": "report.generated",
            "event_id": result.release.output_event_id,
            "row_hash": result.release.output_row_hash,
            "report_path": str(result.report_path),
        },
    }
    summary_text = result.summary_path.read_text(encoding="utf-8")
    for prohibited in (
        api_key,
        "portfolio_values",
        "deepvest_result",
        "controls_passed",
        "answer_statuses",
        '"q1"',
        "reader_key_dir",
        "index_key_path",
    ):
        assert prohibited not in summary_text
    assert result.report_path.stat().st_mtime_ns <= result.summary_path.stat().st_mtime_ns
    transcript = [
        json.loads(line)
        for line in (run_directory / "client" / "http.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [
        item["group"]
        for item in transcript
        if item.get("event") == "release_group_opened"
    ] == ["tn.agents", "default"]
    assert any(item.get("event") == "encrypted_release_verified" for item in transcript)
    assert all(
        name.lower() != "authorization"
        for item in transcript
        for name, _value in item.get("headers", [])
    )
    for artifact in run_directory.rglob("*"):
        if artifact.is_file():
            assert api_key.encode() not in artifact.read_bytes()
    after = {path.relative_to(project_root) for path in project_root.rglob("*")}
    assert all(
        path == Path("experiments") or path.is_relative_to(Path("experiments"))
        for path in after - before
    )


def test_first_pass_scopes_no_bytecode_and_tn_environment_then_restores_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    _install_fake_local_dependencies(monkeypatch, project_root)
    names = (
        "PYTHONDONTWRITEBYTECODE",
        "TN_NO_LINK",
        "TN_NO_STDOUT",
        "TN_AUTOINIT_QUIET",
    )
    original = {
        "PYTHONDONTWRITEBYTECODE": None,
        "TN_NO_LINK": "original-link",
        "TN_NO_STDOUT": "original-stdout",
        "TN_AUTOINIT_QUIET": "original-quiet",
    }
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    for name in names[1:]:
        monkeypatch.setenv(name, original[name])
    observed: list[dict[str, str | None]] = []

    def unavailable(_registry, _method: str, _path: str, payload=None):
        del payload
        observed.append({name: os.environ.get(name) for name in names})
        raise ConnectionError("deterministic Unity outage")

    monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", unavailable)

    with pytest.raises(ConnectionError, match="Unity outage"):
        run_first_pass(
            FirstPassConfig(
                run_directory=(project_root / "experiments" / "results" / "env-failure").resolve(),
                unity_url="http://127.0.0.1:18080",
                llama_port=_free_port(),
                api_key="test-local-key",
            ),
            server_lifecycle_factory=lambda _spec: (_ for _ in ()).throw(
                AssertionError("server must not start")
            ),
        )

    assert observed == [{name: "1" for name in names}]
    assert {name: os.environ.get(name) for name in names} == original


def test_unavailable_getdata_stops_before_deepvest_or_tn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
) -> None:
    _project_root, config = _first_pass_case(tmp_path, monkeypatch, name="getdata-denied")
    config = replace(config, feed_url="http://127.0.0.1:1")
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    calls: list[str] = []

    def unexpected(name: str):
        def denied(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not run after getdata denial")

        return denied

    monkeypatch.setattr(run_module, "execute_deepvest", unexpected("deepvest"))
    monkeypatch.setattr(run_module, "provision_tn_flow", unexpected("tn"))

    with pytest.raises(FeedFetchHTTPError):
        run_first_pass(config)

    assert calls == []


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    [
        ("cross_feed", SchemaPinViolation),
        ("malformed", Exception),
        ("schema", SchemaPinViolation),
        ("dtype", SchemaPinViolation),
        ("media", FeedFetchResponseError),
    ],
)
def test_pre_deepvest_getdata_denials_never_reach_deepvest_or_tn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
    failure: str,
    expected_error: type[Exception],
) -> None:
    import deepvest_tn.polars_stage as polars_stage

    _project_root, config = _first_pass_case(tmp_path, monkeypatch, name=f"getdata-{failure}")
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    real_fetch = polars_stage.fetch_feed
    calls: list[str] = []

    def fetch(getdata_config, feed_name: str) -> FetchedFeed:
        if failure == "media":
            raise FeedFetchResponseError("wrong media type")
        fetched = real_fetch(getdata_config, feed_name)
        if feed_name != POSITIONS_FEED:
            return fetched
        if failure == "cross_feed":
            replacement = real_fetch(getdata_config, MARKET_PRICES_FEED)
            return replace(fetched, body=replacement.body)
        if failure == "malformed":
            return replace(fetched, body=b"not-ndjson\n")
        if failure == "schema":
            rows = [json.loads(line) for line in fetched.body.splitlines()]
            altered = b"".join(
                json.dumps({**row, "unexpected": True}).encode("utf-8") + b"\n"
                for row in rows
            )
            return replace(fetched, body=altered)
        return replace(fetched, body=fetched.body.replace(b'"quantity":125', b'"quantity":"125"'))

    def forbidden(name: str):
        def boundary(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"{name} must not run after getdata denial")

        return boundary

    monkeypatch.setattr(polars_stage, "fetch_feed", fetch)
    monkeypatch.setattr(run_module, "execute_deepvest", forbidden("deepvest"))
    monkeypatch.setattr(run_module, "provision_tn_flow", forbidden("tn"))

    with pytest.raises(expected_error):
        run_first_pass(config)

    assert calls == []


@pytest.mark.parametrize("feed_url", ["https://127.0.0.1:18082", "http://example.test:18082"])
def test_first_pass_rejects_non_loopback_or_non_http_feed_url_before_any_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feed_url: str,
) -> None:
    _project_root, config = _first_pass_case(tmp_path, monkeypatch, name="invalid-feed-url")
    calls: list[str] = []
    monkeypatch.setattr(
        run_module,
        "run_governed_polars_stage",
        lambda *_args, **_kwargs: calls.append("polars"),
    )

    with pytest.raises(ValueError, match="Feed URL"):
        run_first_pass(replace(config, feed_url=feed_url))

    assert calls == []


def test_tn_root_link_is_refused_before_provisioning_or_outside_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
) -> None:
    _project_root, config = _first_pass_case(tmp_path, monkeypatch, name="tn-root-link")
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    outside = tmp_path / "outside-tn"
    outside.mkdir()
    real_deepvest = run_module.execute_deepvest
    provision_calls: list[object] = []

    def create_link_after_deepvest(*args, **kwargs):
        execution = real_deepvest(*args, **kwargs)
        try:
            (config.run_directory / "tn").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory symlinks unavailable: {error}")
        return execution

    def unexpected_provision(*_args, **_kwargs):
        provision_calls.append(object())
        raise AssertionError("Governed provisioning must not receive an escaped path")

    monkeypatch.setattr(run_module, "execute_deepvest", create_link_after_deepvest)
    monkeypatch.setattr(run_module, "provision_tn_flow", unexpected_provision)

    with pytest.raises(ValueError, match="outside the run directory"):
        run_first_pass(config)

    assert provision_calls == []
    assert list(outside.iterdir()) == []
    _assert_no_released_outputs(config.run_directory)


@pytest.mark.parametrize("failure", ("unavailable", "incomplete"))
def test_unity_failure_stops_before_deepvest_tn_server_and_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
    failure: str,
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name=f"unity-{failure}",
    )
    calls: list[str] = []
    if failure == "unavailable":
        def unavailable(_registry, _method: str, _path: str, payload=None):
            del payload
            calls.append("unity")
            raise ConnectionError("local Unity unavailable")

        monkeypatch.setattr("tn_polars.registry.UnityCatalogRegistry._request", unavailable)
        expected_error = ConnectionError
    else:
        copied = tmp_path / "incomplete-bundles"
        shutil.copytree(finance_bundles[POSITIONS_FEED], copied / POSITIONS_FEED)
        shutil.copytree(finance_bundles[MARKET_PRICES_FEED], copied / MARKET_PRICES_FEED)
        (copied / POSITIONS_FEED / "policy.wasm").unlink()
        transport = _install_unity_transport(
            monkeypatch,
            {
                POSITIONS_FEED: copied / POSITIONS_FEED,
                MARKET_PRICES_FEED: copied / MARKET_PRICES_FEED,
            },
            base_url=config.unity_url,
        )
        calls = transport.get_paths
        expected_error = PolicyDenied

    monkeypatch.setattr(
        run_module.StubDeepVest,
        "calculate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DeepVest must not run")
        ),
    )
    monkeypatch.setattr(
        run_module,
        "provision_tn_flow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("TN must not run")
        ),
    )

    with pytest.raises(expected_error):
        run_first_pass(
            config,
            server_lifecycle_factory=lambda _spec: (_ for _ in ()).throw(
                AssertionError("server must not start")
            ),
        )

    assert calls
    _assert_no_released_outputs(config.run_directory)


@pytest.mark.parametrize("failure", ("schema", "governance"))
def test_admission_or_governance_failure_stops_before_deepvest_and_tn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
    local_getdata_service: tuple[str, Path],
    failure: str,
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name=f"polars-{failure}",
    )
    bundles = finance_bundles
    expected_error: type[Exception]
    if failure == "schema":
        _feed_url, data_dir = local_getdata_service
        mutated_positions = data_dir / "internal_positions.ndjson"
        original_positions = mutated_positions.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in original_positions.splitlines()]
        rows[0]["unexpected"] = "drift"
        mutated_positions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        expected_error = SchemaPinViolation
    else:
        copied = tmp_path / "governance-bundles"
        shutil.copytree(finance_bundles[POSITIONS_FEED], copied / POSITIONS_FEED)
        shutil.copytree(finance_bundles[MARKET_PRICES_FEED], copied / MARKET_PRICES_FEED)
        governance_path = copied / MARKET_PRICES_FEED / "tn_governance.json"
        governance = json.loads(governance_path.read_text(encoding="utf-8"))
        governance["instruction"] = "incompatible governance mutation"
        governance_path.write_text(json.dumps(governance), encoding="utf-8")
        bundles = {
            POSITIONS_FEED: copied / POSITIONS_FEED,
            MARKET_PRICES_FEED: copied / MARKET_PRICES_FEED,
        }
        expected_error = ValueError
    _install_unity_transport(
        monkeypatch,
        bundles,
        base_url=config.unity_url,
    )
    calls: list[str] = []

    def forbidden_deepvest(*_args, **_kwargs):
        calls.append("deepvest")
        raise AssertionError("DeepVest must not run")

    def forbidden_tn(*_args, **_kwargs):
        calls.append("tn")
        raise AssertionError("TN must not run")

    monkeypatch.setattr(run_module.StubDeepVest, "calculate", forbidden_deepvest)
    monkeypatch.setattr(run_module, "provision_tn_flow", forbidden_tn)

    try:
        with pytest.raises(expected_error):
            run_first_pass(
                config,
                server_lifecycle_factory=lambda _spec: (_ for _ in ()).throw(
                    AssertionError("server must not start")
                ),
            )
    finally:
        if failure == "schema":
            mutated_positions.write_text(original_positions, encoding="utf-8")

    assert calls == []
    _assert_no_released_outputs(config.run_directory)


def test_deepvest_exception_is_attempted_once_and_stops_before_tn_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name="deepvest-failure",
    )
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    attempts: list[object] = []

    def raising_stub(_stub, request):
        attempts.append(request)
        raise RuntimeError("stub boundary failure")

    monkeypatch.setattr(run_module.StubDeepVest, "calculate", raising_stub)
    monkeypatch.setattr(
        run_module,
        "provision_tn_flow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("TN must not run")
        ),
    )

    with pytest.raises(RuntimeError, match="stub boundary failure"):
        run_first_pass(
            config,
            server_lifecycle_factory=lambda _spec: (_ for _ in ()).throw(
                AssertionError("server must not start")
            ),
        )

    assert len(attempts) == 1
    _assert_no_released_outputs(config.run_directory)


@pytest.mark.parametrize("failure", ("malformed-row", "missing-input-kit"))
def test_invalid_tn_handoff_stops_before_server_and_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
    failure: str,
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name=f"tn-{failure}",
    )
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    real_tn = run_module.provision_tn_flow
    server_calls: list[GovernedServerSpec] = []

    def invalid_handoff(*args, **kwargs):
        flow = real_tn(*args, **kwargs)
        if failure == "malformed-row":
            invalid_row = replace(
                flow.source_row,
                raw_ndjson=flow.source_row.raw_ndjson.removesuffix(b"\n"),
            )
            return replace(flow, source_row=invalid_row)
        (flow.input_reader.reader_key_dir / "finance.btn.mykit").unlink()
        return flow

    monkeypatch.setattr(run_module, "provision_tn_flow", invalid_handoff)

    @contextmanager
    def forbidden_server(spec: GovernedServerSpec):
        server_calls.append(spec)
        raise AssertionError("server must not start")
        yield

    with pytest.raises(ValueError):
        run_first_pass(config, server_lifecycle_factory=forbidden_server)

    assert server_calls == []
    assert not (config.run_directory / "client" / "http.ndjson").exists()
    _assert_no_released_outputs(config.run_directory)


def test_server_start_failure_saves_no_report_or_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name="server-start-failure",
    )
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    calls: list[GovernedServerSpec] = []

    @contextmanager
    def failed_start(spec: GovernedServerSpec):
        calls.append(spec)
        raise TimeoutError("governed child readiness failed")
        yield

    with pytest.raises(TimeoutError, match="readiness"):
        run_first_pass(config, server_lifecycle_factory=failed_start)

    assert len(calls) == 1
    assert not (config.run_directory / "client" / "http.ndjson").exists()
    _assert_no_released_outputs(config.run_directory)


@pytest.mark.parametrize(
    ("failure", "expected_error"),
    (
        ("route-denial", GovernedRequestDenied),
        ("malformed-release", ValueError),
        ("missing-output-kit", FileNotFoundError),
        ("wrong-output-kit", (ValueError, RuntimeError)),
        ("wrong-source-wrapper", ValueError),
    ),
)
def test_governed_or_output_denial_saves_no_report_or_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
    failure: str,
    expected_error,
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name=f"downstream-{failure}",
    )
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    real_tn = run_module.provision_tn_flow
    handoff: dict[str, object] = {}

    def observed_tn(*args, **kwargs):
        flow = real_tn(*args, **kwargs)
        handoff["flow"] = flow
        return flow

    monkeypatch.setattr(run_module, "provision_tn_flow", observed_tn)

    @contextmanager
    def denied_transport(spec: GovernedServerSpec) -> Iterator[None]:
        flow = handoff["flow"]
        status = 200
        if failure == "route-denial":
            body = b'{"error":{"code":"tn_aad_shape"}}'
            status = 400
        elif failure == "malformed-release":
            body = b'{"protocol":"tn.llama.release.v1"}'
        else:
            body = _canonical_release_body(flow)
            if failure == "wrong-source-wrapper":
                wrapper = json.loads(body)
                wrapper["source_event_id"] = "evt-not-the-submitted-source"
                body = json.dumps(wrapper, separators=(",", ":")).encode()
            elif failure == "missing-output-kit":
                (flow.output_writer.reader.reader_key_dir / "default.btn.mykit").unlink()
            elif failure == "wrong-output-kit":
                wrong = flow.input_reader.reader_key_dir / "finance.btn.mykit"
                target = flow.output_writer.reader.reader_key_dir / "default.btn.mykit"
                target.write_bytes(wrong.read_bytes())
        with _serve_response(port=spec.port, body=body, status=status) as requests:
            handoff["requests"] = requests
            yield

    with pytest.raises(expected_error):
        run_first_pass(config, server_lifecycle_factory=denied_transport)

    assert len(handoff["requests"]) == 1
    _assert_no_released_outputs(config.run_directory)
    if failure in {"missing-output-kit", "wrong-output-kit"}:
        assert (config.run_directory / "client" / "received-output.ndjson").read_bytes() == b""


@pytest.mark.parametrize("escaped_directory", ("client", "reports"))
def test_symlinked_output_directory_is_refused_before_http_or_outside_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
    escaped_directory: str,
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name=f"symlink-{escaped_directory}",
    )
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    real_tn = run_module.provision_tn_flow
    handoff: dict[str, object] = {}
    outside = tmp_path / f"outside-{escaped_directory}"
    outside.mkdir()

    def observed_tn(*args, **kwargs):
        flow = real_tn(*args, **kwargs)
        handoff["flow"] = flow
        return flow

    monkeypatch.setattr(run_module, "provision_tn_flow", observed_tn)

    @contextmanager
    def escaped_transport(spec: GovernedServerSpec) -> Iterator[None]:
        link = config.run_directory / escaped_directory
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory symlinks unavailable: {error}")
        body = _canonical_release_body(handoff["flow"])
        with _serve_response(port=spec.port, body=body) as requests:
            handoff["requests"] = requests
            yield

    with pytest.raises(ValueError, match="outside the run directory"):
        run_first_pass(config, server_lifecycle_factory=escaped_transport)

    assert handoff["requests"] == []
    assert list(outside.iterdir()) == []
    _assert_no_released_outputs(config.run_directory)


@pytest.mark.parametrize("escaped_directory", ("runtime", "execution"))
def test_symlinked_prelaunch_directory_is_refused_before_outside_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
    escaped_directory: str,
) -> None:
    _project_root, config = _first_pass_case(
        tmp_path,
        monkeypatch,
        name=f"prelaunch-symlink-{escaped_directory}",
    )
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)
    outside = tmp_path / f"outside-prelaunch-{escaped_directory}"
    outside.mkdir()

    if escaped_directory == "runtime":
        real_boundary = run_module.provision_tn_flow

        def observed_boundary(*args, **kwargs):
            result = real_boundary(*args, **kwargs)
            link = config.run_directory / escaped_directory
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                pytest.skip(f"directory symlinks unavailable: {error}")
            return result

        monkeypatch.setattr(run_module, "provision_tn_flow", observed_boundary)
    else:
        real_boundary = run_module.write_finance_runtime_config

        def observed_boundary(*args, **kwargs):
            result = real_boundary(*args, **kwargs)
            link = config.run_directory / escaped_directory
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                pytest.skip(f"directory symlinks unavailable: {error}")
            return result

        monkeypatch.setattr(run_module, "write_finance_runtime_config", observed_boundary)

    with pytest.raises(ValueError, match="outside the run directory"):
        run_first_pass(config)

    assert list(outside.iterdir()) == []
    _assert_no_released_outputs(config.run_directory)


def test_occupied_port_is_refused_after_runtime_config_and_before_build_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finance_bundles: dict[str, Path],
) -> None:
    _project_root, config = _first_pass_case(tmp_path, monkeypatch, name="occupied-port")
    _install_unity_transport(monkeypatch, finance_bundles, base_url=config.unity_url)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = replace(config, llama_port=int(listener.getsockname()[1]))

        with pytest.raises(RuntimeError, match="occupied"):
            run_first_pass(occupied)

    assert (config.run_directory / "runtime" / "runtime-config.json").is_file()
    assert not (config.run_directory / "execution").exists()
    _assert_no_released_outputs(config.run_directory)


def test_existing_run_directory_is_refused_before_any_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_module, "_PROJECT_ROOT", tmp_path.resolve())
    run_directory = tmp_path / "existing-run"
    run_directory.mkdir()
    config = FirstPassConfig(
        run_directory=run_directory.resolve(),
        unity_url="http://127.0.0.1:18080",
        llama_port=18180,
        api_key="test-local-key",
    )

    with pytest.raises(FileExistsError, match="run directory"):
        run_first_pass(config)


@pytest.mark.parametrize("kind", ("relative", "outside"))
def test_run_directory_must_resolve_beneath_the_project_without_creating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    project_root = (tmp_path / "project").resolve()
    project_root.mkdir()
    monkeypatch.setattr(run_module, "_PROJECT_ROOT", project_root)
    candidate = Path("relative-run") if kind == "relative" else (tmp_path / "outside-run")

    with pytest.raises(ValueError, match="absolute|beneath"):
        run_first_pass(
            FirstPassConfig(
                run_directory=candidate,
                unity_url="http://127.0.0.1:18080",
                llama_port=18180,
                api_key="test-local-key",
            )
        )

    assert not candidate.exists()


def test_run_directory_containment_resolves_a_symlinked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = (tmp_path / "project").resolve()
    outside = (tmp_path / "outside").resolve()
    project_root.mkdir()
    outside.mkdir()
    link = project_root / "linked-outside"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")
    monkeypatch.setattr(run_module, "_PROJECT_ROOT", project_root)
    candidate = link / "run"

    with pytest.raises(ValueError, match="beneath"):
        run_first_pass(
            FirstPassConfig(
                run_directory=candidate.absolute(),
                unity_url="http://127.0.0.1:18080",
                llama_port=18180,
                api_key="test-local-key",
            )
        )

    assert not (outside / "run").exists()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_governed_build_staging_copies_complete_bin_and_binds_runtime_hash(
    tmp_path: Path,
) -> None:
    source_build = tmp_path / "source-build"
    source_bin = source_build / "bin"
    source_bin.mkdir(parents=True)
    source_server = source_bin / "llama-server.exe"
    source_server.write_bytes(b"governed-server")
    (source_bin / "llama.dll").write_bytes(b"companion-dll")
    (source_bin / "nested").mkdir()
    (source_bin / "nested" / "backend.dll").write_bytes(b"backend")
    runtime_config = tmp_path / "run" / "runtime" / "runtime-config.json"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_text(
        json.dumps({"binary_sha256": _sha256(source_server)}),
        encoding="utf-8",
    )

    staged = run_module._stage_governed_build(
        source_build_directory=source_build,
        source_server_path=source_server,
        staged_build_directory=tmp_path / "run" / "execution" / "governed-build",
        runtime_config_path=runtime_config,
    )

    assert staged.server_path.read_bytes() == b"governed-server"
    assert (staged.build_directory / "bin" / "llama.dll").read_bytes() == b"companion-dll"
    assert (staged.build_directory / "bin" / "nested" / "backend.dll").read_bytes() == b"backend"
    assert staged.server_sha256 == _sha256(source_server)


def test_governed_build_staging_refuses_a_runtime_hash_not_matching_executable(
    tmp_path: Path,
) -> None:
    source_build = tmp_path / "source-build"
    source_bin = source_build / "bin"
    source_bin.mkdir(parents=True)
    source_server = source_bin / "llama-server.exe"
    source_server.write_bytes(b"governed-server")
    runtime_config = tmp_path / "runtime-config.json"
    runtime_config.write_text(json.dumps({"binary_sha256": "0" * 64}), encoding="utf-8")
    staged_build = tmp_path / "run" / "execution" / "governed-build"

    with pytest.raises(ValueError, match="runtime.*hash"):
        run_module._stage_governed_build(
            source_build_directory=source_build,
            source_server_path=source_server,
            staged_build_directory=staged_build,
            runtime_config_path=runtime_config,
        )

    assert not staged_build.exists()


class _FakeProcess:
    def __init__(self, pid: int = 43210) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = -1
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -1

    def kill(self) -> None:
        self.returncode = -1


class _TrackingProcess(_FakeProcess):
    def __init__(self, pid: int = 54321) -> None:
        super().__init__(pid)
        self.terminate_calls = 0
        self.kill_calls = 0

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _server_spec(tmp_path: Path, *, api_key: str = "sentinel-api-key") -> GovernedServerSpec:
    work = tmp_path / "run" / "process"
    work.parent.mkdir(parents=True)
    launcher = tmp_path / "run-tn-server.ps1"
    launcher.write_text("# fake launcher", encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    runtime_config = tmp_path / "run" / "runtime" / "runtime-config.json"
    runtime_config.parent.mkdir(parents=True)
    runtime_config.write_text("{}", encoding="utf-8")
    build = tmp_path / "run" / "execution" / "governed-build"
    (build / "bin").mkdir(parents=True)
    (build / "bin" / "llama-server.exe").write_bytes(b"server")
    return GovernedServerSpec(
        run_directory=(tmp_path / "run").resolve(),
        launcher_path=launcher.resolve(),
        model_path=model.resolve(),
        runtime_config_path=runtime_config.resolve(),
        build_directory=build.resolve(),
        working_directory=work.resolve(),
        stdout_path=(tmp_path / "run" / "server.stdout.log").resolve(),
        stderr_path=(tmp_path / "run" / "server.stderr.log").resolve(),
        port=18180,
        api_key=api_key,
        readiness_timeout_seconds=0.05,
    )


def test_server_child_uses_env_reference_authenticated_health_and_pid_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _server_spec(tmp_path)
    process = _FakeProcess()
    popen_calls: list[tuple[list[str], dict[str, object]]] = []
    cleanup_calls: list[list[str]] = []
    readiness_headers: list[dict[str, str]] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeProcess:
        popen_calls.append((command, kwargs))
        return process

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def open_ready(request, timeout: float):
        del timeout
        readiness_headers.append(dict(request.header_items()))
        return Response()

    def fake_run(command: list[str], **_kwargs: object):
        cleanup_calls.append(command)
        process.returncode = -1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(run_module.urllib.request, "urlopen", open_ready)
    monkeypatch.setattr(run_module.shutil, "which", lambda _name: "C:\\Tools\\pwsh.exe")

    with run_module._governed_server(spec):
        assert process.poll() is None

    assert len(popen_calls) == 1
    command, kwargs = popen_calls[0]
    joined = " ".join(command)
    assert "$env:TN_LLAMA_API_KEY" in joined
    assert spec.api_key not in joined
    assert str(spec.build_directory) in joined
    assert kwargs["cwd"] == str(spec.working_directory)
    assert kwargs["env"]["TN_LLAMA_API_KEY"] == spec.api_key
    assert readiness_headers == [{"Authorization": f"Bearer {spec.api_key}"}]
    assert cleanup_calls == [["taskkill", "/PID", str(process.pid), "/T", "/F"]]
    for path in (spec.stdout_path, spec.stderr_path):
        assert spec.api_key.encode() not in path.read_bytes()


def test_readiness_failure_cleans_up_only_the_owned_process_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _server_spec(tmp_path)
    process = _FakeProcess(pid=9876)
    cleanup_calls: list[list[str]] = []

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_kwargs: (
            cleanup_calls.append(command),
            setattr(process, "returncode", -1),
            SimpleNamespace(returncode=0),
        )[-1],
    )
    monkeypatch.setattr(
        run_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("not ready")),
    )
    monkeypatch.setattr(run_module.shutil, "which", lambda _name: "C:\\Tools\\pwsh.exe")
    monkeypatch.setattr(run_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match="health"), run_module._governed_server(spec):
        raise AssertionError("unready server must not enter the context")

    assert cleanup_calls == [["taskkill", "/PID", "9876", "/T", "/F"]]


def test_fake_health_success_is_rejected_if_the_owned_child_has_exited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _server_spec(tmp_path)
    process = _FakeProcess(pid=2468)

    class Response:
        status = 200

        def __enter__(self):
            process.returncode = 17
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(run_module.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(run_module.shutil, "which", lambda _name: "C:\\Tools\\pwsh.exe")

    with pytest.raises(RuntimeError, match="exited"), run_module._governed_server(spec):
        raise AssertionError("dead server must not enter the context")


@pytest.mark.parametrize("taskkill_failure", ("timeout", "oserror", "nonzero"))
def test_owned_cleanup_falls_back_after_taskkill_failure_and_verifies_port(
    monkeypatch: pytest.MonkeyPatch,
    taskkill_failure: str,
) -> None:
    process = _TrackingProcess()
    port = _free_port()
    taskkill_calls: list[list[str]] = []

    def fail_taskkill(command: list[str], **_kwargs: object):
        taskkill_calls.append(command)
        if taskkill_failure == "timeout":
            raise subprocess.TimeoutExpired(command, 15)
        if taskkill_failure == "oserror":
            raise OSError("taskkill unavailable")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", fail_taskkill)

    run_module._stop_owned_process_tree(process, port=port)

    assert taskkill_calls == [["taskkill", "/PID", str(process.pid), "/T", "/F"]]
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.poll() is not None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rebound:
        rebound.bind(("127.0.0.1", port))


def test_owned_cleanup_reports_if_exact_port_remains_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _TrackingProcess()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (
            setattr(process, "returncode", -1),
            SimpleNamespace(returncode=0),
        )[-1],
    )
    monkeypatch.setattr(run_module, "_CLEANUP_PORT_TIMEOUT_SECONDS", 0.01, raising=False)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
        with pytest.raises(RuntimeError, match="port.*remains bound"):
            run_module._stop_owned_process_tree(process, port=port)


def test_owned_cleanup_reports_if_launcher_cannot_be_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StickyProcess(_TrackingProcess):
        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

    process = StickyProcess()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError, match="launcher.*still running"):
        run_module._stop_owned_process_tree(process, port=_free_port())

    assert process.terminate_calls == 1
    assert process.kill_calls == 1


def test_server_lifecycle_refuses_generated_path_outside_owned_run_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = replace(_server_spec(tmp_path), stdout_path=tmp_path / "outside.log")
    popen_calls: list[object] = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: popen_calls.append(object()),
    )

    with pytest.raises(
        ValueError, match="outside the run directory"
    ), run_module._governed_server(spec):
        raise AssertionError("escaped output path must not start a process")

    assert popen_calls == []


def test_server_lifecycle_refuses_an_occupied_port_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _server_spec(tmp_path)
    popen_calls: list[object] = []
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: popen_calls.append(object()),
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        occupied_port = int(listener.getsockname()[1])
        occupied_spec = replace(spec, port=occupied_port)

        with pytest.raises(
            RuntimeError, match="occupied"
        ), run_module._governed_server(occupied_spec):
            raise AssertionError("occupied port must not start a process")

    assert popen_calls == []


def test_cli_uses_env_only_key_real_lifecycle_and_restores_containment_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "cli-key-must-not-print"
    monkeypatch.setenv("TN_LLAMA_API_KEY", sentinel)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "previous-bytecode")
    monkeypatch.setenv("TN_NO_LINK", "previous-link")
    monkeypatch.delenv("TN_NO_STDOUT", raising=False)
    monkeypatch.setenv("TN_AUTOINIT_QUIET", "previous-quiet")
    observed: list[FirstPassConfig] = []
    report_path = tmp_path / "run" / "reports" / "event.html"
    summary_path = tmp_path / "run" / "run-summary.json"

    def fake_run(config: FirstPassConfig) -> object:
        observed.append(config)
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
        assert os.environ["TN_NO_LINK"] == "1"
        assert os.environ["TN_NO_STDOUT"] == "1"
        assert os.environ["TN_AUTOINIT_QUIET"] == "1"
        return SimpleNamespace(report_path=report_path, summary_path=summary_path)

    monkeypatch.setattr(run_module, "run_first_pass", fake_run)

    exit_code = cli_module.main(
        [
            "first-pass",
            "--run-directory",
            str((tmp_path / "run").resolve()),
            "--feed-url",
            "http://127.0.0.1:19082",
            "--llama-port",
            "18182",
        ]
    )

    assert exit_code == 0
    assert observed == [
        FirstPassConfig(
            run_directory=(tmp_path / "run").resolve(),
            unity_url="http://127.0.0.1:18080",
            llama_port=18182,
            api_key=sentinel,
            feed_url="http://127.0.0.1:19082",
        )
    ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "previous-bytecode"
    assert os.environ["TN_NO_LINK"] == "previous-link"
    assert "TN_NO_STDOUT" not in os.environ
    assert os.environ["TN_AUTOINIT_QUIET"] == "previous-quiet"
    output = capsys.readouterr()
    assert str(report_path) in output.out
    assert str(summary_path) in output.out
    assert sentinel not in output.out
    assert sentinel not in output.err


def test_cli_reference_path_selects_the_tn_object_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TN_LLAMA_API_KEY", "reference-key")
    observed: list[ReferenceRunConfig] = []

    def fake_run(config: ReferenceRunConfig) -> object:
        observed.append(config)
        return SimpleNamespace(
            report_path=tmp_path / "report.html",
            summary_path=tmp_path / "summary.json",
        )

    monkeypatch.setattr(run_module, "run_reference_path", fake_run)
    assert cli_module.main(
        [
            "reference-path",
            "--run-directory",
            str((tmp_path / "run").resolve()),
            "--llama-port",
            "18182",
        ]
    ) == 0
    assert observed == [
        ReferenceRunConfig(
            run_directory=(tmp_path / "run").resolve(),
            unity_url="http://127.0.0.1:18080",
            llama_port=18182,
            api_key="reference-key",
        )
    ]


def test_cli_refuses_missing_key_and_has_no_api_key_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("TN_LLAMA_API_KEY", raising=False)
    common = [
        "first-pass",
        "--run-directory",
        str((tmp_path / "run").resolve()),
        "--unity-url",
        "http://127.0.0.1:18080",
        "--llama-port",
        "18182",
    ]

    with pytest.raises(SystemExit):
        cli_module.main(common)
    missing_key = capsys.readouterr()
    assert "TN_LLAMA_API_KEY" in missing_key.err

    with pytest.raises(SystemExit):
        cli_module.main([*common, "--api-key", "must-not-be-accepted"])
    rejected_argument = capsys.readouterr()
    assert "--api-key" in rejected_argument.err


def test_powershell_wrapper_uses_project_command_and_restores_absent_environment(
    tmp_path: Path,
) -> None:
    real_project = Path(__file__).resolve().parents[1]
    real_wrapper = real_project / "scripts" / "run-first-pass.ps1"
    real_command = Path(sys.executable).parent / "deepvest-tn.exe"
    assert real_command.is_file()

    workspace = tmp_path / "workspace"
    project = workspace / "application"
    wrapper = project / "scripts" / "run-first-pass.ps1"
    project_command = project / ".venv" / "Scripts" / "deepvest-tn.exe"
    wrapper.parent.mkdir(parents=True)
    project_command.parent.mkdir(parents=True)
    (project / "dependencies" / "tn-polars").mkdir(parents=True)
    (project / "dependencies" / "tn-llama-client").mkdir(parents=True)
    shutil.copy2(real_wrapper, wrapper)
    shutil.copy2(real_command, project_command)

    shim_root = project / "src"
    shim_package = shim_root / "deepvest_tn"
    shim_package.mkdir(parents=True)
    (shim_package / "__init__.py").write_text("", encoding="utf-8")
    (shim_package / "cli.py").write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "def main():\n"
        "    names = (\n"
        "        'PYTHONDONTWRITEBYTECODE', 'TN_NO_LINK',\n"
        "        'TN_NO_STDOUT', 'TN_AUTOINIT_QUIET',\n"
        "    )\n"
        "    document = {\n"
        "        'arguments': sys.argv[1:],\n"
        "        'environment': {\n"
        "            name: {'present': name in os.environ, 'value': os.environ.get(name)}\n"
        "            for name in names\n"
        "        },\n"
        "        'key_match': os.environ.get('TN_LLAMA_API_KEY')\n"
        "        == os.environ.get('WRAPPER_TEST_EXPECTED_KEY'),\n"
        "    }\n"
        "    Path(os.environ['WRAPPER_TEST_CAPTURE']).write_text(\n"
        "        json.dumps(document), encoding='utf-8'\n"
        "    )\n"
        "    return int(os.environ.get('WRAPPER_TEST_EXIT_CODE', '0'))\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    path_marker = tmp_path / "path-command-ran.txt"
    path_secret = tmp_path / "path-command-secret.txt"
    fake_command = fake_bin / "deepvest-tn.cmd"
    fake_command.write_text(
        "@echo off\n"
        "> \"%WRAPPER_TEST_PATH_MARKER%\" echo RAN\n"
        "> \"%WRAPPER_TEST_PATH_SECRET%\" echo %TN_LLAMA_API_KEY%\n"
        "exit /b 0\n",
        encoding="utf-8",
    )

    success_capture = tmp_path / "success-child.json"
    failure_capture = tmp_path / "failure-child.json"
    state_capture = tmp_path / "wrapper-state.json"
    harness = tmp_path / "wrapper-harness.ps1"
    harness.write_text(
        "param(\n"
        "    [string]$Wrapper, [string]$RunDirectory,\n"
        "    [string]$SuccessCapture, [string]$FailureCapture,\n"
        "    [string]$StateCapture\n"
        ")\n"
        "$ErrorActionPreference = 'Stop'\n"
        "$names = @(\n"
        "    'PYTHONDONTWRITEBYTECODE', 'TN_NO_LINK',\n"
        "    'TN_NO_STDOUT', 'TN_AUTOINIT_QUIET'\n"
        ")\n"
        "function Clear-Containment {\n"
        "    foreach ($name in $names) {\n"
        "        Remove-Item -LiteralPath \"Env:$name\" -ErrorAction SilentlyContinue\n"
        "    }\n"
        "}\n"
        "function Get-Presence {\n"
        "    $result = @{}\n"
        "    foreach ($name in $names) {\n"
        "        $result[$name] = Test-Path -LiteralPath \"Env:$name\"\n"
        "    }\n"
        "    return $result\n"
        "}\n"
        "Clear-Containment\n"
        "$env:WRAPPER_TEST_CAPTURE = $SuccessCapture\n"
        "$env:WRAPPER_TEST_EXIT_CODE = '0'\n"
        "& $Wrapper -RunDirectory $RunDirectory "
        "-FeedUrl 'http://127.0.0.1:19082' -LlamaPort 18182\n"
        "$success = Get-Presence\n"
        "Clear-Containment\n"
        "$env:WRAPPER_TEST_CAPTURE = $FailureCapture\n"
        "$env:WRAPPER_TEST_EXIT_CODE = '9'\n"
        "$failureCaught = $false\n"
        "try {\n"
        "    & $Wrapper -RunDirectory \"$RunDirectory-failure\" "
        "-FeedUrl 'http://127.0.0.1:19082' -LlamaPort 18182\n"
        "} catch {\n"
        "    $failureCaught = $true\n"
        "}\n"
        "$failure = Get-Presence\n"
        "@{\n"
        "    success = $success\n"
        "    failure = $failure\n"
        "    failure_caught = $failureCaught\n"
        "} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StateCapture\n"
        "if (-not $failureCaught) { exit 17 }\n",
        encoding="utf-8",
    )

    sentinel = "wrapper-key-must-not-print"
    run_directory = (tmp_path / "owned-run").resolve()
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "PYTHONPATH": environment.get("PYTHONPATH", ""),
            "WRAPPER_TEST_EXPECTED_KEY": sentinel,
            "WRAPPER_TEST_PATH_MARKER": str(path_marker),
            "WRAPPER_TEST_PATH_SECRET": str(path_secret),
            "TN_LLAMA_API_KEY": sentinel,
        }
    )

    completed = subprocess.run(
        [
            "pwsh",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(harness),
            "-Wrapper",
            str(wrapper),
            "-RunDirectory",
            str(run_directory),
            "-SuccessCapture",
            str(success_capture),
            "-FailureCapture",
            str(failure_capture),
            "-StateCapture",
            str(state_capture),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    expected_environment = {
        name: {"present": True, "value": "1"}
        for name in (
            "PYTHONDONTWRITEBYTECODE",
            "TN_NO_LINK",
            "TN_NO_STDOUT",
            "TN_AUTOINIT_QUIET",
        )
    }
    for capture, expected_run in (
        (success_capture, run_directory),
        (failure_capture, Path(f"{run_directory}-failure")),
    ):
        observed = json.loads(capture.read_text(encoding="utf-8"))
        assert observed == {
            "arguments": [
                "first-pass",
                "--run-directory",
                str(expected_run),
                "--unity-url",
                "http://127.0.0.1:18080",
                "--feed-url",
                "http://127.0.0.1:19082",
                "--llama-port",
                "18182",
            ],
            "environment": expected_environment,
            "key_match": True,
        }
        assert sentinel not in capture.read_text(encoding="utf-8")
    state = json.loads(state_capture.read_text(encoding="utf-8"))
    assert state == {
        "success": {name: False for name in expected_environment},
        "failure": {name: False for name in expected_environment},
        "failure_caught": True,
    }
    assert not path_marker.exists()
    assert not path_secret.exists()
    assert sentinel not in completed.stdout
    assert sentinel not in completed.stderr


@pytest.fixture()
def getdata_launcher_project(tmp_path: Path) -> Path:
    project = tmp_path / "application"
    script = project / "scripts" / "start-local-getdata.ps1"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / script.name
    shutil.copy2(source, script)
    venv.EnvBuilder(with_pip=False).create(project / ".venv")
    return project


def test_start_local_getdata_serves_data_and_refuses_outside_or_occupied_launches(
    tmp_path: Path, getdata_launcher_project: Path,
) -> None:
    project_root = getdata_launcher_project
    script = project_root / "scripts" / "start-local-getdata.ps1"
    port = _free_port()
    data_dir = project_root / ".runtime" / "getdata" / f"pytest-{uuid.uuid4().hex}"
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    pid = None
    try:
        started = subprocess.run(
            [
                "pwsh", "-NoProfile", "-File", str(script), "-Port", str(port),
                "-DataDirectory", str(data_dir),
            ],
            check=False, capture_output=True, text=True, env=environment, timeout=20,
        )
        assert started.returncode == 0, started.stderr
        pid = int(started.stdout.split("PID: ", 1)[1].split(";", 1)[0])
        assert f"base URL: http://127.0.0.1:{port}" in started.stdout
        assert f"data directory: {data_dir}" in started.stdout
        data_url = f"http://127.0.0.1:{port}/v1/feeds/{POSITIONS_FEED}/data"
        with urllib.request.urlopen(data_url) as response:
            assert response.status == 200
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=True, capture_output=True)
        pid = None
        assert run_module._loopback_port_is_bindable(port)

        outside = tmp_path / "outside"
        rejected = subprocess.run(
            [
                "pwsh", "-NoProfile", "-File", str(script), "-Port", str(_free_port()),
                "-DataDirectory", str(outside),
            ],
            check=False, capture_output=True, text=True, env=environment, timeout=20,
        )
        assert rejected.returncode != 0
        assert "beneath this project" in rejected.stderr
        assert not outside.exists()

        occupied = _free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", occupied))
            occupied_data = data_dir.with_name(f"occupied-{uuid.uuid4().hex}")
            blocked = subprocess.run(
                [
                    "pwsh", "-NoProfile", "-File", str(script), "-Port", str(occupied),
                    "-DataDirectory", str(occupied_data),
                ],
                check=False, capture_output=True, text=True, env=environment, timeout=20,
            )
        assert blocked.returncode != 0
        assert "already occupied" in blocked.stderr
    finally:
        if pid is not None:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        if data_dir.exists():
            shutil.rmtree(data_dir)
        for suffix in (".stdout.log", ".stderr.log"):
            Path(f"{data_dir}{suffix}").unlink(missing_ok=True)


def test_start_local_getdata_kills_an_unready_owned_child_and_releases_its_port(
    tmp_path: Path, getdata_launcher_project: Path,
) -> None:
    project_root = getdata_launcher_project
    script = project_root / "scripts" / "start-local-getdata.ps1"
    port = _free_port()
    fake_root = tmp_path / "fake-package" / "deepvest_tn"
    fake_root.mkdir(parents=True)
    (fake_root / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / "feed_server.py").write_text(
        "import socket\nimport sys\nimport time\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "listener = socket.socket()\nlistener.bind(('127.0.0.1', port))\n"
        "listener.listen()\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    data_dir = project_root / ".runtime" / "getdata" / f"pytest-unready-{uuid.uuid4().hex}"
    environment = dict(os.environ)
    previous_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(fake_root.parent) + os.pathsep + previous_python_path
    try:
        completed = subprocess.run(
            [
                "pwsh", "-NoProfile", "-File", str(script), "-TimeoutSeconds", "5", "-Port",
                str(port), "-DataDirectory", str(data_dir),
            ],
            check=False, capture_output=True, text=True, env=environment, timeout=20,
        )
        assert completed.returncode != 0
        assert "did not become ready" in completed.stderr
        assert run_module._loopback_port_is_bindable(port)
    finally:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        for suffix in (".stdout.log", ".stderr.log"):
            Path(f"{data_dir}{suffix}").unlink(missing_ok=True)


def test_start_local_getdata_rejects_a_project_junction_that_targets_outside(
    tmp_path: Path, getdata_launcher_project: Path,
) -> None:
    project_root = getdata_launcher_project
    script = project_root / "scripts" / "start-local-getdata.ps1"
    outside = tmp_path / "outside"
    outside.mkdir()
    junction = project_root / ".runtime" / "getdata" / f"pytest-junction-{uuid.uuid4().hex}"
    junction.parent.mkdir(parents=True, exist_ok=True)
    try:
        created = subprocess.run(
            [
                "pwsh", "-NoProfile", "-Command",
                f"New-Item -ItemType Junction -Path '{junction}' -Target '{outside}' | Out-Null",
            ],
            check=False, capture_output=True, text=True, timeout=20,
        )
        if created.returncode != 0:
            pytest.skip(f"junction creation unavailable: {created.stderr}")
        attempted = subprocess.run(
            [
                "pwsh", "-NoProfile", "-File", str(script), "-Port", str(_free_port()),
                "-DataDirectory", str(junction / "fresh-child"),
            ],
            check=False, capture_output=True, text=True, timeout=20,
        )
        assert attempted.returncode != 0
        assert "beneath this project" in attempted.stderr
        assert list(outside.iterdir()) == []
    finally:
        if junction.exists():
            subprocess.run(
                ["pwsh", "-NoProfile", "-Command", f"Remove-Item -LiteralPath '{junction}' -Force"],
                check=False, capture_output=True, text=True,
            )
        shutil.rmtree(outside)
