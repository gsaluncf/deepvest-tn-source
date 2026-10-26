"""One-command composition of the DeepVest--TN workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from tn_llama_client.artifacts import save_report_html
from tn_llama_client.protocol import ReportRelease

from .deepvest import StubDeepVest, execute_deepvest
from .feed_server import FeedServerConfig, create_feed_server, seed_finance_feed_data
from .llama import submit_finance_report, write_finance_runtime_config
from .pipeline import ProcessingPathResult, run_processing_path
from .polars_stage import GovernedPortfolioResult, run_governed_polars_stage
from .tn_flow import (
    TNFlowResult,
    build_reference_flow,
    provision_reference_path,
    provision_tn_flow,
)
from .unity import UnityCatalogConfig, publish_finance_feeds
from .witness import write_witness

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_ROOT = _PROJECT_ROOT / "fixtures" / "portfolio"
_SOURCE_AGENTS_PATH = _FIXTURE_ROOT / "agents.policy"
_OUTPUT_AGENTS_PATH = _FIXTURE_ROOT / "report-agents.policy"
_TN_LLAMA_ROOT = Path(os.environ.get("TN_LLAMA_ROOT", str(_PROJECT_ROOT.parent / "tn-llama")))
_LLAMA_BUILD_DIRECTORY = _TN_LLAMA_ROOT / "build" / "llama.cpp" / "cpu-tn"
_GOVERNED_SERVER_PATH = _LLAMA_BUILD_DIRECTORY / "bin" / "llama-server.exe"
_LLAMA_MODEL_PATH = _TN_LLAMA_ROOT / "models" / "Qwen3.5-0.8B-BF16.gguf"
_LLAMA_LAUNCHER_PATH = _TN_LLAMA_ROOT / "scripts" / "llama" / "run-tn-server.ps1"
_ACCOUNT_ID = "acct-001"
_GOVERNED_ROUTE = "/v1/tn/chat/completions"
_STUB_DISCLOSURE = (
    "DeepVest calculations in this first-pass report are deterministic stub outputs, "
    "not live DeepVest results."
)
# The installed 0.8B model is used here to exercise the governed emit path, not to support a
# data-specific narrative-generation claim. A stronger configured model is required for that.
_FIXED_NARRATIVE_PARAGRAPH = (
    "This first-pass report demonstrates governed transport of supplied portfolio positions "
    "and provided DeepVest stub risk metrics without recalculation; it does not claim "
    "data-specific narrative generation."
)
_REPORT_PROMPT = (
    "Use only the finance section. This plumbing-first profile does not ask the model to perform "
    "a financial calculation or author data-specific analysis. Copy the following HTML exactly "
    "into report_html:\n"
    "<html><head><title>Portfolio Report</title></head><body>\n"
    "<h1>Portfolio Report</h1>\n"
    f"<p>{_FIXED_NARRATIVE_PARAGRAPH}</p>\n"
    f"<p>{_STUB_DISCLOSURE}</p>\n"
    "</body></html>\n"
    "Do not alter or omit the final disclosure paragraph. Do not call emit_governed_record "
    "until report_html ends with that exact final disclosure paragraph."
)
_CLEANUP_PORT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class FirstPassConfig:
    """User-selectable inputs for the fixed portfolio demonstration profile."""

    run_directory: Path
    unity_url: str
    llama_port: int
    api_key: str = field(repr=False)
    feed_url: str = "http://127.0.0.1:18082"


@dataclass(frozen=True, slots=True)
class ReferenceRunConfig:
    """Inputs for the TN-object reference path."""

    run_directory: Path
    unity_url: str
    llama_port: int
    api_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class StagedGovernedBuild:
    """Run-scoped copy of the governed server execution files."""

    build_directory: Path
    server_path: Path
    server_sha256: str


@dataclass(frozen=True, slots=True)
class GovernedServerSpec:
    """Exact launcher invocation owned by one portfolio demonstration run."""

    run_directory: Path
    launcher_path: Path
    model_path: Path
    runtime_config_path: Path
    build_directory: Path
    working_directory: Path
    stdout_path: Path
    stderr_path: Path
    port: int
    api_key: str = field(repr=False)
    readiness_timeout_seconds: float = 120.0


@dataclass(frozen=True, slots=True)
class FirstPassFacts:
    """Non-secret boundary facts returned beside the verified release."""

    profile: str
    account_id: str
    feed_names: tuple[str, str]
    stub_invocation_count: int
    source_event_id: str
    output_event_id: str
    response_status: int
    recipient_open_groups: tuple[str, str]


@dataclass(frozen=True, slots=True)
class FirstPassResult:
    """Verified report and factual artifacts from one portfolio demonstration execution."""

    release: ReportRelease
    report_path: Path
    summary_path: Path
    facts: FirstPassFacts
    witness_path: Path | None = None


def _validate_new_run_directory(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("run directory must be absolute")
    resolved = candidate.resolve(strict=False)
    project_root = _PROJECT_ROOT.resolve()
    if resolved == project_root or not resolved.is_relative_to(project_root):
        raise ValueError("run directory must be beneath this project")
    if resolved.exists():
        raise FileExistsError(f"run directory already exists: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _require_run_owned_path(
    run_directory: Path,
    path: Path,
    *,
    label: str,
) -> Path:
    """Resolve links/junctions and require a generated path to stay in its run."""
    run_root = Path(run_directory).resolve(strict=True)
    resolved = Path(path).resolve(strict=False)
    if resolved == run_root or not resolved.is_relative_to(run_root):
        raise ValueError(f"{label} resolves outside the run directory")
    return resolved


def _stage_governed_build(
    *,
    source_build_directory: Path,
    source_server_path: Path,
    staged_build_directory: Path,
    runtime_config_path: Path,
) -> StagedGovernedBuild:
    """Copy the complete governed bin directory and bind it to the runtime hash."""
    source_build = Path(source_build_directory).resolve()
    source_bin = source_build / "bin"
    source_server = Path(source_server_path).resolve()
    runtime_config = Path(runtime_config_path).resolve()
    staged_build = Path(staged_build_directory).resolve(strict=False)
    if not source_bin.is_dir():
        raise FileNotFoundError(f"governed build bin directory is missing: {source_bin}")
    if not source_server.is_file() or not source_server.is_relative_to(source_bin):
        raise FileNotFoundError(
            f"governed server must be an existing member of the build bin: {source_server}"
        )
    if not runtime_config.is_file():
        raise FileNotFoundError(f"tn-llama runtime config is missing: {runtime_config}")
    if staged_build.exists():
        raise FileExistsError(f"staged governed build already exists: {staged_build}")
    try:
        runtime_document = json.loads(runtime_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("runtime config is not readable JSON") from error
    runtime_hash = runtime_document.get("binary_sha256")
    source_hash = _sha256_file(source_server)
    if runtime_hash != source_hash:
        raise ValueError("runtime binary hash does not match the governed executable")

    relative_server = source_server.relative_to(source_bin)
    shutil.copytree(source_bin, staged_build / "bin")
    staged_server = staged_build / "bin" / relative_server
    staged_hash = _sha256_file(staged_server)
    if staged_hash != source_hash or staged_hash != runtime_hash:
        raise ValueError("staged governed executable hash differs from the runtime config")
    return StagedGovernedBuild(staged_build, staged_server, staged_hash)


def _powershell_literal(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _launcher_expression(spec: GovernedServerSpec) -> str:
    return " ".join(
        (
            "&",
            _powershell_literal(spec.launcher_path),
            "-ModelPath",
            _powershell_literal(spec.model_path),
            "-RuntimeConfigPath",
            _powershell_literal(spec.runtime_config_path),
            "-ApiKey",
            "$env:TN_LLAMA_API_KEY",
            "-Port",
            str(spec.port),
            "-BuildDirectory",
            _powershell_literal(spec.build_directory),
        )
    )


def _preflight_loopback_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("llama port must be between 1 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(
                f"loopback port {port} is occupied; refusing to attach to another process"
            ) from error


def _loopback_port_is_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _wait_for_authenticated_health(
    process: subprocess.Popen[bytes],
    spec: GovernedServerSpec,
) -> None:
    deadline = time.monotonic() + spec.readiness_timeout_seconds
    health_url = f"http://127.0.0.1:{spec.port}/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"governed tn-llama child {process.pid} exited with code {return_code}"
            )
        request = urllib.request.Request(
            health_url,
            headers={"Authorization": f"Bearer {spec.api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return_code = process.poll()
                    if return_code is not None:
                        raise RuntimeError(
                            f"governed tn-llama child {process.pid} exited with code "
                            f"{return_code} after health response"
                        )
                    return
        except urllib.error.URLError:
            pass
        time.sleep(0.25)
    raise TimeoutError(
        f"governed tn-llama health did not become ready within "
        f"{spec.readiness_timeout_seconds:g} seconds"
    )


def _wait_for_owned_process(process: subprocess.Popen[bytes], timeout: float) -> None:
    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return


def _wait_for_loopback_port_release(port: int) -> bool:
    deadline = time.monotonic() + _CLEANUP_PORT_TIMEOUT_SECONDS
    while True:
        if _loopback_port_is_bindable(port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _stop_owned_process_tree(process: subprocess.Popen[bytes], *, port: int) -> None:
    """Stop only the spawned PID tree and prove its launcher/port are gone."""
    if process.poll() is None:
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (subprocess.TimeoutExpired, OSError):
            completed = None
        if completed is not None and completed.returncode == 0:
            _wait_for_owned_process(process, 10)

    if process.poll() is None:
        with suppress(OSError):
            process.terminate()
        _wait_for_owned_process(process, 5)
    if process.poll() is None:
        with suppress(OSError):
            process.kill()
        _wait_for_owned_process(process, 5)

    launcher_running = process.poll() is None
    port_released = _wait_for_loopback_port_release(port)
    failures = []
    if launcher_running:
        failures.append(f"owned launcher PID {process.pid} is still running")
    if not port_released:
        failures.append(f"loopback port {port} remains bound")
    if failures:
        raise RuntimeError("owned process cleanup failed: " + "; ".join(failures))


@contextmanager
def _governed_server(spec: GovernedServerSpec) -> Iterator[None]:
    """Start, authenticate, and PID-scope cleanup of the model-server launcher."""
    for label, path in (
        ("runtime config", spec.runtime_config_path),
        ("staged governed build", spec.build_directory),
        ("server working directory", spec.working_directory),
        ("server stdout", spec.stdout_path),
        ("server stderr", spec.stderr_path),
    ):
        _require_run_owned_path(spec.run_directory, path, label=label)
    _preflight_loopback_port(spec.port)
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        raise FileNotFoundError("PowerShell 7 executable 'pwsh' was not found")
    for label, path in (
        ("tn-llama launcher", spec.launcher_path),
        ("GGUF model", spec.model_path),
        ("runtime config", spec.runtime_config_path),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if not spec.build_directory.is_dir():
        raise FileNotFoundError(f"staged governed build is missing: {spec.build_directory}")
    if not spec.api_key.strip():
        raise ValueError("TN_LLAMA_API_KEY must be nonempty")

    spec.working_directory.mkdir(parents=True, exist_ok=False)
    spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    spec.stderr_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["TN_LLAMA_API_KEY"] = spec.api_key
    command = [
        pwsh,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        _launcher_expression(spec),
    ]
    process: subprocess.Popen[bytes] | None = None
    with (
        spec.stdout_path.open("xb") as stdout,
        spec.stderr_path.open("xb") as stderr,
    ):
        try:
            process = subprocess.Popen(
                command,
                cwd=str(spec.working_directory),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                ),
            )
            _wait_for_authenticated_health(process, spec)
            yield
        finally:
            if process is not None:
                _stop_owned_process_tree(process, port=spec.port)


def _validate_loopback_http_base_url(value: str, *, label: str) -> str:
    parsed = urllib.parse.urlparse(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{label} must be an HTTP loopback base URL") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or port is None
        or not 1 <= port <= 65535
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be an HTTP loopback base URL")
    return value.rstrip("/")


def _validate_first_pass_config(config: FirstPassConfig) -> tuple[Path, str, str]:
    run_directory = _validate_new_run_directory(config.run_directory)
    if not config.api_key.strip():
        raise ValueError("TN_LLAMA_API_KEY must be nonempty")
    if not 1 <= config.llama_port <= 65535:
        raise ValueError("llama port must be between 1 and 65535")
    unity_url = _validate_loopback_http_base_url(config.unity_url, label="Unity URL")
    feed_url = _validate_loopback_http_base_url(config.feed_url, label="Feed URL")
    for label, path in (
        ("source TN policy", _SOURCE_AGENTS_PATH),
        ("output TN policy", _OUTPUT_AGENTS_PATH),
        ("governed tn-llama executable", _GOVERNED_SERVER_PATH),
        ("tn-llama GGUF model", _LLAMA_MODEL_PATH),
        ("tn-llama launcher", _LLAMA_LAUNCHER_PATH),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required {label} is missing: {path}")
    if not Path(_LLAMA_BUILD_DIRECTORY, "bin").is_dir():
        raise FileNotFoundError(
            f"governed tn-llama bin directory is missing: {_LLAMA_BUILD_DIRECTORY / 'bin'}"
        )
    return run_directory, unity_url, feed_url


def _provision_first_pass_tn(
    run_directory: Path,
    portfolio: GovernedPortfolioResult,
    execution,
) -> TNFlowResult:
    state = run_directory / "tn"
    generated_paths = {
        "source_project_dir": state / "source",
        "source_identity_dir": state / "source-identity",
        "input_recipient_config_path": state / "input-recipient" / "tn.yaml",
        "input_recipient_identity_path": (
            state / "input-recipient-identity" / "identity.json"
        ),
        "input_package_path": state / "packages" / "input-runtime.tnpkg",
        "input_reader_dir": state / "input-reader",
        "input_index_key_path": state / "input.index.key",
        "output_writer_config_path": state / "output-writer" / "tn.yaml",
        "output_writer_identity_path": state / "output-writer-identity" / "identity.json",
        "output_recipient_config_path": state / "output-recipient" / "tn.yaml",
        "output_recipient_identity_path": (
            state / "output-recipient-identity" / "identity.json"
        ),
        "output_package_path": state / "packages" / "output-runtime.tnpkg",
        "output_reader_dir": state / "output-reader",
        "output_index_key_path": state / "output.index.key",
    }
    for name, path in generated_paths.items():
        _require_run_owned_path(run_directory, path, label=f"governed {name}")
    return provision_tn_flow(
        portfolio,
        execution,
        source_agents_path=_SOURCE_AGENTS_PATH,
        output_agents_path=_OUTPUT_AGENTS_PATH,
        **generated_paths,
    )


def _read_transcript_facts(path: Path) -> tuple[int, tuple[str, str]]:
    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    responses = [
        (index, entry)
        for index, entry in enumerate(entries)
        if entry.get("peer") == "llama"
        and entry.get("direction") == "response"
        and entry.get("method") == "POST"
        and urllib.parse.urlparse(str(entry.get("url", ""))).path == _GOVERNED_ROUTE
    ]
    if len(responses) != 1 or not isinstance(responses[0][1].get("status"), int):
        raise ValueError("HTTP transcript lacks one governed tn-llama response")
    response_index, response = responses[0]
    opened = [
        (index, entry.get("group"))
        for index, entry in enumerate(entries)
        if entry.get("event") == "release_group_opened"
    ]
    if [group for _index, group in opened] != ["tn.agents", "default"]:
        raise ValueError("client transcript lacks the ordered official recipient opens")
    verified = [
        index
        for index, entry in enumerate(entries)
        if entry.get("event") == "encrypted_release_verified"
    ]
    if (
        len(verified) != 1
        or response_index >= opened[0][0]
        or opened[-1][0] >= verified[0]
    ):
        raise ValueError("client transcript has inconsistent recipient verification order")
    return int(response["status"]), ("tn.agents", "default")


def _safe_report_path(
    release: ReportRelease,
    report_directory: Path,
    *,
    run_directory: Path,
) -> Path:
    event_id = release.output_event_id
    if not event_id or Path(event_id).name != event_id or event_id in {".", ".."}:
        raise ValueError("verified output event id is not a safe report filename")
    report_root = _require_run_owned_path(
        run_directory,
        report_directory,
        label="report directory",
    )
    expected_path = _require_run_owned_path(
        run_directory,
        report_directory / f"{event_id}.html",
        label="report file",
    )
    report_path = save_report_html(release, report_directory).resolve()
    if report_path != expected_path or not report_path.is_relative_to(report_root):
        raise ValueError("saved report escaped the run report directory")
    return report_path


def _summary_document(
    *,
    unity_url: str,
    feed_url: str,
    portfolio: GovernedPortfolioResult,
    stub: StubDeepVest,
    execution,
    flow: TNFlowResult,
    release: ReportRelease,
    report_path: Path,
    response_status: int,
    recipient_open_groups: tuple[str, str],
) -> dict[str, object]:
    source_event_type = flow.source_row.envelope.get("event_type")
    output_row = json.loads(release.output_ndjson.removesuffix("\n"))
    output_event_type = output_row.get("event_type")
    if source_event_type != "portfolio.deepvest.calculated":
        raise ValueError("governed source event type changed before summary")
    if output_event_type != "report.generated":
        raise ValueError("verified output event type is not report.generated")
    return {
        "profile": "first_pass",
        "account_id": _ACCOUNT_ID,
        "unity": {
            "role": "governance_source",
            "base_url": unity_url,
            "resolved_feeds": list(portfolio.feed_names),
        },
        "getdata": {
            "role": "business_data_source",
            "boundary": "local_http_seeded",
            "base_url": feed_url,
            "fetch_facts": [
                {
                    "event_name": fact.event_name,
                    "feed_name": fact.feed_name,
                    "request_path": fact.request_path,
                    "http_status": fact.http_status,
                    "content_type": fact.content_type,
                }
                for fact in portfolio.feed_fetch_facts
            ],
        },
        "polars": {
            "admitted_schema": {
                name: str(dtype) for name, dtype in portfolio.values.schema.items()
            },
            "column_sources": {
                name: sorted(roots) for name, roots in portfolio.sources.items()
            },
            "row_selection_sources": sorted(portfolio.row_sources),
        },
        "deepvest": {
            "provider": execution.result.provider,
            "live": execution.result.live,
            "invocation_count": len(stub.invocations),
            "calculation_id": execution.result.calculation_id,
            "analysis_type": execution.result.analysis_type,
            "metrics": [
                {"name": metric.name, "value": metric.value, "unit": metric.unit}
                for metric in execution.result.metrics
            ],
        },
        "narrative": {
            "mode": "fixed_template",
            "model_authored": False,
            "data_specific_generation_claimed": False,
        },
        "tn_source": {
            "event_type": source_event_type,
            "event_id": flow.source_row.event_id,
            "row_hash": flow.source_row.row_hash,
            "encrypted_groups": list(flow.source_row.encrypted_groups),
        },
        "tn_llama": {
            "route": _GOVERNED_ROUTE,
            "response_status": response_status,
            "recipient_open_groups": list(recipient_open_groups),
        },
        "verified_output": {
            "event_type": output_event_type,
            "event_id": release.output_event_id,
            "row_hash": release.output_row_hash,
            "report_path": str(report_path),
        },
    }


def _write_summary(path: Path, document: dict[str, object]) -> Path:
    with path.open("x", encoding="utf-8", newline="") as summary:
        json.dump(document, summary, indent=2)
        summary.write("\n")
    return path


@contextmanager
def _scoped_run_environment() -> Iterator[None]:
    names = (
        "PYTHONDONTWRITEBYTECODE",
        "TN_NO_LINK",
        "TN_NO_STDOUT",
        "TN_AUTOINIT_QUIET",
    )
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_reference_run_config(config: ReferenceRunConfig) -> tuple[Path, str]:
    run_directory = _validate_new_run_directory(config.run_directory)
    if not config.api_key.strip():
        raise ValueError("TN_LLAMA_API_KEY must be nonempty")
    if not 1 <= config.llama_port <= 65535:
        raise ValueError("llama port must be between 1 and 65535")
    unity_url = _validate_loopback_http_base_url(config.unity_url, label="Unity URL")
    for label, path in (
        ("source TN policy", _SOURCE_AGENTS_PATH),
        ("output TN policy", _OUTPUT_AGENTS_PATH),
        ("governed tn-llama executable", _GOVERNED_SERVER_PATH),
        ("tn-llama GGUF model", _LLAMA_MODEL_PATH),
        ("tn-llama launcher", _LLAMA_LAUNCHER_PATH),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required {label} is missing: {path}")
    if not Path(_LLAMA_BUILD_DIRECTORY, "bin").is_dir():
        raise FileNotFoundError(
            f"governed tn-llama bin directory is missing: {_LLAMA_BUILD_DIRECTORY / 'bin'}"
        )
    return run_directory, unity_url


def _reference_unity_config(run_directory: Path, unity_url: str) -> UnityCatalogConfig:
    namespace = hashlib.sha256(str(run_directory).encode("utf-8")).hexdigest()[:12]
    return UnityCatalogConfig(
        base_url=unity_url,
        catalog=f"deepvest_{namespace}",
        schema="finance",
    )


@contextmanager
def _reference_feed_server(run_directory: Path, source_config_path: Path) -> Iterator[str]:
    governance = json.loads((_FIXTURE_ROOT / "tn_governance.json").read_text(encoding="utf-8"))
    event_path = run_directory / "evidence" / "getdata.ndjson"
    event_path.parent.mkdir(parents=True)
    data_dir = seed_finance_feed_data(run_directory / "feed-data")
    with event_path.open("x", encoding="utf-8", newline="") as events:
        server = create_feed_server(
            FeedServerConfig(
                data_dir=data_dir,
                event_stream=events,
                port=0,
                publisher_config_path=source_config_path,
                governance=governance,
            )
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address[:2]
            yield f"http://{host}:{port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def _reference_summary(
    *,
    unity_url: str,
    feed_url: str,
    processing: ProcessingPathResult,
    stub: StubDeepVest,
    flow: TNFlowResult,
    release: ReportRelease,
    report_path: Path,
    response_status: int,
    recipient_open_groups: tuple[str, str],
) -> dict[str, object]:
    return {
        "profile": "reference_path",
        "account_id": _ACCOUNT_ID,
        "unity": {
            "role": "governance_source",
            "base_url": unity_url,
            "resolved_feeds": list(processing.polars.feed_names),
        },
        "getdata": {
            "role": "business_data_source",
            "base_url": feed_url,
            "representation": "application/vnd.tn+json",
            "fetch_facts": [
                {
                    "feed_name": fact.feed_name,
                    "request_path": fact.request_path,
                    "http_status": fact.http_status,
                    "content_type": fact.content_type,
                }
                for fact in processing.feed_fetch_facts
            ],
        },
        "polars": {
            "owner": "tn_polars.GovernedPolars",
            "rows": processing.polars.values.height,
            "column_sources": {
                name: sorted(roots) for name, roots in processing.polars.sources.items()
            },
            "row_selection_sources": sorted(processing.polars.row_sources),
        },
        "deepvest": {
            "provider": processing.deepvest.result.provider,
            "live": processing.deepvest.result.live,
            "invocation_count": len(stub.invocations),
            "calculation_id": processing.deepvest.result.calculation_id,
        },
        "tn_handoffs": {
            "feed_groups": ["finance", "tn.agents"],
            "polars_groups": ["finance", "tn.agents"],
            "deepvest_groups": list(flow.source_row.encrypted_groups),
            "model_source_event_id": flow.source_row.event_id,
            "model_source_row_hash": flow.source_row.row_hash,
            "policy_reference": flow.source_row.policy_reference,
        },
        "tn_llama": {
            "route": _GOVERNED_ROUTE,
            "response_status": response_status,
            "recipient_open_groups": list(recipient_open_groups),
        },
        "verified_output": {
            "event_id": release.output_event_id,
            "row_hash": release.output_row_hash,
            "report_path": str(report_path),
        },
    }


def run_reference_path(
    config: ReferenceRunConfig,
    *,
    server_lifecycle_factory: Callable[
        [GovernedServerSpec], AbstractContextManager[None]
    ] = _governed_server,
    witness_writer: Callable[[Path], Path] = write_witness,
) -> FirstPassResult:
    """Run the TN-object reference path through the governed tn-llama route."""
    run_directory, unity_url = _validate_reference_run_config(config)
    with _scoped_run_environment():
        run_directory.mkdir(parents=True, exist_ok=False)
        material = provision_reference_path(
            run_directory / "tn",
            source_agents_path=_SOURCE_AGENTS_PATH,
            output_agents_path=_OUTPUT_AGENTS_PATH,
        )
        unity = _reference_unity_config(run_directory, unity_url)
        publish_finance_feeds(run_directory / "unity-bundles", config=unity)
        stub = StubDeepVest()
        with _reference_feed_server(run_directory, material.source_config_path) as feed_url:
            processing = run_processing_path(
                feed_url=feed_url,
                account_id=_ACCOUNT_ID,
                polars_config_path=material.polars_config_path,
                deepvest_config_path=material.deepvest_config_path,
                deepvest=stub,
                unity=unity,
            )
        if stub.invocations != (processing.deepvest.request,):
            raise RuntimeError("reference StubDeepVest did not execute exactly once")
        flow = build_reference_flow(material, processing.model_row)

        runtime_directory = run_directory / "runtime"
        runtime_config_path = write_finance_runtime_config(
            flow,
            config_path=runtime_directory / "runtime-config.json",
            audit_path=runtime_directory / "decision-events.ndjson",
            raw_trace_dir=runtime_directory / "raw-trace",
            model_path=_LLAMA_MODEL_PATH.resolve(),
            server_path=_GOVERNED_SERVER_PATH.resolve(),
            build_id="deepvest-tn-reference-path",
        )
        _preflight_loopback_port(config.llama_port)
        staged = _stage_governed_build(
            source_build_directory=_LLAMA_BUILD_DIRECTORY,
            source_server_path=_GOVERNED_SERVER_PATH,
            staged_build_directory=run_directory / "execution" / "governed-build",
            runtime_config_path=runtime_config_path,
        )
        spec = GovernedServerSpec(
            run_directory=run_directory,
            launcher_path=_LLAMA_LAUNCHER_PATH.resolve(),
            model_path=_LLAMA_MODEL_PATH.resolve(),
            runtime_config_path=runtime_config_path,
            build_directory=staged.build_directory,
            working_directory=run_directory / "process",
            stdout_path=run_directory / "server.stdout.log",
            stderr_path=run_directory / "server.stderr.log",
            port=config.llama_port,
            api_key=config.api_key,
        )
        transcript_path = run_directory / "client" / "http.ndjson"
        received_output_path = run_directory / "client" / "received-output.ndjson"
        with server_lifecycle_factory(spec):
            release = submit_finance_report(
                flow,
                prompt=_REPORT_PROMPT,
                base_url=f"http://127.0.0.1:{config.llama_port}",
                api_key=config.api_key,
                transcript_path=transcript_path,
                received_output_path=received_output_path,
            )
            response_status, opened_groups = _read_transcript_facts(transcript_path)
            if response_status != 200:
                raise ValueError("governed tn-llama response was not successful")
            report_path = _safe_report_path(
                release,
                run_directory / "reports",
                run_directory=run_directory,
            )
            summary_path = _write_summary(
                run_directory / "run-summary.json",
                _reference_summary(
                    unity_url=unity_url,
                    feed_url=feed_url,
                    processing=processing,
                    stub=stub,
                    flow=flow,
                    release=release,
                    report_path=report_path,
                    response_status=response_status,
                    recipient_open_groups=opened_groups,
                ),
            )
        witness_path = witness_writer(run_directory)
        return FirstPassResult(
            release=release,
            report_path=report_path,
            summary_path=summary_path,
            facts=FirstPassFacts(
                profile="reference_path",
                account_id=_ACCOUNT_ID,
                feed_names=processing.polars.feed_names,
                stub_invocation_count=len(stub.invocations),
                source_event_id=flow.source_row.event_id,
                output_event_id=release.output_event_id,
                response_status=response_status,
                recipient_open_groups=opened_groups,
            ),
            witness_path=witness_path,
        )


def run_first_pass(
    config: FirstPassConfig,
    *,
    server_lifecycle_factory: Callable[
        [GovernedServerSpec], AbstractContextManager[None]
    ] = _governed_server,
) -> FirstPassResult:
    """Run the fixed Unity-to-recipient portfolio demonstration profile exactly once."""
    run_directory, unity_url, feed_url = _validate_first_pass_config(config)
    with _scoped_run_environment():
        return _run_first_pass_scoped(
            config,
            run_directory=run_directory,
            unity_url=unity_url,
            feed_url=feed_url,
            server_lifecycle_factory=server_lifecycle_factory,
        )


def _run_first_pass_scoped(
    config: FirstPassConfig,
    *,
    run_directory: Path,
    unity_url: str,
    feed_url: str,
    server_lifecycle_factory: Callable[
        [GovernedServerSpec], AbstractContextManager[None]
    ],
) -> FirstPassResult:
    run_directory.mkdir(parents=True, exist_ok=False)

    portfolio = run_governed_polars_stage(
        UnityCatalogConfig(base_url=unity_url),
        account_id=_ACCOUNT_ID,
        feed_url=feed_url,
    )
    stub = StubDeepVest()
    execution = execute_deepvest(stub, portfolio)
    if stub.invocations != (execution.request,):
        raise RuntimeError("first-pass StubDeepVest did not execute exactly once")
    if execution.result.provider != "stub" or execution.result.live is not False:
        raise RuntimeError("first-pass DeepVest result is not explicitly stubbed")

    flow = _provision_first_pass_tn(run_directory, portfolio, execution)
    runtime_directory = run_directory / "runtime"
    runtime_config_target = runtime_directory / "runtime-config.json"
    decision_events_path = runtime_directory / "decision-events.ndjson"
    raw_trace_directory = runtime_directory / "raw-trace"
    for label, path in (
        ("runtime config", runtime_config_target),
        ("decision events", decision_events_path),
        ("raw trace directory", raw_trace_directory),
    ):
        _require_run_owned_path(run_directory, path, label=label)
    runtime_config_path = write_finance_runtime_config(
        flow,
        config_path=runtime_config_target,
        audit_path=decision_events_path,
        raw_trace_dir=raw_trace_directory,
        model_path=_LLAMA_MODEL_PATH.resolve(),
        server_path=_GOVERNED_SERVER_PATH.resolve(),
        build_id="deepvest-tn-first-pass",
    )
    _preflight_loopback_port(config.llama_port)
    staged_build_target = run_directory / "execution" / "governed-build"
    _require_run_owned_path(
        run_directory,
        staged_build_target,
        label="staged governed build",
    )
    staged = _stage_governed_build(
        source_build_directory=_LLAMA_BUILD_DIRECTORY,
        source_server_path=_GOVERNED_SERVER_PATH,
        staged_build_directory=staged_build_target,
        runtime_config_path=runtime_config_path,
    )
    server_spec = GovernedServerSpec(
        run_directory=run_directory,
        launcher_path=_LLAMA_LAUNCHER_PATH.resolve(),
        model_path=_LLAMA_MODEL_PATH.resolve(),
        runtime_config_path=runtime_config_path,
        build_directory=staged.build_directory,
        working_directory=run_directory / "process",
        stdout_path=run_directory / "server.stdout.log",
        stderr_path=run_directory / "server.stderr.log",
        port=config.llama_port,
        api_key=config.api_key,
    )
    transcript_path = run_directory / "client" / "http.ndjson"
    received_output_path = run_directory / "client" / "received-output.ndjson"
    with server_lifecycle_factory(server_spec):
        for label, path in (
            ("HTTP transcript", transcript_path),
            ("received output", received_output_path),
            ("report directory", run_directory / "reports"),
            ("run summary", run_directory / "run-summary.json"),
        ):
            _require_run_owned_path(run_directory, path, label=label)
        release = submit_finance_report(
            flow,
            prompt=_REPORT_PROMPT,
            base_url=f"http://127.0.0.1:{config.llama_port}",
            api_key=config.api_key,
            transcript_path=transcript_path,
            received_output_path=received_output_path,
        )
        if (
            release.source_event_id != flow.source_row.event_id
            or release.source_row_hash != flow.source_row.row_hash
        ):
            raise ValueError("verified report release does not identify the submitted TN source")
        response_status, opened_groups = _read_transcript_facts(transcript_path)
        if response_status != 200:
            raise ValueError("governed tn-llama response was not successful")
        report_path = _safe_report_path(
            release,
            run_directory / "reports",
            run_directory=run_directory,
        )
        summary_document = _summary_document(
            unity_url=unity_url,
            feed_url=feed_url,
            portfolio=portfolio,
            stub=stub,
            execution=execution,
            flow=flow,
            release=release,
            report_path=report_path,
            response_status=response_status,
            recipient_open_groups=opened_groups,
        )
        summary_path = _write_summary(run_directory / "run-summary.json", summary_document)

    return FirstPassResult(
        release=release,
        report_path=report_path,
        summary_path=summary_path,
        facts=FirstPassFacts(
            profile="first_pass",
            account_id=_ACCOUNT_ID,
            feed_names=portfolio.feed_names,
            stub_invocation_count=len(stub.invocations),
            source_event_id=flow.source_row.event_id,
            output_event_id=release.output_event_id,
            response_status=response_status,
            recipient_open_groups=opened_groups,
        ),
    )
