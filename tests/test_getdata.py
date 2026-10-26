from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from tn_governed.feed_server import FeedServerConfig, create_feed_server, seed_finance_feed_data
from tn_governed.getdata import (
    FeedFetchHTTPError,
    FeedFetchResponseError,
    GetDataConfig,
    fetch_feed,
)

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "portfolio"

POSITIONS_ROWS = (
    b'{"account_id":"acct-001","portfolio_id":"portfolio-001","symbol":"AAPL","quantity":125,"cost_basis":182.40,"currency":"USD"}\n'
    b'{"account_id":"acct-001","portfolio_id":"portfolio-001","symbol":"MSFT","quantity":80,"cost_basis":412.15,"currency":"USD"}\n'
    b'{"account_id":"acct-001","portfolio_id":"portfolio-001","symbol":"VTI","quantity":210,"cost_basis":265.75,"currency":"USD"}\n'
)
PRICES_ROWS = (
    b'{"symbol":"AAPL","as_of":"2026-09-01T16:00:00Z","last_price":228.15,"currency":"USD","market_source":"deterministic-market-fixture"}\n'
    b'{"symbol":"MSFT","as_of":"2026-09-01T16:00:00Z","last_price":509.90,"currency":"USD","market_source":"deterministic-market-fixture"}\n'
    b'{"symbol":"VTI","as_of":"2026-09-01T16:00:00Z","last_price":311.42,"currency":"USD","market_source":"deterministic-market-fixture"}\n'
)


@contextmanager
def running_server(server: ThreadingHTTPServer) -> Iterator[str]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture()
def feed_service(tmp_path: Path) -> Iterator[tuple[str, io.StringIO, Path]]:
    runtime_data = seed_finance_feed_data(tmp_path / "runtime-feed-data")
    events = io.StringIO()
    server = create_feed_server(
        FeedServerConfig(data_dir=runtime_data, event_stream=events, port=0)
    )
    with running_server(server) as base_url:
        yield base_url, events, runtime_data


def test_seeding_copies_only_the_two_fixture_files(tmp_path: Path) -> None:
    runtime_data = seed_finance_feed_data(tmp_path / "runtime-feed-data")

    assert sorted(path.name for path in runtime_data.iterdir()) == [
        "internal_positions.ndjson",
        "market_prices.ndjson",
    ]
    for filename, expected_rows in (
        ("internal_positions.ndjson", POSITIONS_ROWS),
        ("market_prices.ndjson", PRICES_ROWS),
    ):
        source_bytes = (_FIXTURE_ROOT / filename).read_bytes()
        assert source_bytes.splitlines() == expected_rows.splitlines()
        assert (runtime_data / filename).read_bytes() == source_bytes


@pytest.mark.parametrize(
    ("feed_name", "expected_rows"),
    [
        ("portfolio.internal.positions", POSITIONS_ROWS),
        ("portfolio.market.prices", PRICES_ROWS),
    ],
)
def test_exact_feed_route_returns_raw_rows_and_client_fact(
    feed_service: tuple[str, io.StringIO, Path],
    feed_name: str,
    expected_rows: bytes,
) -> None:
    base_url, events, runtime_data = feed_service
    filename = {
        "portfolio.internal.positions": "internal_positions.ndjson",
        "portfolio.market.prices": "market_prices.ndjson",
    }[feed_name]
    stored_bytes = (runtime_data / filename).read_bytes()
    assert stored_bytes.splitlines() == expected_rows.splitlines()
    request_path = f"/v1/feeds/{feed_name}/data"

    with urlopen(f"{base_url}{request_path}") as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "application/x-ndjson; charset=utf-8"
        assert response.read() == stored_bytes

    fetched = fetch_feed(GetDataConfig(base_url=base_url), feed_name)

    assert fetched.feed_name == feed_name
    assert fetched.body == stored_bytes
    assert fetched.fact.event_name == "getdata.feed_fetched"
    assert fetched.fact.feed_name == feed_name
    assert fetched.fact.request_path == request_path
    assert fetched.fact.http_status == 200
    assert fetched.fact.content_type == "application/x-ndjson; charset=utf-8"
    event = json.loads(events.getvalue().splitlines()[-1])
    assert event == {
        "event": "getdata.feed_served",
        "feed_name": feed_name,
        "request_path": request_path,
        "http_status": 200,
        "content_type": "application/x-ndjson; charset=utf-8",
    }


@pytest.mark.parametrize(
    "path",
    [
        "/v1/feeds/not-a-feed/data",
        "/v1/feeds/%2e%2e%2fmarket_prices/data",
    ],
)
def test_unknown_or_encoded_traversal_route_returns_not_found(
    feed_service: tuple[str, io.StringIO, Path], path: str
) -> None:
    base_url, _events, _runtime_data = feed_service

    with pytest.raises(HTTPError) as caught:
        urlopen(f"{base_url}{path}")

    assert caught.value.code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/v1/feeds/not-a-feed/data",
        "/v1/feeds/%2e%2e%2fmarket_prices/data",
    ],
)
def test_post_to_unknown_or_encoded_traversal_route_returns_not_found(
    feed_service: tuple[str, io.StringIO, Path], path: str
) -> None:
    base_url, _events, _runtime_data = feed_service
    request = Request(f"{base_url}{path}", data=b"", method="POST")

    with pytest.raises(HTTPError) as caught:
        urlopen(request)

    assert caught.value.code == 404


@pytest.mark.parametrize(
    "feed_name",
    ["portfolio.internal.positions", "portfolio.market.prices"],
)
def test_post_to_exact_data_route_is_not_allowed(
    feed_service: tuple[str, io.StringIO, Path], feed_name: str
) -> None:
    base_url, _events, _runtime_data = feed_service
    request = Request(
        f"{base_url}/v1/feeds/{feed_name}/data",
        data=b"",
        method="POST",
    )

    with pytest.raises(HTTPError) as caught:
        urlopen(request)

    assert caught.value.code == 405
    assert caught.value.headers["Allow"] == "GET"


def available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_handle:
        socket_handle.bind(("127.0.0.1", 0))
        return int(socket_handle.getsockname()[1])


def test_server_binds_the_requested_nonzero_loopback_port(tmp_path: Path) -> None:
    requested_port = available_loopback_port()
    runtime_data = seed_finance_feed_data(tmp_path / "runtime-feed-data")
    server = create_feed_server(
        FeedServerConfig(data_dir=runtime_data, event_stream=io.StringIO(), port=requested_port)
    )

    with running_server(server) as base_url:
        assert base_url == f"http://127.0.0.1:{requested_port}"
        with urlopen(f"{base_url}/health") as response:
            assert response.status == 200


def test_module_entry_point_seeds_and_serves_the_requested_port(tmp_path: Path) -> None:
    requested_port = available_loopback_port()
    runtime_data = tmp_path / "module-runtime-feed-data"
    environment = os.environ.copy()
    source_directory = str(Path("src").resolve())
    environment["PYTHONPATH"] = source_directory + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tn_governed.feed_server",
            "--port",
            str(requested_port),
            "--data-directory",
            str(runtime_data),
        ],
        cwd=Path.cwd(),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(process.stderr.read())
            try:
                with urlopen(f"http://127.0.0.1:{requested_port}/health") as response:
                    assert response.status == 200
                    break
            except URLError:
                time.sleep(0.05)
        else:
            pytest.fail("module entry point did not start the requested server")

        for filename in ("internal_positions.ndjson", "market_prices.ndjson"):
            assert (runtime_data / filename).read_bytes() == (_FIXTURE_ROOT / filename).read_bytes()
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)


class FixedResponseHandler(BaseHTTPRequestHandler):
    status = 200
    content_type = "application/x-ndjson; charset=utf-8"
    body = POSITIONS_ROWS
    received_accept: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        type(self).received_accept = self.headers.get("Accept")
        self.send_response(self.status)
        self.send_header("Content-Type", self.content_type)
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@pytest.mark.parametrize(
    ("status", "content_type", "body", "exception_type", "message"),
    [
        (404, "text/plain; charset=utf-8", b"missing", FeedFetchHTTPError, "HTTP 404"),
        (200, "application/x-ndjson; charset=utf-8", b"", FeedFetchResponseError, "empty"),
        (200, "text/plain; charset=utf-8", POSITIONS_ROWS, FeedFetchResponseError, "media type"),
    ],
)
def test_client_rejects_invalid_http_response(
    status: int,
    content_type: str,
    body: bytes,
    exception_type: type[Exception],
    message: str,
) -> None:
    FixedResponseHandler.status = status
    FixedResponseHandler.content_type = content_type
    FixedResponseHandler.body = body
    FixedResponseHandler.received_accept = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixedResponseHandler)

    with running_server(server) as base_url, pytest.raises(exception_type, match=message):
        fetch_feed(GetDataConfig(base_url=base_url), "portfolio.internal.positions")

    assert FixedResponseHandler.received_accept == "application/x-ndjson"
