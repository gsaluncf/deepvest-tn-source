"""Fixed-route HTTP server for the example's seeded internal feed fixtures."""

from __future__ import annotations

import json
import re
import shutil
import sys
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TextIO

_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "portfolio"
_FEED_FILES = {
    "deepvest.internal.positions": "internal_positions.ndjson",
    "deepvest.market.prices": "market_prices.ndjson",
}
_NDJSON_CONTENT_TYPE = "application/x-ndjson; charset=utf-8"
_TN_CONTENT_TYPE = "application/vnd.tn+json; charset=utf-8"


@dataclass(frozen=True)
class FeedServerConfig:
    """Server settings containing only the seeded data directory and event sink."""

    data_dir: Path
    event_stream: TextIO
    port: int
    publisher_config_path: Path | None = None
    governance: Mapping[str, object] | None = None
    publication_store: object | None = None


def seed_finance_feed_data(runtime_data_dir: Path) -> Path:
    """Create a fresh runtime directory containing only the two fixed fixture copies."""
    if runtime_data_dir.exists():
        raise ValueError(f"runtime feed data directory already exists: {runtime_data_dir}")
    runtime_data_dir.mkdir(parents=True)
    for filename in _FEED_FILES.values():
        shutil.copyfile(_FIXTURE_ROOT / filename, runtime_data_dir / filename)
    return runtime_data_dir


def create_feed_server(config: FeedServerConfig) -> ThreadingHTTPServer:
    """Build a loopback-ready server with only the two exact feed routes."""
    if (config.publisher_config_path is None) != (config.governance is None):
        raise ValueError("TN publisher config and governance must be supplied together")

    class FeedRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._write_response(
                    HTTPStatus.OK,
                    "application/json; charset=utf-8",
                    b'{"ready":true}\n',
                )
                return

            publication = re.fullmatch(r"/v1/publications/([0-9a-f]{64})", self.path)
            if publication and config.publication_store is not None:
                identity = f"sha256:{publication[1]}"
                try:
                    body = config.publication_store.get(identity)
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                except ValueError:
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
                    return
                self._write_response(HTTPStatus.OK, _TN_CONTENT_TYPE, body)
                _emit_event(config.event_stream, identity, self.path, _TN_CONTENT_TYPE)
                return

            for feed_name, filename in _FEED_FILES.items():
                request_path = f"/v1/feeds/{feed_name}/data"
                if self.path == request_path:
                    if config.publisher_config_path is None:
                        body = (config.data_dir / filename).read_bytes()
                        content_type = _NDJSON_CONTENT_TYPE
                    else:
                        from .tn_objects import seal_governed_object

                        rows = [
                            json.loads(line)
                            for line in (config.data_dir / filename)
                            .read_text(encoding="utf-8")
                            .splitlines()
                            if line
                        ]
                        sealed = seal_governed_object(
                            config.publisher_config_path,
                            "portfolio.feed.data.v1",
                            business_group="finance",
                            business_fields={"feed_name": feed_name, "rows": rows},
                            governance=config.governance,
                        )
                        body = str(sealed).encode("utf-8")
                        content_type = _TN_CONTENT_TYPE
                    self._write_response(HTTPStatus.OK, content_type, body)
                    _emit_event(
                        config.event_stream,
                        feed_name,
                        request_path,
                        content_type,
                    )
                    return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if (re.fullmatch(r"/v1/publications/[0-9a-f]{64}", self.path) or self.path in {
                f"/v1/feeds/{feed_name}/data" for feed_name in _FEED_FILES
            }):
                self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
                self.send_header("Allow", "GET")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _write_response(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", config.port), FeedRequestHandler)


def _emit_event(
    event_stream: TextIO,
    feed_name: str,
    request_path: str,
    content_type: str = _NDJSON_CONTENT_TYPE,
) -> None:
    event_stream.write(
        json.dumps(
            {
                "event": "getdata.feed_served",
                "feed_name": feed_name,
                "request_path": request_path,
                "http_status": 200,
                "content_type": content_type,
            }
        )
        + "\n"
    )
    event_stream.flush()


def main(argv: Sequence[str] | None = None) -> None:
    """Seed a fresh directory and serve the two fixed feed routes until interrupted."""
    parser = ArgumentParser(description="Serve seeded DeepVest finance feed fixtures.")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--data-directory", required=True, type=Path)
    arguments = parser.parse_args(argv)

    data_dir = seed_finance_feed_data(arguments.data_directory)
    server = create_feed_server(
        FeedServerConfig(data_dir=data_dir, event_stream=sys.stdout, port=arguments.port)
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
