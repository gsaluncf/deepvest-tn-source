"""Client contract for the example's ordinary internal-data feed service."""

from __future__ import annotations

import io
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GetDataConfig:
    """Immutable connection settings for the ordinary feed service."""

    base_url: str
    timeout_seconds: float = 5.0


@dataclass(frozen=True)
class FeedFetchFact:
    """Fetch metadata recorded without including data rows."""

    event_name: str
    feed_name: str
    request_path: str
    http_status: int
    content_type: str


@dataclass(frozen=True)
class FetchedFeed:
    """Buffered NDJSON and the metadata recorded for its successful fetch."""

    feed_name: str
    body: bytes
    fact: FeedFetchFact


class FeedFetchHTTPError(RuntimeError):
    """The service responded with a non-success HTTP status."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"feed fetch failed with HTTP {status}")


class FeedFetchResponseError(RuntimeError):
    """A successful HTTP response violated the feed representation contract."""


def fetch_feed(config: GetDataConfig, feed_name: str) -> FetchedFeed:
    """Fetch one feed as buffered UTF-8 NDJSON from the fixed service route."""
    return _fetch_feed(
        config,
        feed_name,
        accept="application/x-ndjson",
        expected_media_type="application/x-ndjson",
    )


def fetch_governed_feed(config: GetDataConfig, feed_name) -> FetchedFeed:
    """Fetch a native selection's retained source, or a named feed."""
    import tn
    if isinstance(feed_name, tn.DatasetSelection):
        selection = feed_name
        from .publications import object_digest
        fetched = _fetch_feed(
            config, selection.dataset, accept="application/vnd.tn+json",
            expected_media_type="application/vnd.tn+json",
            request_path=f"/v1/publications/{object_digest(selection.source_object_id)}",
        )
        try:
            object = tn.GovernedObject.read(io.BytesIO(fetched.body))
        except Exception as error:
            raise FeedFetchResponseError(
                "publication is not an authenticated native TN object"
            ) from error
        if object.id != selection.source_object_id:
            raise FeedFetchResponseError(
                "publication identity differs from the accepted dataset selection"
            )
        return fetched
    if not isinstance(feed_name, str):
        raise TypeError("governed feed requires a native DatasetSelection or legacy feed name")
    return _fetch_feed(
        config,
        feed_name,
        accept="application/vnd.tn+json",
        expected_media_type="application/vnd.tn+json",
    )


def _fetch_feed(
    config: GetDataConfig,
    feed_name: str,
    *,
    accept: str,
    expected_media_type: str,
    request_path: str | None = None,
) -> FetchedFeed:
    request_path = request_path or f"/v1/feeds/{quote(feed_name, safe='.')}/data"
    request = Request(
        f"{config.base_url.rstrip('/')}{request_path}",
        headers={"Accept": accept},
        method="GET",
    )
    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "")
            media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
            if media_type != expected_media_type:
                raise FeedFetchResponseError(
                    f"feed fetch returned unsupported media type {content_type!r}"
                )
            body = response.read()
            if not body:
                raise FeedFetchResponseError("feed fetch returned an empty body")
            try:
                body.decode("utf-8")
            except UnicodeDecodeError as error:
                raise FeedFetchResponseError("feed fetch body is not valid UTF-8") from error
            fact = FeedFetchFact(
                event_name="getdata.feed_fetched",
                feed_name=feed_name,
                request_path=request_path,
                http_status=response.status,
                content_type=content_type,
            )
            return FetchedFeed(feed_name=feed_name, body=body, fact=fact)
    except HTTPError as error:
        raise FeedFetchHTTPError(error.code) from error
    except URLError as error:
        raise FeedFetchHTTPError(0) from error
