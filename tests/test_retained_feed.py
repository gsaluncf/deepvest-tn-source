import io
from contextlib import contextmanager
from threading import Thread

import pytest
from governed_edition_fixtures import signed_edition

from deepvest_tn.feed_server import FeedServerConfig, create_feed_server
from deepvest_tn.getdata import (
    FeedFetchHTTPError,
    FeedFetchResponseError,
    GetDataConfig,
    fetch_governed_feed,
)
from deepvest_tn.publications import PublicationStore


@contextmanager
def service(tmp_path, store):
    server = create_feed_server(
        FeedServerConfig(tmp_path, io.StringIO(), 0, publication_store=store)
    )
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield GetDataConfig(f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_pinned_selection_repeats_exact_bytes_after_server_and_store_restart(tmp_path):
    fixture = signed_edition(tmp_path / "edition")
    root = tmp_path / "publications"
    store = PublicationStore(root)
    store.put(fixture.source)
    with service(tmp_path, store) as config:
        first = fetch_governed_feed(config, fixture.selection)
        second = fetch_governed_feed(config, fixture.selection)
    with service(tmp_path, PublicationStore(root)) as config:
        restarted = fetch_governed_feed(config, fixture.selection)
    assert first.body == second.body == restarted.body == fixture.source.forward()
    assert first.fact.request_path.endswith(fixture.source.id[7:])


def test_pinned_selection_missing_publication_fails_without_resealing(tmp_path):
    fixture = signed_edition(tmp_path / "edition")
    with (
        service(tmp_path, PublicationStore(tmp_path / "empty")) as config,
        pytest.raises(FeedFetchHTTPError) as failure,
    ):
        fetch_governed_feed(config, fixture.selection)
    assert failure.value.status == 404


def test_route_returning_another_valid_source_is_rejected_before_admission(tmp_path):
    first = signed_edition(tmp_path / "first")
    other = signed_edition(tmp_path / "other", session=first.session, edition="new")

    class WrongRoute:
        def get(self, identity):
            assert identity == first.source.id
            return other.source.forward()

    with (
        service(tmp_path, WrongRoute()) as config,
        pytest.raises(FeedFetchResponseError, match="identity"),
    ):
        fetch_governed_feed(config, first.selection)


def test_corrupt_publication_returns_no_business_body(tmp_path):
    fixture = signed_edition(tmp_path / "edition")
    store = PublicationStore(tmp_path / "publications")
    store.put(fixture.source)
    (store.root / "objects" / f"{fixture.source.id[7:]}.tn").write_bytes(b"substituted")
    with (
        service(tmp_path, store) as config,
        pytest.raises(FeedFetchHTTPError) as failure,
    ):
        fetch_governed_feed(config, fixture.selection)
    assert failure.value.status == 500
