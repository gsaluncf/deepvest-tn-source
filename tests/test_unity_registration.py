"""Edition registration keeps server checks when catalog setup is already complete."""
import json

import pytest
from tn_polars.registry import UnityCatalogRegistry
from tn_polars.release import PolicyDenied

from governed_edition_fixtures import signed_edition


class VolumeTransport:
    def __init__(self):
        self.calls = []
        self.locations = {}
        self.failure = None
        self.return_location = None

    def request(self, method, path, payload=None):
        self.calls.append((method, path))
        if self.failure and path == "/volumes":
            raise PolicyDenied(self.failure)
        if method == "POST" and path == "/volumes":
            name = f"{payload['catalog_name']}.{payload['schema_name']}.{payload['name']}"
            self.locations[name] = payload["storage_location"]
        if method == "GET" and path.startswith("/volumes/"):
            return {"storage_location": self.return_location or self.locations[path.split("/")[-1]]}
        return {}


@pytest.fixture
def registration(tmp_path, monkeypatch):
    source = signed_edition(tmp_path / "source")
    registry = UnityCatalogRegistry("http://catalog", "finance", "sources")
    transport = VolumeTransport()
    monkeypatch.setattr(registry, "_request", transport.request)
    args = (source.record.dataset, source.record.edition, source.record.id, source.bundle)
    yield source, registry, transport, args


@pytest.mark.parametrize("create_namespace", [True, False])
def test_each_edition_keeps_volume_creation_and_location_checks(registration, create_namespace):
    source, registry, transport, args = registration
    if create_namespace:
        registry.register_edition(*args)
    else:
        registry.register_edition(*args, create_namespace=False)
    setup = [("POST", "/catalogs"), ("POST", "/schemas")] if create_namespace else []
    full = f"/volumes/finance.sources.tn_edition_{source.record.id[7:]}"
    assert transport.calls == [*setup, ("POST", "/volumes"), ("GET", full), ("GET", full)]
    transport.calls.clear()
    transport.failure = "HTTP 403: access was revoked"
    with pytest.raises(PolicyDenied, match="403"):
        registry.register_edition(*args, create_namespace=False)
    assert transport.calls == [("POST", "/volumes")]


@pytest.mark.parametrize("failure", ["HTTP 401: expired credentials", "HTTP 404: schema removed"])
def test_volume_failure_stops_registration_without_recreating_namespace(registration, failure):
    _, registry, transport, args = registration
    transport.failure = failure
    with pytest.raises(PolicyDenied, match="401|404"):
        registry.register_edition(*args, create_namespace=False)
    assert transport.calls == [("POST", "/volumes")]


def test_conflicting_volume_location_is_rejected(registration, tmp_path):
    _, registry, transport, args = registration
    transport.return_location = (tmp_path / "other").as_uri()
    with pytest.raises(PolicyDenied, match="another bundle"):
        registry.register_edition(*args, create_namespace=False)
    assert len(transport.calls) == 2


def test_wrong_bundle_identity_stops_before_catalog_request(registration):
    source, registry, transport, args = registration
    path = source.bundle / "bundle.json"
    manifest = json.loads(path.read_bytes())
    manifest["record_id"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(PolicyDenied, match="identity"):
        registry.register_edition(*args, create_namespace=False)
    assert transport.calls == []
